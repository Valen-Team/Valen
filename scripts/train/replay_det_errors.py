"""Replay a sealed Det corpus and select original errors without test-set mining."""
import argparse
from collections import Counter, defaultdict, deque
import copy
import heapq
import json
from pathlib import Path
import random
import shutil

from scripts.train.spatial_worker import atomic_json


def inference_record(record):
    metadata=record.get('meta',{})
    return {'id':record['id'],'group_id':record['group_id'],'request':copy.deepcopy(record['request']),
            'meta':{'referring':True,'allow_empty_target':bool(metadata.get('allow_empty_target',False))}}


def balanced_errors(records, cap=20000, seed=42):
    if len({r['id'] for r in records})!=len(records):
        raise ValueError('Error candidates must be unique original records')
    if len(records)<=cap:
        return list(records)
    rng=random.Random(seed)
    groups=defaultdict(lambda:defaultdict(list))
    for record in records:
        groups[record['meta']['source']][record['group_id']].append(record)
    queues,heaps={},{}
    counts=Counter()
    for source in sorted(groups):
        queues[source]={}
        heaps[source]=[]
        for image,rows in sorted(groups[source].items()):
            rng.shuffle(rows)
            queues[source][image]=deque(rows)
            heapq.heappush(heaps[source],(0,rng.random(),image))
    selected=[]
    while len(selected)<cap:
        for source in sorted(groups):
            heap=heaps[source]
            while heap:
                count,tie,image=heapq.heappop(heap)
                if count!=counts[image]:
                    heapq.heappush(heap,(counts[image],tie,image))
                    continue
                selected.append(queues[source][image].popleft())
                counts[image]+=1
                if queues[source][image]:
                    heapq.heappush(heap,(counts[image],tie,image))
                break
            if len(selected)==cap:
                break
    return selected


def select_errors(records, predictions, model_manifest_sha, source_sha, output, cap=20000):
    from valen.training.checkpoint import file_sha256
    catalog={r['id']:r for r in records}
    if len(catalog)!=len(records) or len(predictions)!=len(records):
        raise ValueError('Replay coverage must include every unique original record')
    observed={p['id'] for p in predictions}
    if len(observed)!=len(predictions) or observed!=set(catalog):
        raise ValueError('Replay IDs differ from the sealed training corpus')
    if any(type(p.get('correct')) is not bool for p in predictions):
        raise ValueError('Replay errors must come from successful scored inference')
    errors=[catalog[p['id']] for p in predictions if not p['correct']]
    selected=balanced_errors(errors,cap)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    train=output/'train.jsonl'
    train.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    manifest={'format':'valen_det_original_errors_v1','validated':True,
              'original_records':len(records),'original_training_sha256':source_sha,
              'model_export_manifest_sha256':model_manifest_sha,'errors':len(errors),
              'selected':len(selected),'records_sha256':file_sha256(train),'seed':42,'cap':cap,
              'images':len({r['group_id'] for r in selected}),
              'source_counts':dict(Counter(r['meta']['source'] for r in selected)),
              'original_records_only':True,'all_errors_when_at_most_cap':len(errors)<=cap,
              'zero_error_skip':not errors}
    atomic_json(output/'manifest.json',manifest)
    return manifest


def score_training_errors(records,predictions,subset,source_sha,model_sha):
    ids={r['id'] for r in records}
    observed={p['id']:p for p in predictions}
    if len(ids)!=len(records) or len(observed)!=len(predictions) or set(observed)!=ids:
        raise ValueError('Training-score replay does not cover the sealed corpus exactly once')
    selected={r['id'] for r in subset}
    if len(selected)!=len(subset) or not selected<=ids:
        raise ValueError('Training-score subset is not a unique subset of the original corpus')
    errors=sum(not p['correct'] for p in predictions)
    selected_errors=sum(not observed[i]['correct'] for i in selected)
    return {'format':'valen_det_training_error_score_v1','original_records':len(records),
            'original_training_sha256':source_sha,'model_export_manifest_sha256':model_sha,
            'errors':errors,'error_rate':errors/len(records),'selected':len(selected),
            'selected_errors':selected_errors,'selected_error_rate':selected_errors/len(selected) if selected else 0.}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['training-data','data-root','model-path','output-dir','subset-dir']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--expected-source-sha256',required=True)
    parser.add_argument('--score-only',action='store_true')
    parser.add_argument('--subset-to-score')
    parser.add_argument('--expected-subset-sha256')
    args=parser.parse_args()
    import torch
    from evaluation.spatial import load_spatial_model
    from valen.data.spatial import read_spatial_jsonl
    from valen.modeling.qwen.spatial import decode_spatial
    from valen.training.checkpoint import file_sha256
    from valen.training.distributed import initialize,close
    from valen.training.spatial_loss import box_iou
    context=initialize()
    try:
        source_sha=file_sha256(args.training_data)
        if source_sha!=args.expected_source_sha256:
            raise ValueError('Replay input differs from the sealed 100k corpus')
        records=read_spatial_jsonl(args.training_data,labeled_only=True)
        if len(records)!=100000:
            raise ValueError('Formal Det replay requires the complete original 100k')
        package=Path(args.model_path)
        identity=file_sha256(package/'export_manifest.json')
        if context.primary:
            export=json.loads((package/'export_manifest.json').read_text())
            expected_training=args.expected_subset_sha256 if args.score_only else source_sha
            if export.get('training_data_sha256')!=expected_training:
                raise ValueError('Replay model was trained on a different original corpus')
            for filename,digest in export['files_sha256'].items():
                if file_sha256(package/filename)!=digest:
                    raise ValueError('Replay package changed after acceptance')
        context.barrier()
        output=Path(args.output_dir)
        shards=output/'shards'
        shards.mkdir(parents=True,exist_ok=True)
        path=shards/f'rank-{context.rank}.jsonl'
        model,compiler=load_spatial_model(package,context.device,data_root=args.data_root)
        with torch.inference_mode(), path.open('w') as stream:
            for record in records[context.rank::context.world_size]:
                question=compiler.compile(inference_record(record)).questions[0]
                prediction=decode_spatial(question,model(question))
                target=next(iter(record['targets'].values()))
                boxes=prediction['boxes']
                truth=target.get('boxes',[])
                if not truth:
                    overlap=float(not boxes)
                    correct=not boxes
                else:
                    overlap=float(box_iou(torch.tensor([boxes[0]['bbox']]),torch.tensor(truth)).max()) if boxes else 0.
                    correct=bool(boxes and overlap>=.5)
                stream.write(json.dumps({'id':record['id'],'correct':correct,'iou':overlap,
                                         'boxes':boxes})+'\n')
        atomic_json(shards/f'rank-{context.rank}.json',{'records':len(records[context.rank::context.world_size]),
                    'sha256':file_sha256(path),'source_sha256':source_sha,'model_manifest_sha256':identity})
        context.barrier()
        if context.primary:
            predictions=[]
            for rank in range(context.world_size):
                part=shards/f'rank-{rank}.jsonl'
                metadata=json.loads((shards/f'rank-{rank}.json').read_text())
                if (metadata['sha256']!=file_sha256(part) or metadata['source_sha256']!=source_sha
                        or metadata['model_manifest_sha256']!=identity):
                    raise ValueError('A replay shard has a different data/model identity')
                predictions.extend(json.loads(line) for line in part.open())
            if args.score_only:
                if not args.subset_to_score or file_sha256(args.subset_to_score)!=args.expected_subset_sha256:
                    raise ValueError('Post-RL training-score subset changed from the RLCD input')
                subset=read_spatial_jsonl(args.subset_to_score,labeled_only=True)
                manifest=score_training_errors(records,predictions,subset,source_sha,identity)
                manifest['subset_sha256']=args.expected_subset_sha256
            else:
                manifest=select_errors(records,predictions,identity,source_sha,args.subset_dir)
            # Keep one compact replay file; temporary per-rank shards are unnecessary after coverage validation.
            with (output/'predictions.jsonl').open('w') as stream:
                for row in predictions:
                    stream.write(json.dumps(row)+'\n')
            manifest['predictions_sha256']=file_sha256(output/'predictions.jsonl')
            atomic_json(output/'pipeline_complete.json',manifest)
            shutil.rmtree(shards)
        context.barrier()
    finally:
        close()


if __name__=='__main__':
    main()
