"""Full UI-Vision, OSWorld-G refined and MMBench-GUI L2 grounding evaluation."""
import argparse
import copy
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

from scripts.data.prepare_spatial import digest_file, write_json


EXPECTED = dict(uivision_basic=1772,uivision_functional=1772,uivision_spatial=1935,
                osworld_g_refined=564,mmbench_gui_l2=3594)


def grounding_point(question, output):
    """Use the same explicit no-object action for every sample, without labels."""
    import torch
    from valen.modeling.qwen.spatial import decode_spatial
    if not torch.isfinite(output.class_logits).all():
        return None
    null_action = bool((output.class_logits.argmax(-1) == len(question.categories)).all())
    inference_question = copy.copy(question)
    inference_question.metadata = dict(question.metadata, allow_empty_target=True)
    prediction = decode_spatial(inference_question, output)
    return [-1., -1.] if null_action else prediction.get('point')


def score(point, metadata):
    """Match official inclusive boxes, ray-cast polygons and explicit refusal."""
    if point is None or len(point)!=2 or not all(math.isfinite(x) for x in point):
        return False
    kind=metadata['target_type']
    if kind=='refusal':
        return all(x<0 for x in point)
    if kind=='bbox':
        x1,y1,x2,y2=metadata['official_box']
        return x1<=point[0]<=x2 and y1<=point[1]<=y2
    if kind!='polygon':
        raise ValueError('Unknown target type')
    polygon=metadata['official_polygon']
    x,y=point
    inside=False
    j=len(polygon)-1
    for i,(xi,yi) in enumerate(polygon):
        xj,yj=polygon[j]
        if (yi>y)!=(yj>y) and x<(xj-xi)*(y-yi)/(yj-yi)+xi:
            inside=not inside
        j=i
    return inside


def aggregate(rows):
    if not rows:
        raise ValueError('Empty benchmark')
    buckets=defaultdict(list)
    for row in rows:
        for key,value in row['groups'].items():
            values=value if isinstance(value,list) else [value]
            for item in values:
                buckets[key+'/'+str(item)].append(row['correct'])
    result = dict(samples=len(rows),accuracy=sum(row['correct'] for row in rows)/len(rows),
        groups={key:dict(samples=len(values),accuracy=sum(values)/len(values)) for key,values in sorted(buckets.items())})
    if all('latency_seconds' in row for row in rows):
        latency=sorted(row['latency_seconds'] for row in rows)
        result['latency_seconds']=dict(mean=sum(latency)/len(latency),
            p50=latency[len(latency)//2],p95=latency[min(len(latency)-1,int(.95*len(latency)))])
    return result


def official_summary(benchmarks):
    levels=benchmarks['mmbench_gui_l2']['groups']
    return dict(uivision_element_average=sum(benchmarks['uivision_'+kind]['accuracy'] for kind in
        ('basic','functional','spatial'))/3,
        mmbench_l2_average=(levels['grounding_type/basic']['accuracy']+
                            levels['grounding_type/advanced']['accuracy'])/2,
        osworld_g_refined_accuracy=benchmarks['osworld_g_refined']['accuracy'])


def results_ready(report, marker, seal, export_path):
    """Require complete scores tied to this export and the sealed evaluation data."""
    try:
        export_path=Path(export_path)
        export=json.loads(export_path.read_text())
        identity=digest_file(export_path)
        rows=report['benchmarks']
        if not (seal.get('validated') is True and report.get('complete_official_splits') is True
                and set(rows)==set(EXPECTED)==set(seal['benchmarks'])
                and marker.get('samples')==sum(EXPECTED.values())
                and marker.get('model')==report.get('model')==export['model_name']
                and marker.get('model_export_manifest_sha256')==report.get('model_export_manifest_sha256')==identity
                and report['training_overlap']['training_records_sha256']==seal['training_records_sha256']==export['training_data_sha256']):
            return False
        for name,count in EXPECTED.items():
            row=rows[name]
            if (row['samples']!=count or row['data_sha256']!=seal['benchmarks'][name]['records_sha256']
                    or type(row['accuracy']) not in (int,float) or not 0<=row['accuracy']<=1
                    or not row.get('groups')):
                return False
            for group in row['groups'].values():
                if type(group['samples']) is not int or group['samples']<=0 or not 0<=group['accuracy']<=1:
                    return False
        expected=official_summary(rows)
        return set(report['official_summary'])==set(expected) and all(
            math.isfinite(report['official_summary'][key]) and
            math.isclose(report['official_summary'][key],value,rel_tol=1e-9,abs_tol=1e-12)
            for key,value in expected.items())
    except (OSError,ValueError,TypeError,KeyError):
        return False


def validate(root):
    seal=json.loads((root/'manifests/ready.json').read_text())
    if not seal.get('validated') or set(seal['benchmarks'])!=set(EXPECTED):
        raise ValueError('Prepare all five complete official grounding splits first')
    overlap=root/'manifests/training-overlap.json'
    if digest_file(overlap)!=seal['training_overlap_manifest_sha256']:
        raise ValueError('Training overlap report changed after sealing')
    if digest_file(root/'manifests/osworld-buckets.json')!=seal['osworld_buckets_sha256']:
        raise ValueError('Official capability groups changed after sealing')
    for name,count in EXPECTED.items():
        manifest=json.loads((root/'manifests'/f'{name}.json').read_text())
        if (manifest['samples']!=count or manifest['records_sha256']!=seal['benchmarks'][name]['records_sha256']
                or digest_file(root/'processed'/f'{name}.jsonl')!=manifest['records_sha256']):
            raise ValueError('Incomplete or modified official split: '+name)
    return seal,json.loads(overlap.read_text())


def worker(args):
    import torch
    from evaluation.spatial import load_spatial_model
    from valen.data.spatial import read_spatial_jsonl
    from valen.training.distributed import initialize,close
    context=initialize()
    root,output=Path(args.data_root),Path(args.output_dir)
    try:
        seal,overlap=validate(root)
        package=Path(args.model_path)
        export_path=package/'export_manifest.json'
        export=json.loads(export_path.read_text())
        if export['training_data_sha256']!=seal['training_records_sha256']:
            raise ValueError('Overlap audit must reference this model training corpus')
        model,compiler=load_spatial_model(package,context.device,data_root=root)
        # OSWorld-G refined contains one original empty instruction. Preserve it
        # verbatim through the evaluation-only opt-in; training stays strict.
        compiler.allow_empty_instruction=True
        benchmarks={}
        buckets=json.loads((root/'manifests/osworld-buckets.json').read_text())
        media={}
        with torch.inference_mode():
            for name,count in EXPECTED.items():
                manifest=json.loads((root/'manifests'/f'{name}.json').read_text())
                # Assign each media hash check to exactly one rank.
                for relative,expected in list(manifest['media_sha256'].items())[context.rank::context.world_size]:
                    if relative not in media and digest_file(root/relative)!=expected:
                        raise ValueError('Benchmark screenshot changed')
                    media[relative]=expected
                records=read_spatial_jsonl(root/'processed'/f'{name}.jsonl',labeled_only=True,allow_empty_instruction=True)
                if len(records)!=count or len({r['id'] for r in records})!=count:
                    raise ValueError('Benchmark contains missing or duplicate IDs')
                rows=[]
                for record in records[context.rank::context.world_size]:
                    started=time.monotonic()
                    question=compiler.compile(record).questions[0]
                    point=grounding_point(question,model(question))
                    meta=record['meta']
                    groups={key:meta[key] for key in ('platform','application','ui_type','target_type') if key in meta}
                    relative=record['request']['state']['messages'][0]['content'][0]['image_url']['url']
                    matches=overlap['overlaps'].get(relative, [])
                    groups['training_overlap']=('exact' if any(m['match'] in {'file','rgb'} for m in matches)
                        else 'near_candidate' if matches else 'none')
                    if name=='mmbench_gui_l2':groups['grounding_type']=meta['group']
                    if name=='osworld_g_refined':
                        groups['capability']=(['refusal'] if meta['target_type']=='refusal' else
                            [group for group,types in buckets.items() if set(types)&set(meta['gui_types'])])
                    rows.append(dict(id=record['id'],point=point,correct=score(point,meta),groups=groups,
                        latency_seconds=time.monotonic()-started))
                all_rows=[row for shard in context.gather(rows) for row in shard]
                if len(all_rows)!=count or {r['id'] for r in all_rows}!={r['id'] for r in records}:
                    raise ValueError('Distributed predictions do not cover the complete split')
                benchmarks[name]=aggregate(all_rows)
                benchmarks[name].update(data_sha256=manifest['records_sha256'],
                    training_overlap_images=overlap['overlap_images'][name],
                    training_overlap_records=sum(row['groups']['training_overlap']!='none' for row in all_rows))
                if context.primary:
                    output.mkdir(parents=True,exist_ok=True)
                    with (output/f'{name}.jsonl').open('w') as stream:
                        for row in sorted(all_rows,key=lambda r:r['id']):stream.write(json.dumps(row)+'\n')
                    write_json(output/'partial-metrics.json',benchmarks)
        if context.primary:
            report=dict(model=export['model_name'],model_export_manifest_sha256=digest_file(export_path),
                benchmarks=benchmarks,official_summary=official_summary(benchmarks),
                complete_official_splits=True,training_overlap=overlap,
                inference=dict(max_pixels=compiler.max_pixels,max_length=compiler.max_length,
                    point='center of highest-confidence non-null box',
                    empty_decision='explicit categorical no-object argmax maps to (-1,-1); invalid output is failure',
                    null_action_enabled_for_every_sample=True,
                    original_empty_instructions_preserved=True,
                    checkpoint_selection='fixed independent development set'),
                evaluation_code_sha256=digest_file(Path(__file__)))
            write_json(output/'metrics.json',report)
    finally:
        close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','model-path','output-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--nodes',type=int,default=1)
    parser.add_argument('--gpus-per-node',type=int,default=8)
    parser.add_argument('--launch-id',default='additional-gui')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        worker(args)
        return
    from valen.training.spatial_cluster import SpatialCluster
    root,output=Path(args.data_root),Path(args.output_dir)
    validate(root)
    cluster=SpatialCluster(output/'coordination'/args.launch_id,args.nodes,args.gpus_per_node)
    cluster.execute(['-m','evaluation.gui.additional','--worker','--data-root',str(root),
        '--model-path',args.model_path,'--output-dir',str(output)],distributed=True)
    if cluster.primary:
        report=json.loads((output/'metrics.json').read_text())
        if set(report['benchmarks'])!=set(EXPECTED) or sum(v['samples'] for v in report['benchmarks'].values())!=9637:
            raise ValueError('Additional grounding evaluation is incomplete')
        write_json(output/'pipeline_complete.json',dict(model=report['model'],samples=9637,
            model_export_manifest_sha256=report['model_export_manifest_sha256']))


if __name__=='__main__':
    main()
