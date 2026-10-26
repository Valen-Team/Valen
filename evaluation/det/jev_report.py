"""Verify complete Jev splits and compare them with accepted local Det models."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import time

from evaluation.det.jev import BENCHMARKS,digest,write_json

REPORTER_SHA256=digest(__file__)

def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def verify_predictions(path, expected_ids, model, metrics, truth_boxes=None):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids=[row['id'] for row in rows]
    if len(ids)!=len(expected_ids) or len(set(ids))!=len(ids) or set(ids)!=expected_ids:
        raise ValueError('Incomplete, duplicate or unexpected prediction IDs')
    if any(row.get('model')!=model or type(row.get('iou')) not in (float,int)
           or not math.isfinite(row['iou']) or not 0<=row['iou']<=1 for row in rows):
        raise ValueError('Wrong model identity or invalid IoU in the prediction cache')
    expected=dict(accuracy=sum(row['iou']>=.5 for row in rows)/len(rows),
                  mean_box_iou=sum(row['iou'] for row in rows)/len(rows))
    if metrics.get('samples')!=len(rows) or any(not math.isclose(metrics.get(key,float('nan')),value,
        rel_tol=1e-10,abs_tol=1e-12) for key,value in expected.items()):
        raise ValueError('Cached predictions disagree with the reported scores')
    if any(row.get('valid_output') is False and row['iou']!=0 for row in rows):
        raise ValueError('Invalid model output cannot receive a correct score')
    if truth_boxes is not None:
        for row in rows:
            prediction=row['box'];boxes=truth_boxes[row['id']]
            if len(boxes)>1:raise ValueError('Referring localization must have at most one target box')
            truth=boxes[0] if boxes else []
            if row.get('valid_output') is False:
                value=0.
            elif not prediction or not truth:
                value=float(not prediction and not truth)
            else:
                if (len(prediction)!=4 or not all(math.isfinite(x) and 0<=x<=1 for x in prediction)
                        or not prediction[0]<prediction[2] or not prediction[1]<prediction[3]):
                    raise ValueError('Prediction boxes must be normalized xyxy')
                x1=max(prediction[0],truth[0]);y1=max(prediction[1],truth[1])
                x2=min(prediction[2],truth[2]);y2=min(prediction[3],truth[3])
                overlap=max(0.,x2-x1)*max(0.,y2-y1)
                area_pred=(prediction[2]-prediction[0])*(prediction[3]-prediction[1])
                area_truth=(truth[2]-truth[0])*(truth[3]-truth[1])
                value=overlap/(area_pred+area_truth-overlap)
            if not math.isclose(value,row['iou'],rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError('Cached IoU differs from the original labeled box')
    if path.stem=='groundingme_test':
        from evaluation.det.groundingme import aggregate
        grouped=aggregate(rows)
        if grouped['groups']!=metrics.get('groups'):
            raise ValueError('GroundingME group scores disagree with the prediction cache')
    latency=sorted(row['elapsed_seconds'] for row in rows)
    return dict(samples=len(rows),predictions_sha256=digest(path),all_expected_ids_verified=True,
        original_box_iou_recomputed=truth_boxes is not None,
        request_latency_seconds=dict(mean=sum(latency)/len(latency),p95=latency[min(len(latency)-1,int(.95*len(latency)))]))


def datasets(config, data_root):
    if set(config['benchmarks'])!=set(BENCHMARKS) or config.get('limit_per_benchmark')!=0:
        raise ValueError('A full report requires all eleven complete benchmark splits')
    result={}
    for name,identity in config['benchmarks'].items():
        base=data_root/('reasonseg' if name.startswith('reasonseg_') else 'groundingme' if name.startswith('groundingme_') else '')
        path=base/'processed'/f'{name}.jsonl'
        if digest(path)!=identity['records_sha256'] or identity['selected_samples']!=identity['full_samples']:
            raise ValueError('Full evaluation data differs from the sealed request configuration')
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids={row['id']:row['targets']['target']['boxes'] for row in rows}
        if len(rows)!=identity['full_samples'] or len(ids)!=len(rows):
            raise ValueError('The benchmark does not contain the expected unique IDs')
        result[name]=ids
    if sum(map(len,result.values()))!=59441:
        raise ValueError('The eleven-split evaluation must cover 59,441 queries')
    return result


def local_baselines(registry, data_root, model_root):
    if registry is None:
        return {}
    from scripts.train.summarize_spatial import collect_results
    state=read_json(registry)
    models=collect_results(state,data_root,model_root)['models']
    result={}
    for size in ('0.8B','2B','4B'):
        row=next(m for m in models if m['task']=='det' and m['size']==size and m['variant']=='lora')
        if not row['deliverables_ready']:
            raise ValueError('Local baseline is not an accepted complete Det model: '+size)
        result[size]=dict(model=row['name'],benchmarks={**row['benchmarks'],**row['reasonseg_benchmarks'],
            **row['groundingme_benchmarks']},inference=row['inference'],export_identity=row['export_identity'])
    return result


def comparison_rows(config, complete, baselines):
    result={}
    for name,identity in config['benchmarks'].items():
        local={}
        for size,row in baselines.items():
            metric=row['benchmarks'][name]
            if metric['samples']!=identity['full_samples'] or metric['data_sha256']!=identity['records_sha256']:
                raise ValueError('Cannot compare benchmark scores from different records')
            local[size]=metric['accuracy']
        api=complete.get(name,{})
        result[name]=dict(samples=identity['full_samples'],data_sha256=identity['records_sha256'],
            jev_accuracy=api.get('accuracy'),valen_lora_accuracy=local,
            valen_minus_jev_pp={size:100*(score-api['accuracy']) for size,score in local.items()} if api else {})
    return result


def markdown(report):
    complete=report['complete_full_evaluation']
    lines=['# Jev / Valen Det benchmark 对照','',
        ('全部11项完整评测已核验。' if complete else '进行中：仅展示已经完整跑完并逐ID核验的split；—表示该项尚未完整完成。'),
        f"已完成split {len(report['jev_benchmarks'])}/11；已保存预测 {report['predictions_saved']:,}/59,441。",'',
        '| Benchmark | 完整样本数 | Jev 1.14.0 | Valen 0.8B LoRA | Valen 2B LoRA | Valen 4B LoRA |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    def percent(value):return '—' if value is None else f'{100*value:.2f}%'
    for name,row in report['comparison'].items():
        local=row['valen_lora_accuracy']
        lines.append(f"| {name} | {row['samples']:,} | {percent(row['jev_accuracy'])} | {percent(local.get('0.8B'))} | {percent(local.get('2B'))} | {percent(local.get('4B'))} |")
    lines+=['','均按原始指令、固定坐标与IoU≥0.5计分。所有对照清单SHA与完整样本数一致。',
        '图像预算均为262,144像素；Jev为满足API边长限制使用居中白色padding并反变换坐标，本地Valen按Qwen预处理。',
        'ReasonSeg/GroundingME的Jev请求统一使用Box+Noul，presence阈值固定0.5；本地模型使用自己的空间无目标动作。',
        'API延迟包含网络/重试，不能直接当作本地GPU推理耗时。验证划分与最终test分别列出，val分数不作独立test成绩。',
        '完整分组、预测/数据身份、API设置及对照差值保存在comparison.json。','']
    return '\n'.join(lines)


def run(args):
    config=read_json(args.run_dir/'request-config.json')
    ids=datasets(config,args.data_root)
    baselines=local_baselines(args.registry,args.data_root.parent.parent,args.model_root)
    if any(row['inference']['max_pixels']!=config['max_pixels'] for row in baselines.values()):
        raise ValueError('Comparison requires matching declared image-pixel budgets')
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    cache={}
    while True:
        complete={};proofs={}
        sources=[args.run_dir]+(sorted(p.parent for p in args.workers_root.glob('*/request-config.json')) if args.workers_root else [])
        for source in sources:
            source_config=read_json(source/'request-config.json')
            # Key routing and sharding may differ; request semantics must match.
            if {k:v for k,v in source_config.items() if k!='benchmarks'}!={k:v for k,v in config.items() if k!='benchmarks'}:
                raise ValueError('A worker changed the sealed request settings')
            for name,identity in source_config['benchmarks'].items():
                if identity!=config['benchmarks'][name]:raise ValueError('A worker changed benchmark identity')
            for filename in ('partial-metrics.json','metrics.json'):
                partial=read_json(source/filename)
                if not partial:continue
                if partial['model']!=config['model']:raise ValueError('A worker returned a different model version')
                for name,metric in partial['benchmarks'].items():
                    if not metric.get('complete_split'):continue
                    if metric['data_sha256']!=config['benchmarks'][name]['records_sha256']:
                        raise ValueError('A score refers to different benchmark records')
                    path=source/(name+'.jsonl');stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
                    if key not in cache:cache[key]=verify_predictions(path,set(ids[name]),config['model'],metric,truth_boxes=ids[name])
                    if name in complete and complete[name]!=metric:raise ValueError('Conflicting complete split reports')
                    complete[name]=metric;proofs[name]=cache[key]
        final=read_json(args.run_dir/'metrics.json')
        finished=(final.get('complete_full_benchmarks') is True and set(complete)==set(BENCHMARKS)
                  and sum(row['samples'] for row in complete.values())==59441)
        status=read_json(args.run_dir/'status.json')
        report=dict(format='valen_jev_det_comparison_v1',time_utc=datetime.now(timezone.utc).isoformat(),
            complete_full_evaluation=finished,model=config['model'],predictions_saved=status.get('predictions_completed',0),
            jev_benchmarks=complete,prediction_verification=proofs,baselines=baselines,
            comparison=comparison_rows(config,complete,baselines),request_config_sha256=digest(args.run_dir/'request-config.json'),
            verification_code_sha256=REPORTER_SHA256)
        write_json(output/'comparison.json',report)
        target=output/'comparison.md';temp=target.with_suffix('.tmp');temp.write_text(markdown(report));temp.replace(target)
        print(json.dumps(dict(complete_full_evaluation=finished,complete_splits=len(complete),predictions_saved=report['predictions_saved'])),flush=True)
        if finished:
            write_json(output/'pipeline_complete.json',dict(model=config['model'],samples=59441,split_count=11,
                comparison_sha256=digest(output/'comparison.json'),all_prediction_ids_verified=True))
        if finished or not args.watch:return
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','data-root','output-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers-root',type=Path)
    parser.add_argument('--registry',type=Path)
    parser.add_argument('--model-root',type=Path)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    if args.registry and not args.model_root:parser.error('--registry requires --model-root')
    from filelock import FileLock
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with FileLock(str(args.output_dir/'.report.lock'),timeout=0):run(args)


if __name__=='__main__':
    main()
