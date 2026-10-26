"""Accepted merged policy -> its original-training errors -> spatial RLCD."""
import argparse
import json
import math
from pathlib import Path
import shutil

from scripts.train.spatial_cluster import SpatialCluster
from scripts.train.spatial_worker import atomic_json


def compare_models(lora_run, full_run, rl_run, full_model, rl_model, reference_variant='full', reference_model=None):
    from valen.training.checkpoint import file_sha256
    rl_label='lora_rlcd' if reference_variant=='lora_merged' else 'full_rlcd'
    runs={'lora':Path(lora_run),rl_label:Path(rl_run)}
    if full_run is not None:
        runs['full']=Path(full_run)
    suites={}
    for name,run in runs.items():
        values={}
        for folder in ['evaluation','evaluation-reasonseg','evaluation-groundingme']:
            report=json.loads((run/folder/'metrics.json').read_text())
            values.update(report['benchmarks'])
        if len(values)!=11:
            raise ValueError('RLCD comparison requires all eleven complete evaluation items')
        suites[name]=values
    if any(set(v)!=set(suites['lora']) for v in suites.values()):
        raise ValueError('LoRA, Full and RLCD use different benchmark sets')
    reference_model=reference_model or full_model
    result={'format':'valen_spatial_rlcd_comparison_v1','benchmarks':{},
            'initialization_variant':reference_variant,
            'reference_export_manifest_sha256':file_sha256(Path(reference_model)/'export_manifest.json'),
            'rlcd_export_manifest_sha256':file_sha256(Path(rl_model)/'export_manifest.json')}
    if full_model is not None:
        result['full_export_manifest_sha256']=file_sha256(Path(full_model)/'export_manifest.json')
    from scripts.train.summarize_spatial import completed_costs
    result['training']={}
    for name,run in runs.items():
        metric=run/'joint/metrics.jsonl'
        config=json.loads((run/'joint/run_manifest.json').read_text())['config']
        result['training'][name]={**completed_costs(str(metric),metric.stat().st_mtime_ns),
            'world_size':config['required_world_size'],
            'global_batch':config['required_world_size']*config['gradient_accumulation'],
            'training_data_sha256':config['data_sha256']}
    before=json.loads((Path(rl_run)/'replay/pipeline_complete.json').read_text())
    after=json.loads((Path(rl_run)/'post-replay/pipeline_complete.json').read_text())
    source_label='lora' if reference_variant=='lora_merged' else 'full'
    result['training_errors']={source_label:{'records':100000,'errors':before['errors'],'error_rate':before['errors']/100000,
                                      'selected_errors':before['selected'],'selected_error_rate':1.},
                               rl_label:after}
    for benchmark,reference in suites['lora'].items():
        if any(row[benchmark]['data_sha256']!=reference['data_sha256'] or
               row[benchmark]['samples']!=reference['samples'] for row in suites.values()):
            raise ValueError('RLCD comparison evaluation data/counts differ')
        fields=[k for k in ['accuracy','mean_box_iou','accuracy_at_075','accuracy_at_09'] if k in reference]
        values={name:{k:row[benchmark][k] for k in fields} for name,row in suites.items()}
        result['benchmarks'][benchmark]={'samples':reference['samples'],'data_sha256':reference['data_sha256'],
            **values,'rlcd_minus_lora':{k:values[rl_label][k]-values['lora'][k] for k in fields},
            'inference_costs':{name:{k:row[benchmark].get(k) for k in
                              ['elapsed_seconds','world_size','latency_seconds','peak_cuda_bytes_per_rank']}
                               for name,row in suites.items()}}
        if 'full' in values:
            result['benchmarks'][benchmark]['rlcd_minus_full']={k:values[rl_label][k]-values['full'][k] for k in fields}
    atomic_json(Path(rl_run)/'comparison.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['model-path','lora-baseline-run','data-root',
                 'output-dir','subset-dir','export-dir','acceptance-marker','launch-id']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--reference-variant',choices=('full','lora_merged'),default='full')
    parser.add_argument('--reference-model')
    parser.add_argument('--full-baseline-run')
    parser.add_argument('--full-model')
    parser.add_argument('--nodes',type=int,default=16)
    parser.add_argument('--gpus-per-node',type=int,default=4)
    args=parser.parse_args()
    from valen.training.checkpoint import file_sha256
    marker=Path(args.acceptance_marker)
    if not marker.is_file():raise ValueError('Accept the complete reference evaluation before RLCD')
    accepted=json.loads(marker.read_text())
    reference_model=args.reference_model or args.full_model
    if not reference_model:parser.error('Provide --reference-model or --full-model')
    reference_name=Path(reference_model).name
    required_names=({reference_name,reference_name[:-5]} if args.reference_variant=='full' else {reference_name})
    if ((args.reference_variant=='full' and (not reference_name.endswith('-Full') or not args.full_baseline_run))
            or not required_names<=set(accepted['models'])):
        raise ValueError('RLCD acceptance gate does not identify the requested matching baselines')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    source=Path(args.lora_baseline_run if args.reference_variant=='lora_merged' else args.full_baseline_run)
    base=json.loads((source/'joint/run_manifest.json').read_text())['config']
    if Path(base['model_path']).resolve()!=Path(args.model_path).resolve():
        raise ValueError('RLCD backbone differs from the accepted reference model')
    export_identity=json.loads((Path(reference_model)/'export_manifest.json').read_text())
    if (accepted['export_identities'][reference_name]['source_checkpoint_sha256']!=export_identity['source_checkpoint_sha256']
            or export_identity['training_data_sha256']!=base['data_sha256']):
        raise ValueError('Reference identity changed after its complete baseline evaluation')
    if args.reference_variant=='lora_merged':
        import struct
        with (Path(reference_model)/'model.safetensors').open('rb') as stream:
            header=json.loads(stream.read(struct.unpack('<Q',stream.read(8))[0]))
        if (export_identity.get('finetuning_type')!='lora'
                or export_identity.get('reload_validation',{}).get('independent_process') is not True
                or any('lora_' in key.lower() or 'adapter' in key.lower() for key in header)):
            raise ValueError('RLCD requires standalone merged LoRA weights, without adapter tensors')
    cluster=SpatialCluster(output/'coordination'/args.launch_id,args.nodes,args.gpus_per_node)
    if cluster.primary:atomic_json(output/'stage.json',{'stage':'replay'})
    if cluster.decision('replay',not (output/'replay/pipeline_complete.json').is_file()):
        cluster.execute(['-m','scripts.train.replay_det_errors','--training-data',base['data'],
            '--data-root',base['data_root'],'--model-path',reference_model,'--output-dir',str(output/'replay'),
            '--subset-dir',args.subset_dir,'--expected-source-sha256',base['data_sha256']],distributed=True)
    subset=Path(args.subset_dir)
    hard=json.loads((subset/'manifest.json').read_text())
    if (hard.get('validated') is not True or hard.get('original_records')!=100000
            or hard.get('original_training_sha256')!=base['data_sha256']
            or not 0<=hard.get('errors',-1)<=100000 or hard.get('selected')!=min(hard['errors'],20000)):
        raise ValueError('Hard cases do not come from the complete sealed original training replay')
    if hard['model_export_manifest_sha256']!=file_sha256(Path(reference_model)/'export_manifest.json'):
        raise ValueError('Hard cases were selected by a different reference model')
    if cluster.decision('zero-errors',hard['errors']==0):
        if cluster.primary:
            atomic_json(output/'pipeline_complete.json',{'zero_error_skip':True,'replay_records':100000,
                'model_export_manifest_sha256':hard['model_export_manifest_sha256'],
                'original_training_sha256':hard['original_training_sha256'],'errors':0})
        return
    export=Path(args.export_dir)
    if cluster.primary:
        if export.is_symlink():
            if (export/'export_manifest.json').exists():raise ValueError('Never replace an existing accepted RLCD model')
            export.unlink()
        export.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in base.items() if k not in ['data_sha256','device','comparison_resource_override','max_steps']}
    code_root=Path(__file__).resolve().parents[2]
    config.update(json.loads((code_root/'configs/train/qwen/det/rlcd.json').read_text()))
    code=code_root/'source_manifest.json'
    config.update(data=str(subset/'train.jsonl'),hard_case_manifest=str(subset/'manifest.json'),
        original_training_sha256=base['data_sha256'],reference_export=reference_model,
        initialization_variant=args.reference_variant,reference_model_name=reference_name,
        initialization_sha256=hard['model_export_manifest_sha256'],
        output=str(output/'joint'),required_world_size=args.nodes*args.gpus_per_node,
        code_sha256=json.loads(code.read_text())['sha256'] if code.exists() else None)
    # Full warmup scheduling is replaced by the short RL schedule, whose length depends on selected errors.
    config.pop('lr_warmup_steps',None)
    group_steps=math.ceil(hard['selected']/(config['required_world_size']*config['gradient_accumulation']))
    config['rl_warmup_updates']=min(20,max(1,math.ceil(group_steps*config['rlcd']['num_iterations']*.05)))
    path=output/'joint.json'
    if cluster.primary:atomic_json(path,config)
    if cluster.decision('rl-preflight',not (output/'joint/latest/checkpoint.pt').is_file()):
        if cluster.primary:atomic_json(output/'stage.json',{'stage':'rl-preflight'})
        smoke=output/'rl-preflight.json'
        if cluster.primary:atomic_json(smoke,dict(config,max_steps=2))
        cluster.execute(['-m','valen.train','--config',str(smoke)],distributed=True)
        if cluster.primary:
            record=json.loads((output/'joint/run_manifest.json').read_text())
            complete=json.loads((output/'joint/training_complete.json').read_text())
            lora_rl=config['finetuning_type']=='lora'
            audit='parameter-update-audit.json' if lora_rl else 'full-update-audit.json'
            gradients=json.loads((output/'joint'/audit).read_text())
            valid_parameters=(0<record['trainable_parameters']<record['total_parameters'] if lora_rl
                              else record['trainable_parameters']==record['total_parameters'])
            if record['world_size']!=args.nodes*args.gpus_per_node or not valid_parameters:
                raise ValueError('RLCD real preflight used the wrong tuning method or world size')
            if not gradients.get('all_groups_have_gradients') or not gradients.get('all_groups_updated'):
                raise ValueError('RLCD real preflight lacks verified trainable parameter updates')
            if lora_rl and (gradients.get('trainable_groups')!=['head','lora','merger']
                            or gradients.get('frozen_backbone_unchanged') is not True):
                raise ValueError('LoRA RLCD changed the frozen backbone or used incorrect parameter groups')
            atomic_json(output/'rl-preflight-complete.json',{'world_size':record['world_size'],'group_steps':complete['step'],
                'finetuning_type':config['finetuning_type'],
                'fixed_reference_manifest_sha256':config['initialization_sha256'],'resume_required':True})
    if cluster.decision('rl-training',not (output/'rl-training-resume-accepted.json').exists()):
        if cluster.primary:atomic_json(output/'stage.json',{'stage':'rl-training'})
        cluster.execute(['-m','valen.train','--config',str(path),'--resume',str(output/'joint/latest')],distributed=True)
        if cluster.primary:
            record=json.loads((output/'joint/run_manifest.json').read_text())
            complete=json.loads((output/'joint/training_complete.json').read_text())
            if complete['stop_reason']!='epochs_complete' or record['starting_step']<1:
                raise ValueError('RLCD formal training did not restore and finish its epoch')
            atomic_json(output/'rl-training-resume-accepted.json',{'world_size':record['world_size'],'starting_step':record['starting_step'],
                'final_step':complete['step'],'selected':hard['selected'],'fixed_reference':True})
    if cluster.decision('export',not (export/'export_manifest.json').is_file()):
        if cluster.primary:atomic_json(output/'stage.json',{'stage':'export'})
        cluster.execute(['-m','scripts.export.qwen.export_spatial','--checkpoint',str(output/'joint/best'),
            '--output-dir',str(export),'--data',config['validation_data'],'--data-root',config['validation_data_root']])
    if cluster.decision('post-replay',not (output/'post-replay/pipeline_complete.json').is_file()):
        if cluster.primary:atomic_json(output/'stage.json',{'stage':'post-replay'})
        cluster.execute(['-m','scripts.train.replay_det_errors','--training-data',base['data'],
            '--data-root',base['data_root'],'--model-path',str(export),'--output-dir',str(output/'post-replay'),
            '--subset-dir',args.subset_dir,'--expected-source-sha256',base['data_sha256'],
            '--score-only','--subset-to-score',str(subset/'train.jsonl'),
            '--expected-subset-sha256',hard['records_sha256']],distributed=True)
    if cluster.primary:atomic_json(output/'stage.json',{'stage':'evaluation'})
    cluster.execute(['-m','evaluation.det.evaluate','--data-root',str(Path(args.data_root)/'eval/det'),
                     '--model-path',str(export),'--output-dir',str(output/'evaluation'),'--coco-protocol','none'],distributed=True)
    cluster.execute(['-m','evaluation.reasonseg','--task','det','--data-root',str(Path(args.data_root)/'eval/det/reasonseg'),
        '--model-path',str(export),'--output-dir',str(output/'evaluation-reasonseg'),'--nodes',str(args.nodes),
        '--gpus-per-node',str(args.gpus_per_node),'--launch-id',args.launch_id+'-reasonseg'],all_nodes=True)
    cluster.execute(['-m','evaluation.det.groundingme','--data-root',str(Path(args.data_root)/'eval/det/groundingme'),
        '--model-path',str(export),'--output-dir',str(output/'evaluation-groundingme')],distributed=True)
    cluster.execute(['-m','evaluation.examples','--task','det','--data-root',str(Path(args.data_root)/'eval/det'),
        '--model-path',str(export),'--output-dir',str(output/'examples'),'--nodes',str(args.nodes),
        '--gpus-per-node',str(args.gpus_per_node),'--launch-id',args.launch_id+'-examples'],all_nodes=True)
    if cluster.primary:
        compare_models(args.lora_baseline_run,args.full_baseline_run,output,args.full_model,export,
                       args.reference_variant,reference_model)
        atomic_json(output/'pipeline_complete.json',{'task':'det','model':export.name,'method':'spatial_rlcd',
                    'finetuning_type':config['finetuning_type'],
                    'training_records':hard['selected'],'initialization_variant':args.reference_variant,
                    'reference_manifest_sha256':hard['model_export_manifest_sha256']})
        atomic_json(output/'stage.json',{'stage':'completed'})
        for folder in ['latest','best']:shutil.rmtree(output/'joint'/folder,ignore_errors=True)


if __name__=='__main__':
    main()
