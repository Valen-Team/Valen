"""Spatial RLCD with LoRA or full updates, a fixed merged policy and exact resume."""
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch

from valen.data.spatial import read_spatial_jsonl
from valen.modeling.factory import build_model, build_compiler, optimizer_groups
from .checkpoint import capture_rank_state, file_sha256, load_checkpoint, save_checkpoint
from .distributed import initialize, epoch_shard, broadcast_trainable, synchronize_gradients
from .spatial import normalize_spatial_config, gpu_telemetry
from .spatial_rlcd_loss import SpatialRLCDObjective


def initialize_full_export(model, path, context):
    """Stream all accepted spatial-package weights into FP32 trainable parameters."""
    from safetensors import safe_open
    path=Path(path)
    manifest=json.loads((path/'export_manifest.json').read_text())
    if context.primary:
        if not manifest.get('files_sha256') or manifest.get('reload_validation',{}).get('independent_process') is not True:
            raise ValueError('RLCD requires an independently reloaded complete reference export')
        for filename,digest in manifest['files_sha256'].items():
            if file_sha256(path/filename)!=digest:
                raise ValueError('Reference package changed after export')
    context.barrier()
    metadata=path/'config.json'
    if metadata.is_file() and json.loads(metadata.read_text()).get('merger_fp32'):
        model.backbone.visual.merger.float()
    state=model.state_dict()
    with safe_open(str(path/'model.safetensors'),framework='pt',device='cpu') as stored:
        if set(stored.keys())!=set(state):
            raise ValueError('Reference package tensors differ from the trainable architecture')
        with torch.no_grad():
            for name in stored.keys():
                value=stored.get_tensor(name)
                if value.shape!=state[name].shape:
                    raise ValueError('Reference tensor shape changed')
                state[name].copy_(value)


def initialized_policy(config, context, make_reference=True):
    """Load the merged SFT base before attaching fresh RL adapters."""
    lora=config.get('finetuning_type')=='lora'
    construction=dict(config,stage='warmup',gradient_checkpointing=False) if lora else config
    model=build_model(construction)
    identity=file_sha256(Path(config['reference_export'])/'export_manifest.json')
    if config.get('initialization_sha256') and identity!=config['initialization_sha256']:
        raise ValueError('The immutable merged initialization export changed')
    initialize_full_export(model,config['reference_export'],context)
    if lora:
        model.backbone.requires_grad_(False)
        model.head.float().requires_grad_(True)
        model.backbone.visual.merger.float().requires_grad_(True)
        model.backbone_autocast_dtype=torch.bfloat16 if config.get('dtype','bf16')=='bf16' else None
    # The fixed reference has the original merged weights and no RL updates.
    reference=copy.deepcopy(model).eval().requires_grad_(False) if make_reference else None
    if lora:
        from peft import LoraConfig,get_peft_model
        targets=[name for name,module in model.backbone.language_model.named_modules()
                 if isinstance(module,torch.nn.Linear)]
        model.backbone.language_model=get_peft_model(model.backbone.language_model,LoraConfig(
            r=config.get('lora_rank',32),lora_alpha=config.get('lora_alpha',64),
            lora_dropout=0.,target_modules=targets,bias='none'))
        model.lora_targets=targets
        if config.get('gradient_checkpointing',True):
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    elif not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Full RLCD must open the entire model')
    return model,reference


def frozen_backbone_matches_reference(model, reference):
    """Check every frozen base parameter, stripping only PEFT wrapper prefixes."""
    baseline=dict(reference.named_parameters())
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        original=name.replace('backbone.language_model.base_model.model.',
                              'backbone.language_model.').replace('.base_layer.','.')
        if original not in baseline or not torch.equal(parameter,baseline[original]):
            return False
    return True


@torch.no_grad()
def validate(model, compiler, records, context):
    from valen.modeling.qwen.spatial import decode_spatial
    from .spatial_loss import box_iou
    model.eval()
    count=correct=overlap_sum=0
    for record in records[context.rank::context.world_size]:
        question=compiler.compile(record,labeled_only=True).questions[0]
        predicted=decode_spatial(question,model(question))['boxes']
        truth=question.target.get('boxes',[])
        if not truth:
            overlap=float(not predicted);hit=not predicted
        else:
            overlap=float(box_iou(torch.tensor([predicted[0]['bbox']]),torch.tensor(truth)).max()) if predicted else 0.
            hit=bool(predicted and overlap>=.5)
        count+=1;correct+=int(hit);overlap_sum+=overlap
    count,correct,overlap_sum=context.sum([count,correct,overlap_sum])
    if not count:raise ValueError('Empty RLCD development set')
    result={'accuracy':correct/count,'mean_box_iou':overlap_sum/count}
    model.train()
    return result


def run(config, resume=None, initialize_from=None):
    if initialize_from:
        raise ValueError('Spatial RLCD initializes from reference_export, not a partial checkpoint')
    config=normalize_spatial_config(config)
    if config['method']!='rlcd':
        raise ValueError('Use method=rlcd for spatial reinforcement learning')
    context=initialize(config.get('device','cuda'))
    if context.world_size!=config['required_world_size']:
        raise ValueError('RLCD training world size differs from the fixed GPU allocation')
    config=dict(config,device=context.device)
    random.seed(config['seed']);torch.manual_seed(config['seed'])
    rng=random.Random(config['seed']+context.rank)
    config['data_sha256']=file_sha256(config['data'])
    config['validation_sha256']=file_sha256(config['validation_data'])
    config['reference_manifest_sha256']=file_sha256(Path(config['reference_export'])/'export_manifest.json')
    config['hard_case_manifest_sha256']=file_sha256(config['hard_case_manifest'])
    manifest=json.loads(Path(config['hard_case_manifest']).read_text())
    records=read_spatial_jsonl(config['data'],labeled_only=True)
    if (manifest.get('validated') is not True or manifest['selected']!=len(records)
            or manifest['records_sha256']!=config['data_sha256']
            or manifest['model_export_manifest_sha256']!=config['reference_manifest_sha256']
            or manifest['original_training_sha256']!=config['original_training_sha256']
            or len(records)>20000):
        raise ValueError('RLCD hard cases differ from the validated original-data/model replay')
    development=read_spatial_jsonl(config['validation_data'],labeled_only=True)
    fingerprint={k:v for k,v in config.items() if k!='device'}
    if any(value!=fingerprint for value in context.gather(fingerprint)):
        raise ValueError('RLCD ranks disagree about data, reference or configuration')
    model,reference=initialized_policy(config,context)
    optimizer=torch.optim.AdamW(optimizer_groups(model,config),weight_decay=config.get('weight_decay',.01))
    rates=[group['lr'] for group in optimizer.param_groups]
    progress={'epoch':0,'cursor':0,'step':0,'order':None,'seen':0,'optimizer_steps':0,
              'best_step':0,'best_accuracy':-1.,'best_iou':-1.}
    if resume:
        payload=load_checkpoint(resume,model,optimizer,rng,rank=context.rank,world_size=context.world_size)
        previous=normalize_spatial_config(payload['config'])
        allowed={'device','output','save_every','max_steps'}
        if {k:v for k,v in config.items() if k not in allowed}!={k:v for k,v in previous.items() if k not in allowed}:
            raise ValueError('RLCD resume changed its data, fixed reference or policy settings')
        training_state=payload.get('training_state',{})
        if training_state.get('reference_manifest_sha256')!=config['reference_manifest_sha256']:
            raise ValueError('RLCD resume lost its original fixed reference identity')
        progress=payload['progress']
        rates=[group['initial_lr'] for group in optimizer.param_groups]
    for group,rate in zip(optimizer.param_groups,rates):group['initial_lr']=rate
    broadcast_trainable(model,context)
    compiler=build_compiler(config,config['data_root'])
    validation_compiler=build_compiler(dict(config,data_root=config['validation_data_root']),'.')
    objective=SpatialRLCDObjective(config['rlcd'])
    iterations=objective.options['num_iterations']
    total_steps=math.ceil(len(records)/(context.world_size*config['gradient_accumulation']))*config['epochs']
    limit=min(config.get('max_steps',total_steps),total_steps)
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    gpu_names=context.gather(torch.cuda.get_device_name() if context.device.startswith('cuda') else None)
    if context.primary:
        (output/'run_manifest.json').write_text(json.dumps({'world_size':context.world_size,'config':config,
            'gpu_names':gpu_names,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_parameters':sum(p.numel() for p in model.parameters()),'fixed_reference':True,
            'starting_step':progress['step'],'group_steps':total_steps,'optimizer_updates_per_group':iterations,
            'versions':{name:importlib.metadata.version(name) for name in ['torch','transformers','peft']}},indent=2)+'\n')

    def save(best=False):
        states=context.gather(capture_rank_state(progress,rng))
        if context.primary:
            save_checkpoint(output/'latest',model,optimizer,config,progress,rng,rank_states=states,
                training_state={'reference_manifest_sha256':config['reference_manifest_sha256'],
                                'fixed_reference_recreated_from_original_export':True,
                                'initialization_variant':config.get('initialization_variant','full')})
            if best:
                target=output/'best';target.mkdir(exist_ok=True)
                link=target/'checkpoint.link';link.unlink(missing_ok=True)
                link.hardlink_to(output/'latest/checkpoint.pt');link.replace(target/'checkpoint.pt')
                shutil.copyfile(output/'latest/config.json',target/'config.json')
        context.barrier()

    model.train()
    if not resume:
        baseline=validate(model,validation_compiler,development,context)
        progress.update(best_accuracy=baseline['accuracy'],best_iou=baseline['mean_box_iou'])
        save(best=True)  # Retain the original merged policy if every RL checkpoint worsens development performance.
    while progress['epoch']<config['epochs'] and progress['step']<limit:
        if progress['order'] is None:
            progress['order']=epoch_shard(len(records),config['seed'],progress['epoch'],context.rank,context.world_size)
        chosen=progress['order'][progress['cursor']:progress['cursor']+config['gradient_accumulation']]
        count=int(context.sum([len(chosen)])[0])
        if not count:
            progress.update(epoch=progress['epoch']+1,cursor=0,order=None)
            continue
        rollouts=[]
        with torch.no_grad():
            for index in chosen:
                question=compiler.compile(records[index],labeled_only=True).questions[0]
                rollouts.append((question,objective.rollout(model(question),reference(question),question)))
        stats={};loss_sum=0.
        for iteration in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            for question,rollout in rollouts:
                loss,parts=objective.loss(model(question),question,rollout)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite spatial RLCD objective')
                (loss/count).backward();loss_sum+=float(loss.detach())
                for key,value in parts.items():stats[key]=stats.get(key,0.)+value
            synchronize_gradients(model,context,bucket_bytes=config['gradient_bucket_mb']*1024**2)
            audit=[]
            if config.get('audit_initial_update') and progress['step']<2:
                expected_groups=({'head','lora','merger'} if config['finetuning_type']=='lora'
                                 else {'head','language','merger','vision'})
                if {g['name'] for g in optimizer.param_groups}!=expected_groups:
                    raise ValueError('RLCD optimizer groups differ from the selected tuning method')
                for group in optimizer.param_groups:
                    for parameter in group['params']:
                        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad)):
                            index=int(parameter.grad.detach().abs().reshape(-1).argmax())
                            audit.append((group['name'],parameter,index,parameter.detach().reshape(-1)[index].clone()))
                            break
            norm=torch.nn.utils.clip_grad_norm_(list(model.parameters()),config.get('max_grad_norm',1.),error_if_nonfinite=True)
            update=progress['optimizer_steps']+1
            warmup=config.get('rl_warmup_updates',min(20,max(1,math.ceil(total_steps*iterations*.05))))
            factor=min(1.,update/warmup)*.5*(1+math.cos(math.pi*max(0,update-warmup)/max(1,total_steps*iterations-warmup)))
            for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate*max(factor,.05)
            optimizer.step();progress['optimizer_steps']=update
            if config.get('audit_initial_update') and progress['step']<2:
                verified=progress.setdefault('parameter_update_audit',{'gradients':{},'updates':{}})
                for name,parameter,index,before in audit:
                    verified['gradients'][name]=True
                    verified['updates'][name]=verified['updates'].get(name,False) or not torch.equal(before,parameter.detach().reshape(-1)[index])
        progress.update(cursor=progress['cursor']+len(chosen),step=progress['step']+1,seen=progress['seen']+count)
        finished=int(context.sum([int(progress['cursor']==len(progress['order']))])[0])==context.world_size
        if finished:progress.update(epoch=progress['epoch']+1,cursor=0,order=None)
        keys=sorted(stats)
        # Idle tail ranks still contribute the same metric vector to every collective.
        keys=context.gather(keys)
        keys=sorted(set().union(*map(set,keys)))
        totals=context.sum([loss_sum,*[stats.get(k,0.) for k in keys]])
        metrics=dict(zip(['loss',*keys],[v/(count*iterations) for v in totals]))
        metrics.update(step=progress['step'],seen=progress['seen'],epoch=progress['epoch'],total_records=len(records)*config['epochs'],
                       optimizer_steps=progress['optimizer_steps'],grad_norm=float(norm),
                       elapsed_seconds=time.monotonic()-started,time_utc=datetime.now(timezone.utc).isoformat(),
                       peak_cuda_bytes=torch.cuda.max_memory_allocated() if context.device.startswith('cuda') else 0)
        due=finished or progress['step']%config['save_every']==0 or progress['step']==limit
        improved=False
        if due:
            quality=validate(model,validation_compiler,development,context)
            metrics.update(validation_accuracy=quality['accuracy'],validation_mean_box_iou=quality['mean_box_iou'])
            improved=(quality['accuracy'],quality['mean_box_iou'])>(progress['best_accuracy'],progress['best_iou'])
            if improved:progress.update(best_step=progress['step'],best_accuracy=quality['accuracy'],best_iou=quality['mean_box_iou'])
            metrics['peak_cuda_bytes']=max(context.gather(metrics['peak_cuda_bytes']));metrics['peak_cuda_scope']='all_ranks'
        if context.primary:
            with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics)+'\n')
            print(json.dumps(metrics),flush=True)
            if due:(output/'gpu.json').write_text(json.dumps({'time_utc':metrics['time_utc'],'gpus':gpu_telemetry()}))
        if due:save(best=improved)
    if config.get('audit_initial_update'):
        audits=context.gather(progress.get('parameter_update_audit',{}))
        names=['head','lora','merger'] if config['finetuning_type']=='lora' else ['head','language','merger','vision']
        valid=all(all(a.get(kind,{}).get(name) is True for name in names)
                  for a in audits for kind in ['gradients','updates'])
        frozen=(frozen_backbone_matches_reference(model,reference) if config['finetuning_type']=='lora' else True)
        frozen=all(context.gather(frozen))
        if not valid or not frozen:raise ValueError('RLCD failed its trainable/frozen parameter update audit')
        if context.primary:
            filename='parameter-update-audit.json' if config['finetuning_type']=='lora' else 'full-update-audit.json'
            (output/filename).write_text(json.dumps({'world_size':context.world_size,
                'finetuning_type':config['finetuning_type'],'trainable_groups':names,
                'all_groups_have_gradients':True,'all_groups_updated':True,
                'frozen_backbone_unchanged':frozen,'ranks':audits},indent=2)+'\n')
    if context.primary:
        (output/'training_complete.json').write_text(json.dumps({'step':progress['step'],'epoch':progress['epoch'],
            'epochs_requested':config['epochs'],'best_step':progress['best_step'],
            'stop_reason':'epochs_complete' if progress['epoch']==config['epochs'] else 'planned_step_limit'},indent=2)+'\n')
    return model,compiler,progress
