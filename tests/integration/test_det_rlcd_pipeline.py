"""RLCD reports matched test data and creates no package for a zero-error replay."""
import json
from pathlib import Path

import pytest

from scripts.train import run_det_rlcd_pipeline as pipeline
from scripts.train.run_spatial_pipeline import required_benchmarks
from valen.training.checkpoint import file_sha256


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_three_way_report_rejects_changed_eval_and_keeps_training_costs(tmp_path):
    names=required_benchmarks('det','none')|{'reasonseg_val','reasonseg_test','groundingme_test'}
    runs=[tmp_path/n for n in ['lora','full','rlcd']]
    models=[tmp_path/n for n in ['FullModel','RLModel']]
    for model in models:put(model/'export_manifest.json',{})
    for index,run in enumerate(runs):
        put(run/'joint/run_manifest.json',{'config':{'required_world_size':16 if index==0 else 64,
            'gradient_accumulation':2,'data_sha256':'hard-cases' if index==2 else 'original'}})
        (run/'joint/metrics.jsonl').write_text(json.dumps({'elapsed_seconds':10,'peak_cuda_bytes':100,
                                                        'peak_cuda_scope':'all_ranks'})+'\n')
        for folder in ['evaluation','evaluation-reasonseg','evaluation-groundingme']:
            selected=[n for n in names if (n.startswith('reasonseg') if folder=='evaluation-reasonseg' else
                       n=='groundingme_test' if folder=='evaluation-groundingme' else n.startswith('refcoco'))]
            put(run/folder/'metrics.json',{'benchmarks':{n:{'samples':2,'data_sha256':n,
                'accuracy':.1*(index+1),'mean_box_iou':.2*(index+1)} for n in selected}})
    put(runs[2]/'replay/pipeline_complete.json',{'errors':100,'selected':100})
    put(runs[2]/'post-replay/pipeline_complete.json',{'errors':80,'selected_errors':50})
    result=pipeline.compare_models(*runs,*models)
    assert len(result['benchmarks'])==11
    assert result['training']['full']['global_batch']==128
    assert result['training_errors']['full']['selected_error_rate']==1.
    item=result['benchmarks']['refcoco_val']
    assert item['rlcd_minus_full']['accuracy']==pytest.approx(.1)
    path=runs[2]/'evaluation/metrics.json';bad=json.loads(path.read_text())
    bad['benchmarks']['refcoco_val']['data_sha256']='changed';put(path,bad)
    with pytest.raises(ValueError,match='data/counts'):
        pipeline.compare_models(*runs,*models)


def test_zero_error_replay_skips_rl_without_creating_a_model_package(tmp_path,monkeypatch):
    full=tmp_path/'Valen-Decision-Det-2B-Full'
    put(full/'export_manifest.json',{'source_checkpoint_sha256':'full-source','training_data_sha256':'original'})
    source=tmp_path/'full-run'
    put(source/'joint/run_manifest.json',{'config':{'model_path':str(tmp_path/'base'),'data_sha256':'original'}})
    gate=tmp_path/'accepted.json'
    put(gate,{'models':['Valen-Decision-Det-2B','Valen-Decision-Det-2B-Full'],
              'export_identities':{'Valen-Decision-Det-2B-Full':{'source_checkpoint_sha256':'full-source'}}})
    output=tmp_path/'rl-run';put(output/'replay/pipeline_complete.json',{})
    subset=tmp_path/'subset'
    put(subset/'manifest.json',{'validated':True,'original_records':100000,'original_training_sha256':'original',
         'model_export_manifest_sha256':file_sha256(full/'export_manifest.json'),'errors':0,'selected':0})
    target=tmp_path/'Valen-Decision-Det-2B-Full-RLCD'
    class Cluster:
        primary=True
        def __init__(self,*args):pass
        def decision(self,name,value):return value
    monkeypatch.setattr(pipeline,'SpatialCluster',Cluster)
    argv=['rlcd']
    for key,value in {'model-path':tmp_path/'base','full-baseline-run':source,'lora-baseline-run':tmp_path/'lora',
        'full-model':full,'data-root':tmp_path/'data','output-dir':output,'subset-dir':subset,'export-dir':target,
        'acceptance-marker':gate,'launch-id':'test'}.items():argv.extend(['--'+key,str(value)])
    monkeypatch.setattr('sys.argv',argv)
    pipeline.main()
    assert json.loads((output/'pipeline_complete.json').read_text())['zero_error_skip'] is True
    assert not target.exists()


@pytest.mark.parametrize('case',['valid','full_mined_errors','unmerged_adapter'])
def test_merged_lora_gate_requires_its_own_error_replay(tmp_path,monkeypatch,case):
    import struct
    package=tmp_path/'Valen-Decision-Det-2B'
    put(package/'export_manifest.json',{'source_checkpoint_sha256':'lora-source','training_data_sha256':'original',
        'finetuning_type':'lora','reload_validation':{'independent_process':True}})
    key='backbone.lora_A.weight' if case=='unmerged_adapter' else 'backbone.weight'
    header=json.dumps({key:{'dtype':'BF16','shape':[1],'data_offsets':[0,2]}}).encode()
    (package/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'\0\0')
    source=tmp_path/'lora-run'
    put(source/'joint/run_manifest.json',{'config':{'model_path':str(tmp_path/'base'),'data_sha256':'original'}})
    gate=tmp_path/'accepted.json'
    put(gate,{'models':[package.name],'export_identities':{package.name:{'source_checkpoint_sha256':'lora-source'}}})
    output=tmp_path/'rl-run';put(output/'replay/pipeline_complete.json',{})
    subset=tmp_path/'subset'
    put(subset/'manifest.json',{'validated':True,'original_records':100000,'original_training_sha256':'original',
        'model_export_manifest_sha256':'full-export' if case=='full_mined_errors' else file_sha256(package/'export_manifest.json'),
        'errors':0,'selected':0})
    target=tmp_path/'Valen-Decision-Det-2B-RLCD'
    class Cluster:
        primary=True
        def __init__(self,*args):pass
        def decision(self,name,value):return value
    monkeypatch.setattr(pipeline,'SpatialCluster',Cluster)
    argv=['rlcd']
    for key,value in {'model-path':tmp_path/'base','lora-baseline-run':source,'reference-model':package,
        'reference-variant':'lora_merged','data-root':tmp_path/'data','output-dir':output,'subset-dir':subset,
        'export-dir':target,'acceptance-marker':gate,'launch-id':'test'}.items():argv.extend(['--'+key,str(value)])
    monkeypatch.setattr('sys.argv',argv)
    if case=='valid':
        pipeline.main()
        assert json.loads((output/'pipeline_complete.json').read_text())['zero_error_skip'] is True
        assert not target.exists()
    else:
        with pytest.raises(ValueError,match='different reference' if case=='full_mined_errors' else 'standalone merged'):
            pipeline.main()
