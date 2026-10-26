"""RLCD resume retains the optimizer, fixed reference identity and rollout RNG."""
import json
from types import SimpleNamespace

import torch

from valen.modeling.qwen.spatial import SpatialOutput
from valen.training import spatial_rlcd
from valen.training.checkpoint import file_sha256


class TinySpatialModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head=torch.nn.Linear(1,18)
        self.base_manifest={'revision':'test-base','weight_sha256':{'base':'test'}}

    def forward(self,question):
        value=self.head(torch.ones(1,1))
        logits=value[:,2:].reshape(1,4,4)
        coordinates=(logits.softmax(-1)*torch.linspace(0,1,4)).sum(-1)
        center,size=coordinates[:,:2],coordinates[:,2:]
        return SpatialOutput(value[:,:2],logits,torch.cat((center-size/2,center+size/2),-1).clamp(0,1))


def test_spatial_rlcd_resume_matches_uninterrupted_group_updates(tmp_path,monkeypatch):
    records=[{'id':str(i),'group_id':'image:'+str(i),'request':{
        'state':{'messages':[{'content':[{'type':'image_url','image_url':{'url':'test.png'}}]}]},
        'questions':{'q':{'type':'det','instructions':'Find the target.'}}},
        'targets':{'q':{'boxes':[[.1,.1,.8,.8]],'labels':[0]}}} for i in range(4)]
    data=tmp_path/'train.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in records))
    package=tmp_path/'full';package.mkdir();(package/'export_manifest.json').write_text('{}')
    manifest=tmp_path/'hard-cases.json'
    manifest.write_text(json.dumps({'validated':True,'selected':4,'records_sha256':file_sha256(data),
        'model_export_manifest_sha256':file_sha256(package/'export_manifest.json'),
        'original_training_sha256':'sealed-original'}))
    compiler=SimpleNamespace(compile=lambda record,**kwargs:SimpleNamespace(questions=[SimpleNamespace(
        kind='det',categories=['target'],metadata={},target=record['targets']['q'])]))
    monkeypatch.setattr(spatial_rlcd,'build_model',lambda config:TinySpatialModel())
    monkeypatch.setattr(spatial_rlcd,'build_compiler',lambda *args,**kwargs:compiler)
    monkeypatch.setattr(spatial_rlcd,'initialize_full_export',lambda *args:None)
    monkeypatch.setattr(spatial_rlcd,'optimizer_groups',lambda model,config:[
        {'params':list(model.parameters()),'lr':.01,'name':'head'}])
    monkeypatch.setattr(spatial_rlcd,'validate',lambda model,*args:{'accuracy':.5,'mean_box_iou':.5})
    config=dict(architecture='qwen_spatial',task='det',model_path='test-base',stage='joint',method='rlcd',
        finetuning_type='full',reference_export=str(package),hard_case_manifest=str(manifest),
        original_training_sha256='sealed-original',data=str(data),validation_data=str(data),
        validation_data_root=str(tmp_path),data_root=str(tmp_path),device='cpu',required_world_size=1,
        gradient_accumulation=1,epochs=1,save_every=2,seed=42,rlcd={'group_size':8})
    full,_,expected=spatial_rlcd.run(dict(config,output=str(tmp_path/'uninterrupted')))
    partial=dict(config,output=str(tmp_path/'resumed'),max_steps=2)
    _,_,middle=spatial_rlcd.run(partial)
    assert middle['step']==2 and middle['optimizer_steps']==4
    resumed,_,observed=spatial_rlcd.run(dict(config,output=str(tmp_path/'resumed')),resume=tmp_path/'resumed/latest')
    assert observed==expected and observed['seen']==4 and observed['optimizer_steps']==8
    for name,value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],rtol=0,atol=0)
    config['original_training_sha256']='changed'
    try:
        spatial_rlcd.run(dict(config,output=str(tmp_path/'bad')),resume=tmp_path/'resumed/latest')
    except ValueError as error:
        assert 'replay' in str(error)
    else:
        raise AssertionError('Changed reference/data identity was accepted')
