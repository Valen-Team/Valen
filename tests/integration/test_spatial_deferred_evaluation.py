"""Release-oriented pipelines finish training/export without launching benchmarks."""
import json
import sys
from pathlib import Path

from scripts.train import run_spatial_pipeline as pipeline
from valen.training.checkpoint import file_sha256


def test_deferred_pipeline_exports_both_epochs_before_returning(tmp_path,monkeypatch):
    data,run,export=tmp_path/'data',tmp_path/'run',tmp_path/'export'
    train=data/'gui/processed/train.jsonl';train.parent.mkdir(parents=True);train.write_text('{}\n')
    dev=data/'eval/gui/processed/gui_dev.jsonl';dev.parent.mkdir(parents=True);dev.write_text('{}\n')
    seal=data/'gui/manifests/ready.json';seal.parent.mkdir(parents=True)
    seal.write_text(json.dumps(dict(validated=True,samples=1,
        records_sha256=file_sha256(train),development_sha256=file_sha256(dev))))
    calls=[]
    class Cluster:
        primary=True
        def __init__(self,*args):pass
        def decision(self,name,value):return value
        def execute(self,args,**kwargs):
            calls.append(args)
            if args[:2]==['-m','valen.train']:
                config=json.loads(Path(args[args.index('--config')+1]).read_text())
                path=Path(config['output']);path.mkdir(parents=True)
                (path/'training_complete.json').write_text(json.dumps(dict(stop_reason='epochs_complete')))
            elif args[:2]==['-m','scripts.export.qwen.export_spatial']:
                export.mkdir();(export/'export_manifest.json').write_text(json.dumps(dict(
                    reload_validation=dict(independent_process=True))))
            else:raise AssertionError('A benchmark ran on the training lease')
    monkeypatch.setattr(pipeline,'SpatialCluster',Cluster)
    monkeypatch.setattr('valen.data.spatial.read_spatial_jsonl',lambda *a,**kw:[{}])
    monkeypatch.setattr(sys,'argv',['pipeline','--task','gui','--model-path',str(tmp_path/'base'),
        '--data-root',str(data),'--output-dir',str(run),'--export-dir',str(export),
        '--training-samples','1','--defer-evaluation'])
    pipeline.main()
    assert len(calls)==3 and sum(c[1]=='valen.train' for c in calls)==2
    marker=json.loads((run/'training_export_complete.json').read_text())
    assert marker['evaluation_deferred'] and marker['benchmarks']==[]
    assert marker['model_export_manifest_sha256']==file_sha256(export/'export_manifest.json')
    assert not (run/'evaluation/metrics.json').exists()
