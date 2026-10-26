"""Run full standard and additional GUI benchmarks on a separate evaluation lease."""
import argparse
import json
from pathlib import Path

from scripts.data.prepare_spatial import digest_file,write_json
from valen.training.spatial_cluster import SpatialCluster


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','additional-root','model-path','output-dir','launch-id'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--nodes',type=int,default=2)
    parser.add_argument('--gpus-per-node',type=int,default=4)
    args=parser.parse_args()
    root,output,package=Path(args.data_root),Path(args.output_dir),Path(args.model_path)
    cluster=SpatialCluster(output/'coordination'/args.launch_id,args.nodes,args.gpus_per_node)
    cluster.execute(['-m','evaluation.gui.evaluate','--data-root',str(root),
        '--model-path',str(package),'--output-dir',str(output/'evaluation')],distributed=True)
    classic=json.loads((output/'evaluation/metrics.json').read_text())
    expected=dict(screenspot_test=1272,screenspot_v2_test=1272,screenspot_pro_test=1581)
    if set(classic['benchmarks'])!=set(expected) or any(classic['benchmarks'][k]['samples']!=v for k,v in expected.items()):
        raise ValueError('Standard GUI benchmark coverage is incomplete')
    common=['--model-path',str(package),'--nodes',str(args.nodes),'--gpus-per-node',str(args.gpus_per_node)]
    cluster.execute(['-m','evaluation.gui.additional','--data-root',args.additional_root,
        '--output-dir',str(output/'evaluation-additional'),'--launch-id',args.launch_id+'-additional',*common],all_nodes=True)
    cluster.execute(['-m','evaluation.examples','--task','gui','--data-root',str(root),
        '--output-dir',str(output/'examples'),'--launch-id',args.launch_id+'-examples',*common],all_nodes=True)
    extra=json.loads((output/'evaluation-additional/metrics.json').read_text())
    identity=digest_file(package/'export_manifest.json')
    if not extra['complete_official_splits'] or extra['model_export_manifest_sha256']!=identity:
        raise ValueError('Additional GUI benchmark identity or coverage is incomplete')
    if cluster.primary:
        marker=json.loads((output/'training_export_complete.json').read_text())
        if marker['model_export_manifest_sha256']!=identity:
            raise ValueError('The evaluated package differs from the completed training export')
        marker.update(format='valen_spatial_gui_evaluated_v1',evaluation_deferred=False,
            benchmarks=sorted(expected),additional_benchmarks=sorted(extra['benchmarks']))
        write_json(output/'pipeline_complete.json',marker)
        write_json(output/'evaluation-suite/pipeline_complete.json',dict(model=package.name,
            model_export_manifest_sha256=identity,standard_samples=4125,additional_samples=9637))


if __name__=='__main__':
    main()
