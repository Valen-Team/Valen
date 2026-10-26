"""Evaluate both official ReasonSeg splits on the task's reserved GPU group."""
import argparse
import json
from pathlib import Path

from valen.training.spatial_cluster import SpatialCluster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("det", "seg"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--launch-id", required=True)
    args = parser.parse_args()
    root, output = Path(args.data_root), Path(args.output_dir)
    expected = {"reasonseg_val", "reasonseg_test"}
    preflight = json.loads((root / "manifests/preflight.json").read_text())
    if not preflight.get("validated") or not all((root / "processed" / (name + ".jsonl")).is_file() for name in expected):
        raise ValueError("Prepare and audit the complete original ReasonSeg validation/test splits")
    if any(preflight.get("baseline_training_sha256_overlap", {}).values()):
        raise ValueError("ReasonSeg evaluation overlaps the baseline training corpus; report contamination before scoring")
    cluster = SpatialCluster(output / "coordination" / args.launch_id, args.nodes, args.gpus_per_node)
    cluster.execute(["-m", f"evaluation.{args.task}.evaluate", "--data-root", str(root),
                     "--model-path", args.model_path, "--output-dir", str(output)], distributed=True)
    report = json.loads((output / "metrics.json").read_text())
    if set(report["benchmarks"]) != expected:
        raise ValueError("ReasonSeg validation/test results are incomplete")
    if cluster.primary:
        (output / "pipeline_complete.json").write_text(json.dumps({"task": args.task,
            "model": Path(args.model_path).name, "benchmarks": sorted(expected),
            "protocol": "first original instruction; native ignore mask; mask-derived bbox for det"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
