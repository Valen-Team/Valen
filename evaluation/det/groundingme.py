"""GroundingME structured-head evaluation with a fixed coordinate convention."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch

from evaluation.spatial import load_spatial_model
from valen.data.spatial import read_spatial_jsonl
from valen.modeling.qwen.spatial import decode_spatial
from valen.training.checkpoint import file_sha256
from valen.training.distributed import close, initialize
from valen.training.spatial_cluster import SpatialCluster


DATASET_REVISION = "78e3c7974b2b1db0ea52266969e40f664a38e330"
SCORER_REVISION = "6867f7a045596d29cd77a14a01d22d4a68d64fb6"
SOURCE = "https://huggingface.co/datasets/lirang04/GroundingME/tree/" + DATASET_REVISION
SCORER = "https://github.com/lirang04/GroundingME/blob/" + SCORER_REVISION + "/evaluate.py"
BENCHMARK = "groundingme_test"
PROTOCOL = "groundingme_top1_fixed_normalized_xyxy_v1"
SOURCE_SHA256 = (
    "4fed5cc62dd46bfc927ba4ec76bdc949fd38cbf8a06a509641fe8504915d42cb",
    "f161147c172cdbd5af154b2e4a5d1e192cc20b9cc9d7db7d1132ba8366c19c32",
    "d905ef749fca4caac2fa102f4334f73ec521ad8c8c6d42d82179f666b66b22fb",
    "403a19480d70174905b5ef87db0e5ac7096e63ad04f2d5abf25e0cb3fb525a43",
)
# Official evaluate.py prompt. Valen returns a structured box instead of generated JSON.
PROMPT = (
    "All spatial relationships are defined from the viewer's perspective, where 'front' means closer "
    "to the viewer and 'back' means farther from the viewer. Please provide the bounding box coordinate "
    "of the object the following statement describes:\n{description}\n"
    "Ensure that all details mentioned about the object are accurate. Provide at most one bounding box. "
    "If a matching object is found, provide its bounding box as a JSON in the format "
    '{{"bbox_2d": [x1, y1, x2, y2]}}. If no matching object is found, output {{"bbox_2d": null}}.'
)


def score_box(prediction, target):
    """Continuous xyxy IoU; an explicit empty prediction matches an empty target."""
    if not prediction or not target:
        return float(not prediction and not target)
    if len(prediction) != 4 or len(target) != 4:
        raise ValueError("GroundingME requires one xyxy box or an empty prediction")
    if any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in prediction + target):
        raise ValueError("GroundingME expects normalized coordinates without format guessing")
    if prediction[0] >= prediction[2] or prediction[1] >= prediction[3]:
        return 0.0
    overlap = max(0, min(prediction[2], target[2]) - max(prediction[0], target[0])) * max(
        0, min(prediction[3], target[3]) - max(prediction[1], target[1]))
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area(prediction) + area(target) - overlap)


def aggregate(cases):
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Empty or duplicate GroundingME predictions")
    def metrics(values):
        return {"samples": len(values), "accuracy": sum(c["iou"] >= .5 for c in values) / len(values),
            "accuracy_at_075": sum(c["iou"] >= .75 for c in values) / len(values),
            "accuracy_at_09": sum(c["iou"] >= .9 for c in values) / len(values),
            "mean_box_iou": sum(c["iou"] for c in values) / len(values)}
    groups = defaultdict(list)
    for case in cases:
        groups[case["subtask_l1"]].append(case)
        if case["subtask_l2"]:
            groups[case["subtask_l1"] + "/" + case["subtask_l2"]].append(case)
    return {**metrics(cases), "groups": {key: metrics(values) for key, values in sorted(groups.items())}}


@torch.inference_mode()
def evaluate(args):
    context = initialize(args.device)
    try:
        root, package, output = map(Path, (args.data_root, args.model_path, args.output_dir))
        path = root / "processed" / (BENCHMARK + ".jsonl")
        manifest = json.loads((root / "manifests" / (BENCHMARK + ".json")).read_text())
        preflight = json.loads((root / "manifests/preflight.json").read_text())
        if (not preflight.get("validated") or preflight.get("baseline_training_sha256_overlap") != 0
                or file_sha256(path) != manifest["records_sha256"] or manifest["samples"] != 1005
                or manifest.get("source_revision") != DATASET_REVISION
                or manifest.get("scorer_revision") != SCORER_REVISION):
            raise ValueError("Prepare and audit the complete pinned GroundingME test split")
        records = read_spatial_jsonl(path, labeled_only=True)
        if len(records) != 1005:
            raise ValueError("GroundingME must include every test example")
        model, compiler = load_spatial_model(package, context.device, data_root=root)
        if str(context.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        cases, timings = [], []
        started = time.monotonic()
        for record in records[context.rank::context.world_size]:
            before = time.monotonic()
            # Labels and task dimensions are used only after inference, never as input metadata.
            request = {key: record[key] for key in ("id", "group_id", "request")}
            request["meta"] = {"referring": True, "allow_empty_target": True}
            question = compiler.compile(request).questions[0]
            prediction = decode_spatial(question, model(question))
            bbox = prediction["boxes"][0]["bbox"] if prediction["boxes"] else []
            targets = record["targets"]["target"]["boxes"]
            if len(targets) > 1:
                raise ValueError("GroundingME expects at most one reference box")
            target = targets[0] if targets else []
            cases.append({"id": record["id"], "subtask_l1": record["meta"]["subtask_l1"],
                "subtask_l2": record["meta"]["subtask_l2"], "prediction": bbox,
                "iou": score_box(bbox, target)})
            if str(context.device).startswith("cuda"):
                torch.cuda.synchronize()
            timings.append(time.monotonic() - before)
        cases = sorted((c for shard in context.gather(cases) for c in shard), key=lambda c: c["id"])
        times = sorted(t for shard in context.gather(timings) for t in shard)
        peaks = context.gather(torch.cuda.max_memory_allocated() if str(context.device).startswith("cuda") else 0)
        result = aggregate(cases)
        if result["samples"] != 1005:
            raise ValueError("Incomplete distributed GroundingME predictions")
        result.update(data_sha256=manifest["records_sha256"], protocol=PROTOCOL,
            source_revision=DATASET_REVISION, scorer_revision=SCORER_REVISION,
            elapsed_seconds=time.monotonic() - started, world_size=context.world_size,
            latency_seconds={"mean": sum(times) / len(times), "p50": times[len(times) // 2],
                             "p95": times[int(len(times) * .95)]}, peak_cuda_bytes_per_rank=max(peaks))
        if context.primary:
            output.mkdir(parents=True, exist_ok=True)
            source = Path(__file__).resolve().parents[2] / "source_manifest.json"
            payload = {"task": "det", "model": package.name, "benchmarks": {BENCHMARK: result},
                "model_export_manifest_sha256": file_sha256(package / "export_manifest.json"),
                "evaluation_code_sha256": json.loads(source.read_text())["sha256"] if source.is_file() else file_sha256(Path(__file__)),
                "inference": {"max_pixels": compiler.max_pixels, "max_length": compiler.max_length,
                    "coordinates": "normalized_xyxy", "selection": "highest-confidence foreground query",
                    "rejection": "background-winner rule for all samples; no benchmark threshold fitting"}}
            SpatialCluster.write(output / "predictions.json", {"data_sha256": manifest["records_sha256"], "cases": cases})
            SpatialCluster.write(output / "metrics.json", payload)
            SpatialCluster.write(output / "pipeline_complete.json", {"model": package.name, "samples": 1005,
                "protocol": PROTOCOL, "data_sha256": manifest["records_sha256"]})
            print(json.dumps(payload), flush=True)
        context.barrier()
    finally:
        close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--launch-id", help="Coordinate evaluation across an existing allocation")
    args = parser.parse_args()
    if args.launch_id:
        cluster = SpatialCluster(Path(args.output_dir) / "coordination" / args.launch_id,
                                 args.nodes, args.gpus_per_node)
        cluster.execute(["-m", "evaluation.det.groundingme", "--data-root", args.data_root,
            "--model-path", args.model_path, "--output-dir", args.output_dir, "--device", args.device], distributed=True)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
