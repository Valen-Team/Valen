"""Shared, streaming evaluation for spatial decision benchmarks."""
import argparse
import json
from pathlib import Path
import time

import torch

from valen.data.spatial import SpatialCompiler, decode_segmentation_target, read_spatial_jsonl
from valen.modeling.factory import build_model, build_compiler
from valen.modeling.qwen.spatial import decode_spatial
from valen.training.checkpoint import load_checkpoint, file_sha256
from valen.training.distributed import initialize, close
from valen.training.spatial_loss import box_iou


def load_spatial_model(path, device, base_model_path=None, data_root="."):
    from transformers import AutoModel, AutoProcessor
    root = Path(path)
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") == "valen_spatial":
        from filelock import FileLock
        from transformers.dynamic_module_utils import HF_MODULES_CACHE
        cache = Path(HF_MODULES_CACHE)
        cache.mkdir(parents=True, exist_ok=True)
        # Dynamic Python-module copies share a cache across ranks and must finish before import.
        with FileLock(str(cache / ".valen-spatial.lock")):
            model = AutoModel.from_pretrained(root, trust_remote_code=True, local_files_only=True,
                                              dtype="auto").to(device).eval()
            compiler = SpatialCompiler(AutoProcessor.from_pretrained(root, trust_remote_code=True,
                                       local_files_only=True), data_root,
                                       config.get("max_length", 4096), config.get("max_pixels", 262144))
    else:
        config.update(device=device, gradient_checkpointing=False, data_root=str(data_root))
        if base_model_path:
            config["model_path"] = str(base_model_path)
        model = build_model(config)
        load_checkpoint(root, model)
        compiler = build_compiler(config, data_root)
    model.requires_grad_(False).eval()
    return model, compiler


@torch.inference_mode()
def evaluate(model, compiler, records, task, context, coco_annotations=None):
    intersection = union = iou_sum = correct = box_correct = samples = 0
    predictions = []
    groups = {}
    latency = []
    if str(context.device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for record in records[context.rank::context.world_size]:
        sample_started = time.monotonic()
        question = compiler.compile(record).questions[0]
        if question.kind != task:
            raise ValueError("Benchmark record belongs to a different spatial task")
        prediction = decode_spatial(question, model(question), score_threshold=0.0 if coco_annotations else .05)
        target = question.target
        if target is None:
            raise ValueError("Evaluation data requires labels")
        if task == "seg":
            truth, valid = decode_segmentation_target(target, question.media_root)
            truth, valid = truth.bool().cpu(), valid.cpu()
            mask = prediction["mask"].bool().cpu()
            if mask.shape != truth.shape:
                raise ValueError("Prediction and original mask sizes differ")
            overlap = int((truth & mask & valid).sum())
            combined = int(((truth | mask) & valid).sum())
            epsilon = 1e-5 if question.metadata.get("source") == "reasonseg" else 0
            iou = overlap / (combined + epsilon) if combined else 1.0
            intersection += overlap
            union += combined
            iou_sum += iou
            correct += int(iou > .5)
        elif coco_annotations:
            from evaluation.det.coco import predictions_to_coco
            predictions.extend(predictions_to_coco(question, prediction))
        elif task == "gui":
            point = prediction.get("point")
            success = bool(point and any(b[0] <= point[0] <= b[2] and b[1] <= point[1] <= b[3]
                                         for b in target["boxes"]))
            correct += int(success)
            if prediction["boxes"] and target["boxes"]:
                iou = float(box_iou(torch.tensor([prediction["boxes"][0]["bbox"]]),
                                    torch.tensor(target["boxes"])).max())
                iou_sum += iou
                box_correct += int(iou >= .5)
        else:
            if not target["boxes"] and question.metadata.get("allow_empty_target", False):
                correct += int(not prediction["boxes"])
                iou_sum += int(not prediction["boxes"])
            elif len(target["boxes"]) != 1:
                raise ValueError("Referring localization requires one reference box")
            elif prediction["boxes"]:
                iou = float(box_iou(torch.tensor([prediction["boxes"][0]["bbox"]]),
                                    torch.tensor(target["boxes"])).max())
                iou_sum += iou
                correct += int(iou >= .5)
        samples += 1
        if str(context.device).startswith("cuda"):
            torch.cuda.synchronize()
        latency.append(time.monotonic() - sample_started)
        if task == "gui":
            for key in ("platform", "ui_type", "application", "group"):
                value = str(question.metadata.get(key, "unknown"))
                group = (key, value)
                bucket = groups.setdefault(group, [0, 0])
                bucket[0] += int(success)
                bucket[1] += 1
    totals = context.sum([samples, correct, iou_sum, intersection, union, box_correct])
    samples, correct, iou_sum, intersection, union, box_correct = totals
    timings = sorted(value for shard in context.gather(latency) for value in shard)
    peaks = context.gather(torch.cuda.max_memory_allocated() if str(context.device).startswith("cuda") else 0)
    result = {"samples": int(samples), "elapsed_seconds": time.monotonic() - started,
              "world_size": context.world_size, "batch_size_per_rank": 1,
              "latency_seconds": {"mean": sum(timings) / len(timings), "p50": timings[len(timings) // 2],
                                  "p95": timings[min(len(timings) - 1, int(len(timings) * .95))]},
              "peak_cuda_bytes_per_rank": max(peaks)}
    if coco_annotations:
        shards = context.gather(predictions)
        if context.primary:
            from evaluation.det.coco import score_coco
            detections = [item for shard in shards for item in shard]
            result.update(score_coco(coco_annotations, detections))
    elif task == "seg":
        result.update(mIoU=iou_sum / samples, cIoU=intersection / union if union else 1.0,
                      precision_at_05=correct / samples)
    else:
        result.update(accuracy=correct / samples, mean_box_iou=iou_sum / samples)
    if task == "gui":
        result["bbox_accuracy_at_05"] = box_correct / samples
        merged = {}
        for shard in context.gather(groups):
            for group, (hits, count) in shard.items():
                bucket = merged.setdefault("/".join(group), [0, 0])
                bucket[0] += hits
                bucket[1] += count
        result["groups"] = {key: {"accuracy": hits / count, "samples": count}
                            for key, (hits, count) in sorted(merged.items())}
    context.barrier()
    return result


def benchmark_paths(root, task, benchmark=None, coco_protocol="none"):
    root = Path(root)
    paths = [root / "processed" / (benchmark + ".jsonl")] if benchmark else sorted(
        path for path in (root / "processed").glob("*.jsonl") if path.stem not in {"dev", "gui_dev"})
    if task != "det":
        return paths
    if coco_protocol == "none":
        if benchmark and benchmark.startswith("coco2017_"):
            raise ValueError("COCO is excluded from the referring suite; select an explicit legacy COCO protocol")
        names = {f"{name}_{split}" for name in ("refcoco", "refcocoplus") for split in ("val", "testA", "testB")}
        names.update({"refcocog_val", "refcocog_test"})
        names.update({"reasonseg_val", "reasonseg_test"})
        return [path for path in paths if path.stem in names]
    from evaluation.det.coco import CONDITIONAL_BENCHMARK as BENCHMARK
    paths = [(root / "protocols/coco_prompt/processed" / (BENCHMARK + ".jsonl"))
             if path.stem == BENCHMARK or (path.stem == "coco2017_val2017" and coco_protocol == "conditional")
             else path for path in paths]
    conditional = root / "protocols/coco_prompt/processed" / (BENCHMARK + ".jsonl")
    if not benchmark and coco_protocol == "conditional" and conditional.is_file() and conditional not in paths:
        paths.append(conditional)
    return paths


def main(task):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", help="Prepared benchmark name, for example refcoco_testA; omit for all")
    parser.add_argument("--base-model-path", help="Override the frozen base path of a training checkpoint")
    parser.add_argument("--device", default="cuda")
    if task == "det":
        parser.add_argument("--coco-protocol", choices=("none", "conditional", "standard"), default="none",
                            help="Referring benchmarks only (default); explicit COCO protocols reproduce legacy results")
    args = parser.parse_args()
    context = initialize(args.device)
    try:
        root = Path(args.data_root)
        paths = benchmark_paths(root, task, args.benchmark, getattr(args, "coco_protocol", "none"))
        if not paths:
            raise ValueError("No prepared benchmark files found")
        model, compiler = load_spatial_model(args.model_path, context.device, args.base_model_path, root)
        results = {}
        for path in paths:
            records = read_spatial_jsonl(path, labeled_only=True)
            conditional = path.stem == "coco2017_conditional_val2017"
            annotations = root / "annotations/instances_val2017.json" if path.stem == "coco2017_val2017" or conditional else None
            if conditional:
                from evaluation.det.coco import SINGLE_CATEGORY_PROTOCOL
                if any(len(r["request"]["questions"]["target"]["categories"]) != 1
                       or r["meta"].get("protocol") != SINGLE_CATEGORY_PROTOCOL for r in records):
                    raise ValueError("Conditional COCO requires exactly one ground-truth present category")
                manifest = json.loads((root / "protocols/coco_prompt/manifests" / (path.stem + ".json")).read_text())
                if file_sha256(path) != manifest["records_sha256"] or file_sha256(annotations) != manifest["annotations_sha256"]:
                    raise ValueError("Conditional COCO records or annotations changed after preparation")
            results[path.stem] = evaluate(model, compiler, records, task, context, annotations)
            if conditional:
                results[path.stem].update(protocol="known_category_conditional_detection", category_source="ground_truth",
                    max_categories_per_prompt=1, queried_images=manifest["images"], catalog_images=manifest["catalog_images"],
                    images_without_positive_queries=manifest["images_without_positive_queries"])
            if task == "seg" and all(record.get("meta", {}).get("source") == "reasonseg" for record in records):
                results[path.stem]["gIoU"] = results[path.stem]["mIoU"]
            results[path.stem]["data_sha256"] = file_sha256(path)
            if context.primary:
                output = Path(args.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                payload = {"task": task, "model": Path(args.model_path).name, "benchmarks": results}
                source_manifest = Path(__file__).resolve().parents[1] / "source_manifest.json"
                payload["evaluation_code_sha256"] = (json.loads(source_manifest.read_text())["sha256"]
                    if source_manifest.is_file() else file_sha256(Path(__file__)))
                payload["inference"] = {"parameters": sum(p.numel() for p in model.parameters()),
                    "max_pixels": compiler.max_pixels, "max_length": compiler.max_length,
                    "dtype": str(next(model.backbone.parameters()).dtype),
                    "backbone_autocast_dtype": str(getattr(model, "backbone_autocast_dtype", None)),
                    "gpu": torch.cuda.get_device_name() if str(context.device).startswith("cuda") else None,
                    "coordinate_format": "normalized_xyxy", "mask_threshold": 0.5,
                    "coco_score_threshold": 0.0, "referring_selection": "highest_confidence"}
                (output / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
                print(json.dumps({"benchmark": path.stem, **results[path.stem]}), flush=True)
    finally:
        close()
