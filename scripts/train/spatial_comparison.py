"""Preserve a shared warmup initialization and compare LoRA with full spatial SFT."""
import argparse
import json
import math
from pathlib import Path


PROTOCOL = ("task", "model_path", "data_sha256", "validation_sha256", "data_seal_sha256",
            "seed", "epochs", "gradient_accumulation", "required_world_size", "dtype",
            "attn_implementation", "max_length", "max_pixels", "spatial_width", "spatial_layers",
            "spatial_heads", "spatial_queries", "coordinate_bins", "coco_protocol")
METRICS = ("accuracy", "mean_box_iou", "bbox_accuracy_at_05", "mIoU", "cIoU", "precision_at_05",
           "AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L")


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def retain_warmup(baseline):
    """Keep head tensors and identity only; optimizer/RNG are unnecessary for a new stage."""
    baseline = Path(baseline)
    target = baseline / "comparison-init"
    if (target / "initialization.json").is_file():
        return True
    completion = baseline / "warmup/training_complete.json"
    if not completion.is_file():
        return False
    import torch
    from valen.training.checkpoint import file_sha256
    source = baseline / "warmup/best/checkpoint.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    completed = json.loads(completion.read_text())
    if (payload["config"]["stage"] != "warmup" or not payload["weights"]
            or any(not key.startswith("head.") for key in payload["weights"])
            or completed["stop_reason"] != "epochs_complete"
            or payload["progress"]["step"] != completed["best_step"]):
        raise ValueError("Comparison initialization must be the completed warmup's best task head")
    target.mkdir(exist_ok=True)
    kept = {key: payload[key] for key in ("weights", "config", "base_manifest")}
    temporary = target / "checkpoint.tmp"
    torch.save(kept, temporary)
    temporary.replace(target / "checkpoint.pt")
    write_json(target / "config.json", payload["config"])
    write_json(target / "initialization.json", {
        "format": "valen_spatial_comparison_init_v1", "warmup_best_step": completed["best_step"],
        "source_checkpoint_sha256": file_sha256(source),
        "checkpoint_sha256": file_sha256(target / "checkpoint.pt"),
        "parameters": sum(value.numel() for value in kept["weights"].values()),
        "optimizer_retained": False})
    return True


def full_config(baseline, output, code_sha256, training_world_size=None):
    baseline = Path(baseline)
    if not (baseline / "pipeline_complete.json").is_file():
        raise ValueError("Complete the LoRA reference pipeline before its full-training comparison")
    config = json.loads((baseline / "joint/run_manifest.json").read_text())["config"]
    if config.get("finetuning_type") != "lora" or config["stage"] != "joint":
        raise ValueError("A full-training comparison requires a LoRA joint-stage reference")
    initialization = baseline / "comparison-init"
    if not (initialization / "initialization.json").is_file():
        raise FileNotFoundError("Retain the warmup best initialization before the reference cleans its checkpoints")
    warmup = json.loads((initialization / "config.json").read_text())
    if any(config.get(key) != warmup.get(key) for key in PROTOCOL):
        raise ValueError("Warmup and reference data, backbone or training protocol differ")
    from valen.training.checkpoint import file_sha256
    identity = json.loads((initialization / "initialization.json").read_text())
    if file_sha256(initialization / "checkpoint.pt") != identity["checkpoint_sha256"]:
        raise ValueError("Retained warmup initialization changed")
    config = {key: value for key, value in config.items() if not key.startswith("lora_")}
    config.update(output=str(Path(output) / "joint"), finetuning_type="full", gradient_checkpointing=True,
                  head_lr=1e-4, language_lr=1e-5, merger_lr=1e-5, vision_lr=2e-6,
                  code_sha256=code_sha256, initialization_sha256=identity["checkpoint_sha256"],
                  reference_code_sha256=warmup.get("code_sha256"))
    if training_world_size is not None:
        if type(training_world_size) is not int or training_world_size not in (8, 16, 64):
            raise ValueError("Explicit Full training world size must be 8, 16 or 64")
        previous = config["required_world_size"]
        config["required_world_size"] = training_world_size
        if previous != training_world_size:
            reference_batch = previous * config["gradient_accumulation"]
            training_batch = training_world_size * config["gradient_accumulation"]
            reference_steps = math.ceil(100000 / reference_batch) * config["epochs"]
            reference_warmup = config.get("lr_warmup_steps", min(100, reference_steps // 10))
            reference_save = config.get("save_every", 100)
            config.update(lr_warmup_steps=max(1, math.ceil(reference_warmup * reference_batch / training_batch)),
                          save_every=max(1, math.ceil(reference_save * reference_batch / training_batch)),
                          gradient_bucket_mb=128 if training_world_size == 64 else 16)
            config["comparison_resource_override"] = {
                "reference_world_size": previous, "training_world_size": training_world_size,
                "reference_global_batch": reference_batch, "training_global_batch": training_batch,
                "reference_lr_warmup_steps": reference_warmup, "training_lr_warmup_steps": config["lr_warmup_steps"],
                "reference_save_every": reference_save, "training_save_every": config["save_every"],
                "gradient_bucket_mb": config["gradient_bucket_mb"], "learning_rate_scaling": "unchanged",
                "scope": "explicit_resource_change; data_and_initialization_matched; not_a_matched_batch_ablation"}
    return config


def compare(baseline, full, export_dir):
    """Write compact per-benchmark deltas plus measured training/inference costs."""
    baseline, full, export_dir = Path(baseline), Path(full), Path(export_dir)
    runs = {"lora": baseline, "full": full}
    manifests = {name: json.loads((run / "joint/run_manifest.json").read_text()) for name, run in runs.items()}
    configs = {name: manifest["config"] for name, manifest in manifests.items()}
    override = configs["full"].get("comparison_resource_override")
    allowed = set()
    if override:
        reference, candidate = configs["lora"], configs["full"]
        valid = (override.get("reference_world_size") == reference["required_world_size"]
                 and override.get("training_world_size") == candidate["required_world_size"]
                 and override.get("reference_global_batch") == reference["required_world_size"] * reference["gradient_accumulation"]
                 and override.get("training_global_batch") == candidate["required_world_size"] * candidate["gradient_accumulation"]
                 and reference["gradient_accumulation"] == candidate["gradient_accumulation"])
        if not valid:
            raise ValueError("Full resource override does not match the recorded training protocols")
        allowed.add("required_world_size")
    if any(configs["lora"].get(key) != configs["full"].get(key) for key in PROTOCOL if key not in allowed):
        raise ValueError("Comparison data or training protocol differs")
    evaluations = {name: json.loads((run / "evaluation/metrics.json").read_text()) for name, run in runs.items()}
    if set(evaluations["lora"]["benchmarks"]) != set(evaluations["full"]["benchmarks"]):
        raise ValueError("Comparison benchmark sets differ")
    result = {"format": "valen_spatial_lora_full_comparison_v1", "task": configs["full"]["task"],
              "shared_warmup_initialization_sha256": configs["full"]["initialization_sha256"],
              "protocol": {key: value for key, value in configs["full"].items() if key in PROTOCOL
                           and key != "model_path"}, "methods": {}, "benchmarks": {}}
    if override:
        result["resource_override"] = override
        result["matched_training_batch"] = False
    from scripts.train.monitor_spatial import last_metric
    from scripts.train.summarize_spatial import completed_costs
    for name, run in runs.items():
        manifest = manifests[name]
        metric_path = run / "joint/metrics.jsonl"
        latest = last_metric(metric_path)
        costs = completed_costs(str(metric_path), metric_path.stat().st_mtime_ns)
        config = configs[name]
        package = (export_dir.parent / json.loads((run / "pipeline_complete.json").read_text())["model"]
                   if name == "lora" else export_dir)
        result["methods"][name] = {"trainable_parameters": manifest["trainable_parameters"],
            "total_parameters_during_training": manifest["total_parameters"],
            "training_elapsed_seconds": costs["recorded_training_seconds"],
            "training_time_scope": "sum_of_recorded_process_segments",
            "training_peak_cuda_bytes": costs["peak_cuda_bytes_all_ranks"] or latest.get("peak_cuda_bytes", 0),
            "world_size": manifest["world_size"], "gpu_names": manifest["gpu_names"],
            "learning_rates": {key: config[key] for key in ("head_lr", "lora_lr", "language_lr", "merger_lr", "vision_lr")
                               if key in config},
            "code_sha256": config.get("code_sha256"),
            "model_weight_bytes": sum(path.stat().st_size for path in package.glob("*.safetensors")),
            "inference": evaluations[name].get("inference")}
    for benchmark, reference in evaluations["lora"]["benchmarks"].items():
        candidate = evaluations["full"]["benchmarks"][benchmark]
        if reference["data_sha256"] != candidate["data_sha256"] or reference["samples"] != candidate["samples"]:
            raise ValueError("Comparison evaluation inputs differ")
        values = {name: {key: metric[key] for key in METRICS if key in metric}
                  for name, metric in (("lora", reference), ("full", candidate))}
        result["benchmarks"][benchmark] = {"samples": reference["samples"], **values,
            "full_minus_lora": {key: values["full"][key] - value for key, value in values["lora"].items()},
            "latency_seconds": {"lora": reference.get("latency_seconds"), "full": candidate.get("latency_seconds")}}
    write_json(full / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retain-warmup", required=True)
    args = parser.parse_args()
    print(json.dumps({"retained": retain_warmup(args.retain_warmup)}), flush=True)


if __name__ == "__main__":
    main()
