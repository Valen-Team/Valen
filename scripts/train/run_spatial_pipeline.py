"""Run head warmup, joint SFT, portable export and the required spatial suite."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import string
import subprocess
import sys

from scripts.train.spatial_cluster import SpatialCluster


def required_benchmarks(task, coco_protocol="standard"):
    ref = {f"{name}_{split}" for name in ("refcoco", "refcocoplus") for split in ("val", "testA", "testB")}
    ref.update({"refcocog_val", "refcocog_test"})
    coco = "coco2017_conditional_val2017" if coco_protocol == "conditional" else "coco2017_val2017"
    return {"det": ref if coco_protocol == "none" else ref | {coco},
            "seg": ref | {"phrasecut_val_miniv", "phrasecut_test"},
            "gui": {"screenspot_test", "screenspot_v2_test", "screenspot_pro_test"}}[task]


def lora_resource_config(config, world_size):
    """Record an explicit topology change and scale schedules by examples."""
    config = dict(config)
    previous_world = config["required_world_size"]
    previous_batch = previous_world * config["gradient_accumulation"]
    batch = max(previous_batch, world_size * 2)
    if batch % world_size:
        raise ValueError("The global batch must divide across all training ranks")
    scale = batch / previous_batch
    previous_steps = math.ceil(config.get("training_samples", 100000) / previous_batch) * config["epochs"]
    warmup = config.get("lr_warmup_steps", min(100, previous_steps // 10))
    config.update(required_world_size=world_size, gradient_accumulation=batch // world_size,
                  save_every=max(1, round(config["save_every"] / scale)),
                  lr_warmup_steps=max(1, round(warmup / scale)),
                  gradient_bucket_mb=128 if world_size == 64 else config.get("gradient_bucket_mb", 16),
                  training_resource_override={"original_world_size": previous_world,
                      "original_global_batch": previous_batch, "world_size": world_size,
                      "global_batch": batch, "learning_rates_changed": False})
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("det", "seg", "gui"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--launch-id", default="local")
    parser.add_argument("--finetuning-type", choices=("lora", "full"), default="lora")
    parser.add_argument("--baseline-run", help="Completed LoRA run whose warmup initialization is reused")
    parser.add_argument("--retain-warmup", action="store_true", help="Retain minimal initialization for a full comparison")
    parser.add_argument("--with-reasonseg", action="store_true", help="Run the independent ReasonSeg val/test supplement")
    parser.add_argument("--defer-evaluation", action="store_true",
                        help="Finish training/export only; schedule benchmarks separately after releasing GPUs")
    parser.add_argument("--training-world-size", type=int, choices=(8, 16, 64),
                        help="Explicit training topology; scale validation and warmup by examples")
    parser.add_argument("--training-samples", type=int, default=100000,
                        help="Expected sealed training count; GUI revisions may use an expanded corpus")
    args = parser.parse_args()
    if (args.finetuning_type == "full") != bool(args.baseline_run):
        parser.error("Full comparisons require --baseline-run; standard pipelines use LoRA")
    if args.training_samples <= 0 or (args.task != "gui" and args.training_samples != 100000):
        parser.error("Only GUI revisions may override the positive 100k training count")
    root = Path(__file__).resolve().parents[2]
    output, data_root = Path(args.output_dir).resolve(), Path(args.data_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cluster = SpatialCluster(output / "coordination" / args.launch_id, args.nodes, args.gpus_per_node)
    template = json.loads((root / "configs/train/qwen" / args.task / "sft.json").read_text())
    environment = dict(os.environ, VALEN_MODEL_PATH=str(Path(args.model_path).resolve()),
                       VALEN_DATA_ROOT=str(data_root), VALEN_RUN_DIR=str(output))
    config = {key: string.Template(value).substitute(environment) if isinstance(value, str) else value
              for key, value in template.items()}
    config["training_samples"] = args.training_samples
    if (root / "source_manifest.json").is_file():
        config["code_sha256"] = json.loads((root / "source_manifest.json").read_text())["sha256"]
    if args.baseline_run:
        from scripts.train.spatial_comparison import full_config
        config = full_config(args.baseline_run, output, config.get("code_sha256"), args.training_world_size)
        if config["task"] != args.task or Path(config["model_path"]).resolve() != Path(args.model_path).resolve():
            raise ValueError("The full-training reference must use the same task and backbone")
    elif args.training_world_size is not None:
        config = lora_resource_config(config, args.training_world_size)
    if config["required_world_size"] != args.nodes * args.gpus_per_node:
        raise ValueError("Training world size differs from the allocated GPU topology")
    if not Path(config["validation_data"]).is_file():
        raise ValueError("Prepare the disjoint development records before training")
    from valen.data.spatial import read_spatial_jsonl
    from valen.training.checkpoint import file_sha256
    seal_path = Path(config.get("data_seal", data_root / args.task / "manifests/ready.json"))
    seal = json.loads(seal_path.read_text())
    config["data_seal_sha256"] = file_sha256(seal_path)
    if seal.get("validated") is not True or file_sha256(config["data"]) != seal["records_sha256"]:
        raise ValueError("Training input is not the audited immutable dataset")
    if file_sha256(config["validation_data"]) != seal["development_sha256"]:
        raise ValueError("Development input changed after the data audit")
    if seal.get("samples") != args.training_samples or len(read_spatial_jsonl(config["data"], labeled_only=True)) != args.training_samples:
        raise ValueError("Training count differs from the explicitly requested sealed corpus")
    coco_protocol = config.get("coco_protocol", "standard")
    evaluation_options = ["--coco-protocol", coco_protocol] if args.task == "det" else []
    for stage in (("joint",) if args.baseline_run else ("warmup", "joint")):
        stage_config = dict(config, stage=stage, output=str(output / stage),
                            gradient_checkpointing=stage == "joint")
        # Both stages see the same complete corpus; the warmup is also the frozen-backbone baseline.
        stage_config.pop("max_steps", None)
        if stage == "joint" and args.finetuning_type == "lora":
            stage_config.update(head_lr=1e-4, lora_lr=5e-5, merger_lr=1e-5, lora_rank=32, lora_alpha=64)
        path = output / (stage + ".json")
        if cluster.primary and not path.exists():
            path.write_text(json.dumps(stage_config, indent=2) + "\n")
        completion = output / stage / "training_complete.json"
        if cluster.decision(stage, not completion.exists()):
            command = ["-m", "valen.train", "--config", str(path)]
            latest = output / stage / "latest"
            if (latest / "checkpoint.pt").is_file():
                command.extend(["--resume", str(latest)])
            elif stage == "joint":
                initialization = (Path(args.baseline_run) / "comparison-init" if args.baseline_run
                                  else output / "warmup/best")
                command.extend(["--initialize", str(initialization)])
            cluster.execute(command, distributed=True)
        if json.loads(completion.read_text())["stop_reason"] != "epochs_complete":
            raise ValueError("A planned full-corpus stage has not finished its epoch")
        if stage == "warmup":
            if args.retain_warmup:
                cluster.execute(["-m", "scripts.train.spatial_comparison", "--retain-warmup", str(output)])
            if args.defer_evaluation:
                continue
            benchmark = {"det": ("refcoco_val" if coco_protocol == "none" else
                                 "coco2017_conditional_val2017" if coco_protocol == "conditional" else "coco2017_val2017"),
                         "seg": "refcoco_val", "gui": "screenspot_v2_test"}[args.task]
            baseline_output = output / "baseline-evaluation"
            if cluster.decision("baseline", not (baseline_output / "metrics.json").exists()):
                cluster.execute(["-m", f"evaluation.{args.task}.evaluate", "--data-root",
                         str(data_root / "eval" / args.task), "--model-path", str(output / "warmup/best"),
                         "--benchmark", benchmark, "--output-dir", str(baseline_output), *evaluation_options], distributed=True)
    export_dir = Path(args.export_dir).resolve()
    if cluster.decision("export", not (export_dir / "export_manifest.json").is_file()):
        cluster.execute(["-m", "scripts.export.qwen.export_spatial", "--checkpoint",
                 str(output / "joint/best"), "--output-dir", str(export_dir), "--data",
                 config["validation_data"], "--data-root", config["validation_data_root"]])
    if args.defer_evaluation:
        if cluster.primary:
            export = json.loads((export_dir / "export_manifest.json").read_text())
            if export.get("reload_validation", {}).get("independent_process") is not True:
                raise ValueError("Release requires independent export reload validation")
            marker = dict(format="valen_spatial_training_export_v1", task=args.task,
                model=export_dir.name, evaluation_deferred=True, benchmarks=[],
                training_epochs={"warmup": 1, "joint": 1}, finetuning_type=args.finetuning_type,
                model_export_manifest_sha256=file_sha256(export_dir / "export_manifest.json"))
            (output / "training_export_complete.json").write_text(json.dumps(marker, indent=2) + "\n")
            (output / "pipeline_complete.json").write_text(json.dumps(marker, indent=2) + "\n")
        return
    evaluation_output = output / "evaluation"
    cluster.execute(["-m", f"evaluation.{args.task}.evaluate", "--data-root",
             str(data_root / "eval" / args.task), "--model-path", str(export_dir),
             "--output-dir", str(evaluation_output), *evaluation_options], distributed=True)
    report = json.loads((evaluation_output / "metrics.json").read_text())
    if set(report["benchmarks"]) != required_benchmarks(args.task, coco_protocol):
        raise ValueError("Required benchmark results are incomplete")
    if args.with_reasonseg:
        if args.task not in {"det", "seg"}:
            raise ValueError("ReasonSeg supports det and seg")
        cluster.execute(["-m", "evaluation.reasonseg", "--task", args.task,
            "--data-root", str(data_root / "eval" / args.task / "reasonseg"),
            "--model-path", str(export_dir), "--output-dir", str(output / "evaluation-reasonseg"),
            "--nodes", str(args.nodes), "--gpus-per-node", str(args.gpus_per_node),
            "--launch-id", args.launch_id + "-reasonseg"], all_nodes=True)
    if cluster.primary:
        if args.baseline_run:
            from scripts.train.spatial_comparison import compare
            compare(args.baseline_run, output, export_dir)
        (output / "pipeline_complete.json").write_text(json.dumps({"task": args.task, "model": export_dir.name,
        "benchmarks": sorted(report["benchmarks"]), "baseline": "frozen backbone, same full training corpus",
        "training_epochs": {"warmup": 1, "joint": 1}, "finetuning_type": args.finetuning_type,
        "warmup_reused": bool(args.baseline_run)}, indent=2) + "\n")
        for stage in ("warmup", "joint"):
            for checkpoint in ("latest", "best"):
                shutil.rmtree(output / stage / checkpoint, ignore_errors=True)
        if args.baseline_run:
            for name in ("checkpoint.pt", "config.json"):
                (Path(args.baseline_run) / "comparison-init" / name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
