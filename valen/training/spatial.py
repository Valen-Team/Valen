"""Spatial SFT using Valen's distributed synchronization and checkpoint contract."""
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import torch

from valen.configuration import flatten_config
from valen.data.spatial import read_spatial_jsonl
from valen.modeling.factory import build_model, build_compiler, get_backend, normalize_model_config, optimizer_groups
from valen.modeling.qwen.spatial import spatial_signature
from .checkpoint import capture_rank_state, file_sha256, load_checkpoint, save_checkpoint
from .distributed import initialize, epoch_shard, broadcast_trainable, synchronize_gradients
from .spatial_loss import SpatialObjective


def normalize_spatial_config(config):
    config = flatten_config(config)
    config = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in config.items()}
    if any(isinstance(v, str) and "${" in v for v in config.values()):
        raise ValueError("Unresolved environment variable in spatial configuration")
    config = normalize_model_config(config)
    if config.get("method", "sft") not in {"sft", "rlcd"}:
        raise ValueError("Spatial prediction supports sft or rlcd")
    if config.get("method") == "rlcd":
        from .rlcd import validate_options
        if (config.get("task") != "det" or config.get("finetuning_type") not in {"full", "lora"}
                or not config.get("reference_export")):
            raise ValueError("Spatial RLCD requires Det training and an immutable reference export")
        config["rlcd"] = validate_options(config.get("rlcd", {}))
    for name, default in (("gradient_accumulation", 4), ("epochs", 1), ("save_every", 100),
                           ("required_world_size", 8), ("max_length", 4096), ("gradient_bucket_mb", 16)):
        value = config.setdefault(name, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    config.setdefault("method", "sft")
    config.setdefault("stage", "joint")
    config.setdefault("seed", 42)
    return config


def gpu_telemetry():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip().splitlines() if result.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        return []


@torch.no_grad()
def validate(model, compiler, records, objective, context):
    model.eval()
    loss = count = 0
    for record in records[context.rank::context.world_size]:
        question = compiler.compile(record, labeled_only=True).questions[0]
        value, _ = objective(model(question), question)
        loss += float(value)
        count += 1
    loss, count = context.sum([loss, count])
    model.train()
    if not count:
        raise ValueError("Empty spatial development set")
    return loss / count


def run(config, resume=None, initialize_from=None):
    config = normalize_spatial_config(config)
    if config["method"] == "rlcd":
        from .spatial_rlcd import run as run_rlcd
        return run_rlcd(config, resume, initialize_from)
    if resume and initialize_from:
        raise ValueError("Use resume or initialize, not both")
    context = initialize(config.get("device", "cuda"))
    if context.world_size != config["required_world_size"]:
        raise ValueError(f"Expected {config['required_world_size']} training ranks, got {context.world_size}")
    config = dict(config, device=context.device)
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed + context.rank)
    data_path = Path(config["data"])
    config["data_sha256"] = file_sha256(data_path)
    records = read_spatial_jsonl(data_path, labeled_only=True)
    development = read_spatial_jsonl(config["validation_data"], labeled_only=True) if config.get("validation_data") else []
    if development:
        config["validation_sha256"] = file_sha256(config["validation_data"])
    fingerprints = context.gather({k: v for k, v in config.items() if k != "device"})
    if any(item != fingerprints[0] for item in fingerprints):
        raise ValueError("Training configuration or data differs between ranks")
    if initialize_from and Path(config["output"]).resolve() == Path(initialize_from).resolve().parent:
        raise ValueError("A new stage requires a distinct output directory")
    model = build_model(config)
    groups = optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.get("weight_decay", .01))
    initial_rates = [group["lr"] for group in groups]
    progress = {"epoch": 0, "cursor": 0, "step": 0, "order": None,
                "best_loss": float("inf"), "best_step": None, "seen": 0}
    if initialize_from:
        payload = load_checkpoint(initialize_from, model, strict=False)
        get_backend(config["architecture"]).validate_initialization(payload["config"], config)
    if resume:
        payload = load_checkpoint(resume, model, optimizer, rng, rank=context.rank, world_size=context.world_size)
        previous = normalize_spatial_config(payload["config"])
        allowed = {"output", "device", "save_every", "max_steps", "epochs"}
        if {k: v for k, v in config.items() if k not in allowed} != {k: v for k, v in previous.items() if k not in allowed}:
            raise ValueError("Resume configuration/data changed; initialize a new stage instead")
        progress = payload["progress"]
        initial_rates = [group.get("initial_lr", lr) for group, lr in zip(groups, initial_rates)]
    for group, lr in zip(optimizer.param_groups, initial_rates):
        group["initial_lr"] = lr
    broadcast_trainable(model, context)
    compiler = build_compiler(config, config.get("data_root", data_path.parent))
    validation_compiler = build_compiler(dict(config, data_root=config.get("validation_data_root",
        str(Path(config["validation_data"]).parent.parent))), ".") if development else None
    objective = SpatialObjective(config)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    max_steps = config.get("max_steps", math.ceil(len(records) / (context.world_size * config["gradient_accumulation"])) * config["epochs"])
    started = time.monotonic()
    last_telemetry = -math.inf
    gpu_names = context.gather(torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    if context.primary:
        manifest = {"world_size": context.world_size, "config": config,
                    "head": spatial_signature(config), "gpu_names": [name for name in gpu_names if name],
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "total_parameters": sum(p.numel() for p in model.parameters()),
                    "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}}
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def save(name):
        states = context.gather(capture_rank_state(progress, rng))
        if context.primary:
            if name == "best":
                destination = output / "best"
                destination.mkdir(exist_ok=True)
                temporary = destination / "checkpoint.link"
                temporary.unlink(missing_ok=True)
                os.link(output / "latest/checkpoint.pt", temporary)
                temporary.replace(destination / "checkpoint.pt")
                shutil.copyfile(output / "latest/config.json", destination / "config.json")
            else:
                save_checkpoint(output / name, model, optimizer, config, progress, rng, rank_states=states)
        context.barrier()

    model.train()
    while progress["epoch"] < config["epochs"] and progress["step"] < max_steps:
        if progress["order"] is None:
            progress["order"] = epoch_shard(len(records), seed, progress["epoch"], context.rank, context.world_size)
        selected = progress["order"][progress["cursor"]:progress["cursor"] + config["gradient_accumulation"]]
        global_count = int(context.sum([len(selected)])[0])
        if not global_count:
            progress.update(epoch=progress["epoch"] + 1, cursor=0, order=None)
            continue
        optimizer.zero_grad(set_to_none=True)
        names = ("bce", "dice", "classification") if config["task"] == "seg" else ("classification", "coordinate", "l1", "giou")
        stats = dict.fromkeys(names, 0.0)
        total_loss = 0.0
        for index in selected:
            question = compiler.compile(records[index], labeled_only=True).questions[0]
            if question.kind != config["task"]:
                raise ValueError("Training record belongs to a different spatial task")
            prediction = model(question)
            loss, parts = objective(prediction, question)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite spatial loss")
            (loss / global_count).backward()
            total_loss += float(loss.detach())
            for name, value in parts.items():
                stats[name] = stats.get(name, 0) + value
        synchronize_gradients(model, context, bucket_bytes=config["gradient_bucket_mb"] * 1024**2)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                             config.get("max_grad_norm", 1.0), error_if_nonfinite=True)
        step = progress["step"] + 1
        warmup = config.get("lr_warmup_steps", min(100, max_steps // 10))
        ratio = min(1.0, step / max(1, warmup)) * .5 * (1 + math.cos(math.pi * max(0, step - warmup) / max(1, max_steps - warmup)))
        for group, lr in zip(optimizer.param_groups, initial_rates):
            group["lr"] = lr * max(ratio, .05)
        optimizer.step()
        progress.update(cursor=progress["cursor"] + len(selected), step=step,
                        seen=progress.get("seen", 0) + global_count)
        values = context.sum([total_loss, *[stats[key] for key in sorted(stats)]])
        metrics = dict(zip(["loss", *sorted(stats)], [value / global_count for value in values]))
        finished = int(progress["cursor"] == len(progress["order"]))
        all_finished = int(context.sum([finished])[0]) == context.world_size
        if all_finished:
            progress.update(epoch=progress["epoch"] + 1, cursor=0, order=None)
        metrics.update(step=step, epoch=progress["epoch"], seen=progress["seen"],
                       total_records=len(records) * config["epochs"], grad_norm=float(norm),
                       elapsed_seconds=time.monotonic() - started,
                       time_utc=datetime.now(timezone.utc).isoformat(),
                       peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
        checkpoint_due = step % config["save_every"] == 0 or all_finished or step == max_steps
        if checkpoint_due:
            metrics["peak_cuda_bytes"] = max(context.gather(metrics["peak_cuda_bytes"]))
            metrics["peak_cuda_scope"] = "all_ranks"
        if checkpoint_due and development:
            metrics["validation_loss"] = validate(model, validation_compiler, development, objective, context)
        quality = metrics.get("validation_loss", metrics["loss"])
        improved = checkpoint_due and quality < progress["best_loss"]
        if improved:
            progress.update(best_loss=quality, best_step=step)
        if context.primary:
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if time.monotonic() - last_telemetry >= 900:
                (output / "gpu.json").write_text(json.dumps({"time_utc": metrics["time_utc"], "gpus": gpu_telemetry()}))
                last_telemetry = time.monotonic()
        if checkpoint_due:
            save("latest")
            if improved:
                save("best")
    if progress["step"] == 0:
        raise ValueError("No spatial training step completed")
    save("latest")
    if progress["best_step"] == progress["step"]:
        save("best")
    if context.primary:
        complete = {"step": progress["step"], "epoch": progress["epoch"],
                    "epochs_requested": config["epochs"], "best_step": progress["best_step"],
                    "stop_reason": "epochs_complete" if progress["epoch"] == config["epochs"] else "planned_step_limit"}
        (output / "training_complete.json").write_text(json.dumps(complete, indent=2) + "\n")
    return model, compiler, progress
