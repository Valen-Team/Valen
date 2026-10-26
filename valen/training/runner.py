"""Shared SFT/RLCD loop: token budgets, distributed updates and exact resume."""
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path
import torch
from valen.training.checkpoint import capture_rank_state, file_sha256, load_checkpoint, save_checkpoint
from valen.training.distributed import initialize as initialize_distributed, close, epoch_shard, broadcast_trainable, synchronize_gradients
from .sft import SFTObjective
from .batching import training_batches
from .rlcd import RLCDObjective, validate_options
from valen.modeling.factory import build_model, build_compiler, get_backend, normalize_model_config, optimizer_groups
from valen.data.schema import read_jsonl
from valen.data.prefetch import PackPrefetcher
from valen.modeling.qwen.attention import ATTENTION_IMPLEMENTATIONS, with_attention_implementation


def normalize_config(config):
    """Normalize architecture, objective settings and disabled legacy fields."""
    config = normalize_model_config(config)
    if config.get("kd_weight", 0) != 0 or config.get("teacher_checkpoint"):
        raise ValueError("Teacher distillation was removed; remove teacher_checkpoint and nonzero kd_weight")
    for key in ("kd_weight", "kd_temperature", "teacher_checkpoint"):
        config.pop(key, None)
    method = config.setdefault("method", "sft")
    for name in ("rps_weight", "brier_weight"):
        value = config.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if method not in {"sft", "rlcd"}:
        raise ValueError(f"Unknown training method: {method}")
    if method == "rlcd":
        config["rlcd"] = validate_options(config.get("rlcd", {}))
        if config.get("rps_weight", 0):
            raise ValueError("RLCD uses Brier calibration; rps_weight must be zero")
    elif config.get("rlcd"):
        raise ValueError("Set method=rlcd to use RLCD options")
    if config.setdefault("loss_reduction", "state_mean") not in ("state_mean", "question_mean"):
        raise ValueError("loss_reduction must be state_mean or question_mean")
    for name, default in (("tokens_per_step", 16384), ("microbatch_size", 1), ("microbatch_max_tokens", 32768)):
        value = config.setdefault(name, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    shared_rl = (method == "rlcd" and config.get("qwen_execution") == "shared_state"
                 and not config.get("cache_frozen_features", False))
    if config["microbatch_size"] > 1 and (config["architecture"] != "qwen" or not (method == "sft" or shared_rl)):
        raise ValueError("Parallel microbatches require Qwen SFT or shared_state RLCD without feature caching")
    if type(config.setdefault("data_prefetch", False)) is not bool:
        raise ValueError("data_prefetch must be a boolean")
    if config["data_prefetch"] and (config["architecture"] != "qwen" or not (method == "sft" or shared_rl)):
        raise ValueError("Data prefetch requires Qwen SFT or shared_state RLCD")
    return config


def run(config, resume=None, initialize=None):
    if normalize_model_config(config)["architecture"] == "qwen_spatial":
        from .spatial import run as run_spatial
        return run_spatial(config, resume, initialize)
    config = normalize_config(config)
    if resume and initialize:
        raise ValueError("Use either resume or initialize")
    if config["method"] == "rlcd" and not (resume or initialize):
        raise ValueError("RLCD requires --initialize from an SFT checkpoint or --resume")
    if initialize and Path(config["output"]).resolve() == Path(initialize).resolve().parent:
        raise ValueError("Initialization requires a new output directory")
    context = initialize_distributed(config.get("device", "cuda"))
    config = dict(config, device=context.device)
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed + context.rank)
    data_path = Path(config["data"])
    backend = get_backend(config["architecture"])
    records = read_jsonl(data_path, candidate_fn=backend.candidates)
    config = dict(config, data_sha256=file_sha256(data_path))
    fingerprints = context.gather({k: v for k, v in config.items() if k != "device"})
    if any(value != fingerprints[0] for value in fingerprints):
        raise ValueError("All ranks must use identical config and dataset contents")
    model = build_model(config)
    groups = optimizer_groups(model, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.get("weight_decay", 0.01))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    progress = {"epoch": 0, "cursor": 0, "step": 0, "order": None}
    payload = None
    if initialize:
        payload = load_checkpoint(initialize, model, strict=False)
        previous = payload["config"]
        previous = normalize_model_config(previous)
        backend.validate_initialization(previous, config)
    if resume:
        payload = load_checkpoint(resume, model, optimizer, rng, rank=context.rank, world_size=context.world_size)
        allowed_changes = {"output", "epochs", "max_steps", "device", "save_every",
                           "microbatch_size", "microbatch_max_tokens", "data_prefetch"}
        if config["architecture"] == "qwen":
            allowed_changes.update({"attn_implementation", "gradient_checkpointing"})
        previous = normalize_config(payload["config"])
        if {k: v for k, v in previous.items() if k not in allowed_changes} != {k: v for k, v in config.items() if k not in allowed_changes}:
            raise ValueError("Resume config/data changed; use --initialize for a new stage")
        progress = payload["progress"]
    broadcast_trainable(model, context)
    if config["method"] == "rlcd":
        if not resume:
            torch.manual_seed(seed + context.rank)
        objective = RLCDObjective(model, config["rlcd"],
                                  payload.get("training_state") if resume else None, resuming=bool(resume),
                                  cache_frozen_features=config.get("cache_frozen_features", False),
                                  batch_config=config)
    else:
        objective = SFTObjective(config)
    progress.setdefault("optimizer_steps", progress["step"])
    compiler = build_compiler(config, data_path.parent)
    report = {"config": config, "world_size": context.world_size, "gradient_sync": "sum_after_local_accumulation",
              "initialization_checkpoint": str(initialize) if initialize else None,
              "initialization_sha256": file_sha256(Path(initialize) / "checkpoint.pt") if initialize else None,
              "tokens_per_step_scope": "per_rank", "adaptation": backend.adaptation(model, config),
              "trainable_parameters": {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
              "optimizer_groups": [{"name": g["name"], "lr": g["lr"], "parameters": sum(p.numel() for p in g["params"])} for g in groups],
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "av")}}
    if context.primary:
        if resume:
            previous_manifest = Path(resume).parent / "run_manifest.json"
            if previous_manifest.exists():
                origin = json.loads(previous_manifest.read_text(encoding="utf-8"))
                for key in ("initialization_checkpoint", "initialization_sha256"):
                    report[key] = origin.get(key)
            report["resume_checkpoint"] = str(resume)
            report["resume_step"] = progress["step"]
            report["resume_runtime_changes"] = {
                key: {"previous": previous.get(key), "current": config.get(key)}
                for key in ("attn_implementation", "gradient_checkpointing", "microbatch_size", "microbatch_max_tokens", "data_prefetch")
                if previous.get(key) != config.get(key)}
        (output / "run_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"world_size": context.world_size, "optimizer_groups": report["optimizer_groups"]}), flush=True)
    max_steps = config.get("max_steps", 2**63-1)
    token_budget = config.get("tokens_per_step", 16384)
    if token_budget <= 0:
        raise ValueError("tokens_per_step must be positive")
    def save():
        rank_states = context.gather(capture_rank_state(progress, rng))
        if context.primary:
            save_checkpoint(output / "latest", model, optimizer, config, progress, rng,
                            rank_states=rank_states, training_state=objective.state_dict())
        context.barrier()

    model.train()
    if config["method"] == "rlcd":
        for module in model.modules():
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                module.eval()
    started = time.monotonic()
    prior_elapsed = 0.0
    if resume and (output / "metrics.jsonl").exists():
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["step"] == progress["step"]:
                prior_elapsed = row.get("elapsed_seconds", 0.0)
    session_states = session_questions = 0
    metric_file = (output / "metrics.jsonl").open("a" if resume else "w", encoding="utf-8") if context.primary else nullcontext(None)
    media_name = "media.jsonl" if context.world_size == 1 else f"media.rank{context.rank}.jsonl"
    with metric_file as log, (output / media_name).open("a" if resume else "w", encoding="utf-8") as media_log, PackPrefetcher(records, compiler, backend, config) as pack_source:
        while progress["epoch"] < config.get("epochs", 1) and progress["step"] < max_steps:
            if progress["order"] is None:
                if context.world_size == 1:
                    progress["order"] = list(range(len(records)))
                    rng.shuffle(progress["order"])
                else:
                    progress["order"] = epoch_shard(len(records), seed, progress["epoch"], context.rank, context.world_size)
            prepare_started = time.monotonic()
            prepared = pack_source.take(progress["order"], progress["cursor"], rng.getstate(),
                                        prefetch_next=progress["step"] + 1 < max_steps)
            prepare_wait_seconds = time.monotonic() - prepare_started
            pack, tokens = prepared.states, prepared.tokens
            progress["cursor"] = prepared.cursor
            rng.setstate(prepared.rng_state)
            for event in prepared.media:
                media_log.write(json.dumps({"epoch": progress["epoch"], "step": progress["step"] + 1,
                                            "rank": context.rank, **event}, default=str) + "\n")
            local_counts = [len(pack), sum(len(s.questions) for s in pack), tokens,
                            int(progress["cursor"] == len(progress["order"]))]
            counts = context.sum(local_counts)
            global_states, global_questions, global_tokens, finished_ranks = map(int, counts)
            if global_states:
                normalization = global_questions if config["loss_reduction"] == "question_mean" else global_states
                rollouts = objective.prepare(model, pack)
                accumulated = {name: 0.0 for name in ("loss", *objective.metric_names)}
                for _ in range(objective.num_iterations):
                    optimizer.zero_grad(set_to_none=True)
                    stats = dict.fromkeys(accumulated, 0.0)
                    for unit in training_batches(model, backend, pack, rollouts, config):
                        losses = [(objective.loss_from_logits(d.logits, d.question, d.rollout), weight / normalization)
                                  for d, weight in unit]
                        combined = torch.stack([loss * scale for (loss, _), scale in losses]).sum()
                        if not torch.isfinite(combined):
                            raise FloatingPointError("Non-finite training unit loss")
                        combined.backward()
                        for (loss, terms), scale in losses:
                            stats["loss"] += loss.item() * scale
                            for name, value in terms.items():
                                stats[name] += value.item() * scale
                    synchronize_gradients(model, context)
                    stats = dict(zip(stats, context.sum(list(stats.values()))))
                    group_norms = {g["name"]: float(torch.stack([p.grad.detach().float().square().sum() for p in g["params"] if p.grad is not None]).sum().sqrt())
                                   for g in groups if any(p.grad is not None for p in g["params"])}
                    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.get("max_grad_norm", 1.0), error_if_nonfinite=True)
                    optimizer.step()
                    progress["optimizer_steps"] += 1
                    for name, value in stats.items():
                        accumulated[name] += value / objective.num_iterations
                stats = accumulated
                progress["step"] += 1
                rank_counts = context.gather({"rank": context.rank, "states": len(pack), "compute_tokens": tokens,
                                              "cursor": progress["cursor"],
                                              "prepare_wait_seconds": prepare_wait_seconds,
                                              "cpu_prepare_seconds": prepared.prepare_seconds,
                                              "peak_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
                session_states += global_states
                session_questions += global_questions
                elapsed = time.monotonic() - started
                epoch_records = sum(r["cursor"] for r in rank_counts)
                completed_records = progress["epoch"] * len(records) + epoch_records
                total_records = config.get("epochs", 1) * len(records)
                stats.update(step=progress["step"], optimizer_steps=progress["optimizer_steps"],
                             method=config["method"], epoch=progress["epoch"], states=global_states,
                             questions=global_questions, compute_tokens=global_tokens, ranks=rank_counts,
                             grad_norm=float(norm), group_grad_norms=group_norms,
                             peak_cuda_bytes=max(r["peak_cuda_bytes"] for r in rank_counts),
                             completed_records=completed_records, total_records=total_records,
                             epoch_fraction=epoch_records / len(records), elapsed_seconds=prior_elapsed + elapsed,
                             states_per_second=session_states / elapsed,
                             questions_per_second=session_questions / elapsed,
                             eta_epoch_seconds=(len(records) - epoch_records) * elapsed / session_states)
                if context.primary:
                    log.write(json.dumps(stats) + "\n")
                    log.flush()
                    print(json.dumps(stats), flush=True)
            if finished_ranks == context.world_size:
                progress.update(epoch=progress["epoch"] + 1, cursor=0, order=None)
            if global_states and progress["step"] % config.get("save_every", 100) == 0:
                media_log.flush()
                save()
        if progress["step"] == 0:
            raise ValueError("No optimizer step: dataset has no labels or epochs/max_steps is zero")
        save()
    return model, compiler, progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--attn-implementation", choices=ATTENTION_IMPLEMENTATIONS,
                        help="Override the Qwen attention backend in the training config")
    parser.add_argument("--method", choices=("sft", "rlcd"), help="Override the training objective (default: SFT)")
    parser.add_argument("--output", help="Override output directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume")
    group.add_argument("--initialize")
    args = parser.parse_args()
    try:
        config = normalize_model_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
        config = with_attention_implementation(config, args.attn_implementation)
        if args.method:
            config["method"] = args.method
        if args.output:
            config["output"] = args.output
        run(config, args.resume, args.initialize)
    finally:
        close()


if __name__ == "__main__":
    main()
