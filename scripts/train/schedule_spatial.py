"""Schedule spatial pipelines on reserved GPU leases within the configured cap."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.train.monitor_spatial import inspect_job


TERMINAL = {"succeeded", "failed", "stopped"}
EVALUATED_CHECKS = ("pipeline_complete", "warmup_epoch", "joint_epoch", "current_training_data",
    "matched_development", "model_identity", "independent_export_reload", "export_files_present",
    "required_benchmarks", "evaluation_model_identity", "reasonseg", "reasonseg_model_identity")


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def snapshot_code(root, destination):
    if destination.exists():
        return json.loads((destination / "source_manifest.json").read_text())["sha256"]
    destination.mkdir(parents=True)
    hashes = {}
    selected = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
                               "valen", "evaluation", "scripts", "configs"], cwd=root,
                              capture_output=True, check=True).stdout.decode().split("\0")
    for name in sorted(filter(None, selected)):
        relative = Path(name)
        path = root / relative
        if not path.is_file() or (path.suffix not in {".py", ".sh", ".json", ".md", ".example"}
                                  and path.name not in {"LICENSE", "COPYING", "NOTICE"}):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("LICENSE", "pyproject.toml"):
        shutil.copyfile(root / name, destination / name)
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    signature = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    atomic_json(destination / "source_manifest.json", {"sha256": signature, "files": hashes})
    return signature


def data_ready(root, task):
    seal = root / task / "manifests/ready.json"
    if not seal.is_file():
        return False
    sealed = json.loads(seal.read_text())
    conditional = task == "det" and sealed.get("format") in {
        "valen_coco_prompt_data_v2", "valen_coco_single_category_data_v3"}
    referring = task == "det" and sealed.get("format") == "valen_det_referring_data_v4"
    if (not conditional and not referring and sealed.get("format") != "valen_spatial_data_v1") or sealed.get("validated") is not True:
        return False
    if sealed.get("global_evaluation_overlap") != 0 or not sealed.get("independent_media"):
        return False
    if sealed.get("format") == "valen_coco_single_category_data_v3" and (
            sealed.get("protocol") != "coco_single_category_v3" or sealed.get("max_categories") != 1
            or sealed.get("max_coco_prompts_per_image") != 1):
        return False
    if referring and (sealed.get("protocol") != "det_referring_reasonseg_v4"
            or sealed.get("coco_training_samples") != 0
            or set(sealed.get("source_counts", {})) != {"refcoco", "refcoco+", "refcocog", "reasonseg"}
            or sum(sealed["source_counts"].values()) != 100000):
        return False
    path = root / task / "processed/train.jsonl"
    manifest = root / task / "manifests/train.json"
    if not path.is_file() or not manifest.is_file():
        return False
    prepared = json.loads(manifest.read_text())
    samples = sealed.get("samples", 100000 if task != "gui" else None)
    if type(samples) is not int or samples <= 0 or (task != "gui" and samples != 100000):
        return False
    if prepared.get("samples") != samples or prepared["records_sha256"] != sealed["records_sha256"]:
        return False
    dev = (root / "eval/det/protocols/coco_prompt/processed/dev.jsonl" if conditional else
           root / "eval" / task / "processed" / ("gui_dev.jsonl" if task == "gui" else "dev.jsonl"))
    if not dev.is_file():
        return False
    ref = {f"{name}_{split}" for name in ("refcoco", "refcocoplus") for split in ("val", "testA", "testB")}
    ref.update({"refcocog_val", "refcocog_test"})
    expected = {"det": ref | {"coco2017_val2017"}, "seg": ref | {"phrasecut_val_miniv", "phrasecut_test"},
                "gui": {"screenspot_test", "screenspot_v2_test", "screenspot_pro_test"}}
    if conditional:
        if not (root / "eval/det/protocols/coco_prompt/processed/coco2017_conditional_val2017.jsonl").is_file():
            return False
        expected["det"] = ref
    if referring:
        expected["det"] = ref
        if not all((root / "eval/det/reasonseg/processed" / (name + ".jsonl")).is_file()
                   for name in ("reasonseg_val", "reasonseg_test")):
            return False
    return all((root / "eval" / task / "processed" / (name + ".jsonl")).is_file() for name in expected[task])


def backbone_ready(root, size):
    path = root / ("Qwen3.5-" + size)
    index = path / "model.safetensors.index.json"
    if not (path / "config.json").is_file() or not index.is_file():
        return False
    weights = set(json.loads(index.read_text())["weight_map"].values())
    return all((path / name).is_file() for name in weights) and any(path.glob("*_manifest.json"))


def retain_comparison_initializations(state):
    owners = {job["run_dir"]: job for job in state["jobs"] if job.get("kind") == "pipeline"}
    for job in state["jobs"]:
        if job.get("variant") != "full" or (Path(job["run_dir"]) / "pipeline_complete.json").is_file():
            continue
        reference = Path(job["baseline_run"])
        # New pipelines retain the head themselves; only legacy pipelines need this controller copy.
        if owners.get(str(reference), {}).get("retain_warmup"):
            continue
        if ((reference / "warmup/training_complete.json").is_file()
                and not (reference / "comparison-init/initialization.json").is_file()):
            subprocess.run([sys.executable, "-m", "scripts.train.spatial_comparison",
                            "--retain-warmup", str(reference)], check=True, timeout=180)


def publish_evaluation_barriers(state, report):
    """Release dependent work after the requested reference deliverables pass."""
    models = {model["name"]: model for model in report["models"]}
    for barrier in state.get("evaluation_barriers", []):
        names = barrier["models"]
        required_checks = tuple(dict.fromkeys((*EVALUATED_CHECKS, *barrier.get("required_checks", []))))
        if not names or len(set(names)) != len(names):
            raise ValueError("An evaluation barrier requires distinct reference models")
        missing = {}
        for name in names:
            model = models.get(name, {})
            checks = (*required_checks, *barrier.get("model_required_checks", {}).get(name, []))
            failures = [key for key in checks if model.get("checks", {}).get(key) is not True]
            expected_variant = barrier.get("expected_variants", {}).get(name, "lora")
            if expected_variant not in {"lora", "full", "rlcd"}:
                raise ValueError("An evaluation barrier supports LoRA, Full or RLCD reference models")
            if model.get("variant") != expected_variant:
                failures.append(expected_variant + "_reference")
            if barrier.get("require_all_deliverables") and model.get("deliverables_ready") is not True:
                failures.append("deliverables_ready")
            if (barrier.get("allow_zero_error_skip") and expected_variant == "rlcd"
                    and model.get("skipped") is True and model.get("deliverables_ready") is True
                    and model.get("variant") == "rlcd"
                    and model.get("checks", {}).get("audited_zero_error_skip") is True):
                failures = []
            if failures:
                missing[name] = failures
        barrier.update(ready=not missing, missing=missing)
        marker = Path(barrier["marker"])
        if missing:
            marker.unlink(missing_ok=True)
        else:
            marker.parent.mkdir(parents=True, exist_ok=True)
            comparisons = {}
            for name in names:
                if barrier.get("expected_variants", {}).get(name) != "full":
                    continue
                reference = models.get(name[:-5], {})
                candidate = models[name]
                def suites(model):
                    return {**model.get("benchmarks", {}), **model.get("reasonseg_benchmarks", {}),
                            **model.get("groundingme_benchmarks", {})}
                baseline, current = suites(reference), suites(candidate)
                table = {}
                for benchmark, values in current.items():
                    other = baseline.get(benchmark, {})
                    if not other or other.get("data_sha256") != values.get("data_sha256") or other.get("samples") != values.get("samples"):
                        raise ValueError("Accepted Full/LoRA comparison has different evaluation inputs")
                    metrics = [key for key in ("accuracy", "mean_box_iou", "accuracy_at_075", "accuracy_at_09")
                               if key in values and key in other]
                    table[benchmark] = {"samples": values["samples"], "data_sha256": values["data_sha256"],
                        "lora": {key: other[key] for key in metrics}, "full": {key: values[key] for key in metrics},
                        "full_minus_lora": {key: values[key] - other[key] for key in metrics}}
                comparisons[name] = {"benchmarks": table,
                    "training": {label: {key: model.get("joint", {}).get(key) for key in
                        ("world_size", "trainable_parameters", "recorded_training_seconds", "peak_cuda_bytes_all_ranks", "settings")}
                        for label, model in (("lora", reference), ("full", candidate))}}
            kind = ("valen_evaluated_rlcd_barrier_v1" if "rlcd" in barrier.get("expected_variants", {}).values()
                    else "valen_evaluated_comparison_barrier_v1" if "full" in barrier.get("expected_variants", {}).values()
                    else "valen_evaluated_lora_barrier_v1")
            atomic_json(marker, {"format": kind, "models": names,
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "verified_checks": list(required_checks),
                "expected_variants": barrier.get("expected_variants", {}),
                "require_all_deliverables": barrier.get("require_all_deliverables", False),
                "audited_zero_error_skips": [name for name in names if models[name].get("skipped") is True],
                "model_required_checks": barrier.get("model_required_checks", {}),
                "metric_comparisons": comparisons,
                "export_identities": {name: models[name]["export_identity"] for name in names}})


def refresh_group_registrations(state):
    """Bind replacement physical replicas only after a complete, disjoint registration."""
    physical = {a["name"]: a for a in state.get("allocations", [])}
    for group in state.get("execution_groups", []):
        directory = group.get("registration_dir")
        if not directory:
            continue
        expected = [(name, rank) for name in group["members"] for rank in range(physical[name]["nodes"])]
        rows = []
        for name, rank in expected:
            path = Path(directory) / f"{name}.node-{rank}.json"
            if not path.is_file():
                break
            row = json.loads(path.read_text())
            if (row.get("allocation") != name or row.get("replica") != rank
                    or not row.get("hostname") or len(row.get("gpu_uuids", [])) != physical[name]["gpus"] // physical[name]["nodes"]):
                raise ValueError("Replacement replica registration differs from its allocation")
            rows.append(row)
        if len(rows) != len(expected):
            continue
        hosts = [r["hostname"] for r in rows]
        cards = [card for r in rows for card in r["gpu_uuids"]]
        if len(set(hosts)) != group["nodes"] or len(set(cards)) != group["gpus"]:
            raise ValueError("Replacement replicas overlap in host identity or GPU UUID")
        group["node_rank_map"] = {host: rank for rank, host in enumerate(hosts)}
        group["gpu_uuids"] = cards
        for name in group["members"]:
            physical[name]["hostnames"] = [r["hostname"] for r in rows if r["allocation"] == name]


def run_allocation_pool(args, state, registry, data_root, model_root, snapshot):
    from scripts.train.monitor_spatial import inspect_registry
    from scripts.train.spatial_pool import configure_allocations
    if not state.get("examples_code_snapshot"):
        examples_code = snapshot
        if not (examples_code / "evaluation/examples.py").is_file():
            examples_code = registry.parent / "code-examples"
            snapshot_code(Path(__file__).resolve().parents[2], examples_code)
        state["examples_code_snapshot"] = str(examples_code)
    while not (registry.parent / "STOP_SCHEDULER").exists():
        refresh_group_registrations(state)
        retain_comparison_initializations(state)
        if state.get("evaluation_barriers"):
            from scripts.train.summarize_spatial import collect_results
            publish_evaluation_barriers(state, collect_results(state, data_root, model_root))
        configure_allocations(state, data_root, model_root, snapshot,
                              lambda task, size: data_ready(data_root, task) and backbone_ready(model_root, size))
        if sum(item["gpus"] for item in state["allocations"]) > args.gpu_limit:
            raise ValueError("Existing allocations exceed the configured GPU budget")
        # Persist the handoff before querying the cluster. Workers never wait for a fresh submission.
        atomic_json(registry, state)
        report = inspect_registry(state, args.scheduler)
        by_name = {item["name"]: item for item in report["jobs"]}
        for job in state["jobs"]:
            job["state"] = by_name[job["name"]]["state"]
        for allocation, observed in zip(state["allocations"], report["allocations"]):
            allocation["state"] = observed["state"]
        for group, observed in zip(state.get("execution_groups", []), report.get("execution_groups", [])):
            group["state"] = observed["state"]
        state.update(last_check_utc=datetime.now(timezone.utc).isoformat(),
                     reserved_gpus=report["reserved_gpus"], running_gpus=report["running_gpus"])
        summary_ready = not args.summary_dir
        if args.summary_dir:
            from scripts.train.summarize_spatial import write_summary
            try:
                summary = write_summary(state, data_root, model_root, args.summary_dir)
                summary_ready = summary["deliverables_ready"]
                state.pop("summary_error", None)
            except (ValueError, TypeError, OSError) as exc:
                state["summary_error"] = str(exc)
                print(json.dumps({"summary_error": str(exc)}), flush=True)
        atomic_json(registry, state)
        if args.progress_file:
            from scripts.train.spatial_progress import write_progress
            write_progress(state, registry, data_root, model_root, args.progress_file)
        pipelines = [job for job in state["jobs"] if job.get("kind") == "pipeline"]
        print(json.dumps({"time_utc": state["last_check_utc"], "policy": state["scheduling_policy"],
                          "reserved_gpus": state["reserved_gpus"],
                          "states": {job["name"]: job["state"] for job in pipelines}}), flush=True)
        reasonseg_complete = (not state.get("reasonseg_support") or all(
            (Path(job["run_dir"]) / "evaluation-reasonseg/pipeline_complete.json").is_file()
            for job in pipelines if job["task"] in {"det", "seg"}))
        examples_complete = all((Path(job["run_dir"]) / "examples/pipeline_complete.json").is_file() for job in pipelines)
        if all(job["state"] == "succeeded" for job in pipelines) and reasonseg_complete and examples_complete and summary_ready:
            atomic_json(registry.parent / "all_models_complete.json",
                        {"models": len(pipelines), "code_sha256": state["code_sha256"]})
            return
        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", default="models")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--code-snapshot", required=True)
    parser.add_argument("--scheduler", default=os.environ.get("RJOB_BIN", "rjob"))
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--gpu-limit", type=int, default=64)
    parser.add_argument("--nodes-per-job", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--reuse-allocations", action="store_true",
                        help="Adopt existing eight-GPU jobs and execute their assigned model queues")
    parser.add_argument("--progress-file", help="Optional Markdown checklist, refreshed after each status check")
    parser.add_argument("--summary-dir", help="Optional compact result archive, refreshed after each status check")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if args.gpu_limit < 8 or args.gpu_limit > 80 or args.gpu_limit % 8:
        parser.error("gpu-limit must be a multiple of eight, at most 80")
    root = Path(__file__).resolve().parents[2]
    data_root = Path(args.data_root).resolve()
    model_root = Path(args.model_root).resolve()
    registry = Path(args.registry).resolve()
    snapshot = Path(args.code_snapshot).resolve()
    signature = snapshot_code(root, snapshot)
    state = json.loads(registry.read_text()) if registry.exists() else {"jobs": []}
    for task in ("det", "seg", "gui"):
        for size in ("0.8B", "2B", "4B", "9B"):
            if any(job.get("task") == task and job.get("size") == size for job in state["jobs"]):
                continue
            name = f"valen-decision-{task}-{size.lower().replace('.', '')}-{args.run_id.lower()}"
            run = root / ("next-generation-" + task) / size / args.run_id
            state["jobs"].append({"name": name, "kind": "pipeline", "task": task, "size": size, "gpus": 8,
                "nodes": args.nodes_per_job,
                "log_dir": str(root / ("next-generation-" + task) / "jobs" / name),
                "outputs": [str(run)], "run_dir": str(run), "state": "not_submitted"})
    state.update(code_sha256=signature, code_snapshot=str(snapshot), gpu_limit=args.gpu_limit)
    if args.progress_file:
        state["progress_file"] = str(Path(args.progress_file).resolve())
    else:
        args.progress_file = state.get("progress_file")
    if args.summary_dir:
        state["results_dir"] = str(Path(args.summary_dir).resolve())
    else:
        args.summary_dir = state.get("results_dir")
    atomic_json(registry, state)
    if args.reuse_allocations:
        run_allocation_pool(args, state, registry, data_root, model_root, snapshot)
        return
    while not (registry.parent / "STOP_SCHEDULER").exists():
        reserved = 0
        smoke_ok = False
        for job in state["jobs"]:
            if job.get("job_id"):
                report = inspect_job(job, args.scheduler)
                job["state"] = report["state"]
                if job["state"] not in TERMINAL:
                    reserved += 8
                if job.get("kind") != "pipeline" and job["state"] == "succeeded":
                    marker = Path(job.get("completion_marker", "missing"))
                    if marker.is_file():
                        validation = json.loads(marker.read_text())
                        smoke_ok = validation.get("ranks") == 8 and set(validation.get("tasks", {})) == {"det", "seg", "gui"}
                if job.get("kind") == "pipeline" and job["state"] == "succeeded":
                    if not (Path(job["run_dir"]) / "pipeline_complete.json").is_file():
                        job["state"] = "failed"
                        job["failure_reason"] = "Missing model/evaluation completion marker"
        # A completed GPU smoke and immutable dataset/base snapshots gate every formal run.
        if smoke_ok:
            for job in state["jobs"]:
                if job.get("kind") != "pipeline" or job.get("job_id") or reserved + 8 > args.gpu_limit:
                    continue
                if not data_ready(data_root, job["task"]) or not backbone_ready(model_root, job["size"]):
                    continue
                job["state"] = "submitting"
                atomic_json(registry, state)
                nodes = job["nodes"]
                environment = dict(os.environ, VALEN_RUN_NAME=job["name"], VALEN_GPUS=str(8 // nodes),
                    VALEN_NODES=str(nodes), VALEN_CPUS=str(64 // nodes),
                    VALEN_ENFORCE_RESOURCES="1",
                    VALEN_MEMORY_MIB=str(524288 // nodes),
                    VALEN_EXPERIMENT_ROOT=str(root / ("next-generation-" + job["task"])),
                    VALEN_CODE_ROOT=str(snapshot), VALEN_PYTHON=str(root / ".venv/bin/python"))
                model_name = "Valen-Decision-" + {"det": "Det", "seg": "Seg", "gui": "GUI"}[job["task"]] + "-" + job["size"]
                command = ["bash", str(root / "scripts/train/submit_experiment.sh"),
                    "-m", "scripts.train.run_spatial_pipeline", "--task", job["task"],
                    "--model-path", str(model_root / ("Qwen3.5-" + job["size"])),
                    "--data-root", str(data_root), "--output-dir", job["run_dir"],
                    "--export-dir", str(model_root / model_name), "--nodes", str(nodes),
                    "--gpus-per-node", str(8 // nodes), "--launch-id", job["name"]]
                result = subprocess.run(command, env=environment, cwd=root, capture_output=True, text=True, timeout=180)
                id_file = Path(job["log_dir"]) / "job_id"
                if result.returncode != 0 or not id_file.is_file():
                    job.update(state="submission_failed", failure_reason=(result.stdout + result.stderr)[-3000:])
                    atomic_json(registry, state)
                    raise RuntimeError("Submission did not return a stable job ID; inspect before retrying")
                job.update(job_id=id_file.read_text().strip(), state="queued")
                reserved += 8
                print(json.dumps({"submitted": job["job_id"], "reserved_gpus": reserved}), flush=True)
                atomic_json(registry, state)
        state["last_check_utc"] = datetime.now(timezone.utc).isoformat()
        state["reserved_gpus"] = reserved
        atomic_json(registry, state)
        pipelines = [job for job in state["jobs"] if job.get("kind") == "pipeline"]
        if all(job.get("state") == "succeeded" for job in pipelines):
            atomic_json(registry.parent / "all_models_complete.json", {"models": len(pipelines), "code_sha256": signature})
            break
        print(json.dumps({"time_utc": state["last_check_utc"], "smoke_passed": smoke_ok,
                          "reserved_gpus": reserved, "states": {job["name"]: job.get("state") for job in pipelines}}), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
