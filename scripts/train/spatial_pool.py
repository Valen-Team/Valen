"""Describe persistent allocations separately from the model runs they execute."""
import json
from pathlib import Path
import sys

from scripts.train.spatial_worker import atomic_json


def configure_allocations(state, data_root, model_root, code_root, ready):
    pipelines = [job for job in state["jobs"] if job.get("kind") == "pipeline"]
    allocations = state.setdefault("allocations", [])
    if not allocations:
        for size in ("0.8B", "2B", "4B", "9B"):
            jobs = [job for job in pipelines if job["size"] == size]
            owners = [job for job in jobs if job.get("job_id")]
            if len(jobs) != 3 or len(owners) != 1:
                raise ValueError("Reuse requires one existing allocation for each size, with three task runs")
            owner = owners[0]
            allocation = {key: owner[key] for key in ("name", "job_id", "gpus", "nodes", "log_dir")}
            allocation.update(kind="allocation", size=size,
                              initial_pipeline=owner["name"],
                              queue=str(Path(owner["log_dir"]) / "allocation/queue.json"))
            allocations.append(allocation)
            for job in jobs:
                job["allocation"] = owner["name"]
    groups = state.get("execution_groups", [])
    physical = {allocation["name"]: allocation for allocation in allocations}
    membership = [name for group in groups for name in group["members"]]
    if len(membership) != len(set(membership)) or any(name not in physical for name in membership):
        raise ValueError("Execution groups must use disjoint reserved allocations")
    for group in groups:
        if (group["gpus"] != sum(physical[name]["gpus"] for name in group["members"])
                or group["nodes"] != sum(physical[name]["nodes"] for name in group["members"])
                or group["gpus"] not in (16, 64)):
            raise ValueError("A sixteen- or sixty-four-GPU group must match its physical members")
    names = [allocation["name"] for allocation in allocations + groups]
    ids = [allocation["job_id"] for allocation in allocations if allocation.get("job_id")]
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("Each allocation must have a unique name and scheduler job ID")
    if sum(allocation["gpus"] for allocation in allocations) > state.get("gpu_limit", 64):
        raise ValueError("Allocations exceed the GPU budget")
    if any(job.get("allocation") not in names for job in pipelines):
        raise ValueError("Each model must belong to exactly one allocation")
    for allocation in allocations + groups:
        jobs = [job for job in pipelines if job["allocation"] == allocation["name"]]
        jobs.sort(key=lambda job: (("gui", "det", "seg").index(job["task"]), job.get("variant") == "full"))
        if ((not jobs and not allocation.get("execution_group")) or
                (allocation["gpus"] not in (8, 16, 64) and not
                 (allocation["gpus"] == 4 and allocation.get("execution_group") and not jobs))
                or allocation["nodes"] not in (1, 2, 4, 8, 16)
                or allocation["gpus"] % allocation["nodes"]):
            raise ValueError("Each allocation requires eight or sixteen GPUs and valid queued work")
        if "initial_pipeline" not in allocation:
            allocation["initial_pipeline"] = allocation["name"]
        initial = next((job for job in jobs if job["name"] == allocation["initial_pipeline"]), None)
        if allocation["initial_pipeline"] is not None and initial is None:
            raise ValueError("The initial pipeline must remain assigned to its allocation")
        allocation["outputs"] = [job["run_dir"] for job in jobs]
        queue = Path(allocation["queue"])
        queue.parent.mkdir(parents=True, exist_ok=True)
        tasks, deferred = [], []
        for job in jobs:
            if job is initial:
                continue
            destination = deferred if job.get("after_queue") else tasks
            size = job["size"]
            job_data_root = Path(job.get("data_root", data_root))
            model_name = "Valen-Decision-" + {"det": "Det", "seg": "Seg", "gui": "GUI"}[job["task"]] + "-" + size
            if job.get("variant") == "full":
                model_name += "-Full"
            elif job.get("variant") == "rlcd":
                model_name += "-RLCD" if job.get("initialization_variant") == "lora_merged" else "-Full-RLCD"
            command = [sys.executable, "-m", "scripts.train.run_spatial_pipeline", "--task", job["task"],
                "--model-path", str(model_root / ("Qwen3.5-" + size)), "--data-root", str(job_data_root),
                "--output-dir", job["run_dir"], "--export-dir", job.get("export_dir", str(model_root / model_name)),
                "--nodes", str(allocation["nodes"]), "--gpus-per-node", str(allocation["gpus"] // allocation["nodes"]),
                "--launch-id", job.get("launch_id", allocation["name"] + "-" + job["task"])]
            if job.get("variant") == "full":
                command.extend(["--finetuning-type", "full", "--baseline-run", job["baseline_run"]])
                if not job.get("launch_id"):
                    command[command.index("--launch-id") + 1] += "-full"
            if job.get("retain_warmup"):
                command.append("--retain-warmup")
            if job.get("defer_evaluation"):
                command.append("--defer-evaluation")
            if job.get("with_reasonseg"):
                command.append("--with-reasonseg")
            if job.get("training_world_size"):
                command.extend(["--training-world-size", str(job["training_world_size"])])
            if job.get("training_samples"):
                command.extend(["--training-samples", str(job["training_samples"])])
            if job.get("variant") == "rlcd":
                reference_variant = job.get("initialization_variant", "full")
                command = [sys.executable, "-m", "scripts.train.run_det_rlcd_pipeline",
                    "--model-path", str(model_root/("Qwen3.5-"+size)),
                    "--lora-baseline-run", job["baseline_run"],
                    "--reference-variant", reference_variant,
                    "--reference-model", job.get("reference_export", str(model_root / (
                        "Valen-Decision-Det-"+size+("" if reference_variant=="lora_merged" else "-Full")))),
                    "--data-root", str(job_data_root), "--output-dir", job["run_dir"],
                    "--subset-dir", job["subset_dir"], "--export-dir", job.get("export_dir", str(model_root/model_name)),
                    "--acceptance-marker", job["evaluation_gate"],
                    "--nodes", str(allocation["nodes"]), "--gpus-per-node", str(allocation["gpus"]//allocation["nodes"]),
                    "--launch-id", job.get("launch_id", job["name"])]
                if reference_variant == "full" or (job.get("full_baseline_run") and
                        (Path(job["full_baseline_run"])/"pipeline_complete.json").is_file()):
                    command.extend(["--full-baseline-run", job["full_baseline_run"],
                                    "--full-model", str(model_root/("Valen-Decision-Det-"+size+"-Full"))])
            destination.append({"name": job["name"], "task": job["task"], "run_dir": job["run_dir"],
                          "log_dir": job["log_dir"], "command": command,
                          "code_root": job.get("code_snapshot", str(code_root)),
                          "ready": not job.get("paused", False) and ready(job["task"], size) and
                          all(Path(path).is_file() for path in job.get("requires_files", []))})
            if job.get("reasonseg_after") and not job.get("defer_evaluation"):
                output = Path(job["run_dir"]) / "evaluation-reasonseg"
                destination.append({"name": job["name"] + "-reasonseg", "task": job["task"], "run_dir": str(output),
                    "log_dir": job["log_dir"] + "-reasonseg", "code_root": job["reasonseg_code_snapshot"],
                    "command": [sys.executable, "-m", "evaluation.reasonseg", "--task", job["task"],
                        "--data-root", str(job_data_root / "eval" / job["task"] / "reasonseg"),
                        "--model-path", job.get("export_dir", str(model_root / model_name)),
                        "--output-dir", str(output), "--nodes", str(allocation["nodes"]),
                        "--gpus-per-node", str(allocation["gpus"] // allocation["nodes"]),
                        "--launch-id", allocation["name"] + "-" + job["task"] + "-reasonseg"],
                    "ready": (Path(job["run_dir"]) / "pipeline_complete.json").is_file() and
                        (job_data_root / "eval" / job["task"] / "reasonseg/manifests/preflight.json").is_file()})
        if state.get("examples_code_snapshot"):
            for job in jobs:
                if job.get("variant") == "rlcd" or job.get("defer_evaluation"):
                    continue  # RLCD owns its examples; a zero-error skip deliberately has no model package.
                job_data_root = Path(job.get("data_root", data_root))
                destination = deferred if job.get("after_queue") else tasks
                model_name = "Valen-Decision-" + {"det": "Det", "seg": "Seg", "gui": "GUI"}[job["task"]] + "-" + job["size"]
                if job.get("variant") == "full":
                    model_name += "-Full"
                elif job.get("variant") == "rlcd":
                    model_name += "-Full-RLCD"
                output = Path(job["run_dir"]) / "examples"
                destination.append({"name": job["name"] + "-examples", "task": job["task"], "run_dir": str(output),
                    "log_dir": job["log_dir"] + "-examples", "code_root": job.get("examples_code_snapshot", state["examples_code_snapshot"]),
                    "command": [sys.executable, "-m", "evaluation.examples", "--task", job["task"],
                        "--data-root", str(job_data_root / "eval" / job["task"]),
                        "--model-path", job.get("export_dir", str(model_root / model_name)),
                        "--output-dir", str(output), "--nodes", str(allocation["nodes"]),
                        "--gpus-per-node", str(allocation["gpus"] // allocation["nodes"]), "--launch-id", job["name"] + "-examples"],
                    "ready": (Path(job["run_dir"]) / "pipeline_complete.json").is_file()})
        preflight = []
        for item in allocation.get("preflight_tasks", []):
            task = {key: value for key, value in item.items() if key != "requires_files"}
            task["ready"] = task.get("ready", True) and all(Path(path).is_file() for path in item.get("requires_files", []))
            preflight.append(task)
        ordered = preflight + tasks + deferred
        for extra in reversed(allocation.get("extra_tasks", [])):
            parent = extra["after"]
            if parent not in {job["name"] for job in jobs}:
                raise ValueError("An extra task must follow a model in its allocation")
            parent_job = next(job for job in jobs if job["name"] == parent)
            task = {key: value for key, value in extra.items() if key != "after"}
            task["ready"] = (task.get("ready", True) and (Path(parent_job["run_dir"]) / "pipeline_complete.json").is_file()
                             and all(Path(p).is_file() for p in task.get("requires_files", [])))
            index = 0 if initial and parent == initial["name"] else next(
                i + 1 for i, item in enumerate(ordered) if item["name"] == parent)
            ordered.insert(index, task)
        if allocation.get("task_order"):
            requested = allocation["task_order"]
            if len(requested) != len(ordered) or set(requested) != {item["name"] for item in ordered}:
                raise ValueError("Explicit task order must contain every queued task exactly once")
            positions = {name: index for index, name in enumerate(requested)}
            ordered.sort(key=lambda item: positions[item["name"]])
        if len({item["name"] for item in ordered}) != len(ordered):
            raise ValueError("Allocation task names must be unique")
        for task in ordered:
            if task.get("task") in state.get("suspended_tasks", []):
                task["ready"] = False
        content = {"format": "valen_spatial_allocation_v1", "name": allocation["name"], "nodes": allocation["nodes"],
                   "gpus_per_node": allocation["gpus"] // allocation["nodes"], "code_root": str(code_root),
                   "initial": {"name": initial["name"], "run_dir": initial["run_dir"]} if initial else None,
                   "tasks": ordered}
        if allocation.get("node_rank_map"):
            content["node_rank_map"] = allocation["node_rank_map"]
        if not queue.exists() or json.loads(queue.read_text()) != content:
            atomic_json(queue, content)
    state["scheduling_policy"] = "reuse_reserved_gpu_allocations" if groups else "reuse_eight_gpu_allocations"


def pipeline_state(job, allocation):
    if (Path(job["run_dir"]) / "pipeline_complete.json").is_file():
        return "succeeded"
    if job.get("paused"):
        return "paused"
    queue = Path(allocation["queue"])
    root = queue.parent
    results = [json.loads(path.read_text()) for path in root.glob(f"result-{job['name']}.node-*.json")]
    if any(row["returncode"] != 0 for row in results):
        return "failed"
    workers = [json.loads(path.read_text()) for path in root.glob("worker.node-*.json")]
    if any(row.get("task") == job["name"] and row.get("state") == "blocked" for row in workers):
        return "failed"
    if allocation["state"] in {"failed", "stopped", "succeeded"}:
        return "failed" if job.get("job_id") else "blocked"
    if job.get("job_id"):
        return allocation["state"]
    if any(row.get("task") == job["name"] and row.get("state") in {"running", "finishing"} for row in workers):
        return "running"
    tasks = json.loads(queue.read_text())["tasks"]
    if any(not Path(path).is_file() for path in job.get("requires_files", [])):
        return "waiting_for_dependencies"
    if not next(item for item in tasks if item["name"] == job["name"])["ready"]:
        return "waiting_for_data"
    return "queued" if allocation["state"] in {"queued", "pending", "inqueue"} else "queued_in_allocation"
