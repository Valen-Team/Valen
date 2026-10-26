"""Record scheduler state, training progress and GPU telemetry every 15 minutes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def last_metric(path):
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 32768))
        for line in reversed(stream.read().splitlines()):
            try:
                return json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
    return None


def inspect_job(job, scheduler):
    result = {"name": job["name"], "job_id": job.get("job_id"), "gpus": job.get("gpus", 8)}
    log_dir = Path(job["log_dir"])
    statuses = [log_dir / "status.log"] if job.get("nodes", 1) == 1 else sorted(log_dir.glob("status.node-*.log"))
    statuses = [path for path in statuses if path.is_file()]
    exits = [path.read_text().strip() for path in statuses]
    failed = any(value != "EXIT_CODE=0" for value in exits)
    complete = len(exits) == job.get("nodes", 1)
    if (failed or complete) and job.get("kind") != "allocation":
        result["exit_status"] = exits
        result["state"] = "failed" if failed else "succeeded"
    elif result["job_id"]:
        try:
            query = subprocess.run([scheduler, "list", result["job_id"]], capture_output=True,
                                   text=True, timeout=45)
            summary = query.stdout + query.stderr
            result["scheduler"] = summary[-5000:]
            result["state"] = next((label.lower() for label in ("Succeeded", "Failed", "Stopped", "Running", "Pending")
                                    if f": {label}" in summary), "unknown")
            if result["state"] == "unknown" and "Inqueue" in summary:
                result["state"] = "queued"
        except (OSError, subprocess.TimeoutExpired) as exc:
            result.update(state="unknown", scheduler=str(exc))
    else:
        result["state"] = "not_submitted"
    outputs = [Path(path) for path in job.get("outputs", [])]
    metrics = [path / "metrics.jsonl" for output in outputs for path in [output, *output.glob("*")]
               if (path / "metrics.jsonl").is_file()]
    if metrics:
        latest = max(metrics, key=lambda p: p.stat().st_mtime)
        result["metrics"] = last_metric(latest)
        result["metrics_age_seconds"] = round(time.time() - latest.stat().st_mtime)
        telemetry = latest.parent / "gpu.json"
        if telemetry.is_file():
            result["gpu_telemetry"] = json.loads(telemetry.read_text())
    return result


def inspect_registry(state, scheduler):
    """Count physical allocations once, even when they serve several model runs."""
    from scripts.train.spatial_pool import pipeline_state
    allocations = []
    physical_allocations = state.get("allocations", [])
    with ThreadPoolExecutor(max_workers=8) as pool:
        observations = list(pool.map(lambda allocation: inspect_job(allocation, scheduler), physical_allocations))
    for allocation, observed in zip(physical_allocations, observations):
        report = dict(allocation, **observed)
        report["workers"] = [json.loads(path.read_text()) for path in
                             Path(allocation["queue"]).parent.glob("worker.node-*.json")]
        if not report["workers"] and allocation.get("execution_group"):
            group = next(item for item in state["execution_groups"] if item["name"] == allocation["execution_group"])
            report["workers"] = [row for path in Path(group["queue"]).parent.glob("worker.node-*.json")
                                 if (row := json.loads(path.read_text()))["hostname"] in allocation["hostnames"]]
        allocations.append(report)
    by_name = {item["name"]: item for item in allocations}
    groups = []
    for group in state.get("execution_groups", []):
        members = [by_name[name] for name in group["members"]]
        states = {member["state"] for member in members}
        assigned = [job for job in state.get("jobs", [])
                    if job.get("kind") == "pipeline" and job.get("allocation") == group["name"]]
        complete = bool(assigned) and all(
            (Path(job["run_dir"]) / marker).is_file()
            for job in assigned for marker in ("pipeline_complete.json", "examples/pipeline_complete.json"))
        observed = ("succeeded" if complete else "running" if states == {"running"} else
                    "failed" if states & {"failed", "stopped", "succeeded"} else
                    "unknown" if "unknown" in states else "pending")
        report = dict(group, state=observed, job_id=None,
                      job_ids=[member["job_id"] for member in members])
        report["workers"] = [json.loads(path.read_text()) for path in Path(group["queue"]).parent.glob("worker.node-*.json")]
        groups.append(report)
        by_name[group["name"]] = report
    jobs, independent = [], []
    for job in state.get("jobs", []):
        allocation = by_name.get(job.get("allocation"))
        if allocation is None:
            report = inspect_job(job, scheduler)
            independent.append(report)
        else:
            report = inspect_job(dict(job, job_id=None), scheduler)
            report.update(state=pipeline_state(job, allocation), allocation=allocation["name"],
                          job_id=allocation.get("job_id"))
            if allocation.get("job_ids"):
                report["job_ids"] = allocation["job_ids"]
        jobs.append(report)
    physical = allocations + independent
    active = [item for item in physical if item["state"] not in
              {"succeeded", "failed", "stopped", "not_submitted"}]
    return {"jobs": jobs, "allocations": allocations, "execution_groups": groups,
            "reserved_gpus": sum(item["gpus"] for item in active),
            "running_gpus": sum(item["gpus"] for item in active if item["state"] == "running")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--interval", type=int, default=900)
    parser.add_argument("--scheduler", default=os.environ.get("RJOB_BIN", "rjob"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be positive")
    registry = Path(args.registry)
    while True:
        started = time.monotonic()
        state = json.loads(registry.read_text())
        report = {"time_utc": datetime.now(timezone.utc).isoformat(), "interval_seconds": args.interval,
                  **inspect_registry(state, args.scheduler)}
        report["gpu_limit"] = state.get("gpu_limit", 64)
        report["budget_exceeded"] = report["reserved_gpus"] > report["gpu_limit"]
        temporary = registry.parent / "monitor-latest.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(registry.parent / "monitor-latest.json")
        print(json.dumps(report), flush=True)
        if args.once:
            break
        deadline = started + args.interval
        while time.monotonic() < deadline and not (registry.parent / "STOP_MONITOR").exists():
            time.sleep(min(5, max(.01, deadline - time.monotonic())))
        if (registry.parent / "STOP_MONITOR").exists():
            break


if __name__ == "__main__":
    main()
