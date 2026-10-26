"""Keep reserved GPU replicas and run their queued pipelines."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class AllocationWorker:
    def __init__(self, queue, interval=2):
        self.queue = Path(queue).resolve()
        self.root = self.queue.parent
        self.interval = interval
        self.rank = int(os.environ.get("KUBEBRAIN_REPLICA", "0"))
        self.config = json.loads(self.queue.read_text())
        self.nodes = self.config["nodes"]
        mapping = self.config.get("node_rank_map")
        replicas = int(os.environ.get("KUBEBRAIN_REPLICA_TOTAL", "1"))
        if mapping is not None:
            if set(mapping.values()) != set(range(self.nodes)) or len(mapping) != self.nodes:
                raise ValueError("Grouped allocations require one unique hostname per logical rank")
            self.rank = mapping[socket.gethostname()]
            replicas = self.nodes
        if (self.nodes * self.config["gpus_per_node"] not in (8, 16, 64)
                or replicas != self.nodes
                or not 0 <= self.rank < self.nodes):
            raise ValueError("A reusable allocation must contain eight, sixteen or sixty-four GPUs")
        self.status_path = self.root / f"worker.node-{self.rank}.json"
        self.status = {"pid": os.getpid(), "hostname": socket.gethostname(), "replica": self.rank,
                       "allocation": self.config["name"], "worker_protocol": 3}
        try:
            self.status["gpu_uuids"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=15).splitlines()
        except (OSError, subprocess.SubprocessError):
            self.status["gpu_uuids"] = []
        self.last_status = 0

    def update(self, state, task=None, **extra):
        now = time.monotonic()
        changed = self.status.get("state") != state or self.status.get("task") != task
        if changed or now - self.last_status >= 30 or extra:
            self.status.update(state=state, task=task, time_utc=datetime.now(timezone.utc).isoformat(), **extra)
            atomic_json(self.status_path, self.status)
            self.last_status = now
            if changed:
                print(json.dumps(self.status), flush=True)

    def stopping(self):
        return (self.root / "STOP").exists()

    def wait(self, predicate, state, task=None):
        while not predicate():
            if self.stopping():
                self.update("stopped", task)
                return False
            self.update(state, task)
            time.sleep(self.interval)
        return True

    def records(self, prefix):
        return [json.loads(path.read_text()) for path in sorted(self.root.glob(prefix + ".node-*.json"))]

    def terminate(self, process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def run_task(self, task):
        name = task["name"]
        # Every replica must reach the handoff before either launches the next pipeline.
        atomic_json(self.root / f"ready-{name}.node-{self.rank}.json", {"ready": True})
        if not self.wait(lambda: len(self.records("ready-" + name)) == self.nodes, "handoff", name):
            return False
        log_dir = Path(task["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.update("running", name)
        code_root = task.get("code_root", self.config["code_root"])
        environment = dict(os.environ)
        environment.update(KUBEBRAIN_REPLICA=str(self.rank), KUBEBRAIN_REPLICA_TOTAL=str(self.nodes))
        environment["PYTHONPATH"] = str(code_root) + os.pathsep + environment.get("PYTHONPATH", "")
        with (log_dir / f"console.node-{self.rank}.log").open("a") as log:
            process = subprocess.Popen(task["command"], cwd=code_root, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    results = self.records("result-" + name)
                    if self.stopping() or any(row["returncode"] != 0 for row in results):
                        self.terminate(process)
                        break
                    self.update("running", name)
                    time.sleep(self.interval)
            finally:
                self.terminate(process)
        atomic_json(self.root / f"result-{name}.node-{self.rank}.json", {"returncode": process.returncode})
        if not self.wait(lambda: len(self.records("result-" + name)) == self.nodes, "finishing", name):
            return False
        results = self.records("result-" + name)
        marker = Path(task["run_dir"]) / "pipeline_complete.json"
        if any(row["returncode"] != 0 for row in results) or not marker.is_file():
            # Preserve the allocation for diagnosis instead of silently moving past a failed model.
            self.update("blocked", name, failure="Pipeline failed; inspect task logs before stopping this allocation")
            self.wait(lambda: False, "blocked", name)
            return False
        self.update("completed_task", name)
        return True

    def run(self):
        initial = self.config.get("initial")
        if initial:
            self.update("waiting_for_initial_pipeline", initial["name"])
            marker = Path(initial["run_dir"]) / "pipeline_complete.json"
            if not self.wait(marker.is_file, "waiting_for_initial_pipeline", initial["name"]):
                return
        else:
            self.update("waiting_for_workers")
        if not self.wait(lambda: len(self.records("worker")) == self.nodes, "handoff"):
            return
        completed = set()
        sequence = 0
        while True:
            selection = self.root / f"selection-{sequence}.json"
            pending = [item for item in json.loads(self.queue.read_text())["tasks"] if item["name"] not in completed
                       and not (Path(item["run_dir"]) / "pipeline_complete.json").is_file()]
            waiting_task = pending[0]["name"] if pending else None
            if self.rank == 0:
                def choose():
                    current = json.loads(self.queue.read_text())
                    pending = [item for item in current["tasks"] if item["name"] not in completed
                               and not (Path(item["run_dir"]) / "pipeline_complete.json").is_file()]
                    available = next((item for item in pending if item["ready"]), None)
                    if available is not None or not pending:
                        atomic_json(selection, {"name": available["name"] if available else None})
                        return True
                    return False
                if not self.wait(choose, "waiting_for_data", waiting_task):
                    return
            elif not self.wait(selection.is_file, "waiting_for_data", waiting_task):
                return
            name = json.loads(selection.read_text())["name"]
            if name is None:
                break
            original = next(item for item in json.loads(self.queue.read_text())["tasks"] if item["name"] == name)
            if not self.run_task(original):
                return
            completed.add(name)
            sequence += 1
        self.update("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    AllocationWorker(args.queue, args.interval).run()


if __name__ == "__main__":
    main()
