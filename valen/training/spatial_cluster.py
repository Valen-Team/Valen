"""Coordinate an eight-, sixteen- or sixty-four-GPU spatial pipeline."""
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


class SpatialCluster:
    def __init__(self, directory, nodes=1, gpus_per_node=8):
        self.nodes, self.gpus = nodes, gpus_per_node
        self.rank = int(os.environ.get("KUBEBRAIN_REPLICA", "0"))
        replicas = int(os.environ.get("KUBEBRAIN_REPLICA_TOTAL", "1"))
        if nodes * gpus_per_node not in (8, 16, 64) or replicas != nodes or not 0 <= self.rank < nodes:
            raise ValueError("Each spatial job requires eight, sixteen or sixty-four GPUs with matching replicas")
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / f"node-{self.rank}.json").exists():
            raise FileExistsError("Use a fresh coordination directory for each cluster submission")
        self.phase = 0
        self.write(self.root / f"node-{self.rank}.json", {
            "ip": os.environ.get("POD_IP") or socket.gethostbyname(socket.gethostname()),
            "gpus": gpus_per_node})

    @property
    def primary(self):
        return self.rank == 0

    @staticmethod
    def write(path, value):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(value) + "\n")
        temporary.replace(path)

    def wait(self, predicate, timeout=1800):
        deadline = time.monotonic() + timeout
        while not predicate():
            if list(self.root.glob("failure-*.json")):
                raise RuntimeError("Another replica failed; inspect the recorded phase")
            if time.monotonic() > deadline:
                raise TimeoutError("A pipeline replica did not reach its synchronization point")
            time.sleep(1)

    def decision(self, name, value):
        path = self.root / ("decision-" + name + ".json")
        if self.primary:
            self.write(path, {"value": value})
        self.wait(path.exists)
        return json.loads(path.read_text())["value"]

    def execute(self, arguments, distributed=False, all_nodes=False):
        """Run one primary command, a torchrun phase, or an existing coordinator per node."""
        if distributed and all_nodes:
            raise ValueError("An existing coordinator must not be wrapped in torchrun")
        phase = self.phase
        self.phase += 1
        keeper = None
        command = [sys.executable, *arguments]
        try:
            if distributed:
                if self.nodes == 1:
                    launch = ["--standalone", f"--nproc_per_node={self.gpus}"]
                else:
                    master_file = self.root / "node-0.json"
                    self.wait(master_file.exists)
                    master = json.loads(master_file.read_text())["ip"]
                    rendezvous = self.root / f"rendezvous-{phase}.json"
                    identity = hashlib.sha256(f"{self.root.resolve()}:{phase}".encode()).hexdigest()[:24]
                    if self.primary:
                        from torch.distributed import TCPStore
                        keeper = TCPStore(master, 0, is_master=True, wait_for_workers=False,
                                          timeout=timedelta(seconds=60))
                        self.write(rendezvous, {"port": keeper.port, "identity": identity})
                    self.wait(rendezvous.exists)
                    record = json.loads(rendezvous.read_text())
                    if record["identity"] != identity:
                        raise ValueError("Rendezvous belongs to another pipeline phase")
                    launch = [f"--nnodes={self.nodes}", f"--nproc_per_node={self.gpus}", f"--node_rank={self.rank}",
                              "--rdzv-backend=c10d", f"--rdzv-endpoint={master}:{record['port']}",
                              f"--rdzv-id={identity}", "--rdzv-conf=is_host=false,join_timeout=600,read_timeout=60"]
                command = [sys.executable, "-m", "torch.distributed.run", *launch, "--max_restarts=0", *arguments]
            if distributed or all_nodes or self.primary:
                print(json.dumps({"phase": phase, "replica": self.rank, "command": command}), flush=True)
                environment = dict(os.environ)
                if self.nodes > 1:
                    environment.update(NCCL_NET="Socket", NCCL_NET_PLUGIN="none", NCCL_IB_DISABLE="1")
                process = subprocess.Popen(command, env=environment, start_new_session=True)
                while process.poll() is None:
                    if list(self.root.glob("failure-*.json")):
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise RuntimeError("Another replica failed during this phase")
                    time.sleep(1)
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
            self.write(self.root / f"phase-{phase}-node-{self.rank}.json", {"complete": True})
            self.wait(lambda: len(list(self.root.glob(f"phase-{phase}-node-*.json"))) == self.nodes)
        except Exception as exc:
            self.write(self.root / f"failure-{self.rank}.json", {"phase": phase, "error": str(exc)})
            raise
        finally:
            del keeper


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--coordination-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A distributed Python command is required")
    SpatialCluster(args.coordination_dir, args.nodes, args.gpus_per_node).execute(command, distributed=True)


if __name__ == "__main__":
    main()
