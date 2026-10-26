"""Two replica supervisors rendezvous eight real CPU ranks without a port race."""
import os
from pathlib import Path
import subprocess
import sys


def test_existing_coordinator_runs_once_on_each_replica(tmp_path):
    root = Path(__file__).resolve().parents[2]
    child = tmp_path / "child.py"
    child.write_text("import os,sys\nfrom pathlib import Path\n"
                     "Path(sys.argv[1], 'child-'+os.environ['KUBEBRAIN_REPLICA']).touch()\n")
    supervisor = tmp_path / "supervisor.py"
    supervisor.write_text("import sys\nfrom scripts.train.spatial_cluster import SpatialCluster\n"
        "SpatialCluster(sys.argv[1],2,4).execute([sys.argv[2],sys.argv[3]],all_nodes=True)\n")
    processes = [subprocess.Popen([sys.executable, str(supervisor), str(tmp_path / "coordination"),
        str(child), str(tmp_path)], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root), KUBEBRAIN_REPLICA=str(rank), KUBEBRAIN_REPLICA_TOTAL="2"))
        for rank in range(2)]
    try:
        assert all(process.wait(timeout=20) == 0 for process in processes)
        assert {path.name for path in tmp_path.glob("child-*")} == {"child-0", "child-1"}
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_two_replicas_coordinate_eight_ranks(tmp_path):
    root = Path(__file__).resolve().parents[2]
    child = tmp_path / "collective.py"
    child.write_text("import os\nimport torch\nimport torch.distributed as dist\n"
                     "dist.init_process_group('gloo')\n"
                     "assert dist.get_world_size()==8\n"
                     "value=torch.tensor(dist.get_rank())\n"
                     "dist.all_reduce(value)\nassert int(value)==28\n"
                     "dist.destroy_process_group()\n")
    processes, logs = [], []
    for rank in range(2):
        log = (tmp_path / f"node-{rank}.log").open("w")
        logs.append(log)
        environment = dict(os.environ, KUBEBRAIN_REPLICA=str(rank), KUBEBRAIN_REPLICA_TOTAL="2",
                           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", POD_IP="127.0.0.1")
        processes.append(subprocess.Popen([sys.executable, "-m", "scripts.train.spatial_cluster",
            "--nodes", "2", "--gpus-per-node", "4", "--coordination-dir", str(tmp_path / "transport"),
            "--", str(child)], cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT))
    try:
        for rank, process in enumerate(processes):
            code = process.wait(timeout=120)
            assert code == 0, (tmp_path / f"node-{rank}.log").read_text()
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
