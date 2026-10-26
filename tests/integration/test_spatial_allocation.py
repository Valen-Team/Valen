"""Persistent replicas hand off models without releasing or double-counting GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scripts.train.monitor_spatial import inspect_job, inspect_registry
from scripts.train.schedule_spatial import retain_comparison_initializations, run_allocation_pool
from scripts.train.spatial_pool import configure_allocations, pipeline_state
from scripts.train.spatial_worker import atomic_json


ROOT = Path(__file__).resolve().parents[2]


def test_suspension_holds_training_evaluation_and_side_tasks_without_releasing_cards(tmp_path):
    jobs = [dict(name=task, kind="pipeline", task=task, size="2B", allocation="pool",
                 run_dir=str(tmp_path/task), log_dir=str(tmp_path/task/"logs"))
            for task in ("det", "seg", "gui")]
    for job in jobs:
        run = Path(job["run_dir"])
        run.mkdir()
        (run/"pipeline_complete.json").write_text("{}")
    side = dict(name="gui-diagnostic", task="gui", after="det", ready=True,
                run_dir=str(tmp_path/"diagnostic"), log_dir=str(tmp_path/"diagnostic-logs"),
                command=["python", "diagnostic.py"])
    preflight = dict(name="seg-preflight", task="seg", ready=True,
                     run_dir=str(tmp_path/"preflight"), log_dir=str(tmp_path/"preflight-logs"),
                     command=["python", "preflight.py"])
    allocation = dict(name="pool", gpus=8, nodes=2, job_id="retained-cards", initial_pipeline=None,
                      queue=str(tmp_path/"queue.json"), extra_tasks=[side], preflight_tasks=[preflight])
    state = dict(jobs=jobs, allocations=[allocation], examples_code_snapshot=str(tmp_path/"code"),
                 suspended_tasks=["seg", "gui"])
    args = (state, tmp_path/"data", tmp_path/"models", tmp_path/"code", lambda *args: True)
    configure_allocations(*args)
    tasks = json.loads((tmp_path/"queue.json").read_text())["tasks"]
    assert len(tasks) == 8
    assert all(task["ready"] == (task["task"] == "det") for task in tasks)
    assert allocation["job_id"] == "retained-cards" and allocation["gpus"] == 8
    state["suspended_tasks"] = []
    configure_allocations(*args)
    assert all(task["ready"] for task in json.loads((tmp_path/"queue.json").read_text())["tasks"])


def test_controller_adds_examples_runtime_when_training_snapshot_predates_it(tmp_path, monkeypatch):
    (tmp_path / "STOP_SCHEDULER").touch()
    copied = []
    def snapshot(root, destination):
        copied.append(destination)
        return "snapshot"
    monkeypatch.setattr("scripts.train.schedule_spatial.snapshot_code", snapshot)
    state = {}
    run_allocation_pool(None, state, tmp_path / "registry.json", tmp_path / "data", tmp_path / "models", tmp_path / "legacy-code")
    assert copied == [tmp_path / "code-examples"]
    assert state["examples_code_snapshot"] == str(copied[0])


def test_controller_does_not_race_a_pipeline_retaining_its_own_head(tmp_path, monkeypatch):
    jobs, calls = [], []
    for builtin in (False, True):
        reference = tmp_path / str(builtin)
        marker = reference / "warmup/training_complete.json"
        marker.parent.mkdir(parents=True)
        marker.write_text("{}")
        jobs.extend([dict(kind="pipeline", run_dir=str(reference), retain_warmup=builtin),
                     dict(kind="pipeline", variant="full", run_dir=str(reference) + "-full", baseline_run=str(reference))])
    monkeypatch.setattr("scripts.train.schedule_spatial.subprocess.run", lambda command, **kwargs: calls.append(command))
    retain_comparison_initializations({"jobs": jobs})
    assert len(calls) == 1 and calls[0][-1] == str(tmp_path / "False")


def test_ready_model_is_queued_until_its_allocation_receives_gpus(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"tasks": [{"name": "det", "ready": True}]}))
    job = dict(name="det", run_dir=str(tmp_path / "run"))
    allocation = dict(queue=str(queue), state="queued")
    assert pipeline_state(job, allocation) == "queued"
    allocation["state"] = "running"
    assert pipeline_state(job, allocation) == "queued_in_allocation"


def test_reasonseg_follows_an_existing_pipeline_in_the_same_allocation(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    job = dict(name="seg", kind="pipeline", task="seg", size="4B", allocation="pool",
        run_dir=str(run), log_dir=str(tmp_path / "logs"), reasonseg_after=True,
        reasonseg_code_snapshot=str(tmp_path / "new-code"))
    allocation = dict(name="pool", kind="allocation", size="4B", gpus=8, nodes=2,
        job_id="same-eight-gpus", log_dir=str(tmp_path / "pool"), queue=str(tmp_path / "queue.json"),
        initial_pipeline=None)
    state = dict(jobs=[job], allocations=[allocation])
    data = tmp_path / "data"
    configure_allocations(state, data, tmp_path / "models", tmp_path / "code", lambda *args: True)
    tasks = json.loads((tmp_path / "queue.json").read_text())["tasks"]
    assert tasks[0]["ready"] and not tasks[1]["ready"]
    (run / "pipeline_complete.json").write_text("{}")
    preflight = data / "eval/seg/reasonseg/manifests/preflight.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text("{}")
    configure_allocations(state, data, tmp_path / "models", tmp_path / "code", lambda *args: True)
    tasks = json.loads((tmp_path / "queue.json").read_text())["tasks"]
    assert tasks[1]["ready"] and tasks[1]["run_dir"] == str(run / "evaluation-reasonseg")
    assert len(state["allocations"]) == 1 and allocation["job_id"] == "same-eight-gpus"


def test_examples_include_initial_models_without_submitting_more_gpus(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    job = dict(name="gui", kind="pipeline", task="gui", size="2B", allocation="pool",
               run_dir=str(run), log_dir=str(tmp_path / "logs"))
    allocation = dict(name="pool", kind="allocation", size="2B", gpus=8, nodes=2,
        job_id="held-eight-gpus", log_dir=str(tmp_path / "pool"), queue=str(tmp_path / "queue.json"),
        initial_pipeline="gui")
    state = dict(jobs=[job], allocations=[allocation], examples_code_snapshot=str(tmp_path / "new-code"))
    arguments = (state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)
    configure_allocations(*arguments)
    task = json.loads((tmp_path / "queue.json").read_text())["tasks"][0]
    assert task["name"] == "gui-examples" and not task["ready"]
    (run / "pipeline_complete.json").write_text("{}")
    configure_allocations(*arguments)
    task = json.loads((tmp_path / "queue.json").read_text())["tasks"][0]
    assert task["ready"] and "evaluation.examples" in task["command"]
    assert len(state["allocations"]) == 1 and allocation["job_id"] == "held-eight-gpus"


def test_stopped_allocation_is_not_success_when_wrapper_exited_cleanly(tmp_path, monkeypatch):
    for rank in range(2):
        (tmp_path / f"status.node-{rank}.log").write_text("EXIT_CODE=0\n")
    monkeypatch.setattr("scripts.train.monitor_spatial.subprocess.run", lambda *args, **kwargs:
                        type("Query", (), {"stdout": ": Stopped", "stderr": ""})())
    job = dict(name="pool", kind="allocation", job_id="stopped-job", nodes=2, gpus=8,
               log_dir=str(tmp_path), outputs=[])
    assert inspect_job(job, "unused")["state"] == "stopped"


def test_diagnostics_follow_the_active_model_before_its_next_training(tmp_path):
    jobs = [dict(name=task, kind="pipeline", task=task, size="0.8B", allocation="held",
                 run_dir=str(tmp_path / task), log_dir=str(tmp_path / task / "logs")) for task in ("det", "seg")]
    auxiliary = dict(name="gui-diagnosis", after="det", task="gui", run_dir=str(tmp_path / "diagnosis"),
                     log_dir=str(tmp_path / "diagnosis/logs"), command=["python", "-m", "evaluation.gui.diagnose"],
                     code_root=str(tmp_path / "diagnostic-code"), ready=True)
    allocation = dict(name="held", size="0.8B", gpus=8, nodes=2, job_id="same-eight-gpus",
                      initial_pipeline=None, queue=str(tmp_path / "queue.json"), extra_tasks=[auxiliary])
    state = dict(jobs=jobs, allocations=[allocation])
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)
    tasks = json.loads((tmp_path / "queue.json").read_text())["tasks"]
    assert [r["name"] for r in tasks] == ["det", "gui-diagnosis", "seg"]
    assert not tasks[1]["ready"] and tasks[1]["code_root"] == auxiliary["code_root"]
    assert allocation["job_id"] == "same-eight-gpus" and len(state["jobs"]) == 2
    (tmp_path / "det").mkdir()
    (tmp_path / "det/pipeline_complete.json").write_text("{}")
    allocation["initial_pipeline"] = "det"
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)
    tasks = json.loads((tmp_path / "queue.json").read_text())["tasks"]
    assert [r["name"] for r in tasks] == ["gui-diagnosis", "seg"]
    assert tasks[0]["ready"]


def test_mixed_sizes_append_unready_comparisons_after_existing_examples(tmp_path):
    jobs = []
    for name, task, size in (("gui4", "gui", "4B"), ("seg4", "seg", "4B"), ("det08full", "det", "0.8B")):
        run = tmp_path / name
        run.mkdir()
        jobs.append(dict(name=name, kind="pipeline", task=task, size=size, allocation="held",
                         run_dir=str(run), log_dir=str(run / "logs")))
    baseline = tmp_path / "det08"
    jobs[-1].update(variant="full", after_queue=True, baseline_run=str(baseline),
                    code_snapshot=str(tmp_path / "pinned-code"), requires_files=[str(baseline / "pipeline_complete.json")])
    (tmp_path / "gui4/pipeline_complete.json").write_text("{}")
    allocation = dict(name="held", size="4B", gpus=8, nodes=2, job_id="unchanged",
                      initial_pipeline=None, queue=str(tmp_path / "queue.json"))
    state = dict(jobs=jobs, allocations=[allocation], examples_code_snapshot=str(tmp_path / "examples-code"))
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)
    tasks = json.loads((tmp_path / "queue.json").read_text())["tasks"]
    assert [item["name"] for item in tasks] == ["gui4", "seg4", "gui4-examples", "seg4-examples",
                                                "det08full", "det08full-examples"]
    assert tasks[2]["ready"] and not tasks[4]["ready"]
    command = tasks[4]["command"]
    assert command[command.index("--model-path") + 1] == str(tmp_path / "models/Qwen3.5-0.8B")
    assert command[command.index("--baseline-run") + 1] == str(baseline)
    assert command[command.index("--gpus-per-node") + 1] == "4"
    assert tasks[4]["code_root"] == str(tmp_path / "pinned-code")
    assert len(state["allocations"]) == 1 and allocation["job_id"] == "unchanged"


def test_mixed_size_append_preserves_running_replicas(tmp_path):
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "pipeline_complete.json").write_text("{}")
    run = tmp_path / "seg4"
    run.mkdir()
    job = dict(name="seg4", kind="pipeline", task="seg", size="4B", allocation="held",
               run_dir=str(run), log_dir=str(run / "logs"))
    allocation = dict(name="held", size="4B", gpus=8, nodes=2, job_id="same-eight-gpus",
                      initial_pipeline=None, queue=str(tmp_path / "queue.json"))
    state = dict(jobs=[job], allocations=[allocation])
    child = tmp_path / "finish.py"
    child.write_text("import os,sys,time\nfrom pathlib import Path\n"
                     "p=Path(sys.argv[1]);r=os.environ['KUBEBRAIN_REPLICA']\n"
                     "(p/('started-'+r)).write_text(str(os.getppid()))\n"
                     "while not (p/'continue').is_file(): time.sleep(.02)\n"
                     "if r=='0': (p/'pipeline_complete.json').write_text('{}')\n")
    fake = tmp_path / "nvidia-smi"
    fake.write_text('#!/bin/sh\nprintf "GPU-%s-0\\nGPU-%s-1\\nGPU-%s-2\\nGPU-%s-3\\n" "$KUBEBRAIN_REPLICA" "$KUBEBRAIN_REPLICA" "$KUBEBRAIN_REPLICA" "$KUBEBRAIN_REPLICA"\n')
    fake.chmod(0o755)
    def refresh():
        configure_allocations(state, tmp_path / "data", tmp_path / "models", ROOT, lambda *args: True)
        config = json.loads((tmp_path / "queue.json").read_text())
        for task in config["tasks"]:
            task["command"] = [sys.executable, str(child), task["run_dir"]]
        atomic_json(tmp_path / "queue.json", config)
    refresh()
    processes = []
    def workers():
        return [json.loads(path.read_text()) for path in tmp_path.glob("worker.node-*.json")]
    try:
        for rank in range(2):
            environment = dict(os.environ, KUBEBRAIN_REPLICA=str(rank), KUBEBRAIN_REPLICA_TOTAL="2",
                               PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
            processes.append(subprocess.Popen([sys.executable, str(ROOT / "scripts/train/spatial_worker.py"),
                "--queue", str(tmp_path / "queue.json"), "--interval", ".02"], env=environment,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        until(lambda: len(list(run.glob("started-*"))) == 2, processes)
        before = {row["replica"]: (row["pid"], row["gpu_uuids"]) for row in workers()}
        assert len({uuid for _, uuids in before.values() for uuid in uuids}) == 8
        extra = tmp_path / "det08full"
        extra.mkdir()
        state["jobs"].append(dict(job, name="det08full", task="det", size="0.8B", variant="full",
            after_queue=True, baseline_run=str(initial), run_dir=str(extra), log_dir=str(extra / "logs")))
        refresh()
        (run / "continue").touch()
        until(lambda: len(list(extra.glob("started-*"))) == 2, processes)
        assert (run / "pipeline_complete.json").is_file()
        (extra / "continue").touch()
        assert all(process.wait(timeout=12) == 0 for process in processes)
        assert {row["replica"]: (row["pid"], row["gpu_uuids"]) for row in workers()} == before
        assert (extra / "pipeline_complete.json").is_file()
    finally:
        (tmp_path / "STOP").touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_ready_task_runs_while_an_earlier_task_waits_for_data(tmp_path):
    child = tmp_path / "finish.py"
    child.write_text("import os,sys,time\nfrom pathlib import Path\n"
        "p=Path(sys.argv[1]);r=os.environ['KUBEBRAIN_REPLICA']\n"
        "(p/('started-'+r)).touch()\n"
        "while len(list(p.glob('started-*')))<2: time.sleep(.02)\n"
        "if r=='0': (p/'pipeline_complete.json').write_text('{}')\n")
    tasks = []
    for name in ("unready", "ready"):
        run = tmp_path / name
        run.mkdir()
        tasks.append(dict(name=name, run_dir=str(run), log_dir=str(run / "logs"), ready=name == "ready",
                          command=[sys.executable, str(child), str(run)]))
    queue = tmp_path / "queue.json"
    config = dict(name="pool", nodes=2, gpus_per_node=4, code_root=str(ROOT), initial=None, tasks=tasks)
    atomic_json(queue, config)
    processes = []
    try:
        for rank in range(2):
            environment = dict(os.environ, KUBEBRAIN_REPLICA=str(rank), KUBEBRAIN_REPLICA_TOTAL="2")
            processes.append(subprocess.Popen([sys.executable, str(ROOT / "scripts/train/spatial_worker.py"),
                "--queue", str(queue), "--interval", ".02"], env=environment,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        until(lambda: (tmp_path / "ready/pipeline_complete.json").exists(), processes)
        assert not (tmp_path / "unready/pipeline_complete.json").exists()
        tasks[0]["ready"] = True
        atomic_json(queue, config)
        assert all(process.wait(timeout=12) == 0 for process in processes)
        assert (tmp_path / "unready/pipeline_complete.json").is_file()
    finally:
        (tmp_path / "STOP").touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def until(predicate, processes=(), timeout=12):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert all(process.poll() is None for process in processes), "A replica exited before handoff"
        if time.monotonic() > deadline:
            pytest.fail("Replicas did not reach the expected state")
        time.sleep(.02)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("has_initial", [False, True])
def test_replicas_keep_the_same_allocation_while_waiting_and_handing_off(tmp_path, fail, has_initial):
    child = tmp_path / "child.py"
    child.write_text("import json,os,sys,time\nfrom pathlib import Path\n"
        "p=Path(sys.argv[1]); rank=os.environ['KUBEBRAIN_REPLICA']\n"
        "(p/('started-'+rank)).write_text(str(os.getppid()))\n"
        "while len(list(p.glob('started-*')))<2: time.sleep(.02)\n"
        "if sys.argv[2]=='fail' and rank=='1': sys.exit(7)\n"
        "if sys.argv[2]=='fail': time.sleep(30)\n"
        "if rank=='0': (p/'pipeline_complete.json').write_text('{}')\n")
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "pipeline_complete.json").write_text("{}")
    tasks = []
    for index in range(2):
        run = tmp_path / ("task" + str(index))
        run.mkdir()
        tasks.append({"name": run.name, "run_dir": str(run), "log_dir": str(run / "logs"),
                      "command": [sys.executable, str(child), str(run), "fail" if fail else "ok"],
                      "ready": False})
    queue = tmp_path / "queue.json"
    config = {"name": "existing-allocation", "nodes": 2, "gpus_per_node": 4,
              "code_root": str(ROOT), "initial": {"name": "initial", "run_dir": str(initial)} if has_initial else None,
              "tasks": tasks}
    atomic_json(queue, config)
    fake = tmp_path / "nvidia-smi"
    fake.write_text('#!/bin/sh\nprintf "GPU-%s-0\\nGPU-%s-1\\nGPU-%s-2\\nGPU-%s-3\\n" "$KUBEBRAIN_REPLICA" "$KUBEBRAIN_REPLICA" "$KUBEBRAIN_REPLICA" "$KUBEBRAIN_REPLICA"\n')
    fake.chmod(0o755)
    processes, logs = [], []
    try:
        for rank in range(2):
            log = (tmp_path / f"node-{rank}.log").open("w")
            logs.append(log)
            env = dict(os.environ, KUBEBRAIN_REPLICA=str(rank), KUBEBRAIN_REPLICA_TOTAL="2",
                       PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
            processes.append(subprocess.Popen([sys.executable, str(ROOT / "scripts/train/spatial_worker.py"),
                "--queue", str(queue), "--interval", ".02"], env=env, stdout=log, stderr=subprocess.STDOUT))
        def states():
            return [json.loads(p.read_text()) for p in tmp_path.glob("worker.node-*.json")]
        until(lambda: len(states()) == 2 and all(row["state"] == "waiting_for_data" for row in states()), processes)
        assert not list((tmp_path / "task0").glob("started-*"))
        workers = {row["replica"]: (row["pid"], row["gpu_uuids"]) for row in states()}
        assert len({uuid for _, uuids in workers.values() for uuid in uuids}) == 8
        tasks[0]["ready"] = True
        atomic_json(queue, config)
        if fail:
            until(lambda: len(states()) == 2 and all(row["state"] == "blocked" for row in states()), processes)
            assert not list((tmp_path / "task1").glob("started-*"))
            (tmp_path / "STOP").touch()
        else:
            until(lambda: len(states()) == 2 and all(row["task"] == "task1" and
                  row["state"] == "waiting_for_data" for row in states()), processes)
            code = tmp_path / "new-code"
            code.mkdir()
            (code / "queued_child.py").write_text(child.read_text())
            extra = tmp_path / "task2"
            extra.mkdir()
            tasks.append({"name": "task2", "run_dir": str(extra), "log_dir": str(extra / "logs"),
                          "code_root": str(code), "ready": True,
                          "command": [sys.executable, "-m", "queued_child", str(extra), "ok"]})
            tasks[1]["ready"] = True
            atomic_json(queue, config)
        for process in processes:
            assert process.wait(timeout=12) == 0
        assert {row["replica"]: (row["pid"], row["gpu_uuids"]) for row in states()} == workers
        if not fail:
            assert all(row["state"] == "complete" for row in states())
            for task in tasks:
                assert (Path(task["run_dir"]) / "pipeline_complete.json").exists()
                assert {int(path.read_text()) for path in Path(task["run_dir"]).glob("started-*")} == {
                    process.pid for process in processes}
    finally:
        (tmp_path / "STOP").touch()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()


def test_models_share_and_expand_allocations_without_duplicate_runs(tmp_path, monkeypatch):
    jobs = []
    for size in ("0.8B", "2B", "4B", "9B"):
        for task in ("det", "seg", "gui"):
            name = task + "-" + size
            run = tmp_path / name
            run.mkdir()
            job = {"name": name, "kind": "pipeline", "task": task, "size": size, "gpus": 8,
                   "nodes": 2, "log_dir": str(run / "logs"), "run_dir": str(run), "outputs": [str(run)]}
            if task == "gui":
                job["job_id"] = "job-" + size
            jobs.append(job)
    state = {"jobs": jobs}
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code",
                          lambda task, size: task == "det")
    monkeypatch.setattr("scripts.train.monitor_spatial.subprocess.run", lambda *args, **kwargs:
                        type("Query", (), {"stdout": ": Running", "stderr": ""})())
    report = inspect_registry(state, "unused")
    assert len(report["allocations"]) == 4
    assert report["reserved_gpus"] == report["running_gpus"] == 32
    assert sum(row["state"] == "running" for row in report["jobs"]) == 4
    assert sum(row["state"] == "waiting_for_data" for row in report["jobs"]) == 4
    assert sum(row["state"] == "queued_in_allocation" for row in report["jobs"]) == 4
    owner = next(job for job in jobs if job["task"] == "gui" and job["size"] == "0.8B")
    (Path(owner["run_dir"]) / "pipeline_complete.json").write_text("{}")
    after = inspect_registry(state, "unused")
    assert next(row for row in after["jobs"] if row["name"] == owner["name"])["state"] == "succeeded"
    assert after["reserved_gpus"] == 32
    assert all(Path(row["queue"]).is_file() for row in state["allocations"])
    for size in ("0.8B", "2B", "4B", "9B"):
        det = next(job for job in jobs if job["task"] == "det" and job["size"] == size)
        name = det["name"] + "-pool"
        allocation = {"name": name, "kind": "allocation", "job_id": "detjob-" + size,
                      "gpus": 8, "nodes": 2, "size": size, "initial_pipeline": None,
                      "log_dir": det["log_dir"], "queue": str(Path(det["log_dir"]) / "worker/queue.json")}
        state["allocations"].append(allocation)
        det["allocation"] = name
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: False)
    expanded = inspect_registry(state, "unused")
    assert len(expanded["allocations"]) == 8
    assert expanded["reserved_gpus"] == expanded["running_gpus"] == 64
    assigned = []
    for allocation in state["allocations"]:
        config = json.loads(Path(allocation["queue"]).read_text())
        if config["initial"]:
            assigned.append(config["initial"]["name"])
            assert [item["task"] for item in config["tasks"]] == ["seg"]
        else:
            assert [item["task"] for item in config["tasks"]] == ["det"]
        assigned.extend(item["name"] for item in config["tasks"])
    assert len(assigned) == len(set(assigned)) == 12
    for base in list(jobs):
        if base["size"] not in ("0.8B", "2B"):
            continue
        run = tmp_path / (base["name"] + "-full")
        run.mkdir()
        jobs.append(dict(base, name=base["name"] + "-full", run_dir=str(run), outputs=[str(run)],
                         log_dir=str(run / "logs"), job_id=None, variant="full", baseline_run=base["run_dir"],
                         code_snapshot=str(tmp_path / "full-code"), requires_files=[str(run / "dependency")]))
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)
    variants = inspect_registry(state, "unused")
    assert variants["reserved_gpus"] == 64
    assert sum(job["state"] == "waiting_for_dependencies" for job in variants["jobs"]) == 6
    for allocation in state["allocations"]:
        queued = json.loads(Path(allocation["queue"]).read_text())["tasks"]
        if allocation["initial_pipeline"] and allocation["size"] in ("0.8B", "2B"):
            assert queued[0]["name"].endswith("-full") and queued[0]["task"] == "gui"
        for task in queued:
            if task["name"].endswith("-full"):
                assert not task["ready"]
                assert task["code_root"] == str(tmp_path / "full-code")
                assert task["command"][-4:-2] == ["--finetuning-type", "full"]
    state["gpu_limit"] = 56
    with pytest.raises(ValueError, match="GPU budget"):
        configure_allocations(state, tmp_path, tmp_path, tmp_path, lambda *args: True)


def test_cannot_adopt_two_independent_jobs_as_one_allocation(tmp_path):
    jobs = [{"name": task, "kind": "pipeline", "task": task, "size": "0.8B", "job_id": task}
            for task in ("det", "seg", "gui")]
    with pytest.raises(ValueError, match="one existing allocation"):
        configure_allocations({"jobs": jobs}, tmp_path, tmp_path, tmp_path, lambda *args: True)


@pytest.mark.parametrize("marker", ["# Retain spatial allocations for the queued models.",
                                  "# Follow replacement queues while retaining the same scheduler allocation."])
def test_running_entrypoint_reads_appended_handoff_without_restarting_child(tmp_path, marker):
    source = (ROOT / "scripts/train/experiment_entry.sh").read_text()
    prefix, tail = source.split("\n" + marker, 1)
    scripts = tmp_path / "scripts/train"
    scripts.mkdir(parents=True)
    entry = scripts / "experiment_entry.sh"
    entry.write_text(prefix)
    proof = tmp_path / "handoff.json"
    (scripts / "spatial_worker.py").write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "q=Path(sys.argv[sys.argv.index('--queue')+1])\n"
        "Path(json.loads(q.read_text())['proof']).write_text(json.dumps({'handoff':True}))\n")
    fake = tmp_path / "nvidia-smi"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    child = tmp_path / "initial.py"
    child.write_text("import os,sys,time\nfrom pathlib import Path\n"
                     "p=Path(sys.argv[1]); (p/'started').write_text(str(os.getpid()))\n"
                     "while not (p/'continue').exists(): time.sleep(.02)\n")
    log_dir = tmp_path / "logs"
    (log_dir / "allocation").mkdir(parents=True)
    queue = log_dir / "allocation/queue.json"
    queue.write_text(json.dumps({"proof": str(proof)}))
    if marker.startswith("# Follow"):
        (log_dir / "handoff").mkdir()
        replacement = log_dir / "handoff/queue.json"
        queue.replace(replacement)
        (log_dir / "handoff/next_queue").write_text(str(replacement) + "\n")
    env = dict(os.environ, VALEN_CODE_ROOT=str(ROOT), VALEN_PYTHON=sys.executable,
               KUBEBRAIN_REPLICA_TOTAL="1", PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    process = subprocess.Popen(["bash", str(entry), str(log_dir), str(child), str(tmp_path)],
                               env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        until(lambda: (tmp_path / "started").exists(), [process])
        original_pid = (tmp_path / "started").read_text()
        with entry.open("a") as stream:
            stream.write("\n" + marker + tail)
        assert (tmp_path / "started").read_text() == original_pid
        (tmp_path / "continue").touch()
        assert process.wait(timeout=12) == 0
        assert json.loads(proof.read_text()) == {"handoff": True}
        assert (log_dir / "status.log").read_text() == "EXIT_CODE=0\n"
    finally:
        (tmp_path / "continue").touch()
        if process.poll() is None:
            process.kill()
            process.wait()
