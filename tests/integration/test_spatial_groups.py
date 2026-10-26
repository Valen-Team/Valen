"""Combine existing leases for sixteen-rank Det without double-counting GPUs."""
import json
import os
from pathlib import Path

import pytest

from scripts.train.monitor_spatial import inspect_registry
from scripts.train.spatial_pool import configure_allocations
from scripts.train.spatial_worker import AllocationWorker
from scripts.train.summarize_spatial import stage_info


def grouped_state(tmp_path):
    members = [dict(name=f"physical-{i}", kind="allocation", job_id=f"job-{i}", gpus=8, nodes=2,
                    initial_pipeline=None, queue=str(tmp_path / f"post-{i}/queue.json"),
                    log_dir=str(tmp_path / f"logs-{i}"), execution_group="det16",
                    hostnames=[f"host-{2*i}", f"host-{2*i+1}"]) for i in range(2)]
    group = dict(name="det16", gpus=16, nodes=4, initial_pipeline=None,
                 queue=str(tmp_path / "group/queue.json"), log_dir=str(tmp_path / "group-logs"),
                 members=[row["name"] for row in members], node_rank_map={f"host-{i}": i for i in range(4)})
    job = dict(name="det", kind="pipeline", task="det", size="0.8B", gpus=16, allocation="det16",
               run_dir=str(tmp_path / "run"), log_dir=str(tmp_path / "run-logs"))
    return dict(jobs=[job], allocations=members, execution_groups=[group], gpu_limit=64)


def test_group_commands_use_four_nodes_and_keep_physical_leases(tmp_path):
    state = grouped_state(tmp_path)
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)
    assert sum(row["gpus"] for row in state["allocations"]) == 16
    assert [row["job_id"] for row in state["allocations"]] == ["job-0", "job-1"]
    queue = json.loads((tmp_path / "group/queue.json").read_text())
    command = queue["tasks"][0]["command"]
    assert command[command.index("--nodes") + 1] == "4"
    assert command[command.index("--gpus-per-node") + 1] == "4"
    assert queue["node_rank_map"] == {f"host-{i}": i for i in range(4)}
    for member in state["allocations"]:
        assert json.loads((tmp_path / member["queue"]).read_text())["tasks"] == []


def test_gui_revision_uses_its_own_corpus_for_training_and_examples(tmp_path):
    state = grouped_state(tmp_path)
    job = state['jobs'][0]
    job.update(task='gui', size='2B', training_world_size=64,
               data_root=str(tmp_path/'gui-revision'), export_dir=str(tmp_path/'versioned-model'))
    state['examples_code_snapshot'] = str(tmp_path/'code')
    configure_allocations(state,tmp_path/'legacy-data',tmp_path/'models',tmp_path/'code',lambda *args:True)
    tasks=json.loads((tmp_path/'group/queue.json').read_text())['tasks']
    training=tasks[0]['command'];examples=tasks[1]['command']
    assert training[training.index('--data-root')+1]==job['data_root']
    assert training[training.index('--training-world-size')+1]=='64'
    assert training[training.index('--export-dir')+1]==job['export_dir']
    assert examples[examples.index('--data-root')+1]==str(tmp_path/'gui-revision/eval/gui')


def test_gui_lora_64_scales_schedule_without_changing_template():
    from scripts.train.run_spatial_pipeline import lora_resource_config
    config=dict(required_world_size=8,gradient_accumulation=4,epochs=1,save_every=125,
                head_lr=2e-4,task='gui',max_pixels=1048576)
    changed=lora_resource_config(config,64)
    assert changed['required_world_size']==64 and changed['gradient_accumulation']==2
    assert changed['lr_warmup_steps']==25 and changed['save_every']==31
    assert changed['head_lr']==2e-4 and changed['max_pixels']==1048576
    assert config['required_world_size']==8 and config['save_every']==125


def test_training_only_queue_defers_evaluation_and_examples_before_gpu_release(tmp_path):
    state=grouped_state(tmp_path)
    state['jobs'][0].update(task='gui',size='2B',defer_evaluation=True)
    state['examples_code_snapshot']=str(tmp_path/'code')
    configure_allocations(state,tmp_path/'data',tmp_path/'models',tmp_path/'code',lambda *args:True)
    tasks=json.loads((tmp_path/'group/queue.json').read_text())['tasks']
    assert len(tasks)==1 and '--defer-evaluation' in tasks[0]['command']


def test_sixty_four_gpu_group_retains_paused_models_without_running_them(tmp_path, monkeypatch):
    state = grouped_state(tmp_path)
    state["allocations"] = [dict(name=f"physical-{i}", kind="allocation", job_id=f"job-{i}",
        gpus=4, nodes=1, initial_pipeline=None, execution_group="det64", hostnames=[f"host-{i}"],
        queue=str(tmp_path/f"post-{i}/queue.json"), log_dir=str(tmp_path/f"logs-{i}")) for i in range(16)]
    group = state["execution_groups"][0]
    group.update(name="det64", gpus=64, nodes=16, members=[a["name"] for a in state["allocations"]],
                 node_rank_map={f"host-{i}":i for i in range(16)})
    job = state["jobs"][0]
    job.update(size="2B", allocation="det64", gpus=64, training_world_size=64,
               variant="full", baseline_run=str(tmp_path/"baseline"))
    state["jobs"].append(dict(job, name="paused4", size="4B", gpus=16,
                              run_dir=str(tmp_path/"paused4"), paused=True))
    configure_allocations(state, tmp_path/"data", tmp_path/"models", tmp_path/"code", lambda *args: True)
    queue = json.loads((tmp_path/"group/queue.json").read_text())
    assert queue["nodes"] == 16 and queue["gpus_per_node"] == 4
    tasks = {task["name"]:task for task in queue["tasks"]}
    assert tasks["det"]["ready"] and not tasks["paused4"]["ready"]
    assert "--training-world-size" in tasks["det"]["command"]
    monkeypatch.setenv("KUBEBRAIN_REPLICA", "0")
    monkeypatch.setenv("KUBEBRAIN_REPLICA_TOTAL", "1")
    monkeypatch.setattr("scripts.train.spatial_worker.socket.gethostname", lambda:"host-15")
    monkeypatch.setattr("scripts.train.spatial_worker.subprocess.check_output", lambda *args,**kwargs:"a\nb\nc\nd\n")
    worker = AllocationWorker(tmp_path/"group/queue.json")
    assert worker.rank == 15 and worker.nodes == 16
    assert sum(a["gpus"] for a in state["allocations"]) == 64


def test_rlcd_queue_uses_its_own_package_and_waits_for_full_comparison(tmp_path):
    state=grouped_state(tmp_path)
    baseline=state['jobs'][0]
    gate=tmp_path/'accepted-comparison.json'
    job=dict(baseline,name='rlcd',variant='rlcd',size='2B',run_dir=str(tmp_path/'rlcd'),
             log_dir=str(tmp_path/'rlcd-logs'),baseline_run=baseline['run_dir'],
             full_baseline_run=str(tmp_path/'full'),subset_dir=str(tmp_path/'subset'),
             evaluation_gate=str(gate),requires_files=[str(gate)])
    state['jobs'].append(job)
    state['examples_code_snapshot']=str(tmp_path/'code')
    args=(state,tmp_path/'data',tmp_path/'models',tmp_path/'code',lambda *args:True)
    configure_allocations(*args)
    tasks=json.loads((tmp_path/'group/queue.json').read_text())['tasks']
    task=next(t for t in tasks if t['name']=='rlcd')
    assert not task['ready'] and 'scripts.train.run_det_rlcd_pipeline' in task['command']
    assert task['command'][task['command'].index('--export-dir')+1].endswith('Valen-Decision-Det-2B-Full-RLCD')
    gate.write_text('{}');configure_allocations(*args)
    task=next(t for t in json.loads((tmp_path/'group/queue.json').read_text())['tasks'] if t['name']=='rlcd')
    assert task['ready']


def test_physical_lease_cannot_appear_in_two_execution_groups(tmp_path):
    state = grouped_state(tmp_path)
    state["execution_groups"].append(dict(state["execution_groups"][0], name="duplicate"))
    with pytest.raises(ValueError, match="disjoint"):
        configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)


def test_four_independent_four_gpu_leases_can_form_one_det_group(tmp_path):
    state = grouped_state(tmp_path)
    state["allocations"] = [dict(name=f"single-{i}", kind="allocation", job_id=f"single-job-{i}",
        gpus=4, nodes=1, initial_pipeline=None, execution_group="det16", hostnames=[f"host-{i}"],
        queue=str(tmp_path / f"single-{i}/queue.json"), log_dir=str(tmp_path / f"single-logs-{i}")) for i in range(4)]
    state["execution_groups"][0]["members"] = [a["name"] for a in state["allocations"]]
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)
    group = json.loads((tmp_path / "group/queue.json").read_text())
    assert group["nodes"] == 4 and group["gpus_per_node"] == 4
    assert sum(a["gpus"] for a in state["allocations"]) == 16
    assert all(json.loads(Path(a["queue"]).read_text())["tasks"] == [] for a in state["allocations"])


def test_full_comparison_and_its_examples_can_precede_nine_b_on_the_same_group(tmp_path):
    state = grouped_state(tmp_path)
    state["examples_code_snapshot"] = str(tmp_path / "code")
    reference = state["jobs"][0]
    for name, size, variant in (("full4", "4B", "full"), ("nine", "9B", "lora")):
        state["jobs"].append(dict(reference, name=name, size=size, variant=variant, after_queue=True,
            baseline_run=reference["run_dir"], run_dir=str(tmp_path / name), log_dir=str(tmp_path / (name + "-logs"))))
    order = ["det", "det-examples", "full4", "full4-examples", "nine", "nine-examples"]
    state["execution_groups"][0]["task_order"] = order
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)
    queue = json.loads((tmp_path / "group/queue.json").read_text())
    assert [task["name"] for task in queue["tasks"]] == order
    state["execution_groups"][0]["task_order"] = order[:-1]
    with pytest.raises(ValueError, match="every queued task"):
        configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)


def test_grouped_worker_remaps_only_child_environment(tmp_path, monkeypatch):
    state = grouped_state(tmp_path)
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)
    monkeypatch.setenv("KUBEBRAIN_REPLICA", "1")
    monkeypatch.setenv("KUBEBRAIN_REPLICA_TOTAL", "2")
    monkeypatch.setattr("scripts.train.spatial_worker.socket.gethostname", lambda: "host-3")
    monkeypatch.setattr("scripts.train.spatial_worker.subprocess.check_output", lambda *a, **k: "gpu1\ngpu2\ngpu3\ngpu4\n")
    worker = AllocationWorker(tmp_path / "group/queue.json")
    assert worker.rank == 3 and worker.nodes == 4
    received = {}
    class Done:
        returncode = 0
        def poll(self):
            return 0
    def launch(command, **kwargs):
        received.update(kwargs["env"])
        return Done()
    monkeypatch.setattr("scripts.train.spatial_worker.subprocess.Popen", launch)
    monkeypatch.setattr(worker, "wait", lambda *args: True)
    monkeypatch.setattr(worker, "records", lambda *args: [{"returncode": 0}] * 4)
    run = tmp_path / "run"
    run.mkdir()
    (run / "pipeline_complete.json").write_text("{}")
    task = json.loads((tmp_path / "group/queue.json").read_text())["tasks"][0]
    assert worker.run_task(task)
    assert received["KUBEBRAIN_REPLICA"] == "3" and received["KUBEBRAIN_REPLICA_TOTAL"] == "4"
    assert os.environ["KUBEBRAIN_REPLICA"] == "1" and os.environ["KUBEBRAIN_REPLICA_TOTAL"] == "2"


def test_monitor_counts_group_members_once(tmp_path, monkeypatch):
    state = grouped_state(tmp_path)
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)
    for rank in range(4):
        (tmp_path / f"group/worker.node-{rank}.json").write_text(json.dumps(
            {"hostname": f"host-{rank}", "replica": rank, "task": "det", "state": "running"}))
    monkeypatch.setattr("scripts.train.monitor_spatial.inspect_job", lambda job, scheduler:
        {"name": job["name"], "job_id": job.get("job_id"), "gpus": job.get("gpus", 8),
         "state": "running" if job.get("job_id") else "not_submitted"})
    report = inspect_registry(state, "unused")
    assert report["reserved_gpus"] == report["running_gpus"] == 16
    assert report["jobs"][0]["state"] == "running"
    assert report["jobs"][0]["job_ids"] == ["job-0", "job-1"]
    assert len(report["execution_groups"][0]["workers"]) == 4
    assert [len(row["workers"]) for row in report["allocations"]] == [2, 2]


def test_sixteen_rank_completion_requires_sixteen_rank_evidence(tmp_path):
    stage = tmp_path / "warmup"
    stage.mkdir()
    (stage / "run_manifest.json").write_text(json.dumps({"world_size": 16, "gpu_names": ["H200"] * 16, "config": {}}))
    (stage / "training_complete.json").write_text(json.dumps({"stop_reason": "epochs_complete", "epoch": 1}))
    (stage / "metrics.jsonl").write_text(json.dumps({"seen": 100000, "total_records": 100000, "elapsed_seconds": 1}) + "\n")
    assert stage_info(tmp_path, "warmup", 16)["complete"]
    assert not stage_info(tmp_path, "warmup", 8)["complete"]


def test_completed_group_survives_release_of_a_physical_member(tmp_path, monkeypatch):
    state = grouped_state(tmp_path)
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)
    monkeypatch.setattr("scripts.train.monitor_spatial.inspect_job", lambda job, scheduler:
        {"name": job["name"], "job_id": job.get("job_id"), "gpus": job.get("gpus", 8),
         "state": "succeeded" if job.get("job_id") == "job-0" else "running"})
    run = tmp_path / "run"
    (run / "examples").mkdir(parents=True)
    (run / "pipeline_complete.json").write_text("{}")
    (run / "examples/pipeline_complete.json").write_text("{}")
    report = inspect_registry(state, "unused")
    assert report["execution_groups"][0]["state"] == "succeeded"
    assert report["jobs"][0]["state"] == "succeeded"
    assert report["running_gpus"] == report["reserved_gpus"] == 8
    (run / "examples/pipeline_complete.json").unlink()
    assert inspect_registry(state, "unused")["execution_groups"][0]["state"] == "failed"
