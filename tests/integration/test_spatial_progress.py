"""Progress checkboxes require completed epochs and distinguish train/eval media."""
import json

from scripts.train.spatial_progress import data_progress, stage_progress, write_progress
from valen.training.checkpoint import file_sha256


def api_progress_case(tmp_path):
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    root=tmp_path/'api';run=root/'full';report_path=root/'reports/comparison.json'
    sources={name:dict(full_samples=count,selected_samples=count,records_sha256=name)
             for name,count in [('refcoco_testA',2),('refcoco_testB',1)]}
    put(run/'request-config.json',dict(model='jev-1.14.0',benchmarks=sources))
    put(run/'metrics.json',dict(model='jev-1.14.0',complete_full_benchmarks=True,
        benchmarks={name:dict(samples=row['full_samples']) for name,row in sources.items()}))
    report=dict(model='jev-1.14.0',complete_full_evaluation=False,
        request_config_sha256=file_sha256(run/'request-config.json'),jev_benchmarks={},prediction_verification={})
    for name,row in sources.items():
        report['jev_benchmarks'][name]=dict(samples=row['full_samples'],full_samples=row['full_samples'],
            complete_split=True,data_sha256=name,accuracy=.75)
        report['prediction_verification'][name]=dict(samples=row['full_samples'],
            all_expected_ids_verified=True,original_box_iou_recomputed=True)
    put(report_path,report)
    state=dict(jobs=[],jev_api_evaluation=dict(model='jev-1.14.0',benchmarks=list(sources),samples=3,
        plan=str(root/'plan.json'),output_dir=str(run)))
    output=tmp_path/'progress.md'
    return (state,tmp_path/'registry.json',tmp_path/'data',tmp_path/'models',output),report,report_path,put


def test_api_split_checkboxes_require_complete_ids_original_iou_and_source_identity(tmp_path):
    args,report,path,put=api_progress_case(tmp_path)
    name='refcoco_testA';line='[x] 已完成：refcoco_testA：2条，Acc@0.5 75.00%'
    write_progress(*args)
    assert line in args[-1].read_text()
    for key in ('all_expected_ids_verified','original_box_iou_recomputed'):
        report['prediction_verification'][name][key]=False
        put(path,report);write_progress(*args)
        assert line not in args[-1].read_text()
        report['prediction_verification'][name][key]=True
    report['jev_benchmarks'][name]['data_sha256']='different-source'
    put(path,report);write_progress(*args)
    assert line not in args[-1].read_text()
    report['jev_benchmarks'][name]['data_sha256']=name
    report['jev_benchmarks'][name]['samples']=1
    put(path,report);write_progress(*args)
    assert line not in args[-1].read_text()


def test_full_api_completion_requires_the_verified_report_marker(tmp_path):
    args,report,path,put=api_progress_case(tmp_path)
    line='[x] 已完成：jev-1.14.0：完整2项、3条'
    write_progress(*args)
    assert line not in args[-1].read_text()
    report['complete_full_evaluation']=True
    put(path,report)
    marker=dict(model='jev-1.14.0',samples=3,split_count=2,all_prediction_ids_verified=True,
        comparison_sha256='different-report')
    put(path.parent/'pipeline_complete.json',marker);write_progress(*args)
    assert line not in args[-1].read_text()
    marker['comparison_sha256']=file_sha256(path)
    put(path.parent/'pipeline_complete.json',marker);write_progress(*args)
    assert line in args[-1].read_text()
    report['time_utc']='changed after verification'
    put(path,report);write_progress(*args)
    assert line not in args[-1].read_text()


def test_training_progress_never_counts_evaluation_media(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    data = tmp_path / "data"
    log = control / "prepare-natural.log"
    log.write_text(json.dumps({"root": str(data / "eval/seg"), "images_ready": 9, "total": 10}) + "\n")
    assert "等待" in data_progress(control, "seg", data)
    with log.open("a") as stream:
        stream.write(json.dumps({"root": str(data / "seg"), "images_ready": 3, "total": 10}) + "\n")
    assert "3/10" in data_progress(control, "seg", data)


def test_expanded_gui_ready_data_is_not_reported_as_still_waiting(tmp_path):
    data=tmp_path/'data'
    path=data/'gui/manifests/ready.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(validated=True,samples=172000,images=23336)))
    state=dict(jobs=[],gui2_lora64_revision=dict(training_samples=172000,data_root=str(data)))
    output=tmp_path/'progress.md'
    write_progress(state,tmp_path/'registry.json',data,tmp_path/'models',output)
    assert '[x] 已完成：GUI 训练数据：172,000 条、23,336 张图' in output.read_text()
    state['gui2_lora64_revision']['training_samples']=172001
    write_progress(state,tmp_path/'registry.json',data,tmp_path/'models',output)
    assert '[x] 已完成：GUI 训练数据' not in output.read_text()


def test_installed_gui_revision_remains_complete_after_staging_is_retired(tmp_path):
    data, control = tmp_path/'data', tmp_path/'control'
    seal = data/'gui/manifests/ready.json'
    seal.parent.mkdir(parents=True)
    control.mkdir()
    current = dict(validated=True, samples=172000, images=23336, records_sha256='installed-data')
    seal.write_text(json.dumps(current))
    (control/'gui-data-revision-v2.json').write_text(json.dumps(dict(
        staging_root=str(tmp_path/'retired-staging'), target_samples=172000,
        install_report_file='install.json', write_access_verified=True)))
    report = control/'install.json'
    report.write_text(json.dumps(dict(applied=True, status='installed_and_verified', records_sha256='installed-data')))
    state = dict(jobs=[], gui2_lora64_revision=dict(training_samples=172000, data_root=str(data)))
    output = tmp_path/'progress.md'
    args = state, control/'registry.json', data, tmp_path/'models', output
    write_progress(*args)
    expected = '[x] 已完成：替换正式 gui 与 eval/gui 开发数据'
    assert expected in output.read_text()
    assert '[x] 已完成：构建并校验新版 GUI 172,000条' in output.read_text()
    # An old successful installation cannot certify a different current corpus.
    current['records_sha256'] = 'different-data'
    seal.write_text(json.dumps(current))
    write_progress(*args)
    assert expected not in output.read_text()
    current['records_sha256'] = 'installed-data'
    seal.write_text(json.dumps(current))
    report.write_text(json.dumps(dict(applied=False, status='preflight_passed', records_sha256='installed-data')))
    write_progress(*args)
    assert expected not in output.read_text()


def test_released_training_cards_are_not_reported_as_still_occupied(tmp_path):
    state = dict(jobs=[], running_gpus=8, reserved_gpus=8,
        gui2_lora64_revision=dict(training_samples=172000, data_root=str(tmp_path/'data')),
        release_gui2_after_training=dict(run_dir=str(tmp_path/'run'), released_utc='verified',
            evaluation_job_id='evaluation-only', status='evaluating_on_separate_8gpu_lease'))
    output = tmp_path/'progress.md'
    args = state, tmp_path/'registry.json', tmp_path/'data', tmp_path/'models', output
    write_progress(*args)
    text = output.read_text()
    assert '已获配 8 卡' in text and '原64卡训练租约已释放' in text
    assert 'LoRA独占64卡' not in text
    assert '评测已提交（独立8卡）' in text
    assert '[x] 已完成：另开8卡完成' not in text
    state['release_gui2_after_training']['status'] = 'evaluated_and_all_leases_released'
    state.update(running_gpus=0, reserved_gpus=0)
    write_progress(*args)
    assert '训练与评测卡组均已释放' in output.read_text()


def test_planned_step_limit_is_not_a_completed_epoch(tmp_path):
    stage = tmp_path / "joint"
    stage.mkdir()
    (stage / "metrics.jsonl").write_text(json.dumps({"seen": 32, "total_records": 100000, "step": 1}) + "\n")
    completion = stage / "training_complete.json"
    completion.write_text(json.dumps({"stop_reason": "planned_step_limit"}))
    done, _, activity = stage_progress(tmp_path, "joint", True, "failed")
    assert not done and activity == "待恢复/检查"
    completion.write_text(json.dumps({"stop_reason": "epochs_complete", "epoch": 1}))
    assert not stage_progress(tmp_path, "joint", True, "running")[0]
    (stage / "metrics.jsonl").write_text(json.dumps({"seen": 100000, "total_records": 100000, "step": 3125}) + "\n")
    assert stage_progress(tmp_path, "joint", True, "running")[0]
    assert not stage_progress(tmp_path, "joint", True, "running", expected_records=172000)[0]
    (stage / "metrics.jsonl").write_text(json.dumps({"seen": 172000, "total_records": 172000, "step": 1344}) + "\n")
    assert stage_progress(tmp_path, "joint", True, "running", expected_records=172000)[0]


def test_reasonseg_marker_cannot_replace_metrics_or_the_full_model_scope(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    manifest = data / "det/manifests/reasonseg.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"validated": True, "splits": {
        split: {"images": count} for split, count in (("train", 239), ("val", 200), ("test", 779))}}))
    run.mkdir()
    (run / "pipeline_complete.json").write_text("{}")
    state = {"jobs": [{"kind": "pipeline", "task": "det", "size": "0.8B", "run_dir": str(run)}]}
    output = tmp_path / "progress.md"
    arguments = (state, tmp_path / "registry.json", data, tmp_path / "models", output)
    write_progress(*arguments)
    assert "[ ] 等待补充评测：Valen-Decision-Det-0.8B：ReasonSeg" in output.read_text()
    assert "[ ] 待办：完成全部模型与对照实验" in output.read_text()
    supplement = run / "evaluation-reasonseg/pipeline_complete.json"
    supplement.parent.mkdir()
    supplement.write_text("{}")
    write_progress(*arguments)
    assert "[ ] 等待补充评测：Valen-Decision-Det-0.8B：ReasonSeg" in output.read_text()
    values = {}
    for name, samples in (("reasonseg_val", 200), ("reasonseg_test", 779)):
        path = data / "eval/det/reasonseg/manifests" / (name + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"samples": samples, "records_sha256": name}))
        values[name] = {"samples": samples, "data_sha256": name, "accuracy": .5}
    (supplement.parent / "metrics.json").write_text(json.dumps({
        "model": "Valen-Decision-Det-0.8B", "benchmarks": values}))
    write_progress(*arguments)
    assert "[x] 已完成：Valen-Decision-Det-0.8B：ReasonSeg" in output.read_text()
    assert "[x] 已完成：Valen-Decision-Det-0.8B\n" not in output.read_text()
    assert "[ ] 待办：完成全部模型与对照实验" in output.read_text()
    assert "[ ] 待办：归档各模型最终指标" in output.read_text()


def test_conditional_gold_requires_matching_group_coverage_and_records(tmp_path):
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    data = tmp_path / "data"
    control = tmp_path / "control"
    put(data / "det/manifests/ready.json", {"protocol": "coco_known_categories_v2"})
    put(data / "det/manifests/reasonseg.json", {})
    put(data / "eval/det/protocols/coco_prompt/manifests/coco2017_conditional_val2017.json",
        {"records_sha256": "current", "catalog_images": 5, "images": 4, "samples": 6})
    gold = {"perfect_prediction_metrics": {"AP": 1}, "records_sha256": "current",
            "protocol": "known_category_conditional_detection", "images": 5, "queried_images": 4, "prompts": 6}
    put(control / "coco-audit/gold-audit.json", gold)
    output = tmp_path / "progress.md"
    arguments = ({"jobs": []}, control / "registry.json", data, tmp_path / "models", output)
    write_progress(*arguments)
    expected = "已完成：COCO 全集真实框回放"
    assert expected not in output.read_text()
    put(control / "coco-prompt-v2-gold/gold-audit.json", gold)
    write_progress(*arguments)
    assert expected in output.read_text()
    gold["prompts"] = 5
    put(control / "coco-prompt-v2-gold/gold-audit.json", gold)
    write_progress(*arguments)
    assert expected not in output.read_text()
