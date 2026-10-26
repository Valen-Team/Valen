"""Result archives distinguish complete deliverables, partial runs and changed inputs."""
import copy
import json

from scripts.train.run_spatial_pipeline import required_benchmarks
from scripts.train.summarize_spatial import benchmark_checks, collect_results, completed_costs, write_summary
from valen.training.checkpoint import file_sha256


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completed_gui(tmp_path):
    data, models, run = tmp_path / "data", tmp_path / "models", tmp_path / "run"
    name = "Valen-Decision-GUI-0.8B"
    put(data / "gui/manifests/ready.json", {"records_sha256": "training", "development_sha256": "development"})
    for stage in ("warmup", "joint"):
        put(run / stage / "training_complete.json", {"stop_reason": "epochs_complete", "epoch": 1, "best_step": 3125})
        put(run / stage / "run_manifest.json", {"world_size": 8, "gpu_names": ["test GPU"] * 8,
            "trainable_parameters": 100, "total_parameters": 1000,
            "config": {"data_sha256": "training", "validation_sha256": "development"}})
        (run / stage / "metrics.jsonl").write_text(json.dumps({"seen": 100000, "total_records": 100000,
            "step": 3125, "loss": 1., "elapsed_seconds": 20, "peak_cuda_bytes": 30, "peak_cuda_scope": "all_ranks"}) + "\n")
    put(run / "pipeline_complete.json", {"task": "gui", "model": name})
    put(models / name / "export_manifest.json", {"model_name": name, "training_data_sha256": "training",
        "reload_validation": {"independent_process": True}, "files_sha256": {"model.safetensors": "fixture"}})
    put(run / "examples/examples.json", {"format": "valen_spatial_examples_v1", "model": name,
        "model_export_manifest_sha256": file_sha256(models / name / "export_manifest.json"), "examples": [{"id": "sample"}]})
    put(run / "examples/pipeline_complete.json", {})
    (run / "examples/examples.jpg").write_bytes(b"preview fixture")
    (models / name / "model.safetensors").write_bytes(b"metadata readiness test; no model loading")
    benchmarks = {}
    for benchmark in required_benchmarks("gui"):
        put(data / "eval/gui/manifests" / (benchmark + ".json"), {"samples": 2, "records_sha256": benchmark})
        benchmarks[benchmark] = {"samples": 2, "data_sha256": benchmark, "accuracy": .5,
            "groups": {"platform/web": {"samples": 2, "accuracy": .5}}}
    put(run / "evaluation/metrics.json", {"model": name, "benchmarks": benchmarks})
    state = {"jobs": [{"kind": "pipeline", "task": "gui", "size": "0.8B", "run_dir": str(run), "state": "succeeded"}]}
    return state, data, models, run, benchmarks


def test_available_models_do_not_shrink_the_eighteen_model_scope(tmp_path):
    state, data, models, _, _ = completed_gui(tmp_path)
    report = write_summary(state, data, models, tmp_path / "results")
    assert report["completed_models"] == 1 and report["models"][0]["deliverables_ready"]
    assert report["expected_models"] == 18 and not report["model_scope_complete"]
    assert not report["deliverables_ready"]
    assert (tmp_path / "results/summary.json").is_file()
    assert "Complete recorded artifact sets: 1/18" in (tmp_path / "results/summary.md").read_text()


def test_revision_deliverables_use_the_jobs_sealed_corpus(tmp_path):
    state, data, models, _, _ = completed_gui(tmp_path)
    state['jobs'][0]['data_root'] = str(data)
    row = collect_results(state, tmp_path/'unrelated-global-data', models)['models'][0]
    assert row['checks']['current_training_data'] and row['checks']['matched_development']
    assert row['checks']['required_benchmarks'] and row['deliverables_ready']


def test_additional_gui_tests_are_required_only_for_the_requested_run(tmp_path):
    state,data,models,run,_=completed_gui(tmp_path)
    assert collect_results(state,data,models)['models'][0]['deliverables_ready']
    state['gui_additional_benchmarks']=dict(after_run=str(tmp_path/'another-run'))
    assert collect_results(state,data,models)['models'][0]['deliverables_ready']
    state['gui_additional_benchmarks']['after_run']=str(run)
    result=collect_results(state,data,models)['models'][0]
    assert not result['checks']['additional_gui_benchmarks'] and not result['deliverables_ready']


def test_historical_model_requires_its_original_unchanged_data_seal(tmp_path):
    state, data, models, _, _ = completed_gui(tmp_path)
    original = tmp_path / 'seals/original.json'
    original.parent.mkdir()
    original.write_bytes((data / 'gui/manifests/ready.json').read_bytes())
    manifest = models / 'Valen-Decision-GUI-0.8B/export_manifest.json'
    export = json.loads(manifest.read_text())
    export['data_seal_sha256'] = file_sha256(original)
    put(manifest, export)
    # Updating the manifest requires matching inference example provenance.
    run = tmp_path / 'run'
    example_path = run / 'examples/examples.json'
    example = json.loads(example_path.read_text())
    example['model_export_manifest_sha256'] = file_sha256(manifest)
    put(example_path, example)
    put(data / 'gui/manifests/ready.json', {'records_sha256': 'new-training',
        'development_sha256': 'new-development'})
    assert not collect_results(state, data, models)['models'][0]['deliverables_ready']
    state['jobs'][0]['data_seal_path'] = str(original)
    assert collect_results(state, data, models)['models'][0]['deliverables_ready']
    # A seal with equivalent fields but altered bytes is not the exported seal.
    original.write_text(original.read_text() + '\n')
    checks = collect_results(state, data, models)['models'][0]['checks']
    assert checks['current_training_data'] and not checks['pinned_data_seal']
    original.unlink()
    assert not collect_results(state, data, models)['models'][0]['checks']['pinned_data_seal']


def test_requested_four_b_full_extends_instead_of_replacing_the_original_scope(tmp_path):
    state, data, models, _, _ = completed_gui(tmp_path)
    state["additional_full_comparisons"] = [{"task": "det", "size": "4B"}]
    assert collect_results(state, data, models)["expected_models"] == 19
    state["jobs"] = []
    for task in ("det", "seg", "gui"):
        variants = [(size, "lora") for size in ("0.8B", "2B", "4B", "9B")]
        variants += [(size, "full") for size in ("0.8B", "2B")]
        if task == "det":
            variants.append(("4B", "full"))
        for size, variant in variants:
            state["jobs"].append({"kind": "pipeline", "task": task, "size": size, "variant": variant,
                "run_dir": str(tmp_path / task / size / variant), "baseline_run": str(tmp_path / "reference")})
    report = collect_results(state, data, models)
    assert report["expected_models"] == 19 and report["model_scope_complete"]
    assert not report["deliverables_ready"]
    state["jobs"] = state["jobs"][:-1]
    assert not collect_results(state, data, models)["model_scope_complete"]


def test_planned_det_rlcd_is_pending_work_in_the_complete_model_scope(tmp_path):
    state, data, models, _, _ = completed_gui(tmp_path)
    state["additional_full_comparisons"] = [{"task": "det", "size": "4B"}]
    state["det_rlcd_plan"] = {"sizes": ["0.8B", "2B", "4B"]}
    report = collect_results(state, data, models)
    assert report["expected_models"] == 22
    assert not report["model_scope_complete"] and not report["deliverables_ready"]


def test_pipeline_marker_does_not_replace_missing_benchmarks_or_weights(tmp_path):
    state, data, models, run, benchmarks = completed_gui(tmp_path)
    benchmarks.pop("screenspot_pro_test")
    put(run / "evaluation/metrics.json", {"benchmarks": benchmarks})
    row = collect_results(state, data, models)["models"][0]
    assert row["checks"]["pipeline_complete"] and not row["checks"]["required_benchmarks"]
    assert not row["deliverables_ready"]
    (models / row["name"] / "model.safetensors").unlink()
    assert not collect_results(state, data, models)["models"][0]["checks"]["export_files_present"]


def test_examples_must_match_the_current_export(tmp_path):
    state, data, models, run, _ = completed_gui(tmp_path)
    assert collect_results(state, data, models)["models"][0]["checks"]["inference_examples"]
    manifest = models / "Valen-Decision-GUI-0.8B/export_manifest.json"
    value = json.loads(manifest.read_text())
    value["source_checkpoint_sha256"] = "changed"
    put(manifest, value)
    assert not collect_results(state, data, models)["models"][0]["checks"]["inference_examples"]


def test_benchmark_results_from_another_model_cannot_complete_a_deliverable(tmp_path):
    state, data, models, run, benchmarks = completed_gui(tmp_path)
    assert collect_results(state, data, models)["models"][0]["deliverables_ready"]
    put(run / "evaluation/metrics.json", {"model": "Valen-Decision-GUI-2B", "benchmarks": benchmarks})
    model = collect_results(state, data, models)["models"][0]
    assert model["checks"]["required_benchmarks"]
    assert not model["checks"]["evaluation_model_identity"]
    assert not model["deliverables_ready"]


def test_progress_rejects_results_assigned_to_another_model(tmp_path):
    from scripts.train.spatial_progress import write_progress
    state, data, models, run, benchmarks = completed_gui(tmp_path)
    output = tmp_path / "progress.md"
    arguments = state, tmp_path / "registry.json", data, models, output
    write_progress(*arguments)
    assert "[x] 已完成：Valen-Decision-GUI-0.8B" in output.read_text()
    put(run / "evaluation/metrics.json", {"model": "Valen-Decision-GUI-2B", "benchmarks": benchmarks})
    write_progress(*arguments)
    text = output.read_text()
    assert "[x] 已完成：Valen-Decision-GUI-0.8B" not in text
    assert "[ ] 待办：全部 3 个必测 benchmark/split 完整评测" in text


def test_benchmark_samples_fingerprints_and_required_metrics_are_verified(tmp_path):
    name = "coco2017_conditional_val2017"
    put(tmp_path / "eval/det/protocols/coco_prompt/manifests" / (name + ".json"),
        {"samples": 2, "records_sha256": "expected"})
    values = {metric: .2 for metric in ("AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L")}
    values.update(samples=2, data_sha256="expected")
    assert benchmark_checks({name: values}, {name}, "det", tmp_path)
    for change in ({"samples": 1}, {"data_sha256": "changed"}, {"AP50": float("nan")}, {"AP75": None}):
        modified = dict(values, **change)
        assert not benchmark_checks({name: modified}, {name}, "det", tmp_path)
    missing = copy.deepcopy(values)
    missing.pop("AP75")
    assert not benchmark_checks({name: missing}, {name}, "det", tmp_path)


def test_recorded_training_cost_adds_resumed_process_segments(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps({"elapsed_seconds": elapsed, "peak_cuda_bytes": peak,
        "peak_cuda_scope": "all_ranks"}) + "\n" for elapsed, peak in ((8, 3), (12, 7), (2, 4), (5, 2))))
    assert completed_costs(str(path), path.stat().st_mtime_ns) == {
        "recorded_training_seconds": 17, "peak_cuda_bytes_all_ranks": 7}


def test_reasonseg_deltas_require_matching_evaluation_records(tmp_path):
    runs = {"lora": tmp_path / "lora", "full": tmp_path / "full"}
    state = {"reasonseg_support": {"validated": True}, "jobs": []}
    for variant, run in runs.items():
        job = {"kind": "pipeline", "task": "det", "size": "2B", "variant": variant, "run_dir": str(run)}
        if variant == "full":
            job["baseline_run"] = str(runs["lora"])
        state["jobs"].append(job)
        values = {}
        for name, samples in (("reasonseg_val", 200), ("reasonseg_test", 779)):
            put(tmp_path / "data/eval/det/reasonseg/manifests" / (name + ".json"),
                {"samples": samples, "records_sha256": name})
            values[name] = {"samples": samples, "data_sha256": name, "accuracy": .65 if variant == "full" else .4}
        name = "Valen-Decision-Det-2B" + ("-Full" if variant == "full" else "")
        put(run / "evaluation-reasonseg/metrics.json", {"model": name, "benchmarks": values})
        put(run / "evaluation-reasonseg/pipeline_complete.json", {})
    full = collect_results(state, tmp_path / "data", tmp_path / "models")["models"][1]
    assert full["reasonseg_comparison"]["reasonseg_test"]["full_minus_lora"]["accuracy"] == .25
    assert full["checks"]["reasonseg_comparison_data"]
    changed = json.loads((runs["full"] / "evaluation-reasonseg/metrics.json").read_text())
    changed["benchmarks"]["reasonseg_test"]["data_sha256"] = "different-records"
    put(runs["full"] / "evaluation-reasonseg/metrics.json", changed)
    full = collect_results(state, tmp_path / "data", tmp_path / "models")["models"][1]
    assert full["reasonseg_comparison"] == {} and not full["checks"]["reasonseg_comparison_data"]
    changed["benchmarks"]["reasonseg_test"]["data_sha256"] = "reasonseg_test"
    changed["model"] = "Valen-Decision-Det-0.8B-Full"
    put(runs["full"] / "evaluation-reasonseg/metrics.json", changed)
    full = collect_results(state, tmp_path / "data", tmp_path / "models")["models"][1]
    assert full["checks"]["reasonseg"]
    assert not full["checks"]["reasonseg_model_identity"]
    assert not full["checks"]["reasonseg_comparison_data"]
    assert full["reasonseg_comparison"] == {}
