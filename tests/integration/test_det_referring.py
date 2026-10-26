"""Referring-only revisions exclude COCO while preserving genuine image diversity."""
from collections import Counter
import json

import pytest

from evaluation.spatial import benchmark_paths
from scripts.data.prepare_det_referring import SOURCES, sample_key, sample_referring
from scripts.data.prepare_spatial import make_record
from scripts.train.run_spatial_pipeline import required_benchmarks
from scripts.train.schedule_spatial import data_ready


def record(source, image, instruction):
    return make_record("det", f"{source}:{image}:{instruction}", f"image:{image}",
                       f"images/{image}.png", instruction, ["target"], [[.1, .2, .7, .8]], [0],
                       source=source, referring=True)


def test_balanced_sampling_caps_images_and_deduplicates_original_instructions():
    pools = {source: [record(source, image, "shared" if image == 0 else source + str(image))
                      for image in range(6)] for source in SOURCES}
    pools["refcoco"].append(record("refcoco", 0, " SHARED. "))
    rows = sample_referring(pools, 9, max_per_image=2)
    assert len(rows) == len({sample_key(row) for row in rows}) == 9
    assert Counter(row["meta"]["source"] for row in rows) == {source: 3 for source in SOURCES}
    assert max(Counter(row["group_id"] for row in rows).values()) <= 2
    assert len({row["group_id"] for row in rows}) >= 5
    assert [row["id"] for row in rows] == [row["id"] for row in sample_referring(pools, 9, max_per_image=2)]


def test_insufficient_unique_data_cannot_be_repeated_to_fill_quota():
    pools = {source: [record(source, 0, source)] for source in SOURCES}
    with pytest.raises(ValueError, match="Insufficient"):
        sample_referring(pools, 12)


def test_default_det_suite_ignores_legacy_coco_files(tmp_path):
    directory = tmp_path / "processed"
    directory.mkdir()
    for name in ("coco2017_val2017", "refcoco_testA", "refcocoplus_val", "refcocog_test", "dev", "unrelated"):
        (directory / (name + ".jsonl")).touch()
    assert {p.stem for p in benchmark_paths(tmp_path, "det")} == {"refcoco_testA", "refcocoplus_val", "refcocog_test"}
    with pytest.raises(ValueError, match="COCO is excluded"):
        benchmark_paths(tmp_path, "det", "coco2017_val2017")
    assert benchmark_paths(tmp_path, "det", "coco2017_val2017", "standard")[0].stem == "coco2017_val2017"


def test_reasonseg_supplement_remains_available_without_coco(tmp_path):
    directory = tmp_path / "processed"
    directory.mkdir()
    for name in ("reasonseg_val", "reasonseg_test"):
        (directory / (name + ".jsonl")).touch()
    assert {p.stem for p in benchmark_paths(tmp_path, "det")} == {"reasonseg_val", "reasonseg_test"}
    assert benchmark_paths(tmp_path, "det", "reasonseg_test")[0].is_file()


def test_referring_pipeline_requires_all_eight_referring_splits():
    names = required_benchmarks("det", "none")
    assert len(names) == 8 and not any(name.startswith("coco2017") for name in names)
    assert {"refcoco_testA", "refcocoplus_testB", "refcocog_test"} <= names
    assert len(required_benchmarks("det", "conditional")) == 9
    assert len(required_benchmarks("seg")) == 10


def test_readiness_requires_reasonseg_and_rejects_coco_training_samples(tmp_path):
    def put(relative, value=""):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value)
        return path
    seal = {"format": "valen_det_referring_data_v4", "protocol": "det_referring_reasonseg_v4",
            "validated": True, "global_evaluation_overlap": 0, "independent_media": True,
            "coco_training_samples": 0, "records_sha256": "fixture",
            "source_counts": {**{source: 32892 for source in SOURCES}, "reasonseg": 1324}}
    path = put("det/manifests/ready.json", seal)
    put("det/processed/train.jsonl")
    put("det/manifests/train.json", {"samples": 100000, "records_sha256": "fixture"})
    put("eval/det/processed/dev.jsonl")
    for name in required_benchmarks("det", "none"):
        put("eval/det/processed/" + name + ".jsonl")
    put("eval/det/reasonseg/processed/reasonseg_val.jsonl")
    assert not data_ready(tmp_path, "det")
    put("eval/det/reasonseg/processed/reasonseg_test.jsonl")
    assert data_ready(tmp_path, "det")
    seal["coco_training_samples"] = 1
    path.write_text(json.dumps(seal))
    assert not data_ready(tmp_path, "det")


def test_deferred_seg_uses_new_rendezvous_and_task_specific_example_code(tmp_path):
    from scripts.train.spatial_pool import configure_allocations
    jobs = [{"name": "seg", "kind": "pipeline", "task": "seg", "size": "4B", "allocation": "held",
             "run_dir": str(tmp_path / "seg"), "log_dir": str(tmp_path / "seg-log"),
             "launch_id": "held-seg-resume-v4", "requires_files": [str(tmp_path / "det-complete")]},
            {"name": "det-full", "kind": "pipeline", "task": "det", "size": "0.8B", "variant": "full",
             "allocation": "held", "run_dir": str(tmp_path / "det-full"), "log_dir": str(tmp_path / "det-log"),
             "baseline_run": str(tmp_path / "det"), "examples_code_snapshot": "new-det-code"}]
    state = {"jobs": jobs, "gpu_limit": 64, "examples_code_snapshot": "old-code",
             "allocations": [{"name": "held", "job_id": "held-id", "gpus": 8, "nodes": 2,
                              "queue": str(tmp_path / "queue.json"), "initial_pipeline": None}]}
    configure_allocations(state, tmp_path / "data", tmp_path / "models", tmp_path / "code", lambda *args: True)
    tasks = {row["name"]: row for row in json.loads((tmp_path / "queue.json").read_text())["tasks"]}
    command = tasks["seg"]["command"]
    assert command[command.index("--launch-id") + 1] == "held-seg-resume-v4"
    assert not tasks["seg"]["ready"]
    assert tasks["det-full-examples"]["code_root"] == "new-det-code"
    assert tasks["seg-examples"]["code_root"] == "old-code"
