"""GroundingME scoring, rejection, data isolation and result identity."""
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from evaluation.det.groundingme import aggregate, score_box
from scripts.data.prepare_groundingme import (
    BENCHMARK, DATASET_REVISION, PROMPT, PROTOCOL, SCORER_REVISION, prepare,
)
from scripts.data.prepare_spatial import digest_file, write_json
from scripts.train.summarize_spatial import groundingme_checks


def original_samples(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), "green").save(buffer, format="PNG")
    raw = buffer.getvalue()
    samples = [{"id": i, "image": {"bytes": raw, "path": None}, "description": "The small green object.",
        "bbox": [10, 8, 50, 40] if dimension != "Rejection" else None,
        "subtask_l1": dimension, "subtask_l2": "Appearance" if i == 0 else "",
        "width": 100, "height": 80} for i, dimension in enumerate(
            ("Discriminative", "Spatial", "Limited", "Rejection"))]
    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(pa.Table.from_pylist(samples), source / "test-00000-of-00001.parquet")
    return source, raw


def test_continuous_iou_boundaries_rejection_and_no_coordinate_guessing():
    assert score_box([], []) == 1
    assert score_box([], [0, 0, 1, 1]) == 0
    assert score_box([0, 0, 1, 1], []) == 0
    assert score_box([0, 0, .5, 1], [0, 0, 1, 1]) == .5
    assert score_box([1, 1, 0, 0], [0, 0, 1, 1]) == 0
    with pytest.raises(ValueError, match="without format guessing"):
        score_box([0, 0, 100, 100], [0, 0, 1, 1])
    cases = [{"id": str(i), "subtask_l1": "Spatial" if i < 3 else "Rejection",
        "subtask_l2": "Relative" if i < 3 else "", "iou": .5 if i < 3 else 0} for i in range(4)]
    values = aggregate(cases)
    assert values["accuracy"] == .75  # sample weighting, not mean of dimension means
    assert values["accuracy_at_075"] == 0
    assert values["groups"]["Spatial/Relative"]["samples"] == 3
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(cases + [cases[0]])


def test_original_bytes_independent_media_uniform_prompt_and_native_coordinates(tmp_path):
    source, raw = original_samples(tmp_path)
    data = tmp_path / "data"
    manifest = prepare(source, data, expected_samples=4)
    root = data / "eval/det/groundingme"
    records = [json.loads(line) for line in (root / "processed" / (BENCHMARK + ".jsonl")).read_text().splitlines()]
    assert len(records) == 4 and manifest["images"] == 1
    assert records[0]["targets"]["target"]["boxes"] == [[.1, .1, .5, .5]]
    assert records[3]["targets"]["target"]["boxes"] == []
    assert all(r["meta"]["allow_empty_target"] is True for r in records)
    assert len({r["request"]["questions"]["target"]["instructions"] for r in records}) == 1
    assert records[0]["request"]["questions"]["target"]["instructions"] == PROMPT.format(description="The small green object.")
    media = root / next(iter(manifest["media_sha256"]))
    assert media.read_bytes() == raw and media.stat().st_nlink == 1 and not media.is_symlink()
    assert prepare(source, data, expected_samples=4)["records_sha256"] == manifest["records_sha256"]
    write_json(data / "det/manifests/train.json", {"media_sha256": {"training.png": digest_file(media)}})
    # A contaminated rerun must not leave a usable validation marker.
    with pytest.raises(ValueError, match="overlap training"):
        prepare(source, data, expected_samples=4)
    assert not (root / "manifests/preflight.json").exists()


def test_supplement_requires_complete_counts_and_exact_export_identity(tmp_path):
    run, package, data = (tmp_path / p for p in ("run", "model", "data"))
    name = "Valen-Decision-Det-4B"
    write_json(package / "export_manifest.json", {"model_name": name, "weight": "current"})
    values = {"samples": 1005, "accuracy": .5, "accuracy_at_075": .3, "accuracy_at_09": .1,
        "mean_box_iou": .4, "protocol": PROTOCOL, "data_sha256": "data",
        "source_revision": DATASET_REVISION, "scorer_revision": SCORER_REVISION,
        "groups": {k: {"samples": n, "accuracy": .5, "accuracy_at_075": .3,
            "accuracy_at_09": .1, "mean_box_iou": .4} for k, n in
            {"Discriminative": 205, "Spatial": 300, "Limited": 300, "Rejection": 200}.items()}}
    write_json(data / "eval/det/groundingme/manifests" / (BENCHMARK + ".json"),
        {"samples": 1005, "records_sha256": "data", "groups": {k: v["samples"] for k, v in values["groups"].items()}})
    report = {"model": name, "model_export_manifest_sha256": digest_file(package / "export_manifest.json"),
              "benchmarks": {BENCHMARK: values}}
    write_json(run / "evaluation-groundingme/metrics.json", report)
    write_json(run / "evaluation-groundingme/pipeline_complete.json", {"model": name,
        "samples": 1005, "data_sha256": "data", "protocol": PROTOCOL})
    assert all(groundingme_checks(run, package, name, data)[0].values())
    values["groups"]["Spatial"]["samples"] = 299
    write_json(run / "evaluation-groundingme/metrics.json", report)
    assert not groundingme_checks(run, package, name, data)[0]["groundingme"]
    write_json(package / "export_manifest.json", {"model_name": name, "weight": "changed"})
    assert not groundingme_checks(run, package, name, data)[0]["groundingme_model_identity"]
