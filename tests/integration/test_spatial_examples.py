"""Qualitative records preserve original geometry and use label-independent selection."""
import json
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from evaluation.examples import overlay, save_examples, select_records
from valen.data.spatial import decode_mask
from valen.modeling.qwen.spatial import SpatialOutput


def test_selection_uses_ids_and_is_invariant_to_labels_and_order():
    records = [{"id": str(i), "targets": {"quality": i}} for i in range(10)]
    expected = [row["id"] for row in select_records(records, 3)]
    changed = [dict(row, targets={"quality": -1000}) for row in reversed(records)]
    assert [row["id"] for row in select_records(changed, 3)] == expected


@pytest.mark.parametrize("task", ["det", "seg", "gui"])
def test_small_examples_keep_real_predictions_and_source_identity(tmp_path, monkeypatch, task):
    image = tmp_path / "image.png"
    Image.new("RGB", (16, 12), "white").save(image)
    target = {"boxes": [[.25, .25, .75, .75]], "labels": [0]}
    if task == "seg":
        target["mask"] = {"size": [12, 16], "polygons": [[4, 3, 12, 3, 12, 9, 4, 9]]}
    record = {"id": "fixed", "group_id": "image", "request": {"state": {"messages": [{"content": [
        {"type": "image_url", "image_url": {"url": "image.png"}}]}]}, "questions": {"target": {
            "type": task, "instructions": "Find the target.", "categories": ["target"]}}}, "targets": {"target": target}}
    data = tmp_path / "processed/smoke.jsonl"
    data.parent.mkdir()
    data.write_text(json.dumps(record) + "\n")
    question = SimpleNamespace(kind=task, categories=["target"], original_size=(12, 16), target=target, metadata={})
    output = SpatialOutput(torch.tensor([[10., -10.]]), boxes=torch.tensor([[.25, .25, .75, .75]]),
                           mask_logits=torch.full((1, 3, 4), 10.) if task == "seg" else None)
    compiler = SimpleNamespace(compile=lambda row: SimpleNamespace(questions=[question]))
    monkeypatch.setattr("evaluation.examples.load_spatial_model", lambda *args, **kwargs: (lambda q: output, compiler))
    model = tmp_path / "model"
    model.mkdir()
    (model / "export_manifest.json").write_text('{"model_name": "fixture"}')
    report = save_examples(task, tmp_path, model, tmp_path / "examples", ["smoke"], device="cpu")
    row = report["examples"][0]
    assert row["record"] == record and row["data_sha256"] and report["model_export_manifest_sha256"]
    assert report["sampling"] == "smallest_sha256_of_record_id"
    if task == "seg":
        assert decode_mask(row["prediction"]["mask"]).shape == (12, 16)
    else:
        assert row["prediction"]["boxes"][0]["bbox"] == [.25, .25, .75, .75]
    with Image.open(tmp_path / "examples/examples.jpg") as preview:
        assert preview.width == 1120 and preview.height > 85


def test_overlay_rejects_misaligned_masks():
    with pytest.raises(ValueError, match="aligned"):
        overlay(Image.new("RGB", (8, 8)), [], torch.ones(2, 2), "red")
