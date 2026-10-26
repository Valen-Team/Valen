"""ReasonSeg preserves original polygon precedence, ignored pixels and independent splits."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

pytest.importorskip("cv2")

from evaluation.spatial import evaluate
from scripts.data.prepare_spatial_reasonseg import encode, prepare, rasterize
from valen.data.spatial import decode_segmentation_target, image_path, read_spatial_jsonl
from valen.modeling.qwen.spatial import SpatialOutput, decode_spatial
from valen.training.distributed import Distributed
from valen.training.spatial_loss import SpatialObjective


def annotation():
    return {"text": ["Locate the useful object", "Find the useful thing"], "is_sentence": True,
            "shapes": [{"label": "target", "points": [[1, 1], [8, 1], [8, 4], [1, 4]]},
                       {"label": "ignore", "points": [[3, 2], [5, 2], [5, 4], [3, 4]]},
                       {"label": "flag", "points": [[0, 0], [9, 0], [9, 5], [0, 5]]}]}


def test_reasonseg_polygon_precedence_and_ignore_are_preserved():
    mask = rasterize(annotation(), (6, 10))
    assert (mask == 1).sum() == 23
    assert (mask == 255).sum() == 9
    assert mask[0, 0] == 0
    truth, valid = decode_segmentation_target({"mask": encode(mask == 1), "ignore_mask": encode(mask == 255)})
    assert truth.sum() == 23 and (~valid).sum() == 9


def test_ignored_pixels_have_no_training_gradient_or_evaluation_penalty():
    mask = rasterize(annotation(), (6, 10))
    target = {"boxes": [[.1, 1/6, .9, 5/6]], "labels": [0],
              "mask": encode(mask == 1), "ignore_mask": encode(mask == 255)}
    logits = torch.tensor(np.where(mask == 1, 10., -10.))[None].float().requires_grad_()
    q = SimpleNamespace(kind="seg", original_size=(6, 10), media_root=Path("."), target=target,
                        metadata={"source": "reasonseg"}, categories=["target"])
    objective = SpatialObjective({})
    first, _ = objective(SpatialOutput(torch.tensor([[10., -10.]]), mask_logits=logits), q)
    first.backward()
    assert logits.grad[0][torch.tensor(mask == 255)].abs().sum() == 0
    changed = logits.detach().clone()
    changed[0][torch.tensor(mask == 255)] = 100
    second, _ = objective(SpatialOutput(torch.tensor([[10., -10.]]), mask_logits=changed), q)
    assert second.item() == pytest.approx(first.item())
    compiler = SimpleNamespace(compile=lambda _: SimpleNamespace(questions=[q]))
    model = lambda _: SpatialOutput(torch.tensor([[10., -10.]]), mask_logits=changed)
    result = evaluate(model, compiler, [{}], "seg", Distributed())
    assert result["mIoU"] == pytest.approx(1, abs=1e-6)
    assert result["cIoU"] == pytest.approx(1)


def test_preparation_materializes_both_tasks_and_keeps_official_first_eval_instruction(tmp_path):
    source, destination = tmp_path / "raw", tmp_path / "data"
    for split, color in (("train", "red"), ("val", "green"), ("test", "blue")):
        directory = source / split
        directory.mkdir(parents=True)
        Image.new("RGB", (10, 6), color).save(directory / "a.png")
        (directory / "a.json").write_text(json.dumps(annotation()))
    result = prepare(source, destination)
    assert result[("det", "train")]["samples"] == 2
    assert result[("seg", "val")]["samples"] == 1
    files = []
    for task in ("det", "seg"):
        for split in ("train", "val", "test"):
            root = destination / task if split == "train" else destination / "eval" / task / "reasonseg"
            rows = read_spatial_jsonl(root / "processed" / ("reasonseg_" + split + ".jsonl"))
            assert rows[0]["targets"]["target"]["boxes"][0] == pytest.approx([.1, 1/6, .9, 5/6])
            path = root / image_path(rows[0])
            assert path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
            files.append(path)
    assert len({path.stat().st_ino for path in files}) == 6
    shutil.rmtree(source)
    assert all(Image.open(path).size == (10, 6) for path in files)


@pytest.mark.parametrize("task", ["det", "seg"])
def test_training_profile_passes_cli_validation_before_environment_expansion(task, monkeypatch):
    from valen.modeling.factory import normalize_model_config
    from valen.training.spatial import normalize_spatial_config
    monkeypatch.setenv("VALEN_MODEL_PATH", "models/base")
    monkeypatch.setenv("VALEN_DATA_ROOT", "data-root")
    monkeypatch.setenv("VALEN_RUN_DIR", "output/reasonseg")
    monkeypatch.setenv("VALEN_STAGE", "joint")
    root = Path(__file__).resolve().parents[2]
    config = normalize_model_config(json.loads((root / "configs/train/qwen" / task / "reasonseg_sft.json").read_text()))
    config = normalize_spatial_config(config)
    assert config["task"] == task and config["required_world_size"] == (16 if task == "det" else 8)
    assert config["required_world_size"] * config["gradient_accumulation"] == 32
    assert config["data"] == f"data-root/{task}/processed/reasonseg_train.jsonl"
    assert config["stage"] == "joint"


def test_empty_targets_and_fully_ignored_regions_are_not_fabricated():
    empty = annotation()
    empty["shapes"] = []
    mask = rasterize(empty, (6, 10))
    assert not mask.any()
    q = SimpleNamespace(kind="det", original_size=(6, 10), categories=["target"],
                        metadata={"source": "reasonseg", "referring": True, "allow_empty_target": True},
                        target={"boxes": [], "labels": []}, media_root=Path("."))
    prediction = SpatialOutput(torch.tensor([[-10., 10.]]), boxes=torch.tensor([[.1, .1, .9, .9]]))
    assert decode_spatial(q, prediction)["boxes"] == []
    compiler = SimpleNamespace(compile=lambda _: SimpleNamespace(questions=[q]))
    assert evaluate(lambda _: prediction, compiler, [{}], "det", Distributed())["accuracy"] == 1
    q.target = {"boxes": [[.1, .1, .9, .9]], "labels": [0]}
    assert decode_spatial(q, prediction)["boxes"] == []
    assert evaluate(lambda _: prediction, compiler, [{}], "det", Distributed())["accuracy"] == 0
    q.kind = "seg"
    q.target = {"mask": encode(np.zeros((6, 10), dtype=np.uint8)),
                "ignore_mask": encode(np.ones((6, 10), dtype=np.uint8))}
    output = SpatialOutput(torch.tensor([[1., -1.]], requires_grad=True),
                           mask_logits=torch.randn(1, 6, 10, requires_grad=True))
    loss, _ = SpatialObjective({})(output, q)
    loss.backward()
    assert loss.item() == 0
    assert not output.mask_logits.grad.any()
