"""Benchmark metrics distinguish clicks, top-1 boxes and global mask area."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluation.spatial import evaluate
from valen.modeling.qwen.spatial import SpatialOutput
from valen.training.distributed import Distributed


def test_phrasecut_uses_original_polygon_boundary_protocol():
    from valen.data.spatial import decode_mask
    mask = decode_mask({"size": [5, 5], "polygons": [[1.9, 1.9, 3.9, 1.9, 3.9, 3.9, 1.9, 3.9]],
                        "rasterization": "phrasecut"})
    assert mask.sum() == 9
    assert mask[1:4, 1:4].all()

def run_cases(task, cases):
    compiler = SimpleNamespace(compile=lambda record: SimpleNamespace(questions=[record["q"]]))
    model = lambda question: question.output
    return evaluate(model, compiler, [{"q": q} for q in cases], task, Distributed())


def question(kind, boxes, prediction):
    return SimpleNamespace(kind=kind, original_size=(2, 2), categories=["target"],
                           metadata={"platform": "desktop", "ui_type": "icon"},
                           media_root=Path("."), target={"boxes": boxes, "labels": [0]},
                           output=prediction)


def test_gui_uses_click_inclusion_rather_than_box_iou():
    output = SpatialOutput(torch.tensor([[10., -10.]]), boxes=torch.tensor([[.2, .2, .4, .4]]))
    result = run_cases("gui", [question("gui", [[.29, .29, .31, .31]], output)])
    assert result["accuracy"] == 1
    assert result["mean_box_iou"] < .1
    assert result["groups"]["platform/desktop"]["samples"] == 1


def test_referring_localization_cannot_choose_a_lower_ranked_oracle_box():
    output = SpatialOutput(torch.tensor([[10., -10.], [0., 10.]]),
                           boxes=torch.tensor([[.6, .6, .9, .9], [.1, .1, .4, .4]]))
    q = question("det", [[.1, .1, .4, .4]], output)
    q.metadata["referring"] = True
    result = run_cases("det", [q])
    assert result["accuracy"] == 0
    assert result["samples"] == 1


def test_cumulative_mask_iou_weights_pixels_instead_of_samples():
    positive = torch.tensor([[10., -10.]])
    first = question("seg", [], SpatialOutput(positive, mask_logits=torch.tensor([[[10., -10.], [-10., -10.]]])))
    first.target["mask"] = {"size": [2, 2], "counts": [0, 1, 3]}
    second = question("seg", [], SpatialOutput(positive, mask_logits=torch.full((1, 2, 2), -10.)))
    second.target["mask"] = {"size": [2, 2], "counts": [0, 4]}
    result = run_cases("seg", [first, second])
    assert result["mIoU"] == pytest.approx(.5)
    assert result["cIoU"] == pytest.approx(.2)


def test_missing_gui_box_counts_as_a_failed_sample():
    output = SpatialOutput(torch.tensor([[10., -10.]]), boxes=torch.zeros((1, 4)))
    result = run_cases("gui", [question("gui", [[.1, .1, .4, .4]], output)])
    assert result["accuracy"] == 0
    assert result["samples"] == 1
