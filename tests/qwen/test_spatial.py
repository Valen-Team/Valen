"""Spatial geometry, set supervision, schema validation and decoder gradients."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from valen.data.spatial import validate_spatial_record
from valen.modeling.qwen.spatial import SpatialHead, decode_spatial
from valen.training.spatial_loss import SpatialObjective, box_iou


def config(task="det"):
    return dict(task=task, spatial_width=32, spatial_heads=4, spatial_queries=4,
                spatial_layers=1, coordinate_bins=32)


def features():
    return torch.randn(12, 48), torch.randn(48), torch.randn(2, 48), (3, 4)


def question(task="det", boxes=None, labels=None):
    return SimpleNamespace(kind=task, categories=["first", "second"], original_size=(24, 32),
                           media_root=Path("."), metadata={},
                           target={"boxes": boxes if boxes is not None else [[.1, .2, .7, .8]],
                                   "labels": labels if labels is not None else [1]})


def test_set_loss_is_invariant_to_target_order_and_has_coordinate_gradients():
    torch.manual_seed(2)
    head = SpatialHead(48, config())
    output = head(*features())
    q = question(boxes=[[.1, .2, .7, .8], [.4, .1, .9, .5]], labels=[1, 0])
    objective = SpatialObjective({})
    loss, _ = objective(output, q)
    reverse = question(boxes=q.target["boxes"][::-1], labels=q.target["labels"][::-1])
    other, _ = objective(output, reverse)
    assert torch.allclose(loss, other)
    loss.backward()
    assert torch.isfinite(head.coordinate.weight.grad).all()
    assert head.coordinate.weight.grad.abs().sum() > 0


def test_empty_detection_target_is_finite():
    head = SpatialHead(48, config())
    loss, _ = SpatialObjective({})(head(*features()), question(boxes=[], labels=[]))
    assert torch.isfinite(loss)
    loss.backward()


def test_mask_loss_and_original_resolution_decode():
    head = SpatialHead(48, config("seg"))
    output = head(*features())
    q = question("seg")
    q.target["mask"] = {"size": [24, 32], "polygons": [[4, 4, 20, 4, 20, 20, 4, 20]]}
    loss, _ = SpatialObjective({})(output, q)
    loss.backward()
    assert head.mask_query.weight.grad.abs().sum() > 0
    assert decode_spatial(q, output)["mask"].shape == (24, 32)


def test_grid_mismatch_is_rejected():
    head = SpatialHead(48, config())
    image, decision, categories, _ = features()
    with pytest.raises(ValueError, match="spatial grid"):
        head(image, decision, categories, (2, 4))


def test_giou_identical_disjoint_and_empty():
    a = torch.tensor([[0., 0., .4, .4]])
    b = torch.tensor([[.6, .6, 1., 1.]])
    assert box_iou(a, a, generalized=True).item() == pytest.approx(1)
    assert box_iou(a, b).item() == 0
    assert box_iou(a, b, generalized=True).item() < 0
    assert box_iou(a, torch.empty(0, 4)).shape == (1, 0)


def test_spatial_schema_rejects_invalid_supervision():
    record = {"group_id": "image-1", "request": {
        "state": {"messages": [{"content": [{"type": "image_url", "image_url": {"url": "images/a.png"}}]}]},
        "questions": {"target": {"type": "det", "instructions": "Find the first", "categories": ["first"]}}},
        "targets": {"target": {"boxes": [[.9, .1, .1, .9]], "labels": [0]}}}
    with pytest.raises(ValueError, match="xyxy"):
        validate_spatial_record(record)
    record["targets"]["target"]["boxes"] = [[.1, .1, .9, .9]]
    assert validate_spatial_record(record) is record
    record["request"]["questions"]["target"]["type"] = "seg"
    with pytest.raises(ValueError, match="genuine mask"):
        validate_spatial_record(record)
