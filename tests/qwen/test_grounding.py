"""Grounding head: targets, losses, decoding and the bbox schema."""
import pytest
import torch

from valen.data.schema import box_target, candidates, validate_record
from valen.modeling.qwen.grounding import (BoxFeatures, GroundingHead, box_loss, decode, generalized_iou,
                                           token_targets)


def test_box_schema():
    assert candidates({"type": "bbox", "instructions": "the dog"}) == []
    assert box_target({"bbox": [0.1, 0.2, 0.3, 0.4]}) == {"bbox": [0.1, 0.2, 0.3, 0.4]}
    assert box_target({"point": [0.5, 0.5]}) == {"point": [0.5, 0.5]}
    with pytest.raises(ValueError):
        box_target({"bbox": [0.3, 0.2, 0.1, 0.4]})
    with pytest.raises(ValueError):
        box_target({"bbox": [0, 0, 1.2, 1]})
    validate_record({"group_id": "g", "request": {"state": "s", "questions": {
        "b": {"type": "bbox", "instructions": "x"}}}, "targets": {"b": {"bbox": [0, 0, 1, 1]}}})


def test_token_targets_inside_box_and_tiny_box():
    weights = token_targets({"bbox": [0.0, 0.0, 0.5, 0.5]}, (4, 4), "cpu").reshape(4, 4)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert weights[:2, :2].sum() == pytest.approx(1.0) and weights[2:].sum() == 0
    tiny = token_targets({"bbox": [0.26, 0.26, 0.27, 0.27]}, (4, 4), "cpu")
    assert tiny.max() == 1.0


def test_giou_identity():
    box = torch.tensor([[0.1, 0.1, 0.4, 0.5]])
    giou, iou = generalized_iou(box, box)
    assert torch.allclose(giou, torch.ones(1)) and torch.allclose(iou, torch.ones(1))


def test_head_learns_a_box():
    torch.manual_seed(0)
    rows, cols, hidden = 4, 6, 16
    tokens = torch.randn(rows * cols, hidden)
    head = GroundingHead(hidden, 16, 32)
    features = BoxFeatures(torch.randn(hidden), tokens, (rows, cols))
    target = {"bbox": [0.5, 0.25, 0.84, 0.75]}
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(300):
        optimizer.zero_grad()
        loss, _ = box_loss(head(features), target)
        loss.backward()
        optimizer.step()
    result = decode(head(features))
    _, iou = generalized_iou(torch.tensor([result["bbox"]]), torch.tensor([target["bbox"]]))
    assert iou.item() > 0.7 and 0 <= result["confidence"] <= 1
