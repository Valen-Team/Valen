"""COCO scoring preserves sparse category IDs, non-square geometry and empty images."""
import json
from types import SimpleNamespace

import pytest
import torch

from evaluation.det.coco import predictions_to_coco
from evaluation.spatial import evaluate
from valen.modeling.qwen.spatial import SpatialHead, SpatialOutput
from valen.training.distributed import Distributed


def test_normalized_xyxy_becomes_pixel_xywh_with_sparse_category_ids():
    q = SimpleNamespace(original_size=(160, 320), categories=["one", "three", "ninety"],
                        metadata={"image_id": 7, "category_ids": [1, 3, 90]})
    p = {"bbox": [.1, .1, .3, .3], "label": 2, "score": .8}
    row = predictions_to_coco(q, {"boxes": [p]})[0]
    assert row["bbox"] == pytest.approx([32, 16, 64, 32])
    assert row["category_id"] == 90
    with pytest.raises(ValueError, match="Invalid normalized"):
        predictions_to_coco(q, {"boxes": [dict(p, bbox=[32, 16, 96, 48])]})


def test_perfect_predictions_score_full_ap_through_the_actual_evaluator(tmp_path):
    categories = [{"id": 1, "name": "one"}, {"id": 3, "name": "three"}, {"id": 90, "name": "ninety"}]
    annotations = tmp_path / "instances.json"
    annotations.write_text(json.dumps({"info": {}, "images": [
        {"id": i, "width": 320, "height": 160} for i in (7, 8)], "categories": categories,
        "annotations": [{"id": 1, "image_id": 7, "category_id": 90, "bbox": [32, 16, 64, 32],
                         "area": 2048, "iscrowd": 0}]}))
    questions = []
    for i in (7, 8):
        questions.append(SimpleNamespace(kind="det", original_size=(160, 320),
            categories=[item["name"] for item in categories], metadata={"image_id": i, "category_ids": [1, 3, 90]},
            target={"boxes": [[.1, .1, .3, .3]] if i == 7 else [], "labels": [2] if i == 7 else []},
            output=SpatialOutput(torch.tensor([[-20., -20., 20., -20.]]) if i == 7 else torch.empty(0, 4),
                                 boxes=torch.tensor([[.1, .1, .3, .3]]) if i == 7 else torch.empty(0, 4))))
    compiler = SimpleNamespace(compile=lambda record: SimpleNamespace(questions=[record]))
    result = evaluate(lambda question: question.output, compiler, questions, "det", Distributed(), annotations)
    assert result["samples"] == 2
    assert result["AP"] == pytest.approx(1)
    assert result["AP50"] == pytest.approx(1)


def test_coordinate_distributions_decode_center_size_to_normalized_corners():
    head = SpatialHead(48, dict(task="det", spatial_width=32, spatial_heads=4,
                               spatial_queries=2, spatial_layers=1, coordinate_bins=11))
    with torch.no_grad():
        head.coordinate.weight.zero_()
        head.coordinate.bias.fill_(-100)
        for axis, index in enumerate((5, 4, 6, 2)):
            head.coordinate.bias[axis * 11 + index] = 100
    output = head(torch.randn(4, 48), torch.randn(48), torch.randn(3, 48), (2, 2))
    assert torch.allclose(output.boxes, torch.tensor([[.2, .3, .8, .5]]).expand(2, -1), atol=1e-6)
