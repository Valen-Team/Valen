"""A mid-epoch checkpoint resumes the same optimizer, order and learning-rate state."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from valen.modeling.qwen.spatial import SpatialOutput
from valen.training import spatial


class TinySpatialModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 18)
        self.base_manifest = {"revision": "test-base", "weight_sha256": {"base": "test"}}

    def forward(self, question):
        output = self.head(torch.ones(1, 1))
        logits = output[:, 2:].reshape(1, 4, 4)
        coordinates = (logits.softmax(-1) * torch.linspace(0, 1, 4)).sum(-1)
        center, size = coordinates[:, :2], coordinates[:, 2:]
        boxes = torch.cat((center - size / 2, center + size / 2), -1).clamp(0, 1)
        return SpatialOutput(output[:, :2], logits, boxes)


def test_spatial_exact_resume(tmp_path, monkeypatch):
    records = [{"id": str(i), "group_id": "image:" + str(i), "request": {
        "state": {"messages": [{"content": [{"type": "image_url", "image_url": {"url": "test.png"}}]}]},
        "questions": {"target": {"type": "det", "instructions": "Find the rectangle."}}},
        "targets": {"target": {"boxes": [[.2, .2, .8, .8]], "labels": [0]}}} for i in range(4)]
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(record) + "\n" for record in records))
    compiler = SimpleNamespace(compile=lambda record, **kwargs: SimpleNamespace(questions=[
        SimpleNamespace(kind="det", categories=["target"], target=record["targets"]["target"])]))
    monkeypatch.setattr(spatial, "build_model", lambda config: TinySpatialModel())
    monkeypatch.setattr(spatial, "build_compiler", lambda *args, **kwargs: compiler)
    monkeypatch.setattr(spatial, "optimizer_groups", lambda model, config: [
        {"params": list(model.parameters()), "lr": .01, "name": "head"}])
    config = dict(architecture="qwen_spatial", task="det", model_path="test-base", data=str(data),
                  device="cpu", required_world_size=1, gradient_accumulation=1, epochs=1, max_steps=4,
                  save_every=2, seed=42, stage="warmup")
    full, _, full_progress = spatial.run(dict(config, output=str(tmp_path / "full")))
    original_save = spatial.save_checkpoint

    class SimulatedInterruption(Exception):
        pass

    def interrupt_after_checkpoint(path, model, optimizer, config, progress, *args, **kwargs):
        original_save(path, model, optimizer, config, progress, *args, **kwargs)
        if Path(path).name == "latest" and progress["step"] == 2:
            raise SimulatedInterruption()

    monkeypatch.setattr(spatial, "save_checkpoint", interrupt_after_checkpoint)
    interrupted_config = dict(config, output=str(tmp_path / "interrupted"))
    with pytest.raises(SimulatedInterruption):
        spatial.run(interrupted_config)
    monkeypatch.setattr(spatial, "save_checkpoint", original_save)
    resumed, _, resumed_progress = spatial.run(interrupted_config, resume=tmp_path / "interrupted/latest")
    assert full_progress == resumed_progress
    for name, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name
