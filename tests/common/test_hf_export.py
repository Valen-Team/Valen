"""Standalone HF runtime packaging and calibration regressions."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from valen.modeling.qwen.hf_adapter_export import bundle_adapter_runtime
from valen.modeling.qwen.hf_export import bundle_responses


def read_answers(folder):
    spec = importlib.util.spec_from_file_location(
        "exported_responses", folder / "responses.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.answer


def test_adapter_export_accepts_existing_systemone_packages(tmp_path):
    merged, stage = tmp_path / "merged", tmp_path / "unmerged"
    merged.mkdir()
    preserved = [
        "configuration_valen.py",
        "heads.py",
        "compiler.py",
        "runtime_model.py",
        "batching.py",
        "schema.py",
        "data_types.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "LICENSE",
    ]
    for name in preserved:
        (merged / name).write_text("checkpoint-specific " + name)
    (merged / "modeling_valen.py").write_text("old SystemOne wrapper")
    (merged / "responses.py").write_text("old probability readout")

    bundle_adapter_runtime(merged, stage)

    assert not (merged / "decisions.py").exists()
    for name in preserved:
        assert (stage / name).read_bytes() == (merged / name).read_bytes()
    assert "allow_video=True" in (stage / "modeling_valen.py").read_text()
    assert "def to_internal" in (stage / "decisions.py").read_text()
    assert "from .decisions import" in (stage / "modeling_valen_lora.py").read_text()
    answer = read_answers(stage)
    q = SimpleNamespace(kind="choice", keys=["cat", "dog"])
    assert answer(q, torch.tensor([2.0, 0.0]), 1e-50)["probabilities"] == {
        "cat": 1.0,
        "dog": 0.0,
    }


@pytest.mark.parametrize("logits", [[2.0, 0.0], [-2.0, -2.0]])
def test_exported_temperature_stays_finite(tmp_path, logits):
    from pathlib import Path

    bundle_responses(Path(__file__).resolve().parents[2], tmp_path)
    answer = read_answers(tmp_path)
    q = SimpleNamespace(kind="choice", keys=["cat", "dog"])
    result = answer(q, torch.tensor(logits), 1e-50)
    probabilities = list(result["probabilities"].values())
    assert all(0 <= p <= 1 for p in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)
