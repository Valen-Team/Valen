"""Standalone HF runtime packaging and calibration regressions."""

import importlib.util
import json
from types import SimpleNamespace

import pytest
import torch

from valen.modeling.qwen.hf_adapter_export import bundle_adapter_runtime
from valen.modeling.qwen.hf_export import base_manifest, bundle_responses, evaluate_records, final_metric, package_versions, sha_file


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


def test_export_reads_the_final_metric_the_trainer_writes(tmp_path):
    checkpoint = tmp_path / "latest"
    checkpoint.mkdir()
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"step": 1, "epoch_fraction": 0.5}) + "\n" + json.dumps({"step": 2, "epoch_fraction": 1.0}) + "\n")
    assert final_metric(checkpoint) == {"step": 2, "epoch_fraction": 1.0}
    (tmp_path / "training_complete.json").write_text(json.dumps({"last_metric": {"step": 9, "epoch_fraction": 1}}))
    assert final_metric(checkpoint)["step"] == 9


def test_export_identifies_a_base_without_a_manifest_by_its_weights(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")
    manifest = base_manifest(tmp_path)
    assert manifest["weight_sha256"] == {"model.safetensors": sha_file(tmp_path / "model.safetensors")}
    assert manifest["revision"].startswith("sha256:") and manifest == base_manifest(tmp_path)
    shipped = {"revision": "pinned", "weight_sha256": {"model.safetensors": "0" * 64}}
    (tmp_path / "valen_manifest.json").write_text(json.dumps(shipped))
    assert base_manifest(tmp_path) == shipped


@pytest.mark.parametrize("mask, executions", [
    ("causal", ["shared_state", "question"]), ("bbox_blocks", ["shared_state"]), ("question_blocks", ["shared_state"])])
def test_export_validation_runs_the_executions_the_mask_supports(mask, executions):
    question = SimpleNamespace(qid="q", keys=["a", "b"])
    seen = []

    class Compiler:
        def __init__(self, processor, max_length, media_kwargs, execution):
            seen.append(execution)

        def compile(self, record):
            return SimpleNamespace(questions=[question])

    class Model(torch.nn.Module):
        full_attention_mask = mask

        def forward(self, question):
            return torch.tensor([1.0, 0.0])

        def forward_state(self, compiled):
            return [torch.tensor([1.0, 0.0])]

    results = evaluate_records(Model(), Compiler, None, {"max_length": 8, "media_kwargs": {}}, [("case", {})])
    assert seen == executions and sorted(results) == sorted(e + "/case/q" for e in executions)


def test_export_records_versions_of_installed_packages_only():
    versions = package_versions("torch", "a-package-that-is-not-installed")
    assert list(versions) == ["torch"]
