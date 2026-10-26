"""Full spatial SFT shares the warmup head and checks matched comparison inputs."""
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from scripts.train.spatial_comparison import compare, full_config, retain_warmup, write_json


def reference(tmp_path):
    baseline = tmp_path / "lora"
    (baseline / "warmup/best").mkdir(parents=True)
    config = dict(architecture="qwen_spatial", task="gui", model_path="same-base", stage="warmup",
                  finetuning_type="lora", seed=42, epochs=1, gradient_accumulation=4,
                  required_world_size=8, dtype="bf16", max_pixels=1000,
                  data_sha256="train-hash", validation_sha256="dev-hash", code_sha256="reference-code")
    torch.save({"weights": {"head.weight": torch.tensor([[1., 2.]])}, "optimizer": {"unused": True},
                "config": config, "base_manifest": {"revision": "base"}, "progress": {"step": 3}},
               baseline / "warmup/best/checkpoint.pt")
    write_json(baseline / "warmup/training_complete.json", {"stop_reason": "epochs_complete", "best_step": 3})
    (baseline / "joint").mkdir()
    config = dict(config, stage="joint", head_lr=1e-4, lora_lr=5e-5, lora_rank=32)
    write_json(baseline / "joint/run_manifest.json", {"config": config, "trainable_parameters": 2,
               "total_parameters": 10, "world_size": 8, "gpu_names": ["test"] * 8})
    write_json(baseline / "pipeline_complete.json", {"model": "reference-model"})
    return baseline


def test_retained_initialization_survives_reference_cleanup_and_checks_protocol(tmp_path):
    baseline = reference(tmp_path)
    assert retain_warmup(baseline)
    kept = torch.load(baseline / "comparison-init/checkpoint.pt", weights_only=False)
    assert set(kept) == {"weights", "config", "base_manifest"}
    torch.testing.assert_close(kept["weights"]["head.weight"], torch.tensor([[1., 2.]]))
    (baseline / "warmup/best/checkpoint.pt").unlink()
    assert retain_warmup(baseline)
    config = full_config(baseline, tmp_path / "full", "comparison-code")
    assert config["finetuning_type"] == "full"
    assert not any(key.startswith("lora_") for key in config)
    assert config["gradient_accumulation"] == 4 and config["epochs"] == 1
    assert config["language_lr"] == 1e-5 and config["vision_lr"] == 2e-6
    path = baseline / "joint/run_manifest.json"
    altered = json.loads(path.read_text())
    altered["config"]["validation_sha256"] = "different-dev"
    write_json(path, altered)
    with pytest.raises(ValueError, match="protocol differ"):
        full_config(baseline, tmp_path / "full", "comparison-code")


def test_explicit_full64_override_preserves_sample_schedule_and_shared_head(tmp_path):
    baseline = reference(tmp_path)
    retain_warmup(baseline)
    for path in [baseline/"comparison-init/config.json", baseline/"joint/run_manifest.json"]:
        value = json.loads(path.read_text())
        config = value.get("config", value)
        config.update(required_world_size=16, gradient_accumulation=2, save_every=125)
        write_json(path, value)
    config = full_config(baseline, tmp_path/"full64", "new-code", training_world_size=64)
    assert config["required_world_size"] == 64 and config["gradient_accumulation"] == 2
    assert config["lr_warmup_steps"] == 25 and config["save_every"] == 32
    assert config["gradient_bucket_mb"] == 128
    assert config["head_lr"] == 1e-4 and config["language_lr"] == 1e-5
    override = config["comparison_resource_override"]
    assert override["reference_global_batch"] == 32 and override["training_global_batch"] == 128
    assert override["learning_rate_scaling"] == "unchanged"
    path = baseline/"joint/run_manifest.json"
    changed = json.loads(path.read_text()); changed["config"]["seed"] = 999; write_json(path, changed)
    with pytest.raises(ValueError, match="protocol differ"):
        full_config(baseline, tmp_path/"full64", "new-code", training_world_size=64)


def test_comparison_deltas_reject_different_evaluation_data(tmp_path):
    baseline = reference(tmp_path)
    retain_warmup(baseline)
    full = tmp_path / "full"
    (full / "joint").mkdir(parents=True)
    config = full_config(baseline, full, "comparison-code")
    write_json(full / "joint/run_manifest.json", {"config": config, "trainable_parameters": 10,
               "total_parameters": 10, "world_size": 8, "gpu_names": ["test"] * 8})
    for run, accuracy in ((baseline, .1), (full, .3)):
        (run / "evaluation").mkdir()
        write_json(run / "evaluation/metrics.json", {"inference": {"parameters": 10}, "benchmarks": {
            "screenspot_test": {"samples": 4, "data_sha256": "same-evaluation", "accuracy": accuracy,
                                "latency_seconds": {"mean": .2}}}})
        # Resuming a process resets its elapsed clock; include both segments in the cost.
        (run / "joint/metrics.jsonl").write_text("".join(json.dumps({"elapsed_seconds": elapsed,
            "peak_cuda_bytes": peak, "peak_cuda_scope": "all_ranks"}) + "\n"
            for elapsed, peak in [(8, 100), (12, 300), (3, 100)]))
    result = compare(baseline, full, tmp_path / "model-full")
    assert result["benchmarks"]["screenspot_test"]["full_minus_lora"]["accuracy"] == pytest.approx(.2)
    assert result["methods"]["full"]["trainable_parameters"] == 10
    for method in result["methods"].values():
        assert method["training_elapsed_seconds"] == 15
        assert method["training_peak_cuda_bytes"] == 300
        assert method["training_time_scope"] == "sum_of_recorded_process_segments"
    assert "model_path" not in result["protocol"]
    path = full / "evaluation/metrics.json"
    changed = json.loads(path.read_text())
    changed["benchmarks"]["screenspot_test"]["data_sha256"] = "different"
    write_json(path, changed)
    with pytest.raises(ValueError, match="evaluation inputs differ"):
        compare(baseline, full, tmp_path / "model-full")


def test_real_full_spatial_update_changes_language_visual_encoder_and_head(tmp_path):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from valen.modeling.factory import build_model, optimizer_groups
    from valen.training.checkpoint import load_checkpoint, save_checkpoint
    from valen.training.spatial_loss import SpatialObjective
    base_config = Qwen3_5Config(text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=["full_attention"], rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                                        "mrope_section": [2, 1, 1]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
            out_hidden_size=32, patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=16),
        image_token_id=5, video_token_id=6, vision_start_token_id=7, vision_end_token_id=8)
    base = tmp_path / "base"
    Qwen3_5ForConditionalGeneration(base_config).save_pretrained(base)
    config = dict(architecture="qwen_spatial", task="gui", model_path=str(base), dtype="bf16", device="cpu",
                  stage="warmup", finetuning_type="lora", attn_implementation="sdpa",
                  gradient_checkpointing=True, spatial_width=32, spatial_heads=4,
                  spatial_queries=1, spatial_layers=1, coordinate_bins=32)
    warmup = build_model(config)
    assert all(p.requires_grad == name.startswith("head.") for name, p in warmup.named_parameters())
    baseline = tmp_path / "reference"
    save_checkpoint(baseline / "warmup/best", warmup, torch.optim.AdamW(optimizer_groups(warmup, config)),
                    config, {"step": 1}, random.Random(42))
    write_json(baseline / "warmup/training_complete.json", {"stop_reason": "epochs_complete", "best_step": 1})
    retain_warmup(baseline)
    config = dict(config, stage="joint", finetuning_type="full")
    model = build_model(config).train()
    load_checkpoint(baseline / "comparison-init", model, strict=False)
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    assert not any("lora_" in name for name, _ in model.named_parameters())
    assert model.backbone_autocast_dtype == torch.bfloat16
    for key, value in warmup.head.state_dict().items():
        torch.testing.assert_close(model.head.state_dict()[key], value, rtol=0, atol=0)
    groups = optimizer_groups(model, config)
    assert {group["name"] for group in groups} == {"head", "language", "merger", "vision"}
    question = SimpleNamespace(inputs={"input_ids": torch.tensor([[7, 5, 8, 9, 10, 11]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long), "pixel_values": torch.randn(4, 1536),
        "image_grid_thw": torch.tensor([[1, 2, 2]]), "mm_token_type_ids": torch.tensor([[0, 1, 0, 0, 0, 0]])},
        category_spans=[(3, 4)], decision_position=5, grid=(1, 1), kind="gui", categories=["target"],
        target={"boxes": [[.1, .2, .7, .8]], "labels": [0]})
    before = {key: p.detach().clone() for key, p in model.named_parameters()}
    loss, _ = SpatialObjective({})(model(question), question)
    loss.backward()
    optimizer = torch.optim.AdamW(groups)
    optimizer.step()
    for prefix in ("head.", "backbone.language_model.", "backbone.visual.patch_embed.",
                   "backbone.visual.blocks.", "backbone.visual.merger."):
        assert any(name.startswith(prefix) and p.grad is not None and p.grad.abs().sum() > 0
                   and not torch.equal(p, before[name]) for name, p in model.named_parameters()), prefix
    save_checkpoint(tmp_path / "full", model, optimizer, config, {"step": 1}, random.Random(42))
    restored = build_model(config)
    restored_optimizer = torch.optim.AdamW(optimizer_groups(restored, config))
    load_checkpoint(tmp_path / "full", restored, restored_optimizer)
    with torch.no_grad():
        torch.testing.assert_close(restored.eval()(question).boxes, model.eval()(question).boxes, rtol=0, atol=0)
