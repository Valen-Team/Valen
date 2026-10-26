"""The standalone runtime preserves mixed precision and native Qwen image readouts."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import pytest
from transformers import AutoModel, Qwen3_5Config

from scripts.export.qwen.export_spatial import bundle_runtime


@pytest.mark.parametrize("mode", ["lora", "full", "lora_rlcd"])
def test_standalone_spatial_runtime_reload(tmp_path, monkeypatch, mode):
    full = mode == "full"
    mixed = mode == "lora_rlcd"
    import transformers.dynamic_module_utils as dynamic_modules
    monkeypatch.setattr(dynamic_modules, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    bundle_runtime(Path(__file__).resolve().parents[2], tmp_path)
    spec = importlib.util.spec_from_file_location("modeling_valen_spatial", tmp_path / "modeling_valen_spatial.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    base = Qwen3_5Config(text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=["full_attention"], rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                                        "mrope_section": [2, 1, 1]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
            out_hidden_size=32, patch_size=16, spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=16),
        image_token_id=5, video_token_id=6, vision_start_token_id=7, vision_end_token_id=8)
    # Official base snapshots advertise BF16 even when full SFT keeps FP32 master weights.
    base.text_config.dtype = base.vision_config.dtype = torch.bfloat16
    config = module.ValenSpatialConfig(backbone_config=base.to_dict(), spatial_settings=dict(task="det",
        spatial_width=32, spatial_heads=4, spatial_queries=3, spatial_layers=1, coordinate_bins=32),
        backbone_autocast_dtype="bfloat16" if full or mixed else None,
        backbone_storage_dtype="float32" if full else "bfloat16", merger_fp32=mixed)
    model = module.ValenSpatialForDecisionMaking(config)
    if not full:
        model.backbone.bfloat16()
    if mixed:
        model.backbone.visual.merger.float()
    config.register_for_auto_class()
    model.register_for_auto_class("AutoModel")
    model.save_pretrained(tmp_path / "model")
    loaded, info = AutoModel.from_pretrained(tmp_path / "model", trust_remote_code=True,
        local_files_only=True, dtype="auto", output_loading_info=True)
    assert not any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    assert next(loaded.head.parameters()).dtype == torch.float32
    assert {p.dtype for p in loaded.backbone.parameters()} == ({torch.float32, torch.bfloat16} if mixed else
                                                            {torch.float32 if full else torch.bfloat16})
    assert loaded.backbone_autocast_dtype == (torch.bfloat16 if full or mixed else None)
    if mixed:
        for key, value in model.backbone.visual.merger.state_dict().items():
            assert torch.equal(value, loaded.backbone.visual.merger.state_dict()[key])
    for key, value in model.head.state_dict().items():
        assert torch.equal(value, loaded.head.state_dict()[key])
    question = SimpleNamespace(inputs={"input_ids": torch.tensor([[7, 5, 8, 9, 10, 11]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long), "pixel_values": torch.randn(4, 1536),
        "image_grid_thw": torch.tensor([[1, 2, 2]]), "mm_token_type_ids": torch.tensor([[0, 1, 0, 0, 0, 0]])},
        category_spans=[(3, 4)], decision_position=5, grid=(1, 1))
    with torch.inference_mode():
        expected = model.eval()(question)
        result = loaded.eval()(question)
    assert result.boxes.shape == (3, 4)
    assert torch.isfinite(result.class_logits).all()
    torch.testing.assert_close(result.class_logits, expected.class_logits, rtol=0, atol=0)
    torch.testing.assert_close(result.boxes, expected.boxes, rtol=0, atol=0)
