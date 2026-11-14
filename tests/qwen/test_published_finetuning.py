"""Continue training portable Valen weights without rebuilding their Mixer."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from valen.data.compilers.qwen import Branch, Question
from valen.modeling.factory import build_compiler, build_model, normalize_model_config, optimizer_groups
from valen.training.checkpoint import file_sha256, load_checkpoint, save_checkpoint
from valen.training.sft import question_loss


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    from transformers import Qwen3_5Config, Qwen3VLProcessor, Qwen2VLImageProcessor, Qwen3VLVideoProcessor, PreTrainedTokenizerFast
    import transformers.dynamic_module_utils as dynamic
    from tokenizers import Tokenizer, models, pre_tokenizers
    from valen.modeling.qwen.hf_export import bundle_runtime

    root = tmp_path_factory.mktemp("published-valen")
    with pytest.MonkeyPatch.context() as patch:
        # Another test may have imported a different temporary HF cache. / 隔离前序测试的动态模块缓存。
        for name in list(sys.modules):
            if name == "transformers_modules" or name.startswith("transformers_modules."):
                patch.delitem(sys.modules, name)
        patch.setattr(dynamic, "HF_MODULES_CACHE", str(root / "module_cache"))
        package = root / "finetune_runtime"
        bundle_runtime(Path(__file__).resolve().parents[2], package)
        patch.syspath_prepend(str(root))
        configuration = importlib.import_module("finetune_runtime.configuration_valen")
        modeling = importlib.import_module("finetune_runtime.modeling_valen")
        configuration.ValenQwenConfig.register_for_auto_class()
        modeling.ValenQwenForDecisionMaking.register_for_auto_class("AutoModel")
        base = Qwen3_5Config(
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
                linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                rope_parameters=dict(rope_type="default", rope_theta=10000,
                    partial_rotary_factor=1.0, mrope_section=[2, 3, 3])),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                out_hidden_size=32, patch_size=2, spatial_merge_size=2,
                temporal_patch_size=1, num_position_embeddings=16),
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
        settings = base.to_dict()
        settings.pop("model_type", None)
        settings.update(dtype="float32", valen_head={"head_type": "mixer", "head_width": 8,
            "head_layers": 2, "head_token_hidden_dim": 4, "head_channel_hidden_dim": 16},
            valen_max_length=1024, qwen_execution="shared_state",
            valen_media_kwargs={"videos_kwargs": {"num_frames": 2, "fps": None}})
        torch.manual_seed(13)
        model = modeling.ValenQwenForDecisionMaking(configuration.ValenQwenConfig(**settings)).eval()
        # Distinguish trained head weights from a fresh initialization. / 标记已训练头，检测意外重建。
        with torch.no_grad():
            model.head.projection.bias.add_(0.125)
        folder = root / "complete"
        model.save_pretrained(folder)
        vocab = {"[UNK]": 0, "[PAD]": 1, "<|im_start|>": 2, "<|im_end|>": 3,
            "A": 4, "cat": 5, "dog": 6, "animal": 7, "Which": 8, "user": 9,
            "true": 10, "false": 11, "red": 12, "blue": 13,
            "<|image_pad|>": 60, "<|video_pad|>": 61, "<|vision_start|>": 62, "<|vision_end|>": 63}
        vocab.update({f"unused_{i}": i for i in range(14, 60)})
        tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
            additional_special_tokens=[key for key in vocab if key.startswith("<|")])
        template = ("{% for m in messages %}<|im_start|>{{m['role']}}\n"
            "{% for c in m['content'] %}{% if c['type']=='text' %}{{c['text']}}{% endif %}{% endfor %}"
            "<|im_end|>\n{% endfor %}")
        processor = Qwen3VLProcessor(tokenizer=tokenizer, chat_template=template,
            image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=1, merge_size=2,
                size={"shortest_edge": 64, "longest_edge": 64}),
            video_processor=Qwen3VLVideoProcessor(patch_size=2, temporal_patch_size=1, merge_size=2,
                size={"shortest_edge": 64, "longest_edge": 64}))
        processor.save_pretrained(folder)
        weight_hash = file_sha256(folder / "model.safetensors")
        (folder / "export_manifest.json").write_text(json.dumps({"format": "valen_qwen_full_sft_v1",
            "source_checkpoint_sha256": weight_hash, "files_sha256": {"model.safetensors": weight_hash}}))
        try:
            yield folder, model
        finally:
            for name in list(sys.modules):
                if name == "transformers_modules" or name.startswith("transformers_modules."):
                    sys.modules.pop(name, None)


def options(folder, **kwargs):
    return dict(model_path=str(folder), device="cpu", dtype="fp32", attn_implementation="sdpa",
                stage="joint", finetuning_type="full", gradient_checkpointing=False, **kwargs)


def test_decision_index_engine_preserves_published_predictions(published):
    pytest.importorskip("decision_index")
    from decision_index.engines import Unsupported, load_engine, validate
    from valen.evaluation.inference import predict
    folder, original = published
    engine = load_engine("evaluation.decision_index.engine:ValenEngine", model=str(folder), device="cpu")
    questions = {
        "animal": {"type": "choice", "instructions": {"task": "Which animal"},
                   "criteria": {"cat": {"label": "cat"}, "dog": None}},
        "truth": {"type": "noul", "instructions": "cat"},
    }
    response, _ = engine({"scene": "A cat"}, questions)
    validate(questions, response)
    compiled = engine.compiler.compile({"request": {
        "state": '{"scene":"A cat"}', "questions": {
            "animal": {"type": "choice", "instructions": '{"task":"Which animal"}',
                       "criteria": {"cat": '{"label":"cat"}', "dog": "dog"}},
            "truth": questions["truth"],
        }}})
    expected = predict(original, compiled)["answers"]
    for key, value in expected["animal"]["probabilities"].items():
        assert response["answers"]["animal"]["probabilities"][key] == pytest.approx(value, abs=1e-6)
    assert response["answers"]["truth"]["noul"] == pytest.approx(expected["truth"]["noul"], abs=1e-6)
    assert engine.config["max_length"] == 1024
    assert not any(p.requires_grad for p in engine.model.parameters())
    with pytest.raises(Unsupported, match="exceed max_length"):
        engine("cat " * 2000, questions)
    engine.close()


def state_for(modality):
    if modality == "text":
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        spans = ((0, 2), (2, 3), [(3, 4), (4, 5)])
    else:
        token = 60 if modality == "image" else 61
        ids = torch.tensor([[1, 62, token, token, token, token, 63, 2, 3, 4, 5, 6]])
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                  "mm_token_type_ids": (ids == token).long() * (1 if modality == "image" else 2)}
        pixels = "pixel_values" if modality == "image" else "pixel_values_videos"
        grid = "image_grid_thw" if modality == "image" else "video_grid_thw"
        inputs.update({pixels: torch.randn(16, 12), grid: torch.tensor([[1, 4, 4]])})
        spans = ((0, 8), (8, 9), [(9, 10), (10, 11)])
    context, instruction, candidates = spans
    branch = Branch(inputs, [end - 1 for _, end in candidates], ids.shape[1] - 1, context, instruction, candidates)
    questions = [Question(kind, kind, ["a", "b"], ["a", "b"], [branch], [1., 0.], modality == "text")
                 for kind in ["choice", "noul", "score"]]
    return SimpleNamespace(inputs=inputs, questions=questions)


@pytest.mark.parametrize("modality", ["text", "image", "video"])
def test_published_loading_preserves_every_tensor_and_shared_logits(published, monkeypatch, modality):
    folder, original = published
    from transformers import Qwen3_5ForConditionalGeneration
    def wrong_loader(*args, **kwargs):
        pytest.fail("Published Valen must not be parsed or loaded as a native Qwen generation model")
    monkeypatch.setattr(Qwen3_5ForConditionalGeneration, "from_pretrained", wrong_loader)
    torch.manual_seed(999)
    loaded = build_model(options(folder)).eval()
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
    state = state_for(modality)
    with torch.inference_mode():
        for expected, actual in zip(original.forward_state(state), loaded.forward_state(state)):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["full", "lora", "warmup"])
def test_published_finetuning_updates_expected_parameters_and_restores_optimizer(published, tmp_path, mode):
    import random
    folder, _ = published
    config = options(folder)
    config.update(finetuning_type="lora" if mode == "lora" else "full",
                  stage="warmup" if mode == "warmup" else "joint", lora_rank=2, lora_alpha=4)
    model = build_model(config).train()
    groups = optimizer_groups(model, config)
    assert {g["name"] for g in groups} == ({"head"} if mode == "warmup" else
        {"head", "lora", "merger"} if mode == "lora" else {"head", "language", "vision", "merger"})
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(groups)
    state = state_for("image")
    outputs = model.forward_state(state)
    sum(question_loss(z, q.target, q.kind)[0] for z, q in zip(outputs, state.questions)).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    changed = {name for name, p in model.named_parameters() if not torch.equal(p, before[name])}
    assert any(name.startswith("head.") for name in changed)
    if mode == "warmup":
        assert all(name.startswith("head.") for name in changed)
    elif mode == "lora":
        assert any("lora_B" in name for name in changed)
        assert any(name.startswith("backbone.visual.merger.") for name in changed)
        assert all("lora_" in name or name.startswith(("head.", "backbone.visual.merger.")) for name in changed)
    else:
        assert any(name.startswith("backbone.language_model.") for name in changed)
        assert any(name.startswith("backbone.visual.blocks.") for name in changed)
    config = normalize_model_config(config)
    save_checkpoint(tmp_path, model, optimizer, config, {"step": 1}, random.Random(42))
    restored = build_model(config)
    restored_optimizer = torch.optim.AdamW(optimizer_groups(restored, config))
    payload = load_checkpoint(tmp_path, restored, restored_optimizer)
    assert payload["progress"]["step"] == 1
    assert len(restored_optimizer.state) == len(optimizer.state)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)


def test_settings_are_inherited_recorded_and_incompatible_heads_rejected(published):
    folder, _ = published
    raw = options(folder)
    before = deepcopy(raw)
    normalized = normalize_model_config(raw)
    assert raw == before
    assert normalized["head_width"] == 8 and normalized["head_layers"] == 2
    assert normalized["qwen_execution"] == "shared_state" and normalized["max_length"] == 1024
    assert normalized["media_kwargs"]["videos_kwargs"]["num_frames"] == 2
    assert normalize_model_config(normalized) == normalized
    with pytest.raises(ValueError, match="head mismatch"):
        build_model(dict(raw, head_width=16))
    overridden = normalize_model_config(dict(raw, qwen_execution="question", max_length=512, media_kwargs={}))
    assert overridden["qwen_execution"] == "question" and overridden["max_length"] == 512
    assert overridden["media_kwargs"] == {}


@pytest.fixture
def published_1_5(published, tmp_path):
    """The published fixture re-saved as Valen-Decision-1.5. / 1.5 代发布模型。"""
    import shutil
    folder, original = published
    settings = original.config.to_dict()
    settings.update(valen_full_attention_mask="bbox_blocks", valen_grounding={"width": 8, "hidden": 16})
    settings.pop("valen_decision_version", None)
    torch.manual_seed(17)
    model = type(original)(type(original.config)(**settings)).eval()
    model.load_state_dict(original.state_dict(), strict=False)
    target = tmp_path / "complete-1.5"
    shutil.copytree(folder, target)
    model.save_pretrained(target)
    weight_hash = file_sha256(target / "model.safetensors")
    (target / "export_manifest.json").write_text(json.dumps({"format": "valen_qwen_full_sft_v1",
        "source_checkpoint_sha256": weight_hash, "files_sha256": {"model.safetensors": weight_hash}}))
    return target, model


def test_published_1_5_architecture_is_inherited_with_its_trained_grounding_head(published, published_1_5):
    folder, original = published_1_5
    normalized = normalize_model_config(options(folder))
    assert normalized["full_attention_mask"] == "bbox_blocks"
    assert normalized["grounding_head"] == {"width": 8, "hidden": 16}
    assert normalize_model_config(normalized) == normalized
    loaded = build_model(options(folder))
    assert loaded.full_attention_mask == "bbox_blocks"
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
    assert all(p.requires_grad for p in loaded.grounding.parameters())
    assert "grounding" in {g["name"] for g in optimizer_groups(loaded, normalized)}
    # 显式设置仍然优先。 / Explicit settings still win.
    explicit = normalize_model_config(options(folder, full_attention_mask="causal", grounding_head=None))
    assert explicit["full_attention_mask"] == "causal" and explicit["grounding_head"] is None
    assert build_model(explicit).grounding is None
    # 1.0 发布模型保持原样。 / A published 1.0 model is unchanged.
    legacy = normalize_model_config(options(published[0]))
    assert legacy.get("full_attention_mask", "causal") == "causal" and "grounding_head" not in legacy


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_published_fp32_weights_are_not_rounded_when_selecting_compute_dtype(published, dtype):
    folder, original = published
    config = options(folder)
    config.update(dtype=dtype, stage="warmup")
    loaded = build_model(config)
    assert all(p.dtype == torch.float32 for p in loaded.parameters())
    assert loaded.backbone_autocast_dtype == (torch.bfloat16 if dtype == "bf16" else None)
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["full", "lora"])
def test_real_published_sft_run_resumes_like_continuous_training(published, tmp_path, monkeypatch, mode):
    from valen.training.runner import run
    folder, _ = published
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    record = {"group_id": "example", "request": {"state": "A cat", "questions": {
        "animal": {"type": "choice", "instructions": "Which animal", "criteria": {"cat": "cat", "dog": "dog"}}}},
        "targets": {"animal": {"probabilities": {"cat": 1., "dog": 0.}}}}
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps(record) + '\n')
    config = options(folder)
    config.update(finetuning_type=mode, data=str(data), output=str(tmp_path / "resumed"),
        epochs=2, max_steps=1, tokens_per_step=1024, save_every=1, lora_rank=2, lora_alpha=4)
    compiler = build_compiler(config, tmp_path)
    assert compiler.execution == "shared_state" and compiler.compile(record).questions
    run(config)
    checkpoint = tmp_path / "resumed" / "latest"
    _, _, progress = run(dict(config, max_steps=2), resume=checkpoint)
    assert progress["step"] == 2 and progress["epoch"] == 2
    continuous = dict(config, max_steps=2, output=str(tmp_path / "continuous"))
    run(continuous)
    resumed = torch.load(checkpoint / "checkpoint.pt", weights_only=False)
    expected = torch.load(tmp_path / "continuous" / "latest" / "checkpoint.pt", weights_only=False)
    for key in resumed["weights"]:
        torch.testing.assert_close(resumed["weights"][key], expected["weights"][key], rtol=0, atol=0)
    assert resumed["progress"] == expected["progress"]
    saved_config = json.loads((checkpoint / "config.json").read_text())
    assert saved_config["head_width"] == 8 and saved_config["qwen_execution"] == "shared_state"
