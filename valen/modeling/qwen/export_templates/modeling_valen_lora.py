"""Original base + separate trained deltas. / 原始基座与未融合的训练权重。"""
import hashlib
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import Qwen3_5ForConditionalGeneration

from .configuration_valen import ValenQwenConfig
from .modeling_valen import ValenQwenForDecisionMaking
# Explicit dependencies keep local HF dynamic loading portable.
# 显式列出依赖，保证 HF 本地动态加载能复制完整的运行代码。
from .heads import build_head
from .compiler import Compiler
from .runtime_model import ValenQwen
from .responses import answer
from .decisions import to_internal, to_public, local_images
from .batching import collate_branches
from .schema import candidates
from .data_types import QuestionSpec


def _verify_local_base(base_path, expected):
    if not Path(base_path).is_dir():
        return
    for filename, digest in expected.items():
        path = Path(base_path) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Original base shard is required: {path}")
        actual = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                actual.update(chunk)
        if actual.hexdigest() != digest:
            raise ValueError(f"Base weights differ from the trained revision: {filename}")


class ValenQwenWithLoRA(ValenQwenForDecisionMaking):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None,
                        base_model_path=None, device=None, dtype=None, attn_implementation=None,
                        is_trainable=False, local_files_only=True, output_loading_info=False,
                        verify_base=True, **kwargs):
        if model_args:
            raise TypeError("Pass loader options as named arguments")
        folder = Path(pretrained_model_name_or_path).resolve(strict=True)
        if config is None:
            config = ValenQwenConfig.from_pretrained(folder, local_files_only=True)
        base = str(base_model_path or config.valen_base["repo"])
        dtype = dtype or kwargs.pop("torch_dtype", None) or config.dtype or torch.bfloat16
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype != torch.bfloat16:
            raise ValueError("Use dtype=torch.bfloat16 to reproduce this adapter bundle")
        attention = attn_implementation or config._attn_implementation or "sdpa"
        if kwargs.pop("device_map", None) is not None:
            raise ValueError("Use device=... or model.to(...) for this adapter bundle")
        # AutoModel also supplies flags for loading the bundle's custom code.
        # AutoModel 会传入自定义代码标记，它们不应改变原始基座的加载配置。
        for name in ("trust_remote_code", "_from_auto", "_commit_hash", "adapter_kwargs", "revision", "subfolder"):
            kwargs.pop(name, None)
        base_options = {name:kwargs.pop(name) for name in
                        ("cache_dir", "token", "force_download", "proxies", "low_cpu_mem_usage") if name in kwargs}
        if kwargs:
            raise TypeError(f"Unsupported loader options: {sorted(kwargs)}")
        if verify_base:
            _verify_local_base(base, config.valen_base["weight_sha256"])
        # Load the published conditional model first to preserve language prefixes.
        # 先完整加载官方模型，再取出主干，避免语言参数前缀被错误映射。
        container, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            base, revision=config.valen_base["revision"], dtype=dtype,
            attn_implementation=attention, local_files_only=local_files_only,
            output_loading_info=True, **base_options)
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Incomplete original base load: {loading}")
        backbone = container.model
        del container
        backbone.requires_grad_(False)
        backbone.language_model = PeftModel.from_pretrained(
            backbone.language_model, folder / "lora", is_trainable=is_trainable,
            autocast_adapter_dtype=True, torch_device="cpu", local_files_only=True)
        backbone.visual.merger.load_state_dict(load_file(folder / "visual_merger.safetensors"), strict=True)
        head = build_head(config.text_config.hidden_size, config.valen_head).float()
        head.load_state_dict(load_file(folder / "mixer.safetensors"), strict=True)
        config._name_or_path = str(folder)
        model = cls(config, backbone=backbone, head=head)
        model.head.requires_grad_(is_trainable)
        model.backbone.visual.merger.requires_grad_(is_trainable)
        if device is not None:
            model.to(device)
        model.eval()
        return (model, loading) if output_loading_info else model
