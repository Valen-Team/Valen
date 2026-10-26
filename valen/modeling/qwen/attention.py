"""Qwen attention selection and explicit FlashAttention prerequisites."""
import importlib

import torch

ATTENTION_IMPLEMENTATIONS = ("eager", "sdpa", "flash_attention_2")


def attention_implementation(config):
    kind = config.get("attn_implementation", "eager")
    if not isinstance(kind, str) or kind not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError(f"attn_implementation must be one of {ATTENTION_IMPLEMENTATIONS}, got {kind!r}")
    if kind != "flash_attention_2":
        return kind
    # 尽早检查，避免加载大模型后失败。 / Check before loading large weights.
    if config.get("dtype", "bf16") != "bf16":
        raise ValueError("Valen FlashAttention-2 requires dtype='bf16'; use eager or sdpa for fp32")
    device = torch.device(config.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("FlashAttention-2 requires an available CUDA GPU; use eager or sdpa on CPU")
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("FlashAttention-2 requires an Ampere or newer CUDA GPU (compute capability >= 8.0)")
    try:
        package = importlib.import_module("flash_attn")
        from packaging.version import Version
        if Version(package.__version__) < Version("2.3.3"):
            raise ImportError("flash-attn >= 2.3.3 is required")
    except (ImportError, OSError) as exc:
        # 不自动下载或回退，以免误记实验后端。 / Never silently replace the requested backend.
        raise ImportError("Cannot load FlashAttention-2. Install a flash-attn build matching PyTorch/CUDA; "
                          "see https://valen-team.github.io/docs/valen/configuration/ for attention settings.") from exc
    return kind


def with_attention_implementation(config, override):
    """推理可覆盖旧 checkpoint 的后端。 / Override without rewriting a checkpoint."""
    if override is None:
        return config
    if config.get("architecture", "qwen") != "qwen":
        raise ValueError("--attn-implementation is supported only for architecture='qwen'")
    if override not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError(f"Unsupported attn_implementation: {override!r}")
    return dict(config, attn_implementation=override)
