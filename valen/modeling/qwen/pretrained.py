"""Read published Valen settings without parsing them as a native Qwen config."""
from copy import deepcopy
import json
from pathlib import Path

from .heads import head_signature


def published_config(model_path):
    if model_path is None:
        return None
    path = Path(model_path) / "config.json"
    if not path.is_file():
        return None
    settings = json.loads(path.read_text(encoding="utf-8"))
    if settings.get("model_type") != "valen_qwen":
        return None
    loader = settings.get("auto_map", {}).get("AutoModel", "")
    if loader.endswith("ValenQwenWithLoRA"):
        raise ValueError("Fine-tuning requires a complete Valen model; download the main revision or merge the adapters first")
    return settings


def with_published_settings(config):
    """Inherit missing settings and reject incompatible heads. / 继承配置并拒绝重建决策头。"""
    settings = published_config(config.get("model_path"))
    if settings is None:
        return config
    source = head_signature(settings.get("valen_head") or {"head_type": "mixer"})
    requested = head_signature(dict(source, **{key: config[key] for key in source if key in config}))
    if requested != source:
        raise ValueError(f"Published Valen head mismatch: saved={source}, requested={requested}")
    config = dict(config)
    for key, value in source.items():
        config.setdefault(key, value)
    for field, saved, default in (
        ("qwen_execution", "qwen_execution", "shared_state"),
        ("max_length", "valen_max_length", 32768),
        ("media_kwargs", "valen_media_kwargs", {}),
    ):
        config.setdefault(field, deepcopy(settings.get(saved, default)))
    # Valen-Decision-1.5 架构随发布权重继承。 / The 1.5 architecture follows the published weights.
    for field, saved in (("full_attention_mask", "valen_full_attention_mask"), ("grounding_head", "valen_grounding")):
        if settings.get(saved) is not None:
            config.setdefault(field, deepcopy(settings[saved]))
    return config
