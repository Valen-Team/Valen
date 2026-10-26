"""Single architecture dispatch point for model, compiler and execution policy."""
from .qwen.backend import QwenBackend
from .dual_encoder.backend import DualEncoderBackend
from .qwen.spatial_backend import SpatialBackend
from valen.configuration import flatten_config

BACKENDS = {backend.name: backend for backend in (QwenBackend(), DualEncoderBackend(), SpatialBackend())}


def get_backend(architecture="qwen"):
    if architecture not in BACKENDS:
        raise ValueError(f"Unsupported architecture: {architecture!r}")
    return BACKENDS[architecture]


def backend_for_model(model):
    # Models used by legacy callers without an architecture tag follow Qwen's
    # existing per-question contract; no method-presence probing is required.
    return get_backend(getattr(model, "architecture", "qwen"))


def normalize_model_config(config):
    config = flatten_config(config)
    get_backend(config.setdefault("architecture", "qwen"))
    if config["architecture"] == "qwen":
        from .qwen.pretrained import with_published_settings
        config = with_published_settings(config)
        finetuning = config.setdefault("finetuning_type", "lora")
        if finetuning not in ("lora", "full"):
            raise ValueError("finetuning_type must be lora or full")
        if finetuning == "full" and config.get("stage", "joint") not in ("warmup", "joint"):
            raise ValueError("Full Qwen training supports warmup and joint stages")
        # 补齐默认值，旧 ckpt 和显式 question 配置等价。 / Canonicalize old and explicit defaults.
        execution = config.setdefault("qwen_execution", "question")
        if execution == "share_state":
            config["qwen_execution"] = "shared_state"
        if config["qwen_execution"] not in ("question", "shared_state"):
            raise ValueError("qwen_execution must be question or shared_state")
    elif config["architecture"] == "qwen_spatial":
        from .qwen.spatial import spatial_signature
        spatial_signature(config)
        if config.setdefault("finetuning_type", "lora") not in ("lora", "full"):
            raise ValueError("finetuning_type must be lora or full")
        if "qwen_execution" in config:
            raise ValueError("Spatial readouts do not use qwen_execution")
    elif "qwen_execution" in config:
        raise ValueError("qwen_execution requires architecture=qwen")
    return config


def build_model(config):
    config = normalize_model_config(config)
    return get_backend(config["architecture"]).build_model(config)


def build_compiler(config, media_root="."):
    config = normalize_model_config(config)
    return get_backend(config["architecture"]).build_compiler(config, media_root)


def optimizer_groups(model, config):
    config = normalize_model_config(config)
    return get_backend(config["architecture"]).optimizer_groups(model, config)
