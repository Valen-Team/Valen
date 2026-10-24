"""Install Valen adapters through SGLang's supported general-plugin hooks."""

_registered = False

HOOKS = (
    (
        "sglang.srt.configs.model_config._get_and_verify_dtype",
        "resolve_dtype",
        "AROUND",
    ),
    (
        "sglang.srt.utils.hf_transformers.config.HfModelConfigParser.parse",
        "normalize_config",
        "AFTER",
    ),
    (
        "sglang.srt.configs.model_config.ModelConfig.__init__",
        "model_config_ready",
        "AFTER",
    ),
    (
        "sglang.srt.arg_groups.model_hook.handle_model_capability_adjustments",
        "configure_capabilities",
        "AROUND",
    ),
    (
        "sglang.srt.models.qwen3_5.Qwen3_5ForConditionalGeneration.__init__",
        "initialize_head",
        "AFTER",
    ),
    (
        "sglang.srt.models.qwen3_5.Qwen3_5ForConditionalGeneration.load_weights",
        "load_weights",
        "AROUND",
    ),
    (
        "sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor.__init__",
        "prepare_image_processor",
        "BEFORE",
    ),
    (
        "sglang.srt.managers.tokenizer_manager.TokenizerManager._validate_joint_schema_request",
        "validate_request",
        "AROUND",
    ),
)
CLASS_HOOKS = (
    "sglang.srt.entrypoints.systemone.protocol.SystemOneRequest",
    "sglang.srt.entrypoints.systemone.serving.SystemOneServing",
)


def register():
    global _registered
    if _registered:
        return
    from transformers import AutoConfig

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from valen_sglang import hooks
    from valen_sglang.configuration import ValenQwenConfig
    from valen_sglang.protocol import SystemOneRequest
    from valen_sglang.serving import ValenSystemOneServing

    AutoConfig.register(ValenQwenConfig.model_type, ValenQwenConfig, exist_ok=True)
    HookRegistry.register(CLASS_HOOKS[0], SystemOneRequest, HookType.REPLACE)
    HookRegistry.register(CLASS_HOOKS[1], ValenSystemOneServing, HookType.REPLACE)
    for target, function, kind in HOOKS:
        HookRegistry.register(target, getattr(hooks, function), getattr(HookType, kind))
    _registered = True
