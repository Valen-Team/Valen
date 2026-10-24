"""Out-of-tree registration; safe to call repeatedly in all vLLM processes."""


def register():
    from transformers import AutoConfig
    from vllm import ModelRegistry

    from valen_vllm.configuration import ValenQwenConfig

    AutoConfig.register("valen_qwen", ValenQwenConfig, exist_ok=True)
    if "ValenQwenForDecisionMaking" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "ValenQwenForDecisionMaking", "valen_vllm.model:ValenQwenForDecisionMaking"
        )
