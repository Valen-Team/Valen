"""Standalone Qwen decision configuration. / 独立的 Qwen 决策模型配置。"""
from transformers import Qwen3_5Config


class ValenQwenConfig(Qwen3_5Config):
    model_type = "valen_qwen"

    def __init__(self, valen_head=None, qwen_execution="shared_state", valen_max_length=32768,
                 valen_media_kwargs=None, valen_backbone_autocast_dtype=None,
                 valen_full_attention_mask="causal", valen_grounding=None, **kwargs):
        if qwen_execution not in {"question", "shared_state"}:
            raise ValueError("qwen_execution must be question or shared_state")
        self.valen_head = valen_head or {"head_type": "mixer"}
        self.qwen_execution = qwen_execution
        self.valen_max_length = valen_max_length
        self.valen_media_kwargs = valen_media_kwargs or {}
        if valen_backbone_autocast_dtype not in {None, "bfloat16"}:
            raise ValueError("valen_backbone_autocast_dtype must be None or bfloat16")
        self.valen_backbone_autocast_dtype = valen_backbone_autocast_dtype
        from .block_attention import MASKS
        if valen_full_attention_mask not in MASKS:
            raise ValueError(f"valen_full_attention_mask must be one of {MASKS}")
        self.valen_full_attention_mask = valen_full_attention_mask
        self.valen_grounding = valen_grounding
        super().__init__(**kwargs)
