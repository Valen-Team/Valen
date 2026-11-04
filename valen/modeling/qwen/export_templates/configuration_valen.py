"""Standalone Qwen decision configuration. / 独立的 Qwen 决策模型配置。"""
from transformers import Qwen3_5Config


class ValenQwenConfig(Qwen3_5Config):
    model_type = "valen_qwen"

    def __init__(self, valen_head=None, qwen_execution="shared_state", valen_max_length=32768,
                 valen_media_kwargs=None, valen_backbone_autocast_dtype=None,
                 valen_full_attention_mask="causal", valen_grounding=None,
                 valen_decision_version=None, **kwargs):
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
        # Generation label. Bundles exported before it existed carry no field, so derive it from
        # the two settings that distinguish the generations rather than assuming a default.
        # 代际标签。更早导出的包没有这个字段，按区分代际的两项设置推导，而不是假定默认值。
        if valen_decision_version is None:
            valen_decision_version = "1.5" if (valen_full_attention_mask != "causal" or valen_grounding) else "1.0"
        self.valen_decision_version = valen_decision_version
        super().__init__(**kwargs)
