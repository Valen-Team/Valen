"""Read published Valen checkpoints without executing checkpoint remote code."""

import torch
from transformers import Qwen3_5Config


class ValenQwenConfig(Qwen3_5Config):
    model_type = "valen_qwen"

    def __init__(
        self,
        valen_head=None,
        qwen_execution="shared_state",
        valen_max_length=32768,
        valen_media_kwargs=None,
        valen_backbone_autocast_dtype=None,
        valen_full_attention_mask="causal",
        valen_grounding=None,
        valen_decision_version=None,
        **kwargs,
    ):
        self.valen_head = valen_head or {"head_type": "mixer"}
        self.qwen_execution = qwen_execution
        self.valen_max_length = valen_max_length
        self.valen_media_kwargs = valen_media_kwargs or {}
        self.valen_backbone_autocast_dtype = valen_backbone_autocast_dtype
        self.valen_full_attention_mask = valen_full_attention_mask
        self.valen_grounding = valen_grounding
        # Bundles exported before the generation label derive it from the two settings.
        self.valen_decision_version = valen_decision_version or (
            "1.5" if valen_full_attention_mask != "causal" or valen_grounding else "1.0"
        )
        super().__init__(**kwargs)
        # Stored weights may be FP32; this setting is the trained execution dtype.
        dtype = (
            torch.bfloat16
            if valen_backbone_autocast_dtype == "bfloat16"
            else torch.float32
        )
        self.dtype = dtype
        self.text_config.dtype = dtype
