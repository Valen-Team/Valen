"""Small, Valen-gated hooks; no engine source replacement or on-disk patching."""

import logging

import torch

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolving_view,
)
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.layers.joint_schema_head import max_joint_prompt_tokens
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_schedule
from valen_sglang.blocks import block_attention
from valen_sglang.grounding import GroundingHead
from valen_sglang.head import (
    ValenMixerHead,
    ValenPooler,
    parse_valen_layout,
    valen_segments,
)

logger = logging.getLogger(__name__)


def normalize_config(config, *args, **kwargs):
    if getattr(config, "architectures", None) != ["ValenQwenForDecisionMaking"]:
        return config
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    config.model_type = "qwen3_5"
    # Without autocast the backbone runs in its stored dtype, as model.predict does.
    stored = str(getattr(config, "dtype", None) or "float32").removeprefix("torch.")
    dtype = config.valen_backbone_autocast_dtype or stored
    config.text_config.dtype = getattr(torch, dtype)
    config.text_config._valen_backbone_dtype = dtype
    return config


def resolve_dtype(original, config, dtype):
    # Stock auto dtype may downcast FP32 checkpoints. Valen's stored autocast
    # setting, including no autocast, is part of its trained execution contract.
    if dtype == "auto" and hasattr(config, "_valen_backbone_dtype"):
        dtype = config._valen_backbone_dtype
    return original(config, dtype)


def model_config_ready(result, self, *args, **kwargs):
    self.valen_head_config = getattr(self.hf_config, "valen_head", None)


def configure_capabilities(original, server_args):
    result = original(server_args)
    model_config = model_config_of(server_args)
    if getattr(model_config, "valen_head_config", None) is None:
        return result
    cfg = resolving_view(server_args)
    for key, value in (
        ("is_embedding", True),
        ("disable_radix_cache", True),
        ("chunked_prefill_size", -1),
        ("skip_server_warmup", True),
    ):
        declare_resolution(
            server_args, "_handle_model_capability_adjustments", **{key: value}
        )
    for phase in (Phase.DECODE, Phase.PREFILL):
        declare_resolution(
            server_args,
            "_handle_model_capability_adjustments",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, phase, backend=Backend.DISABLED
            ),
        )
    logger.info(
        "Valen plugin: one complete prefill, FP32 head, no decode/cache/chunking/CUDA graphs"
    )
    return result


def initialize_head(result, self, config, *args, **kwargs):
    if getattr(config, "valen_head", None) is None:
        return
    hidden_size = config.text_config.hidden_size
    grounding = config.valen_grounding
    self.pooler = ValenPooler(
        ValenMixerHead(hidden_size, config.valen_head),
        GroundingHead(hidden_size, grounding["width"], grounding["hidden"])
        if grounding
        else None,
    )
    mask = config.valen_full_attention_mask
    if mask == "prefix_lm":
        raise ValueError("valen_full_attention_mask=prefix_lm is not served by SGLang")
    if mask != "causal":
        # Valen-Decision-1.5: the full-attention layers follow the request's blocks.
        for layer in self.model.layers:
            if isinstance(getattr(layer, "attn", None), RadixAttention):
                layer.attn.valen_blocks = True


def block_attention_forward(original, self, q, k, v, forward_batch, *args, **kwargs):
    layouts = getattr(self, "valen_blocks", False) and forward_batch.decision_layouts
    if not layouts or not forward_batch.forward_mode.is_extend():
        return original(self, q, k, v, forward_batch, *args, **kwargs)
    # One complete prefill per request, so the KV cache is neither read nor written.
    lengths = forward_batch.extend_seq_lens_cpu
    segments = getattr(forward_batch, "valen_segments", None)
    if segments is None:
        segments = forward_batch.valen_segments = [
            layout and valen_segments(layout, length)
            for layout, length in zip(layouts, lengths)
        ]
    q = q.view(-1, self.tp_q_head_num, self.qk_head_dim)
    k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
    v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
    outputs, offset = [], 0
    for blocks, length in zip(segments, lengths):
        rows = slice(offset, offset + length)
        offset += length
        if blocks is not None:
            blocks = blocks.to(q.device)
        outputs.append(block_attention(q[rows], k[rows], v[rows], blocks, self.scaling))
    return torch.cat(outputs).reshape(-1, self.tp_q_head_num * self.v_head_dim)


def load_weights(original, self, weights, *args, **kwargs):
    if not isinstance(self.pooler, ValenPooler):
        return original(self, weights, *args, **kwargs)
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    params = {"head." + k: v for k, v in self.pooler.head.named_parameters()}
    if self.pooler.grounding is not None:
        params.update(
            ("grounding." + k, v) for k, v in self.pooler.grounding.named_parameters()
        )
    loaded = set()

    def backbone_weights():
        for name, weight in weights:
            if name.startswith(("head.", "grounding.")):
                default_weight_loader(params[name], weight)
                loaded.add(name)
            elif name.startswith("backbone."):
                yield "model." + name.removeprefix("backbone."), weight
            else:
                yield name, weight

    result = original(self, backbone_weights(), *args, **kwargs)
    if isinstance(result, set):
        result.update(f"pooler.{name}" for name in loaded)
    return result


def prepare_image_processor(self, hf_config, server_args, _processor, *args, **kwargs):
    if getattr(hf_config, "valen_head", None) is not None:
        image_kwargs = hf_config.valen_media_kwargs.get("images_kwargs", {})
        if "size" in image_kwargs:
            _processor.image_processor.size = dict(image_kwargs["size"])


def validate_request(original, self, obj, input_ids):
    if getattr(self.model_config, "valen_head_config", None) is None:
        return original(self, obj, input_ids)
    if isinstance(obj, GenerateReqInput) or obj.decision_layout is None:
        if (
            isinstance(obj, EmbeddingReqInput)
            and isinstance(obj.rid, str)
            and obj.rid.startswith(HEALTH_CHECK_RID_PREFIX)
        ):
            return
        raise ValueError("This checkpoint answers Valen decisions; use /v1/systemone or /v1/decisions")
    limit = min(
        self.model_config.hf_config.valen_max_length,
        max_joint_prompt_tokens(
            context_len=self.context_len,
            num_reserved_tokens=self.num_reserved_tokens,
            max_req_input_len=self.max_req_input_len,
            max_prefill_tokens=get_schedule().max_prefill_tokens,
        ),
    )
    if len(input_ids) > limit:
        raise ValueError(
            f"The prompt has {len(input_ids)} tokens; Valen permits at most {limit}"
        )
    parse_valen_layout(obj.decision_layout, len(input_ids))
