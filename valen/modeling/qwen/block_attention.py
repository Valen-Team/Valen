"""Block-structured full attention for Qwen3.5 (issue #40 and Valen-next ablations).

分块注意力：只作用于 full-attention 层，GatedDeltaNet 线性注意力层保持因果。
每行有一个共享前缀 [0, p)（编译器的 context_span，即 state），其后为题目区。
前缀内部双向、永远看不到题目区，因此前缀 KV 仍可在多题间共享。题目区按掩码模式：

- ``block_bidirectional``：题目区整体一块，块内双向并看到完整前缀（Transformers block_sequence_ids 语义）。
- ``prefix_lm``：题目区因果，看到完整前缀（前缀 LM）。
- ``question_blocks``：每题（题面 + 读出位置）一块，块内双向，只看到前缀、公共胶水 token 和本题。

块边界通过 backbone 的 kwargs 传入（``valen_prefix_lengths``，``question_blocks`` 另需
``valen_segments``），gradient checkpointing 反向重算时会收到同一参数，不依赖全局状态。
"""
import torch
import torch.nn.functional as F

ATTENTION_NAME = "valen_block_bidirectional"
MASKS = ("causal", "block_bidirectional", "prefix_lm", "question_blocks")
# Segment ids: prefix, shared suffix glue, padding; questions are 1, 2, ...
PREFIX, COMMON, PADDING = -1, 0, -2
_registered = False


def _register():
    global _registered
    if _registered:
        return
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask
    AttentionInterface.register(ATTENTION_NAME, block_attention_forward)
    # Only the 2D padding mask is needed; blocks come from the kwargs.
    # 只需要 2D padding mask，块结构由 kwargs 给出。
    AttentionMaskInterface.register(ATTENTION_NAME, flash_attention_mask)
    _registered = True


def enable(backbone):
    """Route the language full-attention layers only; vision keeps its backend."""
    _register()
    text = backbone.config.text_config
    if text._attn_implementation != ATTENTION_NAME:
        backbone._valen_kernel = text._attn_implementation
        text._attn_implementation = ATTENTION_NAME


def forward_kwargs(mask, branches, device):
    """Backbone kwargs describing each row's blocks. / 每行块结构。"""
    if mask == "causal":
        return {}
    kwargs = {"valen_prefix_lengths": torch.tensor([b.context_span[1] for b in branches], device=device),
              "valen_mask_mode": mask}
    if mask == "question_blocks":
        if any(getattr(b, "segments", None) is None for b in branches):
            raise ValueError("question_blocks requires compiler segment ids (shared_state execution)")
        length = max(b.segments.numel() for b in branches)
        rows = [F.pad(b.segments, (0, length - b.segments.numel()), value=PADDING) for b in branches]
        kwargs["valen_segments"] = torch.stack(rows).to(device)
    return kwargs


def _flash_available(query):
    if query.device.type != "cuda":
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def _segments(lengths):
    return F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))


def block_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                            valen_prefix_lengths=None, valen_mask_mode="block_bidirectional",
                            valen_segments=None, **kwargs):
    # query/key/value: [B, H, L, D]; attention_mask: None or 2D right-padding mask.
    batch, _, length, _ = query.shape
    if valen_prefix_lengths is None:
        raise ValueError("Block attention requires valen_prefix_lengths")
    prefix = valen_prefix_lengths.to(device=query.device, dtype=torch.long).reshape(-1)
    if prefix.numel() != batch:
        raise ValueError("valen_prefix_lengths must give one prefix length per row")
    if attention_mask is None:
        lengths = torch.full((batch,), length, device=query.device, dtype=torch.long)
    else:
        if attention_mask.ndim != 2:
            raise ValueError("Block attention expects a 2D padding mask")
        lengths = attention_mask.to(query.device).long().sum(-1)
    if bool(((prefix < 1) | (prefix > lengths)).any()):
        raise ValueError("Prefix lengths must lie within each row")
    if valen_mask_mode == "question_blocks":
        if valen_segments is None:
            raise ValueError("question_blocks requires valen_segments")
        segments = valen_segments.to(query.device)[:, :length]
        return _dense(query, key, value, segment_mask(segments, prefix, lengths), dropout, scaling), None
    if valen_mask_mode not in ("block_bidirectional", "prefix_lm"):
        raise ValueError(f"Unknown block mask mode: {valen_mask_mode!r}")
    causal_suffix = valen_mask_mode == "prefix_lm"
    if _flash_available(query):
        return _flash(query, key, value, prefix, lengths, dropout, scaling, causal_suffix), None
    return _dense(query, key, value, block_mask(prefix, lengths, length, causal_suffix), dropout, scaling), None


def _flash(query, key, value, prefix, lengths, dropout, scaling, causal_suffix=False):
    from flash_attn import flash_attn_varlen_func
    dtype = query.dtype
    if dtype == torch.float32:
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else torch.bfloat16
    q, k, v = (x.transpose(1, 2).to(dtype) for x in (query, key, value))  # [B, L, H, D]
    positions = torch.arange(q.shape[1], device=q.device)
    valid = positions < lengths[:, None]
    head = positions < prefix[:, None]
    tail = valid & ~head
    suffix = lengths - prefix
    output = q.new_zeros(q.shape)
    # Prefix block: bidirectional over its own tokens only. / 前缀块：块内双向，看不到后缀。
    cu = _segments(prefix)
    output[head] = flash_attn_varlen_func(
        q[head], k[head], v[head], cu, cu, int(prefix.max()), int(prefix.max()),
        dropout_p=dropout, softmax_scale=scaling, causal=False)
    # Suffix block sees the full prefix; causal masks align bottom-right, i.e. query i
    # of the suffix sees keys up to prefix + i. / 后缀看完整前缀；因果时右下对齐。
    if bool((suffix > 0).any()):
        output[tail] = flash_attn_varlen_func(
            q[tail], k[valid], v[valid], _segments(suffix), _segments(lengths),
            int(suffix.max()), int(lengths.max()), dropout_p=dropout, softmax_scale=scaling,
            causal=causal_suffix)
    return output


def block_mask(prefix, lengths, length, causal_suffix=False):
    """Boolean [B, 1, L, L] reference mask: True means the key is visible."""
    positions = torch.arange(length, device=prefix.device)
    query, key = positions[None, :, None], positions[None, None, :]
    valid_key = key < lengths[:, None, None]
    in_prefix = query < prefix[:, None, None]
    suffix_rule = (key <= query) if causal_suffix else valid_key
    visible = torch.where(in_prefix, key < prefix[:, None, None], suffix_rule)
    return (visible & valid_key)[:, None]


def segment_mask(segments, prefix, lengths):
    """Boolean [B, 1, L, L]: prefix sees prefix; a question sees prefix, glue and itself."""
    length = segments.shape[1]
    positions = torch.arange(length, device=segments.device)
    valid_key = positions[None, None, :] < lengths[:, None, None]
    key_prefix = positions[None, None, :] < prefix[:, None, None]
    query_prefix = positions[None, :, None] < prefix[:, None, None]
    sq, sk = segments[:, :, None], segments[:, None, :]
    suffix_rule = key_prefix | (sk == COMMON) | (sk == sq)
    visible = torch.where(query_prefix, key_prefix, suffix_rule)
    # Padded queries keep one visible key so softmax stays finite. / 保证 padding 行有限。
    visible = visible & valid_key
    visible[:, :, 0] |= ~visible.any(-1)
    return visible[:, None]


def _dense(query, key, value, mask, dropout, scaling):
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask.to(query.device),
                                            dropout_p=dropout, scale=scaling)
    return output.transpose(1, 2).contiguous()
