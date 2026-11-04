"""Block-bidirectional full attention for Qwen3.5 (issue #40).

分块双向注意力：只作用于 full-attention 层，GatedDeltaNet 线性注意力层保持因果。
每行分为两块：共享前缀 [0, p)（编译器的 context_span，即 state）与问题后缀 [p, n)。
块内双向、块间因果：前缀内部双向但永远看不到后缀；后缀内部双向并完整看到前缀。
前缀因此仍可在多题间共享 KV。与 Transformers 的 block_sequence_ids 语义一致。
块边界通过 backbone 的 kwargs ``valen_prefix_lengths`` 传入，gradient
checkpointing 反向重算时会收到同一参数，不依赖全局状态。
"""
import torch
import torch.nn.functional as F

ATTENTION_NAME = "valen_block_bidirectional"
MASKS = ("causal", "block_bidirectional")
_registered = False


def _register():
    global _registered
    if _registered:
        return
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, flash_attention_mask
    AttentionInterface.register(ATTENTION_NAME, block_attention_forward)
    # Only the 2D padding mask is needed; blocks come from the prefix lengths.
    # 只需要 2D padding mask，块结构由前缀长度给出。
    AttentionMaskInterface.register(ATTENTION_NAME, flash_attention_mask)
    _registered = True


def enable(backbone):
    """Route the language full-attention layers only; vision keeps its backend."""
    _register()
    text = backbone.config.text_config
    if text._attn_implementation != ATTENTION_NAME:
        backbone._valen_kernel = text._attn_implementation
        text._attn_implementation = ATTENTION_NAME


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
                            valen_prefix_lengths=None, **kwargs):
    # query/key/value: [B, H, L, D]; attention_mask: None or 2D right-padding mask.
    batch, _, length, _ = query.shape
    if valen_prefix_lengths is None:
        raise ValueError("Block-bidirectional attention requires valen_prefix_lengths")
    prefix = valen_prefix_lengths.to(device=query.device, dtype=torch.long).reshape(-1)
    if prefix.numel() != batch:
        raise ValueError("valen_prefix_lengths must give one prefix length per row")
    if attention_mask is None:
        lengths = torch.full((batch,), length, device=query.device, dtype=torch.long)
    else:
        if attention_mask.ndim != 2:
            raise ValueError("Block-bidirectional attention expects a 2D padding mask")
        lengths = attention_mask.to(query.device).long().sum(-1)
    if bool(((prefix < 1) | (prefix > lengths)).any()):
        raise ValueError("Prefix lengths must lie within each row")
    if _flash_available(query):
        return _flash(query, key, value, prefix, lengths, dropout, scaling), None
    return _dense(module, query, key, value, prefix, lengths, dropout, scaling), None


def _flash(query, key, value, prefix, lengths, dropout, scaling):
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
    # Suffix block: bidirectional over itself plus the full prefix. / 后缀块：看完整行。
    if bool((suffix > 0).any()):
        output[tail] = flash_attn_varlen_func(
            q[tail], k[valid], v[valid], _segments(suffix), _segments(lengths),
            int(suffix.max()), int(lengths.max()), dropout_p=dropout, softmax_scale=scaling, causal=False)
    return output


def block_mask(prefix, lengths, length):
    """Boolean [B, 1, L, L] reference mask: True means the key is visible."""
    positions = torch.arange(length, device=prefix.device)
    query, key = positions[None, :, None], positions[None, None, :]
    valid_key = key < lengths[:, None, None]
    in_prefix = query < prefix[:, None, None]
    visible = torch.where(in_prefix, key < prefix[:, None, None], valid_key)
    return (visible & valid_key)[:, None]


def _dense(module, query, key, value, prefix, lengths, dropout, scaling):
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
    mask = block_mask(prefix, lengths, query.shape[2])
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=dropout, scale=scaling)
    # Padded query rows see nothing valid only if lengths==0, which is rejected above.
    return output.transpose(1, 2).contiguous()
