"""Valen-Decision-1.5 block attention for the full-attention layers.

The state prefix sees itself; shared glue sees the prefix and glue; a question
block also sees itself. Linear-attention layers stay causal.
"""

import torch
import torch.nn.functional as F

PREFIX, COMMON = -1, 0


def block_segments(mask, segments, bbox):
    """Block ids of the suffix tokens under a full-attention mask mode.

    ``segments`` are the compiler's ids: glue 0, question i as i + 1.
    """
    if mask == "block_bidirectional":
        return [COMMON] * len(segments)
    if mask == "bbox_blocks":
        # Localization questions stay isolated; decision questions share one block.
        return [s if s == COMMON or s in bbox else 1 for s in segments]
    return list(segments)


def block_attention(q, k, v, segments, scale):
    """q [L, H, D] and k, v [L, KV, D] of one prompt; returns [L, H, D]."""
    groups = q.shape[1] // k.shape[1]
    q = q.transpose(0, 1)
    k = k.transpose(0, 1).repeat_interleave(groups, 0)
    v = v.transpose(0, 1).repeat_interleave(groups, 0)
    if segments is None:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        return out.transpose(0, 1)
    out = torch.empty_like(q)
    prefix = segments == PREFIX
    shared = prefix | (segments == COMMON)
    for block in segments.unique():
        rows = segments == block
        keys = prefix if block == PREFIX else shared | rows
        out[:, rows] = F.scaled_dot_product_attention(
            q[:, rows], k[:, keys], v[:, keys], scale=scale
        )
    return out.transpose(0, 1)
