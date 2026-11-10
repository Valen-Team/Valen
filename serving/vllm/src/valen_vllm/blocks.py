"""Valen-Decision-1.5 block attention for the full-attention layers.

The state prefix sees itself; shared glue sees the prefix and glue; the question
block also sees itself. Linear-attention layers stay causal.
"""

import torch
import torch.nn.functional as F

PREFIX, COMMON = -1, 0
# The compiler closes the state with these two turns: questions, then readouts.
GLUE = ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")


def token_segments(ids, glue, bidirectional=False):
    """Block ids of one prompt, read from its last two chat turns.

    ``glue`` holds the token ids of GLUE. Decision questions share block 1, as
    ``bbox_blocks`` assigns them; ``bidirectional`` makes the whole suffix shared.
    Prompts without the two turns keep causal attention (None).
    """
    turns = (ids == glue[0][0]).nonzero().flatten().tolist()
    if len(turns) < 2:
        return None
    user, assistant = turns[-2:]
    segments = torch.ones_like(ids)
    segments[:user] = PREFIX
    if bidirectional:
        segments[user:] = COMMON
        return segments
    segments[user : user + len(glue[0])] = COMMON
    start = assistant - glue[1].index(glue[0][0])
    segments[start : start + len(glue[1])] = COMMON
    return segments


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
