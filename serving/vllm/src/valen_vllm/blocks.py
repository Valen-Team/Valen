"""Valen-Decision-1.5 block attention for the full-attention layers.

The state prefix sees itself; shared glue sees the prefix and glue; the question
block also sees itself. Linear-attention layers stay causal.
"""

import torch
import torch.nn.functional as F

PREFIX, COMMON = -1, 0
# The compiler closes the state with these two turns: questions, then readouts.
GLUE = ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")


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


def token_segments(ids, glue, bidirectional=False, encode=None):
    """Block ids of one prompt, read from its last two chat turns.

    ``glue`` holds the token ids of GLUE. Decision questions share block 1, as
    ``bbox_blocks`` assigns them; ``bidirectional`` makes the whole suffix shared.
    With the tokenizer's ``encode``, each bbox question keeps its own block,
    found by the question headers the compiler writes. Prompts without the two
    turns keep causal attention (None).
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
    if encode is None:
        return segments
    tokens = ids.tolist()
    newline = len(encode("\n"))
    cursor, readout, question, marks = user + len(glue[0]), start + len(glue[1]), 0, []
    while readout < len(tokens):
        question += 1
        header = encode(f"Question {question}\nTask:")
        while cursor < start and tokens[cursor : cursor + len(header)] != header:
            cursor += 1
        bbox = encode(f"Question {question}\nTask: bbox")
        end = readout + len(encode(f"Question {question} Decision:")) + newline
        marks.append((cursor, readout, end, tokens[cursor : cursor + len(bbox)] == bbox))
        cursor, readout = cursor + len(header), end
    stops = [mark[0] for mark in marks[1:]] + [start]
    for index, ((begin, readout, end, bbox), stop) in enumerate(zip(marks, stops)):
        if bbox:
            segments[begin:stop] = segments[readout:end] = index + 1
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
