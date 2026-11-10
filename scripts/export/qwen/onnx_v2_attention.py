"""Explicit block attention for Valen-Decision-1.5 ONNX graphs.

GroupQueryAttention is causal by construction. The full-attention layers become explicit
attention over a caller-supplied additive mask: ``block_mask`` [sequence, sequence], float32,
0 where the key is visible. Linear attention stays causal.
"""

import numpy as np
from onnx import TensorProto as T, helper as h, numpy_helper as nh


def use_block_mask(model, head_dim, dtype):
    """Replace every GroupQueryAttention of ``model`` in place; ``dtype`` is its tensor type."""
    scalar = {T.FLOAT: np.float32, T.FLOAT16: np.float16}[dtype]
    model.graph.initializer.extend(
        [
            nh.from_array(np.array([0, 1], np.int64), "valen/attn/mask_axes"),
            nh.from_array(np.array([2], np.int64), "valen/attn/group_axis"),
            nh.from_array(np.array([0, 0, -1], np.int64), "valen/attn/merge_heads"),
        ]
    )
    explicit = [h.make_node("Unsqueeze", ["block_mask", "valen/attn/mask_axes"], ["valen/attn/mask_fp32"])]
    if dtype == T.FLOAT16:
        # The float32 mask minimum lies outside float16; keep hidden keys at a finite float16 value.
        model.graph.initializer.append(
            nh.from_array(np.array(np.finfo(np.float16).min, np.float32), "valen/attn/mask_floor")
        )
        explicit.extend(
            [
                h.make_node("Clip", ["valen/attn/mask_fp32", "valen/attn/mask_floor"], ["valen/attn/mask_fp16_range"]),
                h.make_node("Cast", ["valen/attn/mask_fp16_range"], ["valen/attn/mask"], to=dtype),
            ]
        )
    else:
        explicit.append(h.make_node("Cast", ["valen/attn/mask_fp32"], ["valen/attn/mask"], to=dtype))
    replaced = 0
    for node in model.graph.node:
        if node.domain != "com.microsoft" or node.op_type != "GroupQueryAttention":
            explicit.append(node)
            continue
        attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
        heads, kv_heads = attrs["num_heads"], attrs["kv_num_heads"]
        if attrs.get("do_rotary") or attrs.get("softcap") or attrs.get("local_window_size", -1) != -1:
            raise ValueError("Unexpected GroupQueryAttention settings")
        query, key, value = node.input[:3]
        prefix = "valen/attn/" + node.name.strip("/").replace("/", "_")
        model.graph.initializer.extend(
            [
                nh.from_array(np.array([0, 0, heads, -1], np.int64), prefix + "/q_shape"),
                nh.from_array(np.array([0, 0, kv_heads, -1], np.int64), prefix + "/kv_shape"),
                nh.from_array(np.array([1, 1, heads // kv_heads, 1, 1], np.int64), prefix + "/repeats"),
                nh.from_array(np.array([0, heads, -1, head_dim], np.int64), prefix + "/grouped"),
                nh.from_array(np.array(attrs["scale"], scalar), prefix + "/scale"),
            ]
        )

        def heads_first(source, label, shape):
            explicit.extend(
                [
                    h.make_node("Reshape", [source, prefix + shape], [f"{prefix}/{label}_split"]),
                    h.make_node("Transpose", [f"{prefix}/{label}_split"], [f"{prefix}/{label}_heads"], perm=[0, 2, 1, 3]),
                ]
            )
            return f"{prefix}/{label}_heads"

        def per_query_head(source, label):
            # Each key/value head serves heads // kv_heads consecutive query heads.
            explicit.extend(
                [
                    h.make_node("Unsqueeze", [source, "valen/attn/group_axis"], [f"{prefix}/{label}_group"]),
                    h.make_node("Tile", [f"{prefix}/{label}_group", prefix + "/repeats"], [f"{prefix}/{label}_tiled"]),
                    h.make_node("Reshape", [f"{prefix}/{label}_tiled", prefix + "/grouped"], [f"{prefix}/{label}_full"]),
                ]
            )
            return f"{prefix}/{label}_full"

        q = heads_first(query, "q", "/q_shape")
        k = per_query_head(heads_first(key, "k", "/kv_shape"), "k")
        v = per_query_head(heads_first(value, "v", "/kv_shape"), "v")
        explicit.extend(
            [
                h.make_node("Transpose", [k], [prefix + "/k_t"], perm=[0, 1, 3, 2]),
                h.make_node("MatMul", [q, prefix + "/k_t"], [prefix + "/scores"]),
                h.make_node("Mul", [prefix + "/scores", prefix + "/scale"], [prefix + "/scaled"]),
                h.make_node("Add", [prefix + "/scaled", "valen/attn/mask"], [prefix + "/masked"]),
                h.make_node("Softmax", [prefix + "/masked"], [prefix + "/probs"], axis=-1),
                h.make_node("MatMul", [prefix + "/probs", v], [prefix + "/mixed"]),
                h.make_node("Transpose", [prefix + "/mixed"], [prefix + "/mixed_t"], perm=[0, 2, 1, 3]),
                h.make_node("Reshape", [prefix + "/mixed_t", "valen/attn/merge_heads"], [node.output[0]]),
            ]
        )
        replaced += 1
    if not replaced:
        raise ValueError("The graph has no GroupQueryAttention to replace")
    del model.graph.node[:]
    model.graph.node.extend(explicit)
    model.graph.input.append(h.make_tensor_value_info("block_mask", T.FLOAT, ["sequence_length", "sequence_length"]))
