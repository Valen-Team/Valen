"""Export complete Valen networks without dropping vision or the decision head."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_rotary_pos_emb_vision, torch_recurrent_gated_delta_rule, Qwen3_5RMSNormGated,
)


class DeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, decay, beta):
        result, _ = torch_recurrent_gated_delta_rule(
            query, key, value, decay, beta, None, False, use_qk_l2norm_in_kernel=True,
        )
        return result

    @staticmethod
    def symbolic(graph, query, key, value, decay, beta):
        return graph.op("valen::GatedDelta", query, key, value, decay, beta).setType(value.type())


def export_delta(query, key, value, g, beta, **kwargs):
    return DeltaRule.apply(query, key, value, g, beta), None


def export_conv1d(hidden_states, weight, bias=None, activation=None, **kwargs):
    """The plain causal convolution, without the optional hub kernel."""
    length = hidden_states.shape[-1]
    out = nn.functional.conv1d(hidden_states, weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1,
                               groups=hidden_states.shape[1])[:, :, :length]
    return out if activation is None else modeling_qwen3_5.ACT2FN[activation](out)


def block_type(layer):
    return getattr(layer, "block_type", None) or layer.layer_type


class TextGraph(nn.Module):
    """Language backbone. ``block_mask`` adds the Valen-Decision-1.5 ``attention_mask`` input:
    an additive [sequence, sequence] mask for the full-attention layers, built by the caller
    from the compiler's prefix length and segment ids. Linear attention layers stay causal."""

    def __init__(self, text, block_mask=False):
        super().__init__()
        self.text = text
        self.block_mask = block_mask
        # The linear-attention layers call these module-level kernels; export the traceable ones.
        modeling_qwen3_5.torch_chunk_gated_delta_rule = export_delta
        modeling_qwen3_5.causal_conv1d_fn = export_conv1d
        for layer in text.layers:
            if block_type(layer) == "linear_attention":
                if not isinstance(layer.linear_attn.norm, Qwen3_5RMSNormGated):
                    norm = Qwen3_5RMSNormGated(layer.linear_attn.head_v_dim, eps=layer.linear_attn.layer_norm_epsilon)
                    norm.weight = layer.linear_attn.norm.weight
                    layer.linear_attn.norm = norm

    def forward(self, input_ids, position_ids, visual_positions, visual_features, attention_mask=None):
        hidden = self.text.embed_tokens(input_ids)
        hidden = hidden[0].index_copy(0, visual_positions, visual_features).unsqueeze(0)
        length = input_ids.shape[1]
        if self.block_mask:
            mask = attention_mask.reshape(1, 1, length, length)
        else:
            order = torch.arange(length, device=input_ids.device)
            visible = order[None, :] <= order[:, None]
            mask = torch.where(visible, 0.0, torch.finfo(torch.float32).min).reshape(1, 1, length, length)
        rotary = self.text.rotary_emb(hidden, position_ids)
        for layer in self.text.layers:
            hidden = layer(hidden, position_embeddings=rotary,
                           attention_mask=None if block_type(layer) == "linear_attention" else mask)
        return self.text.norm(hidden)


class VisionGraph(nn.Module):
    def __init__(self, vision):
        super().__init__()
        self.vision = vision

    def forward(self, pixel_values, position_indices, position_weights, rotary_cos, rotary_sin, attention_mask):
        hidden = self.vision.patch_embed(pixel_values)
        positions = self.vision.pos_embed(position_indices) * position_weights.unsqueeze(-1)
        hidden = hidden + (positions[0] + positions[1] + positions[2] + positions[3])
        for block in self.vision.blocks:
            x = block.norm1(hidden)
            attn = block.attn
            query, key, value = attn.qkv(x).reshape(x.shape[0], 3, attn.num_heads, attn.head_dim).permute(1, 0, 2, 3).unbind(0)
            query, key = apply_rotary_pos_emb_vision(query, key, rotary_cos, rotary_sin)
            query, key, value = [t.transpose(0, 1) for t in [query, key, value]]
            scores = query @ key.transpose(-1, -2) * attn.scaling + attention_mask.unsqueeze(0)
            output = (scores.softmax(-1) @ value).transpose(0, 1).reshape(x.shape[0], -1)
            hidden = hidden + attn.proj(output)
            hidden = hidden + block.mlp(block.norm2(hidden))
        return self.vision.merger(hidden)


class HeadGraph(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, features):
        return self.head.score_features(features)


class GroundingGraph(nn.Module):
    """Grounding head for bbox questions: token logits and edge distances in cell units.
    The caller turns distances into a box with the image grid, as grounding.decode does."""

    def __init__(self, grounding):
        super().__init__()
        self.grounding = grounding

    def forward(self, decision, tokens):
        g = self.grounding
        d = g.decision_norm(decision)
        k = g.key(g.token_norm(tokens))
        logits = (k @ g.query(d)) / g.scale
        distances = torch.nn.functional.softplus(g.regress(k * torch.sigmoid(g.gate(d)))) + 0.5
        return logits, distances


def lower_delta_nodes(model, config):
    """Replace export placeholders with standard ONNX Loop, retaining dynamic lengths."""
    import onnx
    from onnx import helper as h, numpy_helper as nh, TensorProto as T
    nodes = []
    heads = config.linear_num_value_heads
    key_dim, value_dim = config.linear_key_head_dim, config.linear_value_head_dim
    for index, node in enumerate(model.graph.node):
        if node.domain != "valen" or node.op_type != "GatedDelta":
            nodes.append(node)
            continue
        prefix = f"delta_{index}_"
        def name(value):
            return prefix + value
        def op(kind, inputs, output, **attrs):
            nodes.append(h.make_node(kind, inputs, [name(output)], name=name(output), **attrs))
            return name(output)
        def const(label, value):
            return op("Constant", [], label, value=nh.from_array(np.asarray(value)))
        axis_last = const("axis_last", np.array([-1], dtype=np.int64))
        epsilon = const("epsilon", np.array(1e-6, dtype=np.float32))
        scale = const("scale", np.array(key_dim ** -0.5, dtype=np.float32))
        q, k, v, decay, beta = node.input
        def normalize(label, tensor):
            square = op("Mul", [tensor, tensor], label + "_square")
            summed = op("ReduceSum", [square, axis_last], label + "_sum", keepdims=1)
            root = op("Sqrt", [op("Add", [summed, epsilon], label + "_eps")], label + "_root")
            return op("Div", [tensor, root], label + "_norm")
        q = op("Mul", [normalize("query", q), scale], "query_scaled")
        k = normalize("key", k)
        outer = {}
        for label, tensor in [("q", q), ("k", k), ("v", v), ("decay", decay), ("beta", beta)]:
            perm = [1, 0, 2, 3] if label in {"q", "k", "v"} else [1, 0, 2]
            outer[label] = op("Transpose", [tensor], label + "_time", perm=perm)
        seq_shape = op("Shape", [outer["q"]], "query_shape")
        zero = const("zero", np.array(0, dtype=np.int64))
        count = op("Gather", [seq_shape, zero], "trip_count", axis=0)
        condition = const("condition", np.array(True, dtype=np.bool_))
        initial = const("initial_state", np.zeros((1, heads, key_dim, value_dim), dtype=np.float32))
        body_nodes = []
        def body_op(kind, inputs, output, **attrs):
            result = name("body_" + output)
            body_nodes.append(h.make_node(kind, inputs, [result], **attrs))
            return result
        def body_const(label, value):
            return body_op("Constant", [], label, value=nh.from_array(np.asarray(value)))
        iteration, cond_in, state_in = [name("body_" + s) for s in ["iteration", "condition", "state"]]
        current = {key: body_op("Gather", [tensor, iteration], "current_" + key, axis=0) for key, tensor in outer.items()}
        axes_last = body_const("axes_last", np.array([-1], dtype=np.int64))
        axes_two = body_const("axes_two", np.array([-2], dtype=np.int64))
        axes_decay = body_const("axes_decay", np.array([-1, -2], dtype=np.int64))
        decay_value = body_op("Unsqueeze", [body_op("Exp", [current["decay"]], "decay_exp"), axes_decay], "decay_dims")
        decayed = body_op("Mul", [state_in, decay_value], "decayed_state")
        key_column = body_op("Unsqueeze", [current["k"], axes_last], "key_column")
        memory = body_op("ReduceSum", [body_op("Mul", [decayed, key_column], "memory_products"), axes_two], "memory", keepdims=0)
        difference = body_op("Sub", [current["v"], memory], "difference")
        beta_value = body_op("Unsqueeze", [current["beta"], axes_last], "beta_dims")
        delta = body_op("Mul", [difference, beta_value], "delta")
        update = body_op("Mul", [key_column, body_op("Unsqueeze", [delta, axes_two], "delta_dims")], "state_update")
        next_state = body_op("Add", [decayed, update], "next_state")
        query_column = body_op("Unsqueeze", [current["q"], axes_last], "query_column")
        output = body_op("ReduceSum", [body_op("Mul", [next_state, query_column], "output_products"), axes_two], "output", keepdims=0)
        cond_out = body_op("Identity", [cond_in], "condition_out")
        body = h.make_graph(body_nodes, name("body"), [
            h.make_tensor_value_info(iteration, T.INT64, []), h.make_tensor_value_info(cond_in, T.BOOL, []),
            h.make_tensor_value_info(state_in, T.FLOAT, [1, heads, key_dim, value_dim]),
        ], [h.make_tensor_value_info(cond_out, T.BOOL, []),
            h.make_tensor_value_info(next_state, T.FLOAT, [1, heads, key_dim, value_dim]),
            h.make_tensor_value_info(output, T.FLOAT, [1, heads, value_dim])])
        nodes.append(h.make_node("Loop", [count, condition, initial], [name("final_state"), name("scan")], body=body))
        nodes.append(h.make_node("Transpose", [name("scan")], list(node.output), perm=[1, 0, 2, 3]))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    imports = [opset for opset in model.opset_import if opset.domain != "valen"]
    del model.opset_import[:]
    model.opset_import.extend(imports)


def save_external_graph(wrapper, inputs, input_names, output_names, dynamic_axes, path, text_config=None):
    import onnx
    from onnx import TensorProto as T
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper.eval(), inputs, str(path), input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=18, export_params=False,
        do_constant_folding=False, dynamo=False,
    )
    graph = onnx.load(str(path), load_external_data=False)
    if text_config is not None:
        lower_delta_nodes(graph, text_config)
    state = wrapper.state_dict()
    graph_inputs = []
    tensor_manifest = []
    data_file, data_name, data_offset, shard = None, None, 0, -1
    try:
        for item in list(graph.graph.input):
            if item.name not in state:
                graph_inputs.append(item)
                continue
            tensor = state[item.name].detach().cpu().contiguous()
            if tensor.dtype != torch.float32:
                raise ValueError(f"Expected FP32 ONNX parameter {item.name}, got {tensor.dtype}")
            array = tensor.numpy()
            size = array.nbytes
            if data_file is None or data_offset + size > 1_800_000_000:
                if data_file is not None:
                    data_file.close()
                shard += 1
                data_name = f"{path.stem}-weights-{shard:03d}.data"
                data_file = (path.parent / data_name).open("wb")
                data_offset = 0
            initializer = T(name=item.name, data_type=T.FLOAT, dims=list(array.shape), data_location=T.EXTERNAL)
            for key, value in [("location", data_name), ("offset", str(data_offset)), ("length", str(size))]:
                entry = initializer.external_data.add()
                entry.key, entry.value = key, value
            graph.graph.initializer.append(initializer)
            array.tofile(data_file)
            tensor_manifest.append({"name": item.name, "shape": list(array.shape), "bytes": size,
                                    "file": data_name, "offset": data_offset})
            data_offset += size
    finally:
        if data_file is not None:
            data_file.close()
    del graph.graph.input[:]
    graph.graph.input.extend(graph_inputs)
    if {item.name for item in graph.graph.input} != set(input_names):
        raise ValueError(f"Unexpected exported inputs: {[item.name for item in graph.graph.input]}")
    if any(node.domain not in {"", "ai.onnx"} for node in graph.graph.node):
        raise ValueError("Export still contains unsupported custom operators")
    onnx.save_model(graph, str(path))
    onnx.checker.check_model(str(path))
    return tensor_manifest


def export_onnx(model, destination):
    if __package__:
        from .format_runtime import vision_inputs
    else:
        from format_runtime import vision_inputs
    directory = Path(destination) / "onnx"
    model.float().cpu().eval()
    # Valen-Decision-1.5 models attend in blocks; their text graph takes the mask as an input.
    block_mask = getattr(model.config, "valen_full_attention_mask", "causal") != "causal"
    text = TextGraph(model.backbone.language_model, block_mask)
    hidden = model.config.text_config.hidden_size
    text_inputs = (torch.arange(8).reshape(1, -1), torch.arange(8).reshape(1, 1, -1).expand(3, 1, -1),
                   torch.tensor([1, 2]), torch.zeros(2, hidden))
    text_names = ["input_ids", "position_ids", "visual_positions", "visual_features"]
    text_axes = {"input_ids": {1: "sequence"}, "position_ids": {2: "sequence"},
                 "visual_positions": {0: "visual_tokens"}, "visual_features": {0: "visual_tokens"},
                 "last_hidden_state": {1: "sequence"}}
    if block_mask:
        text_inputs += (torch.zeros(8, 8),)
        text_names.append("attention_mask")
        text_axes["attention_mask"] = {0: "sequence", 1: "sequence"}
    manifests = {}
    manifests["text"] = save_external_graph(text, text_inputs, text_names, ["last_hidden_state"],
                                            text_axes, directory / "text.onnx", model.config.text_config)
    print("EXPORTED text", destination, flush=True)
    config = model.config.vision_config.to_dict()
    patch_width = config["in_channels"] * config["temporal_patch_size"] * config["patch_size"] ** 2
    pixels = torch.randn(16, patch_width)
    feeds = vision_inputs(pixels, torch.tensor([[1, 4, 4]]), config)
    names = list(feeds)
    values = tuple(torch.from_numpy(feeds[name]) for name in names)
    manifests["vision"] = save_external_graph(VisionGraph(model.backbone.visual), values, names, ["visual_features"],
        {"pixel_values": {0: "patches"}, "position_indices": {1: "patches"}, "position_weights": {1: "patches"},
         "rotary_cos": {0: "patches"}, "rotary_sin": {0: "patches"},
         "attention_mask": {0: "patches", 1: "patches"}, "visual_features": {0: "visual_tokens"}},
        directory / "vision.onnx")
    print("EXPORTED vision", destination, flush=True)
    manifests["head"] = save_external_graph(HeadGraph(model.head), (torch.randn(3, 4, hidden),),
        ["features"], ["logits"], {"features": {0: "candidates"}, "logits": {0: "candidates"}}, directory / "head.onnx")
    print("EXPORTED head", destination, flush=True)
    if getattr(model, "grounding", None) is not None:
        manifests["grounding"] = save_external_graph(
            GroundingGraph(model.grounding.float()), (torch.randn(hidden), torch.randn(6, hidden)),
            ["decision", "tokens"], ["logits", "distances"],
            {"tokens": {0: "image_tokens"}, "logits": {0: "image_tokens"}, "distances": {0: "image_tokens"}},
            directory / "grounding.onnx")
        print("EXPORTED grounding", destination, flush=True)
    (directory / "tensor_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    return manifests


def safe_tensors(source):
    """Read each tensor once without accumulating an entire model in memory."""
    from safetensors import safe_open
    for file in sorted(Path(source).glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as reader:
            for name in reader.keys():
                yield name, reader.get_tensor(name)


def export_gguf(source, destination, repo_id, revision):
    import gguf
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(source)
    config = json.loads((source / "config.json").read_text())
    label = "BF16" if config.get("dtype") == "bfloat16" else "F32"
    path = destination / (source.name + "-" + label + ".gguf")
    writer = gguf.GGUFWriter(str(path), arch="valen_qwen")
    writer.add_name(source.name)
    writer.add_string("general.description", "Complete Valen decision model; use its dedicated ValenGGUFModel runtime.")
    writer.add_string("general.source.huggingface.repository", repo_id)
    writer.add_string("valen.source_revision", revision)
    writer.add_string("valen.config_json", json.dumps(config, separators=(",", ":")))
    writer.add_string("valen.tensor_layout", "original_pytorch_names; original shapes in valen.tensor_shapes_json")
    writer.add_string("valen.precision", "lossless_original_tensor_precision")
    writer.add_uint32("valen.format_version", 1)
    manifests = []
    for name, tensor in safe_tensors(source):
        dtype = gguf.GGMLQuantizationType.BF16 if tensor.dtype == torch.bfloat16 else gguf.GGMLQuantizationType.F32
        array_dtype = np.dtype(np.uint16 if tensor.dtype == torch.bfloat16 else np.float32)
        if tensor.dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError(f"Unsupported native weight dtype: {name}: {tensor.dtype}")
        stored_shape = tuple(tensor.shape) if tensor.ndim <= 4 else (tensor.numel(),)
        writer.add_tensor_info(name, stored_shape, array_dtype, tensor.numel() * tensor.element_size(), raw_dtype=dtype)
        manifests.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    writer.add_string("valen.tensor_shapes_json", json.dumps({item["name"]: item["shape"] for item in manifests}, separators=(",", ":")))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for name, tensor in safe_tensors(source):
        array = tensor.view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
        writer.write_tensor_data(array)
    writer.close()
    # Private mappings also work on filesystems without shared mmap. / 兼容不支持共享映射的文件系统。
    reader = gguf.GGUFReader(str(path), mode="c")
    stored = {t.name: t for t in reader.tensors}
    for name, tensor in safe_tensors(source):
        array = tensor.view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
        restored = stored[name].data
        if tensor.dtype == torch.bfloat16:
            restored = restored.view(np.uint16)
        restored = restored.reshape(array.shape)
        if not np.array_equal(array, restored):
            raise ValueError("GGUF tensor differs from its source: " + name)
    if len(reader.tensors) != len(manifests):
        raise ValueError("GGUF tensor count differs from source")
    print("EXPORTED GGUF", path, "verified tensors", len(manifests), flush=True)
    return {"file": path.name, "tensor_count": len(manifests), "tensors": manifests,
            "all_source_tensors_bitwise_preserved": True}
