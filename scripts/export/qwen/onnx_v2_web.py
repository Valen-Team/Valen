"""Reuse the pinned community Scan graph with only verified Valen weights.

No community weight shard is loaded. Learned initializers come from the tested
Valen CPU FP32 graph, including inline gates and normalization parameters.
"""

from copy import deepcopy
import hashlib, json, os, subprocess, shutil
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import argparse

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--output", required=True)
p.add_argument("--template", required=True)
args = p.parse_args()
source, output, template = map(lambda value: Path(value).resolve(), [args.source, args.output, args.template])
if output.exists() and (not output.is_dir() or any(output.iterdir())):
    p.error("Output must be a new or empty directory")
expected_template = "c2f332890fb1cdb8ee05e21a52e871ba3333f6149a30de97c4cfb0227207e674"
if hashlib.sha256(template.read_bytes()).hexdigest() != expected_template:
    p.error("Unexpected community graph template checksum")
config = json.loads((source / "config.json").read_text())["text_config"]
if (
    config["hidden_size"],
    config["num_hidden_layers"],
    config["num_attention_heads"],
    config["num_key_value_heads"],
    config["head_dim"],
) != (1024, 24, 8, 2, 256):
    p.error("This pinned browser template is verified only for Valen 0.8B")
graphs = output / "onnx"
graphs.mkdir(parents=True, exist_ok=True)
reference = onnx.load(source / "onnx/decision.onnx", load_external_data=False)
model = onnx.load(template, load_external_data=False)
hidden = next(n.input[0] for n in model.graph.node if n.output == ["logits"])
model.graph.node.remove(next(n for n in model.graph.node if n.output == ["logits"]))
start = next(i for i, n in enumerate(reference.graph.node) if "valen/hidden" in n.output)
tail = deepcopy(list(reference.graph.node)[start:])
tail[0].input[0] = "valen/portable_hidden_fp32"
tail.insert(0, h.make_node("Cast", [hidden], ["valen/portable_hidden_fp32"], to=T.FLOAT))
model.graph.node.extend(tail)
produced = {x for n in tail for x in n.output}
needed = {x for n in tail for x in n.input if x not in produced}
model.graph.initializer.extend(deepcopy(x) for x in reference.graph.initializer if x.name in needed)
model.graph.input.extend(deepcopy(x) for x in reference.graph.input if x.name == "role_spans")
del model.graph.output[:]
model.graph.output.extend(deepcopy(reference.graph.output))


# Lexical captures in If/Scan subgraphs are dependencies, too.
def free_graph(g):
    defined = (
        {x.name for x in g.input} | {x.name for x in g.initializer} | {x for n in g.node for x in n.output}
    )
    return set().union(*(dependencies(n) for n in g.node)) - defined


def dependencies(n):
    result = set(n.input) - {""}
    for a in n.attribute:
        if a.type == onnx.AttributeProto.GRAPH:
            result.update(free_graph(a.g))
        elif a.type == onnx.AttributeProto.GRAPHS:
            for g in a.graphs:
                result.update(free_graph(g))
    return result


required = {"logits"}
kept = []
for n in reversed(model.graph.node):
    if required.intersection(n.output):
        kept.append(n)
        required.update(dependencies(n))
del model.graph.node[:]
model.graph.node.extend(reversed(kept))
for c in [model.graph.input, model.graph.initializer, model.graph.value_info]:
    keep = [x for x in c if x.name in required]
    del c[:]
    c.extend(keep)
refs = {x.name: x for x in reference.graph.initializer}
mapping = []
with (graphs / "backbone.onnx.data").open("wb") as target:
    for i, x in enumerate(model.graph.initializer):
        if not x.name.startswith("model."):
            continue
        if x.name == "model.inv_freq":
            config = json.loads((source / "config.json").read_text())["text_config"]
            rope = config["rope_parameters"]
            dim = int(config["head_dim"] * rope["partial_rotary_factor"])
            value = (
                1 / (float(rope["rope_theta"]) ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
            ).reshape(x.dims)
            assert np.allclose(nh.to_array(x), value, atol=1e-7, rtol=1e-6)
            x.CopyFrom(nh.from_array(value, x.name))
            continue
        name = (
            x.name.replace(".gdn.", ".linear_attn.")
            .replace(".A_neg_exp", ".neg_exp_A")
            .replace(".weight_3d", ".weight")
            .replace(".in_proj_qkv.", ".qkv_proj.")
            .replace(".in_proj_z.", ".z_proj.")
            .replace(".in_proj_b.", ".b_proj.")
            .replace(".in_proj_a.", ".a_proj.")
        )
        if ".attn." in name and any("." + part + "_proj.MatMul.weight" in name for part in ["q", "k", "v"]):
            original = name
            name = name.rsplit(".", 3)[0] + ".qkv_proj.MatMul.weight"
            full = nh.to_array(refs[name], base_dir=str(source / "onnx"))
            part = original.split(".attn.")[1][0]
            start_col = {"q": 0, "k": 4096, "v": 4608}[part]
            width = {"q": 4096, "k": 512, "v": 512}[part]
            value = full[:, start_col : start_col + width]
        else:
            assert name in refs, ("Missing trained initializer", x.name, name)
            value = nh.to_array(refs[name], base_dir=str(source / "onnx"))
        if "conv1d.weight" in x.name:
            value = value.reshape(x.dims)
        assert list(value.shape) == list(x.dims), (x.name, value.shape, list(x.dims))
        dtype = {T.FLOAT: np.float32, T.FLOAT16: np.float16}[x.data_type]
        raw = np.asarray(value, dtype=dtype).tobytes()
        offset = target.tell()
        target.write(raw)
        replacement = T(name=x.name, data_type=x.data_type, dims=list(x.dims), data_location=T.EXTERNAL)
        for k, v in [("location", "backbone.onnx.data"), ("offset", str(offset)), ("length", str(len(raw)))]:
            e = replacement.external_data.add()
            e.key, e.value = k, v
        model.graph.initializer[i].CopyFrom(replacement)
        mapping.append({"target": x.name, "source": name, "sha256": hashlib.sha256(raw).hexdigest()})
# Experimental default-domain RMSNorm -> standard operators, with FP32 accumulation.
lowered = []
for index, node in enumerate(model.graph.node):
    if node.domain or node.op_type != "SimplifiedLayerNormalization":
        lowered.append(node)
        continue
    prefix = f"valen/portable_rms_{index}"
    attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
    axis, eps = prefix + "/axis", prefix + "/eps"
    model.graph.initializer.extend(
        [
            nh.from_array(np.array([attrs.get("axis", -1)], np.int64), axis),
            nh.from_array(np.array(attrs.get("epsilon", 1e-5), np.float32), eps),
        ]
    )
    x, w = node.input
    lowered.extend(
        [
            h.make_node("Mul", [x, x], [prefix + "/square"]),
            h.make_node("ReduceMean", [prefix + "/square", axis], [prefix + "/mean"], keepdims=1),
            h.make_node("Add", [prefix + "/mean", eps], [prefix + "/var"]),
            h.make_node("Sqrt", [prefix + "/var"], [prefix + "/root"]),
            h.make_node("Div", [x, prefix + "/root"], [prefix + "/norm"]),
            h.make_node("Mul", [prefix + "/norm", w], list(node.output)),
        ]
    )
del model.graph.node[:]
model.graph.node.extend(lowered)
for op in model.opset_import:
    if op.domain == "":
        op.version = 22
onnx.save(model, graphs / "decision.onnx")
for name in [
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]:
    target = output / name
    if target.exists() and target.stat().st_nlink > 1:
        target.unlink()
    if not target.exists():
        shutil.copy2(source / name, target)
for name in [
    "embedding.onnx",
    "embedding.data",
    "vision.onnx",
    "vision-weights-000.data",
    "head-weights-000.data",
]:
    target = graphs / name
    if target.exists() and target.stat().st_nlink > 1:
        target.unlink()
    if not target.exists():
        shutil.copy2(source / "onnx" / name, target)
onnx.checker.check_model(str(graphs / "decision.onnx"))
files = [str(x.relative_to(output)) for x in output.rglob("*") if x.is_file() and x.name != "valen_onnx.json"]


def digest(path):
    h_ = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h_.update(c)
    return h_.hexdigest()


def schema(g):
    return {
        "inputs": [
            {
                "name": x.name,
                "type": {T.FLOAT: "float32", T.FLOAT16: "float16", T.INT64: "int64"}[
                    x.type.tensor_type.elem_type
                ],
                "shape": [d.dim_param or d.dim_value for d in x.type.tensor_type.shape.dim],
            }
            for x in g.graph.input
        ]
    }


manifest = json.loads((source / "valen_onnx.json").read_text())
variant = {
    "graphs": {
        "embedding": "onnx/embedding.onnx",
        "vision": "onnx/vision.onnx",
        "decision": "onnx/decision.onnx",
    },
    "io_dtype": "float32",
    "minimum_onnxruntime": "1.30.0",
    "precision": {
        "backbone": "mixed_fp16_fp32",
        "embedding": "float32",
        "vision": "float32",
        "mixer": "float32",
    },
    "files": files,
    "sha256": {name: digest(output / name) for name in files},
    "components": {
        "embedding": ["onnx/embedding.data"],
        "vision": ["onnx/vision-weights-000.data"],
        "decision": ["onnx/backbone.onnx.data", "onnx/head-weights-000.data"],
    },
    "io_schema": {"decision": schema(model)},
}
manifest["variants"] = {"web-portable": variant}
manifest["exporter"] = {
    "project": "onnx-community/Qwen3.5-0.8B-ONNX graph template",
    "revision": "8fe9dbe3fb39fe902e18df9be84b6a0ac77ae0f6",
    "template_sha256": digest(template),
    "learned_weights": "exclusively verified Valen FP32 graph initializers",
}
(output / "valen_onnx.json").write_text(json.dumps(manifest, indent=2))
(output / "portable-weight-mapping.json").write_text(json.dumps(mapping, indent=2))
print("PORTABLE_VALEN_GRAPH_READY", len(mapping), len(model.graph.node), flush=True)
