"""Build a portable Valen artifact around the upstream-exported backbone.

The backbone and vision operations are reused. Only Valen role pooling and
the already-trained Mixer readout are attached to the decision graph.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
from safetensors import safe_open

p = argparse.ArgumentParser()
p.add_argument("source")
p.add_argument("legacy")
p.add_argument("backbone")
p.add_argument("output")
p.add_argument("--variant", default="fp32")
p.add_argument("--quantize-embedding", action="store_true")
args = p.parse_args()
source, legacy, backbone, output = map(Path, [args.source, args.legacy, args.backbone, args.output])
graphs = output / "onnx"
graphs.mkdir(parents=True, exist_ok=True)
config = json.loads((source / "config.json").read_text())
hidden = config["text_config"]["hidden_size"]
model = onnx.load(backbone / "backbone.onnx", load_external_data=False)
io_type = next(item.type.tensor_type.elem_type for item in model.graph.input if item.name == "inputs_embeds")
for name in [
    "config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
]:
    shutil.copyfile(source / name, output / name)
for graph_name in ["vision", "head"]:
    for path in (legacy / "onnx").glob(graph_name + "*"):
        if path.is_file():
            target = graphs / path.name
            if target.is_symlink():
                target.unlink()
            if not target.exists():
                shutil.copyfile(path, target)
for path in backbone.glob("*.data"):
    target = graphs / path.name
    if target.is_symlink():
        target.unlink()
    if not target.exists():
        shutil.copyfile(path, target)

# Reusing a graph is safe only when every copied learned tensor matches the
# current native checkpoint, including BF16-to-FP32 conversion for 4B.
index_path = source / "model.safetensors.index.json"
weight_map = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else None
tensor_manifest = json.loads((legacy / "onnx/tensor_manifest.json").read_text())
checked = []
for component in ["vision", "head"]:
    for tensor_record in tensor_manifest[component]:
        native_name = (
            tensor_record["name"].replace("vision.", "backbone.visual.", 1)
            if component == "vision"
            else tensor_record["name"]
        )
        shard = weight_map[native_name] if weight_map else "model.safetensors"
        with safe_open(source / shard, framework="pt", device="cpu") as reader:
            expected = reader.get_tensor(native_name).float().numpy()
            with (legacy / "onnx" / tensor_record["file"]).open("rb") as stream:
                stream.seek(tensor_record["offset"])
                actual = np.frombuffer(stream.read(tensor_record["bytes"]), dtype=np.float32).reshape(
                    tensor_record["shape"]
                )
            if not np.array_equal(expected, actual):
                raise ValueError(f"Reused tensor differs from source checkpoint: {native_name}")
            checked.append(native_name)
            del expected, actual
(output / "reused-tensors.json").write_text(json.dumps({"passed": True, "tensors": checked}, indent=2))

name = "backbone.language_model.embed_tokens.weight"
index_path = source / "model.safetensors.index.json"
shard = json.loads(index_path.read_text())["weight_map"][name] if index_path.exists() else "model.safetensors"
with safe_open(source / shard, framework="pt", device="cpu") as reader:
    tensor_value = reader.get_tensor(name)
    array = tensor_value.half().numpy() if io_type == T.FLOAT16 else tensor_value.float().numpy()
    array.tofile(graphs / "embedding.data")
    tensor = T(name="embedding.weight", data_type=io_type, dims=list(array.shape), data_location=T.EXTERNAL)
    for key, value in [("location", "embedding.data"), ("offset", "0"), ("length", str(array.nbytes))]:
        entry = tensor.external_data.add()
        entry.key, entry.value = key, value
    embedding = h.make_model(
        h.make_graph(
            [h.make_node("Gather", ["embedding.weight", "input_ids"], ["inputs_embeds"], axis=0)],
            "Valen embeddings",
            [h.make_tensor_value_info("input_ids", T.INT64, [1, "sequence"])],
            [h.make_tensor_value_info("inputs_embeds", io_type, [1, "sequence", hidden])],
            [tensor],
        ),
        opset_imports=[h.make_opsetid("", 22)],
        ir_version=10,
    )
    onnx.save_model(embedding, graphs / "embedding.onnx")
del array
if args.quantize_embedding:
    if "q4" not in args.variant:
        raise ValueError("The upstream Gather quantizer supports INT4 embeddings only")
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    quantizer = MatMulNBitsQuantizer(
        str(graphs / "embedding.onnx"),
        bits=4,
        block_size=32,
        is_symmetric=True,
        op_types_to_quantize=("Gather",),
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(graphs / "embedding.onnx"), True)
    (graphs / "embedding.data").unlink()
    del quantizer

head = onnx.load(legacy / "onnx/head.onnx", load_external_data=False)
# The upstream builder emits ORT's experimental default-domain RMSNorm.
# Lower it to standard ops so the artifact also passes the ONNX checker.
lowered = []
for index, node in enumerate(model.graph.node):
    if node.domain or node.op_type != "SimplifiedLayerNormalization":
        lowered.append(node)
        continue
    prefix = f"valen/rms_{index}"
    attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
    axis, eps = prefix + "/axis", prefix + "/epsilon"
    model.graph.initializer.extend(
        [
            nh.from_array(np.array([attrs.get("axis", -1)], np.int64), axis),
            nh.from_array(np.array(attrs.get("epsilon", 1e-5), np.float32), eps),
        ]
    )
    x, weight = node.input
    weight_type = next(tensor.data_type for tensor in model.graph.initializer if tensor.name == weight)
    lowered.extend(
        [
            h.make_node("Cast", [x], [prefix + "/input_fp32"], to=T.FLOAT),
            h.make_node("Cast", [weight], [prefix + "/weight_fp32"], to=T.FLOAT),
            h.make_node("Mul", [prefix + "/input_fp32", prefix + "/input_fp32"], [prefix + "/square"]),
            h.make_node("ReduceMean", [prefix + "/square", axis], [prefix + "/mean"], keepdims=1),
            h.make_node("Add", [prefix + "/mean", eps], [prefix + "/variance"]),
            h.make_node("Sqrt", [prefix + "/variance"], [prefix + "/root"]),
            h.make_node("Div", [prefix + "/input_fp32", prefix + "/root"], [prefix + "/normalized"]),
            h.make_node("Mul", [prefix + "/normalized", prefix + "/weight_fp32"], [prefix + "/result_fp32"]),
            h.make_node("Cast", [prefix + "/result_fp32"], list(node.output), to=weight_type),
        ]
    )
del model.graph.node[:]
model.graph.node.extend(lowered)
hidden_output = next(x.name for x in model.graph.output if x.name == "hidden_states")
nodes = model.graph.node
constants = {
    "valen/axis0": np.array([0], np.int64),
    "valen/axis2": np.array([2], np.int64),
    "valen/scan_axis": np.array(0, np.int64),
    "valen/start": np.array(0, np.int64),
    "valen/end": np.array(1, np.int64),
    "valen/zero_row": np.zeros((1, hidden), np.float32),
}
model.graph.initializer.extend(nh.from_array(value, name) for name, value in constants.items())
nodes.extend(
    [
        h.make_node("Cast", [hidden_output], ["valen/hidden_fp32"], to=T.FLOAT),
        h.make_node("Squeeze", ["valen/hidden_fp32", "valen/axis0"], ["valen/hidden"]),
        h.make_node("CumSum", ["valen/hidden", "valen/scan_axis"], ["valen/cumulative"]),
        h.make_node("Concat", ["valen/zero_row", "valen/cumulative"], ["valen/prefix"], axis=0),
        h.make_node("Gather", ["role_spans", "valen/start"], ["valen/starts"], axis=2),
        h.make_node("Gather", ["role_spans", "valen/end"], ["valen/ends"], axis=2),
        h.make_node("Gather", ["valen/prefix", "valen/starts"], ["valen/start_values"], axis=0),
        h.make_node("Gather", ["valen/prefix", "valen/ends"], ["valen/end_values"], axis=0),
        h.make_node("Sub", ["valen/end_values", "valen/start_values"], ["valen/sums"]),
        h.make_node("Sub", ["valen/ends", "valen/starts"], ["valen/lengths_int"]),
        h.make_node("Cast", ["valen/lengths_int"], ["valen/lengths"], to=T.FLOAT),
        h.make_node("Unsqueeze", ["valen/lengths", "valen/axis2"], ["valen/divisor"]),
        h.make_node("Div", ["valen/sums", "valen/divisor"], ["valen/features"]),
    ]
)
for node in head.graph.node:
    for i, value in enumerate(node.input):
        if value == "features":
            node.input[i] = "valen/features"
nodes.extend(head.graph.node)
model.graph.initializer.extend(head.graph.initializer)
model.graph.input.append(h.make_tensor_value_info("role_spans", T.INT64, ["candidates", 4, 2]))
del model.graph.output[:]
model.graph.output.extend([h.make_tensor_value_info("logits", T.FLOAT, ["candidates"])])

# Remove generation outputs and their unused dependencies, without loading GB weights.
required = {"logits"}
kept = []
for node in reversed(model.graph.node):
    if any(name in required for name in node.output):
        kept.append(node)
        required.update(node.input)
del model.graph.node[:]
model.graph.node.extend(reversed(kept))
for collection in [model.graph.input, model.graph.initializer, model.graph.value_info]:
    values = [item for item in collection if item.name in required]
    del collection[:]
    collection.extend(values)
model.graph.name = "Valen decision prefill and Mixer"
onnx.save_model(model, graphs / "decision.onnx")
onnx.checker.check_model(str(graphs / "decision.onnx"))
onnx.checker.check_model(str(graphs / "embedding.onnx"))

files = [
    str(path.relative_to(output))
    for path in output.rglob("*")
    if path.is_file() and path.name != "valen_onnx.json"
]
files = [name for name in files if Path(name).name != "head.onnx"]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


components = {
    "embedding": [name for name in files if name.startswith("onnx/embedding")],
    "vision": [name for name in files if name.startswith("onnx/vision")],
    "decision": [name for name in files if name.startswith(("onnx/decision", "onnx/backbone", "onnx/head-"))],
}


def graph_schema(graph):
    def value(item):
        tensor = item.type.tensor_type
        return {
            "name": item.name,
            "dtype": T.DataType.Name(tensor.elem_type),
            "shape": [dim.dim_param or dim.dim_value for dim in tensor.shape.dim],
        }

    return {
        "inputs": [value(item) for item in graph.graph.input],
        "outputs": [value(item) for item in graph.graph.output],
    }


manifest = {
    "format_version": 2,
    "contract_version": "valen-decisions-1",
    "model_name": source.name,
    "source": {
        "model": source.name,
        "config_sha256": digest(source / "config.json"),
        "weight_files": [
            {"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in sorted(source.glob("model*.safetensors"))
        ],
    },
    "exporter": {
        "project": "microsoft/onnxruntime-genai",
        "revision": "6b7ff3ca3efa15da219050fdb4ecf8bed67875af",
    },
    "variants": {
        args.variant: {
            "precision": {
                "backbone": "int4_weight_only"
                if "q4" in args.variant
                else "int8_weight_only"
                if "q8" in args.variant
                else "float16"
                if io_type == T.FLOAT16
                else "float32",
                "embedding": "int4_weight_only"
                if args.quantize_embedding
                else "float16"
                if io_type == T.FLOAT16
                else "float32",
                "vision": "float32",
                "mixer": "float32",
            },
            "io_dtype": "float16" if io_type == T.FLOAT16 else "float32",
            "minimum_onnxruntime": "1.31.0",
            "opset": 22,
            "contrib_operators": sorted(
                {node.op_type for node in model.graph.node if node.domain == "com.microsoft"}
            ),
            "graphs": {
                "embedding": "onnx/embedding.onnx",
                "vision": "onnx/vision.onnx",
                "decision": "onnx/decision.onnx",
            },
            "io_schema": {
                "decision": graph_schema(model),
                "embedding": graph_schema(embedding),
                "vision": graph_schema(onnx.load(str(graphs / "vision.onnx"), load_external_data=False)),
            },
            "quality_status": "candidate_requires_target_validation",
            "components": components,
            "files": sorted(files),
            "sha256": {name: digest(output / name) for name in sorted(files)},
        }
    },
}
(output / "valen_onnx.json").write_text(json.dumps(manifest, indent=2))
print("PACKAGE_COMPLETE", output, flush=True)
