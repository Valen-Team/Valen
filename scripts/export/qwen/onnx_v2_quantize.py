"""ORT weight-only quantization restricted to backbone MLPs, protecting recurrence and Mixer."""

import argparse, copy, hashlib, json, time
from pathlib import Path
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--base-variant", default="fp32")
p.add_argument("--bits", type=int, choices=[4, 8], default=8)
a = p.parse_args()
folder = Path(a.model).resolve()
manifest = json.loads((folder / "valen_onnx.json").read_text())
base = manifest["variants"][a.base_variant]
variant = f"q{a.bits}-mlp"
graph = folder / base["graphs"]["decision"]
model = onnx.load(str(graph), load_external_data=False)
if variant in manifest["variants"] or graph.with_name("decision-" + variant + ".onnx").exists():
    raise ValueError("Quantized candidate already exists; choose a new model output directory")

names = [
    n.name
    for n in model.graph.node
    if n.op_type == "MatMul" and n.name.startswith("/model/layers.") and "/mlp/" in n.name
]
assert names
print("QUANTIZE_MLP", len(names), flush=True)
q = MatMulNBitsQuantizer(
    str(graph),
    bits=a.bits,
    block_size=32,
    is_symmetric=True,
    nodes_to_include=names,
    nodes_to_exclude=[n.name for n in model.graph.node if n.name not in names],
    op_types_to_quantize=("MatMul",),
)
q.process()
target = graph.with_name("decision-" + variant + ".onnx")
q.model.save_model_to_file(str(target), True)
onnx.checker.check_model(str(target))


def components(path):
    m = onnx.load(str(folder / path), load_external_data=False)
    return [path] + sorted(
        {
            str((Path(path).parent / d.value).as_posix())
            for t in m.graph.initializer
            for d in t.external_data
            if d.key == "location"
        }
    )


base["components"] = {key: components(value) for key, value in base["graphs"].items()}
new = copy.deepcopy(base)
new["graphs"]["decision"] = str(target.relative_to(folder))
new["components"]["decision"] = [
    str(x.relative_to(folder)) for x in sorted(target.parent.glob(target.name + "*"))
]
new["files"] = sorted(
    (set(base["files"]) - set(base["components"]["decision"])) | set(new["components"]["decision"])
)
new["precision"] = (
    {"embedding": "float32", "vision": "float32", "mixer": "float32", "backbone": new.get("precision")}
    if isinstance(new.get("precision"), str)
    else new["precision"]
)
new["precision"]["backbone"] = f"int{a.bits}_weight_only_MLP_protected_attention"
new["quality_status"] = "candidate_requires_frozen_gate"
new["quantization"] = {
    "tool": "onnxruntime.MatMulNBitsQuantizer",
    "block_size": 32,
    "bits": a.bits,
    "symmetric": True,
    "nodes": names,
}
manifest["variants"][variant] = new
for key in new["components"]["decision"]:
    data = folder / key
    digest = hashlib.sha256()
    with data.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    new.setdefault("sha256", {})[key] = digest.hexdigest()
new["sha256"] = {key: value for key, value in new.get("sha256", {}).items() if key in new["files"]}
(folder / "valen_onnx.json").write_text(json.dumps(manifest, indent=2))
print("QUANTIZE_DONE", target, flush=True)
