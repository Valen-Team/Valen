"""Add one exported variant to a multi-variant Valen ONNX bundle.

``--base`` is the bundle that already holds validated variants, ``--export`` a single-variant
export (``onnx_v2.py`` or ``onnx_v2_web.py`` output). Files whose content the base already holds
keep the base name and are shared; everything else is written as ``onnx/<variant>-<name>`` with the
external-data locations of its graphs rewritten. ``--output`` receives the files of the new variant
and the merged ``valen_onnx.json``; the base directory is only read.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import onnx
from onnx import TensorProto as T

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--base", required=True)
p.add_argument("--export", required=True)
p.add_argument("--variant", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()
base, export, output = (Path(value).resolve() for value in [args.base, args.export, args.output])
(output / "onnx").mkdir(parents=True, exist_ok=True)
merged_path = output / "valen_onnx.json"
manifest = json.loads((merged_path if merged_path.exists() else base / "valen_onnx.json").read_text())
if args.variant in manifest["variants"]:
    p.error(f"The bundle already has a {args.variant} variant")
exported = json.loads((export / "valen_onnx.json").read_text())
variant = copy.deepcopy(exported["variants"][args.variant])
published = {name: digest for entry in manifest["variants"].values() for name, digest in entry["sha256"].items()}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def place(name, content=None):
    """Bundle name for an exported file: the shared name when the content is already published."""
    source = export / name
    value = hashlib.sha256(content).hexdigest() if content is not None else digest(source)
    if published.get(name) == value:
        target = name
    else:
        target = f"{Path(name).parent.as_posix()}/{args.variant}-{Path(name).name}"
        if target in published:
            raise ValueError(f"{target} is already published")
    if content is not None:
        (output / target).write_bytes(content)
    elif not (output / target).exists():
        shutil.copyfile(source, output / target)
    sha256[target] = value
    return target


def schema(model):
    def value(item):
        tensor = item.type.tensor_type
        return {
            "name": item.name,
            "dtype": T.DataType.Name(tensor.elem_type),
            "shape": [dim.dim_param or dim.dim_value for dim in tensor.shape.dim],
        }

    return {"inputs": [value(x) for x in model.graph.input], "outputs": [value(x) for x in model.graph.output]}


sha256, graphs, components, io_schema = {}, {}, {}, {}
for name, graph in variant["graphs"].items():
    model = onnx.load(str(export / graph), load_external_data=False)
    data = {}
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                if entry.value not in data:
                    data[entry.value] = Path(place(f"{Path(graph).parent.as_posix()}/{entry.value}")).name
                entry.value = data[entry.value]
    graphs[name] = place(graph, model.SerializeToString())
    components[name] = [graphs[name]] + sorted(f"{Path(graph).parent.as_posix()}/{value}" for value in data.values())
    io_schema[name] = schema(model)
    onnx.checker.check_model(str(output / graphs[name]))
shared = sorted(name for name in variant["files"] if "/" not in name)
for name in shared:
    if published.get(name) != digest(export / name):
        raise ValueError(f"{name} differs from the published bundle")
    if not (output / name).exists():
        shutil.copyfile(base / name, output / name)
    sha256[name] = published[name]
variant.update(
    graphs=graphs,
    components=components,
    io_schema=io_schema,
    files=sorted(sha256),
    sha256=dict(sorted(sha256.items())),
    download_bytes=sum((output / name).stat().st_size for name in sha256),
)
if exported.get("exporter") != manifest.get("exporter"):
    variant["exporter"] = exported["exporter"]
manifest["variants"][args.variant] = variant
merged_path.write_text(json.dumps(manifest, indent=2))
print("ASSEMBLED", args.variant, json.dumps(graphs), variant["download_bytes"], flush=True)
