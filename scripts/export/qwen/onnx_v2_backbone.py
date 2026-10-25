"""Adapt the pinned Microsoft Qwen3.5 builder to actual Valen weights."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("source")
p.add_argument("output")
p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int4", "int8"])
p.add_argument("--provider", default="cpu")
p.add_argument("--io-dtype", default="auto", choices=["auto", "fp32", "fp16"])
p.add_argument("--upstream", required=True, help="Pinned microsoft/onnxruntime-genai checkout")
args = p.parse_args()
root = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(args.upstream).resolve() / "src/python/py/models"))
import torch
import onnx_ir as ir
from transformers import AutoModel
from builders.qwen import Qwen35TextModel

torch.set_num_threads(8)
source, output = Path(args.source), Path(args.output)
output.mkdir(parents=True, exist_ok=True)
started = time.time()
native = (
    AutoModel.from_pretrained(
        source, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager"
    )
    .float()
    .cpu()
    .eval()
)
text = native.backbone.language_model
config = copy.deepcopy(native.config.text_config)
config.architectures = ["Qwen3_5ForCausalLM"]
config._name_or_path = str(source)


class ValenTextBuilder(Qwen35TextModel):
    def load_weights(self, input_path):
        return text

    def has_final_norm(self, module, orig_model):
        return module is orig_model.norm


options = {"exclude_lm_head": True, "exclude_embeds": True, "filename": "backbone.onnx"}
if args.provider in {"cuda", "webgpu"}:
    options["linear_attn_op"] = "gated_delta_net"
io_half = args.io_dtype == "fp16" or (
    args.io_dtype == "auto" and (args.provider != "cpu" or args.precision == "fp16")
)
builder = ValenTextBuilder(
    config,
    ir.DataType.FLOAT16 if io_half else ir.DataType.FLOAT,
    {
        "fp32": ir.DataType.FLOAT,
        "fp16": ir.DataType.FLOAT16,
        "int4": ir.DataType.INT4,
        "int8": ir.DataType.INT8,
    }[args.precision],
    args.provider,
    str(output / "cache"),
    options,
)
builder.make_model(str(source))
builder.save_model(str(output))
print("EXPORT_COMPLETE", time.time() - started, flush=True)
if args.provider != "cpu":
    (output / "export.json").write_text(
        json.dumps(
            {
                "provider": args.provider,
                "precision": args.precision,
                "io_dtype": "float16" if io_half else "float32",
                "seconds": time.time() - started,
                "runtime_validation": "required",
            },
            indent=2,
        )
    )
    raise SystemExit(0)

import onnxruntime as ort
import numpy as np

opts = ort.SessionOptions()
opts.intra_op_num_threads = 8
session = ort.InferenceSession(str(output / "backbone.onnx"), opts, providers=["CPUExecutionProvider"])
ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
with torch.inference_mode():
    embeds = text.embed_tokens(ids)
    reference = (
        text(
            inputs_embeds=embeds,
            position_ids=torch.arange(8).reshape(1, 1, -1).expand(3, 1, -1),
            use_cache=False,
        )
        .last_hidden_state.float()
        .numpy()
    )
feeds = {}
for item in session.get_inputs():
    if item.name == "inputs_embeds":
        value = embeds.detach().numpy()
    elif item.name == "attention_mask":
        value = np.ones((1, 8), dtype=np.int64)
    elif item.name == "position_ids":
        value = np.broadcast_to(np.arange(8).reshape(1, 1, -1), (3, 1, 8)).copy()
    elif "past" in item.name:
        shape = []
        for dim in item.shape:
            if isinstance(dim, int):
                shape.append(dim)
            elif dim == "batch_size":
                shape.append(1)
            elif dim == "past_sequence_length":
                shape.append(0)
            elif dim == "kv_cache_dim":
                shape.append(config.head_dim)
            else:
                raise ValueError((item.name, item.shape, dim))
        value = np.zeros(shape, dtype=np.float32)
    else:
        raise ValueError((item.name, item.shape))
    feeds[item.name] = value
result = session.run(["hidden_states"], feeds)[0]
error = float(np.max(np.abs(reference - result)))
report = {
    "precision": args.precision,
    "seconds": time.time() - started,
    "hidden_max_abs_error": error,
    "inputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_inputs()],
}
(output / "spike.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
if args.precision == "fp32":
    assert error < 0.03, error
