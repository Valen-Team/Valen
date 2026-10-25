"""Export Valen vision and Mixer through the established component exporter."""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModel

from format_export import HeadGraph, VisionGraph, save_external_graph
from format_runtime import vision_inputs

p = argparse.ArgumentParser()
p.add_argument("source")
p.add_argument("output")
args = p.parse_args()
directory = Path(args.output) / "onnx"
directory.mkdir(parents=True, exist_ok=True)
model = (
    AutoModel.from_pretrained(
        args.source, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager"
    )
    .float()
    .cpu()
    .eval()
)
config = model.config.vision_config.to_dict()
patch_width = config["in_channels"] * config["temporal_patch_size"] * config["patch_size"] ** 2
feeds = vision_inputs(torch.zeros(16, patch_width), torch.tensor([[1, 4, 4]]), config)
names = list(feeds)
manifests = {}
manifests["vision"] = save_external_graph(
    VisionGraph(model.backbone.visual),
    tuple(torch.from_numpy(feeds[name]) for name in names),
    names,
    ["visual_features"],
    {
        "pixel_values": {0: "patches"},
        "position_indices": {1: "patches"},
        "position_weights": {1: "patches"},
        "rotary_cos": {0: "patches"},
        "rotary_sin": {0: "patches"},
        "attention_mask": {0: "patches", 1: "patches"},
        "visual_features": {0: "visual_tokens"},
    },
    directory / "vision.onnx",
)
hidden = model.config.text_config.hidden_size
manifests["head"] = save_external_graph(
    HeadGraph(model.head),
    (torch.zeros(3, 4, hidden),),
    ["features"],
    ["logits"],
    {"features": {0: "candidates"}, "logits": {0: "candidates"}},
    directory / "head.onnx",
)
(directory / "tensor_manifest.json").write_text(json.dumps(manifests, indent=2))
print("COMPONENTS_EXPORTED", flush=True)
