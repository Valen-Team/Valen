"""Export spatial SFT to a self-contained Hugging Face model and verify reload."""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import tempfile

import torch


RUNTIME = '''
class ValenSpatialConfig(PretrainedConfig):
    model_type = "valen_spatial"

    def __init__(self, backbone_config=None, spatial_settings=None, model_name="Valen-Decision-Det",
                 max_length=4096, max_pixels=262144, backbone_autocast_dtype=None,
                 backbone_storage_dtype=None, merger_fp32=False, **kwargs):
        super().__init__(**kwargs)
        self.backbone_config = portable_config(backbone_config or {}, backbone_storage_dtype)
        self.spatial_settings = spatial_settings or {"task": "det"}
        self.model_name = model_name
        self.max_length = max_length
        self.max_pixels = max_pixels
        self.backbone_autocast_dtype = backbone_autocast_dtype
        self.backbone_storage_dtype = backbone_storage_dtype
        self.merger_fp32 = merger_fp32


class ValenSpatialForDecisionMaking(PreTrainedModel):
    config_class = ValenSpatialConfig
    base_model_prefix = "backbone"
    architecture = "qwen_spatial"
    _supports_sdpa = True
    _keep_in_fp32_modules_strict = ["head"]

    def __init__(self, config):
        super().__init__(config)
        backbone_config = Qwen3_5Config.from_dict(config.backbone_config)
        backbone_config._attn_implementation = "sdpa"
        backbone_config.text_config._attn_implementation = "sdpa"
        backbone_config.vision_config._attn_implementation = "sdpa"
        self.backbone = Qwen3_5Model(backbone_config)
        if config.merger_fp32:
            self.backbone.visual.merger.float()
            self._keep_in_fp32_modules_strict = ["head", "backbone.visual.merger"]
        self.head = SpatialHead(backbone_config.text_config.hidden_size, config.spatial_settings)
        self.spatial_config = config.spatial_settings
        self.model_name = config.model_name
        self.backbone_autocast_dtype = (getattr(torch, config.backbone_autocast_dtype)
                                        if config.backbone_autocast_dtype else None)
        self.post_init()

    _hidden = ValenQwen._hidden
    forward = ValenSpatial.forward
    forward_state = ValenSpatial.forward_state

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        loaded = super().from_pretrained(*args, **kwargs)
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        model.head.float()
        if model.config.merger_fp32:
            model.backbone.visual.merger.float()
        return loaded

    @torch.inference_mode()
    def predict(self, record, media_root="."):
        self.eval()
        processor = AutoProcessor.from_pretrained(self.config._name_or_path, local_files_only=True,
                                                 trust_remote_code=True)
        state = SpatialCompiler(processor, media_root, self.config.max_length,
                                self.config.max_pixels).compile(record)
        return {"model": self.model_name,
                "answers": {q.qid: spatial_response(q, self(q)) for q in state.questions},
                "usage": {"input_tokens": state.logical_tokens, "output_tokens": 0}}
'''

ISOLATED_CHECK = '''
import json, sys
from transformers import AutoModel
model = AutoModel.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True, dtype="auto")
model = model.to(sys.argv[4]).eval()
with open(sys.argv[2]) as stream:
    record = json.loads(next(line for line in stream if line.strip()))
response = model.predict(record, media_root=sys.argv[3])
assert response["usage"]["input_tokens"] > 0 and len(response["answers"]) == 1
assert next(iter(response["answers"].values()))["type"] in {"det", "seg", "gui"}
json.dumps(response, allow_nan=False)
print(json.dumps({"independent_process": True, "input_tokens": response["usage"]["input_tokens"]}))
'''


def portable_config(value, storage_dtype=None):
    """Drop machine-local loader metadata, including nested backbone configs."""
    if isinstance(value, dict):
        return {key: (storage_dtype if storage_dtype and key in {"dtype", "torch_dtype"}
                      else portable_config(item, storage_dtype)) for key, item in value.items()
                if key not in {"_name_or_path", "name_or_path", "model_path", "data_root", "output"}}
    if isinstance(value, list):
        return [portable_config(item, storage_dtype) for item in value]
    return value


def bundle_runtime(source_root, destination):
    metadata = (source_root / "valen/data/types.py").read_text()
    data = (source_root / "valen/data/spatial.py").read_text().replace("from .types import SpatialTarget", "")
    spatial = (source_root / "valen/modeling/qwen/spatial.py").read_text().replace("from .model import ValenQwen", "")
    model_text = (source_root / "valen/modeling/qwen/model.py").read_text()
    model_class = next(node for node in ast.parse(model_text).body if isinstance(node, ast.ClassDef) and node.name == "ValenQwen")
    hidden_method = next(node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == "_hidden")
    hidden = textwrap.indent(textwrap.dedent(ast.get_source_segment(model_text, hidden_method)), "    ")
    exporter = (source_root / "scripts/export/qwen/export_spatial.py").read_text()
    portable = next(node for node in ast.parse(exporter).body if isinstance(node, ast.FunctionDef)
                    and node.name == "portable_config")
    runtime = ("from contextlib import nullcontext\nfrom torch import nn\n"
               "from transformers import PretrainedConfig, PreTrainedModel, Qwen3_5Config, Qwen3_5Model, AutoProcessor\n\n"
               + metadata + "\n" + data + "\n\nclass ValenQwen(nn.Module):\n" + hidden + "\n\n" + spatial + "\n"
               + ast.get_source_segment(exporter, portable) + "\n" + RUNTIME)
    compile(runtime, "modeling_valen_spatial.py", "exec")
    (destination / "modeling_valen_spatial.py").write_text(runtime)


def export(checkpoint, output, model_name, data_path, data_root, device="cuda"):
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoProcessor
    from valen.data.spatial import read_spatial_jsonl
    from valen.modeling.factory import build_model, build_compiler
    from valen.modeling.qwen.spatial import spatial_signature
    from valen.training.checkpoint import load_checkpoint, file_sha256

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    source_root = Path(__file__).resolve().parents[3]
    config = json.loads((checkpoint / "config.json").read_text())
    config.update(device=device, gradient_checkpointing=False, data_root=str(data_root))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Final model directory is not empty")
    temporary = output.with_name(output.name + ".exporting")
    if temporary.exists():
        raise FileExistsError("Inspect the previous incomplete export before retrying")
    temporary.mkdir(parents=True)
    if config.get("method") == "rlcd" and config.get("finetuning_type") == "lora":
        from types import SimpleNamespace
        from valen.training.spatial_rlcd import initialized_policy
        model, _ = initialized_policy(config, SimpleNamespace(primary=True, barrier=lambda: None),
                                      make_reference=False)
    else:
        model = build_model(config)
    load_checkpoint(checkpoint, model)
    if hasattr(model.backbone.language_model, "merge_and_unload"):
        model.backbone.language_model = model.backbone.language_model.merge_and_unload()
    model.requires_grad_(False).eval()
    compiler = build_compiler(config, data_root)
    record = read_spatial_jsonl(data_path, labeled_only=True)[0]
    question = compiler.compile(record).questions[0]
    with torch.inference_mode():
        reference = model(question)
        reference = {name: getattr(reference, name).float().cpu() for name in
                     ("class_logits", "coordinate_logits", "boxes", "mask_logits") if getattr(reference, name) is not None}
    processor = AutoProcessor.from_pretrained(config["model_path"], local_files_only=True)
    processor.save_pretrained(temporary)
    bundle_runtime(source_root, temporary)
    backbone_dtype = str(next(model.backbone.parameters()).dtype).removeprefix("torch.")
    backbone_config = portable_config(model.backbone.config.to_dict(), backbone_dtype)
    compute_dtype = (str(model.backbone_autocast_dtype).removeprefix("torch.")
                     if model.backbone_autocast_dtype is not None else None)
    exported_config = {"model_type": "valen_spatial", "architectures": ["ValenSpatialForDecisionMaking"],
        "auto_map": {"AutoConfig": "modeling_valen_spatial.ValenSpatialConfig",
                     "AutoModel": "modeling_valen_spatial.ValenSpatialForDecisionMaking"},
        "backbone_config": backbone_config, "spatial_settings": spatial_signature(config),
        "model_name": model_name, "max_length": config.get("max_length", 4096),
        "max_pixels": config.get("max_pixels", 262144), "dtype": backbone_dtype, "torch_dtype": backbone_dtype,
        "backbone_autocast_dtype": compute_dtype, "backbone_storage_dtype": backbone_dtype,
        "merger_fp32": (next(model.backbone.visual.merger.parameters()).dtype == torch.float32
                        and backbone_dtype != "float32")}
    (temporary / "config.json").write_text(json.dumps(exported_config, indent=2) + "\n")
    tensors = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(tensors, temporary / "model.safetensors", metadata={"format": "pt"})
    base_manifest = getattr(model, "base_manifest", {}) or {}
    checkpoint_hash = file_sha256(checkpoint / "checkpoint.pt")
    del tensors, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded, loading = AutoModel.from_pretrained(temporary, local_files_only=True, trust_remote_code=True,
                                                dtype="auto", output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Exported spatial model did not reload completely")
    reloaded = reloaded.to(device).eval()
    with torch.inference_mode():
        actual = reloaded(question)
    differences = {}
    for name, expected in reference.items():
        tensor = getattr(actual, name).float().cpu()
        if tensor.shape != expected.shape or not torch.isfinite(tensor).all():
            raise ValueError("Invalid prediction after export")
        difference = float((tensor - expected).abs().max())
        differences[name] = difference
        if not torch.allclose(tensor, expected, atol=2e-3, rtol=2e-3):
            raise ValueError(f"Export changed {name}: {difference}")
    del reloaded, actual
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix=".valen-reload-", dir=output.parent) as cache:
        environment = dict(os.environ, HF_MODULES_CACHE=str(Path(cache) / "modules"),
                           HF_HOME=str(Path(cache) / "hub"), XDG_CACHE_HOME=cache)
        isolated = subprocess.run([sys.executable, "-I", "-c", ISOLATED_CHECK, str(temporary),
            str(Path(data_path).resolve()), str(Path(data_root).resolve()), device],
            cwd=temporary, env=environment, capture_output=True, text=True, timeout=900)
        if isolated.returncode:
            raise RuntimeError("Isolated model inference failed: " + isolated.stderr[-4000:])
    isolated_result = json.loads(isolated.stdout.strip().splitlines()[-1])
    shutil.copyfile(source_root / "LICENSE", temporary / "LICENSE")
    base_directory = Path(config["model_path"])
    if not (base_directory / "LICENSE").is_file():
        raise FileNotFoundError("The pinned backbone snapshot must include its original LICENSE")
    for name in ("LICENSE", "NOTICE"):
        if (base_directory / name).is_file():
            shutil.copyfile(base_directory / name, temporary / (name + ".backbone"))
    (temporary / "requirements.txt").write_text(
        "torch>=2.6\ntransformers>=5.4.0\nsafetensors\nPillow\nnumpy\npycocotools>=2.0.7\n")
    (temporary / "README.md").write_text(f"# {model_name}\n\nComplete Qwen3.5 spatial decision model, including the trained head and processor.\n\n"
        "```python\nfrom transformers import AutoModel\n"
        f'model = AutoModel.from_pretrained("{model_name}", trust_remote_code=True, dtype="auto").cuda()\n'
        "response = model.predict(record, media_root=data_root)\n```\n\n"
        "Install `requirements.txt`. Inputs use the spatial JSONL request schema. "
        "Boxes are normalized xyxy; masks use original-resolution COCO RLE. "
        "The export manifest records immutable source identity and prediction reload checks. "
        "Backbone weights retain their original terms in `LICENSE.backbone` and any `NOTICE.backbone`.\n")
    manifest = {"format": "valen_spatial_full_v1", "model_name": model_name,
                "code_sha256": config.get("code_sha256"), "training_data_sha256": config.get("data_sha256"),
                "data_seal_sha256": config.get("data_seal_sha256"),
                "finetuning_type": config.get("finetuning_type", "lora"),
                "initialization_sha256": config.get("initialization_sha256"),
                "backbone_storage_dtype": backbone_dtype, "backbone_autocast_dtype": compute_dtype,
                "base_repo": base_manifest.get("repo", base_manifest.get("repo_id")),
                "base_revision": base_manifest.get("revision"), "source_checkpoint_sha256": checkpoint_hash,
                "files_sha256": {path.name: file_sha256(path) for path in temporary.iterdir() if path.is_file()},
                "reload_validation": {"max_abs_errors": differences, "atol": 2e-3, "rtol": 2e-3,
                                      **isolated_result}}
    if config.get("method") == "rlcd":
        manifest.update(method="rlcd", initialization_variant=config.get("initialization_variant", "full"),
                        reference_model_name=Path(config["reference_export"]).name,
                        reference_export_manifest_sha256=config.get("initialization_sha256"))
    (temporary / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if output.exists():
        output.rmdir()
    temporary.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--data", required=True, help="One labeled record for reload validation")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = export(args.checkpoint, args.output_dir, args.model_name or Path(args.output_dir).name,
                    args.data, args.data_root, args.device)
    print(json.dumps(result["reload_validation"]), flush=True)


if __name__ == "__main__":
    main()
