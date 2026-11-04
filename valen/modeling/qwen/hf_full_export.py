"""Export full SFT without rounding trained weights. / 无损导出全参数 SFT 权重。"""
from datetime import datetime, timezone
import gc
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import torch
from .grounding import grounding_signature

from .hf_export import bundle_runtime, compare_outputs, dump, evaluate_records, log, sha_file, sha_tensor, validation_records


def export_full_checkpoint(checkpoint, output, work, runtime_source, validation_data,
                           device="cuda:0", attention="flash_attention_2"):
    from transformers import AutoModel, AutoProcessor
    from valen.modeling.factory import build_model
    from valen.modeling.manifest import read_base_manifest
    from valen.training.checkpoint import load_checkpoint

    checkpoint, output, work, runtime_source = map(Path, (checkpoint, output, work, runtime_source))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Export to an empty staging directory: {output}")
    work.mkdir(parents=True, exist_ok=True)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if (config.get("method"), config.get("stage"), config.get("finetuning_type")) != ("sft", "joint", "full"):
        raise ValueError("Expected a full joint SFT checkpoint")
    complete = json.loads((checkpoint.parent / "training_complete.json").read_text(encoding="utf-8"))
    if complete["last_metric"]["epoch_fraction"] != 1:
        raise ValueError("Training has not completed its epoch")
    base = Path(config["model_path"])
    manifest = read_base_manifest(base)
    if not manifest or not manifest.get("weight_sha256"):
        raise ValueError("A verified base manifest is required")
    for name, digest in manifest["weight_sha256"].items():
        if sha_file(base / name) != digest:
            raise ValueError(f"Base weight hash differs: {name}")
    config.update(device=device, attn_implementation=attention, gradient_checkpointing=False)
    torch.manual_seed(config.get("seed", 42))
    package = work / ("runtime_" + output.name.replace("-", "_").replace(".", "_"))
    runtime_hashes = bundle_runtime(runtime_source, package)
    sys.path.insert(0, str(package.parent))
    configuration = importlib.import_module(package.name + ".configuration_valen")
    modeling = importlib.import_module(package.name + ".modeling_valen")
    configuration.ValenQwenConfig.register_for_auto_class()
    modeling.ValenQwenForDecisionMaking.register_for_auto_class("AutoModel")
    log("load_full_checkpoint", model=output.name)
    model = build_model(config)
    payload = load_checkpoint(checkpoint, model)
    step = payload["progress"]["step"]
    if step != complete["last_metric"]["step"]:
        raise ValueError("Checkpoint does not match the final training step")
    if any("lora_" in name for name in model.state_dict()):
        raise ValueError("Full SFT export unexpectedly contains LoRA")
    # All parameters must come from the trained checkpoint. / 确认全部参数由训练权重覆盖。
    if set(payload["weights"]) != set(dict(model.named_parameters())):
        raise ValueError("Full checkpoint does not cover every model parameter")
    for name, tensor in model.named_parameters():
        if tensor.dtype != torch.float32 or sha_tensor(tensor) != sha_tensor(payload["weights"][name]):
            raise ValueError(f"Trained tensor changed during loading: {name}")
    source_sha = sha_file(checkpoint / "checkpoint.pt")
    del payload
    gc.collect()
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    records = validation_records(validation_data)
    log("validate_native", model=output.name)
    reference = evaluate_records(model, modeling.Compiler, processor, config, records)
    dump(work / "native_outputs.json", reference)
    dump(work / "validation_requests.json", [{"name": name, "record": record} for name, record in records])
    exported_config = model.backbone.config.to_dict()
    for key in ("model_type", "architectures", "_name_or_path", "auto_map"):
        exported_config.pop(key, None)
    # Parameter storage and compute precision are independent. / 参数存储与计算精度分别保留。
    exported_config["dtype"] = "float32"
    for field in ("text_config", "vision_config"):
        exported_config[field]["dtype"] = "float32"
    exported_config.update(
        valen_head={key: config[key] for key in
                    ("head_type", "head_width", "head_layers", "head_token_hidden_dim", "head_channel_hidden_dim")},
        qwen_execution=config["qwen_execution"], valen_max_length=config["max_length"],
        valen_media_kwargs=config["media_kwargs"],
        valen_backbone_autocast_dtype="bfloat16" if model.backbone_autocast_dtype == torch.bfloat16 else None,
        valen_full_attention_mask=config.get("full_attention_mask", "causal"),
        valen_grounding=grounding_signature(config),
    )
    standalone = modeling.ValenQwenForDecisionMaking(
        configuration.ValenQwenConfig(**exported_config), backbone=model.backbone, head=model.head, grounding=getattr(model, "grounding", None),
    ).eval().requires_grad_(False)
    wrapped_outputs = evaluate_records(standalone, modeling.Compiler, processor, config, records)
    wrapped_delta = compare_outputs(reference, wrapped_outputs, probability_tolerance=1e-6, logit_tolerance=1e-5)
    if wrapped_delta["changed_labels"]:
        raise ValueError("Wrapping changed prediction labels")
    parameters = {name: sha_tensor(tensor) for name, tensor in standalone.state_dict().items()}
    buffers = {name: sha_tensor(tensor) for name, tensor in standalone.named_buffers()}
    output.mkdir(parents=True, exist_ok=True)
    standalone.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    processor.save_pretrained(output)
    if (base / "LICENSE").exists():
        shutil.copyfile(base / "LICENSE", output / "LICENSE")
    del model, standalone
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("reload_full_standalone", model=output.name)
    reloaded, loading = AutoModel.from_pretrained(
        output, trust_remote_code=True, local_files_only=True, dtype="auto",
        attn_implementation=attention, output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete standalone load: {loading}")
    if parameters != {name: sha_tensor(tensor) for name, tensor in reloaded.state_dict().items()}:
        raise ValueError("Save/load changed model tensors")
    if buffers != {name: sha_tensor(tensor) for name, tensor in reloaded.named_buffers()}:
        raise ValueError("Save/load changed model buffers")
    if {p.dtype for p in reloaded.parameters()} != {torch.float32}:
        raise ValueError("Full SFT parameters must retain FP32 storage")
    reloaded.to(device).eval()
    actual = evaluate_records(reloaded, modeling.Compiler, processor, config, records)
    delta = compare_outputs(reference, actual, probability_tolerance=1e-6, logit_tolerance=1e-5)
    if delta["changed_labels"]:
        raise ValueError("Export changed prediction labels")
    response = reloaded.predict(records[0][1])
    if set(response["answers"]) != set(records[0][1]["request"]["questions"]):
        raise ValueError("Public predict omitted questions")
    validation = {
        "passed": True, "native_vs_wrapper": wrapped_delta, "native_vs_reloaded": delta,
        "all_checkpoint_parameters_bitwise_preserved": True, "all_exported_tensors_bitwise_reloaded": True,
        "all_backbone_buffers_bitwise_preserved": True, "public_predict_passed": True,
        "parameter_dtype": "float32", "backbone_compute_dtype": exported_config["valen_backbone_autocast_dtype"],
        "reload_tolerances": {"probability": 1e-6, "logit": 1e-5},
        "scope": "Text/image/video; Choice/Noul/Score including 12-level Score; question/shared_state. Export checks, not a benchmark run.",
    }
    dump(output / "validation.json", validation)
    dump(work / "reload_diagnostics.json", validation)
    provenance = {
        "format": "valen_qwen_full_sft_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": output.name, "architecture": "Qwen3.5 + two-layer Mixer", "execution": config["qwen_execution"],
        "finetuning_type": "full", "source_checkpoint_sha256": source_sha, "training_step": step,
        "training_records": complete["last_metric"]["total_records"], "training_data_sha256": config.get("data_sha256"),
        "base": {key: value for key, value in manifest.items() if key in {"repo", "repo_id", "revision", "weight_sha256"}},
        "parameter_dtype": "float32", "backbone_compute_dtype": exported_config["valen_backbone_autocast_dtype"],
        "runtime_source_sha256": runtime_hashes,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "safetensors", "flash-attn", "torchvision", "Pillow", "av")},
    }
    provenance["files_sha256"] = {path.name: sha_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    dump(output / "export_manifest.json", provenance)
    dump(output / "COMPLETE.json", {"validated": True, "model": output.name,
                                    "source_checkpoint_sha256": source_sha, "training_step": step})
    log("export_complete", model=output.name, source_checkpoint_sha256=source_sha,
        max_probability_difference=delta["max_probability_abs_error"])
    return output
