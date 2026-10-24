"""Merge a trusted SFT checkpoint into a portable HF model. / 导出完整、可搬移的模型。"""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import torch


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def dump(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def log(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def bundle_runtime(source, destination):
    """Freeze the compiler and decision readouts with the weights. / 打包训练时的输入及读出逻辑。"""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "__init__.py").write_text("", encoding="utf-8")
    templates = Path(__file__).parent / "export_templates"
    for name in ("configuration_valen.py", "modeling_valen.py"):
        shutil.copyfile(templates / name, destination / name)
    files = {
        "decisions.py": "valen/decisions.py",
        "heads.py": "valen/modeling/qwen/heads.py",
        "batching.py": "valen/modeling/qwen/batching.py",
        "runtime_model.py": "valen/modeling/qwen/model.py",
        "compiler.py": "valen/data/compilers/qwen.py",
        "schema.py": "valen/data/schema.py",
        "data_types.py": "valen/data/types.py",
    }
    hashes = {}
    for name, relative in files.items():
        path = source / relative
        text = path.read_text(encoding="utf-8")
        if name == "runtime_model.py":
            text = text.replace("from valen import MODEL_NAME", 'MODEL_NAME = "Valen"')
        elif name == "compiler.py":
            text = text.replace("from valen.data.schema import", "from .schema import")
            text = text.replace("from valen.data.types import", "from .data_types import")
        (destination / name).write_text(text, encoding="utf-8")
        hashes[relative] = sha_file(path)
    path = source / "valen/evaluation/inference.py"
    text = path.read_text(encoding="utf-8")
    function = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == "answer")
    (destination / "responses.py").write_text("import math\nimport torch\n\n" + ast.get_source_segment(text, function) + "\n", encoding="utf-8")
    hashes["valen/evaluation/inference.py"] = sha_file(path)
    return hashes


def validation_records(data):
    """Real text/image/video plus a many-question state. / 验证真实模态与多题输入。"""
    data = Path(data)
    questions = {
        "choice": {"type": "choice", "instructions": "Which animal is in the state?",
                   "criteria": {"cat": "A cat", "dog": "A dog", "bird": "A bird"}},
        "noul": {"type": "noul", "instructions": "The state contains a cat."},
        "score": {"type": "score", "instructions": "How many cats are in the state?",
                  "criteria": ["There are no cats.", "There is exactly one cat.", "There are two or more cats."]},
        "score_many": {"type": "score", "instructions": "How many animals are mentioned?",
                       "criteria": [f"Exactly {i} animals are mentioned." for i in range(12)]},
    }
    records = [("text_multi", {"request": {"state": "There is one cat in the room and no other animals.",
                                           "questions": questions}})]
    found = set()
    with (data / "general.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            for qid, question in record["request"]["questions"].items():
                if question["type"] not in found:
                    found.add(question["type"])
                    request = dict(record["request"], questions={qid: question})
                    records.append(("image_" + question["type"], {"request": request}))
            if len(found) == 3:
                break
    assert found == {"choice", "noul", "score"}
    with (data / "video.jsonl").open() as stream:
        video = json.loads(next(stream))
    selected, kinds = {}, set()
    for qid, question in video["request"]["questions"].items():
        if question["type"] not in kinds:
            selected[qid] = question
            kinds.add(question["type"])
    assert kinds == {"choice", "noul", "score"}
    records.append(("video_multi", {"request": dict(video["request"], questions=selected)}))
    return records


@torch.inference_mode()
def evaluate_records(model, compiler_class, processor, config, records):
    # Match FP32 math on both sides of save/load, including DeltaNet fallback.
    # 统一保存前后的 FP32 运算模式，包含 DeltaNet 的 PyTorch 回退路径。
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    results = {}
    model.eval()
    for execution in ("shared_state", "question"):
        compiler = compiler_class(processor, max_length=config["max_length"], media_kwargs=config["media_kwargs"],
                                  execution=execution)
        for name, record in records:
            compiled = compiler.compile(record)
            outputs = model.forward_state(compiled) if execution == "shared_state" else [model(q) for q in compiled.questions]
            for question, logits in zip(compiled.questions, outputs):
                key = execution + "/" + name + "/" + question.qid
                assert torch.isfinite(logits).all()
                results[key] = {"logits": logits.float().cpu().tolist(),
                                "probabilities": logits.float().softmax(-1).cpu().tolist(),
                                "label": question.keys[int(logits.argmax())], "keys": question.keys}
    return results


def compare_outputs(reference, actual, probability_tolerance=None, logit_tolerance=None):
    assert reference.keys() == actual.keys()
    differences, changed = [], []
    for key, old in reference.items():
        new = actual[key]
        assert old["keys"] == new["keys"]
        probability_error = max(abs(a-b) for a,b in zip(old["probabilities"],new["probabilities"]))
        logit_error = max(abs(a-b) for a,b in zip(old["logits"],new["logits"]))
        if probability_tolerance is not None:
            assert probability_error <= probability_tolerance, (key, probability_error)
        if logit_tolerance is not None:
            assert logit_error <= logit_tolerance, (key, logit_error)
        if old["label"] != new["label"]:
            changed.append(key)
        differences.append({"case": key, "probability_max_abs": probability_error, "logit_max_abs": logit_error,
                            "label_changed": old["label"] != new["label"]})
    return {"cases": len(differences), "changed_labels": changed,
            "max_probability_abs_error": max(d["probability_max_abs"] for d in differences),
            "max_logit_abs_error": max(d["logit_max_abs"] for d in differences), "details": differences}


def merge_checkpoint(checkpoint, output, work, runtime_source, validation_data, device="cuda", attention="flash_attention_2"):
    from peft.tuners.lora.layer import LoraLayer
    from transformers import AutoModel, AutoProcessor
    from valen.modeling.factory import build_model
    from valen.modeling.manifest import read_base_manifest
    from valen.training.checkpoint import load_checkpoint

    checkpoint, output, work, runtime_source = map(Path, (checkpoint, output, work, runtime_source))
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("finetuning_type", "lora") == "full":
        from .hf_full_export import export_full_checkpoint
        return export_full_checkpoint(checkpoint, output, work, runtime_source, validation_data, device, attention)
    work.mkdir(parents=True, exist_ok=True)
    if output.exists():
        unexpected = [p.name for p in output.iterdir() if p.name not in {".git", ".gitattributes", "README.md"}]
        if unexpected:
            raise FileExistsError(f"Output already contains model artifacts: {unexpected}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    assert config["method"] == "sft" and config["stage"] == "joint" and config["head_type"] == "mixer"
    complete = json.loads((checkpoint.parent / "training_complete.json").read_text(encoding="utf-8"))
    assert complete["last_metric"]["epoch_fraction"] == 1
    base = Path(config["model_path"])
    manifest = read_base_manifest(base)
    assert manifest and manifest["weight_sha256"]
    for name, digest in manifest["weight_sha256"].items():
        assert sha_file(base / name) == digest, name
    config.update(device=device, attn_implementation=attention, gradient_checkpointing=False)
    torch.manual_seed(config.get("seed", 42))
    package = work / ("runtime_" + output.name.replace("-", "_").replace(".", "_"))
    runtime_hashes = bundle_runtime(runtime_source, package)
    sys.path.insert(0, str(package.parent))
    configuration = importlib.import_module(package.name + ".configuration_valen")
    modeling = importlib.import_module(package.name + ".modeling_valen")
    configuration.ValenQwenConfig.register_for_auto_class()
    modeling.ValenQwenForDecisionMaking.register_for_auto_class("AutoModel")
    log("load_checkpoint", model=output.name)
    model = build_model(config)
    payload = load_checkpoint(checkpoint, model)
    assert payload["progress"]["step"] == complete["last_metric"]["step"]
    head_before = {name: sha_tensor(parameter) for name,parameter in model.head.state_dict().items()}
    merger_before = {name: sha_tensor(parameter) for name,parameter in model.backbone.visual.merger.state_dict().items()}
    source_sha = sha_file(checkpoint / "checkpoint.pt")
    del payload
    gc.collect()
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    records = validation_records(validation_data)
    log("validate_unmerged", model=output.name)
    reference = evaluate_records(model, modeling.Compiler, processor, config, records)
    # Accumulate the effective matrices in FP32, then store BF16 once.
    # FP32 累加 LoRA，最后一次转换为 BF16，避免两次量化 delta。
    expected = {}
    torch.set_float32_matmul_precision("highest")
    backbone_buffers = {name.replace("language_model.base_model.model.", "language_model."):sha_tensor(buffer)
                        for name,buffer in model.backbone.named_buffers()}
    language_buffers = {name.removeprefix("base_model.model."):sha_tensor(buffer)
                        for name,buffer in model.backbone.language_model.named_buffers()}
    with torch.no_grad():
        for name, layer in model.backbone.language_model.named_modules():
            if isinstance(layer, LoraLayer):
                assert layer.active_adapters == ["default"] and not layer.use_dora.get("default", False)
                layer.base_layer.weight.data = layer.base_layer.weight.data.float()
                effective = (layer.base_layer.weight + layer.get_delta_weight("default")).to(torch.bfloat16)
                expected[name.removeprefix("base_model.model.") + ".weight"] = sha_tensor(effective)
                del effective
    assert expected
    model.backbone.language_model = model.backbone.language_model.merge_and_unload(safe_merge=True)
    parameters = dict(model.backbone.language_model.named_parameters())
    # Cast only merged matrices. Casting the whole tower also quantizes RoPE buffers.
    # 只转换合并的矩阵，整体转换会错误地量化 FP32 的 RoPE 频率缓存。
    for name, digest in expected.items():
        parameters[name].data = parameters[name].data.to(torch.bfloat16)
        assert sha_tensor(parameters[name]) == digest, name
    assert language_buffers == {name:sha_tensor(buffer) for name,buffer in model.backbone.language_model.named_buffers()}
    assert backbone_buffers == {name:sha_tensor(buffer) for name,buffer in model.backbone.named_buffers()}
    assert not any("lora_" in name or "base_layer" in name for name in model.state_dict())
    assert head_before == {name:sha_tensor(p) for name,p in model.head.state_dict().items()}
    assert merger_before == {name:sha_tensor(p) for name,p in model.backbone.visual.merger.state_dict().items()}
    merged_config = model.backbone.config.to_dict()
    for key in ("model_type", "architectures", "_name_or_path", "auto_map"):
        merged_config.pop(key, None)
    merged_config.update(valen_head=model.head_config if hasattr(model, "head_config") else
                         {key:config[key] for key in ("head_type","head_width","head_layers","head_token_hidden_dim","head_channel_hidden_dim")},
                         qwen_execution=config["qwen_execution"], valen_max_length=config["max_length"],
                         valen_media_kwargs=config["media_kwargs"])
    merged_config = configuration.ValenQwenConfig(**merged_config)
    merged = modeling.ValenQwenForDecisionMaking(merged_config, backbone=model.backbone, head=model.head).eval()
    merged.requires_grad_(False)
    assert head_before == {name:sha_tensor(p) for name,p in merged.head.state_dict().items()}
    log("validate_merged", model=output.name, merged_linear_layers=len(expected))
    merged_outputs = evaluate_records(merged, modeling.Compiler, processor, config, records)
    merged_repeat = evaluate_records(merged, modeling.Compiler, processor, config, records)
    repeat_delta = compare_outputs(merged_outputs, merged_repeat)
    dump(work / "merged_outputs.json", merged_outputs)
    dump(work / "validation_requests.json", [{"name":name, "record":record} for name,record in records])
    # Separate LoRA branches and fused BF16 matrices have different rounding.
    # 分支 LoRA 与融合 BF16 矩阵有不同的舍入，逐项记录差异而不假定二者数值相等。
    delta = compare_outputs(reference, merged_outputs)
    dump(work / "merge_diagnostics.json", delta)
    dump(work / "adapter_outputs.json", reference)
    stage = work / "staging" / output.name
    stage.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(stage, safe_serialization=True, max_shard_size="4GB")
    processor.save_pretrained(stage)
    if (base / "LICENSE").exists():
        shutil.copyfile(base / "LICENSE", stage / "LICENSE")
    merged_parameters = {name:sha_tensor(p) for name,p in merged.state_dict().items()}
    del parameters, model, merged
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("reload_standalone", model=output.name)
    reloaded, loading = AutoModel.from_pretrained(stage, trust_remote_code=True, local_files_only=True,
                                                dtype=torch.bfloat16, attn_implementation=attention,
                                                output_loading_info=True)
    assert not any(loading.get(key) for key in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")), loading
    assert {p.dtype for p in reloaded.head.parameters()} == {torch.float32}
    assert merged_parameters == {name:sha_tensor(p) for name,p in reloaded.state_dict().items()}
    assert language_buffers == {name:sha_tensor(buffer) for name,buffer in reloaded.backbone.language_model.named_buffers()}
    assert backbone_buffers == {name:sha_tensor(buffer) for name,buffer in reloaded.backbone.named_buffers()}
    reloaded.to(device).eval()
    reloaded_outputs = evaluate_records(reloaded, modeling.Compiler, processor, config, records)
    reload_delta = compare_outputs(merged_outputs, reloaded_outputs)
    dump(work / "reload_diagnostics.json", {"same_model_repeat":repeat_delta, "reload":reload_delta,
                                          "reloaded_outputs":reloaded_outputs})
    compare_outputs(merged_outputs, reloaded_outputs, probability_tolerance=1e-6, logit_tolerance=1e-5)
    assert not reload_delta["changed_labels"]
    response = reloaded.predict(records[0][1])
    assert set(response["answers"]) == set(records[0][1]["request"]["questions"])
    validation = {"passed":True,"merge_vs_adapter":delta,"reload_vs_merged":reload_delta,
                  "all_effective_lora_matrices_verified":True,"head_and_merger_bitwise_preserved":True,
                  "all_exported_tensors_bitwise_reloaded":True,"language_buffers_bitwise_preserved":True,
                  "all_backbone_buffers_bitwise_preserved":True,"float32_matmul_precision":"highest",
                  "same_model_repeat":repeat_delta,"reload_tolerances":{"probability":1e-6,"logit":1e-5},
                  "public_predict_passed":True,
                  "scope":"Text/image/video, Choice/Noul/Score, 12-level Score, question/shared_state; this is export validation, not benchmark evaluation."}
    dump(stage / "validation.json", validation)
    (stage / "MODEL_USAGE.md").write_text(f'''# {output.name}

完整合并的 Qwen3.5 + 两层 Mixer，提供 Choice、Noul、Score 决策和 share_state 多题推理。
Complete Qwen3.5 backbone with merged LoRA, trained visual merger, and the FP32 Mixer head.

训练版本：1195k + JevBench 30k，两阶段 SFT，最终 step {complete["last_metric"]["step"]}。
全体基座参数、训练参数、分词器及输入编译代码均在本目录。

## 加载 / Load

验证环境：Python 3.10、PyTorch 2.6.0、Transformers 5.4.0、torchvision 0.21.0、Pillow、av 16.1.0。
使用 Flash Attention 2 时，另需与 PyTorch/CUDA 匹配的 flash-attn。

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "./{output.name}",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",  # 或 flash_attention_2 / or flash_attention_2
).to("cuda").eval()
# 和导出一致性验证保持相同的 FP32 精度。 / Match export verification precision.
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

response = model.predict({{
    "state": "There is one cat in the room.",
    "questions": {{
        "animal": {{"type": "choice", "instructions": "Which animal is present?",
                   "criteria": {{"cat": "A cat", "dog": "A dog"}}}},
        "is_cat": {{"type": "noul", "instructions": "There is a cat in the room."}},
        "count": {{"type": "score", "instructions": "How many cats are present?",
                  "criteria": ["No cats", "Exactly one cat", "Two or more cats"]}},
    }},
}})
print(response["answers"])
```

图像/视频使用 Valen 的 messages、image_url、video_url 输入格式，媒体路径由 `media_root` 解析。
默认 `execution="shared_state"`；`model.predict(..., execution="question")` 可使用独立问题路径。
视频沿用训练配置，采样 16 帧。所有 Mixer 参数保持 FP32，基座权重为 BF16。

`validation.json` 记录合并前后概率差异、重新加载一致性、文本/图像/视频及三类题型检查。
`export_manifest.json` 记录来源 checkpoint、基座版本及文件 SHA-256。
LoRA 以 FP32 累加后保存为 BF16；融合矩阵的舍入顺序与独立 LoRA 分支不同。
验证样本中最大概率绝对差：{delta["max_probability_abs_error"]:.8f}；预测标签变化：{len(delta["changed_labels"])} / {delta["cases"]}。
该检查只验证导出样本，不能代替完整 benchmark；之前 checkpoint 的成绩属于未融合版本。
''', encoding="utf-8")
    provenance = {"format":"valen_qwen_merged_v1","created_utc":datetime.now(timezone.utc).isoformat(),
                  "model":output.name,"architecture":"Qwen3.5 + two-layer Mixer", "execution":config["qwen_execution"],
                  "source_checkpoint_sha256":source_sha,"training_step":complete["last_metric"]["step"],
                  "training_records":complete["last_metric"]["total_records"],"training_data_sha256":config.get("data_sha256"),
                  "base":manifest,"merge":"FP32 base+LoRA accumulation, stored BF16; trained visual merger and FP32 Mixer retained",
                  "merged_linear_layers":len(expected),"runtime_source_sha256":runtime_hashes,
                  "versions":{p:importlib.metadata.version(p) for p in ("torch","transformers","peft","safetensors","flash-attn","torchvision","Pillow","av")}}
    provenance["files_sha256"] = {p.name:sha_file(p) for p in sorted(stage.iterdir()) if p.is_file()}
    dump(stage / "export_manifest.json", provenance)
    dump(work / (output.name+".validation.json"), validation)
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(stage.iterdir()):
        if path.is_file():
            shutil.copyfile(path, output / path.name)
            assert sha_file(output / path.name) == sha_file(path)
    dump(output / "COMPLETE.json", {"model":output.name,"validated":True,"files":len(provenance["files_sha256"]),
                                    "source_checkpoint_sha256":source_sha})
    log("export_complete", model=output.name, output=str(output), max_probability_difference=delta["max_probability_abs_error"],
        changed_validation_labels=delta["changed_labels"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--runtime-source", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", choices=("eager","sdpa","flash_attention_2"), default="flash_attention_2")
    args = parser.parse_args()
    merge_checkpoint(args.checkpoint, args.output, args.work, args.runtime_source, args.validation_data,
                     args.device, args.attn_implementation)


if __name__ == "__main__":
    main()
