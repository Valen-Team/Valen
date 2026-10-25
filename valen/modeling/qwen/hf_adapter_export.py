"""Export unmerged LoRA, Mixer and visual merger. / 导出未融合的完整训练增量。"""
import argparse
from datetime import datetime, timezone
import gc
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import torch

from .hf_export import bundle_responses, compare_outputs, dump, evaluate_records, log, sha_file, sha_tensor, validation_records


def bundle_adapter_runtime(merged_model, stage):
    """Keep checkpoint-specific internals and add the current public predict API."""
    merged_model, stage = Path(merged_model), Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    for name in ("configuration_valen.py", "heads.py", "compiler.py", "runtime_model.py",
                 "batching.py", "schema.py", "data_types.py", "tokenizer.json",
                 "tokenizer_config.json", "processor_config.json", "chat_template.jinja", "LICENSE"):
        shutil.copyfile(merged_model / name, stage / name)
    templates = Path(__file__).parent / "export_templates"
    for name in ("modeling_valen.py", "modeling_valen_lora.py"):
        shutil.copyfile(templates / name, stage / name)
    source = Path(__file__).resolve().parents[3]
    shutil.copyfile(source / "valen/decisions.py", stage / "decisions.py")
    bundle_responses(source, stage)


def export_adapters(checkpoint, merged_model, work, validation_data, device="cuda:0", attention="flash_attention_2"):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from transformers import AutoModel, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from valen.modeling.factory import build_model
    from valen.training.checkpoint import load_checkpoint

    checkpoint, merged_model, work = map(Path, (checkpoint, merged_model, work))
    output = merged_model / "unmerged"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Adapter output already contains files: {output}")
    work.mkdir(parents=True, exist_ok=True)
    stage = work / "staging" / "unmerged"
    stage.mkdir(parents=True, exist_ok=True)
    parent = json.loads((merged_model / "export_manifest.json").read_text(encoding="utf-8"))
    source_sha = sha_file(checkpoint / "checkpoint.pt")
    assert source_sha == parent["source_checkpoint_sha256"], "Merged and unmerged versions must have the same source"
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    assert config["stage"] == "joint" and config["head_type"] == "mixer"
    config.update(device=device, attn_implementation=attention, gradient_checkpointing=False)
    torch.manual_seed(config.get("seed", 42))
    log("load_adapter_source", model=merged_model.name)
    model = build_model(config)
    payload = load_checkpoint(checkpoint, model)
    assert payload["progress"]["step"] == parent["training_step"]
    source_weights = {name:sha_tensor(tensor) for name,tensor in payload["weights"].items()}
    del payload
    gc.collect()
    buffers = {name:sha_tensor(tensor) for name,tensor in model.backbone.named_buffers()}
    processor = AutoProcessor.from_pretrained(merged_model, trust_remote_code=True, local_files_only=True)
    runtime_cls = get_class_from_dynamic_module("modeling_valen.ValenQwenForDecisionMaking", str(merged_model),
                                               local_files_only=True)
    runtime = importlib.import_module(runtime_cls.__module__)
    records = validation_records(validation_data)
    reference = evaluate_records(model, runtime.Compiler, processor, config, records)
    dump(work / "adapter_outputs.json", reference)
    dump(work / "validation_requests.json", [{"name":name,"record":record} for name,record in records])
    bundle_adapter_runtime(merged_model, stage)
    bundle_config = json.loads((merged_model / "config.json").read_text(encoding="utf-8"))
    bundle_config.update(architectures=["ValenQwenWithLoRA"], valen_base=parent["base"],
                         auto_map={"AutoConfig":"configuration_valen.ValenQwenConfig",
                                   "AutoModel":"modeling_valen_lora.ValenQwenWithLoRA"})
    dump(stage / "config.json", bundle_config)
    adapter = model.backbone.language_model
    adapter.peft_config["default"].base_model_name_or_path = parent["base"]["repo"]
    adapter.peft_config["default"].revision = parent["base"]["revision"]
    adapter.save_pretrained(stage / "lora", safe_serialization=True, save_embedding_layers=False)
    expected_lora = get_peft_model_state_dict(adapter, save_embedding_layers=False)
    assert {name:sha_tensor(t) for name,t in expected_lora.items()} == {
        name:sha_tensor(t) for name,t in load_file(stage / "lora/adapter_model.safetensors").items()}
    for name, module in (("mixer",model.head),("visual_merger",model.backbone.visual.merger)):
        save_file({key:value.detach().cpu().contiguous().clone() for key,value in module.state_dict().items()},
                  stage / (name+".safetensors"))
    log("reload_unmerged", model=merged_model.name, tensors=len(source_weights))
    del expected_lora, adapter, model
    gc.collect()
    torch.cuda.empty_cache()
    loaded, loading = AutoModel.from_pretrained(stage, base_model_path=config["model_path"], trust_remote_code=True,
        dtype=torch.bfloat16, attn_implementation=attention, local_files_only=True, output_loading_info=True)
    assert {p.dtype for p in loaded.head.parameters()} == {torch.float32}
    state = loaded.state_dict()
    assert source_weights == {name:sha_tensor(state[name]) for name in source_weights}
    assert buffers == {name:sha_tensor(tensor) for name,tensor in loaded.backbone.named_buffers()}
    loaded.to(device).eval()
    actual = evaluate_records(loaded, runtime.Compiler, processor, config, records)
    delta = compare_outputs(reference, actual, probability_tolerance=1e-6, logit_tolerance=1e-5)
    assert not delta["changed_labels"]
    response = loaded.predict(records[0][1])
    assert set(response["answers"]) == set(records[0][1]["request"]["questions"])
    validation = {"passed":True,"all_training_deltas_bitwise_reloaded":True,"all_backbone_buffers_bitwise_preserved":True,
                  "trainable_tensor_count":len(source_weights),"reload_vs_checkpoint":delta,"public_predict_passed":True,
                  "scope":"Text/image/video; question/shared_state; Choice/Noul/Score, including 12-level Score."}
    dump(stage / "validation.json", validation)
    (stage / "README.md").write_text(f'''# {merged_model.name} — unmerged LoRA

原始 Qwen3.5 基座 + 未合并的 LoRA + FP32 Mixer + 训练后的视觉 merger。
Original base with separate LoRA, Mixer, and trained visual merger weights.
来源：1195k + JevBench 30k，两阶段 SFT，最终 step {parent["training_step"]}。

## 文件 / Files

- `lora/adapter_model.safetensors`、`lora/adapter_config.json`：标准 PEFT LoRA。
- `mixer.safetensors`：完整 FP32 Mixer 决策头。
- `visual_merger.safetensors`：联合微调时训练的视觉 merger，加载时一并恢复。
- `config.json`、processor、tokenizer 和 Python 文件：模型配置及独立加载代码。

## 完整模型加载 / Load the complete decision model

```bash
python -m pip install torch torchvision transformers pillow av peft safetensors
```
Flash Attention 2 可通过 `attn_implementation="flash_attention_2"` 启用。

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "./{merged_model.name}/unmerged",
    base_model_path="./{parent["base"]["repo"].split("/")[-1]}",  # 原始官方基座 / original published base
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda").eval()
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

result = model.predict({{
    "state": "There is one cat in the room.",
    "questions": {{
        "animal": {{"type": "choice", "instructions": "Which animal is present?",
                   "criteria": {{"cat": "A cat", "dog": "A dog"}}}},
        "is_cat": {{"type": "noul", "instructions": "There is a cat in the room."}},
    }},
}})
print(result["answers"])
```

同一个模型也接受 Decisions 请求：

```python
result = model.predict({{
    "input": "There is one cat in the room.",
    "questions": [{{"name": "is_cat", "type": "predicate",
                   "instructions": "There is a cat in the room."}}],
}}, temperature=1.0)
print(result["answers"][0]["probability"])
```

`base_model_path` 指向原始官方 `{parent["base"]["repo"]}` 权重，本目录中的加载器会核对基座 SHA-256。
省略该参数时使用配置记录的仓库及固定 revision；离线环境需提前缓存该版本。
默认 shared_state，也支持 `model.predict(..., execution="question")`；视频采样 16 帧。
两种协议均支持本地图片和视频；媒体用法见 https://valen-team.github.io/docs/valen/quickstart/。
内部使用 `PeftModel.from_pretrained(backbone.language_model, .../lora)` 加载语言 LoRA，随后恢复 Mixer 和视觉 merger。
适配器不会被融合。`is_trainable=True` 可启用 LoRA、Mixer 和视觉 merger 的梯度。

`validation.json` 记录恢复后与原 checkpoint 的一致性；文件来源和校验和见 `export_manifest.json`。
''', encoding="utf-8")
    provenance = {"format":"valen_qwen_unmerged_v1","created_utc":datetime.now(timezone.utc).isoformat(),
                  "model":merged_model.name,"source_checkpoint_sha256":source_sha,"training_step":parent["training_step"],
                  "training_records":parent["training_records"],"training_data_sha256":parent["training_data_sha256"],
                  "base":parent["base"],"execution":config["qwen_execution"],"trainable_tensor_count":len(source_weights),
                  "versions":{name:importlib.metadata.version(name) for name in ("torch","transformers","peft","safetensors")}}
    provenance["files_sha256"] = {str(p.relative_to(stage)):sha_file(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    dump(stage / "export_manifest.json", provenance)
    shutil.copytree(stage, output, dirs_exist_ok=True)
    for name, digest in provenance["files_sha256"].items():
        assert sha_file(output / name) == digest, name
    dump(output / "COMPLETE.json", {"validated":True,"source_checkpoint_sha256":source_sha})
    log("unmerged_export_complete", model=merged_model.name, output=str(output),
        max_probability_difference=delta["max_probability_abs_error"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","merged-model","work","validation-data"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--attn-implementation",default="flash_attention_2",choices=("eager","sdpa","flash_attention_2"))
    args = parser.parse_args()
    export_adapters(args.checkpoint,args.merged_model,args.work,args.validation_data,args.device,args.attn_implementation)


if __name__ == "__main__":
    main()
