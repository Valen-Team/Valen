import torch
from torch import nn
from .model import ValenQwen
from ..manifest import read_base_manifest
from .heads import head_signature
from .attention import attention_implementation
from .block_attention import MASKS, enable as enable_block_attention
from .grounding import GroundingHead, grounding_signature
from .pretrained import published_config, with_published_settings


def build_qwen(config):
    config = with_published_settings(config)
    head_config = head_signature(config)
    attention = attention_implementation(config)
    mask = config.get("full_attention_mask", "causal")
    if mask not in MASKS:
        raise ValueError(f"full_attention_mask must be one of {MASKS}, got {mask!r}")
    if mask != "causal" and attention == "eager":
        raise ValueError(f"{mask} full attention requires sdpa or flash_attention_2")
    grounding = grounding_signature(config)
    from transformers import AutoModel, Qwen3_5ForConditionalGeneration
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[config.get("dtype", "bf16")]
    published = published_config(config["model_path"]) is not None
    if published:
        # 保留发布权重的精度和已训练 Mixer。 / Preserve released tensors and the trained Mixer.
        container, loading_info = AutoModel.from_pretrained(
            config["model_path"], trust_remote_code=True, dtype="auto", attn_implementation=attention,
            local_files_only=True, output_loading_info=True)
    else:
        # Loading Qwen3_5Model directly can silently remap language prefixes in Transformers 5.4.
        container, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
            config["model_path"], dtype=dtype, attn_implementation=attention,
            local_files_only=True, output_loading_info=True)
    if any(loading_info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Incomplete pretrained model load: {loading_info}")
    backbone = container.backbone if published else container.model
    head = container.head.float().requires_grad_(True) if published else None
    # 保留已训练的定位头。 / Keep the trained grounding head.
    trained_grounding = getattr(container, "grounding", None) if published else None
    if trained_grounding is not None and grounding_signature({"grounding_head": container.config.valen_grounding}) != grounding:
        trained_grounding = None
    del container  # Never execute or retain the generation/vocabulary output layer.
    backbone.requires_grad_(False)
    stage = config.get("stage", "joint")
    if stage not in {"warmup", "text", "joint", "vision_top"}:
        raise ValueError(f"Unknown stage: {stage}")
    targets = []
    finetuning = config.get("finetuning_type", "lora")
    if finetuning == "full" and stage == "joint":
        # 全参数更新语言及视觉主干。 / Update the entire language and vision backbone.
        # FSDP casts its local shards to FP32 after sharding. / FSDP 分片后再转 FP32。
        (backbone if config.get("fsdp") else backbone.float()).requires_grad_(True)
    elif stage != "warmup":
        from peft import LoraConfig, get_peft_model
        # Enumerate exact language Linear paths. This covers DeltaNet, full attention
        # and FFN, without accidentally adapting the visual tower or decision head.
        targets = [name for name, module in backbone.language_model.named_modules() if isinstance(module, nn.Linear)]
        backbone.language_model = get_peft_model(backbone.language_model, LoraConfig(
            r=config.get("lora_rank", 32), lora_alpha=config.get("lora_alpha", 64),
            lora_dropout=0.0, target_modules=targets, bias="none"))
    if stage in {"joint", "vision_top"}:
        backbone.visual.merger.requires_grad_(True)
    if stage == "vision_top":
        for block in backbone.visual.blocks[-4:]:
            block.requires_grad_(True)
    if config.get("gradient_checkpointing", True) and stage != "warmup":
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = ValenQwen(backbone, head_config=head_config, head=head)
    if grounding is not None:
        model.grounding = (trained_grounding.float().requires_grad_(True) if trained_grounding is not None else
                           GroundingHead(backbone.config.text_config.hidden_size, grounding["width"], grounding["hidden"]))
    if dtype == torch.bfloat16 and (published or (finetuning == "full" and stage == "joint")):
        model.backbone_autocast_dtype = torch.bfloat16
    model.base_manifest = read_base_manifest(config["model_path"])
    # FSDP shards on CPU first, then moves local shards. / FSDP 先在 CPU 分片再上 GPU。
    model.to("cpu" if config.get("fsdp") else config.get("device", "cuda"))
    model.lora_targets = targets
    model.full_attention_mask = mask
    if mask != "causal":
        enable_block_attention(backbone)
    model.attention_backends = {
        name: getattr(getattr(backbone.config, field, None), "_attn_implementation", attention)
        for name, field in (("text", "text_config"), ("vision", "vision_config"))}
    return model


def qwen_optimizer_groups(model, config):
    groups = {"head": [], "grounding": [], "lora": [], "language": [], "merger": [], "vision": []}
    rates = {"head": 2e-4, "grounding": config.get("head_lr", 2e-4), "lora": 5e-5, "language": 1e-5, "merger": 1e-5, "vision": 2e-6}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("head."):
            group = "head"
        elif name.startswith("grounding."):
            group = "grounding"
        elif "lora_" in name:
            group = "lora"
        elif ".visual.merger." in name:
            group = "merger"
        elif ".visual.blocks." in name:
            group = "vision"
        elif config.get("finetuning_type", "lora") == "full" and name.startswith("backbone.language_model."):
            group = "language"
        elif config.get("finetuning_type", "lora") == "full" and name.startswith("backbone.visual."):
            group = "vision"
        else:
            raise ValueError(f"Unclassified trainable parameter: {name}")
        groups[group].append(parameter)
    return [{"params": parameters, "lr": config.get(f"{name}_lr", rates[name]), "name": name}
            for name, parameters in groups.items() if parameters]
