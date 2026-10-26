"""Separate spatial contracts with shared Qwen loading and adaptation."""
from ..interfaces import ArchitectureBackend, Capabilities
from .spatial import ValenSpatial, spatial_signature


class SpatialBackend(ArchitectureBackend):
    name = "qwen_spatial"
    unit_scope = "state"
    capabilities = Capabilities(max_images=1)

    def build_model(self, config):
        from .builder import build_qwen
        spatial_signature(config)
        base = build_qwen(dict(config, head_type="bilinear"))
        model = ValenSpatial(base.backbone, config)
        model.backbone_autocast_dtype = base.backbone_autocast_dtype
        model.base_manifest = base.base_manifest
        model.lora_targets = base.lora_targets
        model.attention_backends = base.attention_backends
        return model.to(config.get("device", "cuda"))

    def build_compiler(self, config, media_root):
        from transformers import AutoProcessor
        from valen.data.spatial import SpatialCompiler
        return SpatialCompiler(AutoProcessor.from_pretrained(config["model_path"], local_files_only=True),
                               config.get("data_root", media_root), config.get("max_length", 4096),
                               config.get("max_pixels", 262144))

    def optimizer_groups(self, model, config):
        from .builder import qwen_optimizer_groups
        return qwen_optimizer_groups(model, config)

    def base_paths(self, config):
        return (config["model_path"],)

    def adaptation(self, model, config):
        return {"stage": config.get("stage", "joint"), "finetuning_type": config.get("finetuning_type", "lora"),
                "head": spatial_signature(config), "lora_targets": model.lora_targets}

    def validate_initialization(self, previous, config):
        super().validate_initialization(previous, config)
        if spatial_signature(previous) != spatial_signature(config):
            raise ValueError("Spatial head configuration differs from initialization")
