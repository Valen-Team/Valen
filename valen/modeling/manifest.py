"""Read a base-model manifest across package directory renames."""
import json
import hashlib
from pathlib import Path


def read_base_manifest(model_path):
    root = Path(model_path)
    preferred = root / "valen_manifest.json"
    if preferred.exists():
        return json.loads(preferred.read_text(encoding="utf-8"))
    # A downloaded model may retain its earlier manifest filename. Accept a
    # single manifest with the expected shape, but never guess among several.
    candidates = []
    for path in root.glob("*_manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        # 发布模型以训练权重摘要标识来源。 / Identify released models by their exported tensors.
        if isinstance(value, dict) and str(value.get("format", "")).startswith("valen_qwen_"):
            weights = {name: digest for name, digest in value.get("files_sha256", {}).items()
                       if name.endswith(".safetensors")}
            if weights and value.get("source_checkpoint_sha256"):
                candidates.append({"revision": "sha256:" + value["source_checkpoint_sha256"],
                                   "weight_sha256": weights})
                continue
        if isinstance(value, dict) and isinstance(value.get("revision"), str) and isinstance(value.get("weight_sha256"), dict):
            candidates.append(value)
    if len(candidates) > 1:
        raise ValueError("Multiple base-model manifests; keep only one or create valen_manifest.json")
    return candidates[0] if candidates else None


def read_model_manifest(config):
    from valen.configuration import flatten_config
    config = flatten_config(config)
    if config.get("architecture", "qwen") in {"qwen", "qwen_spatial"}:
        return read_base_manifest(config["model_path"])
    result = {"architecture": "dual_encoder"}
    for role in ("text", "vision"):
        root = Path(config[f"{role}_model_path"])
        manifest = read_base_manifest(root)
        if manifest is None:
            hashes = {}
            for path in sorted(root.iterdir()):
                if path.suffix in {".safetensors", ".json", ".txt", ".model"} and path.is_file():
                    digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                            digest.update(chunk)
                    hashes[path.name] = digest.hexdigest()
            if not any(name.endswith(".safetensors") for name in hashes):
                raise ValueError(f"No encoder weights found: {root}")
            manifest = {"revision": "local", "files_sha256": hashes}
        result[role] = manifest
    return result


def manifests_match(expected, current):
    """Keep legacy single-backbone matching and strict dual-encoder identity."""
    if expected.get("architecture") == "dual_encoder":
        return expected == current
    return bool(current) and current.get("revision") == expected["revision"] and current.get("weight_sha256") == expected.get("weight_sha256")
