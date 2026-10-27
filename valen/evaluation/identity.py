"""Bind calibration to checkpoint bytes and an explicit deployment recipe."""
import json
from pathlib import Path
from .calibration import digest
from valen.training.checkpoint import file_sha256


def checkpoint_identity(checkpoint, config, model, protocol, attn_implementation=None):
    checkpoint = Path(checkpoint)
    hashes = {name: file_sha256(checkpoint / name) for name in ("config.json", "checkpoint.pt")}
    # The checkpoint's base_manifest is serialized inside checkpoint.pt. Include an
    # independently stored manifest too when present; local paths are not identities.
    for name in ("base_manifest.json", "model_manifest.json"):
        if (checkpoint / name).exists():
            hashes[name] = file_sha256(checkpoint / name)
    return {"model": "Valen", "revision": hashes["checkpoint.pt"], "artifacts_sha256": hashes,
            "backend": ":".join((config["architecture"], attn_implementation or "checkpoint-default",
                                  str(next(model.parameters()).device.type))),
            "precision": str(next(model.parameters()).dtype), "protocol_sha256": digest(protocol)}


def read_protocol(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
