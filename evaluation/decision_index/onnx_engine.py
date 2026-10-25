"""ONNX decision-head engine for the pinned upstream Decision Index runner.

Install valen-sdk[onnx] separately; this adapter does not import Torch.
"""
import hashlib
import importlib.metadata
from pathlib import Path

from decision_index.engines.base import Engine, Unsupported, text


class ValenONNXEngine(Engine):
    name = "valen-onnx"

    def __init__(self, model, variant="fp32", revision=None, threads=4, offline=False):
        super().__init__(model=str(model), variant=variant, revision=revision,
                         threads=threads, offline=offline)
        from valen_sdk import Valen
        self.client = Valen.local(model, variant=variant, revision=revision,
                                  num_threads=int(threads), local_files_only=offline is True or str(offline).lower() == "true",
                                  execution="shared_state")
        directory = self.client._backend.directory
        self.provenance = {
            "kind": "native_decision_head_onnx", "model": str(model), "variant": variant,
            "revision": revision, "model_path": str(directory), "qwen_execution": "shared_state",
            "config_sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest(),
            "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "state_serialization": "upstream text(); unchanged strings and compact UTF-8 JSON otherwise",
            "option_policy": "all supplied options retained in original order; no truncation",
        }

    def __call__(self, state, questions):
        normalized = {}
        for key, question in questions.items():
            item = dict(question)
            item["instructions"] = text(item["instructions"])
            if item["type"] == "choice":
                item["criteria"] = {name: name if value is None else text(value)
                                    for name, value in item["criteria"].items()}
            normalized[key] = item
        try:
            return self.client.predict({"state": text(state), "questions": normalized}), None
        except ValueError as exc:
            if "max_length" in str(exc) or "Reserved tokenizer control token" in str(exc):
                raise Unsupported(str(exc)) from exc
            raise

    def runtime(self):
        import onnxruntime
        return {"hardware": "CPU", "providers": self.client._backend.providers,
                "onnxruntime": onnxruntime.__version__,
                "valen_sdk": importlib.metadata.version("valen-sdk")}

    def close(self):
        self.client.close()
