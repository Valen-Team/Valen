# ruff: noqa: E402
# Load the installed entry point before importing patched engine API bindings.
"""Plugin boundaries that native model tests do not exercise."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.plugins import load_plugins

load_plugins()

from valen_sglang import hooks
from valen_sglang.head import ValenMixerHead, ValenPooler
from valen_sglang.plugin import register
from valen_sglang.protocol import SystemOneRequest


class TestPlugin(unittest.TestCase):
    def test_registration_is_idempotent(self):
        from sglang.srt.plugins.hook_registry import HookRegistry

        before = {key: len(value) for key, value in HookRegistry._hooks.items()}
        register()
        register()
        self.assertEqual(
            before, {key: len(value) for key, value in HookRegistry._hooks.items()}
        )

    def test_http_route_uses_extended_request_schema(self):
        from sglang.srt.entrypoints.http_server import app

        route = next(
            route
            for route in app.routes
            if getattr(route, "path", None) == "/v1/systemone"
        )
        self.assertIs(route.body_field.field_info.annotation, SystemOneRequest)
        payload = {
            "model": "valen",
            "state": "Rank this",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "Rank",
                    "criteria": [str(i) for i in range(255)],
                }
            },
        }
        self.assertEqual(
            len(
                route.body_field.field_info.annotation.model_validate(payload)
                .questions["q"]
                .criteria
            ),
            255,
        )

    def test_temperature_extension_overrides_only_the_stock_temperature_rejection(self):
        from pydantic import ValidationError

        payload = {
            "model": "valen",
            "state": "hello",
            "questions": {"q": {"type": "noul", "instructions": "Yes?"}},
        }
        self.assertEqual(SystemOneRequest(**payload, temperature=0.5).temperature, 0.5)
        for temperature in (0, -1, True, float("inf"), float("nan")):
            with (
                self.subTest(temperature=temperature),
                self.assertRaises(ValidationError),
            ):
                SystemOneRequest(**payload, temperature=temperature)
        for field in ("prompt_format_version", "return_prompt_token_ids"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                SystemOneRequest(**payload, **{field: True})

    def test_image_processor_keyword_argument_and_other_models(self):
        processor = SimpleNamespace(image_processor=SimpleNamespace(size={"old": 1}))
        config = SimpleNamespace(
            valen_head={},
            valen_media_kwargs={
                "images_kwargs": {
                    "size": {"shortest_edge": 65536, "longest_edge": 262144}
                }
            },
        )
        hooks.prepare_image_processor(None, config, None, _processor=processor)
        self.assertEqual(
            processor.image_processor.size,
            config.valen_media_kwargs["images_kwargs"]["size"],
        )
        hooks.prepare_image_processor(
            None, SimpleNamespace(), None, _processor=processor
        )
        self.assertEqual(processor.image_processor.size["shortest_edge"], 65536)

    def test_head_loading_reports_loaded_parameters(self):
        head = ValenMixerHead(
            8,
            {
                "head_width": 4,
                "head_layers": 1,
                "head_token_hidden_dim": 3,
                "head_channel_hidden_dim": 8,
            },
        )
        reference = {
            key: torch.randn_like(value) for key, value in head.state_dict().items()
        }
        model = SimpleNamespace(pooler=ValenPooler(head))
        weights = [("head." + key, value) for key, value in reference.items()]
        weights.append(("backbone.language_model.embed_tokens.weight", torch.ones(1)))
        received = []

        def parent(self, values):
            received.extend(name for name, _ in values)
            return set(received)

        loaded = hooks.load_weights(parent, model, iter(weights))
        self.assertEqual(received, ["model.language_model.embed_tokens.weight"])
        self.assertTrue({"pooler.head." + key for key in reference} <= loaded)
        for key, value in reference.items():
            torch.testing.assert_close(head.state_dict()[key], value, rtol=0, atol=0)

    def test_fp32_checkpoint_auto_dtype_does_not_downcast(self):
        from valen_sglang.configuration import ValenQwenConfig

        from sglang.srt.configs.model_config import ModelConfig

        with tempfile.TemporaryDirectory() as directory:
            config = ValenQwenConfig(architectures=["ValenQwenForDecisionMaking"])
            Path(directory, "config.json").write_text(config.to_json_string())
            loaded = ModelConfig(directory, trust_remote_code=False)
            self.assertEqual(loaded.dtype, torch.float32)
            explicit = ModelConfig(directory, trust_remote_code=False, dtype="bfloat16")
            self.assertEqual(explicit.dtype, torch.bfloat16)

    def test_other_models_keep_stock_config_and_loader(self):
        config = SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"])
        self.assertIs(hooks.normalize_config(config), config)
        self.assertFalse(hasattr(config, "valen_head"))
        obj = SimpleNamespace(pooler=object())
        weights = object()
        sentinel = object()
        self.assertIs(
            hooks.load_weights(
                lambda self, value: sentinel if value is weights else None, obj, weights
            ),
            sentinel,
        )


if __name__ == "__main__":
    unittest.main()
