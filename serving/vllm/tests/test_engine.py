"""Integration boundaries; run with the installed wheel and vLLM."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


@unittest.skipUnless(os.environ.get("VALEN_TEST_VLLM") == "1", "requires vLLM runtime")
class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from valen_vllm import register

        register()
        from valen_vllm.model import ValenQwenForDecisionMaking
        from valen_vllm.pooler import ValenPooler

        cls.model_cls = ValenQwenForDecisionMaking
        cls.pooler_cls = ValenPooler

    def test_registration_and_native_pooling_model(self):
        from valen_vllm import register
        from vllm import ModelRegistry
        from vllm.model_executor.models.interfaces import supports_multimodal
        from vllm.model_executor.models.interfaces_base import is_pooling_model

        register()
        register()
        self.assertIn("ValenQwenForDecisionMaking", ModelRegistry.get_supported_archs())
        self.assertTrue(is_pooling_model(self.model_cls))
        self.assertTrue(supports_multimodal(self.model_cls))

    def test_config_loads_without_remote_code(self):
        from vllm.engine.arg_utils import EngineArgs

        args = EngineArgs(
            model=os.environ["VALEN_TEST_MODEL"],
            runner="pooling",
            dtype="bfloat16",
            max_model_len=8192,
            max_num_batched_tokens=8192,
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            limit_mm_per_prompt={"image": 16, "video": 0},
        )
        config = args.create_engine_config()
        self.assertEqual(
            config.model_config.hf_config.architectures, ["ValenQwenForDecisionMaking"]
        )
        self.assertEqual(config.model_config.dtype, torch.bfloat16)
        self.assertFalse(config.scheduler_config.enable_chunked_prefill)

    def test_real_metadata_batching_and_prefix_rejection(self):
        import numpy as np
        from valen_vllm.head import (
            ValenMixerHead,
            ValenQuestion,
            pack_valen_layout,
            pool_features,
        )
        from vllm.pooling_params import PoolingParams
        from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

        head = ValenMixerHead(
            6,
            {
                "head_width": 4,
                "head_layers": 1,
                "head_token_hidden_dim": 3,
                "head_channel_hidden_dim": 8,
            },
        )
        pooler = self.pooler_cls(head)
        layouts = [
            pack_valen_layout(n, 3, [ValenQuestion((3, 5), ((5, 8),), n - 1)])
            for n in (12, 15)
        ]
        params = [
            PoolingParams(task="plugin", extra_kwargs={"valen_layout": layout})
            for layout in layouts
        ]
        metadata = PoolingMetadata(
            prompt_lens=torch.tensor([12, 15]),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=params,
            pooling_states=[PoolingStates(), PoolingStates()],
        )
        metadata.build_pooling_cursor(
            np.array([12, 15]), torch.tensor([12, 15]), torch.device("cpu")
        )
        hidden = torch.randn(27, 6, dtype=torch.bfloat16)
        got = pooler(hidden, metadata)
        for result, hidden_row, layout in zip(got, [hidden[:12], hidden[12:]], layouts):
            torch.testing.assert_close(result, head(pool_features(hidden_row, layout)))
        metadata.build_pooling_cursor(
            np.array([11, 15]), torch.tensor([12, 15]), torch.device("cpu")
        )
        with self.assertRaisesRegex(ValueError, "complete prefill"):
            pooler(hidden, metadata)

    def test_weight_mapping(self):
        from valen_vllm.head import ValenMixerHead

        head = ValenMixerHead(
            4,
            {
                "head_width": 4,
                "head_layers": 1,
                "head_token_hidden_dim": 3,
                "head_channel_hidden_dim": 8,
            },
        )
        owner = SimpleNamespace(pooler=self.pooler_cls(head))
        head_weights = [
            ("head." + n, torch.randn_like(p)) for n, p in head.named_parameters()
        ]
        received = []

        def load_backbone(self, weights):
            received.extend(list(weights))
            return {"visual.fake"}

        with patch(
            "vllm.model_executor.models.qwen3_5.Qwen3_5ForConditionalGeneration.load_weights",
            load_backbone,
        ):
            loaded = self.model_cls.load_weights(
                owner, iter(head_weights + [("backbone.visual.fake", torch.ones(1))])
            )
            self.assertEqual(received[0][0], "model.visual.fake")
            self.assertIn("pooler.head.roles", loaded)
            for n, value in head_weights:
                torch.testing.assert_close(dict(head.named_parameters())[n[5:]], value)

    def test_plugin_route_replacement_and_native_handler(self):
        import asyncio
        import math

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from vllm.plugins.endpoint_plugins.interface import attach_endpoint_plugins

        from test_endpoints import decisions_payload, systemone_payload

        app = FastAPI()

        @app.post("/v1/decisions")
        async def stock_decisions():
            return {"wrong_handler": True}

        @app.get("/v1/models")
        async def models():
            return {"object": "list", "data": [{"id": "valen"}]}

        app.openapi()  # Plugin replacement must also invalidate cached OpenAPI.
        attach_endpoint_plugins(app, ("plugin",))
        routes = [
            r
            for r in app.routes
            if getattr(r, "path", None) == "/v1/decisions"
            and "POST" in getattr(r, "methods", ())
        ]
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].endpoint.__name__, "decisions")
        self.assertNotIn(
            "stock_decisions",
            app.openapi()["paths"]["/v1/decisions"]["post"]["operationId"],
        )
        self.assertEqual(len(app.state.endpoint_plugins), 1)

        class Engine:
            model_config = SimpleNamespace(
                hf_config=SimpleNamespace(
                    model_type="valen_qwen", valen_max_length=8192
                ),
                tokenizer=os.environ["VALEN_TEST_MODEL"],
                tokenizer_revision=None,
                max_model_len=8192,
            )

            async def encode(self, prompt, params, request_id):
                yield SimpleNamespace(
                    outputs=SimpleNamespace(data=torch.tensor([2, 0, 1, 0, 0, 1, 2])),
                    prompt_token_ids=prompt["prompt_token_ids"],
                )

        asyncio.run(
            app.state.endpoint_plugins[0].init_state(
                Engine(), app.state, SimpleNamespace(served_model_name=["valen"])
            )
        )
        client = TestClient(app)
        for temperature in (0.5, 1.0, 2.0):
            old = client.post(
                "/v1/systemone",
                json={**systemone_payload(), "temperature": temperature},
            )
            new = client.post(
                "/v1/decisions",
                json={**decisions_payload(), "temperature": temperature},
            )
            self.assertEqual(old.status_code, 200, old.text)
            self.assertEqual(new.status_code, 200, new.text)
            expected = 1 / (1 + math.exp(-1 / temperature))
            self.assertAlmostEqual(old.json()["answers"]["on_sofa"]["noul"], expected)
            self.assertEqual(
                new.json()["answers"][1]["probability"],
                old.json()["answers"]["on_sofa"]["noul"],
            )
            self.assertEqual(new.json()["usage"], old.json()["usage"])
        self.assertEqual(client.get("/v1/models").json()["data"][0]["id"], "valen")
        for route, payload in [
            ("systemone", systemone_payload()),
            ("decisions", decisions_payload()),
        ]:
            self.assertEqual(
                client.post(
                    "/v1/" + route, json={**payload, "model": "missing"}
                ).status_code,
                404,
            )
