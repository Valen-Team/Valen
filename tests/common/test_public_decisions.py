"""Public contract, deployment-copy drift and HTTP adapter regression checks."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from valen.decisions import to_internal, to_public, local_images

ROOT = Path(__file__).resolve().parents[2]
BODY = {"model": "valen", "input": "A cat.", "questions": [
    {"name": "present", "type": "predicate", "instructions": "A cat is present."},
    {"name": "animal", "type": "choice", "instructions": "Is it a cat?", "choices": [
        {"value": True, "description": "A cat"}, {"value": False, "description": "A dog"}]},
    {"name": "size", "type": "score", "instructions": "How large?", "levels": [
        {"label": "small", "description": "Small"}, {"label": "large", "description": "Large"}]},
]}
OLD = {"model": "valen", "usage": {"input_tokens": 42, "output_tokens": 0}, "answers": {
    "present": {"type": "noul", "noul": .8},
    "animal": {"type": "choice", "choice": "0", "probabilities": {"0": .75, "1": .25}, "confidence": .5},
    "size": {"type": "score", "score": .25, "probabilities": {"0": .75, "1": .25}, "confidence": .5},
}}


class TestDecisions(unittest.TestCase):
    def test_conversion_preserves_values_labels_order_and_usage(self):
        source = copy.deepcopy(BODY)
        internal = to_internal(source)
        self.assertEqual(internal["questions"]["animal"]["criteria"], {"true": "A cat", "false": "A dog"})
        response = to_public(source, OLD)
        self.assertEqual([a["name"] for a in response["answers"]], ["present", "animal", "size"])
        self.assertIs(response["answers"][1]["choice"], True)
        self.assertEqual(response["answers"][2]["probabilities"][1], {"label": "large", "value": 1, "probability": .25})
        self.assertEqual(response["usage"], OLD["usage"])
        self.assertEqual(source, BODY)

    def test_string_values_keep_the_trained_prompt_keys(self):
        source = copy.deepcopy(BODY)
        source["questions"][1]["choices"] = [
            {"value": "cat", "description": "A cat"},
            {"value": "dog", "description": "A dog"},
        ]
        self.assertEqual(to_internal(source)["questions"]["animal"]["criteria"], {"cat": "A cat", "dog": "A dog"})
        source["questions"][1]["choices"] = [
            {"value": True, "description": "Boolean true"},
            {"value": "true", "description": "String true"},
        ]
        criteria = to_internal(source)["questions"]["animal"]["criteria"]
        self.assertEqual(len(criteria), 2)
        self.assertEqual(criteria["true"], "String true")
        self.assertEqual(list(criteria.values()), ["Boolean true", "String true"])

    def test_validation(self):
        for temperature in [True, 0, -1, float("inf"), float("nan"), "1"]:
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                to_internal(dict(BODY, temperature=temperature))
        for questions in [[], BODY["questions"] * 2, [{"type": "unknown", "instructions": "x"}]]:
            with self.subTest(questions=questions), self.assertRaises(ValueError):
                to_internal(dict(BODY, questions=questions))

    def test_images_are_materialized_and_cleaned(self):
        body = dict(BODY, input=[{"role": "user", "content": [
            {"type": "input_text", "text": "Inspect"},
            {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(b"image bytes").decode()},
        ]}])
        internal = to_internal(body)
        with local_images(internal) as prepared:
            path = Path(prepared["state"]["messages"][0]["content"][1]["image_url"]["url"])
            self.assertEqual(path.read_bytes(), b"image bytes")
        self.assertFalse(path.exists())
        self.assertTrue(internal["state"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:"))

    def test_deployment_copies_match(self):
        source = (ROOT / "valen/decisions.py").read_bytes()
        for engine in ("sglang", "vllm"):
            self.assertEqual(source, (ROOT / f"serving/{engine}/src/valen_{engine}/decisions.py").read_bytes())

    def test_vllm_http_adapter_and_calibration(self):
        from valen_vllm.endpoint import ValenEndpointPlugin
        from valen_vllm.protocol import SystemOneRequest
        from valen_vllm.answers import answers
        app = FastAPI()
        ValenEndpointPlugin().attach_router(app)
        async def handle(request, raw):
            self.assertEqual(request.questions["present"].type, "noul")
            return dict(OLD, answers=answers(request, [2., 0., 2., 0., 2., 0.], request.temperature))
        app.state.valen_handler = handle
        client = TestClient(app)
        cold = client.post("/v1/decisions", json=BODY)
        warm = client.post("/v1/decisions", json=dict(BODY, temperature=2))
        self.assertEqual(cold.status_code, 200)
        self.assertGreater(cold.json()["answers"][0]["probability"], warm.json()["answers"][0]["probability"])
        self.assertIs(warm.json()["answers"][1]["choice"], True)
        self.assertEqual(client.post("/v1/decisions", json=dict(BODY, temperature=0)).status_code, 400)
        # Legacy clients still get the mapping they supplied.
        legacy = client.post("/v1/systemone", json=to_internal(BODY))
        self.assertEqual(legacy.status_code, 200)
        self.assertIsInstance(legacy.json()["answers"], dict)
        legacy_warm = client.post("/v1/systemone", json=dict(to_internal(BODY), temperature=2))
        self.assertEqual(legacy_warm.status_code, 200)
        self.assertEqual(legacy_warm.json()["answers"]["present"]["noul"], warm.json()["answers"][0]["probability"])

    def test_exported_predict_accepts_both_contracts(self):
        import ast
        import torch
        from valen.evaluation.inference import answer
        path = ROOT / "valen/modeling/qwen/export_templates/modeling_valen.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "predict")
        method.decorator_list = []
        scope = dict(to_internal=to_internal, to_public=to_public, local_images=local_images, answer=answer)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
        class Compiler:
            def compile(self, record):
                questions = []
                for name, q in record["request"]["questions"].items():
                    keys = ["true", "false"] if q["type"] == "noul" else list(q["criteria"]) if q["type"] == "choice" else ["0", "1"]
                    questions.append(SimpleNamespace(qid=name, kind=q["type"], keys=keys, descriptions=["Small", "Large"]))
                return SimpleNamespace(questions=questions, inputs=True, logical_tokens=42, compute_tokens=50)
        model = SimpleNamespace(eval=lambda: None, make_compiler=lambda *args: Compiler(),
                                model_name="valen", forward_state=lambda state: [torch.tensor([2., 0.])] * len(state.questions))
        public = scope["predict"](model, BODY)
        self.assertIs(public["answers"][1]["choice"], True)
        tiny = scope["predict"](model, dict(BODY, temperature=1e-300))
        self.assertEqual(tiny["answers"][0]["probability"], 1.0)
        legacy = scope["predict"](model, to_internal(BODY))
        self.assertEqual(public, to_public(BODY, legacy))

    def test_example_clients_select_protocol_for_text_and_images(self):
        import contextlib
        import io
        import runpy
        import tempfile
        for engine in ("sglang", "vllm"):
            script = ROOT / f"serving/{engine}/example_client.py"
            for protocol in ("systemone", "decisions"):
                for image in (False, True):
                    with self.subTest(engine=engine, protocol=protocol, image=image), tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "cat.png"
                        path.write_bytes(b"image bytes")
                        argv = [str(script), "--protocol", protocol]
                        if image:
                            argv += ["--image", str(path)]
                        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
                        with patch.object(sys, "argv", argv), patch("requests.post", return_value=response) as post, contextlib.redirect_stdout(io.StringIO()):
                            runpy.run_path(str(script), run_name="__main__")
                        self.assertTrue(post.call_args.args[0].endswith("/v1/" + protocol))
                        payload = post.call_args.kwargs["json"]
                        if protocol == "decisions":
                            payload = to_internal(payload)
                        self.assertEqual([q["type"] for q in payload["questions"].values()], ["choice", "noul", "score"])
                        self.assertEqual(list(payload["questions"]["animal"]["criteria"].values()), ["A cat", "A dog"])
                        if image:
                            part = payload["state"]["messages"][0]["content"][0]
                            self.assertEqual(part["type"], "image_url")
                            self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))
                        else:
                            self.assertIsInstance(payload["state"], str)

    def test_sglang_http_adapter(self):
        class InternalRequest(BaseModel):
            model_config = ConfigDict(extra="allow")
            model: str
            state: str
            questions: dict
            temperature: float = 1
        app = FastAPI()
        async def handle(request, raw):
            self.assertEqual(request.temperature, 2)
            return Response(json.dumps(OLD), media_type="application/json")
        app.state.systemone_serving = SimpleNamespace(valen_head_config={}, handle_request=handle)
        protocol = ModuleType("valen_sglang.protocol")
        protocol.SystemOneRequest = InternalRequest
        http = ModuleType("sglang.srt.entrypoints.http_server")
        http.app = app
        path = ROOT / "serving/sglang/src/valen_sglang/routes.py"
        spec = importlib.util.spec_from_file_location("test_sglang_routes", path)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(ROOT / "serving/sglang/src"))
        try:
            with patch.dict(sys.modules, {"valen_sglang.protocol": protocol, "sglang.srt.entrypoints.http_server": http}):
                spec.loader.exec_module(module)
                module.register_routes()
                module.register_routes()
                client = TestClient(app)
                result = client.post("/v1/decisions", json=dict(BODY, temperature=2))
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json(), to_public(BODY, OLD))
                self.assertEqual(client.post("/v1/decisions", json=dict(BODY, temperature=0)).status_code, 400)
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
