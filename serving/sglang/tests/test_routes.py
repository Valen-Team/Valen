"""Routing regression checks against the FastAPI/SGLang schemas."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient
from sglang.srt.plugins import load_plugins

load_plugins()

from valen_sglang.routes import register_routes  # noqa: E402


BODY = {
    "model": "valen",
    "input": "A cat is present.",
    "questions": [
        {"name": "present", "type": "predicate", "instructions": "A cat is present."}
    ],
}


class TestDecisionsRoutes(unittest.TestCase):
    def make_app(self):
        app = FastAPI()
        stock = AsyncMock(return_value={"wrong": "stock token scoring"})

        async def require_admission(request: Request):
            if request.headers.get("X-Admission") != "allowed":
                raise HTTPException(401, "Admission required")

        @app.post("/v1/decisions", dependencies=[Depends(require_admission)])
        async def stock_decisions(body: dict):
            return await stock(body)

        @app.get("/health")
        async def health():
            return {"ok": True}

        self.stock = stock
        self.handle = AsyncMock(
            return_value=ORJSONResponse(
                {
                    "model": "valen",
                    "answers": {"present": {"type": "noul", "noul": 0.75}},
                    "usage": {"input_tokens": 12, "output_tokens": 0},
                }
            )
        )
        app.state.systemone_serving = SimpleNamespace(
            valen_head_config={"head_type": "mixer"}, handle_request=self.handle
        )
        return app

    def test_replaces_existing_stock_route_and_preserves_dependencies(self):
        app = self.make_app()
        # A generated old schema must not survive route replacement.
        app.openapi()
        index = next(i for i, r in enumerate(app.routes) if r.path == "/v1/decisions")
        register_routes(app)
        self.assertIsNone(app.openapi_schema)
        register_routes(app)
        self.assertEqual(sum(r.path == "/v1/decisions" for r in app.routes), 1)
        self.assertEqual(app.routes[index].name, "valen_decisions")
        client = TestClient(app)
        self.assertEqual(client.get("/health").status_code, 200)
        self.assertEqual(client.post("/v1/decisions", json=BODY).status_code, 401)
        self.handle.assert_not_awaited()
        response = client.post(
            "/v1/decisions", json=BODY, headers={"X-Admission": "allowed"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["answers"][0]["probability"], 0.75)
        self.assertEqual(response.json()["usage"]["input_tokens"], 12)
        self.assertEqual(
            self.handle.await_args.args[0].questions["present"].type, "noul"
        )
        self.stock.assert_not_awaited()

    def test_upstream_errors_are_not_converted_to_success(self):
        app = self.make_app()
        register_routes(app)
        self.handle.return_value = ORJSONResponse(
            {"error": {"message": "Unknown model"}}, status_code=404
        )
        response = TestClient(app).post(
            "/v1/decisions", json=BODY, headers={"X-Admission": "allowed"}
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["message"], "Unknown model")

    def test_invalid_public_requests_do_not_reach_scheduler(self):
        app = self.make_app()
        register_routes(app)
        client = TestClient(app)
        for temperature in (0, -1, True, "warm"):
            with self.subTest(temperature=temperature):
                response = client.post(
                    "/v1/decisions",
                    json=dict(BODY, temperature=temperature),
                    headers={"X-Admission": "allowed"},
                )
                self.assertEqual(response.status_code, 400, response.text)
        self.handle.assert_not_awaited()

    def test_stock_route_is_unchanged_until_valen_explicitly_installs_adapter(self):
        app = self.make_app()
        response = TestClient(app).post(
            "/v1/decisions", json=BODY, headers={"X-Admission": "allowed"}
        )
        self.assertEqual(response.json(), {"wrong": "stock token scoring"})
        self.stock.assert_awaited_once()
        self.handle.assert_not_awaited()
