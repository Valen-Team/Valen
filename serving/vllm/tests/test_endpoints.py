"""HTTP contracts for both protocols, including calibration and typed answers."""

import math
import unittest

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from valen_vllm.answers import answers
from valen_vllm.endpoint import ValenEndpointPlugin
from valen_vllm.protocol import SystemOneRequest


def systemone_payload():
    return {
        "model": "valen",
        "state": "A cat is on the sofa.",
        "questions": {
            "animal": {
                "type": "choice",
                "instructions": "Which animal is present?",
                "criteria": {"cat": "A cat", "dog": "A dog"},
            },
            "on_sofa": {"type": "noul", "instructions": "The cat is on the sofa."},
            "clarity": {
                "type": "score",
                "instructions": "How clear is the evidence?",
                "criteria": ["Unclear", "Partly clear", "Clear"],
            },
        },
    }


def decisions_payload():
    return {
        "model": "valen",
        "input": "A cat is on the sofa.",
        "questions": [
            {
                "name": "animal",
                "type": "choice",
                "instructions": "Which animal is present?",
                "choices": [
                    {"value": "cat", "description": "A cat"},
                    {"value": "dog", "description": "A dog"},
                ],
            },
            {
                "name": "on_sofa",
                "type": "predicate",
                "instructions": "The cat is on the sofa.",
            },
            {
                "name": "clarity",
                "type": "score",
                "instructions": "How clear is the evidence?",
                "levels": [
                    {"label": str(i), "description": text}
                    for i, text in enumerate(["Unclear", "Partly clear", "Clear"])
                ],
            },
        ],
    }


class TestEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        ValenEndpointPlugin().attach_router(self.app)
        self.seen = []

        async def handle(body, raw_request):
            if body.model != "valen":
                raise HTTPException(404, "Unknown served model")
            self.seen.append(body)
            return {
                "model": body.model,
                "answers": answers(body, [2, 0, 1, 0, 0, 1, 2], body.temperature),
                "usage": {"input_tokens": 42, "output_tokens": 0},
            }

        self.app.state.valen_handler = handle
        self.client = TestClient(self.app)

    def test_protocols_share_questions_temperature_and_probability_semantics(self):
        for temperature in (0.5, 1, 2.0):
            with self.subTest(temperature=temperature):
                native = self.client.post(
                    "/v1/systemone",
                    json={**systemone_payload(), "temperature": temperature},
                )
                public = self.client.post(
                    "/v1/decisions",
                    json={**decisions_payload(), "temperature": temperature},
                )
                self.assertEqual(native.status_code, 200, native.text)
                self.assertEqual(public.status_code, 200, public.text)
                self.assertEqual(self.seen[-2].model_dump(), self.seen[-1].model_dump())
                self.assertEqual(self.seen[-1].temperature, temperature)
                old, new = native.json()["answers"], public.json()["answers"]
                self.assertEqual(
                    [a["name"] for a in new], ["animal", "on_sofa", "clarity"]
                )
                self.assertEqual(new[0]["choice"], old["animal"]["choice"])
                self.assertEqual(
                    [p["probability"] for p in new[0]["probabilities"]],
                    list(old["animal"]["probabilities"].values()),
                )
                self.assertEqual(new[1]["probability"], old["on_sofa"]["noul"])
                self.assertEqual(new[2]["score"], old["clarity"]["score"])
                self.assertEqual(new[2]["confidence"], old["clarity"]["confidence"])
                self.assertAlmostEqual(
                    new[1]["probability"], 1 / (1 + math.exp(-1 / temperature))
                )
                self.assertEqual(public.json()["usage"], native.json()["usage"])

    def test_default_temperature(self):
        for route, payload in [
            ("systemone", systemone_payload()),
            ("decisions", decisions_payload()),
        ]:
            response = self.client.post("/v1/" + route, json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.seen[-1].temperature, 1.0)

    def test_invalid_temperature_is_rejected_before_engine(self):
        for temperature in (0, -1, True, False, "0.5", None):
            for route, payload, status in [
                ("systemone", systemone_payload(), 422),
                ("decisions", decisions_payload(), 400),
            ]:
                with self.subTest(route=route, temperature=temperature):
                    response = self.client.post(
                        "/v1/" + route, json={**payload, "temperature": temperature}
                    )
                    self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(self.seen, [])
        for temperature in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValidationError):
                SystemOneRequest.model_validate(
                    {**systemone_payload(), "temperature": temperature}
                )

    def test_unknown_model_and_engine_errors_keep_http_status(self):
        for route, payload in [
            ("systemone", systemone_payload()),
            ("decisions", decisions_payload()),
        ]:
            self.assertEqual(
                self.client.post(
                    "/v1/" + route, json={**payload, "model": "unknown"}
                ).status_code,
                404,
            )

            async def fail(body, raw_request):
                raise HTTPException(400, "Invalid native prompt")

            original = self.app.state.valen_handler
            self.app.state.valen_handler = fail
            response = self.client.post("/v1/" + route, json=payload)
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json(), {"detail": "Invalid native prompt"})
            self.app.state.valen_handler = original

    def test_not_ready_on_both_routes(self):
        del self.app.state.valen_handler
        for route, payload in [
            ("systemone", systemone_payload()),
            ("decisions", decisions_payload()),
        ]:
            self.assertEqual(
                self.client.post("/v1/" + route, json=payload).status_code, 503
            )

    def test_public_candidate_values_and_image_input(self):
        payload = decisions_payload()
        payload["questions"] = payload["questions"][:1]
        payload["questions"][0]["choices"] = [
            {"value": True, "description": "A cat"},
            {"value": False, "description": "A dog"},
        ]
        payload["input"] = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                    {"type": "input_text", "text": "Inspect this image."},
                ],
            }
        ]

        async def handle(body, raw_request):
            self.seen.append(body)
            return {"answers": answers(body, [2, 0], body.temperature)}

        self.app.state.valen_handler = handle
        response = self.client.post("/v1/decisions", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()["answers"][0]
        self.assertIs(answer["choice"], True)
        self.assertEqual([p["value"] for p in answer["probabilities"]], [True, False])
        content = self.seen[-1].state["messages"][0]["content"]
        self.assertEqual(
            content[0],
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        )
        self.assertEqual(content[1], {"type": "text", "text": "Inspect this image."})
