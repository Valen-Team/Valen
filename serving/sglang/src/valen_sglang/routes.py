"""Serve Decisions through the same native Valen pipeline as SystemOne."""

import json

from fastapi import HTTPException, Request
from fastapi.responses import ORJSONResponse
from fastapi.routing import APIRoute

from valen_sglang.decisions import to_internal, to_public
from valen_sglang.protocol import SystemOneRequest


def register_routes(app=None):
    """Replace stock candidate-token scoring only on a Valen server.

    Called by ValenSystemOneServing during app startup. The stock engine
    already owns /v1/decisions; adding another route would leave its first match
    active. Preserve its dependencies and position when replacing it.
    """
    if app is None:
        from sglang.srt.entrypoints.http_server import app

    matches = [
        (index, route)
        for index, route in enumerate(app.router.routes)
        if isinstance(route, APIRoute)
        and route.path == "/v1/decisions"
        and "POST" in route.methods
    ]
    if len(matches) == 1 and getattr(matches[0][1].endpoint, "_valen_decisions", False):
        return
    dependencies = matches[0][1].dependencies if matches else []

    async def decisions(body: dict, raw_request: Request):
        serving = raw_request.app.state.systemone_serving
        if serving.valen_head_config is None:
            raise HTTPException(400, "Decisions requires a Valen checkpoint")
        try:
            internal = to_internal(body)
            internal["temperature"] = body.get("temperature", 1.0)
            request = SystemOneRequest.model_validate(internal)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        response = await serving.handle_request(request, raw_request)
        if response.status_code != 200:
            return response
        return ORJSONResponse(to_public(body, json.loads(response.body)))

    decisions._valen_decisions = True
    app.add_api_route(
        "/v1/decisions",
        decisions,
        methods=["POST"],
        dependencies=dependencies,
        name="valen_decisions",
        summary="Score typed questions with the native Valen decision head",
    )
    if matches:
        replacement = app.router.routes.pop()
        app.router.routes[matches[0][0]] = replacement
    # The old route schema may already have been requested by a client.
    app.openapi_schema = None
