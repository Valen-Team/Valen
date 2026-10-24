"""Explicitly enabled HTTP plugin, using the ordinary EngineClient encode API."""

import asyncio
import base64
import io
import time
import uuid

from fastapi import HTTPException, Request
from PIL import Image

from valen_vllm.answers import answers
from valen_vllm.compiler import encode_valen, valen_messages
from valen_vllm.decisions import to_internal, to_public
from valen_vllm.protocol import SystemOneRequest


def read_image(source):
    # Inline images avoid server-side URL fetching and filesystem interpretation.
    if not source.url.startswith(
        ("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")
    ):
        raise ValueError("Use a PNG, JPEG or WebP base64 data URL for images")
    if len(source.url) > 24 * 1024 * 1024:
        raise ValueError("Encoded image exceeds 24 MiB")
    raw = base64.b64decode(source.url.split(",", 1)[1], validate=True)
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Image content must be PNG, JPEG or WebP")
            if image.width * image.height > 32_000_000:
                raise ValueError("Image exceeds 32 megapixels")
            image.load()
            return image.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise ValueError("Image exceeds the pixel budget") from exc


class ValenEndpointPlugin:
    name = "valen"
    required_tasks = ("plugin",)

    def attach_router(self, app):
        @app.post("/v1/systemone")
        async def systemone(body: SystemOneRequest, raw_request: Request):
            handler = getattr(raw_request.app.state, "valen_handler", None)
            if handler is None:
                raise HTTPException(503, "Valen engine is not ready")
            return await handler(body, raw_request)

        @app.post("/v1/decisions")
        async def decisions(body: dict, raw_request: Request):
            try:
                internal = SystemOneRequest.model_validate(
                    {**to_internal(body), "temperature": body.get("temperature", 1.0)}
                )
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, str(exc)) from exc
            handler = getattr(raw_request.app.state, "valen_handler", None)
            if handler is None:
                raise HTTPException(503, "Valen engine is not ready")
            response = await handler(internal, raw_request)
            return to_public(body, response)

    async def init_state(self, engine_client, state, args):
        from transformers import AutoTokenizer
        from vllm.exceptions import VLLMClientError
        from vllm.pooling_params import PoolingParams

        config = engine_client.model_config
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer,
            revision=config.tokenizer_revision,
            trust_remote_code=False,
        )
        names = args.served_model_name or [config.model]
        if isinstance(names, str):
            names = [names]
        limit = min(config.max_model_len, config.hf_config.valen_max_length)

        async def handle(body, raw_request):
            if body.model not in names:
                raise HTTPException(404, "Unknown served model")
            try:
                messages, sources = valen_messages(body, tokenizer)
                if len(sources) > 16:
                    raise ValueError("At most 16 images are supported")
                images = await asyncio.to_thread(
                    lambda: [read_image(s) for s in sources]
                )
                ids, layout = encode_valen(
                    tokenizer, body, messages, max_length=limit, image_token_counts=[]
                )
            except (ValueError, TypeError, OSError) as exc:
                raise HTTPException(400, str(exc)) from exc
            prompt = {"prompt_token_ids": ids}
            if images:
                prompt["multi_modal_data"] = {"image": images}
            params = PoolingParams(
                task="plugin",
                skip_reading_prefix_cache=True,
                extra_kwargs={"valen_layout": layout},
            )
            request_id = "valen-" + uuid.uuid4().hex
            final = None
            try:
                async for output in engine_client.encode(prompt, params, request_id):
                    if await raw_request.is_disconnected():
                        raise asyncio.CancelledError()
                    final = output
            except asyncio.CancelledError:
                await engine_client.abort(request_id)
                raise
            except (ValueError, VLLMClientError) as exc:
                raise HTTPException(400, str(exc)) from exc
            if final is None:
                raise RuntimeError("Engine returned no Valen result")
            return {
                "id": request_id,
                "object": "structured_decision",
                "created": int(time.time()),
                "model": body.model,
                "answers": answers(body, final.outputs.data.tolist(), body.temperature),
                "usage": {
                    "input_tokens": len(final.prompt_token_ids),
                    "output_tokens": 0,
                },
            }

        state.valen_handler = handle
