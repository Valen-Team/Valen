"""Valen adapter for SGLang's existing System One route and embedding scheduler.

Non-Valen checkpoints retain the stock serving implementation. Response formatting
uses SGLang's System One answer helpers (Apache-2.0; see NOTICE).
"""

import asyncio
import math

from fastapi import HTTPException
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.systemone.protocol import SystemOneResponse, SystemOneUsage
from sglang.srt.entrypoints.systemone.serving import (
    SystemOneServing as StockSystemOneServing,
)
from sglang.srt.entrypoints.systemone.serving import (
    _answer,
    _check_legend,
    _read_image,
    _view,
)
from sglang.srt.layers.joint_schema_head import max_joint_prompt_tokens
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.srt.runtime_context import get_schedule
from valen_sglang.compiler import encode_valen, valen_messages, valen_options
from valen_sglang.grounding import box_answer, image_grids


class ValenSystemOneServing(StockSystemOneServing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valen_head_config = getattr(
            self.tokenizer_manager.model_config, "valen_head_config", None
        )

        if self.valen_head_config is not None:
            from valen_sglang.routes import register_routes

            register_routes()

    def _validate_request(self, request):
        if self.valen_head_config is not None:
            if request.model != self.tokenizer_manager.served_model_name:
                raise HTTPException(404, "Unknown served model")
            return self._validate_joint_schema_request(request)
        if "temperature" in request.model_fields_set:
            return "Temperature calibration requires a Valen checkpoint"
        # The shared HTTP schema admits native heads' larger score lists, but
        # generative models still have only their original ten token labels.
        if self.joint_head_config is None and any(
            q.type == "score" and len(q.criteria) > 10
            for q in request.questions.values()
        ):
            return "Generative decision models support at most 10 score levels"
        return super()._validate_request(request)

    def _convert_to_internal_request(self, request, raw_request=None):
        if self.valen_head_config is None:
            return super()._convert_to_internal_request(request, raw_request)
        config = self.tokenizer_manager.model_config.hf_config
        views = []
        for question_id, question in request.questions.items():
            if question.type == "bbox":
                if not config.valen_grounding:
                    raise ValueError("bbox questions require a grounding head")
                views.append(None)
                continue
            views.append(_view(question))
            if views[-1].kind == "score":
                _check_legend(question_id, views[-1])
        _, images = valen_messages(request, self.tokenizer_manager.tokenizer)
        if images and self.tokenizer_manager.mm_processor is None:
            raise ValueError("Valen images require multimodal input")
        return EmbeddingReqInput(image_data=images or None), (request, views)

    async def _encode_valen_prompt(self, adapted_request, request):
        manager = self.tokenizer_manager
        config = manager.model_config.hf_config
        image_token_counts, grids = [], []
        if adapted_request.image_data:
            # Each image read once; its header gives the size the processor resizes.
            read = await asyncio.gather(
                *(
                    asyncio.to_thread(_read_image, index, image)
                    for index, image in enumerate(adapted_request.image_data)
                )
            )
            adapted_request.image_data = [image for image, _ in read]
            headers = [header for _, header in read]
            image_token_counts = manager.mm_processor.resolve_image_token_counts(
                headers
            )
            if any(q.type == "bbox" for q in request.questions.values()):
                grids = image_grids(
                    manager.mm_processor._processor.image_processor,
                    [(header.height, header.width) for header in headers],
                )
        messages, _ = valen_messages(request, manager.tokenizer)
        adapted_request.input_ids, adapted_request.decision_layout = encode_valen(
            manager.tokenizer,
            request,
            messages,
            max_length=min(
                manager.model_config.hf_config.valen_max_length,
                max_joint_prompt_tokens(
                    context_len=manager.context_len,
                    num_reserved_tokens=manager.num_reserved_tokens,
                    max_req_input_len=manager.max_req_input_len,
                    max_prefill_tokens=get_schedule().max_prefill_tokens,
                ),
            ),
            image_token_counts=image_token_counts,
            fold=config.valen_decision_version == "1.5",
            mask=config.valen_full_attention_mask,
            image_grids=grids,
        )

    async def _handle_non_streaming_request(
        self, adapted_request, processed, raw_request
    ):
        if self.valen_head_config is None:
            return await super()._handle_non_streaming_request(
                adapted_request, processed, raw_request
            )
        request, views = processed
        await self._encode_valen_prompt(adapted_request, request)
        ret = await self.tokenizer_manager.generate_request(
            adapted_request, raw_request
        ).__anext__()
        logits = ret["embedding"]
        # A bbox question returns x1, y1, x2, y2 and confidence instead of logits.
        names = [
            [None] * 5 if q.type == "bbox" else [name for name, _ in valen_options(q)]
            for q in request.questions.values()
        ]
        if len(logits) != sum(map(len, names)) or not all(
            math.isfinite(v) for v in logits
        ):
            raise RuntimeError(
                "The Valen decision head did not return finite scores for every candidate"
            )
        answers, boxes, offset = {}, {}, 0
        for question_id, view, question_names in zip(request.questions, views, names):
            scores = logits[offset : offset + len(question_names)]
            offset += len(question_names)
            if view is None:
                boxes[question_id] = box_answer(scores)
                continue
            top = max(scores)
            weights = [
                math.exp((score - top) / request.temperature) for score in scores
            ]
            total = math.fsum(weights)
            by_name = {
                name: weight / total for name, weight in zip(question_names, weights)
            }
            answers[question_id] = _answer(
                view=view,
                probabilities=(
                    [by_name["true"], by_name["false"]]
                    if view.kind == "yes_no"
                    else [by_name[name] for name in view.names]
                ),
                mass=1.0,
                question_id=question_id,
            )
        response = SystemOneResponse(
            model=self.tokenizer_manager.served_model_name,
            answers=answers,
            usage=SystemOneUsage(input_tokens=ret["meta_info"]["prompt_tokens"]),
        )
        content = response.model_dump(
            exclude={"answers": {"__all__": {"x_label_mass"}}}
        )
        if boxes:
            content["answers"] = {
                question_id: boxes.get(question_id) or content["answers"][question_id]
                for question_id in request.questions
            }
        return ORJSONResponse(content=content)
