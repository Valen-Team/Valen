"""System One's standard fields with Valen's native score-level budget."""

from typing import Annotated, Union

from pydantic import Field, ValidationInfo, field_validator

from sglang.srt.entrypoints.openai.protocol import RequiredDecisionText
from sglang.srt.entrypoints.systemone.protocol import (
    SystemOneChoiceQuestion,
    SystemOneNoulQuestion,
)
from sglang.srt.entrypoints.systemone.protocol import (
    SystemOneRequest as StockSystemOneRequest,
)
from sglang.srt.entrypoints.systemone.protocol import (
    SystemOneScoreQuestion as StockSystemOneScoreQuestion,
)


class SystemOneScoreQuestion(StockSystemOneScoreQuestion):
    criteria: list[RequiredDecisionText] = Field(min_length=1, max_length=255)


Question = Annotated[
    Union[SystemOneNoulQuestion, SystemOneChoiceQuestion, SystemOneScoreQuestion],
    Field(discriminator="type"),
]


class SystemOneRequest(StockSystemOneRequest):
    questions: dict[str, Question] = Field(min_length=1)
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False, strict=True)

    @field_validator("temperature", "prompt_format_version", "return_prompt_token_ids")
    @classmethod
    def _refuse_decisions_fields(cls, value, info: ValidationInfo):
        # Stock SystemOne refuses Decisions-only fields with this validator.
        # Override the inherited validator for Valen's calibration extension,
        # while continuing to reject the other unsupported extensions.
        if info.field_name == "temperature":
            return value
        return super()._refuse_decisions_fields(value, info)
