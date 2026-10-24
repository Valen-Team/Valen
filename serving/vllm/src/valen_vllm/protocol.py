"""Valen's local decision contract, independent of engine-specific schemas."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice", "noul", "score"]
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def check(self):
        if self.type == "noul":
            if self.criteria is not None:
                raise ValueError("noul has fixed true/false criteria")
        elif self.type == "choice":
            if (
                not isinstance(self.criteria, dict)
                or not 1 <= len(self.criteria) <= 255
            ):
                raise ValueError("choice requires 1–255 named criteria")
        elif not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 255:
            raise ValueError("score requires 2–255 ordered criteria")
        return self


class Image(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    detail: str = "auto"


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    state: str | dict
    questions: Annotated[dict[str, Question], Field(min_length=1, max_length=256)]
    images: list[Image] = Field(default_factory=list, max_length=16)
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False, strict=True)
