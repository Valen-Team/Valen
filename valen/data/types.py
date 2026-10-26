"""Model-independent decision metadata; encoded tensors stay with each compiler."""
from dataclasses import dataclass
from typing import TypedDict


@dataclass
class QuestionSpec:
    qid: str
    kind: str
    keys: list
    descriptions: list
    target: list | None
    is_text: bool


class SpatialTarget(TypedDict, total=False):
    """Original-image supervision; masks use PNG references, RLE or polygons."""
    boxes: list[list[float]]
    labels: list[int]
    mask: dict | str
    ignore_mask: dict | str
