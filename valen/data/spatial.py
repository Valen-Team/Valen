"""Image-conditioned spatial records and exact Qwen readout positions."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from PIL import Image

from .types import SpatialTarget


KINDS = {"det", "seg", "gui"}


def validate_spatial_record(record, *, allow_empty_instruction=False):
    request = record["request"]
    if not record.get("group_id") or len(request["questions"]) != 1:
        raise ValueError("Spatial records require group_id and exactly one question")
    qid, question = next(iter(request["questions"].items()))
    if question.get("type") not in KINDS:
        raise ValueError("Expected det, seg or gui")
    if not isinstance(question.get("instructions"), str) or (not question["instructions"].strip() and not allow_empty_instruction):
        raise ValueError("A nonempty spatial instruction is required")
    categories = question.get("categories", ["target"])
    if not categories or any(not isinstance(x, str) or not x.strip() for x in categories):
        raise ValueError("categories must be a nonempty list of names")
    if len(set(categories)) != len(categories):
        raise ValueError("Duplicate categories")
    if set(record.get("targets", {})) - {qid}:
        raise ValueError("Unknown spatial target question")
    target = record.get("targets", {}).get(qid)
    if target is not None:
        boxes, labels = target.get("boxes", []), target.get("labels", [])
        if len(boxes) != len(labels):
            raise ValueError("Each box requires one category index")
        for box, label in zip(boxes, labels):
            if len(box) != 4 or any(isinstance(x, bool) or not isinstance(x, (float, int))
                                    or not math.isfinite(x) for x in box):
                raise ValueError("Expected finite normalized xyxy boxes")
            if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                raise ValueError("Invalid normalized xyxy box")
            if type(label) is not int or not 0 <= label < len(categories):
                raise ValueError("Invalid category index")
        if question["type"] == "seg" and target.get("mask") is None:
            raise ValueError("Segmentation supervision requires a genuine mask")
    image_path(record)
    return record


def image_path(record):
    images = [item["image_url"]["url"]
              for message in record["request"]["state"]["messages"]
              for item in message["content"] if item.get("type") == "image_url"]
    if len(images) != 1 or "://" in images[0]:
        raise ValueError("Spatial tasks require exactly one materialized image")
    return images[0]


def read_spatial_jsonl(path, labeled_only=False, *, allow_empty_instruction=False):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = validate_spatial_record(json.loads(line), allow_empty_instruction=allow_empty_instruction)
                if not labeled_only or record.get("targets"):
                    records.append(record)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not records:
        raise ValueError("Empty spatial dataset")
    return records


def decode_mask(value, root="."):
    """Decode original-resolution PNG, COCO RLE or polygon supervision."""
    import numpy as np
    if isinstance(value, str):
        with Image.open(Path(root) / value) as image:
            return torch.from_numpy((np.asarray(image.convert("L")) > 0).copy()).float()
    if value.get("rasterization") == "phrasecut":
        # The original benchmark truncates vertices and includes polygon boundaries.
        from PIL import ImageDraw
        height, width = value["size"]
        canvas = Image.new("L", (width, height))
        draw = ImageDraw.Draw(canvas)
        for polygon in value["polygons"]:
            vertices = list(zip(map(int, polygon[::2]), map(int, polygon[1::2])))
            if len(vertices) >= 2:
                draw.polygon(vertices, fill=1, outline=1)
        return torch.from_numpy(np.asarray(canvas).copy()).float()
    from pycocotools import mask as mask_api
    if "polygons" in value:
        h, w = value["size"]
        rles = mask_api.frPyObjects(value["polygons"], h, w)
        value = mask_api.merge(rles)
    elif isinstance(value.get("counts"), list):
        value = mask_api.frPyObjects(value, *value["size"])
    elif isinstance(value.get("counts"), str):
        value = dict(value, counts=value["counts"].encode("ascii"))
    array = mask_api.decode(value)
    if array.ndim == 3:
        array = array.max(-1)
    return torch.from_numpy(array.copy()).float()


def decode_segmentation_target(target, root="."):
    """Return the binary target and valid pixels; ignored regions never become foreground."""
    truth = decode_mask(target["mask"], root)
    valid = torch.ones_like(truth, dtype=torch.bool)
    if target.get("ignore_mask") is not None:
        ignored = decode_mask(target["ignore_mask"], root).bool()
        if ignored.shape != truth.shape:
            raise ValueError("Ignored regions and segmentation mask dimensions differ")
        valid &= ~ignored
    return truth, valid


@dataclass
class SpatialQuestion:
    qid: str
    kind: str
    inputs: dict
    categories: list[str]
    category_spans: list[tuple[int, int]]
    decision_position: int
    grid: tuple[int, int]
    original_size: tuple[int, int]
    target: SpatialTarget | None
    media_root: Path
    record_id: str
    metadata: dict


@dataclass
class SpatialState:
    questions: list[SpatialQuestion]
    logical_tokens: int
    compute_tokens: int


class SpatialCompiler:
    def __init__(self, processor, media_root=".", max_length=4096, max_pixels=262144, allow_empty_instruction=False):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.media_root = Path(media_root)
        self.max_length = max_length
        self.max_pixels = max_pixels
        self.allow_empty_instruction = allow_empty_instruction

    def compile(self, record, rng=None, labeled_only=False) -> SpatialState:
        validate_spatial_record(record, allow_empty_instruction=self.allow_empty_instruction)
        qid, question = next(iter(record["request"]["questions"].items()))
        target = record.get("targets", {}).get(qid)
        if labeled_only and target is None:
            return SpatialState([], 0, 0)
        categories = question.get("categories", ["target"])
        for text in [question["instructions"], *categories]:
            if any(token in text for token in self.tokenizer.all_special_tokens):
                raise ValueError("Reserved tokenizer control token in spatial input")
        with Image.open(self.media_root / image_path(record)) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = dict(self.processor(text=[text], images=[image], return_tensors="pt",
                                     max_pixels=self.max_pixels, min_pixels=4096))
        suffix = []
        base_length = inputs["input_ids"].shape[1]

        def append(text):
            start = base_length + len(suffix)
            suffix.extend(self.tokenizer.encode(text, add_special_tokens=False))
            return start, base_length + len(suffix)

        append("<|im_start|>user\nTask: " + question["type"] + "\nInstruction: "
               + question["instructions"] + "\nCategories:\n")
        spans = []
        for name in categories:
            spans.append(append(name))
            append("\n")
        append("Decision:")
        inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([suffix])], -1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.cat([
                inputs["mm_token_type_ids"], torch.zeros((1, len(suffix)), dtype=torch.long)], -1)
        length = inputs["input_ids"].shape[1]
        if length > self.max_length:
            raise ValueError(f"Spatial input has {length} tokens, exceeding {self.max_length}")
        t, h, w = inputs["image_grid_thw"][0].tolist()
        merge = self.processor.image_processor.merge_size
        if t != 1:
            raise ValueError("Expected a single static image grid")
        compiled = SpatialQuestion(qid, question["type"], inputs, categories, spans, length - 1,
                                   (h // merge, w // merge), (image.height, image.width),
                                   target, self.media_root, str(record.get("id", record["group_id"])),
                                   record.get("meta", {}))
        return SpatialState([compiled], length, length)
