"""Compile the shared-state prompt used to train Valen decision checkpoints."""

from copy import deepcopy
from dataclasses import dataclass

from valen_vllm.blocks import block_segments
from valen_vllm.head import ValenQuestion, pack_valen_layout


@dataclass(frozen=True)
class ImageData:
    url: str
    detail: str = "auto"


_NOUL = [
    ("true", "True / 是：满足问题中的条件。"),
    ("false", "False / 否：不满足问题中的条件。"),
]


def valen_options(question):
    if question.type == "noul":
        return _NOUL
    if question.type == "bbox":
        # Localization has no candidates; the image tokens are the choices.
        return []
    if question.type == "choice":
        return list(question.criteria.items())
    if not 2 <= len(question.criteria) <= 255:
        raise ValueError("Valen score questions require 2 to 255 levels")
    return [(str(i), description) for i, description in enumerate(question.criteria)]


def valen_instructions(question, fold):
    """Valen-Decision-1.5 reads a noul rubric as part of the instructions."""
    criteria = question.criteria if question.type == "noul" else None
    if criteria is None:
        return question.instructions
    if not fold:
        raise ValueError(
            "Valen noul questions use fixed true/false descriptions; omit criteria"
        )
    return (
        question.instructions
        + "\nTrue criterion: "
        + criteria["true"]
        + "\nFalse criterion: "
        + criteria["false"]
    )


def _text(value, tokenizer, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError("Valen instructions and criteria must be nonempty strings")
    special = set(tokenizer.all_special_tokens)
    special.update(
        str(token) for token in tokenizer.added_tokens_decoder.values() if token.special
    )
    if any(token in value for token in special):
        raise ValueError("Reserved tokenizer control token in Valen input")
    return value


def valen_messages(request, tokenizer):
    """Render native message state and collect images in placeholder order."""
    if isinstance(request.state, str):
        messages = [{"role": "user", "content": request.state}]
    elif isinstance(request.state, dict) and set(request.state) == {"messages"}:
        messages = deepcopy(request.state["messages"])
    else:
        raise ValueError("Valen state must be text or an object containing messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Valen state.messages must be a nonempty list")
    images = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "user",
            "assistant",
            "system",
        }:
            raise ValueError("Unsupported Valen state message role")
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list) or not content:
            raise ValueError("Valen message content must be text or content blocks")
        normalized = []
        for item in content:
            if not isinstance(item, dict):
                raise TypeError("Invalid Valen content block")
            if item.get("type") == "text":
                normalized.append(
                    {
                        "type": "text",
                        "text": _text(item.get("text"), tokenizer, allow_empty=True),
                    }
                )
            elif item.get("type") == "image_url":
                source = item.get("image_url")
                if not isinstance(source, dict) or not isinstance(
                    source.get("url"), str
                ):
                    raise ValueError("Valen image_url requires a url string")
                images.append(
                    ImageData(url=source["url"], detail=source.get("detail", "auto"))
                )
                normalized.append({"type": "image"})
            else:
                raise ValueError(
                    "Valen currently supports text and image_url content blocks"
                )
        message["content"] = normalized
    if request.images:
        if not isinstance(request.state, str):
            raise ValueError(
                "Use image_url blocks for images in a Valen messages state"
            )
        images.extend(
            ImageData(url=i.url, detail=i.detail or "auto") for i in request.images
        )
        messages[0]["content"] = [{"type": "image"} for _ in request.images] + messages[
            0
        ]["content"]
    return messages, images


def encode_valen(
    tokenizer,
    request,
    messages,
    *,
    max_length,
    image_token_counts,
    fold=False,
    mask="causal",
    image_grids=(),
):
    """``fold``, ``mask`` and ``image_grids`` carry the Valen-Decision-1.5 contract:
    noul rubrics in the instructions, block ids for the full-attention mask, and
    the merged (rows, cols) grid of each image for bbox questions."""
    # Tokenizing only the template leaves one placeholder per image. The native
    # multimodal processor expands them later, and the layout shifts accordingly.
    ids = list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=False
        )
    )
    context_end = len(ids)
    # Block ids of the suffix tokens: glue 0, question i as i + 1.
    segments = []

    def append(text, span=None, segment=0):
        start = len(ids)
        if span is None:
            ids.extend(tokenizer.encode(text, add_special_tokens=False))
            segments.extend([segment] * (len(ids) - start))
            return None
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids.extend(encoded["input_ids"])
        segments.extend([segment] * (len(ids) - start))
        indices = [
            i
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            if a < span[1] and b > span[0]
        ]
        if not indices:
            raise ValueError("Empty token span for Valen decision role")
        return start + indices[0], start + indices[-1] + 1

    append("<|im_start|>user\n")
    questions = []
    for i, question in enumerate(request.questions.values()):
        instruction = _text(valen_instructions(question, fold), tokenizer)
        prefix = f"Question {i + 1}\nTask: {question.type}\nQuestion: "
        if question.type == "bbox":
            if question.image >= len(image_grids):
                raise ValueError(
                    f"bbox refers to image {question.image}, state has {len(image_grids)}"
                )
            rows, cols = image_grids[question.image]
            if rows * cols != image_token_counts[question.image]:
                raise ValueError("Image tokens do not match their grid")
            # The image's first token once the engine expands the placeholders.
            image_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            placeholders = [p for p in range(context_end) if ids[p] == image_token]
            image_start = placeholders[question.image] + sum(
                count - 1 for count in image_token_counts[: question.image]
            )
            instruction_span = append(
                prefix + instruction + f"\nImage: {question.image + 1}\n",
                (len(prefix), len(prefix) + len(instruction)),
                i + 1,
            )
            questions.append((instruction_span, (), (image_start, rows, cols)))
            continue
        instruction_span = append(
            prefix + instruction + "\nCandidates:\n",
            (len(prefix), len(prefix) + len(instruction)),
            i + 1,
        )
        spans = []
        for key, description in valen_options(question):
            _text(key, tokenizer)
            _text(description, tokenizer)
            prefix = key + ": " if question.type != "score" else ""
            spans.append(
                append(
                    prefix + description,
                    (len(prefix), len(prefix) + len(description)),
                    i + 1,
                )
            )
            append("\n", segment=i + 1)
        questions.append((instruction_span, tuple(spans), None))
    append("<|im_end|>\n<|im_start|>assistant\n")
    layout_questions = []
    for i, (instruction, candidates, image) in enumerate(questions):
        append(f"Question {i + 1} Decision:", segment=i + 1)
        layout_questions.append(
            ValenQuestion(instruction, candidates, len(ids) - 1, image)
        )
        append("\n", segment=i + 1)
    length = len(ids) + sum(count - 1 for count in image_token_counts)
    if length > max_length:
        raise ValueError(
            f"Valen prompt has {length} tokens, exceeding max_length={max_length}; no truncation"
        )
    bbox = {i + 1 for i, q in enumerate(request.questions.values()) if q.type == "bbox"}
    return ids, pack_valen_layout(
        len(ids),
        context_end,
        layout_questions,
        None if mask == "causal" else block_segments(mask, segments, bbox),
    )
