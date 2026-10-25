"""Public Decisions contract. Keep serving copies byte-identical to this module."""
import math
from copy import deepcopy
from contextlib import contextmanager


def to_internal(body, *, allow_video=False):
    """Validate public input and convert it to the training/compiler schema."""
    if not isinstance(body, dict):
        raise ValueError("Request must be an object")
    temperature = body.get("temperature", 1.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    source = body.get("input")
    if isinstance(source, str):
        state = source
    elif isinstance(source, list) and source:
        messages = deepcopy(source)
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"user", "assistant", "system"}:
                raise ValueError("input messages require a supported role")
            content = message.get("content")
            if isinstance(content, str):
                continue
            if not isinstance(content, list) or not content:
                raise ValueError("message content must be text or a nonempty list")
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("content parts must be objects")
                if part.get("type") == "input_text" and isinstance(part.get("text"), str):
                    part["type"] = "text"
                elif part.get("type") == "input_image" and isinstance(part.get("image_url"), str):
                    part["type"] = "image_url"
                    part["image_url"] = {"url": part["image_url"]}
                elif allow_video and part.get("type") == "input_video" and isinstance(part.get("video_url"), str):
                    part["type"] = "video_url"
                    part["video_url"] = {"url": part["video_url"]}
                else:
                    kinds = "input_text, input_image or input_video" if allow_video else "input_text or input_image"
                    raise ValueError(f"Use {kinds} content parts")
        state = {"messages": messages}
    else:
        raise ValueError("input must be text or a nonempty message list")
    public = body.get("questions")
    if not isinstance(public, list) or not 1 <= len(public) <= 256:
        raise ValueError("questions must contain 1–256 questions")
    questions = {}
    for index, q in enumerate(public):
        if not isinstance(q, dict):
            raise ValueError("questions must be objects")
        name = q.get("name", str(index))
        if not isinstance(name, str) or not name or name in questions:
            raise ValueError("question names must be unique nonempty strings")
        kind, instructions = q.get("type"), q.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("instructions must be nonempty text")
        item = {"type": "noul" if kind == "predicate" else kind, "instructions": instructions}
        if kind in {"choice", "score"}:
            entries = q.get("choices" if kind == "choice" else "levels")
            minimum = 1 if kind == "choice" else 2
            if not isinstance(entries, list) or not minimum <= len(entries) <= 255:
                raise ValueError(f"{kind} requires {minimum}–255 candidates")
            labels = []
            descriptions = []
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("candidates must be objects")
                label = entry.get("value" if kind == "choice" else "label")
                if (kind == "choice" and not isinstance(label, (str, bool))) or (kind == "score" and not isinstance(label, str)):
                    raise ValueError("choice values must be strings or booleans; level labels must be strings")
                if label in labels:
                    raise ValueError("candidate values/labels must be unique")
                description = entry.get("description")
                if not isinstance(description, str) or not description.strip():
                    raise ValueError("candidate descriptions must be nonempty text")
                labels.append(label)
                descriptions.append(description)
            if kind == "choice":
                # Candidate keys are part of the trained prompt. Preserve string
                # values so equivalent SystemOne/Decisions requests compile alike.
                # Boolean values need string keys internally, while the public
                # response retains their original type.
                used = {label for label in labels if isinstance(label, str)}
                keys = []
                for label in labels:
                    key = label if isinstance(label, str) else str(label).lower()
                    if isinstance(label, bool):
                        while key in used:
                            key = "bool:" + key
                        used.add(key)
                    keys.append(key)
                item["criteria"] = dict(zip(keys, descriptions))
            else:
                item["criteria"] = descriptions
        elif kind != "predicate":
            raise ValueError("question type must be predicate, choice or score")
        questions[name] = item
    result = {"state": state, "questions": questions}
    if "model" in body:
        result["model"] = body["model"]
    return result


def to_public(body, response):
    """Expose ordered typed answers while retaining available usage metadata."""
    result = dict(response)
    output = []
    for index, q in enumerate(body["questions"]):
        name = q.get("name", str(index))
        old = response["answers"][name]
        entry = {"name": name, "type": q["type"]}
        if q["type"] == "predicate":
            entry["probability"] = old["noul"]
        else:
            key = "choices" if q["type"] == "choice" else "levels"
            candidates = q[key]
            probabilities = list(old["probabilities"].values())
            entry["probabilities"] = [
                dict(value=c["value"], probability=p) if key == "choices" else
                dict(label=c["label"], value=i, probability=p)
                for i, (c, p) in enumerate(zip(candidates, probabilities))
            ]
            entry["confidence"] = old["confidence"]
            if key == "choices":
                entry["choice"] = candidates[max(range(len(probabilities)), key=probabilities.__getitem__)]["value"]
            else:
                entry["score"] = old["score"]
        output.append(entry)
    result["answers"] = output
    return result


@contextmanager
def local_images(request):
    """Materialize inline public images for the local compiler's path interface."""
    import base64
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="valen-images-") as directory:
        request = deepcopy(request)
        state = request["state"]
        if isinstance(state, dict):
            for message in state["messages"]:
                if not isinstance(message["content"], list):
                    continue
                for part in message["content"]:
                    if part["type"] != "image_url":
                        continue
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        if header not in {"data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"}:
                            raise ValueError("Use a PNG, JPEG or WebP base64 data URL")
                        if len(data) > 24 * 1024 * 1024:
                            raise ValueError("Encoded image exceeds 24 MiB")
                        path = Path(directory) / str(len(list(Path(directory).iterdir())))
                        path.write_bytes(base64.b64decode(data, validate=True))
                        part["image_url"]["url"] = str(path)
        yield request
