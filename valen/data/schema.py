"""Strict JSONL contract. IDs and labels never enter the model prompt."""
import json
import math
from pathlib import Path


def candidates(question):
    kind = question.get("type")
    if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
        raise ValueError("instructions must be nonempty text")
    criteria = question.get("criteria")
    if kind == "bbox":
        # Localization has no candidates; the image tokens are the choices.
        # 定位题没有文本候选，图像 token 即候选。
        image = question.get("image", 0)
        if type(image) is not int or image < 0:
            raise ValueError("bbox image must be a nonnegative image index")
        return []
    if kind == "noul":
        return [("true", "True / 是：满足问题中的条件。"), ("false", "False / 否：不满足问题中的条件。")]
    if kind == "choice":
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
            raise ValueError("Choice needs 1–255 named criteria")
        result = list(criteria.items())
    elif kind == "score":
        # Counting benchmarks can have eleven or more bins. / 计数题可能超过十档。
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 255:
            raise ValueError("Score criteria must be an ordered list of 2–255 descriptions")
        result = [(str(i), text) for i, text in enumerate(criteria)]
    else:
        raise ValueError(f"Unsupported question type: {kind}")
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip() for k, v in result):
        raise ValueError("Candidate names and descriptions must be nonempty strings")
    return result


def box_target(target):
    """Normalized [x1, y1, x2, y2] box or [x, y] point. / 归一化框或点。"""
    if target is None or (target.get("bbox") is None and target.get("point") is None):
        return None
    if target.get("bbox") is not None:
        box = target["bbox"]
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box)):
            raise ValueError("bbox target must be four finite numbers")
        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("bbox target must be normalized with x1<x2, y1<y2 in [0, 1]")
        return {"bbox": [x1, y1, x2, y2]}
    point = target["point"]
    if (not isinstance(point, (list, tuple)) or len(point) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in point)):
        raise ValueError("point target must be two numbers in [0, 1]")
    return {"point": [float(v) for v in point]}


def question_target(question, target, keys):
    """Training target for any question type. / 任意题型的监督目标。"""
    if question.get("type") == "bbox":
        return box_target(target)
    return target_distribution(target, keys)


def fold_noul_criteria(question):
    """Move a Noul rubric into the instructions, exactly as the eval harness does.

    Qwen's Noul head reads fixed true/false candidate descriptions, so a custom rubric would
    otherwise be invisible during training while JevalKit appends it at evaluation time.
    Qwen 的 Noul 头使用固定的 true/false 候选描述；自定义判定标准必须拼进题面，
    否则训练看不到、评测却能看到，两边不一致。
    """
    criteria = question.get("criteria")
    if question.get("type") != "noul" or not criteria:
        return question
    if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
        raise ValueError("Noul criteria must define true and false")
    question = {k: v for k, v in question.items() if k != "criteria"}
    question["instructions"] = (question["instructions"] + "\nTrue criterion: " + str(criteria["true"])
                                + "\nFalse criterion: " + str(criteria["false"]))
    return question


def target_distribution(target, keys):
    if target is None or target.get("probabilities") is None:
        return None
    probs = target["probabilities"]
    if not isinstance(probs, dict) or set(probs) != set(keys):
        raise ValueError(f"Probability keys must exactly match {keys}")
    values = [probs[k] for k in keys]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Probabilities must be finite nonnegative numbers")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("Target probabilities must sum to 1")
    return values


def validate_record(record, candidate_fn=candidates):
    request = record["request"]
    state = request["state"]
    if not isinstance(state, (str, dict)):
        raise ValueError("state must be text or a messages object")
    if not isinstance(request["questions"], dict) or not request["questions"]:
        raise ValueError("questions must be a nonempty mapping")
    targets = record.get("targets", {})
    if set(targets) - set(request["questions"]):
        raise ValueError("Unknown target question IDs")
    for qid, question in request["questions"].items():
        question_target(question, targets.get(qid), [k for k, _ in candidate_fn(question)])
    if not record.get("group_id"):
        raise ValueError("group_id is required for leakage-free splits")
    return record


def read_jsonl(path, candidate_fn=candidates):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(validate_record(json.loads(line), candidate_fn))
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError("Empty dataset")
    return records
