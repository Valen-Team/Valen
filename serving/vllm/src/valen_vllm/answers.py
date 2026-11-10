"""Return the same native answer semantics as Valen's Transformers model."""

import math

from valen_vllm.compiler import valen_options
from valen_vllm.grounding import box_answer


def answers(request, logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    # A bbox question returns x1, y1, x2, y2 and confidence instead of logits.
    expected = sum(
        5 if q.type == "bbox" else len(valen_options(q))
        for q in request.questions.values()
    )
    if len(logits) != expected or not all(math.isfinite(v) for v in logits):
        raise RuntimeError("Expected finite logits for every Valen candidate")
    result, offset = {}, 0
    for qid, q in request.questions.items():
        if q.type == "bbox":
            result[qid] = box_answer(logits[offset : offset + 5])
            offset += 5
            continue
        options = valen_options(q)
        values = logits[offset : offset + len(options)]
        offset += len(options)
        top = max(values)
        weights = [math.exp((v - top) / temperature) for v in values]
        total = math.fsum(weights)
        probs = [v / total for v in weights]
        keys = [key for key, _ in options]
        entry = {"type": q.type}
        if q.type == "noul":
            entry["noul"] = probs[0]
        else:
            entry["probabilities"] = dict(zip(keys, probs))
            mode = max(range(len(probs)), key=probs.__getitem__)
            if q.type == "choice":
                entry["choice"] = keys[mode]
                entry["confidence"] = (
                    1.0
                    if len(probs) == 1
                    else max(0.0, (probs[mode] - 1 / len(probs)) / (1 - 1 / len(probs)))
                )
            else:
                entry["score"] = sum(i * p for i, p in enumerate(probs))
                entry["legend"] = dict(options)
                distance = sum(p * abs(i - mode) for i, p in enumerate(probs))
                uniform_mad = sum(
                    abs(i - (len(probs) - 1) / 2) for i in range(len(probs))
                ) / len(probs)
                entry["confidence"] = max(0.0, 1.0 - distance / uniform_mad)
        result[qid] = entry
    return result
