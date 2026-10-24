import argparse
import json
import math
from pathlib import Path
import torch
from valen import MODEL_NAME
from valen.training.checkpoint import load_checkpoint
from valen.modeling.factory import build_model, build_compiler, get_backend, backend_for_model, normalize_model_config
from valen.data.schema import read_jsonl
from valen.modeling.qwen.attention import ATTENTION_IMPLEMENTATIONS, with_attention_implementation


def answer(question, logits, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be finite and positive")
    if logits.ndim != 1 or len(logits) != len(question.keys) or not torch.isfinite(logits).all():
        raise ValueError("Expected one finite logit per candidate")
    scores = logits.double()
    probs = ((scores - scores.max()) / temperature).softmax(-1).detach().cpu().tolist()
    total = sum(probs)
    probs = [p / total for p in probs]
    result = {"type": question.kind}
    if question.kind == "noul":
        result["noul"] = probs[question.keys.index("true")]
        return result
    result["probabilities"] = dict(zip(question.keys, probs))
    mode = max(range(len(probs)), key=probs.__getitem__)
    if question.kind == "choice":
        result["choice"] = question.keys[mode]
        result["confidence"] = 1.0 if len(probs) == 1 else max(0.0, (max(probs) - 1/len(probs)) / (1 - 1/len(probs)))
    else:
        result["score"] = sum(i * p for i, p in enumerate(probs))
        result["legend"] = dict(zip(question.keys, question.descriptions))
        # Independently implemented formula from pinned system-one adapter:
        # fb52b1030b7fc1f4f1cf39910afa5da54f9835e3, confidence_metrics.py.
        distance = sum(p * abs(i - mode) for i, p in enumerate(probs))
        uniform_mad = sum(abs(i - (len(probs)-1)/2) for i in range(len(probs))) / len(probs)
        result["confidence"] = max(0.0, 1.0 - distance / uniform_mad)
    return result


@torch.no_grad()
def predict(model, compiled, temperature=1.0):
    model.eval()
    decisions = [d for unit in backend_for_model(model).inference_units(model, compiled) for d in unit]
    def calibrated(q):
        return temperature[q.kind] if isinstance(temperature, dict) else temperature
    return {"model": getattr(model, "model_name", MODEL_NAME),
            "answers": {d.question.qid: answer(d.question, d.logits, calibrated(d.question)) for d in decisions},
            "usage": {"input_tokens": compiled.logical_tokens, "output_tokens": 0},
            "internal_usage": {"compute_tokens": compiled.compute_tokens}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", choices=ATTENTION_IMPLEMENTATIONS,
                        help="Override the Qwen checkpoint's attention backend")
    args = parser.parse_args()
    config = json.loads((Path(args.checkpoint) / "config.json").read_text(encoding="utf-8"))
    config = normalize_model_config(config)
    config = with_attention_implementation(config, args.attn_implementation)
    config["device"] = args.device
    config["gradient_checkpointing"] = False
    model = build_model(config)
    load_checkpoint(args.checkpoint, model)
    backend = get_backend(config["architecture"])
    compiler = build_compiler(config, Path(args.data).parent)
    records = read_jsonl(args.data, candidate_fn=backend.candidates)
    temperature = json.loads(Path(args.calibration).read_text(encoding="utf-8"))["temperature"] if args.calibration else 1.0
    with Path(args.output).open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(predict(model, compiler.compile(record), temperature), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
