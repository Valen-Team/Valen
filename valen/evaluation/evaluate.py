"""Evaluate labeled decisions once, sharded without padding across GPUs."""
import argparse
import json
from pathlib import Path
import time

import torch

from valen.training.checkpoint import file_sha256, load_checkpoint
from valen.training.distributed import initialize, close
from valen.evaluation.inference import answer
from valen.modeling.factory import build_model, build_compiler, get_backend, normalize_model_config
from valen.data.schema import read_jsonl
from .metrics import question_metrics, summarize
from .calibration import load_calibration, temperature_for, write_json
from .identity import checkpoint_identity, read_protocol
from valen.modeling.qwen.attention import ATTENTION_IMPLEMENTATIONS, with_attention_implementation


@torch.no_grad()
def run(checkpoint, data, output, device="cuda", attn_implementation=None, *,
        calibration=None, protocol=None, cache_role=None, group_namespace=None):
    if (calibration or cache_role) and not protocol:
        raise ValueError("--protocol is required for versioned calibration/cache identity")
    if cache_role and not group_namespace:
        raise ValueError("--group-namespace is required for stable split overlap checks")
    run_started = time.monotonic()
    context = initialize(device)
    output, data, checkpoint = Path(output), Path(data), Path(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    config = normalize_model_config(config)
    config = with_attention_implementation(config, attn_implementation)
    config.update(device=context.device, gradient_checkpointing=False)
    torch.manual_seed(config.get("seed", 42))
    model = build_model(config)
    payload = load_checkpoint(checkpoint, model)
    model.eval()
    identity = checkpoint_identity(checkpoint, config, model, read_protocol(protocol), attn_implementation) if protocol else None
    artifact = load_calibration(calibration, identity) if calibration else None
    temperature = artifact["temperature"] if artifact else 1.0
    backend = get_backend(config["architecture"])
    from valen.data.schema import target_distribution
    compiler = build_compiler(config, data.parent)
    candidate_fn = backend.candidates
    records = read_jsonl(data, candidate_fn=candidate_fn)
    indices = list(range(context.rank, len(records), context.world_size))
    context.barrier()
    rows, started = [], time.monotonic()
    def synchronize():
        if context.device.startswith("cuda"):
            torch.cuda.synchronize()
    path = output / f"predictions.rank{context.rank}.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for local_index, index in enumerate(indices):
            record = records[index]
            compile_started = time.monotonic()
            compiled = compiler.compile(record, labeled_only=True)
            compile_seconds = time.monotonic() - compile_started
            units = iter(backend.inference_units(model, compiled))
            while True:
                synchronize()
                forward_started = time.monotonic()
                unit = next(units, None)
                synchronize()
                if unit is None:
                    break
                unit_seconds = time.monotonic() - forward_started
                forward_seconds = unit_seconds / len(unit)
                for decision in unit:
                    question, logits = decision.question, decision.logits
                    meta = record.get("meta", {})
                    row = {"record_index": index, "record_id": meta.get("record_id"),
                           "group_id": record["group_id"], "qid": question.qid, "task": question.kind,
                           "modality": meta.get("modality", "text" if question.is_text else "media"),
                           "language": meta.get("language_bucket", "unknown"),
                           "domain": meta.get("domain", "unknown"),
                           "timing": {"compile_seconds": compile_seconds, "forward_seconds": forward_seconds,
                                      "compile_plus_forward_seconds": compile_seconds + forward_seconds},
                           "metrics": question_metrics(question, logits / temperature_for(temperature, question.kind)),
                           "raw_metrics": question_metrics(question, logits)}
                    if backend.unit_scope == "state" or getattr(compiled, "inputs", None) is not None:
                        row["timing"].update(shared_state_forward_seconds=unit_seconds,
                                             forward_scope="amortized_per_question")
                    rows.append(row)
                    scaled = logits.double() / temperature_for(temperature, question.kind)
                    probabilities = scaled.softmax(-1).cpu().tolist()
                    prediction = dict(row, target=dict(zip(question.keys, question.target)),
                                      probabilities=dict(zip(question.keys, probabilities)),
                                      logits=logits.detach().double().cpu().tolist(),
                                      candidates=list(question.keys),
                                      answer=answer(question, logits, temperature_for(temperature, question.kind)))
                    stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            if (local_index + 1) % 50 == 0 or local_index + 1 == len(indices):
                stream.flush()
                print(json.dumps({"event": "evaluation_progress", "rank": context.rank,
                                  "records": local_index + 1, "total": len(indices),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    rank_timing = context.gather({"rank": context.rank, "records": len(indices),
                                 "evaluation_seconds": time.monotonic() - started,
                                 "load_and_evaluation_seconds": time.monotonic() - run_started,
                                 "device": torch.cuda.get_device_name() if context.device.startswith("cuda") else "cpu"})
    gathered = context.gather(rows)
    if context.primary:
        all_rows = [row for rank_rows in gathered for row in rank_rows]
        # Check coverage without recompiling media.
        expected = {(i, qid) for i, record in enumerate(records)
                    for qid, q in record["request"]["questions"].items()
                    if target_distribution(record.get("targets", {}).get(qid), [k for k, _ in candidate_fn(q)]) is not None}
        actual = [(row["record_index"], row["qid"]) for row in all_rows]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Evaluation duplicated or omitted labeled questions")
        report = {"checkpoint": str(checkpoint), "training_step": payload["progress"]["step"],
                  "base_model_path": backend.base_paths(config), "seed": config.get("seed", 42),
                  "architecture": config.get("architecture", "qwen"),
                  "training_epoch": payload["progress"]["epoch"], "data": str(data),
                  "data_sha256": file_sha256(data),
                  "checkpoint_sha256": file_sha256(checkpoint / "checkpoint.pt"),
                  "records": len(records), "questions": len(all_rows), "world_size": context.world_size,
                  "temperature": temperature, "calibration": artifact, "identity": identity, "elapsed_seconds": time.monotonic() - started,
                  "rank_timing": rank_timing,
                  "evaluation_wall_seconds": max(r["evaluation_seconds"] for r in rank_timing),
                  "load_and_evaluation_wall_seconds": max(r["load_and_evaluation_seconds"] for r in rank_timing),
                  "metrics": summarize(all_rows),
                  "raw_metrics": summarize([dict(row, metrics=row["raw_metrics"]) for row in all_rows])}
        if hasattr(model, "attention_backends"):
            report["attention_backends"] = model.attention_backends
        report["questions_per_second"] = len(all_rows) / report["evaluation_wall_seconds"]
        report["latency_seconds"] = {}
        for name in ("compile_seconds", "forward_seconds", "compile_plus_forward_seconds"):
            values = torch.tensor([row["timing"][name] for row in all_rows], dtype=torch.float64)
            report["latency_seconds"][name] = {"mean": float(values.mean()), "p50": float(values.quantile(.5)),
                                               "p95": float(values.quantile(.95))}
        (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        predictions = []
        for rank in range(context.world_size):
            with (output / f"predictions.rank{rank}.jsonl").open(encoding="utf-8") as stream:
                predictions.extend(json.loads(line) for line in stream)
        predictions.sort(key=lambda row: (row["record_index"], row["qid"]))
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if cache_role:
            cache_rows = []
            for row in predictions:
                if row["group_id"] is None:
                    raise ValueError("Every cache row needs a stable source group_id")
                keys = row["candidates"]
                cache_rows.append({"id": f"{row['record_id'] or row['record_index']}:{row['qid']}",
                                   "group_id": str(row["group_id"]), "type": row["task"],
                                   "candidates": keys, "target": [row["target"][k] for k in keys],
                                   "logits": row["logits"], "status": "ok",
                                   **{key: row[key] for key in ("domain", "modality", "language")}})
            write_json(output / "reliability-cache.json", {
                "schema_version": 1, "role": cache_role, "identity": identity,
                "dataset": {"sha256": file_sha256(data), "source": str(data)},
                "group_namespace": group_namespace, "rows": cache_rows})
        print(json.dumps({"event": "evaluation_complete", "questions": len(all_rows),
                          "metrics_path": str(output / "metrics.json")}), flush=True)
    context.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", choices=ATTENTION_IMPLEMENTATIONS,
                        help="Override the Qwen checkpoint's attention backend")
    parser.add_argument("--calibration", help="Versioned fitted temperature artifact")
    parser.add_argument("--protocol", help="JSON deployment recipe shared across all splits")
    parser.add_argument("--cache-role", choices=("calibration", "policy", "test", "unassigned"))
    parser.add_argument("--group-namespace", help="Stable source-entity namespace, shared across splits")
    args = parser.parse_args()
    try:
        extra = {key: getattr(args, key) for key in ("calibration", "protocol", "cache_role", "group_namespace")
                 if getattr(args, key) is not None}
        run(args.checkpoint, args.data, args.output, args.device, args.attn_implementation, **extra)
    finally:
        close()


if __name__ == "__main__":
    main()
