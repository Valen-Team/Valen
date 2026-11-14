"""Evaluate merged or LoRA releases. / 评估完整模型或未合并 LoRA 包。"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from valen.data.schema import target_distribution
from valen.evaluation.inference import answer
from valen.evaluation.metrics import question_metrics, summarize


def evaluate(model, data, output, device="cuda", execution="shared_state"):
    data, output = Path(data), Path(output)
    source = data / "eval.jsonl" if data.is_dir() else data
    output.mkdir(parents=True, exist_ok=True)
    processor_started = time.perf_counter()
    compiler = model.make_compiler(media_root=source.parent, execution=execution)
    processor_seconds = time.perf_counter() - processor_started
    model.eval()

    def synchronize():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    rows, record_count = [], 0
    synchronize()
    started = time.perf_counter()
    with (
        source.open(encoding="utf-8") as records,
        (output / "predictions.jsonl").open("w", encoding="utf-8") as stream,
        torch.inference_mode(),
    ):
        for line in records:
            if not line.strip():
                continue
            record = json.loads(line)
            index = record_count
            record_count += 1
            questions, targets = record["request"]["questions"], record.get("targets", {})
            if set(targets) != set(questions):
                raise ValueError(f"Record {index}: every question requires a target")
            # Only request and media reach the model. / 标签留在计分侧。
            compiled = compiler.compile({"request": record["request"], "assets": record.get("assets", [])})
            if {q.qid for q in compiled.questions} != set(questions):
                raise ValueError(f"Record {index}: compiled questions differ from the dataset")
            logits = model.forward_state(compiled) if compiled.inputs is not None else [model(q) for q in compiled.questions]
            meta = record.get("meta", {})
            for question, values in zip(compiled.questions, logits, strict=True):
                target = target_distribution(targets[question.qid], question.keys)
                if target is None:
                    raise ValueError(f"Record {index}: missing probabilities for {question.qid}")
                spec = SimpleNamespace(kind=question.kind, keys=question.keys,
                                       descriptions=question.descriptions, target=target)
                row = {"record_index": index, "record_id": meta.get("benchmark_record_id"),
                       "qid": question.qid, "task": question.kind,
                       "subset": meta.get("benchmark_subset", "unknown"),
                       "modality": meta.get("modality", "unknown"),
                       "language": meta.get("language_bucket", "unknown"),
                       "domain": meta.get("domain", "unknown"),
                       "target": dict(zip(question.keys, target)),
                       "probabilities": dict(zip(question.keys, values.float().softmax(-1).cpu().tolist())),
                       "answer": answer(spec, values), "metrics": question_metrics(spec, values)}
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if record_count % 50 == 0:
                stream.flush()
                print(json.dumps({"event": "progress", "records": record_count, "questions": len(rows)}), flush=True)
    synchronize()
    seconds = time.perf_counter() - started
    if not rows:
        raise ValueError("Dataset contains no labeled questions")
    # Keep soft-label Score in probability metrics. / 软标签不计硬标签准确率。
    report = {"data": str(source), "records": record_count, "questions": len(rows),
              "execution": execution, "world_size": 1,
              "device": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else str(device),
              "media_kwargs": compiler.media_kwargs,
              "processor_load_seconds": processor_seconds,
              "evaluation_seconds": seconds, "questions_per_second": len(rows) / seconds,
              "timing_scope": "JSONL read, media verification/preprocessing, forward, scoring and writes; excludes model/processor loading and downloads",
              "metrics": summarize(rows),
              "subsets": {name: summarize([row for row in rows if row["subset"] == name])
                          for name in sorted({row["subset"] for row in rows})}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Valen-Team/Valen-Decision-1.5-2B", help="Hub repository or local release directory")
    parser.add_argument("--data", required=True, help="Unpacked dataset root or native subset JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lora", action="store_true", help="Load the release's unmerged/ bundle")
    parser.add_argument("--revision", default="main", help="Hub branch or tag (for example, lora-ckpt)")
    parser.add_argument("--base-model", help="Optional local original Qwen base for --lora")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution", choices=("question", "shared_state"), default="shared_state")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    args = parser.parse_args()
    if args.base_model and not args.lora:
        parser.error("--base-model requires --lora")

    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    folder = Path(args.model)
    if not folder.is_dir():
        folder = Path(snapshot_download(args.model, revision=args.revision,
                                        allow_patterns="unmerged/*" if args.lora else None))
    options = {"trust_remote_code": True, "dtype": torch.bfloat16 if args.lora else "auto",
               "attn_implementation": args.attn_implementation, "local_files_only": False}
    if args.lora:
        folder /= "unmerged"
        if args.base_model:
            options["base_model_path"] = args.base_model
    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(folder, **options).to(args.device).eval()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    load_seconds = time.perf_counter() - load_started
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    report = evaluate(model, args.data, args.output, args.device, args.execution)
    report.update(model=args.model, revision=args.revision, format="unmerged" if args.lora else "complete",
                  attention=args.attn_implementation, load_seconds=load_seconds)
    path = Path(args.output) / "metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "questions": report["questions"],
                      "accuracy": report["metrics"]["overall"].get("accuracy"), "metrics": str(path)}), flush=True)


if __name__ == "__main__":
    main()
