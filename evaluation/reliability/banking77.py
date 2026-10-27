"""Replay retained BANKING77 predictions; no network or model inference.

Run against the retained JevalKit data root. --stdout-gzip performs no writes
and exports aggregate reports only, for bandwidth-conscious remote replay.
"""
import argparse
import csv
import gzip
import hashlib
import json
import lzma
import re
import sys
from pathlib import Path

from valen.evaluation.calibration import (
    digest, fit, report, select_policy, scored, evaluate_policy, save_report, write_json,
)

SPLIT_SEED = "valen-reliability-banking77-v1"
BUDGETS = [.01, .02, .05, .10]
SOURCE_URL = "https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_group(state):
    # Public corpus has no customer/conversation identifiers. Keep identical
    # normalized utterances together; do not claim semantic/entity deduplication.
    if not isinstance(state, str):
        raise ValueError("BANKING77 replay requires a text state")
    return digest(re.sub(r"\s+", " ", state.casefold()).strip())


def split_role(group):
    bucket = int(digest([SPLIT_SEED, group])[:16], 16) % 100
    return "calibration" if bucket < 30 else "policy" if bucket < 60 else "test"


def target_vector(target, keys):
    if isinstance(target, dict) and "probabilities" in target:
        if set(target["probabilities"]) != set(keys):
            raise ValueError("Target candidate set mismatch")
        return [target["probabilities"][k] for k in keys]
    label = str(target["label"] if isinstance(target, dict) else target)
    if label not in keys:
        raise ValueError("Gold label not in candidates")
    return [float(k == label) for k in keys]


def released_identity(manifest, release):
    runtime, published = manifest["runtime"], release["model_source"]
    recorded = {**runtime["weights"], **runtime["runtime_sha256"], "config.json": runtime["config_sha256"]}
    if any(published["inference_files_sha256"].get(k) != v for k, v in recorded.items()):
        raise ValueError("Historical run does not match the pinned published weights/runtime")
    protocol = {"scenario": "BANKING77", "rendering": "JevalKit retained request, all 77 candidates in source order",
                "execution": runtime["execution"], "attention": runtime["attention"], "raw_temperature": 1.0,
                "source_sha256": manifest["signature"]["source_sha256"]}
    return {"model": published["repo_id"], "revision": published["revision"],
            "backend": f"transformers:{runtime['execution']}:{runtime['attention']}:cuda",
            "precision": runtime["dtype"], "protocol_sha256": digest(protocol),
            "artifacts_sha256": recorded}, protocol


def collect(root, model, source, release_root):
    run = root / "results" / "full" / model / "Banking77"
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["signature"]["limit"] is not None or manifest["runtime"]["temperature"] != 1.0:
        raise ValueError("Require complete, raw T=1 predictions")
    source_path = root / "datasets" / "Banking77.tsv"
    dataset_hash = file_hash(source_path)
    if dataset_hash != manifest["signature"]["dataset_sha256"]:
        raise ValueError("Retained dataset hash changed")
    release = json.loads((release_root / "runs" / model / "environment.json").read_text())
    identity, protocol = released_identity(manifest, release)
    path = run / "predictions.jsonl"
    prediction_hash, manifest_hash = file_hash(path), file_hash(manifest_path)
    original_metrics = json.loads((run / "metrics.json").read_text())
    if (original_metrics.get("predictions_sha256") != prediction_hash
            or original_metrics.get("manifest_sha256") != manifest_hash):
        raise ValueError("Retained run integrity mismatch against original metrics hashes")
    predictions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {p["index"]: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r["index"] for r in source}:
        raise ValueError("Predictions must cover the exact source requests once")
    caches = {role: {"schema_version": 1, "identity": identity, "role": role,
                     "dataset": {"name": "BANKING77 official test / retrospective three-way replay",
                                 "sha256": dataset_hash, "source_url": SOURCE_URL,
                                 "predictions_sha256": prediction_hash, "manifest_sha256": manifest_hash,
                                 "original_metrics_sha256": file_hash(run / "metrics.json")},
                     "group_namespace": "BANKING77-normalized-utterance-v1", "rows": []}
              for role in ("calibration", "policy", "test")}
    for record in source:
        request, targets = record["request"], record["targets"]
        prediction = by_id[record["index"]]
        group = source_group(request["state"])
        role = split_role(group)
        for qid, question in request["questions"].items():
            if question["type"] != "choice" or len(question["criteria"]) != 77:
                raise ValueError("Expected the frozen 77-way intent task")
            keys = list(question["criteria"])
            row = {"id": digest([record["index"], qid]), "group_id": group, "type": "choice",
                   "candidates": keys, "target": target_vector(targets[qid], keys),
                   "status": prediction["status"], "modality": "text", "domain": "banking", "language": "en"}
            if row["status"] == "ok":
                answer = prediction["response"]["answers"][qid]
                p = answer["probabilities"]
                if set(p) != set(keys):
                    raise ValueError("Prediction candidate set mismatch")
                row["probabilities"] = [p[k] for k in keys]
            caches[role]["rows"].append(row)
    return caches, protocol


def replay(root, release_root):
    source = []
    with (root / "datasets" / "Banking77.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            source.append({"index": row["index"], "request": json.loads(row["request"]),
                           "targets": json.loads(row["targets"])})
    results = {}
    for model in ("Valen-0.8B", "Valen-2B", "Valen-4B"):
        caches, protocol = collect(root, model, source, release_root)
        artifact = fit(caches["calibration"])
        policies = [select_policy(caches["policy"], artifact, b) for b in BUDGETS]
        result = report(caches["test"], artifact, policies[0])
        values = scored(caches["test"]["rows"], artifact["temperature"])
        result["operating_points"] = [{"validation_target": p["max_error"], "policy": p,
                                       "test": evaluate_policy(values, p)} for p in policies]
        result["study"] = {
            "name": "BANKING77 retrospective reliability replay v1", "split_seed": SPLIT_SEED,
            "replay_sources_sha256": globals().get("_REPLAY_SOURCE_HASHES") or {
                "valen/evaluation/calibration.py": file_hash(Path(__file__).resolve().parents[2] / "valen/evaluation/calibration.py"),
                "evaluation/reliability/banking77.py": file_hash(__file__)},
            "split": "SHA-256(seed, normalized utterance hash) modulo 100: 0..29 calibration, 30..59 policy, 60..99 test",
            "protocol": protocol, "source_url": SOURCE_URL, "source_license": "CC-BY-4.0",
            "prediction_source": "retained JevalKit full runs, 2026-10-16; no new inference",
            "identity_verification": "original metrics checksums verified retained predictions/manifest; recorded weights, config and runtime-code hashes matched published pinned release metadata",
            "limitations": ["Retrospective partition of a public benchmark already used in prior evaluations; not a fresh blind test.",
                            "No customer/conversation IDs; exact case/whitespace-normalized utterances stay together, semantic near duplicates are not certified absent.",
                            "Pretraining and adaptation overlap with BANKING77 cannot be ruled out.",
                            "English 77-way intent recognition only; not evidence for images, video, open-set rejection or production financial decisions.",
                            "Historical run did not separately hash tokenizer files; artifacts bind recorded weights/config/runtime code.",
                            "Validation Wilson bounds are pointwise diagnostics; threshold search and distribution shift prevent a formal error guarantee."]}
        results[model] = result
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--stdout-gzip", action="store_true")
    transport.add_argument("--stdout-xz", action="store_true", help="Larger compression dictionary for low-bandwidth aggregate export")
    args = parser.parse_args(argv)
    if not args.output and not args.stdout_gzip and not args.stdout_xz:
        parser.error("Use --output, --stdout-gzip or --stdout-xz")
    results = replay(args.root, args.release_root)
    if args.stdout_gzip or args.stdout_xz:
        payload = json.dumps(results, allow_nan=False, separators=(",", ":")).encode()
        packed = lzma.compress(payload) if args.stdout_xz else gzip.compress(payload, mtime=0)
        sys.stdout.buffer.write(packed)
    else:
        for model, result in results.items():
            save_report(args.output / model, result)
            write_json(args.output / model / "calibration.json", result["calibration"])
            write_json(args.output / model / "protocol.json", result["study"]["protocol"])
            for i, operating in enumerate(result["operating_points"]):
                write_json(args.output / model / f"policy-{i}.json", operating["policy"])


if __name__ == "__main__":
    main()
