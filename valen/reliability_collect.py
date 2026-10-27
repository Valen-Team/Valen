"""Collect raw logits from an already downloaded native Valen release, offline."""
import argparse
import hashlib
import json
from pathlib import Path

from valen.data.schema import candidates, target_distribution
from valen.evaluation.calibration import digest, raw_logits, validate_cache, write_json


def fingerprint_folder(folder):
    suffixes = {".safetensors", ".pt", ".pth", ".bin", ".json", ".py", ".model", ".txt", ".jinja", ".tiktoken"}
    ignored = {".git", ".cache", ".venv", "__pycache__"}
    result = {}
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file() or path.suffix not in suffixes or ignored.intersection(path.relative_to(folder).parts):
            continue
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        result[str(path.relative_to(folder))] = h.hexdigest()
    return result


class LocalRelease:
    def __init__(self, folder, protocol, *, model_id=None, revision=None, device="cuda", attention="sdpa",
                 execution="shared_state", media_root="."):
        folder = Path(folder).resolve()
        if not folder.is_dir():
            raise ValueError("Supply an existing local model directory; this tool never downloads weights")
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.model = AutoModel.from_pretrained(str(folder), trust_remote_code=True, local_files_only=True,
                                              dtype="auto", attn_implementation=attention).to(device).eval()
        processor = AutoProcessor.from_pretrained(str(folder), trust_remote_code=True, local_files_only=True)
        self.compiler = self.model.make_compiler(media_root, execution, processor=processor)
        artifacts = fingerprint_folder(folder)
        name = folder.name
        model_id = model_id or ("Valen-Team/" + name if name in ("Valen-0.8B", "Valen-2B", "Valen-4B") else name)
        self.identity = {"model": model_id, "revision": revision or digest(artifacts),
                         "backend": f"transformers:{execution}:{attention}:{torch.device(device).type}",
                         "precision": str(next(self.model.parameters()).dtype), "artifacts_sha256": artifacts,
                         "protocol_sha256": digest(protocol)}

    def logits(self, request):
        if "temperature" in request:
            raise ValueError("Collect raw logits without request temperature overrides")
        with self.torch.inference_mode():
            # Labels, source IDs and metadata never enter the compiler.
            state = self.compiler.compile({"request": request})
            values = self.model.forward_state(state) if state.inputs is not None else [self.model(q) for q in state.questions]
            if len(values) != len(state.questions):
                raise ValueError("Model omitted a question")
            return {q.qid: (list(q.keys), z.detach().double().cpu().tolist()) for q, z in zip(state.questions, values)}


def collect(model, records, role, group_namespace, dataset):
    rows, seen = [], set()
    for index, record in enumerate(records):
        group = record.get("group_id")
        if group is None or not str(group):
            raise ValueError("Each record requires a stable source-entity group_id")
        record_id = str(record.get("meta", {}).get("record_id", index))
        expected = []
        for qid, question in record["request"]["questions"].items():
            keys = [k for k, _ in candidates(question)]
            truth = target_distribution(record.get("targets", {}).get(qid), keys)
            if truth is None:
                continue
            if len(keys) < 2:
                raise ValueError("Single-candidate questions cannot demonstrate calibration")
            row_id = record_id + ":" + qid
            if row_id in seen:
                raise ValueError("Duplicate record/question identity")
            seen.add(row_id)
            expected.append({"id": row_id, "group_id": str(group), "type": question["type"],
                             "candidates": keys, "target": truth,
                             **{k: record.get("meta", {}).get(k, "unknown") for k in ("domain", "modality", "language")}})
        if not expected:
            continue
        try:
            answers = model.logits(record["request"])
            for row in expected:
                qid = row["id"][len(record_id) + 1:]
                keys, logits = answers[qid]
                if set(keys) != set(row["candidates"]):
                    raise ValueError("Native candidate set mismatch")
                row.update(status="ok", logits=[logits[keys.index(k)] for k in row["candidates"]])
                raw_logits(row)
        except Exception as exc:
            # Retain every labeled question in the denominator. Store the error
            # class, never exception text that could contain private inputs.
            for row in expected:
                row.pop("logits", None)
                row.update(status="error", error_type=type(exc).__name__)
        rows.extend(expected)
    return {"schema_version": 1, "identity": model.identity, "role": role,
            "dataset": dataset, "group_namespace": group_namespace, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--revision", help="Pinned release revision; defaults to a local artifact fingerprint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--role", choices=("calibration", "policy", "test", "unassigned"), required=True)
    parser.add_argument("--group-namespace", required=True)
    parser.add_argument("--media-root", default=".")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--execution", choices=("question", "shared_state"), default="shared_state")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    data = Path(args.data)
    model = LocalRelease(args.model_dir, json.loads(Path(args.protocol).read_text()), model_id=args.model_id,
                         revision=args.revision, device=args.device, attention=args.attention,
                         execution=args.execution, media_root=args.media_root)
    with data.open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    artifact = collect(model, records, args.role, args.group_namespace,
                       {"source": str(data), "sha256": hashlib.sha256(data.read_bytes()).hexdigest()})
    validate_cache(artifact, args.role)
    write_json(args.output, artifact)


if __name__ == "__main__":
    main()
