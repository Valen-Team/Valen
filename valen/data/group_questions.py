"""Regroup single-QA records into shared states. / 保留全部 QA，按相同上下文合并。"""
import argparse
from copy import deepcopy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from .schema import validate_record


def local_record(record, media_root):
    def local_path(path):
        return str((media_root / path).resolve())

    def local_item(item):
        kind = item["type"]
        if kind not in {"image_url", "video_url"}:
            return item
        url = item[kind]["url"]
        if "://" in url:
            raise ValueError("Materialize remote media before grouping questions")
        path = local_path(url)
        return item if path == url else {**item, kind: {**item[kind], "url": path}}

    def local_list(func, items):
        new_items = [func(item) for item in items]
        return items if all(new is old for new, old in zip(new_items, items)) else new_items

    def local_message(message):
        if isinstance(message["content"], str):
            return message
        content = local_list(local_item, message["content"])
        return message if content is message["content"] else {**message, "content": content}

    def local_asset(asset):
        path = local_path(asset["path"])
        return asset if path == asset["path"] else {**asset, "path": path}

    record = dict(record)
    state = record["request"]["state"]
    if isinstance(state, dict):
        messages = local_list(local_message, state["messages"])
        if messages is not state["messages"]:
            record["request"] = {**record["request"], "state": {**state, "messages": messages}}
    if "assets" in record:
        record["assets"] = local_list(local_asset, record["assets"])
    return record


def grouping_key(record):
    meta = record.get("meta", {})
    # 同组还必须具有相同输入和来源。 / A group ID alone does not identify identical context.
    value = [record["group_id"], record["request"]["state"], record.get("assets", []),
             {k: meta.get(k) for k in ("source", "domain", "modality", "experiment_modality", "language_bucket")}]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def group_questions(source, output, max_questions=4, media_root=None, work_dir=None):
    """Use a disk index so large text/video mixtures need no in-memory dataset copy."""
    if type(max_questions) is not int or max_questions <= 0:
        raise ValueError("max_questions must be a positive integer")
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if source == output or output.exists() or manifest_path.exists():
        raise ValueError("Choose a new output path; existing datasets are never overwritten")
    media_root = Path(media_root).resolve() if media_root else source.parent
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir).resolve() if work_dir else output.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    counts = dict(input_records=0, input_qa=0, output_records=0, output_qa=0)
    sizes, tasks, sources = Counter(), Counter(), Counter()
    source_hash, output_hash = hashlib.sha256(), hashlib.sha256()

    with tempfile.TemporaryDirectory(prefix=".shared-state-grouping-", dir=work_dir) as work:
        with sqlite3.connect(str(Path(work) / "pending.sqlite")) as index:
            # 仅用于临时索引，输出由原子替换发布。 / This disposable index needs no recovery journal.
            index.execute("PRAGMA journal_mode=OFF")
            index.execute("PRAGMA synchronous=OFF")
            index.execute("PRAGMA cache_size=-32768")
            index.execute("CREATE TABLE pending (key TEXT PRIMARY KEY, record TEXT, first_line INTEGER)")
            temporary = output.parent / (Path(work).name + ".jsonl.tmp")
            try:
                with source.open("rb") as reader, temporary.open("wb") as writer:
                    def emit(record, key):
                        validate_record(record)
                        count = len(record["request"]["questions"])
                        meta = record.setdefault("meta", {})
                        meta["record_id"] = f"shared-{key[:16]}-{counts['output_records']}"
                        meta["shared_state_num_questions"] = count
                        line = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                        writer.write(line)
                        output_hash.update(line)
                        counts["output_records"] += 1
                        counts["output_qa"] += count
                        sizes[count] += 1
                        for q in record["request"]["questions"].values():
                            tasks[q["type"]] += 1
                            sources[meta.get("source", "unknown")] += 1

                    for line_number, line in enumerate(reader, 1):
                        source_hash.update(line)
                        if not line.strip():
                            continue
                        try:
                            record = local_record(validate_record(json.loads(line)), media_root)
                        except (ValueError, KeyError, TypeError) as exc:
                            raise ValueError(f"{source}:{line_number}: {exc}") from exc
                        counts["input_records"] += 1
                        counts["input_qa"] += len(record["request"]["questions"])
                        key = grouping_key(record)
                        pending = index.execute("SELECT record, first_line FROM pending WHERE key=?", (key,)).fetchone()
                        combined = json.loads(pending[0]) if pending else None
                        first_line = pending[1] if pending else line_number
                        for qid, question in record["request"]["questions"].items():
                            if combined is None:
                                combined = deepcopy(record)
                                combined["request"]["questions"] = {}
                                combined["targets"] = {}
                                combined.setdefault("meta", {})["shared_state_questions"] = {}
                                first_line = line_number
                            questions = combined["request"]["questions"]
                            unique, suffix = qid, 2
                            while unique in questions:
                                unique = f"{qid}__{suffix}"
                                suffix += 1
                            questions[unique] = question
                            if qid in record.get("targets", {}):
                                combined["targets"][unique] = record["targets"][qid]
                            combined["meta"]["shared_state_questions"][unique] = {
                                "input_line": line_number, "qid": qid,
                                "record_id": record.get("meta", {}).get("record_id")}
                            if len(questions) == max_questions:
                                emit(combined, key)
                                combined = None
                        if combined is None:
                            index.execute("DELETE FROM pending WHERE key=?", (key,))
                        else:
                            index.execute("INSERT OR REPLACE INTO pending VALUES (?, ?, ?)",
                                          (key, json.dumps(combined, ensure_ascii=False), first_line))
                    for key, record, _ in index.execute("SELECT key, record, first_line FROM pending ORDER BY first_line"):
                        emit(json.loads(record), key)
                if not counts["input_records"]:
                    raise ValueError("Empty dataset")
                assert counts["input_qa"] == counts["output_qa"]
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)

    manifest = {"input": str(source), "output": str(output), "max_questions": max_questions,
                **counts, "question_count_histogram": dict(sorted(sizes.items())),
                "task_qa_counts": dict(tasks), "source_qa_counts": dict(sources),
                "input_sha256": source_hash.hexdigest(), "output_sha256": output_hash.hexdigest(),
                "media_paths": "absolute local paths; source datasets unchanged"}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-questions", type=int, default=4)
    parser.add_argument("--media-root")
    parser.add_argument("--work-dir", help="Directory for a temporary disk index (default: output directory)")
    args = parser.parse_args()
    print(json.dumps(group_questions(args.input, args.output, args.max_questions, args.media_root, args.work_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
