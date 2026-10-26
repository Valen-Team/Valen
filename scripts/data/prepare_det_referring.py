"""Stage and activate 100k RefCOCO-family/ReasonSeg localization examples."""
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil

from PIL import Image

from scripts.data.prepare_spatial import (digest_file, materialize_coco, record_image, ref_record,
                                         save_dataset, source_rows, write_json)
from valen.data.spatial import read_spatial_jsonl, validate_spatial_record


PROTOCOL = "det_referring_reasonseg_v4"
FORMAT = "valen_det_referring_data_v4"
SOURCES = ("refcoco", "refcoco+", "refcocog")
BENCHMARKS = {f"{name}_{split}" for name in ("refcoco", "refcocoplus") for split in ("val", "testA", "testB")}
BENCHMARKS.update({"refcocog_val", "refcocog_test"})


def sample_key(record, image_identity=None):
    question = record["request"]["questions"]["target"]
    instruction = " ".join(question["instructions"].casefold().split()).rstrip(".!?")
    boxes = tuple(tuple(round(x, 6) for x in box) for box in record["targets"]["target"]["boxes"])
    return image_identity or record["group_id"], instruction, boxes


def sample_referring(pools, count, seed=42, max_per_image=12):
    """Balance sources and visit distinct images before drawing more instructions."""
    selected, seen, per_image = [], set(), Counter()
    rng = random.Random(seed)
    for index, source in enumerate(SOURCES):
        quota = count // len(SOURCES) + (index < count % len(SOURCES))
        grouped = defaultdict(list)
        for record in pools[source]:
            grouped[record["group_id"]].append(record)
        images = sorted(grouped)
        rng.shuffle(images)
        for rows in grouped.values():
            rng.shuffle(rows)
        chosen = []
        while len(chosen) < quota:
            advanced = False
            for image in images:
                if per_image[image] >= max_per_image:
                    continue
                while grouped[image]:
                    record = grouped[image].pop()
                    key = sample_key(record)
                    if key in seen:
                        continue
                    seen.add(key)
                    per_image[image] += 1
                    chosen.append(record)
                    advanced = True
                    break
                if len(chosen) == quota:
                    break
            if not advanced:
                raise ValueError(f"Insufficient unique {source} examples for quota {quota}")
        selected.extend(chosen)
    return selected


def evaluation_exclusions(base):
    hashes, groups = set(), set()
    directories = [base / "eval" / task / suffix for task in ("det", "seg", "gui")
                   for suffix in ("manifests", "reasonseg/manifests")]
    directories.append(base / "eval/det/protocols/coco_prompt/manifests")
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            manifest = json.loads(path.read_text())
            media = manifest.get("media_sha256", {})
            hashes.update(media.values())
            records = directory.parent / "processed" / (manifest.get("name", path.stem) + ".jsonl")
            if media and records.is_file():
                for record in source_rows(records):
                    groups.add(record["group_id"])
                    identifier = record.get("meta", {}).get("coco_id")
                    if identifier:
                        groups.add("coco:" + str(int(identifier)))
    return hashes, groups


def development(base, seed):
    root = base / "eval/det"
    records, hashes, seen = [], {}, set()
    rng = random.Random(seed)
    names = ("refcoco_val", "refcocoplus_val", "refcocog_val")
    for index, name in enumerate(names):
        rows = read_spatial_jsonl(root / "processed" / (name + ".jsonl"), labeled_only=True)
        rng.shuffle(rows)
        chosen = []
        for row in rows:
            key = sample_key(row)
            if key not in seen:
                seen.add(key)
                chosen.append(row)
            if len(chosen) == 512 // 3 + (index < 512 % 3):
                break
        records.extend(chosen)
        hashes.update(json.loads((root / "manifests" / (name + ".json")).read_text())["media_sha256"])
    reason_root = root / "reasonseg"
    reason = read_spatial_jsonl(reason_root / "processed/reasonseg_val.jsonl", labeled_only=True)
    if len(reason) != 200:
        raise ValueError("ReasonSeg development requires its complete official 200-image val split")
    for row in reason:
        record = copy.deepcopy(row)
        record["request"]["state"]["messages"][0]["content"][0]["image_url"]["url"] = "reasonseg/" + record_image(row)
        records.append(record)
    hashes.update({"reasonseg/" + name: value for name, value in json.loads(
        (reason_root / "manifests/reasonseg_val.json").read_text())["media_sha256"].items()})
    if len(records) != 712 or len({r["id"] for r in records}) != 712:
        raise ValueError("Development requires 512 RefCOCO-family and 200 ReasonSeg records")
    return records, hashes


def prepare(base, source_root, seed=42, workers=32):
    base, source_root = Path(base), Path(source_root)
    training, evaluation = base / "det", base / "eval/det"
    staging = training / "protocols/referring_v4"
    seal_path = training / "manifests/ready.json"
    seal = json.loads(seal_path.read_text())
    if seal.get("format") == FORMAT:
        if (digest_file(training / "processed/train.jsonl") != seal["records_sha256"]
                or digest_file(evaluation / "processed/dev.jsonl") != seal["development_sha256"]):
            raise ValueError("Activated referring data changed")
        return seal
    prepared_path = staging / "manifests/prepared.json"
    if prepared_path.is_file():
        prepared = json.loads(prepared_path.read_text())
        if prepared["previous_seal_sha256"] != digest_file(seal_path):
            raise ValueError("Original data seal changed while staging")
        if (digest_file(staging / "processed/train.jsonl") == prepared["records_sha256"]
                and digest_file(staging / "processed/dev.jsonl") == prepared["development_sha256"]):
            return prepared
        raise ValueError("Staged referring data changed")
    catalog = json.loads((training / "manifests/sources.json").read_text())
    provenance = next(entry for entry in catalog["sources"] if entry["dataset"].startswith("RefCOCO"))
    source_files = {entry["name"]: entry for entry in provenance["files"]}
    forbidden_hashes, forbidden_groups = evaluation_exclusions(base)
    pools, filtering = {}, {}
    for source in SOURCES:
        path = source_root / (source + "_train.json")
        if digest_file(path) != source_files[path.name]["sha256"]:
            raise ValueError("RefCOCO input differs from the pinned source manifest")
        counts, rows, seen = Counter(), [], set()
        for index, row in enumerate(source_rows(path)):
            counts["original"] += 1
            if "coco:" + str(int(row["id"])) in forbidden_groups:
                counts["evaluation_image"] += 1
                continue
            record = ref_record(row, source, "train", index, "det")
            try:
                validate_spatial_record(record)
            except (ValueError, TypeError):
                counts["invalid"] += 1
                continue
            key = sample_key(record)
            if key in seen:
                counts["duplicate"] += 1
                continue
            seen.add(key)
            rows.append(record)
        pools[source], filtering[source] = rows, dict(counts, eligible=len(rows))
        print(json.dumps({"source": source, **filtering[source]}), flush=True)
    reason = read_spatial_jsonl(training / "processed/reasonseg_train.jsonl", labeled_only=True)
    if len(reason) != 1324 or any(r["meta"]["source"] != "reasonseg" for r in reason):
        raise ValueError("Use all 1,324 unique official ReasonSeg training instructions")
    selected = sample_referring(pools, 100000 - len(reason), seed)
    selected.extend(reason)
    random.Random(seed).shuffle(selected)
    sources = {r["meta"]["image_id"]: f"train2014/COCO_train2014_{r['meta']['image_id']:012d}.jpg"
               for r in selected if r["meta"]["source"] in SOURCES}
    hashes = materialize_coco(staging, sources, preferred=[training, base / "seg"], workers=workers)
    reason_manifest = json.loads((training / "manifests/reasonseg_train.json").read_text())
    for name, expected in reason_manifest["media_sha256"].items():
        destination = staging / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(training / name, destination)
        if digest_file(destination) != expected:
            raise ValueError("ReasonSeg training media changed")
        hashes[name] = expected
    if set(hashes.values()) & forbidden_hashes:
        raise ValueError("Training media overlaps a protected development/evaluation source")
    sizes = {}
    for name in hashes:
        path = staging / name
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError("Training media must be independent regular files")
        with Image.open(path) as image:
            sizes[name] = image.size
            image.verify()
    seen = set()
    for record in selected:
        validate_spatial_record(record)
        name = record_image(record)
        if record["group_id"] in forbidden_groups or sizes[name] != (record["meta"]["width"], record["meta"]["height"]):
            raise ValueError("Training split or image dimensions do not match the annotation")
        key = sample_key(record, hashes[name])
        if key in seen:
            raise ValueError("Duplicate image/instruction/target training example")
        seen.add(key)
    counts = dict(Counter(r["meta"]["source"] for r in selected))
    if len(selected) != 100000 or set(counts) != {*SOURCES, "reasonseg"}:
        raise ValueError("Expected 100k genuine referring records, without COCO category-detection samples")
    train = save_dataset(staging, "train", selected, list(counts), hashes, seed=seed,
                         source_counts=counts, protocol=PROTOCOL, max_records_per_ref_image=12)
    dev_records, dev_hashes = development(base, seed)
    for record in dev_records:
        name = record_image(record)
        path = evaluation / name
        if path.is_symlink() or path.stat().st_nlink != 1 or digest_file(path) != dev_hashes[name]:
            raise ValueError("Development media must be independent verified files")
    dev = save_dataset(staging, "dev", dev_records, ["RefCOCO-family val", "ReasonSeg val"], dev_hashes,
                       seed=seed, purpose="development", media_root="eval/det")
    sources_catalog = {"sources": [dict(provenance, files=[source_files[s + "_train.json"] for s in SOURCES]),
        {"dataset": "LISA/ReasonSeg", "url": "https://github.com/JIA-Lab-research/LISA",
         "source_manifest_sha256": digest_file(training / "manifests/reasonseg-source.json"),
         "split": "official train", "images": 239, "instructions": 1324}],
        "image_origin": "RefCOCO retains its underlying COCO images; COCO category-detection records are excluded."}
    write_json(staging / "manifests/sources.json", sources_catalog)
    evaluations = {"det/" + name: json.loads((evaluation / "manifests" / (name + ".json")).read_text())["records_sha256"]
                   for name in sorted(BENCHMARKS)}
    evaluations.update({"det/" + name: json.loads((evaluation / "reasonseg/manifests" / (name + ".json")).read_text())["records_sha256"]
                        for name in ("reasonseg_val", "reasonseg_test")})
    prepared = {"format": FORMAT, "protocol": PROTOCOL, "validated": True, "task": "det",
        "seed": seed, "samples": 100000, "images": train["images"], "targets": train["targets"],
        "source_counts": counts, "records_sha256": train["records_sha256"],
        "development_sha256": dev["records_sha256"], "development_samples": 712,
        "evaluation_sha256": evaluations, "coco_training_samples": 0,
        "global_evaluation_overlap": 0, "independent_media": True,
        "previous_seal_sha256": digest_file(seal_path), "filtering": filtering,
        "sources_sha256": digest_file(staging / "manifests/sources.json"),
        "media_bytes": sum((staging / name).stat().st_size for name in hashes)}
    write_json(prepared_path, prepared)
    return prepared


def activate(base, prepared):
    base = Path(base)
    training, evaluation = base / "det", base / "eval/det"
    staging, seal = training / "protocols/referring_v4", training / "manifests/ready.json"
    current = json.loads(seal.read_text())
    if current.get("format") == FORMAT:
        return current
    if digest_file(seal) != prepared["previous_seal_sha256"]:
        raise ValueError("Stop legacy readers and preserve their original seal before activation")
    manifest = json.loads((staging / "manifests/train.json").read_text())
    for name, expected in manifest["media_sha256"].items():
        source, target = staging / name, training / name
        if source.exists() and digest_file(source) != expected:
            raise ValueError("Staged media changed before activation")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or target.stat().st_nlink != 1 or digest_file(target) != expected:
                raise ValueError("Existing training media differs from the staged independent copy")
            source.unlink(missing_ok=True)
        elif source.is_file():
            source.replace(target)
        else:
            raise ValueError("Staged training media is missing")
    for name, destination in (("train", training), ("dev", evaluation)):
        if digest_file(staging / "processed" / (name + ".jsonl")) != prepared["records_sha256" if name == "train" else "development_sha256"]:
            raise ValueError("Staged records changed before activation")
        target = destination / "processed" / (name + ".jsonl")
        temporary = target.with_suffix(".tmp")
        shutil.copyfile(staging / "processed" / (name + ".jsonl"), temporary)
        temporary.replace(target)
        write_json(destination / "manifests" / (name + ".json"), json.loads(
            (staging / "manifests" / (name + ".json")).read_text()))
    write_json(training / "manifests/sources.json", json.loads((staging / "manifests/sources.json").read_text()))
    from scripts.data.finalize_spatial import preview
    preview(training, "det")
    final = dict(prepared, time_utc=datetime.now(timezone.utc).isoformat())
    write_json(seal, final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True, help="Pinned original RefCOCO-family train JSON files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--activate", action="store_true", help="Publish only after stopping obsolete Det readers")
    args = parser.parse_args()
    result = prepare(args.data_root, args.source_root, args.seed, args.workers)
    if args.activate:
        result = activate(args.data_root, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"evaluation_sha256", "filtering"}}), flush=True)


if __name__ == "__main__":
    main()
