"""Build present-category COCO prompts without changing sealed legacy datasets."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil

from scripts.data.prepare_spatial import digest_file, record_image, save_dataset, write_json
from evaluation.det.coco import CONDITIONAL_BENCHMARK as BENCHMARK, SINGLE_CATEGORY_PROTOCOL as PROTOCOL
from valen.data.spatial import read_spatial_jsonl, validate_spatial_record


REVISION = "coco_single_category_v3"
FORMAT = "valen_coco_single_category_data_v3"


def development_in_eval(root, records=None, sources=None, hashes=None, **metadata):
    """Publish independent development media without changing a live legacy input."""
    root = Path(root)
    training, evaluation = root / "det", root / "eval/det"
    destination = evaluation / "protocols/coco_prompt"
    path = destination / "processed/dev.jsonl"
    if records is None:
        seal = json.loads((training / "manifests/ready.json").read_text())
        if seal.get("format") not in {"valen_coco_prompt_data_v2", FORMAT} or seal.get("validated") is not True:
            raise ValueError("Relocate only an audited COCO prompt revision")
        if path.is_file():
            manifest = json.loads((destination / "manifests/dev.json").read_text())
            if digest_file(path) != manifest["records_sha256"] or manifest["records_sha256"] != seal["development_sha256"]:
                raise ValueError("Canonical COCO development records changed")
            for name, expected in manifest["media_sha256"].items():
                media = evaluation / name
                if media.is_symlink() or media.stat().st_nlink != 1 or digest_file(media) != expected:
                    raise ValueError("Development media must remain independent verified copies")
            return manifest
        legacy = training / "processed/dev_coco_prompt_v2.jsonl"
        manifest = json.loads((training / "manifests/dev_coco_prompt_v2.json").read_text())
        if digest_file(legacy) != manifest["records_sha256"] or manifest["records_sha256"] != seal["development_sha256"]:
            raise ValueError("Legacy COCO development records changed")
        records = read_spatial_jsonl(legacy, labeled_only=True)
        sources, hashes = manifest["sources"], manifest["media_sha256"]
        metadata = {key: value for key, value in manifest.items() if key not in {
            "name", "samples", "images", "targets", "sources", "records_sha256", "media_sha256"}}
    for name in sorted({record_image(record) for record in records}):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Development media must use relative task paths")
        media = evaluation / name
        if media.is_symlink():
            raise ValueError("Development media must be independent verified copies")
        if not media.exists():
            media.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(training / name, media)
        if media.is_symlink() or media.stat().st_nlink != 1 or digest_file(media) != hashes[name]:
            raise ValueError("Development media must be independent verified copies")
    metadata.update(media_root="../..", purpose="development", validated=True, independent_media=True)
    result = save_dataset(destination, "dev", records, sources, hashes, **metadata)
    if (training / "manifests/ready.json").is_file():
        seal = json.loads((training / "manifests/ready.json").read_text())
        if seal.get("format") in {"valen_coco_prompt_data_v2", FORMAT} and result["records_sha256"] != seal["development_sha256"]:
            raise ValueError("Development relocation must preserve the sealed records byte for byte")
    return result


def grouped_queries(record, max_categories=1, seed=42):
    """Ask each present category separately, retaining every instance of that class."""
    if max_categories != 1:
        raise ValueError("COCO prompts must contain exactly one present category")
    target = record["targets"]["target"]
    positive = sorted(set(target["labels"]))
    key = hashlib.sha256((str(seed) + ":" + record["id"]).encode()).digest()
    random.Random(int.from_bytes(key, "big")).shuffle(positive)
    original = record["request"]["questions"]["target"]
    for index, label in enumerate(positive):
        category_id = record["meta"]["category_ids"][label]
        result = copy.deepcopy(record)
        result["id"] = record["id"].split(":prompt:")[0].split(":category:")[0] + f":category:{category_id}"
        question = result["request"]["questions"]["target"]
        question["categories"] = [original["categories"][label]]
        question["instructions"] = "Find every instance of the category: " + question["categories"][0] + "."
        selected = [box for box, actual in zip(target["boxes"], target["labels"]) if actual == label]
        result["targets"]["target"] = {"boxes": selected, "labels": [0] * len(selected)}
        result["meta"].update(protocol=PROTOCOL, query_index=index, category_ids=[category_id])
        validate_spatial_record(result)
        yield result


def training_queries(records, seed=42):
    """Select one genuine category per image; never create additional image prompts."""
    primary, observed = [], set()
    for record in records:
        if record["group_id"] in observed:
            raise ValueError("COCO training must contain each image exactly once")
        observed.add(record["group_id"])
        first = next(grouped_queries(record, seed=seed), None)
        if first is None:
            raise ValueError("COCO training requires a present annotated category")
        primary.append(first)
    return primary, []


def check_seal(root, seal):
    training, evaluation = root / "det", root / "eval/det"
    checks = ((training / "processed/train.jsonl", "records_sha256"),
              (evaluation / "protocols/coco_prompt/processed/dev.jsonl", "development_sha256"),
              (evaluation / "protocols/coco_prompt/processed" / (BENCHMARK + ".jsonl"), "evaluation_sha256"))
    if not seal.get("validated") or any(digest_file(path) != seal[key] for path, key in checks):
        raise ValueError("Activated COCO prompt data changed")


def prepare(root, seed=42, dev_images=512):
    """Stage a single-category revision without modifying any active training input."""
    root = Path(root)
    training, evaluation = root / "det", root / "eval/det"
    seal_path = training / "manifests/ready.json"
    seal = json.loads(seal_path.read_text())
    if seal.get("format") == FORMAT:
        if seal.get("seed") != seed or seal.get("coco_development_images") != dev_images:
            raise ValueError("An activated immutable revision cannot be silently resampled")
        check_seal(root, seal)
        from scripts.data.finalize_spatial import preview
        preview(training, "det")
        return seal
    if not seal.get("validated"):
        raise ValueError("Prepare from an audited immutable dataset")
    source = training / "processed/train.jsonl"
    if digest_file(source) != seal["records_sha256"]:
        raise ValueError("Training input changed after its audit")
    records = read_spatial_jsonl(source, labeled_only=True)
    rng = random.Random(seed)
    original = json.loads((training / "manifests/train.json").read_text())
    if seal.get("format") == "valen_coco_prompt_data_v2":
        if seal.get("seed") != seed or seal.get("coco_development_images") != dev_images:
            raise ValueError("Preserve the existing image split when simplifying COCO prompts")
        check_seal(root, seal)
        dev_source = evaluation / "protocols/coco_prompt/processed/dev.jsonl"
        dev_manifest = json.loads((evaluation / "protocols/coco_prompt/manifests/dev.json").read_text())
        eval_source = evaluation / "protocols/coco_prompt/processed" / (BENCHMARK + ".jsonl")
        eval_manifest = json.loads((evaluation / "protocols/coco_prompt/manifests" / (BENCHMARK + ".json")).read_text())
        development = read_spatial_jsonl(dev_source, labeled_only=True)
        evaluation_records = read_spatial_jsonl(eval_source, labeled_only=True)
        hashes, sources = original["media_sha256"], original["sources"]
        train = []
        for record in records:
            train.extend(training_queries([record], seed)[0] if record["meta"]["source"] == "coco2017" else [record])
        dev_hashes = dev_manifest["media_sha256"]
        dev_sources = dev_manifest["sources"]
        eval_hashes = eval_manifest["media_sha256"]
        catalog_images = eval_manifest["catalog_images"]
        empty_images = eval_manifest["images_without_positive_queries"]
    else:
        eval_source = evaluation / "processed/coco2017_val2017.jsonl"
        if digest_file(eval_source) != seal["evaluation_sha256"]["det/coco2017_val2017"]:
            raise ValueError("COCO evaluation changed after its audit")
        coco = [r for r in records if r["meta"]["source"] == "coco2017"]
        if not 0 < dev_images < len(coco):
            raise ValueError("Development images must be a nonempty proper subset of COCO training")
        dev_groups = {r["group_id"] for r in rng.sample(coco, dev_images)}
        development = [r for r in coco if r["group_id"] in dev_groups]
        train, _ = training_queries([r for r in coco if r["group_id"] not in dev_groups], seed)
        referring = [r for r in records if r["meta"]["source"] != "coco2017" and r["group_id"] not in dev_groups]
        reasonseg = []
        reason_path = training / "processed/reasonseg_train.jsonl"
        reason_manifest = training / "manifests/reasonseg_train.json"
        provenance = json.loads(reason_manifest.read_text()) if reason_manifest.is_file() else {}
        if provenance:
            if digest_file(reason_path) != provenance["records_sha256"]:
                raise ValueError("ReasonSeg training input changed after preparation")
            reasonseg = read_spatial_jsonl(reason_path, labeled_only=True)
        needed = 100000 - len(train) - len(reasonseg)
        if not 0 <= needed <= len(referring):
            raise ValueError("Cannot fill 100k without repeating COCO images")
        train.extend(rng.sample(referring, needed))
        train.extend(reasonseg)
        rng.shuffle(train)
        ref_dev = read_spatial_jsonl(evaluation / "processed/dev.jsonl", labeled_only=True)
        ref_dev = copy.deepcopy(rng.sample(ref_dev, min(256, len(ref_dev))))
        dev_hashes = dict(original["media_sha256"])
        for record in ref_dev:
            relative = record_image(record)
            dev_hashes[relative] = digest_file(evaluation / relative)
        for record in development:
            name = record_image(record)
            target = evaluation / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(training / name, target)
        development += ref_dev
        hashes = {**original["media_sha256"], **provenance.get("media_sha256", {})}
        sources = original["sources"] + provenance.get("sources", [])
        dev_sources = original["sources"]
        eval_manifest = json.loads((evaluation / "manifests/coco2017_val2017.json").read_text())
        eval_hashes = eval_manifest["media_sha256"]
        evaluation_records = read_spatial_jsonl(eval_source, labeled_only=True)
        catalog_images = len(evaluation_records)
        empty_images = sum(not r["targets"]["target"]["boxes"] for r in evaluation_records)
    development = [q for r in development for q in
                   (grouped_queries(r, seed=seed) if r["meta"]["source"] == "coco2017" else [r])]
    queries = [q for r in evaluation_records for q in grouped_queries(r, seed=seed)]
    coco_train = [r for r in train if r["meta"]["source"] == "coco2017"]
    if (len(train) != 100000 or len({r["id"] for r in train}) != len(train)
            or len({r["group_id"] for r in coco_train}) != len(coco_train)):
        raise ValueError("Require 100k distinct records and exactly one prompt per COCO image")
    if len({r["id"] for r in development}) != len(development) or len({r["id"] for r in queries}) != len(queries):
        raise ValueError("Duplicate single-category development/evaluation queries")
    if {r["group_id"] for r in train} & {r["group_id"] for r in development}:
        raise ValueError("Development image leakage")
    if {hashes[record_image(r)] for r in train} & {dev_hashes[record_image(r)] for r in development}:
        raise ValueError("Development content leakage")
    dev_hashes = {record_image(r): dev_hashes[record_image(r)] for r in development}
    for relative, expected in dev_hashes.items():
        path = evaluation / relative
        if path.is_symlink() or path.stat().st_nlink != 1 or digest_file(path) != expected:
            raise ValueError("Development media must remain independent verified copies")
    saved = save_dataset(training, "train_" + REVISION, train, sources, hashes,
        protocol=PROTOCOL, seed=seed, max_categories=1, max_coco_prompts_per_image=1,
        source_counts=dict(Counter(r["meta"]["source"] for r in train)))
    staging = evaluation / "protocols" / REVISION
    dev = save_dataset(staging, "dev", development, dev_sources, dev_hashes,
        protocol=PROTOCOL, seed=seed, coco_images=dev_images,
        coco_queries=sum(r["meta"]["source"] == "coco2017" for r in development),
        referring_queries=sum(r["meta"]["source"] != "coco2017" for r in development),
        media_root="../..", purpose="development", validated=True, independent_media=True)
    saved_eval = save_dataset(staging, BENCHMARK, queries, ["COCO 2017"], eval_hashes,
        protocol=PROTOCOL, seed=seed, max_categories=1, media_root="../..", catalog_images=catalog_images,
        images_without_positive_queries=empty_images,
        annotations_sha256=digest_file(evaluation / "annotations/instances_val2017.json"),
        note="Ask each ground-truth present category separately, merge all image predictions; conditional COCO AP.")
    result = {"format": FORMAT, "validated": True, "protocol": PROTOCOL, "samples": len(train),
        "images": saved["images"], "seed": seed, "targets": saved["targets"], "source_counts": saved["source_counts"],
        "records_sha256": saved["records_sha256"], "development_sha256": dev["records_sha256"],
        "evaluation_sha256": saved_eval["records_sha256"], "source_seal_sha256": digest_file(seal_path),
        "max_categories": 1, "max_coco_prompts_per_image": 1, "coco_train_queries": len(coco_train),
        "coco_train_images": len(coco_train), "coco_images_with_second_prompt": 0,
        "coco_development_images": dev_images, "development_image_overlap": 0, "development_content_overlap": 0,
        "evaluation_queries": len(queries), "evaluation_catalog_images": catalog_images,
        "evaluation_images_without_queries": empty_images, "independent_media": True,
        "global_evaluation_overlap": seal["global_evaluation_overlap"]}
    write_json(training / "manifests" / ("ready_" + REVISION + ".json"), result)
    return result


def activate(root):
    """Promote staged lists only after all readers of the prior Det revision stop."""
    root = Path(root)
    training, evaluation = root / "det", root / "eval/det"
    current = training / "manifests/ready.json"
    previous = json.loads(current.read_text())
    if previous.get("format") == FORMAT:
        check_seal(root, previous)
        return
    ready = training / "manifests" / ("ready_" + REVISION + ".json")
    seal = json.loads(ready.read_text())
    source = training / "processed" / ("train_" + REVISION + ".jsonl")
    staging = evaluation / "protocols" / REVISION
    checks = ((source, "records_sha256"), (staging / "processed/dev.jsonl", "development_sha256"),
              (staging / "processed" / (BENCHMARK + ".jsonl"), "evaluation_sha256"))
    if (not seal.get("validated") or digest_file(current) != seal["source_seal_sha256"]
            or any(digest_file(path) != seal[key] for path, key in checks)):
        raise ValueError("Audit the staged revision and stop its legacy readers before activating it")
    write_json(training / "manifests/legacy_det_protocol.json", {"obsolete": True, "ready": previous,
        "replacement_protocol": PROTOCOL, "source_records_sha256": previous["records_sha256"]})
    destination = evaluation / "protocols/coco_prompt"
    destination.mkdir(exist_ok=True)
    for folder in ("processed", "manifests"):
        (destination / folder).mkdir(exist_ok=True)
        for path in (staging / folder).iterdir():
            path.replace(destination / folder / path.name)
        (staging / folder).rmdir()
    staging.rmdir()
    source.replace(training / "processed/train.jsonl")
    manifest = training / "manifests" / ("train_" + REVISION + ".json")
    value = json.loads(manifest.read_text())
    value["name"] = "train"
    write_json(training / "manifests/train.json", value)
    manifest.unlink()
    ready.replace(current)
    from scripts.data.finalize_spatial import preview
    preview(training, "det")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-images", type=int, default=512)
    parser.add_argument("--activate", action="store_true", help="Promote the revision after stopping prior Det readers")
    args = parser.parse_args()
    result = prepare(args.data_root, args.seed, args.dev_images)
    if args.activate:
        activate(args.data_root)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
