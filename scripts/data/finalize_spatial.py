"""Audit independent spatial corpora and seal their immutable training inputs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import itertools

from PIL import Image

from scripts.data.prepare_spatial import digest_file, record_image, save_dataset, source_rows, write_json
from valen.data.spatial import decode_mask, validate_spatial_record


def benchmarks(task):
    ref = {f"{name}_{split}" for name in ("refcoco", "refcocoplus") for split in ("val", "testA", "testB")}
    ref.update({"refcocog_val", "refcocog_test"})
    return {"det": ref | {"coco2017_val2017"}, "seg": ref | {"phrasecut_val_miniv", "phrasecut_test"},
            "gui": {"screenspot_test", "screenspot_v2_test", "screenspot_pro_test"}}[task]


def read_manifest(root, name):
    value = json.loads((root / "manifests" / (name + ".json")).read_text())
    if digest_file(root / "processed" / (name + ".jsonl")) != value["records_sha256"]:
        raise ValueError(f"Dataset changed after preparation: {name}")
    return value


def audit(root, name, task, checked_media, forbidden=(), *, allow_empty_instruction=False):
    manifest = read_manifest(root, name)
    ids, groups, counts = set(), set(), Counter()
    media = manifest["media_sha256"]
    if set(media.values()) & set(forbidden):
        raise ValueError(f"Training/evaluation image overlap: {task}/{name}")
    for relative, expected in media.items():
        path = root / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Media escapes its independent dataset directory")
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError("Dataset media must be independently stored regular files")
        if path not in checked_media:
            with Image.open(path) as image:
                size = image.size
                image.verify()
            if digest_file(path) != expected:
                raise ValueError("Media hash mismatch")
            checked_media[path] = (size, expected)
        elif checked_media[path][1] != expected:
            raise ValueError("Inconsistent media hashes")
    for record in source_rows(root / "processed" / (name + ".jsonl")):
        validate_spatial_record(record, allow_empty_instruction=allow_empty_instruction)
        if record["id"] in ids:
            raise ValueError("Duplicate record ID")
        ids.add(record["id"])
        groups.add(record["group_id"])
        question = next(iter(record["request"]["questions"].values()))
        if question["type"] != task or record_image(record) not in media:
            raise ValueError("Wrong task or unregistered media")
        size = checked_media[root / record_image(record)][0]
        metadata = record["meta"]
        if metadata.get("source") == "phrasecut" and metadata.get("coco_id"):
            groups.add("coco:" + str(int(metadata["coco_id"])))
        if size != (metadata["width"], metadata["height"]):
            raise ValueError("Original image dimensions do not match annotation")
        if task == "seg":
            if metadata["source"] == "phrasecut" and record["targets"]["target"]["mask"].get("rasterization") != "phrasecut":
                raise ValueError("PhraseCut requires its original polygon rasterization protocol")
            mask = decode_mask(record["targets"]["target"]["mask"], root)
            if tuple(mask.shape) != (size[1], size[0]) or not mask.any():
                raise ValueError("Invalid original-resolution mask")
        counts[metadata["source"]] += 1
    if len(ids) != manifest["samples"]:
        raise ValueError("Manifest record count mismatch")
    return manifest, counts, groups


def development(root, task, seed):
    if task == "gui":
        return "gui_dev"
    names = ["refcoco_val", "refcocoplus_val", "refcocog_val"]
    if task == "seg":
        names.append("phrasecut_val_miniv")
    rng = random.Random(seed)
    records, hashes = [], {}
    for index, name in enumerate(names):
        rows = list(source_rows(root / "processed" / (name + ".jsonl")))
        rng.shuffle(rows)
        records.extend(rows[:512 // len(names) + (index < 512 % len(names))])
        hashes.update(read_manifest(root, name)["media_sha256"])
    save_dataset(root, "dev", records, names, hashes, seed=seed, purpose="development")
    return "dev"


def preview(root, task):
    from PIL import ImageDraw
    root = Path(root)
    data_sha256 = digest_file(root / "processed/train.jsonl")
    image_path, manifest_path = root / "manifests/samples.jpg", root / "manifests/samples.json"
    if image_path.is_file() and manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text())
        except ValueError:
            previous = {}
        if (previous.get("format") == "valen_annotation_preview_v1" and previous.get("records_sha256") == data_sha256
                and previous.get("preview_sha256") == digest_file(image_path)):
            return previous
    selected = list(itertools.islice(source_rows(root / "processed/train.jsonl"), 4))
    sheet = Image.new("RGB", (960, 280), "white")
    for index, record in enumerate(selected):
        with Image.open(root / record_image(record)) as source:
            image = source.convert("RGB")
        target = record["targets"]["target"]
        if task == "seg":
            import numpy as np
            mask = decode_mask(target["mask"], root).numpy().astype("uint8") * 110
            image.paste(Image.new("RGB", image.size, "lime"), mask=Image.fromarray(mask))
        draw = ImageDraw.Draw(image)
        for x1, y1, x2, y2 in target["boxes"]:
            draw.rectangle((x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height),
                           outline="red", width=3)
        image.thumbnail((236, 250))
        sheet.paste(image, (index * 240, 0))
        ImageDraw.Draw(sheet).text((index * 240, 255), record["id"][:30], fill="black")
    sheet.save(image_path, quality=85)
    manifest = {"format": "valen_annotation_preview_v1", "task": task, "records_sha256": data_sha256,
                "preview_sha256": digest_file(image_path), "selection": "first_four_training_records",
                "samples": [{"id": row["id"], "image": record_image(row), "source": row["meta"]["source"],
                    "instruction": row["request"]["questions"]["target"]["instructions"],
                    "categories": row["request"]["questions"]["target"].get("categories", ["target"]),
                    "boxes": row["targets"]["target"]["boxes"], "labels": row["targets"]["target"]["labels"]}
                    for row in selected]}
    write_json(manifest_path, manifest)
    return manifest


def finalize(base, task, seed=42):
    base = Path(base)
    checked_media, global_eval_hashes, global_eval_groups, evaluations = {}, set(), set(), {}
    # Seal every official suite before any formal training can start.
    for kind in ("det", "seg", "gui"):
        eval_root = base / "eval" / kind
        for name in sorted(benchmarks(kind)):
            manifest, _, groups = audit(eval_root, name, kind, checked_media)
            global_eval_hashes.update(manifest["media_sha256"].values())
            global_eval_groups.update(groups)
            evaluations[f"{kind}/{name}"] = manifest["records_sha256"]
    root, eval_root = base / task, base / "eval" / task
    provenance = root / "manifests/sources.json"
    if not provenance.is_file():
        raise ValueError("Record pinned source identities and checksums before sealing data")
    dev_name = development(eval_root, task, seed)
    dev, _, dev_groups = audit(eval_root, dev_name, task, checked_media)
    global_eval_hashes.update(dev["media_sha256"].values())
    train, counts, train_groups = audit(root, "train", task, checked_media, global_eval_hashes)
    if train["samples"] != 100000 or train_groups & (dev_groups | global_eval_groups):
        raise ValueError("Expected 100k training records disjoint from development")
    result = {"format": "valen_spatial_data_v1", "validated": True,
              "time_utc": datetime.now(timezone.utc).isoformat(), "task": task,
              "samples": train["samples"], "images": train["images"], "targets": train["targets"],
              "source_counts": dict(counts), "records_sha256": train["records_sha256"],
              "development_sha256": dev["records_sha256"], "evaluation_sha256": evaluations,
              "global_evaluation_overlap": 0, "independent_media": True,
              "sources_sha256": digest_file(provenance),
              "media_bytes": sum((root / name).stat().st_size for name in train["media_sha256"])}
    preview(root, task)
    write_json(root / "manifests/ready.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--task", choices=("det", "seg", "gui"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    finalize(args.data_root, args.task, args.seed)


if __name__ == "__main__":
    main()
