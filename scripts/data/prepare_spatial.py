"""Prepare independent spatial datasets from pinned, materialized source files."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import zipfile

from PIL import Image

from scripts.data.download_spatial import download_file


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def source_rows(path):
    with Path(path).open() as stream:
        first = stream.read(1)
        stream.seek(0)
        if first == "[":
            yield from json.load(stream)
        else:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def make_record(task, uid, group_id, image, instruction, categories, boxes, labels, mask=None, **metadata):
    target = {"boxes": boxes, "labels": labels}
    if mask is not None:
        target["mask"] = mask
    return {"id": uid, "group_id": group_id,
            "request": {"state": {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image}}]}]},
                "questions": {"target": {"type": task, "instructions": instruction, "categories": categories}}},
            "targets": {"target": target}, "meta": metadata}


def record_image(record):
    return record["request"]["state"]["messages"][0]["content"][0]["image_url"]["url"]


def image_hashes(root, records):
    paths = sorted({record_image(record) for record in records})
    result = {}
    for name in paths:
        path = root / name
        with Image.open(path) as image:
            image.verify()
        result[name] = digest_file(path)
    return result


def save_dataset(root, name, records, sources, hashes=None, **extra):
    if not records:
        raise ValueError(f"Empty prepared dataset: {name}")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Duplicate spatial record IDs")
    hashes = hashes or image_hashes(root, records)
    path = root / "processed" / (name + ".jsonl")
    write_records(path, records)
    manifest = {"name": name, "samples": len(records), "images": len({record_image(r) for r in records}),
                "targets": sum(len(r["targets"]["target"]["boxes"]) for r in records),
                "sources": sources, "records_sha256": digest_file(path),
                "media_sha256": {name: hashes[name] for name in sorted({record_image(r) for r in records})}, **extra}
    write_json(root / "manifests" / (name + ".json"), manifest)
    print(json.dumps({"dataset": str(path), "samples": manifest["samples"], "images": manifest["images"]}), flush=True)
    return manifest


def coco_image_name(image_id):
    return f"images/coco/{int(image_id):012d}.jpg"


def materialize_coco(root, image_sources, preferred=(), workers=32):
    def one(item):
        image_id, source_name = item
        relative = coco_image_name(image_id)
        destination = root / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            existing = next((Path(p) / relative for p in preferred if (Path(p) / relative).is_file()), None)
            if existing:
                shutil.copyfile(existing, destination)
            else:
                download_file({"url": "https://s3.amazonaws.com/images.cocodataset.org/" + source_name,
                               "destination": str(destination)})
        with Image.open(destination) as image:
            image.verify()
        return relative, digest_file(destination)
    hashes = {}
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(one, item): item[0] for item in image_sources.items()}
        for future in as_completed(pending):
            try:
                name, digest = future.result()
                hashes[name] = digest
            except Exception as exc:
                failures.append({"image_id": pending[future], "error": str(exc)})
            if (len(hashes) + len(failures)) % 1000 == 0:
                print(json.dumps({"root": str(root), "images_ready": len(hashes), "total": len(pending),
                                  "failed": len(failures)}), flush=True)
    if failures:
        write_json(root / "manifests" / "download_failures.json", failures)
        raise RuntimeError(f"{len(failures)} image downloads failed; rerun preparation")
    return hashes


def ref_record(row, source, split, index, task):
    objects = row["objects"]
    if len(objects) != 1 or "rle" not in objects[0]:
        raise ValueError("Expected one genuine RefCOCO object and original RLE")
    item = objects[0]
    image_id = int(row["id"])
    h, w = item["rle"]["size"]
    return make_record(task, f"{source}:{split}:{index}", f"coco:{image_id}", coco_image_name(image_id),
                       item["label"], ["target"], [item["bbox"]], [0],
                       item["rle"] if task == "seg" else None, source=source, split=split,
                       image_id=image_id, height=h, width=w, referring=True)


def reservoir(rows, count, rng):
    selected = []
    for index, row in enumerate(rows):
        if index < count:
            selected.append(row)
        else:
            replacement = rng.randrange(index + 1)
            if replacement < count:
                selected[replacement] = row
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} valid samples are available; expected {count}")
    return selected


def prepare_natural(args):
    root = Path(args.data_root)
    raw = root / "det/raw/ref"
    phrase_root = root / "seg/raw/phrasecut"
    info = json.loads((phrase_root / "image_data_split.json").read_text())
    phrase_eval = {int(item["coco_id"]) for item in info if item["split"] != "train" and item.get("coco_id")}
    source_files = list(raw.glob("*.json"))
    if len(source_files) != 11:
        raise ValueError("Download all eleven RefCOCO source files before preparation")
    annotation_zip = root / "det/raw/annotations_trainval2017.zip"
    with zipfile.ZipFile(annotation_zip) as archive:
        validation = json.loads(archive.read("annotations/instances_val2017.json"))
        training = json.loads(archive.read("annotations/instances_train2017.json"))
    excluded = set(phrase_eval) | {item["id"] for item in validation["images"]}
    for path in source_files:
        if not path.stem.endswith("_train"):
            excluded.update(int(row["id"]) for row in source_rows(path))
    categories = sorted(training["categories"], key=lambda item: item["id"])
    names = [item["name"] for item in categories]
    label_map = {item["id"]: index for index, item in enumerate(categories)}
    by_image = defaultdict(list)
    for item in validation["annotations"]:
        if not item.get("iscrowd"):
            by_image[item["image_id"]].append(item)
    eval_det = root / "eval/det"
    eval_seg = root / "eval/seg"
    eval_sources = {item["id"]: "val2017/" + item["file_name"] for item in validation["images"]}
    coco_records = []
    for item in validation["images"]:
        annotations = by_image[item["id"]]
        boxes = [[a["bbox"][0] / item["width"], a["bbox"][1] / item["height"],
                  min(1, (a["bbox"][0] + a["bbox"][2]) / item["width"]),
                  min(1, (a["bbox"][1] + a["bbox"][3]) / item["height"])] for a in annotations]
        coco_records.append(make_record("det", f"coco2017:val:{item['id']}", f"coco:{item['id']}",
            coco_image_name(item["id"]), "Detect all objects in the listed categories.", names, boxes,
            [label_map[a["category_id"]] for a in annotations], source="coco2017", split="val2017",
            image_id=item["id"], height=item["height"], width=item["width"], category_ids=[c["id"] for c in categories]))
    ref_evaluations = {}
    for source in ("refcoco", "refcoco+", "refcocog"):
        for split in (("val", "test") if source == "refcocog" else ("val", "testA", "testB")):
            records = [ref_record(row, source, split, i, "det")
                       for i, row in enumerate(source_rows(raw / f"{source}_{split}.json"))]
            ref_evaluations[source.replace("+", "plus") + "_" + split] = records
            for record in records:
                i = record["meta"]["image_id"]
                eval_sources[i] = f"train2014/COCO_train2014_{i:012d}.jpg"
    eval_hashes = materialize_coco(eval_det, eval_sources, workers=args.workers)
    save_dataset(eval_det, "coco2017_val2017", coco_records, ["COCO 2017"], eval_hashes)
    write_json(eval_det / "annotations/instances_val2017.json", validation)
    for name, records in ref_evaluations.items():
        save_dataset(eval_det, name, records, ["PaDT-MLLM/RefCOCO@63c2c228081b35e6023f903e1adbebbacc09d0a2"], eval_hashes)
    ref_eval_ids = {r["meta"]["image_id"] for records in ref_evaluations.values() for r in records}
    seg_eval_sources = {i: url for i, url in eval_sources.items() if i in ref_eval_ids}
    seg_hashes = materialize_coco(eval_seg, seg_eval_sources, preferred=[eval_det], workers=args.workers)
    for name in ref_evaluations:
        source, split = name.rsplit("_", 1)
        source = source.replace("plus", "+")
        records = [ref_record(row, source, split, i, "seg") for i, row in enumerate(source_rows(raw / f"{source}_{split}.json"))]
        save_dataset(eval_seg, name, records, ["PaDT-MLLM/RefCOCO@63c2c228081b35e6023f903e1adbebbacc09d0a2"], seg_hashes)
    for task, quotas in (("det", (20000, 20000, 20000)), ("seg", (30000, 30000, 20000))):
        selected, image_sources = [], {}
        selected_keys = set()
        for source, count in zip(("refcoco", "refcoco+", "refcocog"), quotas):
            def eligible():
                seen = set(selected_keys)
                for i, row in enumerate(source_rows(raw / f"{source}_train.json")):
                    if int(row["id"]) not in excluded:
                        key = (int(row["id"]), row["objects"][0]["label"].strip(),
                               tuple(round(v, 6) for v in row["objects"][0]["bbox"]))
                        if key not in seen:
                            seen.add(key)
                            yield ref_record(row, source, "train", i, task)
            chosen = reservoir(eligible(), count, random.Random(args.seed))
            selected.extend(chosen)
            for record in chosen:
                selected_keys.add((record["meta"]["image_id"], record["request"]["questions"]["target"]["instructions"].strip(),
                                   tuple(round(v, 6) for v in record["targets"]["target"]["boxes"][0])))
        for record in selected:
            i = record["meta"]["image_id"]
            image_sources[i] = f"train2014/COCO_train2014_{i:012d}.jpg"
        if task == "det":
            annotations = defaultdict(list)
            for item in training["annotations"]:
                if not item.get("iscrowd") and item["bbox"][2] > 0 and item["bbox"][3] > 0:
                    annotations[item["image_id"]].append(item)
            images = reservoir((item for item in training["images"] if item["id"] not in excluded
                                and 0 < len(annotations[item["id"]]) <= 100), 40000, random.Random(args.seed))
            for item in images:
                boxes, labels = [], []
                for annotation in annotations[item["id"]]:
                    x, y, w, h = annotation["bbox"]
                    box = [max(0, x / item["width"]), max(0, y / item["height"]),
                           min(1, (x + w) / item["width"]), min(1, (y + h) / item["height"])]
                    if box[0] < box[2] and box[1] < box[3]:
                        boxes.append(box)
                        labels.append(label_map[annotation["category_id"]])
                selected.append(make_record("det", f"coco2017:train:{item['id']}", f"coco:{item['id']}",
                    coco_image_name(item["id"]), "Detect all objects in the listed categories.", names, boxes, labels,
                    source="coco2017", split="train2017", image_id=item["id"],
                    height=item["height"], width=item["width"], category_ids=[c["id"] for c in categories]))
                image_sources.setdefault(item["id"], "train2017/" + item["file_name"])
        else:
            # PhraseCut preparation appends its disjoint 20k samples before the ready marker.
            pass
        hashes = materialize_coco(root / task, image_sources, preferred=[root / "det"] if task == "seg" else [], workers=args.workers)
        forbidden_hashes = set(eval_hashes.values()) | set(seg_hashes.values())
        if set(hashes.values()) & forbidden_hashes:
            raise ValueError("Train/evaluation image content overlap; adjust sampling before training")
        random.Random(args.seed).shuffle(selected)
        save_dataset(root / task, "train_refcoco" if task == "seg" else "train", selected,
                     ["COCO 2017", "PaDT-MLLM/RefCOCO"], hashes, seed=args.seed,
                     excluded_image_ids=len(excluded), source_counts=dict(Counter(r["meta"]["source"] for r in selected)))
    write_json(root / "det/manifests/natural_preparation_complete.json", {"det_samples": 100000, "seg_referring_samples": 80000})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", help="Optional local GUI source collection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--task", required=True, choices=("natural",))
    args = parser.parse_args()
    prepare_natural(args)


if __name__ == "__main__":
    main()
