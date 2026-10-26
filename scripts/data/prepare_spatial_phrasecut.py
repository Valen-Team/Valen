"""Prepare genuine PhraseCut masks and finish the independent 100k Seg mixture."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random

from PIL import Image

from scripts.data.download_spatial import download_file
from scripts.data.prepare_spatial import digest_file, make_record, save_dataset, source_rows, write_json


def prepare(args):
    base = Path(args.data_root)
    raw = base / "seg/raw/phrasecut"
    information = {item["image_id"]: item for item in json.loads((raw / "image_data_split.json").read_text())}

    def convert(row, split):
        item = information[row["image_id"]]
        width, height = item["width"], item["height"]
        polygons = [[coordinate for point in polygon for coordinate in point]
                    for instance in row["Polygons"] for polygon in instance if len(polygon) >= 2]
        if not polygons:
            raise ValueError("Empty original PhraseCut polygons")
        boxes = row["instance_boxes"]
        x1 = max(0, min(box[0] for box in boxes) / width)
        y1 = max(0, min(box[1] for box in boxes) / height)
        x2 = min(1, max(box[0] + box[2] for box in boxes) / width)
        y2 = min(1, max(box[1] + box[3] for box in boxes) / height)
        return make_record("seg", "phrasecut:" + row["task_id"], "vg:" + str(row["image_id"]),
            f"images/phrasecut/{row['image_id']}.jpg", row["phrase"], ["target"], [[x1, y1, x2, y2]], [0],
            {"size": [height, width], "polygons": polygons, "rasterization": "phrasecut"}, source="phrasecut", split=split,
            image_id=row["image_id"], coco_id=item.get("coco_id"), width=width, height=height, referring=True)

    def materialize(root, records):
        ids = {r["meta"]["image_id"] for r in records}
        hashes = {}
        failures = []

        def one(i):
            relative = f"images/phrasecut/{i}.jpg"
            destination = root / relative
            download_file({"url": information[i]["url"], "destination": str(destination)})
            with Image.open(destination) as image:
                if image.size != (information[i]["width"], information[i]["height"]):
                    raise ValueError("Original PhraseCut image size mismatch")
                image.verify()
            return relative, digest_file(destination)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one, i): i for i in ids}
            for future in as_completed(futures):
                try:
                    name, digest = future.result()
                    hashes[name] = digest
                except Exception as exc:
                    failures.append({"image_id": futures[future], "error": str(exc)})
                if (len(hashes) + len(failures)) % 500 == 0:
                    print(json.dumps({"images_ready": len(hashes), "total": len(ids), "failed": len(failures)}), flush=True)
        if failures:
            write_json(root / "manifests/phrasecut_download_failures.json", failures)
            raise ValueError(f"{len(failures)} PhraseCut images failed; rerun to resume")
        return hashes

    eval_root = base / "eval/seg"
    source = ["https://github.com/ChenyunWu/PhraseCutDataset"]
    if args.stage in {"eval", "all"}:
        evaluation = {split: [convert(row, split) for row in json.loads((raw / f"refer_{split}.json").read_text())]
                      for split in ("val", "miniv", "test")}
        hashes = materialize(eval_root, evaluation["val"] + evaluation["miniv"] + evaluation["test"])
        save_dataset(eval_root, "phrasecut_val_miniv", evaluation["val"] + evaluation["miniv"], source, hashes,
                     mask_rasterization="phrasecut_original")
        save_dataset(eval_root, "phrasecut_test", evaluation["test"], source, hashes,
                     mask_rasterization="phrasecut_original")
    if args.stage == "eval":
        return
    hashes = {}
    for name in ("phrasecut_val_miniv", "phrasecut_test"):
        hashes.update(json.loads((eval_root / "manifests" / (name + ".json")).read_text())["media_sha256"])
    excluded = set()
    for path in (base / "det/raw/ref").glob("*.json"):
        if not path.stem.endswith("_train"):
            excluded.update(int(row["id"]) for row in source_rows(path))
    coco = json.loads((base / "eval/det/annotations/instances_val2017.json").read_text())
    excluded.update(item["id"] for item in coco["images"])
    excluded.update(int(item["coco_id"]) for item in information.values()
                    if item["split"] != "train" and item.get("coco_id"))
    rows = json.loads((raw / "refer_train.json").read_text())
    excluded_train_images = set(args.exclude_train_image_id)
    eligible = [row for row in rows if not information[row["image_id"]].get("coco_id")
                or int(information[row["image_id"]]["coco_id"]) not in excluded]
    random.Random(args.seed).shuffle(eligible)
    eligible = [row for row in eligible if row["image_id"] not in excluded_train_images]
    records = [convert(row, "train") for row in eligible[:20000]]
    if len(records) != 20000:
        raise ValueError("Insufficient disjoint PhraseCut training phrases")
    train_hashes = materialize(base / "seg", records)
    if set(train_hashes.values()) & set(hashes.values()):
        raise ValueError("PhraseCut train/evaluation image overlap")
    referring = list(source_rows(base / "seg/processed/train_refcoco.jsonl"))
    existing = json.loads((base / "seg/manifests/train_refcoco.json").read_text())
    forbidden_hashes = set()
    for manifest in (base / "eval/seg/manifests").glob("*.json"):
        forbidden_hashes.update(json.loads(manifest.read_text()).get("media_sha256", {}).values())
    combined_hashes = dict(existing["media_sha256"], **train_hashes)
    if set(combined_hashes.values()) & forbidden_hashes:
        raise ValueError("Seg training/evaluation content overlap")
    records.extend(referring)
    random.Random(args.seed).shuffle(records)
    if len(records) != 100000:
        raise ValueError("Expected exactly 100k Seg training samples")
    save_dataset(base / "seg", "train", records, ["RefCOCO family", *source], combined_hashes,
                 seed=args.seed, evaluation_overlap=0, excluded_train_image_ids=sorted(excluded_train_images))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=("eval", "train", "all"), default="all")
    parser.add_argument("--exclude-train-image-id", type=int, action="append", default=[],
                        help="Exclude a verified invalid source image and sample the next genuine phrase")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
