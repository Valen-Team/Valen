"""Materialize independent Det/Seg ReasonSeg splits using the official LISA rasterization.

Protocol: JIA-Lab-research/LISA@3cb2d4301f1af4691bd4f3938335ef06e76f155a,
utils/data_processing.py. Target=1, ignore=255; flag annotations are deprecated.
Rasterization is adapted under Apache-2.0; the upstream license is reproduced
verbatim in the repository LICENSE. Changes add input validation, independent
task copies, explicit ignore masks, bbox conversion and split manifests.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from scripts.data.prepare_spatial import digest_file, make_record, save_dataset, write_json
from valen.data.spatial import validate_spatial_record


REVISION = "3cb2d4301f1af4691bd4f3938335ef06e76f155a"
SOURCE = "https://github.com/JIA-Lab-research/LISA/tree/" + REVISION


def rasterize(annotation, size):
    """Draw large polygons first, then let smaller target/ignore regions overwrite them."""
    import cv2
    height, width = size
    polygons, areas = [], []
    for shape in annotation["shapes"]:
        if shape["label"].lower() == "flag":
            continue
        points = np.asarray(shape["points"])
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
            raise ValueError("Invalid original ReasonSeg polygon")
        points = points.astype(np.int32)[None]
        canvas = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(canvas, points, True, 1, 1)
        cv2.fillPoly(canvas, points, 1)
        polygons.append((points, 255 if "ignore" in shape["label"].lower() else 1))
        areas.append(canvas.sum())
    result = np.zeros((height, width), dtype=np.uint8)
    for index in np.argsort(areas)[::-1]:
        points, value = polygons[index]
        cv2.polylines(result, points, True, value, 1)
        cv2.fillPoly(result, points, value)
    return result


def encode(mask):
    from pycocotools import mask as mask_api
    result = mask_api.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": result["size"], "counts": result["counts"].decode("ascii")}


def read_annotation(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="cp1252"))


def independent_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or target.stat().st_nlink != 1 or digest_file(source) != digest_file(target):
            raise ValueError("Existing ReasonSeg media is linked or differs from its source")
    else:
        shutil.copyfile(source, target)


def prepare(source_root, data_root, splits=("train", "val", "test")):
    source_root, data_root = Path(source_root), Path(data_root)
    manifests = {}
    for split in splits:
        directory = next((path for path in (source_root / split, source_root / "ReasonSeg" / split)
                          if path.is_dir()), None)
        if directory is None:
            raise ValueError(f"Missing official ReasonSeg split directory: {split}")
        annotations = sorted(directory.glob("*.json"))
        if not annotations:
            raise ValueError("ReasonSeg split contains no paired annotation JSON files")
        rows = {task: [] for task in ("det", "seg")}
        hashes, annotation_hashes, ignored_pixels = {}, {}, 0
        for annotation_path in annotations:
            candidates = [annotation_path.with_suffix(extension) for extension in (".jpg", ".jpeg", ".png")
                          if annotation_path.with_suffix(extension).is_file()]
            if len(candidates) != 1:
                raise ValueError("ReasonSeg annotations require exactly one original image")
            image_path = candidates[0]
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
            annotation = read_annotation(annotation_path)
            if annotation.get("imageWidth", width) != width or annotation.get("imageHeight", height) != height:
                raise ValueError("ReasonSeg image and annotation dimensions differ")
            mask = rasterize(annotation, (height, width))
            y, x = np.where(mask == 1)
            boxes = [[int(x.min()) / width, int(y.min()) / height,
                      (int(x.max()) + 1) / width, (int(y.max()) + 1) / height]] if len(x) else []
            target, ignored = encode(mask == 1), encode(mask == 255) if (mask == 255).any() else None
            ignored_pixels += int((mask == 255).sum())
            digest = digest_file(image_path)
            relative = f"images/reasonseg/{split}/{digest[:24]}{image_path.suffix.lower()}"
            hashes[relative] = digest
            annotation_hashes[annotation_path.name] = digest_file(annotation_path)
            texts = annotation["text"]
            if not isinstance(texts, list) or not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
                raise ValueError("ReasonSeg text must be a nonempty list of original instructions")
            texts = list(dict.fromkeys(text.strip() for text in texts)) if split == "train" else [texts[0].strip()]
            for task in ("det", "seg"):
                destination = data_root / task if split == "train" else data_root / "eval" / task / "reasonseg"
                independent_copy(image_path, destination / relative)
                for index, text in enumerate(texts):
                    instruction = text if annotation["is_sentence"] else f"What is {text} in this image?"
                    instruction += " Please output segmentation mask." if task == "seg" else " Please locate it with a bounding box."
                    record = make_record(task, f"reasonseg:{split}:{annotation_path.stem}:{index}",
                        "reasonseg:" + digest, relative, instruction, ["target"], boxes, [0] if boxes else [],
                        target if task == "seg" else None, source="reasonseg", split=split,
                        width=width, height=height, referring=True, allow_empty_target=True,
                        is_sentence=bool(annotation["is_sentence"]),
                        query_index=index, original_instruction=text, annotation_sha256=annotation_hashes[annotation_path.name],
                        bbox_source="tight_union_of_valid_target_pixels")
                    if task == "seg" and ignored is not None:
                        record["targets"]["target"]["ignore_mask"] = ignored
                    validate_spatial_record(record)
                    rows[task].append(record)
        for task in ("det", "seg"):
            destination = data_root / task if split == "train" else data_root / "eval" / task / "reasonseg"
            manifest = save_dataset(destination, "reasonseg_" + split, rows[task], [SOURCE], hashes,
                source_revision=REVISION, split=split, source_images=len(annotations),
                instruction_protocol="all_original_train_paraphrases" if split == "train" else "first_original_instruction",
                ignored_pixels=ignored_pixels, annotation_sha256=annotation_hashes,
                purpose="training" if split == "train" else "development" if split == "val" else "final_evaluation",
                metric="bbox_Acc@0.5_mask_derived" if task == "det" else "LISA_gIoU_cIoU_ignore255")
            manifests[(task, split)] = manifest
    baseline_hashes, existing_eval_hashes = set(), set()
    for task in ("det", "seg", "gui"):
        baseline = data_root / task / "manifests/train.json"
        if baseline.is_file():
            baseline_hashes.update(json.loads(baseline.read_text()).get("media_sha256", {}).values())
        for path in (data_root / "eval" / task / "manifests").glob("*.json"):
            existing_eval_hashes.update(json.loads(path.read_text()).get("media_sha256", {}).values())
    for task in ("det", "seg"):
        split_hashes = {split: set(manifest["media_sha256"].values())
                        for (kind, split), manifest in manifests.items() if kind == task}
        for first in split_hashes:
            for second in split_hashes:
                if first != second and split_hashes[first] & split_hashes[second]:
                    raise ValueError("Original ReasonSeg splits contain overlapping image content")
        overlap = {split: len(values & baseline_hashes) for split, values in split_hashes.items() if split != "train"}
        report = {"validated": True, "source": SOURCE, "source_revision": REVISION,
                  "splits": {split: {"samples": manifest["samples"], "images": manifest["images"],
                                      "records_sha256": manifest["records_sha256"]}
                             for (kind, split), manifest in manifests.items() if kind == task},
                  "independent_media": True, "baseline_training_sha256_overlap": overlap,
                  "training_overlap_with_existing_evaluation_sha256": len(split_hashes.get("train", set()) & existing_eval_hashes),
                  "training_manifest": "reasonseg_train", "evaluation_instructions": "first_original_instruction",
                  "data_license": "Original source terms apply; repository code license is not a dataset license"}
        write_json(data_root / task / "manifests/reasonseg.json", report)
        write_json(data_root / "eval" / task / "reasonseg/manifests/preflight.json", report)
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Extracted original ReasonSeg directory")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    args = parser.parse_args()
    prepare(args.source_root, args.data_root, args.splits)


if __name__ == "__main__":
    main()
