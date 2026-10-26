"""Audit COCO geometry with perfect predictions and separate localization/classification."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

from evaluation.det.coco import predictions_to_coco, score_coco
from evaluation.spatial import load_spatial_model
from valen.data.spatial import image_path, read_spatial_jsonl
from valen.modeling.qwen.spatial import decode_spatial
from valen.training.checkpoint import file_sha256
from valen.training.distributed import close, initialize
from valen.training.spatial_loss import box_iou


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def coco_records(root):
    path = root / "protocols/coco_prompt/processed/coco2017_conditional_val2017.jsonl"
    conditional = path.is_file()
    if not conditional:
        path = root / "processed/coco2017_val2017.jsonl"
    return path, read_spatial_jsonl(path, labeled_only=True), conditional


def gold_audit(root, output):
    root = Path(root)
    annotations = root / "annotations/instances_val2017.json"
    truth = json.loads(annotations.read_text())
    images = {image["id"]: image for image in truth["images"]}
    categories = sorted(truth["categories"], key=lambda item: item["id"])
    ids, names = [c["id"] for c in categories], [c["name"] for c in categories]
    by_image = defaultdict(list)
    for row in truth["annotations"]:
        if not row.get("iscrowd"):
            by_image[row["image_id"]].append(row)
    records_path, records, conditional = coco_records(root)
    queried_categories = defaultdict(set)
    observed, predictions, error = set(), [], 0.0
    for record in records:
        meta, target = record["meta"], record["targets"]["target"]
        question = record["request"]["questions"]["target"]
        identifier = meta["image_id"]
        image = images[identifier]
        local_ids, local_names = meta["category_ids"], question["categories"]
        if conditional:
            if (len(local_ids) != 1 or len(set(local_ids)) != len(local_ids)
                    or any(category not in ids for category in local_ids)
                    or local_names != [names[ids.index(category)] for category in local_ids]
                    or queried_categories[identifier] & set(local_ids)):
                raise ValueError("Conditional COCO categories must be valid disjoint single-category queries")
            queried_categories[identifier].update(local_ids)
        elif identifier in observed or local_ids != ids or local_names != names:
            raise ValueError("Duplicate COCO image or incorrect full category mapping")
        observed.add(identifier)
        with Image.open(root / image_path(record)) as source:
            if source.size != (image["width"], image["height"]) or source.size != (meta["width"], meta["height"]):
                raise ValueError("Image and annotation dimensions differ")
        rows = [row for row in by_image[identifier] if not conditional or row["category_id"] in local_ids]
        if conditional and set(local_ids) != {row["category_id"] for row in rows}:
            raise ValueError("Conditional prompt includes a category absent from its image")
        if len(rows) != len(target["boxes"]):
            raise ValueError("Prepared COCO target count differs from original non-crowd annotations")
        boxes = []
        for box, label, row in zip(target["boxes"], target["labels"], rows):
            x, y, width, height = row["bbox"]
            expected = [x / image["width"], y / image["height"],
                        min(1, (x + width) / image["width"]), min(1, (y + height) / image["height"])]
            if local_ids[label] != row["category_id"] or max(abs(a - b) for a, b in zip(box, expected)) > 1e-9:
                raise ValueError("Prepared COCO labels/coordinates differ from original annotations")
            boxes.append({"bbox": box, "label": label, "score": 1.0})
        q = SimpleNamespace(original_size=(meta["height"], meta["width"]), metadata=meta, categories=local_names)
        converted = predictions_to_coco(q, {"boxes": boxes})
        for prediction, row in zip(converted, rows):
            error = max(error, max(abs(a - b) for a, b in zip(prediction["bbox"], row["bbox"])))
        predictions.extend(converted)
    expected_images = {identifier for identifier in images if by_image[identifier]} if conditional else set(images)
    if observed != expected_images:
        raise ValueError("COCO evaluation must include every original image, including empty images")
    if conditional:
        for identifier in observed:
            if queried_categories[identifier] != {row["category_id"] for row in by_image[identifier]}:
                raise ValueError("Conditional groups do not cover every original positive category")
        for identifier in set(images) - observed:
            with Image.open(root / "images/coco" / f"{identifier:012d}.jpg") as source:
                if source.size != (images[identifier]["width"], images[identifier]["height"]):
                    raise ValueError("Zero-query image is missing or misaligned")
    metrics = score_coco(annotations, predictions)
    report = {"images": len(images), "queried_images": len(observed), "prompts": len(records),
              "protocol": "known_category_conditional_detection" if conditional else "standard_coco_detection",
              "non_crowd_boxes": len(predictions),
              "normalized_xyxy_valid": True, "category_mapping_valid": True,
              "max_xywh_roundtrip_error_pixels": error, "perfect_prediction_metrics": metrics,
              "records_sha256": file_sha256(records_path)}
    save(Path(output) / "gold-audit.json", report)
    if metrics["AP"] < .999:
        raise ValueError("Perfect predictions failed the COCO scoring check; inspect gold-audit.json")
    return report


def preview(root, records, predictions, output):
    by_image = defaultdict(list)
    grouped_records = defaultdict(list)
    for record in records:
        grouped_records[record["meta"]["image_id"]].append(record)
    for row in predictions:
        by_image[row["image_id"]].append(row)
    sheet = Image.new("RGB", (1280, 720), "white")
    for index, image_records in enumerate(list(grouped_records.values())[:8]):
        record = image_records[0]
        with Image.open(root / image_path(record)) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for x1, y1, x2, y2 in [box for query in image_records for box in query["targets"]["target"]["boxes"]]:
            draw.rectangle((x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height),
                           outline="lime", width=2)
        for row in sorted(by_image[record["meta"]["image_id"]], key=lambda row: -row["score"])[:10]:
            x, y, width, height = row["bbox"]
            draw.rectangle((x, y, x + width, y + height), outline="red", width=2)
            draw.text((x, y), f"{row['category_id']}:{row['score']:.2f}", fill="red")
        image.thumbnail((316, 326))
        position = ((index % 4) * 320, (index // 4) * 360)
        sheet.paste(image, position)
        ImageDraw.Draw(sheet).text((position[0], position[1] + 330),
                                  f"COCO {record['meta']['image_id']}: GT green; top10 red", fill="black")
    sheet.save(output, quality=88)


@torch.inference_mode()
def prediction_audit(root, model_path, records, output, context):
    model, compiler = load_spatial_model(model_path, context.device, data_root=root)
    predictions, counts, confusion = [], Counter(), Counter()
    for record in records[context.rank::context.world_size]:
        question = compiler.compile(record).questions[0]
        result = model(question)
        prediction = decode_spatial(question, result, score_threshold=0.0)
        predictions.extend(predictions_to_coco(question, prediction))
        counts["queries"] += len(result.boxes)
        counts["background_wins"] += int((result.class_logits.argmax(-1) == len(question.categories)).sum())
        if not torch.isfinite(result.boxes).all() or (result.boxes < 0).any() or (result.boxes > 1).any():
            raise ValueError("Model produced non-finite or unnormalized coordinates")
        counts["invalid_geometry"] += int(((result.boxes[:, 2:] - result.boxes[:, :2]) <= 0).any(-1).sum())
        target = torch.tensor(question.target["boxes"], dtype=torch.float32).reshape(-1, 4)
        counts["targets"] += len(target)
        if not len(target) or not prediction["boxes"]:
            continue
        boxes = torch.tensor([item["bbox"] for item in prediction["boxes"]])
        labels = torch.tensor([item["label"] for item in prediction["boxes"]])
        truth_labels = torch.tensor(question.target["labels"])
        overlaps = box_iou(boxes, target)
        best, indices = overlaps.max(0)
        localized = best >= .5
        counts["localized_ignoring_class"] += int(localized.sum())
        counts["best_localized_correct_class"] += int((localized & (labels[indices] == truth_labels)).sum())
        counts["localized_with_correct_class"] += int((overlaps.masked_fill(
            labels[:, None] != truth_labels[None, :], 0).max(0).values >= .5).sum())
        for actual, guessed in zip(truth_labels[localized].tolist(), labels[indices][localized].tolist()):
            confusion[(question.categories[actual], question.categories[guessed])] += 1
    shards = context.gather({"predictions": predictions, "counts": dict(counts),
                             "confusion": [(list(key), value) for key, value in confusion.items()]})
    if context.primary:
        predictions, counts, confusion = [], Counter(), Counter()
        for shard in shards:
            predictions.extend(shard["predictions"])
            counts.update(shard["counts"])
            confusion.update({tuple(key): value for key, value in shard["confusion"]})
        annotations = root / "annotations/instances_val2017.json"
        ids = sorted({record["meta"]["image_id"] for record in records})
        report = {"model": Path(model_path).name, "images": len(ids), "prompts": len(records), "diagnostic_only": True,
                  "world_size": context.world_size, "counts": dict(counts),
                  "class_aware": score_coco(annotations, predictions, ids),
                  "class_agnostic": score_coco(annotations, predictions, ids, False),
                  "box_recall_at_05_ignoring_class": counts["localized_ignoring_class"] / max(1, counts["targets"]),
                  "box_recall_at_05_with_class": counts["localized_with_correct_class"] / max(1, counts["targets"]),
                  "category_accuracy_on_best_localized_boxes": counts["best_localized_correct_class"] /
                      max(1, counts["localized_ignoring_class"]),
                  "largest_confusions": [{"truth": key[0], "prediction": key[1], "count": value}
                      for key, value in confusion.most_common(20) if key[0] != key[1]]}
        save(output / Path(model_path).name / "diagnostics.json", report)
        preview(root, records, predictions, output / Path(model_path).name / "examples.jpg")
        print(json.dumps(report), flush=True)
    del model
    if str(context.device).startswith("cuda"):
        torch.cuda.empty_cache()
    context.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", action="append", default=[])
    parser.add_argument("--gold-only", action="store_true")
    parser.add_argument("--samples", type=int, default=512, help="Diagnostic sample count; zero uses all images")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root, output = Path(args.data_root), Path(args.output_dir)
    if args.gold_only:
        print(json.dumps(gold_audit(root, output)), flush=True)
        return
    if not args.model_path or args.samples < 0:
        parser.error("Specify --model-path and a nonnegative diagnostic sample count")
    _, records, _ = coco_records(root)
    if args.samples:
        identifiers = sorted({record["meta"]["image_id"] for record in records})
        selected = set(random.Random(args.seed).sample(identifiers, min(args.samples, len(identifiers))))
        records = [record for record in records if record["meta"]["image_id"] in selected]
    context = initialize(args.device)
    try:
        for model_path in args.model_path:
            prediction_audit(root, model_path, records, output, context)
        if context.primary:
            save(output / "prediction-audit-complete.json", {"models": args.model_path,
                 "images_per_model": len({record["meta"]["image_id"] for record in records}),
                 "prompts_per_model": len(records), "seed": args.seed, "diagnostic_only": True})
    finally:
        close()


if __name__ == "__main__":
    main()
