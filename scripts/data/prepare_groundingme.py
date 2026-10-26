"""Prepare the pinned, evaluation-only GroundingME test split for Det."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path

from PIL import Image

from scripts.data.prepare_spatial import digest_file, make_record, save_dataset, write_json
from valen.data.spatial import validate_spatial_record


from evaluation.det.groundingme import (
    BENCHMARK, DATASET_REVISION, PROMPT, PROTOCOL, SCORER_REVISION, SOURCE, SCORER, SOURCE_SHA256,
)


def prepare(source_root, data_root, expected_samples=1005):
    import hashlib
    import pyarrow.parquet as pq

    source_root, data_root = Path(source_root), Path(data_root)
    files = sorted(source_root.glob("test-*-of-*.parquet"))
    if not files:
        raise ValueError("Missing pinned GroundingME parquet files")
    output = data_root / "eval/det/groundingme"
    (output / "manifests/preflight.json").unlink(missing_ok=True)
    source_hashes = {p.name: digest_file(p) for p in files}
    if expected_samples == 1005 and source_hashes != {
            f"test-{i:05d}-of-00004.parquet": value for i, value in enumerate(SOURCE_SHA256)}:
        raise ValueError("GroundingME source files differ from the pinned official revision")
    rows, hashes, groups, subgroups = [], {}, Counter(), Counter()
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for sample in batch.to_pylist():
                raw = sample["image"]["bytes"]
                if not raw:
                    raise ValueError("GroundingME requires original embedded image bytes")
                digest = hashlib.sha256(raw).hexdigest()
                with Image.open(io.BytesIO(raw)) as image:
                    width, height = image.size
                    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[image.format]
                    image.verify()
                if (width, height) != (sample["width"], sample["height"]):
                    raise ValueError("GroundingME annotation and native image dimensions differ")
                relative = "images/" + digest + extension
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.is_symlink() or target.stat().st_nlink != 1 or digest_file(target) != digest:
                        raise ValueError("GroundingME media must be an independent, unmodified file")
                else:
                    target.write_bytes(raw)
                hashes[relative] = digest
                level = sample["subtask_l1"]
                if level not in {"Discriminative", "Spatial", "Limited", "Rejection"}:
                    raise ValueError("Unknown GroundingME dimension")
                groups[level] += 1
                if sample.get("subtask_l2"):
                    subgroups[level + "/" + sample["subtask_l2"]] += 1
                boxes = []
                if level != "Rejection":
                    bbox = sample["bbox"]
                    if not bbox or len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= width
                                                                        and 0 <= bbox[1] < bbox[3] <= height):
                        raise ValueError("Invalid original GroundingME pixel bbox")
                    boxes = [[v / (width if i % 2 == 0 else height) for i, v in enumerate(bbox)]]
                text = sample["description"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Missing original GroundingME description")
                row = make_record("det", "groundingme:" + str(sample["id"]), digest, relative,
                    PROMPT.format(description=text), ["target"], boxes, [0] if boxes else [],
                    source="groundingme", split="test", width=width, height=height,
                    referring=True, allow_empty_target=True, original_instruction=text,
                    subtask_l1=level, subtask_l2=sample.get("subtask_l2") or "")
                validate_spatial_record(row)
                rows.append(row)
    if len(rows) != expected_samples or set(groups) != {"Discriminative", "Spatial", "Limited", "Rejection"}:
        raise ValueError("Incomplete GroundingME test split")
    train_hashes = set()
    for task in ("det", "seg", "gui"):
        manifest = data_root / task / "manifests/train.json"
        if manifest.is_file():
            train_hashes.update(json.loads(manifest.read_text()).get("media_sha256", {}).values())
    overlap = len(set(hashes.values()) & train_hashes)
    if overlap:
        raise ValueError("GroundingME evaluation image bytes overlap training data")
    manifest = save_dataset(output, BENCHMARK, rows, [SOURCE, SCORER], hashes,
        source_revision=DATASET_REVISION, scorer_revision=SCORER_REVISION, protocol=PROTOCOL,
        purpose="final_evaluation_only", groups=dict(sorted(groups.items())),
        source_files_sha256=source_hashes, subgroups=dict(sorted(subgroups.items())),
        data_license="Original SA-1B and HR-Bench terms apply; research use only",
        coordinates="native pixel xyxy converted deterministically to normalized xyxy",
        prompt="official evaluate.py template; unchanged original description",
        rejection="same background-winner presence rule on every record; no threshold fitting",
        baseline_training_sha256_overlap=overlap, independent_media=True)
    write_json(output / "manifests/preflight.json", {"validated": True,
        "samples": len(rows), "records_sha256": manifest["records_sha256"],
        "source_revision": DATASET_REVISION, "scorer_revision": SCORER_REVISION,
        "independent_media": True, "baseline_training_sha256_overlap": overlap,
        "purpose": "final_evaluation_only", "protocol": PROTOCOL})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Pinned original parquet directory")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    prepare(args.source_root, args.data_root)


if __name__ == "__main__":
    main()
