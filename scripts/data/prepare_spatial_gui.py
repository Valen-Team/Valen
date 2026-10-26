"""Prepare ScreenSpot benchmarks and an independent 100k GUI bbox mixture."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import zipfile

from PIL import Image

from scripts.data.prepare_spatial import make_record, record_image, save_dataset, write_json


def parquet_rows(paths):
    import pyarrow.parquet as pq
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            yield from batch.to_pylist()


def store_image(root, data, source):
    digest = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as image:
        size = image.size
        suffix = ".jpg" if image.format == "JPEG" else ".png"
        image.verify()
    name = f"images/{digest[:2]}/{digest}{suffix}"
    path = root / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return name, digest, size


def normalized(box, size, pixel_xywh=False, pixel_xyxy=False):
    x1, y1, x2, y2 = map(float, box)
    if pixel_xywh:
        x2, y2 = x1 + x2, y1 + y2
    if pixel_xywh or pixel_xyxy:
        x1, x2 = x1 / size[0], x2 / size[0]
        y1, y2 = y1 / size[1], y2 / size[1]
    result = [max(0., x1), max(0., y1), min(1., x2), min(1., y2)]
    if not (result[0] < result[2] and result[1] < result[3]):
        raise ValueError("Invalid GUI box")
    return result


def platform_group(value):
    value = str(value).lower()
    if value in {"ios", "android", "mobile"}:
        return "mobile"
    if value in {"macos", "windows", "linux", "desktop"}:
        return "desktop"
    return "web" if value in {"web", "gitlab", "shop", "forum", "tool"} else value


def evaluation(args):
    root = Path(args.data_root) / "eval/gui"
    source_root = Path(args.source_root)
    hashes, records = {}, []
    paths = sorted((root / "raw/screenspot/data").glob("*.parquet"))
    if len(paths) != 3:
        raise ValueError("Download all three ScreenSpot parquet files first")
    for index, row in enumerate(parquet_rows(paths)):
        name, digest, size = store_image(root, row["image"]["bytes"], "screenspot")
        hashes[name] = digest
        box = normalized(row["bbox"], size)
        records.append(make_record("gui", f"screenspot:{index}", digest, name, row["instruction"],
            ["target"], [box], [0], source="screenspot", split="test", width=size[0], height=size[1],
            ui_type=row.get("data_type", "unknown"),
            platform=platform_group(row.get("data_source", row.get("data_souce", "unknown")))))
    save_dataset(root, "screenspot_test", records, ["rootsautomation/ScreenSpot@0be08781e2e188582f6131625ae1598d443b4d5d"], hashes)
    v2 = source_root / "OS-Copilot--ScreenSpot-v2"
    records = []
    with zipfile.ZipFile(v2 / "screenspotv2_image.zip") as archive:
        entries = {Path(item.filename).name: item for item in archive.infolist() if not item.is_dir()}
        for platform in ("mobile", "desktop", "web"):
            rows = json.loads((v2 / f"screenspot_{platform}_v2.json").read_text())
            for index, row in enumerate(rows):
                data = archive.read(entries[Path(row["img_filename"]).name])
                name, digest, size = store_image(root, data, "screenspot_v2")
                hashes[name] = digest
                records.append(make_record("gui", f"screenspot_v2:{platform}:{index}", digest, name,
                    row["instruction"], ["target"], [normalized(row["bbox"], size, pixel_xywh=True)], [0],
                    source="screenspot_v2", split="test", width=size[0], height=size[1],
                    platform=platform, ui_type=row.get("data_type", "unknown")))
    save_dataset(root, "screenspot_v2_test", records, ["OS-Copilot/ScreenSpot-v2"], hashes)
    records = []
    pro = root / "raw/screenspot-pro"
    annotations = sorted((pro / "annotations").glob("*.json"))
    if len(annotations) != 26:
        raise ValueError("Incomplete ScreenSpot-Pro annotation collection")
    for path in annotations:
        for row in json.loads(path.read_text()):
            data = (pro / "images" / row["img_filename"]).read_bytes()
            name, digest, size = store_image(root, data, "screenspot_pro")
            hashes[name] = digest
            records.append(make_record("gui", "screenspot_pro:" + row["id"], digest, name, row["instruction"],
                ["target"], [normalized(row["bbox"], size, pixel_xyxy=True)], [0],
                source="screenspot_pro", split="test", width=size[0], height=size[1],
                platform=row["platform"], ui_type=row["ui_type"], application=row["application"], group=row["group"]))
    save_dataset(root, "screenspot_pro_test", records, ["likaixin/ScreenSpot-Pro@210e78d3844251110bff86c95835ebd37a6930fa"],
                 hashes, language="en", gt_type="positive", instruction_style="instruction")
    write_json(root / "manifests/evaluation_exclusion.json", {"sha256": sorted(set(hashes.values()))})


def training(args):
    base = Path(args.data_root)
    root = base / "gui"
    source_root = Path(args.source_root)
    forbidden = set(json.loads((base / "eval/gui/manifests/evaluation_exclusion.json").read_text())["sha256"])
    candidates, hashes, per_image = [], {}, Counter()
    seen = set()
    rng = random.Random(args.seed)

    def add(data, source, elements, platform):
        digest = hashlib.sha256(data).hexdigest()
        if digest in forbidden:
            return
        # Hold out whole screenshots, consistently across all source datasets.
        split = "dev" if int(digest[:8], 16) % 20 == 0 else "train"
        with Image.open(io.BytesIO(data)) as image:
            size = image.size
        eligible = list(elements)
        rng.shuffle(eligible)
        selected = []
        for item in eligible:
            if per_image[digest] >= 12:
                break
            instruction = item.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip() or "bbox" not in item:
                continue
            try:
                box = normalized(item["bbox"], size)
            except (ValueError, TypeError):
                continue
            key = (digest, instruction.strip(), tuple(round(x, 6) for x in box))
            if key in seen:
                continue
            seen.add(key)
            per_image[digest] += 1
            selected.append((instruction.strip(), box, item.get("data_type", "unknown")))
        if not selected:
            return
        name, digest, size = store_image(root, data, source)
        hashes[name] = digest
        for instruction, box, ui_type in selected:
            uid = hashlib.sha256((digest + instruction + repr(box)).encode()).hexdigest()
            candidates.append(make_record("gui", source + ":" + uid, digest, name, instruction,
                ["target"], [box], [0], source=source, split=split, platform=platform, ui_type=ui_type,
                width=size[0], height=size[1]))

    desktop_paths = sorted((source_root / "showlab--ShowUI-desktop/data").glob("*.parquet"))
    for row in parquet_rows(desktop_paths):
        add(row["image"]["bytes"], "showui_desktop", [row], "desktop")
    print(json.dumps({"source": "showui_desktop", "candidates": len(candidates)}), flush=True)
    web_paths = sorted((root / "raw/seeclick").glob("*.parquet"))
    if len(web_paths) != 4:
        raise ValueError("Download the four pinned SeeClick parquet source shards first")
    for row in parquet_rows(web_paths):
        add(row["image"]["bytes"], "seeclick_web", row["elements"], "web")
    print(json.dumps({"source": "seeclick_web", "candidates": len(candidates)}), flush=True)
    mobile = source_root / "OS-Copilot--OS-Atlas-data/mobile_domain"
    rows = json.loads((mobile / "aw_mobile.json").read_text())
    with zipfile.ZipFile(mobile / "mobile_images.zip") as archive:
        for row in rows:
            add(archive.read(row["img_filename"]), "osatlas_mobile", row["elements"], "mobile")
    by_source = defaultdict(list)
    development = []
    for record in candidates:
        if record["meta"]["split"] == "dev":
            development.append(record)
        else:
            by_source[record["meta"]["source"]].append(record)
    selected = []
    for source, cap in (("showui_desktop", 20000), ("seeclick_web", 40000)):
        group = by_source[source]
        rng.shuffle(group)
        selected.extend(group[:cap])
    needed = 100000 - len(selected)
    mobile = by_source["osatlas_mobile"]
    rng.shuffle(mobile)
    if len(mobile) < needed:
        raise ValueError(f"Insufficient genuine bbox records: need {needed} mobile records, have {len(mobile)}")
    selected.extend(mobile[:needed])
    rng.shuffle(selected)
    rng.shuffle(development)
    development = development[:512]
    train_groups = {r["group_id"] for r in selected}
    if train_groups & {r["group_id"] for r in development} or train_groups & forbidden:
        raise ValueError("GUI screenshot leakage detected")
    # Keep each selected media item once; unselected candidate images are expendable.
    keep = {record_image(record) for record in selected + development}
    for path in (root / "images").glob("*/*"):
        if path.is_file() and str(path.relative_to(root)) not in keep:
            path.unlink()
    counts = dict(Counter(r["meta"]["source"] for r in selected))
    save_dataset(root, "train", selected, list(counts), hashes, seed=args.seed,
                 source_counts=counts, max_targets_per_screenshot=12, evaluation_overlap=0)
    # Development is self-contained under eval/gui, with independent image copies.
    eval_root = base / "eval/gui"
    for record in development:
        name = record_image(record)
        destination = eval_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(root / name, destination)
    save_dataset(eval_root, "gui_dev", development, list(counts), hashes, seed=args.seed, purpose="development")
    for name in {record_image(r) for r in development} - {record_image(r) for r in selected}:
        (root / name).unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=("eval", "train"), required=True)
    args = parser.parse_args()
    (evaluation if args.stage == "eval" else training)(args)


if __name__ == "__main__":
    main()
