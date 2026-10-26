"""Build a sealed GUI revision from existing data and desktop annotations.

Reads the input corpora without changing them. Only selected, original-resolution
media are copied; benchmark annotations remain unchanged. No synthetic boxes or
rewritten instructions are introduced.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import shutil
import threading
import unicodedata
import zipfile

import numpy as np
from PIL import Image

from scripts.data.prepare_spatial import (
    digest_file, make_record, record_image, save_dataset, source_rows, write_json,
)


GENERIC = {"image", "button", "icon", "text", "input", "menu", "tab", "link",
           "panel", "toolbar", "window", "desktop", "item", "click", "unknown",
           "element", "ui element", "details", "grid", "column", "row"}


def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def is_pointer_annotation(text):
    key = text_key(text)
    return key in {"cursor", "mouse pointer", "cursor movement", "text cursor", "typing cursor",
                   "mouse cursor", "pointer cursor", "i-cursor", "icursor", "shape cursor",
                   "new cursor", "cursor - selected"} or bool(re.match(
        r"^(?:mouse_(?:up|down)\s+)?cursor\s+(?:on\b|0n\b|at\b|position\b|coordinates?\b|"
        r"co-?ordinates?\b|coordintes\b|location\b|clicking\b|right clicking\b|selecting\b|"
        r"unselecting\b|dragging\b|moved\b|structure\b|and\b|after\b|in which\b)", key))


def clean_elements(elements, size, counts, *, desktop=False, exclude_pointer_labels=False):
    """Reject malformed, ambiguous and duplicate targets without repairing labels."""
    grouped = defaultdict(dict)
    width, height = size
    for text, box, kind in elements:
        if (not isinstance(text, str) or not text.strip() or len(text.strip()) > 512
                or re.search(r"<\|.*?\|>|[\x00-\x08\x0b\x0c\x0e-\x1f]", text)):
            counts["invalid_text"] += 1
            continue
        key = text_key(text)
        if exclude_pointer_labels and is_pointer_annotation(text):
            counts["pointer_annotation"] += 1
            continue
        if key in GENERIC or not any(ch.isalpha() for ch in key):
            counts["generic_text"] += 1
            continue
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) for x in box)
                or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
            counts["invalid_box"] += 1
            continue
        bw, bh = box[2] - box[0], box[3] - box[1]
        if min(bw * width, bh * height) < 2 or (desktop and bw * bh > .2):
            counts["unusable_box_extent"] += 1
            continue
        target = tuple(round(x, 7) for x in box)
        if target in grouped[key]:
            counts["duplicate_annotation"] += 1
        grouped[key][target] = (text.strip(), list(box), str(kind))
    eligible = []
    for targets in grouped.values():
        if len(targets) != 1:
            counts["ambiguous_instruction"] += len(targets)
        else:
            eligible.append(next(iter(targets.values())))
    # Multiple aliases for the same exact box do not consume the image quota.
    by_box = {}
    for element in sorted(eligible, key=lambda x: (text_key(x[0]), x[0])):
        key = tuple(round(x, 7) for x in element[1])
        if key in by_box:
            counts["duplicate_target_alias"] += 1
        else:
            by_box[key] = element
    return list(by_box.values())


def image_features(data, cache_dir=None):
    raw = hashlib.sha256(data).hexdigest()
    cached = Path(cache_dir) / raw[:2] / (raw + ".json") if cache_dir else None
    if cached and cached.is_file():
        try:
            payload = json.loads(cached.read_text())
            signature = payload.pop("feature_sha256")
            if (signature != hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                    or payload.pop("algorithm") != "rgb-dhash128-thumbnail64x36-v1" or payload["raw"] != raw):
                raise ValueError("Stale or corrupt feature cache")
            payload["size"] = tuple(payload["size"])
            payload["thumbnail"] = np.frombuffer(bytes.fromhex(payload["thumbnail"]), dtype=np.uint8).reshape(36, 64, 3)
            return payload
        except (OSError, KeyError, ValueError, TypeError):
            pass
    with Image.open(io.BytesIO(data)) as image:
        suffix = ".jpg" if image.format == "JPEG" else ".png"
        rgb = image.convert("RGB")
        size = rgb.size
        pixels = hashlib.sha256(str(size).encode() + b":" + rgb.tobytes()).hexdigest()
        small = np.asarray(rgb.resize((64, 36), Image.Resampling.BILINEAR)).copy()
        gray = np.asarray(rgb.convert("L").resize((9, 16), Image.Resampling.BILINEAR))
        bits = np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes()
    result = dict(raw=raw, pixels=pixels, size=size, suffix=suffix,
                  dhash=int.from_bytes(bits, "big"), thumbnail=small)
    if cached:
        payload = dict(result, thumbnail=small.tobytes().hex(), algorithm="rgb-dhash128-thumbnail64x36-v1")
        payload["feature_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_suffix("." + str(threading.get_ident()) + ".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(cached)
    return result


def near(first, second):
    if first["pixels"] == second["pixels"]:
        return True
    aw, ah = first["size"]
    bw, bh = second["size"]
    if abs((aw / ah) / (bw / bh) - 1) > .01:
        return False
    if (first["dhash"] ^ second["dhash"]).bit_count() > 8:
        return False
    delta = np.abs(first["thumbnail"].astype(np.int16) - second["thumbnail"])
    return float(delta.mean()) <= 3 and float((delta.max(-1) > 25).mean()) <= .08


class NearIndex:
    """Nine disjoint bit bands retrieve every pair within eight hash bits."""
    def __init__(self):
        self.bands = defaultdict(list)
        self.features = []

    @staticmethod
    def keys(feature):
        value = feature["dhash"]
        for index in range(9):
            shift = index * 14
            yield index, (value >> shift) & ((1 << (16 if index == 8 else 14)) - 1)

    def matches(self, feature):
        candidates = set()
        for key in self.keys(feature):
            candidates.update(self.bands.get(key, ()))
        return [index for index in sorted(candidates) if near(feature, self.features[index])]

    def add(self, feature):
        index = len(self.features)
        self.features.append(feature)
        for key in self.keys(feature):
            self.bands[key].append(index)
        return index


def remaining_elements(group, retained=(), rejected=()):
    boxes = {tuple(r["targets"]["target"]["boxes"][0]) for r in retained}
    texts = {text_key(r["request"]["questions"]["target"]["instructions"]) for r in retained}
    return [item for item in group["elements"] if tuple(item[1]) not in boxes
            and text_key(item[0]) not in texts and (not rejected or element_id(group, item) not in rejected)]


def element_id(group, element):
    text, box, _ = element
    uid = hashlib.sha256((group["feature"]["pixels"] + text_key(text) + repr(box)).encode()).hexdigest()
    return group["source"] + ":" + uid


def choose_images(groups, count, rng, limit=12, retained=None, rejected=()):
    """Balance applications and screenshots before selecting another query."""
    if count == 0:
        return []
    retained = retained or {}
    buckets = defaultdict(list)
    for group in groups:
        previous = retained.get(group["feature"].get("pixels"), [])
        items = remaining_elements(group, previous, rejected)
        rng.shuffle(items)
        # Interleave small and larger targets, using annotations from training only.
        small, other = [], []
        width, height = group["feature"]["size"]
        scale = min(1., math.sqrt(1048576 / (width * height)))
        for item in items:
            box = item[1]
            short = min((box[2] - box[0]) * width, (box[3] - box[1]) * height) * scale
            (small if short < 16 else other).append(item)
        ordered = []
        while small or other:
            if small:
                ordered.append(small.pop())
            if other:
                ordered.append(other.pop())
        buckets[group["application"]].append((group, ordered[:max(0, limit - len(previous))]))
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    # Every eligible screenshot gets its first query before its second query.
    for depth in range(limit):
        queues = {app: [(group, elements[depth]) for group, elements in bucket
                        if len(elements) > depth] for app, bucket in buckets.items()}
        while any(queues.values()):
            for app in sorted(queues):
                if queues[app]:
                    selected.append(queues[app].pop())
                    if len(selected) == count:
                        return selected
    raise ValueError(f"Only {len(selected)} clean records available; need {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--extend-from", type=Path, help="Keep every record and the development split of this sealed revision")
    parser.add_argument("--revision", default="gui_desktop_v2")
    parser.add_argument("--groundcua-records", type=int, default=50000)
    parser.add_argument("--osatlas-desktop-records", type=int, default=10000)
    parser.add_argument("--web-records", type=int, default=20000)
    parser.add_argument("--mobile-records", type=int, default=20000)
    parser.add_argument("--desktop-dev-records", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    quotas = dict(groundcua_desktop=args.groundcua_records,
                  osatlas_desktop=args.osatlas_desktop_records,
                  seeclick_web=args.web_records, osatlas_mobile=args.mobile_records)
    target_samples = sum(quotas.values())
    if min(quotas.values()) <= 0:
        raise ValueError("Source quotas must be positive")
    base, sources, out = args.existing_root.resolve(), args.source_root.resolve(), args.output_root.resolve()
    retained_rows, retained, rejected = [], defaultdict(list), set()
    extension = args.extend_from.resolve() if args.extend_from else None
    if extension:
        from scripts.data.install_gui_revision import verify
        previous_seal = verify(extension)
        retained_rows = list(source_rows(extension / "gui/processed/train.jsonl"))
        for row in retained_rows:
            retained[row["group_id"]].append(row)
        if max(map(len, retained.values())) > 12:
            raise ValueError("The previous revision exceeds the per-image query limit")
        curated = extension / "gui/manifests/manual-curation.json"
        if curated.is_file():
            rejected = {item["before"]["id"] for item in json.loads(curated.read_text())["replacements"]}
        if len(retained_rows) >= target_samples:
            raise ValueError("Expansion quotas must exceed the previous sample count")
    if out.exists():
        raise ValueError("Use a new output directory; sealed corpora are immutable")
    if out == base or out.is_relative_to(sources) or base.is_relative_to(out):
        raise ValueError("Output must be independent of the input corpus")
    out.mkdir(parents=True)
    work = out.with_name(out.name + "-feature-cache")
    owner = work / "owner.json"
    identity = dict(format="valen_gui_feature_cache_v1", output_root=str(out))
    if work.is_symlink() or (work.exists() and (not owner.is_file() or json.loads(owner.read_text()) != identity)):
        raise ValueError("Refuse an unrelated feature cache directory")
    write_json(owner, identity)
    features = lambda data: image_features(data, work)
    log = lambda **value: print(json.dumps(value), flush=True)
    counts = Counter()
    sources_proof = []
    initial_proofs = {}
    # Det is observed only to prove it did not change during this operation.
    for relative in ("det/processed/train.jsonl", "det/manifests/ready.json",
                     "gui/processed/train.jsonl", "gui/manifests/ready.json"):
        initial_proofs[relative] = digest_file(base / relative)
    old_manifest = json.loads((base / "gui/manifests/train.json").read_text())
    if initial_proofs["gui/processed/train.jsonl"] != old_manifest["records_sha256"]:
        raise ValueError("Existing GUI records do not match their manifest")
    ground = sources / "Fhrozen--GroundCUA"
    inventory = json.loads((ground / "protocol-v2-inventory.json").read_text())
    pinned = {x["path"]: x for x in inventory["selected"]}
    for path in sorted(ground.glob("*/*.parquet")):
        relative = path.relative_to(ground).as_posix()
        actual = digest_file(path)
        if actual != pinned[relative]["lfs"]["oid"]:
            raise ValueError(f"Source checksum mismatch: {relative}")
        sources_proof.append(dict(dataset=inventory["repo"], revision=inventory["revision"],
                                  file=relative, sha256=actual, bytes=path.stat().st_size))
    atlas = sources / "OS-Copilot--OS-Atlas-data"
    atlas_inventory = json.loads((atlas / "inventory.json").read_text())
    atlas_tree = {x["path"]: x for x in atlas_inventory["tree"]}
    for platform in ("linux", "macos"):
        for suffix in ("_splited.json", "_images.zip"):
            path = atlas / "desktop_domain" / (platform + suffix)
            relative = path.relative_to(atlas).as_posix()
            actual = digest_file(path)
            entry = atlas_tree.get(relative, {})
            if entry.get("lfs") and actual != entry["lfs"]["oid"]:
                raise ValueError(f"Source checksum mismatch: {relative}")
            sources_proof.append(dict(dataset=atlas_inventory["repo"], revision=atlas_inventory["revision"],
                                      file=relative, sha256=actual, bytes=path.stat().st_size))
    write_json(out / "build-status.json", dict(stage="source_checksums_verified"))
    log(phase="source_checksums_verified", files=len(sources_proof))

    protected_paths = {}
    benchmark_hashes = {}
    global_raw = set()
    for manifest in sorted((base / "eval").glob("*/manifests/*.json")):
        value = json.loads(manifest.read_text())
        global_raw.update(value.get("media_sha256", {}).values())
        if manifest.parent.parent.name == "gui" and "media_sha256" in value:
            for relative, expected in value["media_sha256"].items():
                protected_paths[base / "eval/gui" / relative] = expected
            if manifest.stem in {"screenspot_test", "screenspot_v2_test", "screenspot_pro_test"}:
                benchmark_hashes[manifest.stem] = value["records_sha256"]
    ui = sources / "ServiceNow--ui-vision"
    ui_inventory = json.loads((ui / "protocol-v2-inventory.json").read_text())
    for entry in ui_inventory["tree"]:
        if entry["path"].startswith("images/") and entry["path"].endswith(".png"):
            protected_paths[ui / entry["path"]] = entry["lfs"]["oid"]
    protected = NearIndex()
    def load_protected(item):
        path, expected = item
        feature = features(path.read_bytes())
        if feature["raw"] != expected:
            raise ValueError("Protected benchmark media checksum changed")
        return feature
    with ThreadPoolExecutor(max_workers=4) as pool:
        for feature in pool.map(load_protected, sorted(protected_paths.items())):
            protected.add(feature)
    log(phase="protected_media", images=len(protected.features))

    groups = {}
    def add(data, source, application, elements, locator, platform, feature=None):
        if feature is None:
            try:
                feature = features(data)
            except (OSError, ValueError, Image.DecompressionBombError):
                counts["invalid_source_media"] += 1
                return
        if feature["raw"] in global_raw or protected.matches(feature):
            counts["protected_image_rows"] += 1
            return
        key = feature["pixels"]
        if key in groups:
            counts["duplicate_image_rows"] += 1
            groups[key]["raw_elements"].extend(elements)
            return
        groups[key] = dict(feature=feature, source=source, application=application,
                           platform=platform, raw_elements=list(elements), locator=locator)

    # Reuse the sealed Web/Mobile inputs, retaining original text and real boxes.
    old_groups = defaultdict(list)
    for row in source_rows(base / "gui/processed/train.jsonl"):
        if row["meta"]["source"] in {"seeclick_web", "osatlas_mobile"}:
            old_groups[record_image(row)].append(row)
    def old_input(item):
        relative, rows = item
        expected = old_manifest["media_sha256"][relative]
        data = (base / "gui" / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Existing GUI media checksum changed")
        return data, rows, relative, features(data)
    inputs = sorted(old_groups.items())
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(inputs), 32):
            for data, rows, relative, feature in pool.map(old_input, inputs[offset:offset + 32]):
                row = rows[0]
                elements = [(r["request"]["questions"]["target"]["instructions"],
                             r["targets"]["target"]["boxes"][0], r["meta"].get("ui_type", "unknown")) for r in rows]
                add(data, row["meta"]["source"], row["meta"]["source"], elements,
                    ("file", str(base / "gui" / relative)), row["meta"]["platform"], feature)
            if offset % 1024 == 0:
                log(phase="existing_inputs", images=min(offset + 32, len(inputs)))
    import pyarrow.parquet as pq
    def source_feature(row):
        try:
            return row, features(row["image"]["bytes"])
        except (OSError, ValueError, Image.DecompressionBombError):
            return row, None
    with ThreadPoolExecutor(max_workers=4) as pool:
        for path in sorted(ground.glob("*/*.parquet")):
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
                for row, feature in pool.map(source_feature, batch.to_pylist()):
                    row_index, index = index, index + 1
                    if feature is None:
                        counts["invalid_source_media"] += 1
                        continue
                    width, height = feature["size"]
                    if not (len(row["bbox"]) == len(row["text"]) == len(row["category"]) == row["num_boxes"]):
                        raise ValueError("Mismatched GroundCUA annotation columns")
                    elements = [(text, [float(x) / side for x, side in zip(box, (width, height, width, height))], kind)
                                for text, box, kind in zip(row["text"], row["bbox"], row["category"])]
                    add(row["image"]["bytes"], "groundcua_desktop", path.parent.name, elements,
                        ("parquet", str(path), row_index), "desktop", feature)
            log(phase="groundcua", application=path.parent.name, images=len(groups))
    for platform in ("linux", "macos"):
        directory = atlas / "desktop_domain"
        annotations = defaultdict(list)
        for row in json.loads((directory / (platform + "_splited.json")).read_text()):
            annotations[row["img_filename"]].extend((x.get("instruction"), x.get("bbox"),
                                                    x.get("data_type", "unknown")) for x in row["elements"])
        archive_path = directory / (platform + "_images.zip")
        with zipfile.ZipFile(archive_path) as archive:
            members = {Path(x).name: x for x in archive.namelist() if not x.endswith("/")}
            def atlas_input(item):
                name, elements = item
                member = members.get(Path(name).name)
                if member is None:
                    return None, "missing_source_media"
                try:
                    data = archive.read(member)
                    return (data, elements, member, features(data)), None
                except (OSError, ValueError, zipfile.BadZipFile, Image.DecompressionBombError):
                    return None, "invalid_source_media"
            inputs = sorted(annotations.items())
            with ThreadPoolExecutor(max_workers=4) as pool:
                for offset in range(0, len(inputs), 16):
                    for value, error in pool.map(atlas_input, inputs[offset:offset + 16]):
                        if error:
                            counts[error] += 1
                            continue
                        data, elements, member, feature = value
                        add(data, "osatlas_desktop", "osatlas_" + platform, elements,
                            ("zip", str(archive_path), member), "desktop", feature)
        log(phase="osatlas_desktop", platform=platform, images=len(groups))

    # Near-duplicate connected components stay together across train/development.
    index = NearIndex()
    ordered = sorted(groups.values(), key=lambda x: x["feature"]["pixels"])
    parent = list(range(len(ordered)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, group in enumerate(ordered):
        for j in index.matches(group["feature"]):
            first, second = find(i), find(j)
            if first != second:
                parent[max(first, second)] = min(first, second)
                counts["near_duplicate_links"] += 1
        index.add(group["feature"])
    by_source = defaultdict(list)
    development = []
    component_sizes = Counter(find(i) for i in range(len(ordered)))
    previous_training_components = {find(i) for i, g in enumerate(ordered)
                                    if g["source"] in {"seeclick_web", "osatlas_mobile"}}
    for i, group in enumerate(ordered):
        group["component"] = ordered[find(i)]["feature"]["pixels"]
        group["elements"] = clean_elements(group.pop("raw_elements"), group["feature"]["size"], counts,
                                            desktop=group["platform"] not in {"web", "mobile"},
                                            exclude_pointer_labels=group["source"] == "groundcua_desktop")
        if not group["elements"]:
            continue
        # Old source training images never become new development images.
        if (find(i) not in previous_training_components
                and group["source"] in {"groundcua_desktop", "osatlas_desktop"}
                and int(group["component"][:8], 16) % 20 == 0):
            development.append(group)
        else:
            by_source[group["source"]].append(group)
    retained_counts = Counter(r["meta"]["source"] for r in retained_rows)
    eligible_keys = {g["feature"]["pixels"] for value in by_source.values() for g in value}
    if set(retained) - eligible_keys:
        raise ValueError("An inherited training image is no longer in the clean training pool")
    capacities = {key: retained_counts[key] + sum(min(12 - len(retained.get(g["feature"]["pixels"], [])),
        len(remaining_elements(g, retained.get(g["feature"]["pixels"], []), rejected))) for g in value)
        for key, value in by_source.items()}
    log(phase="clean_capacity", capacities=capacities, filters=dict(counts),
        near_components=len(component_sizes), largest_component=max(component_sizes.values(), default=0))
    write_json(out / "build-status.json", dict(stage="clean_candidates", capacities=capacities, filters=dict(counts)))
    requested_quotas = dict(quotas)
    quotas = {key: min(value, capacities.get(key, 0)) for key, value in quotas.items()}
    deficit = target_samples - sum(quotas.values())
    for key in ("groundcua_desktop", "osatlas_desktop", "seeclick_web", "osatlas_mobile"):
        extra = min(deficit, capacities.get(key, 0) - quotas[key])
        quotas[key] += extra
        deficit -= extra
    if deficit:
        raise ValueError(f"Clean pool is {deficit} records short of {target_samples}; never duplicate records to fill quotas")
    if any(quotas[key] < retained_counts[key] for key in retained_counts):
        raise ValueError("Expansion quotas cannot discard inherited records")
    log(phase="fixed_sampling_quotas", requested=requested_quotas, actual=quotas)
    rng = random.Random(args.seed)
    selected = []
    for source, quota in quotas.items():
        if quota:
            selected.extend(choose_images(by_source[source], quota - retained_counts[source], rng,
                                          retained=retained, rejected=rejected))
    selected_dev = [] if extension else choose_images(development, args.desktop_dev_records, rng, limit=3)
    rng.shuffle(selected)

    archives = {}
    def media(group):
        loc = group["locator"]
        if loc[0] == "file":
            return Path(loc[1]).read_bytes()
        if loc[0] == "zip":
            if loc[1] not in archives:
                archives[loc[1]] = zipfile.ZipFile(loc[1])
            return archives[loc[1]].read(loc[2])
        raise ValueError("Parquet media must be materialized in one sequential pass")
    chosen_groups = {g["feature"]["pixels"]: g for g, _ in selected + selected_dev}
    train_root, eval_root = out / "gui", out / "eval/gui"
    train_keys = {g["feature"]["pixels"] for g, _ in selected}
    # Extract only selected parquet images, discarding batch buffers immediately.
    parquet_needed = defaultdict(set)
    for key, group in chosen_groups.items():
        if group["locator"][0] == "parquet":
            parquet_needed[group["locator"][1]].add(group["locator"][2])
    for name, indices in sorted(parquet_needed.items()):
        for i, batch in enumerate(pq.ParquetFile(name).iter_batches(batch_size=1)):
            if i in indices:
                row = batch.to_pylist()[0]
                data = row["image"]["bytes"]
                feature = features(data)
                key = feature["pixels"]
                if key not in chosen_groups:
                    raise ValueError("Selected parquet source changed")
                destination = train_root if key in train_keys else eval_root
                relative = f"images/{feature['raw'][:2]}/{feature['raw']}{feature['suffix']}"
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    hashes = {}
    def materialize(pairs, destination, split):
        rows = []
        for group, (text, box, kind) in pairs:
            feature = group["feature"]
            raw = feature["raw"]
            relative = f"images/{raw[:2]}/{raw}{feature['suffix']}"
            target = destination / relative
            if not target.exists():
                data = media(group)
                if hashlib.sha256(data).hexdigest() != raw:
                    raise ValueError("Selected source image changed")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            hashes[relative] = raw
            uid = hashlib.sha256((feature["pixels"] + text_key(text) + repr(box)).encode()).hexdigest()
            rows.append(make_record("gui", group["source"] + ":" + uid, feature["pixels"], relative,
                text, ["target"], [box], [0], source=group["source"], split=split,
                platform=group["platform"], application=group["application"], ui_type=kind,
                width=feature["size"][0], height=feature["size"][1],
                near_duplicate_group=group["component"], revision=args.revision))
        return rows
    train_rows = materialize(selected, train_root, "train")
    if extension:
        previous_train = json.loads((extension / "gui/manifests/train.json").read_text())
        for relative, expected in previous_train["media_sha256"].items():
            target = train_root / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(extension / "gui" / relative, target)
            if digest_file(target) != expected:
                raise ValueError("An inherited training image changed")
            hashes[relative] = expected
        train_rows = retained_rows + train_rows
        rng.shuffle(train_rows)
    dev_rows = materialize(selected_dev, eval_root, "dev")
    for archive in archives.values():
        archive.close()
    # Keep the original GUI development set and every official benchmark intact.
    dev_input = extension if extension else base
    for row in source_rows(dev_input / "eval/gui/processed/gui_dev.jsonl"):
        relative = record_image(row)
        target = eval_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dev_input / "eval/gui" / relative, target)
        hashes[relative] = digest_file(target)
        dev_rows.append(row)
    for name, expected in benchmark_hashes.items():
        manifest = json.loads((base / f"eval/gui/manifests/{name}.json").read_text())
        if digest_file(base / f"eval/gui/processed/{name}.jsonl") != expected:
            raise ValueError("Benchmark annotations changed")
        for relative, digest in manifest["media_sha256"].items():
            target = eval_root / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base / "eval/gui" / relative, target)
            if digest_file(target) != digest:
                raise ValueError("Benchmark copy checksum mismatch")
        target = eval_root / "processed" / (name + ".jsonl")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base / "eval/gui/processed" / (name + ".jsonl"), target)
        write_json(eval_root / "manifests" / (name + ".json"), manifest)
    train = save_dataset(train_root, "train", train_rows, list(quotas), hashes, seed=args.seed,
                         source_counts=quotas, max_targets_per_screenshot=12, revision=args.revision)
    dev = save_dataset(eval_root, "gui_dev", dev_rows, sorted({r["meta"]["source"] for r in dev_rows}),
                       hashes, seed=args.seed, purpose="development", revision=args.revision)
    if extension:
        if dev["records_sha256"] != previous_seal["development_sha256"]:
            raise ValueError("Expansion changed the fixed development split")
        write_json(train_root / "manifests/extension.json", dict(
            previous_records_sha256=previous_seal["records_sha256"], previous_samples=len(retained_rows),
            retained_records=len(retained_rows), added_records=len(train_rows)-len(retained_rows),
            rejected_record_ids=sorted(rejected), development_unchanged=True))
    inherited = json.loads((base / "gui/manifests/sources.json").read_text())
    write_json(train_root / "manifests/sources.json", dict(
        inherited_sources=[s for s in inherited["sources"] if s["dataset"] != "ShowUI-desktop"],
        inherited_training_sha256=initial_proofs["gui/processed/train.jsonl"],
        added_source_files=sources_proof, protected_ui_vision_revision=ui_inventory["revision"],
        licenses={"GroundCUA": "MIT; original screenshot rights also apply",
                  "OS-Atlas": "Apache-2.0; original component terms also apply"}))
    from scripts.data.finalize_spatial import audit, preview
    checked = {}
    audit(train_root, "train", "gui", checked, global_raw)
    audit(eval_root, "gui_dev", "gui", checked)
    train_groups = {r["group_id"] for r in train_rows}
    dev_groups = {r["group_id"] for r in dev_rows}
    train_components = {r["meta"]["near_duplicate_group"] for r in train_rows}
    dev_components = {r["meta"].get("near_duplicate_group", r["group_id"]) for r in dev_rows}
    if train_groups & dev_groups or train_components & dev_components:
        raise ValueError("Train/development duplicate component overlap")
    for name in benchmark_hashes:
        audit(eval_root, name, "gui", checked)
    for relative, expected in initial_proofs.items():
        if digest_file(base / relative) != expected:
            raise ValueError("An input corpus changed while building the GUI revision")
    used_train = set(train["media_sha256"])
    old_unused = sorted(set(old_manifest["media_sha256"]) - used_train)
    short_sides, areas = [], []
    for row in train_rows:
        box = row["targets"]["target"]["boxes"][0]
        w, h = row["meta"]["width"], row["meta"]["height"]
        short_sides.append(min((box[2] - box[0]) * w, (box[3] - box[1]) * h)
                           * min(1., math.sqrt(1048576 / (w * h))))
        areas.append((box[2] - box[0]) * (box[3] - box[1]))
    write_json(train_root / "manifests/quality.json", dict(
        filters=dict(counts), source_capacities=capacities, protected_images=len(protected.features),
        exact_and_near_protected_overlap=0, train_development_component_overlap=0,
        near_duplicate_components=len(component_sizes), largest_near_component=max(component_sizes.values()),
        platform_counts=dict(Counter(r["meta"]["platform"] for r in train_rows)),
        application_counts=dict(Counter(r["meta"]["application"] for r in train_rows)),
        ui_type_counts=dict(Counter(r["meta"]["ui_type"] for r in train_rows)),
        requested_source_counts=requested_quotas, actual_source_counts=quotas,
        development_source_counts=dict(Counter(r["meta"]["source"] for r in dev_rows)),
        target_area_quantiles=dict(zip(("p10", "p50", "p90"), map(float, np.quantile(areas, [.1, .5, .9])))),
        target_short_side_at_1mp_quantiles=dict(zip(("p10", "p50", "p90"), map(float, np.quantile(short_sides, [.1, .5, .9])))),
        target_short_side_at_1mp_below16=sum(x < 16 for x in short_sides),
        independent_original_resolution_media=True, original_instructions_and_boxes=True,
        unchanged_input_sha256=initial_proofs, benchmark_sha256=benchmark_hashes,
        limitations=["Perceptual similarity detects near duplicates, not all shared sessions; source session IDs are unavailable.",
                     "Structural and ambiguity filters do not replace semantic review of every annotation."]))
    write_json(out / "obsolete-gui-media.json", dict(
        relative_to="previous GUI media root", files=old_unused,
        bytes=sum((base / "gui" / name).stat().st_size for name in old_unused),
        status="review references and filesystem permissions before removing the previous version"))
    preview(train_root, "gui")
    if train["samples"] != target_samples or max(Counter(record_image(r) for r in train_rows).values()) > 12:
        raise ValueError("Final corpus count or per-image quota differs from the sampling plan")
    ready = dict(format="valen_spatial_data_v1", validated=True, task="gui", revision=args.revision,
        time_utc=datetime.now(timezone.utc).isoformat(), samples=train["samples"], images=train["images"],
        targets=train["targets"], source_counts=quotas, records_sha256=train["records_sha256"],
        development_sha256=dev["records_sha256"], evaluation_sha256={"gui/" + k: v for k, v in benchmark_hashes.items()},
        global_evaluation_overlap=0, independent_media=True,
        sources_sha256=digest_file(train_root / "manifests/sources.json"),
        media_bytes=sum((train_root / name).stat().st_size for name in used_train))
    write_json(train_root / "manifests/ready.json", ready)
    (out / "build-status.json").unlink()
    shutil.rmtree(work)
    log(phase="complete", **ready)


if __name__ == "__main__":
    main()
