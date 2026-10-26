"""Validate and replace only GUI data with a previously sealed revision.

The default is a read-only preflight. --apply copies into the destination
filesystem before renaming, rolls back failed swaps, and removes the retired
GUI copies only after the installed corpus passes its checks.
"""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import uuid

from scripts.data.finalize_spatial import audit
from scripts.data.prepare_spatial import digest_file, write_json


def verify(root):
    train_root, eval_root = root / "gui", root / "eval/gui"
    seal = json.loads((train_root / "manifests/ready.json").read_text())
    if not seal.get("validated") or type(seal.get("samples")) is not int or seal["samples"] <= 0:
        raise ValueError("Expected a sealed GUI revision with a positive sample count")
    checked, forbidden = {}, set()
    for name in ("gui_dev", "screenspot_test", "screenspot_v2_test", "screenspot_pro_test"):
        manifest, _, _ = audit(eval_root, name, "gui", checked)
        forbidden.update(manifest["media_sha256"].values())
        expected = (seal["development_sha256"] if name == "gui_dev"
                    else seal["evaluation_sha256"]["gui/" + name])
        if manifest["records_sha256"] != expected:
            raise ValueError("Evaluation records differ from the seal")
    train, _, _ = audit(train_root, "train", "gui", checked, forbidden)
    if train["records_sha256"] != seal["records_sha256"] or train["samples"] != seal["samples"]:
        raise ValueError("Training records differ from the seal")
    if digest_file(train_root / "manifests/sources.json") != seal["sources_sha256"]:
        raise ValueError("Source provenance differs from the seal")
    return seal


def preflight(source, destination, registry):
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Revision and destination must be separate directory trees")
    state = json.loads(registry.read_text())
    active = [j["name"] for j in state["jobs"] if j.get("task") == "gui"
              and j.get("state") in {"running", "submitting", "recovering", "submitted"}]
    if active:
        raise ValueError("Stop active GUI consumers before replacing their data: " + ", ".join(active))
    seal = verify(source)
    quality = json.loads((source / "gui/manifests/quality.json").read_text())
    guard = quality["unchanged_input_sha256"]
    for relative, expected in guard.items():
        if digest_file(destination / relative) != expected:
            raise ValueError("Destination changed since preparation: " + relative)
    for name in ("screenspot_test", "screenspot_v2_test", "screenspot_pro_test"):
        expected = seal["evaluation_sha256"]["gui/" + name]
        if digest_file(destination / f"eval/gui/processed/{name}.jsonl") != expected:
            raise ValueError("The revision changes an official benchmark")
    return seal, guard


def install(source, destination, registry, report, apply=False):
    source, destination = source.resolve(), destination.resolve()
    seal, guard = preflight(source, destination, registry)
    result = dict(format="valen_gui_revision_install_v1", revision=seal.get("revision"),
                  time_utc=datetime.now(timezone.utc).isoformat(), samples=seal["samples"],
                  records_sha256=seal["records_sha256"], status="preflight_passed", applied=False)
    write_json(report, result)
    if not apply:
        return result
    token = uuid.uuid4().hex
    staged = destination / (".gui-revision-" + token)
    staged.mkdir()
    completed = []
    backups = []
    try:
        for relative in ("gui", "eval/gui"):
            original = source / relative
            for path in original.rglob("*"):
                if path.is_symlink() or (path.is_file() and path.stat().st_nlink != 1):
                    raise ValueError("A revision must contain independent regular files")
            shutil.copytree(original, staged / relative)
        verify(staged)
        # Recheck live consumers and input identities immediately before the swap.
        preflight(source, destination, registry)
        previous = {name: json.loads((destination / "gui/manifests" / name).read_text())
                    for name in ("ready.json", "sources.json")}
        # Retain compact baseline annotations for reproducibility, without old media.
        archives = []
        identity = previous["ready.json"].get("records_sha256", "previous")[:12]
        for relative, label in (("gui/processed/train.jsonl", "train.jsonl"),
                                ("gui/manifests/train.json", "manifest.json"),
                                ("eval/gui/processed/gui_dev.jsonl", "dev.jsonl")):
            old = destination / relative
            if old.is_file():
                archived = staged / "gui/manifests/history" / (identity + "-" + label + ".gz")
                archived.parent.mkdir(parents=True, exist_ok=True)
                with old.open("rb") as stream, archived.open("wb") as target:
                    with gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as compressed:
                        shutil.copyfileobj(stream, compressed)
                archives.append(dict(file=str(archived.relative_to(staged / "gui")),
                                     sha256=digest_file(archived), original_sha256=digest_file(old)))
        previous["annotation_archives"] = archives
        write_json(staged / "gui/manifests/previous-version.json", previous)
        for relative in ("gui", "eval/gui"):
            target = destination / relative
            if target.is_symlink():
                raise ValueError("Refuse to replace a symlinked GUI directory")
            backup = target.with_name(".gui-retired-" + token)
            target.rename(backup)
            backups.append((target, backup))
            (staged / relative).rename(target)
            completed.append(target)
        verify(destination)
        for relative, expected in guard.items():
            if relative.startswith("det/") and digest_file(destination / relative) != expected:
                raise ValueError("Det changed during installation")
        result.update(status="installed_and_verified", applied=True,
                      det_unchanged=True, official_benchmarks_unchanged=True)
        write_json(report, result)
    except BaseException:
        for target in reversed(completed):
            failed = staged / ("failed-" + str(len(completed)))
            target.rename(failed)
            completed.pop()
        for target, backup in reversed(backups):
            if not target.exists():
                backup.rename(target)
        raise
    else:
        removed_bytes = 0
        for _, backup in backups:
            removed_bytes += sum(p.stat().st_size for p in backup.rglob("*") if p.is_file())
            shutil.rmtree(backup)
        shutil.rmtree(staged)
        result["retired_gui_bytes_removed"] = removed_bytes
        write_json(report, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(install(args.revision_root, args.data_root, args.registry, args.report, args.apply)))


if __name__ == "__main__":
    main()
