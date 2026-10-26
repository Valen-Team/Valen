"""Prepare merged Det releases, install independent files, and verify Hub commits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct


BENCHMARKS = (
    ("refcoco_val", "RefCOCO val"), ("refcoco_testA", "RefCOCO testA"),
    ("refcoco_testB", "RefCOCO testB"), ("refcocoplus_val", "RefCOCO+ val"),
    ("refcocoplus_testA", "RefCOCO+ testA"), ("refcocoplus_testB", "RefCOCO+ testB"),
    ("refcocog_val", "RefCOCOg val"), ("refcocog_test", "RefCOCOg test"),
    ("reasonseg_val", "ReasonSeg val"), ("reasonseg_test", "ReasonSeg test"),
    ("groundingme_test", "GroundingME test"),
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def file_path(directory, name):
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Release filenames must be plain filenames")
    return Path(directory) / name


def merged_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "export_manifest.json").read_text())
    if (manifest.get("finetuning_type") != "lora"
            or not manifest.get("model_name", "").startswith("Valen-Decision-Det-")
            or manifest.get("method", "sft") != "sft"
            or manifest.get("reload_validation", {}).get("independent_process") is not True):
        raise ValueError("Expected independently validated, merged SFT LoRA exports")
    required = {"model.safetensors", "config.json", "modeling_valen_spatial.py", "README.md",
                "requirements.txt", "LICENSE", "LICENSE.backbone", "tokenizer.json",
                "tokenizer_config.json", "processor_config.json", "chat_template.jinja"}
    if not required <= manifest.get("files_sha256", {}).keys():
        raise ValueError("Incomplete standalone model package")
    for name, expected in manifest["files_sha256"].items():
        if digest(file_path(directory, name)) != expected:
            raise ValueError("Export file hash differs: " + name)
    with (directory / "model.safetensors").open("rb") as stream:
        header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
    if any("lora_" in name.lower() or "adapter" in name.lower() for name in header):
        raise ValueError("Weights still contain unmerged adapter tensors")
    return manifest


def read_benchmarks(run, name, identity):
    values = {}
    for folder in ("evaluation", "evaluation-reasonseg", "evaluation-groundingme"):
        report = json.loads((Path(run) / folder / "metrics.json").read_text())
        if report.get("model") != name:
            raise ValueError("Benchmark results identify another model")
        if folder == "evaluation-groundingme" and report.get("model_export_manifest_sha256") != identity:
            raise ValueError("GroundingME results identify another export")
        if set(values) & report["benchmarks"].keys():
            raise ValueError("Duplicate benchmark names")
        values.update(report["benchmarks"])
    if set(values) != {name for name, _ in BENCHMARKS}:
        raise ValueError("All eleven evaluation items are required")
    for value in values.values():
        if (type(value.get("accuracy")) not in (float, int) or not 0 <= value["accuracy"] <= 1
                or type(value.get("samples")) is not int or value["samples"] <= 0
                or not value.get("data_sha256")):
            raise ValueError("Invalid benchmark result")
    return values


def model_card(name, base_repo, repo, benchmarks):
    lines = ["---", "license: apache-2.0", "library_name: transformers", "base_model: " + base_repo,
             "base_model_relation: finetune", "tags:", "- custom_code", "- visual-grounding",
             "- valen", "---", "", "# " + name, "",
             f"A visual grounding model based on {base_repo}, trained on 100k RefCOCO-family and ReasonSeg examples.",
             "Given an image and a referring instruction, its spatial decision head returns normalized xyxy boxes and confidence scores. The LoRA weights are merged; no separate backbone or adapter is needed.",
             "", "## Inference", "",
             'Install `torch`, `transformers==5.4.0`, `huggingface_hub`, `safetensors`, `Pillow`, `numpy` and `pycocotools`. Replace `example.jpg` with a local image.', "",
             "```python", "from huggingface_hub import snapshot_download", "from transformers import AutoModel", "",
             f'folder = snapshot_download("{repo}")',
             'model = AutoModel.from_pretrained(folder, trust_remote_code=True, dtype="auto").to("cuda").eval()',
             'record = {', '    "id": "demo", "group_id": "demo",', '    "request": {',
             '        "state": {"messages": [{"role": "user", "content": [',
             '            {"type": "image_url", "image_url": {"url": "example.jpg"}}',
             '        ]}]},', '        "questions": {"target": {',
             '            "type": "det", "instructions": "the person on the left",',
             '            "categories": ["target"]', '        }}', '    }', '}',
             'print(model.predict(record, media_root=".")["answers"]["target"])', "```", "",
             "## Evaluation", "", "BBox localization Acc@0.5 (%), using up to 262,144 image pixels. RefCOCO/RefCOCO+ use UNC splits; RefCOCOg uses UMD splits.",
             "", "| Benchmark | Accuracy (%) |", "| --- | ---: |"]
    for key, label in BENCHMARKS:
        if key == "groundingme_test":
            continue
        lines.append(f"| {label} | {benchmarks[key]['accuracy'] * 100:.2f} |")
    return "\n".join(lines) + "\n"


def prepare(source_root, runs_root, run_id, release_root, namespace, sizes):
    release_root = Path(release_root)
    release_root.mkdir(parents=True, exist_ok=True)
    plan = {"format": "valen_merged_det_release_v1", "models": []}
    for size in sizes:
        name = "Valen-Decision-Det-" + size
        source = Path(source_root) / name
        manifest = merged_manifest(source)
        identity = digest(source / "export_manifest.json")
        if manifest["model_name"] != name:
            raise ValueError("Requested size differs from the export")
        run = Path(runs_root) / size / run_id
        benchmarks = read_benchmarks(run, name, identity)
        repo = namespace + "/" + name
        target = release_root / name
        target.mkdir(exist_ok=True)
        for filename in manifest["files_sha256"]:
            destination = file_path(target, filename)
            if filename == "model.safetensors":
                if destination.is_symlink():
                    if destination.resolve() != (source / filename).resolve():
                        raise ValueError("Prepared weights reference another export")
                elif destination.exists():
                    raise ValueError("Inspect the existing release weights before preparing again")
                else:
                    # Staging references avoid a redundant multi-GB weight copy. Installation materializes them.
                    destination.symlink_to((source / filename).resolve())
            else:
                shutil.copyfile(source / filename, destination)
        (target / "README.md").write_text(model_card(name, manifest["base_repo"], repo, benchmarks))
        manifest["files_sha256"]["README.md"] = digest(target / "README.md")
        manifest["publication"] = {"source_export_manifest_sha256": identity, "repo_id": repo,
                                   "weights_unchanged": True, "adapter_tensors": 0}
        write_json(target / "export_manifest.json", manifest)
        files = {name: {"sha256": digest(target / name), "bytes": (target / name).stat().st_size}
                 for name in [*manifest["files_sha256"], "export_manifest.json"]}
        plan["models"].append({"name": name, "repo_id": repo, "files": files,
                               "source_export_manifest_sha256": identity})
        print(json.dumps({"prepared": name, "files": len(files), "bytes": sum(f["bytes"] for f in files.values())}), flush=True)
    write_json(release_root / "publication-plan.json", plan)
    return plan


def verify(directory, files):
    for name, item in files.items():
        path = file_path(directory, name)
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError("Release file changed: " + name)


def install(source, target, files):
    """Preserve checkout metadata; atomically copy only manifested model files."""
    source, target = Path(source).resolve(), Path(target)
    if target.is_symlink() or target.resolve() == source:
        raise ValueError("Model installation requires a separate real directory")
    target.mkdir(parents=True, exist_ok=True)
    verify(source, files)
    for name, item in files.items():
        destination = file_path(target, name)
        if destination.is_file() and not destination.is_symlink() and digest(destination) == item["sha256"]:
            continue
        temporary = destination.with_name(destination.name + ".valen-copying")
        if temporary.is_symlink():
            raise ValueError("Refuse an unexpected copy staging symlink")
        shutil.copyfile(source / name, temporary)
        if digest(temporary) != item["sha256"]:
            raise ValueError("Copied file differs from the release")
        temporary.replace(destination)
    verify(target, files)
    if any((target / name).is_symlink() or (target / name).stat().st_nlink != 1 for name in files):
        raise ValueError("Installed model must contain independent regular files")


def push(directory, specification, api):
    from huggingface_hub import CommitOperationAdd
    repo = specification["repo_id"]
    files = specification["files"]
    verify(directory, files)
    api.create_repo(repo, repo_type="model", exist_ok=True)
    before = api.model_info(repo, revision="main")
    commit = api.create_commit(repo_id=repo, repo_type="model", revision="main", parent_commit=before.sha,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=Path(directory) / name) for name in files],
        commit_message="Release merged Det model, benchmark results and minimal inference example", num_threads=4)
    remote = api.model_info(repo, revision=commit.oid, files_metadata=True)
    siblings = {row.rfilename: row for row in remote.siblings}
    for name, item in files.items():
        row = siblings.get(name)
        if row is None or row.size != item["bytes"]:
            raise ValueError("Remote file is missing or has the wrong size: " + name)
        if row.lfs:
            if row.lfs.sha256 != item["sha256"]:
                raise ValueError("Remote weight hash differs: " + name)
        else:
            data = (Path(directory) / name).read_bytes()
            expected = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if row.blob_id != expected:
                raise ValueError("Remote Git blob differs: " + name)
    return {"status": "verified", "repo_id": repo, "commit_sha": commit.oid,
            "commit_url": commit.commit_url, "files": len(files), "verified_at": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--runs-root", type=Path, required=True)
    build.add_argument("--run-id", required=True)
    build.add_argument("--release-root", type=Path, required=True)
    build.add_argument("--namespace", required=True)
    build.add_argument("--sizes", nargs="+", choices=("0.8B", "2B", "4B"), default=["0.8B", "2B", "4B"])
    publish = commands.add_parser("publish")
    publish.add_argument("--release-root", type=Path, required=True)
    publish.add_argument("--install-root", type=Path, help="Materialize independent model folders here before upload")
    publish.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source_root, args.runs_root, args.run_id, args.release_root, args.namespace, args.sizes)
        return
    from huggingface_hub import HfApi, get_token
    plan = json.loads((args.release_root / "publication-plan.json").read_text())
    if plan.get("format") != "valen_merged_det_release_v1":
        raise ValueError("Unknown publication plan")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=get_token(), headers={"Cache-Control": "no-cache"})
    for specification in plan["models"]:
        receipt = {"repo_id": specification["repo_id"], "status": "pending"}
        result_path = args.work_dir / (specification["name"] + "-publication.json")
        try:
            directory = args.release_root / specification["name"]
            if args.install_root:
                destination = args.install_root / specification["name"]
                install(directory, destination, specification["files"])
                directory = destination
                receipt.update(installed=True, installation_directory=str(destination))
            if not get_token():
                raise ValueError("Hugging Face authentication is unavailable")
            api.whoami()
            receipt.update(push(directory, specification, api))
        except Exception as exc:
            receipt.update(status="incomplete", error_type=type(exc).__name__, error=str(exc))
            write_json(result_path, receipt)
            raise
        write_json(result_path, receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
