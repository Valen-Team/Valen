"""Collect spatial models, benchmark metrics and LoRA/full costs into compact reports."""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path

from scripts.train.monitor_spatial import last_metric
from scripts.train.run_spatial_pipeline import required_benchmarks
from valen.training.checkpoint import file_sha256


def read_json(path):
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=64)
def completed_costs(path, modified_ns):
    elapsed = previous = peak = 0
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            value = row.get("elapsed_seconds", 0)
            if value < previous:
                elapsed += previous
            previous = value
            if row.get("peak_cuda_scope") == "all_ranks":
                peak = max(peak, row.get("peak_cuda_bytes", 0))
    return {"recorded_training_seconds": elapsed + previous, "peak_cuda_bytes_all_ranks": peak}


def stage_info(run, stage, expected_world_size=8, expected_records=100000):
    root = run / stage
    manifest = read_json(root / "run_manifest.json")
    config = manifest.get("config", {})
    completion = read_json(root / "training_complete.json")
    metric = last_metric(root / "metrics.jsonl") or {}
    complete = (completion.get("stop_reason") == "epochs_complete" and completion.get("epoch") == 1
                and metric.get("seen") == metric.get("total_records") == expected_records
                and manifest.get("world_size") == expected_world_size
                and len(manifest.get("gpu_names", [])) == expected_world_size)
    result = {"complete": complete, "seen": metric.get("seen", 0), "records": metric.get("total_records", expected_records),
              "step": metric.get("step", 0), "last_loss": metric.get("loss"),
              "world_size": manifest.get("world_size"), "gpu_names": manifest.get("gpu_names", []),
              "trainable_parameters": manifest.get("trainable_parameters"),
              "total_parameters": manifest.get("total_parameters"),
              "recorded_training_seconds": metric.get("elapsed_seconds"),
              "peak_cuda_bytes_all_ranks": metric.get("peak_cuda_bytes") if metric.get("peak_cuda_scope") == "all_ranks" else None,
              "training_data_sha256": config.get("data_sha256"), "development_sha256": config.get("validation_sha256"),
              "code_sha256": config.get("code_sha256"), "best_step": completion.get("best_step"),
              "settings": {key: config[key] for key in ("seed", "epochs", "dtype", "max_pixels", "gradient_accumulation",
                  "head_lr", "lora_lr", "language_lr", "merger_lr", "vision_lr") if key in config}}
    if complete:
        path = root / "metrics.jsonl"
        result.update(completed_costs(str(path), path.stat().st_mtime_ns))
    return result


def finite_metrics(value):
    if isinstance(value, dict):
        return all(finite_metrics(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_metrics(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def examples_ready(run, package, name):
    directory = Path(run) / "examples"
    report = read_json(directory / "examples.json")
    manifest = Path(package) / "export_manifest.json"
    preview = directory / "examples.jpg"
    return (report.get("format") == "valen_spatial_examples_v1" and report.get("model") == name
            and bool(report.get("examples")) and preview.is_file() and preview.stat().st_size > 0
            and (directory / "pipeline_complete.json").is_file() and manifest.is_file()
            and report.get("model_export_manifest_sha256") == file_sha256(manifest))


def benchmark_checks(benchmarks, expected, task, root, reasonseg=False):
    if set(benchmarks) != expected or not finite_metrics(benchmarks):
        return False
    for name, values in benchmarks.items():
        base = root / "eval" / task
        if reasonseg:
            base = base / "reasonseg"
        elif name == "coco2017_conditional_val2017":
            base = base / "protocols/coco_prompt"
        manifest = read_json(base / "manifests" / (name + ".json"))
        if (not manifest.get("samples") or values.get("samples") != manifest["samples"]
                or values.get("data_sha256") != manifest.get("records_sha256")):
            return False
        metrics = ({"AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L"} if name.startswith("coco2017_") else
                   {"mIoU", "cIoU"} if task == "seg" else {"accuracy"})
        if name.startswith("phrasecut_"):
            metrics.add("precision_at_05")
        if task == "seg" and reasonseg:
            metrics.add("gIoU")
        if not metrics <= values.keys() or any(not isinstance(values[key], (int, float))
                or isinstance(values[key], bool) or not 0 <= values[key] <= 1 for key in metrics):
            return False
        if task == "gui" and not values.get("groups"):
            return False
    return True


def groundingme_checks(run, package, name, data_root):
    from evaluation.det.groundingme import BENCHMARK, DATASET_REVISION, PROTOCOL, SCORER_REVISION
    report = read_json(run / "evaluation-groundingme/metrics.json")
    complete = read_json(run / "evaluation-groundingme/pipeline_complete.json")
    manifest = read_json(data_root / "eval/det/groundingme/manifests" / (BENCHMARK + ".json"))
    values = report.get("benchmarks", {}).get(BENCHMARK, {})
    groups = values.get("groups", {})
    expected = manifest.get("groups", {})
    expected_subgroups = manifest.get("subgroups", {})
    valid = (manifest.get("samples") == values.get("samples") == complete.get("samples") == 1005
        and set(report.get("benchmarks", {})) == {BENCHMARK} and finite_metrics(values)
        and values.get("data_sha256") == complete.get("data_sha256") == manifest.get("records_sha256")
        and values.get("protocol") == complete.get("protocol") == PROTOCOL
        and values.get("source_revision") == DATASET_REVISION and values.get("scorer_revision") == SCORER_REVISION
        and set(expected) == {"Discriminative", "Spatial", "Limited", "Rejection"}
        and sum(expected.values()) == 1005 and all(groups.get(key, {}).get("samples") == n for key, n in expected.items())
        and set(groups) == set(expected) | set(expected_subgroups)
        and all(groups.get(key, {}).get("samples") == n for key, n in expected_subgroups.items())
        and all(type(bucket.get(metric)) in (float, int) and 0 <= bucket[metric] <= 1
            for bucket in [values, *groups.values()] for metric in ("accuracy", "accuracy_at_075", "accuracy_at_09", "mean_box_iou")))
    export = package / "export_manifest.json"
    identity = (report.get("model") == complete.get("model") == name and export.is_file()
        and report.get("model_export_manifest_sha256") == file_sha256(export))
    return {"groundingme": valid, "groundingme_model_identity": identity}, report.get("benchmarks", {})


def collect_results(state, data_root, model_root):
    data_root, model_root = Path(data_root), Path(model_root)
    models = []
    for job in state["jobs"]:
        if job.get("kind") != "pipeline":
            continue
        task, size = job["task"], job["size"]
        job_data_root = Path(job.get("data_root", data_root))
        full = job.get("variant") == "full"
        rlcd = job.get("variant") == "rlcd"
        merged_reference = job.get("initialization_variant") == "lora_merged"
        name = "Valen-Decision-" + {"det": "Det", "seg": "Seg", "gui": "GUI"}[task] + "-" + size + ("-RLCD" if rlcd and merged_reference else "-Full-RLCD" if rlcd else "-Full" if full else "")
        run = Path(job["run_dir"])
        package = Path(job.get("export_dir", model_root / name))
        samples = job.get("training_samples", 100000)
        warmup = stage_info(Path(job["baseline_run"]) if full or rlcd else run, "warmup", job.get("warmup_gpus", job.get("gpus", 8)), samples)
        hard = read_json(Path(job["subset_dir"])/"manifest.json") if rlcd else {}
        joint = stage_info(run, "joint", job.get("gpus", 8), hard.get("selected", 0) if rlcd else samples)
        export = read_json(package / "export_manifest.json")
        evaluation = read_json(run / "evaluation/metrics.json")
        reasonseg = read_json(run / "evaluation-reasonseg/metrics.json")
        comparison = read_json(run / "comparison.json") if full or rlcd else {}
        seal_path = Path(job.get("data_seal_path", job_data_root / task / "manifests/ready.json"))
        seal = read_json(seal_path)
        benchmarks = evaluation.get("benchmarks", {})
        expected = required_benchmarks(task, job.get("coco_protocol", "standard"))
        requires_reasonseg = task in {"det", "seg"}
        checks = {"pipeline_complete": (run / "pipeline_complete.json").is_file(),
            "warmup_epoch": warmup["complete"], "joint_epoch": joint["complete"],
            "current_training_data": bool(seal.get("records_sha256")) and
                joint["training_data_sha256"] == export.get("training_data_sha256") == seal.get("records_sha256"),
            "pinned_data_seal": not job.get("data_seal_path") or (seal_path.is_file()
                and file_sha256(seal_path) == export.get("data_seal_sha256")),
            "matched_development": bool(seal.get("development_sha256")) and
                warmup["development_sha256"] == joint["development_sha256"] == seal.get("development_sha256"),
            "model_identity": export.get("model_name") == name,
            "independent_export_reload": export.get("reload_validation", {}).get("independent_process") is True,
            "export_files_present": bool(export.get("files_sha256")) and all(
                (package / filename).is_file() for filename in export.get("files_sha256", {})),
            "inference_examples": examples_ready(run, package, name),
            "evaluation_model_identity": evaluation.get("model") == name,
            "required_benchmarks": benchmark_checks(benchmarks, expected, task, job_data_root),
            "full_comparison": not full or (comparison.get("format") == "valen_spatial_lora_full_comparison_v1"
                and set(comparison.get("benchmarks", {})) == expected and finite_metrics(comparison)),
            "reasonseg_model_identity": not requires_reasonseg or reasonseg.get("model") == name,
            "reasonseg": not requires_reasonseg or (
                (run / "evaluation-reasonseg/pipeline_complete.json").is_file()
                and benchmark_checks(reasonseg.get("benchmarks", {}), {"reasonseg_val", "reasonseg_test"},
                                     task, job_data_root, reasonseg=True))}
        if rlcd:
            reference = Path(job.get("reference_export", str(Path(model_root)/(
                "Valen-Decision-Det-"+size+("" if merged_reference else "-Full")))))
            reference_manifest = reference/"export_manifest.json"
            identity = file_sha256(reference_manifest) if reference_manifest.is_file() else None
            valid_hard = (hard.get("validated") is True and hard.get("original_records")==100000
                and hard.get("original_training_sha256")==seal.get("records_sha256")
                and hard.get("model_export_manifest_sha256")==identity and hard.get("original_records_only") is True
                and type(hard.get("errors")) is int and 0<=hard["errors"]<=100000
                and hard.get("selected")==min(hard["errors"],20000))
            skipped = read_json(run/"pipeline_complete.json")
            if skipped.get("zero_error_skip") is True and valid_hard and hard["errors"]==0:
                models.append({"name":name,"task":task,"size":size,"variant":"rlcd","state":"skipped_zero_errors",
                    "initialization_variant":job.get("initialization_variant"),
                    "skipped":True,"deliverables_ready":True,"checks":{"audited_zero_error_skip":True},
                    "warmup_reused":True,"warmup":warmup,"joint":joint,"model_weight_bytes":0,
                    "export_identity":{},"benchmarks":{},"reasonseg_benchmarks":{},"groundingme_benchmarks":{},
                    "inference":{},"comparison":{},"skip_evidence":skipped})
                continue
            checks["current_training_data"] = (valid_hard and hard.get("records_sha256")
                == joint["training_data_sha256"] == export.get("training_data_sha256"))
            checks["full_reference_identity"] = (valid_hard and export.get("initialization_sha256")==identity)
            expected_all=expected|{"reasonseg_val","reasonseg_test","groundingme_test"}
            checks["rlcd_comparison"] = (comparison.get("format")=="valen_spatial_rlcd_comparison_v1"
                and set(comparison.get("benchmarks",{}))==expected_all and finite_metrics(comparison)
                and comparison.get("reference_export_manifest_sha256", comparison.get("full_export_manifest_sha256"))==identity
                and comparison.get("initialization_variant", "full")==job.get("initialization_variant", "full")
                and (package/"export_manifest.json").is_file()
                and comparison.get("rlcd_export_manifest_sha256")==file_sha256(package/"export_manifest.json"))
            checks["rlcd_resume_verified"] = read_json(run/"rl-training-resume-accepted.json").get("fixed_reference") is True
            post=read_json(run/"post-replay/pipeline_complete.json")
            checks["training_error_comparison"]=(post.get("original_records")==100000
                and post.get("original_training_sha256")==seal.get("records_sha256")
                and post.get("selected")==hard.get("selected") and post.get("subset_sha256")==hard.get("records_sha256")
                and (package/"export_manifest.json").is_file()
                and post.get("model_export_manifest_sha256")==file_sha256(package/"export_manifest.json"))
        if task == "det" and job.get("coco_protocol") == "conditional":
            checks["conditional_coco_protocol"] = benchmarks.get("coco2017_conditional_val2017", {}).get(
                "protocol") == "known_category_conditional_detection"
        groundingme = {}
        if task == "det" and state.get("groundingme_support"):
            extra_checks, groundingme = groundingme_checks(run, package, name, job_data_root)
            checks.update(extra_checks)
        extra_gui = {}
        requested_gui = state.get('gui_additional_benchmarks', {})
        if task == 'gui' and requested_gui.get('after_run') == str(run):
            from evaluation.gui.additional import results_ready
            extra_root = Path(requested_gui.get('data_root', job_data_root / 'eval/gui/additional'))
            extra_gui = read_json(run / 'evaluation-additional/metrics.json')
            checks['additional_gui_benchmarks'] = results_ready(extra_gui,
                read_json(run / 'evaluation-additional/pipeline_complete.json'),
                read_json(extra_root / 'manifests/ready.json'), package / 'export_manifest.json')
        models.append({"name": name, "task": task, "size": size, "variant": "rlcd" if rlcd else "full" if full else "lora",
            "data_revision": seal.get("revision"), "training_samples": samples,
            "state": job.get("state", "unknown"), "deliverables_ready": all(checks.values()), "checks": checks,
            "initialization_variant": job.get("initialization_variant"),
            "warmup_reused": full or rlcd, "warmup": warmup, "joint": joint,
            "model_weight_bytes": sum(path.stat().st_size for path in package.glob("*.safetensors")),
            "export_identity": {key: export.get(key) for key in ("base_repo", "base_revision", "code_sha256",
                "training_data_sha256", "data_seal_sha256", "source_checkpoint_sha256", "initialization_sha256")},
            "benchmarks": benchmarks, "reasonseg_benchmarks": reasonseg.get("benchmarks", {}),
            "groundingme_benchmarks": groundingme,
            "additional_gui_benchmarks": extra_gui.get('benchmarks', {}),
            "additional_gui_summary": extra_gui.get('official_summary', {}),
            "inference": evaluation.get("inference", {}), "comparison": comparison})
    by_identity = {(model["task"], model["size"], model["variant"]): model for model in models}
    for model in models:
        model["reasonseg_comparison"] = {}
        model["groundingme_comparison"] = {}
        if model["variant"] == "full" and model["task"] == "det" and state.get("groundingme_support"):
            reference = by_identity.get(("det", model["size"], "lora"), {})
            baseline = reference.get("groundingme_benchmarks", {}).get("groundingme_test", {})
            current = model["groundingme_benchmarks"].get("groundingme_test", {})
            matched = (reference.get("checks", {}).get("groundingme") is True
                and reference.get("checks", {}).get("groundingme_model_identity") is True
                and model["checks"]["groundingme"] and model["checks"]["groundingme_model_identity"]
                and current.get("data_sha256") == baseline.get("data_sha256"))
            model["checks"]["groundingme_comparison_data"] = matched
            if matched:
                model["groundingme_comparison"]["groundingme_test"] = {"full_minus_lora": {
                    "accuracy": current["accuracy"] - baseline["accuracy"],
                    **{key: current["groups"][key]["accuracy"] - baseline["groups"][key]["accuracy"]
                       for key in ("Discriminative", "Spatial", "Limited", "Rejection")}}}
        if model["variant"] != "full" or model["task"] == "gui":
            continue
        reference = by_identity.get((model["task"], model["size"], "lora"), {})
        baseline = reference.get("reasonseg_benchmarks", {})
        current = model["reasonseg_benchmarks"]
        matched = (reference.get("checks", {}).get("reasonseg_model_identity") is True
                   and model["checks"]["reasonseg_model_identity"] is True
                   and set(baseline) == set(current) == {"reasonseg_val", "reasonseg_test"})
        if matched:
            matched = all((baseline[name].get("samples") or 0) > 0 and baseline[name].get("data_sha256") and
                          baseline[name].get("samples") == current[name].get("samples") and
                          baseline[name].get("data_sha256") == current[name].get("data_sha256") for name in current)
        model["checks"]["reasonseg_comparison_data"] = matched
        if matched:
            for name, values in current.items():
                model["reasonseg_comparison"][name] = {"full_minus_lora": {
                    metric: values[metric] - baseline[name][metric] for metric in
                    ("accuracy", "mIoU", "gIoU", "cIoU", "precision_at_05") if metric in values and metric in baseline[name]}}
        model["deliverables_ready"] = all(model["checks"].values())
    expected_models = {(task, size, variant) for task in ("det", "seg", "gui")
                       for size, variant in [(size, "lora") for size in ("0.8B", "2B", "4B", "9B")]
                       + [(size, "full") for size in ("0.8B", "2B")]}
    for model in models:
        if model["variant"]=="rlcd":
            reference=by_identity.get((model["task"],model["size"],"full"),{})
            lora=by_identity.get((model["task"],model["size"],"lora"),{})
            model["checks"]["accepted_baselines"]=(lora.get("deliverables_ready") is True and
                (model.get("initialization_variant")=="lora_merged" or reference.get("deliverables_ready") is True))
            model["deliverables_ready"]=all(model["checks"].values())
    for comparison in state.get("additional_full_comparisons", []):
        task, size = comparison["task"], comparison["size"]
        if task not in {"det", "seg", "gui"} or size not in {"0.8B", "2B", "4B", "9B"}:
            raise ValueError("Unknown additional full-training comparison")
        expected_models.add((task, size, "full"))
    for size in state.get("det_rlcd_plan", {}).get("sizes", []):
        if size not in {"0.8B", "2B", "4B"}:
            raise ValueError("Unknown Det full-RLCD size")
        expected_models.add(("det", size, "rlcd"))
    scope_complete = set(by_identity) == expected_models and len(models) == len(expected_models)
    return {"format": "valen_spatial_results_v1", "updated_utc": datetime.now(timezone.utc).isoformat(),
        "models": models, "completed_models": sum(model["deliverables_ready"] for model in models),
        "expected_models": len(expected_models), "model_scope_complete": scope_complete,
        "deliverables_ready": scope_complete and all(model["deliverables_ready"] for model in models),
        "notes": ["Current Det training uses RefCOCO-family and ReasonSeg only; COCO category-detection data and evaluation are retired. Legacy COCO results retain their recorded protocol.",
            "Benchmark metric values are fractions; tables display percentages. Training seconds sum recorded process segments; shared warmup is charged once.",
            "Deliverable checks cover recorded artifacts; final acceptance also verifies model-file hashes and public portability."]}


def percent(value):
    return "—" if value is None else f"{value * 100:.2f}"


def markdown(report):
    lines = ["# Valen spatial results", "", f"Updated: {report['updated_utc']}", "",
        f"Complete recorded artifact sets: {report['completed_models']}/{report['expected_models']}. Metrics below are percentages.", "",
        report["notes"][0], "",
        "## Training and inference cost", "", "| Model | Status | Joint trainable / total parameters | Joint hours | Peak GPU GiB | Weight GiB |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for model in report["models"]:
        joint = model["joint"]
        parameters = (f"{joint['trainable_parameters']:,} / {joint['total_parameters']:,}"
                      if joint.get("total_parameters") is not None else "—")
        hours = f"{joint['recorded_training_seconds']/3600:.2f}" if joint.get("recorded_training_seconds") is not None else "—"
        peak = f"{joint['peak_cuda_bytes_all_ranks']/1024**3:.2f}" if joint.get("peak_cuda_bytes_all_ranks") is not None else "—"
        weights = f"{model['model_weight_bytes']/1024**3:.2f}" if model["model_weight_bytes"] else "—"
        status = model["state"] if model.get("skipped") else "complete" if model["deliverables_ready"] else model["state"]
        lines.append(f"| {model['name']} | {status} | {parameters} | {hours} | {peak} | {weights} |")
    for task in ("det", "seg", "gui"):
        models = [model for model in report["models"] if model["task"] == task]
        if not models:
            continue
        lines.extend(["", "## " + task.upper(), "", "| Model | Benchmark | Metric | Value (%) | Mean latency (s) |",
                      "| --- | --- | --- | ---: | ---: |"])
        overlap_notes=[]
        for model in models:
            metrics = {**model["benchmarks"], **model["reasonseg_benchmarks"], **model["groundingme_benchmarks"],
                       **model.get('additional_gui_benchmarks', {})}
            if not metrics:
                lines.append(f"| {model['name']} | pending | — | — | — |")
            for benchmark, values in sorted(metrics.items()):
                for metric in ("AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "accuracy", "mIoU", "gIoU", "cIoU", "precision_at_05"):
                    if metric not in values:
                        continue
                    latency = values.get("latency_seconds", {}).get("mean")
                    latency = "—" if latency is None else f"{latency:.4f}"
                    lines.append(f"| {model['name']} | {benchmark} | {metric} | {percent(values[metric])} | {latency} |")
            for metric,value in model.get('additional_gui_summary', {}).items():
                lines.append(f"| {model['name']} | official aggregate | {metric} | {percent(value)} | — |")
            for benchmark,values in model.get('additional_gui_benchmarks', {}).items():
                overlap_notes.append(f"{model['name']} / {benchmark}: training duplicate/near-duplicate audit matches {values.get('training_overlap_images', 'unknown')} images / {values.get('training_overlap_records', 'unknown')} queries; full official split retained. Match types and scores by overlap group are in the JSON report.")
        if overlap_notes:
            lines.extend(['',*overlap_notes,''])
    lines.extend(["", "## LoRA / full comparison", "", "| Model | Benchmark | Metric | Full − LoRA (percentage points) |",
                  "| --- | --- | --- | ---: |"])
    for model in report["models"]:
        for benchmark, values in {**model["comparison"].get("benchmarks", {}), **model["reasonseg_comparison"],
                                  **model["groundingme_comparison"]}.items():
            for metric, delta in values.get("full_minus_lora", {}).items():
                lines.append(f"| {model['name']} | {benchmark} | {metric} | {delta * 100:+.2f} |")
    lines.extend(["", "Complete benchmark groups, identities, settings and cost measurements are retained in `summary.json`.", ""])
    return "\n".join(lines)


def write_summary(state, data_root, model_root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = collect_results(state, data_root, model_root)
    for filename, text in (("summary.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                           ("summary.md", markdown(report))):
        destination = output / filename
        temporary = output / (filename + ".tmp")
        temporary.write_text(text)
        temporary.replace(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = write_summary(read_json(Path(args.registry)), args.data_root, args.model_root, args.output_dir)
    print(json.dumps({"models": len(report["models"]), "completed": report["completed_models"]}))


if __name__ == "__main__":
    main()
