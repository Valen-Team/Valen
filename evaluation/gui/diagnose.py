"""Paired native/spatial GUI diagnostics on fixed samples, separate from final scores."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from PIL import Image
import torch

from evaluation.spatial import load_spatial_model
from valen.data.spatial import image_path, read_spatial_jsonl
from valen.modeling.qwen.spatial import decode_spatial
from valen.training.checkpoint import file_sha256
from valen.training.distributed import initialize, close


BENCHMARKS = ("screenspot_pro_test", "screenspot_v2_test")
SYSTEM = (
    "You are a GUI agent. Click the UI element described by the user using the computer_use tool. "
    "Coordinates are normalized independently on each axis to the range 0 to 1000. "
    "Return one tool call and no explanation."
)
TOOLS = [{"type": "function", "function": {"name": "computer_use", "description": "Click a UI element.",
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "enum": ["left_click"]},
        "coordinate": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
        "required": ["action", "coordinate"]}}}]


def select(records, count):
    result = []
    for kind in ("icon", "text"):
        pool = [r for r in records if r["meta"]["ui_type"] == kind]
        if len(pool) < count:
            raise ValueError("Insufficient records for the fixed diagnostic cohort")
        result.extend(sorted(pool, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:count])
    return result


def parse_point(response):
    """Read a tool coordinate or numeric array; never consult the target box."""
    for payload in reversed(re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.S)):
        try:
            value = json.loads(payload)
            coordinates = value.get("arguments", {}).get("coordinate")
            if coordinates is not None:
                return normalized_point(coordinates)
        except (ValueError, TypeError, AttributeError):
            continue
    for payload in reversed(re.findall(r"\[[^\[\]]+\]", response)):
        try:
            point = normalized_point(json.loads(payload))
            if point is not None:
                return point
        except (ValueError, TypeError):
            continue
    return None


def normalized_point(values):
    if not isinstance(values, list) or len(values) not in (2, 4):
        return None
    if any(type(v) not in (int, float) or not 0 <= v <= 1000 for v in values):
        return None
    if len(values) == 4:
        if values[0] >= values[2] or values[1] >= values[3]:
            return None
        values = [(values[0] + values[2]) / 2, (values[1] + values[3]) / 2]
    return [v / 1000 for v in values]


def hit(point, box):
    return bool(point is not None and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3])


def compare_reports(reports):
    identities = [{(r["benchmark"], r["id"], r["max_pixels"], r["data_sha256"]) for r in report["cases"]} for report in reports]
    if (not reports or not all(ids == identities[0] for ids in identities)
            or any(len(ids) != len(report["cases"]) for ids, report in zip(identities, reports))
            or any(r["world_size"] != 8 or len(r.get("gpu_names", [])) != 8
                   or r["gpu_names"] != reports[0]["gpu_names"] for r in reports)
            or len({(r["model"], r["mode"]) for r in reports}) != len(reports)):
        raise ValueError("Diagnostic sample identities or eight-GPU conditions differ")
    table = []
    reference_grids = {(r["benchmark"], r["id"], r["max_pixels"]): r["image_grid_thw"] for r in reports[0]["cases"]}
    for report in reports:
        for benchmark in BENCHMARKS:
            for pixels in report["max_pixels"]:
                cases = [r for r in report["cases"] if r["benchmark"] == benchmark and r["max_pixels"] == pixels]
                table.append(dict(model=report["model"], mode=report["mode"], benchmark=benchmark, max_pixels=pixels,
                    samples=len(cases), hits=sum(r["hit"] for r in cases),
                    invalid_outputs=sum(r["point"] is None for r in cases),
                    image_grid_mismatches=sum(r["image_grid_thw"] != reference_grids[(r["benchmark"], r["id"], pixels)] for r in cases)))
    return dict(purpose="diagnostic subset only; prompt formats differ between native and spatial modes",
                cases_per_model=len(identities[0]), models=len(reports), table=table,
                note="Use each report's input-grid and pixel fingerprints to audit processor differences before attributing model gaps.")


@torch.inference_mode()
def diagnose(args):
    context = initialize(args.device)
    try:
        root, package, output = map(Path, (args.data_root, args.model_path, args.output_dir))
        cohorts = {}
        for benchmark in BENCHMARKS:
            path = root / "processed" / (benchmark + ".jsonl")
            cohorts[benchmark] = (select(read_spatial_jsonl(path, labeled_only=True), args.samples_per_type), file_sha256(path))
        if args.native:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            processor = AutoProcessor.from_pretrained(package, local_files_only=True)
            model = AutoModelForImageTextToText.from_pretrained(package, dtype=torch.bfloat16,
                local_files_only=True, attn_implementation="sdpa").to(context.device).eval().requires_grad_(False)
        else:
            model, compiler = load_spatial_model(package, context.device, data_root=root)
        cases = []
        for benchmark, (records, data_sha) in cohorts.items():
            for record in records[context.rank::context.world_size]:
                truth = record["targets"]["target"]["boxes"][0]
                for pixels in args.max_pixels:
                    extra = {}
                    if args.native:
                        instruction = record["request"]["questions"]["target"]["instructions"]
                        with Image.open(root / image_path(record)) as source:
                            image = source.convert("RGB")
                        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
                            {"type": "image"}, {"type": "text", "text": instruction}]}]
                        text = processor.apply_chat_template(messages, tools=TOOLS, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False)
                        inputs = processor(text=[text], images=[image], return_tensors="pt",
                            max_pixels=pixels, min_pixels=4096).to(context.device)
                        tokens = int(inputs["input_ids"].shape[-1])
                        if tokens > 32768:
                            raise ValueError("Diagnostic input exceeds its declared token budget")
                        generated = model.generate(**inputs, do_sample=False, max_new_tokens=192, use_cache=True)
                        response = processor.tokenizer.decode(generated[0, tokens:], skip_special_tokens=False)
                        point = parse_point(response)
                        grid = inputs["image_grid_thw"][0].tolist()
                        extra = dict(response=response, format_valid=point is not None)
                    else:
                        compiler.max_pixels, compiler.max_length = pixels, 32768
                        question = compiler.compile(record).questions[0]
                        raw = model(question)
                        prediction = decode_spatial(question, raw)
                        point = prediction.get("point")
                        probability = raw.coordinate_logits[0].float().softmax(-1)
                        values = (probability * torch.linspace(0, 1, probability.shape[-1], device=probability.device)).sum(-1)
                        extra = dict(predicted_box=prediction.get("boxes", [{}])[0].get("bbox"),
                            raw_center=values[:2].tolist(), raw_center_hit=hit(values[:2].tolist(), truth),
                            entropy_bits=(-(probability * probability.clamp_min(1e-30).log2()).sum(-1)).tolist())
                        tokens = int(question.inputs["input_ids"].shape[-1])
                        grid = question.inputs["image_grid_thw"][0].tolist()
                        inputs = question.inputs
                    if record["id"] == records[0]["id"]:
                        pixels_tensor = inputs["pixel_values"].detach().float().cpu().contiguous()
                        extra["pixel_values_sha256"] = hashlib.sha256(pixels_tensor.numpy().tobytes()).hexdigest()
                    cases.append(dict(id=record["id"], benchmark=benchmark, ui_type=record["meta"]["ui_type"],
                        data_sha256=data_sha, max_pixels=pixels, input_tokens=tokens, image_grid_thw=grid,
                        truth_box=truth, point=point, hit=hit(point, truth), **extra))
        cases = sorted((r for shard in context.gather(cases) for r in shard), key=lambda r: (r["benchmark"], r["id"], r["max_pixels"]))
        gpu_names = context.gather(torch.cuda.get_device_name() if context.device.startswith("cuda") else "cpu")
        if context.primary:
            expected = len(BENCHMARKS) * 2 * args.samples_per_type * len(args.max_pixels)
            if len(cases) != expected or len({(r["benchmark"], r["id"], r["max_pixels"]) for r in cases}) != expected:
                raise ValueError("Diagnostic cohort coverage is incomplete or duplicated")
            output.mkdir(parents=True, exist_ok=True)
            from valen.training.spatial_cluster import SpatialCluster
            signatures = sorted(package.glob("*_manifest.json")) if args.native else [package / "export_manifest.json"]
            if not signatures or any(not p.is_file() for p in signatures):
                raise ValueError("Diagnostic models require a source or export manifest")
            source = Path(__file__).resolve().parents[2] / "source_manifest.json"
            report = dict(format="valen_gui_paired_diagnostics_v1", model=package.name,
                mode="native" if args.native else "spatial", purpose="diagnostic subset, not final benchmark scores",
                selection="smallest ID SHA256 within each benchmark and UI type", world_size=context.world_size,
                gpu_names=gpu_names,
                max_pixels=args.max_pixels, precision="bf16 backbone, model-defined spatial head precision",
                native_protocol=dict(system=SYSTEM, tools=TOOLS, thinking=False, do_sample=False, max_new_tokens=192) if args.native else None,
                package_manifest_sha256={p.name: file_sha256(p) for p in signatures},
                code_sha256=json.loads(source.read_text())["sha256"] if source.is_file() else file_sha256(Path(__file__)), cases=cases)
            SpatialCluster.write(output / "diagnostics.json", report)
        context.barrier()
    finally:
        close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--model-path")
    models.add_argument("--model-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--samples-per-type", type=int, default=16)
    parser.add_argument("--max-pixels", nargs="+", type=int, default=[1048576, 4194304])
    parser.add_argument("--sizes", nargs="+", choices=("0.8B", "2B", "4B", "9B"), default=["0.8B", "2B", "4B", "9B"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--launch-id", default="gui-diagnostics")
    args = parser.parse_args()
    if (args.samples_per_type <= 0 or any(p <= 0 for p in args.max_pixels)
            or len(set(args.max_pixels)) != len(args.max_pixels) or len(set(args.sizes)) != len(args.sizes)):
        parser.error("Sample counts and unique pixel budgets must be positive")
    if args.model_root:
        from valen.training.spatial_cluster import SpatialCluster
        root, output = Path(args.model_root).resolve(), Path(args.output_dir).resolve()
        cluster = SpatialCluster(output / "coordination" / args.launch_id, args.nodes, args.gpus_per_node)
        for size in args.sizes:
            for mode in ("spatial", "native"):
                path = root / (("Qwen3.5-" if mode == "native" else "Valen-Decision-GUI-") + size)
                destination = output / size / mode
                cluster.execute(["-m", "evaluation.gui.diagnose", "--data-root", args.data_root,
                    "--model-path", str(path), "--output-dir", str(destination), "--samples-per-type", str(args.samples_per_type),
                    "--max-pixels", *map(str, args.max_pixels), *(["--native"] if mode == "native" else [])], distributed=True)
        if cluster.primary:
            reports = [json.loads((output / size / mode / "diagnostics.json").read_text()) for size in args.sizes for mode in ("spatial", "native")]
            summary = compare_reports(reports)
            SpatialCluster.write(output / "paired_summary.json", summary)
            SpatialCluster.write(output / "pipeline_complete.json", dict(models=len(reports), cases_per_model=summary["cases_per_model"], purpose="diagnostics only"))
    else:
        diagnose(args)


if __name__ == "__main__":
    main()
