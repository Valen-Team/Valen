"""Save a few fixed qualitative predictions without replacing benchmark evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw
import torch

from evaluation.spatial import load_spatial_model
from valen.data.spatial import decode_mask, decode_segmentation_target, image_path, read_spatial_jsonl
from valen.modeling.qwen.spatial import spatial_response
from valen.training.checkpoint import file_sha256


BENCHMARKS = {
    "det": ("refcoco_testA", "refcocoplus_testB", "refcocog_test", "reasonseg_test"),
    "seg": ("refcoco_testA", "phrasecut_test", "refcocog_test"),
    "gui": ("screenspot_test", "screenspot_v2_test", "screenspot_pro_test"),
}


def select_records(records, count):
    """Use stable IDs alone; prediction quality and ground truth never select examples."""
    return sorted(records, key=lambda row: (hashlib.sha256(str(row["id"]).encode()).hexdigest(), str(row["id"])))[:count]


def overlay(image, boxes, mask, color):
    panel = image.convert("RGB").copy()
    if mask is not None:
        pixels = np.asarray(mask, dtype=bool)
        if pixels.shape != (image.height, image.width):
            raise ValueError("Qualitative mask is not aligned to the original image")
        tint = Image.new("RGB", panel.size, color)
        panel = Image.composite(Image.blend(panel, tint, .4), panel, Image.fromarray(pixels.astype("uint8") * 255))
    draw = ImageDraw.Draw(panel)
    for box in boxes:
        draw.rectangle([box[0] * image.width, box[1] * image.height,
                        box[2] * image.width, box[3] * image.height], outline=color, width=max(2, image.width // 300))
    panel.thumbnail((560, 360))
    return panel


@torch.inference_mode()
def save_examples(task, data_root, model_path, output, benchmarks=None, count=1, device="cuda"):
    if not 1 <= count <= 3:
        raise ValueError("Keep one to three fixed examples per benchmark")
    data_root, model_path, output = map(Path, (data_root, model_path, output))
    model, compiler = load_spatial_model(model_path, device, data_root=data_root)
    rows, panels = [], []
    for name in benchmarks or BENCHMARKS[task]:
        base = (data_root / "reasonseg" if name.startswith("reasonseg_") else
                data_root / "protocols/coco_prompt" if name == "coco2017_conditional_val2017" else data_root)
        path = base / "processed" / (name + ".jsonl")
        records = select_records(read_spatial_jsonl(path, labeled_only=True), count)
        data_sha = file_sha256(path)
        for record in records:
            if name.startswith("reasonseg_"):
                record = json.loads(json.dumps(record))
                record["request"]["state"]["messages"][0]["content"][0]["image_url"]["url"] = "reasonseg/" + image_path(record)
            question = compiler.compile(record).questions[0]
            if question.kind != task:
                raise ValueError("Example record belongs to a different task")
            prediction = spatial_response(question, model(question), score_threshold=0.0 if name.startswith("coco2017_") else .05)
            with Image.open(data_root / image_path(record)) as source:
                image = source.convert("RGB")
            truth = question.target
            truth_mask = predicted_mask = None
            if task == "seg":
                target_mask, valid = decode_segmentation_target(truth, data_root)
                truth_mask = (target_mask.bool() & valid).numpy()
                predicted_mask = (decode_mask(prediction["mask"]).bool() & valid).numpy()
            truth_panel = overlay(image, truth.get("boxes", []), truth_mask, "lime")
            predicted_panel = overlay(image, [box["bbox"] for box in prediction.get("boxes", [])], predicted_mask, "red")
            question_spec = next(iter(record["request"]["questions"].values()))
            caption = "\n".join(textwrap.wrap(name + ": " + question_spec["instructions"], 120)[:3])
            canvas = Image.new("RGB", (1120, max(truth_panel.height, predicted_panel.height) + 85), "white")
            canvas.paste(truth_panel, (0, 65))
            canvas.paste(predicted_panel, (560, 65))
            draw = ImageDraw.Draw(canvas)
            draw.text((5, 3), caption, fill="black")
            draw.text((5, 50), "Ground truth", fill="black")
            draw.text((565, 50), "Prediction", fill="black")
            panels.append(canvas)
            rows.append({"benchmark": name, "data_sha256": data_sha, "record": record, "prediction": prediction})
    sheet = Image.new("RGB", (1120, sum(panel.height for panel in panels)), "white")
    top = 0
    for panel in panels:
        sheet.paste(panel, (0, top))
        top += panel.height
    output.mkdir(parents=True, exist_ok=True)
    sheet.save(output / "examples.jpg", quality=85)
    report = {"format": "valen_spatial_examples_v1", "model": model_path.name, "task": task,
              "created_utc": datetime.now(timezone.utc).isoformat(), "sampling": "smallest_sha256_of_record_id",
              "purpose": "qualitative examples only; complete benchmark scores are reported separately",
              "model_export_manifest_sha256": file_sha256(model_path / "export_manifest.json"),
              "examples": rows}
    (output / "examples.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "pipeline_complete.json").write_text(json.dumps({"model": model_path.name, "examples": len(rows)}) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("det", "seg", "gui"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--samples-per-benchmark", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--launch-id", help="Coordinate the existing replicas when called from an allocation queue")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    args = parser.parse_args()
    if args.launch_id:
        from valen.training.spatial_cluster import SpatialCluster
        cluster = SpatialCluster(Path(args.output_dir) / "coordination" / args.launch_id, args.nodes, args.gpus_per_node)
        command = ["-m", "evaluation.examples", "--task", args.task, "--data-root", args.data_root,
                   "--model-path", args.model_path, "--output-dir", args.output_dir, "--device", args.device,
                   "--samples-per-benchmark", str(args.samples_per_benchmark)]
        for benchmark in args.benchmark or []:
            command.extend(["--benchmark", benchmark])
        cluster.execute(command)
    else:
        report = save_examples(args.task, args.data_root, args.model_path, args.output_dir,
                               args.benchmark, args.samples_per_benchmark, args.device)
        print(json.dumps({"model": report["model"], "examples": len(report["examples"])}))


if __name__ == "__main__":
    main()
