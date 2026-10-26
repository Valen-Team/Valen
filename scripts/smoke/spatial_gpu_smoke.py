"""Exercise the three spatial heads through actual eight-rank Qwen training."""
import argparse
import gc
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from valen.training.distributed import initialize, close
from valen.training.spatial import run
from scripts.export.qwen.export_spatial import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    context = initialize()
    root = Path(args.output_dir)
    try:
        report = {}
        for task in ("det", "seg", "gui"):
            work = root / task
            if context.primary:
                work.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (128, 96), "white")
                ImageDraw.Draw(image).rectangle((32, 24, 96, 72), fill="red")
                image.save(work / "image.png")
                mask = Image.new("L", image.size)
                ImageDraw.Draw(mask).rectangle((32, 24, 96, 72), fill=255)
                mask.save(work / "mask.png")
                records = []
                for i in range(16):
                    target = {"boxes": [[.25, .25, .75, .75]], "labels": [0]}
                    if task == "seg":
                        target["mask"] = "mask.png"
                    records.append({"id": str(i), "group_id": "synthetic-smoke",
                        "request": {"state": {"messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": "image.png"}}]}]},
                            "questions": {"target": {"type": task, "instructions": "Locate the red rectangle.",
                                                      "categories": ["red rectangle"]}}},
                        "targets": {"target": target}})
                (work / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            context.barrier()
            config = dict(architecture="qwen_spatial", task=task, model_path=args.model_path,
                          data=str(work / "train.jsonl"), data_root=str(work), output=str(work / "warmup"),
                          stage="warmup", dtype="bf16", attn_implementation="sdpa", gradient_checkpointing=False,
                          spatial_width=64, spatial_heads=4, spatial_queries=4, spatial_layers=1, coordinate_bins=64,
                          epochs=16, gradient_accumulation=1, max_steps=32, save_every=32,
                          head_lr=1e-3, max_pixels=16384)
            model, compiler, progress = run(config)
            assert progress["step"] == 32
            del model, compiler
            gc.collect()
            torch.cuda.empty_cache()
            # Reload all eight rank/optimizer/RNG states before opening LoRA parameters.
            model, compiler, restored = run(config, resume=work / "warmup/latest")
            assert restored == progress
            del model, compiler
            gc.collect()
            torch.cuda.empty_cache()
            joint = dict(config, stage="joint", output=str(work / "joint"), epochs=2,
                         max_steps=4, save_every=4, lora_rank=8, lora_alpha=16,
                         lora_lr=5e-5, merger_lr=1e-5, gradient_checkpointing=True)
            model, compiler, _ = run(joint, initialize_from=work / "warmup/best")
            del model, compiler
            gc.collect()
            torch.cuda.empty_cache()
            context.barrier()
            if context.primary:
                metrics = [json.loads(line) for line in (work / "warmup/metrics.jsonl").read_text().splitlines()]
                if metrics[-1]["loss"] >= metrics[0]["loss"]:
                    raise AssertionError("Synthetic overfit loss did not decrease")
                result = export(work / "joint/best", work / "export", "Valen-Decision-" + task.title(),
                                work / "train.jsonl", work)
                report[task] = {"first_loss": metrics[0]["loss"], "last_loss": metrics[-1]["loss"],
                                "reload_validation": result["reload_validation"]}
            context.barrier()
        if context.primary:
            (root / "complete.json").write_text(json.dumps({"tasks": report, "ranks": context.world_size}))
    finally:
        close()


if __name__ == "__main__":
    main()
