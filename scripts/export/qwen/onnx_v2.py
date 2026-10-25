"""Reproducible native Valen -> modular ONNX decision artifact pipeline."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

UPSTREAM_REVISION = "6b7ff3ca3efa15da219050fdb4ecf8bed67875af"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Native merged Valen HF directory")
    parser.add_argument("--output", required=True, help="New or empty output directory")
    parser.add_argument("--upstream", required=True, help="Pinned ONNX Runtime GenAI source checkout")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument(
        "--mlp-bits",
        type=int,
        choices=[4, 8],
        help="Add a candidate MLP-only quantized variant; requires CPU FP32",
    )
    parser.add_argument(
        "--reuse-components",
        help="Optional existing Valen ONNX directory; every vision/Mixer tensor is verified",
    )
    args = parser.parse_args()
    if (args.provider, args.precision) not in {("cpu", "fp32"), ("cuda", "fp16")}:
        parser.error("Validated exports are CPU FP32 and CUDA FP16")
    if args.mlp_bits and (args.provider != "cpu" or args.precision != "fp32"):
        parser.error("MLP-only candidate quantization requires the CPU FP32 base")
    source, output, upstream = (Path(value).resolve() for value in [args.source, args.output, args.upstream])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory must be empty; select a new destination")
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "valen_qwen" or config.get("valen_head", {}).get("head_type") not in {
        "mixer",
        "role_mlp",
    }:
        parser.error("This exporter supports merged Valen Qwen models with Mixer or role-MLP heads")
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != UPSTREAM_REVISION:
        parser.error(f"Expected ONNX Runtime GenAI revision {UPSTREAM_REVISION}; found {actual}")
    scripts = Path(__file__).resolve().parent
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="valen-onnx-export-", dir=output.parent) as temporary:
        work = Path(temporary)
        backbone = work / "backbone"
        precision = args.precision
        subprocess.run(
            [
                sys.executable,
                str(scripts / "onnx_v2_backbone.py"),
                str(source),
                str(backbone),
                "--upstream",
                str(upstream),
                "--provider",
                args.provider,
                "--precision",
                precision,
                "--io-dtype",
                "fp16" if args.provider == "cuda" or precision == "fp16" else "fp32",
            ],
            check=True,
        )
        if args.reuse_components:
            components = Path(args.reuse_components).resolve()
        else:
            components = work / "components"
            subprocess.run(
                [sys.executable, str(scripts / "onnx_v2_components.py"), str(source), str(components)],
                check=True,
            )
        variant = ("cuda-" if args.provider == "cuda" else "") + precision
        command = [
            sys.executable,
            str(scripts / "onnx_v2_graphs.py"),
            str(source),
            str(components),
            str(backbone),
            str(output),
            "--variant",
            variant,
        ]
        subprocess.run(command, check=True)
        if args.mlp_bits:
            subprocess.run(
                [
                    sys.executable,
                    str(scripts / "onnx_v2_quantize.py"),
                    "--model",
                    str(output),
                    "--base-variant",
                    variant,
                    "--bits",
                    str(args.mlp_bits),
                ],
                check=True,
            )
    print(f"Exported {variant} to {output}. Validate complete requests on each target before release.")


if __name__ == "__main__":
    main()
