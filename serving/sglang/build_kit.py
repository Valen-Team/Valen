"""Bundle a plugin wheel with its client, server scripts, and documentation."""

import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("wheel", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if not args.wheel.name.startswith("valen_sglang-") or args.wheel.suffix != ".whl":
    parser.error("expected a built valen_sglang wheel")
root = Path(__file__).resolve().parent
args.output.mkdir(parents=True, exist_ok=False)
for name in [
    "README.md",
    "README_zh.md",
    "LICENSE",
    "NOTICE",
    "install.sh",
    "serve.sh",
    "example_client.py",
]:
    shutil.copy2(root / name, args.output / name)
shutil.copy2(args.wheel, args.output / args.wheel.name)
print(args.output.resolve())
