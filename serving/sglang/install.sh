#!/usr/bin/env bash
set -euo pipefail
KIT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
python3 -m pip install "$KIT"/valen_sglang-*.whl
