#!/usr/bin/env bash
set -euo pipefail
revision=()
if [[ -n "${REVISION:-}" ]]; then revision=(--revision "$REVISION"); fi
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
exec python3 -m valen_vllm.launch_server \
  --model-path "${MODEL:-Valen-Team/Valen-Decision-1.5-2B}" "${revision[@]}" \
  --served-model-name "${MODEL_NAME:-valen}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"
