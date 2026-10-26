#!/usr/bin/env bash
set -euo pipefail
VALEN_LOG_DIR="$1"
shift
VALEN_NODE_SUFFIX=""
if [[ "${KUBEBRAIN_REPLICA_TOTAL:-1}" != 1 ]]; then
  VALEN_NODE_SUFFIX=".node-${KUBEBRAIN_REPLICA:-0}"
fi
exec > >(tee "$VALEN_LOG_DIR/console$VALEN_NODE_SUFFIX.log") 2>&1
trap 'VALEN_EXIT=$?; printf "EXIT_CODE=%s\n" "$VALEN_EXIT" > "$VALEN_LOG_DIR/status$VALEN_NODE_SUFFIX.log"' EXIT
cd -- "${VALEN_CODE_ROOT:-$(dirname -- "${BASH_SOURCE[0]}")/../..}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${VALEN_EXPERIMENT_ROOT:-$PWD/next-generation-exp}/hf-cache"
export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
nvidia-smi
"${VALEN_PYTHON:-$PWD/.venv/bin/python}" "$@"

# Retain spatial allocations for the queued models.
if [[ -f "$VALEN_LOG_DIR/allocation/queue.json" ]]; then
  "${VALEN_PYTHON:-$PWD/.venv/bin/python}" "$(dirname -- "${BASH_SOURCE[0]}")/spatial_worker.py" \
    --queue "$VALEN_LOG_DIR/allocation/queue.json"
fi

# Follow replacement queues while retaining the same scheduler allocation.
VALEN_LAST_QUEUE=""
while [[ -f "$VALEN_LOG_DIR/handoff/next_queue" ]]; do
  IFS= read -r VALEN_NEXT_QUEUE < "$VALEN_LOG_DIR/handoff/next_queue"
  if [[ -z "$VALEN_NEXT_QUEUE" || "$VALEN_NEXT_QUEUE" == "$VALEN_LAST_QUEUE" ]]; then
    break
  fi
  VALEN_LAST_QUEUE="$VALEN_NEXT_QUEUE"
  "${VALEN_PYTHON:-$PWD/.venv/bin/python}" "$(dirname -- "${BASH_SOURCE[0]}")/spatial_worker.py" \
    --queue "$VALEN_NEXT_QUEUE"
done
