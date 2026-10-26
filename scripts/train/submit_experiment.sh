#!/usr/bin/env bash
set -euo pipefail
VALEN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$VALEN_ROOT/scripts/train/cluster_env.sh"
VALEN_NAME="${VALEN_RUN_NAME:?Set a unique VALEN_RUN_NAME}"
VALEN_GPUS="${VALEN_GPUS:-8}"
VALEN_NODES="${VALEN_NODES:-1}"
VALEN_CPUS="${VALEN_CPUS:-$((VALEN_GPUS * 8))}"
VALEN_MEMORY_MIB="${VALEN_MEMORY_MIB:-$((VALEN_GPUS * 65536))}"
VALEN_EXPERIMENT_ROOT="${VALEN_EXPERIMENT_ROOT:-$VALEN_ROOT/next-generation-exp}"
VALEN_LOG_DIR="$VALEN_EXPERIMENT_ROOT/jobs/$VALEN_NAME"
mkdir -p "$VALEN_LOG_DIR"
"$RJOB_BIN" submit --name="$VALEN_NAME" --gpu="$VALEN_GPUS" --cpu="$VALEN_CPUS" --memory="$VALEN_MEMORY_MIB" \
  --charged-group="$RJOB_CHARGED_GROUP" --namespace="$RJOB_NAMESPACE" \
  --private-machine="$RJOB_PRIVATE_MACHINE" "${VALEN_MOUNT_ARGS[@]}" \
  --image="$RJOB_IMAGE" \
  --restart-policy=never --auto-restart=false -P "$VALEN_NODES" --host-network=false \
  -e VALEN_GPUS="$VALEN_GPUS" \
  -e VALEN_EXPERIMENT_ROOT="$VALEN_EXPERIMENT_ROOT" \
  -e VALEN_CODE_ROOT="${VALEN_CODE_ROOT:-$VALEN_ROOT}" \
  "${VALEN_WORKER_ENV[@]}" \
  -- bash "$VALEN_ROOT/scripts/train/experiment_entry.sh" "$VALEN_LOG_DIR" "$@" \
  2>&1 | tee "$VALEN_LOG_DIR/submit.log"
VALEN_ID="$(awk '/created rjob_name:/ {print $NF}' "$VALEN_LOG_DIR/submit.log" | tail -n 1)"
test -n "$VALEN_ID"
printf '%s\n' "$VALEN_ID" > "$VALEN_LOG_DIR/job_id"
if [[ "${VALEN_ENFORCE_RESOURCES:-0}" == 1 ]]; then
  "${VALEN_PYTHON:-$VALEN_ROOT/.venv/bin/python}" -m scripts.train.normalize_job_resources \
    --job-id "$VALEN_ID" --gpus "$VALEN_GPUS" --cpus "$VALEN_CPUS" \
    --memory-mib "$VALEN_MEMORY_MIB" --namespace "$RJOB_NAMESPACE" --replicas "$VALEN_NODES"
fi
printf 'Job: %s\nLogs: %s\n' "$VALEN_ID" "$VALEN_LOG_DIR"
