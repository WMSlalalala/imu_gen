#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713
PYTHON=/home/mwang49/miniconda3/envs/hml/bin/python
CONFIG="$PROJECT/orchestration/formal_pipeline_config.json"
RUN_ROOT="$PROJECT/results/formal_100epoch_100k_20260713"

# PyTorch/CuBLAS must see this before any CUDA handle is created.  Export it
# at the process-tree root so every gate child inherits the same deterministic
# GEMM workspace contract, even when a child imports torch before engine.py.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Gates are the review phase of an explicit false -> true authorization
# transition.  Refuse an already-authorized config before creating run state.
"$PYTHON" -c '
import json,sys
cfg=json.load(open(sys.argv[1], encoding="utf-8"))
if cfg.get("formal_launch_authorized") is not False:
    raise SystemExit("gates-only launch blocked: formal_launch_authorized must be false")
' "$CONFIG"

mkdir -p "$RUN_ROOT/logs"
extra_args=()
if [[ "${RESUME_FAILED:-0}" == "1" ]]; then
  extra_args+=(--resume-failed)
fi
setsid "$PYTHON" -u "$PROJECT/orchestration/formal_supervisor.py" \
  --config "$CONFIG" --gates-only "${extra_args[@]}" \
  >> "$RUN_ROOT/logs/supervisor_gates_only.log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > "$RUN_ROOT/supervisor_gates_only.pid"
printf 'gate-only supervisor started: pid=%s status=%s\n' \
  "$pid" "$RUN_ROOT/supervisor_status.json"
