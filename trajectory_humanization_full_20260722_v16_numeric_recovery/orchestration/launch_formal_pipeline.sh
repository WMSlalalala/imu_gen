#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713
PYTHON=/home/mwang49/miniconda3/envs/hml/bin/python
CONFIG="$PROJECT/orchestration/formal_pipeline_config.json"
RUN_ROOT="$PROJECT/results/formal_100epoch_100k_20260713"

# Bind the complete detached supervisor tree to deterministic CuBLAS GEMM.
# This is deliberately exported before the first Python/PyTorch process.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Fail before creating formal logs unless both the reviewed operational flag
# and a durable gates-only evidence record are present.  The Python
# supervisor re-hashes all three gate artifacts before starting formal work.
"$PYTHON" -c '
import json,pathlib,sys
cfg=json.load(open(sys.argv[1], encoding="utf-8"))
if cfg.get("formal_launch_authorized") is not True:
    raise SystemExit("formal launch blocked: review gates, then set formal_launch_authorized=true")
state_path=pathlib.Path(cfg["run_root"]) / "supervisor_status.json"
if not state_path.is_file():
    raise SystemExit("formal launch blocked: run --gates-only and review its evidence first")
state=json.load(open(state_path, encoding="utf-8"))
evidence=state.get("launch_gate_evidence", {})
stages=state.get("stages", {})
if not (
    evidence.get("schema_version") == "trajectory_launch_gate_evidence_v1"
    and evidence.get("formal_launch_authorized_during_gates") is False
    and state.get("config_sha256") == evidence.get("config_sha256")
    and float(state.get("gates_completed_unix_time", 0.0)) > 0.0
    and all(stages.get(name, {}).get("status") == "complete"
            for name in ("corpus_audit", "e2e_smoke", "condition_preflight"))
):
    raise SystemExit("formal launch blocked: durable gates-only review evidence is missing")
' "$CONFIG"

mkdir -p "$RUN_ROOT/logs"

extra_args=()
if [[ "${RESUME_FAILED:-0}" == "1" ]]; then
  extra_args+=(--resume-failed)
fi

# The supervisor itself is detached from the VS Code terminal.  Every formal
# child also gets a separate session and writes only one key log per job.
setsid "$PYTHON" -u "$PROJECT/orchestration/formal_supervisor.py" \
  --config "$CONFIG" "${extra_args[@]}" \
  >> "$RUN_ROOT/logs/supervisor.log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > "$RUN_ROOT/supervisor.pid"
printf 'formal supervisor started: pid=%s status=%s\n' "$pid" "$RUN_ROOT/supervisor_status.json"
