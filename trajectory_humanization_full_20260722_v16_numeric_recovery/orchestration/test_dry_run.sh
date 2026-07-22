#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713
PYTHON=/home/mwang49/miniconda3/envs/hml/bin/python
CONFIG="$PROJECT/orchestration/formal_pipeline_config.json"
OUTPUT=$(mktemp /tmp/trajectory_formal_dry_run.XXXXXX.json)
trap 'rm -f "$OUTPUT"' EXIT

bash -n "$PROJECT/orchestration/launch_formal_pipeline.sh"
set +e
"$PYTHON" "$PROJECT/orchestration/formal_supervisor.py" --config "$CONFIG" --dry-run > "$OUTPUT"
rc=$?
set -e

# Exit 2 is expected while another agent is still completing a required CLI;
# JSON still has to parse and prove that no formal work was started.
if [[ $rc -ne 0 && $rc -ne 2 ]]; then
  exit "$rc"
fi
"$PYTHON" -c '
import json,sys
x=json.load(open(sys.argv[1]))
m=x["command_manifest"]
c=m["commands"]
assert x["formal_work_started"] is False
assert x["preflight"]["formal_launch_authorized"] is False
assert x["preflight"]["ready_to_launch"] is False
assert m["formal_invariants"]["total_fake"] == 100000
assert m["formal_invariants"]["reference_seed"] == 42
assert m["formal_invariants"]["generation_seed"] == 20260713
assert len(c["training"]) == 5
assert len(c["detector_deep_probes"]) == 10
assert len(c["detector_pair_templates"]) == 25
assert all("run_trajectory_benchmark.py" not in " ".join(row["command"]) for row in c["detector_pair_templates"].values())
condition=c["condition_preflight"]
assert condition[condition.index("--batch-size")+1] == "32"
audit=c["generation_audit"]
assert "--condition-preflight" in audit
' "$OUTPUT"
printf 'dry-run contract parsed successfully (supervisor rc=%s; no formal work started)\n' "$rc"
