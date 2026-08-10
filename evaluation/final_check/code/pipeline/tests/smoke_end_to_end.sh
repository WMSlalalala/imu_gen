#!/bin/bash
# End-to-end smoke: run every builder on a miniature dataset, then verify it.
#
# The unit suite (test_smoke.py) checks each piece in isolation.  What it cannot
# see is the join: a builder that loads the wrong bank, a verifier that passes a
# dataset it never actually read, a summariser that reports a declined action as
# a result.  Those only show up when the real scripts run against a real shard.
#
# Three shards out of a hundred is enough -- it is the code path that is being
# tested here, not the numbers -- and the whole thing finishes in about a minute
# on CPU.  Nothing here touches the released artefacts: everything is written
# under a scratch directory and removed at the end unless KEEP=1 is set.
set -u

C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
# v16, not v12: v12's scroll and swipe fake events carry a deterministic elapsed
# column, and verify_harness check [5] rejects any dataset built on it -- which
# is the point of that check, so the smoke test must not be built on one.
SOURCE=${SOURCE:-/mnt/share/mwang49/data7/results/direct100k/replay_dataset_v16}
PY=${PY:-/home/mwang49/miniconda3/envs/cuhkx/bin/python}
WORK=${WORK:-/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/e2e_smoke}
SHARDS=${SHARDS:-3}

pass=0
fail=0
report() {  # report <name> <exit-status> [detail]
  if [ "$2" -eq 0 ]; then
    echo "  PASS  $1"
    pass=$((pass + 1))
  else
    echo "  FAIL  $1 ${3:-}"
    fail=$((fail + 1))
  fi
}

echo "=== building a ${SHARDS}-shard miniature of $(basename "$SOURCE") ==="
rm -rf "$WORK"
mkdir -p "$WORK/mini/shards"
for shard in $(ls "$SOURCE/shards" | head -"$SHARDS"); do
  cp "$SOURCE/shards/$shard" "$WORK/mini/shards/"
done
cp "$SOURCE/release.json" "$WORK/mini/" 2>/dev/null
cp "$SOURCE/provenance.jsonl" "$WORK/mini/" 2>/dev/null
# The manifest holds one record per split, each naming every shard that split
# owns, and finalise_dataset refuses to rewrite a manifest naming shards that
# were never built -- correctly, since that is what an incomplete build looks
# like.  Each record's shard list is subset to what the miniature actually has,
# and the event counts recomputed, so the manifest stays internally consistent
# and the digest-rewrite path is genuinely exercised rather than skipped.
"$PY" - "$SOURCE/event_manifest.jsonl" "$WORK/mini" <<'PY'
import json, sys
from pathlib import Path
manifest, mini = Path(sys.argv[1]), Path(sys.argv[2])
present = {p.name for p in (mini / "shards").glob("*.npz")}
lines, total = [], 0
for raw in manifest.read_text().splitlines():
    if not raw.strip():
        continue
    record = json.loads(raw)
    shards = [s for s in record.get("shards", []) if Path(s["source"]).name in present]
    if not shards:
        continue
    record["shards"] = shards
    for field, key in (("events", "events"), ("fake_events", "fake"),
                       ("genuine_events", "genuine")):
        if field in record:
            record[field] = sum(int(s.get(key, 0)) for s in shards)
    if "user_ids" in record:
        record["user_ids"] = [s["user_id"] for s in shards if "user_id" in s]
    total += len(shards)
    lines.append(json.dumps(record, sort_keys=True))
(mini / "event_manifest.jsonl").write_text("\n".join(lines) + "\n")
print(f"  manifest: {len(lines)} split record(s) covering {total} shard entries")
PY
echo "  $(ls "$WORK/mini/shards" | wc -l) shards, $(du -sh "$WORK/mini" | cut -f1)"

cd "$C" || exit 1
BIND="$B/fake_target_binding_v12.pkl"

echo
echo "=== 1. sample-bank builder (the path every generative baseline takes) ==="
$PY build_sample_bank_baseline.py --source-dir "$WORK/mini" \
  --output-dir "$WORK/bank_imu" --banks "$B/bank_ttsgan_imu.pkl" \
  --binding "$BIND" --method-name smoke_tts_gan_imu --workers 4 \
  > "$WORK/bank_imu.log" 2>&1
report "build_sample_bank_baseline (imu)" $? "-- see $WORK/bank_imu.log"

echo
echo "=== 2. the same builder declining an action it does not model ==="
$PY build_sample_bank_baseline.py --source-dir "$WORK/mini" \
  --output-dir "$WORK/bank_declined" --banks "$B/bank_ttsgan_imu.pkl" \
  --binding "$BIND" --method-name smoke_declined --decline keystroke \
  --workers 4 > "$WORK/bank_declined.log" 2>&1
report "build_sample_bank_baseline (--decline keystroke)" $?
$PY - "$WORK/bank_declined" <<'PY'
import json, sys
counts = json.loads((__import__("pathlib").Path(sys.argv[1]) / "baseline_counts.json").read_text())
keystroke = counts["per_action"].get("keystroke", {})
assert keystroke.get("fake", 0) > 0, "keystroke had no fake events to decline"
assert keystroke.get("imu_swapped", 0) == 0, f"declined action was still swapped: {keystroke}"
assert keystroke.get("imu_declined", 0) == keystroke["fake"], f"partial decline: {keystroke}"
other = [a for a, t in counts["per_action"].items() if a != "keystroke" and t.get("imu_swapped", 0) > 0]
assert other, "declining one action stopped every other action too"
PY
report "  a declined action is recorded, not silently filled" $?

echo
echo "=== 3. verifier accepts each built dataset ==="
for name in bank_imu; do
  [ -d "$WORK/$name" ] || continue
  $PY verify_harness.py --source-dir "$WORK/mini" --built-dir "$WORK/$name" \
    --kind imu --routing-shards 2 > "$WORK/verify_$name.log" 2>&1
  report "verify_harness $name" $? "-- see $WORK/verify_$name.log"
done

echo
echo "=== 4. verifier REJECTS a dataset whose genuine rows were touched ==="
# A verifier is only worth running if it can fail.  Corrupt one genuine event
# and confirm the check that is supposed to notice actually does.
cp -r "$WORK/bank_imu" "$WORK/corrupt"
$PY - "$WORK/corrupt" <<'PY'
import numpy as np, sys
from pathlib import Path
shard = sorted((Path(sys.argv[1]) / "shards").glob("*.npz"))[0]
arrays = dict(np.load(shard, allow_pickle=True))
genuine = np.flatnonzero(arrays["label"] == 0)
assert len(genuine), "this shard has no genuine events to corrupt"
start = int(arrays["offsets"][genuine[0]])
arrays["imu_flat"][start] += 1.0
np.savez_compressed(shard, **arrays)
print(f"corrupted one genuine row in {shard.name}")
PY
$PY verify_harness.py --source-dir "$WORK/mini" --built-dir "$WORK/corrupt" \
  --kind imu --routing-shards 2 > "$WORK/verify_corrupt.log" 2>&1
if [ $? -ne 0 ]; then
  report "verify_harness rejects corrupted genuine rows" 0
else
  report "verify_harness rejects corrupted genuine rows" 1 \
    "-- IT PASSED A CORRUPTED DATASET, see $WORK/verify_corrupt.log"
fi

echo
echo "=== 5. summariser refuses to report a declined action ==="
$PY summarise_far5.py "$B/detectors_diffts_imu/cells" \
  --dataset "$WORK/bank_declined" > "$WORK/summary.log" 2>&1
report "summarise_far5 runs with --dataset" $? "-- see $WORK/summary.log"
grep -qi "declin\|keystroke" "$WORK/summary.log"
report "  its output names the declined action" $?

echo
echo "=== 6. every module imports cleanly ==="
for module in hmog_baseline_common hmog_event_builder convergence final_release \
              ghost_cursor_path verify_harness summarise_far5 build_comparison \
              score_generator_quality; do
  $PY -c "import $module" 2>"$WORK/import_$module.log"
  report "import $module" $?
done

echo
echo "=== $pass passed, $fail failed ==="
[ "${KEEP:-0}" = "1" ] || { [ "$fail" -eq 0 ] && rm -rf "$WORK"; }
[ "$fail" -eq 0 ]
