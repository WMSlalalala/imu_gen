#!/bin/bash
# Reproduce baseline 01 end to end: build, verify, score.
#
#   bash run.sh <carrier-dataset-dir> <work-dir>
#
# <carrier-dataset-dir> is the direct100k release whose fake events get their
# trajectory replaced (genuine events are never touched).  Everything this
# script writes lands under <work-dir>.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SOURCE=${1:?carrier dataset dir}
WORK=${2:?work dir}
PY=${PY:-/home/mwang49/miniconda3/envs/cuhkx/bin/python}
DETECTOR_PY=${DETECTOR_PY:-/home/mwang49/miniconda3/envs/hml/bin/python}
DETECTOR_CODE=${DETECTOR_CODE:-/mnt/share/mwang49/data7/code/direct100k}
mkdir -p "$WORK"

# The bound targets each fake event has to hit.  Cached once from the source
# release's provenance so the build does not reparse 145,776 records per shard.
if [ ! -f "$WORK/fake_target_binding.pkl" ]; then
  echo "== caching bound targets"
  $PY - "$SOURCE" "$WORK/fake_target_binding.pkl" <<'PYTHON'
import json, pickle, sys
source, out = sys.argv[1], sys.argv[2]
binding = {}
with open(f"{source}/provenance.jsonl") as handle:
    for line in handle:
        record = json.loads(line)
        if record.get("label") != 1:
            continue
        target = record["donor"].get("target_binding") or {}
        orientation = target.get("orientation_id")
        binding[record["event_id"]] = (
            int(orientation) if orientation is not None else 0,
            target.get("gesture_requested_start_px"),
            target.get("gesture_requested_end_px"),
        )
with open(out, "wb") as handle:
    pickle.dump(binding, handle)
print(f"cached {len(binding)} bound targets")
PYTHON
fi

echo "== building the pyclick dataset"
$PY "$HERE/build_pyclick_baseline.py" \
  --source-dir "$SOURCE" \
  --output-dir "$WORK/dataset" \
  --binding "$WORK/fake_target_binding.pkl" \
  --workers "${WORKERS:-32}"

echo "== verifying nothing else moved"
$PY "$HERE/verify_harness.py" \
  --source-dir "$SOURCE" \
  --built-dir "$WORK/dataset" --kind trajectory

echo "== 30 trajectory cells"
rm -rf "$WORK/detectors"
(cd "$DETECTOR_CODE" && $DETECTOR_PY scripts/run_hmog_direct100k_detectors.py \
  --manifest "$WORK/dataset/event_manifest.jsonl" \
  --output-dir "$WORK/detectors" \
  --modality trajectory_xytime \
  --device cuda:0 --device cuda:1 --workers-per-device 3 --cpu-workers 24 \
  --epochs 20 --bootstrap-replicates 10000 --seed 42) > "$WORK/detectors.log" 2>&1
test -f "$WORK/detectors/completion.json" || { echo "grid failed"; tail -30 "$WORK/detectors.log"; exit 1; }

echo "== FAR at the development FRR=5% threshold"
mkdir -p "$WORK/results"
$PY "$HERE/summarise_far5.py" "$WORK/detectors/cells" \
  --dataset "$WORK/dataset" \
  --json-out "$WORK/results/far5.json" | tee "$WORK/results/far5.txt"
