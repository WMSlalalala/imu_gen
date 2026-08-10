#!/bin/bash
# Reproduce one learned baseline end to end: fit, sample, build, verify, score.
#
#   GENERATOR=diffusion_ts|tts_gan KIND=trajectory|imu bash run.sh <carrier-dataset-dir> <work-dir>
#
# Five models are fitted, one per action, on the train users' genuine events
# only.  Each is sampled into a bank, the bank fills the fake carriers, the
# harness is verified, the detector grid runs, and the generator's own quality
# metrics are computed against a real-versus-real control.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SOURCE=${1:?carrier dataset dir}
WORK=${2:?work dir}
GENERATOR=${GENERATOR:-diffusion_ts}
KIND=${KIND:-imu}
PY=${PY:-/home/mwang49/miniconda3/envs/cuhkx/bin/python}
METRIC_PY=${METRIC_PY:-/home/mwang49/.conda/envs/tsmetric/bin/python}
DETECTOR_PY=${DETECTOR_PY:-/home/mwang49/miniconda3/envs/hml/bin/python}
DETECTOR_CODE=${DETECTOR_CODE:-/mnt/share/mwang49/data7/code/direct100k}
SAMPLES=${SAMPLES:-4000}
mkdir -p "$WORK/samples" "$WORK/results"

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

echo "== fitting and sampling five $GENERATOR models ($KIND)"
GPU=0
for action in tap pinch swipe scroll keystroke; do
  if [ -f "$WORK/samples/samples_${action}_${KIND}.npy" ]; then
    echo "   $action already sampled"; continue
  fi
  case $GENERATOR in
    diffusion_ts)
      $PY "$HERE/run_diffusion_ts.py" --dataset-dir "$SOURCE" \
        --output-dir "$WORK/samples" --action "$action" --kind "$KIND" \
        --steps "${STEPS:-12000}" --samples "$SAMPLES" --sample-batch 256 \
        --gpu "$GPU" > "$WORK/samples/log_${action}_${KIND}.txt" 2>&1 &
      ;;
    tts_gan)
      $PY "$HERE/run_tts_gan.py" --dataset-dir "$SOURCE" \
        --output-dir "$WORK/samples" --action "$action" --kind "$KIND" \
        --max-iter "${MAX_ITER:-20000}" --samples "$SAMPLES" --sample-batch 500 \
        --gpu "$GPU" > "$WORK/samples/log_${action}_${KIND}.txt" 2>&1 &
      ;;
    *) echo "unknown generator $GENERATOR"; exit 1 ;;
  esac
  GPU=$((1 - GPU))
done
wait

echo "== assembling the sample bank"
$PY "$HERE/assemble_banks.py" --samples-dir "$WORK/samples" \
  --out "$WORK/bank.pkl" --kind "$KIND"

echo "== building the dataset"
$PY "$HERE/build_sample_bank_baseline.py" \
  --source-dir "$SOURCE" --output-dir "$WORK/dataset" \
  --banks "$WORK/bank.pkl" --binding "$WORK/fake_target_binding.pkl" \
  --method-name "${GENERATOR}_${KIND}" --workers "${WORKERS:-32}"

echo "== verifying nothing else moved"
$PY "$HERE/verify_harness.py" --source-dir "$SOURCE" \
  --built-dir "$WORK/dataset" --kind "$KIND"

MODALITY=trajectory_xytime
[ "$KIND" = imu ] && MODALITY=imu_only
echo "== 30 $MODALITY cells"
rm -rf "$WORK/detectors"
(cd "$DETECTOR_CODE" && $DETECTOR_PY scripts/run_hmog_direct100k_detectors.py \
  --manifest "$WORK/dataset/event_manifest.jsonl" \
  --output-dir "$WORK/detectors" --modality "$MODALITY" \
  --device cuda:0 --device cuda:1 --workers-per-device 3 --cpu-workers 24 \
  --epochs 20 --bootstrap-replicates 10000 --seed 42) > "$WORK/detectors.log" 2>&1
test -f "$WORK/detectors/completion.json" || { echo "grid failed"; tail -30 "$WORK/detectors.log"; exit 1; }

echo "== FAR at the development FRR=5% threshold"
$PY "$HERE/summarise_far5.py" "$WORK/detectors/cells" \
  --dataset "$WORK/dataset" \
  --json-out "$WORK/results/far5.json" | tee "$WORK/results/far5.txt"

echo "== generator quality against a real-versus-real control"
$PY "$HERE/export_real_windows.py" --dataset-dir "$SOURCE" \
  --out-dir "$WORK/real_windows" --split train
for action in tap pinch swipe scroll keystroke; do
  TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3 $METRIC_PY \
    "$HERE/score_generator_quality.py" \
    --real "$WORK/real_windows/real_train_${action}_${KIND}.npy" \
    --generated "$WORK/samples/samples_${action}_${KIND}.npy" \
    --out "$WORK/results/quality_${action}_${KIND}.json" \
    --limit 1000 --repeats 3 2>&1 | grep -E "^(control|generated)" || true
done
echo "done -> $WORK/results"
