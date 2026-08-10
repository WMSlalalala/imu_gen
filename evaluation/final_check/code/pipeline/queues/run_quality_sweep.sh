#!/bin/bash
# Score every generated sample bank with the metrics its own literature uses,
# alongside a real-versus-real control computed on the same windows.
#
# Runs on CPU in a TensorFlow environment kept separate from the training one:
# the metric code is the authors' TF1-style graph code, and mixing it into the
# torch environment would mean pinning versions for one of the two.
set -u
R=/mnt/share/mwang49/data7/results/direct100k/baselines
C=/mnt/share/mwang49/data7/code/baselines
PY=/home/mwang49/.conda/envs/tsmetric/bin/python
REAL=$R/real_windows
SAMPLES_DIR=$1        # e.g. $R/diffts
OUT=$2                # e.g. $R/quality_diffts
KINDS=${3:-"trajectory imu"}
mkdir -p "$OUT"
cd "$C" || exit 1

for kind in $KINDS; do
  for action in tap pinch swipe scroll keystroke; do
    gen="$SAMPLES_DIR/samples_${action}_${kind}.npy"
    [ -f "$gen" ] || { echo "skip ${action}/${kind}: no samples"; continue; }
    echo "== ${action}/${kind}"
    TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3 $PY score_generator_quality.py \
      --real "$REAL/real_train_${action}_${kind}.npy" \
      --generated "$gen" \
      --out "$OUT/quality_${action}_${kind}.json" \
      --limit 1000 --repeats 3 2>&1 | grep -E "^(control|generated)" || true
  done
done
echo "quality sweep done -> $OUT"
