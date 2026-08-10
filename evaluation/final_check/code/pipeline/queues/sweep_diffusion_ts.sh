#!/bin/bash
# Fit and sample one Diffusion-TS per (action, channel set).  Five jobs share a
# card: each model is small enough that one alone leaves an A6000 idle.
set -u
R=/mnt/share/mwang49/data7/results/direct100k
C=/mnt/share/mwang49/data7/code/baselines
OUT=$R/baselines/diffts
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
STEPS=${STEPS:-12000}
SAMPLES=${SAMPLES:-4000}
mkdir -p "$OUT"
cd "$C" || exit 1

launch() {
  local action=$1 kind=$2 gpu=$3
  $PY run_diffusion_ts.py \
    --dataset-dir "$R/replay_dataset_zoh" \
    --output-dir "$OUT" \
    --action "$action" --kind "$kind" \
    --steps "$STEPS" --samples "$SAMPLES" --sample-batch 256 \
    --gpu "$gpu" > "$OUT/log_${action}_${kind}.txt" 2>&1 &
}

for kind in trajectory imu; do
  gpu=0
  [ "$kind" = imu ] && gpu=1
  for action in tap pinch swipe scroll keystroke; do
    launch "$action" "$kind" "$gpu"
  done
done
wait
echo "sweep done"
ls -la "$OUT"/samples_*.npy
