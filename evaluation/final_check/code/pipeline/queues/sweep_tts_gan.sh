#!/bin/bash
# Fit and sample one TTS-GAN per action for the IMU channel set.
#
# The iteration budget is per action rather than global: a transformer GAN's
# step cost grows with the token count (seq_length / patch_size), so a single
# figure would either starve the long actions or waste hours on the short ones.
# Each budget below was chosen to give the model a comparable number of passes
# over its training set; whether that was enough is not asserted here, it is
# measured afterwards by the authors' discriminative score against the
# real-versus-real control.
set -u
R=/mnt/share/mwang49/data7/results/direct100k
C=/mnt/share/mwang49/data7/code/baselines
OUT=$R/baselines/ttsgan
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
KIND=${KIND:-imu}
SAMPLES=${SAMPLES:-4000}
mkdir -p "$OUT"
cd "$C" || exit 1

# action:max_iter:gpu
JOBS=${JOBS:-"tap:30000:0 pinch:20000:1 swipe:15000:0 scroll:15000:1 keystroke:8000:0"}

for job in $JOBS; do
  action=${job%%:*}
  rest=${job#*:}
  iters=${rest%%:*}
  gpu=${rest#*:}
  # Idempotent: an action already sampled is left alone, so this sweep can be
  # started early for the cheap actions and finished later for the rest.
  if [ -f "$OUT/samples_${action}_${KIND}.npy" ]; then
    echo "$action already sampled"; continue
  fi
  $PY run_tts_gan.py \
    --dataset-dir "$R/replay_dataset_zoh" \
    --output-dir "$OUT" \
    --action "$action" --kind "$KIND" \
    --max-iter "$iters" --samples "$SAMPLES" --sample-batch 500 \
    --gpu "$gpu" > "$OUT/log_${action}_${KIND}.txt" 2>&1 &
done
wait
echo "tts-gan sweep done"
ls -la "$OUT"/samples_*.npy
