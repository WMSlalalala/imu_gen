#!/bin/bash
# Retrain the three TTS-GAN actions that were lost when the run was stopped.
#
# pinch (80k iterations) and scroll (60k) survived and are not touched.  The
# three here were killed mid-training by a loop that kept nothing between
# epochs; run_tts_gan.py now checkpoints every --checkpoint-every epochs and
# resumes from the last one, so re-running this script after any interruption
# continues rather than starting over.
#
# Budgets are matched to the two that finished, scaled by how far each action's
# convergence gap still was from the genuine data at the low-budget probe
# (ttsgan_budget_evidence.json).  They are deliberately generous: the point of
# this baseline is to report what TTS-GAN does when it is given enough, so that
# a poor number is a statement about the method and not about my patience.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
S=/mnt/share/mwang49/data7/results/direct100k/replay_dataset_v12
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
OUT=$B/ttsgan
source "$C/gpu_slot.sh"
cd "$C" || exit 1
mkdir -p "$OUT"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$B/PROGRESS.txt"; }

# The training set is the genuine windows of the train-split users, which do not
# depend on which replay_dataset version supplies the fake carriers -- so this
# work is valid whichever carrier version the final tables are built on.
run_one() {   # run_one <action> <iterations> <gpu>
  local action=$1 iters=$2 gpu=$3
  if [ -f "$OUT/samples_${action}_imu.npy" ]; then
    say "ttsgan $action already sampled"
    return 0
  fi
  wait_for_slot "$gpu" 4000
  say "ttsgan $action starting (max_iter=$iters gpu=$gpu)"
  $PY run_tts_gan.py --dataset-dir "$S" --output-dir "$OUT" \
    --action "$action" --kind imu --max-iter "$iters" \
    --samples 4000 --sample-batch 500 --gpu "$gpu" --checkpoint-every 10 \
    > "$OUT/log_${action}_imu.txt" 2>&1
  local status=$?
  if [ $status -eq 0 ]; then
    say "ttsgan $action done"
  else
    say "ERROR ttsgan $action exited $status -- rerun this script to resume"
  fi
  return $status
}

# One action per card, the third behind whichever frees first.  Each waits for
# room, so the ablation queue sharing these cards is never starved and the
# user's reserve holds.
run_one swipe 60000 0 &
run_one keystroke 48000 1 &
wait
run_one tap 45000 0
say "ttsgan retrain queue complete"
