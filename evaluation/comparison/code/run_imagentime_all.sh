#!/bin/bash
# ImagenTime over all five actions.
#
# The authors' full budget: 1000 epochs, no shortfall to disclose.
#
# 1000 epochs was originally ruled out because it measured 79 GPU-hours for the
# five actions at the wide UNet.  Every action now uses the authors' narrower
# UNet from mujoco.yaml (14.4M parameters instead of 151.7M, measured 4.5x
# cheaper per step), which brings the full budget back within reach -- so the
# baseline gets what its authors specified rather than a fraction of it.
#
# Each action checkpoints every 10 epochs and can be sampled from whatever epoch
# it has reached, so this is stoppable at any point without losing the run: if
# the budget turns out not to fit, the answer is a shorter run that is disclosed,
# not a wasted one.
#
# Every action checkpoints every 10 epochs and resumes from the last one, so
# re-running this script continues rather than restarting, and a run stopped
# early can still be sampled from whatever epoch it reached.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
OUT=$B/imagentime
EPOCHS=${EPOCHS:-1000}
SAMPLES=${SAMPLES:-4000}
source "$C/gpu_slot.sh"
cd "$C" || exit 1
mkdir -p "$OUT"
# One instance at a time.  Two of these ran together once -- the scheduler was
# restarted and launched a second -- and because each action resumes from its
# checkpoint, the newcomer picked up the weights but rebuilt the convergence
# monitor empty, so the patience counter reset and early stopping stopped
# working.  A stale lock (holder gone) is taken over rather than obeyed.
LOCK=$B/.imagentime_all.lock
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date '+%m-%d %H:%M:%S')] imagentime_all: pid $(cat "$LOCK") holds the lock; refusing" \
    | tee -a "$B/PROGRESS.txt"
  exit 3
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$B/PROGRESS.txt"; }

run_action() {   # run_action <action> <gpu>
  local action=$1 gpu=$2
  local folder=$OUT/$action
  mkdir -p "$folder"
  if [ -f "$folder/samples_${action}_imu.npy" ]; then
    say "imagentime $action already drawn"
    return 0
  fi
  wait_for_slot "$gpu" 5000
  say "imagentime $action starting (gpu=$gpu epochs=$EPOCHS)"
  $PY run_imagentime.py --real-dir "$B/real_windows" --output-dir "$folder" \
    --action "$action" --epochs "$EPOCHS" --samples "$SAMPLES" --gpu "$gpu" \
    >> "$folder/log_${action}.txt" 2>&1
  if [ $? -eq 0 ]; then
    say "imagentime $action done"
  else
    say "ERROR imagentime $action -- rerun this script to resume"
  fi
}

# All five at once, spread over both cards.  Measured about 4 GB per action, so
# five of them fit alongside the other queues and the sequential arrangement was
# leaving most of both cards unused.  Keystroke is the long pole even at the
# narrow UNet and starts first; the slot gate makes each wait for room, so this
# degrades to sequential rather than failing if the estimate is wrong.
run_action keystroke "$(pick_gpu 1)" &
run_action scroll    "$(pick_gpu 0)" &
run_action swipe     "$(pick_gpu 1)" &
run_action pinch     "$(pick_gpu 0)" &
run_action tap       "$(pick_gpu 0)" &
wait
say "imagentime queue complete"
