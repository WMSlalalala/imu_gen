#!/bin/bash
# A7: train the weighted-sum arm for two actions, one per card.
#
# Each is a full retrain at the released run's own epoch count (scroll 160,
# swipe 145), because the ablation is only meaningful against a model given the
# same budget.  The trainer checkpoints and can resume, so re-running this
# script continues rather than restarting.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
source "$C/gpu_slot.sh"
cd "$C" || exit 1
say(){ echo "[$(date '+%m-%d %H:%M:%S')] a7: $*" | tee -a "$B/PROGRESS.txt"; }

run_one() {  # run_one <action> <gpu>
  wait_for_slot "$2" 6000
  say "$1 starting (gpu=$2, weighted sum, project_conflicts=off, cap=off)"
  $PY run_a7_weighted_sum.py --actions "$1" --gpu "$2" >> "$B/ablations/a7_weighted_sum/queue_$1.log" 2>&1
  say "$1 exit $?"
}

run_one scroll "$(pick_gpu 0)" &
run_one swipe  "$(pick_gpu 1)" &
wait
say "queue complete"
