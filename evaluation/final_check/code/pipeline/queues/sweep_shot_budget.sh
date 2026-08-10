#!/bin/bash
# Fit Diffusion-TS under an attacker's actual shot budget, not a corpus.
#
# The main baselines are given every genuine event of all seventy training
# users -- roughly a thousand times what the paper's own attack is given.  That
# is a deliberately generous setting, and a baseline that still loses there has
# lost on generation quality.  But it cannot answer the question the paper's
# design actually rests on: what does a learned generator do when it only has
# the five events an attacker can realistically capture from one victim?
#
# Two matched budgets are fitted, both with the same architecture, the same
# 12,000 steps and the same sampling as the unrestricted run, so the only thing
# that changes is how much real data the model saw:
#
#   shot5    5 windows total -- the literal five-shot budget of a single victim
#   shot5pu  5 windows per training user (350 total) -- five shots of everyone
#            the attacker can reach, a strictly easier setting than the paper's
#
# Nothing here is expected to work well.  That is the point: the number has to
# exist so the paper can say how badly it fails rather than assert it.
set -u
R=/mnt/share/mwang49/data7/results/direct100k
C=/mnt/share/mwang49/data7/code/baselines
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
SAMPLES=${SAMPLES:-4000}
STEPS=${STEPS:-12000}
KINDS=${KINDS:-"trajectory imu"}
cd "$C" || exit 1

run_budget() {  # run_budget <tag> <flag> <value>
  local tag=$1 flag=$2 value=$3
  local out=$R/baselines/diffts_$tag
  mkdir -p "$out"
  local gpu=0
  for kind in $KINDS; do
    for action in tap pinch swipe scroll keystroke; do
      [ -f "$out/samples_${action}_${kind}.npy" ] && continue
      $PY run_diffusion_ts.py --dataset-dir "$R/replay_dataset_zoh" \
        --output-dir "$out" --action "$action" --kind "$kind" \
        --steps "$STEPS" --samples "$SAMPLES" --sample-batch 256 \
        "$flag" "$value" --gpu "$gpu" \
        > "$out/log_${action}_${kind}.txt" 2>&1 &
      gpu=$((1 - gpu))
    done
    wait
  done
}

run_budget shot5 --train-events 5
echo "shot5 done"
run_budget shot5pu --events-per-user 5
echo "shot5pu done"
