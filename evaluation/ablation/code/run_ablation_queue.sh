#!/bin/bash
# Draw the remaining ablation caches, one after another, both cards filled.
#
# This runs beside whatever else is in flight.  Every launch waits for room, so
# the user's working reserve on both cards survives no matter how many stages
# start at once, and an action whose protocol was never trained upstream is
# dropped by the sampler itself and recorded in that cache's ablation.json.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
# A sampling shard holds about 860 MiB, so memory is never the binding
# constraint here -- contention is.  24 shards was sized for two cards.  On one card, 22 samplers plus the grids,
# A7 and CSDI put 54 compute processes on a single GPU: utilisation reads 100%
# but each process gets about 2% of it, and a deep detector cell that takes five
# minutes alone stopped finishing for 45 minutes.  Total GPU work is unchanged by
# oversubscription -- only the latency of everything gets worse -- so the sampler
# takes a smaller share and leaves room for the grids, which are the critical
# path.
SHARDS=${SHARDS:-8}
source "$C/gpu_slot.sh"
cd "$C" || exit 1
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$B/PROGRESS.txt"; }

# Keystroke is excluded from every ablation arm, and not because it is awkward:
# its fake IMU never passes through the diffusion generator at all.  The released
# keystroke events carry `diffusion_used: false`, `model_used: false` and
# `generator_source: security_exp/keystroke_imu_pulse.py` -- an analytic adapter
# built from the victim's own genuine keystroke events.  Ablating the generator
# therefore cannot move a keystroke IMU cell by construction.
#
# Reporting it would be worse than omitting it: a zero delta reads as "the
# five-shot conditioning does not help keystroke", when the truth is "this
# ablation does not reach keystroke".  It belongs in the table as not-applicable.
ABLATION_ACTIONS="tap scroll swipe pinch"

run_cache() {   # run_cache <tag> <protocol> [k_refs]
  local tag=$1 proto=$2 k=${3:-}
  local out=$B/ablations/$tag
  [ -f "$out/.complete" ] && { say "cache $tag already drawn"; return 0; }
  mkdir -p "$out"
  local extra=""
  [ -n "$k" ] && extra="--k-refs $k"
  say "sampling cache $tag (protocol=$proto k_refs=${k:-protocol default})"
  # Yield to the detector grids.  Sampling and gridding are both GPU-bound and
  # the card is one; running them together does not increase throughput, it just
  # delays both.  The grids finish the ablation tables, and a table nobody can
  # read yet is worth more than the next cache, so sampling waits while grid
  # cells are in flight.  Measured: with 36 processes on one card a deep cell
  # ran 99% CPU for 87 minutes without finishing.
  # See status_daemon.sh: pgrep -c already prints 0 on no match, so `|| echo 0`
  # made this test error out and fall through instead of throttling.
  local cells
  cells=$(pgrep -fc 'run_hmog_direct100k_cel[l].py' 2>/dev/null | head -1); cells=${cells:-0}
  while [ "$cells" -gt 4 ]; do
    say "cache $tag: waiting for the detector grids to drain ($cells running)"
    sleep 300
    cells=$(pgrep -fc 'run_hmog_direct100k_cel[l].py' 2>/dev/null | head -1); cells=${cells:-0}
  done

  local shard=0
  while [ "$shard" -lt "$SHARDS" ]; do
    local gpu; gpu=$(pick_gpu $(( shard % 2 )))
    wait_for_slot "$gpu" 1200
    $PY generate_imu_ablation.py --protocol "$proto" $extra \
      --actions $ABLATION_ACTIONS \
      --out-dir "$out" --num-shards "$SHARDS" --shard-index "$shard" --gpu "$gpu" \
      > "$out/log_shard$shard.txt" 2>&1 &
    sleep 6
    shard=$(( shard + 1 ))
  done
  wait
  local drawn
  drawn=$(find "$out" -name "sample_*.npz" | wc -l)
  say "cache $tag drawn: $drawn samples"
  if [ "$drawn" -gt 30000 ]; then
    touch "$out/.complete"
  else
    say "WARNING cache $tag drew only $drawn samples"
  fi
}

# noshot_adv comes first: it is the k_refs=0 arm of the adversarial-on row and
# the one the paper's "does the victim's five shots matter" claim rests on.
# It was already 10.5k samples in when the run was stopped; the sampler skips
# what it has drawn, so this resumes rather than restarts.
run_cache noshot_adv noshot_adv
run_cache fewshot_nonadv fewshot
run_cache krefs1 fewshot_adv 1
run_cache krefs3 fewshot_adv 3
run_cache krefs8 fewshot_adv 8
say "ablation queue complete"
