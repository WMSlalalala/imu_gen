#!/bin/bash
# Fill both cards until the two-card window closes, then keep going on one.
#
# Ordering rule: SAMPLING AND TRAINING FIRST.  A detector grid spends 20 of its
# 30 cells on CPU-only classical detectors, so losing a card slows it by far
# less than it slows a diffusion sampler, which is pure GPU.  Everything that
# only a GPU can do is therefore packed into the two-card window and the grids
# are left to drain afterwards.
#
# Each stage is idempotent: an output that already exists is skipped, so this
# can be restarted at any point.
set -u
R=/mnt/share/mwang49/data7/results/direct100k
B=$R/baselines
C=/mnt/share/mwang49/data7/code/baselines
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
ABL=$B/ablations
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $B/PROGRESS.txt; }
mkdir -p "$ABL"
cd "$C" || exit 1
# The user keeps a working reserve on both cards; every launch waits for room.
source "$C/gpu_slot.sh"
SAMPLER_MIB=${SAMPLER_MIB:-2600}

# ---- one ablation cache: eight shards across both cards ---------------------
sample_cache() {   # sample_cache <tag> <protocol> [k_refs]
  local tag=$1 protocol=$2 krefs=${3:-}
  local out=$ABL/$tag
  if [ -f "$out/.complete" ]; then say "cache $tag already drawn"; return 0; fi
  say "sampling cache $tag (protocol=$protocol k_refs=${krefs:-default})"
  mkdir -p "$out"
  local extra=""
  [ -n "$krefs" ] && extra="--k-refs $krefs"
  for shard in 0 1 2 3 4 5 6 7; do
    local gpu=$(( shard % 2 ))
    wait_for_slot "$gpu" "$SAMPLER_MIB"
    $PY generate_imu_ablation.py --protocol "$protocol" $extra \
      --out-dir "$out" --num-shards 8 --shard-index "$shard" --gpu "$gpu" \
      > "$out/log_shard$shard.txt" 2>&1 &
    sleep 25   # let the process claim its memory before the next slot check
  done
  wait
  local drawn
  drawn=$(find "$out" -name "sample_*.npz" | wc -l)
  say "cache $tag drawn: $drawn samples"
  [ "$drawn" -gt 1000 ] && touch "$out/.complete" || say "WARNING $tag looks short"
}

# ---- 1. the two protocol ablations the factorial needs ----------------------
sample_cache noshot_adv noshot_adv
sample_cache fewshot_nonadv fewshot

# ---- 2. the reference-count sweep ------------------------------------------
for k in 1 3 8; do
  sample_cache "krefs$k" fewshot_adv "$k"
done

say "IMU ablation sampling complete"
