#!/bin/bash
# Carry each finished ablation cache the rest of the way, without waiting on me.
#
# The sampler writes a cache; the cache has to become a dataset; the dataset has
# to go through the detector grid.  Doing those hand-offs manually leaves the
# cards idle between stages and, worse, leaves a finished cache sitting unused
# overnight.  This watches for caches that are complete and pushes them on.
#
# COMPLETE MEANS EVERY SLOT, NOT `.complete`
# -------------------------------------------
# The upstream sampler writes its `.complete` marker once it has drawn more than
# 30,000 samples, which is not the same as having drawn them all.  A part-filled
# action would mix release IMU with ablation IMU inside one detector's training
# set and produce a number describing neither, so coverage is counted here
# against what the release actually needs: 20,000 fake events per action.
#
# Keystroke is excluded throughout: its fake IMU never passes through the
# diffusion generator (`diffusion_used: false`, generator_source
# `keystroke_imu_pulse.py`), so no ablation of that generator can move it.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
ACTIONS="tap scroll swipe pinch"
NEEDED=20000
say(){ echo "[$(date '+%m-%d %H:%M:%S')] ablpipe: $*" | tee -a "$B/PROGRESS.txt"; }

covered() {   # covered <cache dir> -> 0 if every action has all its slots
  local dir=$1
  for a in $ACTIONS; do
    local n
    n=$(find "$dir" -path "*/$a/*" -name 'sample_*.npz' 2>/dev/null | wc -l)
    [ "$n" -ge "$NEEDED" ] || return 1
  done
  return 0
}

say "started; watching for complete ablation caches"
while true; do
  for dir in "$B"/ablations/*/; do
    [ -d "$dir" ] || continue
    tag=$(basename "$dir")
    # a7_weighted_sum is a training ablation, not a cache; it has no samples.
    [ "$tag" = "a7_weighted_sum" ] && continue
    method="abl_$tag"
    [ -f "$B/final/$method/bundle_manifest.json" ] && continue
    covered "$dir" || continue

    say "$tag is fully covered -> building $method"
    $PY "$C/build_against_final.py" --method "$method" --builder ablation_cache \
      --cache-root "$dir" --kind imu --workers 24 \
      --method-json "{\"arm\":\"$tag\",\"cache\":\"$dir\"}" \
      >> "$B/final_$method.log" 2>&1
    if [ -f "$B/final/$method/bundle_manifest.json" ]; then
      grep -qxF "$method imu_only" "$B/GRID_JOBS.txt" 2>/dev/null \
        || echo "$method imu_only" >> "$B/GRID_JOBS.txt"
      say "$method built and queued for the grid"
    else
      say "ERROR building $method -- see final_$method.log"
    fi
  done
  sleep 300
done
