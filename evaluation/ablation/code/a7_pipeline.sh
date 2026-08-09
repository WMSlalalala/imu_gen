#!/bin/bash
# Carry A7 from a trained generator to a detector table, unattended.
#
# Every other ablation arm starts from a cache the sampler draws off an existing
# checkpoint.  A7 is different: it retrains the generator, so its checkpoint does
# not exist until training ends, and nothing downstream was watching for it.  It
# would have finished training and then sat there.
#
# Three stages, each idempotent:
#   1. draw a cache from the A7 checkpoints (the same sampler, pointed at them)
#   2. build the four release bundles from that cache
#   3. queue the detector grid
#
# A7 trains only scroll and swipe, so its cache covers those two actions and the
# table says so; the other three are not "zero", they were never run.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
RUNS=/mnt/share/mwang49/real-human/imu_gen/final/runs
CACHE=$B/ablations/a7_weighted_sum_cache
ACTIONS="scroll swipe"
NEEDED=20000
say(){ echo "[$(date '+%m-%d %H:%M:%S')] a7pipe: $*" | tee -a "$B/PROGRESS.txt"; }

trained() {   # trained <action> -- has this arm reached its target epoch count?
  $PY - "$1" <<'PYEOF'
import json, re, sys
from pathlib import Path
action = sys.argv[1]
plan = Path("/mnt/share/mwang49/data7/results/direct100k/baselines/ablations/a7_weighted_sum/a7_plan.json")
if not plan.is_file():
    print("no"); raise SystemExit
targets = {r["action"]: int(r["epochs"]) for r in json.loads(plan.read_text())}
total = targets.get(action)
log = Path("/mnt/share/mwang49/real-human/imu_gen/final/runs") / action / \
      "diffusion/fewshot_adv" / f"{action}_a7_weighted_sum/train_log.jsonl"
if total is None or not log.is_file():
    print("no"); raise SystemExit
epochs = re.findall(r'"epoch": (\d+)', log.read_text(errors="ignore"))
print("yes" if epochs and int(epochs[-1]) >= total - 1 else "no")
PYEOF
}

covered_all() {
  for a in $ACTIONS; do
    n=$(find "$CACHE" -path "*/$a/*" -name 'sample_*.npz' 2>/dev/null | wc -l)
    [ "$n" -ge "$NEEDED" ] || return 1
  done
  return 0
}

say "started; waiting for A7 training to finish"
while true; do
  ready=1
  for a in $ACTIONS; do
    [ "$(trained "$a")" = "yes" ] || ready=0
  done
  if [ "$ready" -eq 0 ]; then sleep 600; continue; fi
  say "both A7 arms have trained"
  break
done

# Which checkpoint to draw from.  Selecting the newest inside the A7 run would
# make the arm differ from the release in two ways at once -- the gradient rule
# *and* which checkpoint family was kept -- and only one of those is the ablation.
# So mirror whatever the release pinned for this action: `last.pt` stays
# `last.pt`, and a `best_post_adv_` pin picks the newest of that same family
# (a bare `best_*` glob also matches `best_post_adv_*`, which is how scroll
# first drew from the wrong family).
pick_ckpt() {   # pick_ckpt <action> <a7_run_dir>
  $PY - "$1" "$2" <<'PYEOF2'
import json, sys
from pathlib import Path
action, run = sys.argv[1], Path(sys.argv[2])
pin = json.loads(Path("/mnt/share/mwang49/data7/code/baselines/released_generators.json")
                 .read_text())["runs"][action]["fewshot_adv"]["checkpoint"]
ck = run / "checkpoints"
if pin == "last.pt":
    hit = ck / "last.pt"
    print(hit if hit.is_file() else "", end="")
else:
    family = "best_post_adv_" if pin.startswith("best_post_adv_") else "best_"
    cands = [f for f in ck.glob(family + "*.pt")
             if family == "best_post_adv_" or not f.name.startswith("best_post_adv_")]
    print(max(cands, key=lambda f: f.stat().st_mtime) if cands else "", end="")
PYEOF2
}

# ---- 1. draw the cache from A7's own checkpoints -------------------------
if ! covered_all; then
  say "sampling the A7 cache (scroll, swipe)"
  mkdir -p "$CACHE"
  # Point the sampler at the A7 run instead of the release run.  The registry
  # pins the release, so A7 is passed explicitly rather than by making it the
  # newest -- that is exactly the failure mode the registry exists to prevent.
  for a in $ACTIONS; do
    have=$(find "$CACHE" -path "*/$a/*" -name 'sample_*.npz' 2>/dev/null | wc -l)
    if [ "$have" -ge "$NEEDED" ]; then say "$a already covered ($have); skipping"; continue; fi
    run=$RUNS/$a/diffusion/fewshot_adv/${a}_a7_weighted_sum
    ckpt=$(pick_ckpt "$a" "$run")
    [ -n "$ckpt" ] || { say "ERROR no checkpoint for $a"; continue; }
    say "$a from $(basename "$ckpt")"
    for shard in 0 1 2 3; do
      $PY "$C/generate_imu_ablation.py" --protocol fewshot_adv \
        --run-dir "$run" --checkpoint "$ckpt" \
        --actions "$a" --out-dir "$CACHE" --num-shards 4 --shard-index "$shard" --gpu 0 \
        >> "$CACHE/sample_${a}_$shard.log" 2>&1 &
      sleep 8
    done
  done
  wait
fi

# ---- 2. build, 3. queue the grid ----------------------------------------
if covered_all; then
  if [ ! -f "$B/final/abl_a7_weighted_sum/bundle_manifest.json" ]; then
    say "building abl_a7_weighted_sum"
    $PY "$C/build_against_final.py" --method abl_a7_weighted_sum \
      --builder ablation_cache --cache-root "$CACHE" --kind imu --workers 24 \
      --method-json '{"arm":"A7","description":"gradient merging replaced by a plain weighted sum","actions":"scroll and swipe only"}' \
      >> "$B/final_abl_a7.log" 2>&1
  fi
  grep -qxF "abl_a7_weighted_sum imu_only" "$B/GRID_JOBS.txt" \
    || echo "abl_a7_weighted_sum imu_only" >> "$B/GRID_JOBS.txt"
  say "abl_a7_weighted_sum queued for the grid"
else
  say "ERROR the A7 cache is short; not building a part-filled dataset"
fi
