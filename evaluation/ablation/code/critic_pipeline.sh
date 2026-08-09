#!/bin/bash
# Carry A8-A11 from an untrained arm to a detector table, unattended.
#
# Modelled on a7_pipeline.sh, with its two failures already fixed:
#   - the plan is merged, not rewritten, so training one arm cannot erase
#     another's target and leave this script waiting on an arm that finished;
#   - the checkpoint is the release's own family per action (pick_ckpt), not the
#     newest file, so an arm differs from the release in its own variable only.
#
# Eight runs is more GPU than the machine has spare, so training is serialised
# two at a time and yields to whatever already owns the GPU: the A7 samplers and
# the ImagenTime training finish first, because starting on top of them would
# slow every result rather than add one.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
RUNS=/mnt/share/mwang49/real-human/imu_gen/final/runs
ROOT=$B/ablations
ACTIONS="scroll swipe"
NEEDED=20000
PARALLEL=2

ARMS="A8:a8_no_feature:abl_a8_no_feature A9:a9_no_set:abl_a9_no_set A10:a10_no_waveform:abl_a10_no_waveform A11:a11_no_feature_match:abl_a11_no_feature_match"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] critic: $*" | tee -a "$B/PROGRESS.txt"; }

gpu_busy() {
  local n
  # `pgrep -c` prints 0 *and* exits non-zero when nothing matches, so the usual
  # `|| echo 0` appends a second line and the comparison dies with "integer
  # expression expected" -- which then reads as "not busy" by accident rather
  # than by the count actually being zero.
  n=$(pgrep -fc 'generate_imu_ablation\.py|run_imagentime\.py' 2>/dev/null | head -1)
  n=${n:-0}
  [ "$n" -gt 2 ]
}

trained() {
  $PY - "$1" "$2" <<'PYEOF'
import json, re, sys
from pathlib import Path
arm_dir, action = sys.argv[1], sys.argv[2]
plan = Path("/mnt/share/mwang49/data7/results/direct100k/baselines/ablations") / arm_dir / "plan.json"
if not plan.is_file():
    print("no"); raise SystemExit
targets = {r["action"]: int(r["epochs"]) for r in json.loads(plan.read_text())}
total = targets.get(action)
log = (Path("/mnt/share/mwang49/real-human/imu_gen/final/runs") / action /
       "diffusion/fewshot_adv" / f"{action}_{arm_dir}" / "train_log.jsonl")
if total is None or not log.is_file():
    print("no"); raise SystemExit
epochs = re.findall(r'"epoch": (\d+)', log.read_text(errors="ignore"))
print("yes" if epochs and int(epochs[-1]) >= total - 1 else "no")
PYEOF
}

pick_ckpt() {
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

covered() {
  local a n
  for a in $ACTIONS; do
    n=$(find "$1" -path "*/$a/*" -name 'sample_*.npz' 2>/dev/null | wc -l)
    [ "$n" -ge "$NEEDED" ] || return 1
  done
  return 0
}

say "started; 4 arms x 2 actions"

while true; do
  pending=""
  for entry in $ARMS; do
    arm=${entry%%:*}; rest=${entry#*:}; dir=${rest%%:*}
    for a in $ACTIONS; do
      [ "$(trained "$dir" "$a")" = "yes" ] || pending="$pending $arm/$a"
    done
  done
  [ -z "$pending" ] && { say "all 8 runs trained"; break; }

  if gpu_busy; then
    say "GPU busy (A7 sampling / ImagenTime); waiting -- pending:$pending"
    sleep 900; continue
  fi

  running=0
  launched=0
  for entry in $ARMS; do
    arm=${entry%%:*}; rest=${entry#*:}; dir=${rest%%:*}
    for a in $ACTIONS; do
      [ "$(trained "$dir" "$a")" = "yes" ] && continue
      pgrep -f "run-name ${a}_${dir}\$" >/dev/null 2>&1 && { running=$((running+1)); continue; }
      [ "$running" -ge "$PARALLEL" ] && continue
      mkdir -p "$ROOT/$dir"
      say "training $arm $a"
      $PY "$C/run_critic_ablation.py" --arms "$arm" --actions "$a" --gpu 0 \
        >> "$ROOT/$dir/pipeline_$a.log" 2>&1 &
      running=$((running+1))
      launched=$((launched+1))
      sleep 20
    done
  done
  # `wait` only waits for children *this* shell started.  On a restart the
  # arms already training are other processes: pgrep counts them, nothing new
  # is launched, and wait returns at once -- which spun this loop at 100% CPU
  # until the sleep below was added.
  if [ "$launched" -eq 0 ]; then
    say "nothing new to launch ($running already training); waiting"
    sleep 600
  else
    wait
  fi
done

for entry in $ARMS; do
  arm=${entry%%:*}; rest=${entry#*:}; dir=${rest%%:*}; method=${rest#*:}
  cache=$ROOT/${dir}_cache

  if ! covered "$cache"; then
    say "$arm: sampling"
    mkdir -p "$cache"
    for a in $ACTIONS; do
      have=$(find "$cache" -path "*/$a/*" -name 'sample_*.npz' 2>/dev/null | wc -l)
      [ "$have" -ge "$NEEDED" ] && { say "$arm $a already covered ($have)"; continue; }
      run=$RUNS/$a/diffusion/fewshot_adv/${a}_${dir}
      ckpt=$(pick_ckpt "$a" "$run")
      [ -n "$ckpt" ] || { say "ERROR $arm $a: no checkpoint"; continue; }
      say "$arm $a from $(basename "$ckpt")"
      for shard in 0 1 2 3; do
        $PY "$C/generate_imu_ablation.py" --protocol fewshot_adv \
          --run-dir "$run" --checkpoint "$ckpt" \
          --actions "$a" --out-dir "$cache" --num-shards 4 --shard-index "$shard" --gpu 0 \
          >> "$cache/sample_${a}_$shard.log" 2>&1 &
        sleep 8
      done
      wait
    done
  fi

  if ! covered "$cache"; then
    say "ERROR $arm: cache short; not building a part-filled dataset"
    continue
  fi
  if [ ! -f "$B/final/$method/bundle_manifest.json" ]; then
    say "$arm: building $method"
    $PY "$C/build_against_final.py" --method "$method" \
      --builder ablation_cache --cache-root "$cache" --kind imu --workers 24 \
      --method-json "{\"arm\":\"$arm\",\"actions\":\"scroll and swipe only\"}" \
      >> "$B/final_${method}.log" 2>&1
  fi
  grep -qxF "$method imu_only" "$B/GRID_JOBS.txt" \
    || echo "$method imu_only" >> "$B/GRID_JOBS.txt"
  say "$arm queued for the grid"
done

say "done"
