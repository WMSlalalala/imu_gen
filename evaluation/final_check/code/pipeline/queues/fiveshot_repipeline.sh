#!/bin/bash
# Re-draw every cache whose references changed, then hand the result on.
#
# The inertial channel now heads each victim's bank with the five recordings the
# touch channel was frozen against (final_gen/fiveshot_priority).  Only arms that
# consume references are re-drawn: A2 draws none, and keystroke's inertial
# channel comes from the analytic adapter, so neither is re-run for nothing.
#
# Two things this script is careful about, both learned the hard way:
#   - the arm table is read line by line, not word-split.  An earlier version
#     inlined a space-separated action list into a `for entry in $ARMS` loop,
#     which tore every row into fragments and sent output to `/`.
#   - `--dry-run` prints what would happen and touches nothing, because the
#     fleet costs tens of GPU-hours and a parse bug is invisible until it lands.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
export FIVESHOT_PRIORITY=1

SHARDS_PER_GPU=${SHARDS_PER_GPU:-12}     # concurrency per card
GPUS="0 1"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

say(){ echo "[$(date '+%m-%d %H:%M:%S')] fiveshot: $*" | tee -a "$B/PROGRESS.txt"; }

# arm | protocol | k_refs (-1 = protocol default) | actions | out-dir
read -r -d '' ARM_TABLE <<'TABLE' || true
release|fewshot_adv|-1|tap scroll swipe pinch
abl_fewshot_nonadv|fewshot|-1|tap scroll swipe pinch
abl_krefs1|fewshot_adv|1|tap scroll swipe pinch
abl_krefs3|fewshot_adv|3|tap scroll swipe pinch
abl_krefs8|fewshot_adv|8|tap scroll swipe pinch
TABLE

total_shards=$(( SHARDS_PER_GPU * 2 ))
say "started; priority=on; ${total_shards} concurrent shards across GPUs [$GPUS]"

while IFS='|' read -r arm proto k acts; do
  [ -z "${arm:-}" ] && continue
  out="$B/caches/${arm}_v2"
  case "$out" in */caches/_v2) say "ERROR empty arm name; refusing"; exit 1;; esac
  n_act=$(echo "$acts" | wc -w)
  need=$(( n_act * 20000 ))
  have=$(find "$out" -name 'sample_*.npz' 2>/dev/null | wc -l)
  if [ "$have" -ge "$need" ]; then say "$arm already covered ($have/$need)"; continue; fi

  if [ "$DRY" -eq 1 ]; then
    kflag=""; [ "$k" != "-1" ] && kflag="--k-refs $k"
    echo "  ARM=$arm PROTO=$proto K=$k ACTS=[$acts] N_ACT=$n_act NEED=$need"
    echo "      OUT=$out"
    echo "      例: $PY $C/generate_imu_ablation.py --protocol $proto $kflag \\"
    echo "            --actions $(echo $acts | cut -d' ' -f1) --out-dir $out \\"
    echo "            --num-shards $total_shards --shard-index 0 --gpu 0"
    continue
  fi

  mkdir -p "$out"
  kflag=""; [ "$k" != "-1" ] && kflag="--k-refs $k"
  say "$arm: $proto k=$k over [$acts] -> $need samples"
  for a in $acts; do
    shard=0
    while [ "$shard" -lt "$total_shards" ]; do
      gpu=$(( shard % 2 ))
      $PY "$C/generate_imu_ablation.py" --protocol "$proto" $kflag \
        --actions "$a" --out-dir "$out" \
        --num-shards "$total_shards" --shard-index "$shard" --gpu "$gpu" \
        >> "$out/sample_${a}_${shard}.log" 2>&1 &
      shard=$(( shard + 1 ))
      sleep 3
    done
    wait
    got=$(find "$out" -path "*/$a/*" -name 'sample_*.npz' 2>/dev/null | wc -l)
    say "$arm/$a done ($got/20000)"
    [ "$got" -lt 20000 ] && say "WARNING $arm/$a short by $(( 20000 - got ))"
  done
  say "$arm complete ($(find "$out" -name 'sample_*.npz' | wc -l)/$need)"
done <<< "$ARM_TABLE"

# ---- the retrained arms ----------------------------------------------------
# A7-A11 sample from their own training runs, so each needs its run directory and
# the checkpoint family the release pinned for that action (a bare `best_*` glob
# matches two families and once sent A7's scroll to the wrong one).  They cover
# scroll and swipe only, because those are the actions that were retrained.
pick_ckpt() {   # pick_ckpt <action> <run_dir>
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

RUNS=/mnt/share/mwang49/real-human/imu_gen/final/runs
for dir in a7_weighted_sum a8_no_feature a9_no_set a10_no_waveform a11_no_feature_match; do
  out="$B/caches/abl_${dir}_v2"
  need=40000
  have=$(find "$out" -name 'sample_*.npz' 2>/dev/null | wc -l)
  [ "$have" -ge "$need" ] && { say "abl_$dir already covered ($have/$need)"; continue; }
  for a in scroll swipe; do
    run="$RUNS/$a/diffusion/fewshot_adv/${a}_${dir}"
    [ -d "$run" ] || { say "ERROR abl_$dir/$a: no run at $run"; continue; }
    ckpt=$(pick_ckpt "$a" "$run")
    [ -n "$ckpt" ] || { say "ERROR abl_$dir/$a: no checkpoint"; continue; }
    if [ "$DRY" -eq 1 ]; then
      echo "  ARM=abl_$dir ACT=$a RUN=$(basename $run)"
      echo "      CKPT=$(basename $ckpt)"
      echo "      OUT=$out"
      continue
    fi
    mkdir -p "$out"
    say "abl_$dir/$a from $(basename "$ckpt")"
    shard=0
    while [ "$shard" -lt "$total_shards" ]; do
      $PY "$C/generate_imu_ablation.py" --protocol fewshot_adv \
        --run-dir "$run" --checkpoint "$ckpt" \
        --actions "$a" --out-dir "$out" \
        --num-shards "$total_shards" --shard-index "$shard" --gpu $(( shard % 2 )) \
        >> "$out/sample_${a}_${shard}.log" 2>&1 &
      shard=$(( shard + 1 )); sleep 3
    done
    wait
    say "abl_$dir/$a done ($(find "$out" -path "*/$a/*" -name 'sample_*.npz' | wc -l)/20000)"
  done
done

say "all reference-consuming caches redrawn"
