#!/bin/bash
# CSDI over all five actions, both sampling arms.
#
# One model per action, because the sequence length is baked into the attention
# grid and the positional encoding.  Both arms share that one model: CSDI's
# conditioning is a sampling-time mask, not a training-time setting, so the
# unconditional draw and the five-shot draw come from identical weights.  That
# is worth stating in the paper -- the two arms differ only in what the sampler
# is told, which makes the comparison between them clean.
#
# Budget is the authors' own 200 epochs from config/base.yaml.  Each action
# checkpoints every epoch and resumes from the last one, so re-running this
# script after any interruption continues where it stopped.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
REAL=$B/real_windows
SHOTS=$B/fiveshot_imu_windows.pkl
OUT=$B/csdi
EPOCHS=${EPOCHS:-200}
SAMPLES=${SAMPLES:-4000}
source "$C/gpu_slot.sh"
cd "$C" || exit 1
mkdir -p "$OUT"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$B/PROGRESS.txt"; }

run_action() {   # run_action <action> <gpu>
  local action=$1 gpu=$2
  local folder=$OUT/$action
  mkdir -p "$folder"
  # The five-shot arm is dropped: it shares the trained model with the
  # unconditional one and differs only in what the sampler is told, so it costs
  # ten more deep detector cells (about three hours on one card) to add a second
  # CSDI row that the paper does not need.  Set CSDI_MODES to bring it back.
  for mode in ${CSDI_MODES:-unconditional}; do
    if [ -f "$folder/samples_${action}_${mode}.npy" ]; then
      say "csdi $action/$mode already drawn"
      continue
    fi
    # Peak was measured at 2.4 GiB; ask for 4 GiB so the wait also covers the
    # conditional arm, whose grid is six times longer.
    wait_for_slot "$gpu" 4000
    say "csdi $action/$mode starting (gpu=$gpu epochs=$EPOCHS)"
    local extra=""
    [ "$mode" = "conditional" ] && extra="--shots-cache $SHOTS"
    $PY run_csdi.py --real-dir "$REAL" --output-dir "$folder" \
      --action "$action" --mode "$mode" $extra \
      --epochs "$EPOCHS" --samples "$SAMPLES" --gpu "$gpu" \
      >> "$folder/log_${action}_${mode}.txt" 2>&1
    if [ $? -eq 0 ]; then
      say "csdi $action/$mode done"
    else
      say "ERROR csdi $action/$mode -- rerun this script to resume"
    fi
  done
}

# All five at once, spread over both cards.  CSDI is small -- measured about
# 2-3 GB per action -- so running them in sequence left the cards mostly idle.
# Keystroke (T=512, quadratic attention over both axes) is the long pole and
# starts first on its own card; the slot gate makes each wait for room, so this
# degrades to sequential rather than failing if the estimate is wrong.
run_action keystroke "$(pick_gpu 1)" &
run_action scroll    "$(pick_gpu 0)" &
run_action swipe     "$(pick_gpu 0)" &
run_action pinch     "$(pick_gpu 1)" &
run_action tap       "$(pick_gpu 0)" &
wait
say "csdi queue complete"
