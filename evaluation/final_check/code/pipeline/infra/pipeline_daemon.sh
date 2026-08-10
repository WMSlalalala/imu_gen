#!/bin/bash
# Watch for generators that have finished sampling and carry them the rest of
# the way: assemble a sample bank, build the four release bundles, verify, grid.
#
# The four generative queues finish at very different times -- TTS-GAN in hours,
# CSDI and ImagenTime later, action by action -- and each one's samples are
# useless until they have been through the bundle build and the detector grid.
# Doing that by hand would leave the cards idle between stages, so this drains
# the work as it appears and keeps going.
#
# Every stage is idempotent and marked on disk, so this can be killed and
# restarted at any point without redoing finished work.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
BIND=$B/fake_target_binding_v12.pkl
say(){ echo "[$(date '+%m-%d %H:%M:%S')] pipeline: $*" | tee -a "$B/PROGRESS.txt"; }

# ---- which sample files each method produces, and where -------------------
# method | directory glob for per-action samples | bank name
methods() {
  cat <<'SPEC'
ttsgan|SAMPLES:ttsgan/samples_%s_imu.npy|bank_final_ttsgan_imu.pkl
csdi_unconditional|SAMPLES:csdi/%s/samples_%s_unconditional.npy|bank_final_csdi_uncond.pkl
csdi_fiveshot|SAMPLES:csdi/%s/samples_%s_conditional.npy|bank_final_csdi_fiveshot.pkl
imagentime|SAMPLES:imagentime/%s/samples_%s_imu.npy|bank_final_imagentime.pkl
SPEC
}

sample_path() {   # sample_path <method> <action>
  case $1 in
    ttsgan)             echo "$B/ttsgan/samples_$2_imu.npy" ;;
    csdi_unconditional) echo "$B/csdi/$2/samples_$2_unconditional.npy" ;;
    csdi_fiveshot)      echo "$B/csdi/$2/samples_$2_conditional.npy" ;;
    imagentime)         echo "$B/imagentime/$2/samples_$2_imu.npy" ;;
  esac
}

ready_actions() {  # ready_actions <method> -> the actions that have samples
  local method=$1 out=""
  for action in tap scroll swipe pinch keystroke; do
    [ -f "$(sample_path "$method" "$action")" ] && out="$out $action"
  done
  echo "$out"
}

while true; do
  # csdi_fiveshot dropped; see run_csdi_all.sh
  for method in ttsgan csdi_unconditional imagentime; do
    marker=$B/final/$method/.gridded
    [ -f "$marker" ] && continue

    actions=$(ready_actions "$method")
    count=$(echo "$actions" | wc -w)
    [ "$count" -eq 0 ] && continue

    # Wait for the full set before building: a bank missing an action would
    # produce a dataset that declines it, and a later rebuild would silently
    # change what the published table means.
    if [ "$count" -lt 5 ]; then
      say "$method has $count/5 actions ($actions) -- waiting for the rest"
      continue
    fi

    bank=$B/bank_final_${method}.pkl
    if [ ! -f "$bank" ]; then
      say "$method: assembling sample bank"
      $PY "$C/assemble_banks.py" --out "$bank" --kind imu \
        $(for a in $actions; do echo "--sample $a=$(sample_path "$method" "$a")"; done) \
        >> "$B/pipeline_$method.log" 2>&1 || { say "ERROR $method bank"; continue; }
    fi

    if [ ! -f "$B/final/$method/bundle_manifest.json" ]; then
      say "$method: building the four release bundles"
      $PY "$C/build_against_final.py" --method "$method" --builder sample_bank \
        --banks "$bank" --binding "$BIND" --kind imu --workers 24 \
        >> "$B/pipeline_$method.log" 2>&1 || { say "ERROR $method build"; continue; }
    fi

    say "$method: detector grid (imu_only)"
    bash "$C/grid_against_final.sh" "$method" imu_only >> "$B/pipeline_$method.log" 2>&1
    touch "$marker"
    say "$method: complete"
  done
  sleep 300
done
