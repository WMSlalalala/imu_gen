#!/bin/bash
# Finish every remaining comparison so each experiment the paper runs has a
# baseline counterpart.  Stages are idempotent and ordered by dependency; the
# grids never overlap, because two of them on the same two cards halves both.
#
# What is still missing when this starts:
#   * the joint modality (imu_trajectory_xytime) for every baseline -- the
#     paper's own grid has 90 cells and the baselines so far have 30 each
#   * TTS-GAN as a full baseline (its trajectory models are training when this
#     script begins)
#   * the retrained TTS-GAN IMU dataset and grid
set -u
B=/mnt/share/mwang49/data7/results/direct100k/baselines
C=/mnt/share/mwang49/data7/code/baselines
DC=/mnt/share/mwang49/data7/code/direct100k
S=/mnt/share/mwang49/data7/results/direct100k/replay_dataset_zoh
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
DPY=/home/mwang49/miniconda3/envs/hml/bin/python
BIND=$B/fake_target_binding.pkl
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $B/PROGRESS.txt; }
cd "$C" || exit 1

grid() {  # grid <dataset> <out> <modality>
  local dataset=$1 out=$2 modality=$3
  [ -f "$out/completion.json" ] && { say "grid $(basename $out) already done"; return 0; }
  say "grid $modality on $(basename $dataset)"
  rm -rf "$out"
  (cd "$DC" && $DPY scripts/run_hmog_direct100k_detectors.py \
    --manifest "$dataset/event_manifest.jsonl" --output-dir "$out" \
    --modality "$modality" --device cuda:0 --device cuda:1 \
    --workers-per-device 3 --cpu-workers 24 --epochs 20 \
    --bootstrap-replicates 10000 --seed 42) > "$out.log" 2>&1
  [ -f "$out/completion.json" ] && say "OK $(basename $out)" || say "FAILED $(basename $out)"
}

wait_for_ttsgan() {  # wait_for_ttsgan <kind>
  local kind=$1
  while [ "$(ls $B/ttsgan/samples_*_${kind}.npy 2>/dev/null | wc -l)" -lt 5 ]; do sleep 60; done
  say "ttsgan $kind: 5/5 banks"
}

# ---- 1. the joint modality for the two Diffusion-TS datasets ---------------
# A joint cell needs both channels generated, so it gets its own dataset with
# the trajectory and the IMU bank applied together.  Keystroke is declined here
# for the same reason the trajectory-only dataset declines it: its generated
# trajectory measures the placement transform rather than the generator, and a
# joint cell built on it would understate the baseline.  The keystroke IMU
# result stays where it is valid, in the imu_only table.
if [ ! -f "$B/diffts_both/release.json" ]; then
  $PY - <<'PYTHON'
import pickle
B="/mnt/share/mwang49/data7/results/direct100k/baselines"
banks={}
for kind in ("trajectory","imu"):
    with open(f"{B}/bank_diffts_{kind}.pkl","rb") as h:
        banks.update(pickle.load(h))
with open(f"{B}/bank_diffts_both.pkl","wb") as h:
    pickle.dump(banks,h)
print("combined bank:", {k: sorted(v) for k,v in banks.items()})
PYTHON
  say "building diffts_both"
  $PY build_sample_bank_baseline.py --source-dir "$S" --output-dir "$B/diffts_both" \
    --banks "$B/bank_diffts_both.pkl" --binding "$BIND" \
    --method-name diffusion_ts_trajectory_and_imu --decline keystroke --workers 32 >> $B/PROGRESS.txt 2>&1
fi
$PY verify_harness.py --source-dir "$S" --built-dir "$B/diffts_both" --kind both \
  > $B/verify_diffts_both.txt 2>&1 || say "VERIFY FAILED diffts_both"
grid "$B/diffts_both" "$B/detectors_diffts_joint" imu_trajectory_xytime

# ---- 2. pyclick's joint cells ---------------------------------------------
# Only its trajectory is its own; the IMU stays the paper's.  Labelled that way
# in the report, because it is a mixed condition, not a pyclick result.
grid "$B/pyclick_bezier" "$B/detectors_pyclick_joint" imu_trajectory_xytime

# ---- 3. TTS-GAN IMU, retrained --------------------------------------------
wait_for_ttsgan imu
if [ ! -f "$B/ttsgan_imu/release.json" ] || [ "$B/ttsgan/samples_tap_imu.npy" -nt "$B/ttsgan_imu/release.json" ]; then
  $PY assemble_banks.py --samples-dir "$B/ttsgan" --out "$B/bank_ttsgan_imu.pkl" --kind imu >> $B/PROGRESS.txt 2>&1
  say "building ttsgan_imu"
  rm -rf "$B/ttsgan_imu"
  $PY build_sample_bank_baseline.py --source-dir "$S" --output-dir "$B/ttsgan_imu" \
    --banks "$B/bank_ttsgan_imu.pkl" --binding "$BIND" --method-name tts_gan_imu \
    --workers 32 >> $B/PROGRESS.txt 2>&1
fi
$PY verify_harness.py --source-dir "$S" --built-dir "$B/ttsgan_imu" --kind imu \
  > $B/verify_ttsgan_imu.txt 2>&1 || say "VERIFY FAILED ttsgan_imu"
grid "$B/ttsgan_imu" "$B/detectors_ttsgan_imu" imu_only

# ---- 4. TTS-GAN trajectory and joint --------------------------------------
wait_for_ttsgan trajectory
for kind in trajectory both; do
  if [ "$kind" = trajectory ]; then
    $PY assemble_banks.py --samples-dir "$B/ttsgan" --out "$B/bank_ttsgan_trajectory.pkl" \
      --kind trajectory >> $B/PROGRESS.txt 2>&1
    src=$B/bank_ttsgan_trajectory.pkl; out=$B/ttsgan_trajectory; mod=trajectory_xytime
  else
    $PY - <<'PYTHON'
import pickle
B="/mnt/share/mwang49/data7/results/direct100k/baselines"
banks={}
for kind in ("trajectory","imu"):
    with open(f"{B}/bank_ttsgan_{kind}.pkl","rb") as h:
        banks.update(pickle.load(h))
with open(f"{B}/bank_ttsgan_both.pkl","wb") as h:
    pickle.dump(banks,h)
PYTHON
    src=$B/bank_ttsgan_both.pkl; out=$B/ttsgan_both; mod=imu_trajectory_xytime
  fi
  if [ ! -f "$out/release.json" ]; then
    say "building $(basename $out)"
    $PY build_sample_bank_baseline.py --source-dir "$S" --output-dir "$out" \
      --banks "$src" --binding "$BIND" --method-name "tts_gan_$kind" \
      --decline keystroke --workers 32 >> $B/PROGRESS.txt 2>&1
  fi
  check=$kind; [ "$kind" = both ] && check=both
  $PY verify_harness.py --source-dir "$S" --built-dir "$out" --kind "$check" \
    > $B/verify_ttsgan_$kind.txt 2>&1 || say "VERIFY FAILED $(basename $out)"
  grid "$out" "$B/detectors_ttsgan_$kind" "$mod"
done

# ---- 5. quality scores for the retrained TTS-GAN --------------------------
say "ttsgan quality sweep"
bash run_quality_sweep.sh "$B/ttsgan" "$B/quality_ttsgan" "imu trajectory" >> $B/PROGRESS.txt 2>&1
say "ALL REMAINING STAGES DONE"
