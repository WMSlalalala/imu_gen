#!/bin/bash
# Drive the whole baseline programme to completion, in dependency order.
#
# Stages are idempotent: each skips if its output already exists, so this can be
# restarted after a failure without redoing finished work.  Progress is appended
# to STATUS so a new session can pick up where this left off.
set -u
R=/mnt/share/mwang49/data7/results/direct100k
B=$R/baselines
C=/mnt/share/mwang49/data7/code/baselines
DC=/mnt/share/mwang49/data7/code/direct100k
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
DPY=/home/mwang49/miniconda3/envs/hml/bin/python
SOURCE=$R/replay_dataset_zoh
BIND=$B/fake_target_binding.pkl
STATUS=$B/PROGRESS.txt
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }
cd "$C" || exit 1

grid() {  # grid <dataset> <out> <modality>
  local dataset=$1 out=$2 modality=$3
  if [ -f "$out/completion.json" ]; then say "grid $out already done"; return 0; fi
  say "grid $modality on $(basename "$dataset")"
  rm -rf "$out"
  (cd "$DC" && $DPY scripts/run_hmog_direct100k_detectors.py \
    --manifest "$dataset/event_manifest.jsonl" --output-dir "$out" \
    --modality "$modality" --device cuda:0 --device cuda:1 \
    --workers-per-device 3 --cpu-workers 24 --epochs 20 \
    --bootstrap-replicates 10000 --seed 42) >> "$out.log" 2>&1
  if [ -f "$out/completion.json" ]; then say "grid OK $out"; else say "GRID FAILED $out"; tail -20 "$out.log"; fi
}

# ---- 1. wait for the Diffusion-TS sweep already in flight -------------------
say "waiting for the Diffusion-TS sweep"
while pgrep -f run_diffusion_ts.py > /dev/null; do sleep 60; done
say "Diffusion-TS sweep finished: $(ls $B/diffts/samples_*.npy 2>/dev/null | wc -l)/10 banks"

# ---- 2. pyclick grid (dataset already rebuilt with the audit fixes) ---------
grid "$B/pyclick_bezier" "$B/detectors_pyclick_traj" trajectory_xytime

# ---- 3. Diffusion-TS trajectory and IMU datasets ---------------------------
for kind in trajectory imu; do
  bank=$B/bank_diffts_$kind.pkl
  dataset=$B/diffts_$kind
  if [ ! -f "$bank" ]; then
    $PY assemble_banks.py --samples-dir "$B/diffts" --out "$bank" --kind $kind \
      >> "$STATUS" 2>&1
  fi
  if [ ! -f "$dataset/release.json" ]; then
    say "building $dataset"
    $PY build_sample_bank_baseline.py --source-dir "$SOURCE" \
      --output-dir "$dataset" --banks "$bank" --binding "$BIND" \
      --method-name "diffusion_ts_$kind" --workers 32 >> "$STATUS" 2>&1
  fi
  say "verifying $dataset"
  $PY verify_harness.py --source-dir "$SOURCE" --built-dir "$dataset" \
    --kind $kind >> "$STATUS" 2>&1 || say "VERIFY FAILED $dataset"
  modality=trajectory_xytime; [ $kind = imu ] && modality=imu_only
  grid "$dataset" "$B/detectors_diffts_$kind" "$modality"
done

# ---- 4. TTS-GAN, IMU -------------------------------------------------------
if [ "$(ls $B/ttsgan/samples_*_imu.npy 2>/dev/null | wc -l)" -lt 5 ]; then
  say "TTS-GAN sweep"
  KIND=imu bash sweep_tts_gan.sh >> "$STATUS" 2>&1
fi
say "TTS-GAN banks: $(ls $B/ttsgan/samples_*_imu.npy 2>/dev/null | wc -l)/5"
if [ ! -f "$B/bank_ttsgan_imu.pkl" ]; then
  $PY assemble_banks.py --samples-dir "$B/ttsgan" --out "$B/bank_ttsgan_imu.pkl" \
    --kind imu >> "$STATUS" 2>&1
fi
if [ ! -f "$B/ttsgan_imu/release.json" ]; then
  say "building $B/ttsgan_imu"
  $PY build_sample_bank_baseline.py --source-dir "$SOURCE" \
    --output-dir "$B/ttsgan_imu" --banks "$B/bank_ttsgan_imu.pkl" \
    --binding "$BIND" --method-name tts_gan_imu --workers 32 >> "$STATUS" 2>&1
fi
$PY verify_harness.py --source-dir "$SOURCE" --built-dir "$B/ttsgan_imu" \
  --kind imu >> "$STATUS" 2>&1 || say "VERIFY FAILED ttsgan_imu"
grid "$B/ttsgan_imu" "$B/detectors_ttsgan_imu" imu_only

# ---- 5. generator quality, on CPU, against a real-versus-real control ------
say "quality sweep: Diffusion-TS"
bash run_quality_sweep.sh "$B/diffts" "$B/quality_diffts" "trajectory imu" >> "$STATUS" 2>&1
say "quality sweep: TTS-GAN"
bash run_quality_sweep.sh "$B/ttsgan" "$B/quality_ttsgan" "imu" >> "$STATUS" 2>&1

say "ALL STAGES DONE"
