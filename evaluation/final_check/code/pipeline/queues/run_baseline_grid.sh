#!/bin/bash
# Run one baseline dataset through the same detector grid the method uses.
# Usage: run_baseline_grid.sh <dataset-dir> <output-dir> <modality>
set -u
DATA=$1
OUT=$2
MOD=$3
C=/mnt/share/mwang49/data7/code/direct100k
LOG=$OUT.log
mkdir -p "$(dirname "$OUT")"
echo "[$(date +%H:%M:%S)] $MOD grid on $DATA" | tee -a "$LOG"
cd "$C" || exit 1
rm -rf "$OUT"
/home/mwang49/miniconda3/envs/hml/bin/python scripts/run_hmog_direct100k_detectors.py \
  --manifest "$DATA/event_manifest.jsonl" \
  --output-dir "$OUT" \
  --modality "$MOD" \
  --device cuda:0 --device cuda:1 --workers-per-device 3 --cpu-workers 24 \
  --epochs 20 --bootstrap-replicates 10000 --seed 42 >> "$LOG" 2>&1
if [ -f "$OUT/completion.json" ]; then
  echo "[$(date +%H:%M:%S)] OK $OUT" | tee -a "$LOG"
else
  echo "[$(date +%H:%M:%S)] FAILED $OUT" | tee -a "$LOG"
  tail -30 "$LOG"
  exit 1
fi
