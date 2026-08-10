#!/bin/bash
# Drain a queue of detector grids one at a time, forever, so the cards are never
# idle while the next adapter is being written.  Append a line to GRID_QUEUE.txt
# and it gets picked up; finished lines are moved to GRID_DONE.txt.
set -u
B=/mnt/share/mwang49/data7/results/direct100k/baselines
C=/mnt/share/mwang49/data7/code/direct100k
PY=/home/mwang49/miniconda3/envs/hml/bin/python
Q=$B/GRID_QUEUE.txt
D=$B/GRID_DONE.txt
say(){ echo "[$(date +%H:%M:%S)] $*" >> $B/PROGRESS.txt; }
touch "$D"
while true; do
  line=$(grep -vE '^\s*#|^\s*$' "$Q" 2>/dev/null | head -1)
  if [ -z "$line" ]; then sleep 60; continue; fi
  dataset=$B/$(echo "$line" | cut -d'|' -f1)
  out=$B/$(echo "$line" | cut -d'|' -f2)
  modality=$(echo "$line" | cut -d'|' -f3)
  if [ ! -f "$dataset/event_manifest.jsonl" ]; then
    say "queue: $dataset not built yet, waiting"; sleep 60; continue
  fi
  if [ ! -f "$out/completion.json" ]; then
    say "grid $modality on $(basename $dataset)"
    rm -rf "$out"
    (cd $C && $PY scripts/run_hmog_direct100k_detectors.py \
      --manifest "$dataset/event_manifest.jsonl" --output-dir "$out" \
      --modality "$modality" --device cuda:0 --device cuda:1 \
      --workers-per-device 3 --cpu-workers 24 --epochs 20 \
      --bootstrap-replicates 10000 --seed 42) > "$out.log" 2>&1
    [ -f "$out/completion.json" ] && say "OK $(basename $out)" || say "FAILED $(basename $out)"
  fi
  echo "$line" >> "$D"
  grep -vxF "$line" "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
done
