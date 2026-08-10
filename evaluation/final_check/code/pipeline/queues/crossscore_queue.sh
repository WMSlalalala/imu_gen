#!/bin/bash
# Score every built attack with the release's own frozen detectors, one at a time.
#
# This is the second table, and it answers a different question from the main
# one.  The main table trains a fresh detector against each attack -- the
# honest per-attack number, and what the source method's own pipeline does.
# This one applies the 90 detectors the release actually shipped, with their own
# FRR=5% operating points, and retrains nothing.  That is the deployed-defender
# view, and it is the only way to compare attacks on a single fixed boundary.
#
# The two can disagree sharply, and did: an augmentation baseline scored 0.000
# against a detector trained on itself and 0.772 against the release's frozen
# one.  Reporting only the second would make a weak attack look strong, which is
# why the main table stays the per-attack one and this is a separate section.
#
# Pure inference, one job at a time so it never crowds the grids off the cards.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
PY=/home/mwang49/miniconda3/envs/hml/bin/python
source "$C/gpu_slot.sh"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] crossscore: $*" | tee -a "$B/PROGRESS.txt"; }
mkdir -p "$B/crossscore"

while true; do
  did_work=0
  for dir in "$B"/final/*/; do
    [ -d "$dir" ] || continue
    method=$(basename "$dir")
    [ "$method" = "_excluded" ] && continue
    [ -f "$dir/bundle_manifest.json" ] || continue
    # Ablations are variants of the release's own generator, so scoring them
    # against the release's own detectors asks how well a detector recognises a
    # close relative of what it was trained on.  That is a different question
    # from the one this table exists for -- how a deployed detector fares
    # against an attack it has never seen -- and the family-generalisation point
    # is already made by the third-party rows.  Skip them.
    case "$method" in
      abl_*|_*) say "$method is an ablation of the release; cross-scoring it adds nothing"
                continue ;;
    esac
    out=$B/crossscore/${method}.json
    [ -f "$out" ] && continue
    say "$method starting"
    $PY "$C/score_against_fixed_detector.py" --attack-root "$dir" --out "$out" \
      --device "cuda:$(pick_gpu 1)" >> "$B/crossscore/${method}.log" 2>&1
    if [ -f "$out" ]; then say "$method done"; else say "ERROR $method"; fi
    did_work=1
  done
  [ "$did_work" -eq 0 ] && sleep 600
done
