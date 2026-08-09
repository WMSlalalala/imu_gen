#!/bin/bash
# Run one built baseline through the detector grid, bundle by bundle.
#
# Usage: grid_against_final.sh <method> [modality ...]
#
# The release finalises each action from a different build, so the grid runs
# once per bundle and each run is restricted to the actions that bundle owns
# (`--action`, which the runner already supports).  Across the four bundles that
# is 18 + 18 + 18 + 36 = exactly the 90 cells the paper reports, with every cell
# computed against the same carrier the release used for that action.
#
# Running all four at once would put four grids on two cards while four
# generative queues are already training there, so they run one after another;
# the runner itself already fills both GPUs (`--device cuda:0 --device cuda:1`)
# and the CPU cells scale across cores.  Each bundle is skipped if it already
# has a completion.json, so re-running this script resumes.
set -u
METHOD=$1
shift
MODALITIES=${*:-"trajectory_xytime imu_only imu_trajectory_xytime"}

C=/mnt/share/mwang49/data7/code/direct100k
C_BASE=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
ROOT=$B/final/$METHOD
PY=/home/mwang49/miniconda3/envs/hml/bin/python
PY_BASE=/home/mwang49/miniconda3/envs/cuhkx/bin/python
MAP=/mnt/share/mwang49/data7/direct100k_final/datasets/ACTION_BUNDLE_MAP.json
# Deep cells are a few GB each and spend their tail on CPU bootstrap, so more
# than one per card keeps the GPU busy.  Four per card alongside the generative
# queues; the classical cells never touch a GPU and scale with cores (64 here).
WORKERS_PER_DEVICE=${WORKERS_PER_DEVICE:-4}
CPU_WORKERS=${CPU_WORKERS:-20}
source "$C_BASE/gpu_slot.sh"
# Which cards this run may touch, resolved once at launch.  After the curfew the
# policy names GPU 0 only and the runner is handed a single --device.
DEVICE_FLAGS=""
for g in $(allowed_gpus); do DEVICE_FLAGS="$DEVICE_FLAGS --device cuda:$g"; done
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$B/PROGRESS.txt"; }

if [ ! -f "$ROOT/bundle_manifest.json" ]; then
  say "grid $METHOD: no bundle_manifest.json -- build it first"
  exit 1
fi

# One instance per (method, modality).  This is not belt-and-braces: the loop
# below does `rm -rf "$out"` before each bundle, so two instances on the same
# arguments delete each other's finished cells mid-run.  That happened -- a
# queue was restarted while its child was still going, and the new queue,
# seeing the job neither done nor claimed, started it a second time.
#
# The lock lives with the job rather than with the queue precisely so it holds
# no matter who launches it: the queue, the supervisor, or a hand-typed command.
LOCK=$ROOT/.lock_${MODALITIES// /_}
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  # Non-zero, not zero.  Exiting 0 here told the queue the job had succeeded,
  # so it recorded it done and moved on -- while the instance that actually
  # holds the lock was still only part-way through.  "Someone else is doing
  # this" is not "this is finished".
  say "grid $METHOD/$MODALITIES already running as pid $(cat "$LOCK"); refusing to start a second"
  exit 3
fi
mkdir -p "$ROOT"
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

# Refuse a modality this method never changed, before spending a single cell on
# it.  A trajectory-only method leaves the inertial channel exactly as the
# release wrote it, so an `imu_only` grid here retrains detectors on the
# release's own data and reports it as the baseline's -- hours of GPU to
# reproduce a number we already have.  pyclick and ghost-cursor each burned a
# joint grid that way.
COVERED=$($PY_BASE "$C_BASE/covered_modalities.py" "$METHOD" 2>/dev/null)
for modality in $MODALITIES; do
  case " $COVERED " in
    *" $modality "*) ;;
    *) say "grid $METHOD: $modality is not a channel this method changed (it covers: ${COVERED:-none}) -- refusing"
       exit 4 ;;
  esac
done

failed=0
for bundle in keystroke scroll swipe tap_and_pinch; do
  dataset=$ROOT/$bundle
  [ -f "$dataset/event_manifest.jsonl" ] || { say "grid $METHOD/$bundle: not built"; continue; }

  # The actions this bundle owns AND this baseline actually swapped.  A method
  # that declines an action left the carrier's own signal in place there, so
  # training a detector on it would measure the release rather than the
  # baseline -- an expensive way to produce a number that must never be
  # reported.  Both facts are read from the release map and the build manifest
  # rather than hardcoded.
  actions=$(/home/mwang49/miniconda3/envs/cuhkx/bin/python -c "
import json
owned = json.load(open('$MAP'))['bundles']['$bundle']
manifest = json.load(open('$ROOT/bundle_manifest.json'))
swapped = manifest['bundles'].get('$bundle', {}).get('swapped', {})
kept = []
for a in owned:
    t = swapped.get(a, {})
    if not t.get('fake', 0) or t.get('imu_swapped', 0) or t.get('trajectory_swapped', 0):
        kept.append(a)
print(' '.join(kept))")
  if [ -z "$actions" ]; then
    say "grid $METHOD/$bundle: every owned action declined by this method, skipping"
    continue
  fi
  action_flags=""
  for a in $actions; do action_flags="$action_flags --action $a"; done

  for modality in $MODALITIES; do
    out=$ROOT/cells_${bundle}_${modality}
    if [ -f "$out/completion.json" ]; then
      say "grid $METHOD/$bundle/$modality already done"
      continue
    fi
    say "grid $METHOD/$bundle/$modality (actions:$actions)"
    rm -rf "$out"
    (cd "$C" && $PY scripts/run_hmog_direct100k_detectors.py \
      --manifest "$dataset/event_manifest.jsonl" --output-dir "$out" \
      --modality "$modality" $action_flags \
      $DEVICE_FLAGS \
      --workers-per-device "$WORKERS_PER_DEVICE" --cpu-workers "$CPU_WORKERS" \
      --epochs 20 --bootstrap-replicates 10000 --seed 42) > "$out.log" 2>&1
    if [ -f "$out/completion.json" ]; then
      say "grid $METHOD/$bundle/$modality OK"
    else
      say "ERROR grid $METHOD/$bundle/$modality -- see $out.log"
      failed=$(( failed + 1 ))
    fi
  done
done
if [ "$failed" -gt 0 ]; then
  # Exit non-zero so the queue does not record this job as done.  A job killed
  # part-way -- by the GPU curfew, an OOM, a stray signal -- used to be marked
  # finished anyway, and its missing bundles were never retried: abl_krefs1 and
  # abl_krefs3 were both recorded complete at 18 and 8 cells of 24.
  say "grid $METHOD: $failed bundle(s) incomplete -- NOT marking the job done"
  exit 1
fi
say "grid $METHOD complete"
