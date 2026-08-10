#!/bin/bash
# Drain a list of "<method> <modality>" grid jobs, one at a time, forever.
#
# Grids turned out to be the critical path: the generative work finishes in
# hours, the grids were a day and a half because they ran strictly one at a
# time.  With the cards at 5-7 GB of 49 there was no reason for that, so
# CONCURRENCY jobs run together.  Append a line to GRID_JOBS.txt and it gets
# picked up; the per-bundle skip inside grid_against_final.sh means a re-queued
# job costs nothing if it already finished.
#
# Concurrency is bounded rather than unbounded because each grid holds
# workers-per-device deep cells on every allowed card, and past a point they
# only time-slice each other.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
JOBS=$B/GRID_JOBS.txt
DONE=$B/GRID_JOBS_DONE.txt
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
say(){ echo "[$(date '+%m-%d %H:%M:%S')] gridqueue: $*" | tee -a "$B/PROGRESS.txt"; }
touch "$JOBS" "$DONE"

CONCURRENCY=${CONCURRENCY:-2}
RUNNING=$B/.grid_running
mkdir -p "$RUNNING"

# Claims left by a previous instance of this queue are stale by definition --
# this queue is starting, so nothing it claimed is still its own.  The driver's
# own PID lock is NOT touched: it checks whether its holder is alive, so a stale
# one costs nothing, while deleting a live one lets two drivers into the same
# output directory and they `rm -rf` each other's finished bundles.  That
# happened, from exactly this cleanup done by hand.
rm -f "$RUNNING"/* 2>/dev/null

run_job() {   # run_job "<method> <modality>"
  local line=$1 method modality claim
  method=$(echo "$line" | awk '{print $1}')
  modality=$(echo "$line" | awk '{print $2}')
  claim=$RUNNING/$(echo "$line" | tr ' /' '__')
  say "starting $method $modality"
  bash "$C/grid_against_final.sh" "$method" "$modality" >> "$B/gridqueue_$method.log" 2>&1
  status=$?
  if [ "$status" -eq 4 ]; then
    # The method does not touch this modality; the job is not work, it is a
    # mistake in the queue.  Record it so it is never picked up again.
    say "$method $modality is not applicable to this method -- dropping it"
    grep -qxF "$line" "$DONE" || echo "$line" >> "$DONE"
    rm -f "$claim"
    return
  fi
  if [ "$status" -eq 3 ]; then
    # Another instance holds the lock.  Sleeping here does nothing: run_job is
    # forked, so the parent loop simply launched the job again thirty seconds
    # later and logged twenty back-offs in forty lines.  The parent now skips
    # locked jobs when it picks, so reaching this branch means the lock was
    # taken between the check and the launch -- rare, and one quiet return is
    # the right response.
    rm -f "$claim"
    return
  fi
  # Ask the artefacts whether it finished; the exit status alone has been wrong
  # three different ways (killed mid-run, refused on a held lock, driver failure
  # swallowed).  `grid_job_done.py` checks that every bundle this method owns
  # has the runner's own completion.json.
  if $PY "$C/grid_job_done.py" "$method" "$modality" >/dev/null 2>&1; then
    grep -qxF "$line" "$DONE" || echo "$line" >> "$DONE"
    say "finished $method $modality"
  else
    say "INCOMPLETE $method $modality (missing: $($PY "$C/grid_job_done.py" "$method" "$modality" 2>/dev/null)) -- will retry"
  fi
  rm -f "$claim"
}

while true; do
  # How many are in flight right now.  A claim file is created before the job
  # forks and removed when it returns, so a crashed job releases its slot the
  # next time the supervisor restarts this queue.
  active=$(find "$RUNNING" -type f 2>/dev/null | wc -l)
  if [ "$active" -ge "$CONCURRENCY" ]; then sleep 60; continue; fi

  # The next job that is neither done nor already claimed.
  line=""
  while read -r candidate; do
    case "$candidate" in ''|\#*) continue ;; esac
    $PY "$C/grid_job_done.py" $candidate >/dev/null 2>&1 && continue
    [ -f "$RUNNING/$(echo "$candidate" | tr ' /' '__')" ] && continue
    # Skip a job whose driver lock is held by a live process, whoever started
    # it.  Without this the parent kept launching instances that died on the
    # lock immediately -- the claim file only tracks jobs this queue started,
    # so an orphan from a previous queue was invisible to it.
    cand_method=$(echo "$candidate" | awk '{print $1}')
    cand_mod=$(echo "$candidate" | awk '{print $2}')
    cand_lock=$B/final/$cand_method/.lock_$cand_mod
    if [ -e "$cand_lock" ] && kill -0 "$(cat "$cand_lock" 2>/dev/null)" 2>/dev/null; then
      continue
    fi
    line=$candidate
    break
  done < "$JOBS"

  if [ -z "$line" ]; then sleep 120; continue; fi

  method=$(echo "$line" | awk '{print $1}')
  if [ ! -f "$B/final/$method/bundle_manifest.json" ]; then
    say "$method not built yet, waiting"
    sleep 120
    continue
  fi
  touch "$RUNNING/$(echo "$line" | tr ' /' '__')"
  run_job "$line" &
  sleep 30
done
