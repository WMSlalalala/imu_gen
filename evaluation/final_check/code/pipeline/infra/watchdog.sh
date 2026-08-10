#!/bin/bash
# Emit one line whenever the programme reaches a milestone or gets into trouble.
#
# Silence has to mean "still working", not "died quietly", so this reports every
# terminal state it can distinguish, not only the good ones:
#
#   DONE     a stage finished
#   GONE     a runner that should be alive is not, and its work is unfinished
#   STALL    both cards idle while work remains -- the signature of a hung job
#   ERROR    a new traceback or CUDA failure appeared in a log
#   OOM      a job died for memory, which also means the reserve may have moved
#
# Every message is emitted once: a fingerprint file remembers what has already
# been reported so a persistent watch does not repeat itself every cycle.
set -u
B=/mnt/share/mwang49/data7/results/direct100k/baselines
SEEN=$B/.watchdog_seen
touch "$SEEN"
IDLE_CYCLES=0

report() {   # report <fingerprint> <message>
  local key=$1; shift
  grep -qxF "$key" "$SEEN" && return 0
  echo "$key" >> "$SEEN"
  echo "[$(date '+%H:%M')] $*"
}

alive() { [ "$(pgrep -cf "$1")" -gt 0 ]; }

while true; do
  # ---- stage milestones, read from the programme's own progress log --------
  if [ -f "$B/PROGRESS.txt" ]; then
    while IFS= read -r line; do
      case "$line" in
        *"cache "*" drawn: "*|*"OK detectors_"*|*"ALL"*"DONE"*|*"complete"*)
          report "progress:$line" "DONE  $line" ;;
        *WARNING*|*FAILED*|*"VERIFY FAILED"*)
          report "progress:$line" "ERROR $line" ;;
      esac
    done < <(tail -60 "$B/PROGRESS.txt")
  fi

  # ---- fresh failures in any log ------------------------------------------
  while IFS= read -r hit; do
    local_file=${hit%%:*}
    report "err:$hit" "ERROR $(basename "$local_file"): ${hit#*:}"
  done < <(grep -rlE "Traceback|CUDA out of memory|RuntimeError|AssertionError" \
             "$B"/*.log "$B"/ablations/*/log_*.txt "$B"/ttsgan/log_*.txt 2>/dev/null \
           | while read -r f; do
               msg=$(grep -hoE "CUDA out of memory|RuntimeError: [^\"]{0,90}|AssertionError[^\"]{0,60}" "$f" | tail -1)
               [ -n "$msg" ] && echo "$f:$msg"
             done)

  # ---- runners that should be alive ---------------------------------------
  if ! alive "run_imu_programm[e].sh"; then
    if [ ! -f "$B/ablations/krefs8/.complete" ]; then
      report "gone:programme:$(date +%H)" "GONE  the IMU programme exited with ablations unfinished"
    else
      report "gone:programme:done" "DONE  the IMU ablation programme finished every cache"
    fi
  fi
  if ! alive "grid_daemo[n].sh" && [ -s "$B/GRID_QUEUE.txt" ] \
     && grep -qvE '^\s*#|^\s*$' "$B/GRID_QUEUE.txt" 2>/dev/null; then
    report "gone:grids:$(date +%H)" "GONE  the grid daemon is not running and the queue is not empty"
  fi

  # ---- a stall: cards idle while work is outstanding ----------------------
  busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
         | awk '{ if ($1 > 10) c++ } END { print c+0 }')
  outstanding=0
  alive "run_imu_programm[e].sh" && outstanding=1
  alive "grid_daemo[n].sh" && outstanding=1
  if [ "$busy" -eq 0 ] && [ "$outstanding" -eq 1 ]; then
    IDLE_CYCLES=$(( IDLE_CYCLES + 1 ))
    if [ "$IDLE_CYCLES" -ge 10 ]; then
      report "stall:$(date +%H%M)" "STALL both cards idle for 10 minutes with work outstanding"
      IDLE_CYCLES=0
    fi
  else
    IDLE_CYCLES=0
  fi

  sleep 60
done
