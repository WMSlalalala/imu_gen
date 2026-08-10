#!/bin/bash
# Count grid-queue back-offs in the last N minutes, not in the last N lines.
#
# A line-based check keeps reporting a spin that has already been fixed: the
# tail still holds the pre-fix history, so it re-fires every poll until those
# lines scroll away.  Time is what the question is actually about.
set -u
B=/mnt/share/mwang49/data7/results/direct100k/baselines
MINUTES=${1:-30}
since=$(date -d "$MINUTES minutes ago" '+%m-%d %H:%M')
awk -v s="$since" '{ ts=substr($0,2,11); if (ts >= s) print }' "$B/grid_queue.log" 2>/dev/null \
  | grep -c "backing off\|held by another" || true
