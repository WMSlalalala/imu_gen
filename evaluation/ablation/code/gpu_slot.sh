#!/bin/bash
# Wait until a card has room, so a long programme never crowds the user out.
#
# The user keeps a working reserve on both cards.  Every launcher here calls
# `wait_for_slot <gpu> <needed_mib>` before starting a job and only proceeds
# when the card has the job's footprint plus the reserve still free, so the
# reserve survives even when several stages start at once.
# 6 GB per card.  The user asked to be left 5-10 GB with everything else filled;
# this sits at the bottom of that band deliberately, because the gate is checked
# per launch -- a job that starts just under the line can leave actual free
# memory a little below the nominal reserve, so the nominal number wants headroom
# under the real floor rather than at it.
RESERVE_MIB=${RESERVE_MIB:-6000}
POLL_SECONDS=${POLL_SECONDS:-20}

free_mib() {   # free_mib <gpu>
  nvidia-smi --id="$1" --query-gpu=memory.total,memory.used \
    --format=csv,noheader,nounits | awk -F', *' '{print $1-$2}'
}

wait_for_slot() {   # wait_for_slot <gpu> <needed_mib>
  local gpu=$1 needed=$2 waited=0
  while true; do
    local free
    free=$(free_mib "$gpu")
    if [ "$free" -ge $(( needed + RESERVE_MIB )) ]; then return 0; fi
    if [ $(( waited % 300 )) -eq 0 ]; then
      echo "[$(date '+%H:%M:%S')] gpu$gpu has ${free}MiB free, waiting for $(( needed + RESERVE_MIB ))MiB"
    fi
    sleep "$POLL_SECONDS"
    waited=$(( waited + POLL_SECONDS ))
  done
}

emptier_gpu() {   # print whichever card has more room
  local a b
  a=$(free_mib 0); b=$(free_mib 1)
  [ "$a" -ge "$b" ] && echo 0 || echo 1
}

# ---- which cards may be used right now ------------------------------------
# The user reclaims GPU 1 at a fixed hour, so "which GPU" is not a constant any
# more.  `gpu_curfew.sh` rewrites GPU_POLICY at the cutoff and kills whatever is
# still on the reclaimed card; every queue restarts through the supervisor and
# reads the policy fresh, so nothing has to be edited by hand at the deadline.
#
# The file is the single source of truth.  A missing file means both cards, so a
# fresh checkout behaves as before.
GPU_POLICY=${GPU_POLICY:-/mnt/share/mwang49/data7/results/direct100k/baselines/GPU_POLICY}

allowed_gpus() {
  if [ -f "$GPU_POLICY" ]; then
    tr -s ' \n' ' ' < "$GPU_POLICY" | sed 's/^ *//;s/ *$//'
  else
    echo "0 1"
  fi
}

pick_gpu() {   # pick_gpu <preferred> -> the preferred card if allowed, else the first allowed
  local preferred=$1 allowed
  allowed=$(allowed_gpus)
  for g in $allowed; do
    [ "$g" = "$preferred" ] && { echo "$preferred"; return 0; }
  done
  echo "$allowed" | awk '{print $1}'
}

gpu_is_allowed() {   # gpu_is_allowed <gpu>
  for g in $(allowed_gpus); do [ "$g" = "$1" ] && return 0; done
  return 1
}
