#!/bin/bash
# Wait until a card has room, so a long programme never crowds the user out.
#
# The user keeps a working reserve on both cards.  Every launcher here calls
# `wait_for_slot <gpu> <needed_mib>` before starting a job and only proceeds
# when the card has the job's footprint plus the reserve still free, so the
# reserve survives even when several stages start at once.
# The reserve is per card, because the user's ask is per card: both cards are
# available, and GPU 1 is to keep 10 GB free.  GPU 0 stays at 6 GB, the bottom of
# the 5-10 GB band asked for earlier -- deliberately at the bottom because the
# gate is checked per launch, so a job that starts just under the line can leave
# actual free memory a little below the nominal reserve; the nominal number wants
# headroom under the real floor rather than sitting on it.
#
# Standing instruction (2026-08-10, supersedes the 2026-08-09 one): GPU 0 is used
# in full with no reserve, GPU 1 keeps 10 GB free.  Holds for every later
# experiment until the user says otherwise.
#
# A zero reserve on GPU 0 is the literal instruction and is honoured, but note
# the gate is checked per launch against an estimated footprint: with no margin,
# an underestimate OOMs instead of waiting.  Raise RESERVE_MIB_0 if that bites.
RESERVE_MIB=${RESERVE_MIB:-6000}          # default for any card without its own
RESERVE_MIB_0=${RESERVE_MIB_0:-0}         # GPU 0 is ours in full (2026-08-10)
RESERVE_MIB_1=${RESERVE_MIB_1:-10240}     # 10 GB stays free on GPU 1
POLL_SECONDS=${POLL_SECONDS:-20}

reserve_mib() {   # reserve_mib <gpu> -> that card's reserve
  local named
  named=$(eval "echo \${RESERVE_MIB_$1:-}")
  echo "${named:-$RESERVE_MIB}"
}

free_mib() {   # free_mib <gpu>
  nvidia-smi --id="$1" --query-gpu=memory.total,memory.used \
    --format=csv,noheader,nounits | awk -F', *' '{print $1-$2}'
}

wait_for_slot() {   # wait_for_slot <gpu> <needed_mib>
  local gpu=$1 needed=$2 waited=0 reserve
  reserve=$(reserve_mib "$gpu")
  while true; do
    local free
    free=$(free_mib "$gpu")
    if [ "$free" -ge $(( needed + reserve )) ]; then return 0; fi
    if [ $(( waited % 300 )) -eq 0 ]; then
      echo "[$(date '+%H:%M:%S')] gpu$gpu has ${free}MiB free, waiting for $(( needed + reserve ))MiB (reserve ${reserve}MiB)"
    fi
    sleep "$POLL_SECONDS"
    waited=$(( waited + POLL_SECONDS ))
  done
}

emptier_gpu() {   # print whichever allowed card has more usable room
  # Usable means free-minus-reserve, not raw free: GPU 1 keeps a bigger reserve,
  # so comparing raw free memory would keep sending work to it while it is in
  # fact the tighter card.  Only cards the policy allows are considered.
  local best="" best_room=""
  for g in $(allowed_gpus); do
    local room
    room=$(( $(free_mib "$g") - $(reserve_mib "$g") ))
    if [ -z "$best" ] || [ "$room" -gt "$best_room" ]; then best=$g; best_room=$room; fi
  done
  echo "${best:-0}"
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
