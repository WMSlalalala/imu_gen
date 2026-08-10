#!/bin/bash
# Hand GPU 1 back to the user at the cutoff, without losing work.
#
# Every queue here checkpoints and resumes, so the safe way to vacate a card is
# to stop what is on it and let the supervisor start it again -- the restarted
# queue reads GPU_POLICY fresh and lands on the cards that are still allowed.
# Nothing has to be edited by hand at the deadline, and nothing is lost beyond
# the checkpoint interval.
#
# The order matters:
#   1. rewrite the policy FIRST, so anything restarting during the changeover
#      already sees the new rule;
#   2. then stop the queues, so their in-flight workers die with them rather
#      than being orphaned onto the card we are trying to free;
#   3. then kill whatever is still holding memory on the reclaimed card, which
#      catches anything started outside these queues;
#   4. leave the supervisor to bring everything back.
#
# CUTOFF is "HH:MM" local time today.  Runs once and exits.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
POLICY=$B/GPU_POLICY
CUTOFF=${CUTOFF:-17:00}
RECLAIM=${RECLAIM:-1}
KEEP=${KEEP:-0}
say(){ echo "[$(date '+%m-%d %H:%M:%S')] curfew: $*" | tee -a "$B/PROGRESS.txt"; }

target=$(date -d "$CUTOFF" +%s 2>/dev/null) || { say "bad CUTOFF '$CUTOFF'"; exit 1; }
now=$(date +%s)
if [ "$target" -le "$now" ]; then
  say "$CUTOFF has already passed today; applying immediately"
else
  say "armed: GPU $RECLAIM returns to the user at $CUTOFF ($(( (target-now)/60 )) minutes)"
  while [ "$(date +%s)" -lt "$target" ]; do sleep 30; done
fi

# 1. the rule changes before anything is stopped
echo "$KEEP" > "$POLICY"
say "GPU_POLICY is now '$KEEP' -- GPU $RECLAIM is off limits"

# 2. stop the queues so their workers go with them
for q in run_ablation_queu run_csdi_al run_imagentime_al run_a7_queu \
         grid_queu crossscore_queu; do
  pkill -f "${q}[a-z_]*\.sh" 2>/dev/null && say "stopped $q"
done
sleep 5
for w in generate_imu_ablatio run_csd run_imagentim final_gen.trai \
         run_hmog_direct100k_cel run_hmog_direct100k_detector \
         score_against_fixed_detecto; do
  pkill -f "$w" 2>/dev/null
done
sleep 10

# 3. anything still on the reclaimed card, whoever started it
for pid in $(nvidia-smi --id="$RECLAIM" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  say "killing pid $pid still holding GPU $RECLAIM"
  kill -TERM "$pid" 2>/dev/null
done
sleep 10
for pid in $(nvidia-smi --id="$RECLAIM" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -KILL "$pid" 2>/dev/null
done

# a claimed-but-dead grid slot would otherwise block the queue forever
rm -f "$B"/.grid_running/* 2>/dev/null

sleep 5
free_now=$(nvidia-smi --id="$RECLAIM" --query-gpu=memory.used --format=csv,noheader)
say "GPU $RECLAIM now at $free_now -- the supervisor will restart everything on GPU $KEEP"
