#!/bin/bash
# Run the remaining GPU work in priority order instead of all at once.
#
# WHY ORDERING MATTERS MORE THAN PARALLELISM HERE
# -----------------------------------------------
# Both cards already sit at 100% utilisation with about twenty processes on
# them, so adding more work does not make anything finish sooner -- it makes
# every job finish later, together.  A scroll training step that should cost
# tens of milliseconds was measured at 6.5 seconds under that load; roughly all
# of it is waiting for other processes.
#
# The total GPU work is what it is.  What can be chosen is the order results
# arrive in, so the numbers the paper needs first are not stuck behind the ones
# it needs last.
#
# The order below is by how much the paper depends on each result:
#
#   1. the release's own ablations (A7, then the k_refs sweep) -- these are the
#      claims about the method itself, and nothing substitutes for them
#   2. the third-party baselines already trained or nearly so (TTS-GAN, CSDI)
#   3. ImagenTime, the most expensive and the last one the tables need
#
# Each stage waits for the previous to drain rather than being killed, so no
# checkpoint is lost and every queue stays resumable on its own.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
say(){ echo "[$(date '+%m-%d %H:%M:%S')] scheduler: $*" | tee -a "$B/PROGRESS.txt"; }

wait_for() {   # wait_for <pattern> <label>
  local pattern=$1 label=$2 waited=0 n
  # `pgrep -c` prints 0 and *also* exits non-zero when nothing matches, so the
  # usual `|| echo 0` appended a second line and every comparison below died
  # with "integer expression expected" -- the loop then fell through by accident
  # rather than by the condition being false.
  n=$(pgrep -fc "$pattern" 2>/dev/null | head -1); n=${n:-0}
  while [ "$n" -gt 0 ]; do
    if [ $(( waited % 1800 )) -eq 0 ]; then
      say "waiting for $label ($(pgrep -fc "$pattern") still running)"
    fi
    sleep 120
    waited=$(( waited + 120 ))
    n=$(pgrep -fc "$pattern" 2>/dev/null | head -1); n=${n:-0}
  done
  say "$label done"
}

# ---- stage 1: the method's own ablations ---------------------------------
# A7 and the ablation sampler are already running; this only waits on them so
# the heavier baselines do not compete with them for the cards.
wait_for "run_a7_weighted_su[m].py" "A7 (weighted-sum arm)"
wait_for "generate_imu_ablatio[n].py" "ablation sampling (noshot_adv, fewshot_nonadv, k_refs 1/3/8)"

# ---- stage 2: the baselines that are already most of the way there --------
wait_for "run_tts_ga[n].py" "TTS-GAN"
wait_for "run_csd[i].py" "CSDI"

# ---- stage 3: ImagenTime, alone on both cards ----------------------------
# Only if it is not already running.  The supervisor restarts this scheduler
# when it dies, and an unconditional launch here started a second ImagenTime on
# top of the first: the newcomer resumed each action from its checkpoint with an
# empty convergence monitor, which reset the patience counter and made early
# stopping unable to fire.
if pgrep -f "run_imagentime\.py" >/dev/null 2>&1; then
  say "ImagenTime already running; not starting a second one"
  wait_for "run_imagentim[e]\.py" "ImagenTime (already running)"
else
  say "starting ImagenTime with the cards to itself"
  bash "$C/run_imagentime_all.sh" >> "$B/imagentime_queue.log" 2>&1
fi
say "ImagenTime done"

say "all generative work complete; the grid queue drains what is left"
