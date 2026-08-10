#!/bin/bash
# Keep the machine working: restart any queue that dies while it still has work.
#
# Every queue here is idempotent and resumable -- that was built in deliberately
# -- so restarting one costs at most the checkpoint interval and never redoes
# finished work.  What it buys is that a queue killed by an OOM, a transient CUDA
# error, or a stray signal does not leave both cards idle until somebody notices.
#
# Two things this must NOT do:
#
#   * restart ImagenTime.  `scheduler.sh` deliberately holds it until the
#     ablations and the cheaper baselines have drained, because twenty processes
#     sharing two cards made a 30 ms training step cost 6.5 seconds.  The
#     supervisor restarts the scheduler instead and lets it decide.
#
#   * restart a queue whose work is finished.  Each check below asks whether
#     there is anything left to do, not merely whether a process is alive.
#
# Patterns are bracketed (run_ablation_queu[e].sh) so pgrep cannot match this
# script's own command line -- an unbracketed pattern here counts itself and the
# supervisor concludes everything is fine forever.
set -u
C=/mnt/share/mwang49/data7/code/baselines
B=/mnt/share/mwang49/data7/results/direct100k/baselines
say(){ echo "[$(date '+%m-%d %H:%M:%S')] supervisor: $*" | tee -a "$B/PROGRESS.txt"; }

# `pgrep -fc` prints "0" *and* exits non-zero when nothing matches, so
# `$(pgrep -fc ... || echo 0)` yields two lines and the numeric test errors out.
# It happened to behave correctly -- the error made the test false, which reads
# as "not alive" -- but a supervisor whose liveness check works by accident is
# one refactor away from either never restarting or restarting constantly.
# Exit status alone answers the question and cannot be misread.
alive() { pgrep -f "$1" >/dev/null 2>&1; }

start() {   # start <script> <logfile>
  setsid nohup bash "$C/$1" >> "$B/$2" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

# ---- is there still work for each queue? ---------------------------------
ablation_pending() {
  for tag in noshot_adv fewshot_nonadv krefs1 krefs3 krefs8; do
    [ -f "$B/ablations/$tag/.complete" ] || return 0
  done
  return 1
}

ttsgan_pending() {
  for a in tap swipe keystroke pinch scroll; do
    [ -f "$B/ttsgan/samples_${a}_imu.npy" ] || return 0
  done
  return 1
}

csdi_pending() {
  # Only the arms the driver actually draws.  Waiting on the dropped five-shot
  # arm would leave this permanently pending and the queue permanently
  # restarting.
  for a in tap pinch swipe scroll keystroke; do
    [ -f "$B/csdi/$a/samples_${a}_unconditional.npy" ] || return 0
  done
  return 1
}

a7_pending() {
  # "Finished" is having reached the target epoch count, NOT having a
  # best_*.pt.  The trainer writes best_* whenever validation improves, which
  # happens within the first few epochs -- so the earlier version of this check
  # declared A7 complete at epoch 123 of 160, the curfew stopped it, and the
  # supervisor then refused to bring it back.  The arm would have quietly never
  # finished.
  #
  # The target comes from the ablation plan the driver wrote, and the progress
  # from the trainer's own log, so neither is guessed.
  local plan=$B/ablations/a7_weighted_sum/a7_plan.json
  [ -f "$plan" ] || return 0
  for a in scroll swipe; do
    local run=/mnt/share/mwang49/real-human/imu_gen/final/runs/$a/diffusion/fewshot_adv/${a}_a7_weighted_sum
    local log=$run/train_log.jsonl
    [ -f "$log" ] && return 0   # exists but unfinished -> pending; see below
  done
  return 1
}

# The loop above returns "pending" as soon as any log exists, which is too
# eager once an arm really is done.  The precise test needs python, so it is
# kept separate and consulted only when a log is present.
a7_unfinished() {
  /home/mwang49/miniconda3/envs/cuhkx/bin/python - "$B" <<'PYEOF'
import json, re, sys
from pathlib import Path
B = Path(sys.argv[1])
plan = B / "ablations/a7_weighted_sum/a7_plan.json"
if not plan.is_file():
    print("yes"); raise SystemExit
targets = {r["action"]: int(r["epochs"]) for r in json.loads(plan.read_text())}
root = Path("/mnt/share/mwang49/real-human/imu_gen/final/runs")
for action, total in targets.items():
    log = root / action / "diffusion/fewshot_adv" / f"{action}_a7_weighted_sum" / "train_log.jsonl"
    if not log.is_file():
        print("yes"); raise SystemExit
    epochs = re.findall(r'"epoch": (\d+)', log.read_text(errors="ignore"))
    if not epochs or int(epochs[-1]) < total - 1:
        print("yes"); raise SystemExit
print("no")
PYEOF
}

grid_pending() {
  [ -f "$B/GRID_JOBS.txt" ] || return 1
  local todo=0
  while read -r line; do
    case "$line" in ''|\#*) continue ;; esac
    grep -qxF "$line" "$B/GRID_JOBS_DONE.txt" 2>/dev/null || todo=1
  done < "$B/GRID_JOBS.txt"
  [ "$todo" -eq 1 ]
}

say "started; watching every 3 minutes"
while true; do
  # --- the queues the supervisor owns directly --------------------------
  if ablation_pending && ! alive "run_ablation_queu[e].sh"; then
    say "RESTART ablation queue"
    start run_ablation_queue.sh ablation_queue4.log
  fi
  if ttsgan_pending && ! alive "run_ttsgan_retrai[n].sh"; then
    say "RESTART TTS-GAN queue"
    start run_ttsgan_retrain.sh ttsgan_retrain4.log
  fi
  if csdi_pending && ! alive "run_csdi_al[l].sh"; then
    say "RESTART CSDI queue"
    start run_csdi_all.sh csdi_queue.log
  fi
  if [ "$(a7_unfinished)" = "yes" ] && ! alive "run_a7_queu[e].sh"; then
    say "RESTART A7 queue"
    start run_a7_queue.sh a7_queue.log
  fi
  if grid_pending && ! alive "grid_queu[e].sh"; then
    say "RESTART grid queue"
    start grid_queue.sh grid_queue.log
  fi

  # --- the daemons ------------------------------------------------------
  alive "pipeline_daemo[n].sh" || { say "RESTART pipeline daemon"; start pipeline_daemon.sh pipeline_daemon.log; }
  alive "ablation_pipelin[e].sh" || { say "RESTART ablation pipeline"; start ablation_pipeline.sh ablation_pipeline.log; }
  alive "crossscore_queu[e].sh" || { say "RESTART crossscore queue"; start crossscore_queue.sh crossscore_queue.log; }
  # A7's own path from trained generator -> cache -> dataset -> grid.  Nothing
  # else watches for its checkpoints, so without this the arm would train and
  # then simply stop.
  if ! alive "a7_pipelin[e].sh" && [ ! -f "$B/final/abl_a7_weighted_sum/bundle_manifest.json" ]; then
    say "RESTART a7 pipeline"
    start a7_pipeline.sh a7_pipeline.log
  fi
  # A8-A11 likewise: four arms that train, sample, build and queue themselves.
  # Done when the last arm's bundle exists -- restarting it after that would
  # re-enter a loop whose work is finished.
  if ! alive "critic_pipelin[e].sh" \
     && [ ! -f "$B/final/abl_a11_no_feature_match/bundle_manifest.json" ]; then
    say "RESTART critic pipeline"
    start critic_pipeline.sh critic_pipeline.log
  fi
  # The curfew is NOT restarted automatically any more.
  #
  # It existed for a one-day instruction ("GPU 1 goes back to the user at 17:00")
  # and it works by rewriting GPU_POLICY.  The standing instruction since
  # 2026-08-09 is the opposite -- both cards are usable, with 10 GB kept free on
  # GPU 1 -- so an armed curfew would silently revert that at the cutoff and kill
  # whatever was running there.  Worse, the old trigger was "GPU_POLICY mentions
  # 1", which is now true by design, so it would have re-armed itself every day.
  #
  # `gpu_curfew.sh` is left in place for the next time a card has to be handed
  # back; arm it deliberately and it will do the job:
  #     CUTOFF=17:00 RECLAIM=1 KEEP=0 setsid nohup bash gpu_curfew.sh &
  alive "status_daemo[n].sh"   || { say "RESTART status daemon";   start status_daemon.sh /dev/null; }

  # --- the scheduler, which owns ImagenTime's start time -----------------
  # Only worth having while something is still staged behind it.
  if ! alive "schedule[r].sh"; then
    if [ ! -f "$B/imagentime/keystroke/samples_keystroke_imu.npy" ]; then
      say "RESTART scheduler (ImagenTime still staged behind it)"
      start scheduler.sh scheduler.log
    fi
  fi

  sleep 180
done
