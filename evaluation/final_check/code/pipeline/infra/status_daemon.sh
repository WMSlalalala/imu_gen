#!/bin/bash
# Write a one-glance status of every queue, every five minutes, forever.
#
# A monitor attached to an editor session dies with the session.  This does not:
# it is launched with setsid, reparents to init, and leaves STATUS.md on disk so
# a cold window can read what happened while nobody was watching.  It also keeps
# an append-only EVENTS.txt of anything that looks like a failure, so a crash at
# 3am is still legible in the morning.
set -u
B=/mnt/share/mwang49/data7/results/direct100k/baselines
STATUS=$B/STATUS.md
EVENTS=$B/EVENTS.txt

queue_state() {   # queue_state <pattern> <label>
  local n
  # pgrep -c prints 0 and exits non-zero on no match; `|| echo 0` would add a
  # second line and every numeric test downstream would error.
  n=$(pgrep -fc "$1" 2>/dev/null | head -1); n=${n:-0}
  if [ "$n" -gt 0 ]; then echo "运行中 ($n 进程)"; else echo "**已停止**"; fi
}

while true; do
  now=$(date '+%Y-%m-%d %H:%M:%S')

  {
    echo "# 后台队列状态（自动刷新，最后更新 $now）"
    echo
    echo "> 这个文件由 \`status_daemon.sh\` 每 5 分钟重写一次。"
    echo "> 如果 \"最后更新\" 是很久以前，说明守护进程本身也停了。"
    echo
    echo "| 队列 | 状态 | 恢复命令 |"
    echo "|---|---|---|"
    echo "| 消融 IMU 采样 | $(queue_state 'generate_imu_ablatio[n].py') | \`bash run_ablation_queue.sh\` |"
    echo "| TTS-GAN 重训 | $(queue_state 'run_tts_ga[n].py') | \`bash run_ttsgan_retrain.sh\` |"
    echo "| CSDI | $(queue_state 'run_csd[i].py') | \`bash run_csdi_all.sh\` |"
    echo "| ImagenTime | $(queue_state 'run_imagentim[e].py') | \`bash run_imagentime_all.sh\` |"
    echo "| 建库→格子 流水线 | $(queue_state 'pipeline_daemo[n].sh') | \`bash pipeline_daemon.sh\` |"
    echo "| 检测器格子 | $(queue_state 'run_hmog_direct100k_detector[s].py') | \`bash grid_against_final.sh <method> <modality>\` |"
    echo
    echo "## 对齐最终发布版 \`direct100k_final\`"
    echo
    echo "载体按动作分四个 bundle：keystroke / scroll / swipe / tap_and_pinch。"
    echo "每个 bundle 只换它拥有的动作，其余声明为 declined。"
    echo '```'
    for m in "$B"/final/*/bundle_manifest.json; do
      [ -f "$m" ] || continue
      method=$(basename "$(dirname "$m")")
      built=$(ls -d "$(dirname "$m")"/*/ 2>/dev/null | wc -l)
      cells=$(find "$(dirname "$m")" -name 'test_scores.jsonl' 2>/dev/null | wc -l)
      printf '  %-22s bundles=%d  已出格子=%d/90\n' "$method" "$built" "$cells"
    done
    echo '```'
    echo
    echo "## 显存"
    echo '```'
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader 2>/dev/null
    echo '```'
    echo
    echo "## 进度"
    echo
    echo "**消融缓存**（目标每个约 100,000）"
    echo '```'
    for d in "$B"/ablations/*/; do
      [ -d "$d" ] || continue
      printf '  %-18s %7d %s\n' "$(basename "$d")" \
        "$(find "$d" -name 'sample_*.npz' 2>/dev/null | wc -l)" \
        "$([ -f "$d/.complete" ] && echo COMPLETE)"
    done
    echo '```'
    echo
    echo "**TTS-GAN**"
    echo '```'
    for a in tap swipe keystroke pinch scroll; do
      f=$B/ttsgan/log_${a}_imu.txt
      s=$B/ttsgan/samples_${a}_imu.npy
      if [ -f "$s" ]; then printf '  %-10s 已出样本\n' "$a"
      elif [ -f "$f" ]; then printf '  %-10s %s\n' "$a" "$(grep -o 'Epoch [0-9]*/[0-9]*' "$f" | tail -1)"
      fi
    done
    echo '```'
    echo
    echo "**CSDI**（作者预算 200 epoch，两种采样模式共用一个模型）"
    echo '```'
    for a in tap pinch swipe scroll keystroke; do
      for m in unconditional conditional; do
        s=$B/csdi/$a/samples_${a}_${m}.npy
        f=$B/csdi/$a/log_${a}_${m}.txt
        if [ -f "$s" ]; then printf '  %-10s %-14s 已出样本\n' "$a" "$m"
        elif [ -f "$f" ]; then printf '  %-10s %-14s %s\n' "$a" "$m" "$(grep -oE 'epoch [0-9]+/[0-9]+' "$f" | tail -1)"
        fi
      done
    done
    echo '```'
    echo
    echo "**ImagenTime**（早停：每 25 epoch 比一次 gap，连续 3 次无改善即停；实测停在 374-574 epoch，作者固定 1000。逐动作停止点见 final/imagentime/README.md）"
    echo '```'
    for a in tap pinch swipe scroll keystroke; do
      s=$B/imagentime/$a/samples_${a}_imu.npy
      f=$B/imagentime/$a/log_${a}.txt
      if [ -f "$s" ]; then printf '  %-10s 已出样本\n' "$a"
      elif [ -f "$f" ]; then printf '  %-10s %s\n' "$a" "$(grep -oE 'epoch [0-9]+/[0-9]+' "$f" | tail -1)"
      fi
    done
    echo '```'
    echo
    echo "## 最近的里程碑"
    echo '```'
    tail -12 "$B/PROGRESS.txt" 2>/dev/null
    echo '```'
  } > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"

  # Anything that reads like a failure gets appended once, with a timestamp.
  for log in "$B"/ablation_queue3.log "$B"/ttsgan_retrain4.log \
             "$B"/csdi_queue.log "$B"/imagentime_queue.log \
             "$B"/csdi/*/log_*.txt "$B"/imagentime/*/log_*.txt; do
    [ -f "$log" ] || continue
    grep -hoE "ERROR [a-z]+ [a-z/]+|CUDA out of memory|Traceback \(most recent call last\)|WARNING cache [a-z0-9_]+ drew only [0-9]+" \
      "$log" 2>/dev/null | sort -u | while read -r line; do
        key="$(basename "$log"): $line"
        grep -qF "$key" "$EVENTS" 2>/dev/null || echo "[$now] $key" >> "$EVENTS"
      done
  done

  sleep 300
done
