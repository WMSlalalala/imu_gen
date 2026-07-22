# 正式训练 20 分钟健康检查

`scripts/report_formal_training_health.py` 是独立、只读的正式训练监控器。正式训练启动后，
root agent 约每 20 分钟调用一次；小规模 smoke test 仍应连续监督。它不会启动、停止或恢复
训练，也不会加载/修改 checkpoint tensor、模型、数据、loss、阈值或实验指标。

## 单次检查（root agent 的标准调用）

```bash
cd /home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713
/home/mwang49/miniconda3/envs/hml/bin/python \
  scripts/report_formal_training_health.py \
  --config orchestration/formal_pipeline_config.json \
  --once
```

每次原子替换两个“latest”文件，不累积大量历史日志：

```text
results/formal_100epoch_100k_20260713/monitoring/training_health_latest.json
results/formal_100epoch_100k_20260713/monitoring/training_health_latest.md
```

也支持独立的 20 分钟循环；通常由 root agent 单次调用更便于结合结果判断：

```bash
/home/mwang49/miniconda3/envs/hml/bin/python \
  scripts/report_formal_training_health.py \
  --config orchestration/formal_pipeline_config.json \
  --interval-seconds 1200
```

默认即使发现错误也返回 0，保证监控循环本身不会因一次告警消失。自动化需要非零返回码时，
显式加 `--fail-on-error`；这仍然只报告，不会改训练。

## 检查内容

- 读取 `formal_pipeline_config.json` 和 `supervisor_status.json`，检查 supervisor/job heartbeat、
  PID 是否存在且命令是否仍对应当前 action，默认 heartbeat 超过 600 秒判定为错误。
- 对每个 action 读取小文件 `run_manifest.json`、`metrics.jsonl`、`best_manifest.json`、
  `training_progress.json` 和 `last_state.json`。监控器不用 `torch.load` 打开大 checkpoint，
  但会流式计算 `last.pt` 和全部不可覆盖 best checkpoint 的 SHA-256，并核对
  size、role、EMA 推理声明、source/config 与 sidecar progress。
- `training_progress.json` 必须来自当前 PID/run-instance，并有有限的 loss/grad norm、
  合法计数、时间顺序和最后成功进度；worker 有活 PID 但 600 秒没有有效进度为硬错误。
- train epoch 必须从 1 连续递增、无重复；每个 epoch 都必须标记完整消费 train split，且
  example/batch/valid-feature 计数有效并与 run manifest 一致。
- train/validation loss 必须为有限正数；检查最近一次变化和“前 5 个 epoch 中位数 vs 最近
  5 个 epoch 中位数”的 rolling trend。单次 loss 上升只记 warning，不自动判失败；rolling
  中位数恶化超过 10% 也先 warning，由人结合后续点判断。
- 20/40/60/80/100 epoch 的 validation 必须完整、使用 EMA、loss 有限；到达 milestone 后缺失
  validation 判错误。若恰好处于两阶段原子 epoch commit 的 120 秒窗口，只临时 warning。
- best 必须声明 `selection_split=val`、`test_used_for_selection=false`，best epoch/loss 必须对应
  已记录的 full validation 且是目前最低 val loss；整个 immutable best history 都必须来自
  validation、严格改善且 checkpoint SHA/role/source 全部一致。best 是原子发布的不可变文件，
  因此即使 worker 正在 checkpoint commit，best 损坏也始终是硬错误。
- `last.pt -> last_state.json` 存在一个很短的两文件发布窗口：只有当前 worker 正在
  `checkpoint_commit` 且 `last.pt` 本身在 120 秒内刚替换时，sidecar 短暂不一致才降级为
  warning；旧 mismatch、空 `last.pt` 或任何 best 损坏不会被该窗口掩盖。
- 扫描每个训练日志尾部至多 8 MiB，发现 `NaN`、`Inf`、`non-finite` 或
  `FloatingPointError` 判错误。日志 mtime 会被记录，但训练器在一个 epoch 内本来不打印，
  所以不能只因日志没有变化判 stall；以 worker 自己发布的有效 progress 为主，
  supervisor/job heartbeat 只证明编排器在观察。
- 用 `nvidia-smi` 记录 GPU utilization、temperature、used/total memory。温度 85°C 起 warning、
  90°C 起 error；显存达到 98% warning。运行 action 的一次瞬时利用率低于 5% 只 warning，
  不因单个采样停止训练。

## 状态解释和处置

- `healthy`：训练正在运行且没有错误/告警。
- `warning`：训练仍可继续观察，例如单次 loss 上升、单次 GPU 利用率低或 rolling loss 变差。
- `unhealthy`：NaN/Inf、epoch 缺口、split 不完整、validation/best 来源错误、heartbeat stall、
  PID 不存在或严重 GPU 温度等硬问题。监控器不会自行“修参数”；root agent 应先核实实时日志
  与持久化状态，确认真实原因后才停止/恢复，不能为了指标改数据或测试协议。
- `not_started`/`idle`/`complete`：分别表示尚未启动、存在非运行中状态、五个 action 已完整结束。

正式训练期间每次检查至少看紧凑 Markdown 表中的 epoch、train loss、val loss、best、heartbeat、
GPU 和 E/W 数；发现 `unhealthy` 应立即定位 issue code，而不是等下一次 20 分钟轮询。
