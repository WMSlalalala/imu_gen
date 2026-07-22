# Trajectory Humanization（strict HMOG data + generator + PAD）

本目录包含五类 HMOG 原始触摸轨迹数据层、条件 diffusion 生成器，以及与生成器 critic
独立的 feature/raw-sequence PAD 严格评估层。

入口：

- `preprocess/extract_hmog_trajectories.py`：五动作 raw trajectory extractor；
- `tests/run_one_user_smoke.py`：真实 one-user 运行与结构/语义验证；
- `docs/轨迹数据结构与预处理说明.md`：完整中文 schema；
- `detectors/feature_pad.py`：clean-room feature SVM/XGBoost；
- `detectors/deep_pad.py`：变长 raw TCN/Transformer PAD；
- `detectors/benchmark_runner.py`：五动作 feature+deep 统一评估；
- `scripts/run_trajectory_benchmark.py`：synthetic smoke / formal bundle 入口；
- `scripts/generate_five_shot_trajectories.py`：固定 5 refs、batch=32、50-step DDIM 的 formal shard 生成入口；
- `scripts/audit_five_shot_generation.py`：schema 1.4 / shard manifest v4 / formal audit v4、strict runtime 与 `eta=0` 绑定的 500 units / 100,000 fake 完整生成审计；
- `generation/pad_export.py`：Type-B generated shards 到独立 PAD `RawTrajectoryRecord` 的唯一适配层；
- `scripts/audit_generation_pad_export.py`：生成结果在 PAD ingress 边界的 100×200/count/split/lifecycle 审计；
- `scripts/build_trajectory_pad_bundle.py`：完整 real corpus + 100k generated shards 合并为五动作 detector dataset；
- `docs/generation_protocol.md`：orientation-first、keystroke sequence/letter-XY、1 ms 时间轴、Android 与审计协议；
- `docs/deep_benchmark_protocol.md`：两类 split、raw 通道、checkpoint、阈值、
  bootstrap 和结果文件的完整协议；
- `orchestration/README.md`：100-user 正式流水线、gates-only 人工审核、断电恢复与最终闭包；
- `results/smoke_one_user/`：用户 100669 的五类 NPZ、manifest、audit 和验证报告。

脚本默认只跑一个用户。100-user 全量需要显式 `--confirm-full-run`，以防误启动。

Deep benchmark 的 `--synthetic-smoke` 只验证 pipeline，绝不作为正式 FA/AUC；正式评估
必须传五动作 formal bundle 和固定 `users_seed42.json`，并重新执行严格 real/fake pool
分配。

正式 100-epoch / 100k 运行默认未授权。必须先执行
`orchestration/launch_gates_only.sh`，审核 durable `launch_gate_evidence` 后，再显式把
`formal_launch_authorized` 设为 `true`；直接提前设为 true 也无法绕过 gates-only review point。
