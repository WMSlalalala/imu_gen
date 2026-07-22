# Trajectory estimator pack 20260721

本目录把之前 AHB-style trajectory benchmark 的检测方式封装成可调用的“估计器”接口。

它不改生成器，不使用 generator critic，也不从 test set 反选阈值。核心目标是：

- 离线：按之前 benchmark 协议训练/export detector artifacts；
- 运行时：给定一条 Android/Type-B 风格触摸轨迹，输出 fake-high score、阈值判定、FA/FRR 协议来源和调用耗时；
- 审计：保留 train-only 拟合、validation-only 阈值、test-only 汇报的边界。

关键入口：

- `docs/IMU与轨迹交付状态及问题清单.md`
- `docs/下一步实施方案.md`
- `estimator/feature_estimator.py`
- `estimator/deep_estimator.py`
- `estimator/total_detector.py`
- `estimator/service.py`
- `scripts/train_feature_estimator_artifacts.py`
- `scripts/estimate_trajectory.py`
- `scripts/smoke_estimator_pack.py`
- `scripts/smoke_total_detector.py`
- `scripts/build_paired_detector_table.py`
- `scripts/build_real_pair_indices.py`
- `scripts/build_real_consistency_components.py`
- `estimator/fake_event_plan_archive.py`
- `estimator/fake_imu_pairs.py`
- `scripts/generate_paired_fake_imu.py`
- `scripts/audit_paired_fake_imu.py`
- `scripts/build_fake_consistency_components.py`
- `scripts/merge_detector_components.py`
- `scripts/remap_component_meta_pools.py`
- `scripts/build_trajectory_score_component.py`
- `scripts/build_paired_imu_features.py`
- `scripts/train_paired_imu_scorers.py`
- `scripts/audit_trajectory_kinematics.py`
- `orchestration/formal_total_config_20260722.json`
- `orchestration/formal_total_supervisor.py`
- `scripts/train_total_detector.py`
- `docs/总检测器与联合生成方案.md`
- `docs/共享EventPlan、Trajectory生成与TotalDetector.md`
- `runtime/trajectory_layer.py`
- `runtime/paired_layer.py`
- `scripts/run_paired_event.py`

Smoke：

```bash
cd /home/mwang49/real-human/imu_gen/final/trajectory_estimator_pack_20260721
python scripts/smoke_estimator_pack.py
python scripts/smoke_total_detector.py
```

正式 paired fake 链路不会把 trajectory 与任意 IMU 按行拼接。它先从 trajectory numeric archive 重放并核对原始 `ConditionRequest` 和 shared `EventPlan` digest，再用该 plan 的 user/action/duration/orientation/geometry/text/独立 IMU noise seed 调用在线 five-shot IMU diffusion。每个用户动作单元原子落盘，并要求五个 modality-specific IMU references 在该单元内固定。

`results/paired_fake_imu_online_smoke_20260721/tap/user_000.npz` 是一次真实 checkpoint/CUDA 在线集成 smoke，明确标记 `formal_result=false`。正式模式固定 100,000 个 paired fake 事件，必须在正式 trajectory 100k 审计通过后才能启动。
