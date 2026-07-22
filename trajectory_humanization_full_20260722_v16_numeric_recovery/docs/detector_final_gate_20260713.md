# Detector final gate handoff（2026-07-13）

## 1. 当前结论

Detector 负责范围的代码、协议门禁和五类真实语料小型端到端验证均已通过；**尚未启动正式
25-pair benchmark**。完整 repository-wide `unittest discover` 暂未执行，因为生成管线仍有
一个有界共享修改在进行；待 E2E agent 发出 stable signal 后再运行，避免把并发写入造成的
瞬态状态误判为最终结果。

## 2. `maxDevSigned` 决策证据

- 论文 Appendix B.3 定义：maximum **signed** perpendicular deviation。
- 审计公开 repo commit：`8924fa3e687af6f264d3b91a2d2f48faf8adfd8c`。
- 审计文件：
  `/home/mwang49/real-human/imu_gen/final/trajectory_humanization_ahb_20260712/third_party/ahb/analysis/lib/feature_library.py`。
- 公开函数 `f21_largest_signed_deviation` 的 docstring/注释写 signed，但首个实际 return 是
  `np.max(np.abs(devs))`；后面的 signed 代码不可达。
- 本项目选择论文语义，冻结策略：
  `paper_signed_value_at_argmax_absolute_deviation`。
- 非对称锁定样例 `[0,-3,-1,2,0]`：本项目/论文结果 `-3`，该公开 commit 实际结果 `+3`。

因此只称 **AHB-paper-aligned clean-room**，不称官方公开代码精确数值复现。

## 3. 真实 corpus smoke 证据

结果目录：

```text
results/real_corpus_bundle_v2_smoke_20260713/
  tap.{npz,json}
  scroll.{npz,json}
  swipe.{npz,json}
  pinch.{npz,json}
  keystroke.{npz,json}
  summary.json
  summary.md
```

| action | authoritative source events | Feature dim | adapter/bundle v2 | Feature val-only flow | TCN | Transformer |
| --- | ---: | ---: | --- | --- | --- | --- |
| tap | 19,269 | 24 | pass | pass | finite | finite |
| scroll | 59,937 | 24 | pass | pass | finite | finite |
| swipe | 70,431 | 24 | pass | pass | finite | finite |
| pinch | 58,016 | 49 | pass | pass | finite | finite |
| keystroke | 49,158 | 34 | pass | pass | finite | finite |

总计 256,811 个 authoritative source events。每类实际完成：真实 archive adapter →
`trajectory_pad_bundle_v2` save/load → Feature linear SVM validation-only threshold flow → raw
TCN/Transformer forward。label-1 是 real row 镜像，只用于二分类 API smoke；这些 FA/FRR/AUC
没有科学含义，不得写入生成器结果。

## 4. 正式 detector 合同

- exactly 25 action/detector pairs；
- Deep exactly 40 epochs，`patience=0`，history 必须连续为 1..40；
- exactly 500 user-level bootstrap replicates；
- checkpoint 与 operating threshold 都只由 validation 选择；
- test 只应用固定阈值和绘制完整曲线；
- Deep 必须先通过相同 dataset/model/device 的 longest-real-or-fake-event no-truncation batch probe；
- pair schema 固定为 `trajectory_pad_pair_v2`；supervisor、pair runner、merge 和独立 audit
  必须使用同一 schema，不能把 v2 结果误判为未完成后循环重跑；
- Deep best/last checkpoint 绑定 dataset SHA、fixed split SHA、`real_hash_seed` 和完整 pair
  config；Feature 结果若缺少 current pair manifest 必须归档并重新训练，不能直接复用旧 score；
- 每个 val/test score row 必须按 `row_index`（Feature）或 `sample_id`（Deep）与当前 dataset
  精确重连；500 个 bootstrap replicate arrays 必须按固定 seed 从 score dump 逐值重算；
- merge 后必须运行独立 pair-tree audit，验证 25 pairs、50 operating rows、10 macro rows、
  25 plots、score/bootstrap 重算和全部 output hashes；最终报告内嵌完整 50 行指标、25 张图，
  并逐 pair 保存 10 个 Deep probe 选出的真实 batch size，不把它们折叠成单一最大值。

独立 audit 命令：

```bash
python scripts/audit_trajectory_pair_merge.py \
  --experiment-root /ABS/PATH/TO/benchmark
```

最终必须同时存在：

```text
benchmark/merged/benchmark_manifest.json
benchmark/merged/benchmark_audit.json   # status=passed
```

## 5. 涉及文件

核心逻辑：

- `trajectory/features.py`
- `detectors/deep_pad.py`
- `detectors/trajectory_adapter.py`
- `detectors/pair_runner.py`
- `detectors/pair_merge.py`
- `scripts/run_trajectory_pair.py`
- `scripts/build_trajectory_pad_bundle.py`
- `scripts/probe_deep_batch_size.py`
- `scripts/audit_trajectory_pair_merge.py`
- `scripts/smoke_real_corpus_bundle_v2.py`
- `scripts/merge_real_corpus_bundle_v2_smoke.py`

测试：

- `tests/test_features.py`
- `tests/test_deep_pad.py`
- `tests/test_feature_pad.py`
- `tests/test_trajectory_adapter.py`
- `tests/test_batch_probe.py`
- `tests/test_bundle_builder.py`
- `tests/test_extractor_semantics.py`
- `tests/test_pair_runner.py`
- `tests/test_pair_merge.py`

协议文档：

- `docs/feature_protocol.md`
- `docs/detector_protocol.md`
- `docs/deep_benchmark_protocol.md`
- `docs/deep_benchmark_data_gate_20260713.md`
- `docs/benchmark_audit.md`
- 本文件。

## 6. 已执行测试

语法门禁：

```bash
python -m py_compile \
  trajectory/features.py detectors/pair_runner.py detectors/pair_merge.py \
  scripts/audit_trajectory_pair_merge.py \
  scripts/smoke_real_corpus_bundle_v2.py \
  scripts/merge_real_corpus_bundle_v2_smoke.py
```

Detector 定向测试：

```bash
python -m unittest -v tests.test_features
python -m unittest -v tests.test_pair_runner
python -m unittest -v tests.test_pair_merge
python -m unittest -v \
  tests.test_deep_pad tests.test_feature_pad tests.test_trajectory_adapter \
  tests.test_batch_probe tests.test_bundle_builder tests.test_extractor_semantics
```

当前独立测试集合共 61 tests，全部通过；无 failure/error/skip。观察到的输出仅有 PyTorch
`torch.load(weights_only=False)` FutureWarning 与 Transformer nested-tensor 性能 warning，均非
数值或协议失败。

真实 corpus smoke 命令由 `scripts/smoke_real_corpus_bundle_v2.py --action <action>` 对五动作
分别执行，再由：

```bash
python scripts/merge_real_corpus_bundle_v2_smoke.py \
  --input-dir results/real_corpus_bundle_v2_smoke_20260713
```

做 hash/schema/finiteness 聚合复核，结果 `summary.json: status=passed`。

## 7. 尚未完成 / 阻塞项

1. 等待生成管线共享文件 stable signal 后运行完整 repository-wide discovery；
2. 正式 100,000 neural fake archive 尚未完成，因此 formal bundle 尚未构建；
3. 10 个 Deep longest-event batch probes 尚未运行；
4. 25 个 formal pairs、strict merge 和 merge 后 independent audit 均尚未启动。

这些是正式 benchmark 的前置产物/运行项，不是当前 detector smoke 的失败。按要求当前不启动
formal。
