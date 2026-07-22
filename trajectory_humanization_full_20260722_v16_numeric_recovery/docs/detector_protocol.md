# Trajectory Feature PAD 严格测试协议

## 1. 目标和边界

`detectors/feature_pad.py` 对**预先提取好的数值特征**执行 feature-level PAD。
它不读取 NPZ、不解析轨迹点，也不依赖 `trajectory/features.py` 的具体特征维数。
因此同一个协议可用于：

- AHB-style 单指 24 维；
- pinch 49 维扩展；
- keystroke 34 维扩展；
- 后续冻结的其他特征表。

每次正式调用只训练一个 action、一个 detector。输入包含五个等长数组：

```text
features : float [N,D]
labels   : int   [N]      real=0, fake=1
user_ids :       [N]
pools    : str   [N]      train / val / test
actions  : str   [N]
```

所有特征必须有限。每个目标 action 的 train、validation、test 都必须同时包含 real 和
fake；否则不能定义二分类 AUC/FA/FRR，协议直接停止。
`user_ids` 必须是同一种数值或字符串 dtype；拒绝混合 Python object，以保证 score dump
可在 `allow_pickle=False` 下安全加载。

正式数值输入必须来自 `trajectory_pad_bundle_v2`，并携带
`feature_schema_version=trajectory_features_v2_ahb_table6_hmog_real_up`。该版本把 AHB
Table 6 的 acceleration 固定为 signed `Δspeed/Δt`、`dev20/50/80` 固定为 signed
perpendicular deviation、末项固定为前 5 个 acceleration samples；旧 bundle v1 被 loader
拒绝，不能与本协议结果混合。

## 2. 三类检测器

| `detector_kind` | 实现 | fake-high score |
| --- | --- | --- |
| `linear_svm` | `sklearn.svm.SVC(kernel="linear")` | class 1 的 `decision_function` |
| `rbf_svm` | `sklearn.svm.SVC(kernel="rbf")` | class 1 的 `decision_function` |
| `xgboost` | `xgboost.XGBClassifier` | `P(class=1)` |

模型训练完成后硬检查 `classes_ == [0,1]`，防止分数方向被标签排序悄悄反转。

XGBoost 是可选依赖。如果调用 `xgboost` 但环境没有安装或导入失败，模块抛出包含原始
import error 的 `RuntimeError`；绝不静默改成 SVM，也不省略该 detector。

## 3. 严格的数据流

```text
train features
    │
    ├── fit StandardScaler
    │
    └── transform train → fit detector

validation features
    │
    ├── scaler.transform（不 fit）
    ├── detector score
    └── 只在 validation 上选择 EER 与 FRR<=5% 阈值

test features
    │
    ├── scaler.transform（不 fit）
    ├── detector score
    └── 只应用 validation 已冻结阈值
```

`StandardScaler.fit` 只接收目标 action 的 train rows。validation、test 和输入中的其他
action 都不参与均值/方差估计。

test 上可以扫阈值绘制完整 FA–FRR 曲线，但曲线只用于可视化，不能从 test 曲线重新选
阈值、改 detector 或挑选 fake。

## 4. 分数方向和判定边界

统一约定：

```text
score 越大 -> 越像 fake
score < threshold  -> 接受为 real
score >= threshold -> 拒绝为 fake
```

因此：

```text
FRR = count(real_score >= threshold) / n_real
FA  = count(fake_score <  threshold) / n_fake
AUC = ROC-AUC(label, fake_high_score)
```

等于 threshold 的样本一定被拒绝。`operating_metrics` 是 selection、test 和 bootstrap
共用的唯一实现，避免三个阶段出现 `<`/`<=` 不一致。

## 5. Validation 阈值选择

先对 validation 的唯一 score 构造所有可达到的接受集合；相邻 score 之间使用中点，
并包括“全部拒绝”和“全部接受”端点。重复 score 作为一个整体跨越，不拆散 tie。

### 5.1 EER threshold

选择顺序固定为：

1. 最小化 `abs(FA-FRR)`；
2. 若并列，最小化 `max(FA,FRR)`；
3. 再并列，选择较小 threshold。

主 EER 数值可以报告 `(FA+FRR)/2`，但结果文件保存的是该 validation threshold 以及
threshold 分别应用到 validation/test 后的真实 FA 和 FRR。

### 5.2 Validation FRR<=5% threshold

只保留 validation 上 `FRR <= 0.05` 的候选，并按以下顺序选择：

1. 最小 validation FA；
2. FA 并列时选择最接近 5% 边界、即 FRR 较大的点；
3. 再并列选择较小 threshold。

选择完成后 threshold 固定。test FRR 不保证仍然小于 5%，因为 distribution shift 正是
独立测试需要揭示的现象。
正式 `run_feature_pad_protocol` 将目标硬固定为 `0.05`；不能通过参数把主协议悄悄改成
其他操作点。底层 threshold 函数保留一般化参数仅供明确命名的消融实验使用。

## 6. Score dump 与曲线

`run_feature_pad_protocol` 返回：

- `thresholds`：validation 选择的 EER 和 FRR<=5% 阈值；
- `validation_metrics`：两个固定 operating points；
- `test_metrics`：相同阈值在 test 的结果；
- `score_dumps[val/test]`：`score,label,user_id,pool,action,row_index`；
- `curves[val/test]`：全部 `threshold,fa,frr`；
- 可选 user-level bootstrap 结果。

`save_protocol_outputs` 原子写入：

```text
summary.json
score_dump.npz
curves.npz
bootstrap_summary.json       # 开启 bootstrap 时
bootstrap_replicates.npz     # 开启 bootstrap 时
```

`summary.json` 同时保存 scaler 的 train-only mean/scale、score 方向、判定边界和阈值来源，
便于最终审计。

## 7. User-level bootstrap

接口：

```python
user_level_bootstrap(
    test_labels,
    test_scores,
    test_user_ids,
    fixed_validation_thresholds,
    n_replicates=500,
    seed=...,
)
```

每次 replicate：

1. 从 test real users 中有放回抽取相同数量的 real users；
2. 从 test fake users 中独立、有放回抽取相同数量的 fake users；
3. 某 user 每被抽到一次，就保留该标签下这个 user 的**全部窗口**一次；
4. validation threshold 保持固定，不在 bootstrap sample 上重选；
5. 重新计算 FA、FRR 和 AUC。

如果一个 user 被抽中两次，其完整窗口组也出现两次。这样保留同一用户内部窗口的相关性，
不会把每个 window 错当成独立用户。

输出保存每个 replicate 的数值，以及 mean、median、2.5% 和 97.5% 分位数形成的 95% CI。

## 8. Synthetic 门禁

`tests/test_feature_pad.py` 使用纯合成数组完成整个：

```text
train -> scaler/model fit
      -> validation threshold selection
      -> fixed-threshold test
      -> user-level bootstrap
      -> atomic output bundle
```

测试明确证明：

- scaler mean 精确等于目标 action 的 train mean，且不等于全数据 mean；
- validation/test 的大幅分布偏移不会进入 scaler；
- 只改 test 可改变 test AUC，但 scaler 和 validation thresholds 完全不变；
- fake 的平均 score 高于 real；
- score 恰好等于 threshold 时严格拒绝；
- FA 随 threshold 单调不降、FRR 单调不升；
- bootstrap 抽中 user 后保留该 user 的全部窗口，重复抽中则完整重复；
- XGBoost 缺失时显式失败。

小规模测试命令：

```bash
cd /home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713
python -m unittest discover -s tests -p 'test_feature_pad.py' -v
```

该测试不读取、不训练、不覆盖任何正式轨迹或正式 detector 结果。
