# 五动作 Raw-sequence Deep PAD 与统一评估协议

## 1. 范围

实现文件：

- `detectors/deep_pad.py`：变长 raw trajectory 表示、train-only normalization、
  TCN/Transformer、训练、checkpoint、validation threshold、test、bootstrap；
- `detectors/benchmark_runner.py`：把三类 clean-room feature PAD 与两类 Deep PAD
  纳入同一套五动作评估；
- `scripts/run_trajectory_benchmark.py`：正式命令行入口与明确标注的 synthetic smoke；
- `scripts/build_trajectory_pad_bundle.py`：将同 schema 的 real/fake flat NPZ 转为正式
  numeric bundle，并重新执行两种 pool 规则；
- `scripts/audit_trajectory_benchmark.py`：逐 detector 重算固定阈值指标与完整曲线，
  fail-closed 审计 checkpoint/bootstrap/plot/critic-selector 边界；
- `detectors/trajectory_adapter.py`：不重采样地恢复 global frames、双指与 key gaps；
- `tests/test_deep_pad.py`：时间轴、split、normalization、checkpoint、runner 门禁。

这套 detector 与生成器的三个 adversarial critics 完全分开：不导入 critic、不把 critic
checkpoint 当 detector，也不使用 selector。生成器训练和 detector 测试是两个独立阶段。

统一标签和分数方向为：

```text
real = 0
fake = 1
score 越大 -> 越像 fake
score < threshold  -> 接受为 real
score >= threshold -> 拒绝为 fake
```

不允许根据 test 表现翻转 score，AUC 即 `ROC-AUC(real=0,fake=1,fake-high score)`。

## 2. 两种 pool 边界必须分别处理

### 2.1 Fake：固定 user 70/10/20

fake pool 固定读取：

```text
/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json
```

```text
fake train = 70 users
fake val   = 10 users
fake test  = 20 users
```

三组 user 必须互斥并覆盖 100 users。一个 fake user 的所有事件只进入一个 pool。

### 2.2 Real：每个 user/action 的完整 event group 60/20/20

real 不能套用 fake 的 70/10/20 user 划分。对每个 `(user, action)`，先对完整
`event_group_id` 计算 SHA-256 并按 hash 排序，再按组数切分：

```text
ranked complete event groups
first floor(60% * N)       -> train
next  floor(20% * N)       -> val
remaining groups           -> test
```

`complete_event_group_id` 是完整 gesture / typing event ID，不是 frame、key 或 IMU chunk
编号。同一完整 event 即便表示为多条记录，hash key 相同，所以绝不跨 pool。采用
“hash 后排序再切分”而不是每条独立 `hash % 100`，避免有限样本下某个 real user 偶然
没有 test event；HMOG 正式数据每个 user/action 样本充足时，real test 明确覆盖 100 users。

因此最终 detector pool 的含义是：

```text
train = 每个 real user/action 的约60%完整事件 + 70个fake users
val   = 每个 real user/action 的约20%完整事件 + 10个fake users
test  = 全100个real users各自held-out约20%事件 + 固定20个fake test users
```

尤其不能把 real test 错误缩成 fake test 的 20 个用户。`assign_strict_protocol_pools`
分别记录 real/fake 的事件数和用户数到 `split_audit.json`。

## 3. Raw variable-length 输入

每条 `RawTrajectoryRecord` 保留一个 event-global 时间轴：

| 字段 | shape | 含义 |
| --- | --- | --- |
| `pointer_continuous` | `[2,T,4]` | 两 pointer 的原始 x/y/pressure/size |
| `global_t_ms` | `[T]` | 整个事件共享、单调不减的物理时间；仅 zero-flight 跨键边界可相等 |
| `contact_mask` | `[2,T]` | 每个 pointer 在该全局 token 是否接触屏幕 |
| `active_mask` | `[2,T]` | extractor/generator 审计 annotation；必须是 contact 子集，但绝不输入 PAD 模型 |
| `action_code` | `[2,T]` | DOWN/UP/MOVE/POINTER_DOWN/POINTER_UP |
| `keycode` | `[2,T]` | keystroke contact 的 keycode，缺失为 -1 |
| `event_ids` | `[2,T]` | 离散 contact ID；gap/no-contact 为 -1 |
| `gap_mask` | `[T]` | keystroke 的无接触 flight token |
| `event_group_id` | scalar | 用于 real 60/20/20 分组的完整 event ID |

batch 只在右侧 pad 到该 batch 最大 `T`，`frame_mask` 告诉网络真实 token；不会把正式数据
裁成一个固定长度，也不会把 padding 当成轨迹。

### 3.1 Pinch

两指共用原始 global timestamp。pointer 0 可以先 DOWN，pointer 1 稍后 POINTER_DOWN；
pointer 1 可以先 POINTER_UP，pointer 0 最后 UP。这些错位由两行独立 `contact_mask` 和
`action_code` 保留。

禁止对两根手指分别做：

```text
pointer_i_time -> 独立归一为 0..duration
```

这样做会伪造两指同时落下/抬起并删除真实 lead-in/lead-out。当前只增加一个共享
`time_progress=(global_t-start)/(global_end-start)` 通道，所有指针看到同一物理时刻。

### 3.2 Keystroke

打字不是一条连续滑动轨迹：

```text
key contact 0 -> [positive-flight gap] -> key contact 1 -> ...
```

每次 contact 有独立 `event_ids` 和 canonical keycode；raw negative sentinel
（-1/-2/-5）统一编码为 token 0，flight/no-contact 才保留 `keycode=-1`。正 flight 插入
`gap_mask=1`、`contact_mask=0` 的 token，并用 global `log_dt` 保存 flight 时长；若前一键
UP 与后一键 DOWN 同毫秒，则保留两个相邻 contact token 和相同物理时间，`log_dt=0`，
不虚构 gap 或时间。相等时间只允许发生在不同、递增的 key event ID 边界；同一 key 内
重复时间以及非 keystroke 重复时间一律拒绝。gap 没有 XY，禁止在两个字母之间插值出
一条屏幕直线。keycode 进入独立 embedding，不当连续数值做 StandardScaler。

HMOG 字母码是 ASCII 65–90/97–122，不是 Android KEYCODE_A..Z 29–54。Feature PAD 将
ASCII 字母映射成字符，其他码映射为 `keycode_<code>`；例如 rare raw/canonical 8230
保持 `keycode_8230` 且不计 letter。Deep PAD 使用有界 embedding：gap `-1→index0`，
与生成器共享 `16384` 个 canonical token；非负 `0..16383 → code+1`，所以 8230 保留为
独立的 index8231，超出词表则 fail closed，不截断、不取模、也不进入 overflow bucket。
real extractor 与 generated archive 在进入这两个路径前使用同一个 canonical helper；bundle
manifest 分别记录 real/fake 的 max、8230 数量和越界数量，防止 codebook 漂移。

## 4. Train-only normalization

`RawSequenceNormalizer.fit` 只接受 `pool=train`：

- x/y/pressure/size 只在真实 contact slots 上估计 mean/std；
- `log1p(global_dt_ms)` 只在 train 的全局时间 token 上估计 mean/std；
- contact/gap/action code/keycode/event ID 都是结构或类别字段，不参与连续标准化；
- `active_mask` 只随 record 保存供上游审计，不参与标准化，也不进入 frame encoder，避免
  real label callback/phase annotation 与 generated contact 构造差异成为 label oracle；
- no-contact/padding 连续值在 transform 后强制为 0。

normalizer state 和参与 fit 的 train sample IDs 保存进 best/last checkpoint，便于审计。
validation/test 只能 `transform`，不能重新 fit。

## 5. 两类真实 Deep PAD

每个 action 单独训练同一架构配置，不跨 action 混训。

### 5.1 Raw TCN

`RawTCNPAD` 使用：

1. 两 pointer 物理/结构/key embedding frame encoder；
2. 多层 dilation=`1,2,4,...` 的 temporal Conv1d residual blocks；
3. 每层用 `frame_mask` 清零 padding；
4. deterministic masked mean + masked RMS event pooling；
5. binary classifier 输出 fake logit。

这是对 raw temporal sequence 的卷积网络，不是 24/49/34 维 feature 上的 MLP。

### 5.2 Raw Transformer

`RawTransformerPAD` 使用相同 raw frame encoder，随后用带
`src_key_padding_mask` 的 multi-head TransformerEncoder，再做 masked mean/RMS pooling
和 binary classifier。padding token 不参加 attention 或 pooling。

两类模型均显式输入 pinch 两指结构和 keystroke contact/gap/keycode，不把 pinch 降成
中心线，也不把 typing 降成统计向量。

frame encoder 同时输入物理 `time_progress` 和按保留 token 顺序计算的
`sequence_progress`。因此 zero-flight 边界的两个 token 即使物理时间相同，TCN 和
Transformer 仍保留各自的序列位置；collator 不合并、不排序、不添加 epsilon。

## 6. 训练、best 与 last

训练 loss 为 train 上的 class-balanced BCE-with-logits；`pos_weight` 只由 train 类别数
计算。每个 epoch：

1. 只在 train 更新模型；
2. 在 validation 计算 loss/AUC；
3. 以 validation AUC 最大为主、validation loss 最小为 tie-break 选 best；
4. test 完全不参与 checkpoint 选择。

checkpoint：

```text
checkpoints/best_epoch_XXXX.pt  # 每次改进写不可变新文件，不覆盖旧 best
checkpoints/last.pt             # 每 epoch 原子替换，包含 optimizer，供断电恢复
```

shuffle、dropout 与 DataLoader RNG 不是依赖进程内游标连续推进，而是每个 epoch 用
`SHA256(base_seed|deep_pad_epoch|epoch)` 重新定位。`last.pt` 只在 epoch 边界提交，所以
同一 epoch 无论由连续运行还是断电恢复到达，batch 顺序与随机掩码完全一致；测试逐 tensor
验证 uninterrupted/resumed final state、history 和 test scores bitwise 相同。

`last.pt` 保存模型、optimizer、normalizer、history、当前 best 路径与 early-stopping 状态。
当前 checkpoint schema v2 还把完整 pair run identity 同时写入每个 immutable best 与
`last.pt`：dataset 绝对路径/SHA-256、fixed fake split 路径/SHA-256、`real_hash_seed`、
action/family/detector、完整 pair config 及其 SHA-256。恢复时除 action、detector、model/config
外，还逐字段和 canonical digest 比较该 identity；任一 source/config 漂移都拒绝恢复。
审计同时复算 validation best 规则，核对 best/last epoch、history、optimizer/model finite、
normalizer 及其 train sample IDs。

TCN 每个 residual block 在 conv1 激活之后、进入 conv2 之前立即按 `frame_mask` 清零 padding；
否则 conv1 在尾部 padding 产生的非零响应会被 conv2 卷回有效边界，使同一事件的 score 依赖
batch 中最长事件的 T。TCN 和 Transformer 都有“单独推理 vs 与更长事件同 batch”不变性回归。

## 7. Validation 阈值与 test

加载 validation 选出的 best checkpoint 后：

- 在 validation 选择 EER threshold；
- 在 validation 选择满足 `FRR<=5%` 且 validation FA 最小的 threshold；
- 两个 threshold 冻结后应用到 test；
- test 报固定阈值 FA、FRR、AUC；
- test 上扫完整 threshold 只用于画 FA-FRR 曲线，禁止从曲线反选工作点。

判定边界始终是：

```text
FRR = mean(real_score >= threshold)
FA  = mean(fake_score < threshold)
```

等于 threshold 的样本被拒绝。

## 8. User-level bootstrap

test 置信区间按 user，而不是独立 window 重采样：

1. real test users 有放回抽样；
2. fake test users 独立有放回抽样；
3. 某 user 被抽中一次，就保留该 label 下该 user 的全部 test windows；
4. validation threshold 固定不变；
5. 每次重新计算固定阈值 FA/FRR 与 AUC；
6. 2.5/97.5 percentile 形成 95% CI。

这正确处理“同一 user 内多个 window 相关”的事实，也适配 real test 100 users 与 fake
test 20 users 的不对称边界。

## 9. 统一 feature + deep runner

每个 action 都运行：

| family | detectors |
| --- | --- |
| clean-room Feature PAD | `linear_svm`, `rbf_svm`, `xgboost` |
| Raw Deep PAD | `tcn`, `transformer` |

Feature PAD 正式输入固定为 bundle v2：
`feature_schema_version=trajectory_features_v2_ahb_table6_hmog_real_up`。其中 AHB
Table 6 acceleration 为 signed `Δspeed/Δt`，`dev20/50/80` 为 signed perpendicular
deviation，历史列名 `acc_first5pct_median` 实际取前 5 个 acceleration samples。HMOG
使用真实 UP 时间/坐标，不追加或删除 AHB 容器的 dummy vanishing point。旧 bundle v1
在 loader 处 fail closed。

正式执行采用 25 个独立、可恢复 pair，之后严格合并。pair 目录包含：

```text
pairs/<action>/<family>/<detector>/pair_manifest.json
pairs/<action>/<family>/<detector>/result/summary.json, score_dump.npz, curves.npz, bootstrap_*
pairs/<action>/deep_pad/<detector>/result/history.csv, checkpoints/best_epoch_*.pt, last.pt
pairs/<action>/<family>/<detector>/test_fa_frr.png
merged/plots/<action>/<detector>.png
per_action_detector.csv
macro_by_detector.csv
macro_by_detector.md
summaries/by_action/<action>.csv, <action>.md
summaries/by_detector/<family>__<detector>.csv, .md
merged/benchmark_report.md
merged/benchmark_manifest.json
```

Feature pair 只有存在且通过 source/config/artifact 审计的 current `pair_manifest.json` 才可
跳过。若 summary/scores 存在但 pair manifest 缺失，旧 result 整体移入
`orphaned_unbound_feature_results/`，随后从当前 feature bundle 重新训练；不得把旧 scores
重新贴上当前 dataset hash。Deep pair
通常从配置完全一致的 `last.pt` 恢复；若断电恰好发生在 immutable best 原子落盘之后、
`last.pt` 之前，则只接受唯一的下一 epoch orphan best，按 epoch-addressed RNG 重放并逐
tensor/array/scalar 精确验证后复用，绝不覆盖；任意漂移 fail closed。

pair schema v2 的独立审计不再只检查 score 自洽：它重新读取 dataset 与 fixed split、用固定
`real_hash_seed` 分池，并把 Feature 的每个 `row_index` 或 Deep 的每个 `sample_id` 连同
label/user/pool/action 按顺序逐行重连；要求 val/test 行数精确、identity 唯一且互不重叠。
随后从 validation scores 重选阈值、重建 val/test 曲线，并用 pair seed 的固定 domain
（Feature `seed+31`，Deep `seed+17`）从头精确重算全部 500 个 user-level bootstrap arrays
和 summary，而不只是核对 CI 分位数。merge 会再次执行这套审计；任一删行、字段篡改、
bootstrap 篡改、source hash/config drift 都拒绝写 final manifest。

每个 pair 的 RNG seed 由
`SHA256(base_seed|action|family|detector)` 固定导出，25 个 identity 的 seed 必须唯一；因此
并发启动/结束顺序不影响初始化、shuffle、bootstrap 或断点恢复。formal pair 和 merge
固定且只接受：`epochs=40`、`patience=0`、`user bootstrap=500`。因此所有 Deep pair 都跑
完整 40 epochs；validation 仍只用于选择 immutable best，但不触发提前停止。39/41 epochs、
499/501 bootstrap 或任何正 patience 都 fail closed，不能以“更长/更多”绕开同协议比较。

### 9.1 未截断最长事件显存 probe

Deep pair 正式启动前必须对该 action bundle 的全部 train records 找到最长 real 与最长
fake event，并用二者较长者重复成 batch，执行一次完整 forward/backward/optimizer step。
probe 在 `1..requested_batch` 内二分寻找当前 GPU/模型的最大安全 batch，只允许降低 batch；
不删除、不截断、不重采样任何事件。JSON 固定 dataset/split/model/device/pair seed、最长 `T`、
selected batch、峰值显存与 OOM 尝试。Deep pair 会重算 source hash 和最长 `T`，并要求
`selected_batch_size == train batch_size`；缺 probe 或 probe 漂移即停止。

五动作、五 detector、两个工作点应产生 `5*5*2=50` 条 per-action summary rows；每个
detector/工作点跨五动作做算术 macro average，共 10 条 macro rows。macro 不替代每个
action/detector 的完整结果。最终 `benchmark_report.md` 还必须逐项列出 10 个 Deep
`action/deep_pad/detector -> selected_batch_size`，不能只显示跨 pair 的最大 batch size；
manifest 中的 `batch_size_by_identity` 保留全部 25 个 pair 的配置值。

## 10. Bundle 格式与正式运行

每个 action 一个 numeric-only、`allow_pickle=False` 文件：

```text
<dataset_dir>/tap.npz
<dataset_dir>/scroll.npz
<dataset_dir>/swipe.npz
<dataset_dir>/pinch.npz
<dataset_dir>/keystroke.npz
```

bundle 使用 `sequence_offsets` 和 `flat_*` 保存变长 raw sequences，同时保存同序的
`feature_vectors`。`save_raw_sequence_bundle` / `load_raw_sequence_bundle` 会完整 round-trip
验证 timestamp、双指 mask、action code、gap、keycode、event group 和 feature 行数。

real 使用 extractor v1.1 numeric flat NPZ；fake 直接使用 generator 的 500 个
`shards/shard_*_of_*/<action>/user_*.npz` archive。builder 会重建 Type-B pinch 为
MotionEvent snapshot（union 原时间、pointer lifetime 内 forward-fill），并强制每 action
100 users×200、fake split 14k/2k/4k、100,000 fake ID 全局唯一：

```bash
python scripts/build_trajectory_pad_bundle.py \
  --real-dir /ABS/PATH/TO/real_flat_npz \
  --fake-archive-dir /ABS/PATH/TO/generation_archive_root \
  --fake-user-split /home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json \
  --reference-registry-map /ABS/PATH/TO/reference_registry_map.json \
  --output-dir /ABS/PATH/TO/formal_bundle
```

adapter 按 `flat_frame_index` 恢复 event-global frame；HMOG 同一 frame 偶发的重复 pointer
行用确定性 last-row-wins 合并，不新增时间 token。Android 的 actionMasked 是整个
MotionEvent 的全局值，原始多指行会重复同一个 code；adapter 依据 pointer 在 global
timeline 的出现/消失推导 pointer-local DOWN/POINTER_DOWN/UP/POINTER_UP，避免把 pinch
两指错误标成同时落下或抬起。

正式训练以 one-pair 命令为最小调度单元。三类 Feature 可按 action 做受控 CPU 并发；
TCN/Transformer 分别在 `cuda:0`/`cuda:1` 排队。下面模板对 5 actions×5 detectors 各执行
一次，参数、完整数据、validation threshold 和 bootstrap 都不改变：

```bash
cd /home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713
python scripts/run_trajectory_pair.py \
  --dataset-dir /ABS/PATH/TO/formal_bundle \
  --fake-user-split /home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json \
  --output-root /ABS/PATH/TO/new_result \
  --action tap \
  --family feature_pad \
  --detector linear_svm \
  --epochs 40 \
  --patience 0 \
  --batch-size 64 \
  --bootstrap-replicates 500

# 每个 Deep action/detector 先在将要使用的同一 GPU 做最长事件 one-step probe
python scripts/probe_deep_batch_size.py \
  --dataset-dir /ABS/PATH/TO/formal_bundle \
  --fake-user-split /home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json \
  --action tap \
  --detector tcn \
  --device cuda:0 \
  --requested-batch-size 64 \
  --output /ABS/PATH/TO/new_result/probes/tap__tcn.json

# 将上一步 JSON 的 selected_batch_size 作为 --batch-size
python scripts/run_trajectory_pair.py \
  --dataset-dir /ABS/PATH/TO/formal_bundle \
  --fake-user-split /home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json \
  --output-root /ABS/PATH/TO/new_result \
  --action tap \
  --family deep_pad \
  --detector tcn \
  --epochs 40 \
  --patience 0 \
  --batch-size 64 \
  --bootstrap-replicates 500 \
  --batch-probe-json /ABS/PATH/TO/new_result/probes/tap__tcn.json \
  --device cuda:0
```

全部 25 pair 完成后只运行严格 merge：

```bash
python scripts/merge_trajectory_pairs.py \
  --experiment-root /ABS/PATH/TO/new_result
```

merge 成功后必须再运行独立、只读 pair-tree audit。它不会重跑或改写训练，而是重新打开
25 个 pair 的 score/bootstrap/checkpoint 证据，重算 validation-only 阈值对应指标，核对
50 rows、10 macro rows、全部输出 hash 和 25 张 copied plots；只有通过后才原子写 audit receipt：

```bash
python scripts/audit_trajectory_pair_merge.py \
  --experiment-root /ABS/PATH/TO/new_result
```

正式完成门禁同时要求：

```text
new_result/merged/benchmark_manifest.json
new_result/merged/benchmark_audit.json   # status=passed
```

没有 `--fake-user-split` 的正式运行会停止，不能信任 bundle 内可能错误的 pool 字段。
builder 和 runner 还会逐 action 强制检查 `real source=100 users`、
`real train/val/test event pools=100/100/100 users`、
`fake source=100 users`、`fake pools=70/10/20 users`。历史 v1 raw keystroke 仅覆盖
20 users，不能使用；当前 `results/trajectories_full_v2` 的 v1.1 extraction 已完成并通过
独立 100-user audit（五动作合计 256,811 events，`formal_passed=true`）。正式 detector
仍须等待完整 100k neural fake archive 和 per-Deep-pair batch probe；证据与门禁要求见
`docs/deep_benchmark_data_gate_20260713.md`。

## 11. Synthetic smoke（不是正式结果）

此外还有五动作真实 corpus 的 adapter/bundle v2 小型门禁：

```text
results/real_corpus_bundle_v2_smoke_20260713/summary.json
results/real_corpus_bundle_v2_smoke_20260713/summary.md
```

它实际读取五类共 256,811 个 authoritative source events，并逐类验证 adapter、numeric
bundle v2 round-trip、Feature validation-only 流程和 TCN/Transformer finite forward。
label-1 仅为 real row 镜像以满足二分类 API，因此这项 smoke 只证明管线，绝不是生成器
FA/AUC 结果。

```bash
python scripts/run_trajectory_benchmark.py \
  --synthetic-smoke \
  --output-dir results/smoke_deep_benchmark_20260713_v3
```

已完成的 smoke 覆盖五个 action、三种 feature detector 和两种 deep detector：

```text
per-action rows = 50
macro rows      = 10
plots           = 25
deep best       = 10
deep last       = 10
```

该数据由代码人工构造，只证明 pipeline、边界和落盘产物完整，绝不能作为正式 FA/AUC
结论。正式 full 训练必须等待 generator fake trajectories 和五动作 formal bundle 完成。

smoke 结果的独立审计：

```bash
python scripts/audit_trajectory_benchmark.py \
  --result-dir results/smoke_deep_benchmark_20260713_v3
```

审计逐项从 score dump 重算 validation/test 固定阈值 FA/FRR/AUC 和完整 val/test 曲线，
并确认 25 个 action-detector pair、50 条 operating rows、10 条 macro rows、25 张 test
曲线、10 组 Deep best/last 及全部 bootstrap 产物；当前 `passed=true`。

全测试：

```bash
python -m unittest discover -s tests -v
```

最终共享代码稳定后连续执行三轮完整 discovery，结果分别为：`84 tests / 11.367s / OK`、
`84 tests / 11.220s / OK`、`84 tests / 11.410s / OK`，无 failure、error 或 skip。仅出现
PyTorch Transformer nested-tensor、`torch.load(weights_only=False)` 和旧 AMP API 的
非失败 warning。其中 `tests/test_deep_pad.py` 覆盖：pinch 全局
双指错位 DOWN/UP、keystroke gap/keycode、train-only normalization、real/fake split、
100 real users 的完整 event-group 60/20/20、numeric bundle round-trip，以及
feature+TCN+Transformer 的 runner/checkpoint/曲线、active annotation 反例、finite/multi-seed
exact replay；`tests/test_pair_runner.py` 覆盖 complete skip 与 Deep `last.pt` 恢复，
`tests/test_pair_merge.py` 覆盖 25 pair/50 rows/25 图 gallery，并验证 formal merge 只接受
`epochs=40, patience=0, bootstrap=500`；真实 adapter 测试还覆盖 first-appearance
pointer slot、canonical negative keycode、positive-flight gap、zero-flight 同毫秒双 contact
以及 real/fake adapter 的零飞行语义一致性。
