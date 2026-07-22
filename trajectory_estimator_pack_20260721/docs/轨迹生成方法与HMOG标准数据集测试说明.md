# 轨迹生成方法与 HMOG 标准数据集测试说明

版本：2026-07-22 19:47 EDT（正式实验进行中、逐步审计版）

## 1. 文档目的与结论边界

本文档回答三个问题：

1. HMOG 原始触摸数据如何转换为 tap、scroll、swipe、pinch、keystroke 五类完整轨迹；
2. 如何基于同一用户、同一动作的固定 5 条真实 reference，用神经 diffusion 生成不同时长的新轨迹；
3. 如何在 HMOG 标准数据集上进行轨迹检测和 IMU+轨迹综合检测，以及哪些划分是用户独立、哪些目前不是。

当前可以确认：数据提取、数据协议、five-shot 条件、变长 diffusion、100,000 条条件预检、25 个轨迹检测器图和 5 个综合检测器协议已完成并通过代码/数据门禁。正式训练、100,000 条生成和最终检测指标仍在进行，本文不把 smoke 或部分训练误写为最终实验结果。

## 2. 标准数据集与正式语料

### 2.1 HMOG 来源与范围

本项目的正式外部数据集为 HMOG，使用 100 个匿名用户的原始触摸事件。本地源归档为 `hmog_dataset.zip`，大小 6,132,356,276 bytes，SHA-256 为：

```text
4e3f4216ca7c362bd06493301d7ef9634940af69f939fe02689cb3f84c914346
```

原始 `TouchEvent.csv` 包含 SystemTime、EventTime、ActivityID、PointerCount、PointerID、ActionID、X、Y、Pressure、ContactSize 和 Orientation。处理保留原始不规则毫秒时间轴，不把触摸轨迹重采样为 IMU 的 100 Hz，也不使用 IMU 的固定窗口 padding/裁剪规则。

### 2.2 五类动作的完整事件定义

| 动作 | 完整轨迹语义 |
|---|---|
| tap | 纯单指 DOWN→MOVE*→UP，保留轻微位移、压力和接触面积 |
| scroll | 完整单指 DOWN→UP，同时保留派生 scroll active 区间和完整物理 contact 区间 |
| swipe | HMOG StrokeEvent/fling-style 标签对应的完整单指 contact |
| pinch | 恰好两个 pointer，保留第二指晚 DOWN、第一指早 UP 的错峰生命周期和单指 lead-in/lead-out |
| keystroke | 一系列独立按键 contact；每键保留 DOWN/MOVE/UP、keycode、hold 和 flight，键间离屏阶段不伪造 XY 直线 |

一个原始 contact 可能同时覆盖多个派生 callback 标签。为避免跨类重复和数据泄漏，正式提取使用 `keystroke > pinch > swipe > scroll > tap` 优先级，同一物理 contact 只能保留一次。缺 DOWN、缺 UP、CANCEL、单指动作中出现第二指、pinch 非双指或 keystroke 缺任一 key contact 都会 fail closed，拒绝原因保存在 audit，不静默丢弃。

### 2.3 数据量与用户划分

| 动作 | events | flat rows | keys | train/val/test events | rows min/mean/max |
|---|---:|---:|---:|---:|---:|
| tap | 19,269 | 116,263 | 0 | 14,840 / 1,636 / 2,793 | 2 / 6.03 / 31 |
| scroll | 59,937 | 2,528,173 | 0 | 39,094 / 6,407 / 14,436 | 3 / 42.18 / 526 |
| swipe | 70,431 | 2,212,589 | 0 | 47,852 / 6,718 / 15,861 | 3 / 31.41 / 512 |
| pinch | 58,016 | 4,857,213 | 0 | 40,699 / 6,024 / 11,293 | 9 / 83.72 / 1,316 |
| keystroke | 49,158 | 1,942,826 | 707,088 | 33,659 / 5,406 / 10,093 | 4 / 39.52 / 1,244 |
| 总计 | 256,811 | 11,657,064 | 707,088 | — | — |

生成模型与 fake 轨迹使用固定用户划分 70 train / 10 validation / 20 test，seed=42，split SHA-256 为：

```text
82f2277374be47d5ec9dada2f7e60d0d5afd7ba79ac8a08b67e1607294ff530b
```

三个用户集互斥，并集严格为 0..99。该结论适用于生成模型训练、reference registry、condition prior 和 fake 轨迹划分。PAD builder 对 real 轨迹另采用每个用户内部的 60/20/20 event-group hash 划分，因此当前 PAD 主协议不能笼统写成完全 user-disjoint；详细差异和补充实验要求见第 13.2 节。

### 2.4 数值存储与数据门禁

每个动作使用一个 `hmog_trajectory_<action>.npz`，采用 numeric-only flat+offset 结构：

- `event_offsets`：event 到 flat touch rows；
- `event_key_offsets`：typing event 到 keys；
- `key_touch_offsets`：key 到自己的 contact rows；
- `flat_t_rel_ms/X/Y/pressure/size/pointer_id/action_code`：原始触摸序列；
- event-level duration、orientation、pointer lifetime、用户与溯源字段；
- keystroke-level keycode、hold、flight、letter flag 和每键 contact offset。

所有 NPZ 均要求 `np.load(..., allow_pickle=False)`，object array=0，offset 从 0 开始、单调且末项与数组长度精确一致。正式 corpus audit 结果为 `passed=true`、`formal_no_sample_cap=true`，全部事件被检查，不是抽样审计。

## 3. Five-shot 变长轨迹生成方法

### 3.1 固定五条 reference

对每个 `(action, user, split)`，`ReferenceRegistry` 由 seed、corpus SHA、split 和 user 确定恰好 5 条唯一真实轨迹。五条 reference 必须与 target 同用户、同动作、同 split，不能重复、不能跨用户/跨 split，target 不能出现在 refs 中。同一用户/动作的所有生成请求始终使用同一组 5 refs，符合固定 enrollment 而不是每条重抽参考。

正式启动审计已证明五动作的 train/val/test registry 均为 5 refs，cross-user、cross-split、duplicate、target-in-ref 均为 0，test 不用于训练或 checkpoint 选择。

### 3.2 模型输入与变长表示

五个动作各训练一个独立 `TrajectoryDiffusion`。每根指针在自身 start→end chord 的局部坐标系中表示为：

```text
progress, lateral, log_dt, pressure, size
```

模型同时接收 point/contact/pointer mask、duration、orientation、XY 端点、pointer 起止时间、pinch span/angle、n_keys/n_letters 和有序 keycode sequence。五条 refs 由 permutation-invariant DeepSets encoder 编码，改变 ref 顺序不改变条件，改变 ref 集合会改变生成。

变长 batch 只 pad 到当前 batch 最长序列，`max_points=None`、`drop_last=False`，每个 epoch 覆盖每个 eligible target 恰好一次。长度 bucket 只减少 padding，不裁剪长事件。Keystroke gap 只监督 `log_dt`，不给离屏区间伪造 XY/pressure/size。

### 3.3 Diffusion 训练与采样

正式训练为 epsilon-objective DDPM，每动作 100 epochs，1000-step 线性 beta schedule `1e-4 → 2e-2`，末端 `alpha_bar≈4.04e-5`，EMA=0.999，learning rate=2e-4，weight decay=1e-4，global grad clip=1.0，CUDA AMP 开启。Validation 只在 20/40/60/80/100 epochs 使用全部 val targets 和 EMA 权重运行，best 只由 val masked epsilon MSE 选择，test 不参与。

正式生成从全新 Gaussian noise 开始，在 1000-step 训练时间表上执行 50 次 DDIM 去噪，`eta=0`。这不是 template 检索、真实轨迹变形或从五条 refs 中选一条复制。生成后只做与 detector 无关的物理约束：严格递增毫秒时间、端点一致、指针生命周期、pressure/size 范围、Android DOWN/MOVE/UP 和 pinch 双指 lifecycle。不使用 detector score 挑样本。

### 3.4 不同时长与动作条件

条件不是直接复制某一条 ref，而是由五条 ref 的 robust 统计、两 ref 插值偏离和仅基于 70 个 train users 的 shrinkage prior 组合 duration、起终点、点率、位移、pinch 几何、pointer lifetime 和 keystroke 按键结构。Orientation 在 XY 之前确定，坐标先按方向分组，避免 portrait/landscape 混合后靠 clipping 修复。

100,000 条正式条件预检已完整通过，每动作 20,000 条，覆盖的条件时长为：

| 动作 | duration range | 其他范围 |
|---|---:|---:|
| tap | 14–354 ms | 2–20 points |
| scroll | 130–6,255 ms | 5–268 points |
| swipe | 59–6,260 ms | 2–330 points |
| pinch | 151–2,892 ms | 5–161 timeline points，双指错峰 |
| keystroke | 438–41,590 ms | 2–76 keys，5–559 timeline points |

预检中 100,000 个 fake ID、condition seed 和 DDIM noise seed 各自全局唯一，两类 seed domain 不相交，no retry/no skip，condition-set SHA-256 为：

```text
dc236808d88e0c22fa631be40ee86b9c7ffd425aaad5271e6d7fe39efe16ea9f
```

这证明正式条件空间已经生成和审计，但不等于 100,000 条 DDIM 轨迹已经生成。

### 3.5 正式产物

总计生成 100 用户 × 5 动作 × 200 条 = 100,000 条，分布为 train 70,000、validation 10,000、test 20,000。每个 action-user unit 是纯数值 NPZ，保存轨迹 timeline、Android contact rows、5 reference event IDs、duration/orientation/pointer/key 条件、checkpoint/registry/prior/split hash、clipping 计数与双 seed。两张 GPU 按 shard 并行，每个 unit 先写 staging，审计通过后原子发布，已存在单元只有在全部身份和 hash 通过时才能 resume。

## 4. HMOG 标准数据集测试设计

### 4.1 轨迹检测器

每个动作训练 5 个 real-vs-fake 检测器，总计 25 个：

| 类型 | 模型 | 输入 |
|---|---|---|
| Feature PAD | linear SVM | 冻结轨迹特征 |
| Feature PAD | RBF SVM | 冻结轨迹特征 |
| Feature PAD | XGBoost | 冻结轨迹特征 |
| Deep PAD | TCN | 变长原始轨迹序列与 mask |
| Deep PAD | Transformer | 变长原始轨迹序列与 mask |

单指动作使用 24 维 AHB-style 特征，pinch 使用 49 维双指扩展，keystroke 使用 34 维离散序列扩展。特征包含 duration、路径长、直线距离、速度/加速度、垂直偏差、压力/接触面积、pinch center/span/angle/lifetime，keystroke hold/flight/keycode/contact 结构等。

### 4.2 严格训练、阈值与测试边界

```text
train: fit scaler + fit detector + fit duration bins
validation: 选 EER threshold 和 FRR<=5% threshold
test: 只应用已冻结模型/阈值/时长分箱
```

分数统一为越大越像 fake，`score < threshold` 接受为 real，`score >= threshold` 拒绝为 fake。主指标包含 ROC-AUC、FA（fake 被接受）、FRR（real 被拒绝）、validation EER operating point、validation FRR<=5% operating point，以及 test-user-level 500 次 bootstrap 的均值、中位数和 95% CI。Bootstrap 按用户抽样并保留该用户全部样本，不把同一用户的窗口误当独立样本。

### 4.3 不同时长指标

每个轨迹检测器和综合检测器都输出 4 个 duration bins 的 val/test 指标。分箱边界只由 train duration 拟合，validation/test 不参与；各时长箱应用同一个 validation 全局阈值，不允许在每个箱内重调 threshold。这样才能测量短/中/长轨迹上的真实性能变化。

### 4.4 IMU+轨迹综合检测器

综合检测不只是把两个分数相加。每条配对样本包含：

- 独立 IMU detector score；
- 独立 trajectory detector score；
- 轨迹动力学与 IMU 之间的 10 维一致性 components；
- action、duration、user/split 和 EventPlan 身份。

真实 HMOG 已构建 231,728 个严格配对：tap 18,293、scroll 54,529、swipe 64,421、pinch 51,950、keystroke 42,535。配对使用 user+activity+绝对 start/end 时间的一对一身份，不依赖不可靠的 event_id 或行号。待 fake 轨迹完成后，生成对应 fake IMU 和一致性分量，再对五个动作各训练一个综合检测器。

## 5. 已通过的标准数据集证据

| 证据 | 状态 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| HMOG 100-user 五动作完整提取审计 | 通过 | 数量、时间、pointer、key、schema和五类互斥合法 | 模型生成质量 |
| v16 代码测试 | 172/172 通过、0 skipped | 模型、约束、泄漏门、resume、审计和 detector 协议可执行 | 正式 100k 结果 |
| 100k condition preflight | 通过 | 恰好 100k、5 refs、时长范围、种子和无 skip/retry | DDIM 轨迹已全部生成 |
| tap/scroll/swipe 100-epoch | 完成 | 三个 action 的正式 best-EMA checkpoint 已就绪 | pinch/keystroke 或全任务已完成 |
| pinch/keystroke | 运行中 | 心跳、全量 batch 和固定 5 refs 审计正常 | 最终 checkpoint/100k/检测指标 |
| 25 个轨迹检测器 | 协议完成，正式结果待运行 | 数据流、阈值和指标定义已冻结 | 正式 AUC/FA/FRR |
| 5 个综合检测器 | 真实配对完成，正式 fake 结果待运行 | 真实侧身份和 component 基础可审计 | 正式综合检测性能 |

## 6. 当前正式训练结果

截至 2026-07-22 19:17 EDT：

| 动作 | 状态 | 最新/最终 train loss | best EMA val loss |
|---|---|---:|---:|
| tap | 100/100 完成 | 0.0318582 | 0.0252213（epoch100） |
| scroll | 100/100 完成 | 0.0201320 | 0.0164595（epoch100） |
| swipe | 100/100 完成 | 0.0191806 | 0.0155675（epoch100） |
| pinch | epoch31 已提交，epoch32 训练中 | 0.0214871（epoch31；比epoch30下降0.5868%） | 0.0346967（epoch20） |
| keystroke | epoch3 已提交，epoch4 训练中 | 0.1091779（epoch3；比epoch2下降14.4791%） | 待 epoch20 validation |

三个已完成动作的 best 均明确为 validation-only 选择、EMA 推理权重，test 未参与选择。最终 loss 中的单轮回升、AMP 重试和 GPU 瞬时异常均保存在状态问题清单，没有只报有利数值。

Keystroke epoch1 完整消费 33,309 个 train targets / 521 batches，`full_train_split_consumed=true`，valid feature count 6,879,506，retry0，global step521。`last.pt` 为 29,738,178 bytes，SHA-256 `11658ea0a22c6b1705c1d3f9a09de74766e4c489f2a679d87b9d917d547fdeb1`，与 `last_state.json` 匹配，无 pending/tmp/epoch-commit 残留；复核时 worker 已进入 epoch2 step554，因此未为文档编写暂停训练。

Keystroke epoch2 同样完整消费 33,309 个 targets / 521 batches，valid feature count 6,879,506，retry0，loss `0.1276621683`，比epoch1改善38.3129%。边界checkpoint SHA-256为 `323c99b9f4817fe9efb57a8900b05918c6c211542e5469e50819f7ed17d02bda`，与state匹配且无事务残留；复核时worker已进入epoch3。这是train-loss证据，不替代epoch20 validation或最终PAD指标。

Keystroke epoch3 继续完整消费 33,309 个 targets / 521 batches，valid feature count 6,879,506，retry0，loss `0.1091778939`，比epoch2改善14.4791%。`last.pt`大小29,738,178 bytes，SHA-256 `0d547d0497a270279922f47def238649c45454dc06a01def73c94b85a211105a`，与`last_state.json`匹配；state边界为epoch index3/global step1563/next batch0，且无pending/tmp/epoch-commit残留。120秒健康检查显示worker已继续epoch4，两张GPU均有正常利用率、0个active-action error。

Pinch epoch29 完整消费 40,349 个 targets / 158 batches，retry0，loss `0.0230405671`，比epoch28回升3.032576%，列为 P-TRJ-164 观察项。Epoch30随后同样全量消费，loss下降6.191905%至`0.0216139171`，故回升未延续。Epoch30 checkpoint SHA-256为 `ec58c5100fd16bf417446417bd5c7beffa74fccc3dc6f47be66e2bc4d37f92f5`，与state匹配且无事务残留；复核时已进入epoch31。

Pinch epoch31再次完整消费40,349个targets / 158 batches，valid feature count 16,732,955，retry0，loss继续下降0.5868%至`0.0214870780`。边界global step为4,898，`last.pt`大小29,738,370 bytes，SHA-256 `ffedc21ed5bc992558c9ccf2945d3bb26fd98c54813821ed9aae23ef780c33b7`与state一致，无pending/tmp/epoch-commit残留；worker随后进入epoch32。

## 7. 尚未完成、已知限制与时间

### 7.1 尚未完成

1. pinch 与 keystroke 完成 100 epochs 及最终 checkpoint 审计；
2. 100,000 条真实 DDIM 轨迹的生成、物理门和完整 hash 审计；
3. 25 个轨迹检测器的 val/test 全局与时长分层指标；
4. fake IMU、一致性 components、5 个综合检测器；
5. 最终独立审计、图表和本文的正式结果版。

### 7.2 限制和不能夸大的结论

- 目前的正式外部数据集是 HMOG；生成模型的100用户user-disjoint split用于测试同数据集未见用户生成，不能外推为跨设备、跨数据集泛化。当前PAD主协议也不是real/fake双侧完全user-disjoint，须按第13.2节限定表述。如要声称cross-dataset，必须再接入第二个有完整touch+IMU身份的公开数据集并做独立外测。
- 原始 HMOG zip hash 已补充固定；但提取运行开始时的 extractor bytes 未复制到输出目录。启动前记录的 source hash 为 `243767dc028049f01de8d312744c8fb01cc81330ba838660b5693f896e0fe391`，运行期间发生了仅文档编辑，详细限制已保存在 supplemental provenance，不声称比现有证据更强的 source binding。
- 20-step e2e smoke 的接口通过，但严格正式 clipping 门失败，它只证明管线可运行，不能用作 100-epoch 生成质量。正式 100k 必须通过 aggregate clipping<=5% 且 event max<=25%，否则 pipeline fail closed。

### 7.3 预计时间

以 2026-07-22 18:27 EDT 的实际吞吐估算：

- 剩余训练约 37–43 小时，keystroke 为关键路径；
- 100k 生成与审计约 1–4 小时；
- 25 个轨迹检测器约 6–12 小时；
- 综合检测、最终审计和文档约 3–8 小时。

无返工中心估计约 55 小时，预计 2026-07-24 晚至 2026-07-25 交付最终结果版。如最终物理门、数值门或配对身份审计失败，需如实返工并延长，不会跳过门禁。

## 8. 证据与更新位置

- 唯一当前状态口径：`docs/IMU与轨迹交付状态及问题清单.md`
- 数据结构：`trajectory_humanization_full_20260722_v16_numeric_recovery/docs/轨迹数据结构与预处理说明.md`
- 训练协议：`trajectory_humanization_full_20260722_v16_numeric_recovery/docs/training_protocol.md`
- 生成协议：`trajectory_humanization_full_20260722_v16_numeric_recovery/docs/generation_protocol.md`
- 检测协议：`trajectory_humanization_full_20260722_v16_numeric_recovery/docs/detector_protocol.md`
- 共享 EventPlan 和综合检测：`docs/共享EventPlan、Trajectory生成与TotalDetector.md`
- HMOG 完整语料审计：`trajectory_humanization_full_20260713/results/trajectories_full_v2/formal_audit/formal_data_audit.md`
- 正式 run：`results/formal_eventplan_v16_numeric_recovery_100epoch_100k_20260722`
- 训练后阶段就绪性复核：`results/formal_eventplan_v16_numeric_recovery_100epoch_100k_20260722/audits/post_training_readiness_review_20260722.md`
- GitHub源码镜像：`https://github.com/WMSlalalala/imu_gen`（不包含数据、results、cache或checkpoint）
- Agent续跑轻量缓存：`agent_handoff/latest_state.json`（每30分钟同步前刷新；只作导航，正式状态仍须核验本机manifest/SHA）

本文在后续每个正式阶段完成时更新，最终版将补入 25+5 个检测器的全局指标、4 个时长分层、bootstrap 95% CI、runtime 与所有未关闭问题。

## 9. 正式验收清单

下表是最终交付必须通过的完整清单。任一阻塞项未通过时，总任务不能标记为完成。

| 阶段 | 验收条件 | 当前状态 |
|---|---|---|
| HMOG 语料 | 5 actions、100 users、numeric-only、全事件审计、split/hash 一致 | 通过 |
| 固定 five-shot | 每 action/user/split 恰好 5 个 unique real refs，无 target/cross-user/cross-split 泄漏 | 通过 |
| 训练 | 五动作各 100 epochs，完整 train targets、5 次 full-val、best-EMA/last/hash 一致 | 3/5 actions 完成 |
| 数值恢复 | AMP overflow 必须同 batch 重试、恢复 RNG、有限 optimizer step 后才计数；超上限 fail closed | 运行中持续审计 |
| 100k 条件 | 100×5×200，每条 5 refs，condition/noise/fake ID 唯一，no skip/retry | 预检通过 |
| 100k DDIM | 恰好 500 units/100k，50-step、eta=0、best-EMA、无 selector、请求 digest 与预检一致 | 待训练完成后运行 |
| 物理质量 | 时间严格递增、端点/lifecycle/mask 合法，aggregate clipping<=5%、event clipping<=25% | 待 100k 实测 |
| 轨迹 PAD | 5 actions×5 detectors，train-only scaler/bins，val-only thresholds，test 固定阈值 | 待 100k 后运行 |
| 轨迹指标 | 每 detector 的 val/test AUC、FA、FRR、EER、FRR<=5%、4 duration bins、500× user bootstrap CI | 待运行 |
| 综合配对 | real/fake 的 trajectory、IMU、EventPlan、components 一对一身份和 hash 一致 | real 基础通过，fake 待运行 |
| Total detector | 5 actions×1，只用 train fit，val 冻结 threshold，test 及 4 duration bins 报告 | 待上游完成 |
| Runtime | 轨迹生成、25 个 trajectory detector、5 个 total detector 的冷/热路径延迟与证据 | 待正式产物 |
| 问题清单 | 所有 error/warning/监控误操作均有证据、影响和关闭标准 | 持续更新 |
| 最终独立审计 | 从当前 bytes 重算 count/schema/SHA/身份/指标，不仅信任 sidecar 状态 | 待全部阶段完成 |

## 10. 连续流水线与每小时训练检查约定

文档编写不暂停、重启或修改已冻结的训练。当前同时保留三层监督：

1. 主 trajectory supervisor 负责训练、100k 生成、25 detectors 和阶段门禁；
2. total supervisor 不超前执行，当上游尚未完成时保持 `wait_trajectory`、`jobs={}` 且不占 GPU；
3. 独立只读健康监控按用户要求每 1 小时检查 supervisor stage/heartbeat、worker PID/step、epoch transaction、loss、AMP retry、GPU 利用率/显存/温度和 warning/error。正式 supervisor 本身仍连续运行并执行 fail-closed 阶段门禁。

在此基础上，每 1 小时执行一次训练汇总审计，至少核对：

- 主/下游 supervisor 进程、stage、heartbeat 和返回状态；
- 每个 active action 的 committed epoch、global step、examples、phase、最近成功心跳和 AMP retry；
- 最新完整 epoch 是否全量消费、loss 是否有单轮或连续异常、validation 是否只用 val+EMA；
- checkpoint 事务是否有 pending/tmp 残留，last/best/state/manifest 的 size/SHA/epoch/step 是否一致；
- GPU 上是否只有预期 worker；若有外部任务，记录 PID、显存、温度、降频原因和进度影响，不擅自终止他任务；
- 问题清单和本文的状态/时间是否同步。

任一 error、worker/supervisor 退出、单 batch AMP retry 超上限、NaN/Inf、hash/身份不一致、下游超前运行或任一 fail-closed 门失败仍由正式 supervisor 连续检测并阻止阶段推进；人工健康摘要不需要按分钟重复读取。GitHub源码与Agent轻量交接缓存仍每30分钟同步一次，但它不触发训练级深审计。

2026-07-22 19:57复查时清理了一个旧v15只读健康摘要残留进程；它不是当前v16流水线的一部分。当前训练健康摘要的唯一实时口径是v16 run目录下的每小时`training_health.json`/`training_health.txt`，历史v15报告只能作为失败run归档证据，不能用于判断当前v16训练进度。

### 10.1 已完成的 30 分钟汇总快照（2026-07-22 18:50 EDT）

| 检查项 | 结果 |
|---|---|
| 主 trajectory supervisor | running，stage=`training`，config SHA `063fefe2169e1120cbc8f7b275d7a11e170abee9bb37b85214f143890e40faee` |
| 下游 total supervisor | running，stage=`wait_trajectory`，`jobs={}`，未超前占用 GPU |
| pinch | epoch28训练中，step4418，38,813/40,349 examples，retry0，心跳新鲜 |
| keystroke | epoch1已提交，epoch2 step857，21,469/33,309 examples，retry0，心跳新鲜 |
| 事务残留 | pinch/keystroke 均无 pending/tmp/epoch_commit 残留 |
| GPU0 | keystroke PID2195033，约28%、2,114 MiB、70°C，throttle0 |
| GPU1 | pinch PID2159255 + 外部 IR export PID2218613，100%、12,677 MiB、85–86°C、约296W |
| GPU1 throttle | 仅 power-cap active，hardware thermal slowdown/hardware slowdown 均未活动 |
| errors | 0 |
| 已记录异常 | GPU1 外部 workflow 连续切换训练/导出阶段，分别保存为 P-TRJ-152/155/157/159/160；keystroke GPU0 两次瞬时0%保存为 P-TRJ-154/156 |

该表是18:50且当时采用120秒健康检查时的历史快照；19:55以后独立健康汇总改为每1小时，后续进展以第6节和唯一状态问题清单为准。

## 11. 从原始 HMOG 到最终检测结果：逐步实施记录

本节按实际依赖顺序记录“输入、做了什么、检查什么、产物和当前状态”。“代码通过”只说明实现和门禁可执行；“正式结果通过”必须有当前正式 bytes 的计数、SHA、指标和独立审计，二者不混写。

| 步骤 | 输入 | 实际工作 | 必查内容 | 主要产物 | 当前状态 |
|---:|---|---|---|---|---|
| 1 | `hmog_dataset.zip` | 固定原始 HMOG 归档并计算大小与 SHA-256 | 文件存在、大小 6,132,356,276 bytes、SHA 与记录一致 | 原始数据身份记录 | 通过 |
| 2 | 各 session 的 `TouchEvent.csv` | 按时间读取原始 rows，恢复 frame、pointer 和 contact 生命周期 | 时间字段可解析、pointer/action code 合法、DOWN/MOVE/UP 闭合；CANCEL、孤立帧和未闭合事件单独计数 | extractor audit | 通过，异常见第 12 节 |
| 3 | touch contacts 与 HMOG 派生标签 | 重建 tap/scroll/swipe/pinch/keystroke 五类完整事件 | 单指动作不得混入多指；pinch 必须恰好双指；keystroke 每键必须有完整 contact | 五动作候选事件 | 通过，非法候选 fail closed |
| 4 | 同一物理 contact 的多标签候选 | 按 `keystroke > pinch > swipe > scroll > tap` 做互斥归属 | 一个 raw contact 不得重复进入多个动作；冲突必须记录保留/拒绝原因 | 互斥五动作事件集 | 通过 |
| 5 | 五动作事件 | 转成 numeric-only flat+offset NPZ，不做 object/pickle 存储 | 所有 offset 单调、首项0、末项等于 flat 长度；字段长度一致；全字段 finite | `hmog_trajectory_<action>.npz` | 通过 |
| 6 | numeric NPZ、manifest、用户 split | 对全部 256,811 events 做 uncapped corpus audit | schema、文件 SHA、计数、时间、pointer、key、split 互斥和 canonical restore | `training_corpus_audit.json`、`formal_data_audit.md` | 通过 |
| 7 | 100 个用户 | 固定 seed=42 的 70/10/20 用户划分 | 三集合互斥、并集0..99、split SHA 固定 | `users_seed42.json` | 通过 |
| 8 | 每个 `(action,user,split)` | 检查 fixed-five-shot 最小样本资格 | 至少 6 条真实事件，即 5 refs + 至少1 target；不足则训练前失败 | eligible target 清单 | 通过；旧语料曾失败 |
| 9 | eligible real events | 建立 `ReferenceRegistry`，每组固定5条唯一真实 refs | 同用户/动作/split、无重复、target 不在 refs、registry hash 固定 | 五动作 `reference_registry.json` | 通过 |
| 10 | target、5 refs、动作条件 | 为五动作分别构建变长 diffusion 数据与模型 | mask 正确、无固定长度裁剪、pinch 双指共用 union clock、keystroke gap 不伪造触摸 | 训练 batch 与模型 | 172/172 代码测试通过 |
| 11 | train users 与 fixed refs | 只从 train 拟合条件 shrinkage prior，并枚举 100k ConditionRequest | 恰好100k；每动作20k；每user/action 200；fake ID 与两类 seed 全局唯一；无 retry/skip | 100k preflight receipt | 通过，尚非真实生成 |
| 12 | 冻结源、配置、语料审计 | 执行 launch gates、source/config hash 和 GPU 吞吐门 | source/config 不漂移、测试0 skipped、语料/registry/preflight 全通过 | launch gate receipts | 通过 |
| 13 | 五动作训练集 | 每动作独立训练100 epochs；20/40/60/80/100做完整 EMA validation | 每epoch全量 targets；AMP overflow 同 batch 恢复重试；loss finite；best 仅由 val 选；test 不参与 | `last.pt`、best、metrics、manifest | 3/5动作完成，2/5运行中 |
| 14 | 五个正式 best-EMA | 两 GPU 分 shard 做 50-step、`eta=0` DDIM，生成100k完整轨迹 | 500 units、请求 digest、无 selector、物理约束、原子发布、resume身份一致 | 100k generated units | 待训练完成自动运行 |
| 15 | 100k generated units | 从当前 bytes 重算 generation formal audit 和 PAD ingress audit | 恰好100k、hash map、seed replay、无真实轨迹完整复制、时间/lifecycle/clipping 合法 | formal audit 与 PAD export receipt | 待运行 |
| 16 | HMOG real + generated fake | 构建每动作 PAD bundle 和冻结特征/序列输入 | real/fake 标签、分割、event identity、reference overlap、finite 特征、mask/长度一致 | detector dataset | 待运行 |
| 17 | 五动作 PAD bundle | 每动作训练 linear SVM、RBF SVM、XGBoost、TCN、Transformer | train-only fit；validation-only threshold；test固定阈值；4时长箱和用户bootstrap | 25组模型、scores、metrics | 待运行 |
| 18 | 同事件 trajectory、IMU 与 components | 构建一对一 real/fake paired table，再按动作训练 total detector | user/activity/绝对时间/EventPlan/hash一致；任何缺配或多配失败 | 5个综合检测器及分时长结果 | real基础通过，fake侧待运行 |
| 19 | 所有正式结果 | 独立重算 count/schema/SHA/指标/图表，不只读取 supervisor 的“完成”状态 | 30个检测器结果齐全、置信区间齐全、未关闭问题逐项解释 | 最终报告与审计 receipt | 待全部上游完成 |

### 11.1 已经执行过的修正

1. 第一版 `trajectories_full` 的 uncapped 审计发现 pinch 中存在三指事件，并且大多数用户没有足够的 keystroke refs，因此该语料没有进入正式训练；重新构建并审计 `trajectories_full_v2`。
2. v15 正式训练发现 AMP 数值恢复不能只跳过溢出 batch。v16 改为同一 batch 回退、恢复 RNG、降低 scale 后重试，只有产生有限 optimizer step 才计数；tap/scroll 的迁移状态经过原子事务审计。
3. 20-step e2e smoke 的严格 clipping 门失败。该 smoke 只保留作接口证据，没有被写成生成质量通过；正式100k仍必须通过 aggregate clipping不超过5%、单event不超过25%的硬门。
4. real trajectory 与 IMU 配对不再依赖可能变化的 event ID，而使用 user、activity、绝对 start/end 时间做一对一身份匹配。
5. 文档复核纠正了“整个检测流程均 user-disjoint”的过度表述：目前生成侧是 user-disjoint，当前 PAD real 侧不是；补充要求见第13.2节。

## 12. 数据错误与不规范审计

### 12.1 原始解析层异常

下列数字来自 `trajectories_full_v2/audit.json`，均被保留而非隐去。它们的统计单位不同：rows 是 CSV 行，frames 是解析帧，contacts/gestures 是重建事件，所以不能直接相加后当成“坏样本总数”。

| 项目 | 数量 | 含义与处理 |
|---|---:|---|
| raw rows | 11,336,372 | 实际扫描的原始 touch rows |
| frames | 8,834,172 | 按时间与 pointer 状态解析出的 frames |
| complete gestures | 396,568 | 原始层闭合 gesture，仍需动作标签和互斥门 |
| cancelled gestures | 18 | 出现 CANCEL；拒绝，不拼接成完整轨迹 |
| incomplete at EOF | 1 | 文件结束仍未闭合；拒绝 |
| incomplete restarted | 393 | 新 DOWN 到来前旧 contact 未闭合；旧事件拒绝 |
| mixed-action frames | 91,889 | 同一时间附近 action code 混合；只按严格生命周期恢复，不凭行邻近强行合并 |
| orphan frames | 2,868 | 找不到合法起点/活动 contact；拒绝 |
| one-finger complete contacts | 784,622 | 原始单指完整 contact 候选，不等于最终动作事件数 |
| one-finger rows | 2,451,330 | 单指状态机实际检查的raw rows；与全部rows不是互斥类别，不能相减求坏数据 |
| one-finger incomplete at EOF | 20 | 单指 contact 未闭合；拒绝 |
| one-finger incomplete restarted | 934 | 单指 contact 被新 DOWN 打断；拒绝 |
| one-finger orphan UP | 195,077 | UP 无可配对 DOWN；不补造 DOWN，拒绝 |
| one-finger unsupported action | 686,055 | 不满足正式五动作语义的 action 记录；不映射到最近类别 |

`raw_unsupported_action_codes` 为空，表示没有未知数值 action code；`one_finger_unsupported_action` 指已知 code/生命周期组合不属于本项目五类完整事件，不是解析器不认识编码。

### 12.2 五动作候选的接受与拒绝

| 动作 | 接受 | 拒绝 | 拒绝原因及数量 |
|---|---:|---:|---|
| tap | 19,269 | 5,324 | orientation不一致1；找不到匹配完整raw contact 5,321；单指标签含多指2 |
| scroll | 59,937 | 40,172 | 找不到完整raw contact 1,385；contact被高优先级动作占用37,822；单指标签含多指965 |
| swipe | 70,431 | 6,052 | 找不到完整raw contact5,448；单指标签含多指604 |
| pinch | 58,016 | 4,566 | 找不到完整raw contact209；不是恰好双指1,273；contact被高优先级动作占用3,084 |
| keystroke | 49,158 | 61 | 缺1个按键contact 59；缺2个按键contact 2 |

这些拒绝不是训练阶段随机过滤，而是正式语料构建时的确定性 fail-closed 结果。`raw_contact_reserved_by_higher_priority_event` 主要用于避免同一物理接触重复成为两个动作，并不等同于传感器损坏。`no_matching_complete_raw_contact` 表示派生标签存在但无法找到完整物理生命周期，不能靠插值补成“真实”轨迹。接受数之和严格为256,811。

### 12.3 正式数据规范检查

| 检查 | 规范 | 结果/处理 |
|---|---|---|
| 数值安全 | NPZ 必须 `allow_pickle=False`；object array=0；数值 finite | 正式五动作通过 |
| offset | 从0开始、非递减、末项精确等于对应flat数组长度 | 通过 |
| 时间 | event内时间非降；生成结果必须严格递增整数毫秒 | real通过；fake待100k实测 |
| 单指语义 | tap/scroll/swipe不得出现第二指 | 违规候选拒绝；正式通过 |
| pinch语义 | 恰好两个pointer，保留错峰起止和union clock | 非双指候选拒绝；正式通过 |
| keystroke语义 | 每键完整contact；hold/flight/keycode与offset一致 | 61个不完整候选拒绝；正式通过 |
| 互斥 | 同一raw contact只属于一个正式动作 | 通过；冲突按优先级记录 |
| 用户划分 | 生成侧70/10/20用户互斥 | 通过 |
| five-shot | 每组至少6事件，refs恰好5且固定 | 正式v2通过；旧语料失败后弃用 |
| 长度 | batch可padding但不可裁剪，`drop_last=False` | 代码门通过；训练持续按epoch审计 |
| 特征 | detector输入不得NaN/Inf；序列mask与长度一致 | 代码门通过；正式bundle待生成 |
| 生成物理门 | lifecycle、端点、bounds、clipping率 | 待100k，从当前bytes完整审计 |

### 12.4 已知但尚不能消除的限制

| 编号 | 问题 | 当前影响 | 处理/最终报告要求 |
|---|---|---|---|
| D-01 | extractor启动时的源代码bytes未复制进输出；运行期间源文件发生仅文档编辑 | 现有zip、启动source hash和补充provenance可追溯，但不能声称最强级别的运行时source binding | 永久披露；不改写历史事实 |
| D-02 | HMOG是当前唯一外部数据集 | 只能证明HMOG内评测，不能证明跨设备/跨数据集泛化 | 最终结论限定为HMOG；跨数据集需新数据 |
| D-03 / P-TRJ-167 | 当前PAD中fake按70/10/20用户划分，real按每用户内部60/20/20划分 | 类别两侧用户集合口径不对称，不能宣称PAD完全user-disjoint | 保留当前冻结主协议结果；增加同一70/10/20用户划分的补充PAD评测后才可作user-independent结论 |
| D-04 / P-TRJ-168 | detector real pool与generator的5条reference registry独立hash划分，可能重叠 | detector可能见到作为生成条件的real reference；主协议会完整报告overlap但不删除 | 最终增加“排除所有reference real事件”的敏感性/消融结果，比较指标变化 |
| D-05 | 20-step smoke clipping失败 | 不能用smoke证明正式生成物理质量 | 只接受100-epoch best-EMA下100k完整审计 |
| D-06 | 正式训练期间曾有外部GPU进程争用和85–87°C高温 | 可能延长训练时间；未见hardware thermal slowdown，未改样本或模型定义 | 问题清单逐PID保留；最终runtime解释资源条件 |

## 13. HMOG 标准数据集测试：详细执行与判定

### 13.1 生成模型测试边界

生成模型的 train/validation/test 用户严格为70/10/20互斥。训练只读取70名train users；条件 shrinkage prior 只从train拟合；epoch20/40/60/80/100只在10名validation users上选择best-EMA；20名test users不参与优化、超参数或checkpoint选择。最终100k中每个用户每个动作200条，因而fake为70,000/10,000/20,000。

生成模型测试至少包含四层：

1. 数值层：所有时间、坐标、压力、面积和条件为有限值，offset/mask/shape一致；
2. 物理层：毫秒时间严格递增，Android lifecycle合法，pinch恰好双指且保留错峰，keystroke不在gap伪造接触；
3. 身份层：fake ID、condition seed、noise seed唯一，五条refs固定且身份/hash与训练registry一致；
4. 非复制层：exact trajectory replay、完整metadata copy、完整key-sequence copy均按正式审计规则检查，不用detector筛选生成结果。

### 13.2 PAD 数据划分必须分两种口径报告

当前冻结主协议实际是：

| 类别 | train/validation/test 划分 |
|---|---|
| fake | 固定70/10/20用户，因此test fake只来自20名未见用户 |
| real | 每个 `(action,user)` 内对完整event group做确定性60/20/20 hash划分，因此real train/val/test均含100名用户的不同事件 |

这可评估“对未见用户fake与所有用户held-out real event的区分”，但不是严格的双侧用户独立 PAD。为避免错误结论，最终报告必须同时给出：

1. 冻结主协议结果，确保与已测试代码和既定产物一致；
2. 补充 fully user-disjoint 结果：real与fake均按同一70/10/20用户划分，scaler、模型、时长分箱只拟合train users，threshold只来自validation users；
3. reference-exclusion消融：从real detector pool排除所有固定5条reference事件，或至少单独报告overlap与排除前后差值。

在第2项未完成前，只能写“HMOG固定协议PAD结果”，不能写“完全用户独立检测性能”。这是当前开放方法学问题，不因25个主协议detector跑完而自动关闭。

补充实现已经独立落在`trajectory_pad_supplement_20260722`：它不修改冻结主协议，先从正式primary bundle重建real/fake双侧统一70/10/20用户owner，再构建排除全部500 refs/action的第二变体，并分别运行25个检测器。Registry通过主协议canonical loader从内容重算hash并绑定split SHA，不能靠修改JSON中的自报字段绕过。实现层synthetic unittest 4/4、编译和CLI入口已通过；正式bundle与指标仍必须等待100k和primary detector bundle，不能把代码通过写成补充实验完成。

### 13.3 五类检测器每一步

对每个动作分别执行以下流程，共5×5=25组：

1. 读取该动作的real/fake records，先核对标签、split、user、event/fake ID、duration和来源SHA；
2. 对linear/RBF/XGBoost提取冻结手工特征；只在train拟合scaler及模型；
3. 对TCN/Transformer读取变长序列与mask；只在train优化，不允许test early stopping；
4. 把分数方向统一为“越大越像fake”；保存逐样本score，不只保存汇总表；
5. 在validation分数上选EER threshold与满足FRR不超过5%的threshold；相等时按`score >= threshold`拒绝；
6. 冻结模型、scaler、分箱边界和两个threshold，再应用到test；
7. 输出全局指标、4个duration bins、500次user bootstrap、模型与数据SHA；
8. 独立审计逐样本score重算汇总，防止只信任报告JSON。

### 13.4 指标定义

设“real”为应接受类，“fake”为应拒绝类，分数越大越像fake：

```text
FA  = 被错误接受的 fake 数 / 全部 fake 数
FRR = 被错误拒绝的 real 数 / 全部 real 数
```

- ROC-AUC不依赖某个固定阈值，用于衡量整体排序；0.5附近表示接近随机，越接近1越好。
- EER operating point由validation选择FA与FRR最接近处，然后在test固定应用；test EER列实质是“validation-EER threshold下的test FA/FRR”，不能在test重新寻找等错误率点。
- FRR<=5% operating point也只由validation确定，用于观察在真实用户误拒受限时fake误接受率。
- 若某duration bin只有一个类别、样本为空或用户数不足，必须报告不可计算及原因，不能补0或删除该箱。

### 13.5 不同时长测试

每个动作只用train duration拟合4个分箱边界。validation和test沿用相同边界，且四个箱共用同一个validation全局threshold。最终每个detector至少应有：全局validation、全局test、4个validation bins、4个test bins，两种operating point均可由逐样本score复算。

时长分层的主要问题不是只看“最长箱AUC是否低”，还要同时检查每箱real/fake数量、用户数、FA/FRR和置信区间。极端时长箱样本少时，宽CI必须如实保留。

### 13.6 用户级 bootstrap

正式置信区间做500次user bootstrap。每次有放回抽取用户，被抽中的用户携带其全部样本；同一用户被抽中两次，其整组样本也重复两次。禁止把轨迹逐条独立bootstrap，因为同一用户的轨迹相关，会虚假缩窄置信区间。最终保存随机seed、每次抽样结果或足以确定性重放的记录。

### 13.7 综合检测器测试

每个动作的paired row必须同时拥有trajectory score、IMU score、10维跨模态一致性components、action、duration、user/split、EventPlan身份。real基础已得到231,728个严格pair；fake侧只有100k轨迹与对应IMU完成后才能构建。

综合模型仍遵守train fit、validation定threshold、test固定评估和train-only时长分箱。必须同时报告三类结果：IMU单路、trajectory单路、total融合。只有total相对两条单路的改善及CI可复核，才能声称融合带来收益；不能只报告最好的一条。

## 14. 可复现命令与证据目录

正式源码根目录为：

```text
/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260722_v16_numeric_recovery
```

### 14.1 全语料审计

```bash
cd /home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260722_v16_numeric_recovery
python scripts/audit_training_corpus.py \
  --corpus-dir /home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713/results/trajectories_full_v2 \
  --output /new/audit-output/training_corpus_audit.json
```

该命令无sample cap。正式监督器已执行过；复现者应写入新输出或先确认不会覆盖需要保留的receipt。

### 14.2 单动作训练示例

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_trajectory_diffusion.py \
  --action tap \
  --corpus-dir /home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713/results/trajectories_full_v2 \
  --output-dir /new/training/tap \
  --device cuda:0
```

当前正式run已由supervisor管理，不应在相同输出目录重复启动该示例。恢复必须显式提供对应`last.pt`，且输出目录身份、source/config/corpus/registry hash全部匹配，否则拒绝。

### 14.3 100k 两分片生成与完整审计

```bash
python scripts/generate_five_shot_trajectories.py \
  --confirm-formal-100k --num-shards 2 --shard-id 0 --device cuda:0 \
  --corpus-dir /path/to/trajectories_full_v2 \
  --reference-registry-map /path/to/reference_registry_map.json \
  --checkpoint-map /path/to/best_checkpoint_map.json \
  --output-dir /new/output/path

python scripts/generate_five_shot_trajectories.py \
  --confirm-formal-100k --num-shards 2 --shard-id 1 --device cuda:1 \
  --corpus-dir /path/to/trajectories_full_v2 \
  --reference-registry-map /path/to/reference_registry_map.json \
  --checkpoint-map /path/to/best_checkpoint_map.json \
  --output-dir /new/output/path

python scripts/audit_five_shot_generation.py \
  --num-shards 2 \
  --output-dir /new/output/path \
  --corpus-dir /path/to/trajectories_full_v2 \
  --reference-registry-map /path/to/reference_registry_map.json \
  --condition-preflight /path/to/passed_all_100k_condition_preflight.json
```

`/path/to/...`是需要由该次正式run manifest解析并核对SHA的占位符，不应手工猜测checkpoint文件名。

### 14.4 Detector bundle与25组benchmark

```bash
python scripts/audit_generation_pad_export.py \
  --generation-root /new/output/path \
  --split-json /home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json \
  --require-formal

python scripts/build_trajectory_pad_bundle.py \
  --real-dir /path/to/trajectories_full_v2 \
  --fake-archive-dir /new/output/path \
  --output-dir /new/detector_dataset \
  --fake-user-split /home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json \
  --reference-registry-map /path/to/reference_registry_map.json

python scripts/run_trajectory_benchmark.py \
  --dataset-dir /new/detector_dataset \
  --fake-user-split /home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json \
  --output-dir /new/complete_benchmark
```

### 14.5 当前关键证据位置

| 内容 | 路径 |
|---|---|
| 正式v2语料与全量审计 | `trajectory_humanization_full_20260713/results/trajectories_full_v2` |
| 正式launch gates | `trajectory_humanization_full_20260722_v16_numeric_recovery/results/formal_eventplan_v16_numeric_recovery_launch_gates_20260722` |
| 当前五动作run | `trajectory_humanization_full_20260722_v16_numeric_recovery/results/formal_eventplan_v16_numeric_recovery_100epoch_100k_20260722` |
| 综合检测下游run | `trajectory_estimator_pack_20260721/results/formal_paired_total_eventplan_v16_20260722` |
| 唯一实时状态与问题清单 | `trajectory_estimator_pack_20260721/docs/IMU与轨迹交付状态及问题清单.md` |

## 15. 最终结果应如何填写和判读

正式实验结束后，每动作至少补齐下列汇总；原始逐样本scores、分箱结果和bootstrap不能只浓缩成这一张表。

| action | detector | val AUC | test AUC | EER阈值下test FA/FRR | FRR<=5%阈值下test FA/FRR | 4 bins齐全 | 500×CI | 状态 |
|---|---|---:|---:|---|---|---|---|---|
| tap | linear/RBF/XGB/TCN/Transformer | 待 | 待 | 待 | 待 | 待 | 待 | 待运行 |
| scroll | linear/RBF/XGB/TCN/Transformer | 待 | 待 | 待 | 待 | 待 | 待 | 待运行 |
| swipe | linear/RBF/XGB/TCN/Transformer | 待 | 待 | 待 | 待 | 待 | 待 | 待运行 |
| pinch | linear/RBF/XGB/TCN/Transformer | 待 | 待 | 待 | 待 | 待 | 待 | 待运行 |
| keystroke | linear/RBF/XGB/TCN/Transformer | 待 | 待 | 待 | 待 | 待 | 待 | 待运行 |

最终结论按以下规则写：

- 只有模型与固定阈值均来自train/validation且test从未用于选择，才称为test结果；
- AUC高但FA或FRR不可接受时，不称“检测成功”，必须并列报告工作点；
- 全局指标好但某时长箱显著退化时，必须把退化动作/区间列为限制；
- CI重叠时不凭单点均值宣称某模型显著更优；
- 主协议与fully user-disjoint补充协议分表，不混为一个数字；
- reference-exclusion消融未完成前，不声称结果完全排除了enrollment reference overlap影响；
- 任一100k物理门、身份门、hash门失败时，后续detector指标即使能算也不得作为正式结果发布。

## 16. 当前完成度结论

截至本文版本，轨迹方法、数据结构、标准数据集测试步骤、错误处理规则和最终验收口径已经写全；正式数据提取与审计、固定five-shot、代码/launch gates、100k条件预检以及tap/scroll/swipe训练已通过。pinch和keystroke仍由后台训练，100k真实生成、25个轨迹检测器、5个综合检测器及两个方法学补充评测尚未完成。因此当前状态是“文档与协议完整、正式计算继续”，不是“全部实验完成”。
