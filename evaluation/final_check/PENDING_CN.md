# 交接：真实状态、在跑的、没做的

**状态时间：2026-08-10。** 本文是冷启动接手的唯一入口，读完这一份就知道现在到哪了。目录地图见 [`README.md`](README.md)。

全程不改动那 6 个冻结检测器（只读、仅作对照）。

> `final_check/` 是共享目录，同时挂着两条线：本轮的 session / 任务2 / 实验B / A·B·C 评审，以及并列的 [`AUDIT_CN.md`](AUDIT_CN.md)（威胁模型与统计口径核查，产物 `scores/reference_overlap.json`）。两条线互不冲突。

---

## 一、会话检测器数字（2026-08-10 重跑，以磁盘为准）

**唯一权威源**：

- RF —— [`scores/session_detector_results.json`](scores/session_detector_results.json)
- LogReg —— [`scores/session_detector_logreg.json`](scores/session_detector_logreg.json)

两份都已把键名从 `windows` 改成 `occupancy_bin_widths_s`（旧名会被读成滑窗，是错的），且每个被抓率都带 user-clustered bootstrap CI95。

### 1.1 会话数变了：2394 → 2328

装配现在**拒收**时间线不可能的会话，而不是像以前那样把间隔裁进 `[0, 120]` 硬凑。裁剪会把「时钟故障」偷偷变成「不寻常的人」，然后要检测器去分辨——那个题目本身是错的。

| 项 | 值 |
|---|---|
| 保留 | 2328 |
| 拒收（时间线不可能） | 66 |
| 拒收（可用间隔不足 2 个） | 0 |
| 其中负间隔总数 | 6 |
| 其中超过 120 s 上限的间隔总数 | 74 |
| 最坏间隔 | 1,396,389,999,999,541.2 秒 |

拆分随之变化：train 1677 → **1639**，val 239 → **232**，test 478 → **457**（1639+232+457=2328）。

**拒收报告在 `/mnt/share/mwang49/data7/session_rhythm_detector/results/session_rejects.json`，不在本目录。** 本目录的 `scores/assemble_summary.json` 仍是 8-09 那一版（写着 2394），**已过期**，见第六节。

### 1.2 RF（test 457 真人 / 每攻击臂 457）

FRR5 与 FRR1 两行**完全相同**——这不是 bug，见 §1.4。

| bin | 真人误报 | naive | naive_jitter | paced v1 | paced_emp v2 |
|---|---|---|---|---|---|
| 2 s | 0.4 [0.0, 1.1] | 100 | 100 | **68.0 [62.4, 73.6]** | **5.5 [1.7, 10.2]** |
| 3 s | 0.4 [0.0, 1.1] | 100 | 100 | **67.0 [62.0, 71.9]** | **5.9 [1.9, 10.9]** |
| 5 s | 1.1 [0.0, 3.1] | 100 | 100 | **64.1 [58.5, 69.9]** | **5.5 [1.7, 10.2]** |

（单位 %，方括号是 CI95。阈值 `threshold_frr5 = threshold_frr1 = 0.0`。）

### 1.3 LogReg（test 457 真人 / 每攻击臂 457）

| bin | 操作点 | 真人误报 | naive | naive_jitter | paced v1 | paced_emp v2 |
|---|---|---|---|---|---|---|
| 2 s | FRR5 | 7.9 [3.9, 13.1] | 100 | 100 | **100.0 [100.0, 100.0]** | **4.6 [2.2, 7.8]** |
| 2 s | FRR1 | 1.8 [0.4, 3.6] | 100 | 100 | **99.3 [98.7, 100.0]** | **1.3 [0.2, 2.9]** |
| 3 s | FRR5 | 7.4 [3.7, 12.0] | 100 | 100 | **100.0 [100.0, 100.0]** | **5.0 [2.2, 8.7]** |
| 3 s | FRR1 | 3.1 [0.9, 5.8] | 100 | 100 | **99.6 [98.9, 100.0]** | **1.8 [0.4, 3.5]** |
| 5 s | FRR5 | 8.1 [4.4, 12.6] | 100 | 100 | **100.0 [100.0, 100.0]** | **6.1 [3.1, 9.9]** |
| 5 s | FRR1 | 2.4 [0.5, 4.9] | 100 | 100 | **99.8 [99.3, 100.0]** | **2.6 [0.7, 5.4]** |

### 1.4 四条限定，**凡引用上面数字的地方都必须一起写**

0. **这个实验里没有一条我们生成的假事件——这是最容易被读错的一点。** 五条臂共用同一批**真人**会话骨架（同动作序列、同个数、同手势时长），攻击臂只把**手势间隔**换掉（`code/assemble_sessions.py: rewrite_gaps()`）。所以上面的被抓率量的是**「间隔时序」这一根轴上的假想攻击**，不是我们真正的攻击。生成的假行塌在 `generated-hmog_u006` 这种每用户合成桶里（核实见 §3.2），在任何真人时间线上都没有位置，根本进不了这次装配。任何「我们的假数据逃过了会话检测」的读法都是错的。详见 §3.2。

1. **这不是滑窗。** 一个会话出**一个**分数。W 只是占用率特征给 onset 分箱的箱宽——没有步长、没有逐窗分数、没有因果告警。它**说不出**「第 40 秒开始可疑」。结果 JSON 里现在带一个 `not_a_sliding_window` 字段，写的就是这句。

2. **v2 的结果近乎同义反复。** `paced_emp` 从真人间隔分布抽间隔，而这个检测器**只读间隔统计量**。从人类分布里抽出来的东西，对一个只看间隔统计量的检测器当然难。它证明的是「**手势间隔这条轴可以靠复制人类分布化解**」，**不是**「会话级威胁已解决」。仍然没解决的是：与界面状态的**内容耦合**，以及手势之间的**连续 IMU 背景**（真人的手不会静止；真人 2 s 窗死寂的比例是 0%，见 [`TASK2_IMU_AXIS_CN.md`](TASK2_IMU_AXIS_CN.md)）。

3. **RF 的操作点退化了。** 它的 FRR5 与 FRR1 两行相同，因为 RF 把几乎所有真人会话推到分数 0，两个分位点落在同一个阈值上。它**实际**的误报率是 0.4%（bin 2 s / 3 s；bin 5 s 是 1.1%），远严于名义的 5%。要这么说，别让重复的两行看着像 bug。

### 1.5 与旧文档的差异（方向和量级都没变）

| 项 | 旧 | 新 |
|---|---|---|
| RF paced v1 (bin 2 s) | 72.2% | 68.0% |
| LogReg 真人误报 (bin 2 s, FRR5) | 5.0% | 7.9% |
| LogReg v2 (bin 2 s, FRR5) | 5.0% | 4.6% |
| 会话数 | 2394 | 2328 |

结论不变：**v1 被抓，v2 抓不住。**

---

## 二、已完成

| 主题 | 结论 | 位置 |
|---|---|---|
| 会话节律检测器 | 未整形机器 100% 被抓；v1 pacing 64–100% 被抓；v2 pacing（从真人经验分布抽、带长尾）压到真人误报地板 | 数字见 §1；说明见 `SPEC_CN.md` / `EVALUATION_CN.md`（两份已于 2026-08-10 08:00 同步重写，数字与 §1 一致） |
| 任务 2：连续 IMU 轴 | **更正：能做**——数据在原始 HMOG（98 Hz 连续，`/home/mwang49/Human_agent/hmog_dataset.zip`）；手势之间 0% 真人 2 s 窗死寂 → 比节律更强的判别轴 | `TASK2_IMU_AXIS_CN.md`、`code/imu_background_probe.py` |
| 实验 B：同人/异人距离 | keystroke/tap **保留目标风格**（D_fake < D_inter，user-clustered bootstrap CI 不含 0）；**scroll 待定**（08-10 重跑后 CI 含 0，见 §6.7）；swipe 不确定；**pinch 不保留**（诚实负结果） | `EXPERIMENT_B_CN.md`（数字过期）、`code/style_distance.py`、`scores/style_distance.json`（08-10 08:20 重跑） |
| A/B/C 评审 | 三个质疑都成立；C 指出 joint 只证明 event-alignment，不证明 physical coupling | `ASSESSMENT_ABC_CN.md` |
| A 的门限已冻结 | `gate_rules.json`（`frozen_at: 2026-08-10`）——三种简单攻击的定义、dev AUC 门槛 0.80/0.90、失败处置规则，都在跑任何数字之前写死了 | `gate_rules.json` |
| five-shot 优先级映射 | `code/fiveshot_priority.py`：把已发布的触摸侧五条映射成惯性生成器索引的行号，让 `UserRefBank` 把它们排在最前，k=1⊂k=3⊂k=5。这是 AUDIT §1 / issue #1 重跑的前置件，**已写好、尚未驱动重跑** | `code/fiveshot_priority.py` |

一条贯穿发现：会话节律（间隔轴）可以靠复制人类间隔分布化解（v2 已证），说明真正的会话级威胁在**内容耦合 + 连续 IMU 背景**，不在时间对齐——与 issue C 对 joint 的批评同指一处。

---

## 三、进行中（其他 lane，本轮同时在跑）

**这一节是新加的。以下四项不在本目录里，正在别的 lane 设计/运行；接手前先确认它们的进度，别重复造。**

### 3.1 因果滑窗会话检测器（重做）

现有会话检测器一个会话一个分数（§1.4 第 1 条），因此**不能**回答部署里唯一要紧的问题：「什么时候报警」。正在设计的是真正的因果滑窗版——有步长、有逐窗分数、有累积告警规则。它是**另一个检测器**，不是现有那个改参数，现有结果不能改标签冒充它。

### 3.2 会话 2×2：内容 × 时间

现在这套 `paced` 臂只在**真人骨架**上替换手势间的**间隔**——生成的假事件**根本没进去过**。假行的 `session_id` 是 `generated-hmog_u006` 这样的**每用户合成桶**（已在 `/mnt/share/mwang49/data7/results/direct100k/replay_dataset_v3/shards/hmog_u006.npz` 的 `session_id` 列上核实：真人行是 `hmog_u006_s07` / `hmog_u006_s18` 这类，假行 1000 条全部塌进 `generated-hmog_u006` 一个值），**在时间线上没有位置**。

所以既有的会话实验测的是一个**只有时间、没有内容**的假想攻击，不是我们的攻击。新的 2×2 把两条轴拆开：{真内容, 假内容} × {真时序, 假时序}，四格分别评。

### 3.3 分数空间的会话聚合基线

把逐事件检测器的分数按会话聚合（均值/分位数/越阈计数）当会话级基线，用来回答「会话级信号有多少其实是逐事件检测器早就有的」。这是新会话检测器必须打过的对照。

### 3.4 joint 半攻击消融（**吸收了原来的 C**，见 §4.2）

---

## 四、未完成

### 4.1 A. 深度检测器 competence gate（需 GPU）

- **门限已冻结**：`gate_rules.json`，`frozen_at: 2026-08-10`，`frozen_before` 字段写明选门槛时没看过任何 ActReal 数字。三种简单攻击（`fixed_injection` / `linear_trajectory` / `replay_jitter`）的定义、dev split、逐 action×modality×detector 共 90 格、AUC ≥ 0.80 通过（严 0.90）、未过标 `training_failure` 单独报且**不得**用于支撑 ActReal 高 FAR、**不得**看了 FAR 再删格子——全部在文件里。
- **还要做**：三种攻击的生成器 + 逐配置 dev AUC 表。两种检查：(b) 拿已训练好的检测器直接测简单攻击（先跑，便宜）；(a) 只对 (b) 未过的配置，用简单攻击训练同架构验证学习能力（贵，只在诊断处跑）。
- **复用**：只读检测器运行器 `/mnt/share/mwang49/data7/direct100k_final/code/dataset_test/scripts/run_hmog_direct100k_detectors.py`；冻结模型 `/mnt/share/mwang49/data7/direct100k_final/detector_models/<cell>/model.joblib`（90 个）。
- **验收**：一张逐配置 AUC 表 + training-failure 清单。
- **动机数字（已在冻结前测过，记在 `gate_rules.json` 的 `motivation_measured_before_freezing` 里）**：AUC < 0.55 的格子 10 个，< 0.60 的 31 个；最差的是 `keystroke__trajectory_xytime__hmog_style_svm` = 0.504。这些数是**建门的理由**，不是门的输出。
- 详见 [`ASSESSMENT_ABC_CN.md`](ASSESSMENT_ABC_CN.md) A 节。

### 4.2 C. joint 耦合探针 —— **已被另一条 lane 的「半攻击消融」吸收，不要再单独开**

**C 原本的动机保留不变**（这一段是 C 存在的理由，别丢）：

> **前置事实（已核实）**：IMU 生成的条件是**动作 + 时长 + 设备方向 + five-shot 参考**，**不含最终轨迹方向**。向左 swipe 600 ms 和向右 swipe 600 ms 从**几乎相同的 IMU 分布**采样，系统只是把这条 IMU 和各自的触摸放进同一个时间区间。所以 **FAR = 0.711 只证明 event-alignment（事件对齐），不证明 physical coupling（物理耦合）**——时间同步 ≠ 内容耦合。

**现在由谁做**：另一条 lane 正在设计 **joint 半攻击消融**，在**冻结的 joint 检测器**上打分、**不重训**，网格是：

| 行 | 轨迹 | IMU | 问的问题 |
|---|---|---|---|
| 1 | 假 | 真 | 只有轨迹是假的，joint 抓得住吗 |
| 2 | 真 | 假 | 只有 IMU 是假的，joint 抓得住吗 |
| 3（错配对照） | 真 | 真，但取自**同一用户的另一条真实事件** | 两边都是真人的，只是配错——joint 认不认 |

**第 3 行的错配对照，就是 C 原来那条「同动作内随机交换 IMU」探针。** 两边都是真人数据，joint 若仍判「真」，说明它根本没在看跨模态一致性，「过 joint」就只是分别判了两个单模态。

**接手要做的**：不要在本目录重开 `joint_coupling_probe/`；去那条 lane 看半攻击消融的进度，把 C 的其余三种扰动（IMU 时间平移、左右方向交换、不同时长配对）作为该网格的追加行提出去，别另起炉灶。

### 4.3 任务 2 的攻击侧连续背景流（需真机）

真人连续 IMU 已可从原始 HMOG 抽取；**唯一缺口**是攻击侧要有真机 Agent 跑出的连续 IMU 流才能公正评测——否则用我们自己的背景模型生成再拿去测，就是自证。建议留到有真机 run 之后做。

### 4.4 四处 tex 措辞 —— **没有做，等作者点头**

**状态：未执行。论文是作者的，本轮一个字都没改，也不打算擅自改。**

要改的是把 "touch-conditioned / learned physical coupling" 换成 **"event-aligned" / "time-aligned"**，理由见 §4.2 的前置事实。

`ASSESSMENT_ABC_CN.md` 里记的四个行号是 **2026-08-09 写的，现在已经对不上了**。2026-08-10 在 `/home/mwang49/new/data7/data7_final_monitor_metrics_v1/USENIX8.25/paper/sections/` 下核对：

| 文档里记的 | 现在实际 |
|---|---|
| `threat_model.tex:95` | 仍在，该行为 "jointly generate touch, timing, and IMU, cross-signal consistency no longer" |
| `introduction.tex:84` | 该行现在是 "state. Finally, physical realization must place the IMU response for the current" |
| `method.tex:8` | **对不上**，该行是 ActReal 架构描述，与耦合措辞无关 |
| `evaluation.tex:53` | **对不上**，该行是空行 |

`grep -rn "coupl" sections/` 另外命中 `introduction.tex:67`、`method.tex:166`。

**所以：作者点头之后，第一步是重新定位这四处，不能照着旧行号改。**

---

## 五、v2 pacing 的部署侧改动（在 actreal_agent，不在本仓库）

`/mnt/share/mwang49/data7/actreal_agent/actreal/pacing.py` 增加了 `empirical_gaps` 模式（从真人间隔分布抽、不截长尾）；已核实该字段在文件里（`empirical_gaps: dict[str, list[float]]`，第 95 行附近）。

**`actreal_agent` 仍然不是 git 仓库——2026-08-10 复核过**：`git rev-parse` 在该目录返回 `fatal: not a git repository`，目录下没有 `.git`。这份改动因此没有版本记录，只有文件本身和这一条。改它之前先自己备份。

---

## 六、发现的文档不一致（不在本 lane，交给对应负责人）

本轮核对路径与数字时发现下面几处，**本 lane 只报不改**：

1. **`scores/assemble_summary.json` 过期。** 写着 `genuine_sessions: 2394`、`median_gap_s.genuine: 1.099`，且**没有**任何拒收字段。新的一版在 `/mnt/share/mwang49/data7/session_rhythm_detector/results/assemble_summary.json`（2328 / 1.084），拒收明细在同目录 `session_rejects.json`。需要有人把这两份拷进 `scores/`。
2. ~~`EVALUATION_CN.md` 数字过期~~ —— **已解决（2026-08-10 08:00 重写，08:24 复核）**：两张结果表逐格与 `scores/` 的两个 JSON 一致；2394 / 72.2 只出现在「旧口径 → 新口径」的对照语境里。
3. ~~`EVALUATION_CN.md:88` 路径不存在~~ —— **已解决**：复现段现在写 `$SESSION_DIR/sessions_*.jsonl`，同段给出 `SESSION_DIR=/mnt/share/mwang49/data7/session_rhythm_detector/results`。**五**臂 jsonl 都在那里，`scores/` 下确实没有（事实，非笔误）。
4. ~~`SPEC_CN.md:12` 与实现矛盾~~ —— **已解决**：SPEC 新增第 0 节明写「不是滑动窗」，第 2 节改为「一个会话一个特征向量、一个分数」，全文称 W 为占用箱宽。已无文档再说滑窗。
5. **`ASSESSMENT_ABC_CN.md` 的四个 tex 行号已失效**，见 §4.4。
6. **`ASSESSMENT_ABC_CN.md:78` 建议的 `evaluation/joint_coupling_probe/` 不要再建**，已被 §4.2 的半攻击消融吸收。
7. **`scores/style_distance.json` 于 2026-08-10 08:20 被另一条 lane 重跑覆盖**（新增字段 `metric_fitted_on: "70 train users, genuine events only"`），而 `EXPERIMENT_B_CN.md` 的表还是 8-09 那一版。差异会**改判定**：scroll 的 CI95 新值 `[-0.115, 0.614]`（含 0），旧值 `[0.068, 0.740]`（不含 0）。§2 的「keystroke/tap/scroll 保留」这一行、`README.md` 的同一说法、以及 `EXPERIMENT_B_CN.md` 整张表，都要等那条 lane 定稿后一起重写。**在那之前不要引用 scroll 的「保留」结论。**
