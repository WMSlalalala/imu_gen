# final_check —— 目录地图

> **先读 [`PLAN_AND_RESULTS_CN.md`](PLAN_AND_RESULTS_CN.md)。** 它是本目录的主文档与唯一入口：
> 十四节，覆盖全部实验、每个的状态、已量出的数字，以及每个数字**不能**被读成什么。
> **本文件（README）只回答「哪个文件是干什么的、哪个数字由哪个文件担保」**，不复述任何结论。

本目录路径：`/mnt/share/mwang49/real-human/imu_gen/final/evaluation/final_check/`
Python：`/home/mwang49/miniconda3/envs/cuhkx/bin/python`

## 阅读顺序

| 顺序 | 读什么 | 为什么 |
|---|---|---|
| 1 | [`PLAN_AND_RESULTS_CN.md`](PLAN_AND_RESULTS_CN.md) | 主文档。**第零节的数据源更正必须先读**——两棵形状完全相同的 90 格树，用错那棵会静默得到 0.4447 而不是 0.7746 |
| 2 | [`release_cell_map.json`](release_cell_map.json) | 解析任何一个格子的唯一权威入口。**不要拼路径** |
| 3 | [`PENDING_CN.md`](PENDING_CN.md) | 交接状态：其他 lane 在跑什么、什么没做、已知的文档不一致 |
| 4 | 下面「文档」表里与你的问题相关的那一份 | 支撑细节 |
| 5 | `scores/` 里对应的 JSON | 数字的最终担保物，磁盘为准 |

主文档与支撑文档冲突时，**以 `scores/` 下的 JSON 为准**，其次主文档，最后支撑文档。

---

## 一、目录结构（核实于 2026-08-10 10:40）

```
PLAN_AND_RESULTS_CN.md    主文档：十四节，全部实验 + 状态 + 数字 + 读法边界
README.md                 本文件：目录地图
PENDING_CN.md             交接：在跑的、没做的、文档不一致清单
AUDIT_CN.md               并列线：威胁模型与统计口径核查（6 项判定 + 8 个 issue）
SPEC_CN.md                会话节律检测器 —— 规格
EVALUATION_CN.md          会话节律检测器 —— 结果
SESSION_AGGREGATION_EN.md 会话级聚合 —— 两实现 + 第三方裁判的对账（英文）
EXPERIMENT_B_CN.md        实验 B：同人/异人风格距离
TASK2_IMU_AXIS_CN.md      任务 2：连续 IMU 轴
ASSESSMENT_ABC_CN.md      A/B/C 三个批评的评审（最早的一份，2026-08-09）

gate_rules.json           冻结规则：competence gate 判据
fairness_rules.json       冻结规则：会话级与半攻击实验的 15 条公平性规则
release_cell_map.json     冻结规则：90 格的唯一权威解析入口
EXTERNAL_INPUTS.json      会话链路读写、但因体积不入库的产物清单

code/                     5 个本目录脚本 + 2 个子目录（见第四节）
scores/                   9 个结果 JSON（见第三节）
```

**这个目录正在被多条 lane 同时写。** 本轮盘点期间 `code/pipeline/`（50 个文件）、
`code/session_aggregation/`（5 个）、`scores/` 下 4 个 JSON 都是当天新增或被覆盖的。
文件时间戳比任何文档里的描述都可靠。

---

## 二、文档：各自回答什么

| 文档 | 回答什么 | 状态（核实于 2026-08-10 10:40） |
|---|---|---|
| [`PLAN_AND_RESULTS_CN.md`](PLAN_AND_RESULTS_CN.md) | 全部十四节：数据源更正、主实验、对比、消融、会话聚合、会话节律、风格距离、审计、五-shot 重采样、半攻击联合消融、身份实验、自适应攻击者、滑窗（设计被否）、待办 | **最新，主文档。** 数字已逐表对过磁盘，结论一致；三处需要更正，见第六节 |
| [`PENDING_CN.md`](PENDING_CN.md) | 冷启动交接：会话检测器全部数字、其他 lane 的四项、未完成四项、文档不一致清单 | 数字有效；**§1.1 与 §6.1 已被事实追上**——它说 `session_rejects.json`／新版 `assemble_summary.json`「不在本目录」，两份现在都已在 `scores/` 里（2026-08-10 10:23 提交） |
| [`AUDIT_CN.md`](AUDIT_CN.md) | 触摸与惯性是否共用五条参考（**不是**，四手势实际 10-shot）、主判据有无 CI（已补）、多折、扩散先验来源、联合模态偏低、两次 IMU 透传虚警的证伪 | 有效。§1 的 400/354 与 `scores/reference_overlap.json` 逐项一致 |
| [`SPEC_CN.md`](SPEC_CN.md) | 会话节律检测器的规格与纪律：用户不相交划分、开发集选阈值、paced 从不进训练、**不是滑窗** | 有效，与实现一致（第 0 节明写「不是滑动窗」「生成的假事件不进这个实验」） |
| [`EVALUATION_CN.md`](EVALUATION_CN.md) | 会话节律检测器的结果与读法（RF / LogReg × 三个占用箱宽 × 两个操作点） | 有效。两张结果表逐格与 `scores/` 的两个 JSON 一致（本轮复核） |
| [`SESSION_AGGREGATION_EN.md`](SESSION_AGGREGATION_EN.md) | 会话级聚合：A/B 两个独立实现 + 第三方裁判的逐项对账，52 处分歧全部归因到三个**报告口径**选择 | 有效，是主文档第四节主表的来源。**文末「Files:」列的 `reconciled.json`、`C/referee*.json` 在本仓库里不存在**，见第六节 |
| [`EXPERIMENT_B_CN.md`](EXPERIMENT_B_CN.md) | 生成是否保留目标用户风格（keystroke/tap 保留，scroll/swipe 不确定，pinch 不保留） | **已于 2026-08-10 08:59 按重跑更新**，与 `scores/style_distance.json` 逐格一致。旧 README 说它「数字已过期」，那条已作废 |
| [`TASK2_IMU_AXIS_CN.md`](TASK2_IMU_AXIS_CN.md) | 连续 IMU 轴能不能做（更正：能，数据在原始 HMOG 归档）、为什么比节律轴强、还缺什么 | 有效 |
| [`ASSESSMENT_ABC_CN.md`](ASSESSMENT_ABC_CN.md) | A（competence gate）/ B（距离不等式方向）/ C（joint 证明了什么）三个批评的评审，结论是三个都成立 | 结论有效，**但它是本目录里最旧的一份（2026-08-09）**：四个 tex 行号已失效（见 `PENDING_CN.md` §4.4）；它建议新建的 `competence_gate/`、`style_distance/`、`joint_coupling_probe/` 三个目录**都不要再建**——分别已落到 `scores/competence_gate_classical.json`、`code/style_distance.py`、以及另一条 lane 的半攻击消融里 |

---

## 三、`scores/` —— 哪个 JSON 担保哪个数字

| 文件 | 担保主文档的哪一处 | 由谁生成 | 状态 |
|---|---|---|---|
| `session_aggregation.json` (82 KB) | **第四节全部三张表**：FRR5/FRR1 被抓率与 CI、检测代价、count 规则 k 曲线、标定乐观量 | `code/session_aggregation/make_reconciled.py`（汇总 `sessagg_a.py` / `sessagg_b.py` / `referee.py` 的输出） | 有效，逐格核对一致。`meta.referee_script` 指向一个 `/tmp` 临时路径，脚本本体现已在 `code/session_aggregation/` |
| `session_aggregation_per_cell.json` (23 KB) | 90 格逐格 far5/frr5/farE/frrE。自算均值 = **0.774575**，与主文档第一节 0.775 一致 | 同上 | 有效 |
| `session_detector_results.json` | 第五节 **RF** 行（真人误报 0.4、paced v1 68.0、v2 5.5，及 CI） | `code/session_detector.py --model rf` | 有效；键名 `occupancy_bin_widths_s`，带 `not_a_sliding_window` 字段 |
| `session_detector_logreg.json` | 第五节 **LogReg** 两行 | `code/session_detector.py --model logreg` | 有效 |
| `style_distance.json` | **第六节整张表**：逐动作 D_intra/D_fake/D_inter、中位 Δ、user-clustered bootstrap CI95、理想链成立率、20 名 test user 的 `per_user` 明细 | `code/style_distance.py` | 有效；`metric_fitted_on: "70 train users, genuine events only"`——**读任何一版风格距离前先看这个字段**，它是新旧口径唯一的区分标志 |
| `assemble_summary.json` (315 B) | 第五节「装配拒绝 2394 → 2328」的保留侧，与各臂间隔中位 | `code/assemble_sessions.py` | **已是新版（2328 / 中位 1.084）**，2026-08-10 10:23 替换。旧的 2394 版已不在本仓库 |
| `session_rejects.json` (4 KB) | 第五节「已知缺陷」段：66 段被拒、6 个负间隔、74 个超 120 s 间隔、最坏 1.396e15 秒 | `code/assemble_sessions.py` | **新入库**（同上）。`detail` 只含前 20 条明细，不是全部 66 段 |
| `reference_overlap.json` | 第七节第 1 条「400 组里 354 组完全不相交」、keystroke 20,000/20,000 两通道同源 | `../comparison/code/check_reference_sync.py`（**跨目录**） | 有效 |
| `competence_gate_classical.json` (140 KB) | 第十三节「competence gate 经典四族已在 CPU 上跑」的那份产物 | competence-gate lane，`run: "v2 -- rerun after the data-source correction"` | 有效但**尚未进主文档**：它的 `plain_statement` 里有一条直接限定第一节头条的结论，见第六节 |

**`scores/` 下没有 `sessions_*.jsonl`**——这是事实不是笔误。五臂会话流共 51.5 MiB，按 `EXTERNAL_INPUTS.json` 的记录留在库外（见第五节）。

---

## 四、`code/` —— 哪个脚本产出什么

### 本目录脚本

| 脚本 | 产出 | 备注 |
|---|---|---|
| `code/assemble_sessions.py` | `sessions_*.jsonl`（五臂，写到 `$SESSION_DIR`）+ `assemble_summary.json` + `session_rejects.json` | 真人会话从 `trajectories_full_v2/hmog_trajectory_<action>.npz` 的 `flat_system_time_ms` 重建；五臂共用同一批真人骨架，只换间隔。会话若有**任何**负间隔或超 `MAX_GAP_S = 120.0` 的间隔就整条拒收，不裁剪。`paced` 臂需要 `actreal.pacing.DelayPolicy` |
| `code/session_detector.py` | `session_detector_results.json` / `session_detector_logreg.json` | 一个会话一个分数；`--bins` 是**占用箱宽**不是滑窗（旧别名 `--windows` 仍可用）。只在 genuine vs `naive_jitter` 上训练，`paced` 从不进训练。阈值在 val 真人上取 FRR=5% 与 1% |
| `code/style_distance.py` | `style_distance.json` | 读 `/mnt/share/mwang49/data7/direct100k_final/datasets`；逐事件 45 维特征，度量只在 70 名训练用户的真人事件上拟合；user-clustered bootstrap 10,000 次 |
| `code/imu_background_probe.py` | 无落盘，直接打印 | 任务 2 的证据：从原始 HMOG 一个 session 目录读 `Accelerometer.csv` / `Gyroscope.csv` / `TouchEvent.csv`，量手势之间的 IMU 背景。用法在 docstring（要先从 6.1 GB 归档解出一个 session） |
| `code/fiveshot_priority.py` | 无独立产物，是**库** | 把已发布的触摸侧五条经 `genuine_bindings.jsonl` 映射成惯性生成器索引的行号，供 `UserRefBank` 排序，保证 k=1 ⊂ 3 ⊂ 5。第八节重采样的前置件 |

### `code/session_aggregation/` —— 第四节的三个实现

| 脚本 | 是什么 |
|---|---|
| `sessagg_a.py` / `sessagg_b.py` | 两个**独立**实现（差分测试对），只做冻结 `test_scores` 上的分数空间算术，不加载模型、不推理 |
| `referee.py` | 第三方裁判实现：第三套种子（900001+7r, R=20）、两种用户划分 + 200 个随机均衡划分、两种代价聚合口径、会话切点的暴力验证 |
| `analyze_repo.py` | 对两棵 90 格树各自复算事件级 FAR，用来定位 0.775 属于哪棵树 |
| `make_reconciled.py` | 汇总三方输出 → `scores/session_aggregation.json` |

> **这四个脚本目前不能从克隆直接重跑**：`make_reconciled.py` 与 `analyze_repo.py` 的输入/输出常量指向一个会话级 `/tmp` scratchpad 路径，那个目录不随仓库存在。要复现需先把这些常量改成库内路径。

### `code/pipeline/` —— 从未入过版本控制的 48 个实验驱动

`/mnt/share/mwang49/data7/code/baselines/` 的**逐字节副本**（2026-08-10 快照，48/48 sha256 校验通过），按 `code/ docs/ infra/ queues/ runners/ tests/ vendor/` 分类。
**它是档案，不是第二份工作副本**：这些脚本内部硬编码 `C=/mnt/share/mwang49/data7/code/baselines` 并 `cd "$C"`，从这里启动执行的仍是 data7 那份。详见 [`code/pipeline/README.md`](code/pipeline/README.md)。

### 不在本目录的脚本

AUDIT 那条线在 `/mnt/share/mwang49/real-human/imu_gen/final/evaluation/comparison/code/`：
`check_reference_sync.py`（AUDIT §1，产出 `scores/reference_overlap.json`）、`bootstrap_far5.py`（AUDIT §2，产出 `../BOOTSTRAP_FAR5.md`）、`covered_modalities.py`（AUDIT §6）。

---

## 五、冻结规则文件与库外路径

### 三个冻结规则文件（都在本目录根，都不是结果）

| 文件 | 管什么 | 冻结 |
|---|---|---|
| `gate_rules.json` | competence gate 判据：三种简单攻击定义、dev split、90 格、AUC ≥ 0.80 通过 / 0.90 严格、`training_failure` 的报告方式、**禁止事后按 FAR 删格子**。里面的 `motivation_measured_before_freezing` 是**建门的理由**，不是门的输出 | `frozen_at: 2026-08-10` |
| `fairness_rules.json` | 会话级与半攻击实验的 15 条公平性规则 | 2026-08-10 07:52 |
| `release_cell_map.json` | 90 格的唯一权威解析入口：逐格 `scores` / `thresholds` / `frr5` / `eer` / `bundle` / `model_dir`。本轮逐格核对：**90/90 路径全部存在，90/90 阈值与磁盘一致** | 2026-08-10 09:50（与 data7 那份逐字节相同） |

### 库外路径（本文写到的**全部**已核实存在）

| 路径 | 是什么 |
|---|---|
| `/home/mwang49/new/data7/data7_final_monitor_metrics_v1/USENIX8.25/code/dataset_test/results/cells` | **发布版 90 格**，`test_scores.jsonl.gz`。自算 FAR@frr5 均值 0.7746 |
| `/mnt/share/mwang49/data7/results/direct100k/detectors_90cell/cells` | **r1 修复前基线，不是发布版**，自算 0.4447。形状完全相同，极易误用 |
| `/mnt/share/mwang49/data7/direct100k_final/detector_models/` | 90 个冻结检测器（60 `model.joblib` + 30 `checkpoint.pt`），**只读** |
| `/mnt/share/mwang49/data7/session_rhythm_detector/results/` | 会话链路的落盘目录（`OUT=` / `SESSION_DIR=`）：五臂 jsonl 51.5 MiB。**其中的 `.py` 与旧 `.json` 是 8-09 的过期副本，不要从那里跑复现** |
| `/mnt/share/mwang49/data7/code/baselines/` | 实验驱动的**活的**工作树，非 git 仓库；副本见 `code/pipeline/` |
| `/mnt/share/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713/results/trajectories_full_v2/` | 真人会话时间线来源（`hmog_trajectory_<action>.npz`） |
| `/mnt/share/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json` | 唯一的用户划分，70/10/20，五动作共用 |
| `/mnt/share/mwang49/data7/direct100k_final/datasets/` | 发布版逐事件数据集（4 个 bundle） |
| `/mnt/share/mwang49/data7/direct100k_final/code/dataset_test/scripts/run_hmog_direct100k_detectors.py` | 只读检测器运行器 |
| `/mnt/share/mwang49/data7/actreal_agent/actreal/pacing.py` | v2 pacing 的 `empirical_gaps`。**该目录不是 git 仓库**，改动无版本记录 |
| `/home/mwang49/Human_agent/hmog_dataset.zip` | 原始 HMOG 归档（6.1 GB），任务 2 连续 IMU 的唯一来源 |

完整的库外产物清单（体积、sha256、再生成命令）在 [`EXTERNAL_INPUTS.json`](EXTERNAL_INPUTS.json)（会话链路）与 [`code/EXTERNAL_INPUTS.json`](code/EXTERNAL_INPUTS.json)（pipeline 驱动）。

### 复现命令

```bash
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
CF=/mnt/share/mwang49/real-human/imu_gen/final/evaluation/final_check
OUT=/mnt/share/mwang49/data7/session_rhythm_detector/results   # 输出目录，不是代码目录

# 会话链路：装配 → 打分。SESSION_DIR 必须给，否则 session_detector 读的是默认 scratchpad
$PY $CF/code/assemble_sessions.py --out $OUT
SESSION_DIR=$OUT $PY $CF/code/session_detector.py --model rf     --bins 2,3,5 \
    --out $CF/scores/session_detector_results.json
SESSION_DIR=$OUT $PY $CF/code/session_detector.py --model logreg --bins 2,3,5 \
    --out $CF/scores/session_detector_logreg.json

# 实验 B
$PY $CF/code/style_distance.py --out $CF/scores/style_distance.json

# AUDIT §1
$PY /mnt/share/mwang49/real-human/imu_gen/final/evaluation/comparison/code/check_reference_sync.py \
    --out $CF/scores/reference_overlap.json

# 任务 2 背景探针（先解一个 session 出来，见脚本 docstring）
$PY $CF/code/imu_background_probe.py <解出来的 session 目录>
```

---

## 六、本轮对主文档的核对结果

全部数字都拿磁盘复算过，**主文档的结论一处未被推翻**。逐项核对记录：

**对上了**（逐格一致）：第零节的逐动作 bundle 映射（对 90 个 `frozen_config.json`）、
第一节主 FAR 表（对 `../BOOTSTRAP_FAR5.md`，90 格逐格 0 处不符；逐格 CI 中位宽 0.0855、聚合 0.024）、
第二节对比表与配对差（对 `../RESULTS.md` + `../BOOTSTRAP_COMPARE.md`）、
第三节消融表（对 `../RESULTS.md`）、第四节三张表（对 `scores/session_aggregation.json`）、
第五节三行（对两个 `session_detector_*.json`）、第六节整表（对 `scores/style_distance.json`）、
第七节第 1 条（对 `scores/reference_overlap.json`）。

**独立复算的关键量**（逐事件重算，判据 `accepted ⟺ score < frr5 阈值`）：

| 量 | 主文档 | 本轮复算 |
|---|---|---|
| 发布版 FAR@frr5 均值，90 格 | 0.7746 | **0.774575** |
| r1 基线 FAR@frr5 均值，90 格 | 0.4447 | **0.444733** |
| imu_only 均值（r1 / 发布） | 0.8306 / 0.8350 | **0.830558 / 0.834983** |
| imu_trajectory_xytime（r1 / 发布） | 0.208 / 0.711 | **0.207592 / 0.711267** |
| tap/swipe/pinch 的 18 个 imu_only 格两树相同 | 是 | **18/18 完全相同** |
| `tap__imu_only__paper_xgboost` 的诱饵 far | 0.3817 | **0.38175**（`primary_metrics.far`，取在 dev-EER 阈值上；FAR@frr5 是 0.854） |
| 接受判据对全部 90 格的复算误差 | 0.000e+00 | **0.000e+00**（far 与 frr 都是） |
| `test_threshold_selection_calls` | 0 | **90 格全部为 0**；`test_opened_before_freeze` 全部为 false |

**需要更正的三处**（本 lane 只报不改，主文档不归本 lane）：

1. **第零节「90 格中有 16 格 `frr5 < eer`」属于错的那棵树。** 发布版是 **0 格**，r1 基线才是 16 格。
   这条本意是佐证「阈值是分数值不是错误率」，却拿了这一节正在警告读者不要用的那棵树的统计量。
2. **第五节「只有 6 段是物理上不可能的（负间隔、1.4e15 秒时钟跳变）」与产物口径不符。**
   `scores/session_rejects.json` 记的是 6 个**负间隔**与 74 个**超上限间隔**（都是间隔计数，不是会话计数），
   66 是被拒**会话**数；而 1.396e15 秒那一跳是正的，被算进「超上限」而不是那 6 个里。
3. **第三节 A9/A10/A11 的「CI 排除 0 = 是」在库内没有担保物。** `../BOOTSTRAP_COMPARE.md` 只有
   A7 与 A8 两行配对 bootstrap；A9/A10/A11 没有。同节「EER 8.4×」所需的 12 格 A3′ EER 也不在
   `../RESULTS_EER.md` 里（FAR 侧的 7.4× 可从 `../RESULTS.md` 复算出来，成立）。

**尚未写进主文档、但会限定第一节头条的一条**：`scores/competence_gate_classical.json` 的
`plain_statement` 写着——60 个经典格子里 38 个对**任何**一种简单攻击都到不了 AUC 0.80，
而这 38 个格子的 FAR@frr5 均值 0.7982、通过的 22 个是 0.7056，配对差 0.0925 [0.0793, 0.1066]。
主文档第十三节只把 competence gate 记成一条待办，没有报这个已经量出来的结果。

**路径与链接**：本目录 10 份 `.md` 里的 markdown 链接与绝对路径**逐条测过存在性，全部可解析**
（唯一一处扫描告警是 `SESSION_AGGREGATION_EN.md:137` 行文里把路径写成了省略形式——`/mnt/share` 后接省略号再接 `detectors_90cell/cells`——那不是一条真路径）。
唯一真正断掉的引用是 `SESSION_AGGREGATION_EN.md:246` 的「Files:」行——它列的
`reconciled.json`、`C/referee_repo.json`、`C/referee_mnt.json`、`C/referee_repo.log`、`C/referee_mnt.log`
在本仓库里都不存在；`referee.py` 本体已在 `code/session_aggregation/`，对账结果已在 `scores/session_aggregation.json`。

**两个「会话」不是同一个东西，主文档没有点破**：第四节的会话是 474 段（20 名测试用户，
中位 14 / 均值 18.97 个手势，取自发布版事件池），第五节的会话是 2328 段（测试 457，
均每段约 108 个手势，取自 HMOG 完整时间线）。两节的被抓率不可互相引用。

---

## 七、一条贯穿的发现

会话节律（间隔轴）可以靠复制人类间隔分布化解（v2 已证），说明真正的会话级威胁不在时间对齐，
而在**内容耦合与连续 IMU 背景**——与 `ASSESSMENT_ABC_CN.md` 里 issue C 对 joint 的批评指向同一处：
**时间同步 ≠ 内容耦合**。

**最容易被读错的一点**：会话节律实验（第五节）的五条臂共用同一批**真人**会话骨架，攻击臂只替换
**手势间隔**，**我们生成的假事件一条都没进去**。那些被抓率是「间隔时序」这一根轴上的假想攻击，
不是我们真正的攻击。见 `PLAN_AND_RESULTS_CN.md` 第五节第 4 条限定、`EVALUATION_CN.md` 开头口径警告 2、
`SPEC_CN.md` §0。
