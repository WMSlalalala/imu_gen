# final_check —— 目录地图

审稿式补充调研：对几处方法与评测的质疑，逐个用实验或数据分析核实。**全程不改动那 6 个冻结检测器**（只读、仅作对照）。数据在库外，本目录只装代码、小结果表与说明。

**冷启动请先读 [`PENDING_CN.md`](PENDING_CN.md)** —— 那份是状态入口（现在是什么数字、别的 lane 在跑什么、什么没做）。本份只回答「哪个文件是干什么的」。

本目录路径：`/mnt/share/mwang49/real-human/imu_gen/final/evaluation/final_check/`

```
PENDING_CN.md        状态入口：数字口径、在跑的、没做的、文档不一致清单
README.md            本文件：目录地图
ASSESSMENT_ABC_CN.md A/B/C 三个批评的评审
AUDIT_CN.md          并列的另一条线：威胁模型与统计口径核查
SPEC_CN.md           会话节律检测器规格
EVALUATION_CN.md     会话节律检测器结果
TASK2_IMU_AXIS_CN.md 任务 2：连续 IMU 轴
EXPERIMENT_B_CN.md   实验 B：同人/异人距离
gate_rules.json      competence gate 的冻结规则
code/                5 个脚本
scores/              5 个结果 JSON
```

---

## 一、文档：各自回答什么

| 文档 | 回答什么 | 状态 |
|---|---|---|
| [`PENDING_CN.md`](PENDING_CN.md) | 现在到哪了：会话检测器 2026-08-10 重跑的全部数字与三条限定、其他 lane 在跑的四项、未完成四项、文档不一致清单 | **最新，以它为准** |
| [`ASSESSMENT_ABC_CN.md`](ASSESSMENT_ABC_CN.md) | A（competence gate）/ B（同人-异人距离不等式方向）/ C（joint 到底证明了什么）三个批评的评审——结论是三个都成立 | 结论有效；里面的 4 个 tex 行号已失效，见 `PENDING_CN.md` §4.4；建议的 `joint_coupling_probe/` 已被半攻击消融吸收，别再建 |
| [`AUDIT_CN.md`](AUDIT_CN.md) | 另一条线：触摸与惯性是否共用同一组五条参考（**不是**，四个手势动作实际 10-shot）、主判据有无置信区间（已补）、多折、扩散先验来源、联合模态偏低 | 有效；6 项判定 + 8 个 issue |
| [`SPEC_CN.md`](SPEC_CN.md) | 会话节律检测器的规格与纪律（用户不相交划分、开发集选阈值、paced 从不进训练） | **已于 2026-08-10 08:00 与实现对齐**：新增第 0 节明写「不是滑动窗」、「生成的假事件不进这个实验」；原第 12 行的「滑动窗 W、步长 s」已删除，改为一会话一特征向量一分数、W 只是占用箱宽 |
| [`EVALUATION_CN.md`](EVALUATION_CN.md) | 会话节律检测器的结果与读法 | **已于 2026-08-10 08:00 按重跑重写**：两张结果表逐格与 `scores/` 的两个 JSON 一致，与 `PENDING_CN.md` §1 相同 |
| [`TASK2_IMU_AXIS_CN.md`](TASK2_IMU_AXIS_CN.md) | 连续 IMU 轴能不能做（更正：能，数据在原始 HMOG 归档）、为什么它比节律轴强、还缺什么 | 有效 |
| [`EXPERIMENT_B_CN.md`](EXPERIMENT_B_CN.md) | 生成是否保留目标用户风格（keystroke/tap/scroll 保留，swipe 不确定，pinch 不保留） | **数字已过期**：`scores/style_distance.json` 2026-08-10 08:20 被重跑覆盖，本文表还是 8-09 那版；scroll 的判定会翻转（新 CI95 含 0）。见 `PENDING_CN.md` §6.7 |

---

## 二、`scores/` —— 哪个 JSON 支撑哪个结论

| 文件 | 支撑什么 | 由谁生成 | 状态 |
|---|---|---|---|
| `scores/session_detector_results.json` | 会话检测器 **RF** 全部被抓率与 CI95（`PENDING_CN.md` §1.2） | `code/session_detector.py --model rf` | **最新（2026-08-10 重跑）**；键名 `occupancy_bin_widths_s`；带 `not_a_sliding_window` 字段 |
| `scores/session_detector_logreg.json` | 会话检测器 **LogReg** 全部被抓率与 CI95（`PENDING_CN.md` §1.3） | `code/session_detector.py --model logreg` | 同上 |
| `scores/style_distance.json` | 实验 B 全部数字：逐动作 D_intra / D_fake / D_inter、Δ>0 比例、中位 Δ、user-clustered bootstrap CI95，另含 20 名 test user 的 `per_user` 明细 | `code/style_distance.py` | **2026-08-10 08:20 被重跑覆盖**（新增 `metric_fitted_on: "70 train users, genuine events only"`）。`EXPERIMENT_B_CN.md` 的表还是 8-09 那版、**与本文件已不一致**，且 scroll 的判定会翻转（新 CI95 `[-0.115, 0.614]` 含 0）。见 `PENDING_CN.md` §6.7 |
| `scores/reference_overlap.json` | AUDIT §1：400 个「用户×动作」里 354 组两侧五条完全不相交；keystroke 20,000/20,000 两通道同源 | `../comparison/code/check_reference_sync.py` | 有效 |
| `scores/assemble_summary.json` | 会话装配汇总（会话数、各臂间隔中位） | `code/assemble_sessions.py` | **过期**：写着 2394 / 中位 1.099，且没有拒收字段。最新一版是 `/mnt/share/mwang49/data7/session_rhythm_detector/results/assemble_summary.json`（2328 / 1.084） |

**不在 `scores/` 里、但属于同一批产物的**（都在 `/mnt/share/mwang49/data7/session_rhythm_detector/results/`）：

- `session_rejects.json` —— 66 个被拒会话的明细（负间隔 6、超 120 s 上限 74、最坏间隔 1.396e15 秒）。`PENDING_CN.md` §1.1 的表就出自这里。
- `sessions_{genuine,naive,naive_jitter,paced,paced_emp}.jsonl` —— 五条臂的会话流，每份 9.9–11 MB，可由 `assemble_sessions.py` 再生。每行含 `user/session/actions/durations_s/gaps_s/source/event_ids`；`gaps_s[0]` 是占位的 0.0（与 `actions` 对齐），检测器只用 `gaps_s[1:]`，所以「间隔总数」是 251,333 而不是 253,661。
- `assemble_summary.json` —— 见上。

> `scores/` 下没有 `sessions_*.jsonl`——这是事实，不是笔误；`EVALUATION_CN.md` 的复现段写的是 `$SESSION_DIR/sessions_*.jsonl`，指向上面这个目录。

**`gate_rules.json`**（在本目录根，不在 `scores/`）：competence gate 的冻结规则，`frozen_at: 2026-08-10`。它不是结果，是**跑之前写死的判据**——三种简单攻击的定义、dev split、90 格、AUC ≥ 0.80 通过 / 0.90 严格、`training_failure` 的报告方式、禁止事后按 FAR 删格子。里面的 `motivation_measured_before_freezing`（AUC < 0.55 有 10 格，< 0.60 有 31 格）是**建门的理由**，不是门的输出。

---

## 三、`code/` —— 哪个脚本重生成什么

| 脚本 | 生成什么 | 备注 |
|---|---|---|
| `code/assemble_sessions.py` | 五条臂的 `sessions_*.jsonl` + `assemble_summary.json` + `session_rejects.json` | 真人会话从 `trajectories_full_v2/hmog_trajectory_<action>.npz` 的 `flat_system_time_ms` 重建；五臂共用同一批真人骨架，只换间隔。会话若有**任何**负间隔或超 `MAX_GAP_S = 120.0` 的间隔就整条拒收，不裁剪。需要 `actreal.pacing.DelayPolicy`（`paced` 臂）。 |
| `code/session_detector.py` | `session_detector_results.json` / `session_detector_logreg.json` | 一个会话一个分数；`--bins` 是**占用率箱宽**不是滑窗（旧别名 `--windows` 仍可用）。训练只用 genuine vs `naive_jitter`，`paced` 从不进训练。阈值在 val 真人上取 FRR=5% 与 1%。 |
| `code/style_distance.py` | `style_distance.json` | 读 `/mnt/share/mwang49/data7/direct100k_final/datasets`；逐事件 45 维特征（trajectory + IMU 每通道均值/std/min/max + 帧数），按动作标准化；user-clustered bootstrap 10,000 次。 |
| `code/imu_background_probe.py` | 无落盘结果，直接打印 | 任务 2 的证据：从原始 HMOG 一个 session 目录读 `Accelerometer.csv` / `Gyroscope.csv` / `TouchEvent.csv`，量手势之间的 IMU 背景。用法在脚本 docstring 里（要先从 6.1 GB 归档解出一个 session）。 |
| `code/fiveshot_priority.py` | 无独立产物，是**库** | 把已发布的触摸侧五条（`material_manifest.jsonl`）经 `genuine_bindings.jsonl` 映射成惯性生成器索引的行号，供 `UserRefBank` 排序，保证 k=1 ⊂ k=3 ⊂ k=5。这是 AUDIT §1 / issue #1 重跑的前置件，**已写好、尚未驱动重跑**。 |

AUDIT 那条线的脚本**不在本目录**，在 `/mnt/share/mwang49/real-human/imu_gen/final/evaluation/comparison/code/`：`check_reference_sync.py`（§1）、`bootstrap_far5.py`（§2）。

---

## 四、复现命令

会话链路（装配 → 打分）。**`session_detector.py` 的 `RESULTS` 默认指向一个临时 scratchpad 路径，必须用 `SESSION_DIR` 覆盖**，否则读不到刚装配出来的五臂 jsonl：

```bash
PY=/home/mwang49/miniconda3/envs/cuhkx/bin/python
CF=/mnt/share/mwang49/real-human/imu_gen/final/evaluation/final_check
OUT=/mnt/share/mwang49/data7/session_rhythm_detector/results

$PY $CF/code/assemble_sessions.py --out $OUT

SESSION_DIR=$OUT $PY $CF/code/session_detector.py --model rf     --bins 2,3,5 \
    --out $CF/scores/session_detector_results.json
SESSION_DIR=$OUT $PY $CF/code/session_detector.py --model logreg --bins 2,3,5 \
    --out $CF/scores/session_detector_logreg.json
```

实验 B：

```bash
$PY $CF/code/style_distance.py --out $CF/scores/style_distance.json
```

AUDIT §1：

```bash
$PY /mnt/share/mwang49/real-human/imu_gen/final/evaluation/comparison/code/check_reference_sync.py \
    --out $CF/scores/reference_overlap.json
```

任务 2 的背景探针（先解一个 session 出来，见脚本 docstring）：

```bash
$PY $CF/code/imu_background_probe.py <解出来的 session 目录>
```

---

## 五、库外路径（本文写到的都已核实存在）

| 路径 | 是什么 |
|---|---|
| `/mnt/share/mwang49/data7/session_rhythm_detector/results/` | 会话链路的落盘目录：五臂 jsonl、`assemble_summary.json`、`session_rejects.json`、`style_distance.json` |
| `/mnt/share/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713/results/trajectories_full_v2/` | 真人会话时间线的来源（`hmog_trajectory_<action>.npz`，含 `flat_system_time_ms`、`session_id`、`event_id`） |
| `/mnt/share/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json` | 唯一的用户划分，70 / 10 / 20，五个动作共用 |
| `/mnt/share/mwang49/data7/direct100k_final/datasets/` | 发布版逐事件数据集（4 个 bundle：keystroke / scroll / swipe / tap_and_pinch） |
| `/mnt/share/mwang49/data7/direct100k_final/detector_models/` | 90 个冻结检测器（每格一个 `model.joblib`），**只读** |
| `/mnt/share/mwang49/data7/direct100k_final/code/dataset_test/scripts/run_hmog_direct100k_detectors.py` | 只读检测器运行器（`ASSESSMENT_ABC_CN.md`/旧 PENDING 里那条相对路径 `code/dataset_test/...` 从本目录解不出来，用这条绝对路径） |
| `/mnt/share/mwang49/real-human/imu_gen/final/evaluation/comparison/code/` | AUDIT 线的脚本 |
| `/mnt/share/mwang49/real-human/imu_gen/final/evaluation/BOOTSTRAP_FAR5.md`、`BOOTSTRAP_COMPARE.md` | AUDIT §2 的 FRR=5% 切点 bootstrap 产物 |
| `/home/mwang49/Human_agent/hmog_dataset.zip` | 原始 HMOG 归档（6.1 GB），任务 2 连续 IMU 的唯一来源 |
| `/mnt/share/mwang49/data7/actreal_agent/actreal/pacing.py` | v2 pacing 的 `empirical_gaps` 模式在这里。**该目录不是 git 仓库**（2026-08-10 复核），改动无版本记录 |

**没有的东西**（两个已确认的阻塞项，详见 `PENDING_CN.md` §3）：

- **`dev_scores.jsonl` 不存在**（整个 `/mnt/share/mwang49` 下都没有）。要做分数空间的会话聚合，train/dev 的真人事件必须用冻结的 `model.joblib` 重新打一遍分。现存的逐事件分数只有一格一份：`/mnt/share/mwang49/data7/verification/tap__imu_trajectory_xytime__authconformer/test_scores.jsonl`，而且只有 test split。
- ~~**`assemble_sessions.py` 不逐槽输出 `event_id`**~~ —— **已补上（2026-08-10 08:12 改脚本、08:15 重跑）**：会话 dict 现在多一个 `event_ids`，与 `actions`/`durations_s`/`gaps_s` 逐槽对齐，五份 jsonl 都已重生成。脚本注释里给了到 `test_scores.jsonl` 的连接方式（`genuine_bindings.jsonl` → `source_cluster_id`）。**注意**：`scores/` 下那两个检测器 JSON 是 07:47/07:48 跑的，早于这次重装配；间隔统计逐项复算一致（只多了 `event_ids` 字段），但没人重跑过检测器来确认。

---

## 六、一条贯穿的发现

会话节律（间隔轴）可以靠复制人类间隔分布化解（v2 已证），说明真正的会话级威胁不在时间对齐，而在**内容耦合与连续 IMU 背景**——与实验 A/C 想查的、以及 issue C 对 joint 的批评指向同一处：**时间同步 ≠ 内容耦合**。

需要提醒的是 v2 那个结果近乎同义反复：从人类间隔分布抽出来的东西，对一个只读间隔统计量的检测器当然难。它证明「这条轴可被化解」，**不证明**「会话级威胁已解决」。

**还有一条更容易被读错的**：会话实验的五条臂共用同一批**真人**会话骨架，攻击臂只替换**手势间隔**，**我们生成的假事件一条都没进去**。所以那些被抓率是「间隔时序」这一根轴上的假想攻击，不是我们真正的攻击。见 `PENDING_CN.md` §1.4 第 0 条、`EVALUATION_CN.md` 开头的口径警告 2、`SPEC_CN.md` §0。
