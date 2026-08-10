# 阶段成果与未完成项（交接）

本文记录 final_check 这一轮的**已完成**与**未完成**，让下一次接手能直接续跑。全部不改动那 6 个冻结检测器（只读、仅作对照）。

> 说明：`final_check/` 是共享目录。本轮的 session/任务2/实验B/评审在此，另有一份并列的 `AUDIT_CN.md`（威胁模型与统计口径核查）+ `scores/reference_overlap.json` 是另一条线的审计，互不冲突。

## 一、已完成（阶段成果）

| 主题 | 结论 | 位置 |
|---|---|---|
| 会话节律检测器 | 未整形机器 100% 被抓；v1 pacing（log-normal，只对齐中位）72–100%；**v2 pacing（从真人经验分布抽，带长尾）压到真人误报地板** | `SPEC_CN.md`、`EVALUATION_CN.md`、`code/{assemble_sessions,session_detector}.py`、`scores/session_detector_*.json` |
| 任务 2：连续 IMU 轴 | **更正：能做**——数据在原始 HMOG（98 Hz 连续，`/home/mwang49/Human_agent/hmog_dataset.zip`）；手势之间 0% 真人 2s 窗死寂 → 比节律更强的判别轴 | `TASK2_IMU_AXIS_CN.md`、`code/imu_background_probe.py` |
| 实验 B：同人/异人距离 | keystroke/tap/scroll **保留目标风格**（D_fake<D_inter，user-clustered bootstrap CI 不含 0）；swipe 不确定；**pinch 不保留**（诚实负结果） | `EXPERIMENT_B_CN.md`、`code/style_distance.py`、`scores/style_distance.json` |
| A/B/C 评审 | 三个质疑都成立；C 指出 joint 只证明 event-alignment，不证明 physical coupling | `ASSESSMENT_ABC_CN.md` |

一条贯穿发现：会话节律（间隔轴）可靠复制人类间隔分布化解（v2 已证），说明真正的会话级威胁在**内容耦合 + 连续 IMU 背景**，不在时间对齐——与 issue C 对 joint 的批评同指一处。

## 二、未完成（接手指引）

### A. 深度检测器 competence gate（需 GPU）
- **要做**：造三种明显简单攻击（fixed injection / linear trajectory / replay+jitter）；对每个 action–modality–detector 配置在 **development set** 上算 AUC；预先规定 **dev AUC ≥ 0.8**（严格 0.9），未过标 training failure 单独报、不支撑 ActReal 高 FAR。两种检查：(a) 用简单攻击训练+测同架构验证学习能力；(b) 把 ActReal-trained 检测器直接测简单攻击看边界。
- **纪律**：规则冻结成 `gate_rules.json`，重跑前写死，统一应用，不看 ActReal FAR 后选择性删模型。
- **复用**：现有检测器运行器（`code/dataset_test/scripts/run_hmog_direct100k_detectors.py`，只读那 6 个）。
- **验收**：一张逐配置 AUC 表 + training-failure 清单。
- 详见 `ASSESSMENT_ABC_CN.md` A 节。

### C. joint 耦合扰动探针（需检测器推理）
- **要做**：对已通过 joint 的事件做四种扰动——IMU 时间平移、同动作内随机交换 IMU、左右方向交换、不同时长配对；看 joint 检测器是否还能分开。只有能识别这些扰动，"过 joint" 才等于物理耦合，否则只是分别判两个单模态。
- **前置事实（已核实）**：IMU 生成条件是动作+时长+设备方向+five-shot，**不含最终轨迹方向**——向左/向右 swipe 从几乎相同 IMU 分布采样。所以 FAR=0.711 只证明 event-aligned。
- **验收**：四种扰动各一条被识别率；再看 ActReal 扰动后是否仍通过。
- 详见 `ASSESSMENT_ABC_CN.md` C 节。

### 任务 2 的攻击侧连续背景流（需真机）
- 真人连续 IMU 已可从原始 HMOG 抽取；**唯一缺口**是攻击侧要有真机 Agent 跑出的连续 IMU 流才能公正评测（否则用我们自己的背景模型生成再测就是自证）。建议留到有真机 run 之后做。

### C 的论文措辞（需作者确认）
- 四处 tex 把 "touch-conditioned / learned physical coupling" 改成 **"event-aligned / time-aligned"**：`paper/sections/{threat_model.tex:95, introduction.tex:84, method.tex:8, evaluation.tex:53}`。**论文是作者的，未擅自改。**

## 三、v2 pacing 的部署侧改动（在 actreal_agent，非本仓库）
`actreal/pacing.py` 增加了 empirical_gaps 模式（从真人间隔分布抽、天花板不截长尾）。actreal_agent 目前不是 git 仓库，未版本化，记录于此以免遗漏。
