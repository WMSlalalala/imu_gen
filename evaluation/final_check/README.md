# final_check：审稿式补充调研

针对几处方法与评测的质疑，逐个用实验或数据分析核实。全部**不改动那 6 个冻结检测器**（只读、仅作对照）；数据在库外，本目录只装代码、小结果表与说明。

## 已完成

| 主题 | 结论（一句话） | 文档 |
|---|---|---|
| 会话节律检测器 | 未整形机器 100% 被抓；v1 pacing 72–100%；**v2 pacing（经验分布，带长尾）压到真人误报地板** | SPEC_CN.md, EVALUATION_CN.md |
| 任务 2：连续 IMU 轴 | **更正："能做"，数据在原始 HMOG（98 Hz 连续）**；手势之间 0% 真人窗死寂 → 比节律更强的判别信号；缺口只在攻击侧连续背景流 | TASK2_IMU_AXIS_CN.md |
| 实验 B：同人/异人距离 | keystroke/tap/scroll 保留目标风格（D_fake<D_inter，CI 不含 0）；swipe 不确定；**pinch 不保留**（诚实负结果） | EXPERIMENT_B_CN.md |
| A/B/C 评审 | 三个质疑都成立；C 牵涉论文措辞（event-aligned ≠ physical coupling） | ASSESSMENT_ABC_CN.md |

## 待完成（需检测器基础设施，GPU）

- **A 深度检测器 competence gate**：造 3 种简单攻击（fixed/linear/replay+jitter）+ 逐配置 dev AUC，规则重跑前冻结。见 ASSESSMENT_ABC_CN.md A 节。
- **C joint 耦合扰动探针**：对 IMU 做 4 种扰动（时间平移/同动作换/左右交换/时长错配），看 joint 检测器认不认。见 ASSESSMENT_ABC_CN.md C 节。
- **论文措辞 C**：4 处 tex 从 "physical coupling" 改 "event-aligned"，论文是作者的、未擅自改，待确认。

## 一条贯穿的发现

会话节律（间隔轴）可靠复制人类间隔分布化解（v2），说明真正的会话级威胁不在时间对齐，而在**内容耦合与连续 IMU 背景**——与实验 A/C 想查的、以及 issue C 对 joint 的批评指向同一处：时间同步 ≠ 内容耦合。

## 复现

每个文档末尾都有复现命令。数据不在库里（原始 HMOG 6.1 GB、发布版 6.5 GB、会话流 21 MB 可再生），但代码、判据、小结果表都在，足以核对每个数字。
