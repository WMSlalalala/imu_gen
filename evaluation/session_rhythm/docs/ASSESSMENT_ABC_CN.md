# 三个批评的评审：都成立，且都要改

对你提的三个（competence gate / 同人-异人距离 / joint 到底证明了什么）逐个核实后的判断。**三个都对。** 下面写清每个"对在哪、要改什么、怎么做"。

---

## A. 深度检测器 competence gate —— 成立，必须补，且要在重跑前冻结规则

**为什么对**：声称 ActReal 拿到高 FAR 之前，必须先证明检测器**有能力**——它连"明显的自动化"都抓得住。如果某个检测器连直线轨迹机器人都抓不住，那 ActReal 在它上面的高 FAR 毫无意义（是检测器坏了，不是被骗了）。这是标准的实验卫生，现在确实缺。

**要做什么**（按你说的，规则先冻结、统一应用、不看 ActReal FAR 后选择性删模型）：

1. 三种明显的简单攻击：`fixed injection`（恒定注入）、`linear trajectory`（直线轨迹）、`replay+jitter`（重放加抖动）。
2. 对每个 action–modality–detector 配置，在 **development set** 上算 AUC。
3. 预先规定门槛：**dev AUC ≥ 0.8**（更严 ≥ 0.9）。未过的配置标 **training failure**，单独报告，**不用于支撑 ActReal 的高 FAR**。
4. 两种检查缺一不可：
   - **(a) 学习能力**：用简单攻击训练+测试同一架构 → 证明实现和网络确实学得动（排除"代码写错/网络没学到"）。
   - **(b) 当前边界**：把**已经用 ActReal 训练**的检测器直接拿去测简单攻击 → 看当前决策边界连明显自动化都识别不了吗（若识别不了，说明它的边界被 ActReal 拉偏到只认某种特征）。

**放哪**：`evaluation/` 下新增 `competence_gate/`，与"同人/异人距离"**分开**（你强调过，二者回答不同问题）。规则冻结成一个 `gate_rules.json`（三种攻击定义 + AUC 门槛 + 划分），跑之前写死。

**要新做的量**：三种简单攻击的生成器（fixed/linear/replay-jitter）+ 逐配置 AUC 表。可复用现有检测器运行器（只读，不改那 6 个）。

---

## B. 同人 fake–real 与异人 real–real 的差距 —— 成立，且你把不等式方向纠对了

**为什么对**：这个实验有价值（"生成是否保留目标用户风格"），但不等式方向取决于用**距离**还是**相似度**，很容易写反。按距离 D：

- \(D_{\text{fake}}=D(\text{fake}_u,\text{real}_u)\)：假事件离目标用户本人多远
- \(D_{\text{inter}}=D(\text{real}_v,\text{real}_u),\ v\neq u\)：一个随机异人离目标用户多远
- 支持"fake 保留目标行为"的结果应是 **\(D_{\text{fake}} < D_{\text{inter}}\)**（假比陌生人更像本人）。
- 若 \(D_{\text{fake}} > D_{\text{inter}}\)，反而说明假事件比普通陌生人离本人还远——是坏结果。
- 只有用相似度 S 时才是 \(S_{\text{fake}} > S_{\text{inter}}\)。

**加真人内部距离**（你建议的，对）：\(D_{\text{intra}}=D(\text{real}^{(1)}_u,\text{real}^{(2)}_u)\)。理想链：
\[ D_{\text{intra}} \le D_{\text{fake}} < D_{\text{inter}} \]
即假事件落在"本人两次之间的差异"和"本人 vs 陌生人的差异"之间。

**协议**（按你说的）：按 **action / duration / direction / device orientation 匹配**；每名 test user 产一个差值 \(\Delta_u = D_{\text{inter},u} - D_{\text{fake},u}\)；报 **20 名 user 中 \(\Delta_u>0\) 的比例、中位数、user-clustered bootstrap CI**。

**边界**（你点明的，写进结论）：它回答"生成是否保留目标用户风格"，**不能替代 competence gate**。两个实验分开放、分开报。

**放哪**：`evaluation/` 下 `style_distance/`（与 `competence_gate/` 并列，互不混）。

---

## C. Joint 到底证明了什么 —— 成立，且这是最要紧的一处，牵涉论文措辞

**为什么对（我核实了代码）**：要区分两件事——

- **joint detector**：检测器同时看 touch 和 IMU。（当前实现属于这个）
- **joint generator**：IMU 根据**最终触摸几何**生成。（当前实现**不完全**属于这个）

核实：生成器 API 收 `xy_start/xy_end`、算了 `dx/dy`，但 IMU 的实际条件是"动作类型 + 时长 + 设备方向 + five-shot 参考"。**向左 swipe 600ms 和向右 swipe 600ms 会从几乎相同的 IMU 分布采样**，系统只是把这条 IMU 和各自的触摸放进相同时间区间。就是你那个类比：视频和音频同时开始同时结束，但音频内容未必对得上嘴型——**时间同步 ≠ 内容耦合**。

**所以 FAR=0.711 只能说明**：这 6 类 joint detector 未能可靠识别"当前事件对齐的 touch–IMU 组合"。
**不能说明**：IMU 已根据最终触摸的方向、距离、速度产生物理一致的响应。

**要改论文措辞**：把 "touch-conditioned / learned physical coupling" 改成 **"event-aligned" 或 "time-aligned"**。四处：

- `paper/sections/threat_model.tex:95`
- `paper/sections/introduction.tex:84`
- `paper/sections/method.tex:8`
- `paper/sections/evaluation.tex:53`

> 论文是你的，我没擅自改。上面四处我可以按"event-aligned"改好给你审，或你自己改。**要改说一声。**

**要证明物理耦合，得先加一组扰动实验**（确认 detector 到底能不能识别，否则它可能只是分别判断两个单模态各自像不像真人）：

1. IMU 时间平移
2. 同动作内随机交换 IMU
3. 左右方向交换
4. 不同持续时间配对

对每种扰动，看 joint detector 是否还能分开。**只有 detector 能识别这些扰动，"通过 joint" 才等于"物理耦合"**；然后再看 ActReal 是否仍能通过。这组扰动实验和上面的 session-detector 发现指向同一件事：真正的联合信号在**内容耦合**，不在时间对齐。

**放哪**：`evaluation/` 下 `joint_coupling_probe/`。

---

## 优先级建议

| 项 | 类型 | 依赖 | 我可以现在做 |
|---|---|---|---|
| C 的论文措辞 | 改 4 行 tex | 需你点头（论文是你的） | 待确认 |
| C 的扰动探针 | 新实验 | 只读现有 6 检测器 + 造 4 种扰动 | 能做 |
| A competence gate | 新实验 | 造 3 种简单攻击 + 逐配置 AUC | 能做 |
| B style distance | 新实验 | 定距离度量 + 匹配 + bootstrap | 能做 |

三个都不碰那 6 个冻结检测器（只读、只用来打分/对照）。A 和 C 的扰动探针共享"检测器只读运行器"，可一起做。
