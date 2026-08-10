# 定稿前核查：威胁模型与统计口径

对两个方法学问题的核查：**触摸与惯性是否共用同一组五条参考**，以及 **主判据有没有置信区间**。做法是三路独立取证，每路结论再交给一个只负责**反驳**它的检查者。本文记录经得起反驳的、被推翻的、以及仍判不了的。

每条都给出可自行重跑的命令。没查清的写"未测量"，不推断。

## 速览

| # | 事项 | 判定 | 处置 | Issue |
|---|---|---|---|---|
| 1 | 触摸与惯性是否共用同一组五条参考 | **不是**（四个手势动作） | **重跑**（改表述已否决） | [#1](https://github.com/WMSlalalala/imu_gen/issues/1) |
| 2 | 主判据 FAR 有没有置信区间 | 之前没有 | 已补 | — |
| 3 | 是否跑过多折 / 换划分 | 没有，只有一份划分 | 待补 | [#4](https://github.com/WMSlalalala/imu_gen/issues/4) |
| 4 | 「imu_only 在 3/5 动作上没测 IMU 攻击」 | **不成立**，虚警 | 不用动 | — |
| 5 | 扩散先验来自哪个用户 | 来自**别的**用户 | 待查规则 | [#3](https://github.com/WMSlalalala/imu_gen/issues/3) |
| 6 | 联合模态偏低是否源于参考错配 | **判不了**，缺对照 | 建双通道对照 | [#2](https://github.com/WMSlalalala/imu_gen/issues/2) |

---

## 1. 触摸与惯性各抽各的五条（论文级问题）

**在 tap / scroll / swipe / pinch 上，两个通道各自独立抽了五条真人录制，两组几乎不重叠。攻击者每个「受害者 × 动作」实际消耗的真实录制最多是 10 条，不是 5 条。**

400 个「用户 × 动作」组合里 **354 组（88.5%）两组完全不相交**：

| 动作 | 组数 | 完全不相交 | 交集分布 | 并集均值 |
|---|---|---|---|---|
| tap | 100 | 64 | {0:64, 1:26, 2:10} | 9.54 |
| scroll | 100 | 97 | {0:97, 1:3} | 9.97 |
| swipe | 100 | 97 | {0:97, 1:3} | 9.97 |
| pinch | 100 | 96 | {0:96, 1:3, 2:1} | 9.95 |

少数组重合 1–2 条，是两套独立抽样偶然撞上，不是设计上的绑定。

**keystroke 是唯一例外，它确实是真 five-shot。** 它的惯性通道由解析式适配器写出（不走扩散模型），适配器拿到的就是触摸侧那五条：发布版 **20,000 / 20,000** 条 keystroke 伪造事件的 `donor.material_source_event_ids` 与 `donor.imu.source_event_ids` 逐条同序完全相同。

### 为什么两边不可能对齐

两套算法、两个种子、两个候选池，没有任何共享状态：

```
惯性侧   UserRefBank（final_gen/data.py）
         np.random.default_rng(345 + user*1009).permutation(该用户全部行)[:5]
         种子 345 = 42 + 303，在生成器里写死为 EXPECTED_REF_BANK_SEED

触摸侧   冻结素材（freeze_hmog_fiveshot_material.py）
         按 sha256("<domain>|42|<event_id>") 升序取前五
```

而且触摸素材是在惯性缓存**之后**才冻结的——它的 `input_manifest` 指向 `detector_dataset_100k`，那里面已经含扩散生成的 IMU——因果上就不可能与之匹配。

### 影响范围

90 格里 **72 格**（四个手势 × 3 模态 × 6 检测器）建立在「10-shot」之上；三个模态的汇总 FAR（轨迹 0.777 / 惯性 0.835 / 联合 0.711）都含这 72 格。keystroke 的 18 格不受影响。

需要说清楚的是：**这不影响任何一个数字的正确性**，它们都是如实测出来的。受影响的是这些数字挂在什么标题下——「five-shot 攻击」这个说法在 4/5 的动作上与实际不符。

### 处置：重跑，不改表述

曾考虑过「改表述」——写成「触摸 5 条 + 惯性 5 条，共 10 条」，不重跑任何实验。**这条已被否决。**

要做的是真正实现 five-shot：把 `UserRefBank` 的候选限制到 `fiveshot_material` 那五条，再重跑扩散缓存（100 用户 × 4 动作 × 200 条）→ 事件合成 → 建库 → 72 格。按本项目实测速率约数天。keystroke 不用动。

跟踪见 [#1](https://github.com/WMSlalalala/imu_gen/issues/1)。

### 自己验

```bash
python ../comparison/code/check_reference_sync.py --out /tmp/refcheck.json
```

脚本先**复算惯性侧的选择规则**并与缓存实录比对，对不上直接中止——否则后面的比较没有意义。通过后再把两侧映射到同一个 HMOG event id 空间：惯性侧 `used_ref_indices` → processed 行 → `event_id`；触摸侧 `source_cluster_id` → `genuine_bindings.jsonl` → `source_event_id`。输出见 [`scores/reference_overlap.json`](scores/reference_overlap.json)。

---

## 2. 主判据的置信区间（已补）

发布版每个格子本来就带 `bootstrap_95ci`，重抽单位 `user_cluster`、10,000 次——规格没问题。**问题是它括的是 `primary_metrics.far`，取在开发集选定的 EER 阈值上**，而论文报的 0.775 在 FRR=5% 切点上。唯一存在的区间恰好不在主判据上。

现已按 FRR=5% 切点补齐（[`../BOOTSTRAP_FAR5.md`](../BOOTSTRAP_FAR5.md)、[`../BOOTSTRAP_COMPARE.md`](../BOOTSTRAP_COMPARE.md)）：

```
FAR 全 90 格   0.775  [0.763, 0.787]
FRR 全 90 格   0.052  [0.046, 0.059]     ← 盖住 5% 目标
惯性 0.835 [0.817, 0.852]  ⎫ 两区间不重叠
联合 0.711 [0.696, 0.726]  ⎭
```

重抽单位必须是用户：一个测试格有 6,000 条事件，但只有 20 个人、每人 300 条，同一个人的事件共享习惯、设备、握持与少数几段 session。按事件重抽会让区间按簇大小的平方根缩小，宣称数据不支持的精度。

还有一个容易做错的细节：**每次抽出的那组人在全部 90 格上共用**。90 格考的是同一批人；各格独立抽人的话，逐格区间仍对，但它们的平均会表现得像各格互相独立，聚合区间会窄得离谱。

配对比较（同一组人同时给两边打分，区间落在差值上）里**唯一不排除 0 的是 A5**（k_refs=3）：`+0.004 [−0.003, +0.011]`。这是 k_refs 曲线在 3 处饱和的统计确认；其区间半宽约 0.007，与用 A1/A5/A6 离散度估出的噪声量级 0.005 相符，两条独立路径互为佐证。

**一处更正**：核查中发现冻结发布树里**本来就有**能在 frr5 切点做 bootstrap 的实现（`code/dataset_test/security_exp/hmog_direct100k_test.py` 的 `_user_bootstrap` 同时接收 `eer_threshold` 与 `target_threshold`）。准确的说法是「能力早就有，只是发表的那份数字没用它」，不是「这条链路上没有 bootstrap」。

---

## 3. 没有多折，也没有换种子重复

用户划分只有一份：`users_seed42.json`，用户不相交 70/10/20，五个动作共用，90 格的 `frozen_config.json` 全部 `seed=42`。评测代码树里没有任何 k 折或换划分的实现。

**bootstrap 不能替代换划分**：前者在这一份测试划分之内重抽 20 名测试用户，答的是"换 20 个同池子的人这个数会怎么动"；后者会改变生成器与检测器各自训在谁身上，需要把 90 个检测器连同上游生成器全部重训。两者量的是不同的变异来源。

---

## 4. 一次虚警：惯性通道的 0.835 是成立的

核查中曾出现一条指控——「tap/pinch/swipe 的 `imu_only` 用的 manifest 里伪造 IMU 与真人相同，0.835 在 3/5 动作上根本没在测 IMU 攻击，必须重跑格子」。

**这条不成立，已验伪。** 取发布版一条 tap 伪造事件的惯性窗口，与它 `imu_source` 指向的扩散缓存 `user_cache_eval_200/user_000/tap/train/sample_0000.npz` 逐位比对，**最大差 0**；同一分片前 400 条真人事件里没有任何一条与之相同。

误判来源是把 `input_imu_sha256 == output_imu_sha256` 读成了「没有 IMU 攻击」。正确含义是「惯性通道由上游扩散生成，触摸重建这一阶段原样透传」。

记在这里是因为这个误读很容易再犯：**这个字段只说明某个阶段没改动，不说明通道内容是什么。**

---

## 5. 扩散先验来自别的用户

抽查 20 组（5 用户 × 4 手势动作），**19 组的 `prior_audit.prior_source_user_id` 不等于目标用户**：user000 的 tap 先验来自 user007、scroll 来自 user046、swipe 来自 user097、pinch 来自 user084。

这大概率是设计如此——先验只提供起点，个性化由那五条参考完成——但文中应当交代一句，否则读者无法判断"到底用了目标用户的什么"。**先验取自哪个用户、依据什么规则选，本次未查清，标为未测量。**

---

## 6. 联合模态偏低：判不了

联合 0.711 是三个模态里最低的（轨迹 0.777、惯性 0.835）。自然的假说是：既然两通道的参考不是同一组（§1），跨模态一致性就是坏的，而联合检测器恰好能抓这个。

**这个假说既没被证实也没被证伪。**

支持它的：deep 族检测器第一层就是跨全部 15 通道的卷积（`event_detectors.py` 里的 `nn.Conv1d(input_channels, ...)`），是最有能力利用跨模态不一致的一族，而它掉得最多（−0.070）。

不足以判定的三条：

- 唯一能判别的对照**没跑过**——`control_genuine` 只有 `imu_only` 的 30 格，没有联合格子。
- **补跑它也判别不了**：`control_genuine` 只替换惯性通道、保留发布版的合成触摸，等于复刻现状。真正能判别的是「真人惯性 + 真人触摸」的双通道真人对照。
- 用 `diffts_both` 做旁证也不行：按 `covered_modalities.py` 的规则，只有两通道都替换的方法才上报联合格子，所以整棵评测树里的联合格子全部来自双通道方法——等于拿同一种配置给自己做对照。

要判别就得建那个双通道真人对照并跑 30 格。在此之前，联合模态偏低的原因**未确定**。

---

## 产物

- [`scores/reference_overlap.json`](scores/reference_overlap.json) —— §1 的全量比对输出，含选择规则的自校验结果
- 脚本在 [`../comparison/code/`](../comparison/code/)：`check_reference_sync.py`（§1）、`bootstrap_far5.py`（§2）

---

## 未决事项都已登记为 issue

| Issue | 事项 | 标签 |
|---|---|---|
| [#1](https://github.com/WMSlalalala/imu_gen/issues/1) | 四个手势动作实际是 10-shot，不是 five-shot | `threat-model` `blocking-paper` |
| [#2](https://github.com/WMSlalalala/imu_gen/issues/2) | 联合模态偏低的原因判不了，缺双通道真人对照 | `undetermined` |
| [#3](https://github.com/WMSlalalala/imu_gen/issues/3) | 扩散先验来自别的用户，规则未查清 | `threat-model` `documentation` |
| [#4](https://github.com/WMSlalalala/imu_gen/issues/4) | 只有一份用户划分，没有 k 折 | `statistics` |
| [#5](https://github.com/WMSlalalala/imu_gen/issues/5) | 生成器重训的运行间方差从未测量 | `statistics` |
| [#6](https://github.com/WMSlalalala/imu_gen/issues/6) | EER 表没有置信区间 | `statistics` |
| [#7](https://github.com/WMSlalalala/imu_gen/issues/7) | 第三方基线用的是早停停止点，基线被低估 | `baseline-fairness` |
| [#8](https://github.com/WMSlalalala/imu_gen/issues/8) | gitignore 的 `**/results/` 会静默丢文件 | `reproducibility` |

本文只登记核查结果；进展在 issue 里跟。
