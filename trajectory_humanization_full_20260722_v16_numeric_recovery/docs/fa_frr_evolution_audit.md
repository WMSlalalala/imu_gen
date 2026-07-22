# IMU 生成、Android 封装与 FA/FRR 结果演进审计

更新时间：2026-07-13

## 0. 文档目的

本文回答三个问题：

1. 为什么最开始的 diffusion 在完整 PAD 测试中还有中等 FA，换成 Android 缓存封装后却几乎被检测器 100% 识别；
2. 中间具体尝试过哪些修复或新方法，各自结果怎样；
3. 目前哪些数字可以作为正式结论，哪些只是 gate、诊断、selector 或非神经 train-prior 结果，不能混在一起比较。

本文不启动新训练，也不把尚未完成的 trajectory 版本写成已完成结果。所有表格均以现有落盘报告、score summary 和审计文件为依据。

---

## 1. 最终结论

### 1.1 Android 封装后的崩溃不是原 diffusion checkpoint 突然失效

第一次 Android cache 版本把 `raw_ddpm` checkpoint 当成 residual-level 模型使用：

1. 模型输出已经由 raw normalizer 反归一化回带重力的原始 IMU；
2. wrapper 又额外加了一次 `level_raw`；
3. few-shot refs 同时又被按 residual 方式处理。

这相当于把重力/DC level 加了两次。以 tap 为例：

| 数据 | 加速度三轴均值（前三维） | 重力模长中位数 |
| --- | --- | ---: |
| real test | `[-0.398, 5.277, 7.510]` | 9.621 |
| 错误 Android fake train | `[0.209, 10.178, 14.284]` | 17.813 |

这种偏移不需要复杂 detector：只看窗口均值或重力模长就能区分。因此错误封装版得到 EER FA 约 0、AUC 约 1 是合理结果，但它反映的是 wrapper representation bug，不是 diffusion 本身的真实生成能力。

### 1.2 修复 wrapper 后，完整 Android v2 基本恢复早期 baseline

在同一类 raw-DDPM checkpoint 上修复表示、five-shot refs、orientation、pinch XY 和 train-only metadata prior 后，100,000 条完整 Android v2 的 IMU-only 宏平均为：

| 正式实验 | EER FA | EER FRR | AUC | FA@val-FRR≤5% | test FRR@该阈值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 早期完整 raw-DDPM baseline | 0.2658 | 0.1859 | 0.8547 | 0.5826 | 0.0403 |
| 修复后的 Android cache v2 | 0.2626 | 0.1994 | 0.8464 | 0.5922 | 0.0448 |
| v2 − baseline | -0.0032 | +0.0135 | -0.0083 | +0.0096 | +0.0045 |

两个结果处在同一水平。这是当前最强的因果证据：第一次封装失败来自接口和数据协议错误；正确封装不会天然摧毁 diffusion 的 FA/FRR。

### 1.3 目前可引用的神经生成正式结果

若要求：

- 真正由 diffusion 采样；
- 不进行候选后筛选（即使 selector 只在 train 上拟合，它仍属于额外后处理）；
- 不复制真实训练窗口作为最终 fake；
- fake 用户固定 70/10/20；
- real test 使用全部 100 个用户的 held-out real；
- 五个 action、每 action 9 个 detector、不删结果；

则当前最完整、最适合引用的是修复后的 Android cache v2。早期 baseline 可作为独立复现实验对照。

### 1.4 当前 trajectory-humanization 项目还没有正式 FA/FRR

`trajectory_humanization_full_20260713` 当前已有完整数据抽取、协议代码、smoke/gate 和静态审计，但尚未完成：

- 五个 action 的正式 100-epoch best-EMA checkpoint；
- 100,000 条正式 neural fake archive；
- 25 个 formal detector pair 及最终合并审计。

目录中的 `best_smoke_step_0020.pt`、1–2 epoch detector checkpoint 和 smoke summary 只证明管线能执行，不能当成正式模型或正式 FA/FRR。本文不为该版本填写任何正式性能数字。

---

## 2. 指标与阈值到底表示什么

统一 score 方向为：

- score 越大，越像 fake；
- `score < threshold` 时，被认证系统接受为 real。

### 2.1 FA、FRR 与 AUC

```text
FA  = 被当作 real 接受的 fake 数 / 全部 fake 数
FRR = 被错误拒绝的 real 数 / 全部 real 数
```

对生成器而言：

- FA 越高越好；
- FRR 是认证系统在真实数据上的使用代价，不是生成器“成功率”的补数；
- AUC 使用 fake-high 方向，越接近 0.5 表示 real/fake 越难排序区分，越接近 1 表示越容易检测 fake。

所以 `FA` 不等于 `1 - real accuracy`。FA 的分母是 fake，real accuracy/FRR 的分母是 real，二者属于不同类别。

### 2.2 “EER FA/FRR”为什么在 test 上不一定相等

EER threshold 是在 validation real/fake 上选择的，使 validation FAR 与 FRR 尽量接近；然后把这个固定 threshold 应用到 test。

因此：

- validation 上 FAR≈FRR；
- test 分布不同，test FA 与 test FRR 不需要相等；
- 不能在 test 上重新调 threshold，否则会把 test 信息泄漏回模型选择。

### 2.3 为什么 FA@val-FRR≤5% 很高，但 AUC 仍然较好

AUC 衡量的是所有 threshold 上的整体排序能力；`FA@val-FRR≤5%` 只看一个固定 operating point。

为了让 validation real 的 FRR 不超过 5%，threshold 通常会设得较宽松，从而接受更多 real，同时也会接受更多 fake。因此完全可能出现：

- AUC≈0.85：总体上 detector 能较好排序 real 和 fake；
- FA@低 FRR 阈值≈0.6：在“少拒绝真人”的严格可用性要求下，仍有很多 fake 被接受。

二者不矛盾。

### 2.4 FA-FRR 曲线与 user-level bootstrap

- FA-FRR 曲线是在 test scores 上扫所有 threshold 得到的可视化；
- 主表的 operating point 不从 test 反选，而是把 validation 选定 threshold 固定应用到 test；
- user-level bootstrap 只估计置信区间：有放回抽 test user，每次保留该 user 的全部 windows，再计算指标；
- 它与把每个 window 当成独立样本重采样不同，能保留同一用户窗口之间的相关性。

---

## 3. 只有先对齐协议，结果才可以比较

| 实验 | 真正 neural sampling | selector | fake 数量 | fake 用户 | real test | detector 范围 | 能否与正式 baseline 直接比较 |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| 早期 raw-DDPM baseline | 是 | 无 | 每 action 14k/2k/4k | 70/10/20 | 全 100 users | 9/action，PAD 与 XY | 是 |
| 第一次 Android sparse cache | 是，但 wrapper 错误 | 无 | 总共仅 5,200 events | 70/10/20 | 完整 real pool | IMU+XY+time；fake 极稀疏 | 否，只能诊断 |
| 修复 Android cache v2 | 是 | 无 | 总共 100,000；每 action 14k/2k/4k | 70/10/20 | 全 100 users | 9/action，IMU-only 与 IMU+XY+time | 是 |
| action-wise train-prior exact/mix | 否，复制/混合 train windows | 无 | 每 action 14k/2k/4k | 70/10/20 | 旧 matched-user 方案 | PAD 与 XY | 否，只是上界诊断 |
| diffusion + train-only selector | 是 | K→200 后筛选 | 选后 200/user | 固定 split | 旧 matched-user 方案 | PAD | 不能当纯生成器结果 |
| v2/v4/v5/v6 小 gate | 是 | 无 | 20/user 等小样本 | split 不重叠 | gate real pool | 通常 feature-only | 否，只能筛方向 |
| trajectory-humanization 当前版 | 设计为是 | 明确禁止 | 正式目标 100,000 | 70/10/20 | 全 100 users | 正式尚未运行 | 无结果可比较 |

最容易造成误解的三种混用是：

1. 把 matched 20-user real test 与 all-100-user real test 混为同一协议；
2. 把 K 个候选中筛 200 个的 selector 结果称为“直接生成结果”；
3. 把 train-prior exact 复制真实训练窗口称为 diffusion 生成。

---

## 4. 早期完整 raw-DDPM baseline

### 4.1 模型与输入

该版本使用五个 action 独立的 few-shot + ADV raw-DDPM checkpoint：tap、scroll、swipe、pinch、keystroke。波形模型的条件包含目标 mask/active length、orientation 和 five-shot refs；keystroke 还包含字母/按键相关条件。checkpoint 本身不直接以 XY 为波形条件，XY-aware 是 detector 输入变体。

测试协议：

- fake users：70 train / 10 validation / 20 test，互斥；
- fake test：每 action 4,000 条、20 个 held-out fake users；
- real test：该 action 的全部 100 个 real users 的 held-out real windows；
- no_select：直接采样后评估，不从 test candidates 中挑样本；
- 9 detectors/action：3 个 HMOG-style、2 个 paper-style、4 个 deep；
- threshold 只在 validation 上选择。

### 4.2 PAD / IMU-only 正式结果

| action | FA@val-FRR≤5% | test FRR | EER FA | EER FRR | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| tap | 0.8012 | 0.0256 | 0.3854 | 0.2055 | 0.7829 |
| scroll | 0.4443 | 0.0543 | 0.2061 | 0.1813 | 0.8859 |
| swipe | 0.4859 | 0.0455 | 0.2050 | 0.1660 | 0.8946 |
| pinch | 0.6203 | 0.0453 | 0.2821 | 0.2161 | 0.8312 |
| keystroke | 0.5611 | 0.0310 | 0.2503 | 0.1607 | 0.8786 |
| 五动作宏平均 | **0.5826** | **0.0403** | **0.2658** | **0.1859** | **0.8547** |

### 4.3 早期 XY-aware / IMU+XY 结果

| action | FA@val-FRR≤5% | test FRR | EER FA | EER FRR | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| tap | 0.7591 | 0.0286 | 0.3491 | 0.1972 | 0.8073 |
| scroll | 0.4890 | 0.0538 | 0.2179 | 0.1941 | 0.8750 |
| swipe | 0.5177 | 0.0447 | 0.2125 | 0.1761 | 0.8865 |
| pinch | 0.6516 | 0.0457 | 0.3036 | 0.2258 | 0.8140 |
| keystroke | 0.5633 | 0.0314 | 0.2445 | 0.1674 | 0.8770 |
| 五动作宏平均 | **0.5961** | **0.0408** | **0.2655** | **0.1921** | **0.8520** |

该输入是 `IMU+XY`。修复后的 Android v2 使用 `IMU+XY+log(duration)`，多了 time，因此两张 XY 表可分别分析，但不能把差值全部归因于 wrapper。

### 4.4 该结果应怎样解释

这不是“fake 已经无法检测”。AUC 仍为 0.78–0.89，说明 detector 仍有明显排序能力。更准确的结论是：

- 直接 neural sampling 与 real 有中等程度重叠；
- tap 最难检测，scroll/swipe 最容易检测；
- 在低 FRR operating point 下 FA 较高，但在 validation-EER threshold 下 FA 只有约 0.21–0.39；
- 这是后续 Android 封装应该复现的神经 baseline，而不是“所有 action 都达到 0.4”的结论。

---

## 5. 第一次 Android cache 为什么彻底失败

### 5.1 观察到的结果

第一次 cache 只产生 5,200 个事件：

| action | fake train | fake val | fake test |
| --- | ---: | ---: | ---: |
| tap | 280 | 40 | 80 |
| scroll | 840 | 120 | 240 |
| swipe | 840 | 120 | 240 |
| pinch | 840 | 120 | 240 |
| keystroke chunks | 3,010 | 409 | 866 |

在 IMU+XY+time detector 下，按 5 个 feature 与 4 个 deep detector 加权后的 action 结果约为：

| action | EER FA | AUC |
| --- | ---: | ---: |
| tap | 0.004167 | 0.997634 |
| scroll | 0.001389 | 0.999248 |
| swipe | 0.000463 | 0.999782 |
| pinch | 0.003241 | 0.999811 |
| keystroke | 0.000642 | 0.999984 |
| 五动作宏平均 | **0.001980** | **0.999292** |

这个结果表示 fake 几乎全部被拒绝。

### 5.2 第一主因：raw/residual 表示合同不一致

checkpoint normalizer 的均值本身包含真实重力，例如 tap 约为 `[-0.37, 5.18, 7.55, ...]`，证明它是 raw-DDPM。

错误 wrapper 却：

- 导入 residual-level 实验实现；
- 将 five-shot refs 做 residual normalization；
- 对模型反归一化后的 raw 输出再次加 `level_raw`。

结果就是前述重力模长从 real 的约 9.62 变成 fake 的约 17.81。该单一错误足以解释 AUC≈1。

### 5.3 第二主因：test few-shot 静默退化成 no-shot

诊断 cache 中，五个 action 的全部 `fake_test_used_ref_indices` 都是：

```text
[-1, -1, -1, -1, -1]
```

原因是 Android layer 的 mutable ref bank：train/val 显式用户调用替换了当前 bank；随后 test user 虽属于配置的 test split，代码却没有检查当前 bank 是否真的含该 user，于是没有重建 refs。

因此报告名义上说“5 refs”，实际 test fake 没有使用 five-shot。这不仅改变生成质量，也使 train/val/test 的生成机制不一致。

### 5.4 其他会制造检测侧信号的问题

1. **orientation 没有硬传递**：request 的真实 metadata direction 与输出方向可能不一致；
2. **pinch XY 端点坍缩**：wrapper 只保存一个 center，并把它同时写成 start/end；真实 pinch 起终 focus 位移中位数约 437.46 px，错误 fake 恒为 0；
3. **target-user metadata 越过 five-shot 边界**：早期从目标用户全部 enroll rows 抽 duration/XY/orientation/text，而不是只用五 refs 加 train-only prior；
4. **duration、distance 和 bin 独立抽样**：破坏了真实联合分布；
5. **初始 query 只按 action 随机抽**：没有严格按 duration、XY、orientation、letter count 查询；
6. **样本过少且极不均衡**：tap fake test 只有 80 条，和正式每 action 4,000 条相差 50 倍；
7. **detector 输入也变了**：该轮只报告 IMU+XY+time，不是早期 IMU-only 的同输入复现。

因此第一次 cache 的数字不能用于判断“Android 封装是否可行”，只能作为错误实现的诊断证据。

---

## 6. Android cache v2 的修复与完整结果

### 6.1 代码和协议修复

v2 做了以下硬修复：

1. 导入与 checkpoint 匹配的 root `final_gen` raw-DDPM 实现；
2. 只反归一化一次，删除第二次 `level_raw`；
3. 添加 `representation=raw_ddpm` guard；
4. 每条 fake 强制 `ref_count=5`，五个 ref index 必须非负且都属于目标 user；
5. request orientation 必须传入，返回 orientation 必须一致；
6. pinch 同时保存 phase-0 start center 与 phase-2 end center；
7. metadata 只从 70 个 train users 的联合 prior 抽取，目标 val/test user 信息限定为五 refs；
8. 每 user/action 生成 200 条，恢复正式 100,000 条规模；
9. detector-set gate 检查 14k/2k/4k 数量、split 互斥、mask、IMU 有限性、分布和 ref ownership；
10. 同时完整测试 IMU-only 与 IMU+XY+time，所有 detector 行均保留。

最终审计为：

```text
passed=true
formal_rows=90
bootstrap_rows=140
plots=400
deep_checkpoints=40
```

### 6.2 IMU-only：九 detector 宏平均

| action | FA@val-FRR≤5% | test FRR | EER FA | EER FRR | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| tap | 0.8236 | 0.0344 | 0.3889 | 0.2316 | 0.7593 |
| scroll | 0.4399 | 0.0510 | 0.1892 | 0.1797 | 0.8951 |
| swipe | 0.4867 | 0.0478 | 0.2009 | 0.1775 | 0.8905 |
| pinch | 0.5973 | 0.0608 | 0.2668 | 0.2456 | 0.8159 |
| keystroke | 0.6134 | 0.0299 | 0.2671 | 0.1628 | 0.8710 |
| 五动作宏平均 | **0.5922** | **0.0448** | **0.2626** | **0.1994** | **0.8464** |

### 6.3 IMU+XY+time：九 detector 宏平均

| action | FA@val-FRR≤5% | test FRR | EER FA | EER FRR | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| tap | 0.6342 | 0.0321 | 0.2999 | 0.1849 | 0.8219 |
| scroll | 0.4770 | 0.0466 | 0.1976 | 0.1802 | 0.8896 |
| swipe | 0.5175 | 0.0460 | 0.2139 | 0.1813 | 0.8836 |
| pinch | 0.6214 | 0.0509 | 0.2639 | 0.2291 | 0.8314 |
| keystroke | 0.6019 | 0.0327 | 0.2624 | 0.1783 | 0.8637 |
| 五动作宏平均 | **0.5704** | **0.0417** | **0.2476** | **0.1908** | **0.8580** |

### 6.4 与早期 baseline 的逐动作复现

下表只比较最可比的 IMU-only EER FA：

| action | 早期 baseline | 修复 Android v2 | 差值 |
| --- | ---: | ---: | ---: |
| tap | 0.3854 | 0.3889 | +0.0035 |
| scroll | 0.2061 | 0.1892 | -0.0169 |
| swipe | 0.2050 | 0.2009 | -0.0041 |
| pinch | 0.2821 | 0.2668 | -0.0153 |
| keystroke | 0.2503 | 0.2671 | +0.0168 |
| 宏平均 | 0.2658 | 0.2626 | -0.0032 |

没有任何 action 出现第一次 cache 那种从约 0.2–0.39 跌到约 0 的崩溃。这进一步证明修复是有效的。

---

## 7. 各轮新方法与尝试：哪些完整，哪些只是 gate

### 7.1 总表

| 版本/方法 | 核心改动 | 测试级别 | 主要结果 | 结论 |
| --- | --- | --- | --- | --- |
| trajectory-conditioned v1 | 新轨迹条件 diffusion | 四动作 formal no_select PAD | tap/swipe FA=0；scroll/pinch≈0；AUC≈1 | gravity/DC level 严重错位，失败 |
| residual-level v2 eps | 生成 residual，再恢复 level | tap、20/user、feature gate | FA=0.0035，AUC≈0.9997 | level 修好，但动态仍易检测 |
| residual-level v2 x0 | 改 x0 objective | tap、20/user、feature gate | FA≈0.1620，AUC≈0.8851 | 有改善，仍不足；未完成四动作 formal |
| raw detector-space ADV v3 | critic 直接约束 raw detector 特征/频谱 | **四动作完整 formal** | no-select PAD FA=0.1433/0.1122/0.1119/0.1116 | 完整结果显著低于 gate 和旧 baseline |
| template-anchor v4 | 真实模板 carrier + partial denoise | tap gate | FA=0.1880，AUC=0.9276 | 不启动 full |
| transition-aware v5 | 加 velocity/acceleration/jerk/burst transition | tap、scroll gates | tap 0.2515/0.8243；scroll 0.2140/0.8486 | scroll 低于结构门，停止 full |
| frequency-detail v6 | 频带能量与细节条件/约束 | tap/scroll/swipe gates | FA=0.1780/0.2270/0.2110 | 三项均低于 0.25，停止 full |
| Android cache v1 | 把旧 raw-DDPM 封装进缓存 API | 5,200 条诊断 | 宏 FA≈0.002，AUC≈0.999 | wrapper 错误，不是模型结论 |
| Android cache v2 | 修复 raw/ref/metadata/orientation/pinch | **五动作 100k formal** | IMU-only EER FA=0.2626，AUC=0.8464 | 成功恢复旧 neural baseline |
| trajectory-humanization 20260713 | Android 轨迹 + IMU joint diffusion 正式协议 | 仅 smoke/gate/代码审计 | **无正式 FA/FRR** | 尚未训练和完整检测 |

### 7.2 raw detector-space ADV v3 的完整结果

v3 是少数真正完成四动作 full sampling、PAD 与 XY-aware 的新神经版本，因此应单独保留：

| action | no-select PAD FA/AUC | no-select XY FA/AUC | select PAD FA/AUC | select XY FA/AUC |
| --- | --- | --- | --- | --- |
| tap | 0.1433 / 0.9242 | 0.1578 / 0.9155 | 0.1950 / 0.8702 | 0.1995 / 0.8687 |
| scroll | 0.1122 / 0.9321 | 0.1142 / 0.9320 | 0.1559 / 0.8807 | 0.1573 / 0.8805 |
| swipe | 0.1119 / 0.9346 | 0.1127 / 0.9344 | 0.1789 / 0.8719 | 0.1765 / 0.8720 |
| pinch | 0.1116 / 0.9198 | 0.0937 / 0.9309 | 0.1520 / 0.8606 | 0.1541 / 0.8618 |

小 gate 中 tap feature FA=0.2515，但完整九-detector结果只有 0.1433。这说明：

- 小规模 feature gate 只适合决定“是否值得继续”，不能预测完整结果；
- 对 critic proxy 的改进不一定迁移到多步 diffusion samples；
- deep/paper detector 可能利用 gate 未覆盖的统计量；
- 更多 ADV 轮次并不自动等价于更高最终 FA。

### 7.3 为什么“多训几轮 ADV”不一定更好

ADV 训练优化的是训练时 critic 的近似目标，而最终评估看的是完整多步采样和独立 detectors。两者存在三层 gap：

1. critic 可能看到随机 t 的一步 `x0_pred`，不是最终 240/320-step sample；
2. critic 只覆盖有限特征、架构和训练数据，独立 PAD 可以利用其他统计量；
3. 过强 ADV 容易牺牲多样性或产生针对 critic 的模式，导致 selector/critic 分数更好但真实 detector 更容易识别。

v3 的 gate→formal 下跌就是实证。因此正确做法是：每个结构改动先 gate，但最终只以 no-select full sampling + 全 detector 为准。

---

## 8. Selector 与 train-prior：数字较高，但含义不同

### 8.1 train-only selector

旧 selector 从每 user 先生成 K 个候选，再用只在 train 数据上建立的 selector 选 200 个。v1 K=600→200 的旧 matched-user 协议得到：

| action | selector PAD FA |
| --- | ---: |
| tap | 0.4648 |
| scroll | 0.2818 |
| swipe | 0.2510 |
| pinch | 0.3688 |
| 宏平均 | 0.3416 |

这不是 test 泄漏，但也不是“生成器直接生成 200 条”的性能，而是：

```text
ADV diffusion 生成 600 个候选
        ↓
train-only selector 进行后处理
        ↓
保留 200 个再送 detector
```

因此它应标为 `generator + post-selection`。此外该结果使用旧 matched-user real 协议，不能和 all-100-real baseline/v2 直接做数值排名。

后续 selector 并没有稳定变好：

- tree calibration v2 宏 FA 约 0.2650；
- coverage v3 仅产生局部 gate；
- K=1200 v4 因成本和未完成流程没有正式最终指标；
- raw-ADV v3 完整 select 宏 FA 也只有约 0.1704，且 deep PAD 约 0.0069。

这说明 selector 会放大 mode collapse 或只优化 feature proxy，不能保证跨 detector 提升。

### 8.2 action-wise train-prior exact/mix

该方法的旧结果很高：

| action | PAD FA/AUC | XY-aware FA/AUC |
| --- | --- | --- |
| tap, `train_prior_mix05` | 0.4654 / 0.5357 | 0.4695 / 0.7513 |
| scroll, `train_prior_exact` | 0.4948 / 0.5029 | 0.4692 / 0.6997 |
| swipe, `train_prior_mix05` | 0.4564 / 0.6336 | 0.4117 / 0.7560 |
| pinch, `train_prior_exact` | 0.4701 / 0.4997 | 0.4772 / 0.7057 |

但它不是 neural diffusion：

- `exact` 直接复用 train-user 的真实窗口；
- `mix05` 对真实 train windows 做小比例混合/扰动；
- 旧测试又采用 matched-user real 协议。

所以它只能说明“如果输出非常接近真实训练片段，PAD 上界可以多高”，不能作为生成模型的正式结果，也不能用于证明可生成全新轨迹。

---

## 9. 当前 trajectory-humanization 版本的客观状态

目录：

```text
/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713
```

### 9.1 已完成

- 五 action 的 HMOG 轨迹/IMU 数据抽取与正式数据审计；
- train/val/test user split、five-shot registry、train-only condition prior；
- duration、orientation、XY、pointer lifecycle、pinch phase、keystroke letters 的条件结构；
- strict checkpoint、best/last、resume、seed、hash 和 100k archive 审计协议；
- feature/deep detector 独立协议；
- smoke、E2E、throughput projection 与相关代码测试。

最近一次代码审计中，training pipeline、throughput projection 与 supervisor 的 49 个 targeted tests 通过；五 action 抽样的 canonical timeline 长度一致。这里的 5-epoch profile 只用于估算 batch size/ETA，120 秒只是一项候选 wall-clock 资源门，不是正式数据或测试的简化。

### 9.2 未完成

- 100-epoch 五 action 正式训练；
- 五个正式 best-EMA 1000-step checkpoint；
- 100 users × 5 actions × 200 = 100,000 条正式 fake；
- feature/deep 全 formal detector 训练与 test；
- validation-selected threshold、test FA/FRR/AUC、user-level CI 和最终曲线。

因此当前状态必须写成：

> 代码与协议准备/审计完成，正式模型和正式性能尚未产生。

不能写成“trajectory-humanization 已达到某个 FA”。

---

## 10. 最终可用结论与禁止结论

### 10.1 可以写进论文或项目报告

1. 早期 action-specific few-shot ADV raw-DDPM 在五动作、all-100-real、no-select、九 detector 的 IMU-only 测试中，宏 EER FA 为 0.2658，AUC 为 0.8547。
2. 第一次 Android wrapper 因 raw/residual 表示错误、test refs 退化和 metadata/XY bug，使 fake 出现巨大物理侧信号，宏 EER FA 降至约 0.0020、AUC 升至约 0.9993。
3. 修复后 100k Android v2 在同类完整协议下恢复到宏 EER FA 0.2626、AUC 0.8464，与早期 baseline 基本一致。
4. 因而“封装后不行”的主要原因是 wrapper/API/data-contract 错误，而不是 Android cache 查询或 raw-DDPM checkpoint 天然无效。
5. 更强或更多 ADV、selector 与小 gate 并未稳定改善独立 detector 的完整结果；最终模型选择必须依赖 no-select full sampling 和完整 detector 套件。

### 10.2 不应写出的结论

1. 不应说第一次 Android cache 证明 diffusion 完全失败；它是已确认的错误实现。
2. 不应把 train-prior exact/mix 的 FA>0.4 写成生成模型结果。
3. 不应把 selector 宏 FA=0.3416 写成直接 diffusion 采样性能。
4. 不应把 20/user feature gate 写成九 detector 完整结果。
5. 不应把 matched-20 real 与 all-100 real 的结果直接排名。
6. 不应给当前 trajectory-humanization 版本填写尚未产生的正式 FA/FRR。

---

## 11. 证据与结果文件索引

### 11.1 早期完整 baseline

```text
/home/mwang49/real-human/imu_gen/final/android_physical_layer_20260709/docs/five_action_all_real_test_summary.md
```

### 11.2 第一次 Android sparse cache

```text
/home/mwang49/real-human/imu_gen/final/android_user_cache_xytime_20260710/docs/technical_report.md
```

### 11.3 wrapper 根因、各次停止与修复日志

```text
/home/mwang49/real-human/imu_gen/final/android_user_cache_xytime_full_20260710/docs/dev_log.md
/home/mwang49/real-human/imu_gen/final/android_user_cache_xytime_full_20260710/results/diagnostic_cache_audit.json
```

### 11.4 修复后完整 Android v2

```text
/home/mwang49/real-human/imu_gen/final/android_user_cache_xytime_full_20260710/docs/technical_report_v2.md
/home/mwang49/real-human/imu_gen/final/android_user_cache_xytime_full_20260710/v2/results/final_result_audit.json
```

### 11.5 train-prior 与 selector

```text
/home/mwang49/real-human/imu_gen/final/per_action_search_20260706/METHOD_TRAIN_TEST_ANALYSIS.md
/home/mwang49/real-human/imu_gen/final/docs/当前最优diffusion与selector客观说明_20260702.md
```

### 11.6 新神经方法 v2–v6

```text
/home/mwang49/real-human/imu_gen/final/residual_level_diffusion_v2_20260707/
/home/mwang49/real-human/imu_gen/final/raw_detector_adv_diffusion_v3_20260707/SUMMARY.md
/home/mwang49/real-human/imu_gen/final/template_anchor_diffusion_v4_20260707/
/home/mwang49/real-human/imu_gen/final/transition_aware_diffusion_v5_20260707/
/home/mwang49/real-human/imu_gen/final/frequency_detail_diffusion_v6_20260707/
```

### 11.7 当前 trajectory-humanization 协议与未完成状态

```text
/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713/docs/training_protocol.md
/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713/docs/generation_protocol.md
/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713/docs/detector_final_gate_20260713.md
```

---

## 12. 一句话总结

早期 diffusion 的中等 FA 是真实的完整 neural baseline；第一次 Android 封装跌到近零是 raw/residual 双重重力、five-shot 失效及 metadata/XY 错误共同造成的实现事故；把这些问题修复并按 100k、all-100-real、全 detector 协议重跑后，结果恢复到早期水平。后续若继续研究新 trajectory 模型，必须重新完成正式训练与完整评估，不能拿 smoke、gate、selector 或 train-prior 数字代替。
