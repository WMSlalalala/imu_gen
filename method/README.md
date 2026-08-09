# 方法：从原始数据到冻结发布版

本目录是本文攻击方法的完整链路索引。攻击的产物——伪造的触摸轨迹与惯性事件——在 `../evaluation/` 里被逐格评测；本目录回答的是这些产物**怎么来的**：从 HMOG 原始语料，到 five-shot 变长扩散生成，到二十五个 PAD 与五个综合检测器，再到冻结成发布版的统一发布接口。

与 `../evaluation/` 一样，本目录只装文本。方法代码不复制到这里，而是留在仓库根部已跟踪的几个工作目录中；下表把每个方法阶段指到它的实现目录与已跟踪的设计文档。事件数据、检查点与生成产物都在仓库外，见 [`DATA.md`](DATA.md)。

## 链路七阶段

| # | 阶段 | 做什么 | 实现目录（仓库根，已跟踪） | 设计文档 |
|---|---|---|---|---|
| 1 | **数据处理 / 轨迹提取** | 从 HMOG 原始语料提取 tap/scroll/swipe/pinch/keystroke 五动作的完整触摸轨迹，落成定长 padding 窗口，记录逐事件来源 | `trajectory_humanization_full_20260722_v16_numeric_recovery/` | `docs/formal_extraction_provenance_20260713.md`、`docs/extraction_static_audit_20260713.md` |
| 2 | **载体与 five-shot 参考** | 每用户 × 动作固定五条真实事件作参考（`k_refs=5`），构造变长条件与先验；这是威胁模型的边界——攻击者只有这五条 | `trajectory_humanization_full_.../` | `docs/strict_protocol.md`、`docs/generation_protocol.md` |
| 3 | **生成 / 变长扩散** | five-shot 变长扩散模型；ConditionRequest 支持任意时长；100,000 条正式轨迹生成协议 | `trajectory_humanization_full_.../` | `docs/model_design.md`、`docs/training_protocol.md`、`docs/generation_protocol.md` |
| 4 | **轨迹 PAD 检测器** | 五动作 × 五模型 = 25 个 trajectory PAD（presentation-attack detection）检测器，含特征法与深度法 | `trajectory_humanization_full_.../` | `docs/detector_protocol.md`、`docs/feature_protocol.md`、`docs/deep_benchmark_protocol.md`、`docs/detector_final_gate_20260713.md` |
| 5 | **综合检测器 / 跨模态一致性** | IMU score、trajectory score 与跨模态一致性合成五个 total detector；轨迹 runtime 与配对 IMU 链路 | `trajectory_estimator_pack_20260721/` | 该目录 `README.md` 及 `docs/` |
| 6 | **补充敏感性评测** | fully user-disjoint 与排除 five-shot 参考两条更严的评测臂 | `trajectory_pad_supplement_20260722/` | 该目录 `README.md` |
| 7 | **IMU 统一发布接口** | cache / online 双后端、fail-closed 门禁的 IMU 发布接口，把生成能力冻结成可调用的发布版 | `android_duration_time_fixed_20260720/imu_release_20260721/` | 该目录 `README.md` |
| — | **部署 / 注入**（预留） | 把发布版接进移动 Agent、在真机上以触摸+IMU 落地。**本目录只留占位，代码在本地另写**，见 [`deployment/`](deployment/) | — | [`deployment/README.md`](deployment/README.md) |

阶段 1–4 都在 `trajectory_humanization_full_20260722_v16_numeric_recovery/` 里，它是提取、扩散、100k 生成、25 个 PAD 与正式 supervisor 的合并工作树。阶段 5–7 各自独立成目录。

## 与 evaluation 的关系

本目录是「怎么造」，`../evaluation/` 是「造得有多好」。评测的三部分——主实验（发布版 90 格）、对比实验（八个第三方基线）、消融实验（拆发布版部件）——都以本目录阶段 1–7 产出的冻结发布版为对象。判据、逐格分数与可复现说明都在 `../evaluation/`，本目录不重复。

一条边界要说清：阶段 3 的 100k 正式生成、阶段 4 的 25 个检测器、阶段 5 的 5 个综合检测器，都只在全部上游 fail-closed 门通过后才运行（见顶层 `../README.md` 的「当前状态」）。GitHub 上代码可执行不等于正式指标已产出；哪些已完成、哪些在跑，以 `trajectory_estimator_pack_20260721/docs/IMU与轨迹交付状态及问题清单.md` 为准。

## 冻结发布版是什么

阶段 1–7 的产物冻结在 `/mnt/share/mwang49/data7/direct100k_final/`（6.5 GB，不在仓库）：五个动作的扩散检查点、四个 per-action bundle 的生成事件、90 个格子各一份拟合好的检测器。这是 `../evaluation/` 主实验直接评测的对象。获取方式与目录结构见 [`DATA.md`](DATA.md) 与 `../evaluation/DATA.md`。

## 复现的入口

不需要大文件就能做的：通读本目录指向的设计文档，核对协议；按上游 commit 钉版本重跑第三方基线（见 `../evaluation/DATA.md`）。需要原始数据与 GPU 才能做的：重跑提取、重训扩散、重拟合检测器、复现 100k——这些的数据准备与门禁都写在上表的设计文档里。
