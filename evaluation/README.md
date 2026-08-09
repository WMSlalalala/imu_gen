# 评测：主实验 / 对比实验 / 消融实验

> **路径约定**：本页与三份子 README 里的所有路径都相对 `evaluation/` 根书写，跨文件引用一律用可点击的相对链接。
> **语言约定**：本页、[EXPERIMENTS_CN.md](EXPERIMENTS_CN.md)、[DATA.md](DATA.md) 与三份子 README 为中文；[RESULTS.md](RESULTS.md) 与 19 份逐方法 notes 为英文，因为它们由 `final_tables.py` 与 `write_baseline_readmes.py` 从构建产物自动生成。英文读者见文末 [English summary](#english-summary)。

## 这是什么

**被评测的对象是一套针对手机行为生物特征认证的呈现攻击（presentation attack）方法。** 给定受害者每个动作 **5 条**真实事件（`k_refs = 5`，即「五-shot」威胁模型），它合成伪造的触摸轨迹与六轴惯性（IMU）信号，目标是让部署好的「真人 vs 伪造」检测器把它们当作真人事件放行。方法由三部分组成：**五-shot 条件变长扩散生成器**；**对抗训练**——feature / set / waveform 三个 critic 加一项直接特征匹配损失 `feature_match`，四者的梯度在**梯度层面**合并（而非标量加权相加）；以及 keystroke 动作专用的**解析式 IMU 适配器**（该动作的伪造 IMU 不经过扩散生成器）。攻击面是 HMOG 五个动作（tap / scroll / swipe / pinch / keystroke）的触摸通道与惯性通道；方法侧的完整链路索引在 [`../method/README.md`](../method/README.md)。

**「发布版（release）」= 论文定稿时冻结的那一份产物**（生成器检查点 + 四个 per-action bundle + 90 个已拟合检测器，6.5 GB，不在仓库），也就是消融表里的 **A1**。全文只用「发布版」这一个叫法，不再出现「本文方法」「ours」等别称。

论文尚未公开，题名 `<论文题名：待填>`、预印本链接 `<预印本链接：待填>`、bibtex `<待填>`（与 [DATA.md](DATA.md) 末尾的归档链接占位同一口径）。代码镜像：<https://github.com/WMSlalalala/imu_gen>。

## 结论

评测网格：五个动作 × 三个模态（轨迹 / 惯性 / 联合）× 六个检测器 = **90 个「格子（cell）」**，一格一个分数。

> **在 HMOG 的 90 个格子上，发布版在 FRR=5% 操作点的 FAR 均值 0.775、中位 0.794、77/90 格 ≥ 0.60。就惯性通道的同一 30 格比较：发布版 0.835，最强第三方基线 ImagenTime 0.683，真人对照 0.773。**

**FAR 越高＝攻击越强**，成功线取 0.60。判据是**开发集上选定的 FRR=5% 阈值处的 FAR**，不是 EER 阈值上的——两者在本项目上给出方向相反的结论，见[下一节](#为什么是-frr5-上的-far不是-eer)。跨模态的数字不要混着比：0.775 是三个模态 90 格的合计，0.683 / 0.773 只在惯性通道的 30 格上。

| | 格子 | 均值 | 中位 | ≥0.60 |
|---|---|---|---|---|
| **发布版 · 全部 90 格** | 90 | **0.775** | 0.794 | 77 |
| **发布版 · 轨迹**（`trajectory_xytime`） | 30 | **0.777** | 0.779 | 26 |
| **发布版 · 惯性**（`imu_only`） | 30 | **0.835** | 0.840 | 30 |
| **发布版 · 联合**（`imu_trajectory_xytime`） | 30 | **0.711** | 0.734 | 21 |
| 最强第三方基线：ImagenTime（惯性） | 30 | 0.683 | 0.778 | 20 |
| 最弱：TTS-GAN（惯性） | 30 | 0.124 | 0.008 | 1 |
| 对照：把真人惯性窗口当作伪造通道灌进同一流水线 | 30 | 0.773 | 0.844 | 22 |
| 最大消融效应：A8，移除 feature critic | 12 | 0.682 | 0.653 | 8 |

消融中效应最大的是 A8，逐格配对后比发布版低 0.107。**A1–A11 十一个臂已全部落地**（A11 移除直接特征匹配损失，12 格均值 0.717、配对差 +0.072），[RESULTS.md](RESULTS.md) 表 2 里没有任何一行再标 *not finished*。对照行是同流水线的参照点，不是上界。

## 三个子目录

| 子目录 | 回答什么 | 内容 |
|---|---|---|
| [`main/`](main/README.md) | 发布版自身在 90 格上的成绩 | [`main/scores/release_90_cells.csv`](main/scores/release_90_cells.csv)（90 行）。生成与检测代码不在本目录，而在仓库根部已有的几个目录中，路径见 [`main/README.md`](main/README.md) |
| [`comparison/`](comparison/README.md) | 八个第三方基线加一条真人对照，在同一实验台上各能到多少 | `comparison/code/`（30 个文件：29 个脚本 + 1 份数据文件 `released_generators.json`）、[`comparison/scores/comparison_cells.csv`](comparison/scores/comparison_cells.csv)（258 行）、`comparison/notes/`（9 份逐方法说明） |
| [`ablation/`](ablation/README.md) | 发布版的哪个部件在起作用 | `ablation/code/`（12 个脚本）、[`ablation/scores/ablation_cells.csv`](ablation/scores/ablation_cells.csv)（180 行）、`ablation/notes/`（10 份，覆盖 A2–A11） |

## 为什么是 FRR=5% 上的 FAR，不是 EER

部署时厂商先规定用户能忍受的误拒比例，再看这个前提下漏进来多少假货。在这 90 格上逐格重算，EER 阈值平均要拒掉 33.6% 的真人事件，没有产品会这样发货；FRR=5% 阈值上实测拒真率均值 5.2%。两个判据在本项目上给出方向相反的结论：同一批 90 格，按 EER 读是均值 0.386、0 格过 0.60，按 FRR=5% 读是均值 0.775、77 格过 0.60。格子产物默认写出的 `far` 字段正是 EER 阈值上的值，直接汇总会系统性低估攻击面。细节见 [EXPERIMENTS_CN.md](EXPERIMENTS_CN.md) §1.4。

## 环境

**只复算分数：不需要环境。** 本页与三份子 README 里的每一个均值 / 中位 / ≥0.60 计数，都能从三份 CSV 用 **Python 3 标准库**（`csv` + `statistics`）算出来，秒级、无需 GPU。[`main/README.md`](main/README.md) 里的示例片段另需 pandas。

**重跑基线或重训检测器：需要完整环境。** 作者机器实测配置（见 [requirements.txt](requirements.txt) 的钉版本）：

| 项 | 实测值 |
|---|---|
| Python | 3.10.20（conda env `cuhkx`；`code/` 下脚本把该解释器绝对路径写死了，换机器须先改） |
| PyTorch | 2.5.1+cu121（torchvision 0.20.1+cu124、cuDNN 9.1.0） |
| GPU / 驱动 | 2 × NVIDIA RTX A6000（各 49,140 MiB），驱动 610.43.02。脚本以 `--gpu <id>` 单卡执行，**单作业显存下限未测量** |
| 检测器侧 | numpy 2.2.6、scikit-learn 1.7.2、xgboost 3.2.0、joblib 1.5.3 |
| 其它 | PyYAML 6.0.3、pandas 2.3.3 |
| 上游基线仓库 | 各自的依赖按其自带说明安装；同一 env 里实测共存的版本为 einops 0.8.2、ema-pytorch 0.7.9、tqdm 4.68.1、matplotlib 3.10.9、accelerate 1.13.0、transformers 4.53.3 |
| 磁盘 / 机时 | 从零复现整条链约需 150 GB 磁盘与双卡 GPU 数日机时 |

**版本漂移会改结果，这不是空话**：两个树模型（`paper_xgboost`、`hmog_style_rf`）对数值细节极其敏感——本项目付过一次学费，`elapsed` 时钟列的一次 float32 往返差异就让某格 FAR 从 0.847 掉到 0.000（[EXPERIMENTS_CN.md](EXPERIMENTS_CN.md) §1.5）。跨大版本重训检测器，逐格分数不保证复现。

## 仓库里有什么、没有什么

有：全部评测与基线构建代码、逐格分数 CSV（列为 `action` / `modality` / `detector` / `far_at_frr5`，对比与消融两张表另有 `method` 列）、19 份由构建产物自动生成的逐方法可复现说明（因此不会与实际跑的东西漂移；**一处已知例外**——`comparison/notes/csdi_unconditional.md` 里关于五-shot 臂的那一句是脚本硬编码、不是从产物读出来的，文字本身已按日志证据改正并重新生成，但它仍靠人工维护，见 [comparison/README.md](comparison/README.md) §7 与 §9）。分数可自行核对，例如复算发布版总成绩：

```bash
cd evaluation
python3 - <<'PY'
import csv, statistics
v = [float(r["far_at_frr5"]) for r in csv.DictReader(open("main/scores/release_90_cells.csv"))]
print(len(v), round(statistics.mean(v), 3), round(statistics.median(v), 3), sum(x >= 0.60 for x in v))
PY
```

输出 `90 0.775 0.794 77`。把同一文件筛到 `imu_only` 且动作非 keystroke，得到消融基线 A1 的 24 格、均值 0.835、24 格过线。

没有：118 GB 的生成事件与逐格产物、6.5 GB 的冻结发布版（生成器检查点、检测器权重、四个动作 bundle）。两者都在机构存储上，不进仓库。没有它们仍可以：核对全部分数、通读代码与说明、按上游 commit 钉版本重跑第三方基线（CSDI `7f24a436f08d`、Diffusion-TS `566307e6cf2d`、ImagenTime `f372626ed20a`、TTS-GAN `3f8b36ab84d1`、pyclick `bf0edd19892d`；第三方仓库本身不入库）。不能做的是复现发布版自身的 90 格，那需要冻结的生成器与检测器权重。获取方式与目录结构见 [DATA.md](DATA.md)。

**还有一件必须同样说明白的事：`comparison/code/` 与 `ablation/code/` 里的脚本不能独立运行。** 它们是真跑过的那一份、出处忠实，但另有 6 个模块（`hmog_baseline_common.py`、`export_real_windows.py`、`build_sample_bank_baseline.py`、`ghost_cursor_path.py`、`grid_against_final.sh`、`gpu_slot.sh`）与两棵上游代码树（`data7/code/baselines/`、`data7/code/direct100k/`）不随本仓库发布，缺它们**训练、构建、出表三步都跑不起来**；此外脚本里的机器绝对路径换机器必须先改。逐个清单、缺在哪一步、要改哪一行，见 [comparison/README.md](comparison/README.md) §6 前置与 §7、[ablation/README.md](ablation/README.md) 的依赖清单。

## 名字对照：CSV id ↔ 中文名 ↔ 表格显示名

**CSV 里的 id 是唯一权威**。正文一律用 CSV id；表头可以用显示名，但首次出现处要括注 id。

模态（CSV 列 `modality`）：

| CSV id | 中文名 | 表格显示名 |
|---|---|---|
| `trajectory_xytime` | 轨迹 | trajectory |
| `imu_only` | 惯性 / IMU | imu |
| `imu_trajectory_xytime` | 联合 | joint |

检测器（CSV 列 `detector`，表格中的固定列序即下表顺序）：

| CSV id | 中文名 | 表格显示名 |
|---|---|---|
| `hmog_style_svm` | HMOG 风格线性 SVM | HMOG-SVM |
| `hmog_style_rf` | HMOG 风格随机森林 | HMOG-RF |
| `paper_svm` | 论文特征线性 SVM | Paper-SVM |
| `paper_xgboost` | 论文特征 XGBoost | Paper-XGB |
| `behaveformer_stdat` | BehaveFormer STDAT 深度序列模型 | BehaveFormer |
| `authconformer` | AuthConFormer 深度序列模型 | AuthConformer |

方法与消融臂（CSV 列 `method`）：

| CSV id | 中文名 / 臂号 | 表格显示名 |
|---|---|---|
| （`main` 表无 `method` 列） | 发布版 = A1 | Ours (released) / A1 |
| `diffts_trajectory` / `diffts_imu` / `diffts_both` | Diffusion-TS 轨迹臂 / IMU 臂 / 双通道臂 | Diffusion-TS (traj / IMU / dual arm) |
| `csdi_unconditional` | CSDI（无条件采样臂） | CSDI |
| `imagentime` | ImagenTime | ImagenTime |
| `ttsgan` | TTS-GAN | TTS-GAN |
| `pyclick` | pyclick（贝塞尔拟人光标库） | pyclick |
| `ghostcursor` | ghost-cursor（Fitts 定律拟人光标库） | ghost-cursor |
| `control_genuine` | 真人对照 | Control |
| `abl_noshot_adv` | A2，去掉五-shot 条件（`k_refs=0`） | A2 |
| `abl_fewshot_nonadv` | A3，关掉整套对抗训练 | A3 |
| `abl_krefs1` / `abl_krefs3` / `abl_krefs8` | A4 / A5 / A6，`k_refs` 改为 1 / 3 / 8 | A4 / A5 / A6 |
| `abl_a7_weighted_sum` | A7，梯度合并换成标量加权相加 | A7 |
| `abl_a8_no_feature` | A8，移除 feature critic | A8 |
| `abl_a9_no_set` | A9，移除 set critic | A9 |
| `abl_a10_no_waveform` | A10，移除 waveform critic | A10 |
| `abl_a11_no_feature_match` | A11，移除直接特征匹配损失 | A11 |

## 术语速查

| 词 | 一句话定义 | 详见 |
|---|---|---|
| **HMOG** | 一个公开的手机行为生物特征数据集，100 名匿名志愿者，同时记录触摸事件与惯性读数 | §1.1 |
| **PAD**（Presentation Attack Detection，呈现攻击检测） | 本项目的检测器不是「本人 vs 他人」的身份核验器，而是每格一个「真人 vs 伪造」二分类器 | §1.3 |
| **IMU**（Inertial Measurement Unit，惯性测量单元） | 加速度计 + 陀螺仪，六轴 | §1.1 |
| **FAR**（False Acceptance Rate，假接受率） | 伪造样本被检测器接受的比例；本项目的唯一判据，越高攻击越强 | §1.4 |
| **FRR**（False Rejection Rate，拒真率） | 真人样本被误拒的比例；阈值固定在 FRR=5% 这一点上 | §1.4 |
| **EER**（Equal Error Rate，等错误率） | 「拒真率 = 假接受率」那一点；本项目**不用**它做判据 | §1.4 |
| **格子（cell）** | 评测的最小单位，即「动作 × 模态 × 检测器」一个三元组，出一个数；完整网格 5×3×6 = 90 格 | §1.3、§1.6 |
| **臂（arm）** | 同一方法被评测的一个变体配置，例如消融的 A1–A11、Diffusion-TS 的三个通道臂 | §1.6 |
| **发布版（release）** | 论文定稿时冻结的产物，即被评测的攻击方法本体，也是消融表里的 A1、所有基线共用的载体来源 | §1.2 |
| **载体（carrier）** | 一条伪造事件里**除被替换通道以外**的全部内容：行数、时钟列 `elapsed`、无接触哨兵 `(0,0)`、以及全部真人事件 | §1.3、§2.1 |
| **原位替换（in-place swap）** | 各方法只提供一条通道，由实验台写回同一批伪造事件的同一批行，其余逐字节相同——保证两行之间唯一的差别就是那条通道 | §2.1 |
| **bundle** | 发布版按动作组织的四个数据集包（`tap_and_pinch`、`scroll`、`swipe`、`keystroke`）；每个 bundle 都带全五个动作的事件，但只有 `owned_actions` 里的动作属于该发布版 | [DATA.md](DATA.md) |
| **declined（拒绝报数）** | 一个方法不建模某个动作时显式声明拒绝，那些格子不占分母也不记 0 分——与「未测量」是两件事 | §2.5 |
| **critic** | 对抗训练里的判别器；本方法有 feature（只看分布统计量）、waveform（直接看整段信号）、set（判断是否「属于」那五条参考）三个，外加不属于 critic 的 `feature_match` 直接匹配损失 | §5.3 |
| **`k_refs` / 五-shot** | 采样时给生成器的受害者参考条数；发布版为 5，消融扫 0/1/3/8 | §1.2 |
| **合库 / 建库 / 排格（子）** | 合库＝把逐动作采样出的样本拼成一个 bank；建库＝把 bank 原位替换进发布版副本、装配成可打分的样本库；排格＝在该样本库上逐格从零训练检测器、在开发集上切阈值、再算测试集 FAR | §2.5 |

章节号均指 [EXPERIMENTS_CN.md](EXPERIMENTS_CN.md)。

## 预期用途与伦理

**用途声明。** 本目录发布的是**防御研究材料**：攻击代码、逐格分数与完整方法学说明，用于衡量现有行为生物特征 PAD 检测器在五-shot 条件下的真实脆弱程度，并为改进检测器提供可复现的对照基准。**不得**用于绕过任何真实系统的认证、冒充任何真人用户，或在未获授权的设备与账号上使用。

**为什么攻击结果值得公开。** 六个检测器覆盖传统特征工程、树模型与深度序列模型三类范式，在 FRR=5% 这个厂商实际会选的操作点上被普遍击穿（90 格里 77 格 ≥0.60），而按 EER 读这批检测器看起来毫无问题（0 格过线）。不把这个口径差与具体失效模式讲清楚，检测器一侧无从修起；§3.7 逐项量化了各方法的失效模式（防御方可以直接照着找线索），§7 则把本文的局限、哪些结论不成立一并列出，包括对本文不利的部分。

**数据合规。** 全部实验建在公开数据集 HMOG 上，**本项目未采集任何新被试数据**，也未做任何再识别（re-identification）尝试。HMOG 的被试为 100 名匿名志愿者，数据集自带匿名编号（文档里的 `user_000`–`user_099` 即该编号，不含任何直接标识信息）；「受害者」一词指的是这些匿名编号对应的行为样本，不指向任何可识别的自然人。使用 HMOG 请遵守其官方发布条款并引用其主论文：Sitová et al., *HMOG: New Behavioral Biometric Features for Continuous Authentication of Smartphone Users*, IEEE TIFS, 2016。本仓库不分发 HMOG 原始数据，也不分发由其派生的事件数据（见 [DATA.md](DATA.md)）。

**发布范围。** 冻结发布版（生成器检查点与检测器权重）与 118 GB 生成事件不随仓库公开分发，获取需按 [DATA.md](DATA.md) 的方式联系作者。

## 完整实验说明

[EXPERIMENTS_CN.md](EXPERIMENTS_CN.md)，1148 行、七节（开头附一张「文中路径 → 仓库路径」对照表，正文引用的内部产物都标了「未发布」）：§1 问题与判据；§2 实验台的公平性保证（原位替换、declined 与「未测量」的区别、自检回执）；§3 八个第三方基线逐个说明；§4 A1–A7；§5 A8–A11；§6 迁移表——用发布版出厂的检测器去打所有攻击，回答的是另一个问题，不能当攻击强度读；§7 局限与可复现性。三张主表另见 [RESULTS.md](RESULTS.md)。

## English summary

This directory is the complete evaluation of a **five-shot presentation attack on smartphone behavioural-biometric authentication**: given five genuine events per victim per action, the method synthesises touch trajectories and six-axis IMU signals that a genuine-vs-spoof detector accepts. Over the 90 cells of the HMOG grid (5 actions × 3 modalities × 6 detectors), the released attack reaches a **mean FAR of 0.775 at the development-selected FRR = 5% operating point, with 77/90 cells ≥ 0.60**; restricted to the 30 inertial cells it scores **0.835**, against **0.683** for the strongest third-party baseline (ImagenTime) and **0.773** for the genuine-data control. Higher FAR means a stronger attack; EER is deliberately not used as the criterion and reverses the conclusion — see [RESULTS.md](RESULTS.md) for the three main tables and [EXPERIMENTS_CN.md](EXPERIMENTS_CN.md) §1.4 for why.

The protocol write-ups (this page, [EXPERIMENTS_CN.md](EXPERIMENTS_CN.md), [DATA.md](DATA.md) and the three sub-READMEs) are in Chinese; [RESULTS.md](RESULTS.md) and the 18 per-method notes are in English because they are generated from build artefacts by `final_tables.py` and `write_baseline_readmes.py`.
