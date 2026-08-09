# 对比实验：八个第三方基线

判据是**开发集选定的 FRR=5% 阈值上的 FAR**，越高攻击越强。格子（cell）= 动作 × 模态 × 检测器，5 × 3 × 6 = 90 格为完整网格。逐格原始分见 `scores/comparison_cells.csv`（258 行，列 `method,action,modality,detector,far_at_frr5`）。

## 1. 八项是什么

| 目录名 | 方法 | 出处 | 上游版本 |
|---|---|---|---|
| `diffts_trajectory` / `diffts_imu` / `diffts_both` | Diffusion-TS，三个臂（只供轨迹 / 只供惯性 / 两条都供） | Yuan & Qiao, ICLR 2024 | `566307e6cf2d` |
| `csdi_unconditional` | CSDI（无条件采样臂） | Tashiro et al., NeurIPS 2021 | `7f24a436f08d` |
| `imagentime` | ImagenTime | Naiman et al., NeurIPS 2024 | `f372626ed20a` |
| `ttsgan` | TTS-GAN | Li et al., AIME 2022 | `3f8b36ab84d1` |
| `pyclick` | 贝塞尔拟人光标库 | patrikoss/pyclick (MIT) | `bf0edd19892d` |
| `ghostcursor` | Fitts 定律拟人光标库 | 算法出自 Xetera/ghost-cursor（TypeScript）；被 import 的是其 Python 移植 mcolella14/python_ghost_cursor | PyPI `python-ghost-cursor` **0.1.1**（wheel，不是 git 检出，没有 commit 可钉） |

前四项是论文里的时序生成模型；后两项是工程界现成的拟人光标库，不含学习成分、没见过任何受害者数据，给出的是「非学习方法能做到什么程度」的下界，不是竞争者。

**ghost-cursor 的出处分三层，缺一层就拿不到与作者相同的起点。** ①**算法来源**是 TypeScript 原库 `github.com/Xetera/ghost-cursor`——`notes/ghostcursor.md` 的 "Code" 行给的就是它，那是算法出处，不是可安装的依赖。②**实际依赖**是它的 Python 移植：PyPI 包 `python-ghost-cursor` 0.1.1（MIT，主页 `github.com/mcolella14/python_ghost_cursor`；`pip download python_ghost_cursor==0.1.1` 即得 wheel）。这个 wheel 装出两个顶层包，本项目用的是其中的 **`pyppeteer_ghost_cursor`**——§6 前置①里那个目录名就是这个包名，不是随手起的。本机 `code/pyppeteer_ghost_cursor/` 是从该 wheel 取出的同名目录，五个文件（`__init__.py`、`math.py`、`mouseHelper.py`、`spoof.py`、`js/mouseHelper.js`）与 wheel `RECORD` 记录的 sha256 逐字节一致。③本项目对该移植版做的**两处更正**（弧长采样取代参数等距采样、恢复被移植版丢掉的 `spreadOverride`；详见 `../EXPERIMENTS_CN.md` §3.6 与 `notes/ghostcursor.md` 的 "Every deviation"）**不改这个包**，而是在 `ghost_cursor_path.py` 里重新实现——它 import `pyppeteer_ghost_cursor.math` 与 `.spoof`（该文件 67、74、131 行），**而这个文件不在本仓库**，见 §6 前置⓪与 §7。也就是说：`pyppeteer_ghost_cursor` 归档代码里确实无人 import，但它并非可省，import 它的是那个未发布的移植文件。

`control_genuine`（真人对照）**不是基线**，是流水线参照点：把真人惯性窗口原封不动当伪造通道灌进同一条管道，回答「实验台本身有没有制造差距」。下表以斜体单列，不计入这八项。

## 2. 完整结果表

主表口径：每格都用**那一行自己的**伪造数据从零训练检测器、在开发集上重选阈值。八行基线与对照可由 `scores/comparison_cells.csv` 逐格复算，发布版参照行来自 `../main/scores/release_90_cells.csv`；两者与上游结果文档 `RESULTS.md` 表 1 一致。

| 方法 | 本表报告的模态 | 格子数 | 均值 | 中位 | ≥0.60 |
|---|---|---|---|---|---|
| **发布版（参照）** | 轨迹 / IMU / 联合 | 30 / 30 / 30 | **0.777 / 0.835 / 0.711** | 0.779 / 0.840 / 0.734 | 26 / 30 / 21 |
| Diffusion-TS 轨迹臂 | 轨迹 | 30 | 0.460 | 0.393 | 11 |
| Diffusion-TS IMU 臂 | IMU | 30 | 0.325 | 0.343 | 2 |
| Diffusion-TS 双通道臂 | 联合（轨迹、IMU 也在其供给范围内，但 60 格未计算，见 §9） | 30 | 0.266 | 0.205 | 3 |
| CSDI（无条件） | IMU | 30 | 0.292 | 0.175 | 4 |
| **ImagenTime** | IMU | 30 | **0.683** | 0.778 | 20 |
| TTS-GAN | IMU | 30 | 0.124 | 0.008 | 1 |
| pyclick | 轨迹 | 24 | 0.294 | 0.198 | 6 |
| ghost-cursor | 轨迹 | 24 | 0.335 | 0.232 | 6 |
| *真人对照* | *IMU* | *30* | *0.773* | *0.844* | *22* |

两个光标库是 24 格：keystroke 在网格上是一串位置恒定的按住，两点间没有路径，光标生成器无从产出，故记为 **declined（拒绝报数）**——不替作者编造他从未提出的东西，declined 不占格子也不记 0 分。

**没有供给的模态不报数字。** 只换惯性的方法，其 `trajectory_xytime` 格跑的其实是发布版自己的轨迹；反过来只换轨迹的方法，其 `imu_only` 格跑的是发布版自己的惯性。这不是推论，可以逐格核对：迁移打分产物里，`crossscore/pyclick.json` 与 `crossscore/ghostcursor.json` 的 24 个 `imu_only` 格与 `crossscore/_release_selfcheck.json` 的同名格**逐格 max |差| = 0.0000**，三者均值都是 **0.8355**（发布版 30 格口径 0.8350，差别只是这两个库 declined 掉了 keystroke）；`crossscore/diffts_imu.json` 的 30 个 `trajectory_xytime` 格同理，均值 0.7775 就是发布版的轨迹口径。**所以在 `scores/comparison_cells.csv` 里查不到这批格子是对的**：按此规则，pyclick / ghost-cursor 的 `imu_only` 主表格子从未排过；它们**已经排过**的 48 个联合模态格子（各 24 格，`gridqueue_pyclick.log`、`gridqueue_ghostcursor.log` 08-07 的 `imu_trajectory_xytime` 条目）则已按规则删除，不进 CSV。上面这五个文件都在工作树 `data7/results/direct100k/baselines/` 下，属 §8 所说「不在仓库里」的那部分。该规则由 `code/covered_modalities.py` 从各产物的 `release.json` 判定，不看方法名；联合模态更严，两条通道都换才报。**「没有供给」与「供给了但网格没跑」是两回事**：前者永远不出数字，后者是未完成项，逐条记在 §9，各方法 note 的 "Modalities supplied / reported below / Not yet computed" 三行把两者分开写。

## 3. 这张表为什么成立

**原位替换。** 各方法都不造数据集，只提供**一条通道**（触摸坐标或六轴惯性），由实验台写回同一批伪造事件的**同一批行**：行数、偏移、用户/会话/事件 ID、划分序号全不动。载体其余部分——更新时序、无接触哨兵 `(0,0)`、时钟列 `elapsed`、以及全部真人事件——逐字节相同。轨迹替换只许动 x、y，dx/dy 由 `recompute_deltas` 确定性重算，接触标志/压力/指针数/`elapsed`/可用性五列逐位相等（`verify_harness.py` 的 `UNCLAIMED_TRAJECTORY_COLUMNS`）。两行之间唯一的差别就是那条通道。

**真人一侧共用。** 每个 bundle 是 145,776 条事件 = 100,000 伪造 + 45,776 真人、100 个用户，按 `fixed_user_disjoint_70_10_20` 做用户不相交三划分；真人事件谁都不许碰。`verify_harness.py` 逐分片核对五项：路由（生成器写入由事件 ID 派生的指纹，必须落回该事件自己的行）、真人行未改、不该动的列未改、替换确实发生、载体时钟不是破绽；18 方法 × 4 bundle = 72 份日志，第五项全部 PASS。

**两条限定。** 威胁模型不对称且对基线有利：第三方生成器用 70 个训练用户的**全部**真人事件，本方法只有受害者每动作 5 条、共 25 条。其次，轨迹通道约 13–18% 的伪造事件只剩 ≤2 行可自由改写（首行锚点、末行绑定目标），任何两方法都写出逐位相同的坐标，会把轨迹类方法一齐往载体拉近。

## 4. ImagenTime 是唯一进入同一量级的第三方方法

0.683，20/30 格过 0.60。第三方第二名（Diffusion-TS 轨迹臂）0.460，它是其 1.5 倍；同在 IMU 通道上第二名只有 0.325，差距超两倍；它甚至逼近真人对照的 0.773。**「第三方方法都很弱」不成立**，正确说法是：只有「现代扩散 + 图像变换」这一类能接近，且仍比发布版的 0.835 低 0.152。

短板也清楚：behaveformer_stdat 0.869、hmog_style_svm 0.847、paper_svm 0.816 都很高，两个树模型却垮了——hmog_style_rf 0.464、paper_xgboost 0.491，而发布版在同两个检测器上是 0.821 与 0.774。它还被略微低估：**采样用的是早停那一刻的模型而非历史最优检查点**（代码只有一个反复覆盖的检查点文件）。逐动作停点与 gap 见 `notes/imagentime.md`，完整历史在工作树的 `data7/results/direct100k/baselines/imagentime/<action>/summary_<action>.json`——**该文件不在仓库里**，属 §8 所说的工作树部分。

## 5. 弱的那几个为什么弱

两种方向相反、都能直接量出来的失效（完整对照表在上游实验文档 `EXPERIMENTS_CN.md` §3.7；std 比取自各方法自己的 `summary_*.json`，lag-1 用 `code/convergence.py::lag1` 对已存 `samples_*.npy` 现算）。**本节引用的所有 `summary_*.json` / `samples_*.npy` 都在工作树 `data7/results/direct100k/baselines/` 下、不在仓库里**（§8）：Diffusion-TS 在 `diffts/summary_<action>_<kind>.json`，CSDI 在 `csdi/<action>/summary_<action>_<mode>.json`，ImagenTime 在 `imagentime/<action>/summary_<action>.json`，TTS-GAN 在 `ttsgan/summary_<action>_imu.json`。

- **Diffusion-TS：方差不足。** 时间连续性完美（lag-1 自相关 0.951–0.996，与真人几乎重合），离散程度却系统性偏小——六通道 std 比值中位数只有真人的 0.561–0.762 倍，陀螺仪最重，swipe 0.477、keystroke 0.298。形状对，动态范围不够。CSDI 同类失效且更重（scroll/keystroke 的加速度通道 0.38–0.44 倍），分数也同档（0.292 对 0.325）。
- **TTS-GAN：几乎没有时间相关性。** 方向正相反——边缘离散度几乎完美（std 比 0.949–1.084），相邻点相关性却远低于真人：scroll 0.729、swipe 0.745（真人 0.996），keystroke 只有 0.364（真人 0.971）。**这是上调预算之后的数字**：低预算时 keystroke 0.388，迭代加 6 倍只到 0.364，不是预算问题；证据在 `data7/results/direct100k/baselines/ttsgan_budget_evidence.json`（同样在工作树，不在仓库里）。
- **ImagenTime 两项都对**（离散度 0.739–1.074、lag-1 与真人差 0.005–0.038），这是它进入同一量级的机制性解释。

另提醒一句：这个区间里 TimeGAN 系列的判别分数**对攻击成功率没有判别力**——它已饱和到与「真人对真人」对照同一量级，差值落在重复噪声里，而这些方法的 FAR 相差 0.336。

## 6. 怎么复现

**前置。**

**⓪ 依赖状态。** 早先版本的本节曾提示 `code/` 缺 6 个未发布模块，照命令逐字执行会在第一步缺件。**发布前已把这 6 个第一方模块一并归档**：`hmog_baseline_common.py`、`export_real_windows.py`、`build_sample_bank_baseline.py`、`ghost_cursor_path.py`、`grid_against_final.sh`、`gpu_slot.sh`。现在 `code/` 内的第一方 import 全部可解析（在 `code/` 目录下 `python -c "import hmog_baseline_common, final_release, covered_modalities, summarise_final, released_generators, convergence, export_real_windows"` 七个模块全通）。

仍然**不随本仓库发布**的只剩两项外部依赖，都不该由本仓库分发：

- **`security_exp`**——发布版自带的检测器与事件处理包，在上游树 `data7/code/direct100k/` 里；`hmog_event_builder.py:55` 与 `score_against_fixed_detector.py:42` 把它硬插进 `sys.path`。它是被评测的那套检测器的实现，属发布版而非本评测代码。
- **`pyppeteer_ghost_cursor`**——PyPI `python-ghost-cursor` 0.1.1 的顶层包（见 ①）。

此外脚本里的绝对路径常量仍写死在作者机器的位置（清单见 ④），换机器要先改。所以准确的说法是：**第一方代码完整、外部依赖两项、路径需改**，不是「跑不起来」。

**① 上游仓库按 §1 表的版本 clone 到 `code/` 目录内部、与脚本同级**（**不是** `code/` 的兄弟位置——放错位置四个训练脚本全部 `ModuleNotFoundError`）。脚本一律用 `Path(__file__).resolve().parent` 定位，目录名写死：`code/DiffusionTS`（`run_diffusion_ts.py:32-34`）、`code/CSDI`（`run_csdi.py:72-74`，另 `:125` 读 `CSDI/config/base.yaml`）、`code/ImagenTime`（`run_imagentime.py:64-66`）、`code/TTSGAN`（`run_tts_gan.py:52-54`）、`code/pyclick`（`build_pyclick_baseline.py:47`）。第六个 `code/pyppeteer_ghost_cursor` **不是 git clone**，是 PyPI `python-ghost-cursor` 0.1.1 wheel 里的同名顶层包（见 §1 那段说明）；本仓库内没有文件 import 它，import 它的是未发布的 `ghost_cursor_path.py`，所以**要跑 ghost-cursor 这一臂就仍然要放它**，只跑其余七项则不需要。

**② 数据。** 原始 HMOG 事件数据与冻结发布版（§8）。

**③ 绑定表 `fake_target_binding_v12.pkl`（下面 5 条构建命令的 `--binding` 必填参数）。** 它不是数据集，是一张缓存表：`{event_id: (orientation_id, gesture_requested_start_px, gesture_requested_end_px)}`，10 万条、每条伪造事件一条，从载体 bundle 自己的 `provenance.jsonl` 中 `label == 1` 记录的 `donor.target_binding` 抽出来，唯一目的是免得每个分片重新解析 145,776 条记录。生成它的那段内联 Python 在上游 `runners/run_pyclick.sh:20-42`（`runners/run_learned.sh:23-45` 是逐字相同的一段），这两个 runner 也不在本仓库；有冻结发布版就能照该段重建（它读的就是发布版每个 bundle 的 `provenance.jsonl`）。它本体在工作树 `data7/results/direct100k/baselines/fake_target_binding_v12.pkl`（7.2 MB），**不在 6.5 GB 冻结包内部**。**文件名里的 `v12` 与被时钟指纹判 FAIL 的载体 `replay_dataset_v12` 无关**，是早期文件名遗留：实测这一份与冻结发布版 `direct100k_final/datasets/{scroll,swipe,keystroke,tap_and_pinch}/provenance.jsonl` 逐条一致（四个 bundle 各 100,000/100,000 全中、无缺失），而同目录下那个无后缀的旧 `fake_target_binding.pkl` 抽查只有 18,682/20,000 对得上——**别拿错那个**。

**④ 硬编码的本机绝对路径，换机器一律先改。** 分两类，两类都要动，只改文档里点名的那两个文件是不够的。

*Python（`code/` 下）*：

| 文件:行 | 变量 | 应改成什么 |
|---|---|---|
| `build_against_final.py:28` | `PYTHON` | 你的解释器绝对路径（它用 `subprocess` 拉起子脚本） |
| `build_against_final.py:29` | `OUTPUT_ROOT` | 构建产物根，即 `$B/final` |
| `final_release.py:38` | `RELEASE` | 冻结发布版根（`direct100k_final`） |
| `final_release.py:47` | `WORKING_RESULTS` | 发布版自身格子所在的工作结果根，即 `$B` 的上一级 |
| `summarise_final.py:34`、`covered_modalities.py:30`、`final_tables.py:43`、`write_baseline_readmes.py:34` | `ROOT` | 同 `$B/final` |
| `final_tables.py:44` | `CROSS` | `$B/crossscore` |
| `write_baseline_readmes.py:35` / `:336` | `RESULTS` / `ABL_CACHE` | `$B` / `$B/ablations` |
| `grid_job_done.py:27` / `:29` | `B` / `BUNDLE_MAP` | `$B` / 发布版的 `datasets/ACTION_BUNDLE_MAP.json` |
| `score_against_fixed_detector.py:42`、`hmog_event_builder.py:55` | `DIRECT` / `sys.path.insert` | 上游树 `data7/code/direct100k/` |
| `released_generators.py:42`（`:122` 的文档字符串同） | `FINAL` | 本仓库根（`imu_gen/final/`） |

*Shell*：三个训练驱动 `run_csdi_all.sh:15-17`、`run_imagentime_all.sh:21-23`、`run_ttsgan_retrain.sh:16-19` 各自在头部写死 `C=`（上游代码目录）、`B=`（工作目录）、`PY=`（解释器），`run_ttsgan_retrain.sh:18` 另有 `S=`（TTS-GAN 的训练语料目录）；三者都 `cd "$C" || exit 1`（分别在 `:24`、`:28`、`:22`），换机器不改就直接 `exit 1`。**它们不读下文定义的 `$C` / `$B`，必须编辑脚本头部。** `../ablation/code/` 下的 `a7_pipeline.sh:17-20`、`critic_pipeline.sh:15-18`、`run_ablation_queue.sh:9-11` 同样写死 `C` / `B` / `PY`（前两个还多一个 `RUNS=`）。

下文 `$C` = `code/`，`$B` = 工作目录，`$SRC` = 真人事件数据集——**这两个变量只对下面写出的 `python` 命令有效**，`bash $C/run_*.sh` 那三行不接受它们（见④）。

**第一步，训练与采样**（每动作一个模型）：

```bash
# Diffusion-TS：5 动作 × {trajectory, imu} = 10 个作业
python $C/run_diffusion_ts.py --dataset-dir $SRC --output-dir $B/diffts \
    --action swipe --kind imu --steps 12000 --samples 4000 --sample-batch 256 --gpu 0

# CSDI：作者自带的 200 epoch，可断点续跑
bash $C/run_csdi_all.sh
python $C/run_csdi.py --real-dir $B/real_windows --output-dir $B/csdi/tap \
    --action tap --mode unconditional --epochs 200 --samples 4000 --gpu 0

# ImagenTime：作者全额 1000 epoch，--eval-every 25 --patience 3 为早停
bash $C/run_imagentime_all.sh
python $C/run_imagentime.py --real-dir $B/real_windows --output-dir $B/imagentime/tap \
    --action tap --epochs 1000 --samples 4000 --gpu 0

# TTS-GAN：上调后的预算（swipe 60000 / keystroke 48000 / tap 45000 次迭代）
bash $C/run_ttsgan_retrain.sh
python $C/run_tts_gan.py --dataset-dir $SRC --output-dir $B/ttsgan \
    --action swipe --kind imu --max-iter 60000 --samples 4000 --sample-batch 500 \
    --gpu 0 --checkpoint-every 10
```

**第二步，合库与构建**（对发布版四个 bundle 逐个原位替换并验证）：

```bash
# Diffusion-TS 输出在一个扁平目录，用 --samples-dir；双通道臂给两个 --kind
python $C/assemble_banks.py --samples-dir $B/diffts --out $B/bank_diffts_imu.pkl --kind imu
python $C/assemble_banks.py --samples-dir $B/diffts --out $B/bank_diffts_both.pkl \
    --kind trajectory --kind imu

# CSDI / ImagenTime 每动作一个子目录，逐个点名（--sample 可重复）
python $C/assemble_banks.py --kind imu --out $B/bank_final_imagentime.pkl \
    --sample tap=$B/imagentime/tap/samples_tap_imu.npy   # …其余四个动作同理

python $C/build_against_final.py --method imagentime --builder sample_bank \
    --banks $B/bank_final_imagentime.pkl --binding $B/fake_target_binding_v12.pkl \
    --kind imu --workers 24
python $C/build_against_final.py --method diffts_both --builder sample_bank \
    --banks $B/bank_diffts_both.pkl --binding $B/fake_target_binding_v12.pkl --kind both

# 两个光标库无需训练，直接构建；各自内部声明 keystroke declined
python $C/build_against_final.py --method pyclick --builder pyclick \
    --binding $B/fake_target_binding_v12.pkl --kind trajectory --workers 24
python $C/build_against_final.py --method ghostcursor --builder ghostcursor \
    --binding $B/fake_target_binding_v12.pkl --kind trajectory --workers 24

# 构建已内置逐 bundle 的验证（--skip-verify 可跳过），也可单独跑
python $C/verify_harness.py --source-dir <bundle 源> --built-dir <构建产物> --kind imu
```

**第三步，出表。** `summarise_final.py` 按 bundle manifest 只取该 bundle 拥有的动作、拼成 90 格；`final_tables.py` 出三张主表；`score_against_fixed_detector.py` 出迁移表（纯推理）：

```bash
python $C/summarise_final.py imagentime
python $C/final_tables.py --out RESULTS.md
python $C/score_against_fixed_detector.py --attack-root $B/final/imagentime \
    --out $B/crossscore/imagentime.json
```

把构建好的数据集训成 90 格的网格驱动脚本不在本目录，见 §7。

## 7. 仓库里有什么、没有什么

`code/` 的 30 个文件都是真跑过的那一份。第一方模块已在发布前补齐（§6 前置⓪），仍需自备的是两项外部依赖与写死的路径常量。`notes/<方法>.md` 共 9 份（八个基线 + `control_genuine`），由 `code/write_baseline_readmes.py` 从构建产物自动生成——`release.json` 的 `method_detail`、样本旁的训练摘要、`bundle_manifest.json` 的逐动作替换计数——因此不会与实际跑的东西漂移。**一处例外**：`notes/csdi_unconditional.md` 里讲五-shot 条件臂的那一条是脚本里**硬编码的文字**、不是从产物读出的，所以它得靠人工维护。原文写的是 "The five-shot arm was dropped after measurement"，与日志不符；现已在 `write_baseline_readmes.py` 里改成「从未跑完、只有 tap 与 pinch 出样本、无 bank 无 bundle 无格子」并重新生成该 note（见 §9）。

**以下曾经缺失的第一方依赖，发布前已归档进本目录**（原在上游代码树 `data7/code/baselines/`）：

- `hmog_baseline_common.py`——`run_diffusion_ts.py:36`、`run_csdi.py:336`、`run_tts_gan.py:56`、`build_pyclick_baseline.py:35`、`build_ghostcursor_baseline.py:33`、`verify_harness.py:38`、`hmog_event_builder.py:66` 都 import 它。
- `build_sample_bank_baseline.py`——`--builder sample_bank` 实际调用的那个（`build_against_final.py:80`）。
- `ghost_cursor_path.py`——ghost-cursor 的移植实现与两处移植更正，同时是唯一 import `pyppeteer_ghost_cursor` 的文件（§1）。
- `export_real_windows.py`——产出 `$B/real_windows`，CSDI 与 ImagenTime 的 `--real-dir` 指向它。
- `gpu_slot.sh`——三个训练驱动 `run_csdi_all.sh:23`、`run_imagentime_all.sh:27`、`run_ttsgan_retrain.sh:21` 会 `source` 它（注意第三个文件名不带 `_all`）；上游 `run_ablation_queue.sh:21` 同。
- `grid_against_final.sh`——检测器网格驱动。

另有一棵上游树 `data7/code/direct100k/` 同样不发布：`hmog_event_builder.py:55` 与 `score_against_fixed_detector.py:42` 把它硬插进 `sys.path`（前者取 `security_exp.*`）。

**三步都缺件，不是只有第二、三步**：第一步缺 `hmog_baseline_common.py`（Diffusion-TS / CSDI / TTS-GAN 三个训练脚本 import 它）与 `export_real_windows.py`（CSDI、ImagenTime 的 `--real-dir` 靠它产出）；第二步缺 `build_sample_bank_baseline.py`、`ghost_cursor_path.py`；第三步缺 `grid_against_final.sh`。（另：`--builder ablation_cache` 调用的 `build_ablation_cache_baseline.py` 在本仓库、但在 `../ablation/code/` 下，须复制到 `code/` 同级；对比实验本身不走这个 builder。）

## 8. 大数据在哪里

两份大件**不在仓库里**：**冻结发布版 6.5 GB**（`data7/direct100k_final/`，四个 bundle——`tap_and_pinch` 拥有 tap 与 pinch，`scroll`、`swipe`、`keystroke` 各拥有一个动作；含 90 个已拟合检测器与各自的 FRR=5% 操作点）；**构建产物 118 GB**（`data7/results/direct100k/baselines/final/`，18 方法 × 4 bundle 的替换后数据集与逐格结果）。

§4、§5 引用的 `summary_*.json`、`ttsgan_budget_evidence.json`、§2 引用的 `crossscore/*.json` 与 `gridqueue_*.log` 也都在同一棵工作树 `data7/results/direct100k/baselines/` 下，同样不在仓库里。

**没有它们仍能做**：读全部逐格分数、复算任何均值/中位/≥0.60 计数、审计各基线改动记录、审读采样与替换代码、按 commit clone 上游重跑**第一步**（前提是先补齐 §6 前置⓪列的缺件）并与 `notes/` 记的预算、停点、std 比、lag-1 对照。**做不到**：第二步原位替换（需发布版 bundle，以及 `fake_target_binding_v12.pkl`——它在工作树里，或按 §6 前置③那段脚本从发布版 `provenance.jsonl` 重建）、第三步网格与三张表（需 118 GB 产物或重训 90 个检测器）。从零复现整条链需要原始 HMOG 数据、约 150 GB 磁盘与双卡 GPU 数日机时。

## 9. 已知的未完成项

- **Diffusion-TS 双通道臂只算了联合模态的 30 格**，其轨迹与 IMU 单模态共 **60 格未计算**（各 30 格；`notes/diffts_both.md` 的 "Not yet computed" 与表下脚注原样标注）。它两条通道都换了，这两个模态本就在它的供给范围内，所以是「没跑」，不是「不报」。
- **CSDI 五-shot 条件臂未跑完**：只有 tap、pinch 出样本，无 bank、无 bundle、无格子。因此**没有任何第三方基线拿到与本方法相同的受害者条件信息**（`notes/csdi_unconditional.md` 原先那句 "dropped after measurement" 与此矛盾，已按本条更正）。
- **除 Diffusion-TS 外的三个学习型基线未在原论文语料上复现过**（Diffusion-TS 复现了作者的 Sines 基准：0.0108 ± 0.0061 对论文 0.006 ± 0.007）。对 CSDI、ImagenTime、TTS-GAN，「它们弱是因为接法有问题」这个竞争解释**未被排除**。
- ghost-cursor 的退化路径（起终点重合）事件数**未计数**，产物里没有这一项。
- 消融侧的 **A11（`abl_a11_no_feature_match`）已跑完**（12 格、均值 0.717、配对差 +0.072）；属消融实验，不在本表内，见 `../ablation/README.md`。
