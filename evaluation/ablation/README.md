# 消融实验：A1–A11

本方法（下称**发布版**）自身的部件消融。判据是**开发集选定的 FRR=5% 阈值上的 FAR**，越高攻击越强；一格（cell）= 动作 × 模态 × 检测器，每格检测器都在该臂的伪造数据上重训。所有臂只改惯性生成器，故只报 `imu_only`。

## 目录内容

| 路径 | 内容 |
|---|---|
| `scores/ablation_cells.csv` | 10 个已完成臂的逐格分数，180 行，列 `method,action,modality,detector,far_at_frr5` |
| `notes/` | 10 份逐臂说明：`abl_noshot_adv.md`、`abl_fewshot_nonadv.md`、`abl_krefs1/3/8.md`、`abl_a7_weighted_sum.md`、`abl_a8_no_feature.md`、`abl_a9_no_set.md`、`abl_a10_no_waveform.md`、`abl_a11_no_feature_match.md`。A1 即发布版本身，无单独说明 |
| `code/` | `generate_imu_ablation.py`、`run_ablation_queue.sh`、`run_a7_weighted_sum.py`、`a7_pipeline.sh`、`run_critic_ablation.py`、`critic_pipeline.sh`、`build_ablation_cache_baseline.py`、`verify_reconstruction.py` |

## 十一个臂

协议名两段拼接：`noshot`/`fewshot` 表示采样时给不给参考样本，后缀 `_adv` 表示训练时开不开对抗目标。

| 臂 | 拿掉/改掉什么 | 协议 | 重训生成器 |
|---|---|---|---|
| A1 | 无，即发布版本身 | `fewshot_adv`，k_refs=5 | — |
| A2 | 五-shot 条件（k_refs=0） | `noshot_adv` | 否，该协议检查点已存在 |
| A3 | 整套对抗训练 | `fewshot` | 否，同上 |
| A4/A5/A6 | k_refs 改为 1 / 3 / 8 | `fewshot_adv` | 否，与 A1 共用检查点 |
| A7 | 梯度层面合并 → 标量加权相加 | `fewshot_adv` | **是** |
| A8/A9/A10 | `adv.critics.feature` / `.set` / `.waveform` 之一 | `fewshot_adv` | **是** |
| A11 | 直接特征匹配损失（`adv.feature_match_weight`→0） | `fewshot_adv` | **是** |

## 完整结果表

**两种口径不可混。** A2–A6 是 4 动作 × 6 检测器 = 24 格；A7–A11 只重训了 scroll 与 swipe，是 12 格。A1 的同格基准分别为 **0.835**（24 格）与 **0.789**（12 格，精确值 0.7887），不可互相相减。**「配对差」逐格配对后再平均，跨臂可比**；均值只应在格子数相同的臂之间读。

| 臂 | 拿掉什么 | 格 | 均值 | 对 A1 配对差 | ≥0.60 |
|---|---|---|---|---|---|
| **A1（24 格）** | 无 | 24 | **0.835** | — | **24 / 24** |
| A2 | 五-shot | 24 | 0.779 | +0.056 | 21 |
| A3（24 格） | 整套对抗 | 24 | 0.802 | +0.034 | 21 |
| A4 | k_refs=1 | 24 | 0.798 | +0.038 | 23 |
| A5 | k_refs=3 | 24 | 0.832 | +0.004 | 23 |
| A6 | k_refs=8 | 24 | 0.830 | +0.005 | 23 |
| **A1（12 格）** | 无（scroll+swipe） | 12 | **0.789** | — | **12 / 12** |
| A3（12 格） | 整套对抗（限定 scroll+swipe） | 12 | 0.744 | +0.045 | 9 |
| A7 | 梯度合并→加权和 | 12 | 0.734 | +0.054 | 9 |
| A8 | feature critic | 12 | 0.682 | +0.107 | 8 |
| A9 | set critic | 12 | 0.716 | +0.073 | 9 |
| A10 | waveform critic | 12 | 0.706 | +0.083 | 9 |
| A11 | feature_match | 12 | 0.717 | +0.072 | 9 |

**A1 与 A3 各占两行**，不是两个臂：同一个臂分别在 24 格（tap/scroll/swipe/pinch）与 12 格（scroll+swipe，与 A7–A11 同格）上重算，各自只跟同格数的臂配对。两行之间**不可互减**（`EXPERIMENTS_CN.md` §5.7 是同一套读法）。

A1 是唯一全部格子越过 0.60 的臂。**A5–A6** 与 A1 理论上应一致（k_refs 在 3 处已饱和），实测配对差 +0.004 / +0.005，即「什么都没真变」时的抖动约 **0.005**（A4 不在此列：k=1 远未饱和，配对差 +0.038）；但它只含采样随机性，**重训噪声未测量**（没做多种子重复），故只是下界。A4–A6 复用 A1 的检查点，测的是参考编码器的泛化，不是「用 k 条参考从头训练」。

## keystroke 按构造排除

分母是 24 而非 30：keystroke 的伪造 IMU 由解析式适配器写出（产物记为 `diffusion_used: false` / `model_used: false`，生成源 `security_exp/keystroke_imu_pulse.py`），**从不经过扩散生成器**，再多算力也够不着。建库脚本 `build_ablation_cache_baseline.py` 因此硬编码 `ALWAYS_DECLINED = frozenset({"keystroke"})`——填 0 会被读成「这个部件对 keystroke 没帮助」，事实却是「消融够不着它」。另需区分：A7–A11 的 tap 与 pinch 属「未重训（算力预算）」，而不是 declined（拒绝报数）——后者指方法**按构造**产不出这个动作（同属 `tap_and_pinch` 一个 bundle，重训按 bundle 走，要么整体做要么整体不做）。

## A7：最重要的一个臂

**被消融的机制。** 发布版有重建与对抗两个目标。常规做法是两个标量损失加权相加再反传；发布版在**梯度层面**合并，由上游 `final_gen/train.py::backward_with_projected_adversarial_gradients` 做三件事：重建损失单独反传、留下干净的 `g_recon` 作参照系；仅当 `dot(g_adv, g_recon) < 0` 时投影掉冲突分量；把余下的对抗贡献每步重新钉到 `max_grad_ratio · ‖g_recon‖`。

**A7 不改上游一行代码。** `run_a7_weighted_sum.py` 只把 `project_conflicts` 置 `false`、`max_grad_ratio` 顶到 `1e9`（常量 `UNCAPPED_RATIO`），作者自己的合并路径便**在代数上退化**成加权相加：`projection_coeff = 0`、`merge_scale = clamp(inf, max=1) = 1`，于是 `p.grad = g_recon + g_weight · g_adv`。`g_weight` 原样保留（scroll/swipe 均 0.04，`max_grad_ratio` 原值 0.5，两个不同的量别对调）。日志统计：投影只在 0–25% 的步上触发、多数动作 <1%，范数上限却**每步都起作用**，中位 `merge_scale` 为 0.005–0.07，把对抗梯度压小 15–200 倍——A7 关掉的主体是那道上限。

**只看均值会错过全部信息**（逐检测器，scroll+swipe 严格同格；CSV 里的 id 依次是 `hmog_style_svm`、`hmog_style_rf`、`paper_svm`、`paper_xgboost`、`behaveformer_stdat`、`authconformer`）：

| | HMOG-SVM | HMOG-RF | Paper-SVM | Paper-XGB | BehaveFormer | AuthConformer |
|---|---|---|---|---|---|---|
| A1（scroll+swipe） | 0.859 | 0.772 | 0.852 | 0.725 | 0.775 | 0.749 |
| A7 | **0.901** | **0.581** | 0.824 | **0.607** | 0.859 | **0.634** |
| A8 | 0.865 | **0.564** | 0.794 | **0.548** | 0.704 | 0.617 |
| A9 | 0.867 | **0.623** | 0.807 | **0.571** | 0.802 | 0.627 |
| A10 | 0.879 | **0.585** | 0.816 | **0.561** | 0.795 | 0.600 |
| A11 | 0.873 | **0.614** | 0.812 | **0.584** | 0.784 | 0.633 |

**树模型崩，SVM 不降反升。** A7 上 HMOG-RF 0.772→0.581、Paper-XGB 0.725→0.607、AuthConformer 0.749→0.634；而 HMOG-SVM 反升到 0.901，Paper-SVM 微降到 0.824，BehaveFormer 微升到 0.859。这不是「整体变差」，而是加权相加让某类**结构性线索重新暴露**：线性 SVM（`hmog_style_svm` 即 `LinearSVC`）只看一个线性方向，轴对齐树模型却能对单个统计量切区间。**只报一个检测器，结论会完全反过来。**（`RESULTS.md` 的同一张逐检测器表已按口径分块：A1–A6 标「four, 24 cells」，A7–A11 标「scroll+swipe, 12 cells」，并在 A7 上方重列 12 格的 A1 基准；`EXPERIMENTS_CN.md` §4.6 与 §5.7 用的是同一行。）

## A8–A11：部件之和远大于整体

原计划是把 A3 的效应拆成四份，实测拆不出来：同样 12 格上，**整套对抗全关只损失 0.045，单独关掉任一部件却损失 0.072–0.107**，四臂之和 0.334——与整体 **差着一个量级**。方向性解释（非结论）：四样不是可加的贡献项，去掉一个会让剩下三个失衡，而**失衡的对抗目标比没有对抗目标更有害**。

**限制与数字一起读。** 这批数据**能**支持「没有单一部件可以被去掉」：四臂极差只有 0.035，而它们对同格基准 0.789 的差距是 0.072–0.107，量级一致。**不能**支持四者之间的精细排序：相邻两臂（A9 +0.073 与 A11 +0.072）只差 0.001，相隔最远的 A8 与 A11 也只差 0.035，而重训噪声从未测量、门槛只会比 0.005 更高。**具体倍数（0.334 / 0.045）建立在两个都带着未测量重训方差的数之上，不应被引用**，只有「差着一个量级」这个定性结论成立。

## 为什么是四个臂，不是三个

`adv.critics` 底下只有三个开关，但训练器里还有第四样——`feature_match`。它**不在** `adv.critics` 里，只由 `adv.feature_match_weight > 0` 控制，却位于**同一个 `adv_update` 块**内，其加权损失与三个 critic 的损失汇入同一条梯度、过同一套投影与上限。所以 A3 关掉的是四样。只做三个 critic 臂，整体效应会有一块无人认领；更糟的是 A8 测出的是**偏小的假效应**——critic 关了，直接匹配损失还在供给同一批统计量。（实测 A8 的 +0.107 反而最大，与此方向相反；上述论证给的是**下界**而非排名。A11 落地后实测 **+0.072**，与另外三臂同量级——feature 通路的另一半确实携带可观效应，只做三个 critic 臂会整块漏掉它；但四臂极差只有 0.035，主导的更可能是失衡效应而非单通路效应。）

映射写在 `run_critic_ablation.py` 的 `ARMS` 表。四臂底座是**发布版那次运行自己的 `effective_config.json`**，梯度保护保持开启、只翻一个开关；脚本有硬断言，底座若 `project_conflicts` ≠ `True` 或 `max_grad_ratio` ≠ `0.5` 就 `SystemExit`（「这看起来是 A7 的配置而不是发布版的」）。

## 怎么复现

### 只用本仓库（无需大数据与 GPU）

上表每个数字都能从两份 CSV 重算：

```bash
cd <repo>/evaluation
# 各臂均值与 ≥0.60 计数
awk -F, 'NR>1{s[$1]+=$5;n[$1]++; if($5>=0.6)c[$1]++} END{for(m in s) printf "%-24s n=%d mean=%.3f >=0.6=%d\n", m,n[m],s[m]/n[m],c[m]}' ablation/scores/ablation_cells.csv | sort
# 对 A1 的逐格配对差
awk -F, 'FNR==NR{if(FNR>1) r[$1","$2","$3]=$4; next} FNR>1{k=$2","$3","$4; if(k in r){d[$1]+=r[k]-$5; n[$1]++}} END{for(m in d) printf "%-24s %+.3f over %d\n", m,d[m]/n[m],n[m]}' main/scores/release_90_cells.csv ablation/scores/ablation_cells.csv | sort
```

参考值 `main/scores/release_90_cells.csv` 即发布版 90 格，A1 两个基准也由它得出：`imu_only` 且动作 ≠ keystroke 得 24 格 0.8355，再限定 scroll+swipe 得 12 格 0.7887。

### 需要大数据与算力

`code/` 下脚本原样归档，含**机器绝对路径**（`/mnt/share/mwang49/...`，外加一处硬编码的 conda 解释器路径——三个 `.sh` 的 `PY=`、`run_a7_weighted_sum.py:59` 与 `run_critic_ablation.py:55` 的 `PYTHON=`），换机器须先改。外部依赖分三类，缺哪一类都跑不动：

**① 在本仓库里，但不在 `ablation/code/` 下。** 这些脚本按 `Path(__file__).resolve().parent` 解析依赖，所以要求**同目录**，运行前须复制或软链过来：

- `released_generators.py` / `released_generators.json`（钉住发布版的 run 与检查点）在 `../comparison/code/`。`generate_imu_ablation.py:37,58`、`run_a7_weighted_sum.py:80-81`、`run_critic_ablation.py:81-82` 都是先 `sys.path.insert(0, Path(__file__).resolve().parent)` 再 `from released_generators import resolve`。这三处 import 都在函数体内，所以 `--help` 照样能过，**真去解析发布版 run 时才 `ImportError`**。
- `build_against_final.py`（以及它调用的 `verify_harness.py`）同样在 `../comparison/code/`，而它是用自己的 `BASE = Path(__file__).resolve().parent` 找子脚本的：`--builder ablation_cache` 走的正是 `BASE / "build_ablation_cache_baseline.py"`。所以这两个文件也必须放在一起。

**② 完全不在本仓库**，在上游代码树 `data7/code/baselines/`（与 `../comparison/README.md` §7 是同一份清单）：

- `hmog_baseline_common.py`——`build_ablation_cache_baseline.py:73` 与 `verify_reconstruction.py:36` 直接 import 它。缺它这两个脚本**连模块 import 都过不去**（`ModuleNotFoundError: No module named 'hmog_baseline_common'`），下面「建库与校验」那两条命令在归档树上必然跑不起来。须把它放到脚本同目录，或加进 `PYTHONPATH`。
- `gpu_slot.sh`——`run_ablation_queue.sh` 会 `source` 它。

**③ 上游 `data7/code/direct100k/`。** `build_ablation_cache_baseline.py:71` 与 `verify_reconstruction.py:29` 把这个路径硬编码进了 `sys.path`，为的是 `security_exp.fiveshot_gesture_timing.carrier_window_imu`（padded-window 那几个动作的载体切窗）。注意它只提供 `security_exp/`，`hmog_baseline_common.py` **不在**这里，在 ② 说的 `baselines/` 下。

三个 `.sh` 还都写死了 `C=/mnt/share/mwang49/data7/code/baselines` 并 `cd "$C"` 后运行——它们原本就在上游那棵树里跑，上述文件在那里本来同处一个目录；归档时按用途拆进了 `ablation/code/` 与 `comparison/code/`，同目录关系随之断掉。

- **A2–A6（只采样）**：`python generate_imu_ablation.py --protocol noshot_adv --actions tap scroll swipe pinch --out-dir <cache> --num-shards 8 --shard-index <i> --gpu <g>`；k_refs 臂把 `--protocol` 换成 `fewshot_adv` 并加 `--k-refs 1|3|8`。`run_ablation_queue.sh` 是五臂的串行驱动。
- **A7（必须重训）**：`python run_a7_weighted_sum.py --actions scroll swipe --gpu 0`（先加 `--dry-run` 只写配置、打印命令）。它生成的训练命令形如 `python -m final_gen.train train --config <yaml> --action scroll --method diffusion --protocol fewshot_adv --adv true --run-name scroll_a7_weighted_sum`；训练完由 `a7_pipeline.sh` 无人值守采样 → 建库 → 排格。
- **A8–A11（必须重训）**：`python run_critic_ablation.py --arms A8 A9 A10 A11 --actions scroll swipe --gpu 0`，随后 `critic_pipeline.sh`（每动作至少 `NEEDED=20000` 个样本才建库，短了宁可报错）。
- **建库与校验**：`python build_against_final.py --method abl_a8_no_feature --builder ablation_cache --cache-root <cache> --kind imu --workers 24`。消融数据集是把该臂的 IMU **逐槽替换**进发布版副本；该规则由 `verify_reconstruction.py` 对着发布版自己的缓存重放、要求与已发布 IMU **逐位相同**，建库前先跑。

**成本。** GPU 空闲时实测 162 epoch/小时，一个重训臂全程约 **21 小时**（训练 4h + 采样 13h + 排格 4h）——这是开跑前的预算口径；critic 那批实测为 8 个训练运行（4 臂 × 2 动作、两两并发）约 9.3 小时，其后每臂「采样 + 建库 + 排格」约 4.7 小时。tap/pinch/keystroke 没重训正因这个成本，**两动作的结论能否外推未验证**。

## 大数据不在仓库里

- **冻结发布版**：`/mnt/share/mwang49/data7/direct100k_final/`，实测 **6.5 GB**（`datasets/` 3.9 GB，另有 `detector_models/`、`generator_checkpoints/`、`provenance.json`）。所有臂以它为底座做原位替换。
- **替换后的数据集与逐格结果**：`/mnt/share/mwang49/data7/results/direct100k/baselines/final/`，约 **118 GB**（18 个方法目录，与 `DATA.md`、`../comparison/README.md` §8 同一口径）。
- **完整实验产物树**（上一条再加消融缓存、样本 bank、日志、A11 采样）：其父目录 `/mnt/share/mwang49/data7/results/direct100k/baselines/`，同一快照下实测 **142 GB**；再往上整个 `results/direct100k/` 是 179 GB。这三个数取自 A11 落地前后的同一次快照，仅供量级参考。
- **没有它们能做**：复算本页所有表、审阅 `code/` 与 `notes/`。**不能做**：重跑任何一格检测器、重训任何一个臂、逐位校验替换规则。

## A11 已落地（如实记录时序）

A11 的两个训练运行随该批 8 个运行于 2026-08-08 23:17 训完，采样自 08-09 09:28 开始，成稿当天完成建库与排格，12 格全部取到：均值 **0.717**、配对差 **+0.072**、9/12 格 ≥ 0.60，`RESULTS.md` 表 2 与 `notes/abl_a11_no_feature_match.md` 都已收录，没有任何一行再标 *not finished*。**「为什么是四个臂」那段闭合因此从论证上的变成实测上的**：此前只有日志证据（A11 三个 `adv_acc_*` 键都在、`adv_feature_match_*` 整组消失，A8/A9/A10 反之），现在有了配对数字。

---

延伸阅读：`EXPERIMENTS_CN.md` 第 4 节、第 5 节、第 7.2–7.5 节；主表 `RESULTS.md`；第三方对比 `../comparison/`。
