# 主实验：发布版自身的 90 格成绩

本目录只回答一个问题：**本文方法（下称「发布版」）伪造的行为事件，能被针对它训练的检测器接受多少。**

威胁模型是**五-shot**：攻击者只有受害者的五条真实事件（`k_refs = 5`），没有该用户的其他数据，也不读任何检测器分数、标签或真人总体统计。

判据：**在开发集上选定、测试集前冻结的 FRR=5% 阈值处的 FAR**，越高攻击越强，成功线记为 0.60。注意每格产物默认写出的 `far` 字段是 **EER 阈值**上的值，两个判据在本项目上给出方向相反的结论（同一 90 格，EER 口径均值 0.386、0 格过线），不要混用。口径细节见 `EXPERIMENTS_CN.md` 第 1.4 节。

评测单元是**格子**＝动作 × 模态 × 检测器，5 × 3 × 6 = 90 格全部跑出了结果，无缺格。

## 成绩

全 90 格：**均值 0.775、中位 0.794、77/90 ≥ 0.60**（最小 0.482，最大 0.961）。

| 模态 | 格子 | 均值 | 中位 | ≥0.60 |
|---|---|---|---|---|
| `trajectory_xytime` | 30 | 0.777 | 0.779 | 26 |
| `imu_only` | 30 | 0.835 | 0.840 | 30 |
| `imu_trajectory_xytime` | 30 | 0.711 | 0.734 | 21 |

联合模态是最难骗的一档，比两个单通道都低。逐动作 × 逐模态（六个检测器均值）：

| 动作 | 轨迹 | IMU | 联合 |
|---|---|---|---|
| tap | 0.901 | 0.881 | 0.820 |
| scroll | 0.685 | 0.726 | 0.607 |
| swipe | 0.689 | 0.852 | 0.653 |
| pinch | 0.729 | 0.883 | 0.721 |
| keystroke | 0.883 | 0.833 | 0.756 |

keystroke × 轨迹一格恰好落在进位边界上（六格精确均值 0.88350）：本表按下面复算片段的口径写成 0.883，手工四舍五入会得到 0.884，两者说的是同一个数。

IMU 模态逐检测器（五动作均值），跨度 0.09：

| hmog_style_svm | hmog_style_rf | paper_svm | paper_xgboost | behaveformer_stdat | authconformer |
|---|---|---|---|---|---|
| 0.845 | 0.821 | 0.867 | 0.774 | 0.868 | 0.835 |

铺满 90 格是必要的：只看 tap（轨迹 0.901）或只看 scroll（轨迹 0.685）会得到相反印象。

## `main/scores/release_90_cells.csv`

90 行 + 表头，四列：`action`（tap/scroll/swipe/pinch/keystroke）、`modality`（三个模态名如上）、`detector`（六个检测器名如上）、`far_at_frr5`（该格的 FAR，六位小数）。

**本目录所有命令都从 `evaluation/` 下执行，文件路径一律相对 `evaluation/`**，与 `../README.md`、`../ablation/README.md` 的约定一致（因此本节写 `main/scores/...` 而不是 `scores/...`）。上面三张表全部可由这一个文件重算：

```bash
cd path/to/repo/evaluation
python3 - <<'PY'
import csv, statistics

rows = list(csv.DictReader(open("main/scores/release_90_cells.csv")))
far = lambda rs: [float(r["far_at_frr5"]) for r in rs]
MODS = ["trajectory_xytime", "imu_only", "imu_trajectory_xytime"]
DETS = ["hmog_style_svm", "hmog_style_rf", "paper_svm",
        "paper_xgboost", "behaveformer_stdat", "authconformer"]

for m in MODS:                                              # 表一：逐模态
    x = far([r for r in rows if r["modality"] == m])
    print(f"{m:24s} n={len(x)} mean={statistics.mean(x):.3f} "
          f"median={statistics.median(x):.3f} >=0.60={sum(y >= 0.60 for y in x)}")

for a in ["tap", "scroll", "swipe", "pinch", "keystroke"]:  # 表二：逐动作 × 逐模态
    cells = [statistics.mean(far([r for r in rows if r["action"] == a
                                  and r["modality"] == m])) for m in MODS]
    print(f"{a:10s}" + "".join(f"{c:8.3f}" for c in cells))

for d in DETS:                                              # 表三：IMU 逐检测器
    x = far([r for r in rows if r["modality"] == "imu_only" and r["detector"] == d])
    print(f"{d:20s} {statistics.mean(x):.3f}")

v = far(rows)
print(len(v), round(statistics.mean(v), 3), round(statistics.median(v), 3),
      sum(y >= 0.60 for y in v))
PY
```

输出逐行对应上面三张表，最后一行是 `90 0.775 0.794 77`。只用标准库（`csv` + `statistics`），任何 `python3` 都能直接跑，不需要 pandas，也不需要大数据。

## 代码与数据在哪

| 要看什么 | 去哪 |
|---|---|
| 上游轨迹提取 / 变长条件 diffusion / 独立 PAD | 仓库根 `trajectory_humanization_full_20260722_v16_numeric_recovery/` |
| 检测估计器封装（feature / deep / total detector）、配对 IMU 链路 | 仓库根 `trajectory_estimator_pack_20260721/` |
| IMU 统一发布接口（cache / online 双后端，fail-closed 门禁） | 仓库根 `android_duration_time_fixed_20260720/imu_release_20260721/` |
| user-disjoint 与 reference-exclusion 补充评测 | 仓库根 `trajectory_pad_supplement_20260722/` |
| **本实验的事件构造器与检测器运行器** | 冻结发布版 `code/generation/`、`code/dataset_test/`（入口 `scripts/run_hmog_direct100k_detectors.py`） |

冻结发布版在 **`/mnt/share/mwang49/data7/direct100k_final/`，6.5 GB**：`datasets/` 3.9 GB（按 tap_and_pinch / scroll / swipe / keystroke 四个 bundle 组织）、`detector_models/` 2.5 GB（90 个格子各一份拟合好的模型）、`generator_checkpoints/` 117 MB（五个载体扩散检查点）、`provenance.json`（逐格记录来源）。逐格 `thresholds.json` / `summary.json` / `pre_test_freeze_receipt.json` / `test_scores.jsonl.gz` 在冻结包内的 `code/dataset_test/results/cells/` 下（本段路径相对冻结包根目录，不是相对 `evaluation/`）。

**不在仓库里、也不在冻结包里**的是完整实验工作树 `/mnt/share/mwang49/data7/results/direct100k/`（撰写时实测 179 GB，其中 `baselines/` 子树 142 GB；A11 完成后略有增长）。

**没有这些大文件你能做什么**：用本目录的 CSV 复算全部汇总表与任何自定阈值下的过线计数；读 `code/docs_zh/方法.md`、`code/docs_en/REPRODUCE.md` 核对协议。**不能做什么**：重跑检测器、重新选阈值、逐事件核对替换。替代路径是从冻结包出发（`cd /mnt/share/mwang49/data7/direct100k_final/code/dataset_test/results && python summarise.py 0.48` 从已发布分数重算主表），而不是从 HMOG 原始数据重建——后者需要 100 用户全量语料与多卡 GPU。

## 与另外两部分的关系

`../comparison/` 用**同一批真人事件、同一套载体**评测八个第三方基线：只替换该方法真正生成的那条通道，其余内容继承载体，因此只在改动过的模态上报数（pyclick、ghost-cursor 不建模 keystroke，记为 **declined（拒绝报数）**，分母 24 格）。`../ablation/` 拆的是发布版自己的部件（k_refs、对抗训练、梯度合并、三个 critic、直接特征匹配损失），只在 IMU 通道、keystroke 由解析适配器写成故排除；**A1–A11 十一个臂已全部落地**，`RESULTS.md` 表 2 里没有任何一行标 *not finished*。本目录只提供发布版自身的 90 格，消融数字见 `../ablation/`。
