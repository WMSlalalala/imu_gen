# 数据：在哪、为什么不在仓库里、没有它能做什么

本仓库只装文本：代码、逐格分数 CSV、说明。所有事件数据都在仓库外。下面的绝对路径是作者机器上的位置，用于交叉核对来源；对外分发的位置见文末。

**本文是体积与获取途径的唯一口径。** `main/README.md`、`comparison/README.md`、`ablation/README.md` 里各自复述过的 GB 数与 A11 进度是各自撰写当天的快照，与本文不一致时以本文为准。

## 体积：从 185 GB 到 1.71 MiB，量级差五个数量级

三层是原始归档、生成产物、逐格结果；中间两行是产物层的父目录，单列出来是因为别处引过它们的数字。体积全部为 `du -sh` 实测，快照时刻 **2026-08-09 13:01 EDT**；A11 当时正在写盘（见文末），带 * 的行仍在增长。

| 层 | 路径（口径） | 实测体积 | 含 A11 采样/建库中数据？ | 在仓库里？ |
|---|---|---|---|---|
| 原始 HMOG 归档 | `/mnt/share/mwang49/Human_agent/hmog_dataset.zip` | 6,132,356,276 字节（5.7 GiB） | — | 否 |
| 生成的对比/消融数据集 | `data7/results/direct100k/baselines/final/` | 125 GB*（19 个方法目录，每个 6.4–6.9 GB） | 是，含 `abl_a11_no_feature_match/` 6.4 GB | 否 |
| 基线工作目录（上一行的父目录） | `data7/results/direct100k/baselines/` | 148 GB*（= `final/` 125 GB + `ablations/` 采样缓存 5.5 GB + 各方法中间产物、`bank_*.pkl`、日志） | 是 | 否 |
| 整棵实验工作树 | `data7/results/direct100k/` | 185 GB*（含上一行，另加 `detectors_v*`、`replay_dataset_v*` 等历史构建） | 是 | 否 |
| 冻结发布版 | `data7/direct100k_final/` | 6.5 GB（`datasets/` 3.9 GB、`detector_models/` 2.5 GB、`generator_checkpoints/` 117 MB、`code/` 26 MB） | 否，冻结不再变 | 否 |
| 逐格结果 | 逐格 `summary.json`（90 个发布版格子在 `detectors_90cell/`，其余基线与消融格子在 `baselines/final/*/cells_*/`）；快照时是 516 个，A11 的 12 格落地后为 528 个 | 1.71 MiB（516 格时） | A11 落地后已含 | **是，已提取成 CSV（528 行）** |

125 GB 里绝大部分是 `shards/*.npz` 与 `provenance.jsonl`：每个方法都要把 4 个 bundle 各 100 个分片整份复制一遍，只改被替换的那一条通道，这样"唯一变量"才成立（见 `EXPERIMENTS_CN.md` §2.1）。代价就是体积。

### 复现需要多少磁盘

- **只核对分数、只读代码**：0。仓库本身几 MB，`evaluation/` 全是文本。
- **从冻结发布版出发，重跑一个基线方法**：约 **14 GB**（发布版 6.5 GB + 该方法目录 6.4–6.9 GB）。
- **重建三张主表的全部 19 个方法目录**：约 **132 GB**（6.5 + 125），留出临时文件余量按 **150 GB** 准备。
- **从原始 HMOG 起重建整条链**：作者机器上整棵 `results/direct100k/` 落盘 185 GB，另需归档 5.7 GB 与解压中间产物，按 **200 GB** 准备。

## 为什么不进 git

体积是第一条，第二条是仓库根目录的历史包袱。`.gitignore` 采用**默认拒绝**：先 `/*` 排除一切，再按白名单逐项放行（`!/evaluation/`、`!/evaluation/**` 等）。此外还有一层与白名单无关的硬排除：`**/results/`、`**/*.npz`、`**/*.npy`、`**/*.pt`、`**/*.ckpt`、`**/*.log`。也就是说，即使有人把数据软链进白名单目录，也不会被提交。

## 只有仓库里的 CSV，能做什么

三份 CSV，共 528 行，列都是 `far_at_frr5`（开发集选定的 FRR=5% 阈值上的 FAR）：

- `main/scores/release_90_cells.csv`（90 行）
- `comparison/scores/comparison_cells.csv`（258 行）
- `ablation/scores/ablation_cells.csv`（180 行，仅 `imu_only`，含 A11）

足以：**逐格核对 `RESULTS.md` 的表 1 与表 2 的每一个数字**（已验证：表 1 各行的格子数/均值/中位/≥0.60 计数、表 2 各臂均值与 A1 的 24 格 0.835，全部由这三份 CSV 重算得出）；重画任何图；重算任意子集（按动作、按模态、按检测器）的均值/中位/计数。不需要任何原始数据，几行 `csv` + `statistics` 即可。

一处例外要说清楚：**表 3（迁移）的逐格分数不在仓库里**，它们在 `results/direct100k/baselines/crossscore/*.json`（396 KB）。表 3 目前只能按表级数字核对，不能从仓库重算。

## 必须有原始数据才能做什么

- 重训六个检测器（`detector_models/` 下 90 个已拟合模型，2.5 GB）
- 重新采样任何生成器（`generator_checkpoints/`，五个动作）
- 用 `ablation/code/verify_reconstruction.py` 验证替换规则的**逐位一致性**——它要求重放结果与已发布 IMU **bit identical**，不是"接近"（结果见 §2.2，80,000/80,000）
- 用 `comparison/code/verify_harness.py` 复核原位替换的四项不变量
- `comparison/code/final_tables.py` / `summarise_final.py` 直接读 `results/.../final`，无原始数据无法运行

## 原始数据从哪来

**数据集**：HMOG，100 个匿名用户的手机行为数据。本仓库不转发它，也不复述其使用条款——条款以发布方页面为准，使用前请自行确认。

**本项目实际用的归档**（版本标识就是这两行，请以此核对，不要以文件名核对）：

```text
文件   /mnt/share/mwang49/Human_agent/hmog_dataset.zip
大小   6,132,356,276 字节
SHA-256  4e3f4216ca7c362bd06493301d7ef9634940af69f939fe02689cb3f84c914346
```

**获取途径**：2026-05-29 15:21 EDT 用 `wget` 从发布方的 Box 共享链接下载，完整下载日志留在 `/mnt/share/mwang49/Human_agent/hmog_dataset_official.wget.log`，当时的链接是

```text
https://wm1693.app.box.com/index.php?rm=box_download_shared_file&shared_name=jkjodaw2scunua7b7qaxt3ker5w9lwt7&file_id=f_770692248712
```

这是**当日实际使用并成功的链接**（HTTP 206，Content-Length 6132356276，与上面的大小一致）。Box 共享链接会随发布方的设置变动而失效；失效时请按数据集名 HMOG 回到发布方页面重新索取，拿到后用上面的 SHA-256 核对是否同一版本。**已知缺口**：本仓库目前没有记录 HMOG 的引用文献条目，也没有转述其许可条款文本，上面的 SHA-256 是唯一可核对的版本锚点。

**解压后的目录布局**（实测自该归档）：

```
public_dataset/<6 位用户号>.zip        ← 100 个，一个用户一个
  └─ <用户号>/<用户号>_session_<N>/    ← 例：100669/100669_session_1，该用户 24 个 session
       Accelerometer.csv  Activity.csv  Gyroscope.csv  KeyPressEvent.csv
       Magnetometer.csv   OneFingerTouchEvent.csv      PinchEvent.csv
       ScrollEvent.csv    StrokeEvent.csv              TouchEvent.csv
```

**从原始归档到脚本能吃的格式，两步**：

1. **抽轨迹**：`trajectory_humanization_full_20260722_v16_numeric_recovery/preprocess/extract_hmog_trajectories.py`（**在本仓库里**，已被 `.gitignore` 白名单放行并已 track；`DEFAULT_HMOG_ZIP` 默认指向上面那个 zip，须按自己的路径改）。正式提取当日用的命令与耗时、以及归档大小/SHA 的独立复核，记在同目录 `docs/formal_extraction_provenance_20260713.md`（也在仓库里）：

   ```bash
   python -u preprocess/extract_hmog_trajectories.py \
     --output-dir results/trajectories_full_v2 \
     --max-users 100 \
     --confirm-full-run
   ```

2. **建分片数据集**：`security_exp/replay_dataset_builder.py`。它**随冻结发布版一起发**，在 `direct100k_final/code/generation/security_exp/replay_dataset_builder.py`（作者机器上的工作副本是 `data7/code/direct100k/security_exp/replay_dataset_builder.py`）。它产出 `replay_dataset_v*/`，四个 bundle 各自定稿于哪一个构建，记在 `direct100k_final/provenance.json`：keystroke←`replay_dataset_v10`、scroll←`replay_dataset_v15`、swipe←`replay_dataset_v8`、tap_and_pinch←`replay_dataset_v3`。

**关于 `$SRC` 的一处澄清**：`comparison/README.md` 里基线脚本的 `--dataset-dir $SRC` **不是**原始 HMOG 归档。`hmog_baseline_common.iter_shards()` 直接 glob `<dataset_dir>/shards/*.npz`，所以 `$SRC` 必须是**发布版布局的 bundle 目录**（例如 `direct100k_final/datasets/scroll/`）。也就是说：跑第三方基线只需要冻结发布版，不需要原始 HMOG；原始 HMOG 只在你要从头重建 bundle 时才用得上。

## 想要数据

发布版是**冻结**的，带校验：`event_manifest.jsonl` 中每个分片有 `source_sha256`，`release.json` 中有 `event_manifest_sha256` 与 `provenance_sha256`，`provenance.json` 记录每个 bundle 与每一格来自哪个工作目录。

**获取方式**：6.5 GB 的冻结发布版与 125 GB 的构建产物**暂不公开分发**（尚未上传任何归档站点，因此本文不给下载链接——给了也是死链）。需要的话请在本仓库 GitHub 提 issue 说明用途与所需部分（整份发布版 / 某个动作的 bundle / 某个方法目录 / 只要 `crossscore/*.json`），仓库地址 `https://github.com/WMSlalalala/imu_gen`。逐格分数不必申请，三份 CSV 已在 `*/scores/` 里。

## 许可与第三方代码

**本仓库**：根目录目前**没有 LICENSE 文件**。在补上之前请按"保留所有权利"对待——引用、复用或再分发前先在 issue 里问一声。这是已知缺口，不是有意的限制。

**第三方基线**：本仓库**不再分发任何上游源码**，只给改动记录与调用脚本。请按 `comparison/README.md` 表里钉死的 commit 自行 clone：Diffusion-TS `566307e6cf2d`、CSDI `7f24a436f08d`、ImagenTime `f372626ed20a`、TTS-GAN `3f8b36ab84d1`、pyclick `bf0edd19892d`。下面是**作者机器上那几份 clone 里实际读到的**许可文件，仅供参考，以你 clone 到的版本为准：

| 上游 | 本地 clone 里的 LICENSE | 说明 |
|---|---|---|
| Diffusion-TS | MIT | `baselines_release/02_traj_diffusion_ts/DiffusionTS/LICENSE` |
| CSDI | MIT（Copyright (c) 2021 Yusuke Tashiro） | `code/baselines/CSDI/LICENSE` |
| TTS-GAN | Apache License 2.0 | `code/baselines/TTSGAN/LICENSE` |
| pyclick | MIT 文本（Copyright (c) 2018 The Python Packaging Authority） | `code/baselines/pyclick/LICENSE` |
| ImagenTime | **本地 clone 内没有 LICENSE 文件** | 再分发或商用前须回上游确认 |
| ghost-cursor（`pyppeteer_ghost_cursor`） | **本地 clone 内没有 LICENSE 文件** | 同上 |

**HMOG 数据**：本仓库既不转发数据也不转述条款，见上一节。

## 数据怎么组织：四个 per-action bundle

发布版不是一个数据集，是四个 bundle，因为不同动作定稿自不同构建（`datasets/ACTION_BUNDLE_MAP.json`）：

```
keystroke     -> keystroke        scroll -> scroll
swipe         -> swipe            tap, pinch -> tap_and_pinch
```

**每个 bundle 都携带全部五个动作的事件，但只有 `owned_actions` 列出的动作属于该发布版。** 例如 scroll bundle 里也有 tap 事件，但基线在 scroll bundle 上只许替换 scroll——替换其它四个会产出论文不报告的载体上的数字。这条规则由 `comparison/code/final_release.py` 的 `bundle_map()` / `action_to_bundle()` 从发布版读取，不硬编码；每个方法目录下的 `bundle_manifest.json` 记录了实际替换计数（拥有的动作 20,000，其余为 0）。

**每个 bundle 100 个分片、145,776 条事件。** `event_manifest.jsonl` 是**三行**，一行一个划分，不是一行一个 bundle——按 `fixed_user_disjoint_70_10_20` 做用户不相交三划分，一个用户一个分片：

| 划分 | 事件 | 假 | 真 | 分片 |
|---|---|---|---|---|
| train | 102,186 | 70,000 | 32,186 | 70 |
| development | 14,599 | 10,000 | 4,599 | 10 |
| test | 28,991 | 20,000 | 8,991 | 20 |
| **合计** | **145,776** | **100,000** | **45,776** | **100** |

四个 bundle 这张表**完全相同**，也与各自 `release.json` 的 `events` / `fake_events` / `genuine_events`（145,776 / 100,000 / 45,776）一致，与 `comparison/README.md` §3 和 `EXPERIMENTS_CN.md` §1.3、§2.1 的说法一致。**102,186（70,000 假 / 32,186 真）只是 train 划分，不是 bundle 的总量**——早先版本的本文引过这个数当总量，是错的。

## A11 的实时状态（会变，看时间戳）

**A11（`abl_a11_no_feature_match`，去掉直接特征匹配损失）不是"没跑"，而是正在收尾，且状态每小时都在变。** 下面是 **2026-08-09 13:01 EDT** 的实测：

- **训练**：已完成。两条运行随该批于 2026-08-08 23:17 训完，日志 `results/direct100k/baselines/ablations/a11_no_feature_match/train_{scroll,swipe}.log`。
- **采样缓存**：已完成。`ablations/a11_no_feature_match_cache/` 有 100 个受害者目录、40,009 个文件、373 MB，与已完成的 A8/A9/A10（各 100 目录、40,009 文件、372–374 MB）持平。
- **建库**：已完成。四个 bundle 于当日 12:53 落盘在 `baselines/final/abl_a11_no_feature_match/`（6.4 GB），`bundle_manifest.json` 与四份 `verify_*.log` 都在。
- **逐格评分**：已完成。12:54 起跑，scroll 与 swipe 的 `imu_only` 共 12 格全部出格。

**所以（成稿时的终态）：A11 已落地，12 格均值 0.717、配对差 +0.072、9/12 格 ≥ 0.60。** `RESULTS.md` 表 2 里已没有任何一行标 *not finished*；三份 CSV 已重新导出，`ablation/scores/ablation_cells.csv` 是 **180 行**（10 个臂），三份合计 **528 行**。本页上文与 `main/README.md`、`comparison/README.md` 里凡写"尚未建库、仍在采样、168 行、516 格"的地方都是更早的快照，一律以本段为准。
