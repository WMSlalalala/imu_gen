# 方法侧的数据：在哪、为什么不在仓库、没有它能做什么

与 `../evaluation/DATA.md` 同一套原则：仓库只装文本，事件数据、检查点与生成产物都在仓库外。评测侧记的是「结果数据」（118 GB 生成事件 + 6.5 GB 冻结发布版），本文件记的是方法侧的**输入与中间产物**——原始语料、提取后的窗口、five-shot 素材、训练检查点。下面的绝对路径是作者机器上的位置，用于交叉核对。

## 方法链路要吃的东西

| 阶段 | 输入 / 中间产物 | 位置 | 在仓库里？ |
|---|---|---|---|
| 1 提取 | HMOG 原始归档与解压后的用户语料 | 机构存储（HMOG 数据集，需按其许可获取） | 否 |
| 1→2 | 提取后的定长 padding 窗口 NPZ | `data/processed_xy4_20260702/`（体量随动作，keystroke 最大） | 否（`.npz` 硬排除） |
| 2 | five-shot 素材清单（每用户×动作五条参考） | `android_physical_layer_20260709/results/five_action_pad_root/detector_sets/*.npz` | 否 |
| 3 | 扩散训练检查点（五个动作） | `runs/<action>/diffusion/fewshot_adv/.../checkpoints/*.pt` | 否（`.pt` 硬排除） |
| 3 | 100k 正式生成轨迹 | 生成工作树 | 否 |
| 4–5 | 拟合好的 PAD 与综合检测器 | 冻结发布版 `direct100k_final/detector_models/`（2.5 GB） | 否 |
| 冻结 | 发布版整包 | `/mnt/share/mwang49/data7/direct100k_final/`（6.5 GB） | 否 |

## 为什么不进 git

和评测侧一样两条。第一条是体积：单个提取 NPZ、100k 轨迹、检查点都在 GB 量级。第二条是仓库根目录的 `.gitignore` 采用**默认拒绝**（先 `/*` 排除一切，再白名单逐项放行），且有一层与白名单无关的硬排除：`**/*.npz`、`**/*.npy`、`**/*.pt`、`**/*.ckpt`、`**/*.log`、`**/results/`。即使把数据软链进 `method/`，也不会被提交——本目录因此只可能有文本。

## 只看仓库里的文本，能做什么

- 通读 `README.md` 指向的十余份设计文档（提取来源审计、模型设计、训练/生成协议、检测器协议与门禁），核对方法的每一步与每一道 fail-closed 门；
- 对照 `../evaluation/` 的逐格 CSV 与说明，把「方法怎么造」与「造得多好」两侧接上；
- 按上游 commit 钉版本重跑第三方基线（commit 号见 `../evaluation/DATA.md`）。

## 必须有原始数据 / GPU 才能做什么

- **重跑提取**：需要 HMOG 全量语料，产出阶段 1 的 padding 窗口；
- **重训扩散**：需要提取窗口 + five-shot 素材 + 多卡 GPU，按 `docs/training_protocol.md` 的门禁；
- **复现 100k 生成**：需要冻结检查点，按 `docs/generation_protocol.md`；
- **重拟合 / 重跑检测器**：需要 `detector_models/` 或从生成事件重训，按 `docs/detector_protocol.md`；
- 逐位一致性核对（重放结果与已发布 IMU **bit identical**）在评测侧，见 `../evaluation/DATA.md`。

## 数据组织：four per-action bundle

与评测侧共享同一套 bundle 约定：发布版不是一个数据集，是四个 per-action bundle（`keystroke`、`scroll`、`swipe`、`tap_and_pinch`），每个 bundle 携带全部五动作事件但只有 `owned_actions` 列出的动作属于该发布版。规则由发布版读取、不硬编码。细节见 `../evaluation/DATA.md` 与 `direct100k_final/datasets/ACTION_BUNDLE_MAP.json`。

## 想要数据

发布版是**冻结**的、带校验（分片 `source_sha256`、`release.json` 的清单哈希、`provenance.json` 逐格来源）。获取方式：`<归档链接：待填>`；联系作者 `<邮箱：待填>`。

原始 HMOG 语料请按 HMOG 数据集自身的发布方与许可获取，本仓库不转发。
