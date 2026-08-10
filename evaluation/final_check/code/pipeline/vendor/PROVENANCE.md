# vendor/ 的出处与许可状态

这个目录里的代码**不是我们写的**，也不是我们的贡献的一部分。它在这里只有一个理由：没有它，某些基线无法复现。

**这是一个公开仓库，而这些文件都没有随附许可证文本。** 在对外发布前必须逐项处理，下面记录每一项的实际状态和证据，不做推断。

## `pyppeteer_ghost_cursor/`（5 个文件）

`build_ghostcursor_baseline.py` 经由 `ghost_cursor_path.py` 使用它，产出论文对比表里的 ghost-cursor 轨迹基线（24 格，均值 FAR 0.335）。

**状态：作者未发布过的 Python 移植版，无许可证。**

证据是文件自己的头部——`spoof.py` 开头就是作者的待办清单：

```
# TODO:
# - Add click and moveTo
# - Double check for completion
# - Actually make a repo
# - Publish
```

`Actually make a repo` / `Publish` 这两条说明这份移植在被我们拿到时**从未进过任何仓库、从未发布**。磁盘上的源码树里没有 `.git`、没有 `setup.py`、没有 `LICENSE`，当前 Python 环境里也查不到对应的 pip 元数据。所以它不是"上游还在、我们没记下来"，而是**确实没有可追溯的上游**——这是把它保存进来的理由，丢了这份拷贝，ghost-cursor 基线就不可复现。

被移植的**算法**本身是公开的（JS 的 ghost-cursor 项目，基于 Fitts 定律与贝塞尔曲线的鼠标轨迹模拟），论文里也是这样引用的。**但这份具体的 Python 代码的许可状态未知。**

**发布前要做的**：要么取得授权，要么换成一份许可清楚的实现重跑该基线，要么把这个目录从公开发行版里移除、只在论文里描述算法。**未决，不要默认它可以照原样公开。**

## `ImagenTime_additions/data/`（3 个文件）

**状态：这是我们自己写的代码，不是第三方代码。**

ImagenTime（Naiman et al., NeurIPS 2024）把 `data/` 作为空包分发，语料由使用者自备，所以在这三个文件存在之前仓库里没有任何东西会 import 它们。内容：

- `long_range.py` —— 把 HMOG 窗口接到他们的 `fred_md` 分支上，沿用他们每个语料都用的逐通道 min-max 缩放（EDM 里 `sigma_data` 硬编码为 0.5）
- `data_provider/data_factory.py` —— 故意对未知数据集名抛 `NotImplementedError`，让写错的数据集名直接中止，而不是静默去加载 ETT
- `data_provider/__init__.py` —— 空包标记

**clone ImagenTime 到 `f372626` 也拿不到它们**，因为它们是未进版本控制的本地新增。复现方式是拷进 `ImagenTime/data/`。

许可上没有问题：**这是我们的代码**，写来适配一个第三方框架。它放在 `vendor/` 下只是因为它必须落在对方的目录结构里才能生效，不代表它的作者是对方。

---

**一句话**：`ImagenTime_additions/` 可以随仓库公开；`pyppeteer_ghost_cursor/` 不行，先解决许可再说。
