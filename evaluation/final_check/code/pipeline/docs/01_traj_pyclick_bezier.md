# 基线 01 — pyclick 贝塞尔机器人（轨迹）

## 方法出处

[patrikoss/pyclick](https://github.com/patrikoss/pyclick)（MIT）。反爬虫 / 自动化圈里
生成「像人的指针轨迹」的事实标准实现。`HumanCurve(fromPoint, toPoint, **kwargs)` 做四件事：

1. 在起终点包围盒外扩 100 px 的矩形里**随机取 2 个内部结点**；
2. 用起点 + 结点 + 终点作控制点画一条**贝塞尔曲线**，取 `max(|dx|,|dy|,2)` 个点；
3. 对中间点按正态分布（均值 1、标准差 1 px、频率 0.5）**随机加扰动**（只加在 y 上，这是库自身的写法）；
4. 用缓动函数 `easeOutQuad` 从曲线上**重采样出目标点数** —— 这一步决定速度剖面。

## 我们做了什么

- 库源码**一行未改**，vendored 在 `pyclick/`。
- `pyclick/__init__.py` 会 import `HumanClicker` → `pyautogui` → 需要 X display。
  我们**只用曲线数学**，所以在 import 前往 `sys.modules` 注册一个空的 `pyautogui`
  占位模块。这不改变库的行为，只是绕开一个我们不调用的 GUI 驱动。
- 端点 = 载体自己绑定的目标请求（`gesture_requested_start_px` / `_end_px`），取整到像素
  （pyclick 的结点采样用 `range()`，要求整数边界；真实派发本来也是整数像素）。
  pinch 没有单指端点请求（它按区域派发），用载体质心路径的首尾点。
- 点数 = 载体的行数，`targetPoints` 直接给。
- 每条事件的随机种子 = `sha256("pyclick|" + event_id)[:4]`，所以重建是**确定性**的
  （已验证：从 vendored 路径重建一次，与已评分数据集逐字节相同）。
- **keystroke 拒绝生成**（见总 README 第 1 节）。

## 一个必须知道的库行为：tap 会退化成静止点

`generatePoints` 里 `midPtsCnt = max(|from.x-to.x|, |from.y-to.y|, 2)`。
tap 的请求起点 == 终点，于是只有 **2 个**贝塞尔点，`tweenPoints` 再怎么取都只能落在这两点上
→ **整条 tap 轨迹是一个完全静止的点**。

这是库本身的性质（它是为「鼠标从 A 移到 B」写的），不是我们的误用。
报告里必须写明，因为 tap 那一行的高 FAR 是「静止点比真人 tap 更难被识破」的结果，
而不是贝塞尔曲线质量好。

## 复现

```bash
bash run.sh /path/to/replay_dataset_zoh /path/to/output
```

`run.sh` 做三步：建数据集 → 跑 `verify_harness.py` → 跑 30 格检测器 → 出 FAR 表。

## 运行统计

| 项 | 值 |
|---|---|
| 假事件总数 | 100,000 |
| 生成轨迹 | 80,000（tap / scroll / swipe / pinch） |
| 拒绝 | 20,000（keystroke） |
| 被屏幕边界裁掉的采样点 | 361 / 约 3,400,000（0.01%） |
| 构建耗时 | 4.6 秒（32 进程） |
