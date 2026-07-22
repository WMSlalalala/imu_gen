# 轨迹 Humanization Benchmark 审计

## 上游基准的实际边界

审计对象：

- GitHub: https://github.com/Gebro13/Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark
- Paper: https://arxiv.org/abs/2604.09574
- Dataset: https://huggingface.co/datasets/lyyang2766/Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark
- 审计 commit: `8924fa3e687af6f264d3b91a2d2f48faf8adfd8c`

上游输入是 Android `getevent -lt` 的 Type-B 单指事件，解析后为按时间排序的
`(timestamp_us, x, y)`。公开 parser 遇到 `ABS_MT_SLOT` 会直接拒绝，因此不支持
双指 pinch；动作只分 tap/swipe，typing 在论文中被列为 future work。

公开论文使用 24 维轨迹统计特征：持续时间、起终点、位移、路径长度、方向、
速度/加速度分位点、末端速度、最大/分位偏差、平均方向、端点路径比等。原实现
仅报告分类 accuracy，并且在随机 gesture 切分前拟合 StandardScaler，不能作为
本项目的严格主结果。

对 Table 6 与公开实现的逐项数值审计还确认：`a20/a50/a80` 是 signed
`Δspeed/Δt`，不是速度向量差的模；`dev20/dev50/dev80` 对 signed perpendicular
deviation 取分位数；历史列名 `acc_first5pct_median` 的实际窗口是最前 5 个
acceleration samples。当前 clean-room schema 已冻结为
`trajectory_features_v2_ahb_table6_hmog_real_up`，旧数值定义不再被正式 loader 接受。

`maxDevSigned` 还有一处必须显式保留的论文/代码冲突：论文 Appendix B.3 定义
“maximum signed perpendicular deviation”；但上述审计 commit 的
`analysis/lib/feature_library.py::f21_largest_signed_deviation` 首个可执行返回是
`max(abs(devs))`，其后 signed 代码不可达。本项目选择论文语义
`paper_signed_value_at_argmax_absolute_deviation`。例如 `[0,-3,-1,2,0]` 返回 `-3`，而该
公开代码实际返回 `+3`。因此本项目是 paper-aligned clean-room 实现，不宣称与公开仓库
数值完全一致。

上游 app/getevent 容器尾部存在 dummy/vanishing point：抬指时间来自 dummy，几何终点取
前一个有效坐标。HMOG 则直接记录真实 UP 时间与坐标，本项目保留该真实 UP 并让它参与
路径统计，不人为构造上游 dummy；这是数据源适配而非逐行容器复刻。

上游 GitHub 与 Hugging Face 页面均未发现机器可读 license。本项目不复制或再
分发其代码/数据，只依据公开方法说明 clean-room 实现可比特征。

## 本项目的修正协议

1. 五类动作全部来自 HMOG 同一原始数据域；tap/scroll/swipe/pinch/keystroke
   标签和互斥规则与已审计 IMU v2 一致。
2. **fake** 使用固定、互斥的 70/10/20 user split；同一 fake user 不跨 pool。
   **real** 不套用该 user split，而是在每个 `(user, action)` 内按完整 event group 的
   SHA-256 排序做 60/20/20 event split，因此 real train/validation/test 都覆盖全部
   100 users，同一完整事件不跨 pool。
3. generator prior/reference 只使用固定 70 个 train users。detector/scaler 只使用其
   detector train pool：即全部 100 real users 的约 60% held-out-by-event train groups，
   加 70 个 fake train users；validation/test 不参与拟合。
4. validation 选择阈值，test 只应用固定阈值并报告 FA/FRR/AUC。
5. 单指动作补充 clean-room AHB-style 24-feature SVM/XGBoost；pinch 使用双指
   center/span/angle 扩展；keystroke 使用逐键 DOWN/MOVE/UP、hold/flight、
   keycode/letter 序列。
6. 本目录额外报告 trajectory-only raw-sequence deep detector、合法事件率、endpoint
   成功率、exact replay、train-nearest DTW/Fréchet 和 user-level bootstrap CI。
   IMU-only 与 IMU+trajectory multimodal detector 尚未在本目录实现，仅列为未来扩展，
   不属于当前正式 benchmark。

## 原始数据与外部泛化集

- HMOG（主训练/测试）：https://hmog-dataset.github.io/hmog/ ，论文
  https://arxiv.org/abs/1501.01199 。本地完整 outer zip 包含 100 users 的
  raw `TouchEvent.csv`，同时具备五动作标签、触摸、IMU 和用户信息。
- AITouch（pinch 外部域）：https://data.mendeley.com/datasets/9v7bxv3dcc/1 ，
  论文 https://doi.org/10.1016/j.dib.2025.112323 ，CC BY-NC 4.0。只可作为
  非商用外部 domain-shift 测试，不能混入 HMOG test。
- BrainRun（tap/swipe 外部域）：https://doi.org/10.5281/zenodo.2598135 ，
  数据论文 https://doi.org/10.3390/data4020060 ，CC0。
- Touchalytics 方法参考：https://arxiv.org/abs/1207.6231 。

原始 HMOG touch 帧间隔约 16--17 ms，即约 60 Hz。提取层保留原时间戳；只有
模型内部张量化时才允许显式重采样，报告中不得把原始 touch 宣称为 100 Hz。
