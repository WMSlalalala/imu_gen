# 轨迹特征协议（clean-room）

## 1. 范围与来源边界

本模块独立实现三组特征：

1. AHB-style 单指轨迹 24 维特征；
2. 我们为双指 pinch 增加的中心轨迹、指间距离和旋转特征；
3. 我们为离散 keystroke 增加的 hold、flight、burst、纠错和键位移动特征。

公开依据为论文 [Turing Test on Screen: A Benchmark for Mobile GUI Agent
Humanization](https://zeyu-zheng.github.io/GUI.pdf) 的 Appendix B.3、Table 6
和公开的[项目主页](https://github.com/Gebro13/Passing-the-Turing-Test-on-Screen-Agent-Humanization-Benchmark)。
论文公开了 24 个特征名称、类别和部分数学定义，例如 mean resultant
length、路径效率及 signed perpendicular deviation。

本目录没有导入、复制或改写第三方仓库中的特征代码。论文没有明确规定的数值细节，
例如重复时间戳、零位移、速度差分位置及空统计量，均在本协议中重新定义。因此这里称为
**AHB-style**，不能宣称与某个未公开或版本可变的官方脚本逐 bit 相同。

公开文本及公开 benchmark 实现之间存在命名/说明差异，本实现冻结如下，避免以后悄悄改变协议：

- Table 6 写 `v_last3_median`，相邻正文却描述“最后五点平均速度”；本实现按正式特征名，
  使用最后至多 3 段速度的中位数；
- `acc_first5pct_median` 是公开列名中的历史命名；公开 benchmark 的实际计算取最前
  5 个加速度样本，本实现也固定取 `acceleration[:5]`，不是前 5%；
- 表格使用 `maxDev`，相关性图和正文强调 signed deviation；本实现明确输出
  `maxDevSigned`，即绝对值最大的垂距并保留其符号。

### 1.1 `maxDevSigned`：论文定义与公开代码实际行为不一致

论文 Appendix B.3 将该量明确写为 **maximum signed perpendicular deviation**，因此本项目
冻结的策略是：先找 `argmax(abs(deviation))`，再返回该位置原来的有符号值。机器可读策略名为：

```text
paper_signed_value_at_argmax_absolute_deviation
```

但审计 commit `8924fa3e687af6f264d3b91a2d2f48faf8adfd8c` 的公开文件
`analysis/lib/feature_library.py::f21_largest_signed_deviation` 时发现：注释和 docstring 写
signed，首个可执行 `return` 却是 `np.max(np.abs(devs))`，后面的 signed 实现不可达。因此
偏离值 `[0,-3,-1,2,0]` 在本项目/论文定义下为 `-3`，该公开 commit 的实际运行值为 `+3`。

本项目选择论文定义，因为符号被论文用于刻画弧线凸向/方向；测试用非对称轨迹锁定了该
差异。后续报告只能称 **AHB-paper-aligned clean-room feature**，不能称与公开仓库数值完全
一致或逐 bit 复现。

实现文件：`trajectory/features.py`。

当前持久化数值定义版本为：

```text
trajectory_features_v2_ahb_table6_hmog_real_up
```

PAD bundle 同步升级为 `trajectory_pad_bundle_v2`；loader 对旧 v1 fail closed，避免用旧的
无符号加速度/绝对偏离特征与新结果混跑。

## 2. 输入单位与公共规则

- 坐标单位：输入坐标单位，手机轨迹通常为屏幕像素 `px`；模块不擅自归一化。
- 时间单位：`ms`。
- 速度单位：`px/s`（或一般的 coordinate-unit/s）。
- 加速度单位：`px/s²`。
- 角度单位：弧度，使用 `atan2(y, x)`。
- 分位数：NumPy 线性插值 percentile。
- 所有输出为固定顺序的 `float64`，且保证有限值；非法输入抛出 `ValueError`。

### 2.1 时间单调与重复时间戳

单指和双指轨迹要求时间戳**单调不降**。如果存在 `t[i] < t[i-1]`，不排序、不修补，
直接报错，以避免改变真实事件顺序。

连续事件具有相同时间戳时，使用 `last-event-wins`：同一个时间组只保留最后一个完整
位置。这样不会用 `dt=0` 计算无限速度，而且保留该时间点最后上报的 Android 状态。

双指 pinch 对完整的 `(finger1, finger2)` 点对联合去重，不允许分别去重后错位。

### 2.2 重复空间点

坐标相同但时间不同的点不删除。它表示手指停顿，生成一段速度为零的有效时间。
零长度段不参与方向圆统计，但仍参与速度和加速度序列。

### 2.3 短序列

- 单点单指轨迹：保留起终点；duration、速度、加速度和几何运动量为 0。
- 单点 pinch：保留中心、span 和 angle；变化率为 0。
- 空单指/pinch：没有可定义的端点，因此报错。
- 空 keystroke：返回固定维度全 0，便于批处理；`nKeys=0`。
- 统计序列为空时，对应统计量为 0，不写 `NaN/Inf`。

### 2.4 HMOG 真实 UP 与 AHB dummy vanishing point

AHB 公开 benchmark 的单指容器在尾部附有一个 dummy/vanishing point：该点提供抬指时间，
几何终点取它前一个仍有有效坐标的样本，速度/路径则先删除 dummy。HMOG 本项目的 raw
`TouchEvent`/`OneFingerTouchEvent` 已直接记录真实 UP 的时间和坐标，不存在需要删除的
无坐标 dummy。因此本协议：

- duration 使用真实 `UP_time - DOWN_time`；
- `endX/endY` 使用真实 UP 坐标；
- UP 是数据集实际观测 contact，参与几何路径和相邻段统计；
- 不人为追加或删除一个 AHB 风格 vanishing point。

这是一项明确的数据源适配，不应宣称 HMOG 端点数组与 AHB app/getevent 容器逐行相同；
24 个统计量的列含义和下述 signed 数值定义保持一致。

## 3. 单指 24 维 AHB-style 特征

固定顺序由 `SINGLE_FINGER_FEATURE_NAMES` 定义，模型、CSV 和 checkpoint 不得自行重排。

设轨迹点为 `p_i=(x_i,y_i)`、时间为 `t_i`，相邻差为：

```text
dt_i = (t_{i+1}-t_i) / 1000
d_i  = p_{i+1}-p_i
l_i  = ||d_i||
v_i  = ||d_i|| / dt_i
```

AHB Table 6 的加速度是相邻**标量速度**的有符号差分。对第二个及后续速度样本：

```text
a_i = (v_i-v_{i-1}) / dt_i
```

因此减速为负，加速为正；仅改变运动方向但 speed 不变时该 tangential acceleration 为 0。

| 顺序 | 名称 | 本实现定义 |
| ---: | --- | --- |
| 1 | `duration` | `t_last-t_first`，ms |
| 2–5 | `startX,startY,endX,endY` | 起点、终点原始坐标 |
| 6 | `displacement` | `||p_last-p_first||` |
| 7 | `meanResultantLength` | 非零局部方向单位复向量均值的模，范围 `[0,1]` |
| 8 | `direction` | 端点向量的 `atan2`；零位移时为 0 |
| 9–11 | `v20,v50,v80` | 相邻段速度 `v_i` 的 20/50/80 分位数 |
| 12–14 | `a20,a50,a80` | 上述有符号 `Δspeed/Δt` 的 20/50/80 分位数 |
| 15 | `v_last3_median` | 最后至多 3 段速度的中位数 |
| 16 | `maxDevSigned` | 对端点直线的绝对值最大有符号垂距，保留符号 |
| 17–19 | `dev20,dev50,dev80` | 所有点到端点直线**有符号**垂距的分位数 |
| 20 | `avgDirection` | 非零局部方向的 circular mean |
| 21 | `length` | `sum(l_i)` |
| 22 | `ratio_end_to_length` | `displacement/length`；零路径为 0 |
| 23 | `speed` | `length/(duration/1000)` |
| 24 | `acc_first5pct_median` | 最前至多 5 个加速度样本的中位数（列名保留历史 `pct`） |

有符号偏离采用：

```text
cross(p_last-p_first, p_i-p_first) / displacement
```

手机坐标通常 y 轴向下，因此符号对应屏幕坐标系，而不是数学坐标系的“左/右”。
端点完全相同时参考直线不可定义，`maxDevSigned` 和 `dev*` 置 0；实际绕行仍由
`length` 和速度特征保留。

## 4. 双指 pinch 扩展

`extract_pinch_features(finger1_points, finger2_points, times_ms)` 要求两个手指共享同一
时间序列，输入形状均为 `[N,2]`。

首先计算焦点轨迹：

```text
center_i = (finger1_i + finger2_i) / 2
```

并对 `center` 提取完整 24 维单指特征，名称加 `center_` 前缀。随后增加 25 维：

- span 基础量：`startSpan,endSpan,spanDelta,absSpanDelta,minSpan,maxSpan,`
  `meanSpan,stdSpan,spanPathLength`；
- span 速度：`spanRate20,spanRate50,spanRate80,meanAbsSpanRate`；
- 双指轴角度：`startAngle,endAngle,angleDelta,anglePathLength`；
- 角速度：`angularSpeed20,angularSpeed50,angularSpeed80,meanAbsAngularSpeed`；
- 两指运动平衡：`finger1Length,finger2Length,fingerLengthRatio,`
  `fingerSpeedCorrelation`。

总维数为 `24+25=49`，固定顺序见 `PINCH_FEATURE_NAMES`。

定义：

```text
span_i  = ||finger2_i-finger1_i||
angle_i = unwrap(atan2(finger2_y-finger1_y,
                       finger2_x-finger1_x))
```

若个别时刻两指完全重合，轴角度不可定义；实现从非零 span 的相邻角度进行 unwrap 后
插值，首尾使用最近有效角度。若全部 span 都是 0，则角度序列为 0。这样不会产生 NaN。

`fingerLengthRatio=min(length1,length2)/max(length1,length2)`，范围 `[0,1]`；两指均不动
时为 0。速度相关性只有在至少两段且两侧速度方差均非零时才计算，否则为 0。

## 5. Keystroke 离散序列扩展

接口：

```python
extract_keystroke_features(
    keys,
    down_times_ms,
    up_times_ms=None,
    key_points=None,
    pause_threshold_ms=500.0,
)
```

固定输出 34 维，顺序见 `KEYSTROKE_FEATURE_NAMES`。

### 5.1 事件规则

- `keys` 与 DOWN 数量必须相同。
- DOWN 时间必须单调不降；相同 DOWN 时间属于不同离散按键，不合并。
- 若提供 UP，则每个 `up_i >= down_i`。
- 不要求按 DOWN 排序后的 UP 也严格递增；长按重叠可能导致 `up_i > down_{i+1}`。
- `hold_i=up_i-down_i`。
- `flight_i=down_{i+1}-up_i`，允许负数；负数表示两个键在时间上重叠。
- `downDown_i=down_{i+1}-down_i`。
- event duration 使用 `max(up)-first_down`；缺少 UP 时使用 `last_down-first_down`。

### 5.2 特征组

- 计数：`nKeys,nLetters,nUnique,uniqueKeyRatio,correctionRatio`；
- 总时间：`duration,keyRate`；
- hold：20/50/80 分位数、均值、标准差；
- flight：20/50/80 分位数、均值、标准差、`overlapFraction`；
- down-down：20/50/80 分位数、均值、标准差；
- burst：`burstCount,meanBurstSize,maxBurstSize,pauseFraction`；
- 可选键位轨迹：相邻键位距离的 20/50/80 分位数、均值及总路径长度；
- 可用性：`hasUpTimes,hasKeyXY`。

`nLetters` 只统计单字符且 `isalpha()` 的键。纠错键大小写不敏感，识别
`BACKSPACE/DELETE/DEL/<BS>/KEYCODE_DEL/\b`。burst 在相邻 DOWN 间隔严格大于
`pause_threshold_ms` 时断开。

缺少 UP 或键位坐标时，相应特征组置 0，同时 availability flag 为 0；检测器应同时使用
flag，不能把“缺失”误解为“真实测量值恰好为 0”。

## 6. 测试门禁

`tests/test_features.py` 覆盖：

- 24 维顺序和匀速直线的解析结果；
- 时间倒序拒绝；
- 相同时间戳 last-event-wins；
- 空间重复点保留停顿；
- 单点、静止和零 span 不产生 NaN；
- signed deviation；
- signed `Δspeed/Δt` 与前 5 个加速度样本中位数；
- bundle v2/feature schema 绑定；
- pinch 联合去重、对称扩张和双指长度平衡；
- keystroke DOWN/UP、hold、flight、burst、纠错和键位距离；
- 同时 DOWN 不合并、缺少可选流、空序列及非法 UP。

运行小单测：

```bash
cd /home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713
python -m unittest discover -s tests -v
```
