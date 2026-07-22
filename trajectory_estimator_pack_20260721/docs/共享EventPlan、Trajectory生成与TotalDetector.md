# 共享 EventPlan、Trajectory 生成与 Total Detector

## 1. 本轮确定的结构

IMU 与 trajectory 的生成器、检测器仍然分别训练和调用，但两路生成不能各自随机抽条件。
正确结构是：

```text
目标用户每个 action 的 5 条真实 refs
                  +
              train-user prior
                  +
调用方指定的 duration/orientation/XY/text/pointer lifetime
                  ↓
            resolved EventPlan
              ├── IMU generator
              └── trajectory generator
                  ↓
      IMU detector + trajectory detector
                  ↓
 total detector（两边分数 + 跨模态物理一致性）
```

`EventPlan` 必须先解析完成，再运行任一生成器。两路结果同时保存：

- 相同的 `sample_id`；
- 相同的 `event_plan_sha256`；
- 相同的 `action/user/duration/orientation`；
- 相同的 XY、文字和 pointer 结构；
- 不同且域隔离的 IMU/trajectory noise seed。

不能用“IMU 第 i 行配 trajectory 第 i 行”的方式事后拼接。

## 2. Five-shot 的含义

每个目标用户、每个 action 固定使用 5 条真实参考事件。5 条参考负责提供该用户的局部风格，
train users 的全局 prior 负责收缩和补足小样本下不稳定的分布。

5 条 refs 不是候选 template 库，最终轨迹也不是从 5 条中挑一条后做缩放。模型输入包含完整的
five-shot reference tensors，event plan 由 refs 与 train-only prior 共同确定，扩散噪声再生成新的轨迹。

以下内容不会读取目标 test user 的其他事件：

- duration 与不规则 touch point rate；
- orientation 对应的屏幕坐标范围；
- start/end XY 与位移；
- pinch 双指 lifetime、span 和 angle；
- keystroke 的按键序列、字母数、contact 点数与 zero-flight 结构。

## 3. EventPlan 的完整字段

### 3.1 公共接口条件

| 字段 | 含义 |
|---|---|
| `sample_id` | IMU、trajectory 和 total detector 共用的事件身份 |
| `action` | tap/scroll/swipe/pinch/keystroke |
| `user_id` | five-shot 参考所属用户 |
| `duration_ms` | 事件逻辑持续时间；trajectory 使用整数毫秒格点 |
| `start_time_ns` | 可选的 Android 绝对开始时间 |
| `orientation_id` | -1/0/1/3 |
| `start_xy/end_xy` | 单指为一个点；pinch 为两根手指各自的点 |
| `pointer_start/end_offset_ms` | 每根手指相对事件开始的 DOWN/UP lifetime |
| `text/keycodes/n_keys/n_letters` | 只用于 keystroke |

### 3.2 可重放与审计字段

EventPlan 还保存：

- 每个 pointer 的不规则 touch token 数量；
- `contact_masks` 与 `event_ids`；
- keystroke 的 `zero_flight_after_key`；
- 恰好 5 个 reference event id 与 canonical SHA-256；
- train-only prior SHA-256；
- condition、trajectory noise、IMU noise 三个互不相同的 seed；
- EventPlan 自身的 canonical SHA-256。

因此 plan 写入 JSON 后可以在另一个进程中重放，不依赖上一个 Python 进程里的临时状态。

## 4. 时间接口怎么统一

### 4.1 逻辑时间

`duration_ms` 是两路共同的真实事件持续时间。它不是训练窗口长度，也不是程序运行耗时。

trajectory 保留原始风格的不规则 touch timestamps。例如 100 ms 的 swipe 可能产生：

```text
0, 7, 18, 31, 49, 72, 88, 100 ms
```

IMU 固定为 100 Hz，因此使用 10 ms buffer 格点。逻辑事件仍是 100 ms；输出点数由原 IMU
time contract 计算。两边不需要具有相同点数，只需要共享相同的逻辑起止时间。

### 4.2 绝对时间

调用方提供 `start_time_ns` 后：

```text
trajectory_timestamp_ns = start_time_ns + trajectory_relative_ms * 1e6
IMU_timestamp_ns        = start_time_ns + IMU_relative_timestamp_ns
```

pair audit 会检查两路第一条时间戳都等于同一 `start_time_ns`。

### 4.3 范围限制

显式 duration 必须落在对应 action 的 train-only duration 支持范围内。超出范围时接口会报错，
不会让模型无依据外推。长 typing/session 应由上层切成多个完整子事件，每个子事件使用独立
sample_id，再由 parent session id 串联；不能静默裁掉后半段。

## 5. 每个 action 的接口条件

| action | 必要条件 | pointer 结构 |
|---|---|---|
| tap | x/y、duration、orientation | 1 个 pointer，DOWN→MOVE(可选)→UP |
| scroll | start XY、end XY、duration、orientation | 1 个 pointer，不规则时间与速度曲线 |
| swipe | start XY、end XY、duration、orientation | 1 个 pointer，较快的速度/加减速结构 |
| pinch | 两指 start/end XY、duration、orientation | 2 个 pointer，共享全局时间轴，lifetime 必须重叠 |
| keystroke | text 或 keycodes、duration、orientation | 单 pointer slot 中多个独立 key contact，每个 key 有 DOWN/UP |

keystroke 不会把整段文字插值成一条连续滑动轨迹。正 flight 插入 no-contact gap；0 ms flight
保留同一毫秒内“上一键 UP 在前、下一键 DOWN 在后”的两个有序事件。

## 6. 两个独立 detector

### 6.1 IMU detector

IMU detector 使用已经完成的 IMU 波形、mask/valid_mask 与相应 metadata。它不读取 touch
trajectory 的原始点。

### 6.2 Trajectory detector

trajectory detector 独立训练，输入为 `RawTrajectoryRecord`：

- 两个 canonical pointer slot；
- event-global 不规则 `global_t_ms`；
- x/y/pressure/size；
- contact/active mask；
- Android action code；
- keycode、key event id 与 gap mask。

包括两类：

- Feature PAD：基于 AHB/HMOG trajectory 特征的 SVM/RF/XGBoost 类模型；
- Deep PAD：直接读取 raw variable-length trajectory 的 TCN/Transformer 类模型。

运行时 trajectory 输出与离线 detector bundle 使用同一个转换函数
`record_from_android_trajectory`，避免线上线下预处理漂移。

## 7. Total detector 不只是拼分数

旧的第一版 total detector 只拼接：

```text
IMU detector scores + trajectory detector scores
```

现在正式训练默认还必须输入 `consistency__*` 特征：

- trajectory path length；
- mean touch speed；
- IMU acceleration delta RMS；
- gyro RMS；
- touch speed 与 acceleration dynamic 的相关系数；
- touch speed 与 gyro 的相关系数；
- touch peak 与 IMU motion peak 的时间差；
- trajectory point rate；
- pointer count；
- n_keys。

这样 total detector 检查的不只是“两边各自像不像 fake”，还检查“这段触摸运动是否能解释同一时刻的
IMU 响应”。

## 8. 训练与测试边界

三个 detector 都保持相同边界：

- fake users：train 70 / validation 10 / test 20；
- 生成目标用户只读取固定的 5 条 refs；
- scaler、模型参数只用 train；
- EER 或目标 FRR threshold 只从 validation 选择；
- test 只计算 FA、FRR、AUC 与曲线；
- total detector 的 train/val/test 行必须分别含 real 和 fake；
- real/fake pair 都必须具有真正的 shared sample_id；
- test label、test detector score 不参与 event plan、生成或筛选。

## 9. 代码位置

- 共享 plan：[event_plan.py](../../trajectory_humanization_full_20260713/generation/event_plan.py)
- trajectory 运行时：[trajectory_layer.py](../runtime/trajectory_layer.py)
- paired 生成与一致性：[paired_layer.py](../runtime/paired_layer.py)
- trajectory 独立估计器：[service.py](../estimator/service.py)
- total detector：[total_detector.py](../estimator/total_detector.py)
- 单事件命令行：[run_paired_event.py](../scripts/run_paired_event.py)

## 10. 当前完成度

已经完成并测试：

- EventPlan JSON/SHA-256/条件重放；
- 显式 duration/orientation/XY/text/pointer lifetime 绑定；
- five-shot reference 与 train-only prior provenance；
- trajectory diffusion 运行时 API；
- trajectory 输出直接转 detector record；
- paired sample_id 与 plan hash 审计；
- 跨模态物理一致性特征；
- total detector 强制 consistency feature；
- 原生成协议 20 项回归测试全部通过；
- 新增 EventPlan/paired tests 3 项通过；
- total detector smoke 通过。

当前不能宣称 trajectory 正式模型已经完成：正式目录中尚无五动作 100-epoch best checkpoint map
和 reference registry map，现有 checkpoint 只有 smoke 训练产物。必须完成正式 trajectory 训练后，
才能运行真实的 paired generation、trajectory-only 完整检测和 total detector 完整检测。
