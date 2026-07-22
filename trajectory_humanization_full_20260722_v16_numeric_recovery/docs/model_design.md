# 五动作 five-shot 神经轨迹 diffusion 设计

## 1. 模型边界

`tap / scroll / swipe / pinch / keystroke` 分别训练一个独立
`TrajectoryDiffusion`。它从新高斯噪声反向采样，不读取、选择、扭曲或重放某一条
真实 template。五类模型共享张量协议和物理约束代码，但参数彼此独立。

正式模型是 **five-shot**：每个 candidate 必须同时带 5 条同一 user、同一 action、
同一 split 的真实 reference trajectories。reference ID 必须唯一，candidate 自身不能
进入 refs；缺少、重复、跨用户、跨 split 或 fake ref 都直接报错。验证/测试用户的
非 ref 真实轨迹不会进入生成条件。

## 2. Canonical batch

| 字段 | shape | 含义 |
| --- | --- | --- |
| `features` | `[B,2,T,5]` | `progress/lateral/log_dt/pressure/size` |
| `point_mask` | `[B,2,T]` | 有效全局时间线；可包含 keystroke 离屏 gap |
| `contact_mask` | `[B,2,T]` | 真正向 Android 发出的触点 |
| `event_ids` | `[B,2,T]` | contact 所属 key/contact ID；gap/pad 为 -1 |
| `pointer_mask` | `[B,2]` | 有效手指 |
| `duration_ms` | `[B]` | 完整事件 union 的总时长 |
| `pointer_start_offset_ms` | `[B,2]` | 每根手指相对事件起点的 DOWN 偏移 |
| `pointer_end_offset_ms` | `[B,2]` | 每根手指相对事件起点的 UP 偏移 |
| `start_xy/end_xy` | `[B,2,2]` | 每根手指自身 DOWN/UP 的硬端点 |
| `orientation_id` | `[B]` | `-1/0/1/3` |
| `pinch_span/angle` | `[B,2]` | 双指开始/结束几何 |
| `n_keys/n_letters` | `[B]` | 总 key contact 数、其中 alphabetic key 数 |
| `keycodes/keycode_mask` | `[B,K]` | 有序 keycode 序列 |

reference 使用对应的 `[B,5,2,Tr,*]` 张量、keycode 张量、pointer lifetime 和来源
ID/user/split。所有 padding 都有独立 mask，不能成为训练目标。

单指动作必须覆盖完整 `0 → duration`。Pinch 保留真实 Android 多指状态机：例如
pointer 0 可在 0 ms DOWN、250 ms UP，而 pointer 1 在 20 ms POINTER_DOWN、235 ms
POINTER_UP。至少一根指针从 0 开始，最后一根指针的 UP 等于总时长；不会再把两根
手指错误压成相同时间范围。

## 3. 原始输入

单指示例：

```python
{
  "action": "swipe",
  "sample_id": "event_123",
  "user_id": 7,
  "split": "train",
  "is_real": True,
  "orientation_id": 0,
  "duration_ms": 420.0,
  "pointers": [{
    "xy": float_array_Nx2,
    "timestamps_ms": float_array_N,  # 同一事件全局时间轴
    "pressure": float_array_N,
    "size": float_array_N,
  }],
}
```

Pinch 必须有两个 pointer，两个 `timestamps_ms` 可以错开。`duration_ms` 必须等于
所有 pointer lifetime 的 union，不一致会 fail closed。

Keystroke 不接受一条虚构连续线，而是：

```python
{
  "action": "keystroke",
  "contacts": [
    {"keycode": 97, "xy": xy_down_move_up, "timestamps_ms": t_down_move_up},
    {"keycode": 98, "xy": xy_down_move_up, "timestamps_ms": t_down_move_up},
  ],
  "n_letters": 2,
  # 同样必须保存 sample_id/user_id/split/is_real/orientation_id
}
```

这里的 `97/98` 是 HMOG `KeyPress.csv` 的 ASCII `a/b`，不是 Android
`KEYCODE_A=29` / `KEYCODE_B=30`。正式 embedding 使用 `0..16383`，因此完整保留
corpus 中的稀有码 `8230`；所有负 sentinel 只在神经条件中统一成 token 0，原始值仍
保存在 metadata。

键间仅增加一个无接触 gap timeline slot 来学习 flight `log_dt`，其 XY、pressure、
size 不监督也不输出。每个 contact 单独恢复 DOWN/MOVE/UP。

## 4. 轨迹表示与条件

每根手指以自身 start→end chord 建立局部坐标：

1. `progress`：沿 chord 投影；
2. `lateral`：法向偏移；
3. `log_dt = log(dt_ms / 10ms)`；
4. pressure；
5. size。

调用条件编码器使用 duration、pointer lifetime fractions、orientation、每指 XY 端点、
pinch span/angle、n_keys、n_letters、有序 keycode GRU，以及 pointer count/长度。
Keystroke 还把每个 contact 对应的 keycode embedding 作为逐时间点通道输入 denoiser，
所以 `[A,B]` 与 `[B,A]` 不是相同条件。

5 条 refs 由 permutation-invariant DeepSets encoder 编码。它使用 masked 轨迹统计、
触点/gap 比例、pointer lifetime、pointer count 和 keycode 统计；换 ref 顺序不改变
condition，而换成另一组 refs 会改变 condition 和同噪声下的生成结果。

## 5. Diffusion 与 loss

模型为 epsilon-objective DDPM。正式训练使用 1000-step 线性噪声表
`beta=1e-4 -> 2e-2`，其末端 `alpha_bar` 约为 `4e-5`，因此训练前向过程的
终点与生成时使用的标准高斯起点一致。旧的 200-step 同参数噪声表末端
`alpha_bar` 仍约为 `0.132`，不用于正式训练。noisy
features、point/contact/event/keycode channels 进入
1D residual denoiser，timestep、metadata 和 ref-style condition 通过 FiLM 注入每层。

监督 mask 为：

- contact 点：5 个 feature 全监督；
- keystroke gap：只监督 `log_dt`，不监督不存在的 XY/pressure/size；
- inactive pointer/padding：完全不监督。

因此 loss 是所有有效 feature element 的 masked epsilon MSE，不是对 padding 或离屏
几何补零后的普通 MSE。

保留完整 DDPM sampler，同时正式大量生成使用 50-step DDIM。DDIM timestep 是训练
schedule 的严格子序列，denoiser 确实调用 50 次；它不是随机 t 的 one-step `x0_pred`。
`eta=0` 在相同初始 seed 下确定性复现，`eta>0` 可加入合法随机性。

## 6. 采样后物理约束

`constrain_and_decode` 只施加接口/物理约束，不看 detector 分数：

1. 验证 action 的 pointer 数、prefix timeline、contact/event IDs；
2. 每根指针的 timestamp 严格递增，并精确落在自身
   `pointer_start_offset_ms → pointer_end_offset_ms`；
3. 完整事件总时长等于所有指针的 union；
4. 每根指针局部首尾精确解码为 caller 指定 XY；
5. pressure/size 裁到 `[0,1]`；
6. gap、inactive pointer 和 padding 的 XY/feature 精确为零；
7. contact phase 恢复为 DOWN/MOVE/UP，pinch 的两指时间保持错开。

这些约束不会选择“更能骗过 detector”的候选，也不读取 test 数据。

## 7. 调用示例

```python
from trajectory.data import build_fewshot_examples, collate_fewshot_trajectories
from trajectory.model import TrajectoryDiffusion

examples = build_fewshot_examples(train_targets, real_train_pool, seed=42)
batch = collate_fewshot_trajectories(examples)
model = TrajectoryDiffusion("swipe")

loss = model.training_loss(batch)["loss"]
loss.backward()

# sampling_batch 同样必须有恰好五条合法 refs
fake = model.sample_ddim(sampling_batch, inference_steps=50, eta=0.0)
```

正式 checkpoint 必须保存 action、schema、diffusion schedule、optimizer、AMP scaler、
EMA、split/hash、best validation 与 last resume 状态。五个 action 各自保存，不覆盖
best。

## 8. 已通过的基础测试

`tests/test_model_smoke.py` 已验证：masked loss 下降、padding 不进 loss、完整 DDPM、
多步 DDIM 调用次数、五 refs 泄漏门禁与顺序不变性、不同 refs 的生成影响、keycode
顺序影响、离散 key contacts/gaps、双指错开 DOWN/UP、精确 XY/时间端点和合法 mask。

这些是模型基础模块验证，不替代正式 100-user 训练、100,000 条 fake 生成和独立
Feature/Deep PAD benchmark。
