# 五类真实轨迹 corpus 与 action-specific diffusion 严格训练协议

## 1. 范围与当前状态

本协议只负责：

1. 从 `hmog_touch_trajectory_v1` 的 numeric flat+offset NPZ 无损恢复真实触摸事件；
2. 固定 70/10/20 用户划分和每用户 5 条 reference；
3. 分别训练 tap / scroll / swipe / pinch / keystroke 五个 diffusion；
4. 保存可审计、可断点恢复且不覆盖历史 best 的 checkpoint。

它不修改 `trajectory/data.py` 或 `trajectory/model.py`，不复用旧 IMU generator 的
checkpoint，也不把 template 检索当作生成。当前只完成代码、synthetic 测试和真实
one-user smoke；**没有启动正式 full training**。

旧目录 `results/trajectories_full` 已被 uncapped audit 明确拒绝：其中有三指 pinch，
且多数用户缺少 keystroke refs。正式输入只能使用修正 extractor 生成并通过本文全部
门禁的 `results/trajectories_full_v2`；不能因为旧文件已存在而回退使用。

## 2. 输入 corpus 与反序列化

正式输入每个 action 一个文件：

```text
hmog_trajectory_tap.npz
hmog_trajectory_scroll.npz
hmog_trajectory_swipe.npz
hmog_trajectory_pinch.npz
hmog_trajectory_keystroke.npz
```

`training/corpus.py` 始终使用：

```python
np.load(path, allow_pickle=False)
```

启动审计会访问 NPZ 内每个字段，并拒绝任何 object dtype。恢复仅依赖数值列和：

- `event_offsets`：事件到 flat touch rows；
- `event_key_offsets`：typing event 到 keys；
- `key_touch_offsets`：每个 key 到它自己的 DOWN–UP touch rows。

所有 offset 必须从 0 开始、单调不减、末项精确等于对应数组长度。`n_rows`、
`n_keys`、`n_letters` 还会与 offsets 和 `key_is_letter` 逐事件交叉验证。

### 2.1 sample identity

正式 `sample_id` 直接使用非负 numeric `event_id` 的十进制字符串：

```text
sample_id = str(event_id)
```

action 已是独立字段，不再拼成 `action:event_id`。每个 action 内所有 event id 必须
唯一。

### 2.2 单指动作

tap / scroll / swipe 恢复一条完整单指 contact，并保留：

- 原始不规则 `timestamps_ms`，不重采样；
- XY、pressure、size、Android action code、frame index；
- 完整动作时间轴上的 pointer start/end offset。

### 2.3 pinch 双指

pinch 必须恢复恰好两条 pointer stream。pointer 顺序按原始 pointer id 稳定排序，
不假设 id 必为 0/1；这与正式 generation corpus loader 完全一致。

两根手指可以错开 DOWN/UP。例如：

```text
pointer 0: [0 ms, 100 ms]
pointer 7: [20 ms, 80 ms]
event:     [0 ms, 100 ms]
```

保存语义为：

```text
pointer_start_offset_ms = [0, 20]
pointer_end_offset_ms   = [100, 80]
```

`end_offset` 是相对完整事件起点的绝对 UP 时间，不是“距离事件尾还剩多少时间”。
训练层不会把两根手指都强制改成 `0 -> duration`。

### 2.4 keystroke 离散 contacts

keystroke 严格按 `event_key_offsets + key_touch_offsets` 恢复多个离散 key contacts。
相邻 key 的 flight 时间保留，但 flight 期间没有伪造 XY 连线。模型使用的 timeline
可以有 contact gap，gap 只承载时间信息。

两个计数分开保存并交叉检查：

```text
n_keys    = 全部 key contacts 数
n_letters = 其中字母 key 数，0 <= n_letters <= n_keys
```

HMOG 原始 keycode 实测包含 `-1/-2/-3/-5`，并出现两次 Unicode U+2026
（十进制 `8230`）。原值完整保存在 metadata；模型与正式
generation 共用以下固定映射：

```text
raw_keycode < 0  -> UNKNOWN token 0
raw_keycode >= 0 -> token = raw_keycode
允许非负 raw 范围 [0, 16383]
embedding vocabulary size = 16384
```

负值的具体原值不会丢失，只是不让负下标进入 `torch.Embedding`。这个范围完整覆盖
当前正式 HMOG corpus 的最大值 `8230`；大于 `16383` 的未审计输入会 fail closed，
而不是丢键、取模或静默截断。Feature PAD 仍保留 `keycode_8230` 的独立符号身份。

## 3. 固定用户划分

唯一正式划分文件：

```text
/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json
```

固定 SHA-256：

```text
82f2277374be47d5ec9dada2f7e60d0d5afd7ba79ac8a08b67e1607294ff530b
```

加载时同时检查：

| split | 用户数 | 用途 |
| --- | ---: | --- |
| train | 70 | optimizer 更新 |
| val | 10 | 20% milestones 的 checkpoint 选择 |
| test | 20 | 只预留给最终生成/检测，不参与训练或 best 选择 |

三组必须互斥且并集严格为 user id `0..99`。split 文件内容或 hash 改变即拒绝运行。

## 4. 固定五条 ReferenceRegistry

正式协议不是“每个 target 临时再抽 5 条”。`ReferenceRegistry` 对每个
`action/user/split` 只确定一次固定的 5 条真实 reference：

```text
(action, user, split) -> exactly 5 unique real sample ids
```

选择由 `seed + corpus SHA-256 + split + user` 决定，因此跨进程、resume 和
DataLoader worker 可重现。固定 refs 会从 target pool 排除：

```text
all real events = 5 fixed refs + all eligible targets
```

因此每组至少需要 6 条真实事件。少于 6 条时整个 action 训练在 optimizer 创建前
fail closed，不会少用 reference，也不会悄悄删除该用户。

每个 action 的 Registry 写入：

```text
reference_registry.json
```

其中保存 source indices、numeric sample ids、corpus/split hash 及 registry 自身的
canonical JSON SHA-256。checkpoint 和 run manifest 都绑定该 hash。同一个用户的
所有训练 targets、validation targets 和之后的 fake request 始终使用该组的同一
五条 reference；target 永远不在 refs 中。

JSON 的 `schema_version/producer/split_sha256/entries/registry_sha256` 与
`generation.protocol.ReferenceRegistry` 完全兼容；generation 会直接加载训练产出的
per-action registry，不会重新抽 refs。entries 的 canonical 顺序固定为
`action -> split(train,val,test) -> user_id`。

## 5. 变长 batch：不截断、不丢事件

`StrictVariableLengthCollator` 的规则：

- `max_points=None`；
- target 只 pad 到当前 batch 最长 target；
- refs 只 pad 到当前 batch 最长 reference；
- collate 后逐 pointer 检查 mask 长度必须等于原始长度；
- `drop_last=False`；
- 一个 epoch 覆盖 target pool 中每个 target 恰好一次。

为了避免极长 typing event 把随机 batch 的 padding 撑大，使用 deterministic
length-bucket sampler：

1. 排序 key 是 target 与其固定 refs 的最大原始行数；
2. 相邻长度先形成 bucket；
3. train 在 bucket 内和 batch 间用 `seed+epoch` 确定性打乱；
4. 每次都验证 sampler 输出恰好是 `range(len(dataset))` 的一个排列。

这只减少 padding，不裁剪 outlier，也不按长度删除事件。

## 6. Diffusion 训练

五类 action 分别运行一个独立 `TrajectoryDiffusion`。正式默认：

| 参数 | 默认值 |
| --- | ---: |
| epochs | 100 |
| batch size | 32 |
| learning rate | 2e-4 |
| weight decay | 1e-4 |
| diffusion steps | 1000 |
| beta | linear, 1e-4 -> 2e-2 |
| EMA decay | 0.999 |
| gradient clip | global norm 1.0 |
| seed | 42 |
| AMP | CUDA 上启用 |

### 6.1 CUDA 运行时确定性契约

训练引擎在导入 `torch` 之前直接固定：

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

这不是依赖启动 shell 的可选建议值；进程会将它设为上述精确值，避免
resume 因继承不同外部环境而改变 cuBLAS CUDA GEMM 的确定性契约。
`seed_everything()` 还必须得到以下精确运行时状态：

```text
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

`warn_only=False` 是正式契约：PyTorch 遇到没有确定性实现的算子时直接失败，
不允许长训练只记一条警告后继续。已知的 post-DDIM timing `cumsum` 已在生成
物理投影中改为 CPU 确定性扫描，见 `generation_protocol.md`。

吞吐 probe 的 result JSON 和正式训练的 `run_manifest.json` 都保存：

```json
"runtime_determinism": {
  "cublas_workspace_config": ":4096:8",
  "deterministic_algorithms_enabled": true,
  "deterministic_algorithms_warn_only": false,
  "cudnn_benchmark": false,
  "cudnn_deterministic": true
}
```

Supervisor 对吞吐候选、选中的 100-step probe 以及正式训练完成检查都要求
该字典与上述值完全一致；字段缺失或任一值不同都不能用于固化 batch，也不能被
认定为正式 training complete。该契约控制同一设备与固定 batch 边界下的可重现性；它不把
不同 GPU/驱动/PyTorch 版本之间的逐 bit 相同当作额外承诺。

训练 checkpoint 协议已升级为 `trajectory_diffusion_strict_five_ref_v2`。每个 mid-epoch、
best 和 last checkpoint，以及 epoch commit journal，都内嵌同一精确字典。恢复时先核对
protocol/config/source/runtime；orphan staged checkpoint 还必须证明 embedded journal 与
checkpoint runtime 完全一致，之后才允许晋升 best、替换 last 或追加 metrics。因而旧 v1
checkpoint 不能被新的 run manifest 重新标记为严格确定性产物。

### 6.2 terminal Gaussian gate

200-step linear schedule 的 `alpha_bar_T` 约为 0.132，终点仍含大量数据成分，却从
纯 Gaussian 开始生成，训练/采样边界不一致。因此正式默认改为 1000 steps；此时
`alpha_bar_final` 约 `4e-5`。

正式启动前计算并记录 `alpha_bar_final`，要求：

```text
alpha_bar_final <= 1e-3
```

不满足就 fail closed。小 schedule 只可由 synthetic unit test 显式打开 test-only
override，正式 CLI 不提供这个开关。训练每个 batch 仍只抽一个随机 diffusion t；
1000 steps 不会把单个训练 batch 变成 1000 次网络前向。后续生成仍可用 50-step
DDIM。

### 6.3 条件与 loss

训练 batch 包含：

- action-specific metadata；
- duration、orientation、XY endpoints；
- pointer start/end offsets；
- pinch 两指几何；
- keystroke 的 n_keys、n_letters、keycode sequence 和 contact gaps；
- 固定 5-reference set embedding。

loss 只在各 feature 的真实有效 mask 上计算；padding 不产生梯度。tap 的零 chord
不假设末点 progress 必须为 1，坐标互逆规则完全交给当前 canonical data/constraint
实现。

### 6.4 validation milestones

完整 validation 只在总 epoch 的：

```text
20%, 40%, 60%, 80%, 100%
```

执行。若百分比不是整数 epoch，使用 `ceil` 后去重。每次使用：

- 全部 val target pool，无 sample cap；
- 固定 registry refs；
- 固定 validation timestep/noise RNG；
- EMA 参数；
- `val masked epsilon MSE` 越低越好。

记录的 `fraction = completed_epoch / total_epochs`，不会错误地永远写成 1。test
split 不计算 loss、不选 threshold、不选 best。

多 GPU 时 validation RNG 绑定完整 device，例如 `cuda:1`，不会错误创建在默认
`cuda:0` 上。

## 7. Checkpoint 与断电恢复

输出包括：

```text
last.pt
last_state.json
training_progress.json
best_epoch_XXXX_step_XXXXXXXXX_valloss_XXXXXXXX.pt
best_manifest.json
run_manifest.json
metrics.jsonl
source_audit.json
reference_audit.json
reference_registry.json
```

### 7.1 last

`last.pt` 用同目录临时文件写完、flush/fsync 后 `os.replace` 原子更新。保存：

- model、EMA、optimizer、AMP GradScaler；
- Python / NumPy / Torch CPU / 全 CUDA RNG；
- epoch、下一 batch、global step；
- 当前 epoch 已累计 examples、weighted loss、feature count、batch count；
- best metric、最后一次 validation；
- model config、corpus hash、split hash、registry hash。

默认每 1000 optimizer steps 以及每个 epoch 边界更新。mid-epoch resume 重新构造
相同 length-bucket 顺序，跳过已完成 batch，并继续原累计器，所以 epoch loss 仍是
完整 epoch 的 loss，不会只统计断点后的 remainder。

正式配置把该间隔固定为 500 optimizer steps。每次原子发布 `last.pt` 后，
`last_state.json` 记录其实际 SHA-256、size、source/config 和 progress；
`training_progress.json` 由训练 worker 自己按成功 optimizer step、validation batch、
checkpoint commit 和 complete 阶段更新。它包含 run-instance、PID、epoch/batch/global-step、
最后成功进度时间、有限 loss/grad norm 和 device。supervisor 不把自身存活当作训练进度；
同一 worker 的 counter/sequence 必须单调，600 秒没有有效进度即失败。

### 7.2 best

每次 val loss 创新低都新建一个带 epoch/step/metric 的 versioned 文件；文件已存在
时拒绝覆盖。`best_manifest.json` 保存全部 improvement history 和当前 best。best
只按 val 选择，不允许 test 或生成后 detector 结果反选。

正式 sampler 把 `best_manifest.json` 作为独立 trust boundary：先核 exact protocol、val-only
metric、top/best role、`ema.shadow`、history tail、source、schedule、epoch/global-step/val-loss，
再读取 checkpoint bytes 并核 `best.checkpoint_sha256`。只有哈希通过后才从同一份内存 bytes
反序列化，避免 manifest 检查与 `torch.load` 之间的文件替换竞态。`.pt` 自身必须声明
`training_state_with_raw_model_and_ema`，manifest 则声明 `validation_selected_best`；两层角色
不能混用。

未提供 `--resume` 时，如果输出目录已有 `last/best/metrics/run_manifest`，训练拒绝
启动，防止误覆盖旧实验。

## 8. Source、schema、split 与 count 审计

正式训练前先执行：

```bash
python scripts/audit_training_corpus.py \
  --corpus-dir results/trajectories_full_v2 \
  --output results/trajectories_full_v2/training_corpus_audit.json
```

该命令没有 `max_samples`，会检查五个 action 的全部事件。审计包括：

- 每个 NPZ path、size、SHA-256；
- extraction manifest path 与 SHA-256，并核对 manifest 内 action NPZ hash/count；
- 所有字段均可 `allow_pickle=False` 访问且 object count=0；
- schema/version/unit/sampling；
- offsets、event/key/letter counts；
- 70/10/20 split 文件 hash、成员和互斥；
- per action/split/user event counts；
- 每组是否至少有 `5 refs + 1 target`；
- 全部事件逐条 canonical restore；
- pointer global time offsets 是否保留；
- diffusion terminal Gaussian gate。

## 9. 正式运行命令

一个 action 的例子：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_trajectory_diffusion.py \
  --action tap \
  --corpus-dir results/trajectories_full_v2 \
  --output-dir results/training/tap \
  --device cuda:0
```

断点恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_trajectory_diffusion.py \
  --action tap \
  --corpus-dir results/trajectories_full_v2 \
  --output-dir results/training/tap \
  --device cuda:0 \
  --resume results/training/tap/last.pt
```

不同 action 可分配到两张 GPU 并行，但每个 action 仍只有自己的 model、optimizer、
registry、last 和 best 目录。

## 10. 已执行测试

Synthetic 测试覆盖：

- numeric-only/`allow_pickle=False`；
- numeric sample id；
- staggered pinch `[0,100]` 与 `[20,80]`；
- keystroke offsets、离散 contacts、负 raw keycode 映射；
- `n_keys != n_letters`；
- 固定 registry、refs 从 targets 排除；
- fail-closed 少于 6 条；
- deterministic length buckets 全覆盖；
- 无截断 collate；
- 20% validation milestones；
- `cuda:1` RNG device 绑定；
- cuBLAS workspace、CUDNN 与 deterministic-algorithm runtime audit；
- AMP scaler state、EMA、atomic last、versioned best；
- 模拟断电后的 exact mid-epoch resume，最终 model/EMA/metrics 与未中断运行一致。

真实 one-user smoke：

```bash
python tests/run_training_one_user_smoke.py
```

已对五个 action 各执行一次真实 NPZ restore、固定 5 refs、无截断 collate、loss
backward、gradient clip 和 optimizer step。结果写入：

```text
results/training_one_user_smoke.json
```

one-user smoke 明确不是正式结果，因为一个用户不能覆盖 70/10/20；它只证明真实
NPZ 到模型训练 step 的端到端接口可运行。

## 11. 正式耗时与显存预估

正式训练尚未启动；下面是启动前的容量规划区间，不是实测结果。按旧 full corpus 中非 keystroke 的数量、batch 32、
100 epochs 粗估 optimizer steps：

| action | 约 train targets（扣除 5 refs/user） | 约 optimizer steps | 初始耗时区间 |
| --- | ---: | ---: | ---: |
| tap | 14.5k | 45k | 1–3 h |
| scroll | 38.7k | 121k | 4–8 h |
| swipe | 47.5k | 148k | 5–10 h |
| pinch | 以 v2 audit 为准 | 约 120k–140k | 5–10 h |
| keystroke | 以 v2 fallback 后 audit 为准 | 约 80k–150k | 4–10 h |

两张 GPU 并行时应按 action 分进程调度，不做同一 action 的不透明混合。AMP 下常规
batch 预计约 3–6 GB VRAM；最长 pinch/typing bucket 保守按 8–12 GB 预留。模型参数
本身很小，峰值主要来自 target + 5 refs 的变长 activation。

正式启动流程必须先在每类跑相同协议的 optimizer throughput/peak-VRAM probe，再用：

```bash
python scripts/benchmark_training_throughput.py \
  --action tap \
  --corpus-dir results/trajectories_full_v2 \
  --batch-size 128 \
  --steps 100 \
  --warmup-steps 5 \
  --device cuda:0 \
  --output results/throughput_probe/tap_batch128.json
```

该入口读取与正式训练完全相同的 uncapped train target、固定 5 refs、变长 bucket、
AMP、反向传播和 optimizer step。长度键使用 canonical model timeline；keystroke 的正
flight midpoint 会计入长度，零 flight 不虚构点。target `T`、五 refs 的 `Tr` 以及
keystroke target/ref key-token padding `K/Kr` 分开保存。
probe 先枚举 5 个确定性 train epoch 的全部 batch 长度分布，再实测普通区、尾部和人工
组合的全局最坏满 batch；最坏批次不仅检查 OOM，其耗时也进入单调分段完整 epoch 投影。
因此不会再以恰好遗漏长序列的随机前缀代表完整训练。不读取 validation/test target，
也不创建或覆盖正式 checkpoint。JSON 同时记录 corpus/split/registry/profile hash、完整
epoch 投影、最坏耗时、loss 有限性和 CUDA 峰值。
批大小只按 `projected_full_epoch_examples_per_second` 与资源门禁确定；候选运行超过固定
120 秒预算会记录为可审计的 runtime-budget failure。该选择不能改变数据、序列长度或模型。

随后用：

```text
estimated_hours = remaining_epochs * projected_full_epoch_optimizer_seconds / 3600
```

更新 ETA。probe 只能测速度，不允许改数据、截断长事件或据此选择模型；随后从干净
正式输出目录启动 full run。
