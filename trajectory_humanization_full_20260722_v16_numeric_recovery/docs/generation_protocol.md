# 五动作 5-shot 神经轨迹生成与 Android 序列化协议

## 1. 结论与范围

本目录实现的是正式轨迹生成管线，不是模板查询、真实轨迹变形或 detector 后选择：

1. `tap / scroll / swipe / pinch / keystroke` 分别加载一个 action-specific neural diffusion；
2. 每个 user/action 使用训练阶段登记的固定 5 条真实 reference；
3. 从全新 Gaussian noise 经过 **50 次 DDIM 神经去噪**生成；
4. 保存完整条件、5 个 reference event ID、全局时间轴、Android lifecycle 和审计字段；
5. 不实现、也不接受 selector 参数，生成后不会根据 detector score 丢弃样本。

当前已完成 synthetic + HMOG one-user 数据读取 smoke 和五动作真实 50-step 神经执行测试。按要求，本阶段没有误启动正式 100,000 条生成；正式运行必须等五个 1000-step best-EMA checkpoint 和训练阶段 `ReferenceRegistry` 均就绪。

## 2. 正式数量与用户边界

唯一用户划分来自：

```text
/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json
```

| split | users | fake/action | 五动作合计 |
| --- | ---: | ---: | ---: |
| train | 70 | 14,000 | 70,000 |
| validation | 10 | 2,000 | 10,000 |
| test | 20 | 4,000 | 20,000 |
| 总计 | 100 | 20,000 | 100,000 |

每个 user/action 必须恰好 200 条。`build_generation_units()` 在 formal 模式强制生成 500 个 action-user unit，共 100,000 条；不能改成少用户、少动作或少 fake。

## 3. 固定 5 references

### 3.1 ReferenceRegistry

正式采样不在运行时重新抽 refs，而是直接读取五个 action 训练目录各自产生的 `reference_registry.json`；命令行的 `reference_registry_map.json` 只负责把 action 映射到这五个文件，不重新抽样。每个 Registry 包含：

- `producer = trajectory_training_pipeline`；
- 固定 split 文件 SHA-256；
- 500 个 `(action, user_id, split)` entry；
- 每个 entry 恰好 5 个唯一 numeric real event ID；
- registry 内容 SHA-256。

同一个 user/action 的全部 200 条 fake 保存完全相同的 `[5]` reference ID。每个 ref 必须同时满足：

- real；
- 与 fake 同 user；
- 同 action；
- 同 split；
- 五条互不重复；
- fake ID 不属于 real/reference ID 空间。

Refs 只作为 reference encoder 的输入，不进入 fake target 集，也不会作为生成结果直接输出。

### 3.2 为什么不是每条重新抽 5 个

固定 refs 对应真实 few-shot enrollment 场景：一个人的参考集合在部署时先确定，之后多次生成都使用同一组参考。若每条 fake 重新抽 refs，200 条样本面对的是 200 个不同 enrollment context，无法解释为同一个实际调用协议。

## 4. 条件采样：5 refs + train-only shrinkage prior

### 4.1 禁止 carrier-only

不能随机选一条 ref 后原样复制它的 duration、XY、length 和 pointer lifetime。那样 200 条最多只有五组 metadata，既不满足随机时间/距离，也会制造明显重复。

当前策略对连续条件使用：

```text
five-ref robust median
  + 0.75 × two-ref interpolation deviation
  + bounded residual sampled from global train-only prior
```

正数变量在 log 空间混合，角度在 `(cos θ, sin θ)` 空间混合。若极端情况下完整 metadata 与任意 ref 完全相同，会以范围内的微小 duration 缩放打破完整复制；审计要求 `exact_metadata_copy_count = 0`。

### 4.2 train-only prior

每个 action 单独拟合 prior，输入只能是固定 70 个 train users：

- duration；
- 每根指针的 point rate；
- start/end 与 displacement；
- tap endpoint jitter；
- pinch center、span、angle；
- 两根手指在全局事件时间轴上的 DOWN/UP lifetime fractions；
- keystroke duration/key、每个 contact 的点数；
- keystroke 的完整 train-only key length/token/事件内 transition prior；
- 按 `(orientation,keycode)` 建立的 key DOWN/UP/center XY prior；
- 每个 orientation 的 observed XY bounds。

Prior 保存所有 source event IDs、source user IDs 和 SHA-256。审计重新验证 source users 是 train user 集的子集。Validation/test 的非-ref 轨迹没有进入 `ReferenceConditionPolicy` 的参数，因此不能被读取。

### 4.3 各动作条件

| action | 完整条件 |
| --- | --- |
| tap | duration、orientation、start/end XY、point count/rate、轻微 tap displacement |
| scroll | duration、orientation、start XY、end XY/displacement、point rate |
| swipe | duration、orientation、start XY、end XY/displacement、point rate |
| pinch | duration、orientation、两指 start/end、start/end span、start/end angle、两指绝对 DOWN/UP 时间、各指 point rate |
| keystroke | duration、orientation、n_keys、n_letters、完整 keycode sequence、每个独立 contact 点数、键间 gap slots、首末 XY |

### 4.4 Orientation-first，而不是生成后再裁剪

方向在任何 XY、displacement、pinch span/angle 条件之前选定。随后：

- 几何只使用五条 refs 中同一 orientation 的子集；
- prior 几何只使用 70 个 train users 中同一 orientation 的行；
- screen bounds 也取同一 orientation；
- duration、point rate、pointer lifetime 仍使用 action-global train-only prior，因为这些量不属于屏幕坐标系，且代码与审计明确区分。

因此不会再发生“portrait 与 landscape 几何先混合，再靠 clip 压回某个方向边界”的伪影。

### 4.5 Keystroke sequence 与首末键 XY

正式 prior 模式不复制五条 refs 中任一完整 keycode sequence。目标 `n_keys/n_letters` 由五条 refs 与 train-only prior 混合；序列逐位置组合：

1. 五条 refs 的位置 token；
2. refs 与 train-only event 内部的前后键 transition；
3. train-only token pool；
4. 每次组合后拒绝与任一完整 ref sequence 完全相同的结果。

transition 永远不会跨真实 typing event 边界。HMOG `KeyPress.csv` 的字母 codebook 是 ASCII：`65..90/97..122` 分别为 `A..Z/a..z`；其他非负值仍可能是 Unicode/扩展键，例如正式 corpus 中出现的 `8230`。负 sentinel 在权威 corpus 中编码为 token 0；非负 token 在 `0..16383` 内保持原值，超界 fail closed。这里不能误套 Android `KEYCODE_A=29..KEYCODE_Z=54` 的物理键码表。唯一字母判定函数是 `trajectory.features.is_hmog_ascii_letter_keycode()`；extractor audit 和 training corpus 不只检查 `n_letters == sum(key_is_letter)`，还独立强制每个 `key_is_letter` 等于该 ASCII predicate，防止“标注数量自洽但 codebook 整体错误”。

首键 DOWN XY 和末键 UP XY 不是 generic endpoint：先按所选 orientation 和对应 keycode 查询五条 refs，再与 train-only `(orientation,keycode)` 位置 prior 混合。若五条 refs 没有该键但 train prior 有，使用 train-only exact-token prior；若 token 0 或稀有 token 在 train prior 也不存在，才回退到同方向 keyboard-position prior。每个 endpoint 保存来源码：

| code | 来源 |
| ---: | --- |
| 1 | same-orientation ref + train exact-token prior |
| 2 | train-only exact-token prior |
| 3 | same-orientation keyboard fallback |

显式 caller text 是保留的部署扩展，入口接收 canonical token（原始负 sentinel 必须先经 `encode_raw_keycode` 变成 0），`condition_source_code=3`；正式 benchmark 只允许自动 prior 模式 `condition_source_code=2`，防止两种协议混入同一结果。

## 5. Pinch 的全局 union 时间轴

`pointer_start_offset_ms` 和 `pointer_end_offset_ms` 均相对事件起点：

```text
pointer_start_offset_ms = 该指 DOWN 的绝对事件相对时间
pointer_end_offset_ms   = 该指 UP 的绝对事件相对时间
```

后者不是“距离事件末尾还剩多少”。合法条件为：

```text
0 <= start[p] < end[p] <= duration
min(active starts) = 0
max(active ends) = duration
```

因此真实 pinch 中第二指稍晚 DOWN、第一指稍早 UP 的错峰结构被保留。模型和约束在各自 `[start[p], end[p]]` 内构造严格递增时间，不会再把两指强制拉成相同 `0 → duration`。

### 5.1 与 HMOG 相同的 1 ms 时间分辨率

HMOG 的 `evt_t/t_rel_ms` 是整数毫秒。为避免 detector 仅凭“小数时间戳”识别 fake：

- condition 的 duration 和每根指针的绝对 DOWN/UP 均量化为整数 ms；
- 点数强制不超过 `pointer_lifetime_ms + 1`；
- DDIM 预测的相对 interval 权重通过 largest-remainder 分配到整数 interval；
- 每个 interval 至少 1 ms，严格递增；
- 首点精确等于 DOWN、末点精确等于 UP，事件末端精确等于 duration；
- 量化后同步重算 archive 中的 `log_dt`，避免 feature/timestamp 自相矛盾。

Android pinch 不是两条互不关联的线。序列化器在 union 时间轴合并两指，生命周期为：

```text
ACTION_DOWN
→ ACTION_POINTER_DOWN
→ ACTION_MOVE*（两个 Type-B slot）
→ ACTION_POINTER_UP
→ ACTION_UP
```

同时保存 `slot`、稳定 tracking ID、Type-B tracking update（DOWN=id、MOVE=no-update sentinel、UP=-1）、frame index 和 `SYN_REPORT` 等价的 `frame_end`。

PAD 适配时，Type-B 行被解释为“slot 更新”而不是完整 MotionEvent。适配器取所有原始更新时间的 union；在某 pointer 的 DOWN..UP lifetime 内，对没有新 slot update 的帧 forward-fill 最近状态，从而恢复完整双指 snapshot，同时保留两指错峰生命周期和全部原始时间点，不做固定频率重采样。

## 6. Keystroke 是独立 contacts

Keystroke 不被插值成一条持续按住的线：

- `contact_mask=1` 的每个 key event 至少有 DOWN 和 UP；
- 键间 flight 使用 `contact_mask=0,event_id=-1` 的 gap timeline slot；
- 每个 contact 的 `event_id` 为 `0..n_keys-1`；
- Android 输出为每个 key 单独的 `DOWN → MOVE* → UP`；
- keycodes 与 `n_keys/n_letters` 一同保存。

## 7. 1000-step 训练 schedule 上的 50-step DDIM

正式 checkpoint 必须：

- 由 validation 选择并标为 best；
- 含 EMA shadow state，正式加载拒绝 raw/last-only 权重；
- `model_config.diffusion_steps = 1000`；
- action 与请求 action 一致。

正式采样在 1000-step 训练表上选择严格递增的 50 个 DDIM timestep，并从 fresh Gaussian noise 反向去噪 50 次，`eta=0`。不是 one-step `x0_pred`，也不是 template replay。

50 次神经去噪结束后，`trajectory/constraints.py` 才执行一次 hard timing
projection。每根有效 pointer 的 interval 权重是一条很短的 1-D 序列；PyTorch 明确会把
CUDA `cumsum` 报告为非确定性算子，因此 `_deterministic_cumsum_1d()` 先 detach 该序列，
在 CPU 上计算 prefix sum，再以原 dtype 拷回原 device。正常权重路径和极端浮点塌缩时的
equal-increment fallback 都走同一个 helper。该阶段是采样后的纯推理物理投影，不需要
gradient，不改变 DDIM 步数、噪声或网络输出，只消除时间轴构造中的非确定性 CUDA scan。

1000-step 的必要性也被审计：archive 保存 `training_diffusion_steps` 和 `alpha_bar_final`，正式要求 `alpha_bar_final <= 0.001`。这避免旧 200-step 线性表末端仍保留较强信号，却从纯噪声启动造成 train/inference mismatch。

每个样本使用绑定到完整设备名的独立 `torch.Generator(device=batch.features.device)`；`cuda:1` 不会误用 `cuda:0` generator。初始噪声先逐样本确定，再组成 batch，因此 **noise tensor** 与 batch grouping、shard 或 resume 边界严格无关。模型内部使用 masked GroupNorm，并在每层卷积后重新屏蔽 padding；reference occupancy 也按每条 reference 自身 valid 长度计算。因此同一 request+noise 与更长样本合批时，有效区输出在严格数值容差内一致。这里不宣称不同 batch shape 的浮点 kernel 逐 bit 相同：CPU 50-step 回归的 feature/timestamp 最大差分别为 `4.18e-6` / `2.68e-5 ms`，测试上限为 `1e-5` / `1e-4 ms`。正式协议固定 batch=32、每 unit 固定 200 条且按 sample_index 顺序分成 32/32/32/32/32/32/8，resume 只按完整 unit 跳过，所以正式 resume 保持完全相同的 batch 边界。

正式 condition/generation 协议固定 `base_seed=20260713`，它与 reference registry 的训练 seed 42 是两个不同角色。第 `sample_index` 条的 request seed 必须精确等于：

```text
stable_seed(20260713, action, user_id, sample_index)
```

正式 CLI 若收到其他 `--seed` 或非 32 的 `--batch-size` 会在读取 corpus/checkpoint 前立即拒绝。DDIM 初始高斯噪声使用独立 domain：

```text
ddim_noise_seed = stable_seed(
    condition_request_seed xor 0xDD1A50,
    action,
    user_id,
    sample_index,
)
```

100k preflight、shard manifest 和最终 audit 还要求两类 seed 各自全局唯一、两个 seed domain 互不相交。因此 ConditionRequest seed 与 DDIM noise seed 的角色不会混淆，且 shard 数量和 resume 边界不会改变逐样本 seed/noise。

## 8. Screen bounds 与 clipping

Bounds 不是从 validation/test 统计，而是按 action 和 orientation 从 70 个 train users 的端点拟合。生成条件先被投影到对应 orientation 的 train-only bounds。

神经模型仍可能让内部 lateral point 越界，因此 Android contact 输出前执行最后一次物理 clip，并逐条保存：

- `screen_min_xy/screen_max_xy`；
- `clipped_point_count`；
- `contact_point_count`；
- `clipped_point_rate`。

正式审计阈值：

- 全 unit aggregate clipping rate 不得超过 5%；
- 任一 event clipping rate 不得超过 25%。

Clip 是显式物理合法投影，不使用 detector/test score。若 clipping 异常则 unit 审计失败，不允许静默发布。

## 9. Numeric flat+offset NPZ

每个 action-user unit 是一个 `allow_pickle=False` 可读取的纯 numeric NPZ，包含 200 条 fake。没有 object、dict 或 Python pickle。

主要 event-level 字段：

- `fake_id/user_id/split_id/action_id`；
- `reference_event_ids[N,5]`、carrier ref ID、registry/prior/split/checkpoint SHA-256；
- duration、orientation、lengths、start/end XY；
- pointer absolute start/end time；
- pinch span/angle；
- n_keys/n_letters 与 `key_offsets/keycodes`；
- `key_endpoint_source_code[N,2]`；
- clipping counts/rates；
- 50-step DDIM、`ddim_eta_scalar=float32(0)` 与训练 schedule 元数据。
- `generation_base_seed_scalar`、`generation_batch_size_scalar`；
- `runtime_determinism_sha256[32]`：严格 PyTorch/cuDNN/cuBLAS 运行时契约的 canonical digest；
- 逐样本 ConditionRequest `seed` 与 `ddim_noise_seed`；
- `condition_request_sha256[N,32]`：覆盖 `ConditionRequest` 全部 30 个字段的 canonical SHA-256。

当前 numeric archive schema 为 `1.4`。旧 `1.1` 不含 base seed/request digest；旧 `1.2` 不含 generation batch 与 DDIM noise-seed 证据；旧 `1.3` 不含 strict runtime digest。所有旧 schema 都不能作为正式 resume 或 PAD 输入，必须 fail closed，不能靠可重写的 sidecar 重新贴标签。

canonical request digest 的唯一实现位于 `generation/protocol.py`。它对 dataclass 字段清单做 schema 漂移断言；preflight、archive builder、unit audit 和 100k audit 共用同一函数。集合 digest 按 `fake_id` 数值排序后聚合，因此与 action traversal、batch 和 shard 顺序无关。

两套 flat+offset：

1. `trajectory_offsets + flat_trajectory_*`：完整模型 timeline，含 keystroke gap；
2. `android_offsets + flat_android_*`：只含真实 contact rows，可直接恢复 Android lifecycle。

`trajectory_pointer_offsets[N,3]` 可在每个 event 内恢复 pointer 0/1。

## 10. Resume、shard 与原子发布

一个 unit 对应一个不可覆盖文件：

```text
shards/shard_000_of_002/tap/user_000.npz
```

规则：

- shard assignment 只由 `(action_id,user_id,num_shards)` 决定；
- 已存在文件先验证 action/user/count/50-step DDIM/`eta=0`/training schedule、当前 checkpoint SHA、archive base seed、archive batch=32，并逐条重算 ConditionRequest seed 与 domain-separated DDIM noise seed；然后从 refs+train prior 重构完整 `ConditionRequest`，核对其 canonical digest 与全部 archive condition 字段，最后重新运行 geometry/lifecycle 审计，全部匹配才 resume；
- 不提供覆盖正式 NPZ 的 force 选项；
- 新结果先写唯一 staging 文件；
- staging 完成严格审计后才 `os.replace` 原子发布；
- 审计失败删除 staging，不产生可被误判为完成的正式文件；
- output 不能等于或位于 source corpus 内部。

两张 GPU 可并行运行不同 shard：

`reference_registry_map.json` 直接指向五个训练目录的原始 registry：

```json
{
  "tap": "/train/tap/reference_registry.json",
  "scroll": "/train/scroll/reference_registry.json",
  "swipe": "/train/swipe/reference_registry.json",
  "pinch": "/train/pinch/reference_registry.json",
  "keystroke": "/train/keystroke/reference_registry.json"
}
```

```bash
python scripts/generate_five_shot_trajectories.py \
  --confirm-formal-100k --num-shards 2 --shard-id 0 --device cuda:0 \
  --corpus-dir /path/to/trajectories_full \
  --reference-registry-map /path/from/training/reference_registry_map.json \
  --checkpoint-map /path/from/training/best_checkpoint_map.json \
  --output-dir /new/output/path

python scripts/generate_five_shot_trajectories.py \
  --confirm-formal-100k --num-shards 2 --shard-id 1 --device cuda:1 \
  --corpus-dir /path/to/trajectories_full \
  --reference-registry-map /path/from/training/reference_registry_map.json \
  --checkpoint-map /path/from/training/best_checkpoint_map.json \
  --output-dir /new/output/path
```

每个 shard 写独立 `five_shot_generation_shard_manifest_v4` manifest，避免双进程互相覆盖。v4 在 v3 的 batch/双 seed 证据之外，保存精确 `runtime_determinism` 字典及其 canonical SHA-256；正式入口在导入 PyTorch 前固定 cuBLAS，并以 warn-only=false 启用严格确定性。manifest 顶层、每个 unit audit、NPZ 内 `ddim_eta_scalar` 必须同时证明 `eta=0`，sidecar 不能掩盖 NPZ 的非零或非法 eta。

## 11. 正式完整审计

两 shard 完成后运行：

```bash
python scripts/audit_five_shot_generation.py \
  --num-shards 2 \
  --output-dir /new/output/path \
  --corpus-dir /path/to/trajectories_full \
  --reference-registry-map /path/from/training/reference_registry_map.json \
  --condition-preflight /path/to/passed_all_100k_condition_preflight.json
```

完整审计强制验证：

- 恰好 500 unit、100,000 unique fake IDs；
- 每 action 为 14,000/2,000/4,000 train/val/test；
- 每 user/action 恰好 200；
- 同 unit 的五个 refs 固定且与训练 registry/hash 一致；
- refs 的 user/action/split/real/unique 合法；
- prior source users 全属于 70 train users；
- exact trajectory replay = 0；
- complete metadata copy = 0；
- complete key-sequence copy = 0，并报告 unique sequence count；
- 首/末 key endpoint 来源与可用 ref/train prior 一致，并报告稀有/token-0 fallback；
- duration、pointer lifetime、trajectory/Android timestamps 全部位于整数 ms lattice；
- endpoint 和 pointer time error <= 1e-3；
- 所有 contact 在 train-only orientation bounds 内；
- clipping rate 合法；
- Android/Type-B lifecycle 合法；
- 五动作各自只使用一个非零 best-EMA checkpoint digest；
- 1000-step schedule、50-step DDIM、`eta=0`；
- selector_used = false。
- shard、unit NPZ、unit audit 与 formal audit 的 strict runtime 字典/digest 必须完全一致；
- 所有 shard manifest 和 500 unit 均绑定 `generation_base_seed=20260713` 与 `generation_batch_size=32`；
- 逐条重算 100,000 个 ConditionRequest seed、100,000 个 DDIM noise seed 和完整 request digest；
- 100,000 个 ConditionRequest seed 全局唯一、100,000 个 DDIM noise seed 全局唯一，且两类 seed 集合互不相交；
- 按 `fake_id` 聚合后的 `condition_set_sha256` 必须与 exhaustive pre-DDIM preflight 逐字一致。
- 完整 audit 额外发布 500 个 unit NPZ 的 SHA-256 map；最终 supervisor 会按当前 bytes 重算，
  detector builder 的 archive hash map 也必须与它逐项相同。

完整审计 receipt schema 为 `five_shot_generation_formal_audit_v4`；v3 及更旧 receipt 没有 schema 1.4 的 immutable runtime provenance，正式 supervisor 必须拒绝。

### 11.1 Detector ingress 审计

正式 500 个 unit 通过生成审计后，还要在独立 PAD 输入边界运行：

```bash
python scripts/audit_generation_pad_export.py \
  --generation-root /new/output/path \
  --split-json /home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json \
  --require-formal
```

`generation.pad_export.load_generated_action_tree()` 是 generator shard 到 detector builder 的唯一适配 API。每 action 返回 20,000 条 `RawTrajectoryRecord` 和 `trajectory/features.py` 特征；正式 gate 再次验证 schema 1.4、strict runtime digest、batch=32、两类 seed derivation、100 users × 200、fixed split、全局 fake ID、50-step DDIM、`eta=0`、1000-step schedule、单一 best checkpoint、无 selector、pinch Type-B snapshot 与 keystroke gap。最终 detector builder 把这些 fake 与同一 v2 real corpus 合并：fake 按固定用户做 70/10/20；real 则在每个 user/action 内按完整 event group 做 60/20/20，因此 real test 含全部 100 users 的 held-out event，而不是只含 20 个用户。

唯一正式 generation → detector dataset 命令为：

```bash
python scripts/build_trajectory_pad_bundle.py \
  --real-dir /path/to/trajectories_full_v2 \
  --fake-archive-dir /new/output/path \
  --output-dir /new/detector_dataset \
  --fake-user-split /home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json \
  --reference-registry-map /path/from/training/reference_registry_map.json

python scripts/run_trajectory_benchmark.py \
  --dataset-dir /new/detector_dataset \
  --fake-user-split /home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json \
  --output-dir /new/complete_benchmark
```

Builder 不接受旧的 extractor-style fake 文件；它直接读取 500 个 generated unit archive，强制 fake 为 14k/2k/4k，每动作 20k、全局 100k，并把 reference 与 detector real pool 的实际 overlap 完整记录。Real event pool 的 60/20/20 hash ranking 与 reference registry 相互独立；若某 ref 落在 detector real train/val/test，它会按该 pool 正常参与，并不会被隐瞒、重分配或删除。

## 12. 已运行 smoke

```bash
python -m unittest tests.test_generation_pipeline -v
```

generation 定向回归覆盖：

- 500 units / 100,000 计划；
- 200 条固定同一组 5 refs；
- registry round-trip 与 hash；
- train-only shrinkage 多样性及 test non-ref 隔离；
- orientation-first 混合方向反例；
- 200 条 keystroke 新序列多样性、完整 ref sequence copy=0；
- 首末 keycode 改变会改变 endpoint，token 0 fallback 有明确来源；
- staggered pinch absolute DOWN/UP；
- HMOG one-user 五动作 numeric corpus 读取；
- 五动作各一次真实 50 denoiser-call DDIM；
- pinch global lifecycle、keystroke discrete contacts；
- numeric archive、严格 audit；
- atomic publish/resume；
- 同 seed/batch exact resume、不同 base seed 或不同 batch 拒绝、checkpoint SHA mismatch 拒绝与完整重审计；
- 篡改逐样本 ConditionRequest seed、DDIM noise seed 或 `condition_request_sha256` 均 fail closed；
- canonical request set digest 不受输入/shard 顺序影响，重复 fake ID 拒绝；
- 1 ms largest-remainder timeline、精确端点；
- 五动作 generated shard → PAD round-trip，pinch Type-B snapshot 与 keystroke gap；
- formal best-EMA / 1000-step checkpoint gate。

此外保留了早期真实 HMOG one-user tap archive smoke：

```text
results/generation_smoke_real_tap/
```

该目录早于 integer-ms 与 key endpoint 修正，不作为当前版本最终证据。当前版另行运行并保存在：

```text
results/generation_smoke_real_tap_integer_v2/
```

`integer_v2` 目录仅保留作历史追踪：它的 keystroke letter 判定误用了 Android 29..54 codebook，**不得在报告、表格或模型选择中引用**。修正 ASCII codebook 后的五动作 smoke 使用 `integer_v3` 新目录，绝不覆盖历史产物。五动作均为固定 reference row=1、trajectory/Android timestamp 全为 integer ms、endpoint/time error=0、exact replay=0、complete metadata copy=0、selector=false；keystroke 还要求 sequence 唯一、complete key-sequence copy=0、`n_letters` 与 ASCII code一致、首末 endpoint source 可审计。随机小模型的 clipping 不用于正式结论，正式 checkpoint 仍须通过 5%/25% clip gate。所有 smoke 只验证管线，不冒充正式模型；正式入口拒绝该 schedule，并只接受 1000-step best-EMA checkpoint。
