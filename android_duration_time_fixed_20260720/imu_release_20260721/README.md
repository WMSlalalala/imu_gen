# Audited IMU release 20260721

这里把原先分散的 IMU 在线生成器和正式部署 cache 查询统一为一个入口：

```python
from imu_release import IMUReleaseService

# 经过 138,200 条 runtime cache 审计的低延迟查询。
cache = IMUReleaseService(mode="cache", seed=42)
result = cache.generate("swipe", user_id=6, duration_ms=550,
                        xy_start=(300, 1500), xy_end=(800, 600),
                        orientation_id=0, noise_seed=123)

# 真正加载 action-specific checkpoint 做 five-shot diffusion 在线采样。
online = IMUReleaseService(mode="online", seed=42, device="cuda:0")
result = online.generate("keystroke", user_id=6, text="hello",
                         n_keys=5, n_letters=5, duration_ms=1500,
                         noise_seed=123)
```

两个后端都输出 100 Hz、有限、C-contiguous 的 `float32[N,6]`、逻辑时间和纳秒时间轴。

## Fail-closed 启动门禁

默认初始化会同时核对：

- `results/formal_pipeline_status.txt` 最新行必须是 `complete`；
- runtime cache 必须是 strict formal `pass`，并覆盖 138,200/138,200 文件；
- detector finalizer 必须 `passed=true` 且含 90 行正式 IMU 结果。

任一项不满足就拒绝启动。

## 两个后端的边界

- `cache`：适合部署查询，支持 user/action/duration/active_len/start time/orientation/XY/n_letters 和 strict/nearest matching。显式 `noise_seed` 让同一查询不依赖调用顺序。
- `online`：真正重新采样 diffusion，支持完整 EventPlan 的 text/n_keys、pinch span、显式 noise seed 等条件。

cache 文件没有完整的 pinch span 以及逐字符 text/n_keys 检索索引。为防止条件被静默忽略，向 cache 后端传这些字段会直接报错并要求改用 `mode=online`。

## CLI

```bash
cd /home/mwang49/real-human/imu_gen/final/android_duration_time_fixed_20260720/imu_release_20260721
python scripts/query_or_generate_imu.py \
  --mode cache --action tap --user-id 6 --noise-seed 123 \
  --output results/tap.npz --summary results/tap.json
```

## 验证

```bash
python -m unittest discover -s tests -v
```

测试包含正式审计门禁、五动作真实 cache 查询、输出 schema，以及显式 seed 的调用顺序独立性。
