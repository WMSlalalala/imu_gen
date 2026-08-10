# 基线 02 / 03 — Diffusion-TS（轨迹 / IMU）

## 方法出处

[Y-debug-sys/Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS)，
Xinyu Yuan & Yan Qiao，*Diffusion-TS: Interpretable Diffusion for General Time Series
Generation*，**ICLR 2024**。编码器-解码器 transformer 的去噪扩散模型，
解码端把序列拆成趋势项（多项式回归）与季节项（傅里叶三角基），
训练目标是**直接重建样本**（而非噪声）并附加傅里叶域损失。
是当前通用多变量时序生成的代表性开源实现。

## 我们做了什么

模型、扩散过程、训练循环、采样全部是作者的代码，仓库 vendored 在 `DiffusionTS/`，
**未做任何修改**。我们的 `run_diffusion_ts.py` 只提供三样东西：

1. **数据**：`collect_genuine_windows(..., split="train", ...)` —— 只取 **70 个 train
   用户**的真人事件，dev / test 用户一条都不进生成器（否则会泄漏到后面要评分的集合）。
   每条事件按动作重采样到冻结的检测器窗口（tap 16 / pinch 100 / swipe 176 /
   scroll 208 / keystroke 512），用的是真人路径本身走的那个零阶保持观测器
   （IMU 是连续物理量，用线性插值）。往返畸变实测见总 README 3.2。
2. **归一化**：与作者 `CustomDataset` 完全一致 —— `MinMaxScaler` 铺平拟合，
   再线性映射到 [-1, 1]；采样后反向。
   （注：作者模型在采样中已把 `x_start` clamp 到 [-1,1]，所以我们的 clip 是空操作。）
3. **采样**：`Trainer.sample`，用作者保留的 **EMA 模型**。

超参照搬作者发布的真实数据集配置（`Config/etth.yaml` / `Config/sines.yaml`）：
`n_layer_enc=3, n_layer_dec=2, d_model=64, timesteps=500, sampling_timesteps=500,
loss='l1', beta_schedule='cosine', n_heads=4, mlp_hidden_times=4, kernel_size=1`；
求解器 `base_lr=1e-5, gradient_accumulate_every=2, EMA decay 0.995 / 每 10 步,
ReduceLROnPlateauWithWarmup(warmup 500, warmup_lr 8e-4)`。
**只有 `seq_length` 和 `feature_size` 随动作/通道数变化**。

训练步数 12,000（作者 sines 配置的值；etth 用 18,000，我们的数据集比它小）。
每个 (动作, 通道集) 各拟合一个模型 —— 序列长度在构造时固定，而每个动作有自己的冻结窗口。

- 轨迹模型：`feature_size=2`（x, y）
- IMU 模型：`feature_size=6`

## 端点绑定（轨迹专有）

见总 README 第 1 节。生成的路径经「一次旋转 + 一次等比缩放 + 一次平移」
搬到载体请求的端点上；tap 请求无位移，退化为纯平移（强行钉住两端会把 tap 的微移压平，
那正是我们自己方法里怀疑的 bug，不能拿它去坑基线）。

## 复现

```bash
bash run.sh /path/to/replay_dataset_zoh /path/to/output
```

依次：训练 5 个模型 → 采样 4,000 条/动作 → 合成样本库 → 建数据集 →
`verify_harness.py` → 30 格检测器 → FAR 表 → 生成质量指标。

## 质量验证

`results/quality_*.json` 里每个模型都有：

- `generated_vs_real`：作者自己的 discriminative / predictive score
- `control_real_vs_real`：真人窗口对半分跑同样指标 —— 数据允许的下限

两者接近 = 拟合到位。差距大 = 没训好，那么该模型的检测器分数不能当作 Diffusion-TS 的上限。
