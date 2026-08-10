# 基线 04 — TTS-GAN（IMU）

## 方法出处

[imics-lab/tts-gan](https://github.com/imics-lab/tts-gan)，
*TTS-GAN: A Transformer-based Time-Series Generative Adversarial Network*。
生成器和判别器都是 transformer：生成器把隐向量线性展开成 `seq_len × embed_dim`，
过 transformer 编码块，再用 1×1 卷积映回通道数；判别器把序列切成 patch 做线性嵌入
（带 CLS token），过 transformer 后接分类头。

选它做 GAN 家族基线的理由很直接：**它论文里的实验对象就是 UniMiB-SHAR 加速度计数据**，
和我们要生成的惯性信号同类。

（原计划用官方 TimeGAN（NeurIPS'19）。它是 TensorFlow 1.15 + `tf.contrib`，
只能 CPU 跑 GRU，5 个模型里最长序列 512 步，实际跑不完；硬压迭代数会造成
「基线差是因为我们没训好」——正是要避免的情况。所以换成同家族但能在 GPU 上
获得公平训练预算的 TTS-GAN。）

## 我们做了什么

- `GANModels.Generator` / `GANModels.Discriminator` **一行未改** —— 它们本来就把
  `seq_len` / `channels` / `patch_size` / `latent_dim` 作为构造参数，我们只是传了不同的值。
- 优化器、`LinearLrDecay` 调度、平均生成器（`copy_params` / `load_params`）
  全部照 `train_GAN.py` 第 163–290 行搭；每个 epoch 直接调用作者的 `functions.train`。
- 超参用作者发布的 UniMiB 配方（`RunningGAN_Train.py` 的命令行），逐项照抄：
  `loss=lsgan, g_lr=1e-4, d_lr=3e-4, beta=(0.9,0.999), wd=1e-3, n_critic=1,
  latent_dim=100, patch_size=2, init=xavier_uniform, ema=0.9999`。
  这些是通过作者自己的 `cfg.parse_args()` 解析出来的，不是我们手写的字典。
- 数据张量按他们的形状 `(N, channels, 1, seq_len)` 组织；归一化用 `MinMaxScaler`
  映到 [-1, 1]，采样后反向（通道轴语义两个方向一致，已逐轴核对）。
- 采样用**平均生成器**（作者保留的那个），不是原始生成器。
- 训练集同样只取 **70 个 train 用户**的真人事件。

## 训练预算

作者对 UniMiB 发布的是 `max_iter=500000`。他们的数据集是单类别、几百条窗口；
我们每个动作有 4,300–7,000 条窗口，而且要拟合 5 个模型，其中 keystroke 是 512 步长序列
（patch 2 → 256 个 token）。500,000 次迭代在这里等于几十小时一个模型。

所以迭代数按动作缩放，写在 `sweep_tts_gan.sh` 里。
**预算够不够不是靠声明，而是靠测**：`results/quality_*.json` 里给出
作者自己的 discriminative score 和 real-vs-real 对照，两者的距离就是答案。

## 复现

```bash
bash run.sh /path/to/replay_dataset_zoh /path/to/output
```
