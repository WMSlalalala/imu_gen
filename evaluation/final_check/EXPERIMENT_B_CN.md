# 实验 B：同人 fake–real 与异人 real–real 的距离

回答"生成是否保留**目标用户**的行为风格"。用距离 D（不是相似度），支持"保留"的正确不等式是 **\(D_{\text{fake}} < D_{\text{inter}}\)**（假比陌生人更靠近目标本人）。理想链 \(D_{\text{intra}} \le D_{\text{fake}} < D_{\text{inter}}\)。

- \(D_{\text{fake}}=D(\text{fake}_u,\text{real}_u)\)、\(D_{\text{inter}}=D(\text{real}_v,\text{real}_u),v\neq u\)、\(D_{\text{intra}}=D(\text{real}^{(1)}_u,\text{real}^{(2)}_u)\)。
- 特征：逐事件的 trajectory + IMU 每通道 [均值/std/min/max] + 帧数（45 维），按动作标准化。距离=到该用户 real 质心的中位欧氏距离。
- **按动作匹配**；每名 test user 一个 \(\Delta_u=D_{\text{inter},u}-D_{\text{fake},u}\)；报 20 名 test user 中 \(\Delta_u>0\) 比例、中位、**user-clustered bootstrap CI95**（重采样用户，10000 次）。

## 结果（20 test users）

| 动作 | \(D_{\text{intra}}\) | \(D_{\text{fake}}\) | \(D_{\text{inter}}\) | 理想链成立% | \(\Delta>0\) | 中位\(\Delta\) | CI95 | 判定 |
|---|---|---|---|---|---|---|---|---|
| keystroke | 4.64 | **4.13** | 6.07 | 25% | 100% | 2.198 | [1.40, 2.48] | **强保留** |
| tap | 5.02 | **5.14** | 5.91 | 40% | 80% | 0.784 | [0.49, 0.98] | **保留**（理想链最干净） |
| scroll | 5.01 | **5.57** | 5.92 | 60% | 80% | 0.364 | [0.07, 0.74] | **保留**（较弱，CI 不含 0） |
| swipe | 5.09 | 5.52 | 5.63 | 40% | 50% | −0.004 | [−0.11, 0.26] | **不确定**（CI 含 0） |
| pinch | 4.62 | 5.82 | 5.30 | 15% | 20% | −0.384 | [−0.57, −0.09] | **不保留**（诚实负结果） |

## 读法

- **keystroke / tap / scroll**：\(D_{\text{fake}} < D_{\text{inter}}\) 且 bootstrap CI 不含 0 → 假事件确实比一个普通陌生人更靠近目标本人，**保留了目标风格**。tap 的理想链最干净（\(5.02 \le 5.14 < 5.91\)）。
- **swipe**：\(D_{\text{fake}} \approx D_{\text{inter}}\)，CI 含 0 → 假事件离目标和离陌生人差不多，**不确定**。
- **pinch**：\(D_{\text{fake}} > D_{\text{inter}}\)，CI 全在负侧 → 假 pinch 比陌生人离目标**更远**，**没有保留目标风格**。这是诚实的负结果，如实报，不挑。

## 两条要写进论文的诚实注记

1. **keystroke 的 \(D_{\text{fake}} < D_{\text{intra}}\)（4.13 < 4.64）**：假事件比目标本人的两半还靠近质心，说明假 keystroke **彼此过于相似**（多样性低于真人），是"太干净"而非"不像"。方向上支持保留风格，但要标注多样性偏低。
2. **这个实验只回答"是否保留目标风格"，不能替代 detector competence gate**（实验 A）。距离近不等于检测器抓不住，反之亦然。两个实验分开报。

## 复现

```bash
python code/style_distance.py     # → scores/style_distance.json（含逐用户 D 值）
```

判据、特征、bootstrap 全部写死在 `code/style_distance.py`，不看检测器结果、不挑用户。
