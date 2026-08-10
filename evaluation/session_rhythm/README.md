# 会话节律检测器（补充评测）

一个新的、会话级的检测器，问那 6 个逐手势检测器结构上答不了的问题：**一串动作的节律像不像人**。原 6 个检测器全程未动（只读、仅作"逐事件视角"的对照）。

这是主实验之外的补充：主实验（`../main/`）逐手势评测发布版；本目录把动作连成会话，测**动作间的机器节律**是否露馅，以及我们的 pacing 能否化解。

## 一句话

未整形的机器节律 100% 被抓；v1 pacing（log-normal，只对齐中位）被抓 72–100%；**v2 pacing（从真人经验分布抽，带长尾）被抓率压到真人误报的地板**。但这几乎是套套逻辑（从人类间隔分布抽、又只用间隔统计去检测），真正的会话级威胁在**内容耦合 + 连续 IMU**，不在间隔时序——与本仓库对 joint 的批评（issue C）指向同一处。

## 文档

| 文件 | 内容 |
|---|---|
| `docs/SPEC_CN.md` | 检测器规格：判据、FRR=5%/1% 操作点、三臂、防自证协议 |
| `docs/EVALUATION_CN.md` | 结果、机制（间隔分布对比）、v1→v2 改进、诚实边界、复现 |
| `docs/ASSESSMENT_ABC_CN.md` | 对三个批评的评审：competence gate / 同人-异人距离 / joint 到底证明了什么 |

## 代码与产物

- `code/assemble_sessions.py`：从 HMOG 真实时间线重建真人会话节律，naive/paced/paced_emp 同骨架换间隔。
- `code/session_detector.py`：训练（genuine vs naive_jitter）+ 开发集选 FRR 操作点冻结 + 测试集报被抓率。
- `results/*.json`：逐窗逐操作点的被抓率（RF 与 LogReg）。

**不入库的**：`sessions_*.jsonl`（四臂会话流，约 21 MB，可由 `assemble_sessions.py` 从原始 HMOG 时间线一键再生）。与主实验同一原则——仓库只装代码与小结果表，事件数据在库外。

## 复现

```bash
python code/assemble_sessions.py                 # 重建四臂（需原始 HMOG 时间线 + actreal pacing）
python code/session_detector.py --model logreg
python code/session_detector.py --model rf
```

`assemble_sessions.py` 依赖 `/mnt/share/mwang49/data7/actreal_agent/actreal/pacing.py` 的 DelayPolicy（paced 臂）；v2 的经验分布模式即在该 DelayPolicy 中实现。
