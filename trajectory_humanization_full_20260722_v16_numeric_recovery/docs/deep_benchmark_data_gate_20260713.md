# Formal Deep PAD 数据覆盖门禁与 v1 历史问题（2026-07-13）

## 结论

历史输出 `results/trajectories_full` 的 raw keystroke 只覆盖 20 users，不能用于正式 PAD。
该历史阻塞已经由 `results/trajectories_full_v2` 解决：五动作均覆盖 100 users，完整
extractor audit 的 `formal_passed=true`。builder/runner 仍保留 fail-closed 门禁，不会静默
缩小任何 action 的用户或样本协议。

该问题已经定位：很多 HMOG soft-keyboard contact 不在低层 `TouchEvent.csv`，但
`OneFingerTouchEvent.csv` 对 100 users 仍保存与 KeyPress EventTime 精确匹配的真实
DOWN/UP x/y/pressure/size。extractor v2 已改为：优先完整 raw TouchEvent；缺失时使用
真实 OneFinger DOWN/UP 两端点，不插 MOVE、不推断键盘中心、不伪造 XY。缺失用户
smoke 已恢复 503/503 typing events，旧用户也补齐先前遗漏的 19 events。

v2 keystroke detector adapter 已实际转换全部 49,158 个 typing events，Feature 形状为
`[49158,34]` 且全 finite。其中 530 个 event 含 577 个合法 zero-flight 边界：相邻 key 的
UP/DOWN 同毫秒时保留两个有序 contact token，不插入虚假 gap；正 flight 才插 gap。

## v2 正式 real 产物（当前 formal source）

| action | events | raw rows | keys | users |
| --- | ---: | ---: | ---: | ---: |
| tap | 19,269 | 116,263 | 0 | 100 |
| scroll | 59,937 | 2,528,173 | 0 | 100 |
| swipe | 70,431 | 2,212,589 | 0 | 100 |
| pinch | 58,016 | 4,857,213 | 0 | 100 |
| keystroke | 49,158 | 1,942,826 | 707,088 | 100 |

权威审计：

- `results/trajectories_full_v2/formal_audit/formal_data_audit.json`；
- `formal_passed=true`，`completion_state=complete`；
- 五个 action 各自 100 users，合计 256,811 个 unique event IDs；
- 数据仍保留原始 label/touch 双时间轴和允许端点容差，没有为了审计通过而裁剪 swipe。

## v1 full real 产物实测（不可作最终 formal source）

| action | events | raw rows | users | min events/user（已出现 users） |
| --- | ---: | ---: | ---: | ---: |
| tap | 19,269 | 116,263 | 100 | 27 |
| scroll | 59,937 | 2,528,173 | 100 | 126 |
| swipe | 70,431 | 2,212,589 | 100 | 69 |
| pinch | 59,285 | 4,988,617 | 100 | 81 |
| keystroke | 9,818 | 750,797 | **20** | 222 |

证据：

- `results/trajectories_full/manifest.json`；
- 五个 `hmog_trajectory_<action>.npz` 的 `user_id/event_id/event_offsets`；
- `results/trajectories_full/audit.json`。

处理层原始 `hmog_keystroke.npz` 确实有 100 users、42,589 个 unique
`typing_event_id`；raw extractor 的 49,219 个 typing event 候选中只接受 9,818，拒绝
39,401，主要原因是一个或多个 key 没有匹配到完整 raw contact。接受的 20 个 user ID
并非简单的前 20 个，说明不是 `max_users=20`：

```text
0, 9, 14, 18, 25, 29, 35, 36, 39, 44,
46, 47, 54, 56, 60, 61, 62, 64, 70, 71
```

这个历史问题不能靠复制用户、允许 partial typing event、插值键间 XY 或把 IMU typing
chunk 当 raw 触摸轨迹来补足；v2 的 OneFinger fallback 只使用数据集中真实观测端点。

## 正式门禁

`scripts/build_trajectory_pad_bundle.py` 对每个 action 要求：

```text
real source users = 100
fake source users = 100
real train/val/test users = 100 / 100 / 100（每个 user 的 complete event groups 分别按 60/20/20 held out）
fake pools users  = 70 / 10 / 20
```

`scripts/run_trajectory_pair.py` 在加载每个 formal action bundle 并重新分 pool 后再次检查同一
覆盖条件。任一不满足即抛错，不生成 `benchmark_manifest.json`，也不会把降级结果标为
formal complete。

## Pipeline smoke 与尚待外部产物

真实语料 adapter/bundle 小型门禁也已逐动作执行，而不是只在 synthetic record 上测试：

```text
results/real_corpus_bundle_v2_smoke_20260713/summary.json
results/real_corpus_bundle_v2_smoke_20260713/summary.md
```

它分别读取五个 authoritative HMOG v2 archive，覆盖源事件
`19,269 + 59,937 + 70,431 + 58,016 + 49,158 = 256,811`，并对每类选出的完整真实事件执行
adapter → `trajectory_pad_bundle_v2` save/load round-trip → Feature linear SVM validation-only
流程 → raw TCN/Transformer finite forward。五类均通过，Feature 维数依次为
24/24/24/49/34。为了只测试 binary API，smoke 的 label-1 行是 real 行的精确镜像，不是
generator fake；其 FA/FRR/AUC 没有科学含义，不能进入正式结果表。

代码级 synthetic smoke 已完整覆盖五动作、25 个 action-detector pairs、50 条 operating
rows、25 张 test FA-FRR 图及 10 组 Deep best/last；该 smoke 仅证明 pipeline。正式 real
data gate 已通过，但正式 benchmark 仍必须等待满足 100 users × 5 actions × 200 的神经
fake archive、严格 bundle 和每个 Deep pair 的 longest-event batch probe。不得用 smoke
结果替代这些产物，也不得在 fake archive 缺失时启动或宣称 formal benchmark 完成。

最终共享代码稳定后，完整 `unittest discover -s tests -v` 连续三轮均为 84/84 通过，
耗时分别为 11.367s、11.220s、11.410s；该测试结论只证明实现门禁与可恢复执行链路，
不替代尚未产生的正式 fake archive 与正式 detector 指标。
