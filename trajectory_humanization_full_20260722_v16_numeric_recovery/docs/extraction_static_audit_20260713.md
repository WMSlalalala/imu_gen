# HMOG 五动作轨迹提取：静态与 one-user 运行审计

审计日期：2026-07-13。

本报告只审计已经冻结的轨迹提取算法与输出 schema。100-user 正式任务启动后，
没有修改提取脚本、字段定义、匹配容差或输出目录，也没有重启或覆盖正在写入的
正式结果。

## 1. 审计对象

| 对象 | SHA256 |
| --- | --- |
| `preprocess/extract_hmog_trajectories.py` | `621fb45d11ad6939f7576e6c7457a32d1818c0514b1f03d153eb1b403763d9fb` |
| `tests/run_one_user_smoke.py` | `c405877078f4edc8992c069c7951ee49b649c2c21970dfa4467bdce91f363324` |
| `docs/轨迹数据结构与预处理说明.md` | `c050da9e288b92e09b68ce922b4ed46a4707a8ae2d8e63541be8fbca8366269b` |
| 已审计事件层 `1_data_processing/preprocess.py` | `79c9d59058dc3b3720270ebd0b59f784e1275cfb7754f2c03a34b72a99540364` |

上述 hash 用于界定本报告结论。后续若修改任一文件，应重新执行 smoke 和静态审计，
不能沿用本报告的“通过”结论。

## 2. 静态审计结论

通过以下检查：

1. 提取器无条件用五个 action 调用已审计的 `extract_all`，再筛选输出，因此不会因
   只导出某一类而削弱五类互斥。
2. 原始 contact 以 `ActivityID + EventTime` 的 MotionEvent frame 重建，不把同一
   双指 frame 的两行误拆成两个时间点。
3. `tap/scroll/swipe/keystroke` 只接受完整单指 DOWN--UP contact；`pinch` 要求
   active phase 中实际同时存在两个 pointer。
4. 物理 contact 去重优先级固定为
   `keystroke > pinch > swipe > scroll > tap`。较低优先级派生事件不能复用已被较高
   优先级占用的同一原始 contact。
5. keystroke 使用已修正的 `PressType=0 DOWN, PressType=1 UP` 和 PressTime，逐键
   对齐原始 contact；默认缺任意一键即拒绝整个 typing event。
6. 输出使用 numeric-only flat arrays 和 offsets；没有 object dtype，不依赖 pickle。
7. 写入使用 append-only 临时二进制表，结束后才生成压缩 NPZ，并写 manifest/hash；
   不把全量触摸点常驻 Python list。
8. 命令行默认最多一个用户；100 users 必须同时显式指定 `--max-users 100` 与
   `--confirm-full-run`。
9. `py_compile` 通过；已有 one-user 结果再次用 `allow_pickle=False` 加载并执行完整
   validator，结果为 passed。

## 3. one-user smoke 结果

用户：HMOG external id `100669`，24 sessions。

| action | accepted events | rejected events | flat rows | frames | keys |
| --- | ---: | ---: | ---: | ---: | ---: |
| tap | 70 | 63 | 355 | 354 | 0 |
| scroll | 758 | 485 | 27,728 | 27,715 | 0 |
| swipe | 651 | 1 | 17,682 | 17,670 | 0 |
| pinch | 540 | 1 | 46,691 | 23,255 | 0 |
| keystroke | 958 | 19 | 30,726 | 30,642 | 6,809 |

验证器确认：

- 五个 NPZ 均可由 `np.load(..., allow_pickle=False)` 读取，object array 数为 0；
- `event_offsets`、`event_key_offsets`、`key_touch_offsets` 单调且末项精确等于对应
  flat/key 总数；
- 每个事件内原始时间非降序，事件首尾与保存的 touch 首尾一致；
- `active_mask <= valid_mask`，当前 flat row 全部 valid；
- 单指类 pointer count 严格为 1；pinch 最大 pointer count 为 2，且每个事件的
  active phase 均出现两个 pointer id；
- 每个保留 key 都有非空触摸片段，hold/flight、keycode 和 event 内 key index 一致；
- 原始时间未被伪装成 100Hz：各类中位帧间隔为 16--17ms，并存在大量非 10ms
  间隔。

机器可读结果：

- `results/smoke_one_user/smoke_validation.json`
- `results/smoke_one_user/audit.json`
- `results/smoke_one_user/event_audit.csv`

## 4. scroll/tap 高 rejection 的原因

### scroll

485 个拒绝中：

- 479 个：`raw_contact_reserved_by_higher_priority_event`；
- 2 个：`single_pointer_action_contains_multitouch`；
- 4 个：`no_matching_complete_raw_contact`。

因此 scroll 的高 rejection 主要不是容差或时间匹配失败，而是派生 scroll 和 pinch
共享同一个完整 DOWN--UP 物理 contact。若两类都保留完整轨迹，会造成跨类别 exact
contact 重复和数据泄漏。严格层只让高优先级 pinch 保留该 contact。

### tap

63 个拒绝全部为 `no_matching_complete_raw_contact`。人工检查最近 raw contact 后，
候选之间通常相差数秒到数十秒，并非把容差增大几毫秒即可恢复。严格输出不使用
起终点插值、不用不完整 contact，也不把不存在的轨迹补成 tap。

这两种拒绝都应保留在 audit 中；不能为了提高样本量而静默放宽。

## 5. 正式任务早期只读健康检查

正式命令：

```text
extract_hmog_trajectories.py --output-dir results/trajectories_full \
  --max-users 100 --confirm-full-run
```

在约第 11/100 位用户时检查：

- Python worker 持续运行，约占一个 CPU core；RSS 约 206MB；
- `.build` flat/event/key 二进制表和 `.event_audit.csv.tmp` 持续增长；
- 日志未出现 Traceback、Exception、OOM、NaN 或 killed；
- 多用户均同时出现单指、双指和 typing session，数量级随 session 类型变化，未见
  全零用户或字段爆炸。

### 已修复的 zip 目录枚举 quirk

部分 HMOG 用户 zip 含 macOS 元数据目录：

```text
__MACOSX/<user>/<user>_session_*/
```

复用的已审计 `sessions()` 原先会因此额外枚举一个
`session=<external_user_id>`。正式全量任务在形成任何最终 NPZ/manifest 前停止，
提取器现已增加 exact dataset-root gate：候选 session 只有在内层 zip 中真实存在
`<external_user_id>/<session_name>/` 前缀才会被处理。被过滤的资源目录名称写入每个
用户 manifest 的 `ignored_non_dataset_session_entries`，同时保存
`n_sessions_discovered` 和实际 `n_sessions`。因此新的正式结果中
`processed_session_count` 只统计真实数据 session，不再受 `__MACOSX` 影响。

## 6. 最终结论

当前版本通过静态检查和 one-user 完整运行验证；session-root 修复后已重新通过
`py_compile` 并从 user 1 重新启动 100-user 提取。
正式输出完成后仍必须做一次全量后验验证：numeric-only 加载、offset 终值、逐事件
时间单调性、pointer 约束、keystroke key-touch 非空、manifest SHA256 和 action
接收/拒绝原因汇总。one-user passed 不能替代最终全量后验验证。
