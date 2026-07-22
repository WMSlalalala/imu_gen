# Generation seed / resume provenance audit（2026-07-13）

## 结论

旧 archive schema 1.1 保存了逐样本 `seed` 和大部分 condition/request 字段，但没有保存产生这些 seed 的 generation base seed。因此，同 checkpoint、registry、prior 下，调用方改用另一个 base seed 时，旧 `generate_unit(..., resume=True)` 仍可能错误复用已有 NPZ。旧 unit audit 也没有按协议公式重算 seed。这是正式生成的 provenance 阻断，已在任何 100k 生成启动前修复。

## 修复后的协议

- 正式 generation base seed 固定为 `20260713`；reference registry seed 仍独立固定为 42。
- archive schema 先由 1.1 升为 1.2 保存 `generation_base_seed_scalar`，再升为 `1.3` 保存 `generation_batch_size_scalar` 与逐样本 `ddim_noise_seed`；当前 `1.4` 进一步保存 immutable `runtime_determinism_sha256[32]`。
- 每条 ConditionRequest `seed` 必须等于 `stable_seed(base_seed, action, user_id, sample_index)`。
- 每条 DDIM noise seed 必须等于 `stable_seed(condition_request_seed xor 0xDD1A50, action, user_id, sample_index)`；两类 seed 分开命名、保存和逐条重算。
- exhaustive preflight、shard manifest 和最终 audit 额外要求：100,000 个 request seed 各自唯一、100,000 个 noise seed 各自唯一，且两类 seed 集合互不相交。
- pipeline 把传给 `sample_ddim_seeded_batch()` 的同一组 noise seeds 传给 archive builder；builder 先与公式逐条比较再保存，避免“采样用一组 seed、receipt 事后写另一组 seed”。
- archive 新增 `condition_request_sha256[N,32]`。
- canonical request digest 覆盖 `ConditionRequest` dataclass 的全部 30 个字段，并有字段清单漂移断言。
- request set digest 按 `fake_id` 数值排序聚合，所以不依赖 batch、shard 或 resume 顺序。
- 独立 generator 保证逐样本初始 noise 与 batch/shard/resume 严格无关；masked hidden normalization、逐层 padding mask 与 per-reference local occupancy 保证不同 padded batch shape 下有效输出严格数值近似一致，但不声称浮点 kernel 逐 bit 相同。正式 batch 固定为 32，200 条 unit 固定为 `32×6+8`，unit-level resume 不改变边界。
- resume 先校验 schema/base seed/batch size/sample index/两类 seed/fake ID/checkpoint，再从固定 refs 与 train-only prior 重构完整 request，并核对 request digest、condition 数组、topology、geometry 和 Android lifecycle。
- 正式生成入口在导入 PyTorch 前固定 cuBLAS，并启用 strict deterministic、warn-only=false；unit NPZ、unit audit、shard manifest 和 formal audit 都绑定同一 runtime 字典/digest。sidecar 即使被重写，也不能替代 NPZ 内的 digest。
- archive builder、resume validator、独立 unit audit 与 PAD ingress 四层都要求 `ddim_eta_scalar` 为 finite float32 scalar 且精确等于 0；shard/final receipt 同时保存并复核 eta，不能由 manifest 的 0 掩盖 NPZ 非零值。
- 正式 checkpoint loader 在任何 `torch.load` 前验证 best manifest 的 protocol、SHA、source、schedule、role、EMA、history 和 progress，并从已验哈希的同一份 bytes 反序列化。
- 正式 CLI 在任何 corpus/checkpoint I/O 前拒绝非 `20260713` 的 `--seed` 或非 32 的 `--batch-size`。
- PAD adapter 只接受当前 schema；正式模式再次检查 base seed、batch=32、两类逐样本 seed、fake ID 与 request digest shape。
- 最终 100k audit 要求传入 exhaustive condition preflight，并要求全局与逐 action 的 condition set digest 完全一致。
- shard manifest 与完整 audit receipt 分别升级为 `five_shot_generation_shard_manifest_v4` 和 `five_shot_generation_formal_audit_v4`；旧 v2 receipt 必须 fail closed。

## archive 字段充分性

schema 1.1 已有 `sample_index/seed/fake_id`、固定五 refs 与 canonical hashes、carrier、duration/orientation/XY、pointer lifetime、pinch、keycodes/zero-flight、screen bounds 和 prior/registry/split/checkpoint digest，因此可以重构并核对几乎全部显式 condition。`contact_masks/event_ids` 没有作为第二份 request-only 数组重复保存，但它们在 generated trajectory 的 `flat_trajectory_contact_mask/flat_trajectory_event_id` 中按 pointer offsets 无损表示，audit 会与重构 request 逐点比较。

schema 1.1 不能可靠反推 **base seed 本身**；schema 1.2 虽保存 base seed/request digest，但仍不能证明 generation batch 与 DDIM noise seed；schema 1.3 仍不能证明生成时的 PyTorch/cuDNN/cuBLAS 确定性状态。因此不能靠 sidecar 对旧 archive 重新贴标签，必须使用 schema 1.4 显式保存并复核全部证据。

## 回归证据

定向测试覆盖：

- 相同 base seed 可 resume；
- 不同 base seed 拒绝 resume；
- 不同 generation batch 拒绝 resume；
- 篡改任一 per-sample seed 拒绝；
- 篡改任一 DDIM noise seed 拒绝；
- 篡改任一 request digest 拒绝；
- 篡改/删除 runtime digest、旧 schema 1.3、或仅伪造当前 sidecar runtime 均拒绝 resume；
- 非零/NaN/非 scalar eta 在 builder、resume、unit audit、PAD ingress 任一层均拒绝；
- best manifest 缺失/错误 SHA、source/schedule/role/history/progress 不一致或 checkpoint bytes 被替换时，在反序列化前拒绝；
- request set digest 对输入顺序不敏感，重复 fake ID 拒绝；
- 同一 request+noise 单独运行或与更长样本合批，完整 50-step DDIM 有效区 feature/timestamp 分别在 `1e-5` / `1e-4 ms` 内一致；
- 两类 seed 的 unit/shard/100k 唯一性与 domain-disjoint receipt 缺失或不一致时拒绝；
- 五动作 archive audit 与 PAD round-trip 继续通过。

执行命令：

```bash
/home/mwang49/miniconda3/envs/hml/bin/python -m unittest -v \
  tests.test_generation_pipeline tests.test_zero_flight_generation
```

定向测试使用 synthetic/临时目录与小规模真实 condition worker 等价检查，不启动任何正式训练或正式 100k 生成；以实际测试命令输出作为通过数量的权威证据。
