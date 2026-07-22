# 正式流水线 supervisor

这个目录只负责编排，不实现或修改 diffusion、生成、轨迹转换、detector 和指标。
正式顺序固定为：

```text
v2 corpus 全事件审计
  -> 独立 E2E smoke（非正式结果）
  -> 100 users × 5 actions × 200 的完整 ConditionRequest gate
     （正式 batch=32；共 3,500 batches；共享 canonical digest）
  -> 每 action 真实 train+固定5refs 吞吐/显存 probe（统一在干净 cuda:0 串行比较）
  -> 5 action / 100 epoch diffusion（cuda:0 与 cuda:1 各自串行队列）
  -> validation 20/40/60/80/100%，best EMA + last
  -> 两个 GPU shard / 100,000 条 / 50-step DDIM
  -> 500 unit 完整性、seed、ConditionRequest digest、泄漏、replay、clipping 审计
  -> 5-action detector bundle
  -> 10 个 Deep longest-event batch probes
  -> 25 个 action-detector pair：3 Feature PAD + 2 Deep PAD
  -> 严格 merge、独立指标复算审计和最终报告
```

## 1. 先做 dry-run

```bash
cd /home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713
bash orchestration/test_dry_run.sh
```

也可查看完整 JSON：

```bash
/home/mwang49/miniconda3/envs/hml/bin/python orchestration/formal_supervisor.py \
  --config orchestration/formal_pipeline_config.json --dry-run
```

dry-run 会逐个执行所有正式 CLI 的 `--help`，检查后续阶段所需参数真的存在，并构造
命令模板；它不会建立 checkpoint、fake 或 detector 结果，也不会查询/占用 GPU。CLI
参数缺失返回码 2。training batch 要等 throughput probe 后固化，Deep pair batch 要等
longest-event probe 后固化；manifest 用 finalized 字段区分模板与 exact argv。

当前正式 builder 的必要接口是 `--fake-archive-dir`，即直接消费 generation 的 500 个
action-user archive。旧 `--fake-dir fake_trajectory_{action}.npz` 不是本正式链路，避免把
不同 schema 误接起来。

## 2. 先跑 launch gates，审核后再授权正式启动

配置初始固定为：

```json
"formal_launch_authorized": false
```

此时普通 formal run 在创建/修改正式 state 之前就会被拒绝；只能后台执行 gates：

```bash
bash orchestration/launch_gates_only.sh
```

它先运行 CLI / split / disk preflight，再运行 corpus audit、E2E smoke 和完整 100k
ConditionRequest preflight，绝不会进入 throughput、training、generation 或 detector。E2E
会等待指定 GPU 干净。完成状态为 `gates_complete_awaiting_formal_authorization`。

E2E smoke 只用 20 个 optimizer steps，因此它的硬门槛是 loss/EMA/checkpoint、完整
50-step DDIM、Android archive 的结构与物理合法性以及 25 个 detector 接口。它同时用正式
5% aggregate / 25% per-event clipping 阈值做诊断并原样保存结果，但短训练模型不以该质量
诊断决定 smoke 成败；这两个 clipping 阈值只在 100-epoch checkpoint 生成全部 100,000 条
后的正式 generation audit 中作为硬门槛。这样既不把 quick smoke 冒充模型质量，也不会因
未训练充分的随机输出阻断本来通过的接口测试。

人工审核的三份主文件是：

```text
<run_root>/audits/corpus_audit.json
<gate_root>/v2_e2e_smoke/e2e_smoke.json
<gate_root>/all_condition_requests.json
```

state 同时保存 `launch_gate_evidence`，绑定三份文件、五个 corpus 和 split 的 SHA-256。

审核三个 gate 的路径、SHA-256、计数与状态后，才把配置中的操作授权改为 `true`，然后：

```bash
bash orchestration/launch_formal_pipeline.sh
```

授权布尔值是操作确认，不属于实验语义，因此被明确排除在 immutable experiment config
hash 之外；false 下完成的 gates state 可以在 false→true 后原地续跑。除这一布尔值外，
任何数据、模型、seed、指标、资源策略或协议参数变化都会改变 config identity，并拒绝接续
旧 state。formal launcher 自身也会在 `setsid` 前检查授权和 durable gates-only evidence；
Supervisor 在改变正式 state 前重新计算三个 gate 的当前结果和 SHA。没有先完成 gates-only，
或审核后任一 gate/corpus 已变化，即使布尔值为 `true` 也不能开始 training，必须重新审核。

launcher 用 `setsid` 脱离 VS Code/terminal；supervisor 的每个长任务也使用独立 session。
退出前端不会给它们发送终端 hangup。关键日志只有：

```text
<run_root>/logs/supervisor.log
<run_root>/logs/supervisor_gates_only.log
<run_root>/logs/corpus_audit.log
<run_root>/logs/e2e_smoke.log
<run_root>/logs/condition_preflight.log
<run_root>/logs/throughput_probe__<action>__bs<batch>.log
<run_root>/logs/throughput_probe__<action>__selected_100steps.log
<run_root>/logs/training__<action>.log
<run_root>/logs/generation__shard_<id>.log
<run_root>/logs/generation_audit.log
<run_root>/logs/detector_bundle.log
<run_root>/logs/detector_probe__<action>__deep_pad__<detector>.log
<run_root>/logs/detector_pair__<action>__<family>__<detector>.log
<run_root>/logs/benchmark_merge.log
<run_root>/logs/benchmark_audit.log
```

## 3. 断电恢复与不覆盖

- `supervisor_status.json`、`command_manifest.json`、checkpoint maps 都原子发布。
- 正式训练 batch size 不靠猜测：每 action 都在同一张干净 `cuda:0` 上串行跑
  32/64/128/256 四个
  短候选。长度使用 canonical model timeline，而不是 raw flat-row 数；target `T` 与五个
  refs 的 `Tr` 分开统计，keystroke 还独立统计 target/ref key-token padding `K/Kr`。每个候选对 5 个确定性 train epoch 的全部 batch 建立长度分布，
  实测覆盖普通区、昂贵尾部和人工组合的全局最坏满 batch，再用单调分段插值累计预计完整
  epoch optimizer 时间。CUDA OOM 或超过 120 秒候选预算只记为该 candidate 资源/运行时间
  不合格；其他异常使流水线失败。按 `projected_full_epoch_examples_per_second` 选最优稳定
  candidate，同分取较大 batch，并额外跑 100 个覆盖全分布的 measured steps。正式 training 启动前两张卡
  都必须满足所选 peak VRAM + 6 GiB 安全余量。所选 batch、候选失败、peak allocated/reserved VRAM、corpus/
  split/registry/result hash 全写入 `manifests/throughput_selection.json`，随后才固化 exact
  training argv；probe 不读 val/test target，不改模型、数据或截断策略，也不写正式 checkpoint。
- throughput probe 与正式 training 的完成条件包含实际 runtime determinism audit。
  每个 candidate result、选中的 100-step result 和正式 `run_manifest.json` 中的
  `runtime_determinism` 必须与下列精确字典相等：

  ```json
  {
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms_enabled": true,
    "deterministic_algorithms_warn_only": false,
    "cudnn_benchmark": false,
    "cudnn_deterministic": true
  }
  ```

  引擎在导入 `torch` 前固定 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，并调用
  `torch.use_deterministic_algorithms(True)`（warn-only 关闭）、
  `torch.backends.cudnn.benchmark=False` 和
  `torch.backends.cudnn.deterministic=True`。candidate 不匹配则不得参与 batch
  选择；selected probe 不匹配则不得固化 training argv；正式 run manifest 不匹配
  则该 action 不得标记 training complete。字段缺失、多出或任一类型/值不一致均
  fail closed，不能只凭进程退出码接续。
- 50-step DDIM 后的 hard timing projection 不使用非确定性的 CUDA `cumsum`：每根 pointer
  的 1-D interval 权重在 CPU 做 deterministic prefix sum，再按原 dtype/device 拷回；这一步
  只发生在无梯度的采样后物理投影，不把轨迹生成替换成 CPU 模板。
- E2E、throughput、正式 training 和 Deep longest-event probe 这四个 clean-GPU gate 若遇到
  外部任务或 `nvidia-smi` 瞬时失败，state 写
  `waiting_for_clean_gpu`、snapshot/error 和 heartbeat；不会启动 child，也不会把资源等待误记成
  实验失败。GPU 变干净后自动继续；`STOP_REQUESTED` 仍可终止等待。generation 和正式
  detector pair 使用设备串行队列，但不另设 clean-GPU gate。
- training 每个 epoch 以 staged checkpoint → durable journal → immutable best → best manifest →
  last → metrics 的事务提交；七个断电窗口均有 fault-injection 回归。启动时先 reconcile，
  再从 `last.pt` 恢复，metrics `(type, completed_epoch)` 不允许重复。checkpoint 和 epoch
  journal 均保存精确 runtime 字典；orphan/staged/last 在任何 best/last/metrics 发布前都必须
  通过 config/source/runtime 一致性校验，训练协议为 `trajectory_diffusion_strict_five_ref_v2`。
- training worker 独立原子发布 `training_progress.json`；supervisor 核对 PID/run-instance、
  source/config/device、有限 loss/grad norm、counter/sequence 单调性和最后成功进度。600 秒无有效
  worker progress 会立即失败，不会被 supervisor 自身 heartbeat 掩盖。supervisor 重启后
  reattach 已有 detached child 时仍会刷新观测 heartbeat；若发现无效/停滞 progress，包括
  reattached child 在内的当前并行任务都会收到 SIGTERM，避免 supervisor 失败后留下孤儿训练。
- 每个 best entry 记录并复核实际 checkpoint SHA-256、`validation_selected_best` role 和
  `ema.shadow` 推理权重；`last.pt` 通过 `last_state.json` 绑定实际 SHA/size/progress/
  source/config。最终 gate 还会加载 best/last，检查 model/EMA 键与 shape、全部 tensor 有限、
  optimizer/scaler/RNG 可恢复、完整 100 epochs 的 loss/count/step 以及 validation record-low history。
- generation sampler 在反序列化前再次把 manifest protocol/source/schedule/role/history/progress
  与同一份 checkpoint bytes 的 SHA-256 绑定，关闭 supervisor 检查后到生成加载前的替换窗口。
- 若首次 epoch/checkpoint 前断电，training 目录非空但没有 `last.pt`，旧目录会整体移动到
  `orphaned_attempts/`，而不是删除或覆盖；随后从头启动该 action。
- generation 每个 action-user unit 原子发布；重启同一 shard 时先核 schema 1.4、`eta=0`、immutable
  strict-runtime SHA-256、base seed、
  batch=32、逐样本 ConditionRequest seed 与 domain-separated DDIM noise seed、checkpoint/registry/
  split 与 canonical ConditionRequest digest，再严格验证/跳过。
- 两类逐样本 seed 在 100k 范围内分别唯一且 domain 互不相交；DDIM hidden normalization/conv
  排除 padding，reference occupancy 使用自身 valid 长度。不同 batch shape 只承诺严格数值容差一致，
  不宣称浮点 kernel 逐 bit 相同；正式 200 条 unit 始终使用固定 32/32/32/32/32/32/8 边界，
  unit-level resume 不改变该边界。
- Condition gate 和 generation archive/final audit 共用
  `trajectory_condition_request_canonical_v1`；最终 100k set digest 必须逐字一致，防止 preflight
  检查一套条件、DDIM 实际消费另一套条件。
- 同一个 device 同时最多一个本流水线长任务；五动作完整训练，不做 sample cap。
- 旧 condition JSON、E2E 目录或 partial detector bundle 若与当前协议不符，会先整体移动到
  `orphaned/` / `orphaned_attempts/`，不会同名覆盖历史证据。
- 一个当前子任务返回非零或完成 gate 不通过，立即终止本 supervisor 启动的并行 sibling，
  标记 `failed` 并停止后续阶段。
- 代码或数据问题修好后才能显式传 `--resume-failed`。默认不会无限重试失败代码。

状态：

```bash
/home/mwang49/miniconda3/envs/hml/bin/python orchestration/formal_supervisor.py \
  --config orchestration/formal_pipeline_config.json --status
```

协作停止请求：

```bash
/home/mwang49/miniconda3/envs/hml/bin/python orchestration/formal_supervisor.py \
  --config orchestration/formal_pipeline_config.json --request-stop
```

恢复前先明确移除 `STOP_REQUESTED`，再使用 `--resume-failed`。这不会删除已完成 checkpoint、
unit archive 或 detector 产物。

后台恢复可直接使用：

```bash
RESUME_FAILED=1 bash orchestration/launch_formal_pipeline.sh
```

gate 阶段失败修复后相应用
`RESUME_FAILED=1 bash orchestration/launch_gates_only.sh`。

## 4. 完成 gate

Supervisor 不以“进程退出 0”单独判完成：

- corpus：完整 action audit 必须绑定当前 split 路径/SHA、五个 NPZ 路径/SHA/size 和逐事件计数；
- E2E：绑定当前 corpus manifest/audit/formal-audit、split、exact runtime 字典/digest、exact smoke 配置、五 action training /
  generation / 25 detector interfaces 和全部 artifact hashes；
- Condition：100,000 条、batch=32（700 batches/action）、全部 source code=2、全局 fake_id 唯一、
  当前 corpus/split/registry/prior hashes 和完整 request-set digest；
- throughput：所有参与比较的 candidate 与选中的 100-step result 都必须包含
  上述精确 `runtime_determinism` 契约、canonical `T/Tr/K/Kr`、5-epoch profile hash、最坏批次
  实测耗时和完整 epoch 投影，否则不能完成 batch 选择；
- 每个训练：100 个完整 train epoch、20/40/60/80/100 五次 full validation、val-only
  best、EMA、last、固定 registry，且 `run_manifest.json.runtime_determinism` 必须与上述
  精确契约完全相等；
- generation：两个 `five_shot_generation_shard_manifest_v4` 必须精确覆盖各自 `(action,user)` 和
  unit path，并绑定 exact runtime 字典/digest、NPZ/unit/manifest 三层 `eta=0`、batch=32 与两类 seed derivation/count；随后
  `five_shot_generation_formal_audit_v4` 强制 500 units / 100,000 fake / 5 refs / no selector /
  no replay，并发布、重验 500 个 NPZ 的 SHA map，detector bundle 必须绑定同一 map；
- detector bundle：500 个 generation archive hashes、registry map、当前 real corpus、五 action
  output SHA 全绑定；fake users 固定 70/10/20（14k/2k/4k），real test 保留全部 100 users；
- benchmark：每个 pair 绑定当前 dataset SHA 和完整 config；25 个 action-detector pair、
  validation-only 阈值、固定阈值 test 指标、曲线、
  500 次 user-level bootstrap、Deep best+last；
- 最后重新执行整个 completion closure，包括 25 个 pair 的 score/curve/bootstrap/best/last
  hash 与指标复算。先原子发布 `FINAL_REPORT.md`，再把带 report SHA 和全部权威 receipt
  snapshot 的 `final_audit.json` 作为最后 commit marker；发布后再次复核才标记 complete。

配置和逐阶段固化的 exact argv 写入 `<run_root>/command_manifest.json`；probe 前对应命令仍是
显式模板。状态文件绑定 experiment config SHA-256（唯一排除操作授权字段），修改任何实验参数
后不能误接旧 run。
