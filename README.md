# IMU 与完整触摸轨迹生成、检测和综合评测

本仓库是当前正式研究代码与文档的安全源码镜像，覆盖：

GitHub：<https://github.com/WMSlalalala/imu_gen>

- IMU 正式评估、runtime cache 与统一双后端发布接口；
- HMOG 五动作完整轨迹提取：tap、scroll、swipe、pinch、keystroke；
- 每用户/动作固定五条真实参考的five-shot变长diffusion；
- 不同时长ConditionRequest和100,000条正式轨迹生成协议；
- 5动作×5模型的25个trajectory PAD检测器；
- IMU score、trajectory score和跨模态一致性组成的5个total detector；
- fully user-disjoint与排除five-shot reference的补充敏感性评测。

## 当前状态

IMU正式交付已经通过接口、缓存、封装、runtime和评测审计。轨迹侧正式训练仍在进行：tap、scroll、swipe已经完成100 epochs，pinch和keystroke继续由后台supervisor训练。100k真实生成、25个轨迹检测器和5个综合检测器只有在全部上游fail-closed门通过后才会运行。

唯一详细状态、问题和未完成项见：

- [IMU与轨迹交付状态及问题清单](trajectory_estimator_pack_20260721/docs/IMU与轨迹交付状态及问题清单.md)
- [轨迹生成方法与HMOG标准数据集测试说明](trajectory_estimator_pack_20260721/docs/轨迹生成方法与HMOG标准数据集测试说明.md)
- [共享EventPlan、Trajectory生成与TotalDetector](trajectory_estimator_pack_20260721/docs/共享EventPlan、Trajectory生成与TotalDetector.md)

## 目录

| 目录 | 内容 |
|---|---|
| `trajectory_humanization_full_20260722_v16_numeric_recovery/` | 变长轨迹diffusion、100k生成、25个PAD、测试和正式supervisor |
| `trajectory_estimator_pack_20260721/` | 轨迹runtime、IMU配对、跨模态一致性、total detector与独立审计 |
| `trajectory_pad_supplement_20260722/` | 双侧user-disjoint及reference-exclusion补充评测 |
| `android_duration_time_fixed_20260720/imu_release_20260721/` | IMU cache/online统一发布接口 |
| `github_sync/` | 安全增量提交与推送脚本 |
| `agent_handoff/` | 下一次Agent恢复工作所需的轻量状态缓存、路径和续跑规则 |

## 数据与大文件

仓库不包含以下内容：

- HMOG原始归档或解压后的用户数据；
- 训练/生成NPZ、100k轨迹、IMU cache；
- `.pt/.pth/.ckpt` checkpoint；
- 正式results、score dump、bootstrap大数组、日志；
- 凭据、token或本机环境文件。

复现实验时需按中文方法文档准备数据，并用manifest/SHA绑定本地输入。GitHub上的测试通过只说明代码与协议可执行，不等于正式100k指标已经完成。

## 测试

轨迹v16回归：

```bash
cd trajectory_humanization_full_20260722_v16_numeric_recovery
/home/mwang49/miniconda3/envs/hml/bin/python -m unittest discover -s tests -v
```

综合检测器回归：

```bash
cd trajectory_estimator_pack_20260721
/home/mwang49/miniconda3/envs/hml/bin/python -m unittest discover -s tests -v
```

补充PAD协议回归：

```bash
cd trajectory_pad_supplement_20260722
/home/mwang49/miniconda3/envs/hml/bin/python -m unittest discover -s tests -v
```

## 发布纪律

正式结果只在当前bytes的count、schema、identity、SHA、阈值来源、时长分箱和用户级bootstrap审计全部通过后更新。任何失败、数据不规范、监控误操作或尚未解决的方法学问题均保存在问题清单，不能用删除失败记录的方式制造“全部通过”。
