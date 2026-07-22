# Agent持续工作缓存说明

本目录用于让后续Agent在上下文被压缩、会话重启或机器重新连接后继续正式任务。

`latest_state.json`由`github_sync/update_agent_handoff.py`生成，并在每次30分钟GitHub同步前刷新。它只保存轻量、可公开的恢复信息：当前阶段、五动作epoch/step、监督器状态、GPU摘要、关键本机路径、checkpoint SHA和恢复规则。

它不上传HMOG原始数据、正式results、NPZ/NPY、训练checkpoint、大型runtime cache、日志或凭据。以上本机产物仍由正式manifest、count/schema/identity审计和SHA约束；新Agent必须按缓存中的路径重新校验，不能只凭缓存声称阶段完成。

恢复顺序：

1. 阅读`latest_state.json`定位当前run和监督器。
2. 阅读`trajectory_estimator_pack_20260721/docs/IMU与轨迹交付状态及问题清单.md`，以其中未关闭问题为唯一问题口径。
3. 阅读`trajectory_estimator_pack_20260721/docs/轨迹生成方法与HMOG标准数据集测试说明.md`，恢复数据、训练、生成和检测协议。
4. 读取本机正式`supervisor_status.json`、`training_health.json`和manifest，并验证当前文件SHA；缓存只用于导航。
5. 不修改冻结v16源码/配置和冻结estimator-pack源码/配置；所有新问题都继续追加到中文问题文档。

公开仓库：<https://github.com/WMSlalalala/imu_gen>
