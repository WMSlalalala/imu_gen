# 严格实验协议

## 1. 用户与数据边界

- Generator/fake user split 固定读取
  `/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json`：
  70 train、10 validation、20 test，三者互斥。
- 轨迹条件分布、标准化、reference encoder、diffusion、feature detector 和
  deep detector 都只能在 train 部分拟合。
- 每个目标 user/action 固定选择 5 条 reference trajectories。refs 不作为
  detector fake test 样本；fake 必须保存五个 ref event IDs 以便审计同用户与唯一性。
- Validation 选择 threshold/checkpoint；test 不参与模型、阈值或超参数选择。

## 2. Real detector pools

轨迹 detector 的 real 数据覆盖 100 users。为避免 keystroke 同一 typing event 的
不同 chunk 泄漏，real enroll/validation/test 按完整 trajectory event ID 分组后做
确定性 60/20/20 划分，禁止逐 frame、逐 key 或逐 chunk 随机切分。同一 event 的
全部原始帧只能属于一个 pool。

## 3. Fake 数量

正式目标与 IMU 实验一致：每 user/action 200 条，即每个 action 的 fake train/
validation/test 为 14,000/2,000/4,000，总计 100,000 条轨迹。每条 fake 的
metadata condition 来自 train-user real distribution 或调用方显式输入；不能从
目标 validation/test user 的非 ref 轨迹抽条件。

## 4. 输入和合法性

- 原始 touch timestamp 使用毫秒并保持严格单调；不伪称原始 100 Hz。
- 单指轨迹必须具有合法 DOWN → MOVE* → UP 结构。
- Pinch 必须保留两个 pointer，并可序列化为 Android Type-B slots；不能降成
  一条 focus-center 线。
- Keystroke 是多个独立 key contacts，加 keycode/letter、hold、flight；键间无接触
  区间不插值成连续手指移动。
- 生成后硬约束只负责物理合法性与显式条件，不能用 test detector 分数筛选 fake。

## 5. 检测与指标

每个 action 独立报告：

1. clean-room AHB-style feature SVM/XGBoost；
2. raw-sequence deep PAD；

本目录当前正式完成范围是 **trajectory-only**：三类 Feature PAD
（linear/RBF SVM、XGBoost）与两类 raw-sequence Deep PAD（TCN、Transformer）。
IMU-only 和 IMU+trajectory multimodal 只属于 future/optional extension，不计入本目录
的 25-pair 正式结果，也不得在报告中写成已经完成。

分数方向统一为 fake-high。Validation 单独选择 EER threshold 和指定 FRR budget
threshold；test 报告固定阈值下 FA/FRR、ROC AUC、完整 FA--FRR 曲线及 user-level
bootstrap 95% CI。另报告 endpoint error、duration error、合法 event rate、exact
replay rate、train-nearest DTW/Fréchet 和任务成功率。

## 6. 禁止事项

- 不允许在完整数据上先 fit scaler；
- 不允许随机 gesture split 代替 user/event group split；
- 不允许用 test score 选 checkpoint、threshold、generator 或候选；
- 不允许用 template replay 冒充 neural generation；
- 不允许为达到目标 FA 删除困难 detector、减少 test users、减少 fake 数量或改变
  指标方向。
