# 轨迹 PAD 补充评测包

该目录解决正式冻结主协议中两个已经公开记录的方法学问题，不修改v16正式源码、配置或正在运行的产物：

1. real与fake两侧统一使用同一份70/10/20 user-disjoint split；
2. 额外构建排除全部fixed-five real references的敏感性数据集。

主协议结果仍单独保留。补充结果不得覆盖或改名成主协议结果，而应并列报告。

正式100k生成和primary detector bundle完成后构建两个变体：

```bash
python scripts/build_supplement_bundles.py \
  --primary-bundle-dir /path/to/formal/detector_bundle \
  --output-dir /path/to/supplement/fully_user_disjoint \
  --split-json /home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json \
  --reference-registry-map /path/to/formal/manifests/reference_registry_map.json

python scripts/build_supplement_bundles.py \
  --primary-bundle-dir /path/to/formal/detector_bundle \
  --output-dir /path/to/supplement/fully_user_disjoint_reference_excluded \
  --split-json /home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json \
  --reference-registry-map /path/to/formal/manifests/reference_registry_map.json \
  --exclude-references
```

两个变体可分别占用一张空闲GPU运行25个检测器：

```bash
python scripts/run_supplement_benchmark.py \
  --bundle-dir /path/to/supplement/fully_user_disjoint \
  --output-dir /path/to/results/fully_user_disjoint \
  --device cuda:0 --epochs 40 --bootstrap-replicates 500 \
  --confirm-formal-supplement

python scripts/run_supplement_benchmark.py \
  --bundle-dir /path/to/supplement/fully_user_disjoint_reference_excluded \
  --output-dir /path/to/results/fully_user_disjoint_reference_excluded \
  --device cuda:1 --epochs 40 --bootstrap-replicates 500 \
  --confirm-formal-supplement
```

测试：

```bash
/home/mwang49/miniconda3/envs/hml/bin/python -m unittest discover -s tests -v
```

当前阶段仅完成实现与synthetic回归。没有100k primary bundle之前，不能把本目录写成正式指标完成。
