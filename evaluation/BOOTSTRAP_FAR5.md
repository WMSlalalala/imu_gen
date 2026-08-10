# User-clustered bootstrap at the FRR=5% operating point

Resampling unit: the 20 test users, drawn with replacement, all events of each kept. 2000 replicates, seed 42. One user multiset per replicate, shared across all cells (see the module docstring). Thresholds are the frozen `frr5` cuts; nothing is re-selected.

## Aggregate

| Quantity | Point | 95% CI |
|---|---|---|
| FAR, all 90 cells | 0.775 | [0.763, 0.787] |
| FRR, all 90 cells | 0.052 | [0.046, 0.059] |

## By modality

| Modality | Cells | Point FAR | 95% CI | CI width |
|---|---|---|---|---|
| trajectory_xytime | 30 | 0.777 | [0.766, 0.789] | 0.022 |
| imu_only | 30 | 0.835 | [0.817, 0.852] | 0.035 |
| imu_trajectory_xytime | 30 | 0.711 | [0.696, 0.726] | 0.030 |

## Per cell

Per-cell intervals are much wider than the aggregate -- median width 0.085 against 0.013 for the 90-cell FRR and the aggregate FAR band above -- because averaging over 90 cells cancels a good deal of the per-cell noise while the user-level variation, being shared, does not cancel.

| Action | Modality | Detector | Point FAR | 95% CI |
|---|---|---|---|---|
| keystroke | imu_only | authconformer | 0.914 | [0.855, 0.960] |
| keystroke | imu_only | behaveformer_stdat | 0.950 | [0.933, 0.964] |
| keystroke | imu_only | hmog_style_rf | 0.819 | [0.735, 0.889] |
| keystroke | imu_only | hmog_style_svm | 0.712 | [0.613, 0.809] |
| keystroke | imu_only | paper_svm | 0.812 | [0.741, 0.882] |
| keystroke | imu_only | paper_xgboost | 0.792 | [0.697, 0.874] |
| keystroke | imu_trajectory_xytime | authconformer | 0.556 | [0.488, 0.629] |
| keystroke | imu_trajectory_xytime | behaveformer_stdat | 0.961 | [0.952, 0.970] |
| keystroke | imu_trajectory_xytime | hmog_style_rf | 0.822 | [0.745, 0.886] |
| keystroke | imu_trajectory_xytime | hmog_style_svm | 0.739 | [0.639, 0.832] |
| keystroke | imu_trajectory_xytime | paper_svm | 0.796 | [0.731, 0.866] |
| keystroke | imu_trajectory_xytime | paper_xgboost | 0.663 | [0.566, 0.750] |
| keystroke | trajectory_xytime | authconformer | 0.883 | [0.858, 0.906] |
| keystroke | trajectory_xytime | behaveformer_stdat | 0.946 | [0.935, 0.957] |
| keystroke | trajectory_xytime | hmog_style_rf | 0.949 | [0.933, 0.962] |
| keystroke | trajectory_xytime | hmog_style_svm | 0.944 | [0.902, 0.975] |
| keystroke | trajectory_xytime | paper_svm | 0.837 | [0.777, 0.889] |
| keystroke | trajectory_xytime | paper_xgboost | 0.742 | [0.680, 0.801] |
| pinch | imu_only | authconformer | 0.943 | [0.924, 0.958] |
| pinch | imu_only | behaveformer_stdat | 0.936 | [0.922, 0.949] |
| pinch | imu_only | hmog_style_rf | 0.821 | [0.794, 0.846] |
| pinch | imu_only | hmog_style_svm | 0.909 | [0.879, 0.934] |
| pinch | imu_only | paper_svm | 0.918 | [0.899, 0.938] |
| pinch | imu_only | paper_xgboost | 0.772 | [0.748, 0.793] |
| pinch | imu_trajectory_xytime | authconformer | 0.897 | [0.879, 0.916] |
| pinch | imu_trajectory_xytime | behaveformer_stdat | 0.861 | [0.846, 0.876] |
| pinch | imu_trajectory_xytime | hmog_style_rf | 0.542 | [0.480, 0.602] |
| pinch | imu_trajectory_xytime | hmog_style_svm | 0.696 | [0.651, 0.741] |
| pinch | imu_trajectory_xytime | paper_svm | 0.806 | [0.780, 0.834] |
| pinch | imu_trajectory_xytime | paper_xgboost | 0.523 | [0.459, 0.585] |
| pinch | trajectory_xytime | authconformer | 0.935 | [0.929, 0.941] |
| pinch | trajectory_xytime | behaveformer_stdat | 0.841 | [0.826, 0.854] |
| pinch | trajectory_xytime | hmog_style_rf | 0.544 | [0.480, 0.608] |
| pinch | trajectory_xytime | hmog_style_svm | 0.705 | [0.647, 0.762] |
| pinch | trajectory_xytime | paper_svm | 0.790 | [0.753, 0.824] |
| pinch | trajectory_xytime | paper_xgboost | 0.559 | [0.493, 0.620] |
| scroll | imu_only | authconformer | 0.628 | [0.581, 0.675] |
| scroll | imu_only | behaveformer_stdat | 0.707 | [0.654, 0.763] |
| scroll | imu_only | hmog_style_rf | 0.707 | [0.632, 0.776] |
| scroll | imu_only | hmog_style_svm | 0.828 | [0.761, 0.891] |
| scroll | imu_only | paper_svm | 0.806 | [0.722, 0.879] |
| scroll | imu_only | paper_xgboost | 0.680 | [0.607, 0.750] |
| scroll | imu_trajectory_xytime | authconformer | 0.522 | [0.481, 0.564] |
| scroll | imu_trajectory_xytime | behaveformer_stdat | 0.728 | [0.671, 0.787] |
| scroll | imu_trajectory_xytime | hmog_style_rf | 0.484 | [0.412, 0.556] |
| scroll | imu_trajectory_xytime | hmog_style_svm | 0.709 | [0.644, 0.766] |
| scroll | imu_trajectory_xytime | paper_svm | 0.715 | [0.646, 0.775] |
| scroll | imu_trajectory_xytime | paper_xgboost | 0.482 | [0.430, 0.534] |
| scroll | trajectory_xytime | authconformer | 0.573 | [0.528, 0.617] |
| scroll | trajectory_xytime | behaveformer_stdat | 0.754 | [0.708, 0.801] |
| scroll | trajectory_xytime | hmog_style_rf | 0.615 | [0.547, 0.690] |
| scroll | trajectory_xytime | hmog_style_svm | 0.759 | [0.724, 0.800] |
| scroll | trajectory_xytime | paper_svm | 0.765 | [0.739, 0.792] |
| scroll | trajectory_xytime | paper_xgboost | 0.642 | [0.600, 0.681] |
| swipe | imu_only | authconformer | 0.871 | [0.851, 0.888] |
| swipe | imu_only | behaveformer_stdat | 0.844 | [0.829, 0.858] |
| swipe | imu_only | hmog_style_rf | 0.837 | [0.810, 0.861] |
| swipe | imu_only | hmog_style_svm | 0.890 | [0.868, 0.912] |
| swipe | imu_only | paper_svm | 0.898 | [0.876, 0.918] |
| swipe | imu_only | paper_xgboost | 0.770 | [0.753, 0.786] |
| swipe | imu_trajectory_xytime | authconformer | 0.598 | [0.561, 0.636] |
| swipe | imu_trajectory_xytime | behaveformer_stdat | 0.676 | [0.637, 0.715] |
| swipe | imu_trajectory_xytime | hmog_style_rf | 0.557 | [0.504, 0.614] |
| swipe | imu_trajectory_xytime | hmog_style_svm | 0.747 | [0.694, 0.793] |
| swipe | imu_trajectory_xytime | paper_svm | 0.741 | [0.689, 0.790] |
| swipe | imu_trajectory_xytime | paper_xgboost | 0.598 | [0.548, 0.648] |
| swipe | trajectory_xytime | authconformer | 0.502 | [0.442, 0.561] |
| swipe | trajectory_xytime | behaveformer_stdat | 0.769 | [0.732, 0.803] |
| swipe | trajectory_xytime | hmog_style_rf | 0.627 | [0.575, 0.680] |
| swipe | trajectory_xytime | hmog_style_svm | 0.742 | [0.692, 0.788] |
| swipe | trajectory_xytime | paper_svm | 0.790 | [0.739, 0.835] |
| swipe | trajectory_xytime | paper_xgboost | 0.707 | [0.656, 0.756] |
| tap | imu_only | authconformer | 0.821 | [0.800, 0.842] |
| tap | imu_only | behaveformer_stdat | 0.904 | [0.888, 0.919] |
| tap | imu_only | hmog_style_rf | 0.922 | [0.909, 0.934] |
| tap | imu_only | hmog_style_svm | 0.884 | [0.867, 0.903] |
| tap | imu_only | paper_svm | 0.902 | [0.888, 0.916] |
| tap | imu_only | paper_xgboost | 0.854 | [0.838, 0.872] |
| tap | imu_trajectory_xytime | authconformer | 0.774 | [0.749, 0.799] |
| tap | imu_trajectory_xytime | behaveformer_stdat | 0.744 | [0.708, 0.773] |
| tap | imu_trajectory_xytime | hmog_style_rf | 0.835 | [0.809, 0.861] |
| tap | imu_trajectory_xytime | hmog_style_svm | 0.886 | [0.853, 0.911] |
| tap | imu_trajectory_xytime | paper_svm | 0.883 | [0.834, 0.922] |
| tap | imu_trajectory_xytime | paper_xgboost | 0.797 | [0.774, 0.820] |
| tap | trajectory_xytime | authconformer | 0.903 | [0.842, 0.952] |
| tap | trajectory_xytime | behaveformer_stdat | 0.916 | [0.874, 0.951] |
| tap | trajectory_xytime | hmog_style_rf | 0.866 | [0.844, 0.891] |
| tap | trajectory_xytime | hmog_style_svm | 0.907 | [0.844, 0.952] |
| tap | trajectory_xytime | paper_svm | 0.896 | [0.836, 0.942] |
| tap | trajectory_xytime | paper_xgboost | 0.915 | [0.896, 0.931] |
