# Paired user-clustered bootstrap: the release against each method

2000 replicates, seed 42, resampled by test user. Each replicate scores both sides on the same drawn people, so the interval is on the paired difference. Only the cells the two share are used, so the comparison is like for like.

| Method | Shared cells | Ours | Theirs | Difference | 95% CI | Excludes 0 |
|---|---|---|---|---|---|---|
| imagentime | 30 | 0.835 | 0.683 | +0.152 | [+0.136, +0.168] | yes |
| diffts_imu | 30 | 0.835 | 0.325 | +0.510 | [+0.495, +0.526] | yes |
| csdi_unconditional | 30 | 0.835 | 0.292 | +0.543 | [+0.527, +0.559] | yes |
| ttsgan | 30 | 0.835 | 0.124 | +0.711 | [+0.694, +0.728] | yes |
| control_genuine | 30 | 0.835 | 0.773 | +0.062 | [+0.046, +0.077] | yes |
| abl_noshot_adv | 24 | 0.835 | 0.779 | +0.056 | [+0.039, +0.072] | yes |
| abl_fewshot_nonadv | 24 | 0.835 | 0.802 | +0.034 | [+0.023, +0.045] | yes |
| abl_krefs3 | 24 | 0.835 | 0.832 | +0.004 | [-0.003, +0.011] | **no** |
| abl_a7_weighted_sum | 12 | 0.789 | 0.734 | +0.054 | [+0.034, +0.076] | yes |
| abl_a8_no_feature | 12 | 0.789 | 0.682 | +0.107 | [+0.087, +0.127] | yes |
