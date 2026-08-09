# Results

FAR at the development-selected FRR = 5% threshold; higher is a stronger attack.

## Table 1 - Comparison: each attack against a detector trained on it

| Method | Reference | Channel | Cells | Mean | Median | Cells >= 0.60 |
|---|---|---|---|---|---|---|
| **Ours (released)** | - | trajectory | 30 | **0.777** | 0.779 | **26** |
| **Ours (released)** | - | imu | 30 | **0.835** | 0.840 | **30** |
| **Ours (released)** | - | joint | 30 | **0.711** | 0.734 | **21** |
| Diffusion-TS (traj arm) | Yuan & Qiao, ICLR 2024 | trajectory | 30 | 0.460 | 0.393 | 11 |
| Diffusion-TS (IMU arm) | Yuan & Qiao, ICLR 2024 | imu | 30 | 0.325 | 0.343 | 2 |
| Diffusion-TS (dual arm) | Yuan & Qiao, ICLR 2024 | joint | 30 | 0.266 | 0.205 | 3 |
| CSDI | Tashiro et al., NeurIPS 2021 | imu | 30 | 0.292 | 0.175 | 4 |
| ImagenTime | Naiman et al., NeurIPS 2024 | imu | 30 | 0.683 | 0.778 | 20 |
| TTS-GAN | Li et al., AIME 2022 | imu | 30 | 0.124 | 0.008 | 1 |
| pyclick | Bezier human-cursor library | trajectory | 24 | 0.294 | 0.198 | 6 |
| ghost-cursor | Fitts-law cursor library | trajectory | 24 | 0.335 | 0.232 | 6 |
| *Control* | *genuine windows swapped in as the fake channel* | *IMU* | *30* | *0.773* | *0.844* | *22* |

The italic Control row, here and in Table 3, is not a baseline: no generator is being measured, genuine inertial windows are simply fed through the forged channel. It is **a reference point on the same pipeline, not an upper bound** -- those windows come from a different genuine event and are resampled to the detector's fixed window length, two mismatches the release's own fake channel never carries, so a method can score above this row without being "more real than real" (`EXPERIMENTS_CN.md` 2.4). pyclick and ghost-cursor decline keystroke, hence 24 cells rather than 30.

## Table 2 - Ablation: the release's own components, removed one at a time

Inertial channel only, and keystroke is excluded by construction: its fake IMU is written by an analytic adapter (`diffusion_used: false`), so no ablation of the diffusion generator can reach it.

| Arm | What is removed | Cells | Mean | vs release | Cells >= 0.60 |
|---|---|---|---|---|---|
| **A1** | nothing (the released method) | 24 | **0.835** | - | **24** |
| A2 | no five-shot conditioning (k_refs = 0) | 24 | 0.779 | **+0.056** | 21 |
| A3 | adversarial training off | 24 | 0.802 | **+0.034** | 21 |
| A4 | k_refs = 1 | 24 | 0.798 | **+0.038** | 23 |
| A5 | k_refs = 3 | 24 | 0.832 | **+0.004** | 23 |
| A6 | k_refs = 8 | 24 | 0.830 | **+0.005** | 23 |
| **A1** | nothing (matched baseline for A7-A11) | 12 | **0.789** | - | **12** |
| A3' | adversarial training off (same 12 cells) | 12 | 0.744 | **+0.045** | 9 |
| A7 | gradient merging replaced by a weighted sum | 12 | 0.734 | **+0.054** | 9 |
| A8 | feature critic removed | 12 | 0.682 | **+0.107** | 8 |
| A9 | set critic removed | 12 | 0.716 | **+0.073** | 9 |
| A10 | waveform critic removed | 12 | 0.706 | **+0.083** | 9 |
| A11 | direct feature-matching loss removed | 12 | 0.717 | **+0.072** | 9 |

A7-A11 retrain the generator and cover scroll and swipe only, so their means are over 12 cells rather than 24 and must be read against the 12-cell A1 row (0.789), never against the 24-cell one (0.835). A1 and A3 each appear twice: one arm, recomputed on each of the two cell sets. The `vs release` column is paired cell by cell, so it remains comparable across every arm; the means are not.

### Per detector

A1-A6 are means over the four actions (24 cells); A7-A11 cover scroll and swipe only (12 cells). The two calibers do not mix, so the matched 12-cell A1 baseline is repeated above the A7-A11 block and is the only row those five should be compared with.

| Arm | Actions | HMOG-SVM | HMOG-RF | Paper-SVM | Paper-XGB | BehaveFormer | AuthConformer |
|---|---|---|---|---|---|---|---|
| **A1 (released)** | four, 24 cells | **0.878** | **0.822** | **0.881** | **0.769** | **0.848** | **0.816** |
| A2 | four, 24 cells | 0.864 | 0.680 | 0.856 | 0.665 | 0.830 | 0.780 |
| A3 | four, 24 cells | 0.875 | 0.734 | 0.878 | 0.698 | 0.825 | 0.800 |
| A4 | four, 24 cells | 0.851 | 0.732 | 0.849 | 0.692 | 0.890 | 0.772 |
| A5 | four, 24 cells | 0.883 | 0.806 | 0.881 | 0.751 | 0.837 | 0.834 |
| A6 | four, 24 cells | 0.869 | 0.808 | 0.878 | 0.763 | 0.850 | 0.815 |
| **A1 (released)** | scroll+swipe, 12 cells | **0.859** | **0.772** | **0.852** | **0.725** | **0.775** | **0.749** |
| A7 | scroll+swipe, 12 cells | 0.901 | 0.581 | 0.824 | 0.607 | 0.859 | 0.634 |
| A8 | scroll+swipe, 12 cells | 0.865 | 0.564 | 0.794 | 0.548 | 0.704 | 0.617 |
| A9 | scroll+swipe, 12 cells | 0.867 | 0.623 | 0.807 | 0.571 | 0.802 | 0.627 |
| A10 | scroll+swipe, 12 cells | 0.879 | 0.585 | 0.816 | 0.561 | 0.795 | 0.600 |
| A11 | scroll+swipe, 12 cells | 0.873 | 0.614 | 0.812 | 0.584 | 0.784 | 0.633 |

Against its matched baseline, A7 still rises on HMOG-SVM (0.901 vs 0.859) while the tree-based detectors collapse (HMOG-RF 0.581 vs 0.772, Paper-XGB 0.607 vs 0.725); reporting a single detector would reverse the conclusion. `ablation/README.md` works through why.


## Table 3 - Transfer: every attack against the detectors the release shipped

No retraining and no threshold re-selection. This answers a different question from Table 1 and must not be read as attack strength: the weakest attack in Table 1 scores near the top here, because these detectors were trained on the release's artefacts and a different generator's artefacts simply pass.

| Method | Channel | Cells | Mean | Median |
|---|---|---|---|---|
| **Ours (released)** | imu | 30 | **0.835** | 0.840 |
| | | | | |
| *Control* | *imu* | *30* | *0.942* | *0.950* |
| ImagenTime | imu | 30 | 0.924 | 0.933 |
| Diffusion-TS (IMU arm) | imu | 30 | 0.913 | 0.940 |
| CSDI | imu | 30 | 0.892 | 0.929 |
| TTS-GAN | imu | 30 | 0.877 | 0.901 |
| Diffusion-TS (dual arm) | joint | 30 | 0.769 | 0.794 |
| Diffusion-TS (traj arm) | trajectory | 30 | 0.763 | 0.750 |
| ghost-cursor | trajectory | 24 | 0.704 | 0.692 |
| pyclick | trajectory | 24 | 0.658 | 0.663 |

Self-check: scoring the release with its own detectors reproduces its published cells exactly (30 cells, max |difference| 0.0000). The release's two columns coincide by construction -- for it alone, the detector trained on the attack *is* the shipped detector.
