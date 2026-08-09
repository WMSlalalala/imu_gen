# Results at the test-set equal-error rate

A companion to `RESULTS.md`, which reports FAR at a FRR=5% threshold selected on the development split. This file reports EER located on the test split, because much of the literature does. It is the weaker protocol -- the cut is chosen knowing the test scores -- and the release's own artefacts call the same quantity `descriptive_test_eer` for that reason.

## Table E1 - Comparison at the test-set EER

Higher is a stronger attack, as in the FAR table -- but the scale is different and the two must not be quoted against each other.

| Method | Channel | Cells | Mean EER | Median EER |
|---|---|---|---|---|
| **Ours (released)** | trajectory | 30 | **0.367** | 0.372 |
| **Ours (released)** | IMU | 30 | **0.397** | 0.405 |
| **Ours (released)** | joint | 30 | **0.319** | 0.324 |
| Diffusion-TS (traj arm) | trajectory | 30 | 0.208 | 0.176 |
| Diffusion-TS (IMU arm) | IMU | 30 | 0.124 | 0.131 |
| Diffusion-TS (dual arm) | joint | 30 | 0.104 | 0.097 |
| CSDI | IMU | 30 | 0.105 | 0.078 |
| ImagenTime | IMU | 30 | 0.301 | 0.325 |
| TTS-GAN | IMU | 30 | 0.063 | 0.019 |
| pyclick | trajectory | 24 | 0.150 | 0.103 |
| ghost-cursor | trajectory | 24 | 0.156 | 0.108 |
| *Control* | *IMU* | *30* | *0.447* | *0.440* |

## Table E2 - Ablation at the test-set EER

Inertial channel only; keystroke is excluded by construction. A7-A11 retrain the generator and cover scroll and swipe only, so their means are over 12 cells against A1's 24 -- the `vs release` column is paired cell by cell and stays comparable, the means do not.

| Arm | What is removed | Cells | Mean EER | vs release |
|---|---|---|---|---|
| **A1** | nothing (the released method) | 24 | **0.393** | - |
| A2 | no five-shot conditioning (k_refs = 0) | 24 | 0.358 | **+0.035** |
| A3 | adversarial training off | 24 | 0.370 | **+0.023** |
| A4 | k_refs = 1 | 24 | 0.371 | **+0.022** |
| A5 | k_refs = 3 | 24 | 0.396 | **-0.003** |
| A6 | k_refs = 8 | 24 | 0.383 | **+0.010** |
| A7 | gradient merging replaced by a weighted sum | 12 | 0.315 | **+0.047** |
| A8 | feature critic removed | 12 | 0.288 | **+0.074** |
| A9 | set critic removed | 12 | 0.306 | **+0.056** |
| A10 | waveform critic removed | 12 | 0.301 | **+0.061** |
| A11 | direct feature-matching loss removed | 12 | 0.307 | **+0.056** |

## Table E3 - The two operating points side by side

Same cells, same scores, two ways of choosing the cut. The ordering is preserved; the spread is not.

| Method | Channel | FAR @ FRR=5% | Test EER |
|---|---|---|---|
| **Ours (released)** | trajectory | **0.777** | **0.367** |
| **Ours (released)** | IMU | **0.835** | **0.397** |
| **Ours (released)** | joint | **0.711** | **0.319** |
| Diffusion-TS (traj arm) | trajectory | 0.460 | 0.208 |
| Diffusion-TS (IMU arm) | IMU | 0.325 | 0.124 |
| Diffusion-TS (dual arm) | joint | 0.266 | 0.104 |
| CSDI | IMU | 0.292 | 0.105 |
| ImagenTime | IMU | 0.683 | 0.301 |
| TTS-GAN | IMU | 0.124 | 0.063 |
| pyclick | trajectory | 0.294 | 0.150 |
| ghost-cursor | trajectory | 0.335 | 0.156 |
| *Control* | *IMU* | *0.773* | *0.447* |

### Per detector, released method, inertial channel

| HMOG-SVM | HMOG-RF | Paper-SVM | Paper-XGB | BehaveFormer | AuthConformer |
|---|---|---|---|---|---|
| 0.399 | 0.369 | 0.428 | 0.353 | 0.424 | 0.407 |
