# Diffusion-TS as a baseline on HMOG

**Paper.** X. Yuan and Y. Qiao, "Diffusion-TS: Interpretable Diffusion for General Time Series Generation", ICLR 2024

**Code.** https://github.com/Y-debug-sys/Diffusion-TS

## What ran unchanged

- the model, the interpretable trend/seasonality decomposition, the diffusion schedule and the training loop, unmodified
- the authors' own hyper-parameters for each sequence length

## Every deviation

- One dataset class was written to feed HMOG windows, because the repository ships loaders only for its own corpora.

## Sanity check

The authors' Sines benchmark was reproduced first: discriminative score 0.0108 +/- 0.0061 against the paper's 0.006 +/- 0.007, so the training path is sound before it was pointed at HMOG.

## How it was scored

The generator supplies both channels. Everything else -- the carrier's update timing, its no-contact sentinel, its clock column, and the genuine events -- is the release's and is left byte-identical, which `verify_harness.py` checks.

Reported on: trajectory, IMU, joint. A modality this method does not supply is not reported, because those cells would be running the release's own data.

## Result

| Modality | Cells | Mean FAR | Cells >= 0.60 |
|---|---|---|---|
| joint | 30 | 0.266 | 3 |

FAR at the development-selected FRR = 5% threshold, against a detector trained on this attack.

*60 cell(s) not yet computed.*
