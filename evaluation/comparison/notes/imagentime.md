# ImagenTime as a baseline on HMOG

**Paper.** I. Naiman, N. Berman, I. Pemper, I. Arbiv, G. Fadlon and O. Azencot, "Utilizing Image Transforms and Diffusion Models for Generative Modeling of Short and Long Time Series", NeurIPS 2024

**Code.** https://github.com/azencot-group/ImagenTime

## What ran unchanged

- the delay-embedding transform, the UNet, the EDM loss, the EMA and the reverse process, unmodified

## Every deviation

- `data/long_range.py` and `data/data_provider/` were written: the repository ships `data/` as an empty package because the corpora are distributed separately, so this is its intended extension point rather than a modification.
- A delay/embedding pair per action, found by exhaustive search over the pairs that fill the square exactly so the transform inverts. tap admits no exact fill (T=16) and uses the smallest square with two zero columns, which still inverts exactly.
- Per-channel min-max scaling into [0,1], which every corpus they ship receives and which EDM's hard-coded `sigma_data = 0.5` assumes.
- A training driver and a sampler. Their `run_unconditional.py` scores the model by refitting S4 classifiers ten times per evaluation and checkpoints only inside that block on improvement; the repository has no sampling script at all.
- The authors' narrower UNet from their own `mujoco.yaml` (14.4M parameters against 151.7M) for every action. This is a configuration choice of theirs, not a model change.
- Early stopping, against the authors' fixed 1000 epochs. Every 25 epochs a draw is compared with genuine data on lag-1 autocorrelation, per-channel dispersion and a window-summary energy distance; training stops when that gap has not improved for 3 consecutive checks. The samples are drawn from the model at the stopping epoch, not from the best-scoring check -- both are given below so the difference is visible rather than assumed away. Per action: tap stopped at 574 (gap 0.370; the best check was epoch 499 at 0.361); pinch stopped at 399 (gap 0.399; the best check was epoch 324 at 0.373); swipe stopped at 374 (gap 0.421; the best check was epoch 299 at 0.352); scroll stopped at 449 (gap 0.427; the best check was epoch 374 at 0.334); keystroke stopped at 424 (gap 0.778; the best check was epoch 349 at 0.568). The full histories are in `summary_<action>.json`, so a reader can see the gap had flattened rather than take the stop on trust.

## Sanity check

`img_to_ts` unpads against a shape cached by the first `ts_to_img` call, so one real batch is always pushed through before any draw -- otherwise sampling silently returns the wrong length.

## How it was scored

The generator supplies the inertial channel. Everything else -- the carrier's update timing, its no-contact sentinel, its clock column, and the genuine events -- is the release's and is left byte-identical, which `verify_harness.py` checks.

Reported on: IMU. A modality this method does not supply is not reported, because those cells would be running the release's own data.

## Result

| Modality | Cells | Mean FAR | Cells >= 0.60 |
|---|---|---|---|
| IMU | 30 | 0.683 | 20 |

FAR at the development-selected FRR = 5% threshold, against a detector trained on this attack.
