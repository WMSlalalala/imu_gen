# CSDI as a baseline on HMOG

**Paper.** Y. Tashiro, J. Song, Y. Song and S. Ermon, "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", NeurIPS 2021

**Code.** https://github.com/ermongroup/CSDI

## What ran unchanged

- `CSDI_Physio` and its denoiser, unmodified
- every hyper-parameter from the authors' own `config/base.yaml`
- their loss, optimiser, and MultiStepLR schedule

## Every deviation

- A Dataset returning the four keys `process_data` reads, because the repository ships loaders for three medical corpora only.
- Per-channel z-scoring, which is mandatory rather than optional: the model has no normalisation layer and all three official datasets standardise before the DataLoader, while accelerometer and gyroscope differ by roughly 7x in scale.
- An unconditional draw. CSDI is an imputer with no generation entry point; passing an all-zero conditioning mask turns the authors' own `impute` into a plain DDPM over the whole window. No model change.
- Checkpointing every epoch. The authors' `train()` writes one file after the final epoch and never uses the best validation loss it computes, so an interrupted run keeps nothing.
- The five-shot arm was dropped after measurement: it shares the trained model with the unconditional one and differs only in what the sampler is told, so it would add a second CSDI row at the cost of ten more detector cells.

## Sanity check

`linear_attention_transformer` was installed rather than stubbed, so `diff_models.py` is byte-identical to the authors' file; `target_strategy` was left at `random`, because their historical-masking branch makes the training loss identically zero on fully-observed windows.

## How it was scored

The generator supplies the inertial channel. Everything else -- the carrier's update timing, its no-contact sentinel, its clock column, and the genuine events -- is the release's and is left byte-identical, which `verify_harness.py` checks.

Reported on: IMU. A modality this method does not supply is not reported, because those cells would be running the release's own data.

## Result

| Modality | Cells | Mean FAR | Cells >= 0.60 |
|---|---|---|---|
| IMU | 30 | 0.292 | 4 |

FAR at the development-selected FRR = 5% threshold, against a detector trained on this attack.
