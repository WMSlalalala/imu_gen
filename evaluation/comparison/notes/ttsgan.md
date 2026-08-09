# TTS-GAN as a baseline on HMOG

**Paper.** X. Li, V. Metsis, H. Wang and A. H. H. Ngu, "TTS-GAN: A Transformer-based Time-Series Generative Adversarial Network", AIME 2022

**Code.** https://github.com/imics-lab/tts-gan

## What ran unchanged

- the transformer generator and discriminator, the training loop, the EMA and the learning-rate schedule, unmodified
- the authors' per-window z-score normalisation

## Every deviation

- Mid-training checkpointing was added. The authors' loop keeps nothing between epochs, so an interrupted run loses everything.
- The budget was raised after a first run proved undertrained: its lag-1 autocorrelation was 0.387-0.850 against genuine data's 0.945-0.996. The evidence for that decision is kept in `ttsgan_budget_evidence.json` so the reported result cannot be read as a statement about the first budget.

## How it was scored

The generator supplies the inertial channel. Everything else -- the carrier's update timing, its no-contact sentinel, its clock column, and the genuine events -- is the release's and is left byte-identical, which `verify_harness.py` checks.

Reported on: IMU. A modality this method does not supply is not reported, because those cells would be running the release's own data.

## Result

| Modality | Cells | Mean FAR | Cells >= 0.60 |
|---|---|---|---|
| IMU | 30 | 0.124 | 1 |

FAR at the development-selected FRR = 5% threshold, against a detector trained on this attack.
