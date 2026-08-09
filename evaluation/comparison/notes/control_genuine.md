# Control: genuine windows as the fake channel as a baseline on HMOG

**Paper.** not a baseline

## What ran unchanged

- genuine inertial windows, unaltered

## Every deviation

- None beyond the harness described below.

## Sanity check

This is the ceiling the pipeline itself allows. If a detector separated these from the release's genuine events, the harness would be manufacturing the gap rather than measuring it.

## How it was scored

The generator supplies the inertial channel. Everything else -- the carrier's update timing, its no-contact sentinel, its clock column, and the genuine events -- is the release's and is left byte-identical, which `verify_harness.py` checks.

Reported on: IMU. A modality this method does not supply is not reported, because those cells would be running the release's own data.

## Result

| Modality | Cells | Mean FAR | Cells >= 0.60 |
|---|---|---|---|
| IMU | 30 | 0.773 | 22 |

FAR at the development-selected FRR = 5% threshold, against a detector trained on this attack.
