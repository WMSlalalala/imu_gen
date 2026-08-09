# A2: five-shot conditioning removed (k_refs = 0)

An ablation of the released method, not a third-party baseline. Everything the release does is kept except the one thing named above.

## What this arm changes

- five-shot conditioning removed (k_refs = 0)
- sampling protocol: `noshot_adv`
- generator retrained: no -- this arm draws from checkpoints that already existed, so only the sampler was told anything different
- generator source modified: no -- a config switch, not an edit to the trainer or the model

## What stays identical to the release

- the carrier's update timing, its no-contact sentinel, its clock column and the genuine events are the release's and left byte-identical.

## Confounds

- pinch, scroll, swipe, tap: drawn from this protocol's own training run, not the release's, so the number carries that run's training as well as the change above. Run-to-run variation between two trainings of the same config is not measured anywhere in this project, so it cannot be subtracted out.

## The change took effect

Read back from the cache rather than assumed: a drawn sample carries `ref_count = 0`, protocol `noshot_adv`.

**Declined by construction.** keystroke. This arm did draw keystroke samples -- `noshot_adv` has a keystroke checkpoint of its own -- but the released keystroke fake IMU is written by the analytic adapter (`diffusion_used: false`, generator source `keystroke_imu_pulse.py`) and never passes through the diffusion generator, so no ablation of that generator can move a keystroke cell. `build_ablation_cache_baseline.py` declines it in `ALWAYS_DECLINED`; it carries no number rather than a substituted one, so the table below covers the 4 actions listed above.

## Result

| Modality | Cells | Mean FAR | Cells >= 0.60 |
|---|---|---|---|
| IMU | 24 | 0.779 | 21 |

FAR at the development-selected FRR = 5% threshold, against a detector trained on this arm. Inertial channel only: this arm changes no touch coordinate.
