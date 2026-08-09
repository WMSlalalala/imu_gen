# A3: adversarial training removed

An ablation of the released method, not a third-party baseline. Everything the release does is kept except the one thing named above.

## What this arm changes

- adversarial training removed
- sampling protocol: `fewshot`
- generator retrained: no -- this arm draws from checkpoints that already existed, so only the sampler was told anything different
- generator source modified: no -- a config switch, not an edit to the trainer or the model

## What stays identical to the release

- the carrier's update timing, its no-contact sentinel, its clock column and the genuine events are the release's and left byte-identical.

## Confounds

- pinch, scroll, swipe, tap: drawn from this protocol's own training run, not the release's, so the number carries that run's training as well as the change above. Run-to-run variation between two trainings of the same config is not measured anywhere in this project, so it cannot be subtracted out.

## The change took effect

Read back from the cache rather than assumed: a drawn sample carries `ref_count = 5` with 5 reference indices, protocol `fewshot`.

**Actions without a checkpoint.** keystroke. These carry no number rather than a substituted one.

## Result

| Modality | Cells | Mean FAR | Cells >= 0.60 |
|---|---|---|---|
| IMU | 24 | 0.802 | 21 |

FAR at the development-selected FRR = 5% threshold, against a detector trained on this arm. Inertial channel only: this arm changes no touch coordinate.
