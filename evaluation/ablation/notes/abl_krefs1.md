# A4: k_refs = 1 at sampling time

An ablation of the released method, not a third-party baseline. Everything the release does is kept except the one thing named above.

## What this arm changes

- k_refs = 1 at sampling time
- sampling protocol: `fewshot_adv`
- `k_refs` overridden to 1
- generator retrained: no -- this arm draws from checkpoints that already existed, so only the sampler was told anything different
- generator source modified: no -- a config switch, not an edit to the trainer or the model

## What stays identical to the release

- pinch, scroll, swipe, tap: the same checkpoint the release samples from, so the arm differs from it only in what the sampler was told.
- the carrier's update timing, its no-contact sentinel, its clock column and the genuine events are the release's and left byte-identical.

## The change took effect

Read back from the cache rather than assumed: a drawn sample carries `ref_count = 1` with 1 reference indices, protocol `fewshot_adv`.

## What this arm does not measure

checkpoints were trained at their protocol's own reference count; sampling at a different count measures how the reference encoder generalises, not a model trained for that count.

## Result

| Modality | Cells | Mean FAR | Cells >= 0.60 |
|---|---|---|---|
| IMU | 24 | 0.798 | 23 |

FAR at the development-selected FRR = 5% threshold, against a detector trained on this arm. Inertial channel only: this arm changes no touch coordinate.
