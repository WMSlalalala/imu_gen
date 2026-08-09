# A7: gradient merging replaced by a plain weighted sum

An ablation of the released method, not a third-party baseline. Everything the release does is kept except the one thing named above.

## What this arm changes

- gradient merging replaced by a plain weighted sum
- sampling protocol: `fewshot_adv`
- generator retrained: yes -- scroll, swipe, from the released run's own `effective_config.json` with the switch above flipped (`a7_weighted_sum/plan.json`, `a7_weighted_sum/config_<action>.yaml`)
- generator source modified: no -- a config switch, not an edit to the trainer or the model

## What stays identical to the release

- the carrier's update timing, its no-contact sentinel, its clock column and the genuine events are the release's and left byte-identical.

## Confounds

- scroll, swipe: drawn from this arm's own training run, not the release's, so the number carries that run's training as well as the change above. Run-to-run variation between two trainings of the same config is not measured anywhere in this project, so it cannot be subtracted out.
- pinch, tap: not retrained for this arm and therefore carrying no number -- a different thing from an action the method declines to model, though the bundle manifest lists both the same way. The arm's conclusion is measured on scroll, swipe only; whether it extrapolates to the rest is untested.
- keystroke: out of reach of this or any generator ablation. Its released fake IMU is written by the analytic adapter (`diffusion_used: false`, generator source `keystroke_imu_pulse.py`) and never passes through the diffusion generator, so `build_ablation_cache_baseline.py` declines it in `ALWAYS_DECLINED`.

## The change took effect

Read back from this arm's own training logs, not from the config that requested the change. The trainer prints one JSON line per logged step and names each live component in it -- `adv_acc_<name>` is critic `<name>`'s discrimination accuracy at that step -- so a component that was really switched off leaves no key behind at all. Counts are the number of logged JSON lines carrying the key.

| Training log | `adv_acc_feature` | `adv_acc_set` | `adv_acc_waveform` | `adv_feature_match_*` |
|---|---|---|---|---|
| `train_scroll.log` | 1036 | 1036 | 1036 | 1036 on each of 6 keys |
| `train_swipe.log` | 738 | 738 | 738 | 738 on each of 6 keys |

The `adv_feature_match_*` group is `corr`, `kurt`, `mean`, `skew`, `std`, `weight` -- the matched statistics and the weight the trainer applies to them.

The plan its driver wrote for these runs asked for exactly that: `max_grad_ratio` `0.5` -> `1000000000.0` (scroll, swipe); `project_conflicts` `true` -> `false` (scroll, swipe).

No key disappears for this arm -- every critic keeps running and only the merge changes -- so the read-back is the merged quantity itself. `adv_grad_merge_scale` is the factor the trainer pins the adversarial gradient to each step: `train_scroll.log`: 1 on all 1036 logged steps; `train_swipe.log`: 1 on all 738 logged steps. A value of 1 on every step is the cap never binding, which is what removing it is supposed to look like.

The sampler is not this arm's variable and its read-back says only that much: a drawn sample carries `ref_count = 5` with 5 reference indices, protocol `fewshot_adv`.

## Result

| Modality | Cells | Mean FAR | Cells >= 0.60 |
|---|---|---|---|
| IMU | 12 | 0.734 | 9 |

FAR at the development-selected FRR = 5% threshold, against a detector trained on this arm. Inertial channel only: this arm changes no touch coordinate.
