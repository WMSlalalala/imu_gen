#!/usr/bin/env python3
"""Swap an ablation's IMU cache into a copy of the release, slot for slot.

WHY THIS IS A SUBSTITUTION AND NOT A REBUILD
---------------------------------------------
The program that composed the release -- the one that bound cache samples to
fake events and wrote the shards -- is not on disk any more; only its outputs
survive.  It cannot be re-run for an ablation cache.

It does not need to be.  The binding it made is recorded inside the released
shards themselves: every fake event carries the `sample_idx` of the cache sample
it was given, and its `user_id`, `action` and the shard's `split` name the rest
of the path.  So the release can be reproduced from its own cache, and an
ablation dataset is the same reconstruction reading a different cache.

That claim is not taken on faith.  `verify_reconstruction.py` replays the rule
against the *release* cache over every shard and asserts the result is
bit-identical to the published IMU; run it before trusting a build.

THE ROUTING RULE (verified bit-exact on the released scroll bundle)
--------------------------------------------------------------------
    user   = int(str(user_id)[-3:])                     "hmog_u037" -> 37
    split  = {"train": "train",
              "development": "val",                     the cache calls it val
              "test": "test"}[shard split]
    path   = <cache>/user_%03d/<action>/<split>/sample_%04d.npz

`sample_idx` is present on genuine events too -- it is their own ordinal -- so
only fake events are ever looked up.

THE LENGTH RULE, WHICH DIFFERS BY ACTION
-----------------------------------------
The event's row count is frozen by the release and is never changed: the
carrier, the clock and the detector window all depend on it.

  scroll            a contiguous cut of the *padded* window around the mask
                    span, via the pipeline's own `carrier_window_imu`.  Not
                    `active_imu`, and no resampling.
  tap/pinch/swipe   `linear_resample(active_imu, n)`, which is the identity
                    whenever the sample is already n rows long.

KEYSTROKE IS DECLINED, AND THAT IS NOT A CHOICE
------------------------------------------------
Released keystroke fakes record `diffusion_used: false`, `model_used: false`
and a generator source of `security_exp/keystroke_imu_pulse.py` -- an analytic
adapter driven by the victim's own genuine keystroke events.  Their IMU never
passes through the diffusion generator, so no ablation of that generator can
move a keystroke cell.  Declining it keeps the number out of the table
entirely, which is right: a zero delta would read as "the ablated component
does not help keystroke" when the truth is "the ablation does not reach it".

A PARTIALLY DRAWN ACTION IS REFUSED
------------------------------------
If any slot an action needs is missing from the cache, the whole action is
declined rather than part-filled.  Mixing release IMU and ablation IMU inside
one detector's training set produces a number that describes neither.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, "/mnt/share/mwang49/data7/code/direct100k")

from hmog_baseline_common import (  # noqa: E402
    ACTIONS,
    finalise_dataset,
    iter_shards,
    linear_resample,
    load_shard,
    merge_counts,
    rewrite_shard_arrays,
    save_shard,
)

SPLIT_DIR = {"train": "train", "development": "val", "test": "test"}
# Actions the release re-cuts from the padded window rather than resampling the
# active span.  Measured against the published bundles, not assumed.
FROM_PADDED_WINDOW = frozenset({"scroll"})
# The generator never touches this action's IMU; see the module docstring.
ALWAYS_DECLINED = frozenset({"keystroke"})

_STATE: dict = {}


def cache_path(root: Path, event: dict) -> Path:
    user = int(str(event["user_id"])[-3:])
    split = SPLIT_DIR[str(event["split"])]
    return (
        root
        / f"user_{user:03d}"
        / str(event["action"])
        / split
        / f"sample_{int(event['sample_idx']):04d}.npz"
    )


def generate_imu(event: dict):
    action = str(event["action"])
    if action in ALWAYS_DECLINED or action in _STATE["decline"]:
        return None
    path = cache_path(_STATE["cache_root"], event)
    if not path.is_file():
        # Counted as declined and never invented.  `--require-complete` turns a
        # single missing slot into a hard failure for the whole action.
        return None
    with np.load(path, allow_pickle=False) as cache:
        if action in FROM_PADDED_WINDOW:
            from security_exp.fiveshot_gesture_timing import carrier_window_imu

            window, _audit = carrier_window_imu(
                window=np.asarray(cache["window"]),
                mask=np.asarray(cache["mask"]),
                samples=int(event["samples"]),
            )
            return np.asarray(window, dtype=np.float32)
        return linear_resample(
            np.asarray(cache["active_imu"], dtype=np.float32), int(event["samples"])
        )


def _process(job):
    source, output = job
    arrays = load_shard(Path(source))
    counts = rewrite_shard_arrays(arrays, imu_generator=generate_imu)
    return Path(source).name, save_shard(Path(output), arrays), counts


def _initialise(cache_root: str, decline: str) -> None:
    _STATE["cache_root"] = Path(cache_root)
    _STATE["decline"] = frozenset(a for a in decline.split(",") if a)


def coverage(cache_root: Path, source_dir: Path, actions) -> dict:
    """How many of each action's slots the cache actually holds.

    Checked before building rather than after, because a part-filled action is
    refused outright and it is cheaper to find that out from the file system
    than from a half-written dataset.
    """

    needed: dict = {action: [0, 0] for action in actions}
    for path in iter_shards(source_dir):
        arrays = load_shard(path)
        for index in np.flatnonzero(arrays["label"] == 1):
            action = str(arrays["action"][index])
            if action not in needed:
                continue
            event = {
                "user_id": str(arrays["user_id"][index]),
                "action": action,
                "split": str(arrays["split"]),
                "sample_idx": int(arrays["sample_idx"][index]),
            }
            needed[action][1] += 1
            needed[action][0] += cache_path(cache_root, event).is_file()
    return needed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="a release bundle to copy and substitute into")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--method-json", default="{}")
    parser.add_argument("--owned-actions", default="",
                        help="actions this release bundle owns; the rest are declined")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--require-complete", action="store_true", default=True,
                        help="decline any action whose slots are not all present")
    parser.add_argument("--allow-partial", dest="require_complete",
                        action="store_false",
                        help="build an action even with missing slots (not for results)")
    args = parser.parse_args()

    owned = {a for a in args.owned_actions.split(",") if a} or set(ACTIONS)
    unknown = owned - set(ACTIONS)
    if unknown:
        raise SystemExit(f"unknown action(s) in --owned-actions: {sorted(unknown)}")
    declined = set(ACTIONS) - owned

    considered = sorted(owned - ALWAYS_DECLINED)
    have = coverage(args.cache_root, args.source_dir, considered)
    for action in considered:
        present, total = have[action]
        if total and present < total:
            print(f"  {action}: cache holds {present}/{total} slots", flush=True)
            if args.require_complete:
                declined.add(action)
                print(f"  {action}: DECLINED (partial coverage would mix release "
                      "and ablation IMU in one detector's training set)", flush=True)
        elif total:
            print(f"  {action}: {present}/{total} slots present", flush=True)

    (args.output_dir / "shards").mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(path), str(args.output_dir / "shards" / path.name))
        for path in iter_shards(args.source_dir)
    ]
    digests: dict = {}
    totals: dict = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialise,
        initargs=(str(args.cache_root), ",".join(sorted(declined))),
    ) as pool:
        for name, digest, counts in pool.map(_process, jobs):
            digests[name] = digest
            merge_counts(totals, counts)

    detail = json.loads(args.method_json)
    detail["cache_root"] = str(args.cache_root)
    detail["actions_declined"] = sorted(declined | ALWAYS_DECLINED)
    detail["keystroke_note"] = (
        "declined by construction: released keystroke fakes record "
        "diffusion_used=false and a generator_source of keystroke_imu_pulse.py, "
        "so no ablation of the diffusion generator can move a keystroke cell"
    )
    detail["slot_coverage"] = {a: {"present": have[a][0], "total": have[a][1]}
                               for a in considered}
    detail["substitution"] = (
        "the release's own binding, replayed from each fake event's sample_idx "
        "against a different cache; scroll re-cut from the padded window via "
        "carrier_window_imu, the rest linear_resample(active_imu, n)"
    )
    ablation_json = args.cache_root / "ablation.json"
    if ablation_json.is_file():
        detail["ablation"] = json.loads(ablation_json.read_text())

    finalise_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        shard_digests=digests,
        counts=totals,
        method_name=args.method_name,
        method_detail=detail,
        swapped={"trajectory_xy": False, "imu": True},
    )
    print(json.dumps({k: v for k, v in totals.items() if k != "per_action"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
