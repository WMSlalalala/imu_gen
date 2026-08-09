#!/usr/bin/env python3
"""Prove the substitution rule by replaying it against the release's own cache.

The ablation builder claims it can reproduce the release's binding from each
fake event's `sample_idx`.  If that claim is wrong anywhere, every ablation
number built on it is wrong there too, silently -- the dataset would still
verify, still train, still produce a plausible FAR.

So the rule is checked the only way that settles it: run it against the cache
the release itself was composed from, and require the result to be **bit
identical** to the published IMU.  Not close, not correlated -- equal.

Run this before trusting any ablation build.  It reads only, and a failure
names the exact event so the disagreement can be looked at rather than guessed
about.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, "/mnt/share/mwang49/data7/code/direct100k")

from build_ablation_cache_baseline import (  # noqa: E402
    ALWAYS_DECLINED,
    FROM_PADDED_WINDOW,
    cache_path,
)
from hmog_baseline_common import iter_shards, linear_resample, load_shard  # noqa: E402

RELEASE_CACHE = Path(
    "/mnt/share/mwang49/real-human/imu_gen/final/"
    "android_duration_time_fixed_20260720/user_cache_eval_200"
)
BUNDLES = Path("/mnt/share/mwang49/data7/direct100k_final/datasets")


def rebuild(cache: Path, action: str, samples: int) -> np.ndarray | None:
    with np.load(cache, allow_pickle=False) as archive:
        if action in FROM_PADDED_WINDOW:
            from security_exp.fiveshot_gesture_timing import carrier_window_imu

            window, _ = carrier_window_imu(
                window=np.asarray(archive["window"]),
                mask=np.asarray(archive["mask"]),
                samples=samples,
            )
            return np.asarray(window, dtype=np.float32)
        return linear_resample(
            np.asarray(archive["active_imu"], dtype=np.float32), samples
        )


def check_bundle(bundle: Path, owned, cache_root: Path, limit: int | None) -> dict:
    tally: dict = {}
    failures = []
    shards = list(iter_shards(bundle))
    if limit:
        shards = shards[:limit]
    for path in shards:
        arrays = load_shard(path)
        offsets = arrays["offsets"]
        for index in np.flatnonzero(arrays["label"] == 1):
            action = str(arrays["action"][index])
            if action not in owned or action in ALWAYS_DECLINED:
                continue
            start, stop = int(offsets[index]), int(offsets[index + 1])
            event = {
                "user_id": str(arrays["user_id"][index]),
                "action": action,
                "split": str(arrays["split"]),
                "sample_idx": int(arrays["sample_idx"][index]),
            }
            counts = tally.setdefault(action, {"checked": 0, "exact": 0, "missing": 0})
            sample = cache_path(cache_root, event)
            if not sample.is_file():
                counts["missing"] += 1
                continue
            counts["checked"] += 1
            try:
                rebuilt = rebuild(sample, action, stop - start)
            except Exception as error:  # noqa: BLE001
                failures.append(f"{event['user_id']}/{action}/{event['sample_idx']}: "
                                f"{type(error).__name__}: {error}")
                continue
            if np.array_equal(rebuilt, arrays["imu_flat"][start:stop]):
                counts["exact"] += 1
            elif len(failures) < 20:
                published = arrays["imu_flat"][start:stop]
                gap = (float(np.abs(rebuilt - published).max())
                       if rebuilt.shape == published.shape else float("nan"))
                failures.append(
                    f"{event['user_id']}/{action}/sample_{event['sample_idx']:04d}: "
                    f"rebuilt {rebuilt.shape} vs published {published.shape}, "
                    f"max|diff|={gap:.3e}"
                )
    return {"per_action": tally, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=RELEASE_CACHE)
    parser.add_argument("--bundles", type=Path, default=BUNDLES)
    parser.add_argument("--limit-shards", type=int, default=None,
                        help="check only the first N shards per bundle (a smoke run)")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    bundle_map = json.loads((args.bundles / "ACTION_BUNDLE_MAP.json").read_text())
    report: dict = {}
    total_exact = total_checked = total_missing = 0
    all_failures = []

    for name, owned in sorted(bundle_map["bundles"].items()):
        result = check_bundle(args.bundles / name, set(owned), args.cache_root,
                              args.limit_shards)
        report[name] = result
        for action, counts in sorted(result["per_action"].items()):
            total_exact += counts["exact"]
            total_checked += counts["checked"]
            total_missing += counts["missing"]
            rate = counts["exact"] / counts["checked"] if counts["checked"] else 0.0
            flag = "OK " if rate == 1.0 else "BAD"
            print(f"  {flag} {name:14s} {action:10s} "
                  f"{counts['exact']}/{counts['checked']} bit-exact"
                  + (f", {counts['missing']} slots missing" if counts["missing"] else ""))
        all_failures.extend(result["failures"])

    print(f"\n{total_exact}/{total_checked} fake events reproduced bit-exactly"
          + (f", {total_missing} cache slots missing" if total_missing else ""))
    if all_failures:
        print(f"\nfirst {min(len(all_failures), 20)} disagreements:")
        for line in all_failures[:20]:
            print(f"  {line}")

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True))

    if total_checked == 0:
        raise SystemExit("nothing was checked -- is the cache root right?")
    if total_exact != total_checked:
        raise SystemExit(
            f"{total_checked - total_exact} events did not reproduce; the "
            "substitution rule is wrong there and no ablation built on it can "
            "be trusted"
        )
    print("the substitution rule reproduces the release exactly")


if __name__ == "__main__":
    main()
