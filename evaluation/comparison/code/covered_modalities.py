#!/usr/bin/env python3
"""Which modalities a built attack may legitimately be reported on.

A detector cell reads specific channels.  A method that swapped only the
inertial channel leaves the trajectory exactly as the release wrote it, so its
`trajectory_xytime` cells are not measuring that method at all -- they
reproduce the release's own numbers, and averaging them in drags the method's
score toward the release's.

That is not hypothetical: cross-scoring showed pyclick and ghost-cursor at
0.835 on `imu_only`, identical to the release, because neither touches inertial
data; and diffts_imu at 0.763 on `trajectory_xytime` for the same reason in
reverse.

The joint modality is stricter still.  It reads both channels, so for a
single-channel method a joint cell is half the method and half the release --
a hybrid attack nobody proposed.  Only methods that swapped both channels are
reported there.

Truth comes from the built dataset's own `release.json`, written by
`finalise_dataset`, not from the method's name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mnt/share/mwang49/data7/results/direct100k/baselines/final")
BUNDLES = ("scroll", "swipe", "keystroke", "tap_and_pinch")


def swapped_channels(method: str) -> dict:
    """{'imu': bool, 'trajectory_xy': bool} as the build recorded it."""

    for bundle in BUNDLES:
        release = ROOT / method / bundle / "release.json"
        if release.is_file():
            return json.loads(release.read_text()).get("baseline_swapped", {})
    return {}


def covered(method: str) -> list:
    channels = swapped_channels(method)
    imu = bool(channels.get("imu"))
    trajectory = bool(channels.get("trajectory_xy"))
    modalities = []
    if trajectory:
        modalities.append("trajectory_xytime")
    if imu:
        modalities.append("imu_only")
    if imu and trajectory:
        modalities.append("imu_trajectory_xytime")
    return modalities


def main() -> None:
    if len(sys.argv) == 2:
        print(" ".join(covered(sys.argv[1])))
        return
    for path in sorted(ROOT.glob("*")):
        if path.name.startswith("_") or not (path / "bundle_manifest.json").is_file():
            continue
        channels = swapped_channels(path.name)
        print(f"  {path.name:22s} swapped={channels}  -> {covered(path.name)}")


if __name__ == "__main__":
    main()
