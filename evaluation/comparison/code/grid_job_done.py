#!/usr/bin/env python3
"""Is a grid job finished?  Ask the artefacts, not a bookkeeping file.

A queue that records "done" in a text file is only as right as every path that
writes to it, and three separate paths got it wrong here in one evening:

  * a job killed by the GPU curfew was recorded done at 8 cells of 24;
  * a job that refused to start because another instance held its lock exited
    0, and was recorded done without running at all;
  * a job whose driver failed was recorded done because the queue never looked
    at the exit status.

Each of those is a one-line fix, and each would have gone unnoticed until the
tables came out short.  The durable answer is to stop keeping a separate record:
a job is finished when every bundle it owns has the runner's own
`completion.json` on disk, and nothing else counts.

Exits 0 when finished, 1 when not, so it can be used directly in shell tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

B = Path("/mnt/share/mwang49/data7/results/direct100k/baselines")
BUNDLE_MAP = Path(
    "/mnt/share/mwang49/data7/direct100k_final/datasets/ACTION_BUNDLE_MAP.json"
)


def required_bundles(method: str) -> list:
    """The bundles this method actually contributes cells for.

    A bundle whose every owned action the method declined produces no cells and
    must not be waited on -- pyclick declines keystroke, so its keystroke bundle
    is legitimately absent rather than missing.
    """

    root = B / "final" / method
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text())
    needed = []
    for bundle in json.loads(BUNDLE_MAP.read_text())["bundles"]:
        entry = manifest["bundles"].get(bundle)
        if not entry:
            continue
        swapped = entry.get("swapped", {})
        for action in entry["owned_actions"]:
            tally = swapped.get(action, {})
            if (not tally.get("fake", 0)) or tally.get("imu_swapped", 0) \
                    or tally.get("trajectory_swapped", 0):
                needed.append(bundle)
                break
    return needed


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: grid_job_done.py <method> <modality>")
    method, modality = sys.argv[1], sys.argv[2]
    bundles = required_bundles(method)
    if not bundles:
        raise SystemExit(1)   # not built yet -> not done
    missing = [
        b for b in bundles
        if not (B / "final" / method / f"cells_{b}_{modality}" / "completion.json").is_file()
    ]
    if missing:
        print(" ".join(missing))
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
