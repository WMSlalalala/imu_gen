#!/usr/bin/env python3
"""Build one action's five base-trajectory-detector duration report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from estimator.trajectory_duration_report import (
    build_trajectory_duration_report,
    validate_trajectory_duration_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=("tap", "scroll", "swipe", "pinch", "keystroke"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--detector-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-bins", type=int, default=4)
    args = parser.parse_args()
    build_trajectory_duration_report(
        action=args.action,
        bundle_path=args.bundle,
        detector_root=args.detector_root,
        output_path=args.output,
        n_bins=args.duration_bins,
    )
    report = validate_trajectory_duration_report(
        args.output,
        expected_action=args.action,
        expected_bundle=args.bundle,
        expected_detector_root=args.detector_root,
        expected_bins=args.duration_bins,
    )
    print(json.dumps({
        "status": "complete",
        "action": args.action,
        "detector_count": report["detector_count"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
