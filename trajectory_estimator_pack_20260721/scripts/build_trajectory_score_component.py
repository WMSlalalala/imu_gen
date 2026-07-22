#!/usr/bin/env python3
"""Build one action's exact paired trajectory-score component."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from estimator.trajectory_score_component import build_trajectory_score_component


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=("tap", "scroll", "swipe", "pinch", "keystroke"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--detector-root", type=Path, required=True)
    parser.add_argument("--real-pair-index", type=Path, required=True)
    parser.add_argument("--fake-imu-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-nonformal", action="store_true")
    args = parser.parse_args()
    report = build_trajectory_score_component(
        action=args.action, bundle_path=args.bundle, detector_root=args.detector_root,
        real_pair_index_path=args.real_pair_index, fake_imu_root=args.fake_imu_root,
        output_path=args.output, manifest_path=args.manifest,
        require_formal=not args.allow_nonformal,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
