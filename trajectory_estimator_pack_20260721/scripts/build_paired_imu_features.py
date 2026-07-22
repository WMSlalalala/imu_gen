#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from estimator.paired_imu_scorer import build_paired_imu_feature_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=("tap", "scroll", "swipe", "pinch", "keystroke"))
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--real-imu-source", type=Path, required=True)
    parser.add_argument("--fake-imu-root", type=Path, required=True)
    parser.add_argument("--pad-detectors-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--allow-nonformal", action="store_true")
    args = parser.parse_args()
    report = build_paired_imu_feature_table(
        action=args.action, pair_index_path=args.pair_index,
        real_imu_source_path=args.real_imu_source, fake_imu_root=args.fake_imu_root,
        pad_detectors_root=args.pad_detectors_root, output_path=args.output,
        manifest_path=args.manifest, batch_size=args.batch_size,
        require_formal=not args.allow_nonformal,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
