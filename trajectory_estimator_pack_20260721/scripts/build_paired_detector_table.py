#!/usr/bin/env python3
"""Build a strict formal total-detector table from three modality components."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from estimator.paired_dataset_builder import build_paired_detector_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu-table", type=Path, required=True)
    parser.add_argument("--trajectory-table", type=Path, required=True)
    parser.add_argument("--consistency-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest or args.output.with_name(args.output.stem + "_manifest.json")
    result = build_paired_detector_table(
        imu_path=args.imu_table,
        trajectory_path=args.trajectory_table,
        consistency_path=args.consistency_table,
        output_path=args.output,
        manifest_path=manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
