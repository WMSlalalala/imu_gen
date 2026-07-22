#!/usr/bin/env python3
"""Create leakage-safe level-2 train/val/test component pools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from estimator.meta_pool import remap_component_to_meta_pools  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("imu", "trajectory", "consistency"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = remap_component_to_meta_pools(
        input_path=args.input, expected_component=args.component,
        split_json=args.split_json, output_path=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps({key: report[key] for key in ("status", "component", "action", "meta_rows", "meta_pool_counts")}, indent=2))


if __name__ == "__main__":
    main()
