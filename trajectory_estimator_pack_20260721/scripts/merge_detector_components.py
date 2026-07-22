#!/usr/bin/env python3
"""Merge disjoint real/fake component tables for formal total detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from estimator.component_merge import merge_component_tables  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("imu", "trajectory", "consistency"), required=True)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    report = merge_component_tables(
        input_paths=args.inputs,
        expected_component=args.component,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps({key: report[key] for key in ("status", "component", "action", "rows", "features")}, indent=2))


if __name__ == "__main__":
    main()
