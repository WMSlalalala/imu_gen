#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from estimator.paired_imu_scorer import FORMAL_IMU_SCORERS, train_paired_imu_scorers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=("tap", "scroll", "swipe", "pinch", "keystroke"))
    parser.add_argument("--feature-table", type=Path, required=True)
    parser.add_argument("--output-component", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scorers", nargs="+", default=list(FORMAL_IMU_SCORERS))
    parser.add_argument("--allow-nonformal", action="store_true")
    args = parser.parse_args()
    report = train_paired_imu_scorers(
        feature_table_path=args.feature_table, action=args.action,
        output_component_path=args.output_component, artifact_path=args.artifact,
        manifest_path=args.manifest, seed=args.seed, scorers=args.scorers,
        require_formal=not args.allow_nonformal,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
