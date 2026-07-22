#!/usr/bin/env python3
"""Build the formal 25-detector runtime trajectory estimator release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from estimator.trajectory_release import build_trajectory_estimator_release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--detector-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=20260713)
    args = parser.parse_args()
    report = build_trajectory_estimator_release(
        bundle_dir=args.bundle_dir,
        detector_root=args.detector_root,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
    )
    print(json.dumps({
        "status": report["status"],
        "detector_count": report["detector_count"],
        "manifest": str((args.output_dir / "estimator_manifest.json").resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
