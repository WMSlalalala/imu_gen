#!/usr/bin/env python3
"""Build strict real HMOG IMU+trajectory pair indices for all five actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from estimator.paired_dataset import ALLOWED_ACTIONS  # noqa: E402
from estimator.real_pair_index import build_real_pair_index, write_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--imu-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-hash-seed", type=int, default=20260713)
    args = parser.parse_args()
    reports = {}
    for action in ALLOWED_ACTIONS:
        output = args.output_dir / ("real_pair_index_%s.npz" % action)
        report = build_real_pair_index(
            action=action,
            trajectory_path=args.trajectory_dir / ("hmog_trajectory_%s.npz" % action),
            imu_path=args.imu_dir / ("hmog_%s.npz" % action),
            output_path=output,
            real_hash_seed=args.real_hash_seed,
        )
        write_audit(args.output_dir / ("real_pair_index_%s_audit.json" % action), report)
        reports[action] = report
    combined = {
        "schema_version": "hmog_real_pair_indices_manifest_v1",
        "status": "pass",
        "real_hash_seed": int(args.real_hash_seed),
        "actions": reports,
        "total_paired_events": int(sum(row["paired_events"] for row in reports.values())),
    }
    manifest = args.output_dir / "manifest.json"
    write_audit(manifest, combined)
    print(json.dumps({"status": "pass", "manifest": str(manifest), "total_paired_events": combined["total_paired_events"]}, indent=2))


if __name__ == "__main__":
    main()
