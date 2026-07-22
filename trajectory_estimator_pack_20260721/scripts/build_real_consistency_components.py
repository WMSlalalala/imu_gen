#!/usr/bin/env python3
"""Build five formal real cross-modal consistency component tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = Path("/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713")
for path in (ROOT, CORE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from detectors.trajectory_adapter import load_extracted_trajectory_npz  # noqa: E402
from estimator.consistency_component import (  # noqa: E402
    build_real_consistency_component,
    validate_real_consistency_audit,
    write_json,
)
from estimator.paired_dataset import ALLOWED_ACTIONS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-index-dir", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--imu-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--actions", nargs="+", choices=ALLOWED_ACTIONS, default=list(ALLOWED_ACTIONS))
    args = parser.parse_args()
    reports = {}
    for action in ALLOWED_ACTIONS:
        audit_path = args.output_dir / ("real_consistency_%s_audit.json" % action)
        if action not in args.actions:
            if not audit_path.is_file():
                raise ValueError("cannot skip %s without an existing passing audit" % action)
            prior = validate_real_consistency_audit(audit_path, expected_action=action)
            reports[action] = prior
            print("[consistency] %s reused %d" % (action, prior["rows"]), flush=True)
            continue
        trajectory_path = args.trajectory_dir / ("hmog_trajectory_%s.npz" % action)
        records, _ = load_extracted_trajectory_npz(
            trajectory_path, label=0, default_pool="train", sample_prefix="real:"
        )
        output = args.output_dir / ("real_consistency_%s.npz" % action)
        report = build_real_consistency_component(
            pair_index_path=args.pair_index_dir / ("real_pair_index_%s.npz" % action),
            trajectory_records=records,
            imu_path=args.imu_dir / ("hmog_%s.npz" % action),
            output_path=output,
            trajectory_source_path=trajectory_path,
        )
        write_json(audit_path, report)
        reports[action] = report
        print("[consistency] %s %d" % (action, report["rows"]), flush=True)
    manifest = {
        "schema_version": "real_consistency_components_manifest_v1",
        "status": "pass",
        "actions": reports,
        "total_rows": int(sum(row["rows"] for row in reports.values())),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "pass", "total_rows": manifest["total_rows"]}, indent=2))


if __name__ == "__main__":
    main()
