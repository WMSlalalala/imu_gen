#!/usr/bin/env python3
"""Build five formal fake cross-modal consistency component tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from estimator.fake_consistency_component import build_fake_consistency_component  # noqa: E402
from estimator.fake_imu_pairs import sha256_file, write_json  # noqa: E402
from estimator.paired_dataset import ALLOWED_ACTIONS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-archive-root", type=Path, required=True)
    parser.add_argument("--fake-imu-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-user", type=int, default=200)
    parser.add_argument("--actions", nargs="+", choices=ALLOWED_ACTIONS, default=list(ALLOWED_ACTIONS))
    parser.add_argument("--confirm-formal-100k-paired", action="store_true")
    args = parser.parse_args()
    if args.confirm_formal_100k_paired and (
        set(args.actions) != set(ALLOWED_ACTIONS) or args.samples_per_user != 200
    ):
        raise ValueError("formal fake consistency fixes five actions and 200 samples/user/action")
    reports = {}
    for action in args.actions:
        output = args.output_dir / ("fake_consistency_%s.npz" % action)
        report = build_fake_consistency_component(
            action=action,
            trajectory_archive_root=args.trajectory_archive_root,
            fake_imu_root=args.fake_imu_root,
            split_json=args.split_json,
            output_path=output,
            samples_per_user=args.samples_per_user,
            require_formal=bool(args.confirm_formal_100k_paired),
        )
        report["output_sha256"] = sha256_file(output)
        audit_path = output.with_name(output.stem + "_audit.json")
        write_json(audit_path, report)
        reports[action] = report
        print("[fake-consistency] %s %d" % (action, report["rows"]), flush=True)
    manifest = {
        "schema_version": "fake_consistency_components_manifest_v1",
        "status": "pass",
        "formal_100k_paired": bool(args.confirm_formal_100k_paired),
        "total_rows": int(sum(report["rows"] for report in reports.values())),
        "actions": reports,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "pass", "total_rows": manifest["total_rows"]}, indent=2))


if __name__ == "__main__":
    main()
