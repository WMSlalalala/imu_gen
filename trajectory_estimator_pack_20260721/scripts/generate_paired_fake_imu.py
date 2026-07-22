#!/usr/bin/env python3
"""Generate fake IMU from the exact EventPlans archived with formal trajectories."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMU_RELEASE = Path(
    "/home/mwang49/real-human/imu_gen/final/android_duration_time_fixed_20260720/"
    "imu_release_20260721"
)
for path in (ROOT, IMU_RELEASE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from estimator.fake_imu_pairs import (  # noqa: E402
    build_fake_imu_unit,
    sha256_file,
    validate_fake_imu_unit,
    write_json,
)
from estimator.paired_dataset import ALLOWED_ACTIONS  # noqa: E402
from imu_release import IMUReleaseService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--actions", nargs="+", choices=ALLOWED_ACTIONS, default=list(ALLOWED_ACTIONS))
    parser.add_argument("--samples-per-user", type=int, default=200)
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--confirm-formal-100k-paired", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.confirm_formal_100k_paired == args.smoke:
        raise ValueError("choose exactly one of --confirm-formal-100k-paired or --smoke")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard id/count")
    actions = tuple(args.actions)
    if args.confirm_formal_100k_paired and (
        set(actions) != set(ALLOWED_ACTIONS) or args.samples_per_user != 200
    ):
        raise ValueError("formal paired generation fixes five actions and 200 samples/user/action")
    if args.smoke and not 1 <= args.samples_per_user <= 5:
        raise ValueError("smoke limits samples-per-user to 1..5")
    all_units = []
    for action in ALLOWED_ACTIONS:
        if action not in actions:
            continue
        paths = sorted(
            args.trajectory_archive_root.glob("shards/shard_*_of_*/%s/user_*.npz" % action)
        )
        expected = 100 if args.confirm_formal_100k_paired else len(paths)
        if len(paths) != expected or not paths:
            raise ValueError("%s trajectory archive count is %d; expected %d" % (action, len(paths), expected))
        all_units.extend((action, path) for path in paths)
    all_units.sort(key=lambda item: (ALLOWED_ACTIONS.index(item[0]), str(item[1])))
    selected = [unit for index, unit in enumerate(all_units) if index % args.num_shards == args.shard_id]
    if not selected:
        raise ValueError("this shard contains no units")

    service = IMUReleaseService(mode="online", seed=42, device=args.device)
    reports = []
    started = time.time()
    for unit_index, (action, trajectory_path) in enumerate(selected, start=1):
        output = args.output_dir / action / trajectory_path.name
        audit_path = output.with_suffix(".audit.json")
        reused = False
        if output.is_file() and audit_path.is_file():
            prior = json.loads(audit_path.read_text(encoding="utf-8"))
            if (
                prior.get("status") == "pass"
                and prior.get("output_sha256") == sha256_file(output)
                and prior.get("trajectory_archive_sha256") == sha256_file(trajectory_path)
            ):
                validate_fake_imu_unit(
                    output, trajectory_archive_path=trajectory_path,
                    expected_action=action, expected_samples=args.samples_per_user,
                )
                report = prior
                reused = True
            else:
                raise ValueError("existing paired fake IMU unit has stale audit: %s" % output)
        else:
            if output.exists() or audit_path.exists():
                raise ValueError("partial paired fake IMU unit exists: %s" % output)
            report = build_fake_imu_unit(
                trajectory_archive_path=trajectory_path,
                output_path=output,
                service=service,
                expected_action=action,
                samples_per_user=args.samples_per_user,
                sample_steps=args.sample_steps,
            )
            report["trajectory_archive_sha256"] = sha256_file(trajectory_path)
            report["formal_100k_paired"] = bool(args.confirm_formal_100k_paired)
            write_json(audit_path, report)
        reports.append(report)
        print(
            "[paired-imu] %d/%d %s %s rows=%d reused=%s"
            % (unit_index, len(selected), action, trajectory_path.stem, report["rows"], reused),
            flush=True,
        )
    manifest = {
        "schema_version": "paired_fake_imu_shard_manifest_v1",
        "status": "pass",
        "formal_100k_paired": bool(args.confirm_formal_100k_paired),
        "trajectory_archive_root": str(args.trajectory_archive_root.resolve()),
        "device": args.device,
        "actions": list(actions),
        "samples_per_user": int(args.samples_per_user),
        "sample_steps_override": args.sample_steps,
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "planned_units": len(selected),
        "completed_units": len(reports),
        "completed_events": int(sum(report["rows"] for report in reports)),
        "elapsed_seconds": time.time() - started,
        "results": reports,
    }
    manifest_path = args.output_dir / (
        "manifest_shard_%03d_of_%03d.json" % (args.shard_id, args.num_shards)
    )
    write_json(manifest_path, manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "completed_units", "completed_events", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
