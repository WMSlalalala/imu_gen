#!/usr/bin/env python3
"""Fail-closed audit for all EventPlan-bound fake IMU units."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from estimator.fake_imu_pairs import sha256_file, validate_fake_imu_unit, write_json  # noqa: E402
from estimator.paired_dataset import ALLOWED_ACTIONS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-archive-root", type=Path, required=True)
    parser.add_argument("--fake-imu-root", type=Path, required=True)
    parser.add_argument("--trajectory-generation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-user", type=int, default=200)
    parser.add_argument("--confirm-formal-100k-paired", action="store_true")
    args = parser.parse_args()
    if args.confirm_formal_100k_paired and args.samples_per_user != 200:
        raise ValueError("formal paired audit fixes 200 samples/user/action")
    trajectory_audit = json.loads(args.trajectory_generation_audit.read_text(encoding="utf-8"))
    if trajectory_audit.get("passed") is not True:
        raise ValueError("formal trajectory generation audit did not pass")

    counts = Counter()
    all_ids = set()
    all_plan_ids = set()
    condition_seeds = set()
    trajectory_seeds = set()
    imu_seeds = set()
    unit_hashes = {}
    for action in ALLOWED_ACTIONS:
        trajectory_paths = sorted(
            args.trajectory_archive_root.glob("shards/shard_*_of_*/%s/user_*.npz" % action)
        )
        if len(trajectory_paths) != 100:
            raise ValueError("formal %s trajectory tree must contain 100 user units" % action)
        for trajectory_path in trajectory_paths:
            fake_imu_path = args.fake_imu_root / action / trajectory_path.name
            audit_path = fake_imu_path.with_suffix(".audit.json")
            if not fake_imu_path.is_file() or not audit_path.is_file():
                raise ValueError("missing paired fake IMU unit/audit for %s" % trajectory_path)
            prior = json.loads(audit_path.read_text(encoding="utf-8"))
            if (
                prior.get("status") != "pass"
                or prior.get("output_sha256") != sha256_file(fake_imu_path)
                or prior.get("trajectory_archive_sha256") != sha256_file(trajectory_path)
            ):
                raise ValueError("paired fake IMU unit audit/hash mismatch")
            report = validate_fake_imu_unit(
                fake_imu_path, trajectory_archive_path=trajectory_path,
                expected_action=action, expected_samples=args.samples_per_user,
            )
            with np.load(str(fake_imu_path), allow_pickle=False) as source:
                sample_ids = np.asarray(source["sample_ids"]).astype(str).tolist()
                plan_ids = np.asarray(source["event_plan_sha256"]).astype(str).tolist()
                pools = np.asarray(source["pools"]).astype(str).tolist()
                condition = np.asarray(source["condition_seeds"], dtype=np.int64).tolist()
                trajectory = np.asarray(source["trajectory_noise_seeds"], dtype=np.int64).tolist()
                imu = np.asarray(source["imu_noise_seeds"], dtype=np.int64).tolist()
            if all_ids.intersection(sample_ids) or all_plan_ids.intersection(plan_ids):
                raise ValueError("duplicate paired fake identity across units")
            if (
                condition_seeds.intersection(condition)
                or trajectory_seeds.intersection(trajectory)
                or imu_seeds.intersection(imu)
            ):
                raise ValueError("duplicate seed inside a global paired seed domain")
            all_ids.update(sample_ids)
            all_plan_ids.update(plan_ids)
            condition_seeds.update(condition)
            trajectory_seeds.update(trajectory)
            imu_seeds.update(imu)
            counts[action] += report["rows"]
            counts.update(pools)
            unit_hashes[str(fake_imu_path.resolve())] = prior["output_sha256"]
    expected_total = 5 * 100 * int(args.samples_per_user)
    if len(all_ids) != expected_total or len(all_plan_ids) != expected_total:
        raise ValueError("paired fake IMU total identity count mismatch")
    if condition_seeds & trajectory_seeds or condition_seeds & imu_seeds or trajectory_seeds & imu_seeds:
        raise ValueError("paired condition/trajectory/IMU seed domains overlap")
    if args.confirm_formal_100k_paired:
        expected = {action: 20000 for action in ALLOWED_ACTIONS}
        expected.update(train=70000, val=10000, test=20000)
        if any(counts[name] != value for name, value in expected.items()):
            raise ValueError("formal paired fake IMU action/split counts mismatch")
    report = {
        "schema_version": "paired_fake_imu_formal_audit_v1",
        "passed": True,
        "formal_100k_paired": bool(args.confirm_formal_100k_paired),
        "total_events": expected_total,
        "counts": dict(sorted(counts.items())),
        "unique_sample_ids": len(all_ids),
        "unique_event_plan_sha256": len(all_plan_ids),
        "unique_condition_seeds": len(condition_seeds),
        "unique_trajectory_noise_seeds": len(trajectory_seeds),
        "unique_imu_noise_seeds": len(imu_seeds),
        "all_three_seed_domains_disjoint": True,
        "trajectory_generation_audit": str(args.trajectory_generation_audit.resolve()),
        "trajectory_generation_audit_sha256": sha256_file(args.trajectory_generation_audit),
        "unit_output_sha256": unit_hashes,
    }
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("passed", "total_events", "counts")}, indent=2))


if __name__ == "__main__":
    main()
