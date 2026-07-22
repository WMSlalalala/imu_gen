#!/usr/bin/env python3
"""Smoke test for paired IMU+trajectory total detector."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PACK_ROOT = Path(__file__).resolve().parents[1]
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from estimator.total_detector import TotalDetectorArtifact, write_training_outputs  # noqa: E402


def _make_synthetic(seed: int = 20260721):
    rng = np.random.RandomState(seed)
    pools = []
    labels = []
    users = []
    sample_ids = []
    rows = []
    names = (
        "imu__feature__score",
        "imu__feature__val_frr_margin",
        "imu__deep__score",
        "imu__deep__val_frr_margin",
        "trajectory__feature__score",
        "trajectory__feature__val_frr_margin",
        "trajectory__deep__score",
        "trajectory__deep__val_frr_margin",
        "consistency__touch_speed_accel_corr",
        "consistency__motion_touch_peak_delta_ms",
    )
    for pool, n_users, per_user in (("train", 6, 12), ("val", 4, 8), ("test", 4, 8)):
        for label in (0, 1):
            for user in range(label * 100 + {"train": 0, "val": 20, "test": 40}[pool], label * 100 + {"train": 0, "val": 20, "test": 40}[pool] + n_users):
                for rep in range(per_user):
                    # fake rows have moderately higher scores in both modalities.
                    center = 0.0 if label == 0 else 1.25
                    imu = rng.normal(center, 0.9, size=4)
                    traj = rng.normal(center, 0.9, size=4)
                    consistency = rng.normal(center * 0.5, 0.7, size=2)
                    rows.append(np.concatenate([imu, traj, consistency]))
                    labels.append(label)
                    users.append(user)
                    pools.append(pool)
                    sample_ids.append("%s_%d_%d_%d" % (pool, label, user, rep))
    return np.asarray(rows, dtype=np.float64), np.asarray(labels), np.asarray(users), np.asarray(pools), np.asarray(sample_ids), names


def main() -> None:
    x, y, users, pools, sample_ids, names = _make_synthetic()
    artifact = TotalDetectorArtifact.train(
        features=x,
        labels=y,
        user_ids=users,
        pools=pools,
        sample_ids=sample_ids,
        action="tap",
        feature_names=names,
        model_kind="logistic",
        bootstrap_replicates=8,
    )
    out = PACK_ROOT / "results" / "smoke_total_detector"
    paths = write_training_outputs(out, artifact)
    loaded = TotalDetectorArtifact.load(out / "total_detector.joblib")
    score = loaded.score_feature_row(x[0], names)
    report = {
        "schema_version": "total_detector_smoke_v1",
        "status": "passed",
        "paths": paths,
        "loaded_score_finite": bool(np.isfinite(score)),
        "test_metrics": loaded.test_metrics,
    }
    report_path = PACK_ROOT / "results" / "smoke_total_detector_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "passed", "report": str(report_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
