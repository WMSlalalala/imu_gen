#!/usr/bin/env python3
"""End-to-end smoke for feature artifact export and runtime estimation."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PROJECT = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713"
)
for path in (PACK_ROOT, TRAJECTORY_PROJECT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from estimator.service import TrajectoryEstimatorService  # noqa: E402
from scripts.run_trajectory_benchmark import synthetic_five_action_dataset  # noqa: E402
from scripts.train_feature_estimator_artifacts import export_feature_artifacts  # noqa: E402


def main() -> None:
    out = PACK_ROOT / "results" / "smoke_feature_estimator"
    if out.exists():
        shutil.rmtree(out)
    manifest = export_feature_artifacts(
        output_dir=out,
        synthetic_smoke=True,
        dataset_dir=None,
        fake_user_split=None,
        actions=("tap", "scroll", "swipe", "pinch", "keystroke"),
        detectors=("linear_svm", "rbf_svm"),
        bootstrap_replicates=0,
        seed=20260713,
    )
    service = TrajectoryEstimatorService.load(out / "estimator_manifest.json", load_deep=False)
    records, _ = synthetic_five_action_dataset(seed=13)
    estimates = []
    for action in ("tap", "scroll", "swipe", "pinch", "keystroke"):
        record = next(row for row in records if row.action == action and row.pool == "test")
        estimates.append(service.estimate_record(record))
    report = {
        "schema_version": "trajectory_estimator_pack_smoke_v1",
        "status": "passed",
        "manifest": str(out / "estimator_manifest.json"),
        "n_manifest_rows": len(manifest["summary_rows"]),
        "actions": service.actions(),
        "estimate_count": len(estimates),
        "estimates": estimates,
    }
    result_path = PACK_ROOT / "results" / "smoke_estimator_report.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "passed", "report": str(result_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

