from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from estimator.trajectory_duration_report import (
    TRAJECTORY_DURATION_REPORT_SCHEMA,
    build_trajectory_duration_report,
    validate_trajectory_duration_report,
)
from estimator.trajectory_score_component import DETECTORS


class TrajectoryDurationReportTest(unittest.TestCase):
    def _fixture(self, root: Path):
        bundle_path = root / "tap.npz"
        pools = []
        labels = []
        users = []
        sample_ids = []
        durations = []
        for pool_index, pool in enumerate(("train", "val", "test")):
            for label in (0, 1):
                for rep in range(6):
                    pools.append(pool)
                    labels.append(label)
                    users.append(pool_index * 100 + label * 10 + rep // 2)
                    sample_ids.append("%s-%d-%d" % (pool, label, rep))
                    durations.append(25.0 + rep * 20.0 + label * 3.0 + pool_index)
        pools = np.asarray(pools)
        labels = np.asarray(labels, dtype=np.int64)
        users = np.asarray(users, dtype=np.int64)
        sample_ids = np.asarray(sample_ids)
        durations = np.asarray(durations, dtype=np.float64)
        offsets = np.arange(0, 2 * (len(labels) + 1), 2, dtype=np.int64)
        flat_t = np.empty(2 * len(labels), dtype=np.float64)
        flat_t[0::2] = 0.0
        flat_t[1::2] = durations
        np.savez_compressed(
            str(bundle_path),
            schema_version=np.asarray("trajectory_pad_bundle_v2"),
            sequence_offsets=offsets,
            flat_global_t_ms=flat_t,
            label=labels,
            user_id=users,
            pool=pools,
            action=np.asarray(["tap"] * len(labels)),
            sample_id=sample_ids,
        )
        detector_root = root / "pairs"
        for detector_index, (family, detector) in enumerate(DETECTORS):
            result_root = detector_root / "tap" / family / detector / "result"
            result_root.mkdir(parents=True)
            arrays = {}
            for pool in ("val", "test"):
                rows = np.flatnonzero(pools == pool)
                arrays[pool + "_score"] = (
                    labels[rows] * 0.8 + 0.1 + detector_index * 0.001
                ).astype(np.float64)
                arrays[pool + "_label"] = labels[rows]
                arrays[pool + "_user_id"] = users[rows]
                arrays[pool + "_pool"] = pools[rows]
                arrays[pool + "_action"] = np.asarray(["tap"] * len(rows))
                if family == "feature_pad":
                    arrays[pool + "_row_index"] = rows
                else:
                    arrays[pool + "_sample_id"] = sample_ids[rows]
            np.savez_compressed(str(result_root / "score_dump.npz"), **arrays)
            (result_root / "summary.json").write_text(
                json.dumps({
                    "action": "tap",
                    "detector_kind": detector,
                    "score_direction": "fake_high",
                    "acceptance_rule": "score < threshold",
                    "threshold_selection_pool": "validation_only",
                    "thresholds": {
                        "eer": 0.5 + detector_index * 0.001,
                        "val_frr_le_5pct": 0.7 + detector_index * 0.001,
                        "target_frr": 0.05,
                    },
                }),
                encoding="utf-8",
            )
        return bundle_path, detector_root, pools, labels

    def test_all_five_detectors_have_train_only_bins_and_fixed_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, detector_root, pools, labels = self._fixture(root)
            output = root / "duration.json"
            report = build_trajectory_duration_report(
                action="tap", bundle_path=bundle, detector_root=detector_root,
                output_path=output, n_bins=4,
            )
            self.assertEqual(report["schema_version"], TRAJECTORY_DURATION_REPORT_SCHEMA)
            self.assertEqual(report["detector_count"], 5)
            self.assertEqual(report["bin_spec"]["fit_pool"], "train_only")
            self.assertEqual(report["bin_spec"]["train_count"], int(np.sum(pools == "train")))
            for value in report["detectors"].values():
                self.assertEqual(set(value["pools"]), {"val", "test"})
                for pool in ("val", "test"):
                    pool_report = value["pools"][pool]
                    self.assertEqual(
                        pool_report["threshold_source"],
                        "validation_global_not_refit_per_bin",
                    )
                    self.assertEqual(
                        sum(row["n"] for row in pool_report["rows"]),
                        int(np.sum(pools == pool)),
                    )
                    for row in pool_report["rows"]:
                        self.assertEqual(
                            set(row["operating_points"]),
                            {"eer", "val_frr_le_5pct"},
                        )
            observed = validate_trajectory_duration_report(
                output, expected_action="tap", expected_bundle=bundle,
                expected_detector_root=detector_root, expected_bins=4,
            )
            self.assertTrue(observed["passed"])
            tampered = json.loads(output.read_text(encoding="utf-8"))
            row = next(
                value
                for value in tampered["detectors"]["feature_pad/linear_svm"]["pools"]["test"]["rows"]
                if value["auc"] is not None
            )
            row["auc"] = 1.0 - float(row["auc"])
            output.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differ from score recomputation"):
                validate_trajectory_duration_report(
                    output, expected_action="tap", expected_bundle=bundle,
                    expected_detector_root=detector_root, expected_bins=4,
                )

    def test_exact_score_metadata_relink_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, detector_root, _, _ = self._fixture(root)
            score_path = detector_root / "tap" / "feature_pad" / "linear_svm" / "result" / "score_dump.npz"
            with np.load(str(score_path), allow_pickle=False) as source:
                arrays = {name: np.asarray(source[name]) for name in source.files}
            arrays["val_user_id"] = arrays["val_user_id"].copy()
            arrays["val_user_id"][0] += 1
            np.savez_compressed(str(score_path), **arrays)
            with self.assertRaisesRegex(ValueError, "metadata does not relink exactly"):
                build_trajectory_duration_report(
                    action="tap", bundle_path=bundle, detector_root=detector_root,
                    output_path=root / "duration.json", n_bins=4,
                )

    def test_revalidation_rejects_source_hash_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, detector_root, _, _ = self._fixture(root)
            output = root / "duration.json"
            build_trajectory_duration_report(
                action="tap", bundle_path=bundle, detector_root=detector_root,
                output_path=output, n_bins=4,
            )
            summary = detector_root / "tap" / "deep_pad" / "tcn" / "result" / "summary.json"
            value = json.loads(summary.read_text(encoding="utf-8"))
            value["unrelated_but_hash_relevant"] = True
            summary.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source identity/hash mismatch"):
                validate_trajectory_duration_report(
                    output, expected_action="tap", expected_bundle=bundle,
                    expected_detector_root=detector_root, expected_bins=4,
                )


if __name__ == "__main__":
    unittest.main()
