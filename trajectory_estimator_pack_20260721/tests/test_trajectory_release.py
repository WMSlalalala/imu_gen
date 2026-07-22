from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TRAJECTORY_PROJECT = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713"
)
if str(TRAJECTORY_PROJECT) not in sys.path:
    sys.path.insert(0, str(TRAJECTORY_PROJECT))

from detectors.deep_pad import save_raw_sequence_bundle
from detectors.feature_pad import run_feature_pad_protocol, save_protocol_outputs
from detectors.pair_runner import stable_pair_seed
from scripts.run_trajectory_benchmark import synthetic_five_action_dataset

from estimator.feature_estimator import feature_vector_from_record
from estimator.trajectory_release import (
    FEATURE_DETECTORS,
    build_trajectory_estimator_release,
    validate_trajectory_estimator_release,
)
from estimator.runtime_benchmark import (
    benchmark_trajectory_estimator_latency,
    validate_trajectory_latency_report,
)


class TrajectoryReleaseTest(unittest.TestCase):
    def _formal_layout(self, root: Path):
        records, _ = synthetic_five_action_dataset(seed=17)
        records = [record for record in records if record.action == "tap"]
        features = np.stack([feature_vector_from_record(record) for record in records])
        bundle_dir = root / "bundles"
        bundle_dir.mkdir()
        save_raw_sequence_bundle(bundle_dir / "tap.npz", records, features)
        labels = np.asarray([record.label for record in records])
        users = np.asarray([record.user_id for record in records])
        pools = np.asarray([record.pool for record in records])
        actions = np.asarray([record.action for record in records])
        detector_root = root / "pairs"
        for detector in FEATURE_DETECTORS:
            result = run_feature_pad_protocol(
                features, labels, users, pools, actions,
                action="tap", detector_kind=detector,
                random_state=stable_pair_seed(20260713, "tap", "feature_pad", detector),
                bootstrap_replicates=0,
            )
            target = detector_root / "tap" / "feature_pad" / detector / "result"
            save_protocol_outputs(result, target)
        for detector in ("tcn", "transformer"):
            target = detector_root / "tap" / "deep_pad" / detector / "result"
            target.mkdir(parents=True)
            checkpoint = target / "best.pt"
            checkpoint.write_bytes(("checkpoint-" + detector).encode("utf-8"))
            (target / "summary.json").write_text(json.dumps({
                "schema_version": "trajectory_deep_pad_result_v2",
                "action": "tap", "detector_kind": detector,
                "score_direction": "fake_high", "acceptance_rule": "score < threshold",
                "checkpoint_selection_pool": "validation_only",
                "threshold_selection_pool": "validation_only",
                "last_epoch": 40,
                "checkpoint_paths": {"best": "best.pt"},
            }), encoding="utf-8")
        return bundle_dir, detector_root

    def test_release_reconstructs_formal_feature_models_and_binds_deep_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_dir, detector_root = self._formal_layout(root)
            output = root / "release"
            report = build_trajectory_estimator_release(
                bundle_dir=bundle_dir, detector_root=detector_root,
                output_dir=output, actions=("tap",),
                require_formal_five_actions=False,
            )
            self.assertEqual(report["detector_count"], 5)
            self.assertEqual(
                report["feature_model_policy"],
                "exact_reconstruction_verified_against_formal_score_dump",
            )
            manifest = output / "estimator_manifest.json"
            validate_trajectory_estimator_release(
                manifest, expected_bundle_dir=bundle_dir,
                expected_detector_root=detector_root,
                expected_actions=("tap",), require_formal_five_actions=False,
            )
            latency_path = output / "runtime_latency.json"
            benchmark_trajectory_estimator_latency(
                manifest_path=manifest, bundle_dir=bundle_dir,
                output_path=latency_path, device="cpu", actions=("tap",),
                iterations_per_action=4, warmup_per_action=1, load_deep=False,
            )
            validate_trajectory_latency_report(
                latency_path, manifest_path=manifest, bundle_dir=bundle_dir,
                expected_actions=("tap",), expected_iterations=4,
                expected_detectors_per_action=3, expected_device="cpu",
                expected_load_deep=False, expected_warmup_iterations=1,
            )
            original_latency = latency_path.read_text(encoding="utf-8")
            tampered_latency = json.loads(original_latency)
            tampered_latency["actions"]["tap"]["service_latency"]["mean_ms"] += 0.01
            latency_path.write_text(json.dumps(tampered_latency), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match raw samples"):
                validate_trajectory_latency_report(
                    latency_path, manifest_path=manifest, bundle_dir=bundle_dir,
                    expected_actions=("tap",), expected_iterations=4,
                    expected_detectors_per_action=3, expected_device="cpu",
                    expected_load_deep=False, expected_warmup_iterations=1,
                )
            latency_path.write_text(original_latency, encoding="utf-8")
            artifact = output / "feature_pad" / "tap" / "linear_svm" / "artifact.joblib"
            artifact.write_bytes(artifact.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "source/artifact hash mismatch"):
                validate_trajectory_estimator_release(
                    manifest, expected_bundle_dir=bundle_dir,
                    expected_detector_root=detector_root,
                    expected_actions=("tap",), require_formal_five_actions=False,
                )


if __name__ == "__main__":
    unittest.main()
