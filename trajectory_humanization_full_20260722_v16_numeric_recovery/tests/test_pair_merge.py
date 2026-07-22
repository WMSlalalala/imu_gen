import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from detectors.deep_pad import ACTIONS, DEEP_DETECTORS, DeepTrainConfig
from detectors.feature_pad import ALLOWED_DETECTORS
from detectors.pair_merge import (
    _read_and_audit_pair,
    audit_merged_pair_tree,
    expected_pairs,
    merge_and_audit_pairs,
)
from detectors.pair_runner import (
    PAIR_SCHEMA,
    run_or_resume_pair,
    sha256_file,
    stable_pair_seed,
)


class FormalMergeGateTests(unittest.TestCase):
    @staticmethod
    def _mock_pairs(*, epochs: int, bootstrap: int, patience: int = 0):
        base_seed = 20260713
        output = []
        for action, family, detector in expected_pairs():
            pair_seed = stable_pair_seed(base_seed, action, family, detector)
            train = DeepTrainConfig(
                epochs=epochs, batch_size=64, patience=patience,
                bootstrap_replicates=bootstrap, seed=pair_seed,
            )
            manifest = {
                "action": action,
                "family": family,
                "detector": detector,
                "dataset_sha256": hashlib.sha256(action.encode()).hexdigest(),
                "dataset_file": "/formal/%s.npz" % action,
                "fake_user_split": "/formal/fixed_user_split.json",
                "fake_user_split_sha256": "a" * 64,
                "config": {
                    "formal_protocol": True,
                    "seed": pair_seed,
                    "base_seed": base_seed,
                    "real_hash_seed": base_seed,
                    "feature_bootstrap_replicates": bootstrap,
                    "deep_train": train.__dict__,
                    "feature_model_params": {},
                    "deep_model_params": {},
                    "batch_probe": None,
                },
            }
            output.append((manifest, {}, Path("/formal/pair_manifest.json")))
        return output

    def test_formal_merge_rejects_any_epoch_budget_other_than_40(self):
        with mock.patch(
            "detectors.pair_merge._read_and_audit_pair",
            side_effect=self._mock_pairs(epochs=39, bootstrap=500),
        ):
            with self.assertRaisesRegex(ValueError, "epochs == 40"):
                merge_and_audit_pairs(Path("/not/read"), require_formal=True)
        with mock.patch(
            "detectors.pair_merge._read_and_audit_pair",
            side_effect=self._mock_pairs(epochs=41, bootstrap=500),
        ):
            with self.assertRaisesRegex(ValueError, "epochs == 40"):
                merge_and_audit_pairs(Path("/not/read"), require_formal=True)

    def test_formal_merge_rejects_any_bootstrap_budget_other_than_500(self):
        with mock.patch(
            "detectors.pair_merge._read_and_audit_pair",
            side_effect=self._mock_pairs(epochs=40, bootstrap=499),
        ):
            with self.assertRaisesRegex(ValueError, "bootstrap replicates == 500"):
                merge_and_audit_pairs(Path("/not/read"), require_formal=True)
        with mock.patch(
            "detectors.pair_merge._read_and_audit_pair",
            side_effect=self._mock_pairs(epochs=40, bootstrap=501),
        ):
            with self.assertRaisesRegex(ValueError, "bootstrap replicates == 500"):
                merge_and_audit_pairs(Path("/not/read"), require_formal=True)

    def test_formal_merge_disables_validation_early_stopping(self):
        with mock.patch(
            "detectors.pair_merge._read_and_audit_pair",
            side_effect=self._mock_pairs(epochs=40, bootstrap=500, patience=8),
        ):
            with self.assertRaisesRegex(ValueError, "patience == 0"):
                merge_and_audit_pairs(Path("/not/read"), require_formal=True)


class PairMergeTests(unittest.TestCase):
    @staticmethod
    def _current_pair_fixtures(experiment: Path):
        base_seed = 20260713
        fixtures = {}
        for action_index, (action, family, detector) in enumerate(expected_pairs()):
            pair_seed = stable_pair_seed(base_seed, action, family, detector)
            batch_size = 8 + 4 * (action_index % 3) if family == "deep_pad" else 64
            deep_train = DeepTrainConfig(
                epochs=1, batch_size=batch_size, patience=0,
                bootstrap_replicates=4, seed=pair_seed,
            )
            config = {
                "action": action,
                "family": family,
                "detector": detector,
                "seed": pair_seed,
                "base_seed": base_seed,
                "seed_policy": "sha256(base_seed|action|family|detector)_uint32",
                "formal_protocol": False,
                "real_hash_seed": base_seed,
                "feature_bootstrap_replicates": 4,
                "deep_train": deep_train.__dict__,
                "feature_model_params": {},
                "deep_model_params": {},
                "batch_probe": None,
            }
            pair_root = experiment / "pairs" / action / family / detector
            pair_root.mkdir(parents=True)
            plot = pair_root / "test_fa_frr.png"
            plot.write_bytes(("plot:%s:%s:%s" % (action, family, detector)).encode())
            manifest_path = pair_root / "pair_manifest.json"
            manifest = {
                "schema_version": PAIR_SCHEMA,
                "status": "complete",
                "action": action,
                "family": family,
                "detector": detector,
                "dataset_file": "/formal/%s.npz" % action,
                "dataset_sha256": hashlib.sha256(action.encode()).hexdigest(),
                "fake_user_split": "/formal/fixed_user_split.json",
                "fake_user_split_sha256": "a" * 64,
                "config": config,
                "plot": str(plot.resolve()),
                "plot_sha256": sha256_file(plot),
            }
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, allow_nan=False), encoding="utf-8"
            )
            rows = []
            for point_index, point in enumerate(("eer", "val_frr_le_5pct")):
                value = 0.10 + 0.001 * action_index + 0.01 * point_index
                rows.append({
                    "action": action,
                    "detector_family": family,
                    "detector": detector,
                    "operating_point": point,
                    "threshold_from_validation": 0.0,
                    "validation_fa": value,
                    "validation_frr": value,
                    "validation_auc": 0.70,
                    "test_fa": value + 0.01,
                    "test_frr": value + 0.02,
                    "test_auc": 0.68,
                    "test_fa_ci95_low": value,
                    "test_fa_ci95_high": value + 0.02,
                    "test_frr_ci95_low": value + 0.01,
                    "test_frr_ci95_high": value + 0.03,
                    "test_auc_ci95_low": 0.65,
                    "test_auc_ci95_high": 0.71,
                    "n_test_real": 100,
                    "n_test_fake": 100,
                    "best_epoch": "" if family == "feature_pad" else 1,
                })
            fixtures[(action, family, detector)] = (
                manifest, {"rows": rows, "deep_training_audit": {}}, manifest_path
            )
        return fixtures

    def test_strict_25_pair_merge_writes_50_rows_and_embedded_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory)
            fixtures = self._current_pair_fixtures(experiment)

            def read_fixture(_root, action, family, detector):
                return fixtures[(action, family, detector)]

            with mock.patch(
                "detectors.pair_merge._read_and_audit_pair", side_effect=read_fixture
            ):
                merged = merge_and_audit_pairs(experiment, require_formal=False)
                self.assertEqual(merged["n_pairs"], 25)
                self.assertEqual(merged["n_operating_rows"], 50)
                self.assertEqual(merged["plot_count"], 25)
                report = Path(merged["outputs"]["report"]).read_text(encoding="utf-8")
                for action in ACTIONS:
                    self.assertIn("### " + action, report)
                for detector in tuple(ALLOWED_DETECTORS) + tuple(DEEP_DETECTORS):
                    self.assertIn(detector + ".png", report)
                deep_batch_map = merged["protocol_config"]["batch_size_by_identity"]
                expected_deep = {
                    key: value for key, value in deep_batch_map.items()
                    if "/deep_pad/" in key
                }
                self.assertEqual(len(expected_deep), 10)
                for identity, batch_size in expected_deep.items():
                    self.assertIn('"%s": %d' % (identity, batch_size), report)
                independent = audit_merged_pair_tree(
                    experiment, require_formal=False, write_audit=True
                )
                self.assertEqual(independent["status"], "passed")
                self.assertEqual(independent["n_reaudited_pairs"], 25)
                self.assertEqual(independent["n_recomputed_operating_rows"], 50)
                self.assertEqual(independent["n_verified_plots"], 25)
                self.assertTrue((experiment / "merged" / "benchmark_audit.json").is_file())
                with self.assertRaisesRegex(ValueError, "non-formal/quick"):
                    merge_and_audit_pairs(experiment, require_formal=True)

    def test_pair_merge_reader_accepts_only_current_source_relinked_pair(self):
        # This small real runner fixture prevents the merge test from silently
        # skipping whenever an old on-disk smoke has a pre-provenance schema.
        from tests.test_pair_runner import _small_action_bundle

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, split, _ = _small_action_bundle(root)
            config = DeepTrainConfig(
                epochs=1, batch_size=16, bootstrap_replicates=2, seed=13,
            )
            run_or_resume_pair(
                dataset_file=dataset_dir / "swipe.npz",
                fake_user_split=split,
                output_root=root / "benchmark",
                action="swipe", family="feature_pad", detector="linear_svm",
                deep_config=config, feature_bootstrap_replicates=2,
                seed=13, real_hash_seed=13, require_formal=False,
            )
            manifest, audited, _ = _read_and_audit_pair(
                root / "benchmark", "swipe", "feature_pad", "linear_svm"
            )
            self.assertEqual(manifest["schema_version"], PAIR_SCHEMA)
            self.assertEqual(
                manifest["dataset_relink_audit"], audited["dataset_relink_audit"]
            )
            manifest_path = (
                root / "benchmark" / "pairs" / "swipe" / "feature_pad"
                / "linear_svm" / "pair_manifest.json"
            )
            tampered = json.loads(manifest_path.read_text())
            tampered["dataset_relink_audit"]["unique_identity_count"] += 1
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset relink audit drift"):
                _read_and_audit_pair(
                    root / "benchmark", "swipe", "feature_pad", "linear_svm"
                )


if __name__ == "__main__":
    unittest.main()
