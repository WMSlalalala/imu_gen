from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from estimator.duration_metrics import duration_stratified_metrics, fit_duration_bins
from estimator.consistency_component import validate_real_consistency_audit
from estimator.component_merge import merge_component_tables
from estimator.meta_pool import remap_component_to_meta_pools
from estimator.trajectory_score_component import build_trajectory_score_component
from estimator.fake_imu_pairs import FAKE_IMU_PAIR_SCHEMA
from estimator.real_pair_index import REAL_PAIR_INDEX_SCHEMA
from estimator.paired_imu_scorer import (
    IMU_FEATURE_TABLE_SCHEMA, train_paired_imu_scorers,
)
from estimator.trajectory_kinematics_audit import KINEMATIC_METRICS, _event_metrics
from estimator.paired_dataset import PAIRED_DATASET_SCHEMA, PairedDetectorTable
from estimator.paired_dataset_builder import (
    COMPONENT_TABLE_SCHEMA, DetectorComponentTable,
    build_paired_detector_table,
)
from estimator.total_detector import TotalDetectorArtifact, write_training_outputs
from estimator.total_detector_audit import validate_total_detector_outputs
from estimator.runtime_benchmark import benchmark_total_detector_latency


class TotalDetectorProtocolTest(unittest.TestCase):
    def synthetic(self):
        rng = np.random.RandomState(7)
        rows, labels, users, pools, sample_ids, durations = [], [], [], [], [], []
        for pool_index, pool in enumerate(("train", "val", "test")):
            for label in (0, 1):
                for user_index in range(3):
                    for rep in range(4):
                        rows.append(rng.normal(label * 1.2, 0.5, size=3))
                        labels.append(label)
                        users.append(pool_index * 100 + label * 10 + user_index)
                        pools.append(pool)
                        sample_ids.append("%s-%d-%d-%d" % (pool, label, user_index, rep))
                        durations.append(50.0 + 25.0 * rep + pool_index)
        return (
            np.asarray(rows), np.asarray(labels), np.asarray(users),
            np.asarray(pools), np.asarray(sample_ids), np.asarray(durations),
        )

    @staticmethod
    def identities(sample_ids):
        return np.asarray([
            hashlib.sha256(str(value).encode("utf-8")).hexdigest() for value in sample_ids
        ])

    def test_numeric_paired_table_and_consistency_gate(self):
        x, y, users, pools, sample_ids, duration = self.synthetic()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.npz"
            np.savez_compressed(
                str(path), schema_version=np.asarray(PAIRED_DATASET_SCHEMA),
                features=x, feature_names=np.asarray([
                    "imu__score", "trajectory__score", "consistency__peak_delta"
                ]), labels=y, user_ids=users, pools=pools, sample_ids=sample_ids,
                actions=np.asarray(["tap"] * len(y)), duration_ms=duration,
                pair_identity_sha256=self.identities(sample_ids),
            )
            table = PairedDetectorTable.load(path)
            self.assertEqual(table.features.shape, x.shape)
            self.assertEqual(table.feature_names[-1], "consistency__peak_delta")

    def test_train_outputs_bootstrap_and_duration_slices(self):
        x, y, users, pools, sample_ids, duration = self.synthetic()
        names = ("imu__score", "trajectory__score", "consistency__peak_delta")
        artifact = TotalDetectorArtifact.train(
            features=x, labels=y, user_ids=users, pools=pools,
            sample_ids=sample_ids, action="tap", feature_names=names,
            bootstrap_replicates=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_training_outputs(Path(directory), artifact)
            self.assertTrue(Path(paths["bootstrap_summary"]).is_file())
            self.assertTrue(Path(paths["bootstrap_replicates"]).is_file())
        spec = fit_duration_bins(duration[pools == "train"], 4)
        dump = artifact._training_cache["score_dumps"]["test"]
        report = duration_stratified_metrics(
            labels=dump["label"], scores=dump["score"],
            duration_ms=duration[pools == "test"],
            thresholds={"eer": artifact.thresholds["eer"]},
            bin_spec=spec, pool="test",
        )
        self.assertEqual(len(report["rows"]), spec["effective_bins"])
        self.assertEqual(sum(row["n"] for row in report["rows"]), len(dump["label"]))
        self.assertEqual(report["threshold_source"], "validation_global_not_refit_per_bin")

    def test_formal_cli_writes_auditable_outputs(self):
        x, y, users, pools, sample_ids, duration = self.synthetic()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "paired.npz"
            output = root / "output"
            np.savez_compressed(
                str(dataset), schema_version=np.asarray(PAIRED_DATASET_SCHEMA),
                features=x, feature_names=np.asarray([
                    "imu__score", "trajectory__score", "consistency__peak_delta"
                ]), labels=y, user_ids=users, pools=pools, sample_ids=sample_ids,
                actions=np.asarray(["tap"] * len(y)), duration_ms=duration,
                pair_identity_sha256=self.identities(sample_ids),
            )
            script = Path(__file__).resolve().parents[1] / "scripts" / "train_total_detector.py"
            result = subprocess.run(
                [
                    sys.executable, str(script), "--dataset", str(dataset),
                    "--action", "tap", "--output-dir", str(output),
                    "--bootstrap-replicates", "5", "--duration-bins", "4",
                ],
                check=True, capture_output=True, text=True,
            )
            self.assertIn('"status": "complete"', result.stdout)
            manifest = json.loads((output / "training_manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["normalization_fit_pool"], "train_only")
            self.assertEqual(manifest["threshold_selection_pool"], "validation_only")
            self.assertEqual(manifest["test_role"], "fixed_threshold_reporting_only")
            for name in (
                "total_detector.joblib", "summary.json", "score_dump.npz",
                "curves.npz", "bootstrap_summary.json", "bootstrap_replicates.npz",
                "duration_stratified_metrics.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            benchmark_total_detector_latency(
                artifact_path=output / "total_detector.joblib", dataset_path=dataset,
                expected_action="tap", output_path=output / "runtime_latency.json",
                iterations=20, warmup_iterations=2,
            )
            reaudit = validate_total_detector_outputs(
                output, dataset_path=dataset, expected_action="tap",
                expected_bootstrap_replicates=5, expected_duration_bins=4,
                require_runtime_latency=True, expected_latency_iterations=20,
                expected_latency_warmup_iterations=2,
            )
            self.assertTrue(reaudit["passed"])
            self.assertTrue(reaudit["model_inference_exact"])
            self.assertTrue(reaudit["bootstrap_recomputed_exact"])
            latency_path = output / "runtime_latency.json"
            original_latency = latency_path.read_text(encoding="utf-8")
            latency = json.loads(original_latency)
            latency["latency"]["mean_ms"] += 0.01
            latency_path.write_text(json.dumps(latency), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match raw samples"):
                validate_total_detector_outputs(
                    output, dataset_path=dataset, expected_action="tap",
                    expected_bootstrap_replicates=5, expected_duration_bins=4,
                    require_runtime_latency=True, expected_latency_iterations=20,
                    expected_latency_warmup_iterations=2,
                )
            latency_path.write_text(original_latency, encoding="utf-8")
            duration_path = output / "duration_stratified_metrics.json"
            original_duration = duration_path.read_text(encoding="utf-8")
            duration = json.loads(original_duration)
            duration["pools"]["test"]["rows"][0]["operating_points"]["eer"]["threshold"] += 0.1
            duration_path.write_text(json.dumps(duration), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed a global validation threshold"):
                validate_total_detector_outputs(
                    output, dataset_path=dataset, expected_action="tap",
                    expected_bootstrap_replicates=5, expected_duration_bins=4,
                    require_runtime_latency=True, expected_latency_iterations=20,
                    expected_latency_warmup_iterations=2,
                )
            duration_path.write_text(original_duration, encoding="utf-8")
            duration = json.loads(original_duration)
            row = next(
                value for value in duration["pools"]["test"]["rows"]
                if value["auc"] is not None
            )
            row["auc"] = 1.0 - float(row["auc"])
            duration_path.write_text(json.dumps(duration), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differ from paired-data recomputation"):
                validate_total_detector_outputs(
                    output, dataset_path=dataset, expected_action="tap",
                    expected_bootstrap_replicates=5, expected_duration_bins=4,
                    require_runtime_latency=True, expected_latency_iterations=20,
                    expected_latency_warmup_iterations=2,
                )
            duration_path.write_text(original_duration, encoding="utf-8")
            score_path = output / "score_dump.npz"
            with np.load(str(score_path), allow_pickle=False) as source:
                arrays = {name: np.asarray(source[name]) for name in source.files}
            arrays["test_score"] = arrays["test_score"].copy()
            arrays["test_score"][0] += 0.01
            np.savez_compressed(str(score_path), **arrays)
            with self.assertRaisesRegex(ValueError, "do not equal model inference"):
                validate_total_detector_outputs(
                    output, dataset_path=dataset, expected_action="tap",
                    expected_bootstrap_replicates=5, expected_duration_bins=4,
                    require_runtime_latency=True, expected_latency_iterations=20,
                    expected_latency_warmup_iterations=2,
                )

    def test_component_builder_joins_by_identity_not_row_position(self):
        x, y, users, pools, sample_ids, duration = self.synthetic()
        identities = self.identities(sample_ids)
        rng = np.random.RandomState(99)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            component_data = {
                "imu": (x[:, :1], ["imu__score"]),
                "trajectory": (x[:, 1:2], ["trajectory__score"]),
                "consistency": (x[:, 2:3], ["consistency__peak_delta"]),
            }
            for component, (values, names) in component_data.items():
                order = rng.permutation(len(y))
                path = root / (component + ".npz")
                np.savez_compressed(
                    str(path), schema_version=np.asarray(COMPONENT_TABLE_SCHEMA),
                    component=np.asarray(component), features=values[order],
                    feature_names=np.asarray(names), sample_ids=sample_ids[order],
                    labels=y[order], user_ids=users[order], pools=pools[order],
                    actions=np.asarray(["tap"] * len(y))[order], duration_ms=duration[order],
                    pair_identity_sha256=identities[order],
                )
                paths[component] = path
            output = root / "paired.npz"
            manifest = root / "manifest.json"
            report = build_paired_detector_table(
                imu_path=paths["imu"], trajectory_path=paths["trajectory"],
                consistency_path=paths["consistency"], output_path=output,
                manifest_path=manifest,
            )
            table = PairedDetectorTable.load(output)
            expected_order = np.argsort(sample_ids, kind="stable")
            self.assertTrue(np.array_equal(table.sample_ids, sample_ids[expected_order]))
            self.assertTrue(np.allclose(table.features, x[expected_order]))
            self.assertEqual(report["join_key"], "sample_id_exact_set_then_canonical_sort")
            self.assertTrue(manifest.is_file())

    def test_component_builder_rejects_pair_identity_mismatch(self):
        x, y, users, pools, sample_ids, duration = self.synthetic()
        identities = self.identities(sample_ids)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for component, column in (("imu", 0), ("trajectory", 1), ("consistency", 2)):
                observed_identity = identities.copy()
                if component == "consistency":
                    observed_identity[0] = "0" * 64
                path = root / (component + ".npz")
                np.savez_compressed(
                    str(path), schema_version=np.asarray(COMPONENT_TABLE_SCHEMA),
                    component=np.asarray(component), features=x[:, column:column + 1],
                    feature_names=np.asarray([component + "__score"]), sample_ids=sample_ids,
                    labels=y, user_ids=users, pools=pools,
                    actions=np.asarray(["tap"] * len(y)), duration_ms=duration,
                    pair_identity_sha256=observed_identity,
                )
                paths[component] = path
            with self.assertRaisesRegex(ValueError, "pair_identity_sha256"):
                build_paired_detector_table(
                    imu_path=paths["imu"], trajectory_path=paths["trajectory"],
                    consistency_path=paths["consistency"], output_path=root / "paired.npz",
                    manifest_path=root / "manifest.json",
                )

    def test_real_consistency_resume_rechecks_all_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {}
            for name in ("pair_index", "trajectory_source", "imu_source"):
                path = root / (name + ".npz")
                path.write_bytes((name + "-frozen").encode("utf-8"))
                sources[name] = path
            output = root / "consistency.npz"
            sample_ids = np.asarray(["real:0", "real:1", "real:2"])
            np.savez_compressed(
                str(output), schema_version=np.asarray(COMPONENT_TABLE_SCHEMA),
                component=np.asarray("consistency"), features=np.ones((3, 1)),
                feature_names=np.asarray(["consistency__score"]), sample_ids=sample_ids,
                labels=np.zeros(3, dtype=np.int64), user_ids=np.arange(3),
                pools=np.asarray(["train", "val", "test"]),
                actions=np.asarray(["tap"] * 3), duration_ms=np.asarray([10.0, 20.0, 30.0]),
                pair_identity_sha256=self.identities(sample_ids),
            )
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            report = {
                "schema_version": "real_consistency_component_audit_v1",
                "status": "pass", "action": "tap", "rows": 3, "features": 1,
                "feature_names": ["consistency__score"], "output": str(output),
                "output_sha256": digest(output),
            }
            for name, path in sources.items():
                report[name] = str(path)
                report[name + "_sha256"] = digest(path)
            audit = root / "audit.json"
            audit.write_text(json.dumps(report), encoding="utf-8")
            observed = validate_real_consistency_audit(audit, expected_action="tap")
            self.assertEqual(observed["rows"], 3)
            output.write_bytes(output.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "output SHA-256 mismatch"):
                validate_real_consistency_audit(audit, expected_action="tap")

    def test_component_merge_requires_disjoint_real_fake_identities(self):
        x, y, users, pools, sample_ids, duration = self.synthetic()
        identities = self.identities(sample_ids)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for label, name in ((0, "real"), (1, "fake")):
                keep = y == label
                path = root / (name + ".npz")
                np.savez_compressed(
                    str(path), schema_version=np.asarray(COMPONENT_TABLE_SCHEMA),
                    component=np.asarray("consistency"), features=x[keep, 2:3],
                    feature_names=np.asarray(["consistency__peak_delta"]),
                    sample_ids=sample_ids[keep], labels=y[keep], user_ids=users[keep],
                    pools=pools[keep], actions=np.asarray(["tap"] * int(np.sum(keep))),
                    duration_ms=duration[keep], pair_identity_sha256=identities[keep],
                )
                inputs.append(path)
            output = root / "merged.npz"
            report = merge_component_tables(
                input_paths=inputs, expected_component="consistency",
                output_path=output, manifest_path=root / "manifest.json",
            )
            self.assertEqual(report["rows"], len(y))
            self.assertEqual(report["labels"], {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))})

    def test_meta_pool_excludes_base_train_and_splits_base_validation(self):
        split_path = Path(
            "/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json"
        )
        split = json.loads(split_path.read_text())
        rows = []
        for index in range(80):
            rows.append(("real-val-%d" % index, 0, index % 100, "val"))
        for index in range(20):
            rows.append(("real-test-%d" % index, 0, index, "test"))
        for user in split["val_users"]:
            rows.append(("fake-val-%d" % user, 1, user, "val"))
        for user in split["test_users"]:
            rows.append(("fake-test-%d" % user, 1, user, "test"))
        rows.append(("excluded-real-train", 0, 0, "train"))
        rows.append(("excluded-fake-train", 1, split["train_users"][0], "train"))
        sample_ids = np.asarray([row[0] for row in rows])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            np.savez_compressed(
                str(source), schema_version=np.asarray(COMPONENT_TABLE_SCHEMA),
                component=np.asarray("consistency"), features=np.ones((len(rows), 1)),
                feature_names=np.asarray(["consistency__score"]), sample_ids=sample_ids,
                labels=np.asarray([row[1] for row in rows]),
                user_ids=np.asarray([row[2] for row in rows]),
                pools=np.asarray([row[3] for row in rows]),
                actions=np.asarray(["tap"] * len(rows)),
                duration_ms=np.full(len(rows), 50.0),
                pair_identity_sha256=self.identities(sample_ids),
            )
            output = root / "meta.npz"
            report = remap_component_to_meta_pools(
                input_path=source, expected_component="consistency",
                split_json=split_path, output_path=output,
                manifest_path=root / "manifest.json",
            )
            self.assertEqual(report["base_train_rows_excluded"], 2)
            self.assertEqual(len(report["meta_fake_users"]["train"]), 6)
            self.assertEqual(len(report["meta_fake_users"]["val"]), 4)
            self.assertEqual(len(report["meta_fake_users"]["test"]), 20)

    def test_trajectory_scores_relink_rows_ids_and_pair_identity(self):
        action = "tap"
        bundle_ids = np.asarray([
            "real:tap:100", "8001", "real:tap:101", "real:tap:999", "8002",
            "real:tap:102", "8003",
        ])
        labels = np.asarray([0, 1, 0, 0, 1, 0, 1])
        users = np.asarray([1, 11, 2, 9, 12, 3, 13])
        pools = np.asarray(["train", "train", "val", "val", "val", "test", "test"])
        offsets = np.arange(0, 2 * (len(bundle_ids) + 1), 2, dtype=np.int64)
        timeline = np.tile(np.asarray([0.0, 10.0]), len(bundle_ids))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "tap.npz"
            np.savez_compressed(
                str(bundle), schema_version=np.asarray("trajectory_pad_bundle_v2"),
                sequence_offsets=offsets, flat_global_t_ms=timeline,
                label=labels, user_id=users, pool=pools,
                action=np.asarray([action] * len(labels)), sample_id=bundle_ids,
            )
            real_index = root / "real_pairs.npz"
            pair_event_ids = np.asarray([100, 101, 102])
            pair_ids = np.asarray(["real-pair-train", "real-pair-val", "real-pair-test"])
            np.savez_compressed(
                str(real_index), schema_version=np.asarray(REAL_PAIR_INDEX_SCHEMA),
                action=np.asarray(action), sample_ids=pair_ids,
                pair_identity_sha256=self.identities(pair_ids), labels=np.zeros(3, dtype=np.int8),
                user_ids=np.asarray([1, 2, 3]), pools=np.asarray(["train", "val", "test"]),
                duration_ms=np.full(3, 10.0), trajectory_event_ids=pair_event_ids,
            )
            fake_root = root / "fake_imu" / action
            fake_root.mkdir(parents=True)
            fake_ids = np.asarray(["8001", "8002", "8003"])
            np.savez_compressed(
                str(fake_root / "user_000.npz"), schema_version=np.asarray(FAKE_IMU_PAIR_SCHEMA),
                action=np.asarray(action), sample_ids=fake_ids,
                event_plan_sha256=self.identities(fake_ids), user_ids=np.asarray([11, 12, 13]),
                pools=np.asarray(["train", "val", "test"]), duration_ms=np.full(3, 10.0),
            )
            detector_root = root / "detectors"
            score_rows = np.flatnonzero(np.isin(pools, ["val", "test"]))
            for family, detector in (
                ("feature_pad", "linear_svm"), ("feature_pad", "rbf_svm"),
                ("feature_pad", "xgboost"), ("deep_pad", "tcn"),
                ("deep_pad", "transformer"),
            ):
                target = detector_root / action / family / detector / "result"
                target.mkdir(parents=True)
                arrays = {}
                for pool in ("val", "test"):
                    rows = score_rows[pools[score_rows] == pool]
                    arrays[pool + "_score"] = rows.astype(np.float64) / 10.0
                    arrays[pool + "_label"] = labels[rows]
                    arrays[pool + "_user_id"] = users[rows]
                    arrays[pool + "_pool"] = pools[rows]
                    arrays[pool + "_action"] = np.asarray([action] * len(rows))
                    if family == "feature_pad":
                        arrays[pool + "_row_index"] = rows
                    else:
                        arrays[pool + "_sample_id"] = bundle_ids[rows]
                np.savez_compressed(str(target / "score_dump.npz"), **arrays)
            output = root / "trajectory_component.npz"
            report = build_trajectory_score_component(
                action=action, bundle_path=bundle, detector_root=detector_root,
                real_pair_index_path=real_index, fake_imu_root=root / "fake_imu",
                output_path=output, manifest_path=root / "manifest.json",
                require_formal=False,
            )
            table = DetectorComponentTable.load(output, "trajectory")
            self.assertEqual(report["rows"], 4)
            self.assertEqual(report["unpaired_real_trajectory_events_excluded"], 1)
            self.assertEqual(table.features.shape, (4, 5))
            self.assertEqual(set(table.sample_ids.tolist()), {"real-pair-val", "real-pair-test", "8002", "8003"})

    def test_paired_imu_scorer_fits_train_and_exports_only_val_test(self):
        rng = np.random.RandomState(123)
        rows = []
        for pool in ("train", "val", "test"):
            for label in (0, 1):
                for repeat in range(5):
                    rows.append((pool, label, repeat))
        pools = np.asarray([row[0] for row in rows])
        labels = np.asarray([row[1] for row in rows], dtype=np.int64)
        sample_ids = np.asarray(["imu-%s-%d-%d" % row for row in rows])
        hmog = rng.normal(labels[:, None], 0.3, size=(len(rows), 6)).astype(np.float32)
        paper = np.concatenate((hmog, rng.normal(size=(len(rows), 2))), axis=1).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table_path = root / "features.npz"
            np.savez_compressed(
                str(table_path), schema_version=np.asarray(IMU_FEATURE_TABLE_SCHEMA),
                action=np.asarray("tap"), hmog_features=hmog, paper_features=paper,
                sample_ids=sample_ids, pair_identity_sha256=self.identities(sample_ids),
                labels=labels, user_ids=np.arange(len(rows)), pools=pools,
                duration_ms=np.full(len(rows), 50.0), actions=np.asarray(["tap"] * len(rows)),
            )
            component = root / "imu_component.npz"
            report = train_paired_imu_scorers(
                feature_table_path=table_path, action="tap", output_component_path=component,
                artifact_path=root / "scorer.joblib", manifest_path=root / "manifest.json",
                scorers=("hmog_style_svm",), require_formal=False,
            )
            strict = DetectorComponentTable.load(component, "imu")
            self.assertEqual(report["training_rows"], 10)
            self.assertEqual(report["scored_rows"], 20)
            self.assertEqual(set(strict.pools.tolist()), {"val", "test"})
            self.assertFalse(report["base_train_scores_exported"])

    def test_trajectory_kinematics_uses_physical_irregular_time(self):
        values = np.zeros((3, 2, 4), dtype=np.float32)
        values[:, 0, 0] = [0.0, 1.0, 3.0]
        values[:, 0, 2] = [0.2, 0.3, 0.4]
        values[:, 0, 3] = 0.1
        contact = np.zeros((3, 2), dtype=bool)
        contact[:, 0] = True
        event_ids = np.full((3, 2), -1, dtype=np.int64)
        event_ids[:, 0] = 0
        metrics = _event_metrics(
            timeline=np.asarray([0.0, 10.0, 20.0]), values=values,
            contact=contact, event_ids=event_ids,
        )
        observed = dict(zip(KINEMATIC_METRICS, metrics.tolist()))
        self.assertAlmostEqual(observed["duration_ms"], 20.0)
        self.assertAlmostEqual(observed["path_px"], 3.0)
        self.assertAlmostEqual(observed["mean_speed_px_s"], 150.0)
        self.assertAlmostEqual(observed["max_speed_px_s"], 200.0)
        self.assertAlmostEqual(observed["max_accel_px_s2"], 10000.0)


if __name__ == "__main__":
    unittest.main()
