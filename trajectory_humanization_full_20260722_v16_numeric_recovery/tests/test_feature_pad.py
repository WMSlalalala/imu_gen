import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import detectors.feature_pad as feature_pad
from detectors.feature_pad import (
    FeaturePAD,
    fa_frr_curve,
    operating_metrics,
    resample_user_groups,
    run_feature_pad_protocol,
    save_protocol_outputs,
    select_validation_thresholds,
    user_level_bootstrap,
)


def synthetic_table(seed=17):
    """Return a schema-independent, multi-action train/val/test table."""

    rng = np.random.RandomState(seed)
    rows = []
    # Each class has three users and every user contributes four windows.  Large
    # pool offsets in feature 2 make leakage into StandardScaler easy to detect,
    # while feature 1 remains the real/fake signal.
    for pool, pool_offset, user_base in (
        ("train", 0.0, 0),
        ("val", 100.0, 100),
        ("test", 1000.0, 200),
    ):
        for label, center in ((0, -2.0), (1, 2.0)):
            for user_local in range(3):
                user_id = user_base + label * 10 + user_local
                for _ in range(4):
                    feature = np.asarray(
                        [
                            center + rng.normal(scale=0.25),
                            pool_offset + rng.normal(scale=0.3),
                            rng.normal(scale=0.5),
                        ]
                    )
                    rows.append((feature, label, user_id, pool, "swipe"))

    # An unrelated action has extreme train values. It must not enter swipe's
    # scaler or detector fit.
    for label in (0, 1):
        for index in range(4):
            rows.append(
                (
                    np.asarray([5000.0 + label, -7000.0, 9000.0]),
                    label,
                    900 + label * 10 + index,
                    "train",
                    "tap",
                )
            )
    features = np.stack([row[0] for row in rows])
    labels = np.asarray([row[1] for row in rows], dtype=np.int64)
    users = np.asarray([row[2] for row in rows], dtype=np.int64)
    pools = np.asarray([row[3] for row in rows])
    actions = np.asarray([row[4] for row in rows])
    return features, labels, users, pools, actions


class ThresholdSemanticsTests(unittest.TestCase):
    def test_strict_equality_is_rejected_for_both_classes(self):
        metric = operating_metrics(
            labels=np.asarray([0, 0, 1, 1]),
            scores=np.asarray([-1.0, 0.0, 0.0, 1.0]),
            threshold=0.0,
        )
        # real score == threshold is rejected -> FRR includes it.
        self.assertAlmostEqual(metric["frr"], 0.5)
        # fake score == threshold is not accepted -> FA excludes it.
        self.assertAlmostEqual(metric["fa"], 0.0)
        self.assertAlmostEqual(metric["auc"], 0.875)

    def test_curve_direction_and_validation_selection(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        scores = np.asarray([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
        curve = fa_frr_curve(labels, scores)
        self.assertTrue(np.all(np.diff(curve["threshold"]) > 0.0))
        self.assertTrue(np.all(np.diff(curve["fa"]) >= 0.0))
        self.assertTrue(np.all(np.diff(curve["frr"]) <= 0.0))
        selected = select_validation_thresholds(labels, scores)
        frr5 = operating_metrics(labels, scores, selected["val_frr_le_5pct"])
        self.assertLessEqual(frr5["frr"], 0.05)


class FullSyntheticProtocolTests(unittest.TestCase):
    def test_train_only_scaler_val_only_threshold_and_fake_high_scores(self):
        features, labels, users, pools, actions = synthetic_table()
        result = run_feature_pad_protocol(
            features,
            labels,
            users,
            pools,
            actions,
            action="swipe",
            detector_kind="linear_svm",
            bootstrap_replicates=12,
            bootstrap_seed=9,
        )

        train_keep = (pools == "train") & (actions == "swipe")
        expected_train_mean = np.mean(features[train_keep], axis=0)
        all_swipe_mean = np.mean(features[actions == "swipe"], axis=0)
        np.testing.assert_allclose(result.detector.scaler.mean_, expected_train_mean)
        self.assertFalse(np.allclose(result.detector.scaler.mean_, all_swipe_mean))
        self.assertEqual(result.detector.train_row_count, int(np.sum(train_keep)))

        val = result.score_dumps["val"]
        test = result.score_dumps["test"]
        selected_again = select_validation_thresholds(val["label"], val["score"])
        self.assertEqual(result.thresholds["eer"], selected_again["eer"])
        self.assertEqual(
            result.thresholds["val_frr_le_5pct"],
            selected_again["val_frr_le_5pct"],
        )
        self.assertGreater(
            float(np.mean(val["score"][val["label"] == 1])),
            float(np.mean(val["score"][val["label"] == 0])),
        )
        self.assertGreater(result.validation_metrics["eer"]["auc"], 0.99)
        self.assertGreater(result.test_metrics["eer"]["auc"], 0.99)

        for name in ("eer", "val_frr_le_5pct"):
            manual = operating_metrics(
                test["label"], test["score"], result.thresholds[name]
            )
            self.assertEqual(result.test_metrics[name], manual)
        self.assertEqual(result.bootstrap["n_replicates"], 12)
        self.assertEqual(len(result.bootstrap["replicates"]["auc"]), 12)

    def test_changing_test_cannot_change_scaler_or_validation_threshold(self):
        features, labels, users, pools, actions = synthetic_table()
        first = run_feature_pad_protocol(
            features,
            labels,
            users,
            pools,
            actions,
            action="swipe",
            detector_kind="linear_svm",
        )
        changed = features.copy()
        test_keep = (pools == "test") & (actions == "swipe")
        changed[test_keep, 0] *= -1.0
        changed[test_keep, 1] += 100000.0
        second = run_feature_pad_protocol(
            changed,
            labels,
            users,
            pools,
            actions,
            action="swipe",
            detector_kind="linear_svm",
        )
        np.testing.assert_array_equal(first.detector.scaler.mean_, second.detector.scaler.mean_)
        self.assertEqual(first.thresholds, second.thresholds)
        self.assertNotEqual(
            first.test_metrics["eer"]["auc"], second.test_metrics["eer"]["auc"]
        )

    def test_rbf_svm_runs_with_the_same_protocol(self):
        features, labels, users, pools, actions = synthetic_table()
        # Remove the deliberate cross-pool nuisance offset used by the leakage
        # test.  An RBF kernel correctly treats a 1000-sigma shift as OOD and
        # need not preserve accuracy there; this test targets implementation and
        # score direction under an in-distribution split.
        features[pools == "val", 1] -= 100.0
        features[pools == "test", 1] -= 1000.0
        result = run_feature_pad_protocol(
            features,
            labels,
            users,
            pools,
            actions,
            action="swipe",
            detector_kind="rbf_svm",
            model_params={"C": 2.0},
        )
        self.assertGreater(result.validation_metrics["eer"]["auc"], 0.99)
        self.assertEqual(result.detector.detector_kind, "rbf_svm")

    def test_output_bundle_contains_scores_curves_and_protocol_summary(self):
        features, labels, users, pools, actions = synthetic_table()
        result = run_feature_pad_protocol(
            features,
            labels,
            users,
            pools,
            actions,
            action="swipe",
            detector_kind="linear_svm",
            bootstrap_replicates=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = save_protocol_outputs(result, Path(directory))
            self.assertEqual(
                set(paths),
                {
                    "summary",
                    "score_dump",
                    "curves",
                    "bootstrap_summary",
                    "bootstrap_replicates",
                },
            )
            summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
            self.assertEqual(summary["score_direction"], "fake_high")
            self.assertEqual(summary["acceptance_rule"], "score < threshold")
            self.assertEqual(summary["scaler_fit_pool"], "train_only")
            self.assertEqual(summary["threshold_selection_pool"], "validation_only")
            with np.load(paths["score_dump"], allow_pickle=False) as dump:
                self.assertIn("val_score", dump.files)
                self.assertIn("test_user_id", dump.files)
            with np.load(paths["curves"], allow_pickle=False) as curves:
                self.assertIn("test_fa", curves.files)
                self.assertIn("val_frr", curves.files)


class UserBootstrapTests(unittest.TestCase):
    def test_resampling_keeps_every_window_of_each_drawn_user(self):
        labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        users = np.asarray([10, 10, 20, 20, 20, 30, 30, 40, 40, 40, 40])
        indices, audit = resample_user_groups(labels, users, np.random.RandomState(5))

        for label, name in ((0, "real"), (1, "fake")):
            for user in audit[name + "_users"]:
                draw_count = int(np.sum(audit[name + "_draws"] == user))
                original = np.flatnonzero((labels == label) & (users == user))
                for index in original:
                    self.assertEqual(int(np.sum(indices == index)), draw_count)
        self.assertEqual(set(np.unique(labels[indices]).tolist()), {0, 1})

    def test_bootstrap_uses_fixed_thresholds(self):
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        users = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])
        scores = np.asarray([-2.0, -1.0, -0.5, 0.0, 0.1, 0.5, 1.0, 2.0])
        thresholds = {"eer": 0.05, "val_frr_le_5pct": 0.01}
        output = user_level_bootstrap(
            labels, scores, users, thresholds, n_replicates=8, seed=4
        )
        self.assertEqual(output["thresholds"], thresholds)
        self.assertEqual(output["n_real_users"], 2)
        self.assertEqual(output["n_fake_users"], 2)
        self.assertEqual(output["n_replicates"], 8)
        self.assertEqual(len(output["replicates"]["eer_fa"]), 8)


class OptionalXGBoostTests(unittest.TestCase):
    @unittest.skipIf(feature_pad.XGBClassifier is None, "xgboost is not installed")
    def test_installed_xgboost_runs_the_full_split_protocol(self):
        features, labels, users, pools, actions = synthetic_table()
        result = run_feature_pad_protocol(
            features,
            labels,
            users,
            pools,
            actions,
            action="swipe",
            detector_kind="xgboost",
            model_params={"n_estimators": 12, "max_depth": 2},
        )
        self.assertEqual(result.detector.detector_kind, "xgboost")
        self.assertGreater(result.validation_metrics["eer"]["auc"], 0.99)
        self.assertTrue(np.all(np.isfinite(result.score_dumps["test"]["score"])))

    def test_unavailable_xgboost_is_an_explicit_error(self):
        with mock.patch.object(feature_pad, "XGBClassifier", None), mock.patch.object(
            feature_pad, "_XGB_IMPORT_ERROR", ImportError("synthetic missing package")
        ):
            detector = FeaturePAD("xgboost")
            with self.assertRaisesRegex(RuntimeError, "xgboost is unavailable"):
                detector.fit(
                    np.asarray([[-1.0], [-0.5], [0.5], [1.0]]),
                    np.asarray([0, 0, 1, 1]),
                )


if __name__ == "__main__":
    unittest.main()
