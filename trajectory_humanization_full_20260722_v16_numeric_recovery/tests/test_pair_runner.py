import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from detectors.deep_pad import (
    DeepTrainConfig,
    assign_strict_protocol_pools,
    load_fake_user_split,
    load_raw_sequence_bundle,
    run_deep_pad_protocol,
    save_raw_sequence_bundle,
)
from detectors.pair_merge import expected_pairs
from detectors.pair_runner import (
    _load_and_assign_action,
    _require_formal_deep_completion,
    audit_protocol_result,
    run_or_resume_pair,
    sha256_file,
    stable_pair_seed,
)
from scripts.run_trajectory_benchmark import synthetic_five_action_dataset


def _small_action_bundle(root: Path):
    records, feature_map = synthetic_five_action_dataset(seed=811)
    original = [row for row in records if row.action == "swipe"]
    original_features = feature_map["swipe"]
    rows, features = [], []
    for row, feature in zip(original, original_features):
        rows.append(row)
        features.append(feature)
        if row.label == 0:
            for copy_index in range(3):
                identity = row.sample_id + "_copy%d" % copy_index
                rows.append(replace(row, sample_id=identity, event_group_id=identity))
                features.append(feature.copy())
    dataset_dir = root / "dataset"
    dataset_dir.mkdir()
    save_raw_sequence_bundle(dataset_dir / "swipe.npz", rows, np.stack(features))

    val = {25, 26, *range(70, 78)}
    test = {45, 46, *range(78, 96)}
    train = set(range(100)) - val - test
    split_path = root / "split.json"
    split_path.write_text(json.dumps({
        "train_users": sorted(train),
        "val_users": sorted(val),
        "test_users": sorted(test),
    }))
    return dataset_dir, split_path, rows


class PairRunnerTests(unittest.TestCase):
    @staticmethod
    def _rewrite_npz(path: Path, arrays):
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    def test_formal_deep_completion_requires_observed_history_exactly_one_to_forty(self):
        valid = {
            "summary": {"last_epoch": 40},
            "deep_training_audit": {
                "history_epoch_count": 40,
                "history_last_epoch": 40,
            },
        }
        _require_formal_deep_completion(valid)
        for field, value in (
            ("last_epoch", 39),
            ("history_epoch_count", 39),
            ("history_last_epoch", 41),
        ):
            candidate = {
                "summary": dict(valid["summary"]),
                "deep_training_audit": dict(valid["deep_training_audit"]),
            }
            target = (
                candidate["summary"]
                if field == "last_epoch" else candidate["deep_training_audit"]
            )
            target[field] = value
            with self.assertRaisesRegex(ValueError, "exactly 40 completed"):
                _require_formal_deep_completion(candidate)

    def test_formal_pair_requires_all_100_real_users_in_every_event_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, split, _ = _small_action_bundle(root)
            with self.assertRaisesRegex(
                ValueError, "real event pools must each include all 100 users"
            ):
                _load_and_assign_action(
                    dataset / "swipe.npz", split, "swipe", 20260713, True
                )

    def test_formal_pair_seeds_are_unique_and_quick_configs_fail_before_data_access(self):
        base = 20260713
        seeds = {
            stable_pair_seed(base, action, family, detector)
            for action, family, detector in expected_pairs()
        }
        self.assertEqual(len(seeds), 25)
        pair_seed = stable_pair_seed(base, "tap", "deep_pad", "tcn")
        with self.assertRaisesRegex(ValueError, "epochs"):
            run_or_resume_pair(
                dataset_file=Path("/not/read.npz"), fake_user_split=Path("/not/read.json"),
                output_root=Path("/not/write"), action="tap", family="deep_pad",
                detector="tcn",
                deep_config=DeepTrainConfig(
                    epochs=1, bootstrap_replicates=500, seed=pair_seed
                ),
                feature_bootstrap_replicates=500, seed=pair_seed,
                real_hash_seed=base, base_seed=base, require_formal=True,
            )
        with self.assertRaisesRegex(ValueError, "500"):
            run_or_resume_pair(
                dataset_file=Path("/not/read.npz"), fake_user_split=Path("/not/read.json"),
                output_root=Path("/not/write"), action="tap", family="deep_pad",
                detector="tcn",
                deep_config=DeepTrainConfig(
                    epochs=40, bootstrap_replicates=4, seed=pair_seed
                ),
                feature_bootstrap_replicates=4, seed=pair_seed,
                real_hash_seed=base, base_seed=base, require_formal=True,
            )
        with self.assertRaisesRegex(ValueError, "exactly 40"):
            run_or_resume_pair(
                dataset_file=Path("/not/read.npz"), fake_user_split=Path("/not/read.json"),
                output_root=Path("/not/write"), action="tap", family="deep_pad",
                detector="tcn",
                deep_config=DeepTrainConfig(
                    epochs=41, patience=0, bootstrap_replicates=500, seed=pair_seed
                ),
                feature_bootstrap_replicates=500, seed=pair_seed,
                real_hash_seed=base, base_seed=base, require_formal=True,
            )
        with self.assertRaisesRegex(ValueError, "patience"):
            run_or_resume_pair(
                dataset_file=Path("/not/read.npz"), fake_user_split=Path("/not/read.json"),
                output_root=Path("/not/write"), action="tap", family="deep_pad",
                detector="tcn",
                deep_config=DeepTrainConfig(
                    epochs=40, patience=8, bootstrap_replicates=500, seed=pair_seed
                ),
                feature_bootstrap_replicates=500, seed=pair_seed,
                real_hash_seed=base, base_seed=base, require_formal=True,
            )
        with self.assertRaisesRegex(ValueError, "exactly 500"):
            run_or_resume_pair(
                dataset_file=Path("/not/read.npz"), fake_user_split=Path("/not/read.json"),
                output_root=Path("/not/write"), action="tap", family="deep_pad",
                detector="tcn",
                deep_config=DeepTrainConfig(
                    epochs=40, patience=0, bootstrap_replicates=501, seed=pair_seed
                ),
                feature_bootstrap_replicates=501, seed=pair_seed,
                real_hash_seed=base, base_seed=base, require_formal=True,
            )

    def test_feature_completed_pair_is_strictly_audited_and_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, split, _ = _small_action_bundle(root)
            config = DeepTrainConfig(epochs=1, batch_size=16, bootstrap_replicates=3, seed=9)
            first = run_or_resume_pair(
                dataset_file=dataset / "swipe.npz", fake_user_split=split,
                output_root=root / "run", action="swipe", family="feature_pad",
                detector="linear_svm", deep_config=config,
                feature_bootstrap_replicates=3, seed=9, real_hash_seed=9,
                require_formal=False,
            )
            second = run_or_resume_pair(
                dataset_file=dataset / "swipe.npz", fake_user_split=split,
                output_root=root / "run", action="swipe", family="feature_pad",
                detector="linear_svm", deep_config=config,
                feature_bootstrap_replicates=3, seed=9, real_hash_seed=9,
                require_formal=False,
            )
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "already_complete")

    def test_feature_without_pair_commit_archives_old_scores_and_retrains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, split_path, _ = _small_action_bundle(root)
            dataset_file = dataset_dir / "swipe.npz"
            config = DeepTrainConfig(
                epochs=1, batch_size=16, bootstrap_replicates=3, seed=23,
            )
            kwargs = dict(
                dataset_file=dataset_file, fake_user_split=split_path,
                output_root=root / "run", action="swipe",
                family="feature_pad", detector="linear_svm",
                deep_config=config, feature_bootstrap_replicates=3,
                seed=23, real_hash_seed=23, require_formal=False,
            )
            run_or_resume_pair(**kwargs)
            pair_root = (
                root / "run" / "pairs" / "swipe" / "feature_pad"
                / "linear_svm"
            )
            old_score_sha = sha256_file(pair_root / "result" / "score_dump.npz")
            (pair_root / "pair_manifest.json").unlink()

            records, features = load_raw_sequence_bundle(dataset_file)
            changed = features.copy()
            changed[:, 0] += np.linspace(-3.0, 5.0, len(changed))
            save_raw_sequence_bundle(dataset_file, records, changed)
            rerun = run_or_resume_pair(**kwargs)
            self.assertEqual(rerun["status"], "completed")
            manifest = json.loads((pair_root / "pair_manifest.json").read_text())
            archived = Path(manifest["archived_unbound_feature_result"])
            self.assertTrue(archived.is_dir())
            self.assertEqual(
                sha256_file(archived / "score_dump.npz"), old_score_sha,
            )
            self.assertEqual(manifest["dataset_sha256"], sha256_file(dataset_file))

            # Even with unchanged source bytes, a result tree without its
            # source/config commit cannot be safely claimed as current.
            (pair_root / "pair_manifest.json").unlink()
            second = run_or_resume_pair(**kwargs)
            self.assertEqual(second["status"], "completed")
            second_manifest = json.loads((pair_root / "pair_manifest.json").read_text())
            self.assertIsNotNone(second_manifest["archived_unbound_feature_result"])
            self.assertEqual(
                len(list((pair_root / "orphaned_unbound_feature_results").iterdir())),
                2,
            )

    def test_pair_audit_rejects_deleted_or_tampered_score_rows_and_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, split_path, _ = _small_action_bundle(root)
            dataset_file = dataset_dir / "swipe.npz"
            config = DeepTrainConfig(
                epochs=1, batch_size=16, bootstrap_replicates=3, seed=29,
            )
            output_root = root / "run"
            run_or_resume_pair(
                dataset_file=dataset_file, fake_user_split=split_path,
                output_root=output_root, action="swipe", family="feature_pad",
                detector="linear_svm", deep_config=config,
                feature_bootstrap_replicates=3, seed=29, real_hash_seed=29,
                require_formal=False,
            )
            result_dir = (
                output_root / "pairs" / "swipe" / "feature_pad"
                / "linear_svm" / "result"
            )
            score_path = result_dir / "score_dump.npz"
            bootstrap_path = result_dir / "bootstrap_replicates.npz"
            with np.load(score_path, allow_pickle=False) as archive:
                original_scores = {name: archive[name].copy() for name in archive.files}
            with np.load(bootstrap_path, allow_pickle=False) as archive:
                original_bootstrap = {name: archive[name].copy() for name in archive.files}

            audit_args = dict(
                action="swipe", family="feature_pad", detector="linear_svm",
                expected_bootstrap_replicates=3,
                dataset_file=dataset_file, fake_user_split=split_path,
                real_hash_seed=29,
                expected_dataset_sha256=sha256_file(dataset_file),
                expected_fake_user_split_sha256=sha256_file(split_path),
                expected_bootstrap_seed=60,
            )

            deleted = {
                name: value[1:].copy() if name.startswith("val_") else value.copy()
                for name, value in original_scores.items()
            }
            self._rewrite_npz(score_path, deleted)
            with self.assertRaisesRegex(ValueError, "row count mismatch"):
                audit_protocol_result(result_dir, **audit_args)

            tampered = {name: value.copy() for name, value in original_scores.items()}
            tampered["test_user_id"][0] += 1
            self._rewrite_npz(score_path, tampered)
            with self.assertRaisesRegex(ValueError, "do not relink"):
                audit_protocol_result(result_dir, **audit_args)

            self._rewrite_npz(score_path, original_scores)
            summary_path = result_dir / "summary.json"
            original_summary_text = summary_path.read_text(encoding="utf-8")
            tampered_summary = json.loads(original_summary_text)
            tampered_summary["scaler_mean"][0] = float(np.nextafter(
                tampered_summary["scaler_mean"][0], np.inf
            ))
            summary_path.write_text(json.dumps(tampered_summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature scaler/train-row audit"):
                audit_protocol_result(result_dir, **audit_args)
            summary_path.write_text(original_summary_text, encoding="utf-8")

            tampered_bootstrap = {
                name: value.copy() for name, value in original_bootstrap.items()
            }
            tampered_bootstrap["auc"][0] = np.nextafter(
                tampered_bootstrap["auc"][0], np.inf
            )
            self._rewrite_npz(bootstrap_path, tampered_bootstrap)
            with self.assertRaisesRegex(ValueError, "fixed-seed recomputation"):
                audit_protocol_result(result_dir, **audit_args)

    def test_deep_resume_rejects_dataset_split_seed_or_full_identity_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir, split_path, _ = _small_action_bundle(root)
            dataset_file = dataset_dir / "swipe.npz"
            config = DeepTrainConfig(
                epochs=1, batch_size=24, learning_rate=2e-3,
                patience=0, bootstrap_replicates=3, seed=37,
            )
            output_root = root / "run"
            kwargs = dict(
                dataset_file=dataset_file, fake_user_split=split_path,
                output_root=output_root, action="swipe", family="deep_pad",
                detector="tcn", deep_config=config,
                feature_bootstrap_replicates=3, seed=37, real_hash_seed=37,
                device="cpu",
                deep_model_params={"hidden_dim": 12, "n_blocks": 1, "dropout": 0.0},
                require_formal=False,
            )
            run_or_resume_pair(**kwargs)
            pair_root = output_root / "pairs" / "swipe" / "deep_pad" / "tcn"
            deep_score_path = pair_root / "result" / "score_dump.npz"
            with np.load(deep_score_path, allow_pickle=False) as archive:
                original_deep_scores = {
                    name: archive[name].copy() for name in archive.files
                }
            tampered_deep_scores = {
                name: value.copy() for name, value in original_deep_scores.items()
            }
            tampered_deep_scores["test_sample_id"][0] = "not-the-dataset-row"
            self._rewrite_npz(deep_score_path, tampered_deep_scores)
            with self.assertRaisesRegex(ValueError, "sample_id.*relink"):
                run_or_resume_pair(**kwargs)
            self._rewrite_npz(deep_score_path, original_deep_scores)

            (pair_root / "pair_manifest.json").unlink()
            (pair_root / "result" / "summary.json").unlink()
            original_dataset = dataset_file.read_bytes()
            original_split = split_path.read_bytes()

            records, features = load_raw_sequence_bundle(dataset_file)
            changed_features = features.copy()
            changed_features[0, 0] = np.nextafter(changed_features[0, 0], np.inf)
            save_raw_sequence_bundle(dataset_file, records, changed_features)
            with self.assertRaisesRegex(ValueError, "(run identity|source/config identity) mismatch"):
                run_or_resume_pair(**kwargs)
            dataset_file.write_bytes(original_dataset)

            split_payload = json.loads(original_split.decode("utf-8"))
            split_path.write_text(json.dumps(split_payload, indent=4) + "\n")
            with self.assertRaisesRegex(ValueError, "(run identity|source/config identity) mismatch"):
                run_or_resume_pair(**kwargs)
            split_path.write_bytes(original_split)

            changed_seed = dict(kwargs)
            changed_seed["real_hash_seed"] = 38
            with self.assertRaisesRegex(ValueError, "(run identity|source/config identity) mismatch"):
                run_or_resume_pair(**changed_seed)

            changed_identity = dict(kwargs)
            changed_identity["base_seed"] = 123456
            with self.assertRaisesRegex(ValueError, "(run identity|source/config identity) mismatch"):
                run_or_resume_pair(**changed_identity)

    def test_deep_partial_with_last_checkpoint_resumes_without_overwriting_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, split_path, rows = _small_action_bundle(root)
            split = load_fake_user_split(split_path)
            assigned, _ = assign_strict_protocol_pools(rows, split, real_hash_seed=17)
            config = DeepTrainConfig(
                epochs=1, batch_size=24, learning_rate=2e-3,
                patience=0, bootstrap_replicates=3, seed=17,
            )
            output_root = root / "run"
            result_dir = output_root / "pairs" / "swipe" / "deep_pad" / "tcn" / "result"
            trained = run_or_resume_pair(
                dataset_file=dataset / "swipe.npz", fake_user_split=split_path,
                output_root=output_root, action="swipe", family="deep_pad",
                detector="tcn", deep_config=config,
                feature_bootstrap_replicates=3, seed=17, real_hash_seed=17,
                device="cpu",
                deep_model_params={"hidden_dim": 12, "n_blocks": 1, "dropout": 0.0},
                require_formal=False,
            )
            best = result_dir / "checkpoints" / "best_epoch_0001.pt"
            best_hash_before = __import__("hashlib").sha256(best.read_bytes()).hexdigest()
            score_path = result_dir / "score_dump.npz"
            with np.load(score_path, allow_pickle=False) as archive:
                original_scores = {
                    name: archive[name].copy() for name in archive.files
                }
            tampered = {name: value.copy() for name, value in original_scores.items()}
            tampered["test_score"][0] = np.nextafter(
                tampered["test_score"][0], np.inf
            )
            self._rewrite_npz(score_path, tampered)
            (output_root / "pairs" / "swipe" / "deep_pad" / "tcn" / "pair_manifest.json").unlink()
            resumed = run_or_resume_pair(
                dataset_file=dataset / "swipe.npz", fake_user_split=split_path,
                output_root=output_root, action="swipe", family="deep_pad",
                detector="tcn", deep_config=config,
                feature_bootstrap_replicates=3, seed=17, real_hash_seed=17,
                device="cpu",
                deep_model_params={"hidden_dim": 12, "n_blocks": 1, "dropout": 0.0},
                require_formal=False,
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(
                best_hash_before,
                __import__("hashlib").sha256(best.read_bytes()).hexdigest(),
            )
            with np.load(score_path, allow_pickle=False) as archive:
                for name, expected in original_scores.items():
                    np.testing.assert_array_equal(archive[name], expected)
            self.assertTrue(
                resumed["manifest"]
                and (output_root / "pairs" / "swipe" / "deep_pad" / "tcn"
                     / "orphaned_unbound_deep_outputs").is_dir()
            )

    def test_pair_recovers_exact_best_to_last_power_loss_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, split_path, rows = _small_action_bundle(root)
            split = load_fake_user_split(split_path)
            assigned, _ = assign_strict_protocol_pools(rows, split, real_hash_seed=19)
            config = DeepTrainConfig(
                epochs=1, batch_size=24, learning_rate=2e-3,
                patience=0, bootstrap_replicates=3, seed=19,
            )
            output_root = root / "run"
            result_dir = output_root / "pairs" / "swipe" / "deep_pad" / "tcn" / "result"
            import detectors.deep_pad as deep_module
            original_save = deep_module._atomic_torch_save

            def save_best_then_interrupt(path, payload):
                original_save(path, payload)
                if Path(path).name == "best_epoch_0001.pt":
                    raise RuntimeError("pair best-to-last interruption")

            with mock.patch(
                "detectors.deep_pad._atomic_torch_save",
                side_effect=save_best_then_interrupt,
            ):
                with self.assertRaisesRegex(RuntimeError, "best-to-last"):
                    run_or_resume_pair(
                        dataset_file=dataset / "swipe.npz", fake_user_split=split_path,
                        output_root=output_root, action="swipe", family="deep_pad",
                        detector="tcn", deep_config=config,
                        feature_bootstrap_replicates=3, seed=19,
                        real_hash_seed=19, device="cpu",
                        deep_model_params={
                            "hidden_dim": 12, "n_blocks": 1, "dropout": 0.0,
                        },
                        require_formal=False,
                    )
            best = result_dir / "checkpoints" / "best_epoch_0001.pt"
            best_hash = __import__("hashlib").sha256(best.read_bytes()).hexdigest()
            self.assertFalse((result_dir / "checkpoints" / "last.pt").exists())
            resumed = run_or_resume_pair(
                dataset_file=dataset / "swipe.npz", fake_user_split=split_path,
                output_root=output_root, action="swipe", family="deep_pad",
                detector="tcn", deep_config=config,
                feature_bootstrap_replicates=3, seed=19, real_hash_seed=19,
                device="cpu",
                deep_model_params={"hidden_dim": 12, "n_blocks": 1, "dropout": 0.0},
                require_formal=False,
            )
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(
                best_hash,
                __import__("hashlib").sha256(best.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
