import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from detectors.benchmark_runner import BenchmarkConfig, run_complete_benchmark
from detectors.deep_pad import (
    DeepTrainConfig,
    RawSequenceNormalizer,
    RawTCNPAD,
    RawTransformerPAD,
    assign_strict_protocol_pools,
    collate_raw_sequences,
    deep_keycode_embedding_index,
    load_raw_sequence_bundle,
    run_deep_pad_protocol,
    save_raw_sequence_bundle,
)
from scripts.run_trajectory_benchmark import synthetic_five_action_dataset


class GlobalTimelineAndChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.records, cls.features = synthetic_five_action_dataset(seed=31)

    def test_pinch_keeps_staggered_two_pointer_down_up_on_one_global_clock(self):
        records = [row for row in self.records if row.action == "pinch" and row.pool == "train"]
        normalizer = RawSequenceNormalizer().fit(records)
        row = records[0]
        batch = collate_raw_sequences([row], normalizer)
        n = len(row.global_t_ms)
        self.assertTrue(torch.equal(batch.contact_mask[0, :, :n], torch.from_numpy(row.contact_mask)))
        self.assertTrue(batch.contact_mask[0, 0, 0])
        self.assertFalse(batch.contact_mask[0, 1, 0])
        self.assertTrue(batch.contact_mask[0, 1, 2])
        self.assertFalse(batch.contact_mask[0, 1, n - 1])
        self.assertEqual(int(batch.action_code[0, 0, 0]), 0)
        self.assertEqual(int(batch.action_code[0, 1, 2]), 5)
        self.assertEqual(int(batch.action_code[0, 1, n - 2]), 6)
        expected = (row.global_t_ms - row.global_t_ms[0]) / (row.global_t_ms[-1] - row.global_t_ms[0])
        np.testing.assert_allclose(batch.time_progress[0, :n].numpy(), expected, rtol=0, atol=1e-7)

    def test_keystroke_has_explicit_no_contact_gaps_and_keycode_affects_models(self):
        rows = [row for row in self.records if row.action == "keystroke" and row.pool == "train"]
        normalizer = RawSequenceNormalizer().fit(rows)
        row = rows[0]
        self.assertTrue(np.array_equal(np.flatnonzero(row.gap_mask), [2, 5]))
        self.assertFalse(np.any(row.contact_mask[:, row.gap_mask]))
        batch = collate_raw_sequences([row], normalizer)
        changed_codes = row.keycode.copy()
        changed_codes[changed_codes == 29] = 99
        changed = replace(row, keycode=changed_codes, sample_id=row.sample_id + "_changed")
        changed.validate()
        changed_batch = collate_raw_sequences([changed], normalizer)
        for model in (
            RawTCNPAD(hidden_dim=12, n_blocks=1, dropout=0.0),
            RawTransformerPAD(hidden_dim=12, n_layers=1, n_heads=2, feedforward_dim=24, dropout=0.0),
        ):
            model.eval()
            with torch.no_grad():
                first = model(batch)
                second = model(changed_batch)
            self.assertGreater(float(torch.max(torch.abs(first - second))), 1e-8)

    def test_tcn_and_transformer_scores_are_invariant_to_batch_padding_length(self):
        rows = [row for row in self.records if row.pool == "train"]
        short = min(rows, key=lambda row: len(row.global_t_ms))
        long = max(rows, key=lambda row: len(row.global_t_ms))
        self.assertLess(len(short.global_t_ms), len(long.global_t_ms))
        normalizer = RawSequenceNormalizer().fit(rows)
        short_only = collate_raw_sequences([short], normalizer)
        mixed = collate_raw_sequences([short, long], normalizer)
        for model in (
            RawTCNPAD(hidden_dim=12, n_blocks=2, dropout=0.0),
            RawTransformerPAD(
                hidden_dim=12, n_layers=1, n_heads=2,
                feedforward_dim=24, dropout=0.0,
            ),
        ):
            model.eval()
            with torch.no_grad():
                score_alone = model(short_only)[0]
                score_padded = model(mixed)[0]
            torch.testing.assert_close(
                score_alone, score_padded, rtol=0.0, atol=2.0e-7,
                msg="score changed only because the batch maximum T changed",
            )

    def test_rare_keycode_8230_keeps_distinct_shared_token_for_real_and_fake(self):
        row = next(
            row for row in self.records
            if row.action == "keystroke" and row.pool == "train"
        )
        keycode = row.keycode.copy()
        keycode[row.contact_mask] = 8230
        real = replace(row, keycode=keycode, label=0, sample_id="rare_real")
        fake = replace(row, keycode=keycode, label=1, sample_id="rare_fake")
        real.validate()
        fake.validate()
        self.assertEqual(deep_keycode_embedding_index(8230), 8231)
        self.assertNotEqual(
            deep_keycode_embedding_index(8230),
            deep_keycode_embedding_index(1024),
        )
        self.assertEqual(deep_keycode_embedding_index(-1), 0)
        with self.assertRaises(ValueError):
            deep_keycode_embedding_index(16384)
        normalizer = RawSequenceNormalizer().fit([real, fake])
        batch = collate_raw_sequences([real, fake], normalizer)
        for model in (
            RawTCNPAD(hidden_dim=12, n_blocks=1, dropout=0.0),
            RawTransformerPAD(
                hidden_dim=12, n_layers=1, n_heads=2,
                feedforward_dim=24, dropout=0.0,
            ),
        ):
            model.eval()
            with torch.no_grad():
                logits = model(batch)
            torch.testing.assert_close(logits[0], logits[1], rtol=0.0, atol=0.0)

    def test_active_annotation_is_not_a_model_input_but_contact_is(self):
        rows = [row for row in self.records if row.action == "swipe" and row.pool == "train"][:2]
        normalizer = RawSequenceNormalizer().fit(rows)
        batch = collate_raw_sequences(rows, normalizer)
        changed_active = replace(batch, active_mask=torch.zeros_like(batch.active_mask))
        changed_contact_values = batch.contact_mask.clone()
        changed_contact_values[0, 0, 1] = ~changed_contact_values[0, 0, 1]
        changed_contact = replace(batch, contact_mask=changed_contact_values)
        for model in (
            RawTCNPAD(hidden_dim=12, n_blocks=1, dropout=0.0),
            RawTransformerPAD(
                hidden_dim=12, n_layers=1, n_heads=2,
                feedforward_dim=24, dropout=0.0,
            ),
        ):
            model.eval()
            with torch.no_grad():
                original = model(batch)
                annotation_only = model(changed_active)
                physical_contact = model(changed_contact)
            torch.testing.assert_close(original, annotation_only, rtol=0.0, atol=0.0)
            self.assertGreater(
                float(torch.max(torch.abs(original - physical_contact))), 1.0e-8
            )

    def test_transformer_clears_only_padding_nan_and_rejects_valid_nan(self):
        rows = [row for row in self.records if row.action == "swipe" and row.pool == "train"][:2]
        # Force unequal lengths so the second event has semantic padding.
        rows[1] = replace(
            rows[1],
            pointer_continuous=rows[1].pointer_continuous[:, :-2],
            global_t_ms=rows[1].global_t_ms[:-2],
            contact_mask=rows[1].contact_mask[:, :-2],
            active_mask=rows[1].active_mask[:, :-2],
            action_code=np.concatenate((rows[1].action_code[:, :-3], rows[1].action_code[:, -1:]), axis=1),
            keycode=rows[1].keycode[:, :-2],
            event_ids=rows[1].event_ids[:, :-2],
            gap_mask=rows[1].gap_mask[:-2],
        )
        rows[1].validate()
        normalizer = RawSequenceNormalizer().fit(rows)
        batch = collate_raw_sequences(rows, normalizer)
        model = RawTransformerPAD(
            hidden_dim=12, n_layers=1, n_heads=2, feedforward_dim=24, dropout=0.0
        )
        model.eval()
        original = model.encoder.forward

        def padding_nan(*args, **kwargs):
            output = original(*args, **kwargs).clone()
            output[~batch.frame_mask] = float("nan")
            return output

        with mock.patch.object(model.encoder, "forward", side_effect=padding_nan):
            logits = model(batch)
        self.assertTrue(torch.isfinite(logits).all())

        def valid_nan(*args, **kwargs):
            output = original(*args, **kwargs).clone()
            output[0, 0, 0] = float("nan")
            return output

        with mock.patch.object(model.encoder, "forward", side_effect=valid_nan):
            with self.assertRaisesRegex(FloatingPointError, "valid timeline"):
                model(batch)

    def test_normalizer_fits_train_only_and_ignores_val_test_extremes(self):
        rows = [row for row in self.records if row.action == "swipe"]
        train = [row for row in rows if row.pool == "train"]
        first = RawSequenceNormalizer().fit(train)
        changed = []
        for row in rows:
            if row.pool == "train":
                changed.append(row)
            else:
                values = row.pointer_continuous.copy()
                values[row.contact_mask] += 1e6
                changed.append(replace(row, pointer_continuous=values))
        second = RawSequenceNormalizer().fit([row for row in changed if row.pool == "train"])
        np.testing.assert_array_equal(first.pointer_mean, second.pointer_mean)
        np.testing.assert_array_equal(first.pointer_scale, second.pointer_scale)
        self.assertEqual(first.fit_sample_ids, second.fit_sample_ids)


class SplitAndBundleTests(unittest.TestCase):
    def test_real_event_hash_and_fake_user_split_are_separate(self):
        records, _ = synthetic_five_action_dataset(seed=41)
        split = {
            "train": tuple(range(70)),
            "val": tuple(range(70, 80)),
            "test": tuple(range(80, 100)),
        }
        # Synthetic fake users are not the formal 0..99 split, so remap only
        # for this structural split test.
        remapped = []
        fake_counter = {"train": 0, "val": 70, "test": 80}
        for row in records:
            if row.label == 1:
                user = fake_counter[row.pool]
                remapped.append(replace(row, user_id=user))
            else:
                remapped.append(row)
        assigned, audit = assign_strict_protocol_pools(remapped, split, real_hash_seed=7)
        for row in assigned:
            if row.label == 1:
                expected = "train" if row.user_id < 70 else ("val" if row.user_id < 80 else "test")
                self.assertEqual(row.pool, expected)
        groups = {}
        for row in assigned:
            if row.label == 0:
                key = (row.user_id, row.action, row.event_group_id)
                groups.setdefault(key, set()).add(row.pool)
        self.assertTrue(all(len(pools) == 1 for pools in groups.values()))
        self.assertEqual(audit["fake_policy"], "fixed_disjoint_users_70_10_20")
        self.assertIn("complete_event_group", audit["real_policy"])

    def test_real_ranked_hash_gives_all_100_users_60_20_20_event_groups(self):
        records, _ = synthetic_five_action_dataset(seed=43)
        base = next(row for row in records if row.action == "swipe" and row.label == 0)
        real = []
        for user in range(100):
            for event in range(10):
                identity = "real_u%03d_swipe_e%02d" % (user, event)
                real.append(replace(
                    base, user_id=user, sample_id=identity, event_group_id=identity,
                ))
        split = {
            "train": tuple(range(70)), "val": tuple(range(70, 80)),
            "test": tuple(range(80, 100)),
        }
        assigned, audit = assign_strict_protocol_pools(real, split, real_hash_seed=11)
        for pool, expected_per_user in (("train", 6), ("val", 2), ("test", 2)):
            rows = [row for row in assigned if row.pool == pool]
            self.assertEqual(len(rows), 100 * expected_per_user)
            self.assertEqual(len(set(row.user_id for row in rows)), 100)
        self.assertEqual(
            audit["real_complete_event_group_counts"],
            {"train": 600, "val": 200, "test": 200},
        )

    def test_numeric_flat_offset_bundle_roundtrip(self):
        records, features = synthetic_five_action_dataset(seed=51)
        rows = [row for row in records if row.action == "pinch"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pinch.npz"
            save_raw_sequence_bundle(path, rows, features["pinch"])
            with np.load(path, allow_pickle=False) as bundle:
                self.assertEqual(str(bundle["schema_version"].item()), "trajectory_pad_bundle_v2")
                self.assertEqual(
                    str(bundle["feature_schema_version"].item()),
                    "trajectory_features_v2_ahb_table6_hmog_real_up",
                )
                legacy_arrays = {name: bundle[name] for name in bundle.files}
            loaded, loaded_features = load_raw_sequence_bundle(path)
            self.assertEqual(len(loaded), len(rows))
            np.testing.assert_array_equal(loaded_features, features["pinch"])
            for before, after in zip(rows, loaded):
                self.assertEqual(before.sample_id, after.sample_id)
                self.assertEqual(before.event_group_id, after.event_group_id)
                np.testing.assert_array_equal(before.global_t_ms, after.global_t_ms)
                np.testing.assert_array_equal(before.contact_mask, after.contact_mask)
                np.testing.assert_array_equal(before.action_code, after.action_code)
            legacy_arrays["schema_version"] = np.asarray("trajectory_pad_bundle_v1")
            legacy = Path(directory) / "legacy_v1.npz"
            np.savez_compressed(legacy, **legacy_arrays)
            with self.assertRaisesRegex(ValueError, "unsupported bundle schema"):
                load_raw_sequence_bundle(legacy)


class UnifiedRunnerSmokeTests(unittest.TestCase):
    def test_power_loss_between_immutable_best_and_last_replays_and_reuses_exactly(self):
        records, _ = synthetic_five_action_dataset(seed=632)
        config = DeepTrainConfig(
            epochs=2, batch_size=7, learning_rate=1e-3,
            patience=0, bootstrap_replicates=2, seed=124,
        )
        params = {"hidden_dim": 12, "n_blocks": 1, "dropout": 0.2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted = run_deep_pad_protocol(
                records, action="swipe", detector_kind="tcn",
                output_dir=root / "uninterrupted", config=config,
                model_params=params, device="cpu",
            )

            import detectors.deep_pad as deep_module
            original_save = deep_module._atomic_torch_save
            interrupted = {"done": False}

            def save_best_then_interrupt(path, payload):
                original_save(path, payload)
                if (
                    not interrupted["done"]
                    and Path(path).name == "best_epoch_0001.pt"
                ):
                    interrupted["done"] = True
                    raise RuntimeError("synthetic power loss between best and last")

            with mock.patch(
                "detectors.deep_pad._atomic_torch_save",
                side_effect=save_best_then_interrupt,
            ):
                with self.assertRaisesRegex(RuntimeError, "between best and last"):
                    run_deep_pad_protocol(
                        records, action="swipe", detector_kind="tcn",
                        output_dir=root / "resumed", config=config,
                        model_params=params, device="cpu",
                    )
            orphan = root / "resumed" / "checkpoints" / "best_epoch_0001.pt"
            self.assertTrue(orphan.is_file())
            self.assertFalse((root / "resumed" / "checkpoints" / "last.pt").exists())
            orphan_hash = __import__("hashlib").sha256(orphan.read_bytes()).hexdigest()

            resumed = run_deep_pad_protocol(
                records, action="swipe", detector_kind="tcn",
                output_dir=root / "resumed", config=config,
                model_params=params, device="cpu", resume=True,
            )
            self.assertEqual(
                orphan_hash,
                __import__("hashlib").sha256(orphan.read_bytes()).hexdigest(),
            )
            self.assertEqual(uninterrupted.history, resumed.history)
            np.testing.assert_array_equal(
                uninterrupted.score_dumps["test"]["score"],
                resumed.score_dumps["test"]["score"],
            )
            last_a = torch.load(
                root / "uninterrupted" / "checkpoints" / "last.pt",
                map_location="cpu",
            )
            last_b = torch.load(
                root / "resumed" / "checkpoints" / "last.pt",
                map_location="cpu",
            )
            for name in last_a["model_state"]:
                torch.testing.assert_close(
                    last_a["model_state"][name], last_b["model_state"][name],
                    rtol=0.0, atol=0.0,
                )

    def test_epoch_addressed_shuffle_and_dropout_make_resume_exact(self):
        records, _ = synthetic_five_action_dataset(seed=631)
        config = DeepTrainConfig(
            epochs=3, batch_size=7, learning_rate=1e-3,
            patience=0, bootstrap_replicates=2, seed=123,
        )
        params = {"hidden_dim": 12, "n_blocks": 1, "dropout": 0.2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted = run_deep_pad_protocol(
                records, action="swipe", detector_kind="tcn",
                output_dir=root / "uninterrupted", config=config,
                model_params=params, device="cpu",
            )

            import detectors.deep_pad as deep_module
            original_save = deep_module._atomic_torch_save

            def save_then_interrupt(path, payload):
                original_save(path, payload)
                if Path(path).name == "last.pt" and int(payload["epoch"]) == 1:
                    raise RuntimeError("synthetic power loss after epoch-1 last.pt")

            with mock.patch(
                "detectors.deep_pad._atomic_torch_save", side_effect=save_then_interrupt
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic power loss"):
                    run_deep_pad_protocol(
                        records, action="swipe", detector_kind="tcn",
                        output_dir=root / "resumed", config=config,
                        model_params=params, device="cpu",
                    )
            resumed = run_deep_pad_protocol(
                records, action="swipe", detector_kind="tcn",
                output_dir=root / "resumed", config=config,
                model_params=params, device="cpu", resume=True,
            )
            np.testing.assert_array_equal(
                uninterrupted.score_dumps["test"]["score"],
                resumed.score_dumps["test"]["score"],
            )
            self.assertEqual(uninterrupted.history, resumed.history)
            state_a = torch.load(
                root / "uninterrupted" / "checkpoints" / "last.pt", map_location="cpu"
            )
            state_b = torch.load(
                root / "resumed" / "checkpoints" / "last.pt", map_location="cpu"
            )
            self.assertEqual(state_a["epoch"], state_b["epoch"])
            self.assertEqual(state_a["history"], state_b["history"])
            for name in state_a["model_state"]:
                torch.testing.assert_close(
                    state_a["model_state"][name], state_b["model_state"][name],
                    rtol=0.0, atol=0.0,
                )

    def test_deep_training_is_finite_across_seeds_and_exactly_repeatable(self):
        records, _ = synthetic_five_action_dataset(seed=67)
        model_params = {
            "tcn": {"hidden_dim": 12, "n_blocks": 1, "dropout": 0.0},
            "transformer": {
                "hidden_dim": 12, "n_layers": 1, "n_heads": 2,
                "feedforward_dim": 24, "dropout": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for detector_kind in ("tcn", "transformer"):
                repeated = []
                for run_index, seed in enumerate((3, 17, 101, 17)):
                    result = run_deep_pad_protocol(
                        records, action="swipe", detector_kind=detector_kind,
                        output_dir=root / detector_kind / ("run_%d" % run_index),
                        config=DeepTrainConfig(
                            epochs=2, batch_size=24, learning_rate=2e-3,
                            patience=0, bootstrap_replicates=2, seed=seed,
                        ),
                        model_params=model_params[detector_kind], device="cpu",
                    )
                    for pool in ("val", "test"):
                        self.assertTrue(np.all(np.isfinite(result.score_dumps[pool]["score"])))
                    if seed == 17:
                        repeated.append(result.score_dumps["test"]["score"].copy())
                self.assertEqual(len(repeated), 2)
                np.testing.assert_array_equal(repeated[0], repeated[1])

    def test_feature_and_both_deep_families_write_complete_outputs(self):
        records, features = synthetic_five_action_dataset(seed=61)
        config = BenchmarkConfig(
            actions=("swipe",),
            feature_detectors=("linear_svm",),
            deep_detectors=("tcn", "transformer"),
            deep_model_params={
                "tcn": {"hidden_dim": 12, "n_blocks": 1, "dropout": 0.0},
                "transformer": {
                    "hidden_dim": 12, "n_layers": 1, "n_heads": 2,
                    "feedforward_dim": 24, "dropout": 0.0,
                },
            },
            deep_train=DeepTrainConfig(
                epochs=1, batch_size=24, learning_rate=2e-3,
                patience=0, bootstrap_replicates=3, seed=5,
            ),
            feature_bootstrap_replicates=3,
            seed=5,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            outputs = run_complete_benchmark(
                records, features, output_dir=root, config=config, device="cpu"
            )
            self.assertTrue(Path(outputs["manifest"]).exists())
            self.assertTrue(Path(outputs["macro_markdown"]).exists())
            self.assertTrue((Path(outputs["by_action"]) / "swipe.md").exists())
            self.assertTrue((Path(outputs["by_detector"]) / "deep_pad__tcn.csv").exists())
            with Path(outputs["per_action"]).open(encoding="utf-8") as handle:
                rows = list(__import__("csv").DictReader(handle))
            self.assertEqual(len(rows), 6)  # 3 detectors x 2 validation operating points
            self.assertEqual({row["detector"] for row in rows}, {"linear_svm", "tcn", "transformer"})
            for detector in ("tcn", "transformer"):
                detector_root = root / "deep_pad" / "swipe" / detector
                summary = json.loads((detector_root / "summary.json").read_text())
                self.assertEqual(summary["score_direction"], "fake_high")
                self.assertEqual(summary["checkpoint_selection_pool"], "validation_only")
                self.assertFalse(summary["uses_critic"])
                self.assertFalse(summary["uses_selector"])
                self.assertTrue(Path(summary["checkpoint_paths"]["best"]).exists())
                self.assertTrue(Path(summary["checkpoint_paths"]["last"]).exists())
                self.assertTrue((root / "plots" / "swipe" / (detector + ".png")).exists())


if __name__ == "__main__":
    unittest.main()
