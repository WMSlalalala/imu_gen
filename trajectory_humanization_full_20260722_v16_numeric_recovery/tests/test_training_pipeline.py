#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

from tests.synthetic_training_corpus import (
    keystroke_events,
    staggered_pinch_events,
    tap_events_for_all_users,
    write_archive,
)
from training.corpus import (
    FORMAL_SPLIT_PATH, NumericTrajectoryCorpus, SplitDefinition,
    canonical_sample_sha256,
)
from training.engine import (
    CUBLAS_WORKSPACE_CONFIG, TrainingConfig,
    _backoff_amp_scaler_without_optimizer_step, _restore_rng_state, _rng_state,
    _validated_loss_result,
    evaluate_full_validation, make_seeded_generator, runtime_determinism_audit,
    seed_everything, train_action, validation_epochs,
)
from training.fewshot_dataset import (
    DeterministicLengthBucketBatchSampler,
    ReferenceRegistry,
    StrictFiveReferenceDataset,
    StrictVariableLengthCollator,
)


class StrictTrainingPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.splits = SplitDefinition.load(FORMAL_SPLIT_PATH, require_pinned_hash=True)

    def test_runtime_determinism_contract_is_enabled_and_auditable(self):
        seed_everything(20260713)
        expected = {
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms_enabled": True,
            "deterministic_algorithms_warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        }
        self.assertEqual(CUBLAS_WORKSPACE_CONFIG, ":4096:8")
        self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")
        self.assertEqual(runtime_determinism_audit(), expected)

    def test_numeric_sample_id_staggered_pinch_offsets_and_no_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_archive(Path(directory) / "hmog_trajectory_pinch.npz", "pinch", staggered_pinch_events())
            with np.load(str(path), allow_pickle=False) as archive:
                self.assertTrue(all(np.asarray(archive[name]).dtype.kind != "O" for name in archive.files))
            corpus = NumericTrajectoryCorpus(path, self.splits, expected_action="pinch")
            exact = corpus.event_flat_rows(0)
            self.assertEqual(exact["flat_x"].shape, (9,))
            self.assertEqual(exact["flat_pointer_id"].tolist(), [0, 0, 0, 7, 0, 7, 0, 7, 0])
            raw = corpus.raw_sample(0)
            self.assertEqual(raw["sample_id"], "3000")
            self.assertEqual(raw["metadata"]["pointer_ids"], [0, 7])
            self.assertEqual(raw["metadata"]["pointer_start_offsets_ms"], [0.0, 20.0])
            self.assertEqual(raw["metadata"]["pointer_end_offsets_ms"], [100.0, 80.0])
            self.assertEqual(raw["pointers"][1]["timestamps_ms"].tolist(), [20.0, 50.0, 80.0])
            self.assertEqual(raw["pointers"][0]["timestamps_ms"].tolist(), [0.0, 20.0, 50.0, 80.0, 100.0])
            self.assertEqual(float(raw["pointers"][0]["xy"][1, 0]), 106.0)
            sample = corpus.canonical_sample(0)
            np.testing.assert_array_equal(sample.pointer_start_offset_ms, [0.0, 20.0])
            np.testing.assert_array_equal(sample.pointer_end_offset_ms, [100.0, 80.0])
            self.assertEqual(corpus.event_length(0), 9)
            self.assertEqual(corpus.canonical_max_points(0), 5)
            self.assertEqual(
                corpus.canonical_max_points(0),
                max(int(values.shape[0]) for values in sample.pointer_features),
            )

    def test_keystroke_offsets_restore_discrete_contacts_and_separate_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_archive(Path(directory) / "hmog_trajectory_keystroke.npz", "keystroke", keystroke_events())
            corpus = NumericTrajectoryCorpus(path, self.splits, expected_action="keystroke")
            raw = corpus.raw_sample(0)
            self.assertEqual(len(raw["contacts"]), 2)
            self.assertEqual(raw["metadata"]["n_keys"], 2)
            self.assertEqual(raw["metadata"]["n_letters"], 1)
            self.assertEqual(raw["metadata"]["raw_keycodes"], [-5, 97])
            self.assertEqual([contact["keycode"] for contact in raw["contacts"]], [0, 97])
            self.assertEqual(raw["contacts"][0]["end_offset_ms"], 30.0)
            self.assertEqual(raw["contacts"][1]["start_offset_ms"], 80.0)
            sample = corpus.canonical_sample(0)
            self.assertEqual(sample.n_keys, 2)
            self.assertEqual(sample.n_letters, 1)
            self.assertTrue(np.any(~sample.pointer_contact_masks[0]))
            self.assertTrue(np.all(sample.pointer_event_ids[0][~sample.pointer_contact_masks[0]] == -1))
            self.assertEqual(corpus.event_length(0), 4)
            self.assertEqual(corpus.canonical_max_points(0), 5)
            self.assertEqual(corpus.canonical_max_points(0), sample.pointer_features[0].shape[0])

            # Formal generation must see exactly the same canonical refs as
            # training, including UNKNOWN key token and contact gaps.
            from generation.corpus import load_action_corpus
            from generation.protocol import FixedUserSplit
            generation_split = FixedUserSplit.load(str(FORMAL_SPLIT_PATH), require_formal=True)
            generation_sample = load_action_corpus(
                str(path), "keystroke", generation_split, user_ids=[0], strict=True
            )[0]
            np.testing.assert_allclose(sample.pointer_features[0], generation_sample.pointer_features[0])
            np.testing.assert_array_equal(sample.pointer_contact_masks[0], generation_sample.pointer_contact_masks[0])
            np.testing.assert_array_equal(sample.pointer_event_ids[0], generation_sample.pointer_event_ids[0])
            np.testing.assert_array_equal(sample.keycodes, generation_sample.keycodes)
            self.assertEqual(canonical_sample_sha256(sample), canonical_sample_sha256(generation_sample))

    def test_zero_flight_canonical_length_does_not_invent_midpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            event = keystroke_events(count=1)[0]
            # Make the second DOWN coincide exactly with the first UP.  The
            # canonical topology must keep adjacent contacts, not add a gap.
            for name, value in (
                ("flat_system_time_ms", 1000030),
                ("flat_event_time_ms", 30),
                ("flat_t_rel_ms", 30),
            ):
                event["flat_rows"][2][name] = value
            for name, value in (
                ("flat_system_time_ms", 1000070),
                ("flat_event_time_ms", 70),
                ("flat_t_rel_ms", 70),
            ):
                event["flat_rows"][3][name] = value
            second = event["key_rows"][1]
            second.update({
                "key_down_ms": 30,
                "key_up_ms": 70,
                "key_hold_ms": 40,
                "key_flight_from_previous_ms": 0,
                "key_touch_start_ms": 30,
                "key_touch_end_ms": 70,
            })
            event["duration_ms"] = 70
            path = write_archive(
                Path(directory) / "hmog_trajectory_keystroke.npz",
                "keystroke",
                [event],
            )
            corpus = NumericTrajectoryCorpus(path, self.splits, expected_action="keystroke")
            sample = corpus.canonical_sample(0)
            self.assertEqual(corpus.event_length(0), 4)
            self.assertEqual(corpus.canonical_max_points(0), 4)
            self.assertEqual(sample.pointer_features[0].shape[0], 4)
            self.assertFalse(np.any(~sample.pointer_contact_masks[0]))

    def test_ascii_keycode_flag_is_independently_audited(self):
        source = Path(
            "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713/"
            "results/smoke_fallback_user100669/hmog_trajectory_keystroke.npz"
        )
        if not source.is_file():
            self.skipTest("v1.1 keystroke smoke is unavailable")
        with np.load(str(source), allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        positions = np.flatnonzero(arrays["keycode"] == 97)
        self.assertGreater(positions.size, 0)
        key_index = int(positions[0])
        self.assertEqual(int(arrays["key_is_letter"][key_index]), 1)
        # Keep the old aggregate invariant internally self-consistent so only
        # the independent ASCII codebook gate can detect the corruption.
        arrays["key_is_letter"][key_index] = 0
        event_index = int(np.searchsorted(arrays["event_key_offsets"], key_index, side="right") - 1)
        arrays["n_letters"][event_index] -= 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hmog_trajectory_keystroke.npz"
            np.savez_compressed(str(path), **arrays)
            with self.assertRaisesRegex(ValueError, "ASCII keycode codebook"):
                NumericTrajectoryCorpus(path, self.splits, expected_action="keystroke")
            from scripts.audit_extractor_v11 import InvariantError, audit_action_npz
            with self.assertRaisesRegex(InvariantError, "ASCII keycode codebook"):
                audit_action_npz(path, "keystroke", expected_users=1)

    def test_extended_hmog_keycode_is_preserved_in_audited_model_vocabulary(self):
        """U+2026 occurs in the full HMOG archive and must not be dropped."""
        with tempfile.TemporaryDirectory() as directory:
            events = keystroke_events()
            for event in events:
                # Replace the second ASCII key in every synthetic event by the
                # observed HMOG ellipsis codepoint.  It is deliberately not a
                # letter, while its exact raw/token identity remains 8230.
                event["key_rows"][1]["keycode"] = 8230
                event["key_rows"][1]["key_is_letter"] = 0
                for row in event["flat_rows"]:
                    if int(row["flat_key_index"]) == 1:
                        row["flat_keycode"] = 8230
            path = write_archive(
                Path(directory) / "hmog_trajectory_keystroke.npz",
                "keystroke",
                events,
            )
            corpus = NumericTrajectoryCorpus(
                path, self.splits, expected_action="keystroke"
            )
            raw = corpus.raw_sample(0)
            self.assertEqual(raw["metadata"]["raw_keycodes"], [-5, 8230])
            self.assertEqual(
                [contact["keycode"] for contact in raw["contacts"]],
                [0, 8230],
            )
            sample = corpus.canonical_sample(0)
            np.testing.assert_array_equal(sample.keycodes, [0, 8230])
            self.assertEqual(sample.n_letters, 0)

    def test_training_and_generation_loaders_match_staggered_pinch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_archive(Path(directory) / "hmog_trajectory_pinch.npz", "pinch", staggered_pinch_events())
            corpus = NumericTrajectoryCorpus(path, self.splits, expected_action="pinch")
            training_sample = corpus.canonical_sample(0)
            from generation.corpus import load_action_corpus
            from generation.protocol import FixedUserSplit
            generation_split = FixedUserSplit.load(str(FORMAL_SPLIT_PATH), require_formal=True)
            generation_sample = load_action_corpus(
                str(path), "pinch", generation_split, user_ids=[0], strict=True
            )[0]
            for left, right in zip(training_sample.pointer_features, generation_sample.pointer_features):
                np.testing.assert_allclose(left, right)
            np.testing.assert_array_equal(training_sample.pointer_start_offset_ms, generation_sample.pointer_start_offset_ms)
            np.testing.assert_array_equal(training_sample.pointer_end_offset_ms, generation_sample.pointer_end_offset_ms)
            self.assertEqual(
                canonical_sample_sha256(training_sample),
                canonical_sample_sha256(generation_sample),
            )

    def test_pinch_pointer_slots_follow_first_appearance_not_numeric_id(self):
        with tempfile.TemporaryDirectory() as directory:
            events = staggered_pinch_events()
            for event in events:
                for row in event["flat_rows"]:
                    row["flat_pointer_id"] = 9 if row["flat_pointer_id"] == 0 else 2
            path = write_archive(
                Path(directory) / "hmog_trajectory_pinch.npz", "pinch", events
            )
            corpus = NumericTrajectoryCorpus(path, self.splits, expected_action="pinch")
            raw = corpus.raw_sample(0)
            self.assertEqual(raw["metadata"]["pointer_ids"], [9, 2])
            self.assertEqual(
                raw["metadata"]["pointer_start_offsets_ms"], [0.0, 20.0]
            )
            self.assertEqual(
                raw["metadata"]["pointer_end_offsets_ms"], [100.0, 80.0]
            )

    def test_fixed_registry_targets_exclude_refs_and_bucket_is_lossless(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_archive(Path(directory) / "hmog_trajectory_tap.npz", "tap", tap_events_for_all_users())
            corpus = NumericTrajectoryCorpus(path, self.splits, expected_action="tap")
            registry_a = ReferenceRegistry.build(corpus, seed=42)
            registry_b = ReferenceRegistry.build(corpus, seed=42)
            self.assertEqual(registry_a.sha256, registry_b.sha256)
            registry_path = Path(directory) / "reference_registry.json"
            registry_a.save(registry_path)
            from generation.protocol import ReferenceRegistry as GenerationReferenceRegistry
            generation_registry = GenerationReferenceRegistry.load(
                str(registry_path), self.splits.sha256
            )
            self.assertEqual(generation_registry.registry_sha256, registry_a.sha256)
            train = StrictFiveReferenceDataset(corpus, "train", registry_a, seed=42, cache_size=32)
            self.assertEqual(len(train), 70)  # 6 real - 5 fixed refs, for 70 train users
            first = train[0]
            first_user = first.target.user_id
            expected_ids = registry_a.sample_ids("train", first_user)
            self.assertEqual(tuple(ref.sample_id for ref in first.references), expected_ids)
            from generation.corpus import load_action_corpus
            from generation.protocol import FixedUserSplit
            generation_split = FixedUserSplit.load(str(FORMAL_SPLIT_PATH), require_formal=True)
            generation_pool = load_action_corpus(
                str(path), "tap", generation_split, user_ids=[first_user], strict=True
            )
            generation_refs = generation_registry.resolve(
                generation_pool, "tap", first_user, "train"
            )
            self.assertEqual(
                [canonical_sample_sha256(ref) for ref in first.references],
                [canonical_sample_sha256(ref) for ref in generation_refs],
            )
            same_user_positions = [i for i in range(len(train)) if train[i].target.user_id == first_user]
            self.assertTrue(same_user_positions)
            for position in same_user_positions:
                example = train[position]
                self.assertEqual(tuple(ref.sample_id for ref in example.references), expected_ids)
                self.assertNotIn(example.target.sample_id, expected_ids)
            sampler_a = DeterministicLengthBucketBatchSampler(train, batch_size=9, epoch=3, shuffle=True)
            sampler_b = DeterministicLengthBucketBatchSampler(train, batch_size=9, epoch=3, shuffle=True)
            batches_a = list(iter(sampler_a))
            batches_b = list(iter(sampler_b))
            self.assertEqual(batches_a, batches_b)
            flat = [x for batch in batches_a for x in batch]
            self.assertEqual(len(flat), len(train))
            self.assertEqual(set(flat), set(range(len(train))))
            example_batch = [train[position] for position in batches_a[0]]
            collated = StrictVariableLengthCollator()(example_batch)
            self.assertEqual(collated.features.shape[0], len(example_batch))
            for i, example in enumerate(example_batch):
                target_t, reference_t = train.padded_length_components(batches_a[0][i])
                shape_components = train.padded_shape_components(batches_a[0][i])
                self.assertEqual(shape_components[:2], (target_t, reference_t))
                self.assertEqual(
                    target_t,
                    max(int(values.shape[0]) for values in example.target.pointer_features),
                )
                self.assertEqual(
                    reference_t,
                    max(
                        int(values.shape[0])
                        for reference in example.references
                        for values in reference.pointer_features
                    ),
                )
                self.assertEqual(train.padded_length_key(batches_a[0][i]), max(target_t, reference_t))
                self.assertEqual(shape_components[2], max(1, int(example.target.n_keys)))
                self.assertEqual(
                    shape_components[3],
                    max(1, max(int(reference.n_keys) for reference in example.references)),
                )
                self.assertEqual(
                    int(collated.point_mask[i, 0].sum()),
                    int(example.target.pointer_features[0].shape[0]),
                )

    def test_registry_fail_closed_and_validation_milestones(self):
        with tempfile.TemporaryDirectory() as directory:
            events = tap_events_for_all_users()
            events.pop()  # one test user now has only five events
            path = write_archive(Path(directory) / "hmog_trajectory_tap.npz", "tap", events)
            corpus = NumericTrajectoryCorpus(path, self.splits, expected_action="tap")
            with self.assertRaisesRegex(ValueError, "fail closed"):
                ReferenceRegistry.build(corpus, seed=42)
        self.assertEqual(validation_epochs(100), (20, 40, 60, 80, 100))
        self.assertEqual(validation_epochs(7), (2, 3, 5, 6, 7))
        formal = TrainingConfig(
            action="tap", corpus_npz="unused", split_json="unused", output_dir="unused"
        )
        formal.validate()
        self.assertLessEqual(formal.alpha_bar_final, 1e-3)
        unsafe = TrainingConfig(
            action="tap", corpus_npz="unused", split_json="unused", output_dir="unused",
            diffusion_steps=200,
        )
        with self.assertRaisesRegex(ValueError, "terminal"):
            unsafe.validate()

    def test_generator_is_bound_to_full_cuda_device_index(self):
        fake = mock.Mock()
        with mock.patch("training.engine.torch.Generator", return_value=fake) as constructor:
            result = make_seeded_generator(torch.device("cuda:1"), 1234)
        constructor.assert_called_once_with(device=torch.device("cuda:1"))
        fake.manual_seed.assert_called_once_with(1234)
        self.assertIs(result, fake)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_resume_restores_rng_tensors_even_if_loader_mapped_them_to_cuda(self):
        seed_everything(731)
        expected = _rng_state()
        mapped = dict(expected)
        mapped["torch_cpu"] = expected["torch_cpu"].to("cuda:0")
        mapped["torch_cuda"] = [
            value.to("cuda:0") for value in expected["torch_cuda"]
        ]
        _restore_rng_state(mapped)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected["torch_cpu"]))
        for actual, wanted in zip(torch.cuda.get_rng_state_all(), expected["torch_cuda"]):
            self.assertTrue(torch.equal(actual, wanted))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_amp_backoff_changes_only_scale_and_never_steps_optimizer(self):
        model = torch.nn.Linear(2, 1).to("cuda:0")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.cuda.amp.GradScaler(
            init_scale=1024.0, growth_interval=2, enabled=True
        )
        loss = model(torch.ones(3, 2, device="cuda:0")).square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        before_parameters = [value.detach().clone() for value in model.parameters()]
        before, after = _backoff_amp_scaler_without_optimizer_step(scaler)
        self.assertEqual(before, 1024.0)
        self.assertEqual(after, 512.0)
        self.assertEqual(scaler.get_scale(), 512.0)
        self.assertEqual(scaler.state_dict()["_growth_tracker"], 0)
        for actual, wanted in zip(model.parameters(), before_parameters):
            self.assertTrue(torch.equal(actual, wanted))

    def test_atomic_checkpoint_ema_best_and_exact_mid_epoch_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_archive(
                root / "hmog_trajectory_tap.npz", "tap", tap_events_for_all_users()
            )

            def config(output):
                return TrainingConfig(
                    action="tap", corpus_npz=str(source), split_json=str(FORMAL_SPLIT_PATH),
                    output_dir=str(output), epochs=1, batch_size=32, learning_rate=1e-3,
                    diffusion_steps=1000, base_channels=8, cond_dim=16, time_dim=8,
                    n_blocks=1, dropout=0.0, seed=42, num_workers=0, amp=False,
                    checkpoint_every_steps=1, reference_cache_size=32, device="cpu",
                )

            uninterrupted_dir = root / "uninterrupted"
            uninterrupted = train_action(config(uninterrupted_dir))
            self.assertEqual(uninterrupted["status"], "complete")
            expected_determinism = {
                "cublas_workspace_config": ":4096:8",
                "deterministic_algorithms_enabled": True,
                "deterministic_algorithms_warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            }
            self.assertEqual(
                uninterrupted["runtime_determinism"], expected_determinism
            )
            persisted_manifest = json.loads(
                (uninterrupted_dir / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                persisted_manifest["runtime_determinism"], expected_determinism
            )
            self.assertTrue((uninterrupted_dir / "last.pt").is_file())
            progress_receipt = json.loads((uninterrupted_dir / "training_progress.json").read_text())
            last_receipt = json.loads((uninterrupted_dir / "last_state.json").read_text())
            self.assertEqual(progress_receipt["phase"], "complete")
            self.assertEqual(progress_receipt["epoch_index"], 1)
            self.assertEqual(progress_receipt["run_instance_id"], last_receipt["run_instance_id"])
            self.assertEqual(
                last_receipt["checkpoint_sha256"],
                hashlib.sha256((uninterrupted_dir / "last.pt").read_bytes()).hexdigest(),
            )
            self.assertEqual(last_receipt["progress"]["epoch_index"], 1)
            best = json.loads((uninterrupted_dir / "best_manifest.json").read_text())
            self.assertEqual(len(best["history"]), 1)
            self.assertTrue(Path(best["best"]["path"]).is_file())
            self.assertFalse(best["test_used_for_selection"])
            self.assertEqual(best["checkpoint_role"], "validation_selected_best")
            self.assertEqual(best["inference_weights"], "ema.shadow")
            self.assertEqual(
                best["best"]["checkpoint_sha256"],
                hashlib.sha256(Path(best["best"]["path"]).read_bytes()).hexdigest(),
            )
            registry = json.loads((uninterrupted_dir / "reference_registry.json").read_text())
            self.assertTrue(registry["reference_samples_excluded_from_target_pool"])
            self.assertEqual(registry["references_per_group"], 5)
            from generation.sampler import load_model_checkpoint
            loaded_model, loaded_sha = load_model_checkpoint(
                best["best"]["path"], "tap", torch.device("cpu"), require_best_ema=True,
                expected_registry_sha256=registry["registry_sha256"],
                expected_split_sha256=self.splits.sha256,
            )
            self.assertEqual(loaded_model.diffusion_steps, 1000)
            self.assertEqual(len(loaded_sha), 64)

            resumed_dir = root / "resumed"
            from training import engine as engine_module
            original_save = engine_module.atomic_torch_save
            interrupted = {"raised": False}

            def save_then_interrupt(path, payload, overwrite=True):
                original_save(path, payload, overwrite=overwrite)
                progress = payload.get("progress", {})
                if (
                    Path(path).name == "last.pt"
                    and progress.get("epoch_index") == 0
                    and progress.get("next_batch_in_epoch") == 1
                    and not interrupted["raised"]
                ):
                    interrupted["raised"] = True
                    raise RuntimeError("simulated power loss")

            with mock.patch("training.engine.atomic_torch_save", side_effect=save_then_interrupt):
                with self.assertRaisesRegex(RuntimeError, "power loss"):
                    train_action(config(resumed_dir))
            mid = torch.load(str(resumed_dir / "last.pt"), map_location="cpu")
            self.assertEqual(mid["progress"]["next_batch_in_epoch"], 1)
            self.assertEqual(mid["progress"]["examples_seen_in_epoch"], 32)
            self.assertGreater(mid["progress"]["epoch_feature_count"], 0)
            self.assertEqual(mid["progress"]["epoch_new_batches"], 1)
            resumed = train_action(config(resumed_dir), resume=resumed_dir / "last.pt")
            self.assertEqual(resumed["status"], "complete")

            state_a = torch.load(str(uninterrupted_dir / "last.pt"), map_location="cpu")
            state_b = torch.load(str(resumed_dir / "last.pt"), map_location="cpu")
            self.assertEqual(state_b["runtime_determinism"], expected_determinism)
            self.assertEqual(state_a["progress"]["global_step"], state_b["progress"]["global_step"])
            for name in state_a["model"]:
                self.assertTrue(torch.equal(state_a["model"][name], state_b["model"][name]), name)
            for name in state_a["ema"]["shadow"]:
                self.assertTrue(torch.equal(state_a["ema"]["shadow"][name], state_b["ema"]["shadow"][name]), name)
            metrics_a = [json.loads(line) for line in (uninterrupted_dir / "metrics.jsonl").read_text().splitlines()]
            metrics_b = [json.loads(line) for line in (resumed_dir / "metrics.jsonl").read_text().splitlines()]
            for records in (metrics_a, metrics_b):
                for record in records:
                    record.pop("unix_time", None)
            self.assertEqual(metrics_a, metrics_b)
            protected = (
                resumed_dir / "last.pt", resumed_dir / "best_manifest.json",
                resumed_dir / "metrics.jsonl",
            )
            protected_before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in protected
            }
            poisoned = dict(state_b)
            poisoned["runtime_determinism"] = dict(expected_determinism)
            poisoned["runtime_determinism"]["cudnn_deterministic"] = False
            poisoned_path = root / "poisoned_runtime.pt"
            torch.save(poisoned, poisoned_path)
            with self.assertRaisesRegex(ValueError, "runtime determinism"):
                train_action(config(resumed_dir), resume=poisoned_path)
            self.assertEqual(protected_before, {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in protected
            })
            with self.assertRaises(FileExistsError):
                train_action(config(uninterrupted_dir))

    def test_validation_nonfinite_loss_or_feature_count_fails_immediately(self):
        class Dataset:
            def set_epoch(self, _epoch):
                pass

            def __len__(self):
                return 1

        class Batch:
            def __init__(self):
                self.features = torch.zeros((1, 6, 2), dtype=torch.float32)

            def to(self, _device):
                return self

        class Model:
            diffusion_steps = 2

            def __init__(self, loss, count):
                self.loss = loss
                self.count = count

            def eval(self):
                pass

            def train(self):
                pass

            def load_state_dict(self, _state, strict=True):
                pass

            def training_loss(self, _batch, timesteps=None, noise=None):
                return {
                    "loss": torch.tensor(self.loss),
                    "valid_feature_count": torch.tensor(self.count),
                }

        ema = mock.Mock()
        ema.copy_to.return_value = {}
        cases = ((float("nan"), 12.0, "validation loss"), (1.0, float("nan"), "feature count"))
        for loss, count, message in cases:
            with self.subTest(message=message), mock.patch(
                "training.engine.make_epoch_loader", return_value=[Batch()]
            ), mock.patch("training.engine.ExponentialMovingAverage.restore"):
                with self.assertRaisesRegex(FloatingPointError, message):
                    evaluate_full_validation(
                        Model(loss, count), ema, Dataset(), batch_size=1, num_workers=0,
                        device=torch.device("cpu"), amp_enabled=False, seed=1,
                        completed_epoch=1, total_epochs=1,
                    )

    def test_training_loss_feature_count_fails_before_optimizer_aggregation(self):
        good_loss = torch.tensor(1.0, requires_grad=True)
        for count in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(count=count), self.assertRaisesRegex(
                FloatingPointError, "training feature count"
            ):
                _validated_loss_result({
                    "loss": good_loss,
                    "valid_feature_count": torch.tensor(count),
                }, "training")
        loss, count = _validated_loss_result({
            "loss": good_loss,
            "valid_feature_count": torch.tensor(12.0),
        }, "training")
        self.assertIs(loss, good_loss)
        self.assertEqual(count, 12.0)

    def test_epoch_commit_faults_reconcile_without_duplicate_metrics_or_best_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_archive(
                root / "hmog_trajectory_tap.npz", "tap", tap_events_for_all_users()
            )

            def config(output):
                return TrainingConfig(
                    action="tap", corpus_npz=str(source), split_json=str(FORMAL_SPLIT_PATH),
                    output_dir=str(output), epochs=1, batch_size=128, learning_rate=1e-3,
                    diffusion_steps=1000, base_channels=8, cond_dim=16, time_dim=8,
                    n_blocks=1, dropout=0.0, seed=42, num_workers=0, amp=False,
                    checkpoint_every_steps=1, reference_cache_size=32, device="cpu",
                )

            from training import engine as engine_module
            baseline_dir = root / "baseline"
            train_action(config(baseline_dir))
            baseline_checkpoint = torch.load(str(baseline_dir / "last.pt"), map_location="cpu")
            baseline_metrics = [
                json.loads(line)
                for line in (baseline_dir / "metrics.jsonl").read_text().splitlines()
            ]
            for row in baseline_metrics:
                row.pop("unix_time", None)
            for phase in (
                "staged_checkpoint", "journal", "best_checkpoint", "best_manifest", "last_checkpoint",
                "train_metric", "validation_metric",
            ):
                with self.subTest(phase=phase):
                    output = root / phase
                    raised = {"value": False}

                    def inject(observed):
                        if observed == phase and not raised["value"]:
                            raised["value"] = True
                            raise RuntimeError("fault:%s" % phase)

                    with mock.patch("training.engine._fault_inject", side_effect=inject):
                        with self.assertRaisesRegex(RuntimeError, "fault:%s" % phase):
                            train_action(config(output))
                    if phase == "staged_checkpoint":
                        self.assertFalse((output / "epoch_commit.json").exists())
                        pending = list(output.glob(".epoch_*_next.pt.pending"))
                        self.assertEqual(len(pending), 1)
                        pre_recovery_last_sha = hashlib.sha256(
                            (output / "last.pt").read_bytes()
                        ).hexdigest()
                        clean_pending = torch.load(str(pending[0]), map_location="cpu")
                        poisoned_pending = dict(clean_pending)
                        poisoned_pending["runtime_determinism"] = dict(
                            clean_pending["runtime_determinism"]
                        )
                        poisoned_pending["runtime_determinism"][
                            "deterministic_algorithms_warn_only"
                        ] = True
                        torch.save(poisoned_pending, pending[0])
                        with self.assertRaisesRegex(ValueError, "runtime determinism"):
                            train_action(config(output), resume=output / "last.pt")
                        self.assertFalse((output / "epoch_commit.json").exists())
                        self.assertEqual(
                            hashlib.sha256((output / "last.pt").read_bytes()).hexdigest(),
                            pre_recovery_last_sha,
                        )
                        self.assertFalse((output / "best_manifest.json").exists())
                        self.assertFalse((output / "metrics.jsonl").exists())
                        torch.save(clean_pending, pending[0])
                    else:
                        self.assertTrue((output / "epoch_commit.json").is_file())
                    # ``last.pt`` may not yet have been promoted.  Naming its
                    # eventual path is intentional: recovery reconciles the
                    # journal before opening the resume checkpoint.
                    result = train_action(config(output), resume=output / "last.pt")
                    self.assertEqual(result["status"], "complete")
                    recovered_progress = json.loads(
                        (output / "training_progress.json").read_text()
                    )
                    self.assertEqual(recovered_progress["phase"], "complete")
                    self.assertGreater(recovered_progress["last_loss"], 0.0)
                    self.assertGreaterEqual(recovered_progress["grad_norm"], 0.0)
                    self.assertFalse((output / "epoch_commit.json").exists())
                    rows = [
                        json.loads(line)
                        for line in (output / "metrics.jsonl").read_text().splitlines()
                    ]
                    identities = [(row["type"], row["completed_epoch"]) for row in rows]
                    self.assertEqual(identities, [("train_epoch", 1), ("validation", 1)])
                    for row in rows:
                        row.pop("unix_time", None)
                    self.assertEqual(rows, baseline_metrics)
                    best = json.loads((output / "best_manifest.json").read_text())
                    self.assertEqual(len(best["history"]), 1)
                    self.assertTrue(Path(best["best"]["path"]).is_file())
                    self.assertEqual(
                        best["best"]["checkpoint_sha256"],
                        hashlib.sha256(Path(best["best"]["path"]).read_bytes()).hexdigest(),
                    )
                    self.assertEqual(len(list(output.glob("best_epoch_*.pt"))), 1)
                    checkpoint = torch.load(str(output / "last.pt"), map_location="cpu")
                    self.assertEqual(checkpoint["progress"]["epoch_index"], 1)
                    for name in baseline_checkpoint["model"]:
                        self.assertTrue(torch.equal(
                            baseline_checkpoint["model"][name], checkpoint["model"][name]
                        ), name)
                    for name in baseline_checkpoint["ema"]["shadow"]:
                        self.assertTrue(torch.equal(
                            baseline_checkpoint["ema"]["shadow"][name],
                            checkpoint["ema"]["shadow"][name],
                        ), name)


if __name__ == "__main__":
    unittest.main()
