"""End-to-end regression for HMOG same-ms inter-key boundaries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from runtime_determinism import STRICT_RUNTIME_DETERMINISM_SHA256
from generation.android import PHASE_DOWN, PHASE_UP, record_from_generated
from generation.archive import atomic_save_npz, build_numeric_archive
from generation.audit import audit_generated_unit
from generation.batching import build_sampling_batch
from generation.corpus import load_action_corpus, open_shared_corpus
from generation.protocol import (
    FixedUserSplit,
    ReferenceConditionPolicy,
    ReferenceRegistry,
    TrainGlobalPrior,
    _mean_contiguous_groups_bitwise,
    ddim_noise_seed,
)
from generation.sampler import sample_ddim_seeded_batch
from scripts.preflight_keystroke_conditions import _registry_for_full_corpus
from trajectory.data import canonicalize_sample, keystroke_zero_flight_flags
from trajectory.constraints import _deterministic_cumsum_1d
from trajectory.model import TrajectoryDiffusion
from training.fewshot_dataset import ReferenceRegistry as TrainingReferenceRegistry


SPLIT_PATH = "/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json"
TRAJECTORY_DATA_ROOT = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713"
)


def _same_ms_keystroke(event_id: int, variant: int):
    contacts = []
    cursor = 0.0
    for key_index in range(3 + variant % 2):
        count = 2 + (variant + key_index) % 3
        timestamps = np.linspace(
            cursor, cursor + 40.0 + variant + key_index, count, dtype=np.float32
        )
        x = 100.0 + 40.0 * key_index + variant
        xy = np.stack([
            x + np.linspace(0.0, 1.0, count),
            200.0 + key_index + np.linspace(0.0, 0.5, count),
        ], axis=-1).astype(np.float32)
        contacts.append({
            "xy": xy,
            "timestamps_ms": timestamps,
            "keycode": 97 + (variant + key_index) % 20,
        })
        # Deliberately no positive flight: next DOWN shares this UP's integer ms.
        cursor = float(timestamps[-1])
    return canonicalize_sample({
        "action": "keystroke",
        "user_id": 0,
        "split": "train",
        "sample_id": str(event_id),
        "orientation_id": 0,
        "is_real": True,
        "duration_ms": cursor,
        "contacts": contacts,
        "n_letters": len(contacts),
    })


class ZeroFlightGenerationTest(unittest.TestCase):
    def test_timing_projection_cumsum_is_exact_cpu_scan_with_source_contract(self):
        source_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        values = torch.tensor(
            [0.125, 1.5, 0.25, 2.0],
            dtype=torch.float64,
            device=source_device,
            requires_grad=True,
        )
        original_cumsum = torch.cumsum
        observed_devices = []

        def audited_cumsum(tensor, dim):
            observed_devices.append(tensor.device.type)
            return original_cumsum(tensor, dim=dim)

        from unittest import mock
        with mock.patch(
            "trajectory.constraints.torch.cumsum", side_effect=audited_cumsum
        ):
            result = _deterministic_cumsum_1d(values)

        self.assertEqual(observed_devices, ["cpu"])
        self.assertEqual(result.device, values.device)
        self.assertEqual(result.dtype, values.dtype)
        self.assertFalse(result.requires_grad)
        torch.testing.assert_close(
            result.cpu(),
            torch.tensor([0.125, 1.625, 1.875, 3.875], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        with self.assertRaisesRegex(ValueError, "1-D"):
            _deterministic_cumsum_1d(torch.ones((2, 2), dtype=torch.float32))

    def test_grouped_float32_means_are_bitwise_equal_for_multiple_lengths(self):
        rng = np.random.default_rng(20260713)
        lengths = rng.integers(2, 41, size=4096, dtype=np.int64)
        offsets = np.concatenate([
            np.zeros(1, np.int64), np.cumsum(lengths, dtype=np.int64)
        ])
        values = rng.normal(size=(int(offsets[-1]), 2)).astype(np.float32)
        expected = np.stack([
            np.mean(values[int(left):int(right)], axis=0)
            for left, right in zip(offsets[:-1], offsets[1:])
        ]).astype(np.float32)
        observed = _mean_contiguous_groups_bitwise(values, offsets)
        np.testing.assert_array_equal(observed, expected)

    def test_reference_seed_and_generation_seed_are_strictly_separate(self):
        split = FixedUserSplit.load(SPLIT_PATH, require_formal=True)
        path = str(
            TRAJECTORY_DATA_ROOT
            / "results/trajectories_full_v2/hmog_trajectory_tap.npz"
        )
        corpus = open_shared_corpus(path, "tap", split)
        # Changing only the generation seed cannot reach registry creation.
        registry_by_generation_seed = {
            generation_seed: _registry_for_full_corpus(corpus, 42).registry_sha256
            for generation_seed in (20260713, 20260714)
        }
        self.assertEqual(len(set(registry_by_generation_seed.values())), 1)
        registry_42 = _registry_for_full_corpus(corpus, 42)
        registry_43 = _registry_for_full_corpus(corpus, 43)
        self.assertNotEqual(registry_42.registry_sha256, registry_43.registry_sha256)
        training_registry = TrainingReferenceRegistry.build(corpus, seed=42)
        self.assertEqual(registry_42.registry_sha256, training_registry.sha256)

    def test_streaming_numeric_prior_is_digest_identical_to_materialized_records(self):
        split = FixedUserSplit.load(SPLIT_PATH, require_formal=True)
        path = str(
            TRAJECTORY_DATA_ROOT
            / "results/smoke_one_user/hmog_trajectory_keystroke.npz"
        )
        records = load_action_corpus(
            path, "keystroke", split, user_ids=[0], strict=True
        )
        materialized = TrainGlobalPrior.fit("keystroke", records, [0])
        numeric = open_shared_corpus(path, "keystroke", split)
        streaming = TrainGlobalPrior.fit("keystroke", numeric, [0])
        self.assertEqual(streaming.digest, materialized.digest)
        for name, expected in materialized.__dict__.items():
            observed = getattr(streaming, name)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(observed, expected, err_msg=name)

    def test_condition_neural_projection_android_and_archive_preserve_zero_flight(self):
        split = FixedUserSplit.load(SPLIT_PATH, require_formal=True)
        pool = [_same_ms_keystroke(4_900_000 + index, index) for index in range(8)]
        refs = tuple(pool[:5])
        prior = TrainGlobalPrior.fit("keystroke", pool, split.train_users)
        request = ReferenceConditionPolicy(prior).sample(
            "keystroke", 0, "train", 0, 713, refs
        )
        self.assertEqual(request.zero_flight_probability, 1.0)
        self.assertTrue(np.all(request.zero_flight_after_key))

        batch = build_sampling_batch([request], [refs], torch.device("cpu"))
        model = TrajectoryDiffusion(
            "keystroke", diffusion_steps=50, base_channels=8, cond_dim=16,
            time_dim=8, n_blocks=1, dropout=0.0,
        ).eval()
        noise_seed = ddim_noise_seed(
            request.seed, "keystroke", 0, request.sample_index
        )
        generated = sample_ddim_seeded_batch(
            model, batch, [noise_seed], inference_steps=50
        )
        record = record_from_generated(generated, batch, 0, request)

        left, right = record.trajectory_pointer_offsets[:2]
        timeline = record.trajectory_t_ms[int(left):int(right)]
        contact = record.trajectory_contact_mask[int(left):int(right)].astype(bool)
        events = record.trajectory_event_id[int(left):int(right)]
        np.testing.assert_array_equal(
            keystroke_zero_flight_flags(contact, events, request.n_keys),
            request.zero_flight_after_key,
        )
        for key_index in range(request.n_keys - 1):
            current = np.flatnonzero(events == key_index)
            following = np.flatnonzero(events == key_index + 1)
            self.assertEqual(timeline[current[-1]], timeline[following[0]])

            up = np.flatnonzero(
                (record.android_key_index == key_index)
                & (record.android_phase == PHASE_UP)
            )[-1]
            down = np.flatnonzero(
                (record.android_key_index == key_index + 1)
                & (record.android_phase == PHASE_DOWN)
            )[0]
            self.assertEqual(record.android_t_ms[up], record.android_t_ms[down])
            self.assertLess(up, down)
            self.assertNotEqual(record.android_frame_index[up], record.android_frame_index[down])

        registry = ReferenceRegistry.build(
            {("keystroke", 0, "train"): request.reference_ids}, split.source_sha256
        )
        arrays = build_numeric_archive(
            [record], 50, 0.0, 50, float(model.alpha_bar[-1]),
            "0" * 64, split.source_sha256, registry.registry_sha256,
            STRICT_RUNTIME_DETERMINISM_SHA256, 713, 1, [noise_seed],
        )
        np.testing.assert_array_equal(
            arrays["flat_zero_flight_after_key"],
            np.ones(request.n_keys - 1, np.uint8),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keystroke_zero_flight.npz"
            atomic_save_npz(str(path), arrays)
            report = audit_generated_unit(
                str(path), pool, split, registry, prior,
                expected_count=1, expected_base_seed=713,
                expected_generation_batch_size=1,
                max_aggregate_clip_rate=1.0,
                max_event_clip_rate=1.0, max_alpha_bar_final=1.0,
            )
        self.assertTrue(report["passed"])
        self.assertEqual(report["zero_flight_boundary_count"], request.n_keys - 1)
        self.assertEqual(report["positive_flight_boundary_count"], 0)


if __name__ == "__main__":
    unittest.main()
