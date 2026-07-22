#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
import torch

from trajectory.constraints import constrain_and_decode, pinch_endpoints_from_center
from trajectory.data import (
    FewShotExample,
    TrajectoryBatch,
    build_fewshot_examples,
    canonicalize_sample,
    collate_trajectories,
    collate_fewshot_trajectories,
    make_sampling_batch,
    validate_fewshot_references,
)
from trajectory.model import TrajectoryDiffusion


def swipe_raw(offset: float, sample_id: str, user_id: int = 7):
    n = 10
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x = 100.0 + offset + 600.0 * t
    y = 900.0 - 420.0 * t + (12.0 + 0.3 * offset) * np.sin(math.pi * t)
    return {
        "action": "swipe", "sample_id": sample_id, "user_id": user_id, "split": "train", "is_real": True,
        "orientation_id": 0, "duration_ms": 360.0 + offset,
        "pointers": [{
            "xy": np.stack([x, y], axis=-1), "timestamps_ms": np.linspace(0.0, 360.0 + offset, n),
            "pressure": 0.40 + 0.002 * offset + 0.08 * np.sin(math.pi * t),
            "size": np.full(n, 0.25 + 0.001 * offset, dtype=np.float32),
        }],
    }


def pinch_raw(offset: float, sample_id: str, user_id: int = 9):
    n = 6
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    center = np.stack([500.0 + 25.0 * t, 800.0 - 30.0 * t], axis=-1)
    half = (60.0 + offset + 100.0 * t)[:, None] * np.stack([np.cos(0.3 * t), np.sin(0.3 * t)], axis=-1)
    timestamps = np.linspace(0.0, 250.0 + offset, n)
    second_timestamps = np.linspace(20.0 + 0.1 * offset, 235.0 + offset, n)
    return {
        "action": "pinch", "sample_id": sample_id, "user_id": user_id, "split": "train", "is_real": True,
        "orientation_id": 3, "duration_ms": 250.0 + offset,
        "pointers": [
            {"xy": center - half, "timestamps_ms": timestamps},
            {"xy": center + half, "timestamps_ms": second_timestamps},
        ],
    }


def keystroke_raw(offset: float, sample_id: str, user_id: int = 11):
    contacts = []
    for key_index, keycode in enumerate((97, 98, 32, 99)):
        start = key_index * (110.0 + offset)
        xy0 = np.asarray([220.0 + key_index * 150.0, 1200.0 + (key_index % 2) * 35.0], dtype=np.float32)
        contacts.append({
            "keycode": keycode,
            "xy": np.stack([xy0, xy0 + [2.0 + offset * 0.1, -1.0], xy0 + [1.0, 1.0]], axis=0),
            "timestamps_ms": np.asarray([start, start + 15.0, start + 32.0 + offset], dtype=np.float32),
            "pressure": [0.35, 0.55 + offset * 0.005, 0.25],
            "size": [0.25, 0.30, 0.24],
        })
    return {
        "action": "keystroke", "sample_id": sample_id, "user_id": user_id, "split": "train", "is_real": True,
        "orientation_id": 0, "contacts": contacts, "n_letters": 3,
    }


def permute_refs(batch: TrajectoryBatch, order):
    values = dict(batch.__dict__)
    for name in (
        "ref_features", "ref_point_mask", "ref_contact_mask", "ref_event_ids",
        "ref_pointer_mask", "ref_pointer_start_offset_ms", "ref_pointer_end_offset_ms",
        "ref_mask", "ref_keycodes", "ref_keycode_mask",
    ):
        values[name] = getattr(batch, name)[:, order].clone()
    for name in ("ref_sample_ids", "ref_user_ids", "ref_splits"):
        rows = getattr(batch, name)
        values[name] = tuple(tuple(row[j] for j in order) for row in rows)
    result = TrajectoryBatch(**values)
    result.validate(require_references=True)
    return result


class NeuralTrajectorySmokeTest(unittest.TestCase):
    def test_zero_chord_tap_preserves_two_dimensional_jitter(self):
        raw = {
            "action": "tap",
            "sample_id": "9001",
            "user_id": 1,
            "split": "train",
            "is_real": True,
            "orientation_id": 0,
            "duration_ms": 30.0,
            "pointers": [{
                "xy": np.asarray(
                    [[100.0, 200.0], [103.0, 198.0], [97.0, 204.0], [100.0, 200.0]],
                    dtype=np.float32,
                ),
                "timestamps_ms": np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float32),
                "pressure": np.asarray([0.4, 0.5, 0.6, 0.4], dtype=np.float32),
                "size": np.asarray([0.2, 0.3, 0.25, 0.2], dtype=np.float32),
            }],
        }
        sample = canonicalize_sample(raw)
        batch = collate_trajectories([sample])
        output = constrain_and_decode(batch.features.clone(), batch)
        observed = output.xy[0, 0, :4].detach().cpu().numpy()
        np.testing.assert_allclose(
            observed, raw["pointers"][0]["xy"], rtol=0.0, atol=1e-6
        )

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_cpu_masked_loss_decreases_and_sample_is_constrained(self):
        pool = [canonicalize_sample(swipe_raw(float(i), "swipe_%02d" % i)) for i in range(14)]
        examples = build_fewshot_examples(pool[:4], pool, seed=8)
        batch = collate_fewshot_trajectories(examples)
        model = TrajectoryDiffusion(
            "swipe", diffusion_steps=8, base_channels=16, cond_dim=32, time_dim=16, n_blocks=2, dropout=0.0
        )
        fixed_t = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        fixed_noise = torch.randn(batch.features.shape, generator=torch.Generator().manual_seed(99))
        optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
        initial_result = model.training_loss(batch, fixed_t, fixed_noise)
        initial = float(initial_result["loss"].item())
        self.assertEqual(int(initial_result["valid_feature_count"].item()), 200)
        changed_padding = fixed_noise.clone()
        changed_padding[~batch.feature_mask] = 1e6
        self.assertTrue(torch.equal(initial_result["loss"], model.training_loss(batch, fixed_t, changed_padding)["loss"]))
        for _ in range(60):
            optimizer.zero_grad()
            loss = model.training_loss(batch, fixed_t, fixed_noise)["loss"]
            loss.backward()
            optimizer.step()
        final = float(model.training_loss(batch, fixed_t, fixed_noise)["loss"].item())
        self.assertLess(final, initial * 0.45, (initial, final))

        target = make_sampling_batch(
            action="swipe", lengths=[[9, 0], [7, 0]], duration_ms=[410.0, 270.0], orientation_id=[0, 1],
            start_xy=[[[50.0, 800.0], [0.0, 0.0]], [[200.0, 900.0], [0.0, 0.0]]],
            end_xy=[[[750.0, 300.0], [0.0, 0.0]], [[650.0, 500.0], [0.0, 0.0]]],
            reference_sets=[examples[0].references, examples[1].references],
            target_sample_ids=["request_a", "request_b"], user_ids=[7, 7], splits=["train", "train"],
        )
        output = model.sample(target, generator=torch.Generator().manual_seed(123))
        self.assertTrue(torch.equal(output.point_mask, target.point_mask))
        self.assertTrue(torch.all(output.features[~target.feature_mask] == 0))
        self.assertTrue(torch.all(output.xy[~target.contact_mask] == 0))
        for b in range(2):
            n = int(target.point_mask[b, 0].sum())
            self.assertTrue(torch.equal(output.xy[b, 0, 0], target.start_xy[b, 0]))
            self.assertTrue(torch.equal(output.xy[b, 0, n - 1], target.end_xy[b, 0]))
            times = output.timestamps_ms[b, 0, :n]
            self.assertAlmostEqual(float(times[-1]), float(target.duration_ms[b]), places=4)
            self.assertTrue(torch.all(times[1:] > times[:-1]))

        with mock.patch.object(model.denoiser, "forward", wraps=model.denoiser.forward) as denoiser_forward:
            ddim_output = model.sample_ddim(
                target, inference_steps=4, eta=0.0, generator=torch.Generator().manual_seed(321)
            )
        self.assertEqual(denoiser_forward.call_count, 4)
        self.assertEqual(model.ddim_timesteps(4).tolist()[0], 0)
        self.assertEqual(model.ddim_timesteps(4).tolist()[-1], model.diffusion_steps - 1)
        self.assertTrue(torch.equal(ddim_output.point_mask, target.point_mask))
        self.assertTrue(torch.all(ddim_output.features[~target.feature_mask] == 0))
        ddim_repeat = model.sample_ddim(
            target, inference_steps=4, eta=0.0, generator=torch.Generator().manual_seed(321)
        )
        self.assertTrue(torch.equal(ddim_output.features, ddim_repeat.features))

    def test_reference_set_invariance_effect_and_leakage_gates(self):
        pool = [canonicalize_sample(swipe_raw(float(i * 3), "style_%02d" % i)) for i in range(16)]
        target = pool[0]
        refs_a, refs_b = pool[1:6], pool[8:13]
        batch_a = collate_fewshot_trajectories([FewShotExample(target, refs_a)])
        batch_b = collate_fewshot_trajectories([FewShotExample(target, refs_b)])
        model = TrajectoryDiffusion(
            "swipe", diffusion_steps=4, base_channels=16, cond_dim=32, time_dim=16, n_blocks=1, dropout=0.0
        )
        model.eval()
        condition_a = model.denoiser.encode_condition(batch_a)
        permuted = permute_refs(batch_a, [4, 2, 0, 3, 1])
        condition_permuted = model.denoiser.encode_condition(permuted)
        self.assertTrue(torch.allclose(condition_a, condition_permuted, rtol=1e-6, atol=1e-6))
        condition_b = model.denoiser.encode_condition(batch_b)
        self.assertGreater(float(torch.max(torch.abs(condition_a - condition_b))), 1e-6)
        out_a = model.sample(batch_a, generator=torch.Generator().manual_seed(55)).features
        out_b = model.sample(batch_b, generator=torch.Generator().manual_seed(55)).features
        self.assertGreater(float(torch.max(torch.abs(out_a - out_b))), 1e-7)

        with self.assertRaisesRegex(ValueError, "missing refs"):
            validate_fewshot_references(target, refs_a[:4])
        with self.assertRaisesRegex(ValueError, "own ref"):
            validate_fewshot_references(target, [target] + refs_a[:4])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_fewshot_examples([target], pool + [pool[2]])
        with self.assertRaisesRegex(ValueError, "missing refs"):
            build_fewshot_examples([target], [target] + refs_a[:4])

    def test_two_pointer_pinch_endpoints_and_mask(self):
        pool = [canonicalize_sample(pinch_raw(float(i), "pinch_%02d" % i)) for i in range(8)]
        refs = pool[1:6]
        centers0, centers1 = torch.tensor([[500.0, 800.0]]), torch.tensor([[530.0, 760.0]])
        span, angle = torch.tensor([[120.0, 360.0]]), torch.tensor([[0.0, math.pi / 4.0]])
        start_xy, end_xy = pinch_endpoints_from_center(centers0, centers1, span, angle)
        batch = make_sampling_batch(
            action="pinch", lengths=[[6, 6]], duration_ms=[250.0], orientation_id=[3],
            start_xy=start_xy.tolist(), end_xy=end_xy.tolist(), pinch_span=span.tolist(), pinch_angle=angle.tolist(),
            pointer_start_offset_ms=[[0.0, 20.0]], pointer_end_offset_ms=[[250.0, 235.0]],
            reference_sets=[refs], target_sample_ids=["pinch_request"], user_ids=[9], splits=["train"],
        )
        model = TrajectoryDiffusion("pinch", diffusion_steps=4, base_channels=16, cond_dim=32, time_dim=16, n_blocks=1, dropout=0.0)
        output = model.sample(batch, generator=torch.Generator().manual_seed(2))
        for pointer_id in range(2):
            self.assertTrue(torch.equal(output.xy[0, pointer_id, 0], start_xy[0, pointer_id]))
            self.assertTrue(torch.equal(output.xy[0, pointer_id, 5], end_xy[0, pointer_id]))
            self.assertTrue(torch.all(output.timestamps_ms[0, pointer_id, 1:6] > output.timestamps_ms[0, pointer_id, :5]))
            self.assertAlmostEqual(
                float(output.timestamps_ms[0, pointer_id, 0]),
                float(batch.pointer_start_offset_ms[0, pointer_id]),
                places=5,
            )
            self.assertAlmostEqual(
                float(output.timestamps_ms[0, pointer_id, 5]),
                float(batch.pointer_end_offset_ms[0, pointer_id]),
                places=5,
            )
        self.assertGreater(float(output.timestamps_ms[0, 1, 0]), 0.0)
        self.assertLess(float(output.timestamps_ms[0, 1, 5]), float(output.duration_ms[0]))

    def test_keystroke_contacts_gaps_phases_and_keycode_condition(self):
        pool = [canonicalize_sample(keystroke_raw(float(i), "key_%02d" % i)) for i in range(8)]
        target = pool[0]
        batch_train = collate_fewshot_trajectories([FewShotExample(target, pool[1:6])])
        self.assertTrue(torch.any(batch_train.point_mask & ~batch_train.contact_mask))
        self.assertTrue(torch.all(batch_train.event_ids[~batch_train.contact_mask] == -1))

        contact_pattern = [True, True, False, True, True, False, True, True, False, True, True]
        event_pattern = [0, 0, -1, 1, 1, -1, 2, 2, -1, 3, 3]
        batch = make_sampling_batch(
            action="keystroke", lengths=[[11, 0]], duration_ms=[700.0], orientation_id=[0],
            start_xy=[[[200.0, 1200.0], [0.0, 0.0]]], end_xy=[[[800.0, 1200.0], [0.0, 0.0]]],
            n_keys=[4], n_letters=[3], keycodes=[[97, 98, 32, 99]],
            contact_masks=[[contact_pattern, [False] * 11]], event_ids=[[event_pattern, [-1] * 11]],
            reference_sets=[pool[1:6]], target_sample_ids=["key_request"], user_ids=[11], splits=["train"],
        )
        model = TrajectoryDiffusion("keystroke", diffusion_steps=4, base_channels=16, cond_dim=32, time_dim=16, n_blocks=1, dropout=0.0)
        self.assertTrue(torch.isfinite(model.training_loss(batch)["loss"]))
        self.assertEqual(int(batch.n_keys[0]), 4)
        self.assertEqual(int(batch.n_letters[0]), 3)
        output = model.sample(batch, generator=torch.Generator().manual_seed(4))
        self.assertTrue(torch.equal(output.contact_mask, batch.contact_mask))
        self.assertTrue(torch.all(output.xy[~batch.contact_mask] == 0))
        self.assertTrue(torch.all(output.contact_phase[~batch.contact_mask] == -1))
        expected_phases = torch.tensor([0, 2, -1, 0, 2, -1, 0, 2, -1, 0, 2])
        self.assertTrue(torch.equal(output.contact_phase[0, 0, :11], expected_phases))
        times = output.timestamps_ms[0, 0, :11]
        self.assertTrue(torch.all(times[1:] > times[:-1]))
        # Gap timing remains a supervised/generated feature; geometry does not.
        self.assertTrue(torch.all(batch.feature_mask[0, 0, [2, 5, 8], 2]))
        self.assertTrue(torch.all(~batch.feature_mask[0, 0, [2, 5, 8], :2]))

        invalid_events = [[event_pattern[:4] + [-1] + event_pattern[5:], [-1] * 11]]
        with self.assertRaises(ValueError):
            make_sampling_batch(
                action="keystroke", lengths=[[11, 0]], duration_ms=[700.0], orientation_id=[0],
                start_xy=[[[200.0, 1200.0], [0.0, 0.0]]], end_xy=[[[800.0, 1200.0], [0.0, 0.0]]],
                n_keys=[4], n_letters=[3], keycodes=[[97, 98, 32, 99]], contact_masks=[[contact_pattern, [False] * 11]],
                event_ids=invalid_events,
            )

        swapped_values = dict(batch.__dict__)
        swapped_values["keycodes"] = batch.keycodes.clone()
        swapped_values["keycodes"][:, 0], swapped_values["keycodes"][:, 1] = (
            batch.keycodes[:, 1].clone(), batch.keycodes[:, 0].clone()
        )
        swapped = TrajectoryBatch(**swapped_values)
        swapped.validate(require_references=True)
        model.eval()
        condition_original = model.denoiser.encode_condition(batch)
        condition_swapped = model.denoiser.encode_condition(swapped)
        self.assertGreater(float(torch.max(torch.abs(condition_original - condition_swapped))), 1e-7)
        x_t = torch.randn(batch.features.shape, generator=torch.Generator().manual_seed(88)) * batch.feature_mask
        timestep = torch.tensor([2], dtype=torch.long)
        predicted_original = model.denoiser(x_t, timestep, batch)
        predicted_swapped = model.denoiser(x_t, timestep, swapped)
        self.assertGreater(float(torch.max(torch.abs(predicted_original - predicted_swapped))), 1e-7)


if __name__ == "__main__":
    unittest.main()
