import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from runtime_determinism import (
    EXPECTED_RUNTIME_DETERMINISM,
    STRICT_RUNTIME_DETERMINISM_SHA256,
    seed_everything,
)
from generation.android import (
    ACTION_DOWN, ACTION_POINTER_DOWN, ACTION_POINTER_UP, ACTION_UP,
    record_from_generated,
)
from generation.archive import atomic_save_npz, build_numeric_archive, validate_existing_unit
from generation.audit import audit_generated_unit
from generation.batching import build_sampling_batch
from generation.corpus import load_action_corpus, open_shared_corpus
from generation.event_plan import EventPlan, bind_explicit_event_conditions
from generation.protocol import (
    ACTIONS, ACTION_TO_ID, FixedUserSplit, GenerationUnit, ReferenceConditionPolicy,
    ReferenceRegistry, TrainGlobalPrior, build_generation_units,
    canonical_condition_request_digest, condition_request_set_sha256,
    choose_reference_sets, ddim_noise_seed,
)
from generation.pipeline import generate_unit, unit_output_path
from generation.pad_export import load_generated_action_tree
from generation.sampler import (
    checkpoint_sha256, load_model_checkpoint, sample_ddim_seeded_batch,
)
from trajectory.data import canonicalize_sample
from trajectory.model import TrajectoryDiffusion
from training.corpus import canonical_sample_sha256
from scripts import generate_five_shot_trajectories as generation_cli


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_DATA_ROOT = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713"
)
SPLIT_PATH = "/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json"


def synthetic(action, user_id, split, event_id, variant, orientation=0):
    duration = float(80 + variant * 7)
    base_x = float(100 + variant * 13 + (user_id % 3) * 4)
    base_y = float(200 + variant * 9 + (user_id % 5) * 3)
    common = {
        "action": action, "user_id": user_id, "split": split,
        "sample_id": str(event_id), "orientation_id": orientation,
        "is_real": True,
    }
    if action == "tap":
        t = np.linspace(0, duration, 4, dtype=np.float32)
        xy = np.stack([base_x + np.linspace(0, 1 + variant * .1, 4), base_y + np.linspace(0, .8, 4)], axis=-1)
        common.update(duration_ms=duration, pointers=[{"xy": xy, "timestamps_ms": t}])
    elif action in ("scroll", "swipe"):
        t = np.linspace(0, duration * (3 if action == "scroll" else 2), 7, dtype=np.float32)
        dx = 40 + variant * 3
        dy = (120 if action == "scroll" else 70) + variant * 5
        xy = np.stack([base_x + np.linspace(0, dx, 7), base_y + np.linspace(0, dy, 7)], axis=-1)
        common.update(duration_ms=float(t[-1]), pointers=[{"xy": xy, "timestamps_ms": t}])
    elif action == "pinch":
        total = duration * 4
        t0 = np.linspace(0, total - 10 - variant, 8, dtype=np.float32)
        t1 = np.linspace(10 + variant, total, 9, dtype=np.float32)
        center = np.asarray([base_x + 250, base_y + 250], np.float32)
        p0 = np.stack([
            np.linspace(center[0] - 25, center[0] - 70 - variant, 8),
            np.linspace(center[1], center[1] - 10, 8),
        ], axis=-1)
        p1 = np.stack([
            np.linspace(center[0] + 25, center[0] + 70 + variant, 9),
            np.linspace(center[1], center[1] + 10, 9),
        ], axis=-1)
        common.update(duration_ms=total, pointers=[
            {"xy": p0, "timestamps_ms": t0}, {"xy": p1, "timestamps_ms": t1},
        ])
    else:
        contacts = []
        cursor = 0.0
        keys = [97 + (variant + j) % 20 for j in range(3 + variant % 2)]
        for key_index, keycode in enumerate(keys):
            count = 2 + ((variant + key_index) % 3)
            hold = 45.0 + 3 * key_index + variant
            t = np.linspace(cursor, cursor + hold, count, dtype=np.float32)
            x = base_x + key_index * 35
            y = base_y + (key_index % 2) * 25
            xy = np.stack([x + np.linspace(0, 1, count), y + np.linspace(0, .5, count)], axis=-1)
            contacts.append({"xy": xy, "timestamps_ms": t, "keycode": keycode})
            cursor = float(t[-1] + 60 + variant)
        event_duration = float(contacts[-1]["timestamps_ms"][-1])
        common.update(duration_ms=event_duration, contacts=contacts, n_letters=len(keys))
    return canonicalize_sample(common)


def make_pool(action, target_user=0, target_split="train"):
    train = [synthetic(action, 0, "train", ACTION_TO_ID[action] * 100000 + 1000 + i, i) for i in range(8)]
    if target_user == 0 and target_split == "train":
        return train, train
    refs = [synthetic(action, target_user, target_split, ACTION_TO_ID[action] * 100000 + 9000 + i, i + 1) for i in range(7)]
    return train + refs, refs


class TestGenerationProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = FixedUserSplit.load(SPLIT_PATH, require_formal=True)

    def test_formal_plan_is_exactly_100k_and_shard_complete(self):
        units = build_generation_units(self.split, num_shards=2, shard_id=None, require_formal=True)
        self.assertEqual(len(units), 500)
        self.assertEqual(sum(x.samples for x in units), 100000)
        self.assertEqual({x.shard_id for x in units}, {0, 1})

    def test_formal_cli_rejects_non_protocol_generation_seed_before_io(self):
        argv = [
            "generate_five_shot_trajectories.py",
            "--corpus-dir", "/does/not/exist",
            "--output-dir", "/also/does/not/exist",
            "--confirm-formal-100k",
            "--seed", "20260714",
        ]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(
            ValueError, "formal generation fixes --seed=20260713"
        ):
            generation_cli.main()

    def test_formal_cli_rejects_non_protocol_batch_before_io(self):
        argv = [
            "generate_five_shot_trajectories.py",
            "--corpus-dir", "/does/not/exist",
            "--output-dir", "/also/does/not/exist",
            "--confirm-formal-100k",
            "--batch-size", "64",
        ]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(
            ValueError, "formal generation fixes --batch-size=32"
        ):
            generation_cli.main()

    def test_reference_set_fixed_for_all_200_and_registry_hash(self):
        pool, _ = make_pool("tap")
        sets = choose_reference_sets(pool, "tap", 0, "train", 200, 123)
        first = tuple(x.sample_id for x in sets[0])
        self.assertEqual(len(set(first)), 5)
        self.assertTrue(all(tuple(x.sample_id for x in row) == first for row in sets))
        registry = ReferenceRegistry.build(
            {("tap", 0, "train"): tuple(int(x) for x in first)}, self.split.source_sha256
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry.write(str(path))
            loaded = ReferenceRegistry.load(str(path), self.split.source_sha256)
            self.assertEqual(loaded.registry_sha256, registry.registry_sha256)
            self.assertEqual(tuple(x.sample_id for x in loaded.resolve(pool, "tap", 0, "train")), first)

    def test_train_only_shrinkage_not_carrier_copy_and_no_nonref_access(self):
        pool, test_refs = make_pool("scroll", target_user=6, target_split="test")
        prior = TrainGlobalPrior.fit("scroll", pool, self.split.train_users)
        self.assertEqual(set(prior.source_user_ids.tolist()), {0})
        refs = tuple(test_refs[:5])
        policy = ReferenceConditionPolicy(prior)
        requests = [policy.sample("scroll", 6, "test", i, 99, refs) for i in range(50)]
        self.assertTrue(all(x.reference_ids == requests[0].reference_ids for x in requests))
        self.assertGreater(len({round(x.duration_ms, 4) for x in requests}), 5)
        for request in requests:
            self.assertEqual(request.condition_source_code, 2)
            self.assertEqual(request.train_prior_digest, prior.digest)
            self.assertFalse(any(
                abs(request.duration_ms - ref.duration_ms) <= 1e-7
                and np.array_equal(request.start_xy, ref.start_xy)
                and np.array_equal(request.end_xy, ref.end_xy)
                for ref in refs
            ))

    def test_staggered_pinch_absolute_lifetimes(self):
        pool, refs = make_pool("pinch")
        prior = TrainGlobalPrior.fit("pinch", pool, self.split.train_users)
        request = ReferenceConditionPolicy(prior).sample("pinch", 0, "train", 0, 5, tuple(refs[:5]))
        self.assertAlmostEqual(float(np.min(request.pointer_start_offset_ms)), 0.0, places=4)
        self.assertAlmostEqual(float(np.max(request.pointer_end_offset_ms)), request.duration_ms, places=3)
        self.assertTrue(np.all(request.pointer_end_offset_ms[:2] > request.pointer_start_offset_ms[:2]))
        self.assertTrue(np.any(request.pointer_start_offset_ms[:2] > 0) or np.any(request.pointer_end_offset_ms[:2] < request.duration_ms))

    def test_orientation_is_selected_before_geometry_composition(self):
        records = []
        for orientation, shift in ((0, 0.0), (1, 800.0)):
            for variant in range(8):
                item = synthetic("tap", 0, "train", 700000 + orientation * 100 + variant, variant, orientation)
                item.start_xy[:, 0] += shift
                item.end_xy[:, 0] += shift
                records.append(item)
        prior = TrainGlobalPrior.fit("tap", records, self.split.train_users)
        # Four portrait refs and one landscape ref make orientation-late
        # mixing an adversarial counterexample: landscape geometry would be
        # pulled toward the portrait majority before clipping.
        refs = tuple(records[:4] + [records[8 + 4]])
        requests = [ReferenceConditionPolicy(prior).sample("tap", 0, "train", i, 701, refs) for i in range(200)]
        landscape = [row for row in requests if row.orientation_id == 1]
        portrait = [row for row in requests if row.orientation_id == 0]
        self.assertTrue(landscape and portrait)
        self.assertTrue(all(float(row.start_xy[0, 0]) > 700.0 for row in landscape))
        self.assertTrue(all(float(row.start_xy[0, 0]) < 400.0 for row in portrait))

    def test_keystroke_sequence_is_composed_not_one_of_five_refs(self):
        pool, refs_all = make_pool("keystroke")
        refs = tuple(refs_all[:5])
        prior = TrainGlobalPrior.fit("keystroke", pool, self.split.train_users)
        policy = ReferenceConditionPolicy(prior)
        requests = [policy.sample("keystroke", 0, "train", i, 811, refs) for i in range(200)]
        reference_sequences = {tuple(int(x) for x in ref.keycodes.tolist()) for ref in refs}
        generated_sequences = {tuple(int(x) for x in row.keycodes.tolist()) for row in requests}
        self.assertGreater(len(generated_sequences), 5)
        self.assertTrue(generated_sequences.isdisjoint(reference_sequences))
        for row in requests:
            self.assertEqual(row.n_keys, row.keycodes.size)
            self.assertEqual(row.n_letters, int(np.sum(
                ((row.keycodes >= 65) & (row.keycodes <= 90))
                | ((row.keycodes >= 97) & (row.keycodes <= 122))
            )))
            self.assertEqual(row.condition_source_code, 2)
        explicit = policy.sample(
            "keystroke", 0, "train", 500, 811, refs,
            explicit_keycodes=[97, 98, 32], explicit_n_letters=2,
        )
        np.testing.assert_array_equal(explicit.keycodes, [97, 98, 32])
        self.assertEqual(explicit.condition_source_code, 3)

    def test_keystroke_first_last_tokens_condition_endpoints_and_rare_fallback(self):
        pool, refs_all = make_pool("keystroke")
        refs = tuple(refs_all[:5])
        prior = TrainGlobalPrior.fit("keystroke", pool, self.split.train_users)
        policy = ReferenceConditionPolicy(prior)
        first = policy.sample(
            "keystroke", 0, "train", 901, 912, refs,
            explicit_keycodes=[97, 98], explicit_n_letters=2,
        )
        second = policy.sample(
            "keystroke", 0, "train", 901, 912, refs,
            explicit_keycodes=[101, 102], explicit_n_letters=2,
        )
        self.assertFalse(np.array_equal(first.start_xy[0], second.start_xy[0]))
        self.assertFalse(np.array_equal(first.end_xy[0], second.end_xy[0]))
        for row in (first, second):
            self.assertTrue(np.all(row.start_xy[0] >= row.screen_min_xy))
            self.assertTrue(np.all(row.end_xy[0] <= row.screen_max_xy))
        # Canonical token 0 represents every raw negative HMOG sentinel.  It
        # is absent from this synthetic train prior, so endpoint provenance
        # must explicitly record the same-orientation keyboard fallback.
        rare = policy.sample(
            "keystroke", 0, "train", 902, 912, refs,
            explicit_keycodes=[0, 0], explicit_n_letters=0,
        )
        np.testing.assert_array_equal(rare.key_endpoint_source_code, [3, 3])

    def test_external_event_plan_binds_time_xy_and_disjoint_modality_seeds(self):
        pool, refs_all = make_pool("scroll")
        refs = tuple(refs_all[:5])
        prior = TrainGlobalPrior.fit("scroll", pool, self.split.train_users)
        base = ReferenceConditionPolicy(prior).sample(
            "scroll", 0, "train", 17, 1234, refs,
            explicit_orientation_id=0,
        )
        duration = float(np.clip(210, np.min(prior.duration_ms), np.max(prior.duration_ms)))
        duration = float(round(duration))
        start = np.asarray([140.0, 240.0], np.float32)
        end = np.asarray([180.0, 330.0], np.float32)
        bound = bind_explicit_event_conditions(
            base, prior, refs,
            duration_ms=duration,
            start_xy=start,
            end_xy=end,
        )
        plan = EventPlan.from_condition_request(
            bound, sample_id="paired-scroll-000017", start_time_ns=9_000_000_000
        )
        self.assertEqual(plan.condition_source_code, 3)
        self.assertEqual(plan.duration_ms, duration)
        self.assertEqual(plan.start_xy[0], tuple(start.tolist()))
        self.assertEqual(plan.end_xy[0], tuple(end.tolist()))
        self.assertEqual(len(plan.plan_sha256), 64)
        self.assertEqual(plan.to_dict()["plan_sha256"], plan.plan_sha256)
        self.assertEqual(len({plan.condition_seed, plan.trajectory_noise_seed, plan.imu_noise_seed}), 3)
        imu = plan.to_imu_kwargs()
        self.assertEqual(imu["duration_ms"], duration)
        self.assertEqual(imu["xy_start"], tuple(start.tolist()))
        self.assertEqual(imu["start_time_ns"], 9_000_000_000)

    def test_external_event_plan_keystroke_text_is_one_shared_condition(self):
        pool, refs_all = make_pool("keystroke")
        refs = tuple(refs_all[:5])
        prior = TrainGlobalPrior.fit("keystroke", pool, self.split.train_users)
        text = "abc"
        base = ReferenceConditionPolicy(prior).sample(
            "keystroke", 0, "train", 19, 4321, refs,
            explicit_keycodes=[ord(ch) for ch in text],
            explicit_n_letters=3,
            explicit_orientation_id=0,
        )
        bound = bind_explicit_event_conditions(base, prior, refs)
        plan = EventPlan.from_condition_request(
            bound, sample_id="paired-key-000019", text=text
        )
        self.assertEqual(plan.keycodes, (97, 98, 99))
        self.assertEqual(plan.n_keys, 3)
        self.assertEqual(plan.n_letters, 3)
        imu = plan.to_imu_kwargs()
        self.assertEqual(imu["text"], text)
        self.assertEqual(imu["n_keys"], 3)

    def test_explicit_orientation_requires_five_shot_support(self):
        pool, refs_all = make_pool("tap")
        refs = tuple(refs_all[:5])
        prior = TrainGlobalPrior.fit("tap", pool, self.split.train_users)
        with self.assertRaisesRegex(ValueError, "explicit orientation"):
            ReferenceConditionPolicy(prior).sample(
                "tap", 0, "train", 20, 777, refs,
                explicit_orientation_id=1,
            )

    def test_real_one_user_numeric_corpus_reader(self):
        smoke = TRAJECTORY_DATA_ROOT / "results" / "smoke_one_user"
        for action in ACTIONS:
            records = load_action_corpus(
                str(smoke / ("hmog_trajectory_%s.npz" % action)), action,
                self.split, user_ids=[0], strict=True,
            )
            self.assertGreaterEqual(len(records), 5)
            self.assertTrue(all(x.user_id == 0 and x.split == "train" for x in records))
            authoritative = open_shared_corpus(
                str(smoke / ("hmog_trajectory_%s.npz" % action)), action, self.split
            )
            authoritative_by_id = {
                str(int(authoritative.event_ids[index])): authoritative.canonical_sample(index)
                for index in range(len(authoritative))
            }
            for generation_sample in records[:5]:
                self.assertEqual(
                    canonical_sample_sha256(generation_sample),
                    canonical_sample_sha256(authoritative_by_id[generation_sample.sample_id]),
                )
            if action == "pinch":
                self.assertTrue(any(
                    np.any(x.pointer_start_offset_ms[:2] > 0) or np.any(x.pointer_end_offset_ms[:2] < x.duration_ms)
                    for x in records
                ))

    def test_formal_checkpoint_loader_requires_best_ema_and_1000_schedule(self):
        model = TrajectoryDiffusion(
            "tap", diffusion_steps=1000, base_channels=8, cond_dim=16,
            time_dim=8, n_blocks=1, dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "tap_best.pt"
            schedule = {
                "terminal_gaussian_gate_passed": True,
                "alpha_bar_final": float(model.alpha_bar[-1]),
            }
            source = {
                "corpus_sha256": "1" * 64,
                "split_sha256": "2" * 64,
                "reference_registry_sha256": "3" * 64,
            }
            payload = {
                "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
                "runtime_determinism": dict(EXPECTED_RUNTIME_DETERMINISM),
                "checkpoint_role": "training_state_with_raw_model_and_ema",
                "inference_weights_for_validation_selected_best": "ema.shadow",
                "config": {"action": "tap"},
                "model_config": {
                    "action": "tap", "diffusion_steps": 1000, "base_channels": 8, "cond_dim": 16,
                    "time_dim": 8, "n_blocks": 1, "dropout": 0.0,
                },
                "diffusion_schedule": schedule,
                "source": source,
                "progress": {
                    "epoch_index": 20,
                    "global_step": 200,
                    "best_val_loss": 1.25,
                    "last_validation": {"completed_epoch": 20, "val_loss": 1.25},
                },
                "ema": {"decay": 0.999, "shadow": model.state_dict()},
            }
            torch.save(payload, good)
            best_entry = {
                "path": str(good),
                "completed_epoch": 20,
                "global_step": 200,
                "val_loss": 1.25,
                "source_sha256": source["corpus_sha256"],
                "split_sha256": source["split_sha256"],
                "reference_registry_sha256": source["reference_registry_sha256"],
                "checkpoint_sha256": checkpoint_sha256(str(good)),
                "checkpoint_role": "validation_selected_best",
                "inference_weights": "ema.shadow",
            }
            manifest = {
                "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
                "selection_split": "val",
                "selection_metric": "full_val_masked_epsilon_mse_ema",
                "lower_is_better": True,
                "test_used_for_selection": False,
                "checkpoint_role": "validation_selected_best",
                "inference_weights": "ema.shadow",
                "source": source,
                "diffusion_schedule": schedule,
                "best": best_entry,
                "history": [best_entry],
            }
            manifest_path = Path(directory) / "best_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            loaded, digest = load_model_checkpoint(str(good), "tap", torch.device("cpu"))
            self.assertEqual(loaded.diffusion_steps, 1000)
            self.assertEqual(len(digest), 64)

            for mutation, message in (
                (("best", "checkpoint_sha256", "0" * 64), "SHA-256"),
                (("best", "checkpoint_sha256", None), "fully bound"),
                (("protocol_version", None, "wrong"), "fully bound"),
                (("checkpoint_role", None, "wrong"), "fully bound"),
            ):
                broken = json.loads(json.dumps(manifest))
                parent, key, value = mutation
                if key is None:
                    broken[parent] = value
                else:
                    broken[parent][key] = value
                    broken["history"][-1] = dict(broken["best"])
                manifest_path.write_text(json.dumps(broken))
                with self.assertRaisesRegex(ValueError, message):
                    load_model_checkpoint(str(good), "tap", torch.device("cpu"))
            broken = json.loads(json.dumps(manifest))
            broken["history"][-1]["global_step"] += 1
            manifest_path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "fully bound"):
                load_model_checkpoint(str(good), "tap", torch.device("cpu"))
            broken = json.loads(json.dumps(manifest))
            broken["source"]["split_sha256"] = "9" * 64
            manifest_path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "source/schedule"):
                load_model_checkpoint(str(good), "tap", torch.device("cpu"))
            broken = json.loads(json.dumps(manifest))
            broken["best"]["completed_epoch"] = 19
            broken["history"][-1] = dict(broken["best"])
            manifest_path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "role/progress"):
                load_model_checkpoint(str(good), "tap", torch.device("cpu"))

            original_bytes = good.read_bytes()
            changed_payload = dict(payload)
            changed_payload["unbound_replacement_marker"] = True
            torch.save(changed_payload, good)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_model_checkpoint(str(good), "tap", torch.device("cpu"))
            good.write_bytes(original_bytes)

            wrong_role_payload = dict(payload)
            wrong_role_payload["checkpoint_role"] = "validation_selected_best"
            torch.save(wrong_role_payload, good)
            role_manifest = json.loads(json.dumps(manifest))
            role_manifest["best"]["checkpoint_sha256"] = checkpoint_sha256(str(good))
            role_manifest["history"][-1] = dict(role_manifest["best"])
            manifest_path.write_text(json.dumps(role_manifest))
            with self.assertRaisesRegex(ValueError, "role/progress"):
                load_model_checkpoint(str(good), "tap", torch.device("cpu"))
            good.write_bytes(original_bytes)

            smoke = Path(directory) / "tap_smoke.pt"
            smoke_payload = dict(payload)
            smoke_payload.update({
                "protocol_version": "trajectory_finalized_v2_e2e_smoke_v2",
                "checkpoint_role": "smoke_validation_best_ema",
                "inference_weights": "ema.shadow",
                "selection": {
                    "split": "val",
                    "metric": "fixed_smoke_val_masked_epsilon_mse_ema",
                    "best_step": 7,
                    "best_value": 1.5,
                    "test_used": False,
                },
            })
            torch.save(smoke_payload, smoke)
            smoke_entry = {
                "path": str(smoke), "step": 7, "val_loss": 1.5,
                "source_sha256": source["corpus_sha256"],
                "split_sha256": source["split_sha256"],
                "reference_registry_sha256": source["reference_registry_sha256"],
                "checkpoint_sha256": checkpoint_sha256(str(smoke)),
                "checkpoint_role": "validation_selected_best",
                "inference_weights": "ema.shadow",
            }
            manifest_path.write_text(json.dumps({
                "protocol_version": "trajectory_finalized_v2_e2e_smoke_v2",
                "selection_split": "val",
                "selection_metric": "fixed_smoke_val_masked_epsilon_mse_ema",
                "lower_is_better": True,
                "test_used_for_selection": False,
                "checkpoint_role": "validation_selected_best",
                "inference_weights": "ema.shadow",
                "source": source,
                "diffusion_schedule": schedule,
                "best": smoke_entry,
                "history": [smoke_entry],
            }))
            with self.assertRaisesRegex(ValueError, "fully bound"):
                load_model_checkpoint(str(smoke), "tap", torch.device("cpu"))
            smoke_loaded, _ = load_model_checkpoint(
                str(smoke), "tap", torch.device("cpu"),
                allow_e2e_smoke_checkpoint=True,
            )
            self.assertEqual(smoke_loaded.diffusion_steps, 1000)
            bad = Path(directory) / "tap_last.pt"
            torch.save({
                "action": "tap", "model_config": {"diffusion_steps": 1000},
                "model_state_dict": {},
            }, bad)
            with self.assertRaises(ValueError):
                load_model_checkpoint(str(bad), "tap", torch.device("cpu"))


class TestNeuralGenerationSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(20260713)
        cls.split = FixedUserSplit.load(SPLIT_PATH, require_formal=True)

    def _run_action(self, action):
        pool, refs_all = make_pool(action)
        refs = tuple(refs_all[:5])
        prior = TrainGlobalPrior.fit(action, pool, self.split.train_users)
        policy = ReferenceConditionPolicy(prior)
        request = policy.sample(action, 0, "train", 0, 2026, refs)
        batch = build_sampling_batch([request], [refs], torch.device("cpu"))
        torch.manual_seed(7)
        model = TrajectoryDiffusion(
            action, diffusion_steps=50, base_channels=8, cond_dim=16,
            time_dim=8, n_blocks=1, dropout=0.0,
        ).eval()
        calls = {"n": 0}
        handle = model.denoiser.register_forward_hook(lambda *args: calls.__setitem__("n", calls["n"] + 1))
        noise_seed = ddim_noise_seed(request.seed, action, 0, request.sample_index)
        try:
            output = sample_ddim_seeded_batch(model, batch, [noise_seed], inference_steps=50)
        finally:
            handle.remove()
        self.assertEqual(calls["n"], 50)
        record = record_from_generated(output, batch, 0, request)
        self.assertTrue(np.all(record.trajectory_t_ms == np.rint(record.trajectory_t_ms)))
        self.assertTrue(np.all(record.android_t_ms == np.rint(record.android_t_ms)))
        for pointer_id, length in enumerate(request.lengths):
            if not length:
                continue
            left, right = record.trajectory_pointer_offsets[pointer_id:pointer_id + 2]
            timeline = record.trajectory_t_ms[int(left):int(right)]
            self.assertTrue(np.all(np.diff(timeline) >= 1.0))
            self.assertEqual(float(timeline[0]), float(request.pointer_start_offset_ms[pointer_id]))
            self.assertEqual(float(timeline[-1]), float(request.pointer_end_offset_ms[pointer_id]))
        registry = ReferenceRegistry.build(
            {(action, 0, "train"): request.reference_ids}, self.split.source_sha256
        )
        arrays = build_numeric_archive(
            [record], 50, 0.0, 50, float(model.alpha_bar[-1]),
            "0" * 64, self.split.source_sha256, registry.registry_sha256,
            STRICT_RUNTIME_DETERMINISM_SHA256, 2026, 1, [noise_seed],
        )
        with self.assertRaisesRegex(ValueError, "DDIM sampler seeds"):
            build_numeric_archive(
                [record], 50, 0.0, 50, float(model.alpha_bar[-1]),
                "0" * 64, self.split.source_sha256,
                registry.registry_sha256, STRICT_RUNTIME_DETERMINISM_SHA256,
                2026, 1, [noise_seed ^ 1],
            )
        for invalid_eta in (0.5, float("nan")):
            with self.assertRaisesRegex(ValueError, "eta=0"):
                build_numeric_archive(
                    [record], 50, invalid_eta, 50, float(model.alpha_bar[-1]),
                    "0" * 64, self.split.source_sha256,
                    registry.registry_sha256, STRICT_RUNTIME_DETERMINISM_SHA256,
                    2026, 1, [noise_seed],
                )
        self.assertEqual(tuple(arrays["schema_version"].tolist()), (1, 5))
        self.assertEqual(int(arrays["generation_base_seed_scalar"]), 2026)
        self.assertEqual(int(arrays["generation_batch_size_scalar"]), 1)
        self.assertEqual(
            int(arrays["ddim_noise_seed"][0]),
            ddim_noise_seed(request.seed, action, 0, request.sample_index),
        )
        np.testing.assert_array_equal(
            arrays["condition_request_sha256"][0],
            np.frombuffer(canonical_condition_request_digest(request), dtype=np.uint8),
        )
        expected_plan = EventPlan.from_condition_request(
            request, sample_id=str(request.fake_id), start_time_ns=None,
            text=None,
        )
        np.testing.assert_array_equal(
            arrays["event_plan_sha256"][0],
            np.frombuffer(bytes.fromhex(expected_plan.plan_sha256), dtype=np.uint8),
        )
        self.assertTrue(all(value.dtype.kind in "biufc" for value in arrays.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory) / "shards" / "shard_000_of_001" / action
                / "user_000.npz"
            )
            atomic_save_npz(str(path), arrays)
            report = audit_generated_unit(
                str(path), pool, self.split, registry, prior, expected_count=1,
                expected_base_seed=2026,
                expected_generation_batch_size=1,
                max_aggregate_clip_rate=1.0, max_event_clip_rate=1.0,
                max_alpha_bar_final=1.0,
            )
            self.assertTrue(report["passed"])
            self.assertTrue(validate_existing_unit(
                str(path), ACTION_TO_ID[action], 0, 1, 50,
                generation_base_seed=2026,
                generation_batch_size=1,
                runtime_determinism_sha256=STRICT_RUNTIME_DETERMINISM_SHA256,
            ))
            with np.load(str(path), allow_pickle=False) as archive:
                nonzero_eta = {
                    name: archive[name].copy() for name in archive.files
                }
            nonzero_eta["ddim_eta_scalar"] = np.asarray(0.5, np.float32)
            path.unlink()
            atomic_save_npz(str(path), nonzero_eta)
            with self.assertRaisesRegex(ValueError, "eta=0"):
                validate_existing_unit(
                    str(path), ACTION_TO_ID[action], 0, 1, 50,
                    generation_base_seed=2026,
                    generation_batch_size=1,
                    runtime_determinism_sha256=STRICT_RUNTIME_DETERMINISM_SHA256,
                )
            with self.assertRaisesRegex(ValueError, "eta=0"):
                audit_generated_unit(
                    str(path), pool, self.split, registry, prior,
                    expected_count=1, expected_base_seed=2026,
                    expected_generation_batch_size=1,
                    max_aggregate_clip_rate=1.0, max_event_clip_rate=1.0,
                    max_alpha_bar_final=1.0,
                )
            with self.assertRaisesRegex(ValueError, "eta=0"):
                load_generated_action_tree(
                    Path(directory), action, self.split, require_formal=False,
                )
        if action == "pinch":
            masked = record.android_action & 0xFF
            self.assertEqual(int(np.sum(masked == ACTION_DOWN)), 1)
            self.assertEqual(int(np.sum(masked == ACTION_POINTER_DOWN)), 1)
            self.assertEqual(int(np.sum(masked == ACTION_POINTER_UP)), 1)
            self.assertEqual(int(np.sum(masked == ACTION_UP)), 1)
        if action == "keystroke":
            self.assertEqual(len(set(record.android_key_index.tolist()) - {-1}), request.n_keys)

    def test_ddim_valid_output_is_invariant_to_longer_cobatched_padding(self):
        """A request must not change when another request increases padded T."""

        action = "keystroke"
        pool, refs_all = make_pool(action)
        refs = tuple(refs_all[:5])
        prior = TrainGlobalPrior.fit(action, pool, self.split.train_users)
        policy = ReferenceConditionPolicy(prior)
        requests = [
            policy.sample(action, 0, "train", index, 20260713, refs)
            for index in range(30)
        ]
        shorter = longer = None
        for candidate in requests:
            for other in requests:
                if max(other.lengths) > max(candidate.lengths):
                    shorter, longer = candidate, other
                    break
            if shorter is not None:
                break
        self.assertIsNotNone(shorter)
        single = build_sampling_batch([shorter], [refs], torch.device("cpu"))
        paired = build_sampling_batch(
            [shorter, longer], [refs, refs], torch.device("cpu")
        )
        self.assertGreater(paired.features.shape[2], single.features.shape[2])

        torch.manual_seed(991)
        model = TrajectoryDiffusion(
            action, diffusion_steps=50, base_channels=8, cond_dim=16,
            time_dim=8, n_blocks=1, dropout=0.0,
        ).eval()
        seeds = [
            ddim_noise_seed(row.seed, action, 0, row.sample_index)
            for row in (shorter, longer)
        ]
        single_output = sample_ddim_seeded_batch(model, single, seeds[:1], 50)
        paired_output = sample_ddim_seeded_batch(model, paired, seeds, 50)
        valid = single.point_mask[0]
        paired_prefix = paired_output.features[0, :, :single.features.shape[2]]
        paired_t_prefix = paired_output.timestamps_ms[0, :, :single.features.shape[2]]
        # Batched floating kernels need not be bitwise identical, but padding
        # may not cause a physically meaningful change.  Before masked hidden
        # normalization this differed by 0.277 features / 5.4 ms.
        np.testing.assert_allclose(
            single_output.features[0][valid].numpy(),
            paired_prefix[valid].numpy(), rtol=0.0, atol=1.0e-5,
        )
        np.testing.assert_allclose(
            single_output.timestamps_ms[0][valid].numpy(),
            paired_t_prefix[valid].numpy(), rtol=0.0, atol=1.0e-4,
        )

    def test_reference_encoding_ignores_other_samples_reference_padding(self):
        def tap_ref(event_id, points):
            duration = float(70 + points)
            timestamps = np.linspace(0.0, duration, points, dtype=np.float32)
            xy = np.stack([
                np.linspace(100.0, 103.0, points),
                np.linspace(200.0, 202.0, points),
            ], axis=-1).astype(np.float32)
            return canonicalize_sample({
                "action": "tap", "user_id": 0, "split": "train",
                "sample_id": str(event_id), "orientation_id": 0,
                "is_real": True, "duration_ms": duration,
                "pointers": [{"xy": xy, "timestamps_ms": timestamps}],
            })

        short_refs = tuple(tap_ref(8_100_000 + index, 4 + index) for index in range(5))
        long_refs = tuple(tap_ref(8_200_000 + index, 30 + index) for index in range(5))
        pool = list(short_refs + long_refs)
        prior = TrainGlobalPrior.fit("tap", pool, self.split.train_users)
        policy = ReferenceConditionPolicy(prior)
        short_request = policy.sample("tap", 0, "train", 0, 713, short_refs)
        long_request = policy.sample("tap", 0, "train", 1, 713, long_refs)
        single = build_sampling_batch(
            [short_request], [short_refs], torch.device("cpu")
        )
        paired = build_sampling_batch(
            [short_request, long_request], [short_refs, long_refs],
            torch.device("cpu"),
        )
        self.assertGreater(paired.ref_features.shape[3], single.ref_features.shape[3])
        torch.manual_seed(117)
        model = TrajectoryDiffusion(
            "tap", diffusion_steps=50, base_channels=8, cond_dim=16,
            time_dim=8, n_blocks=1, dropout=0.0,
        ).eval()
        with torch.no_grad():
            encoded_single = model.denoiser.encode_condition(single)[0]
            encoded_paired = model.denoiser.encode_condition(paired)[0]
        np.testing.assert_allclose(
            encoded_single.numpy(), encoded_paired.numpy(), rtol=0.0, atol=1.0e-6
        )

    def test_all_five_actions_real_50_step_ddim_and_android_archive(self):
        for action in ACTIONS:
            with self.subTest(action=action):
                self._run_action(action)

    def test_atomic_unit_pipeline_and_resume(self):
        action = "tap"
        pool, refs_all = make_pool(action)
        refs = tuple(refs_all[:5])
        prior = TrainGlobalPrior.fit(action, pool, self.split.train_users)
        registry = ReferenceRegistry.build(
            {(action, 0, "train"): tuple(int(x.sample_id) for x in refs)}, self.split.source_sha256
        )
        unit = GenerationUnit(action, 0, "train", 2, 0, 1)
        model = TrajectoryDiffusion(
            action, diffusion_steps=50, base_channels=8, cond_dim=16,
            time_dim=8, n_blocks=1, dropout=0.0,
        ).eval()
        with tempfile.TemporaryDirectory() as directory:
            first = generate_unit(
                unit, pool, self.split, registry, prior, model, "0" * 64,
                directory, 77, 2, torch.device("cpu"),
                max_aggregate_clip_rate=1.0, max_event_clip_rate=1.0,
            )
            self.assertEqual(first["status"], "generated")
            second = generate_unit(
                unit, pool, self.split, registry, prior, model, "0" * 64,
                directory, 77, 2, torch.device("cpu"), resume=True,
                max_aggregate_clip_rate=1.0, max_event_clip_rate=1.0,
            )
            self.assertEqual(second["status"], "resumed")
            with self.assertRaisesRegex(ValueError, "protocol"):
                generate_unit(
                    unit, pool, self.split, registry, prior, model, "0" * 64,
                    directory, 77, 1, torch.device("cpu"), resume=True,
                    max_aggregate_clip_rate=1.0, max_event_clip_rate=1.0,
                )
            with self.assertRaisesRegex(ValueError, "protocol|base seed"):
                generate_unit(
                    unit, pool, self.split, registry, prior, model, "0" * 64,
                    directory, 78, 2, torch.device("cpu"), resume=True,
                    max_aggregate_clip_rate=1.0, max_event_clip_rate=1.0,
                )
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                generate_unit(
                    unit, pool, self.split, registry, prior, model, "1" * 64,
                    directory, 77, 2, torch.device("cpu"), resume=True,
                    max_aggregate_clip_rate=1.0, max_event_clip_rate=1.0,
                )
            archive_path = unit_output_path(directory, unit)
            with np.load(str(archive_path), allow_pickle=False) as archive:
                tampered = {name: archive[name].copy() for name in archive.files}
            tampered["runtime_determinism_sha256"][0] ^= np.uint8(1)
            archive_path.unlink()
            atomic_save_npz(str(archive_path), tampered)
            # The sidecar still claims the current runtime, but the immutable
            # NPZ digest is authoritative and must block resume.
            with self.assertRaisesRegex(ValueError, "protocol"):
                generate_unit(
                    unit, pool, self.split, registry, prior, model, "0" * 64,
                    directory, 77, 2, torch.device("cpu"), resume=True,
                    max_aggregate_clip_rate=1.0, max_event_clip_rate=1.0,
                )
            tampered["runtime_determinism_sha256"][0] ^= np.uint8(1)
            old_schema = {name: value.copy() for name, value in tampered.items()}
            old_schema["schema_version"] = np.asarray((1, 3), np.int16)
            old_schema.pop("runtime_determinism_sha256")
            archive_path.unlink()
            atomic_save_npz(str(archive_path), old_schema)
            with self.assertRaisesRegex(ValueError, "protocol"):
                generate_unit(
                    unit, pool, self.split, registry, prior, model, "0" * 64,
                    directory, 77, 2, torch.device("cpu"), resume=True,
                    max_aggregate_clip_rate=1.0, max_event_clip_rate=1.0,
                )
            tampered["ddim_noise_seed"][0] ^= np.int64(1)
            archive_path.unlink()
            atomic_save_npz(str(archive_path), tampered)
            with self.assertRaisesRegex(ValueError, "protocol"):
                generate_unit(
                    unit, pool, self.split, registry, prior, model, "0" * 64,
                    directory, 77, 2, torch.device("cpu"), resume=True,
                    max_aggregate_clip_rate=1.0, max_event_clip_rate=1.0,
                )
            tampered["ddim_noise_seed"][0] ^= np.int64(1)
            tampered["seed"][0] ^= np.int64(1)
            archive_path.unlink()
            atomic_save_npz(str(archive_path), tampered)
            with self.assertRaisesRegex(ValueError, "protocol"):
                generate_unit(
                    unit, pool, self.split, registry, prior, model, "0" * 64,
                    directory, 77, 2, torch.device("cpu"), resume=True,
                    max_aggregate_clip_rate=1.0, max_event_clip_rate=1.0,
                )
            tampered["seed"][0] ^= np.int64(1)
            tampered["condition_request_sha256"][0, 0] ^= np.uint8(1)
            archive_path.unlink()
            atomic_save_npz(str(archive_path), tampered)
            with self.assertRaisesRegex(ValueError, "ConditionRequest digest"):
                generate_unit(
                    unit, pool, self.split, registry, prior, model, "0" * 64,
                    directory, 77, 2, torch.device("cpu"), resume=True,
                    max_aggregate_clip_rate=1.0, max_event_clip_rate=1.0,
                )
            self.assertEqual(len(list(Path(directory).rglob("*.npz"))), 1)
            self.assertFalse(list(Path(directory).rglob("*.staging.*")))

    def test_condition_request_digest_and_set_are_order_independent_and_complete(self):
        pool, refs_all = make_pool("tap")
        refs = tuple(refs_all[:5])
        prior = TrainGlobalPrior.fit("tap", pool, self.split.train_users)
        policy = ReferenceConditionPolicy(prior)
        first = policy.sample("tap", 0, "train", 0, 77, refs)
        second = policy.sample("tap", 0, "train", 1, 77, refs)
        first_digest = canonical_condition_request_digest(first)
        second_digest = canonical_condition_request_digest(second)
        self.assertEqual(len(first_digest), 32)
        self.assertNotEqual(first_digest, second_digest)
        forward = condition_request_set_sha256([
            (first.fake_id, first_digest), (second.fake_id, second_digest),
        ])
        reverse = condition_request_set_sha256([
            (second.fake_id, second_digest), (first.fake_id, first_digest),
        ])
        self.assertEqual(forward, reverse)
        with self.assertRaisesRegex(ValueError, "duplicate fake_id"):
            condition_request_set_sha256([
                (first.fake_id, first_digest), (first.fake_id, second_digest),
            ])

    def test_generated_shard_to_pad_roundtrip_preserves_five_action_semantics(self):
        for action in ACTIONS:
            with self.subTest(action=action), tempfile.TemporaryDirectory() as directory:
                pool, refs_all = make_pool(action)
                refs = tuple(refs_all[:5])
                prior = TrainGlobalPrior.fit(action, pool, self.split.train_users)
                request = ReferenceConditionPolicy(prior).sample(action, 0, "train", 0, 303, refs)
                batch = build_sampling_batch([request], [refs], torch.device("cpu"))
                model = TrajectoryDiffusion(
                    action, diffusion_steps=50, base_channels=8, cond_dim=16,
                    time_dim=8, n_blocks=1, dropout=0.0,
                ).eval()
                noise_seed = ddim_noise_seed(
                    request.seed, action, 0, request.sample_index
                )
                output = sample_ddim_seeded_batch(
                    model, batch, [noise_seed], inference_steps=50
                )
                generated = record_from_generated(output, batch, 0, request)
                registry = ReferenceRegistry.build(
                    {(action, 0, "train"): request.reference_ids}, self.split.source_sha256
                )
                arrays = build_numeric_archive(
                    [generated], 50, 0.0, 50, float(model.alpha_bar[-1]),
                    "0" * 64, self.split.source_sha256,
                    registry.registry_sha256, STRICT_RUNTIME_DETERMINISM_SHA256,
                    303, 1, [noise_seed],
                )
                path = Path(directory) / "shards" / "shard_000_of_001" / action / "user_000.npz"
                atomic_save_npz(str(path), arrays)
                records, features = load_generated_action_tree(
                    Path(directory), action, self.split, require_formal=False
                )
                self.assertEqual(len(records), 1)
                self.assertTrue(np.all(np.isfinite(features)))
                converted = records[0]
                converted.validate()
                np.testing.assert_array_equal(np.unique(converted.global_t_ms), converted.global_t_ms)
                if action == "pinch":
                    self.assertTrue(np.any(converted.contact_mask[0] & converted.contact_mask[1]))
                    self.assertEqual(
                        float(converted.global_t_ms[np.flatnonzero(converted.contact_mask[0])[0]]),
                        float(request.pointer_start_offset_ms[0]),
                    )
                    self.assertEqual(
                        float(converted.global_t_ms[np.flatnonzero(converted.contact_mask[1])[-1]]),
                        float(request.pointer_end_offset_ms[1]),
                    )
                    # At least one union frame is a slot update from only one
                    # native pointer but a complete snapshot still contains
                    # the other pointer's forward-filled active state.
                    native_t0 = set(generated.android_t_ms[generated.android_pointer_id == 0].tolist())
                    native_t1 = set(generated.android_t_ms[generated.android_pointer_id == 1].tolist())
                    shared_active = converted.global_t_ms[converted.contact_mask[0] & converted.contact_mask[1]]
                    self.assertTrue(any((float(t) not in native_t0) or (float(t) not in native_t1) for t in shared_active))
                if action == "keystroke":
                    self.assertEqual(int(np.sum(converted.gap_mask)), request.n_keys - 1)
                    self.assertFalse(np.any(converted.contact_mask[:, converted.gap_mask]))
                    self.assertTrue(np.all(converted.keycode[converted.contact_mask] >= 0))
                    self.assertEqual(float(features[0, 0]), float(request.n_keys))
                    self.assertEqual(float(features[0, 1]), float(request.n_letters))


if __name__ == "__main__":
    unittest.main()
