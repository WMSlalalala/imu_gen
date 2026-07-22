from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from estimator.fake_event_plan_archive import load_event_plans_from_archive
from estimator.fake_imu_pairs import build_fake_imu_unit, validate_fake_imu_unit


SMOKE_ROOT = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713/"
    "results/formal_eventplan_v15_launch_gates_20260721/v2_e2e_smoke/generated_archives"
)


class FakeEventPlanArchiveTest(unittest.TestCase):
    def test_all_five_action_smoke_archives_replay_exact_plans(self):
        paths = sorted(SMOKE_ROOT.glob("shards/shard_*_of_*/*/user_*.npz"))
        self.assertEqual(len(paths), 15)
        actions = set()
        identities = set()
        for path in paths:
            action = path.parent.name
            plans = load_event_plans_from_archive(path, expected_action=action)
            self.assertEqual(len(plans), 1)
            plan = plans[0]
            self.assertEqual(plan.to_imu_kwargs()["noise_seed"], plan.imu_noise_seed)
            actions.add(action)
            identities.add(plan.plan_sha256)
        self.assertEqual(actions, {"tap", "scroll", "swipe", "pinch", "keystroke"})
        self.assertEqual(len(identities), 15)

    def test_condition_digest_tampering_fails_closed(self):
        source = next(SMOKE_ROOT.glob("shards/shard_*_of_*/tap/user_*.npz"))
        with np.load(str(source), allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        arrays["condition_request_sha256"] = arrays["condition_request_sha256"].copy()
        arrays["condition_request_sha256"][0, 0] ^= np.uint8(1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.npz"
            np.savez_compressed(str(path), **arrays)
            with self.assertRaisesRegex(ValueError, "ConditionRequest digest"):
                load_event_plans_from_archive(path, expected_action="tap")

    def test_fake_imu_unit_preserves_exact_event_plan_identity(self):
        source = next(SMOKE_ROOT.glob("shards/shard_*_of_*/tap/user_*.npz"))

        class FakeOnlineService:
            def generate(self, action, **kwargs):
                rows = max(1, int(round(float(kwargs["duration_ms"]) / 10.0)))
                active = np.zeros((rows, 6), dtype=np.float32)
                active[:, 2] = 9.8
                return {
                    "action": action,
                    "hz": 100.0,
                    "active_imu": active,
                    "relative_timestamps_ns": np.arange(rows, dtype=np.int64) * 10_000_000,
                    "release_backend": "online_five_shot_diffusion",
                    "metadata": {
                        "user_id": kwargs["user_id"],
                        "orientation_id": kwargs["orientation_id"],
                        "noise_seed": kwargs["noise_seed"],
                        "logical_event_duration_ms": kwargs["duration_ms"],
                        "ref_count": 5,
                        "used_ref_indices": [11, 12, 13, 14, 15],
                        "sample_steps": kwargs.get("sample_steps", 240),
                    },
                }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fake_imu.npz"
            report = build_fake_imu_unit(
                trajectory_archive_path=source, output_path=output,
                service=FakeOnlineService(), expected_action="tap",
                samples_per_user=1, sample_steps=17,
            )
            self.assertEqual(report["rows"], 1)
            replay = validate_fake_imu_unit(
                output, trajectory_archive_path=source,
                expected_action="tap", expected_samples=1,
            )
            self.assertEqual(replay["sample_steps"], [17])
            with np.load(str(output), allow_pickle=False) as archive:
                self.assertEqual(archive["event_plan_sha256"].shape, (1,))
                self.assertEqual(archive["imu_reference_indices"].shape, (1, 5))


if __name__ == "__main__":
    unittest.main()
