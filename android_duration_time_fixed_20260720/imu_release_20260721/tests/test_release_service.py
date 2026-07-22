from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from imu_release import ACTIONS, IMUReleaseService  # noqa: E402


class ReleaseServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = IMUReleaseService(mode="cache", seed=20260721)

    def test_formal_health_gate(self):
        health = self.service.health()
        self.assertTrue(health["audit"]["verified"])
        self.assertEqual(health["audit"]["runtime_cache_files"], 138200)
        self.assertEqual(tuple(health["actions"]), ACTIONS)

    def test_five_action_cache_outputs(self):
        for action in ACTIONS:
            result = self.service.generate(action, user_id=6, noise_seed=100 + len(action))
            self.assertEqual(result["release_schema_version"], "audited_imu_release_result_v1")
            self.assertEqual(result["release_backend"], "audited_runtime_cache")
            self.assertEqual(result["active_imu"].dtype, np.float32)
            self.assertTrue(result["active_imu"].flags["C_CONTIGUOUS"])
            self.assertTrue(np.all(np.isfinite(result["active_imu"])))
            self.assertEqual(result["relative_timestamps_ns"].shape[0], result["active_imu"].shape[0])

    def test_selection_seed_is_call_order_independent(self):
        first = self.service.generate("tap", user_id=6, noise_seed=12345)
        self.service.generate("tap", user_id=6, noise_seed=99999)
        second = self.service.generate("tap", user_id=6, noise_seed=12345)
        self.assertEqual(first["path"], second["path"])
        self.assertTrue(np.array_equal(first["active_imu"], second["active_imu"]))

    def test_cache_rejects_unrepresentable_eventplan_condition(self):
        with self.assertRaisesRegex(ValueError, "use mode=online"):
            self.service.generate("keystroke", user_id=6, text="hello", n_keys=5)


if __name__ == "__main__":
    unittest.main()
