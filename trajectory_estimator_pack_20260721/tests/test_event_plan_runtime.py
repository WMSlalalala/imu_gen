from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PACK_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_ROOT = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713"
)
for path in (PACK_ROOT, TRAJECTORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generation.event_plan import EventPlan  # noqa: E402
from runtime.paired_layer import (  # noqa: E402
    audit_paired_generation,
    cross_modal_consistency_from_record,
    cross_modal_consistency_features,
)
from detectors.deep_pad import make_record  # noqa: E402


def tap_plan() -> EventPlan:
    return EventPlan(
        sample_id="paired-tap-smoke", action="tap", user_id=7, split="test",
        fake_id=8_000_000_000_700_001, sample_index=1,
        duration_ms=100.0, start_time_ns=1_000_000_000, orientation_id=0,
        start_xy=((100.0, 200.0), (0.0, 0.0)),
        end_xy=((101.0, 201.0), (0.0, 0.0)),
        pointer_start_offset_ms=(0.0, 0.0), pointer_end_offset_ms=(100.0, 0.0),
        lengths=(3, 0), n_keys=0, n_letters=0, keycodes=(), text=None,
        zero_flight_after_key=(), zero_flight_probability=0.0,
        contact_masks=((1, 1, 1), ()), event_ids=((0, 0, 0), ()),
        key_endpoint_source_code=(0, 0), condition_seed=1,
        trajectory_noise_seed=2, imu_noise_seed=3, condition_source_code=3,
        screen_min_xy=(0.0, 0.0), screen_max_xy=(1000.0, 2000.0),
        reference_ids=(1, 2, 3, 4, 5),
        reference_canonical_sha256=("1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64),
        carrier_ref_id=1, train_prior_digest="a" * 64,
    )


def paired_results(plan: EventPlan):
    imu_t = np.arange(10, dtype=np.int64) * 10_000_000
    imu = np.stack(
        [
            np.sin(np.arange(10) / 3.0), np.cos(np.arange(10) / 4.0),
            np.ones(10) * 9.8, np.arange(10) * 0.01,
            np.arange(10) * 0.02, np.arange(10) * 0.03,
        ],
        axis=1,
    ).astype(np.float32)
    imu_result = {
        "sample_id": plan.sample_id, "event_plan_sha256": plan.plan_sha256,
        "action": plan.action, "user_id": plan.user_id,
        "orientation_id": plan.orientation_id,
        "active_imu": imu, "relative_timestamps_ns": imu_t,
        "timestamps_ns": imu_t + plan.start_time_ns,
        "metadata": {
            "sample_id": plan.sample_id, "event_plan_sha256": plan.plan_sha256,
            "action": plan.action, "user_id": plan.user_id,
            "orientation_id": plan.orientation_id,
            "logical_event_duration_ms": plan.duration_ms,
        },
    }
    trajectory_t = np.asarray([0, 50, 100], np.int64) * 1_000_000
    trajectory_result = {
        "sample_id": plan.sample_id, "event_plan_sha256": plan.plan_sha256,
        "action": plan.action, "user_id": plan.user_id, "split": plan.split,
        "duration_ms": plan.duration_ms, "orientation_id": plan.orientation_id,
        "relative_timestamps_ns": trajectory_t,
        "timestamps_ns": trajectory_t + plan.start_time_ns,
        "x": np.asarray([100.0, 100.4, 101.0]),
        "y": np.asarray([200.0, 200.3, 201.0]),
        "frame_index": np.asarray([0, 1, 2]),
        "key_index": np.asarray([-1, -1, -1]),
        "metadata": {"event_plan_sha256": plan.plan_sha256},
    }
    return imu_result, trajectory_result


class TestEventPlanRuntime(unittest.TestCase):
    def test_plan_json_and_condition_roundtrip(self):
        plan = tap_plan()
        restored = EventPlan.from_dict(plan.to_dict())
        self.assertEqual(restored.plan_sha256, plan.plan_sha256)
        request = restored.to_condition_request()
        self.assertEqual(request.duration_ms, plan.duration_ms)
        np.testing.assert_array_equal(request.contact_masks[0], [1, 1, 1])

    def test_pair_audit_and_consistency_features(self):
        plan = tap_plan()
        imu, trajectory = paired_results(plan)
        audit = audit_paired_generation(plan, imu, trajectory)
        self.assertTrue(audit["passed"])
        names, values = cross_modal_consistency_features(plan, imu, trajectory)
        self.assertEqual(len(names), len(values))
        self.assertTrue(all(name.startswith("consistency__") for name in names))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_pair_audit_rejects_identity_mismatch(self):
        plan = tap_plan()
        imu, trajectory = paired_results(plan)
        trajectory["sample_id"] = "wrong-event"
        with self.assertRaisesRegex(ValueError, "sample_id mismatch"):
            audit_paired_generation(plan, imu, trajectory)

    def test_real_and_generated_record_consistency_path_is_shared(self):
        plan = tap_plan()
        imu, trajectory = paired_results(plan)
        old_names, old_values = cross_modal_consistency_features(plan, imu, trajectory)
        values = np.zeros((2, 3, 4), dtype=np.float32)
        values[0, :, 0] = trajectory["x"]
        values[0, :, 1] = trajectory["y"]
        contact = np.zeros((2, 3), dtype=bool)
        contact[0] = True
        record = make_record(
            action="tap", label=1, user_id=plan.user_id, pool="test",
            sample_id=plan.sample_id, pointer_continuous=values,
            global_t_ms=np.asarray([0.0, 50.0, 100.0]), contact_mask=contact,
        )
        names, observed = cross_modal_consistency_from_record(
            action="tap", logical_duration_ms=plan.duration_ms,
            active_imu=imu["active_imu"],
            imu_relative_timestamps_ms=imu["relative_timestamps_ns"] / 1.0e6,
            trajectory_record=record,
        )
        self.assertEqual(names, old_names)
        # Raw detector records intentionally freeze screen coordinates as
        # float32.  The legacy synthetic fallback above uses float64 arrays.
        np.testing.assert_allclose(observed, old_values, rtol=1e-5, atol=1e-5)

    def test_single_imu_frame_has_defined_zero_dynamics(self):
        values = np.zeros((2, 2, 4), dtype=np.float32)
        values[0, :, :2] = [[10.0, 20.0], [11.0, 21.0]]
        contact = np.zeros((2, 2), dtype=bool)
        contact[0] = True
        record = make_record(
            action="tap", label=0, user_id=1, pool="train", sample_id="one-frame",
            pointer_continuous=values, global_t_ms=np.asarray([0.0, 10.0]),
            contact_mask=contact,
        )
        names, observed = cross_modal_consistency_from_record(
            action="tap", logical_duration_ms=10.0,
            active_imu=np.asarray([[0.0, 0.0, 9.8, 0.1, 0.2, 0.3]], dtype=np.float32),
            imu_relative_timestamps_ms=np.asarray([0.0]), trajectory_record=record,
        )
        row = dict(zip(names, observed.tolist()))
        self.assertEqual(row["consistency__accel_delta_rms"], 0.0)
        self.assertEqual(row["consistency__touch_speed_accel_corr"], 0.0)
        self.assertEqual(row["consistency__touch_speed_gyro_corr"], 0.0)
        self.assertEqual(row["consistency__motion_touch_peak_delta_ms"], 0.0)
        self.assertTrue(np.all(np.isfinite(observed)))


if __name__ == "__main__":
    unittest.main()
