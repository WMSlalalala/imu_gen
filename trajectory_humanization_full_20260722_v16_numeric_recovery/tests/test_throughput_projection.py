#!/usr/bin/env python3
from __future__ import annotations

import unittest

from scripts.benchmark_training_throughput import (
    _project_epoch_seconds,
    _representative_profiles,
)
from orchestration.formal_supervisor import throughput_projection_matches


def profile(epoch: int, batch_index: int, work: int):
    return {
        "epoch": epoch,
        "batch_index": batch_index,
        "positions": (batch_index,),
        "batch_size": 1,
        "target_padded_t": work,
        "reference_padded_t": 1,
        "padded_work": work,
    }


class ThroughputProjectionTest(unittest.TestCase):
    def test_representatives_always_cover_expensive_tail(self):
        epochs = [
            [profile(epoch, index, 10 * (index + 1)) for index in range(20)]
            for epoch in range(5)
        ]
        selected = _representative_profiles(epochs, 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(max(row["padded_work"] for row in selected), 200)
        self.assertGreaterEqual(sum(row["padded_work"] >= 180 for row in selected), 3)

    def test_worst_measurement_changes_projected_full_epoch_time(self):
        epochs = [
            [profile(epoch, 0, 10), profile(epoch, 1, 100)]
            for epoch in range(5)
        ]
        ordinary_only = _project_epoch_seconds(
            epochs,
            [
                {"padded_work": 10, "elapsed_seconds": 1.0},
                {"padded_work": 100, "elapsed_seconds": 1.0},
            ],
        )
        with_worst = _project_epoch_seconds(
            epochs,
            [
                {"padded_work": 10, "elapsed_seconds": 1.0},
                {"padded_work": 100, "elapsed_seconds": 50.0},
            ],
        )
        self.assertEqual(ordinary_only["mean_epoch_optimizer_seconds"], 2.0)
        self.assertEqual(with_worst["mean_epoch_optimizer_seconds"], 51.0)
        self.assertFalse(with_worst["projection_has_extrapolation"])
        self.assertGreater(
            with_worst["mean_epoch_optimizer_seconds"],
            ordinary_only["mean_epoch_optimizer_seconds"],
        )

    def test_projection_producer_matches_supervisor_consumer(self):
        epochs = [
            [profile(epoch, 0, 10), profile(epoch, 1, 100)]
            for epoch in range(5)
        ]
        measurements = [
            {
                "label": "artificial_global_worst_case",
                "padded_work": 100,
                "elapsed_seconds": 50.0,
                "batch_size": 1,
                "target_padded_t": 10,
                "reference_padded_t": 10,
                "target_keycode_padded_k": 1,
                "reference_keycode_padded_k": 1,
            },
            {
                "label": "profile_000",
                "padded_work": 10,
                "elapsed_seconds": 1.0,
                "batch_size": 1,
                "target_padded_t": 2,
                "reference_padded_t": 1,
                "target_keycode_padded_k": 1,
                "reference_keycode_padded_k": 1,
            },
        ]
        projection = _project_epoch_seconds(epochs, measurements)
        mean_seconds = projection["mean_epoch_optimizer_seconds"]
        value = {
            "dataset_target_count": 2,
            "batch_size": 1,
            "profile_epoch_count": 5,
            "profile_epoch_batch_counts": [2] * 5,
            "profile_target_occurrences": 10,
            "profile_each_epoch_covers_dataset_once": True,
            "measured_optimizer_steps": 1,
            "projection_measurement_count": 2,
            "optimizer_state_initialization_steps": 2,
            "optimizer_state_initialization_excluded_from_projection": True,
            "shape_specific_warmup_optimizer_steps": 2,
            "shape_specific_warmup_excluded_from_projection": True,
            "total_unmeasured_optimizer_steps": 4,
            "projection_has_extrapolation": False,
            "worst_case_padded_t": 10,
            "worst_case_reference_padded_t": 10,
            "worst_case_keycode_padded_k": 1,
            "worst_case_reference_keycode_padded_k": 1,
            "worst_case_padded_work": 100,
            "worst_case_elapsed_seconds": 50.0,
            "projection_measurements": measurements,
            "epoch_projection": projection,
            "projected_full_epoch_optimizer_seconds": mean_seconds,
            "projected_full_epoch_examples_per_second": 2.0 / mean_seconds,
            "epoch_length_profile_sha256": "a" * 64,
        }
        self.assertTrue(throughput_projection_matches(value, 1, 2))


if __name__ == "__main__":
    unittest.main()
