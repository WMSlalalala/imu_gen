from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supplement.bundle import reassign_user_disjoint
from detectors.deep_pad import make_record


def _record(user_id: int, label: int, suffix: int):
    contact = np.asarray([[1, 1], [0, 0]], dtype=bool)
    values = np.zeros((2, 2, 4), dtype=np.float32)
    values[0, :, 0] = [float(user_id), float(user_id) + 1.0]
    values[0, :, 1] = [0.0, 1.0]
    values[0, :, 2:] = 0.5
    event_id = str(user_id * 100 + suffix)
    return make_record(
        action="tap",
        label=label,
        user_id=user_id,
        pool="train",
        sample_id=("real:" if label == 0 else "fake:") + event_id,
        event_group_id=event_id,
        pointer_continuous=values,
        global_t_ms=np.asarray([0.0, 10.0], dtype=np.float32),
        contact_mask=contact,
    )


class SupplementBundleTests(unittest.TestCase):
    def setUp(self):
        self.split = {
            "train": tuple(range(70)),
            "val": tuple(range(70, 80)),
            "test": tuple(range(80, 100)),
        }
        self.records = []
        self.references = set()
        for user_id in range(100):
            self.records.append(_record(user_id, 0, 1))
            self.records.append(_record(user_id, 0, 2))
            self.records.append(_record(user_id, 1, 3))
            self.references.add(str(user_id * 100 + 1))
        self.features = np.arange(len(self.records) * 3, dtype=np.float64).reshape(-1, 3)

    def test_same_user_owns_both_labels(self):
        rows, features, audit = reassign_user_disjoint(
            self.records,
            self.features,
            self.split,
            self.references,
            exclude_references=False,
            require_formal_fake_counts=False,
        )
        self.assertEqual(len(rows), 300)
        self.assertEqual(features.shape, (300, 3))
        self.assertEqual(audit["reference_rows_remaining"], 100)
        owner = {
            user: pool for pool, users in self.split.items() for user in users
        }
        self.assertTrue(all(row.pool == owner[row.user_id] for row in rows))
        for label in (0, 1):
            for pool, users in self.split.items():
                observed = {r.user_id for r in rows if r.label == label and r.pool == pool}
                self.assertEqual(observed, set(users))

    def test_reference_exclusion_preserves_non_reference_real(self):
        rows, features, audit = reassign_user_disjoint(
            self.records,
            self.features,
            self.split,
            self.references,
            exclude_references=True,
            require_formal_fake_counts=False,
        )
        self.assertEqual(len(rows), 200)
        self.assertEqual(features.shape, (200, 3))
        self.assertEqual(audit["reference_rows_dropped"], 100)
        self.assertEqual(audit["reference_rows_remaining"], 0)
        remaining_real = [row for row in rows if row.label == 0]
        self.assertEqual(len(remaining_real), 100)
        self.assertTrue(all(str(row.event_group_id).endswith("2") for row in remaining_real))

    def test_missing_reference_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "registry references are absent"):
            reassign_user_disjoint(
                self.records,
                self.features,
                self.split,
                self.references | {"999999999"},
                exclude_references=True,
                require_formal_fake_counts=False,
            )


if __name__ == "__main__":
    unittest.main()
