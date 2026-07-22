from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from estimator.real_pair_index import REAL_PAIR_INDEX_SCHEMA, build_real_pair_index


def write_trajectory(path: Path, action: str, count: int = 20) -> None:
    starts = np.arange(count, dtype=np.int64) * 100 + 1000
    np.savez_compressed(
        str(path), action_name=np.asarray(action), event_id=np.arange(100, 100 + count),
        user_id=np.zeros(count, dtype=np.int64), activity_id=np.full(count, 9001),
        session_id=np.ones(count, dtype=np.int64), orientation_id=np.zeros(count, dtype=np.int64),
        label_start_ms=starts, label_end_ms=starts + 50,
        label_duration_ms=np.full(count, 50, dtype=np.int64),
    )


def write_imu(path: Path, action: str, count: int = 20) -> None:
    starts = np.arange(count, dtype=np.int64) * 100 + 1000
    repeats = 2 if action == "keystroke" else 1
    start_rows = np.repeat(starts, repeats)
    fields = dict(
        event_id=np.repeat(np.arange(500, 500 + count), repeats),
        user_id=np.zeros(count * repeats, dtype=np.int64),
        activity_id=np.full(count * repeats, 9001),
        session_id=np.ones(count * repeats, dtype=np.int64),
        orientation_id=np.zeros(count * repeats, dtype=np.int64),
        event_start_ms=start_rows, event_end_ms=start_rows + 50,
        event_duration_ms=np.full(count * repeats, 50, dtype=np.int64),
        active_len=np.full(count * repeats, 5, dtype=np.int64),
    )
    if action == "keystroke":
        fields["chunk_idx"] = np.tile(np.asarray([0, 1], dtype=np.int64), count)
    np.savez_compressed(str(path), **fields)


class RealPairIndexTest(unittest.TestCase):
    def run_build(self, action: str):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        trajectory = root / "trajectory.npz"
        imu = root / "imu.npz"
        output = root / "index.npz"
        write_trajectory(trajectory, action)
        write_imu(imu, action)
        report = build_real_pair_index(
            action=action, trajectory_path=trajectory, imu_path=imu,
            output_path=output, real_hash_seed=7,
        )
        return directory, output, report

    def test_absolute_time_identity_allows_event_id_mismatch(self):
        directory, output, report = self.run_build("tap")
        try:
            self.assertEqual(report["paired_events"], 20)
            self.assertEqual(report["event_id_match_count"], 0)
            self.assertEqual(report["event_id_mismatch_count"], 20)
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(str(data["schema_version"].item()), REAL_PAIR_INDEX_SCHEMA)
                self.assertEqual(len(data["sample_ids"]), 20)
                self.assertEqual(data["imu_row_offsets"][-1], 20)
                self.assertEqual(set(data["pools"].tolist()), {"train", "val", "test"})
        finally:
            directory.cleanup()

    def test_keystroke_preserves_complete_chunk_order(self):
        directory, output, report = self.run_build("keystroke")
        try:
            self.assertEqual(report["paired_events"], 20)
            with np.load(output, allow_pickle=False) as data:
                self.assertTrue(np.all(np.diff(data["imu_row_offsets"]) == 2))
                self.assertEqual(len(data["imu_rows"]), 40)
        finally:
            directory.cleanup()

    def test_incomplete_keystroke_chunk_sequence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / "trajectory.npz"
            imu = root / "imu.npz"
            write_trajectory(trajectory, "keystroke")
            write_imu(imu, "keystroke")
            with np.load(imu, allow_pickle=False) as source:
                arrays = {name: source[name] for name in source.files}
            arrays["chunk_idx"] = arrays["chunk_idx"].copy()
            arrays["chunk_idx"][1] = 2
            np.savez_compressed(str(imu), **arrays)
            with self.assertRaisesRegex(ValueError, "chunk sequence"):
                build_real_pair_index(
                    action="keystroke", trajectory_path=trajectory, imu_path=imu,
                    output_path=root / "index.npz",
                )


if __name__ == "__main__":
    unittest.main()
