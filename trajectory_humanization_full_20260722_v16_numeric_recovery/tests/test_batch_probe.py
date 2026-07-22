import tempfile
import unittest
from pathlib import Path

from detectors.batch_probe import probe_max_safe_batch
from detectors.deep_pad import assign_strict_protocol_pools, load_fake_user_split
from tests.test_pair_runner import _small_action_bundle


class DeepBatchProbeTests(unittest.TestCase):
    def test_probe_uses_longest_full_event_and_records_no_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, split_path, rows = _small_action_bundle(root)
            split = load_fake_user_split(split_path)
            assigned, _ = assign_strict_protocol_pools(rows, split, real_hash_seed=33)
            output = root / "probe.json"
            result = probe_max_safe_batch(
                assigned, action="swipe", detector="tcn",
                model_params={"hidden_dim": 12, "n_blocks": 1, "dropout": 0.0},
                requested_batch_size=4, device="cpu", seed=77,
                dataset_file=dataset / "swipe.npz", fake_user_split=split_path,
                output_path=output,
            )
            expected = max(
                len(row.global_t_ms) for row in assigned
                if row.action == "swipe" and row.pool == "train"
            )
            self.assertEqual(result["selected_batch_size"], 4)
            self.assertEqual(result["longest_observed_train_event_length"], expected)
            self.assertFalse(result["truncation"])
            self.assertFalse(result["resampling"])
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
