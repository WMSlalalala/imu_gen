import unittest
from dataclasses import replace

from generation.protocol import FixedUserSplit, ReferenceRegistry
from scripts.build_trajectory_pad_bundle import _reference_overlap
from scripts.run_trajectory_benchmark import synthetic_five_action_dataset
from detectors.deep_pad import assign_strict_protocol_pools


class FormalBundleReferenceAuditTests(unittest.TestCase):
    def test_five_refs_per_user_are_reported_against_independent_real_event_pools(self):
        records, _ = synthetic_five_action_dataset(seed=991)
        base = next(row for row in records if row.action == "tap" and row.label == 0)
        real = []
        registry_entries = {}
        for user in range(100):
            event_ids = []
            for event in range(10):
                event_id = 1_000_000 + user * 100 + event
                event_ids.append(event_id)
                real.append(replace(
                    base, user_id=user, sample_id=str(event_id),
                    event_group_id=str(event_id),
                ))
            pool = "train" if user < 70 else ("val" if user < 80 else "test")
            registry_entries[("tap", user, pool)] = event_ids[:5]
        split_map = {
            "train": tuple(range(70)), "val": tuple(range(70, 80)),
            "test": tuple(range(80, 100)),
        }
        assigned, _ = assign_strict_protocol_pools(real, split_map, real_hash_seed=17)
        fixed = FixedUserSplit(
            train_users=split_map["train"], val_users=split_map["val"],
            test_users=split_map["test"], source_path="<test>", source_sha256="b" * 64,
        )
        registry = ReferenceRegistry.build(registry_entries, fixed.source_sha256)
        result = _reference_overlap(registry, "tap", assigned, fixed)
        self.assertEqual(result["n_reference_events"], 500)
        self.assertEqual(sum(result["detector_pool_totals"].values()), 500)
        self.assertIn("participates according to that pool", result["semantics"])


if __name__ == "__main__":
    unittest.main()
