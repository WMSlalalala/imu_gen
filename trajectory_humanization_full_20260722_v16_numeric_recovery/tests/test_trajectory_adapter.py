import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from detectors.deep_pad import (
    RawSequenceNormalizer,
    RawTCNPAD,
    RawTransformerPAD,
    collate_raw_sequences,
    make_record,
)
from detectors.trajectory_adapter import load_extracted_trajectory_npz
from generation.pad_export import (
    _feature_vector as generated_feature_vector,
    _single_pointer_frames as generated_single_pointer_frames,
)
from tests.synthetic_training_corpus import _flat_row, keystroke_events, write_archive
from trajectory.features import extract_keystroke_features


ROOT = Path(__file__).resolve().parents[1]
SMOKE = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713/"
    "results/smoke_one_user"
)


class ZeroFlightAdapterTests(unittest.TestCase):
    @staticmethod
    def _zero_flight_event():
        event = copy.deepcopy(keystroke_events(user_id=0, count=1)[0])
        event["duration_ms"] = 70
        event["flat_rows"] = [
            _flat_row(0, 0, 0, 100, 500, 0, 1, 0, -5),
            _flat_row(30, 0, 1, 101, 501, 1, 1, 0, -5),
            # A later key may receive a different raw Android tracking ID; it
            # is still the sole finger and maps to canonical pointer slot 0.
            _flat_row(30, 7, 0, 300, 500, 2, 1, 1, 97),
            _flat_row(70, 7, 1, 301, 501, 3, 1, 1, 97),
        ]
        event["key_rows"][0].update({
            "key_down_ms": 0, "key_up_ms": 30, "key_hold_ms": 30,
            "key_touch_start_ms": 0, "key_touch_end_ms": 30,
        })
        event["key_rows"][1].update({
            "key_down_ms": 30, "key_up_ms": 70, "key_hold_ms": 40,
            "key_flight_from_previous_ms": 0,
            "key_touch_start_ms": 30, "key_touch_end_ms": 70,
        })
        return event

    def _load_real(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_archive(
                Path(directory) / "keystroke.npz",
                "keystroke",
                [self._zero_flight_event()],
            )
            records, features = load_extracted_trajectory_npz(path, label=0)
        return records[0], features[0]

    @staticmethod
    def _generated_rows():
        return {
            "t": np.asarray([0, 30, 30, 70], np.float32),
            "x": np.asarray([100, 101, 300, 301], np.float32),
            "y": np.asarray([500, 501, 500, 501], np.float32),
            "pressure": np.full(4, 0.5, np.float32),
            "size": np.full(4, 0.2, np.float32),
            "pointer": np.zeros(4, np.int8),
            "phase": np.asarray([0, 2, 0, 2], np.int8),
            "action": np.asarray([0, 1, 0, 1], np.int16),
            "key_index": np.asarray([0, 0, 1, 1], np.int32),
            "keycode": np.asarray([-5, -5, 97, 97], np.int32),
            "frame": np.arange(4, dtype=np.int64),
        }

    def test_zero_flight_real_and_fake_keep_two_equal_time_tokens_without_gap(self):
        real, real_features = self._load_real()
        generated = generated_single_pointer_frames(self._generated_rows(), "keystroke")
        values, times, contact, active, codes, keycodes, events, gap = generated
        fake = make_record(
            action="keystroke", label=1, user_id=0, pool="train",
            sample_id="zero_flight_fake", pointer_continuous=values,
            global_t_ms=times, contact_mask=contact, active_mask=active,
            action_code=codes, keycode=keycodes, event_ids=events,
            gap_mask=gap,
        )
        np.testing.assert_array_equal(real.global_t_ms, [0.0, 30.0, 30.0, 70.0])
        np.testing.assert_array_equal(fake.global_t_ms, real.global_t_ms)
        np.testing.assert_array_equal(fake.gap_mask, real.gap_mask)
        np.testing.assert_array_equal(fake.event_ids, real.event_ids)
        np.testing.assert_array_equal(fake.keycode, real.keycode)
        self.assertFalse(np.any(real.gap_mask))
        self.assertEqual(int(real.event_ids[0, 1]), 0)
        self.assertEqual(int(real.event_ids[0, 2]), 1)

        fake_features = generated_feature_vector(
            "keystroke", values, times, contact, keycodes, events
        )
        expected_features = extract_keystroke_features(
            ["keycode_0", "a"],
            np.asarray([0.0, 30.0]),
            up_times_ms=np.asarray([30.0, 70.0]),
            key_points=np.asarray([[100.0, 500.0], [300.0, 500.0]]),
        )
        np.testing.assert_allclose(real_features, expected_features, rtol=0, atol=0)
        np.testing.assert_allclose(fake_features, expected_features, rtol=0, atol=0)
        # All flight summaries are exactly zero, including overlapFraction.
        np.testing.assert_array_equal(real_features[12:18], np.zeros(6))

    def test_intra_key_equal_time_is_rejected_but_deep_models_accept_zero_flight(self):
        real, _ = self._load_real()
        illegal_times = real.global_t_ms.copy()
        illegal_times[1] = illegal_times[0]
        with self.assertRaisesRegex(ValueError, "ordered key boundary"):
            replace(real, global_t_ms=illegal_times).validate()
        skipped_event_ids = real.event_ids.copy()
        skipped_event_ids[skipped_event_ids == 1] = 2
        with self.assertRaisesRegex(ValueError, "canonical contiguous"):
            replace(real, event_ids=skipped_event_ids).validate()

        normalizer = RawSequenceNormalizer().fit([real])
        batch = collate_raw_sequences([real], normalizer)
        self.assertEqual(int(batch.frame_mask.sum()), 4)
        self.assertEqual(float(batch.time_progress[0, 1]), float(batch.time_progress[0, 2]))
        self.assertTrue(torch.isfinite(batch.log_dt).all())
        for model in (
            RawTCNPAD(hidden_dim=12, n_blocks=1, dropout=0.0),
            RawTransformerPAD(
                hidden_dim=12, n_layers=1, n_heads=2,
                feedforward_dim=24, dropout=0.0,
            ),
        ):
            model.eval()
            with torch.no_grad():
                encoded = model.frame_encoder(batch)
                score = model(batch)
            self.assertEqual(tuple(encoded.shape[:2]), (1, 4))
            self.assertTrue(torch.isfinite(encoded).all())
            self.assertTrue(torch.isfinite(score).all())
            # The equal-time UP and DOWN remain separate ordered sequence slots.
            self.assertGreater(
                float(torch.max(torch.abs(encoded[0, 1] - encoded[0, 2]))),
                0.0,
            )


@unittest.skipUnless(SMOKE.exists(), "one-user extraction smoke is unavailable")
class AuditedFlatNPZAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = {}
        for action in ("tap", "scroll", "swipe", "pinch", "keystroke"):
            cls.loaded[action] = load_extracted_trajectory_npz(
                SMOKE / ("hmog_trajectory_%s.npz" % action),
                label=0,
                sample_prefix="real:",
            )

    def test_all_five_actions_convert_to_expected_cleanroom_feature_dimensions(self):
        expected = {"tap": 24, "scroll": 24, "swipe": 24, "pinch": 49, "keystroke": 34}
        for action, dimension in expected.items():
            records, features = self.loaded[action]
            self.assertGreater(len(records), 0)
            self.assertEqual(features.shape, (len(records), dimension))
            self.assertTrue(np.all(np.isfinite(features)))
            for record in records[:10]:
                record.validate()

    def test_pinch_pointer_local_transitions_remain_staggered_on_global_time(self):
        records, _ = self.loaded["pinch"]
        staggered = None
        for record in records:
            first0 = int(np.flatnonzero(record.contact_mask[0])[0])
            first1 = int(np.flatnonzero(record.contact_mask[1])[0])
            last0 = int(np.flatnonzero(record.contact_mask[0])[-1])
            last1 = int(np.flatnonzero(record.contact_mask[1])[-1])
            if first0 != first1 or last0 != last1:
                staggered = (record, first0, first1, last0, last1)
                break
        self.assertIsNotNone(staggered)
        record, first0, first1, last0, last1 = staggered
        self.assertEqual(int(record.action_code[0, first0]), 0)
        self.assertEqual(int(record.action_code[1, first1]), 5)
        self.assertEqual(int(record.action_code[0, last0]), 1)
        self.assertEqual(int(record.action_code[1, last1]), 6)
        if first1 > first0:
            self.assertEqual(int(record.action_code[0, first1]), 2)
        self.assertTrue(np.all(np.diff(record.global_t_ms) > 0))

    def test_keystroke_contact_transitions_have_real_no_xy_gap_tokens(self):
        records, _ = self.loaded["keystroke"]
        multi = next(record for record in records if len(np.unique(record.event_ids[record.contact_mask])) >= 3)
        self.assertGreater(int(np.sum(multi.gap_mask)), 1)
        self.assertFalse(np.any(multi.contact_mask[:, multi.gap_mask]))
        self.assertTrue(np.all(multi.pointer_continuous[:, multi.gap_mask] == 0.0))
        contact_events = multi.event_ids[0, multi.contact_mask[0]]
        self.assertGreater(len(np.unique(contact_events)), 1)

    def test_negative_raw_keycode_sentinels_share_one_canonical_token(self):
        base = keystroke_events(user_id=0, count=1)[0]
        events = []
        for index, sentinel in enumerate((-1, -2, -5)):
            event = copy.deepcopy(base)
            event["event_id"] = 9100 + index
            for row in event["flat_rows"][:2]:
                row["flat_keycode"] = sentinel
            event["key_rows"][0]["keycode"] = sentinel
            events.append(event)
        with tempfile.TemporaryDirectory() as directory:
            path = write_archive(Path(directory) / "keystroke.npz", "keystroke", events)
            records, features = load_extracted_trajectory_npz(path, label=0)
        for record in records:
            first_contact = record.contact_mask[0] & (record.event_ids[0] == 0)
            self.assertTrue(np.all(record.keycode[0, first_contact] == 0))
            self.assertTrue(np.all(record.keycode[:, record.gap_mask] == -1))
        np.testing.assert_array_equal(features[0], features[1])
        np.testing.assert_array_equal(features[1], features[2])
        # First key is a canonical non-letter sentinel; second raw code 97 is
        # ASCII 'a' and must contribute exactly one letter feature.
        self.assertTrue(np.all(features[:, 1] == 1.0))

    def test_pinch_slots_follow_first_appearance_not_numeric_pointer_id(self):
        rows = [
            _flat_row(0, 9, 0, 900, 100, 0, 1),
            _flat_row(20, 9, 5, 910, 100, 1, 2),
            _flat_row(20, 2, 5, 200, 100, 1, 2),
            _flat_row(50, 9, 2, 920, 100, 2, 2),
            _flat_row(50, 2, 2, 210, 100, 2, 2),
            _flat_row(80, 9, 6, 930, 100, 3, 2),
            _flat_row(80, 2, 6, 220, 100, 3, 2),
            _flat_row(100, 9, 1, 940, 100, 4, 1),
        ]
        event = {
            "event_id": 9900, "user_id": 0, "duration_ms": 100,
            "flat_rows": rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = write_archive(Path(directory) / "pinch.npz", "pinch", [event])
            records, _ = load_extracted_trajectory_npz(path, label=0)
        record = records[0]
        first0 = int(np.flatnonzero(record.contact_mask[0])[0])
        first1 = int(np.flatnonzero(record.contact_mask[1])[0])
        self.assertEqual(float(record.pointer_continuous[0, first0, 0]), 900.0)
        self.assertEqual(float(record.pointer_continuous[1, first1, 0]), 200.0)
        self.assertEqual(int(record.action_code[0, first0]), 0)
        self.assertEqual(int(record.action_code[1, first1]), 5)


if __name__ == "__main__":
    unittest.main()
