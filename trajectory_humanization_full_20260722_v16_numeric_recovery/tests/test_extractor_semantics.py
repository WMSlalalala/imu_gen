#!/usr/bin/env python3
from __future__ import annotations

import unittest

import pandas as pd

from preprocess.extract_hmog_trajectories import (
    ACTION_DOWN,
    ACTION_UP,
    RawGesture,
    find_gesture_match,
    process_non_key_event,
    reconstruct_one_finger_contacts,
    validate_pointer_semantics,
)


class ExtractorSemanticsTest(unittest.TestCase):
    def test_filtered_higher_priority_action_still_reserves_raw_contact(self):
        rows = pd.DataFrame(
            [
                [1000, 100, 17, 1, 0, ACTION_DOWN, 10.0, 20.0, 0.5, 0.2, 0, 0, 0],
                [1010, 110, 17, 1, 0, ACTION_UP, 30.0, 40.0, 0.4, 0.2, 0, 1, 1],
            ],
            columns=(
                "sys_t", "evt_t", "act_id", "pointer_count", "pointer_id", "action",
                "x", "y", "pressure", "size", "orient", "source_row",
                "gesture_frame_index",
            ),
        )
        raw = RawGesture(9, 17, rows)

        class Event:
            start_ms = 100
            end_ms = 110
            activity_id = 17
            orientation_id = 0
            meta = {}

        class DP:
            ACTION_NAME_TO_ID = {"swipe": 2}
            XY_FLOAT_FIELDS = ()
            XY_INT_FIELDS = ()

        class Audit:
            rows = []

            def add_event(self, row):
                self.rows.append(row)

        used = set()
        audit = Audit()
        process_non_key_event(
            dp=DP,
            writer=None,
            audit=audit,
            event=Event(),
            action="swipe",
            event_id=1,
            user_id=0,
            user_external_id=123,
            session_id=1,
            activity_meta={},
            gestures_by_activity={17: [raw]},
            used_gesture_ids=used,
            match_tolerance_ms=0,
            container_margin_ms=0,
        )
        self.assertEqual(used, {9})
        self.assertEqual(audit.rows[0]["status"], "reserved_not_emitted")
        self.assertEqual(audit.rows[0]["reason"], "requested_action_filter")
        self.assertIsNone(
            find_gesture_match(Event(), "tap", {17: [raw]}, used, 0, 0)
        )

    def test_one_finger_fallback_uses_observed_down_up_without_interpolation(self):
        # HMOG file order/SystemTime can be opposite to the authoritative
        # EventTime order.  The fallback must still restore exactly two real
        # endpoint rows and must not invent MOVE samples.
        frame = pd.DataFrame(
            [
                [2000, 110, 77, 9, 1, ACTION_UP, 104.0, 205.0, 0.2, 0.3, 0],
                [2010, 100, 77, 9, 0, ACTION_DOWN, 100.0, 200.0, 0.5, 0.4, 0],
            ],
            columns=(
                "sys_t", "evt_t", "act_id", "tap_id", "tap_type", "action",
                "x", "y", "pressure", "size", "orient",
            ),
        )
        contacts, audit = reconstruct_one_finger_contacts(frame)
        self.assertEqual(audit["one_finger_complete_contacts"], 1)
        self.assertEqual(len(contacts), 1)
        contact = contacts[0]
        self.assertLess(contact.gesture_id, 0)
        self.assertEqual(contact.rows["action"].tolist(), [ACTION_DOWN, ACTION_UP])
        self.assertEqual(contact.rows["evt_t"].tolist(), [100, 110])
        self.assertEqual(contact.rows[["x", "y"]].to_numpy().tolist(), [[100.0, 200.0], [104.0, 205.0]])

        class Key:
            start_ms = 100
            end_ms = 110
            activity_id = 77
            orientation_id = 0
            meta = {}

        match = find_gesture_match(Key(), "tap", {77: contacts}, set(), 0, 0)
        self.assertIsNotNone(match)
        self.assertEqual(match.start_error_ms, 0)
        self.assertEqual(match.end_error_ms, 0)

    def test_pinch_rejects_three_pointer_contact(self):
        rows = []
        for frame_index, event_time in enumerate((0, 20)):
            for pointer_id in range(3):
                rows.append(
                    {
                        "sys_t": event_time,
                        "evt_t": event_time,
                        "act_id": 1,
                        "pointer_count": 3,
                        "pointer_id": pointer_id,
                        "action": ACTION_DOWN if frame_index == 0 else ACTION_UP,
                        "x": float(pointer_id),
                        "y": float(pointer_id),
                        "pressure": 0.5,
                        "size": 0.2,
                        "orient": 0,
                        "source_row": len(rows),
                        "gesture_frame_index": frame_index,
                    }
                )
        raw = RawGesture(1, 1, pd.DataFrame(rows))

        class Pinch:
            start_ms = 0
            end_ms = 20

        self.assertEqual(
            validate_pointer_semantics(Pinch(), "pinch", raw),
            "pinch_is_not_exactly_two_pointer_contact",
        )


if __name__ == "__main__":
    unittest.main()
