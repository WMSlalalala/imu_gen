"""Small numeric flat+offset archives used only by unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from preprocess.extract_hmog_trajectories import (
    EVENT_BASE_SPECS,
    FLAT_SPECS,
    KEY_SPECS,
    SCHEMA_VERSION,
)


ACTION_IDS = {"tap": 0, "scroll": 1, "swipe": 2, "pinch": 3, "keystroke": 4}


def _scalar_defaults(action: str) -> Dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "action_name": np.asarray(action),
        "action_id_scalar": np.asarray(ACTION_IDS[action], dtype=np.int8),
        "time_unit": np.asarray("ms"),
        "coordinate_unit": np.asarray("raw_screen_pixel"),
        "sampling": np.asarray("raw_irregular_touch_events"),
    }


def _append(storage: Dict[str, List], specs: Dict[str, str], row: Dict[str, object]) -> None:
    for name, dtype in specs.items():
        storage[name].append(row.get(name, 0))


def _event_row(action: str, event_id: int, user_id: int, n_rows: int, duration: int, n_keys: int, n_letters: int) -> Dict[str, object]:
    return {
        "event_id": event_id,
        "user_id": user_id,
        "user_external_id": 100000 + user_id,
        "session_id": 1,
        "action_id": ACTION_IDS[action],
        "activity_id": 100000000 + event_id,
        "orientation_id": (0, 1, 3)[user_id % 3],
        "label_start_ms": 0,
        "label_end_ms": duration,
        "label_duration_ms": duration,
        "touch_start_ms": 0,
        "touch_end_ms": duration,
        "touch_duration_ms": duration,
        "active_start_rel_ms": 0,
        "active_end_rel_ms": duration,
        "n_rows": n_rows,
        "n_frames": n_rows,
        "n_pointers": 2 if action == "pinch" else 1,
        "max_pointer_count": 2 if action == "pinch" else 1,
        "active_row_count": n_rows,
        "raw_gesture_id": event_id,
        "n_raw_gestures": n_keys if action == "keystroke" else 1,
        "match_start_error_ms": 0,
        "match_end_error_ms": 0,
        "n_keys": n_keys,
        "n_letters": n_letters,
        "motion_id_raw": 1,
        "posture_id": 0,
        "task_id": 1,
        "activity_subtask_id": 1,
    }


def _flat_row(
    t: int,
    pointer_id: int,
    action_code: int,
    x: float,
    y: float,
    frame: int,
    pointer_count: int,
    key_index: int = -1,
    keycode: int = -1,
) -> Dict[str, object]:
    return {
        "flat_system_time_ms": 1000000 + t,
        "flat_event_time_ms": t,
        "flat_t_rel_ms": t,
        "flat_frame_index": frame,
        "flat_pointer_count": pointer_count,
        "flat_pointer_id": pointer_id,
        "flat_action_code": action_code,
        "flat_x": x,
        "flat_y": y,
        "flat_pressure": 0.5,
        "flat_size": 0.2,
        "flat_orientation_id": 0,
        "flat_active_mask": 1,
        "flat_valid_mask": 1,
        "flat_key_index": key_index,
        "flat_keycode": keycode,
    }


def write_archive(path: Path, action: str, events: Sequence[Dict[str, object]]) -> Path:
    flat_store = {name: [] for name in FLAT_SPECS}
    event_store = {name: [] for name in EVENT_BASE_SPECS}
    key_store = {name: [] for name in KEY_SPECS}
    event_offsets = [0]
    event_key_offsets = [0]
    key_touch_offsets = [0]
    for event in events:
        flat_rows = list(event["flat_rows"])
        key_rows = list(event.get("key_rows", []))
        key_lengths = list(event.get("key_lengths", []))
        if len(key_rows) != len(key_lengths):
            raise ValueError("key test fixture mismatch")
        for row in flat_rows:
            _append(flat_store, FLAT_SPECS, row)
        row = _event_row(
            action,
            int(event["event_id"]),
            int(event["user_id"]),
            len(flat_rows),
            int(event["duration_ms"]),
            len(key_rows),
            sum(int(value.get("key_is_letter", 0)) for value in key_rows),
        )
        _append(event_store, EVENT_BASE_SPECS, row)
        for key, length in zip(key_rows, key_lengths):
            _append(key_store, KEY_SPECS, key)
            key_touch_offsets.append(key_touch_offsets[-1] + int(length))
        event_offsets.append(event_offsets[-1] + len(flat_rows))
        event_key_offsets.append(event_key_offsets[-1] + len(key_rows))
    arrays = {
        name: np.asarray(values, dtype=np.dtype(dtype))
        for name, (values, dtype) in {
            **{name: (flat_store[name], dtype) for name, dtype in FLAT_SPECS.items()},
            **{name: (event_store[name], dtype) for name, dtype in EVENT_BASE_SPECS.items()},
            **{name: (key_store[name], dtype) for name, dtype in KEY_SPECS.items()},
        }.items()
    }
    arrays.update(
        {
            "event_offsets": np.asarray(event_offsets, dtype=np.int64),
            "event_key_offsets": np.asarray(event_key_offsets, dtype=np.int64),
            "key_touch_offsets": np.asarray(key_touch_offsets, dtype=np.int64),
        }
    )
    arrays.update(_scalar_defaults(action))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)
    return path


def tap_events_for_all_users(events_per_user: int = 6) -> List[Dict[str, object]]:
    events = []
    event_id = 1
    for user_id in range(100):
        for local in range(events_per_user):
            duration = 80 + local
            rows = [
                _flat_row(0, 0, 0, 100 + local, 400 + user_id, 0, 1),
                _flat_row(duration // 2, 0, 2, 101 + local, 401 + user_id, 1, 1),
                _flat_row(duration, 0, 1, 101 + local, 401 + user_id, 2, 1),
            ]
            events.append(
                {"event_id": event_id, "user_id": user_id, "duration_ms": duration, "flat_rows": rows}
            )
            event_id += 1
    return events


def staggered_pinch_events(user_id: int = 0, count: int = 6) -> List[Dict[str, object]]:
    events = []
    for local in range(count):
        # Pointer 0 spans [0,100], pointer 7 spans [20,80].  Rows are globally
        # time ordered to mimic the numeric extractor output.
        rows = [
            _flat_row(0, 0, 0, 100, 100, 0, 1),
            _flat_row(20, 0, 5, 105, 105, 1, 2),
            _flat_row(20, 0, 5, 106, 106, 1, 2),  # repeated Android state snapshot
            _flat_row(20, 7, 5, 200, 200, 1, 2),
            _flat_row(50, 0, 2, 110, 110, 2, 2),
            _flat_row(50, 7, 2, 210, 210, 2, 2),
            _flat_row(80, 0, 6, 115, 115, 3, 2),
            _flat_row(80, 7, 6, 220, 220, 3, 2),
            _flat_row(100, 0, 1, 120, 120, 4, 1),
        ]
        events.append(
            {"event_id": 3000 + local, "user_id": user_id, "duration_ms": 100, "flat_rows": rows}
        )
    return events


def keystroke_events(user_id: int = 0, count: int = 6) -> List[Dict[str, object]]:
    events = []
    for local in range(count):
        raw_codes = (-5, 97)
        rows = [
            _flat_row(0, 0, 0, 100, 500, 0, 1, 0, raw_codes[0]),
            _flat_row(30, 0, 1, 101, 501, 1, 1, 0, raw_codes[0]),
            _flat_row(80, 0, 0, 300, 500, 2, 1, 1, raw_codes[1]),
            _flat_row(120, 0, 1, 301, 501, 3, 1, 1, raw_codes[1]),
        ]
        key_rows = [
            {
                "key_index_in_event": 0, "keycode": raw_codes[0], "key_is_letter": 0,
                "key_down_ms": 0, "key_up_ms": 30, "key_hold_ms": 30,
                "key_flight_from_previous_ms": 0, "key_orientation_id": 0,
                "key_raw_gesture_id": 1, "key_touch_start_ms": 0, "key_touch_end_ms": 30,
                "key_match_start_error_ms": 0, "key_match_end_error_ms": 0, "key_touch_found": 1,
            },
            {
                "key_index_in_event": 1, "keycode": raw_codes[1], "key_is_letter": 1,
                "key_down_ms": 80, "key_up_ms": 120, "key_hold_ms": 40,
                "key_flight_from_previous_ms": 50, "key_orientation_id": 0,
                "key_raw_gesture_id": 2, "key_touch_start_ms": 80, "key_touch_end_ms": 120,
                "key_match_start_error_ms": 0, "key_match_end_error_ms": 0, "key_touch_found": 1,
            },
        ]
        events.append(
            {
                "event_id": 4000 + local, "user_id": user_id, "duration_ms": 120,
                "flat_rows": rows, "key_rows": key_rows, "key_lengths": [2, 2],
            }
        )
    return events
