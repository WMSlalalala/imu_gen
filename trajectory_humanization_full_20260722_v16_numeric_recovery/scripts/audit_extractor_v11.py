#!/usr/bin/env python3
"""Fail-closed audit for HMOG trajectory extractor v1.1.0 outputs.

Complete mode validates finalized NPZ/manifest/audit artifacts.  In-progress
mode only reports stable binary-table prefixes and can never return a formal
pass.  It is safe to run while the extractor appends to ``.build``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory.features import is_hmog_ascii_letter_keycode


ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
ACTION_ID = {"tap": 0, "scroll": 1, "swipe": 2, "pinch": 3, "keystroke": 4}
EXPECTED_SCHEMA = "hmog_touch_trajectory_v1"
EXPECTED_SCRIPT_VERSION = "1.1.0"
ALLOWED_ORIENTATIONS = {-1, 0, 1, 3}

FLAT_SPECS = {
    "flat_system_time_ms": "<i8", "flat_event_time_ms": "<i8",
    "flat_t_rel_ms": "<i8", "flat_frame_index": "<i4",
    "flat_pointer_count": "i1", "flat_pointer_id": "<i2",
    "flat_action_code": "i1", "flat_x": "<f4", "flat_y": "<f4",
    "flat_pressure": "<f4", "flat_size": "<f4",
    "flat_orientation_id": "i1", "flat_active_mask": "u1",
    "flat_valid_mask": "u1", "flat_key_index": "<i4",
    "flat_keycode": "<i4",
}

EVENT_SPECS = {
    "event_id": "<i8", "user_id": "<i4", "user_external_id": "<i8",
    "session_id": "<i2", "action_id": "i1", "activity_id": "<i8",
    "orientation_id": "i1", "label_start_ms": "<i8",
    "label_end_ms": "<i8", "label_duration_ms": "<i4",
    "touch_start_ms": "<i8", "touch_end_ms": "<i8",
    "touch_duration_ms": "<i4", "active_start_rel_ms": "<i4",
    "active_end_rel_ms": "<i4", "n_rows": "<i4", "n_frames": "<i4",
    "n_pointers": "i1", "max_pointer_count": "i1",
    "active_row_count": "<i4", "raw_gesture_id": "<i4",
    "n_raw_gestures": "<i2", "match_start_error_ms": "<i4",
    "match_end_error_ms": "<i4", "n_keys": "<i2",
    "n_letters": "<i2", "motion_id_raw": "<i2", "posture_id": "i1",
    "task_id": "<i2", "activity_subtask_id": "<i2",
}

KEY_SPECS = {
    "key_index_in_event": "<i2", "keycode": "<i4", "key_is_letter": "u1",
    "key_down_ms": "<i8", "key_up_ms": "<i8", "key_hold_ms": "<i4",
    "key_flight_from_previous_ms": "<i4", "key_orientation_id": "i1",
    "key_raw_gesture_id": "<i4", "key_touch_start_ms": "<i8",
    "key_touch_end_ms": "<i8", "key_match_start_error_ms": "<i4",
    "key_match_end_error_ms": "<i4", "key_touch_found": "u1",
    "key_touch_source": "u1",
}


class InvariantError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not bool(condition):
        raise InvariantError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _require_dtype(data: Mapping[str, np.ndarray], specs: Mapping[str, str], context: str) -> None:
    missing = sorted(set(specs) - set(data))
    require(not missing, "%s missing arrays: %s" % (context, missing))
    for name, dtype in specs.items():
        require(np.dtype(data[name].dtype) == np.dtype(dtype), "%s dtype mismatch: %s" % (context, name))


def _offsets(array: np.ndarray, expected_last: int, expected_length: int, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.int64)
    require(values.shape == (expected_length,), "%s length mismatch" % name)
    require(values[0] == 0 and values[-1] == expected_last, "%s endpoints mismatch" % name)
    require(np.all(np.diff(values) >= 0), "%s must be monotonic" % name)
    return values


def _frame_groups(frame_index: np.ndarray) -> List[np.ndarray]:
    if len(frame_index) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(frame_index) != 0) + 1
    return list(np.split(np.arange(len(frame_index)), boundaries))


def audit_action_npz(path: Path, action: str, expected_users: int) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        require(all(archive[name].dtype.kind != "O" for name in archive.files), "%s contains object arrays" % action)
        data = {name: archive[name] for name in archive.files}
    _require_dtype(data, FLAT_SPECS, action + "/flat")
    _require_dtype(data, EVENT_SPECS, action + "/event")
    _require_dtype(data, KEY_SPECS, action + "/key")
    require(str(data["schema_version"].item()) == EXPECTED_SCHEMA, "%s schema mismatch" % action)
    require(str(data["action_name"].item()) == action, "%s action_name mismatch" % action)
    require(int(data["action_id_scalar"].item()) == ACTION_ID[action], "%s action scalar mismatch" % action)
    require(str(data["time_unit"].item()) == "ms", "%s time unit mismatch" % action)
    require(str(data["coordinate_unit"].item()) == "raw_screen_pixel", "%s coordinate unit mismatch" % action)
    require(str(data["sampling"].item()) == "raw_irregular_touch_events", "%s sampling claim mismatch" % action)

    e = len(data["event_id"])
    f = len(data["flat_x"])
    k = len(data["keycode"])
    require(e > 0 and f > 0, "%s must contain events and raw rows" % action)
    for name in FLAT_SPECS:
        require(len(data[name]) == f, "%s flat length mismatch: %s" % (action, name))
    for name in EVENT_SPECS:
        require(len(data[name]) == e, "%s event length mismatch: %s" % (action, name))
    for name in KEY_SPECS:
        require(len(data[name]) == k, "%s key length mismatch: %s" % (action, name))
    event_offsets = _offsets(data["event_offsets"], f, e + 1, action + "/event_offsets")
    event_key_offsets = _offsets(data["event_key_offsets"], k, e + 1, action + "/event_key_offsets")
    key_touch_offsets = _offsets(data["key_touch_offsets"], f if action == "keystroke" else 0, k + 1, action + "/key_touch_offsets")

    require(np.array_equal(np.diff(event_offsets), data["n_rows"].astype(np.int64)), "%s n_rows != offsets" % action)
    require(np.array_equal(np.diff(event_key_offsets), data["n_keys"].astype(np.int64)), "%s n_keys != key offsets" % action)
    require(np.all(data["label_duration_ms"] == data["label_end_ms"] - data["label_start_ms"]), "%s label duration mismatch" % action)
    require(np.all(data["touch_duration_ms"] == data["touch_end_ms"] - data["touch_start_ms"]), "%s touch duration mismatch" % action)
    require(np.all(data["label_duration_ms"] > 0) and np.all(data["touch_duration_ms"] > 0), "%s nonpositive duration" % action)
    # Keep the unmodified label and raw-touch clocks.  The relative interval
    # is derived directly from those clocks and may extend outside the raw
    # contact while the two intervals still overlap.  For non-key gestures the
    # stored match errors are the same two endpoint differences.  Keystroke
    # event match errors are instead rounded means over all key contacts and
    # are checked from the key table below.
    require(
        np.array_equal(
            data["active_start_rel_ms"],
            data["label_start_ms"] - data["touch_start_ms"],
        ),
        "%s active start is not label_start-touch_start" % action,
    )
    require(
        np.array_equal(
            data["active_end_rel_ms"],
            data["label_end_ms"] - data["touch_start_ms"],
        ),
        "%s active end is not label_end-touch_start" % action,
    )
    if action != "keystroke":
        require(
            np.array_equal(
                data["active_start_rel_ms"], -data["match_start_error_ms"]
            ),
            "%s active start/match error mismatch" % action,
        )
        require(
            np.array_equal(
                data["active_end_rel_ms"] - data["touch_duration_ms"],
                -data["match_end_error_ms"],
            ),
            "%s active end/match error mismatch" % action,
        )
    require(np.all(data["active_start_rel_ms"] <= data["touch_duration_ms"]), "%s label starts after touch" % action)
    require(np.all(data["active_end_rel_ms"] >= 0), "%s label ends before touch" % action)
    require(np.all(data["active_end_rel_ms"] >= data["active_start_rel_ms"]), "%s invalid active interval" % action)
    require(np.all(data["active_row_count"] >= 1), "%s event without active rows" % action)
    require(np.array_equal(data["active_row_count"], np.add.reduceat(data["flat_active_mask"].astype(np.int64), event_offsets[:-1])), "%s active row count mismatch" % action)
    row_label_start = np.repeat(data["label_start_ms"], data["n_rows"])
    row_label_end = np.repeat(data["label_end_ms"], data["n_rows"])
    expected_active = (
        (data["flat_event_time_ms"] >= row_label_start)
        & (data["flat_event_time_ms"] <= row_label_end)
    ).astype(np.uint8)
    require(np.array_equal(data["flat_active_mask"], expected_active), "%s flat active mask/label interval mismatch" % action)
    require(set(np.unique(data["flat_valid_mask"]).tolist()) == {1}, "%s valid_mask must be all one" % action)
    require(set(np.unique(data["flat_active_mask"]).tolist()).issubset({0, 1}), "%s active mask not binary" % action)
    require(set(np.unique(data["orientation_id"]).tolist()).issubset(ALLOWED_ORIENTATIONS), "%s invalid event orientation" % action)
    require(set(np.unique(data["flat_orientation_id"]).tolist()).issubset(ALLOWED_ORIENTATIONS), "%s invalid flat orientation" % action)
    require(np.all(np.isfinite(data["flat_x"])) and np.all(np.isfinite(data["flat_y"])), "%s nonfinite XY" % action)
    require(np.all(np.isfinite(data["flat_pressure"])) and np.all(np.isfinite(data["flat_size"])), "%s nonfinite contact values" % action)
    require(np.all(data["action_id"] == ACTION_ID[action]), "%s event action IDs mismatch" % action)
    require(len(np.unique(data["event_id"])) == e, "%s duplicate event IDs" % action)
    require(np.all((data["session_id"] >= 1) & (data["session_id"] <= 24)), "%s invalid session ID" % action)
    users = np.unique(data["user_id"])
    require(len(users) == expected_users, "%s expected %d users, found %d" % (action, expected_users, len(users)))

    if action == "keystroke":
        require(k > 0, "keystroke has no keys")
        require(np.all(data["n_keys"] > 0), "keystroke event without keys")
        require(np.all(data["n_letters"] >= 0) and np.all(data["n_letters"] <= data["n_keys"]), "keystroke n_letters invalid")
        expected_letter_flags = np.asarray([
            is_hmog_ascii_letter_keycode(int(raw_keycode))
            for raw_keycode in data["keycode"].tolist()
        ], dtype=np.uint8)
        require(
            np.array_equal(data["key_is_letter"].astype(np.uint8), expected_letter_flags),
            "keystroke key_is_letter contradicts HMOG ASCII keycode codebook",
        )
        require(np.all(data["n_raw_gestures"] == data["n_keys"]), "keystroke raw gesture count mismatch")
        require(np.all(data["raw_gesture_id"] == -1), "keystroke event raw_gesture_id must be -1")
        require(set(np.unique(data["key_touch_source"]).tolist()).issubset({0, 1}), "unknown keystroke touch source")
        require(np.all(data["key_touch_found"] == 1), "formal keystroke contains missing contact")
        require(np.all(data["key_up_ms"] >= data["key_down_ms"]), "key UP before DOWN")
        require(np.all(data["key_hold_ms"] == data["key_up_ms"] - data["key_down_ms"]), "key hold mismatch")
        require(np.all(data["key_touch_start_ms"] - data["key_down_ms"] == data["key_match_start_error_ms"]), "key start error mismatch")
        require(np.all(data["key_touch_end_ms"] - data["key_up_ms"] == data["key_match_end_error_ms"]), "key end error mismatch")
        key_counts = np.diff(event_key_offsets)
        require(np.all(key_counts > 0), "keystroke event without aligned keys")
        mean_start_error = np.rint(
            np.add.reduceat(
                data["key_match_start_error_ms"].astype(np.float64),
                event_key_offsets[:-1],
            ) / key_counts
        ).astype(data["match_start_error_ms"].dtype)
        mean_end_error = np.rint(
            np.add.reduceat(
                data["key_match_end_error_ms"].astype(np.float64),
                event_key_offsets[:-1],
            ) / key_counts
        ).astype(data["match_end_error_ms"].dtype)
        require(
            np.array_equal(data["match_start_error_ms"], mean_start_error),
            "keystroke event match_start_error is not the rounded per-key mean",
        )
        require(
            np.array_equal(data["match_end_error_ms"], mean_end_error),
            "keystroke event match_end_error is not the rounded per-key mean",
        )
        require(np.all(data["key_orientation_id"] == np.repeat(data["orientation_id"], data["n_keys"])), "key/event orientation mismatch")
        require(np.array_equal(key_touch_offsets[event_key_offsets[:-1]], event_offsets[:-1]), "event/key flat left offsets disagree")
        require(np.array_equal(key_touch_offsets[event_key_offsets[1:]], event_offsets[1:]), "event/key flat right offsets disagree")
    else:
        require(k == 0, "%s unexpectedly contains key rows" % action)
        require(np.all(data["n_keys"] == 0) and np.all(data["n_letters"] == 0), "%s key counts nonzero" % action)
        require(np.all(data["n_raw_gestures"] == 1), "%s must map to one raw gesture" % action)

    pointer_summary = Counter()
    source_summary = Counter()
    repeated_pointer_rows = 0
    raw_identities: List[Tuple[int, int, int, str]] = []
    for index in range(e):
        left, right = int(event_offsets[index]), int(event_offsets[index + 1])
        frames = data["flat_frame_index"][left:right]
        times = data["flat_t_rel_ms"][left:right]
        event_times = data["flat_event_time_ms"][left:right]
        pointer_count = data["flat_pointer_count"][left:right]
        pointer_id = data["flat_pointer_id"][left:right]
        codes = data["flat_action_code"][left:right]
        active = data["flat_active_mask"][left:right].astype(bool)
        orientation = data["flat_orientation_id"][left:right]
        require(np.all(np.diff(frames) >= 0), "%s event frame index goes backward" % action)
        groups = _frame_groups(frames)
        require(len(groups) == int(data["n_frames"][index]), "%s n_frames mismatch" % action)
        frame_times = []
        for group in groups:
            require(np.all(times[group] == times[group[0]]), "%s frame t_rel disagreement" % action)
            require(np.all(event_times[group] == event_times[group[0]]), "%s frame EventTime disagreement" % action)
            frame_pointer_ids = np.unique(pointer_id[group])
            # HMOG can log multiple updates for one pointer with the same
            # Android EventTime (different SystemTime/XY).  These are observed
            # rows, not invented samples.  The canonical detector adapter uses
            # deterministic last-row-wins inside that MotionEvent frame.  The
            # declared pointer_count must therefore match unique pointer IDs,
            # not the number of repeated log rows.
            require(
                int(np.max(pointer_count[group])) == len(frame_pointer_ids)
                and np.all(pointer_count[group] >= 1)
                and np.all(pointer_count[group] <= len(frame_pointer_ids)),
                "%s pointer_count/unique frame pointers mismatch" % action,
            )
            repeated_pointer_rows += int(len(group) - len(frame_pointer_ids))
            frame_times.append(int(times[group[0]]))
        frame_times_array = np.asarray(frame_times, dtype=np.int64)
        frame_deltas = np.diff(frame_times_array)
        if action == "keystroke":
            # Key contacts are stored contact-major.  HMOG contains legitimate
            # zero-flight transitions where one key UP and the next key DOWN
            # share the same millisecond; time may be equal only across that
            # contact boundary, never go backwards or duplicate within a key.
            require(
                frame_times_array[0] == 0 and np.all(frame_deltas >= 0),
                "keystroke global contact-major frame time invalid",
            )
            frame_keys = np.asarray(
                [int(data["flat_key_index"][left + int(group[0])]) for group in groups],
                dtype=np.int64,
            )
            equal_positions = np.flatnonzero(frame_deltas == 0)
            require(
                np.all(frame_keys[equal_positions + 1] > frame_keys[equal_positions]),
                "keystroke duplicate timestamp occurs inside one key contact",
            )
        else:
            require(
                frame_times_array[0] == 0 and np.all(frame_deltas > 0),
                "%s global frame time invalid" % action,
            )
        require(frame_times[-1] == int(data["touch_duration_ms"][index]), "%s last frame != touch duration" % action)
        unique_pointers = np.unique(pointer_id)
        if action == "keystroke":
            # A typing event concatenates independent key contacts.  Android
            # may reuse a different tracking id for a later contact, so event-
            # global unique pointer IDs are not a simultaneous pointer count.
            require(
                int(data["n_pointers"][index]) == 1
                and int(data["max_pointer_count"][index]) == 1
                and np.all(pointer_count == 1),
                "keystroke event must be single-pointer per key contact",
            )
        else:
            require(len(unique_pointers) == int(data["n_pointers"][index]), "%s n_pointers mismatch" % action)
            require(int(np.max(pointer_count)) == int(data["max_pointer_count"][index]), "%s max pointer count mismatch" % action)
        require(np.all(orientation == data["orientation_id"][index]), "%s orientation changes within event" % action)
        require(set(np.unique(codes).tolist()).issubset({0, 1, 2, 5, 6}), "%s unsupported action code" % action)
        pointer_summary.update([int(data["n_pointers"][index])])

        if action in ("tap", "scroll", "swipe"):
            require(len(unique_pointers) == 1 and np.all(pointer_count == 1), "%s must be single pointer" % action)
        if action in ("tap", "scroll", "swipe"):
            require(codes[0] == 0 and codes[-1] == 1, "%s incomplete DOWN/UP" % action)
            raw_identities.append((int(data["user_id"][index]), int(data["session_id"][index]), int(data["raw_gesture_id"][index]), action))
        elif action == "pinch":
            require(len(unique_pointers) == 2, "pinch must contain exactly two unique pointer IDs")
            require(int(data["max_pointer_count"][index]) == 2 and np.all(pointer_count <= 2), "pinch exceeds exactly two pointers")
            for group in groups:
                if np.any(active[group]):
                    require(
                        len(np.unique(pointer_id[group])) == 2,
                        "pinch active frame must contain both pointers",
                    )
            raw_identities.append((int(data["user_id"][index]), int(data["session_id"][index]), int(data["raw_gesture_id"][index]), action))
        else:
            key_left, key_right = int(event_key_offsets[index]), int(event_key_offsets[index + 1])
            require(np.array_equal(data["key_index_in_event"][key_left:key_right], np.arange(key_right - key_left)), "keystroke key indices invalid")
            require(int(np.sum(data["key_is_letter"][key_left:key_right])) == int(data["n_letters"][index]), "keystroke letter count mismatch")
            require(
                int(data["label_start_ms"][index]) == int(data["key_down_ms"][key_left])
                and int(data["label_end_ms"][index]) == int(data["key_up_ms"][key_right - 1]),
                "keystroke event label is not first DOWN..last UP",
            )
            require(
                int(data["touch_start_ms"][index]) == int(data["key_touch_start_ms"][key_left])
                and int(data["touch_end_ms"][index]) == int(data["key_touch_end_ms"][key_right - 1]),
                "keystroke event touch bounds are not first..last key contact",
            )
            expected_start_summary = int(round(float(np.mean(
                data["key_match_start_error_ms"][key_left:key_right]
            ))))
            expected_end_summary = int(round(float(np.mean(
                data["key_match_end_error_ms"][key_left:key_right]
            ))))
            require(
                int(data["match_start_error_ms"][index]) == expected_start_summary
                and int(data["match_end_error_ms"][index]) == expected_end_summary,
                "keystroke event match-error summary mismatch",
            )
            require(data["key_flight_from_previous_ms"][key_left] == 0, "first key flight must be zero")
            if key_right - key_left > 1:
                expected_flight = data["key_down_ms"][key_left + 1:key_right] - data["key_up_ms"][key_left:key_right - 1]
                require(np.array_equal(data["key_flight_from_previous_ms"][key_left + 1:key_right], expected_flight), "key flight mismatch")
            for key_index in range(key_left, key_right):
                kl, kr = int(key_touch_offsets[key_index]), int(key_touch_offsets[key_index + 1])
                local_key_index = key_index - key_left
                require(kl >= left and kr <= right and kr > kl, "key touch offsets outside event")
                require(
                    len(np.unique(data["flat_pointer_id"][kl:kr])) == 1
                    and np.all(data["flat_pointer_count"][kl:kr] == 1),
                    "each keystroke contact must be single pointer",
                )
                require(np.all(data["flat_key_index"][kl:kr] == local_key_index), "flat key index mismatch")
                require(np.all(data["flat_keycode"][kl:kr] == data["keycode"][key_index]), "flat keycode mismatch")
                require(data["flat_action_code"][kl] == 0 and data["flat_action_code"][kr - 1] == 1, "key contact incomplete DOWN/UP")
                source = int(data["key_touch_source"][key_index])
                source_summary.update([source])
                raw_id = int(data["key_raw_gesture_id"][key_index])
                if source == 0:
                    require(raw_id >= 0, "raw TouchEvent key must use nonnegative gesture ID")
                    raw_identities.append((int(data["user_id"][index]), int(data["session_id"][index]), raw_id, action))
                else:
                    require(raw_id < 0, "OneFinger fallback key must use negative gesture ID")
                    require(kr - kl == 2, "OneFinger fallback must contain exactly observed DOWN/UP rows")
                    require(np.array_equal(data["flat_action_code"][kl:kr], np.asarray([0, 1])), "fallback invented a MOVE/action")

    # No raw TouchEvent gesture can be reused within this action.  Negative
    # OneFinger IDs occupy a separate per-session source namespace.
    positive = [identity[:3] for identity in raw_identities if identity[2] >= 0]
    require(len(positive) == len(set(positive)), "%s reuses a raw TouchEvent gesture" % action)
    per_user_counts = {str(int(user)): int(np.sum(data["user_id"] == user)) for user in users}
    return {
        "action": action,
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "n_events": e,
        "n_flat_rows": f,
        "n_keys": k,
        "n_users": len(users),
        "user_ids": users.astype(int).tolist(),
        "per_user_event_counts": per_user_counts,
        "pointer_event_counts": {str(key): int(value) for key, value in sorted(pointer_summary.items())},
        "key_touch_source_counts": {str(key): int(value) for key, value in sorted(source_summary.items())},
        "repeated_same_pointer_rows_within_frame": int(repeated_pointer_rows),
        "event_ids": data["event_id"].astype(np.int64),
        "user_external_pairs": set(zip(data["user_id"].astype(int).tolist(), data["user_external_id"].astype(int).tolist())),
        "raw_identities": raw_identities,
    }


def audit_complete(root: Path, expected_users: int) -> Dict[str, Any]:
    require(not (root / ".build").exists(), "finalized output must not retain .build")
    require(not (root / ".event_audit.csv.tmp").exists(), "finalized output retains temporary event audit")
    manifest_path = root / "manifest.json"
    audit_path = root / "audit.json"
    event_audit_path = root / "event_audit.csv"
    for path in (manifest_path, audit_path, event_audit_path):
        require(path.exists(), "missing finalized artifact: %s" % path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    extraction_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(manifest["schema_version"] == EXPECTED_SCHEMA, "manifest schema mismatch")
    require(manifest["script_version"] == EXPECTED_SCRIPT_VERSION, "manifest extractor version mismatch")
    require(extraction_audit["schema_version"] == EXPECTED_SCHEMA, "audit schema mismatch")
    require(extraction_audit["script_version"] == EXPECTED_SCRIPT_VERSION, "audit extractor version mismatch")
    require(extraction_audit["processed_user_count"] == expected_users, "processed user count mismatch")
    require(manifest["matching"]["allow_partial_keystroke"] is False, "partial keystroke must be disabled")
    require(manifest["matching"]["raw_timestamps_resampled"] is False, "raw timestamps were resampled")
    require(manifest["matching"]["pinch_requires_exactly_two_pointers"] is True, "pinch exact-two gate missing")
    require(manifest["matching"]["keystroke_fallback_interpolates_move_or_xy"] is False, "fallback interpolation claim invalid")
    require(manifest["matching"]["keystroke_touch_sources"] == {
        "0": "complete raw TouchEvent contact",
        "1": "observed OneFingerTouchEvent DOWN/UP endpoints",
    }, "keystroke source map mismatch")
    require(set(manifest["outputs"]) == set(ACTIONS), "manifest action outputs incomplete")

    action_results = {}
    all_event_ids: List[int] = []
    user_mapping: Dict[int, int] = {}
    raw_owner: Dict[Tuple[int, int, int], str] = {}
    for action in ACTIONS:
        path = root / ("hmog_trajectory_%s.npz" % action)
        require(path.exists(), "missing action NPZ: %s" % action)
        result = audit_action_npz(path, action, expected_users)
        summary = manifest["outputs"][action]
        for field in ("n_events", "n_flat_rows", "n_keys", "size_bytes", "sha256"):
            require(result[field] == summary[field], "manifest/%s mismatch: %s" % (action, field))
        require(Path(summary["path"]).resolve() == path.resolve(), "manifest output path mismatch")
        accepted = extraction_audit["action_status"][action]["accepted"]
        require(result["n_events"] == accepted, "audit accepted count mismatch: %s" % action)
        all_event_ids.extend(result["event_ids"].tolist())
        for user, external in result["user_external_pairs"]:
            previous = user_mapping.setdefault(user, external)
            require(previous == external, "user_id maps to multiple external IDs")
        for user, session, raw_id, owner in result["raw_identities"]:
            if raw_id < 0:
                continue
            key = (user, session, raw_id)
            require(key not in raw_owner, "raw gesture reused across actions: %r by %s/%s" % (key, raw_owner.get(key), owner))
            raw_owner[key] = owner
        del result["event_ids"], result["user_external_pairs"], result["raw_identities"]
        action_results[action] = result
    require(len(all_event_ids) == len(set(all_event_ids)), "event IDs are not globally unique")
    require(len(user_mapping) == expected_users, "global user mapping does not cover expected users")

    csv_counts: Dict[str, Counter] = {action: Counter() for action in ACTIONS}
    accepted_ids: Dict[str, set] = {action: set() for action in ACTIONS}
    with event_audit_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            action = row["action"]
            require(action in ACTIONS, "event audit contains unknown action")
            status = row["status"]
            require(status in ("accepted", "rejected"), "event audit status invalid")
            csv_counts[action][status] += 1
            if status == "accepted":
                accepted_ids[action].add(int(row["event_id"]))
    for action in ACTIONS:
        status = extraction_audit["action_status"][action]
        require(csv_counts[action]["accepted"] == status.get("accepted", 0), "event CSV accepted mismatch")
        require(csv_counts[action]["rejected"] == status.get("rejected", 0), "event CSV rejected mismatch")
        with np.load(root / ("hmog_trajectory_%s.npz" % action), allow_pickle=False) as data:
            require(accepted_ids[action] == set(data["event_id"].astype(int).tolist()), "accepted event IDs mismatch")

    warnings = []
    source = manifest.get("source", {})
    if "extractor_sha256" not in source:
        warnings.append("manifest does not cryptographically bind the extractor script (missing extractor_sha256)")
    if "hmog_zip_sha256" not in source:
        warnings.append("manifest records HMOG zip size but not HMOG zip SHA256")
    for field in ("event_preprocessor", "config"):
        path = Path(source[field])
        require(path.exists(), "manifest source path missing: %s" % path)
        require(sha256_file(path) == source[field + "_sha256"], "source hash drift: %s" % field)
    return {
        "schema_version": "extractor_v11_formal_audit_v1",
        "completion_state": "complete",
        "formal_passed": True,
        "source_dir": str(root),
        "expected_users": expected_users,
        "manifest_sha256": sha256_file(manifest_path),
        "audit_sha256": sha256_file(audit_path),
        "event_audit_sha256": sha256_file(event_audit_path),
        "actions": action_results,
        "global": {
            "n_unique_event_ids": len(all_event_ids),
            "n_users": len(user_mapping),
            "n_owned_raw_touch_gestures": len(raw_owner),
        },
        "warnings": warnings,
        "checks": [
            "numeric-only schema/dtypes and flat/event/key offsets",
            "global raw timestamp and frame invariants without resampling",
            "single-pointer canonical DOWN/MOVE/UP actions",
            "pinch exactly two unique pointers and two-pointer active frames",
            "keystroke complete discrete contacts, hold/flight/keycode/letter consistency",
            "OneFinger fallback source=1 uses exactly observed DOWN/UP and no MOVE",
            "100-user per-action coverage gate",
            "global event ID uniqueness and raw gesture mutual exclusion",
            "manifest hashes/counts and event-audit accepted/rejected IDs",
        ],
    }


def _binary_counts(directory: Path, specs: Mapping[str, str]) -> Dict[str, int]:
    counts = {}
    for name, dtype in specs.items():
        path = directory / (name + ".bin")
        if not path.exists():
            counts[name] = 0
            continue
        size = path.stat().st_size
        itemsize = np.dtype(dtype).itemsize
        require(size % itemsize == 0, "partial binary has torn element: %s" % path)
        counts[name] = size // itemsize
    return counts


def audit_in_progress(root: Path) -> Dict[str, Any]:
    build = root / ".build"
    require(build.is_dir(), "in-progress audit requires .build")
    actions = {}
    for action in ACTIONS:
        action_root = build / action
        flat_counts = _binary_counts(action_root / "flat", FLAT_SPECS)
        event_counts = _binary_counts(action_root / "events", EVENT_SPECS)
        key_counts = _binary_counts(action_root / "keys", KEY_SPECS)
        event_offset_path = action_root / "event_offsets.bin"
        key_event_offset_path = action_root / "event_key_offsets.bin"
        key_touch_offset_path = action_root / "key_touch_offsets.bin"
        offset_counts = {
            "event_offsets": event_offset_path.stat().st_size // 8 if event_offset_path.exists() else 0,
            "event_key_offsets": key_event_offset_path.stat().st_size // 8 if key_event_offset_path.exists() else 0,
            "key_touch_offsets": key_touch_offset_path.stat().st_size // 8 if key_touch_offset_path.exists() else 0,
        }
        stable_events = min(list(event_counts.values()) + [max(offset_counts["event_offsets"] - 1, 0), max(offset_counts["event_key_offsets"] - 1, 0)])
        stable_flat = min(flat_counts.values())
        stable_keys = min(list(key_counts.values()) + [max(offset_counts["key_touch_offsets"] - 1, 0)])
        user_count = 0
        if stable_events > 0:
            users = np.fromfile(action_root / "events" / "user_id.bin", dtype=EVENT_SPECS["user_id"], count=stable_events)
            user_count = len(np.unique(users))
        source_counts = {}
        if action == "keystroke" and stable_keys > 0:
            sources = np.fromfile(action_root / "keys" / "key_touch_source.bin", dtype=KEY_SPECS["key_touch_source"], count=stable_keys)
            source_counts = {str(int(value)): int(np.sum(sources == value)) for value in np.unique(sources)}
        actions[action] = {
            "stable_complete_event_prefix": int(stable_events),
            "stable_complete_flat_row_prefix": int(stable_flat),
            "stable_complete_key_prefix": int(stable_keys),
            "observed_user_count_in_stable_event_prefix": int(user_count),
            "key_touch_source_counts_in_stable_prefix": source_counts,
            "table_count_ranges": {
                "event": [int(min(event_counts.values())), int(max(event_counts.values()))],
                "flat": [int(min(flat_counts.values())), int(max(flat_counts.values()))],
                "key": [int(min(key_counts.values())), int(max(key_counts.values()))],
            },
            "offset_counts": offset_counts,
        }
    return {
        "schema_version": "extractor_v11_formal_audit_v1",
        "completion_state": "in_progress",
        "formal_passed": False,
        "source_dir": str(root),
        "actions": actions,
        "blocking_reason": "extractor has not finalized NPZ/manifest/audit; stable-prefix counts are progress evidence only",
        "warnings": ["do not train detectors from .build binaries or this partial audit"],
    }


def report_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# HMOG trajectory extractor v1.1.0 formal data audit",
        "",
        "- completion_state: `%s`" % result["completion_state"],
        "- formal_passed: `%s`" % str(result["formal_passed"]).lower(),
        "- source: `%s`" % result["source_dir"],
        "",
    ]
    if result["completion_state"] == "in_progress":
        lines.extend([
            "> 当前只审计 append-only binary table 的稳定前缀；不能作为 formal dataset。",
            "", "| action | stable events | stable flat rows | stable keys | observed users | key source 0/1 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for action in ACTIONS:
            row = result["actions"][action]
            sources = row["key_touch_source_counts_in_stable_prefix"]
            lines.append("| %s | %d | %d | %d | %d | %s / %s |" % (
                action, row["stable_complete_event_prefix"], row["stable_complete_flat_row_prefix"],
                row["stable_complete_key_prefix"], row["observed_user_count_in_stable_event_prefix"],
                sources.get("0", 0), sources.get("1", 0),
            ))
        lines.extend(["", "阻断：" + result["blocking_reason"], ""])
        return "\n".join(lines)

    lines.extend([
        "| action | events | flat rows | keys | users | key source 0/1 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for action in ACTIONS:
        row = result["actions"][action]
        sources = row["key_touch_source_counts"]
        lines.append("| %s | %d | %d | %d | %d | %s / %s |" % (
            action, row["n_events"], row["n_flat_rows"], row["n_keys"], row["n_users"],
            sources.get("0", 0), sources.get("1", 0),
        ))
    lines.extend(["", "## Checks", ""])
    for check in result["checks"]:
        lines.append("- PASS: " + check)
    if result.get("warnings"):
        lines.extend(["", "## Provenance warnings", ""])
        for warning in result["warnings"]:
            lines.append("- " + warning)
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--expected-users", type=int, default=100)
    parser.add_argument("--in-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_users <= 0:
        raise ValueError("expected-users must be positive")
    try:
        result = audit_in_progress(args.source_dir) if args.in_progress else audit_complete(args.source_dir, args.expected_users)
    except Exception as error:
        result = {
            "schema_version": "extractor_v11_formal_audit_v1",
            "completion_state": "in_progress" if args.in_progress else "complete_or_expected_complete",
            "formal_passed": False,
            "source_dir": str(args.source_dir),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _atomic_json(args.report_dir / "formal_data_audit.json", result)
        _atomic_text(args.report_dir / "formal_data_audit.md", "# Formal data audit\n\n- formal_passed: `false`\n- error: `%s`\n" % str(error).replace("`", "'"))
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(1)
    result["created_unix_time"] = time.time()
    _atomic_json(args.report_dir / "formal_data_audit.json", result)
    _atomic_text(args.report_dir / "formal_data_audit.md", report_markdown(result))
    print(json.dumps({
        "completion_state": result["completion_state"],
        "formal_passed": result["formal_passed"],
        "json": str(args.report_dir / "formal_data_audit.json"),
        "markdown": str(args.report_dir / "formal_data_audit.md"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
