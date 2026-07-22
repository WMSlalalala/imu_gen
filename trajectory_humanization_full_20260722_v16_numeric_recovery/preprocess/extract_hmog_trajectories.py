#!/usr/bin/env python3
"""Extract leakage-auditable HMOG touch trajectories for five action classes.

This module deliberately reuses the already-audited event construction and
mutual-exclusion logic in ``1_data_processing/preprocess.py``.  It adds a
second, independent layer: the derived event intervals are aligned back to
``TouchEvent.csv`` so the original irregular touch timeline is retained.  For
soft-keyboard contacts absent from that low-level stream, genuine observed
DOWN/UP endpoints from ``OneFingerTouchEvent.csv`` are used fail-closed.

The output is one numeric-only, flat+offset NPZ per action.  No object arrays
and no pickle files are written.  ``numpy.load(..., allow_pickle=False)`` is
therefore sufficient to read every output.

Important semantics
-------------------
* tap/scroll/swipe must map to a complete, single-pointer DOWN..UP contact;
* pinch must map to a complete contact containing exactly two pointers, while
  the phase-0..phase-2 interval remains available as ``flat_active_mask``;
* keystroke is a complete typing event made from discrete per-key contacts.
  Low-level TouchEvent is preferred; observed OneFinger DOWN/UP endpoints are
  the fallback. No fictitious MOVE or on-screen line is inserted;
* raw TouchEvent timestamps are never resampled.  HMOG touch samples are not
  assumed to be 100 Hz.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


SCHEMA_VERSION = "hmog_touch_trajectory_v1"
SCRIPT_VERSION = "1.1.1"
ALL_ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")

DEFAULT_HMOG_ZIP = Path("/home/mwang49/Human_agent/hmog_dataset.zip")
DEFAULT_EVENT_PREPROCESS = Path(
    "/home/mwang49/real-human/imu_gen/1_data_processing/preprocess.py"
)
DEFAULT_CONFIG = Path(
    "/home/mwang49/real-human/imu_gen/1_data_processing/config.yaml"
)

TOUCH_COLS = [
    "sys_t",
    "evt_t",
    "act_id",
    "pointer_count",
    "pointer_id",
    "action",
    "x",
    "y",
    "pressure",
    "size",
    "orient",
]

# HMOG records keyboard DOWN/UP endpoints in OneFingerTouchEvent even in many
# writing sessions where the lower-level TouchEvent stream omits soft-keyboard
# contacts.  These rows are genuine observed touch endpoints, not inferred
# keyboard centres.  They are used only as a fail-closed keystroke fallback.
ONE_FINGER_COLS = [
    "sys_t",
    "evt_t",
    "act_id",
    "tap_id",
    "tap_type",
    "action",
    "x",
    "y",
    "pressure",
    "size",
    "orient",
]

# Android MotionEvent actionMasked values used by HMOG.
ACTION_DOWN = 0
ACTION_UP = 1
ACTION_MOVE = 2
ACTION_CANCEL = 3
ACTION_POINTER_DOWN = 5
ACTION_POINTER_UP = 6
SUPPORTED_TOUCH_ACTIONS = {
    ACTION_DOWN,
    ACTION_UP,
    ACTION_MOVE,
    ACTION_CANCEL,
    ACTION_POINTER_DOWN,
    ACTION_POINTER_UP,
}


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_dump_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_event_preprocessor(path: Path):
    """Load the audited preprocessor by path without duplicating its logic."""

    if not path.is_file():
        raise FileNotFoundError(f"audited event preprocessor not found: {path}")
    module_name = "audited_hmog_event_preprocessor"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve the defining module through sys.modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    required = {
        "extract_all",
        "sessions",
        "event_identifier",
        "read_activity_meta_map",
        "ACTION_NAME_TO_ID",
        "ID_TO_NAME",
        "XY_FLOAT_FIELDS",
        "XY_INT_FIELDS",
        "is_alpha",
    }
    missing = sorted(name for name in required if not hasattr(module, name))
    if missing:
        raise AttributeError(
            f"audited event preprocessor is missing required symbols: {missing}"
        )
    return module


@dataclass(frozen=True)
class RawGesture:
    """A complete raw primary DOWN..UP contact from TouchEvent.csv."""

    gesture_id: int
    activity_id: int
    rows: pd.DataFrame

    @property
    def start_ms(self) -> int:
        return int(self.rows["evt_t"].iloc[0])

    @property
    def end_ms(self) -> int:
        return int(self.rows["evt_t"].iloc[-1])

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def max_pointer_count(self) -> int:
        return int(self.rows["pointer_count"].max())

    @property
    def n_pointers(self) -> int:
        return int(self.rows["pointer_id"].nunique())

    @property
    def n_frames(self) -> int:
        return int(self.rows["gesture_frame_index"].max()) + 1

    @property
    def orientations(self) -> tuple[int, ...]:
        values = sorted(set(int(v) for v in self.rows["orient"].tolist()))
        return tuple(values)


@dataclass(frozen=True)
class GestureMatch:
    raw: RawGesture
    start_error_ms: int
    end_error_ms: int
    score: float


def _numeric_touch_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[*TOUCH_COLS, "source_row"])
    x = df.copy()
    x["source_row"] = np.arange(len(x), dtype=np.int64)
    for column in TOUCH_COLS:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x = x.dropna(subset=TOUCH_COLS)
    if len(x) == 0:
        return pd.DataFrame(columns=[*TOUCH_COLS, "source_row"])
    integer_columns = [
        "sys_t",
        "evt_t",
        "act_id",
        "pointer_count",
        "pointer_id",
        "action",
        "orient",
        "source_row",
    ]
    for column in integer_columns:
        x[column] = x[column].astype(np.int64)
    for column in ["x", "y", "pressure", "size"]:
        x[column] = x[column].astype(np.float64)
    # EventTime is authoritative.  System time and file order only make the
    # order deterministic for the multiple pointer rows in one MotionEvent.
    return x.sort_values(
        ["evt_t", "sys_t", "source_row"], kind="stable"
    ).reset_index(drop=True)


def reconstruct_raw_gestures(
    touch_df: pd.DataFrame | None,
) -> tuple[list[RawGesture], dict[str, Any]]:
    """Reconstruct complete contacts from the raw Android event state machine."""

    x = _numeric_touch_frame(touch_df)
    stats: Counter[str] = Counter()
    unsupported_codes: Counter[int] = Counter()
    stats["rows"] = len(x)
    if len(x) == 0:
        return [], {"counts": dict(stats), "unsupported_action_codes": {}}

    current: dict[int, list[pd.DataFrame]] = {}
    gestures: list[RawGesture] = []
    next_id = 0

    # Rows with the same ActivityID and EventTime are one Android MotionEvent;
    # each active pointer appears as its own CSV row.
    for (activity_raw, event_time_raw), frame in x.groupby(
        ["act_id", "evt_t"], sort=False
    ):
        activity_id = int(activity_raw)
        event_time_ms = int(event_time_raw)
        del event_time_ms  # kept in frame; name documents the grouping key.
        stats["frames"] += 1
        action_codes = {int(v) for v in frame["action"].tolist()}
        for code in action_codes:
            if code not in SUPPORTED_TOUCH_ACTIONS:
                unsupported_codes[code] += 1
        if len(action_codes) > 1:
            stats["mixed_action_frames"] += 1

        has_down = ACTION_DOWN in action_codes
        has_up = ACTION_UP in action_codes
        has_cancel = ACTION_CANCEL in action_codes

        if has_down:
            if activity_id in current:
                stats["incomplete_restarted"] += 1
                del current[activity_id]
            current[activity_id] = []

        if activity_id not in current:
            stats["orphan_frames"] += 1
            continue

        current[activity_id].append(frame.copy())

        if has_cancel:
            stats["cancelled_gestures"] += 1
            del current[activity_id]
            continue

        if has_up:
            rows = pd.concat(current.pop(activity_id), ignore_index=True)
            # The concat order is frame order; make the per-gesture frame id
            # explicit so two-pointer rows can be reconstructed losslessly.
            # ``action`` is an event-level value copied onto every pointer row.
            # A few HMOG frames nevertheless contain mixed values, so EventTime
            # alone is the authoritative frame boundary.
            frame_keys = rows["evt_t"].tolist()
            frame_index: list[int] = []
            previous: int | None = None
            current_index = -1
            for key in frame_keys:
                key_int = int(key)
                if key_int != previous:
                    current_index += 1
                    previous = key_int
                frame_index.append(current_index)
            rows["gesture_frame_index"] = np.asarray(frame_index, dtype=np.int32)
            gestures.append(
                RawGesture(
                    gesture_id=next_id,
                    activity_id=activity_id,
                    rows=rows.reset_index(drop=True),
                )
            )
            next_id += 1
            stats["complete_gestures"] += 1

    if current:
        stats["incomplete_at_eof"] += len(current)

    return gestures, {
        "counts": {key: int(value) for key, value in sorted(stats.items())},
        "unsupported_action_codes": {
            str(key): int(value) for key, value in sorted(unsupported_codes.items())
        },
    }


def reconstruct_one_finger_contacts(
    one_finger_df: pd.DataFrame | None,
) -> tuple[list[RawGesture], dict[str, int]]:
    """Restore genuine OneFingerTouchEvent DOWN/UP endpoint contacts.

    Pairing is sequential per ActivityID on authoritative EventTime, matching
    the audited label preprocessor.  Returned gesture IDs are negative so they
    can never collide with raw TouchEvent gesture IDs.  Every accepted contact
    contains exactly two observed rows (DOWN and UP); no MOVE point or XY is
    interpolated.
    """

    stats: Counter[str] = Counter()
    if one_finger_df is None or len(one_finger_df) == 0:
        return [], {"one_finger_rows": 0, "one_finger_complete_contacts": 0}
    x = one_finger_df.copy()
    x["source_row"] = np.arange(len(x), dtype=np.int64)
    for column in ONE_FINGER_COLS:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x = x.dropna(subset=ONE_FINGER_COLS)
    stats["one_finger_rows"] = len(x)
    if len(x) == 0:
        return [], {key: int(value) for key, value in stats.items()}
    x = x.sort_values(["evt_t", "sys_t", "source_row"], kind="stable")
    opened: dict[int, Any] = {}
    contacts: list[RawGesture] = []
    next_id = -1_000_000_000
    for row in x.itertuples(index=False):
        activity_id = int(row.act_id)
        action = int(row.action)
        if action == ACTION_DOWN:
            if activity_id in opened:
                stats["one_finger_incomplete_restarted"] += 1
            opened[activity_id] = row
            continue
        if action != ACTION_UP:
            stats["one_finger_unsupported_action"] += 1
            continue
        if activity_id not in opened:
            stats["one_finger_orphan_up"] += 1
            continue
        down = opened.pop(activity_id)
        if int(row.evt_t) <= int(down.evt_t):
            stats["one_finger_nonpositive_contact"] += 1
            continue
        rows = pd.DataFrame(
            {
                "sys_t": [int(down.sys_t), int(row.sys_t)],
                "evt_t": [int(down.evt_t), int(row.evt_t)],
                "act_id": [activity_id, activity_id],
                "pointer_count": [1, 1],
                "pointer_id": [0, 0],
                "action": [ACTION_DOWN, ACTION_UP],
                "x": [float(down.x), float(row.x)],
                "y": [float(down.y), float(row.y)],
                "pressure": [float(down.pressure), float(row.pressure)],
                "size": [float(down.size), float(row.size)],
                "orient": [int(down.orient), int(row.orient)],
                "source_row": [int(down.source_row), int(row.source_row)],
                "gesture_frame_index": [0, 1],
            }
        )
        contacts.append(
            RawGesture(
                gesture_id=next_id,
                activity_id=activity_id,
                rows=rows,
            )
        )
        next_id -= 1
        stats["one_finger_complete_contacts"] += 1
    stats["one_finger_incomplete_at_eof"] += len(opened)
    return contacts, {key: int(value) for key, value in sorted(stats.items())}


def _primary_xy(raw: RawGesture, first: bool) -> tuple[float, float]:
    rows = raw.rows
    primary = rows[rows["pointer_id"] == 0]
    if len(primary) == 0:
        primary = rows
    row = primary.iloc[0 if first else -1]
    return float(row["x"]), float(row["y"])


def _event_xy_error(event: Any, raw: RawGesture) -> float:
    values = [
        event.meta.get("xy_start_x", np.nan),
        event.meta.get("xy_start_y", np.nan),
        event.meta.get("xy_end_x", np.nan),
        event.meta.get("xy_end_y", np.nan),
    ]
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        return 0.0
    raw_start = _primary_xy(raw, first=True)
    raw_end = _primary_xy(raw, first=False)
    start_error = float(
        np.hypot(raw_start[0] - float(values[0]), raw_start[1] - float(values[1]))
    )
    end_error = float(
        np.hypot(raw_end[0] - float(values[2]), raw_end[1] - float(values[3]))
    )
    return start_error + end_error


def _candidate_score(
    event: Any,
    raw: RawGesture,
    action: str,
    match_tolerance_ms: int,
    container_margin_ms: int,
) -> float | None:
    if int(event.activity_id) != raw.activity_id:
        return None
    start_error = abs(raw.start_ms - int(event.start_ms))
    end_error = abs(raw.end_ms - int(event.end_ms))

    if action in {"tap", "swipe"}:
        if start_error > match_tolerance_ms or end_error > match_tolerance_ms:
            return None
    elif action == "scroll":
        # ScrollEvent.CurrentTime usually stops at the last movement callback;
        # retain the complete raw contact through ACTION_UP.
        if start_error > match_tolerance_ms:
            return None
        if raw.end_ms < int(event.end_ms) - match_tolerance_ms:
            return None
        if raw.end_ms - int(event.end_ms) > container_margin_ms:
            return None
    elif action == "pinch":
        # Pinch phase 0/2 is a detector interval inside the complete two-finger
        # contact.  Preserve the one-finger lead-in/out instead of truncating it.
        if raw.start_ms > int(event.start_ms) + match_tolerance_ms:
            return None
        if raw.end_ms < int(event.end_ms) - match_tolerance_ms:
            return None
        before = max(0, int(event.start_ms) - raw.start_ms)
        after = max(0, raw.end_ms - int(event.end_ms))
        if before + after > 2 * container_margin_ms:
            return None
    else:
        raise ValueError(f"unsupported non-key action: {action}")

    # Time dominates. XY is a deterministic tie-breaker and never changes the
    # interval admissibility rule.
    return float(start_error + end_error) + 0.01 * _event_xy_error(event, raw)


def find_gesture_match(
    event: Any,
    action: str,
    gestures_by_activity: Mapping[int, Sequence[RawGesture]],
    used_gesture_ids: set[int],
    match_tolerance_ms: int,
    container_margin_ms: int,
) -> GestureMatch | None:
    candidates: list[tuple[float, RawGesture]] = []
    for raw in gestures_by_activity.get(int(event.activity_id), []):
        if raw.gesture_id in used_gesture_ids:
            continue
        score = _candidate_score(
            event,
            raw,
            action,
            match_tolerance_ms,
            container_margin_ms,
        )
        if score is not None:
            candidates.append((score, raw))
    if not candidates:
        return None
    score, raw = min(candidates, key=lambda item: (item[0], item[1].gesture_id))
    return GestureMatch(
        raw=raw,
        start_error_ms=raw.start_ms - int(event.start_ms),
        end_error_ms=raw.end_ms - int(event.end_ms),
        score=float(score),
    )


def validate_pointer_semantics(
    event: Any, action: str, raw: RawGesture
) -> str | None:
    if action in {"tap", "scroll", "swipe", "keystroke"}:
        if raw.max_pointer_count != 1 or raw.n_pointers != 1:
            return "single_pointer_action_contains_multitouch"
    elif action == "pinch":
        active = raw.rows[
            (raw.rows["evt_t"] >= int(event.start_ms))
            & (raw.rows["evt_t"] <= int(event.end_ms))
        ]
        if (
            raw.max_pointer_count != 2
            or raw.n_pointers != 2
            or len(active) == 0
            or int(active["pointer_count"].max()) != 2
            or int(active["pointer_id"].nunique()) != 2
        ):
            return "pinch_is_not_exactly_two_pointer_contact"
    return None


def validate_orientation(event: Any, raw: RawGesture) -> str | None:
    orientations = raw.orientations
    if len(orientations) != 1:
        return "orientation_changes_inside_contact"
    expected = int(event.orientation_id)
    if expected != -1 and orientations[0] != expected:
        return "event_touch_orientation_mismatch"
    return None


FLAT_SPECS: dict[str, str] = {
    "flat_system_time_ms": "<i8",
    "flat_event_time_ms": "<i8",
    "flat_t_rel_ms": "<i8",
    "flat_frame_index": "<i4",
    "flat_pointer_count": "i1",
    "flat_pointer_id": "<i2",
    "flat_action_code": "i1",
    "flat_x": "<f4",
    "flat_y": "<f4",
    "flat_pressure": "<f4",
    "flat_size": "<f4",
    "flat_orientation_id": "i1",
    "flat_active_mask": "u1",
    "flat_valid_mask": "u1",
    "flat_key_index": "<i4",
    "flat_keycode": "<i4",
}

EVENT_BASE_SPECS: dict[str, str] = {
    "event_id": "<i8",
    "user_id": "<i4",
    "user_external_id": "<i8",
    "session_id": "<i2",
    "action_id": "i1",
    "activity_id": "<i8",
    "orientation_id": "i1",
    "label_start_ms": "<i8",
    "label_end_ms": "<i8",
    "label_duration_ms": "<i4",
    "touch_start_ms": "<i8",
    "touch_end_ms": "<i8",
    "touch_duration_ms": "<i4",
    "active_start_rel_ms": "<i4",
    "active_end_rel_ms": "<i4",
    "n_rows": "<i4",
    "n_frames": "<i4",
    "n_pointers": "i1",
    "max_pointer_count": "i1",
    "active_row_count": "<i4",
    "raw_gesture_id": "<i4",
    "n_raw_gestures": "<i2",
    "match_start_error_ms": "<i4",
    "match_end_error_ms": "<i4",
    "n_keys": "<i2",
    "n_letters": "<i2",
    "motion_id_raw": "<i2",
    "posture_id": "i1",
    "task_id": "<i2",
    "activity_subtask_id": "<i2",
}

KEY_SPECS: dict[str, str] = {
    "key_index_in_event": "<i2",
    "keycode": "<i4",
    "key_is_letter": "u1",
    "key_down_ms": "<i8",
    "key_up_ms": "<i8",
    "key_hold_ms": "<i4",
    "key_flight_from_previous_ms": "<i4",
    "key_orientation_id": "i1",
    "key_raw_gesture_id": "<i4",
    "key_touch_start_ms": "<i8",
    "key_touch_end_ms": "<i8",
    "key_match_start_error_ms": "<i4",
    "key_match_end_error_ms": "<i4",
    "key_touch_found": "u1",
    # 0 = full raw TouchEvent contact; 1 = genuine observed
    # OneFingerTouchEvent DOWN/UP endpoint fallback.
    "key_touch_source": "u1",
}


class BinaryTable:
    """Append fixed-schema columns to raw files without retaining a full corpus."""

    def __init__(self, directory: Path, specs: Mapping[str, str]) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.specs = {name: np.dtype(dtype) for name, dtype in specs.items()}
        self.paths = {name: directory / f"{name}.bin" for name in self.specs}
        self.handles = {name: path.open("wb") for name, path in self.paths.items()}
        self.count = 0

    def append(self, columns: Mapping[str, Any], expected_count: int | None = None) -> int:
        arrays: dict[str, np.ndarray] = {}
        count: int | None = expected_count
        for name, dtype in self.specs.items():
            if name not in columns:
                raise KeyError(f"missing column {name}")
            array = np.asarray(columns[name], dtype=dtype).reshape(-1)
            if count is None:
                count = len(array)
            if len(array) != count:
                raise ValueError(
                    f"column length mismatch: {name}={len(array)} expected={count}"
                )
            arrays[name] = array
        if count is None:
            count = 0
        for name, array in arrays.items():
            array.tofile(self.handles[name])
        self.count += count
        return count

    def append_scalar(self, columns: Mapping[str, Any]) -> None:
        self.append({name: [value] for name, value in columns.items()}, expected_count=1)

    def close(self) -> None:
        for handle in self.handles.values():
            if not handle.closed:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()

    def arrays(self) -> dict[str, np.ndarray]:
        self.close()
        result: dict[str, np.ndarray] = {}
        for name, dtype in self.specs.items():
            if self.count == 0:
                result[name] = np.empty(0, dtype=dtype)
            else:
                result[name] = np.memmap(
                    self.paths[name], mode="r", dtype=dtype, shape=(self.count,)
                )
        return result


class ActionWriter:
    def __init__(
        self,
        action: str,
        action_id: int,
        output_path: Path,
        build_root: Path,
        float_meta_fields: Sequence[str],
        int_meta_fields: Sequence[str],
    ) -> None:
        self.action = action
        self.action_id = int(action_id)
        self.output_path = output_path
        self.build_dir = build_root / action
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.build_dir.mkdir(parents=True)

        event_specs = dict(EVENT_BASE_SPECS)
        for field in float_meta_fields:
            event_specs[f"meta_{field}"] = "<f4"
        for field in int_meta_fields:
            event_specs[f"meta_{field}"] = "<i4"
        self.float_meta_fields = tuple(float_meta_fields)
        self.int_meta_fields = tuple(int_meta_fields)
        self.flat = BinaryTable(self.build_dir / "flat", FLAT_SPECS)
        self.events = BinaryTable(self.build_dir / "events", event_specs)
        self.keys = BinaryTable(self.build_dir / "keys", KEY_SPECS)
        self.event_offsets_path = self.build_dir / "event_offsets.bin"
        self.event_key_offsets_path = self.build_dir / "event_key_offsets.bin"
        self.key_touch_offsets_path = self.build_dir / "key_touch_offsets.bin"
        self.event_offsets_handle = self.event_offsets_path.open("wb")
        self.event_key_offsets_handle = self.event_key_offsets_path.open("wb")
        self.key_touch_offsets_handle = self.key_touch_offsets_path.open("wb")
        np.asarray([0], dtype="<i8").tofile(self.event_offsets_handle)
        np.asarray([0], dtype="<i8").tofile(self.event_key_offsets_handle)
        np.asarray([0], dtype="<i8").tofile(self.key_touch_offsets_handle)
        self.event_offset_count = 1
        self.event_key_offset_count = 1
        self.key_touch_offset_count = 1
        self._keys_touch_total = 0

    def append_event(
        self,
        event_columns: Mapping[str, Any],
        flat_columns: Mapping[str, Any],
        key_columns: Sequence[Mapping[str, Any]],
        key_flat_lengths: Sequence[int],
    ) -> None:
        flat_count = self.flat.append(flat_columns)
        self.events.append_scalar(event_columns)
        if len(key_columns) != len(key_flat_lengths):
            raise ValueError("key records and key flat lengths differ")
        for key_record, key_flat_length in zip(key_columns, key_flat_lengths):
            self.keys.append_scalar(key_record)
            np.asarray([self.keys_touch_total + int(key_flat_length)], dtype="<i8").tofile(
                self.key_touch_offsets_handle
            )
            self.keys_touch_total += int(key_flat_length)
            self.key_touch_offset_count += 1

        np.asarray([self.flat.count], dtype="<i8").tofile(self.event_offsets_handle)
        np.asarray([self.keys.count], dtype="<i8").tofile(
            self.event_key_offsets_handle
        )
        self.event_offset_count += 1
        self.event_key_offset_count += 1

        if flat_count != int(event_columns["n_rows"]):
            raise ValueError("event n_rows does not match appended flat rows")

    @property
    def keys_touch_total(self) -> int:
        return getattr(self, "_keys_touch_total", 0)

    @keys_touch_total.setter
    def keys_touch_total(self, value: int) -> None:
        self._keys_touch_total = int(value)

    def _close_offsets(self) -> None:
        for handle in [
            self.event_offsets_handle,
            self.event_key_offsets_handle,
            self.key_touch_offsets_handle,
        ]:
            if not handle.closed:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()

    @staticmethod
    def _offset_array(path: Path, count: int) -> np.ndarray:
        return np.memmap(path, mode="r", dtype="<i8", shape=(count,))

    def finalize(self) -> dict[str, Any]:
        self._close_offsets()
        arrays: dict[str, Any] = {}
        arrays.update(self.flat.arrays())
        arrays.update(self.events.arrays())
        arrays.update(self.keys.arrays())
        arrays["event_offsets"] = self._offset_array(
            self.event_offsets_path, self.event_offset_count
        )
        arrays["event_key_offsets"] = self._offset_array(
            self.event_key_offsets_path, self.event_key_offset_count
        )
        arrays["key_touch_offsets"] = self._offset_array(
            self.key_touch_offsets_path, self.key_touch_offset_count
        )
        arrays["schema_version"] = np.asarray(SCHEMA_VERSION)
        arrays["action_name"] = np.asarray(self.action)
        arrays["action_id_scalar"] = np.asarray(self.action_id, dtype=np.int8)
        arrays["time_unit"] = np.asarray("ms")
        arrays["coordinate_unit"] = np.asarray("raw_screen_pixel")
        arrays["sampling"] = np.asarray("raw_irregular_touch_events")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.output_path.name}.",
            suffix=".tmp",
            dir=str(self.output_path.parent),
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary_name, self.output_path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

        summary = {
            "action": self.action,
            "action_id": self.action_id,
            "n_events": int(self.events.count),
            "n_flat_rows": int(self.flat.count),
            "n_keys": int(self.keys.count),
            "path": str(self.output_path),
            "size_bytes": int(self.output_path.stat().st_size),
            "sha256": _sha256_file(self.output_path),
        }
        # Release memmaps before deleting their files (important on non-POSIX
        # platforms; harmless here).
        arrays.clear()
        shutil.rmtree(self.build_dir)
        return summary

    def abort(self) -> None:
        self.flat.close()
        self.events.close()
        self.keys.close()
        self._close_offsets()


AUDIT_FIELDS = [
    "user_id",
    "user_external_id",
    "session_id",
    "action",
    "event_id",
    "activity_id",
    "label_start_ms",
    "label_end_ms",
    "status",
    "reason",
    "raw_gesture_id",
    "match_start_error_ms",
    "match_end_error_ms",
    "raw_max_pointer_count",
    "raw_n_pointers",
    "raw_n_rows",
    "n_keys",
    "n_keys_matched",
]


class AuditWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = path.with_name(f".{path.name}.tmp")
        self.handle = self.temporary_path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=AUDIT_FIELDS)
        self.writer.writeheader()
        self.action_status: dict[str, Counter[str]] = defaultdict(Counter)
        self.action_reasons: dict[str, Counter[str]] = defaultdict(Counter)
        self.raw_parser_totals: Counter[str] = Counter()
        self.raw_unsupported_codes: Counter[str] = Counter()
        self.sessions: list[dict[str, Any]] = []

    def add_event(self, row: Mapping[str, Any]) -> None:
        normalized = {field: row.get(field, "") for field in AUDIT_FIELDS}
        self.writer.writerow(normalized)
        action = str(normalized["action"])
        status = str(normalized["status"])
        reason = str(normalized["reason"])
        self.action_status[action][status] += 1
        if reason:
            self.action_reasons[action][reason] += 1

    def add_raw_session(
        self,
        user_id: int,
        user_external_id: int,
        session_id: int,
        raw_audit: Mapping[str, Any],
    ) -> None:
        counts = {str(k): int(v) for k, v in raw_audit.get("counts", {}).items()}
        unsupported = {
            str(k): int(v)
            for k, v in raw_audit.get("unsupported_action_codes", {}).items()
        }
        self.raw_parser_totals.update(counts)
        self.raw_unsupported_codes.update(unsupported)
        self.sessions.append(
            {
                "user_id": int(user_id),
                "user_external_id": int(user_external_id),
                "session_id": int(session_id),
                "raw_touch": counts,
                "unsupported_action_codes": unsupported,
            }
        )

    def finalize(self) -> dict[str, Any]:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        os.replace(self.temporary_path, self.path)
        return {
            "action_status": {
                action: {k: int(v) for k, v in sorted(counter.items())}
                for action, counter in sorted(self.action_status.items())
            },
            "rejection_reasons": {
                action: {k: int(v) for k, v in sorted(counter.items())}
                for action, counter in sorted(self.action_reasons.items())
            },
            "raw_parser_totals": {
                k: int(v) for k, v in sorted(self.raw_parser_totals.items())
            },
            "raw_unsupported_action_codes": {
                k: int(v) for k, v in sorted(self.raw_unsupported_codes.items())
            },
            "sessions": self.sessions,
            "event_audit_csv": str(self.path),
        }

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()


def raw_to_flat_columns(
    raw: RawGesture,
    label_start_ms: int,
    label_end_ms: int,
    key_index: int = -1,
    keycode: int = -1,
    relative_origin_ms: int | None = None,
) -> dict[str, np.ndarray]:
    rows = raw.rows.sort_values(
        ["gesture_frame_index", "pointer_id", "source_row"], kind="stable"
    ).reset_index(drop=True)
    event_times = rows["evt_t"].to_numpy(np.int64)
    origin = raw.start_ms if relative_origin_ms is None else int(relative_origin_ms)
    active = (event_times >= int(label_start_ms)) & (event_times <= int(label_end_ms))
    n = len(rows)
    return {
        "flat_system_time_ms": rows["sys_t"].to_numpy(np.int64),
        "flat_event_time_ms": event_times,
        "flat_t_rel_ms": event_times - origin,
        "flat_frame_index": rows["gesture_frame_index"].to_numpy(np.int32),
        "flat_pointer_count": rows["pointer_count"].to_numpy(np.int8),
        "flat_pointer_id": rows["pointer_id"].to_numpy(np.int16),
        "flat_action_code": rows["action"].to_numpy(np.int8),
        "flat_x": rows["x"].to_numpy(np.float32),
        "flat_y": rows["y"].to_numpy(np.float32),
        "flat_pressure": rows["pressure"].to_numpy(np.float32),
        "flat_size": rows["size"].to_numpy(np.float32),
        "flat_orientation_id": rows["orient"].to_numpy(np.int8),
        "flat_active_mask": active.astype(np.uint8),
        "flat_valid_mask": np.ones(n, dtype=np.uint8),
        "flat_key_index": np.full(n, key_index, dtype=np.int32),
        "flat_keycode": np.full(n, keycode, dtype=np.int32),
    }


def concatenate_flat(parts: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        return {
            name: np.empty(0, dtype=np.dtype(dtype)) for name, dtype in FLAT_SPECS.items()
        }
    result: dict[str, np.ndarray] = {}
    for name, dtype in FLAT_SPECS.items():
        result[name] = np.concatenate(
            [np.asarray(part[name], dtype=np.dtype(dtype)) for part in parts]
        )
    return result


def build_event_columns(
    dp: Any,
    event: Any,
    action: str,
    event_id: int,
    user_id: int,
    user_external_id: int,
    session_id: int,
    touch_start_ms: int,
    touch_end_ms: int,
    n_rows: int,
    n_frames: int,
    n_pointers: int,
    max_pointer_count: int,
    active_row_count: int,
    raw_gesture_id: int,
    n_raw_gestures: int,
    match_start_error_ms: int,
    match_end_error_ms: int,
    n_keys: int,
    n_letters: int,
    activity_meta: Mapping[int, tuple[int, int, int, int]],
) -> dict[str, Any]:
    motion_raw, posture_id, task_id, content_id = activity_meta.get(
        int(event.activity_id), (0, -1, -1, -1)
    )
    values: dict[str, Any] = {
        "event_id": event_id,
        "user_id": user_id,
        "user_external_id": user_external_id,
        "session_id": session_id,
        "action_id": int(dp.ACTION_NAME_TO_ID[action]),
        "activity_id": int(event.activity_id),
        "orientation_id": int(event.orientation_id),
        "label_start_ms": int(event.start_ms),
        "label_end_ms": int(event.end_ms),
        "label_duration_ms": int(event.end_ms - event.start_ms),
        "touch_start_ms": int(touch_start_ms),
        "touch_end_ms": int(touch_end_ms),
        "touch_duration_ms": int(touch_end_ms - touch_start_ms),
        "active_start_rel_ms": int(event.start_ms - touch_start_ms),
        "active_end_rel_ms": int(event.end_ms - touch_start_ms),
        "n_rows": int(n_rows),
        "n_frames": int(n_frames),
        "n_pointers": int(n_pointers),
        "max_pointer_count": int(max_pointer_count),
        "active_row_count": int(active_row_count),
        "raw_gesture_id": int(raw_gesture_id),
        "n_raw_gestures": int(n_raw_gestures),
        "match_start_error_ms": int(match_start_error_ms),
        "match_end_error_ms": int(match_end_error_ms),
        "n_keys": int(n_keys),
        "n_letters": int(n_letters),
        "motion_id_raw": int(motion_raw),
        "posture_id": int(posture_id),
        "task_id": int(task_id),
        "activity_subtask_id": int(content_id),
    }
    for field in dp.XY_FLOAT_FIELDS:
        values[f"meta_{field}"] = float(event.meta.get(field, np.nan))
    for field in dp.XY_INT_FIELDS:
        values[f"meta_{field}"] = int(event.meta.get(field, -1))
    return values


def _audit_event_row(
    *,
    user_id: int,
    user_external_id: int,
    session_id: int,
    action: str,
    event_id: int,
    event: Any,
    status: str,
    reason: str,
    match: GestureMatch | None = None,
    n_keys: int = 0,
    n_keys_matched: int = 0,
) -> dict[str, Any]:
    raw = match.raw if match is not None else None
    return {
        "user_id": user_id,
        "user_external_id": user_external_id,
        "session_id": session_id,
        "action": action,
        "event_id": event_id,
        "activity_id": int(event.activity_id),
        "label_start_ms": int(event.start_ms),
        "label_end_ms": int(event.end_ms),
        "status": status,
        "reason": reason,
        "raw_gesture_id": raw.gesture_id if raw is not None else -1,
        "match_start_error_ms": match.start_error_ms if match is not None else "",
        "match_end_error_ms": match.end_error_ms if match is not None else "",
        "raw_max_pointer_count": raw.max_pointer_count if raw is not None else "",
        "raw_n_pointers": raw.n_pointers if raw is not None else "",
        "raw_n_rows": len(raw.rows) if raw is not None else "",
        "n_keys": n_keys,
        "n_keys_matched": n_keys_matched,
    }


def process_non_key_event(
    *,
    dp: Any,
    writer: ActionWriter | None,
    audit: AuditWriter,
    event: Any,
    action: str,
    event_id: int,
    user_id: int,
    user_external_id: int,
    session_id: int,
    activity_meta: Mapping[int, tuple[int, int, int, int]],
    gestures_by_activity: Mapping[int, Sequence[RawGesture]],
    used_gesture_ids: set[int],
    match_tolerance_ms: int,
    container_margin_ms: int,
) -> None:
    match = find_gesture_match(
        event,
        action,
        gestures_by_activity,
        used_gesture_ids,
        match_tolerance_ms,
        container_margin_ms,
    )
    if match is None:
        # Distinguish a truly missing raw contact from a lower-priority derived
        # interval that is merely a prefix/suffix of a physical contact already
        # reserved by pinch/swipe/keystroke.  Both are rejected, but the audit
        # reason is materially different.
        reserved_match = find_gesture_match(
            event,
            action,
            gestures_by_activity,
            set(),
            match_tolerance_ms,
            container_margin_ms,
        )
        reason = (
            "raw_contact_reserved_by_higher_priority_event"
            if reserved_match is not None
            and reserved_match.raw.gesture_id in used_gesture_ids
            else "no_matching_complete_raw_contact"
        )
        audit.add_event(
            _audit_event_row(
                user_id=user_id,
                user_external_id=user_external_id,
                session_id=session_id,
                action=action,
                event_id=event_id,
                event=event,
                status="rejected",
                reason=reason,
                match=reserved_match,
            )
        )
        return

    # Reserve the physical contact even when a semantic validation rejects it;
    # otherwise another label could silently reuse the same contaminated trace.
    used_gesture_ids.add(match.raw.gesture_id)
    reason = validate_pointer_semantics(event, action, match.raw)
    if reason is None:
        reason = validate_orientation(event, match.raw)
    if reason is not None:
        audit.add_event(
            _audit_event_row(
                user_id=user_id,
                user_external_id=user_external_id,
                session_id=session_id,
                action=action,
                event_id=event_id,
                event=event,
                status="rejected",
                reason=reason,
                match=match,
            )
        )
        return

    flat = raw_to_flat_columns(
        match.raw,
        int(event.start_ms),
        int(event.end_ms),
        relative_origin_ms=match.raw.start_ms,
    )
    n_rows = len(flat["flat_event_time_ms"])
    active_count = int(np.asarray(flat["flat_active_mask"]).sum())
    event_columns = build_event_columns(
        dp,
        event,
        action,
        event_id,
        user_id,
        user_external_id,
        session_id,
        match.raw.start_ms,
        match.raw.end_ms,
        n_rows,
        match.raw.n_frames,
        match.raw.n_pointers,
        match.raw.max_pointer_count,
        active_count,
        match.raw.gesture_id,
        1,
        match.start_error_ms,
        match.end_error_ms,
        0,
        0,
        activity_meta,
    )
    if writer is not None:
        writer.append_event(event_columns, flat, [], [])
    audit.add_event(
        _audit_event_row(
            user_id=user_id,
            user_external_id=user_external_id,
            session_id=session_id,
            action=action,
            event_id=event_id,
            event=event,
            status="accepted" if writer is not None else "reserved_not_emitted",
            reason="" if writer is not None else "requested_action_filter",
            match=match,
        )
    )


def process_keystroke_event(
    *,
    dp: Any,
    writer: ActionWriter | None,
    audit: AuditWriter,
    event: Any,
    event_id: int,
    user_id: int,
    user_external_id: int,
    session_id: int,
    activity_meta: Mapping[int, tuple[int, int, int, int]],
    gestures_by_activity: Mapping[int, Sequence[RawGesture]],
    one_finger_by_activity: Mapping[int, Sequence[RawGesture]],
    used_gesture_ids: set[int],
    used_one_finger_ids: set[int],
    match_tolerance_ms: int,
    allow_partial: bool,
) -> None:
    keys = sorted(event.meta.get("keys", []), key=lambda key: (key.down_ms, key.up_ms))
    matched: list[tuple[Any, GestureMatch | None, str | None, int]] = []
    for key in keys:
        # A key is itself a single-pointer down/up contact.  Reuse the generic
        # exact interval matcher through a minimal event-like proxy.
        class KeyProxy:
            pass

        proxy = KeyProxy()
        proxy.start_ms = int(key.down_ms)
        proxy.end_ms = int(key.up_ms)
        proxy.activity_id = int(key.activity_id)
        proxy.orientation_id = int(key.orientation_id)
        proxy.meta = {}
        raw_match = find_gesture_match(
            proxy,
            "tap",
            gestures_by_activity,
            used_gesture_ids,
            match_tolerance_ms,
            match_tolerance_ms,
        )
        reason: str | None = None
        source_code = 0
        match = raw_match
        if raw_match is not None:
            used_gesture_ids.add(raw_match.raw.gesture_id)
            reason = validate_pointer_semantics(proxy, "keystroke", raw_match.raw)
            if reason is None:
                reason = validate_orientation(proxy, raw_match.raw)

        # Many HMOG writing sessions contain exact soft-keyboard DOWN/UP rows
        # only in OneFingerTouchEvent.csv.  Fall back to those observed
        # endpoints when the lower-level TouchEvent contact is absent or
        # invalid.  No MOVE point or keyboard-centre coordinate is invented.
        if match is None or reason is not None:
            fallback = find_gesture_match(
                proxy,
                "tap",
                one_finger_by_activity,
                used_one_finger_ids,
                match_tolerance_ms,
                match_tolerance_ms,
            )
            if fallback is not None:
                used_one_finger_ids.add(fallback.raw.gesture_id)
                fallback_reason = validate_pointer_semantics(
                    proxy, "keystroke", fallback.raw
                )
                if fallback_reason is None:
                    fallback_reason = validate_orientation(proxy, fallback.raw)
                if fallback_reason is None:
                    match = fallback
                    reason = None
                    source_code = 1
                elif reason is None:
                    reason = fallback_reason
            elif match is None:
                reason = "key_has_no_matching_raw_or_one_finger_contact"
        matched.append((key, match, reason, source_code))

    valid_matches = [
        (key, match, source_code)
        for key, match, reason, source_code in matched
        if match is not None and reason is None
    ]
    if len(valid_matches) != len(keys) and not allow_partial:
        reasons = Counter(
            reason or "unknown"
            for _, _, reason, _ in matched
            if reason is not None
        )
        audit.add_event(
            _audit_event_row(
                user_id=user_id,
                user_external_id=user_external_id,
                session_id=session_id,
                action="keystroke",
                event_id=event_id,
                event=event,
                status="rejected",
                reason="incomplete_key_touch_alignment:" + ";".join(
                    f"{name}={count}" for name, count in sorted(reasons.items())
                ),
                n_keys=len(keys),
                n_keys_matched=len(valid_matches),
            )
        )
        return
    if not valid_matches:
        audit.add_event(
            _audit_event_row(
                user_id=user_id,
                user_external_id=user_external_id,
                session_id=session_id,
                action="keystroke",
                event_id=event_id,
                event=event,
                status="rejected",
                reason="typing_event_has_zero_aligned_keys",
                n_keys=len(keys),
                n_keys_matched=0,
            )
        )
        return

    # Points are emitted key-by-key.  The EventTime gaps between successive
    # contacts retain flight timing, but there are deliberately no invented XY
    # samples while the finger is off screen.
    event_origin = int(valid_matches[0][1].raw.start_ms)
    flat_parts: list[dict[str, np.ndarray]] = []
    key_records: list[dict[str, Any]] = []
    key_flat_lengths: list[int] = []
    previous_up: int | None = None
    n_frames = 0
    n_letters = 0
    start_errors: list[int] = []
    end_errors: list[int] = []
    for key_index, (key, match, source_code) in enumerate(valid_matches):
        assert match is not None
        part = raw_to_flat_columns(
            match.raw,
            int(key.down_ms),
            int(key.up_ms),
            key_index=key_index,
            keycode=int(key.keycode),
            relative_origin_ms=event_origin,
        )
        # Frame indices must be monotonic over the complete typing event.
        part["flat_frame_index"] = part["flat_frame_index"] + n_frames
        n_frames += match.raw.n_frames
        flat_parts.append(part)
        key_flat_lengths.append(len(part["flat_event_time_ms"]))
        is_letter = bool(dp.is_alpha(int(key.keycode)))
        n_letters += int(is_letter)
        flight = 0 if previous_up is None else int(key.down_ms - previous_up)
        previous_up = int(key.up_ms)
        start_errors.append(match.start_error_ms)
        end_errors.append(match.end_error_ms)
        key_records.append(
            {
                "key_index_in_event": key_index,
                "keycode": int(key.keycode),
                "key_is_letter": int(is_letter),
                "key_down_ms": int(key.down_ms),
                "key_up_ms": int(key.up_ms),
                "key_hold_ms": int(key.up_ms - key.down_ms),
                "key_flight_from_previous_ms": flight,
                "key_orientation_id": int(key.orientation_id),
                "key_raw_gesture_id": match.raw.gesture_id,
                "key_touch_start_ms": match.raw.start_ms,
                "key_touch_end_ms": match.raw.end_ms,
                "key_match_start_error_ms": match.start_error_ms,
                "key_match_end_error_ms": match.end_error_ms,
                "key_touch_found": 1,
                "key_touch_source": source_code,
            }
        )

    flat = concatenate_flat(flat_parts)
    touch_start = int(valid_matches[0][1].raw.start_ms)
    touch_end = int(valid_matches[-1][1].raw.end_ms)
    n_rows = len(flat["flat_event_time_ms"])
    event_columns = build_event_columns(
        dp,
        event,
        "keystroke",
        event_id,
        user_id,
        user_external_id,
        session_id,
        touch_start,
        touch_end,
        n_rows,
        n_frames,
        1,
        1,
        int(np.asarray(flat["flat_active_mask"]).sum()),
        -1,
        len(valid_matches),
        int(round(float(np.mean(start_errors)))),
        int(round(float(np.mean(end_errors)))),
        len(valid_matches),
        n_letters,
        activity_meta,
    )
    if writer is not None:
        writer.append_event(event_columns, flat, key_records, key_flat_lengths)
    audit.add_event(
        _audit_event_row(
            user_id=user_id,
            user_external_id=user_external_id,
            session_id=session_id,
            action="keystroke",
            event_id=event_id,
            event=event,
            status="accepted" if writer is not None else "reserved_not_emitted",
            reason=(
                "partial_alignment_allowed"
                if writer is not None and len(valid_matches) != len(keys)
                else "requested_action_filter"
                if writer is None
                else ""
            ),
            n_keys=len(keys),
            n_keys_matched=len(valid_matches),
        )
    )


def parse_actions(value: str) -> tuple[str, ...]:
    actions = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    unknown = sorted(set(actions) - set(ALL_ACTIONS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown actions: {unknown}")
    if not actions:
        raise argparse.ArgumentTypeError("at least one action is required")
    return actions


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract raw HMOG trajectories aligned to audited five-action events."
    )
    parser.add_argument("--hmog-zip", type=Path, default=DEFAULT_HMOG_ZIP)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--event-preprocess", type=Path, default=DEFAULT_EVENT_PREPROCESS
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "trajectories",
    )
    parser.add_argument("--actions", type=parse_actions, default=ALL_ACTIONS)
    parser.add_argument(
        "--max-users",
        type=int,
        default=1,
        help="Safety default is one user. Use 100 with --confirm-full-run for full data.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=0,
        help="0 keeps all sessions of each selected user.",
    )
    parser.add_argument(
        "--user-external-id",
        type=int,
        default=None,
        help="Optionally process one exact HMOG external user id.",
    )
    parser.add_argument("--match-tolerance-ms", type=int, default=25)
    parser.add_argument("--container-margin-ms", type=int, default=1500)
    parser.add_argument("--allow-partial-keystroke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    started_at = time.time()
    actions = tuple(args.actions)
    if args.max_users <= 0:
        raise ValueError("--max-users must be positive")
    if args.max_users >= 100 and not args.confirm_full_run:
        raise ValueError("a 100-user run requires explicit --confirm-full-run")
    if args.match_tolerance_ms < 0 or args.container_margin_ms < 0:
        raise ValueError("matching tolerances must be non-negative")
    if not args.hmog_zip.is_file():
        raise FileNotFoundError(args.hmog_zip)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        action: output_dir / f"hmog_trajectory_{action}.npz" for action in actions
    }
    control_paths = [
        output_dir / "manifest.json",
        output_dir / "audit.json",
        output_dir / "event_audit.csv",
    ]
    existing = [path for path in [*output_paths.values(), *control_paths] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "output exists; pass --overwrite: " + ", ".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    config_text = args.config.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)["data"]
    gesture_cfg = dict(config["gesture"])
    key_cfg = dict(config["keystroke"])
    dp = load_event_preprocessor(args.event_preprocess.resolve())
    preprocessor_sha256 = _sha256_file(args.event_preprocess.resolve())

    build_root = output_dir / ".build"
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)
    writers = {
        action: ActionWriter(
            action,
            int(dp.ACTION_NAME_TO_ID[action]),
            output_paths[action],
            build_root,
            dp.XY_FLOAT_FIELDS,
            dp.XY_INT_FIELDS,
        )
        for action in actions
    }
    audit = AuditWriter(output_dir / "event_audit.csv")

    outer = zipfile.ZipFile(args.hmog_zip)
    all_user_members = sorted(
        name
        for name in outer.namelist()
        if name.startswith("public_dataset/") and name.endswith(".zip")
    )
    indexed_members: list[tuple[int, str, int]] = []
    for global_user_id, member in enumerate(all_user_members):
        external_id = int(Path(member).stem)
        if args.user_external_id is not None and external_id != args.user_external_id:
            continue
        indexed_members.append((global_user_id, member, external_id))
    if args.user_external_id is not None and not indexed_members:
        raise ValueError(f"user {args.user_external_id} is not present in HMOG archive")
    selected_members = indexed_members[: int(args.max_users)]

    processed_users: list[dict[str, Any]] = []
    session_count = 0
    try:
        for selected_index, (user_id, member, external_id) in enumerate(selected_members):
            print(
                f"[user {selected_index + 1}/{len(selected_members)}] "
                f"user_id={user_id} external={external_id}",
                flush=True,
            )
            inner_bytes = outer.read(member)
            user_session_count = 0
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as zf:
                # ``preprocess.sessions`` intentionally mirrors the original
                # HMOG preprocessor, but a few user archives also contain
                # ``__MACOSX/<user>/<user>_session_*`` resource-fork entries.
                # Its broad directory scan can turn those entries into a
                # bogus session named only ``<user>``.  Require the exact
                # dataset root ``<user>/<session>/`` here so empty metadata
                # directories cannot inflate the auditable session count.
                archive_names = tuple(zf.namelist())
                discovered_session_items = dp.sessions(zf)
                session_items = [
                    (session_name, session_id)
                    for session_name, session_id in discovered_session_items
                    if any(
                        name.startswith(f"{external_id}/{session_name}/")
                        for name in archive_names
                    )
                ]
                ignored_session_items = [
                    session_name
                    for session_name, session_id in discovered_session_items
                    if (session_name, session_id) not in session_items
                ]
                if args.max_sessions > 0:
                    session_items = session_items[: int(args.max_sessions)]
                for session_name, session_id in session_items:
                    base = f"{external_id}/{session_name}"
                    touch_df = dp.read_csv(
                        zf, f"{base}/TouchEvent.csv", TOUCH_COLS
                    )
                    one_finger_df = dp.read_csv(
                        zf, f"{base}/OneFingerTouchEvent.csv", ONE_FINGER_COLS
                    )
                    raw_gestures, raw_audit = reconstruct_raw_gestures(touch_df)
                    one_finger_contacts, one_finger_audit = (
                        reconstruct_one_finger_contacts(one_finger_df)
                    )
                    raw_audit.setdefault("counts", {}).update(one_finger_audit)
                    audit.add_raw_session(
                        user_id, external_id, session_id, raw_audit
                    )
                    gestures_by_activity: dict[int, list[RawGesture]] = defaultdict(list)
                    for raw in raw_gestures:
                        gestures_by_activity[raw.activity_id].append(raw)
                    for values in gestures_by_activity.values():
                        values.sort(key=lambda raw: (raw.start_ms, raw.end_ms, raw.gesture_id))
                    one_finger_by_activity: dict[int, list[RawGesture]] = defaultdict(list)
                    for raw in one_finger_contacts:
                        one_finger_by_activity[raw.activity_id].append(raw)
                    for values in one_finger_by_activity.values():
                        values.sort(
                            key=lambda raw: (raw.start_ms, raw.end_ms, raw.gesture_id)
                        )

                    # Always construct all five labels before filtering outputs.
                    # Passing only requested actions would weaken exclusivity.
                    events = dp.extract_all(
                        zf,
                        str(external_id),
                        session_name,
                        set(ALL_ACTIONS),
                        gesture_cfg,
                        key_cfg,
                    )
                    activity_meta = dp.read_activity_meta_map(
                        zf, str(external_id), session_name
                    )
                    by_action: dict[str, list[Any]] = defaultdict(list)
                    for event in events:
                        by_action[str(dp.ID_TO_NAME[int(event.action_id)])].append(event)
                    for values in by_action.values():
                        values.sort(key=lambda event: (event.start_ms, event.end_ms))

                    used_gesture_ids: set[int] = set()
                    used_one_finger_ids: set[int] = set()
                    # Mirror the label priority so key contacts and multi-touch
                    # contacts are reserved before lower-priority actions.
                    for action in ["keystroke", "pinch", "swipe", "scroll", "tap"]:
                        action_events = by_action.get(action, [])
                        for event_index, event in enumerate(action_events):
                            event_id = int(
                                dp.event_identifier(
                                    user_id,
                                    session_id,
                                    int(dp.ACTION_NAME_TO_ID[action]),
                                    event_index,
                                )
                            )
                            if (
                                action != "keystroke"
                                and (touch_df is None or len(touch_df) == 0)
                            ):
                                audit.add_event(
                                    _audit_event_row(
                                        user_id=user_id,
                                        user_external_id=external_id,
                                        session_id=session_id,
                                        action=action,
                                        event_id=event_id,
                                        event=event,
                                        status="rejected",
                                        reason="missing_or_empty_touch_event_file",
                                        n_keys=len(event.meta.get("keys", [])),
                                    )
                                )
                                continue
                            if action == "keystroke":
                                process_keystroke_event(
                                    dp=dp,
                                    writer=writers.get(action),
                                    audit=audit,
                                    event=event,
                                    event_id=event_id,
                                    user_id=user_id,
                                    user_external_id=external_id,
                                    session_id=session_id,
                                    activity_meta=activity_meta,
                                    gestures_by_activity=gestures_by_activity,
                                    one_finger_by_activity=one_finger_by_activity,
                                    used_gesture_ids=used_gesture_ids,
                                    used_one_finger_ids=used_one_finger_ids,
                                    match_tolerance_ms=int(args.match_tolerance_ms),
                                    allow_partial=bool(args.allow_partial_keystroke),
                                )
                            else:
                                process_non_key_event(
                                    dp=dp,
                                    writer=writers.get(action),
                                    audit=audit,
                                    event=event,
                                    action=action,
                                    event_id=event_id,
                                    user_id=user_id,
                                    user_external_id=external_id,
                                    session_id=session_id,
                                    activity_meta=activity_meta,
                                    gestures_by_activity=gestures_by_activity,
                                    used_gesture_ids=used_gesture_ids,
                                    match_tolerance_ms=int(args.match_tolerance_ms),
                                    container_margin_ms=int(args.container_margin_ms),
                                )

                    user_session_count += 1
                    session_count += 1
                    print(
                        f"  session={session_id:02d} raw_gestures={len(raw_gestures):4d} "
                        + " ".join(
                            f"{name}={len(by_action.get(name, []))}"
                            for name in ALL_ACTIONS
                        ),
                        flush=True,
                    )
            processed_users.append(
                {
                    "user_id": int(user_id),
                    "user_external_id": int(external_id),
                    "outer_member": member,
                    "inner_zip_size_bytes": len(inner_bytes),
                    "n_sessions": user_session_count,
                    "n_sessions_discovered": len(discovered_session_items),
                    "ignored_non_dataset_session_entries": ignored_session_items,
                }
            )

        output_summaries: dict[str, Any] = {}
        for action, writer in writers.items():
            print(f"[save] {action}", flush=True)
            output_summaries[action] = writer.finalize()
        audit_payload = audit.finalize()
        audit_payload.update(
            {
                "schema_version": SCHEMA_VERSION,
                "script_version": SCRIPT_VERSION,
                "created_unix_time": time.time(),
                "processed_user_count": len(processed_users),
                "processed_session_count": session_count,
            }
        )
        _json_dump_atomic(output_dir / "audit.json", audit_payload)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            "created_unix_time": time.time(),
            "elapsed_seconds": time.time() - started_at,
            "source": {
                "hmog_zip": str(args.hmog_zip.resolve()),
                "hmog_zip_size_bytes": int(args.hmog_zip.stat().st_size),
                "event_preprocessor": str(args.event_preprocess.resolve()),
                "event_preprocessor_sha256": preprocessor_sha256,
                "event_pipeline_version": getattr(dp, "PIPELINE_VERSION", "unknown"),
                "config": str(args.config.resolve()),
                "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
            },
            "selection": {
                "actions": list(actions),
                "max_users": int(args.max_users),
                "max_sessions": int(args.max_sessions),
                "user_external_id": args.user_external_id,
                "processed_users": processed_users,
            },
            "matching": {
                "match_tolerance_ms": int(args.match_tolerance_ms),
                "container_margin_ms": int(args.container_margin_ms),
                "allow_partial_keystroke": bool(args.allow_partial_keystroke),
                "single_pointer_required_for": ["tap", "scroll", "swipe", "keystroke"],
                "two_pointer_required_for": ["pinch"],
                "pinch_requires_exactly_two_pointers": True,
                "keystroke_touch_sources": {
                    "0": "complete raw TouchEvent contact",
                    "1": "observed OneFingerTouchEvent DOWN/UP endpoints",
                },
                "keystroke_fallback_interpolates_move_or_xy": False,
                "raw_timestamps_resampled": False,
            },
            "action_id_map": {
                name: int(dp.ACTION_NAME_TO_ID[name]) for name in ALL_ACTIONS
            },
            "outputs": output_summaries,
            "audit_json": str(output_dir / "audit.json"),
            "event_audit_csv": str(output_dir / "event_audit.csv"),
        }
        _json_dump_atomic(output_dir / "manifest.json", manifest)
        print(
            "[done] "
            + " ".join(
                f"{action}={summary['n_events']}"
                for action, summary in output_summaries.items()
            )
            + f" elapsed={manifest['elapsed_seconds']:.1f}s",
            flush=True,
        )
        return 0
    except Exception:
        for writer in writers.values():
            writer.abort()
        audit.abort()
        raise
    finally:
        outer.close()
        if build_root.exists() and not any(build_root.iterdir()):
            build_root.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
