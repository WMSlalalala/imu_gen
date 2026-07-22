"""Adapter from audited flat HMOG-style trajectory NPZ to PAD raw records.

The adapter never resamples a trajectory.  It collapses rows belonging to one
Android MotionEvent into a shared global frame, preserves independently absent
pinch pointers, and inserts one explicit no-contact token for every positive-
flight keystroke transition.  A zero-flight UP/DOWN pair remains two ordered
contact tokens at the same physical millisecond, with no invented gap.  The same numeric schema can be emitted by the neural
generator for a like-for-like real/fake detector bundle.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np

from detectors.deep_pad import RawTrajectoryRecord, make_record
from training.corpus import encode_raw_keycode
from trajectory.features import (
    canonical_keycode_feature_token,
    extract_keystroke_features,
    extract_pinch_features,
    extract_single_finger_features,
)


REQUIRED_FLAT_FIELDS = (
    "event_offsets", "event_id", "user_id", "flat_frame_index", "flat_t_rel_ms",
    "flat_pointer_id", "flat_action_code", "flat_x", "flat_y", "flat_pressure",
    "flat_size", "flat_active_mask", "flat_key_index", "flat_keycode",
)


def _validate_source(data: Mapping[str, np.ndarray]) -> Tuple[str, int]:
    missing = [name for name in REQUIRED_FLAT_FIELDS if name not in data]
    if missing:
        raise ValueError("trajectory NPZ is missing fields: %s" % missing)
    action = str(data["action_name"].item())
    offsets = np.asarray(data["event_offsets"], dtype=np.int64)
    n = len(data["event_id"])
    if offsets.shape != (n + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("event_offsets must define non-empty event slices")
    if offsets[-1] != len(data["flat_x"]):
        raise ValueError("event_offsets do not terminate at flat row count")
    flat_length = int(offsets[-1])
    for name in REQUIRED_FLAT_FIELDS[2:]:
        if name.startswith("flat_") and len(data[name]) != flat_length:
            raise ValueError("flat field length mismatch: %s" % name)
    return action, n


def _event_frames(data: Mapping[str, np.ndarray], left: int, right: int, action: str):
    frame_index = np.asarray(data["flat_frame_index"][left:right], dtype=np.int64)
    if np.any(np.diff(frame_index) < 0):
        raise ValueError("flat_frame_index must be monotonic within an event")
    unique_frames = np.unique(frame_index)
    raw_pointer_ids = np.asarray(data["flat_pointer_id"][left:right], dtype=np.int64)
    # Android pointer slots are assigned by tracking lifecycle/first
    # appearance.  Numeric tracking IDs are arbitrary and may be 9 then 2;
    # sorting them would swap the primary ACTION_DOWN pointer between real and
    # generated records and create a detector artifact.
    pointer_ids = []
    for pointer in raw_pointer_ids.tolist():
        if int(pointer) not in pointer_ids:
            pointer_ids.append(int(pointer))
    if action == "pinch":
        if len(pointer_ids) != 2:
            raise ValueError("pinch must contain exactly two pointer lifecycles")
        pointer_slot = {
            int(pointer): index for index, pointer in enumerate(pointer_ids)
        }
    elif action == "keystroke":
        # A typing event concatenates independent one-finger contacts.  Android
        # may assign a new tracking ID to a later key, but those IDs are never
        # simultaneous pointers and must all occupy canonical slot 0.
        pointer_slot = {int(pointer): 0 for pointer in pointer_ids}
    else:
        if len(pointer_ids) != 1:
            raise ValueError("single-finger gesture must contain exactly one pointer")
        pointer_slot = {int(pointer_ids[0]): 0}

    values: List[np.ndarray] = []
    times: List[float] = []
    contacts: List[np.ndarray] = []
    active: List[np.ndarray] = []
    codes: List[np.ndarray] = []
    keycodes: List[np.ndarray] = []
    event_ids: List[np.ndarray] = []
    gap: List[bool] = []
    previous_key_index = None
    previous_time = None
    for frame in unique_frames:
        local = np.flatnonzero(frame_index == frame) + left
        frame_times = np.asarray(data["flat_t_rel_ms"][local], dtype=np.float64)
        if not np.all(frame_times == frame_times[0]):
            raise ValueError("rows in one MotionEvent frame disagree on global time")
        frame_time = float(frame_times[0])
        frame_key_indices = np.asarray(data["flat_key_index"][local], dtype=np.int64)
        current_key_index = int(frame_key_indices[0]) if action == "keystroke" else 0
        if action == "keystroke" and not np.all(frame_key_indices == current_key_index):
            raise ValueError("one keystroke frame cannot belong to multiple keys")
        if action == "keystroke" and len(np.unique(data["flat_pointer_id"][local])) != 1:
            raise ValueError("one keystroke frame must contain exactly one pointer lifecycle")

        key_transition = (
            action == "keystroke" and previous_key_index is not None
            and current_key_index != previous_key_index
        )
        if previous_time is not None and frame_time < previous_time:
            raise ValueError("distinct global frames cannot move backward in time")
        if previous_time is not None and frame_time == previous_time and not key_transition:
            raise ValueError(
                "equal global frame times are legal only across a zero-flight key boundary"
            )
        if key_transition and frame_time > previous_time:
            # A float midpoint preserves both original contact timestamps and
            # provides a genuine no-contact flight token even for a 1ms gap.
            # For a 0ms flight, the two original contact frames remain directly
            # adjacent and retain their equal timestamp; no fake time is made.
            midpoint = previous_time + 0.5 * (frame_time - previous_time)
            values.append(np.zeros((2, 4), dtype=np.float32))
            times.append(float(midpoint))
            contacts.append(np.zeros((2,), dtype=bool))
            active.append(np.zeros((2,), dtype=bool))
            codes.append(np.full((2,), -1, dtype=np.int16))
            keycodes.append(np.full((2,), -1, dtype=np.int32))
            event_ids.append(np.full((2,), -1, dtype=np.int32))
            gap.append(True)

        frame_values = np.zeros((2, 4), dtype=np.float32)
        frame_contact = np.zeros((2,), dtype=bool)
        frame_active = np.zeros((2,), dtype=bool)
        frame_codes = np.full((2,), -1, dtype=np.int16)
        frame_keycodes = np.full((2,), -1, dtype=np.int32)
        frame_events = np.full((2,), -1, dtype=np.int32)
        for row in local:
            slot = pointer_slot[int(data["flat_pointer_id"][row])]
            # HMOG contains a small number of byte-identical repeated pointer
            # rows at the same frame/time.  Android exposes one pointer state
            # per MotionEvent, so use deterministic last-row-wins rather than
            # creating a fake additional temporal token.
            frame_contact[slot] = True
            frame_active[slot] = bool(data["flat_active_mask"][row])
            frame_codes[slot] = int(data["flat_action_code"][row])
            frame_keycodes[slot] = (
                encode_raw_keycode(int(data["flat_keycode"][row]))
                if action == "keystroke" else -1
            )
            frame_events[slot] = current_key_index if action == "keystroke" else 0
            frame_values[slot] = np.asarray([
                data["flat_x"][row], data["flat_y"][row],
                data["flat_pressure"][row], data["flat_size"][row],
            ], dtype=np.float32)
        values.append(frame_values)
        times.append(frame_time)
        contacts.append(frame_contact)
        active.append(frame_active)
        codes.append(frame_codes)
        keycodes.append(frame_keycodes)
        event_ids.append(frame_events)
        gap.append(False)
        previous_key_index = current_key_index
        previous_time = frame_time
    value_array = np.stack(values, axis=1)
    contact_array = np.stack(contacts, axis=1)
    active_array = np.stack(active, axis=1)
    # Android actionMasked is global to the MotionEvent and is repeated on all
    # pointer rows by HMOG.  Feeding that value as though it were pointer-local
    # would falsely mark both fingers DOWN/UP together.  Derive pointer-local
    # phases from appearance/disappearance on the preserved global timeline;
    # pointer 0 uses DOWN/UP and the secondary pointer uses POINTER_DOWN/UP.
    code_array = np.full(contact_array.shape, -1, dtype=np.int16)
    for pointer in range(2):
        indices = np.flatnonzero(contact_array[pointer])
        if len(indices) == 0:
            continue
        code_array[pointer, indices] = 2
        starts = indices[np.concatenate(([True], np.diff(indices) > 1))]
        ends = indices[np.concatenate((np.diff(indices) > 1, [True]))]
        code_array[pointer, starts] = 0 if pointer == 0 else 5
        code_array[pointer, ends] = 1 if pointer == 0 else 6
    return (
        value_array, np.asarray(times, dtype=np.float32),
        contact_array, active_array,
        code_array, np.stack(keycodes, axis=1),
        np.stack(event_ids, axis=1), np.asarray(gap, dtype=bool),
    )


def _feature_vector(
    data: Mapping[str, np.ndarray],
    action: str,
    event_index: int,
    pointer_values: np.ndarray,
    times: np.ndarray,
    contact: np.ndarray,
    gap: np.ndarray,
) -> np.ndarray:
    if action in ("tap", "scroll", "swipe"):
        keep = contact[0]
        return extract_single_finger_features(pointer_values[0, keep, :2], times[keep])
    if action == "pinch":
        keep = contact[0] & contact[1]
        if not np.any(keep):
            raise ValueError("pinch event has no simultaneous two-pointer frame")
        return extract_pinch_features(
            pointer_values[0, keep, :2], pointer_values[1, keep, :2], times[keep]
        )
    if action == "keystroke":
        if "event_key_offsets" not in data:
            raise ValueError("keystroke feature extraction needs event_key_offsets")
        left, right = np.asarray(data["event_key_offsets"][event_index:event_index + 2], dtype=np.int64)
        for name in ("keycode", "key_down_ms", "key_up_ms"):
            if name not in data:
                raise ValueError("keystroke source is missing %s" % name)
        # Use exactly the same auditable vocabulary as the neural training and
        # generation path: every raw negative HMOG sentinel maps to token 0,
        # while non-negative keycodes remain unchanged.  Otherwise real -1/-2/
        # -5 and generated 0 would form a trivial PAD artifact.
        keys = [
            canonical_keycode_feature_token(encode_raw_keycode(int(value)))
            for value in data["keycode"][left:right]
        ]
        down = np.asarray(data["key_down_ms"][left:right], dtype=np.float64)
        up = np.asarray(data["key_up_ms"][left:right], dtype=np.float64)
        # Use the first raw contact point of every key when present.
        points = []
        flat_key = np.asarray(data["flat_key_index"], dtype=np.int64)
        event_left, event_right = np.asarray(data["event_offsets"][event_index:event_index + 2], dtype=np.int64)
        for key_index in range(int(right - left)):
            rows = np.flatnonzero(flat_key[event_left:event_right] == key_index) + event_left
            if len(rows) == 0:
                points = []
                break
            points.append([float(data["flat_x"][rows[0]]), float(data["flat_y"][rows[0]])])
        key_points = np.asarray(points, dtype=np.float64) if len(points) == len(keys) else None
        return extract_keystroke_features(keys, down, up_times_ms=up, key_points=key_points)
    raise ValueError("unsupported action: %s" % action)


def load_extracted_trajectory_npz(
    path: Path,
    *,
    label: int,
    default_pool: str = "train",
    sample_prefix: str = "",
) -> Tuple[List[RawTrajectoryRecord], np.ndarray]:
    """Convert every event in one audited flat NPZ without pickle or resampling."""

    records: List[RawTrajectoryRecord] = []
    features: List[np.ndarray] = []
    # Materialize every compressed array once.  Repeated ``NpzFile[name]``
    # access can decompress the same member for every event and becomes
    # quadratic-time on the formal files.
    with np.load(Path(path), allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
        action, n = _validate_source(data)
        offsets = np.asarray(data["event_offsets"], dtype=np.int64)
        for index in range(n):
            left, right = int(offsets[index]), int(offsets[index + 1])
            values, times, contact, active, codes, keycodes, events, gap = _event_frames(
                data, left, right, action
            )
            event_group_id = str(int(data["event_id"][index]))
            sample_id = "%s%s:%s" % (sample_prefix, action, event_group_id)
            records.append(make_record(
                action=action, label=int(label), user_id=int(data["user_id"][index]),
                pool=default_pool, sample_id=sample_id, event_group_id=event_group_id,
                pointer_continuous=values, global_t_ms=times, contact_mask=contact,
                active_mask=active, action_code=codes, keycode=keycodes,
                event_ids=events, gap_mask=gap,
            ))
            features.append(_feature_vector(data, action, index, values, times, contact, gap))
    result = np.stack(features).astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("feature adapter produced non-finite values")
    return records, result


__all__ = ["load_extracted_trajectory_npz"]
