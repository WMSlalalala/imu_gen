"""Lossless numeric Android/Type-B serialization of generated trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from trajectory.constraints import ConstrainedTrajectory
from trajectory.data import BASE_DT_MS, TrajectoryBatch, keystroke_zero_flight_flags

from .protocol import ConditionRequest


ACTION_DOWN = 0
ACTION_UP = 1
ACTION_MOVE = 2
ACTION_POINTER_DOWN = 5
ACTION_POINTER_UP = 6
ACTION_POINTER_INDEX_SHIFT = 8
TYPE_B_NO_TRACKING_UPDATE = np.iinfo(np.int32).min
PHASE_DOWN = 0
PHASE_MOVE = 1
PHASE_UP = 2


@dataclass
class AndroidTrajectoryRecord:
    request: ConditionRequest
    trajectory_features: np.ndarray       # [M,5], pointer-major valid timeline
    trajectory_t_ms: np.ndarray           # [M]
    trajectory_pointer_id: np.ndarray
    trajectory_contact_mask: np.ndarray
    trajectory_event_id: np.ndarray
    trajectory_pointer_offsets: np.ndarray  # [3], offsets per pointer into trajectory arrays
    android_t_ms: np.ndarray
    android_x: np.ndarray
    android_y: np.ndarray
    android_pressure: np.ndarray
    android_size: np.ndarray
    android_pointer_id: np.ndarray
    android_slot: np.ndarray
    android_tracking_id: np.ndarray
    android_type_b_tracking_value: np.ndarray
    android_phase: np.ndarray
    android_action: np.ndarray
    android_key_index: np.ndarray
    android_keycode: np.ndarray
    android_frame_index: np.ndarray
    android_frame_end: np.ndarray
    clipped_point_count: int
    contact_point_count: int


def _cpu(value) -> np.ndarray:
    return value.detach().cpu().numpy()


def _integer_ms_timeline(
    predicted: np.ndarray,
    start_ms: float,
    end_ms: float,
    allow_zero_interval: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Largest-remainder allocation on the source HMOG 1 ms lattice.

    Every ordinary interval keeps at least 1 ms.  A declared keystroke
    zero-flight boundary keeps exactly 0 ms, preserving two ordered UP/DOWN
    MotionEvents that share HMOG's integer EventTime.  Remaining milliseconds
    are assigned proportionally to the neural positive interval weights.
    """
    values = np.asarray(predicted, np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("predicted pointer timeline must contain at least two points")
    intervals = values.size - 1
    allowed_zero = (
        np.zeros(intervals, dtype=np.bool_)
        if allow_zero_interval is None
        else np.asarray(allow_zero_interval, dtype=np.bool_).reshape(-1)
    )
    if allowed_zero.shape != (intervals,):
        raise ValueError("allow_zero_interval must have shape [N-1]")
    predicted_dt = np.diff(values)
    if np.any(predicted_dt[~allowed_zero] <= 0) or np.any(predicted_dt[allowed_zero] != 0):
        raise ValueError("predicted timeline contradicts positive/zero interval topology")
    start = int(round(float(start_ms)))
    end = int(round(float(end_ms)))
    total = end - start
    positive = ~allowed_zero
    minimum_duration = int(np.sum(positive))
    if total < minimum_duration:
        raise ValueError("integer-ms lifetime is shorter than its positive intervals")
    weights = predicted_dt[positive]
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("invalid predicted relative timing weights")
    remaining = total - minimum_duration
    allocated_positive = np.ones(minimum_duration, np.int64)
    if remaining:
        ideal = weights / float(np.sum(weights)) * remaining
        extra = np.floor(ideal).astype(np.int64)
        leftover = int(remaining - int(np.sum(extra)))
        if leftover:
            # Stable tie-break by earlier interval makes the result identical
            # across devices/shard layouts.
            order = np.lexsort((np.arange(minimum_duration), -(ideal - extra)))
            extra[order[:leftover]] += 1
        allocated_positive += extra
    allocated = np.zeros(intervals, np.int64)
    allocated[positive] = allocated_positive
    result = np.concatenate(([start], start + np.cumsum(allocated))).astype(np.float32)
    observed = np.diff(result)
    if (
        result[0] != start
        or result[-1] != end
        or np.any(observed[positive] < 1)
        or np.any(observed[allowed_zero] != 0)
    ):
        raise AssertionError("largest-remainder integer timeline invariant failed")
    return result


def _android_action_maps(action: str, pointer_lifetimes: List[Tuple[int, float, float]]) -> Tuple[Dict[int, int], Dict[int, int]]:
    if action != "pinch":
        return {0: ACTION_DOWN}, {0: ACTION_UP}
    down_order = sorted(pointer_lifetimes, key=lambda x: (x[1], x[0]))
    up_order = sorted(pointer_lifetimes, key=lambda x: (x[2], -x[0]))
    down_actions: Dict[int, int] = {}
    up_actions: Dict[int, int] = {}
    active = 0
    for pointer_id, _, _ in down_order:
        down_actions[pointer_id] = ACTION_DOWN if active == 0 else (
            ACTION_POINTER_DOWN | (pointer_id << ACTION_POINTER_INDEX_SHIFT)
        )
        active += 1
    for pointer_id, _, _ in up_order:
        up_actions[pointer_id] = ACTION_UP if active == 1 else (
            ACTION_POINTER_UP | (pointer_id << ACTION_POINTER_INDEX_SHIFT)
        )
        active -= 1
    if active != 0:
        raise AssertionError("pinch active-pointer accounting failed")
    return down_actions, up_actions


def record_from_generated(
    output: ConstrainedTrajectory,
    batch: TrajectoryBatch,
    batch_index: int,
    request: ConditionRequest,
) -> AndroidTrajectoryRecord:
    """Extract one generated item and emit legal contact lifecycle rows."""
    if str(batch.target_sample_ids[batch_index]) != str(request.fake_id):
        raise ValueError("generated batch/request id mismatch")
    features = _cpu(output.features[batch_index]).astype(np.float32)
    xy = _cpu(output.xy[batch_index]).astype(np.float32)
    timestamps = _cpu(output.timestamps_ms[batch_index]).astype(np.float32)
    pressure = _cpu(output.pressure[batch_index]).astype(np.float32)
    size = _cpu(output.size[batch_index]).astype(np.float32)
    point_mask = _cpu(output.point_mask[batch_index]).astype(np.bool_)
    contact_mask = _cpu(output.contact_mask[batch_index]).astype(np.bool_)
    event_ids = _cpu(output.event_ids[batch_index]).astype(np.int64)
    phases = _cpu(output.contact_phase[batch_index]).astype(np.int8)
    pointer_mask = _cpu(output.pointer_mask[batch_index]).astype(np.bool_)

    # HMOG event timestamps are integer milliseconds.  Quantize the generated
    # relative timing weights before any archive/Android serialization and
    # rewrite log_dt so the stored feature tensor remains self-consistent.
    for pointer_id in range(2):
        indices = np.flatnonzero(point_mask[pointer_id])
        if not pointer_mask[pointer_id]:
            continue
        allow_zero = np.zeros(max(indices.size - 1, 0), dtype=np.bool_)
        if request.action == "keystroke":
            local_contact = contact_mask[pointer_id, indices]
            local_events = event_ids[pointer_id, indices]
            allow_zero = (
                local_contact[:-1]
                & local_contact[1:]
                & (local_events[:-1] >= 0)
                & (local_events[1:] == local_events[:-1] + 1)
            )
            observed_flags = keystroke_zero_flight_flags(
                local_contact, local_events, request.n_keys
            )
            if not np.array_equal(observed_flags, request.zero_flight_after_key):
                raise ValueError("generated topology lost a requested zero-flight boundary")
        quantized = _integer_ms_timeline(
            timestamps[pointer_id, indices],
            request.pointer_start_offset_ms[pointer_id],
            request.pointer_end_offset_ms[pointer_id],
            allow_zero_interval=allow_zero,
        )
        timestamps[pointer_id, indices] = quantized
        log_dt = np.log(
            np.maximum(np.diff(quantized).astype(np.float64), 1.0e-3) / BASE_DT_MS
        ).astype(np.float32)
        features[pointer_id, indices[1:], 2] = log_dt
        features[pointer_id, indices[0], 2] = log_dt[0]

    # Bounds are learned only from train users for the sampled orientation.
    # Endpoint conditions were already projected into the same bounds; this
    # last contact-wise clip is a hard physical deployment guard, not a test
    # detector selector.
    original_xy = xy.copy()
    for pointer_id in range(2):
        valid_contact = contact_mask[pointer_id]
        xy[pointer_id, valid_contact] = np.clip(
            xy[pointer_id, valid_contact], request.screen_min_xy, request.screen_max_xy
        )
    clipped_mask = contact_mask & np.any(np.abs(xy - original_xy) > 1e-6, axis=-1)
    clipped_count = int(np.sum(clipped_mask))
    contact_count = int(np.sum(contact_mask))

    trajectory_parts = {"features": [], "t": [], "pointer": [], "contact": [], "event": []}
    pointer_offsets = [0]
    lifetimes: List[Tuple[int, float, float]] = []
    for pointer_id in range(2):
        indices = np.flatnonzero(point_mask[pointer_id])
        if not pointer_mask[pointer_id]:
            pointer_offsets.append(pointer_offsets[-1])
            continue
        if indices.size < 2:
            raise ValueError("generated active pointer has fewer than two points")
        trajectory_parts["features"].append(features[pointer_id, indices])
        trajectory_parts["t"].append(timestamps[pointer_id, indices])
        trajectory_parts["pointer"].append(np.full(indices.size, pointer_id, np.int8))
        trajectory_parts["contact"].append(contact_mask[pointer_id, indices])
        trajectory_parts["event"].append(event_ids[pointer_id, indices])
        pointer_offsets.append(pointer_offsets[-1] + indices.size)
        lifetimes.append((pointer_id, float(timestamps[pointer_id, indices[0]]), float(timestamps[pointer_id, indices[-1]])))
    down_actions, up_actions = _android_action_maps(request.action, lifetimes)

    rows: List[Tuple[float, int, int, int, int, int, float, float, float, float, int, int]] = []
    tracking_base = int(request.sample_index * 100 + 1)
    for pointer_id in range(2):
        if not pointer_mask[pointer_id]:
            continue
        contact_indices = np.flatnonzero(contact_mask[pointer_id])
        for index in contact_indices.tolist():
            phase = int(phases[pointer_id, index])
            event_id = int(event_ids[pointer_id, index])
            if phase not in (PHASE_DOWN, PHASE_MOVE, PHASE_UP) or event_id < 0:
                raise ValueError("contact row lacks legal phase/event id")
            if request.action == "keystroke":
                tracking_id = tracking_base + event_id
                action_code = ACTION_DOWN if phase == PHASE_DOWN else ACTION_UP if phase == PHASE_UP else ACTION_MOVE
                key_index = event_id
                keycode = int(request.keycodes[event_id])
            else:
                tracking_id = tracking_base + pointer_id
                action_code = down_actions[pointer_id] if phase == PHASE_DOWN else up_actions[pointer_id] if phase == PHASE_UP else ACTION_MOVE
                key_index = -1
                keycode = -1
            type_b = tracking_id if phase == PHASE_DOWN else -1 if phase == PHASE_UP else int(TYPE_B_NO_TRACKING_UPDATE)
            # Sorting key fields are appended at the end and removed below.
            lifecycle_order = 0 if phase == PHASE_DOWN else 2 if phase == PHASE_UP else 1
            tie_pointer = pointer_id if phase != PHASE_UP else -pointer_id
            rows.append((
                float(timestamps[pointer_id, index]), lifecycle_order, tie_pointer,
                pointer_id, tracking_id, type_b,
                float(xy[pointer_id, index, 0]), float(xy[pointer_id, index, 1]),
                float(pressure[pointer_id, index]), float(size[pointer_id, index]),
                phase, action_code, key_index, keycode,
            ))
    if request.action == "keystroke":
        # At a zero-flight boundary the previous key's UP must precede the
        # next key's DOWN even though both carry the same integer EventTime.
        rows.sort(key=lambda x: (x[0], x[12], x[1]))
    else:
        rows.sort(key=lambda x: (x[0], x[1], x[2]))
    if not rows:
        raise ValueError("generated trajectory has no Android contact rows")
    array = np.asarray(rows, dtype=np.float64)
    android_t = array[:, 0].astype(np.float32)
    pointer = array[:, 3].astype(np.int8)
    tracking = array[:, 4].astype(np.int32)
    type_b = array[:, 5].astype(np.int32)
    ax, ay = array[:, 6].astype(np.float32), array[:, 7].astype(np.float32)
    ap, az = array[:, 8].astype(np.float32), array[:, 9].astype(np.float32)
    phase = array[:, 10].astype(np.int8)
    action_code = array[:, 11].astype(np.int16)
    key_index = array[:, 12].astype(np.int16)
    keycode = array[:, 13].astype(np.int32)
    # Separate keystroke lifecycle callbacks remain separate MotionEvent
    # frames even when coarse integer EventTime is equal.  Pinch slot updates
    # at one timestamp still share one Type-B SYN_REPORT frame.
    if request.action == "keystroke":
        frame_index = np.arange(len(rows), dtype=np.int32)
        frame_end = np.ones(len(rows), np.uint8)
    else:
        frame_index = np.zeros(len(rows), np.int32)
        frame = 0
        for row_index in range(1, len(rows)):
            if android_t[row_index] != android_t[row_index - 1]:
                frame += 1
            frame_index[row_index] = frame
        frame_end = np.ones(len(rows), np.uint8)
        frame_end[:-1] = (frame_index[:-1] != frame_index[1:]).astype(np.uint8)

    def concat(name: str, dtype) -> np.ndarray:
        values = trajectory_parts[name]
        return np.concatenate(values).astype(dtype, copy=False) if values else np.zeros(0, dtype=dtype)

    return AndroidTrajectoryRecord(
        request=request,
        trajectory_features=concat("features", np.float32).reshape(-1, 5),
        trajectory_t_ms=concat("t", np.float32),
        trajectory_pointer_id=concat("pointer", np.int8),
        trajectory_contact_mask=concat("contact", np.uint8),
        trajectory_event_id=concat("event", np.int16),
        trajectory_pointer_offsets=np.asarray(pointer_offsets, np.int64),
        android_t_ms=android_t, android_x=ax, android_y=ay, android_pressure=ap, android_size=az,
        android_pointer_id=pointer, android_slot=pointer.copy(), android_tracking_id=tracking,
        android_type_b_tracking_value=type_b, android_phase=phase, android_action=action_code,
        android_key_index=key_index, android_keycode=keycode, android_frame_index=frame_index,
        android_frame_end=frame_end, clipped_point_count=clipped_count,
        contact_point_count=contact_count,
    )
