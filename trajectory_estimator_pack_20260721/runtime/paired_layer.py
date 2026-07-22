"""Generate and audit paired IMU + touch trajectories from one EventPlan."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .trajectory_layer import EventPlan, TrajectoryDiffusionLayer


PAIR_AUDIT_SCHEMA = "paired_imu_trajectory_audit_v1"


def _metadata(result: Mapping[str, Any]) -> Mapping[str, Any]:
    value = result.get("metadata", {})
    if not isinstance(value, Mapping):
        raise ValueError("generator result metadata must be a mapping")
    return value


def _float_meta(result: Mapping[str, Any], *names: str) -> float:
    metadata = _metadata(result)
    for name in names:
        if name in result and result[name] is not None:
            value = float(result[name])
            if math.isfinite(value):
                return value
        if name in metadata and metadata[name] is not None:
            value = float(metadata[name])
            if math.isfinite(value):
                return value
    raise ValueError("missing finite metadata field: %s" % (names,))


def audit_paired_generation(
    plan: EventPlan,
    imu_result: Mapping[str, Any],
    trajectory_result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fail closed when two modalities do not describe the same event."""

    errors = []
    imu_meta = _metadata(imu_result)
    expected = {
        "sample_id": plan.sample_id,
        "action": plan.action,
        "user_id": int(plan.user_id),
        "orientation_id": int(plan.orientation_id),
        "event_plan_sha256": plan.plan_sha256,
    }
    for name, value in expected.items():
        observed_imu = imu_result.get(name, imu_meta.get(name))
        observed_trajectory = trajectory_result.get(name, _metadata(trajectory_result).get(name))
        if observed_imu != value:
            errors.append("IMU %s mismatch: %r != %r" % (name, observed_imu, value))
        if observed_trajectory != value:
            errors.append("trajectory %s mismatch: %r != %r" % (name, observed_trajectory, value))
    try:
        imu_duration = _float_meta(
            imu_result, "logical_event_duration_ms", "event_duration_ms", "duration_ms"
        )
    except ValueError as exc:
        errors.append(str(exc))
        imu_duration = float("nan")
    trajectory_duration = float(trajectory_result.get("duration_ms", float("nan")))
    if not math.isfinite(trajectory_duration) or abs(trajectory_duration - plan.duration_ms) > 1.0e-6:
        errors.append("trajectory logical duration mismatch")
    if not math.isfinite(imu_duration) or abs(imu_duration - plan.duration_ms) > 1.0e-6:
        errors.append("IMU logical duration mismatch")
    relative = np.asarray(trajectory_result.get("relative_timestamps_ns", []), dtype=np.int64)
    if relative.size < 2 or int(np.min(relative)) != 0 or abs(int(np.max(relative)) / 1.0e6 - plan.duration_ms) > 1.0e-6:
        errors.append("trajectory physical timeline does not span [0,duration_ms]")
    imu_t = np.asarray(imu_result.get("relative_timestamps_ns", []), dtype=np.int64)
    imu = np.asarray(imu_result.get("active_imu", []), dtype=np.float32)
    if imu.ndim != 2 or imu.shape[1:] != (6,) or imu_t.shape != (imu.shape[0],):
        errors.append("IMU active_imu/timestamp shape mismatch")
    if plan.start_time_ns is not None:
        trajectory_abs = trajectory_result.get("timestamps_ns")
        imu_abs = imu_result.get("timestamps_ns")
        if trajectory_abs is None or int(np.asarray(trajectory_abs)[0]) != int(plan.start_time_ns):
            errors.append("trajectory absolute start time mismatch")
        if imu_abs is None or int(np.asarray(imu_abs)[0]) != int(plan.start_time_ns):
            errors.append("IMU absolute start time mismatch")
    report = {
        "schema_version": PAIR_AUDIT_SCHEMA,
        "passed": not errors,
        "sample_id": plan.sample_id,
        "event_plan_sha256": plan.plan_sha256,
        "action": plan.action,
        "user_id": int(plan.user_id),
        "logical_duration_ms": float(plan.duration_ms),
        "imu_active_points": int(imu.shape[0]) if imu.ndim == 2 else 0,
        "trajectory_rows": int(relative.size),
        "errors": errors,
    }
    if errors:
        raise ValueError("paired generation audit failed: %s" % "; ".join(errors))
    return report


def _trajectory_speed_series(result: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
    t = np.asarray(result["relative_timestamps_ns"], dtype=np.float64) / 1.0e6
    x = np.asarray(result["x"], dtype=np.float64)
    y = np.asarray(result["y"], dtype=np.float64)
    frame = np.asarray(result["frame_index"], dtype=np.int64)
    key = np.asarray(result.get("key_index", np.full(t.shape, -1)), dtype=np.int64)
    if not (t.shape == x.shape == y.shape == frame.shape == key.shape):
        raise ValueError("trajectory arrays have inconsistent shapes")
    unique = np.unique(frame)
    times = np.empty(unique.size, dtype=np.float64)
    centers = np.empty((unique.size, 2), dtype=np.float64)
    frame_key = np.empty(unique.size, dtype=np.int64)
    for index, value in enumerate(unique):
        rows = np.flatnonzero(frame == value)
        times[index] = float(t[rows[0]])
        centers[index] = [float(np.mean(x[rows])), float(np.mean(y[rows]))]
        observed_keys = key[rows][key[rows] >= 0]
        frame_key[index] = int(observed_keys[0]) if observed_keys.size else -1
    dt = np.diff(times)
    distance = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    valid = dt > 0
    # Do not treat the spatial jump between distinct typed keys as finger
    # velocity; each key is an independent contact.
    valid &= (frame_key[1:] == frame_key[:-1]) | (frame_key[1:] < 0) | (frame_key[:-1] < 0)
    midpoint = 0.5 * (times[:-1] + times[1:])
    speed = np.zeros(dt.shape, dtype=np.float64)
    speed[valid] = distance[valid] / dt[valid] * 1000.0
    return midpoint, speed, float(np.sum(distance[valid]))


def _record_speed_series(record: Any) -> Tuple[np.ndarray, np.ndarray, float, int, int, int]:
    """Return the same center-speed semantics from a detector RawTrajectoryRecord."""

    record.validate()
    times = np.asarray(record.global_t_ms, dtype=np.float64)
    values = np.asarray(record.pointer_continuous, dtype=np.float64)
    contact = np.asarray(record.contact_mask, dtype=bool)
    event_ids = np.asarray(record.event_ids, dtype=np.int64)
    has_contact = np.any(contact, axis=0)
    centers = np.zeros((times.size, 2), dtype=np.float64)
    frame_event = np.full(times.size, -1, dtype=np.int64)
    for index in np.flatnonzero(has_contact):
        pointers = np.flatnonzero(contact[:, index])
        centers[index] = np.mean(values[pointers, index, :2], axis=0)
        observed = event_ids[pointers, index]
        observed = observed[observed >= 0]
        frame_event[index] = int(observed[0]) if observed.size else -1
    dt = np.diff(times)
    distance = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    valid = (dt > 0) & has_contact[:-1] & has_contact[1:]
    if str(record.action) == "keystroke":
        valid &= frame_event[:-1] == frame_event[1:]
    midpoint = 0.5 * (times[:-1] + times[1:])
    speed = np.zeros(dt.shape, dtype=np.float64)
    speed[valid] = distance[valid] / dt[valid] * 1000.0
    pointer_count = int(np.max(np.sum(contact, axis=0)))
    keys = np.unique(event_ids[event_ids >= 0]) if str(record.action) == "keystroke" else np.asarray([])
    n_keys = int(keys.size)
    physical_frames = int(np.sum(has_contact))
    return midpoint, speed, float(np.sum(distance[valid])), physical_frames, pointer_count, n_keys


def cross_modal_consistency_from_record(
    *,
    action: str,
    logical_duration_ms: float,
    active_imu: np.ndarray,
    imu_relative_timestamps_ms: np.ndarray,
    trajectory_record: Any,
) -> Tuple[Tuple[str, ...], np.ndarray]:
    """Shared real/fake consistency features on the detector record schema."""

    duration = float(logical_duration_ms)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("logical_duration_ms must be finite and positive")
    imu = np.asarray(active_imu, dtype=np.float64)
    imu_time = np.asarray(imu_relative_timestamps_ms, dtype=np.float64).reshape(-1)
    if imu.ndim != 2 or imu.shape[1] != 6 or imu.shape[0] < 1 or imu_time.shape != (imu.shape[0],):
        raise ValueError("active IMU and relative time must be [N,6]/[N] with N>=1")
    if not np.all(np.isfinite(imu)) or not np.all(np.isfinite(imu_time)) or np.any(np.diff(imu_time) <= 0):
        raise ValueError("active IMU/timeline must be finite and strictly increasing")
    if str(trajectory_record.action) != str(action):
        raise ValueError("trajectory record action disagrees with paired action")
    speed_t, speed, path_length, frame_count, pointer_count, n_keys = _record_speed_series(
        trajectory_record
    )
    accel_delta = np.linalg.norm(np.diff(imu[:, :3], axis=0), axis=1)
    gyro = np.linalg.norm(imu[:, 3:6], axis=1)
    imu_mid = 0.5 * (imu_time[:-1] + imu_time[1:])
    speed_on_imu = (
        np.interp(imu_mid, speed_t, speed, left=0.0, right=0.0)
        if speed_t.size else np.zeros_like(imu_mid)
    )

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        if left.size < 3 or right.size != left.size or np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    gyro_mid = 0.5 * (gyro[:-1] + gyro[1:])
    if gyro_mid.size != speed_on_imu.size:
        gyro_mid = np.interp(imu_mid, imu_time, gyro, left=gyro[0], right=gyro[-1])
    speed_peak_ms = float(imu_mid[int(np.argmax(speed_on_imu))]) if speed_on_imu.size else 0.0
    motion = accel_delta + gyro_mid
    motion_peak_ms = float(imu_mid[int(np.argmax(motion))]) if motion.size else 0.0
    names = (
        "consistency__path_length_px",
        "consistency__mean_touch_speed_px_s",
        "consistency__accel_delta_rms",
        "consistency__gyro_rms",
        "consistency__touch_speed_accel_corr",
        "consistency__touch_speed_gyro_corr",
        "consistency__motion_touch_peak_delta_ms",
        "consistency__trajectory_rows_per_second",
        "consistency__pointer_count",
        "consistency__n_keys",
    )
    values = np.asarray([
        path_length,
        float(np.mean(speed)) if speed.size else 0.0,
        float(np.sqrt(np.mean(accel_delta ** 2))) if accel_delta.size else 0.0,
        float(np.sqrt(np.mean(gyro ** 2))) if gyro.size else 0.0,
        correlation(speed_on_imu, accel_delta),
        correlation(speed_on_imu, gyro_mid),
        motion_peak_ms - speed_peak_ms,
        float(frame_count) * 1000.0 / duration,
        float(pointer_count),
        float(n_keys),
    ], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("cross-modal consistency features are non-finite")
    return names, values


def cross_modal_consistency_features(
    plan: EventPlan,
    imu_result: Mapping[str, Any],
    trajectory_result: Mapping[str, Any],
) -> Tuple[Tuple[str, ...], np.ndarray]:
    """Extract physical consistency signals for the trainable total detector."""

    audit_paired_generation(plan, imu_result, trajectory_result)
    if trajectory_result.get("raw_detector_record") is not None:
        return cross_modal_consistency_from_record(
            action=plan.action,
            logical_duration_ms=plan.duration_ms,
            active_imu=np.asarray(imu_result["active_imu"], dtype=np.float64),
            imu_relative_timestamps_ms=(
                np.asarray(imu_result["relative_timestamps_ns"], dtype=np.float64) / 1.0e6
            ),
            trajectory_record=trajectory_result["raw_detector_record"],
        )
    imu = np.asarray(imu_result["active_imu"], dtype=np.float64)
    imu_time = np.asarray(imu_result["relative_timestamps_ns"], dtype=np.float64) / 1.0e6
    speed_t, speed, path_length = _trajectory_speed_series(trajectory_result)
    accel_delta = np.linalg.norm(np.diff(imu[:, :3], axis=0), axis=1)
    gyro = np.linalg.norm(imu[:, 3:6], axis=1)
    imu_mid = 0.5 * (imu_time[:-1] + imu_time[1:])
    speed_on_imu = np.interp(imu_mid, speed_t, speed, left=0.0, right=0.0) if speed_t.size else np.zeros_like(imu_mid)

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        if left.size < 3 or right.size != left.size or np.std(left) < 1.0e-12 or np.std(right) < 1.0e-12:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    accel_corr = correlation(speed_on_imu, accel_delta)
    gyro_mid = 0.5 * (gyro[:-1] + gyro[1:]) if gyro.size > 1 else np.zeros_like(speed_on_imu)
    if gyro_mid.size != speed_on_imu.size:
        gyro_mid = np.interp(imu_mid, imu_time, gyro, left=gyro[0], right=gyro[-1])
    gyro_corr = correlation(speed_on_imu, gyro_mid)
    speed_peak_ms = float(imu_mid[int(np.argmax(speed_on_imu))]) if speed_on_imu.size else 0.0
    motion = accel_delta + gyro_mid
    motion_peak_ms = float(imu_mid[int(np.argmax(motion))]) if motion.size else 0.0
    names = (
        "consistency__path_length_px",
        "consistency__mean_touch_speed_px_s",
        "consistency__accel_delta_rms",
        "consistency__gyro_rms",
        "consistency__touch_speed_accel_corr",
        "consistency__touch_speed_gyro_corr",
        "consistency__motion_touch_peak_delta_ms",
        "consistency__trajectory_rows_per_second",
        "consistency__pointer_count",
        "consistency__n_keys",
    )
    values = np.asarray(
        [
            path_length,
            float(np.mean(speed)) if speed.size else 0.0,
            float(np.sqrt(np.mean(accel_delta ** 2))) if accel_delta.size else 0.0,
            float(np.sqrt(np.mean(gyro ** 2))) if gyro.size else 0.0,
            accel_corr,
            gyro_corr,
            motion_peak_ms - speed_peak_ms,
            float(len(np.unique(trajectory_result["frame_index"]))) * 1000.0 / plan.duration_ms,
            float(plan.pointer_count),
            float(plan.n_keys),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("cross-modal consistency features are non-finite")
    return names, values


class PairedGenerationService:
    """Run both neural generators from one already-resolved EventPlan."""

    def __init__(self, trajectory_layer: TrajectoryDiffusionLayer, imu_layer: Any) -> None:
        self.trajectory_layer = trajectory_layer
        self.imu_layer = imu_layer

    def generate_plan(self, plan: EventPlan) -> Dict[str, Any]:
        started = time.perf_counter()
        trajectory = self.trajectory_layer.generate_plan(plan)
        imu = self.imu_layer.generate(plan.action, **plan.to_imu_kwargs())
        imu.setdefault("metadata", {})
        imu["sample_id"] = plan.sample_id
        imu["event_plan_sha256"] = plan.plan_sha256
        imu["user_id"] = int(plan.user_id)
        imu["orientation_id"] = int(plan.orientation_id)
        imu["metadata"].update(
            sample_id=plan.sample_id,
            event_plan_sha256=plan.plan_sha256,
            user_id=int(plan.user_id),
            orientation_id=int(plan.orientation_id),
        )
        audit = audit_paired_generation(plan, imu, trajectory)
        feature_names, feature_values = cross_modal_consistency_features(plan, imu, trajectory)
        return {
            "sample_id": plan.sample_id,
            "event_plan": plan.to_dict(),
            "event_plan_sha256": plan.plan_sha256,
            "imu": imu,
            "trajectory": trajectory,
            "pair_audit": audit,
            "consistency_feature_names": feature_names,
            "consistency_features": feature_values,
            "paired_generation_wall_ms": float((time.perf_counter() - started) * 1000.0),
        }


__all__ = [
    "PAIR_AUDIT_SCHEMA", "PairedGenerationService", "audit_paired_generation",
    "cross_modal_consistency_features", "cross_modal_consistency_from_record",
]
