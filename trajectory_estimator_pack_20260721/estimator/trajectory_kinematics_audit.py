"""Supplemental human-likeness audit for complete variable-time trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from .fake_imu_pairs import sha256_file


KINEMATIC_METRICS = (
    "duration_ms", "path_px", "displacement_px", "path_efficiency",
    "mean_speed_px_s", "p95_speed_px_s", "max_speed_px_s",
    "mean_accel_px_s2", "max_accel_px_s2", "median_dt_ms", "dt_cv",
    "pressure_mean", "pressure_std", "size_mean", "size_std",
)
PRIMARY_GATES = (
    "path_px", "mean_speed_px_s", "p95_speed_px_s", "max_speed_px_s",
    "max_accel_px_s2", "median_dt_ms", "pressure_mean", "size_mean",
)


def _event_metrics(
    *, timeline: np.ndarray, values: np.ndarray, contact: np.ndarray, event_ids: np.ndarray,
) -> np.ndarray:
    times = np.asarray(timeline, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) < 0):
        raise ValueError("trajectory event timeline is invalid")
    duration = float(times[-1] - times[0])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("trajectory event duration must be positive")
    paths: List[float] = []
    displacements: List[float] = []
    all_speeds: List[float] = []
    all_accels: List[float] = []
    all_dt: List[float] = []
    pressure: List[float] = []
    size: List[float] = []
    for pointer in range(2):
        positions = np.flatnonzero(contact[:, pointer])
        if not len(positions):
            continue
        xy = values[positions, pointer, :2].astype(np.float64)
        pressure.extend(values[positions, pointer, 2].astype(float).tolist())
        size.extend(values[positions, pointer, 3].astype(float).tolist())
        displacements.append(float(np.linalg.norm(xy[-1] - xy[0])))
        if len(positions) < 2:
            paths.append(0.0)
            continue
        left = positions[:-1]
        right = positions[1:]
        dt = times[right] - times[left]
        same_discrete_event = event_ids[right, pointer] == event_ids[left, pointer]
        adjacent = right == left + 1
        keep = (dt > 0) & same_discrete_event & adjacent
        if not np.any(keep):
            paths.append(0.0)
            continue
        delta = values[right[keep], pointer, :2].astype(np.float64) - values[left[keep], pointer, :2].astype(np.float64)
        distance = np.linalg.norm(delta, axis=1)
        segment_dt = dt[keep]
        speed = distance / (segment_dt / 1000.0)
        paths.append(float(np.sum(distance)))
        all_speeds.extend(speed.tolist())
        all_dt.extend(segment_dt.tolist())
        if len(speed) >= 2:
            accel_dt = (segment_dt[1:] + segment_dt[:-1]) / 2000.0
            valid = accel_dt > 0
            all_accels.extend((np.abs(np.diff(speed)[valid]) / accel_dt[valid]).tolist())
    if not pressure:
        raise ValueError("trajectory event has no contact values")
    speed_values = np.asarray(all_speeds or [0.0], dtype=np.float64)
    accel_values = np.asarray(all_accels or [0.0], dtype=np.float64)
    dt_values = np.asarray(all_dt or [duration], dtype=np.float64)
    path = float(sum(paths))
    displacement = float(sum(displacements))
    pressure_values = np.asarray(pressure, dtype=np.float64)
    size_values = np.asarray(size, dtype=np.float64)
    output = np.asarray([
        duration, path, displacement, displacement / path if path > 1.0e-12 else 1.0,
        path / (duration / 1000.0), float(np.quantile(speed_values, 0.95)), float(np.max(speed_values)),
        float(np.mean(accel_values)), float(np.max(accel_values)), float(np.median(dt_values)),
        float(np.std(dt_values) / max(np.mean(dt_values), 1.0e-12)),
        float(np.mean(pressure_values)), float(np.std(pressure_values)),
        float(np.mean(size_values)), float(np.std(size_values)),
    ], dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise ValueError("trajectory kinematic metrics are non-finite")
    return output


def extract_bundle_kinematics(path: Path, *, expected_action: str) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as source:
        required = {
            "schema_version", "sequence_offsets", "flat_pointer_continuous",
            "flat_global_t_ms", "flat_contact_mask", "flat_event_ids",
            "label", "pool", "user_id", "sample_id", "action",
        }
        missing = required - set(source.files)
        if missing:
            raise ValueError("trajectory bundle lacks kinematic fields: %s" % sorted(missing))
        if str(np.asarray(source["schema_version"]).item()) != "trajectory_pad_bundle_v2":
            raise ValueError("trajectory bundle schema mismatch")
        arrays = {name: np.asarray(source[name]) for name in required - {"schema_version"}}
    n = len(arrays["label"])
    offsets = np.asarray(arrays["sequence_offsets"], dtype=np.int64)
    if offsets.shape != (n + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 2):
        raise ValueError("trajectory bundle offsets are invalid")
    total = int(offsets[-1])
    if any(len(arrays[name]) != total for name in (
        "flat_pointer_continuous", "flat_global_t_ms", "flat_contact_mask", "flat_event_ids"
    )):
        raise ValueError("trajectory bundle flat arrays disagree with offsets")
    if set(np.asarray(arrays["action"]).astype(str).tolist()) != {expected_action}:
        raise ValueError("trajectory bundle action mismatch")
    matrix = np.empty((n, len(KINEMATIC_METRICS)), dtype=np.float64)
    for index in range(n):
        left, right = int(offsets[index]), int(offsets[index + 1])
        matrix[index] = _event_metrics(
            timeline=arrays["flat_global_t_ms"][left:right],
            values=arrays["flat_pointer_continuous"][left:right],
            contact=np.asarray(arrays["flat_contact_mask"][left:right], dtype=bool),
            event_ids=np.asarray(arrays["flat_event_ids"][left:right], dtype=np.int64),
        )
    return {
        "metrics": matrix, "labels": np.asarray(arrays["label"], dtype=np.int64),
        "pools": np.asarray(arrays["pool"]).astype(str),
        "user_ids": np.asarray(arrays["user_id"], dtype=np.int64),
        "sample_ids": np.asarray(arrays["sample_id"]).astype(str),
    }


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    levels = (0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0)
    observed = np.quantile(values, levels)
    return {("p%g" % (level * 100.0)): float(value) for level, value in zip(levels, observed)}


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    x = np.sort(np.asarray(left, dtype=np.float64))
    y = np.sort(np.asarray(right, dtype=np.float64))
    points = np.unique(np.concatenate((x, y)))
    return float(np.max(np.abs(
        np.searchsorted(x, points, side="right") / len(x)
        - np.searchsorted(y, points, side="right") / len(y)
    )))


def _comparison(real: np.ndarray, fake: np.ndarray) -> Dict[str, Any]:
    lo, q25, median, q75, hi, p99 = np.quantile(real, [0.001, 0.25, 0.5, 0.75, 0.999, 0.99])
    fake_median = float(np.median(fake))
    iqr = float(q75 - q25)
    robust_z = abs(fake_median - float(median)) / max(iqr, abs(float(median)) * 0.01, 1.0e-9)
    outlier_rate = float(np.mean((fake < lo) | (fake > hi)))
    fake_p99 = float(np.quantile(fake, 0.99))
    p99_ratio = fake_p99 / float(p99) if abs(float(p99)) > 1.0e-12 else None
    return {
        "real_train": _quantiles(real), "fake_all": _quantiles(fake),
        "ks_statistic": _ks_statistic(real, fake),
        "fake_outside_real_p0.1_p99.9_rate": outlier_rate,
        "median_robust_iqr_distance": float(robust_z), "fake_to_real_p99_ratio": p99_ratio,
    }


def audit_trajectory_kinematics(
    *, bundle_dir: Path, output_path: Path, actions: Tuple[str, ...] = (
        "tap", "scroll", "swipe", "pinch", "keystroke"
    ),
) -> Dict[str, Any]:
    action_reports = {}
    violations = []
    hashes = {}
    for action in actions:
        path = Path(bundle_dir) / (action + ".npz")
        data = extract_bundle_kinematics(path, expected_action=action)
        hashes[action] = sha256_file(path)
        real_train = (data["labels"] == 0) & (data["pools"] == "train")
        fake_all = data["labels"] == 1
        if not np.any(real_train) or not np.any(fake_all):
            raise ValueError("kinematic audit requires real-train and fake events")
        comparisons = {}
        for index, name in enumerate(KINEMATIC_METRICS):
            comparison = _comparison(data["metrics"][real_train, index], data["metrics"][fake_all, index])
            comparisons[name] = comparison
            if name in PRIMARY_GATES:
                if comparison["fake_outside_real_p0.1_p99.9_rate"] > 0.20:
                    violations.append("%s:%s outlier_rate>0.20" % (action, name))
                if comparison["median_robust_iqr_distance"] > 8.0:
                    violations.append("%s:%s median_robust_distance>8" % (action, name))
                ratio = comparison["fake_to_real_p99_ratio"]
                if ratio is not None and (ratio < 0.05 or ratio > 20.0):
                    violations.append("%s:%s p99_ratio_outside_[0.05,20]" % (action, name))
        duration = data["metrics"][real_train, KINEMATIC_METRICS.index("duration_ms")]
        cuts = np.unique(np.quantile(duration, [0.25, 0.5, 0.75]))
        duration_bins = []
        all_duration = data["metrics"][:, KINEMATIC_METRICS.index("duration_ms")]
        bin_ids = np.searchsorted(cuts, all_duration, side="right")
        for bin_index in range(len(cuts) + 1):
            duration_bins.append({
                "bin": bin_index,
                "lower_ms": None if bin_index == 0 else float(cuts[bin_index - 1]),
                "upper_ms": None if bin_index == len(cuts) else float(cuts[bin_index]),
                "real_train_n": int(np.sum(real_train & (bin_ids == bin_index))),
                "fake_n": int(np.sum(fake_all & (bin_ids == bin_index))),
            })
        action_reports[action] = {
            "rows": int(len(data["labels"])), "real_train_rows": int(np.sum(real_train)),
            "fake_rows": int(np.sum(fake_all)), "metrics": comparisons,
            "duration_bins_fit_on_real_train": duration_bins,
        }
    report = {
        "schema_version": "trajectory_kinematics_human_likeness_audit_v1",
        "passed": not violations, "violations": violations,
        "gate_policy": {
            "primary_metrics": list(PRIMARY_GATES), "max_fake_real_tail_outlier_rate": 0.20,
            "max_median_robust_iqr_distance": 8.0, "allowed_fake_to_real_p99_ratio": [0.05, 20.0],
            "reference": "real detector-train events only", "evaluation": "all generated events",
        },
        "bundle_dir": str(Path(bundle_dir).resolve()), "bundle_sha256": hashes,
        "actions": action_reports,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    report["output_sha256"] = sha256_file(target)
    return report


__all__ = ["KINEMATIC_METRICS", "extract_bundle_kinematics", "audit_trajectory_kinematics"]
