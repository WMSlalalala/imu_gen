"""Auditable runtime-latency benchmarks for trajectory and total estimators."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from detectors.deep_pad import load_raw_sequence_bundle

from .fake_imu_pairs import sha256_file
from .paired_dataset import PairedDetectorTable
from .service import TrajectoryEstimatorService
from .total_detector import TotalDetectorArtifact


TRAJECTORY_LATENCY_SCHEMA = "trajectory_estimator_runtime_latency_v1"
TOTAL_LATENCY_SCHEMA = "total_detector_runtime_latency_v1"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _latency_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("latency samples must be a finite nonnegative vector")
    return {
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.percentile(array, 50.0)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "max_ms": float(np.max(array)),
    }


def _validate_latency_summary(value: Mapping[str, Any], context: str) -> None:
    if set(value) != {"mean_ms", "p50_ms", "p95_ms", "max_ms"}:
        raise ValueError("latency summary schema mismatch: %s" % context)
    numbers = {name: float(item) for name, item in value.items()}
    if not all(math.isfinite(item) and item >= 0 for item in numbers.values()):
        raise ValueError("latency summary contains invalid values: %s" % context)
    if not numbers["p50_ms"] <= numbers["p95_ms"] <= numbers["max_ms"]:
        raise ValueError("latency percentiles are not monotonic: %s" % context)


def _validate_latency_samples(
    samples: Any, summary: Mapping[str, Any], *, expected_count: int, context: str,
) -> None:
    if not isinstance(samples, list) or len(samples) != int(expected_count):
        raise ValueError("latency raw sample count mismatch: %s" % context)
    recomputed = _latency_summary(samples)
    _validate_latency_summary(summary, context)
    if dict(summary) != recomputed:
        raise ValueError("latency summary does not match raw samples: %s" % context)


def benchmark_trajectory_estimator_latency(
    *, manifest_path: Path, bundle_dir: Path, output_path: Path,
    device: str, actions: Sequence[str], iterations_per_action: int = 25,
    warmup_per_action: int = 2, load_deep: bool = True,
) -> Dict[str, Any]:
    if int(iterations_per_action) < 1 or int(warmup_per_action) < 0:
        raise ValueError("trajectory latency iteration counts are invalid")
    manifest_path = Path(manifest_path).resolve()
    bundle_dir = Path(bundle_dir).resolve()
    load_started = time.perf_counter()
    service = TrajectoryEstimatorService.load(
        manifest_path, load_feature=True, load_deep=bool(load_deep), device=device,
    )
    service_load_ms = (time.perf_counter() - load_started) * 1000.0
    if set(service.actions()) != set(actions):
        raise ValueError("trajectory latency service action closure mismatch")
    action_reports: Dict[str, Any] = {}
    bundle_hashes: Dict[str, Any] = {}
    for action in actions:
        bundle = bundle_dir / (str(action) + ".npz")
        records, _ = load_raw_sequence_bundle(bundle)
        test_records = [record for record in records if record.pool == "test"]
        real = [record for record in test_records if record.label == 0]
        fake = [record for record in test_records if record.label == 1]
        if not real or not fake:
            raise ValueError("trajectory latency benchmark requires both test classes")
        candidates = [value for pair in zip(real, fake) for value in pair]
        for index in range(int(warmup_per_action)):
            service.estimate_record(candidates[index % len(candidates)])
        service_samples = []
        wall_samples = []
        detector_samples: Dict[str, list] = {}
        labels = []
        detector_count = None
        for index in range(int(iterations_per_action)):
            record = candidates[index % len(candidates)]
            started = time.perf_counter()
            result = service.estimate_record(record)
            wall_ms = (time.perf_counter() - started) * 1000.0
            rows = result.get("detectors", [])
            if detector_count is None:
                detector_count = len(rows)
            if len(rows) != detector_count or not rows:
                raise ValueError("trajectory latency detector count changed across calls")
            service_samples.append(float(result["latency_ms"]))
            wall_samples.append(float(wall_ms))
            labels.append(int(record.label))
            for row in rows:
                key = "%s/%s" % (row["family"], row["detector"])
                detector_samples.setdefault(key, []).append(float(row["latency_ms"]))
        if set(labels) != {0, 1}:
            raise ValueError("trajectory latency measurements did not cover both labels")
        if any(len(values) != int(iterations_per_action) for values in detector_samples.values()):
            raise ValueError("trajectory latency per-detector sample count mismatch")
        action_reports[str(action)] = {
            "iterations": int(iterations_per_action),
            "warmup_iterations": int(warmup_per_action),
            "detector_count": int(detector_count),
            "labels_covered": [0, 1],
            "service_latency": _latency_summary(service_samples),
            "wall_latency": _latency_summary(wall_samples),
            "service_samples_ms": [float(value) for value in service_samples],
            "wall_samples_ms": [float(value) for value in wall_samples],
            "detector_latency": {
                key: _latency_summary(values) for key, values in sorted(detector_samples.items())
            },
            "detector_samples_ms": {
                key: [float(value) for value in values]
                for key, values in sorted(detector_samples.items())
            },
        }
        bundle_hashes[str(action)] = {
            "path": str(bundle.resolve()), "sha256": sha256_file(bundle),
        }
    report = {
        "schema_version": TRAJECTORY_LATENCY_SCHEMA,
        "status": "passed",
        "device": str(device),
        "load_deep": bool(load_deep),
        "service_load_ms": float(service_load_ms),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "bundle_sources": bundle_hashes,
        "actions": action_reports,
    }
    _atomic_json(output_path, report)
    return report


def validate_trajectory_latency_report(
    path: Path, *, manifest_path: Path, bundle_dir: Path,
    expected_actions: Sequence[str], expected_iterations: int,
    expected_detectors_per_action: int, expected_device: str,
    expected_load_deep: bool, expected_warmup_iterations: int,
) -> Dict[str, Any]:
    value = _read_json(path)
    manifest_path = Path(manifest_path).resolve()
    bundle_dir = Path(bundle_dir).resolve()
    if (
        value.get("schema_version") != TRAJECTORY_LATENCY_SCHEMA
        or value.get("status") != "passed"
        or value.get("device") != str(expected_device)
        or value.get("load_deep") is not bool(expected_load_deep)
        or Path(value.get("manifest", "")).resolve() != manifest_path
        or value.get("manifest_sha256") != sha256_file(manifest_path)
        or set(value.get("actions", {})) != set(expected_actions)
        or set(value.get("bundle_sources", {})) != set(expected_actions)
        or not math.isfinite(float(value.get("service_load_ms", -1)))
        or float(value.get("service_load_ms", -1)) < 0
    ):
        raise ValueError("trajectory estimator latency report protocol mismatch")
    for action in expected_actions:
        bundle = bundle_dir / (str(action) + ".npz")
        source = value["bundle_sources"][action]
        if Path(source.get("path", "")).resolve() != bundle or source.get("sha256") != sha256_file(bundle):
            raise ValueError("trajectory latency bundle source/hash mismatch")
        report = value["actions"][action]
        expected_detector_keys = {
            "feature_pad/linear_svm", "feature_pad/rbf_svm", "feature_pad/xgboost",
            "deep_pad/tcn", "deep_pad/transformer",
        } if int(expected_detectors_per_action) == 5 else None
        if (
            int(report.get("iterations", -1)) != int(expected_iterations)
            or int(report.get("warmup_iterations", -1)) != int(expected_warmup_iterations)
            or int(report.get("detector_count", -1)) != int(expected_detectors_per_action)
            or report.get("labels_covered") != [0, 1]
            or len(report.get("detector_latency", {})) != int(expected_detectors_per_action)
            or set(report.get("detector_samples_ms", {}))
            != set(report.get("detector_latency", {}))
            or (
                expected_detector_keys is not None
                and set(report.get("detector_latency", {})) != expected_detector_keys
            )
        ):
            raise ValueError("trajectory latency action coverage mismatch")
        _validate_latency_samples(
            report.get("service_samples_ms"), report.get("service_latency", {}),
            expected_count=expected_iterations, context=action + "/service",
        )
        _validate_latency_samples(
            report.get("wall_samples_ms"), report.get("wall_latency", {}),
            expected_count=expected_iterations, context=action + "/wall",
        )
        for detector, summary in report["detector_latency"].items():
            _validate_latency_samples(
                report["detector_samples_ms"][detector], summary,
                expected_count=expected_iterations, context=action + "/" + detector,
            )
    return value


def benchmark_total_detector_latency(
    *, artifact_path: Path, dataset_path: Path, expected_action: str,
    output_path: Path, iterations: int = 1000, warmup_iterations: int = 20,
) -> Dict[str, Any]:
    if int(iterations) < 1 or int(warmup_iterations) < 0:
        raise ValueError("total detector latency iteration counts are invalid")
    artifact_path = Path(artifact_path).resolve()
    dataset_path = Path(dataset_path).resolve()
    load_started = time.perf_counter()
    artifact = TotalDetectorArtifact.load(artifact_path)
    artifact_load_ms = (time.perf_counter() - load_started) * 1000.0
    table = PairedDetectorTable.load(dataset_path)
    rows = np.flatnonzero((table.actions == expected_action) & (table.pools == "test"))
    if artifact.action != expected_action or artifact.feature_names != table.feature_names or rows.size == 0:
        raise ValueError("total detector latency artifact/dataset identity mismatch")
    for index in range(int(warmup_iterations)):
        artifact.score_feature_row(table.features[rows[index % len(rows)]], table.feature_names)
    latencies = []
    scores = []
    for index in range(int(iterations)):
        row = rows[index % len(rows)]
        started = time.perf_counter()
        score = artifact.score_feature_row(table.features[row], table.feature_names)
        latencies.append((time.perf_counter() - started) * 1000.0)
        scores.append(score)
    if not np.all(np.isfinite(scores)):
        raise ValueError("total detector latency benchmark produced non-finite scores")
    report = {
        "schema_version": TOTAL_LATENCY_SCHEMA,
        "status": "passed",
        "action": expected_action,
        "iterations": int(iterations),
        "warmup_iterations": int(warmup_iterations),
        "feature_count": int(len(table.feature_names)),
        "test_rows_available": int(len(rows)),
        "artifact_load_ms": float(artifact_load_ms),
        "latency": _latency_summary(latencies),
        "latency_samples_ms": [float(value) for value in latencies],
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
    }
    _atomic_json(output_path, report)
    return report


def validate_total_latency_report(
    path: Path, *, artifact_path: Path, dataset_path: Path,
    expected_action: str, expected_iterations: int,
    expected_warmup_iterations: int = 20,
) -> Dict[str, Any]:
    value = _read_json(path)
    artifact_path = Path(artifact_path).resolve()
    dataset_path = Path(dataset_path).resolve()
    table = PairedDetectorTable.load(dataset_path)
    test_rows = np.flatnonzero(
        (table.actions == expected_action) & (table.pools == "test")
    )
    if (
        value.get("schema_version") != TOTAL_LATENCY_SCHEMA
        or value.get("status") != "passed"
        or value.get("action") != expected_action
        or int(value.get("iterations", -1)) != int(expected_iterations)
        or int(value.get("warmup_iterations", -1)) != int(expected_warmup_iterations)
        or Path(value.get("artifact", "")).resolve() != artifact_path
        or value.get("artifact_sha256") != sha256_file(artifact_path)
        or Path(value.get("dataset", "")).resolve() != dataset_path
        or value.get("dataset_sha256") != sha256_file(dataset_path)
        or int(value.get("feature_count", -1)) != len(table.feature_names)
        or int(value.get("test_rows_available", -1)) != len(test_rows)
        or not math.isfinite(float(value.get("artifact_load_ms", -1)))
        or float(value.get("artifact_load_ms", -1)) < 0
    ):
        raise ValueError("total detector latency report protocol/source mismatch")
    _validate_latency_samples(
        value.get("latency_samples_ms"), value.get("latency", {}),
        expected_count=expected_iterations, context=expected_action,
    )
    return value


__all__ = [
    "TRAJECTORY_LATENCY_SCHEMA", "TOTAL_LATENCY_SCHEMA",
    "benchmark_trajectory_estimator_latency", "validate_trajectory_latency_report",
    "benchmark_total_detector_latency", "validate_total_latency_report",
]
