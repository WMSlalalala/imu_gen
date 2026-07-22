"""Duration-stratified reports for all five base trajectory PAD detectors.

Duration bins are fitted once from the trajectory bundle's train rows.  Every
validation/test slice applies the detector's already selected global validation
thresholds; no duration-bin-specific threshold is fitted.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .duration_metrics import (
    DURATION_REPORT_SCHEMA,
    duration_stratified_metrics,
    fit_duration_bins,
)
from .trajectory_score_component import DETECTORS, _detector_scores, _load_bundle
from .fake_imu_pairs import sha256_file


TRAJECTORY_DURATION_REPORT_SCHEMA = "trajectory_detector_duration_report_v1"
OPERATING_POINTS = ("eer", "val_frr_le_5pct")
RELINK_PROTOCOL = {
    "feature_pad": "score_dump_row_index_to_bundle_row_exact",
    "deep_pad": "score_dump_sample_id_to_bundle_sample_id_exact",
    "metadata_checks": ["label", "user_id", "pool", "action"],
    "coverage": "exact_bundle_validation_and_test_rows",
}


def _detector_key(family: str, detector: str) -> str:
    return "%s/%s" % (family, detector)


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _validated_thresholds(summary: Mapping[str, Any], *, action: str, detector: str) -> Dict[str, float]:
    if (
        summary.get("action") != action
        or summary.get("detector_kind") != detector
        or summary.get("score_direction") != "fake_high"
        or summary.get("acceptance_rule") != "score < threshold"
        or summary.get("threshold_selection_pool") != "validation_only"
    ):
        raise ValueError("trajectory detector summary protocol mismatch: %s/%s" % (action, detector))
    source = summary.get("thresholds")
    if not isinstance(source, dict) or not set(OPERATING_POINTS).issubset(source):
        raise ValueError("trajectory detector summary lacks required validation thresholds")
    thresholds = {name: float(source[name]) for name in OPERATING_POINTS}
    if not all(math.isfinite(value) for value in thresholds.values()):
        raise ValueError("trajectory detector thresholds must be finite")
    return thresholds


def _validate_pool_report(
    report: Mapping[str, Any], *, pool: str, bin_spec: Mapping[str, Any],
    thresholds: Mapping[str, float], expected_labels: np.ndarray,
) -> None:
    if (
        report.get("schema_version") != DURATION_REPORT_SCHEMA
        or report.get("pool") != pool
        or report.get("threshold_source") != "validation_global_not_refit_per_bin"
        or report.get("bin_spec") != dict(bin_spec)
    ):
        raise ValueError("duration pool report protocol mismatch: %s" % pool)
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != int(bin_spec["effective_bins"]):
        raise ValueError("duration pool report bin count mismatch: %s" % pool)
    if [row.get("bin_index") for row in rows] != list(range(len(rows))):
        raise ValueError("duration pool report bin indices mismatch: %s" % pool)
    expected_total = int(len(expected_labels))
    expected_real = int(np.sum(expected_labels == 0))
    expected_fake = int(np.sum(expected_labels == 1))
    if (
        sum(int(row.get("n", -1)) for row in rows) != expected_total
        or sum(int(row.get("n_real", -1)) for row in rows) != expected_real
        or sum(int(row.get("n_fake", -1)) for row in rows) != expected_fake
    ):
        raise ValueError("duration pool report row coverage mismatch: %s" % pool)
    for row in rows:
        n = int(row.get("n", -1))
        n_real = int(row.get("n_real", -1))
        n_fake = int(row.get("n_fake", -1))
        if min(n, n_real, n_fake) < 0 or n_real + n_fake != n:
            raise ValueError("duration pool report class counts are invalid")
        auc = row.get("auc")
        if auc is not None and (not math.isfinite(float(auc)) or not 0.0 <= float(auc) <= 1.0):
            raise ValueError("duration pool report AUC is invalid")
        operating = row.get("operating_points")
        if not isinstance(operating, dict) or set(operating) != set(OPERATING_POINTS):
            raise ValueError("duration pool report operating points mismatch")
        for name in OPERATING_POINTS:
            point = operating[name]
            if not math.isclose(float(point.get("threshold")), thresholds[name], rel_tol=0.0, abs_tol=0.0):
                raise ValueError("duration pool report changed a global validation threshold")
            for metric in ("fa", "frr"):
                value = point.get(metric)
                if value is not None and (
                    not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
                ):
                    raise ValueError("duration pool report contains an invalid %s" % metric)


def build_trajectory_duration_report(
    *, action: str, bundle_path: Path, detector_root: Path,
    output_path: Path, n_bins: int = 4,
) -> Dict[str, Any]:
    """Build and atomically save one action's five-detector duration report."""

    bundle_path = Path(bundle_path).resolve()
    detector_root = Path(detector_root).resolve()
    bundle = _load_bundle(bundle_path, action)
    score_matrix, score_names, score_paths = _detector_scores(
        detector_root=detector_root, bundle=bundle, action=action,
    )
    bundle_rows = np.flatnonzero(np.isin(bundle["pools"], ["val", "test"]))
    if score_matrix.shape != (len(bundle_rows), len(DETECTORS)):
        raise ValueError("trajectory score matrix shape mismatch")
    bin_spec = fit_duration_bins(
        bundle["duration_ms"][bundle["pools"] == "train"], n_bins=int(n_bins),
    )
    detectors: Dict[str, Any] = {}
    for column, ((family, detector), score_path, score_name) in enumerate(
        zip(DETECTORS, score_paths, score_names)
    ):
        result_root = detector_root / action / family / detector / "result"
        expected_score_path = result_root / "score_dump.npz"
        if Path(score_path).resolve() != expected_score_path.resolve():
            raise ValueError("trajectory detector score path is not canonical")
        summary_path = result_root / "summary.json"
        if not summary_path.is_file():
            raise ValueError("trajectory detector summary is missing: %s" % summary_path)
        summary = _read_json(summary_path)
        thresholds = _validated_thresholds(summary, action=action, detector=detector)
        pools: Dict[str, Any] = {}
        for pool in ("val", "test"):
            selected = bundle["pools"][bundle_rows] == pool
            labels = bundle["labels"][bundle_rows][selected]
            if set(labels.tolist()) != {0, 1}:
                raise ValueError("trajectory duration pool must contain both classes: %s" % pool)
            pool_report = duration_stratified_metrics(
                labels=labels,
                scores=score_matrix[selected, column],
                duration_ms=bundle["duration_ms"][bundle_rows][selected],
                thresholds=thresholds,
                bin_spec=bin_spec,
                pool=pool,
            )
            _validate_pool_report(
                pool_report, pool=pool, bin_spec=bin_spec,
                thresholds=thresholds, expected_labels=labels,
            )
            pools[pool] = pool_report
        key = _detector_key(family, detector)
        detectors[key] = {
            "family": family,
            "detector_kind": detector,
            "score_feature_name": score_name,
            "score_dump": str(expected_score_path.resolve()),
            "score_dump_sha256": sha256_file(expected_score_path),
            "summary": str(summary_path.resolve()),
            "summary_sha256": sha256_file(summary_path),
            "thresholds": thresholds,
            "pools": pools,
        }
    report: Dict[str, Any] = {
        "schema_version": TRAJECTORY_DURATION_REPORT_SCHEMA,
        "passed": True,
        "action": action,
        "detector_count": len(detectors),
        "duration_source": "bundle_irregular_global_timeline_last_minus_first_ms",
        "duration_bin_policy": "train_only_quantiles",
        "threshold_policy": "validation_global_not_refit_per_bin",
        "row_relink_protocol": RELINK_PROTOCOL,
        "bundle": str(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "bin_spec": bin_spec,
        "detectors": detectors,
    }
    _atomic_json(output_path, report)
    return report


def validate_trajectory_duration_report(
    path: Path, *, expected_action: str, expected_bundle: Path,
    expected_detector_root: Path, expected_bins: int = 4,
) -> Dict[str, Any]:
    """Fail closed on protocol, coverage, identities, metrics, and source hashes."""

    report = _read_json(path)
    if (
        report.get("schema_version") != TRAJECTORY_DURATION_REPORT_SCHEMA
        or report.get("passed") is not True
        or report.get("action") != expected_action
        or report.get("detector_count") != len(DETECTORS)
        or report.get("duration_bin_policy") != "train_only_quantiles"
        or report.get("threshold_policy") != "validation_global_not_refit_per_bin"
        or report.get("row_relink_protocol") != RELINK_PROTOCOL
    ):
        raise ValueError("trajectory detector duration report protocol mismatch")
    expected_bundle = Path(expected_bundle).resolve()
    expected_detector_root = Path(expected_detector_root).resolve()
    if Path(report.get("bundle", "")).resolve() != expected_bundle:
        raise ValueError("trajectory duration report bundle path mismatch")
    if report.get("bundle_sha256") != sha256_file(expected_bundle):
        raise ValueError("trajectory duration report bundle SHA-256 mismatch")
    bundle = _load_bundle(expected_bundle, expected_action)
    score_matrix, score_names, score_paths = _detector_scores(
        detector_root=expected_detector_root, bundle=bundle, action=expected_action,
    )
    bundle_rows = np.flatnonzero(np.isin(bundle["pools"], ["val", "test"]))
    if score_matrix.shape != (len(bundle_rows), len(DETECTORS)):
        raise ValueError("trajectory duration report score matrix shape mismatch")
    expected_spec = fit_duration_bins(
        bundle["duration_ms"][bundle["pools"] == "train"], n_bins=int(expected_bins),
    )
    if report.get("bin_spec") != expected_spec:
        raise ValueError("trajectory duration report train-only bin spec mismatch")
    detectors = report.get("detectors")
    expected_keys = {_detector_key(family, detector) for family, detector in DETECTORS}
    if not isinstance(detectors, dict) or set(detectors) != expected_keys:
        raise ValueError("trajectory duration report detector closure mismatch")
    for column, (family, detector) in enumerate(DETECTORS):
        key = _detector_key(family, detector)
        value = detectors[key]
        expected_root = expected_detector_root / expected_action / family / detector / "result"
        summary_path = expected_root / "summary.json"
        score_path = expected_root / "score_dump.npz"
        if (
            value.get("family") != family
            or value.get("detector_kind") != detector
            or value.get("score_feature_name") != score_names[column]
            or Path(value.get("summary", "")).resolve() != summary_path.resolve()
            or Path(value.get("score_dump", "")).resolve() != score_path.resolve()
            or Path(score_paths[column]).resolve() != score_path.resolve()
            or value.get("summary_sha256") != sha256_file(summary_path)
            or value.get("score_dump_sha256") != sha256_file(score_path)
        ):
            raise ValueError("trajectory duration report source identity/hash mismatch: %s" % key)
        thresholds = _validated_thresholds(
            _read_json(summary_path), action=expected_action, detector=detector,
        )
        if value.get("thresholds") != thresholds or set(value.get("pools", {})) != {"val", "test"}:
            raise ValueError("trajectory duration report thresholds/pools mismatch: %s" % key)
        for pool in ("val", "test"):
            selected = bundle["pools"][bundle_rows] == pool
            labels = bundle["labels"][bundle_rows][selected]
            _validate_pool_report(
                value["pools"][pool], pool=pool, bin_spec=expected_spec,
                thresholds=thresholds, expected_labels=labels,
            )
            recomputed_duration = duration_stratified_metrics(
                labels=labels,
                scores=score_matrix[selected, column],
                duration_ms=bundle["duration_ms"][bundle_rows][selected],
                thresholds=thresholds,
                bin_spec=expected_spec,
                pool=pool,
            )
            if value["pools"][pool] != recomputed_duration:
                raise ValueError(
                    "trajectory detector duration metrics differ from score recomputation: %s/%s"
                    % (key, pool)
                )
    return report


__all__ = [
    "TRAJECTORY_DURATION_REPORT_SCHEMA",
    "build_trajectory_duration_report",
    "validate_trajectory_duration_report",
]
