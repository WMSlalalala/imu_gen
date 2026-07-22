"""Independent, fail-closed re-audit of one formal total detector output."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from detectors.feature_pad import (
    fa_frr_curve,
    operating_metrics,
    select_validation_thresholds,
    user_level_bootstrap,
)

from .duration_metrics import duration_stratified_metrics, fit_duration_bins
from .fake_imu_pairs import sha256_file
from .paired_dataset import PairedDetectorTable
from .total_detector import (
    ACCEPTANCE_RULE,
    ARTIFACT_SCHEMA,
    SCORE_DIRECTION,
    TotalDetectorArtifact,
    _score_from_detector,
)
from .trajectory_duration_report import OPERATING_POINTS, _validate_pool_report
from .runtime_benchmark import validate_total_latency_report


TOTAL_DETECTOR_REAUDIT_SCHEMA = "total_detector_formal_reaudit_v1"
TOTAL_DURATION_SCHEMA = "total_detector_duration_report_v1"


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def _finite_thresholds(value: Mapping[str, Any]) -> Dict[str, float]:
    if not isinstance(value, dict) or not set(OPERATING_POINTS).issubset(value):
        raise ValueError("total detector lacks required validation thresholds")
    result = {name: float(value[name]) for name in OPERATING_POINTS}
    if not all(math.isfinite(item) for item in result.values()):
        raise ValueError("total detector thresholds are non-finite")
    return result


def _require_metric_equal(observed: Mapping[str, Any], expected: Mapping[str, Any], context: str) -> None:
    if set(observed) != set(expected):
        raise ValueError("total detector metric schema mismatch: %s" % context)
    for name, expected_value in expected.items():
        observed_value = observed[name]
        if isinstance(expected_value, (int, np.integer)):
            if int(observed_value) != int(expected_value):
                raise ValueError("total detector metric count mismatch: %s/%s" % (context, name))
        else:
            if not math.isclose(
                float(observed_value), float(expected_value), rel_tol=0.0, abs_tol=1.0e-15,
            ):
                raise ValueError("total detector metric value mismatch: %s/%s" % (context, name))


def validate_total_detector_outputs(
    root: Path, *, dataset_path: Path, expected_action: str,
    expected_bootstrap_replicates: int = 500, expected_duration_bins: int = 4,
    require_runtime_latency: bool = False, expected_latency_iterations: int = 1000,
    expected_latency_warmup_iterations: int = 20,
) -> Dict[str, Any]:
    """Recompute the complete formal evidence chain from model and paired data."""

    root = Path(root).resolve()
    dataset_path = Path(dataset_path).resolve()
    required = {
        "total_detector.joblib", "summary.json", "score_dump.npz", "curves.npz",
        "bootstrap_summary.json", "bootstrap_replicates.npz",
        "duration_stratified_metrics.json", "training_manifest.json",
    }
    if require_runtime_latency:
        required.add("runtime_latency.json")
    missing = [name for name in sorted(required) if not (root / name).is_file()]
    if missing:
        raise ValueError("total detector formal outputs are missing: %s" % missing)
    table = PairedDetectorTable.load(dataset_path)
    action_mask = table.actions == expected_action
    if not np.any(action_mask) or set(table.actions[action_mask].tolist()) != {expected_action}:
        raise ValueError("paired dataset action mismatch")
    summary = _read_json(root / "summary.json")
    if (
        summary.get("schema_version") != ARTIFACT_SCHEMA
        or summary.get("action") != expected_action
        or summary.get("score_direction") != SCORE_DIRECTION
        or summary.get("acceptance_rule") != ACCEPTANCE_RULE
        or summary.get("normalization_fit_pool") != "train_only"
        or summary.get("threshold_selection_pool") != "validation_only"
        or summary.get("requires_paired_sample_id") is not True
        or summary.get("requires_consistency") is not True
        or tuple(summary.get("feature_names", ())) != table.feature_names
    ):
        raise ValueError("total detector summary protocol mismatch")
    manifest = _read_json(root / "training_manifest.json")
    if (
        manifest.get("schema_version") != "total_detector_training_manifest_v1"
        or manifest.get("status") != "complete"
        or manifest.get("action") != expected_action
        or Path(manifest.get("dataset", "")).resolve() != dataset_path
        or manifest.get("dataset_sha256") != sha256_file(dataset_path)
        or int(manifest.get("bootstrap_replicates", -1)) != int(expected_bootstrap_replicates)
        or int(manifest.get("duration_bins", -1)) != int(expected_duration_bins)
        or manifest.get("normalization_fit_pool") != "train_only"
        or manifest.get("threshold_selection_pool") != "validation_only"
        or manifest.get("test_role") != "fixed_threshold_reporting_only"
        or tuple(manifest.get("feature_names", ())) != table.feature_names
        or int(manifest.get("rows", -1)) != int(np.sum(action_mask))
    ):
        raise ValueError("total detector training manifest mismatch")

    artifact = TotalDetectorArtifact.load(root / "total_detector.joblib")
    if (
        artifact.action != expected_action
        or artifact.model_kind != summary.get("model_kind")
        or artifact.feature_names != table.feature_names
        or artifact.requires_consistency is not True
    ):
        raise ValueError("total detector artifact metadata mismatch")
    train_rows = np.flatnonzero(action_mask & (table.pools == "train"))
    expected_mean = np.mean(table.features[train_rows], axis=0)
    expected_var = np.var(table.features[train_rows], axis=0)
    if (
        not np.allclose(artifact.scaler.mean_, expected_mean, rtol=1.0e-12, atol=1.0e-12)
        or not np.allclose(artifact.scaler.var_, expected_var, rtol=1.0e-12, atol=1.0e-12)
    ):
        raise ValueError("total detector scaler is not the exact train-only fit")

    with np.load(str(root / "score_dump.npz"), allow_pickle=False) as source:
        expected_score_keys = {
            "%s_%s" % (pool, name)
            for pool in ("val", "test")
            for name in ("score", "label", "user_id", "sample_id")
        }
        if set(source.files) != expected_score_keys:
            raise ValueError("total detector score dump schema mismatch")
        scores = {name: np.asarray(source[name]) for name in source.files}
    for pool in ("val", "test"):
        rows = np.flatnonzero(action_mask & (table.pools == pool))
        score = np.asarray(scores[pool + "_score"], dtype=np.float64)
        if (
            score.shape != (len(rows),)
            or not np.all(np.isfinite(score))
            or not np.array_equal(np.asarray(scores[pool + "_label"], dtype=np.int64), table.labels[rows])
            or not np.array_equal(np.asarray(scores[pool + "_user_id"]), table.user_ids[rows])
            or not np.array_equal(np.asarray(scores[pool + "_sample_id"]).astype(str), table.sample_ids[rows])
        ):
            raise ValueError("total detector score dump does not exactly relink to paired dataset: %s" % pool)
        model_score = _score_from_detector(
            artifact.model_kind,
            artifact.model,
            artifact.scaler.transform(table.features[rows]),
        )
        if not np.array_equal(score, model_score):
            raise ValueError("total detector saved scores do not equal model inference: %s" % pool)

    selected_thresholds = select_validation_thresholds(
        scores["val_label"], scores["val_score"], target_frr=0.05,
    )
    thresholds = _finite_thresholds(summary.get("thresholds", {}))
    if summary.get("thresholds") != selected_thresholds:
        raise ValueError("total detector thresholds are not the exact validation recomputation")
    if dict(artifact.thresholds) != selected_thresholds:
        raise ValueError("total detector artifact thresholds differ from summary/validation")
    for pool, summary_key in (("val", "validation_metrics"), ("test", "test_metrics")):
        if set(summary.get(summary_key, {})) != set(OPERATING_POINTS):
            raise ValueError("total detector summary operating-point closure mismatch")
        artifact_metrics = artifact.validation_metrics if pool == "val" else artifact.test_metrics
        for point in OPERATING_POINTS:
            expected_metric = operating_metrics(
                scores[pool + "_label"], scores[pool + "_score"], thresholds[point],
            )
            _require_metric_equal(summary[summary_key][point], expected_metric, "%s/%s" % (pool, point))
            _require_metric_equal(artifact_metrics[point], expected_metric, "artifact/%s/%s" % (pool, point))

    with np.load(str(root / "curves.npz"), allow_pickle=False) as source:
        curve_arrays = {name: np.asarray(source[name]) for name in source.files}
    expected_curve_keys = {
        "%s_%s" % (pool, name)
        for pool in ("val", "test") for name in ("threshold", "fa", "frr")
    }
    if set(curve_arrays) != expected_curve_keys:
        raise ValueError("total detector saved curve schema mismatch")
    for pool in ("val", "test"):
        expected_curve = fa_frr_curve(scores[pool + "_label"], scores[pool + "_score"])
        for name in ("threshold", "fa", "frr"):
            if not np.array_equal(curve_arrays[pool + "_" + name], expected_curve[name]):
                raise ValueError("total detector saved curve differs from score dump")

    bootstrap_summary = _read_json(root / "bootstrap_summary.json")
    expected_seed = int(manifest["seed"]) + 17
    if (
        int(bootstrap_summary.get("n_replicates", -1)) != int(expected_bootstrap_replicates)
        or int(bootstrap_summary.get("seed", -1)) != expected_seed
        or bootstrap_summary.get("protocol")
        != "separate_real_fake_user_resampling_all_windows_fixed_val_thresholds"
    ):
        raise ValueError("total detector bootstrap protocol/count/seed mismatch")
    recomputed_bootstrap = user_level_bootstrap(
        scores["test_label"], scores["test_score"], scores["test_user_id"],
        thresholds, n_replicates=int(expected_bootstrap_replicates), seed=expected_seed,
    )
    expected_bootstrap_summary = {
        key: value for key, value in recomputed_bootstrap.items() if key != "replicates"
    }
    if bootstrap_summary != expected_bootstrap_summary:
        raise ValueError("total detector bootstrap summary differs from fixed-seed recomputation")
    with np.load(str(root / "bootstrap_replicates.npz"), allow_pickle=False) as source:
        replicate_arrays = {name: np.asarray(source[name]) for name in source.files}
    if set(replicate_arrays) != set(recomputed_bootstrap["replicates"]):
        raise ValueError("total detector bootstrap replicate schema mismatch")
    for name, expected in recomputed_bootstrap["replicates"].items():
        observed = np.asarray(replicate_arrays[name], dtype=np.float64)
        if observed.shape != (int(expected_bootstrap_replicates),) or not np.array_equal(observed, expected):
            raise ValueError("total detector bootstrap replicates differ from recomputation: %s" % name)

    duration = _read_json(root / "duration_stratified_metrics.json")
    expected_bin_spec = fit_duration_bins(
        table.duration_ms[train_rows], n_bins=int(expected_duration_bins),
    )
    if (
        duration.get("schema_version") != TOTAL_DURATION_SCHEMA
        or duration.get("action") != expected_action
        or duration.get("bin_spec") != expected_bin_spec
        or set(duration.get("pools", {})) != {"val", "test"}
    ):
        raise ValueError("total detector duration report protocol/bin mismatch")
    for pool in ("val", "test"):
        rows = np.flatnonzero(action_mask & (table.pools == pool))
        _validate_pool_report(
            duration["pools"][pool], pool=pool, bin_spec=expected_bin_spec,
            thresholds=thresholds, expected_labels=table.labels[rows],
        )
        recomputed_duration = duration_stratified_metrics(
            labels=table.labels[rows],
            scores=scores[pool + "_score"],
            duration_ms=table.duration_ms[rows],
            thresholds=thresholds,
            bin_spec=expected_bin_spec,
            pool=pool,
        )
        if duration["pools"][pool] != recomputed_duration:
            raise ValueError(
                "total detector duration metrics differ from paired-data recomputation: %s" % pool
            )

    runtime_latency = None
    if require_runtime_latency:
        runtime_latency = validate_total_latency_report(
            root / "runtime_latency.json",
            artifact_path=root / "total_detector.joblib",
            dataset_path=dataset_path,
            expected_action=expected_action,
            expected_iterations=expected_latency_iterations,
            expected_warmup_iterations=expected_latency_warmup_iterations,
        )

    hashes = {
        name: sha256_file(root / name)
        for name in sorted(required)
    }
    return {
        "schema_version": TOTAL_DETECTOR_REAUDIT_SCHEMA,
        "passed": True,
        "action": expected_action,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "row_counts": {
            pool: int(np.sum(action_mask & (table.pools == pool)))
            for pool in ("train", "val", "test")
        },
        "model_inference_exact": True,
        "train_only_scaler_exact": True,
        "validation_threshold_recomputation_exact": True,
        "score_identity_relink_exact": True,
        "curves_recomputed_exact": True,
        "bootstrap_recomputed_exact": True,
        "bootstrap_replicates": int(expected_bootstrap_replicates),
        "duration_report_recomputed_exact": True,
        "runtime_latency_validated": bool(require_runtime_latency),
        "runtime_latency": runtime_latency,
        "bin_spec": expected_bin_spec,
        "thresholds": thresholds,
        "artifact_sha256": hashes,
    }


__all__ = ["TOTAL_DETECTOR_REAUDIT_SCHEMA", "validate_total_detector_outputs"]
