"""Total IMU+trajectory detector for paired runtime estimates.

The total detector is intentionally a second-stage detector over already
validation-calibrated modality outputs.  It requires one shared sample identity
per row, so IMU and trajectory signals must be produced for the same event.
Historical independent score dumps without a common ``sample_id`` are rejected
for formal use instead of being silently aligned by row order.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TRAJECTORY_PROJECT = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713"
)
if str(TRAJECTORY_PROJECT) not in sys.path:
    sys.path.insert(0, str(TRAJECTORY_PROJECT))

from detectors.feature_pad import (  # noqa: E402
    FAKE_LABEL,
    REAL_LABEL,
    fa_frr_curve,
    operating_metrics,
    select_validation_thresholds,
    user_level_bootstrap,
)


ARTIFACT_SCHEMA = "imu_trajectory_total_detector_artifact_v1"
SCORE_DIRECTION = "fake_high"
ACCEPTANCE_RULE = "score < threshold"
ALLOWED_MODELS = ("logistic", "linear_svm", "rbf_svm")


def _finite_2d(features: Sequence[Sequence[float]]) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must be a non-empty [N,D] matrix")
    if not np.all(np.isfinite(x)):
        raise ValueError("features contain non-finite values")
    return x


def _as_1d(values: Sequence[Any], n: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != n:
        raise ValueError("%s must be a 1-D array of length %d" % (name, n))
    return array


def _require_both(labels: np.ndarray, context: str) -> None:
    if set(np.unique(labels).astype(int).tolist()) != {REAL_LABEL, FAKE_LABEL}:
        raise ValueError("%s must contain both real=0 and fake=1" % context)


def _score_from_detector(model_kind: str, model: Any, scaled: np.ndarray) -> np.ndarray:
    if model_kind in ("linear_svm", "rbf_svm"):
        return np.asarray(model.decision_function(scaled), dtype=np.float64).reshape(-1)
    classes = np.asarray(model.classes_, dtype=np.int64)
    if classes.tolist() != [REAL_LABEL, FAKE_LABEL]:
        raise RuntimeError("total detector class order is not [0,1]")
    return np.asarray(model.predict_proba(scaled)[:, 1], dtype=np.float64).reshape(-1)


def _make_model(model_kind: str, seed: int, params: Optional[Mapping[str, Any]] = None) -> Any:
    if model_kind not in ALLOWED_MODELS:
        raise ValueError("model_kind must be one of %r" % (ALLOWED_MODELS,))
    values = dict(params or {})
    if model_kind == "logistic":
        defaults: Dict[str, Any] = {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": int(seed),
        }
        defaults.update(values)
        return LogisticRegression(**defaults)
    defaults = {
        "kernel": "linear" if model_kind == "linear_svm" else "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "probability": False,
        "random_state": int(seed),
    }
    defaults.update(values)
    return SVC(**defaults)


def detector_result_features(
    result: Mapping[str, Any],
    *,
    prefix: str,
    include_points: Sequence[str] = ("eer", "val_frr_le_5pct"),
) -> Tuple[List[str], List[float]]:
    """Convert one modality estimator result into scale-safe fusion features.

    The raw detector score is retained, but the main comparable signals are
    margins relative to validation-selected thresholds and binary rejection
    flags.  These fields are available at runtime and do not use test labels.
    """

    names: List[str] = []
    values: List[float] = []
    detectors = list(result.get("detectors", []))
    for detector_index, row in enumerate(detectors):
        family = str(row.get("family", "unknown"))
        detector = str(row.get("detector", detector_index))
        base = "%s__%s__%s" % (prefix, family, detector)
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError("non-finite modality score for %s" % base)
        names.append(base + "__score")
        values.append(score)
        decisions = row.get("decisions", {})
        for point in include_points:
            if point not in decisions:
                continue
            detail = decisions[point]
            margin = float(detail["margin_score_minus_threshold"])
            reject = 1.0 if bool(detail["rejected_as_fake"]) else 0.0
            if not math.isfinite(margin):
                raise ValueError("non-finite modality margin for %s/%s" % (base, point))
            names.extend([base + "__%s_margin" % point, base + "__%s_reject" % point])
            values.extend([margin, reject])
    if not names:
        raise ValueError("estimator result contains no detector rows for %s" % prefix)
    return names, values


def paired_estimate_feature_row(
    *,
    imu_result: Mapping[str, Any],
    trajectory_result: Mapping[str, Any],
    consistency_feature_names: Optional[Sequence[str]] = None,
    consistency_features: Optional[Sequence[float]] = None,
) -> Tuple[List[str], np.ndarray]:
    """Build one total-detector row from modality scores and physical consistency.

    ``consistency_*`` is optional for backward compatibility with old score-
    fusion smoke tests.  Formal paired evaluation must provide it; the runtime
    pack computes these values directly from the shared EventPlan, IMU samples
    and touch trajectory.
    """

    imu_names, imu_values = detector_result_features(imu_result, prefix="imu")
    traj_names, traj_values = detector_result_features(trajectory_result, prefix="trajectory")
    action_i = str(imu_result.get("action", ""))
    action_t = str(trajectory_result.get("action", ""))
    if action_i and action_t and action_i != action_t:
        raise ValueError("paired modalities disagree on action: %s vs %s" % (action_i, action_t))
    names = imu_names + traj_names
    values = imu_values + traj_values
    if (consistency_feature_names is None) != (consistency_features is None):
        raise ValueError("consistency feature names/values must be supplied together")
    if consistency_feature_names is not None:
        extra_names = [str(name) for name in consistency_feature_names]
        extra = np.asarray(consistency_features, dtype=np.float64).reshape(-1)
        if len(extra_names) != extra.size or not np.all(np.isfinite(extra)):
            raise ValueError("invalid cross-modal consistency feature vector")
        if any(not name.startswith("consistency__") for name in extra_names):
            raise ValueError("cross-modal feature names must use consistency__ prefix")
        if len(set(names + extra_names)) != len(names) + len(extra_names):
            raise ValueError("duplicate total-detector feature name")
        names.extend(extra_names)
        values.extend(extra.tolist())
    return names, np.asarray(values, dtype=np.float64)


@dataclass
class TotalDetectorArtifact:
    action: str
    model_kind: str
    model: Any
    scaler: StandardScaler
    feature_names: Tuple[str, ...]
    thresholds: Mapping[str, float]
    validation_metrics: Mapping[str, Mapping[str, float]]
    test_metrics: Mapping[str, Mapping[str, float]]
    artifact_path: Path
    requires_consistency: bool = True

    @classmethod
    def train(
        cls,
        *,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
        user_ids: Sequence[Any],
        pools: Sequence[str],
        sample_ids: Sequence[Any],
        action: str,
        feature_names: Sequence[str],
        model_kind: str = "logistic",
        model_params: Optional[Mapping[str, Any]] = None,
        seed: int = 20260721,
        bootstrap_replicates: int = 0,
        require_consistency: bool = True,
    ) -> "TotalDetectorArtifact":
        x = _finite_2d(features)
        n = x.shape[0]
        y = _as_1d(labels, n, "labels").astype(np.int64)
        users = _as_1d(user_ids, n, "user_ids")
        pool = _as_1d(pools, n, "pools").astype(str)
        sample = _as_1d(sample_ids, n, "sample_ids").astype(str)
        if len(tuple(feature_names)) != x.shape[1]:
            raise ValueError("feature_names length does not match feature matrix width")
        if require_consistency and not any(
            str(name).startswith("consistency__") for name in feature_names
        ):
            raise ValueError(
                "formal total detector requires physical consistency features, not score concatenation alone"
            )
        if set(np.unique(y).tolist()) - {REAL_LABEL, FAKE_LABEL}:
            raise ValueError("labels must use real=0/fake=1")
        if len(set(sample.tolist())) != len(sample):
            raise ValueError("sample_ids must be unique; total detector needs paired events")
        split_index: Dict[str, np.ndarray] = {}
        for name in ("train", "val", "test"):
            idx = np.flatnonzero(pool == name)
            if idx.size == 0:
                raise ValueError("missing %s rows" % name)
            _require_both(y[idx], "%s split" % name)
            split_index[name] = idx

        scaler = StandardScaler()
        train_idx = split_index["train"]
        train_x = scaler.fit_transform(x[train_idx])
        model = _make_model(model_kind, seed, model_params)
        model.fit(train_x, y[train_idx])
        classes = np.asarray(getattr(model, "classes_", []), dtype=np.int64)
        if classes.tolist() != [REAL_LABEL, FAKE_LABEL]:
            raise RuntimeError("total detector class order is not [0,1]")

        scores: Dict[str, np.ndarray] = {}
        for name in ("val", "test"):
            idx = split_index[name]
            scores[name] = _score_from_detector(model_kind, model, scaler.transform(x[idx]))

        thresholds = select_validation_thresholds(y[split_index["val"]], scores["val"], target_frr=0.05)
        selected = {"eer": thresholds["eer"], "val_frr_le_5pct": thresholds["val_frr_le_5pct"]}
        validation_metrics = {
            key: operating_metrics(y[split_index["val"]], scores["val"], threshold)
            for key, threshold in selected.items()
        }
        test_metrics = {
            key: operating_metrics(y[split_index["test"]], scores["test"], threshold)
            for key, threshold in selected.items()
        }
        bootstrap = None
        if int(bootstrap_replicates) > 0:
            bootstrap = user_level_bootstrap(
                y[split_index["test"]],
                scores["test"],
                users[split_index["test"]],
                selected,
                n_replicates=int(bootstrap_replicates),
                seed=int(seed) + 17,
            )
        artifact = cls(
            action=str(action),
            model_kind=str(model_kind),
            model=model,
            scaler=scaler,
            feature_names=tuple(str(name) for name in feature_names),
            thresholds=thresholds,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            artifact_path=Path(""),
            requires_consistency=bool(require_consistency),
        )
        artifact._training_cache = {
            "score_dumps": {
                name: {
                    "score": scores[name],
                    "label": y[split_index[name]],
                    "user_id": users[split_index[name]],
                    "sample_id": sample[split_index[name]],
                }
                for name in ("val", "test")
            },
            "curves": {
                name: fa_frr_curve(y[split_index[name]], scores[name])
                for name in ("val", "test")
            },
            "bootstrap": bootstrap,
            "model_params": dict(model_params or {}),
            "seed": int(seed),
        }
        return artifact

    def save(self, path: Path) -> None:
        payload = {
            "schema_version": ARTIFACT_SCHEMA,
            "action": self.action,
            "model_kind": self.model_kind,
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": list(self.feature_names),
            "thresholds": dict(self.thresholds),
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "score_direction": SCORE_DIRECTION,
            "acceptance_rule": ACCEPTANCE_RULE,
            "normalization_fit_pool": "train_only",
            "threshold_selection_pool": "validation_only",
            "requires_consistency": bool(self.requires_consistency),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        joblib.dump(payload, temporary)
        temporary.replace(path)
        object.__setattr__(self, "artifact_path", path)

    @classmethod
    def load(cls, path: Path) -> "TotalDetectorArtifact":
        payload = joblib.load(Path(path))
        if payload.get("schema_version") != ARTIFACT_SCHEMA:
            raise ValueError("total detector artifact schema mismatch")
        if payload.get("score_direction") != SCORE_DIRECTION:
            raise ValueError("total detector score direction mismatch")
        if payload.get("acceptance_rule") != ACCEPTANCE_RULE:
            raise ValueError("total detector acceptance rule mismatch")
        return cls(
            action=str(payload["action"]),
            model_kind=str(payload["model_kind"]),
            model=payload["model"],
            scaler=payload["scaler"],
            feature_names=tuple(str(v) for v in payload["feature_names"]),
            thresholds={str(k): float(v) for k, v in payload["thresholds"].items()},
            validation_metrics=payload.get("validation_metrics", {}),
            test_metrics=payload.get("test_metrics", {}),
            artifact_path=Path(path),
            requires_consistency=bool(payload.get("requires_consistency", False)),
        )

    def score_feature_row(self, feature_row: Sequence[float], feature_names: Sequence[str]) -> float:
        names = tuple(str(v) for v in feature_names)
        if names != self.feature_names:
            raise ValueError("total detector feature order/name mismatch")
        x = np.asarray(feature_row, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != len(self.feature_names) or not np.all(np.isfinite(x)):
            raise ValueError("invalid total detector feature row")
        score = _score_from_detector(self.model_kind, self.model, self.scaler.transform(x))[0]
        if not math.isfinite(float(score)):
            raise RuntimeError("total detector produced non-finite score")
        return float(score)

    def estimate_pair(
        self,
        *,
        imu_result: Mapping[str, Any],
        trajectory_result: Mapping[str, Any],
        consistency_feature_names: Optional[Sequence[str]] = None,
        consistency_features: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        names, row = paired_estimate_feature_row(
            imu_result=imu_result, trajectory_result=trajectory_result,
            consistency_feature_names=consistency_feature_names,
            consistency_features=consistency_features,
        )
        if self.requires_consistency and consistency_features is None:
            raise ValueError("this total detector requires cross-modal consistency features")
        score = self.score_feature_row(row, names)
        decisions = {}
        for point in ("eer", "val_frr_le_5pct"):
            if point not in self.thresholds:
                continue
            threshold = float(self.thresholds[point])
            accepted = bool(score < threshold)
            decisions[point] = {
                "threshold": threshold,
                "accepted_as_real": accepted,
                "rejected_as_fake": not accepted,
                "margin_score_minus_threshold": score - threshold,
                "acceptance_rule": ACCEPTANCE_RULE,
            }
        return {
            "schema_version": "imu_trajectory_total_detector_result_v1",
            "action": self.action,
            "family": "total_imu_trajectory_pad",
            "model_kind": self.model_kind,
            "score": score,
            "score_direction": SCORE_DIRECTION,
            "acceptance_rule": ACCEPTANCE_RULE,
            "decisions": decisions,
            "latency_ms": float((time.perf_counter() - started) * 1000.0),
        }


def write_training_outputs(root: Path, artifact: TotalDetectorArtifact) -> Dict[str, str]:
    """Persist artifact plus val/test score dumps and a compact summary."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    artifact_path = root / "total_detector.joblib"
    artifact.save(artifact_path)
    cache = getattr(artifact, "_training_cache", {})
    summary = {
        "schema_version": ARTIFACT_SCHEMA,
        "action": artifact.action,
        "model_kind": artifact.model_kind,
        "feature_names": list(artifact.feature_names),
        "thresholds": dict(artifact.thresholds),
        "validation_metrics": artifact.validation_metrics,
        "test_metrics": artifact.test_metrics,
        "score_direction": SCORE_DIRECTION,
        "acceptance_rule": ACCEPTANCE_RULE,
        "normalization_fit_pool": "train_only",
        "threshold_selection_pool": "validation_only",
        "requires_paired_sample_id": True,
        "requires_consistency": bool(artifact.requires_consistency),
        "modality_policy": "paired_imu_plus_trajectory_scores_plus_physical_consistency",
    }
    paths: Dict[str, str] = {"artifact": str(artifact_path)}
    summary_path = root / "summary.json"
    tmp = summary_path.with_name(summary_path.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(summary_path)
    paths["summary"] = str(summary_path)
    if "score_dumps" in cache:
        arrays = {}
        for pool, dump in cache["score_dumps"].items():
            for key, value in dump.items():
                arrays["%s_%s" % (pool, key)] = np.asarray(value)
        score_path = root / "score_dump.npz"
        tmp_npz = score_path.with_name(score_path.name + ".tmp")
        with tmp_npz.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp_npz.replace(score_path)
        paths["score_dump"] = str(score_path)
    if "curves" in cache:
        arrays = {}
        for pool, curve in cache["curves"].items():
            for key, value in curve.items():
                arrays["%s_%s" % (pool, key)] = np.asarray(value)
        curve_path = root / "curves.npz"
        tmp_npz = curve_path.with_name(curve_path.name + ".tmp")
        with tmp_npz.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp_npz.replace(curve_path)
        paths["curves"] = str(curve_path)
    bootstrap = cache.get("bootstrap")
    if bootstrap is not None:
        bootstrap_summary = {
            key: value for key, value in bootstrap.items() if key != "replicates"
        }
        bootstrap_path = root / "bootstrap_summary.json"
        temporary_json = bootstrap_path.with_name(bootstrap_path.name + ".tmp")
        temporary_json.write_text(
            json.dumps(bootstrap_summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary_json.replace(bootstrap_path)
        paths["bootstrap_summary"] = str(bootstrap_path)
        replicates_path = root / "bootstrap_replicates.npz"
        temporary_npz = replicates_path.with_name(replicates_path.name + ".tmp")
        with temporary_npz.open("wb") as handle:
            np.savez_compressed(
                handle,
                **{key: np.asarray(value) for key, value in bootstrap["replicates"].items()}
            )
        temporary_npz.replace(replicates_path)
        paths["bootstrap_replicates"] = str(replicates_path)
    return paths


__all__ = [
    "TotalDetectorArtifact",
    "detector_result_features",
    "paired_estimate_feature_row",
    "write_training_outputs",
]
