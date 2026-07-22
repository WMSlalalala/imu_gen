"""Strict feature-level PAD protocol for pre-extracted trajectory features.

Labels are fixed to real=0 and fake=1.  Every detector score is oriented so a
larger value means "more fake".  The scaler and detector are fitted on train
only; validation alone selects operating thresholds; test only applies those
fixed thresholds.  The authentication decision is deliberately strict:

    score < threshold  -> accept as real
    score >= threshold -> reject as fake

This module operates on plain arrays and has no dependency on a trajectory file
schema or on the feature extractor used to create those arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:  # Optional by protocol: absence must be explicit, never a silent fallback.
    from xgboost import XGBClassifier

    _XGB_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised by a mocked unit test.
    XGBClassifier = None  # type: ignore
    _XGB_IMPORT_ERROR = exc


REAL_LABEL = 0
FAKE_LABEL = 1
ALLOWED_POOLS = ("train", "val", "test")
ALLOWED_DETECTORS = ("linear_svm", "rbf_svm", "xgboost")


@dataclass
class FeaturePADProtocolResult:
    """In-memory result of one action and one detector protocol."""

    action: str
    detector_kind: str
    detector: "FeaturePAD"
    thresholds: Dict[str, float]
    validation_metrics: Dict[str, Dict[str, float]]
    test_metrics: Dict[str, Dict[str, float]]
    score_dumps: Dict[str, Dict[str, np.ndarray]]
    curves: Dict[str, Dict[str, np.ndarray]]
    bootstrap: Optional[Dict[str, Any]] = None


def _as_1d(values: Sequence[Any], name: str, n: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != int(n):
        raise ValueError("%s must be a one-dimensional array of length N" % name)
    return array


def validate_feature_table(
    features: np.ndarray,
    labels: Sequence[int],
    user_ids: Sequence[Any],
    pools: Sequence[str],
    actions: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize the schema-independent detector input arrays."""

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("features must have non-empty shape [N, D]")
    if not np.all(np.isfinite(x)):
        raise ValueError("features must contain only finite values")
    n = x.shape[0]
    y = _as_1d(labels, "labels", n).astype(np.int64)
    users = _as_1d(user_ids, "user_ids", n)
    if users.dtype.kind == "O":
        raise ValueError(
            "user_ids must use one homogeneous numeric or string dtype, not object"
        )
    pool = _as_1d(pools, "pools", n).astype(str)
    action = _as_1d(actions, "actions", n).astype(str)
    if not set(np.unique(y).tolist()).issubset({REAL_LABEL, FAKE_LABEL}):
        raise ValueError("labels must use real=0 and fake=1 only")
    unknown_pools = set(np.unique(pool).tolist()) - set(ALLOWED_POOLS)
    if unknown_pools:
        raise ValueError("unknown pools: %s" % sorted(unknown_pools))
    if np.any(action == ""):
        raise ValueError("action names must be non-empty")
    return x, y, users, pool, action


def _require_both_classes(labels: np.ndarray, context: str) -> None:
    if set(np.unique(labels).astype(int).tolist()) != {REAL_LABEL, FAKE_LABEL}:
        raise ValueError("%s must contain both real=0 and fake=1" % context)


class FeaturePAD:
    """StandardScaler plus a fake-high binary feature detector."""

    def __init__(
        self,
        detector_kind: str,
        random_state: int = 20260713,
        model_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if detector_kind not in ALLOWED_DETECTORS:
            raise ValueError(
                "detector_kind must be one of %s" % (ALLOWED_DETECTORS,)
            )
        self.detector_kind = str(detector_kind)
        self.random_state = int(random_state)
        self.model_params = dict(model_params or {})
        self.scaler = StandardScaler()
        self.model = None
        self.train_row_count = 0
        self._fitted = False

    def _make_model(self):
        if self.detector_kind in ("linear_svm", "rbf_svm"):
            defaults: Dict[str, Any] = {
                "kernel": "linear" if self.detector_kind == "linear_svm" else "rbf",
                "C": 1.0,
                "gamma": "scale",
                "class_weight": "balanced",
                "probability": False,
                "random_state": self.random_state,
            }
            defaults.update(self.model_params)
            return SVC(**defaults)

        if XGBClassifier is None:
            detail = repr(_XGB_IMPORT_ERROR) if _XGB_IMPORT_ERROR is not None else "unknown import error"
            raise RuntimeError(
                "xgboost detector requested, but xgboost is unavailable: %s" % detail
            )
        defaults = {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": self.random_state,
            "verbosity": 0,
            "use_label_encoder": False,
        }
        defaults.update(self.model_params)
        return XGBClassifier(**defaults)

    def fit(self, train_features: np.ndarray, train_labels: Sequence[int]) -> "FeaturePAD":
        x = np.asarray(train_features, dtype=np.float64)
        y = np.asarray(train_labels, dtype=np.int64)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) == 0:
            raise ValueError("train features/labels must have shapes [N,D] and [N]")
        if not np.all(np.isfinite(x)):
            raise ValueError("train features must be finite")
        _require_both_classes(y, "train")

        # This is intentionally two explicit calls rather than a sklearn
        # Pipeline so tests and saved protocol metadata can prove train-only fit.
        scaled = self.scaler.fit_transform(x)
        self.model = self._make_model()
        self.model.fit(scaled, y)
        classes = np.asarray(getattr(self.model, "classes_", []), dtype=np.int64)
        if classes.shape != (2,) or classes.tolist() != [REAL_LABEL, FAKE_LABEL]:
            raise RuntimeError("detector class order is not [real=0, fake=1]")
        self.train_row_count = int(len(x))
        self._fitted = True
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted or self.model is None:
            raise RuntimeError("detector must be fitted before scoring")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != int(self.scaler.n_features_in_):
            raise ValueError("score features must have shape [N, fitted_D]")
        if not np.all(np.isfinite(x)):
            raise ValueError("score features must be finite")
        scaled = self.scaler.transform(x)
        if self.detector_kind in ("linear_svm", "rbf_svm"):
            score = np.asarray(self.model.decision_function(scaled), dtype=np.float64)
        else:
            classes = np.asarray(self.model.classes_, dtype=np.int64)
            fake_column = int(np.flatnonzero(classes == FAKE_LABEL)[0])
            score = np.asarray(
                self.model.predict_proba(scaled)[:, fake_column], dtype=np.float64
            )
        score = score.reshape(-1)
        if len(score) != len(x) or not np.all(np.isfinite(score)):
            raise RuntimeError("detector produced invalid scores")
        return score


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(scores, dtype=np.float64))
    if len(unique) == 0 or not np.all(np.isfinite(unique)):
        raise ValueError("scores must be a non-empty finite array")
    if len(unique) == 1:
        return np.asarray([unique[0], np.nextafter(unique[0], np.inf)])
    midpoint = unique[:-1] + 0.5 * (unique[1:] - unique[:-1])
    return np.concatenate(
        (unique[:1], midpoint, np.asarray([np.nextafter(unique[-1], np.inf)]))
    )


def operating_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> Dict[str, float]:
    """Evaluate one fixed threshold with strict score<threshold acceptance."""

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
        raise ValueError("labels and scores must be non-empty matching vectors")
    if not np.all(np.isfinite(s)) or not np.isfinite(float(threshold)):
        raise ValueError("scores and threshold must be finite")
    _require_both_classes(y, "metric input")
    real = y == REAL_LABEL
    fake = y == FAKE_LABEL
    # Equality is rejected.  Keep these comparisons centralized so threshold
    # selection, test evaluation, and bootstrap cannot drift semantically.
    frr = float(np.mean(s[real] >= float(threshold)))
    fa = float(np.mean(s[fake] < float(threshold)))
    auc = float(roc_auc_score(y, s))
    return {
        "threshold": float(threshold),
        "fa": fa,
        "frr": frr,
        "auc": auc,
        "n_real": int(np.sum(real)),
        "n_fake": int(np.sum(fake)),
    }


def fa_frr_curve(
    labels: Sequence[int], scores: Sequence[float]
) -> Dict[str, np.ndarray]:
    """Return every attainable FA/FRR state for a fake-high score."""

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
        raise ValueError("labels and scores must be non-empty matching vectors")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    _require_both_classes(y, "curve input")
    thresholds = _threshold_candidates(s)
    real_scores = s[y == REAL_LABEL]
    fake_scores = s[y == FAKE_LABEL]
    frr = np.asarray(
        [np.mean(real_scores >= threshold) for threshold in thresholds],
        dtype=np.float64,
    )
    fa = np.asarray(
        [np.mean(fake_scores < threshold) for threshold in thresholds],
        dtype=np.float64,
    )
    return {"threshold": thresholds, "fa": fa, "frr": frr}


def select_validation_thresholds(
    labels: Sequence[int],
    scores: Sequence[float],
    target_frr: float = 0.05,
) -> Dict[str, float]:
    """Select EER and minimum-FA validation thresholds without test access."""

    if not np.isfinite(float(target_frr)) or not 0.0 <= target_frr <= 1.0:
        raise ValueError("target_frr must lie in [0,1]")
    curve = fa_frr_curve(labels, scores)
    threshold = curve["threshold"]
    fa = curve["fa"]
    frr = curve["frr"]

    # Primary: |FA-FRR|. Secondary: lower worst error. Tertiary: lower threshold.
    eer_order = np.lexsort((threshold, np.maximum(fa, frr), np.abs(fa - frr)))
    eer_index = int(eer_order[0])

    eligible = np.flatnonzero(frr <= float(target_frr) + 1.0e-15)
    if len(eligible) == 0:  # The curve includes an accept-all state, so defensive only.
        raise RuntimeError("no validation threshold satisfies the target FRR")
    # Among feasible states, minimize FA; on ties stay closest to the allowed
    # FRR boundary, then use the smaller threshold.
    feasible_order = np.lexsort(
        (threshold[eligible], -frr[eligible], fa[eligible])
    )
    frr_index = int(eligible[int(feasible_order[0])])
    return {
        "eer": float(threshold[eer_index]),
        "val_frr_le_5pct": float(threshold[frr_index]),
        "target_frr": float(target_frr),
    }


def resample_user_groups(
    labels: Sequence[int],
    user_ids: Sequence[Any],
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Sample real and fake users independently, retaining every user window.

    A user drawn twice contributes its complete class-specific window group
    twice.  Returned indices intentionally contain duplicates.
    """

    y = np.asarray(labels, dtype=np.int64)
    users = np.asarray(user_ids)
    if y.ndim != 1 or users.ndim != 1 or len(y) != len(users) or len(y) == 0:
        raise ValueError("labels and user_ids must be non-empty matching vectors")
    _require_both_classes(y, "bootstrap input")
    parts = []
    audit: Dict[str, np.ndarray] = {}
    for label, name in ((REAL_LABEL, "real"), (FAKE_LABEL, "fake")):
        class_mask = y == label
        class_users = np.unique(users[class_mask])
        if len(class_users) == 0:
            raise ValueError("bootstrap class has no users")
        draws = rng.choice(class_users, size=len(class_users), replace=True)
        audit[name + "_users"] = class_users.copy()
        audit[name + "_draws"] = np.asarray(draws).copy()
        for user in draws:
            group = np.flatnonzero(class_mask & (users == user))
            if len(group) == 0:
                raise RuntimeError("sampled user group is unexpectedly empty")
            parts.append(group)
    return np.concatenate(parts).astype(np.int64), audit


def _bootstrap_summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
    }


def user_level_bootstrap(
    labels: Sequence[int],
    scores: Sequence[float],
    user_ids: Sequence[Any],
    thresholds: Mapping[str, float],
    n_replicates: int = 500,
    seed: int = 20260713,
) -> Dict[str, Any]:
    """Bootstrap fixed-threshold test metrics with user as the sampling unit."""

    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    users = np.asarray(user_ids)
    if s.ndim != 1 or len(s) != len(y) or not np.all(np.isfinite(s)):
        raise ValueError("bootstrap scores must be a finite vector matching labels")
    if users.ndim != 1 or len(users) != len(y):
        raise ValueError("bootstrap user_ids must match labels")
    if int(n_replicates) <= 0:
        raise ValueError("n_replicates must be positive")
    if not thresholds:
        raise ValueError("at least one fixed threshold is required")
    fixed_thresholds = {str(k): float(v) for k, v in thresholds.items()}
    if not np.all(np.isfinite(list(fixed_thresholds.values()))):
        raise ValueError("bootstrap thresholds must be finite")
    _require_both_classes(y, "bootstrap input")

    rng = np.random.RandomState(int(seed))
    auc_values = np.zeros((int(n_replicates),), dtype=np.float64)
    fa_values = {
        name: np.zeros((int(n_replicates),), dtype=np.float64)
        for name in fixed_thresholds
    }
    frr_values = {
        name: np.zeros((int(n_replicates),), dtype=np.float64)
        for name in fixed_thresholds
    }
    for replicate in range(int(n_replicates)):
        indices, _ = resample_user_groups(y, users, rng)
        yy = y[indices]
        ss = s[indices]
        auc_values[replicate] = float(roc_auc_score(yy, ss))
        for name, threshold in fixed_thresholds.items():
            metric = operating_metrics(yy, ss, threshold)
            fa_values[name][replicate] = metric["fa"]
            frr_values[name][replicate] = metric["frr"]

    summary: Dict[str, Any] = {"auc": _bootstrap_summary(auc_values)}
    for name in fixed_thresholds:
        summary[name] = {
            "threshold": fixed_thresholds[name],
            "fa": _bootstrap_summary(fa_values[name]),
            "frr": _bootstrap_summary(frr_values[name]),
        }
    return {
        "protocol": "separate_real_fake_user_resampling_all_windows_fixed_val_thresholds",
        "n_replicates": int(n_replicates),
        "seed": int(seed),
        "n_real_users": int(len(np.unique(users[y == REAL_LABEL]))),
        "n_fake_users": int(len(np.unique(users[y == FAKE_LABEL]))),
        "thresholds": fixed_thresholds,
        "summary": summary,
        "replicates": {
            "auc": auc_values,
            **{"%s_fa" % name: values for name, values in fa_values.items()},
            **{"%s_frr" % name: values for name, values in frr_values.items()},
        },
    }


def run_feature_pad_protocol(
    features: np.ndarray,
    labels: Sequence[int],
    user_ids: Sequence[Any],
    pools: Sequence[str],
    actions: Sequence[str],
    action: str,
    detector_kind: str,
    random_state: int = 20260713,
    model_params: Optional[Mapping[str, Any]] = None,
    target_frr: float = 0.05,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 20260713,
) -> FeaturePADProtocolResult:
    """Run train -> validation threshold selection -> fixed-threshold test."""

    if not np.isclose(float(target_frr), 0.05, rtol=0.0, atol=1.0e-15):
        raise ValueError("the strict protocol fixes validation target_frr at 0.05")
    x, y, users, pool, action_array = validate_feature_table(
        features, labels, user_ids, pools, actions
    )
    requested_action = str(action)
    action_mask = action_array == requested_action
    if not np.any(action_mask):
        raise ValueError("requested action has no rows: %s" % requested_action)

    split_indices: Dict[str, np.ndarray] = {}
    for split in ALLOWED_POOLS:
        indices = np.flatnonzero(action_mask & (pool == split))
        if len(indices) == 0:
            raise ValueError("action %s has no %s rows" % (requested_action, split))
        _require_both_classes(y[indices], "%s/%s" % (requested_action, split))
        split_indices[split] = indices

    detector = FeaturePAD(
        detector_kind=detector_kind,
        random_state=random_state,
        model_params=model_params,
    )
    train_index = split_indices["train"]
    detector.fit(x[train_index], y[train_index])

    score_dumps: Dict[str, Dict[str, np.ndarray]] = {}
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("val", "test"):
        index = split_indices[split]
        score = detector.score(x[index])
        score_dumps[split] = {
            "score": score.astype(np.float64),
            "label": y[index].astype(np.int64),
            "user_id": users[index].copy(),
            "pool": pool[index].copy(),
            "action": action_array[index].copy(),
            "row_index": index.astype(np.int64),
        }
        curves[split] = fa_frr_curve(y[index], score)

    thresholds = select_validation_thresholds(
        score_dumps["val"]["label"],
        score_dumps["val"]["score"],
        target_frr=target_frr,
    )
    operating_thresholds = {
        "eer": thresholds["eer"],
        "val_frr_le_5pct": thresholds["val_frr_le_5pct"],
    }
    validation_metrics = {
        name: operating_metrics(
            score_dumps["val"]["label"], score_dumps["val"]["score"], threshold
        )
        for name, threshold in operating_thresholds.items()
    }
    test_metrics = {
        name: operating_metrics(
            score_dumps["test"]["label"], score_dumps["test"]["score"], threshold
        )
        for name, threshold in operating_thresholds.items()
    }

    bootstrap = None
    if int(bootstrap_replicates) > 0:
        bootstrap = user_level_bootstrap(
            score_dumps["test"]["label"],
            score_dumps["test"]["score"],
            score_dumps["test"]["user_id"],
            operating_thresholds,
            n_replicates=int(bootstrap_replicates),
            seed=int(bootstrap_seed),
        )
    return FeaturePADProtocolResult(
        action=requested_action,
        detector_kind=str(detector_kind),
        detector=detector,
        thresholds=thresholds,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        score_dumps=score_dumps,
        curves=curves,
        bootstrap=bootstrap,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("not JSON serializable: %r" % (type(value),))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def save_protocol_outputs(
    result: FeaturePADProtocolResult, output_dir: Path
) -> Dict[str, str]:
    """Atomically persist summary, score dump, curves, and bootstrap arrays."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "summary.json"
    score_path = root / "score_dump.npz"
    curve_path = root / "curves.npz"
    summary = {
        "action": result.action,
        "detector_kind": result.detector_kind,
        "score_direction": "fake_high",
        "acceptance_rule": "score < threshold",
        "scaler_fit_pool": "train_only",
        "threshold_selection_pool": "validation_only",
        "thresholds": result.thresholds,
        "validation_metrics": result.validation_metrics,
        "test_metrics": result.test_metrics,
        "train_row_count": result.detector.train_row_count,
        "random_state": result.detector.random_state,
        "model_params": result.detector.model_params,
        "scaler_mean": result.detector.scaler.mean_.astype(float).tolist(),
        "scaler_scale": result.detector.scaler.scale_.astype(float).tolist(),
    }
    _atomic_json(summary_path, summary)
    _atomic_npz(
        score_path,
        {
            "%s_%s" % (split, key): value
            for split, dump in result.score_dumps.items()
            for key, value in dump.items()
        },
    )
    _atomic_npz(
        curve_path,
        {
            "%s_%s" % (split, key): value
            for split, curve in result.curves.items()
            for key, value in curve.items()
        },
    )
    paths = {
        "summary": str(summary_path),
        "score_dump": str(score_path),
        "curves": str(curve_path),
    }
    if result.bootstrap is not None:
        bootstrap_summary_path = root / "bootstrap_summary.json"
        bootstrap_replicates_path = root / "bootstrap_replicates.npz"
        _atomic_json(
            bootstrap_summary_path,
            {key: value for key, value in result.bootstrap.items() if key != "replicates"},
        )
        _atomic_npz(bootstrap_replicates_path, result.bootstrap["replicates"])
        paths["bootstrap_summary"] = str(bootstrap_summary_path)
        paths["bootstrap_replicates"] = str(bootstrap_replicates_path)
    return paths
