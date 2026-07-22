"""Duration-stratified metrics with train-only bin fitting.

The detector threshold remains the validation-selected global threshold.  Time
bins are descriptive test slices only: they never select a model or threshold.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


DURATION_REPORT_SCHEMA = "duration_stratified_fixed_threshold_metrics_v1"


def fit_duration_bins(train_duration_ms: Sequence[float], n_bins: int = 4) -> Dict[str, Any]:
    values = np.asarray(train_duration_ms, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("train durations must be a non-empty finite positive vector")
    if int(n_bins) < 1:
        raise ValueError("n_bins must be positive")
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    cuts = np.unique(np.quantile(values, quantiles)).astype(np.float64)
    # Repeated durations can collapse adjacent quantile bins.  Keeping only
    # unique cuts makes every emitted bin attainable and deterministic.
    return {
        "fit_pool": "train_only",
        "requested_bins": int(n_bins),
        "effective_bins": int(cuts.size + 1),
        "cut_points_ms": cuts.tolist(),
        "train_min_ms": float(np.min(values)),
        "train_max_ms": float(np.max(values)),
        "train_count": int(values.size),
    }


def _bin_indices(duration_ms: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    cuts = np.asarray(spec.get("cut_points_ms", []), dtype=np.float64)
    if cuts.ndim != 1 or not np.all(np.isfinite(cuts)) or np.any(np.diff(cuts) <= 0):
        raise ValueError("duration bin cut points must be finite and strictly increasing")
    return np.searchsorted(cuts, duration_ms, side="right").astype(np.int64)


def duration_stratified_metrics(
    *,
    labels: Sequence[int],
    scores: Sequence[float],
    duration_ms: Sequence[float],
    thresholds: Mapping[str, float],
    bin_spec: Mapping[str, Any],
    pool: str,
) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    duration = np.asarray(duration_ms, dtype=np.float64).reshape(-1)
    if not (y.size == score.size == duration.size) or y.size == 0:
        raise ValueError("labels, scores and durations must be matching non-empty vectors")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("labels must use real=0/fake=1")
    if not np.all(np.isfinite(score)) or not np.all(np.isfinite(duration)) or np.any(duration <= 0):
        raise ValueError("scores/durations must be finite and durations positive")
    fixed = {str(name): float(value) for name, value in thresholds.items()}
    if not fixed or not np.all(np.isfinite(list(fixed.values()))):
        raise ValueError("thresholds must be a non-empty finite mapping")
    indices = _bin_indices(duration, bin_spec)
    n_bins = int(bin_spec["effective_bins"])
    if n_bins != len(np.asarray(bin_spec.get("cut_points_ms", []))) + 1:
        raise ValueError("duration bin spec effective count mismatch")
    rows = []
    for bin_index in range(n_bins):
        selected = indices == bin_index
        yy = y[selected]
        ss = score[selected]
        dd = duration[selected]
        real = yy == 0
        fake = yy == 1
        operating = {}
        for name, threshold in fixed.items():
            operating[name] = {
                "threshold": threshold,
                "fa": None if not np.any(fake) else float(np.mean(ss[fake] < threshold)),
                "frr": None if not np.any(real) else float(np.mean(ss[real] >= threshold)),
            }
        auc = None
        if np.any(real) and np.any(fake):
            auc = float(roc_auc_score(yy, ss))
        rows.append({
            "bin_index": int(bin_index),
            "lower_cut_ms": None if bin_index == 0 else float(bin_spec["cut_points_ms"][bin_index - 1]),
            "upper_cut_ms": None if bin_index == n_bins - 1 else float(bin_spec["cut_points_ms"][bin_index]),
            "interval": "[lower,upper)" if bin_index < n_bins - 1 else "[lower,+inf)",
            "n": int(selected.sum()),
            "n_real": int(real.sum()),
            "n_fake": int(fake.sum()),
            "observed_min_ms": None if dd.size == 0 else float(np.min(dd)),
            "observed_max_ms": None if dd.size == 0 else float(np.max(dd)),
            "auc": auc,
            "operating_points": operating,
        })
    return {
        "schema_version": DURATION_REPORT_SCHEMA,
        "pool": str(pool),
        "threshold_source": "validation_global_not_refit_per_bin",
        "bin_spec": dict(bin_spec),
        "rows": rows,
    }


__all__ = [
    "DURATION_REPORT_SCHEMA",
    "fit_duration_bins",
    "duration_stratified_metrics",
]
