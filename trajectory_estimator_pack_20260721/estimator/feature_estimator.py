"""Feature-level trajectory estimator artifact and runtime scoring.

This wraps the clean-room trajectory feature PAD protocol from
``trajectory_humanization_full_20260713``.  The persisted artifact contains the
trained sklearn/xgboost model, the train-only scaler, validation-selected
thresholds, and metadata required to prove the score direction.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import joblib
import numpy as np


TRAJECTORY_PROJECT = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713"
)
if str(TRAJECTORY_PROJECT) not in sys.path:
    sys.path.insert(0, str(TRAJECTORY_PROJECT))

from detectors.deep_pad import FAKE_LABEL, REAL_LABEL, RawTrajectoryRecord  # noqa: E402
from trajectory.features import (  # noqa: E402
    TRAJECTORY_FEATURE_SCHEMA_VERSION,
    canonical_keycode_feature_token,
    extract_keystroke_features,
    extract_pinch_features,
    extract_single_finger_features,
)


ARTIFACT_SCHEMA = "trajectory_feature_estimator_artifact_v1"
ACCEPTANCE_RULE = "score < threshold"
SCORE_DIRECTION = "fake_high"


def _finite_vector(values: np.ndarray, *, name: str = "feature_vector") -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("%s must be a non-empty 1-D vector" % name)
    if not np.all(np.isfinite(vector)):
        raise ValueError("%s contains non-finite values" % name)
    return vector


def _decision(score: float, threshold: float) -> Dict[str, Any]:
    if not (math.isfinite(float(score)) and math.isfinite(float(threshold))):
        raise ValueError("score and threshold must be finite")
    accepted = bool(float(score) < float(threshold))
    return {
        "threshold": float(threshold),
        "accepted_as_real": accepted,
        "rejected_as_fake": not accepted,
        "margin_score_minus_threshold": float(score) - float(threshold),
        "acceptance_rule": ACCEPTANCE_RULE,
    }


def feature_vector_from_record(record: RawTrajectoryRecord) -> np.ndarray:
    """Extract the exact frozen feature vector used by the benchmark.

    The function consumes a ``RawTrajectoryRecord`` on the shared event-global
    timeline.  It does not resample timestamps and it uses the same single
    source of truth as the previous benchmark code.
    """

    record.validate()
    action = str(record.action)
    values = np.asarray(record.pointer_continuous, dtype=np.float32)
    times = np.asarray(record.global_t_ms, dtype=np.float64)
    contact = np.asarray(record.contact_mask, dtype=bool)

    if action in ("tap", "scroll", "swipe"):
        keep = contact[0]
        if not np.any(keep):
            raise ValueError("%s record has no pointer-0 contact" % action)
        return extract_single_finger_features(values[0, keep, :2], times[keep])

    if action == "pinch":
        keep = contact[0] & contact[1]
        if not np.any(keep):
            raise ValueError("pinch record has no simultaneous two-finger frame")
        return extract_pinch_features(values[0, keep, :2], values[1, keep, :2], times[keep])

    if action == "keystroke":
        event_ids = np.asarray(record.event_ids[0], dtype=np.int64)
        keycodes = np.asarray(record.keycode[0], dtype=np.int64)
        pointer_contact = contact[0]
        observed = sorted(set(int(v) for v in event_ids[pointer_contact].tolist() if int(v) >= 0))
        if not observed:
            raise ValueError("keystroke record has no key contact events")
        keys = []
        down = []
        up = []
        points = []
        for event_id in observed:
            local = np.flatnonzero(pointer_contact & (event_ids == event_id))
            if local.size == 0:
                raise RuntimeError("missing local rows for observed key event")
            local_times = times[local]
            local_keycodes = keycodes[local]
            valid_codes = local_keycodes[local_keycodes >= 0]
            if valid_codes.size == 0:
                raise ValueError("keystroke contact lacks canonical keycode")
            keys.append(canonical_keycode_feature_token(int(valid_codes[0])))
            down.append(float(local_times[0]))
            up.append(float(local_times[-1]))
            points.append(values[0, local[0], :2].astype(np.float64))
        return extract_keystroke_features(
            keys,
            np.asarray(down, dtype=np.float64),
            up_times_ms=np.asarray(up, dtype=np.float64),
            key_points=np.asarray(points, dtype=np.float64),
        )

    raise ValueError("unsupported action: %s" % action)


@dataclass(frozen=True)
class FeatureEstimatorArtifact:
    """Loaded runtime artifact for one action and one feature detector."""

    action: str
    detector_kind: str
    model: Any
    scaler: Any
    thresholds: Mapping[str, float]
    validation_metrics: Mapping[str, Mapping[str, float]]
    test_metrics: Mapping[str, Mapping[str, float]]
    train_row_count: int
    model_params: Mapping[str, Any]
    artifact_path: Path

    @classmethod
    def from_protocol_result(cls, result: Any, artifact_path: Path) -> "FeatureEstimatorArtifact":
        """Persist and return an artifact from ``run_feature_pad_protocol`` output."""

        detector = result.detector
        payload = {
            "schema_version": ARTIFACT_SCHEMA,
            "feature_schema_version": TRAJECTORY_FEATURE_SCHEMA_VERSION,
            "score_direction": SCORE_DIRECTION,
            "acceptance_rule": ACCEPTANCE_RULE,
            "action": result.action,
            "detector_kind": result.detector_kind,
            "model": detector.model,
            "scaler": detector.scaler,
            "thresholds": dict(result.thresholds),
            "validation_metrics": result.validation_metrics,
            "test_metrics": result.test_metrics,
            "train_row_count": int(detector.train_row_count),
            "model_params": dict(detector.model_params),
            "artifact_role": "runtime_feature_pad_estimator",
            "threshold_selection_pool": "validation_only",
            "normalization_fit_pool": "train_only",
        }
        artifact_path = Path(artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_path.with_name(artifact_path.name + ".tmp")
        joblib.dump(payload, temporary)
        temporary.replace(artifact_path)
        return cls.load(artifact_path)

    @classmethod
    def load(cls, artifact_path: Path) -> "FeatureEstimatorArtifact":
        payload = joblib.load(Path(artifact_path))
        if payload.get("schema_version") != ARTIFACT_SCHEMA:
            raise ValueError("feature estimator artifact schema mismatch: %s" % artifact_path)
        if payload.get("feature_schema_version") != TRAJECTORY_FEATURE_SCHEMA_VERSION:
            raise ValueError("feature schema mismatch in %s" % artifact_path)
        if payload.get("score_direction") != SCORE_DIRECTION:
            raise ValueError("score direction mismatch in %s" % artifact_path)
        if payload.get("acceptance_rule") != ACCEPTANCE_RULE:
            raise ValueError("acceptance rule mismatch in %s" % artifact_path)
        thresholds = {str(k): float(v) for k, v in dict(payload["thresholds"]).items()}
        if "eer" not in thresholds or "val_frr_le_5pct" not in thresholds:
            raise ValueError("artifact is missing required validation thresholds")
        return cls(
            action=str(payload["action"]),
            detector_kind=str(payload["detector_kind"]),
            model=payload["model"],
            scaler=payload["scaler"],
            thresholds=thresholds,
            validation_metrics=payload.get("validation_metrics", {}),
            test_metrics=payload.get("test_metrics", {}),
            train_row_count=int(payload.get("train_row_count", -1)),
            model_params=dict(payload.get("model_params", {})),
            artifact_path=Path(artifact_path),
        )

    def score_vector(self, feature_vector: Sequence[float]) -> float:
        vector = _finite_vector(np.asarray(feature_vector, dtype=np.float64))
        expected = int(getattr(self.scaler, "n_features_in_", vector.size))
        if vector.size != expected:
            raise ValueError(
                "%s/%s expects %d features, got %d"
                % (self.action, self.detector_kind, expected, vector.size)
            )
        scaled = self.scaler.transform(vector.reshape(1, -1))
        if self.detector_kind in ("linear_svm", "rbf_svm"):
            score = np.asarray(self.model.decision_function(scaled), dtype=np.float64).reshape(-1)[0]
        else:
            classes = np.asarray(getattr(self.model, "classes_", []), dtype=np.int64)
            if FAKE_LABEL not in classes:
                raise RuntimeError("xgboost/sklearn artifact lacks fake class")
            fake_col = int(np.flatnonzero(classes == FAKE_LABEL)[0])
            score = np.asarray(self.model.predict_proba(scaled)[:, fake_col], dtype=np.float64).reshape(-1)[0]
        if not math.isfinite(float(score)):
            raise RuntimeError("detector produced a non-finite score")
        return float(score)

    def estimate_vector(self, feature_vector: Sequence[float]) -> Dict[str, Any]:
        started = time.perf_counter()
        score = self.score_vector(feature_vector)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "family": "feature_pad",
            "action": self.action,
            "detector": self.detector_kind,
            "score": score,
            "score_direction": SCORE_DIRECTION,
            "decisions": {name: _decision(score, value) for name, value in self.thresholds.items() if name in ("eer", "val_frr_le_5pct")},
            "feature_schema_version": TRAJECTORY_FEATURE_SCHEMA_VERSION,
            "latency_ms": float(elapsed_ms),
        }

    def estimate_record(self, record: RawTrajectoryRecord) -> Dict[str, Any]:
        if str(record.action) != self.action:
            raise ValueError("record action %s does not match artifact action %s" % (record.action, self.action))
        vector = feature_vector_from_record(record)
        result = self.estimate_vector(vector)
        result["feature_dim"] = int(vector.size)
        return result

    def metadata(self) -> Dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA,
            "action": self.action,
            "detector_kind": self.detector_kind,
            "feature_schema_version": TRAJECTORY_FEATURE_SCHEMA_VERSION,
            "score_direction": SCORE_DIRECTION,
            "acceptance_rule": ACCEPTANCE_RULE,
            "thresholds": dict(self.thresholds),
            "train_row_count": int(self.train_row_count),
            "model_params": dict(self.model_params),
            "artifact_path": str(self.artifact_path),
        }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)

