"""Unified trajectory estimator service.

The service loads a manifest that points to feature PAD artifacts and/or raw
Deep PAD summaries/checkpoints.  It scores one ``RawTrajectoryRecord`` with all
available detectors for that action and returns detector-level decisions plus
simple ensemble vote summaries.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

from .deep_estimator import DeepEstimatorArtifact
from .feature_estimator import FeatureEstimatorArtifact


MANIFEST_SCHEMA = "trajectory_estimator_manifest_v1"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class TrajectoryEstimatorService:
    """Runtime facade for trajectory human-likeness estimation."""

    manifest_path: Path
    feature_estimators: Mapping[str, Mapping[str, FeatureEstimatorArtifact]]
    deep_estimators: Mapping[str, Mapping[str, DeepEstimatorArtifact]]

    @classmethod
    def load(
        cls,
        manifest_path: Path,
        *,
        load_feature: bool = True,
        load_deep: bool = True,
        device: Optional[str] = None,
    ) -> "TrajectoryEstimatorService":
        manifest_path = Path(manifest_path)
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != MANIFEST_SCHEMA:
            raise ValueError("estimator manifest schema mismatch: %s" % manifest_path)
        root = manifest_path.parent
        feature: Dict[str, Dict[str, FeatureEstimatorArtifact]] = {}
        deep: Dict[str, Dict[str, DeepEstimatorArtifact]] = {}
        artifacts = manifest.get("artifacts", {})

        for action, groups in artifacts.items():
            action = str(action)
            if load_feature:
                for detector, spec in dict(groups.get("feature_pad", {})).items():
                    path = Path(spec["artifact"])
                    if not path.is_absolute():
                        path = (root / path).resolve()
                    feature.setdefault(action, {})[str(detector)] = FeatureEstimatorArtifact.load(path)
            if load_deep:
                for detector, spec in dict(groups.get("deep_pad", {})).items():
                    summary = Path(spec["summary"])
                    if not summary.is_absolute():
                        summary = (root / summary).resolve()
                    checkpoint = spec.get("checkpoint")
                    checkpoint_path = None
                    if checkpoint:
                        checkpoint_path = Path(checkpoint)
                        if not checkpoint_path.is_absolute():
                            checkpoint_path = (root / checkpoint_path).resolve()
                    deep.setdefault(action, {})[str(detector)] = DeepEstimatorArtifact.load(
                        summary, checkpoint_path=checkpoint_path, device=device
                    )
        return cls(manifest_path=manifest_path, feature_estimators=feature, deep_estimators=deep)

    def actions(self) -> List[str]:
        return sorted(set(self.feature_estimators) | set(self.deep_estimators))

    def estimate_record(self, record: Any) -> Dict[str, Any]:
        started = time.perf_counter()
        action = str(record.action)
        detector_results: List[Dict[str, Any]] = []
        for artifact in self.feature_estimators.get(action, {}).values():
            detector_results.append(artifact.estimate_record(record))
        for artifact in self.deep_estimators.get(action, {}).values():
            detector_results.append(artifact.estimate_record(record))
        if not detector_results:
            raise ValueError("no estimator artifacts loaded for action %s" % action)

        summary: Dict[str, Any] = {}
        for point in ("eer", "val_frr_le_5pct"):
            votes = [
                bool(row["decisions"][point]["rejected_as_fake"])
                for row in detector_results
                if point in row.get("decisions", {})
            ]
            if votes:
                summary[point] = {
                    "n_detectors": int(len(votes)),
                    "fake_vote_count": int(np.sum(votes)),
                    "fake_vote_rate": float(np.mean(votes)),
                    "accepted_by_all": bool(not any(votes)),
                    "accepted_by_majority": bool(np.mean(votes) < 0.5),
                }
        return {
            "schema_version": "trajectory_estimator_result_v1",
            "action": action,
            "sample_id": str(getattr(record, "sample_id", "")),
            "score_direction": "fake_high",
            "acceptance_rule": "score < threshold",
            "detectors": detector_results,
            "ensemble": summary,
            "latency_ms": float((time.perf_counter() - started) * 1000.0),
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA,
            "manifest_path": str(self.manifest_path),
            "actions": self.actions(),
            "feature_estimators": {
                action: {detector: artifact.metadata() for detector, artifact in detectors.items()}
                for action, detectors in self.feature_estimators.items()
            },
            "deep_estimators": {
                action: {detector: artifact.metadata() for detector, artifact in detectors.items()}
                for action, detectors in self.deep_estimators.items()
            },
        }

