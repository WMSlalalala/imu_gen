"""Runtime wrapper for raw-sequence Deep PAD trajectory checkpoints."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


TRAJECTORY_PROJECT = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713"
)
if str(TRAJECTORY_PROJECT) not in sys.path:
    sys.path.insert(0, str(TRAJECTORY_PROJECT))

from detectors.deep_pad import (  # noqa: E402
    RawSequenceNormalizer,
    RawTrajectoryRecord,
    collate_raw_sequences,
    make_deep_model,
)


SCORE_DIRECTION = "fake_high"
ACCEPTANCE_RULE = "score < threshold"


def _decision(score: float, threshold: float) -> Dict[str, Any]:
    accepted = bool(float(score) < float(threshold))
    return {
        "threshold": float(threshold),
        "accepted_as_real": accepted,
        "rejected_as_fake": not accepted,
        "margin_score_minus_threshold": float(score) - float(threshold),
        "acceptance_rule": ACCEPTANCE_RULE,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class DeepEstimatorArtifact:
    """Loaded runtime artifact for one action and one raw-sequence detector."""

    action: str
    detector_kind: str
    model: torch.nn.Module
    normalizer: RawSequenceNormalizer
    thresholds: Mapping[str, float]
    summary_path: Path
    checkpoint_path: Path
    device: torch.device
    model_params: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        summary_path: Path,
        *,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> "DeepEstimatorArtifact":
        summary_path = Path(summary_path)
        summary = _read_json(summary_path)
        if summary.get("schema_version") != "trajectory_deep_pad_result_v2":
            raise ValueError("deep summary schema mismatch: %s" % summary_path)
        if summary.get("score_direction") != SCORE_DIRECTION:
            raise ValueError("deep summary score direction mismatch")
        if summary.get("acceptance_rule") != ACCEPTANCE_RULE:
            raise ValueError("deep summary acceptance rule mismatch")
        if summary.get("checkpoint_selection_pool") != "validation_only":
            raise ValueError("deep checkpoint must be selected on validation only")
        if summary.get("threshold_selection_pool") != "validation_only":
            raise ValueError("deep thresholds must be selected on validation only")

        paths = summary.get("checkpoint_paths", {})
        selected = Path(checkpoint_path or paths.get("best", ""))
        if not selected.is_absolute():
            selected = (summary_path.parent / selected).resolve()
        if not selected.exists():
            raise FileNotFoundError("deep best checkpoint not found: %s" % selected)

        selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(str(selected), map_location=selected_device)
        if checkpoint.get("schema_version") != "trajectory_deep_pad_v2":
            raise ValueError("deep checkpoint schema mismatch: %s" % selected)
        action = str(summary["action"])
        detector = str(summary["detector_kind"])
        if checkpoint.get("action") != action or checkpoint.get("detector_kind") != detector:
            raise ValueError("deep checkpoint action/detector does not match summary")

        model_params = dict(summary.get("model_params", {}))
        model = make_deep_model(detector, model_params)
        model.load_state_dict(checkpoint["model_state"])
        model.to(selected_device)
        model.eval()
        normalizer = RawSequenceNormalizer.from_state_dict(checkpoint["normalizer"])
        thresholds = {str(k): float(v) for k, v in summary["thresholds"].items()}
        if "eer" not in thresholds or "val_frr_le_5pct" not in thresholds:
            raise ValueError("deep summary is missing required validation thresholds")
        return cls(
            action=action,
            detector_kind=detector,
            model=model,
            normalizer=normalizer,
            thresholds=thresholds,
            summary_path=summary_path,
            checkpoint_path=selected,
            device=selected_device,
            model_params=model_params,
        )

    def score_records(self, records: Sequence[RawTrajectoryRecord]) -> np.ndarray:
        if not records:
            raise ValueError("records must be non-empty")
        for record in records:
            if str(record.action) != self.action:
                raise ValueError("record action %s does not match artifact action %s" % (record.action, self.action))
        batch = collate_raw_sequences(records, self.normalizer).to(self.device)
        started = time.perf_counter()
        with torch.no_grad():
            logits = self.model(batch)
            scores = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if scores.shape != (len(records),) or not np.all(np.isfinite(scores)):
            raise RuntimeError("deep estimator produced invalid scores")
        self._last_forward_ms = float(elapsed_ms)
        return scores

    def estimate_record(self, record: RawTrajectoryRecord) -> Dict[str, Any]:
        started = time.perf_counter()
        score = float(self.score_records([record])[0])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not math.isfinite(score):
            raise RuntimeError("deep score is non-finite")
        return {
            "family": "deep_pad",
            "action": self.action,
            "detector": self.detector_kind,
            "score": score,
            "score_direction": SCORE_DIRECTION,
            "decisions": {name: _decision(score, value) for name, value in self.thresholds.items() if name in ("eer", "val_frr_le_5pct")},
            "latency_ms": float(elapsed_ms),
            "device": str(self.device),
            "checkpoint_path": str(self.checkpoint_path),
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "detector_kind": self.detector_kind,
            "detector_family": "deep_pad",
            "score_direction": SCORE_DIRECTION,
            "acceptance_rule": ACCEPTANCE_RULE,
            "thresholds": dict(self.thresholds),
            "summary_path": str(self.summary_path),
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "model_params": dict(self.model_params),
        }

