"""Build a runtime trajectory-estimator release from the formal 25-detector run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from detectors.deep_pad import load_raw_sequence_bundle
from detectors.feature_pad import run_feature_pad_protocol
from detectors.pair_runner import sha256_file, stable_pair_seed

from .feature_estimator import FeatureEstimatorArtifact
from .service import MANIFEST_SCHEMA, TrajectoryEstimatorService


TRAJECTORY_RELEASE_PROTOCOL = "formal_trajectory_estimator_release_v1"
ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
FEATURE_DETECTORS = ("linear_svm", "rbf_svm", "xgboost")
DEEP_DETECTORS = ("tcn", "transformer")


def _read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object: %s" % path)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _feature_arrays(records: Sequence[Any], features: np.ndarray) -> Tuple[np.ndarray, ...]:
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray([record.label for record in records], dtype=np.int64),
        np.asarray([record.user_id for record in records], dtype=np.int64),
        np.asarray([record.pool for record in records]),
        np.asarray([record.action for record in records]),
    )


def _checkpoint_path(summary_path: Path, value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = summary_path.parent / path
    return path.resolve()


def _compare_feature_result(result: Any, result_root: Path) -> Dict[str, str]:
    summary_path = result_root / "summary.json"
    score_path = result_root / "score_dump.npz"
    summary = _read_json(summary_path)
    expected_summary = {
        "action": result.action,
        "detector_kind": result.detector_kind,
        "score_direction": "fake_high",
        "acceptance_rule": "score < threshold",
        "threshold_selection_pool": "validation_only",
    }
    if any(summary.get(name) != value for name, value in expected_summary.items()):
        raise ValueError("formal feature result summary protocol mismatch")
    for name in ("thresholds", "validation_metrics", "test_metrics"):
        if summary.get(name) != getattr(result, name):
            raise ValueError("reconstructed feature model differs from formal %s" % name)
    with np.load(str(score_path), allow_pickle=False) as source:
        expected_keys = {
            "%s_%s" % (pool, name)
            for pool in ("val", "test")
            for name in ("score", "label", "user_id", "pool", "action", "row_index")
        }
        if set(source.files) != expected_keys:
            raise ValueError("formal feature score dump schema mismatch")
        for pool in ("val", "test"):
            for name, value in result.score_dumps[pool].items():
                if not np.array_equal(np.asarray(source[pool + "_" + name]), np.asarray(value)):
                    raise ValueError("reconstructed feature model score identity mismatch")
    return {
        "formal_summary": str(summary_path.resolve()),
        "formal_summary_sha256": sha256_file(summary_path),
        "formal_score_dump": str(score_path.resolve()),
        "formal_score_dump_sha256": sha256_file(score_path),
    }


def build_trajectory_estimator_release(
    *, bundle_dir: Path, detector_root: Path, output_dir: Path,
    actions: Sequence[str] = ACTIONS, base_seed: int = 20260713,
    require_formal_five_actions: bool = True,
) -> Dict[str, Any]:
    """Reconstruct feature models exactly and bind formal Deep PAD checkpoints."""

    actions = tuple(str(action) for action in actions)
    if len(set(actions)) != len(actions) or set(actions) - set(ACTIONS):
        raise ValueError("trajectory release actions are invalid")
    if require_formal_five_actions and set(actions) != set(ACTIONS):
        raise ValueError("formal trajectory release requires exactly five actions")
    bundle_dir = Path(bundle_dir).resolve()
    detector_root = Path(detector_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Any] = {}
    sources: Dict[str, Any] = {}
    for action in actions:
        bundle_path = bundle_dir / (action + ".npz")
        records, features = load_raw_sequence_bundle(bundle_path)
        if set(record.action for record in records) != {action}:
            raise ValueError("trajectory release bundle action mismatch")
        x, y, users, pools, action_array = _feature_arrays(records, features)
        action_artifacts: Dict[str, Any] = {"feature_pad": {}, "deep_pad": {}}
        action_sources: Dict[str, Any] = {
            "bundle": str(bundle_path), "bundle_sha256": sha256_file(bundle_path),
        }
        for detector in FEATURE_DETECTORS:
            pair_seed = stable_pair_seed(int(base_seed), action, "feature_pad", detector)
            result = run_feature_pad_protocol(
                x, y, users, pools, action_array,
                action=action, detector_kind=detector, random_state=pair_seed,
                bootstrap_replicates=0,
            )
            formal_root = detector_root / action / "feature_pad" / detector / "result"
            formal_sources = _compare_feature_result(result, formal_root)
            artifact_path = output_dir / "feature_pad" / action / detector / "artifact.joblib"
            FeatureEstimatorArtifact.from_protocol_result(result, artifact_path)
            action_artifacts["feature_pad"][detector] = {
                "artifact": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                **formal_sources,
            }
        for detector in DEEP_DETECTORS:
            formal_root = detector_root / action / "deep_pad" / detector / "result"
            summary_path = formal_root / "summary.json"
            summary = _read_json(summary_path)
            if (
                summary.get("schema_version") != "trajectory_deep_pad_result_v2"
                or summary.get("action") != action
                or summary.get("detector_kind") != detector
                or summary.get("score_direction") != "fake_high"
                or summary.get("acceptance_rule") != "score < threshold"
                or summary.get("checkpoint_selection_pool") != "validation_only"
                or summary.get("threshold_selection_pool") != "validation_only"
                or int(summary.get("last_epoch", -1)) != 40
            ):
                raise ValueError("formal deep estimator summary protocol mismatch")
            checkpoint = _checkpoint_path(
                summary_path, summary.get("checkpoint_paths", {}).get("best", ""),
            )
            try:
                checkpoint.relative_to(formal_root.resolve())
            except ValueError as exc:
                raise ValueError("formal deep best checkpoint escapes result root") from exc
            if not checkpoint.is_file():
                raise ValueError("formal deep best checkpoint is missing")
            action_artifacts["deep_pad"][detector] = {
                "summary": str(summary_path.resolve()),
                "summary_sha256": sha256_file(summary_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        artifacts[action] = action_artifacts
        sources[action] = action_sources
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "release_protocol": TRAJECTORY_RELEASE_PROTOCOL,
        "status": "complete",
        "formal_result": bool(require_formal_five_actions),
        "score_direction": "fake_high",
        "acceptance_rule": "score < threshold",
        "normalization_fit_pool": "train_only",
        "checkpoint_selection_pool": "validation_only",
        "threshold_selection_pool": "validation_only",
        "actions": list(actions),
        "detector_count": int(len(actions) * (len(FEATURE_DETECTORS) + len(DEEP_DETECTORS))),
        "base_seed": int(base_seed),
        "feature_model_policy": "exact_reconstruction_verified_against_formal_score_dump",
        "deep_model_policy": "formal_validation_selected_best_checkpoint_by_hash",
        "sources": sources,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "estimator_manifest.json"
    _atomic_json(manifest_path, manifest)
    validate_trajectory_estimator_release(
        manifest_path,
        expected_bundle_dir=bundle_dir,
        expected_detector_root=detector_root,
        expected_actions=actions,
        require_formal_five_actions=require_formal_five_actions,
    )
    return manifest


def validate_trajectory_estimator_release(
    manifest_path: Path, *, expected_bundle_dir: Path, expected_detector_root: Path,
    expected_actions: Sequence[str] = ACTIONS, require_formal_five_actions: bool = True,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    value = _read_json(manifest_path)
    actions = tuple(str(action) for action in expected_actions)
    expected_count = len(actions) * 5
    if (
        value.get("schema_version") != MANIFEST_SCHEMA
        or value.get("release_protocol") != TRAJECTORY_RELEASE_PROTOCOL
        or value.get("status") != "complete"
        or value.get("formal_result") is not bool(require_formal_five_actions)
        or value.get("score_direction") != "fake_high"
        or value.get("acceptance_rule") != "score < threshold"
        or value.get("normalization_fit_pool") != "train_only"
        or value.get("checkpoint_selection_pool") != "validation_only"
        or value.get("threshold_selection_pool") != "validation_only"
        or set(value.get("actions", ())) != set(actions)
        or int(value.get("detector_count", -1)) != expected_count
        or set(value.get("artifacts", {})) != set(actions)
        or set(value.get("sources", {})) != set(actions)
    ):
        raise ValueError("trajectory estimator release manifest protocol mismatch")
    bundle_dir = Path(expected_bundle_dir).resolve()
    detector_root = Path(expected_detector_root).resolve()
    for action in actions:
        bundle = bundle_dir / (action + ".npz")
        if (
            Path(value["sources"][action].get("bundle", "")).resolve() != bundle
            or value["sources"][action].get("bundle_sha256") != sha256_file(bundle)
        ):
            raise ValueError("trajectory estimator release bundle hash mismatch")
        groups = value["artifacts"][action]
        if set(groups.get("feature_pad", {})) != set(FEATURE_DETECTORS):
            raise ValueError("trajectory estimator release feature closure mismatch")
        if set(groups.get("deep_pad", {})) != set(DEEP_DETECTORS):
            raise ValueError("trajectory estimator release deep closure mismatch")
        for detector in FEATURE_DETECTORS:
            spec = groups["feature_pad"][detector]
            artifact_path = Path(spec.get("artifact", "")).resolve()
            formal_root = detector_root / action / "feature_pad" / detector / "result"
            if (
                spec.get("artifact_sha256") != sha256_file(artifact_path)
                or Path(spec.get("formal_summary", "")).resolve() != (formal_root / "summary.json").resolve()
                or spec.get("formal_summary_sha256") != sha256_file(formal_root / "summary.json")
                or Path(spec.get("formal_score_dump", "")).resolve() != (formal_root / "score_dump.npz").resolve()
                or spec.get("formal_score_dump_sha256") != sha256_file(formal_root / "score_dump.npz")
            ):
                raise ValueError("trajectory feature release source/artifact hash mismatch")
            artifact = FeatureEstimatorArtifact.load(artifact_path)
            if artifact.action != action or artifact.detector_kind != detector:
                raise ValueError("trajectory feature release artifact identity mismatch")
        for detector in DEEP_DETECTORS:
            spec = groups["deep_pad"][detector]
            formal_root = detector_root / action / "deep_pad" / detector / "result"
            summary = formal_root / "summary.json"
            summary_value = _read_json(summary)
            checkpoint = _checkpoint_path(summary, summary_value["checkpoint_paths"]["best"])
            if (
                Path(spec.get("summary", "")).resolve() != summary.resolve()
                or spec.get("summary_sha256") != sha256_file(summary)
                or Path(spec.get("checkpoint", "")).resolve() != checkpoint
                or spec.get("checkpoint_sha256") != sha256_file(checkpoint)
            ):
                raise ValueError("trajectory deep release source/checkpoint hash mismatch")
    service = TrajectoryEstimatorService.load(
        manifest_path, load_feature=True, load_deep=False,
    )
    if set(service.actions()) != set(actions):
        raise ValueError("trajectory estimator release service action closure mismatch")
    return value


__all__ = [
    "TRAJECTORY_RELEASE_PROTOCOL",
    "build_trajectory_estimator_release",
    "validate_trajectory_estimator_release",
]
