#!/usr/bin/env python3
"""Train/export runtime feature estimator artifacts under the strict benchmark protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


PACK_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PROJECT = Path(
    "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713"
)
for path in (PACK_ROOT, TRAJECTORY_PROJECT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from detectors.benchmark_runner import load_benchmark_dataset  # noqa: E402
from detectors.deep_pad import ACTIONS, assign_strict_protocol_pools, load_fake_user_split  # noqa: E402
from detectors.feature_pad import ALLOWED_DETECTORS, run_feature_pad_protocol, save_protocol_outputs  # noqa: E402
from scripts.run_trajectory_benchmark import synthetic_five_action_dataset  # noqa: E402
from estimator.feature_estimator import (  # noqa: E402
    FeatureEstimatorArtifact,
    feature_vector_from_record,
    write_json,
)
from estimator.service import MANIFEST_SCHEMA  # noqa: E402


def _feature_arrays(records, feature_vectors: np.ndarray, action: str):
    action_records = [record for record in records if record.action == action]
    features = np.asarray(feature_vectors, dtype=np.float64)
    if len(features) != len(action_records):
        raise ValueError("feature row count does not match %s records" % action)
    return (
        features,
        np.asarray([record.label for record in action_records], dtype=np.int64),
        np.asarray([record.user_id for record in action_records], dtype=np.int64),
        np.asarray([record.pool for record in action_records], dtype="U5"),
        np.asarray([record.action for record in action_records], dtype="U16"),
    )


def _parse_list(values: Sequence[str] | None, allowed: Sequence[str], name: str) -> List[str]:
    if not values:
        return list(allowed)
    result: List[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                result.append(item)
    unknown = sorted(set(result) - set(allowed))
    if unknown:
        raise ValueError("unknown %s: %s" % (name, unknown))
    return result


def export_feature_artifacts(
    *,
    output_dir: Path,
    synthetic_smoke: bool,
    dataset_dir: Path | None,
    fake_user_split: Path | None,
    actions: Sequence[str],
    detectors: Sequence[str],
    bootstrap_replicates: int,
    seed: int,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if synthetic_smoke:
        records, _ = synthetic_five_action_dataset(seed=13)
        features = {
            action: np.stack(
                [feature_vector_from_record(record) for record in records if record.action == action],
                axis=0,
            )
            for action in actions
        }
        split_audit = {"mode": "synthetic_smoke_only", "formal_result": False}
    else:
        if dataset_dir is None or fake_user_split is None:
            raise ValueError("formal export requires --dataset-dir and --fake-user-split")
        records, features = load_benchmark_dataset(Path(dataset_dir), actions)
        split = load_fake_user_split(Path(fake_user_split))
        records, split_audit = assign_strict_protocol_pools(records, split)

    artifacts: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for action in actions:
        x, y, users, pools, action_array = _feature_arrays(records, features[action], action)
        for detector in detectors:
            model_params = {"n_estimators": 8, "max_depth": 2} if synthetic_smoke and detector == "xgboost" else None
            result = run_feature_pad_protocol(
                x,
                y,
                users,
                pools,
                action_array,
                action=action,
                detector_kind=detector,
                random_state=int(seed),
                model_params=model_params,
                bootstrap_replicates=int(bootstrap_replicates),
                bootstrap_seed=int(seed) + 31,
            )
            result_dir = output_dir / "feature_pad" / action / detector
            save_protocol_outputs(result, result_dir)
            artifact_path = result_dir / "artifact.joblib"
            artifact = FeatureEstimatorArtifact.from_protocol_result(result, artifact_path)
            artifacts.setdefault(action, {}).setdefault("feature_pad", {})[detector] = {
                "artifact": str(artifact_path),
                "summary": str(result_dir / "summary.json"),
                "score_dump": str(result_dir / "score_dump.npz"),
                "curves": str(result_dir / "curves.npz"),
            }
            for point in ("eer", "val_frr_le_5pct"):
                rows.append({
                    "action": action,
                    "family": "feature_pad",
                    "detector": detector,
                    "point": point,
                    "threshold": float(result.thresholds[point]),
                    "validation_fa": float(result.validation_metrics[point]["fa"]),
                    "validation_frr": float(result.validation_metrics[point]["frr"]),
                    "test_fa": float(result.test_metrics[point]["fa"]),
                    "test_frr": float(result.test_metrics[point]["frr"]),
                    "test_auc": float(result.test_metrics[point]["auc"]),
                })

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "pack_root": str(PACK_ROOT),
        "source_trajectory_project": str(TRAJECTORY_PROJECT),
        "mode": "synthetic_smoke" if synthetic_smoke else "formal_feature_export",
        "score_direction": "fake_high",
        "acceptance_rule": "score < threshold",
        "threshold_selection_pool": "validation_only",
        "normalization_fit_pool": "train_only",
        "actions": list(actions),
        "feature_detectors": list(detectors),
        "artifacts": artifacts,
        "split_audit": split_audit,
        "summary_rows": rows,
    }
    write_json(output_dir / "estimator_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--fake-user-split", type=Path, default=None)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--actions", nargs="*", default=None)
    parser.add_argument("--feature-detectors", nargs="*", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions = _parse_list(args.actions, ACTIONS, "actions")
    detectors = _parse_list(args.feature_detectors, ALLOWED_DETECTORS, "feature detectors")
    manifest = export_feature_artifacts(
        output_dir=args.output_dir,
        synthetic_smoke=bool(args.synthetic_smoke),
        dataset_dir=args.dataset_dir,
        fake_user_split=args.fake_user_split,
        actions=actions,
        detectors=detectors,
        bootstrap_replicates=int(args.bootstrap_replicates),
        seed=int(args.seed),
    )
    print(json.dumps({
        "manifest": str(Path(args.output_dir) / "estimator_manifest.json"),
        "mode": manifest["mode"],
        "actions": manifest["actions"],
        "feature_detectors": manifest["feature_detectors"],
        "n_summary_rows": len(manifest["summary_rows"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
