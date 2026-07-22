"""Build formal fake consistency components from paired trajectory/IMU trees."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from generation.pad_export import load_generated_action_tree
from generation.protocol import FixedUserSplit
from runtime.paired_layer import cross_modal_consistency_from_record

from .fake_imu_pairs import validate_fake_imu_unit
from .paired_dataset_builder import COMPONENT_TABLE_SCHEMA, DetectorComponentTable


def build_fake_consistency_component(
    *,
    action: str,
    trajectory_archive_root: Path,
    fake_imu_root: Path,
    split_json: Path,
    output_path: Path,
    samples_per_user: int = 200,
    require_formal: bool = True,
) -> Dict[str, Any]:
    split = FixedUserSplit.load(str(split_json), require_formal=True)
    records, _ = load_generated_action_tree(
        Path(trajectory_archive_root), action, split, require_formal=require_formal
    )
    by_id = {str(record.sample_id): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("generated trajectory records contain duplicate sample ids")
    imu_paths = sorted((Path(fake_imu_root) / action).glob("user_*.npz"))
    expected_units = 100 if require_formal else len(imu_paths)
    if not imu_paths or len(imu_paths) != expected_units:
        raise ValueError("paired fake IMU unit count mismatch for %s" % action)

    sample_ids = []
    labels = []
    user_ids = []
    pools = []
    actions = []
    durations = []
    pair_identities = []
    matrix = []
    feature_names = None
    seen_trajectory_archives = set()
    for imu_path in imu_paths:
        audit_path = imu_path.with_suffix(".audit.json")
        if not audit_path.is_file():
            raise ValueError("paired fake IMU unit lacks audit: %s" % imu_path)
        # The exact trajectory source path is stored in the unit audit.
        import json
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        trajectory_path = Path(str(audit.get("trajectory_archive", "")))
        validate_fake_imu_unit(
            imu_path,
            trajectory_archive_path=trajectory_path,
            expected_action=action,
            expected_samples=samples_per_user,
        )
        seen_trajectory_archives.add(str(trajectory_path.resolve()))
        with np.load(str(imu_path), allow_pickle=False) as source:
            ids = np.asarray(source["sample_ids"]).astype(str)
            users = np.asarray(source["user_ids"], dtype=np.int64)
            unit_pools = np.asarray(source["pools"]).astype(str)
            unit_duration = np.asarray(source["duration_ms"], dtype=np.float64)
            identities = np.asarray(source["event_plan_sha256"]).astype(str)
            offsets = np.asarray(source["imu_offsets"], dtype=np.int64)
            active = np.asarray(source["flat_active_imu"], dtype=np.float32)
            relative = np.asarray(source["flat_relative_timestamps_ns"], dtype=np.int64)
        for index, sample_id in enumerate(ids.tolist()):
            if sample_id not in by_id:
                raise ValueError("fake IMU sample is absent from generated trajectory tree")
            record = by_id.pop(sample_id)
            if (
                int(record.user_id) != int(users[index])
                or str(record.pool) != str(unit_pools[index])
                or str(record.action) != action
            ):
                raise ValueError("fake paired record metadata mismatch")
            left, right = int(offsets[index]), int(offsets[index + 1])
            names, values = cross_modal_consistency_from_record(
                action=action,
                logical_duration_ms=float(unit_duration[index]),
                active_imu=active[left:right],
                imu_relative_timestamps_ms=relative[left:right].astype(np.float64) / 1.0e6,
                trajectory_record=record,
            )
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise RuntimeError("fake consistency feature order drifted")
            matrix.append(values)
            sample_ids.append(sample_id)
            labels.append(1)
            user_ids.append(int(users[index]))
            pools.append(str(unit_pools[index]))
            actions.append(action)
            durations.append(float(unit_duration[index]))
            pair_identities.append(str(identities[index]))
    if by_id:
        raise ValueError("generated trajectory tree has samples missing from paired fake IMU")
    expected_rows = expected_units * int(samples_per_user)
    if len(matrix) != expected_rows:
        raise ValueError("fake consistency row count mismatch")
    features = np.stack(matrix, axis=0).astype(np.float64)
    if not np.all(np.isfinite(features)):
        raise ValueError("fake consistency features contain non-finite values")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        str(temporary),
        schema_version=np.asarray(COMPONENT_TABLE_SCHEMA),
        component=np.asarray("consistency"),
        features=features,
        feature_names=np.asarray(feature_names),
        sample_ids=np.asarray(sample_ids),
        labels=np.asarray(labels, dtype=np.int64),
        user_ids=np.asarray(user_ids, dtype=np.int64),
        pools=np.asarray(pools),
        actions=np.asarray(actions),
        duration_ms=np.asarray(durations, dtype=np.float64),
        pair_identity_sha256=np.asarray(pair_identities),
    )
    temporary.replace(output)
    table = DetectorComponentTable.load(output, "consistency")
    return {
        "schema_version": "fake_consistency_component_audit_v1",
        "status": "pass",
        "action": action,
        "rows": int(table.features.shape[0]),
        "features": int(table.features.shape[1]),
        "feature_names": list(table.feature_names),
        "trajectory_archive_root": str(Path(trajectory_archive_root).resolve()),
        "fake_imu_root": str(Path(fake_imu_root).resolve()),
        "trajectory_units": len(seen_trajectory_archives),
        "imu_units": len(imu_paths),
        "pair_identity": "event_plan_sha256_exact",
        "output": str(output.resolve()),
    }


__all__ = ["build_fake_consistency_component"]
