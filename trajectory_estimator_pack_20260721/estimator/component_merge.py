"""Strictly concatenate disjoint real/fake detector component tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from .fake_imu_pairs import sha256_file
from .paired_dataset_builder import COMPONENT_TABLE_SCHEMA, DetectorComponentTable


def merge_component_tables(
    *,
    input_paths: Sequence[Path],
    expected_component: str,
    output_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    paths = [Path(path) for path in input_paths]
    if len(paths) < 2 or len({str(path.resolve()) for path in paths}) != len(paths):
        raise ValueError("component merge requires at least two unique inputs")
    tables = [DetectorComponentTable.load(path, expected_component) for path in paths]
    names = tables[0].feature_names
    action_sets = [set(table.actions.tolist()) for table in tables]
    if any(table.feature_names != names for table in tables[1:]):
        raise ValueError("component tables have different feature schemas")
    if any(len(values) != 1 for values in action_sets) or any(values != action_sets[0] for values in action_sets[1:]):
        raise ValueError("component tables do not represent the same single action")
    all_ids = np.concatenate([table.sample_ids for table in tables])
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("component inputs overlap in sample ids")
    order = np.argsort(all_ids, kind="stable")
    arrays = {
        "features": np.concatenate([table.features for table in tables], axis=0)[order],
        "sample_ids": all_ids[order],
    }
    for name in ("labels", "user_ids", "pools", "actions", "duration_ms", "pair_identity_sha256"):
        arrays[name] = np.concatenate([np.asarray(getattr(table, name)) for table in tables])[order]
    if set(arrays["labels"].tolist()) != {0, 1}:
        raise ValueError("merged component must contain real=0 and fake=1")
    if set(arrays["pools"].astype(str).tolist()) != {"train", "val", "test"}:
        raise ValueError("merged component must cover train/val/test")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        str(temporary),
        schema_version=np.asarray(COMPONENT_TABLE_SCHEMA),
        component=np.asarray(expected_component),
        feature_names=np.asarray(names),
        **arrays,
    )
    temporary.replace(output)
    merged = DetectorComponentTable.load(output, expected_component)
    manifest: Dict[str, Any] = {
        "schema_version": "detector_component_merge_manifest_v1",
        "status": "complete",
        "component": expected_component,
        "action": next(iter(action_sets[0])),
        "rows": int(merged.features.shape[0]),
        "features": int(merged.features.shape[1]),
        "labels": {str(value): int(np.sum(merged.labels == value)) for value in (0, 1)},
        "pools": {
            value: int(np.sum(merged.pools == value)) for value in ("train", "val", "test")
        },
        "join_policy": "disjoint_sample_ids_then_canonical_sort",
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths
        ],
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_manifest = target.with_name(target.name + ".tmp")
    temp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_manifest.replace(target)
    return manifest


__all__ = ["merge_component_tables"]
