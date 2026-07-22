"""Deterministically merge IMU, trajectory and consistency component tables."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .paired_dataset import PAIRED_DATASET_SCHEMA, PairedDetectorTable


COMPONENT_TABLE_SCHEMA = "paired_detector_component_table_v1"
COMPONENT_PREFIX = {
    "imu": "imu__",
    "trajectory": "trajectory__",
    "consistency": "consistency__",
}
METADATA_FIELDS = (
    "labels", "user_ids", "pools", "actions", "duration_ms", "pair_identity_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError("%s must be a scalar string" % name)
    return str(array.item())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


@dataclass(frozen=True)
class DetectorComponentTable:
    component: str
    features: np.ndarray
    feature_names: Tuple[str, ...]
    sample_ids: np.ndarray
    labels: np.ndarray
    user_ids: np.ndarray
    pools: np.ndarray
    actions: np.ndarray
    duration_ms: np.ndarray
    pair_identity_sha256: np.ndarray

    @classmethod
    def load(cls, path: Path, expected_component: str) -> "DetectorComponentTable":
        with np.load(str(path), allow_pickle=False) as archive:
            required = {
                "schema_version", "component", "features", "feature_names", "sample_ids",
                *METADATA_FIELDS,
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError("component table missing fields: %s" % sorted(missing))
            if _scalar_text(archive["schema_version"], "schema_version") != COMPONENT_TABLE_SCHEMA:
                raise ValueError("component table schema mismatch")
            component = _scalar_text(archive["component"], "component")
            table = cls(
                component=component,
                features=np.asarray(archive["features"], dtype=np.float64),
                feature_names=tuple(str(value) for value in np.asarray(archive["feature_names"]).tolist()),
                sample_ids=np.asarray(archive["sample_ids"]).astype(str),
                labels=np.asarray(archive["labels"], dtype=np.int64),
                user_ids=np.asarray(archive["user_ids"]),
                pools=np.asarray(archive["pools"]).astype(str),
                actions=np.asarray(archive["actions"]).astype(str),
                duration_ms=np.asarray(archive["duration_ms"], dtype=np.float64),
                pair_identity_sha256=np.asarray(archive["pair_identity_sha256"]).astype(str),
            )
        table.validate(expected_component)
        return table

    def validate(self, expected_component: str) -> None:
        if expected_component not in COMPONENT_PREFIX or self.component != expected_component:
            raise ValueError("component mismatch: expected %s, found %s" % (expected_component, self.component))
        if self.features.ndim != 2 or not self.features.shape[0] or not self.features.shape[1]:
            raise ValueError("component features must be non-empty [N,D]")
        if not np.all(np.isfinite(self.features)):
            raise ValueError("component features contain non-finite values")
        n, width = self.features.shape
        if len(self.feature_names) != width or len(set(self.feature_names)) != width:
            raise ValueError("component feature names must be unique and match width")
        prefix = COMPONENT_PREFIX[expected_component]
        if any(not name.startswith(prefix) for name in self.feature_names):
            raise ValueError("%s features must use %s prefix" % (expected_component, prefix))
        for name in ("sample_ids",) + METADATA_FIELDS:
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or len(value) != n:
                raise ValueError("%s must be a vector of length %d" % (name, n))
            if value.dtype.kind == "O":
                raise ValueError("%s must not use object dtype" % name)
        if np.any(self.sample_ids == "") or len(np.unique(self.sample_ids)) != n:
            raise ValueError("component sample_ids must be non-empty and unique")
        if any(_SHA256.fullmatch(value) is None for value in self.pair_identity_sha256.tolist()):
            raise ValueError("pair_identity_sha256 must contain lowercase 64-hex identities")
        if set(np.unique(self.labels).tolist()) - {0, 1}:
            raise ValueError("labels must use real=0/fake=1")
        if not np.all(np.isfinite(self.duration_ms)) or np.any(self.duration_ms <= 0):
            raise ValueError("duration_ms must be finite and positive")


def _canonical_order(table: DetectorComponentTable) -> np.ndarray:
    return np.argsort(table.sample_ids, kind="stable")


def _assert_same_metadata(
    reference: DetectorComponentTable,
    other: DetectorComponentTable,
    reference_order: np.ndarray,
    other_order: np.ndarray,
) -> None:
    left_ids = reference.sample_ids[reference_order]
    right_ids = other.sample_ids[other_order]
    if not np.array_equal(left_ids, right_ids):
        missing = sorted(set(left_ids.tolist()) - set(right_ids.tolist()))[:5]
        extra = sorted(set(right_ids.tolist()) - set(left_ids.tolist()))[:5]
        raise ValueError("component sample-id sets differ; missing=%r extra=%r" % (missing, extra))
    for name in METADATA_FIELDS:
        left = np.asarray(getattr(reference, name))[reference_order]
        right = np.asarray(getattr(other, name))[other_order]
        equal = (
            np.allclose(left, right, rtol=0.0, atol=1.0e-6)
            if name == "duration_ms" else np.array_equal(left, right)
        )
        if not equal:
            mismatch = int(np.flatnonzero(left != right)[0]) if name != "duration_ms" else 0
            raise ValueError("paired metadata mismatch for %s near sample %s" % (name, left_ids[mismatch]))


def build_paired_detector_table(
    *,
    imu_path: Path,
    trajectory_path: Path,
    consistency_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    paths = {
        "imu": Path(imu_path),
        "trajectory": Path(trajectory_path),
        "consistency": Path(consistency_path),
    }
    tables = {name: DetectorComponentTable.load(path, name) for name, path in paths.items()}
    reference = tables["imu"]
    order = {name: _canonical_order(table) for name, table in tables.items()}
    for name in ("trajectory", "consistency"):
        _assert_same_metadata(reference, tables[name], order["imu"], order[name])
    feature_names = tuple(
        name for component in ("imu", "trajectory", "consistency")
        for name in tables[component].feature_names
    )
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("duplicate feature names across component tables")
    features = np.concatenate(
        [tables[name].features[order[name]] for name in ("imu", "trajectory", "consistency")],
        axis=1,
    )
    ref_order = order["imu"]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        str(temporary),
        schema_version=np.asarray(PAIRED_DATASET_SCHEMA),
        features=features,
        feature_names=np.asarray(feature_names),
        labels=reference.labels[ref_order],
        user_ids=reference.user_ids[ref_order],
        pools=reference.pools[ref_order],
        sample_ids=reference.sample_ids[ref_order],
        actions=reference.actions[ref_order],
        duration_ms=reference.duration_ms[ref_order],
        pair_identity_sha256=reference.pair_identity_sha256[ref_order],
    )
    temporary.replace(output)
    # Re-open the exact bytes with the public strict loader before declaring success.
    PairedDetectorTable.load(output)
    manifest: Dict[str, Any] = {
        "schema_version": "paired_detector_table_build_manifest_v1",
        "status": "complete",
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]),
        "feature_names": list(feature_names),
        "join_key": "sample_id_exact_set_then_canonical_sort",
        "identity_gate": "pair_identity_sha256_exact_match",
        "duration_tolerance_ms": 1.0e-6,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    target_manifest = Path(manifest_path)
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    temp_manifest = target_manifest.with_name(target_manifest.name + ".tmp")
    temp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_manifest.replace(target_manifest)
    return manifest


__all__ = [
    "COMPONENT_TABLE_SCHEMA", "DetectorComponentTable", "build_paired_detector_table",
]
