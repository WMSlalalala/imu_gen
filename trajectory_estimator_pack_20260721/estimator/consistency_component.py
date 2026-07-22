"""Build the consistency component table from strict real-pair indices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from runtime.paired_layer import cross_modal_consistency_from_record
from .paired_dataset_builder import COMPONENT_TABLE_SCHEMA, DetectorComponentTable
from .real_pair_index import REAL_PAIR_INDEX_SCHEMA


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def build_real_consistency_component(
    *,
    pair_index_path: Path,
    trajectory_records: Sequence[Any],
    imu_path: Path,
    output_path: Path,
    trajectory_source_path: Path,
) -> Dict[str, Any]:
    with np.load(str(pair_index_path), allow_pickle=False) as source:
        required = {
            "schema_version", "action", "sample_ids", "pair_identity_sha256", "labels",
            "user_ids", "pools", "duration_ms", "trajectory_rows", "imu_row_offsets", "imu_rows",
        }
        if required - set(source.files):
            raise ValueError("real pair index lacks fields: %s" % sorted(required - set(source.files)))
        if str(np.asarray(source["schema_version"]).item()) != REAL_PAIR_INDEX_SCHEMA:
            raise ValueError("real pair index schema mismatch")
        index = {name: np.asarray(source[name]) for name in required if name not in {"schema_version"}}
    action = str(np.asarray(index["action"]).item())
    n = len(index["sample_ids"])
    if len(trajectory_records) <= int(np.max(index["trajectory_rows"])):
        raise ValueError("trajectory record list is shorter than pair-index rows")
    if np.any(index["labels"] != 0):
        raise ValueError("real pair index must contain label=0 only")
    offsets = np.asarray(index["imu_row_offsets"], dtype=np.int64)
    flat_rows = np.asarray(index["imu_rows"], dtype=np.int64)
    if offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(flat_rows) or np.any(np.diff(offsets) <= 0):
        raise ValueError("pair-index IMU row offsets are invalid")

    with np.load(str(imu_path), allow_pickle=False) as source:
        required_imu = {"windows", "mask", "active_len", "hz"}
        if required_imu - set(source.files):
            raise ValueError("IMU source lacks waveform/mask fields")
        windows = np.asarray(source["windows"], dtype=np.float32)
        masks = np.asarray(source["mask"], dtype=bool)
        active_len = np.asarray(source["active_len"], dtype=np.int64)
        hz = float(np.asarray(source["hz"]).item())
    if windows.ndim != 3 or windows.shape[2] != 6 or masks.shape != windows.shape[:2]:
        raise ValueError("IMU windows/mask have invalid shapes")
    if active_len.shape != (windows.shape[0],) or np.any(np.sum(masks, axis=1) != active_len):
        raise ValueError("IMU active_len disagrees with masks")
    if not np.isclose(hz, 100.0, rtol=0.0, atol=1.0e-6):
        raise ValueError("formal consistency component requires 100 Hz IMU")
    if flat_rows.size and (np.min(flat_rows) < 0 or np.max(flat_rows) >= windows.shape[0]):
        raise ValueError("pair index references an out-of-range IMU row")

    matrix = []
    expected_names = None
    for pair_index in range(n):
        row_ids = flat_rows[offsets[pair_index]:offsets[pair_index + 1]]
        chunks = [np.ascontiguousarray(windows[row][masks[row]], dtype=np.float32) for row in row_ids]
        imu = np.concatenate(chunks, axis=0)
        relative_ms = np.arange(len(imu), dtype=np.float64) * (1000.0 / hz)
        record = trajectory_records[int(index["trajectory_rows"][pair_index])]
        names, values = cross_modal_consistency_from_record(
            action=action,
            logical_duration_ms=float(index["duration_ms"][pair_index]),
            active_imu=imu,
            imu_relative_timestamps_ms=relative_ms,
            trajectory_record=record,
        )
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise RuntimeError("consistency feature order drifted within one component")
        matrix.append(values)
    features = np.stack(matrix, axis=0).astype(np.float64)
    if not np.all(np.isfinite(features)):
        raise ValueError("real consistency component contains non-finite features")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        str(temporary), schema_version=np.asarray(COMPONENT_TABLE_SCHEMA),
        component=np.asarray("consistency"), features=features,
        feature_names=np.asarray(expected_names), sample_ids=index["sample_ids"],
        labels=index["labels"], user_ids=index["user_ids"], pools=index["pools"],
        actions=np.full(n, action), duration_ms=index["duration_ms"],
        pair_identity_sha256=index["pair_identity_sha256"],
    )
    temporary.replace(output)
    report: Dict[str, Any] = {
        "schema_version": "real_consistency_component_audit_v1",
        "status": "pass",
        "action": action,
        "rows": int(n),
        "features": int(features.shape[1]),
        "feature_names": list(expected_names or ()),
        "pair_index": str(Path(pair_index_path).resolve()),
        "pair_index_sha256": _sha256(pair_index_path),
        "trajectory_source": str(Path(trajectory_source_path).resolve()),
        "trajectory_source_sha256": _sha256(trajectory_source_path),
        "imu_source": str(Path(imu_path).resolve()),
        "imu_source_sha256": _sha256(imu_path),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "imu_hz": hz,
        "timeline_policy": "concatenate_complete_active_chunks_then_100hz_relative_time",
        "trajectory_policy": "frozen_raw_detector_record_no_resampling",
    }
    return report


def validate_real_consistency_audit(
    audit_path: Path, *, expected_action: str
) -> Dict[str, Any]:
    """Fail closed before reusing a previously built real component."""

    audit_file = Path(audit_path)
    report = json.loads(audit_file.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != "real_consistency_component_audit_v1"
        or report.get("status") != "pass"
        or report.get("action") != expected_action
    ):
        raise ValueError("prior real consistency audit identity/status is invalid")
    for path_key, digest_key in (
        ("pair_index", "pair_index_sha256"),
        ("trajectory_source", "trajectory_source_sha256"),
        ("imu_source", "imu_source_sha256"),
        ("output", "output_sha256"),
    ):
        source = Path(str(report.get(path_key, "")))
        expected_digest = str(report.get(digest_key, ""))
        if not source.is_file() or len(expected_digest) != 64:
            raise ValueError("prior real consistency %s evidence is missing" % path_key)
        if _sha256(source) != expected_digest:
            raise ValueError("prior real consistency %s SHA-256 mismatch" % path_key)
    table = DetectorComponentTable.load(Path(report["output"]), "consistency")
    if (
        table.features.shape != (int(report.get("rows", -1)), int(report.get("features", -1)))
        or list(table.feature_names) != list(report.get("feature_names", ()))
        or set(table.actions.tolist()) != {expected_action}
        or set(table.labels.tolist()) != {0}
    ):
        raise ValueError("prior real consistency output disagrees with its audit")
    return report


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


__all__ = [
    "build_real_consistency_component", "validate_real_consistency_audit", "write_json",
]
