"""Generate resumable fake IMU units bound to archived trajectory EventPlans."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .fake_event_plan_archive import load_event_plans_from_archive


FAKE_IMU_PAIR_SCHEMA = "paired_fake_imu_eventplan_archive_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.%d.%s" % (os.getpid(), uuid.uuid4().hex))
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_generated_result(result: Mapping[str, Any], plan: Any) -> None:
    if result.get("release_backend") != "online_five_shot_diffusion":
        raise ValueError("formal paired fake IMU requires the online diffusion backend")
    active = np.asarray(result.get("active_imu"), dtype=np.float32)
    relative = np.asarray(result.get("relative_timestamps_ns"), dtype=np.int64)
    if active.ndim != 2 or active.shape[1] != 6 or not len(active) or not np.all(np.isfinite(active)):
        raise ValueError("paired fake IMU waveform is invalid")
    if relative.shape != (len(active),) or relative[0] != 0 or np.any(np.diff(relative) <= 0):
        raise ValueError("paired fake IMU timeline is invalid")
    metadata = result.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("paired fake IMU metadata is missing")
    exact = (
        str(result.get("action")) == plan.action,
        int(metadata.get("user_id", -1)) == int(plan.user_id),
        int(metadata.get("orientation_id", -999)) == int(plan.orientation_id),
        int(metadata.get("noise_seed", -1)) == int(plan.imu_noise_seed),
        int(metadata.get("ref_count", -1)) == 5,
    )
    if not all(exact):
        raise ValueError("paired fake IMU metadata disagrees with EventPlan/five-shot contract")
    if not np.isclose(
        float(metadata.get("logical_event_duration_ms", np.nan)),
        float(plan.duration_ms), rtol=0.0, atol=1.0e-6,
    ):
        raise ValueError("paired fake IMU logical duration disagrees with EventPlan")
    used_refs = metadata.get("used_ref_indices")
    if not isinstance(used_refs, list) or len(used_refs) != 5 or len(set(map(int, used_refs))) != 5:
        raise ValueError("paired fake IMU did not bind five unique IMU references")


def build_fake_imu_unit(
    *,
    trajectory_archive_path: Path,
    output_path: Path,
    service: Any,
    expected_action: str,
    samples_per_user: int,
    sample_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate the first N deterministic plans in one action/user archive."""

    if isinstance(samples_per_user, bool) or int(samples_per_user) != samples_per_user:
        raise ValueError("samples_per_user must be an integer")
    count = int(samples_per_user)
    if count < 1 or count > 200:
        raise ValueError("samples_per_user must lie in [1,200]")
    if sample_steps is not None and (
        isinstance(sample_steps, bool) or int(sample_steps) != sample_steps or int(sample_steps) < 1
    ):
        raise ValueError("sample_steps must be a positive integer or None")
    plans = load_event_plans_from_archive(
        trajectory_archive_path, expected_action=expected_action
    )
    selected = [plan for plan in plans if int(plan.sample_index) < count]
    if len(selected) != count or sorted(plan.sample_index for plan in selected) != list(range(count)):
        raise ValueError("trajectory unit does not contain the exact requested sample-index prefix")
    if len({plan.user_id for plan in selected}) != 1 or len({plan.split for plan in selected}) != 1:
        raise ValueError("one paired fake IMU unit must contain one user/split")

    active_rows = []
    relative_rows = []
    output_metadata = []
    offsets = np.zeros(count + 1, dtype=np.int64)
    for index, plan in enumerate(selected):
        kwargs = plan.to_imu_kwargs()
        if sample_steps is not None:
            kwargs["sample_steps"] = int(sample_steps)
        result = service.generate(plan.action, **kwargs)
        _validate_generated_result(result, plan)
        active = np.ascontiguousarray(result["active_imu"], dtype=np.float32)
        relative = np.asarray(result["relative_timestamps_ns"], dtype=np.int64)
        active_rows.append(active)
        relative_rows.append(relative)
        output_metadata.append(result["metadata"])
        offsets[index + 1] = offsets[index] + len(active)
    arrays = {
        "schema_version": np.asarray(FAKE_IMU_PAIR_SCHEMA),
        "source_trajectory_archive_sha256": np.asarray(sha256_file(trajectory_archive_path)),
        "action": np.asarray(expected_action),
        "sample_ids": np.asarray([plan.sample_id for plan in selected]),
        "fake_ids": np.asarray([plan.fake_id for plan in selected], dtype=np.int64),
        "sample_indices": np.asarray([plan.sample_index for plan in selected], dtype=np.int32),
        "user_ids": np.asarray([plan.user_id for plan in selected], dtype=np.int16),
        "pools": np.asarray([plan.split for plan in selected]),
        "duration_ms": np.asarray([plan.duration_ms for plan in selected], dtype=np.float64),
        "event_plan_sha256": np.asarray([plan.plan_sha256 for plan in selected]),
        "condition_seeds": np.asarray([plan.condition_seed for plan in selected], dtype=np.int64),
        "trajectory_noise_seeds": np.asarray(
            [plan.trajectory_noise_seed for plan in selected], dtype=np.int64
        ),
        "imu_noise_seeds": np.asarray([plan.imu_noise_seed for plan in selected], dtype=np.int64),
        "imu_reference_indices": np.asarray(
            [row["used_ref_indices"] for row in output_metadata], dtype=np.int64
        ),
        "imu_sample_steps": np.asarray(
            [int(row["sample_steps"]) for row in output_metadata], dtype=np.int32
        ),
        "imu_offsets": offsets,
        "flat_active_imu": np.concatenate(active_rows, axis=0).astype(np.float32, copy=False),
        "flat_relative_timestamps_ns": np.concatenate(relative_rows).astype(np.int64, copy=False),
    }
    _atomic_npz(output_path, arrays)
    audit = validate_fake_imu_unit(
        output_path,
        trajectory_archive_path=trajectory_archive_path,
        expected_action=expected_action,
        expected_samples=count,
    )
    audit.update(
        schema_version="paired_fake_imu_unit_audit_v1",
        status="pass",
        output=str(Path(output_path).resolve()),
        output_sha256=sha256_file(output_path),
        trajectory_archive=str(Path(trajectory_archive_path).resolve()),
    )
    return audit


def validate_fake_imu_unit(
    path: Path,
    *,
    trajectory_archive_path: Path,
    expected_action: str,
    expected_samples: int,
) -> Dict[str, Any]:
    plans = load_event_plans_from_archive(
        trajectory_archive_path, expected_action=expected_action
    )
    expected = [plan for plan in plans if plan.sample_index < int(expected_samples)]
    with np.load(str(path), allow_pickle=False) as source:
        required = {
            "schema_version", "source_trajectory_archive_sha256", "action", "sample_ids",
            "fake_ids", "sample_indices", "user_ids", "pools", "duration_ms",
            "event_plan_sha256", "condition_seeds", "trajectory_noise_seeds",
            "imu_noise_seeds", "imu_reference_indices", "imu_sample_steps", "imu_offsets",
            "flat_active_imu", "flat_relative_timestamps_ns",
        }
        if required - set(source.files) or any(source[name].dtype.kind == "O" for name in source.files):
            raise ValueError("paired fake IMU unit has an invalid numeric schema")
        data = {name: np.asarray(source[name]) for name in required}
    n = int(expected_samples)
    if str(data["schema_version"].item()) != FAKE_IMU_PAIR_SCHEMA:
        raise ValueError("paired fake IMU schema mismatch")
    if str(data["action"].item()) != expected_action:
        raise ValueError("paired fake IMU action mismatch")
    if str(data["source_trajectory_archive_sha256"].item()) != sha256_file(trajectory_archive_path):
        raise ValueError("paired fake IMU source archive SHA-256 mismatch")
    vector_expectations = {
        "sample_ids": np.asarray([plan.sample_id for plan in expected]),
        "fake_ids": np.asarray([plan.fake_id for plan in expected], dtype=np.int64),
        "sample_indices": np.arange(n, dtype=np.int32),
        "user_ids": np.asarray([plan.user_id for plan in expected], dtype=np.int16),
        "pools": np.asarray([plan.split for plan in expected]),
        "duration_ms": np.asarray([plan.duration_ms for plan in expected], dtype=np.float64),
        "event_plan_sha256": np.asarray([plan.plan_sha256 for plan in expected]),
        "condition_seeds": np.asarray([plan.condition_seed for plan in expected], dtype=np.int64),
        "trajectory_noise_seeds": np.asarray(
            [plan.trajectory_noise_seed for plan in expected], dtype=np.int64
        ),
        "imu_noise_seeds": np.asarray([plan.imu_noise_seed for plan in expected], dtype=np.int64),
    }
    if any(not np.array_equal(data[name], value) for name, value in vector_expectations.items()):
        raise ValueError("paired fake IMU identity/EventPlan vectors disagree with trajectory archive")
    offsets = np.asarray(data["imu_offsets"], dtype=np.int64)
    active = np.asarray(data["flat_active_imu"], dtype=np.float32)
    relative = np.asarray(data["flat_relative_timestamps_ns"], dtype=np.int64)
    if (
        offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(active)
        or np.any(np.diff(offsets) <= 0) or active.ndim != 2 or active.shape[1] != 6
        or relative.shape != (len(active),) or not np.all(np.isfinite(active))
    ):
        raise ValueError("paired fake IMU ragged waveform arrays are invalid")
    for index in range(n):
        left, right = int(offsets[index]), int(offsets[index + 1])
        timeline = relative[left:right]
        if timeline[0] != 0 or np.any(np.diff(timeline) <= 0):
            raise ValueError("paired fake IMU per-event timeline is invalid")
    references = np.asarray(data["imu_reference_indices"], dtype=np.int64)
    if (
        references.shape != (n, 5)
        or any(len(set(row.tolist())) != 5 for row in references)
        or np.unique(references, axis=0).shape[0] != 1
    ):
        raise ValueError("paired fake IMU five-shot reference matrix is invalid")
    steps = np.asarray(data["imu_sample_steps"], dtype=np.int32)
    if steps.shape != (n,) or np.any(steps < 1):
        raise ValueError("paired fake IMU sample-step vector is invalid")
    return {
        "action": expected_action,
        "rows": n,
        "user_id": int(data["user_ids"][0]),
        "pool": str(data["pools"][0]),
        "active_imu_rows": int(len(active)),
        "sample_steps": sorted(set(int(value) for value in steps.tolist())),
        "pair_identity": "event_plan_sha256_exact",
        "five_shot_policy": "five_unique_fixed_imu_reference_indices_per_user_action",
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


__all__ = [
    "FAKE_IMU_PAIR_SCHEMA", "build_fake_imu_unit", "sha256_file",
    "validate_fake_imu_unit", "write_json",
]
