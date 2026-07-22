"""Atomic, resumable, numeric-only NPZ output for generated trajectories."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .android import AndroidTrajectoryRecord
from .event_plan import EventPlan
from .protocol import (
    ACTION_TO_ID, ID_TO_ACTION, SPLIT_TO_ID, canonical_condition_request_digest,
    ddim_noise_seed, make_fake_id, stable_seed,
)


SCHEMA_VERSION = (1, 5)


def _require_zero_eta(value, *, archive_scalar: bool = False) -> float:
    """Validate the formal deterministic DDIM contract at every trust boundary."""

    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iuf":
        raise ValueError("DDIM eta must be one real scalar")
    if archive_scalar and array.dtype != np.dtype(np.float32):
        raise ValueError("archived DDIM eta must be a float32 scalar")
    eta = float(array)
    if not np.isfinite(eta) or eta != 0.0:
        raise ValueError("formal generation requires deterministic DDIM eta=0")
    return eta


def _digest_bytes(value: str) -> np.ndarray:
    if not value:
        return np.zeros(32, np.uint8)
    if len(value) != 64:
        raise ValueError("digest must be a 64-character SHA-256 hex string")
    return np.frombuffer(bytes.fromhex(value), dtype=np.uint8).copy()


def _concat(records: Sequence[AndroidTrajectoryRecord], name: str, dtype, width: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    values = [np.asarray(getattr(item, name)) for item in records]
    offsets = np.zeros(len(values) + 1, np.int64)
    for i, value in enumerate(values):
        offsets[i + 1] = offsets[i] + value.shape[0]
    if values:
        result = np.concatenate(values, axis=0).astype(dtype, copy=False)
    else:
        result = np.zeros((0, width), dtype=dtype) if width else np.zeros(0, dtype=dtype)
    return result, offsets


def build_numeric_archive(
    records: Sequence[AndroidTrajectoryRecord],
    ddim_steps: int,
    eta: float,
    training_diffusion_steps: int,
    alpha_bar_final: float,
    checkpoint_sha256: str,
    split_sha256: str,
    reference_registry_sha256: str,
    runtime_determinism_sha256: str,
    generation_base_seed: int,
    generation_batch_size: int,
    ddim_noise_seeds: Sequence[int],
) -> Dict[str, np.ndarray]:
    if not records:
        raise ValueError("cannot archive an empty generation unit")
    _require_zero_eta(eta)
    action = records[0].request.action
    user_id = records[0].request.user_id
    split = records[0].request.split
    if any(x.request.action != action or x.request.user_id != user_id or x.request.split != split for x in records):
        raise ValueError("one unit archive must contain one action/user/split")
    if len({x.request.fake_id for x in records}) != len(records):
        raise ValueError("duplicate fake ids in archive")
    prior_digests = {x.request.train_prior_digest for x in records}
    if len(prior_digests) != 1:
        raise ValueError("mixed train prior provenance")
    expected_seeds = np.asarray([
        stable_seed(generation_base_seed, action, user_id, x.request.sample_index)
        for x in records
    ], np.int64)
    observed_seeds = np.asarray([x.request.seed for x in records], np.int64)
    if not np.array_equal(observed_seeds, expected_seeds):
        raise ValueError("condition request seed is not derived from generation base seed")
    if int(generation_batch_size) <= 0:
        raise ValueError("generation batch size must be positive")
    expected_noise_seeds = np.asarray([
        ddim_noise_seed(
            x.request.seed, action, user_id, x.request.sample_index
        )
        for x in records
    ], np.int64)
    observed_noise_seeds = np.asarray(ddim_noise_seeds, np.int64)
    if observed_noise_seeds.shape != (len(records),) or not np.array_equal(
        observed_noise_seeds, expected_noise_seeds
    ):
        raise ValueError(
            "DDIM sampler seeds are not the audited domain-separated derivation"
        )

    trajectory_features, trajectory_offsets = _concat(records, "trajectory_features", np.float32, width=5)
    trajectory_t, _ = _concat(records, "trajectory_t_ms", np.float32)
    trajectory_pointer, _ = _concat(records, "trajectory_pointer_id", np.int8)
    trajectory_contact, _ = _concat(records, "trajectory_contact_mask", np.uint8)
    trajectory_event, _ = _concat(records, "trajectory_event_id", np.int16)
    android_t, android_offsets = _concat(records, "android_t_ms", np.float32)
    android_x, _ = _concat(records, "android_x", np.float32)
    android_y, _ = _concat(records, "android_y", np.float32)
    android_pressure, _ = _concat(records, "android_pressure", np.float32)
    android_size, _ = _concat(records, "android_size", np.float32)
    android_pointer, _ = _concat(records, "android_pointer_id", np.int8)
    android_slot, _ = _concat(records, "android_slot", np.int8)
    android_tracking, _ = _concat(records, "android_tracking_id", np.int32)
    android_type_b, _ = _concat(records, "android_type_b_tracking_value", np.int32)
    android_phase, _ = _concat(records, "android_phase", np.int8)
    android_action, _ = _concat(records, "android_action", np.int16)
    android_key_index, _ = _concat(records, "android_key_index", np.int16)
    android_keycode, _ = _concat(records, "android_keycode", np.int32)
    android_frame_index, _ = _concat(records, "android_frame_index", np.int32)
    android_frame_end, _ = _concat(records, "android_frame_end", np.uint8)

    key_offsets = np.zeros(len(records) + 1, np.int64)
    key_flight_offsets = np.zeros(len(records) + 1, np.int64)
    for i, record in enumerate(records):
        key_offsets[i + 1] = key_offsets[i] + record.request.keycodes.size
        key_flight_offsets[i + 1] = (
            key_flight_offsets[i] + record.request.zero_flight_after_key.size
        )
    flat_keycodes = np.concatenate([x.request.keycodes for x in records]).astype(np.int32) if key_offsets[-1] else np.zeros(0, np.int32)
    flat_zero_flight = (
        np.concatenate([x.request.zero_flight_after_key for x in records]).astype(np.uint8)
        if key_flight_offsets[-1]
        else np.zeros(0, np.uint8)
    )

    arrays: Dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, np.int16),
        "action_id_scalar": np.asarray(ACTION_TO_ID[action], np.int8),
        "ddim_steps_scalar": np.asarray(ddim_steps, np.int16),
        "ddim_eta_scalar": np.asarray(eta, np.float32),
        "training_diffusion_steps_scalar": np.asarray(training_diffusion_steps, np.int32),
        "alpha_bar_final_scalar": np.asarray(alpha_bar_final, np.float64),
        "checkpoint_sha256": _digest_bytes(checkpoint_sha256),
        "fixed_split_sha256": _digest_bytes(split_sha256),
        "reference_registry_sha256": _digest_bytes(reference_registry_sha256),
        "runtime_determinism_sha256": _digest_bytes(runtime_determinism_sha256),
        "train_prior_sha256": _digest_bytes(next(iter(prior_digests))),
        "generation_base_seed_scalar": np.asarray(generation_base_seed, np.int64),
        "generation_batch_size_scalar": np.asarray(generation_batch_size, np.int32),
        "condition_request_sha256": np.stack([
            np.frombuffer(canonical_condition_request_digest(x.request), dtype=np.uint8)
            for x in records
        ]).astype(np.uint8),
        "event_plan_sha256": np.stack([
            _digest_bytes(EventPlan.from_condition_request(
                x.request,
                sample_id=str(x.request.fake_id),
                start_time_ns=None,
                text=None,
            ).plan_sha256)
            for x in records
        ]).astype(np.uint8),
        "fake_id": np.asarray([x.request.fake_id for x in records], np.int64),
        "sample_index": np.asarray([x.request.sample_index for x in records], np.int32),
        "seed": np.asarray([x.request.seed for x in records], np.int64),
        "ddim_noise_seed": observed_noise_seeds,
        "user_id": np.asarray([x.request.user_id for x in records], np.int16),
        "split_id": np.asarray([SPLIT_TO_ID[x.request.split] for x in records], np.int8),
        "orientation_id": np.asarray([x.request.orientation_id for x in records], np.int8),
        "duration_ms": np.asarray([x.request.duration_ms for x in records], np.float32),
        "lengths": np.asarray([x.request.lengths for x in records], np.int32),
        "pointer_start_offset_ms": np.stack([x.request.pointer_start_offset_ms for x in records]).astype(np.float32),
        "pointer_end_offset_ms": np.stack([x.request.pointer_end_offset_ms for x in records]).astype(np.float32),
        "start_xy": np.stack([x.request.start_xy for x in records]).astype(np.float32),
        "end_xy": np.stack([x.request.end_xy for x in records]).astype(np.float32),
        "pinch_span": np.stack([x.request.pinch_span for x in records]).astype(np.float32),
        "pinch_angle": np.stack([x.request.pinch_angle for x in records]).astype(np.float32),
        "n_keys": np.asarray([x.request.n_keys for x in records], np.int16),
        "n_letters": np.asarray([x.request.n_letters for x in records], np.int16),
        "zero_flight_probability": np.asarray([
            x.request.zero_flight_probability for x in records
        ], np.float64),
        "key_endpoint_source_code": np.stack([
            x.request.key_endpoint_source_code for x in records
        ]).astype(np.int8),
        "condition_source_code": np.asarray([x.request.condition_source_code for x in records], np.int8),
        "condition_carrier_ref_id": np.asarray([x.request.carrier_ref_id for x in records], np.int64),
        "reference_event_ids": np.asarray([x.request.reference_ids for x in records], np.int64),
        "reference_canonical_sha256": np.stack([
            np.stack([_digest_bytes(value) for value in x.request.reference_canonical_sha256])
            for x in records
        ]).astype(np.uint8),
        "screen_min_xy": np.stack([x.request.screen_min_xy for x in records]).astype(np.float32),
        "screen_max_xy": np.stack([x.request.screen_max_xy for x in records]).astype(np.float32),
        "clipped_point_count": np.asarray([x.clipped_point_count for x in records], np.int32),
        "contact_point_count": np.asarray([x.contact_point_count for x in records], np.int32),
        "clipped_point_rate": np.asarray([
            x.clipped_point_count / max(x.contact_point_count, 1) for x in records
        ], np.float32),
        "key_offsets": key_offsets,
        "keycodes": flat_keycodes,
        "key_flight_offsets": key_flight_offsets,
        "flat_zero_flight_after_key": flat_zero_flight,
        "trajectory_offsets": trajectory_offsets,
        "trajectory_pointer_offsets": np.stack([x.trajectory_pointer_offsets for x in records]).astype(np.int64),
        "flat_trajectory_features": trajectory_features,
        "flat_trajectory_t_ms": trajectory_t,
        "flat_trajectory_pointer_id": trajectory_pointer,
        "flat_trajectory_contact_mask": trajectory_contact,
        "flat_trajectory_event_id": trajectory_event,
        "android_offsets": android_offsets,
        "flat_android_t_ms": android_t,
        "flat_android_x": android_x,
        "flat_android_y": android_y,
        "flat_android_pressure": android_pressure,
        "flat_android_size": android_size,
        "flat_android_pointer_id": android_pointer,
        "flat_android_slot": android_slot,
        "flat_android_tracking_id": android_tracking,
        "flat_android_type_b_tracking_value": android_type_b,
        "flat_android_phase": android_phase,
        "flat_android_action": android_action,
        "flat_android_key_index": android_key_index,
        "flat_android_keycode": android_keycode,
        "flat_android_frame_index": android_frame_index,
        "flat_android_frame_end": android_frame_end,
    }
    for name, value in arrays.items():
        if value.dtype.kind not in "biufc":
            raise TypeError("archive field %s is not numeric" % name)
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("archive field %s contains non-finite values" % name)
    return arrays


def validate_existing_unit(
    path: str, action_id: int, user_id: int, expected_count: int, ddim_steps: int,
    generation_base_seed: int, generation_batch_size: int,
    runtime_determinism_sha256: str,
    training_diffusion_steps: int = None,
) -> bool:
    source = Path(path)
    if not source.is_file():
        return False
    with np.load(str(source), allow_pickle=False) as a:
        required = {
            "schema_version", "runtime_determinism_sha256", "ddim_eta_scalar",
            "event_plan_sha256",
        }
        if not required.issubset(a.files):
            raise ValueError(
                "existing resume unit does not match requested protocol: %s" % source
            )
        if any(a[name].dtype.kind == "O" for name in a.files):
            raise ValueError("existing unit contains object arrays")
        _require_zero_eta(a["ddim_eta_scalar"], archive_scalar=True)
        action = ID_TO_ACTION.get(int(action_id))
        if action is None:
            raise ValueError("unknown requested action id")
        sample_index = np.asarray(a["sample_index"], np.int64)
        expected_indices = np.arange(expected_count, dtype=np.int64)
        expected_seeds = np.asarray([
            stable_seed(generation_base_seed, action, user_id, int(index))
            for index in expected_indices.tolist()
        ], np.int64)
        expected_fake_ids = np.asarray([
            make_fake_id(action, user_id, int(index))
            for index in expected_indices.tolist()
        ], np.int64)
        expected_noise_seeds = np.asarray([
            ddim_noise_seed(
                int(expected_seeds[index]), action, user_id, int(sample_index)
            )
            for index, sample_index in enumerate(expected_indices.tolist())
        ], np.int64)
        checks = (
            tuple(a["schema_version"].tolist()) == SCHEMA_VERSION,
            int(a["action_id_scalar"]) == action_id,
            int(a["ddim_steps_scalar"]) == ddim_steps,
            training_diffusion_steps is None or int(a["training_diffusion_steps_scalar"]) == training_diffusion_steps,
            int(a["generation_base_seed_scalar"]) == int(generation_base_seed),
            int(a["generation_batch_size_scalar"]) == int(generation_batch_size),
            np.array_equal(
                np.asarray(a["runtime_determinism_sha256"], np.uint8),
                _digest_bytes(runtime_determinism_sha256),
            ),
            int(a["fake_id"].size) == expected_count,
            np.all(a["user_id"] == user_id),
            np.array_equal(sample_index, expected_indices),
            np.array_equal(np.asarray(a["seed"], np.int64), expected_seeds),
            np.array_equal(
                np.asarray(a["ddim_noise_seed"], np.int64), expected_noise_seeds
            ),
            np.array_equal(np.asarray(a["fake_id"], np.int64), expected_fake_ids),
            a["condition_request_sha256"].shape == (expected_count, 32),
            a["event_plan_sha256"].shape == (expected_count, 32),
            np.unique(a["reference_event_ids"], axis=0).shape[0] == 1,
        )
        if not all(checks):
            raise ValueError("existing resume unit does not match requested protocol: %s" % source)
    return True


def atomic_save_npz(path: str, arrays: Mapping[str, np.ndarray]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp.%d.%s" % (os.getpid(), uuid.uuid4().hex))
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(target))
        directory_fd = os.open(str(target.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
