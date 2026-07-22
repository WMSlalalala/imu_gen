"""Build a strict real-event index between HMOG touch trajectories and IMU windows."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .paired_dataset import ALLOWED_ACTIONS


REAL_PAIR_INDEX_SCHEMA = "hmog_real_imu_trajectory_pair_index_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _identity_key(user: int, activity: int, start_ms: int, end_ms: int) -> Tuple[int, int, int, int]:
    key = (int(user), int(activity), int(start_ms), int(end_ms))
    if key[0] < 0 or key[2] < 0 or key[3] <= key[2]:
        raise ValueError("invalid real event identity fields: %r" % (key,))
    return key


def _pair_digest(action: str, key: Tuple[int, int, int, int]) -> str:
    value = "real_hmog_pair_v1|%s|%d|%d|%d|%d" % ((action,) + key)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _full_trajectory_pools(
    user_ids: np.ndarray,
    event_ids: np.ndarray,
    action: str,
    seed: int,
) -> np.ndarray:
    pools = np.empty(len(user_ids), dtype="U5")
    rows_by_user: Dict[int, List[int]] = defaultdict(list)
    for index, user in enumerate(user_ids):
        rows_by_user[int(user)].append(index)
    for user, rows in rows_by_user.items():
        ranked = sorted(
            rows,
            key=lambda index: hashlib.sha256(
                ("%d|%d|%s|%s" % (int(seed), user, action, int(event_ids[index]))).encode("utf-8")
            ).digest(),
        )
        n_train = int(math.floor(0.60 * len(ranked)))
        n_val = int(math.floor(0.20 * len(ranked)))
        for rank, index in enumerate(ranked):
            pools[index] = "train" if rank < n_train else ("val" if rank < n_train + n_val else "test")
    return pools


def build_real_pair_index(
    *,
    action: str,
    trajectory_path: Path,
    imu_path: Path,
    output_path: Path,
    real_hash_seed: int = 20260713,
) -> Dict[str, Any]:
    if action not in ALLOWED_ACTIONS:
        raise ValueError("action is outside the frozen five-action protocol")
    with np.load(str(trajectory_path), allow_pickle=False) as source:
        required_t = {
            "action_name", "event_id", "user_id", "activity_id", "session_id",
            "orientation_id", "label_start_ms", "label_end_ms", "label_duration_ms",
        }
        if required_t - set(source.files):
            raise ValueError("trajectory source lacks identity fields: %s" % sorted(required_t - set(source.files)))
        if str(np.asarray(source["action_name"]).item()) != action:
            raise ValueError("trajectory action mismatch")
        trajectory = {name: np.asarray(source[name]) for name in required_t if name != "action_name"}
    with np.load(str(imu_path), allow_pickle=False) as source:
        required_i = {
            "event_id", "user_id", "activity_id", "session_id", "orientation_id",
            "event_start_ms", "event_end_ms", "event_duration_ms", "active_len",
        }
        if action == "keystroke":
            required_i.add("chunk_idx")
        if required_i - set(source.files):
            raise ValueError("IMU source lacks identity fields: %s" % sorted(required_i - set(source.files)))
        imu = {name: np.asarray(source[name]) for name in required_i}

    n_trajectory = len(trajectory["event_id"])
    n_imu_rows = len(imu["event_id"])
    for name, value in trajectory.items():
        if value.ndim != 1 or len(value) != n_trajectory:
            raise ValueError("trajectory identity field shape mismatch: %s" % name)
    for name, value in imu.items():
        if value.ndim != 1 or len(value) != n_imu_rows:
            raise ValueError("IMU identity field shape mismatch: %s" % name)

    trajectory_by_identity: Dict[Tuple[int, int, int, int], int] = {}
    for index in range(n_trajectory):
        key = _identity_key(
            trajectory["user_id"][index], trajectory["activity_id"][index],
            trajectory["label_start_ms"][index], trajectory["label_end_ms"][index],
        )
        if key in trajectory_by_identity:
            raise ValueError("duplicate trajectory absolute-time identity: %r" % (key,))
        trajectory_by_identity[key] = index
    imu_by_identity: Dict[Tuple[int, int, int, int], List[int]] = defaultdict(list)
    for index in range(n_imu_rows):
        key = _identity_key(
            imu["user_id"][index], imu["activity_id"][index],
            imu["event_start_ms"][index], imu["event_end_ms"][index],
        )
        imu_by_identity[key].append(index)

    full_pools = _full_trajectory_pools(
        trajectory["user_id"], trajectory["event_id"], action, real_hash_seed
    )
    common = sorted(set(trajectory_by_identity) & set(imu_by_identity))
    sample_ids: List[str] = []
    digests: List[str] = []
    users: List[int] = []
    pools: List[str] = []
    durations: List[float] = []
    trajectory_rows: List[int] = []
    trajectory_event_ids: List[int] = []
    imu_event_ids: List[int] = []
    event_id_matches: List[int] = []
    imu_rows_flat: List[int] = []
    imu_row_offsets = [0]
    for key in common:
        trajectory_index = trajectory_by_identity[key]
        imu_indices = list(imu_by_identity[key])
        if action == "keystroke":
            imu_indices.sort(key=lambda index: int(imu["chunk_idx"][index]))
            chunks = [int(imu["chunk_idx"][index]) for index in imu_indices]
            if chunks != list(range(len(chunks))):
                raise ValueError("keystroke IMU chunk sequence is incomplete for %r" % (key,))
        elif len(imu_indices) != 1:
            raise ValueError("non-keystroke real event maps to multiple IMU rows: %r" % (key,))
        expected_session = int(trajectory["session_id"][trajectory_index])
        expected_orientation = int(trajectory["orientation_id"][trajectory_index])
        expected_duration = int(trajectory["label_duration_ms"][trajectory_index])
        if expected_duration != key[3] - key[2]:
            raise ValueError("trajectory label duration does not equal absolute endpoints")
        for imu_index in imu_indices:
            if int(imu["session_id"][imu_index]) != expected_session:
                raise ValueError("paired event session mismatch: %r" % (key,))
            if int(imu["orientation_id"][imu_index]) != expected_orientation:
                raise ValueError("paired event orientation mismatch: %r" % (key,))
            if int(imu["event_duration_ms"][imu_index]) != expected_duration:
                raise ValueError("paired event duration mismatch: %r" % (key,))
            if int(imu["active_len"][imu_index]) < 1:
                raise ValueError("paired IMU row has non-positive active_len")
        digest = _pair_digest(action, key)
        sample_ids.append("real:%s:%s" % (action, digest))
        digests.append(digest)
        users.append(key[0])
        pools.append(str(full_pools[trajectory_index]))
        durations.append(float(expected_duration))
        trajectory_rows.append(int(trajectory_index))
        trajectory_event_id = int(trajectory["event_id"][trajectory_index])
        imu_event_id = int(imu["event_id"][imu_indices[0]])
        trajectory_event_ids.append(trajectory_event_id)
        imu_event_ids.append(imu_event_id)
        event_id_matches.append(int(all(int(imu["event_id"][index]) == trajectory_event_id for index in imu_indices)))
        imu_rows_flat.extend(imu_indices)
        imu_row_offsets.append(len(imu_rows_flat))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        str(temporary), schema_version=np.asarray(REAL_PAIR_INDEX_SCHEMA), action=np.asarray(action),
        sample_ids=np.asarray(sample_ids), pair_identity_sha256=np.asarray(digests),
        labels=np.zeros(len(common), dtype=np.int8), user_ids=np.asarray(users, dtype=np.int64),
        pools=np.asarray(pools), duration_ms=np.asarray(durations, dtype=np.float64),
        trajectory_rows=np.asarray(trajectory_rows, dtype=np.int64),
        trajectory_event_ids=np.asarray(trajectory_event_ids, dtype=np.int64),
        imu_event_ids=np.asarray(imu_event_ids, dtype=np.int64),
        event_id_matches=np.asarray(event_id_matches, dtype=np.uint8),
        imu_row_offsets=np.asarray(imu_row_offsets, dtype=np.int64),
        imu_rows=np.asarray(imu_rows_flat, dtype=np.int64),
    )
    temporary.replace(output)
    pool_array = np.asarray(pools)
    user_array = np.asarray(users, dtype=np.int64)
    pool_counts = {pool: int(np.sum(pool_array == pool)) for pool in ("train", "val", "test")}
    user_counts = {pool: int(len(np.unique(user_array[pool_array == pool]))) for pool in ("train", "val", "test")}
    if not common or any(pool_counts[pool] == 0 for pool in pool_counts):
        raise ValueError("strict real intersection leaves an empty detector pool")
    report: Dict[str, Any] = {
        "schema_version": "hmog_real_pair_index_audit_v1",
        "status": "pass",
        "action": action,
        "identity_key": "user_id+activity_id+absolute_event_start_ms+absolute_event_end_ms",
        "event_id_role": "audit_only_not_join_key",
        "pool_policy": "full_trajectory_event_id_sha256_rank_60_20_20_before_pair_intersection",
        "real_hash_seed": int(real_hash_seed),
        "trajectory_events": int(n_trajectory),
        "imu_rows": int(n_imu_rows),
        "imu_unique_events": int(len(imu_by_identity)),
        "paired_events": int(len(common)),
        "trajectory_unmatched_events": int(len(trajectory_by_identity) - len(common)),
        "imu_unmatched_events": int(len(imu_by_identity) - len(common)),
        "event_id_match_count": int(sum(event_id_matches)),
        "event_id_mismatch_count": int(len(common) - sum(event_id_matches)),
        "pool_counts": pool_counts,
        "pool_user_counts": user_counts,
        "trajectory_path": str(Path(trajectory_path).resolve()),
        "trajectory_sha256": _sha256_file(trajectory_path),
        "imu_path": str(Path(imu_path).resolve()),
        "imu_sha256": _sha256_file(imu_path),
        "output": str(output.resolve()),
        "output_sha256": _sha256_file(output),
    }
    return report


def write_audit(path: Path, report: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


__all__ = ["REAL_PAIR_INDEX_SCHEMA", "build_real_pair_index", "write_audit"]
