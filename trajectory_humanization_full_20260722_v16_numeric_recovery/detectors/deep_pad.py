"""Leakage-safe raw-sequence PAD for variable-length touch trajectories.

This module deliberately has no dependency on the trajectory generator or its
adversarial critics.  It implements two independent detector families:

* ``tcn``: a dilated temporal convolutional network;
* ``transformer``: a masked self-attention encoder.

Both consume the same event-global timeline.  Pinch pointers therefore retain
their independently staggered DOWN/UP states, and keystrokes retain explicit
no-contact flight tokens and per-contact keycodes.  No pointer is independently
warped to ``0 -> duration``.

Labels and decisions are fixed throughout the module::

    real = 0, fake = 1, larger score = more fake
    score < threshold  -> accepted as real
    score >= threshold -> rejected as fake

Only train rows fit normalization/model parameters, only validation selects a
checkpoint and operating thresholds, and test only applies frozen choices.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
# Required by deterministic CUDA matrix multiplication.  It must be present
# before CUDA context initialization; setdefault preserves an explicit caller
# choice while giving formal pair workers a safe default.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from detectors.feature_pad import (
    ALLOWED_POOLS,
    FAKE_LABEL,
    REAL_LABEL,
    fa_frr_curve,
    operating_metrics,
    select_validation_thresholds,
    user_level_bootstrap,
)
from trajectory.features import (
    KEYCODE_TOKEN_MAX,
    KEYCODE_VOCAB_SIZE,
    TRAJECTORY_FEATURE_SCHEMA_VERSION,
)


ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
DEEP_DETECTORS = ("tcn", "transformer")
POINTERS = 2
POINTER_CONTINUOUS_NAMES = ("x", "y", "pressure", "size")
N_POINTER_CONTINUOUS = len(POINTER_CONTINUOUS_NAMES)
ACTION_CODES = (-1, 0, 1, 2, 5, 6)


def deep_keycode_embedding_index(
    raw_or_canonical_keycode: int,
    keycode_vocab: int = KEYCODE_VOCAB_SIZE,
) -> int:
    """Return the lossless Deep PAD embedding index for one canonical code.

    Contact keycodes are already canonical (raw negative sentinels -> 0).
    ``-1`` is reserved for gap/no-contact padding and maps to embedding index
    zero.  Each audited non-negative code has a distinct ``code+1`` index;
    notably HMOG 8230 maps to 8231 rather than an overflow bucket.  Inputs
    outside the shared vocabulary fail closed instead of being clipped.
    """

    vocabulary = int(keycode_vocab)
    if vocabulary != KEYCODE_VOCAB_SIZE:
        raise ValueError(
            "Deep PAD keycode_vocab must equal the shared value %d"
            % KEYCODE_VOCAB_SIZE
        )
    value = int(raw_or_canonical_keycode)
    if value < 0:
        return 0
    if value >= vocabulary:
        raise ValueError(
            "canonical keycode %d outside shared [0,%d] vocabulary"
            % (value, vocabulary - 1)
        )
    return value + 1


def _finite_1d(values: np.ndarray, name: str, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != int(length):
        raise ValueError("%s must have shape [%d]" % (name, length))
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("%s must be finite" % name)
    return array


@dataclass
class RawTrajectoryRecord:
    """One variable-length event on a shared global frame timeline.

    ``pointer_continuous`` has shape ``[2,T,4]`` in raw screen units.  A row is
    meaningful only where ``contact_mask`` is true.  ``global_t_ms`` is never
    normalized per pointer; both pointer slots reference the same physical
    timestamp.  ``gap_mask`` denotes an explicit no-contact flight token (used
    by keystroke).  ``event_ids`` distinguish discrete key contacts.
    """

    action: str
    label: int
    user_id: int
    pool: str
    sample_id: str
    pointer_continuous: np.ndarray       # [2,T,4]: x,y,pressure,size
    global_t_ms: np.ndarray              # [T]
    contact_mask: np.ndarray             # [2,T]
    active_mask: np.ndarray              # [2,T]
    action_code: np.ndarray              # [2,T], -1 for no contact
    keycode: np.ndarray                  # [2,T], -1 for absent/unknown
    event_ids: np.ndarray                # [2,T], -1 for gap/no contact
    gap_mask: np.ndarray                 # [T]
    event_group_id: str = ""             # complete event ID; never frame/chunk ID

    def validate(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError("unsupported action: %r" % self.action)
        if int(self.label) not in (REAL_LABEL, FAKE_LABEL):
            raise ValueError("label must use real=0/fake=1")
        if self.pool not in ALLOWED_POOLS:
            raise ValueError("pool must be train/val/test")
        if not str(self.sample_id):
            raise ValueError("sample_id must be non-empty")
        if self.event_group_id and not str(self.event_group_id):
            raise ValueError("event_group_id must be a string")
        values = np.asarray(self.pointer_continuous, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != POINTERS or values.shape[2] != N_POINTER_CONTINUOUS:
            raise ValueError("pointer_continuous must have shape [2,T,4]")
        t = int(values.shape[1])
        if t < 2 or not np.all(np.isfinite(values)):
            raise ValueError("an event needs at least two finite global timeline tokens")
        times = _finite_1d(self.global_t_ms, "global_t_ms", t).astype(np.float64)
        deltas = np.diff(times)
        if np.any(deltas < 0.0):
            raise ValueError("global_t_ms must be monotonic non-decreasing")
        shape = (POINTERS, t)
        for name in ("contact_mask", "active_mask", "action_code", "keycode", "event_ids"):
            array = np.asarray(getattr(self, name))
            if tuple(array.shape) != shape:
                raise ValueError("%s must have shape [2,T]" % name)
        contact = np.asarray(self.contact_mask, dtype=bool)
        active = np.asarray(self.active_mask, dtype=bool)
        gap = _finite_1d(self.gap_mask, "gap_mask", t).astype(bool)
        codes = np.asarray(self.action_code, dtype=np.int64)
        keycodes = np.asarray(self.keycode, dtype=np.int64)
        events = np.asarray(self.event_ids, dtype=np.int64)
        if not set(np.unique(codes).tolist()).issubset(set(ACTION_CODES)):
            raise ValueError("action_code contains an unsupported Android actionMasked value")
        if np.any(active & ~contact):
            raise ValueError("active_mask must be a subset of contact_mask")
        if np.any(gap & np.any(contact, axis=0)):
            raise ValueError("gap tokens cannot contain a pointer contact")
        if np.any((~contact) & (codes != -1)):
            raise ValueError("no-contact pointer slots must use action_code=-1")
        if np.any((~contact) & (events != -1)):
            raise ValueError("no-contact pointer slots must use event_ids=-1")
        if np.any((~contact) & (keycodes != -1)):
            raise ValueError("no-contact pointer slots must use keycode=-1")
        if np.any(keycodes < -1):
            raise ValueError("raw negative keycode sentinels must be canonicalized before Deep PAD")
        if np.any(keycodes[contact] > KEYCODE_TOKEN_MAX):
            raise ValueError(
                "contact keycode outside shared [0,%d] vocabulary" % KEYCODE_TOKEN_MAX
            )
        if np.any(~contact) and not np.all(values[~contact] == 0.0):
            raise ValueError("no-contact pointer continuous values must be zero")
        if not np.any(contact):
            raise ValueError("event contains no touch contact")

        if self.action == "pinch":
            if not (np.any(contact[0]) and np.any(contact[1])):
                raise ValueError("pinch must retain both pointer trajectories")
            if np.any(gap):
                raise ValueError("pinch must not use keystroke flight tokens")
        elif np.any(contact[1]):
            raise ValueError("non-pinch actions must not populate pointer 1")

        if self.action == "keystroke":
            if np.any(keycodes[contact] < 0):
                raise ValueError("keystroke contacts require canonical non-negative keycodes")
            contact_events = events[contact]
            observed_events = sorted(
                set(int(value) for value in contact_events[contact_events >= 0].tolist())
            )
            if not observed_events:
                raise ValueError("keystroke must retain discrete contact event IDs")
            if observed_events != list(range(len(observed_events))):
                raise ValueError("keystroke event IDs must be canonical contiguous 0..n-1")
            if np.any(np.asarray(self.keycode)[1] >= 0):
                raise ValueError("keystroke keycodes belong to pointer 0 only")
            # Positive-flight transitions use one or more explicit no-contact
            # tokens; a true zero-flight UP/DOWN boundary instead keeps the two
            # ordered contact tokens directly adjacent at the same millisecond.
            event0 = events[0]
            contacts = contact[0]
            # Equal physical timestamps are legitimate only for a zero-flight
            # boundary: the previous key UP and next key DOWN remain two
            # ordered sequence tokens even though both were logged in the same
            # millisecond.  Never accept an intra-key duplicate or a duplicate
            # involving an invented gap token.
            for left_index in np.flatnonzero(deltas == 0.0):
                right_index = int(left_index + 1)
                if (
                    not contacts[left_index]
                    or not contacts[right_index]
                    or gap[left_index]
                    or gap[right_index]
                    or int(event0[left_index]) < 0
                    or int(event0[right_index]) <= int(event0[left_index])
                ):
                    raise ValueError(
                        "equal keystroke timestamps are allowed only across an ordered key boundary"
                    )
            previous_event = None
            for index in np.flatnonzero(contacts):
                current_event = int(event0[index])
                if previous_event is not None and current_event != previous_event:
                    if current_event != previous_event + 1:
                        raise ValueError("keystroke contact event IDs must advance by exactly one")
                    transition_dt = float(times[index] - times[last_contact_index])
                    between = gap[last_contact_index + 1:index]
                    if transition_dt > 0.0 and not np.any(between):
                        raise ValueError("positive-flight keystroke transition lacks a gap token")
                    if transition_dt == 0.0 and (
                        index != last_contact_index + 1 or np.any(between)
                    ):
                        raise ValueError("zero-flight keystroke transition must be directly adjacent")
                previous_event = current_event
                last_contact_index = int(index)
        else:
            if np.any(deltas == 0.0):
                raise ValueError("non-keystroke global_t_ms must be strictly increasing")
            if np.any(gap):
                raise ValueError("gap tokens are reserved for keystroke")


def make_record(
    *,
    action: str,
    label: int,
    user_id: int,
    pool: str,
    sample_id: str,
    pointer_continuous: np.ndarray,
    global_t_ms: np.ndarray,
    contact_mask: np.ndarray,
    active_mask: Optional[np.ndarray] = None,
    action_code: Optional[np.ndarray] = None,
    keycode: Optional[np.ndarray] = None,
    event_ids: Optional[np.ndarray] = None,
    gap_mask: Optional[np.ndarray] = None,
    event_group_id: Optional[str] = None,
) -> RawTrajectoryRecord:
    """Construct and validate a record, filling only explicit neutral fields."""

    values = np.asarray(pointer_continuous, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("pointer_continuous must have rank three")
    p, t, _ = values.shape
    if p != POINTERS:
        raise ValueError("exactly two pointer slots are required")
    contact = np.asarray(contact_mask, dtype=bool)
    default_codes = np.where(contact, 2, -1).astype(np.int16)
    record = RawTrajectoryRecord(
        action=str(action),
        label=int(label),
        user_id=int(user_id),
        pool=str(pool),
        sample_id=str(sample_id),
        pointer_continuous=values,
        global_t_ms=np.asarray(global_t_ms, dtype=np.float32),
        contact_mask=contact,
        active_mask=np.asarray(active_mask if active_mask is not None else contact, dtype=bool),
        action_code=np.asarray(action_code if action_code is not None else default_codes, dtype=np.int16),
        keycode=np.asarray(keycode if keycode is not None else np.full((p, t), -1), dtype=np.int32),
        event_ids=np.asarray(event_ids if event_ids is not None else np.where(contact, 0, -1), dtype=np.int32),
        gap_mask=np.asarray(gap_mask if gap_mask is not None else np.zeros((t,)), dtype=bool),
        event_group_id=str(event_group_id if event_group_id is not None else sample_id),
    )
    record.validate()
    return record


def load_fake_user_split(path: Path) -> Dict[str, Tuple[int, ...]]:
    """Load and strictly validate the fixed 70/10/20 fake-user split JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    output = {
        "train": tuple(int(value) for value in payload["train_users"]),
        "val": tuple(int(value) for value in payload["val_users"]),
        "test": tuple(int(value) for value in payload["test_users"]),
    }
    if tuple(len(output[name]) for name in ALLOWED_POOLS) != (70, 10, 20):
        raise ValueError("fake user split must contain exactly 70/10/20 users")
    sets = [set(output[name]) for name in ALLOWED_POOLS]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("fake train/val/test users must be disjoint")
    if len(set.union(*sets)) != 100:
        raise ValueError("fake user split must cover exactly 100 users")
    return output


def assign_strict_protocol_pools(
    records: Sequence[RawTrajectoryRecord],
    fake_user_split: Mapping[str, Sequence[int]],
    *,
    real_hash_seed: int = 20260713,
) -> Tuple[List[RawTrajectoryRecord], Dict[str, Any]]:
    """Assign the detector pools without mixing the real/fake boundaries.

    Fake events use the externally frozen 70/10/20 user split.  Real events
    rank complete event groups by a deterministic SHA-256 value separately
    within every user/action, then allocate 60/20/20.  Consequently test real
    covers held-out events from all 100 real users (when each user/action has
    enough events), whereas fake test contains only the frozen 20 fake users.
    """

    normalized_split = {
        pool: tuple(int(value) for value in fake_user_split[pool])
        for pool in ALLOWED_POOLS
    }
    fake_owner: Dict[int, str] = {}
    for pool, users in normalized_split.items():
        for user in users:
            if user in fake_owner:
                raise ValueError("fake user appears in multiple pools: %s" % user)
            fake_owner[user] = pool
    if tuple(len(normalized_split[name]) for name in ALLOWED_POOLS) != (70, 10, 20):
        raise ValueError("strict fake split must be 70/10/20")

    # Rank unique complete real events *within each user/action*.  A direct
    # hash%100 assignment is not sufficient: with finite samples it can leave
    # a real user with no held-out test event.  Hash ranking is deterministic
    # and input-order invariant while the explicit counts implement the stated
    # 60/20/20 event-group split.
    groups_by_user_action: Dict[Tuple[int, str], set] = {}
    for record in records:
        record.validate()
        if record.label == REAL_LABEL:
            key = (int(record.user_id), str(record.action))
            groups_by_user_action.setdefault(key, set()).add(
                str(record.event_group_id or record.sample_id)
            )
    group_pool: Dict[Tuple[int, str, str], str] = {}
    real_group_counts: Dict[str, int] = {pool: 0 for pool in ALLOWED_POOLS}
    for (user_id, action), group_ids in sorted(groups_by_user_action.items()):
        ranked = sorted(
            group_ids,
            key=lambda group_id: hashlib.sha256(
                ("%d|%d|%s|%s" % (int(real_hash_seed), user_id, action, group_id)).encode("utf-8")
            ).digest(),
        )
        n_groups = len(ranked)
        n_train = int(math.floor(0.60 * n_groups))
        n_val = int(math.floor(0.20 * n_groups))
        # Test receives the rounding remainder.  For every non-empty group this
        # is at least one event; formal HMOG user/actions have much more than one.
        for index, group_id in enumerate(ranked):
            pool = "train" if index < n_train else ("val" if index < n_train + n_val else "test")
            group_pool[(user_id, action, group_id)] = pool
            real_group_counts[pool] += 1

    assigned: List[RawTrajectoryRecord] = []
    counts: Dict[str, Dict[str, int]] = {
        "real": {pool: 0 for pool in ALLOWED_POOLS},
        "fake": {pool: 0 for pool in ALLOWED_POOLS},
    }
    users: Dict[str, Dict[str, set]] = {
        "real": {pool: set() for pool in ALLOWED_POOLS},
        "fake": {pool: set() for pool in ALLOWED_POOLS},
    }
    for record in records:
        record.validate()
        if record.label == FAKE_LABEL:
            if record.user_id not in fake_owner:
                raise ValueError("fake user is absent from fixed split: %s" % record.user_id)
            pool = fake_owner[record.user_id]
            label_name = "fake"
        else:
            group_id = str(record.event_group_id or record.sample_id)
            group_key = (int(record.user_id), str(record.action), group_id)
            if group_key not in group_pool:
                raise RuntimeError("complete real event group was not assigned")
            pool = group_pool[group_key]
            label_name = "real"
        assigned.append(replace(record, pool=pool))
        counts[label_name][pool] += 1
        users[label_name][pool].add(int(record.user_id))
    audit = {
        "schema_version": "trajectory_detector_split_v1",
        "fake_policy": "fixed_disjoint_users_70_10_20",
        "real_policy": "sha256_ranked_complete_event_group_per_user_action_60_20_20",
        "real_hash_seed": int(real_hash_seed),
        "counts": counts,
        "real_complete_event_group_counts": real_group_counts,
        "user_counts": {
            label: {pool: len(values) for pool, values in pools.items()}
            for label, pools in users.items()
        },
        "fake_users": {pool: list(normalized_split[pool]) for pool in ALLOWED_POOLS},
    }
    return assigned, audit


class RawSequenceNormalizer:
    """Train-only normalization for physical continuous channels and frame dt."""

    def __init__(self) -> None:
        self.pointer_mean = np.zeros((N_POINTER_CONTINUOUS,), dtype=np.float64)
        self.pointer_scale = np.ones((N_POINTER_CONTINUOUS,), dtype=np.float64)
        self.log_dt_mean = 0.0
        self.log_dt_scale = 1.0
        self.fit_sample_ids: Tuple[str, ...] = tuple()
        self.fitted = False

    def fit(self, records: Sequence[RawTrajectoryRecord]) -> "RawSequenceNormalizer":
        if not records:
            raise ValueError("normalizer needs non-empty train records")
        pointer_values: List[np.ndarray] = []
        log_dt_values: List[np.ndarray] = []
        sample_ids: List[str] = []
        for record in records:
            record.validate()
            if record.pool != "train":
                raise ValueError("normalizer.fit accepts train records only")
            values = np.asarray(record.pointer_continuous, dtype=np.float64)
            contact = np.asarray(record.contact_mask, dtype=bool)
            pointer_values.append(values[contact])
            times = np.asarray(record.global_t_ms, dtype=np.float64)
            dt = np.diff(times, prepend=times[0])
            positive = dt[dt > 0.0]
            dt[0] = float(np.median(positive)) if len(positive) else 1.0
            log_dt_values.append(np.log1p(dt))
            sample_ids.append(record.sample_id)
        stacked = np.concatenate(pointer_values, axis=0)
        if len(stacked) == 0 or not np.all(np.isfinite(stacked)):
            raise ValueError("train records contain no finite contact values")
        self.pointer_mean = np.mean(stacked, axis=0)
        self.pointer_scale = np.std(stacked, axis=0)
        self.pointer_scale[self.pointer_scale < 1.0e-8] = 1.0
        all_log_dt = np.concatenate(log_dt_values)
        self.log_dt_mean = float(np.mean(all_log_dt))
        self.log_dt_scale = float(np.std(all_log_dt))
        if self.log_dt_scale < 1.0e-8:
            self.log_dt_scale = 1.0
        self.fit_sample_ids = tuple(sample_ids)
        self.fitted = True
        return self

    def transform(self, record: RawTrajectoryRecord) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("normalizer must be fitted first")
        record.validate()
        values = (
            np.asarray(record.pointer_continuous, dtype=np.float64) - self.pointer_mean[None, None, :]
        ) / self.pointer_scale[None, None, :]
        values[~np.asarray(record.contact_mask, dtype=bool)] = 0.0
        times = np.asarray(record.global_t_ms, dtype=np.float64)
        dt = np.diff(times, prepend=times[0])
        positive = dt[dt > 0.0]
        dt[0] = float(np.median(positive)) if len(positive) else 1.0
        log_dt = (np.log1p(dt) - self.log_dt_mean) / self.log_dt_scale
        duration = max(float(times[-1] - times[0]), 1.0)
        progress = (times - times[0]) / duration
        return values.astype(np.float32), log_dt.astype(np.float32), progress.astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("normalizer is not fitted")
        return {
            "pointer_mean": self.pointer_mean.copy(),
            "pointer_scale": self.pointer_scale.copy(),
            "log_dt_mean": float(self.log_dt_mean),
            "log_dt_scale": float(self.log_dt_scale),
            "fit_sample_ids": list(self.fit_sample_ids),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "RawSequenceNormalizer":
        result = cls()
        result.pointer_mean = np.asarray(state["pointer_mean"], dtype=np.float64)
        result.pointer_scale = np.asarray(state["pointer_scale"], dtype=np.float64)
        if result.pointer_mean.shape != (N_POINTER_CONTINUOUS,) or result.pointer_scale.shape != (N_POINTER_CONTINUOUS,):
            raise ValueError("invalid normalizer pointer shape")
        result.log_dt_mean = float(state["log_dt_mean"])
        result.log_dt_scale = float(state["log_dt_scale"])
        result.fit_sample_ids = tuple(str(value) for value in state.get("fit_sample_ids", []))
        if not np.all(np.isfinite(result.pointer_mean)) or not np.all(np.isfinite(result.pointer_scale)):
            raise ValueError("invalid normalizer values")
        if np.any(result.pointer_scale <= 0.0) or not np.isfinite(result.log_dt_scale) or result.log_dt_scale <= 0.0:
            raise ValueError("invalid normalizer scale")
        result.fitted = True
        return result


@dataclass
class DeepBatch:
    pointer_continuous: torch.Tensor  # [B,2,T,4]
    log_dt: torch.Tensor              # [B,T]
    time_progress: torch.Tensor       # [B,T]
    frame_mask: torch.Tensor          # [B,T]
    contact_mask: torch.Tensor        # [B,2,T]
    active_mask: torch.Tensor         # [B,2,T]
    action_code: torch.Tensor         # [B,2,T]
    keycode: torch.Tensor             # [B,2,T]
    event_ids: torch.Tensor           # [B,2,T]
    gap_mask: torch.Tensor            # [B,T]
    labels: torch.Tensor              # [B]
    user_ids: torch.Tensor            # [B]
    sample_ids: Tuple[str, ...]
    pools: Tuple[str, ...]
    actions: Tuple[str, ...]

    def to(self, device: torch.device) -> "DeepBatch":
        values = dict(self.__dict__)
        for name in (
            "pointer_continuous", "log_dt", "time_progress", "frame_mask",
            "contact_mask", "active_mask", "action_code", "keycode",
            "event_ids", "gap_mask", "labels", "user_ids",
        ):
            values[name] = getattr(self, name).to(device)
        return DeepBatch(**values)


class _RecordDataset(Dataset):
    def __init__(self, records: Sequence[RawTrajectoryRecord]) -> None:
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> RawTrajectoryRecord:
        return self.records[index]


def collate_raw_sequences(
    records: Sequence[RawTrajectoryRecord], normalizer: RawSequenceNormalizer
) -> DeepBatch:
    if not records:
        raise ValueError("cannot collate an empty batch")
    for record in records:
        record.validate()
    max_t = max(len(record.global_t_ms) for record in records)
    b = len(records)
    pointer = np.zeros((b, POINTERS, max_t, N_POINTER_CONTINUOUS), dtype=np.float32)
    log_dt = np.zeros((b, max_t), dtype=np.float32)
    progress = np.zeros((b, max_t), dtype=np.float32)
    frame = np.zeros((b, max_t), dtype=bool)
    contact = np.zeros((b, POINTERS, max_t), dtype=bool)
    active = np.zeros_like(contact)
    action_code = np.full((b, POINTERS, max_t), -1, dtype=np.int64)
    keycode = np.full((b, POINTERS, max_t), -1, dtype=np.int64)
    event_ids = np.full((b, POINTERS, max_t), -1, dtype=np.int64)
    gap = np.zeros((b, max_t), dtype=bool)
    for i, record in enumerate(records):
        n = len(record.global_t_ms)
        normalized, log_values, time_values = normalizer.transform(record)
        pointer[i, :, :n] = normalized
        log_dt[i, :n] = log_values
        progress[i, :n] = time_values
        frame[i, :n] = True
        contact[i, :, :n] = record.contact_mask
        active[i, :, :n] = record.active_mask
        action_code[i, :, :n] = record.action_code
        keycode[i, :, :n] = record.keycode
        event_ids[i, :, :n] = record.event_ids
        gap[i, :n] = record.gap_mask
    return DeepBatch(
        pointer_continuous=torch.from_numpy(pointer),
        log_dt=torch.from_numpy(log_dt),
        time_progress=torch.from_numpy(progress),
        frame_mask=torch.from_numpy(frame),
        contact_mask=torch.from_numpy(contact),
        active_mask=torch.from_numpy(active),
        action_code=torch.from_numpy(action_code),
        keycode=torch.from_numpy(keycode),
        event_ids=torch.from_numpy(event_ids),
        gap_mask=torch.from_numpy(gap),
        labels=torch.tensor([record.label for record in records], dtype=torch.float32),
        user_ids=torch.tensor([record.user_id for record in records], dtype=torch.long),
        sample_ids=tuple(record.sample_id for record in records),
        pools=tuple(record.pool for record in records),
        actions=tuple(record.action for record in records),
    )


class _FrameEncoder(nn.Module):
    """Shared raw frame encoder; categorical channels are never normalized."""

    def __init__(
        self,
        hidden_dim: int,
        key_embedding_dim: int = 8,
        keycode_vocab: int = KEYCODE_VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.keycode_vocab = int(keycode_vocab)
        if self.keycode_vocab != KEYCODE_VOCAB_SIZE:
            raise ValueError(
                "keycode_vocab must equal the shared %d-token vocabulary"
                % KEYCODE_VOCAB_SIZE
            )
        # index 0 is gap/padding; canonical code c uses the distinct index c+1.
        self.key_embedding = nn.Embedding(self.keycode_vocab + 1, key_embedding_dim, padding_idx=0)
        # Per pointer: 4 physical + key embedding + contact/down/move/up.
        # ``active_mask`` is deliberately excluded: it is an extractor-side
        # annotation (for example a callback/phase interval), not an Android
        # MotionEvent observation available equally for real and generated
        # trajectories.  Feeding it would create a label-construction oracle.
        per_pointer = N_POINTER_CONTINUOUS + key_embedding_dim + 4
        # Global: normalized log-dt, physical event-time progress, ordered
        # sequence progress, and explicit flight gap.  The separate sequence
        # progress keeps two zero-flight key boundary tokens distinguishable
        # even when their physical timestamps are equal.
        input_dim = POINTERS * per_pointer + 4
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, batch: DeepBatch) -> torch.Tensor:
        if torch.any(batch.keycode >= self.keycode_vocab):
            raise ValueError(
                "Deep PAD keycode outside configured [0,%d] vocabulary"
                % (self.keycode_vocab - 1)
            )
        key = torch.where(
            batch.keycode >= 0,
            batch.keycode + 1,
            torch.zeros_like(batch.keycode),
        )
        key_emb = self.key_embedding(key)  # [B,P,T,E]
        code = batch.action_code
        pointer_cat = torch.stack(
            (
                batch.contact_mask.float(),
                ((code == 0) | (code == 5)).float(),
                (code == 2).float(),
                ((code == 1) | (code == 6)).float(),
            ),
            dim=-1,
        )
        pointer = torch.cat((batch.pointer_continuous, key_emb, pointer_cat), dim=-1)
        b, p, t, c = pointer.shape
        pointer = pointer.permute(0, 2, 1, 3).reshape(b, t, p * c)
        b, t = batch.frame_mask.shape
        sequence_index = torch.arange(t, device=batch.frame_mask.device).expand(b, t)
        sequence_length = batch.frame_mask.sum(dim=1, keepdim=True)
        sequence_progress = sequence_index.to(batch.log_dt.dtype) / torch.clamp(
            sequence_length.to(batch.log_dt.dtype) - 1.0, min=1.0
        )
        global_values = torch.stack(
            (
                batch.log_dt,
                batch.time_progress,
                sequence_progress,
                batch.gap_mask.float(),
            ),
            dim=-1,
        )
        encoded = self.projection(torch.cat((pointer, global_values), dim=-1))
        # Multiplication is not a valid padding clear operation for a possible
        # non-finite invalid token: NaN * 0 remains NaN.  masked_fill only
        # replaces semantically invalid padding and leaves every valid token
        # available for the explicit finite-value gate in _masked_pool.
        return encoded.masked_fill(~batch.frame_mask.unsqueeze(-1), 0.0)


def _masked_pool(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = sequence[mask]
    if valid.numel() == 0:
        raise RuntimeError("raw Deep PAD batch has no valid timeline tokens")
    if not torch.isfinite(valid).all():
        raise FloatingPointError("non-finite Deep PAD representation on a valid timeline token")
    sequence = sequence.masked_fill(~mask.unsqueeze(-1), 0.0)
    weights = mask.unsqueeze(-1).to(sequence.dtype)
    mean = torch.sum(sequence * weights, dim=1) / torch.clamp(torch.sum(weights, dim=1), min=1.0)
    # A masked maximum has tie-sensitive backward routing and caused rare
    # cross-run divergence/non-finite gradients on CPU even with deterministic
    # algorithms enabled.  The RMS statistic is smooth, deterministic, and
    # retains an event-amplitude summary alongside the mean without inspecting
    # semantic padding.
    second_moment = torch.sum(sequence.square() * weights, dim=1) / torch.clamp(
        torch.sum(weights, dim=1), min=1.0
    )
    rms = torch.sqrt(torch.clamp(second_moment, min=0.0) + 1.0e-8)
    return torch.cat((mean, rms), dim=-1)


class _TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = int(dilation)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = x
        value = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        value = self.dropout(F.gelu(self.norm1(value)))
        # Conv1 creates non-zero activations in semantic padding next to the
        # final valid token.  If those activations enter conv2 they are folded
        # back into the valid boundary, so the same event receives a different
        # score solely because another batch member has a longer timeline.
        # Clear padding between convolutions, not only after the residual.
        value = value.masked_fill(~mask.unsqueeze(-1), 0.0)
        value = self.conv2(value.transpose(1, 2)).transpose(1, 2)
        value = self.dropout(F.gelu(self.norm2(value)))
        value = (value + residual).masked_fill(~mask.unsqueeze(-1), 0.0)
        return value


class RawTCNPAD(nn.Module):
    """Dilated raw temporal convolution PAD, not a feature MLP."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_blocks: int = 4,
        dropout: float = 0.15,
        keycode_vocab: int = KEYCODE_VOCAB_SIZE,
    ) -> None:
        super().__init__()
        self.frame_encoder = _FrameEncoder(hidden_dim, keycode_vocab=keycode_vocab)
        self.blocks = nn.ModuleList(
            [_TemporalResidualBlock(hidden_dim, 2 ** index, dropout) for index in range(n_blocks)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch: DeepBatch) -> torch.Tensor:
        value = self.frame_encoder(batch)
        for block in self.blocks:
            value = block(value, batch.frame_mask)
        return self.classifier(_masked_pool(value, batch.frame_mask)).squeeze(-1)


class RawTransformerPAD(nn.Module):
    """Masked self-attention raw-sequence PAD on the event-global timeline."""

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        feedforward_dim: int = 128,
        dropout: float = 0.15,
        keycode_vocab: int = KEYCODE_VOCAB_SIZE,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.frame_encoder = _FrameEncoder(hidden_dim, keycode_vocab=keycode_vocab)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, norm=nn.LayerNorm(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    def forward(self, batch: DeepBatch) -> torch.Tensor:
        value = self.frame_encoder(batch)
        value = self.encoder(value, src_key_padding_mask=~batch.frame_mask)
        value = value.masked_fill(~batch.frame_mask.unsqueeze(-1), 0.0)
        return self.classifier(_masked_pool(value, batch.frame_mask)).squeeze(-1)


def make_deep_model(detector_kind: str, model_params: Optional[Mapping[str, Any]] = None) -> nn.Module:
    params = dict(model_params or {})
    if detector_kind == "tcn":
        return RawTCNPAD(**params)
    if detector_kind == "transformer":
        return RawTransformerPAD(**params)
    raise ValueError("detector_kind must be one of %r" % (DEEP_DETECTORS,))


@dataclass
class DeepTrainConfig:
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    patience: int = 0
    num_workers: int = 0
    seed: int = 20260713
    bootstrap_replicates: int = 500
    gradient_clip_norm: float = 5.0

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0.0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")
        if self.patience < 0 or self.num_workers < 0 or self.bootstrap_replicates < 0:
            raise ValueError("patience/workers/bootstrap cannot be negative")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")


@dataclass
class DeepPADProtocolResult:
    action: str
    detector_kind: str
    model: nn.Module
    normalizer: RawSequenceNormalizer
    thresholds: Dict[str, float]
    validation_metrics: Dict[str, Dict[str, float]]
    test_metrics: Dict[str, Dict[str, float]]
    score_dumps: Dict[str, Dict[str, np.ndarray]]
    curves: Dict[str, Dict[str, np.ndarray]]
    bootstrap: Optional[Dict[str, Any]]
    history: List[Dict[str, float]]
    best_epoch: int
    last_epoch: int
    checkpoint_paths: Dict[str, str]
    run_identity: Dict[str, Any]
    run_identity_sha256: str


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    # Repeated formal runs with the same seed must not depend on cuDNN kernel
    # autotuning or TF32 choices.  warn_only=False is intentionally fail-closed
    # if a future model introduces a nondeterministic operation.
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _configure_deterministic_backend(device: torch.device) -> None:
    """Remove backend choices that may diverge despite deterministic mode."""

    if device.type == "cpu":
        # oneDNN convolution reductions have shown cross-run gradient routing
        # differences in the raw TCN on this host.  The formal CPU fallback is
        # correctness-first; GPU formal jobs are unaffected by these settings.
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = False
        torch.set_num_threads(1)
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)
    if device.type == "cuda" and hasattr(torch.backends, "cuda"):
        # Force the auditable math SDP kernel; flash/memory-efficient kernels
        # can choose hardware-dependent reduction orders across resumptions.
        for name, enabled in (
            ("enable_flash_sdp", False),
            ("enable_mem_efficient_sdp", False),
            ("enable_math_sdp", True),
        ):
            function = getattr(torch.backends.cuda, name, None)
            if function is not None:
                function(enabled)


def _seed_training_epoch(base_seed: int, epoch: int) -> int:
    """Reset every stochastic stream to an epoch-addressable seed.

    ``last.pt`` is committed only at epoch boundaries.  Addressing shuffle and
    dropout randomness by epoch makes epoch ``k`` identical whether reached in
    one process or after a power-loss resume; no hidden DataLoader generator
    cursor needs to be reconstructed.
    """

    payload = "%d|deep_pad_epoch|%d" % (int(base_seed), int(epoch))
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little")
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    return value


def _require_finite_model(model: nn.Module, context: str) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError("%s: non-finite model parameter %s" % (context, name))
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("%s: non-finite gradient %s" % (context, name))


def _require_finite_optimizer(optimizer: torch.optim.Optimizer, context: str) -> None:
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError("%s: non-finite optimizer state %s" % (context, name))


def _records_for_protocol(
    records: Sequence[RawTrajectoryRecord], action: str
) -> Dict[str, List[RawTrajectoryRecord]]:
    if action not in ACTIONS:
        raise ValueError("unsupported action: %s" % action)
    result: Dict[str, List[RawTrajectoryRecord]] = {pool: [] for pool in ALLOWED_POOLS}
    seen_ids = set()
    for record in records:
        record.validate()
        identity = (record.label, record.sample_id)
        if identity in seen_ids:
            raise ValueError("duplicate label/sample_id: %r" % (identity,))
        seen_ids.add(identity)
        if record.action == action:
            result[record.pool].append(record)
    for pool, rows in result.items():
        if not rows:
            raise ValueError("%s has no %s records" % (action, pool))
        if set(record.label for record in rows) != {REAL_LABEL, FAKE_LABEL}:
            raise ValueError("%s/%s must contain real=0 and fake=1" % (action, pool))
    # A sample may not cross pools even if a caller changed its label.
    sample_pools: Dict[str, str] = {}
    for pool, rows in result.items():
        for record in rows:
            previous = sample_pools.setdefault(record.sample_id, pool)
            if previous != pool:
                raise ValueError("sample_id crosses pools: %s" % record.sample_id)
    return result


def _make_loader(
    records: Sequence[RawTrajectoryRecord],
    normalizer: RawSequenceNormalizer,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        _RecordDataset(records),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        generator=generator,
        collate_fn=lambda rows: collate_raw_sequences(rows, normalizer),
        pin_memory=torch.cuda.is_available(),
    )


def _evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, Any]:
    model.eval()
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_users: List[np.ndarray] = []
    all_ids: List[str] = []
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            if not torch.isfinite(logits).all():
                bad = np.asarray(batch.sample_ids)[~torch.isfinite(logits).detach().cpu().numpy()]
                raise FloatingPointError(
                    "non-finite evaluation logit for sample IDs: %s" % bad[:8].tolist()
                )
            loss = F.binary_cross_entropy_with_logits(logits, batch.labels, reduction="sum")
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite evaluation loss")
            total_loss += float(loss.item())
            total_count += int(len(batch.labels))
            all_scores.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64))
            all_labels.append(batch.labels.detach().cpu().numpy().astype(np.int64))
            all_users.append(batch.user_ids.detach().cpu().numpy().astype(np.int64))
            all_ids.extend(batch.sample_ids)
    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    users = np.concatenate(all_users)
    return {
        "loss": total_loss / max(total_count, 1),
        "auc": float(roc_auc_score(labels, scores)),
        "score": scores,
        "label": labels,
        "user_id": users,
        "sample_id": np.asarray(all_ids, dtype="U128"),
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(str(temporary), str(path))


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _records_sha256(records: Sequence[RawTrajectoryRecord]) -> str:
    """Hash every value consumed by Deep PAD for standalone resume identity.

    Formal pair runs bind the immutable dataset-file SHA instead.  This
    fallback keeps the public in-memory API fail-closed as well: changing a
    label, pool, user, sample ID, timeline or trajectory tensor cannot resume
    an unrelated checkpoint merely because its hyperparameters match.
    """

    digest = hashlib.sha256()
    for record in records:
        metadata = (
            record.action, int(record.label), int(record.user_id), record.pool,
            record.sample_id, record.event_group_id or record.sample_id,
        )
        digest.update(json.dumps(metadata, separators=(",", ":")).encode("utf-8"))
        for value in (
            record.pointer_continuous, record.global_t_ms, record.contact_mask,
            record.active_mask, record.action_code, record.keycode,
            record.event_ids, record.gap_mask,
        ):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _deep_run_identity(
    records: Sequence[RawTrajectoryRecord],
    *,
    action: str,
    detector_kind: str,
    train_config: DeepTrainConfig,
    model_params: Optional[Mapping[str, Any]],
    input_identity: Optional[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    if input_identity is None:
        source: Dict[str, Any] = {
            "schema_version": "trajectory_deep_pad_in_memory_input_v1",
            "records_sha256": _records_sha256(records),
            "record_count": int(len(records)),
        }
    else:
        source = dict(input_identity)
        required = {
            "schema_version", "dataset_file", "dataset_sha256",
            "fake_user_split", "fake_user_split_sha256", "real_hash_seed",
            "action", "family", "detector", "pair_config",
            "pair_config_sha256",
        }
        if set(source) != required:
            raise ValueError(
                "Deep pair input identity fields mismatch: missing=%r extra=%r"
                % (sorted(required - set(source)), sorted(set(source) - required))
            )
        if source.get("schema_version") != "trajectory_deep_pad_pair_input_v1":
            raise ValueError("Deep pair input identity schema mismatch")
        if (
            source.get("action") != action
            or source.get("family") != "deep_pad"
            or source.get("detector") != detector_kind
        ):
            raise ValueError("Deep pair input identity action/family/detector mismatch")
        for name in ("dataset_sha256", "fake_user_split_sha256", "pair_config_sha256"):
            value = source.get(name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("Deep pair input identity has invalid %s" % name)
        for name in ("dataset_file", "fake_user_split"):
            if not Path(str(source.get(name, ""))).is_absolute():
                raise ValueError("Deep pair input identity requires absolute %s" % name)
        pair_config = source.get("pair_config")
        if not isinstance(pair_config, dict):
            raise ValueError("Deep pair input identity lacks canonical pair_config")
        if _canonical_json_sha256(pair_config) != source["pair_config_sha256"]:
            raise ValueError("Deep pair input identity pair_config digest mismatch")
        if (
            pair_config.get("action") != action
            or pair_config.get("family") != "deep_pad"
            or pair_config.get("detector") != detector_kind
            or int(pair_config.get("real_hash_seed", -1))
            != int(source["real_hash_seed"])
        ):
            raise ValueError("Deep pair input identity disagrees with pair_config")

    identity = {
        "schema_version": "trajectory_deep_pad_run_identity_v2",
        "action": action,
        "detector_kind": detector_kind,
        "model_params": dict(model_params or {}),
        "train_config": asdict(train_config),
        "selection_pool": "validation_only",
        "input_identity": source,
    }
    return identity, _canonical_json_sha256(identity)


def _assert_checkpoint_value_equal(expected: Any, observed: Any, field: str) -> None:
    """Recursively prove that a replayed checkpoint equals an immutable file.

    ``torch.save`` byte streams are not a stable semantic comparison (container
    metadata may differ), so recovery compares every tensor/array/scalar and
    nested key exactly.  The replayed epoch has already passed all finite
    gates; an orphan with any state or metadata drift is rejected fail-closed.
    """

    if torch.is_tensor(expected) or torch.is_tensor(observed):
        if not (torch.is_tensor(expected) and torch.is_tensor(observed)):
            raise ValueError("immutable best checkpoint type mismatch at %s" % field)
        if (
            expected.dtype != observed.dtype
            or tuple(expected.shape) != tuple(observed.shape)
            or not torch.equal(expected.detach().cpu(), observed.detach().cpu())
        ):
            raise ValueError("immutable best checkpoint tensor mismatch at %s" % field)
        return
    if isinstance(expected, np.ndarray) or isinstance(observed, np.ndarray):
        if not (isinstance(expected, np.ndarray) and isinstance(observed, np.ndarray)):
            raise ValueError("immutable best checkpoint type mismatch at %s" % field)
        if (
            expected.dtype != observed.dtype
            or expected.shape != observed.shape
            or not np.array_equal(expected, observed)
        ):
            raise ValueError("immutable best checkpoint array mismatch at %s" % field)
        return
    if isinstance(expected, Mapping) or isinstance(observed, Mapping):
        if not (isinstance(expected, Mapping) and isinstance(observed, Mapping)):
            raise ValueError("immutable best checkpoint type mismatch at %s" % field)
        if set(expected) != set(observed):
            raise ValueError("immutable best checkpoint keys mismatch at %s" % field)
        for key in sorted(expected, key=lambda value: repr(value)):
            _assert_checkpoint_value_equal(
                expected[key], observed[key], "%s.%s" % (field, key)
            )
        return
    if isinstance(expected, (list, tuple)) or isinstance(observed, (list, tuple)):
        if type(expected) is not type(observed) or len(expected) != len(observed):
            raise ValueError("immutable best checkpoint sequence mismatch at %s" % field)
        for index, (left, right) in enumerate(zip(expected, observed)):
            _assert_checkpoint_value_equal(left, right, "%s[%d]" % (field, index))
        return
    if type(expected) is not type(observed) or expected != observed:
        raise ValueError("immutable best checkpoint scalar mismatch at %s" % field)


def _best_checkpoint_epoch(path: Path) -> int:
    name = path.name
    prefix, suffix = "best_epoch_", ".pt"
    token = name[len(prefix) : -len(suffix)] if name.startswith(prefix) and name.endswith(suffix) else ""
    if len(token) != 4 or not token.isdigit() or int(token) < 1:
        raise ValueError("invalid immutable best checkpoint filename: %s" % path)
    return int(token)


def _commit_or_reuse_immutable_best(
    path: Path,
    payload: Mapping[str, Any],
    *,
    replay_orphan: Optional[Path],
    map_location: torch.device,
) -> bool:
    """Create a best checkpoint, or exactly verify one left before ``last``.

    Returns ``True`` only when a deterministic orphan was verified and reused.
    Any unrelated pre-existing immutable filename remains an overwrite error.
    """

    if not path.exists():
        _atomic_torch_save(path, payload)
        return False
    if replay_orphan is None or path.resolve() != replay_orphan.resolve():
        raise FileExistsError(
            "refusing to overwrite immutable best checkpoint: %s" % path
        )
    observed = torch.load(str(path), map_location=map_location)
    _assert_checkpoint_value_equal(dict(payload), observed, "checkpoint")
    return True


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError("not JSON serializable: %r" % type(value))


def run_deep_pad_protocol(
    records: Sequence[RawTrajectoryRecord],
    *,
    action: str,
    detector_kind: str,
    output_dir: Path,
    config: Optional[DeepTrainConfig] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    device: Optional[str] = None,
    resume: bool = False,
    run_identity: Optional[Mapping[str, Any]] = None,
) -> DeepPADProtocolResult:
    """Train one action/detector and execute the strict validation/test protocol."""

    if detector_kind not in DEEP_DETECTORS:
        raise ValueError("detector_kind must be one of %r" % (DEEP_DETECTORS,))
    train_config = config or DeepTrainConfig()
    train_config.validate()
    split = _records_for_protocol(records, action)
    canonical_run_identity, run_identity_sha256 = _deep_run_identity(
        records, action=action, detector_kind=detector_kind,
        train_config=train_config, model_params=model_params,
        input_identity=run_identity,
    )
    root = Path(output_dir)
    summary_path = root / "summary.json"
    last_path = root / "checkpoints" / "last.pt"
    if summary_path.exists() and not resume:
        raise FileExistsError("refusing to overwrite completed run: %s" % root)
    root.mkdir(parents=True, exist_ok=True)

    _seed_everything(train_config.seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _configure_deterministic_backend(selected_device)
    normalizer = RawSequenceNormalizer().fit(split["train"])
    model = make_deep_model(detector_kind, model_params).to(selected_device)
    _require_finite_model(model, "initialization")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay, foreach=False,
    )
    n_real = sum(record.label == REAL_LABEL for record in split["train"])
    n_fake = sum(record.label == FAKE_LABEL for record in split["train"])
    pos_weight = torch.tensor(float(n_real) / float(n_fake), device=selected_device)

    val_loader = _make_loader(
        split["val"], normalizer, train_config.batch_size, False,
        train_config.seed + 1, train_config.num_workers,
    )
    test_loader = _make_loader(
        split["test"], normalizer, train_config.batch_size, False,
        train_config.seed + 2, train_config.num_workers,
    )

    start_epoch = 1
    history: List[Dict[str, float]] = []
    best_auc = -math.inf
    best_loss = math.inf
    best_epoch = 0
    best_path: Optional[Path] = None
    epochs_without_improvement = 0
    replay_orphan_best: Optional[Path] = None
    if resume:
        checkpoint_dir = root / "checkpoints"
        immutable_best = sorted(checkpoint_dir.glob("best_epoch_*.pt"))
        immutable_epochs = [_best_checkpoint_epoch(path) for path in immutable_best]
        if len(set(immutable_epochs)) != len(immutable_epochs):
            raise ValueError("duplicate immutable best checkpoint epochs")
        if last_path.exists():
            checkpoint = torch.load(str(last_path), map_location=selected_device)
            if checkpoint.get("schema_version") != "trajectory_deep_pad_v2":
                raise ValueError("last checkpoint schema mismatch")
            if checkpoint["action"] != action or checkpoint["detector_kind"] != detector_kind:
                raise ValueError("last checkpoint action/detector mismatch")
            if dict(checkpoint["model_params"]) != dict(model_params or {}):
                raise ValueError("last checkpoint model_params mismatch")
            if dict(checkpoint["train_config"]) != asdict(train_config):
                raise ValueError("last checkpoint train_config mismatch; refusing unsafe resume")
            if checkpoint.get("selection_pool") != "validation_only":
                raise ValueError("last checkpoint selection policy mismatch")
            if (
                checkpoint.get("run_identity") != canonical_run_identity
                or checkpoint.get("run_identity_sha256") != run_identity_sha256
                or _canonical_json_sha256(checkpoint.get("run_identity", {}))
                != run_identity_sha256
            ):
                raise ValueError(
                    "last checkpoint run identity mismatch; refusing unsafe resume"
                )
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            normalizer = RawSequenceNormalizer.from_state_dict(checkpoint["normalizer"])
            history = list(checkpoint["history"])
            best_auc = float(checkpoint["best_auc"])
            best_loss = float(checkpoint["best_loss"])
            best_epoch = int(checkpoint["best_epoch"])
            best_path = Path(checkpoint["best_path"])
            epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
            completed_epoch = int(checkpoint["epoch"])
            if len(history) != completed_epoch or int(history[-1]["epoch"]) != completed_epoch:
                raise ValueError("last checkpoint history/epoch mismatch")
            start_epoch = completed_epoch + 1
            future_best = [
                path for path, epoch_value in zip(immutable_best, immutable_epochs)
                if epoch_value > completed_epoch
            ]
            if future_best:
                if (
                    len(future_best) != 1
                    or _best_checkpoint_epoch(future_best[0]) != start_epoch
                ):
                    raise ValueError(
                        "immutable best checkpoints beyond last.pt do not form one recoverable epoch"
                    )
                replay_orphan_best = future_best[0]
        else:
            # Epoch one is always a validation improvement over (-inf,+inf).
            # Therefore the only legitimate no-last state is an atomic epoch-1
            # best committed immediately before power loss.  Reinitialize from
            # the fixed seed, replay epoch one, and compare every payload value.
            expected = checkpoint_dir / "best_epoch_0001.pt"
            if immutable_best != [expected]:
                raise FileNotFoundError(
                    "resume without last.pt requires exactly one epoch-1 immutable best checkpoint"
                )
            orphan = torch.load(str(expected), map_location=selected_device)
            if (
                orphan.get("schema_version") != "trajectory_deep_pad_v2"
                or orphan.get("action") != action
                or orphan.get("detector_kind") != detector_kind
                or dict(orphan.get("model_params", {})) != dict(model_params or {})
                or dict(orphan.get("train_config", {})) != asdict(train_config)
                or int(orphan.get("epoch", -1)) != 1
                or int(orphan.get("best_epoch", -1)) != 1
                or Path(str(orphan.get("best_path", ""))).resolve() != expected.resolve()
                or orphan.get("selection_pool") != "validation_only"
                or orphan.get("run_identity") != canonical_run_identity
                or orphan.get("run_identity_sha256") != run_identity_sha256
                or _canonical_json_sha256(orphan.get("run_identity", {}))
                != run_identity_sha256
            ):
                raise ValueError("epoch-1 orphan checkpoint metadata mismatch")
            replay_orphan_best = expected

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, train_config.epochs + 1):
        epoch_seed = _seed_training_epoch(train_config.seed, epoch)
        train_loader = _make_loader(
            split["train"], normalizer, train_config.batch_size, True,
            epoch_seed, train_config.num_workers,
        )
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            batch = batch.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            if not torch.isfinite(logits).all():
                bad = np.asarray(batch.sample_ids)[~torch.isfinite(logits).detach().cpu().numpy()]
                raise FloatingPointError(
                    "non-finite training logit for sample IDs: %s" % bad[:8].tolist()
                )
            loss = F.binary_cross_entropy_with_logits(logits, batch.labels, pos_weight=pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            _require_finite_model(model, "after backward")
            total_norm = nn.utils.clip_grad_norm_(
                model.parameters(), train_config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            if not torch.isfinite(total_norm):
                raise FloatingPointError("non-finite gradient norm")
            optimizer.step()
            _require_finite_model(model, "after optimizer step")
            _require_finite_optimizer(optimizer, "after optimizer step")
            train_loss_sum += float(loss.item()) * len(batch.labels)
            train_count += int(len(batch.labels))
        validation = _evaluate_model(model, val_loader, selected_device)
        row = {
            "epoch": int(epoch),
            "train_loss": train_loss_sum / max(train_count, 1),
            "val_loss": float(validation["loss"]),
            "val_auc": float(validation["auc"]),
        }
        history.append(row)
        improved = (
            row["val_auc"] > best_auc + 1.0e-12
            or (abs(row["val_auc"] - best_auc) <= 1.0e-12 and row["val_loss"] < best_loss - 1.0e-12)
        )
        if improved:
            best_auc, best_loss, best_epoch = row["val_auc"], row["val_loss"], int(epoch)
            epochs_without_improvement = 0
            candidate = root / "checkpoints" / ("best_epoch_%04d.pt" % epoch)
            best_path = candidate
        else:
            epochs_without_improvement += 1

        if (
            replay_orphan_best is not None
            and _best_checkpoint_epoch(replay_orphan_best) == int(epoch)
            and not improved
        ):
            raise RuntimeError(
                "deterministic replay did not reproduce the orphan best-checkpoint decision"
            )

        payload = {
            "schema_version": "trajectory_deep_pad_v2",
            "action": action,
            "detector_kind": detector_kind,
            "model_params": dict(model_params or {}),
            "train_config": asdict(train_config),
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
            "history": history,
            "best_auc": float(best_auc),
            "best_loss": float(best_loss),
            "best_epoch": int(best_epoch),
            "best_path": str(best_path) if best_path is not None else "",
            "epochs_without_improvement": int(epochs_without_improvement),
            "selection_pool": "validation_only",
            "checkpoint_commit_policy": (
                "immutable_best_then_last_with_exact_epoch_replay_v3_source_bound"
            ),
            "train_epoch_seed_policy": "sha256(base_seed|deep_pad_epoch|epoch)_uint32",
            "epoch_seed": int(epoch_seed),
            "run_identity": canonical_run_identity,
            "run_identity_sha256": run_identity_sha256,
        }
        if improved:
            reused = _commit_or_reuse_immutable_best(
                best_path, payload,
                replay_orphan=replay_orphan_best,
                map_location=selected_device,
            )
            if reused:
                replay_orphan_best = None
        _atomic_torch_save(last_path, payload)
        last_epoch = int(epoch)
        if train_config.patience > 0 and epochs_without_improvement >= train_config.patience:
            break

    if best_path is None or not best_path.exists():
        raise RuntimeError("training produced no best checkpoint")
    best_checkpoint = torch.load(str(best_path), map_location=selected_device)
    if (
        best_checkpoint.get("schema_version") != "trajectory_deep_pad_v2"
        or best_checkpoint.get("run_identity") != canonical_run_identity
        or best_checkpoint.get("run_identity_sha256") != run_identity_sha256
        or _canonical_json_sha256(best_checkpoint.get("run_identity", {}))
        != run_identity_sha256
    ):
        raise ValueError("selected best checkpoint run identity mismatch")
    model.load_state_dict(best_checkpoint["model_state"])
    normalizer = RawSequenceNormalizer.from_state_dict(best_checkpoint["normalizer"])
    # Recreate loaders with the exact normalizer embedded in the selected checkpoint.
    val_loader = _make_loader(split["val"], normalizer, train_config.batch_size, False, train_config.seed + 1, train_config.num_workers)
    test_loader = _make_loader(split["test"], normalizer, train_config.batch_size, False, train_config.seed + 2, train_config.num_workers)
    validation = _evaluate_model(model, val_loader, selected_device)
    test = _evaluate_model(model, test_loader, selected_device)

    thresholds = select_validation_thresholds(validation["label"], validation["score"], target_frr=0.05)
    operating_thresholds = {
        "eer": thresholds["eer"],
        "val_frr_le_5pct": thresholds["val_frr_le_5pct"],
    }
    validation_metrics = {
        name: operating_metrics(validation["label"], validation["score"], threshold)
        for name, threshold in operating_thresholds.items()
    }
    test_metrics = {
        name: operating_metrics(test["label"], test["score"], threshold)
        for name, threshold in operating_thresholds.items()
    }
    score_dumps: Dict[str, Dict[str, np.ndarray]] = {}
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for pool, values in (("val", validation), ("test", test)):
        score_dumps[pool] = {
            "score": values["score"], "label": values["label"],
            "user_id": values["user_id"], "sample_id": values["sample_id"],
            "pool": np.full(len(values["label"]), pool, dtype="U5"),
            "action": np.full(len(values["label"]), action, dtype="U16"),
        }
        curves[pool] = fa_frr_curve(values["label"], values["score"])

    bootstrap = None
    if train_config.bootstrap_replicates > 0:
        bootstrap = user_level_bootstrap(
            test["label"], test["score"], test["user_id"], operating_thresholds,
            n_replicates=train_config.bootstrap_replicates,
            seed=train_config.seed + 17,
        )
    result = DeepPADProtocolResult(
        action=action, detector_kind=detector_kind, model=model, normalizer=normalizer,
        thresholds=thresholds, validation_metrics=validation_metrics, test_metrics=test_metrics,
        score_dumps=score_dumps, curves=curves, bootstrap=bootstrap, history=history,
        best_epoch=int(best_epoch), last_epoch=int(last_epoch),
        checkpoint_paths={"best": str(best_path), "last": str(last_path)},
        run_identity=canonical_run_identity,
        run_identity_sha256=run_identity_sha256,
    )
    save_deep_protocol_outputs(result, root, train_config, model_params)
    return result


def save_deep_protocol_outputs(
    result: DeepPADProtocolResult,
    output_dir: Path,
    config: DeepTrainConfig,
    model_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    root = Path(output_dir)
    effective_keycode_vocab = int(
        (model_params or {}).get("keycode_vocab", KEYCODE_VOCAB_SIZE)
    )
    summary = {
        "schema_version": "trajectory_deep_pad_result_v2",
        "action": result.action,
        "detector_kind": result.detector_kind,
        "detector_family": "raw_sequence_deep_pad",
        "score_direction": "fake_high",
        "acceptance_rule": "score < threshold",
        "normalization_fit_pool": "train_only",
        "checkpoint_selection_pool": "validation_only",
        "threshold_selection_pool": "validation_only",
        "test_role": "fixed_threshold_evaluation_and_curve_only",
        "train_epoch_seed_policy": "sha256(base_seed|deep_pad_epoch|epoch)_uint32",
        "uses_critic": False,
        "uses_selector": False,
        "global_timeline": True,
        "global_time_policy": (
            "nondecreasing; equality only for directly adjacent ordered "
            "keystroke event IDs with zero flight"
        ),
        "positive_flight_gap_token": True,
        "zero_flight_gap_token": False,
        "ordered_sequence_progress_channel": True,
        "independent_pointer_time_warp": False,
        "keycode_embedding_policy": {
            "gap_or_absent_negative_index": 0,
            "shared_vocabulary_size": effective_keycode_vocab,
            "canonical_range": [0, effective_keycode_vocab - 1],
            "canonical_nonnegative_index": "keycode+1",
            "out_of_range_policy": "fail_closed",
            "rare_hmog_8230_index": 8231,
            "real_fake_shared_canonical_input": True,
        },
        "train_config": asdict(config),
        "model_params": dict(model_params or {}),
        "thresholds": result.thresholds,
        "validation_metrics": result.validation_metrics,
        "test_metrics": result.test_metrics,
        "best_epoch": result.best_epoch,
        "last_epoch": result.last_epoch,
        "checkpoint_paths": result.checkpoint_paths,
        "run_identity": result.run_identity,
        "run_identity_sha256": result.run_identity_sha256,
        "normalizer": result.normalizer.state_dict(),
    }
    summary_path = root / "summary.json"
    scores_path = root / "score_dump.npz"
    curves_path = root / "curves.npz"
    history_path = root / "history.csv"
    _atomic_json(summary_path, summary)
    _atomic_npz(scores_path, {
        "%s_%s" % (pool, key): value
        for pool, dump in result.score_dumps.items() for key, value in dump.items()
    })
    _atomic_npz(curves_path, {
        "%s_%s" % (pool, key): value
        for pool, curve in result.curves.items() for key, value in curve.items()
    })
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = history_path.with_name(history_path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "val_loss", "val_auc"))
        writer.writeheader()
        writer.writerows(result.history)
    os.replace(str(temporary), str(history_path))
    paths = {
        "summary": str(summary_path), "scores": str(scores_path),
        "curves": str(curves_path), "history": str(history_path),
        **result.checkpoint_paths,
    }
    if result.bootstrap is not None:
        bootstrap_summary = root / "bootstrap_summary.json"
        bootstrap_values = root / "bootstrap_replicates.npz"
        _atomic_json(bootstrap_summary, {key: value for key, value in result.bootstrap.items() if key != "replicates"})
        _atomic_npz(bootstrap_values, result.bootstrap["replicates"])
        paths["bootstrap_summary"] = str(bootstrap_summary)
        paths["bootstrap_replicates"] = str(bootstrap_values)
    return paths


def save_raw_sequence_bundle(
    path: Path,
    records: Sequence[RawTrajectoryRecord],
    feature_vectors: np.ndarray,
) -> None:
    """Persist one action's variable sequences as numeric flat+offset NPZ."""

    if not records:
        raise ValueError("bundle needs records")
    for record in records:
        record.validate()
    action = records[0].action
    if any(record.action != action for record in records):
        raise ValueError("one bundle must contain one action")
    features = np.asarray(feature_vectors, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(records) or not np.all(np.isfinite(features)):
        raise ValueError("feature_vectors must be finite [N,D]")
    offsets = np.zeros((len(records) + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum([len(record.global_t_ms) for record in records])
    arrays = {
        "sequence_offsets": offsets,
        "flat_pointer_continuous": np.concatenate(
            [record.pointer_continuous.transpose(1, 0, 2) for record in records], axis=0
        ).astype(np.float32),
        "flat_global_t_ms": np.concatenate([record.global_t_ms for record in records]).astype(np.float32),
        "flat_contact_mask": np.concatenate([record.contact_mask.T for record in records], axis=0).astype(np.uint8),
        "flat_active_mask": np.concatenate([record.active_mask.T for record in records], axis=0).astype(np.uint8),
        "flat_action_code": np.concatenate([record.action_code.T for record in records], axis=0).astype(np.int16),
        "flat_keycode": np.concatenate([record.keycode.T for record in records], axis=0).astype(np.int32),
        "flat_event_ids": np.concatenate([record.event_ids.T for record in records], axis=0).astype(np.int32),
        "flat_gap_mask": np.concatenate([record.gap_mask for record in records]).astype(np.uint8),
        "feature_vectors": features,
        "label": np.asarray([record.label for record in records], dtype=np.int8),
        "user_id": np.asarray([record.user_id for record in records], dtype=np.int32),
        "pool": np.asarray([record.pool for record in records], dtype="U5"),
        "action": np.asarray([record.action for record in records], dtype="U16"),
        "sample_id": np.asarray([record.sample_id for record in records], dtype="U128"),
        "event_group_id": np.asarray(
            [record.event_group_id or record.sample_id for record in records], dtype="U128"
        ),
        "schema_version": np.asarray("trajectory_pad_bundle_v2"),
        "feature_schema_version": np.asarray(TRAJECTORY_FEATURE_SCHEMA_VERSION),
    }
    _atomic_npz(Path(path), arrays)


def load_raw_sequence_bundle(path: Path) -> Tuple[List[RawTrajectoryRecord], np.ndarray]:
    """Load the no-pickle numeric bundle consumed by the complete runner."""

    with np.load(Path(path), allow_pickle=False) as data:
        if str(data["schema_version"].item()) != "trajectory_pad_bundle_v2":
            raise ValueError("unsupported bundle schema")
        if (
            str(data["feature_schema_version"].item())
            != TRAJECTORY_FEATURE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported trajectory feature schema")
        offsets = np.asarray(data["sequence_offsets"], dtype=np.int64)
        n = len(offsets) - 1
        if n <= 0 or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise ValueError("invalid sequence_offsets")
        total = int(offsets[-1])
        flat_names = (
            "flat_pointer_continuous", "flat_global_t_ms", "flat_contact_mask",
            "flat_active_mask", "flat_action_code", "flat_keycode",
            "flat_event_ids", "flat_gap_mask",
        )
        if any(len(data[name]) != total for name in flat_names):
            raise ValueError("flat bundle arrays disagree with offsets")
        for name in ("label", "user_id", "pool", "action", "sample_id", "event_group_id"):
            if len(data[name]) != n:
                raise ValueError("event metadata length mismatch: %s" % name)
        records: List[RawTrajectoryRecord] = []
        for i in range(n):
            left, right = int(offsets[i]), int(offsets[i + 1])
            records.append(make_record(
                action=str(data["action"][i]), label=int(data["label"][i]),
                user_id=int(data["user_id"][i]), pool=str(data["pool"][i]),
                sample_id=str(data["sample_id"][i]),
                pointer_continuous=np.asarray(data["flat_pointer_continuous"][left:right]).transpose(1, 0, 2),
                global_t_ms=np.asarray(data["flat_global_t_ms"][left:right]),
                contact_mask=np.asarray(data["flat_contact_mask"][left:right], dtype=bool).T,
                active_mask=np.asarray(data["flat_active_mask"][left:right], dtype=bool).T,
                action_code=np.asarray(data["flat_action_code"][left:right]).T,
                keycode=np.asarray(data["flat_keycode"][left:right]).T,
                event_ids=np.asarray(data["flat_event_ids"][left:right]).T,
                gap_mask=np.asarray(data["flat_gap_mask"][left:right], dtype=bool),
                event_group_id=str(data["event_group_id"][i]),
            ))
        feature_vectors = np.asarray(data["feature_vectors"], dtype=np.float64)
    if feature_vectors.ndim != 2 or len(feature_vectors) != len(records) or not np.all(np.isfinite(feature_vectors)):
        raise ValueError("invalid feature_vectors in bundle")
    return records, feature_vectors


__all__ = [
    "ACTIONS", "DEEP_DETECTORS", "RawTrajectoryRecord", "RawSequenceNormalizer",
    "DeepBatch", "DeepTrainConfig", "DeepPADProtocolResult", "RawTCNPAD",
    "RawTransformerPAD", "make_record", "load_fake_user_split",
    "assign_strict_protocol_pools", "collate_raw_sequences", "make_deep_model",
    "run_deep_pad_protocol", "save_deep_protocol_outputs",
    "save_raw_sequence_bundle", "load_raw_sequence_bundle",
    "deep_keycode_embedding_index",
]
