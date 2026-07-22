"""Formal split, work-unit, reference and condition provenance rules.

This module intentionally separates condition construction from the corpus.
``ReferenceConditionPolicy`` receives exactly five references and an optional
immutable train-only prior; it has no method that can query validation/test
non-reference examples.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from trajectory.data import (
    ACTIONS,
    FORMAL_REF_COUNT,
    KEYCODE_TOKEN_MAX,
    ORIENTATION_IDS,
    CanonicalTrajectory,
    keystroke_zero_flight_flags,
)
from trajectory.features import is_hmog_ascii_letter_keycode
from training.corpus import canonical_sample_sha256


ACTION_TO_ID = {name: index for index, name in enumerate(ACTIONS)}
ID_TO_ACTION = {value: key for key, value in ACTION_TO_ID.items()}
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
ID_TO_SPLIT = {value: key for key, value in SPLIT_TO_ID.items()}
FORMAL_USERS = 100
FORMAL_SAMPLES_PER_USER_ACTION = 200
FORMAL_TOTAL = FORMAL_USERS * len(ACTIONS) * FORMAL_SAMPLES_PER_USER_ACTION
FORMAL_DDIM_STEPS = 50
# This is the protocol seed for condition requests and, transitively, DDIM
# noise.  It is deliberately independent of the training/reference-registry
# seed (42).  Formal generation must never silently resume an archive created
# with a different value.
FORMAL_GENERATION_BASE_SEED = 20260713
CONDITION_REQUEST_DIGEST_SCHEMA = "trajectory_condition_request_canonical_v1"
CONDITION_SET_DIGEST_SCHEMA = "trajectory_condition_request_set_v1"


class _LazyCanonicalRecords(Sequence[CanonicalTrajectory]):
    """Recreate canonical rows on demand without retaining a full corpus.

    ``NumericTrajectoryCorpus`` already owns the compact numeric archive.
    Formal prior fitting needs several deterministic passes for exact
    preallocation, but it does not need tens of thousands of simultaneously
    live ``CanonicalTrajectory`` objects.  This private sequence makes each
    pass streaming and releases a row before advancing to the next one.
    """

    def __init__(self, corpus, indices: np.ndarray):
        self.corpus = corpus
        self.indices = np.asarray(indices, np.int64).reshape(-1)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> CanonicalTrajectory:
        value = int(index)
        if value < 0:
            value += len(self)
        if value < 0 or value >= len(self):
            raise IndexError(index)
        return self.corpus.canonical_sample(int(self.indices[value]))


def _mean_contiguous_groups_bitwise(
    values: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Vectorize per-slice float32 means without changing reduction order."""

    array = np.asarray(values, np.float32)
    boundaries = np.asarray(offsets, np.int64).reshape(-1)
    if (
        array.ndim != 2
        or boundaries.size < 2
        or boundaries[0] != 0
        or boundaries[-1] != array.shape[0]
        or np.any(np.diff(boundaries) <= 0)
    ):
        raise ValueError("invalid contiguous-group mean inputs")
    lengths = np.diff(boundaries)
    result = np.empty((lengths.size, array.shape[1]), np.float32)
    for length in np.unique(lengths).tolist():
        rows = np.flatnonzero(lengths == int(length))
        indices = boundaries[rows, None] + np.arange(int(length), dtype=np.int64)[None, :]
        # Each gathered row keeps the exact source element order.  NumPy's
        # mean over axis 1 is therefore bitwise identical to calling
        # mean(axis=0) on every original contiguous slice.
        result[rows] = np.mean(array[indices], axis=1)
    return result


def stable_seed(base_seed: int, action: str, user_id: int, sample_index: int) -> int:
    """Platform-independent 63-bit seed, stable across shard layouts/resume."""
    payload = "%d|%s|%d|%d" % (int(base_seed), action, int(user_id), int(sample_index))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


DDIM_NOISE_SEED_XOR_DOMAIN = 0xDD1A50


def ddim_noise_seed(
    condition_request_seed: int, action: str, user_id: int, sample_index: int
) -> int:
    """Domain-separated seed for the initial Gaussian DDIM noise."""

    return stable_seed(
        int(condition_request_seed) ^ DDIM_NOISE_SEED_XOR_DOMAIN,
        action,
        user_id,
        sample_index,
    )


def numeric_sample_id(value: str) -> int:
    """Require the formal source event id to be losslessly numeric."""
    text = str(value)
    try:
        result = int(text)
    except ValueError as exc:
        raise ValueError("formal reference sample_id must be numeric: %r" % text) from exc
    if str(result) != text and text.lstrip("+").lstrip("0") != str(result):
        raise ValueError("sample_id is not losslessly numeric: %r" % text)
    if result < 0 or result > np.iinfo(np.int64).max:
        raise ValueError("sample_id outside non-negative int64 range")
    return result


@dataclass(frozen=True)
class FixedUserSplit:
    train_users: Tuple[int, ...]
    val_users: Tuple[int, ...]
    test_users: Tuple[int, ...]
    source_path: str
    source_sha256: str

    @classmethod
    def load(cls, path: str, require_formal: bool = True) -> "FixedUserSplit":
        source = Path(path)
        raw_bytes = source.read_bytes()
        data = json.loads(raw_bytes.decode("utf-8"))
        values = {
            "train": tuple(int(x) for x in data["train_users"]),
            "val": tuple(int(x) for x in data["val_users"]),
            "test": tuple(int(x) for x in data["test_users"]),
        }
        flat = values["train"] + values["val"] + values["test"]
        if len(flat) != len(set(flat)):
            raise ValueError("train/val/test user splits overlap")
        if any(x < 0 for x in flat):
            raise ValueError("user ids must be non-negative")
        if require_formal:
            expected_counts = {"train": 70, "val": 10, "test": 20}
            for split, expected in expected_counts.items():
                if len(values[split]) != expected:
                    raise ValueError("formal %s split must contain %d users" % (split, expected))
            if set(flat) != set(range(FORMAL_USERS)):
                raise ValueError("formal split must cover user ids 0..99 exactly")
        return cls(
            train_users=values["train"], val_users=values["val"], test_users=values["test"],
            source_path=str(source.resolve()), source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )

    def split_for_user(self, user_id: int) -> str:
        user_id = int(user_id)
        for split in ("train", "val", "test"):
            if user_id in getattr(self, split + "_users"):
                return split
        raise KeyError("user %d is absent from the fixed split" % user_id)

    @property
    def all_users(self) -> Tuple[int, ...]:
        return self.train_users + self.val_users + self.test_users


@dataclass(frozen=True)
class ReferenceRegistry:
    """Training-created immutable five-reference enrollment registry."""
    entries: Mapping[Tuple[str, int, str], Tuple[int, ...]]
    split_sha256: str
    registry_sha256: str
    source_path: str

    @staticmethod
    def _payload(entries: Mapping[Tuple[str, int, str], Tuple[int, ...]], split_sha256: str) -> Dict[str, object]:
        rows = []
        for (action, user_id, split), ids in sorted(
            entries.items(), key=lambda x: (
                ACTION_TO_ID[x[0][0]], SPLIT_TO_ID[x[0][2]], x[0][1]
            )
        ):
            rows.append({
                "action": action, "user_id": int(user_id), "split": split,
                "reference_event_ids": [int(x) for x in ids],
            })
        return {
            "schema_version": 1,
            "producer": "trajectory_training_pipeline",
            "split_sha256": split_sha256,
            "entries": rows,
        }

    @classmethod
    def build(
        cls, entries: Mapping[Tuple[str, int, str], Sequence[int]], split_sha256: str,
        source_path: str = "<memory>",
    ) -> "ReferenceRegistry":
        normalized: Dict[Tuple[str, int, str], Tuple[int, ...]] = {}
        for key, values in entries.items():
            action, user_id, split = key
            ids = tuple(int(x) for x in values)
            if action not in ACTIONS or split not in SPLIT_TO_ID or len(ids) != FORMAL_REF_COUNT or len(set(ids)) != FORMAL_REF_COUNT:
                raise ValueError("invalid reference registry entry %r" % (key,))
            normalized[(action, int(user_id), split)] = ids
        payload = cls._payload(normalized, split_sha256)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return cls(entries=normalized, split_sha256=split_sha256, registry_sha256=digest, source_path=source_path)

    @classmethod
    def load(cls, path: str, expected_split_sha256: str) -> "ReferenceRegistry":
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        if (
            int(data.get("schema_version", -1)) != 1
            or data.get("producer") != "trajectory_training_pipeline"
            or data.get("split_sha256") != expected_split_sha256
        ):
            raise ValueError("reference registry schema/split hash mismatch")
        entries = {}
        for row in data.get("entries", []):
            key = (str(row["action"]), int(row["user_id"]), str(row["split"]))
            if key in entries:
                raise ValueError("duplicate reference registry key")
            entries[key] = tuple(int(x) for x in row["reference_event_ids"])
        result = cls.build(entries, expected_split_sha256, source_path=str(source.resolve()))
        if data.get("registry_sha256") != result.registry_sha256:
            raise ValueError("reference registry content hash mismatch")
        return result

    def write(self, path: str) -> None:
        target = Path(path)
        if target.exists():
            raise FileExistsError(str(target))
        payload = self._payload(self.entries, self.split_sha256)
        payload["registry_sha256"] = self.registry_sha256
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)

    def resolve(
        self, pool: Sequence[CanonicalTrajectory], action: str, user_id: int, split: str,
    ) -> Tuple[CanonicalTrajectory, ...]:
        key = (action, int(user_id), split)
        if key not in self.entries:
            raise KeyError("reference registry missing %r" % (key,))
        ids = self.entries[key]
        index = {numeric_sample_id(x.sample_id): x for x in pool if x.action == action and x.user_id == user_id and x.split == split}
        if len(index) != len([x for x in pool if x.action == action and x.user_id == user_id and x.split == split]):
            raise ValueError("duplicate source ids while resolving registry")
        try:
            refs = tuple(index[x] for x in ids)
        except KeyError as exc:
            raise ValueError("registry references missing from real corpus") from exc
        if len(refs) != FORMAL_REF_COUNT or any(not x.is_real for x in refs):
            raise ValueError("registry did not resolve to five real refs")
        return refs


@dataclass(frozen=True)
class GenerationUnit:
    action: str
    user_id: int
    split: str
    samples: int
    shard_id: int
    num_shards: int

    @property
    def key(self) -> str:
        return "%s/user_%03d" % (self.action, self.user_id)


def build_generation_units(
    split: FixedUserSplit,
    samples_per_user_action: int = FORMAL_SAMPLES_PER_USER_ACTION,
    actions: Sequence[str] = ACTIONS,
    num_shards: int = 1,
    shard_id: Optional[int] = None,
    require_formal: bool = True,
) -> List[GenerationUnit]:
    if samples_per_user_action <= 0:
        raise ValueError("samples_per_user_action must be positive")
    if require_formal and samples_per_user_action != FORMAL_SAMPLES_PER_USER_ACTION:
        raise ValueError("formal protocol fixes 200 fake per user/action")
    if num_shards <= 0 or (shard_id is not None and not 0 <= shard_id < num_shards):
        raise ValueError("invalid shard configuration")
    action_values = tuple(actions)
    if not action_values or len(set(action_values)) != len(action_values) or any(x not in ACTIONS for x in action_values):
        raise ValueError("actions must be a unique non-empty subset of the five formal actions")
    if require_formal and set(action_values) != set(ACTIONS):
        raise ValueError("formal generation must include all five actions")
    units: List[GenerationUnit] = []
    for action in ACTIONS:
        if action not in action_values:
            continue
        for user_id in sorted(split.all_users):
            assignment = (ACTION_TO_ID[action] * FORMAL_USERS + user_id) % num_shards
            if shard_id is None or assignment == shard_id:
                units.append(
                    GenerationUnit(
                        action=action, user_id=user_id, split=split.split_for_user(user_id),
                        samples=samples_per_user_action, shard_id=assignment, num_shards=num_shards,
                    )
                )
    if require_formal and shard_id is None:
        if len(units) != FORMAL_USERS * len(ACTIONS) or sum(x.samples for x in units) != FORMAL_TOTAL:
            raise AssertionError("formal work plan must contain exactly 500 units / 100,000 samples")
    return units


@dataclass(frozen=True)
class TrainGlobalPrior:
    """Immutable shrinkage prior fitted exclusively on fixed train users.

    Arrays retain the joint train-only rows instead of independent histograms,
    allowing the policy to draw coherent residuals without ever looking up a
    validation/test non-reference event.
    """
    action: str
    train_user_ids: Tuple[int, ...]
    orientation_id: np.ndarray
    duration_ms: np.ndarray
    point_rate_hz: np.ndarray
    start_xy: np.ndarray
    end_xy: np.ndarray
    pointer_start_fraction: np.ndarray
    pointer_end_fraction: np.ndarray
    pinch_span: np.ndarray
    pinch_angle: np.ndarray
    key_duration_per_key_ms: np.ndarray
    key_contact_points: np.ndarray
    key_n_keys: np.ndarray
    key_n_letters: np.ndarray
    key_zero_flight: np.ndarray
    keycode_offsets: np.ndarray
    keycodes: np.ndarray
    key_position_token: np.ndarray
    key_position_orientation: np.ndarray
    key_position_down_xy: np.ndarray
    key_position_up_xy: np.ndarray
    key_position_center_xy: np.ndarray
    screen_min_xy_by_orientation: np.ndarray  # [4,2], ORIENTATION_IDS order
    screen_max_xy_by_orientation: np.ndarray
    screen_orientation_observed: np.ndarray   # [4]
    source_event_ids: np.ndarray
    source_user_ids: np.ndarray
    digest: str

    @classmethod
    def _fit_numeric_keystroke(cls, corpus, allowed: set) -> "TrainGlobalPrior":
        """Fit the exact keystroke prior directly from compact source arrays.

        The numeric archive is already the authoritative, independently
        audited source.  Reconstructing every train event as a Python object
        merely to aggregate the same rows is both slower and, at formal scale,
        memory unsafe.  This path reproduces ``raw_sample``'s model-row rule:
        within each key and integer timestamp, keep the last row of the
        earliest frame, then retain one row per distinct timestamp.
        """

        arrays = corpus._arrays
        event_selected = np.isin(
            arrays["user_id"], np.asarray(sorted(allowed), arrays["user_id"].dtype)
        )
        event_indices = np.flatnonzero(event_selected).astype(np.int64)
        if event_indices.size == 0:
            raise ValueError("cannot fit empty train-only prior for keystroke")
        if any(corpus.splits.split_for_user(int(user)) != "train" for user in allowed):
            raise ValueError("train prior user set contains non-train users")

        event_key_offsets_global = arrays["event_key_offsets"].astype(np.int64, copy=False)
        global_event_key_counts = np.diff(event_key_offsets_global)
        global_key_event = np.repeat(
            np.arange(len(corpus), dtype=np.int32), global_event_key_counts
        )
        key_selected = event_selected[global_key_event]
        selected_key_indices = np.flatnonzero(key_selected).astype(np.int64)
        selected_key_counts = global_event_key_counts[event_indices].astype(np.int64)
        keycode_offsets = np.zeros(event_indices.size + 1, np.int64)
        keycode_offsets[1:] = np.cumsum(selected_key_counts)
        if int(keycode_offsets[-1]) != int(selected_key_indices.size):
            raise AssertionError("selected numeric key offsets are inconsistent")

        # Restore the exact model-row representative for every key/timestamp.
        key_touch_offsets = arrays["key_touch_offsets"].astype(np.int64, copy=False)
        global_row_key = np.repeat(
            np.arange(selected_key_indices.max() + 1 if selected_key_indices.size else 0, dtype=np.int32),
            np.diff(key_touch_offsets)[: selected_key_indices.max() + 1],
        )
        # Formal key offsets span all keys.  Recreate the complete row mapping
        # when validation/test keys follow the largest selected train key.
        if global_row_key.size != arrays["flat_t_rel_ms"].size:
            global_row_key = np.repeat(
                np.arange(arrays["keycode"].size, dtype=np.int32),
                np.diff(key_touch_offsets),
            )
        flat_time = arrays["flat_t_rel_ms"]
        flat_frame = arrays["flat_frame_index"]
        group_start_mask = np.ones(flat_time.size, np.bool_)
        group_start_mask[1:] = (
            (global_row_key[1:] != global_row_key[:-1])
            | (flat_time[1:] != flat_time[:-1])
        )
        group_starts = np.flatnonzero(group_start_mask).astype(np.int64)
        group_key = global_row_key[group_starts]
        minimum_frame = np.minimum.reduceat(flat_frame, group_starts)
        group_id = np.cumsum(group_start_mask, dtype=np.int64) - 1
        row_number = np.arange(flat_time.size, dtype=np.int64)
        candidates = np.where(flat_frame == minimum_frame[group_id], row_number, -1)
        representative = np.maximum.reduceat(candidates, group_starts)
        if np.any(representative < 0):
            raise AssertionError("numeric model-row grouping lost a timestamp")
        keep_group = key_selected[group_key]
        representative = representative[keep_group]
        representative_key = group_key[keep_group].astype(np.int64, copy=False)
        del group_id, candidates, row_number, minimum_frame, group_start_mask

        global_to_local_key = np.full(arrays["keycode"].size, -1, np.int32)
        global_to_local_key[selected_key_indices] = np.arange(
            selected_key_indices.size, dtype=np.int32
        )
        representative_local_key = global_to_local_key[representative_key]
        if np.any(representative_local_key < 0):
            raise AssertionError("numeric representative escaped selected train keys")
        key_points_int64 = np.bincount(
            representative_local_key, minlength=selected_key_indices.size
        ).astype(np.int64)
        if np.any(key_points_int64 < 2):
            raise ValueError("train-only keystroke contact lacks DOWN/UP geometry")
        point_offsets = np.zeros(selected_key_indices.size + 1, np.int64)
        point_offsets[1:] = np.cumsum(key_points_int64)
        if int(point_offsets[-1]) != int(representative.size):
            raise AssertionError("numeric representative/key point counts differ")

        raw_xy = np.stack([
            arrays["flat_x"][representative], arrays["flat_y"][representative]
        ], axis=-1).astype(np.float32, copy=False)
        key_down_raw = raw_xy[point_offsets[:-1]]
        key_up_raw = raw_xy[point_offsets[1:] - 1]
        event_start_raw = key_down_raw[keycode_offsets[:-1]]
        event_end_raw = key_up_raw[keycode_offsets[1:] - 1]
        local_key_event = np.repeat(
            np.arange(event_indices.size, dtype=np.int32), selected_key_counts
        )
        point_event = local_key_event[representative_local_key]

        # Match trajectory.data._canonical_pointer encode/decode float32
        # arithmetic.  The original prior intentionally uses decoded canonical
        # coordinates rather than bypassing the model representation.
        chord = (event_end_raw - event_start_raw).astype(np.float32)
        chord_length = np.linalg.norm(chord, axis=1).astype(np.float32)
        scale = np.maximum(chord_length, np.float32(1.0)).astype(np.float32)
        unit = np.zeros_like(chord)
        nonzero = chord_length > np.float32(1.0e-6)
        unit[nonzero] = chord[nonzero] / chord_length[nonzero, None]
        unit[~nonzero, 0] = 1.0
        normal = np.stack([-unit[:, 1], unit[:, 0]], axis=-1).astype(np.float32)
        local_event_start = event_start_raw[point_event]
        delta = (raw_xy - local_event_start).astype(np.float32)
        local_unit = unit[point_event]
        local_normal = normal[point_event]
        local_scale = scale[point_event]
        progress = (
            np.sum(delta * local_unit, axis=1) / local_scale
        ).astype(np.float32)
        lateral = (
            np.sum(delta * local_normal, axis=1) / local_scale
        ).astype(np.float32)
        decoded_xy = (
            local_event_start
            + progress[:, None] * local_scale[:, None] * local_unit
            + lateral[:, None] * local_scale[:, None] * local_normal
        ).astype(np.float32)

        key_position_down_array = decoded_xy[point_offsets[:-1]].copy()
        key_position_up_array = decoded_xy[point_offsets[1:] - 1].copy()
        key_position_center_array = _mean_contiguous_groups_bitwise(
            decoded_xy, point_offsets
        )

        duration_array = arrays["touch_duration_ms"][event_indices].astype(np.float32)
        positive_flight_per_key = arrays["key_flight_from_previous_ms"] > 0
        is_first_global_key = np.zeros(arrays["keycode"].size, np.bool_)
        is_first_global_key[event_key_offsets_global[:-1]] = True
        selected_transition_keys = selected_key_indices[~is_first_global_key[selected_key_indices]]
        key_zero_flight_array = (
            arrays["key_flight_from_previous_ms"][selected_transition_keys] == 0
        ).astype(np.uint8)
        event_contact_points = np.add.reduceat(
            key_points_int64, keycode_offsets[:-1]
        )
        event_positive_flights = np.add.reduceat(
            positive_flight_per_key[selected_key_indices].astype(np.int64),
            keycode_offsets[:-1],
        )
        # The first key's conventionally-zero flight is never a gap.
        event_positive_flights -= positive_flight_per_key[
            selected_key_indices[keycode_offsets[:-1]]
        ].astype(np.int64)
        event_point_count = event_contact_points + event_positive_flights
        rate_array = np.zeros((event_indices.size, 2), np.float32)
        rate_array[:, 0] = (
            np.float32(1000.0) * event_point_count.astype(np.float32)
            / duration_array
        )

        start_array = np.zeros((event_indices.size, 2, 2), np.float32)
        end_array = np.zeros_like(start_array)
        start_array[:, 0] = event_start_raw
        end_array[:, 0] = event_end_raw
        pointer_start_array = np.zeros((event_indices.size, 2), np.float32)
        pointer_end_array = np.zeros_like(pointer_start_array)
        pointer_end_array[:, 0] = 1.0
        span_array = np.zeros((event_indices.size, 2), np.float32)
        angle_array = np.zeros_like(span_array)
        orientation_values = arrays["orientation_id"][event_indices].astype(np.int8)
        id_array = arrays["event_id"][event_indices].astype(np.int64)
        source_user_array = arrays["user_id"][event_indices].astype(np.int16)
        key_duration_array = duration_array / np.maximum(
            selected_key_counts.astype(np.float32), np.float32(1.0)
        )
        key_points_array = key_points_int64.astype(np.int16)
        key_n_keys_array = selected_key_counts.astype(np.int16)
        key_n_letters_array = arrays["n_letters"][event_indices].astype(np.int16)
        raw_keycodes = arrays["keycode"][selected_key_indices].astype(np.int32)
        keycodes_array = np.where(raw_keycodes < 0, 0, raw_keycodes).astype(np.int32)
        key_position_token_array = keycodes_array.copy()
        key_position_orientation_array = orientation_values[local_key_event].astype(np.int8)

        global_min = np.min(decoded_xy, axis=0).astype(np.float32)
        global_max = np.max(decoded_xy, axis=0).astype(np.float32)
        if np.any(global_max <= global_min):
            raise ValueError("train-only screen bounds are degenerate")
        screen_min = np.zeros((len(ORIENTATION_IDS), 2), np.float32)
        screen_max = np.zeros_like(screen_min)
        screen_observed = np.zeros(len(ORIENTATION_IDS), np.uint8)
        point_orientation = orientation_values[point_event]
        for orientation_index, orientation_id in enumerate(ORIENTATION_IDS):
            selected_points = point_orientation == orientation_id
            if np.any(selected_points):
                screen_min[orientation_index] = np.min(decoded_xy[selected_points], axis=0)
                screen_max[orientation_index] = np.max(decoded_xy[selected_points], axis=0)
                screen_observed[orientation_index] = 1
            else:
                screen_min[orientation_index] = global_min
                screen_max[orientation_index] = global_max

        digest_builder = hashlib.sha256()
        values = (
            id_array, source_user_array, duration_array, rate_array, start_array, end_array,
            pointer_start_array, pointer_end_array, span_array, angle_array,
            key_duration_array, key_points_array, key_n_keys_array, key_n_letters_array,
            key_zero_flight_array, keycode_offsets, keycodes_array,
            orientation_values, screen_min, screen_max, screen_observed,
            key_position_token_array, key_position_orientation_array,
            key_position_down_array, key_position_up_array, key_position_center_array,
        )
        for value in values:
            digest_builder.update(np.ascontiguousarray(value).tobytes())
            value.setflags(write=False)
        return cls(
            action="keystroke", train_user_ids=tuple(sorted(allowed)),
            orientation_id=orientation_values, duration_ms=duration_array,
            point_rate_hz=rate_array, start_xy=start_array, end_xy=end_array,
            pointer_start_fraction=pointer_start_array,
            pointer_end_fraction=pointer_end_array,
            pinch_span=span_array, pinch_angle=angle_array,
            key_duration_per_key_ms=key_duration_array,
            key_contact_points=key_points_array,
            key_n_keys=key_n_keys_array, key_n_letters=key_n_letters_array,
            key_zero_flight=key_zero_flight_array,
            keycode_offsets=keycode_offsets, keycodes=keycodes_array,
            key_position_token=key_position_token_array,
            key_position_orientation=key_position_orientation_array,
            key_position_down_xy=key_position_down_array,
            key_position_up_xy=key_position_up_array,
            key_position_center_xy=key_position_center_array,
            screen_min_xy_by_orientation=screen_min,
            screen_max_xy_by_orientation=screen_max,
            screen_orientation_observed=screen_observed,
            source_event_ids=id_array, source_user_ids=source_user_array,
            digest=digest_builder.hexdigest(),
        )

    @classmethod
    def fit(cls, action: str, records, train_users: Iterable[int]) -> "TrainGlobalPrior":
        allowed = set(int(x) for x in train_users)
        lazy_numeric = all(
            hasattr(records, name)
            for name in ("canonical_sample", "user_ids", "action", "splits")
        )
        if lazy_numeric:
            if str(records.action) != action:
                raise ValueError("numeric corpus action does not match train prior")
            if any(records.splits.split_for_user(user_id) != "train" for user_id in allowed):
                raise ValueError("train prior user set contains non-train users")
            if action == "keystroke":
                return cls._fit_numeric_keystroke(records, allowed)
            selected = _LazyCanonicalRecords(
                records,
                np.flatnonzero(np.isin(records.user_ids, np.asarray(sorted(allowed)))).astype(np.int64),
            )
        else:
            selected = [
                x for x in records
                if x.action == action and x.user_id in allowed and x.split == "train"
            ]
        if len(selected) == 0:
            raise ValueError("cannot fit empty train-only prior for %s" % action)
        if not lazy_numeric and any(
            x.split != "train" or x.user_id not in allowed for x in selected
        ):
            raise AssertionError("non-train sample reached train prior")
        # A formal keystroke prior covers about half a million individual
        # keys.  Storing one Python int/small ndarray per key inflated the
        # transient working set by hundreds of MB and could OOM before formal
        # generation.  Preallocate the exact numeric representation instead;
        # this preserves ordering, dtypes and digest semantics while keeping
        # memory proportional to the actual numeric payload.
        record_count = len(selected)
        duration_array = np.empty(record_count, np.float32)
        rate_array = np.zeros((record_count, 2), np.float32)
        start_array = np.empty((record_count, 2, 2), np.float32)
        end_array = np.empty_like(start_array)
        pointer_start_array = np.empty((record_count, 2), np.float32)
        pointer_end_array = np.empty_like(pointer_start_array)
        span_array = np.empty((record_count, 2), np.float32)
        angle_array = np.empty_like(span_array)
        orientation_values = np.empty(record_count, np.int8)
        id_array = np.empty(record_count, np.int64)
        source_user_array = np.empty(record_count, np.int16)

        if action == "keystroke":
            key_counts = np.fromiter(
                (int(item.n_keys) for item in selected),
                dtype=np.int64, count=record_count,
            )
            total_keys = int(np.sum(key_counts))
            total_flights = int(np.sum(np.maximum(key_counts - 1, 0)))
            key_duration_array = np.empty(record_count, np.float32)
            key_points_array = np.empty(total_keys, np.int16)
            key_n_keys_array = np.empty(record_count, np.int16)
            key_n_letters_array = np.empty(record_count, np.int16)
            key_zero_flight_array = np.empty(total_flights, np.uint8)
            keycode_offsets = np.zeros(record_count + 1, np.int64)
            keycode_offsets[1:] = np.cumsum(key_counts)
            keycodes_array = np.empty(total_keys, np.int32)
            key_position_token_array = np.empty(total_keys, np.int32)
            key_position_orientation_array = np.empty(total_keys, np.int8)
            key_position_down_array = np.empty((total_keys, 2), np.float32)
            key_position_up_array = np.empty((total_keys, 2), np.float32)
            key_position_center_array = np.empty((total_keys, 2), np.float32)
        else:
            key_duration_array = np.asarray([0.0], np.float32)
            key_points_array = np.asarray([0], np.int16)
            key_n_keys_array = np.asarray([0], np.int16)
            key_n_letters_array = np.asarray([0], np.int16)
            key_zero_flight_array = np.zeros(0, np.uint8)
            keycode_offsets = np.zeros(1, np.int64)
            keycodes_array = np.zeros(0, np.int32)
            key_position_token_array = np.zeros(0, np.int32)
            key_position_orientation_array = np.zeros(0, np.int8)
            key_position_down_array = np.zeros((0, 2), np.float32)
            key_position_up_array = np.zeros((0, 2), np.float32)
            key_position_center_array = np.zeros((0, 2), np.float32)

        global_min = np.full(2, np.inf, np.float32)
        global_max = np.full(2, -np.inf, np.float32)
        screen_min = np.full((len(ORIENTATION_IDS), 2), np.inf, np.float32)
        screen_max = np.full((len(ORIENTATION_IDS), 2), -np.inf, np.float32)
        screen_observed = np.zeros(len(ORIENTATION_IDS), np.uint8)
        flight_cursor = 0

        for record_index, item in enumerate(selected):
            duration_array[record_index] = float(item.duration_ms)
            rate_row = np.zeros(2, np.float32)
            for pointer_id, pointer in enumerate(item.pointer_features):
                lifetime = float(item.pointer_end_offset_ms[pointer_id] - item.pointer_start_offset_ms[pointer_id])
                rate_row[pointer_id] = 1000.0 * len(pointer) / max(lifetime, 1e-3)
            rate_array[record_index] = rate_row
            start_array[record_index] = np.asarray(item.start_xy, np.float32)
            end_array[record_index] = np.asarray(item.end_xy, np.float32)
            pointer_start_array[record_index] = (
                np.asarray(item.pointer_start_offset_ms, np.float32) / item.duration_ms
            )
            pointer_end_array[record_index] = (
                np.asarray(item.pointer_end_offset_ms, np.float32) / item.duration_ms
            )
            span_array[record_index] = np.asarray(item.pinch_span, np.float32)
            angle_array[record_index] = np.asarray(item.pinch_angle, np.float32)
            orientation_values[record_index] = int(item.orientation_id)
            id_array[record_index] = numeric_sample_id(item.sample_id)
            source_user_array[record_index] = int(item.user_id)
            decoded_contacts = []
            decoded_by_pointer: List[np.ndarray] = []
            for pointer_id, pointer_features in enumerate(item.pointer_features):
                start_xy_value = item.start_xy[pointer_id]
                chord = item.end_xy[pointer_id] - start_xy_value
                chord_length = float(np.linalg.norm(chord))
                scale = max(chord_length, 1.0)
                unit = chord / chord_length if chord_length > 1e-6 else np.asarray([1.0, 0.0], np.float32)
                normal = np.asarray([-unit[1], unit[0]], np.float32)
                decoded = (
                    start_xy_value[None, :]
                    + pointer_features[:, 0:1] * scale * unit[None, :]
                    + pointer_features[:, 1:2] * scale * normal[None, :]
                )
                decoded_by_pointer.append(decoded.astype(np.float32))
                decoded_contacts.append(decoded[item.pointer_contact_masks[pointer_id]])
            contact_xy = np.concatenate(decoded_contacts, axis=0).astype(np.float32)
            if contact_xy.size == 0:
                raise ValueError("train-only prior event has no contact geometry")
            local_min = np.min(contact_xy, axis=0)
            local_max = np.max(contact_xy, axis=0)
            global_min = np.minimum(global_min, local_min)
            global_max = np.maximum(global_max, local_max)
            orientation_index = ORIENTATION_IDS.index(int(item.orientation_id))
            screen_min[orientation_index] = np.minimum(
                screen_min[orientation_index], local_min
            )
            screen_max[orientation_index] = np.maximum(
                screen_max[orientation_index], local_max
            )
            screen_observed[orientation_index] = 1
            if action == "keystroke":
                key_duration_array[record_index] = (
                    float(item.duration_ms) / max(item.n_keys, 1)
                )
                key_n_keys_array[record_index] = int(item.n_keys)
                key_n_letters_array[record_index] = int(item.n_letters)
                zero_flags = keystroke_zero_flight_flags(
                    item.pointer_contact_masks[0],
                    item.pointer_event_ids[0],
                    item.n_keys,
                ).astype(np.uint8)
                key_zero_flight_array[
                    flight_cursor : flight_cursor + zero_flags.size
                ] = zero_flags
                flight_cursor += zero_flags.size
                key_left = int(keycode_offsets[record_index])
                key_right = int(keycode_offsets[record_index + 1])
                keycodes_array[key_left:key_right] = np.asarray(item.keycodes, np.int32)
                key_position_token_array[key_left:key_right] = np.asarray(
                    item.keycodes, np.int32
                )
                key_position_orientation_array[key_left:key_right] = int(item.orientation_id)
                for event_id in range(item.n_keys):
                    positions = np.flatnonzero(item.pointer_event_ids[0] == event_id)
                    if positions.size < 2:
                        raise ValueError("train-only keystroke contact lacks DOWN/UP geometry")
                    slot = key_left + event_id
                    key_points_array[slot] = int(positions.size)
                    key_position_down_array[slot] = decoded_by_pointer[0][positions[0]]
                    key_position_up_array[slot] = decoded_by_pointer[0][positions[-1]]
                    key_position_center_array[slot] = np.mean(
                        decoded_by_pointer[0][positions], axis=0
                    )

        if action == "keystroke" and flight_cursor != key_zero_flight_array.size:
            raise AssertionError("keystroke flight preallocation was not filled exactly")
        if np.any(global_max <= global_min):
            raise ValueError("train-only screen bounds are degenerate")
        for orientation_index in range(len(ORIENTATION_IDS)):
            if screen_observed[orientation_index] == 0:
                # Values are present only to keep the archive rectangular;
                # policy refuses to use an unobserved orientation.
                screen_min[orientation_index] = global_min
                screen_max[orientation_index] = global_max
        digest_builder = hashlib.sha256()
        arrays = (
            id_array, source_user_array, duration_array, rate_array, start_array, end_array,
            pointer_start_array, pointer_end_array, span_array, angle_array,
            key_duration_array, key_points_array, key_n_keys_array, key_n_letters_array,
            key_zero_flight_array, keycode_offsets, keycodes_array,
            orientation_values, screen_min, screen_max, screen_observed,
            key_position_token_array, key_position_orientation_array, key_position_down_array,
            key_position_up_array, key_position_center_array,
        )
        for value in arrays:
            digest_builder.update(np.ascontiguousarray(value).tobytes())
            value.setflags(write=False)
        return cls(
            action=action, train_user_ids=tuple(sorted(allowed)), orientation_id=orientation_values,
            duration_ms=duration_array,
            point_rate_hz=rate_array, start_xy=start_array, end_xy=end_array,
            pointer_start_fraction=pointer_start_array, pointer_end_fraction=pointer_end_array,
            pinch_span=span_array, pinch_angle=angle_array,
            key_duration_per_key_ms=key_duration_array, key_contact_points=key_points_array,
            key_n_keys=key_n_keys_array, key_n_letters=key_n_letters_array,
            key_zero_flight=key_zero_flight_array,
            keycode_offsets=keycode_offsets, keycodes=keycodes_array,
            key_position_token=key_position_token_array,
            key_position_orientation=key_position_orientation_array,
            key_position_down_xy=key_position_down_array,
            key_position_up_xy=key_position_up_array,
            key_position_center_xy=key_position_center_array,
            screen_min_xy_by_orientation=screen_min, screen_max_xy_by_orientation=screen_max,
            screen_orientation_observed=screen_observed,
            source_event_ids=id_array, source_user_ids=source_user_array,
            digest=digest_builder.hexdigest(),
        )


@dataclass(frozen=True)
class ConditionRequest:
    action: str
    user_id: int
    split: str
    fake_id: int
    sample_index: int
    seed: int
    reference_ids: Tuple[int, ...]
    reference_canonical_sha256: Tuple[str, ...]
    carrier_ref_id: int
    lengths: Tuple[int, int]
    duration_ms: float
    orientation_id: int
    start_xy: np.ndarray
    end_xy: np.ndarray
    pinch_span: np.ndarray
    pinch_angle: np.ndarray
    pointer_start_offset_ms: np.ndarray
    pointer_end_offset_ms: np.ndarray
    n_keys: int
    n_letters: int
    keycodes: np.ndarray
    zero_flight_after_key: np.ndarray  # [n_keys-1], 1 means next DOWN has same ms as current UP
    zero_flight_probability: float     # refs + train-only prior shrinkage probability
    key_endpoint_source_code: np.ndarray  # [2]: 0 non-key, 1 ref+prior exact, 2 train exact, 3 orientation fallback
    contact_masks: Tuple[np.ndarray, np.ndarray]
    event_ids: Tuple[np.ndarray, np.ndarray]
    condition_source_code: int  # 2 = formal refs+train prior; 3 = explicit external event plan
    train_prior_digest: str
    screen_min_xy: np.ndarray
    screen_max_xy: np.ndarray


CONDITION_REQUEST_DIGEST_FIELDS = (
    "action", "user_id", "split", "fake_id", "sample_index", "seed",
    "reference_ids", "reference_canonical_sha256", "carrier_ref_id",
    "lengths", "duration_ms", "orientation_id", "start_xy", "end_xy",
    "pinch_span", "pinch_angle", "pointer_start_offset_ms",
    "pointer_end_offset_ms", "n_keys", "n_letters", "keycodes",
    "zero_flight_after_key", "zero_flight_probability",
    "key_endpoint_source_code", "contact_masks", "event_ids",
    "condition_source_code", "train_prior_digest", "screen_min_xy",
    "screen_max_xy",
)


def _condition_digest_add_array(digest, value) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())


def _condition_digest_add_text(digest, value: str) -> None:
    raw = str(value).encode("utf-8")
    digest.update(np.asarray([len(raw)], np.int64).tobytes())
    digest.update(raw)


def canonical_condition_request_digest(request: ConditionRequest) -> bytes:
    """Return a schema-bound SHA-256 over every ``ConditionRequest`` field.

    This is the single encoder shared by preflight, archive publication,
    resume audit and the final 100k audit.  A dataclass field change fails
    closed until this versioned encoder is deliberately updated.
    """

    observed_fields = tuple(field.name for field in dataclass_fields(ConditionRequest))
    if observed_fields != CONDITION_REQUEST_DIGEST_FIELDS:
        raise RuntimeError(
            "ConditionRequest changed without a digest-schema update: %r != %r"
            % (observed_fields, CONDITION_REQUEST_DIGEST_FIELDS)
        )
    digest = hashlib.sha256()
    _condition_digest_add_text(digest, CONDITION_REQUEST_DIGEST_SCHEMA)
    _condition_digest_add_text(digest, request.action)
    _condition_digest_add_text(digest, request.split)
    _condition_digest_add_array(digest, [
        request.fake_id, request.user_id, request.sample_index, request.seed,
        request.carrier_ref_id, request.orientation_id, request.n_keys,
        request.n_letters, request.condition_source_code,
    ])
    _condition_digest_add_array(digest, request.reference_ids)
    for value in request.reference_canonical_sha256:
        _condition_digest_add_text(digest, value)
    for value in (
        request.lengths, [request.duration_ms], request.start_xy, request.end_xy,
        request.pinch_span, request.pinch_angle,
        request.pointer_start_offset_ms, request.pointer_end_offset_ms,
        request.keycodes, request.zero_flight_after_key,
        [request.zero_flight_probability], request.key_endpoint_source_code,
        request.contact_masks[0], request.contact_masks[1],
        request.event_ids[0], request.event_ids[1],
    ):
        _condition_digest_add_array(digest, value)
    _condition_digest_add_text(digest, request.train_prior_digest)
    _condition_digest_add_array(digest, request.screen_min_xy)
    _condition_digest_add_array(digest, request.screen_max_xy)
    return digest.digest()


def canonical_condition_request_sha256(request: ConditionRequest) -> str:
    return canonical_condition_request_digest(request).hex()


def condition_request_set_sha256(
    fake_id_and_digest: Iterable[Tuple[int, object]],
) -> str:
    """Aggregate request digests independent of shard/batch traversal order."""

    normalized = []
    for fake_id, value in fake_id_and_digest:
        raw = bytes.fromhex(value) if isinstance(value, str) else bytes(value)
        if len(raw) != 32:
            raise ValueError("condition request digest must be exactly 32 bytes")
        normalized.append((int(fake_id), raw))
    normalized.sort(key=lambda item: item[0])
    if len({fake_id for fake_id, _ in normalized}) != len(normalized):
        raise ValueError("condition request set contains duplicate fake_id")
    digest = hashlib.sha256()
    _condition_digest_add_text(digest, CONDITION_SET_DIGEST_SCHEMA)
    for fake_id, raw in normalized:
        _condition_digest_add_array(digest, np.asarray([fake_id], np.int64))
        _condition_digest_add_array(digest, np.frombuffer(raw, dtype=np.uint8))
    return digest.hexdigest()


def make_fake_id(action: str, user_id: int, sample_index: int) -> int:
    if action not in ACTION_TO_ID or not 0 <= user_id < 1000 or not 0 <= sample_index < 100000:
        raise ValueError("fake id components outside supported range")
    return 8_000_000_000_000_000 + ACTION_TO_ID[action] * 100_000_000 + user_id * 100_000 + sample_index


class ReferenceConditionPolicy:
    """Construct complete conditions from five refs without corpus access.

    Five refs provide the per-user robust centre and two-ref interpolation;
    the train-only prior contributes a bounded residual.  Orientation is
    selected before any spatial condition and all geometry is then composed
    only from same-orientation refs/prior rows.  Keystroke sequences are
    composed from the five refs and train-only transition statistics rather
    than copied from one carrier.  This yields far more than five metadata
    combinations while preserving strict user/split provenance.
    """

    def __init__(self, train_prior: Optional[TrainGlobalPrior] = None):
        if train_prior is None:
            raise ValueError("five-shot generation requires an audited train-only shrinkage prior")
        self.train_prior = train_prior

    ZERO_FLIGHT_PRIOR_STRENGTH = 32.0

    @staticmethod
    def _reference_zero_flight_values(
        references: Sequence[CanonicalTrajectory],
    ) -> np.ndarray:
        values: List[np.ndarray] = []
        for ref in references:
            if ref.action != "keystroke":
                raise ValueError("zero-flight reference audit is keystroke-only")
            values.append(
                keystroke_zero_flight_flags(
                    ref.pointer_contact_masks[0],
                    ref.pointer_event_ids[0],
                    ref.n_keys,
                ).astype(np.float64)
            )
        return np.concatenate(values) if values else np.zeros(0, np.float64)

    @classmethod
    def zero_flight_probability(
        cls,
        references: Sequence[CanonicalTrajectory],
        prior: TrainGlobalPrior,
    ) -> float:
        """Shrink five-reference topology toward the train-only global rate."""

        if prior.action != "keystroke":
            raise ValueError("zero-flight probability requires a keystroke prior")
        ref_values = cls._reference_zero_flight_values(references)
        prior_values = np.asarray(prior.key_zero_flight, np.float64).reshape(-1)
        if prior_values.size == 0 or np.any(~np.isin(prior_values, [0.0, 1.0])):
            raise ValueError("train-only prior lacks audited inter-key flight topology")
        prior_rate = float(np.mean(prior_values))
        strength = float(cls.ZERO_FLIGHT_PRIOR_STRENGTH)
        probability = (
            float(np.sum(ref_values)) + strength * prior_rate
        ) / (float(ref_values.size) + strength)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("invalid refs+train zero-flight probability")
        return probability

    @staticmethod
    def _blend_linear(ref_values: np.ndarray, prior_values: np.ndarray, rng: random.Random, residual_scale: float = 0.12) -> np.ndarray:
        values = np.asarray(ref_values, np.float64)
        prior = np.asarray(prior_values, np.float64)
        if values.shape[0] < 1 or prior.shape[0] < 1 or prior.shape[1:] != values.shape[1:]:
            raise ValueError("shrinkage value shape mismatch")
        if values.shape[0] == 1:
            i = j = 0
            alpha = 0.5
        else:
            i, j = rng.sample(range(values.shape[0]), 2)
            alpha = rng.uniform(0.15, 0.85)
        center = np.median(values, axis=0)
        interpolation = (1.0 - alpha) * values[i] + alpha * values[j]
        prior_index = rng.randrange(prior.shape[0])
        prior_center = np.median(prior, axis=0)
        result = center + 0.75 * (interpolation - center) + residual_scale * (prior[prior_index] - prior_center)
        return np.asarray(result, np.float32)

    @classmethod
    def _blend_positive(cls, ref_values: np.ndarray, prior_values: np.ndarray, rng: random.Random, residual_scale: float = 0.12) -> np.ndarray:
        ref = np.log(np.maximum(np.asarray(ref_values, np.float64), 1e-3))
        prior = np.log(np.maximum(np.asarray(prior_values, np.float64), 1e-3))
        return np.exp(cls._blend_linear(ref, prior, rng, residual_scale)).astype(np.float32)

    @staticmethod
    def _blend_angles(ref_values: np.ndarray, prior_values: np.ndarray, rng: random.Random) -> np.ndarray:
        ref = np.asarray(ref_values, np.float64)
        prior = np.asarray(prior_values, np.float64)
        ref_vec = np.stack([np.cos(ref), np.sin(ref)], axis=-1)
        prior_vec = np.stack([np.cos(prior), np.sin(prior)], axis=-1)
        mixed = ReferenceConditionPolicy._blend_linear(ref_vec, prior_vec, rng, residual_scale=0.08)
        return np.arctan2(mixed[..., 1], mixed[..., 0]).astype(np.float32)

    @staticmethod
    def _pinch_endpoints(center: np.ndarray, span: np.ndarray, angle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_vector = 0.5 * span[0] * np.asarray([math.cos(float(angle[0])), math.sin(float(angle[0]))], np.float32)
        end_vector = 0.5 * span[1] * np.asarray([math.cos(float(angle[1])), math.sin(float(angle[1]))], np.float32)
        start = np.stack([center[0] - start_vector, center[0] + start_vector]).astype(np.float32)
        end = np.stack([center[1] - end_vector, center[1] + end_vector]).astype(np.float32)
        return start, end

    @staticmethod
    def _clip_pinch_geometry(
        centers: np.ndarray, span: np.ndarray, angle: np.ndarray, low: np.ndarray, high: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        span = np.maximum(np.asarray(span, np.float32), 1.0)
        centers = np.asarray(centers, np.float32).copy()
        # Reduce an outlying span before clamping the centre.  This preserves
        # the exact two-pointer span/angle relationship.
        for endpoint in range(2):
            unit = np.abs(np.asarray([math.cos(float(angle[endpoint])), math.sin(float(angle[endpoint]))], np.float32))
            limits = np.full(2, np.inf, np.float32)
            nonzero = unit > 1e-6
            limits[nonzero] = (high - low)[nonzero] / unit[nonzero]
            span[endpoint] = min(float(span[endpoint]), float(np.min(limits)))
            margin = 0.5 * span[endpoint] * unit
            centers[endpoint] = np.minimum(np.maximum(centers[endpoint], low + margin), high - margin)
        start, end = ReferenceConditionPolicy._pinch_endpoints(centers, span, angle)
        return start, end, span

    @staticmethod
    def _is_letter_token(value: int) -> bool:
        # HMOG KeyPress.csv uses ASCII codes.  The authoritative corpus keeps
        # non-negative values unchanged and maps negative sentinels to 0.
        return is_hmog_ascii_letter_keycode(int(value))

    @staticmethod
    def _quantize_duration(value: float, prior_values: np.ndarray) -> float:
        low = int(math.ceil(float(np.min(prior_values))))
        high = int(math.floor(float(np.max(prior_values))))
        if low <= 0 or high < low:
            raise ValueError("train-only duration range has no positive integer millisecond")
        return float(int(np.clip(int(round(float(value))), low, high)))

    @staticmethod
    def _quantize_lifetimes(
        start_fraction: np.ndarray, end_fraction: np.ndarray, duration_ms: float, active_pointers: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        duration = int(round(float(duration_ms)))
        if duration < 1:
            raise ValueError("integer event duration must be positive")
        starts = np.zeros(2, np.float32)
        ends = np.zeros(2, np.float32)
        starts[:active_pointers] = np.rint(np.asarray(start_fraction[:active_pointers]) * duration)
        ends[:active_pointers] = np.rint(np.asarray(end_fraction[:active_pointers]) * duration)
        starts[:active_pointers] = np.clip(starts[:active_pointers], 0, duration - 1)
        ends[:active_pointers] = np.clip(ends[:active_pointers], 1, duration)
        starts[int(np.argmin(starts[:active_pointers]))] = 0
        ends[int(np.argmax(ends[:active_pointers]))] = duration
        for pointer_id in range(active_pointers):
            if ends[pointer_id] <= starts[pointer_id]:
                if starts[pointer_id] < duration:
                    ends[pointer_id] = starts[pointer_id] + 1
                else:
                    starts[pointer_id] = duration - 1
                    ends[pointer_id] = duration
        if active_pointers == 2 and float(np.max(starts[:2])) >= float(np.min(ends[:2])):
            later = int(np.argmax(starts[:2]))
            earlier_up = int(np.argmin(ends[:2]))
            if ends[earlier_up] < duration:
                ends[earlier_up] = min(duration, starts[later] + 1)
            else:
                starts[later] = max(0, ends[earlier_up] - 1)
        if float(np.min(starts[:active_pointers])) != 0.0 or float(np.max(ends[:active_pointers])) != float(duration):
            raise AssertionError("integer pointer lifetimes lost event-union endpoints")
        if np.any(starts[:active_pointers] >= ends[:active_pointers]):
            raise ValueError("integer pointer lifetime collapsed")
        if active_pointers == 2 and float(np.max(starts[:2])) >= float(np.min(ends[:2])):
            raise ValueError("integer pinch lifetime lost overlap")
        return starts, ends

    @staticmethod
    def _decoded_pointer_xy(item: CanonicalTrajectory, pointer_id: int = 0) -> np.ndarray:
        features = np.asarray(item.pointer_features[pointer_id], np.float32)
        start = np.asarray(item.start_xy[pointer_id], np.float32)
        chord = np.asarray(item.end_xy[pointer_id] - item.start_xy[pointer_id], np.float32)
        length = float(np.linalg.norm(chord))
        scale = max(length, 1.0)
        unit = chord / length if length > 1e-6 else np.asarray([1.0, 0.0], np.float32)
        normal = np.asarray([-unit[1], unit[0]], np.float32)
        return (
            start[None, :] + features[:, 0:1] * scale * unit[None, :]
            + features[:, 1:2] * scale * normal[None, :]
        ).astype(np.float32)

    @classmethod
    def _key_endpoint(
        cls,
        token: int,
        endpoint: int,
        orientation_id: int,
        references: Sequence[CanonicalTrajectory],
        prior: TrainGlobalPrior,
        rng: random.Random,
    ) -> Tuple[np.ndarray, int]:
        if endpoint not in (0, 1):
            raise ValueError("key endpoint index must be 0/1")
        ref_candidates: List[np.ndarray] = []
        for ref in references:
            if ref.orientation_id != orientation_id:
                continue
            decoded = cls._decoded_pointer_xy(ref, 0)
            for event_id, ref_token in enumerate(ref.keycodes.tolist()):
                if int(ref_token) != int(token):
                    continue
                positions = np.flatnonzero(ref.pointer_event_ids[0] == event_id)
                if positions.size:
                    ref_candidates.append(decoded[positions[0 if endpoint == 0 else -1]])
        orientation_mask = np.asarray(prior.key_position_orientation) == int(orientation_id)
        exact_mask = orientation_mask & (np.asarray(prior.key_position_token) == int(token))
        prior_values = prior.key_position_down_xy if endpoint == 0 else prior.key_position_up_xy
        if np.any(exact_mask):
            exact_prior = prior_values[exact_mask]
            if ref_candidates:
                return cls._blend_linear(np.stack(ref_candidates), exact_prior, rng, residual_scale=0.08), 1
            # No user ref contains this token: use only train-user exact-token
            # positions rather than borrowing a mismatched validation/test key.
            index = rng.randrange(exact_prior.shape[0])
            center = np.median(exact_prior, axis=0)
            return np.asarray(center + 0.25 * (exact_prior[index] - center), np.float32), 2
        fallback = np.asarray(prior.key_position_center_xy)[orientation_mask]
        if fallback.size == 0:
            raise ValueError("selected orientation has no train-only keyboard position prior")
        ref_fallback = []
        for ref in references:
            if ref.orientation_id == orientation_id:
                decoded = cls._decoded_pointer_xy(ref, 0)
                ref_fallback.extend(decoded[ref.pointer_contact_masks[0]].tolist())
        if ref_fallback:
            return cls._blend_linear(np.asarray(ref_fallback, np.float32), fallback, rng, residual_scale=0.08), 3
        index = rng.randrange(fallback.shape[0])
        return np.asarray(fallback[index], np.float32), 3

    @staticmethod
    def _prior_key_rows(prior: TrainGlobalPrior) -> List[np.ndarray]:
        offsets = np.asarray(prior.keycode_offsets, np.int64)
        if offsets.ndim != 1 or offsets.size != prior.key_n_keys.size + 1:
            raise ValueError("train-only key prior offsets contradict event count")
        rows = [
            np.asarray(prior.keycodes[int(offsets[i]):int(offsets[i + 1])], np.int64)
            for i in range(offsets.size - 1)
        ]
        if any(row.size != int(prior.key_n_keys[i]) for i, row in enumerate(rows)):
            raise ValueError("train-only key prior offsets contradict n_keys")
        return rows

    @classmethod
    def _compose_key_sequence(
        cls,
        references: Sequence[CanonicalTrajectory],
        prior: TrainGlobalPrior,
        rng: random.Random,
    ) -> Tuple[np.ndarray, int]:
        """Compose, never replay, a sequence from refs + train-only Markov rows."""
        ref_rows = [np.asarray(ref.keycodes, np.int64) for ref in references]
        if any(row.size < 1 for row in ref_rows):
            raise ValueError("keystroke reference has no key sequence")
        prior_rows = cls._prior_key_rows(prior)
        if not prior_rows or not any(row.size for row in prior_rows):
            raise ValueError("keystroke train-only prior has no key sequences")

        ref_lengths = np.asarray([row.size for row in ref_rows], np.float32)
        prior_lengths = np.asarray(prior.key_n_keys, np.float32)
        sampled_length = float(cls._blend_positive(ref_lengths, prior_lengths, rng, residual_scale=0.10))
        n_keys = int(np.clip(round(sampled_length), int(np.min(prior.key_n_keys)), int(np.max(prior.key_n_keys))))
        n_keys = max(1, n_keys)

        ref_ratios = np.asarray([
            float(ref.n_letters) / max(int(ref.n_keys), 1) for ref in references
        ], np.float32)
        prior_ratios = np.asarray(prior.key_n_letters, np.float32) / np.maximum(
            np.asarray(prior.key_n_keys, np.float32), 1.0
        )
        target_ratio = float(cls._blend_linear(ref_ratios, prior_ratios, rng, residual_scale=0.08))
        target_letters = int(np.clip(round(np.clip(target_ratio, 0.0, 1.0) * n_keys), 0, n_keys))

        all_rows = ref_rows + prior_rows
        token_pool = [int(token) for row in all_rows for token in row.tolist()]
        if not token_pool or min(token_pool) < 0 or max(token_pool) > KEYCODE_TOKEN_MAX:
            raise ValueError(
                "key token prior escaped the audited [0,%d] vocabulary"
                % KEYCODE_TOKEN_MAX
            )
        by_letter = {
            False: [token for token in token_pool if not cls._is_letter_token(token)],
            True: [token for token in token_pool if cls._is_letter_token(token)],
        }
        if target_letters and not by_letter[True]:
            target_letters = 0
        if target_letters < n_keys and not by_letter[False]:
            target_letters = n_keys

        transitions: Dict[Tuple[int, bool], List[int]] = {}
        for row in all_rows:
            for left, right in zip(row[:-1].tolist(), row[1:].tolist()):
                transitions.setdefault((int(left), cls._is_letter_token(int(right))), []).append(int(right))

        reference_tuples = {tuple(int(x) for x in row.tolist()) for row in ref_rows}
        for attempt in range(64):
            letter_flags = [True] * target_letters + [False] * (n_keys - target_letters)
            rng.shuffle(letter_flags)
            output: List[int] = []
            # Position candidates draw from all five refs.  Transition
            # candidates additionally draw from every train-only event but
            # never cross an event boundary.
            for position, want_letter in enumerate(letter_flags):
                positional = [
                    int(row[position % row.size]) for row in ref_rows
                    if cls._is_letter_token(int(row[position % row.size])) == want_letter
                ]
                transition = transitions.get((output[-1], want_letter), []) if output else []
                global_candidates = by_letter[want_letter]
                candidates = transition * 3 + positional * 2 + global_candidates
                if not candidates:
                    raise ValueError("cannot realize requested letter/non-letter key composition")
                output.append(int(candidates[rng.randrange(len(candidates))]))
            candidate = tuple(output)
            if candidate not in reference_tuples:
                result = np.asarray(candidate, np.int64)
                return result, int(sum(cls._is_letter_token(x) for x in candidate))
        raise ValueError("refs/train prior are too degenerate to compose a non-replayed key sequence")

    def sample(
        self,
        action: str,
        user_id: int,
        split: str,
        sample_index: int,
        base_seed: int,
        references: Sequence[CanonicalTrajectory],
        *,
        explicit_keycodes: Optional[Sequence[int]] = None,
        explicit_n_letters: Optional[int] = None,
        explicit_orientation_id: Optional[int] = None,
    ) -> ConditionRequest:
        if len(references) != FORMAL_REF_COUNT:
            raise ValueError("condition policy requires exactly five refs")
        ref_ids = tuple(numeric_sample_id(x.sample_id) for x in references)
        if len(set(ref_ids)) != FORMAL_REF_COUNT:
            raise ValueError("reference ids must be unique")
        for ref in references:
            if not ref.is_real or ref.action != action or ref.user_id != user_id or ref.split != split:
                raise ValueError("condition ref provenance mismatch")
        seed = stable_seed(base_seed, action, user_id, sample_index)
        rng = random.Random(seed)
        carrier = references[rng.randrange(FORMAL_REF_COUNT)]
        carrier_id = numeric_sample_id(carrier.sample_id)
        prior = self.train_prior
        if prior.action != action:
            raise ValueError("train prior action mismatch")
        if explicit_n_letters is not None and explicit_keycodes is None:
            raise ValueError("explicit_n_letters requires explicit_keycodes")
        if explicit_keycodes is not None and action != "keystroke":
            raise ValueError("explicit text/keycodes are valid only for keystroke")
        # Select orientation before spatial sampling.  Geometry may use only
        # reference and train-prior rows in that selected coordinate frame.
        eligible_orientation_refs = [
            x for x in references
            if x.orientation_id in ORIENTATION_IDS
            and prior.screen_orientation_observed[ORIENTATION_IDS.index(x.orientation_id)] != 0
        ]
        if not eligible_orientation_refs:
            raise ValueError("none of the five ref orientations has train-only observed screen bounds")
        if explicit_orientation_id is None:
            orientation_id = int(eligible_orientation_refs[rng.randrange(len(eligible_orientation_refs))].orientation_id)
        else:
            orientation_id = int(explicit_orientation_id)
            if orientation_id not in ORIENTATION_IDS:
                raise ValueError("explicit_orientation_id must be one of %r" % (ORIENTATION_IDS,))
            if prior.screen_orientation_observed[ORIENTATION_IDS.index(orientation_id)] == 0:
                raise ValueError("explicit orientation has no train-only geometry support")
            if not any(ref.orientation_id == orientation_id for ref in eligible_orientation_refs):
                raise ValueError("explicit orientation is absent from this user's five references")
        geometry_refs = [x for x in references if x.orientation_id == orientation_id]
        prior_orientation_mask = np.asarray(prior.orientation_id) == orientation_id
        if not geometry_refs or not np.any(prior_orientation_mask):
            raise AssertionError("selected orientation lacks geometry support")
        orientation_index = ORIENTATION_IDS.index(orientation_id)
        low = prior.screen_min_xy_by_orientation[orientation_index]
        high = prior.screen_max_xy_by_orientation[orientation_index]
        active_pointers = 2 if action == "pinch" else 1
        ref_duration = np.asarray([x.duration_ms for x in references], np.float32)
        if action == "keystroke":
            if explicit_keycodes is None:
                keycodes, n_letters = self._compose_key_sequence(references, prior, rng)
                condition_source_code = 2
            else:
                keycodes = np.asarray(list(explicit_keycodes), np.int64)
                if (
                    keycodes.ndim != 1
                    or keycodes.size < 1
                    or np.any(keycodes < 0)
                    or np.any(keycodes > KEYCODE_TOKEN_MAX)
                ):
                    raise ValueError(
                        "explicit keycodes must be a non-empty [0,%d] vector"
                        % KEYCODE_TOKEN_MAX
                    )
                inferred = int(sum(self._is_letter_token(x) for x in keycodes.tolist()))
                n_letters = inferred if explicit_n_letters is None else int(explicit_n_letters)
                if n_letters != inferred:
                    raise ValueError("explicit_n_letters contradicts HMOG ASCII A..Z/a..z keycodes")
                condition_source_code = 3  # external caller text; forbidden in formal benchmark audit
            n_keys = int(keycodes.size)
            zero_flight_probability = self.zero_flight_probability(references, prior)
            zero_flight_after_key = np.asarray(
                [rng.random() < zero_flight_probability for _ in range(max(n_keys - 1, 0))],
                dtype=np.bool_,
            )
            ref_per_key = np.asarray([x.duration_ms / max(x.n_keys, 1) for x in references], np.float32)
            duration = float(self._blend_positive(ref_per_key, prior.key_duration_per_key_ms, rng) * n_keys)
            duration = self._quantize_duration(duration, prior.duration_ms)
            prior_points = prior.key_contact_points.astype(np.float32)
            eligible_points = prior_points[prior_points >= 2]
            if eligible_points.size == 0:
                raise ValueError("train-only key contact prior has no valid DOWN/UP contacts")
            point_center = float(np.median(eligible_points))
            contact_counts = []
            for event_id in range(n_keys):
                topology_ref = carrier if event_id == 0 else references[
                    (event_id + rng.randrange(FORMAL_REF_COUNT)) % FORMAL_REF_COUNT
                ]
                topology_event = event_id % max(topology_ref.n_keys, 1)
                count = float(np.sum(topology_ref.pointer_event_ids[0] == topology_event))
                sampled = float(eligible_points[rng.randrange(eligible_points.size)])
                value = int(round(count + 0.20 * (sampled - point_center)))
                contact_counts.append(max(2, value))
            contact0: List[bool] = []
            event0: List[int] = []
            for event_id, count in enumerate(contact_counts):
                if event_id and not bool(zero_flight_after_key[event_id - 1]):
                    contact0.append(False)
                    event0.append(-1)
                contact0.extend([True] * count)
                event0.extend([event_id] * count)
            contacts = [np.asarray(contact0, np.bool_), np.zeros(0, np.bool_)]
            events = [np.asarray(event0, np.int64), np.zeros(0, np.int64)]
            lengths = (len(contact0), 0)
            positive_intervals = lengths[0] - 1 - int(np.sum(zero_flight_after_key))
            if positive_intervals > int(duration):
                raise ValueError(
                    "integer keystroke duration cannot hold its positive-time contact/gap intervals"
                )
            pointer_start = np.asarray([0.0, 0.0], np.float32)
            pointer_end = np.asarray([duration, 0.0], np.float32)
        else:
            condition_source_code = 2
            n_keys, n_letters = 0, 0
            keycodes = np.zeros(0, np.int64)
            zero_flight_after_key = np.zeros(0, np.bool_)
            zero_flight_probability = 0.0
            duration = float(self._blend_positive(ref_duration, prior.duration_ms, rng))
            duration = self._quantize_duration(duration, prior.duration_ms)
            ref_start_fraction = np.stack([x.pointer_start_offset_ms / x.duration_ms for x in references])
            ref_end_fraction = np.stack([x.pointer_end_offset_ms / x.duration_ms for x in references])
            start_fraction = self._blend_linear(ref_start_fraction, prior.pointer_start_fraction, rng, residual_scale=0.08)
            end_fraction = self._blend_linear(ref_end_fraction, prior.pointer_end_fraction, rng, residual_scale=0.08)
            start_fraction = np.clip(start_fraction, 0.0, 0.95)
            end_fraction = np.clip(end_fraction, 0.05, 1.0)
            start_fraction[active_pointers:] = 0.0
            end_fraction[active_pointers:] = 0.0
            start_fraction[:active_pointers] -= np.min(start_fraction[:active_pointers])
            end_fraction[:active_pointers] /= max(float(np.max(end_fraction[:active_pointers])), 1e-6)
            for pointer_id in range(active_pointers):
                if end_fraction[pointer_id] <= start_fraction[pointer_id] + 0.02:
                    end_fraction[pointer_id] = min(1.0, start_fraction[pointer_id] + 0.02)
            if action == "pinch" and float(np.max(start_fraction[:2])) >= float(np.min(end_fraction[:2])):
                midpoint = 0.5 * (
                    float(np.max(start_fraction[:2])) + float(np.min(end_fraction[:2]))
                )
                start_fraction[:2] = np.minimum(start_fraction[:2], midpoint - 0.01)
                end_fraction[:2] = np.maximum(end_fraction[:2], midpoint + 0.01)
            # Re-establish the event-union invariants after minimum-lifetime repair.
            start_fraction[:active_pointers] -= np.min(start_fraction[:active_pointers])
            end_fraction[:active_pointers] /= np.max(end_fraction[:active_pointers])
            pointer_start, pointer_end = self._quantize_lifetimes(
                start_fraction, end_fraction, duration, active_pointers
            )
            ref_rates = np.zeros((FORMAL_REF_COUNT, 2), np.float32)
            for ref_index, ref in enumerate(references):
                for pointer_id, values in enumerate(ref.pointer_features):
                    lifetime = ref.pointer_end_offset_ms[pointer_id] - ref.pointer_start_offset_ms[pointer_id]
                    ref_rates[ref_index, pointer_id] = 1000.0 * len(values) / max(float(lifetime), 1e-3)
            rates = self._blend_positive(ref_rates[:, :active_pointers], prior.point_rate_hz[:, :active_pointers], rng)
            prior_rates = prior.point_rate_hz[:, :active_pointers]
            rates = np.clip(rates, np.min(prior_rates, axis=0), np.max(prior_rates, axis=0))
            length_values = [
                min(
                    int(pointer_end[p] - pointer_start[p]) + 1,
                    max(2, int(round(float(pointer_end[p] - pointer_start[p]) * float(rates[p]) / 1000.0))),
                )
                for p in range(active_pointers)
            ]
            lengths = tuple(length_values + [0] * (2 - active_pointers))
            contacts = [np.ones(length_values[p], np.bool_) for p in range(active_pointers)] + [np.zeros(0, np.bool_)] * (2 - active_pointers)
            events = [np.zeros(length_values[p], np.int64) for p in range(active_pointers)] + [np.zeros(0, np.int64)] * (2 - active_pointers)

        key_endpoint_source = np.zeros(2, np.int8)
        if action == "pinch":
            ref_centers = np.stack([
                np.stack([np.mean(x.start_xy[:2], axis=0), np.mean(x.end_xy[:2], axis=0)]) for x in geometry_refs
            ])
            prior_centers = np.stack([
                np.mean(prior.start_xy[prior_orientation_mask, :2], axis=1),
                np.mean(prior.end_xy[prior_orientation_mask, :2], axis=1)
            ], axis=1)
            centers = self._blend_linear(ref_centers, prior_centers, rng)
            span = self._blend_positive(
                np.stack([x.pinch_span for x in geometry_refs]), prior.pinch_span[prior_orientation_mask], rng
            )
            angle = self._blend_angles(
                np.stack([x.pinch_angle for x in geometry_refs]), prior.pinch_angle[prior_orientation_mask], rng
            )
            start_xy, end_xy, span = self._clip_pinch_geometry(centers, span, angle, low, high)
            angle = np.asarray(angle, np.float32)
        elif action == "keystroke":
            start0, key_endpoint_source[0] = self._key_endpoint(
                int(keycodes[0]), 0, orientation_id, geometry_refs, prior, rng
            )
            end0, key_endpoint_source[1] = self._key_endpoint(
                int(keycodes[-1]), 1, orientation_id, geometry_refs, prior, rng
            )
            start_xy = np.zeros((2, 2), np.float32)
            end_xy = np.zeros((2, 2), np.float32)
            start_xy[0] = np.clip(start0, low, high)
            end_xy[0] = np.clip(end0, low, high)
            span = np.zeros(2, np.float32)
            angle = np.zeros(2, np.float32)
        else:
            ref_starts = np.stack([x.start_xy[0] for x in geometry_refs])
            ref_displacements = np.stack([x.end_xy[0] - x.start_xy[0] for x in geometry_refs])
            prior_starts = prior.start_xy[prior_orientation_mask, 0]
            prior_displacements = (
                prior.end_xy[prior_orientation_mask, 0] - prior.start_xy[prior_orientation_mask, 0]
            )
            start0 = self._blend_linear(ref_starts, prior_starts, rng)
            displacement = self._blend_linear(
                ref_displacements, prior_displacements, rng, residual_scale=0.18 if action == "tap" else 0.10
            )
            start0 = np.clip(start0, low, high)
            end0 = np.clip(start0 + displacement, low, high)
            start_xy = np.zeros((2, 2), np.float32)
            end_xy = np.zeros((2, 2), np.float32)
            start_xy[0], end_xy[0] = start0, end0
            span = np.zeros(2, np.float32)
            angle = np.zeros(2, np.float32)

        # Continuous shrinkage should never collapse to a complete metadata
        # copy.  Add a deterministic sub-pixel perturbation only in the exact
        # equality corner case (usually identical five refs).
        exact_copy = any(
            abs(duration - ref.duration_ms) <= 1e-7
            and np.array_equal(start_xy, ref.start_xy)
            and np.array_equal(end_xy, ref.end_xy)
            and np.array_equal(pointer_start, ref.pointer_start_offset_ms)
            and np.array_equal(pointer_end, ref.pointer_end_offset_ms)
            for ref in references
        )
        if exact_copy:
            # Preserve the source domain's 1 ms timing lattice.  If the prior
            # range cannot support an integer duration change, move the whole
            # endpoint geometry by one representable subpixel instead.
            low_duration = int(math.ceil(float(np.min(prior.duration_ms))))
            high_duration = int(math.floor(float(np.max(prior.duration_ms))))
            if int(duration) < high_duration:
                old_duration = duration
                duration = float(int(duration) + 1)
                pointer_start = np.rint(pointer_start / old_duration * duration).astype(np.float32)
                pointer_end = np.rint(pointer_end / old_duration * duration).astype(np.float32)
                pointer_start[int(np.argmin(pointer_start[:active_pointers]))] = 0
                pointer_end[int(np.argmax(pointer_end[:active_pointers]))] = duration
            elif int(duration) > low_duration:
                old_duration = duration
                duration = float(int(duration) - 1)
                pointer_start = np.rint(pointer_start / old_duration * duration).astype(np.float32)
                pointer_end = np.rint(pointer_end / old_duration * duration).astype(np.float32)
                pointer_start[int(np.argmin(pointer_start[:active_pointers]))] = 0
                pointer_end[int(np.argmax(pointer_end[:active_pointers]))] = duration
            else:
                epsilon = np.asarray([1e-4, 0.0], np.float32)
                if np.all(end_xy[:active_pointers] + epsilon <= high):
                    start_xy[:active_pointers] += epsilon
                    end_xy[:active_pointers] += epsilon
                elif np.all(start_xy[:active_pointers] - epsilon >= low):
                    start_xy[:active_pointers] -= epsilon
                    end_xy[:active_pointers] -= epsilon
                else:
                    raise ValueError("degenerate refs/prior cannot avoid complete metadata copy")
        request = ConditionRequest(
            action=action, user_id=int(user_id), split=split,
            fake_id=make_fake_id(action, user_id, sample_index), sample_index=int(sample_index), seed=seed,
            reference_ids=ref_ids,
            reference_canonical_sha256=tuple(canonical_sample_sha256(x) for x in references),
            carrier_ref_id=carrier_id, lengths=(int(lengths[0]), int(lengths[1])),
            duration_ms=duration, orientation_id=orientation_id,
            start_xy=start_xy, end_xy=end_xy,
            pinch_span=span, pinch_angle=angle,
            pointer_start_offset_ms=pointer_start, pointer_end_offset_ms=pointer_end,
            n_keys=n_keys, n_letters=n_letters, keycodes=keycodes,
            zero_flight_after_key=zero_flight_after_key,
            zero_flight_probability=float(zero_flight_probability),
            key_endpoint_source_code=key_endpoint_source,
            contact_masks=(np.asarray(contacts[0], np.bool_).copy(), np.asarray(contacts[1], np.bool_).copy()),
            event_ids=(np.asarray(events[0], np.int64).copy(), np.asarray(events[1], np.int64).copy()),
            condition_source_code=condition_source_code,
            train_prior_digest=prior.digest,
            screen_min_xy=np.asarray(low, np.float32).copy(), screen_max_xy=np.asarray(high, np.float32).copy(),
        )
        self.validate_request(request)
        return request

    @staticmethod
    def validate_request(request: ConditionRequest) -> None:
        if request.action not in ACTIONS or request.split not in SPLIT_TO_ID:
            raise ValueError("invalid request action/split")
        if len(request.reference_ids) != FORMAL_REF_COUNT or len(set(request.reference_ids)) != FORMAL_REF_COUNT:
            raise ValueError("request must preserve five unique ref ids")
        if len(request.reference_canonical_sha256) != FORMAL_REF_COUNT or any(
            len(value) != 64 for value in request.reference_canonical_sha256
        ):
            raise ValueError("request must bind five canonical reference SHA-256 values")
        if request.carrier_ref_id not in request.reference_ids:
            raise ValueError("contact-topology anchor is not in the five refs")
        if request.condition_source_code not in (2, 3):
            raise ValueError("unknown condition source")
        if not request.train_prior_digest:
            raise ValueError("hybrid condition lacks train-only prior digest")
        if request.orientation_id not in ORIENTATION_IDS or not math.isfinite(request.duration_ms) or request.duration_ms <= 0:
            raise ValueError("invalid duration/orientation condition")
        if abs(request.duration_ms - round(request.duration_ms)) > 1e-6:
            raise ValueError("formal condition duration must lie on the HMOG 1 ms lattice")
        if request.screen_min_xy.shape != (2,) or request.screen_max_xy.shape != (2,) or np.any(request.screen_max_xy <= request.screen_min_xy):
            raise ValueError("invalid train-only orientation screen bounds")
        expected_pointers = 2 if request.action == "pinch" else 1
        if sum(int(x > 0) for x in request.lengths) != expected_pointers or any(x < 0 for x in request.lengths):
            raise ValueError("condition pointer count mismatch")
        if request.action == "keystroke":
            if request.n_keys <= 0 or request.keycodes.size != request.n_keys or not 0 <= request.n_letters <= request.n_keys:
                raise ValueError("incomplete keystroke conditions")
            if sum(is_hmog_ascii_letter_keycode(int(value)) for value in request.keycodes) != request.n_letters:
                raise ValueError("n_letters contradicts HMOG ASCII A..Z/a..z tokens")
            if request.key_endpoint_source_code.shape != (2,) or np.any(
                ~np.isin(request.key_endpoint_source_code, [1, 2, 3, 4])
            ):
                raise ValueError("keystroke lacks first/last key-conditioned endpoint provenance")
            if request.zero_flight_after_key.shape != (request.n_keys - 1,) or request.zero_flight_after_key.dtype != np.bool_:
                raise ValueError("keystroke zero-flight topology must have shape [n_keys-1]")
            if not math.isfinite(float(request.zero_flight_probability)) or not 0.0 <= float(
                request.zero_flight_probability
            ) <= 1.0:
                raise ValueError("invalid zero-flight sampling probability")
            observed_zero = keystroke_zero_flight_flags(
                request.contact_masks[0], request.event_ids[0], request.n_keys
            )
            if not np.array_equal(observed_zero, request.zero_flight_after_key):
                raise ValueError("keystroke contact topology contradicts zero-flight condition")
        elif request.n_keys != 0 or request.n_letters != 0 or request.keycodes.size:
            raise ValueError("non-keystroke cannot carry key conditions")
        elif request.key_endpoint_source_code.shape != (2,) or np.any(request.key_endpoint_source_code != 0):
            raise ValueError("non-keystroke cannot carry key endpoint provenance")
        elif request.zero_flight_after_key.size or float(request.zero_flight_probability) != 0.0:
            raise ValueError("non-keystroke cannot carry zero-flight conditions")
        if request.action == "pinch":
            if np.any(request.pinch_span < 0) or request.pinch_span.shape != (2,) or request.pinch_angle.shape != (2,):
                raise ValueError("incomplete pinch span/angle")
        active = 2 if request.action == "pinch" else 1
        endpoint_values = np.concatenate([request.start_xy[:active], request.end_xy[:active]], axis=0)
        if np.any(endpoint_values < request.screen_min_xy - 1e-4) or np.any(endpoint_values > request.screen_max_xy + 1e-4):
            raise ValueError("endpoint condition is outside train-only orientation screen bounds")
        starts = request.pointer_start_offset_ms[:active]
        ends = request.pointer_end_offset_ms[:active]
        if np.any(starts < 0) or np.any(starts >= ends) or np.any(ends > request.duration_ms + 1e-3):
            raise ValueError("invalid global pointer lifetimes")
        if np.any(np.abs(starts - np.rint(starts)) > 1e-6) or np.any(np.abs(ends - np.rint(ends)) > 1e-6):
            raise ValueError("pointer lifetimes must lie on the HMOG 1 ms lattice")
        if request.action == "keystroke":
            required_positive_intervals = (
                int(request.lengths[0]) - 1 - int(np.sum(request.zero_flight_after_key))
            )
            if required_positive_intervals > int(round(float(ends[0] - starts[0]))):
                raise ValueError("keystroke positive-time intervals do not fit integer-ms lifetime")
        elif any(request.lengths[p] > int(round(float(ends[p] - starts[p]))) + 1 for p in range(active)):
            raise ValueError("point count cannot fit a strictly increasing integer-ms timeline")
        if abs(float(np.min(starts))) > 1e-3 or abs(float(np.max(ends)) - request.duration_ms) > 1e-3:
            raise ValueError("pointer lifetimes must span complete event duration")
        if request.action == "pinch" and float(np.max(starts)) >= float(np.min(ends)):
            raise ValueError("pinch pointers must overlap on the global union timeline")


def choose_reference_sets(
    pool: Sequence[CanonicalTrajectory], action: str, user_id: int, split: str,
    n_samples: int, base_seed: int,
) -> List[Tuple[CanonicalTrajectory, ...]]:
    """Deterministically draw exact five unique refs for every fake."""
    candidates = [x for x in pool if x.is_real and x.action == action and x.user_id == user_id and x.split == split]
    by_id: Dict[str, CanonicalTrajectory] = {}
    for item in candidates:
        if not item.sample_id or item.sample_id in by_id:
            raise ValueError("real pool contains missing/duplicate reference id")
        by_id[item.sample_id] = item
    candidates = list(by_id.values())
    if len(candidates) < FORMAL_REF_COUNT:
        raise ValueError("%s user %d split %s has %d refs; five required" % (action, user_id, split, len(candidates)))
    # One immutable enrollment/reference set per user/action/split.  All 200
    # generated candidates for that unit use the same five refs; only Gaussian
    # noise and condition residual draws vary by sample_index.
    rng = random.Random(stable_seed(base_seed, action, user_id, 0) ^ 0x5EED5EED)
    refs = tuple(rng.sample(candidates, FORMAL_REF_COUNT))
    if len({x.sample_id for x in refs}) != FORMAL_REF_COUNT:
        raise AssertionError("reference sampler returned duplicates")
    return [refs] * n_samples
