"""Numeric-only HMOG trajectory corpus loader and strict source audit.

The extractor writes compressed numeric columns plus three offset tables.  This
module is the inverse operation used by training: one event is reconstructed
without pickle, without resampling and without joining two key contacts by a
fictional on-screen line.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from trajectory.data import (
    ACTIONS,
    KEYCODE_TOKEN_MAX,
    CanonicalTrajectory,
    canonicalize_sample,
)
from trajectory.features import is_hmog_ascii_letter_keycode


SCHEMA_VERSION = "hmog_touch_trajectory_v1"
FORMAL_SPLIT_SHA256 = "82f2277374be47d5ec9dada2f7e60d0d5afd7ba79ac8a08b67e1607294ff530b"
FORMAL_SPLIT_PATH = Path(
    "/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json"
)
KEYCODE_TOKEN_OFFSET = 0
KEYCODE_TOKEN_MAX_RAW = KEYCODE_TOKEN_MAX

FLAT_FIELDS = (
    "flat_system_time_ms",
    "flat_event_time_ms",
    "flat_t_rel_ms",
    "flat_frame_index",
    "flat_pointer_count",
    "flat_pointer_id",
    "flat_action_code",
    "flat_x",
    "flat_y",
    "flat_pressure",
    "flat_size",
    "flat_orientation_id",
    "flat_active_mask",
    "flat_valid_mask",
    "flat_key_index",
    "flat_keycode",
)
EVENT_FIELDS = (
    "event_id",
    "user_id",
    "user_external_id",
    "session_id",
    "action_id",
    "activity_id",
    "orientation_id",
    "label_start_ms",
    "label_end_ms",
    "label_duration_ms",
    "touch_start_ms",
    "touch_end_ms",
    "touch_duration_ms",
    "active_start_rel_ms",
    "active_end_rel_ms",
    "n_rows",
    "n_frames",
    "n_pointers",
    "max_pointer_count",
    "active_row_count",
    "raw_gesture_id",
    "n_raw_gestures",
    "match_start_error_ms",
    "match_end_error_ms",
    "n_keys",
    "n_letters",
)
KEY_FIELDS = (
    "key_index_in_event",
    "keycode",
    "key_is_letter",
    "key_down_ms",
    "key_up_ms",
    "key_hold_ms",
    "key_flight_from_previous_ms",
    "key_orientation_id",
    "key_raw_gesture_id",
    "key_touch_start_ms",
    "key_touch_end_ms",
    "key_match_start_error_ms",
    "key_match_end_error_ms",
    "key_touch_found",
)
OFFSET_FIELDS = ("event_offsets", "event_key_offsets", "key_touch_offsets")
SCALAR_FIELDS = (
    "schema_version",
    "action_name",
    "action_id_scalar",
    "time_unit",
    "coordinate_unit",
    "sampling",
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class SplitDefinition:
    path: Path
    sha256: str
    seed: int
    train_users: Tuple[int, ...]
    val_users: Tuple[int, ...]
    test_users: Tuple[int, ...]

    @classmethod
    def load(cls, path: Path = FORMAL_SPLIT_PATH, require_pinned_hash: bool = True) -> "SplitDefinition":
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError("split file not found: %s" % path)
        digest = sha256_file(path)
        if require_pinned_hash and digest != FORMAL_SPLIT_SHA256:
            raise ValueError(
                "formal users_seed42.json hash changed: %s != %s"
                % (digest, FORMAL_SPLIT_SHA256)
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = {
            name: tuple(int(x) for x in payload["%s_users" % name])
            for name in ("train", "val", "test")
        }
        if int(payload.get("seed", -1)) != 42:
            raise ValueError("formal split seed must be 42")
        if tuple(payload.get("actions", ())) != ACTIONS:
            raise ValueError("split action order must match the five formal actions")
        expected_sizes = {"train": 70, "val": 10, "test": 20}
        for name, users in values.items():
            if len(users) != expected_sizes[name] or len(set(users)) != len(users):
                raise ValueError("%s split must contain %d unique users" % (name, expected_sizes[name]))
        sets = {name: set(users) for name, users in values.items()}
        if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
            raise ValueError("user splits overlap")
        if sets["train"] | sets["val"] | sets["test"] != set(range(100)):
            raise ValueError("formal splits must partition user ids 0..99")
        return cls(
            path=path,
            sha256=digest,
            seed=42,
            train_users=values["train"],
            val_users=values["val"],
            test_users=values["test"],
        )

    def split_for_user(self, user_id: int) -> str:
        value = int(user_id)
        if value in self.train_users:
            return "train"
        if value in self.val_users:
            return "val"
        if value in self.test_users:
            return "test"
        raise ValueError("user_id outside fixed split: %d" % value)

    def users(self, split: str) -> Tuple[int, ...]:
        if split not in ("train", "val", "test"):
            raise ValueError("unknown split: %s" % split)
        return getattr(self, "%s_users" % split)

    def audit_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "seed": self.seed,
            "counts": {"train": 70, "val": 10, "test": 20},
            "train_users": list(self.train_users),
            "val_users": list(self.val_users),
            "test_users": list(self.test_users),
        }


def _validate_offsets(name: str, offsets: np.ndarray, expected_last: int) -> None:
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or offsets.size < 1:
        raise ValueError("%s must be a non-empty integer vector" % name)
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(expected_last):
        raise ValueError("%s boundary mismatch" % name)
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("%s is not monotonic" % name)


def encode_raw_keycode(raw_keycode: int) -> int:
    value = int(raw_keycode)
    # HMOG uses negative sentinel codes (-1/-2/-5).  They remain losslessly
    # recorded in metadata, while the neural vocabulary uses token 0 for all
    # non-key/sentinel values.  Positive ASCII/key values stay unchanged so
    # training and formal generation share one codebook.
    if value < 0:
        return 0
    if value > KEYCODE_TOKEN_MAX_RAW:
        raise ValueError(
            "raw keycode %d outside auditable token range [0,%d]"
            % (value, KEYCODE_TOKEN_MAX_RAW)
        )
    return value


def canonical_sample_sha256(sample: CanonicalTrajectory) -> str:
    """Element-level digest used to prove training/generation loader identity."""
    digest = hashlib.sha256()

    def add_text(value: Any) -> None:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    def add_array(value: Any) -> None:
        array = np.ascontiguousarray(np.asarray(value))
        add_text(array.dtype.str)
        add_text(tuple(int(x) for x in array.shape))
        digest.update(array.tobytes(order="C"))

    for value in (sample.action, sample.sample_id, sample.user_id, sample.split, int(sample.is_real)):
        add_text(value)
    for values in sample.pointer_features:
        add_array(values)
    for values in sample.pointer_contact_masks:
        add_array(values)
    for values in sample.pointer_event_ids:
        add_array(values)
    for value in (
        np.asarray([sample.duration_ms], dtype=np.float32),
        sample.pointer_start_offset_ms,
        sample.pointer_end_offset_ms,
        np.asarray([sample.orientation_id], dtype=np.int64),
        sample.start_xy,
        sample.end_xy,
        sample.pinch_span,
        sample.pinch_angle,
        np.asarray([sample.n_keys, sample.n_letters], dtype=np.int64),
        sample.keycodes,
    ):
        add_array(value)
    return digest.hexdigest()


class NumericTrajectoryCorpus:
    """One action archive, restored from numeric flat+offset columns only."""

    def __init__(
        self,
        path: Path,
        splits: SplitDefinition,
        expected_action: Optional[str] = None,
        verify_sha256: bool = True,
        extraction_manifest: Optional[Path] = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.splits = splits
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha256 = sha256_file(self.path) if verify_sha256 else "not_computed"
        self.extraction_manifest_path = (
            Path(extraction_manifest).resolve()
            if extraction_manifest is not None
            else self.path.parent / "manifest.json"
        )
        self.extraction_manifest_sha256 = (
            sha256_file(self.extraction_manifest_path)
            if self.extraction_manifest_path.is_file()
            else None
        )
        self._arrays: Dict[str, np.ndarray] = {}
        self._all_field_dtypes: Dict[str, str] = {}
        self._read_and_validate_archive(expected_action)
        # ``event_length`` is the number of lossless flat archive rows, not
        # necessarily the temporal length seen by the model.  Cache the latter
        # separately: canonicalization removes repeated Android state snapshots
        # and, for a positive keystroke flight, inserts one explicit no-contact
        # midpoint.  Only int64 lengths are cached, never decoded trajectories.
        self._canonical_max_points_cache = np.full(len(self), -1, dtype=np.int64)
        self._indices_by_split_user: Dict[Tuple[str, int], np.ndarray] = {}
        for split in ("train", "val", "test"):
            for user_id in self.splits.users(split):
                indices = np.flatnonzero(self._arrays["user_id"] == user_id).astype(np.int64)
                self._indices_by_split_user[(split, user_id)] = indices

    def _read_and_validate_archive(self, expected_action: Optional[str]) -> None:
        required = set(FLAT_FIELDS + EVENT_FIELDS + KEY_FIELDS + OFFSET_FIELDS + SCALAR_FIELDS)
        with np.load(str(self.path), allow_pickle=False) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError("archive missing fields: %s" % missing)
            # Access every field with allow_pickle=False.  This is an explicit
            # proof that no hidden object/pickle column exists.
            for name in archive.files:
                array = np.asarray(archive[name])
                if array.dtype.kind == "O":
                    raise ValueError("object array is forbidden: %s" % name)
                self._all_field_dtypes[name] = str(array.dtype)
            for name in required:
                self._arrays[name] = np.asarray(archive[name]).copy()

        def scalar(name: str) -> Any:
            value = self._arrays[name]
            if value.ndim != 0:
                raise ValueError("%s must be scalar" % name)
            return value.item()

        self.schema_version = str(scalar("schema_version"))
        self.action = str(scalar("action_name"))
        self.action_id = int(scalar("action_id_scalar"))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported schema: %s" % self.schema_version)
        if self.action not in ACTIONS or (expected_action is not None and self.action != expected_action):
            raise ValueError("unexpected action: %s" % self.action)
        if str(scalar("time_unit")) != "ms" or str(scalar("coordinate_unit")) != "raw_screen_pixel":
            raise ValueError("unexpected time/coordinate units")
        if str(scalar("sampling")) != "raw_irregular_touch_events":
            raise ValueError("source touch samples must remain irregular and unresampled")

        n_events = int(self._arrays["event_id"].size)
        n_flat = int(self._arrays["flat_x"].size)
        n_keys = int(self._arrays["keycode"].size)
        for name in FLAT_FIELDS:
            if self._arrays[name].ndim != 1 or self._arrays[name].size != n_flat:
                raise ValueError("flat column length mismatch: %s" % name)
        for name in EVENT_FIELDS:
            if self._arrays[name].ndim != 1 or self._arrays[name].size != n_events:
                raise ValueError("event column length mismatch: %s" % name)
        for name in KEY_FIELDS:
            if self._arrays[name].ndim != 1 or self._arrays[name].size != n_keys:
                raise ValueError("key column length mismatch: %s" % name)
        _validate_offsets("event_offsets", self._arrays["event_offsets"], n_flat)
        _validate_offsets("event_key_offsets", self._arrays["event_key_offsets"], n_keys)
        _validate_offsets("key_touch_offsets", self._arrays["key_touch_offsets"], n_flat if self.action == "keystroke" else 0)
        if self._arrays["event_offsets"].size != n_events + 1 or self._arrays["event_key_offsets"].size != n_events + 1:
            raise ValueError("event offset table length mismatch")
        if self._arrays["key_touch_offsets"].size != n_keys + 1:
            raise ValueError("key_touch_offsets length mismatch")
        if len(set(int(x) for x in self._arrays["event_id"].tolist())) != n_events:
            raise ValueError("event_id must be globally unique within action")
        if np.any(self._arrays["flat_valid_mask"] != 1):
            raise ValueError("strict extracted rows must all be valid")
        if np.any(self._arrays["n_rows"] != np.diff(self._arrays["event_offsets"])):
            raise ValueError("n_rows contradicts event_offsets")
        if np.any(self._arrays["n_letters"] < 0) or np.any(self._arrays["n_letters"] > self._arrays["n_keys"]):
            raise ValueError("n_letters must be a subset of n_keys")
        if np.any(self._arrays["action_id"] != self.action_id):
            raise ValueError("event action_id contradicts archive scalar")
        for user_id in self._arrays["user_id"].tolist():
            self.splits.split_for_user(int(user_id))

        key_counts = np.diff(self._arrays["event_key_offsets"])
        if self.action == "keystroke":
            if np.any(key_counts != self._arrays["n_keys"]) or np.any(key_counts <= 0):
                raise ValueError("keystroke n_keys contradicts key offsets")
            letter_prefix = np.concatenate(
                [np.zeros(1, dtype=np.int64), np.cumsum(self._arrays["key_is_letter"], dtype=np.int64)]
            )
            starts = self._arrays["event_key_offsets"][:-1]
            ends = self._arrays["event_key_offsets"][1:]
            if np.any(letter_prefix[ends] - letter_prefix[starts] != self._arrays["n_letters"]):
                raise ValueError("n_letters contradicts key_is_letter")
            expected_letter_flags = np.asarray([
                is_hmog_ascii_letter_keycode(int(raw_keycode))
                for raw_keycode in self._arrays["keycode"].tolist()
            ], dtype=np.uint8)
            if not np.array_equal(self._arrays["key_is_letter"].astype(np.uint8), expected_letter_flags):
                raise ValueError("key_is_letter contradicts the HMOG ASCII keycode codebook")
            if np.any(self._arrays["key_touch_found"] != 1):
                raise ValueError("formal keystroke corpus cannot contain unmatched keys")
            for raw_keycode in self._arrays["keycode"].tolist():
                encode_raw_keycode(int(raw_keycode))
        else:
            if n_keys != 0 or np.any(key_counts != 0) or np.any(self._arrays["n_keys"] != 0) or np.any(self._arrays["n_letters"] != 0):
                raise ValueError("non-keystroke archive contains key records")

        # If an extraction manifest accompanies the archive, verify the file
        # hash and counts instead of merely recording the manifest path.
        if self.extraction_manifest_path.is_file():
            manifest = json.loads(self.extraction_manifest_path.read_text(encoding="utf-8"))
            output = manifest.get("outputs", {}).get(self.action)
            if not isinstance(output, dict):
                raise ValueError("extraction manifest lacks action output: %s" % self.action)
            if output.get("sha256") != self.sha256:
                raise ValueError("NPZ hash contradicts extraction manifest")
            expected_counts = (int(output["n_events"]), int(output["n_flat_rows"]), int(output["n_keys"]))
            if expected_counts != (n_events, n_flat, n_keys):
                raise ValueError("NPZ counts contradict extraction manifest")

    def __len__(self) -> int:
        return int(self._arrays["event_id"].size)

    @property
    def event_ids(self) -> np.ndarray:
        return self._arrays["event_id"]

    @property
    def user_ids(self) -> np.ndarray:
        return self._arrays["user_id"]

    def indices_for_split(self, split: str) -> np.ndarray:
        users = np.asarray(self.splits.users(split), dtype=self.user_ids.dtype)
        return np.flatnonzero(np.isin(self.user_ids, users)).astype(np.int64)

    def indices_for_user(self, split: str, user_id: int) -> np.ndarray:
        if self.splits.split_for_user(user_id) != split:
            raise ValueError("user %d is not in %s" % (user_id, split))
        return self._indices_by_split_user[(split, int(user_id))].copy()

    def event_length(self, index: int) -> int:
        i = self._check_index(index)
        return int(self._arrays["event_offsets"][i + 1] - self._arrays["event_offsets"][i])

    def event_key_count(self, index: int) -> int:
        """Return the exact canonical key-token count without decoding."""
        i = self._check_index(index)
        return int(self._arrays["n_keys"][i])

    def canonical_max_points(self, index: int) -> int:
        """Return the exact maximum model-timeline length for one event.

        This mirrors :func:`trajectory.data.canonicalize_sample` without
        materializing feature arrays.  For ordinary actions it is the longest
        de-duplicated pointer timeline.  For keystroke it is the sum of all
        de-duplicated key contacts plus one midpoint for every strictly
        positive inter-key flight; equal-time UP/DOWN boundaries remain
        adjacent and do not invent a gap token.
        """
        i = self._check_index(index)
        cached = int(self._canonical_max_points_cache[i])
        if cached >= 0:
            return cached

        left, right, key_left, key_right = self._event_bounds(i)
        if self.action == "keystroke":
            length = 0
            previous_end_time: Optional[int] = None
            for global_key_index in range(key_left, key_right):
                contact_left = int(self._arrays["key_touch_offsets"][global_key_index])
                contact_right = int(self._arrays["key_touch_offsets"][global_key_index + 1])
                model_rows = self._model_pointer_rows(
                    np.arange(contact_left, contact_right, dtype=np.int64)
                )
                if model_rows.size < 2:
                    raise ValueError("canonical key contact has fewer than two points")
                start_time = int(self._arrays["flat_t_rel_ms"][model_rows[0]])
                end_time = int(self._arrays["flat_t_rel_ms"][model_rows[-1]])
                if previous_end_time is not None:
                    if start_time < previous_end_time:
                        raise ValueError("canonical key contacts overlap")
                    if start_time > previous_end_time:
                        length += 1
                length += int(model_rows.size)
                previous_end_time = end_time
        else:
            pointer_values = self._arrays["flat_pointer_id"][left:right]
            pointer_ids = list(dict.fromkeys(int(value) for value in pointer_values.tolist()))
            lengths = []
            for pointer_id in pointer_ids:
                rows = np.flatnonzero(pointer_values == pointer_id).astype(np.int64) + left
                lengths.append(int(self._model_pointer_rows(rows).size))
            if not lengths:
                raise ValueError("canonical event has no pointer timeline")
            length = max(lengths)

        if length < 2:
            raise ValueError("canonical event needs at least two points")
        self._canonical_max_points_cache[i] = int(length)
        return int(length)

    def _check_index(self, index: int) -> int:
        value = int(index)
        if value < 0:
            value += len(self)
        if value < 0 or value >= len(self):
            raise IndexError(index)
        return value

    def _event_bounds(self, index: int) -> Tuple[int, int, int, int]:
        i = self._check_index(index)
        left = int(self._arrays["event_offsets"][i])
        right = int(self._arrays["event_offsets"][i + 1])
        key_left = int(self._arrays["event_key_offsets"][i])
        key_right = int(self._arrays["event_key_offsets"][i + 1])
        if right <= left:
            raise ValueError("empty event at index %d" % i)
        return left, right, key_left, key_right

    def event_flat_rows(self, index: int) -> Dict[str, np.ndarray]:
        """Return every extractor row exactly, before model-view de-duplication."""
        left, right, _, _ = self._event_bounds(index)
        return {
            name: self._arrays[name][left:right].copy()
            for name in FLAT_FIELDS
        }

    def _model_pointer_rows(self, indices: np.ndarray) -> np.ndarray:
        """Collapse repeated Android pointer-state snapshots deterministically.

        The exact rows remain available through :meth:`event_flat_rows`.  The
        canonical model view keeps one row per pointer/frame/time and then one
        row per pointer timestamp, matching formal generation corpus loading.
        """
        selected: Dict[Tuple[int, int, int], int] = {}
        for raw_index in np.asarray(indices, dtype=np.int64).tolist():
            key = (
                int(self._arrays["flat_pointer_id"][raw_index]),
                int(self._arrays["flat_frame_index"][raw_index]),
                int(self._arrays["flat_t_rel_ms"][raw_index]),
            )
            selected[key] = int(raw_index)  # keep last exact state snapshot
        ordered = sorted(
            selected.values(),
            key=lambda row: (
                int(self._arrays["flat_t_rel_ms"][row]),
                int(self._arrays["flat_frame_index"][row]),
                int(self._arrays["flat_pointer_id"][row]),
                int(row),
            ),
        )
        result = []
        previous_time = None
        for row in ordered:
            current_time = int(self._arrays["flat_t_rel_ms"][row])
            if previous_time is None or current_time > previous_time:
                result.append(row)
                previous_time = current_time
        return np.asarray(result, dtype=np.int64)

    def raw_sample(self, index: int) -> Dict[str, Any]:
        """Restore one raw event with lossless contact timing and provenance."""
        i = self._check_index(index)
        left, right, key_left, key_right = self._event_bounds(i)
        user_id = int(self._arrays["user_id"][i])
        split = self.splits.split_for_user(user_id)
        duration_ms = float(self._arrays["touch_duration_ms"][i])
        if not math.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError("event duration must be positive")
        event_id = int(self._arrays["event_id"][i])
        if event_id < 0:
            raise ValueError("formal source event_id must be non-negative")
        # action is already a separate, required field.  Keeping the identity
        # as the decimal numeric id also matches generation-side provenance.
        sample_id = str(event_id)
        base_metadata: Dict[str, Any] = {
            "source_npz": str(self.path),
            "source_npz_sha256": self.sha256,
            "source_event_index": i,
            "source_event_id": int(self._arrays["event_id"][i]),
            "user_external_id": int(self._arrays["user_external_id"][i]),
            "session_id": int(self._arrays["session_id"][i]),
            "activity_id": int(self._arrays["activity_id"][i]),
            "raw_sampling": "irregular_touch_events",
            "source_flat_row_count": int(right - left),
            "n_keys": int(self._arrays["n_keys"][i]),
            "n_letters": int(self._arrays["n_letters"][i]),
        }
        result: Dict[str, Any] = {
            "action": self.action,
            "sample_id": sample_id,
            "user_id": user_id,
            "split": split,
            "is_real": True,
            "orientation_id": int(self._arrays["orientation_id"][i]),
            "duration_ms": duration_ms,
            "n_letters": int(self._arrays["n_letters"][i]),
            "metadata": base_metadata,
        }

        if self.action == "keystroke":
            if key_right - key_left != int(self._arrays["n_keys"][i]):
                raise ValueError("key count mismatch while restoring event")
            contacts = []
            raw_keycodes = []
            pointer_start_offsets = []
            pointer_end_offsets = []
            for global_key_index in range(key_left, key_right):
                contact_left = int(self._arrays["key_touch_offsets"][global_key_index])
                contact_right = int(self._arrays["key_touch_offsets"][global_key_index + 1])
                if contact_left < left or contact_right > right or contact_right - contact_left < 2:
                    raise ValueError("key_touch_offsets escape event or lack DOWN/UP")
                raw_code = int(self._arrays["keycode"][global_key_index])
                token = encode_raw_keycode(raw_code)
                model_rows = self._model_pointer_rows(
                    np.arange(contact_left, contact_right, dtype=np.int64)
                )
                if model_rows.size < 2:
                    raise ValueError("key contact has fewer than two distinct timestamps")
                timestamps = self._arrays["flat_t_rel_ms"][model_rows].astype(np.float32, copy=True)
                if np.any(np.diff(timestamps) < 0):
                    raise ValueError("key contact timestamps go backwards")
                start_offset = float(timestamps[0])
                end_offset = float(timestamps[-1])
                if start_offset < 0 or end_offset > duration_ms + 1e-3:
                    raise ValueError("key contact lies outside full typing event")
                contacts.append(
                    {
                        "keycode": token,
                        "raw_keycode": raw_code,
                        "is_letter": bool(self._arrays["key_is_letter"][global_key_index]),
                        "xy": np.stack(
                            [self._arrays["flat_x"][model_rows], self._arrays["flat_y"][model_rows]],
                            axis=-1,
                        ).astype(np.float32, copy=False),
                        "timestamps_ms": timestamps,
                        "pressure": self._arrays["flat_pressure"][model_rows].astype(np.float32, copy=True),
                        "size": self._arrays["flat_size"][model_rows].astype(np.float32, copy=True),
                        "start_offset_ms": start_offset,
                        "end_offset_ms": end_offset,
                        "source_key_index": int(self._arrays["key_index_in_event"][global_key_index]),
                    }
                )
                raw_keycodes.append(raw_code)
                pointer_start_offsets.append(start_offset)
                pointer_end_offsets.append(end_offset)
            if key_left < key_right:
                if int(self._arrays["key_touch_offsets"][key_left]) != left or int(self._arrays["key_touch_offsets"][key_right]) != right:
                    raise ValueError("keystroke key contacts do not exactly cover flat event rows")
            if sum(int(x["is_letter"]) for x in contacts) != int(self._arrays["n_letters"][i]):
                raise ValueError("restored letters contradict event n_letters")
            result["contacts"] = contacts
            result["metadata"].update(
                {
                    "raw_keycodes": raw_keycodes,
                    "keycode_token_offset": KEYCODE_TOKEN_OFFSET,
                    "contact_start_offsets_ms": pointer_start_offsets,
                    "contact_end_offsets_ms": pointer_end_offsets,
                }
            )
            return result

        pointer_values = self._arrays["flat_pointer_id"][left:right]
        # Android pointer ids are stable identifiers, not an ordering key.
        # Preserve slot order by first appearance (the primary ACTION_DOWN
        # pointer first), even for non-monotonic ids such as 9 then 2.
        pointer_ids = list(dict.fromkeys(int(value) for value in pointer_values.tolist()))
        expected_pointers = 2 if self.action == "pinch" else 1
        if len(pointer_ids) != expected_pointers:
            raise ValueError("%s event has %d pointers, expected %d" % (self.action, len(pointer_ids), expected_pointers))
        pointers = []
        start_offsets = []
        end_offsets = []
        for pointer_id in pointer_ids:
            local = np.flatnonzero(pointer_values == pointer_id) + left
            local = self._model_pointer_rows(local)
            if local.size < 2:
                raise ValueError("each pointer needs at least DOWN and UP samples")
            timestamps = self._arrays["flat_t_rel_ms"][local].astype(np.float32, copy=True)
            if np.any(np.diff(timestamps) < 0):
                raise ValueError("pointer timestamps go backwards")
            start_offset = float(timestamps[0])
            end_offset = float(timestamps[-1])
            if start_offset < 0 or end_offset > duration_ms + 1e-3:
                raise ValueError("pointer contact lies outside full event")
            pointer = {
                "pointer_id": pointer_id,
                "xy": np.stack([self._arrays["flat_x"][local], self._arrays["flat_y"][local]], axis=-1).astype(np.float32, copy=False),
                "timestamps_ms": timestamps,
                "pressure": self._arrays["flat_pressure"][local].astype(np.float32, copy=True),
                "size": self._arrays["flat_size"][local].astype(np.float32, copy=True),
                "action_code": self._arrays["flat_action_code"][local].astype(np.int8, copy=True),
                "frame_index": self._arrays["flat_frame_index"][local].astype(np.int32, copy=True),
                "active_mask": self._arrays["flat_active_mask"][local].astype(np.bool_, copy=True),
                "start_offset_ms": start_offset,
                "end_offset_ms": end_offset,
            }
            pointers.append(pointer)
            start_offsets.append(start_offset)
            end_offsets.append(end_offset)
        result["pointers"] = pointers
        result["metadata"].update(
            {
                "pointer_ids": pointer_ids,
                "pointer_start_offsets_ms": start_offsets,
                "pointer_end_offsets_ms": end_offsets,
                "model_pointer_row_count": int(sum(pointer["timestamps_ms"].size for pointer in pointers)),
            }
        )
        return result

    def canonical_sample(self, index: int) -> CanonicalTrajectory:
        sample = canonicalize_sample(self.raw_sample(index))
        if sample.n_keys != int(self._arrays["n_keys"][self._check_index(index)]):
            raise ValueError("canonical n_keys changed during restoration")
        if sample.n_letters != int(self._arrays["n_letters"][self._check_index(index)]):
            raise ValueError("canonical n_letters changed during restoration")
        return sample

    def audit(self, require_all_users: bool = False, validate_every_event: bool = False) -> Dict[str, Any]:
        split_counts: Dict[str, int] = {}
        split_users: Dict[str, list] = {}
        per_user_counts: Dict[str, Dict[str, int]] = {}
        missing_reference_users: Dict[str, list] = {}
        for split in ("train", "val", "test"):
            indices = self.indices_for_split(split)
            split_counts[split] = int(indices.size)
            present = sorted(set(int(x) for x in self.user_ids[indices].tolist()))
            split_users[split] = present
            counts = {str(user): int(self.indices_for_user(split, user).size) for user in self.splits.users(split)}
            per_user_counts[split] = counts
            missing_reference_users[split] = [int(user) for user, count in ((u, counts[str(u)]) for u in self.splits.users(split)) if count < 6]
            if require_all_users and missing_reference_users[split]:
                raise ValueError(
                    "%s/%s has users with fewer than target+5 real events: %s"
                    % (self.action, split, missing_reference_users[split])
                )
        validation_counts = Counter()
        if validate_every_event:
            for index in range(len(self)):
                sample = self.canonical_sample(index)
                validation_counts["events"] += 1
                validation_counts["pointer_streams"] += len(sample.pointer_features)
                validation_counts["keys"] += sample.n_keys
        offsets = self._arrays["event_offsets"]
        lengths = np.diff(offsets)
        raw_keycodes = self._arrays["keycode"]
        return {
            "schema_version": self.schema_version,
            "action": self.action,
            "source": {
                "npz": str(self.path),
                "size_bytes": int(self.path.stat().st_size),
                "sha256": self.sha256,
                "extraction_manifest": str(self.extraction_manifest_path) if self.extraction_manifest_path.is_file() else None,
                "extraction_manifest_sha256": self.extraction_manifest_sha256,
                "all_fields_allow_pickle_false": True,
                "object_array_count": 0,
            },
            "split": self.splits.audit_dict(),
            "counts": {
                "events": len(self),
                "flat_rows": int(self._arrays["flat_x"].size),
                "keys": int(raw_keycodes.size),
                "letters": int(self._arrays["n_letters"].sum()),
                "by_split": split_counts,
                "users_present_by_split": split_users,
                "per_user_by_split": per_user_counts,
            },
            "lengths": {
                "min_rows": int(lengths.min()) if lengths.size else 0,
                "max_rows": int(lengths.max()) if lengths.size else 0,
                "mean_rows": float(lengths.mean()) if lengths.size else 0.0,
            },
            "reference_gate": {
                "required_unique_refs": 5,
                "minimum_real_events_per_user_action_split": 6,
                "users_with_fewer_than_six": missing_reference_users,
                "require_all_users": bool(require_all_users),
            },
            "keycode_encoding": {
                "raw_min": int(raw_keycodes.min()) if raw_keycodes.size else None,
                "raw_max": int(raw_keycodes.max()) if raw_keycodes.size else None,
                "token_offset": KEYCODE_TOKEN_OFFSET,
                "raw_values_preserved_in_metadata": True,
            },
            "pointer_timing": {
                "timestamps_relative_to_complete_event_origin": True,
                "per_pointer_start_end_offsets_preserved": True,
                "pinch_pointers_forced_to_common_zero_or_duration": False,
            },
            "full_event_validation": dict(validation_counts),
            "all_field_dtypes": dict(sorted(self._all_field_dtypes.items())),
        }


def audit_corpus_directory(
    corpus_dir: Path,
    split_path: Path = FORMAL_SPLIT_PATH,
    require_pinned_split: bool = True,
    require_all_users: bool = True,
    validate_every_event: bool = True,
) -> Dict[str, Any]:
    corpus_dir = Path(corpus_dir).resolve()
    split = SplitDefinition.load(split_path, require_pinned_hash=require_pinned_split)
    action_audits: Dict[str, Any] = {}
    total_events = 0
    total_rows = 0
    total_keys = 0
    for action in ACTIONS:
        path = corpus_dir / ("hmog_trajectory_%s.npz" % action)
        corpus = NumericTrajectoryCorpus(path, split, expected_action=action)
        audit = corpus.audit(
            require_all_users=require_all_users,
            validate_every_event=validate_every_event,
        )
        action_audits[action] = audit
        total_events += int(audit["counts"]["events"])
        total_rows += int(audit["counts"]["flat_rows"])
        total_keys += int(audit["counts"]["keys"])
    return {
        "protocol": "strict_five_action_trajectory_corpus_v1",
        "corpus_dir": str(corpus_dir),
        "split": split.audit_dict(),
        "actions": action_audits,
        "totals": {"events": total_events, "flat_rows": total_rows, "keys": total_keys},
        "formal_no_sample_cap": True,
        "passed": True,
    }
