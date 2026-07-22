"""Shared, auditable event plans for paired trajectory and IMU generation.

An :class:`EventPlan` is resolved *before* either modality is sampled.  It is
therefore the cross-modal source of truth for action, user, logical event
time, orientation, touch geometry, key text/tokens and pointer topology.

The formal 100k trajectory benchmark still uses condition_source_code=2
(five references plus a train-user-only prior).  Explicit Android requests
are bound with condition_source_code=3 and cannot silently enter that formal
benchmark, whose ingress audit requires code 2.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from trajectory.data import ACTIONS, KEYCODE_TOKEN_MAX, ORIENTATION_IDS, CanonicalTrajectory

from .protocol import (
    ConditionRequest, ReferenceConditionPolicy, TrainGlobalPrior,
    ddim_noise_seed,
)


EVENT_PLAN_SCHEMA = "paired_event_plan_v1"
EXTERNAL_CONDITION_SOURCE_CODE = 3
EXTERNAL_KEY_ENDPOINT_SOURCE_CODE = 4


def _xy_matrix(value: Optional[Sequence[Sequence[float]]], active_pointers: int, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if active_pointers == 1 and array.shape == (2,):
        result = np.zeros((2, 2), dtype=np.float32)
        result[0] = array
        array = result
    if array.shape != (2, 2):
        raise ValueError("%s must be [x,y] for one pointer or [[x0,y0],[x1,y1]] for pinch" % name)
    if not np.all(np.isfinite(array)):
        raise ValueError("%s contains non-finite coordinates" % name)
    if active_pointers == 1 and np.any(array[1] != 0):
        raise ValueError("%s populated inactive pointer slot 1" % name)
    return array


def _offset_vector(value: Optional[Sequence[float]], active_pointers: int, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == active_pointers:
        result = np.zeros(2, dtype=np.float32)
        result[:active_pointers] = array
        array = result
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError("%s must contain one value per active pointer" % name)
    if active_pointers == 1 and float(array[1]) != 0.0:
        raise ValueError("%s populated inactive pointer slot 1" % name)
    return array


def _integer_ms(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("%s must be finite and > 0" % name)
    if abs(result - round(result)) > 1.0e-6:
        raise ValueError("%s must lie on the trajectory model's integer-ms lattice" % name)
    return float(round(result))


def _seed_for_domain(plan_seed: int, domain: str) -> int:
    digest = hashlib.sha256(("%d|%s" % (int(plan_seed), domain)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


@dataclass(frozen=True)
class EventPlan:
    """Fully resolved cross-modal event contract.

    Arrays are serialized as immutable tuples so the canonical SHA-256 is
    stable across processes and can be copied into both modality outputs.
    ``lengths`` describes trajectory touch tokens; IMU keeps its own 100 Hz
    buffer length while sharing the logical ``duration_ms``.
    """

    sample_id: str
    action: str
    user_id: int
    split: str
    fake_id: int
    sample_index: int
    duration_ms: float
    start_time_ns: Optional[int]
    orientation_id: int
    start_xy: Tuple[Tuple[float, float], Tuple[float, float]]
    end_xy: Tuple[Tuple[float, float], Tuple[float, float]]
    pointer_start_offset_ms: Tuple[float, float]
    pointer_end_offset_ms: Tuple[float, float]
    lengths: Tuple[int, int]
    n_keys: int
    n_letters: int
    keycodes: Tuple[int, ...]
    text: Optional[str]
    zero_flight_after_key: Tuple[int, ...]
    zero_flight_probability: float
    contact_masks: Tuple[Tuple[int, ...], Tuple[int, ...]]
    event_ids: Tuple[Tuple[int, ...], Tuple[int, ...]]
    key_endpoint_source_code: Tuple[int, int]
    condition_seed: int
    trajectory_noise_seed: int
    imu_noise_seed: int
    condition_source_code: int
    screen_min_xy: Tuple[float, float]
    screen_max_xy: Tuple[float, float]
    reference_ids: Tuple[int, ...]
    reference_canonical_sha256: Tuple[str, ...]
    carrier_ref_id: int
    train_prior_digest: str

    def __post_init__(self) -> None:
        self.validate()

    @property
    def pointer_count(self) -> int:
        return 2 if self.action == "pinch" else 1

    @property
    def plan_sha256(self) -> str:
        payload = self.to_dict(include_digest=False)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, include_digest: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": EVENT_PLAN_SCHEMA,
            "sample_id": self.sample_id,
            "action": self.action,
            "user_id": int(self.user_id),
            "split": self.split,
            "fake_id": int(self.fake_id),
            "sample_index": int(self.sample_index),
            "duration_ms": float(self.duration_ms),
            "start_time_ns": None if self.start_time_ns is None else int(self.start_time_ns),
            "orientation_id": int(self.orientation_id),
            "start_xy": [list(row) for row in self.start_xy],
            "end_xy": [list(row) for row in self.end_xy],
            "pointer_start_offset_ms": list(self.pointer_start_offset_ms),
            "pointer_end_offset_ms": list(self.pointer_end_offset_ms),
            "lengths": list(self.lengths),
            "n_keys": int(self.n_keys),
            "n_letters": int(self.n_letters),
            "keycodes": list(self.keycodes),
            "text": self.text,
            "zero_flight_after_key": list(self.zero_flight_after_key),
            "zero_flight_probability": float(self.zero_flight_probability),
            "contact_masks": [list(row) for row in self.contact_masks],
            "event_ids": [list(row) for row in self.event_ids],
            "key_endpoint_source_code": list(self.key_endpoint_source_code),
            "condition_seed": int(self.condition_seed),
            "trajectory_noise_seed": int(self.trajectory_noise_seed),
            "imu_noise_seed": int(self.imu_noise_seed),
            "condition_source_code": int(self.condition_source_code),
            "screen_min_xy": list(self.screen_min_xy),
            "screen_max_xy": list(self.screen_max_xy),
            "reference_ids": list(self.reference_ids),
            "reference_canonical_sha256": list(self.reference_canonical_sha256),
            "carrier_ref_id": int(self.carrier_ref_id),
            "train_prior_digest": self.train_prior_digest,
        }
        if include_digest:
            payload["plan_sha256"] = self.plan_sha256
        return payload

    def to_imu_kwargs(self) -> Dict[str, Any]:
        """Return conditions accepted by ``AndroidIMUDiffusionLayer.generate``."""

        result: Dict[str, Any] = {
            "user_id": int(self.user_id),
            "duration_ms": float(self.duration_ms),
            "orientation_id": int(self.orientation_id),
            "start_time_ns": self.start_time_ns,
            "noise_seed": int(self.imu_noise_seed),
        }
        if self.action in ("tap", "scroll", "swipe"):
            result["xy_start"] = tuple(self.start_xy[0])
            result["xy_end"] = tuple(self.end_xy[0])
        elif self.action == "pinch":
            start = np.asarray(self.start_xy[:2], dtype=np.float64)
            end = np.asarray(self.end_xy[:2], dtype=np.float64)
            result.update(
                xy_start=tuple(np.mean(start, axis=0).tolist()),
                xy_end=tuple(np.mean(end, axis=0).tolist()),
                pinch_start_span=float(np.linalg.norm(start[1] - start[0])),
                pinch_end_span=float(np.linalg.norm(end[1] - end[0])),
            )
        else:
            result.update(
                text=self.text or "".join(chr(value) for value in self.keycodes),
                n_keys=int(self.n_keys),
                n_letters=int(self.n_letters),
            )
        return result

    def validate(self) -> None:
        if not self.sample_id or self.action not in ACTIONS or self.split not in ("train", "val", "test"):
            raise ValueError("event plan has invalid identity/action/split")
        if int(self.user_id) < 0 or int(self.user_id) != self.user_id:
            raise ValueError("event plan user_id must be a non-negative integer")
        if int(self.fake_id) < 0 or int(self.sample_index) < 0:
            raise ValueError("event plan fake_id/sample_index must be non-negative")
        _integer_ms(self.duration_ms, "duration_ms")
        if self.start_time_ns is not None and (int(self.start_time_ns) != self.start_time_ns or int(self.start_time_ns) < 0):
            raise ValueError("start_time_ns must be a non-negative integer or None")
        if int(self.orientation_id) not in ORIENTATION_IDS:
            raise ValueError("orientation_id must be one of %r" % (ORIENTATION_IDS,))
        start = np.asarray(self.start_xy, dtype=np.float64)
        end = np.asarray(self.end_xy, dtype=np.float64)
        low = np.asarray(self.screen_min_xy, dtype=np.float64)
        high = np.asarray(self.screen_max_xy, dtype=np.float64)
        starts = np.asarray(self.pointer_start_offset_ms, dtype=np.float64)
        ends = np.asarray(self.pointer_end_offset_ms, dtype=np.float64)
        lengths = np.asarray(self.lengths, dtype=np.int64)
        if start.shape != (2, 2) or end.shape != (2, 2) or low.shape != (2,) or high.shape != (2,):
            raise ValueError("event plan geometry has the wrong shape")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)) or np.any(high <= low):
            raise ValueError("event plan geometry is non-finite or has invalid bounds")
        if starts.shape != (2,) or ends.shape != (2,) or lengths.shape != (2,):
            raise ValueError("event plan pointer structure has the wrong shape")
        active = self.pointer_count
        if np.any(lengths[:active] < 2) or np.any(lengths[active:] != 0):
            raise ValueError("event plan point counts contradict action pointer count")
        if np.any(starts[:active] < 0) or np.any(ends[:active] > self.duration_ms) or np.any(starts[:active] >= ends[:active]):
            raise ValueError("event plan pointer lifetimes are outside event duration")
        if float(np.min(starts[:active])) != 0.0 or float(np.max(ends[:active])) != float(self.duration_ms):
            raise ValueError("pointer lifetime union must equal [0,duration_ms]")
        if np.any(np.abs(starts - np.rint(starts)) > 1.0e-6) or np.any(np.abs(ends - np.rint(ends)) > 1.0e-6):
            raise ValueError("pointer lifetimes must lie on integer milliseconds")
        if active == 2 and float(np.max(starts[:2])) >= float(np.min(ends[:2])):
            raise ValueError("pinch pointer lifetimes must overlap")
        if np.any(start[:active] < low) or np.any(start[:active] > high) or np.any(end[:active] < low) or np.any(end[:active] > high):
            raise ValueError("event plan endpoints are outside train-only screen bounds")
        if active == 1 and (np.any(start[1] != 0) or np.any(end[1] != 0)):
            raise ValueError("single-pointer event populated pointer slot 1")
        if self.action == "keystroke":
            if self.n_keys < 1 or len(self.keycodes) != self.n_keys or not 0 <= self.n_letters <= self.n_keys:
                raise ValueError("keystroke plan has inconsistent key counts")
            if any(value < 0 or value > KEYCODE_TOKEN_MAX for value in self.keycodes):
                raise ValueError("keystroke keycode is outside the model vocabulary")
            if self.text is not None and tuple(ord(ch) for ch in self.text) != tuple(self.keycodes):
                raise ValueError("text and keycodes are not the same explicit sequence")
            if len(self.zero_flight_after_key) != max(self.n_keys - 1, 0) or any(
                value not in (0, 1) for value in self.zero_flight_after_key
            ):
                raise ValueError("keystroke zero-flight topology has the wrong length/value")
        elif self.n_keys != 0 or self.n_letters != 0 or self.keycodes or self.text is not None:
            raise ValueError("non-keystroke event carries key metadata")
        elif self.zero_flight_after_key:
            raise ValueError("non-keystroke event carries zero-flight topology")
        if not math.isfinite(float(self.zero_flight_probability)) or not 0.0 <= float(self.zero_flight_probability) <= 1.0:
            raise ValueError("zero_flight_probability must lie in [0,1]")
        if len(self.contact_masks) != 2 or len(self.event_ids) != 2:
            raise ValueError("event plan topology must contain two canonical pointer slots")
        for pointer in range(2):
            if len(self.contact_masks[pointer]) != int(self.lengths[pointer]) or len(self.event_ids[pointer]) != int(self.lengths[pointer]):
                raise ValueError("event plan topology length disagrees with pointer lengths")
            if any(value not in (0, 1) for value in self.contact_masks[pointer]):
                raise ValueError("contact mask must be binary")
        if len(self.key_endpoint_source_code) != 2:
            raise ValueError("key endpoint provenance must contain two values")
        if len(self.reference_ids) != 5 or len(set(self.reference_ids)) != 5:
            raise ValueError("event plan must bind exactly five unique references")
        if len(self.reference_canonical_sha256) != 5 or any(len(value) != 64 for value in self.reference_canonical_sha256):
            raise ValueError("event plan must bind five canonical reference digests")
        if int(self.carrier_ref_id) not in self.reference_ids or len(self.train_prior_digest) != 64:
            raise ValueError("event plan reference/prior provenance is incomplete")
        for seed_name, value in (
            ("condition_seed", self.condition_seed),
            ("trajectory_noise_seed", self.trajectory_noise_seed),
            ("imu_noise_seed", self.imu_noise_seed),
        ):
            if int(value) != value or not 0 <= int(value) < (1 << 63):
                raise ValueError("%s must be a non-negative 63-bit integer" % seed_name)
        if len({int(self.condition_seed), int(self.trajectory_noise_seed), int(self.imu_noise_seed)}) != 3:
            raise ValueError("condition/trajectory/IMU seed domains must be disjoint")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EventPlan":
        if payload.get("schema_version") != EVENT_PLAN_SCHEMA:
            raise ValueError("event plan schema mismatch")
        expected_digest = payload.get("plan_sha256")
        plan = cls(
            sample_id=str(payload["sample_id"]), action=str(payload["action"]),
            user_id=int(payload["user_id"]), split=str(payload["split"]),
            fake_id=int(payload["fake_id"]), sample_index=int(payload["sample_index"]),
            duration_ms=float(payload["duration_ms"]),
            start_time_ns=None if payload.get("start_time_ns") is None else int(payload["start_time_ns"]),
            orientation_id=int(payload["orientation_id"]),
            start_xy=tuple(tuple(float(x) for x in row) for row in payload["start_xy"]),
            end_xy=tuple(tuple(float(x) for x in row) for row in payload["end_xy"]),
            pointer_start_offset_ms=tuple(float(x) for x in payload["pointer_start_offset_ms"]),
            pointer_end_offset_ms=tuple(float(x) for x in payload["pointer_end_offset_ms"]),
            lengths=tuple(int(x) for x in payload["lengths"]),
            n_keys=int(payload["n_keys"]), n_letters=int(payload["n_letters"]),
            keycodes=tuple(int(x) for x in payload["keycodes"]),
            text=None if payload.get("text") is None else str(payload["text"]),
            zero_flight_after_key=tuple(int(x) for x in payload["zero_flight_after_key"]),
            zero_flight_probability=float(payload["zero_flight_probability"]),
            contact_masks=tuple(tuple(int(x) for x in row) for row in payload["contact_masks"]),
            event_ids=tuple(tuple(int(x) for x in row) for row in payload["event_ids"]),
            key_endpoint_source_code=tuple(int(x) for x in payload["key_endpoint_source_code"]),
            condition_seed=int(payload["condition_seed"]),
            trajectory_noise_seed=int(payload["trajectory_noise_seed"]),
            imu_noise_seed=int(payload["imu_noise_seed"]),
            condition_source_code=int(payload["condition_source_code"]),
            screen_min_xy=tuple(float(x) for x in payload["screen_min_xy"]),
            screen_max_xy=tuple(float(x) for x in payload["screen_max_xy"]),
            reference_ids=tuple(int(x) for x in payload["reference_ids"]),
            reference_canonical_sha256=tuple(str(x) for x in payload["reference_canonical_sha256"]),
            carrier_ref_id=int(payload["carrier_ref_id"]),
            train_prior_digest=str(payload["train_prior_digest"]),
        )
        if expected_digest is not None and str(expected_digest) != plan.plan_sha256:
            raise ValueError("event plan SHA-256 mismatch")
        return plan

    @classmethod
    def from_condition_request(
        cls,
        request: ConditionRequest,
        *,
        sample_id: str,
        start_time_ns: Optional[int] = None,
        text: Optional[str] = None,
    ) -> "EventPlan":
        return cls(
            sample_id=str(sample_id), action=request.action, user_id=int(request.user_id),
            split=request.split, fake_id=int(request.fake_id), sample_index=int(request.sample_index),
            duration_ms=float(request.duration_ms),
            start_time_ns=None if start_time_ns is None else int(start_time_ns),
            orientation_id=int(request.orientation_id),
            start_xy=tuple(tuple(float(x) for x in row) for row in request.start_xy),
            end_xy=tuple(tuple(float(x) for x in row) for row in request.end_xy),
            pointer_start_offset_ms=tuple(float(x) for x in request.pointer_start_offset_ms),
            pointer_end_offset_ms=tuple(float(x) for x in request.pointer_end_offset_ms),
            lengths=tuple(int(x) for x in request.lengths),
            n_keys=int(request.n_keys), n_letters=int(request.n_letters),
            keycodes=tuple(int(x) for x in request.keycodes), text=text,
            zero_flight_after_key=tuple(int(x) for x in request.zero_flight_after_key),
            zero_flight_probability=float(request.zero_flight_probability),
            contact_masks=tuple(tuple(int(x) for x in row) for row in request.contact_masks),
            event_ids=tuple(tuple(int(x) for x in row) for row in request.event_ids),
            key_endpoint_source_code=tuple(int(x) for x in request.key_endpoint_source_code),
            condition_seed=int(request.seed),
            # Record the exact seed consumed by the formal trajectory DDIM
            # sampler.  This makes EventPlan provenance describe the sampled
            # waveform, rather than a merely nominal trajectory seed.
            trajectory_noise_seed=ddim_noise_seed(
                request.seed, request.action, request.user_id,
                request.sample_index,
            ),
            imu_noise_seed=_seed_for_domain(request.seed, "imu"),
            condition_source_code=int(request.condition_source_code),
            screen_min_xy=tuple(float(x) for x in request.screen_min_xy),
            screen_max_xy=tuple(float(x) for x in request.screen_max_xy),
            reference_ids=tuple(int(x) for x in request.reference_ids),
            reference_canonical_sha256=tuple(str(x) for x in request.reference_canonical_sha256),
            carrier_ref_id=int(request.carrier_ref_id),
            train_prior_digest=str(request.train_prior_digest),
        )

    def to_condition_request(self) -> ConditionRequest:
        """Rebuild the exact trajectory-model condition after persistence."""

        request = ConditionRequest(
            action=self.action, user_id=int(self.user_id), split=self.split,
            fake_id=int(self.fake_id), sample_index=int(self.sample_index), seed=int(self.condition_seed),
            reference_ids=tuple(int(x) for x in self.reference_ids),
            reference_canonical_sha256=tuple(self.reference_canonical_sha256),
            carrier_ref_id=int(self.carrier_ref_id), lengths=tuple(int(x) for x in self.lengths),
            duration_ms=float(self.duration_ms), orientation_id=int(self.orientation_id),
            start_xy=np.asarray(self.start_xy, np.float32), end_xy=np.asarray(self.end_xy, np.float32),
            pinch_span=np.asarray([
                np.linalg.norm(np.asarray(self.start_xy[1]) - np.asarray(self.start_xy[0])),
                np.linalg.norm(np.asarray(self.end_xy[1]) - np.asarray(self.end_xy[0])),
            ], np.float32) if self.action == "pinch" else np.zeros(2, np.float32),
            pinch_angle=np.asarray([
                math.atan2(self.start_xy[1][1] - self.start_xy[0][1], self.start_xy[1][0] - self.start_xy[0][0]),
                math.atan2(self.end_xy[1][1] - self.end_xy[0][1], self.end_xy[1][0] - self.end_xy[0][0]),
            ], np.float32) if self.action == "pinch" else np.zeros(2, np.float32),
            pointer_start_offset_ms=np.asarray(self.pointer_start_offset_ms, np.float32),
            pointer_end_offset_ms=np.asarray(self.pointer_end_offset_ms, np.float32),
            n_keys=int(self.n_keys), n_letters=int(self.n_letters),
            keycodes=np.asarray(self.keycodes, np.int64),
            zero_flight_after_key=np.asarray(self.zero_flight_after_key, np.bool_),
            zero_flight_probability=float(self.zero_flight_probability),
            key_endpoint_source_code=np.asarray(self.key_endpoint_source_code, np.int8),
            contact_masks=tuple(np.asarray(row, np.bool_) for row in self.contact_masks),
            event_ids=tuple(np.asarray(row, np.int64) for row in self.event_ids),
            condition_source_code=int(self.condition_source_code),
            train_prior_digest=self.train_prior_digest,
            screen_min_xy=np.asarray(self.screen_min_xy, np.float32),
            screen_max_xy=np.asarray(self.screen_max_xy, np.float32),
        )
        ReferenceConditionPolicy.validate_request(request)
        return request


def bind_explicit_event_conditions(
    base: ConditionRequest,
    prior: TrainGlobalPrior,
    references: Sequence[CanonicalTrajectory],
    *,
    duration_ms: Optional[float] = None,
    start_xy: Optional[Sequence[Sequence[float]]] = None,
    end_xy: Optional[Sequence[Sequence[float]]] = None,
    pointer_start_offset_ms: Optional[Sequence[float]] = None,
    pointer_end_offset_ms: Optional[Sequence[float]] = None,
) -> ConditionRequest:
    """Bind caller time/geometry to a five-shot request without test lookup.

    The request must already have been sampled with the caller's requested
    orientation and, for keystroke, explicit keycodes.  Point counts are
    rescaled from the five-shot request's irregular sampling rate; no fixed-Hz
    touch trajectory is invented.
    """

    if prior.action != base.action or len(references) != 5:
        raise ValueError("explicit binding requires matching prior and exactly five refs")
    if any(ref.action != base.action or ref.user_id != base.user_id or ref.split != base.split for ref in references):
        raise ValueError("explicit binding reference provenance mismatch")
    active = 2 if base.action == "pinch" else 1
    duration = float(base.duration_ms) if duration_ms is None else _integer_ms(duration_ms, "duration_ms")
    prior_low = int(math.ceil(float(np.min(prior.duration_ms))))
    prior_high = int(math.floor(float(np.max(prior.duration_ms))))
    if duration < prior_low or duration > prior_high:
        raise ValueError(
            "duration_ms=%.0f is outside the train-only model support [%d,%d]; split/stitch the event instead of extrapolating"
            % (duration, prior_low, prior_high)
        )

    requested_starts = _offset_vector(pointer_start_offset_ms, active, "pointer_start_offset_ms")
    requested_ends = _offset_vector(pointer_end_offset_ms, active, "pointer_end_offset_ms")
    if (requested_starts is None) != (requested_ends is None):
        raise ValueError("pointer start/end offsets must be supplied together")
    if requested_starts is None:
        if base.action == "keystroke":
            pointer_starts = np.asarray([0.0, 0.0], dtype=np.float32)
            pointer_ends = np.asarray([duration, 0.0], dtype=np.float32)
        else:
            pointer_starts = np.rint(np.asarray(base.pointer_start_offset_ms) / base.duration_ms * duration).astype(np.float32)
            pointer_ends = np.rint(np.asarray(base.pointer_end_offset_ms) / base.duration_ms * duration).astype(np.float32)
            pointer_starts[active:] = 0.0
            pointer_ends[active:] = 0.0
            pointer_starts[int(np.argmin(pointer_starts[:active]))] = 0.0
            pointer_ends[int(np.argmax(pointer_ends[:active]))] = duration
    else:
        pointer_starts = np.rint(requested_starts).astype(np.float32)
        pointer_ends = np.rint(requested_ends).astype(np.float32)
        if np.any(np.abs(requested_starts - pointer_starts) > 1.0e-6) or np.any(np.abs(requested_ends - pointer_ends) > 1.0e-6):
            raise ValueError("explicit pointer lifetimes must lie on integer milliseconds")

    if base.action == "keystroke":
        lengths = tuple(int(x) for x in base.lengths)
        positive_intervals = lengths[0] - 1 - int(np.sum(base.zero_flight_after_key))
        if positive_intervals > int(duration):
            raise ValueError("requested keystroke duration is too short for its DOWN/MOVE/UP topology")
        contacts = base.contact_masks
        events = base.event_ids
    else:
        length_values = []
        contacts_list = []
        events_list = []
        for pointer in range(active):
            old_lifetime = float(base.pointer_end_offset_ms[pointer] - base.pointer_start_offset_ms[pointer])
            new_lifetime = float(pointer_ends[pointer] - pointer_starts[pointer])
            if new_lifetime < 1.0:
                raise ValueError("explicit pointer lifetime must be at least 1 ms")
            old_intervals = max(int(base.lengths[pointer]) - 1, 1)
            intervals = int(round(old_intervals * new_lifetime / max(old_lifetime, 1.0)))
            intervals = max(1, min(intervals, int(new_lifetime)))
            length = intervals + 1
            length_values.append(length)
            contacts_list.append(np.ones(length, dtype=np.bool_))
            events_list.append(np.zeros(length, dtype=np.int64))
        while len(length_values) < 2:
            length_values.append(0)
            contacts_list.append(np.zeros(0, dtype=np.bool_))
            events_list.append(np.zeros(0, dtype=np.int64))
        lengths = tuple(length_values)
        contacts = tuple(contacts_list)
        events = tuple(events_list)

    resolved_start = _xy_matrix(start_xy, active, "start_xy")
    resolved_end = _xy_matrix(end_xy, active, "end_xy")
    if (resolved_start is None) != (resolved_end is None):
        raise ValueError("start_xy and end_xy must be supplied together")
    if resolved_start is None:
        resolved_start = np.asarray(base.start_xy, dtype=np.float32).copy()
        resolved_end = np.asarray(base.end_xy, dtype=np.float32).copy()
    low = np.asarray(base.screen_min_xy, dtype=np.float32)
    high = np.asarray(base.screen_max_xy, dtype=np.float32)
    if np.any(resolved_start[:active] < low) or np.any(resolved_start[:active] > high) or np.any(resolved_end[:active] < low) or np.any(resolved_end[:active] > high):
        raise ValueError("explicit XY lies outside train-only bounds for the selected orientation")

    span = np.zeros(2, dtype=np.float32)
    angle = np.zeros(2, dtype=np.float32)
    if base.action == "pinch":
        for endpoint, points in enumerate((resolved_start[:2], resolved_end[:2])):
            vector = points[1] - points[0]
            span[endpoint] = float(np.linalg.norm(vector))
            angle[endpoint] = float(math.atan2(float(vector[1]), float(vector[0])))
        if np.any(span <= 0):
            raise ValueError("pinch endpoints must define two non-coincident fingers")

    endpoint_source = np.asarray(base.key_endpoint_source_code, dtype=np.int8).copy()
    if base.action == "keystroke" and start_xy is not None:
        endpoint_source[:] = EXTERNAL_KEY_ENDPOINT_SOURCE_CODE
    bound = replace(
        base,
        duration_ms=float(duration),
        start_xy=resolved_start,
        end_xy=resolved_end,
        pinch_span=span,
        pinch_angle=angle,
        pointer_start_offset_ms=pointer_starts,
        pointer_end_offset_ms=pointer_ends,
        lengths=(int(lengths[0]), int(lengths[1])),
        contact_masks=(np.asarray(contacts[0], np.bool_), np.asarray(contacts[1], np.bool_)),
        event_ids=(np.asarray(events[0], np.int64), np.asarray(events[1], np.int64)),
        key_endpoint_source_code=endpoint_source,
        condition_source_code=EXTERNAL_CONDITION_SOURCE_CODE,
    )
    ReferenceConditionPolicy.validate_request(bound)
    return bound


__all__ = [
    "EVENT_PLAN_SCHEMA", "EXTERNAL_CONDITION_SOURCE_CODE",
    "EXTERNAL_KEY_ENDPOINT_SOURCE_CODE", "EventPlan",
    "bind_explicit_event_conditions",
]
