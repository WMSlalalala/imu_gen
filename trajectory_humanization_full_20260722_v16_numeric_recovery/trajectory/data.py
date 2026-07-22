"""Canonical data and leakage-safe five-shot construction for trajectories."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .features import (
    KEYCODE_TOKEN_MAX,
    KEYCODE_VOCAB_SIZE,
    is_hmog_ascii_letter_keycode,
)


ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
SPLITS = ("train", "val", "test")
ORIENTATION_IDS = (-1, 0, 1, 3)
FEATURE_NAMES = ("progress", "lateral", "log_dt", "pressure", "size")
FEATURE_DIM = len(FEATURE_NAMES)
LOG_DT_INDEX = 2
MAX_POINTERS = 2
BASE_DT_MS = 10.0
FORMAL_REF_COUNT = 5
# Re-export the shared constants for the existing training API.  Their single
# definition lives in ``trajectory.features`` so generation and both PAD
# families cannot silently drift to different representations.


def keystroke_zero_flight_flags(
    contact_mask: np.ndarray,
    event_ids: np.ndarray,
    n_keys: int,
) -> np.ndarray:
    """Return one flag per inter-key boundary from lossless topology.

    HMOG timestamps are integer milliseconds.  A key UP and the next key DOWN
    can therefore be two ordered MotionEvents with the same timestamp.  Such a
    zero-flight boundary is represented by adjacent contact tokens belonging
    to consecutive key events.  A positive flight is represented by one or
    more explicit no-contact tokens.  No timestamp or interpolated XY is
    needed to recover this distinction.
    """

    contact = np.asarray(contact_mask, dtype=np.bool_).reshape(-1)
    events = np.asarray(event_ids, dtype=np.int64).reshape(-1)
    count = int(n_keys)
    if contact.shape != events.shape or count < 1:
        raise ValueError("keystroke topology needs matching vectors and n_keys>=1")
    if np.any(events[contact] < 0) or np.any(events[~contact] != -1):
        raise ValueError("keystroke topology has invalid contact/event ids")
    flags = np.zeros(max(count - 1, 0), dtype=np.bool_)
    previous_last = -1
    for event_id in range(count):
        positions = np.flatnonzero(contact & (events == event_id))
        if positions.size < 2 or np.any(np.diff(positions) != 1):
            raise ValueError("each key must be one contiguous DOWN..UP contact")
        first, last = int(positions[0]), int(positions[-1])
        if first <= previous_last:
            raise ValueError("key events are not ordered")
        if event_id:
            between = slice(previous_last + 1, first)
            if np.any(contact[between]) or np.any(events[between] != -1):
                raise ValueError("inter-key flight contains an invalid contact/event")
            flags[event_id - 1] = first == previous_last + 1
        previous_last = last
    observed = sorted(set(int(value) for value in events[contact].tolist()))
    if observed != list(range(count)):
        raise ValueError("keystroke event ids must be exactly 0..n_keys-1")
    return flags


@dataclass
class CanonicalTrajectory:
    action: str
    pointer_features: List[np.ndarray]
    pointer_contact_masks: List[np.ndarray]
    pointer_event_ids: List[np.ndarray]
    duration_ms: float
    pointer_start_offset_ms: np.ndarray
    pointer_end_offset_ms: np.ndarray
    orientation_id: int
    start_xy: np.ndarray
    end_xy: np.ndarray
    pinch_span: np.ndarray
    pinch_angle: np.ndarray
    n_keys: int
    n_letters: int
    keycodes: np.ndarray
    sample_id: str
    user_id: int
    split: str
    is_real: bool
    metadata: Dict[str, Any]


@dataclass
class FewShotExample:
    target: CanonicalTrajectory
    references: List[CanonicalTrajectory]


@dataclass
class TrajectoryBatch:
    """Action-homogeneous target batch plus its leakage-audited reference set."""

    action: str
    features: torch.Tensor                 # [B,2,T,5]
    point_mask: torch.Tensor               # valid timeline, including contact gaps
    contact_mask: torch.Tensor             # emitted touch contact points only
    event_ids: torch.Tensor                # contact id, -1 for gap/padding
    pointer_mask: torch.Tensor              # [B,2]
    duration_ms: torch.Tensor
    pointer_start_offset_ms: torch.Tensor   # [B,2], global event timeline
    pointer_end_offset_ms: torch.Tensor     # [B,2], global event timeline
    orientation_id: torch.Tensor
    start_xy: torch.Tensor
    end_xy: torch.Tensor
    pinch_span: torch.Tensor
    pinch_angle: torch.Tensor
    n_keys: torch.Tensor
    n_letters: torch.Tensor
    keycodes: torch.Tensor
    keycode_mask: torch.Tensor
    ref_features: torch.Tensor             # [B,R,2,Tr,5]
    ref_point_mask: torch.Tensor            # [B,R,2,Tr]
    ref_contact_mask: torch.Tensor
    ref_event_ids: torch.Tensor
    ref_pointer_mask: torch.Tensor          # [B,R,2]
    ref_pointer_start_offset_ms: torch.Tensor  # [B,R,2]
    ref_pointer_end_offset_ms: torch.Tensor    # [B,R,2]
    ref_mask: torch.Tensor                  # [B,R]
    ref_keycodes: torch.Tensor              # [B,R,Kr]
    ref_keycode_mask: torch.Tensor
    target_sample_ids: Tuple[str, ...]
    target_user_ids: Tuple[int, ...]
    target_splits: Tuple[str, ...]
    ref_sample_ids: Tuple[Tuple[str, ...], ...]
    ref_user_ids: Tuple[Tuple[int, ...], ...]
    ref_splits: Tuple[Tuple[str, ...], ...]

    def to(self, device: torch.device) -> "TrajectoryBatch":
        tensor_names = (
            "features", "point_mask", "contact_mask", "event_ids", "pointer_mask",
            "duration_ms", "pointer_start_offset_ms", "pointer_end_offset_ms",
            "orientation_id", "start_xy", "end_xy", "pinch_span",
            "pinch_angle", "n_keys", "n_letters", "keycodes", "keycode_mask", "ref_features",
            "ref_point_mask", "ref_contact_mask", "ref_event_ids", "ref_pointer_mask",
            "ref_pointer_start_offset_ms", "ref_pointer_end_offset_ms", "ref_mask",
            "ref_keycodes", "ref_keycode_mask",
        )
        values = {name: getattr(self, name).to(device) for name in tensor_names}
        return TrajectoryBatch(
            action=self.action,
            target_sample_ids=self.target_sample_ids,
            target_user_ids=self.target_user_ids,
            target_splits=self.target_splits,
            ref_sample_ids=self.ref_sample_ids,
            ref_user_ids=self.ref_user_ids,
            ref_splits=self.ref_splits,
            **values
        )

    def with_features(self, features: torch.Tensor) -> "TrajectoryBatch":
        values = dict(self.__dict__)
        values["features"] = features
        return TrajectoryBatch(**values)

    @property
    def feature_mask(self) -> torch.Tensor:
        # Geometry/contact attributes exist only while the finger is down;
        # log_dt is also supervised on a valid gap slot to learn flight time.
        mask = self.contact_mask.unsqueeze(-1).expand_as(self.features).clone()
        mask[..., LOG_DT_INDEX] = self.point_mask
        return mask

    @property
    def ref_feature_mask(self) -> torch.Tensor:
        mask = self.ref_contact_mask.unsqueeze(-1).expand_as(self.ref_features).clone()
        mask[..., LOG_DT_INDEX] = self.ref_point_mask
        return mask & self.ref_mask[:, :, None, None, None]

    def validate(self, require_references: bool = False, expected_refs: int = FORMAL_REF_COUNT) -> None:
        if self.action not in ACTIONS:
            raise ValueError("unsupported action: %r" % self.action)
        if self.features.dim() != 4 or self.features.shape[1] != MAX_POINTERS or self.features.shape[-1] != FEATURE_DIM:
            raise ValueError("features must have shape [B,2,T,5]")
        if not torch.is_floating_point(self.features) or not torch.all(torch.isfinite(self.features)):
            raise ValueError("features must be finite floating point")
        b, p, t, _ = self.features.shape
        shapes = {
            "point_mask": (b, p, t), "contact_mask": (b, p, t), "event_ids": (b, p, t),
            "pointer_mask": (b, p), "duration_ms": (b,), "orientation_id": (b,),
            "pointer_start_offset_ms": (b, p), "pointer_end_offset_ms": (b, p),
            "start_xy": (b, p, 2), "end_xy": (b, p, 2), "pinch_span": (b, 2),
            "pinch_angle": (b, 2), "n_keys": (b,), "n_letters": (b,),
        }
        for name, shape in shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError("%s must have shape %r" % (name, shape))
        if tuple(self.keycodes.shape) != tuple(self.keycode_mask.shape) or self.keycodes.dim() != 2 or self.keycodes.shape[0] != b:
            raise ValueError("keycodes/keycode_mask must have matching [B,K] shape")
        valid_keycodes = self.keycodes[self.keycode_mask]
        if torch.any(valid_keycodes < 0) or torch.any(valid_keycodes > KEYCODE_TOKEN_MAX):
            raise ValueError(
                "valid keycode token outside audited [0,%d] vocabulary"
                % KEYCODE_TOKEN_MAX
            )
        for name in ("point_mask", "contact_mask", "pointer_mask", "keycode_mask", "ref_point_mask", "ref_contact_mask", "ref_pointer_mask", "ref_mask", "ref_keycode_mask"):
            if getattr(self, name).dtype != torch.bool:
                raise ValueError("%s must be bool" % name)
        for name in (
            "duration_ms", "pointer_start_offset_ms", "pointer_end_offset_ms",
            "start_xy", "end_xy", "pinch_span", "pinch_angle",
        ):
            if not torch.all(torch.isfinite(getattr(self, name))):
                raise ValueError("%s contains non-finite values" % name)
        if not torch.all(self.duration_ms > 0):
            raise ValueError("duration_ms must be positive")
        _validate_pointer_time_bounds(
            self.pointer_mask,
            self.pointer_start_offset_ms,
            self.pointer_end_offset_ms,
            self.duration_ms,
            self.action,
            "target",
        )
        for value in self.orientation_id.detach().cpu().tolist():
            if int(value) not in ORIENTATION_IDS:
                raise ValueError("invalid orientation_id: %r" % value)
        _validate_target_timelines(self)
        _validate_geometry(self)
        _validate_reference_tensors(self, require_references, expected_refs)


def _validate_prefix(point_mask: torch.Tensor, name: str) -> torch.Tensor:
    lengths = point_mask.long().sum(dim=-1)
    t = point_mask.shape[-1]
    prefix = torch.arange(t, device=point_mask.device).view(*([1] * (point_mask.dim() - 1)), t) < lengths.unsqueeze(-1)
    if not torch.equal(prefix, point_mask):
        raise ValueError("%s must be a contiguous valid-timeline prefix" % name)
    return lengths


def _validate_pointer_time_bounds(
    pointer_mask: torch.Tensor,
    start_offset_ms: torch.Tensor,
    end_offset_ms: torch.Tensor,
    duration_ms: torch.Tensor,
    action: str,
    name: str,
) -> None:
    """Validate pointer lifetimes on one shared Android event timeline."""

    if not torch.all(torch.isfinite(start_offset_ms)) or not torch.all(torch.isfinite(end_offset_ms)):
        raise ValueError("%s pointer time offsets must be finite" % name)
    inactive = ~pointer_mask
    if torch.any(start_offset_ms[inactive] != 0) or torch.any(end_offset_ms[inactive] != 0):
        raise ValueError("%s inactive pointer time offsets must be zero" % name)
    active_start = start_offset_ms[pointer_mask]
    active_end = end_offset_ms[pointer_mask]
    expanded_duration = duration_ms.unsqueeze(-1).expand_as(start_offset_ms)
    active_duration = expanded_duration[pointer_mask]
    tolerance = 1.0e-3
    if torch.any(active_start < -tolerance):
        raise ValueError("%s pointer start offsets cannot be negative" % name)
    if torch.any(active_end <= active_start):
        raise ValueError("%s pointer end must be after pointer start" % name)
    if torch.any(active_end > active_duration + tolerance):
        raise ValueError("%s pointer lifetime exceeds event duration" % name)

    # The stored event is the complete union of all active pointer lifetimes.
    masked_start = torch.where(pointer_mask, start_offset_ms, torch.full_like(start_offset_ms, float("inf")))
    masked_end = torch.where(pointer_mask, end_offset_ms, torch.zeros_like(end_offset_ms))
    if torch.any(torch.abs(masked_start.min(dim=-1).values) > tolerance):
        raise ValueError("%s event must begin with an active pointer at t=0" % name)
    if torch.any(torch.abs(masked_end.max(dim=-1).values - duration_ms) > tolerance):
        raise ValueError("%s event duration must equal the last pointer UP" % name)
    if action != "pinch":
        first = start_offset_ms[:, 0]
        last = end_offset_ms[:, 0]
        if torch.any(torch.abs(first) > tolerance) or torch.any(torch.abs(last - duration_ms) > tolerance):
            raise ValueError("%s single-pointer event must cover the full event timeline" % name)


def _validate_one_timeline(
    action: str,
    point: torch.Tensor,
    contact: torch.Tensor,
    events: torch.Tensor,
    active: bool,
    expected_events: Optional[int] = None,
) -> None:
    if not active:
        if torch.any(point) or torch.any(contact) or torch.any(events != -1):
            raise ValueError("inactive pointer contains timeline data")
        return
    n = int(point.long().sum().item())
    if n < 2 or not bool(contact[0].item()) or not bool(contact[n - 1].item()):
        raise ValueError("active timeline needs contact DOWN/UP endpoints")
    if torch.any(contact & ~point):
        raise ValueError("contact_mask must be a subset of point_mask")
    if torch.any(events[contact] < 0) or torch.any(events[~contact] != -1):
        raise ValueError("event_ids must be non-negative exactly at contact points")
    if action != "keystroke":
        if not torch.equal(contact, point) or torch.any(events[point] != 0):
            raise ValueError("non-keystroke timelines are one contiguous contact event")
        return
    ids = events[contact].detach().cpu().tolist()
    unique_ids = sorted(set(int(x) for x in ids))
    required = list(range(len(unique_ids)))
    if unique_ids != required or (expected_events is not None and len(unique_ids) != expected_events):
        raise ValueError("keystroke event ids must be 0..n_keys-1")
    previous_last = -1
    for event_id in unique_ids:
        positions = torch.nonzero(events[:n] == event_id, as_tuple=False).reshape(-1)
        if positions.numel() < 2:
            raise ValueError("each key contact needs at least DOWN and UP")
        first = int(positions[0].item())
        last = int(positions[-1].item())
        if not torch.all(events[first : last + 1] == event_id):
            raise ValueError("a key contact cannot contain an internal gap")
        if first <= previous_last:
            raise ValueError("keystroke event ids must occur in order")
        previous_last = last
    # This additionally proves that every positive flight has explicit
    # no-contact topology, while a zero-flight boundary is represented by
    # adjacent contacts rather than an invented midpoint sample.
    keystroke_zero_flight_flags(
        contact[:n].detach().cpu().numpy(),
        events[:n].detach().cpu().numpy(),
        len(unique_ids),
    )


def _validate_target_timelines(batch: TrajectoryBatch) -> None:
    expected_pointers = 2 if batch.action == "pinch" else 1
    if not torch.all(batch.pointer_mask.sum(dim=1) == expected_pointers):
        raise ValueError("%s requires exactly %d pointer(s)" % (batch.action, expected_pointers))
    _validate_prefix(batch.point_mask, "point_mask")
    if torch.any(batch.point_mask & ~batch.pointer_mask.unsqueeze(-1)):
        raise ValueError("inactive pointer has valid points")
    for i in range(batch.features.shape[0]):
        if batch.action == "keystroke":
            n_keys = int(batch.n_keys[i].item())
            n_letters = int(batch.n_letters[i].item())
            if n_keys <= 0 or n_letters < 0 or n_letters > n_keys or int(batch.keycode_mask[i].sum().item()) != n_keys:
                raise ValueError("keystroke requires n_keys=keycode/contact count and 0<=n_letters<=n_keys")
        else:
            n_keys = None
            if int(batch.n_keys[i].item()) != 0 or int(batch.n_letters[i].item()) != 0 or torch.any(batch.keycode_mask[i]):
                raise ValueError("non-keystroke cannot carry keycodes")
        for pointer_id in range(MAX_POINTERS):
            _validate_one_timeline(
                batch.action,
                batch.point_mask[i, pointer_id],
                batch.contact_mask[i, pointer_id],
                batch.event_ids[i, pointer_id],
                bool(batch.pointer_mask[i, pointer_id].item()),
                n_keys,
            )


def _validate_geometry(batch: TrajectoryBatch) -> None:
    inactive = ~batch.pointer_mask.unsqueeze(-1)
    if torch.any(torch.where(inactive, batch.start_xy, torch.zeros_like(batch.start_xy)) != 0) or torch.any(
        torch.where(inactive, batch.end_xy, torch.zeros_like(batch.end_xy)) != 0
    ):
        raise ValueError("inactive pointer endpoints must be zero")
    if batch.action == "pinch":
        sv = batch.start_xy[:, 1] - batch.start_xy[:, 0]
        ev = batch.end_xy[:, 1] - batch.end_xy[:, 0]
        span = torch.stack([torch.linalg.norm(sv, dim=-1), torch.linalg.norm(ev, dim=-1)], dim=-1)
        angle = torch.stack([torch.atan2(sv[:, 1], sv[:, 0]), torch.atan2(ev[:, 1], ev[:, 0])], dim=-1)
        if not torch.allclose(batch.pinch_span, span, rtol=1e-4, atol=1e-3):
            raise ValueError("pinch_span contradicts endpoints")
        error = torch.atan2(torch.sin(batch.pinch_angle - angle), torch.cos(batch.pinch_angle - angle)).abs()
        if torch.any(error > 1e-4):
            raise ValueError("pinch_angle contradicts endpoints")
    elif torch.any(batch.pinch_span != 0) or torch.any(batch.pinch_angle != 0):
        raise ValueError("pinch geometry is only valid for pinch")


def _validate_reference_tensors(batch: TrajectoryBatch, required: bool, expected_refs: int) -> None:
    b = batch.features.shape[0]
    if batch.ref_features.dim() != 5 or batch.ref_features.shape[0] != b or batch.ref_features.shape[2] != MAX_POINTERS or batch.ref_features.shape[-1] != FEATURE_DIM:
        raise ValueError("ref_features must have shape [B,R,2,Tr,5]")
    _, r, p, tr, _ = batch.ref_features.shape
    shapes = {
        "ref_point_mask": (b, r, p, tr), "ref_contact_mask": (b, r, p, tr),
        "ref_event_ids": (b, r, p, tr), "ref_pointer_mask": (b, r, p),
        "ref_pointer_start_offset_ms": (b, r, p),
        "ref_pointer_end_offset_ms": (b, r, p), "ref_mask": (b, r),
    }
    for name, shape in shapes.items():
        if tuple(getattr(batch, name).shape) != shape:
            raise ValueError("%s has wrong shape" % name)
    if not torch.all(torch.isfinite(batch.ref_features)):
        raise ValueError("reference features contain non-finite values")
    if not torch.all(torch.isfinite(batch.ref_pointer_start_offset_ms)) or not torch.all(
        torch.isfinite(batch.ref_pointer_end_offset_ms)
    ):
        raise ValueError("reference pointer time offsets contain non-finite values")
    if batch.ref_keycodes.dim() != 3 or batch.ref_keycodes.shape[:2] != (b, r) or tuple(batch.ref_keycode_mask.shape) != tuple(batch.ref_keycodes.shape):
        raise ValueError("ref_keycodes/ref_keycode_mask must have shape [B,R,Kr]")
    provenance = (batch.target_sample_ids, batch.target_user_ids, batch.target_splits, batch.ref_sample_ids, batch.ref_user_ids, batch.ref_splits)
    if any(len(item) != b for item in provenance):
        raise ValueError("reference provenance batch length mismatch")
    if required and (r != expected_refs or not torch.all(batch.ref_mask)):
        raise ValueError("formal model requires exactly %d valid refs" % expected_refs)
    if r == 0:
        return
    padded = ~batch.ref_mask.unsqueeze(-1)
    if torch.any(batch.ref_pointer_start_offset_ms[padded.expand_as(batch.ref_pointer_start_offset_ms)] != 0) or torch.any(
        batch.ref_pointer_end_offset_ms[padded.expand_as(batch.ref_pointer_end_offset_ms)] != 0
    ):
        raise ValueError("padded reference pointer time offsets must be zero")
    flat_active = batch.ref_mask.reshape(-1)
    flat_pointer = batch.ref_pointer_mask.reshape(b * r, p)[flat_active]
    flat_start = batch.ref_pointer_start_offset_ms.reshape(b * r, p)[flat_active]
    flat_end = batch.ref_pointer_end_offset_ms.reshape(b * r, p)[flat_active]
    flat_duration = flat_end.max(dim=-1).values
    _validate_pointer_time_bounds(
        flat_pointer,
        flat_start,
        flat_end,
        flat_duration,
        batch.action,
        "reference",
    )
    _validate_prefix(batch.ref_point_mask.reshape(b * r, p, tr), "ref_point_mask")
    for i in range(b):
        active_slots = torch.nonzero(batch.ref_mask[i], as_tuple=False).reshape(-1).detach().cpu().tolist()
        ids = [batch.ref_sample_ids[i][j] for j in active_slots]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate reference ids")
        if batch.target_sample_ids[i] in ids:
            raise ValueError("target/candidate leakage into reference set")
        for j in range(r):
            if not bool(batch.ref_mask[i, j].item()):
                if torch.any(batch.ref_point_mask[i, j]) or torch.any(batch.ref_contact_mask[i, j]):
                    raise ValueError("padded reference slot contains data")
                continue
            if batch.ref_user_ids[i][j] != batch.target_user_ids[i] or batch.ref_splits[i][j] != batch.target_splits[i]:
                raise ValueError("reference must have target user and split")
            expected_pointers = 2 if batch.action == "pinch" else 1
            if int(batch.ref_pointer_mask[i, j].sum().item()) != expected_pointers:
                raise ValueError("reference pointer count mismatch")
            for pointer_id in range(MAX_POINTERS):
                expected_events = int(batch.ref_keycode_mask[i, j].sum().item()) if batch.action == "keystroke" else None
                _validate_one_timeline(
                    batch.action,
                    batch.ref_point_mask[i, j, pointer_id],
                    batch.ref_contact_mask[i, j, pointer_id],
                    batch.ref_event_ids[i, j, pointer_id],
                    bool(batch.ref_pointer_mask[i, j, pointer_id].item()),
                    expected_events,
                )


def _as_vector(values: Optional[Sequence[float]], n: int, default: float) -> np.ndarray:
    if values is None:
        return np.full(n, default, dtype=np.float32)
    result = np.asarray(values, dtype=np.float32).reshape(-1)
    if result.size != n or not np.all(np.isfinite(result)):
        raise ValueError("expected %d finite values" % n)
    return result


def _canonical_pointer(
    pointer: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    xy = np.asarray(pointer["xy"], dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 2 or not np.all(np.isfinite(xy)):
        raise ValueError("pointer xy must be finite [N,2], N>=2")
    n = xy.shape[0]
    timestamps = _as_vector(pointer.get("timestamps_ms"), n, 0.0)
    if pointer.get("timestamps_ms") is None:
        timestamps = np.arange(n, dtype=np.float32) * BASE_DT_MS
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps cannot go backwards")
    pressure = np.clip(_as_vector(pointer.get("pressure"), n, 0.5), 0.0, 1.0)
    size = np.clip(_as_vector(pointer.get("size"), n, 0.5), 0.0, 1.0)
    start, end = xy[0].copy(), xy[-1].copy()
    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    scale = max(chord_length, 1.0)
    unit = chord / chord_length if chord_length > 1e-6 else np.asarray([1.0, 0.0], dtype=np.float32)
    normal = np.asarray([-unit[1], unit[0]], dtype=np.float32)
    delta = xy - start[None, :]
    progress = np.sum(delta * unit[None, :], axis=1) / scale
    lateral = np.sum(delta * normal[None, :], axis=1) / scale
    dt = np.diff(timestamps, prepend=timestamps[0]).astype(np.float32)
    positive = dt[dt > 0]
    dt[0] = float(np.median(positive)) if positive.size else BASE_DT_MS
    log_dt = np.log(np.maximum(dt, 1e-3) / BASE_DT_MS)
    features = np.stack([progress, lateral, log_dt, pressure, size], axis=-1).astype(np.float32)
    pointer_start_ms = float(timestamps[0])
    pointer_end_ms = float(timestamps[-1])
    if pointer_end_ms <= pointer_start_ms:
        pointer_end_ms = pointer_start_ms + float((n - 1) * BASE_DT_MS)
    return features, start, end, pointer_start_ms, pointer_end_ms


def _assemble_keystroke(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    contacts = list(raw.get("contacts", []))
    if not contacts:
        raise ValueError("keystroke requires discrete contacts, not one continuous pointer")
    xy_parts: List[np.ndarray] = []
    time_parts: List[np.ndarray] = []
    pressure_parts: List[np.ndarray] = []
    size_parts: List[np.ndarray] = []
    contact_parts: List[np.ndarray] = []
    event_parts: List[np.ndarray] = []
    keycodes: List[int] = []
    previous_end_time: Optional[float] = None
    previous_end_xy: Optional[np.ndarray] = None
    for event_id, contact in enumerate(contacts):
        xy = np.asarray(contact["xy"], dtype=np.float32)
        ts = np.asarray(contact["timestamps_ms"], dtype=np.float32).reshape(-1)
        if xy.ndim != 2 or xy.shape != (ts.size, 2) or ts.size < 2 or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(ts)):
            raise ValueError("each key contact needs finite DOWN..UP samples")
        if np.any(np.diff(ts) < 0) or (previous_end_time is not None and float(ts[0]) < previous_end_time):
            raise ValueError("key contacts must be time ordered and non-overlapping")
        if previous_end_time is not None and float(ts[0]) > previous_end_time:
            midpoint_time = 0.5 * (previous_end_time + float(ts[0]))
            midpoint_xy = 0.5 * (previous_end_xy + xy[0])
            xy_parts.append(midpoint_xy.reshape(1, 2))
            time_parts.append(np.asarray([midpoint_time], dtype=np.float32))
            pressure_parts.append(np.zeros(1, dtype=np.float32))
            size_parts.append(np.zeros(1, dtype=np.float32))
            contact_parts.append(np.zeros(1, dtype=np.bool_))
            event_parts.append(np.full(1, -1, dtype=np.int64))
        n = ts.size
        xy_parts.append(xy)
        time_parts.append(ts)
        pressure_parts.append(_as_vector(contact.get("pressure"), n, 0.5))
        size_parts.append(_as_vector(contact.get("size"), n, 0.5))
        contact_parts.append(np.ones(n, dtype=np.bool_))
        event_parts.append(np.full(n, event_id, dtype=np.int64))
        keycode = int(contact["keycode"])
        if keycode < 0:
            raise ValueError("keycodes must be non-negative")
        keycodes.append(keycode)
        previous_end_time = float(ts[-1])
        previous_end_xy = xy[-1]
    pointer = {
        "xy": np.concatenate(xy_parts), "timestamps_ms": np.concatenate(time_parts),
        "pressure": np.concatenate(pressure_parts), "size": np.concatenate(size_parts),
    }
    return pointer, np.concatenate(contact_parts), np.concatenate(event_parts), np.asarray(keycodes, dtype=np.int64)


def _pinch_geometry(start_xy: np.ndarray, end_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sv, ev = start_xy[1] - start_xy[0], end_xy[1] - end_xy[0]
    span = np.asarray([np.linalg.norm(sv), np.linalg.norm(ev)], dtype=np.float32)
    angle = np.asarray([math.atan2(float(sv[1]), float(sv[0])), math.atan2(float(ev[1]), float(ev[0]))], dtype=np.float32)
    return span, angle


def canonicalize_sample(raw: Dict[str, Any]) -> CanonicalTrajectory:
    action = str(raw["action"])
    if action not in ACTIONS:
        raise ValueError("unsupported action: %r" % action)
    orientation = int(raw.get("orientation_id", -1))
    if orientation not in ORIENTATION_IDS:
        raise ValueError("invalid orientation_id")
    if action == "keystroke":
        pointer, contact, events, keycodes = _assemble_keystroke(raw)
        pointers = [pointer]
        raw_contact_masks = [contact]
        raw_event_ids = [events]
    else:
        pointers = list(raw["pointers"])
        expected = 2 if action == "pinch" else 1
        if len(pointers) != expected:
            raise ValueError("%s requires %d pointer streams" % (action, expected))
        raw_contact_masks = [None] * len(pointers)
        raw_event_ids = [None] * len(pointers)
        keycodes = np.zeros(0, dtype=np.int64)

    features_list: List[np.ndarray] = []
    contacts_list: List[np.ndarray] = []
    events_list: List[np.ndarray] = []
    starts = np.zeros((MAX_POINTERS, 2), dtype=np.float32)
    ends = np.zeros((MAX_POINTERS, 2), dtype=np.float32)
    pointer_start_times: List[float] = []
    pointer_end_times: List[float] = []
    for pointer_id, pointer in enumerate(pointers):
        features, start, end, pointer_start, pointer_end = _canonical_pointer(pointer)
        if action == "keystroke":
            contact_mask = raw_contact_masks[pointer_id]
            event_ids = raw_event_ids[pointer_id]
            features[~contact_mask, 0] = 0.0
            features[~contact_mask, 1] = 0.0
            features[~contact_mask, 3:] = 0.0
        else:
            contact_mask = np.ones(features.shape[0], dtype=np.bool_)
            event_ids = np.zeros(features.shape[0], dtype=np.int64)
        features_list.append(features)
        contacts_list.append(contact_mask)
        events_list.append(event_ids)
        starts[pointer_id], ends[pointer_id] = start, end
        pointer_start_times.append(pointer_start)
        pointer_end_times.append(pointer_end)
    event_origin_ms = min(pointer_start_times)
    pointer_start_offsets = np.zeros(MAX_POINTERS, dtype=np.float32)
    pointer_end_offsets = np.zeros(MAX_POINTERS, dtype=np.float32)
    for pointer_id, (pointer_start, pointer_end) in enumerate(
        zip(pointer_start_times, pointer_end_times)
    ):
        pointer_start_offsets[pointer_id] = float(pointer_start - event_origin_ms)
        pointer_end_offsets[pointer_id] = float(pointer_end - event_origin_ms)
    derived_duration_ms = float(max(pointer_end_times) - event_origin_ms)
    duration_ms = float(raw.get("duration_ms", derived_duration_ms))
    if not np.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if abs(duration_ms - derived_duration_ms) > 1.0e-3:
        raise ValueError(
            "duration_ms must equal the complete union of pointer lifetimes "
            "(got %.6f, derived %.6f)" % (duration_ms, derived_duration_ms)
        )
    if action == "pinch":
        span, angle = _pinch_geometry(starts, ends)
    else:
        span, angle = np.zeros(2, np.float32), np.zeros(2, np.float32)
    n_keys = int(keycodes.size) if action == "keystroke" else 0
    # HMOG KeyPress.csv uses ASCII for letters.  Keep inference on the shared
    # predicate even when audited n_letters is also supplied explicitly.
    inferred_letters = int(sum(
        is_hmog_ascii_letter_keycode(int(value)) for value in keycodes
    ))
    n_letters = int(raw.get("n_letters", inferred_letters)) if action == "keystroke" else 0
    if n_letters < 0 or n_letters > n_keys:
        raise ValueError("n_letters must be the alphabetic subset of n_keys")
    return CanonicalTrajectory(
        action=action, pointer_features=features_list, pointer_contact_masks=contacts_list,
        pointer_event_ids=events_list, duration_ms=duration_ms,
        pointer_start_offset_ms=pointer_start_offsets,
        pointer_end_offset_ms=pointer_end_offsets, orientation_id=orientation,
        start_xy=starts, end_xy=ends, pinch_span=span, pinch_angle=angle,
        n_keys=n_keys, n_letters=n_letters, keycodes=keycodes, sample_id=str(raw.get("sample_id", "")),
        user_id=int(raw.get("user_id", -1)), split=str(raw.get("split", "")),
        is_real=bool(raw.get("is_real", False)), metadata=dict(raw.get("metadata", {})),
    )


def validate_fewshot_references(target: CanonicalTrajectory, references: Sequence[CanonicalTrajectory], k_refs: int = FORMAL_REF_COUNT) -> None:
    if not target.sample_id or target.user_id < 0 or target.split not in SPLITS:
        raise ValueError("target lacks auditable sample/user/split provenance")
    if len(references) != k_refs:
        raise ValueError("missing refs: expected exactly %d" % k_refs)
    ids = [ref.sample_id for ref in references]
    if any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("references must have unique non-empty ids")
    if target.sample_id in ids:
        raise ValueError("target/candidate cannot be used as its own ref")
    for ref in references:
        if not ref.is_real:
            raise ValueError("reference must come from the real pool")
        if ref.action != target.action or ref.user_id != target.user_id or ref.split != target.split:
            raise ValueError("reference must match target action, user, and split")


def build_fewshot_examples(
    targets: Sequence[CanonicalTrajectory],
    real_pool: Sequence[CanonicalTrajectory],
    k_refs: int = FORMAL_REF_COUNT,
    seed: int = 42,
) -> List[FewShotExample]:
    """Select five unique real same-user/split/action refs without replacement."""
    if k_refs != FORMAL_REF_COUNT:
        raise ValueError("formal protocol fixes k_refs=%d" % FORMAL_REF_COUNT)
    seen = set()
    for ref in real_pool:
        key = (ref.action, ref.user_id, ref.split, ref.sample_id)
        if not ref.sample_id or key in seen:
            raise ValueError("real reference pool has missing/duplicate sample id")
        seen.add(key)
        if not ref.is_real:
            raise ValueError("reference pool contains non-real data")
    rng = random.Random(seed)
    examples: List[FewShotExample] = []
    for target in targets:
        candidates = [
            ref for ref in real_pool
            if ref.action == target.action and ref.user_id == target.user_id and ref.split == target.split
            and ref.sample_id != target.sample_id
        ]
        if len(candidates) < k_refs:
            raise ValueError("missing refs for target %s: %d < %d" % (target.sample_id, len(candidates), k_refs))
        refs = rng.sample(candidates, k_refs)
        validate_fewshot_references(target, refs, k_refs)
        examples.append(FewShotExample(target=target, references=refs))
    return examples


def _empty_refs(b: int) -> Dict[str, Any]:
    return {
        "ref_features": torch.zeros((b, 0, MAX_POINTERS, 1, FEATURE_DIM), dtype=torch.float32),
        "ref_point_mask": torch.zeros((b, 0, MAX_POINTERS, 1), dtype=torch.bool),
        "ref_contact_mask": torch.zeros((b, 0, MAX_POINTERS, 1), dtype=torch.bool),
        "ref_event_ids": torch.full((b, 0, MAX_POINTERS, 1), -1, dtype=torch.long),
        "ref_pointer_mask": torch.zeros((b, 0, MAX_POINTERS), dtype=torch.bool),
        "ref_pointer_start_offset_ms": torch.zeros((b, 0, MAX_POINTERS), dtype=torch.float32),
        "ref_pointer_end_offset_ms": torch.zeros((b, 0, MAX_POINTERS), dtype=torch.float32),
        "ref_mask": torch.zeros((b, 0), dtype=torch.bool),
        "ref_keycodes": torch.zeros((b, 0, 1), dtype=torch.long),
        "ref_keycode_mask": torch.zeros((b, 0, 1), dtype=torch.bool),
        "ref_sample_ids": tuple(tuple() for _ in range(b)),
        "ref_user_ids": tuple(tuple() for _ in range(b)),
        "ref_splits": tuple(tuple() for _ in range(b)),
    }


def collate_trajectories(samples: Sequence[CanonicalTrajectory], max_points: Optional[int] = None, max_keycodes: Optional[int] = None) -> TrajectoryBatch:
    if not samples:
        raise ValueError("empty batch")
    action = samples[0].action
    if any(item.action != action for item in samples):
        raise ValueError("action-specific batch cannot mix actions")
    observed = max(x.shape[0] for item in samples for x in item.pointer_features)
    t = observed if max_points is None else int(max_points)
    if t < observed or t < 2:
        raise ValueError("max_points too small")
    observed_keys = max(item.keycodes.size for item in samples)
    k = max(1, observed_keys if max_keycodes is None else int(max_keycodes))
    if k < observed_keys:
        raise ValueError("max_keycodes too small")
    b = len(samples)
    features = np.zeros((b, MAX_POINTERS, t, FEATURE_DIM), np.float32)
    point = np.zeros((b, MAX_POINTERS, t), np.bool_)
    contact = np.zeros_like(point)
    events = np.full((b, MAX_POINTERS, t), -1, np.int64)
    pointers = np.zeros((b, MAX_POINTERS), np.bool_)
    duration = np.asarray([x.duration_ms for x in samples], np.float32)
    pointer_start_offset = np.stack([x.pointer_start_offset_ms for x in samples]).astype(np.float32)
    pointer_end_offset = np.stack([x.pointer_end_offset_ms for x in samples]).astype(np.float32)
    orientation = np.asarray([x.orientation_id for x in samples], np.int64)
    start = np.stack([x.start_xy for x in samples]).astype(np.float32)
    end = np.stack([x.end_xy for x in samples]).astype(np.float32)
    span = np.stack([x.pinch_span for x in samples]).astype(np.float32)
    angle = np.stack([x.pinch_angle for x in samples]).astype(np.float32)
    letters = np.asarray([x.n_letters for x in samples], np.int64)
    n_keys = np.asarray([x.n_keys for x in samples], np.int64)
    keys = np.zeros((b, k), np.int64)
    key_mask = np.zeros((b, k), np.bool_)
    for i, sample in enumerate(samples):
        for pointer_id, values in enumerate(sample.pointer_features):
            n = values.shape[0]
            features[i, pointer_id, :n] = values
            point[i, pointer_id, :n] = True
            contact[i, pointer_id, :n] = sample.pointer_contact_masks[pointer_id]
            events[i, pointer_id, :n] = sample.pointer_event_ids[pointer_id]
            pointers[i, pointer_id] = True
        if sample.keycodes.size:
            keys[i, : sample.keycodes.size] = sample.keycodes
            key_mask[i, : sample.keycodes.size] = True
    batch = TrajectoryBatch(
        action=action, features=torch.from_numpy(features), point_mask=torch.from_numpy(point),
        contact_mask=torch.from_numpy(contact), event_ids=torch.from_numpy(events), pointer_mask=torch.from_numpy(pointers),
        duration_ms=torch.from_numpy(duration),
        pointer_start_offset_ms=torch.from_numpy(pointer_start_offset),
        pointer_end_offset_ms=torch.from_numpy(pointer_end_offset),
        orientation_id=torch.from_numpy(orientation),
        start_xy=torch.from_numpy(start), end_xy=torch.from_numpy(end), pinch_span=torch.from_numpy(span),
        pinch_angle=torch.from_numpy(angle), n_keys=torch.from_numpy(n_keys), n_letters=torch.from_numpy(letters), keycodes=torch.from_numpy(keys),
        keycode_mask=torch.from_numpy(key_mask), target_sample_ids=tuple(x.sample_id for x in samples),
        target_user_ids=tuple(x.user_id for x in samples), target_splits=tuple(x.split for x in samples),
        **_empty_refs(b)
    )
    batch.validate(require_references=False)
    return batch


def _attach_references(batch: TrajectoryBatch, reference_sets: Sequence[Sequence[CanonicalTrajectory]]) -> TrajectoryBatch:
    b = batch.features.shape[0]
    if len(reference_sets) != b:
        raise ValueError("one reference set is required per target")
    r = FORMAL_REF_COUNT
    for i, refs in enumerate(reference_sets):
        pseudo_target = CanonicalTrajectory(
            action=batch.action, pointer_features=[], pointer_contact_masks=[], pointer_event_ids=[], duration_ms=float(batch.duration_ms[i]),
            pointer_start_offset_ms=batch.pointer_start_offset_ms[i].detach().cpu().numpy().copy(),
            pointer_end_offset_ms=batch.pointer_end_offset_ms[i].detach().cpu().numpy().copy(),
            orientation_id=int(batch.orientation_id[i]), start_xy=np.zeros((2, 2), np.float32), end_xy=np.zeros((2, 2), np.float32),
            pinch_span=np.zeros(2, np.float32), pinch_angle=np.zeros(2, np.float32), n_keys=0, n_letters=0, keycodes=np.zeros(0, np.int64),
            sample_id=batch.target_sample_ids[i], user_id=batch.target_user_ids[i], split=batch.target_splits[i], is_real=False, metadata={},
        )
        validate_fewshot_references(pseudo_target, refs, r)
    tr = max(x.shape[0] for refs in reference_sets for ref in refs for x in ref.pointer_features)
    rf = np.zeros((b, r, MAX_POINTERS, tr, FEATURE_DIM), np.float32)
    rp = np.zeros((b, r, MAX_POINTERS, tr), np.bool_)
    rc = np.zeros_like(rp)
    re = np.full((b, r, MAX_POINTERS, tr), -1, np.int64)
    rpointer = np.zeros((b, r, MAX_POINTERS), np.bool_)
    rpointer_start = np.zeros((b, r, MAX_POINTERS), np.float32)
    rpointer_end = np.zeros((b, r, MAX_POINTERS), np.float32)
    kr = max(1, max(ref.keycodes.size for refs in reference_sets for ref in refs))
    ref_keys = np.zeros((b, r, kr), np.int64)
    ref_key_mask = np.zeros((b, r, kr), np.bool_)
    for i, refs in enumerate(reference_sets):
        for j, ref in enumerate(refs):
            for pointer_id, values in enumerate(ref.pointer_features):
                n = values.shape[0]
                rf[i, j, pointer_id, :n] = values
                rp[i, j, pointer_id, :n] = True
                rc[i, j, pointer_id, :n] = ref.pointer_contact_masks[pointer_id]
                re[i, j, pointer_id, :n] = ref.pointer_event_ids[pointer_id]
                rpointer[i, j, pointer_id] = True
            rpointer_start[i, j] = ref.pointer_start_offset_ms
            rpointer_end[i, j] = ref.pointer_end_offset_ms
            if ref.keycodes.size:
                ref_keys[i, j, : ref.keycodes.size] = ref.keycodes
                ref_key_mask[i, j, : ref.keycodes.size] = True
    values = dict(batch.__dict__)
    values.update(
        ref_features=torch.from_numpy(rf), ref_point_mask=torch.from_numpy(rp), ref_contact_mask=torch.from_numpy(rc),
        ref_event_ids=torch.from_numpy(re), ref_pointer_mask=torch.from_numpy(rpointer), ref_mask=torch.ones((b, r), dtype=torch.bool),
        ref_pointer_start_offset_ms=torch.from_numpy(rpointer_start),
        ref_pointer_end_offset_ms=torch.from_numpy(rpointer_end),
        ref_keycodes=torch.from_numpy(ref_keys), ref_keycode_mask=torch.from_numpy(ref_key_mask),
        ref_sample_ids=tuple(tuple(ref.sample_id for ref in refs) for refs in reference_sets),
        ref_user_ids=tuple(tuple(ref.user_id for ref in refs) for refs in reference_sets),
        ref_splits=tuple(tuple(ref.split for ref in refs) for refs in reference_sets),
    )
    result = TrajectoryBatch(**values)
    result.validate(require_references=True)
    return result


def collate_fewshot_trajectories(examples: Sequence[FewShotExample], max_points: Optional[int] = None) -> TrajectoryBatch:
    if not examples:
        raise ValueError("empty few-shot batch")
    for item in examples:
        validate_fewshot_references(item.target, item.references)
    base = collate_trajectories([item.target for item in examples], max_points=max_points)
    return _attach_references(base, [item.references for item in examples])


def make_sampling_batch(
    action: str, lengths: Sequence[Sequence[int]], duration_ms: Sequence[float], orientation_id: Sequence[int],
    start_xy: Sequence[Sequence[Sequence[float]]], end_xy: Sequence[Sequence[Sequence[float]]],
    pointer_start_offset_ms: Optional[Sequence[Sequence[float]]] = None,
    pointer_end_offset_ms: Optional[Sequence[Sequence[float]]] = None,
    pinch_span: Optional[Sequence[Sequence[float]]] = None, pinch_angle: Optional[Sequence[Sequence[float]]] = None,
    n_keys: Optional[Sequence[int]] = None, n_letters: Optional[Sequence[int]] = None, keycodes: Optional[Sequence[Sequence[int]]] = None,
    contact_masks: Optional[Sequence[Sequence[Sequence[bool]]]] = None,
    event_ids: Optional[Sequence[Sequence[Sequence[int]]]] = None,
    reference_sets: Optional[Sequence[Sequence[CanonicalTrajectory]]] = None,
    target_sample_ids: Optional[Sequence[str]] = None, user_ids: Optional[Sequence[int]] = None,
    splits: Optional[Sequence[str]] = None,
) -> TrajectoryBatch:
    lengths_array = np.asarray(lengths, np.int64)
    if action not in ACTIONS or lengths_array.ndim != 2 or lengths_array.shape[1] != MAX_POINTERS or np.any(lengths_array < 0):
        raise ValueError("invalid action/lengths")
    b, t = lengths_array.shape[0], int(np.max(lengths_array))
    if t < 2:
        raise ValueError("sampling timeline too short")
    point = torch.arange(t).view(1, 1, t) < torch.from_numpy(lengths_array).unsqueeze(-1)
    pointer = torch.from_numpy(lengths_array > 0)
    if action == "keystroke":
        if contact_masks is None or event_ids is None:
            raise ValueError("keystroke sampling requires contact_masks and event_ids")
        contact = torch.as_tensor(contact_masks, dtype=torch.bool)
        events = torch.as_tensor(event_ids, dtype=torch.long)
    else:
        contact = point.clone()
        events = torch.where(point, torch.zeros_like(point, dtype=torch.long), torch.full_like(point, -1, dtype=torch.long))
    keys_list = [[] for _ in range(b)] if keycodes is None else [list(x) for x in keycodes]
    if len(keys_list) != b:
        raise ValueError("one keycode sequence per item")
    k = max(1, max((len(x) for x in keys_list), default=0))
    keys, key_mask = torch.zeros((b, k), dtype=torch.long), torch.zeros((b, k), dtype=torch.bool)
    for i, row in enumerate(keys_list):
        if row:
            keys[i, : len(row)] = torch.as_tensor(row)
            key_mask[i, : len(row)] = True
    ids = tuple(target_sample_ids or ["request_%d" % i for i in range(b)])
    users = tuple(int(x) for x in (user_ids or [-1] * b))
    split_values = tuple(splits or [""] * b)
    duration_tensor = torch.as_tensor(duration_ms, dtype=torch.float32)
    if tuple(duration_tensor.shape) != (b,):
        raise ValueError("duration_ms must contain one value per item")
    if pointer_start_offset_ms is None:
        pointer_start_tensor = torch.zeros((b, MAX_POINTERS), dtype=torch.float32)
    else:
        pointer_start_tensor = torch.as_tensor(pointer_start_offset_ms, dtype=torch.float32)
    if pointer_end_offset_ms is None:
        pointer_end_tensor = torch.where(
            pointer,
            duration_tensor.unsqueeze(-1).expand(b, MAX_POINTERS),
            torch.zeros((b, MAX_POINTERS), dtype=torch.float32),
        )
    else:
        pointer_end_tensor = torch.as_tensor(pointer_end_offset_ms, dtype=torch.float32)
    if tuple(pointer_start_tensor.shape) != (b, MAX_POINTERS) or tuple(pointer_end_tensor.shape) != (b, MAX_POINTERS):
        raise ValueError("pointer time offsets must have shape [B,2]")
    refs = _empty_refs(b)
    batch = TrajectoryBatch(
        action=action, features=torch.zeros((b, MAX_POINTERS, t, FEATURE_DIM)), point_mask=point,
        contact_mask=contact, event_ids=events, pointer_mask=pointer,
        duration_ms=duration_tensor,
        pointer_start_offset_ms=pointer_start_tensor,
        pointer_end_offset_ms=pointer_end_tensor,
        orientation_id=torch.as_tensor(orientation_id, dtype=torch.long),
        start_xy=torch.as_tensor(start_xy, dtype=torch.float32), end_xy=torch.as_tensor(end_xy, dtype=torch.float32),
        pinch_span=torch.zeros((b, 2)) if pinch_span is None else torch.as_tensor(pinch_span, dtype=torch.float32),
        pinch_angle=torch.zeros((b, 2)) if pinch_angle is None else torch.as_tensor(pinch_angle, dtype=torch.float32),
        n_keys=torch.as_tensor([len(x) for x in keys_list] if n_keys is None else n_keys, dtype=torch.long),
        n_letters=torch.zeros(b, dtype=torch.long) if n_letters is None else torch.as_tensor(n_letters, dtype=torch.long),
        keycodes=keys, keycode_mask=key_mask, target_sample_ids=ids, target_user_ids=users, target_splits=split_values,
        **refs
    )
    batch.validate(require_references=False)
    return _attach_references(batch, reference_sets) if reference_sets is not None else batch
