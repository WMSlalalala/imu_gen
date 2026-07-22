"""Strict generated-shard adapter for the independent trajectory PAD stack.

The generator archive is a Type-B slot-update log.  PAD consumes complete
MotionEvent snapshots on one shared timeline.  For pinch, this adapter uses the
union of every original slot-update timestamp and forward-fills the most recent
state of every pointer that is still inside its DOWN..UP lifetime.  No original
timestamp is removed and no fixed-rate resampling is performed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from detectors.deep_pad import FAKE_LABEL, RawTrajectoryRecord, make_record
from runtime_determinism import STRICT_RUNTIME_DETERMINISM_SHA256
from training.corpus import encode_raw_keycode
from trajectory.features import (
    canonical_keycode_feature_token,
    extract_keystroke_features,
    extract_pinch_features,
    extract_single_finger_features,
)

from .archive import SCHEMA_VERSION
from .android import AndroidTrajectoryRecord
from .protocol import (
    ACTIONS, ACTION_TO_ID, FORMAL_GENERATION_BASE_SEED, ID_TO_SPLIT,
    FixedUserSplit, ddim_noise_seed, make_fake_id, stable_seed,
)


REQUIRED_FIELDS = (
    "schema_version", "action_id_scalar", "ddim_steps_scalar", "ddim_eta_scalar",
    "training_diffusion_steps_scalar", "alpha_bar_final_scalar",
    "generation_base_seed_scalar", "generation_batch_size_scalar",
    "runtime_determinism_sha256",
    "condition_request_sha256", "event_plan_sha256", "seed", "ddim_noise_seed",
    "checkpoint_sha256", "fixed_split_sha256", "reference_event_ids",
    "key_endpoint_source_code", "fake_id", "sample_index", "user_id", "split_id",
    "condition_source_code", "android_offsets", "flat_android_t_ms",
    "flat_android_x", "flat_android_y", "flat_android_pressure",
    "flat_android_size", "flat_android_pointer_id", "flat_android_phase",
    "flat_android_action", "flat_android_key_index", "flat_android_keycode",
    "flat_android_frame_index", "pointer_start_offset_ms",
    "pointer_end_offset_ms", "duration_ms", "n_keys", "n_letters",
)


def _materialize(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in REQUIRED_FIELDS if name not in archive]
        if missing:
            raise ValueError("generated archive %s lacks %s" % (path, missing))
        data = {name: archive[name] for name in archive.files}
    for name, value in data.items():
        if value.dtype.kind not in "biufc":
            raise ValueError("generated archive field is not numeric: %s" % name)
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("generated archive field is non-finite: %s" % name)
    return data


def _event_slice(data: Mapping[str, np.ndarray], index: int) -> Dict[str, np.ndarray]:
    left, right = (int(value) for value in data["android_offsets"][index:index + 2])
    if right <= left:
        raise ValueError("generated Android event is empty")
    return {
        "t": np.asarray(data["flat_android_t_ms"][left:right], np.float32),
        "x": np.asarray(data["flat_android_x"][left:right], np.float32),
        "y": np.asarray(data["flat_android_y"][left:right], np.float32),
        "pressure": np.asarray(data["flat_android_pressure"][left:right], np.float32),
        "size": np.asarray(data["flat_android_size"][left:right], np.float32),
        "pointer": np.asarray(data["flat_android_pointer_id"][left:right], np.int8),
        "phase": np.asarray(data["flat_android_phase"][left:right], np.int8),
        "action": np.asarray(data["flat_android_action"][left:right], np.int16) & 0xFF,
        "key_index": np.asarray(data["flat_android_key_index"][left:right], np.int32),
        "keycode": np.asarray(data["flat_android_keycode"][left:right], np.int32),
        "frame": np.asarray(data["flat_android_frame_index"][left:right], np.int64),
    }


def _single_pointer_frames(rows: Mapping[str, np.ndarray], action: str):
    if np.any(rows["pointer"] != 0):
        raise ValueError("non-pinch generated event populated pointer slot 1")
    if np.any(np.diff(rows["frame"]) < 0):
        raise ValueError("generated Android frames are not monotonic")
    frames = np.unique(rows["frame"])
    values: List[np.ndarray] = []
    times: List[float] = []
    contacts: List[np.ndarray] = []
    active: List[np.ndarray] = []
    codes: List[np.ndarray] = []
    keycodes: List[np.ndarray] = []
    events: List[np.ndarray] = []
    gaps: List[bool] = []
    previous_event = None
    previous_time = None
    for frame in frames.tolist():
        positions = np.flatnonzero(rows["frame"] == frame)
        frame_times = rows["t"][positions]
        if positions.size != 1 or not np.all(frame_times == frame_times[0]):
            raise ValueError("single-pointer generated frame is not one unambiguous slot update")
        position = int(positions[-1])
        now = float(rows["t"][position])
        if abs(now - round(now)) > 1e-6:
            raise ValueError("generated source exposes a fractional-ms fake cue")
        event_id = int(rows["key_index"][position]) if action == "keystroke" else 0
        if action == "keystroke" and event_id < 0:
            raise ValueError("keystroke contact lacks event/key index")
        key_transition = (
            action == "keystroke" and previous_event is not None
            and event_id != previous_event
        )
        if previous_time is not None and now < previous_time:
            raise ValueError("generated global frames cannot move backward in time")
        if previous_time is not None and now == previous_time and not key_transition:
            raise ValueError(
                "equal generated frame times are legal only across a zero-flight key boundary"
            )
        if key_transition and now > previous_time:
            midpoint = previous_time + 0.5 * (now - previous_time)
            values.append(np.zeros((2, 4), np.float32))
            times.append(midpoint)
            contacts.append(np.zeros(2, np.bool_))
            active.append(np.zeros(2, np.bool_))
            codes.append(np.full(2, -1, np.int16))
            keycodes.append(np.full(2, -1, np.int32))
            events.append(np.full(2, -1, np.int32))
            gaps.append(True)
        frame_value = np.zeros((2, 4), np.float32)
        frame_value[0] = [
            rows["x"][position], rows["y"][position],
            rows["pressure"][position], rows["size"][position],
        ]
        frame_contact = np.asarray([True, False], np.bool_)
        frame_code = np.asarray([int(rows["action"][position]), -1], np.int16)
        frame_event = np.asarray([event_id, -1], np.int32)
        frame_key = np.asarray([
            encode_raw_keycode(int(rows["keycode"][position])) if action == "keystroke" else -1,
            -1,
        ], np.int32)
        values.append(frame_value)
        times.append(now)
        contacts.append(frame_contact)
        # Generated trajectories contain only the requested active event; the
        # extractor's label interval is not an observable detector feature.
        active.append(frame_contact.copy())
        codes.append(frame_code)
        keycodes.append(frame_key)
        events.append(frame_event)
        gaps.append(False)
        previous_event = event_id
        previous_time = now
    return (
        np.stack(values, axis=1), np.asarray(times, np.float32),
        np.stack(contacts, axis=1), np.stack(active, axis=1),
        np.stack(codes, axis=1), np.stack(keycodes, axis=1),
        np.stack(events, axis=1), np.asarray(gaps, np.bool_),
    )


def _pinch_type_b_snapshots(rows: Mapping[str, np.ndarray], starts: np.ndarray, ends: np.ndarray):
    # These are generator canonical slots, not raw Android pointer IDs.  The
    # shared training/generation corpus loader already maps raw IDs to slots by
    # first appearance (including non-monotonic raw IDs such as 9 -> 2).  Never
    # sort or reinterpret them here.
    if set(int(value) for value in np.unique(rows["pointer"]).tolist()) != {0, 1}:
        raise ValueError("pinch Type-B log must contain exactly slots 0 and 1")
    timeline = np.unique(np.asarray(rows["t"], np.float32))
    if timeline.size < 2 or np.any(np.diff(timeline) <= 0):
        raise ValueError("pinch union timeline must be strictly increasing")
    if np.any(np.abs(timeline - np.rint(timeline)) > 1e-6):
        raise ValueError("pinch generated source is not on the integer-ms lattice")
    values = np.zeros((2, timeline.size, 4), np.float32)
    contact = np.zeros((2, timeline.size), np.bool_)
    codes = np.full((2, timeline.size), -1, np.int16)
    events = np.full((2, timeline.size), -1, np.int32)
    keycodes = np.full((2, timeline.size), -1, np.int32)
    for pointer in range(2):
        positions = np.flatnonzero(rows["pointer"] == pointer)
        pointer_t = rows["t"][positions]
        if pointer_t.size < 2 or np.any(np.diff(pointer_t) <= 0):
            raise ValueError("pinch slot updates are not strictly increasing")
        if float(pointer_t[0]) != float(starts[pointer]) or float(pointer_t[-1]) != float(ends[pointer]):
            raise ValueError("pinch slot lifecycle contradicts archived pointer lifetime")
        active_indices = np.flatnonzero((timeline >= starts[pointer]) & (timeline <= ends[pointer]))
        source_values = np.stack([
            rows["x"][positions], rows["y"][positions],
            rows["pressure"][positions], rows["size"][positions],
        ], axis=-1)
        for global_index in active_indices.tolist():
            source_index = int(np.searchsorted(pointer_t, timeline[global_index], side="right") - 1)
            if source_index < 0:
                raise AssertionError("active pointer has no DOWN state to forward-fill")
            values[pointer, global_index] = source_values[source_index]
        contact[pointer, active_indices] = True
        events[pointer, active_indices] = 0
        codes[pointer, active_indices] = 2
        source_first_action = int(rows["action"][positions[0]])
        source_last_action = int(rows["action"][positions[-1]])
        codes[pointer, active_indices[0]] = source_first_action
        codes[pointer, active_indices[-1]] = source_last_action
    return values, timeline, contact, contact.copy(), codes, keycodes, events, np.zeros(timeline.size, np.bool_)


def _feature_vector(action: str, values: np.ndarray, times: np.ndarray, contact: np.ndarray,
                    keycodes: np.ndarray, events: np.ndarray) -> np.ndarray:
    if action in ("tap", "scroll", "swipe"):
        keep = contact[0]
        return extract_single_finger_features(values[0, keep, :2], times[keep])
    if action == "pinch":
        keep = contact[0] & contact[1]
        if np.sum(keep) < 1:
            raise ValueError("pinch Type-B snapshots have no shared active lifetime")
        return extract_pinch_features(values[0, keep, :2], values[1, keep, :2], times[keep])
    if action == "keystroke":
        event_values = sorted(int(value) for value in np.unique(events[0][contact[0]]).tolist())
        if event_values != list(range(len(event_values))):
            raise ValueError("keystroke contact event ids must be contiguous from zero")
        keys, downs, ups, points = [], [], [], []
        for event_id in event_values:
            positions = np.flatnonzero(contact[0] & (events[0] == event_id))
            tokens = np.unique(keycodes[0, positions])
            if positions.size < 2 or tokens.size != 1 or int(tokens[0]) < 0:
                raise ValueError("keystroke event lacks stable keycode/DOWN/UP")
            keys.append(canonical_keycode_feature_token(int(tokens[0])))
            downs.append(float(times[positions[0]]))
            ups.append(float(times[positions[-1]]))
            points.append(values[0, positions[0], :2])
        return extract_keystroke_features(
            keys, np.asarray(downs), up_times_ms=np.asarray(ups),
            key_points=np.asarray(points, np.float64),
        )
    raise ValueError("unsupported action: %s" % action)


def _record_from_archive(data: Mapping[str, np.ndarray], index: int, action: str) -> Tuple[RawTrajectoryRecord, np.ndarray]:
    rows = _event_slice(data, index)
    if action == "pinch":
        values, times, contact, active, codes, keycodes, events, gap = _pinch_type_b_snapshots(
            rows, np.asarray(data["pointer_start_offset_ms"][index, :2], np.float32),
            np.asarray(data["pointer_end_offset_ms"][index, :2], np.float32),
        )
    else:
        values, times, contact, active, codes, keycodes, events, gap = _single_pointer_frames(rows, action)
    split = ID_TO_SPLIT[int(data["split_id"][index])]
    fake_id = str(int(data["fake_id"][index]))
    record = make_record(
        action=action, label=FAKE_LABEL, user_id=int(data["user_id"][index]), pool=split,
        sample_id=fake_id, event_group_id=fake_id,
        pointer_continuous=values, global_t_ms=times, contact_mask=contact,
        active_mask=active, action_code=codes, keycode=keycodes,
        event_ids=events, gap_mask=gap,
    )
    if action == "keystroke":
        event_count = len(set(int(value) for value in events[0][contact[0]].tolist()))
        if event_count != int(data["n_keys"][index]):
            raise ValueError("generated keystroke PAD record contradicts archived n_keys")
    elif int(data["n_keys"][index]) != 0 or int(data["n_letters"][index]) != 0:
        raise ValueError("non-keystroke generated record carries key counts")
    feature = _feature_vector(action, values, times, contact, keycodes, events)
    return record, np.asarray(feature, np.float64)


def record_from_android_trajectory(
    source: AndroidTrajectoryRecord,
    *,
    sample_id: str,
    pool: str,
    label: int = FAKE_LABEL,
) -> Tuple[RawTrajectoryRecord, np.ndarray]:
    """Convert one in-memory generated event to the detector's raw schema.

    This is the runtime counterpart of ``_record_from_archive``.  Both paths
    use the same MotionEvent snapshot reconstruction and feature definitions;
    the runtime service therefore cannot drift from offline PAD evaluation.
    ``sample_id`` comes from the shared IMU+trajectory event plan rather than
    from array position.
    """

    request = source.request
    rows = {
        "t": np.asarray(source.android_t_ms, np.float32),
        "x": np.asarray(source.android_x, np.float32),
        "y": np.asarray(source.android_y, np.float32),
        "pressure": np.asarray(source.android_pressure, np.float32),
        "size": np.asarray(source.android_size, np.float32),
        "pointer": np.asarray(source.android_pointer_id, np.int8),
        "phase": np.asarray(source.android_phase, np.int8),
        "action": np.asarray(source.android_action, np.int16) & 0xFF,
        "key_index": np.asarray(source.android_key_index, np.int32),
        "keycode": np.asarray(source.android_keycode, np.int32),
        "frame": np.asarray(source.android_frame_index, np.int64),
    }
    row_count = rows["t"].size
    if row_count < 2 or any(np.asarray(value).shape != (row_count,) for value in rows.values()):
        raise ValueError("in-memory Android trajectory fields have inconsistent lengths")
    if request.action == "pinch":
        values, times, contact, active, codes, keycodes, events, gap = _pinch_type_b_snapshots(
            rows,
            np.asarray(request.pointer_start_offset_ms[:2], np.float32),
            np.asarray(request.pointer_end_offset_ms[:2], np.float32),
        )
    else:
        values, times, contact, active, codes, keycodes, events, gap = _single_pointer_frames(
            rows, request.action
        )
    record = make_record(
        action=request.action, label=int(label), user_id=int(request.user_id), pool=str(pool),
        sample_id=str(sample_id), event_group_id=str(sample_id),
        pointer_continuous=values, global_t_ms=times, contact_mask=contact,
        active_mask=active, action_code=codes, keycode=keycodes,
        event_ids=events, gap_mask=gap,
    )
    feature = _feature_vector(request.action, values, times, contact, keycodes, events)
    return record, np.asarray(feature, np.float64)


def _action_paths(root: Path, action: str) -> List[Path]:
    return sorted(Path(root).glob("shards/shard_*_of_*/%s/user_*.npz" % action))


def load_generated_action_tree(
    root: Path,
    action: str,
    fixed_split: FixedUserSplit,
    require_formal: bool = True,
) -> Tuple[List[RawTrajectoryRecord], np.ndarray]:
    """Load one action's 100x200 generated shards into detector records."""
    if action not in ACTIONS:
        raise ValueError("unsupported action: %s" % action)
    paths = _action_paths(Path(root), action)
    if not paths:
        raise FileNotFoundError("no generated %s unit archives under %s" % (action, root))
    if require_formal and len(paths) != 100:
        raise ValueError("formal %s export needs exactly 100 user archives; found %d" % (action, len(paths)))
    records: List[RawTrajectoryRecord] = []
    features: List[np.ndarray] = []
    seen_users = set()
    seen_fake_ids = set()
    checkpoint_hashes = set()
    for path in paths:
        match = re.fullmatch(r"user_(\d+)\.npz", path.name)
        if not match:
            raise ValueError("generated unit filename is not user_XXX.npz: %s" % path)
        filename_user = int(match.group(1))
        data = _materialize(path)
        n = int(data["fake_id"].size)
        if tuple(int(value) for value in data["schema_version"].tolist()) != SCHEMA_VERSION:
            raise ValueError("unsupported generated unit schema: %s" % path)
        runtime_digest = bytes(
            np.asarray(data["runtime_determinism_sha256"], np.uint8).tolist()
        ).hex()
        if runtime_digest != STRICT_RUNTIME_DETERMINISM_SHA256:
            raise ValueError("generated detector ingress runtime determinism mismatch")
        if int(data["action_id_scalar"]) != ACTION_TO_ID[action]:
            raise ValueError("generated unit action/path mismatch: %s" % path)
        generation_base_seed = int(data["generation_base_seed_scalar"])
        if require_formal and generation_base_seed != FORMAL_GENERATION_BASE_SEED:
            raise ValueError("formal PAD export rejected the wrong generation base seed")
        generation_batch_size = int(data["generation_batch_size_scalar"])
        if generation_batch_size <= 0 or (require_formal and generation_batch_size != 32):
            raise ValueError("generated detector ingress batch-size provenance mismatch")
        if int(data["ddim_steps_scalar"]) != 50:
            raise ValueError("formal PAD export requires genuine 50-step DDIM")
        eta_array = np.asarray(data["ddim_eta_scalar"])
        if (
            eta_array.shape != ()
            or eta_array.dtype != np.dtype(np.float32)
            or not np.isfinite(float(eta_array))
            or float(eta_array) != 0.0
        ):
            raise ValueError("generated detector ingress requires DDIM eta=0")
        if require_formal and (
            int(data["training_diffusion_steps_scalar"]) != 1000
            or float(data["alpha_bar_final_scalar"]) > 0.001
        ):
            raise ValueError("formal PAD export rejected non-1000-step/non-Gaussian checkpoint")
        if any("selector" in name.lower() for name in data):
            raise ValueError("selector output is forbidden")
        if require_formal and n != 200:
            raise ValueError("formal user/action archive must contain exactly 200 fake")
        user_values = np.unique(data["user_id"])
        if user_values.size != 1 or int(user_values[0]) != filename_user or filename_user in seen_users:
            raise ValueError("generated user archive/path is duplicate or inconsistent")
        seen_users.add(filename_user)
        expected_split = fixed_split.split_for_user(filename_user)
        expected_split_id = {"train": 0, "val": 1, "test": 2}[expected_split]
        if np.any(data["split_id"] != expected_split_id):
            raise ValueError("generated fake user split contradicts fixed users_seed42")
        if np.any(data["condition_source_code"] != 2):
            raise ValueError("formal generated fake must use five refs + train-only prior")
        split_digest = bytes(np.asarray(data["fixed_split_sha256"], np.uint8).tolist()).hex()
        if split_digest != fixed_split.source_sha256:
            raise ValueError("generated archive fixed-split SHA does not match users_seed42")
        if data["reference_event_ids"].shape != (n, 5) or np.unique(
            data["reference_event_ids"], axis=0
        ).shape[0] != 1 or len(set(int(value) for value in data["reference_event_ids"][0])) != 5:
            raise ValueError("generated detector ingress lost one fixed set of five unique refs")
        if np.any(np.abs(data["duration_ms"] - np.rint(data["duration_ms"])) > 1e-6) or np.any(
            np.abs(data["pointer_start_offset_ms"] - np.rint(data["pointer_start_offset_ms"])) > 1e-6
        ) or np.any(np.abs(data["pointer_end_offset_ms"] - np.rint(data["pointer_end_offset_ms"])) > 1e-6):
            raise ValueError("generated detector ingress metadata is not on the integer-ms lattice")
        if action == "keystroke":
            if data["key_endpoint_source_code"].shape != (n, 2) or np.any(
                ~np.isin(data["key_endpoint_source_code"], [1, 2, 3])
            ):
                raise ValueError("generated keystroke endpoint provenance is missing")
        elif np.any(data["key_endpoint_source_code"] != 0):
            raise ValueError("non-keystroke carries key endpoint provenance")
        if require_formal and set(int(value) for value in data["sample_index"].tolist()) != set(range(200)):
            raise ValueError("formal user/action sample_index must be exactly 0..199")
        sample_indices = np.asarray(data["sample_index"], np.int64)
        expected_seeds = np.asarray([
            stable_seed(generation_base_seed, action, filename_user, int(sample_index))
            for sample_index in sample_indices.tolist()
        ], np.int64)
        if not np.array_equal(np.asarray(data["seed"], np.int64), expected_seeds):
            raise ValueError("generated detector ingress seed provenance mismatch")
        expected_noise_seeds = np.asarray([
            ddim_noise_seed(
                int(expected_seeds[index]), action, filename_user, int(sample_index)
            )
            for index, sample_index in enumerate(sample_indices.tolist())
        ], np.int64)
        if not np.array_equal(
            np.asarray(data["ddim_noise_seed"], np.int64), expected_noise_seeds
        ):
            raise ValueError("generated detector ingress DDIM noise-seed provenance mismatch")
        expected_fake_ids = np.asarray([
            make_fake_id(action, filename_user, int(sample_index))
            for sample_index in sample_indices.tolist()
        ], np.int64)
        if not np.array_equal(np.asarray(data["fake_id"], np.int64), expected_fake_ids):
            raise ValueError("generated detector ingress fake_id provenance mismatch")
        if data["condition_request_sha256"].shape != (n, 32):
            raise ValueError("generated detector ingress lacks per-request SHA-256")
        if data["event_plan_sha256"].shape != (n, 32):
            raise ValueError("generated detector ingress lacks per-plan SHA-256")
        unit_ids = set(int(value) for value in data["fake_id"].tolist())
        if len(unit_ids) != n or seen_fake_ids.intersection(unit_ids):
            raise ValueError("duplicate generated fake_id")
        seen_fake_ids.update(unit_ids)
        checkpoint_hash = bytes(np.asarray(data["checkpoint_sha256"], np.uint8).tolist()).hex()
        if require_formal and checkpoint_hash == "0" * 64:
            raise ValueError("formal detector ingress rejected a zero/smoke checkpoint digest")
        checkpoint_hashes.add(checkpoint_hash)
        for index in range(n):
            record, feature = _record_from_archive(data, index, action)
            records.append(record)
            features.append(feature)
    if len(checkpoint_hashes) != 1:
        raise ValueError("one action's formal units must use one identical best-EMA checkpoint")
    if require_formal:
        if seen_users != set(range(100)) or len(records) != 20_000:
            raise ValueError("formal action export is not exactly 100 users x 200 fake")
    feature_array = np.stack(features).astype(np.float64)
    if not np.all(np.isfinite(feature_array)):
        raise ValueError("generated PAD features are non-finite")
    return records, feature_array


def audit_generated_archive_tree(
    root: Path, fixed_split: FixedUserSplit, require_formal: bool = True,
) -> Dict[str, object]:
    """End-to-end detector-ingress count/identity/lifecycle audit."""
    per_action: Dict[str, Dict[str, int]] = {}
    all_ids = set()
    for action in ACTIONS:
        records, features = load_generated_action_tree(root, action, fixed_split, require_formal=require_formal)
        ids = {int(record.sample_id) for record in records}
        if all_ids.intersection(ids):
            raise ValueError("fake_id collides across actions")
        all_ids.update(ids)
        per_action[action] = {
            "records": len(records), "users": len({row.user_id for row in records}),
            "feature_dim": int(features.shape[1]),
        }
    expected = 100_000 if require_formal else sum(value["records"] for value in per_action.values())
    if len(all_ids) != expected:
        raise ValueError("generation tree global fake count mismatch")
    return {
        "passed": True, "root": str(Path(root).resolve()),
        "n_fake": len(all_ids), "per_action": per_action,
        "type_b_pinch_snapshot": "union_timeline_forward_fill_inside_pointer_lifetime",
        "keystroke_gap_policy": "one_midpoint_no_contact_token_between_contacts",
        "selector_used": False,
    }


__all__ = [
    "load_generated_action_tree", "audit_generated_archive_tree",
    "record_from_android_trajectory",
]
