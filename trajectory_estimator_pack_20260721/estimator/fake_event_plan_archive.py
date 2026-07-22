"""Recover exact shared EventPlans from numeric formal trajectory archives."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from generation.event_plan import EventPlan
from generation.protocol import (
    ConditionRequest,
    ID_TO_ACTION,
    ID_TO_SPLIT,
    canonical_condition_request_digest,
)


REQUIRED_FIELDS = {
    "action_id_scalar", "condition_request_sha256", "event_plan_sha256",
    "fake_id", "sample_index", "seed", "ddim_noise_seed", "user_id", "split_id",
    "orientation_id", "duration_ms", "lengths", "start_xy", "end_xy",
    "pinch_span", "pinch_angle", "pointer_start_offset_ms", "pointer_end_offset_ms",
    "n_keys", "n_letters", "zero_flight_probability", "key_endpoint_source_code",
    "condition_source_code", "condition_carrier_ref_id", "reference_event_ids",
    "reference_canonical_sha256", "screen_min_xy", "screen_max_xy",
    "key_offsets", "keycodes", "key_flight_offsets", "flat_zero_flight_after_key",
    "trajectory_offsets", "trajectory_pointer_offsets", "flat_trajectory_contact_mask",
    "flat_trajectory_event_id", "train_prior_sha256",
}


def _digest_hex(value: np.ndarray, name: str) -> str:
    array = np.asarray(value, dtype=np.uint8)
    if array.shape != (32,):
        raise ValueError("%s must contain exactly 32 digest bytes" % name)
    return bytes(array).hex()


def load_event_plans_from_archive(
    path: Path, *, expected_action: Optional[str] = None
) -> List[EventPlan]:
    """Load plans only after exact ConditionRequest and EventPlan digest replay."""

    source_path = Path(path)
    with np.load(str(source_path), allow_pickle=False) as source:
        missing = REQUIRED_FIELDS - set(source.files)
        if missing:
            raise ValueError("trajectory archive lacks EventPlan fields: %s" % sorted(missing))
        if any(source[name].dtype.kind == "O" for name in source.files):
            raise ValueError("trajectory archive contains object arrays")
        arrays = {name: np.asarray(source[name]) for name in REQUIRED_FIELDS}
    for name, value in arrays.items():
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("trajectory archive field %s contains non-finite values" % name)

    try:
        action = ID_TO_ACTION[int(arrays["action_id_scalar"])]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("trajectory archive has an unknown action id") from exc
    if expected_action is not None and action != expected_action:
        raise ValueError("trajectory archive action mismatch")
    n = int(arrays["fake_id"].size)
    if n < 1 or np.asarray(arrays["fake_id"]).shape != (n,):
        raise ValueError("trajectory archive has no event rows")
    if len(np.unique(arrays["fake_id"])) != n:
        raise ValueError("trajectory archive contains duplicate fake ids")
    if arrays["condition_request_sha256"].shape != (n, 32) or arrays["event_plan_sha256"].shape != (n, 32):
        raise ValueError("trajectory archive digest matrices have invalid shapes")

    plans: List[EventPlan] = []
    for index in range(n):
        trajectory_left, trajectory_right = (
            int(value) for value in arrays["trajectory_offsets"][index:index + 2]
        )
        pointer_offsets = np.asarray(arrays["trajectory_pointer_offsets"][index], dtype=np.int64)
        if pointer_offsets.shape != (3,) or pointer_offsets[0] != 0 or pointer_offsets[-1] != trajectory_right - trajectory_left:
            raise ValueError("trajectory pointer offsets are invalid")
        contact_masks = []
        event_ids = []
        for pointer in range(2):
            left = trajectory_left + int(pointer_offsets[pointer])
            right = trajectory_left + int(pointer_offsets[pointer + 1])
            contact_masks.append(
                np.asarray(arrays["flat_trajectory_contact_mask"][left:right], dtype=np.bool_)
            )
            event_ids.append(
                np.asarray(arrays["flat_trajectory_event_id"][left:right], dtype=np.int64)
            )
        key_left, key_right = (int(value) for value in arrays["key_offsets"][index:index + 2])
        flight_left, flight_right = (
            int(value) for value in arrays["key_flight_offsets"][index:index + 2]
        )
        try:
            split = ID_TO_SPLIT[int(arrays["split_id"][index])]
        except KeyError as exc:
            raise ValueError("trajectory archive has an unknown split id") from exc
        request = ConditionRequest(
            action=action,
            user_id=int(arrays["user_id"][index]),
            split=split,
            fake_id=int(arrays["fake_id"][index]),
            sample_index=int(arrays["sample_index"][index]),
            seed=int(arrays["seed"][index]),
            reference_ids=tuple(int(value) for value in arrays["reference_event_ids"][index]),
            reference_canonical_sha256=tuple(
                bytes(value).hex() for value in arrays["reference_canonical_sha256"][index]
            ),
            carrier_ref_id=int(arrays["condition_carrier_ref_id"][index]),
            lengths=tuple(int(value) for value in arrays["lengths"][index]),
            duration_ms=float(arrays["duration_ms"][index]),
            orientation_id=int(arrays["orientation_id"][index]),
            start_xy=np.asarray(arrays["start_xy"][index], dtype=np.float32),
            end_xy=np.asarray(arrays["end_xy"][index], dtype=np.float32),
            pinch_span=np.asarray(arrays["pinch_span"][index], dtype=np.float32),
            pinch_angle=np.asarray(arrays["pinch_angle"][index], dtype=np.float32),
            pointer_start_offset_ms=np.asarray(
                arrays["pointer_start_offset_ms"][index], dtype=np.float32
            ),
            pointer_end_offset_ms=np.asarray(
                arrays["pointer_end_offset_ms"][index], dtype=np.float32
            ),
            n_keys=int(arrays["n_keys"][index]),
            n_letters=int(arrays["n_letters"][index]),
            # These are the original in-memory dtypes used by the canonical
            # request digest, not merely the archive storage dtypes.
            keycodes=np.asarray(arrays["keycodes"][key_left:key_right], dtype=np.int64),
            zero_flight_after_key=np.asarray(
                arrays["flat_zero_flight_after_key"][flight_left:flight_right], dtype=np.bool_
            ),
            zero_flight_probability=float(arrays["zero_flight_probability"][index]),
            key_endpoint_source_code=np.asarray(
                arrays["key_endpoint_source_code"][index], dtype=np.int8
            ),
            contact_masks=tuple(contact_masks),
            event_ids=tuple(event_ids),
            condition_source_code=int(arrays["condition_source_code"][index]),
            train_prior_digest=_digest_hex(arrays["train_prior_sha256"], "train_prior_sha256"),
            screen_min_xy=np.asarray(arrays["screen_min_xy"][index], dtype=np.float32),
            screen_max_xy=np.asarray(arrays["screen_max_xy"][index], dtype=np.float32),
        )
        observed_request_digest = canonical_condition_request_digest(request)
        if observed_request_digest != bytes(arrays["condition_request_sha256"][index]):
            raise ValueError("archived ConditionRequest digest does not replay")
        plan = EventPlan.from_condition_request(
            request, sample_id=str(request.fake_id), start_time_ns=None, text=None
        )
        if plan.plan_sha256 != bytes(arrays["event_plan_sha256"][index]).hex():
            raise ValueError("archived EventPlan digest does not replay")
        if int(plan.trajectory_noise_seed) != int(arrays["ddim_noise_seed"][index]):
            raise ValueError("archived trajectory noise seed disagrees with EventPlan")
        plans.append(plan)
    return plans


__all__ = ["REQUIRED_FIELDS", "load_event_plans_from_archive"]
