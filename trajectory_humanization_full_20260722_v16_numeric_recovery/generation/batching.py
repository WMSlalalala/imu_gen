"""Convert audited condition requests into exact five-shot sampling batches."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

from trajectory.data import CanonicalTrajectory, TrajectoryBatch, make_sampling_batch
from training.corpus import canonical_sample_sha256

from .protocol import ConditionRequest, ReferenceConditionPolicy


def build_sampling_batch(
    requests: Sequence[ConditionRequest],
    reference_sets: Sequence[Sequence[CanonicalTrajectory]],
    device: torch.device,
) -> TrajectoryBatch:
    if not requests or len(requests) != len(reference_sets):
        raise ValueError("requests and reference_sets must be matching non-empty sequences")
    action = requests[0].action
    if any(item.action != action for item in requests):
        raise ValueError("one sampling batch cannot mix actions")
    max_points = max(max(item.lengths) for item in requests)
    contacts = np.zeros((len(requests), 2, max_points), dtype=np.bool_)
    events = np.full((len(requests), 2, max_points), -1, dtype=np.int64)
    for batch_id, item in enumerate(requests):
        ReferenceConditionPolicy.validate_request(item)
        if tuple(item.reference_ids) != tuple(int(x.sample_id) for x in reference_sets[batch_id]):
            raise ValueError("request/reference ordering mismatch")
        if tuple(item.reference_canonical_sha256) != tuple(
            canonical_sample_sha256(x) for x in reference_sets[batch_id]
        ):
            raise ValueError("request/reference canonical SHA mismatch")
        for pointer_id, length in enumerate(item.lengths):
            if not length:
                continue
            if item.contact_masks[pointer_id].shape != (length,) or item.event_ids[pointer_id].shape != (length,):
                raise ValueError("request topology length mismatch")
            contacts[batch_id, pointer_id, :length] = item.contact_masks[pointer_id]
            events[batch_id, pointer_id, :length] = item.event_ids[pointer_id]
    batch = make_sampling_batch(
        action=action,
        lengths=[item.lengths for item in requests],
        duration_ms=[item.duration_ms for item in requests],
        orientation_id=[item.orientation_id for item in requests],
        start_xy=np.stack([item.start_xy for item in requests]),
        end_xy=np.stack([item.end_xy for item in requests]),
        pointer_start_offset_ms=np.stack([item.pointer_start_offset_ms for item in requests]),
        pointer_end_offset_ms=np.stack([item.pointer_end_offset_ms for item in requests]),
        pinch_span=np.stack([item.pinch_span for item in requests]),
        pinch_angle=np.stack([item.pinch_angle for item in requests]),
        n_keys=[item.n_keys for item in requests],
        n_letters=[item.n_letters for item in requests],
        keycodes=[item.keycodes.tolist() for item in requests],
        contact_masks=contacts,
        event_ids=events,
        reference_sets=reference_sets,
        target_sample_ids=[str(item.fake_id) for item in requests],
        user_ids=[item.user_id for item in requests],
        splits=[item.split for item in requests],
    )
    batch.validate(require_references=True)
    return batch.to(device)
