"""Hard physical/schema constraints applied after neural trajectory sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .data import BASE_DT_MS, FEATURE_DIM, MAX_POINTERS, TrajectoryBatch


PROGRESS = 0
LATERAL = 1
LOG_DT = 2
PRESSURE = 3
SIZE = 4


def _deterministic_cumsum_1d(values: torch.Tensor) -> torch.Tensor:
    """Deterministic inference-only prefix sum for one variable-length trace.

    CUDA ``cumsum`` is explicitly reported by PyTorch as non-deterministic.
    Hard timing projection runs once after DDIM and does not require gradients,
    so compute this small scan on CPU and copy it back to the source device.
    Dtype is preserved.
    """

    if values.ndim != 1:
        raise ValueError("deterministic cumsum expects a 1-D tensor")
    cpu_values = values.detach().to(device="cpu")
    return torch.cumsum(cpu_values, dim=0).to(
        device=values.device, dtype=values.dtype
    )


@dataclass
class ConstrainedTrajectory:
    """Projected local features plus decoded screen-space trajectory."""

    features: torch.Tensor
    xy: torch.Tensor
    timestamps_ms: torch.Tensor
    pressure: torch.Tensor
    size: torch.Tensor
    point_mask: torch.Tensor
    contact_mask: torch.Tensor
    event_ids: torch.Tensor
    contact_phase: torch.Tensor
    pointer_mask: torch.Tensor
    duration_ms: torch.Tensor
    pointer_start_offset_ms: torch.Tensor
    pointer_end_offset_ms: torch.Tensor


def pinch_endpoints_from_center(
    center_start: torch.Tensor,
    center_end: torch.Tensor,
    span_start_end: torch.Tensor,
    angle_start_end: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert pinch centers plus span/angle into two pointer endpoints.

    Args use shapes ``center_*=[B,2]`` and ``span/angle=[B,2]`` where the last
    dimension denotes start and end geometry.
    """
    if center_start.dim() != 2 or center_start.shape[1] != 2 or center_end.shape != center_start.shape:
        raise ValueError("pinch centers must have shape [B,2]")
    if span_start_end.shape != center_start.shape or angle_start_end.shape != center_start.shape:
        raise ValueError("pinch span/angle must have shape [B,2]")
    if torch.any(span_start_end < 0):
        raise ValueError("pinch spans must be non-negative")
    start_vector = torch.stack(
        [torch.cos(angle_start_end[:, 0]), torch.sin(angle_start_end[:, 0])], dim=-1
    ) * (0.5 * span_start_end[:, 0:1])
    end_vector = torch.stack(
        [torch.cos(angle_start_end[:, 1]), torch.sin(angle_start_end[:, 1])], dim=-1
    ) * (0.5 * span_start_end[:, 1:2])
    start_xy = torch.stack([center_start - start_vector, center_start + start_vector], dim=1)
    end_xy = torch.stack([center_end - end_vector, center_end + end_vector], dim=1)
    return start_xy, end_xy


def validate_sampling_masks(batch: TrajectoryBatch) -> None:
    """Validate pointer count, prefix point masks, and inactive padding."""
    batch.validate()
    # validate() already enforces all schema invariants.  Keeping this named
    # entry point makes the hard sampling gate explicit and independently
    # callable by deployment code.


def project_local_features(
    features: torch.Tensor,
    batch: TrajectoryBatch,
    min_dt_ms: float = 0.1,
    max_dt_ms: float = 1000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project a sampled local tensor and produce strict timestamps.

    Hard invariants:
    - inactive/padded positions are exactly zero;
    - first/last local points decode to requested endpoints;
    - the local basis is exactly inverse to canonicalization even when a tap
      starts and ends at the same pixel;
    - every within-contact/positive-flight timestamp is strictly increasing;
    - a keystroke zero-flight UP->DOWN boundary alone may repeat one integer ms;
    - every pointer ends at duration_ms;
    - pressure and size are in [0,1].
    """
    validate_sampling_masks(batch)
    if tuple(features.shape) != tuple(batch.features.shape):
        raise ValueError("sample feature shape does not match target batch")
    if not torch.all(torch.isfinite(features)):
        raise ValueError("sampled features contain non-finite values")
    if min_dt_ms <= 0 or max_dt_ms <= min_dt_ms:
        raise ValueError("invalid dt bounds")

    projected = features.clone()
    mask4 = batch.feature_mask
    projected = torch.where(mask4, projected, torch.zeros_like(projected))
    projected[..., PRESSURE] = torch.clamp(projected[..., PRESSURE], 0.0, 1.0)
    projected[..., SIZE] = torch.clamp(projected[..., SIZE], 0.0, 1.0)
    timestamps = torch.zeros_like(projected[..., LOG_DT])

    b, p, _, f = projected.shape
    if p != MAX_POINTERS or f != FEATURE_DIM:
        raise ValueError("expected [B,2,T,5] local features")
    lengths = batch.point_mask.long().sum(dim=-1)
    for batch_id in range(b):
        for pointer_id in range(p):
            if not bool(batch.pointer_mask[batch_id, pointer_id].item()):
                projected[batch_id, pointer_id] = 0.0
                continue
            n = int(lengths[batch_id, pointer_id].item())
            projected[batch_id, pointer_id, 0, PROGRESS] = 0.0
            projected[batch_id, pointer_id, 0, LATERAL] = 0.0
            chord_length = torch.linalg.norm(
                batch.end_xy[batch_id, pointer_id]
                - batch.start_xy[batch_id, pointer_id]
            )
            # Canonicalization divides both local axes by max(chord, 1 px).
            # A zero-chord tap therefore has endpoint progress 0, not 1.  The
            # old hard-coded value made encoding/decoding non-invertible and
            # discarded one axis of tap jitter when start_xy == end_xy.
            local_scale = torch.clamp(chord_length, min=1.0)
            projected[batch_id, pointer_id, n - 1, PROGRESS] = (
                chord_length / local_scale
            )
            projected[batch_id, pointer_id, n - 1, LATERAL] = 0.0

            raw_log_dt = torch.clamp(
                projected[batch_id, pointer_id, 1:n, LOG_DT],
                min=float(torch.log(torch.tensor(min_dt_ms / BASE_DT_MS)).item()),
                max=float(torch.log(torch.tensor(max_dt_ms / BASE_DT_MS)).item()),
            )
            weights = torch.exp(raw_log_dt)
            allowed_zero = torch.zeros(n - 1, dtype=torch.bool, device=projected.device)
            if batch.action == "keystroke":
                local_contact = batch.contact_mask[batch_id, pointer_id, :n]
                local_events = batch.event_ids[batch_id, pointer_id, :n]
                # Adjacent contacts whose event id changes encode an observed
                # zero-ms flight.  Positive flights have an explicit gap token
                # and therefore retain positive timing weights on both sides.
                allowed_zero = (
                    local_contact[:-1]
                    & local_contact[1:]
                    & (local_events[:-1] >= 0)
                    & (local_events[1:] == local_events[:-1] + 1)
                )
                weights = torch.where(allowed_zero, torch.zeros_like(weights), weights)
            if not bool(torch.any(weights > 0).item()):
                raise ValueError("trajectory topology has no positive-time interval")
            cumulative = torch.cat(
                [weights.new_zeros(1), _deterministic_cumsum_1d(weights)], dim=0
            )
            # Positive weights define the intended relative timing; explicit
            # checks below guard against rare finite-precision collapse.
            pointer_start = batch.pointer_start_offset_ms[batch_id, pointer_id]
            pointer_end = batch.pointer_end_offset_ms[batch_id, pointer_id]
            pointer_duration = pointer_end - pointer_start
            time_values = pointer_start + cumulative / cumulative[-1] * pointer_duration
            time_values[0] = pointer_start
            time_values[-1] = pointer_end
            intervals_ms = time_values[1:] - time_values[:-1]
            legal = torch.all(intervals_ms[~allowed_zero] > 0) & torch.all(
                intervals_ms[allowed_zero] == 0
            )
            if not bool(legal.item()):
                # Extremely unbalanced predicted intervals can collapse after
                # float32 rounding.  Fall back to equal positive increments
                # while preserving every declared zero-flight boundary.
                fallback_weights = (~allowed_zero).to(projected.dtype)
                fallback_cumulative = torch.cat(
                    [
                        fallback_weights.new_zeros(1),
                        _deterministic_cumsum_1d(fallback_weights),
                    ]
                )
                time_values = pointer_start + (
                    fallback_cumulative / fallback_cumulative[-1] * pointer_duration
                )
                time_values[0] = pointer_start
                time_values[-1] = pointer_end
                intervals_ms = time_values[1:] - time_values[:-1]
            if torch.any(intervals_ms[~allowed_zero] <= 0) or torch.any(
                intervals_ms[allowed_zero] != 0
            ):
                raise ValueError("duration/topology cannot form a legal nondecreasing timeline")
            timestamps[batch_id, pointer_id, :n] = time_values
            consistent_log_dt = torch.log(
                torch.clamp(intervals_ms, min=1.0e-3) / BASE_DT_MS
            )
            projected[batch_id, pointer_id, 1:n, LOG_DT] = consistent_log_dt
            projected[batch_id, pointer_id, 0, LOG_DT] = consistent_log_dt[0]

    projected = torch.where(mask4, projected, torch.zeros_like(projected))
    timestamps = torch.where(batch.point_mask, timestamps, torch.zeros_like(timestamps))
    return projected, timestamps


def decode_local_xy(projected: torch.Tensor, batch: TrajectoryBatch) -> torch.Tensor:
    """Decode local progress/lateral into global screen pixels."""
    if tuple(projected.shape) != tuple(batch.features.shape):
        raise ValueError("projected feature shape does not match batch")
    chord = batch.end_xy - batch.start_xy
    chord_length = torch.linalg.norm(chord, dim=-1, keepdim=True)
    safe_length = torch.clamp(chord_length, min=1.0)
    default_unit = torch.zeros_like(chord)
    default_unit[..., 0] = 1.0
    unit = torch.where(chord_length > 1e-6, chord / torch.clamp(chord_length, min=1e-6), default_unit)
    normal = torch.stack([-unit[..., 1], unit[..., 0]], dim=-1)
    progress = projected[..., PROGRESS : PROGRESS + 1]
    lateral = projected[..., LATERAL : LATERAL + 1]
    # Canonicalization represents both local axes in units of
    # max(chord_length, 1 px).  Use that same scale and basis here; multiplying
    # progress by the raw chord loses the along-axis component for a
    # zero-displacement tap.
    xy = (
        batch.start_xy.unsqueeze(2)
        + progress * safe_length.unsqueeze(2) * unit.unsqueeze(2)
        + lateral * safe_length.unsqueeze(2) * normal.unsqueeze(2)
    )
    xy = torch.where(batch.contact_mask.unsqueeze(-1), xy, torch.zeros_like(xy))

    lengths = batch.point_mask.long().sum(dim=-1)
    for batch_id in range(xy.shape[0]):
        for pointer_id in range(MAX_POINTERS):
            if not bool(batch.pointer_mask[batch_id, pointer_id].item()):
                continue
            n = int(lengths[batch_id, pointer_id].item())
            # Explicit assignment removes floating-point endpoint drift.
            xy[batch_id, pointer_id, 0] = batch.start_xy[batch_id, pointer_id]
            xy[batch_id, pointer_id, n - 1] = batch.end_xy[batch_id, pointer_id]
    return xy


def constrain_and_decode(features: torch.Tensor, batch: TrajectoryBatch) -> ConstrainedTrajectory:
    projected, timestamps = project_local_features(features, batch)
    xy = decode_local_xy(projected, batch)
    phases = torch.full_like(batch.event_ids, -1)
    for batch_id in range(batch.features.shape[0]):
        for pointer_id in range(MAX_POINTERS):
            if not bool(batch.pointer_mask[batch_id, pointer_id].item()):
                continue
            ids = torch.unique(batch.event_ids[batch_id, pointer_id][batch.contact_mask[batch_id, pointer_id]])
            for event_id_tensor in ids:
                event_id = int(event_id_tensor.item())
                positions = torch.nonzero(batch.event_ids[batch_id, pointer_id] == event_id, as_tuple=False).reshape(-1)
                phases[batch_id, pointer_id, positions] = 1  # MOVE
                phases[batch_id, pointer_id, positions[0]] = 0  # DOWN
                phases[batch_id, pointer_id, positions[-1]] = 2  # UP
    return ConstrainedTrajectory(
        features=projected,
        xy=xy,
        timestamps_ms=timestamps,
        pressure=projected[..., PRESSURE],
        size=projected[..., SIZE],
        point_mask=batch.point_mask.clone(),
        contact_mask=batch.contact_mask.clone(),
        event_ids=batch.event_ids.clone(),
        contact_phase=phases,
        pointer_mask=batch.pointer_mask.clone(),
        duration_ms=batch.duration_ms.clone(),
        pointer_start_offset_ms=batch.pointer_start_offset_ms.clone(),
        pointer_end_offset_ms=batch.pointer_end_offset_ms.clone(),
    )
