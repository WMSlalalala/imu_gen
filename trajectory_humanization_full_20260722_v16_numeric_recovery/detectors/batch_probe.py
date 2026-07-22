"""Worst-length one-step memory probe for untruncated raw Deep PAD batches."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from detectors.deep_pad import (
    RawSequenceNormalizer,
    RawTrajectoryRecord,
    _configure_deterministic_backend,
    _require_finite_model,
    _seed_everything,
    collate_raw_sequences,
    make_deep_model,
)
from detectors.pair_runner import sha256_file


PROBE_SCHEMA = "trajectory_deep_batch_probe_v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in text


def _probe_once(
    record: RawTrajectoryRecord,
    normalizer: RawSequenceNormalizer,
    detector: str,
    model_params: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> int:
    _seed_everything(seed)
    _configure_deterministic_backend(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = make_deep_model(detector, model_params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, foreach=False)
    batch = collate_raw_sequences([record] * int(batch_size), normalizer).to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = F.binary_cross_entropy_with_logits(logits, batch.labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("batch probe produced non-finite loss")
    loss.backward()
    _require_finite_model(model, "batch probe backward")
    optimizer.step()
    _require_finite_model(model, "batch probe optimizer")
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    del loss, logits, batch, optimizer, model
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return peak


def probe_max_safe_batch(
    records: Sequence[RawTrajectoryRecord],
    *,
    action: str,
    detector: str,
    model_params: Optional[Mapping[str, Any]],
    requested_batch_size: int,
    device: str,
    seed: int,
    dataset_file: Path,
    fake_user_split: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Binary-search the largest batch <= requested on the longest raw event."""

    if requested_batch_size <= 0:
        raise ValueError("requested batch size must be positive")
    action_rows = [row for row in records if row.action == action and row.pool == "train"]
    if not action_rows:
        raise ValueError("batch probe has no train records for %s" % action)
    real_rows = [row for row in action_rows if row.label == 0]
    fake_rows = [row for row in action_rows if row.label == 1]
    if not real_rows or not fake_rows:
        raise ValueError("batch probe requires real and fake train records")
    longest_real = max(real_rows, key=lambda row: len(row.global_t_ms))
    longest_fake = max(fake_rows, key=lambda row: len(row.global_t_ms))
    longest = max((longest_real, longest_fake), key=lambda row: len(row.global_t_ms))
    max_observed_length = max(len(row.global_t_ms) for row in action_rows)
    if len(longest.global_t_ms) != max_observed_length:
        raise AssertionError("probe did not select the longest untruncated event")
    normalizer = RawSequenceNormalizer().fit(action_rows)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA batch probe requested but CUDA is unavailable")
    params = dict(model_params or {})
    low, high = 1, int(requested_batch_size)
    best = 0
    peak_by_batch: Dict[str, int] = {}
    oom_batches = []
    while low <= high:
        candidate = (low + high) // 2
        try:
            peak = _probe_once(
                longest, normalizer, detector, params, candidate,
                selected_device, seed + candidate,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            if not _is_oom(error):
                raise
            oom_batches.append(candidate)
            if selected_device.type == "cuda":
                torch.cuda.empty_cache()
            high = candidate - 1
            continue
        peak_by_batch[str(candidate)] = peak
        best = candidate
        low = candidate + 1
    if best < 1:
        raise RuntimeError("even batch_size=1 OOMs on the longest untruncated event")
    payload = {
        "schema_version": PROBE_SCHEMA,
        "status": "passed",
        "action": action,
        "detector": detector,
        "device": str(selected_device),
        "device_name": (
            torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda"
            else "cpu"
        ),
        "seed": int(seed),
        "model_params": params,
        "dataset_file": str(Path(dataset_file).resolve()),
        "dataset_sha256": sha256_file(dataset_file),
        "fake_user_split": str(Path(fake_user_split).resolve()),
        "fake_user_split_sha256": sha256_file(fake_user_split),
        "requested_batch_size": int(requested_batch_size),
        "max_safe_batch_size": int(best),
        "selected_batch_size": int(best),
        "longest_observed_train_event_length": int(max_observed_length),
        "longest_real_train_event_length": int(len(longest_real.global_t_ms)),
        "longest_fake_train_event_length": int(len(longest_fake.global_t_ms)),
        "probed_sample_id": str(longest.sample_id),
        "probed_sample_label": int(longest.label),
        "truncation": False,
        "resampling": False,
        "probe": "one_full_forward_backward_optimizer_step_on_repeated_longest_event",
        "peak_memory_bytes_by_successful_batch": peak_by_batch,
        "oom_batch_attempts": sorted(oom_batches),
    }
    _atomic_json(output_path, payload)
    return payload


__all__ = ["PROBE_SCHEMA", "probe_max_safe_batch"]
