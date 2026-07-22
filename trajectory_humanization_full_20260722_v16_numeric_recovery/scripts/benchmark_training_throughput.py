#!/usr/bin/env python3
"""Measure formal trajectory-training throughput without creating checkpoints.

The benchmark consumes the real, uncapped training split through the exact
five-reference dataset/collator and runs genuine AMP forward/backward/optimizer
steps.  It never reads validation/test targets and never mutates a formal run
directory.  One JSON result is written atomically so batch-size selection is
auditable rather than guessed from allocated VRAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_determinism import CUBLAS_WORKSPACE_CONFIG  # noqa: F401,E402

import numpy as np
import torch

from trajectory.data import FORMAL_REF_COUNT, KEYCODE_VOCAB_SIZE
from trajectory.model import TrajectoryDiffusion
from training.corpus import ACTIONS, FORMAL_SPLIT_PATH, NumericTrajectoryCorpus, SplitDefinition, atomic_json_dump
from training.engine import (
    ExponentialMovingAverage,
    runtime_determinism_audit,
    seed_everything,
)
from training.fewshot_dataset import (
    ReferenceRegistry,
    DeterministicLengthBucketBatchSampler,
    StrictFiveReferenceDataset,
    StrictVariableLengthCollator,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real optimizer steps to measure one action/batch-size candidate."
    )
    parser.add_argument("--action", required=True, choices=ACTIONS)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, default=FORMAL_SPLIT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=96)
    parser.add_argument("--cond-dim", type=int, default=192)
    parser.add_argument("--time-dim", type=int, default=96)
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--keycode-vocab", type=int, default=KEYCODE_VOCAB_SIZE)
    parser.add_argument("--reference-cache-size", type=int, default=2048)
    parser.add_argument("--no-amp", action="store_true")
    return parser


PROFILE_EPOCHS = (0, 1, 2, 3, 4)


def _batch_profiles(dataset, batch_size: int):
    """Describe exact target/ref padding for five deterministic train epochs."""
    profiles = []
    for epoch in PROFILE_EPOCHS:
        sampler = DeterministicLengthBucketBatchSampler(
            dataset, batch_size=batch_size, epoch=epoch, shuffle=True
        )
        epoch_profiles = []
        for batch_index, positions in enumerate(sampler):
            components = [dataset.padded_shape_components(position) for position in positions]
            target_t = max(value[0] for value in components)
            reference_t = max(value[1] for value in components)
            target_k = max(value[2] for value in components)
            reference_k = max(value[3] for value in components)
            count = len(positions)
            # Keep all four independent padded dimensions in the compute proxy.
            # Runtime is still measured empirically; this scalar only orders
            # interpolation anchors and is not claimed to be exact FLOPs.
            padded_work = count * (
                target_t + FORMAL_REF_COUNT * reference_t
                + target_k + FORMAL_REF_COUNT * reference_k
            )
            epoch_profiles.append({
                "epoch": int(epoch),
                "batch_index": int(batch_index),
                "positions": tuple(int(value) for value in positions),
                "batch_size": int(count),
                "target_padded_t": int(target_t),
                "reference_padded_t": int(reference_t),
                "target_keycode_padded_k": int(target_k),
                "reference_keycode_padded_k": int(reference_k),
                "padded_work": int(padded_work),
            })
        if len(epoch_profiles) != len(sampler):
            raise AssertionError("epoch profile batch count mismatch")
        profiles.append(epoch_profiles)
    return profiles


def _representative_profiles(epoch_profiles, count: int):
    """Choose deterministic, tail-dense measured batches from the full profile."""
    available = sorted(
        [profile for epoch in epoch_profiles for profile in epoch],
        key=lambda value: (
            value["padded_work"], value["target_padded_t"],
            value["reference_padded_t"], value["epoch"], value["batch_index"],
        ),
    )
    if count <= 0 or count > len(available):
        raise ValueError("representative count escapes available epoch batches")
    if count == 1:
        ranks = [len(available) - 1]
    elif count < 4:
        ranks = np.rint(np.linspace(0, len(available) - 1, count)).astype(np.int64).tolist()
    else:
        # Reserve three points for the expensive tail.  The previous random
        # prefix missed this region entirely for keystroke.
        quantiles = np.concatenate([
            np.linspace(0.0, 0.90, count - 3),
            np.asarray([0.95, 0.99, 1.0], dtype=np.float64),
        ])
        ranks = np.rint(quantiles * float(len(available) - 1)).astype(np.int64).tolist()
    chosen = []
    used = set()
    for rank in ranks:
        rank = int(rank)
        if rank not in used:
            chosen.append(available[rank])
            used.add(rank)
    if len(chosen) < count:
        for rank in range(len(available) - 1, -1, -1):
            if rank not in used:
                chosen.append(available[rank])
                used.add(rank)
            if len(chosen) == count:
                break
    return chosen


def _project_epoch_seconds(epoch_profiles, measurements):
    """Monotone piecewise interpolation over exact padded target/ref work."""
    by_work = {}
    for value in measurements:
        work = int(value["padded_work"])
        by_work.setdefault(work, []).append(float(value["elapsed_seconds"]))
    x = np.asarray(sorted(by_work), dtype=np.float64)
    # Use the slower observation for duplicate work and enforce a monotone
    # envelope.  This is deliberately conservative under timing noise.
    y = np.asarray([max(by_work[int(work)]) for work in x], dtype=np.float64)
    y = np.maximum.accumulate(y)
    all_work = np.asarray(
        [row["padded_work"] for epoch in epoch_profiles for row in epoch],
        dtype=np.float64,
    )
    has_extrapolation = bool(
        all_work.size == 0 or all_work.min() < x.min() or all_work.max() > x.max()
    )
    if has_extrapolation:
        raise AssertionError("epoch projection would require extrapolation")
    if x.size == 1:
        predicted = [[float(y[0])] * len(epoch) for epoch in epoch_profiles]
    else:
        predicted = [
            np.interp(
                np.asarray([row["padded_work"] for row in epoch], dtype=np.float64),
                x, y,
            ).tolist()
            for epoch in epoch_profiles
        ]
    epoch_seconds = [float(sum(values)) for values in predicted]
    return {
        "method": "monotone_piecewise_linear_exact_t_tr_k_kr_padding_v2",
        "fit_padded_work": [int(value) for value in x.tolist()],
        "fit_elapsed_seconds": [float(value) for value in y.tolist()],
        "projection_has_extrapolation": False,
        "profile_epochs": [int(value) for value in PROFILE_EPOCHS],
        "epoch_optimizer_seconds": epoch_seconds,
        "mean_epoch_optimizer_seconds": float(statistics.fmean(epoch_seconds)),
        "min_epoch_optimizer_seconds": float(min(epoch_seconds)),
        "max_epoch_optimizer_seconds": float(max(epoch_seconds)),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.steps <= 0 or args.warmup_steps <= 0:
        raise ValueError("batch-size/steps/warmup must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    split = SplitDefinition.load(args.split_json, require_pinned_hash=True)
    corpus_path = args.corpus_dir.resolve() / ("hmog_trajectory_%s.npz" % args.action)
    corpus = NumericTrajectoryCorpus(corpus_path, split, expected_action=args.action, verify_sha256=True)
    registry = ReferenceRegistry.build(corpus, seed=args.seed)
    dataset = StrictFiveReferenceDataset(
        corpus, "train", registry, seed=args.seed, cache_size=args.reference_cache_size
    )
    epoch_profiles = _batch_profiles(dataset, int(args.batch_size))
    representatives = _representative_profiles(epoch_profiles, int(args.steps))
    flat_profiles = [profile for epoch in epoch_profiles for profile in epoch]
    if not flat_profiles:
        raise ValueError("empty benchmark epoch profile")

    all_components = [dataset.padded_shape_components(position) for position in range(len(dataset))]
    worst_count = min(int(args.batch_size), len(dataset))
    # The formal bucket shuffles can combine a maximum target and a maximum
    # reference item in later epochs.  Force both extrema into a full safety
    # batch, then fill it by exact target+five-reference padded work.
    ordered = sorted(
        range(len(dataset)),
        key=lambda position: (
            all_components[position][0] + FORMAL_REF_COUNT * all_components[position][1]
            + all_components[position][2] + FORMAL_REF_COUNT * all_components[position][3],
            *all_components[position], position,
        ),
        reverse=True,
    )
    required = [
        max(range(len(dataset)), key=lambda position: (all_components[position][0], position)),
        max(range(len(dataset)), key=lambda position: (all_components[position][1], position)),
        max(range(len(dataset)), key=lambda position: (all_components[position][2], position)),
        max(range(len(dataset)), key=lambda position: (all_components[position][3], position)),
    ]
    longest_positions = []
    for position in required + ordered:
        if position not in longest_positions:
            longest_positions.append(int(position))
        if len(longest_positions) == worst_count:
            break
    worst_batch = StrictVariableLengthCollator()(
        [dataset[position] for position in longest_positions]
    )
    worst_case_padded_t = int(worst_batch.features.shape[2])
    worst_case_reference_padded_t = int(worst_batch.ref_features.shape[3])
    worst_case_keycode_padded_k = int(worst_batch.keycodes.shape[1])
    worst_case_reference_keycode_padded_k = int(worst_batch.ref_keycodes.shape[2])
    worst_case_padded_work = int(
        worst_count * (
            worst_case_padded_t + FORMAL_REF_COUNT * worst_case_reference_padded_t
            + worst_case_keycode_padded_k
            + FORMAL_REF_COUNT * worst_case_reference_keycode_padded_k
        )
    )

    profile_payload = [
        {
            "epoch": int(row["epoch"]),
            "batch_index": int(row["batch_index"]),
            "positions": list(row["positions"]),
            "batch_size": int(row["batch_size"]),
            "target_padded_t": int(row["target_padded_t"]),
            "reference_padded_t": int(row["reference_padded_t"]),
            "target_keycode_padded_k": int(row["target_keycode_padded_k"]),
            "reference_keycode_padded_k": int(row["reference_keycode_padded_k"]),
            "padded_work": int(row["padded_work"]),
        }
        for row in flat_profiles
    ]
    epoch_profile_sha256 = hashlib.sha256(
        json.dumps(profile_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    model = TrajectoryDiffusion(
        args.action,
        diffusion_steps=args.diffusion_steps,
        base_channels=args.base_channels,
        cond_dim=args.cond_dim,
        time_dim=args.time_dim,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
        keycode_vocab=args.keycode_vocab,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_enabled = bool(not args.no_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ema = ExponentialMovingAverage(model, args.ema_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 8849)

    losses = []
    examples = 0
    valid_features = 0.0
    padded_points = 0
    max_padded_t = 0
    max_reference_padded_t = 0
    timed_seconds = 0.0
    measured_steps = 0
    projection_measurements = []
    model.train()

    collator = StrictVariableLengthCollator()

    def collate_profile(profile):
        return collator([dataset[position] for position in profile["positions"]])

    def optimizer_step(cpu_batch, label: str, padded_work: int):
        batch = cpu_batch.to(device)
        b = int(batch.features.shape[0])
        observed_padded_work = int(
            b * (
                int(batch.features.shape[2])
                + FORMAL_REF_COUNT * int(batch.ref_features.shape[3])
                + int(batch.keycodes.shape[1])
                + FORMAL_REF_COUNT * int(batch.ref_keycodes.shape[2])
            )
        )
        if observed_padded_work != int(padded_work):
            raise AssertionError("profile padded work contradicts collated batch")
        timesteps = torch.randint(
            0, model.diffusion_steps, (b,), device=device, dtype=torch.long,
            generator=generator,
        )
        noise = torch.randn(
            batch.features.shape, dtype=batch.features.dtype, device=device,
            generator=generator,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            result = model.training_loss(batch, timesteps=timesteps, noise=noise)
            loss = result["loss"]
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite benchmark loss at %s" % label)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
            raise FloatingPointError("non-finite benchmark gradient norm at %s" % label)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        return {
            "label": label,
            "elapsed_seconds": float(elapsed),
            "loss": float(loss.detach().item()),
            "batch_size": b,
            "valid_feature_count": float(result["valid_feature_count"].detach().item()),
            "target_padded_t": int(batch.features.shape[2]),
            "reference_padded_t": int(batch.ref_features.shape[3]),
            "target_keycode_padded_k": int(batch.keycodes.shape[1]),
            "reference_keycode_padded_k": int(batch.ref_keycodes.shape[2]),
            "padded_work": int(padded_work),
            "padded_pointer_points": int(np.prod(batch.features.shape[:-1])),
        }

    warmup_profile = sorted(flat_profiles, key=lambda value: value["padded_work"])[
        len(flat_profiles) // 2
    ]
    warmup_batch = collate_profile(warmup_profile)
    for warmup_index in range(int(args.warmup_steps)):
        optimizer_step(
            warmup_batch, "warmup_%03d" % warmup_index,
            int(warmup_profile["padded_work"]),
        )

    warmed_shapes = set()
    worst_shape = (
        int(worst_batch.features.shape[0]),
        worst_case_padded_t,
        worst_case_reference_padded_t,
        worst_case_keycode_padded_k,
        worst_case_reference_keycode_padded_k,
    )
    optimizer_step(
        worst_batch, "shape_warmup_global_worst_case", worst_case_padded_work
    )
    warmed_shapes.add(worst_shape)
    shape_specific_warmup_steps = 1
    worst_measurement = optimizer_step(
        worst_batch, "artificial_global_worst_case", worst_case_padded_work
    )
    projection_measurements.append(worst_measurement)

    for representative_index, profile in enumerate(representatives):
        profile_batch = collate_profile(profile)
        profile_shape = (
            int(profile_batch.features.shape[0]),
            int(profile_batch.features.shape[2]),
            int(profile_batch.ref_features.shape[3]),
            int(profile_batch.keycodes.shape[1]),
            int(profile_batch.ref_keycodes.shape[2]),
        )
        if profile_shape not in warmed_shapes:
            optimizer_step(
                profile_batch,
                "shape_warmup_profile_%03d" % representative_index,
                int(profile["padded_work"]),
            )
            warmed_shapes.add(profile_shape)
            shape_specific_warmup_steps += 1
        measurement = optimizer_step(
            profile_batch,
            "profile_%03d_epoch_%d_batch_%d" % (
                representative_index, profile["epoch"], profile["batch_index"]
            ),
            int(profile["padded_work"]),
        )
        projection_measurements.append(measurement)
        losses.append(float(measurement["loss"]))
        timed_seconds += float(measurement["elapsed_seconds"])
        measured_steps += 1
        examples += int(measurement["batch_size"])
        valid_features += float(measurement["valid_feature_count"])
        padded_points += int(measurement["padded_pointer_points"])
        max_padded_t = max(max_padded_t, int(measurement["target_padded_t"]))
        max_reference_padded_t = max(
            max_reference_padded_t, int(measurement["reference_padded_t"])
        )

    if measured_steps != args.steps or timed_seconds <= 0 or not losses:
        raise AssertionError("benchmark measurement count mismatch")
    projection = _project_epoch_seconds(epoch_profiles, projection_measurements)
    projected_epoch_seconds = float(projection["mean_epoch_optimizer_seconds"])
    if not math.isfinite(projected_epoch_seconds) or projected_epoch_seconds <= 0:
        raise AssertionError("invalid projected full-epoch optimizer time")
    projected_epoch_examples_per_second = float(len(dataset) / projected_epoch_seconds)
    window = max(1, min(20, len(losses) // 5 if len(losses) >= 5 else 1))
    peak_allocated = 0
    peak_reserved = 0
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    benchmark_config = {
        "action": args.action,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "measured_steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "grad_clip_norm": float(args.grad_clip_norm),
        "ema_decay": float(args.ema_decay),
        "diffusion_steps": int(args.diffusion_steps),
        "base_channels": int(args.base_channels),
        "cond_dim": int(args.cond_dim),
        "time_dim": int(args.time_dim),
        "n_blocks": int(args.n_blocks),
        "dropout": float(args.dropout),
        "keycode_vocab": int(args.keycode_vocab),
        "reference_cache_size": int(args.reference_cache_size),
        "amp": bool(amp_enabled),
        "optimizer": "AdamW",
        "profile_epochs": list(PROFILE_EPOCHS),
    }
    benchmark_config_sha256 = hashlib.sha256(
        json.dumps(benchmark_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "schema_version": "trajectory_training_throughput_v2",
        "formal_training": False,
        "uses_exact_formal_train_loader_and_model": True,
        "reads_validation_or_test_targets": False,
        "creates_or_updates_formal_checkpoint": False,
        "action": args.action,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "warmup_steps": int(args.warmup_steps),
        "worst_case_safety_optimizer_steps": 1,
        "worst_case_batch_size": int(worst_count),
        "worst_case_padded_t": int(worst_case_padded_t),
        "worst_case_reference_padded_t": int(worst_case_reference_padded_t),
        "worst_case_keycode_padded_k": int(worst_case_keycode_padded_k),
        "worst_case_reference_keycode_padded_k": int(
            worst_case_reference_keycode_padded_k
        ),
        "worst_case_padded_work": int(worst_case_padded_work),
        "worst_case_elapsed_seconds": float(worst_measurement["elapsed_seconds"]),
        "longest_uncapped_batch_exercised": True,
        "exact_canonical_target_and_reference_lengths": True,
        "exact_target_and_reference_keycode_lengths": True,
        "projection_includes_worst_case_measurement": True,
        "optimizer_state_initialization_steps": int(args.warmup_steps),
        "optimizer_state_initialization_excluded_from_projection": True,
        "shape_specific_warmup_optimizer_steps": int(shape_specific_warmup_steps),
        "shape_specific_warmup_excluded_from_projection": True,
        "total_unmeasured_optimizer_steps": int(
            args.warmup_steps + shape_specific_warmup_steps
        ),
        "projection_has_extrapolation": False,
        "profile_epoch_count": len(PROFILE_EPOCHS),
        "profile_epoch_batch_counts": [len(epoch) for epoch in epoch_profiles],
        "profile_target_occurrences": int(
            sum(row["batch_size"] for row in flat_profiles)
        ),
        "profile_each_epoch_covers_dataset_once": True,
        "epoch_length_profile_sha256": epoch_profile_sha256,
        "projection_measurement_count": len(projection_measurements),
        "measured_optimizer_steps": int(measured_steps),
        "dataset_target_count": int(len(dataset)),
        "corpus_npz": str(corpus.path),
        "corpus_sha256": corpus.sha256,
        "split_sha256": split.sha256,
        "reference_registry_sha256": registry.sha256,
        "amp": amp_enabled,
        "runtime_determinism": runtime_determinism_audit(),
        "diffusion_steps": int(args.diffusion_steps),
        "keycode_vocab": int(args.keycode_vocab),
        "benchmark_config": benchmark_config,
        "benchmark_config_sha256": benchmark_config_sha256,
        "elapsed_seconds": float(timed_seconds),
        "steps_per_second": float(measured_steps / timed_seconds),
        "examples_per_second": float(examples / timed_seconds),
        "padded_pointer_points_per_second": float(padded_points / timed_seconds),
        "valid_features_per_second": float(valid_features / timed_seconds),
        "projected_full_epoch_optimizer_seconds": projected_epoch_seconds,
        "projected_full_epoch_examples_per_second": projected_epoch_examples_per_second,
        "projected_100_epoch_optimizer_hours": float(
            100.0 * projected_epoch_seconds / 3600.0
        ),
        "epoch_projection": projection,
        "projection_measurements": projection_measurements,
        "max_padded_t": int(max_padded_t),
        "max_reference_padded_t": int(max_reference_padded_t),
        "loss": {
            "first": float(losses[0]),
            "last": float(losses[-1]),
            "mean": float(statistics.fmean(losses)),
            "first_window_median": float(statistics.median(losses[:window])),
            "last_window_median": float(statistics.median(losses[-window:])),
            "window_size": int(window),
            "all_finite": all(math.isfinite(value) for value in losses),
        },
        "cuda_peak_memory_allocated_bytes": peak_allocated,
        "cuda_peak_memory_reserved_bytes": peak_reserved,
        "passed": True,
    }
    atomic_json_dump(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
