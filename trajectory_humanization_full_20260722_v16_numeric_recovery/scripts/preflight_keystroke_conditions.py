#!/usr/bin/env python3
"""Deterministic 100-user x 200-sample keystroke condition preflight.

This is a metadata/topology gate, not neural generation and not a reported PAD
result.  It exercises the exact fixed-five-reference registry, train-only
prior, condition policy, sampling-batch construction and hard timeline
projection for every formal user/sample index before expensive DDIM sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.batching import build_sampling_batch  # noqa: E402
from generation.protocol import (  # noqa: E402
    FixedUserSplit,
    ReferenceConditionPolicy,
    ReferenceRegistry as GenerationReferenceRegistry,
    TrainGlobalPrior,
)
from trajectory.constraints import constrain_and_decode  # noqa: E402
from training.corpus import NumericTrajectoryCorpus, SplitDefinition, sha256_file  # noqa: E402
from training.fewshot_dataset import ReferenceRegistry as TrainingReferenceRegistry  # noqa: E402


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def _registry_for_full_corpus(corpus: NumericTrajectoryCorpus, seed: int):
    training = TrainingReferenceRegistry.build(corpus, seed=seed)
    entries = {}
    for row in training.payload["entries"]:
        entries[(row["action"], int(row["user_id"]), row["split"])] = tuple(
            int(value) for value in row["reference_event_ids"]
        )
    generation = GenerationReferenceRegistry.build(entries, corpus.splits.sha256)
    if generation.registry_sha256 != training.sha256:
        raise AssertionError("training/generation reference-registry hashes differ")
    return generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "results/trajectories_full_v2/hmog_trajectory_keystroke.npz",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        default=Path("/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/v2_keystroke_condition_preflight/preflight.json",
    )
    parser.add_argument("--reference-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=20260713)
    parser.add_argument("--samples-per-user", type=int, default=200)
    args = parser.parse_args()
    if args.samples_per_user != 200:
        raise ValueError("formal preflight requires exactly 200 samples per user")

    started = time.time()
    split = SplitDefinition.load(args.split_json, require_pinned_hash=True)
    fixed_split = FixedUserSplit.load(str(args.split_json), require_formal=True)
    corpus = NumericTrajectoryCorpus(
        args.corpus, split, expected_action="keystroke", verify_sha256=True
    )
    corpus.audit(require_all_users=True, validate_every_event=False)
    print("[preflight] authoritative corpus + 100-user audit loaded", flush=True)
    registry = _registry_for_full_corpus(corpus, args.reference_seed)
    print("[preflight] fixed five-reference registry built", flush=True)
    # Fit directly from the compact authoritative numeric corpus.  The prior
    # loader recreates canonical rows lazily and never retains all ~34k train
    # CanonicalTrajectory objects simultaneously.
    prior = TrainGlobalPrior.fit("keystroke", corpus, fixed_split.train_users)
    print("[preflight] streaming 70-user train-only prior fitted", flush=True)
    if set(int(value) for value in prior.source_user_ids.tolist()) != set(fixed_split.train_users):
        raise ValueError("train-only prior does not cover exactly the 70 fixed train users")
    event_to_index = {
        int(event_id): index for index, event_id in enumerate(corpus.event_ids.tolist())
    }
    required_reference_ids = sorted({
        int(event_id)
        for ids in registry.entries.values()
        for event_id in ids
    })
    if len(required_reference_ids) != 500:
        raise ValueError("formal keystroke registry must contain 500 distinct references")
    pool = [corpus.canonical_sample(event_to_index[event_id]) for event_id in required_reference_ids]
    print("[preflight] exact 500 fixed references restored", flush=True)
    policy = ReferenceConditionPolicy(prior)

    totals = {
        "requests": 0,
        "users": 0,
        "zero_flight_boundaries": 0,
        "positive_flight_boundaries": 0,
        "projected_zero_intervals": 0,
        "projected_positive_intervals": 0,
    }
    min_values = {"duration_ms": float("inf"), "n_keys": 1 << 30, "points": 1 << 30}
    max_values = {"duration_ms": 0.0, "n_keys": 0, "points": 0}
    signature = hashlib.sha256()

    for user_id in sorted(fixed_split.all_users):
        split_name = fixed_split.split_for_user(user_id)
        refs = registry.resolve(pool, "keystroke", user_id, split_name)
        requests = [
            policy.sample(
                "keystroke", user_id, split_name, sample_index,
                args.generation_seed, refs,
            )
            for sample_index in range(args.samples_per_user)
        ]
        batch = build_sampling_batch(
            requests, [refs] * len(requests), torch.device("cpu")
        )
        output = constrain_and_decode(torch.zeros_like(batch.features), batch)
        for row_index, request in enumerate(requests):
            n = int(request.lengths[0])
            times = output.timestamps_ms[row_index, 0, :n].detach().cpu().numpy()
            observed_zero = np.diff(times) == 0
            contacts = request.contact_masks[0]
            events = request.event_ids[0]
            allowed_zero = (
                contacts[:-1]
                & contacts[1:]
                & (events[:-1] >= 0)
                & (events[1:] == events[:-1] + 1)
            )
            if not np.array_equal(observed_zero, allowed_zero):
                raise ValueError(
                    "projected zero/positive timeline mismatch for user=%d sample=%d"
                    % (user_id, request.sample_index)
                )
            totals["requests"] += 1
            totals["zero_flight_boundaries"] += int(np.sum(request.zero_flight_after_key))
            totals["positive_flight_boundaries"] += int(
                request.zero_flight_after_key.size - np.sum(request.zero_flight_after_key)
            )
            totals["projected_zero_intervals"] += int(np.sum(observed_zero))
            totals["projected_positive_intervals"] += int(np.sum(~observed_zero))
            values = {
                "duration_ms": float(request.duration_ms),
                "n_keys": int(request.n_keys),
                "points": n,
            }
            for key, value in values.items():
                min_values[key] = min(min_values[key], value)
                max_values[key] = max(max_values[key], value)
            signature.update(np.asarray([
                request.fake_id, request.seed, request.n_keys, request.n_letters, n,
            ], np.int64).tobytes())
            signature.update(np.asarray(request.keycodes, np.int32).tobytes())
            signature.update(np.asarray(request.zero_flight_after_key, np.uint8).tobytes())
        totals["users"] += 1
        if totals["users"] % 10 == 0:
            print(
                "[preflight] users=%d/100 requests=%d/%d"
                % (totals["users"], totals["requests"], 100 * args.samples_per_user),
                flush=True,
            )

    expected = 100 * args.samples_per_user
    if totals["users"] != 100 or totals["requests"] != expected:
        raise AssertionError("preflight did not cover exact 100x200 conditions")
    result = {
        "schema_version": "keystroke_condition_preflight_v1",
        "status": "passed",
        "formal_result": False,
        "purpose": "pre-DDIM full-count deterministic metadata/topology gate",
        "source": str(args.corpus.resolve()),
        "source_sha256": sha256_file(args.corpus),
        "split_sha256": fixed_split.source_sha256,
        "reference_registry_sha256": registry.registry_sha256,
        "train_prior_sha256": prior.digest,
        "reference_seed": int(args.reference_seed),
        "generation_seed": int(args.generation_seed),
        "samples_per_user": int(args.samples_per_user),
        "counts": totals,
        "minimum": min_values,
        "maximum": max_values,
        "condition_signature_sha256": signature.hexdigest(),
        "exact_fixed_reference_count": 5,
        "train_prior_user_count": len(set(int(x) for x in prior.source_user_ids.tolist())),
        "train_prior_source_event_count": int(prior.source_event_ids.size),
        "train_prior_key_count": int(prior.keycodes.size),
        "train_prior_flight_count": int(prior.key_zero_flight.size),
        "train_prior_numeric_bytes": int(sum(
            value.nbytes for value in prior.__dict__.values()
            if isinstance(value, np.ndarray)
        )),
        "train_prior_only_fixed_train_users": True,
        "all_requests_validated": True,
        "all_sampling_batches_validated": True,
        "all_hard_timeline_projections_validated": True,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
