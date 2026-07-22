#!/usr/bin/env python3
"""Build and validate all 100,000 formal metadata requests before DDIM.

This is a deterministic metadata/topology launch gate, not generated data and
not a PAD result.  Reference enrollment uses the training seed (42); request
sampling uses the independent generation seed (20260713).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
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
    ACTIONS, CONDITION_REQUEST_DIGEST_FIELDS, CONDITION_REQUEST_DIGEST_SCHEMA,
    CONDITION_SET_DIGEST_SCHEMA, FixedUserSplit, ReferenceConditionPolicy,
    ReferenceRegistry as GenerationReferenceRegistry, TrainGlobalPrior,
    canonical_condition_request_digest, condition_request_set_sha256,
    ddim_noise_seed,
)
from trajectory.constraints import constrain_and_decode  # noqa: E402
from trajectory.data import KEYCODE_VOCAB_SIZE  # noqa: E402
from training.corpus import (  # noqa: E402
    NumericTrajectoryCorpus, SplitDefinition, canonical_sample_sha256, sha256_file,
)
from training.fewshot_dataset import ReferenceRegistry as TrainingReferenceRegistry  # noqa: E402


SCHEMA = "trajectory_all_condition_requests_preflight_v1"
PRODUCER_SOURCE_RELATIVE_PATHS = (
    "scripts/preflight_all_condition_requests.py",
    "generation/batching.py",
    "generation/protocol.py",
    "trajectory/constraints.py",
    "trajectory/data.py",
    "training/corpus.py",
    "training/fewshot_dataset.py",
)
_WORKER_CONTEXT = None


def producer_source_identity(root: Path) -> dict:
    files = {
        relative: sha256_file(Path(root) / relative)
        for relative in PRODUCER_SOURCE_RELATIVE_PATHS
    }
    tree = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "trajectory_condition_preflight_source_v1",
        "files": files,
        "tree_sha256": tree,
    }


def _worker_initialize() -> None:
    torch.set_num_threads(1)


def _audit_one_user(user_id: int):
    context = _WORKER_CONTEXT
    if context is None:
        raise RuntimeError("condition-preflight fork context was not initialized")
    action = context["action"]
    fixed_split = context["fixed_split"]
    registry = context["registry"]
    pool = context["pool"]
    policy = context["policy"]
    prior = context["prior"]
    samples = int(context["samples"])
    batch_size = int(context["batch_size"])
    generation_seed = int(context["generation_seed"])
    split_name = fixed_split.split_for_user(user_id)
    refs = registry.resolve(pool, action, user_id, split_name)
    requests = [
        policy.sample(action, user_id, split_name, sample_index, generation_seed, refs)
        for sample_index in range(samples)
    ]
    pairs = []
    fake_ids = []
    condition_seeds = []
    noise_seeds = []
    counts = {
        "requests": 0, "sampling_batches": 0, "condition_source_code_2": 0,
        "zero_flight_boundaries": 0, "positive_flight_boundaries": 0,
        "hard_timeline_projections": 0, "projected_zero_intervals": 0,
    }
    minimum = {"duration_ms": float("inf"), "points": 1 << 30, "n_keys": 1 << 30}
    maximum = {"duration_ms": 0.0, "points": 0, "n_keys": 0}
    reference_hashes = tuple(canonical_sample_sha256(item) for item in refs)
    for start in range(0, len(requests), batch_size):
        request_batch = requests[start : start + batch_size]
        batch = build_sampling_batch(
            request_batch, [refs] * len(request_batch), torch.device("cpu")
        )
        counts["sampling_batches"] += 1
        projected = None
        if action == "keystroke":
            projected = constrain_and_decode(torch.zeros_like(batch.features), batch)
            counts["hard_timeline_projections"] += len(request_batch)
        for index, request in enumerate(request_batch):
            if request.condition_source_code != 2 or request.train_prior_digest != prior.digest:
                raise ValueError("formal request provenance mismatch")
            if tuple(request.reference_canonical_sha256) != reference_hashes:
                raise ValueError("request/reference canonical hash mismatch")
            fake_ids.append(int(request.fake_id))
            condition_seeds.append(int(request.seed))
            noise_seeds.append(int(ddim_noise_seed(
                request.seed, request.action, request.user_id,
                request.sample_index,
            )))
            counts["condition_source_code_2"] += 1
            n_points = int(max(request.lengths))
            minimum["duration_ms"] = min(minimum["duration_ms"], float(request.duration_ms))
            maximum["duration_ms"] = max(maximum["duration_ms"], float(request.duration_ms))
            minimum["points"] = min(minimum["points"], n_points)
            maximum["points"] = max(maximum["points"], n_points)
            minimum["n_keys"] = min(minimum["n_keys"], int(request.n_keys))
            maximum["n_keys"] = max(maximum["n_keys"], int(request.n_keys))
            pairs.append((int(request.fake_id), canonical_condition_request_digest(request)))
            if action == "keystroke":
                n = int(request.lengths[0])
                times = projected.timestamps_ms[index, 0, :n].detach().cpu().numpy()
                observed_zero = np.diff(times) == 0
                contacts = request.contact_masks[0]
                events = request.event_ids[0]
                allowed_zero = (
                    contacts[:-1] & contacts[1:] & (events[:-1] >= 0)
                    & (events[1:] == events[:-1] + 1)
                )
                if not np.array_equal(observed_zero, allowed_zero):
                    raise ValueError("keystroke hard timeline zero-flight mismatch")
                counts["zero_flight_boundaries"] += int(np.sum(request.zero_flight_after_key))
                counts["positive_flight_boundaries"] += int(
                    request.zero_flight_after_key.size - np.sum(request.zero_flight_after_key)
                )
                counts["projected_zero_intervals"] += int(np.sum(observed_zero))
            counts["requests"] += 1
        del batch, projected
    return {
        "user_id": int(user_id), "pairs": pairs, "fake_ids": fake_ids,
        "condition_seeds": condition_seeds, "noise_seeds": noise_seeds,
        "counts": counts, "minimum": minimum, "maximum": maximum,
    }


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def compatible_registry(corpus: NumericTrajectoryCorpus, reference_seed: int):
    training = TrainingReferenceRegistry.build(corpus, seed=reference_seed)
    entries = {
        (row["action"], int(row["user_id"]), row["split"]): tuple(
            int(value) for value in row["reference_event_ids"]
        )
        for row in training.payload["entries"]
    }
    generation = GenerationReferenceRegistry.build(entries, corpus.splits.sha256)
    if generation.registry_sha256 != training.sha256:
        raise AssertionError("training/generation registry serialization/hash mismatch")
    return training, generation


def validate_codebook(corpus: NumericTrajectoryCorpus) -> dict:
    raw = np.asarray(corpus._arrays["keycode"], np.int32)
    observed = np.asarray(corpus._arrays["key_is_letter"], np.uint8)
    expected = (((raw >= 65) & (raw <= 90)) | ((raw >= 97) & (raw <= 122))).astype(np.uint8)
    if not np.array_equal(observed, expected):
        raise ValueError("keystroke key_is_letter is not the ASCII A-Z/a-z codebook")
    offsets = np.asarray(corpus._arrays["event_key_offsets"], np.int64)
    counts = np.add.reduceat(expected, offsets[:-1])
    empty = np.diff(offsets) == 0
    counts[empty] = 0
    if not np.array_equal(counts.astype(np.int16), corpus._arrays["n_letters"].astype(np.int16)):
        raise ValueError("event n_letters disagrees with ASCII keycode count")
    # Negative HMOG sentinel/special codes are preserved losslessly in raw
    # metadata and map to neural token 0; only nonnegative codes consume the
    # finite model vocabulary.
    if raw.size and int(raw.max()) >= KEYCODE_VOCAB_SIZE:
        raise ValueError("raw keycode exceeds lossless model vocabulary")
    return {
        "definition": "ASCII 65-90 or 97-122",
        "keycode_vocab": int(KEYCODE_VOCAB_SIZE),
        "raw_min": None if not raw.size else int(raw.min()),
        "raw_max": None if not raw.size else int(raw.max()),
        "ellipsis_u2026_observed": bool(np.any(raw == 8230)),
        "per_key_and_event_counts_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-seed", type=int, default=42)
    parser.add_argument("--generation-seed", type=int, default=20260713)
    parser.add_argument("--samples-per-user-action", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.reference_seed != 42 or args.generation_seed != 20260713:
        raise ValueError("formal gate fixes reference_seed=42 and generation_seed=20260713")
    if args.samples_per_user_action != 200:
        raise ValueError("formal gate requires 200 requests per user/action")
    if args.batch_size != 32:
        raise ValueError("formal gate must reproduce generation batch_size=32")
    if not 1 <= int(args.workers) <= 32:
        raise ValueError("condition preflight workers must be in [1,32]")

    started = time.time()
    source_identity = producer_source_identity(ROOT)
    split = SplitDefinition.load(args.split_json, require_pinned_hash=True)
    fixed_split = FixedUserSplit.load(str(args.split_json), require_formal=True)
    total_requests = 0
    all_request_digests = []
    per_action = {}
    registry_hashes = {}
    prior_hashes = {}
    codebook = None
    seen_fake_ids = set()
    seen_condition_seeds = set()
    seen_noise_seeds = set()
    expected_batches_per_user = (
        args.samples_per_user_action + args.batch_size - 1
    ) // args.batch_size

    for action in ACTIONS:
        corpus_path = args.corpus_dir / ("hmog_trajectory_%s.npz" % action)
        corpus = NumericTrajectoryCorpus(
            corpus_path, split, expected_action=action, verify_sha256=True
        )
        corpus.audit(require_all_users=True, validate_every_event=False)
        training_registry, registry = compatible_registry(corpus, args.reference_seed)
        registry_hashes[action] = training_registry.sha256
        prior = TrainGlobalPrior.fit(action, corpus, fixed_split.train_users)
        prior_hashes[action] = prior.digest
        if set(int(value) for value in prior.source_user_ids.tolist()) != set(fixed_split.train_users):
            raise ValueError("%s prior does not use exactly the fixed 70 train users" % action)
        event_to_index = {
            int(event_id): index for index, event_id in enumerate(corpus.event_ids.tolist())
        }
        reference_ids = sorted({
            int(event_id) for ids in registry.entries.values() for event_id in ids
        })
        if len(reference_ids) != 500:
            raise ValueError("%s registry must contain 500 distinct refs" % action)
        pool = [corpus.canonical_sample(event_to_index[event_id]) for event_id in reference_ids]
        policy = ReferenceConditionPolicy(prior)
        action_request_digests = []
        action_fake_ids = set()
        action_condition_seeds = set()
        action_noise_seeds = set()
        counts = {
            "users": 0, "requests": 0, "sampling_batches": 0,
            "condition_source_code_2": 0, "unique_fake_ids": 0,
            "zero_flight_boundaries": 0, "positive_flight_boundaries": 0,
            "hard_timeline_projections": 0, "projected_zero_intervals": 0,
        }
        minimum = {"duration_ms": float("inf"), "points": 1 << 30, "n_keys": 1 << 30}
        maximum = {"duration_ms": 0.0, "points": 0, "n_keys": 0}
        global _WORKER_CONTEXT
        _WORKER_CONTEXT = {
            "action": action, "fixed_split": fixed_split, "registry": registry,
            "pool": pool, "policy": policy, "prior": prior,
            "samples": args.samples_per_user_action, "batch_size": args.batch_size,
            "generation_seed": args.generation_seed,
        }
        user_ids = sorted(fixed_split.all_users)
        if int(args.workers) == 1:
            user_results = [_audit_one_user(user_id) for user_id in user_ids]
        else:
            if "fork" not in mp.get_all_start_methods():
                raise RuntimeError("formal parallel condition preflight requires fork")
            with mp.get_context("fork").Pool(
                processes=int(args.workers), initializer=_worker_initialize,
            ) as worker_pool:
                user_results = worker_pool.map(_audit_one_user, user_ids, chunksize=1)
        _WORKER_CONTEXT = None
        for user_result in user_results:
            user_fake_ids = [int(value) for value in user_result["fake_ids"]]
            user_condition_seeds = [
                int(value) for value in user_result["condition_seeds"]
            ]
            user_noise_seeds = [int(value) for value in user_result["noise_seeds"]]
            if len(user_fake_ids) != len(set(user_fake_ids)):
                raise ValueError("duplicate formal fake_id within user")
            if len(user_condition_seeds) != len(set(user_condition_seeds)):
                raise ValueError("duplicate formal ConditionRequest seed within user")
            if len(user_noise_seeds) != len(set(user_noise_seeds)):
                raise ValueError("duplicate formal DDIM noise seed within user")
            if set(user_condition_seeds).intersection(user_noise_seeds):
                raise ValueError("ConditionRequest/DDIM seed domains overlap within user")
            for fake_id in user_fake_ids:
                if fake_id in seen_fake_ids or fake_id in action_fake_ids:
                    raise ValueError("duplicate formal fake_id: %d" % fake_id)
                seen_fake_ids.add(fake_id)
                action_fake_ids.add(fake_id)
            for condition_seed in user_condition_seeds:
                if (
                    condition_seed in seen_condition_seeds
                    or condition_seed in action_condition_seeds
                ):
                    raise ValueError("duplicate formal ConditionRequest seed")
                seen_condition_seeds.add(condition_seed)
                action_condition_seeds.add(condition_seed)
            for noise_seed in user_noise_seeds:
                if noise_seed in seen_noise_seeds or noise_seed in action_noise_seeds:
                    raise ValueError("duplicate formal DDIM noise seed")
                seen_noise_seeds.add(noise_seed)
                action_noise_seeds.add(noise_seed)
            pairs = list(user_result["pairs"])
            all_request_digests.extend(pairs)
            action_request_digests.extend(pairs)
            for name, value in user_result["counts"].items():
                counts[name] += int(value)
            counts["users"] += 1
            for name in minimum:
                minimum[name] = min(minimum[name], user_result["minimum"][name])
                maximum[name] = max(maximum[name], user_result["maximum"][name])
        counts["unique_fake_ids"] = len(action_fake_ids)
        counts["unique_condition_request_seeds"] = len(action_condition_seeds)
        counts["unique_ddim_noise_seeds"] = len(action_noise_seeds)
        expected_batches = 100 * expected_batches_per_user
        if (
            counts["users"] != 100 or counts["requests"] != 20000
            or counts["sampling_batches"] != expected_batches
            or counts["condition_source_code_2"] != 20000
            or counts["unique_fake_ids"] != 20000
            or counts["unique_condition_request_seeds"] != 20000
            or counts["unique_ddim_noise_seeds"] != 20000
        ):
            raise AssertionError("%s preflight did not cover exact 100x200" % action)
        if action == "keystroke" and counts["hard_timeline_projections"] != 20000:
            raise AssertionError("keystroke hard projection did not cover all 20,000 requests")
        if action != "keystroke" and counts["hard_timeline_projections"] != 0:
            raise AssertionError("non-keystroke unexpectedly entered hard timeline gate")
        per_action[action] = {
            "corpus": str(corpus_path.resolve()), "corpus_sha256": corpus.sha256,
            "training_reference_registry_sha256": training_registry.sha256,
            "generation_registry_compatibility_sha256": registry.registry_sha256,
            "train_prior_sha256": prior.digest, "counts": counts,
            "condition_set_sha256": condition_request_set_sha256(action_request_digests),
            "minimum": minimum, "maximum": maximum,
        }
        if action == "keystroke":
            codebook = validate_codebook(corpus)
        total_requests += counts["requests"]
        print("[condition-preflight] %s 20,000/20,000" % action, flush=True)
        del pool, policy, prior, registry, training_registry, corpus

    if total_requests != 100000 or set(per_action) != set(ACTIONS):
        raise AssertionError("formal condition preflight total is not exactly 100,000")
    if len(seen_fake_ids) != 100000:
        raise AssertionError("formal fake IDs are not globally unique")
    if len(seen_condition_seeds) != 100000:
        raise AssertionError("formal ConditionRequest seeds are not globally unique")
    if len(seen_noise_seeds) != 100000:
        raise AssertionError("formal DDIM noise seeds are not globally unique")
    if seen_condition_seeds.intersection(seen_noise_seeds):
        raise AssertionError("ConditionRequest and DDIM noise seed domains overlap")
    if producer_source_identity(ROOT) != source_identity:
        raise RuntimeError("condition-preflight executable source changed while running")
    result = {
        "schema_version": SCHEMA, "status": "passed", "formal_result": False,
        "purpose": "pre-DDIM exhaustive metadata/topology launch gate",
        "producer_source": source_identity,
        "worker_count": int(args.workers),
        "parallelization": "fork_per_user_deterministic_parent_aggregation",
        "reference_seed": int(args.reference_seed),
        "generation_seed": int(args.generation_seed),
        "seed_roles_are_distinct": args.reference_seed != args.generation_seed,
        "split_json": str(args.split_json.resolve()),
        "split_sha256": sha256_file(args.split_json),
        "samples_per_user_action": 200, "n_users": 100,
        "generation_batch_size": int(args.batch_size),
        "sampling_batches_per_user_action": int(expected_batches_per_user),
        "sampling_batches_per_action": int(100 * expected_batches_per_user),
        "sampling_batches_total": int(5 * 100 * expected_batches_per_user),
        "n_actions": 5, "total_requests": total_requests,
        "keystroke_hard_timeline_requests": per_action["keystroke"]["counts"]["hard_timeline_projections"],
        "no_retries": True, "no_skips": True,
        "fixed_references_per_user_action": 5,
        "train_prior_only_fixed_train_users": True,
        "all_condition_source_code_eq_2": True,
        "condition_source_code_2_count": total_requests,
        "all_fake_ids_globally_unique": True,
        "unique_fake_id_count": len(seen_fake_ids),
        "all_condition_request_seeds_globally_unique": True,
        "unique_condition_request_seed_count": len(seen_condition_seeds),
        "all_ddim_noise_seeds_globally_unique": True,
        "unique_ddim_noise_seed_count": len(seen_noise_seeds),
        "condition_and_noise_seed_domains_disjoint": True,
        "training_reference_registry_sha256_by_action": registry_hashes,
        "train_prior_sha256_by_action": prior_hashes,
        "condition_request_digest_schema": CONDITION_REQUEST_DIGEST_SCHEMA,
        "condition_set_digest_schema": CONDITION_SET_DIGEST_SCHEMA,
        "condition_request_digest_fields": list(CONDITION_REQUEST_DIGEST_FIELDS),
        "condition_set_sha256": condition_request_set_sha256(all_request_digests),
        "per_action_condition_set_sha256": {
            action: per_action[action]["condition_set_sha256"] for action in ACTIONS
        },
        "keycode": codebook, "per_action": per_action,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
