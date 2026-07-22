#!/usr/bin/env python3
"""Formal 100k or explicitly marked smoke five-shot DDIM generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_determinism import (  # noqa: E402
    require_strict_runtime_determinism,
    runtime_determinism_sha256,
    seed_everything,
)

import numpy as np
import torch

from generation.corpus import load_action_corpus
from generation.pipeline import generate_unit
from generation.protocol import (
    ACTIONS, CONDITION_REQUEST_DIGEST_SCHEMA, CONDITION_SET_DIGEST_SCHEMA,
    FORMAL_GENERATION_BASE_SEED, FORMAL_SAMPLES_PER_USER_ACTION, FixedUserSplit,
    ReferenceRegistry, TrainGlobalPrior, build_generation_units,
    choose_reference_sets, condition_request_set_sha256,
)
from generation.sampler import load_model_checkpoint
from trajectory.model import TrajectoryDiffusion


DEFAULT_SPLIT = "/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--split-json", default=DEFAULT_SPLIT)
    parser.add_argument("--reference-registry")
    parser.add_argument("--reference-registry-map", help="JSON action->training reference_registry.json")
    parser.add_argument("--checkpoint-map", help="JSON object mapping each action to a trained checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=FORMAL_GENERATION_BASE_SEED)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--confirm-formal-100k", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-action", choices=ACTIONS, default="tap")
    parser.add_argument("--smoke-user", type=int, default=0)
    parser.add_argument("--smoke-samples", type=int, default=2)
    return parser.parse_args()


def assert_safe_paths(corpus_dir: Path, output_dir: Path) -> None:
    corpus, output = corpus_dir.resolve(), output_dir.resolve()
    if corpus == output or str(output).startswith(str(corpus) + "/"):
        raise ValueError("output cannot equal or live inside the formal input corpus")


def main() -> int:
    args = parse_args()
    if args.smoke == args.confirm_formal_100k:
        raise ValueError("choose exactly one of --smoke or --confirm-formal-100k")
    if not args.smoke and args.seed != FORMAL_GENERATION_BASE_SEED:
        raise ValueError(
            "formal generation fixes --seed=%d; reference selection uses the independent training seed 42"
            % FORMAL_GENERATION_BASE_SEED
        )
    if not args.smoke and args.batch_size != 32:
        raise ValueError("formal generation fixes --batch-size=32")
    seed_everything(args.seed)
    runtime_determinism = require_strict_runtime_determinism()
    runtime_digest = runtime_determinism_sha256(runtime_determinism)
    corpus_dir, output_dir = Path(args.corpus_dir), Path(args.output_dir)
    assert_safe_paths(corpus_dir, output_dir)
    split = FixedUserSplit.load(args.split_json, require_formal=True)
    device = torch.device(args.device)
    if args.smoke:
        actions = (args.smoke_action,)
        units = [x for x in build_generation_units(
            split, samples_per_user_action=args.smoke_samples, actions=actions,
            num_shards=1, shard_id=None, require_formal=False,
        ) if x.user_id == args.smoke_user]
    else:
        actions = ACTIONS
        units = build_generation_units(
            split, samples_per_user_action=FORMAL_SAMPLES_PER_USER_ACTION, actions=actions,
            num_shards=args.num_shards, shard_id=args.shard_id, require_formal=True,
        )
        if (not args.reference_registry and not args.reference_registry_map) or not args.checkpoint_map:
            raise ValueError("formal generation requires training ReferenceRegistry map and checkpoint map")
        if args.reference_registry and args.reference_registry_map:
            raise ValueError("use either merged --reference-registry or per-action --reference-registry-map")

    checkpoint_map = {}
    action_pools = {}
    if args.checkpoint_map:
        checkpoint_map = json.loads(Path(args.checkpoint_map).read_text(encoding="utf-8"))
    if args.reference_registry_map:
        registry_paths = json.loads(Path(args.reference_registry_map).read_text(encoding="utf-8"))
        if set(registry_paths) != set(ACTIONS):
            raise ValueError("reference registry map must contain exactly the five actions")
        registries = {
            action: ReferenceRegistry.load(registry_paths[action], split.source_sha256)
            for action in ACTIONS
        }
    elif args.reference_registry:
        registry = ReferenceRegistry.load(args.reference_registry, split.source_sha256)
        registries = {action: registry for action in actions}
    else:
        # Smoke derives a registry only to exercise the interface.  Formal mode
        # is forbidden from taking this path and must consume training output.
        entries = {}
        for action in actions:
            requested_users = set(split.train_users) | {x.user_id for x in units if x.action == action}
            pool = load_action_corpus(
                str(corpus_dir / ("hmog_trajectory_%s.npz" % action)), action, split,
                user_ids=requested_users, strict=True,
            )
            action_pools[action] = pool
            for unit in (x for x in units if x.action == action):
                refs = choose_reference_sets(
                    pool, unit.action, unit.user_id, unit.split,
                    n_samples=1, base_seed=args.seed,
                )[0]
                entries[(unit.action, unit.user_id, unit.split)] = tuple(int(x.sample_id) for x in refs)
        registry = ReferenceRegistry.build(entries, split.source_sha256, source_path="smoke-derived")
        registries = {action: registry for action in actions}
        registry_path = output_dir / "smoke_reference_registry.json"
        if not registry_path.exists():
            registry.write(str(registry_path))

    if not args.smoke:
        for action in ACTIONS:
            expected_action_keys = {
                (action, user_id, split.split_for_user(user_id)) for user_id in split.all_users
            }
            action_keys = {key for key in registries[action].entries if key[0] == action}
            if action_keys != expected_action_keys:
                raise ValueError("%s training ReferenceRegistry must cover exactly 100 users" % action)
        if set(checkpoint_map) != set(ACTIONS):
            raise ValueError("formal checkpoint map must contain exactly the five actions")

    results = []
    started = time.time()
    # Process one action at a time.  Canonical full-action records can be
    # large; retaining all five corpora and models simultaneously is needless.
    for action in actions:
        if args.smoke and action in action_pools:
            pool = action_pools[action]
        else:
            requested_users = set(split.train_users) | {x.user_id for x in units if x.action == action}
            pool = load_action_corpus(
                str(corpus_dir / ("hmog_trajectory_%s.npz" % action)), action, split,
                user_ids=requested_users, strict=True,
            )
        prior = TrainGlobalPrior.fit(action, pool, split.train_users)
        if args.smoke and action not in checkpoint_map:
            torch.manual_seed(args.seed)
            model = TrajectoryDiffusion(
                action, diffusion_steps=50, base_channels=8,
                cond_dim=16, time_dim=8, n_blocks=1, dropout=0.0,
            ).to(device).eval()
            checkpoint_digest = "0" * 64
        else:
            if action not in checkpoint_map:
                raise ValueError("checkpoint map lacks %s" % action)
            model, checkpoint_digest = load_model_checkpoint(
                checkpoint_map[action], action, device,
                expected_registry_sha256=registries[action].registry_sha256,
                expected_split_sha256=split.source_sha256,
            )
        for unit in (x for x in units if x.action == action):
            results.append(generate_unit(
                unit, pool, split, registries[action], prior, model,
                checkpoint_digest, str(output_dir), args.seed, args.batch_size, device,
                resume=True,
                max_aggregate_clip_rate=1.0 if args.smoke else 0.05,
                max_event_clip_rate=1.0 if args.smoke else 0.25,
            ))
        del pool, prior, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    request_digest_pairs = []
    request_digest_pairs_by_action = {action: [] for action in actions}
    condition_seeds = set()
    noise_seeds = set()
    for result in results:
        with np.load(str(result["path"]), allow_pickle=False) as archive:
            pairs = list(zip(
                np.asarray(archive["fake_id"], np.int64).tolist(),
                [bytes(row) for row in np.asarray(archive["condition_request_sha256"], np.uint8)],
            ))
            unit_condition_seeds = set(
                int(value) for value in np.asarray(archive["seed"], np.int64).tolist()
            )
            unit_noise_seeds = set(
                int(value)
                for value in np.asarray(archive["ddim_noise_seed"], np.int64).tolist()
            )
            if (
                len(unit_condition_seeds) != int(archive["seed"].size)
                or len(unit_noise_seeds) != int(archive["ddim_noise_seed"].size)
                or condition_seeds.intersection(unit_condition_seeds)
                or noise_seeds.intersection(unit_noise_seeds)
            ):
                raise ValueError("generation shard contains a per-sample seed collision")
            condition_seeds.update(unit_condition_seeds)
            noise_seeds.update(unit_noise_seeds)
        request_digest_pairs.extend(pairs)
        request_digest_pairs_by_action[str(result["action"])].extend(pairs)
    expected_seed_count = int(sum(int(result["n_fake"]) for result in results))
    if (
        len(condition_seeds) != expected_seed_count
        or len(noise_seeds) != expected_seed_count
        or condition_seeds.intersection(noise_seeds)
    ):
        raise ValueError("generation shard seed domains are not unique and disjoint")
    checkpoint_sets_by_action = {
        action: {
            str(result["checkpoint_sha256"])
            for result in results if result["action"] == action
        }
        for action in actions
    }
    if any(len(values) != 1 for values in checkpoint_sets_by_action.values()):
        raise ValueError("one generation shard mixed checkpoints within an action")
    manifest = {
        "schema_version": "five_shot_generation_shard_manifest_v4",
        "formal": not args.smoke,
        "runtime_determinism": runtime_determinism,
        "runtime_determinism_sha256": runtime_digest,
        "generation_base_seed": int(args.seed),
        "generation_batch_size": int(args.batch_size),
        "condition_request_seed_derivation":
        "stable_seed(base_seed,action,user_id,sample_index)",
        "ddim_noise_seed_derivation":
        "stable_seed(condition_request_seed_xor_0xDD1A50,action,user_id,sample_index)",
        "condition_request_digest_schema": CONDITION_REQUEST_DIGEST_SCHEMA,
        "condition_set_digest_schema": CONDITION_SET_DIGEST_SCHEMA,
        "condition_set_sha256": condition_request_set_sha256(request_digest_pairs),
        "per_action_condition_set_sha256": {
            action: condition_request_set_sha256(request_digest_pairs_by_action[action])
            for action in actions if request_digest_pairs_by_action[action]
        },
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "selector_used": False,
        "ddim_steps": 50,
        "eta": 0.0,
        "fixed_refs_per_user_action": 5,
        "planned_units": len(units),
        "planned_fake": int(sum(x.samples for x in units)),
        "completed_units": len(results),
        "completed_fake": int(sum(int(x["n_fake"]) for x in results)),
        "elapsed_seconds": time.time() - started,
        "fixed_split_sha256": split.source_sha256,
        "reference_registry_sha256_by_action": {
            action: registries[action].registry_sha256 for action in actions
        },
        "checkpoint_sha256_by_action": {
            action: next(iter(checkpoint_sets_by_action[action]))
            for action in actions
        },
        "condition_request_seed_recomputed_count": int(sum(
            int(result["condition_request_seed_recomputed_count"])
            for result in results
        )),
        "ddim_noise_seed_recomputed_count": int(sum(
            int(result["ddim_noise_seed_recomputed_count"])
            for result in results
        )),
        "unique_condition_request_seed_count": len(condition_seeds),
        "unique_ddim_noise_seed_count": len(noise_seeds),
        "condition_and_noise_seed_domains_disjoint": not bool(
            condition_seeds.intersection(noise_seeds)
        ),
        "condition_request_replay_count": int(sum(
            int(result["condition_request_replay_count"]) for result in results
        )),
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / (
        "generation_manifest.json" if args.smoke
        else "generation_manifest_shard_%03d_of_%03d.json" % (args.shard_id, args.num_shards)
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(manifest_path))
    directory_fd = os.open(str(manifest_path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps({key: manifest[key] for key in ("formal", "planned_fake", "completed_fake", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
