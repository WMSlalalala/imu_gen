#!/usr/bin/env python3
"""Full 500-unit / 100,000-fake completion and leakage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_determinism import (
    EXPECTED_RUNTIME_DETERMINISM,
    STRICT_RUNTIME_DETERMINISM_SHA256,
    runtime_determinism_matches,
)
from generation.audit import audit_generated_unit
from generation.corpus import load_action_corpus
from generation.pipeline import unit_output_path
from generation.protocol import (
    ACTIONS, CONDITION_REQUEST_DIGEST_SCHEMA, CONDITION_SET_DIGEST_SCHEMA,
    FORMAL_GENERATION_BASE_SEED, FORMAL_TOTAL, FixedUserSplit,
    ReferenceRegistry, TrainGlobalPrior, build_generation_units,
    condition_request_set_sha256,
)


DEFAULT_SPLIT = "/home/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--split-json", default=DEFAULT_SPLIT)
    parser.add_argument("--reference-registry")
    parser.add_argument("--reference-registry-map", help="JSON action->training reference_registry.json")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--condition-preflight", required=True,
        help="passed exhaustive 100k condition preflight using the shared digest schema",
    )
    return parser.parse_args()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d.%s" % (os.getpid(), uuid.uuid4().hex))
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        directory_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    split = FixedUserSplit.load(args.split_json, require_formal=True)
    if bool(args.reference_registry) == bool(args.reference_registry_map):
        raise ValueError("provide exactly one merged registry or per-action registry map")
    if args.reference_registry_map:
        registry_paths = json.loads(Path(args.reference_registry_map).read_text(encoding="utf-8"))
        if set(registry_paths) != set(ACTIONS):
            raise ValueError("reference registry map must contain five actions")
        registries = {action: ReferenceRegistry.load(registry_paths[action], split.source_sha256) for action in ACTIONS}
    else:
        registry = ReferenceRegistry.load(args.reference_registry, split.source_sha256)
        registries = {action: registry for action in ACTIONS}
    units = build_generation_units(split, num_shards=args.num_shards, shard_id=None, require_formal=True)
    expected_manifest_paths = {
        output / ("generation_manifest_shard_%03d_of_%03d.json" % (shard_id, args.num_shards))
        for shard_id in range(args.num_shards)
    }
    actual_manifest_paths = set(output.glob("generation_manifest_shard_*_of_*.json"))
    if actual_manifest_paths != expected_manifest_paths:
        raise ValueError("formal generation shard manifest set is incomplete or contains extras")
    manifest_hashes = {}
    manifest_checkpoint_by_action = defaultdict(set)
    manifest_unit_paths = set()
    for manifest_path in sorted(expected_manifest_paths):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_id = int(manifest["shard_id"])
        shard_units = [unit for unit in units if unit.shard_id == shard_id]
        if (
            manifest.get("schema_version") != "five_shot_generation_shard_manifest_v4"
            or manifest.get("formal") is not True
            or not runtime_determinism_matches(manifest.get("runtime_determinism"))
            or manifest.get("runtime_determinism_sha256")
            != STRICT_RUNTIME_DETERMINISM_SHA256
            or int(manifest.get("generation_base_seed", -1)) != FORMAL_GENERATION_BASE_SEED
            or int(manifest.get("generation_batch_size", -1)) != 32
            or manifest.get("selector_used") is not False
            or int(manifest.get("ddim_steps", -1)) != 50
            or float(manifest.get("eta", float("nan"))) != 0.0
            or int(manifest.get("fixed_refs_per_user_action", -1)) != 5
            or manifest.get("condition_request_seed_derivation")
            != "stable_seed(base_seed,action,user_id,sample_index)"
            or manifest.get("ddim_noise_seed_derivation")
            != "stable_seed(condition_request_seed_xor_0xDD1A50,action,user_id,sample_index)"
            or int(manifest.get("num_shards", -1)) != args.num_shards
            or not 0 <= shard_id < args.num_shards
            or int(manifest.get("planned_units", -1)) != len(shard_units)
            or int(manifest.get("completed_units", -1)) != len(shard_units)
            or int(manifest.get("planned_fake", -1)) != sum(unit.samples for unit in shard_units)
            or int(manifest.get("completed_fake", -1)) != sum(unit.samples for unit in shard_units)
            or int(manifest.get("condition_request_seed_recomputed_count", -1))
            != sum(unit.samples for unit in shard_units)
            or int(manifest.get("ddim_noise_seed_recomputed_count", -1))
            != sum(unit.samples for unit in shard_units)
            or int(manifest.get("unique_condition_request_seed_count", -1))
            != sum(unit.samples for unit in shard_units)
            or int(manifest.get("unique_ddim_noise_seed_count", -1))
            != sum(unit.samples for unit in shard_units)
            or manifest.get("condition_and_noise_seed_domains_disjoint") is not True
            or int(manifest.get("condition_request_replay_count", -1)) != sum(unit.samples for unit in shard_units)
            or manifest.get("fixed_split_sha256") != split.source_sha256
        ):
            raise ValueError("formal generation shard manifest protocol mismatch: %s" % manifest_path)
        expected_registry_hashes = {
            action: registries[action].registry_sha256 for action in ACTIONS
        }
        if manifest.get("reference_registry_sha256_by_action") != expected_registry_hashes:
            raise ValueError("formal generation shard manifest registry mismatch: %s" % manifest_path)
        results = manifest.get("results")
        if not isinstance(results, list) or len(results) != len(shard_units):
            raise ValueError("formal generation shard manifest results mismatch: %s" % manifest_path)
        expected_paths_for_shard = {
            str(unit_output_path(str(output), unit).resolve()) for unit in shard_units
        }
        observed_paths_for_shard = {str(result.get("path")) for result in results}
        if observed_paths_for_shard != expected_paths_for_shard:
            raise ValueError("formal generation shard manifest unit path mismatch: %s" % manifest_path)
        for result in results:
            if (
                result.get("passed") is not True
                or not runtime_determinism_matches(result.get("runtime_determinism"))
                or result.get("runtime_determinism_sha256")
                != STRICT_RUNTIME_DETERMINISM_SHA256
                or int(result.get("generation_base_seed", -1)) != FORMAL_GENERATION_BASE_SEED
                or int(result.get("generation_batch_size", -1)) != 32
                or int(result.get("ddim_steps", -1)) != 50
                or float(result.get("ddim_eta", float("nan"))) != 0.0
                or int(result.get("condition_request_seed_recomputed_count", -1)) != 200
                or int(result.get("ddim_noise_seed_recomputed_count", -1)) != 200
                or int(result.get("unique_condition_request_seed_count", -1)) != 200
                or int(result.get("unique_ddim_noise_seed_count", -1)) != 200
                or result.get("condition_and_noise_seed_domains_disjoint") is not True
                or int(result.get("condition_request_replay_count", -1)) != 200
            ):
                raise ValueError("formal generation shard manifest contains an unaudited unit")
            manifest_checkpoint_by_action[str(result["action"])].add(str(result["checkpoint_sha256"]))
        observed_result_actions = {str(result["action"]) for result in results}
        current_checkpoint_sets = {
            action: {
                str(result["checkpoint_sha256"])
                for result in results if str(result["action"]) == action
            }
            for action in observed_result_actions
        }
        if any(len(values) != 1 for values in current_checkpoint_sets.values()) or manifest.get(
            "checkpoint_sha256_by_action"
        ) != {
            action: next(iter(current_checkpoint_sets[action]))
            for action in observed_result_actions
        }:
            raise ValueError("formal generation shard checkpoint manifest mismatch")
        shard_digest_pairs = []
        shard_digest_pairs_by_action = defaultdict(list)
        for unit in shard_units:
            with np.load(str(unit_output_path(str(output), unit)), allow_pickle=False) as archive:
                pairs = list(zip(
                    np.asarray(archive["fake_id"], np.int64).tolist(),
                    [bytes(row) for row in np.asarray(archive["condition_request_sha256"], np.uint8)],
                ))
            shard_digest_pairs.extend(pairs)
            shard_digest_pairs_by_action[unit.action].extend(pairs)
        expected_per_action_digest = {
            action: condition_request_set_sha256(pairs)
            for action, pairs in shard_digest_pairs_by_action.items()
        }
        if (
            manifest.get("condition_request_digest_schema") != CONDITION_REQUEST_DIGEST_SCHEMA
            or manifest.get("condition_set_digest_schema") != CONDITION_SET_DIGEST_SCHEMA
            or manifest.get("condition_set_sha256")
            != condition_request_set_sha256(shard_digest_pairs)
            or manifest.get("per_action_condition_set_sha256") != expected_per_action_digest
        ):
            raise ValueError("formal generation shard manifest condition digest mismatch")
        manifest_unit_paths.update(observed_paths_for_shard)
        manifest_hashes[str(shard_id)] = sha256_file(manifest_path)
    expected_paths = {unit_output_path(str(output), unit).resolve() for unit in units}
    actual_paths = {path.resolve() for path in output.glob("shards/shard_*_of_*/*/user_*.npz")}
    if actual_paths != expected_paths:
        missing = sorted(str(x) for x in expected_paths - actual_paths)
        extra = sorted(str(x) for x in actual_paths - expected_paths)
        raise ValueError("unit file set mismatch: missing=%d extra=%d" % (len(missing), len(extra)))

    reports = []
    all_fake_ids = set()
    all_condition_seeds = set()
    all_noise_seeds = set()
    per_action_split = defaultdict(int)
    checkpoint_by_action = defaultdict(set)
    request_digest_pairs = []
    request_digest_pairs_by_action = defaultdict(list)
    for action in ACTIONS:
        corpus_path = Path(args.corpus_dir) / ("hmog_trajectory_%s.npz" % action)
        pool = load_action_corpus(str(corpus_path), action, split, user_ids=split.all_users, strict=True)
        prior = TrainGlobalPrior.fit(action, pool, split.train_users)
        for unit in (x for x in units if x.action == action):
            path = unit_output_path(str(output), unit)
            report = audit_generated_unit(
                str(path), pool, split, registries[action], prior, expected_count=200,
                expected_base_seed=FORMAL_GENERATION_BASE_SEED,
                expected_generation_batch_size=32,
                expected_ddim_steps=50, max_aggregate_clip_rate=0.05,
                max_event_clip_rate=0.25, max_alpha_bar_final=0.001,
            )
            reports.append(report)
            per_action_split[(action, unit.split)] += 200
            with np.load(str(path), allow_pickle=False) as archive:
                ids = set(int(x) for x in archive["fake_id"])
                if all_fake_ids.intersection(ids):
                    raise ValueError("fake id collision across units")
                all_fake_ids.update(ids)
                condition_seeds = set(
                    int(value) for value in np.asarray(archive["seed"], np.int64).tolist()
                )
                noise_seeds = set(
                    int(value)
                    for value in np.asarray(archive["ddim_noise_seed"], np.int64).tolist()
                )
                if (
                    len(condition_seeds) != 200
                    or len(noise_seeds) != 200
                    or all_condition_seeds.intersection(condition_seeds)
                    or all_noise_seeds.intersection(noise_seeds)
                ):
                    raise ValueError("per-sample seed collision across generation units")
                all_condition_seeds.update(condition_seeds)
                all_noise_seeds.update(noise_seeds)
                checkpoint_by_action[action].add(bytes(archive["checkpoint_sha256"].tolist()).hex())
                pairs = list(zip(
                    np.asarray(archive["fake_id"], np.int64).tolist(),
                    [bytes(row) for row in np.asarray(archive["condition_request_sha256"], np.uint8)],
                ))
                request_digest_pairs.extend(pairs)
                request_digest_pairs_by_action[action].extend(pairs)
        del pool, prior

    expected_counts = {"train": 14000, "val": 2000, "test": 4000}
    for action in ACTIONS:
        for split_name, count in expected_counts.items():
            if per_action_split[(action, split_name)] != count:
                raise ValueError("wrong %s/%s count" % (action, split_name))
        if len(checkpoint_by_action[action]) != 1 or next(iter(checkpoint_by_action[action])) == "0" * 64:
            raise ValueError("formal action units must share one nonzero best-EMA checkpoint digest")
    if len(all_fake_ids) != FORMAL_TOTAL or len(reports) != 500:
        raise ValueError("formal total must be 500 units / 100,000 unique fake")
    if (
        len(all_condition_seeds) != FORMAL_TOTAL
        or len(all_noise_seeds) != FORMAL_TOTAL
        or all_condition_seeds.intersection(all_noise_seeds)
    ):
        raise ValueError("formal ConditionRequest/DDIM seed domains are not unique/disjoint")
    if manifest_unit_paths != {str(path) for path in expected_paths}:
        raise ValueError("shard manifests do not cover the exact 500 archive units")
    if {
        action: checkpoint_by_action[action] for action in ACTIONS
    } != {
        action: manifest_checkpoint_by_action[action] for action in ACTIONS
    }:
        raise ValueError("shard manifest checkpoint provenance contradicts unit archives")
    condition_set_digest = condition_request_set_sha256(request_digest_pairs)
    per_action_condition_set_digest = {
        action: condition_request_set_sha256(request_digest_pairs_by_action[action])
        for action in ACTIONS
    }
    preflight_path = Path(args.condition_preflight)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "passed"
        or int(preflight.get("generation_seed", -1)) != FORMAL_GENERATION_BASE_SEED
        or int(preflight.get("total_requests", -1)) != FORMAL_TOTAL
        or int(preflight.get("unique_condition_request_seed_count", -1)) != FORMAL_TOTAL
        or int(preflight.get("unique_ddim_noise_seed_count", -1)) != FORMAL_TOTAL
        or preflight.get("condition_and_noise_seed_domains_disjoint") is not True
        or preflight.get("condition_request_digest_schema") != CONDITION_REQUEST_DIGEST_SCHEMA
        or preflight.get("condition_set_digest_schema") != CONDITION_SET_DIGEST_SCHEMA
        or preflight.get("condition_set_sha256") != condition_set_digest
        or preflight.get("per_action_condition_set_sha256") != per_action_condition_set_digest
    ):
        raise ValueError("generated 100k ConditionRequest set differs from exhaustive preflight")

    # Publish a byte-level receipt for all 500 neural archives.  The final
    # supervisor and detector bundle both bind this same map, so a structurally
    # valid NPZ cannot be changed after the expensive generation audit and then
    # silently flow into detection.
    archive_hashes = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(expected_paths)
    }
    archive_hash_path = output / "generation_archive_file_hashes.json"
    atomic_text(
        archive_hash_path,
        json.dumps(archive_hashes, indent=2, sort_keys=True) + "\n",
    )

    summary = {
        "schema_version": "five_shot_generation_formal_audit_v4",
        "passed": True,
        "formal": True,
        "runtime_determinism": dict(EXPECTED_RUNTIME_DETERMINISM),
        "runtime_determinism_sha256": STRICT_RUNTIME_DETERMINISM_SHA256,
        "generation_base_seed": FORMAL_GENERATION_BASE_SEED,
        "generation_batch_size": 32,
        "condition_request_seed_derivation":
        "stable_seed(base_seed,action,user_id,sample_index)",
        "ddim_noise_seed_derivation":
        "stable_seed(condition_request_seed_xor_0xDD1A50,action,user_id,sample_index)",
        "condition_request_seed_recomputed_count": int(sum(
            x["condition_request_seed_recomputed_count"] for x in reports
        )),
        "ddim_noise_seed_recomputed_count": int(sum(
            x["ddim_noise_seed_recomputed_count"] for x in reports
        )),
        "unique_condition_request_seed_count": len(all_condition_seeds),
        "unique_ddim_noise_seed_count": len(all_noise_seeds),
        "condition_and_noise_seed_domains_disjoint": True,
        "condition_request_replay_count": int(sum(x["condition_request_replay_count"] for x in reports)),
        "condition_request_digest_schema": CONDITION_REQUEST_DIGEST_SCHEMA,
        "condition_set_digest_schema": CONDITION_SET_DIGEST_SCHEMA,
        "condition_set_sha256": condition_set_digest,
        "per_action_condition_set_sha256": per_action_condition_set_digest,
        "condition_preflight": str(preflight_path.resolve()),
        "condition_preflight_sha256": sha256_file(preflight_path),
        "generation_manifest_sha256_by_shard": manifest_hashes,
        "generation_archive_file_count": len(archive_hashes),
        "generation_archive_file_hashes": str(archive_hash_path.resolve()),
        "generation_archive_file_hashes_sha256": sha256_file(archive_hash_path),
        "selector_used": False,
        "n_units": len(reports),
        "n_fake": len(all_fake_ids),
        "ddim_steps": 50,
        "eta": 0.0,
        "training_diffusion_steps": 1000,
        "fixed_refs_per_user_action": 5,
        "split_counts_per_action": expected_counts,
        "checkpoint_sha256_by_action": {key: next(iter(value)) for key, value in checkpoint_by_action.items()},
        "aggregate_clip_rate_by_action": {
            action: float(np.mean([x["aggregate_clipped_point_rate"] for x in reports if x["action"] == action]))
            for action in ACTIONS
        },
        "exact_replay_total": int(sum(x["exact_replay_count"] for x in reports)),
        "exact_metadata_copy_total": int(sum(x["exact_metadata_copy_count"] for x in reports)),
        "complete_key_sequence_copy_total": int(sum(x["complete_key_sequence_copy_count"] for x in reports)),
        "key_endpoint_orientation_fallback_total": int(sum(x["key_endpoint_orientation_fallback_count"] for x in reports)),
        "key_endpoint_zero_token_total": int(sum(x["key_endpoint_zero_token_count"] for x in reports)),
        "unit_reports": reports,
    }
    atomic_text(output / "formal_generation_audit.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows = [
        "# 五动作正式生成完整性审计", "", "- 结论：PASS", "- fake 总数：100,000",
        "- 单元：500（100 users × 5 actions）", "- 每 user/action：200",
        "- 固定 refs：5，同一 user/action 的 200 条完全相同", "- 采样：best EMA checkpoint，1000-step 训练表上的 50-step DDIM，eta=0",
        "- generation base seed：20260713；batch：32；100,000 条 ConditionRequest seed 与 100,000 条 DDIM noise seed 已逐条重算",
        "- condition set SHA-256：`%s`（与 exhaustive preflight 完全一致）" % condition_set_digest,
        "- selector：未使用", "- exact replay：0", "- complete metadata copy：0", "",
        "- complete key-sequence copy：0", "- 时间分辨率：HMOG integer ms lattice", "",
        "| action | train | val | test | mean clipped point rate |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for action in ACTIONS:
        rows.append("| %s | 14000 | 2000 | 4000 | %.6f |" % (action, summary["aggregate_clip_rate_by_action"][action]))
    atomic_text(output / "formal_generation_audit.md", "\n".join(rows) + "\n")
    print(json.dumps({key: summary[key] for key in ("passed", "n_units", "n_fake", "selector_used")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
