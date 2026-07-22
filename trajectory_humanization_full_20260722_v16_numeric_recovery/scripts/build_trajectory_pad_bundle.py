#!/usr/bin/env python3
"""Build formal PAD bundles from audited real NPZ and generated shard archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajectory.features import (
    KEYCODE_TOKEN_MAX,
    KEYCODE_VOCAB_SIZE,
    TRAJECTORY_FEATURE_SCHEMA_VERSION,
)
from detectors.deep_pad import (
    ACTIONS,
    assign_strict_protocol_pools,
    load_fake_user_split,
    save_raw_sequence_bundle,
)
from detectors.trajectory_adapter import load_extracted_trajectory_npz
from generation.pad_export import load_generated_action_tree
from generation.protocol import FixedUserSplit, ReferenceRegistry


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--fake-archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fake-user-split", type=Path, required=True)
    parser.add_argument("--reference-registry-map", type=Path, required=True)
    parser.add_argument("--real-pattern", default="hmog_trajectory_{action}.npz")
    parser.add_argument("--real-hash-seed", type=int, default=20260713)
    return parser.parse_args()


def _reference_overlap(
    registry: ReferenceRegistry,
    action: str,
    assigned_records,
    fixed_split: FixedUserSplit,
) -> dict:
    real_by_event = {}
    for record in assigned_records:
        if record.label != 0:
            continue
        event_id = int(record.event_group_id)
        if event_id in real_by_event:
            raise ValueError("duplicate real event_group_id in %s" % action)
        real_by_event[event_id] = (int(record.user_id), str(record.pool))
    entries = {
        key: ids for key, ids in registry.entries.items() if key[0] == action
    }
    if len(entries) != 100:
        raise ValueError("%s registry must contain exactly one entry for each of 100 users" % action)
    matrix = {
        generator_pool: {detector_pool: 0 for detector_pool in ("train", "val", "test")}
        for generator_pool in ("train", "val", "test")
    }
    seen_ids = set()
    for (entry_action, user_id, generator_pool), ids in sorted(entries.items()):
        if entry_action != action or generator_pool != fixed_split.split_for_user(user_id):
            raise ValueError("reference registry action/user/split mismatch")
        if len(ids) != 5 or len(set(ids)) != 5:
            raise ValueError("reference registry entry is not five unique real events")
        for event_id in ids:
            if event_id in seen_ids:
                raise ValueError("reference event reused across user entries")
            seen_ids.add(event_id)
            if event_id not in real_by_event:
                raise ValueError("reference event is absent from detector real corpus: %d" % event_id)
            event_user, detector_pool = real_by_event[event_id]
            if event_user != user_id:
                raise ValueError("reference event/user mismatch")
            matrix[generator_pool][detector_pool] += 1
    if len(seen_ids) != 500:
        raise ValueError("%s reference overlap audit expected exactly 500 refs" % action)
    return {
        "n_reference_events": 500,
        "by_generator_user_split_and_detector_real_pool": matrix,
        "detector_pool_totals": {
            pool: sum(matrix[source][pool] for source in matrix)
            for pool in ("train", "val", "test")
        },
        "semantics": (
            "The five real events are fixed enrollment/conditioning references, not generated "
            "targets. Detector real event pools are independently complete-event hash-ranked, "
            "without consulting the reference registry. A reference may therefore also be a "
            "detector real train/validation/test sample and participates according to that pool "
            "(including fit or validation selection when assigned there). The full overlap is "
            "reported and no event is reassigned or removed based on the overlap."
        ),
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.output_dir / "bundle_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("refusing to overwrite completed bundle: %s" % args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = load_fake_user_split(args.fake_user_split)
    fixed_split = FixedUserSplit.load(str(args.fake_user_split), require_formal=True)
    registry_paths = json.loads(args.reference_registry_map.read_text(encoding="utf-8"))
    if set(registry_paths) != set(ACTIONS):
        raise ValueError("reference-registry-map must map exactly the five actions")
    registries = {
        action: ReferenceRegistry.load(
            str(registry_paths[action]), fixed_split.source_sha256
        ) for action in ACTIONS
    }
    sources = []
    per_action = {}
    output_files = {}
    split_audit = {
        "schema_version": "trajectory_detector_split_by_action_v1",
        "per_action": {},
    }
    fake_archive_paths = sorted(args.fake_archive_dir.glob(
        "shards/shard_*_of_*/*/user_*.npz"
    ))
    if len(fake_archive_paths) != 500:
        raise ValueError(
            "formal generated archive tree must contain exactly 500 user/action files; found %d"
            % len(fake_archive_paths)
        )
    fake_archive_hashes = {
        str(path.relative_to(args.fake_archive_dir)): sha256(path)
        for path in fake_archive_paths
    }
    fake_archive_hash_path = args.output_dir / "fake_archive_file_hashes.json"
    fake_archive_hash_path.write_text(
        json.dumps(fake_archive_hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    reference_overlap = {}
    global_fake_ids = set()
    for action in ACTIONS:
        real_path = args.real_dir / args.real_pattern.format(action=action)
        real_records, real_features = load_extracted_trajectory_npz(
            real_path, label=0, default_pool="train", sample_prefix="real:"
        )
        fake_records, fake_features = load_generated_action_tree(
            args.fake_archive_dir, action, fixed_split, require_formal=True
        )
        real_users = {record.user_id for record in real_records}
        fake_users = {record.user_id for record in fake_records}
        if len(real_users) != 100:
            raise ValueError(
                "%s formal real source must cover 100 users; found %d" % (action, len(real_users))
            )
        if len(fake_users) != 100:
            raise ValueError(
                "%s formal fake source must cover 100 users; found %d" % (action, len(fake_users))
            )
        if len(fake_records) != 20_000:
            raise ValueError("%s formal fake source must contain 20,000 records" % action)
        action_fake_ids = {int(record.sample_id) for record in fake_records}
        if len(action_fake_ids) != 20_000 or global_fake_ids.intersection(action_fake_ids):
            raise ValueError("%s fake_id uniqueness/cross-action collision" % action)
        global_fake_ids.update(action_fake_ids)
        action_records = real_records + fake_records
        action_features = np.concatenate((real_features, fake_features), axis=0)
        assigned, action_split_audit = assign_strict_protocol_pools(
            action_records, split, real_hash_seed=args.real_hash_seed
        )
        if action_split_audit["user_counts"]["real"] != {
            "train": 100, "val": 100, "test": 100,
        }:
            raise ValueError(
                "%s real train/val/test event pools must each cover all 100 users"
                % action
            )
        if action_split_audit["user_counts"]["fake"]["test"] != 20:
            raise ValueError("%s fake test must contain the fixed 20 users" % action)
        fake_pool_counts = {
            pool: sum(record.label == 1 and record.pool == pool for record in assigned)
            for pool in ("train", "val", "test")
        }
        if fake_pool_counts != {"train": 14000, "val": 2000, "test": 4000}:
            raise ValueError("%s fake pool counts are not 14k/2k/4k" % action)
        if len(action_features) != len(assigned):
            raise RuntimeError("feature/order mismatch for %s" % action)
        path = args.output_dir / (action + ".npz")
        save_raw_sequence_bundle(path, assigned, action_features)
        output_files[action] = {"path": str(path), "sha256": sha256(path), "n": len(assigned)}
        split_audit["per_action"][action] = action_split_audit
        split_audit["per_action"][action]["fake_sample_counts"] = fake_pool_counts
        reference_overlap[action] = _reference_overlap(
            registries[action], action, assigned, fixed_split
        )
        sources.extend([
            {"action": action, "label": "real", "path": str(real_path), "sha256": sha256(real_path), "n": len(real_records)},
            {"action": action, "label": "fake", "path": str(args.fake_archive_dir), "n": len(fake_records)},
        ])
        per_action[action] = {"real": len(real_records), "fake": len(fake_records), "feature_dim": int(action_features.shape[1])}
        if action == "keystroke":
            keycode_audit = {}
            for label, name in ((0, "real"), (1, "fake")):
                values = np.concatenate([
                    record.keycode[record.contact_mask]
                    for record in assigned if record.label == label
                ]).astype(np.int64)
                if np.any(values < 0):
                    raise ValueError("keystroke contact contains non-canonical negative keycode")
                keycode_audit[name] = {
                    "contact_tokens": int(values.size),
                    "min": int(values.min()),
                    "max": int(values.max()),
                    "count_8230": int(np.sum(values == 8230)),
                    "outside_shared_vocabulary_count": int(
                        np.sum(values > KEYCODE_TOKEN_MAX)
                    ),
                }
            keycode_audit["deep_embedding_policy"] = (
                "negative gap=-1 -> index0; canonical nonnegative 0..%d -> code+1; "
                "rare 8230 keeps distinct index8231 identically for real/fake; "
                "out-of-range fails closed" % KEYCODE_TOKEN_MAX
            )
            keycode_audit["shared_vocabulary_size"] = KEYCODE_VOCAB_SIZE
            keycode_audit["feature_policy"] = (
                "ASCII 65-90/97-122 -> letter character; all other canonical codes, "
                "including 8230, -> keycode_<code>"
            )
            per_action[action]["keycode_audit"] = keycode_audit

    if len(global_fake_ids) != 100_000:
        raise ValueError("formal generated archive must contain 100,000 globally unique fake IDs")

    (args.output_dir / "split_audit.json").write_text(
        json.dumps(split_audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema_version": "trajectory_pad_bundle_manifest_v2",
        "status": "complete",
        "feature_schema_version": TRAJECTORY_FEATURE_SCHEMA_VERSION,
        "sources": sources,
        "fake_user_split": str(args.fake_user_split),
        "fake_user_split_sha256": sha256(args.fake_user_split),
        "fake_archive_dir": str(args.fake_archive_dir.resolve()),
        "fake_archive_file_count": len(fake_archive_paths),
        "fake_archive_file_hashes": str(fake_archive_hash_path),
        "fake_archive_file_hashes_sha256": sha256(fake_archive_hash_path),
        "reference_registry_map": str(args.reference_registry_map.resolve()),
        "reference_registry_map_sha256": sha256(args.reference_registry_map),
        "reference_registry_sha256_by_action": {
            action: registries[action].registry_sha256 for action in ACTIONS
        },
        "reference_overlap_with_detector_real_event_pools": reference_overlap,
        "real_hash_seed": args.real_hash_seed,
        "per_action": per_action,
        "outputs": output_files,
        "split_audit": str(args.output_dir / "split_audit.json"),
    }
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
