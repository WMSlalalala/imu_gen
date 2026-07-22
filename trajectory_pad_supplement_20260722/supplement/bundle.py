"""Build and audit fully user-disjoint trajectory PAD sensitivity bundles.

The frozen primary PAD protocol intentionally remains unchanged.  This module
reads its numeric bundle and creates a separate dataset in which *both* real
and fake samples use the same frozen 70/10/20 user split.  An optional second
variant removes every real event used by the generator's fixed-five reference
registry, allowing the enrollment-reference overlap sensitivity to be measured.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np


DEFAULT_V16_ROOT = Path(
    "/home/mwang49/real-human/imu_gen/final/"
    "trajectory_humanization_full_20260722_v16_numeric_recovery"
)
V16_ROOT = Path(os.environ.get("TRAJECTORY_V16_ROOT", str(DEFAULT_V16_ROOT))).resolve()
if str(V16_ROOT) not in sys.path:
    sys.path.insert(0, str(V16_ROOT))

from detectors.deep_pad import (  # noqa: E402
    ACTIONS,
    RawTrajectoryRecord,
    load_fake_user_split,
    load_raw_sequence_bundle,
    save_raw_sequence_bundle,
)


POOLS = ("train", "val", "test")
SCHEMA = "trajectory_pad_user_disjoint_supplement_v1"
AUDIT_SCHEMA = "trajectory_pad_user_disjoint_supplement_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _owner_map(split: Mapping[str, Sequence[int]]) -> Dict[int, str]:
    owner: Dict[int, str] = {}
    expected_sizes = {"train": 70, "val": 10, "test": 20}
    for pool in POOLS:
        users = tuple(int(value) for value in split[pool])
        if len(users) != expected_sizes[pool] or len(set(users)) != len(users):
            raise ValueError("fixed user split must contain unique 70/10/20 users")
        for user_id in users:
            if user_id in owner:
                raise ValueError("user appears in more than one pool: %d" % user_id)
            owner[user_id] = pool
    if set(owner) != set(range(100)):
        raise ValueError("fixed user split must cover exactly users 0..99")
    return owner


def load_reference_ids(
    registry_path: Path,
    *,
    action: str,
    owner: Mapping[int, str],
) -> Tuple[Set[str], Dict[str, int], str]:
    """Load exactly five unique reference events for each of 100 users."""

    path = Path(registry_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("action") != action:
        raise ValueError("reference registry action mismatch for %s" % action)
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 100:
        raise ValueError("%s registry must contain exactly 100 user entries" % action)
    seen_users: Set[int] = set()
    reference_ids: Set[str] = set()
    counts = {pool: 0 for pool in POOLS}
    for entry in entries:
        user_id = int(entry["user_id"])
        if user_id in seen_users or user_id not in owner:
            raise ValueError("invalid/duplicate registry user for %s" % action)
        seen_users.add(user_id)
        if entry.get("action") != action or entry.get("split") != owner[user_id]:
            raise ValueError("registry user/action/split mismatch for %s" % action)
        ids = tuple(str(int(value)) for value in entry["reference_event_ids"])
        if len(ids) != 5 or len(set(ids)) != 5:
            raise ValueError("registry entry is not exactly five unique references")
        if reference_ids.intersection(ids):
            raise ValueError("reference event reused across users")
        reference_ids.update(ids)
        counts[owner[user_id]] += len(ids)
    if seen_users != set(range(100)) or len(reference_ids) != 500:
        raise ValueError("%s registry does not cover 100 users / 500 refs" % action)
    if counts != {"train": 350, "val": 50, "test": 100}:
        raise ValueError("reference pool counts do not follow fixed 70/10/20 users")
    declared_hash = str(payload.get("registry_sha256", ""))
    if not declared_hash:
        raise ValueError("registry does not declare registry_sha256")
    return reference_ids, counts, declared_hash


def reassign_user_disjoint(
    records: Sequence[RawTrajectoryRecord],
    feature_vectors: np.ndarray,
    fixed_split: Mapping[str, Sequence[int]],
    reference_ids: Iterable[str],
    *,
    exclude_references: bool,
    require_formal_fake_counts: bool = True,
) -> Tuple[List[RawTrajectoryRecord], np.ndarray, Dict[str, Any]]:
    """Use the same user owner for both labels, optionally dropping real refs."""

    owner = _owner_map(fixed_split)
    references = {str(value) for value in reference_ids}
    if len(references) == 0:
        raise ValueError("reference set must not be empty")
    features = np.asarray(feature_vectors, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(records) or not np.all(np.isfinite(features)):
        raise ValueError("feature matrix must be finite [N,D] and match records")

    assigned: List[RawTrajectoryRecord] = []
    kept_features: List[np.ndarray] = []
    found_references: Set[str] = set()
    dropped_references: Set[str] = set()
    identity: Set[Tuple[int, str]] = set()
    for record, feature in zip(records, features):
        record.validate()
        if int(record.user_id) not in owner:
            raise ValueError("record user is absent from fixed split")
        row_identity = (int(record.label), str(record.sample_id))
        if row_identity in identity:
            raise ValueError("duplicate label/sample_id in input bundle")
        identity.add(row_identity)
        event_id = str(record.event_group_id or record.sample_id)
        is_reference = int(record.label) == 0 and event_id in references
        if is_reference:
            if event_id in found_references:
                raise ValueError("reference real event occurs more than once")
            found_references.add(event_id)
            if exclude_references:
                dropped_references.add(event_id)
                continue
        assigned.append(replace(record, pool=owner[int(record.user_id)]))
        kept_features.append(np.asarray(feature, dtype=np.float64))

    missing = references - found_references
    if missing:
        raise ValueError("%d registry references are absent from real bundle" % len(missing))
    if exclude_references and dropped_references != references:
        raise RuntimeError("reference exclusion was incomplete")

    output_features = np.stack(kept_features, axis=0)
    counts = {
        label_name: {
            pool: sum(
                int(row.label) == label and row.pool == pool for row in assigned
            )
            for pool in POOLS
        }
        for label, label_name in ((0, "real"), (1, "fake"))
    }
    users = {
        label_name: {
            pool: sorted({
                int(row.user_id)
                for row in assigned
                if int(row.label) == label and row.pool == pool
            })
            for pool in POOLS
        }
        for label, label_name in ((0, "real"), (1, "fake"))
    }
    for label_name in ("real", "fake"):
        for pool in POOLS:
            if set(users[label_name][pool]) != set(int(x) for x in fixed_split[pool]):
                raise ValueError(
                    "%s %s users do not exactly equal the frozen user split"
                    % (label_name, pool)
                )
    fake_per_user = {
        user_id: sum(
            int(row.label) == 1 and int(row.user_id) == user_id for row in assigned
        )
        for user_id in range(100)
    }
    if require_formal_fake_counts:
        if counts["fake"] != {"train": 14000, "val": 2000, "test": 4000}:
            raise ValueError("formal fake counts are not 14k/2k/4k")
        if set(fake_per_user.values()) != {200}:
            raise ValueError("formal fake records are not exactly 200/user")

    remaining_reference_rows = sum(
        int(row.label) == 0
        and str(row.event_group_id or row.sample_id) in references
        for row in assigned
    )
    expected_remaining = 0 if exclude_references else len(references)
    if remaining_reference_rows != expected_remaining:
        raise RuntimeError("unexpected number of reference rows after reassignment")
    return assigned, output_features, {
        "schema_version": "trajectory_pad_user_disjoint_assignment_v1",
        "policy": "same_fixed_user_70_10_20_for_real_and_fake",
        "exclude_references": bool(exclude_references),
        "input_records": int(len(records)),
        "output_records": int(len(assigned)),
        "reference_ids": int(len(references)),
        "reference_rows_found": int(len(found_references)),
        "reference_rows_dropped": int(len(dropped_references)),
        "reference_rows_remaining": int(remaining_reference_rows),
        "counts": counts,
        "users": users,
        "fake_samples_per_user_min_max": [
            min(fake_per_user.values()), max(fake_per_user.values())
        ],
    }


def build_supplement_bundle(
    primary_bundle_dir: Path,
    output_dir: Path,
    split_json: Path,
    registry_map_path: Path,
    *,
    exclude_references: bool,
) -> Dict[str, Any]:
    """Build one immutable supplementary bundle through a staging directory."""

    source_dir = Path(primary_bundle_dir).resolve()
    output = Path(output_dir).resolve()
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError("refusing to overwrite output/staging: %s" % output)
    source_manifest = source_dir / "bundle_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError("primary bundle manifest is missing")
    primary = json.loads(source_manifest.read_text(encoding="utf-8"))
    if primary.get("status") != "complete":
        raise ValueError("primary PAD bundle is not complete")

    split_path = Path(split_json).resolve()
    split = load_fake_user_split(split_path)
    owner = _owner_map(split)
    registry_map_file = Path(registry_map_path).resolve()
    registry_map = json.loads(registry_map_file.read_text(encoding="utf-8"))
    if set(registry_map) != set(ACTIONS):
        raise ValueError("registry map must contain exactly five actions")

    staging.mkdir(parents=True)
    try:
        action_audits: Dict[str, Any] = {}
        outputs: Dict[str, Any] = {}
        registry_hashes: Dict[str, str] = {}
        for action in ACTIONS:
            source_file = source_dir / (action + ".npz")
            records, features = load_raw_sequence_bundle(source_file)
            if any(row.action != action for row in records):
                raise ValueError("primary bundle action mismatch for %s" % action)
            references, expected_reference_pools, registry_hash = load_reference_ids(
                Path(registry_map[action]), action=action, owner=owner
            )
            assigned, assigned_features, audit = reassign_user_disjoint(
                records,
                features,
                split,
                references,
                exclude_references=exclude_references,
                require_formal_fake_counts=True,
            )
            destination = staging / (action + ".npz")
            save_raw_sequence_bundle(destination, assigned, assigned_features)
            outputs[action] = {
                "path": str(output / (action + ".npz")),
                "sha256": sha256_file(destination),
                "records": len(assigned),
                "feature_dim": int(assigned_features.shape[1]),
                "source_sha256": sha256_file(source_file),
            }
            audit["expected_reference_counts_by_user_pool"] = expected_reference_pools
            action_audits[action] = audit
            registry_hashes[action] = registry_hash

        manifest = {
            "schema_version": SCHEMA,
            "status": "complete",
            "variant": (
                "fully_user_disjoint_reference_excluded"
                if exclude_references
                else "fully_user_disjoint"
            ),
            "split_policy": "same_fixed_user_70_10_20_for_real_and_fake",
            "exclude_references": bool(exclude_references),
            "primary_bundle_dir": str(source_dir),
            "primary_bundle_manifest": str(source_manifest),
            "primary_bundle_manifest_sha256": sha256_file(source_manifest),
            "fixed_user_split": str(split_path),
            "fixed_user_split_sha256": sha256_file(split_path),
            "reference_registry_map": str(registry_map_file),
            "reference_registry_map_sha256": sha256_file(registry_map_file),
            "reference_registry_sha256_by_action": registry_hashes,
            "v16_root": str(V16_ROOT),
            "v16_dependency_sha256": {
                "detectors/deep_pad.py": sha256_file(V16_ROOT / "detectors/deep_pad.py"),
                "trajectory/features.py": sha256_file(V16_ROOT / "trajectory/features.py"),
            },
            "actions": action_audits,
            "outputs": outputs,
        }
        _atomic_json(staging / "bundle_manifest.json", manifest)
        os.replace(str(staging), str(output))
        return manifest
    except BaseException:
        # Preserve partial staging for forensic review; never publish it as complete.
        raise


def audit_supplement_bundle(
    bundle_dir: Path,
    *,
    require_variant: str | None = None,
) -> Dict[str, Any]:
    """Recompute hashes, user ownership, class counts and reference exclusion."""

    root = Path(bundle_dir).resolve()
    manifest_path = root / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("supplement bundle manifest is incomplete/unsupported")
    if require_variant is not None and manifest.get("variant") != require_variant:
        raise ValueError("supplement variant mismatch")
    split_path = Path(manifest["fixed_user_split"])
    registry_map_path = Path(manifest["reference_registry_map"])
    source_manifest = Path(manifest["primary_bundle_manifest"])
    bindings = {
        "fixed_user_split_sha256": sha256_file(split_path),
        "reference_registry_map_sha256": sha256_file(registry_map_path),
        "primary_bundle_manifest_sha256": sha256_file(source_manifest),
    }
    for key, observed in bindings.items():
        if observed != manifest.get(key):
            raise ValueError("supplement source binding changed: %s" % key)
    split = load_fake_user_split(split_path)
    owner = _owner_map(split)
    registry_map = json.loads(registry_map_path.read_text(encoding="utf-8"))
    exclude = bool(manifest["exclude_references"])
    recomputed: Dict[str, Any] = {}
    global_identity: Set[Tuple[int, str]] = set()
    for action in ACTIONS:
        path = root / (action + ".npz")
        declared = manifest["outputs"][action]
        if sha256_file(path) != declared["sha256"]:
            raise ValueError("supplement output hash changed for %s" % action)
        records, features = load_raw_sequence_bundle(path)
        references, _, registry_hash = load_reference_ids(
            Path(registry_map[action]), action=action, owner=owner
        )
        if registry_hash != manifest["reference_registry_sha256_by_action"][action]:
            raise ValueError("registry identity changed for %s" % action)
        counts = {
            label_name: {
                pool: sum(row.label == label and row.pool == pool for row in records)
                for pool in POOLS
            }
            for label, label_name in ((0, "real"), (1, "fake"))
        }
        for row in records:
            row.validate()
            if row.pool != owner[int(row.user_id)]:
                raise ValueError("row pool does not equal fixed user owner")
            identity = (int(row.label), str(row.sample_id))
            if identity in global_identity:
                raise ValueError("duplicate sample identity across supplement bundle")
            global_identity.add(identity)
        users = {
            label_name: {
                pool: sorted({
                    int(row.user_id) for row in records
                    if row.label == label and row.pool == pool
                })
                for pool in POOLS
            }
            for label, label_name in ((0, "real"), (1, "fake"))
        }
        for label_name in ("real", "fake"):
            for pool in POOLS:
                if set(users[label_name][pool]) != set(split[pool]):
                    raise ValueError("audit user coverage mismatch")
        remaining = sum(
            row.label == 0
            and str(row.event_group_id or row.sample_id) in references
            for row in records
        )
        expected = 0 if exclude else 500
        if remaining != expected:
            raise ValueError("reference exclusion audit mismatch for %s" % action)
        if counts != manifest["actions"][action]["counts"]:
            raise ValueError("recomputed class/pool counts changed for %s" % action)
        recomputed[action] = {
            "records": len(records),
            "feature_dim": int(features.shape[1]),
            "counts": counts,
            "users": {label: {pool: len(value) for pool, value in pools.items()} for label, pools in users.items()},
            "reference_rows_remaining": int(remaining),
            "sha256": sha256_file(path),
        }
    receipt = {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed",
        "variant": manifest["variant"],
        "bundle_manifest": str(manifest_path),
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "actions": recomputed,
        "global_unique_label_sample_id_count": len(global_identity),
    }
    _atomic_json(root / "bundle_audit.json", receipt)
    return receipt

