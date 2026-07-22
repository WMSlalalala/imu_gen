"""Leakage-safe level-2 pools built only from base validation/test events."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from generation.protocol import FixedUserSplit

from .fake_imu_pairs import sha256_file
from .paired_dataset_builder import COMPONENT_TABLE_SCHEMA, DetectorComponentTable


META_POOL_SCHEMA = "paired_total_detector_meta_pool_v1"


def _real_validation_meta_pool(sample_id: str) -> str:
    raw = hashlib.sha256(("total-meta-real|" + str(sample_id)).encode("utf-8")).digest()
    return "train" if int.from_bytes(raw[:8], "little") % 10 < 6 else "val"


def remap_component_to_meta_pools(
    *,
    input_path: Path,
    expected_component: str,
    split_json: Path,
    output_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Exclude base train, split base validation 60/40, preserve base test."""

    table = DetectorComponentTable.load(input_path, expected_component)
    split = FixedUserSplit.load(str(split_json), require_formal=True)
    meta_train_fake_users = tuple(sorted(split.val_users)[:6])
    meta_val_fake_users = tuple(sorted(split.val_users)[6:])
    if len(meta_train_fake_users) != 6 or len(meta_val_fake_users) != 4:
        raise ValueError("formal base validation split must contain exactly ten fake users")
    keep = np.isin(table.pools, ["val", "test"])
    if not np.any(keep):
        raise ValueError("component has no base validation/test rows")
    selected = np.flatnonzero(keep)
    new_pools = []
    for index in selected.tolist():
        original = str(table.pools[index])
        label = int(table.labels[index])
        user = int(table.user_ids[index])
        if original == "test":
            new_pools.append("test")
        elif label == 1:
            if user in meta_train_fake_users:
                new_pools.append("train")
            elif user in meta_val_fake_users:
                new_pools.append("val")
            else:
                raise ValueError("fake base-validation user is absent from fixed split")
        else:
            new_pools.append(_real_validation_meta_pool(str(table.sample_ids[index])))
    new_pools_array = np.asarray(new_pools)
    labels = table.labels[selected]
    users = table.user_ids[selected]
    for pool in ("train", "val", "test"):
        if set(np.unique(labels[new_pools_array == pool]).tolist()) != {0, 1}:
            raise ValueError("meta pool %s lacks real/fake rows" % pool)
    fake_users = {
        pool: set(users[(new_pools_array == pool) & (labels == 1)].astype(int).tolist())
        for pool in ("train", "val", "test")
    }
    if (
        fake_users["train"] != set(meta_train_fake_users)
        or fake_users["val"] != set(meta_val_fake_users)
        or fake_users["test"] != set(split.test_users)
        or any(fake_users[a] & fake_users[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    ):
        raise ValueError("meta fake-user pools violate the 6/4/20 disjoint protocol")

    order = np.argsort(table.sample_ids[selected], kind="stable")
    selected = selected[order]
    new_pools_array = new_pools_array[order]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        str(temporary), schema_version=np.asarray(COMPONENT_TABLE_SCHEMA),
        component=np.asarray(expected_component), features=table.features[selected],
        feature_names=np.asarray(table.feature_names), sample_ids=table.sample_ids[selected],
        labels=table.labels[selected], user_ids=table.user_ids[selected], pools=new_pools_array,
        actions=table.actions[selected], duration_ms=table.duration_ms[selected],
        pair_identity_sha256=table.pair_identity_sha256[selected],
    )
    temporary.replace(output)
    remapped = DetectorComponentTable.load(output, expected_component)
    report: Dict[str, Any] = {
        "schema_version": META_POOL_SCHEMA,
        "status": "complete",
        "component": expected_component,
        "action": str(remapped.actions[0]),
        "base_train_rows_excluded": int(np.sum(table.pools == "train")),
        "base_validation_rows_partitioned": int(np.sum(table.pools == "val")),
        "base_test_rows_preserved": int(np.sum(table.pools == "test")),
        "meta_rows": int(len(remapped.labels)),
        "meta_pool_counts": {
            pool: {
                "real": int(np.sum((remapped.pools == pool) & (remapped.labels == 0))),
                "fake": int(np.sum((remapped.pools == pool) & (remapped.labels == 1))),
            }
            for pool in ("train", "val", "test")
        },
        "meta_fake_users": {pool: sorted(values) for pool, values in fake_users.items()},
        "level_1_fit_source": "base_train_only",
        "level_2_fit_source": "base_validation_partition_A_only",
        "level_2_threshold_source": "base_validation_partition_B_only",
        "level_2_test_source": "base_test_fixed_reporting_only",
        "real_validation_partition": "sha256(total-meta-real|sample_id) modulo 10: 0..5 train, 6..9 val",
        "input": str(Path(input_path).resolve()),
        "input_sha256": sha256_file(input_path),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "fixed_split_sha256": split.source_sha256,
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = target.with_name(target.name + ".tmp")
    temporary_manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(target)
    return report


__all__ = ["META_POOL_SCHEMA", "remap_component_to_meta_pools"]
