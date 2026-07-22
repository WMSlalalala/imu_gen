"""Strict numeric input table for formal IMU+trajectory detector training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import re

import numpy as np


PAIRED_DATASET_SCHEMA = "paired_imu_trajectory_detector_table_v1"
ALLOWED_ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _scalar_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("schema_version must be a scalar string")
    return str(array.item())


@dataclass(frozen=True)
class PairedDetectorTable:
    features: np.ndarray
    feature_names: Tuple[str, ...]
    labels: np.ndarray
    user_ids: np.ndarray
    pools: np.ndarray
    sample_ids: np.ndarray
    actions: np.ndarray
    duration_ms: np.ndarray
    pair_identity_sha256: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "PairedDetectorTable":
        source = Path(path)
        with np.load(str(source), allow_pickle=False) as archive:
            required = {
                "schema_version", "features", "feature_names", "labels",
                "user_ids", "pools", "sample_ids", "actions", "duration_ms",
                "pair_identity_sha256",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError("paired detector table missing fields: %s" % sorted(missing))
            if _scalar_text(archive["schema_version"]) != PAIRED_DATASET_SCHEMA:
                raise ValueError("paired detector table schema mismatch")
            features = np.asarray(archive["features"], dtype=np.float64)
            feature_names = tuple(str(value) for value in np.asarray(archive["feature_names"]).tolist())
            labels = np.asarray(archive["labels"], dtype=np.int64)
            user_ids = np.asarray(archive["user_ids"])
            pools = np.asarray(archive["pools"]).astype(str)
            sample_ids = np.asarray(archive["sample_ids"]).astype(str)
            actions = np.asarray(archive["actions"]).astype(str)
            duration_ms = np.asarray(archive["duration_ms"], dtype=np.float64)
            pair_identity_sha256 = np.asarray(archive["pair_identity_sha256"]).astype(str)
        table = cls(
            features=features, feature_names=feature_names, labels=labels,
            user_ids=user_ids, pools=pools, sample_ids=sample_ids,
            actions=actions, duration_ms=duration_ms,
            pair_identity_sha256=pair_identity_sha256,
        )
        table.validate()
        return table

    def validate(self) -> None:
        if self.features.ndim != 2 or self.features.shape[0] == 0 or not np.all(np.isfinite(self.features)):
            raise ValueError("features must be finite non-empty [N,D]")
        n, width = self.features.shape
        if len(self.feature_names) != width or len(set(self.feature_names)) != width:
            raise ValueError("feature_names must uniquely describe every feature column")
        for prefix in ("imu__", "trajectory__", "consistency__"):
            if not any(name.startswith(prefix) for name in self.feature_names):
                raise ValueError("formal paired table requires %s features" % prefix)
        for name, value in (
            ("labels", self.labels), ("user_ids", self.user_ids),
            ("pools", self.pools), ("sample_ids", self.sample_ids),
            ("actions", self.actions), ("duration_ms", self.duration_ms),
            ("pair_identity_sha256", self.pair_identity_sha256),
        ):
            if np.asarray(value).ndim != 1 or len(value) != n:
                raise ValueError("%s must be a vector of length %d" % (name, n))
        if self.user_ids.dtype.kind == "O":
            raise ValueError("user_ids must not require pickle/object dtype")
        if set(np.unique(self.labels).tolist()) - {0, 1}:
            raise ValueError("labels must use real=0/fake=1")
        if set(np.unique(self.pools).tolist()) != {"train", "val", "test"}:
            raise ValueError("pools must contain exactly train/val/test")
        if np.any(self.sample_ids == "") or len(np.unique(self.sample_ids)) != n:
            raise ValueError("sample_ids must be non-empty and globally unique")
        if np.any(self.actions == ""):
            raise ValueError("actions must be non-empty")
        if set(np.unique(self.actions).tolist()) - set(ALLOWED_ACTIONS):
            raise ValueError("actions contain values outside the frozen five-action protocol")
        if any(_SHA256.fullmatch(value) is None for value in self.pair_identity_sha256.tolist()):
            raise ValueError("pair_identity_sha256 must contain lowercase 64-hex identities")
        if not np.all(np.isfinite(self.duration_ms)) or np.any(self.duration_ms <= 0):
            raise ValueError("duration_ms must be finite and positive")
        for pool in ("train", "val", "test"):
            labels = set(np.unique(self.labels[self.pools == pool]).tolist())
            if labels != {0, 1}:
                raise ValueError("%s must contain both real and fake rows" % pool)
        for action in np.unique(self.actions):
            for pool in ("train", "val", "test"):
                selected = (self.actions == action) & (self.pools == pool)
                if set(np.unique(self.labels[selected]).tolist()) != {0, 1}:
                    raise ValueError("%s/%s must contain both real and fake rows" % (action, pool))
        # The formal fake protocol is user-disjoint 70/10/20.  Real events may
        # use within-user event-group splits, so this gate applies to fake only.
        fake = self.labels == 1
        for user in np.unique(self.user_ids[fake]):
            observed = np.unique(self.pools[fake & (self.user_ids == user)])
            if observed.size != 1:
                raise ValueError("fake user %r crosses train/val/test pools" % user)


__all__ = ["ALLOWED_ACTIONS", "PAIRED_DATASET_SCHEMA", "PairedDetectorTable"]
