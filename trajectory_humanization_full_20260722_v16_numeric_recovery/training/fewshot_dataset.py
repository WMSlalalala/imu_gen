"""Fail-closed five-reference dataset and lossless variable-length collation."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from trajectory.data import (
    FORMAL_REF_COUNT,
    FewShotExample,
    TrajectoryBatch,
    collate_fewshot_trajectories,
    validate_fewshot_references,
)

from .corpus import NumericTrajectoryCorpus, atomic_json_dump


class ReferenceRegistry:
    """One immutable set of five refs for every action/user/split group."""

    PROTOCOL = "fixed_five_real_refs_per_user_action_split_v1"

    def __init__(
        self,
        corpus: NumericTrajectoryCorpus,
        seed: int,
        mapping: Dict[str, Dict[int, Tuple[int, ...]]],
    ) -> None:
        self.corpus = corpus
        self.seed = int(seed)
        self.mapping = mapping
        self._validate()
        self.payload = self._payload()
        canonical = json.dumps(self.payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.sha256 = hashlib.sha256(canonical).hexdigest()

    @classmethod
    def build(cls, corpus: NumericTrajectoryCorpus, seed: int = 42) -> "ReferenceRegistry":
        mapping: Dict[str, Dict[int, Tuple[int, ...]]] = {}
        for split in ("train", "val", "test"):
            mapping[split] = {}
            for user_id in corpus.splits.users(split):
                candidates = corpus.indices_for_user(split, user_id)
                if candidates.size < FORMAL_REF_COUNT + 1:
                    raise ValueError(
                        "fail closed: %s/%s/user=%d has %d events; fixed refs plus target need >=6"
                        % (corpus.action, split, user_id, candidates.size)
                    )
                # Rank each immutable numeric event id by SHA-256.  This is
                # independent of NumPy/Python RNG implementations and corpus
                # row order, while remaining seed/source/split/user bound.
                ranked = []
                for candidate in candidates.tolist():
                    event_id = int(corpus.event_ids[int(candidate)])
                    material = "%d|%s|%s|%d|%d" % (
                        int(seed), corpus.sha256, split, int(user_id), event_id
                    )
                    ranked.append((hashlib.sha256(material.encode("utf-8")).digest(), event_id, int(candidate)))
                ranked.sort(key=lambda item: (item[0], item[1]))
                selected = tuple(item[2] for item in ranked[:FORMAL_REF_COUNT])
                mapping[split][int(user_id)] = selected
        return cls(corpus, seed, mapping)

    def _validate(self) -> None:
        if set(self.mapping) != {"train", "val", "test"}:
            raise ValueError("reference registry must cover all three splits")
        for split in ("train", "val", "test"):
            if set(self.mapping[split]) != set(self.corpus.splits.users(split)):
                raise ValueError("reference registry user coverage mismatch for %s" % split)
            for user_id, indices in self.mapping[split].items():
                if len(indices) != FORMAL_REF_COUNT or len(set(indices)) != FORMAL_REF_COUNT:
                    raise ValueError("registry needs exactly five unique refs")
                valid = set(int(x) for x in self.corpus.indices_for_user(split, user_id).tolist())
                if any(int(index) not in valid for index in indices):
                    raise ValueError("registry reference outside same user/action/split")

    def _payload(self) -> Dict[str, object]:
        entries = []
        for split in ("train", "val", "test"):
            for user_id in sorted(self.mapping[split]):
                indices = self.mapping[split][user_id]
                entries.append(
                    {
                        "action": self.corpus.action,
                        "user_id": int(user_id),
                        "split": split,
                        "reference_event_ids": [int(self.corpus.event_ids[index]) for index in indices],
                    }
                )
        split_order = {"train": 0, "val": 1, "test": 2}
        entries.sort(key=lambda row: (split_order[str(row["split"])], int(row["user_id"])))
        return {
            # This is deliberately byte/hash compatible with
            # generation.protocol.ReferenceRegistry.  A per-action training
            # registry can therefore be consumed directly by that action's
            # formal generation process, or merged across five actions.
            "schema_version": 1,
            "producer": "trajectory_training_pipeline",
            "split_sha256": self.corpus.splits.sha256,
            "entries": entries,
        }

    def indices(self, split: str, user_id: int) -> Tuple[int, ...]:
        return self.mapping[split][int(user_id)]

    def sample_ids(self, split: str, user_id: int) -> Tuple[str, ...]:
        return tuple(str(int(self.corpus.event_ids[index])) for index in self.indices(split, user_id))

    def save(self, path) -> None:
        payload = dict(self.payload)
        payload["registry_sha256"] = self.sha256
        # Extra audit fields are not part of the canonical registry hash;
        # generation's strict loader ignores them while checking the same
        # entries/split digest.
        payload.update(
            {
                "action": self.corpus.action,
                "seed": self.seed,
                "corpus_npz": str(self.corpus.path),
                "corpus_sha256": self.corpus.sha256,
                "split_json": str(self.corpus.splits.path),
                "references_per_group": FORMAL_REF_COUNT,
                "reference_samples_excluded_from_target_pool": True,
            }
        )
        path = Path(path)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise ValueError("existing reference registry differs; refusing overwrite")
            return
        atomic_json_dump(path, payload)


class StrictFiveReferenceDataset(Dataset):
    """Every real target receives five unique same-user/action/split refs.

    No target is filtered because it is long or because its reference pool is
    small.  A user/action/split pool with fewer than six real events raises at
    construction time, before the first optimizer update.
    """

    def __init__(
        self,
        corpus: NumericTrajectoryCorpus,
        split: str,
        registry: ReferenceRegistry,
        seed: int = 42,
        cache_size: int = 2048,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train/val/test")
        if int(cache_size) < 0:
            raise ValueError("cache_size cannot be negative")
        self.corpus = corpus
        self.split = split
        self.seed = int(seed)
        self.registry = registry
        if registry.corpus is not corpus or registry.seed != self.seed:
            raise ValueError("registry must be built for this exact corpus and seed")
        self.epoch = 0
        all_indices = corpus.indices_for_split(split)
        reference_indices = {
            index
            for user_id in corpus.splits.users(split)
            for index in registry.indices(split, user_id)
        }
        self.indices = np.asarray(
            [int(index) for index in all_indices.tolist() if int(index) not in reference_indices],
            dtype=np.int64,
        )
        # There is deliberately no max_samples argument and no slicing.  The
        # only excluded events are the protocol-defined fixed refs, which are
        # still consumed by the reference encoder.
        self.expected_full_count = int(self.indices.size)
        if self.expected_full_count == 0:
            raise ValueError("empty formal %s split for %s" % (split, corpus.action))
        self._groups: Dict[int, np.ndarray] = {}
        for user_id in corpus.splits.users(split):
            group = corpus.indices_for_user(split, user_id)
            if group.size < FORMAL_REF_COUNT + 1:
                raise ValueError(
                    "fail closed: %s/%s/user=%d has %d events; target+5 refs needs >=6"
                    % (corpus.action, split, user_id, group.size)
                )
            self._groups[int(user_id)] = group

        # lru_cache is per DataLoader process.  It bounds memory while avoiding
        # repeated canonicalization of references reused in nearby batches.
        @lru_cache(maxsize=int(cache_size))
        def cached(index: int):
            return self.corpus.canonical_sample(index)

        self._cached_sample = cached

    def __len__(self) -> int:
        return self.expected_full_count

    def set_epoch(self, epoch: int) -> None:
        value = int(epoch)
        if value < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = value

    def _reference_indices(self, target_index: int, user_id: int) -> np.ndarray:
        refs = np.asarray(self.registry.indices(self.split, int(user_id)), dtype=np.int64)
        if len(set(int(x) for x in refs.tolist())) != FORMAL_REF_COUNT or int(target_index) in refs:
            raise AssertionError("reference sampling violated uniqueness/exclusion")
        return refs

    def __getitem__(self, item: int) -> FewShotExample:
        target_index = int(self.indices[int(item)])
        target = self._cached_sample(target_index)
        if target.split != self.split or target.action != self.corpus.action or not target.is_real:
            raise ValueError("target provenance contradicts dataset split/action")
        ref_indices = self._reference_indices(target_index, target.user_id)
        references = [self._cached_sample(int(index)) for index in ref_indices]
        validate_fewshot_references(target, references, FORMAL_REF_COUNT)
        return FewShotExample(target=target, references=references)

    def padded_length_components(self, item: int) -> Tuple[int, int]:
        """Exact target/reference temporal padding components for one item."""
        target_points, reference_points, _, _ = self.padded_shape_components(item)
        return int(target_points), int(reference_points)

    def padded_shape_components(self, item: int) -> Tuple[int, int, int, int]:
        """Exact target/ref timeline and key-token padding components."""
        target_index = int(self.indices[int(item)])
        user_id = int(self.corpus.user_ids[target_index])
        reference_indices = self.registry.indices(self.split, user_id)
        target_points = self.corpus.canonical_max_points(target_index)
        reference_points = max(
            self.corpus.canonical_max_points(index)
            for index in reference_indices
        )
        # Collation always allocates at least one key-token slot, including
        # non-keystroke actions whose semantic key count is zero.
        target_keys = max(1, self.corpus.event_key_count(target_index))
        reference_keys = max(
            1, max(self.corpus.event_key_count(index) for index in reference_indices)
        )
        return (
            int(target_points), int(reference_points),
            int(target_keys), int(reference_keys),
        )

    def padded_length_key(self, item: int) -> int:
        """Exact scalar sort key; compute cost keeps both components."""
        return max(self.padded_length_components(item))

    def reference_audit(self) -> Dict[str, object]:
        """Exhaustively prove all targets have valid reference provenance."""
        checked = 0
        unique_pairs = set()
        for position in range(len(self)):
            target_index = int(self.indices[position])
            user_id = int(self.corpus.user_ids[target_index])
            refs = self._reference_indices(target_index, user_id)
            if refs.size != FORMAL_REF_COUNT or target_index in refs:
                raise AssertionError("reference audit failed")
            for ref in refs.tolist():
                if int(self.corpus.user_ids[int(ref)]) != user_id:
                    raise AssertionError("cross-user reference")
                unique_pairs.add((target_index, int(ref)))
            checked += 1
        return {
            "action": self.corpus.action,
            "split": self.split,
            "targets_checked": checked,
            "references_per_target": FORMAL_REF_COUNT,
            "target_in_refs": 0,
            "duplicate_refs": 0,
            "cross_user_refs": 0,
            "cross_split_refs": 0,
            "unique_target_ref_pairs": len(unique_pairs),
            "registry_sha256": self.registry.sha256,
            "fixed_refs_shared_by_all_targets_in_group": True,
            "reference_samples_excluded_from_target_pool": True,
            "passed": checked == len(self),
        }


class StrictVariableLengthCollator:
    """Pad only to the longest item in this batch; never truncate or drop."""

    def __call__(self, examples: Sequence[FewShotExample]) -> TrajectoryBatch:
        if not examples:
            raise ValueError("empty batch")
        target_lengths = [
            tuple(int(pointer.shape[0]) for pointer in example.target.pointer_features)
            for example in examples
        ]
        reference_lengths = [
            [tuple(int(pointer.shape[0]) for pointer in ref.pointer_features) for ref in example.references]
            for example in examples
        ]
        # max_points=None is intentional: collate raises rather than truncates.
        batch = collate_fewshot_trajectories(examples, max_points=None)
        batch.validate(require_references=True, expected_refs=FORMAL_REF_COUNT)
        if batch.features.shape[0] != len(examples):
            raise AssertionError("collation dropped targets")
        for i, lengths in enumerate(target_lengths):
            observed = tuple(int(batch.point_mask[i, p].sum().item()) for p in range(len(lengths)))
            if observed != lengths:
                raise AssertionError("target was truncated during collation")
            for j, ref_lengths in enumerate(reference_lengths[i]):
                ref_observed = tuple(
                    int(batch.ref_point_mask[i, j, p].sum().item())
                    for p in range(len(ref_lengths))
                )
                if ref_observed != ref_lengths:
                    raise AssertionError("reference was truncated during collation")
        return batch


class DeterministicLengthBucketBatchSampler(BatchSampler):
    """Group similar event lengths while covering every target exactly once."""

    def __init__(
        self,
        dataset: StrictFiveReferenceDataset,
        batch_size: int,
        epoch: int,
        shuffle: bool,
        bucket_batches: int = 20,
    ) -> None:
        if batch_size <= 0 or bucket_batches <= 0:
            raise ValueError("invalid batch/bucket size")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.epoch = int(epoch)
        self.shuffle = bool(shuffle)
        self.bucket_batches = int(bucket_batches)
        lengths = np.asarray(
            [dataset.padded_length_key(position) for position in range(len(dataset))],
            dtype=np.int64,
        )
        self.sorted_positions = np.argsort(lengths, kind="stable").astype(np.int64)

    def __len__(self) -> int:
        return int(math.ceil(len(self.dataset) / float(self.batch_size)))

    def __iter__(self):
        positions = self.sorted_positions.copy()
        bucket_size = self.batch_size * self.bucket_batches
        batches = []
        for left in range(0, positions.size, bucket_size):
            bucket = positions[left : left + bucket_size].copy()
            if self.shuffle:
                bucket = np.asarray(
                    sorted(
                        bucket.tolist(),
                        key=lambda position: hashlib.sha256(
                            ("bucket|%d|%d|%d" % (self.dataset.seed, self.epoch, int(position))).encode("utf-8")
                        ).digest(),
                    ),
                    dtype=np.int64,
                )
            for batch_left in range(0, bucket.size, self.batch_size):
                batch = bucket[batch_left : batch_left + self.batch_size]
                if batch.size:
                    batches.append([int(x) for x in batch.tolist()])
        if self.shuffle:
            batches.sort(
                key=lambda batch: hashlib.sha256(
                    ("batch|%d|%d|%s" % (self.dataset.seed, self.epoch, ",".join(str(x) for x in batch))).encode("utf-8")
                ).digest()
            )
        flattened = [value for batch in batches for value in batch]
        if len(flattened) != len(self.dataset) or set(flattened) != set(range(len(self.dataset))):
            raise AssertionError("length bucket sampler dropped or duplicated targets")
        return iter(batches)


def make_epoch_loader(
    dataset: StrictFiveReferenceDataset,
    batch_size: int,
    epoch: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("invalid batch_size/num_workers")
    dataset.set_epoch(epoch if dataset.split == "train" else 0)
    generator = torch.Generator()
    generator.manual_seed(dataset.seed + 1000003 * int(epoch))
    batch_sampler = DeterministicLengthBucketBatchSampler(
        dataset, batch_size=batch_size, epoch=epoch, shuffle=shuffle
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=int(num_workers),
        collate_fn=StrictVariableLengthCollator(),
        pin_memory=bool(pin_memory),
        generator=generator,
        persistent_workers=False,
    )
