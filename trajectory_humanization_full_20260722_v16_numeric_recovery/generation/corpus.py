"""Generation-side view of the single authoritative training corpus loader.

There is deliberately no second NPZ-to-canonical implementation here.  Both
training and formal generation call ``training.corpus.NumericTrajectoryCorpus``
so timestamp de-duplication, UNKNOWN key tokens, pointer offsets and every
canonical tensor are byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from trajectory.data import ACTIONS, CanonicalTrajectory
from training.corpus import NumericTrajectoryCorpus, SplitDefinition

from .protocol import FixedUserSplit


def _shared_split(fixed: FixedUserSplit) -> SplitDefinition:
    shared = SplitDefinition.load(Path(fixed.source_path), require_pinned_hash=True)
    if shared.sha256 != fixed.source_sha256:
        raise ValueError("generation/training fixed split digest mismatch")
    if (
        shared.train_users != fixed.train_users
        or shared.val_users != fixed.val_users
        or shared.test_users != fixed.test_users
    ):
        raise ValueError("generation/training fixed split membership mismatch")
    return shared


def open_shared_corpus(path: str, action: str, split: FixedUserSplit) -> NumericTrajectoryCorpus:
    if action not in ACTIONS:
        raise ValueError("unknown action: %r" % action)
    return NumericTrajectoryCorpus(
        Path(path), _shared_split(split), expected_action=action,
        verify_sha256=True,
    )


def load_action_corpus(
    path: str,
    action: str,
    split: FixedUserSplit,
    user_ids: Optional[Iterable[int]] = None,
    strict: bool = True,
) -> List[CanonicalTrajectory]:
    """Return canonical samples produced by the exact training code path."""
    corpus = open_shared_corpus(path, action, split)
    allowed = None if user_ids is None else set(int(x) for x in user_ids)
    records: List[CanonicalTrajectory] = []
    errors = []
    for index in range(len(corpus)):
        if allowed is not None and int(corpus.user_ids[index]) not in allowed:
            continue
        try:
            records.append(corpus.canonical_sample(index))
        except Exception as exc:
            if strict:
                raise ValueError(
                    "shared canonical loader failed action=%s index=%d event_id=%s: %s"
                    % (action, index, corpus.event_ids[index], exc)
                ) from exc
            errors.append((index, str(exc)))
    if not records:
        raise ValueError("no usable %s events in %s" % (action, Path(path).resolve()))
    ids = [item.sample_id for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("shared corpus returned duplicate sample ids")
    return records


def load_corpus_directory(
    directory: str,
    split: FixedUserSplit,
    actions: Sequence[str] = ACTIONS,
    user_ids: Optional[Iterable[int]] = None,
    strict: bool = True,
) -> Dict[str, List[CanonicalTrajectory]]:
    root = Path(directory)
    return {
        action: load_action_corpus(
            str(root / ("hmog_trajectory_%s.npz" % action)), action, split,
            user_ids=user_ids, strict=strict,
        )
        for action in actions
    }
