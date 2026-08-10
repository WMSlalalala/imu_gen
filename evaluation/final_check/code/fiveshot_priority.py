"""Which five real recordings the victim actually gave up.

The threat model says the attacker holds five recordings of the victim.  Two
generators run per fake event -- one writes the touch trajectory, one writes the
inertial window -- and until now each drew its own five independently: different
algorithm, different seed, different candidate pool.  Measured over the four
gesture actions, 354 of 400 victim-action pairs came out completely disjoint, so
the attacker was in fact consuming ten recordings, not five.

The touch side's five are already frozen and published (`material_manifest.jsonl`,
five shots per victim per action).  They are the ones to keep, because they are
the published artefact and because a real event carries *both* channels -- the
touch trajectory and the inertial window belong to the same recording, which is
what "five recordings" ought to mean.

This module maps those five onto row numbers in the processed arrays the
inertial generator indexes, so `UserRefBank` can put them first.

    material_manifest.jsonl   source_cluster_id  ->  (user, action, shot ordinal)
    genuine_bindings.jsonl    source_cluster_id  ->  source_event_id
    hmog_<action>.npz         event_id           ->  row

Order is preserved: shot 0 of the material becomes the first reference, so a
`k_refs` sweep takes nested prefixes -- k=1 is inside k=3 is inside k=5, and k=5
is exactly the published five.  Beyond five the bank extends from the victim's
remaining events, which is the honest reading of an arm that asks what a
better-resourced attacker would get; such an arm is outside the five-shot claim
and has to be labelled that way.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

MATERIAL = Path("/mnt/share/mwang49/data7/results/direct100k/fiveshot_material/material_manifest.jsonl")
BINDINGS = Path("/mnt/share/mwang49/data7/results/direct100k/genuine_bindings_v1/genuine_bindings.jsonl")


@lru_cache(maxsize=1)
def _cluster_to_event() -> dict:
    out = {}
    for line in BINDINGS.open():
        row = json.loads(line)
        out[row["source_cluster_id"]] = int(row["source_event_id"])
    return out


@lru_cache(maxsize=8)
def _material_events(action: str) -> dict:
    """{user index -> [source_event_id, ...] in shot order} for one action."""
    mapping = _cluster_to_event()
    shots: dict[int, list[tuple[int, int]]] = {}
    unmapped = 0
    for line in MATERIAL.open():
        row = json.loads(line)
        if row["action"] != action:
            continue
        event = mapping.get(row["source_cluster_id"])
        if event is None:
            unmapped += 1
            continue
        user = int(str(row["user_id"]).split("_u")[-1])
        shots.setdefault(user, []).append((int(row.get("shot_ordinal", 0)), event))
    if unmapped:
        raise SystemExit(
            f"{unmapped} {action} material shots have no genuine binding; the "
            f"mapping is incomplete and a bank built on it would silently fall "
            f"back to the old behaviour"
        )
    return {u: [e for _o, e in sorted(v)] for u, v in shots.items()}


def priority_rows(action: str, event_ids: np.ndarray, user_ids: np.ndarray) -> dict:
    """{user -> row indices of that user's published five, in shot order}.

    `event_ids` and `user_ids` are the processed array's own columns, so the row
    numbers this returns are the ones `UserRefBank` indexes with.
    """
    wanted = _material_events(action)
    out: dict[int, np.ndarray] = {}
    for user, events in wanted.items():
        rows = np.flatnonzero(user_ids == user)
        if len(rows) == 0:
            continue
        by_event = {int(event_ids[r]): int(r) for r in rows}
        ordered = [by_event[e] for e in events if e in by_event]
        if ordered:
            out[user] = np.asarray(ordered, dtype=np.int64)
    return out


def coverage(action: str, event_ids: np.ndarray, user_ids: np.ndarray) -> dict:
    """How completely the five map through, per action -- for the build receipt.

    A victim whose five do not all resolve would silently get a shorter bank, so
    this is recorded rather than left to be noticed later.
    """
    rows = priority_rows(action, event_ids, user_ids)
    counts = [len(v) for v in rows.values()]
    return {
        "action": action,
        "victims_with_priority": len(rows),
        "victims_with_all_five": sum(1 for c in counts if c == 5),
        "min_resolved": min(counts) if counts else 0,
        "total_resolved": int(sum(counts)),
    }
