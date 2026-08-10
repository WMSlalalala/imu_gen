#!/usr/bin/env python3
"""Assemble genuine / naive / paced session rhythm sequences.

A session here is a sequence of (action, gap-before) pairs -- the rhythm, not
the gesture shapes.  Genuine rhythms are reconstructed from the real HMOG
timeline (``flat_system_time_ms`` per touch sample, grouped by session).  The
naive and paced arms reuse the *same* genuine skeletons -- same action order,
same gesture count -- and replace only the inter-gesture gaps, so the sole
variable across the three arms is where the gaps come from.

Genuine gaps are the human's own.  Naive gaps are a constant machine interval.
Paced gaps are drawn by the ActReal DelayPolicy.  The paced arm is never used
to train the detector (see SPEC section 4); it is only ever evaluated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW_ROOT = Path(
    "/mnt/share/mwang49/real-human/imu_gen/final/"
    "trajectory_humanization_full_20260713/results/trajectories_full_v2"
)
SPLIT_JSON = Path("/mnt/share/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json")
ACTREAL = Path("/mnt/share/mwang49/data7/actreal_agent")

ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
# Below this a "gap" is really one gesture continuing, not a new action start.
# Filled by genuine_sessions() and written out by main(); the rejects are a
# result in their own right, not a debug aside.
REJECT_REPORT: dict = {}

MIN_GAP_S = 0.0
# Above this the user put the phone down; not part of a continuous rhythm.
MAX_GAP_S = 120.0


def load_split() -> dict:
    if SPLIT_JSON.is_file():
        return json.loads(SPLIT_JSON.read_text())
    return {}


def genuine_sessions() -> list[dict]:
    """One dict per (user, session): ordered actions and the gap before each."""

    # Collect every event's (user, session, action, start_ms, end_ms).
    events = defaultdict(list)  # (user, session) -> [(start, end, action)]
    for action in ACTIONS:
        path = RAW_ROOT / f"hmog_trajectory_{action}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=True) as d:
            st = d["flat_system_time_ms"]
            off = d["event_offsets"]
            sid = d["session_id"]
            uid = d["user_id"]
            for i in range(len(off) - 1):
                a, b = int(off[i]), int(off[i + 1])
                if b <= a:
                    continue
                events[(str(uid[i]), str(sid[i]))].append(
                    (float(st[a]), float(st[b - 1]), action)
                )

    sessions, rejected, short = [], [], []
    for (user, session), evs in events.items():
        evs.sort()
        if len(evs) < 3:
            continue
        actions = [e[2] for e in evs]
        durations = [(e[1] - e[0]) / 1000.0 for e in evs]
        gaps = [0.0]
        for prev, cur in zip(evs, evs[1:]):
            gaps.append((cur[0] - prev[1]) / 1000.0)

        # A session is kept only if *every* inner gap is physically possible.
        #
        # The earlier rule asked only that two gaps be plausible and then stored
        # the rest as they came, which let a session through carrying negative
        # gaps -- a later event starting before the previous one ended -- and
        # clock jumps as large as 1.4e15 seconds.  Those were not caught
        # downstream either: the feature builder clipped gaps into [0, 120], so
        # an impossible timeline silently became a merely unusual one and the
        # detector was asked to tell a human from a clock fault.
        #
        # Anomalies are counted rather than repaired, because a repaired gap is
        # a number nobody measured.
        inner = gaps[1:]
        bad_negative = sum(1 for g in inner if g < 0.0)
        bad_long = sum(1 for g in inner if g > MAX_GAP_S)
        if bad_negative or bad_long:
            rejected.append({
                "user": user, "session": session, "events": len(evs),
                "negative_gaps": bad_negative, "gaps_over_max": bad_long,
                "worst_gap_s": max(inner) if inner else None,
                "most_negative_gap_s": min(inner) if inner else None,
            })
            continue
        if sum(1 for g in inner if MIN_GAP_S <= g <= MAX_GAP_S) < 2:
            short.append({"user": user, "session": session, "events": len(evs)})
            continue

        sessions.append(
            {
                "user": user,
                "session": session,
                "actions": actions,
                # Kept so onset times can include how long each action lasted;
                # accumulating gaps alone places every onset too early by the
                # total gesture time before it.
                "durations_s": [round(d, 4) for d in durations],
                "gaps_s": [round(g, 4) for g in gaps],
                "source": "genuine",
            }
        )

    if rejected or short:
        summary = {
            "kept": len(sessions),
            "rejected_impossible_timeline": len(rejected),
            "rejected_too_few_usable_gaps": len(short),
            "negative_gaps_total": sum(r["negative_gaps"] for r in rejected),
            "gaps_over_max_total": sum(r["gaps_over_max"] for r in rejected),
            "worst_gap_s": max((r["worst_gap_s"] for r in rejected
                                if r["worst_gap_s"] is not None), default=None),
            "detail": rejected[:20],
        }
        REJECT_REPORT.clear()
        REJECT_REPORT.update(summary)
        print(f"  rejected {len(rejected)} sessions with impossible timelines "
              f"({summary['negative_gaps_total']} negative gaps, "
              f"{summary['gaps_over_max_total']} over {MAX_GAP_S}s); "
              f"see session_rejects.json")
    return sessions


def _delay_policy():
    sys.path.insert(0, str(ACTREAL))
    from actreal.pacing import DelayPolicy

    return DelayPolicy(enabled=False)  # enabled=False: we sample targets, never sleep


# Map HMOG actions onto the DelayPolicy's action keys.
POLICY_KEY = {
    "tap": "tap",
    "scroll": "gesture",
    "swipe": "gesture",
    "pinch": "gesture",
    "keystroke": "type",
}


def empirical_gap_pools(sessions: list[dict], train_users: set) -> dict:
    """Per-action gap pools from TRAIN-user genuine sessions only.

    An attacker may know the training population's rhythm; drawing from it is
    fair and never touches a test user.  Bucketing by the action that follows
    the gap keeps taps-after-taps distinct from a scroll landing after a pause.
    """

    import collections

    pools = collections.defaultdict(list)
    for s in sessions:
        if s["user"] not in train_users:
            continue
        for action, gap in zip(s["actions"][1:], s["gaps_s"][1:]):
            if 0.0 <= gap <= 120.0:
                pools[action].append(gap)
    return {a: np.asarray(v, dtype=float) for a, v in pools.items() if v}


def rewrite_gaps(sessions: list[dict], arm: str, *, naive_interval_s: float,
                 seed: int, gap_pools: dict | None = None) -> list[dict]:
    """Replace each session's inter-gesture gaps, keeping its skeleton.

    ``naive`` uses a constant interval; ``paced`` draws each gap from the
    DelayPolicy for that action.  Neither touches the action order or count.
    """

    out = []
    if arm == "paced":
        policy = _delay_policy()
    rng = np.random.default_rng(seed)
    for idx, s in enumerate(sessions):
        gaps = [0.0]
        for j, action in enumerate(s["actions"][1:], start=1):
            if arm == "naive":
                # A machine with a hard-coded sleep: one constant gap.
                gaps.append(naive_interval_s)
            elif arm == "naive_jitter":
                # A less naive agent: sleep(random.uniform(lo, hi)).  This is
                # the machine class the detector is trained against, so the
                # test is not won by "zero variance" alone.
                gaps.append(round(float(rng.uniform(naive_interval_s - 1.0,
                                                    naive_interval_s + 1.0)), 4))
            elif arm == "paced":
                key = POLICY_KEY.get(action, "default")
                # Deterministic per (session, position) so the arm is reproducible.
                gap = policy.target_for(key, seed * 100003 + idx * 131 + j)
                gaps.append(round(gap, 4))
            elif arm == "paced_emp":
                # v2 pacing: draw the gap from the training population's own
                # per-action gap distribution, so the heavy tail and burstiness
                # the log-normal policy could not reproduce come along for free.
                pool = gap_pools.get(action) if gap_pools else None
                if pool is None or len(pool) == 0:
                    pool = np.concatenate(list(gap_pools.values())) if gap_pools else np.array([1.0])
                gaps.append(round(float(pool[rng.integers(len(pool))]), 4))
            else:
                raise ValueError(arm)
        out.append({**s, "gaps_s": gaps, "source": arm})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/mnt/share/mwang49/data7/session_rhythm_detector/results")
    parser.add_argument("--naive-interval-s", type=float, default=4.0,
                        help="the constant machine gap; a plausible fixed agent sleep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sessions", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    genuine = genuine_sessions()
    if args.max_sessions:
        genuine = genuine[: args.max_sessions]
    split = load_split()
    train_users = {str(u) for u in split.get("train_users", [])}
    pools = empirical_gap_pools(genuine, train_users)

    naive = rewrite_gaps(genuine, "naive", naive_interval_s=args.naive_interval_s, seed=args.seed)
    naive_jitter = rewrite_gaps(genuine, "naive_jitter", naive_interval_s=args.naive_interval_s,
                                seed=args.seed)
    paced = rewrite_gaps(genuine, "paced", naive_interval_s=args.naive_interval_s, seed=args.seed)
    paced_emp = rewrite_gaps(genuine, "paced_emp", naive_interval_s=args.naive_interval_s,
                             seed=args.seed, gap_pools=pools)

    for arm, data in (("genuine", genuine), ("naive", naive),
                      ("naive_jitter", naive_jitter), ("paced", paced),
                      ("paced_emp", paced_emp)):
        path = out / f"sessions_{arm}.jsonl"
        with path.open("w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")

    # A quick sanity read-out so the three arms are visibly one skeleton, three gaps.
    def med_gap(data):
        g = [x for s in data for x in s["gaps_s"][1:]]
        return float(np.median(g)) if g else 0.0

    summary = {
        "genuine_sessions": len(genuine),
        "median_gap_s": {
            "genuine": round(med_gap(genuine), 3),
            "naive": round(med_gap(naive), 3),
            "naive_jitter": round(med_gap(naive_jitter), 3),
            "paced": round(med_gap(paced), 3),
            "paced_emp": round(med_gap(paced_emp), 3),
        },
        "naive_interval_s": args.naive_interval_s,
        "split_users": {
            k: len(v) for k, v in load_split().items() if isinstance(v, list)
        },
    }
    (out / "assemble_summary.json").write_text(json.dumps(summary, indent=2))
    if REJECT_REPORT:
        (out / "session_rejects.json").write_text(
            json.dumps(REJECT_REPORT, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
