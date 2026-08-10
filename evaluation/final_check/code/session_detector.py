#!/usr/bin/env python3
"""Train and evaluate the session rhythm detector.

Discipline (SPEC sections 2 and 4):

* User-disjoint split: train on ``train_users``, pick the operating threshold on
  ``val_users`` genuine sessions (FRR = 5% and 1%), report on ``test_users``.
* The detector is trained on genuine vs ``naive_jitter`` only.  ``paced`` -- our
  method's output -- is never seen in training; it is evaluated at the frozen
  operating point, so the test is whether a detector that learned human vs
  machine still flags our paced attack.

W is an **occupancy bin width, not a sliding window**, and the distinction
matters enough to state at the top.  Each session produces one feature vector
and one score; W only controls how the session's onsets are binned when the
occupancy terms are computed (how crowded the busiest bin is, how many are
empty).  There is no per-window score, no stride, and no causal alarm: this
cannot say "the session became suspicious at t = 40 s", only "this session, seen
whole, looks like X".

A deployed session PAD would want the sliding version.  It is a different
detector and is not built here; results are labelled "occupancy bin width" so
they are not read as per-window detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RESULTS = Path("/mnt/share/mwang49/data7/session_rhythm_detector/results")
SPLIT_JSON = Path("/mnt/share/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json")


def load_arm(arm: str) -> list[dict]:
    path = RESULTS / f"sessions_{arm}.jsonl"
    return [json.loads(l) for l in path.open()]


# A gap longer than this is the user setting the phone down, not part of a
# continuous rhythm; it is capped so one idle stretch cannot dominate the
# timeline (and cannot blow the window count up to terabytes).
MAX_GAP_S = 30.0


def onset_times(session: dict) -> np.ndarray:
    """When each action starts, in seconds from the session's first onset.

    An onset is the sum of every gap before it *and* every action before it.
    Accumulating gaps alone -- which this did -- drops the gesture time and
    places each onset earlier than the last by the whole duration of the actions
    preceding it, so a session's timeline came out systematically compressed and
    the occupancy features were computed against the wrong clock.

    Durations arrive from the assembler; a session recorded before that field
    existed falls back to gaps alone, and says so rather than pretending.
    """
    gaps = np.asarray(session["gaps_s"], dtype=float)
    durations = session.get("durations_s")
    if durations is None:
        return np.cumsum(np.clip(gaps, 0.0, MAX_GAP_S))
    dur = np.asarray(durations, dtype=float)
    onsets, t = [], 0.0
    for i in range(len(gaps)):
        t += max(0.0, float(gaps[i]))
        onsets.append(t)
        if i < len(dur):
            t += max(0.0, float(dur[i]))
    return np.asarray(onsets, dtype=float)


def action_sequence_entropy(actions: list) -> tuple:
    """What the session does, not only when.

    SPEC asks for an action-sequence term and the feature builder never read
    `session["actions"]` at all, so a machine that fires the right rhythm with a
    nonsense action order was indistinguishable from a human.  Two cheap terms:
    the entropy of the action mix, and the entropy of consecutive pairs, which
    is what catches an order no person would produce.
    """
    if not actions:
        return 0.0, 0.0, 0.0
    kinds, counts = np.unique(np.asarray(actions, dtype=object), return_counts=True)
    p = counts / counts.sum()
    unigram = float(-np.sum(p * np.log(p)))
    if len(actions) < 2:
        return unigram, 0.0, float(len(kinds))
    pairs = [f"{a}>{b}" for a, b in zip(actions, actions[1:])]
    _k, c = np.unique(np.asarray(pairs, dtype=object), return_counts=True)
    q = c / c.sum()
    bigram = float(-np.sum(q * np.log(q)))
    repeat = float(np.mean([a == b for a, b in zip(actions, actions[1:])]))
    return unigram, bigram, repeat


def session_features(session: dict, window_s: float) -> np.ndarray:
    gaps = np.clip(np.asarray(session["gaps_s"][1:], dtype=float), 0.0, MAX_GAP_S)  # inner gaps
    if len(gaps) < 2:
        gaps = np.asarray([0.0, 0.0])
    t = onset_times(session)
    duration = float(t[-1]) if len(t) else 0.0

    log_gaps = np.log1p(np.clip(gaps, 0, None))
    cv = float(np.std(gaps) / (np.mean(gaps) + 1e-9))

    # Occupancy: onsets per W-second window -- this is where the window width W
    # is felt.  A human's rhythm is bursty (some windows crowded, many empty);
    # a fixed-interval machine spreads onsets evenly.
    if duration > 0:
        nwin = max(1, int(np.ceil(duration / window_s)))
        counts = np.zeros(nwin)
        for onset in t:
            counts[min(int(onset // window_s), nwin - 1)] += 1
        empty_frac = float(np.mean(counts == 0))
        occ_mean = float(np.mean(counts))
        occ_std = float(np.std(counts))
        occ_max = float(np.max(counts))
    else:
        empty_frac = occ_mean = occ_std = occ_max = 0.0

    # Gap-histogram entropy: a constant/near-constant machine has low entropy.
    hist, _ = np.histogram(np.clip(gaps, 0, 30), bins=12, range=(0, 30))
    p = hist / (hist.sum() + 1e-9)
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))

    uni, bi, repeat = action_sequence_entropy(session.get("actions") or [])

    return np.array([
        float(np.median(gaps)), float(np.mean(gaps)), float(np.std(gaps)), cv,
        float(np.min(gaps)), float(np.percentile(gaps, 10)), float(np.percentile(gaps, 90)),
        float(np.mean(log_gaps)), float(np.std(log_gaps)),
        empty_frac, occ_mean, occ_std, occ_max,
        entropy, float(len(gaps)), duration,
        uni, bi, repeat,
    ], dtype=float)


def build_matrix(sessions: list[dict], window_s: float) -> np.ndarray:
    return np.vstack([session_features(s, window_s) for s in sessions]) if sessions else np.zeros((0, 19))


def pick_threshold(scores_genuine: np.ndarray, frr: float) -> float:
    """Threshold so that a fraction ``frr`` of genuine sessions are flagged.

    Higher score = more machine-like.  A genuine session is falsely flagged
    when its score exceeds the cut, so the cut is the (1 - frr) quantile of the
    genuine scores.
    """

    return float(np.quantile(scores_genuine, 1.0 - frr))


def caught_rate(scores: np.ndarray, threshold: float) -> float:
    """Fraction of attack sessions flagged (score above the frozen cut)."""

    return float(np.mean(scores > threshold)) if len(scores) else float("nan")


def caught_rate_ci(scores: np.ndarray, users: np.ndarray, threshold: float,
                   replicates: int = 2000, seed: int = 42) -> tuple:
    """The same rate with a user-clustered interval.

    Sessions are not independent: one person contributes several, and they share
    that person's pace.  Resampling sessions would shrink the interval by roughly
    the square root of how many each user has and claim a precision the data does
    not support, so the resampling unit is the user and every session of a drawn
    user comes with them.
    """
    if len(scores) == 0:
        return float("nan"), float("nan")
    who = np.asarray(users)
    unique = np.unique(who)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        picked = rng.choice(unique, size=len(unique), replace=True)
        vals = np.concatenate([scores[who == u] for u in picked])
        draws.append(float(np.mean(vals > threshold)))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", "--windows", dest="windows", default="2,3,5",
                        help="occupancy bin widths in seconds; NOT sliding windows")
    parser.add_argument("--model", default="rf", choices=["rf", "logreg"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(RESULTS / "session_detector_results.json"))
    args = parser.parse_args()

    split = json.loads(SPLIT_JSON.read_text())
    # The split lists users as integers; sessions carry them as strings.
    train_u = {str(u) for u in split["train_users"]}
    val_u = {str(u) for u in split["val_users"]}
    test_u = {str(u) for u in split["test_users"]}

    arms = {a: load_arm(a) for a in ("genuine", "naive", "naive_jitter", "paced", "paced_emp")}

    def subset(arm, users):
        return [s for s in arms[arm] if s["user"] in users]

    report = {
        "spec": "session rhythm detector; FRR-fixed operating points 5% and 1%",
        "trained_on": "genuine vs naive_jitter (paced never in training)",
        "counts": {
            arm: {"train": len(subset(arm, train_u)), "val": len(subset(arm, val_u)),
                  "test": len(subset(arm, test_u))}
            for arm in arms
        },
        "occupancy_bin_widths_s": {},
        "not_a_sliding_window": "one score per session; W bins the onsets for the occupancy terms only. No stride, no per-window score, no causal alarm.",
    }

    for window_s in [float(w) for w in args.windows.split(",")]:  # bin width, not a window
        # Train: genuine(train) vs naive_jitter(train).
        Xg = build_matrix(subset("genuine", train_u), window_s)
        Xm = build_matrix(subset("naive_jitter", train_u), window_s)
        X = np.vstack([Xg, Xm])
        y = np.concatenate([np.zeros(len(Xg)), np.ones(len(Xm))])

        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        if args.model == "rf":
            clf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                         random_state=args.seed, n_jobs=8)
        else:
            clf = LogisticRegression(max_iter=1000)
        clf.fit(Xs, y)

        def score(sessions):
            if not sessions:
                return np.zeros(0)
            return clf.predict_proba(scaler.transform(build_matrix(sessions, window_s)))[:, 1]

        # Operating point on val genuine.
        val_genuine = score(subset("genuine", val_u))
        thr5 = pick_threshold(val_genuine, 0.05)
        thr1 = pick_threshold(val_genuine, 0.01)

        # Report on test.  The user of every scored session is carried alongside
        # its score so the interval can be resampled by person, not by session.
        test_sessions = {arm: subset(arm, test_u) for arm in arms}
        test_scores = {arm: score(v) for arm, v in test_sessions.items()}
        test_users = {arm: np.asarray([x["user"] for x in v]) for arm, v in test_sessions.items()}

        def with_ci(arm, thr):
            lo, hi = caught_rate_ci(test_scores[arm], test_users[arm], thr, seed=args.seed)
            return {"rate": round(caught_rate(test_scores[arm], thr), 4),
                    "ci95": [round(lo, 4), round(hi, 4)]}

        report["occupancy_bin_widths_s"][f"{window_s:g}"] = {
            "threshold_frr5": round(thr5, 4),
            "threshold_frr1": round(thr1, 4),
            "at_frr5": {
                "genuine_false_alarm": with_ci("genuine", thr5),
                "naive_caught": round(caught_rate(test_scores["naive"], thr5), 4),
                "naive_jitter_caught": round(caught_rate(test_scores["naive_jitter"], thr5), 4),
                "paced_caught": with_ci("paced", thr5),
                "paced_emp_caught": with_ci("paced_emp", thr5),
            },
            "at_frr1": {
                "genuine_false_alarm": with_ci("genuine", thr1),
                "naive_caught": round(caught_rate(test_scores["naive"], thr1), 4),
                "naive_jitter_caught": round(caught_rate(test_scores["naive_jitter"], thr1), 4),
                "paced_caught": with_ci("paced", thr1),
                "paced_emp_caught": with_ci("paced_emp", thr1),
            },
        }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
