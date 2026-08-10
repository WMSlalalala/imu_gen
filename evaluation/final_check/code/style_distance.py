#!/usr/bin/env python3
"""Same-person fake-real vs different-person real-real distance (experiment B).

Does the generation preserve the *target user's* behavioural style?  With a
distance D (not a similarity), the claim "fake preserves the target" is

    D_fake  = D(fake_u,  real_u)         # fake to its own target
    D_inter = D(real_v,  real_u), v!=u   # a stranger to the target
    D_intra = D(real_u1, real_u2)        # the target to itself

    ideal:  D_intra <= D_fake < D_inter

i.e. a fake sits between "the user vs themselves" and "the user vs a stranger".
If instead D_fake > D_inter, the fake is *farther* from the target than an
ordinary stranger -- the opposite of preserving style.  (Only with a
similarity would the inequality flip to S_fake > S_inter.)

Matched by action.  Per test user u we form Delta_u = D_inter,u - D_fake,u and
report the fraction of the 20 test users with Delta_u > 0, its median, and a
user-clustered bootstrap CI.  This answers "does generation keep the target's
style"; it is NOT a substitute for the detector competence gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RELEASE = Path("/mnt/share/mwang49/data7/direct100k_final/datasets")
SPLIT = Path("/mnt/share/mwang49/real-human/imu_gen/final/data/splits/users_seed42.json")
BUNDLE = {"tap": "tap_and_pinch", "pinch": "tap_and_pinch", "scroll": "scroll",
          "swipe": "swipe", "keystroke": "keystroke"}
ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")


def event_features(traj: np.ndarray, imu: np.ndarray) -> np.ndarray:
    """A fixed vector per event: per-channel stats of trajectory and inertia.

    These are generic per-channel statistics -- the same family the
    hmog_style detectors read -- so the distance lives in a behaviour space,
    not a raw-sample space that length alone would dominate.
    """

    feats = []
    for arr, chans in ((traj, (1, 2, 3, 5, 6)), (imu, (0, 1, 2, 3, 4, 5))):
        for c in chans:
            col = arr[:, c].astype(np.float64)
            feats += [col.mean(), col.std(), col.min(), col.max()]
    feats.append(len(traj))
    return np.asarray(feats, dtype=np.float64)


def load_user_events(action: str, user: str):
    path = RELEASE / BUNDLE[action] / "shards" / f"{user}.npz"
    if not path.is_file():
        return None, None
    with np.load(path, allow_pickle=True) as d:
        keep = d["action"] == action
        labels = d["label"][keep]
        offsets = d["offsets"]
        traj, imu = d["trajectory_flat"], d["imu_flat"]
        idx = np.flatnonzero(keep)
        fake, real = [], []
        for i in idx:
            s, e = int(offsets[i]), int(offsets[i + 1])
            if e - s < 2:
                continue
            f = event_features(traj[s:e], imu[s:e])
            (fake if d["label"][i] == 1 else real).append(f)
    return (np.vstack(fake) if fake else None,
            np.vstack(real) if real else None)


def median_dist_to_centroid(points: np.ndarray, centroid: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(points - centroid, axis=1)))


def user_clustered_bootstrap(deltas: np.ndarray, n: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        pick = rng.integers(0, len(deltas), len(deltas))  # resample users
        stats.append(float(np.median(deltas[pick])))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/mnt/share/mwang49/data7/session_rhythm_detector/results/style_distance.json")
    parser.add_argument("--actions", default="tap,scroll,swipe,pinch,keystroke")
    args = parser.parse_args()

    split = json.loads(SPLIT.read_text())
    test_users = [f"hmog_u{u:03d}" for u in split["test_users"]]

    report = {
        "spec": "D_fake=D(fake_u,real_u); D_inter=D(real_v,real_u); D_intra=D(real_u1,real_u2)",
        "claim": "fake preserves target style  <=>  D_fake < D_inter (distance, not similarity)",
        "ideal_chain": "D_intra <= D_fake < D_inter",
        "test_users": len(test_users),
        "actions": {},
    }

    for action in [a.strip() for a in args.actions.split(",") if a.strip()]:
        # Load all test users' events for this action.
        per_user = {}
        for u in test_users:
            fake, real = load_user_events(action, u)
            if real is not None and len(real) >= 4:
                per_user[u] = (fake, real)
        if len(per_user) < 3:
            report["actions"][action] = {"skipped": "too few users with real events"}
            continue

        # Standardize features across all events of this action.
        allf = np.vstack([r for f, r in per_user.values()]
                         + [f for f, r in per_user.values() if f is not None])
        mu, sd = allf.mean(0), allf.std(0) + 1e-9

        def z(x):
            return (x - mu) / sd

        centroids = {u: z(r).mean(0) for u, (f, r) in per_user.items()}
        deltas, rows = [], {}
        for u, (fake, real) in per_user.items():
            rz = z(real)
            # D_intra: split this user's real events in two, distance across.
            half = len(rz) // 2
            c1 = rz[:half].mean(0)
            d_intra = median_dist_to_centroid(rz[half:], c1)
            # D_fake: this user's fakes to their own real centroid.
            d_fake = median_dist_to_centroid(z(fake), centroids[u]) if fake is not None else float("nan")
            # D_inter: other users' reals to this user's real centroid.
            others = np.vstack([z(per_user[v][1]) for v in per_user if v != u])
            d_inter = median_dist_to_centroid(others, centroids[u])
            if not np.isnan(d_fake):
                deltas.append(d_inter - d_fake)
                rows[u] = {"D_intra": round(d_intra, 4), "D_fake": round(d_fake, 4),
                           "D_inter": round(d_inter, 4), "delta": round(d_inter - d_fake, 4),
                           "chain_ok": bool(d_intra <= d_fake < d_inter)}
        deltas = np.asarray(deltas)
        lo, hi = user_clustered_bootstrap(deltas)
        report["actions"][action] = {
            "users": len(deltas),
            "fraction_delta_pos": round(float(np.mean(deltas > 0)), 3),
            "median_delta": round(float(np.median(deltas)), 4),
            "bootstrap_ci95": [lo, hi],
            "median_D_intra": round(float(np.median([r["D_intra"] for r in rows.values()])), 4),
            "median_D_fake": round(float(np.median([r["D_fake"] for r in rows.values()])), 4),
            "median_D_inter": round(float(np.median([r["D_inter"] for r in rows.values()])), 4),
            "chain_ok_fraction": round(float(np.mean([r["chain_ok"] for r in rows.values()])), 3),
            "per_user": rows,
        }
        r = report["actions"][action]
        print(f"{action:10s} users={r['users']:2d}  "
              f"D_intra={r['median_D_intra']:.2f} <= D_fake={r['median_D_fake']:.2f} "
              f"< D_inter={r['median_D_inter']:.2f}  "
              f"Delta>0: {r['fraction_delta_pos']:.0%}  CI{r['bootstrap_ci95']}")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
