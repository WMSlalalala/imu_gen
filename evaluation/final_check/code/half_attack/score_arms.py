#!/usr/bin/env python
"""Half-attack SCORING phase -- the three new arms on the 20 classical joint cells.

Construction is NOT re-implemented here.  Every event is built by importing
`build_arms` (plan / build_one) unchanged, so the arms scored below are exactly
the arms the build phase validated.

Pipeline order, and it is not negotiable:

  1. `reproduce()` re-scores the release's OWN published test events with the
     same two functions used for every new arm (`extract_event_features` +
     `classical_scores`) and compares against the published test_scores.jsonl.gz
     of each cell.  If the max absolute difference is not at float round-off,
     the pipeline is not the frozen one and nothing downstream means anything.
  2. Only then are rows 3, 4, 5 (and the row5_unmatched diagnostic) built and
     scored.

METRIC.  FAR at the development-selected FRR=5% cut, the `frr5` scalar in each
cell's thresholds.json.  Accepted iff score < threshold (score_direction is
larger_is_more_fake in all 20 cells; asserted).  Rows 1 and 2 are read from the
published scores -- they ARE the release and are never rebuilt.

UNCERTAINTY (fairness rule F9).  The resampling unit is the user: the 20 test
users are resampled as clusters, 2000 replicates, ONE shared user multiset per
replicate across every cell and every arm, so paired differences are paired.

DURATION STRATIFICATION.  The build phase measured that row3-row2, row4-row5
and row5-row2 are duration-confounded (row3/row5 share the genuine carrier,
row4/row2 share the release fake carrier).  Every count is therefore also
carried in 10 duration bins whose edges are the release fake pool's own
duration deciles for that action, so the three confounded differences can be
re-read with the duration distribution standardised to the release arm's.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DIRECT = "/mnt/share/mwang49/data7/code/direct100k"
for p in (DIRECT, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import build_arms as B  # noqa: E402
from security_exp.event_detectors import (  # noqa: E402
    classical_scores,
    extract_event_features,
)

ACTIONS = B.ACTIONS
CLASSICAL = B.CLASSICAL
MODALITY = B.MODALITY
ARMS = ("row1", "row2", "row3", "row4", "row5", "row5_unmatched")
BUILT_ARMS = ("row3", "row4", "row5", "row5_unmatched")
FEATURE_FAMILY = {"hmog_style_svm": "feature", "hmog_style_rf": "feature",
                  "paper_svm": "paper", "paper_xgboost": "paper"}
FAMILY_REP = {"feature": "hmog_style_svm", "paper": "paper_svm"}
NBINS = 10
NBOOT = 2000
BOOT_SEED = 909090

CELLS = [f"{a}__{MODALITY}__{d}" for a in ACTIONS for d in CLASSICAL]
CELL_INDEX = {c: i for i, c in enumerate(CELLS)}
ARM_INDEX = {a: i for i, a in enumerate(ARMS)}
NDET = len(CLASSICAL)


# --------------------------------------------------------------------------- #
# frozen artefacts
# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_checksums(cells: dict) -> dict:
    out = {}
    for name in CELLS:
        c = cells[name]
        out[name] = {
            "model.joblib": sha256(Path(c["model_dir"]) / "model.joblib"),
            "thresholds.json": sha256(Path(c["thresholds"])),
        }
    return out


def load_models(action: str, cells: dict) -> dict:
    """Load the four frozen classical models for one action, single-threaded."""
    import joblib
    out = {}
    for det in CLASSICAL:
        c = cells[f"{action}__{MODALITY}__{det}"]
        assert c["score_direction"] == "larger_is_more_fake", c
        model = joblib.load(Path(c["model_dir"]) / "model.joblib")
        # Respect the 4-worker budget: no model may fan out threads on its own.
        try:
            model.set_params(n_jobs=1)
        except (ValueError, AttributeError, TypeError):
            pass
        thr = json.loads(Path(c["thresholds"]).read_text())
        assert thr["selection_split"] == "development", thr
        out[det] = (model, float(thr["frr5"]))
    return out


# --------------------------------------------------------------------------- #
# STEP 1 -- pipeline validation against the published scores
# --------------------------------------------------------------------------- #
def reproduce(cells: dict, users: list[str], book: B.ShardBook, actions) -> list[dict]:
    """Re-score the release's own published test events, exactly the way the new
    arms are scored: one feature vector per family per event, then every frozen
    classical model of that action."""
    out = []
    for action in actions:
        bundle = cells[f"{action}__{MODALITY}__{CLASSICAL[0]}"]["bundle"]
        models = load_models(action, cells)
        published = {det: {} for det in CLASSICAL}
        labels = {}
        for det in CLASSICAL:
            with gzip.open(cells[f"{action}__{MODALITY}__{det}"]["scores"], "rt") as f:
                for line in f:
                    d = json.loads(line)
                    if d["action"] != action:
                        continue
                    published[det][d["event_id"]] = float(d["fake_high_score"])
                    labels[d["event_id"]] = int(d["label"])
        ids, feats = [], {fam: [] for fam in FAMILY_REP}
        missing = 0
        for u in users:
            s = book.get(bundle, u)
            for i in range(len(s.label)):
                if s.action[i] != action:
                    continue
                eid = str(s.event_id[i])
                if eid not in labels:
                    missing += 1
                    continue
                imu, traj = s.signals(i)
                for fam, rep_det in FAMILY_REP.items():
                    feats[fam].append(
                        extract_event_features(rep_det, MODALITY, imu, traj))
                ids.append(eid)
        mats = {fam: np.stack(v).astype(np.float32) for fam, v in feats.items()}
        for det in CLASSICAL:
            model, cut = models[det]
            got = classical_scores(model, mats[FEATURE_FAMILY[det]])
            want = np.asarray([published[det][e] for e in ids])
            out.append({
                "cell": f"{action}__{MODALITY}__{det}",
                "published_events": len(published[det]),
                "rescored_events": int(got.size),
                "published_not_found_in_shards": int(len(published[det]) - len(ids)),
                "shard_events_not_in_published": int(missing),
                "max_abs_diff": float(np.max(np.abs(got - want))),
                "mean_abs_diff": float(np.mean(np.abs(got - want))),
                "exact_bitwise_fraction": float(np.mean(got == want)),
                "threshold_frr5": cut,
                "published_far5_label1": float(np.mean(
                    [published[det][e] < cut for e in published[det]
                     if labels[e] == 1])),
                "rescored_far5_label1": float(np.mean(
                    [g < cut for g, e in zip(got, ids) if labels[e] == 1])),
            })
            print("  reproduce", json.dumps(out[-1]), flush=True)
    return out


# --------------------------------------------------------------------------- #
# STEP 2 -- build and score the new arms
# --------------------------------------------------------------------------- #
_G: dict = {}          # populated in the PARENT before the fork; children inherit


def _init_child():
    _G["models"] = {}   # model cache is per-process, never inherited


def _models_for(action):
    if action not in _G["models"]:
        _G["models"].clear()          # one action's models resident at a time
        _G["models"][action] = load_models(action, _G["cells"])
    return _G["models"][action]


def work(task):
    """Build and score every new arm for one (action, victim)."""
    action, user, reps = task
    ctx, edges = _G["ctx"], _G["edges"]
    models = _models_for(action)
    shard, fake_rows, gen_rows = B.pools(ctx, user, action)
    five = ctx["five"][(user, action)]
    edge = edges[action]

    acc = np.zeros((NDET, len(BUILT_ARMS), reps, NBINS), np.int64)
    tot = np.zeros((len(BUILT_ARMS), reps, NBINS), np.int64)
    ssum = np.zeros((NDET, len(BUILT_ARMS)), np.float64)
    scnt = np.zeros((NDET, len(BUILT_ARMS)), np.int64)
    rejects = collections.Counter()
    t0 = time.time()

    for rep in range(reps):
        p = B.plan(user, action, rep, five, fake_rows, gen_rows, shard)
        for ai, arm in enumerate(BUILT_ARMS):
            vals, dur = [], []
            for t_row, c_row in p[arm]:
                try:
                    values, traj, imu, audit = B.build_one(ctx, action, shard,
                                                           t_row, c_row)
                except B.BuildRejected as exc:
                    rejects[f"{action}|{arm}|{str(exc)[:70]}"] += 1
                    continue
                vals.append(values)
                dur.append(audit["duration_ms"])
            if not vals:
                continue
            bins = np.searchsorted(edge, np.asarray(dur, float), side="right")
            feats = {}
            for fam, rep_det in FAMILY_REP.items():
                feats[fam] = np.stack([
                    extract_event_features(rep_det, MODALITY, v[:, :6], v[:, 6:])
                    for v in vals]).astype(np.float32)
            np.add.at(tot[ai, rep], bins, 1)
            for di, det in enumerate(CLASSICAL):
                model, cut = models[det]
                sc = classical_scores(model, feats[FEATURE_FAMILY[det]])
                np.add.at(acc[di, ai, rep], bins[sc < cut], 1)
                ssum[di, ai] += float(sc.sum())
                scnt[di, ai] += int(sc.size)
    return {"action": action, "user": user, "acc": acc, "tot": tot,
            "score_sum": ssum, "score_n": scnt, "rejects": dict(rejects),
            "seconds": time.time() - t0}


# --------------------------------------------------------------------------- #
# published rows 1 and 2
# --------------------------------------------------------------------------- #
def published_rows(cells, users, edges, book):
    """Per-user, per-duration-bin accept counts for the genuine and release arms."""
    acc = np.zeros((len(CELLS), 2, len(users), 1, NBINS), np.int64)
    tot = np.zeros((len(ACTIONS), 2, len(users), 1, NBINS), np.int64)
    ssum = np.zeros((len(CELLS), 2), np.float64)
    scnt = np.zeros((len(CELLS), 2), np.int64)
    uidx = {u: i for i, u in enumerate(users)}
    for ai, action in enumerate(ACTIONS):
        bundle = cells[f"{action}__{MODALITY}__{CLASSICAL[0]}"]["bundle"]
        dur = {}
        for u in users:
            s = book.get(bundle, u)
            for i in range(len(s.label)):
                if s.action[i] == action:
                    dur[str(s.event_id[i])] = s.duration_ms(i)
        first = True
        for det in CLASSICAL:
            name = f"{action}__{MODALITY}__{det}"
            c = cells[name]
            cut = float(json.loads(Path(c["thresholds"]).read_text())["frr5"])
            ci_ = CELL_INDEX[name]
            with gzip.open(c["scores"], "rt") as f:
                for line in f:
                    d = json.loads(line)
                    if d["action"] != action:
                        continue
                    eid, u = d["event_id"], d["user_id"]
                    if u not in uidx:
                        continue
                    arm = 0 if d["label"] == 0 else 1
                    b = int(np.searchsorted(edges[action], dur[eid], side="right"))
                    if float(d["fake_high_score"]) < cut:
                        acc[ci_, arm, uidx[u], 0, b] += 1
                    ssum[ci_, arm] += float(d["fake_high_score"])
                    scnt[ci_, arm] += 1
                    if first:
                        tot[ai, arm, uidx[u], 0, b] += 1
            first = False
    return acc, tot, ssum, scnt


# --------------------------------------------------------------------------- #
# statistics -- everything vectorised over the shared bootstrap index
# --------------------------------------------------------------------------- #
class Stats:
    """ACC[cell, arm, user, rep, bin]; TOT[action, arm, user, rep, bin].

    Totals are per ACTION because all four detectors of an action see the
    identical event set -- carrying them per cell would only duplicate them.
    """

    def __init__(self, acc, tot, boot_idx):
        self.A = acc.sum(axis=(3, 4))                      # (cell, arm, user)
        self.T = tot.sum(axis=(3, 4))                      # (action, arm, user)
        self.boot = boot_idx
        Ab = self.A[:, :, boot_idx].sum(axis=3)            # (cell, arm, NBOOT)
        Tb = self.T[:, :, boot_idx].sum(axis=3)            # (action, arm, NBOOT)
        Tb_cell = np.repeat(Tb, NDET, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.far_boot = np.where(Tb_cell > 0, Ab / np.maximum(Tb_cell, 1), np.nan)
            Tp = np.repeat(self.T.sum(axis=2), NDET, axis=0)
            self.far_pt = np.where(Tp > 0, self.A.sum(axis=2) / np.maximum(Tp, 1),
                                   np.nan)

    def group_point(self, cids, ai):
        return float(np.nanmean(self.far_pt[cids, ai]))

    def group_boot(self, cids, ai):
        return np.nanmean(self.far_boot[cids, ai, :], axis=0)

    def cell_point(self, cid, ai):
        return float(self.far_pt[cid, ai])

    def cell_boot(self, cid, ai):
        return self.far_boot[cid, ai, :]


def ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v, dtype=float)
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def standardised(acc, tot, cid, ai, ref_ai):
    """Cell FAR with the duration distribution standardised to arm `ref_ai`."""
    action_i = cid // NDET
    a = acc[cid, ai].sum(axis=(0, 1))
    t = tot[action_i, ai].sum(axis=(0, 1))
    w = tot[action_i, ref_ai].sum(axis=(0, 1)).astype(float)
    ok = (t > 0) & (w > 0)
    if not ok.any():
        return np.nan
    w = w[ok] / w[ok].sum()
    return float((w * (a[ok] / t[ok])).sum())


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--stage", default="all", choices=("reproduce", "score"))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = B.cell_map()
    t0 = time.time()
    ctx = B.context()
    users = ctx["users"]
    print(f"context {time.time()-t0:.1f}s, {len(users)} test users", flush=True)

    before = frozen_checksums(cells)
    (out_dir / "checksums_score_before.json").write_text(json.dumps(before, indent=1))

    # ---- STEP 1: MANDATORY pipeline validation ------------------------------
    rep = reproduce(cells, users, ctx["book"], ACTIONS)
    (out_dir / "reproduce_all20.json").write_text(json.dumps(rep, indent=1))
    worst = max(r["max_abs_diff"] for r in rep)
    print(f"REPRODUCTION worst max_abs_diff over {len(rep)} cells = {worst:.3e}",
          flush=True)
    if worst > 1e-9:
        raise SystemExit("PIPELINE DOES NOT REPRODUCE -- stopping before any arm")
    if args.stage == "reproduce":
        return

    # ---- duration bin edges: the release fake pool's own deciles -------------
    edges = {}
    for action in ACTIONS:
        bundle = cells[f"{action}__{MODALITY}__{CLASSICAL[0]}"]["bundle"]
        d = []
        for u in users:
            s = ctx["book"].get(bundle, u)
            for i in range(len(s.label)):
                if s.action[i] == action and s.label[i] == 1:
                    d.append(s.duration_ms(i))
        edges[action] = np.unique(np.percentile(
            d, np.linspace(0, 100, NBINS + 1)[1:-1])).astype(float)
    print("duration bins", {a: len(edges[a]) + 1 for a in ACTIONS}, flush=True)

    # ---- rows 1 and 2 from the published scores ------------------------------
    p_acc, p_tot, p_ssum, p_scnt = published_rows(cells, users, edges, ctx["book"])
    print("published rows read", flush=True)

    # ---- rows 3/4/5 (+ diagnostic): build and score --------------------------
    ACC = np.zeros((len(CELLS), len(ARMS), len(users), args.reps, NBINS), np.int64)
    TOT = np.zeros((len(ACTIONS), len(ARMS), len(users), args.reps, NBINS), np.int64)
    SSUM = np.zeros((len(CELLS), len(ARMS)), np.float64)
    SCNT = np.zeros((len(CELLS), len(ARMS)), np.int64)
    ACC[:, 0:2, :, 0:1, :] = p_acc
    TOT[:, 0:2, :, 0:1, :] = p_tot
    SSUM[:, 0:2] = p_ssum
    SCNT[:, 0:2] = p_scnt

    _G["ctx"], _G["cells"], _G["edges"] = ctx, cells, edges   # inherited by fork
    tasks = [(a, u, args.reps) for a in ACTIONS for u in users]
    uidx = {u: i for i, u in enumerate(users)}
    rejects = collections.Counter()
    done = 0
    t0 = time.time()
    ctxmp = mp.get_context("fork")
    with ctxmp.Pool(args.workers, initializer=_init_child) as pool:
        for r in pool.imap_unordered(work, tasks, chunksize=1):
            ai_action = ACTIONS.index(r["action"])
            ui = uidx[r["user"]]
            for k, arm in enumerate(BUILT_ARMS):
                gi = ARM_INDEX[arm]
                TOT[ai_action, gi, ui] = r["tot"][k]
                for di, det in enumerate(CLASSICAL):
                    ci_ = CELL_INDEX[f"{r['action']}__{MODALITY}__{det}"]
                    ACC[ci_, gi, ui] = r["acc"][di, k]
                    SSUM[ci_, gi] += r["score_sum"][di, k]
                    SCNT[ci_, gi] += r["score_n"][di, k]
            rejects.update(r["rejects"])
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} tasks, {time.time()-t0:.0f}s",
                      flush=True)
    build_score_seconds = time.time() - t0
    np.savez_compressed(out_dir / "counts.npz", ACC=ACC, TOT=TOT, SSUM=SSUM,
                        SCNT=SCNT, cells=np.array(CELLS), arms=np.array(ARMS),
                        users=np.array(users))

    after = frozen_checksums(cells)
    (out_dir / "checksums_score_after.json").write_text(json.dumps(after, indent=1))

    # ---- statistics ---------------------------------------------------------
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, len(users), size=(NBOOT, len(users)))
    st = Stats(ACC, TOT, boot_idx)

    groups = {"all_20_cells": list(range(len(CELLS))),
              "tap_scroll_swipe_12_cells": [CELL_INDEX[f"{a}__{MODALITY}__{d}"]
                                            for a in ("tap", "scroll", "swipe")
                                            for d in CLASSICAL]}
    for a in ACTIONS:
        groups[f"action:{a}"] = [CELL_INDEX[f"{a}__{MODALITY}__{d}"]
                                 for d in CLASSICAL]
    for d in CLASSICAL:
        groups[f"detector:{d}"] = [CELL_INDEX[f"{a}__{MODALITY}__{d}"]
                                   for a in ACTIONS]

    PAIRS = [("row3", "row2"), ("row4", "row2"), ("row3", "row5"),
             ("row4", "row5"), ("row5", "row2"),
             ("row3", "row5_unmatched"), ("row5", "row5_unmatched"),
             ("row2", "row1"), ("row3", "row1")]

    summary, diffs, boot_cache = {}, {}, {}
    for gname, cids in groups.items():
        entry = {}
        for arm in ARMS:
            gi = ARM_INDEX[arm]
            bv = st.group_boot(cids, gi)
            boot_cache[(gname, arm)] = bv
            entry[arm] = {"far_at_frr5": st.group_point(cids, gi),
                          "ci95": ci(bv), "boot_sd": float(np.std(bv, ddof=1))}
        summary[gname] = entry
        d = {}
        for x, y in PAIRS:
            bv = boot_cache[(gname, x)] - boot_cache[(gname, y)]
            point = st.group_point(cids, ARM_INDEX[x]) - st.group_point(cids,
                                                                        ARM_INDEX[y])
            lo, hi = ci(bv)
            d[f"{x}_minus_{y}"] = {"difference": point, "ci95": [lo, hi],
                                   "excludes_zero": bool(lo > 0 or hi < 0),
                                   "boot_sd": float(np.std(bv, ddof=1))}
        diffs[gname] = d
    print("bootstrap done", flush=True)

    per_cell = {}
    for name in CELLS:
        cid = CELL_INDEX[name]
        e = {"threshold_frr5": float(json.loads(
            Path(cells[name]["thresholds"]).read_text())["frr5"])}
        for arm in ARMS:
            gi = ARM_INDEX[arm]
            e[arm] = {"far_at_frr5": st.cell_point(cid, gi),
                      "ci95": ci(st.cell_boot(cid, gi)),
                      "events": int(TOT[cid // NDET, gi].sum()),
                      "mean_score": (float(SSUM[cid, gi] / SCNT[cid, gi])
                                     if SCNT[cid, gi] else None)}
        e["paired_differences"] = {}
        for x, y in PAIRS:
            bv = st.cell_boot(cid, ARM_INDEX[x]) - st.cell_boot(cid, ARM_INDEX[y])
            lo, hi = ci(bv)
            e["paired_differences"][f"{x}_minus_{y}"] = {
                "difference": st.cell_point(cid, ARM_INDEX[x])
                - st.cell_point(cid, ARM_INDEX[y]),
                "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0)}
        per_cell[name] = e

    # construction spread over the R repetitions (point estimates, no bootstrap)
    spread = {}
    per_rep_stats = [Stats(ACC[:, :, :, r:r + 1, :], TOT[:, :, :, r:r + 1, :],
                           boot_idx[:1]) for r in range(args.reps)]
    for gname, cids in groups.items():
        s = {}
        for arm in BUILT_ARMS:
            gi = ARM_INDEX[arm]
            vals = [ps.group_point(cids, gi) for ps in per_rep_stats]
            s[arm] = {"per_repetition_far": [float(v) for v in vals],
                      "mean": float(np.mean(vals)),
                      "sd": float(np.std(vals, ddof=1)),
                      "min": float(np.min(vals)), "max": float(np.max(vals)),
                      "range": float(np.max(vals) - np.min(vals))}
        spread[gname] = s

    # duration-standardised FAR (reference = the release fake pool's deciles)
    std_tab = {}
    for gname, cids in groups.items():
        e = {}
        for arm in ARMS:
            gi = ARM_INDEX[arm]
            e[arm] = float(np.nanmean([standardised(ACC, TOT, c, gi,
                                                    ARM_INDEX["row2"])
                                       for c in cids]))
        e["_reference_distribution"] = ("row2, the release fake pool, its own "
                                        "duration deciles per action")
        for x, y in [("row3", "row2"), ("row4", "row5"), ("row5", "row2"),
                     ("row3", "row5"), ("row4", "row2")]:
            e[f"{x}_minus_{y}"] = e[x] - e[y]
        std_tab[gname] = e

    out = {
        "reproduce": rep,
        "reproduce_worst_max_abs_diff": worst,
        "duration_bin_edges_ms": {a: edges[a].tolist() for a in ACTIONS},
        "summary": summary,
        "paired_differences": diffs,
        "per_cell": per_cell,
        "construction_spread_over_repetitions": spread,
        "duration_standardised": std_tab,
        "build_rejects": dict(rejects),
        "events_per_arm": {arm: int(TOT[:, ARM_INDEX[arm]].sum()) for arm in ARMS},
        "events_per_arm_per_action": {
            arm: {a: int(TOT[ACTIONS.index(a), ARM_INDEX[arm]].sum())
                  for a in ACTIONS} for arm in ARMS},
        "checksums_before": before, "checksums_after": after,
        "checksums_unchanged": before == after,
        "bootstrap": {"unit": "user", "n_users": len(users), "replicates": NBOOT,
                      "shared_multiset_across_cells": True, "seed": BOOT_SEED},
        "repetitions": args.reps,
        "build_and_score_seconds": build_score_seconds,
        "workers": args.workers,
    }
    (out_dir / "score_result.json").write_text(json.dumps(out, indent=1))
    print("WROTE", out_dir / "score_result.json", flush=True)
    print(json.dumps({"summary_all20": summary["all_20_cells"],
                      "diffs_all20": diffs["all_20_cells"]}, indent=1))


if __name__ == "__main__":
    main()
