#!/usr/bin/env python
"""
sessagg_a.py -- SESSION-LEVEL aggregation of per-event PAD scores.

Question: the attack wins per-event (FAR = 0.775 at the dev-selected FRR=5% cut).
Does it still win when the attacker runs a whole SESSION of gestures?

Everything here is score-space arithmetic over the frozen test_scores files.
No GPU, no model loading, no re-inference.

Agent A of a differential-testing pair. Written for clarity over speed.

Outputs (all under OUT_DIR):
  run.log                     -- full console transcript
  verify.json                 -- schema / accept-rule / roster verification
  sessions.json               -- genuine session inventory
  session_stats_<rel>.json    -- per-cell per-stat session-level summary
  curves_<rel>.json           -- full operating curves (held-out and oracle)
  results_<rel>.json          -- headline numbers + bootstrap CIs
  RESULTS.json                -- everything, both releases
"""

import gzip
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

OUT_DIR = ("/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/"
           "e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/sessagg/A")

RELEASES = {
    # PRIMARY: the repo-published release. This is the ONLY one that reproduces
    # the framing fact of the task, per-event FAR@frr5 = 0.775.
    "repo": ("/home/mwang49/new/data7/data7_final_monitor_metrics_v1/USENIX8.25/"
             "code/dataset_test/results/cells"),
    # SECONDARY: the directory named in the task's DATA section. Verified in
    # reconnaissance to be the r1 pre-fix baseline (per-event FAR@frr5 ~ 0.445).
    "mnt": "/mnt/share/mwang49/data7/results/direct100k/detectors_90cell/cells",
}

ACTIONS = ["tap", "scroll", "swipe", "pinch", "keystroke"]
MODALITIES = ["trajectory_xytime", "imu_only", "imu_trajectory_xytime"]
DETECTORS = ["hmog_style_svm", "hmog_style_rf", "paper_svm",
             "paper_xgboost", "behaveformer_stdat", "authconformer"]

STATS = ["S1_COUNT", "S2_MEAN", "S3_MAX", "S4_TRIMMED", "S5_LOGODDS"]

R_DRAWS = 20                 # mirrored-fake redraws
BASE_SEED = 20260810         # fixed
N_BOOT = 2000                # user-clustered bootstrap replicates
BOOT_SEED = 777
LOGODDS_EPS = 1e-6           # clip scores away from 0 and 1 for S5
TARGET_FRRS = (0.05, 0.01)   # session-level operating points reported with CIs
CAUGHT_TARGETS = (0.50, 0.80, 0.95)   # inverse reading: price of detection

# q-grid used to trace the cross-fitted (held-out) session operating curve.
Q_GRID = np.unique(np.concatenate([
    np.array([0.0]),
    np.geomspace(0.001, 0.05, 25),
    np.linspace(0.05, 1.0, 191),
]))

# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------

_LOG_FH = None


def log(msg=""):
    print(msg, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(str(msg) + "\n")
        _LOG_FH.flush()


def dump(obj, name):
    p = os.path.join(OUT_DIR, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_jsonify)
    log("  wrote %s (%.1f KB)" % (p, os.path.getsize(p) / 1024.0))
    return p


def _jsonify(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(type(o)))


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def cell_dir(root, action, modality, detector):
    return os.path.join(root, "%s__%s__%s" % (action, modality, detector))


def read_scores(cdir):
    """Return list of dict rows from test_scores.jsonl[.gz]."""
    p_gz = os.path.join(cdir, "test_scores.jsonl.gz")
    p_pl = os.path.join(cdir, "test_scores.jsonl")
    rows = []
    if os.path.exists(p_gz):
        fh = gzip.open(p_gz, "rt")
    else:
        fh = open(p_pl, "rt")
    with fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(cdir, name):
    with open(os.path.join(cdir, name)) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# STEP 0 -- verification of the conventions we rely on
# --------------------------------------------------------------------------

def verify_conventions(root, tag):
    """Re-verify threshold field names, score direction, accept rule."""
    log("\n[verify:%s] re-verifying threshold schema + accept rule over 90 cells" % tag)
    keysets = Counter()
    dirs = Counter()
    tfrr = Counter()
    split = Counter()
    frr5_lt_eer = 0
    far_dev_eer_max_abs_err = 0.0
    frr_dev_eer_max_abs_err = 0.0
    far_frr5 = []
    frr_frr5 = []
    sd_in_frozen = 0
    sd_in_summary = 0
    per_cell = {}

    for a in ACTIONS:
        for m in MODALITIES:
            for d in DETECTORS:
                cdir = cell_dir(root, a, m, d)
                th = read_json(cdir, "thresholds.json")
                fc = read_json(cdir, "frozen_config.json")
                sm = read_json(cdir, "summary.json")
                keysets[tuple(sorted(th))] += 1
                dirs[th["score_direction"]] += 1
                tfrr[th["target_frr"]] += 1
                split[th["selection_split"]] += 1
                if th["frr5"] < th["eer"]:
                    frr5_lt_eer += 1
                if "score_direction" in fc:
                    sd_in_frozen += 1
                if "score_direction" in sm:
                    sd_in_summary += 1

                rows = read_scores(cdir)
                s = np.array([r["fake_high_score"] for r in rows], dtype=np.float64)
                y = np.array([r["label"] for r in rows], dtype=np.int64)
                fake = y == 1
                real = y == 0
                # project accept rule: ACCEPTED <=> score < threshold
                far_eer = float(np.mean(s[fake] < th["eer"]))
                frr_eer = float(np.mean(s[real] >= th["eer"]))
                pm = sm["primary_metrics"]
                far_dev_eer_max_abs_err = max(far_dev_eer_max_abs_err,
                                              abs(far_eer - pm["far"]))
                frr_dev_eer_max_abs_err = max(frr_dev_eer_max_abs_err,
                                              abs(frr_eer - pm["frr"]))
                f5 = float(np.mean(s[fake] < th["frr5"]))
                r5 = float(np.mean(s[real] >= th["frr5"]))
                far_frr5.append(f5)
                frr_frr5.append(r5)
                per_cell["%s__%s__%s" % (a, m, d)] = {
                    "frr5": th["frr5"], "eer": th["eer"],
                    "far_at_frr5": f5, "frr_at_frr5": r5,
                    "far_at_eer": far_eer, "summary_far": pm["far"],
                    "n_genuine": int(real.sum()), "n_fake": int(fake.sum()),
                    "score_min": float(s.min()), "score_max": float(s.max()),
                    # S5 LOGODDS is only well defined on (0,1); record how much
                    # of each cell gets destroyed by the clip.
                    "frac_scores_outside_open01":
                        float(np.mean((s <= 0.0) | (s >= 1.0))),
                }

    out = {
        "release": tag,
        "threshold_keysets": {str(k): v for k, v in keysets.items()},
        "score_direction_counts": dict(dirs),
        "target_frr_counts": {str(k): v for k, v in tfrr.items()},
        "selection_split_counts": dict(split),
        "cells_with_frr5_below_eer": frr5_lt_eer,
        "score_direction_present_in_frozen_config": sd_in_frozen,
        "score_direction_present_in_summary": sd_in_summary,
        "accept_rule_used": "ACCEPTED <=> score < threshold ; CAUGHT <=> score >= threshold",
        "max_abs_err_recomputed_far_at_dev_eer_vs_summary_far": far_dev_eer_max_abs_err,
        "max_abs_err_recomputed_frr_at_dev_eer_vs_summary_frr": frr_dev_eer_max_abs_err,
        "event_far_at_frr5_mean_over_90_cells": float(np.mean(far_frr5)),
        "event_far_at_frr5_median_over_90_cells": float(np.median(far_frr5)),
        "event_frr_at_frr5_mean_over_90_cells": float(np.mean(frr_frr5)),
        "per_cell": per_cell,
    }
    log("  threshold key sets: %d distinct" % len(keysets))
    log("  score_direction: %s ; target_frr: %s ; selection_split: %s"
        % (dict(dirs), dict(tfrr), dict(split)))
    log("  score_direction in frozen_config: %d/90 ; in summary: %d/90"
        % (sd_in_frozen, sd_in_summary))
    log("  cells with frr5 < eer: %d/90" % frr5_lt_eer)
    log("  ACCEPT RULE CHECK  max|recomputed FAR@eer - summary.far| = %.3e"
        % far_dev_eer_max_abs_err)
    log("                     max|recomputed FRR@eer - summary.frr| = %.3e"
        % frr_dev_eer_max_abs_err)
    log("  EVENT-LEVEL FAR@frr5 : mean %.6f  median %.6f" %
        (out["event_far_at_frr5_mean_over_90_cells"],
         out["event_far_at_frr5_median_over_90_cells"]))
    log("  EVENT-LEVEL FRR@frr5 : mean %.6f" % out["event_frr_at_frr5_mean_over_90_cells"])
    clipped = {}
    for k, v in per_cell.items():
        det = k.split("__")[2]
        clipped.setdefault(det, []).append(v["frac_scores_outside_open01"])
    log("  S5 clip damage -- fraction of events with score <=0 or >=1, by detector:")
    for det in DETECTORS:
        v = clipped[det]
        log("    %-20s mean %.4f  max %.4f" % (det, float(np.mean(v)), float(np.max(v))))
    out["s5_clip_fraction_by_detector"] = {d: {"mean": float(np.mean(clipped[d])),
                                               "max": float(np.max(clipped[d]))}
                                           for d in DETECTORS}
    return out


# --------------------------------------------------------------------------
# STEP 1 -- assemble the per-(modality,detector) tables
# --------------------------------------------------------------------------

class Roster:
    """The event roster, identical across all 18 (modality, detector) cells."""

    def __init__(self):
        # genuine
        self.gen_ids = {}        # action -> list of event_id (canonical order)
        self.gen_index = {}      # action -> {event_id: pos}
        self.gen_user = {}       # action -> list user
        self.gen_sess = {}       # action -> list session_id
        # fake
        self.fake_ids = {}       # (action,user) -> list of event_id (canonical order)
        self.fake_index = {}     # action -> {event_id: (user, pos)}


def build_roster(root):
    """Build canonical rosters from the trajectory_xytime/hmog_style_svm cells,
    then VERIFY that every other cell of the same action has the identical
    event-id -> (user, session, label) mapping."""
    log("\n[roster] building canonical event roster")
    ros = Roster()
    ref = {}
    for a in ACTIONS:
        cdir = cell_dir(root, a, MODALITIES[0], DETECTORS[0])
        rows = read_scores(cdir)
        g = sorted([r for r in rows if r["label"] == 0], key=lambda r: r["event_id"])
        f = sorted([r for r in rows if r["label"] == 1], key=lambda r: r["event_id"])
        ros.gen_ids[a] = [r["event_id"] for r in g]
        ros.gen_index[a] = {e: i for i, e in enumerate(ros.gen_ids[a])}
        ros.gen_user[a] = [r["user_id"] for r in g]
        ros.gen_sess[a] = [r["session_id"] for r in g]
        by_user = defaultdict(list)
        for r in f:
            by_user[r["user_id"]].append(r["event_id"])
        ros.fake_index[a] = {}
        for u, ids in by_user.items():
            ids = sorted(ids)
            ros.fake_ids[(a, u)] = ids
            for i, e in enumerate(ids):
                ros.fake_index[a][e] = (u, i)
        ref[a] = {r["event_id"]: (r["user_id"], r["session_id"], r["label"]) for r in rows}
        log("  %-10s genuine %5d  fake %5d  fake pools %s"
            % (a, len(g), len(f),
               sorted(set(len(v) for k, v in ros.fake_ids.items() if k[0] == a))))

    mismatches = 0
    for a in ACTIONS:
        for m in MODALITIES:
            for d in DETECTORS:
                cdir = cell_dir(root, a, m, d)
                rows = read_scores(cdir)
                cur = {r["event_id"]: (r["user_id"], r["session_id"], r["label"])
                       for r in rows}
                if cur != ref[a]:
                    mismatches += 1
                    log("  ROSTER MISMATCH %s %s %s" % (a, m, d))
    log("  roster identical across all 18 cells for every action: %s"
        % (mismatches == 0))
    return ros, mismatches


def load_cell_scores(root, modality, detector, ros):
    """For one (modality, detector): return per-action genuine score vectors,
    per-(action,user) fake score vectors, and the per-action frr5 threshold."""
    gen = {}
    fake = {}
    thr = {}
    for a in ACTIONS:
        cdir = cell_dir(root, a, modality, detector)
        th = read_json(cdir, "thresholds.json")
        thr[a] = float(th["frr5"])
        rows = read_scores(cdir)
        gvec = np.full(len(ros.gen_ids[a]), np.nan)
        fvecs = {u: np.full(len(ros.fake_ids[(a, u)]), np.nan)
                 for (aa, u) in ros.fake_ids if aa == a}
        for r in rows:
            eid = r["event_id"]
            if r["label"] == 0:
                gvec[ros.gen_index[a][eid]] = r["fake_high_score"]
            else:
                u, i = ros.fake_index[a][eid]
                fvecs[u][i] = r["fake_high_score"]
        assert not np.isnan(gvec).any()
        for u in fvecs:
            assert not np.isnan(fvecs[u]).any()
        gen[a] = gvec
        fake[a] = fvecs
    return gen, fake, thr


# --------------------------------------------------------------------------
# STEP 2 -- genuine sessions
# --------------------------------------------------------------------------

def build_sessions(ros):
    """Group genuine events by real session_id.
    Returns list of dicts in a fixed (sorted by session_id) order."""
    log("\n[sessions] grouping genuine events by real session_id")
    bysess = defaultdict(lambda: {"user": None, "events": []})
    for a in ACTIONS:
        for i, sid in enumerate(ros.gen_sess[a]):
            rec = bysess[sid]
            u = ros.gen_user[a][i]
            if rec["user"] is None:
                rec["user"] = u
            assert rec["user"] == u, "session %s spans users" % sid
            rec["events"].append((a, i))
    sessions = []
    for sid in sorted(bysess):
        rec = bysess[sid]
        # canonical intra-session ordering: action order, then roster position
        ev = sorted(rec["events"], key=lambda t: (ACTIONS.index(t[0]), t[1]))
        counts = Counter(a for a, _ in ev)
        sessions.append({
            "session_id": sid,
            "user": rec["user"],
            "n": len(ev),
            "events": ev,
            "action_counts": {a: counts.get(a, 0) for a in ACTIONS},
        })
    ns = np.array([s["n"] for s in sessions])
    log("  %d genuine sessions, %d genuine events, %d users"
        % (len(sessions), int(ns.sum()), len(set(s["user"] for s in sessions))))
    log("  session length n: min %d p25 %g median %g mean %.2f p75 %g p95 %.2f max %d"
        % (ns.min(), np.percentile(ns, 25), np.median(ns), ns.mean(),
           np.percentile(ns, 75), np.percentile(ns, 95), ns.max()))
    nact = Counter(sum(1 for a in ACTIONS if s["action_counts"][a] > 0) for s in sessions)
    log("  distinct actions per session: %s" % dict(sorted(nact.items())))
    return sessions


def check_shortfalls(sessions, ros):
    """Per (user, action): genuine slots required vs fake pool available,
    both cumulatively (all sessions of the user) and per single session."""
    need = Counter()
    pool = {}
    worst_single = Counter()
    for s in sessions:
        for a in ACTIONS:
            c = s["action_counts"][a]
            if c:
                need[(s["user"], a)] += c
                worst_single[(s["user"], a)] = max(worst_single[(s["user"], a)], c)
    for (a, u), ids in ros.fake_ids.items():
        pool[(u, a)] = len(ids)
    rows = []
    n_short_cum = 0
    n_short_single = 0
    for (u, a), nn in sorted(need.items()):
        av = pool.get((u, a), 0)
        rows.append({"user": u, "action": a, "gen_slots": nn, "fake_pool": av,
                     "max_single_session_demand": worst_single[(u, a)],
                     "cumulative_shortfall": max(0, nn - av),
                     "single_session_shortfall": max(0, worst_single[(u, a)] - av)})
        if nn > av:
            n_short_cum += 1
        if worst_single[(u, a)] > av:
            n_short_single += 1
    log("  (user,action) pairs used: %d ; cumulative shortfalls: %d ; "
        "single-session shortfalls: %d" % (len(rows), n_short_cum, n_short_single))
    tightest = max(rows, key=lambda r: r["max_single_session_demand"] / r["fake_pool"])
    log("  tightest SINGLE-session demand: %s %s -> %d of pool %d (ratio %.3f)"
        % (tightest["user"], tightest["action"],
           tightest["max_single_session_demand"], tightest["fake_pool"],
           tightest["max_single_session_demand"] / tightest["fake_pool"]))
    return rows, n_short_cum, n_short_single


# --------------------------------------------------------------------------
# STEP 3 -- mirrored fake sessions
# --------------------------------------------------------------------------

def draw_fake_sessions(sessions, ros, seed):
    """One mirrored draw. For each genuine session (user u, action multiset A),
    build a fake session for the SAME user with the SAME action multiset,
    drawing fake events of each action WITHOUT REPLACEMENT within the session.
    Draws are independent across sessions (i.e. with replacement across sessions).

    Returns list (parallel to `sessions`) of lists of (action, pool_pos).
    """
    rng = np.random.default_rng(seed)
    out = []
    for s in sessions:                       # sessions are in fixed sorted order
        u = s["user"]
        picks = []
        for a in ACTIONS:                    # fixed action order
            c = s["action_counts"][a]
            if c == 0:
                continue
            pool_n = len(ros.fake_ids[(a, u)])
            if c > pool_n:
                raise RuntimeError("shortfall %s %s %d>%d" % (u, a, c, pool_n))
            idx = rng.choice(pool_n, size=c, replace=False)
            for j in sorted(idx.tolist()):
                picks.append((a, j))
        out.append(picks)
    return out


# --------------------------------------------------------------------------
# STEP 4 -- session statistics
# --------------------------------------------------------------------------

def session_stats(scores, thr_vec):
    """scores : 1-D float array of per-event scores of ONE session
       thr_vec: 1-D float array, each event's OWN action's dev FRR=5% threshold
       Returns the 5 scalars in STATS order. All are increasing in fakeness."""
    n = scores.shape[0]
    s1 = float(np.count_nonzero(scores >= thr_vec))          # COUNT
    s2 = float(scores.mean())                                # MEAN
    s3 = float(scores.max())                                 # MAX
    k = int(math.ceil(n / 3.0))
    top = np.sort(scores)[::-1][:k]
    s4 = float(top.mean())                                   # TRIMMED
    c = np.clip(scores, LOGODDS_EPS, 1.0 - LOGODDS_EPS)      # LOGODDS
    s5 = float(np.sum(np.log(c / (1.0 - c))))
    return (s1, s2, s3, s4, s5)


def compute_all_session_stats(sessions, fake_draws, gen, fake, thr):
    """For one (modality, detector) cell.
    Returns gen_stats (S,n_sess) and fake_stats (S,R,n_sess)."""
    S = len(STATS)
    ns = len(sessions)
    R = len(fake_draws)
    gs = np.zeros((S, ns))
    fs = np.zeros((S, R, ns))
    for si, s in enumerate(sessions):
        ev = s["events"]
        sc = np.array([gen[a][i] for a, i in ev])
        tv = np.array([thr[a] for a, _ in ev])
        gs[:, si] = session_stats(sc, tv)
    for r in range(R):
        draws = fake_draws[r]
        for si in range(ns):
            picks = draws[si]
            sc = np.array([fake[a][sessions[si]["user"]][j] for a, j in picks])
            tv = np.array([thr[a] for a, _ in picks])
            fs[:, r, si] = session_stats(sc, tv)
    return gs, fs


# --------------------------------------------------------------------------
# STEP 5 -- session threshold selection and evaluation
# --------------------------------------------------------------------------

def pick_threshold_sorted(gsorted, q):
    """SESSION-level threshold rule, calibrated on genuine sessions only.

    t* = the smallest threshold whose calibration session-FRR = #{g >= t}/m is
    still <= q.  With k = floor(q*m) genuine sessions allowed at-or-above the
    cut, the infimum of admissible thresholds is just above the (k+1)-th
    largest calibration genuine statistic, so

        t* = nextafter( g_[(k+1)-th largest] , +inf )

    This is exactly the project's own event-level rule
    (event_pad.py::select_development_thresholds: eligible = {FRR <= target},
    take the FAR-minimising one, and FAR is monotone in t so that is the
    smallest eligible candidate on a grid of unique observed scores). The two
    give IDENTICAL catch rates: the smallest pooled-grid candidate strictly
    above g_[(k+1)] is <= every fake value strictly above g_[(k+1)], so both
    rules catch exactly the fake sessions with stat > g_[(k+1)]. The nextafter
    form is used because it needs no fake scores at calibration time.

    k >= m  -> t = -inf  (flag everything, FA = 1)
    k == 0  -> t just above the calibration genuine maximum
    """
    m = gsorted.shape[0]
    if m == 0:
        return np.inf
    k = int(math.floor(q * m + 1e-9))
    if k >= m:
        return -np.inf
    v = gsorted[m - k - 1]                     # (k+1)-th largest
    return float(np.nextafter(v, np.inf))


def pick_threshold(gcal, q):
    return pick_threshold_sorted(np.sort(gcal), q)


def evaluate(gev, fev, t):
    """gev: (n_gen,) genuine session stats; fev: (R,n_fake) fake session stats."""
    fa = float(np.mean(gev >= t)) if gev.size else float("nan")
    ct = float(np.mean(fev >= t)) if fev.size else float("nan")
    return fa, ct


def crossfit_operating_point(gs, fs, fold_of_sess, q, cache=None):
    """User-disjoint 2-fold cross-fitting, pooled.
    Calibrate on fold 0 -> evaluate fold 1, and vice versa; pool the counts."""
    if cache is None:
        cache = {}
        for f in (0, 1):
            sel = fold_of_sess == f
            cache[f] = (np.sort(gs[sel]), gs[sel], fs[:, sel])
    fa_num = 0.0
    fa_den = 0.0
    ct_num = 0.0
    ct_den = 0.0
    ts = []
    for f_cal, f_ev in ((0, 1), (1, 0)):
        t = pick_threshold_sorted(cache[f_cal][0], q)
        ts.append(t)
        gev, fev = cache[f_ev][1], cache[f_ev][2]
        fa_num += float(np.count_nonzero(gev >= t))
        fa_den += gev.size
        ct_num += float(np.count_nonzero(fev >= t))
        ct_den += fev.size
    return fa_num / fa_den, ct_num / ct_den, ts


def oracle_operating_point(gs, fs, q, gsorted=None):
    t = pick_threshold_sorted(np.sort(gs) if gsorted is None else gsorted, q)
    return evaluate(gs, fs, t) + (t,)


def roc_auc(gvals, fvals):
    """AUC with fake as the positive class (Mann-Whitney, ties = 0.5)."""
    ng = gvals.size
    nf = fvals.size
    allv = np.concatenate([gvals, fvals])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rf = ranks[ng:].sum()
    return float((rf - nf * (nf + 1) / 2.0) / (ng * nf))


# --------------------------------------------------------------------------
# MAIN per-release pipeline
# --------------------------------------------------------------------------

def run_release(tag, root, sessions_ref=None):
    log("\n" + "=" * 78)
    log("RELEASE %s  root=%s" % (tag, root))
    log("=" * 78)

    ver = verify_conventions(root, tag)
    ros, mism = build_roster(root)
    sessions = build_sessions(ros)
    short_rows, n_sc, n_ss = check_shortfalls(sessions, ros)

    users = sorted(set(s["user"] for s in sessions))
    assert len(users) == 20, users
    uidx = {u: i for i, u in enumerate(users)}
    sess_user = np.array([uidx[s["user"]] for s in sessions])

    # FIXED, DOCUMENTED user split: user ids sorted ascending, alternating.
    #   fold 0 = users at even positions (0,2,...,18)
    #   fold 1 = users at odd  positions (1,3,...,19)
    fold_of_user = np.array([i % 2 for i in range(20)])
    fold_of_sess = fold_of_user[sess_user]
    log("\n[split] fixed user-disjoint halves (sorted user ids, alternating):")
    log("  fold0 = %s" % [users[i] for i in range(20) if fold_of_user[i] == 0])
    log("  fold1 = %s" % [users[i] for i in range(20) if fold_of_user[i] == 1])
    log("  sessions per fold: %d / %d"
        % (int((fold_of_sess == 0).sum()), int((fold_of_sess == 1).sum())))

    # mirrored fake draws, shared across all 18 cells (same synthetic sessions
    # judged by every detector -- keeps the 18 cells comparable)
    log("\n[draws] building %d mirrored fake-session draws (base seed %d)"
        % (R_DRAWS, BASE_SEED))
    t0 = time.time()
    fake_draws = [draw_fake_sessions(sessions, ros, BASE_SEED + 1000 * r)
                  for r in range(R_DRAWS)]
    log("  done in %.1f s" % (time.time() - t0))

    # sanity: composition identical
    for r in range(R_DRAWS):
        for si, s in enumerate(sessions):
            c = Counter(a for a, _ in fake_draws[r][si])
            assert all(c.get(a, 0) == s["action_counts"][a] for a in ACTIONS)
            assert len(fake_draws[r][si]) == s["n"]
            assert len(set(fake_draws[r][si])) == s["n"]   # no repeats in session
    log("  composition check passed: fake sessions match genuine length + "
        "action multiset exactly, no within-session repeats")

    # ----------------------------------------------------------------- cells
    combos = [(m, d) for m in MODALITIES for d in DETECTORS]
    G = {}      # (m,d) -> (S, n_sess)
    F = {}      # (m,d) -> (S, R, n_sess)
    thr_all = {}
    log("\n[cells] computing session statistics for 18 (modality,detector) combos")
    for (m, d) in combos:
        t0 = time.time()
        gen, fake, thr = load_cell_scores(root, m, d, ros)
        gs, fs = compute_all_session_stats(sessions, fake_draws, gen, fake, thr)
        G[(m, d)] = gs
        F[(m, d)] = fs
        thr_all["%s|%s" % (m, d)] = thr
        log("  %-22s %-19s  %.1f s" % (m, d, time.time() - t0))

    # ------------------------------------------------- headline point results
    log("\n[eval] operating points, user-disjoint (primary) and oracle")
    results = {}
    for (m, d) in combos:
        key = "%s|%s" % (m, d)
        results[key] = {}
        gs = G[(m, d)]
        fs = F[(m, d)]
        for si, st in enumerate(STATS):
            e = {}
            for q in TARGET_FRRS:
                fa, ct, ts = crossfit_operating_point(gs[si], fs[si], fold_of_sess, q)
                # per-draw spread of the caught rate
                per_draw = []
                for r in range(fs.shape[1]):
                    n_, dn_ = 0.0, 0
                    for f_cal, f_ev in ((0, 1), (1, 0)):
                        t = ts[0] if f_cal == 0 else ts[1]
                        evm = fold_of_sess == f_ev
                        n_ += float(np.count_nonzero(fs[si, r, evm] >= t))
                        dn_ += int(np.count_nonzero(evm))
                    per_draw.append(n_ / dn_)
                per_draw = np.array(per_draw)
                ofa, oct_, ot = oracle_operating_point(gs[si], fs[si], q)
                e["q%g" % q] = {
                    "xval_session_fa": fa, "xval_caught": ct,
                    "xval_thresholds": [None if not np.isfinite(x) else x for x in ts],
                    "xval_caught_per_draw_mean": float(per_draw.mean()),
                    "xval_caught_per_draw_std": float(per_draw.std(ddof=1)),
                    "xval_caught_per_draw_min": float(per_draw.min()),
                    "xval_caught_per_draw_max": float(per_draw.max()),
                    "oracle_session_fa": ofa, "oracle_caught": oct_,
                    "oracle_threshold": None if not np.isfinite(ot) else ot,
                    "gap_oracle_minus_xval": oct_ - ct,
                }
            # AUC per draw
            aucs = [roc_auc(gs[si], fs[si, r]) for r in range(fs.shape[1])]
            e["auc_mean"] = float(np.mean(aucs))
            e["auc_std"] = float(np.std(aucs, ddof=1))
            e["auc_min"] = float(np.min(aucs))
            e["auc_max"] = float(np.max(aucs))
            results[key][st] = e

    # ------------------------------------------------------- operating curves
    log("[curves] tracing cross-fitted and oracle operating curves over q-grid")
    curves = {}
    for (m, d) in combos:
        key = "%s|%s" % (m, d)
        curves[key] = {}
        gs = G[(m, d)]
        fs = F[(m, d)]
        for si, st in enumerate(STATS):
            cache = {}
            for f in (0, 1):
                sel = fold_of_sess == f
                cache[f] = (np.sort(gs[si][sel]), gs[si][sel], fs[si][:, sel])
            gall = np.sort(gs[si])
            xf_fa, xf_ct, or_fa, or_ct = [], [], [], []
            for q in Q_GRID:
                fa, ct, _ = crossfit_operating_point(gs[si], fs[si], fold_of_sess,
                                                     q, cache=cache)
                xf_fa.append(fa)
                xf_ct.append(ct)
                ofa, oct_, _ = oracle_operating_point(gs[si], fs[si], q, gsorted=gall)
                or_fa.append(ofa)
                or_ct.append(oct_)
            curves[key][st] = {
                "q_grid": Q_GRID.tolist(),
                "xval_fa": xf_fa, "xval_caught": xf_ct,
                "oracle_fa": or_fa, "oracle_caught": or_ct,
            }

    # ------------------------------------------------------- inverse readings
    def price_of_detection(fa_list, ct_list, target):
        """Smallest session false-alarm rate at which caught >= target."""
        best = None
        for fa, ct in zip(fa_list, ct_list):
            if ct >= target and (best is None or fa < best):
                best = fa
        return best   # None == unreachable even at FA=1

    # aggregate curve = macro-mean over the 18 combos at each q
    agg_curves = {}
    for st in STATS:
        xf_fa = np.mean([curves["%s|%s" % (m, d)][st]["xval_fa"] for (m, d) in combos], axis=0)
        xf_ct = np.mean([curves["%s|%s" % (m, d)][st]["xval_caught"] for (m, d) in combos], axis=0)
        or_fa = np.mean([curves["%s|%s" % (m, d)][st]["oracle_fa"] for (m, d) in combos], axis=0)
        or_ct = np.mean([curves["%s|%s" % (m, d)][st]["oracle_caught"] for (m, d) in combos], axis=0)
        agg_curves[st] = {"q_grid": Q_GRID.tolist(),
                          "xval_fa": xf_fa.tolist(), "xval_caught": xf_ct.tolist(),
                          "oracle_fa": or_fa.tolist(), "oracle_caught": or_ct.tolist(),
                          "price_xval": {("caught%g" % c):
                                         price_of_detection(xf_fa, xf_ct, c)
                                         for c in CAUGHT_TARGETS},
                          "price_oracle": {("caught%g" % c):
                                           price_of_detection(or_fa, or_ct, c)
                                           for c in CAUGHT_TARGETS}}
    # per-cell price
    for key in curves:
        for st in STATS:
            c = curves[key][st]
            c["price_xval"] = {("caught%g" % t):
                               price_of_detection(c["xval_fa"], c["xval_caught"], t)
                               for t in CAUGHT_TARGETS}
            c["price_oracle"] = {("caught%g" % t):
                                 price_of_detection(c["oracle_fa"], c["oracle_caught"], t)
                                 for t in CAUGHT_TARGETS}

    # ------------------------------------------------- S1 integer count rule
    log("[countrule] explicit integer-k table for S1 COUNT")
    s1 = STATS.index("S1_COUNT")
    kmax = int(max(max(G[c][s1].max() for c in combos),
                   max(F[c][s1].max() for c in combos)))
    count_rule = {"k_max_observed": kmax, "per_k": {}, "per_cell_best_k": {}}
    for k in range(0, kmax + 2):
        fas, cts = [], []
        for (m, d) in combos:
            fas.append(float(np.mean(G[(m, d)][s1] >= k)))
            cts.append(float(np.mean(F[(m, d)][s1] >= k)))
        count_rule["per_k"][str(k)] = {
            "macro_session_fa": float(np.mean(fas)),
            "macro_caught": float(np.mean(cts)),
            "fa_min": float(np.min(fas)), "fa_max": float(np.max(fas)),
            "caught_min": float(np.min(cts)), "caught_max": float(np.max(cts)),
        }
    # best integer k under the user-disjoint calibration at FRR<=5%
    for (m, d) in combos:
        key = "%s|%s" % (m, d)
        ks = []
        for f_cal in (0, 1):
            t = pick_threshold(G[(m, d)][s1][fold_of_sess == f_cal], 0.05)
            ks.append(None if not np.isfinite(t) else int(math.ceil(t)))
        count_rule["per_cell_best_k"][key] = {
            "k_calibrated_on_fold0": ks[0], "k_calibrated_on_fold1": ks[1],
            "k_oracle_all20": (lambda t: None if not np.isfinite(t) else int(math.ceil(t)))(
                pick_threshold(G[(m, d)][s1], 0.05)),
        }

    # ------------------------------------------------------- regime breakout
    # keystroke never co-occurs with scroll or pinch: sessions come in two
    # regimes. Report the held-out caught rate separately for each.
    log("[regime] caught-rate by session regime (q=0.05, user-disjoint)")
    ks_only = np.array([s["action_counts"]["keystroke"] == s["n"] for s in sessions])
    regime = {}
    for si, st in enumerate(STATS):
        for name, mask in (("keystroke_only", ks_only), ("touch_or_mixed", ~ks_only)):
            # threshold is per-cell and global (calibrated on the whole
            # calibration fold); only the EVALUATION set is split by regime.
            accc = []
            accf = []
            for (m, d) in combos:
                gs = G[(m, d)][si]
                fs = F[(m, d)][si]
                cn = cd = 0
                fn = fd = 0
                for f_cal, f_ev in ((0, 1), (1, 0)):
                    t = pick_threshold(gs[fold_of_sess == f_cal], 0.05)
                    sel = (fold_of_sess == f_ev) & mask
                    cn += int(np.count_nonzero(fs[:, sel] >= t))
                    cd += int(fs.shape[0] * np.count_nonzero(sel))
                    fn += int(np.count_nonzero(gs[sel] >= t))
                    fd += int(np.count_nonzero(sel))
                accc.append(cn / cd)
                accf.append(fn / fd)
            regime["%s|%s" % (st, name)] = {
                "n_sessions": int(mask.sum()),
                "macro_caught": float(np.mean(accc)),
                "macro_session_fa": float(np.mean(accf)),
            }
    for st in STATS:
        a = regime["%s|keystroke_only" % st]
        b = regime["%s|touch_or_mixed" % st]
        log("  %-12s keystroke-only(n=%d) caught %.4f FA %.4f | "
            "touch/mixed(n=%d) caught %.4f FA %.4f"
            % (st, a["n_sessions"], a["macro_caught"], a["macro_session_fa"],
               b["n_sessions"], b["macro_caught"], b["macro_session_fa"]))

    # --------------------------------------------------------- bootstrap CIs
    log("[bootstrap] user-clustered bootstrap, %d replicates "
        "(stratified 10+10 within the fixed folds; one user multiset shared "
        "across all 18 cells per replicate)" % N_BOOT)
    boot = run_bootstrap(G, F, sessions, sess_user, users, fold_of_user,
                         fold_of_sess, combos)

    out = {
        "release": tag, "root": root,
        "verify": ver,
        "roster_mismatches": mism,
        "n_sessions": len(sessions), "n_users": len(users),
        "users": users,
        "fold0": [users[i] for i in range(20) if fold_of_user[i] == 0],
        "fold1": [users[i] for i in range(20) if fold_of_user[i] == 1],
        "shortfalls": {"rows": short_rows, "n_cumulative": n_sc, "n_single_session": n_ss},
        "thresholds_frr5": thr_all,
        "per_cell": results,
        "curves": curves,
        "aggregate_curves": agg_curves,
        "count_rule": count_rule,
        "regime": regime,
        "bootstrap": boot,
        "config": {"R_DRAWS": R_DRAWS, "BASE_SEED": BASE_SEED, "N_BOOT": N_BOOT,
                   "BOOT_SEED": BOOT_SEED, "LOGODDS_EPS": LOGODDS_EPS,
                   "target_frrs": list(TARGET_FRRS),
                   "caught_targets": list(CAUGHT_TARGETS)},
    }
    return out, sessions, G, F, combos


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------

def group_key(m, d):
    return "%s|%s" % (m, d)


def run_bootstrap(G, F, sessions, sess_user, users, fold_of_user, fold_of_sess,
                  combos):
    """User-clustered bootstrap. One user multiset per replicate, applied to
    every cell so aggregates are consistent. Stratified 10 + 10 inside the two
    fixed folds so the cross-fitting design survives resampling (an unstratified
    draw can empty a fold)."""
    rng = np.random.default_rng(BOOT_SEED)
    sess_of_user = [np.where(sess_user == i)[0] for i in range(20)]
    fold0_users = [i for i in range(20) if fold_of_user[i] == 0]
    fold1_users = [i for i in range(20) if fold_of_user[i] == 1]

    # accumulators: mode -> q -> agg/breakout -> list over replicates
    acc = defaultdict(lambda: defaultdict(list))

    t0 = time.time()
    for b in range(N_BOOT):
        u0 = rng.choice(fold0_users, size=10, replace=True)
        u1 = rng.choice(fold1_users, size=10, replace=True)
        idx0 = np.concatenate([sess_of_user[u] for u in u0])
        idx1 = np.concatenate([sess_of_user[u] for u in u1])
        idx_all = np.concatenate([idx0, idx1])

        for si, st in enumerate(STATS):
            percell = {}
            for (m, d) in combos:
                gs = G[(m, d)][si]
                fs = F[(m, d)][si]
                g0 = gs[idx0]
                g1 = gs[idx1]
                ga = np.concatenate([g0, g1])
                g0s = np.sort(g0)
                g1s = np.sort(g1)
                gas = np.sort(ga)
                f0 = fs[:, idx0]
                f1 = fs[:, idx1]
                nsess = idx0.size + idx1.size
                nfake = f0.size + f1.size
                cell = {}
                for q in TARGET_FRRS:
                    # ---- user-disjoint cross-fit
                    t01 = pick_threshold_sorted(g0s, q)   # calib fold0 -> eval fold1
                    t10 = pick_threshold_sorted(g1s, q)   # calib fold1 -> eval fold0
                    ctn = (np.count_nonzero(f1 >= t01)
                           + np.count_nonzero(f0 >= t10))
                    fan = (np.count_nonzero(g1 >= t01) + np.count_nonzero(g0 >= t10))
                    cell[("xval", q)] = (fan / nsess, ctn / nfake)
                    # ---- oracle
                    to = pick_threshold_sorted(gas, q)
                    cell[("oracle", q)] = (
                        float(np.count_nonzero(ga >= to)) / nsess,
                        float(np.count_nonzero(f0 >= to)
                              + np.count_nonzero(f1 >= to)) / nfake)
                percell[(m, d)] = cell
            for mode in ("xval", "oracle"):
                for q in TARGET_FRRS:
                    vals = {c: percell[c][(mode, q)] for c in combos}
                    key = "%s|q%g|%s" % (mode, q, st)
                    acc[key]["ALL"].append(np.mean([v[1] for v in vals.values()]))
                    acc[key]["ALL_fa"].append(np.mean([v[0] for v in vals.values()]))
                    for m in MODALITIES:
                        acc[key]["MOD:" + m].append(
                            np.mean([vals[(m, d)][1] for d in DETECTORS]))
                    for d in DETECTORS:
                        acc[key]["DET:" + d].append(
                            np.mean([vals[(m, d)][1] for m in MODALITIES]))
                    for c in combos:
                        acc[key]["CELL:" + group_key(*c)].append(vals[c][1])
        if (b + 1) % 250 == 0:
            log("    bootstrap %d/%d  (%.0f s)" % (b + 1, N_BOOT, time.time() - t0))

    out = {}
    for key, groups in acc.items():
        out[key] = {}
        for g, vals in groups.items():
            v = np.asarray(vals, dtype=np.float64)
            out[key][g] = {"mean": float(v.mean()),
                           "ci_low": float(np.percentile(v, 2.5)),
                           "ci_high": float(np.percentile(v, 97.5))}
    log("  bootstrap done in %.0f s" % (time.time() - t0))
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def point_aggregate(res, combos, mode, q, st, field):
    vals = []
    for (m, d) in combos:
        vals.append(res["per_cell"][group_key(m, d)][st]["q%g" % q][field])
    return float(np.mean(vals))


def report(res, combos, tag):
    log("\n" + "#" * 78)
    log("# HEADLINE TABLE -- release %s" % tag)
    log("#" * 78)
    boot = res["bootstrap"]
    for q in TARGET_FRRS:
        log("\n--- session FRR target = %g ---" % q)
        log("%-12s | %-32s | %-32s | %s"
            % ("stat", "USER-DISJOINT caught [95% CI]",
               "ORACLE caught [95% CI]", "gap  realisedFA  AUC"))
        for st in STATS:
            xk = "xval|q%g|%s" % (q, st)
            ok = "oracle|q%g|%s" % (q, st)
            xc = point_aggregate(res, combos, "xval", q, st, "xval_caught")
            oc = point_aggregate(res, combos, "xval", q, st, "oracle_caught")
            xfa = point_aggregate(res, combos, "xval", q, st, "xval_session_fa")
            ofa = point_aggregate(res, combos, "xval", q, st, "oracle_session_fa")
            auc = float(np.mean([res["per_cell"][group_key(m, d)][st]["auc_mean"]
                                 for (m, d) in combos]))
            log("%-12s | %.4f [%.4f, %.4f]            | %.4f [%.4f, %.4f]            "
                "| %+.4f  %.4f/%.4f  %.4f"
                % (st, xc, boot[xk]["ALL"]["ci_low"], boot[xk]["ALL"]["ci_high"],
                   oc, boot[ok]["ALL"]["ci_low"], boot[ok]["ALL"]["ci_high"],
                   oc - xc, xfa, ofa, auc))

    log("\n--- PRICE OF DETECTION (macro over 18 combos; session FA needed) ---")
    log("%-12s | %-26s | %s" % ("stat", "USER-DISJOINT FA @caught",
                                "ORACLE FA @caught"))
    for st in STATS:
        a = res["aggregate_curves"][st]
        px = a["price_xval"]
        po = a["price_oracle"]
        def f(v):
            return "  n/a " if v is None else "%.3f" % v
        log("%-12s | 50%%:%s 80%%:%s 95%%:%s | 50%%:%s 80%%:%s 95%%:%s"
            % (st, f(px["caught0.5"]), f(px["caught0.8"]), f(px["caught0.95"]),
               f(po["caught0.5"]), f(po["caught0.8"]), f(po["caught0.95"])))

    log("\n--- PRICE OF DETECTION, per-cell distribution (user-disjoint) ---")
    for st in STATS:
        line = ["%-12s" % st]
        for t in CAUGHT_TARGETS:
            vals = [res["curves"][group_key(m, d)][st]["price_xval"]["caught%g" % t]
                    for (m, d) in combos]
            vals = [1.0 if v is None else v for v in vals]
            line.append("caught%g%%: median %.3f [%.3f,%.3f]"
                        % (t * 100, float(np.median(vals)), min(vals), max(vals)))
        log("  " + " | ".join(line))

    log("\n--- S1 COUNT RULE, integer k (macro over 18 combos, pooled all users) ---")
    log("  k :  session FA   caught      (fa range)          (caught range)")
    pk = res["count_rule"]["per_k"]
    for k in sorted(pk, key=lambda x: int(x)):
        v = pk[k]
        if v["macro_session_fa"] < 1e-9 and v["macro_caught"] < 1e-9 and int(k) > 3:
            continue
        log("  %2s:  %.4f      %.4f      [%.4f,%.4f]   [%.4f,%.4f]"
            % (k, v["macro_session_fa"], v["macro_caught"],
               v["fa_min"], v["fa_max"], v["caught_min"], v["caught_max"]))

    log("\n--- BREAKOUTS at session FRR=5%, statistic S1_COUNT (user-disjoint) ---")
    xk = "xval|q0.05|S1_COUNT"
    for m in MODALITIES:
        v = res["bootstrap"][xk]["MOD:" + m]
        log("  modality %-22s caught %.4f [%.4f, %.4f]"
            % (m, v["mean"], v["ci_low"], v["ci_high"]))
    for d in DETECTORS:
        v = res["bootstrap"][xk]["DET:" + d]
        log("  detector %-22s caught %.4f [%.4f, %.4f]"
            % (d, v["mean"], v["ci_low"], v["ci_high"]))

    log("\n--- BREAKOUTS at session FRR=5%, best statistic per group (user-disjoint) ---")
    for st in STATS:
        k = "xval|q0.05|%s" % st
        row = ["%s %.3f" % (m.split("_")[0][:4], res["bootstrap"][k]["MOD:" + m]["mean"])
               for m in MODALITIES]
        row2 = ["%s %.3f" % (d[:9], res["bootstrap"][k]["DET:" + d]["mean"])
                for d in DETECTORS]
        log("  %-12s | %s | %s" % (st, "  ".join(row), "  ".join(row2)))

    log("\n--- per-cell, S1_COUNT, session FRR=5%, user-disjoint ---")
    log("%-22s %-19s  caught   FA     oracle  gap    AUC" % ("modality", "detector"))
    for (m, d) in combos:
        e = res["per_cell"][group_key(m, d)]["S1_COUNT"]["q0.05"]
        auc = res["per_cell"][group_key(m, d)]["S1_COUNT"]["auc_mean"]
        log("%-22s %-19s  %.4f  %.4f  %.4f  %+.4f  %.4f"
            % (m, d, e["xval_caught"], e["xval_session_fa"],
               e["oracle_caught"], e["gap_oracle_minus_xval"], auc))

    log("\n--- draw-to-draw spread of the caught rate (S1_COUNT, q=0.05, xval) ---")
    sds = [res["per_cell"][group_key(m, d)]["S1_COUNT"]["q0.05"]["xval_caught_per_draw_std"]
           for (m, d) in combos]
    rng_ = [res["per_cell"][group_key(m, d)]["S1_COUNT"]["q0.05"]["xval_caught_per_draw_max"] -
            res["per_cell"][group_key(m, d)]["S1_COUNT"]["q0.05"]["xval_caught_per_draw_min"]
            for (m, d) in combos]
    log("  across R=%d draws: max SD over cells %.5f, max (max-min) %.5f, "
        "median SD %.5f" % (R_DRAWS, max(sds), max(rng_), float(np.median(sds))))


# --------------------------------------------------------------------------

def main():
    global _LOG_FH
    os.makedirs(OUT_DIR, exist_ok=True)
    _LOG_FH = open(os.path.join(OUT_DIR, "run.log"), "w")
    log("sessagg_a.py  agent A  start %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    log("python %s" % sys.version.replace("\n", " "))
    log("numpy %s" % np.__version__)

    allres = {}
    for tag in ("repo", "mnt"):
        res, sessions, G, F, combos = run_release(tag, RELEASES[tag])
        report(res, combos, tag)
        dump(res, "results_%s.json" % tag)
        allres[tag] = res
        if tag == "repo":
            inv = [{k: v for k, v in s.items() if k != "events"} for s in sessions]
            dump({"sessions": inv}, "sessions.json")

    # cross-release comparison of the headline
    log("\n" + "#" * 78)
    log("# CROSS-RELEASE COMPARISON (S1_COUNT, q=0.05, user-disjoint, macro)")
    log("#" * 78)
    for tag in ("repo", "mnt"):
        b = allres[tag]["bootstrap"]["xval|q0.05|S1_COUNT"]["ALL"]
        ev = allres[tag]["verify"]["event_far_at_frr5_mean_over_90_cells"]
        log("  %-5s  event FAR@frr5 %.4f  ->  session caught %.4f [%.4f, %.4f]"
            % (tag, ev, b["mean"], b["ci_low"], b["ci_high"]))

    dump(allres, "RESULTS.json")
    log("\nDONE %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    _LOG_FH.close()


if __name__ == "__main__":
    main()
