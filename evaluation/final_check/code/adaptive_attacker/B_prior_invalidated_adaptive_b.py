#!/usr/bin/env python
"""
adaptive_b.py -- Adaptive-attacker session-level analysis (agent B, independent implementation).

Pure score-space + duration-space arithmetic on already-existing files.  No GPU, no model
loading, no inference.

Stages (select with --stage):
  prep    build the cached event table, score matrix, duration table, A2 features, sessions
  partB   duration-only session LLR, control, quantile mapping, bulk stats, timing consequence
  partA   adaptive-attacker curve (A0/A1/A2/A3) + baselines B0/B1/B2 + Part C per-user price
  all     prep, partB, partA

Everything is dumped to JSON under OUT/.
"""
import argparse
import json
import math
import os
import sys
import time
import hashlib
from collections import defaultdict

import numpy as np

# ----------------------------------------------------------------------------- constants
ROOT = "/mnt/share/mwang49/data7/results/direct100k"
CELLS = os.path.join(ROOT, "detectors_90cell", "cells")
SHARDS = os.path.join(ROOT, "replay_dataset_full", "shards")
BINDINGS = os.path.join(ROOT, "genuine_bindings_v1", "genuine_bindings.jsonl")
RELEASE = os.path.join(ROOT, "replay_dataset_full", "event_manifest.jsonl")
OUT = ("/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/"
       "e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/B/out_agentB")

ACTIONS = ["tap", "scroll", "swipe", "pinch", "keystroke"]
AIDX = {a: i for i, a in enumerate(ACTIONS)}
MODALITIES = ["imu_only", "trajectory_xytime", "imu_trajectory_xytime"]
DETECTORS = ["authconformer", "behaveformer_stdat", "hmog_style_rf",
             "hmog_style_svm", "paper_svm", "paper_xgboost"]
CELLKEYS = [(m, d) for m in MODALITIES for d in DETECTORS]      # 18 victim / surrogate cells
NCELL = len(CELLKEYS)

POOL_N = 200                     # published fake events per (user, action)
RATIOS = [1, 2, 5, 10, 20]       # rejection ratios r
NQ = 41                          # quantiles -> 40 bins
LLR_CLIP = 5.0
R_SEEDS = 20                     # replicate draws
BOOT = 1000                      # user-clustered bootstrap replicates
SESSIONS_PER_DAY = 10.0          # stated parameter for the lockout cadence
FRR_TARGETS = [0.05, 0.01]
CAUGHT_TARGETS = [0.50, 0.80, 0.95]

# ----------------------------------------------------------------------------- small utils


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


T0 = time.time()


def jdump(obj, name):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, default=_jdefault)
    log("wrote", p, "%.1f KB" % (os.path.getsize(p) / 1024.0))
    return p


def _jdefault(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(repr(type(o)))


def quantile_llr_fit(x_fake, x_gen, nq=NQ, clip=LLR_CLIP):
    """41-quantile LLR: edges from the pooled sample, Laplace(0.5) smoothed bin masses."""
    pooled = np.concatenate([x_fake, x_gen])
    edges = np.quantile(pooled, np.linspace(0.0, 1.0, nq))
    edges = np.unique(edges)
    if edges.size < 3:                      # degenerate -> single bin, LLR 0
        edges = np.array([pooled.min(), pooled.max() + 1e-9])
    nb = edges.size - 1
    bf = np.clip(np.searchsorted(edges, x_fake, side="right") - 1, 0, nb - 1)
    bg = np.clip(np.searchsorted(edges, x_gen, side="right") - 1, 0, nb - 1)
    cf = np.bincount(bf, minlength=nb).astype(np.float64) + 0.5
    cg = np.bincount(bg, minlength=nb).astype(np.float64) + 0.5
    pf = cf / cf.sum()
    pg = cg / cg.sum()
    llr = np.clip(np.log(pf) - np.log(pg), -clip, clip)
    return edges, llr


def quantile_llr_apply(edges, llr, x):
    nb = edges.size - 1
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)
    return llr[b]


# ----------------------------------------------------------------------------- stage: prep

def load_splits():
    out = {}
    txt = open(RELEASE).read()
    dec = json.JSONDecoder()
    i = 0
    while i < len(txt):
        while i < len(txt) and txt[i] in " \n\r\t":
            i += 1
        if i >= len(txt):
            break
        blk, j = dec.raw_decode(txt, i)
        i = j
        out[blk["split"]] = sorted(blk["user_ids"])
    return out


def event_features(z):
    """Per-event summary features from the ragged shard.  Attacker-visible only (its own
    generated IMU + trajectory).  No genuine data, no PAD model, no labels."""
    off = z["offsets"].astype(np.int64)
    imu = z["imu_flat"].astype(np.float64)
    tr = z["trajectory_flat"].astype(np.float64)
    n = off.size - 1
    cnt = (off[1:] - off[:-1]).astype(np.float64)
    starts = off[:-1]
    cols = [imu[:, c] for c in range(6)] + [tr[:, c] for c in (1, 2, 5, 6)]
    feats = []
    for v in cols:
        s = np.add.reduceat(v, starts)
        s2 = np.add.reduceat(v * v, starts)
        mu = s / cnt
        var = np.maximum(s2 / cnt - mu * mu, 0.0)
        feats.append(mu)
        feats.append(np.sqrt(var))
    # x/y span
    for c in (1, 2):
        v = tr[:, c]
        mx = np.maximum.reduceat(v, starts)
        mn = np.minimum.reduceat(v, starts)
        feats.append(mx)
        feats.append(mn)
    feats.append(np.log(cnt))                       # log n_samples  (duration-derived)
    dur = tr[off[1:] - 1, 7]
    feats.append(np.log(np.maximum(dur, 1e-3)))     # log duration   (duration-derived)
    F = np.stack(feats, axis=1).astype(np.float32)
    assert F.shape[0] == n
    return F, dur.astype(np.float64), cnt.astype(np.int64)


N_DUR_FEATS = 2   # the last two feature columns are duration-derived (A2nodur drops them)


def prep():
    splits = load_splits()
    test_users = splits["test"]
    dev_users = splits["development"]
    log("test users", len(test_users), "dev users", len(dev_users))

    def load_users(users, want_feats):
        rec = dict(user=[], action=[], label=[], sess=[], eid=[], scid=[], dur=[],
                   nsamp=[], feat=[])
        for ui, u in enumerate(users):
            z = np.load(os.path.join(SHARDS, u + ".npz"), allow_pickle=True)
            F, dur, cnt = event_features(z)
            n = dur.size
            assert (z["user_id"] == u).all()
            rec["user"].append(np.full(n, ui, np.int32))
            rec["action"].append(np.array([AIDX[a] for a in z["action"]], np.int8))
            rec["label"].append(z["label"].astype(np.int8))
            rec["sess"].append(z["session_id"].astype("U24"))
            rec["eid"].append(z["event_id"].astype("U64"))
            rec["scid"].append(z["source_cluster_id"].astype("U64"))
            rec["dur"].append(dur)
            rec["nsamp"].append(cnt)
            if want_feats:
                rec["feat"].append(F)
        out = {k: (np.concatenate(v) if v else None) for k, v in rec.items() if k != "feat"}
        out["feat"] = np.concatenate(rec["feat"]) if want_feats else None
        return out

    te = load_users(test_users, True)
    dv = load_users(dev_users, False)
    N = te["dur"].size
    log("test events", N, "dev events", dv["dur"].size)

    # ---- scores: [18, N]; every test event must be found in its (action, cell) file
    S = np.full((NCELL, N), np.nan, np.float64)
    eid2row = {e: i for i, e in enumerate(te["eid"])}
    assert len(eid2row) == N, "duplicate event_id in test shards"
    thr = np.zeros((NCELL, len(ACTIONS)))
    missing = []
    for ci, (m, d) in enumerate(CELLKEYS):
        for ai, a in enumerate(ACTIONS):
            cd = os.path.join(CELLS, "%s__%s__%s" % (a, m, d))
            th = json.load(open(os.path.join(cd, "thresholds.json")))
            assert th["score_direction"] == "larger_is_more_fake"
            assert th["selection_split"] == "development" and th["target_frr"] == 0.05
            thr[ci, ai] = th["frr5"]
            nfound = 0
            for line in open(os.path.join(cd, "test_scores.jsonl")):
                r = json.loads(line)
                row = eid2row.get(r["event_id"])
                if row is None:
                    missing.append((a, m, d, r["event_id"]))
                    continue
                S[ci, row] = r["fake_high_score"]
                nfound += 1
            assert nfound > 0
    nan = np.isnan(S).sum()
    log("score matrix filled; NaNs", nan, "unmatched score rows", len(missing))
    assert nan == 0, "some test event has no score in some cell"
    assert not missing

    # ---- sessions from genuine_bindings (full slot timeline) ------------------------
    tset = set(test_users)
    dset = set(dev_users)
    bind = defaultdict(list)
    bind_dev = defaultdict(list)
    for line in open(BINDINGS):
        r = json.loads(line)
        u = r["user_id"]
        if u in tset:
            bind[r["session_id"]].append(r)
        elif u in dset:
            bind_dev[r["session_id"]].append(r)

    scid2row = {}
    gmask = te["label"] == 0
    for i in np.nonzero(gmask)[0]:
        scid2row[te["scid"][i]] = int(i)
    assert len(scid2row) == int(gmask.sum())

    uidx = {u: i for i, u in enumerate(test_users)}

    def build_sessions(binddict, scid2row_local, uidx_local):
        sess = []
        for sid, rs in sorted(binddict.items()):
            rs.sort(key=lambda r: r["start_sample"])
            onset = np.array([r["start_sample"] for r in rs], np.float64) / 100.0
            dur = np.array([r["duration_ms"] for r in rs], np.float64) / 1000.0
            acts = [r["action"] for r in rs]
            rows = [scid2row_local.get(r["source_cluster_id"], -1) for r in rs]
            rows = np.array(rows, np.int64)
            keep = rows >= 0
            sess.append(dict(sid=sid, user=uidx_local[rs[0]["user_id"]],
                             n_slots=len(rs), n_scored=int(keep.sum()),
                             rows=rows[keep], onset=onset[keep], dur_full=dur[keep],
                             act=np.array([AIDX[a] for a in acts], np.int8)[keep],
                             onset_all=onset, dur_all=dur,
                             act_all=np.array([AIDX[a] for a in acts], np.int8)))
        return sess

    sess = build_sessions(bind, scid2row, uidx)
    tot_slots = sum(s["n_slots"] for s in sess)
    tot_scored = sum(s["n_scored"] for s in sess)
    log("test sessions in bindings", len(sess), "slots", tot_slots,
        "scored", tot_scored, "= %.1f%%" % (100.0 * tot_scored / tot_slots))
    assert tot_scored == int(gmask.sum()), (tot_scored, int(gmask.sum()))

    # dev sessions (for the timing-feature detector fitted off-test)
    dgm = dv["label"] == 0
    dscid2row = {}
    for i in np.nonzero(dgm)[0]:
        dscid2row[dv["scid"][i]] = int(i)
    duidx = {u: i for i, u in enumerate(dev_users)}
    dsess = build_sessions(bind_dev, dscid2row, duidx)

    # ---- fake pool [n_users, n_actions, 200] of row indices ------------------------
    pool = np.full((len(test_users), len(ACTIONS), POOL_N), -1, np.int64)
    fk = np.nonzero(te["label"] == 1)[0]
    cnt = defaultdict(int)
    for i in fk:
        u = int(te["user"][i]); a = int(te["action"][i])
        k = cnt[(u, a)]
        assert k < POOL_N
        pool[u, a, k] = i
        cnt[(u, a)] = k + 1
    assert (pool >= 0).all(), "pool not full for every (user, action)"

    np.savez_compressed(
        os.path.join(OUT, "cache.npz"),
        test_users=np.array(test_users), dev_users=np.array(dev_users),
        te_user=te["user"], te_action=te["action"], te_label=te["label"],
        te_dur=te["dur"], te_nsamp=te["nsamp"], te_feat=te["feat"],
        dv_user=dv["user"], dv_action=dv["action"], dv_label=dv["label"], dv_dur=dv["dur"],
        S=S.astype(np.float32), thr=thr, pool=pool)
    # sessions -> pickle-free json-ish npz (object arrays)
    np.save(os.path.join(OUT, "sessions.npy"), np.array(sess, dtype=object),
            allow_pickle=True)
    np.save(os.path.join(OUT, "dev_sessions.npy"), np.array(dsess, dtype=object),
            allow_pickle=True)

    # coverage / denominator report (F11, F15)
    cov = dict(
        test_users=test_users, dev_users=dev_users,
        n_test_events=int(N), n_test_genuine=int(gmask.sum()),
        n_test_fake=int((te["label"] == 1).sum()),
        n_sessions_in_bindings=len(sess),
        n_sessions_with_at_least_one_scored=int(sum(1 for s in sess if s["n_scored"] > 0)),
        total_session_slots=int(tot_slots), total_scored_slots=int(tot_scored),
        scored_slot_fraction=float(tot_scored) / tot_slots,
        pool_per_user_action=POOL_N,
        score_source="detectors_90cell/cells/<action>__<modality>__<detector>/test_scores.jsonl"
                     " field fake_high_score",
        duration_source="trajectory_flat[offsets[i+1]-1, 7] (elapsed_seconds); cross-checked "
                        "against genuine_bindings duration_ms -- max abs diff 0.0 s",
        onset_source="genuine_bindings start_sample / 100 Hz (n_samples == duration_ms/10 + 1 "
                     "exactly for all 46231 test slots)")
    jdump(cov, "coverage_prep.json")
    return cov


# ----------------------------------------------------------------------------- load cache

class Data(object):
    def __init__(self):
        z = np.load(os.path.join(OUT, "cache.npz"), allow_pickle=True)
        self.test_users = [str(x) for x in z["test_users"]]
        self.dev_users = [str(x) for x in z["dev_users"]]
        self.user = z["te_user"]
        self.action = z["te_action"]
        self.label = z["te_label"]
        self.dur = z["te_dur"]
        self.nsamp = z["te_nsamp"]
        self.feat = z["te_feat"]
        self.dv_user = z["dv_user"]
        self.dv_action = z["dv_action"]
        self.dv_label = z["dv_label"]
        self.dv_dur = z["dv_dur"]
        self.S = z["S"].astype(np.float64)
        self.thr = z["thr"]
        self.pool = z["pool"]
        self.N = self.dur.size
        self.sess = list(np.load(os.path.join(OUT, "sessions.npy"), allow_pickle=True))
        self.dsess = list(np.load(os.path.join(OUT, "dev_sessions.npy"), allow_pickle=True))
        # fixed, recorded user halves (F2): sorted user list, even index -> half 0
        self.half = np.array([i % 2 for i in range(len(self.test_users))], np.int8)

    def regime_sessions(self, regime, min_scored=1):
        out = []
        drop = 0
        for s in self.sess:
            if s["n_scored"] < min_scored:
                drop += 1
                continue
            acts = set(self.action[s["rows"]].tolist())
            has_ks = AIDX["keystroke"] in acts
            has_touch = bool(acts - {AIDX["keystroke"]})
            if regime == "touch":
                ok = has_touch and not has_ks
            elif regime == "keystroke":
                ok = has_ks and not has_touch
            else:
                ok = has_ks and has_touch
            if ok:
                out.append(s)
        return out


# ----------------------------------------------------------------------------- duration LLR


def fit_duration_llr(dat, transform=None, dv_dur=None):
    """Per-action duration LLR fitted on the 10 DEVELOPMENT users (F1: never test)."""
    src = dat.dv_dur if dv_dur is None else dv_dur
    edges, llrs = {}, {}
    for ai in range(len(ACTIONS)):
        m = dat.dv_action == ai
        dg = src[m & (dat.dv_label == 0)]
        df = src[m & (dat.dv_label == 1)]
        if transform is not None:
            dg, df = transform(ai, dg, df)
        e, l = quantile_llr_fit(df, dg)
        edges[ai], llrs[ai] = e, l
    return edges, llrs


def apply_per_action(edges, llrs, action, x):
    out = np.zeros(x.size)
    for ai in range(len(ACTIONS)):
        m = action == ai
        if m.any():
            out[m] = quantile_llr_apply(edges[ai], llrs[ai], x[m])
    return out


def make_control(dat, rng):
    """Reviewer's prescribed control: clip BOTH arms at the per-action fake cap, then add
    U(-5,+5) ms dequantising jitter to both arms.  Caps taken per action from the fake
    durations of the arm's own split (dev cap for dev fitting, test cap for test)."""
    caps_dev, caps_te = {}, {}
    for ai in range(len(ACTIONS)):
        caps_dev[ai] = dat.dv_dur[(dat.dv_action == ai) & (dat.dv_label == 1)].max()
        caps_te[ai] = dat.dur[(dat.action == ai) & (dat.label == 1)].max()

    def tf_dev(ai, dg, df):
        c = caps_dev[ai]
        return (np.minimum(dg, c) + rng.uniform(-0.005, 0.005, dg.size),
                np.minimum(df, c) + rng.uniform(-0.005, 0.005, df.size))

    d_te = dat.dur.copy()
    for ai in range(len(ACTIONS)):
        m = dat.action == ai
        d_te[m] = np.minimum(d_te[m], caps_te[ai])
    d_te = d_te + rng.uniform(-0.005, 0.005, d_te.size)
    return tf_dev, d_te, caps_dev, caps_te


def make_qmap(dat, mode="grid", fit_split="dev", jitter_s=0.005, seed=99):
    """Quantile-map every FAKE duration onto the per-action GENUINE duration distribution.

    mode='grid'   : inverse CDF by `inverted_cdf`, so a mapped value is always an actual
                    genuine sample value.  Durations are quantised to 10 ms in this corpus;
                    a linearly-interpolated inverse CDF would place every mapped fake value
                    strictly OFF that 10 ms grid and hand the defender a perfect new tell,
                    which is a numerical artefact of the fix, not a residual of the attack.
    mode='jitter' : dequantise BOTH arms with U(-jitter, +jitter) first, then map with a
                    linearly-interpolated inverse CDF.  Both arms are then continuous.

    fit_split='dev'  -> deployable: the map is fitted on the 10 development users (F1).
    fit_split='test' -> DIAGNOSTIC ONLY, oracle-fitted map, reported as an upper bound on
                        how well any such mapping could do."""
    rng = np.random.default_rng(seed)
    jit = jitter_s if mode == "jitter" else 0.0

    dv_dur = dat.dv_dur + (rng.uniform(-jit, jit, dat.dv_dur.size) if jit else 0.0)
    te_dur = dat.dur + (rng.uniform(-jit, jit, dat.dur.size) if jit else 0.0)

    maps = {}
    for ai in range(len(ACTIONS)):
        if fit_split == "dev":
            m = dat.dv_action == ai
            dg = np.sort(dv_dur[m & (dat.dv_label == 0)])
            df = np.sort(dv_dur[m & (dat.dv_label == 1)])
        else:
            m = dat.action == ai
            dg = np.sort(te_dur[m & (dat.label == 0)])
            df = np.sort(te_dur[m & (dat.label == 1)])
        maps[ai] = (df, dg)

    def apply(ai, x):
        df, dg = maps[ai]
        q = np.clip((np.searchsorted(df, x, side="right") - 0.5) / df.size, 0.0, 1.0)
        if mode == "grid":
            return dg[np.clip((q * dg.size).astype(np.int64), 0, dg.size - 1)]
        return np.quantile(dg, q)

    def tf_dev(ai, dg_in, df_in):
        # dg_in / df_in already carry the jitter, they come from dv_dur via fit_duration_llr
        return dg_in, apply(ai, df_in)

    d_te = te_dur.copy()
    for ai in range(len(ACTIONS)):
        m = (dat.action == ai) & (dat.label == 1)
        d_te[m] = apply(ai, te_dur[m])
    return tf_dev, d_te, dv_dur


# ----------------------------------------------------------------------------- part B


def duration_bulk_stats(dat):
    out = {}
    for ai, a in enumerate(ACTIONS):
        m = dat.action == ai
        g = dat.dur[m & (dat.label == 0)]
        f = dat.dur[m & (dat.label == 1)]
        cap = f.max()
        row = {}
        for nm, v in (("genuine", g), ("fake", f)):
            row[nm] = dict(n=int(v.size), p5=float(np.percentile(v, 5)),
                           p50=float(np.percentile(v, 50)),
                           median=float(np.median(v)),
                           p95=float(np.percentile(v, 95)),
                           mean=float(v.mean()), max=float(v.max()), min=float(v.min()))
        row["fake_cap_s"] = float(cap)
        row["frac_fake_exactly_at_cap"] = float(np.mean(f >= cap - 1e-12))
        row["frac_fake_within_1ms_of_cap"] = float(np.mean(f >= cap - 1.001e-3))
        row["frac_genuine_above_fake_cap"] = float(np.mean(g > cap))
        row["median_shift_s"] = float(np.median(g) - np.median(f))
        # dev split, for reference
        md = dat.dv_action == ai
        row["dev_genuine_median"] = float(np.median(dat.dv_dur[md & (dat.dv_label == 0)]))
        row["dev_fake_median"] = float(np.median(dat.dv_dur[md & (dat.dv_label == 1)]))
        row["dev_fake_cap_s"] = float(dat.dv_dur[md & (dat.dv_label == 1)].max())
        out[a] = row
    # genuine durations over ALL 46231 session slots (not only the 8991 scored ones), so a
    # reader can see the scored subsample is not a biased slice of the genuine timeline (F11)
    all_d = np.concatenate([s["dur_all"] for s in dat.sess])
    all_a = np.concatenate([s["act_all"] for s in dat.sess])
    for ai, a in enumerate(ACTIONS):
        v = all_d[all_a == ai]
        out[a]["genuine_all_session_slots"] = dict(
            n=int(v.size), p5=float(np.percentile(v, 5)), median=float(np.median(v)),
            p95=float(np.percentile(v, 95)), max=float(v.max()))
    return out


# --------------------------------------------------------------- session machinery


def seg_layout(sessions):
    """Flatten scored slots of a session list; return row indices, segment starts, sizes."""
    rows = np.concatenate([s["rows"] for s in sessions])
    sizes = np.array([s["rows"].size for s in sessions], np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    users = np.array([s["user"] for s in sessions], np.int32)
    return rows, starts, sizes, users


def segment_mean(ch, starts, sizes):
    """ch: [C, n_slots] -> [C, n_seg] mean."""
    s = np.add.reduceat(ch, starts, axis=1)
    return s / sizes[None, :]


def segment_sum(ch, starts, sizes):
    return np.add.reduceat(ch, starts, axis=1)


def caught_rate_at_thresholds(gen, fake, thr):
    """gen: [n_gen] genuine session scores; fake: [...] fake session scores; thr scalar."""
    return float(np.mean(fake >= thr)), float(np.mean(gen >= thr))


def threshold_for_frr(gen_scores, target):
    """Smallest cut t with mean(gen >= t) <= target, taken from the observed score grid."""
    if gen_scores.size == 0:
        return np.inf, 0.0
    s = np.sort(gen_scores)[::-1]
    k = int(math.floor(target * s.size))       # allow at most k genuine alarms
    if k <= 0:
        t = np.nextafter(s[0], np.inf)
    elif k >= s.size:
        t = -np.inf
    else:
        t = s[k - 1]
        # move just above the k-th largest so exactly the top k (or fewer, with ties) alarm
        t = np.nextafter(t, np.inf)
    return float(t), float(np.mean(gen_scores >= t))




def fit_beta_binomial(k, n):
    """Method-of-moments beta-binomial fit to per-user (alarms, sessions)."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / np.maximum(n, 1)
    m = float(np.average(p, weights=n))
    if m <= 0 or m >= 1:
        return dict(method="moments", alpha=None, beta=None, mean=m,
                    note="degenerate (mean 0 or 1)")
    v = float(np.average((p - m) ** 2, weights=n))
    nbar = float(n.mean())
    # binomial part of the variance
    vb = m * (1 - m) / nbar
    if v <= vb:
        return dict(method="moments", alpha=None, beta=None, mean=m,
                    note="no overdispersion detected (var <= binomial var)")
    rho = (v - vb) / (m * (1 - m) * (1 - 1.0 / nbar))
    rho = float(min(max(rho, 1e-6), 0.999))
    s = (1 - rho) / rho
    a, b = m * s, (1 - m) * s
    from scipy.stats import beta as _beta
    return dict(method="moments", alpha=float(a), beta=float(b), mean=m, icc=rho,
                fitted_p50=float(_beta.ppf(0.5, a, b)),
                fitted_p90=float(_beta.ppf(0.9, a, b)),
                fitted_p99=float(_beta.ppf(0.99, a, b)))


def per_user_frr_block(gen_scores, gen_users, thr, n_users):
    alarms = np.zeros(n_users)
    tot = np.zeros(n_users)
    al = (gen_scores >= thr).astype(float)
    np.add.at(alarms, gen_users, al)
    np.add.at(tot, gen_users, 1.0)
    seen = tot > 0
    p = alarms[seen] / tot[seen]
    blk = dict(n_users=int(seen.sum()), min=float(p.min()), median=float(np.median(p)),
               p90=float(np.percentile(p, 90)), max=float(p.max()),
               pooled=float(alarms.sum() / tot.sum()),
               ratio_max_over_min=(float(p.max() / p.min()) if p.min() > 0 else None),
               per_user=[float(x) for x in p])
    blk["fit"] = fit_beta_binomial(alarms[seen], tot[seen])
    # lockout cadence: days between false lockouts at SESSIONS_PER_DAY sessions/day
    def cadence(x):
        return (1.0 / (x * SESSIONS_PER_DAY)) if x > 0 else None
    blk["sessions_per_day"] = SESSIONS_PER_DAY
    blk["days_between_false_lockouts_median_user"] = cadence(float(np.median(p)))
    blk["days_between_false_lockouts_p90_user"] = cadence(float(np.percentile(p, 90)))
    blk["days_between_false_lockouts_worst_user"] = cadence(float(p.max()))
    return blk


def threshold_for_p90_user_frr(gen_scores, gen_users, n_users, cap=0.01):
    """Smallest cut such that the 90th-percentile per-user session FRR is <= cap."""
    cand = np.unique(gen_scores)
    cand = np.concatenate([cand, [np.nextafter(cand.max(), np.inf)]]) if cand.size else \
        np.array([np.inf])
    lo, hi = 0, cand.size - 1
    best = cand[-1]
    while lo <= hi:
        mid = (lo + hi) // 2
        t = cand[mid]
        alarms = np.zeros(n_users)
        tot = np.zeros(n_users)
        np.add.at(alarms, gen_users, (gen_scores >= t).astype(float))
        np.add.at(tot, gen_users, 1.0)
        seen = tot > 0
        p = alarms[seen] / tot[seen]
        if np.percentile(p, 90) <= cap:
            best = t
            hi = mid - 1
        else:
            lo = mid + 1
    return float(best)


# ----------------------------------------------------------------------------- main partB

def part_b(dat):
    res = dict(bulk=duration_bulk_stats(dat))
    rng = np.random.default_rng(20260810)

    variants = {}
    e, l = fit_duration_llr(dat)
    variants["raw"] = (e, l, dat.dur)
    tf_dev, d_ctrl, caps_dev, caps_te = make_control(dat, rng)
    e, l = fit_duration_llr(dat, tf_dev)
    variants["control_clip_jitter"] = (e, l, d_ctrl)
    for nm, mode, fs in (("quantile_mapped_grid_devfit", "grid", "dev"),
                         ("quantile_mapped_jitter_devfit", "jitter", "dev"),
                         ("quantile_mapped_grid_testfit_DIAGNOSTIC", "grid", "test"),
                         ("quantile_mapped_jitter_testfit_DIAGNOSTIC", "jitter", "test")):
        tf, d_q, dvj = make_qmap(dat, mode=mode, fit_split=fs)
        e, l = fit_duration_llr(dat, tf, dv_dur=dvj)
        variants[nm] = (e, l, d_q)
    # naive linear-interpolation map, kept to document the artefact it creates
    tf, d_q, dvj = make_qmap(dat, mode="jitter", fit_split="dev", jitter_s=0.0)
    e, l = fit_duration_llr(dat, tf, dv_dur=dvj)
    variants["quantile_mapped_linterp_NAIVE_offgrid"] = (e, l, d_q)

    res["caps"] = dict(dev={ACTIONS[k]: float(v) for k, v in caps_dev.items()},
                       test={ACTIONS[k]: float(v) for k, v in caps_te.items()})

    # per-event duration LLR channels
    durllr = {}
    for name, (e, l, dv) in variants.items():
        durllr[name] = apply_per_action(e, l, dat.action, dv)
    np.save(os.path.join(OUT, "durllr.npy"),
            np.stack([durllr[k] for k in ("raw", "control_clip_jitter",
                                          "quantile_mapped_grid_devfit")]))

    nu = len(dat.test_users)
    out = {}
    for regime in ("touch", "keystroke"):
        sessions = dat.regime_sessions(regime)
        rows, starts, sizes, susers = seg_layout(sessions)
        out[regime] = dict(n_sessions=len(sessions), n_scored_slots=int(sizes.sum()),
                           median_scored_slots=float(np.median(sizes)),
                           scored_slots_p5_p95=[float(np.percentile(sizes, 5)),
                                                float(np.percentile(sizes, 95))],
                           median_total_slots=float(np.median([s["n_slots"] for s in sessions])),
                           variants={})
        for name in variants:
            ch = durllr[name][None, :]
            gen = segment_mean(ch[:, rows], starts, sizes)[0]
            # fake arm: A0 uniform draws, R seeds
            fk = []
            for seed in range(R_SEEDS):
                D, short = draw_fake(dat, sessions, np.ones((nu, 5, POOL_N), bool),
                                     seed=1000 + seed)
                fk.append(segment_mean(durllr[name][None, D], starts, sizes)[0])
            fk = np.stack(fk)                        # [R, n_sess]
            out[regime]["variants"][name] = eval_operating_points(
                gen, susers, fk, nu, dat.half, sessions)
    res["duration_only_session_llr"] = out

    # ---- consequence analysis: timing features built from onsets --------------------
    res["timing_consequence"] = timing_consequence(dat)
    jdump(res, "partB_duration.json")
    return res


# ----------------------------------------------------------------------------- draws


def build_keep_order(key, pool, ratio, ascending=True):
    """key: [nu, na, POOL_N] attacker-visible ordering statistic (lower = keep).
    Returns order [nu, na, m] of pool positions kept, m = ceil(200/ratio)."""
    m = int(math.ceil(POOL_N / float(ratio)))
    k = key if ascending else -key
    order = np.argsort(k, axis=2, kind="stable")[:, :, :m]
    return order


def draw_fake(dat, sessions, keep_mask_unused, seed, order=None):
    """Draw one fake event per scored slot, without replacement inside a (session, action)
    group, from the attacker's kept subpool.  Returns row indices [n_slots] and a shortfall
    counter."""
    rng = np.random.default_rng(seed)
    if order is None:
        nu, na = dat.pool.shape[0], dat.pool.shape[1]
        order = np.tile(np.arange(POOL_N)[None, None, :], (nu, na, 1))
    m = order.shape[2]
    # group by (session, action)
    groups = []      # (u, a, positions_in_flat)
    flat_off = 0
    for s in sessions:
        acts = dat.action[s["rows"]]
        for ai in np.unique(acts):
            idx = np.nonzero(acts == ai)[0] + flat_off
            groups.append((s["user"], int(ai), idx))
        flat_off += s["rows"].size
    n_slots = flat_off
    G = len(groups)
    kmax = max(len(g[2]) for g in groups)
    U = rng.random((G, m))
    perm = np.argsort(U, axis=1)                 # random permutation of the kept subpool
    out = np.empty(n_slots, np.int64)
    shortfall = 0
    shortfall_slots = 0
    for gi, (u, a, idx) in enumerate(groups):
        k = idx.size
        if k <= m:
            pos = order[u, a, perm[gi, :k]]
        else:
            extra = rng.integers(0, m, size=k - m)
            pos = np.concatenate([order[u, a, perm[gi, :m]], order[u, a, perm[gi, extra]]])
            shortfall += 1
            shortfall_slots += k - m
        out[idx] = dat.pool[u, a, pos]
    return out, (shortfall, shortfall_slots, kmax, m)


# ----------------------------------------------------------------------------- eval core


def r5(x):
    return None if x is None else round(float(x), 6)


def user_agg(per_sess, users, n_users):
    s = np.bincount(users, weights=per_sess, minlength=n_users)
    c = np.bincount(users, minlength=n_users).astype(float)
    return s, c


def boot_ci(per_sess, users, n_users, rng, nboot=BOOT):
    """User-clustered bootstrap (F9): resample the 20 users as clusters, keep the threshold
    fixed at the value chosen on the calibration half."""
    s, c = user_agg(np.asarray(per_sess, float), users, n_users)
    present = np.nonzero(c > 0)[0]
    if present.size == 0:
        return [None, None]
    pick = rng.integers(0, present.size, size=(nboot, present.size))
    S = s[present][pick].sum(1)
    C = c[present][pick].sum(1)
    v = S / C
    return [r5(np.percentile(v, 2.5)), r5(np.percentile(v, 97.5))]


def peruser_block(alarm, users, n_users, want_list):
    s, c = user_agg(np.asarray(alarm, float), users, n_users)
    seen = c > 0
    p = s[seen] / c[seen]
    fit = fit_beta_binomial(s[seen], c[seen])
    blk = dict(n_users=int(seen.sum()), min=r5(p.min()), median=r5(np.median(p)),
               p90=r5(np.percentile(p, 90)), max=r5(p.max()),
               pooled=r5(s.sum() / c.sum()),
               ratio_max_over_min=(r5(p.max() / p.min()) if p.min() > 0 else None),
               bb_alpha=r5(fit.get("alpha")), bb_beta=r5(fit.get("beta")),
               bb_p90=r5(fit.get("fitted_p90")), bb_note=fit.get("note"),
               sessions_per_day=SESSIONS_PER_DAY)
    for nm, val in (("median_user", float(np.median(p))),
                    ("p90_user", float(np.percentile(p, 90))),
                    ("worst_user", float(p.max()))):
        blk["days_between_false_lockouts_" + nm] = \
            r5(1.0 / (val * SESSIONS_PER_DAY)) if val > 0 else None
    if want_list:
        blk["per_user"] = [r5(x) for x in p]
    return blk


def threshold_p90user(gen, users, n_users, cap=0.01):
    """Smallest cut such that the 90th-percentile per-user session FRR is <= cap.
    Fully vectorised over candidate cuts (the observed genuine score grid)."""
    cand = np.unique(gen)
    cand = np.concatenate([cand, [np.nextafter(cand[-1], np.inf)]])
    A = (gen[None, :] >= cand[:, None]).astype(np.float64)     # [nc, nS]
    oh = np.zeros((gen.size, n_users))
    oh[np.arange(gen.size), users] = 1.0
    tot = oh.sum(0)
    seen = tot > 0
    P = (A @ oh)[:, seen] / tot[seen][None, :]                 # [nc, nu]
    p90 = np.percentile(P, 90, axis=1)
    ok = np.nonzero(p90 <= cap)[0]
    return float(cand[ok[0]]) if ok.size else float(cand[-1])


def session_auc(gen, fake):
    from scipy.stats import rankdata
    x = np.concatenate([fake, gen])
    r = rankdata(x)
    n1, n0 = fake.size, gen.size
    return r5((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def far_for_caught(gen, fake, target):
    if fake.size == 0:
        return None
    t = np.nextafter(np.quantile(fake, 1.0 - target), -np.inf)
    return dict(session_false_alarm_rate=r5(np.mean(gen >= t)),
                achieved_caught=r5(np.mean(fake >= t)), threshold=r5(t))


def evaluate(gen_f, fake_f, users, half, n_users, rng, peruser=False):
    """gen_f/fake_f: dict fold -> genuine [nS] / fake [R, nS] session scores.
    fold 0/1 = fitted on user half 0/1 (calibration), fold 2 = all-20 oracle fit.
    Session FRR is the denominator everywhere (F11)."""
    hs = half[users]
    out = {}
    ops = [("frr0.05", ("frr", 0.05)), ("frr0.01", ("frr", 0.01)),
           ("p90user_frr1pct", ("p90", 0.01))]

    for scheme in ("split_pooled", "oracle_all20"):
        blk = {}
        for opname, (kind, param) in ops:
            if scheme == "oracle_all20":
                g, fk = gen_f[2], fake_f[2]
                if kind == "frr":
                    t, _ = threshold_for_frr(g, param)
                else:
                    t = threshold_p90user(g, users, n_users, param)
                caught = float((fk >= t).mean())
                per_sess = (fk >= t).mean(axis=0)
                alarm = (g >= t).astype(float)
                frr = float(alarm.mean())
                ev_users = users
                thrs = [r5(t)]
            else:
                caught_bits, alarm_bits, ev_u, thrs = [], [], [], []
                per_sess = np.zeros(gen_f[2].size)
                for f in (0, 1):
                    cal = hs == f
                    ev = ~cal
                    if cal.sum() == 0 or ev.sum() == 0:
                        continue
                    g, fk = gen_f[f], fake_f[f]
                    if kind == "frr":
                        t, _ = threshold_for_frr(g[cal], param)
                    else:
                        t = threshold_p90user(g[cal], users[cal], n_users, param)
                    caught_bits.append((fk[:, ev] >= t).ravel())
                    alarm_bits.append((g[ev] >= t).astype(float))
                    ev_u.append(users[ev])
                    per_sess[ev] = (fk[:, ev] >= t).mean(axis=0)
                    thrs.append(r5(t))
                caught = float(np.concatenate(caught_bits).mean())
                alarm = np.concatenate(alarm_bits)
                frr = float(alarm.mean())
                ev_users = np.concatenate(ev_u)
            blk[opname] = dict(
                caught=r5(caught),
                caught_ci95=boot_ci(per_sess, users, n_users, rng),
                realised_session_frr=r5(frr),
                thresholds=thrs,
                per_user_frr=peruser_block(alarm, ev_users, n_users, peruser))
        out[scheme] = blk

    g2, f2 = gen_f[2], fake_f[2].ravel()
    out["price_curve"] = {("caught%g" % c): far_for_caught(g2, f2, c)
                          for c in CAUGHT_TARGETS}
    out["price_curve_note"] = ("threshold-free ROC over all test sessions x all seeds using "
                               "the all-20 (oracle) LLR fit; no threshold is selected here")
    out["auc_session"] = session_auc(g2, f2)
    out["gap_oracle_minus_split"] = {
        k: r5(out["oracle_all20"][k]["caught"] - out["split_pooled"][k]["caught"])
        for k, _ in ops}
    # cost of the p90-user operating point relative to the pooled-FRR-5% cut
    for scheme in ("split_pooled", "oracle_all20"):
        a = out[scheme]["frr0.05"]["caught"]
        b = out[scheme]["p90user_frr1pct"]["caught"]
        out[scheme]["p90user_cost_vs_frr5"] = dict(
            caught_at_frr5=a, caught_at_p90user=b, absolute_loss=r5(a - b),
            relative_loss=(r5((a - b) / a) if a and a > 0 else None))
    return out


def eval_simple(gen, fake, users, half, n_users, rng, peruser=False):
    return evaluate({0: gen, 1: gen, 2: gen}, {0: fake, 1: fake, 2: fake},
                    users, half, n_users, rng, peruser)


# kept for compatibility with part_b / timing code
def eval_operating_points(gen, gen_users, fake, n_users, half, sessions, peruser=False):
    return eval_simple(gen, fake, gen_users, half, n_users,
                       np.random.default_rng(11), peruser)


def bootstrap_caught_ci(per_sess_caught, sess_users, n_users, nboot=BOOT, seed=7):
    return boot_ci(per_sess_caught, sess_users, n_users,
                   np.random.default_rng(seed), nboot)


# ----------------------------------------------------------------------------- timing


def timing_features(onsets, durs):
    """Onset-derived session features only.  Duration enters solely through
    onset[i] = onset[i-1] + duration[i-1] + gap[i]."""
    n = onsets.size
    if n < 2:
        return None
    ioi = np.diff(onsets)
    span = onsets[-1] + durs[-1] - onsets[0]
    return np.array([
        np.log(max(ioi.mean(), 1e-3)),
        np.log(max(ioi.std() + 1e-6, 1e-6)),
        np.log(max(np.percentile(ioi, 10), 1e-3)),
        np.log(max(np.percentile(ioi, 90), 1e-3)),
        np.log(max(span, 1e-3)),
        np.log(n / max(span, 1e-3))])


def build_timing_arms(action_of, sessions, fake_dur, gap_mode, rng, med_dur, gapbank):
    G, F = [], []
    for si, s in enumerate(sessions):
        o = s["onset"]
        if o.size < 2:
            continue
        dg_real = s["dur_full"]
        df_real = fake_dur[si]
        acts = action_of(s)
        if med_dur is not None:
            dg = med_dur[acts]
            df = med_dur[acts]
        else:
            dg, df = dg_real, df_real
        gap_g = np.maximum(o[1:] - (o[:-1] + dg_real[:-1]), 0.0)
        # paced: the attacker replays the victim's own pacing, so the gaps are IDENTICAL in
        # both arms and duration is the only variable (near-tautological, F14).
        # emp: the attacker paces from the pooled human gap distribution instead, so the fake
        # arm carries a pacing mismatch on top of the duration difference.
        gap_f = (rng.choice(gapbank, size=gap_g.size, replace=True)
                 if gap_mode == "emp" else gap_g)
        og = np.concatenate([[o[0]], o[0] + np.cumsum(dg[:-1] + gap_g)])
        of = np.concatenate([[o[0]], o[0] + np.cumsum(df[:-1] + gap_f)])
        fg = timing_features(og, dg)
        ff = timing_features(of, df)
        if fg is None or ff is None:
            continue
        G.append(fg)
        F.append(ff)
    return np.array(G), np.array(F)


def gap_bank(sessions):
    g = []
    for s in sessions:
        o, d = s["onset"], s["dur_full"]
        if o.size >= 2:
            g.append(np.maximum(o[1:] - (o[:-1] + d[:-1]), 0.0))
    return np.concatenate(g) if g else np.array([1.0])


def timing_consequence(dat):
    med_dur = np.array([np.median(dat.dur[(dat.action == ai) & (dat.label == 0)])
                        for ai in range(len(ACTIONS))])
    out = dict(genuine_median_duration_s={ACTIONS[i]: r5(med_dur[i])
                                          for i in range(len(ACTIONS))},
               design=("timing features are onset-derived only (IOI mean/std/p10/p90, span, "
                       "rate).  Gaps are IDENTICAL in both arms under gap_mode=paced (F3/F8 "
                       "arm parity: duration is the only variable), and resampled from the "
                       "pooled genuine gap bank under gap_mode=emp.  The LLR is fitted on the "
                       "10 DEVELOPMENT users only (F1)."),
               regimes={})

    dev_pool_dur = defaultdict(list)
    for i in np.nonzero(dat.dv_label == 1)[0]:
        dev_pool_dur[(int(dat.dv_user[i]), int(dat.dv_action[i]))].append(dat.dv_dur[i])
    dev_pool_dur = {k: np.array(v) for k, v in dev_pool_dur.items()}
    te_pool_dur = dat.dur[dat.pool]

    for regime in ("touch", "keystroke"):
        te_sess = [s for s in dat.regime_sessions(regime) if s["onset"].size >= 2]
        dv_sess = []
        for s in dat.dsess:
            if s["onset"].size < 2:
                continue
            acts = set(dat.dv_action[s["rows"]].tolist())
            hk = AIDX["keystroke"] in acts
            ht = bool(acts - {AIDX["keystroke"]})
            if (regime == "touch" and ht and not hk) or (regime == "keystroke" and hk and not ht):
                dv_sess.append(s)
        te_users = np.array([s["user"] for s in te_sess], np.int64)
        gb_te, gb_dv = gap_bank(te_sess), gap_bank(dv_sess)
        blk = dict(n_test_sessions=len(te_sess), n_dev_sessions=len(dv_sess))

        for gap_mode in ("paced", "emp"):
            for med in (False, True):
                md = med_dur if med else None
                rngd = np.random.default_rng(555)
                dfk = [np.array([rngd.choice(dev_pool_dur[(s["user"], int(a))])
                                 for a in dat.dv_action[s["rows"]]]) for s in dv_sess]
                Gd, Fd = build_timing_arms(lambda s: dat.dv_action[s["rows"]], dv_sess,
                                           dfk, gap_mode, rngd, md, gb_dv)
                if Gd.size == 0 or Fd.size == 0:
                    continue
                nf = Gd.shape[1]
                E = [quantile_llr_fit(Fd[:, j], Gd[:, j], nq=21) for j in range(nf)]

                def sc(M):
                    return np.sum([quantile_llr_apply(E[j][0], E[j][1], M[:, j])
                                   for j in range(nf)], axis=0)

                fakes = []
                gen = None
                for seed in range(R_SEEDS):
                    rr = np.random.default_rng(9000 + seed)
                    tfk = [te_pool_dur[s["user"], dat.action[s["rows"]],
                                       rr.integers(0, POOL_N, size=s["rows"].size)]
                           for s in te_sess]
                    Gt, Ft = build_timing_arms(lambda s: dat.action[s["rows"]], te_sess,
                                               tfk, gap_mode, rr, md, gb_te)
                    if gen is None:
                        gen = sc(Gt)
                    fakes.append(sc(Ft))
                fakes = np.stack(fakes)
                key = gap_mode + ("_durmedian_both_arms" if med else "")
                blk[key] = eval_simple(gen, fakes, te_users, dat.half,
                                       len(dat.test_users), np.random.default_rng(31))
        for gap_mode in ("paced", "emp"):
            a, b = blk.get(gap_mode), blk.get(gap_mode + "_durmedian_both_arms")
            if a and b:
                for k in ("frr0.05", "frr0.01"):
                    ca = a["split_pooled"][k]["caught"]
                    cb = b["split_pooled"][k]["caught"]
                    blk.setdefault("duration_share", {})["%s_%s" % (gap_mode, k)] = dict(
                        caught_with_real_durations=ca,
                        caught_with_genuine_median_durations_both_arms=cb,
                        absolute_loss=r5(ca - cb),
                        relative_share_of_caught=(r5((ca - cb) / ca) if ca > 0 else None))
        out["regimes"][regime] = blk
    return out


# ----------------------------------------------------------------------------- part A


def a2_keys(dat, use_duration_feats=True):
    """A2 self-consistency statistic.  For each (user, action) pool of 200 generated events
    the attacker standardises its own summary-feature matrix inside the pool and takes the
    Euclidean norm of the z-scored vector, i.e. the diagonal-Mahalanobis distance to its own
    pool centroid.  Inputs: only the attacker's own generated IMU + trajectory arrays.  No
    genuine data, no detector, no labels, no test-split quantity."""
    nu, na = dat.pool.shape[0], dat.pool.shape[1]
    F = dat.feat
    ncol = F.shape[1] - (0 if use_duration_feats else N_DUR_FEATS)
    key = np.zeros((nu, na, POOL_N))
    for u in range(nu):
        for a in range(na):
            X = F[dat.pool[u, a], :ncol].astype(np.float64)
            sd = X.std(0)
            sd[sd <= 0] = 1.0
            Z = (X - X.mean(0)) / sd
            key[u, a] = np.sqrt((Z * Z).sum(1))
    return key


def stable_seed(name, seed):
    h = hashlib.md5(name.encode()).hexdigest()[:8]
    return (int(h, 16) % 1000003) * 101 + seed


def part_a(dat, regimes=("touch", "keystroke"), victims=None, out_name="partA_adaptive.json"):
    if victims is None:
        victims = list(range(NCELL))
    durllr = np.load(os.path.join(OUT, "durllr.npy"))

    caught = np.zeros((NCELL, dat.N), np.float32)
    for ci in range(NCELL):
        caught[ci] = (dat.S[ci] >= dat.thr[ci][dat.action]).astype(np.float32)
    raw = dat.S.astype(np.float32)

    llr = np.zeros((3, NCELL, dat.N), np.float32)
    fitrec = {}
    for f in range(3):
        umask = (np.isin(dat.user, np.nonzero(dat.half == f)[0]) if f < 2
                 else np.ones(dat.N, bool))
        fitrec["fold%d" % f] = dict(
            fit_users=[dat.test_users[i] for i in np.nonzero(
                dat.half == f)[0]] if f < 2 else dat.test_users,
            n_fit_genuine=int((umask & (dat.label == 0)).sum()),
            n_fit_fake=int((umask & (dat.label == 1)).sum()))
        for ci in range(NCELL):
            for ai in range(len(ACTIONS)):
                m = dat.action == ai
                fm = m & umask
                e, l = quantile_llr_fit(dat.S[ci][fm & (dat.label == 1)],
                                        dat.S[ci][fm & (dat.label == 0)])
                llr[f, ci, m] = quantile_llr_apply(e, l, dat.S[ci][m])
    log("channels built")

    keys = {("cell", ci): dat.S[ci][dat.pool] for ci in range(NCELL)}
    keys[("a2", 0)] = a2_keys(dat, True)
    keys[("a2nodur", 0)] = a2_keys(dat, False)
    log("ordering keys built")

    configs = [dict(name="A0_uniform", rule="A0", r=1, order_key=None)]
    for ci in range(NCELL):
        for r in RATIOS[1:]:
            configs.append(dict(name="ORD_%s__%s__r%d" % (CELLKEYS[ci] + (r,)),
                                rule="ORDER_CELL", r=r, order_key=("cell", ci), order_cell=ci))
    for r in RATIOS[1:]:
        configs.append(dict(name="A2_selfconsistency_r%d" % r, rule="A2", r=r,
                            order_key=("a2", 0)))
        configs.append(dict(name="A2nodur_selfconsistency_r%d" % r, rule="A2nodur", r=r,
                            order_key=("a2nodur", 0)))
        configs.append(dict(name="A2inv_selfconsistency_r%d" % r, rule="A2inv", r=r,
                            order_key=("a2", 0), descending=True))
    log("configs", len(configs))

    PERUSER_CFG = set(["A0_uniform", "A2_selfconsistency_r10", "A2nodur_selfconsistency_r10",
                       "A2_selfconsistency_r20"])
    results = dict(llr_fit_provenance=fitrec, n_seeds=R_SEEDS,
                   n_bootstrap=BOOT, regimes={})

    for regime in regimes:
        sessions = dat.regime_sessions(regime)
        rows, starts, sizes, susers = seg_layout(sessions)
        nslot = int(sizes.sum())
        nu = len(dat.test_users)
        log(regime, "sessions", len(sessions), "slots", nslot)

        gen_llr = {(f, v): segment_mean(llr[f, v][None, rows], starts, sizes)[0]
                   for f in range(3) for v in victims}
        gen_b0 = {v: segment_sum(caught[v][None, rows], starts, sizes)[0] for v in victims}
        gen_b1 = {v: segment_mean(raw[v][None, rows], starts, sizes)[0] for v in victims}
        gen_b2 = segment_mean(durllr[0][None, rows], starts, sizes)[0]

        reg = dict(n_sessions=len(sessions), n_scored_slots=nslot,
                   scored_slots_per_session=dict(
                       min=int(sizes.min()), p25=r5(np.percentile(sizes, 25)),
                       median=r5(np.median(sizes)), p75=r5(np.percentile(sizes, 75)),
                       max=int(sizes.max()), mean=r5(sizes.mean())),
                   median_total_slots_per_session=r5(
                       np.median([s["n_slots"] for s in sessions])),
                   n_sessions_with_single_scored_slot=int((sizes == 1).sum()),
                   n_users_present=int(len(set(susers.tolist()))),
                   configs={})

        for cfg in configs:
            t0 = time.time()
            if cfg["order_key"] is None:
                order, m = None, POOL_N
            else:
                order = build_keep_order(keys[cfg["order_key"]], dat.pool, cfg["r"],
                                         ascending=not cfg.get("descending", False))
                m = order.shape[2]
            F_llr = {(f, v): np.empty((R_SEEDS, len(sessions))) for f in range(3)
                     for v in victims}
            F_b0 = {v: np.empty((R_SEEDS, len(sessions))) for v in victims}
            F_b1 = {v: np.empty((R_SEEDS, len(sessions))) for v in victims}
            F_b2 = np.empty((R_SEEDS, len(sessions)))
            pef = np.zeros((R_SEEDS, NCELL))
            sg = sl = 0
            kneed = 0
            for seed in range(R_SEEDS):
                D, sh = draw_fake(dat, sessions, None,
                                  seed=stable_seed(cfg["name"] + regime, seed), order=order)
                sg += sh[0]; sl += sh[1]; kneed = max(kneed, sh[2])
                for f in range(3):
                    Lm = segment_mean(llr[f][:, D], starts, sizes)
                    for v in victims:
                        F_llr[(f, v)][seed] = Lm[v]
                C = caught[:, D]
                Cs = segment_sum(C, starts, sizes)
                Rm = segment_mean(raw[:, D], starts, sizes)
                for v in victims:
                    F_b0[v][seed] = Cs[v]
                    F_b1[v][seed] = Rm[v]
                F_b2[seed] = segment_mean(durllr[0][None, D], starts, sizes)[0]
                pef[seed] = C.mean(axis=1)

            want_pu = cfg["name"] in PERUSER_CFG
            rng = np.random.default_rng(2026)
            entry = dict(
                rule=cfg["rule"], r=cfg["r"], kept_pool_size=m,
                order_cell=(list(CELLKEYS[cfg["order_cell"]]) if "order_cell" in cfg else None),
                generated_events_per_submitted_event=r5(float(POOL_N) / m),
                pool_size=POOL_N,
                shortfall_group_draws=int(sg), shortfall_slot_draws=int(sl),
                shortfall_fraction_of_slots=r5(sl / float(nslot * R_SEEDS)),
                max_single_action_need_in_a_session=int(kneed),
                B2durLLR=eval_simple(gen_b2, F_b2, susers, dat.half, nu, rng, want_pu),
                victims={})
            for v in victims:
                mv, dv_ = CELLKEYS[v]
                if cfg["rule"] == "ORDER_CELL":
                    om, od = CELLKEYS[cfg["order_cell"]]
                    lab = ("A1_surrogate_other_family" if od != dv_ else
                           ("A3_ORACLE_not_a_threat_model" if om == mv else
                            "A3b_partial_oracle_same_family_other_modality"))
                elif cfg["rule"] == "A0":
                    lab = "A0_uniform"
                else:
                    lab = {"A2": "A2_self_consistency",
                           "A2nodur": "A2_self_consistency_no_duration_features",
                           "A2inv": "A2_self_consistency_inverted_direction"}[cfg["rule"]]
                entry["victims"]["%s__%s" % (mv, dv_)] = dict(
                    label=lab,
                    per_event_far_caught_rate=r5(pef[:, v].mean()),
                    per_event_accept_rate=r5(1.0 - pef[:, v].mean()),
                    scoreLLR=evaluate({f: gen_llr[(f, v)] for f in range(3)},
                                      {f: F_llr[(f, v)] for f in range(3)},
                                      susers, dat.half, nu, rng, want_pu),
                    B0count=eval_simple(gen_b0[v], F_b0[v], susers, dat.half, nu, rng, want_pu),
                    B1mean=eval_simple(gen_b1[v], F_b1[v], susers, dat.half, nu, rng, want_pu))
            reg["configs"][cfg["name"]] = entry
            log("  %-10s %-42s %.1fs" % (regime, cfg["name"], time.time() - t0))
        results["regimes"][regime] = reg
    results["preregistered_outcomes"] = PREREG
    jdump(results, out_name)
    return results


# ----------------------------------------------------------------------------- main

PREREG = {
    "written_before_results": True,
    "A2_defeats_defence": "if A2 (no PAD model at all) drives session caught below the genuine "
                          "session false-alarm rate at r <= 10, the session-level defence is "
                          "defeated by a free, knowledge-free attacker, and THAT is the finding",
    "only_A3_collapses": "the defence holds against realistic attackers and the oracle result "
                         "is a curiosity",
    "A1_but_not_A2": "the defence costs the attacker one surrogate model, and the price of that "
                     "model is the real security margin",
    "otherwise": "inconclusive; say so",
    "duration_acceptance_test": "after quantile mapping the duration-only session LLR must be at "
                                "chance (caught ~= session FRR); if not, the mapping failed",
}

COMPLIANCE = {
    "experiment": "adaptive-attacker session-level curve + duration confound (agent B)",
    "rules_satisfied": {
        "F1": "no threshold or fit uses test data for selection except where F2 explicitly "
              "permits a user-disjoint calibration half; the duration LLR and the timing LLR "
              "are fitted on the 10 DEVELOPMENT users only; per-event cuts come from the "
              "frozen thresholds.json (selection_split=development, target_frr=0.05)",
        "F2": "every operating point is reported twice: split_pooled (threshold and score-LLR "
              "fitted on one user half, evaluated on the other, swapped and pooled) and "
              "oracle_all20; the gap is emitted as gap_oracle_minus_split",
        "F3": "genuine and fake arms share the identical session skeletons (same user, same "
              "action multiset, same length, same scored-slot layout)",
        "F5": "A0/A1/A2 use only attacker-computable orderings; A3 uses the victim cell's own "
              "score and is labelled A3_ORACLE_not_a_threat_model everywhere and is never the "
              "headline",
        "F9": "all intervals are user-clustered bootstraps over the 20 test users as clusters",
        "F10": "B0 count rule, B1 mean score and B2 duration-only LLR are reported at every "
               "operating point beside every headline number",
        "F11": "the denominator is the SESSION for every false-alarm rate; per-event rates are "
               "named per_event_far_caught_rate / per_event_accept_rate",
        "F12": "price_curve gives the session false-alarm rate required for caught 50/80/95",
        "F13": "preregistered_outcomes is written into every results file",
        "F15": "coverage_prep.json records the denominators; every config records "
               "shortfall_group_draws / shortfall_slot_draws / max_single_action_need",
    },
    "exemptions": [
        {"rule_id": "F4",
         "reason": "this experiment consumes no new recordings; it re-uses the already published "
                   "200-event fake pool per (user, action) and the frozen per-event scores"},
        {"rule_id": "F6",
         "reason": "no detector is loaded, retrained or re-thresholded; only the frozen "
                   "test_scores.jsonl and thresholds.json are read"},
        {"rule_id": "F7",
         "reason": "all 18 (modality, detector) cells are scored because the selection rule, "
                   "not the signal content, is the variable under test"},
        {"rule_id": "F8",
         "reason": "no splicing is performed; the mirrored fake session uses whole published "
                   "fake events in the genuine skeleton"},
        {"rule_id": "F14",
         "reason": "the timing paced arm is labelled near-tautological at the point of use"},
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--regimes", default="touch,keystroke")
    ap.add_argument("--out", default="partA_adaptive.json")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jdump(dict(preregistered_outcomes=PREREG, compliance=COMPLIANCE),
          "prereg_and_compliance.json")
    if a.stage in ("prep", "all"):
        prep()
    if a.stage in ("partB", "all"):
        part_b(Data())
    if a.stage in ("partA", "all"):
        part_a(Data(), tuple(a.regimes.split(",")), None, a.out)


if __name__ == "__main__":
    main()
