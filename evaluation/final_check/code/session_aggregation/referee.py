#!/usr/bin/env python
"""
referee.py -- third, independent implementation used ONLY to adjudicate the
points where agent A and agent B disagree.

Deliberately written from the spec again, with:
  * a third seed scheme for the mirrored fake draw (so agreement is not a
    seed artefact),
  * BOTH user splits computed side by side plus 200 random balanced 10/10
    splits, so the split choice can be quantified instead of argued about,
  * BOTH price-of-detection aggregation conventions computed side by side,
  * a brute-force check that the session cut rule is the max-caught cut
    subject to calibration FRR <= q.

CPU only, score-space arithmetic on the frozen JSONL files.
"""
import gzip, json, math, os, sys, time
from collections import Counter, defaultdict
import numpy as np

TAG = sys.argv[1] if len(sys.argv) > 1 else "repo"
ROOT = {"repo": ("/home/mwang49/new/data7/data7_final_monitor_metrics_v1/USENIX8.25/"
                 "code/dataset_test/results/cells"),
        "mnt": "/mnt/share/mwang49/data7/results/direct100k/detectors_90cell/cells"}[TAG]
OUT = ("/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/"
       "e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/sessagg/C")
ACTIONS = ["tap", "scroll", "swipe", "pinch", "keystroke"]
MODALITIES = ["trajectory_xytime", "imu_only", "imu_trajectory_xytime"]
DETECTORS = ["hmog_style_svm", "hmog_style_rf", "paper_svm",
             "paper_xgboost", "behaveformer_stdat", "authconformer"]
STATS = ["S1_COUNT", "S2_MEAN", "S3_MAX", "S4_TRIMMED", "S5_LOGODDS"]
R = 20
SEEDS = [900001 + 7 * r for r in range(R)]   # third, independent seed scheme
EPS = 1e-6
QS = (0.05, 0.01)
TARGETS = (0.50, 0.80, 0.95)


def read_jsonl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def sfile(cdir):
    for nm in ("test_scores.jsonl.gz", "test_scores.jsonl"):
        p = os.path.join(cdir, nm)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(cdir)


# ------------------------------------------------------------------ load
t0 = time.time()
gen_rows, fake_rows = [], []
for a in ACTIONS:
    cd = os.path.join(ROOT, "%s__%s__%s" % (a, MODALITIES[0], DETECTORS[0]))
    for r in read_jsonl(sfile(cd)):
        (gen_rows if r["label"] == 0 else fake_rows).append(
            (r["event_id"], a, r["user_id"], r.get("session_id")))
gen_rows.sort(key=lambda t: t[0])
fake_rows.sort(key=lambda t: t[0])
gid = {t[0]: i for i, t in enumerate(gen_rows)}
fid = {t[0]: i for i, t in enumerate(fake_rows)}
gA = np.array([ACTIONS.index(t[1]) for t in gen_rows])
fA = np.array([ACTIONS.index(t[1]) for t in fake_rows])
gU = [t[2] for t in gen_rows]
fU = [t[2] for t in fake_rows]
users = sorted(set(gU))
uix = {u: i for i, u in enumerate(users)}
gUi = np.array([uix[u] for u in gU])
fUi = np.array([uix[u] for u in fU])
gS = [t[3] for t in gen_rows]
NG, NF, NU = len(gen_rows), len(fake_rows), len(users)

GS, FS, THR = {}, {}, {}
far90, frr90 = [], []
for m in MODALITIES:
    for d in DETECTORS:
        g = np.full(NG, np.nan)
        f = np.full(NF, np.nan)
        for a in ACTIONS:
            cd = os.path.join(ROOT, "%s__%s__%s" % (a, m, d))
            th = json.load(open(os.path.join(cd, "thresholds.json")))
            THR[(a, m, d)] = float(th["frr5"])
            ga, fa = [], []
            for r in read_jsonl(sfile(cd)):
                s = float(r["fake_high_score"])
                if r["label"] == 0:
                    g[gid[r["event_id"]]] = s; ga.append(s)
                else:
                    f[fid[r["event_id"]]] = s; fa.append(s)
            far90.append(float(np.mean(np.asarray(fa) < THR[(a, m, d)])))
            frr90.append(float(np.mean(np.asarray(ga) >= THR[(a, m, d)])))
        assert not np.isnan(g).any() and not np.isnan(f).any()
        GS[(m, d)] = g; FS[(m, d)] = f
print("loaded in %.1fs  event FAR@frr5 mean over 90 = %.6f  FRR = %.6f"
      % (time.time() - t0, np.mean(far90), np.mean(frr90)))

# --------------------------------------------------------------- sessions
by = defaultdict(list)
for i, s in enumerate(gS):
    by[s].append(i)
sids = sorted(by)
sev = [np.array(sorted(by[s])) for s in sids]
NS = len(sids)
suser = np.array([gUi[e[0]] for e in sev])
slen = np.array([len(e) for e in sev])
sac = np.zeros((NS, 5), int)
for i, e in enumerate(sev):
    for a in gA[e]:
        sac[i, a] += 1
seg_start = np.concatenate([[0], np.cumsum(slen)[:-1]])
seg_end = np.cumsum(slen)
gflat = np.concatenate(sev)
pool = {(u, a): np.flatnonzero((fUi == u) & (fA == a)) for u in range(NU) for a in range(5)}
print("%d sessions, %d users, len min %d median %g mean %.2f max %d"
      % (NS, NU, slen.min(), np.median(slen), slen.mean(), slen.max()))


def draw(seed):
    rng = np.random.default_rng(seed)
    out = np.empty(int(seg_end[-1]), np.int64)
    for si in range(NS):
        u = int(suser[si]); pos = int(seg_start[si])
        for a in range(5):
            c = int(sac[si, a])
            if c:
                out[pos:pos + c] = rng.choice(pool[(u, a)], size=c, replace=False)
                pos += c
        assert pos == seg_end[si]
    return out


draws = [draw(s) for s in SEEDS]
# mirror audit
for fl in draws:
    for si in range(NS):
        seg = fl[seg_start[si]:seg_end[si]]
        assert len(set(seg.tolist())) == seg.size
        assert (fUi[seg] == suser[si]).all()
        assert Counter(fA[seg].tolist()) == Counter({a: int(sac[si, a]) for a in range(5) if sac[si, a]})
print("mirror audit passed for all %d draws" % R)


def stats_of(sc, caught):
    """sc, caught laid out in the shared segment layout."""
    out = {}
    c = np.concatenate([[0.0], np.cumsum(sc)])
    out["S2_MEAN"] = (c[seg_end] - c[seg_start]) / slen
    cc = np.concatenate([[0], np.cumsum(caught.astype(np.int64))])
    out["S1_COUNT"] = (cc[seg_end] - cc[seg_start]).astype(float)
    out["S3_MAX"] = np.maximum.reduceat(sc, seg_start)
    lg = np.log(np.clip(sc, EPS, 1 - EPS) / (1 - np.clip(sc, EPS, 1 - EPS)))
    cl = np.concatenate([[0.0], np.cumsum(lg)])
    out["S5_LOGODDS"] = cl[seg_end] - cl[seg_start]
    segid = np.repeat(np.arange(NS), slen)
    order = np.lexsort((sc, segid))
    ss = sc[order]
    cs = np.concatenate([[0.0], np.cumsum(ss)])
    k = np.ceil(slen / 3.0).astype(np.int64)
    out["S4_TRIMMED"] = (cs[seg_end] - cs[seg_end - k]) / k
    return out


cells = [(m, d) for m in MODALITIES for d in DETECTORS]
G = {}; F = {}
for (m, d) in cells:
    g = GS[(m, d)]; f = FS[(m, d)]
    tg = np.array([THR[(ACTIONS[a], m, d)] for a in gA])
    tf = np.array([THR[(ACTIONS[a], m, d)] for a in fA])
    cg = g >= tg; cf = f >= tf
    G[(m, d)] = stats_of(g[gflat], cg[gflat])
    F[(m, d)] = [stats_of(f[fl], cf[fl]) for fl in draws]
print("session statistics computed  (%.1fs)" % (time.time() - t0))


# ------------------------------------------------------- threshold + eval
def cut(cal, q):
    n = cal.size
    if n == 0:
        return np.inf
    k = int(math.floor(q * n))
    if k >= n:
        return -np.inf
    return float(np.sort(cal)[::-1][k])     # (k+1)-th largest ; flag iff x > cut


def brute_best(cal_g, ev_g, ev_f, q):
    """max caught on the eval fold subject to calibration FRR <= q, over all cuts."""
    cands = np.unique(np.concatenate([cal_g, ev_g, ev_f]))
    cands = np.concatenate([[-np.inf], cands])
    best = -1.0
    for c in cands:
        if np.mean(cal_g > c) <= q + 1e-12:
            v = float(np.mean(ev_f > c))
            if v > best:
                best = v
    return best


def xval(gst, fst, maskA, q):
    """pooled two-fold user-disjoint operating point, averaged over draws"""
    ct = fa = 0.0
    n = 0
    for cal, ev in ((maskA, ~maskA), (~maskA, maskA)):
        c = cut(gst[cal], q)
        fa += float(np.count_nonzero(gst[ev] > c))
        n += int(ev.sum())
        ct += float(np.mean([np.count_nonzero(f[ev] > c) for f in fst]))
    return fa / n, ct / n


def oracle(gst, fst, q):
    c = cut(gst, q)
    return (float(np.mean(gst > c)),
            float(np.mean([np.mean(f > c) for f in fst])))


maskA_alt = np.isin(suser, [i for i in range(NU) if i % 2 == 0])     # A primary
maskA_f10 = np.isin(suser, list(range(10)))                          # B primary

# brute-force validation of the cut rule on 3 spanning cells
for (m, d) in [("trajectory_xytime", "paper_svm"), ("imu_only", "authconformer"),
               ("imu_trajectory_xytime", "paper_xgboost")]:
    for st in STATS:
        g = G[(m, d)][st]; f = F[(m, d)][0][st]
        for q in QS:
            for cal, ev in ((maskA_alt, ~maskA_alt), (~maskA_alt, maskA_alt)):
                c = cut(g[cal], q)
                got = float(np.mean(f[ev] > c))
                want = brute_best(g[cal], g[ev], f[ev], q)
                assert abs(got - want) < 1e-12, (m, d, st, q, got, want)
print("cut rule == brute-force max-caught subject to cal FRR<=q : VERIFIED")

res = {"event_far_at_frr5_mean90": float(np.mean(far90)),
       "event_frr_at_frr5_mean90": float(np.mean(frr90)),
       "n_sessions": NS, "n_users": NU,
       "seeds": SEEDS, "R": R}

# ------------------------------------------------------- headline points
per_cell = {}
for (m, d) in cells:
    per_cell["%s|%s" % (m, d)] = {}
    for st in STATS:
        g = G[(m, d)][st]; fs = [F[(m, d)][r][st] for r in range(R)]
        e = {}
        for q in QS:
            fa1, c1 = xval(g, fs, maskA_alt, q)
            fa2, c2 = xval(g, fs, maskA_f10, q)
            ofa, oc = oracle(g, fs, q)
            e["q%g" % q] = {"xval_alt_fa": fa1, "xval_alt_caught": c1,
                            "xval_f10_fa": fa2, "xval_f10_caught": c2,
                            "oracle_fa": ofa, "oracle_caught": oc}
        # AUC (Mann-Whitney, tie corrected), mean over draws
        aucs = []
        for f in fs:
            allv = np.concatenate([g, f])
            o = np.argsort(allv, kind="mergesort")
            srt = allv[o]
            new = np.empty(srt.size, bool); new[0] = True
            np.not_equal(srt[1:], srt[:-1], out=new[1:])
            dense = np.cumsum(new); gsr = np.flatnonzero(new)
            bnd = np.concatenate([gsr, [srt.size]])
            mid = 0.5 * (bnd[dense - 1] + bnd[dense] + 1)
            ranks = np.empty(srt.size); ranks[o] = mid
            aucs.append(float((ranks[g.size:].sum() - f.size * (f.size + 1) / 2) / (g.size * f.size)))
        e["auc"] = float(np.mean(aucs))
        per_cell["%s|%s" % (m, d)][st] = e
res["per_cell"] = per_cell


def macro(field, st, sel=None):
    sel = sel or cells
    return float(np.mean([per_cell["%s|%s" % (m, d)][st][field[0]][field[1]]
                          if isinstance(field, tuple) else
                          per_cell["%s|%s" % (m, d)][st][field] for (m, d) in sel]))


res["aggregate"] = {}
for st in STATS:
    a = {"auc": macro("auc", st)}
    for q in QS:
        for k in ("xval_alt_caught", "xval_alt_fa", "xval_f10_caught", "xval_f10_fa",
                  "oracle_caught", "oracle_fa"):
            a["%s@q%g" % (k, q)] = macro(("q%g" % q, k), st)
    res["aggregate"][st] = a

# ------------------------------------------------------ price of detection
QGRID = np.unique(np.concatenate([np.array([0.0]),
                                  np.arange(0, 238) / 237.0,
                                  np.arange(0, 475) / 474.0]))
QGRID = np.clip(QGRID + 1e-12, 0, 1.0)


def curve_xval(g, fs, maskA):
    fa, ct = [], []
    for q in QGRID:
        a, c = xval(g, fs, maskA, q)
        fa.append(a); ct.append(c)
    return np.array(fa), np.array(ct)


def curve_oracle(g, fs):
    fa, ct = [], []
    for q in QGRID:
        a, c = oracle(g, fs, q)
        fa.append(a); ct.append(c)
    return np.array(fa), np.array(ct)


def price(fa, ct, t):
    ok = ct >= t - 1e-12
    if not ok.any():
        return 1.0
    return float(np.min(np.where(ok, fa, np.inf)))


price_cell = {}
agg_curves = {}
for st in STATS:
    XA = []; XC = []; OA = []; OC = []
    for (m, d) in cells:
        g = G[(m, d)][st]; fs = [F[(m, d)][r][st] for r in range(R)]
        fa, ct = curve_xval(g, fs, maskA_alt)
        ofa, oct_ = curve_oracle(g, fs)
        XA.append(fa); XC.append(ct); OA.append(ofa); OC.append(oct_)
        price_cell["%s|%s|%s" % (m, d, st)] = {
            "xval": {("caught%g" % t): price(fa, ct, t) for t in TARGETS},
            "oracle": {("caught%g" % t): price(ofa, oct_, t) for t in TARGETS}}
    XA = np.array(XA); XC = np.array(XC); OA = np.array(OA); OC = np.array(OC)
    agg_curves[st] = {
        "mean_of_percell_price_xval": {("caught%g" % t): float(np.mean(
            [price_cell["%s|%s|%s" % (m, d, st)]["xval"]["caught%g" % t] for (m, d) in cells]))
            for t in TARGETS},
        "mean_of_percell_price_oracle": {("caught%g" % t): float(np.mean(
            [price_cell["%s|%s|%s" % (m, d, st)]["oracle"]["caught%g" % t] for (m, d) in cells]))
            for t in TARGETS},
        "median_of_percell_price_xval": {("caught%g" % t): float(np.median(
            [price_cell["%s|%s|%s" % (m, d, st)]["xval"]["caught%g" % t] for (m, d) in cells]))
            for t in TARGETS},
        "range_of_percell_price_xval": {("caught%g" % t): [float(np.min(
            [price_cell["%s|%s|%s" % (m, d, st)]["xval"]["caught%g" % t] for (m, d) in cells])),
            float(np.max([price_cell["%s|%s|%s" % (m, d, st)]["xval"]["caught%g" % t] for (m, d) in cells]))]
            for t in TARGETS},
        "macro_curve_price_xval": {("caught%g" % t): price(XA.mean(0), XC.mean(0), t) for t in TARGETS},
        "macro_curve_price_oracle": {("caught%g" % t): price(OA.mean(0), OC.mean(0), t) for t in TARGETS},
    }
res["price_cell"] = price_cell
res["price_agg"] = agg_curves
print("price curves done (%.1fs)" % (time.time() - t0))

# -------------------------------------------------------- S1 integer rule
kmax = int(slen.max())
kt = {}
for k in range(0, kmax + 1):
    fas = [float(np.mean(G[(m, d)]["S1_COUNT"] >= k)) for (m, d) in cells]
    cts = [float(np.mean([np.mean(F[(m, d)][r]["S1_COUNT"] >= k) for r in range(R)]))
           for (m, d) in cells]
    kt[k] = {"fa": float(np.mean(fas)), "caught": float(np.mean(cts))}
best5 = min([k for k in kt if kt[k]["fa"] <= 0.05])
bestJ = max(kt, key=lambda k: kt[k]["caught"] - kt[k]["fa"])
res["count_rule"] = {"table": {str(k): kt[k] for k in sorted(kt)},
                     "best_k_under_fa5": best5, "best_k_under_fa5_point": kt[best5],
                     "best_k_youden": bestJ, "best_k_youden_point": kt[bestJ],
                     "youden_J": kt[bestJ]["caught"] - kt[bestJ]["fa"]}

# ------------------------------------------------- split-choice sensitivity
rng = np.random.default_rng(4242)
NSPLIT = 200
splitres = {st: {q: [] for q in QS} for st in STATS}
splitcell = {st: {q: np.zeros(len(cells)) for q in QS} for st in STATS}
splitcell_fa = {st: {q: np.zeros(len(cells)) for q in QS} for st in STATS}
splitgrp = {}
for b in range(NSPLIT):
    perm = rng.permutation(NU)[:10]
    mk = np.isin(suser, perm)
    for st in STATS:
        for q in QS:
            fv = [xval(G[(m, d)][st], [F[(m, d)][r][st] for r in range(R)], mk, q)
                  for (m, d) in cells]
            v = [x[1] for x in fv]
            splitres[st][q].append(float(np.mean(v)))
            splitcell[st][q] += np.array(v) / NSPLIT
            splitcell_fa[st][q] += np.array([x[0] for x in fv]) / NSPLIT
            splitgrp.setdefault((st, q), []).append(np.array(v))
res["split_marginal_per_cell"] = {
    st: {("q%g" % q): {"%s|%s" % (m, d): float(splitcell[st][q][i])
                       for i, (m, d) in enumerate(cells)} for q in QS} for st in STATS}
res["split_marginal_per_cell_fa"] = {
    st: {("q%g" % q): {"%s|%s" % (m, d): float(splitcell_fa[st][q][i])
                       for i, (m, d) in enumerate(cells)} for q in QS} for st in STATS}
res["split_marginal_breakouts"] = {}
for st in STATS:
    b_ = {}
    for q in QS:
        d_ = {}
        for m in MODALITIES:
            ii = [i for i, (mm, dd) in enumerate(cells) if mm == m]
            d_["MOD:" + m] = float(np.mean(splitcell[st][q][ii]))
        for dd0 in DETECTORS:
            ii = [i for i, (mm, dd) in enumerate(cells) if dd == dd0]
            d_["DET:" + dd0] = float(np.mean(splitcell[st][q][ii]))
        d_["ALL"] = float(np.mean(splitcell[st][q]))
        d_["ALL_fa"] = float(np.mean(splitcell_fa[st][q]))
        M = np.vstack(splitgrp[(st, q)])      # (NSPLIT, n_cells)
        sd = {}
        for m in MODALITIES:
            ii = [i for i, (mm, dd) in enumerate(cells) if mm == m]
            sd["MOD:" + m] = float(np.std(M[:, ii].mean(1), ddof=1))
        for dd0 in DETECTORS:
            ii = [i for i, (mm, dd) in enumerate(cells) if dd == dd0]
            sd["DET:" + dd0] = float(np.std(M[:, ii].mean(1), ddof=1))
        sd["ALL"] = float(np.std(M.mean(1), ddof=1))
        d_["_split_sd"] = sd
        b_["q%g" % q] = d_
    res["split_marginal_breakouts"][st] = b_
res["split_sensitivity"] = {
    st: {("q%g" % q): {"mean": float(np.mean(splitres[st][q])),
                       "sd": float(np.std(splitres[st][q], ddof=1)),
                       "p2.5": float(np.percentile(splitres[st][q], 2.5)),
                       "p97.5": float(np.percentile(splitres[st][q], 97.5)),
                       "min": float(np.min(splitres[st][q])),
                       "max": float(np.max(splitres[st][q]))}
         for q in QS} for st in STATS}

# -------------------------------------------------------------- breakouts
res["breakouts"] = {}
for st in STATS:
    b = {}
    for m in MODALITIES:
        sel = [(m, d) for d in DETECTORS]
        b["MOD:" + m] = {
            "xval_alt@0.05": float(np.mean([per_cell["%s|%s" % c][st]["q0.05"]["xval_alt_caught"] for c in sel])),
            "xval_f10@0.05": float(np.mean([per_cell["%s|%s" % c][st]["q0.05"]["xval_f10_caught"] for c in sel])),
            "auc": float(np.mean([per_cell["%s|%s" % c][st]["auc"] for c in sel]))}
    for d in DETECTORS:
        sel = [(m, d) for m in MODALITIES]
        b["DET:" + d] = {
            "xval_alt@0.05": float(np.mean([per_cell["%s|%s" % c][st]["q0.05"]["xval_alt_caught"] for c in sel])),
            "xval_f10@0.05": float(np.mean([per_cell["%s|%s" % c][st]["q0.05"]["xval_f10_caught"] for c in sel])),
            "auc": float(np.mean([per_cell["%s|%s" % c][st]["auc"] for c in sel]))}
    res["breakouts"][st] = b

json.dump(res, open(os.path.join(OUT, "referee_%s.json" % TAG), "w"), indent=1)
print("\n=== REFEREE AGGREGATE (repo release, R=%d, third seed scheme) ===" % R)
print("%-11s %8s | %-24s | %-24s | %-16s" % ("stat", "AUC", "xval ALT split c@5/c@1",
                                             "xval F10 split c@5/c@1", "oracle c@5/c@1"))
for st in STATS:
    a = res["aggregate"][st]
    print("%-11s %8.4f | %.4f / %.4f  (fa %.4f) | %.4f / %.4f  (fa %.4f) | %.4f / %.4f"
          % (st, a["auc"], a["xval_alt_caught@q0.05"], a["xval_alt_caught@q0.01"],
             a["xval_alt_fa@q0.05"], a["xval_f10_caught@q0.05"], a["xval_f10_caught@q0.01"],
             a["xval_f10_fa@q0.05"], a["oracle_caught@q0.05"], a["oracle_caught@q0.01"]))
print("\n=== PRICE OF DETECTION, two conventions ===")
for st in STATS:
    p = res["price_agg"][st]
    print("%-11s meanPerCell(xval) 50/80/95 = %.4f %.4f %.4f | macroCurve(xval) = %.4f %.4f %.4f | medianPerCell = %.4f %.4f %.4f"
          % (st, p["mean_of_percell_price_xval"]["caught0.5"], p["mean_of_percell_price_xval"]["caught0.8"],
             p["mean_of_percell_price_xval"]["caught0.95"], p["macro_curve_price_xval"]["caught0.5"],
             p["macro_curve_price_xval"]["caught0.8"], p["macro_curve_price_xval"]["caught0.95"],
             p["median_of_percell_price_xval"]["caught0.5"], p["median_of_percell_price_xval"]["caught0.8"],
             p["median_of_percell_price_xval"]["caught0.95"]))
print("\n=== S1 INTEGER COUNT RULE ===")
for k in range(0, 14):
    print("  k=%2d  fa %.4f  caught %.4f" % (k, kt[k]["fa"], kt[k]["caught"]))
print("  best k under 5%% macro FA: %d -> %s" % (best5, kt[best5]))
print("  Youden-best k: %d -> %s  J=%.4f" % (bestJ, kt[bestJ], kt[bestJ]["caught"] - kt[bestJ]["fa"]))
print("\n=== SPLIT SENSITIVITY (200 random balanced 10/10 user splits) ===")
for st in STATS:
    s = res["split_sensitivity"][st]["q0.05"]
    s1 = res["split_sensitivity"][st]["q0.01"]
    print("  %-11s q=5%%: mean %.4f sd %.4f [%.4f,%.4f]  | q=1%%: mean %.4f sd %.4f [%.4f,%.4f]"
          % (st, s["mean"], s["sd"], s["p2.5"], s["p97.5"], s1["mean"], s1["sd"], s1["p2.5"], s1["p97.5"]))
print("\ntotal %.1fs" % (time.time() - t0))
