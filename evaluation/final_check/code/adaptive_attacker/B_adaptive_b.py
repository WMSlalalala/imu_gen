#!/usr/bin/env python
"""
Adaptive-attacker curve (Part A), duration confound (Part B) and false-alarm
incidence (Part C) for the session-level defence over the PUBLISHED 90-cell release.

Agent B implementation.  Pure score-space / duration-space arithmetic, no GPU,
no model loading.  Every cell is resolved through release_cell_map.json.

Run:  python adaptive_b.py            (builds cache/ first via build_cache.py)
"""
import os, sys, json, math, time, itertools, zlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

CELLMAP_PATH = "/mnt/share/mwang49/data7/code/baselines/release_cell_map.json"

ACTIONS = ["tap", "scroll", "swipe", "pinch", "keystroke"]
AIDX = {a: i for i, a in enumerate(ACTIONS)}
MODALITIES = ["imu_only", "trajectory_xytime", "imu_trajectory_xytime"]
DETECTORS = ["authconformer", "behaveformer_stdat", "hmog_style_rf",
             "hmog_style_svm", "paper_svm", "paper_xgboost"]
COMBOS = [(m, d) for m in MODALITIES for d in DETECTORS]

R_VALUES = [1, 2, 5, 10, 20]
N_SEEDS = 20
N_BOOT = 2000
ALPHAS = [0.05, 0.01]
NQ = 41                      # quantile edges -> 40 bins
SESSIONS_PER_DAY = 12.0      # stated parameter for the lockout cadence
STATS = ["llr_sum", "llr_mean", "B0_count", "B1_meanscore", "B2_durllr"]
PRICE_TARGETS = [0.50, 0.80, 0.95]
AGRID = np.unique(np.concatenate([np.logspace(-3, 0, 150), np.linspace(0.0, 1.0, 101),
                                  np.array(ALPHAS)]))

# ----------------------------------------------------------------------------- loaders
def _log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_all():
    cm = json.load(open(CELLMAP_PATH))["cells"]
    C = np.load(os.path.join(CACHE, "cells.npz"), allow_pickle=True)
    bundles = sorted({v["bundle"] for v in cm.values()})
    B = {b: np.load(os.path.join(CACHE, f"bundle_{b}.npz"), allow_pickle=True) for b in bundles}
    bd = np.load(os.path.join(CACHE, "bindings.npz"), allow_pickle=True)
    return cm, C, B, bd


def build_index(cm, C, B, bd):
    """Canonical event indexing shared by every cell and every bundle."""
    # ---- test users
    any_cell = "tap__imu_only__paper_xgboost"
    users = sorted(set(C[any_cell + "|user_id"].tolist()))
    for a in ACTIONS:
        for m, d in COMBOS:
            pass
    UIDX = {u: i for i, u in enumerate(users)}
    NU = len(users)

    # ---- fake pool: canonical order = sorted event_id within (action, user)
    pool_eid = np.empty((5, NU, 200), dtype=object)
    for a in ACTIONS:
        cell = f"{a}__imu_only__paper_xgboost"
        eid = C[cell + "|event_id"]; lab = C[cell + "|label"]; usr = C[cell + "|user_id"]
        fe, fu = eid[lab == 1], usr[lab == 1]
        for u in users:
            sel = np.sort(fe[fu == u])
            assert len(sel) == 200, (a, u, len(sel))
            pool_eid[AIDX[a], UIDX[u]] = sel
    pool_pos = {}
    for ai in range(5):
        for ui in range(NU):
            for p, e in enumerate(pool_eid[ai, ui]):
                pool_pos[e] = (ai, ui, p)

    # ---- genuine events (scored): gid space
    gen_rows = []           # (event_id, cluster_id, aidx, uidx)
    for a in ACTIONS:
        cell = f"{a}__imu_only__paper_xgboost"
        eid = C[cell + "|event_id"]; lab = C[cell + "|label"]; usr = C[cell + "|user_id"]
        ge, gu = eid[lab == 0], usr[lab == 0]
        o = np.argsort(ge)
        for e, u in zip(ge[o].tolist(), gu[o].tolist()):
            gen_rows.append((e, e[len("genuine-"):], AIDX[a], UIDX[u]))
    gid_of_eid = {r[0]: i for i, r in enumerate(gen_rows)}
    gid_of_cluster = {r[1]: i for i, r in enumerate(gen_rows)}
    gen_action = np.array([r[2] for r in gen_rows], np.int64)
    gen_user = np.array([r[3] for r in gen_rows], np.int64)
    NG = len(gen_rows)

    # ---- per-cell score arrays
    cellS = {}
    for cell in cm:
        eid = C[cell + "|event_id"]; sc = C[cell + "|score"]; lab = C[cell + "|label"]
        g = np.full(NG, np.nan); f = np.full((5, NU, 200), np.nan)
        for e, s, l in zip(eid.tolist(), sc.tolist(), lab.tolist()):
            if l == 0:
                g[gid_of_eid[e]] = s
            else:
                ai, ui, p = pool_pos[e]
                f[ai, ui, p] = s
        a_of_cell = cell.split("__")[0]
        ai = AIDX[a_of_cell]
        assert not np.isnan(f[ai]).any()
        cellS[cell] = dict(gen=g, fake=f, action=ai,
                           tau=json.load(open(cm[cell]["thresholds"]))["frr5"],
                           bundle=cm[cell]["bundle"])

    # ---- per-bundle duration arrays (test) + dev arrays
    bundleD, devD = {}, {}
    for b, z in B.items():
        eid = z["event_id"]; dur = z["duration"]; lab = z["label"]; spl = z["split"]
        act = z["action"]
        g = np.full(NG, np.nan); f = np.full((5, NU, 200), np.nan)
        tm = spl == "test"
        for e, dd, l in zip(eid[tm].tolist(), dur[tm].tolist(), lab[tm].tolist()):
            if l == 0:
                j = gid_of_eid.get(e)
                if j is not None:
                    g[j] = dd
            else:
                pp = pool_pos.get(e)
                if pp is not None:
                    ai, ui, p = pp
                    f[ai, ui, p] = dd
        bundleD[b] = dict(gen=g, fake=f)
        dm = spl == "development"
        dv = {}
        for a in ACTIONS:
            sel = dm & (act == a)
            dv[a] = dict(gen=dur[sel & (lab == 0)].copy(), fake=dur[sel & (lab == 1)].copy())
        devD[b] = dv
    for b in bundleD:
        assert not np.isnan(bundleD[b]["fake"]).any()
        assert not np.isnan(bundleD[b]["gen"]).any()

    # ---- A2 self-consistency distances per (modality, action)
    FEAT_SETS = {"imu_only": ["feat_imu"],
                 "trajectory_xytime": ["feat_traj"],
                 "imu_trajectory_xytime": ["feat_imu", "feat_traj"]}
    bundle_of = {(cell.split("__")[0], cell.split("__")[1]): cm[cell]["bundle"] for cell in cm}
    a2c = {m: np.zeros((5, NU, 200)) for m in MODALITIES}
    a2k = {m: np.zeros((5, NU, 200)) for m in MODALITIES}
    for m in MODALITIES:
        for a in ACTIONS:
            b = bundle_of[(a, m)]
            z = B[b]
            eid = z["event_id"]; lab = z["label"]; spl = z["split"]; act = z["action"]
            sel = (spl == "test") & (lab == 1) & (act == a)
            F = np.concatenate([z[k][sel] for k in FEAT_SETS[m]], axis=1).astype(np.float64)
            E = eid[sel]
            ui = np.array([pool_pos[e][1] for e in E.tolist()])
            pi = np.array([pool_pos[e][2] for e in E.tolist()])
            for u in range(NU):
                msk = ui == u
                X = F[msk]; pp = pi[msk]
                mu = X.mean(0); sd = X.std(0); sd[sd < 1e-12] = 1.0
                Z = (X - mu) / sd
                dc = np.linalg.norm(Z, axis=1)
                q = (Z * Z).sum(1)
                D = np.sqrt(np.maximum(q[:, None] + q[None, :] - 2.0 * (Z @ Z.T), 0.0))
                np.fill_diagonal(D, np.inf)
                k = 10
                dk = np.sort(D, axis=1)[:, :k].mean(1)
                a2c[m][AIDX[a], u, pp] = dc
                a2k[m][AIDX[a], u, pp] = dk

    # ---- sessions from the FULL genuine binding inventory (test split)
    tm = bd["split"] == "test"
    b_cluster = bd["source_cluster_id"][tm]; b_sid = bd["session_id"][tm]
    b_act = bd["action"][tm]; b_ord = bd["ordinal"][tm]; b_start = bd["start_sample"][tm]
    b_user = bd["user_id"][tm]
    sess_slots = {}
    sess_total = {}
    sess_actions_full = {}
    for i in range(len(b_sid)):
        s = b_sid[i]
        sess_total[s] = sess_total.get(s, 0) + 1
        sess_actions_full.setdefault(s, set()).add(b_act[i])
        gid = gid_of_cluster.get(b_cluster[i])
        if gid is not None:
            sess_slots.setdefault(s, []).append((int(b_ord[i]), gid, int(b_start[i]), UIDX[b_user[i]]))

    sessions = {"touch": [], "keystroke": []}
    dropped = {"no_scored_slot": 0, "mixed_keystroke_touch": 0}
    covariate = {"touch": [], "keystroke": []}
    for s, tot in sess_total.items():
        acts = sess_actions_full[s]
        if "keystroke" in acts and len(acts) > 1:
            dropped["mixed_keystroke_touch"] += 1
            continue
        reg = "keystroke" if acts == {"keystroke"} else "touch"
        sl = sorted(sess_slots.get(s, []))
        if not sl:
            dropped["no_scored_slot"] += 1
            continue
        sessions[reg].append(dict(sid=s, user=sl[0][3],
                                  gids=[x[1] for x in sl],
                                  starts=[x[2] for x in sl],
                                  n_total_slots=tot))
        covariate[reg].append((len(sl), tot))

    # genuine pool per (action, user), for the F8 same-mechanics re-assembly control
    cnt = np.zeros((5, NU), np.int64)
    for ai, ui in zip(gen_action.tolist(), gen_user.tolist()):
        cnt[ai, ui] += 1
    Nmax = int(cnt.max())
    gen_pool = np.zeros((5, NU, Nmax), np.int64)
    fill = np.zeros((5, NU), np.int64)
    for g in range(NG):
        ai, ui = gen_action[g], gen_user[g]
        gen_pool[ai, ui, fill[ai, ui]] = g
        fill[ai, ui] += 1

    return dict(gen_pool=gen_pool, gen_pool_n=cnt,
                users=users, UIDX=UIDX, NU=NU, NG=NG, gen_action=gen_action,
                gen_user=gen_user, cellS=cellS, bundleD=bundleD, devD=devD,
                a2c=a2c, a2k=a2k, sessions=sessions, dropped=dropped,
                covariate=covariate, bundle_of=bundle_of)


# ----------------------------------------------------------------------------- quantile LLR
def fit_qllr(gen_vals, fake_vals, nq=NQ):
    both = np.concatenate([gen_vals, fake_vals])
    edges = np.unique(np.quantile(both, np.linspace(0, 1, nq)))
    if len(edges) < 3:
        edges = np.array([both.min() - 1.0, both.max() + 1.0])
    inner = edges[1:-1]
    nb = len(inner) + 1
    bg = np.bincount(np.searchsorted(inner, gen_vals, side="right"), minlength=nb).astype(float)
    bf = np.bincount(np.searchsorted(inner, fake_vals, side="right"), minlength=nb).astype(float)
    pg = (bg + 0.5) / (bg.sum() + 0.5 * nb)
    pf = (bf + 0.5) / (bf.sum() + 0.5 * nb)
    return dict(inner=inner, llr=np.log(pf) - np.log(pg))


def apply_qllr(tab, vals):
    return tab["llr"][np.searchsorted(tab["inner"], vals, side="right")]


# ----------------------------------------------------------------------------- regime tensors
class Regime:
    """Flattened slot arrays for one regime, shared by all combos."""

    def __init__(self, sess_list, gen_action, gen_user, NU):
        self.sessions = sess_list
        self.n_sess = len(sess_list)
        self.sess_user = np.array([s["user"] for s in sess_list], np.int64)
        self.sess_len = np.array([len(s["gids"]) for s in sess_list], np.int64)
        self.sess_scored_frac = np.array([len(s["gids"]) / s["n_total_slots"] for s in sess_list])
        slot_sess, slot_gid, slot_start = [], [], []
        for i, s in enumerate(sess_list):
            slot_sess += [i] * len(s["gids"])
            slot_gid += s["gids"]
            slot_start += s["starts"]
        self.slot_sess = np.array(slot_sess, np.int64)
        self.slot_gid = np.array(slot_gid, np.int64)
        self.slot_start = np.array(slot_start, np.float64) / 100.0     # seconds, 100 Hz
        self.slot_action = gen_action[self.slot_gid]
        self.slot_user = gen_user[self.slot_gid]
        self.M = len(self.slot_gid)
        # (session, action) groups + rank within group
        key = self.slot_sess * 5 + self.slot_action
        uk, inv = np.unique(key, return_inverse=True)
        self.group = inv
        self.n_groups = len(uk)
        self.group_user = np.zeros(self.n_groups, np.int64)
        self.group_action = np.zeros(self.n_groups, np.int64)
        self.group_user[inv] = self.slot_user
        self.group_action[inv] = self.slot_action
        order = np.lexsort((np.arange(self.M), inv))
        rank = np.zeros(self.M, np.int64)
        cnt = np.zeros(self.n_groups, np.int64)
        for j in order:
            g = inv[j]; rank[j] = cnt[g]; cnt[g] += 1
        self.rank = rank
        self.group_count = cnt
        self.n_slots_per_sess = np.bincount(self.slot_sess, minlength=self.n_sess).astype(float)

    def agg_sum(self, w):
        return np.bincount(self.slot_sess, weights=w, minlength=self.n_sess)

    def agg_mean(self, w):
        return self.agg_sum(w) / self.n_slots_per_sess


# ----------------------------------------------------------------------------- helpers
def pvals(stat, null_sorted):
    n = len(null_sorted)
    ge = n - np.searchsorted(null_sorted, stat, side="left")
    return (1.0 + ge) / (1.0 + n)


def curve_counts(u, grid):
    """#(u <= alpha) for each alpha in grid."""
    us = np.sort(u)
    return np.searchsorted(us, grid, side="right")


def inverse_price(caught_curve, frr_curve, targets):
    out = {}
    for t in targets:
        idx = np.nonzero(caught_curve >= t)[0]
        out[f"frr_for_caught_{int(t*100)}"] = float(frr_curve[idx[0]]) if len(idx) else None
    return out


def boot_rate(num_u, den_u, bmat):
    n = num_u[bmat].sum(1); d = den_u[bmat].sum(1)
    r = np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)
    return float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


def betabinom_mom(k, n):
    """Method-of-moments beta-binomial fit over users."""
    k = np.asarray(k, float); n = np.asarray(n, float)
    p = k / n
    m = np.average(p, weights=n)
    if m <= 0 or m >= 1:
        return dict(alpha=None, beta=None, mean=float(m), rho=None, note="degenerate")
    v = np.average((p - m) ** 2, weights=n)
    nbar = n.mean()
    denom = m * (1 - m)
    rho = (v - denom / nbar) / (denom * (1 - 1.0 / nbar)) if denom > 0 else 0.0
    rho = float(min(max(rho, 1e-6), 0.999))
    s = (1 - rho) / rho
    return dict(alpha=float(m * s), beta=float((1 - m) * s), mean=float(m), rho=rho)


def _days(rate):
    """Mean days between false lockouts at SESSIONS_PER_DAY sessions/day."""
    if rate is None or rate <= 0:
        return None
    return float(1.0 / (rate * SESSIONS_PER_DAY))


def betabinom_quantile_p(fit, q):
    """Quantile of the per-user Bernoulli rate under the fitted Beta."""
    if fit.get("alpha") is None:
        return None
    from scipy.stats import beta as _beta
    return float(_beta.ppf(q, fit["alpha"], fit["beta"]))


# ============================================================================= PART A
GLOB = {}


def _combine_gen(per_action_vec, gen_action):
    """per_action_vec[a] is a length-NG array valid only where gen_action==a."""
    out = np.zeros(len(gen_action))
    for ai in range(5):
        sel = gen_action == ai
        out[sel] = per_action_vec[ai][sel]
    return out


def make_regimes(IDX):
    return {reg: Regime(IDX["sessions"][reg], IDX["gen_action"], IDX["gen_user"], IDX["NU"])
            for reg in ("touch", "keystroke")}


def slot_lookup_llr(R, tabs, vals):
    out = np.empty(R.M)
    for ai in range(5):
        m = R.slot_action == ai
        if m.any():
            out[m] = apply_qllr(tabs[ai], vals[m])
    return out


def session_stats(R, sc, du, tabs_by_src, durtab, tau_slot, slot_src):
    """Return dict stat -> (n_sess,) for the cross-fitted source and for the oracle."""
    llr = {}
    for src in (0, 1, 2):
        llr[src] = slot_lookup_llr(R, tabs_by_src[src], sc)
    llr_cf = np.where(slot_src == 0, llr[0], llr[1])
    dl = slot_lookup_llr(R, durtab, du)
    caught = (sc >= tau_slot).astype(float)
    out_cf = dict(llr_sum=R.agg_sum(llr_cf), llr_mean=R.agg_mean(llr_cf),
                  B0_count=R.agg_sum(caught), B1_meanscore=R.agg_mean(sc),
                  B2_durllr=R.agg_sum(dl))
    out_or = dict(out_cf)
    out_or["llr_sum"] = R.agg_sum(llr[2]); out_or["llr_mean"] = R.agg_mean(llr[2])
    return out_cf, out_or


def run_combo(args):
    mod, det = args
    IDX = GLOB["IDX"]; REG = GLOB["REG"]; bmat = GLOB["bmat"]
    NU = IDX["NU"]; users = IDX["users"]
    gen_action = IDX["gen_action"]
    fold_of_user = np.array([i % 2 for i in range(NU)])
    cells = {a: IDX["cellS"][f"{a}__{mod}__{det}"] for a in ACTIONS}

    gen_score = _combine_gen([cells[ACTIONS[ai]]["gen"] for ai in range(5)], gen_action)
    fake_score = np.stack([cells[ACTIONS[ai]]["fake"][ai] for ai in range(5)])       # (5,NU,200)
    gen_dur = _combine_gen([IDX["bundleD"][IDX["bundle_of"][(ACTIONS[ai], mod)]]["gen"]
                            for ai in range(5)], gen_action)
    fake_dur = np.stack([IDX["bundleD"][IDX["bundle_of"][(ACTIONS[ai], mod)]]["fake"][ai]
                         for ai in range(5)])
    tau = np.array([cells[a]["tau"] for a in ACTIONS])

    # ---- duration LLR fitted on DEVELOPMENT users only (F1)
    durtab = []
    for ai, a in enumerate(ACTIONS):
        dv = IDX["devD"][IDX["bundle_of"][(a, mod)]][a]
        durtab.append(fit_qllr(dv["gen"], dv["fake"]))

    # ---- score LLR tables per calibration source
    tabs_by_src = {}
    for src in (0, 1, 2):
        cal_users = np.arange(NU) if src == 2 else np.nonzero(fold_of_user == src)[0]
        tt = []
        for ai in range(5):
            gsel = (gen_action == ai) & np.isin(IDX["gen_user"], cal_users)
            gv = gen_score[gsel]
            fv = fake_score[ai][cal_users].ravel()
            tt.append(fit_qllr(gv, fv))
        tabs_by_src[src] = tt

    results = []
    partC = []
    PU = dict(caught=[], den=[], cfg=[], gen_alarm={}, gen_den={}, alpha90={})
    for reg in ("touch", "keystroke"):
        R = REG[reg]
        if R.n_sess == 0:
            continue
        slot_src = 1 - fold_of_user[R.slot_user]
        sess_src = 1 - fold_of_user[R.sess_user]
        tau_slot = tau[R.slot_action]
        g_sc = gen_score[R.slot_gid]; g_du = gen_dur[R.slot_gid]
        gcf, gor = session_stats(R, g_sc, g_du, tabs_by_src, durtab, tau_slot, slot_src)

        # nulls: calibration genuine sessions
        nulls_cf, nulls_or = {}, {}
        for st in STATS:
            nulls_cf[st] = [np.sort(gcf[st][R.sess_user % 2 == s]) for s in (0, 1)]
            nulls_or[st] = np.sort(gor[st])

        # genuine p-values
        pg_cf, pg_or = {}, {}
        for st in STATS:
            p0 = pvals(gcf[st], nulls_cf[st][0]); p1 = pvals(gcf[st], nulls_cf[st][1])
            pg_cf[st] = np.where(sess_src == 0, p0, p1)
            pg_or[st] = pvals(gor[st], nulls_or[st])

        n_gen_u = np.bincount(R.sess_user, minlength=NU).astype(float)

        # ---- Part C: per-user FRR curve, alpha90, operating points
        alpha90 = {}
        peruser = {}
        for st in STATS:
            cnt = np.zeros((len(AGRID), NU))
            for u in range(NU):
                m = R.sess_user == u
                cnt[:, u] = curve_counts(pg_cf[st][m], AGRID)
            rate = cnt / np.maximum(n_gen_u, 1e-12)
            p90 = np.percentile(rate, 90, axis=1)
            ok = np.nonzero(p90 <= 0.01)[0]
            a90 = float(AGRID[ok[-1]]) if len(ok) else 0.0
            alpha90[st] = a90
            peruser[st] = dict(rate_curve=rate, n_gen_u=n_gen_u)
        alist = ALPHAS + [alpha90[st] for st in STATS]

        # genuine per-user alarm counts at the operating points
        gen_op = {}
        for st in STATS:
            gen_op[st] = {}
            for tag, al in [("0.05", 0.05), ("0.01", 0.01), ("a90", alpha90[st])]:
                al_eff = al
                alarm = (pg_cf[st] <= al_eff).astype(float)
                ku = np.bincount(R.sess_user, weights=alarm, minlength=NU)
                alarm_or = (pg_or[st] <= al_eff).astype(float)
                ku_or = np.bincount(R.sess_user, weights=alarm_or, minlength=NU)
                gen_op[st][tag] = dict(alpha=al_eff, k_u=ku, n_u=n_gen_u,
                                       frr=float(ku.sum() / n_gen_u.sum()),
                                       frr_oracle=float(ku_or.sum() / n_gen_u.sum()))
            gen_op[st]["curve"] = curve_counts(pg_cf[st], AGRID) / R.n_sess

        # ---- Part C record
        for st in STATS:
            rec = dict(modality=mod, detector=det, regime=reg, statistic=st,
                       n_sessions=int(R.n_sess), n_users=int(NU),
                       sessions_per_day=SESSIONS_PER_DAY)
            for tag in ("0.05", "0.01", "a90"):
                d = gen_op[st][tag]
                pu = d["k_u"] / np.maximum(d["n_u"], 1e-12)
                fit = betabinom_mom(d["k_u"], np.maximum(d["n_u"], 1))
                rec[tag] = dict(
                    alpha=d["alpha"], pooled_frr=d["frr"], pooled_frr_oracle=d["frr_oracle"],
                    per_user_frr=dict(min=float(pu.min()), median=float(np.median(pu)),
                                      p90=float(np.percentile(pu, 90)), max=float(pu.max())),
                    betabinom=fit,
                    betabinom_p50=betabinom_quantile_p(fit, 0.50),
                    betabinom_p90=betabinom_quantile_p(fit, 0.90),
                    betabinom_p99=betabinom_quantile_p(fit, 0.99),
                    lockout_days_fitted_median_user=_days(betabinom_quantile_p(fit, 0.50)),
                    lockout_days_fitted_p90_user=_days(betabinom_quantile_p(fit, 0.90)),
                    lockout_days_between_median_user=(
                        float(1.0 / (np.median(pu) * SESSIONS_PER_DAY)) if np.median(pu) > 0 else None),
                    lockout_days_between_p90_user=(
                        float(1.0 / (np.percentile(pu, 90) * SESSIONS_PER_DAY))
                        if np.percentile(pu, 90) > 0 else None),
                )
            partC.append(rec)

        # ---- selection configurations
        rank_keys = {}
        rank_keys[("A0_uniform", None)] = None
        for sm, sd in COMBOS:
            if (sm, sd) == (mod, det):
                continue
            sur = np.stack([IDX["cellS"][f"{ACTIONS[ai]}__{sm}__{sd}"]["fake"][ai] for ai in range(5)])
            rank_keys[("A1_surrogate", f"{sm}|{sd}")] = sur
        # knowledge-free ensembles of surrogates: the attacker averages the pool RANK
        # given by every PAD it owns, so it never has to guess which surrogate transfers.
        def pool_rank(x):
            o = np.argsort(np.argsort(x, axis=2), axis=2).astype(np.float64)
            return o / (x.shape[2] - 1.0)
        ens_all, ens_same, n_all, n_same = 0.0, 0.0, 0, 0
        for (sm, sd) in COMBOS:
            if (sm, sd) == (mod, det):
                continue
            rk = pool_rank(np.stack([IDX["cellS"][f"{ACTIONS[ai]}__{sm}__{sd}"]["fake"][ai]
                                     for ai in range(5)]))
            ens_all = ens_all + rk; n_all += 1
            if sm == mod:
                ens_same = ens_same + rk; n_same += 1
        rank_keys[("A1_ensemble_all17", None)] = ens_all / n_all
        rank_keys[("A1_ensemble_same_modality5", None)] = ens_same / n_same
        rank_keys[("A2_centroid", None)] = IDX["a2c"][mod]
        rank_keys[("A2_knn", None)] = IDX["a2k"][mod]
        rank_keys[("A2_centroid_reverse", None)] = -IDX["a2c"][mod]
        rank_keys[("A3_oracle", None)] = fake_score

        configs = []
        configs.append(("A0_uniform", None, 1))
        for (rule, sur) in rank_keys:
            if rule == "A0_uniform":
                continue
            for r in R_VALUES[1:]:
                configs.append((rule, sur, r))

        # ---- F8 control: genuine material put through the IDENTICAL session-assembly
        gp = IDX["gen_pool"]; gpn = IDX["gen_pool_n"]
        Nmax = gp.shape[2]
        cu = {st: {t: np.zeros(NU) for t in ("0.05", "0.01", "a90")} for st in STATS}
        n_fake_u = np.bincount(R.sess_user, minlength=NU).astype(float) * N_SEEDS
        gk_n = gpn[R.group_action, R.group_user]
        for sd_ in range(N_SEEDS):
            rr = np.random.default_rng(int(zlib.crc32(f"ctl|{mod}|{det}|{reg}|{sd_}".encode())))
            K = rr.random((R.n_groups, Nmax))
            K[np.arange(Nmax)[None, :] >= gk_n[:, None]] = np.inf
            om = np.argsort(K, axis=1)
            col = om[R.group, R.rank % gk_n[R.group]]
            gids = gp[R.slot_action, R.slot_user, col]
            cs = gen_score[gids]; cd = gen_dur[gids]
            ccf, _ = session_stats(R, cs, cd, tabs_by_src, durtab, tau_slot, slot_src)
            for st in STATS:
                p0 = pvals(ccf[st], nulls_cf[st][0]); p1 = pvals(ccf[st], nulls_cf[st][1])
                pc_ = np.where(sess_src == 0, p0, p1)
                for tag, al in (("0.05", 0.05), ("0.01", 0.01), ("a90", alpha90[st])):
                    cu[st][tag] += np.bincount(R.sess_user, weights=(pc_ <= al).astype(float),
                                               minlength=NU)
        ctl = dict(modality=mod, detector=det, regime=reg, rule="C0_genuine_reassembled_F8_control",
                   surrogate=None, r=1, pool_kept=int(Nmax), generated_per_submitted=1.0,
                   surrogate_same_modality=None, n_sessions=int(R.n_sess), n_slots=int(R.M),
                   forced_reuse_slots_per_draw=0, forced_reuse_frac=0.0,
                   per_event_FAR_at_dev_frr5=float((gen_score[R.slot_gid] < tau_slot).mean()),
                   stats={})
        for st in STATS:
            d = {}
            for tag in ("0.05", "0.01", "a90"):
                lo, hi = boot_rate(cu[st][tag], n_fake_u, bmat)
                g = gen_op[st][tag]
                d[tag] = dict(alpha=float(alpha90[st] if tag == "a90" else float(tag)),
                              caught=float(cu[st][tag].sum() / n_fake_u.sum()), caught_ci=[lo, hi],
                              session_frr=g["frr"], session_frr_ci=[0.0, 0.0])
            d["caught_seed_mean_at_0.05"] = d["0.05"]["caught"]
            d["caught_seed_sd_at_0.05"] = 0.0
            d["price"] = dict(frr_for_caught_50=None, frr_for_caught_80=None, frr_for_caught_95=None)
            ctl["stats"][st] = d
        results.append(ctl)

        for (rule, sur, r) in configs:
            P = 200 // r
            if rule == "A0_uniform":
                keep = np.tile(np.arange(200), (5, NU, 1))
            else:
                keep = np.argsort(rank_keys[(rule, sur)], axis=2)[:, :, :P]
            rec, pu, den = run_selection(R, keep, P, fake_score, fake_dur, tabs_by_src, durtab,
                                tau, tau_slot, slot_src, sess_src, nulls_cf, nulls_or,
                                gen_op, alpha90, bmat, NU, n_gen_u, pg_cf,
                                seed_base=int(zlib.crc32(f'{mod}|{det}|{reg}|{rule}|{sur}|{r}'.encode())))
            rec.update(modality=mod, detector=det, regime=reg, rule=rule,
                       surrogate=sur, r=r, pool_kept=P,
                       generated_per_submitted=float(r),
                       surrogate_same_modality=(sur.split("|")[0] == mod) if sur else None)
            results.append(rec)
            PU["caught"].append(pu); PU["den"].append(den)
            PU["cfg"].append((reg, rule, str(sur), r))
        ga = np.zeros((len(STATS), 3, NU))
        for i, st in enumerate(STATS):
            for j, tag in enumerate(("0.05", "0.01", "a90")):
                ga[i, j] = gen_op[st][tag]["k_u"]
        PU["gen_alarm"][reg] = ga; PU["gen_den"][reg] = n_gen_u
        PU["alpha90"][reg] = np.array([gen_op[st]["a90"]["alpha"] for st in STATS])
    np.savez_compressed(os.path.join(OUT, f"peruser_{mod}__{det}.npz"),
                        caught=np.array(PU["caught"]), den=np.array(PU["den"]),
                        cfg=np.array(PU["cfg"], dtype=object),
                        gen_alarm_touch=PU["gen_alarm"].get("touch", np.zeros((len(STATS), 3, NU))),
                        gen_den_touch=PU["gen_den"].get("touch", np.zeros(NU)),
                        gen_alarm_keystroke=PU["gen_alarm"].get("keystroke", np.zeros((len(STATS), 3, NU))),
                        gen_den_keystroke=PU["gen_den"].get("keystroke", np.zeros(NU)),
                        stats=np.array(STATS), tags=np.array(["0.05", "0.01", "a90"]),
                        alpha90_touch=PU["alpha90"].get("touch", np.zeros(len(STATS))),
                        alpha90_keystroke=PU["alpha90"].get("keystroke", np.zeros(len(STATS))))
    return results, partC


def run_selection(R, keep, P, fake_score, fake_dur, tabs_by_src, durtab, tau, tau_slot,
                  slot_src, sess_src, nulls_cf, nulls_or, gen_op, alpha90, bmat, NU,
                  n_gen_u, pg_cf, seed_base):
    n_sess = R.n_sess
    caught_u = {st: {t: np.zeros(NU) for t in ("0.05", "0.01", "a90")} for st in STATS}
    caught_u_or = {st: {t: np.zeros(NU) for t in ("0.05", "0.01")} for st in STATS}
    curve = {st: np.zeros(len(AGRID)) for st in STATS}
    seed_rates = {st: [] for st in STATS}
    far_num = 0.0; far_den = 0.0
    short_num = 0
    n_fake_u = np.bincount(R.sess_user, minlength=NU).astype(float) * N_SEEDS
    reuse_slots = int((R.rank >= P).sum())

    for s in range(N_SEEDS):
        rng = np.random.default_rng(seed_base + 7919 * s)
        ordm = np.argsort(rng.random((R.n_groups, P)), axis=1)
        col = ordm[R.group, R.rank % P]
        pos = keep[R.slot_action, R.slot_user, col]
        fs = fake_score[R.slot_action, R.slot_user, pos]
        fd = fake_dur[R.slot_action, R.slot_user, pos]
        far_num += float((fs < tau_slot).sum()); far_den += R.M
        short_num += reuse_slots

        fcf, forc = session_stats(R, fs, fd, tabs_by_src, durtab, tau_slot, slot_src)
        for st in STATS:
            p0 = pvals(fcf[st], nulls_cf[st][0]); p1 = pvals(fcf[st], nulls_cf[st][1])
            pf = np.where(sess_src == 0, p0, p1)
            pf_or = pvals(forc[st], nulls_or[st])
            curve[st] += curve_counts(pf, AGRID)
            for tag, al in (("0.05", 0.05), ("0.01", 0.01), ("a90", alpha90[st])):
                caught_u[st][tag] += np.bincount(R.sess_user, weights=(pf <= al).astype(float),
                                                 minlength=NU)
            for tag, al in (("0.05", 0.05), ("0.01", 0.01)):
                caught_u_or[st][tag] += np.bincount(R.sess_user,
                                                    weights=(pf_or <= al).astype(float), minlength=NU)
            seed_rates[st].append(float((pf <= 0.05).mean()))

    out = dict(n_sessions=int(n_sess), n_slots=int(R.M),
               forced_reuse_slots_per_draw=reuse_slots,
               forced_reuse_frac=float(reuse_slots / R.M),
               per_event_FAR_at_dev_frr5=float(far_num / far_den),
               stats={})
    frr_curve = None
    for st in STATS:
        c_curve = curve[st] / (n_sess * N_SEEDS)
        f_curve = gen_op[st]["curve"]
        d = dict()
        for tag in ("0.05", "0.01", "a90"):
            num = caught_u[st][tag]; den = n_fake_u
            lo, hi = boot_rate(num, den, bmat)
            g = gen_op[st][tag]
            glo, ghi = boot_rate(g["k_u"], np.maximum(g["n_u"], 1e-12), bmat)
            d[tag] = dict(alpha=float(alpha90[st] if tag == "a90" else float(tag)),
                          caught=float(num.sum() / den.sum()), caught_ci=[lo, hi],
                          session_frr=g["frr"], session_frr_ci=[glo, ghi])
        for tag in ("0.05", "0.01"):
            num = caught_u_or[st][tag]
            d[tag]["caught_oracle_calib"] = float(num.sum() / n_fake_u.sum())
            d[tag]["session_frr_oracle_calib"] = gen_op[st][tag]["frr_oracle"]
        d["caught_seed_mean_at_0.05"] = float(np.mean(seed_rates[st]))
        d["caught_seed_sd_at_0.05"] = float(np.std(seed_rates[st], ddof=1))
        d["price"] = inverse_price(c_curve, f_curve, PRICE_TARGETS)
        out["stats"][st] = d
    pu = np.zeros((len(STATS), 3, NU))
    for i, st in enumerate(STATS):
        for j, tag in enumerate(("0.05", "0.01", "a90")):
            pu[i, j] = caught_u[st][tag]
    return out, pu, n_fake_u


# ============================================================================= PART B
def duration_bulk_stats(IDX):
    out = {}
    for mod in ("imu_only", "trajectory_xytime"):
        for ai, a in enumerate(ACTIONS):
            b = IDX["bundle_of"][(a, mod)]
            key = f"{a}|{b}"
            if key in out:
                continue
            g = IDX["bundleD"][b]["gen"][IDX["gen_action"] == ai]
            f = IDX["bundleD"][b]["fake"][ai].ravel()
            dv = IDX["devD"][b][a]
            cap = float(f.max())
            out[key] = dict(
                action=a, bundle=b, n_genuine=int(len(g)), n_fake=int(len(f)),
                genuine=dict(p5=float(np.percentile(g, 5)), p50=float(np.median(g)),
                             mean=float(g.mean()), p95=float(np.percentile(g, 95)),
                             max=float(g.max())),
                fake=dict(p5=float(np.percentile(f, 5)), p50=float(np.median(f)),
                          mean=float(f.mean()), p95=float(np.percentile(f, 95)),
                          max=cap),
                median_shift_s=float(np.median(g) - np.median(f)),
                frac_fake_exactly_on_cap=float(np.mean(np.abs(f - cap) < 1e-9)),
                frac_fake_within_1ms_of_cap=float(np.mean(np.abs(f - cap) < 1e-3)),
                dev_fake_cap=float(dv["fake"].max()),
                dev_genuine_max=float(dv["gen"].max()),
                frac_genuine_above_fake_cap=float(np.mean(g > cap)),
            )
    return out


def _dur_eval(R, gen_du, fake_du_draws, durtab, fold_of_user, bmat, NU):
    """Evaluate a duration-only (or any per-slot additive) session LLR."""
    slot_src = 1 - fold_of_user[R.slot_user]
    sess_src = 1 - fold_of_user[R.sess_user]
    gS = R.agg_sum(slot_lookup_llr(R, durtab, gen_du))
    nulls = [np.sort(gS[R.sess_user % 2 == s]) for s in (0, 1)]
    p0 = pvals(gS, nulls[0]); p1 = pvals(gS, nulls[1])
    pg = np.where(sess_src == 0, p0, p1)
    n_gen_u = np.bincount(R.sess_user, minlength=NU).astype(float)
    res = dict(n_sessions=int(R.n_sess))
    fcurve = curve_counts(pg, AGRID) / R.n_sess
    ccurve = np.zeros(len(AGRID))
    caught_u = {t: np.zeros(NU) for t in ("0.05", "0.01")}
    n_fake_u = n_gen_u * len(fake_du_draws)
    for fd in fake_du_draws:
        fS = R.agg_sum(slot_lookup_llr(R, durtab, fd))
        q0 = pvals(fS, nulls[0]); q1 = pvals(fS, nulls[1])
        pf = np.where(sess_src == 0, q0, q1)
        ccurve += curve_counts(pf, AGRID)
        for tag, al in (("0.05", 0.05), ("0.01", 0.01)):
            caught_u[tag] += np.bincount(R.sess_user, weights=(pf <= al).astype(float), minlength=NU)
    ccurve /= (R.n_sess * len(fake_du_draws))
    for tag, al in (("0.05", 0.05), ("0.01", 0.01)):
        alarm = (pg <= al).astype(float)
        ku = np.bincount(R.sess_user, weights=alarm, minlength=NU)
        lo, hi = boot_rate(caught_u[tag], n_fake_u, bmat)
        glo, ghi = boot_rate(ku, n_gen_u, bmat)
        pu = ku / np.maximum(n_gen_u, 1e-12)
        res[tag] = dict(caught=float(caught_u[tag].sum() / n_fake_u.sum()), caught_ci=[lo, hi],
                        session_frr=float(ku.sum() / n_gen_u.sum()), session_frr_ci=[glo, ghi],
                        per_user_frr=dict(min=float(pu.min()), median=float(np.median(pu)),
                                          p90=float(np.percentile(pu, 90)), max=float(pu.max())))
    res["price"] = inverse_price(ccurve, fcurve, PRICE_TARGETS)
    return res


def draw_fake_slots(R, seed, P=200, keep=None):
    rng = np.random.default_rng(seed)
    if keep is None:
        keep = np.tile(np.arange(200), (5, R.slot_user.max() + 1, 1))
    ordm = np.argsort(rng.random((R.n_groups, P)), axis=1)
    col = ordm[R.group, R.rank % P]
    return keep[R.slot_action, R.slot_user, col]


def part_b(IDX, REG, bmat):
    NU = IDX["NU"]
    fold_of_user = np.array([i % 2 for i in range(NU)])
    res = dict(bulk=duration_bulk_stats(IDX), regimes={})
    rng_j = np.random.default_rng(20260810)

    for mod in ("imu_only", "trajectory_xytime"):
        gen_dur = _combine_gen([IDX["bundleD"][IDX["bundle_of"][(ACTIONS[ai], mod)]]["gen"]
                                for ai in range(5)], IDX["gen_action"])
        fake_dur = np.stack([IDX["bundleD"][IDX["bundle_of"][(ACTIONS[ai], mod)]]["fake"][ai]
                             for ai in range(5)])
        dev = [IDX["devD"][IDX["bundle_of"][(a, mod)]][a] for a in ACTIONS]
        cap = np.array([d["fake"].max() for d in dev])
        genmed = np.array([np.median(d["gen"]) for d in dev])

        tab_raw = [fit_qllr(dev[ai]["gen"], dev[ai]["fake"]) for ai in range(5)]

        # control: clip both arms at the per-action DEV fake cap + U(-5,+5) ms jitter
        def clipjit(v, ai_arr, rng):
            return np.minimum(v, cap[ai_arr]) + rng.uniform(-0.005, 0.005, size=v.shape)
        tab_ctl = []
        for ai in range(5):
            r = np.random.default_rng(1000 + ai)
            g = np.minimum(dev[ai]["gen"], cap[ai]) + r.uniform(-0.005, 0.005, dev[ai]["gen"].shape)
            f = np.minimum(dev[ai]["fake"], cap[ai]) + r.uniform(-0.005, 0.005, dev[ai]["fake"].shape)
            tab_ctl.append(fit_qllr(g, f))

        # quantile mapping fake -> genuine.  Randomised (mid-rank) PIT so that the
        # heavy 10 ms tie structure of the duration variable does not collapse mass.
        def make_qmap(src_fake, src_gen):
            return [(np.sort(src_fake[ai]), np.sort(src_gen[ai])) for ai in range(5)]

        qmap_dev = make_qmap([dev[ai]["fake"] for ai in range(5)],
                             [dev[ai]["gen"] for ai in range(5)])
        qmap_test = make_qmap([fake_dur[ai].ravel() for ai in range(5)],
                              [gen_dur[IDX["gen_action"] == ai] for ai in range(5)])

        def apply_qmap(v, ai_arr, qm, rng):
            out = np.empty_like(v)
            for ai in range(5):
                m = ai_arr == ai
                if not m.any():
                    continue
                fs, gs = qm[ai]
                lo = np.searchsorted(fs, v[m], side="left").astype(np.float64)
                hi = np.searchsorted(fs, v[m], side="right").astype(np.float64)
                u = (lo + rng.random(int(m.sum())) * np.maximum(hi - lo, 1.0)) / len(fs)
                j = np.clip((u * len(gs)).astype(np.int64), 0, len(gs) - 1)
                out[m] = gs[j]
            return out

        rq = np.random.default_rng(5150)
        tab_qm = []
        for ai in range(5):
            fm = apply_qmap(dev[ai]["fake"], np.full(len(dev[ai]["fake"]), ai), qmap_dev, rq)
            tab_qm.append(fit_qllr(dev[ai]["gen"], fm))

        for reg in ("touch", "keystroke"):
            R = REG[reg]
            g_du = gen_dur[R.slot_gid]
            draws = [fake_dur[R.slot_action, R.slot_user, draw_fake_slots(R, 4242 + 13 * s)]
                     for s in range(N_SEEDS)]
            key = f"{mod}|{reg}"
            entry = {}
            entry["raw"] = _dur_eval(R, g_du, draws, tab_raw, fold_of_user, bmat, NU)
            # control
            rr = np.random.default_rng(777)
            gc = np.minimum(g_du, cap[R.slot_action]) + rr.uniform(-0.005, 0.005, g_du.shape)
            dc = [np.minimum(d, cap[R.slot_action]) + rr.uniform(-0.005, 0.005, d.shape)
                  for d in draws]
            entry["control_clip_cap_plus_jitter"] = _dur_eval(R, gc, dc, tab_ctl,
                                                              fold_of_user, bmat, NU)
            # quantile mapping (deployed detector kept fixed = tab_raw)
            dq = [apply_qmap(d, R.slot_action, qmap_dev, rq) for d in draws]
            dq_or = [apply_qmap(d, R.slot_action, qmap_test, rq) for d in draws]
            entry["quantile_mapped_fixed_detector"] = _dur_eval(R, g_du, dq, tab_raw,
                                                                fold_of_user, bmat, NU)
            entry["quantile_mapped_refit_detector"] = _dur_eval(R, g_du, dq, tab_qm,
                                                                fold_of_user, bmat, NU)
            entry["quantile_mapped_testfit_ORACLE_fixed_detector"] = _dur_eval(
                R, g_du, dq_or, tab_raw, fold_of_user, bmat, NU)
            entry["quantile_mapped_testfit_ORACLE_refit_detector"] = _dur_eval(
                R, g_du, dq_or, tab_qm, fold_of_user, bmat, NU)
            # mapping diagnostic: did the mapped fake arm really match genuine?
            chk = {}
            for ai, a in enumerate(ACTIONS):
                k = R.slot_action == ai
                if not k.any():
                    continue
                gv = g_du[k]; mv = np.concatenate([d[k] for d in dq]); rv = np.concatenate([d[k] for d in draws])
                chk[a] = dict(
                    genuine_p50=float(np.median(gv)), raw_fake_p50=float(np.median(rv)),
                    mapped_fake_p50=float(np.median(mv)),
                    genuine_p5=float(np.percentile(gv, 5)), mapped_fake_p5=float(np.percentile(mv, 5)),
                    genuine_p95=float(np.percentile(gv, 95)), mapped_fake_p95=float(np.percentile(mv, 95)),
                    ks=float(np.max(np.abs(
                        np.searchsorted(np.sort(gv), np.sort(np.unique(np.concatenate([gv, mv])))) / len(gv)
                        - np.searchsorted(np.sort(mv), np.sort(np.unique(np.concatenate([gv, mv])))) / len(mv)))),
                    ks_raw_fake_vs_genuine=float(np.max(np.abs(
                        np.searchsorted(np.sort(gv), np.sort(np.unique(np.concatenate([gv, rv])))) / len(gv)
                        - np.searchsorted(np.sort(rv), np.sort(np.unique(np.concatenate([gv, rv])))) / len(rv)))),
                )
            entry["quantile_map_diagnostic"] = chk
            # ---- consequence analysis: timing features
            entry["timing"] = timing_consequence(R, g_du, draws, genmed, fold_of_user, bmat, NU)
            for pac in ("pacing_replay", "pacing_permuted"):
                t = entry["timing"][pac]
                for tag in ("0.05", "0.01"):
                    ex_full = t["real_durations"][tag]["caught"] - t["real_durations"][tag]["session_frr"]
                    ex_neut = (t["median_durations_both_arms"][tag]["caught"]
                               - t["median_durations_both_arms"][tag]["session_frr"])
                    t.setdefault("duration_share", {})[tag] = dict(
                        excess_over_chance_with_real_durations=float(ex_full),
                        excess_over_chance_with_durations_neutralised=float(ex_neut),
                        share_of_excess_attributable_to_duration=(
                            float((ex_full - ex_neut) / ex_full) if abs(ex_full) > 1e-9 else None))
            res["regimes"][key] = entry
    return res


def _sess_gaps(R, gen_du):
    """gap[i] = onset[i]-onset[i-1]-dur[i-1]; gap of first slot in a session = 0."""
    gap = np.zeros(R.M)
    prev_sess = np.concatenate([[-1], R.slot_sess[:-1]])
    same = R.slot_sess == prev_sess
    d = np.concatenate([[0.0], R.slot_start[:-1]])
    dd = np.concatenate([[0.0], gen_du[:-1]])
    gap[same] = (R.slot_start - d - dd)[same]
    neg = int((gap < 0).sum())
    gap = np.maximum(gap, 0.0)
    return gap, neg


def timing_consequence(R, gen_du, draws, genmed, fold_of_user, bmat, NU):
    """How much of a timing-feature session result is duration?

    Timing statistic = sum_i llr_dur(d_i, a_i) + llr_span(sum_i d_i + sum_i g_i),
    with g_i = onset_i - onset_{i-1} - d_{i-1} taken from the genuine session.
    Two pacing models for the fake arm:
      pacing_replay   : the attacker replays the victim session's gaps exactly.
      pacing_permuted : the gaps are permuted within the session (same marginal
                        distribution, broken coupling to duration).
    In both, durations are then replaced by the per-action genuine median in BOTH
    arms and the whole pipeline is re-run; the drop in caught-rate is duration's share.
    """
    gap, neg = _sess_gaps(R, gen_du)
    sess_src = 1 - fold_of_user[R.sess_user]
    n_gen_u = np.bincount(R.sess_user, minlength=NU).astype(float)

    def stat(du, gp, tabs):
        td, tg, ts = tabs
        out = np.zeros(R.M)
        for ai in range(5):
            k = R.slot_action == ai
            if not k.any():
                continue
            if td[ai] is not None:
                out[k] += apply_qllr(td[ai], du[k])
            if tg[ai] is not None:
                out[k] += apply_qllr(tg[ai], gp[k])
        span = R.agg_sum(du) + R.agg_sum(gp)
        return R.agg_sum(out) + apply_qllr(ts, span)

    def build_tabs(cal, du_g, gp_g, du_f, gp_f):
        m = np.isin(R.slot_user, cal); ms = np.isin(R.sess_user, cal)
        td, tg = [], []
        for ai in range(5):
            k = m & (R.slot_action == ai)
            td.append(fit_qllr(du_g[k], du_f[k]) if k.sum() >= 20 else None)
            tg.append(fit_qllr(gp_g[k], gp_f[k]) if k.sum() >= 20 else None)
        sg = R.agg_sum(du_g) + R.agg_sum(gp_g)
        sf = R.agg_sum(du_f) + R.agg_sum(gp_f)
        ts = fit_qllr(sg[ms], sf[ms])
        return td, tg, ts

    def run(du_g, du_f_list, gp_f_list):
        tabs = {s: build_tabs(np.nonzero(fold_of_user == s)[0], du_g, gap,
                              du_f_list[0], gp_f_list[0]) for s in (0, 1)}
        gS = {s: stat(du_g, gap, tabs[s]) for s in (0, 1)}
        nulls = [np.sort(gS[s][R.sess_user % 2 == s]) for s in (0, 1)]
        pg = np.where(sess_src == 0, pvals(gS[0], nulls[0]), pvals(gS[1], nulls[1]))
        caught_u = {t: np.zeros(NU) for t in ("0.05", "0.01")}
        for du_f, gp_f in zip(du_f_list, gp_f_list):
            fS = {s: stat(du_f, gp_f, tabs[s]) for s in (0, 1)}
            pf = np.where(sess_src == 0, pvals(fS[0], nulls[0]), pvals(fS[1], nulls[1]))
            for tag, al in (("0.05", 0.05), ("0.01", 0.01)):
                caught_u[tag] += np.bincount(R.sess_user, weights=(pf <= al).astype(float),
                                             minlength=NU)
        nf = n_gen_u * len(du_f_list)
        o = {}
        for tag, al in (("0.05", 0.05), ("0.01", 0.01)):
            ku = np.bincount(R.sess_user, weights=(pg <= al).astype(float), minlength=NU)
            lo, hi = boot_rate(caught_u[tag], nf, bmat)
            o[tag] = dict(caught=float(caught_u[tag].sum() / nf.sum()), caught_ci=[lo, hi],
                          session_frr=float(ku.sum() / n_gen_u.sum()))
        return o

    med = genmed[R.slot_action]
    rep = [gap] * len(draws)
    perm = []
    rg = np.random.default_rng(31337)
    order = np.lexsort((rg.random(R.M), R.slot_sess))
    for _ in draws:
        o2 = np.lexsort((rg.random(R.M), R.slot_sess))
        gp = np.empty(R.M); gp[np.sort(o2)] = gap[o2]
        perm.append(gp)
    out = dict(
        negative_gaps_clipped=neg,
        gap_seconds=dict(p5=float(np.percentile(gap[gap > 0], 5)),
                         p50=float(np.median(gap[gap > 0])),
                         p95=float(np.percentile(gap[gap > 0], 95))),
        pacing_replay=dict(
            real_durations=run(gen_du, draws, rep),
            median_durations_both_arms=run(med, [med] * len(draws), rep),
            note="gaps identical in both arms; with durations neutralised the two arms "
                 "become byte-identical, so caught==FRR is TRUE BY CONSTRUCTION (F14)."),
        pacing_permuted=dict(
            real_durations=run(gen_du, draws, perm),
            median_durations_both_arms=run(med, [med] * len(draws), perm),
            note="fake-arm gaps permuted within session: same gap marginal, coupling broken. "
                 "The residual caught-rate after neutralising duration is the non-duration part."),
    )
    return out


# ============================================================================= driver
def jsan(o):
    if isinstance(o, dict):
        return {str(k): jsan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsan(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsan(o.tolist())
    if isinstance(o, (np.str_, str)):
        return str(o)
    return o


def coverage_block(IDX, REG):
    cov = dict(
        test_users=IDX["users"], n_test_users=IDX["NU"],
        scored_genuine_events=int(IDX["NG"]),
        fake_pool_per_user_action=200,
        sessions_dropped=IDX["dropped"],
        regimes={},
    )
    for reg in ("touch", "keystroke"):
        R = REG[reg]
        sc = np.array([c[0] for c in IDX["covariate"][reg]])
        tt = np.array([c[1] for c in IDX["covariate"][reg]])
        cov["regimes"][reg] = dict(
            n_sessions=int(R.n_sess), n_scored_slots=int(R.M),
            total_session_slots_in_full_inventory=int(tt.sum()),
            scored_slot_fraction=float(sc.sum() / tt.sum()),
            scored_slots_per_session=dict(min=int(sc.min()), p50=float(np.median(sc)),
                                          p90=float(np.percentile(sc, 90)), max=int(sc.max())),
            max_slots_of_one_action_in_one_session=int(R.group_count.max()),
            sessions_per_user=dict(min=int(np.bincount(R.sess_user, minlength=IDX["NU"]).min()),
                                   median=float(np.median(np.bincount(R.sess_user,
                                                                      minlength=IDX["NU"]))),
                                   max=int(np.bincount(R.sess_user, minlength=IDX["NU"]).max())),
        )
        sf = {}
        for r in R_VALUES:
            P = 200 // r
            sf[str(r)] = dict(pool_kept=P,
                              forced_reuse_slots=int((R.rank >= P).sum()),
                              forced_reuse_frac=float((R.rank >= P).mean()),
                              groups_short=int((R.group_count > P).sum()),
                              n_groups=int(R.n_groups))
        cov["regimes"][reg]["rejection_sampling_shortfall"] = sf
    return cov


def main():
    t0 = time.time()
    _log("loading")
    cm, C, B, bd = load_all()
    IDX = build_index(cm, C, B, bd)
    _log("index built", IDX["NG"], "genuine events")
    REG = make_regimes(IDX)
    NU = IDX["NU"]
    bmat = np.random.default_rng(12345).integers(0, NU, size=(N_BOOT, NU))
    GLOB["IDX"] = IDX; GLOB["REG"] = REG; GLOB["bmat"] = bmat

    cov = coverage_block(IDX, REG)
    json.dump(jsan(cov), open(os.path.join(OUT, "coverage.json"), "w"), indent=1)
    _log("coverage written", {k: v["n_sessions"] for k, v in cov["regimes"].items()})

    # per-cell FAR@frr5 sanity (must reproduce the release mean 0.774575)
    fars = {}
    for cell in cm:
        cs = IDX["cellS"][cell]
        fars[cell] = float((cs["fake"][cs["action"]] < cs["tau"]).mean())
    json.dump(jsan(dict(per_cell=fars, mean=float(np.mean(list(fars.values()))))),
              open(os.path.join(OUT, "far_frr5_check.json"), "w"), indent=1)
    _log("FAR@frr5 mean over 90 cells =", np.mean(list(fars.values())))

    _log("PART B")
    pb = part_b(IDX, REG, bmat)
    json.dump(jsan(pb), open(os.path.join(OUT, "part_b.json"), "w"), indent=1)
    _log("part B done", time.time() - t0)

    _log("PART A/C over", len(COMBOS), "combos")
    allA, allC = [], []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, (ra, rc) in enumerate(ex.map(run_combo, COMBOS)):
            allA += ra; allC += rc
            _log(f"  combo {i+1}/{len(COMBOS)} done, {len(allA)} records, {time.time()-t0:.0f}s")
    json.dump(jsan(allA), open(os.path.join(OUT, "part_a.json"), "w"))
    json.dump(jsan(allC), open(os.path.join(OUT, "part_c.json"), "w"), indent=1)
    _log("ALL DONE", time.time() - t0)


if __name__ == "__main__":
    main()
