#!/usr/bin/env python
"""Reviewer check 6: defender moves the measurement does not rule out.

The released session statistic is a SUM OF PER-EVENT LLRs fitted against UNSELECTED fakes.
Selection moves the attacker into the left tail of that score, which is exactly the
direction the statistic rewards.  Two defenders that do not have that property:
  D1  one-class typicality: -log p_genuine(score), no fake model at all (two-sided by
      construction: a score that is too good is as atypical as one that is too bad).
  D2  within-session score DISPERSION: selection compresses the score spread.
  D3  exact-duplicate rule (counted separately in rv1).
Everything else -- sessions, slots, folds, nulls, alpha -- is identical to agent B.
"""
import os, sys, json, pickle, zlib, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/B")
import adaptive_b as AB

RV = "/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/RV"
ACT = AB.ACTIONS
NSEED = 20
G = {}


def fit_neglogp(vals, nb=40):
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        hi = lo + 1.0
    inner = np.linspace(lo, hi, nb + 1)[1:-1]          # EQUAL-WIDTH bins -> a real density
    b = np.bincount(np.searchsorted(inner, vals, side="right"), minlength=nb).astype(float)
    p = (b + 0.5) / (b.sum() + 0.5 * nb)
    return dict(inner=inner, val=-np.log(p))


def apply_tab(tab, v):
    return tab["val"][np.searchsorted(tab["inner"], v, side="right")]


def run_cell(args):
    mod, det = args
    IDX, REG = G["IDX"], G["REG"]
    NU = IDX["NU"]; R = REG["touch"]
    fold = np.arange(NU) % 2
    ga = IDX["gen_action"]
    cells = {a: IDX["cellS"][f"{a}__{mod}__{det}"] for a in ACT}
    gen_score = AB._combine_gen([cells[ACT[ai]]["gen"] for ai in range(5)], ga)
    fake_score = np.stack([cells[ACT[ai]]["fake"][ai] for ai in range(5)])
    slot_src = 1 - fold[R.slot_user]; sess_src = 1 - fold[R.sess_user]

    # calibration tables per fold
    llr_t, one_t, mu, sd = {}, {}, {}, {}
    for src in (0, 1):
        cal = np.nonzero(fold == src)[0]
        lt, ot, m_, s_ = [], [], [], []
        for ai in range(5):
            gsel = (ga == ai) & np.isin(IDX["gen_user"], cal)
            gv = gen_score[gsel]
            lt.append(AB.fit_qllr(gv, fake_score[ai][cal].ravel()))
            ot.append(fit_neglogp(gv))
            m_.append(float(gv.mean())); s_.append(float(gv.std() + 1e-12))
        llr_t[src] = lt; one_t[src] = ot; mu[src] = np.array(m_); sd[src] = np.array(s_)

    def stat_llr(sc):
        l0 = AB.slot_lookup_llr(R, llr_t[0], sc); l1 = AB.slot_lookup_llr(R, llr_t[1], sc)
        return R.agg_sum(np.where(slot_src == 0, l0, l1))

    def stat_one(sc):
        o = np.empty(R.M)
        for src in (0, 1):
            for ai in range(5):
                m = (slot_src == src) & (R.slot_action == ai)
                if m.any():
                    o[m] = apply_tab(one_t[src][ai], sc[m])
        return R.agg_sum(o)

    def stat_disp(sc):
        z = np.empty(R.M)
        for src in (0, 1):
            m = slot_src == src
            z[m] = (sc[m] - mu[src][R.slot_action[m]]) / sd[src][R.slot_action[m]]
        mn = R.agg_mean(z)
        var = R.agg_mean(z * z) - mn ** 2
        return -np.sqrt(np.maximum(var, 0.0))          # low spread -> high statistic

    # D4: the defender ANTICIPATES rejection sampling and fits the per-event LLR against a
    # fake calibration population that has already been filtered by the same rule at r=10.
    ens_c = 0.0; nc = 0
    for sd2 in AB.DETECTORS:
        if sd2 == det:
            continue
        x = np.stack([IDX["cellS"][f"{ACT[ai]}__{mod}__{sd2}"]["fake"][ai] for ai in range(5)])
        ens_c = ens_c + np.argsort(np.argsort(x, axis=2), axis=2).astype(float) / 199.0
        nc += 1
    ens_c = ens_c / nc
    keep_c = np.argsort(ens_c, axis=2)[:, :, :20]
    llr_a = {}
    for src in (0, 1):
        cal = np.nonzero(fold == src)[0]
        tt = []
        for ai in range(5):
            gsel = (ga == ai) & np.isin(IDX["gen_user"], cal)
            fv = np.concatenate([fake_score[ai][u][keep_c[ai, u]] for u in cal])
            tt.append(AB.fit_qllr(gen_score[gsel], fv))
        llr_a[src] = tt

    def stat_llr_adapt(sc):
        l0 = AB.slot_lookup_llr(R, llr_a[0], sc); l1 = AB.slot_lookup_llr(R, llr_a[1], sc)
        return R.agg_sum(np.where(slot_src == 0, l0, l1))

    STATS = dict(llr_sum=stat_llr, oneclass=stat_one, dispersion=stat_disp,
                 llr_defender_anticipates=stat_llr_adapt)
    gsc = gen_score[R.slot_gid]
    nulls, pg = {}, {}
    for k, f in STATS.items():
        gS = f(gsc)
        nl = [np.sort(gS[R.sess_user % 2 == s]) for s in (0, 1)]
        nulls[k] = nl
        pg[k] = np.where(sess_src == 0, AB.pvals(gS, nl[0]), AB.pvals(gS, nl[1]))
    frr = {k: float((v <= 0.05).mean()) for k, v in pg.items()}

    # selection keys
    ens = 0.0; n = 0
    for sd_ in AB.DETECTORS:
        if sd_ == det:
            continue
        x = np.stack([IDX["cellS"][f"{ACT[ai]}__{mod}__{sd_}"]["fake"][ai] for ai in range(5)])
        ens = ens + np.argsort(np.argsort(x, axis=2), axis=2).astype(float) / 199.0
        n += 1
    keys = {"A1_ens5": ens / n, "A2_centroid": IDX["a2c"][mod], "A3_oracle": fake_score}

    res = {}
    for arm, r in [("A0", 1), ("A1_ens5", 5), ("A1_ens5", 10), ("A1_ens5", 20),
                   ("A2_centroid", 10), ("A3_oracle", 5), ("A3_oracle", 10)]:
        P = 200 // r
        keep = (np.tile(np.arange(200), (5, NU, 1)) if arm == "A0"
                else np.argsort(keys[arm], axis=2)[:, :, :P])
        acc = {k: 0.0 for k in STATS}
        for s in range(NSEED):
            rng = np.random.default_rng(zlib.crc32(f"{mod}|{det}|{arm}|{r}|{s}".encode()))
            ordm = np.argsort(rng.random((R.n_groups, P)), axis=1)
            pos = keep[R.slot_action, R.slot_user, ordm[R.group, R.rank % P]]
            fs = fake_score[R.slot_action, R.slot_user, pos]
            for k, f in STATS.items():
                fS = f(fs)
                pf = np.where(sess_src == 0, AB.pvals(fS, nulls[k][0]), AB.pvals(fS, nulls[k][1]))
                acc[k] += float((pf <= 0.05).mean())
        res[f"{arm}|r{r}"] = {k: v / NSEED for k, v in acc.items()}
    return (mod, det), dict(frr=frr, res=res)


if __name__ == "__main__":
    D = pickle.load(open(os.path.join(RV, "idx.pkl"), "rb"))
    G["IDX"] = D["IDX"]; G["REG"] = D["REG"]
    jobs = [(m, d) for m in AB.MODALITIES for d in AB.DETECTORS]
    outs = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        for k, v in ex.map(run_cell, jobs):
            outs["|".join(k)] = v
    json.dump(outs, open(os.path.join(RV, "rv6.json"), "w"), indent=1)
    print(f"{time.time()-t0:.0f}s  18 touch cells, alpha=0.05, B-style cross-fit\n")
    print(f"{'':22s} " + "  ".join(f"{k:>12s}" for k in ("llr_sum", "oneclass", "dispersion", "llr_defender_anticipates")))
    fr = {k: np.mean([v["frr"][k] for v in outs.values()]) for k in ("llr_sum", "oneclass", "dispersion", "llr_defender_anticipates")}
    print(f"{'genuine session FRR':22s} " + "  ".join(f"{fr[k]:12.4f}" for k in fr))
    for arm in sorted(next(iter(outs.values()))["res"]):
        row = {k: np.mean([v["res"][arm][k] for v in outs.values()]) for k in fr}
        nb = {k: sum(1 for v in outs.values() if v["res"][arm][k] <= v["frr"][k]) for k in fr}
        print(f"{arm:22s} " + "  ".join(f"{row[k]:12.4f}" for k in fr)
              + "   collapsed/18: " + " ".join(f"{k}={nb[k]}" for k in fr))
