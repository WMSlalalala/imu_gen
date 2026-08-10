#!/usr/bin/env python
"""Reviewer check 5: is the duration quantile map physically executable, and does it
remove the leak, or only remove it from the one variable it edits?

  (a) duration <-> sample count coupling (the redundant channel the map does not touch)
  (b) a sample-count-only session detector, fitted on DEV, before and after the map
  (c) feasibility of the mapped durations: values the generator has never emitted
  (d) the map when the attacker only owns FIVE genuine recordings per (victim, action)
"""
import os, sys, json, pickle, collections
import numpy as np
sys.path.insert(0, "/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/B")
import adaptive_b as AB

RV = "/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/RV"
CACHE = "/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/B/cache"
D = pickle.load(open(os.path.join(RV, "idx.pkl"), "rb"))
IDX, REG = D["IDX"], D["REG"]
ACT = AB.ACTIONS
NU = IDX["NU"]
out = {}

# ---------------- rebuild nsamp arrays in the same (gid) / (5,NU,200) layout -----------
C = np.load(os.path.join(CACHE, "cells.npz"), allow_pickle=True)
users = sorted(set(C["tap__imu_only__paper_xgboost|user_id"].tolist()))
UIDX = {u: i for i, u in enumerate(users)}
pool_pos = {}
for ai, a in enumerate(ACT):
    cell = f"{a}__imu_only__paper_xgboost"
    eid = C[cell + "|event_id"]; lab = C[cell + "|label"]; usr = C[cell + "|user_id"]
    fe, fu = eid[lab == 1], usr[lab == 1]
    for u in users:
        for p, e in enumerate(np.sort(fe[fu == u])):
            pool_pos[e] = (ai, UIDX[u], p)
gid_of_eid = {}
gi = 0
for ai, a in enumerate(ACT):
    cell = f"{a}__imu_only__paper_xgboost"
    eid = C[cell + "|event_id"]; lab = C[cell + "|label"]
    for e in np.sort(eid[lab == 0]):
        gid_of_eid[e] = gi; gi += 1
NG = gi

bundle_ns = {}
dev_ns = {}
for b in sorted({v for v in IDX["bundle_of"].values()}):
    z = np.load(os.path.join(CACHE, f"bundle_{b}.npz"), allow_pickle=True)
    eid = z["event_id"]; ns = z["nsamp"].astype(float); lab = z["label"]; spl = z["split"]
    act = z["action"]; dur = z["duration"]
    g = np.full(NG, np.nan); f = np.full((5, NU, 200), np.nan)
    tm = spl == "test"
    for e, v, l in zip(eid[tm].tolist(), ns[tm].tolist(), lab[tm].tolist()):
        if l == 0:
            j = gid_of_eid.get(e)
            if j is not None:
                g[j] = v
        else:
            pp = pool_pos.get(e)
            if pp is not None:
                f[pp[0], pp[1], pp[2]] = v
    bundle_ns[b] = dict(gen=g, fake=f)
    dm = spl == "development"
    dv = {}
    for a in ACT:
        s = dm & (act == a)
        dv[a] = dict(gen=ns[s & (lab == 0)].copy(), fake=ns[s & (lab == 1)].copy(),
                     gen_dur=dur[s & (lab == 0)].copy(), fake_dur=dur[s & (lab == 1)].copy())
    dev_ns[b] = dv

# ---------------- (a) duration <-> nsamp coupling -------------------------------------
cpl = {}
for mod in ("imu_only", "trajectory_xytime"):
    for ai, a in enumerate(ACT):
        b = IDX["bundle_of"][(a, mod)]
        k = f"{a}|{b}"
        if k in cpl:
            continue
        gd = IDX["bundleD"][b]["gen"][IDX["gen_action"] == ai]
        gn = bundle_ns[b]["gen"][IDX["gen_action"] == ai]
        fd = IDX["bundleD"][b]["fake"][ai].ravel()
        fn = bundle_ns[b]["fake"][ai].ravel()
        # conditional spread of duration given nsamp, on genuine
        sp = []
        for v in np.unique(gn):
            m = gn == v
            if m.sum() >= 5:
                sp.append(gd[m].std())
        cpl[k] = dict(
            corr_gen=float(np.corrcoef(gd, gn)[0, 1]), corr_fake=float(np.corrcoef(fd, fn)[0, 1]),
            r2_gen=float(np.corrcoef(gd, gn)[0, 1] ** 2),
            mean_sd_of_duration_given_nsamp_genuine_s=float(np.mean(sp)) if sp else None,
            sd_of_duration_genuine_s=float(gd.std()),
            frac_variance_of_duration_left_after_nsamp=(
                float(np.mean(np.array(sp) ** 2) / gd.var()) if sp else None),
        )
out["duration_vs_samplecount"] = cpl

# ---------------- (b) sample-count-only session detector -------------------------------
def eval_slot_var(R, gvals, fvals_draws, tabs, fold):
    slot_src = 1 - fold[R.slot_user]; sess_src = 1 - fold[R.sess_user]
    gS = R.agg_sum(AB.slot_lookup_llr(R, tabs, gvals))
    nulls = [np.sort(gS[R.sess_user % 2 == s]) for s in (0, 1)]
    pg = np.where(sess_src == 0, AB.pvals(gS, nulls[0]), AB.pvals(gS, nulls[1]))
    c = 0.0
    for fv in fvals_draws:
        fS = R.agg_sum(AB.slot_lookup_llr(R, tabs, fv))
        pf = np.where(sess_src == 0, AB.pvals(fS, nulls[0]), AB.pvals(fS, nulls[1]))
        c += float((pf <= 0.05).mean())
    return dict(caught=c / len(fvals_draws), frr=float((pg <= 0.05).mean()))


fold = np.arange(NU) % 2
rng = np.random.default_rng(20260810)
res_b = {}
for mod in ("imu_only", "trajectory_xytime"):
    gen_dur = AB._combine_gen([IDX["bundleD"][IDX["bundle_of"][(ACT[ai], mod)]]["gen"]
                               for ai in range(5)], IDX["gen_action"])
    fake_dur = np.stack([IDX["bundleD"][IDX["bundle_of"][(ACT[ai], mod)]]["fake"][ai]
                         for ai in range(5)])
    gen_ns = AB._combine_gen([bundle_ns[IDX["bundle_of"][(ACT[ai], mod)]]["gen"]
                              for ai in range(5)], IDX["gen_action"])
    fake_ns = np.stack([bundle_ns[IDX["bundle_of"][(ACT[ai], mod)]]["fake"][ai]
                        for ai in range(5)])
    dev = [IDX["devD"][IDX["bundle_of"][(a, mod)]][a] for a in ACT]
    devn = [dev_ns[IDX["bundle_of"][(a, mod)]][a] for a in ACT]
    tab_dur = [AB.fit_qllr(dev[ai]["gen"], dev[ai]["fake"]) for ai in range(5)]
    tab_ns = [AB.fit_qllr(devn[ai]["gen"], devn[ai]["fake"]) for ai in range(5)]

    # B's randomised mid-rank PIT map, fake arm only, dev-fitted
    qmap = [(np.sort(dev[ai]["fake"]), np.sort(dev[ai]["gen"])) for ai in range(5)]

    def apply_qmap(v, ai_arr, rr):
        o = np.empty_like(v)
        for ai in range(5):
            m = ai_arr == ai
            if not m.any():
                continue
            fs, gs = qmap[ai]
            lo = np.searchsorted(fs, v[m], side="left").astype(float)
            hi = np.searchsorted(fs, v[m], side="right").astype(float)
            u = (lo + rr.random(int(m.sum())) * np.maximum(hi - lo, 1.0)) / len(fs)
            j = np.clip((u * len(gs)).astype(np.int64), 0, len(gs) - 1)
            o[m] = gs[j]
        return o

    for reg in ("touch",):
        R = REG[reg]
        draws_pos = [AB.draw_fake_slots(R, 4242 + 13 * s) for s in range(20)]
        d_draws = [fake_dur[R.slot_action, R.slot_user, p] for p in draws_pos]
        n_draws = [fake_ns[R.slot_action, R.slot_user, p] for p in draws_pos]
        rq = np.random.default_rng(5150)
        dq = [apply_qmap(d, R.slot_action, rq) for d in d_draws]
        e = {}
        e["duration_raw"] = eval_slot_var(R, gen_dur[R.slot_gid], d_draws, tab_dur, fold)
        e["duration_quantile_mapped"] = eval_slot_var(R, gen_dur[R.slot_gid], dq, tab_dur, fold)
        e["samplecount_raw"] = eval_slot_var(R, gen_ns[R.slot_gid], n_draws, tab_ns, fold)
        e["samplecount_after_duration_map"] = eval_slot_var(R, gen_ns[R.slot_gid], n_draws,
                                                            tab_ns, fold)
        # (c) feasibility of the mapped durations
        cap = np.array([max(dev[ai]["fake"].max(), fake_dur[ai].max()) for ai in range(5)])
        allmapped = np.concatenate(dq)
        allact = np.tile(R.slot_action, len(dq))
        infeas = allmapped > cap[allact] + 1e-9
        # never-emitted values: mapped value not present in the union of all generator outputs
        emitted = {ai: set(np.round(np.concatenate([dev[ai]["fake"], fake_dur[ai].ravel()]), 6))
                   for ai in range(5)}
        never = np.zeros(len(allmapped), bool)
        for ai in range(5):
            m = allact == ai
            never[m] = np.array([round(float(x), 6) not in emitted[ai] for x in allmapped[m]])
        e["mapped_duration_feasibility"] = dict(
            frac_above_generator_max=float(infeas.mean()),
            frac_never_emitted_by_generator=float(never.mean()),
            per_action={ACT[ai]: dict(
                frac_above_max=float(infeas[allact == ai].mean()),
                frac_never_emitted=float(never[allact == ai].mean()),
                gen_max_s=float(cap[ai])) for ai in range(4)},
        )
        # (d) five-shot-limited map: attacker owns 5 genuine durations per (victim, action)
        rr5 = np.random.default_rng(99)
        five = {}
        for ai in range(5):
            g = IDX["bundleD"][IDX["bundle_of"][(ACT[ai], mod)]]["gen"]
            for u in range(NU):
                sel = (IDX["gen_action"] == ai) & (IDX["gen_user"] == u)
                v = g[sel]
                five[(ai, u)] = np.sort(rr5.choice(v, size=min(5, len(v)), replace=False))

        def apply_qmap5(v, ai_arr, ui_arr, rr):
            o = np.empty_like(v)
            for ai in range(5):
                for u in range(NU):
                    m = (ai_arr == ai) & (ui_arr == u)
                    if not m.any():
                        continue
                    fs = qmap[ai][0]; gs = five[(ai, u)]
                    lo = np.searchsorted(fs, v[m], side="left").astype(float)
                    hi = np.searchsorted(fs, v[m], side="right").astype(float)
                    uu = (lo + rr.random(int(m.sum())) * np.maximum(hi - lo, 1.0)) / len(fs)
                    j = np.clip((uu * len(gs)).astype(np.int64), 0, len(gs) - 1)
                    o[m] = gs[j]
            return o

        rq5 = np.random.default_rng(717)
        dq5 = [apply_qmap5(d, R.slot_action, R.slot_user, rq5) for d in d_draws]
        e["duration_map_fiveshot_target"] = eval_slot_var(R, gen_dur[R.slot_gid], dq5,
                                                          tab_dur, fold)
        res_b[f"{mod}|{reg}"] = e
out["duration_channel"] = res_b
json.dump(out, open(os.path.join(RV, "rv5.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
