#!/usr/bin/env python
"""Reviewer check 2.

Three things the released measurement does not do:
  (S1) honest SEQUENTIAL rejection sampling  (generate r fresh, keep the best 1, per slot)
       vs the released rule (keep the global best 200/r once, then permute with wraparound).
  (S2) the released rule with within-session EXACT DUPLICATES forbidden.
  (S3) a knowledge-free selection rule the study never ran: duration typicality.
Calibration follows agent B (cross-fitted, out-of-sample null).  18 touch cells.
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


def load():
    D = pickle.load(open(os.path.join(RV, "idx.pkl"), "rb"))
    return D["IDX"], D["REG"], D["cm"]


def rank_from_key(key, P):
    """global top-P positions per (action,user) from a ranking key (lower = attacker prefers)."""
    return np.argsort(key, axis=2)[:, :, :P]


def draw_topk(R, keep, P, rng):
    ordm = np.argsort(rng.random((R.n_groups, P)), axis=1)
    col = ordm[R.group, R.rank % P]
    return keep[R.slot_action, R.slot_user, col]


def draw_topk_nodup(R, key, P, rng):
    """No within-session duplicates: a group of size c takes the best max(c,P) items,
    i.e. it extends into the pool rather than repeating an event."""
    order = np.argsort(key, axis=2)                      # (5,NU,200) best-first
    pos = np.empty(R.M, np.int64)
    need = R.group_count[R.group]
    take = np.maximum(need, P)                            # how deep into the ranking this group goes
    for g in range(R.n_groups):
        m = R.group == g
        c = int(R.group_count[g])
        t = int(max(c, P))
        sel = rng.choice(t, size=c, replace=False)
        pos[m] = order[R.group_action[g], R.group_user[g], sel]
    return pos


def draw_seq(R, key, r, rng):
    """Honest sequential rejection sampling: r fresh candidates per slot, keep argmin(key)."""
    cand = rng.integers(0, 200, size=(R.M, r))
    k = key[R.slot_action[:, None], R.slot_user[:, None], cand]
    return cand[np.arange(R.M), np.argmin(k, axis=1)]


def run_cell(args):
    mod, det, reg = args
    IDX, REG = G["IDX"], G["REG"]
    NU = IDX["NU"]
    R = REG[reg]
    fold = np.arange(NU) % 2
    gen_action = IDX["gen_action"]
    cells = {a: IDX["cellS"][f"{a}__{mod}__{det}"] for a in ACT}
    gen_score = AB._combine_gen([cells[ACT[ai]]["gen"] for ai in range(5)], gen_action)
    fake_score = np.stack([cells[ACT[ai]]["fake"][ai] for ai in range(5)])
    tau = np.array([cells[a]["tau"] for a in ACT])
    tau_slot = tau[R.slot_action]
    gen_dur = AB._combine_gen([IDX["bundleD"][IDX["bundle_of"][(ACT[ai], mod)]]["gen"]
                               for ai in range(5)], gen_action)
    fake_dur = np.stack([IDX["bundleD"][IDX["bundle_of"][(ACT[ai], mod)]]["fake"][ai]
                         for ai in range(5)])

    # cross-fitted score LLR tables (agent-B construction)
    tabs = {}
    for src in (0, 1):
        cal = np.nonzero(fold == src)[0]
        tt = []
        for ai in range(5):
            gsel = (gen_action == ai) & np.isin(IDX["gen_user"], cal)
            tt.append(AB.fit_qllr(gen_score[gsel], fake_score[ai][cal].ravel()))
        tabs[src] = tt
    slot_src = 1 - fold[R.slot_user]
    sess_src = 1 - fold[R.sess_user]

    def sess_llr(sc):
        l0 = AB.slot_lookup_llr(R, tabs[0], sc)
        l1 = AB.slot_lookup_llr(R, tabs[1], sc)
        return R.agg_sum(np.where(slot_src == 0, l0, l1))

    gS = sess_llr(gen_score[R.slot_gid])
    nulls = [np.sort(gS[R.sess_user % 2 == s]) for s in (0, 1)]
    pg = np.where(sess_src == 0, AB.pvals(gS, nulls[0]), AB.pvals(gS, nulls[1]))
    n_gen_u = np.bincount(R.sess_user, minlength=NU).astype(float)
    frr = float((pg <= 0.05).mean())

    # ---- selection keys
    keys = {}
    ens = 0.0; n = 0
    for sm in AB.MODALITIES:
        for sd in AB.DETECTORS:
            if (sm, sd) == (mod, det):
                continue
            if sm != mod:
                continue
            x = np.stack([IDX["cellS"][f"{ACT[ai]}__{sm}__{sd}"]["fake"][ai] for ai in range(5)])
            o = np.argsort(np.argsort(x, axis=2), axis=2).astype(np.float64) / 199.0
            ens = ens + o; n += 1
    keys["A1_ens5"] = ens / n
    keys["A2_centroid"] = IDX["a2c"][mod]
    keys["A3_oracle"] = fake_score
    # knowledge-free duration-typicality selection (attacker knows human duration stats)
    dev = [IDX["devD"][IDX["bundle_of"][(a, mod)]][a] for a in ACT]
    devmed = np.array([np.median(d["gen"]) for d in dev])
    keys["A4_dur_typicality"] = np.abs(fake_dur - devmed[:, None, None])
    # duration-LLR selection: attacker fits the SAME kind of dev-based duration LLR
    durtab = [AB.fit_qllr(dev[ai]["gen"], dev[ai]["fake"]) for ai in range(5)]
    dl = np.empty_like(fake_dur)
    for ai in range(5):
        dl[ai] = AB.apply_qllr(durtab[ai], fake_dur[ai].ravel()).reshape(fake_dur[ai].shape)
    keys["A5_dur_llr"] = dl

    res = {}
    for kname, key in keys.items():
        for r in (5, 10, 20):
            P = 200 // r
            keep = rank_from_key(key, P)
            for mode in ("topk", "topk_nodup", "seq"):
                cu = 0.0; den = 0.0; farn = 0.0; fard = 0.0
                dupn = 0.0
                for s in range(NSEED):
                    rng = np.random.default_rng(zlib.crc32(
                        f"{mod}|{det}|{reg}|{kname}|{r}|{mode}|{s}".encode()))
                    if mode == "topk":
                        pos = draw_topk(R, keep, P, rng)
                    elif mode == "topk_nodup":
                        pos = draw_topk_nodup(R, key, P, rng)
                    else:
                        pos = draw_seq(R, key, r, rng)
                    fs = fake_score[R.slot_action, R.slot_user, pos]
                    farn += float((fs < tau_slot).sum()); fard += R.M
                    # within-session duplicate count
                    gk = R.slot_sess.astype(np.int64) * 1000 + pos
                    dupn += R.M - len(np.unique(gk))
                    fS = sess_llr(fs)
                    pf = np.where(sess_src == 0, AB.pvals(fS, nulls[0]), AB.pvals(fS, nulls[1]))
                    cu += float((pf <= 0.05).sum()); den += R.n_sess
                res[f"{kname}|r{r}|{mode}"] = dict(
                    caught=cu / den, per_event_FAR=farn / fard,
                    dup_slots_frac=dupn / fard)
    return (mod, det, reg), dict(frr=frr, res=res)


if __name__ == "__main__":
    IDX, REG, cm = load()
    G["IDX"] = IDX; G["REG"] = REG
    jobs = [(m, d, "touch") for m in AB.MODALITIES for d in AB.DETECTORS]
    t0 = time.time()
    outs = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for k, v in ex.map(run_cell, jobs):
            outs["|".join(k)] = v
            print("done", k, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(outs, open(os.path.join(RV, "rv2_touch.json"), "w"), indent=1)
    # aggregate
    agg = {}
    frrs = np.mean([v["frr"] for v in outs.values()])
    keysall = sorted(next(iter(outs.values()))["res"])
    print(f"\nmean touch genuine session FRR = {frrs:.4f}  (18 cells, B-style cross-fit)")
    for kk in keysall:
        c = np.mean([v["res"][kk]["caught"] for v in outs.values()])
        f = np.mean([v["res"][kk]["per_event_FAR"] for v in outs.values()])
        d = np.mean([v["res"][kk]["dup_slots_frac"] for v in outs.values()])
        nbelow = sum(1 for v in outs.values() if v["res"][kk]["caught"] <= v["frr"])
        agg[kk] = dict(caught=c, per_event_FAR=f, dup_frac=d, cells_below_frr=nbelow)
        print(f"{kk:42s} caught={c:.4f}  FAR={f:.4f}  dup={d:.4f}  cells_collapsed={nbelow}/18")
    json.dump(dict(mean_frr=float(frrs), agg=agg),
              open(os.path.join(RV, "rv2_touch_agg.json"), "w"), indent=1)
