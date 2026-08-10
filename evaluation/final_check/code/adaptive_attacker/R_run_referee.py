#!/usr/bin/env python
"""Referee Part A: 18 cells x 2 regimes x 3 calibration constructions."""
import os, json, pickle, collections
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import referee as RF

OUT = os.path.dirname(os.path.abspath(__file__))
ACTIONS = RF.ACTIONS
COMBOS = RF.COMBOS
DETECTORS = RF.DETECTORS
POOL = RF.POOL
R_LIST = RF.R_LIST
N_REPS = RF.N_REPS
G = {}


def prepare():
    users, UIDX, gen_rows, gen_index, fake_pos, cellS = RF.read_release()
    NU = len(users)
    S = RF.build_sessions(users, UIDX, gen_rows)
    # UNION rule: a session is mixed if EITHER inventory shows keystroke with touch
    mixed = set(map(str, S["binding"]["mixed_ids"])) | set(map(str, S["artifact"]["mixed_ids"]))
    sess = {"touch": S["binding"]["sessions"]["touch"],
            "keystroke": [s for s in S["binding"]["sessions"]["keystroke"]
                          if str(s["sid"]) not in mixed]}
    A2 = RF.build_a2(users, fake_pos)
    gu = {a: np.array([UIDX[u] for _, u, _ in gen_rows[a]]) for a in ACTIONS}
    return dict(users=users, NU=NU, cellS=cellS, A2=A2, gen_user_of_row=gu,
                sess=sess, S=S, mixed_union=sorted(mixed))


def worker(arg):
    m, d = arg
    NU = G["NU"]; cellS = G["cellS"]; A2 = G["A2"]; gur = G["gen_user_of_row"]
    parity = np.array([i % 2 for i in range(NU)])
    mod3 = np.array([i % 3 for i in range(NU)])
    out = []
    for regime in ("touch", "keystroke"):
        reg = G["REG"][regime]
        acts = sorted(set(reg.act.tolist()))
        cells = {a: cellS[f"{a}__{m}__{d}"] for a in ACTIONS}
        gen = {a: cells[a]["gen"] for a in ACTIONS}
        fake = np.stack([cells[a]["fake"] for a in ACTIONS])
        tau = np.array([cells[a]["tau"] for a in ACTIONS])
        g_sc = np.zeros(reg.M)
        for ai in acts:
            mk = reg.act == ai
            g_sc[mk] = gen[ACTIONS[ai]][reg.row[mk]]

        def fit_tables(groups, gid):
            T = []
            for grp in groups:
                cu = np.nonzero(gid == grp)[0]
                t = []
                for ai in range(5):
                    a = ACTIONS[ai]
                    sel = np.isin(gur[a], cu)
                    t.append(RF.qllr_fit(gen[a][sel], fake[ai][cu].ravel()))
                T.append(t)
            return T
        T2 = fit_tables([0, 1], parity)
        T3 = fit_tables([0, 1, 2], mod3)

        def sess_stats(sc):
            o = {}
            for tag, TT in (("p", T2), ("m", T3)):
                for k, t in enumerate(TT):
                    v = np.zeros(reg.M)
                    for ai in acts:
                        mk = reg.act == ai
                        v[mk] = RF.qllr(t[ai], sc[mk])
                    o[f"{tag}{k}"] = reg.agg(v)
            return o
        GS = sess_stats(g_sc)
        sp = parity[reg.user]; sm = mod3[reg.user]

        def pvec(ST):
            P = {}
            p = np.full(reg.n, np.nan)
            for f in (0, 1):
                null = np.sort(GS[f"p{f}"][sp == f]); ev = sp != f
                p[ev] = RF.tail_p(null, ST[f"p{f}"][ev])
            P["A_insample_null"] = p
            p = np.full(reg.n, np.nan)
            for s in (0, 1):
                ev = sp == s
                null = np.sort(GS[f"p{s}"][sp == 1 - s])
                p[ev] = RF.tail_p(null, ST[f"p{1-s}"][ev])
            P["B_crossfit_null"] = p
            p = np.full(reg.n, np.nan)
            for e in (0, 1, 2):
                f = (e + 1) % 3; n_ = (e + 2) % 3
                null = np.sort(GS[f"m{f}"][sm == n_]); ev = sm == e
                p[ev] = RF.tail_p(null, ST[f"m{f}"][ev])
            P["R_threeway_disjoint"] = p
            return P
        PG = pvec(GS)

        orders = {"A0|uniform": np.tile(np.arange(POOL), (5, NU, 1))}
        for tag in ("A", "B"):
            for var in ("centroid", "knn"):
                st = np.stack([A2[f"{tag}|{m}|{a}|{var}"] for a in ACTIONS])
                orders[f"A2{tag}_{var}|-"] = np.argsort(st, axis=2, kind="stable")
        ens = 0.0
        for sd_ in DETECTORS:
            if sd_ == d:
                continue
            sx = np.stack([cellS[f"{a}__{m}__{sd_}"]["fake"] for a in ACTIONS])
            orders[f"A1|{m}|{sd_}"] = np.argsort(sx, axis=2, kind="stable")
            ens = ens + np.stack([RF.pool_rank(sx[ai]) for ai in range(5)])
        orders["A1ENS5|same_modality"] = np.argsort(ens / 5.0, axis=2, kind="stable")
        orders["A3|oracle"] = np.argsort(fake, axis=2, kind="stable")

        specs = [("A0|uniform", 1)]
        for k in orders:
            if k == "A0|uniform":
                continue
            for r in R_LIST[1:]:
                specs.append((k, r))

        for okey, r in specs:
            P = max(1, POOL // r)
            keep = np.ascontiguousarray(orders[okey][:, :, :P])
            acc = {k: [] for k in PG}
            far = 0.0
            for rep in range(N_REPS):
                rng = np.random.default_rng(
                    (hash((m, d, regime, okey, r, rep)) & 0x7fffffff))
                pos = RF.draw(reg, keep, rng)
                f_sc = fake[reg.act, reg.uid, pos]
                far += float((f_sc < tau[reg.act]).mean())
                PF = pvec(sess_stats(f_sc))
                for k in acc:
                    acc[k].append(PF[k])
            rec = dict(regime=regime, modality=m, detector=d, arm=okey, r=r,
                       n_sessions=int(reg.n), n_slots=int(reg.M),
                       forced_reuse_frac=float((reg.rank >= P).mean()),
                       per_event_far_slotweighted=far / N_REPS)
            for k in PG:
                pf = np.array(acc[k])
                for al in (0.05, 0.01):
                    rec[f"{k}|caught@{al}"] = float((pf <= al).mean())
                    rec[f"{k}|frr@{al}"] = float((PG[k] <= al).mean())
            out.append(rec)
    return out


def init(state):
    G.update(state)
    G["REG"] = {r: RF.Regime(state["sess"][r], state["NU"]) for r in ("touch", "keystroke")}


def main():
    st = prepare()
    print("touch", len(st["sess"]["touch"]), "keystroke", len(st["sess"]["keystroke"]),
          "mixed_union", len(st["mixed_union"]), flush=True)
    init(st)
    for r in ("touch", "keystroke"):
        R = G["REG"][r]
        print(r, "sessions", R.n, "slots", R.M, "max group", R.gcount.max(), flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=6, initializer=init, initargs=(st,)) as ex:
        for i, o in enumerate(ex.map(worker, COMBOS)):
            res += o
            print("combo", i + 1, "/", len(COMBOS), flush=True)
    json.dump(res, open(os.path.join(OUT, "referee_partA.json"), "w"))
    json.dump(dict(mixed_union=st["mixed_union"],
                   n_touch=len(st["sess"]["touch"]),
                   n_keystroke=len(st["sess"]["keystroke"]),
                   binding_mixed=sorted(map(str, st["S"]["binding"]["mixed_ids"])),
                   artifact_mixed=sorted(map(str, st["S"]["artifact"]["mixed_ids"]))),
              open(os.path.join(OUT, "referee_sessions.json"), "w"), indent=1)
    print("done", len(res))


if __name__ == "__main__":
    main()
