#!/usr/bin/env python
"""Referee Part B: duration-only session floor, the prescribed control, and the
four quantile-map variants that A and B disagree about."""
import os, json, glob, pickle, collections
import numpy as np
import referee as RF

OUT = os.path.dirname(os.path.abspath(__file__))
ACTIONS = RF.ACTIONS
AIDX = RF.AIDX
POOL = 200
N_REPS = 20
SH = "/mnt/share/mwang49/data7/results/direct100k/{b}/shards/{u}.npz"
REL_BUNDLE = {a: ("replay_dataset_v15" if a == "scroll" else "replay_dataset_full")
              for a in ACTIONS}
FULL_BUNDLE = {a: "replay_dataset_full" for a in ACTIONS}


def user_split():
    sp = {}
    for p in sorted(glob.glob(SH.format(b="replay_dataset_full", u="*"))):
        z = np.load(p, allow_pickle=True)
        sp[os.path.basename(p)[:-4]] = str(z["split"])
    return sp


def durations(bundle_map, users):
    """(action, label) -> {user -> array}, plus event-keyed table."""
    tab = {}
    for b in sorted(set(bundle_map.values())):
        for u in users:
            z = np.load(SH.format(b=b, u=u), allow_pickle=True)
            tf, off, eid, act, lab = (z["trajectory_flat"], z["offsets"], z["event_id"],
                                      z["action"], z["label"])
            dd = tf[off[1:] - 1, 7].astype(float)
            for i in range(len(eid)):
                if bundle_map[str(act[i])] == b:
                    tab[str(eid[i])] = (dd[i], str(act[i]), int(lab[i]), u)
    return tab


def main():
    users, UIDX, gen_rows, gen_index, fake_pos, cellS = RF.read_release()
    NU = len(users)
    S = RF.build_sessions(users, UIDX, gen_rows)
    mixed = set(map(str, S["binding"]["mixed_ids"])) | set(map(str, S["artifact"]["mixed_ids"]))
    sess = {"touch": S["binding"]["sessions"]["touch"],
            "keystroke": [s for s in S["binding"]["sessions"]["keystroke"]
                          if str(s["sid"]) not in mixed]}
    REG = {r: RF.Regime(sess[r], NU) for r in ("touch", "keystroke")}
    spl = user_split()
    dev_users = sorted(u for u, s in spl.items() if s == "development")
    print("dev users", len(dev_users), dev_users)

    res = {}
    for tag, BM in (("release", REL_BUNDLE), ("full", FULL_BUNDLE)):
        tabT = durations(BM, users)
        tabD = durations(BM, dev_users)
        gd = {a: np.array([tabT[e][0] for e, _, _ in gen_rows[a]]) for a in ACTIONS}
        fd = np.zeros((5, NU, POOL))
        for a in ACTIONS:
            for e, (ui, p) in fake_pos[a].items():
                fd[AIDX[a], ui, p] = tabT[e][0]
        devg = {a: np.array([v[0] for v in tabD.values() if v[1] == a and v[2] == 0])
                for a in ACTIONS}
        devf = {a: np.array([v[0] for v in tabD.values() if v[1] == a and v[2] == 1])
                for a in ACTIONS}
        bulk = {}
        for a in ACTIONS:
            G, F = gd[a], fd[AIDX[a]].ravel()
            cap = float(F.max())
            bulk[a] = dict(gen_p5=float(np.percentile(G, 5)), gen_p50=float(np.median(G)),
                           gen_p95=float(np.percentile(G, 95)), gen_max=float(G.max()),
                           fake_p5=float(np.percentile(F, 5)), fake_p50=float(np.median(F)),
                           fake_p95=float(np.percentile(F, 95)), fake_max=cap,
                           frac_fake_on_cap=float(np.mean(np.isclose(F, cap))),
                           frac_genuine_above_cap=float(np.mean(G > cap)),
                           median_shift_gen_minus_fake=float(np.median(G) - np.median(F)))
        res[f"bulk|{tag}"] = bulk

        # ---------- transforms
        cap = {a: float(devf[a].max()) for a in ACTIONS}
        rj = np.random.default_rng(909)

        def clipjit(v, a):
            return np.minimum(v, cap[a]) + rj.uniform(-0.005, 0.005, np.shape(v))

        # (1) A's dequantised interpolating map, applied to BOTH arms
        dqg = {a: devg[a] + rj.uniform(-0.005, 0.005, devg[a].shape) for a in ACTIONS}
        dqf = {a: devf[a] + rj.uniform(-0.005, 0.005, devf[a].shape) for a in ACTIONS}
        dqsrc = {a: np.sort(dqf[a]) for a in ACTIONS}
        dqtgt = {a: np.sort(dqg[a]) for a in ACTIONS}

        def A_dq_gen(v, a):
            return np.asarray(v, float) + rj.uniform(-0.005, 0.005, np.shape(v))

        def A_dq_fake(v, a):
            x = np.asarray(v, float) + rj.uniform(-0.005, 0.005, np.shape(v))
            u = np.clip((np.searchsorted(dqsrc[a], x, side="right") - 0.5) / len(dqsrc[a]), 0, 1)
            return np.quantile(dqtgt[a], u)

        # (2) A's deterministic rank-preserving map (fake only)
        src = {a: np.sort(devf[a]) for a in ACTIONS}
        tgt = {a: np.sort(devg[a]) for a in ACTIONS}

        def A_rank_fake(v, a):
            x = np.asarray(v, float)
            u = np.clip((np.searchsorted(src[a], x, side="right") - 0.5) / len(src[a]), 0, 1)
            i = np.clip(np.rint(u * (len(tgt[a]) - 1)).astype(int), 0, len(tgt[a]) - 1)
            return tgt[a][i]

        # (3) B's randomised mid-rank PIT (fake only, lands on observed genuine values)
        rq = np.random.default_rng(5150)

        def B_rand_fake(v, a):
            x = np.asarray(v, float)
            lo = np.searchsorted(src[a], x, side="left").astype(float)
            hi = np.searchsorted(src[a], x, side="right").astype(float)
            u = (lo + rq.random(np.shape(x)) * np.maximum(hi - lo, 1.0)) / len(src[a])
            j = np.clip((u * len(tgt[a])).astype(int), 0, len(tgt[a]) - 1)
            return tgt[a][j]

        ident = lambda v, a: np.asarray(v, float)
        variants = {
            "raw": (ident, ident, "fit_on_transformed_dev"),
            "control_clip_cap_plus_jitter": (clipjit, clipjit, "fit_on_transformed_dev"),
            "A_dequantised_interpolating_map_both_arms": (A_dq_gen, A_dq_fake,
                                                          "fit_on_transformed_dev"),
            "A_rank_preserving_map_fake_only": (ident, A_rank_fake, "fit_on_transformed_dev"),
            "B_randomised_midrank_map_fake_only__refit": (ident, B_rand_fake,
                                                          "fit_on_transformed_dev"),
            "B_randomised_midrank_map_fake_only__fixed_detector": (ident, B_rand_fake, "fixed"),
        }
        for vname, (gt, ft, fitmode) in variants.items():
            if fitmode == "fixed":
                fit = {a: RF.qllr_fit(devg[a], devf[a]) for a in ACTIONS}
            else:
                fit = {a: RF.qllr_fit(gt(devg[a], a), ft(devf[a], a)) for a in ACTIONS}
            GD = {a: gt(gd[a], a) for a in ACTIONS}
            FD = np.stack([ft(fd[AIDX[a]], a) for a in ACTIONS])
            for regime in ("touch", "keystroke"):
                reg = REG[regime]
                acts = sorted(set(reg.act.tolist()))
                gl = np.zeros(reg.M)
                for ai in acts:
                    mk = reg.act == ai
                    gl[mk] = RF.qllr(fit[ACTIONS[ai]], GD[ACTIONS[ai]][reg.row[mk]])
                GS = reg.agg(gl)
                par = np.array([i % 2 for i in range(NU)])[reg.user]
                pg = np.full(reg.n, np.nan)
                for f in (0, 1):
                    null = np.sort(GS[par == f])
                    pg[par != f] = RF.tail_p(null, GS[par != f])
                keep = np.tile(np.arange(POOL), (5, NU, 1))
                PF = []
                for rep in range(N_REPS):
                    rng = np.random.default_rng((hash((vname, tag, regime, rep)) & 0x7fffffff))
                    pos = RF.draw(reg, keep, rng)
                    fl = np.zeros(reg.M)
                    for ai in acts:
                        mk = reg.act == ai
                        fl[mk] = RF.qllr(fit[ACTIONS[ai]], FD[ai][reg.uid[mk], pos[mk]])
                    FS = reg.agg(fl)
                    p = np.full(reg.n, np.nan)
                    for f in (0, 1):
                        null = np.sort(GS[par == f])
                        p[par != f] = RF.tail_p(null, FS[par != f])
                    PF.append(p)
                PF = np.array(PF)
                d = {}
                for al in (0.05, 0.01):
                    d[f"caught@{al}"] = float((PF <= al).mean())
                    d[f"frr@{al}"] = float((pg <= al).mean())
                d["ongrid_fake"] = float(np.mean(np.abs(FD * 100 - np.rint(FD * 100)) < 1e-6))
                d["ongrid_gen"] = float(np.mean([np.mean(np.abs(GD[a] * 100 -
                                                np.rint(GD[a] * 100)) < 1e-6) for a in ACTIONS]))
                res[f"dur|{tag}|{vname}|{regime}"] = d
            print(tag, vname, "done", flush=True)
    json.dump(res, open(os.path.join(OUT, "referee_partB.json"), "w"), indent=1)
    print("wrote referee_partB.json")


if __name__ == "__main__":
    main()
