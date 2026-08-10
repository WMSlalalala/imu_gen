#!/usr/bin/env python
"""Referee implementation: independent third recomputation to arbitrate A vs B.

Reads ONLY frozen release files (through release_cell_map.json) and the two
agents' caches for the heavy per-event feature arrays (which are themselves
verified against the release here).  No GPU, no models.
"""
import json, gzip, os, sys, collections
import numpy as np

CM = json.load(open("/mnt/share/mwang49/data7/code/baselines/release_cell_map.json"))
CELLS = CM["cells"]
ACTIONS = ["tap", "scroll", "swipe", "pinch", "keystroke"]
AIDX = {a: i for i, a in enumerate(ACTIONS)}
MODALITIES = ["imu_only", "trajectory_xytime", "imu_trajectory_xytime"]
DETECTORS = ["authconformer", "behaveformer_stdat", "hmog_style_rf",
             "hmog_style_svm", "paper_svm", "paper_xgboost"]
COMBOS = [(m, d) for m in MODALITIES for d in DETECTORS]
ACACHE = ("/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/"
          "e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/A/cache")
BCACHE = ("/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/"
          "e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/B/cache")
OUT = os.path.dirname(os.path.abspath(__file__))
NQ, SMOOTH, POOL = 41, 0.5, 200
R_LIST = [1, 2, 5, 10, 20]
N_REPS = 20


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- release read
def read_release():
    """Per-cell score arrays indexed in a canonical event order."""
    gen_rows = {}          # action -> list of (event_id, user, session)
    fake_pos = {}          # action -> eid -> (ui, p)
    users = None
    for a in ACTIONS:
        ref = CELLS[f"{a}__imu_only__paper_xgboost"]
        g, f = [], collections.defaultdict(list)
        for line in gzip.open(ref["scores"], "rt"):
            d = json.loads(line)
            if d["label"] == 0:
                g.append((d["event_id"], d["user_id"], d["session_id"]))
            else:
                f[d["user_id"]].append(d["event_id"])
        g.sort()
        gen_rows[a] = g
        if users is None:
            users = sorted(f)
        assert users == sorted(f)
        fp = {}
        for u in users:
            s = sorted(f[u])
            assert len(s) == POOL
            for p, e in enumerate(s):
                fp[e] = (users.index(u), p)
        fake_pos[a] = fp
    NU = len(users)
    UIDX = {u: i for i, u in enumerate(users)}
    gen_index = {a: {e: i for i, (e, _, _) in enumerate(gen_rows[a])} for a in ACTIONS}
    cellS = {}
    for a in ACTIONS:
        for (m, d) in COMBOS:
            name = f"{a}__{m}__{d}"
            info = CELLS[name]
            gv = np.full(len(gen_rows[a]), np.nan)
            fv = np.full((NU, POOL), np.nan)
            for line in gzip.open(info["scores"], "rt"):
                r = json.loads(line)
                if r["label"] == 0:
                    gv[gen_index[a][r["event_id"]]] = r["fake_high_score"]
                else:
                    ui, p = fake_pos[a][r["event_id"]]
                    fv[ui, p] = r["fake_high_score"]
            assert not np.isnan(gv).any() and not np.isnan(fv).any()
            assert info["score_direction"] == "larger_is_more_fake"
            cellS[name] = dict(gen=gv, fake=fv, tau=float(info["frr5"]))
    return users, UIDX, gen_rows, gen_index, fake_pos, cellS


# ---------------------------------------------------------------- sessions
def build_sessions(users, UIDX, gen_rows):
    """Two constructions:
      'binding'  = full genuine binding inventory (agent B)
      'artifact' = session_rhythm_detector genuine artefact + fallback (agent A)
    Both keep only sessions with >=1 scored genuine event."""
    bd = np.load(os.path.join(BCACHE, "bindings.npz"), allow_pickle=True)
    tm = bd["split"] == "test"
    b_cluster = bd["source_cluster_id"][tm]; b_sid = bd["session_id"][tm]
    b_act = bd["action"][tm]; b_user = bd["user_id"][tm]
    b_start = bd["start_sample"][tm]; b_ord = bd["ordinal"][tm]
    # canonical genuine slot id = (action, row)
    gid_of_cluster = {}
    for a in ACTIONS:
        for i, (e, u, s) in enumerate(gen_rows[a]):
            gid_of_cluster[e[len("genuine-"):]] = (AIDX[a], i, UIDX[u], s)
    inv_actions = collections.defaultdict(list)
    inv_slots = collections.defaultdict(list)
    for i in range(len(b_sid)):
        inv_actions[b_sid[i]].append(b_act[i])
        g = gid_of_cluster.get(b_cluster[i])
        if g is not None:
            inv_slots[b_sid[i]].append((int(b_ord[i]), g[0], g[1], g[2], int(b_start[i])))
    out = {}
    for name in ("binding", "artifact"):
        sess = []
        drop = collections.Counter()
        for sid in sorted(inv_actions):
            slots = sorted(inv_slots.get(sid, []))
            if not slots:
                drop["no_scored_slot"] += 1
                continue
            if name == "binding":
                aset = set(inv_actions[sid])
            else:
                aset = None       # filled below
            sess.append((sid, slots, aset))
        out[name] = (sess, drop)
    # artifact composition
    art = {}
    for line in open("/mnt/share/mwang49/data7/session_rhythm_detector/results/sessions_genuine.jsonl"):
        r = json.loads(line)
        art[(int(r["user"]), int(r["session"]))] = r
    sess_art, drop_art = [], collections.Counter()
    for sid, slots, _ in out["artifact"][0]:
        uu = int(sid.split("_u")[1].split("_s")[0]); ss = int(sid.split("_s")[1])
        rec = art.get((uu, ss))
        aset = set(rec["actions"]) if rec is not None else {ACTIONS[s[1]] for s in slots}
        sess_art.append((sid, slots, aset))
    out["artifact"] = (sess_art, drop_art)
    fin = {}
    for name, (sess, drop) in out.items():
        keep = {"touch": [], "keystroke": []}
        nmixed = 0
        mixed_ids = []
        for sid, slots, aset in sess:
            if aset == {"keystroke"}:
                rg = "keystroke"
            elif "keystroke" not in aset:
                rg = "touch"
            else:
                nmixed += 1; mixed_ids.append(sid); continue
            keep[rg].append(dict(sid=sid, user=slots[0][3],
                                 slots=[(s[1], s[2]) for s in slots],
                                 starts=[s[4] for s in slots],
                                 n_total=len(aset) if False else None))
        fin[name] = dict(sessions=keep, dropped_mixed=nmixed, mixed_ids=mixed_ids,
                         dropped_noscore=int(drop.get("no_scored_slot", 0)))
    # inventory total slots per session (for the scored-slot fraction)
    fin["_inv_total"] = {s: len(v) for s, v in inv_actions.items()}
    return fin


class Regime:
    def __init__(self, sess_list, NU):
        self.sess = sess_list
        self.n = len(sess_list)
        self.user = np.array([s["user"] for s in sess_list], np.int64)
        si, ai, gr = [], [], []
        for i, s in enumerate(sess_list):
            for (a, row) in s["slots"]:
                si.append(i); ai.append(a); gr.append(row)
        self.sess_idx = np.array(si, np.int64)
        self.act = np.array(ai, np.int64)
        self.row = np.array(gr, np.int64)
        self.uid = self.user[self.sess_idx]
        self.M = len(si)
        key = self.sess_idx * 5 + self.act
        uk, inv = np.unique(key, return_inverse=True)
        self.group = inv
        self.ngroups = len(uk)
        rank = np.zeros(self.M, np.int64); cnt = np.zeros(self.ngroups, np.int64)
        for j in range(self.M):
            g = inv[j]; rank[j] = cnt[g]; cnt[g] += 1
        self.rank = rank
        self.gcount = cnt
        self.guser = np.zeros(self.ngroups, np.int64); self.guser[inv] = self.uid
        self.gact = np.zeros(self.ngroups, np.int64); self.gact[inv] = self.act
        self.nslots = np.bincount(self.sess_idx, minlength=self.n).astype(float)

    def agg(self, w):
        return np.bincount(self.sess_idx, weights=w, minlength=self.n)


# ---------------------------------------------------------------- LLR
def qllr_fit(gv, fv, nq=NQ, smooth=SMOOTH):
    both = np.concatenate([gv, fv])
    edges = np.unique(np.quantile(both, np.linspace(0, 1, nq)))[1:-1]
    if len(edges) == 0:
        edges = np.array([float(np.median(both))])
    nb = len(edges) + 1
    hg = np.bincount(np.searchsorted(edges, gv, side="right"), minlength=nb).astype(float)
    hf = np.bincount(np.searchsorted(edges, fv, side="right"), minlength=nb).astype(float)
    pg = (hg + smooth) / (hg.sum() + smooth * nb)
    pf = (hf + smooth) / (hf.sum() + smooth * nb)
    return edges, np.log(pf / pg)


def qllr(fit, v):
    return fit[1][np.searchsorted(fit[0], v, side="right")]


def tail_p(null_sorted, stat):
    n = len(null_sorted)
    ge = n - np.searchsorted(null_sorted, stat, side="left")
    return (ge + 1.0) / (n + 1.0)


def draw(reg, keep, rng):
    P = keep.shape[2]
    om = np.argsort(rng.random((reg.ngroups, P)), axis=1)
    col = om[reg.group, reg.rank % P]
    return keep[reg.act, reg.uid, col]


# ---------------------------------------------------------------- A2 features
SHARD = "/mnt/share/mwang49/data7/results/direct100k/{bundle}/shards/{user}.npz"
IMU_CH = list(range(6))
TRAJ_CH_B = [1, 2, 5, 6]
TRAJ_CH_A = [0, 1, 2, 3, 4, 5, 6, 8]


def bundle_features(bundle, users):
    """fake-event features, keyed by event_id.  Returns dict eid -> (fA54, fB_imu30, fB_traj20)."""
    out = {}
    for u in users:
        z = np.load(SHARD.format(bundle=bundle, user=u), allow_pickle=True)
        imu, tf, off = z["imu_flat"], z["trajectory_flat"], z["offsets"]
        eid, act, lab = z["event_id"], z["action"], z["label"]
        for i in range(len(eid)):
            if lab[i] != 1:
                continue
            lo, hi = int(off[i]), int(off[i + 1])
            X = imu[lo:hi].astype(np.float64); T = tf[lo:hi].astype(np.float64)
            fa = []
            for c in IMU_CH:
                v = X[:, c]
                dv = np.diff(v) if len(v) > 1 else np.zeros(1)
                fa += [v.mean(), v.std(), np.percentile(v, 5), np.percentile(v, 50),
                       np.percentile(v, 95), np.abs(dv).mean()]
            fa += [T[-1, 7], float(hi - lo)]
            for c in TRAJ_CH_A:
                fa += [T[:, c].mean(), T[:, c].std()]
            si = X[:, IMU_CH]; st = T[:, TRAJ_CH_B]
            fb_i = np.concatenate([si.mean(0), si.std(0),
                                   np.percentile(si, 5, axis=0), np.percentile(si, 50, axis=0),
                                   np.percentile(si, 95, axis=0)])
            fb_t = np.concatenate([st.mean(0), st.std(0),
                                   np.percentile(st, 5, axis=0), np.percentile(st, 50, axis=0),
                                   np.percentile(st, 95, axis=0)])
            out[str(eid[i])] = (np.array(fa), fb_i, fb_t)
    return out


def build_a2(users, fake_pos):
    """A2 selection statistics under BOTH feature specifications."""
    cache = os.path.join(OUT, "a2_referee.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    NU = len(users)
    bundle_of = {(a, m): CELLS[f"{a}__{m}__paper_xgboost"]["bundle"]
                 for a in ACTIONS for m in MODALITIES}
    needed = sorted({v for v in bundle_of.values()})
    feats = {}
    for b in needed:
        log("  features for", b)
        feats[b] = bundle_features(b, users)
    out = {}
    for m in MODALITIES:
        for a in ACTIONS:
            fb = feats[bundle_of[(a, m)]]
            XA = np.zeros((NU, POOL, 54)); XI = None; XT = None
            n_i = n_t = None
            for eid, (ui, p) in fake_pos[a].items():
                fa, fi, ft = fb[eid]
                if XI is None:
                    n_i, n_t = len(fi), len(ft)
                    XI = np.zeros((NU, POOL, n_i)); XT = np.zeros((NU, POOL, n_t))
                XA[ui, p] = fa; XI[ui, p] = fi; XT[ui, p] = ft
            XB = {"imu_only": XI, "trajectory_xytime": XT,
                  "imu_trajectory_xytime": np.concatenate([XI, XT], axis=2)}[m]
            for tag, X in (("A", XA), ("B", XB)):
                dc = np.zeros((NU, POOL)); dk = np.zeros((NU, POOL))
                for u in range(NU):
                    Z = X[u]
                    sd = Z.std(0); sd = np.where(sd < 1e-12, 1.0, sd)
                    Z = (Z - Z.mean(0)) / sd
                    dc[u] = np.linalg.norm(Z, axis=1)
                    q = (Z * Z).sum(1)
                    D = np.sqrt(np.maximum(q[:, None] + q[None, :] - 2.0 * (Z @ Z.T), 0.0))
                    np.fill_diagonal(D, np.inf)
                    dk[u] = np.sort(D, axis=1)[:, :10].mean(1)
                out[f"{tag}|{m}|{a}|centroid"] = dc
                out[f"{tag}|{m}|{a}|knn"] = dk
    np.savez_compressed(cache, **out)
    return out


# ---------------------------------------------------------------- main
def pool_rank(x):
    o = np.argsort(np.argsort(x, axis=1), axis=1).astype(np.float64)
    return o / (x.shape[1] - 1.0)


def main():
    log("reading release ...")
    users, UIDX, gen_rows, gen_index, fake_pos, cellS = read_release()
    NU = len(users)
    fars = {c: float((cellS[c]["fake"] < cellS[c]["tau"]).mean()) for c in cellS}
    log("FAR@frr5 mean over 90 cells = %.6f  (release says %.6f)"
        % (np.mean(list(fars.values())), CM["far5_mean_release"]))
    log("tap__imu_only__paper_xgboost FAR@frr5 = %.5f" % fars["tap__imu_only__paper_xgboost"])

    S = build_sessions(users, UIDX, gen_rows)
    for name in ("binding", "artifact"):
        d = S[name]
        log(f"sessions[{name}]: touch={len(d['sessions']['touch'])} "
            f"keystroke={len(d['sessions']['keystroke'])} "
            f"dropped_mixed={d['dropped_mixed']} dropped_noscore={d['dropped_noscore']}")
    only_art = set(x["sid"] for x in S["artifact"]["sessions"]["keystroke"]) - \
        set(x["sid"] for x in S["binding"]["sessions"]["keystroke"])
    log("keystroke sessions kept by artifact-construction but not by binding:", sorted(only_art))
    log("their binding composition:",
        {s: sorted(set(json.loads('[]')) | set()) for s in []})

    log("A2 features ...")
    A2 = build_a2(users, fake_pos)

    gen_user_of_row = {a: np.array([UIDX[u] for _, u, _ in gen_rows[a]]) for a in ACTIONS}

    parity = np.array([i % 2 for i in range(NU)])
    mod3 = np.array([i % 3 for i in range(NU)])

    results = []
    for sess_src in ("binding",):
        for regime in ("touch", "keystroke"):
            reg = Regime(S[sess_src]["sessions"][regime], NU)
            log(f"== {sess_src} {regime}: {reg.n} sessions, {reg.M} slots, "
                f"max slots of one action in one session = {reg.gcount.max()}")
            acts = sorted(set(reg.act.tolist()))
            for (m, d) in COMBOS:
                cells = {a: cellS[f"{a}__{m}__{d}"] for a in ACTIONS}
                gen = {a: cells[a]["gen"] for a in ACTIONS}
                fake = np.stack([cells[a]["fake"] for a in ACTIONS])
                tau = np.array([cells[a]["tau"] for a in ACTIONS])
                # per-slot genuine score
                g_sc = np.zeros(reg.M)
                for ai in acts:
                    mk = reg.act == ai
                    g_sc[mk] = gen[ACTIONS[ai]][reg.row[mk]]
                # ---- LLR tables
                def fit_tables(groups, gid):
                    T = []
                    for grp in groups:
                        cu = np.nonzero(gid == grp)[0]
                        t = []
                        for ai in range(5):
                            a = ACTIONS[ai]
                            sel = np.isin(gen_user_of_row[a], cu)
                            t.append(qllr_fit(gen[a][sel], fake[ai][cu].ravel()))
                        T.append(t)
                    return T
                T2 = fit_tables([0, 1], parity)
                T3 = fit_tables([0, 1, 2], mod3)

                def sess_stats(sc):
                    out = {}
                    for tag, TT in (("p", T2), ("m", T3)):
                        for k, t in enumerate(TT):
                            v = np.zeros(reg.M)
                            for ai in acts:
                                mk = reg.act == ai
                                v[mk] = qllr(t[ai], sc[mk])
                            out[f"{tag}{k}"] = reg.agg(v)
                    return out
                GS = sess_stats(g_sc)
                sp = parity[reg.user]; sm = mod3[reg.user]

                def pvec(ST):
                    """p-values under the three calibration schemes."""
                    P = {}
                    # A: null in-sample for the table, eval out-of-sample
                    p = np.full(reg.n, np.nan)
                    for f in (0, 1):
                        null = np.sort(GS[f"p{f}"][sp == f])
                        ev = sp != f
                        p[ev] = tail_p(null, ST[f"p{f}"][ev])
                    P["A_insample_null"] = p
                    # B: stat from the other fold's table, null from the other fold's sessions
                    p = np.full(reg.n, np.nan)
                    for s in (0, 1):
                        ev = sp == s
                        null = np.sort(GS[f"p{s}"][sp == 1 - s])
                        p[ev] = tail_p(null, ST[f"p{1-s}"][ev])
                    P["B_crossfit_null"] = p
                    # 3-way: fit / null / eval fully disjoint
                    p = np.full(reg.n, np.nan)
                    for e in (0, 1, 2):
                        f = (e + 1) % 3; n_ = (e + 2) % 3
                        null = np.sort(GS[f"m{f}"][sm == n_])
                        ev = sm == e
                        p[ev] = tail_p(null, ST[f"m{f}"][ev])
                    P["R_threeway_disjoint"] = p
                    return P
                PG = pvec(GS)

                # ---- selection orderings
                orders = {"A0|uniform": np.tile(np.arange(POOL), (5, NU, 1))}
                for tag in ("A", "B"):
                    for var in ("centroid", "knn"):
                        st = np.stack([A2[f"{tag}|{m}|{a}|{var}"] for a in ACTIONS])
                        orders[f"A2{tag}|{var}"] = np.argsort(st, axis=2, kind="stable")
                sur_same = [(m, dd) for dd in DETECTORS if dd != d]
                ens = 0.0
                for (sm_, sd_) in sur_same:
                    sx = np.stack([cellS[f"{a}__{sm_}__{sd_}"]["fake"] for a in ACTIONS])
                    orders[f"A1|{sm_}|{sd_}"] = np.argsort(sx, axis=2, kind="stable")
                    ens = ens + np.stack([pool_rank(sx[ai]) for ai in range(5)])
                orders["A1ENS5|same"] = np.argsort(ens / len(sur_same), axis=2, kind="stable")
                orders["A3|oracle"] = np.argsort(fake, axis=2, kind="stable")

                specs = [("A0", "uniform", 1)]
                for k in orders:
                    if k == "A0|uniform":
                        continue
                    for r in R_LIST[1:]:
                        specs.append((k.split("|")[0], k, r))

                for rule, okey, r in specs:
                    O = orders[okey] if rule != "A0" else orders["A0|uniform"]
                    P = max(1, POOL // r)
                    keep = np.ascontiguousarray(O[:, :, :P])
                    acc = {k: [] for k in PG}
                    far = 0.0
                    for rep in range(N_REPS):
                        rng = np.random.default_rng(
                            abs(hash((m, d, regime, okey, r, rep))) % (2 ** 31))
                        pos = draw(reg, keep, rng)
                        f_sc = fake[reg.act, reg.uid, pos]
                        far += float((f_sc < tau[reg.act]).mean())
                        PF = pvec(sess_stats(f_sc))
                        for k in acc:
                            acc[k].append(PF[k])
                    rec = dict(sess_src=sess_src, regime=regime, modality=m, detector=d,
                               rule=rule, arm=okey, r=r,
                               per_event_far_slotweighted=far / N_REPS)
                    for k in PG:
                        pf = np.array(acc[k])
                        for al in (0.05, 0.01):
                            rec[f"{k}|caught@{al}"] = float((pf <= al).mean())
                            rec[f"{k}|frr@{al}"] = float((PG[k] <= al).mean())
                    results.append(rec)
                log(f"   {m}|{d} done ({len(specs)} arms)")
    json.dump(results, open(os.path.join(OUT, f"referee_partA.json"), "w"))
    log("wrote referee_partA.json", len(results), "records")


if __name__ == "__main__":
    main()
