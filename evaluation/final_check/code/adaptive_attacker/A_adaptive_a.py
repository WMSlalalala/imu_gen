#!/usr/bin/env python
"""
adaptive_a.py -- Adaptive-attacker curve (Part A), duration confound (Part B),
who-pays-the-false-alarms (Part C).

Stages:
  prereg  : write the pre-registration + fairness compliance block FIRST
  cache   : build score / session / duration / A2-feature caches
  partA   : the adaptive attacker curve for one victim cell family
  partB   : duration-only session detector, controls, timing consequence
  collect : aggregate

Read-only on /mnt/share.  No GPU, no torch, no model loading.
"""
import argparse
import collections
import gzip
import json
import os
import time

import numpy as np

# ----------------------------------------------------------------------------
BASE = ("/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/"
        "e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/A")
CACHE = os.path.join(BASE, "cache")
RES = os.path.join(BASE, "results")
for d in (BASE, CACHE, RES):
    os.makedirs(d, exist_ok=True)

CELL_MAP_PATH = "/mnt/share/mwang49/data7/code/baselines/release_cell_map.json"
SESS_ART = "/mnt/share/mwang49/data7/session_rhythm_detector/results/sessions_{arm}.jsonl"
SHARD_TPL = "/mnt/share/mwang49/data7/results/direct100k/{bundle}/shards/{user}.npz"

ACTIONS = ["tap", "scroll", "swipe", "pinch", "keystroke"]
AIDX = {a: i for i, a in enumerate(ACTIONS)}
MODALITIES = ["imu_only", "trajectory_xytime", "imu_trajectory_xytime"]
DETECTORS = ["authconformer", "behaveformer_stdat", "hmog_style_rf",
             "hmog_style_svm", "paper_svm", "paper_xgboost"]

TEST_USERS = ["hmog_u006", "hmog_u008", "hmog_u011", "hmog_u012", "hmog_u013",
              "hmog_u014", "hmog_u019", "hmog_u022", "hmog_u035", "hmog_u036",
              "hmog_u041", "hmog_u047", "hmog_u049", "hmog_u053", "hmog_u064",
              "hmog_u065", "hmog_u066", "hmog_u077", "hmog_u085", "hmog_u090"]
DEV_USERS = ["hmog_u030", "hmog_u034", "hmog_u044", "hmog_u054", "hmog_u058",
             "hmog_u067", "hmog_u074", "hmog_u082", "hmog_u088", "hmog_u098"]
UIDX = {u: i for i, u in enumerate(TEST_USERS)}
DUIDX = {u: i for i, u in enumerate(DEV_USERS)}
NU = len(TEST_USERS)
POOL = 200

R_LIST = [1, 2, 5, 10, 20]
N_REPS = 20
N_BOOT = 2000
NQUANT = 41
SMOOTH = 0.5
FRR_TARGETS = [0.05, 0.01]
SESSIONS_PER_DAY = 6
ALPHA_GRID = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10,
                       0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
                       0.90, 1.00])
FOLD_OF_USER = {u: (i % 2) for i, u in enumerate(TEST_USERS)}
STAT_NAMES = ["llr_sum", "llr_mean", "b0_count", "b0_rate", "b1_meanscore", "b2_durllr"]

PREREG = {
    "experiment": "adaptive_attacker_and_duration_confound_session_v1",
    "written_before_results": True,
    "preregistered_outcomes": {
        "defence_defeated_by_free_attacker":
            "A2 (no PAD model at all; filters only on a self-consistency statistic inside its "
            "own 200-event pool) drives session caught-rate below the realised genuine session "
            "false-alarm rate at the same operating point, at r <= 10. Then the session-level "
            "defence is defeated by a free, knowledge-free attacker and THAT is the finding.",
        "defence_holds_oracle_curiosity":
            "Only A3 (victim's own score; oracle, not a threat model) collapses it. Then the "
            "defence holds against realistic attackers and the oracle number is a curiosity.",
        "security_margin_is_one_surrogate_model":
            "A1 (surrogate PAD of a different detector family) collapses it but A2 does not. "
            "Then the defence costs the attacker exactly one surrogate model and the price of "
            "that model is the real security margin.",
        "inconclusive": "Anything else. Say so."
    },
    "rules_satisfied": {
        "F1": "Nothing is selected on the evaluated data. Event-level PAD cuts are the frozen "
              "thresholds.json frr5 values (selection_split=development). Session operating "
              "points come from a user-disjoint calibration half of the test users. The "
              "duration LLR is fitted on the 10 DEVELOPMENT users, disjoint from all 20 test "
              "users.",
        "F2": "Two-fold user-disjoint calibration (fold = position parity in the sorted "
              "test-user list), swapped and pooled; the oracle (all-20) variant is reported "
              "alongside with the gap.",
        "F3": "Mirrored arms: each fake session copies a genuine session's user, action "
              "multiset, per-action slot counts and length. Only event content changes.",
        "F5": "A0/A1/A2 consume no victim-detector output. A3 does and is labelled "
              "'oracle, not a threat model' wherever it appears; never the headline.",
        "F9": "User-clustered bootstrap over the 20 test users, 2000 replicates, shared "
              "resampled user multiset across arms within a replicate.",
        "F10": "B0 count rule, B0 count-rate, B1 mean score, B2 duration-only LLR reported at "
               "every operating point, same selection rule, same threshold-search budget.",
        "F11": "Every rate names its denominator: caught and FRR are per SESSION; FAR is "
               "explicitly per EVENT.",
        "F12": "Full price curve plus the session false-alarm rate needed for caught 50/80/95%.",
        "F13": "This block, written before any result file exists.",
        "F15": "coverage.json records every dropped session, every pool shortfall, every "
               "reduced-scope decision."
    },
    "exemptions": [
        {"rule_id": "F4", "reason": "No material is re-generated; the published five-shot fake "
                                    "events are re-used as-is, so the budget is inherited."},
        {"rule_id": "F6", "reason": "No model is loaded (hard no-GPU constraint); only frozen "
                                    "score and threshold files are read, nothing under "
                                    "/mnt/share is written."},
        {"rule_id": "F7", "reason": "Each cell is scored on its own published modality. A1 "
                                    "surrogates may live in another modality; that is the "
                                    "attacker's tool choice, and cross-modality surrogates are "
                                    "reported separately from same-modality ones."},
        {"rule_id": "F8", "reason": "No splicing arm is built; the mirrored fake session applies "
                                    "identical assembly mechanics in both arms."}
    ],
    "threshold_provenance": {
        "event_level_pad_threshold": "thresholds.json frr5, selection_split=development, frozen "
                                     "before test. CAUGHT iff score >= threshold.",
        "session_operating_point": "tail probability of the session statistic against the "
                                   "calibration fold's GENUINE sessions; alarm iff p <= alpha.",
        "score_llr_bins": "41 quantiles of the calibration fold users' pooled genuine+fake event "
                          "scores, per action, per cell.",
        "duration_llr_bins": "41 quantiles of the 10 DEVELOPMENT users' pooled genuine+fake "
                             "durations, per action.",
        "quantile_map": "fitted on DEVELOPMENT users' genuine and fake durations, per action."
    },
    "parity_table": {
        "users": "identical", "session_length": "identical",
        "action_composition": "identical", "detector": "identical",
        "session_threshold": "identical",
        "variable_under_test": "the attacker's selection rule over its own 200-event pool"
    },
    "definitions": {
        "A0": "uniform draw from the published 200 fake events per (user, action).",
        "A1": "attacker scores its OWN pool with a surrogate PAD cell whose DETECTOR FAMILY "
              "differs from the victim's, keeps the lowest-scoring 200/r, draws from that.",
        "A2": "attacker owns NO PAD model. It z-scores a 54-d description of each of its own "
              "generated events (per-IMU-channel mean/std/p5/p50/p95/mean|diff|, duration, "
              "sample count, per-trajectory-channel mean/std) using ONLY its own (user, action) "
              "pool, and keeps the 200/r events closest to that pool's own centroid "
              "(variant 'centroid') or with the smallest mean distance to their own 10 nearest "
              "neighbours inside the pool (variant 'knn'). Inputs: the attacker's own generated "
              "arrays. Nothing else.",
        "A3": "ORACLE, NOT A THREAT MODEL. The attacker filters on the victim cell's own score.",
        "session_statistic_primary": "llr_sum = sum over the session's scored slots of a "
                                     "per-action 41-quantile log-likelihood ratio of the event "
                                     "score. llr_mean is reported alongside.",
        "caught": "fraction of mirrored FAKE sessions that alarm. Denominator = sessions.",
        "session_frr": "fraction of GENUINE sessions that alarm. Denominator = sessions.",
        "per_event_far": "fraction of the attacker's SELECTED fake events with score < the "
                         "cell's frozen frr5 threshold. Denominator = events."
    }
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))


def jdump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_default)


def cell_name(a, m, d):
    return f"{a}__{m}__{d}"


def load_cell_map():
    return json.load(open(CELL_MAP_PATH))["cells"]


# ============================================================================
# STAGE cache
# ============================================================================
def stage_cache():
    cmap = load_cell_map()
    cov = {}

    log("reading 90 cells")
    gen_rows, gen_order, fake_order = {}, {}, {}
    for action in ACTIONS:
        ref = cmap[cell_name(action, "imu_only", "paper_xgboost")]
        g, f = [], collections.defaultdict(list)
        for line in gzip.open(ref["scores"], "rt"):
            d = json.loads(line)
            if d["label"] == 0:
                g.append((d["event_id"], d["user_id"], d["session_id"]))
            else:
                f[d["user_id"]].append(d["event_id"])
        gen_rows[action] = g
        gen_order[action] = {e: i for i, (e, _, _) in enumerate(g)}
        np.save(os.path.join(CACHE, f"genuser_{action}.npy"),
                np.array([UIDX[u] for _, u, _ in g], np.int32))
        fo = {}
        for u in TEST_USERS:
            assert len(f[u]) == POOL, (action, u, len(f[u]))
            for s, e in enumerate(f[u]):
                fo[e] = (UIDX[u], s)
        fake_order[action] = fo

    for modality in MODALITIES:
        for detector in DETECTORS:
            gsc, fsc, thr = {}, {}, {}
            for action in ACTIONS:
                cell = cmap[cell_name(action, modality, detector)]
                gv = np.full(len(gen_rows[action]), np.nan)
                fv = np.full((NU, POOL), np.nan)
                for line in gzip.open(cell["scores"], "rt"):
                    d = json.loads(line)
                    if d["label"] == 0:
                        gv[gen_order[action][d["event_id"]]] = d["fake_high_score"]
                    else:
                        u, s = fake_order[action][d["event_id"]]
                        fv[u, s] = d["fake_high_score"]
                assert not np.isnan(gv).any() and not np.isnan(fv).any()
                t = json.load(open(cell["thresholds"]))
                assert t["score_direction"] == "larger_is_more_fake"
                gsc[action], fsc[action], thr[action] = gv, fv, float(t["frr5"])
            np.savez_compressed(
                os.path.join(CACHE, f"scores__{modality}__{detector}.npz"),
                **{f"gen_{a}": gsc[a] for a in ACTIONS},
                **{f"fake_{a}": fsc[a] for a in ACTIONS},
                thr=np.array([thr[a] for a in ACTIONS]))
    log("  scores cached")

    # ---- sessions -----------------------------------------------------------
    art = {}
    for line in open(SESS_ART.format(arm="genuine")):
        r = json.loads(line)
        art[(int(r["user"]), int(r["session"]))] = r

    sess_slots = collections.defaultdict(list)
    for action in ACTIONS:
        for i, (e, u, s) in enumerate(gen_rows[action]):
            sess_slots[s].append((action, i))

    sessions, dropped_mixed, missing_art = [], [], []
    for sid in sorted(sess_slots):
        slots = sess_slots[sid]
        user = sid.split("_s")[0]
        uu = int(user.split("_u")[1]); ss = int(sid.split("_s")[1])
        rec = art.get((uu, ss))
        if rec is None:
            missing_art.append(sid)
            full_actions = [a for a, _ in slots]
        else:
            full_actions = rec["actions"]
        aset = set(full_actions)
        if aset == {"keystroke"}:
            regime = "keystroke"
        elif "keystroke" not in aset:
            regime = "touch"
        else:
            dropped_mixed.append(sid)
            continue
        sessions.append(dict(
            session_id=sid, user=user, regime=regime,
            slots=sorted(slots, key=lambda t: (AIDX[t[0]], t[1])),
            n_scored=len(slots), n_total=len(full_actions),
            touch4=(aset == {"pinch", "scroll", "swipe", "tap"}),
            in_artifact=rec is not None))
    cov["sessions_reconstructed_from_scored_events"] = len(sess_slots)
    cov["sessions_dropped_mixed_keystroke_touch"] = len(dropped_mixed)
    cov["dropped_mixed_ids"] = dropped_mixed
    cov["sessions_absent_from_session_assembler_artifact"] = len(missing_art)
    cov["absent_ids"] = missing_art
    cov["sessions_kept"] = len(sessions)
    for rg in ("touch", "keystroke"):
        sub = [s for s in sessions if s["regime"] == rg]
        ln = np.array([s["n_scored"] for s in sub])
        cov[rg] = dict(
            sessions=len(sub), users=len({s["user"] for s in sub}),
            scored_slots=int(sum(s["n_scored"] for s in sub)),
            total_slots=int(sum(s["n_total"] for s in sub)),
            scored_slot_fraction=float(sum(s["n_scored"] for s in sub) /
                                       max(1, sum(s["n_total"] for s in sub))),
            scored_len=dict(min=int(ln.min()), p25=float(np.percentile(ln, 25)),
                            median=float(np.median(ln)), p75=float(np.percentile(ln, 75)),
                            max=int(ln.max()), mean=float(ln.mean())))
    cov["touch4_strict_subset_sessions"] = sum(
        1 for s in sessions if s["regime"] == "touch" and s["touch4"])
    jdump(sessions, os.path.join(CACHE, "sessions.json"))

    # ---- dev sessions (for the fully dev-calibrated duration operating point)
    devsess = build_dev_sessions()
    jdump(devsess, os.path.join(CACHE, "dev_sessions.json"))
    cov["dev_sessions"] = {rg: sum(1 for s in devsess if s["regime"] == rg)
                           for rg in ("touch", "keystroke")}

    # ---- durations ----------------------------------------------------------
    log("reading durations from shards")
    bundles = sorted({cmap[c]["bundle"] for c in cmap})
    cov["bundles_in_release"] = bundles
    tabs = {b: _durations_for_bundle(b, TEST_USERS + DEV_USERS) for b in bundles}
    ref = tabs["replay_dataset_full"]
    diffs = collections.Counter()
    for b in bundles:
        if b == "replay_dataset_full":
            continue
        for eid, (d, a, lab, u) in tabs[b].items():
            if abs(d - ref[eid][0]) > 1e-9:
                diffs[(b, a, lab)] += 1
    cov["duration_bundle_disagreements_vs_full"] = {
        f"{b}|{a}|label{l}": n for (b, a, l), n in diffs.items()}
    for a in ACTIONS:
        cov.setdefault("bundles_used_by_action", {})[a] = sorted(
            {cmap[cell_name(a, m, d)]["bundle"] for m in MODALITIES for d in DETECTORS})
    ACTION_BUNDLE = {a: ("replay_dataset_v15" if a == "scroll" else "replay_dataset_full")
                     for a in ACTIONS}
    cov["duration_source_release"] = ACTION_BUNDLE
    cov["duration_source_taskprescribed"] = {a: "replay_dataset_full" for a in ACTIONS}

    for tag, src in (("release", ACTION_BUNDLE),
                     ("full", {a: "replay_dataset_full" for a in ACTIONS})):
        gd = {a: np.zeros(len(gen_rows[a])) for a in ACTIONS}
        fd = {a: np.zeros((NU, POOL)) for a in ACTIONS}
        devg = {a: [] for a in ACTIONS}
        devf = {a: collections.defaultdict(list) for a in ACTIONS}
        for a in ACTIONS:
            tab = tabs[src[a]]
            for i, (e, u, s) in enumerate(gen_rows[a]):
                gd[a][i] = tab[e][0]
            for e, (ui, si) in fake_order[a].items():
                fd[a][ui, si] = tab[e][0]
            for eid, (d, act, lab, u) in tab.items():
                if act == a and u in DEV_USERS:
                    if lab == 1:
                        devf[a][u].append(d)
                    else:
                        devg[a].append(d)
        np.savez_compressed(
            os.path.join(CACHE, f"durations__{tag}.npz"),
            **{f"gen_{a}": gd[a] for a in ACTIONS},
            **{f"fake_{a}": fd[a] for a in ACTIONS},
            **{f"devgen_{a}": np.array(devg[a]) for a in ACTIONS},
            **{f"devfake_{a}": np.array([devf[a][u] for u in DEV_USERS]) for a in ACTIONS})
        log("  durations cached:", tag)

    # ---- A2 self-consistency orderings --------------------------------------
    log("A2 self-consistency features")
    bcfg = {m: {a: cmap[cell_name(a, m, "paper_xgboost")]["bundle"] for a in ACTIONS}
            for m in MODALITIES}
    uniq = {}
    for m in MODALITIES:
        uniq.setdefault(json.dumps(bcfg[m], sort_keys=True), bcfg[m])
    cfg_keys = list(uniq.keys())
    mod2cfg = {m: cfg_keys.index(json.dumps(bcfg[m], sort_keys=True)) for m in MODALITIES}
    cov["a2_bundle_config_per_modality"] = bcfg
    a2 = {}
    fdim = None
    for ci, k in enumerate(cfg_keys):
        cfg = uniq[k]
        feats = {b: _features_for_bundle(b) for b in sorted(set(cfg.values()))}
        for a in ACTIONS:
            F = feats[cfg[a]][a]
            fdim = F.shape[-1]
            for variant in ("centroid", "knn"):
                order = np.zeros((NU, POOL), np.int32)
                for u in range(NU):
                    X = F[u]
                    sd = X.std(0); sd = np.where(sd < 1e-12, 1.0, sd)
                    Z = (X - X.mean(0)) / sd
                    if variant == "centroid":
                        stat = np.linalg.norm(Z, axis=1)
                    else:
                        D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
                        np.fill_diagonal(D2, np.inf)
                        stat = np.sort(D2, axis=1)[:, :10].mean(1)
                    order[u] = np.argsort(stat, kind="stable")
                a2[f"cfg{ci}|{a}|{variant}"] = order
        log("  a2 cfg", ci, "done")
    np.savez_compressed(os.path.join(CACHE, "a2_orders.npz"), **a2)
    jdump(mod2cfg, os.path.join(CACHE, "mod2cfg.json"))
    cov["a2_feature_dim"] = int(fdim)
    jdump(cov, os.path.join(RES, "coverage.json"))
    log("cache done")


def build_dev_sessions():
    """Genuine sessions of the 10 development users, from the shards."""
    slots = collections.defaultdict(list)
    counters = {a: collections.defaultdict(int) for a in ACTIONS}
    idx = {a: {} for a in ACTIONS}
    for u in DEV_USERS:
        z = np.load(SHARD_TPL.format(bundle="replay_dataset_full", user=u), allow_pickle=True)
        act, lab, sid = z["action"], z["label"], z["session_id"]
        for i in range(len(act)):
            if lab[i] != 0:
                continue
            a = str(act[i])
            row = counters[a][u]
            counters[a][u] += 1
            idx[a][(u, row)] = None
            slots[str(sid[i])].append((a, u, row))
    out = []
    for s, sl in sorted(slots.items()):
        aset = {a for a, _, _ in sl}
        if aset == {"keystroke"}:
            rg = "keystroke"
        elif "keystroke" not in aset:
            rg = "touch"
        else:
            continue
        out.append(dict(session_id=s, user=sl[0][1], regime=rg,
                        slots=[[a, u, r] for a, u, r in sl], n_scored=len(sl)))
    return out


def _durations_for_bundle(bundle, users):
    out = {}
    for u in users:
        z = np.load(SHARD_TPL.format(bundle=bundle, user=u), allow_pickle=True)
        tf, off = z["trajectory_flat"], z["offsets"]
        eid, act, lab = z["event_id"], z["action"], z["label"]
        for i in range(len(eid)):
            a, b = int(off[i]), int(off[i + 1])
            assert abs(float(tf[a, 7])) < 1e-9
            out[str(eid[i])] = (float(tf[b - 1, 7]), str(act[i]), int(lab[i]), u)
    return out


def _features_for_bundle(bundle):
    out = {a: np.zeros((NU, POOL, 54)) for a in ACTIONS}
    for u in TEST_USERS:
        ui = UIDX[u]
        z = np.load(SHARD_TPL.format(bundle=bundle, user=u), allow_pickle=True)
        imu, tf, off = z["imu_flat"], z["trajectory_flat"], z["offsets"]
        act, lab = z["action"], z["label"]
        cnt = collections.Counter()
        for i in range(len(act)):
            if lab[i] != 1:
                continue
            a = str(act[i]); s = cnt[a]; cnt[a] += 1
            lo, hi = int(off[i]), int(off[i + 1])
            X = imu[lo:hi].astype(np.float64); T = tf[lo:hi].astype(np.float64)
            f = []
            for c in range(6):
                v = X[:, c]
                dv = np.diff(v) if len(v) > 1 else np.zeros(1)
                f += [v.mean(), v.std(), np.percentile(v, 5), np.percentile(v, 50),
                      np.percentile(v, 95), np.abs(dv).mean()]
            f += [T[-1, 7], float(hi - lo)]
            for c in [0, 1, 2, 3, 4, 5, 6, 8]:
                f += [T[:, c].mean(), T[:, c].std()]
            out[a][ui, s] = f
    return out


# ============================================================================
# session machinery
# ============================================================================
class Regime:
    def __init__(self, sessions, regime):
        sub = sorted([s for s in sessions if s["regime"] == regime],
                     key=lambda s: s["session_id"])
        self.regime, self.sessions = regime, sub
        self.n_sessions = len(sub)
        self.user_of_session = np.array([UIDX[s["user"]] for s in sub], np.int32)
        self.nscored = np.array([s["n_scored"] for s in sub], np.float64)
        self.ntotal = np.array([s["n_total"] for s in sub], np.int32)
        si_, ai_, gr_, ui_ = [], [], [], []
        for i, s in enumerate(sub):
            for a, row in s["slots"]:
                si_.append(i); ai_.append(AIDX[a]); gr_.append(row); ui_.append(UIDX[s["user"]])
        self.sess_idx = np.array(si_, np.int32)
        self.act_idx = np.array(ai_, np.int32)
        self.gen_row = np.array(gr_, np.int32)
        self.uidx = np.array(ui_, np.int32)
        self.n_slots = len(si_)
        self.amask = [self.act_idx == a for a in range(5)]
        self.groups = {}
        for a in range(5):
            m = np.where(self.amask[a])[0]
            if len(m) == 0:
                self.groups[a] = None
                continue
            uniq, ginv = np.unique(self.sess_idx[m], return_inverse=True)
            rank = np.zeros(len(m), np.int32)
            c = collections.Counter()
            for j, g in enumerate(ginv):
                rank[j] = c[g]; c[g] += 1
            self.groups[a] = dict(slots=m, ginv=ginv.astype(np.int32), rank=rank,
                                  ngroups=len(uniq), u=self.uidx[m])


def draw_fake(reg, allowed, rng):
    pos = np.zeros(reg.n_slots, np.int32)
    reuse = 0
    for a in range(5):
        g = reg.groups[a]
        if g is None:
            continue
        P = allowed[a].shape[1]
        order = np.argsort(rng.random((g["ngroups"], P)), axis=1)
        rk = g["rank"]
        over = rk >= P
        reuse += int(over.sum())
        chosen = order[g["ginv"], np.where(over, rk % P, rk)]
        pos[g["slots"]] = allowed[a][g["u"], chosen]
    return pos, reuse


def qllr_fit(gen_vals, fake_vals, nq=NQUANT, smooth=SMOOTH):
    both = np.concatenate([gen_vals, fake_vals])
    edges = np.unique(np.quantile(both, np.linspace(0.0, 1.0, nq)))[1:-1]
    if len(edges) == 0:
        edges = np.array([float(np.median(both))])
    nb = len(edges) + 1
    hg = np.bincount(np.searchsorted(edges, gen_vals, side="right"), minlength=nb).astype(float)
    hf = np.bincount(np.searchsorted(edges, fake_vals, side="right"), minlength=nb).astype(float)
    pg = (hg + smooth) / (hg.sum() + smooth * nb)
    pf = (hf + smooth) / (hf.sum() + smooth * nb)
    return edges, np.log(pf / pg)


def qllr_apply(fit, vals):
    edges, llr = fit
    return llr[np.searchsorted(edges, vals, side="right")]


def ssum(reg, x):
    return np.bincount(reg.sess_idx, weights=x, minlength=reg.n_sessions)


def tail_p(cal_sorted, stats):
    n = len(cal_sorted)
    ge = n - np.searchsorted(cal_sorted, stats, side="left")
    return (ge + 1.0) / (n + 1.0)


def boot_ci(vals, users, boot_idx):
    su = np.bincount(users, weights=np.asarray(vals, float), minlength=NU)
    nu = np.bincount(users, minlength=NU).astype(float)
    den = nu[boot_idx].sum(1)
    v = su[boot_idx].sum(1) / np.where(den == 0, np.nan, den)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def per_user_frr(p_gen, users, alpha):
    a = (p_gen <= alpha).astype(float)
    k = np.bincount(users, weights=a, minlength=NU)
    n = np.bincount(users, minlength=NU).astype(float)
    keep = n > 0
    return k[keep], n[keep]


def per_user_block(p_gen, users, alpha):
    k, n = per_user_frr(p_gen, users, alpha)
    f = k / n
    lam_med = float(np.median(f)) * SESSIONS_PER_DAY
    lam_p90 = float(np.percentile(f, 90)) * SESSIONS_PER_DAY
    return dict(n_users=int(len(f)), min=float(f.min()), median=float(np.median(f)),
                p90=float(np.percentile(f, 90)), max=float(f.max()),
                pooled=float(k.sum() / n.sum()), per_user=[float(x) for x in f],
                sessions_per_user=[int(x) for x in n],
                spread=fit_beta_binomial(k, n),
                lockout_median_user_days_between=(1.0 / lam_med) if lam_med > 0 else None,
                lockout_p90_user_days_between=(1.0 / lam_p90) if lam_p90 > 0 else None,
                lockouts_per_day_median_user=lam_med,
                lockouts_per_day_p90_user=lam_p90)


def fit_beta_binomial(k, n):
    p = k / n
    m = float(np.average(p, weights=n))
    if m <= 0:
        return dict(method="moments", mean=m, note="all-zero: no spread estimable")
    if m >= 1:
        return dict(method="moments", mean=m, note="all-one")
    v = float(np.average((p - m) ** 2, weights=n))
    nb = float(np.mean(n))
    exp_bin = m * (1 - m) / nb
    if v <= exp_bin:
        return dict(method="moments", mean=m, var=v, binomial_var=exp_bin,
                    note="no overdispersion detected")
    rho = (v - exp_bin) / (m * (1 - m) * (1 - 1.0 / nb))
    rho = float(min(max(rho, 1e-6), 0.999))
    s = (1 - rho) / rho
    al, be = m * s, (1 - m) * s
    out = dict(method="moments", mean=m, alpha=al, beta=be, icc=rho)
    try:
        from scipy.stats import beta as _b
        for q in (0.10, 0.50, 0.90, 0.99):
            out[f"q{int(q*100)}"] = float(_b.ppf(q, al, be))
    except Exception:
        pass
    return out


def price_readings(p_gen, p_fake):
    frr = np.array([(p_gen <= a).mean() for a in ALPHA_GRID])
    cau = np.array([(p_fake <= a).mean() for a in ALPHA_GRID])
    out = {}
    for t in (0.50, 0.80, 0.95):
        ok = np.where(cau >= t)[0]
        out[f"frr_for_caught_{int(t*100)}"] = float(frr[ok[0]]) if len(ok) else None
    return out, frr, cau


def alpha_for_p90_user_frr(p_gen, users, cap=0.01):
    """Largest ATTAINABLE cut whose per-user session FRR has p90 <= cap.
    Candidate cuts are the observed p-values themselves -- those are the only
    operating points the calibration set can actually realise."""
    best = None
    cands = np.unique(p_gen[np.isfinite(p_gen)])
    for a in cands:
        k, n = per_user_frr(p_gen, users, a)
        if np.percentile(k / n, 90) <= cap:
            best = float(a)
        else:
            break
    return best


def build_allowed(order, r):
    return np.ascontiguousarray(order[:, :max(1, POOL // r)])


# ============================================================================
# STAGE partA
# ============================================================================
_GEN_UIDX = {}


def gen_user_index(a):
    if a not in _GEN_UIDX:
        _GEN_UIDX[a] = np.load(os.path.join(CACHE, f"genuser_{a}.npy"))
    return _GEN_UIDX[a]


def gen_slot(reg, table):
    out = np.zeros(reg.n_slots)
    for ai, a in enumerate(ACTIONS):
        m = reg.amask[ai]
        if m.any():
            out[m] = table[a][reg.gen_row[m]]
    return out


def stage_partA(vm, vd, surrogate_scope):
    sessions = json.load(open(os.path.join(CACHE, "sessions.json")))
    mod2cfg = json.load(open(os.path.join(CACHE, "mod2cfg.json")))
    a2 = np.load(os.path.join(CACHE, "a2_orders.npz"))
    dz = np.load(os.path.join(CACHE, "durations__release.npz"))

    vkey = f"{vm}|{vd}"
    vz = np.load(os.path.join(CACHE, f"scores__{vm}__{vd}.npz"))
    gen = {a: vz[f"gen_{a}"] for a in ACTIONS}
    fake = {a: vz[f"fake_{a}"] for a in ACTIONS}
    thr = {a: float(vz["thr"][i]) for i, a in enumerate(ACTIONS)}

    dur_fit = {a: qllr_fit(dz[f"devgen_{a}"], dz[f"devfake_{a}"].ravel()) for a in ACTIONS}
    gdur = {a: dz[f"gen_{a}"] for a in ACTIONS}
    fdur = {a: dz[f"fake_{a}"] for a in ACTIONS}

    # ---- selection orderings ------------------------------------------------
    orders = {"A0|uniform": np.tile(np.arange(POOL, dtype=np.int32), (5, NU, 1))}
    surr = [(m, d) for m in MODALITIES for d in DETECTORS
            if d != vd and (surrogate_scope != "same_modality" or m == vm)]
    for (m, d) in surr:
        sz = np.load(os.path.join(CACHE, f"scores__{m}__{d}.npz"))
        orders[f"A1|{m}|{d}"] = np.stack(
            [np.argsort(sz[f"fake_{a}"], axis=1, kind="stable").astype(np.int32)
             for a in ACTIONS])
    ci = mod2cfg[vm]
    for variant in ("centroid", "knn"):
        orders[f"A2|{variant}"] = np.stack([a2[f"cfg{ci}|{a}|{variant}"] for a in ACTIONS])
    orders["A3|oracle"] = np.stack(
        [np.argsort(fake[a], axis=1, kind="stable").astype(np.int32) for a in ACTIONS])

    arm_specs = [("A0", "uniform", 1, "A0|uniform")]
    for (m, d) in surr:
        for r in R_LIST[1:]:
            arm_specs.append(("A1", f"{m}|{d}", r, f"A1|{m}|{d}"))
    for variant in ("centroid", "knn"):
        for r in R_LIST[1:]:
            arm_specs.append(("A2", variant, r, f"A2|{variant}"))
    for r in R_LIST[1:]:
        arm_specs.append(("A3_ORACLE_NOT_A_THREAT_MODEL", "oracle", r, "A3|oracle"))

    results = dict(victim=vkey, surrogate_scope=surrogate_scope,
                   n_surrogates=len(surr), arms=[], reference={})
    boot_idx = np.random.default_rng(777).integers(0, NU, size=(N_BOOT, NU))

    for regime in ("touch", "keystroke"):
        reg = Regime(sessions, regime)
        if reg.n_sessions == 0:
            continue
        fold_of_sess = np.array([FOLD_OF_USER[TEST_USERS[u]] for u in reg.user_of_session])
        scopes = {}
        for f in (0, 1):
            cal_ui = np.array([UIDX[u] for u in TEST_USERS if FOLD_OF_USER[u] == f])
            fit = {}
            for a in ACTIONS:
                gu = gen_user_index(a)
                gv = gen[a][np.isin(gu, cal_ui)]
                fit[a] = qllr_fit(gv, fake[a][cal_ui].ravel()) if len(gv) >= 20 else None
            scopes.setdefault("split", []).append(
                dict(fit=fit, cal=np.where(fold_of_sess == f)[0],
                     ev=np.where(fold_of_sess != f)[0]))
        scopes["oracle"] = [dict(
            fit={a: qllr_fit(gen[a], fake[a].ravel()) for a in ACTIONS},
            cal=np.arange(reg.n_sessions), ev=np.arange(reg.n_sessions))]

        # genuine per-slot quantities (arm-independent)
        g_score = gen_slot(reg, gen)
        g_dur = gen_slot(reg, gdur)
        g_caught = np.zeros(reg.n_slots)
        g_durllr = np.zeros(reg.n_slots)
        for ai, a in enumerate(ACTIONS):
            m = reg.amask[ai]
            if not m.any():
                continue
            g_caught[m] = (g_score[m] >= thr[a]).astype(float)
            g_durllr[m] = qllr_apply(dur_fit[a], g_dur[m])
        gc = ssum(reg, g_caught)
        base_g = {
            "b0_count": gc, "b0_rate": gc / reg.nscored,
            "b1_meanscore": ssum(reg, g_score) / reg.nscored,
            "b2_durllr": ssum(reg, g_durllr)}

        # genuine session stats + p-values per scope/fold
        GST, PG, CALSORT = {}, {}, {}
        for scope, fl in scopes.items():
            PG[scope] = {sn: np.full(reg.n_sessions, np.nan) for sn in STAT_NAMES}
            CALSORT[scope] = []
            GST[scope] = []
            for fd in fl:
                gl = np.zeros(reg.n_slots)
                for ai, a in enumerate(ACTIONS):
                    m = reg.amask[ai]
                    if m.any() and fd["fit"][a] is not None:
                        gl[m] = qllr_apply(fd["fit"][a], g_score[m])
                s = dict(base_g)
                s["llr_sum"] = ssum(reg, gl)
                s["llr_mean"] = s["llr_sum"] / reg.nscored
                GST[scope].append(s)
                cs = {sn: np.sort(s[sn][fd["cal"]]) for sn in STAT_NAMES}
                CALSORT[scope].append(cs)
                for sn in STAT_NAMES:
                    PG[scope][sn][fd["ev"]] = tail_p(cs[sn], s[sn][fd["ev"]])

        # arm-independent Part C operating point + attainability of alpha
        alt_alpha, min_alpha = {}, {}
        for scope in scopes:
            n_cal = min(len(fd["cal"]) for fd in scopes[scope])
            for sn in STAT_NAMES:
                alt_alpha[f"{scope}|{sn}"] = alpha_for_p90_user_frr(
                    PG[scope][sn], reg.user_of_session, cap=0.01)
                min_alpha[f"{scope}|{sn}"] = 1.0 / (n_cal + 1.0)

        results["reference"][regime] = dict(
            n_sessions=int(reg.n_sessions), n_slots=int(reg.n_slots),
            users=int(len(set(reg.user_of_session.tolist()))),
            scored_len_mean=float(reg.nscored.mean()),
            scored_slot_fraction=float(reg.nscored.sum() / reg.ntotal.sum()),
            min_attainable_alpha=min_alpha,
            alpha001_attainable={k: bool(v <= 0.01) for k, v in min_alpha.items()},
            alt_operating_point_alpha_p90user_frr_le_1pct=alt_alpha,
            genuine_session_frr={
                f"{scope}|{sn}|alpha{al}": float((PG[scope][sn] <= al).mean())
                for scope in scopes for sn in STAT_NAMES for al in FRR_TARGETS},
            mean_per_event_llr_genuine=float(np.mean([
                GST["split"][k]["llr_sum"][fd["ev"]].sum()
                / reg.nscored[fd["ev"]].sum()
                for k, fd in enumerate(scopes["split"])])),
            per_user_frr={
                **{f"{scope}|{sn}|alpha{al}": per_user_block(PG[scope][sn],
                                                             reg.user_of_session, al)
                   for scope in scopes for sn in STAT_NAMES for al in FRR_TARGETS},
                **{f"{scope}|{sn}|alt_op": per_user_block(
                    PG[scope][sn], reg.user_of_session,
                    alt_alpha[f"{scope}|{sn}"] if alt_alpha[f"{scope}|{sn}"] else 0.0)
                   for scope in scopes for sn in STAT_NAMES}})

        for rule, tag, r, okey in arm_specs:
            O = orders[okey]
            allowed = {a: build_allowed(O[a], r) for a in range(5)}
            P = allowed[0].shape[1]
            far_ev = {}
            for ai, a in enumerate(ACTIONS):
                sel = np.take_along_axis(fake[a], allowed[ai], axis=1)
                far_ev[a] = float((sel < thr[a]).mean())
            acts_here = [ACTIONS[ai] for ai in range(5) if reg.groups[ai] is not None]
            far_reg = float(np.mean([far_ev[a] for a in acts_here]))

            short = {}
            for ai, a in enumerate(ACTIONS):
                if reg.groups[ai] is None:
                    continue
                need = int(reg.groups[ai]["rank"].max()) + 1
                if need > P:
                    short[a] = dict(pool_after_filter=int(P), max_slots_needed=need,
                                    slot_draws_forced_to_reuse=int(
                                        (reg.groups[ai]["rank"] >= P).sum()),
                                    sessions_affected=int(len(np.unique(
                                        reg.sess_idx[reg.groups[ai]["slots"]][
                                            reg.groups[ai]["rank"] >= P]))))

            PFacc = {sc: {sn: [] for sn in STAT_NAMES} for sc in scopes}
            reuse_tot = 0
            mean_llr_fake = []
            for rep in range(N_REPS):
                rng = np.random.default_rng(
                    abs(hash((okey, r, regime, rep))) % (2 ** 31))
                pos, reuse = draw_fake(reg, allowed, rng)
                reuse_tot += reuse
                f_score = np.zeros(reg.n_slots); f_dur = np.zeros(reg.n_slots)
                f_caught = np.zeros(reg.n_slots); f_durllr = np.zeros(reg.n_slots)
                for ai, a in enumerate(ACTIONS):
                    m = reg.amask[ai]
                    if not m.any():
                        continue
                    f_score[m] = fake[a][reg.uidx[m], pos[m]]
                    f_dur[m] = fdur[a][reg.uidx[m], pos[m]]
                    f_caught[m] = (f_score[m] >= thr[a]).astype(float)
                    f_durllr[m] = qllr_apply(dur_fit[a], f_dur[m])
                fc = ssum(reg, f_caught)
                base_f = {"b0_count": fc, "b0_rate": fc / reg.nscored,
                          "b1_meanscore": ssum(reg, f_score) / reg.nscored,
                          "b2_durllr": ssum(reg, f_durllr)}
                for scope, fl in scopes.items():
                    pf = {sn: np.full(reg.n_sessions, np.nan) for sn in STAT_NAMES}
                    for k, fd in enumerate(fl):
                        flr = np.zeros(reg.n_slots)
                        for ai, a in enumerate(ACTIONS):
                            m = reg.amask[ai]
                            if m.any() and fd["fit"][a] is not None:
                                flr[m] = qllr_apply(fd["fit"][a], f_score[m])
                        s = dict(base_f)
                        s["llr_sum"] = ssum(reg, flr)
                        s["llr_mean"] = s["llr_sum"] / reg.nscored
                        if scope == "split":
                            ev_slots = np.isin(reg.sess_idx, fd["ev"])
                            mean_llr_fake.append(float(flr[ev_slots].mean()))
                        for sn in STAT_NAMES:
                            pf[sn][fd["ev"]] = tail_p(CALSORT[scope][k][sn], s[sn][fd["ev"]])
                    for sn in STAT_NAMES:
                        PFacc[scope][sn].append(pf[sn])

            arm = dict(rule=rule, tag=tag, r=r, regime=regime, n_reps=N_REPS,
                       pool_after_filter=int(P), generated_per_submitted=int(r),
                       per_event_far_at_frr5_by_action=far_ev,
                       per_event_far_at_frr5_regime_mean=far_reg,
                       pool_shortfall=short, reuse_slot_draws_total=int(reuse_tot),
                       n_sessions=int(reg.n_sessions),
                       mean_per_event_llr_fake=float(np.mean(mean_llr_fake)),
                       stats={})
            for scope in scopes:
                for sn in STAT_NAMES:
                    PF = np.array(PFacc[scope][sn])
                    pg = PG[scope][sn]
                    d = {}
                    for lab, al in ([(f"alpha{x}", x) for x in FRR_TARGETS] +
                                    [("alpha_min_attainable",
                                      min_alpha[f"{scope}|{sn}"])]):
                        crep = (PF <= al).mean(1)
                        cs = (PF <= al).mean(0)
                        d[lab] = dict(
                            alpha=float(al),
                            caught=float(crep.mean()),
                            caught_sd_over_reps=float(crep.std()),
                            caught_min_rep=float(crep.min()),
                            caught_max_rep=float(crep.max()),
                            caught_ci95_userboot=boot_ci(cs, reg.user_of_session, boot_idx),
                            session_frr=float((pg <= al).mean()),
                            session_frr_ci95_userboot=boot_ci(
                                (pg <= al).astype(float), reg.user_of_session, boot_idx),
                            caught_minus_frr=float(crep.mean() - (pg <= al).mean()))
                        if sn == "llr_sum":
                            nu_ = np.bincount(reg.user_of_session, minlength=NU)
                            d[lab]["per_user_caught_sum"] = np.bincount(
                                reg.user_of_session, weights=cs, minlength=NU).tolist()
                            d[lab]["per_user_frr_sum"] = np.bincount(
                                reg.user_of_session, weights=(pg <= al).astype(float),
                                minlength=NU).tolist()
                            d[lab]["per_user_sessions"] = nu_.tolist()
                    pr, frr_c, cau_c = price_readings(pg, PF.reshape(-1))
                    d["price"] = pr
                    aa = alt_alpha[f"{scope}|{sn}"]
                    if aa is not None:
                        d["alt_op_p90user_frr_le_1pct"] = dict(
                            alpha=aa, caught=float((PF <= aa).mean()),
                            session_frr=float((pg <= aa).mean()))
                    else:
                        d["alt_op_p90user_frr_le_1pct"] = None
                    if sn == "llr_sum":
                        d["curve"] = dict(alpha=ALPHA_GRID.tolist(),
                                          frr=frr_c.tolist(), caught=cau_c.tolist())
                    arm["stats"][f"{scope}|{sn}"] = d
            results["arms"].append(arm)
        log(f"{vkey} {regime} done ({len(arm_specs)} arms)")

    jdump(results, os.path.join(RES, f"partA__{vm}__{vd}.json"))
    log("wrote", f"partA__{vm}__{vd}.json")


# ============================================================================
# STAGE partB
# ============================================================================
def stage_partB():
    sessions = json.load(open(os.path.join(CACHE, "sessions.json")))
    devsess = json.load(open(os.path.join(CACHE, "dev_sessions.json")))
    out = {}
    boot_idx = np.random.default_rng(777).integers(0, NU, size=(N_BOOT, NU))

    for tag in ("release", "full"):
        dz = np.load(os.path.join(CACHE, f"durations__{tag}.npz"))
        gd = {a: dz[f"gen_{a}"] for a in ACTIONS}
        fd = {a: dz[f"fake_{a}"] for a in ACTIONS}
        dgen = {a: dz[f"devgen_{a}"] for a in ACTIONS}
        dfak = {a: dz[f"devfake_{a}"] for a in ACTIONS}   # (10, POOL)

        # ---- bulk statistics -------------------------------------------------
        bulk = {}
        for a in ACTIONS:
            G, F = gd[a], fd[a].ravel()
            cap = float(F.max())
            bulk[a] = dict(
                genuine=dict(n=int(G.size), p5=float(np.percentile(G, 5)),
                             p50=float(np.median(G)), median=float(np.median(G)),
                             p95=float(np.percentile(G, 95)), max=float(G.max()),
                             mean=float(G.mean())),
                fake=dict(n=int(F.size), p5=float(np.percentile(F, 5)),
                          p50=float(np.median(F)), median=float(np.median(F)),
                          p95=float(np.percentile(F, 95)), max=cap, mean=float(F.mean())),
                fake_fraction_exactly_at_cap=float(np.mean(np.isclose(F, cap))),
                fake_fraction_above_genuine_max=float(np.mean(F > G.max())),
                genuine_fraction_above_fake_cap=float(np.mean(G > cap)),
                median_shift_genuine_minus_fake=float(np.median(G) - np.median(F)))
        out[f"bulk__{tag}"] = bulk

        # ---- three duration variants ----------------------------------------
        # Each variant is a pair of per-action transforms applied identically
        # everywhere the arm appears (dev fit data, test events, dev sessions).
        rng = np.random.default_rng(4242)
        ident = {a: (lambda x, a=a: np.asarray(x, float)) for a in ACTIONS}
        capd = {a: float(dfak[a].max()) for a in ACTIONS}      # cap from DEV users only

        def mk_clipjit(a):
            def t(x):
                x = np.asarray(x, float)
                return np.minimum(x, capd[a]) + rng.uniform(-0.005, 0.005, size=x.shape)
            return t

        qmap = {a: (np.sort(dfak[a].ravel()), np.sort(dgen[a])) for a in ACTIONS}

        def mk_qmap(a, rank_preserving):
            src, tgt = qmap[a]
            def t(x):
                x = np.asarray(x, float)
                u = np.clip((np.searchsorted(src, x, side="right") - 0.5) / len(src),
                            0.0, 1.0)
                if rank_preserving:
                    # map onto OBSERVED genuine values -> stays on the 10 ms grid
                    i = np.clip(np.rint(u * (len(tgt) - 1)).astype(int), 0, len(tgt) - 1)
                    return tgt[i]
                return np.quantile(tgt, u)     # interpolating -> leaves the grid
            return t

        # dequantised quantile map: durations sit on a 10 ms grid, so the plain
        # ECDF is massively tied and a rank map cannot reproduce the target.
        # Break the ties first (U(-5,+5) ms on BOTH arms), then map.
        jrng = np.random.default_rng(909)
        dq_devg = {a: dgen[a] + jrng.uniform(-0.005, 0.005, dgen[a].shape) for a in ACTIONS}
        dq_devf = {a: dfak[a].ravel() + jrng.uniform(-0.005, 0.005, dfak[a].ravel().shape)
                   for a in ACTIONS}
        dqmap = {a: (np.sort(dq_devf[a]), np.sort(dq_devg[a])) for a in ACTIONS}

        def mk_dq_g(a):
            def t(x):
                x = np.asarray(x, float)
                return x + jrng.uniform(-0.005, 0.005, x.shape)
            return t

        def mk_dq_f(a):
            src, tgt = dqmap[a]
            def t(x):
                x = np.asarray(x, float) + jrng.uniform(-0.005, 0.005, np.shape(x))
                u = np.clip((np.searchsorted(src, x, side="right") - 0.5) / len(src),
                            0.0, 1.0)
                return np.quantile(tgt, u)
            return t

        variants = {
            "raw": dict(gt=ident, ft=ident),
            "fix_dequantised_quantile_map": dict(
                gt={a: mk_dq_g(a) for a in ACTIONS},
                ft={a: mk_dq_f(a) for a in ACTIONS}),
            "control_clip_cap_plus_jitter": dict(
                gt={a: mk_clipjit(a) for a in ACTIONS},
                ft={a: mk_clipjit(a) for a in ACTIONS}),
            "fix_quantile_map_interpolating": dict(
                gt=ident, ft={a: mk_qmap(a, False) for a in ACTIONS}),
            "fix_quantile_map_rank_preserving": dict(
                gt=ident, ft={a: mk_qmap(a, True) for a in ACTIONS}),
        }
        # grid diagnostic: durations live on a 10 ms grid; a mapping that leaves
        # that grid manufactures a brand-new tell.
        grid = {}
        for vn, TRx in variants.items():
            for a in ACTIONS:
                fx = TRx["ft"][a](fd[a]).ravel()
                gx = TRx["gt"][a](gd[a]).ravel()
                grid[f"{vn}|{a}"] = dict(
                    fake_frac_on_10ms_grid=float(np.mean(
                        np.abs(fx * 100 - np.rint(fx * 100)) < 1e-6)),
                    genuine_frac_on_10ms_grid=float(np.mean(
                        np.abs(gx * 100 - np.rint(gx * 100)) < 1e-6)),
                    fake_median=float(np.median(fx)),
                    genuine_median=float(np.median(gx)))
        out[f"grid_diagnostic__{tag}"] = grid

        for vname, TR in variants.items():
            gt, ft = TR["gt"], TR["ft"]
            V = dict(g={a: gt[a](gd[a]) for a in ACTIONS},
                     f={a: ft[a](fd[a]) for a in ACTIONS},
                     devg={a: gt[a](dgen[a]) for a in ACTIONS},
                     devf={a: ft[a](dfak[a].ravel()) for a in ACTIONS})
            fit = {a: qllr_fit(V["devg"][a], V["devf"][a]) for a in ACTIONS}
            res = {}
            for regime in ("touch", "keystroke"):
                reg = Regime(sessions, regime)
                if reg.n_sessions == 0:
                    continue
                g_dur = gen_slot(reg, V["g"])
                g_llr = np.zeros(reg.n_slots)
                for ai, a in enumerate(ACTIONS):
                    m = reg.amask[ai]
                    if m.any():
                        g_llr[m] = qllr_apply(fit[a], g_dur[m])
                GS = ssum(reg, g_llr)

                fold_of_sess = np.array([FOLD_OF_USER[TEST_USERS[u]]
                                         for u in reg.user_of_session])
                scopes = {"split": [dict(cal=np.where(fold_of_sess == f)[0],
                                         ev=np.where(fold_of_sess != f)[0]) for f in (0, 1)],
                          "oracle": [dict(cal=np.arange(reg.n_sessions),
                                          ev=np.arange(reg.n_sessions))]}
                # fully dev-calibrated operating point (F1-clean: fit AND cut from dev)
                dev_cal = dev_session_duration_stats(devsess, regime, fit, gt, tag)

                allowed = {a: np.tile(np.arange(POOL, dtype=np.int32), (NU, 1))
                           for a in range(5)}
                PFs = {sc: [] for sc in list(scopes) + ["dev"]}
                for rep in range(N_REPS):
                    r2 = np.random.default_rng(abs(hash((vname, regime, rep, tag))) % 2 ** 31)
                    pos, _ = draw_fake(reg, allowed, r2)
                    f_llr = np.zeros(reg.n_slots)
                    for ai, a in enumerate(ACTIONS):
                        m = reg.amask[ai]
                        if not m.any():
                            continue
                        f_llr[m] = qllr_apply(fit[a], V["f"][a][reg.uidx[m], pos[m]])
                    FS = ssum(reg, f_llr)
                    for sc, fl in scopes.items():
                        pf = np.full(reg.n_sessions, np.nan)
                        for fdd in fl:
                            cs = np.sort(GS[fdd["cal"]])
                            pf[fdd["ev"]] = tail_p(cs, FS[fdd["ev"]])
                        PFs[sc].append(pf)
                    PFs["dev"].append(tail_p(dev_cal, FS))
                PGs = {}
                for sc, fl in scopes.items():
                    pg = np.full(reg.n_sessions, np.nan)
                    for fdd in fl:
                        cs = np.sort(GS[fdd["cal"]])
                        pg[fdd["ev"]] = tail_p(cs, GS[fdd["ev"]])
                    PGs[sc] = pg
                PGs["dev"] = tail_p(dev_cal, GS)

                rr = {}
                for sc in PFs:
                    PF = np.array(PFs[sc]); pg = PGs[sc]
                    e = {}
                    for al in FRR_TARGETS:
                        crep = (PF <= al).mean(1)
                        e[f"alpha{al}"] = dict(
                            caught=float(crep.mean()),
                            caught_sd_over_reps=float(crep.std()),
                            caught_ci95_userboot=boot_ci((PF <= al).mean(0),
                                                         reg.user_of_session, boot_idx),
                            session_frr=float((pg <= al).mean()),
                            caught_minus_frr=float(crep.mean() - (pg <= al).mean()))
                    pr, frr_c, cau_c = price_readings(pg, PF.reshape(-1))
                    e["price"] = pr
                    e["curve"] = dict(alpha=ALPHA_GRID.tolist(), frr=frr_c.tolist(),
                                      caught=cau_c.tolist())
                    e["per_user_frr"] = {f"alpha{al}": per_user_block(
                        pg, reg.user_of_session, al) for al in FRR_TARGETS}
                    rr[sc] = e
                res[regime] = rr
            out[f"durationLLR__{tag}__{vname}"] = res
            log("partB", tag, vname, "done")

    out["timing_consequence"] = timing_consequence()
    jdump(out, os.path.join(RES, "partB.json"))
    log("partB written")


_DEVDUR = {}


def dev_genuine_durations(tag):
    """(user, action, row) -> genuine duration for the 10 development users."""
    if tag in _DEVDUR:
        return _DEVDUR[tag]
    dur = {}
    for u in DEV_USERS:
        zf = np.load(SHARD_TPL.format(bundle="replay_dataset_full", user=u), allow_pickle=True)
        z15 = np.load(SHARD_TPL.format(bundle="replay_dataset_v15", user=u), allow_pickle=True)
        tf, off, act, lab = zf["trajectory_flat"], zf["offsets"], zf["action"], zf["label"]
        tf15, off15 = z15["trajectory_flat"], z15["offsets"]
        cnt = collections.Counter()
        for i in range(len(act)):
            if lab[i] != 0:
                continue
            a = str(act[i]); row = cnt[a]; cnt[a] += 1
            if tag == "release" and a == "scroll":
                dur[(u, a, row)] = float(tf15[int(off15[i + 1]) - 1, 7])
            else:
                dur[(u, a, row)] = float(tf[int(off[i + 1]) - 1, 7])
    _DEVDUR[tag] = dur
    return dur


def dev_session_duration_stats(devsess, regime, fit, gt, tag):
    """genuine session duration-LLR statistics for the 10 development users."""
    dur = dev_genuine_durations(tag)
    stats = []
    for s in devsess:
        if s["regime"] != regime:
            continue
        tot = 0.0
        for a, u, row in s["slots"]:
            d = gt[a](np.array([dur[(u, a, row)]]))
            tot += float(qllr_apply(fit[a], d)[0])
        stats.append(tot)
    return np.sort(np.array(stats)) if stats else np.array([0.0])


def timing_consequence():
    """How much of a timing-feature session result is duration?

    onset[i] = onset[i-1] + duration[i-1] + gap[i], so the detector that reads
    only the ONSET sequence sees inter-onset intervals IOI[i] = duration[i-1] + gap[i].
    Two fake gap policies:
      gaps_copy : the attacker reproduces the victim session's own gap sequence
                  (perfect pacing).  Then the arms differ ONLY through duration --
                  this arm is NEAR-TAUTOLOGICAL and is labelled as such.
      gaps_emp  : the attacker samples gaps from the published paced_emp arm
                  (gaps drawn from the human interval distribution).
    Ablation: replace every duration with the per-action GENUINE median in BOTH
    arms and re-run.  The caught-rate lost is the duration share.
    """
    art = {}
    for line in open(SESS_ART.format(arm="genuine")):
        r = json.loads(line)
        art[(int(r["user"]), int(r["session"]))] = r
    pemp = {}
    for line in open(SESS_ART.format(arm="paced_emp")):
        r = json.loads(line)
        pemp[(int(r["user"]), int(r["session"]))] = r

    dz = np.load(os.path.join(CACHE, "durations__release.npz"))
    fpool = {a: dz[f"fake_{a}"] for a in ACTIONS}                 # (20, 200) test users
    fpool_dev = {a: dz[f"devfake_{a}"] for a in ACTIONS}          # (10, 200) dev users
    gmed = {a: float(np.median(dz[f"devgen_{a}"])) for a in ACTIONS}

    def collect(users, uidx_map):
        rows = []
        for (uu, ss), rec in sorted(art.items()):
            uname = f"hmog_u{uu:03d}"
            if uname not in uidx_map:
                continue
            acts = rec["actions"]
            aset = set(acts)
            if aset == {"keystroke"}:
                rg = "keystroke"
            elif "keystroke" not in aset:
                rg = "touch"
            else:
                continue
            rows.append(dict(u=uidx_map[uname], regime=rg,
                             a=np.array([AIDX[x] for x in acts], np.int32),
                             d=np.round(np.array(rec["durations_s"]), 2),
                             g=np.array(rec["gaps_s"]),
                             ge=np.array(pemp[(uu, ss)]["gaps_s"])))
        return rows

    dev_rows = collect(DEV_USERS, DUIDX)
    test_rows = collect(TEST_USERS, UIDX)
    GMED_VEC = np.array([gmed[a] for a in ACTIONS])

    def sess_ioi(rows, which, ablate, rng, pool):
        """which in {'genuine','gaps_copy','gaps_emp'}"""
        out = []
        for r in rows:
            n = len(r["a"])
            if which == "genuine":
                d = r["d"].copy(); g = r["g"]
            else:
                idx = rng.integers(0, POOL, size=n)
                d = np.empty(n)
                for ai in range(5):
                    m = r["a"] == ai
                    if m.any():
                        d[m] = pool[ACTIONS[ai]][r["u"], idx[m]]
                g = r["g"] if which == "gaps_copy" else r["ge"]
            if ablate:
                d = GMED_VEC[r["a"]]
            out.append((r["u"], r["regime"],
                        (d[:-1] + g[1:]) if n > 1 else np.array([])))
        return out

    res = {}
    boot_idx = np.random.default_rng(777).integers(0, NU, size=(N_BOOT, NU))
    for ablate in (False, True):
        for which in ("gaps_copy", "gaps_emp"):
            rng = np.random.default_rng(31337)
            devg = sess_ioi(dev_rows, "genuine", ablate, rng, fpool_dev)
            devf = sess_ioi(dev_rows, which, ablate, rng, fpool_dev)
            gvals = np.concatenate([x[2] for x in devg if len(x[2])])
            fvals = np.concatenate([x[2] for x in devf if len(x[2])])
            fit = qllr_fit(np.log(np.clip(gvals, 1e-3, None)),
                           np.log(np.clip(fvals, 1e-3, None)))
            for regime in ("touch", "keystroke"):
                cal = np.array(sorted(
                    float(qllr_apply(fit, np.log(np.clip(x[2], 1e-3, None))).sum())
                    for x in devg if x[1] == regime and len(x[2])))
                tg = sess_ioi(test_rows, "genuine", ablate, rng, fpool)
                gs = np.array([qllr_apply(fit, np.log(np.clip(x[2], 1e-3, None))).sum()
                               for x in tg if x[1] == regime and len(x[2])])
                gu = np.array([x[0] for x in tg if x[1] == regime and len(x[2])], np.int32)
                pg = tail_p(cal, gs)
                caught = {al: [] for al in FRR_TARGETS}
                for rep in range(N_REPS):
                    r2 = np.random.default_rng(abs(hash((which, ablate, regime, rep))) % 2 ** 31)
                    tf_ = sess_ioi(test_rows, which, ablate, r2, fpool)
                    fs = np.array([qllr_apply(fit, np.log(np.clip(x[2], 1e-3, None))).sum()
                                   for x in tf_ if x[1] == regime and len(x[2])])
                    pf = tail_p(cal, fs)
                    for al in FRR_TARGETS:
                        caught[al].append(float((pf <= al).mean()))
                res[f"{which}|{'duration_ablated' if ablate else 'raw'}|{regime}"] = dict(
                    n_sessions=int(len(gs)),
                    caught={f"alpha{al}": float(np.mean(caught[al])) for al in FRR_TARGETS},
                    caught_sd={f"alpha{al}": float(np.std(caught[al])) for al in FRR_TARGETS},
                    session_frr={f"alpha{al}": float((pg <= al).mean()) for al in FRR_TARGETS},
                    note=("NEAR-TAUTOLOGICAL: the attacker copies the victim session's own gap "
                          "sequence, so the two arms differ only through duration."
                          if which == "gaps_copy" else
                          "attacker paces from the human interval distribution (paced_emp gaps)"))
    # duration share
    for which in ("gaps_copy", "gaps_emp"):
        for regime in ("touch", "keystroke"):
            a = res.get(f"{which}|raw|{regime}")
            b = res.get(f"{which}|duration_ablated|{regime}")
            if a and b:
                for al in FRR_TARGETS:
                    k = f"alpha{al}"
                    res.setdefault("duration_share", {})[f"{which}|{regime}|{k}"] = dict(
                        caught_raw=a["caught"][k], caught_after_duration_ablation=b["caught"][k],
                        absolute_loss=a["caught"][k] - b["caught"][k],
                        fraction_of_caught_attributable_to_duration=(
                            (a["caught"][k] - b["caught"][k]) / a["caught"][k]
                            if a["caught"][k] > 0 else None))
    return res


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--modality")
    ap.add_argument("--detector")
    ap.add_argument("--surrogate-scope", default="all")
    a = ap.parse_args()
    if a.stage == "prereg":
        jdump(PREREG, os.path.join(RES, "preregistration.json"))
        log("prereg written")
    elif a.stage == "cache":
        stage_cache()
    elif a.stage == "partA":
        stage_partA(a.modality, a.detector, a.surrogate_scope)
    elif a.stage == "partB":
        stage_partB()
    else:
        raise SystemExit("unknown stage " + a.stage)


if __name__ == "__main__":
    main()
