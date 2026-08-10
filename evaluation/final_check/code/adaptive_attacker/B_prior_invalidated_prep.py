"""Stage 1 preparation for the adaptive-attacker / duration-confound experiment (agent B).

Pure score-space and duration-space arithmetic.  No GPU, no model loading.

Builds, under CACHE/:
  cells_<action>.npz      per-action event table + (18, n_event) score matrix
  shards_test.npz         per-event duration + attacker-side features for test users
  shards_devtrain.npz     per-event duration for development and train users
  onsets_test.json        real HMOG onset/offset for every scored genuine test event
  session_inventory.json  total raw slots per real session (denominator for scored-slot share)
"""
import json
import os
import sys
import time
import collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

CELLROOT = "/mnt/share/mwang49/data7/results/direct100k/detectors_90cell/cells"
SHARDS = "/mnt/share/mwang49/data7/results/direct100k/replay_dataset_full/shards"
PROV = "/mnt/share/mwang49/data7/results/direct100k/replay_dataset_full/provenance.jsonl"
RAWTRAJ = ("/mnt/share/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713"
           "/results/trajectories_full_v2/hmog_trajectory_%s.npz")

ACTIONS = ["tap", "scroll", "swipe", "pinch", "keystroke"]
MODALITIES = ["imu_only", "imu_trajectory_xytime", "trajectory_xytime"]
DETECTORS = ["authconformer", "behaveformer_stdat", "hmog_style_rf",
             "hmog_style_svm", "paper_svm", "paper_xgboost"]
CELLKEYS = [(m, d) for m in MODALITIES for d in DETECTORS]   # 18, fixed order


def log(*a):
    print("[prep]", *a, flush=True)


# ----------------------------------------------------------------------------
# 1. cells
# ----------------------------------------------------------------------------
def build_cells():
    thresholds = {}
    for action in ACTIONS:
        out = os.path.join(CACHE, "cells_%s.npz" % action)
        eids = users = sess = labels = None
        scores = np.zeros((len(CELLKEYS), 0), dtype=np.float64)
        cols = []
        for (m, d) in CELLKEYS:
            cd = os.path.join(CELLROOT, "%s__%s__%s" % (action, m, d))
            rows = [json.loads(l) for l in open(os.path.join(cd, "test_scores.jsonl"))]
            e = np.array([r["event_id"] for r in rows])
            if eids is None:
                eids = e
                users = np.array([r["user_id"] for r in rows])
                sess = np.array([r["session_id"] for r in rows])
                labels = np.array([int(r["label"]) for r in rows], dtype=np.int8)
            else:
                assert np.array_equal(eids, e), "event order differs %s %s %s" % (action, m, d)
            cols.append(np.array([float(r["fake_high_score"]) for r in rows]))
            th = json.load(open(os.path.join(cd, "thresholds.json")))
            assert th["score_direction"] == "larger_is_more_fake"
            thresholds["%s|%s|%s" % (action, m, d)] = {
                "frr5": float(th["frr5"]), "eer": float(th["eer"]),
                "selection_split": th["selection_split"], "target_frr": th["target_frr"]}
        scores = np.vstack(cols)
        np.savez_compressed(out, event_id=eids, user_id=users, session_id=sess,
                            label=labels, scores=scores,
                            cellkeys=np.array(["%s|%s" % k for k in CELLKEYS]))
        log("cells", action, scores.shape, "genuine", int((labels == 0).sum()),
            "fake", int((labels == 1).sum()))
    json.dump(thresholds, open(os.path.join(CACHE, "thresholds.json"), "w"), indent=1)


# ----------------------------------------------------------------------------
# 2. shards: duration + attacker features
# ----------------------------------------------------------------------------
FEATNAMES = None


def event_features(imu, traj):
    """Attacker-side summary features of ONE event.  Uses only the generated signal."""
    f = []
    names = []
    for c in range(6):
        x = imu[:, c]
        dx = np.diff(x) if len(x) > 1 else np.zeros(1)
        f += [x.mean(), x.std(), np.percentile(x, 5), np.percentile(x, 95), np.abs(dx).mean()]
        names += ["imu%d_%s" % (c, s) for s in ("mean", "std", "p05", "p95", "adiff")]
    for c in (1, 2, 5, 6):
        x = traj[:, c]
        f += [x.mean(), x.std(), x.min(), x.max()]
        names += ["tr%d_%s" % (c, s) for s in ("mean", "std", "min", "max")]
    # path length in normalised screen coordinates
    if len(traj) > 1:
        pl = float(np.abs(np.diff(traj[:, 1])).sum() + np.abs(np.diff(traj[:, 2])).sum())
    else:
        pl = 0.0
    f += [pl, float(len(traj)), float(traj[-1, 7])]
    names += ["pathlen", "n_samples", "duration_s"]
    return np.array(f, dtype=np.float64), names


def build_shards():
    global FEATNAMES
    out_test = collections.defaultdict(list)
    out_dev = collections.defaultdict(list)
    for u in range(100):
        p = os.path.join(SHARDS, "hmog_u%03d.npz" % u)
        d = np.load(p, allow_pickle=True)
        split = str(d["split"])
        off = d["offsets"]
        traj = d["trajectory_flat"]
        imu = d["imu_flat"]
        lab = d["label"]
        act = d["action"]
        eid = d["event_id"]
        uid = d["user_id"]
        sid = d["session_id"]
        # duration = last row of channel 7 (elapsed_seconds); first row must be 0
        first = traj[off[:-1], 7]
        assert np.abs(first).max() == 0.0, "channel 7 does not start at 0 in %s" % p
        dur = traj[off[1:] - 1, 7].astype(np.float64)
        nsamp = (off[1:] - off[:-1]).astype(np.int64)
        tgt = out_test if split == "test" else out_dev
        tgt["event_id"].append(eid)
        tgt["user_id"].append(uid)
        tgt["session_id"].append(sid)
        tgt["action"].append(act)
        tgt["label"].append(lab.astype(np.int8))
        tgt["duration"].append(dur)
        tgt["n_samples"].append(nsamp)
        tgt["split"].append(np.array([split] * len(lab)))
        if split == "test":
            feats = np.zeros((len(lab), 0))
            fl = []
            for i in range(len(lab)):
                if lab[i] != 1:
                    fl.append(None)
                    continue
                v, names = event_features(imu[off[i]:off[i + 1]], traj[off[i]:off[i + 1]])
                FEATNAMES = names
                fl.append(v)
            nf = len(FEATNAMES)
            feats = np.full((len(lab), nf), np.nan)
            for i, v in enumerate(fl):
                if v is not None:
                    feats[i] = v
            tgt["features"].append(feats)
        if u % 20 == 0:
            log("shard", u, split)
    def cat(dd, keys):
        return {k: np.concatenate(dd[k]) for k in keys}
    kt = ["event_id", "user_id", "session_id", "action", "label", "duration", "n_samples", "split"]
    T = cat(out_test, kt)
    T["features"] = np.concatenate(out_test["features"])
    np.savez_compressed(os.path.join(CACHE, "shards_test.npz"),
                        featnames=np.array(FEATNAMES), **T)
    D = cat(out_dev, kt)
    np.savez_compressed(os.path.join(CACHE, "shards_devtrain.npz"), **D)
    log("shards_test", len(T["label"]), "shards_devtrain", len(D["label"]))


# ----------------------------------------------------------------------------
# 3. provenance -> real onsets for scored genuine test events, and slot inventory
# ----------------------------------------------------------------------------
def build_onsets():
    want = {}                      # our event_id -> (action, raw index)
    t0 = time.time()
    n = 0
    with open(PROV) as fh:
        for line in fh:
            n += 1
            if '"label": 0' not in line:
                continue
            r = json.loads(line)
            if r.get("label") != 0:
                continue
            dn = r.get("donor") or {}
            src = dn.get("raw_trajectory_source")
            idx = dn.get("raw_trajectory_event_index")
            if src is None or idx is None:
                continue
            act = os.path.basename(src).replace("hmog_trajectory_", "").replace(".npz", "")
            want[r["event_id"]] = (act, int(idx), dn.get("source_user_id"))
    log("provenance scanned", n, "lines; genuine test events with raw index", len(want),
        "%.1fs" % (time.time() - t0))

    # raw trajectory tables
    raw = {}
    inv = collections.Counter()          # (user_external_id, raw_session_id) -> total events
    for a in ACTIONS:
        d = np.load(RAWTRAJ % a, allow_pickle=True)
        raw[a] = {"start": d["label_start_ms"].astype(np.float64),
                  "end": d["label_end_ms"].astype(np.float64),
                  "dur": d["label_duration_ms"].astype(np.float64),
                  "uext": d["user_external_id"],
                  "sid": d["session_id"],
                  "uid": d["user_id"]}
        for ue, si in zip(raw[a]["uext"], raw[a]["sid"]):
            inv[(int(ue), int(si))] += 1
        log("raw", a, len(raw[a]["start"]))

    rec = {}
    miss = 0
    for eid, (a, idx, su) in want.items():
        if a not in raw or idx >= len(raw[a]["start"]):
            miss += 1
            continue
        rec[eid] = {"action": a, "start_ms": float(raw[a]["start"][idx]),
                    "end_ms": float(raw[a]["end"][idx]),
                    "dur_ms": float(raw[a]["dur"][idx]),
                    "uext": int(raw[a]["uext"][idx]), "raw_sid": int(raw[a]["sid"][idx]),
                    "raw_uid": int(raw[a]["uid"][idx]), "donor_user": su}
    json.dump({"events": rec, "n_missing": miss},
              open(os.path.join(CACHE, "onsets_test.json"), "w"))
    json.dump({"%d|%d" % k: v for k, v in inv.items()},
              open(os.path.join(CACHE, "session_inventory.json"), "w"))
    log("onsets recovered", len(rec), "missing", miss, "raw sessions in inventory", len(inv))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "cells"):
        build_cells()
    if which in ("all", "shards"):
        build_shards()
    if which in ("all", "onsets"):
        build_onsets()
