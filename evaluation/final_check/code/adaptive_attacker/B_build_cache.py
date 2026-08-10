#!/usr/bin/env python
"""Cache builder for the adaptive-attacker / duration-confound experiment (agent B).

Produces, under cache/:
  bundle_<name>.npz   per-event duration + A2 self-consistency feature stats, per bundle
  bindings.npz        full genuine event inventory (session skeleton denominator + dev durations)
  cells.npz           per-event scores for all 90 released cells

Read-only w.r.t. /mnt/share.  No GPU, no model loading.
"""
import os, sys, json, gzip, glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
os.makedirs(CACHE, exist_ok=True)

D100K = "/mnt/share/mwang49/data7/results/direct100k"
CELLMAP = "/mnt/share/mwang49/data7/code/baselines/release_cell_map.json"
BINDINGS = os.path.join(D100K, "genuine_bindings_v1/genuine_bindings.jsonl")

IMU_CH = list(range(6))
TRAJ_CH = [1, 2, 5, 6]           # x, y, dx, dy   (0=contact flag, 7=elapsed, 8=mask const)
QS = (5.0, 50.0, 95.0)


def shard_features(path):
    z = np.load(path, allow_pickle=True)
    off = z["offsets"].astype(np.int64)
    tf = z["trajectory_flat"]
    imu = z["imu_flat"]
    n = len(off) - 1
    dur = tf[off[1:] - 1, 7].astype(np.float64)
    first = tf[off[:-1], 7].astype(np.float64)
    nsamp = (off[1:] - off[:-1]).astype(np.int64)
    fi = np.zeros((n, len(IMU_CH) * 5), np.float32)
    ft = np.zeros((n, len(TRAJ_CH) * 5), np.float32)
    for i in range(n):
        a, b = off[i], off[i + 1]
        seg_i = imu[a:b][:, IMU_CH]
        seg_t = tf[a:b][:, TRAJ_CH]
        for seg, out in ((seg_i, fi), (seg_t, ft)):
            m = seg.mean(0); s = seg.std(0)
            p = np.percentile(seg, QS, axis=0)
            out[i] = np.concatenate([m, s, p[0], p[1], p[2]])
    return dict(
        event_id=z["event_id"], user_id=z["user_id"], session_id=z["session_id"],
        action=z["action"], label=z["label"].astype(np.int8),
        split=np.full(n, str(z["split"])), duration=dur, first_elapsed=first,
        nsamp=nsamp, feat_imu=fi, feat_traj=ft,
    )


def build_bundle(bundle):
    out = os.path.join(CACHE, f"bundle_{bundle}.npz")
    if os.path.exists(out):
        return bundle, "cached"
    paths = sorted(glob.glob(os.path.join(D100K, bundle, "shards", "*.npz")))
    parts = [shard_features(p) for p in paths]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(out, **merged)
    return bundle, f"{len(merged['event_id'])} events from {len(paths)} shards"


def build_bindings():
    out = os.path.join(CACHE, "bindings.npz")
    if os.path.exists(out):
        return "cached"
    uid, sid, act, dur, spl, ordn, scl, startS = [], [], [], [], [], [], [], []
    with open(BINDINGS) as fh:
        for line in fh:
            d = json.loads(line)
            uid.append(d["user_id"]); sid.append(d["session_id"]); act.append(d["action"])
            dur.append(d.get("duration_ms", np.nan)); spl.append(d["split"])
            ordn.append(d.get("source_event_ordinal", -1))
            scl.append(d.get("source_cluster_id", ""))
            startS.append(d.get("start_sample", -1))
    np.savez_compressed(out, user_id=np.array(uid), session_id=np.array(sid),
                        action=np.array(act), duration_ms=np.array(dur, np.float64),
                        split=np.array(spl), ordinal=np.array(ordn, np.int64),
                        source_cluster_id=np.array(scl), start_sample=np.array(startS, np.int64))
    return f"{len(uid)} genuine binding rows"


def build_cells():
    out = os.path.join(CACHE, "cells.npz")
    if os.path.exists(out):
        return "cached"
    cm = json.load(open(CELLMAP))["cells"]
    store = {}
    for cell, info in sorted(cm.items()):
        p = info["scores"]
        ev, sc, lb, us, ss, ac = [], [], [], [], [], []
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as fh:
            for line in fh:
                d = json.loads(line)
                ev.append(d["event_id"]); sc.append(d["fake_high_score"]); lb.append(d["label"])
                us.append(d["user_id"]); ss.append(d["session_id"]); ac.append(d["action"])
        store[cell + "|event_id"] = np.array(ev)
        store[cell + "|score"] = np.array(sc, np.float64)
        store[cell + "|label"] = np.array(lb, np.int8)
        store[cell + "|user_id"] = np.array(us)
        store[cell + "|session_id"] = np.array(ss)
        store[cell + "|action"] = np.array(ac)
    np.savez_compressed(out, **store)
    return f"{len(cm)} cells"


if __name__ == "__main__":
    bundles = ["replay_dataset_full", "replay_dataset_v3", "replay_dataset_v8",
               "replay_dataset_v10", "replay_dataset_v15"]
    with ProcessPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(build_bundle, b) for b in bundles]
        futs.append(ex.submit(build_bindings))
        futs.append(ex.submit(build_cells))
        for f in futs:
            print(f.result(), flush=True)
    print("DONE")
