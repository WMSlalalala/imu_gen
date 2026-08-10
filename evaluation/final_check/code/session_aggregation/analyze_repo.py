#!/usr/bin/env python
"""Same recomputation against the REPO-published (gzipped) 90-cell release, to
locate which release the 0.775 aggregate belongs to."""
import json, os, gzip, statistics, collections
import numpy as np

ROOT = "/home/mwang49/new/data7/data7_final_monitor_metrics_v1/USENIX8.25/code/dataset_test/results/cells"
MNT = "/mnt/share/mwang49/data7/results/direct100k/detectors_90cell/cells"
OUT = "/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/sessagg"
ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
MODALITIES = ("trajectory_xytime", "imu_only", "imu_trajectory_xytime")
DETECTORS = ("hmog_style_svm", "hmog_style_rf", "paper_svm",
             "paper_xgboost", "behaveformer_stdat", "authconformer")

rows = []
gen_sess_action = {}
fake_user_action = collections.Counter()
manifests = collections.Counter()
identical = []
for a in ACTIONS:
    for m in MODALITIES:
        for d in DETECTORS:
            name = f"{a}__{m}__{d}"
            cp = os.path.join(ROOT, name)
            t = json.load(open(os.path.join(cp, "thresholds.json")))
            manifests[t["manifest_sha256"]] += 1
            tm = json.load(open(os.path.join(MNT, name, "thresholds.json")))
            if t == tm:
                identical.append(name)
            cut5, cutE = float(t["frr5"]), float(t["eer"])
            s = json.load(open(os.path.join(cp, "summary.json")))
            gs, fs = [], []
            for line in gzip.open(os.path.join(cp, "test_scores.jsonl.gz"), "rt"):
                r = json.loads(line)
                sc = float(r["fake_high_score"])
                if r["label"] == 1:
                    fs.append(sc)
                    if (m, d) == (MODALITIES[0], DETECTORS[0]):
                        fake_user_action[(r["user_id"], a)] += 1
                else:
                    gs.append(sc)
                    if (m, d) == (MODALITIES[0], DETECTORS[0]):
                        k = (r["user_id"], r["session_id"], a)
                        gen_sess_action[k] = gen_sess_action.get(k, 0) + 1
            g = np.asarray(gs); f = np.asarray(fs)
            rows.append(dict(action=a, modality=m, detector=d,
                             n_gen=len(g), n_fake=len(f),
                             far5=float((f < cut5).mean()),
                             frr5=float((g >= cut5).mean()),
                             farE=float((f < cutE).mean()),
                             frrE=float((g >= cutE).mean()),
                             sum_far=s["primary_metrics"]["far"],
                             sum_frr=s["primary_metrics"]["frr"]))

json.dump(rows, open(os.path.join(OUT, "cell_table_repo.json"), "w"), indent=1)
far5 = sorted(r["far5"] for r in rows)
frr5 = sorted(r["frr5"] for r in rows)
farE = sorted(r["farE"] for r in rows)
print("REPO-PUBLISHED RELEASE", ROOT)
print("manifest_sha256 histogram:", dict(manifests))
print(f"cells with thresholds.json identical to /mnt/share run: {len(identical)}")
print("   ", sorted(identical))
print()
print(f"FAR@dev-frr5: mean {statistics.mean(far5):.6f}  median "
      f"{statistics.median(far5):.6f}  min {far5[0]:.4f}  max {far5[-1]:.4f}")
print(f"   >=0.5: {sum(v>=0.5 for v in far5)}/90   >=0.6: {sum(v>=0.6 for v in far5)}/90"
      f"   >=0.48: {sum(v>=0.48 for v in far5)}/90")
print(f"FRR@dev-frr5: mean {statistics.mean(frr5):.6f}  median {statistics.median(frr5):.6f}")
print(f"FAR@dev-eer : mean {statistics.mean(farE):.6f}  median {statistics.median(farE):.6f}")
tot = sum(r['far5']*r['n_fake'] for r in rows)/sum(r['n_fake'] for r in rows)
print(f"event-pooled FAR@frr5: {tot:.6f}")
mx = max(abs(r["farE"]-r["sum_far"]) for r in rows)
print(f"max |FAR@eer - summary.far| = {mx:.3e}")
print()
byk = {(r['action'],r['modality'],r['detector']): r for r in rows}
for k in [("scroll","trajectory_xytime","hmog_style_svm"),
          ("tap","imu_only","paper_xgboost"),
          ("keystroke","imu_trajectory_xytime","authconformer"),
          ("swipe","trajectory_xytime","behaveformer_stdat"),
          ("pinch","imu_only","hmog_style_rf"),
          ("scroll","imu_trajectory_xytime","paper_svm")]:
    r = byk[k]
    print(f"  {k[0]:<10s}{k[1]:<24s}{k[2]:<20s} FAR@frr5={r['far5']:.4f} "
          f"FRR@frr5={r['frr5']:.4f}  FAR@eer={r['farE']:.4f} (summary {r['sum_far']:.4f})"
          f"  n_gen={r['n_gen']} n_fake={r['n_fake']}")
print()
for label in ("action","modality","detector"):
    gr = collections.defaultdict(list)
    for r in rows: gr[r[label]].append(r['far5'])
    print(f"-- by {label}")
    for k,v in sorted(gr.items(), key=lambda kv:-statistics.median(kv[1])):
        v=sorted(v)
        print(f"   {k:<24s} n={len(v):<3d} mean {statistics.mean(v):.4f} median "
              f"{statistics.median(v):.4f} min {v[0]:.4f} max {v[-1]:.4f}")
print()
print("genuine counts per action (repo release):",
      {a: byk[(a,'trajectory_xytime','hmog_style_svm')]['n_gen'] for a in ACTIONS})
print("fake counts per action (repo release):",
      {a: byk[(a,'trajectory_xytime','hmog_style_svm')]['n_fake'] for a in ACTIONS})
sessions = sorted({(u,s) for (u,s,a) in gen_sess_action})
users = sorted({u for (u,s,a) in gen_sess_action})
print("users:", len(users), " sessions:", len(sessions),
      " total genuine:", sum(gen_sess_action.values()),
      " total fake:", sum(fake_user_action.values()))
lens = [sum(gen_sess_action.get((u,s,a),0) for a in ACTIONS) for (u,s) in sessions]
nact = collections.Counter(sum(1 for a in ACTIONS if gen_sess_action.get((u,s,a),0)>0)
                           for (u,s) in sessions)
print("session length: min",min(lens),"median",statistics.median(lens),
      "mean",round(statistics.mean(lens),2),"max",max(lens))
print("distinct actions per session:", dict(sorted(nact.items())))
short = [(u,a,sum(gen_sess_action.get((u,s,a),0) for (uu,s) in sessions if uu==u),
          fake_user_action.get((u,a),0))
         for u in users for a in ACTIONS]
bad = [x for x in short if x[2] > x[3]]
print("shortfalls (gen slots > fake avail):", len(bad), bad[:20])
