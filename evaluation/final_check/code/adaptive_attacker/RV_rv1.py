#!/usr/bin/env python
"""Reviewer check 1: plumbing sanity, pool provenance x selection, reuse/duplication."""
import os, sys, json, pickle, collections
import numpy as np
sys.path.insert(0, "/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/B")
import adaptive_b as AB

RV = "/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/RV"
D = pickle.load(open(os.path.join(RV, "idx.pkl"), "rb"))
IDX, REG, cm = D["IDX"], D["REG"], D["cm"]
ACT = AB.ACTIONS
out = {}

# ---------- plumbing
fars = {c: float((IDX["cellS"][c]["fake"][IDX["cellS"][c]["action"]] < IDX["cellS"][c]["tau"]).mean())
        for c in cm}
out["far_frr5_mean_90cells"] = float(np.mean(list(fars.values())))
out["far_tap_imu_xgb"] = fars["tap__imu_only__paper_xgboost"]

# ---------- session inventory
for reg in ("touch", "keystroke"):
    R = REG[reg]
    out[f"{reg}_n_sessions"] = int(R.n_sess)
    out[f"{reg}_n_slots"] = int(R.M)
    out[f"{reg}_max_group"] = int(R.group_count.max())
    out[f"{reg}_group_count_hist_top"] = collections.Counter(R.group_count.tolist()).most_common(8)

# ---------- REUSE: exact within-session duplicate events under B's draw rule
reuse = {}
for reg in ("touch", "keystroke"):
    R = REG[reg]
    n_sess = R.n_sess
    for r in (1, 2, 5, 10, 20):
        P = 200 // r
        dup_slot = R.rank >= P                       # slot forced onto an already-used pool item
        sess_has_dup = np.zeros(n_sess, bool)
        np.logical_or.at(sess_has_dup, R.slot_sess, dup_slot)
        # multiplicity: how many times each distinct submitted event is repeated in a session
        mult = collections.Counter()
        for g in range(R.n_groups):
            c = int(R.group_count[g])
            for j in range(min(c, P)):
                k = 1 + (c - 1 - j) // P if c > j else 1
                mult[k] += 1
        reuse[f"{reg}|r{r}"] = dict(
            pool_kept=P,
            forced_reuse_slots=int(dup_slot.sum()),
            forced_reuse_frac=float(dup_slot.mean()),
            sessions_with_at_least_one_exact_duplicate=int(sess_has_dup.sum()),
            frac_sessions_with_duplicate=float(sess_has_dup.mean()),
            max_multiplicity=int(max(mult)),
            distinct_events_per_victim_action=P,
        )
out["within_session_reuse"] = reuse

# ---------- CROSS-SESSION reuse: pool of P events serves the whole campaign
xs = {}
for reg in ("touch", "keystroke"):
    R = REG[reg]
    # slots per (user, action) over the whole regime
    key = R.slot_user * 5 + R.slot_action
    cnt = collections.Counter(key.tolist())
    for r in (1, 5, 10, 20):
        P = 200 // r
        dem = np.array(sorted(cnt.values()))
        xs[f"{reg}|r{r}"] = dict(
            pool_kept=P,
            slots_per_user_action_median=float(np.median(dem)),
            slots_per_user_action_max=int(dem.max()),
            mean_submissions_per_distinct_event=float(np.mean(dem) / P),
            max_submissions_per_distinct_event=float(dem.max() / P),
        )
out["campaign_level_reuse"] = xs
json.dump(out, open(os.path.join(RV, "rv1.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
