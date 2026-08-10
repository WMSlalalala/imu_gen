#!/usr/bin/env python
"""Collate the Part A / Part B / Part C outputs into summary tables + text."""
import json
import os
import sys
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
SESSIONS_PER_DAY = 8.0

A = json.load(open(os.path.join(OUT, "raw", "partA_records.json")))
B = json.load(open(os.path.join(OUT, "raw", "partB_results.json")))
D = json.load(open(os.path.join(OUT, "raw", "diagnostics.json")))
REC = A["records"]

DETS = ["llr_mean", "llr_sum", "count", "mean_score", "dur_llr"]
VICTIMS = sorted({r["victim"] for r in REC})


def sel(**kw):
    out = [r for r in REC if all(r.get(k) == v for k, v in kw.items())]
    # A3 is the ORACLE: the attacker filters with the VICTIM cell's own score.
    # The runner also produced key != victim combinations (which are just extra
    # A1 surrogate pairings); keep only the true oracle rows here.
    if kw.get("rule") == "A3":
        out = [r for r in out if r["key"] == r["victim"]]
    return out


def g(r, mode, target, field):
    return r["result"][f"{mode}_frr{target}"][field]


summary = {}

# ---------------------------------------------------------------- A: main curve
curve = {}
for rg in ("touch", "keystroke"):
    for det in DETS:
        for rule in ("A0", "A1", "A2", "A2inv", "A3"):
            for r in ([1] if rule == "A0" else [2, 5, 10, 20]):
                rr = sel(rule=rule, r=r, detector_stat=det, regime=rg)
                if not rr:
                    continue
                # per victim: median over surrogate keys (A1) / modalities (A2)
                byv = {}
                for x in rr:
                    byv.setdefault(x["victim"], []).append(x)
                c5, f5, c1, f1, far, c5o = [], [], [], [], [], []
                for v, xs in byv.items():
                    c5.append(np.median([g(x, "split", 0.05, "caught") for x in xs]))
                    f5.append(np.median([g(x, "split", 0.05, "frr") for x in xs]))
                    c1.append(np.median([g(x, "split", 0.01, "caught") for x in xs]))
                    f1.append(np.median([g(x, "split", 0.01, "frr") for x in xs]))
                    c5o.append(np.median([g(x, "oracle", 0.05, "caught") for x in xs]))
                    far.append(np.median([x["per_event_far"] for x in xs]))
                # pooled CI across all victims x keys: take the mean of the
                # user-clustered bootstrap bounds (cells are not independent)
                lo = np.mean([g(x, "split", 0.05, "caught_ci")[0] for x in rr])
                hi = np.mean([g(x, "split", 0.05, "caught_ci")[1] for x in rr])
                lo1 = np.mean([g(x, "split", 0.01, "caught_ci")[0] for x in rr])
                hi1 = np.mean([g(x, "split", 0.01, "caught_ci")[1] for x in rr])
                price = {}
                for k in ("far_for_caught_50", "far_for_caught_80", "far_for_caught_95"):
                    vals = [x["result"]["price"][k].get("session_far") for x in rr]
                    vals = [v for v in vals if v is not None]
                    price[k] = dict(n_reachable=len(vals), n_total=len(rr),
                                    median_session_far=(float(np.median(vals))
                                                        if vals else None))
                p90 = [x["result"]["p90_user_le_1pct"] for x in rr]
                p90c = [q["caught"] for q in p90 if "caught" in q]
                p90f = [q["pooled_frr"] for q in p90 if "pooled_frr" in q]
                curve[f"{rg}|{det}|{rule}|r{r}"] = dict(
                    n_victims=len(byv), n_records=len(rr),
                    caught_frr5_mean_over_victims=float(np.mean(c5)),
                    caught_frr5_median=float(np.median(c5)),
                    caught_frr5_min=float(np.min(c5)), caught_frr5_max=float(np.max(c5)),
                    caught_frr5_ci_mean=[float(lo), float(hi)],
                    achieved_frr5=float(np.mean(f5)),
                    caught_frr1_mean_over_victims=float(np.mean(c1)),
                    caught_frr1_ci_mean=[float(lo1), float(hi1)],
                    achieved_frr1=float(np.mean(f1)),
                    caught_frr5_oracle_calib=float(np.mean(c5o)),
                    per_event_far_mean=float(np.mean(far)),
                    price=price,
                    p90user_caught_mean=(float(np.mean(p90c)) if p90c else None),
                    p90user_pooled_frr_mean=(float(np.mean(p90f)) if p90f else None),
                    n_p90_reachable=len(p90c))
summary["A_curve"] = curve

# per-event FAR on the SUBMITTED events, unweighted mean over the 90
# (action, cell) pairs -- directly comparable with the paper's 0.7746
pe = {}
for rule in ("A0", "A1", "A2", "A3"):
    for r in ([1] if rule == "A0" else [2, 5, 10, 20]):
        rows = [x for x in sel(rule=rule, r=r, detector_stat="llr_mean",
                               regime="touch")]
        if not rows:
            continue
        byv = {}
        for x in rows:
            byv.setdefault(x["victim"], []).append(x)
        per_cell = []
        for v, xs in byv.items():
            for a in xs[0]["per_event_far_by_action"]:
                per_cell.append(float(np.median([x["per_event_far_by_action"][a]
                                                 for x in xs])))
        pe[f"{rule}|r{r}"] = dict(n_action_cells=len(per_cell),
                                  mean_far_over_90=float(np.mean(per_cell)),
                                  min=float(np.min(per_cell)),
                                  max=float(np.max(per_cell)))
summary["A_per_event_far_90cells"] = pe

# ------------------------------------------------- A1 spread over surrogates
spread = {}
for rg in ("touch", "keystroke"):
    for r in (2, 5, 10, 20):
        rows = sel(rule="A1", r=r, detector_stat="llr_mean", regime=rg)
        byv = {}
        for x in rows:
            byv.setdefault(x["victim"], []).append(x)
        allc, samemod, diffdet = [], [], []
        per_v = {}
        for v, xs in byv.items():
            cs = [g(x, "split", 0.05, "caught") for x in xs]
            allc.extend(cs)
            vm, vd = v.split("|")
            sm = [g(x, "split", 0.05, "caught") for x in xs
                  if x["key"].split("|")[0] == vm]
            dd = [g(x, "split", 0.05, "caught") for x in xs
                  if x["key"].split("|")[1] != vd]
            samemod.extend(sm)
            diffdet.extend(dd)
            per_v[v] = dict(min=float(np.min(cs)), median=float(np.median(cs)),
                            max=float(np.max(cs)),
                            worst_surrogate=xs[int(np.argmax(cs))]["key"],
                            best_surrogate=xs[int(np.argmin(cs))]["key"])
        spread[f"{rg}|r{r}"] = dict(
            all=dict(n=len(allc), min=float(np.min(allc)),
                     p25=float(np.percentile(allc, 25)),
                     median=float(np.median(allc)),
                     p75=float(np.percentile(allc, 75)), max=float(np.max(allc)),
                     mean=float(np.mean(allc))),
            same_modality_surrogates=dict(n=len(samemod), median=float(np.median(samemod)),
                                          min=float(np.min(samemod)), max=float(np.max(samemod))),
            different_detector_surrogates=dict(n=len(diffdet),
                                               median=float(np.median(diffdet)),
                                               min=float(np.min(diffdet)),
                                               max=float(np.max(diffdet))),
            per_victim=per_v)
summary["A1_surrogate_spread"] = spread

# ------------------------------------------------- collapse point
collapse = {}
for rg in ("touch", "keystroke"):
    for det in DETS:
        for rule in ("A1", "A2", "A3"):
            got = None
            for r in (2, 5, 10, 20):
                k = f"{rg}|{det}|{rule}|r{r}"
                if k not in curve:
                    continue
                c = curve[k]["caught_frr5_mean_over_victims"]
                f = curve[k]["achieved_frr5"]
                if c < f and got is None:
                    got = dict(r=r, caught=c, frr=f)
            collapse[f"{rg}|{det}|{rule}"] = got or dict(
                never=True,
                caught_at_r20=curve.get(f"{rg}|{det}|{rule}|r20", {}).get(
                    "caught_frr5_mean_over_victims"),
                frr=curve.get(f"{rg}|{det}|{rule}|r20", {}).get("achieved_frr5"))
summary["collapse_point_frr5"] = collapse

# ------------------------------------------------- Part C
partC = {}
for rg in ("touch", "keystroke"):
    for det in DETS:
        for rule, r in (("A0", 1), ("A1", 5), ("A2", 5), ("A3", 5)):
            rows = sel(rule=rule, r=r, detector_stat=det, regime=rg)
            if not rows:
                continue
            mins, meds, p90s, maxs, rhos = [], [], [], [], []
            for x in rows:
                pu = x["result"]["split_frr0.05"]["per_user_frr"]
                mins.append(pu["min"])
                meds.append(pu["median"])
                p90s.append(pu["p90"])
                maxs.append(pu["max"])
                bb = x["result"].get("per_user_betabinom_frr5", {})
                if bb.get("rho") is not None:
                    rhos.append(bb["rho"])
            med = float(np.mean(meds))
            p90 = float(np.mean(p90s))
            partC[f"{rg}|{det}|{rule}|r{r}"] = dict(
                per_user_frr_min=float(np.mean(mins)),
                per_user_frr_median=med, per_user_frr_p90=p90,
                per_user_frr_max=float(np.mean(maxs)),
                spread_ratio_max_over_min=(float(np.mean(maxs)) /
                                           max(float(np.mean(mins)), 1e-9)),
                betabinom_rho_mean=(float(np.mean(rhos)) if rhos else None),
                lockout_days_median_user=(1.0 / (med * SESSIONS_PER_DAY)
                                          if med > 0 else None),
                lockout_days_p90_user=(1.0 / (p90 * SESSIONS_PER_DAY)
                                       if p90 > 0 else None),
                sessions_per_day=SESSIONS_PER_DAY,
                p90user_operating_point=curve[f"{rg}|{det}|{rule}|r{r}"]
                .get("p90user_caught_mean"),
                p90user_operating_point_frr=curve[f"{rg}|{det}|{rule}|r{r}"]
                .get("p90user_pooled_frr_mean"),
                caught_at_pooled_5pct=curve[f"{rg}|{det}|{rule}|r{r}"]
                ["caught_frr5_mean_over_victims"])
summary["C_per_user"] = partC

# ------------------------------------------------- Part B
summary["B"] = {k: {rg: dict(
    caught_frr5=v[rg]["split_frr0.05"]["caught"],
    frr5=v[rg]["split_frr0.05"]["frr"],
    caught_frr5_ci=v[rg]["split_frr0.05"]["caught_ci"],
    caught_frr1=v[rg]["split_frr0.01"]["caught"],
    frr1=v[rg]["split_frr0.01"]["frr"],
    caught_matched_frr5=v[rg]["at_achieved_frr0.05"]["caught"],
    matched_frr5=v[rg]["at_achieved_frr0.05"]["frr"],
    caught_oracle_frr5=v[rg]["oracle_frr0.05"]["caught"],
    price=v[rg]["price"],
    per_user=v[rg]["split_frr0.05"]["per_user_frr"],
) for rg in ("touch", "keystroke")}
    for k, v in B.items() if k.startswith("duration_only") or k.startswith("timing_")}
summary["B_bulk_stats"] = B["bulk_stats"]
summary["B_qmap_stats"] = {k: v for k, v in B.items() if k.startswith("quantile_map_stats")}
summary["B_gap_pool"] = B["gap_pool"]

# ------------------------------------------------- diagnostics + shortfalls
sv = D["surrogate_vs_victim"]
summary["diag_surrogate_rank_corr"] = dict(
    n_pairs=len(sv),
    mean=float(np.mean([v["mean"] for v in sv.values()])),
    min=float(np.min([v["mean"] for v in sv.values()])),
    max=float(np.max([v["mean"] for v in sv.values()])),
    lowest5=sorted(sv.items(), key=lambda kv: kv[1]["mean"])[:5],
    highest5=sorted(sv.items(), key=lambda kv: -kv[1]["mean"])[:5])
av = D["a2_vs_victim"]
summary["diag_a2_rank_corr"] = dict(
    mean=float(np.mean([v["mean"] for v in av.values()])),
    min=float(np.min([v["mean"] for v in av.values()])),
    max=float(np.max([v["mean"] for v in av.values()])))
summary["shortfalls"] = {k: v for k, v in A["shortfalls"].items()
                         if v["events_short"] > 0}
summary["shortfall_summary"] = {
    f"keep{v['keep_per_pool']}": dict(blocks_short=v["blocks_short"],
                                      events_short=v["events_short"],
                                      total_blocks=v["total_blocks"],
                                      total_slots=v["total_slots"])
    for v in A["shortfalls"].values()}

json.dump(summary, open(os.path.join(OUT, "raw", "summary_b.json"), "w"),
          indent=1, sort_keys=True, default=float)
print("wrote", os.path.join(OUT, "raw", "summary_b.json"))
