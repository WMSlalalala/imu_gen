#!/usr/bin/env python
"""Aggregate the 18 per-victim Part A files + Part B into one summary JSON."""
import collections
import glob
import json
import os

import numpy as np

BASE = ("/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/"
        "e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/A")
RES = os.path.join(BASE, "results")
NU = 20
N_BOOT = 2000
BOOT = np.random.default_rng(777).integers(0, NU, size=(N_BOOT, NU))
DETECTORS = ["authconformer", "behaveformer_stdat", "hmog_style_rf",
             "hmog_style_svm", "paper_svm", "paper_xgboost"]
MODALITIES = ["imu_only", "trajectory_xytime", "imu_trajectory_xytime"]
ALPHAS = ["alpha0.05", "alpha0.01", "alpha_min_attainable"]


def shared_boot(num_by_cell, den_by_cell):
    """Cross-cell aggregate with ONE resampled user multiset per replicate (F9)."""
    num = np.array(num_by_cell)      # (cells, NU)
    den = np.array(den_by_cell)
    vals = []
    for b in range(N_BOOT):
        idx = BOOT[b]
        n = num[:, idx].sum(1)
        d = den[:, idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_cell = np.where(d > 0, n / np.maximum(d, 1e-12), np.nan)
        vals.append(np.nanmean(per_cell))
    v = np.array(vals)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    files = sorted(glob.glob(os.path.join(RES, "partA__*.json")))
    cells = {}
    for f in files:
        d = json.load(open(f))
        cells[d["victim"]] = d
    assert len(cells) == 18, len(cells)

    out = {"n_victim_cells": len(cells), "victims": sorted(cells)}

    # ---- reference block ----------------------------------------------------
    ref = {}
    for rg in ("touch", "keystroke"):
        r0 = next(iter(cells.values()))["reference"][rg]
        ref[rg] = dict(n_sessions=r0["n_sessions"], n_slots=r0["n_slots"],
                       users=r0["users"], scored_len_mean=r0["scored_len_mean"],
                       scored_slot_fraction=r0["scored_slot_fraction"],
                       min_attainable_alpha=r0["min_attainable_alpha"]["split|llr_sum"],
                       alpha001_attainable=r0["alpha001_attainable"]["split|llr_sum"])
        # genuine session FRR averaged over cells
        for scope in ("split", "oracle"):
            for sn in ("llr_sum", "llr_mean", "b0_count", "b1_meanscore", "b2_durllr"):
                for al in (0.05, 0.01):
                    k = f"{scope}|{sn}|alpha{al}"
                    ref[rg][f"genuine_session_frr__{k}"] = float(np.mean(
                        [c["reference"][rg]["genuine_session_frr"][k] for c in cells.values()]))
    out["reference"] = ref

    # ---- arm index ----------------------------------------------------------
    def arms(cell, rg):
        return {(a["rule"], a["tag"], a["r"]): a
                for a in cells[cell]["arms"] if a["regime"] == rg}

    idx = {(c, rg): arms(c, rg) for c in cells for rg in ("touch", "keystroke")}

    def agg(keysel, rg, scope="split", sn="llr_sum", alab="alpha0.05"):
        """Average an arm statistic over the 18 victim cells, with a shared-user
        cross-cell bootstrap.  keysel(cell)->arm key, or None to skip the cell."""
        cs, fs, num, den, fnum, ev_far, sd, mins, maxs = [], [], [], [], [], [], [], [], []
        used = []
        for c in sorted(cells):
            k = keysel(c)
            if k is None or k not in idx[(c, rg)]:
                continue
            a = idx[(c, rg)][k]
            s = a["stats"][f"{scope}|{sn}"][alab]
            cs.append(s["caught"]); fs.append(s["session_frr"])
            sd.append(s["caught_sd_over_reps"])
            mins.append(s["caught_min_rep"]); maxs.append(s["caught_max_rep"])
            ev_far.append(a["per_event_far_at_frr5_regime_mean"])
            if "per_user_caught_sum" in s:
                num.append(s["per_user_caught_sum"]); den.append(s["per_user_sessions"])
                fnum.append(s["per_user_frr_sum"])
            used.append(c)
        if not cs:
            return None
        o = dict(n_cells=len(cs), caught_mean=float(np.mean(cs)),
                 caught_sd_over_cells=float(np.std(cs)),
                 caught_min_cell=float(np.min(cs)), caught_max_cell=float(np.max(cs)),
                 session_frr_mean=float(np.mean(fs)),
                 caught_minus_frr=float(np.mean(cs) - np.mean(fs)),
                 caught_sd_over_reps_mean=float(np.mean(sd)),
                 caught_rep_range_mean=float(np.mean(np.array(maxs) - np.array(mins))),
                 per_event_far_mean=float(np.mean(ev_far)),
                 alpha=float(next(iter(
                     idx[(used[0], rg)][keysel(used[0])]["stats"][f"{scope}|{sn}"][alab]
                 )) if False else
                     idx[(used[0], rg)][keysel(used[0])]["stats"][f"{scope}|{sn}"][alab]["alpha"]))
        if num:
            o["caught_ci95_shared_user_boot"] = shared_boot(num, den)
            o["session_frr_ci95_shared_user_boot"] = shared_boot(fnum, den)
        return o

    # ---- A0 / A2 / A3 curves ------------------------------------------------
    curve = {}
    for rg in ("touch", "keystroke"):
        for alab in ALPHAS:
            for scope in ("split", "oracle"):
                base = f"{rg}|{scope}|{alab}"
                curve[f"{base}|A0|uniform|r1"] = agg(
                    lambda c: ("A0", "uniform", 1), rg, scope, "llr_sum", alab)
                for variant in ("centroid", "knn"):
                    for r in (2, 5, 10, 20):
                        curve[f"{base}|A2|{variant}|r{r}"] = agg(
                            lambda c, v=variant, rr=r: ("A2", v, rr), rg, scope,
                            "llr_sum", alab)
                for r in (2, 5, 10, 20):
                    curve[f"{base}|A3_ORACLE_NOT_A_THREAT_MODEL|oracle|r{r}"] = agg(
                        lambda c, rr=r: ("A3_ORACLE_NOT_A_THREAT_MODEL", "oracle", rr),
                        rg, scope, "llr_sum", alab)
    out["adaptive_curve"] = curve

    # ---- A1: per surrogate, then spread -------------------------------------
    a1 = {}
    for rg in ("touch", "keystroke"):
        for alab in ("alpha0.05", "alpha0.01"):
            for r in (2, 5, 10, 20):
                per_surr = {}
                for sm in MODALITIES:
                    for sd_ in DETECTORS:
                        tag = f"{sm}|{sd_}"
                        v = agg(lambda c, t=tag, rr=r: (
                            "A1", t, rr) if c.split("|")[1] != t.split("|")[1] else None,
                            rg, "split", "llr_sum", alab)
                        if v:
                            per_surr[tag] = v
                # split same-modality vs cross-modality relative to each victim
                same, cross = [], []
                for c in sorted(cells):
                    vm, vd = c.split("|")
                    for sm in MODALITIES:
                        for sd_ in DETECTORS:
                            if sd_ == vd:
                                continue
                            k = ("A1", f"{sm}|{sd_}", r)
                            if k not in idx[(c, rg)]:
                                continue
                            s = idx[(c, rg)][k]["stats"][f"split|llr_sum"][alab]
                            (same if sm == vm else cross).append(s["caught"])
                a1[f"{rg}|{alab}|r{r}"] = dict(
                    per_surrogate={k: dict(caught_mean=v["caught_mean"],
                                           caught_min_cell=v["caught_min_cell"],
                                           caught_max_cell=v["caught_max_cell"],
                                           session_frr_mean=v["session_frr_mean"],
                                           per_event_far_mean=v["per_event_far_mean"],
                                           n_cells=v["n_cells"],
                                           ci95=v.get("caught_ci95_shared_user_boot"))
                                   for k, v in per_surr.items()},
                    spread_over_surrogates=dict(
                        min=float(min(v["caught_mean"] for v in per_surr.values())),
                        median=float(np.median([v["caught_mean"] for v in per_surr.values()])),
                        max=float(max(v["caught_mean"] for v in per_surr.values())),
                        argmin=min(per_surr, key=lambda k: per_surr[k]["caught_mean"]),
                        argmax=max(per_surr, key=lambda k: per_surr[k]["caught_mean"])),
                    all_victim_surrogate_pairings=dict(
                        n_same_modality=len(same), n_cross_modality=len(cross),
                        same_modality_caught_mean=float(np.mean(same)) if same else None,
                        same_modality_caught_p05=float(np.percentile(same, 5)) if same else None,
                        same_modality_caught_p95=float(np.percentile(same, 95)) if same else None,
                        cross_modality_caught_mean=float(np.mean(cross)) if cross else None,
                        cross_modality_caught_p05=float(np.percentile(cross, 5)) if cross else None,
                        cross_modality_caught_p95=float(np.percentile(cross, 95)) if cross else None,
                        pairings_below_genuine_frr=int(sum(
                            1 for c in sorted(cells)
                            for sm in MODALITIES for sd_ in DETECTORS
                            if sd_ != c.split("|")[1]
                            and ("A1", f"{sm}|{sd_}", r) in idx[(c, rg)]
                            and idx[(c, rg)][("A1", f"{sm}|{sd_}", r)]["stats"]["split|llr_sum"][alab]["caught"] <
                            idx[(c, rg)][("A1", f"{sm}|{sd_}", r)]["stats"]["split|llr_sum"][alab]["session_frr"])),
                        pairings_total=len(same) + len(cross)))
    out["A1_surrogate"] = a1

    # ---- collapse point -----------------------------------------------------
    collapse = {}
    for rg in ("touch", "keystroke"):
        for alab in ("alpha0.05", "alpha0.01"):
            for rule, tag in (("A2", "centroid"), ("A2", "knn"),
                              ("A3_ORACLE_NOT_A_THREAT_MODEL", "oracle")):
                got = None
                for r in (1, 2, 5, 10, 20):
                    key = ("A0", "uniform", 1) if r == 1 else (rule, tag, r)
                    v = agg(lambda c, k=key: k, rg, "split", "llr_sum", alab)
                    if v and v["caught_mean"] < v["session_frr_mean"]:
                        got = r
                        break
                collapse[f"{rg}|{alab}|{rule}|{tag}"] = got
            # A1: per surrogate
            per = {}
            for sm in MODALITIES:
                for sd_ in DETECTORS:
                    tg = f"{sm}|{sd_}"
                    got = None
                    for r in (1, 2, 5, 10, 20):
                        key = ("A0", "uniform", 1) if r == 1 else ("A1", tg, r)
                        v = agg(lambda c, k=key, t=tg: (
                            k if (k[0] == "A0" or c.split("|")[1] != t.split("|")[1]) else None),
                            rg, "split", "llr_sum", alab)
                        if v and v["caught_mean"] < v["session_frr_mean"]:
                            got = r
                            break
                    per[tg] = got
            collapse[f"{rg}|{alab}|A1|per_surrogate"] = per
    out["collapse_point"] = collapse

    # ---- baselines at the same operating point ------------------------------
    base = {}
    for rg in ("touch", "keystroke"):
        for alab in ("alpha0.05", "alpha0.01"):
            for lab, key in (("A0", ("A0", "uniform", 1)),
                             ("A2_centroid_r10", ("A2", "centroid", 10)),
                             ("A2_knn_r10", ("A2", "knn", 10)),
                             ("A3_ORACLE_r10", ("A3_ORACLE_NOT_A_THREAT_MODEL", "oracle", 10))):
                for sn in ("llr_sum", "llr_mean", "b0_count", "b0_rate",
                           "b1_meanscore", "b2_durllr"):
                    v = agg(lambda c, k=key: k, rg, "split", sn, alab)
                    if v:
                        base[f"{rg}|{alab}|{lab}|{sn}"] = dict(
                            caught=v["caught_mean"], frr=v["session_frr_mean"],
                            n_cells=v["n_cells"])
    out["baselines"] = base

    # ---- price curve --------------------------------------------------------
    price = {}
    for rg in ("touch", "keystroke"):
        for lab, key in (("A0", ("A0", "uniform", 1)),
                         ("A2_centroid_r10", ("A2", "centroid", 10)),
                         ("A1_best_surrogate_r10", None),
                         ("A3_ORACLE_r10", ("A3_ORACLE_NOT_A_THREAT_MODEL", "oracle", 10))):
            for sn in ("llr_sum", "b0_count", "b1_meanscore", "b2_durllr"):
                vals = collections.defaultdict(list)
                for c in sorted(cells):
                    if key is None:
                        ks = [k for k in idx[(c, rg)] if k[0] == "A1" and k[2] == 10]
                        if not ks:
                            continue
                        k = min(ks, key=lambda kk: idx[(c, rg)][kk]["stats"][
                            f"split|{sn}"]["alpha0.05"]["caught"])
                    else:
                        k = key
                        if k not in idx[(c, rg)]:
                            continue
                    p = idx[(c, rg)][k]["stats"][f"split|{sn}"]["price"]
                    for t in (50, 80, 95):
                        v = p[f"frr_for_caught_{t}"]
                        vals[t].append(v if v is not None else 1.0)
                price[f"{rg}|{lab}|{sn}"] = {
                    f"session_frr_for_caught_{t}": float(np.mean(vals[t]))
                    for t in (50, 80, 95)}
                price[f"{rg}|{lab}|{sn}"]["n_cells"] = len(vals[50])
                price[f"{rg}|{lab}|{sn}"]["note_unreachable_counted_as_1.0"] = int(
                    sum(1 for t in (95,) for v in vals[t] if v == 1.0))
    out["price_curve"] = price

    # ---- Part C: who pays ---------------------------------------------------
    partc = {}
    for rg in ("touch", "keystroke"):
        for scope in ("split", "oracle"):
            for sn in ("llr_sum", "b2_durllr"):
                for al in (0.05, 0.01):
                    k = f"{scope}|{sn}|alpha{al}"
                    blocks = [c["reference"][rg]["per_user_frr"][k] for c in cells.values()]
                    partc[f"{rg}|{k}"] = dict(
                        min=float(np.mean([b["min"] for b in blocks])),
                        median=float(np.mean([b["median"] for b in blocks])),
                        p90=float(np.mean([b["p90"] for b in blocks])),
                        max=float(np.mean([b["max"] for b in blocks])),
                        pooled=float(np.mean([b["pooled"] for b in blocks])),
                        worst_cell_max=float(np.max([b["max"] for b in blocks])),
                        ratio_p90_over_median=float(np.mean(
                            [b["p90"] / b["median"] if b["median"] > 0 else np.nan
                             for b in blocks])),
                        beta_binomial_icc=float(np.nanmean(
                            [b["spread"].get("icc", np.nan) for b in blocks])),
                        beta_binomial_q90=float(np.nanmean(
                            [b["spread"].get("q90", np.nan) for b in blocks])),
                        beta_binomial_q99=float(np.nanmean(
                            [b["spread"].get("q99", np.nan) for b in blocks])),
                        lockout_days_between_median_user=float(np.nanmean(
                            [b["lockout_median_user_days_between"] or np.nan
                             for b in blocks])),
                        lockout_days_between_p90_user=float(np.nanmean(
                            [b["lockout_p90_user_days_between"] or np.nan
                             for b in blocks])),
                        sessions_per_day_assumed=6)
        # alternative operating point
        alt = [c["reference"][rg]["alt_operating_point_alpha_p90user_frr_le_1pct"]["split|llr_sum"]
               for c in cells.values()]
        have = [a for a in alt if a is not None]
        rows = []
        for c in sorted(cells):
            a = idx[(c, rg)].get(("A0", "uniform", 1))
            if a is None:
                continue
            s = a["stats"]["split|llr_sum"]
            if s["alt_op_p90user_frr_le_1pct"]:
                rows.append((s["alt_op_p90user_frr_le_1pct"]["caught"],
                             s["alt_op_p90user_frr_le_1pct"]["session_frr"],
                             s["alpha0.05"]["caught"]))
        partc[f"{rg}|alt_operating_point_p90user_frr_le_1pct"] = dict(
            n_cells_with_a_feasible_cut=len(have), n_cells=len(alt),
            alpha_mean=float(np.mean(have)) if have else None,
            caught_at_alt_op_A0=float(np.mean([r[0] for r in rows])) if rows else None,
            session_frr_at_alt_op_A0=float(np.mean([r[1] for r in rows])) if rows else None,
            caught_at_pooled_5pct_cut_A0=float(np.mean([r[2] for r in rows])) if rows else None,
            caught_cost_of_tail_safe_cut=(
                float(np.mean([r[2] - r[0] for r in rows])) if rows else None))
    out["part_c"] = partc

    # ---- pool shortfall -----------------------------------------------------
    sf = {}
    for rg in ("touch", "keystroke"):
        for r in (1, 2, 5, 10, 20):
            tot_reuse, tot_arms, worst = 0, 0, {}
            for c in sorted(cells):
                for k, a in idx[(c, rg)].items():
                    if k[2] != r:
                        continue
                    tot_reuse += a["reuse_slot_draws_total"]
                    tot_arms += 1
                    for act, v in a["pool_shortfall"].items():
                        worst[act] = v
            sf[f"{rg}|r{r}"] = dict(pool_after_filter=200 // r,
                                    arms=tot_arms,
                                    reuse_slot_draws_total=tot_reuse,
                                    reuse_slot_draws_per_arm=(tot_reuse / tot_arms
                                                              if tot_arms else 0),
                                    shortfall_by_action=worst)
    out["pool_shortfall"] = sf

    json.dump(out, open(os.path.join(RES, "summary_partA.json"), "w"), indent=1,
              sort_keys=True, default=float)
    print("wrote summary_partA.json")


if __name__ == "__main__":
    main()
