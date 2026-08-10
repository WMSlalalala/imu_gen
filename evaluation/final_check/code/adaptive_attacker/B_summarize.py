#!/usr/bin/env python
"""Aggregate the raw per-cell records into the reportable summary.

User-clustered bootstrap (F9): one shared resampled user multiset per replicate,
applied identically to every cell inside a replicate, so macro-aggregates keep the
correlation between cells.
"""
import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
STATS = ["llr_sum", "llr_mean", "B0_count", "B1_meanscore", "B2_durllr"]
TAGS = ["0.05", "0.01", "a90"]
NB = 2000
MODALITIES = ["imu_only", "trajectory_xytime", "imu_trajectory_xytime"]
DETECTORS = ["authconformer", "behaveformer_stdat", "hmog_style_rf",
             "hmog_style_svm", "paper_svm", "paper_xgboost"]

A = json.load(open(os.path.join(OUT, "part_a.json")))
NU = 20
bmat = np.random.default_rng(12345).integers(0, NU, size=(NB, NU))

PU = {}
for m in MODALITIES:
    for d in DETECTORS:
        z = np.load(os.path.join(OUT, f"peruser_{m}__{d}.npz"), allow_pickle=True)
        cfg = {tuple(c): i for i, c in enumerate(z["cfg"].tolist())}
        PU[(m, d)] = dict(z=z, cfg=cfg)


def num_den(combo, regime, rule, sur, r, stat, tag):
    p = PU[combo]
    i = p["cfg"][(regime, rule, str(sur), r)]
    si = STATS.index(stat); ti = TAGS.index(tag)
    return p["z"]["caught"][i, si, ti], p["z"]["den"][i]


def gen_num_den(combo, regime, stat, tag):
    p = PU[combo]["z"]
    si = STATS.index(stat); ti = TAGS.index(tag)
    return p[f"gen_alarm_{regime}"][si, ti], p[f"gen_den_{regime}"]


def macro_boot(pairs):
    """pairs: list of (num_u, den_u).  Returns point estimate + shared-user CI."""
    pt = float(np.mean([n.sum() / d.sum() for n, d in pairs if d.sum() > 0]))
    acc = np.zeros(NB); k = 0
    for n, d in pairs:
        nb = n[bmat].sum(1); db = d[bmat].sum(1)
        ok = db > 0
        v = np.where(ok, nb / np.maximum(db, 1e-12), np.nan)
        acc = acc + np.nan_to_num(v, nan=np.nanmean(v) if np.isfinite(v).any() else 0.0)
        k += 1
    acc /= max(k, 1)
    return pt, [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]


def macro_boot_diff(pairs_c, pairs_g):
    """Paired (caught - session FRR) macro difference with a shared-user bootstrap (F9)."""
    pt = float(np.mean([nc.sum() / dc.sum() - ng.sum() / dg.sum()
                        for (nc, dc), (ng, dg) in zip(pairs_c, pairs_g)]))
    acc = np.zeros(NB)
    for (nc, dc), (ng, dg) in zip(pairs_c, pairs_g):
        a = nc[bmat].sum(1) / np.maximum(dc[bmat].sum(1), 1e-12)
        b = ng[bmat].sum(1) / np.maximum(dg[bmat].sum(1), 1e-12)
        acc += (a - b)
    acc /= len(pairs_c)
    return pt, [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]


def _med(vals):
    v = [x for x in vals if x is not None]
    return dict(median=float(np.median(v)) if v else None,
                n_reached=len(v), n_total=len(vals))


def sel(**kw):
    return [r for r in A if all(r.get(k) == v for k, v in kw.items())]


summary = {}

# ---------------------------------------------------------------- 1. adaptive curve
RULE_GROUPS = {
    "A0_uniform": lambda r: r["rule"] == "A0_uniform",
    "A1_surrogate_same_modality": lambda r: r["rule"] == "A1_surrogate" and r["surrogate_same_modality"],
    "A1_surrogate_all_17": lambda r: r["rule"] == "A1_surrogate",
    "A1_surrogate_cross_modality": lambda r: r["rule"] == "A1_surrogate" and not r["surrogate_same_modality"],
    "A1_ensemble_same_modality5": lambda r: r["rule"] == "A1_ensemble_same_modality5",
    "A1_ensemble_all17": lambda r: r["rule"] == "A1_ensemble_all17",
    "A2_centroid": lambda r: r["rule"] == "A2_centroid",
    "A2_knn": lambda r: r["rule"] == "A2_knn",
    "A2_centroid_reverse": lambda r: r["rule"] == "A2_centroid_reverse",
    "A3_oracle_UPPER_BOUND_not_a_threat_model": lambda r: r["rule"] == "A3_oracle",
}

curve = {}
for reg in ("touch", "keystroke"):
    curve[reg] = {}
    # genuine session FRR (attacker independent)
    for stat in STATS:
        pairs = [gen_num_den((m, d), reg, stat, "0.05") for m in MODALITIES for d in DETECTORS]
        pt, ci = macro_boot(pairs)
        curve[reg].setdefault("_genuine_session_FRR", {})[stat] = dict(macro_mean=pt, ci=ci)
        pairs = [gen_num_den((m, d), reg, stat, "0.01") for m in MODALITIES for d in DETECTORS]
        pt, ci = macro_boot(pairs)
        curve[reg]["_genuine_session_FRR"][stat + "@1%"] = dict(macro_mean=pt, ci=ci)
    for gname, pred in RULE_GROUPS.items():
        curve[reg][gname] = {}
        for r in [1, 2, 5, 10, 20]:
            recs = [x for x in A if x["regime"] == reg and pred(x) and x["r"] == r]
            if not recs:
                continue
            ent = dict(n_cells_or_pairs=len(recs),
                       generated_per_submitted=float(r),
                       per_event_FAR=float(np.mean([x["per_event_FAR_at_dev_frr5"] for x in recs])),
                       per_event_FAR_range=[float(np.min([x["per_event_FAR_at_dev_frr5"] for x in recs])),
                                            float(np.max([x["per_event_FAR_at_dev_frr5"] for x in recs]))],
                       forced_reuse_frac=float(recs[0]["forced_reuse_frac"]))
            for stat in STATS:
                pairs = [num_den((x["modality"], x["detector"]), reg, x["rule"], x["surrogate"],
                                 x["r"], stat, "0.05") for x in recs]
                pt5, ci5 = macro_boot(pairs)
                pairs1 = [num_den((x["modality"], x["detector"]), reg, x["rule"], x["surrogate"],
                                  x["r"], stat, "0.01") for x in recs]
                pt1, ci1 = macro_boot(pairs1)
                per_cell5 = [x["stats"][stat]["0.05"]["caught"] for x in recs]
                per_cell1 = [x["stats"][stat]["0.01"]["caught"] for x in recs]
                frrs = [x["stats"][stat]["0.05"]["session_frr"] for x in recs]
                ent[stat] = dict(
                    caught_at_frr5_macro=pt5, caught_at_frr5_ci=ci5,
                    caught_at_frr1_macro=pt1, caught_at_frr1_ci=ci1,
                    caught_at_frr5_percell=dict(min=float(np.min(per_cell5)),
                                                p25=float(np.percentile(per_cell5, 25)),
                                                median=float(np.median(per_cell5)),
                                                p75=float(np.percentile(per_cell5, 75)),
                                                max=float(np.max(per_cell5))),
                    caught_at_frr1_percell_median=float(np.median(per_cell1)),
                    frac_cells_at_or_below_chance=float(np.mean(
                        [c <= f for c, f in zip(per_cell5, frrs)])),
                    caught_seed_sd_mean=float(np.mean([x["stats"][stat]["caught_seed_sd_at_0.05"]
                                                       for x in recs])),
                    caught_at_a90=float(np.mean([x["stats"][stat]["a90"]["caught"] for x in recs])),
                    price_frr_for_caught_50=_med([x["stats"][stat]["price"]["frr_for_caught_50"]
                                                  for x in recs]),
                    price_frr_for_caught_80=_med([x["stats"][stat]["price"]["frr_for_caught_80"]
                                                  for x in recs]),
                    price_frr_for_caught_95=_med([x["stats"][stat]["price"]["frr_for_caught_95"]
                                                  for x in recs]),
                    caught_oracle_calibration=float(np.mean(
                        [x["stats"][stat]["0.05"]["caught_oracle_calib"] for x in recs])),
                )
                pg5 = [gen_num_den((x["modality"], x["detector"]), reg, stat, "0.05") for x in recs]
                dpt, dci = macro_boot_diff(pairs, pg5)
                ent[stat]["caught_minus_frr_at_5pct"] = dpt
                ent[stat]["caught_minus_frr_at_5pct_ci"] = dci
                ent[stat]["collapsed_macro"] = bool(dci[1] <= 0.0)
                ent[stat]["indistinguishable_from_chance_macro"] = bool(dci[0] <= 0.0 <= dci[1])
            curve[reg][gname][str(r)] = ent
summary["adaptive_curve"] = curve

# ---------------------------------------------------------------- 2. collapse point
collapse = {}
for reg in ("touch", "keystroke"):
    collapse[reg] = {}
    for gname, pred in RULE_GROUPS.items():
        for stat in ("llr_sum", "B0_count"):
            per_unit = {}
            for x in A:
                if x["regime"] != reg or not pred(x):
                    continue
                key = (x["modality"], x["detector"], str(x["surrogate"]))
                c = x["stats"][stat]["0.05"]["caught"]
                f = x["stats"][stat]["0.05"]["session_frr"]
                per_unit.setdefault(key, {})[x["r"]] = (c, f)
            first = []
            for key, d in per_unit.items():
                rr = None
                for r in [1, 2, 5, 10, 20]:
                    if r in d and d[r][0] <= d[r][1]:
                        rr = r; break
                first.append(rr)
            got = [r for r in first if r is not None]
            collapse[reg].setdefault(gname, {})[stat] = dict(
                n_units=len(first), n_collapsed=len(got),
                frac_collapsed=float(len(got) / max(len(first), 1)),
                smallest_r_collapse_median=float(np.median(got)) if got else None,
                smallest_r_collapse_min=int(np.min(got)) if got else None,
                collapsed_at_r_le_10=float(np.mean([r is not None and r <= 10 for r in first])),
                macro_caught_vs_frr_at_r20=None,
            )
summary["collapse_point"] = collapse

# ---------------------------------------------------------------- 3. surrogate spread
spread = {}
for reg in ("touch", "keystroke"):
    spread[reg] = {}
    for r in [2, 5, 10, 20]:
        rows = [x for x in A if x["regime"] == reg and x["rule"] == "A1_surrogate" and x["r"] == r]
        by_victim = {}
        for x in rows:
            by_victim.setdefault((x["modality"], x["detector"]), []).append(x)
        v_stats = []
        for v, xs in by_victim.items():
            c = [y["stats"]["llr_sum"]["0.05"]["caught"] for y in xs]
            cs = [y["stats"]["llr_sum"]["0.05"]["caught"] for y in xs if y["surrogate_same_modality"]]
            a0 = [y for y in A if y["regime"] == reg and y["rule"] == "A0_uniform"
                  and y["modality"] == v[0] and y["detector"] == v[1]][0]
            v_stats.append(dict(victim=f"{v[0]}|{v[1]}", A0_caught=a0["stats"]["llr_sum"]["0.05"]["caught"],
                                across_17_surrogates=dict(min=float(np.min(c)), median=float(np.median(c)),
                                                          max=float(np.max(c))),
                                across_5_same_modality=dict(min=float(np.min(cs)), median=float(np.median(cs)),
                                                            max=float(np.max(cs))),
                                best_surrogate=min(xs, key=lambda y: y["stats"]["llr_sum"]["0.05"]["caught"])["surrogate"],
                                worst_surrogate=max(xs, key=lambda y: y["stats"]["llr_sum"]["0.05"]["caught"])["surrogate"]))
        # which surrogate family is best on average
        by_sur = {}
        for x in rows:
            by_sur.setdefault(x["surrogate"], []).append(x["stats"]["llr_sum"]["0.05"]["caught"])
        spread[reg][str(r)] = dict(
            per_victim=v_stats,
            mean_caught_by_surrogate={k: float(np.mean(v)) for k, v in sorted(by_sur.items())},
            best_case_attacker_macro=float(np.mean([v["across_17_surrogates"]["min"] for v in v_stats])),
            worst_case_attacker_macro=float(np.mean([v["across_17_surrogates"]["max"] for v in v_stats])),
            median_surrogate_macro=float(np.mean([v["across_17_surrogates"]["median"] for v in v_stats])),
        )
summary["surrogate_spread"] = spread

# ---------------------------------------------------------------- 4. baselines table
base = {}
for reg in ("touch", "keystroke"):
    base[reg] = {}
    for gname, pred in RULE_GROUPS.items():
        for r in [1, 2, 5, 10, 20]:
            recs = [x for x in A if x["regime"] == reg and pred(x) and x["r"] == r]
            if not recs:
                continue
            row = {}
            for stat in STATS:
                row[stat] = dict(
                    caught_frr5=float(np.mean([x["stats"][stat]["0.05"]["caught"] for x in recs])),
                    caught_frr1=float(np.mean([x["stats"][stat]["0.01"]["caught"] for x in recs])),
                    frr5_realised=float(np.mean([x["stats"][stat]["0.05"]["session_frr"] for x in recs])),
                )
            base[reg].setdefault(gname, {})[str(r)] = row
summary["baselines"] = base

# ---------------------------------------------------------------- 5. reviewer's cell
rev = {}
for reg in ("touch", "keystroke"):
    rows = [x for x in A if x["regime"] == reg and x["modality"] == "imu_only"
            and x["detector"] == "behaveformer_stdat"]
    rev[reg] = {}
    for x in rows:
        k = f"{x['rule']}|{x['surrogate']}|r{x['r']}"
        rev[reg][k] = {st: dict(caught5=x["stats"][st]["0.05"]["caught"],
                                caught5_ci=x["stats"][st]["0.05"]["caught_ci"],
                                caught1=x["stats"][st]["0.01"]["caught"],
                                frr5=x["stats"][st]["0.05"]["session_frr"])
                       for st in ("llr_sum", "llr_mean", "B0_count", "B2_durllr")}
        rev[reg][k]["per_event_FAR"] = x["per_event_FAR_at_dev_frr5"]
summary["reviewer_cell_imu_only_behaveformer_stdat"] = rev

# ---------------------------------------------------------------- 6. part C roll-up
C = json.load(open(os.path.join(OUT, "part_c.json")))
pc = {}
for reg in ("touch", "keystroke"):
    pc[reg] = {}
    for stat in STATS:
        rows = [x for x in C if x["regime"] == reg and x["statistic"] == stat]
        d = {}
        for tag in TAGS:
            d[tag] = dict(
                alpha_median=float(np.median([x[tag]["alpha"] for x in rows])),
                pooled_frr=float(np.mean([x[tag]["pooled_frr"] for x in rows])),
                pooled_frr_oracle_calibration=float(np.mean([x[tag]["pooled_frr_oracle"] for x in rows])),
                per_user_min=float(np.mean([x[tag]["per_user_frr"]["min"] for x in rows])),
                per_user_median=float(np.mean([x[tag]["per_user_frr"]["median"] for x in rows])),
                per_user_p90=float(np.mean([x[tag]["per_user_frr"]["p90"] for x in rows])),
                per_user_max=float(np.mean([x[tag]["per_user_frr"]["max"] for x in rows])),
                per_user_max_over_cells=float(np.max([x[tag]["per_user_frr"]["max"] for x in rows])),
                betabinom_rho=float(np.mean([x[tag]["betabinom"]["rho"] for x in rows
                                             if x[tag]["betabinom"]["rho"] is not None]))
                if any(x[tag]["betabinom"]["rho"] is not None for x in rows) else None,
                betabinom_p50=float(np.mean([x[tag]["betabinom_p50"] for x in rows
                                             if x[tag]["betabinom_p50"] is not None]))
                if any(x[tag]["betabinom_p50"] is not None for x in rows) else None,
                betabinom_p90=float(np.mean([x[tag]["betabinom_p90"] for x in rows
                                             if x[tag]["betabinom_p90"] is not None]))
                if any(x[tag]["betabinom_p90"] is not None for x in rows) else None,
                lockout_days_fitted_median_user=float(np.mean(
                    [x[tag]["lockout_days_fitted_median_user"] for x in rows
                     if x[tag]["lockout_days_fitted_median_user"] is not None]))
                if any(x[tag]["lockout_days_fitted_median_user"] is not None for x in rows) else None,
                lockout_days_fitted_p90_user=float(np.mean(
                    [x[tag]["lockout_days_fitted_p90_user"] for x in rows
                     if x[tag]["lockout_days_fitted_p90_user"] is not None]))
                if any(x[tag]["lockout_days_fitted_p90_user"] is not None for x in rows) else None,
                sessions_per_day=rows[0]["sessions_per_day"],
            )
        # cost of moving to the p90-user operating point, under A0
        a0 = [x for x in A if x["regime"] == reg and x["rule"] == "A0_uniform"]
        d["cost_of_p90_user_operating_point_under_A0"] = dict(
            caught_at_pooled_5pct=float(np.mean([x["stats"][stat]["0.05"]["caught"] for x in a0])),
            caught_at_a90=float(np.mean([x["stats"][stat]["a90"]["caught"] for x in a0])),
        )
        pc[reg][stat] = d
summary["per_user_price"] = pc

summary["part_b"] = json.load(open(os.path.join(OUT, "part_b.json")))
summary["coverage"] = json.load(open(os.path.join(OUT, "coverage.json")))
summary["far_frr5_check_mean"] = json.load(open(os.path.join(OUT, "far_frr5_check.json")))["mean"]
summary["preregistration"] = json.load(open(os.path.join(ROOT, "preregistration.json")))

json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print("summary written", os.path.join(OUT, "summary.json"))
