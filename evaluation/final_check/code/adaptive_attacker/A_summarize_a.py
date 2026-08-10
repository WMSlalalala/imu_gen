#!/usr/bin/env python
"""Condense results_final/part_a.json + part_b.json into readable tables."""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_final")
A = json.load(open(os.path.join(OUT, "part_a.json")))
try:
    B = json.load(open(os.path.join(OUT, "part_b.json")))
except FileNotFoundError:
    B = None

CELLS = list(A["cells"].keys())
REG = ["touch", "keystroke"]
LINES = []


def p(s=""):
    print(s)
    LINES.append(s)


def rule_of(label):
    if label.startswith("A0"):
        return "A0_uniform", 1
    m = re.search(r"_r(\d+)$", label.replace("_SANITY_NOT_HEADLINE", ""))
    r = int(m.group(1)) if m else 1
    if label.startswith("A1_same-mod"):
        return "A1_same_modality_surrogate", r
    if label.startswith("A1_cross-mod"):
        return "A1_cross_modality_surrogate", r
    if label.startswith("A2rev"):
        return "A2rev_atypical_SANITY", r
    if label.startswith("A2"):
        return "A2_self_consistency", r
    if label.startswith("A3llr"):
        return "A3llr_ORACLE", r
    if label.startswith("A3"):
        return "A3_ORACLE", r
    return label, r


def collect(regime, stat, cal, target):
    """rows: (rule, r) -> list over (cell, surrogate) of dict."""
    rows = {}
    for cn, c in A["cells"].items():
        rr = c["regimes"][regime]
        if "configs" not in rr:
            continue
        for lab, blk in rr["configs"].items():
            k = rule_of(lab)
            d = blk[stat][target][cal]
            rows.setdefault(k, []).append(dict(
                cell=cn, label=lab, caught=d["caught"], frr=d["session_frr"],
                lo=d["caught_ci"][0], hi=d["caught_ci"][1],
                sd=d["caught_per_seed_sd"],
                far=blk["per_event_far"]["_mean_over_actions"]))
    return rows


def q(v):
    v = np.array(v, float)
    return np.median(v), np.percentile(v, 25), np.percentile(v, 75), v.min(), v.max()


ORDER = ["A0_uniform", "A1_same_modality_surrogate", "A1_cross_modality_surrogate",
         "A2_self_consistency", "A2rev_atypical_SANITY", "A3_ORACLE", "A3llr_ORACLE"]

p("=" * 118)
p("DENOMINATORS AND COVERAGE (fairness rule F11 / F15)")
p("=" * 118)
cov = json.load(open(os.path.join(OUT, "coverage.json")))
for k, v in cov.items():
    p("  %-42s %s" % (k, v))

for regime in REG:
    any_cell = A["cells"][CELLS[0]]["regimes"][regime]
    p("  %-42s %s" % (regime + " sessions", any_cell.get("n_sessions")))
    p("  %-42s %s" % (regime + " scored slots / session slots",
                      "%s / %s" % (any_cell.get("scored_slots"), any_cell.get("session_slots"))))
    p("  %-42s %s" % (regime + " scored events per session",
                      any_cell.get("nscored_per_session")))

for target in ["0.05", "0.01"]:
    for cal in ["split", "oracle"]:
        for regime in REG:
            p()
            p("=" * 118)
            p("PART A  regime=%s  statistic=session score-LLR (sum)  calibration=%s  "
              "target session FRR=%s" % (regime, cal, target))
            p("=" * 118)
            rows = collect(regime, "llr", cal, target)
            p("%-34s %3s | %-32s | %-9s | %-7s | %s"
              % ("selection rule", "r", "session caught  med [IQR] (min,max)",
                 "realizedFRR", "evFAR", "cells with caught<FRR"))
            for rl in ORDER:
                for r in (1, 2, 5, 10, 20):
                    if (rl, r) not in rows:
                        continue
                    v = rows[(rl, r)]
                    m, lo, hi, mn, mx = q([x["caught"] for x in v])
                    frr = np.median([x["frr"] for x in v])
                    far = np.median([x["far"] for x in v])
                    below = np.mean([x["caught"] < x["frr"] for x in v])
                    p("%-34s %3d | %.3f [%.3f,%.3f] (%.3f,%.3f) | %9.3f | %7.3f | %.0f%% of %d"
                      % (rl, r, m, lo, hi, mn, mx, frr, far, 100 * below, len(v)))

p()
p("=" * 118)
p("PART A  MANDATORY BASELINES at the SAME operating point (F10), selection = A0 uniform")
p("=" * 118)
for regime in REG:
    for target in ["0.05", "0.01"]:
        line = "%-10s target=%s |" % (regime, target)
        for stat, nm in (("llr", "scoreLLR"), ("b0", "B0 count"), ("b1", "B1 meanZ"),
                         ("b2", "B2 durLLR")):
            v = [A["cells"][c]["regimes"][regime]["configs"]["A0_uniform"][stat][target]["split"]
                 for c in CELLS]
            line += "  %s caught %.3f (frr %.3f)" % (
                nm, np.median([x["caught"] for x in v]), np.median([x["session_frr"] for x in v]))
        p(line)

p()
p("=" * 118)
p("PART A  BASELINES UNDER SELECTION (A2 r=10, A1 same-mod r=10) -- split calibration, 5%")
p("=" * 118)
for regime in REG:
    for rl, r in (("A2_self_consistency", 10), ("A1_same_modality_surrogate", 10),
                  ("A3llr_ORACLE", 10)):
        line = "%-10s %-30s r=%2d |" % (regime, rl, r)
        for stat, nm in (("llr", "scoreLLR"), ("b0", "B0"), ("b1", "B1"), ("b2", "B2")):
            rows = collect(regime, stat, "split", "0.05")
            v = rows.get((rl, r), [])
            if v:
                line += "  %s %.3f" % (nm, np.median([x["caught"] for x in v]))
        p(line)

p()
p("=" * 118)
p("PART A  PRICE CURVE (F12): session false-alarm rate needed to reach caught 50/80/95%")
p("=" * 118)
for regime in REG:
    for lab in ["A0_uniform", "A1_same-mod_paper_xgboost_r10", "A2_selfconsistency_r10",
                "A3llr_ORACLE_NOT_A_THREAT_MODEL_r10"]:
        vals = {k: [] for k in (50, 80, 95)}
        for c in CELLS:
            pc = A["cells"][c]["regimes"][regime]["configs"][lab]["llr"]["price_curve"]["oracle"]
            for k in (50, 80, 95):
                x = pc["frr_for_caught_%d" % k]
                vals[k].append(np.nan if x is None else x)
        p("%-10s %-42s  FRR@caught50 %s  @80 %s  @95 %s" % (
            regime, lab,
            "%.3f" % np.nanmedian(vals[50]) if np.any(~np.isnan(vals[50])) else "  n/a",
            "%.3f" % np.nanmedian(vals[80]) if np.any(~np.isnan(vals[80])) else "  n/a",
            "%.3f" % np.nanmedian(vals[95]) if np.any(~np.isnan(vals[95])) else "  n/a"))

p()
p("=" * 118)
p("PART A  A1 SURROGATE SPREAD -- caught per surrogate family, median over 18 victim cells")
p("=" * 118)
for regime in REG:
    for r in (2, 5, 10, 20):
        per = {}
        for c in CELLS:
            for lab, blk in A["cells"][c]["regimes"][regime]["configs"].items():
                if not lab.startswith("A1") or not lab.endswith("_r%d" % r):
                    continue
                fam = lab.split("_")[1] + ":" + "_".join(lab.split("_")[2:-1])
                per.setdefault(fam, []).append(blk["llr"]["0.05"]["split"]["caught"])
        if not per:
            continue
        s = " ".join("%s=%.3f" % (k.replace("same-mod:", "S/").replace("cross-mod", "X"),
                                  np.median(v)) for k, v in sorted(per.items()))
        med = [np.median(v) for v in per.values()]
        p("%-10s r=%2d  spread over %d surrogate families: min %.3f max %.3f range %.3f"
          % (regime, r, len(per), min(med), max(med), max(med) - min(med)))
        p("            %s" % s)

p()
p("=" * 118)
p("PART A  COLLAPSE POINT: smallest r at which caught < realized session FRR (split, 5%)")
p("=" * 118)
for regime in REG:
    for rl in ORDER:
        cps = []
        for c in CELLS:
            best = None
            for r in (2, 5, 10, 20):
                rows = collect(regime, "llr", "split", "0.05").get((rl, r), [])
                v = [x for x in rows if x["cell"] == c]
                if v and np.median([x["caught"] < x["frr"] for x in v]) >= 0.5:
                    best = r
                    break
            cps.append(best)
        n = sum(x is not None for x in cps)
        p("%-10s %-32s collapses in %2d/%2d cells; smallest r (median over collapsing cells)=%s"
          % (regime, rl, n, len(cps),
             "%.0f" % np.median([x for x in cps if x is not None]) if n else "never"))

p()
p("=" * 118)
p("PART A  ATTACKER COST AND POOL SUFFICIENCY (F15)")
p("=" * 118)
for regime in REG:
    for r in (1, 2, 5, 10, 20):
        lab = "A0_uniform" if r == 1 else "A2_selfconsistency_r%d" % r
        if lab not in A["cells"][CELLS[0]]["regimes"][regime]["configs"]:
            continue
        ac = A["cells"][CELLS[0]]["regimes"][regime]["configs"][lab]["attacker_cost"]
        p("%-10s r=%2d  generated/submitted=%2d  pool 200 -> %3d submitted; "
          "max one-action count in a session=%d; slots short %d/%d, sessions affected %d, "
          "events drawn with replacement %d"
          % (regime, r, ac["generated_per_submitted"], ac["submitted_pool_per_user_action"],
             ac["max_events_of_one_action_in_a_session"],
             ac["shortfall"]["session_action_slots_needing_more_than_pool"],
             ac["shortfall"]["session_action_slots_total"],
             ac["shortfall"]["sessions_affected"],
             ac["shortfall"]["events_drawn_with_replacement_over_all_seeds"]))

p()
p("=" * 118)
p("PART C  WHO PAYS -- per-user session FRR at the pooled 5% cut (split calibration)")
p("=" * 118)
for regime in REG:
    for stat in ("llr", "b0", "b2"):
        mn, md, p9, mx, bb = [], [], [], [], []
        for c in CELLS:
            d = A["cells"][c]["regimes"][regime]["configs"]["A0_uniform"][stat]["0.05"]["split"]["per_user_frr"]
            mn.append(d["min"]); md.append(d["median"]); p9.append(d["p90"]); mx.append(d["max"])
            if d["betabinom"].get("p90") is not None:
                bb.append(d["betabinom"]["p90"])
        p("%-10s %-5s per-user FRR  min %.4f  median %.4f  p90 %.4f  max %.4f   "
          "(ratio max/min %s)  betabinom p90 %.4f"
          % (regime, stat, np.median(mn), np.median(md), np.median(p9), np.median(mx),
             "%.0fx" % (np.median(mx) / np.median(mn)) if np.median(mn) > 0 else "inf",
             np.median(bb) if bb else float("nan")))
    d = A["cells"][CELLS[0]]["regimes"][regime]["configs"]["A0_uniform"]["llr"]["0.05"]["split"]["per_user_frr"]
    p("           lockout cadence at this cut (%s sessions/day assumed): median user 1 false "
      "lockout every %s days; p90 user every %s days"
      % (8, _f(d["days_between_false_lockouts_median_user"]),
         _f(d["days_between_false_lockouts_p90_user"])))

p()
p("=" * 118)
p("PART C  ALTERNATIVE OPERATING POINT: cut with p90-user session FRR <= 1%")
p("=" * 118)
for regime in REG:
    for lab in ("A0_uniform", "A2_selfconsistency_r10", "A3llr_ORACLE_NOT_A_THREAT_MODEL_r10"):
        a, b, fa, fb = [], [], [], []
        for c in CELLS:
            x = A["cells"][c]["regimes"][regime]["configs"][lab]["llr"]
            a.append(x["0.05"]["split"]["caught"]); fa.append(x["0.05"]["split"]["session_frr"])
            b.append(x["p90user_frr_le_1pct"]["split"]["caught"])
            fb.append(x["p90user_frr_le_1pct"]["split"]["session_frr"])
        p("%-10s %-42s pooled5%% cut: caught %.3f (frr %.3f)  ->  p90user<=1%% cut: "
          "caught %.3f (frr %.3f)   cost %.3f"
          % (regime, lab, np.median(a), np.median(fa), np.median(b), np.median(fb),
             np.median(a) - np.median(b)))


def _f(x):
    return "inf" if x is None else "%.1f" % x


if B is not None:
    p()
    p("=" * 118)
    p("PART B  DURATION CONFOUND -- duration-only session LLR (development-user fit)")
    p("=" * 118)
    for var, blk in B["duration_only_session_llr"].items():
        for regime in REG:
            if regime not in blk:
                continue
            for t in ("0.05", "0.01"):
                s = blk[regime][t]["split"]; o = blk[regime][t]["oracle"]
                p("%-16s %-10s target %s | split caught %.3f (frr %.3f) | oracle caught %.3f "
                  "(frr %.3f)" % (var, regime, t, s["caught"], s["session_frr"],
                                  o["caught"], o["session_frr"]))
    p()
    p("PART B  per-action duration bulk statistics")
    for tag, blk in B["bulk_stats"].items():
        p("  --- %s ---" % tag)
        p("  %-10s %-38s %-38s %8s %8s" % ("action", "genuine p5/p50/p95/max",
                                           "fake p5/p50/p95/max", "medshift", "on-cap"))
        for a, d in blk.items():
            p("  %-10s %-38s %-38s %8.3f %8.4f" % (
                a,
                "%.3f/%.3f/%.3f/%.3f" % (d["genuine"]["p5"], d["genuine"]["p50"],
                                         d["genuine"]["p95"], d["genuine"]["max"]),
                "%.3f/%.3f/%.3f/%.3f" % (d["fake"]["p5"], d["fake"]["p50"],
                                         d["fake"]["p95"], d["fake"]["max"]),
                d["median_shift_s"], d["frac_fake_exactly_on_cap"]))
    p()
    p("PART B  timing consequence (onset = onset + duration + gap)")
    for k, v in B["timing_consequence"]["duration_share"].items():
        p("  %-24s caught(real dur) %.3f -> caught(median dur) %.3f   lost %.3f  (%s of caught)"
          % (k, v["caught_real_durations"], v["caught_median_durations"],
             v["caught_lost_to_duration"],
             "n/a" if v["fraction_of_caught_attributable_to_duration"] is None
             else "%.0f%%" % (100 * v["fraction_of_caught_attributable_to_duration"])))

open(os.path.join(OUT, "SUMMARY_A.txt"), "w").write("\n".join(LINES) + "\n")
print("\nwritten", os.path.join(OUT, "SUMMARY_A.txt"))
