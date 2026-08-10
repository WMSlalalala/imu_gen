#!/usr/bin/env python3
"""User-clustered bootstrap for FAR/FRR at the FRR=5% operating point.

WHY THIS EXISTS
----------------
Every cell already ships a `bootstrap_95ci` block, resampled by user cluster with
10,000 replicates -- but it brackets `primary_metrics.far`, which sits at the
*development-selected EER threshold*.  The paper's headline number is FAR at the
FRR=5% threshold, and that one had no interval at all.  This fills that gap from
the scores already on disk; nothing is retrained and no threshold is re-selected.

WHY THE RESAMPLING UNIT IS THE USER
------------------------------------
A test split of 6,000 events is not 6,000 independent observations.  The 300
events belonging to one person share that person's habits, their handset, their
grip, and a handful of sessions.  Treating events as independent would shrink the
interval by roughly the square root of the cluster size and claim a precision the
data does not support.  The independent units here are the 20 test users, so the
resample draws 20 users with replacement and keeps every event of each one.

WHY ONE RESAMPLE IS SHARED ACROSS CELLS
----------------------------------------
The 90 cells are scored on the *same* 20 people.  If each cell drew its own users,
the per-cell intervals would be right but their average would behave as if the
cells were independent, and the interval on the 90-cell mean would come out far
too narrow.  So each replicate draws one user multiset and applies it everywhere,
which propagates user-level variation into the aggregate the way it actually
flows.

WHAT THIS IS NOT
-----------------
Not a k-fold or repeated-split experiment.  The user split is fixed
(`users_seed42.json`, 70/10/20, user-disjoint) and this resamples *within* the
one test split; it quantifies "how much would this number move on 20 other people
drawn from the same pool", not "how much would it move under a different
train/dev/test partition".  The latter needs retraining and is not done here.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

# WORKING_RESULTS is taken from final_release rather than hard-coded, so this
# reads exactly the cells reference_scores() reads and the point estimates here
# cannot drift from the published table.
from final_release import (MODALITIES, WORKING_RESULTS,  # noqa: E402
                           cell_dir, cell_sources)


def load_cell(cell: Path):
    """(cut, {user -> (fake scores, genuine scores)}) or None."""
    thresholds_path = cell / "thresholds.json"
    scores_path = cell / "test_scores.jsonl"
    if not scores_path.is_file():
        scores_path = cell / "test_scores.jsonl.gz"
    if not thresholds_path.is_file() or not scores_path.is_file():
        return None
    thresholds = json.loads(thresholds_path.read_text())
    if thresholds["score_direction"] != "larger_is_more_fake":
        raise SystemExit(f"unexpected score direction in {cell}")
    cut = float(thresholds["frr5"])
    per_user = defaultdict(lambda: ([], []))
    opener = gzip.open if scores_path.suffix == ".gz" else open
    with opener(scores_path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            fake, genuine = per_user[row["user_id"]]
            (fake if int(row["label"]) == 1 else genuine).append(
                float(row["fake_high_score"]))
    return cut, {u: (np.asarray(f), np.asarray(g)) for u, (f, g) in per_user.items()}


def load_release() -> dict:
    cells = {}
    for key, directory in sorted(cell_sources().items()):
        action, modality, detector = key
        cell = cell_dir(action, modality, detector, directory)
        loaded = load_cell(cell)
        if loaded is None:
            raise SystemExit(
                f"released cell {action}__{modality}__{detector} at {cell} has "
                f"no readable scores. Refusing to continue: skipping it would "
                f"shrink every interval below without saying so.")
        cells[key] = loaded
    return cells


def load_method(method: str) -> dict:
    """A baseline or ablation arm's cells, located exactly as summarise_final does.

    Going through `collect` first means this honours the same two rules the FAR
    table honours -- only the modalities the build says it changed, and no cells
    for actions the method declined -- so a bootstrap here can never quietly
    cover more ground than the number it is bracketing.
    """
    from summarise_final import ROOT, collect
    from final_release import action_to_bundle
    scores, _missing, _declined = collect(method, "far5")
    owner = action_to_bundle()
    cells = {}
    for key in scores:
        action, modality, detector = key
        cell = (ROOT / method / f"cells_{owner[action]}_{modality}"
                / "cells" / f"{action}__{modality}__{detector}")
        loaded = load_cell(cell)
        if loaded is None:
            raise SystemExit(
                f"{method} declares {action}__{modality}__{detector} in its "
                f"bundle manifest but {cell} has no readable scores. Refusing "
                f"to continue: a paired comparison that quietly drops cells "
                f"reports a different quantity from the one it names.")
        cells[key] = loaded
    return cells


def eer_on(fake: np.ndarray, genuine: np.ndarray) -> float:
    """The equal-error point located on whatever sample is handed in.

    Deliberately re-located per replicate rather than held at the full-sample
    cut: the test EER *is* a cut chosen from the data, so an interval that froze
    that cut would describe a different quantity from the one reported.
    """
    if len(fake) == 0 or len(genuine) == 0:
        return float("nan")
    best = None
    for cut in np.unique(np.concatenate([fake, genuine])):
        far = float((fake < cut).mean())
        frr = float((genuine >= cut).mean())
        gap = abs(far - frr)
        if best is None or gap < best[0]:
            best = (gap, (far + frr) / 2.0)
    return best[1]


def paired_eer(cells: dict, names: list) -> float:
    """Mean test EER over a cell set for one resampled user multiset."""
    out = []
    for _cut, per_user in cells.values():
        fake = np.concatenate([per_user[n][0] for n in names])
        genuine = np.concatenate([per_user[n][1] for n in names])
        out.append(eer_on(fake, genuine))
    return float(np.mean(out))


def paired_far(cells: dict, names: list) -> float:
    """Mean FAR over a cell set for one resampled user multiset."""
    out = []
    for cut, per_user in cells.values():
        fake = np.concatenate([per_user[n][0] for n in names])
        out.append(float((fake < cut).mean()))
    return float(np.mean(out))


def compare(args) -> None:
    """Bootstrap the release against each named method on the cells they share.

    The difference is resampled jointly -- the same user multiset scores both
    sides in every replicate -- so the interval is on the *paired* difference and
    not on two independent means, which would be wider and would ignore that both
    numbers move together when the people change.
    """
    release = load_release()
    rng = np.random.default_rng(args.seed)
    rows = []
    for method in args.compare:
        theirs = load_method(method)
        shared = sorted(set(release) & set(theirs))
        if not shared:
            rows.append((method, 0, None, None, None, None, None))
            continue
        users = sorted(next(iter({frozenset(pu) for _c, pu in theirs.values()})))
        ours_c = {k: release[k] for k in shared}
        thrs_c = {k: theirs[k] for k in shared}
        diffs, ours_s, thrs_s = [], [], []
        for _ in range(args.replicates):
            names = [users[i] for i in rng.choice(len(users), size=len(users), replace=True)]
            a, b = paired_far(ours_c, names), paired_far(thrs_c, names)
            ours_s.append(a); thrs_s.append(b); diffs.append(a - b)
        pt_o = paired_far(ours_c, users)
        pt_t = paired_far(thrs_c, users)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append((method, len(shared), pt_o, pt_t, pt_o - pt_t, float(lo), float(hi)))

    lines = ["# Paired user-clustered bootstrap: the release against each method", "",
             f"{args.replicates} replicates, seed {args.seed}, resampled by test user. "
             "Each replicate scores both sides on the same drawn people, so the "
             "interval is on the paired difference. Only the cells the two share are "
             "used, so the comparison is like for like.", "",
             "| Method | Shared cells | Ours | Theirs | Difference | 95% CI | Excludes 0 |",
             "|---|---|---|---|---|---|---|"]
    for method, n, o, t, d, lo, hi in rows:
        if o is None:
            lines.append(f"| {method} | 0 | - | - | - | - | - |")
            continue
        excl = "yes" if (lo > 0 or hi < 0) else "**no**"
        lines.append(f"| {method} | {n} | {o:.3f} | {t:.3f} | {d:+.3f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] | {excl} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--metric", default="far5", choices=["far5", "test_eer"],
                        help="far5 brackets FAR at the frozen FRR=5% cut; test_eer\n                             re-locates the equal-error point in every replicate")
    parser.add_argument("--method", default=None,
                        help="a baseline or ablation arm under final/ instead of the "
                             "release; the cells are located the same way "
                             "summarise_final does, so the same modality and "
                             "declined-action rules apply")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="methods to bootstrap alongside the release on the "
                             "cells they share, and report the paired difference")
    args = parser.parse_args()

    if args.compare:
        return compare(args)

    cells = load_release() if args.method is None else load_method(args.method)
    if not cells:
        raise SystemExit("no cells loaded")

    # Every cell must be scored on the same people, or a shared resample is
    # meaningless.  Checked rather than assumed.
    user_sets = {frozenset(per_user) for _cut, per_user in cells.values()}
    if len(user_sets) != 1:
        raise SystemExit(f"cells do not share one user set: {len(user_sets)} distinct sets")
    users = sorted(next(iter(user_sets)))
    print(f"{len(cells)} cells, {len(users)} test users, "
          f"{args.replicates} replicates, seed {args.seed}", file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    per_cell = {key: [] for key in cells}
    aggregate, aggregate_frr = [], []
    by_modality = {m: [] for m in MODALITIES}

    for _ in range(args.replicates):
        drawn = rng.choice(len(users), size=len(users), replace=True)
        names = [users[i] for i in drawn]
        fars, frrs = {}, {}
        for key, (cut, per_user) in cells.items():
            fake = np.concatenate([per_user[n][0] for n in names])
            genuine = np.concatenate([per_user[n][1] for n in names])
            if args.metric == "test_eer":
                fars[key] = eer_on(fake, genuine)
                frrs[key] = fars[key]      # at the equal-error point the two coincide
            else:
                fars[key] = float((fake < cut).mean())
                frrs[key] = float((genuine >= cut).mean())
            per_cell[key].append(fars[key])
        aggregate.append(float(np.mean(list(fars.values()))))
        aggregate_frr.append(float(np.mean(list(frrs.values()))))
        for m in MODALITIES:
            v = [x for k, x in fars.items() if k[1] == m]
            if v:
                by_modality[m].append(float(np.mean(v)))

    def band(samples):
        a = np.asarray(samples)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    lines = ["# User-clustered bootstrap at the FRR=5% operating point\n",
             f"Resampling unit: the {len(users)} test users, drawn with replacement, "
             f"all events of each kept. {args.replicates} replicates, seed {args.seed}. "
             "One user multiset per replicate, shared across all cells (see the "
             "module docstring). Thresholds are the frozen `frr5` cuts; nothing is "
             "re-selected.\n",
             "## Aggregate\n",
             "| Quantity | Point | 95% CI |", "|---|---|---|"]
    lo, hi = band(aggregate)
    point = float(np.mean([  # the point estimate is the unresampled value
        (np.concatenate([per_user[u][0] for u in users]) < cut).mean()
        for cut, per_user in cells.values()]))
    lines.append(f"| FAR, all {len(cells)} cells | {point:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lo_f, hi_f = band(aggregate_frr)
    point_frr = float(np.mean([
        (np.concatenate([per_user[u][1] for u in users]) >= cut).mean()
        for cut, per_user in cells.values()]))
    lines.append(f"| FRR, all {len(cells)} cells | {point_frr:.3f} | [{lo_f:.3f}, {hi_f:.3f}] |")
    lines.append("")
    lines += ["## By modality\n", "| Modality | Cells | Point FAR | 95% CI | CI width |",
              "|---|---|---|---|---|"]
    for m in MODALITIES:
        if not by_modality[m]:
            continue
        keys = [k for k in cells if k[1] == m]
        pt = float(np.mean([
            (np.concatenate([cells[k][1][u][0] for u in users]) < cells[k][0]).mean()
            for k in keys]))
        lo, hi = band(by_modality[m])
        lines.append(f"| {m} | {len(keys)} | {pt:.3f} | [{lo:.3f}, {hi:.3f}] | {hi - lo:.3f} |")
    lines.append("")

    widths = []
    lines += ["## Per cell\n", "| Action | Modality | Detector | Point FAR | 95% CI |",
              "|---|---|---|---|---|"]
    for key in sorted(cells):
        cut, per_user = cells[key]
        pt = float((np.concatenate([per_user[u][0] for u in users]) < cut).mean())
        lo, hi = band(per_cell[key])
        widths.append(hi - lo)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {pt:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines.insert(len(lines) - len(cells) - 2,
                 f"Per-cell intervals are much wider than the aggregate -- median width "
                 f"{np.median(widths):.3f} against {hi_f - lo_f:.3f} for the 90-cell FRR "
                 f"and the aggregate FAR band above -- because averaging over 90 cells "
                 f"cancels a good deal of the per-cell noise while the user-level "
                 f"variation, being shared, does not cancel.\n")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
