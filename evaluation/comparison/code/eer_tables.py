#!/usr/bin/env python3
"""The same comparison and ablation, scored at the test-set equal-error rate.

WHY A SECOND TABLE AT ALL
--------------------------
The paper's primary criterion is FAR at a FRR=5% threshold that was selected on
the development split.  That is what a deployed detector does: the vendor fixes
how many genuine users it will tolerate rejecting, then lives with whatever gets
through.  The test set is never consulted when the cut is chosen.

Much of the behavioural-biometrics literature reports EER instead, and reports it
on the test split -- the cut is moved until the two error rates meet, on the same
data the number is quoted from.  That is a weaker protocol, but it is the one
readers will want to compare against, so both are published side by side.

WHAT THE TWO NUMBERS MEAN, AND WHY THEY DISAGREE
-------------------------------------------------
FAR at FRR=5% asks: at a fixed, realistic tolerance for rejecting genuine users,
how much fake traffic is let through?  Test EER asks: what is the best balance a
detector could strike if it were allowed to tune the cut against this very data?

The second question flatters the detector, so every attack looks weaker under it.
Direction is unchanged -- higher is still a stronger attack in both -- but the
scales are not comparable and a number from one table must never be quoted
against a number from the other.

`descriptive_test_eer` in each cell's `summary.json` is the same quantity; this
script recomputes it from the raw scores so every method goes through one code
path, and `--check` verifies the two agree.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from covered_modalities import covered  # noqa: E402
from final_release import (ACTIONS, DETECTORS, MODALITIES,  # noqa: E402
                           reference_scores, reference_scores_eer)
from summarise_final import ROOT, collect, read_cell  # noqa: E402

SHORT = {"hmog_style_svm": "HMOG-SVM", "hmog_style_rf": "HMOG-RF",
         "paper_svm": "Paper-SVM", "paper_xgboost": "Paper-XGB",
         "behaveformer_stdat": "BehaveFormer", "authconformer": "AuthConformer"}
MODEN = {"trajectory_xytime": "trajectory", "imu_only": "IMU",
         "imu_trajectory_xytime": "joint"}

COMPARISON = [
    ("diffts_trajectory", "Diffusion-TS (traj arm)"),
    ("diffts_imu", "Diffusion-TS (IMU arm)"),
    ("diffts_both", "Diffusion-TS (dual arm)"),
    ("csdi_unconditional", "CSDI"),
    ("imagentime", "ImagenTime"),
    ("ttsgan", "TTS-GAN"),
    ("pyclick", "pyclick"),
    ("ghostcursor", "ghost-cursor"),
]
ABLATION = [
    ("abl_noshot_adv", "A2", "no five-shot conditioning (k_refs = 0)"),
    ("abl_fewshot_nonadv", "A3", "adversarial training off"),
    ("abl_krefs1", "A4", "k_refs = 1"),
    ("abl_krefs3", "A5", "k_refs = 3"),
    ("abl_krefs8", "A6", "k_refs = 8"),
    ("abl_a7_weighted_sum", "A7", "gradient merging replaced by a weighted sum"),
    ("abl_a8_no_feature", "A8", "feature critic removed"),
    ("abl_a9_no_set", "A9", "set critic removed"),
    ("abl_a10_no_waveform", "A10", "waveform critic removed"),
    ("abl_a11_no_feature_match", "A11", "direct feature-matching loss removed"),
]
CONTROL = ("control_genuine", "Control")


def cells_eer(method: str) -> dict:
    try:
        scores, _missing, _declined = collect(method, "test_eer")
    except SystemExit:
        return {}
    # collect returns read_cell's whole tuple; keep the EER only.
    return {k: (v[0] if isinstance(v, tuple) else v) for k, v in scores.items()}


def stat(values: list) -> str:
    return (f"{statistics.mean(values):.3f} | {statistics.median(values):.3f} | "
            f"{len(values)}")


def table_comparison(ref_eer: dict) -> None:
    print("## Table E1 - Comparison at the test-set EER\n")
    print("Higher is a stronger attack, as in the FAR table -- but the scale is "
          "different and the two must not be quoted against each other.\n")
    print("| Method | Channel | Cells | Mean EER | Median EER |")
    print("|---|---|---|---|---|")
    for modality in MODALITIES:
        v = [c["eer"] for k, c in ref_eer.items() if k[1] == modality]
        print(f"| **Ours (released)** | {MODEN[modality]} | {len(v)} | "
              f"**{statistics.mean(v):.3f}** | {statistics.median(v):.3f} |")
    for method, label in COMPARISON:
        cells = cells_eer(method)
        if not cells:
            print(f"| {label} | - | *not finished* | | |")
            continue
        for modality in covered(method):
            v = [x for k, x in cells.items() if k[1] == modality]
            if v:
                print(f"| {label} | {MODEN[modality]} | {len(v)} | "
                      f"{statistics.mean(v):.3f} | {statistics.median(v):.3f} |")
    cells = cells_eer(CONTROL[0])
    if cells:
        v = list(cells.values())
        print(f"| *{CONTROL[1]}* | *IMU* | *{len(v)}* | *{statistics.mean(v):.3f}* | "
              f"*{statistics.median(v):.3f}* |")


def table_ablation(ref_eer: dict) -> None:
    print("\n## Table E2 - Ablation at the test-set EER\n")
    print("Inertial channel only; keystroke is excluded by construction. A7-A11 "
          "retrain the generator and cover scroll and swipe only, so their means "
          "are over 12 cells against A1's 24 -- the `vs release` column is paired "
          "cell by cell and stays comparable, the means do not.\n")
    ACT = [a for a in ACTIONS if a != "keystroke"]
    base_cells = {k: v["eer"] for k, v in ref_eer.items()
                  if k[1] == "imu_only" and k[0] in ACT}
    print("| Arm | What is removed | Cells | Mean EER | vs release |")
    print("|---|---|---|---|---|")
    print(f"| **A1** | nothing (the released method) | {len(base_cells)} | "
          f"**{statistics.mean(list(base_cells.values())):.3f}** | - |")
    for method, arm, what in ABLATION:
        cells = {k: v for k, v in cells_eer(method).items()
                 if k[1] == "imu_only" and k[0] in ACT}
        if not cells:
            print(f"| {arm} | {what} | *not finished* | | |")
            continue
        paired = [(v, base_cells[k]) for k, v in cells.items() if k in base_cells]
        mine = statistics.mean(list(cells.values()))
        theirs = statistics.mean([b for _, b in paired]) if paired else float("nan")
        print(f"| {arm} | {what} | {len(cells)} | {mine:.3f} | "
              f"**{theirs - mine:+.3f}** |")


def table_both(ref_far: dict, ref_eer: dict) -> None:
    print("\n## Table E3 - The two operating points side by side\n")
    print("Same cells, same scores, two ways of choosing the cut. The ordering is "
          "preserved; the spread is not.\n")
    print("| Method | Channel | FAR @ FRR=5% | Test EER |")
    print("|---|---|---|---|")
    for modality in MODALITIES:
        f = [c["far"] for k, c in ref_far.items() if k[1] == modality]
        e = [c["eer"] for k, c in ref_eer.items() if k[1] == modality]
        print(f"| **Ours (released)** | {MODEN[modality]} | **{statistics.mean(f):.3f}** | "
              f"**{statistics.mean(e):.3f}** |")
    for method, label in COMPARISON + [CONTROL]:
        far_cells, eer_cells = {}, cells_eer(method)
        try:
            far_cells, _m, _d = collect(method, "far5")
        except SystemExit:
            pass
        if not eer_cells:
            continue
        for modality in covered(method):
            f = [x for k, x in far_cells.items() if k[1] == modality]
            e = [x for k, x in eer_cells.items() if k[1] == modality]
            if f and e:
                italic = "*" if method == CONTROL[0] else ""
                print(f"| {italic}{label}{italic} | {italic}{MODEN[modality]}{italic} | "
                      f"{italic}{statistics.mean(f):.3f}{italic} | "
                      f"{italic}{statistics.mean(e):.3f}{italic} |")


def per_detector(ref_eer: dict) -> None:
    print("\n### Per detector, released method, inertial channel\n")
    print("| " + " | ".join(SHORT[d] for d in DETECTORS) + " |")
    print("|---" * len(DETECTORS) + "|")
    row = []
    for d in DETECTORS:
        v = [c["eer"] for k, c in ref_eer.items() if k[1] == "imu_only" and k[2] == d]
        row.append(f"{statistics.mean(v):.3f}" if v else "-")
    print("| " + " | ".join(row) + " |")


def check() -> int:
    """Every recomputed EER must match the cell's own descriptive_test_eer."""
    bad = total = 0
    for method, _label in COMPARISON + [CONTROL] + [(m, a) for m, a, _w in ABLATION]:
        for cell in sorted((ROOT / method).glob("cells_*/cells/*")):
            summary = cell / "summary.json"
            if not summary.is_file():
                continue
            stored = json.loads(summary.read_text()).get(
                "primary_metrics", {}).get("descriptive_test_eer")
            result = read_cell(cell, "test_eer")
            if stored is None or result is None:
                continue
            total += 1
            if abs(result[0] - stored) > 0.005:
                bad += 1
                print(f"  MISMATCH {method}/{cell.name}: "
                      f"recomputed {result[0]:.4f} vs stored {stored:.4f}")
    print(f"cross-check: {total} cells, {bad} mismatched")
    return 1 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--check", action="store_true",
                        help="verify against each cell's own descriptive_test_eer")
    args = parser.parse_args()

    if args.check:
        raise SystemExit(check())

    import contextlib
    import io
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print("# Results at the test-set equal-error rate\n")
        print("A companion to `RESULTS.md`, which reports FAR at a FRR=5% threshold "
              "selected on the development split. This file reports EER located on "
              "the test split, because much of the literature does. It is the "
              "weaker protocol -- the cut is chosen knowing the test scores -- and "
              "the release's own artefacts call the same quantity "
              "`descriptive_test_eer` for that reason.\n")
        ref_far = reference_scores()
        ref_eer = reference_scores_eer()
        table_comparison(ref_eer)
        table_ablation(ref_eer)
        table_both(ref_far, ref_eer)
        per_detector(ref_eer)
    text = buffer.getvalue()
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
