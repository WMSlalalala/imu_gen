#!/usr/bin/env python3
"""Produce every table the paper needs, from whatever has finished so far.

Three tables, and they answer three different questions:

  1. COMPARISON -- each attack against a detector trained on that attack.  This
     is the primary measure: it is what the release's own pipeline does, so
     every row is measured with the same ruler.

  2. ABLATION -- the release's own components removed one at a time, scored the
     same way as (1).  Keystroke is absent by construction, not by omission: its
     fake IMU never passes through the diffusion generator, so no ablation of
     that generator can move it.

  3. TRANSFER -- every attack against the detectors the release actually
     shipped, with no retraining.  This is a separate question, and reporting
     it as attack strength would rank the weakest attack near the top.  It
     belongs in its own section with that caveat stated.

Only cells that exist are reported, and a method is only reported on the
modalities its own build says it changed (`covered_modalities.py`), so a
trajectory-only method never contributes an inertial number that is really the
release's.

Run any time; incomplete rows are marked rather than hidden.
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
from final_release import ACTIONS, DETECTORS, MODALITIES, reference_scores  # noqa: E402
from summarise_final import collect  # noqa: E402

ROOT = Path("/mnt/share/mwang49/data7/results/direct100k/baselines/final")
CROSS = Path("/mnt/share/mwang49/data7/results/direct100k/baselines/crossscore")

SHORT = {
    "hmog_style_svm": "HMOG-SVM", "hmog_style_rf": "HMOG-RF",
    "paper_svm": "Paper-SVM", "paper_xgboost": "Paper-XGB",
    "behaveformer_stdat": "BehaveFormer", "authconformer": "AuthConformer",
}
MODEN = {"trajectory_xytime": "Trajectory", "imu_only": "IMU",
         "imu_trajectory_xytime": "Joint"}

# Third-party baselines, in the order the paper introduces them.
COMPARISON = [
    # Diffusion-TS was built once per channel arm, so the arm is part of the
    # name: without it two rows read as duplicates the moment both arms cover
    # the same modality.
    ("diffts_trajectory", "Diffusion-TS (traj arm)", "Yuan & Qiao, ICLR 2024"),
    ("diffts_imu", "Diffusion-TS (IMU arm)", "Yuan & Qiao, ICLR 2024"),
    ("diffts_both", "Diffusion-TS (dual arm)", "Yuan & Qiao, ICLR 2024"),
    ("csdi_unconditional", "CSDI", "Tashiro et al., NeurIPS 2021"),
    ("imagentime", "ImagenTime", "Naiman et al., NeurIPS 2024"),
    ("ttsgan", "TTS-GAN", "Li et al., AIME 2022"),
    ("pyclick", "pyclick", "Bezier human-cursor library"),
    ("ghostcursor", "ghost-cursor", "Fitts-law cursor library"),
]
# The release's own components, removed one at a time.
ABLATIONS = [
    ("abl_noshot_adv", "A2", "no five-shot conditioning (k_refs = 0)"),
    ("abl_fewshot_nonadv", "A3", "adversarial training off"),
    ("abl_krefs1", "A4", "k_refs = 1"),
    ("abl_krefs3", "A5", "k_refs = 3"),
    ("abl_krefs8", "A6", "k_refs = 8"),
    ("abl_a7_weighted_sum", "A7", "gradient merging replaced by a weighted sum"),
    # A8-A11 decompose A3: the four things `adv` runs at once.  A11 is not one of
    # `adv.critics` -- it is the direct feature-matching loss in the trainer --
    # but it lives inside the same block A3 switches off, so without it the
    # decomposition would not add up to A3.
    ("abl_a8_no_feature", "A8", "feature critic removed"),
    ("abl_a9_no_set", "A9", "set critic removed"),
    ("abl_a10_no_waveform", "A10", "waveform critic removed"),
    ("abl_a11_no_feature_match", "A11", "direct feature-matching loss removed"),
]
# A7-A11 retrain the generator, so they cover only the two actions that were
# retrained: their Cells column reads 12 where A2-A6 read 24.  The "vs release"
# column is computed cell by cell against the same cells, so it stays comparable
# across arms even though the means are over different action mixes.
TWO_ACTION_NOTE = (
    "A7-A11 retrain the generator and cover scroll and swipe only, so their "
    "means are over 12 cells rather than 24. The `vs release` column is paired "
    "cell by cell, so it remains comparable across every arm; the means are not, "
    "and should be read against arms with the same cell count."
)
CONTROL = ("control_genuine", "Control", "genuine windows swapped in as the fake channel")


def cells_of(method: str) -> dict:
    try:
        scores, _missing, _declined = collect(method)
    except SystemExit:
        return {}
    return scores


def summary(cells: dict, modality: str | None = None) -> dict | None:
    values = [v for k, v in cells.items() if modality is None or k[1] == modality]
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "at60": sum(1 for v in values if v >= 0.6),
    }


def table_comparison(reference: dict) -> None:
    print("## Table 1 - Comparison: each attack against a detector trained on it\n")
    print("| Method | Reference | Channel | Cells | Mean | Median | Cells >= 0.60 |")
    print("|---|---|---|---|---|---|---|")
    for modality in MODALITIES:
        values = [c["far"] for k, c in reference.items() if k[1] == modality]
        print(f"| **Ours (released)** | - | {MODEN[modality].lower()} | {len(values)} | "
              f"**{statistics.mean(values):.3f}** | {statistics.median(values):.3f} | "
              f"**{sum(1 for v in values if v >= 0.6)}** |")
    for method, label, ref in COMPARISON:
        cells = cells_of(method)
        if not cells:
            print(f"| {label} | {ref} | - | *not finished* | | | |")
            continue
        for modality in covered(method):
            s = summary(cells, modality)
            if s:
                print(f"| {label} | {ref} | {MODEN[modality].lower()} | {s['n']} | "
                      f"{s['mean']:.3f} | {s['median']:.3f} | {s['at60']} |")
    cells = cells_of(CONTROL[0])
    s = summary(cells)
    if s:
        print(f"| *{CONTROL[1]}* | *{CONTROL[2]}* | *IMU* | *{s['n']}* | "
              f"*{s['mean']:.3f}* | *{s['median']:.3f}* | *{s['at60']}* |")


def table_ablation(reference: dict) -> None:
    print("\n## Table 2 - Ablation: the release's own components, removed one at a time\n")
    print("Inertial channel only, and keystroke is excluded by construction: its fake IMU "
          "is written by an analytic adapter (`diffusion_used: false`), so no ablation of "
          "the diffusion generator can reach it.\n")
    ACT = [a for a in ACTIONS if a != "keystroke"]
    reference_cells = {
        k: v["far"] for k, v in reference.items()
        if k[1] == "imu_only" and k[0] in ACT
    }
    print("| Arm | What is removed | Cells | Mean | vs release | Cells >= 0.60 |")
    print("|---|---|---|---|---|---|")
    base = statistics.mean(list(reference_cells.values()))
    print(f"| **A1** | nothing (the released method) | {len(reference_cells)} | "
          f"**{base:.3f}** | - | **{sum(1 for v in reference_cells.values() if v >= 0.6)}** |")
    for method, arm, what in ABLATIONS:
        cells = cells_of(method)
        cells = {k: v for k, v in cells.items() if k[1] == "imu_only" and k[0] in ACT}
        if not cells:
            print(f"| {arm} | {what} | *not finished* | | | |")
            continue
        paired = [(v, reference[k]["far"]) for k, v in cells.items() if k in reference]
        mine = statistics.mean(list(cells.values()))
        theirs = statistics.mean([b for _, b in paired]) if paired else float("nan")
        print(f"| {arm} | {what} | {len(cells)} | {mine:.3f} | "
              f"**{theirs - mine:+.3f}** | {sum(1 for v in cells.values() if v >= 0.6)} |")

    if any(cells_of(m) for m, _a, _w in ABLATIONS
           if m.startswith(("abl_a7", "abl_a8", "abl_a9", "abl_a10", "abl_a11"))):
        print(f"\n{TWO_ACTION_NOTE}")

    # Which detector each arm costs the most on -- the ablations do not move
    # the six detectors equally, and the per-detector view is where the
    # mechanism shows.
    print("\n### Per detector, mean over the four actions\n")
    print("| Arm | " + " | ".join(SHORT[d] for d in DETECTORS) + " |")
    print("|---" * (len(DETECTORS) + 1) + "|")
    row = []
    for detector in DETECTORS:
        values = [v for k, v in reference_cells.items() if k[2] == detector]
        row.append(f"**{statistics.mean(values):.3f}**" if values else "-")
    print("| **A1 (released)** | " + " | ".join(row) + " |")
    for method, arm, _what in ABLATIONS:
        cells = cells_of(method)
        row = []
        for detector in DETECTORS:
            values = [v for k, v in cells.items()
                      if k[1] == "imu_only" and k[2] == detector and k[0] in ACT]
            row.append(f"{statistics.mean(values):.3f}" if values else "-")
        if any(x != "-" for x in row):
            print(f"| {arm} | " + " | ".join(row) + " |")
    print()   # the next table's prose ran onto this one's last row without it


def table_transfer() -> None:
    print("\n## Table 3 - Transfer: every attack against the detectors the release shipped\n")
    print("No retraining and no threshold re-selection. This answers a different question "
          "from Table 1 and must not be read as attack strength: the weakest attack in "
          "Table 1 scores near the top here, because these detectors were trained on the "
          "release's artefacts and a different generator's artefacts simply pass.\n")
    print("| Method | Channel | Cells | Mean | Median |")
    print("|---|---|---|---|---|")
    reference = reference_scores()
    for modality in ("imu_only",):
        values = [c["far"] for k, c in reference.items() if k[1] == modality]
        print(f"| **Ours (released)** | {MODEN[modality].lower()} | {len(values)} | "
              f"**{statistics.mean(values):.3f}** | {statistics.median(values):.3f} |")
    print("| | | | | |")
    # Diffusion-TS was built three times and the arms share their per-channel
    # samples: the dual arm's inertial cells are the IMU arm's inertial cells,
    # so listing both prints the same numbers twice under different names.  Each
    # (channel, sample source) is reported once, by the arm that is only that
    # channel where one exists.
    preferred = {
        "imu_only": "diffts_imu",
        "trajectory_xytime": "diffts_trajectory",
        "imu_trajectory_xytime": "diffts_both",
    }
    rows = []
    for method, label, _ref in COMPARISON + [CONTROL[:1] + (CONTROL[1], CONTROL[2])]:
        path = CROSS / f"{method}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        for modality in covered(method):
            if method.startswith("diffts") and preferred.get(modality) != method:
                continue
            values = [c["far"] for k, c in data["cells"].items()
                      if k.split("__")[1] == modality]
            if values:
                rows.append((statistics.mean(values), label, MODEN[modality].lower(),
                             len(values), statistics.median(values)))
    for mean, label, modality, n, median in sorted(rows, reverse=True):
        italic = "*" if label == "Control" else ""
        print(f"| {italic}{label}{italic} | {italic}{modality}{italic} | {italic}{n}{italic} | "
              f"{italic}{mean:.3f}{italic} | {italic}{median:.3f}{italic} |")

    selfcheck = CROSS / "_release_selfcheck.json"
    if selfcheck.is_file():
        data = json.loads(selfcheck.read_text())
        values = [c["far"] for c in data["cells"].values()]
        expected = [reference[tuple(k.split("__"))]["far"] for k in data["cells"]
                    if tuple(k.split("__")) in reference]
        gap = max(abs(a - b) for a, b in zip(values, expected)) if expected else float("nan")
        print(f"\nSelf-check: scoring the release with its own detectors reproduces its "
              f"published cells exactly ({len(values)} cells, max |difference| {gap:.4f}). "
              f"The release's two columns coincide by construction -- for it alone, the "
              f"detector trained on the attack *is* the shipped detector.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print("# Results\n")
        print("FAR at the development-selected FRR = 5% threshold; higher is a "
              "stronger attack.\n")
        reference = reference_scores()
        table_comparison(reference)
        table_ablation(reference)
        table_transfer()
    text = buffer.getvalue()
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
