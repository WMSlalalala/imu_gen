#!/usr/bin/env python3
"""Write one reproducibility note per baseline, from what the build recorded.

A reviewer's first question about a third-party baseline is whether it was run
faithfully or quietly weakened.  The honest answer has to be specific: which
code is the authors', which file was changed and why, what budget it was given
against what its paper specifies, and what it declined to model at all.

All of that was recorded at build time -- `method_detail` in each dataset's
`release.json`, the training summaries beside the samples, the bundle manifest's
per-action swap counts -- so these notes are generated from the artefacts rather
than written from memory, and they cannot drift from what actually ran.

Anything the generator could not be made to do is stated as a limitation rather
than omitted.  A baseline that looks weak because it was given a quarter of its
authors' budget is a different claim from one that is weak on the task, and the
note has to let a reader tell those apart.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from covered_modalities import covered, swapped_channels  # noqa: E402
from final_release import ACTIONS, MODALITIES  # noqa: E402
from summarise_final import collect  # noqa: E402

ROOT = Path("/mnt/share/mwang49/data7/results/direct100k/baselines/final")
RESULTS = Path("/mnt/share/mwang49/data7/results/direct100k/baselines")

MODEN = {"trajectory_xytime": "trajectory", "imu_only": "IMU",
         "imu_trajectory_xytime": "joint"}

def imagentime_budget() -> str:
    """State the real budget from the training summaries, not from intent.

    The authors specify 1000 epochs.  Training stopped earlier, on a convergence
    rule, so claiming their budget would be false; claiming "we could not afford
    it" would also be false.  The honest form is the rule, the epoch it fired,
    and the epoch actually kept -- read back per action so this line cannot drift
    from what ran.
    """
    rows = []
    for action in ("tap", "pinch", "swipe", "scroll", "keystroke"):
        summary = RESULTS / "imagentime" / action / f"summary_{action}.json"
        if not summary.is_file():
            continue
        data = json.loads(summary.read_text())
        stopped = data.get("stopped_early_at")
        history = data.get("convergence_history") or []
        if stopped is None:
            rows.append(f"{action} ran the full {data.get('epochs', 1000)}")
            continue
        # The sampler draws from the model as it stands when training stops, not
        # from the best-scoring checkpoint: one checkpoint file is rewritten each
        # save and `best_step` is reported rather than restored.  Quoting both
        # gaps lets a reader see how much that costs instead of taking the word
        # "early stopping" to mean something the code does not do.
        at_stop = next((r for r in reversed(history) if r.get("step") == stopped), None)
        best = min(history, key=lambda r: r.get("gap", float("inf"))) if history else None
        detail = f"{action} stopped at {stopped}"
        if at_stop and best and best.get("step") != stopped:
            detail += (f" (gap {at_stop['gap']:.3f}; the best check was epoch "
                       f"{best['step']} at {best['gap']:.3f})")
        elif at_stop:
            detail += f" (gap {at_stop['gap']:.3f}, its own best)"
        rows.append(detail)
    if not rows:
        return ("Training was stopped on a convergence rule rather than at the "
                "authors' 1000 epochs; per-action figures are in each "
                "`summary_<action>.json`.")
    return (
        "Early stopping, against the authors' fixed 1000 epochs. Every 25 epochs "
        "a draw is compared with genuine data on lag-1 autocorrelation, "
        "per-channel dispersion and a window-summary energy distance; training "
        "stops when that gap has not improved for 3 consecutive checks. The "
        "samples are drawn from the model at the stopping epoch, not from the "
        "best-scoring check -- both are given below so the difference is visible "
        "rather than assumed away. Per action: " + "; ".join(rows) +
        ". The full histories are in `summary_<action>.json`, so a reader can "
        "see the gap had flattened rather than take the stop on trust."
    )


# What each baseline is, what was reproduced verbatim, and every deviation.
# Deviations are listed even when they favour the baseline.
NOTES = {
    "diffts": {
        "title": "Diffusion-TS",
        "citation": "X. Yuan and Y. Qiao, \"Diffusion-TS: Interpretable Diffusion for "
                    "General Time Series Generation\", ICLR 2024",
        "repository": "https://github.com/Y-debug-sys/Diffusion-TS",
        "verbatim": [
            "the model, the interpretable trend/seasonality decomposition, the "
            "diffusion schedule and the training loop, unmodified",
            "the authors' own hyper-parameters for each sequence length",
        ],
        "deviations": [
            "One dataset class was written to feed HMOG windows, because the "
            "repository ships loaders only for its own corpora.",
        ],
        "sanity": "The authors' Sines benchmark was reproduced first: "
                  "discriminative score 0.0108 +/- 0.0061 against the paper's "
                  "0.006 +/- 0.007, so the training path is sound before it "
                  "was pointed at HMOG.",
    },
    "csdi_unconditional": {
        "title": "CSDI",
        "citation": "Y. Tashiro, J. Song, Y. Song and S. Ermon, \"CSDI: Conditional "
                    "Score-based Diffusion Models for Probabilistic Time Series "
                    "Imputation\", NeurIPS 2021",
        "repository": "https://github.com/ermongroup/CSDI",
        "verbatim": [
            "`CSDI_Physio` and its denoiser, unmodified",
            "every hyper-parameter from the authors' own `config/base.yaml`",
            "their loss, optimiser, and MultiStepLR schedule",
        ],
        "deviations": [
            "A Dataset returning the four keys `process_data` reads, because "
            "the repository ships loaders for three medical corpora only.",
            "Per-channel z-scoring, which is mandatory rather than optional: "
            "the model has no normalisation layer and all three official "
            "datasets standardise before the DataLoader, while accelerometer "
            "and gyroscope differ by roughly 7x in scale.",
            "An unconditional draw. CSDI is an imputer with no generation entry "
            "point; passing an all-zero conditioning mask turns the authors' own "
            "`impute` into a plain DDPM over the whole window. No model change.",
            "Checkpointing every epoch. The authors' `train()` writes one file "
            "after the final epoch and never uses the best validation loss it "
            "computes, so an interrupted run keeps nothing.",
            "The five-shot conditional arm was never completed, so it is not "
            "measured anywhere: after four consecutive `ERROR ... rerun this "
            "script to resume` entries in `csdi_queue.log`, only tap and pinch "
            "ever produced samples, and the arm reached no bank, no bundle and "
            "no detector cell. `run_csdi_all.sh` then left `unconditional` as "
            "its only default mode (`CSDI_MODES` brings the arm back); the "
            "reason its comment gives -- that both arms come from identical "
            "weights and differ only in what the sampler is told, so a second "
            "CSDI row would cost ten more deep detector cells -- is why the arm "
            "was not resumed, not a conclusion drawn from measuring it.",
        ],
        "sanity": "`linear_attention_transformer` was installed rather than "
                  "stubbed, so `diff_models.py` is byte-identical to the "
                  "authors' file; `target_strategy` was left at `random`, "
                  "because their historical-masking branch makes the training "
                  "loss identically zero on fully-observed windows.",
    },
    "imagentime": {
        "title": "ImagenTime",
        "citation": "I. Naiman, N. Berman, I. Pemper, I. Arbiv, G. Fadlon and O. Azencot, "
                    "\"Utilizing Image Transforms and Diffusion Models for Generative "
                    "Modeling of Short and Long Time Series\", NeurIPS 2024",
        "repository": "https://github.com/azencot-group/ImagenTime",
        "verbatim": [
            "the delay-embedding transform, the UNet, the EDM loss, the EMA and "
            "the reverse process, unmodified",
        ],
        "deviations": [
            "`data/long_range.py` and `data/data_provider/` were written: the "
            "repository ships `data/` as an empty package because the corpora "
            "are distributed separately, so this is its intended extension "
            "point rather than a modification.",
            "A delay/embedding pair per action, found by exhaustive search over "
            "the pairs that fill the square exactly so the transform inverts. "
            "tap admits no exact fill (T=16) and uses the smallest square with "
            "two zero columns, which still inverts exactly.",
            "Per-channel min-max scaling into [0,1], which every corpus they "
            "ship receives and which EDM's hard-coded `sigma_data = 0.5` "
            "assumes.",
            "A training driver and a sampler. Their `run_unconditional.py` "
            "scores the model by refitting S4 classifiers ten times per "
            "evaluation and checkpoints only inside that block on improvement; "
            "the repository has no sampling script at all.",
            "The authors' narrower UNet from their own `mujoco.yaml` (14.4M "
            "parameters against 151.7M) for every action. This is a "
            "configuration choice of theirs, not a model change.",
            imagentime_budget,
        ],
        "sanity": "`img_to_ts` unpads against a shape cached by the first "
                  "`ts_to_img` call, so one real batch is always pushed through "
                  "before any draw -- otherwise sampling silently returns the "
                  "wrong length.",
    },
    "ttsgan": {
        "title": "TTS-GAN",
        "citation": "X. Li, V. Metsis, H. Wang and A. H. H. Ngu, \"TTS-GAN: A "
                    "Transformer-based Time-Series Generative Adversarial Network\", "
                    "AIME 2022",
        "repository": "https://github.com/imics-lab/tts-gan",
        "verbatim": [
            "the transformer generator and discriminator, the training loop, "
            "the EMA and the learning-rate schedule, unmodified",
            "the authors' per-window z-score normalisation",
        ],
        "deviations": [
            "Mid-training checkpointing was added. The authors' loop keeps "
            "nothing between epochs, so an interrupted run loses everything.",
            # Two genuine lag-1 figures are in circulation and both are right:
            # this note quotes the value recorded in `ttsgan_budget_evidence.json`
            # when the budget decision was taken, EXPERIMENTS_CN.md 3.7 quotes a
            # later recomputation of the same statistic over the saved samples.
            # A reader holding only one of the two documents would otherwise take
            # them for two disagreeing measurements, so the note names both and
            # states the size of the gap.
            "The budget was raised after a first run proved undertrained: its "
            "lag-1 autocorrelation was 0.387-0.850 against genuine data's "
            "0.945-0.996 (the recomputed genuine figure quoted in "
            "`EXPERIMENTS_CN.md` 3.7 is 0.947-0.996 -- the same quantity "
            "computed twice, differing by <=0.002 per action, not two "
            "measurements). The evidence for that decision is kept in "
            "`ttsgan_budget_evidence.json` so the reported result cannot be "
            "read as a statement about the first budget.",
        ],
        "sanity": None,
    },
    "pyclick": {
        "title": "pyclick",
        "citation": "pyclick, a Bezier human-like cursor path library (MIT)",
        "repository": "https://github.com/patrikoss/pyclick",
        "verbatim": ["`HumanCurve` and its distortion/tweening, unmodified"],
        "deviations": [
            "`pyautogui` is imported by the library at module scope but never "
            "used on this path; an empty stub module satisfies the import "
            "without touching the library.",
        ],
        "sanity": None,
    },
    "ghostcursor": {
        "title": "ghost-cursor",
        "citation": "ghost-cursor, a Fitts-law cursor path library",
        "repository": "https://github.com/Xetera/ghost-cursor",
        "verbatim": [
            "the Bezier path construction, the overshoot rule and its "
            "corrective second segment, ported faithfully from `spoof.ts`",
        ],
        "deviations": [
            "Two documented port corrections: arc-length sampling in place of "
            "parameter-space sampling, and an honoured `spreadOverride`.",
            "A degenerate-chord guard returning a constant point, recorded as "
            "declined-by-degeneracy rather than crashing.",
        ],
        "sanity": None,
    },
    "control_genuine": {
        "title": "Control: genuine windows as the fake channel",
        "citation": "not a baseline",
        "repository": None,
        "verbatim": ["genuine inertial windows, unaltered"],
        "deviations": [],
        # Not "the ceiling": the control carries two mismatches the release's
        # own fake channel does not, so the release scoring above it is expected
        # rather than paradoxical.  Calling this row an upper bound would
        # contradict the measured numbers and the write-up (EXPERIMENTS_CN 2.4).
        "sanity": "This is a same-pipeline reference point, not an upper "
                  "bound. These windows come from a different genuine event "
                  "and are resampled to the detector's fixed window length -- "
                  "two mismatches the release's own fake channel never carries "
                  "-- so a method can score above this row without being more "
                  "real than real. What the number measures is how much "
                  "separability the harness itself leaks, not a ceiling on "
                  "synthesis quality.",
    },
}


def note_for(method: str) -> dict | None:
    for key, note in NOTES.items():
        if method == key or method.startswith(key):
            return note
    return None


def write(method: str) -> bool:
    root = ROOT / method
    if not (root / "bundle_manifest.json").is_file():
        return False
    note = note_for(method)
    if note is None:
        return False

    manifest = json.loads((root / "bundle_manifest.json").read_text())
    channels = swapped_channels(method)
    lines = [f"# {note['title']} as a baseline on HMOG", ""]
    if note["citation"]:
        lines += [f"**Paper.** {note['citation']}", ""]
    if note["repository"]:
        lines += [f"**Code.** {note['repository']}", ""]

    lines += ["## What ran unchanged", ""]
    lines += [f"- {item}" for item in note["verbatim"]] + [""]

    lines += ["## Every deviation", ""]
    if note["deviations"]:
        lines += [f"- {item() if callable(item) else item}"
                  for item in note["deviations"]]
    else:
        lines += ["- None beyond the harness described below."]
    lines += [""]

    if note["sanity"]:
        lines += ["## Sanity check", "", note["sanity"], ""]

    # A modality can be absent from the result table for two opposite reasons,
    # and one line saying "reported on" cannot carry both.  "Supplied" is what
    # this build's own `release.json` says the method wrote, so it is what the
    # method may legitimately be scored on; "reported" is what actually has
    # cells on disk.  An unsupplied modality is withheld on purpose -- those
    # cells would be running the release's own data.  A supplied modality with
    # no cells is only an unfinished detector-grid job.  Both are read back
    # from the artefacts so neither can drift.
    try:
        cells, missing, _ = collect(method)
    except SystemExit:
        cells, missing = {}, []
    supplied = covered(method)
    reported = [m for m in MODALITIES if any(key[1] == m for key in cells)]
    pending = [m for m in supplied if m not in reported]
    pending_cells = sum(1 for item in missing
                        if any(f"__{m}__" in item for m in pending))

    lines += [
        "## How it was scored", "",
        "The generator supplies "
        + ("the inertial channel" if channels.get("imu") and not channels.get("trajectory_xy")
           else "the touch coordinates" if channels.get("trajectory_xy") and not channels.get("imu")
           else "both channels")
        + ". Everything else -- the carrier's update timing, its no-contact "
        "sentinel, its clock column, and the genuine events -- is the release's "
        "and is left byte-identical, which `verify_harness.py` checks.",
        "",
        f"Modalities supplied: {', '.join(MODEN[m] for m in supplied) or 'none'}. "
        f"Modalities reported below: {', '.join(MODEN[m] for m in reported) or 'none'}."
        + (f" Not yet computed: {', '.join(MODEN[m] for m in pending)} "
           f"({pending_cells} cells)." if pending else ""),
        "",
        "A modality this method does not supply is absent from every one of "
        "those lists and carries no number anywhere, because those cells would "
        "be running the release's own data."
        + (" A supplied modality listed as not yet computed is the opposite "
           "case -- a detector-grid job that has not been run -- so its "
           "absence from the table below is a gap in the measurement, not a "
           "reporting decision and not a result." if pending else ""),
        "",
    ]

    # The manifest records each declined entry as "<bundle>/<action>", and a
    # bundle that owns a single action is named after it -- so the raw string
    # reads "keystroke/keystroke" and invites a reader to look for two different
    # entities.  Every action is owned by exactly one bundle, so the action alone
    # names it unambiguously; the bundle is an internal routing detail.
    declined = [entry.split("/")[-1]
                for entry in manifest.get("declined_by_method", [])]
    if declined:
        lines += [
            "**Declined actions.** " + ", ".join(declined) + ". A method that "
            "does not model an action declines it rather than having something "
            "its authors never proposed invented for it; declined actions carry "
            "no number.", "",
        ]

    if cells:
        lines += ["## Result", "", "| Modality | Cells | Mean FAR | Cells >= 0.60 |",
                  "|---|---|---|---|"]
        for modality in MODALITIES:
            values = [v for k, v in cells.items() if k[1] == modality]
            if values:
                lines.append(f"| {MODEN[modality]} | {len(values)} | "
                             f"{statistics.mean(values):.3f} | "
                             f"{sum(1 for v in values if v >= 0.6)} |")
        lines += ["", "FAR at the development-selected FRR = 5% threshold, "
                  "against a detector trained on this attack.", ""]
        if missing:
            # Name the modality that is waiting, not just the cell count: a bare
            # number under a table with fewer rows than the method supplies
            # leaves a reader unable to tell a pending job from a withheld one.
            breakdown = ", ".join(
                f"{MODEN[m]} {sum(1 for item in missing if f'__{m}__' in item)}"
                for m in MODALITIES
                if any(f"__{m}__" in item for item in missing)
            )
            lines += [f"*{len(missing)} cell(s) not yet computed"
                      + (f": {breakdown}" if breakdown else "")
                      + " -- supplied modalities still awaiting the detector "
                        "grid, not modalities withheld.*", ""]

    (root / "README.md").write_text("\n".join(lines))
    return True


# ---------------------------------------------------------------------------
# The ablation arms.  These are the paper's own claims, so they need the same
# treatment the third-party baselines get: what changed, what stayed bit-identical,
# and where the reader can check that the change actually took effect.
ABL_CACHE = Path("/mnt/share/mwang49/data7/results/direct100k/baselines/ablations")
ABL_NOTES = {
    "abl_noshot_adv":       ("A2", "noshot_adv",          "five-shot conditioning removed (k_refs = 0)"),
    "abl_fewshot_nonadv":   ("A3", "fewshot_nonadv",      "adversarial training removed"),
    "abl_krefs1":           ("A4", "krefs1",              "k_refs = 1 at sampling time"),
    "abl_krefs3":           ("A5", "krefs3",              "k_refs = 3 at sampling time"),
    "abl_krefs8":           ("A6", "krefs8",              "k_refs = 8 at sampling time"),
    "abl_a7_weighted_sum":  ("A7", "a7_weighted_sum_cache",
                             "gradient merging replaced by a plain weighted sum"),
    # A8-A11 decompose A3.  A11 removes a loss weight rather than a critic --
    # `feature_match` is not in `adv.critics` -- but it sits in the same
    # `adv_update` block, so A3 switches it off too and the decomposition needs it.
    "abl_a8_no_feature":    ("A8", "a8_no_feature_cache",  "the feature critic"),
    "abl_a9_no_set":        ("A9", "a9_no_set_cache",      "the set critic"),
    "abl_a10_no_waveform":  ("A10", "a10_no_waveform_cache", "the waveform critic"),
    "abl_a11_no_feature_match": ("A11", "a11_no_feature_match_cache",
                                 "the direct feature-matching loss"),
}


def effective_setting(cache: Path) -> str | None:
    """Read back what a drawn sample actually carries.

    `k_refs` was once overridden on the wrong attribute and silently did
    nothing, so the arm's setting is quoted from a sample rather than from the
    request that was meant to produce it.
    """
    for sample in sorted(cache.glob("user_*/*/*/sample_0000.npz"))[:1]:
        try:
            import numpy as np
            meta = json.loads(str(np.load(sample, allow_pickle=True)["metadata_json"]))
        except Exception:
            return None
        count, used = meta.get("ref_count"), meta.get("used_ref_indices")
        if count is None:
            return None
        return (f"a drawn sample carries `ref_count = {count}`"
                + (f" with {len(used)} reference indices" if used else "")
                + f", protocol `{meta.get('protocol')}`")
    return None


# The trainer prints one JSON line per logged step.  `adv_acc_<name>` is critic
# `<name>`'s discrimination accuracy at that step and `adv_feature_match_<stat>`
# is the direct match on statistic `<stat>`, so a component that was really
# switched off leaves no key of its own anywhere in the log.  That makes the key
# census the evidence for a training-side arm; the sampler read-back above can
# only ever speak for the sampling protocol.
ADV_ACC = "adv_acc_"
FEATURE_MATCH = "adv_feature_match_"
# Fields whose change is visible in the merged gradient rather than in which
# keys exist -- A7 keeps every critic and moves these instead.
GRADIENT_FIELDS = {"project_conflicts", "max_grad_ratio", "g_weight"}
MERGE_SCALE = "adv_grad_merge_scale"
# Mirrors `ALWAYS_DECLINED` in `build_ablation_cache_baseline.py`: an action whose
# fake IMU never passes through the diffusion generator cannot be moved by an
# ablation of it, so it is declined before any coverage check runs.
ALWAYS_DECLINED = frozenset({"keystroke"})


def arm_training_dir(cache: Path) -> Path | None:
    """Where this arm trained its own generator, if it trained one at all.

    A2-A6 sample from checkpoints that already existed, so there is nothing to
    read.  The retrained arms are exactly the ones whose cache is named
    `<arm>_cache` beside an `<arm>/` directory holding the config their driver
    wrote and the training log it produced.
    """
    if not cache.name.endswith("_cache"):
        return None
    directory = cache.with_name(cache.name[: -len("_cache")])
    return directory if directory.is_dir() else None


def arm_plan(train_dir: Path) -> list[dict]:
    """The per-action record of what the driver flipped, as it wrote it."""
    for name in ("plan.json", "a7_plan.json"):
        path = train_dir / name
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except ValueError:
                return []
    return []


def plan_changes(plan: list[dict]) -> dict[str, list[str]]:
    """`"critics.feature `true` -> `false`"` -> the actions it applies to.

    Diffed from the plan's own before/after rather than restated, so an arm that
    was re-run against a corrected config cannot leave this note describing the
    first attempt.  Grouped by description because the same flip usually applies
    to every action, and printing it once per action reads as more evidence than
    there is.
    """
    grouped: dict[str, list[str]] = {}
    for record in plan:
        action = record.get("action", "?")
        before, after = record.get("before") or {}, record.get("after") or {}
        for key in sorted(set(before) | set(after)):
            first, second = before.get(key), after.get(key)
            pairs = ([(f"{key}.{inner}", (first or {}).get(inner), (second or {}).get(inner))
                      for inner in sorted(set(first or {}) | set(second or {}))]
                     if isinstance(first, dict) or isinstance(second, dict)
                     else [(key, first, second)])
            for name, was, now in pairs:
                if was != now:
                    grouped.setdefault(
                        f"`{name}` `{json.dumps(was)}` -> `{json.dumps(now)}`", []
                    ).append(action)
    return grouped


def training_census(train_dir: Path) -> dict:
    """Per-action counts of the adversarial keys the trainer actually printed."""
    census = {}
    for log in sorted(train_dir.glob("train_*.log")):
        critics, matched, merge = Counter(), Counter(), []
        with log.open(errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line.startswith("{") or not line.endswith("}"):
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                for key, value in row.items():
                    if key.startswith(ADV_ACC):
                        critics[key[len(ADV_ACC):]] += 1
                    elif key.startswith(FEATURE_MATCH):
                        matched[key] += 1
                    elif key == MERGE_SCALE and isinstance(value, (int, float)):
                        merge.append(float(value))
        if critics or matched:
            census[log.stem[len("train_"):]] = {
                "log": log.name, "critics": critics, "match": matched, "merge": merge}
    return census


def _uniform(counts: Counter) -> str:
    """A count column that says how many distinct keys carry it."""
    if not counts:
        return "**0**"
    values = set(counts.values())
    body = str(next(iter(values))) if len(values) == 1 else \
        f"{min(values)}-{max(values)}"
    return body if len(counts) == 1 else f"{body} on each of {len(counts)} keys"


def training_evidence(train_dir: Path, plan: list[dict]) -> list[str]:
    """What the arm's own training logs show about the component it removed.

    Written because the sampler read-back is not evidence for these arms: a
    `ref_count` of 5 says the five-shot protocol was in force and says nothing
    at all about whether a critic was switched off.  The counts below are the
    thing a reader has to be able to check.
    """
    census = training_census(train_dir)
    if not census:
        return []
    changes = plan_changes(plan)

    # Every critic the released base ran gets a column, so one that was switched
    # off shows a zero instead of quietly not being mentioned.
    names = set()
    for record in plan:
        names |= set((record.get("before") or {}).get("critics") or {})
    for entry in census.values():
        names |= set(entry["critics"])
    names = sorted(names)

    lines = [
        "## The change took effect", "",
        "Read back from this arm's own training logs, not from the config that "
        "requested the change. The trainer prints one JSON line per logged step "
        "and names each live component in it -- `adv_acc_<name>` is critic "
        "`<name>`'s discrimination accuracy at that step -- so a component that "
        "was really switched off leaves no key behind at all. Counts are the "
        "number of logged JSON lines carrying the key.", "",
        "| Training log | " + " | ".join(f"`{ADV_ACC}{n}`" for n in names)
        + f" | `{FEATURE_MATCH}*` |",
        "|---" * (len(names) + 2) + "|",
    ]
    for action in sorted(census):
        entry = census[action]
        row = [f"`{entry['log']}`"]
        row += [str(entry["critics"][n]) if entry["critics"].get(n) else "**0**"
                for n in names]
        row.append(_uniform(entry["match"]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Name the group's members once.  "6 keys" alone invites a reader who knows
    # the five matched statistics to think one of them has been miscounted; the
    # sixth is the weight the trainer applies to them, printed alongside.
    members = sorted({key[len(FEATURE_MATCH):]
                      for entry in census.values() for key in entry["match"]})
    if members:
        lines += [f"The `{FEATURE_MATCH}*` group is `"
                  + "`, `".join(members) + "` -- the matched statistics and the "
                  "weight the trainer applies to them.", ""]

    if changes:
        lines += ["The plan its driver wrote for these runs asked for exactly "
                  "that: " + "; ".join(
                      f"{description} ({', '.join(actions)})"
                      for actions, description in sorted(
                          (v, k) for k, v in changes.items())
                  ) + ".", ""]
    # The one derivation mistake that would silently invalidate a critic arm is
    # building it on A7's config instead of the release's, and the plan records
    # what the base actually held -- so quote it rather than assert the base.
    protection = {json.dumps(record["gradient_protection"], sort_keys=True)
                  for record in plan if record.get("gradient_protection")}
    if len(protection) == 1 and not any(
            field in "".join(changes) for field in GRADIENT_FIELDS):
        settings = json.loads(protection.pop())
        lines += ["The gradient protection is A7's variable, not this arm's, and "
                  "the plan records it unchanged at the release's own "
                  + ", ".join(f"`{k}: {json.dumps(v)}`"
                              for k, v in sorted(settings.items()))
                  + " for every run here.", ""]

    gone = [n for n in names
            if not any(entry["critics"].get(n) for entry in census.values())]
    kept = [n for n in names if n not in gone]
    if gone:
        lines += [f"The zero column is the removal: `{ADV_ACC}{gone[0]}` never "
                  "appears"
                  + (f", while `{'`, `'.join(ADV_ACC + n for n in kept)}` appear on "
                     "every logged step." if kept else "."), ""]
    match_alive = any(entry["match"] for entry in census.values())
    if gone and match_alive:
        lines.append(
            f"`{FEATURE_MATCH}*` is still counted on every step, which is this "
            "arm's honest limit: it removes a critic, not the direct "
            "feature-matching loss that runs in the same `adv_update` block."
            + (" Because the critic removed here is the feature one, the feature "
               "pathway is only half removed and the direct match keeps supplying "
               "the same statistics -- that other half is A11's variable."
               if gone == ["feature"] else ""))
    elif not match_alive and kept:
        lines.append(
            f"`{FEATURE_MATCH}*` is the removal here -- the whole group is absent -- "
            f"while `{'`, `'.join(ADV_ACC + n for n in kept)}` keep their counts, "
            "so this arm moved the loss weight and left the critics running.")
    if gone or not match_alive:
        lines.append("")

    # An arm that changes how the adversarial gradient is merged removes no key
    # at all, so its evidence has to be the merged quantity itself.
    if any(field in "".join(changes) for field in GRADIENT_FIELDS):
        merged = [f"`{census[a]['log']}`: "
                  + (f"{min(census[a]['merge']):.4g} on all "
                     f"{len(census[a]['merge'])} logged steps"
                     if min(census[a]["merge"]) == max(census[a]["merge"]) else
                     f"{len(census[a]['merge'])} steps, median "
                     f"{statistics.median(census[a]['merge']):.4g}, range "
                     f"{min(census[a]['merge']):.4g}-{max(census[a]['merge']):.4g}")
                  for a in sorted(census) if census[a]["merge"]]
        if merged:
            lines += [
                f"No key disappears for this arm -- every critic keeps running and "
                f"only the merge changes -- so the read-back is the merged quantity "
                f"itself. `{MERGE_SCALE}` is the factor the trainer pins the "
                f"adversarial gradient to each step: " + "; ".join(merged)
                + ". A value of 1 on every step is the cap never binding, which is "
                  "what removing it is supposed to look like.", ""]
    return lines


def write_ablation(method: str) -> bool:
    root, entry = ROOT / method, ABL_NOTES.get(method)
    if entry is None or not (root / "bundle_manifest.json").is_file():
        return False
    arm, cache_name, what = entry
    cache = ABL_CACHE / cache_name
    spec = json.loads((cache / "ablation.json").read_text()) if (cache / "ablation.json").is_file() else {}
    train_dir = arm_training_dir(cache)
    plan = arm_plan(train_dir) if train_dir else []
    trained = sorted({record["action"] for record in plan if record.get("action")}
                     or {log.stem[len("train_"):]
                         for log in (train_dir.glob("train_*.log") if train_dir else [])})

    lines = [f"# {arm}: {what}", "",
             "An ablation of the released method, not a third-party baseline. "
             "Everything the release does is kept except the one thing named above.", ""]

    lines += ["## What this arm changes", "", f"- {what}",
              f"- sampling protocol: `{spec.get('protocol', '?')}`"]
    if spec.get("k_refs_override") is not None:
        lines.append(f"- `k_refs` overridden to {spec['k_refs_override']}")
    # Two separate facts, because collapsing them into one "generator code
    # modified: no" reads as "nothing was retrained" -- which is false for every
    # arm that had to train, and is the opposite of what its own driver did.
    if trained:
        lines.append(f"- generator retrained: yes -- {', '.join(trained)}, from the "
                     f"released run's own `effective_config.json` with the switch "
                     f"above flipped (`{train_dir.name}/plan.json`, "
                     f"`{train_dir.name}/config_<action>.yaml`)")
    else:
        lines.append("- generator retrained: no -- this arm draws from checkpoints "
                     "that already existed, so only the sampler was told anything "
                     "different")
    lines.append(f"- generator source modified: "
                 f"{'yes' if spec.get('generator_modified') else 'no'}"
                 + ("" if spec.get("generator_modified") else
                    " -- a config switch, not an edit to the trainer or the model"))
    lines.append("")

    # Whether this arm samples from the release's own checkpoints decides what
    # the number means: same checkpoint isolates the sampling change, a different
    # run also carries whatever that training run differed in.
    #
    # Read per action from a drawn sample, not from the cache's `ablation.json`:
    # the sampler rewrites that file on every invocation, so for an arm sampled
    # one action at a time it records only the last one and the others vanish.
    # Compare the *run* rather than the checkpoint filename -- every retrained
    # arm's scroll is also called `last.pt`, so filenames alone would report a
    # retrained arm as sharing the release's checkpoint.
    # An action can be drawn and still carry no cell: the cell build declines
    # keystroke unconditionally, whether or not this arm has a checkpoint for it.
    # Listing such an action here would put five actions above a 24-cell table and
    # leave the reader to work out which one vanished, so the declined ones are
    # held back and explained on their own terms below.
    declined = {item.split("/")[-1]
                for item in json.loads((root / "bundle_manifest.json").read_text())
                .get("declined_by_method", [])}

    pins = json.loads(Path("released_generators.json").read_text())["runs"]
    same, differ, declined_but_drawn = [], [], []
    for action in sorted(ACTIONS):
        drawn = sorted(cache.glob(f"user_*/{action}/*/sample_0000.npz"))[:1]
        if not drawn:
            continue
        if action in declined:
            declined_but_drawn.append(action)
            continue
        try:
            import numpy as np
            meta = json.loads(str(np.load(drawn[0], allow_pickle=True)["metadata_json"]))
        except Exception:
            continue
        run = Path(meta.get("run_dir", "")).name
        pin = pins.get(action, {}).get("fewshot_adv", {})
        (same if run and run == pin.get("run") else differ).append(action)
    lines += ["## What stays identical to the release", ""]
    if same:
        lines.append(f"- {', '.join(same)}: the same checkpoint the release samples from, "
                     f"so the arm differs from it only in what the sampler was told.")
    lines += ["- the carrier's update timing, its no-contact sentinel, its clock column "
              "and the genuine events are the release's and left byte-identical.", ""]

    # A different training run and an untouched action are warnings about what
    # the number carries, not reassurances -- listing them under "stays identical
    # to the release" said the opposite of what they mean, so they get their own
    # heading.
    confounds = []
    if differ:
        confounds.append(
            f"- {', '.join(differ)}: drawn from "
            + ("this arm's own training run" if trained
               else "this protocol's own training run")
            + ", not the release's, so the number carries that run's training as "
              "well as the change above. Run-to-run variation between two "
              "trainings of the same config is not measured anywhere in this "
              "project, so it cannot be subtracted out.")
    # An action this arm never trained is absent from the table for a reason the
    # bundle manifest cannot express: it lands in the same `declined_by_method`
    # list as keystroke, which is out of reach of any generator ablation by
    # construction.  Reading both as "the method does not model this" would be
    # wrong about the first and right about the second, so they are separated.
    absent = [a for a in sorted(ACTIONS)
              if a not in trained and a not in same and a not in differ
              and a not in declined_but_drawn]
    not_retrained = [a for a in absent if a not in ALWAYS_DECLINED]
    if trained and not_retrained:
        confounds.append(
            f"- {', '.join(not_retrained)}: not retrained for this arm and "
            f"therefore carrying no number -- a different thing from an action the "
            f"method declines to model, though the bundle manifest lists both the "
            f"same way. The arm's conclusion is measured on {', '.join(trained)} "
            f"only; whether it extrapolates to the rest is untested.")
    if trained and [a for a in absent if a in ALWAYS_DECLINED]:
        confounds.append(
            "- keystroke: out of reach of this or any generator ablation. Its "
            "released fake IMU is written by the analytic adapter "
            "(`diffusion_used: false`, generator source `keystroke_imu_pulse.py`) "
            "and never passes through the diffusion generator, so "
            "`build_ablation_cache_baseline.py` declines it in `ALWAYS_DECLINED`.")
    if confounds:
        lines += ["## Confounds", ""] + confounds + [""]

    if train_dir:
        lines += training_evidence(train_dir, plan)

    proof = effective_setting(cache)
    if proof:
        if train_dir:
            lines += ["The sampler is not this arm's variable and its read-back "
                      f"says only that much: {proof}.", ""]
        else:
            lines += ["## The change took effect", "",
                      f"Read back from the cache rather than assumed: {proof}.", ""]

    if spec.get("k_refs_note"):
        lines += ["## What this arm does not measure", "", spec["k_refs_note"] + ".", ""]

    missing = spec.get("actions_without_checkpoint") or []
    if missing:
        lines += [f"**Actions without a checkpoint.** {', '.join(missing)}. These carry no "
                  "number rather than a substituted one.", ""]

    # Only a declined action that was actually drawn needs saying: one this arm
    # never sampled is already absent from the list above, so its action count
    # and its cell count agree without any explanation.  A drawn-then-declined
    # action is the case that silently breaks that arithmetic.
    if declined_but_drawn:
        names = ", ".join(declined_but_drawn)
        ckpt = (f" -- `{spec.get('protocol', '?')}` has a {names} checkpoint of its own -- "
                if all(a in (spec.get("checkpoints") or {}) for a in declined_but_drawn)
                else ", ")
        why = ("the released keystroke fake IMU is written by the analytic adapter "
               "(`diffusion_used: false`, generator source `keystroke_imu_pulse.py`) and "
               "never passes through the diffusion generator, so no ablation of that "
               "generator can move a keystroke cell. `build_ablation_cache_baseline.py` "
               "declines it in `ALWAYS_DECLINED`;"
               if declined_but_drawn == ["keystroke"] else
               "the cell build declined it, so those draws were never scored;")
        lines += [f"**Declined by construction.** {names}. This arm did draw {names} "
                  f"samples{ckpt}but {why} it carries no number rather than a substituted "
                  f"one, so the table below covers the {len(same) + len(differ)} actions "
                  "listed above.", ""]

    try:
        cells, gaps, _ = collect(method)
    except SystemExit:
        cells, gaps = {}, []
    if cells:
        values = [v for k, v in cells.items() if k[1] == "imu_only"]
        if values:
            lines += ["## Result", "",
                      f"| Modality | Cells | Mean FAR | Cells >= 0.60 |", "|---|---|---|---|",
                      f"| IMU | {len(values)} | {statistics.mean(values):.3f} | "
                      f"{sum(1 for v in values if v >= 0.6)} |", "",
                      "FAR at the development-selected FRR = 5% threshold, against a "
                      "detector trained on this arm. Inertial channel only: this arm "
                      "changes no touch coordinate.", ""]
        if gaps:
            lines += [f"*{len(gaps)} cell(s) not yet computed.*", ""]

    (root / "README.md").write_text("\n".join(lines))
    return True


def main() -> None:
    written = []
    for path in sorted(ROOT.glob("*")):
        if path.name.startswith("_"):
            continue
        if write(path.name) or write_ablation(path.name):
            written.append(path.name)
    print(f"wrote {len(written)} baseline notes:")
    for name in written:
        print(f"  {ROOT / name / 'README.md'}")
    skipped = [p.name for p in sorted(ROOT.glob("*"))
               if not p.name.startswith("_") and p.name not in written
               and (p / "bundle_manifest.json").is_file()]
    if skipped:
        print(f"no note defined for: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
