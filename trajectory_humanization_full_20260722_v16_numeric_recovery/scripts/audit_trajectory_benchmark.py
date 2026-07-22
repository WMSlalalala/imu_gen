#!/usr/bin/env python3
"""Fail-closed audit of a completed feature + raw Deep PAD result tree."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.deep_pad import ACTIONS, DEEP_DETECTORS
from detectors.feature_pad import ALLOWED_DETECTORS, fa_frr_curve, operating_metrics


def close_dict(actual, expected, context):
    for key in ("threshold", "fa", "frr", "auc"):
        if not np.isclose(float(actual[key]), float(expected[key]), rtol=0.0, atol=1e-12):
            raise AssertionError("%s metric mismatch: %s" % (context, key))
    for key in ("n_real", "n_fake"):
        if int(actual[key]) != int(expected[key]):
            raise AssertionError("%s count mismatch: %s" % (context, key))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_dir
    manifest = json.loads((root / "benchmark_manifest.json").read_text(encoding="utf-8"))
    actions = tuple(manifest["actions"])
    feature = tuple(manifest["feature_detectors"])
    deep = tuple(manifest["deep_detectors"])
    if manifest["status"] != "complete" or set(actions) - set(ACTIONS):
        raise AssertionError("invalid completion manifest")
    if set(feature) - set(ALLOWED_DETECTORS) or set(deep) - set(DEEP_DETECTORS):
        raise AssertionError("unknown detector in manifest")

    checked = []
    for action in actions:
        for family, detectors in (("feature_pad", feature), ("deep_pad", deep)):
            for detector in detectors:
                detector_root = root / family / action / detector
                summary = json.loads((detector_root / "summary.json").read_text(encoding="utf-8"))
                if summary["score_direction"] != "fake_high" or summary["acceptance_rule"] != "score < threshold":
                    raise AssertionError("score semantics mismatch")
                if summary["threshold_selection_pool"] != "validation_only":
                    raise AssertionError("threshold was not validation-only")
                if family == "deep_pad":
                    if summary["checkpoint_selection_pool"] != "validation_only":
                        raise AssertionError("checkpoint was not validation-only")
                    if summary["uses_critic"] or summary["uses_selector"]:
                        raise AssertionError("deep detector improperly uses critic/selector")
                    for path in summary["checkpoint_paths"].values():
                        if not Path(path).exists():
                            raise AssertionError("missing checkpoint: %s" % path)

                with np.load(detector_root / "score_dump.npz", allow_pickle=False) as scores:
                    for pool in ("val", "test"):
                        labels = scores[pool + "_label"]
                        values = scores[pool + "_score"]
                        if not np.all(np.isfinite(values)):
                            raise AssertionError("non-finite score")
                        for point in ("eer", "val_frr_le_5pct"):
                            manual = operating_metrics(labels, values, summary["thresholds"][point])
                            expected = summary["validation_metrics" if pool == "val" else "test_metrics"][point]
                            close_dict(manual, expected, "%s/%s/%s/%s/%s" % (family, action, detector, pool, point))
                with np.load(detector_root / "curves.npz", allow_pickle=False) as curves, np.load(
                    detector_root / "score_dump.npz", allow_pickle=False
                ) as scores:
                    for pool in ("val", "test"):
                        manual = fa_frr_curve(scores[pool + "_label"], scores[pool + "_score"])
                        for key in ("threshold", "fa", "frr"):
                            np.testing.assert_array_equal(curves[pool + "_" + key], manual[key])
                for name in ("bootstrap_summary.json", "bootstrap_replicates.npz"):
                    if not (detector_root / name).exists():
                        raise AssertionError("missing bootstrap output: %s" % (detector_root / name))
                if not (root / "plots" / action / (detector + ".png")).exists():
                    raise AssertionError("missing test curve plot")
                checked.append({"action": action, "family": family, "detector": detector})

    with (root / "per_action_detector.csv").open(encoding="utf-8") as handle:
        per_rows = list(csv.DictReader(handle))
    with (root / "macro_by_detector.csv").open(encoding="utf-8") as handle:
        macro_rows = list(csv.DictReader(handle))
    expected_detector_pairs = len(actions) * (len(feature) + len(deep))
    if len(checked) != expected_detector_pairs or len(per_rows) != expected_detector_pairs * 2:
        raise AssertionError("per-action detector row count mismatch")
    if len(macro_rows) != (len(feature) + len(deep)) * 2:
        raise AssertionError("macro row count mismatch")
    for action in actions:
        for suffix in (".csv", ".md"):
            if not (root / "summaries" / "by_action" / (action + suffix)).exists():
                raise AssertionError("missing per-action summary")
    for family, detectors in (("feature_pad", feature), ("deep_pad", deep)):
        for detector in detectors:
            for suffix in (".csv", ".md"):
                name = family + "__" + detector + suffix
                if not (root / "summaries" / "by_detector" / name).exists():
                    raise AssertionError("missing per-detector summary")
    if not (root / "macro_by_detector.md").exists():
        raise AssertionError("missing macro Markdown")

    audit = {
        "schema_version": "trajectory_benchmark_audit_v1",
        "passed": True,
        "n_actions": len(actions),
        "n_action_detector_pairs": len(checked),
        "n_per_action_operating_rows": len(per_rows),
        "n_macro_rows": len(macro_rows),
        "n_plots": len(list((root / "plots").rglob("*.png"))),
        "checks": [
            "fake-high score and strict acceptance boundary",
            "validation-only checkpoint/threshold provenance",
            "fixed-threshold val/test metrics exactly recomputed",
            "full val/test FA-FRR curves exactly recomputed",
            "user-level bootstrap artifacts present",
            "deep best/last checkpoints present",
            "critic/selector absent from Deep PAD",
            "per-action/per-detector/macro CSV and Markdown summaries present",
        ],
    }
    (root / "benchmark_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    (root / "benchmark_audit.md").write_text(
        "# Benchmark audit\n\n- passed: true\n- actions: %d\n- action-detector pairs: %d\n- operating rows: %d\n- macro rows: %d\n- plots: %d\n"
        % (audit["n_actions"], audit["n_action_detector_pairs"], audit["n_per_action_operating_rows"], audit["n_macro_rows"], audit["n_plots"]),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
