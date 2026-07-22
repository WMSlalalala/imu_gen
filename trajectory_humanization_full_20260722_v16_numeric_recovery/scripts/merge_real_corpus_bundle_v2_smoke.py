#!/usr/bin/env python3
"""Fail-closed aggregate for the five independent real-corpus smoke runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
FEATURE_DIMS = {"tap": 24, "scroll": 24, "swipe": 24, "pinch": 49, "keystroke": 34}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(str(temporary), str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for action in ACTIONS:
        path = args.input_dir / (action + ".json")
        item = json.loads(path.read_text(encoding="utf-8"))
        if (
            item.get("schema_version") != "real_corpus_bundle_v2_smoke_v1"
            or item.get("status") != "passed"
            or item.get("formal_result") is not False
            or item.get("action") != action
            or item.get("bundle_schema") != "trajectory_pad_bundle_v2"
            or item.get("feature_schema_version")
            != "trajectory_features_v2_ahb_table6_hmog_real_up"
            or item.get("feature_shape") != [12, FEATURE_DIMS[action]]
            or item.get("feature_all_finite") is not True
        ):
            raise ValueError("invalid real-corpus smoke report: %s" % path)
        source = Path(item["source"])
        bundle = Path(item["bundle"])
        if sha256_file(source) != item["source_sha256"]:
            raise ValueError("source hash drift: %s" % source)
        if sha256_file(bundle) != item["bundle_sha256"]:
            raise ValueError("bundle hash drift: %s" % bundle)
        for detector in ("tcn", "transformer"):
            if (
                item.get("deep_forward", {}).get(detector, {}).get("all_finite") is not True
                or int(item["deep_forward"][detector].get("n_scores", -1)) != 12
            ):
                raise ValueError("Deep smoke failed for %s/%s" % (action, detector))
        if item.get("feature_linear_svm", {}).get("checkpoint_selection_pool") != "validation_only":
            raise ValueError("Feature smoke threshold provenance drift: %s" % action)
        reports.append(item)

    output = {
        "schema_version": "real_corpus_bundle_v2_five_action_smoke_summary_v1",
        "status": "passed",
        "formal_result": False,
        "metric_interpretation": (
            "pipeline-only: label-1 rows are exact mirrors of real rows, not generated fake; "
            "FA/FRR/AUC from this smoke must not be reported as generator quality"
        ),
        "actions": list(ACTIONS),
        "n_actions": 5,
        "source_event_count_by_action": {
            item["action"]: int(item["source_event_count"]) for item in reports
        },
        "source_event_count_total": sum(int(item["source_event_count"]) for item in reports),
        "bundle_schema": "trajectory_pad_bundle_v2",
        "feature_schema_version": "trajectory_features_v2_ahb_table6_hmog_real_up",
        "feature_dimension_by_action": FEATURE_DIMS,
        "feature_adapter_bundle_roundtrip": "passed_all_five",
        "feature_linear_svm_validation_only_flow": "passed_all_five",
        "deep_tcn_forward_finite": "passed_all_five",
        "deep_transformer_forward_finite": "passed_all_five",
        "report_sha256_by_action": {
            action: sha256_file(args.input_dir / (action + ".json")) for action in ACTIONS
        },
        "bundle_sha256_by_action": {
            item["action"]: item["bundle_sha256"] for item in reports
        },
    }
    summary_json = args.input_dir / "summary.json"
    summary_md = args.input_dir / "summary.md"
    atomic_text(
        summary_json,
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    rows = [
        "# Five-action real-corpus bundle v2 smoke",
        "",
        "- status: **passed**",
        "- scope: adapter → numeric bundle v2 round-trip → Feature linear SVM validation-only flow → raw TCN/Transformer finite forward",
        "- formal result: **no**",
        "- critical warning: label-1 rows are exact real-row mirrors, not generator fake. These smoke FA/FRR/AUC values are scientifically meaningless and are not generator results.",
        "",
        "| action | authoritative real events | feature dim | bundle v2 | Feature flow | TCN finite | Transformer finite |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for item in reports:
        rows.append(
            "| {action} | {count} | {dim} | pass | pass | pass | pass |".format(
                action=item["action"], count=item["source_event_count"],
                dim=FEATURE_DIMS[item["action"]],
            )
        )
    rows.extend([
        "",
        "Total authoritative events audited through the source adapters: **%d**." % output["source_event_count_total"],
        "",
    ])
    atomic_text(summary_md, "\n".join(rows))
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
