"""Strict 25-pair merge, report/gallery generation, and formal audit."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from detectors.benchmark_runner import (
    BenchmarkConfig,
    _macro_rows,
    _write_csv,
    _write_macro_markdown,
    _write_operating_markdown,
    _write_report,
)
from detectors.deep_pad import ACTIONS, DEEP_DETECTORS, DeepTrainConfig
from detectors.feature_pad import ALLOWED_DETECTORS
from detectors.pair_runner import (
    FORMAL_MIN_BOOTSTRAP,
    FORMAL_MIN_EPOCHS,
    PAIR_SCHEMA,
    _build_deep_pair_input_identity,
    _config_digest,
    audit_protocol_result,
    sha256_file,
    stable_pair_seed,
)


MERGE_SCHEMA = "trajectory_pad_25pair_merge_v1"
MERGE_AUDIT_SCHEMA = "trajectory_pad_25pair_independent_audit_v1"


def expected_pairs() -> Tuple[Tuple[str, str, str], ...]:
    return tuple(
        (action, family, detector)
        for action in ACTIONS
        for family, detectors in (
            ("feature_pad", ALLOWED_DETECTORS),
            ("deep_pad", DEEP_DETECTORS),
        )
        for detector in detectors
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(str(temporary), str(target))


def _read_and_audit_pair(
    experiment_root: Path, action: str, family: str, detector: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    pair_root = experiment_root / "pairs" / action / family / detector
    manifest_path = pair_root / "pair_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("missing pair manifest: %s" % manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != PAIR_SCHEMA
        or manifest.get("status") != "complete"
        or (manifest.get("action"), manifest.get("family"), manifest.get("detector"))
        != (action, family, detector)
    ):
        raise ValueError("pair manifest identity/status mismatch: %s" % manifest_path)
    cfg = manifest["config"]
    if _config_digest(cfg) != manifest.get("config_sha256"):
        raise ValueError("pair config digest mismatch")
    probe = cfg.get("batch_probe")
    if probe is not None:
        probe_path = Path(probe["path"])
        if not probe_path.is_file() or sha256_file(probe_path) != probe.get("sha256"):
            raise ValueError("pair batch-probe artifact missing/hash drift")
    replicates = (
        int(cfg["feature_bootstrap_replicates"])
        if family == "feature_pad" else int(cfg["deep_train"]["bootstrap_replicates"])
    )
    dataset_file = Path(manifest["dataset_file"])
    split_file = Path(manifest["fake_user_split"])
    if (
        not dataset_file.is_file()
        or sha256_file(dataset_file) != manifest.get("dataset_sha256")
    ):
        raise ValueError("pair source dataset missing/hash drift")
    if (
        not split_file.is_file()
        or sha256_file(split_file) != manifest.get("fake_user_split_sha256")
    ):
        raise ValueError("pair fake-user split missing/hash drift")
    expected_deep_identity = (
        _build_deep_pair_input_identity(
            dataset_file=dataset_file,
            dataset_sha256=manifest["dataset_sha256"],
            fake_user_split=split_file,
            fake_user_split_sha256=manifest["fake_user_split_sha256"],
            real_hash_seed=int(cfg["real_hash_seed"]),
            action=action,
            detector=detector,
            pair_config=cfg,
            pair_config_sha256=manifest["config_sha256"],
        )
        if family == "deep_pad" else None
    )
    result_dir = Path(manifest["result_dir"])
    if result_dir.resolve() != (pair_root / "result").resolve():
        raise ValueError("pair result directory escapes canonical pair root")
    audited = audit_protocol_result(
        result_dir, action=action, family=family,
        detector=detector, expected_bootstrap_replicates=replicates,
        dataset_file=dataset_file, fake_user_split=split_file,
        real_hash_seed=int(cfg["real_hash_seed"]),
        expected_dataset_sha256=manifest["dataset_sha256"],
        expected_fake_user_split_sha256=manifest["fake_user_split_sha256"],
        expected_bootstrap_seed=int(cfg["seed"]) + (
            31 if family == "feature_pad" else 17
        ),
        expected_deep_run_identity=expected_deep_identity,
    )
    summary = audited["summary"]
    if family == "feature_pad":
        if (
            int(summary.get("random_state", -1)) != int(cfg["seed"])
            or summary.get("model_params", {}) != cfg.get("feature_model_params", {})
        ):
            raise ValueError("Feature pair summary seed/model config drift")
    elif (
        summary.get("train_config") != cfg.get("deep_train")
        or summary.get("model_params", {}) != cfg.get("deep_model_params", {})
    ):
        raise ValueError("Deep pair summary train/model config drift")
    if audited["artifact_hashes"] != manifest.get("artifact_hashes"):
        raise ValueError("pair artifact hash drift: %s" % manifest_path)
    if audited["dataset_relink_audit"] != manifest.get("dataset_relink_audit"):
        raise ValueError("pair dataset relink audit drift: %s" % manifest_path)
    plot = Path(manifest["plot"])
    if plot.resolve() != (pair_root / "test_fa_frr.png").resolve():
        raise ValueError("pair plot path escapes canonical pair root")
    if not plot.is_file() or sha256_file(plot) != manifest.get("plot_sha256"):
        raise ValueError("pair plot missing/hash drift: %s" % plot)
    if manifest.get("operating_rows") != audited["rows"] or len(audited["rows"]) != 2:
        raise ValueError("pair manifest operating rows differ from independently audited scores")
    return manifest, audited, manifest_path


def merge_and_audit_pairs(experiment_root: Path, require_formal: bool = True) -> Dict[str, Any]:
    """Require all 25 independent pairs, then atomically derive final outputs."""

    root = Path(experiment_root)
    merged_root = root / "merged"
    final_manifest = merged_root / "benchmark_manifest.json"
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any], Path]] = []
    for action, family, detector in expected_pairs():
        pairs.append(_read_and_audit_pair(root, action, family, detector))
    if len(pairs) != 25:
        raise AssertionError("formal merge requires exactly 25 pairs")

    identities = {
        (manifest["action"], manifest["family"], manifest["detector"])
        for manifest, _, _ in pairs
    }
    if identities != set(expected_pairs()):
        raise ValueError("pair set has duplicates or omissions")
    split_hashes = {manifest["fake_user_split_sha256"] for manifest, _, _ in pairs}
    split_paths = {manifest["fake_user_split"] for manifest, _, _ in pairs}
    if len(split_hashes) != 1 or len(split_paths) != 1:
        raise ValueError("pairs do not share one fixed fake-user split")
    for action in ACTIONS:
        action_pairs = [manifest for manifest, _, _ in pairs if manifest["action"] == action]
        if len({item["dataset_sha256"] for item in action_pairs}) != 1:
            raise ValueError("%s detector pairs use different dataset bytes" % action)
        if len({item["dataset_file"] for item in action_pairs}) != 1:
            raise ValueError("%s detector pairs use different dataset paths" % action)

    first_config = pairs[0][0]["config"]
    base_seed = int(first_config.get("base_seed", first_config["seed"]))
    deep_template = dict(first_config["deep_train"])
    deep_template.pop("seed", None)
    deep_template.pop("batch_size", None)
    pair_seeds = {}
    batch_sizes = {}
    for manifest, audited, _ in pairs:
        cfg = manifest["config"]
        identity = (manifest["action"], manifest["family"], manifest["detector"])
        if int(cfg.get("base_seed", cfg["seed"])) != base_seed:
            raise ValueError("pair base seed drift")
        if int(cfg["real_hash_seed"]) != int(first_config["real_hash_seed"]):
            raise ValueError("pair real hash seed drift")
        if int(cfg["feature_bootstrap_replicates"]) != int(first_config["feature_bootstrap_replicates"]):
            raise ValueError("feature bootstrap config drift")
        candidate_template = dict(cfg["deep_train"])
        candidate_template.pop("seed", None)
        candidate_template.pop("batch_size", None)
        if candidate_template != deep_template:
            raise ValueError("deep training template drift across pairs")
        pair_seed = int(cfg["seed"])
        if int(cfg["deep_train"]["seed"]) != pair_seed:
            raise ValueError("pair/deep seed mismatch")
        pair_seeds["/".join(identity)] = pair_seed
        batch_sizes["/".join(identity)] = int(cfg["deep_train"]["batch_size"])
        if require_formal:
            if cfg.get("formal_protocol") is not True:
                raise ValueError("formal merge rejects a non-formal/quick pair")
            expected_seed = stable_pair_seed(base_seed, *identity)
            if pair_seed != expected_seed:
                raise ValueError("formal pair seed does not match stable identity policy")
            if int(cfg["deep_train"]["epochs"]) != FORMAL_MIN_EPOCHS:
                raise ValueError("formal merge requires configured epochs == 40")
            if int(cfg["deep_train"]["patience"]) != 0:
                raise ValueError("formal merge requires patience == 0")
            if (
                int(cfg["feature_bootstrap_replicates"]) != FORMAL_MIN_BOOTSTRAP
                or int(cfg["deep_train"]["bootstrap_replicates"]) != FORMAL_MIN_BOOTSTRAP
            ):
                raise ValueError("formal merge requires bootstrap replicates == 500")
            if manifest["family"] == "deep_pad":
                training = audited.get("deep_training_audit", {})
                if (
                    int(audited.get("summary", {}).get("last_epoch", -1))
                    != FORMAL_MIN_EPOCHS
                    or int(training.get("history_epoch_count", -1))
                    != FORMAL_MIN_EPOCHS
                    or int(training.get("history_last_epoch", -1))
                    != FORMAL_MIN_EPOCHS
                ):
                    raise ValueError(
                        "formal merge requires exactly 40 completed Deep history epochs"
                    )
                probe = cfg.get("batch_probe")
                if (
                    not probe
                    or int(probe.get("selected_batch_size", -1)) != int(cfg["deep_train"]["batch_size"])
                    or probe.get("truncation") is not False
                    or probe.get("resampling") is not False
                ):
                    raise ValueError("formal Deep pair lacks matching no-truncation batch probe")
    if require_formal and len(set(pair_seeds.values())) != 25:
        raise ValueError("formal pair RNG seeds must be unique across all 25 identities")

    rows: List[Dict[str, Any]] = []
    pair_manifest_hashes: Dict[str, str] = {}
    feature_params: Dict[str, Dict[str, Any]] = {}
    deep_params: Dict[str, Dict[str, Any]] = {}
    for manifest, audited, manifest_path in pairs:
        action, family, detector = (
            manifest["action"], manifest["family"], manifest["detector"]
        )
        rows.extend(audited["rows"])
        key = "%s/%s/%s" % (action, family, detector)
        pair_manifest_hashes[key] = sha256_file(manifest_path)
        params = dict(manifest["config"][
            "feature_model_params" if family == "feature_pad" else "deep_model_params"
        ])
        target = feature_params if family == "feature_pad" else deep_params
        previous = target.setdefault(detector, params)
        if previous != params:
            raise ValueError("model parameter drift across actions: %s" % detector)
        plot_target = merged_root / "plots" / action / (detector + ".png")
        _atomic_copy(Path(manifest["plot"]), plot_target)

    row_keys = {
        (row["action"], row["detector_family"], row["detector"], row["operating_point"])
        for row in rows
    }
    expected_row_keys = {
        (action, family, detector, point)
        for action, family, detector in expected_pairs()
        for point in ("eer", "val_frr_le_5pct")
    }
    if len(rows) != 50 or row_keys != expected_row_keys:
        raise ValueError("formal merge requires exactly 50 unique operating rows")
    rows.sort(key=lambda row: (
        ACTIONS.index(row["action"]),
        0 if row["detector_family"] == "feature_pad" else 1,
        (ALLOWED_DETECTORS if row["detector_family"] == "feature_pad" else DEEP_DETECTORS).index(row["detector"]),
        0 if row["operating_point"] == "eer" else 1,
    ))
    macro = _macro_rows(rows)
    if len(macro) != 10:
        raise ValueError("formal merge requires 5 detectors × 2 operating points macro rows")

    per_action_fields = (
        "action", "detector_family", "detector", "operating_point",
        "threshold_from_validation", "validation_fa", "validation_frr", "validation_auc",
        "test_fa", "test_frr", "test_auc",
        "test_fa_ci95_low", "test_fa_ci95_high",
        "test_frr_ci95_low", "test_frr_ci95_high",
        "test_auc_ci95_low", "test_auc_ci95_high",
        "n_test_real", "n_test_fake", "best_epoch",
    )
    macro_fields = (
        "detector_family", "detector", "operating_point", "n_actions",
        "macro_validation_fa", "macro_validation_frr", "macro_validation_auc",
        "macro_test_fa", "macro_test_frr", "macro_test_auc",
    )
    per_action_path = merged_root / "per_action_detector.csv"
    macro_path = merged_root / "macro_by_detector.csv"
    macro_md = merged_root / "macro_by_detector.md"
    report_path = merged_root / "benchmark_report.md"
    _write_csv(per_action_path, rows, per_action_fields)
    _write_csv(macro_path, macro, macro_fields)
    _write_macro_markdown(macro_md, macro)
    by_action_root = merged_root / "summaries" / "by_action"
    by_detector_root = merged_root / "summaries" / "by_detector"
    for action in ACTIONS:
        action_rows = [row for row in rows if row["action"] == action]
        _write_csv(by_action_root / (action + ".csv"), action_rows, per_action_fields)
        _write_operating_markdown(
            by_action_root / (action + ".md"),
            action + " detector results", action_rows,
        )
    for family, detectors in (
        ("feature_pad", ALLOWED_DETECTORS), ("deep_pad", DEEP_DETECTORS)
    ):
        for detector in detectors:
            detector_rows = [
                row for row in rows
                if row["detector_family"] == family and row["detector"] == detector
            ]
            name = family + "__" + detector
            _write_csv(by_detector_root / (name + ".csv"), detector_rows, per_action_fields)
            _write_operating_markdown(
                by_detector_root / (name + ".md"), name + " results", detector_rows,
            )
    report_deep_config = dict(deep_template)
    report_deep_config["seed"] = base_seed
    report_deep_config["batch_size"] = max(
        int(manifest["config"]["deep_train"]["batch_size"])
        for manifest, _, _ in pairs if manifest["family"] == "deep_pad"
    )
    config = BenchmarkConfig(
        actions=ACTIONS,
        feature_detectors=ALLOWED_DETECTORS,
        deep_detectors=DEEP_DETECTORS,
        feature_model_params=feature_params,
        deep_model_params=deep_params,
        deep_train=DeepTrainConfig(**report_deep_config),
        feature_bootstrap_replicates=int(first_config["feature_bootstrap_replicates"]),
        seed=base_seed,
    )
    _write_report(
        report_path, rows, macro, config,
        deep_batch_size_by_pair={
            identity: size for identity, size in batch_sizes.items()
            if "/deep_pad/" in identity
        },
    )

    outputs = {
        "per_action": str(per_action_path.resolve()),
        "macro": str(macro_path.resolve()),
        "macro_markdown": str(macro_md.resolve()),
        "report": str(report_path.resolve()),
        "plots": str((merged_root / "plots").resolve()),
        "by_action": str(by_action_root.resolve()),
        "by_detector": str(by_detector_root.resolve()),
    }
    summary_files = sorted(
        path for path in (merged_root / "summaries").rglob("*") if path.is_file()
    )
    summary_hashes = {
        str(path.relative_to(merged_root)): sha256_file(path) for path in summary_files
    }
    if len(summary_hashes) != 20:
        raise ValueError("formal summaries require exactly 20 by-action/by-detector files")
    manifest = {
        "schema_version": MERGE_SCHEMA,
        "status": "complete",
        "formal_protocol": bool(require_formal),
        "n_pairs": 25,
        "n_operating_rows": 50,
        "n_macro_rows": 10,
        "actions": list(ACTIONS),
        "feature_detectors": list(ALLOWED_DETECTORS),
        "deep_detectors": list(DEEP_DETECTORS),
        "fake_user_split_sha256": next(iter(split_hashes)),
        "fake_user_split": next(iter(split_paths)),
        "dataset_sha256_by_action": {
            action: next(
                manifest["dataset_sha256"] for manifest, _, _ in pairs
                if manifest["action"] == action
            ) for action in ACTIONS
        },
        "pair_manifest_sha256": pair_manifest_hashes,
        "protocol_config": {
            "base_seed": base_seed,
            "pair_seed_policy": "sha256(base_seed|action|family|detector)_uint32",
            "pair_seed_by_identity": pair_seeds,
            "batch_size_by_identity": batch_sizes,
            "real_hash_seed": int(first_config["real_hash_seed"]),
            "feature_bootstrap_replicates": int(first_config["feature_bootstrap_replicates"]),
            "deep_train_template_without_pair_seed": deep_template,
        },
        "outputs": outputs,
        "output_sha256": {
            name: sha256_file(Path(path)) for name, path in outputs.items()
            if name not in {"plots", "by_action", "by_detector"}
        },
        "summary_artifact_sha256": summary_hashes,
        "plot_count": len(list((merged_root / "plots").rglob("*.png"))),
    }
    if manifest["plot_count"] != 25:
        raise ValueError("formal gallery requires exactly 25 plots")
    _atomic_json(final_manifest, manifest)
    return manifest


def _require_csv_equals(
    path: Path,
    expected_rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Compare a persisted CSV to independently reconstructed rows exactly."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(fieldnames):
            raise ValueError("merged CSV header mismatch: %s" % path)
        observed = list(reader)
    expected = [
        {
            field: "" if row.get(field) is None else str(row.get(field))
            for field in fieldnames
        }
        for row in expected_rows
    ]
    if observed != expected:
        raise ValueError("merged CSV does not equal independently audited pair rows: %s" % path)


def audit_merged_pair_tree(
    experiment_root: Path,
    *,
    require_formal: bool = True,
    write_audit: bool = True,
) -> Dict[str, Any]:
    """Independently re-audit a completed pair tree without rewriting results.

    This is deliberately separate from :func:`merge_and_audit_pairs`: it
    reopens all 25 pair score dumps and bootstrap artifacts, reconstructs the
    50 operating rows and 10 macro rows, verifies every merged output/hash and
    copied plot, and only then commits an audit receipt.
    """

    root = Path(experiment_root)
    merged_root = root / "merged"
    manifest_path = merged_root / "benchmark_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("missing merged benchmark manifest: %s" % manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != MERGE_SCHEMA
        or manifest.get("status") != "complete"
        or int(manifest.get("n_pairs", -1)) != 25
        or int(manifest.get("n_operating_rows", -1)) != 50
        or int(manifest.get("n_macro_rows", -1)) != 10
        or tuple(manifest.get("actions", ())) != ACTIONS
        or tuple(manifest.get("feature_detectors", ())) != ALLOWED_DETECTORS
        or tuple(manifest.get("deep_detectors", ())) != DEEP_DETECTORS
    ):
        raise ValueError("merged benchmark manifest identity/count mismatch")
    if require_formal and manifest.get("formal_protocol") is not True:
        raise ValueError("independent formal audit rejects non-formal merge")

    audited_pairs: List[Tuple[Dict[str, Any], Dict[str, Any], Path]] = []
    for action, family, detector in expected_pairs():
        audited_pairs.append(_read_and_audit_pair(root, action, family, detector))
    if len(audited_pairs) != 25:
        raise AssertionError("independent audit requires exactly 25 pairs")

    pair_hashes = {
        "%s/%s/%s" % (item[0]["action"], item[0]["family"], item[0]["detector"]):
        sha256_file(item[2])
        for item in audited_pairs
    }
    if manifest.get("pair_manifest_sha256") != pair_hashes:
        raise ValueError("merged manifest pair hashes disagree with pair tree")
    split_hashes = {item[0]["fake_user_split_sha256"] for item in audited_pairs}
    split_paths = {item[0]["fake_user_split"] for item in audited_pairs}
    if (
        len(split_hashes) != 1
        or len(split_paths) != 1
        or manifest.get("fake_user_split_sha256") != next(iter(split_hashes))
        or manifest.get("fake_user_split") != next(iter(split_paths))
    ):
        raise ValueError("merged fixed user-split provenance mismatch")
    dataset_hashes = {
        action: {
            item[0]["dataset_sha256"]
            for item in audited_pairs if item[0]["action"] == action
        }
        for action in ACTIONS
    }
    if any(len(values) != 1 for values in dataset_hashes.values()):
        raise ValueError("detectors within an action do not share dataset bytes")
    if manifest.get("dataset_sha256_by_action") != {
        action: next(iter(values)) for action, values in dataset_hashes.items()
    }:
        raise ValueError("merged dataset hashes disagree with pair tree")

    if require_formal:
        base_seed_values = {
            int(item[0]["config"].get("base_seed", item[0]["config"]["seed"]))
            for item in audited_pairs
        }
        if len(base_seed_values) != 1:
            raise ValueError("formal pair base-seed drift")
        base_seed = next(iter(base_seed_values))
        pair_seed_values = set()
        for pair_manifest, audited, _ in audited_pairs:
            cfg = pair_manifest["config"]
            identity = (
                pair_manifest["action"], pair_manifest["family"], pair_manifest["detector"]
            )
            pair_seed = int(cfg["seed"])
            pair_seed_values.add(pair_seed)
            if (
                cfg.get("formal_protocol") is not True
                or pair_seed != stable_pair_seed(base_seed, *identity)
                or int(cfg["deep_train"]["seed"]) != pair_seed
                or int(cfg["deep_train"]["epochs"]) != FORMAL_MIN_EPOCHS
                or int(cfg["deep_train"]["patience"]) != 0
                or int(cfg["feature_bootstrap_replicates"]) != FORMAL_MIN_BOOTSTRAP
                or int(cfg["deep_train"]["bootstrap_replicates"]) != FORMAL_MIN_BOOTSTRAP
            ):
                raise ValueError("formal pair config/seed/budget drift: %s" % (identity,))
            if pair_manifest["family"] == "deep_pad":
                training = audited.get("deep_training_audit", {})
                probe = cfg.get("batch_probe")
                if (
                    int(audited["summary"].get("last_epoch", -1)) != FORMAL_MIN_EPOCHS
                    or int(training.get("history_epoch_count", -1)) != FORMAL_MIN_EPOCHS
                    or int(training.get("history_last_epoch", -1)) != FORMAL_MIN_EPOCHS
                    or not probe
                    or int(probe.get("selected_batch_size", -1))
                    != int(cfg["deep_train"]["batch_size"])
                    or probe.get("truncation") is not False
                    or probe.get("resampling") is not False
                ):
                    raise ValueError("formal Deep completion/probe drift: %s" % (identity,))
        if len(pair_seed_values) != 25:
            raise ValueError("formal pair seeds are not unique")

    rows: List[Dict[str, Any]] = []
    for _, audited, _ in audited_pairs:
        rows.extend(audited["rows"])
    rows.sort(key=lambda row: (
        ACTIONS.index(row["action"]),
        0 if row["detector_family"] == "feature_pad" else 1,
        (ALLOWED_DETECTORS if row["detector_family"] == "feature_pad" else DEEP_DETECTORS).index(row["detector"]),
        0 if row["operating_point"] == "eer" else 1,
    ))
    if len(rows) != 50:
        raise ValueError("independent audit did not reconstruct exactly 50 rows")
    macro = _macro_rows(rows)
    if len(macro) != 10:
        raise ValueError("independent audit did not reconstruct exactly 10 macro rows")
    per_action_fields = (
        "action", "detector_family", "detector", "operating_point",
        "threshold_from_validation", "validation_fa", "validation_frr", "validation_auc",
        "test_fa", "test_frr", "test_auc",
        "test_fa_ci95_low", "test_fa_ci95_high",
        "test_frr_ci95_low", "test_frr_ci95_high",
        "test_auc_ci95_low", "test_auc_ci95_high",
        "n_test_real", "n_test_fake", "best_epoch",
    )
    macro_fields = (
        "detector_family", "detector", "operating_point", "n_actions",
        "macro_validation_fa", "macro_validation_frr", "macro_validation_auc",
        "macro_test_fa", "macro_test_frr", "macro_test_auc",
    )
    _require_csv_equals(merged_root / "per_action_detector.csv", rows, per_action_fields)
    _require_csv_equals(merged_root / "macro_by_detector.csv", macro, macro_fields)

    by_action_root = merged_root / "summaries" / "by_action"
    by_detector_root = merged_root / "summaries" / "by_detector"
    for action in ACTIONS:
        _require_csv_equals(
            by_action_root / (action + ".csv"),
            [row for row in rows if row["action"] == action],
            per_action_fields,
        )
    for family, detectors in (
        ("feature_pad", ALLOWED_DETECTORS), ("deep_pad", DEEP_DETECTORS)
    ):
        for detector in detectors:
            _require_csv_equals(
                by_detector_root / (family + "__" + detector + ".csv"),
                [
                    row for row in rows
                    if row["detector_family"] == family
                    and row["detector"] == detector
                ],
                per_action_fields,
            )

    expected_outputs = {
        "per_action": merged_root / "per_action_detector.csv",
        "macro": merged_root / "macro_by_detector.csv",
        "macro_markdown": merged_root / "macro_by_detector.md",
        "report": merged_root / "benchmark_report.md",
        "plots": merged_root / "plots",
        "by_action": by_action_root,
        "by_detector": by_detector_root,
    }
    if set(manifest.get("outputs", {})) != set(expected_outputs):
        raise ValueError("merged output declaration set mismatch")
    for name, expected_path in expected_outputs.items():
        declared = Path(manifest.get("outputs", {}).get(name, ""))
        if declared.resolve() != expected_path.resolve() or not expected_path.exists():
            raise ValueError("merged output path/existence mismatch: %s" % name)
        if name not in {"plots", "by_action", "by_detector"}:
            if manifest.get("output_sha256", {}).get(name) != sha256_file(expected_path):
                raise ValueError("merged output hash mismatch: %s" % name)

    observed_summary_hashes = {
        str(path.relative_to(merged_root)): sha256_file(path)
        for path in sorted(
            path for path in (merged_root / "summaries").rglob("*")
            if path.is_file()
        )
    }
    expected_summary_paths = {
        "summaries/by_action/%s.%s" % (action, suffix)
        for action in ACTIONS for suffix in ("csv", "md")
    } | {
        "summaries/by_detector/%s__%s.%s" % (family, detector, suffix)
        for family, detectors in (
            ("feature_pad", ALLOWED_DETECTORS), ("deep_pad", DEEP_DETECTORS)
        )
        for detector in detectors for suffix in ("csv", "md")
    }
    if (
        set(observed_summary_hashes) != expected_summary_paths
        or manifest.get("summary_artifact_sha256") != observed_summary_hashes
    ):
        raise ValueError("merged summary artifact set/hash mismatch")

    expected_plot_paths = set()
    for pair_manifest, _, _ in audited_pairs:
        action, detector = pair_manifest["action"], pair_manifest["detector"]
        target = merged_root / "plots" / action / (detector + ".png")
        expected_plot_paths.add(target.resolve())
        if not target.is_file() or sha256_file(target) != pair_manifest["plot_sha256"]:
            raise ValueError("merged plot differs from audited pair plot: %s" % target)
    observed_plot_paths = {path.resolve() for path in (merged_root / "plots").rglob("*.png")}
    if (
        observed_plot_paths != expected_plot_paths
        or int(manifest.get("plot_count", -1)) != 25
    ):
        raise ValueError("merged gallery must contain exactly the expected 25 plots")

    audit = {
        "schema_version": MERGE_AUDIT_SCHEMA,
        "status": "passed",
        "formal_protocol": bool(require_formal),
        "experiment_root": str(root.resolve()),
        "merged_manifest": str(manifest_path.resolve()),
        "merged_manifest_sha256": sha256_file(manifest_path),
        "n_reaudited_pairs": 25,
        "n_recomputed_operating_rows": 50,
        "n_recomputed_macro_rows": 10,
        "n_verified_plots": 25,
        "threshold_selection_pool": "validation_only",
        "formal_epochs": FORMAL_MIN_EPOCHS if require_formal else None,
        "formal_patience": 0 if require_formal else None,
        "formal_bootstrap_replicates": FORMAL_MIN_BOOTSTRAP if require_formal else None,
        "checks": [
            "25 pair manifests and artifacts independently reopened and rehashed",
            "every validation/test score row relinked to the current source dataset",
            "validation thresholds and fixed-threshold val/test metrics recomputed from scores",
            "500 fixed-seed user-bootstrap replicate arrays and confidence intervals exactly recomputed",
            "50 operating rows and 10 macro rows reconstructed and matched to CSV",
            "20 by-action/by-detector summary artifacts verified",
            "all declared merged output hashes verified",
            "25 copied plots matched byte-for-byte to audited pair plots",
        ],
    }
    if write_audit:
        _atomic_json(merged_root / "benchmark_audit.json", audit)
    return audit


__all__ = [
    "MERGE_SCHEMA", "MERGE_AUDIT_SCHEMA", "expected_pairs",
    "merge_and_audit_pairs", "audit_merged_pair_tree",
]
