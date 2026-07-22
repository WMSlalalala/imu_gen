"""Unified clean-room feature PAD and raw Deep PAD evaluation runner."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detectors.deep_pad import (
    ACTIONS,
    DEEP_DETECTORS,
    DeepTrainConfig,
    RawTrajectoryRecord,
    load_raw_sequence_bundle,
    run_deep_pad_protocol,
)
from detectors.feature_pad import (
    ALLOWED_DETECTORS,
    run_feature_pad_protocol,
    save_protocol_outputs,
)


@dataclass
class BenchmarkConfig:
    actions: Tuple[str, ...] = ACTIONS
    feature_detectors: Tuple[str, ...] = ALLOWED_DETECTORS
    deep_detectors: Tuple[str, ...] = DEEP_DETECTORS
    feature_model_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    deep_model_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    deep_train: DeepTrainConfig = field(default_factory=DeepTrainConfig)
    feature_bootstrap_replicates: int = 500
    seed: int = 20260713

    def validate(self) -> None:
        if not self.actions or set(self.actions) - set(ACTIONS):
            raise ValueError("actions must be a non-empty subset of the five formal actions")
        if not self.feature_detectors or set(self.feature_detectors) - set(ALLOWED_DETECTORS):
            raise ValueError("invalid feature_detectors")
        if not self.deep_detectors or set(self.deep_detectors) - set(DEEP_DETECTORS):
            raise ValueError("invalid deep_detectors")
        if self.feature_bootstrap_replicates < 0:
            raise ValueError("feature bootstrap count cannot be negative")
        self.deep_train.validate()


def load_benchmark_dataset(
    dataset_dir: Path, actions: Sequence[str] = ACTIONS
) -> Tuple[List[RawTrajectoryRecord], Dict[str, np.ndarray]]:
    """Load one no-pickle ``<action>.npz`` raw bundle per action."""

    records: List[RawTrajectoryRecord] = []
    features: Dict[str, np.ndarray] = {}
    sample_identity = set()
    for action in actions:
        path = Path(dataset_dir) / (action + ".npz")
        action_records, action_features = load_raw_sequence_bundle(path)
        if any(record.action != action for record in action_records):
            raise ValueError("bundle action mismatch: %s" % path)
        for record in action_records:
            identity = (record.label, record.sample_id)
            if identity in sample_identity:
                raise ValueError("duplicate sample across bundles: %r" % (identity,))
            sample_identity.add(identity)
        records.extend(action_records)
        features[action] = action_features
    return records, features


def _feature_arrays(
    records: Sequence[RawTrajectoryRecord], feature_vectors: np.ndarray, action: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_records = [record for record in records if record.action == action]
    features = np.asarray(feature_vectors, dtype=np.float64)
    if len(features) != len(action_records):
        raise ValueError("feature row count does not match %s records" % action)
    return (
        features,
        np.asarray([record.label for record in action_records], dtype=np.int64),
        np.asarray([record.user_id for record in action_records], dtype=np.int64),
        np.asarray([record.pool for record in action_records], dtype="U5"),
        np.asarray([record.action for record in action_records], dtype="U16"),
    )


def _row_from_metrics(
    *,
    action: str,
    family: str,
    detector: str,
    operating_point: str,
    threshold: float,
    validation: Mapping[str, float],
    test: Mapping[str, float],
    best_epoch: Optional[int] = None,
    bootstrap: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "action": action,
        "detector_family": family,
        "detector": detector,
        "operating_point": operating_point,
        "threshold_from_validation": float(threshold),
        "validation_fa": float(validation["fa"]),
        "validation_frr": float(validation["frr"]),
        "validation_auc": float(validation["auc"]),
        "test_fa": float(test["fa"]),
        "test_frr": float(test["frr"]),
        "test_auc": float(test["auc"]),
        "n_test_real": int(test["n_real"]),
        "n_test_fake": int(test["n_fake"]),
        "best_epoch": "" if best_epoch is None else int(best_epoch),
    }
    if bootstrap is None:
        for metric in ("fa", "frr", "auc"):
            row["test_%s_ci95_low" % metric] = float("nan")
            row["test_%s_ci95_high" % metric] = float("nan")
    else:
        summary = bootstrap["summary"]
        point = summary[operating_point]
        row.update({
            "test_fa_ci95_low": float(point["fa"]["ci95_low"]),
            "test_fa_ci95_high": float(point["fa"]["ci95_high"]),
            "test_frr_ci95_low": float(point["frr"]["ci95_low"]),
            "test_frr_ci95_high": float(point["frr"]["ci95_high"]),
            "test_auc_ci95_low": float(summary["auc"]["ci95_low"]),
            "test_auc_ci95_high": float(summary["auc"]["ci95_high"]),
        })
    return row


def _plot_curve(
    curve: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, float]],
    action: str,
    detector: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.8, 4.4), dpi=140)
    axis.plot(curve["frr"], curve["fa"], color="#1f77b4", linewidth=1.7, label="test sweep")
    markers = (("eer", "validation EER threshold", "#d62728"), ("val_frr_le_5pct", "validation FRR<=5% threshold", "#2ca02c"))
    for name, label, color in markers:
        metric = metrics[name]
        axis.scatter([metric["frr"]], [metric["fa"]], s=42, color=color, label=label, zorder=3)
    axis.set_xlabel("Test FRR (real rejected)")
    axis.set_ylabel("Test FA (fake accepted)")
    axis.set_xlim(-0.01, 1.01)
    axis.set_ylim(-0.01, 1.01)
    axis.grid(alpha=0.25)
    axis.set_title("%s — %s" % (action, detector))
    axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    temporary = output_path.with_name(output_path.name + ".tmp.png")
    fig.savefig(temporary)
    plt.close(fig)
    os.replace(str(temporary), str(output_path))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _macro_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["detector_family"]), str(row["detector"]), str(row["operating_point"]))
        groups.setdefault(key, []).append(row)
    output: List[Dict[str, Any]] = []
    for (family, detector, operating), values in sorted(groups.items()):
        output.append({
            "detector_family": family,
            "detector": detector,
            "operating_point": operating,
            "n_actions": len(values),
            "macro_validation_fa": float(np.mean([row["validation_fa"] for row in values])),
            "macro_validation_frr": float(np.mean([row["validation_frr"] for row in values])),
            "macro_validation_auc": float(np.mean([row["validation_auc"] for row in values])),
            "macro_test_fa": float(np.mean([row["test_fa"] for row in values])),
            "macro_test_frr": float(np.mean([row["test_frr"] for row in values])),
            "macro_test_auc": float(np.mean([row["test_auc"] for row in values])),
        })
    return output


def _write_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    macro_rows: Sequence[Mapping[str, Any]],
    config: BenchmarkConfig,
    deep_batch_size_by_pair: Optional[Mapping[str, int]] = None,
) -> None:
    lines = [
        "# 五动作轨迹 PAD 完整评估报告",
        "",
        "本报告同时运行 clean-room feature PAD 与独立 raw-sequence Deep PAD。",
        "标签固定为 real=0/fake=1，分数越大越像 fake；`score < threshold` 才接受为 real。",
        "所有 normalization/model 只在 train 拟合；checkpoint 与 EER/FRR<=5% 阈值只由 validation 选择；test 不参与调参。",
        "test FA-FRR 曲线只作完整可视化，不从 test 曲线反选阈值。Deep PAD 不复用 generator critic，也不使用 selector。",
        "",
        "## Per action / detector",
        "",
        "| action | family | detector | point | val FA | val FRR | test FA | test FRR | test AUC |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {action} | {detector_family} | {detector} | {operating_point} | {validation_fa:.4f} | {validation_frr:.4f} | {test_fa:.4f} | {test_frr:.4f} | {test_auc:.4f} |".format(**row)
        )
    lines.extend([
        "", "## Macro over actions", "",
        "| family | detector | point | actions | test FA | test FRR | test AUC |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in macro_rows:
        lines.append(
            "| {detector_family} | {detector} | {operating_point} | {n_actions} | {macro_test_fa:.4f} | {macro_test_frr:.4f} | {macro_test_auc:.4f} |".format(**row)
        )
    lines.extend([
        "", "## 分动作固定阈值结果与曲线", "",
        "以下每张图横轴为 test FRR、纵轴为 test FA；红点/绿点分别是 validation EER 与 validation FRR<=5% 固定阈值应用到 test 的位置。",
        "95% CI 使用 test user 为重采样单位，real/fake users 分别有放回抽样，并保留每个被抽中 user 的全部 windows。",
    ])

    def metric_ci(row: Mapping[str, Any], metric: str) -> str:
        value = float(row["test_" + metric])
        low = float(row["test_%s_ci95_low" % metric])
        high = float(row["test_%s_ci95_high" % metric])
        if not (np.isfinite(low) and np.isfinite(high)):
            return "%.4f (CI unavailable)" % value
        return "%.4f [%.4f, %.4f]" % (value, low, high)

    detector_order = list(config.feature_detectors) + list(config.deep_detectors)
    for action in config.actions:
        lines.extend(["", "### " + action, ""])
        action_rows = [row for row in rows if row["action"] == action]
        for point, title in (
            ("eer", "Validation EER threshold → test"),
            ("val_frr_le_5pct", "Validation FRR<=5% threshold → test"),
        ):
            lines.extend([
                "#### " + title, "",
                "| detector | test FA (95% CI) | test FRR (95% CI) | test AUC (95% CI) |",
                "| --- | ---: | ---: | ---: |",
            ])
            indexed = {
                str(row["detector"]): row
                for row in action_rows if row["operating_point"] == point
            }
            for detector in detector_order:
                if detector not in indexed:
                    continue
                row = indexed[detector]
                lines.append("| %s | %s | %s | %s |" % (
                    detector, metric_ci(row, "fa"), metric_ci(row, "frr"),
                    metric_ci(row, "auc"),
                ))
            lines.append("")
        lines.extend(["#### Test FA–FRR curves", ""])
        present = [name for name in detector_order if any(row["detector"] == name for row in action_rows)]
        for start in range(0, len(present), 2):
            group = present[start:start + 2]
            lines.append("| " + " | ".join(group) + " |")
            lines.append("| " + " | ".join(["---"] * len(group)) + " |")
            lines.append("| " + " | ".join(
                "![%s %s FA-FRR](plots/%s/%s.png)" % (action, name, action, name)
                for name in group
            ) + " |")
            lines.append("")
    if deep_batch_size_by_pair is not None:
        exact_batches = {
            str(identity): int(batch_size)
            for identity, batch_size in sorted(deep_batch_size_by_pair.items())
        }
        lines.extend([
            "", "## Deep PAD no-truncation batch sizes", "",
            "Each value below is the exact batch size selected by that pair's "
            "audited longest-event probe; it is not a maximum collapsed across pairs.",
            "The single `deep_train.batch_size` in the generic configuration block is "
            "only the maximum used for compact dataclass display; the map here is authoritative.",
            "", "```json", json.dumps(exact_batches, indent=2, sort_keys=True),
            "```", "",
        ])
    lines.extend([
        "", "## Configuration", "", "```json",
        json.dumps(asdict(config), indent=2, sort_keys=True), "```", "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _write_operating_markdown(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# " + title,
        "",
        "| action | family | detector | point | val FA | val FRR | test FA | test FRR | test AUC |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {action} | {detector_family} | {detector} | {operating_point} | {validation_fa:.4f} | {validation_frr:.4f} | {test_fa:.4f} | {test_frr:.4f} | {test_auc:.4f} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def _write_macro_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Macro average over actions", "",
        "| family | detector | point | actions | test FA | test FRR | test AUC |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {detector_family} | {detector} | {operating_point} | {n_actions} | {macro_test_fa:.4f} | {macro_test_frr:.4f} | {macro_test_auc:.4f} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def run_complete_benchmark(
    records: Sequence[RawTrajectoryRecord],
    feature_vectors_by_action: Mapping[str, np.ndarray],
    *,
    output_dir: Path,
    config: Optional[BenchmarkConfig] = None,
    device: Optional[str] = None,
) -> Dict[str, str]:
    """Run every configured action through all feature and deep detectors."""

    cfg = config or BenchmarkConfig()
    cfg.validate()
    root = Path(output_dir)
    completion = root / "benchmark_manifest.json"
    if completion.exists():
        raise FileExistsError("refusing to overwrite completed benchmark: %s" % root)
    root.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []

    for action in cfg.actions:
        if action not in feature_vectors_by_action:
            raise ValueError("missing feature vectors for %s" % action)
        x, y, users, pools, actions = _feature_arrays(records, feature_vectors_by_action[action], action)
        for detector in cfg.feature_detectors:
            result = run_feature_pad_protocol(
                x, y, users, pools, actions,
                action=action,
                detector_kind=detector,
                random_state=cfg.seed,
                model_params=cfg.feature_model_params.get(detector),
                bootstrap_replicates=cfg.feature_bootstrap_replicates,
                bootstrap_seed=cfg.seed + 31,
            )
            result_dir = root / "feature_pad" / action / detector
            save_protocol_outputs(result, result_dir)
            for point in ("eer", "val_frr_le_5pct"):
                all_rows.append(_row_from_metrics(
                    action=action, family="feature_pad", detector=detector,
                    operating_point=point, threshold=result.thresholds[point],
                    validation=result.validation_metrics[point], test=result.test_metrics[point],
                    bootstrap=result.bootstrap,
                ))
            _plot_curve(result.curves["test"], result.test_metrics, action, detector, root / "plots" / action / (detector + ".png"))

        for detector in cfg.deep_detectors:
            result = run_deep_pad_protocol(
                records,
                action=action,
                detector_kind=detector,
                output_dir=root / "deep_pad" / action / detector,
                config=cfg.deep_train,
                model_params=cfg.deep_model_params.get(detector),
                device=device,
            )
            for point in ("eer", "val_frr_le_5pct"):
                all_rows.append(_row_from_metrics(
                    action=action, family="deep_pad", detector=detector,
                    operating_point=point, threshold=result.thresholds[point],
                    validation=result.validation_metrics[point], test=result.test_metrics[point],
                    best_epoch=result.best_epoch,
                    bootstrap=result.bootstrap,
                ))
            _plot_curve(result.curves["test"], result.test_metrics, action, detector, root / "plots" / action / (detector + ".png"))

    macro = _macro_rows(all_rows)
    per_action_fields = (
        "action", "detector_family", "detector", "operating_point", "threshold_from_validation",
        "validation_fa", "validation_frr", "validation_auc", "test_fa", "test_frr", "test_auc",
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
    per_action_path = root / "per_action_detector.csv"
    macro_path = root / "macro_by_detector.csv"
    macro_md_path = root / "macro_by_detector.md"
    report_path = root / "benchmark_report.md"
    _write_csv(per_action_path, all_rows, per_action_fields)
    _write_csv(macro_path, macro, macro_fields)
    _write_macro_markdown(macro_md_path, macro)
    by_action_root = root / "summaries" / "by_action"
    for action in cfg.actions:
        action_rows = [row for row in all_rows if row["action"] == action]
        _write_csv(by_action_root / (action + ".csv"), action_rows, per_action_fields)
        _write_operating_markdown(by_action_root / (action + ".md"), action + " detector results", action_rows)
    by_detector_root = root / "summaries" / "by_detector"
    for family in ("feature_pad", "deep_pad"):
        detectors = cfg.feature_detectors if family == "feature_pad" else cfg.deep_detectors
        for detector in detectors:
            detector_rows = [
                row for row in all_rows
                if row["detector_family"] == family and row["detector"] == detector
            ]
            name = family + "__" + detector
            _write_csv(by_detector_root / (name + ".csv"), detector_rows, per_action_fields)
            _write_operating_markdown(by_detector_root / (name + ".md"), name + " results", detector_rows)
    _write_report(report_path, all_rows, macro, cfg)
    manifest = {
        "schema_version": "trajectory_complete_benchmark_v1",
        "status": "complete",
        "score_direction": "fake_high",
        "acceptance_rule": "score < threshold",
        "feature_detectors": list(cfg.feature_detectors),
        "deep_detectors": list(cfg.deep_detectors),
        "actions": list(cfg.actions),
        "n_per_action_operating_rows": len(all_rows),
        "n_macro_rows": len(macro),
        "outputs": {
            "per_action": str(per_action_path), "macro": str(macro_path),
            "macro_markdown": str(macro_md_path), "report": str(report_path),
            "by_action": str(by_action_root), "by_detector": str(by_detector_root),
            "plots": str(root / "plots"),
        },
    }
    temporary = completion.with_name(completion.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(completion))
    return {"manifest": str(completion), **manifest["outputs"]}


__all__ = [
    "BenchmarkConfig", "load_benchmark_dataset", "run_complete_benchmark",
]
