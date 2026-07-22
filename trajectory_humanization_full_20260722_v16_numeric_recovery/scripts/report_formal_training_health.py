#!/usr/bin/env python3
"""Read-only health report for the five formal diffusion training jobs.

The monitor deliberately does not import torch and never opens a checkpoint.
It reads small JSON/JSONL files, file metadata, supervisor heartbeats, logs and
``nvidia-smi``.  Its only writes are atomic replacements of its own latest
JSON and Markdown report files.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_determinism import runtime_determinism_matches

DEFAULT_CONFIG = ROOT / "orchestration" / "formal_pipeline_config.json"
NONFINITE_LOG_PATTERN = re.compile(
    r"(?i)(?:\b(?:nan|inf|infinity)\b|non[- ]finite|floatingpointerror)"
)


def _issue(
    issues: List[Dict[str, Any]], severity: str, code: str,
    zh: str, en: str, action: Optional[str] = None,
) -> None:
    value = {
        "severity": severity,
        "code": code,
        "message_zh": zh,
        "message_en": en,
    }
    if action is not None:
        value["action"] = action
    issues.append(value)


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _nonnegative_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and int(value) >= 0
    )


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256_file(path: Path, attempts: int = 2) -> Tuple[str, os.stat_result]:
    """Hash one stable inode, retrying an atomic replacement race."""

    if attempts <= 0:
        raise ValueError("stable hash attempts must be positive")
    for _attempt in range(attempts):
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
        before_identity = (
            before.st_dev, before.st_ino, before.st_size,
            getattr(before, "st_mtime_ns", int(before.st_mtime * 1e9)),
        )
        after_identity = (
            after.st_dev, after.st_ino, after.st_size,
            getattr(after, "st_mtime_ns", int(after.st_mtime * 1e9)),
        )
        if before_identity == after_identity:
            return digest, after
    raise OSError("file changed during both stable hash attempts: %s" % path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, str(path))
        try:
            directory = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _file_info(path: Path, now: float) -> Dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_unix_time": float(stat.st_mtime),
        "age_seconds": max(0.0, float(now) - float(stat.st_mtime)),
    }


def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None, "top-level JSON is not an object"
        return value, None
    except (OSError, ValueError, TypeError) as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)


def _read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return rows, [{"line": 0, "error": "%s: %s" % (type(exc).__name__, exc)}]
    lines = raw.splitlines()
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("row is not an object")
            rows.append(value)
        except (ValueError, TypeError) as exc:
            failures.append({
                "line": index,
                "is_last_line": index == len(lines),
                "file_ended_with_newline": raw.endswith("\n"),
                "error": "%s: %s" % (type(exc).__name__, exc),
            })
    return rows, failures


def _validation_epochs(total_epochs: int) -> List[int]:
    return sorted(set(int(math.ceil(total_epochs * fraction / 5.0)) for fraction in range(1, 6)))


def _default_process_probe(pid: int, command: Sequence[str]) -> Dict[str, Any]:
    result = {"pid": int(pid), "alive": False, "command_matches": False, "cmdline": None}
    if pid <= 0:
        return result
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        cmdline = raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return result
    result["alive"] = True
    result["cmdline"] = cmdline
    anchors: List[str] = []
    if command:
        for token in command:
            if token.endswith("train_trajectory_diffusion.py"):
                anchors.append(Path(token).name)
        for flag in ("--action", "--output-dir"):
            if flag in command and command.index(flag) + 1 < len(command):
                anchors.append(str(command[command.index(flag) + 1]))
    result["command_matches"] = bool(anchors) and all(anchor in cmdline for anchor in anchors)
    return result


def query_gpus(attempts: int = 3, retry_delay_seconds: float = 0.25) -> Dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    if int(attempts) <= 0:
        raise ValueError("GPU query attempts must be positive")
    completed = None
    last_error = None
    for attempt in range(int(attempts)):
        try:
            completed = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, timeout=10, check=False,
            )
            if completed.returncode == 0:
                break
            last_error = completed.stderr.strip() or "nvidia-smi return code %d" % completed.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_error = "%s: %s" % (type(exc).__name__, exc)
        if attempt + 1 < int(attempts):
            time.sleep(float(retry_delay_seconds))
    if completed is None or completed.returncode != 0:
        return {"available": False, "error": last_error or "nvidia-smi failed", "gpus": []}
    gpus = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        try:
            index = int(fields[0])
            utilization = float(fields[2])
            memory_used = float(fields[3])
            memory_total = float(fields[4])
            temperature = float(fields[5])
        except ValueError:
            continue
        gpus.append({
            "index": index,
            "name": fields[1],
            "utilization_percent": utilization,
            "memory_used_mib": memory_used,
            "memory_total_mib": memory_total,
            "memory_fraction": memory_used / memory_total if memory_total > 0 else None,
            "temperature_c": temperature,
        })
    return {"available": bool(gpus), "error": None if gpus else "no parseable GPU rows", "gpus": gpus}


def _scan_log(path: Path, maximum_bytes: int = 8 * 1024 * 1024) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "nonfinite_matches": []}
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            if size > maximum_bytes:
                stream.seek(size - maximum_bytes)
            raw = stream.read()
        text = raw.decode("utf-8", "replace")
    except OSError as exc:
        return {
            "path": str(path), "exists": True,
            "read_error": "%s: %s" % (type(exc).__name__, exc),
            "nonfinite_matches": [],
        }
    matches = []
    for line in text.splitlines():
        if NONFINITE_LOG_PATTERN.search(line):
            matches.append(line.strip()[:500])
    return {
        "path": str(path),
        "exists": True,
        "scanned_tail_bytes": len(raw),
        "tail_only": size > maximum_bytes,
        "nonfinite_matches": matches[-20:],
    }


def _loss_trend(values: Sequence[float], window: int = 5) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0, "first": None, "latest": None,
            "latest_change_fraction": None, "rolling_status": "insufficient_data",
        }
    result: Dict[str, Any] = {
        "count": len(values),
        "first": float(values[0]),
        "latest": float(values[-1]),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "latest_change_fraction": None,
        "rolling_window": min(window, len(values)),
        "rolling_previous_median": None,
        "rolling_recent_median": None,
        "rolling_change_fraction": None,
        "rolling_status": "insufficient_data",
    }
    if len(values) >= 2 and values[-2] != 0:
        result["latest_change_fraction"] = float(values[-1] / values[-2] - 1.0)
    if len(values) >= 2 * window:
        previous = float(statistics.median(values[-2 * window:-window]))
        recent = float(statistics.median(values[-window:]))
        change = recent / previous - 1.0 if previous != 0 else None
        result.update({
            "rolling_previous_median": previous,
            "rolling_recent_median": recent,
            "rolling_change_fraction": change,
            "rolling_status": (
                "regressing" if change is not None and change > 0.10
                else "improving" if change is not None and change < -0.02
                else "flat_or_noisy"
            ),
        })
    return result


def _add_loss_issues(
    action: str, name: str, values: Sequence[float], trend: Mapping[str, Any],
    issues: List[Dict[str, Any]],
) -> None:
    if len(values) >= 2 and values[-1] > values[-2]:
        _issue(
            issues, "warning", "%s_LATEST_INCREASE" % name.upper(),
            "%s 最新一次 loss 上升；单次上升只告警，不判训练失败。" % action,
            "%s latest %s loss increased; one increase is warning-only, not failure." % (action, name),
            action,
        )
    if trend.get("rolling_status") == "regressing":
        _issue(
            issues, "warning", "%s_ROLLING_REGRESSION" % name.upper(),
            "%s 的 %s loss 最近 5 个中位数比此前 5 个高超过 10%%。" % (action, name),
            "%s %s loss recent-5 median is over 10%% above the previous-5 median." % (action, name),
            action,
        )


def _check_metrics(
    action: str, rows: Sequence[Mapping[str, Any]], total_epochs: int,
    counts: Mapping[str, Any], transaction_pending: bool, metrics_age: Optional[float],
    run_complete: bool, issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    train_rows = [row for row in rows if row.get("type") == "train_epoch"]
    val_rows = [row for row in rows if row.get("type") == "validation"]
    unknown = [row.get("type") for row in rows if row.get("type") not in ("train_epoch", "validation")]
    if unknown:
        _issue(
            issues, "error", "UNKNOWN_METRIC_ROW",
            "%s metrics.jsonl 含未知记录类型。" % action,
            "%s metrics.jsonl contains unknown row types." % action,
            action,
        )

    train_epochs: List[int] = []
    train_losses: List[float] = []
    train_global_steps: List[int] = []
    expected_train_examples = counts.get("train")
    for row in train_rows:
        try:
            epoch = int(row.get("completed_epoch"))
        except (TypeError, ValueError):
            epoch = -1
        train_epochs.append(epoch)
        try:
            step = int(row.get("global_step"))
        except (TypeError, ValueError):
            step = -1
        train_global_steps.append(step)
        loss = _safe_float(row.get("loss"))
        if loss is None or loss <= 0:
            _issue(
                issues, "error", "NONFINITE_OR_NONPOSITIVE_TRAIN_LOSS",
                "%s epoch %s 的训练 loss 不是有限正数。" % (action, epoch),
                "%s epoch %s train loss is not finite and positive." % (action, epoch),
                action,
            )
        else:
            train_losses.append(loss)
        if row.get("full_train_split_consumed") is not True:
            _issue(
                issues, "error", "INCOMPLETE_TRAIN_SPLIT",
                "%s epoch %s 未标记完整消费 train split。" % (action, epoch),
                "%s epoch %s did not mark the full train split consumed." % (action, epoch),
                action,
            )
        if expected_train_examples is not None and row.get("examples_total_in_epoch") != expected_train_examples:
            _issue(
                issues, "error", "TRAIN_EXAMPLE_COUNT_MISMATCH",
                "%s epoch %s 的训练样本数与 run manifest 不一致。" % (action, epoch),
                "%s epoch %s train example count disagrees with run manifest." % (action, epoch),
                action,
            )
        if not _finite_positive(row.get("valid_feature_count_total")):
            _issue(
                issues, "error", "INVALID_TRAIN_FEATURE_COUNT",
                "%s epoch %s 的有效 feature 数不是有限正数。" % (action, epoch),
                "%s epoch %s valid-feature count is not finite and positive." % (action, epoch),
                action,
            )
        if not _finite_positive(row.get("batches_total_in_epoch")):
            _issue(
                issues, "error", "INVALID_TRAIN_BATCH_COUNT",
                "%s epoch %s 的 batch 数无效。" % (action, epoch),
                "%s epoch %s batch count is invalid." % (action, epoch),
                action,
            )

    expected_sequence = list(range(1, max(train_epochs) + 1)) if train_epochs and min(train_epochs) >= 1 else []
    if train_epochs != expected_sequence or len(train_epochs) != len(set(train_epochs)):
        _issue(
            issues, "error", "TRAIN_EPOCH_GAP_OR_DUPLICATE",
            "%s 的训练 epoch 不连续、乱序或重复：%s。" % (action, train_epochs),
            "%s train epochs have a gap, reordering, or duplicate: %s." % (action, train_epochs),
            action,
        )
    if any(epoch > total_epochs for epoch in train_epochs):
        _issue(
            issues, "error", "TRAIN_EPOCH_EXCEEDS_CONFIG",
            "%s 的 metrics epoch 超过配置的 %d。" % (action, total_epochs),
            "%s metrics epoch exceeds configured %d." % (action, total_epochs),
            action,
        )
    if (
        any(step <= 0 for step in train_global_steps)
        or any(later <= earlier for earlier, later in zip(train_global_steps, train_global_steps[1:]))
    ):
        _issue(
            issues, "error", "TRAIN_GLOBAL_STEP_NOT_STRICTLY_INCREASING",
            "%s train epoch 的 global_step 不是严格递增正整数。" % action,
            "%s train-epoch global_step is not a strictly increasing positive integer." % action,
            action,
        )

    val_epochs: List[int] = []
    val_losses: List[float] = []
    val_by_epoch: Dict[int, float] = {}
    train_step_by_epoch = dict(zip(train_epochs, train_global_steps))
    expected_val_examples = counts.get("val")
    milestones = _validation_epochs(total_epochs)
    for row in val_rows:
        try:
            epoch = int(row.get("completed_epoch"))
        except (TypeError, ValueError):
            epoch = -1
        val_epochs.append(epoch)
        try:
            val_step = int(row.get("global_step"))
        except (TypeError, ValueError):
            val_step = -1
        if train_step_by_epoch.get(epoch) != val_step:
            _issue(
                issues, "error", "VALIDATION_GLOBAL_STEP_MISMATCH",
                "%s validation epoch %s 的 global_step 与 train epoch 不一致。" % (action, epoch),
                "%s validation epoch %s global_step disagrees with its train epoch." % (action, epoch),
                action,
            )
        loss = _safe_float(row.get("val_loss"))
        if loss is None or loss <= 0:
            _issue(
                issues, "error", "NONFINITE_OR_NONPOSITIVE_VAL_LOSS",
                "%s epoch %s 的 validation loss 不是有限正数。" % (action, epoch),
                "%s epoch %s validation loss is not finite and positive." % (action, epoch),
                action,
            )
        else:
            val_losses.append(loss)
            val_by_epoch[epoch] = loss
        if epoch not in milestones:
            _issue(
                issues, "error", "UNEXPECTED_VALIDATION_EPOCH",
                "%s 在非约定 epoch %s 做了 validation。" % (action, epoch),
                "%s validation occurred at unexpected epoch %s." % (action, epoch),
                action,
            )
        if row.get("full_validation_split") is not True or row.get("ema_weights") is not True:
            _issue(
                issues, "error", "INVALID_FULL_EMA_VALIDATION",
                "%s epoch %s 不是完整 validation split + EMA 权重。" % (action, epoch),
                "%s epoch %s is not full-split EMA validation." % (action, epoch),
                action,
            )
        if expected_val_examples is not None and row.get("n_examples") != expected_val_examples:
            _issue(
                issues, "error", "VAL_EXAMPLE_COUNT_MISMATCH",
                "%s epoch %s 的 validation 样本数与 run manifest 不一致。" % (action, epoch),
                "%s epoch %s validation count disagrees with run manifest." % (action, epoch),
                action,
            )
        if not _finite_positive(row.get("valid_feature_count")) or not _finite_positive(row.get("n_batches")):
            _issue(
                issues, "error", "INVALID_VAL_COUNTS",
                "%s epoch %s 的 validation batch/feature 计数无效。" % (action, epoch),
                "%s epoch %s validation batch/feature counts are invalid." % (action, epoch),
                action,
            )

    if len(val_epochs) != len(set(val_epochs)) or val_epochs != sorted(val_epochs):
        _issue(
            issues, "error", "VALIDATION_DUPLICATE_OR_REORDERED",
            "%s 的 validation epoch 重复或乱序。" % action,
            "%s validation epochs are duplicated or reordered." % action,
            action,
        )
    max_train_epoch = max(train_epochs) if train_epochs else 0
    required = [epoch for epoch in milestones if epoch <= max_train_epoch]
    missing = [epoch for epoch in required if epoch not in val_epochs]
    if missing:
        transient = transaction_pending
        _issue(
            issues, "warning" if transient and not run_complete else "error",
            "MISSING_VALIDATION_MILESTONE",
            "%s 缺少已到达里程碑的完整 validation：%s%s。" % (
                action, missing, "（epoch 事务可能正在提交）" if transient else "",
            ),
            "%s is missing reached validation milestones %s%s." % (
                action, missing, " (epoch transaction may be committing)" if transient else "",
            ),
            action,
        )
    if run_complete and (train_epochs != list(range(1, total_epochs + 1)) or val_epochs != milestones):
        _issue(
            issues, "error", "COMPLETE_RUN_METRICS_INCOMPLETE",
            "%s 标记 complete，但没有完整 %d epochs 和全部 validation 里程碑。" % (action, total_epochs),
            "%s is marked complete without all %d epochs and validation milestones." % (action, total_epochs),
            action,
        )

    train_trend = _loss_trend(train_losses)
    val_trend = _loss_trend(val_losses)
    _add_loss_issues(action, "train", train_losses, train_trend, issues)
    _add_loss_issues(action, "validation", val_losses, val_trend, issues)
    return {
        "train_epochs": train_epochs,
        "train_global_steps": train_global_steps,
        "latest_train_global_step": train_global_steps[-1] if train_global_steps else 0,
        "validation_epochs": val_epochs,
        "expected_validation_epochs": milestones,
        "train_loss": train_trend,
        "validation_loss": val_trend,
        "validation_loss_by_epoch": {str(key): value for key, value in sorted(val_by_epoch.items())},
    }


def _check_best(
    action: str, action_root: Path, best: Optional[Mapping[str, Any]],
    metrics: Mapping[str, Any], issues: List[Dict[str, Any]],
    transaction_pending: bool = False,
) -> Dict[str, Any]:
    val_by_epoch = {
        int(key): float(value)
        for key, value in metrics.get("validation_loss_by_epoch", {}).items()
    }
    if best is None:
        if val_by_epoch:
            _issue(
                issues, "error", "BEST_MANIFEST_MISSING_AFTER_VALIDATION",
                "%s 已有 validation，但 best_manifest.json 缺失或不可读。" % action,
                "%s has validation metrics but no readable best_manifest.json." % action,
                action,
            )
        return {"exists": False, "selected_epoch": None, "val_loss": None, "checkpoint_exists": False}
    if best.get("selection_split") != "val" or best.get("test_used_for_selection") is not False:
        _issue(
            issues, "error", "BEST_NOT_VALIDATION_ONLY",
            "%s 的 best checkpoint 不是严格 validation-only 选择。" % action,
            "%s best checkpoint is not strictly validation-only selected." % action,
            action,
        )
    if (
        best.get("selection_metric") != "full_val_masked_epsilon_mse_ema"
        or best.get("lower_is_better") is not True
    ):
        _issue(
            issues, "error", "BEST_SELECTION_METRIC_INVALID",
            "%s 的 best selection metric 不是 full-val EMA loss。" % action,
            "%s best selection metric is not full-validation EMA loss." % action,
            action,
        )
    selected = best.get("best")
    history = best.get("history")
    if not isinstance(selected, dict) or not isinstance(history, list) or not history:
        _issue(
            issues, "error", "BEST_MANIFEST_SCHEMA_INVALID",
            "%s 的 best manifest 结构无效。" % action,
            "%s best manifest schema is invalid." % action,
            action,
        )
        return {"exists": True, "selected_epoch": None, "val_loss": None, "checkpoint_exists": False}
    try:
        selected_epoch = int(selected.get("completed_epoch"))
    except (TypeError, ValueError):
        selected_epoch = -1
    selected_loss = _safe_float(selected.get("val_loss"))
    selected_waits_for_commit = transaction_pending and selected_epoch not in val_by_epoch
    if selected_waits_for_commit:
        _issue(
            issues, "warning", "BEST_AWAITS_VALIDATION_METRIC_COMMIT",
            "%s 新 best 已发布，但对应 validation metric 仍在 epoch 原子事务中提交。" % action,
            "%s new best is published while its validation metric is still in the atomic epoch commit." % action,
            action,
        )
    elif selected_epoch not in val_by_epoch or selected_loss is None or not math.isclose(
        selected_loss, val_by_epoch.get(selected_epoch, float("inf")), rel_tol=1e-9, abs_tol=1e-12,
    ):
        _issue(
            issues, "error", "BEST_NOT_BACKED_BY_VALIDATION_ROW",
            "%s 的 best epoch/loss 没有对应的 validation 记录。" % action,
            "%s best epoch/loss is not backed by a validation row." % action,
            action,
        )
    if not selected_waits_for_commit and val_by_epoch and selected_loss is not None and not math.isclose(
        selected_loss, min(val_by_epoch.values()), rel_tol=1e-9, abs_tol=1e-12,
    ):
        _issue(
            issues, "error", "BEST_IS_NOT_MINIMUM_VALIDATION_LOSS",
            "%s 的 best loss 不是目前最小 validation loss。" % action,
            "%s best loss is not the minimum observed validation loss." % action,
            action,
        )
    history_epochs: List[int] = []
    history_losses: List[float] = []
    history_paths: List[str] = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        try:
            epoch = int(entry.get("completed_epoch"))
        except (TypeError, ValueError):
            epoch = -1
        loss = _safe_float(entry.get("val_loss"))
        history_epochs.append(epoch)
        if loss is not None:
            history_losses.append(loss)
        raw_path = str(entry.get("path", ""))
        checkpoint = Path(raw_path)
        if not checkpoint.is_absolute():
            checkpoint = action_root / checkpoint
        history_paths.append(str(checkpoint))
        if not checkpoint.is_file():
            _issue(
                issues, "error", "BEST_HISTORY_CHECKPOINT_MISSING",
                "%s 的不可覆盖 best history checkpoint 缺失：%s。" % (action, checkpoint),
                "%s immutable best-history checkpoint is missing: %s." % (action, checkpoint),
                action,
            )
        entry_waits_for_commit = transaction_pending and epoch == selected_epoch and epoch not in val_by_epoch
        if not entry_waits_for_commit and (epoch not in val_by_epoch or loss is None or not math.isclose(
            loss, val_by_epoch.get(epoch, float("inf")), rel_tol=1e-9, abs_tol=1e-12,
        )):
            _issue(
                issues, "error", "BEST_HISTORY_NOT_FROM_VALIDATION",
                "%s best history 中有条目不来自 validation。" % action,
                "%s best history contains an entry not backed by validation." % action,
                action,
            )
    if history_epochs != sorted(set(history_epochs)) or any(
        later >= earlier for earlier, later in zip(history_losses, history_losses[1:])
    ):
        _issue(
            issues, "error", "BEST_HISTORY_NOT_STRICT_IMPROVEMENTS",
            "%s best history 不是按 epoch 递增且 loss 严格改善。" % action,
            "%s best history is not epoch-ordered strict loss improvement." % action,
            action,
        )
    if history[-1] != selected:
        _issue(
            issues, "error", "BEST_HISTORY_TAIL_MISMATCH",
            "%s best 字段不等于 history 最后一条。" % action,
            "%s best entry differs from the final history entry." % action,
            action,
        )
    selected_path = Path(str(selected.get("path", "")))
    if not selected_path.is_absolute():
        selected_path = action_root / selected_path
    return {
        "exists": True,
        "selection_split": best.get("selection_split"),
        "test_used_for_selection": best.get("test_used_for_selection"),
        "selected_epoch": selected_epoch,
        "val_loss": selected_loss,
        "checkpoint_path": str(selected_path),
        "checkpoint_exists": selected_path.is_file(),
        "history_length": len(history),
        "history_paths": history_paths,
    }


def _check_best_checkpoint_integrity(
    action: str,
    action_root: Path,
    best: Optional[Mapping[str, Any]],
    run_manifest: Optional[Mapping[str, Any]],
    issues: List[Dict[str, Any]],
    transient_checkpoint_publish: bool = False,
) -> Dict[str, Any]:
    # Best files are immutable hard links and best_manifest.json is replaced
    # atomically only after the new file exists.  Unlike last.pt/last_state,
    # there is no legitimate two-file publication window.  Never let an
    # unrelated live checkpoint phase downgrade best corruption to a warning.
    del transient_checkpoint_publish
    result = {"all_entries_verified": False, "verified_entries": 0}
    if best is None:
        return result
    source = run_manifest.get("source") if isinstance(run_manifest, Mapping) else None
    if (
        best.get("checkpoint_role") != "validation_selected_best"
        or best.get("inference_weights") != "ema.shadow"
        or not isinstance(source, Mapping)
        or best.get("source") != source
    ):
        _issue(
            issues, "error", "BEST_MANIFEST_PROVENANCE_INVALID",
            "%s best manifest 的 role/inference/source 绑定无效。" % action,
            "%s best manifest role/inference/source binding is invalid." % action,
            action,
        )
    history = best.get("history")
    if not isinstance(history, list) or not history:
        return result
    verified = 0
    for entry in history:
        if not isinstance(entry, Mapping):
            continue
        path = Path(str(entry.get("path", "")))
        if not path.is_absolute():
            path = action_root / path
        identity_ok = (
            entry.get("checkpoint_role") == "validation_selected_best"
            and entry.get("inference_weights") == "ema.shadow"
            and isinstance(source, Mapping)
            and entry.get("source_sha256") == source.get("corpus_sha256")
            and entry.get("split_sha256") == source.get("split_sha256")
            and entry.get("reference_registry_sha256")
            == source.get("reference_registry_sha256")
        )
        expected_sha = entry.get("checkpoint_sha256")
        try:
            bytes_ok = (
                path.is_file() and path.stat().st_size > 0
                and isinstance(expected_sha, str) and len(expected_sha) == 64
                and _sha256_file(path) == expected_sha
            )
        except OSError:
            bytes_ok = False
        if not identity_ok or not bytes_ok:
            _issue(
                issues, "error", "BEST_CHECKPOINT_INTEGRITY_INVALID",
                "%s best history checkpoint 的 SHA/role/inference/source 无效：%s。"
                % (action, path),
                "%s best-history checkpoint SHA/role/inference/source is invalid: %s."
                % (action, path),
                action,
            )
        else:
            verified += 1
    result.update({
        "all_entries_verified": verified == len(history),
        "verified_entries": verified,
        "expected_entries": len(history),
    })
    return result


def _check_last_checkpoint_integrity(
    action: str,
    action_root: Path,
    run_manifest: Optional[Mapping[str, Any]],
    metric_report: Mapping[str, Any],
    issues: List[Dict[str, Any]],
    transient_checkpoint_publish: bool = False,
    observed_at: Optional[float] = None,
) -> Dict[str, Any]:
    checkpoint = action_root / "last.pt"
    state_path = action_root / "last_state.json"
    result = {
        "checkpoint": str(checkpoint), "state": str(state_path),
        "verified": False,
    }
    if not checkpoint.is_file():
        return result
    initial_stat = checkpoint.stat()
    # Recompute freshness from the current inode here rather than trusting the
    # earlier directory snapshot: last.pt can be atomically replaced between
    # those reads.  This is the only legitimate two-file warning window.
    current_time = time.time() if observed_at is None else float(observed_at)
    actual_transient = bool(
        transient_checkpoint_publish
        and max(0.0, current_time - float(initial_stat.st_mtime)) <= 120.0
    )
    if initial_stat.st_size <= 0:
        _issue(
            issues, "error", "LAST_CHECKPOINT_EMPTY",
            "%s last.pt 是空文件。" % action,
            "%s last.pt is empty." % action,
            action,
        )
        return result
    state, error = _read_json(state_path) if state_path.is_file() else (None, "missing")
    if state is None:
        _issue(
            issues, "warning" if actual_transient else "error", "LAST_STATE_MISSING_OR_UNREADABLE",
            "%s last_state.json 缺失或不可读。" % action,
            "%s last_state.json is missing or unreadable." % action,
            action,
        )
        result["error"] = error
        return result
    source = run_manifest.get("source") if isinstance(run_manifest, Mapping) else None
    config = run_manifest.get("config") if isinstance(run_manifest, Mapping) else None
    train_epochs = metric_report.get("train_epochs", [])
    max_epoch = max(train_epochs) if train_epochs else 0
    progress = state.get("progress") if isinstance(state.get("progress"), Mapping) else {}
    try:
        actual_sha, checkpoint_stat = _stable_sha256_file(checkpoint)
        actual_transient = bool(
            transient_checkpoint_publish
            and max(0.0, current_time - float(checkpoint_stat.st_mtime)) <= 120.0
        )
    except OSError as exc:
        _issue(
            issues, "warning" if actual_transient else "error",
            "LAST_CHECKPOINT_READ_FAILED",
            "%s last.pt 在完整性检查时不可读。" % action,
            "%s last.pt could not be read during integrity checking." % action,
            action,
        )
        result["error"] = str(exc)
        return result
    latest_step = int(metric_report.get("latest_train_global_step", 0))
    checkpoint_epoch = (
        int(progress["epoch_index"])
        if _nonnegative_int(progress.get("epoch_index")) else -1
    )
    checkpoint_batch = (
        int(progress["next_batch_in_epoch"])
        if _nonnegative_int(progress.get("next_batch_in_epoch")) else -1
    )
    checkpoint_step = (
        int(progress["global_step"])
        if _nonnegative_int(progress.get("global_step")) else -1
    )
    progress_sidecar, _progress_error = _read_json(
        action_root / "training_progress.json"
    ) if (action_root / "training_progress.json").is_file() else (None, "missing")
    run_complete = bool(
        isinstance(run_manifest, Mapping)
        and run_manifest.get("status") == "complete"
    )
    run_instance_ok = (
        not run_complete
        or (
            isinstance(progress_sidecar, Mapping)
            and isinstance(state.get("run_instance_id"), str)
            and bool(state.get("run_instance_id"))
            and progress_sidecar.get("run_instance_id")
            == state.get("run_instance_id")
        )
    )
    valid = (
        state.get("schema_version") == "trajectory_last_state_v1"
        and state.get("protocol_version") == "trajectory_diffusion_strict_five_ref_v2"
        and state.get("action") == action
        and Path(str(state.get("checkpoint_path", ""))).resolve() == checkpoint.resolve()
        and state.get("checkpoint_sha256") == actual_sha
        and _nonnegative_int(state.get("checkpoint_size_bytes"))
        and int(state["checkpoint_size_bytes"]) == int(checkpoint_stat.st_size)
        and isinstance(source, Mapping) and state.get("source") == source
        and isinstance(config, Mapping)
        and state.get("config_sha256") == _canonical_sha256(config)
        and checkpoint_epoch == max_epoch
        and checkpoint_batch >= 0
        and checkpoint_step >= latest_step
        and (checkpoint_batch != 0 or checkpoint_step == latest_step)
        and run_instance_ok
    )
    if not valid:
        _issue(
            issues, "warning" if actual_transient else "error", "LAST_CHECKPOINT_STATE_MISMATCH",
            "%s last.pt 与 last_state.json 的 SHA/progress/source/config 不一致。" % action,
            "%s last.pt disagrees with last_state.json SHA/progress/source/config." % action,
            action,
        )
    result.update({
        "verified": valid,
        "checkpoint_sha256": actual_sha,
        "progress": dict(progress),
    })
    return result


def _check_worker_progress(
    action: str,
    action_root: Path,
    job: Mapping[str, Any],
    job_status: str,
    process: Mapping[str, Any],
    run_manifest: Optional[Mapping[str, Any]],
    observed_at: float,
    stale_seconds: float,
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    path = action_root / "training_progress.json"
    info = _file_info(path, observed_at)
    result: Dict[str, Any] = {
        "path": str(path), "exists": bool(info.get("exists")),
        "status": "not_required", "heartbeat_age_seconds": None,
    }
    if not path.is_file():
        if job_status == "running":
            started = _safe_float(job.get("started_unix_time"))
            age = None if started is None else max(0.0, observed_at - started)
            result.update({"status": "awaiting_initial_publish", "heartbeat_age_seconds": age})
            if age is None or age > stale_seconds:
                _issue(
                    issues, "error", "TRAINING_PROGRESS_MISSING",
                    "%s 运行中但 training_progress.json 缺失或超过宽限期。" % action,
                    "%s is running without training_progress.json beyond its grace period." % action,
                    action,
                )
        elif run_manifest and run_manifest.get("status") == "complete":
            _issue(
                issues, "error", "TRAINING_PROGRESS_MISSING",
                "%s 完成运行缺少 training_progress.json。" % action,
                "%s completed run lacks training_progress.json." % action,
                action,
            )
        return result
    value, error = _read_json(path)
    if value is None:
        _issue(
            issues, "error", "TRAINING_PROGRESS_UNREADABLE",
            "%s training_progress.json 不可读。" % action,
            "%s training_progress.json is unreadable." % action,
            action,
        )
        result.update({"status": "invalid", "error": error})
        return result
    started = _safe_float(job.get("started_unix_time"))
    mtime = _safe_float(info.get("mtime_unix_time"))
    old_instance = (
        job_status == "running" and started is not None and mtime is not None
        and mtime <= started
        and int(value.get("pid", -1)) != int(job.get("pid", -2))
    )
    if old_instance:
        age = max(0.0, observed_at - started)
        result.update({
            "status": "awaiting_new_run_publish", "heartbeat_age_seconds": age,
            "old_run_instance_id": value.get("run_instance_id"),
        })
        if age > stale_seconds:
            _issue(
                issues, "error", "TRAINING_PROGRESS_NEW_INSTANCE_TIMEOUT",
                "%s resume worker 未在宽限期内发布新 progress。" % action,
                "%s resumed worker did not publish new progress within the grace period." % action,
                action,
            )
        return result
    source = run_manifest.get("source") if isinstance(run_manifest, Mapping) else None
    config = run_manifest.get("config") if isinstance(run_manifest, Mapping) else None
    updated = _safe_float(value.get("updated_unix_time"))
    successful = _safe_float(value.get("last_successful_progress_unix_time"))
    worker_started = _safe_float(value.get("started_unix_time"))
    total_epochs = (
        int(config["epochs"])
        if isinstance(config, Mapping) and _nonnegative_int(config.get("epochs"))
        else -1
    )
    numeric_fields_ok = all(
        _nonnegative_int(value.get(name))
        for name in (
            "epoch_index", "next_batch_in_epoch", "global_step",
            "examples_seen_in_epoch", "last_successful_step",
            "heartbeat_sequence",
        )
    )
    if numeric_fields_ok:
        numeric_fields_ok = (
            int(value["heartbeat_sequence"]) > 0
            and int(value["epoch_index"]) <= total_epochs
            and int(value["last_successful_step"]) <= int(value["global_step"])
        )
    last_loss = value.get("last_loss")
    grad_norm = value.get("grad_norm")
    scalar_fields_ok = (
        (last_loss is None or _finite_positive(last_loss))
        and (grad_norm is None or _finite_nonnegative(grad_norm))
    )
    validation_index = value.get("validation_batch_index")
    validation_total = value.get("validation_batches_total")
    validation_fields_ok = (
        (validation_index is None and validation_total is None)
        or (
            _nonnegative_int(validation_index)
            and _nonnegative_int(validation_total)
            and int(validation_index) > 0
            and int(validation_total) > 0
            and int(validation_index) <= int(validation_total)
        )
    )
    identity_ok = (
        value.get("schema_version") == "trajectory_training_progress_v1"
        and value.get("protocol_version") == "trajectory_diffusion_strict_five_ref_v2"
        and value.get("action") == action
        and isinstance(value.get("run_instance_id"), str) and bool(value.get("run_instance_id"))
        and isinstance(source, Mapping) and value.get("source") == source
        and isinstance(config, Mapping)
        and value.get("config_sha256") == _canonical_sha256(config)
        and value.get("device") == config.get("device")
        and numeric_fields_ok and scalar_fields_ok and validation_fields_ok
    )
    if job_status == "running":
        identity_ok = identity_ok and int(value.get("pid", -1)) == int(job.get("pid", -2))
    phase = value.get("phase")
    if phase not in {"init", "train", "validation", "checkpoint_commit", "complete"}:
        identity_ok = False
    if phase == "init":
        identity_ok = identity_ok and last_loss is None and grad_norm is None
    elif numeric_fields_ok:
        identity_ok = identity_ok and (
            _finite_positive(last_loss)
            and _finite_nonnegative(grad_norm)
            and int(value["last_successful_step"]) == int(value["global_step"])
        )
    if job_status == "running" and started is not None and worker_started is not None:
        identity_ok = identity_ok and worker_started >= started - 60.0
    timestamp_ok = (
        updated is not None and mtime is not None and worker_started is not None
        and worker_started <= updated
        and (successful is None or worker_started <= successful <= updated)
        and updated <= observed_at + 60.0 and mtime <= observed_at + 60.0
        and (successful is None or successful <= observed_at + 60.0)
        and abs(float(mtime) - float(updated)) <= 60.0
    )
    if phase != "init" and successful is None:
        timestamp_ok = False
    if phase == "complete" and numeric_fields_ok:
        identity_ok = identity_ok and (
            int(value["epoch_index"]) == total_epochs
            and int(value["next_batch_in_epoch"]) == 0
            and int(value["examples_seen_in_epoch"]) == 0
            and int(value["last_successful_step"]) == int(value["global_step"])
        )
    if not identity_ok or not timestamp_ok:
        _issue(
            issues, "error", "TRAINING_PROGRESS_IDENTITY_INVALID",
            "%s worker progress 的 PID/action/source/config/device/timestamp 无效。" % action,
            "%s worker progress PID/action/source/config/device/timestamp is invalid." % action,
            action,
        )
        result["status"] = "invalid"
        return result
    freshness = successful if successful is not None else updated
    age = max(0.0, observed_at - float(freshness))
    file_age = max(0.0, observed_at - float(mtime))
    result.update({
        "status": "fresh" if max(age, file_age) <= stale_seconds else "stalled",
        "heartbeat_age_seconds": age,
        "file_age_seconds": file_age,
        "run_instance_id": value.get("run_instance_id"),
        "phase": phase,
        "epoch_index": value.get("epoch_index"),
        "next_batch_in_epoch": value.get("next_batch_in_epoch"),
        "global_step": value.get("global_step"),
        "last_successful_step": value.get("last_successful_step"),
        "last_loss": value.get("last_loss"),
        "grad_norm": value.get("grad_norm"),
    })
    if job_status == "running" and max(age, file_age) > stale_seconds:
        _issue(
            issues, "error", "TRAINING_PROGRESS_STALLED",
            "%s worker 有活 PID，但有效进度已停滞 %.0f 秒。" % (action, max(age, file_age)),
            "%s worker has a live PID but useful progress stalled for %.0f seconds."
            % (action, max(age, file_age)),
            action,
        )
    if run_manifest and run_manifest.get("status") == "complete" and phase != "complete":
        _issue(
            issues, "error", "COMPLETE_PROGRESS_PHASE_INVALID",
            "%s 已完成但 worker progress phase 不是 complete。" % action,
            "%s is complete but worker progress phase is not complete." % action,
            action,
        )
    process_age = _safe_float(job.get("heartbeat_unix_time"))
    result["supervisor_observation_age_seconds"] = (
        None if process_age is None else max(0.0, observed_at - process_age)
    )
    return result


def build_report(
    config: Mapping[str, Any], now: Optional[float] = None, stale_seconds: float = 600.0,
    gpu_probe: Optional[Callable[[], Mapping[str, Any]]] = None,
    process_probe: Optional[Callable[[int, Sequence[str]], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build one read-only snapshot; callers may inject probes for tests."""
    observed_at = time.time() if now is None else float(now)
    run_root = Path(str(config["run_root"])).resolve()
    actions = [str(value) for value in config.get("actions", [])]
    total_epochs = int(config.get("training", {}).get("epochs", 100))
    issues: List[Dict[str, Any]] = []
    probe_process = process_probe or _default_process_probe
    state_path = run_root / "supervisor_status.json"
    state, state_error = _read_json(state_path) if state_path.is_file() else (None, None)
    if state_error:
        _issue(
            issues, "error", "SUPERVISOR_STATUS_UNREADABLE",
            "supervisor_status.json 不可读。", "supervisor_status.json is unreadable.",
        )
    state = state or {}
    supervisor_status = state.get("status", "not_started")
    supervisor_updated = _safe_float(state.get("updated_unix_time"))
    supervisor_age = None if supervisor_updated is None else max(0.0, observed_at - supervisor_updated)
    current_stage = state.get("current_stage")
    if supervisor_status == "failed":
        _issue(
            issues, "error", "SUPERVISOR_FAILED",
            "正式 supervisor 已标记 failed。", "Formal supervisor is marked failed.",
        )
    if supervisor_status in ("running", "running_gates_only"):
        if supervisor_updated is None:
            _issue(
                issues, "error", "SUPERVISOR_HEARTBEAT_MISSING",
                "运行中的 supervisor 缺少 updated_unix_time。",
                "Running supervisor has no updated_unix_time.",
            )
        elif supervisor_age is not None and supervisor_age > stale_seconds:
            _issue(
                issues, "error", "SUPERVISOR_HEARTBEAT_STALE",
                "supervisor heartbeat 已 %.0f 秒未更新。" % supervisor_age,
                "Supervisor heartbeat is stale by %.0f seconds." % supervisor_age,
            )
        try:
            supervisor_process = dict(probe_process(int(state.get("supervisor_pid", -1)), []))
        except (TypeError, ValueError, OSError) as exc:
            supervisor_process = {"alive": False, "probe_error": str(exc)}
        if not supervisor_process.get("alive"):
            _issue(
                issues, "error", "SUPERVISOR_PROCESS_MISSING",
                "supervisor 标记运行，但 PID 不存在。",
                "Supervisor is marked running, but its PID is absent.",
            )

    jobs = state.get("jobs") if isinstance(state.get("jobs"), dict) else {}
    action_reports: Dict[str, Any] = {}
    active_actions: List[str] = []
    for action in actions:
        action_issues_start = len(issues)
        action_root = run_root / "training" / action
        manifest_path = action_root / "run_manifest.json"
        metrics_path = action_root / "metrics.jsonl"
        best_path = action_root / "best_manifest.json"
        last_path = action_root / "last.pt"
        last_state_path = action_root / "last_state.json"
        progress_path = action_root / "training_progress.json"
        registry_path = action_root / "reference_registry.json"
        journal_path = action_root / "epoch_commit.json"
        pending = sorted(action_root.glob(".epoch_*_next.pt.pending")) if action_root.is_dir() else []
        artifacts = {
            "run_manifest": _file_info(manifest_path, observed_at),
            "metrics": _file_info(metrics_path, observed_at),
            "best_manifest": _file_info(best_path, observed_at),
            "last_checkpoint": _file_info(last_path, observed_at),
            "last_state": _file_info(last_state_path, observed_at),
            "training_progress": _file_info(progress_path, observed_at),
            "reference_registry": _file_info(registry_path, observed_at),
            "epoch_commit": _file_info(journal_path, observed_at),
            "pending_epoch_checkpoints": [_file_info(path, observed_at) for path in pending],
        }
        run_manifest, run_error = _read_json(manifest_path) if manifest_path.is_file() else (None, None)
        best_manifest, best_error = _read_json(best_path) if best_path.is_file() else (None, None)
        rows, row_failures = _read_jsonl(metrics_path) if metrics_path.is_file() else ([], [])
        job_name = "training/" + action
        job = jobs.get(job_name) if isinstance(jobs.get(job_name), dict) else {}
        job_status = job.get("status", "not_started")
        process: Dict[str, Any] = {"pid": job.get("pid"), "alive": False, "command_matches": False}
        if job_status == "running":
            active_actions.append(action)
            try:
                process = dict(probe_process(int(job.get("pid", -1)), job.get("command", [])))
            except (TypeError, ValueError, OSError) as exc:
                process = {"pid": job.get("pid"), "alive": False, "command_matches": False, "probe_error": str(exc)}
            if not process.get("alive") or not process.get("command_matches"):
                _issue(
                    issues, "error", "TRAINING_PROCESS_MISSING_OR_MISMATCH",
                    "%s job 标记 running，但 PID 不存在或命令不匹配。" % action,
                    "%s job is running in state, but PID is absent or command mismatches." % action,
                    action,
                )
        if job_status == "failed":
            _issue(
                issues, "error", "TRAINING_JOB_FAILED",
                "%s 训练 job 已标记 failed。" % action,
                "%s training job is marked failed." % action,
                action,
            )

        worker_progress = _check_worker_progress(
            action, action_root, job, str(job_status), process, run_manifest,
            observed_at, stale_seconds, issues,
        )
        process["heartbeat_unix_time"] = (
            None if worker_progress.get("heartbeat_age_seconds") is None
            else observed_at - float(worker_progress["heartbeat_age_seconds"])
        )
        process["heartbeat_age_seconds"] = worker_progress.get("heartbeat_age_seconds")
        process["worker_progress_status"] = worker_progress.get("status")

        any_started_artifact = any(
            artifacts[name]["exists"]
            for name in (
                "run_manifest", "metrics", "best_manifest", "last_checkpoint",
                "last_state", "training_progress", "reference_registry", "epoch_commit",
            )
        ) or bool(pending)
        if run_error:
            _issue(
                issues, "error", "RUN_MANIFEST_UNREADABLE",
                "%s run_manifest.json 不可读。" % action,
                "%s run_manifest.json is unreadable." % action,
                action,
            )
        if best_error:
            _issue(
                issues, "error", "BEST_MANIFEST_UNREADABLE",
                "%s best_manifest.json 不可读。" % action,
                "%s best_manifest.json is unreadable." % action,
                action,
            )
        for failure in row_failures:
            transient = (
                job_status == "running"
                and failure.get("is_last_line")
                and not failure.get("file_ended_with_newline")
            )
            _issue(
                issues, "warning" if transient else "error", "METRICS_JSONL_PARSE_ERROR",
                "%s metrics.jsonl 第 %s 行不可解析%s。" % (
                    action, failure.get("line"), "（可能正写入）" if transient else "",
                ),
                "%s metrics.jsonl line %s is invalid%s." % (
                    action, failure.get("line"), " (possibly being appended)" if transient else "",
                ),
                action,
            )

        run_status = run_manifest.get("status") if run_manifest else None
        if run_manifest:
            if (
                run_manifest.get("protocol_version")
                != "trajectory_diffusion_strict_five_ref_v2"
                or not runtime_determinism_matches(
                    run_manifest.get("runtime_determinism")
                )
            ):
                _issue(
                    issues, "error", "RUN_RUNTIME_DETERMINISM_MISMATCH",
                    "%s run manifest 缺少精确 strict runtime 契约。" % action,
                    "%s run manifest lacks the exact strict runtime contract." % action,
                    action,
                )
            if run_manifest.get("action") != action:
                _issue(
                    issues, "error", "RUN_ACTION_MISMATCH",
                    "%s run manifest 的 action 不匹配。" % action,
                    "%s run manifest action mismatches." % action,
                    action,
                )
            manifest_epochs = run_manifest.get("config", {}).get("epochs")
            if manifest_epochs != total_epochs:
                _issue(
                    issues, "error", "RUN_EPOCH_CONFIG_MISMATCH",
                    "%s run manifest 的 epoch 配置不是 %d。" % (action, total_epochs),
                    "%s run manifest epoch setting is not %d." % (action, total_epochs),
                    action,
                )
            if run_status not in ("running", "complete"):
                _issue(
                    issues, "error", "RUN_STATUS_INVALID",
                    "%s run manifest 状态无效：%s。" % (action, run_status),
                    "%s run manifest status is invalid: %s." % (action, run_status),
                    action,
                )
        counts = run_manifest.get("counts", {}) if run_manifest else {}
        metrics_age = artifacts["metrics"].get("age_seconds")
        transaction_files = []
        if artifacts["epoch_commit"].get("exists"):
            transaction_files.append(artifacts["epoch_commit"])
        transaction_files.extend(artifacts["pending_epoch_checkpoints"])
        transaction_present = bool(transaction_files)
        transaction_fresh = (
            transaction_present
            and job_status == "running"
            and process.get("alive") is True
            and process.get("command_matches") is True
            and all(
                _safe_float(item.get("age_seconds")) is not None
                and float(item["age_seconds"]) <= 120.0
                for item in transaction_files
            )
        )
        if transaction_present and not transaction_fresh:
            _issue(
                issues, "error", "STALE_EPOCH_TRANSACTION",
                "%s 存在 epoch_commit/pending，但不是 live worker 的 <=120s 新鲜事务。" % action,
                "%s has epoch_commit/pending files that are not a <=120s transaction of a live worker."
                % action,
                action,
            )
        checkpoint_publish_transient = (
            job_status == "running"
            and process.get("alive") is True
            and process.get("command_matches") is True
            and worker_progress.get("phase") == "checkpoint_commit"
            and _safe_float(worker_progress.get("file_age_seconds")) is not None
            and float(worker_progress["file_age_seconds"]) <= 120.0
        )
        metric_report = _check_metrics(
            action, rows, total_epochs, counts,
            transaction_pending=transaction_fresh,
            metrics_age=metrics_age,
            run_complete=run_status == "complete",
            issues=issues,
        )
        completed_epochs = metric_report["train_epochs"]
        if completed_epochs and not last_path.is_file():
            _issue(
                issues, "error", "LAST_CHECKPOINT_MISSING",
                "%s 已完成 epoch，但 last.pt 缺失。" % action,
                "%s has completed epochs but last.pt is missing." % action,
                action,
            )
        if run_status == "complete":
            for name in (
                "last_checkpoint", "last_state", "training_progress",
                "reference_registry", "best_manifest",
            ):
                if not artifacts[name]["exists"]:
                    _issue(
                        issues, "error", "COMPLETE_ARTIFACT_MISSING",
                        "%s 标记 complete，但 %s 缺失。" % (action, name),
                        "%s is complete but %s is missing." % (action, name),
                        action,
                    )
        if any_started_artifact and run_manifest is None and job_status != "running":
            _issue(
                issues, "error", "PARTIAL_RUN_WITHOUT_MANIFEST",
                "%s 有部分训练产物，但无可读 run manifest 且进程未运行。" % action,
                "%s has partial artifacts without a readable run manifest or live job." % action,
                action,
            )

        best_report = _check_best(
            action, action_root, best_manifest, metric_report, issues,
            transaction_pending=transaction_fresh,
        )
        best_integrity = _check_best_checkpoint_integrity(
            action, action_root, best_manifest, run_manifest, issues,
            transient_checkpoint_publish=checkpoint_publish_transient,
        )
        last_integrity = _check_last_checkpoint_integrity(
            action, action_root, run_manifest, metric_report, issues,
            transient_checkpoint_publish=checkpoint_publish_transient,
            observed_at=observed_at,
        )
        configured_log = job.get("log") if isinstance(job.get("log"), str) else None
        log_path = Path(configured_log) if configured_log else run_root / "logs" / ("training__%s.log" % action)
        log_report = _scan_log(log_path)
        log_report["file"] = _file_info(log_path, observed_at)
        if log_report.get("read_error"):
            _issue(
                issues, "warning", "TRAINING_LOG_UNREADABLE",
                "%s 训练日志不可读。" % action,
                "%s training log is unreadable." % action,
                action,
            )
        if log_report.get("nonfinite_matches"):
            _issue(
                issues, "error", "NONFINITE_TOKEN_IN_TRAINING_LOG",
                "%s 训练日志发现 NaN/Inf/non-finite。" % action,
                "%s training log contains NaN/Inf/non-finite." % action,
                action,
            )

        if run_status == "complete" and len(completed_epochs) == total_epochs:
            action_status = "complete"
        elif job_status == "running":
            action_status = "running"
        elif not any_started_artifact:
            action_status = "not_started"
        else:
            action_status = "incomplete"
        if run_status == "running" and job_status not in ("running", "interrupted"):
            _issue(
                issues, "error", "RUNNING_MANIFEST_WITHOUT_RUNNING_JOB",
                "%s run manifest 仍为 running，但 supervisor 中没有运行/待恢复 job。" % action,
                "%s run manifest is running without a running/interrupted supervisor job." % action,
                action,
            )
        action_issue_slice = issues[action_issues_start:]
        action_reports[action] = {
            "status": action_status,
            "configured_device": config.get("action_device", {}).get(action),
            "job_status": job_status,
            "process": process,
            "worker_progress": worker_progress,
            "run_manifest_status": run_status,
            "completed_epoch": max(completed_epochs) if completed_epochs else 0,
            "total_epochs": total_epochs,
            "artifacts": artifacts,
            "metrics": metric_report,
            "best": best_report,
            "best_checkpoint_integrity": best_integrity,
            "last_checkpoint_integrity": last_integrity,
            "log": log_report,
            "error_count": sum(item["severity"] == "error" for item in action_issue_slice),
            "warning_count": sum(item["severity"] == "warning" for item in action_issue_slice),
        }

    gpu = dict((gpu_probe or query_gpus)())
    if active_actions and not gpu.get("available"):
        _issue(
            issues, "warning", "GPU_TELEMETRY_UNAVAILABLE",
            "有训练正在运行，但 nvidia-smi GPU 遥测不可用。",
            "Training is active, but nvidia-smi GPU telemetry is unavailable.",
        )
    gpu_by_index = {int(item["index"]): item for item in gpu.get("gpus", []) if "index" in item}
    active_device_indices = set()
    for action in active_actions:
        try:
            active_device_indices.add(int(str(config.get("action_device", {}).get(action, "")).split(":")[-1]))
        except ValueError:
            pass
    for item in gpu.get("gpus", []):
        if int(item.get("index", -1)) not in active_device_indices:
            continue
        temperature = _safe_float(item.get("temperature_c"))
        fraction = _safe_float(item.get("memory_fraction"))
        if temperature is not None and temperature >= 90.0:
            _issue(
                issues, "error", "GPU_TEMPERATURE_CRITICAL",
                "GPU %s 温度 %.1f°C，达到严重阈值。" % (item.get("index"), temperature),
                "GPU %s temperature %.1f°C reached the critical threshold." % (item.get("index"), temperature),
            )
        elif temperature is not None and temperature >= 85.0:
            _issue(
                issues, "warning", "GPU_TEMPERATURE_HIGH",
                "GPU %s 温度 %.1f°C 偏高。" % (item.get("index"), temperature),
                "GPU %s temperature %.1f°C is high." % (item.get("index"), temperature),
            )
        if fraction is not None and fraction >= 0.98:
            _issue(
                issues, "warning", "GPU_MEMORY_NEAR_CAPACITY",
                "GPU %s 显存使用达到 %.1f%%。" % (item.get("index"), 100.0 * fraction),
                "GPU %s memory use reached %.1f%%." % (item.get("index"), 100.0 * fraction),
            )
    for action in active_actions:
        if not gpu.get("available"):
            continue
        device = str(config.get("action_device", {}).get(action, ""))
        try:
            index = int(device.split(":")[-1])
        except ValueError:
            continue
        telemetry = gpu_by_index.get(index)
        if telemetry is None:
            _issue(
                issues, "warning", "ACTIVE_GPU_MISSING_FROM_TELEMETRY",
                "%s 的设备 %s 没有 GPU 遥测。" % (action, device),
                "%s device %s is absent from GPU telemetry." % (action, device),
                action,
            )
            continue
        utilization = _safe_float(telemetry.get("utilization_percent"))
        heartbeat_age = action_reports[action]["process"].get("heartbeat_age_seconds")
        started = _safe_float(jobs.get("training/" + action, {}).get("started_unix_time"))
        age = None if started is None else observed_at - started
        if (
            utilization is not None and utilization < 5.0
            and heartbeat_age is not None and heartbeat_age <= stale_seconds
            and age is not None and age > 180.0
        ):
            _issue(
                issues, "warning", "ACTIVE_GPU_LOW_INSTANT_UTILIZATION",
                "%s 正在运行，但 %s 当前瞬时利用率仅 %.1f%%；单次采样只告警。" % (action, device, utilization),
                "%s is running but %s instantaneous utilization is %.1f%%; one sample is warning-only." % (
                    action, device, utilization,
                ),
                action,
            )

    # GPU issues are appended after the per-action filesystem checks, so
    # recompute the compact table counters once all action-attributed issues
    # are known.
    for action, value in action_reports.items():
        attributed = [item for item in issues if item.get("action") == action]
        value["error_count"] = sum(item["severity"] == "error" for item in attributed)
        value["warning_count"] = sum(item["severity"] == "warning" for item in attributed)

    errors = [item for item in issues if item["severity"] == "error"]
    warnings = [item for item in issues if item["severity"] == "warning"]
    statuses = [item["status"] for item in action_reports.values()]
    if errors:
        overall = "unhealthy"
    elif warnings:
        overall = "warning"
    elif statuses and all(value == "complete" for value in statuses):
        overall = "complete"
    elif any(value == "running" for value in statuses):
        overall = "healthy"
    elif all(value == "not_started" for value in statuses):
        overall = "not_started"
    else:
        overall = "idle"
    return {
        "schema_version": "trajectory_formal_training_health_v1",
        "read_only_monitor": True,
        "checked_unix_time": observed_at,
        "checked_local_time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(observed_at)),
        "run_root": str(run_root),
        "overall_status": overall,
        "supervisor": {
            "status_file": _file_info(state_path, observed_at),
            "status": supervisor_status,
            "current_stage": current_stage,
            "updated_unix_time": supervisor_updated,
            "heartbeat_age_seconds": supervisor_age,
            "pid": state.get("supervisor_pid"),
        },
        "stale_threshold_seconds": float(stale_seconds),
        "actions": action_reports,
        "gpu": gpu,
        "issues": issues,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "monitoring_note_zh": "该报告只读训练产物；单次 loss 上升或单次低 GPU 利用率只告警，不自动停止或修改训练。",
        "monitoring_note_en": "This report only reads training artifacts; one loss increase or one low-GPU sample is warning-only and never changes or stops training.",
    }


def render_summary(report: Mapping[str, Any]) -> str:
    rows = [
        "# 正式训练健康检查 / Formal training health",
        "",
        "- 时间 / checked: `%s`" % report.get("checked_local_time"),
        "- 总体 / overall: **%s**" % report.get("overall_status"),
        "- Supervisor: `%s`, stage=`%s`, heartbeat_age=%s s" % (
            report.get("supervisor", {}).get("status"),
            report.get("supervisor", {}).get("current_stage"),
            "-" if report.get("supervisor", {}).get("heartbeat_age_seconds") is None
            else "%.1f" % report.get("supervisor", {}).get("heartbeat_age_seconds"),
        ),
        "- Errors/Warnings: `%d / %d`" % (report.get("error_count", 0), report.get("warning_count", 0)),
        "",
        "| Action | State | Epoch | Train loss | Val loss | Best val (epoch) | Heartbeat | E/W |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for action, value in report.get("actions", {}).items():
        train = value.get("metrics", {}).get("train_loss", {}).get("latest")
        val = value.get("metrics", {}).get("validation_loss", {}).get("latest")
        best = value.get("best", {})
        heartbeat = value.get("process", {}).get("heartbeat_age_seconds")
        rows.append(
            "| %s | %s | %d/%d | %s | %s | %s (%s) | %s | %d/%d |" % (
                action,
                value.get("status"),
                value.get("completed_epoch", 0),
                value.get("total_epochs", 0),
                "-" if train is None else "%.6g" % train,
                "-" if val is None else "%.6g" % val,
                "-" if best.get("val_loss") is None else "%.6g" % best.get("val_loss"),
                "-" if best.get("selected_epoch") is None else best.get("selected_epoch"),
                "-" if heartbeat is None else "%.1fs" % heartbeat,
                value.get("error_count", 0),
                value.get("warning_count", 0),
            )
        )
    rows.extend(["", "## GPU", ""])
    if report.get("gpu", {}).get("available"):
        rows.extend([
            "| GPU | Util | Memory | Temp |",
            "| ---: | ---: | ---: | ---: |",
        ])
        for gpu in report.get("gpu", {}).get("gpus", []):
            rows.append(
                "| %s | %.1f%% | %.0f/%.0f MiB | %.1f°C |" % (
                    gpu.get("index"), gpu.get("utilization_percent", 0.0),
                    gpu.get("memory_used_mib", 0.0), gpu.get("memory_total_mib", 0.0),
                    gpu.get("temperature_c", 0.0),
                )
            )
    else:
        rows.append("- unavailable: `%s`" % report.get("gpu", {}).get("error"))
    if report.get("issues"):
        rows.extend(["", "## 告警 / Issues", ""])
        for item in report["issues"]:
            prefix = "[%s][%s]" % (item["severity"].upper(), item["code"])
            action = "[%s]" % item["action"] if item.get("action") else ""
            rows.append("- %s%s %s / %s" % (prefix, action, item["message_zh"], item["message_en"]))
    rows.extend([
        "",
        "> 只读 / Read-only: 不加载 checkpoint tensor，不改模型、数据、loss、阈值或训练进程。",
        "",
    ])
    return "\n".join(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only formal diffusion training health monitor.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--once", action="store_true", help="Write one snapshot and exit.")
    parser.add_argument("--interval-seconds", type=float, default=1200.0)
    parser.add_argument("--stale-seconds", type=float, default=600.0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--fail-on-error", action="store_true", help="Return 2 when the snapshot is unhealthy.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.interval_seconds <= 0 or args.stale_seconds <= 0:
        raise ValueError("interval-seconds and stale-seconds must be positive")
    config, error = _read_json(args.config.resolve())
    if config is None:
        raise ValueError("cannot read config %s: %s" % (args.config, error))
    run_root = Path(str(config["run_root"])).resolve()
    json_output = args.json_output or run_root / "monitoring" / "training_health_latest.json"
    summary_output = args.summary_output or run_root / "monitoring" / "training_health_latest.md"
    lock_path = run_root / "monitoring" / "training_health.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_stream.close()
        raise RuntimeError("another formal training health monitor owns %s" % lock_path)
    while True:
        report = build_report(config, stale_seconds=args.stale_seconds)
        summary = render_summary(report)
        _atomic_json(json_output.resolve(), report)
        _atomic_text(summary_output.resolve(), summary)
        print(summary, flush=True)
        if args.once:
            return 2 if args.fail_on_error and report["error_count"] else 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
