import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime_determinism import EXPECTED_RUNTIME_DETERMINISM
from scripts import report_formal_training_health as health


class FormalTrainingHealthMonitorTest(unittest.TestCase):
    NOW = 2_000_000_000.0

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.action_root = self.run / "training" / "tap"
        self.action_root.mkdir(parents=True)
        (self.run / "logs").mkdir(parents=True)
        self.config = {
            "run_root": str(self.run),
            "actions": ["tap"],
            "action_device": {"tap": "cuda:0"},
            "training": {"epochs": 100},
        }

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def gpu():
        return {
            "available": True,
            "error": None,
            "gpus": [{
                "index": 0,
                "name": "synthetic",
                "utilization_percent": 80.0,
                "memory_used_mib": 8000.0,
                "memory_total_mib": 24000.0,
                "memory_fraction": 1.0 / 3.0,
                "temperature_c": 65.0,
            }],
        }

    def _write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _healthy_fixture(self):
        self._write_json(self.run / "supervisor_status.json", {
            "status": "complete",
            "current_stage": None,
            "updated_unix_time": self.NOW - 1,
            "supervisor_pid": 11,
            "jobs": {"training/tap": {"status": "complete"}},
        })
        run_manifest = {
            "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
            "runtime_determinism": dict(EXPECTED_RUNTIME_DETERMINISM),
            "action": "tap",
            "status": "complete",
            "config": {"epochs": 100, "device": "cuda:0"},
            "source": {
                "corpus_sha256": "1" * 64,
                "split_sha256": "2" * 64,
                "reference_registry_sha256": "3" * 64,
            },
            "counts": {"train": 70, "val": 10, "test_reserved": 20},
        }
        self._write_json(self.action_root / "run_manifest.json", run_manifest)
        rows = []
        val_losses = {}
        for epoch in range(1, 101):
            rows.append({
                "type": "train_epoch",
                "completed_epoch": epoch,
                "global_step": epoch * 3,
                "loss": 1.0 / (epoch + 1.0),
                "batches_total_in_epoch": 3,
                "examples_total_in_epoch": 70,
                "valid_feature_count_total": 700.0,
                "full_train_split_consumed": True,
                "unix_time": self.NOW - (101 - epoch),
            })
            if epoch in (20, 40, 60, 80, 100):
                loss = 1.0 / epoch
                val_losses[epoch] = loss
                rows.append({
                    "type": "validation",
                    "completed_epoch": epoch,
                    "fraction": epoch / 100.0,
                    "val_loss": loss,
                    "n_examples": 10,
                    "n_batches": 2,
                    "valid_feature_count": 100.0,
                    "ema_weights": True,
                    "full_validation_split": True,
                    "global_step": epoch * 3,
                    "unix_time": self.NOW - (101 - epoch),
                })
        (self.action_root / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        history = []
        for epoch, loss in val_losses.items():
            checkpoint = self.action_root / ("best_epoch_%04d.pt" % epoch)
            checkpoint.write_bytes(b"best")
            history.append({
                "path": str(checkpoint),
                "filename": checkpoint.name,
                "completed_epoch": epoch,
                "global_step": epoch * 3,
                "val_loss": loss,
                "source_sha256": "1" * 64,
                "split_sha256": "2" * 64,
                "reference_registry_sha256": "3" * 64,
                "checkpoint_sha256": health._sha256_file(checkpoint),
                "checkpoint_role": "validation_selected_best",
                "inference_weights": "ema.shadow",
            })
        self._write_json(self.action_root / "best_manifest.json", {
            "checkpoint_role": "validation_selected_best",
            "inference_weights": "ema.shadow",
            "selection_split": "val",
            "selection_metric": "full_val_masked_epsilon_mse_ema",
            "lower_is_better": True,
            "best": history[-1],
            "history": history,
            "test_used_for_selection": False,
            "source": run_manifest["source"],
        })
        (self.action_root / "last.pt").write_bytes(b"last")
        self._write_json(self.action_root / "last_state.json", {
            "schema_version": "trajectory_last_state_v1",
            "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
            "run_instance_id": "fixture-run",
            "action": "tap",
            "checkpoint_path": str((self.action_root / "last.pt").resolve()),
            "checkpoint_sha256": health._sha256_file(self.action_root / "last.pt"),
            "checkpoint_size_bytes": 4,
            "progress": {
                "epoch_index": 100,
                "next_batch_in_epoch": 0,
                "examples_seen_in_epoch": 0,
                "global_step": 300,
            },
            "source": run_manifest["source"],
            "config_sha256": health._canonical_sha256(run_manifest["config"]),
            "updated_unix_time": self.NOW - 1,
        })
        self._write_json(self.action_root / "training_progress.json", {
            "schema_version": "trajectory_training_progress_v1",
            "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
            "run_instance_id": "fixture-run",
            "action": "tap",
            "pid": 999,
            "source": run_manifest["source"],
            "config_sha256": health._canonical_sha256(run_manifest["config"]),
            "phase": "complete",
            "epoch_index": 100,
            "next_batch_in_epoch": 0,
            "global_step": 300,
            "examples_seen_in_epoch": 0,
            "last_successful_step": 300,
            "last_successful_progress_unix_time": self.NOW - 1,
            "last_loss": 0.01,
            "grad_norm": 0.5,
            "device": "cuda:0",
            "validation_batch_index": None,
            "validation_batches_total": None,
            "heartbeat_sequence": 100,
            "started_unix_time": self.NOW - 1000,
            "updated_unix_time": self.NOW - 1,
        })
        os.utime(
            self.action_root / "training_progress.json",
            (self.NOW - 1, self.NOW - 1),
        )
        self._write_json(self.action_root / "reference_registry.json", {"ok": True})
        (self.run / "logs" / "training__tap.log").write_text("training complete\n", encoding="utf-8")

    def _report(self, process_probe=None):
        return health.build_report(
            self.config,
            now=self.NOW,
            stale_seconds=600,
            gpu_probe=self.gpu,
            process_probe=process_probe,
        )

    def _make_running(self, progress_age=1.0, phase="train"):
        state = {
            "status": "running",
            "current_stage": "training",
            "updated_unix_time": self.NOW - 1,
            "supervisor_pid": 11,
            "jobs": {"training/tap": {
                "status": "running",
                "pid": 123,
                "started_unix_time": self.NOW - 5000,
                "heartbeat_unix_time": self.NOW - 1,
                "command": [
                    "python", "scripts/train_trajectory_diffusion.py",
                    "--action", "tap", "--output-dir", str(self.action_root),
                ],
            }},
        }
        self._write_json(self.run / "supervisor_status.json", state)
        manifest = json.loads((self.action_root / "run_manifest.json").read_text())
        manifest["status"] = "running"
        self._write_json(self.action_root / "run_manifest.json", manifest)
        progress_path = self.action_root / "training_progress.json"
        progress = json.loads(progress_path.read_text())
        progress.update({
            "pid": 123,
            "phase": phase,
            "updated_unix_time": self.NOW - progress_age,
            "last_successful_progress_unix_time": self.NOW - progress_age,
        })
        self._write_json(progress_path, progress)
        os.utime(progress_path, (self.NOW - progress_age, self.NOW - progress_age))

        def alive(pid, _command):
            return {"pid": pid, "alive": True, "command_matches": True}

        return alive

    def test_complete_healthy_run(self):
        self._healthy_fixture()
        report = self._report()
        self.assertEqual(report["overall_status"], "complete")
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(report["warning_count"], 0)
        self.assertEqual(report["actions"]["tap"]["completed_epoch"], 100)
        self.assertEqual(report["actions"]["tap"]["metrics"]["validation_epochs"], [20, 40, 60, 80, 100])
        self.assertTrue(report["actions"]["tap"]["best"]["checkpoint_exists"])

    def test_nonfinite_loss_is_unhealthy(self):
        self._healthy_fixture()
        path = self.action_root / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        next(row for row in rows if row["type"] == "train_epoch" and row["completed_epoch"] == 50)["loss"] = float("nan")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        report = self._report()
        self.assertEqual(report["overall_status"], "unhealthy")
        self.assertIn("NONFINITE_OR_NONPOSITIVE_TRAIN_LOSS", [item["code"] for item in report["issues"]])

    def test_epoch_gap_is_unhealthy(self):
        self._healthy_fixture()
        path = self.action_root / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [
            row for row in rows
            if not (row["type"] == "train_epoch" and row["completed_epoch"] == 50)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        report = self._report()
        self.assertEqual(report["overall_status"], "unhealthy")
        codes = [item["code"] for item in report["issues"]]
        self.assertIn("TRAIN_EPOCH_GAP_OR_DUPLICATE", codes)
        self.assertIn("COMPLETE_RUN_METRICS_INCOMPLETE", codes)

    def test_stale_running_heartbeat_is_unhealthy(self):
        self._healthy_fixture()
        self._write_json(self.run / "supervisor_status.json", {
            "status": "running",
            "current_stage": "training",
            "updated_unix_time": self.NOW - 700,
            "supervisor_pid": 11,
            "jobs": {"training/tap": {
                "status": "running",
                "pid": 123,
                "started_unix_time": self.NOW - 5000,
                "heartbeat_unix_time": self.NOW - 700,
                "command": ["python", "scripts/train_trajectory_diffusion.py", "--action", "tap"],
            }},
        })

        def alive(_pid, _command):
            return {"pid": 123, "alive": True, "command_matches": True}

        report = self._report(process_probe=alive)
        self.assertEqual(report["overall_status"], "unhealthy")
        codes = [item["code"] for item in report["issues"]]
        self.assertIn("SUPERVISOR_HEARTBEAT_STALE", codes)
        # A wrong-PID sidecar modified after the new job start is not stale
        # inherited state; it is an immediate identity violation.
        self.assertIn("TRAINING_PROGRESS_IDENTITY_INVALID", codes)

    def test_one_latest_loss_increase_is_warning_only(self):
        self._healthy_fixture()
        path = self.action_root / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        train = [row for row in rows if row["type"] == "train_epoch"]
        train[-1]["loss"] = train[-2]["loss"] * 1.01
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        report = self._report()
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(report["overall_status"], "warning")
        issue = next(item for item in report["issues"] if item["code"] == "TRAIN_LATEST_INCREASE")
        self.assertEqual(issue["severity"], "warning")

    def test_live_pid_with_stale_worker_progress_is_unhealthy(self):
        self._healthy_fixture()
        alive = self._make_running(progress_age=700.0)
        report = self._report(process_probe=alive)
        self.assertIn("TRAINING_PROGRESS_STALLED", [item["code"] for item in report["issues"]])
        self.assertEqual(report["overall_status"], "unhealthy")

    def test_fresh_validation_worker_progress_is_not_stalled(self):
        self._healthy_fixture()
        alive = self._make_running(progress_age=1.0, phase="validation")
        report = self._report(process_probe=alive)
        codes = [item["code"] for item in report["issues"]]
        self.assertNotIn("TRAINING_PROGRESS_STALLED", codes)
        self.assertNotIn("TRAINING_PROGRESS_IDENTITY_INVALID", codes)

    def test_stale_transaction_does_not_hide_missing_validation(self):
        self._healthy_fixture()
        manifest = json.loads((self.action_root / "run_manifest.json").read_text())
        manifest["status"] = "running"
        self._write_json(self.action_root / "run_manifest.json", manifest)
        path = self.action_root / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [
            row for row in rows
            if not (row.get("type") == "validation" and row.get("completed_epoch") == 100)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        journal = self.action_root / "epoch_commit.json"
        self._write_json(journal, {"schema_version": "trajectory_epoch_commit_v1"})
        os.utime(journal, (self.NOW - 500, self.NOW - 500))
        report = self._report()
        codes = [item["code"] for item in report["issues"]]
        self.assertIn("STALE_EPOCH_TRANSACTION", codes)
        self.assertIn("MISSING_VALIDATION_MILESTONE", codes)
        missing = next(item for item in report["issues"] if item["code"] == "MISSING_VALIDATION_MILESTONE")
        self.assertEqual(missing["severity"], "error")

    def test_fresh_live_transaction_only_temporarily_warns(self):
        self._healthy_fixture()
        alive = self._make_running(progress_age=1.0, phase="checkpoint_commit")
        path = self.action_root / "metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [
            row for row in rows
            if not (row.get("type") == "validation" and row.get("completed_epoch") == 100)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        journal = self.action_root / "epoch_commit.json"
        self._write_json(journal, {"schema_version": "trajectory_epoch_commit_v1"})
        os.utime(journal, (self.NOW - 1, self.NOW - 1))
        report = self._report(process_probe=alive)
        matching = [item for item in report["issues"] if item["code"] == "MISSING_VALIDATION_MILESTONE"]
        self.assertEqual([item["severity"] for item in matching], ["warning"])
        self.assertNotIn("STALE_EPOCH_TRANSACTION", [item["code"] for item in report["issues"]])

    def test_best_tamper_and_zero_byte_last_are_hard_errors(self):
        self._healthy_fixture()
        best = json.loads((self.action_root / "best_manifest.json").read_text())
        Path(best["history"][0]["path"]).write_bytes(b"tampered")
        (self.action_root / "last.pt").write_bytes(b"")
        report = self._report()
        codes = [item["code"] for item in report["issues"]]
        self.assertIn("BEST_CHECKPOINT_INTEGRITY_INVALID", codes)
        self.assertIn("LAST_CHECKPOINT_EMPTY", codes)
        self.assertEqual(report["overall_status"], "unhealthy")

    def test_fresh_checkpoint_publish_race_is_warning_then_must_converge(self):
        self._healthy_fixture()
        alive = self._make_running(progress_age=1.0, phase="checkpoint_commit")
        (self.action_root / "last.pt").write_bytes(b"new-last-not-yet-bound")
        os.utime(
            self.action_root / "last.pt",
            (self.NOW - 1.0, self.NOW - 1.0),
        )
        report = self._report(process_probe=alive)
        issues = [item for item in report["issues"] if item["code"] == "LAST_CHECKPOINT_STATE_MISMATCH"]
        self.assertEqual([item["severity"] for item in issues], ["warning"])

    def test_checkpoint_phase_never_excuses_best_or_empty_last_corruption(self):
        self._healthy_fixture()
        alive = self._make_running(progress_age=1.0, phase="checkpoint_commit")
        best = json.loads((self.action_root / "best_manifest.json").read_text())
        Path(best["history"][0]["path"]).write_bytes(b"tampered-during-unrelated-commit")
        (self.action_root / "last.pt").write_bytes(b"")
        os.utime(
            self.action_root / "last.pt",
            (self.NOW - 1.0, self.NOW - 1.0),
        )
        report = self._report(process_probe=alive)
        by_code = {}
        for item in report["issues"]:
            by_code.setdefault(item["code"], []).append(item["severity"])
        self.assertEqual(by_code["BEST_CHECKPOINT_INTEGRITY_INVALID"], ["error"])
        self.assertEqual(by_code["LAST_CHECKPOINT_EMPTY"], ["error"])

    def test_old_last_mismatch_is_error_even_during_fresh_checkpoint_phase(self):
        self._healthy_fixture()
        alive = self._make_running(progress_age=1.0, phase="checkpoint_commit")
        (self.action_root / "last.pt").write_bytes(b"old-unbound-corruption")
        os.utime(
            self.action_root / "last.pt",
            (self.NOW - 500.0, self.NOW - 500.0),
        )
        report = self._report(process_probe=alive)
        issues = [
            item for item in report["issues"]
            if item["code"] == "LAST_CHECKPOINT_STATE_MISMATCH"
        ]
        self.assertEqual([item["severity"] for item in issues], ["error"])

    def test_progress_nonfinite_counter_and_complete_instance_mismatch_are_errors(self):
        self._healthy_fixture()
        alive = self._make_running(progress_age=1.0, phase="train")
        path = self.action_root / "training_progress.json"
        value = json.loads(path.read_text())
        value["last_loss"] = float("nan")
        self._write_json(path, value)
        os.utime(path, (self.NOW - 1.0, self.NOW - 1.0))
        report = self._report(process_probe=alive)
        self.assertIn(
            "TRAINING_PROGRESS_IDENTITY_INVALID",
            [item["code"] for item in report["issues"]],
        )

        self._healthy_fixture()
        state_path = self.action_root / "last_state.json"
        state = json.loads(state_path.read_text())
        state["run_instance_id"] = "different-completed-run"
        self._write_json(state_path, state)
        report = self._report()
        self.assertIn(
            "LAST_CHECKPOINT_STATE_MISMATCH",
            [item["code"] for item in report["issues"]],
        )

    def test_gpu_unavailable_is_scoped_to_active_training_once(self):
        self._healthy_fixture()
        unavailable = lambda: {"available": False, "error": "temporary", "gpus": []}
        complete = health.build_report(self.config, now=self.NOW, gpu_probe=unavailable)
        self.assertNotIn("GPU_TELEMETRY_UNAVAILABLE", [item["code"] for item in complete["issues"]])
        alive = self._make_running(progress_age=1.0)
        active = health.build_report(
            self.config, now=self.NOW, gpu_probe=unavailable, process_probe=alive,
        )
        self.assertEqual(
            sum(item["code"] == "GPU_TELEMETRY_UNAVAILABLE" for item in active["issues"]), 1,
        )

    def test_gpu_query_retries_transient_failure(self):
        failed = SimpleNamespace(returncode=9, stdout="", stderr="temporary")
        passed = SimpleNamespace(
            returncode=0,
            stdout="0, GPU, 80, 1000, 24000, 65\n",
            stderr="",
        )
        with mock.patch.object(health.subprocess, "run", side_effect=[failed, passed]) as run, mock.patch.object(
            health.time, "sleep"
        ):
            result = health.query_gpus(attempts=3, retry_delay_seconds=0.01)
        self.assertTrue(result["available"])
        self.assertEqual(run.call_count, 2)

    def test_stable_checkpoint_hash_retries_atomic_replacement(self):
        path = self.root / "changing.pt"
        path.write_bytes(b"old")
        replacement = self.root / "replacement.pt"
        replacement.write_bytes(b"new-stable")
        original = health._sha256_file
        calls = {"count": 0}

        def hash_then_replace(observed):
            digest = original(observed)
            calls["count"] += 1
            if calls["count"] == 1:
                os.replace(replacement, path)
            return digest

        with mock.patch.object(health, "_sha256_file", side_effect=hash_then_replace):
            digest, stat = health._stable_sha256_file(path, attempts=2)
        self.assertEqual(calls["count"], 2)
        self.assertEqual(digest, original(path))
        self.assertEqual(stat.st_size, len(b"new-stable"))

    def test_log_nan_is_detected_and_outputs_are_atomic_json_and_markdown(self):
        self._healthy_fixture()
        (self.run / "logs" / "training__tap.log").write_text(
            "FloatingPointError: non-finite training loss NaN\n", encoding="utf-8"
        )
        report = self._report()
        self.assertIn("NONFINITE_TOKEN_IN_TRAINING_LOG", [item["code"] for item in report["issues"]])
        json_path = self.root / "latest.json"
        md_path = self.root / "latest.md"
        health._atomic_json(json_path, report)
        health._atomic_text(md_path, health.render_summary(report))
        loaded = json.loads(json_path.read_text())
        self.assertEqual(loaded["schema_version"], "trajectory_formal_training_health_v1")
        self.assertIn("Formal training health", md_path.read_text())
        self.assertEqual(list(self.root.glob(".latest.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
