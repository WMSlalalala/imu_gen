import copy
import hashlib
import json
import os
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import torch

from orchestration import formal_supervisor as fs


PROJECT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT / "orchestration" / "formal_pipeline_config.json"


class FormalSupervisorStateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        with BASE_CONFIG.open(encoding="utf-8") as stream:
            self.config = json.load(stream)
        self.config["run_root"] = str(root / "run")
        self.config["gate_root"] = str(root / "gates")
        self.config["formal_launch_authorized"] = False

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _close(supervisor):
        supervisor.lock_stream.close()

    @staticmethod
    def _preflight_report():
        return {
            "passed": True,
            "ready_for_gates": True,
            "ready_to_launch": True,
            "formal_launch_authorized": True,
            "runtime_inputs_required": True,
            "blockers": [],
            "cli_checks": [],
            "corpus_files_present": 5,
        }

    def _write_canonical_registry(self, supervisor, action, path):
        from generation.protocol import ReferenceRegistry

        split = fs.read_json(supervisor.p["split"])
        entries = {
            (action, int(user_id), split_name): tuple(
                int(user_id) * 10 + offset for offset in range(5)
            )
            for split_name in ("train", "val", "test")
            for user_id in split[split_name + "_users"]
        }
        registry = ReferenceRegistry.build(entries, self.config["split_sha256"])
        payload = ReferenceRegistry._payload(registry.entries, self.config["split_sha256"])
        payload["registry_sha256"] = registry.registry_sha256
        fs.atomic_json(path, payload)
        return registry.registry_sha256

    def _complete_gates_without_commands(self, supervisor):
        gate_report = dict(self._preflight_report())
        gate_report.update({
            "ready_to_launch": False,
            "formal_launch_authorized": False,
            "runtime_inputs_required": False,
        })
        evidence = {
            "schema_version": "trajectory_launch_gate_evidence_v1",
            "formal_launch_authorized_during_gates": False,
            "config_sha256": fs.experiment_config_sha256(supervisor.config),
            "split_sha256": supervisor.config["split_sha256"],
            "corpus_sha256_by_action": {},
            "artifacts": {
                name: {"path": name, "sha256": "0" * 64, "size_bytes": 0}
                for name in ("corpus_audit", "e2e_smoke", "condition_preflight")
            },
        }
        with mock.patch.object(fs, "preflight", return_value=gate_report), mock.patch.object(
            supervisor, "_run_launch_gates", return_value=None
        ), mock.patch.object(
            supervisor, "_current_launch_gate_evidence", return_value=evidence
        ):
            supervisor.run_gates_only()

    def test_experiment_identity_excludes_only_operational_authorization(self):
        authorized = copy.deepcopy(self.config)
        authorized["formal_launch_authorized"] = True
        self.assertEqual(
            fs.experiment_config_sha256(self.config),
            fs.experiment_config_sha256(authorized),
        )
        changed = copy.deepcopy(authorized)
        changed["training"]["learning_rate"] *= 2
        self.assertNotEqual(
            fs.experiment_config_sha256(self.config),
            fs.experiment_config_sha256(changed),
        )

    def test_validate_config_locks_training_generation_seed_and_batch(self):
        with BASE_CONFIG.open(encoding="utf-8") as stream:
            base = json.load(stream)
        fs.validate_config(base)
        for section, field, value in (
            ("training", "seed", 43),
            ("generation", "seed", 7),
            ("generation", "batch_size", 64),
        ):
            changed = copy.deepcopy(base)
            changed[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                fs.validate_config(changed)

    def test_e2e_bounded_optimizer_extension_accepts_requested_through_double(self):
        self.assertTrue(fs.bounded_e2e_optimizer_steps(20, 20))
        self.assertTrue(fs.bounded_e2e_optimizer_steps(20, 40))
        self.assertFalse(fs.bounded_e2e_optimizer_steps(20, 19))
        self.assertFalse(fs.bounded_e2e_optimizer_steps(20, 41))

    def test_false_gates_state_is_reused_after_explicit_authorization(self):
        first = fs.Supervisor(self.config)
        self._complete_gates_without_commands(first)
        identity = first.state["config_sha256"]
        self.assertEqual(first.state["status"], "gates_complete_awaiting_formal_authorization")
        self._close(first)

        authorized = copy.deepcopy(self.config)
        authorized["formal_launch_authorized"] = True
        resumed = fs.Supervisor(authorized)
        self.assertEqual(resumed.state["config_sha256"], identity)
        self.assertEqual(resumed.state["status"], "gates_complete_awaiting_formal_authorization")
        self._close(resumed)

    def test_unauthorized_formal_run_never_enters_a_formal_stage_or_poison_gates(self):
        supervisor = fs.Supervisor(self.config)
        self._complete_gates_without_commands(supervisor)
        before = copy.deepcopy(supervisor.state)
        with mock.patch.object(supervisor, "_run_launch_gates") as launch:
            with self.assertRaises(PermissionError):
                supervisor.run()
        launch.assert_not_called()
        self.assertEqual(supervisor.state["status"], before["status"])
        self.assertNotIn("failed_unix_time", supervisor.state)
        self._close(supervisor)

    def test_real_experiment_config_change_refuses_resume(self):
        first = fs.Supervisor(self.config)
        self._complete_gates_without_commands(first)
        self._close(first)

        changed = copy.deepcopy(self.config)
        changed["formal_launch_authorized"] = True
        changed["training"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "different config"):
            second = fs.Supervisor(changed)
            self._close(second)

    def test_regular_run_exception_durably_fails_active_stage_with_traceback(self):
        authorized = copy.deepcopy(self.config)
        authorized["formal_launch_authorized"] = True
        supervisor = fs.Supervisor(authorized)

        def fail_in_gate():
            supervisor.set_stage("e2e_smoke", "running")
            raise RuntimeError("fault after formal authorization")

        with mock.patch.object(
            supervisor, "_reviewed_launch_gates_are_current", return_value=True
        ), mock.patch.object(fs, "preflight", return_value=self._preflight_report()), mock.patch.object(
            supervisor, "_run_launch_gates", side_effect=fail_in_gate
        ):
            with self.assertRaisesRegex(RuntimeError, "fault after formal authorization"):
                supervisor.run()

        persisted = fs.read_json(Path(authorized["run_root"]) / "supervisor_status.json")
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(persisted["current_stage"], "e2e_smoke")
        self.assertEqual(persisted["stages"]["e2e_smoke"]["status"], "failed")
        self.assertIn("fault after formal authorization", persisted["traceback"])
        self.assertIn("fault after formal authorization", persisted["stages"]["e2e_smoke"]["traceback"])
        self._close(supervisor)

    def test_detached_child_reattach_refreshes_heartbeat_without_duplicate_launch(self):
        supervisor = fs.Supervisor(self.config)
        command = ["python", "worker.py", "--output-dir", str(supervisor.p["generation"])]
        supervisor.state["jobs"]["generation/reattach"] = {
            "status": "running", "pid": 123, "command": command,
            "started_unix_time": time.time() - 10,
        }
        calls = {"pid": 0, "done": False}

        def pid_matches(_pid, _command):
            calls["pid"] += 1
            if calls["pid"] <= 2:
                return True
            calls["done"] = True
            return False

        def completion():
            return calls["done"]

        with mock.patch.object(supervisor, "_pid_matches", side_effect=pid_matches), mock.patch.object(
            supervisor, "_launch"
        ) as launch, mock.patch.object(fs.time, "sleep"):
            supervisor._run_parallel(
                "generation",
                {"generation/reattach": ("cuda:0", command, completion)},
            )
        launch.assert_not_called()
        job = supervisor.state["jobs"]["generation/reattach"]
        self.assertEqual(job["status"], "complete")
        self.assertGreater(job["heartbeat_unix_time"], job["started_unix_time"])
        self.assertIn("supervisor_observed_unix_time", job)
        self._close(supervisor)

    def test_worker_progress_old_resume_sidecar_has_bounded_grace(self):
        supervisor = fs.Supervisor(self.config)
        action = "tap"
        root = supervisor.p["training"] / action
        root.mkdir(parents=True)
        now = time.time()
        source = {"corpus_sha256": "1" * 64}
        cfg = {"device": self.config["action_device"][action]}
        fs.atomic_json(root / "run_manifest.json", {"source": source, "config": cfg})
        progress_path = root / "training_progress.json"
        fs.atomic_json(progress_path, {
            "schema_version": "trajectory_training_progress_v1",
            "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
            "run_instance_id": "old-run", "action": action, "pid": 111,
        })
        started = now - 10
        os.utime(progress_path, (started - 1, started - 1))
        supervisor.state["jobs"]["training/tap"] = {
            "status": "running", "pid": 222, "started_unix_time": started,
        }
        with mock.patch.object(fs.time, "time", return_value=now):
            supervisor._observe_training_progress("training/tap")
        self.assertEqual(
            supervisor.state["jobs"]["training/tap"]["worker_progress"]["status"],
            "awaiting_new_run_publish",
        )
        with mock.patch.object(fs.time, "time", return_value=started + 601):
            with self.assertRaisesRegex(RuntimeError, "did not replace old progress"):
                supervisor._observe_training_progress("training/tap")
        self._close(supervisor)

    def test_worker_progress_rejects_nonfinite_and_backward_updates(self):
        supervisor = fs.Supervisor(self.config)
        action = "tap"
        root = supervisor.p["training"] / action
        root.mkdir(parents=True)
        now = time.time()
        source = {"corpus_sha256": "1" * 64}
        cfg = {"device": self.config["action_device"][action]}
        fs.atomic_json(root / "run_manifest.json", {"source": source, "config": cfg})
        progress_path = root / "training_progress.json"

        def progress(sequence, global_step, last_loss=1.0):
            payload = {
                "schema_version": "trajectory_training_progress_v1",
                "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
                "run_instance_id": "current-run", "action": action, "pid": 222,
                "source": source, "config_sha256": fs.canonical_sha256(cfg),
                "device": self.config["action_device"][action], "phase": "train",
                "epoch_index": 0, "next_batch_in_epoch": 2,
                "global_step": global_step, "examples_seen_in_epoch": 64,
                "last_successful_step": global_step,
                "validation_batch_index": None, "validation_batches_total": None,
                "heartbeat_sequence": sequence, "last_loss": last_loss,
                "grad_norm": 0.5, "started_unix_time": now - 9,
                "updated_unix_time": now,
                "last_successful_progress_unix_time": now,
            }
            fs.atomic_json(progress_path, payload)
            os.utime(progress_path, (now, now))

        supervisor.state["jobs"]["training/tap"] = {
            "status": "running", "pid": 222, "started_unix_time": now - 10,
        }
        progress(1, 10)
        with mock.patch.object(fs.time, "time", return_value=now):
            supervisor._observe_training_progress("training/tap")
        progress(2, 11, float("nan"))
        with mock.patch.object(fs.time, "time", return_value=now):
            with self.assertRaisesRegex(RuntimeError, "loss/gradient"):
                supervisor._observe_training_progress("training/tap")
        progress(2, 9)
        with mock.patch.object(fs.time, "time", return_value=now):
            with self.assertRaisesRegex(RuntimeError, "moved backwards"):
                supervisor._observe_training_progress("training/tap")
        self._close(supervisor)

    def test_wait_one_reattach_refreshes_heartbeat_without_relaunch(self):
        supervisor = fs.Supervisor(self.config)
        command = ["python", "worker.py", "--output-dir", str(supervisor.p["bundle"])]
        supervisor.state["jobs"]["detector_bundle"] = {
            "status": "running", "pid": 321, "command": command,
            "started_unix_time": time.time() - 10,
        }
        calls = {"pid": 0, "done": False}

        def pid_matches(_pid, _command):
            calls["pid"] += 1
            if calls["pid"] <= 2:
                return True
            calls["done"] = True
            return False

        with mock.patch.object(
            supervisor, "_pid_matches", side_effect=pid_matches
        ), mock.patch.object(supervisor, "_launch") as launch, mock.patch.object(
            fs.time, "sleep"
        ):
            supervisor._wait_one(
                "detector_bundle", command, lambda: calls["done"]
            )
        launch.assert_not_called()
        job = supervisor.state["jobs"]["detector_bundle"]
        self.assertEqual(job["status"], "complete")
        self.assertGreater(job["heartbeat_unix_time"], job["started_unix_time"])
        self.assertIn("supervisor_observed_unix_time", job)
        self._close(supervisor)

    def test_parallel_failure_terminates_reattached_training_child(self):
        supervisor = fs.Supervisor(self.config)
        command = [
            "python", "worker.py", "--output-dir",
            str(supervisor.p["training"] / "tap"),
        ]
        supervisor.state["jobs"]["training/tap"] = {
            "status": "running", "pid": 654, "command": command,
            "started_unix_time": time.time() - 10,
        }
        with mock.patch.object(
            supervisor, "_pid_matches", return_value=True
        ), mock.patch.object(
            supervisor, "_observe_training_progress",
            side_effect=RuntimeError("invalid worker progress"),
        ), mock.patch.object(supervisor, "_launch") as launch, mock.patch.object(
            fs.time, "sleep"
        ), mock.patch.object(fs.os, "killpg") as killpg:
            with self.assertRaisesRegex(RuntimeError, "invalid worker progress"):
                supervisor._run_parallel(
                    "training",
                    {"training/tap": ("cuda:0", command, lambda: False)},
                )
        launch.assert_not_called()
        killpg.assert_called_once_with(654, fs.signal.SIGTERM)
        self.assertIn(
            "termination_requested_unix_time",
            supervisor.state["jobs"]["training/tap"],
        )
        self._close(supervisor)

    def test_authorization_true_without_reviewed_gates_cannot_start_formal_work(self):
        authorized = copy.deepcopy(self.config)
        authorized["formal_launch_authorized"] = True
        supervisor = fs.Supervisor(authorized)
        before = copy.deepcopy(supervisor.state)
        with mock.patch.object(fs, "preflight") as preflight_call, mock.patch.object(
            supervisor, "_run_launch_gates"
        ) as launch:
            with self.assertRaisesRegex(PermissionError, "completed --gates-only"):
                supervisor.run()
        preflight_call.assert_not_called()
        launch.assert_not_called()
        self.assertEqual(supervisor.state["status"], before["status"])
        self._close(supervisor)

    def test_main_refuses_authorized_launch_without_gates_before_constructing_state(self):
        authorized = copy.deepcopy(self.config)
        authorized["formal_launch_authorized"] = True
        config_path = Path(self.temporary.name) / "authorized_without_gates.json"
        fs.atomic_json(config_path, authorized)
        with mock.patch.object(fs, "validate_config", return_value=None), self.assertRaisesRegex(
            PermissionError, "durable completed --gates-only"
        ):
            fs.main(["--config", str(config_path)])
        self.assertFalse(Path(authorized["run_root"]).joinpath("supervisor_status.json").exists())

    def test_gates_only_failure_uses_one_shared_failure_transaction(self):
        supervisor = fs.Supervisor(self.config)

        def fail_in_gate():
            supervisor.set_stage("condition_preflight", "running")
            raise ValueError("condition gate fault")

        gate_report = dict(self._preflight_report())
        gate_report["runtime_inputs_required"] = False
        with mock.patch.object(fs, "preflight", return_value=gate_report), mock.patch.object(
            supervisor, "_run_launch_gates", side_effect=fail_in_gate
        ), mock.patch.object(supervisor, "_record_failure", wraps=supervisor._record_failure) as record:
            with self.assertRaisesRegex(ValueError, "condition gate fault"):
                supervisor.run_gates_only()
        record.assert_called_once()
        persisted = fs.read_json(Path(self.config["run_root"]) / "supervisor_status.json")
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(persisted["stages"]["condition_preflight"]["status"], "failed")
        self._close(supervisor)

    def test_gates_only_rejects_failed_overall_preflight_before_launch(self):
        supervisor = fs.Supervisor(self.config)
        report = self._preflight_report()
        report.update({
            "passed": False,
            "ready_for_gates": False,
            "runtime_inputs_required": False,
            "blockers": ["disk gate"],
        })
        with mock.patch.object(fs, "preflight", return_value=report), mock.patch.object(
            supervisor, "_run_launch_gates"
        ) as launch:
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                supervisor.run_gates_only()
        launch.assert_not_called()
        self.assertEqual(supervisor.state["status"], "failed")
        self.assertEqual(supervisor.state["stages"]["preflight"]["status"], "failed")
        self._close(supervisor)

    def test_gates_only_rejects_authorized_config_before_state_transition(self):
        authorized = copy.deepcopy(self.config)
        authorized["formal_launch_authorized"] = True
        supervisor = fs.Supervisor(authorized)
        before = copy.deepcopy(supervisor.state)
        with self.assertRaisesRegex(PermissionError, "requires formal_launch_authorized=false"):
            supervisor.run_gates_only()
        self.assertEqual(supervisor.state["status"], before["status"])
        self._close(supervisor)

    def test_final_writer_rejects_any_failed_closure_without_publishing_pass(self):
        supervisor = fs.Supervisor(self.config)
        with mock.patch.object(
            supervisor,
            "_final_completion_failures",
            return_value=["generation/shard_1"],
        ):
            with self.assertRaisesRegex(RuntimeError, "generation/shard_1"):
                supervisor._write_final()
        self.assertFalse(supervisor.p["final_audit"].exists())
        self.assertFalse(supervisor.p["final_report"].exists())
        self._close(supervisor)

    def test_final_commit_marker_binds_report_bytes_and_current_snapshot(self):
        supervisor = fs.Supervisor(self.config)
        evidence = {"schema_version": "trajectory_launch_gate_evidence_v1"}
        supervisor.state["launch_gate_evidence"] = evidence
        supervisor.p["final_report"].parent.mkdir(parents=True, exist_ok=True)
        supervisor.p["final_report"].write_text("formal report\n", encoding="utf-8")
        generation = {
            "n_fake": 100000,
            "n_units": 500,
            "selector_used": False,
            "condition_set_sha256": "a" * 64,
            "runtime_determinism": dict(fs.EXPECTED_RUNTIME_DETERMINISM),
            "runtime_determinism_sha256": fs.STRICT_RUNTIME_DETERMINISM_SHA256,
        }
        benchmark = {"status": "passed"}
        fs.atomic_json(
            supervisor.p["generation"] / "formal_generation_audit.json",
            generation,
        )
        fs.atomic_json(
            supervisor.p["benchmark_merged"] / "benchmark_audit.json",
            benchmark,
        )
        fs.atomic_json(supervisor.p["final_audit"], {
            "schema_version": "trajectory_formal_end_to_end_audit_v2",
            "passed": True,
            "commit_marker": True,
            "config_sha256": fs.experiment_config_sha256(self.config),
            "config_identity_excludes": sorted(fs.OPERATIONAL_CONFIG_KEYS),
            "source_code": supervisor.manifest["source_code"],
            "split_sha256": self.config["split_sha256"],
            "launch_gate_evidence": evidence,
            "training_complete_actions": list(fs.ACTIONS),
            "generation": generation,
            "benchmark": benchmark,
            "invariants": supervisor.manifest["formal_invariants"],
            "artifact_snapshot": {},
            "report": {
                "path": str(supervisor.p["final_report"].resolve()),
                "sha256": fs.sha256_file(supervisor.p["final_report"]),
                "size_bytes": supervisor.p["final_report"].stat().st_size,
            },
        })
        with mock.patch.object(
            supervisor, "_final_completion_failures", return_value=[]
        ), mock.patch.object(
            supervisor, "_final_artifact_snapshot", return_value={}
        ):
            self.assertTrue(supervisor._final_complete())
            receipt = fs.read_json(supervisor.p["final_audit"])
            receipt["generation"]["n_fake"] = 99999
            fs.atomic_json(supervisor.p["final_audit"], receipt)
            self.assertFalse(supervisor._final_complete())
            receipt["generation"] = generation
            fs.atomic_json(supervisor.p["final_audit"], receipt)
            supervisor.p["final_report"].write_text("tampered\n", encoding="utf-8")
            self.assertFalse(supervisor._final_complete())
        self._close(supervisor)

    def test_clean_gpu_wait_retries_query_error_and_busy_snapshot_without_launch(self):
        supervisor = fs.Supervisor(self.config)
        supervisor.poll_seconds = 0
        busy = {
            "cuda:0": {
                "total_bytes": 48 << 30,
                "free_bytes": 20 << 30,
                "used_bytes": 28 << 30,
                "utilization_percent": 100,
                "temperature_c": 85,
            }
        }
        clean = {
            "cuda:0": {
                "total_bytes": 48 << 30,
                "free_bytes": 48 << 30,
                "used_bytes": 0,
                "utilization_percent": 0,
                "temperature_c": 40,
            }
        }
        with mock.patch.object(
            supervisor,
            "_gpu_memory_snapshot",
            side_effect=[RuntimeError("transient nvidia-smi rc9"), busy, clean],
        ) as snapshots, mock.patch.object(fs.time, "sleep") as sleep, mock.patch.object(
            supervisor, "_launch"
        ) as launch:
            observed = supervisor._wait_for_clean_gpus(
                ("cuda:0",), "unit clean gate", "throughput_probe"
            )
        self.assertEqual(observed, clean)
        self.assertEqual(snapshots.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        launch.assert_not_called()
        self.assertEqual(supervisor.state["stages"]["throughput_probe"]["status"], "running")
        self._close(supervisor)

    def test_throughput_selection_recomputes_argmax_and_rejects_tampered_choice(self):
        supervisor = fs.Supervisor(self.config)
        metric = "projected_full_epoch_examples_per_second"
        registry_hashes = {action: "a" * 64 for action in fs.ACTIONS}
        fs.atomic_json(supervisor.p["condition_preflight"], {
            "training_reference_registry_sha256_by_action": registry_hashes,
        })
        candidates, selected_results = {}, {}
        selected = {action: 128 for action in fs.ACTIONS}
        metric_by_batch = {32: 1.0, 64: 2.0, 128: 4.0, 256: 3.0}
        memory_limit = (100 << 30) - (6 << 30)
        for action in fs.ACTIONS:
            rows = []
            for batch_size in (32, 64, 128, 256):
                path = supervisor.p["probe"] / action / (
                    "candidate_bs%03d.json" % batch_size
                )
                payload = {
                    metric: metric_by_batch[batch_size],
                    "projected_full_epoch_optimizer_seconds": 1000.0 / metric_by_batch[batch_size],
                    "worst_case_elapsed_seconds": float(batch_size),
                    "cuda_peak_memory_reserved_bytes": batch_size << 20,
                }
                fs.atomic_json(path, payload)
                rows.append({
                    "batch_size": batch_size,
                    "path": str(path),
                    "result_sha256": fs.sha256_file(path),
                    "expected_resource_failure": False,
                    "peak_vram_reserved_bytes": batch_size << 20,
                    "memory_limit_after_margin_bytes": memory_limit,
                    "memory_safety_passed": True,
                    metric: metric_by_batch[batch_size],
                    "projected_full_epoch_optimizer_seconds": 1000.0 / metric_by_batch[batch_size],
                    "worst_case_elapsed_seconds": float(batch_size),
                })
            candidates[action] = rows
            selected_path = supervisor.p["probe"] / action / "selected_bs128_100steps.json"
            selected_payload = {
                metric: 4.0,
                "projected_full_epoch_optimizer_seconds": 250.0,
                "worst_case_elapsed_seconds": 128.0,
                "cuda_peak_memory_allocated_bytes": 100,
                "cuda_peak_memory_reserved_bytes": 200,
            }
            fs.atomic_json(selected_path, selected_payload)
            selected_results[action] = {
                "path": str(selected_path),
                "sha256": fs.sha256_file(selected_path),
                "batch_size": 128,
                metric: 4.0,
                "projected_full_epoch_optimizer_seconds": 250.0,
                "worst_case_elapsed_seconds": 128.0,
                "peak_vram_allocated_bytes": 100,
                "peak_vram_reserved_bytes": 200,
                "measured_optimizer_steps": 100,
            }
        selection = {
            "schema_version": "trajectory_formal_throughput_selection_v2",
            "passed": True,
            "selection_uses_validation_or_test": False,
            "changes_model_data_or_truncation": False,
            "selection_metric": metric,
            "tie_break": "larger_stable_batch",
            "candidate_batch_sizes": [32, 64, 128, 256],
            "candidate_measured_steps": 8,
            "selected_measured_steps": 100,
            "candidate_wall_time_limit_seconds": float(
                self.config["throughput_probe"]["candidate_wall_time_limit_seconds"]
            ),
            "gpu_safety_margin_bytes": 6 << 30,
            "gpu_memory_before_probe": {
                "cuda:0": {"free_bytes": 100 << 30}
            },
            "split_sha256": self.config["split_sha256"],
            "corpus_sha256_by_action": {
                action: fs.sha256_file(
                    supervisor.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
                )
                for action in fs.ACTIONS
            },
            "reference_registry_sha256_by_action": registry_hashes,
            "selected_batch_size_by_action": selected,
            "selected_results": selected_results,
            "candidates": candidates,
        }
        fs.atomic_json(supervisor.p["probe_selection"], selection)
        with mock.patch.object(
            supervisor, "_probe_candidate_complete", return_value=True
        ), mock.patch.object(
            supervisor, "_selected_probe_complete", return_value=True
        ):
            self.assertEqual(supervisor._load_probe_selection(), selected)
            selection["selected_batch_size_by_action"]["tap"] = 32
            fs.atomic_json(supervisor.p["probe_selection"], selection)
            self.assertIsNone(supervisor._load_probe_selection())
        self._close(supervisor)

    def test_probe_completion_requires_exact_typed_runtime_determinism(self):
        supervisor = fs.Supervisor(self.config)
        action = "tap"
        batch_size = 32
        registry_sha = "a" * 64
        fs.atomic_json(supervisor.p["condition_preflight"], {
            "training_reference_registry_sha256_by_action": {
                action: registry_sha,
            },
        })
        def projection_contract(steps, warmup):
            measurements = [{
                "label": "artificial_global_worst_case",
                "elapsed_seconds": 1.0,
                "batch_size": 6,
                "target_padded_t": 2,
                "reference_padded_t": 2,
                "target_keycode_padded_k": 1,
                "reference_keycode_padded_k": 1,
                "padded_work": 20,
            }]
            measurements.extend({
                "label": "profile_%03d" % index,
                "elapsed_seconds": 1.0,
                "batch_size": 6,
                "target_padded_t": 2,
                "reference_padded_t": 2,
                "target_keycode_padded_k": 1,
                "reference_keycode_padded_k": 1,
                "padded_work": 10,
            } for index in range(steps))
            return {
                "dataset_target_count": 6,
                "profile_epoch_batch_counts": [1] * 5,
                "profile_target_occurrences": 30,
                "profile_each_epoch_covers_dataset_once": True,
                "epoch_length_profile_sha256": "b" * 64,
                "worst_case_padded_t": 2,
                "worst_case_reference_padded_t": 2,
                "worst_case_keycode_padded_k": 1,
                "worst_case_reference_keycode_padded_k": 1,
                "worst_case_padded_work": 20,
                "worst_case_elapsed_seconds": 1.0,
                "optimizer_state_initialization_steps": warmup,
                "optimizer_state_initialization_excluded_from_projection": True,
                "shape_specific_warmup_optimizer_steps": 1,
                "shape_specific_warmup_excluded_from_projection": True,
                "total_unmeasured_optimizer_steps": warmup + 1,
                "projection_measurement_count": steps + 1,
                "projection_measurements": measurements,
                "projected_full_epoch_optimizer_seconds": 2.0,
                "projected_full_epoch_examples_per_second": 3.0,
                "epoch_projection": {
                    "method": "monotone_piecewise_linear_exact_t_tr_k_kr_padding_v2",
                    "fit_padded_work": [10, 20],
                    "fit_elapsed_seconds": [1.0, 2.0],
                    "projection_has_extrapolation": False,
                    "profile_epochs": [0, 1, 2, 3, 4],
                    "epoch_optimizer_seconds": [2.0] * 5,
                    "mean_epoch_optimizer_seconds": 2.0,
                },
            }
        def benchmark_config(steps, warmup):
            training = self.config["training"]
            value = {
                "action": action,
                "device": self.config["throughput_probe"]["device"],
                "batch_size": batch_size,
                "measured_steps": steps,
                "warmup_steps": warmup,
                "num_workers": training["num_workers"],
                "seed": training["seed"],
                "learning_rate": training["learning_rate"],
                "weight_decay": training["weight_decay"],
                "grad_clip_norm": training["grad_clip_norm"],
                "ema_decay": training["ema_decay"],
                "diffusion_steps": training["diffusion_steps"],
                "base_channels": training["base_channels"],
                "cond_dim": training["cond_dim"],
                "time_dim": training["time_dim"],
                "n_blocks": training["n_blocks"],
                "dropout": training["dropout"],
                "keycode_vocab": training["keycode_vocab"],
                "reference_cache_size": training["reference_cache_size"],
                "amp": True,
                "optimizer": "AdamW",
                "profile_epochs": [0, 1, 2, 3, 4],
            }
            return {
                "benchmark_config": value,
                "benchmark_config_sha256": fs.canonical_sha256(value),
            }
        common = {
            "schema_version": "trajectory_training_throughput_v2",
            "passed": True,
            "uses_exact_formal_train_loader_and_model": True,
            "reads_validation_or_test_targets": False,
            "creates_or_updates_formal_checkpoint": False,
            "action": action,
            "device": self.config["throughput_probe"]["device"],
            "batch_size": batch_size,
            "diffusion_steps": 1000,
            "keycode_vocab": 16384,
            "longest_uncapped_batch_exercised": True,
            "worst_case_safety_optimizer_steps": 1,
            "exact_canonical_target_and_reference_lengths": True,
            "exact_target_and_reference_keycode_lengths": True,
            "projection_includes_worst_case_measurement": True,
            "projection_has_extrapolation": False,
            "profile_epoch_count": 5,
            "loss": {"all_finite": True},
            "split_sha256": self.config["split_sha256"],
            "corpus_sha256": fs.sha256_file(
                supervisor.p["corpus"] / "hmog_trajectory_tap.npz"
            ),
            "reference_registry_sha256": registry_sha,
            "runtime_determinism": dict(fs.EXPECTED_RUNTIME_DETERMINISM),
        }
        candidate_path = (
            supervisor.p["probe"] / action / "candidate_bs032.json"
        )
        candidate = dict(common)
        candidate["measured_optimizer_steps"] = int(
            self.config["throughput_probe"]["candidate_measured_steps"]
        )
        candidate["projection_measurement_count"] = int(
            self.config["throughput_probe"]["candidate_measured_steps"]
        ) + 1
        candidate_steps = int(self.config["throughput_probe"]["candidate_measured_steps"])
        candidate_warmup = int(self.config["throughput_probe"]["candidate_warmup_steps"])
        candidate.update(projection_contract(candidate_steps, candidate_warmup))
        candidate.update(benchmark_config(candidate_steps, candidate_warmup))
        fs.atomic_json(candidate_path, candidate)
        self.assertTrue(supervisor._probe_candidate_complete(action, batch_size))

        selected_path = (
            supervisor.p["probe"] / action / "selected_bs032_100steps.json"
        )
        selected = dict(common)
        selected["measured_optimizer_steps"] = 100
        selected["projection_measurement_count"] = 101
        selected_warmup = int(self.config["throughput_probe"]["selected_warmup_steps"])
        selected.update(projection_contract(100, selected_warmup))
        selected.update(benchmark_config(100, selected_warmup))
        fs.atomic_json(selected_path, selected)
        self.assertTrue(supervisor._selected_probe_complete(action, batch_size))

        invalid_contracts = {
            "missing": None,
            "wrong_cublas": {
                **fs.EXPECTED_RUNTIME_DETERMINISM,
                "cublas_workspace_config": ":16:8",
            },
            "wrong_warn_only": {
                **fs.EXPECTED_RUNTIME_DETERMINISM,
                "deterministic_algorithms_warn_only": True,
            },
            "deterministic_algorithms_disabled": {
                **fs.EXPECTED_RUNTIME_DETERMINISM,
                "deterministic_algorithms_enabled": False,
            },
            "wrong_cudnn_benchmark": {
                **fs.EXPECTED_RUNTIME_DETERMINISM,
                "cudnn_benchmark": True,
            },
            "cudnn_nondeterministic": {
                **fs.EXPECTED_RUNTIME_DETERMINISM,
                "cudnn_deterministic": False,
            },
            "extra_field": {
                **fs.EXPECTED_RUNTIME_DETERMINISM,
                "unreviewed_runtime_flag": True,
            },
            "wrong_boolean_type": {
                **fs.EXPECTED_RUNTIME_DETERMINISM,
                "deterministic_algorithms_enabled": 1,
            },
        }
        for label, contract in invalid_contracts.items():
            with self.subTest(result="candidate", invalid=label):
                candidate["runtime_determinism"] = contract
                fs.atomic_json(candidate_path, candidate)
                self.assertFalse(
                    supervisor._probe_candidate_complete(action, batch_size)
                )
            with self.subTest(result="selected", invalid=label):
                selected["runtime_determinism"] = contract
                fs.atomic_json(selected_path, selected)
                self.assertFalse(
                    supervisor._selected_probe_complete(action, batch_size)
                )
        self._close(supervisor)

    def test_probe_runtime_budget_failure_is_command_and_limit_bound(self):
        supervisor = fs.Supervisor(self.config)
        action, batch_size = "keystroke", 64
        command = list(
            supervisor.manifest["commands"]["throughput_probe_candidates"][action][str(batch_size)]
        )
        inner = command[command.index("--") + 1 :]
        command_sha = hashlib.sha256(
            json.dumps(inner, sort_keys=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        path = supervisor.p["probe"] / action / "candidate_bs064.json"
        payload = {
            "schema_version": "trajectory_training_throughput_candidate_failure_v2",
            "passed": False,
            "expected_resource_failure": True,
            "failure_kind": "runtime_budget_exceeded",
            "timeout_seconds": float(
                self.config["throughput_probe"]["candidate_wall_time_limit_seconds"]
            ),
            "elapsed_seconds": 120.1,
            "process_group_terminated": True,
            "termination_signal": "SIGTERM",
            "command_sha256": command_sha,
        }
        fs.atomic_json(path, payload)
        self.assertTrue(supervisor._probe_candidate_complete(action, batch_size))
        payload["timeout_seconds"] += 1.0
        fs.atomic_json(path, payload)
        self.assertFalse(supervisor._probe_candidate_complete(action, batch_size))
        self._close(supervisor)

    def test_training_completion_binds_config_source_best_sha_ema_and_last_epoch(self):
        supervisor = fs.Supervisor(self.config)
        action = "tap"
        root = supervisor.p["training"] / action
        root.mkdir(parents=True)
        supervisor.p["probe_selection"].parent.mkdir(parents=True, exist_ok=True)
        fs.atomic_json(supervisor.p["probe_selection"], {
            "selected_batch_size_by_action": {name: 32 for name in fs.ACTIONS}
        })
        corpus = supervisor.p["corpus"] / "hmog_trajectory_tap.npz"
        corpus_sha = fs.sha256_file(corpus)
        extraction_manifest = supervisor.p["corpus"] / "manifest.json"
        from generation.protocol import ReferenceRegistry as CanonicalRegistry
        split_payload = fs.read_json(supervisor.p["split"])
        reference_entries = {
            (action, int(user_id), split_name): tuple(
                int(user_id) * 10 + offset for offset in range(5)
            )
            for split_name in ("train", "val", "test")
            for user_id in split_payload[split_name + "_users"]
        }
        canonical_registry = CanonicalRegistry.build(
            reference_entries, self.config["split_sha256"]
        )
        registry_sha = canonical_registry.registry_sha256
        source = {
            "corpus_npz": str(corpus.resolve()),
            "corpus_sha256": corpus_sha,
            "schema_version": 2,
            "action": action,
            "split_json": str(supervisor.p["split"]),
            "split_sha256": self.config["split_sha256"],
            "split_seed": 42,
            "extraction_manifest": str(extraction_manifest.resolve()),
            "extraction_manifest_sha256": fs.sha256_file(extraction_manifest),
            "reference_registry_sha256": registry_sha,
            "reference_registry_protocol": "fixed_five_reference_registry_v1",
        }
        training = self.config["training"]
        cfg = {
            "action": action,
            "corpus_npz": str(corpus.resolve()),
            "split_json": str(supervisor.p["split"]),
            "output_dir": str(root.resolve()),
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "grad_clip_norm": float(training["grad_clip_norm"]),
            "ema_decay": float(training["ema_decay"]),
            "diffusion_steps": 1000,
            "base_channels": int(training["base_channels"]),
            "cond_dim": int(training["cond_dim"]),
            "time_dim": int(training["time_dim"]),
            "n_blocks": int(training["n_blocks"]),
            "dropout": float(training["dropout"]),
            "keycode_vocab": 16384,
            "seed": 42,
            "num_workers": int(training["num_workers"]),
            "amp": True,
            "checkpoint_every_steps": int(training["checkpoint_every_steps"]),
            "reference_cache_size": int(training["reference_cache_size"]),
            "device": self.config["action_device"][action],
            "amp_overflow_max_retries": int(training["amp_overflow_max_retries"]),
            "allow_non_gaussian_terminal_for_test": False,
        }
        best_path = root / "best_epoch_0020_step_000000001_valloss_1.00000000.pt"
        last_path = root / "last.pt"

        model_config = {
            "action": action, "diffusion_steps": 1000,
            "base_channels": cfg["base_channels"], "cond_dim": cfg["cond_dim"],
            "time_dim": cfg["time_dim"], "n_blocks": cfg["n_blocks"],
            "dropout": cfg["dropout"], "keycode_vocab": 16384,
        }
        diffusion_schedule = {
            "diffusion_steps": 1000, "terminal_gaussian_gate_passed": True,
        }

        def checkpoint(epoch):
            return {
                "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
                "runtime_determinism": dict(fs.EXPECTED_RUNTIME_DETERMINISM),
                "checkpoint_role": "training_state_with_raw_model_and_ema",
                "inference_weights_for_validation_selected_best": "ema.shadow",
                "config": cfg,
                "model_config": model_config,
                "diffusion_schedule": diffusion_schedule,
                "numeric_recovery_policy": fs.numeric_recovery_policy(self.config),
                "source": source,
                "model": {"weight": torch.tensor([1.0])},
                "ema": {"decay": cfg["ema_decay"], "shadow": {"weight": torch.tensor([1.0])}},
                "optimizer": {"state": {1: {}}, "param_groups": [{"params": [1]}]},
                "amp_scaler": {"scale": 65536.0},
                "rng_state": {
                    "python": (), "numpy": (), "torch_cpu": torch.tensor([1]),
                    "torch_cuda": [],
                },
                "progress": {
                    "epoch_index": epoch,
                    "next_batch_in_epoch": 0,
                    "examples_seen_in_epoch": 0,
                    "global_step": epoch * 10,
                    "best_val_loss": 1.0,
                    "last_step_loss": 0.01,
                    "last_grad_norm": 0.5,
                    "amp_overflow_retries_total": 0,
                    "epoch_amp_overflow_events": [],
                    "last_validation": {
                        "completed_epoch": epoch, "val_loss": 1.0,
                    },
                },
            }

        torch.save(checkpoint(20), best_path)
        torch.save(checkpoint(100), last_path)
        best_entry = {
            "path": str(best_path.resolve()),
            "filename": best_path.name,
            "completed_epoch": 20,
            "global_step": 200,
            "val_loss": 1.0,
            "source_sha256": corpus_sha,
            "split_sha256": self.config["split_sha256"],
            "reference_registry_sha256": registry_sha,
            "checkpoint_sha256": fs.sha256_file(best_path),
            "checkpoint_role": "validation_selected_best",
            "inference_weights": "ema.shadow",
            "numeric_recovery_policy": fs.numeric_recovery_policy(self.config),
        }
        fs.atomic_json(root / "best_manifest.json", {
            "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
            "numeric_recovery_policy": fs.numeric_recovery_policy(self.config),
            "selection_split": "val",
            "selection_metric": "full_val_masked_epsilon_mse_ema",
            "lower_is_better": True,
            "test_used_for_selection": False,
            "checkpoint_role": "validation_selected_best",
            "inference_weights": "ema.shadow",
            "best": best_entry,
            "history": [best_entry],
            "source": source,
        })
        fs.atomic_json(root / "reference_registry.json", {
            **CanonicalRegistry._payload(
                canonical_registry.entries, self.config["split_sha256"]
            ),
            "registry_sha256": registry_sha,
            "action": action,
            "seed": 42,
            "corpus_npz": str(corpus.resolve()),
            "corpus_sha256": corpus_sha,
            "split_sha256": self.config["split_sha256"],
            "references_per_group": 5,
        })
        fs.atomic_json(root / "run_manifest.json", {
            "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
            "status": "complete",
            "action": action,
            "config": cfg,
            "source": source,
            "counts": {"train": 70, "val": 10, "test_reserved": 20},
            "validation_completed_epochs": [20, 40, 60, 80, 100],
            "validation_fractions": [0.2, 0.4, 0.6, 0.8, 1.0],
            "full_corpus_no_sample_cap": True,
            "drop_last": False,
            "truncation": False,
            "amp_effective": True,
            "numeric_recovery_policy": fs.numeric_recovery_policy(self.config),
            "amp_overflow_retries_total": 0,
            "runtime_determinism": dict(fs.EXPECTED_RUNTIME_DETERMINISM),
            "global_step": 1000,
            "best_val_loss": 1.0,
            "best_checkpoint": str(best_path.resolve()),
            "last_checkpoint": str(last_path.resolve()),
        })
        metrics = []
        for epoch in range(1, 101):
            metrics.append({
                "type": "train_epoch", "completed_epoch": epoch,
                "global_step": epoch * 10, "loss": 1.0 / (epoch + 1),
                "batches_total_in_epoch": 3,
                "examples_total_in_epoch": 70,
                "valid_feature_count_total": 700.0,
                "full_train_split_consumed": True,
                "amp_overflow_retries": 0,
                "amp_overflow_events": [],
            })
            if epoch % 20 == 0:
                metrics.append({
                    "type": "validation", "completed_epoch": epoch,
                    "global_step": epoch * 10, "fraction": epoch / 100.0,
                    "val_loss": 1.0, "n_examples": 10, "n_batches": 2,
                    "valid_feature_count": 100.0,
                    "full_validation_split": True, "ema_weights": True,
                })
        (root / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8"
        )
        fs.atomic_json(root / "last_state.json", {
            "schema_version": "trajectory_last_state_v1",
            "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
            "run_instance_id": "fixture-run",
            "action": action,
            "checkpoint_path": str(last_path.resolve()),
            "checkpoint_sha256": fs.sha256_file(last_path),
            "checkpoint_size_bytes": last_path.stat().st_size,
            "progress": {
                "epoch_index": 100,
                "next_batch_in_epoch": 0,
                "examples_seen_in_epoch": 0,
                "global_step": 1000,
            },
            "source": source,
            "config_sha256": fs.canonical_sha256(cfg),
        })
        fs.atomic_json(root / "training_progress.json", {
            "schema_version": "trajectory_training_progress_v1",
            "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
            "run_instance_id": "fixture-run",
            "action": action,
            "source": source,
            "config_sha256": fs.canonical_sha256(cfg),
            "phase": "complete",
            "epoch_index": 100,
            "next_batch_in_epoch": 0,
            "examples_seen_in_epoch": 0,
            "global_step": 1000,
            "last_successful_step": 1000,
            "heartbeat_sequence": 100,
            "last_loss": 0.01,
            "grad_norm": 0.5,
            "amp_overflow_retries_total": 0,
            "device": cfg["device"],
            "started_unix_time": 100.0,
            "updated_unix_time": 200.0,
            "last_successful_progress_unix_time": 200.0,
        })
        self.assertTrue(supervisor._training_complete(action))

        run_manifest_path = root / "run_manifest.json"
        original_run_manifest = fs.read_json(run_manifest_path)
        invalid_run_manifest = copy.deepcopy(original_run_manifest)
        invalid_run_manifest["runtime_determinism"]["cudnn_deterministic"] = False
        fs.atomic_json(run_manifest_path, invalid_run_manifest)
        self.assertFalse(supervisor._training_complete(action))
        invalid_run_manifest = copy.deepcopy(original_run_manifest)
        invalid_run_manifest["runtime_determinism"]["extra"] = True
        fs.atomic_json(run_manifest_path, invalid_run_manifest)
        self.assertFalse(supervisor._training_complete(action))
        fs.atomic_json(run_manifest_path, original_run_manifest)
        self.assertTrue(supervisor._training_complete(action))

        metrics_path = root / "metrics.jsonl"
        original_metrics = metrics_path.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in original_metrics.splitlines()]
        rows[0]["loss"] = float("nan")
        metrics_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        self.assertFalse(supervisor._training_complete(action))
        metrics_path.write_text(original_metrics, encoding="utf-8")

        original_best_bytes = best_path.read_bytes()
        poisoned_runtime = torch.load(str(best_path), map_location="cpu")
        poisoned_runtime["runtime_determinism"]["deterministic_algorithms_enabled"] = 1
        torch.save(poisoned_runtime, best_path)
        manifest = fs.read_json(root / "best_manifest.json")
        manifest["best"]["checkpoint_sha256"] = fs.sha256_file(best_path)
        manifest["history"][-1]["checkpoint_sha256"] = fs.sha256_file(best_path)
        fs.atomic_json(root / "best_manifest.json", manifest)
        self.assertFalse(supervisor._training_complete(action))
        best_path.write_bytes(original_best_bytes)
        manifest["best"]["checkpoint_sha256"] = fs.sha256_file(best_path)
        manifest["history"][-1]["checkpoint_sha256"] = fs.sha256_file(best_path)
        fs.atomic_json(root / "best_manifest.json", manifest)
        self.assertTrue(supervisor._training_complete(action))

        poisoned = torch.load(str(best_path), map_location="cpu")
        poisoned["ema"]["shadow"]["weight"] = torch.tensor([float("nan")])
        torch.save(poisoned, best_path)
        manifest = fs.read_json(root / "best_manifest.json")
        manifest["best"]["checkpoint_sha256"] = fs.sha256_file(best_path)
        manifest["history"][-1]["checkpoint_sha256"] = fs.sha256_file(best_path)
        fs.atomic_json(root / "best_manifest.json", manifest)
        self.assertFalse(supervisor._training_complete(action))
        best_path.write_bytes(original_best_bytes)
        manifest["best"]["checkpoint_sha256"] = fs.sha256_file(best_path)
        manifest["history"][-1]["checkpoint_sha256"] = fs.sha256_file(best_path)
        fs.atomic_json(root / "best_manifest.json", manifest)
        self.assertTrue(supervisor._training_complete(action))

        manifest = fs.read_json(root / "best_manifest.json")
        manifest["best"]["checkpoint_sha256"] = "0" * 64
        manifest["history"][-1]["checkpoint_sha256"] = "0" * 64
        fs.atomic_json(root / "best_manifest.json", manifest)
        self.assertFalse(supervisor._training_complete(action))
        self._close(supervisor)

    def test_training_bootstrap_binds_source_target_and_internal_receipt(self):
        from scripts.migrate_v15_training_state import expected_training_config

        supervisor = fs.Supervisor(self.config)
        selected = {name: 32 for name in fs.ACTIONS}
        selected["tap"] = 128
        selected["scroll"] = 256
        supervisor.p["probe_selection"].parent.mkdir(parents=True, exist_ok=True)
        fs.atomic_json(supervisor.p["probe_selection"], {
            "selected_batch_size_by_action": selected,
        })
        bootstrap = self.config["training_bootstrap"]
        source_root = Path(bootstrap["source_run_root"]).resolve()
        action_rows = []
        protected = {
            name: "0" * 64
            for name in ("model", "ema", "optimizer", "amp_scaler", "rng_state")
        }
        for action in ("tap", "scroll"):
            target_root = supervisor.p["training"] / action
            target_root.mkdir(parents=True)
            target_last = target_root / "last.pt"
            target_last.write_bytes(("migrated-" + action).encode("ascii"))
            source_last = source_root / "training" / action / "last.pt"
            expected = expected_training_config(
                self.config, action, selected[action], target_root,
            )
            fs.atomic_json(target_root / "run_manifest.json", {
                "config": asdict(expected),
                "status": "migrated_awaiting_resume",
            })
            last_receipt = {
                "source_path": str(source_last),
                "source_sha256": fs.sha256_file(source_last),
                "target_path": str(target_last),
                "target_sha256": fs.sha256_file(target_last),
                "protected_content_sha256": protected,
                "progress": {
                    "epoch_index": 1,
                    "next_batch_in_epoch": 2,
                    "examples_seen_in_epoch": 3,
                    "global_step": 4,
                    "amp_overflow_retries_total": 0,
                },
            }
            action_rows.append({
                "action": action,
                "source_last_checkpoint_sha256": bootstrap["actions"][action][
                    "last_checkpoint_sha256"
                ],
                "target_last_checkpoint_sha256": last_receipt["target_sha256"],
                "expected_config_sha256": fs.canonical_sha256(asdict(expected)),
                "checkpoint_receipts": {"last.pt": last_receipt},
            })
        receipt = {
            "schema_version": "trajectory_v15_to_v16_training_bootstrap_v1",
            "passed": True,
            "source_run_root": str(source_root),
            "source_tree_sha256": bootstrap["source_tree_sha256"],
            "target_run_root": str(supervisor.p["run"]),
            "config_sha256": fs.experiment_config_sha256(self.config),
            "selection_path": str(supervisor.p["probe_selection"]),
            "selection_sha256": fs.sha256_file(supervisor.p["probe_selection"]),
            "actions": action_rows,
        }
        internal = supervisor.p["training"] / ".v15_to_v16_bootstrap_receipt.json"
        fs.atomic_json(internal, receipt)
        fs.atomic_json(supervisor.p["training_bootstrap_receipt"], receipt)
        self.assertTrue(supervisor._training_bootstrap_complete())
        tap_last = supervisor.p["training"] / "tap" / "last.pt"
        tap_bytes = tap_last.read_bytes()
        tap_last.write_bytes(b"tampered")
        self.assertFalse(supervisor._training_bootstrap_complete())
        tap_last.write_bytes(tap_bytes)
        poisoned = copy.deepcopy(receipt)
        poisoned["source_tree_sha256"] = "f" * 64
        fs.atomic_json(supervisor.p["training_bootstrap_receipt"], poisoned)
        self.assertFalse(supervisor._training_bootstrap_complete())
        self._close(supervisor)

    def test_condition_gate_rejects_split_corpus_and_request_protocol_tampering(self):
        from generation.protocol import CONDITION_REQUEST_DIGEST_FIELDS

        supervisor = fs.Supervisor(self.config)
        registry = {action: ("%x" % (index + 1)) * 64 for index, action in enumerate(fs.ACTIONS)}
        prior = {action: ("%x" % (index + 6)) * 64 for index, action in enumerate(fs.ACTIONS)}
        per_action_digest = {action: "a" * 64 for action in fs.ACTIONS}
        per_action = {}
        for action in fs.ACTIONS:
            corpus = supervisor.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
            per_action[action] = {
                "corpus": str(corpus.resolve()),
                "corpus_sha256": fs.sha256_file(corpus),
                "training_reference_registry_sha256": registry[action],
                "generation_registry_compatibility_sha256": registry[action],
                "train_prior_sha256": prior[action],
                "condition_set_sha256": per_action_digest[action],
                "counts": {
                    "users": 100,
                    "requests": 20000,
                    "sampling_batches": 700,
                    "condition_source_code_2": 20000,
                    "unique_fake_ids": 20000,
                    "unique_condition_request_seeds": 20000,
                    "unique_ddim_noise_seeds": 20000,
                    "hard_timeline_projections": 20000 if action == "keystroke" else 0,
                },
            }
        gate = {
            "schema_version": "trajectory_all_condition_requests_preflight_v1",
            "status": "passed",
            "formal_result": False,
            "producer_source": __import__(
                "scripts.preflight_all_condition_requests",
                fromlist=["producer_source_identity"],
            ).producer_source_identity(supervisor.p["project"]),
            "worker_count": 8,
            "parallelization": "fork_per_user_deterministic_parent_aggregation",
            "reference_seed": 42,
            "generation_seed": 20260713,
            "seed_roles_are_distinct": True,
            "split_json": str(supervisor.p["split"]),
            "split_sha256": self.config["split_sha256"],
            "samples_per_user_action": 200,
            "generation_batch_size": 32,
            "sampling_batches_per_user_action": 7,
            "sampling_batches_per_action": 700,
            "sampling_batches_total": 3500,
            "total_requests": 100000,
            "keystroke_hard_timeline_requests": 20000,
            "no_retries": True,
            "no_skips": True,
            "train_prior_only_fixed_train_users": True,
            "all_condition_source_code_eq_2": True,
            "condition_source_code_2_count": 100000,
            "all_fake_ids_globally_unique": True,
            "unique_fake_id_count": 100000,
            "all_condition_request_seeds_globally_unique": True,
            "unique_condition_request_seed_count": 100000,
            "all_ddim_noise_seeds_globally_unique": True,
            "unique_ddim_noise_seed_count": 100000,
            "condition_and_noise_seed_domains_disjoint": True,
            "condition_request_digest_schema": "trajectory_condition_request_canonical_v1",
            "condition_set_digest_schema": "trajectory_condition_request_set_v1",
            "condition_request_digest_fields": list(CONDITION_REQUEST_DIGEST_FIELDS),
            "condition_set_sha256": "b" * 64,
            "per_action_condition_set_sha256": per_action_digest,
            "training_reference_registry_sha256_by_action": registry,
            "train_prior_sha256_by_action": prior,
            "keycode": {
                "keycode_vocab": 16384,
                "per_key_and_event_counts_exact": True,
                "ellipsis_u2026_observed": True,
            },
            "per_action": per_action,
        }
        fs.atomic_json(supervisor.p["condition_preflight"], gate)
        self.assertTrue(supervisor._condition_preflight_complete())

        for field, replacement in (
            ("split_sha256", "0" * 64),
            ("generation_batch_size", 64),
            ("all_condition_source_code_eq_2", False),
            ("condition_and_noise_seed_domains_disjoint", False),
        ):
            tampered = copy.deepcopy(gate)
            tampered[field] = replacement
            fs.atomic_json(supervisor.p["condition_preflight"], tampered)
            with self.subTest(field=field):
                self.assertFalse(supervisor._condition_preflight_complete())
        tampered = copy.deepcopy(gate)
        tampered["per_action"]["tap"]["corpus_sha256"] = "0" * 64
        fs.atomic_json(supervisor.p["condition_preflight"], tampered)
        self.assertFalse(supervisor._condition_preflight_complete())
        self._close(supervisor)

    def test_e2e_gate_binds_current_split_corpus_and_exact_smoke_configuration(self):
        supervisor = fs.Supervisor(self.config)
        smoke_root = supervisor.p["e2e_smoke"]
        smoke_root.mkdir(parents=True)
        artifact = smoke_root / "artifact.bin"
        artifact.write_bytes(b"audited-smoke")
        split = supervisor._expected_split_audit()
        selected = {
            name: split[name + "_users"][:1] for name in ("train", "val", "test")
        }
        source_files = {}
        training, generation, detectors = {}, {}, {}
        for action in fs.ACTIONS:
            corpus = supervisor.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
            digest = fs.sha256_file(corpus)
            source_files[action] = {"path": str(corpus.resolve()), "sha256": digest, "n_events": 1}
            training[action] = {
                "action": action,
                "passed": True,
                "corpus_sha256": digest,
                "reference_seed": 42,
                "training_seed": 42,
                "loss": {"optimizer_steps": 40},
            }
            generation[action] = {
                "n_users": 3,
                "n_fake": 3,
                "ddim_steps": 50,
                "selector_used": False,
                "smoke_physical_validity_gate_passed": True,
                "formal_physical_gate_evaluated": True,
                "formal_physical_gate_passed": False,
                "formal_physical_gate_failures": [{
                    "user_id": int(selected["test"][0]),
                    "archive": "smoke-only.npz",
                    "error": "ValueError: formal clipping diagnostic exceeded",
                }],
                "train_prior_contains_only_fixed_train_users": True,
                "denoiser_calls": 150,
                "expected_denoiser_calls": 150,
            }
            detectors[action] = {
                "action": action,
                "passed": True,
                "detector_kind_count": 5,
                "detectors": {
                    name: {"passed": True} for name in (
                        "linear_svm", "rbf_svm", "xgboost", "tcn", "transformer"
                    )
                },
            }
        source = {
            "root": str(supervisor.p["corpus"]),
            "manifest": str(supervisor.p["corpus"] / "manifest.json"),
            "manifest_sha256": fs.sha256_file(supervisor.p["corpus"] / "manifest.json"),
            "audit": str(supervisor.p["corpus"] / "audit.json"),
            "audit_sha256": fs.sha256_file(supervisor.p["corpus"] / "audit.json"),
            "formal_audit": str(supervisor.p["corpus"] / "formal_audit" / "formal_data_audit.json"),
            "formal_audit_sha256": fs.sha256_file(
                supervisor.p["corpus"] / "formal_audit" / "formal_data_audit.json"
            ),
            "formal_audit_passed": True,
            "processed_users": 100,
            "files": source_files,
        }
        report = {
            "schema_version": "trajectory_finalized_v2_e2e_smoke_v2",
            "status": "passed",
            "formal_result": False,
            "runtime_determinism": dict(fs.EXPECTED_RUNTIME_DETERMINISM),
            "runtime_determinism_sha256": fs.STRICT_RUNTIME_DETERMINISM_SHA256,
            "source": source,
            "split": split,
            "selected_users": selected,
            "configuration": {
                "device": "cuda:0",
                "users_per_pool": 1,
                "samples_per_user_action": 1,
                "optimizer_steps_requested": 20,
                "training_diffusion_steps": 1000,
                "ddim_inference_steps": 50,
                "reference_seed": 42,
                "training_seed": 42,
                "generation_seed": 20260713,
            },
            "training": training,
            "generation": generation,
            "detectors": detectors,
            "all_training_loss_finite_and_decreased": True,
            "all_checkpoint_schedule_best_ema_gates_passed": True,
            "all_archive_adapter_paths_passed": True,
            "all_smoke_physical_validity_gates_passed": True,
            "all_formal_physical_gates_evaluated": True,
            "all_formal_physical_gates_passed": False,
            "all_25_detector_interface_smokes_passed": True,
            "detector_pairs_completed": 25,
            "artifact_hashes": {
                "artifact.bin": {
                    "sha256": fs.sha256_file(artifact),
                    "size_bytes": artifact.stat().st_size,
                }
            },
        }
        fs.atomic_json(smoke_root / "e2e_smoke.json", report)
        self.assertTrue(supervisor._e2e_smoke_complete())
        for mutate in (
            lambda row: row["configuration"].__setitem__("optimizer_steps_requested", 19),
            lambda row: row["split"].__setitem__("sha256", "0" * 64),
            lambda row: row["source"]["files"]["tap"].__setitem__("sha256", "0" * 64),
            lambda row: row.__setitem__("runtime_determinism_sha256", "0" * 64),
            lambda row: row.__setitem__("all_smoke_physical_validity_gates_passed", False),
            lambda row: row["generation"]["tap"].__setitem__("formal_physical_gate_failures", []),
        ):
            tampered = copy.deepcopy(report)
            mutate(tampered)
            fs.atomic_json(smoke_root / "e2e_smoke.json", tampered)
            self.assertFalse(supervisor._e2e_smoke_complete())
        self._close(supervisor)

    def test_corpus_audit_completion_rejects_stale_archive_hash(self):
        supervisor = fs.Supervisor(self.config)
        actions = {}
        totals = {"events": 0, "flat_rows": 0, "keys": 0}
        for index, action in enumerate(fs.ACTIONS, start=1):
            corpus = supervisor.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
            counts = {"events": index, "flat_rows": index * 10, "keys": index * 2}
            for name in totals:
                totals[name] += counts[name]
            actions[action] = {
                "action": action,
                "source": {
                    "npz": str(corpus.resolve()),
                    "sha256": fs.sha256_file(corpus),
                    "size_bytes": corpus.stat().st_size,
                    "all_fields_allow_pickle_false": True,
                    "object_array_count": 0,
                },
                "split": supervisor._expected_split_audit(),
                "counts": counts,
                "full_event_validation": {"events": index},
                "reference_gate": {
                    "require_all_users": True,
                    "users_with_fewer_than_six": {"train": [], "val": [], "test": []},
                },
            }
        audit = {
            "protocol": "strict_five_action_trajectory_corpus_v1",
            "corpus_dir": str(supervisor.p["corpus"]),
            "split": supervisor._expected_split_audit(),
            "actions": actions,
            "totals": totals,
            "formal_no_sample_cap": True,
            "passed": True,
        }
        fs.atomic_json(supervisor.p["corpus_audit"], audit)
        self.assertTrue(supervisor._completion_corpus())
        audit["actions"]["tap"]["source"]["sha256"] = "0" * 64
        fs.atomic_json(supervisor.p["corpus_audit"], audit)
        self.assertFalse(supervisor._completion_corpus())
        self._close(supervisor)

    def test_pair_completion_binds_current_bundle_bytes_and_full_pair_config(self):
        from detectors.pair_runner import PAIR_SCHEMA, stable_pair_seed

        supervisor = fs.Supervisor(self.config)
        identity = "tap/feature_pad/linear_svm"
        command = list(
            supervisor.manifest["commands"]["detector_pair_templates"][identity]["command"]
        )
        supervisor.manifest["commands"]["detector_pairs"] = {
            identity: {"resource": "cpu:0", "command": command}
        }
        dataset = supervisor.p["bundle"] / "tap.npz"
        dataset.parent.mkdir(parents=True)
        dataset.write_bytes(b"current-bundle-bytes")
        pair_root = supervisor.p["benchmark"] / "pairs" / "tap" / "feature_pad" / "linear_svm"
        result_dir = pair_root / "result"
        result_dir.mkdir(parents=True)
        plot = pair_root / "test_fa_frr.png"
        plot.write_bytes(b"plot")
        pair_seed = stable_pair_seed(20260713, "tap", "feature_pad", "linear_svm")
        pair_config = {
            "action": "tap",
            "family": "feature_pad",
            "detector": "linear_svm",
            "seed": pair_seed,
            "base_seed": 20260713,
            "seed_policy": "sha256(base_seed|action|family|detector)_uint32",
            "formal_protocol": True,
            "real_hash_seed": 20260713,
            "feature_bootstrap_replicates": 500,
            "deep_train": {
                "epochs": 40,
                "batch_size": 64,
                "learning_rate": 0.0003,
                "weight_decay": 0.0001,
                "patience": 0,
                "num_workers": 0,
                "seed": pair_seed,
                "bootstrap_replicates": 500,
                "gradient_clip_norm": 5.0,
            },
            "feature_model_params": {},
            "deep_model_params": {},
            "batch_probe": None,
        }
        manifest = {
            "schema_version": PAIR_SCHEMA,
            "status": "complete",
            "action": "tap",
            "family": "feature_pad",
            "detector": "linear_svm",
            "dataset_file": str(dataset.resolve()),
            "dataset_sha256": fs.sha256_file(dataset),
            "fake_user_split": str(supervisor.p["split"]),
            "fake_user_split_sha256": self.config["split_sha256"],
            "config": pair_config,
            "config_sha256": fs.canonical_sha256(pair_config),
            "result_dir": str(result_dir.resolve()),
            "plot": str(plot.resolve()),
            "plot_sha256": fs.sha256_file(plot),
            "artifact_hashes": {"score_dump.npz": "a" * 64},
            "dataset_relink_audit": {
                "protocol": "every_score_row_exactly_relinked_to_assigned_dataset_v1",
                "dataset_sha256": fs.sha256_file(dataset),
                "fake_user_split_sha256": self.config["split_sha256"],
                "real_hash_seed": 20260713,
            },
            "operating_rows": [
                {"operating_point": "eer"},
                {"operating_point": "val_frr_le_5pct"},
            ],
            "split_audit": {
                "fake_sample_counts": {"train": 14000, "val": 2000, "test": 4000}
            },
        }
        fs.atomic_json(pair_root / "pair_manifest.json", manifest)
        audited = {
            "artifact_hashes": manifest["artifact_hashes"],
            "rows": manifest["operating_rows"],
            "summary": {},
            "deep_training_audit": {},
            "dataset_relink_audit": manifest["dataset_relink_audit"],
        }
        with mock.patch(
            "detectors.pair_runner.audit_protocol_result", return_value=audited
        ) as audit_mock:
            self.assertTrue(supervisor._pair_complete(identity))
            call = audit_mock.call_args.kwargs
            self.assertEqual(Path(call["dataset_file"]), dataset)
            self.assertEqual(Path(call["fake_user_split"]), supervisor.p["split"])
            self.assertEqual(call["real_hash_seed"], 20260713)
            self.assertEqual(call["expected_dataset_sha256"], fs.sha256_file(dataset))
            self.assertEqual(
                call["expected_fake_user_split_sha256"], self.config["split_sha256"]
            )
            self.assertEqual(call["expected_bootstrap_seed"], pair_seed + 31)
            dataset.write_bytes(b"tampered-bundle-bytes")
            self.assertFalse(supervisor._pair_complete(identity))
        self._close(supervisor)

    def test_deep_pair_completion_passes_source_bound_identity_to_reaudit(self):
        from detectors.pair_runner import PAIR_SCHEMA, stable_pair_seed

        supervisor = fs.Supervisor(self.config)
        identity = "tap/deep_pad/tcn"
        command = list(
            supervisor.manifest["commands"]["detector_pair_templates"][identity]["command"]
        )
        supervisor._replace_cli_value(command, "--batch-size", 8)
        supervisor.manifest["commands"]["detector_pairs"] = {
            identity: {"resource": "cuda:0", "command": command}
        }
        dataset = supervisor.p["bundle"] / "tap.npz"
        dataset.parent.mkdir(parents=True)
        dataset.write_bytes(b"current-deep-bundle-bytes")
        probe_path = Path(command[command.index("--batch-probe-json") + 1])
        fs.atomic_json(probe_path, {
            "selected_batch_size": 8,
            "longest_observed_train_event_length": 123,
        })
        pair_root = supervisor.p["benchmark"] / "pairs" / "tap" / "deep_pad" / "tcn"
        result_dir = pair_root / "result"
        result_dir.mkdir(parents=True)
        plot = pair_root / "test_fa_frr.png"
        plot.write_bytes(b"deep-plot")
        pair_seed = stable_pair_seed(20260713, "tap", "deep_pad", "tcn")
        pair_config = {
            "action": "tap",
            "family": "deep_pad",
            "detector": "tcn",
            "seed": pair_seed,
            "base_seed": 20260713,
            "seed_policy": "sha256(base_seed|action|family|detector)_uint32",
            "formal_protocol": True,
            "real_hash_seed": 20260713,
            "feature_bootstrap_replicates": 500,
            "deep_train": {
                "epochs": 40,
                "batch_size": 8,
                "learning_rate": 0.0003,
                "weight_decay": 0.0001,
                "patience": 0,
                "num_workers": 0,
                "seed": pair_seed,
                "bootstrap_replicates": 500,
                "gradient_clip_norm": 5.0,
            },
            "feature_model_params": {},
            "deep_model_params": {},
            "batch_probe": {
                "path": str(probe_path.resolve()),
                "sha256": fs.sha256_file(probe_path),
                "selected_batch_size": 8,
                "longest_observed_train_event_length": 123,
                "truncation": False,
                "resampling": False,
            },
        }
        relink = {
            "protocol": "every_score_row_exactly_relinked_to_assigned_dataset_v1",
            "dataset_sha256": fs.sha256_file(dataset),
            "fake_user_split_sha256": self.config["split_sha256"],
            "real_hash_seed": 20260713,
        }
        rows = [
            {"operating_point": "eer"},
            {"operating_point": "val_frr_le_5pct"},
        ]
        manifest = {
            "schema_version": PAIR_SCHEMA,
            "status": "complete",
            "action": "tap",
            "family": "deep_pad",
            "detector": "tcn",
            "dataset_file": str(dataset.resolve()),
            "dataset_sha256": fs.sha256_file(dataset),
            "fake_user_split": str(supervisor.p["split"]),
            "fake_user_split_sha256": self.config["split_sha256"],
            "config": pair_config,
            "config_sha256": fs.canonical_sha256(pair_config),
            "result_dir": str(result_dir.resolve()),
            "plot": str(plot.resolve()),
            "plot_sha256": fs.sha256_file(plot),
            "artifact_hashes": {"score_dump.npz": "a" * 64},
            "dataset_relink_audit": relink,
            "operating_rows": rows,
            "split_audit": {
                "fake_sample_counts": {"train": 14000, "val": 2000, "test": 4000}
            },
        }
        fs.atomic_json(pair_root / "pair_manifest.json", manifest)
        audited = {
            "artifact_hashes": manifest["artifact_hashes"],
            "dataset_relink_audit": relink,
            "rows": rows,
            "summary": {"last_epoch": 40},
            "deep_training_audit": {
                "history_epoch_count": 40,
                "history_last_epoch": 40,
            },
        }
        with mock.patch(
            "detectors.pair_runner.audit_protocol_result", return_value=audited
        ) as audit_mock:
            self.assertTrue(supervisor._pair_complete(identity))
            call = audit_mock.call_args.kwargs
            source = call["expected_deep_run_identity"]
            self.assertEqual(source["dataset_sha256"], fs.sha256_file(dataset))
            self.assertEqual(source["fake_user_split_sha256"], self.config["split_sha256"])
            self.assertEqual(source["real_hash_seed"], 20260713)
            self.assertEqual(source["pair_config"], pair_config)
            self.assertEqual(call["expected_bootstrap_seed"], pair_seed + 17)
        self._close(supervisor)

    def test_bundle_completion_binds_generation_tree_registry_split_and_outputs(self):
        supervisor = fs.Supervisor(self.config)
        registry_paths = {}
        registry_hashes = {}
        for index, action in enumerate(fs.ACTIONS, start=1):
            path = supervisor.p["run"] / "registries" / (action + ".json")
            digest = self._write_canonical_registry(supervisor, action, path)
            registry_paths[action] = str(path.resolve())
            registry_hashes[action] = digest
        fs.atomic_json(supervisor.p["registry_map"], registry_paths)

        archive_hashes = {}
        for action_index, action in enumerate(fs.ACTIONS):
            for user_id in range(100):
                shard = (action_index * 100 + user_id) % 2
                path = (
                    supervisor.p["generation"] / "shards"
                    / ("shard_%03d_of_002" % shard) / action
                    / ("user_%03d.npz" % user_id)
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("%s:%d" % (action, user_id)).encode())
                archive_hashes[str(path.relative_to(supervisor.p["generation"]))] = fs.sha256_file(path)

        bundle = supervisor.p["bundle"]
        bundle.mkdir(parents=True)
        archive_hash_path = bundle / "fake_archive_file_hashes.json"
        fs.atomic_json(archive_hash_path, archive_hashes)
        fs.atomic_json(
            supervisor.p["generation"] / "generation_archive_file_hashes.json",
            archive_hashes,
        )
        sources, outputs, per_action, overlaps, split_rows = [], {}, {}, {}, {}
        split_users = fs.read_json(supervisor.p["split"])
        for action in fs.ACTIONS:
            output = bundle / (action + ".npz")
            output.write_bytes(("bundle:" + action).encode())
            real = supervisor.p["corpus"] / ("hmog_trajectory_%s.npz" % action)
            outputs[action] = {
                "path": str(output.resolve()),
                "sha256": fs.sha256_file(output),
                "n": 20300,
            }
            per_action[action] = {"real": 300, "fake": 20000, "feature_dim": 1}
            overlaps[action] = {"n_reference_events": 500}
            split_rows[action] = {
                "schema_version": "trajectory_detector_split_v1",
                "fake_policy": "fixed_disjoint_users_70_10_20",
                "real_policy": "sha256_ranked_complete_event_group_per_user_action_60_20_20",
                "real_hash_seed": 20260713,
                "fake_sample_counts": {"train": 14000, "val": 2000, "test": 4000},
                "counts": {
                    "real": {"train": 100, "val": 100, "test": 100},
                    "fake": {"train": 14000, "val": 2000, "test": 4000},
                },
                "real_complete_event_group_counts": {
                    "train": 100, "val": 100, "test": 100,
                },
                "user_counts": {
                    "real": {"train": 100, "val": 100, "test": 100},
                    "fake": {"train": 70, "val": 10, "test": 20},
                },
                "fake_users": {
                    pool: list(split_users[pool + "_users"])
                    for pool in ("train", "val", "test")
                },
            }
            sources.extend((
                {
                    "action": action,
                    "label": "real",
                    "path": str(real.resolve()),
                    "sha256": fs.sha256_file(real),
                    "n": 300,
                },
                {
                    "action": action,
                    "label": "fake",
                    "path": str(supervisor.p["generation"]),
                    "n": 20000,
                },
            ))
        split_audit_path = bundle / "split_audit.json"
        fs.atomic_json(split_audit_path, {
            "schema_version": "trajectory_detector_split_by_action_v1",
            "per_action": split_rows,
        })
        manifest = {
            "schema_version": "trajectory_pad_bundle_manifest_v2",
            "status": "complete",
            "fake_user_split": str(supervisor.p["split"]),
            "fake_user_split_sha256": self.config["split_sha256"],
            "fake_archive_dir": str(supervisor.p["generation"]),
            "fake_archive_file_count": 500,
            "fake_archive_file_hashes": str(archive_hash_path.resolve()),
            "fake_archive_file_hashes_sha256": fs.sha256_file(archive_hash_path),
            "reference_registry_map": str(supervisor.p["registry_map"]),
            "reference_registry_map_sha256": fs.sha256_file(supervisor.p["registry_map"]),
            "reference_registry_sha256_by_action": registry_hashes,
            "reference_overlap_with_detector_real_event_pools": overlaps,
            "real_hash_seed": 20260713,
            "per_action": per_action,
            "outputs": outputs,
            "sources": sources,
            "split_audit": str(split_audit_path.resolve()),
        }
        fs.atomic_json(bundle / "bundle_manifest.json", manifest)
        self.assertTrue(supervisor._bundle_complete())
        (bundle / "tap.npz").write_bytes(b"tampered")
        self.assertFalse(supervisor._bundle_complete())
        self._close(supervisor)

    def test_generation_shard_gate_requires_exact_unique_unit_coverage_and_provenance(self):
        supervisor = fs.Supervisor(self.config)
        checkpoint_paths, registry_paths = {}, {}
        checkpoint_hashes, registry_hashes = {}, {}
        for index, action in enumerate(fs.ACTIONS, start=1):
            checkpoint = supervisor.p["run"] / "checkpoints" / (action + ".pt")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(("checkpoint:" + action).encode())
            checkpoint_paths[action] = str(checkpoint.resolve())
            checkpoint_hashes[action] = fs.sha256_file(checkpoint)
            registry = supervisor.p["run"] / "registries" / (action + ".json")
            registry_hash = self._write_canonical_registry(
                supervisor, action, registry
            )
            registry_paths[action] = str(registry.resolve())
            registry_hashes[action] = registry_hash
        fs.atomic_json(supervisor.p["checkpoint_map"], checkpoint_paths)
        fs.atomic_json(supervisor.p["registry_map"], registry_paths)

        split = fs.read_json(supervisor.p["split"])
        split_by_user = {
            int(user): name
            for name in ("train", "val", "test")
            for user in split[name + "_users"]
        }
        rows = []
        for action_index, action in enumerate(fs.ACTIONS):
            for user_id in range(100):
                if (action_index * 100 + user_id) % 2 != 0:
                    continue
                archive = (
                    supervisor.p["generation"] / "shards" / "shard_000_of_002"
                    / action / ("user_%03d.npz" % user_id)
                )
                archive.parent.mkdir(parents=True, exist_ok=True)
                archive.write_bytes(("%s:%d" % (action, user_id)).encode())
                row = {
                    "passed": True,
                    "runtime_determinism": dict(fs.EXPECTED_RUNTIME_DETERMINISM),
                    "runtime_determinism_sha256": fs.STRICT_RUNTIME_DETERMINISM_SHA256,
                    "status": "generated",
                    "path": str(archive.resolve()),
                    "output_path": str(archive.resolve()),
                    "action": action,
                    "user_id": user_id,
                    "split": split_by_user[user_id],
                    "n_fake": 200,
                    "ddim_steps": 50,
                    "ddim_eta": 0.0,
                    "training_diffusion_steps": 1000,
                    "generation_base_seed": 20260713,
                    "generation_batch_size": 32,
                    "condition_request_seed_derivation":
                    "stable_seed(base_seed,action,user_id,sample_index)",
                    "ddim_noise_seed_derivation":
                    "stable_seed(condition_request_seed_xor_0xDD1A50,action,user_id,sample_index)",
                    "condition_request_seed_recomputed_count": 200,
                    "ddim_noise_seed_recomputed_count": 200,
                    "unique_condition_request_seed_count": 200,
                    "unique_ddim_noise_seed_count": 200,
                    "condition_and_noise_seed_domains_disjoint": True,
                    "condition_request_replay_count": 200,
                    "neural_ddim": True,
                    "selector_used": False,
                    "batch_size": 32,
                    "fixed_split_sha256": self.config["split_sha256"],
                    "checkpoint_sha256": checkpoint_hashes[action],
                    "reference_registry_sha256": registry_hashes[action],
                    "exact_replay_count": 0,
                    "exact_metadata_copy_count": 0,
                    "complete_key_sequence_copy_count": 0,
                    "condition_set_sha256": "a" * 64,
                }
                fs.atomic_json(archive.with_suffix(".audit.json"), row)
                rows.append(row)
        manifest = {
            "schema_version": "five_shot_generation_shard_manifest_v4",
            "formal": True,
            "runtime_determinism": dict(fs.EXPECTED_RUNTIME_DETERMINISM),
            "runtime_determinism_sha256": fs.STRICT_RUNTIME_DETERMINISM_SHA256,
            "selector_used": False,
            "generation_base_seed": 20260713,
            "generation_batch_size": 32,
            "condition_request_seed_derivation":
            "stable_seed(base_seed,action,user_id,sample_index)",
            "ddim_noise_seed_derivation":
            "stable_seed(condition_request_seed_xor_0xDD1A50,action,user_id,sample_index)",
            "condition_request_digest_schema": "trajectory_condition_request_canonical_v1",
            "condition_set_digest_schema": "trajectory_condition_request_set_v1",
            "condition_set_sha256": "b" * 64,
            "per_action_condition_set_sha256": {
                action: "c" * 64 for action in fs.ACTIONS
            },
            "shard_id": 0,
            "num_shards": 2,
            "planned_fake": 50000,
            "completed_fake": 50000,
            "planned_units": 250,
            "completed_units": 250,
            "ddim_steps": 50,
            "eta": 0.0,
            "fixed_refs_per_user_action": 5,
            "fixed_split_sha256": self.config["split_sha256"],
            "checkpoint_sha256_by_action": checkpoint_hashes,
            "reference_registry_sha256_by_action": registry_hashes,
            "condition_request_seed_recomputed_count": 50000,
            "ddim_noise_seed_recomputed_count": 50000,
            "unique_condition_request_seed_count": 50000,
            "unique_ddim_noise_seed_count": 50000,
            "condition_and_noise_seed_domains_disjoint": True,
            "condition_request_replay_count": 50000,
            "results": rows,
        }
        manifest_path = (
            supervisor.p["generation"] / "generation_manifest_shard_000_of_002.json"
        )
        fs.atomic_json(manifest_path, manifest)
        self.assertTrue(supervisor._generation_shard_complete(0))

        wrong_runtime_digest = copy.deepcopy(manifest)
        wrong_runtime_digest["runtime_determinism_sha256"] = "0" * 64
        fs.atomic_json(manifest_path, wrong_runtime_digest)
        self.assertFalse(supervisor._generation_shard_complete(0))

        wrong_row_runtime = copy.deepcopy(manifest)
        wrong_row_runtime["results"][0]["runtime_determinism"]["extra"] = True
        first_sidecar = Path(wrong_row_runtime["results"][0]["path"]).with_suffix(
            ".audit.json"
        )
        fs.atomic_json(first_sidecar, wrong_row_runtime["results"][0])
        fs.atomic_json(manifest_path, wrong_row_runtime)
        self.assertFalse(supervisor._generation_shard_complete(0))
        fs.atomic_json(first_sidecar, manifest["results"][0])
        fs.atomic_json(manifest_path, manifest)
        self.assertTrue(supervisor._generation_shard_complete(0))

        wrong_eta = copy.deepcopy(manifest)
        wrong_eta["results"][0]["ddim_eta"] = 0.5
        fs.atomic_json(first_sidecar, wrong_eta["results"][0])
        fs.atomic_json(manifest_path, wrong_eta)
        self.assertFalse(supervisor._generation_shard_complete(0))
        fs.atomic_json(first_sidecar, manifest["results"][0])
        fs.atomic_json(manifest_path, manifest)

        duplicated = copy.deepcopy(manifest)
        duplicated["results"][-1] = copy.deepcopy(duplicated["results"][0])
        fs.atomic_json(manifest_path, duplicated)
        self.assertFalse(supervisor._generation_shard_complete(0))
        self._close(supervisor)


if __name__ == "__main__":
    unittest.main()
