import json
import io
import sys
import tempfile
import unittest
from dataclasses import asdict
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch

from scripts.migrate_v15_training_state import (
    expected_training_config,
    main,
    recursive_digest,
    sha256_file,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class NumericRecoveryMigrationTest(unittest.TestCase):
    def _config(self, root: Path):
        return {
            "formal_launch_authorized": False,
            "corpus_dir": str(root / "corpus"),
            "split_json": str(root / "split.json"),
            "run_root": str(root / "target"),
            "action_device": {"tap": "cuda:0", "scroll": "cuda:1"},
            "training": {
                "epochs": 100,
                "learning_rate": 0.0002,
                "weight_decay": 0.0001,
                "grad_clip_norm": 1.0,
                "ema_decay": 0.999,
                "diffusion_steps": 1000,
                "base_channels": 96,
                "cond_dim": 192,
                "time_dim": 96,
                "n_blocks": 8,
                "dropout": 0.05,
                "keycode_vocab": 16384,
                "seed": 42,
                "num_workers": 0,
                "checkpoint_every_steps": 500,
                "reference_cache_size": 2048,
                "amp_overflow_max_retries": 4,
            },
        }

    def _checkpoint(self, config, action, batch_size, output, incompatible=False):
        expected = expected_training_config(config, action, batch_size, output)
        source_config = asdict(expected)
        source_config.pop("amp_overflow_max_retries")
        source_config["output_dir"] = str(output)
        if incompatible:
            source_config["batch_size"] += 1
        return {
            "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
            "config": source_config,
            "model": {"weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)},
            "ema": {"shadow": torch.arange(3, dtype=torch.float32)},
            "optimizer": {"state": {0: {"step": torch.tensor(2)}}},
            "amp_scaler": {"scale": 1048576.0, "growth_tracker": 0},
            "rng_state": {"torch": torch.arange(8, dtype=torch.uint8)},
            "progress": {
                "epoch_index": 1,
                "next_batch_in_epoch": 2,
                "examples_seen_in_epoch": 3,
                "global_step": 4,
            },
            "source": {"action": action},
        }

    def _write_action(
        self, root, config, action, batch_size,
        incompatible_last=False, with_best=True,
    ):
        action_root = root / "source" / "training" / action
        action_root.mkdir(parents=True)
        best = action_root / "best_epoch_0020.pt"
        last = action_root / "last.pt"
        if with_best:
            torch.save(self._checkpoint(config, action, batch_size, action_root), best)
        torch.save(
            self._checkpoint(
                config, action, batch_size, action_root,
                incompatible=incompatible_last,
            ),
            last,
        )
        for name in ("source_audit.json", "reference_audit.json", "reference_registry.json"):
            write_json(action_root / name, {"action": action, "name": name})
        if with_best:
            write_json(action_root / "best_manifest.json", {
                "history": [{
                    "filename": best.name,
                    "path": str(best),
                    "checkpoint_sha256": sha256_file(best),
                }],
                "best": {"filename": best.name, "path": str(best)},
            })
        (action_root / "metrics.jsonl").write_text(
            json.dumps({"type": "train_epoch", "completed_epoch": 1}) + "\n",
            encoding="utf-8",
        )
        write_json(action_root / "run_manifest.json", {
            "protocol_version": "trajectory_diffusion_strict_five_ref_v2",
            "action": action,
        })
        return last

    def test_recursive_digest_supports_bfloat16(self):
        value = torch.arange(9, dtype=torch.bfloat16)
        self.assertEqual(recursive_digest(value), recursive_digest(value.clone()))

    def test_two_action_bootstrap_is_atomic_and_recoverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            selection = {"selected_batch_size_by_action": {"tap": 8, "scroll": 16}}
            write_json(root / "split.json", {})
            write_json(root / "source" / "supervisor_status.json", {
                "status": "failed", "source_tree_sha256": "a" * 64,
            })
            tap_last = self._write_action(root, config, "tap", 8)
            scroll_last = self._write_action(
                root, config, "scroll", 16,
                incompatible_last=True, with_best=False,
            )
            config["training_bootstrap"] = {
                "source_run_root": str(root / "source"),
                "source_tree_sha256": "a" * 64,
                "actions": {
                    "tap": {"last_checkpoint_sha256": sha256_file(tap_last)},
                    "scroll": {"last_checkpoint_sha256": sha256_file(scroll_last)},
                },
            }
            config_path = root / "config.json"
            selection_path = root / "selection.json"
            output = root / "target" / "manifests" / "bootstrap.json"
            write_json(config_path, config)
            write_json(selection_path, selection)
            argv = ["migrate", "--config", str(config_path), "--selection",
                    str(selection_path), "--output", str(output)]
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "incompatible"):
                        main()
            self.assertFalse((root / "target" / "training").exists())

            torch.save(
                self._checkpoint(config, "scroll", 16, scroll_last.parent),
                scroll_last,
            )
            config["training_bootstrap"]["actions"]["scroll"][
                "last_checkpoint_sha256"
            ] = sha256_file(scroll_last)
            write_json(config_path, config)
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
            internal = root / "target" / "training" / ".v15_to_v16_bootstrap_receipt.json"
            self.assertTrue(internal.is_file())
            self.assertEqual(json.loads(output.read_text()), json.loads(internal.read_text()))
            receipt = json.loads(output.read_text())
            for action in receipt["actions"]:
                for filename, checkpoint in action["checkpoint_receipts"].items():
                    self.assertEqual(
                        Path(checkpoint["target_path"]),
                        root / "target" / "training" / action["action"] / filename,
                    )
            self.assertEqual(
                set(path.name for path in (root / "target" / "training").iterdir()),
                {"tap", "scroll", ".v15_to_v16_bootstrap_receipt.json"},
            )
            output.unlink()
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
            self.assertTrue(output.is_file())

    def test_real_v15_checkpoints_migrate_without_changing_protected_state(self):
        project = Path(__file__).resolve().parents[1]
        source = Path(
            "/home/mwang49/real-human/imu_gen/final/trajectory_humanization_full_20260713/"
            "results/formal_eventplan_v15_100epoch_100k_20260721"
        )
        if not (source / "training" / "tap" / "last.pt").is_file():
            self.skipTest("frozen v15 checkpoint source is unavailable")
        formal_config = project / "orchestration" / (
            "formal_pipeline_config_eventplan_v16_numeric_recovery_20260722.json"
        )
        config = json.loads(formal_config.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config["run_root"] = str(root / "target")
            config_path = root / "config.json"
            selection_path = root / "selection.json"
            output = root / "target" / "manifests" / "bootstrap.json"
            write_json(config_path, config)
            write_json(selection_path, {
                "selected_batch_size_by_action": {
                    "tap": 128, "scroll": 256,
                }
            })
            argv = ["migrate", "--config", str(config_path), "--selection",
                    str(selection_path), "--output", str(output)]
            with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                [row["action"] for row in receipt["actions"]],
                ["tap", "scroll"],
            )
            for action_row in receipt["actions"]:
                for checkpoint_receipt in action_row["checkpoint_receipts"].values():
                    source_checkpoint = torch.load(
                        checkpoint_receipt["source_path"], map_location="cpu"
                    )
                    target_checkpoint = torch.load(
                        checkpoint_receipt["target_path"], map_location="cpu"
                    )
                    for name in ("model", "ema", "optimizer", "amp_scaler", "rng_state"):
                        expected_digest = checkpoint_receipt[
                            "protected_content_sha256"
                        ][name]
                        self.assertEqual(recursive_digest(source_checkpoint[name]), expected_digest)
                        self.assertEqual(recursive_digest(target_checkpoint[name]), expected_digest)
            scroll = next(row for row in receipt["actions"] if row["action"] == "scroll")
            self.assertEqual(set(scroll["checkpoint_receipts"]), {"last.pt"})
            self.assertFalse(
                (root / "target" / "training" / "scroll" / "best_manifest.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
