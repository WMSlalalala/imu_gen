#!/usr/bin/env python3
"""Assemble the four baselines into four self-contained, rerunnable folders.

Each folder holds the third-party code exactly as its authors published it, the
driver that feeds it HMOG data, a one-command reproduction script, the evidence
that the generator was fitted properly, and the detector-grid result.  A folder
can be copied elsewhere and rerun without reaching back into this tree for
anything except the carrier dataset, whose path is a script argument.

The shared harness is copied into every folder rather than referenced from one
place: a reader auditing folder 03 should not have to trust that folder 00 held
the same code when 03 was run.  ``00_common`` keeps the canonical copy and the
packaging records a digest of each file so drift is detectable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

CODE = Path("/mnt/share/mwang49/data7/code/baselines")
RESULTS = Path("/mnt/share/mwang49/data7/results/direct100k/baselines")
SOURCE_DATASET = Path("/mnt/share/mwang49/data7/results/direct100k/replay_dataset_zoh")

COMMON_FILES = (
    "hmog_baseline_common.py",
    "verify_harness.py",
    "summarise_far5.py",
    "build_comparison.py",
    "assemble_banks.py",
    "export_real_windows.py",
    "score_generator_quality.py",
    "diagnose_placement.py",
)

# folder -> (driver files, vendored repo dirs, runner source, doc source,
#            result globs relative to RESULTS)
LAYOUT = {
    "01_traj_pyclick_bezier": {
        "drivers": ("build_pyclick_baseline.py",),
        "vendor": ("pyclick",),
        "runner": "runners/run_pyclick.sh",
        "doc": "docs/01_traj_pyclick_bezier.md",
        "results": {
            "far5.txt": "far5_pyclick_traj.txt",
            "far5.json": "far5_pyclick_traj.json",
            "baseline_counts.json": "pyclick_bezier/baseline_counts.json",
            "release.json": "pyclick_bezier/release.json",
            "verify.txt": "verify_pyclick.txt",
        },
    },
    "02_traj_diffusion_ts": {
        "drivers": ("run_diffusion_ts.py", "build_sample_bank_baseline.py",
                    "reproduce_diffusion_ts_sines.py", "sweep_diffusion_ts.sh"),
        "vendor": ("DiffusionTS",),
        "runner": "runners/run_learned.sh",
        "doc": "docs/02_03_diffusion_ts.md",
        "results": {
            "far5.txt": "far5_diffts_trajectory.txt",
            "far5.json": "far5_diffts_trajectory.json",
            "baseline_counts.json": "diffts_trajectory/baseline_counts.json",
            "release.json": "diffts_trajectory/release.json",
            "placement.json": "placement_diffts_trajectory.json",
            "verify.txt": "verify_diffts_trajectory.txt",
            "quality": "quality_diffts",
            "generator_summaries": "diffts_summaries_trajectory",
        },
    },
    "03_imu_diffusion_ts": {
        "drivers": ("run_diffusion_ts.py", "build_sample_bank_baseline.py",
                    "reproduce_diffusion_ts_sines.py", "sweep_diffusion_ts.sh"),
        "vendor": ("DiffusionTS",),
        "runner": "runners/run_learned.sh",
        "doc": "docs/02_03_diffusion_ts.md",
        "results": {
            "far5.txt": "far5_diffts_imu.txt",
            "far5.json": "far5_diffts_imu.json",
            "baseline_counts.json": "diffts_imu/baseline_counts.json",
            "release.json": "diffts_imu/release.json",
            "verify.txt": "verify_diffts_imu.txt",
            "quality": "quality_diffts",
            "generator_summaries": "diffts_summaries_imu",
        },
    },
    "04_imu_tts_gan": {
        "drivers": ("run_tts_gan.py", "build_sample_bank_baseline.py",
                    "sweep_tts_gan.sh"),
        "vendor": ("TTSGAN",),
        "runner": "runners/run_learned.sh",
        "doc": "docs/04_imu_tts_gan.md",
        "results": {
            "far5.txt": "far5_ttsgan_imu.txt",
            "far5.json": "far5_ttsgan_imu.json",
            "baseline_counts.json": "ttsgan_imu/baseline_counts.json",
            "release.json": "ttsgan_imu/release.json",
            "verify.txt": "verify_ttsgan_imu.txt",
            "quality": "quality_ttsgan",
            "generator_summaries": "ttsgan_summaries_imu",
        },
    },
}

IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".git", "logs", "tb", "Checkpoints*", "OUTPUT",
    "*.npy", "*.pt", "*.pth",
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=IGNORE)


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=True)

    digests: dict[str, str] = {}
    common = root / "00_common"
    common.mkdir(exist_ok=True)
    for name in COMMON_FILES:
        copy_file(CODE / name, common / name)
        digests[f"00_common/{name}"] = digest(CODE / name)
    copy_file(CODE / "README_CN.md", root / "README_CN.md")
    copy_file(CODE / "docs" / "AUDIT_FIXES_CN.md", common / "AUDIT_FIXES_CN.md")
    copy_file(CODE / "run_everything.sh", common / "run_everything.sh")
    copy_file(CODE / "run_quality_sweep.sh", common / "run_quality_sweep.sh")

    missing: list[str] = []
    for folder_name, spec in LAYOUT.items():
        folder = root / folder_name
        folder.mkdir(exist_ok=True)
        for name in spec["drivers"]:
            copy_file(CODE / name, folder / name)
            digests[f"{folder_name}/{name}"] = digest(CODE / name)
        for name in COMMON_FILES:
            copy_file(CODE / name, folder / name)
        for name in spec["vendor"]:
            copy_tree(CODE / name, folder / name)
        copy_file(CODE / spec["runner"], folder / "run.sh")
        (folder / "run.sh").chmod(0o755)
        copy_file(CODE / spec["doc"], folder / "README_CN.md")
        for target_name, source_name in spec["results"].items():
            source = RESULTS / source_name
            if source.is_file():
                copy_file(source, folder / "results" / target_name)
            elif source.is_dir():
                copy_tree(source, folder / "results" / target_name)
            else:
                missing.append(f"{folder_name}: {source_name}")

    (root / "MANIFEST.json").write_text(
        json.dumps(
            {
                "carrier_dataset": str(SOURCE_DATASET),
                "folders": sorted(spec for spec in LAYOUT),
                "file_digests": digests,
                "missing_results": sorted(missing),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"packaged into {root}")
    if missing:
        print("results not yet available:")
        for item in sorted(missing):
            print(f"  {item}")


if __name__ == "__main__":
    main()
