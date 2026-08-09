#!/usr/bin/env python3
"""A7: replace the gradient-merging mechanism with a conventional weighted sum.

WHAT THIS ABLATION ISOLATES
----------------------------
The release merges the diffusion and adversarial objectives at the gradient
level rather than by adding two scalars.  `backward_with_projected_adversarial_
gradients` in the upstream trainer does three things together:

  1. backpropagates the reconstruction loss on its own, so the reconstruction
     gradient stays intact and can serve as the reference frame;
  2. when the adversarial gradient opposes it (dot < 0), projects the
     conflicting component away;
  3. rescales what is left so its norm is a fixed fraction (`max_grad_ratio`)
     of the reconstruction gradient norm -- recomputed every step.

The conventional alternative is L = L_recon + lambda * L_adv, whose lambda is
hard to calibrate: too large and the reconstruction is overwhelmed, too small
and the adversarial term does nothing.  This ablation measures exactly that.

HOW IT IS DONE WITHOUT TOUCHING THE UPSTREAM TRAINER
-----------------------------------------------------
No code is modified.  Setting `project_conflicts: false` and an effectively
infinite `max_grad_ratio` makes the authors' own merge path degenerate,
algebraically, into the weighted sum:

    projection_coeff = 0                      (projection disabled)
    merge_scale      = clamp(inf, max=1.0) = 1  (cap never binds)
    applied_weight   = lambda * 1 = lambda
    p.grad           = g_recon + lambda * g_adv

which is precisely the gradient of `L_recon + lambda * L_adv`, since gradients
add linearly and a separate backward pass changes nothing about the sum.  So
this arm differs from the release in the merge rule and in nothing else -- not
by inspection, but by construction.

EVERYTHING ELSE IS COPIED FROM THE RUN THE RELEASE ACTUALLY SAMPLED
--------------------------------------------------------------------
Each action's released checkpoint came from a run with its own hyper-parameters
(tap trains at g_weight 0.08 / ratio 0.6, scroll and swipe at 0.04 / 0.5,
keystroke at 0.02 / 0.25), and several of those were CLI overrides rather than
values in the base YAML.  This driver reads `effective_config.json` from the
exact run `generate_imu_ablation.py` selects, so the ablation inherits the
configuration that was really used rather than the file's defaults.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

FINAL = Path("/mnt/share/mwang49/real-human/imu_gen/final")
TRAINER = FINAL / "final_gen" / "train.py"
PYTHON = "/home/mwang49/miniconda3/envs/cuhkx/bin/python"
OUT_ROOT = Path("/mnt/share/mwang49/data7/results/direct100k/baselines/ablations/a7_weighted_sum")

# Large enough that `allowed_norm / weighted_norm` always exceeds 1 and the
# clamp at 1.0 makes the cap a no-op, without being an actual infinity that
# could turn into a nan when multiplied by a zero norm.
UNCAPPED_RATIO = 1.0e9


def released_run(action: str, protocol: str = "fewshot_adv") -> Path:
    """The run whose checkpoint the release samples, resolved by pinned name.

    The same registry `generate_imu_ablation.py` reads, deliberately: if the two
    ever disagreed, this ablation would be configured against a run the release
    never used, and the comparison would be against the wrong baseline.

    Pinning also protects this script from itself.  A7 trains under
    `fewshot_adv` -- the trainer accepts no other name for this protocol -- so
    an mtime-based lookup would make each A7 run the "release" for the next one.
    """

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from released_generators import resolve

    run, _checkpoint = resolve(action, protocol)
    return run


def build_config(action: str, output: Path) -> dict:
    """The released run's effective config, with exactly two fields changed."""

    run = released_run(action)
    effective = json.loads((run / "effective_config.json").read_text())
    adv = effective.setdefault("adv", {})

    before = {
        "project_conflicts": adv.get("project_conflicts"),
        "max_grad_ratio": adv.get("max_grad_ratio"),
        "g_weight": adv.get("g_weight"),
    }
    if before["project_conflicts"] is not True:
        raise SystemExit(
            f"{action}: the released run already had project_conflicts="
            f"{before['project_conflicts']}; this ablation assumes it was on"
        )

    adv["project_conflicts"] = False
    adv["max_grad_ratio"] = UNCAPPED_RATIO
    adv["enabled"] = True

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(effective, sort_keys=True))
    return {
        "action": action,
        "released_run": str(run),
        "epochs": int(effective.get("train", {}).get("epochs", 0)),
        "batch_size": int(effective.get("train", {}).get("batch_size", 0)),
        "before": before,
        "after": {
            "project_conflicts": adv["project_conflicts"],
            "max_grad_ratio": adv["max_grad_ratio"],
            "g_weight": adv["g_weight"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", nargs="+", default=["scroll", "swipe"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override; default is the released run's own count")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke only: cap steps per epoch")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the configs and print the commands, train nothing")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    plan = []
    for action in args.actions:
        config_path = OUT_ROOT / f"config_{action}.yaml"
        record = build_config(action, config_path)
        record["config"] = str(config_path)
        plan.append(record)
        print(f"{action}: {record['before']} -> {record['after']}"
              f"  ({record['epochs']} epochs, from {Path(record['released_run']).name})",
              flush=True)

    (OUT_ROOT / "a7_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))

    for record in plan:
        action = record["action"]
        run_name = f"{action}_a7_weighted_sum"
        # The trainer uses relative imports (`from . import __version__`), so it
        # runs as a module from the package's parent, not as a script path.
        command = [
            PYTHON, "-m", "final_gen.train", "train",
            "--config", record["config"],
            "--action", action,
            "--method", "diffusion",
            "--protocol", "fewshot_adv",
            "--adv", "true",
            "--run-name", run_name,
        ]
        if args.epochs is not None:
            command += ["--epochs", str(args.epochs)]
        if args.max_steps is not None:
            command += ["--max-steps", str(args.max_steps)]
        print("  " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        log = OUT_ROOT / f"train_{action}.log"
        import os

        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
        with log.open("a") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                    env=environment, cwd=str(FINAL))
        print(f"  {action}: exit {result.returncode}", flush=True)


if __name__ == "__main__":
    main()
