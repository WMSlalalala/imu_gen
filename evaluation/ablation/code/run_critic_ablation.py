#!/usr/bin/env python3
"""A8-A11: remove one part of the adversarial objective at a time.

WHAT THESE ARMS ISOLATE
------------------------
A3 already answers "is the adversarial machinery worth anything" -- it removes
the whole of it and costs 0.034 FAR.  It cannot say which part earned that,
because four things run at once under `adv`:

    feature        a critic over distributional statistics of the window
    waveform       a critic over the sample as a signal, discriminated directly
    set            a critic over the five-shot reference set as a whole, so a
                   draw has to belong with its references rather than merely
                   look real alone
    feature_match  a direct statistical match on the same differentiable HMOG
                   features -- not a critic and not in `adv.critics`, but inside
                   the same `adv_update` block, so A3 removes it too

Each arm removes exactly one and keeps the other three, so the four numbers
decompose A3's 0.034 rather than repeating it.  Dropping A11 would leave the
feature pathway half-measured: A8 removes only its adversarial half, and the
direct match would keep supplying the same statistics.

WHY THESE ARMS ARE NOT DERIVED FROM A7'S CONFIG
------------------------------------------------
A7 disables the gradient protection (`project_conflicts: false`, an uncapped
`max_grad_ratio`).  Starting from that file would change two things at once and
the number would mean nothing.  The base here is the released run's own
`effective_config.json`, exactly as A7 derives its own base, with the gradient
protection left on and one `adv.critics.<name>` flipped to false.

A PRIOR THAT SHOULD BE STATED
------------------------------
The whole adversarial effect is 0.034 and arm means vary run-to-run by about
0.005 (measured on A1/A5/A6, which saturate and should agree).  A component
carrying an even quarter of it (0.008) therefore sits at roughly 1.7x that
variation.  These arms are powered to find a component that dominates, and are
not powered to separate four near-equal contributors.  That is a property of the
measurement, and belongs in the paper next to the numbers rather than being
discovered afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

FINAL = Path("/mnt/share/mwang49/real-human/imu_gen/final")
PYTHON = "/home/mwang49/miniconda3/envs/cuhkx/bin/python"
ROOT = Path("/mnt/share/mwang49/data7/results/direct100k/baselines/ablations")

# arm -> (kind, key, directory and run-name suffix)
#
# Four arms, not three.  `feature_match` is not one of `adv.critics` -- it lives
# in the trainer, gated only by `feature_match_weight > 0` -- but it sits inside
# the same `adv_update` block, and its weighted loss joins the critics' in the
# gradient that gets projected and capped.  So A3 ("adversarial training off")
# removes four things, and three critic arms would leave a quarter of its 0.034
# unattributed.  A11 closes that.
ARMS = {
    "A8":  ("critic", "feature",              "a8_no_feature"),
    "A9":  ("critic", "set",                  "a9_no_set"),
    "A10": ("critic", "waveform",             "a10_no_waveform"),
    "A11": ("weight", "feature_match_weight", "a11_no_feature_match"),
}


def released_run(action: str) -> Path:
    """The run the release actually samples, by pinned name -- never by mtime.

    Same registry `generate_imu_ablation.py` reads.  These arms train under
    `fewshot_adv` like the release does, so an mtime lookup would let the first
    critic arm become the "release" the next one is configured against.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from released_generators import resolve

    run, _checkpoint = resolve(action, "fewshot_adv")
    return run


def build_config(action: str, arm: str, output: Path) -> dict:
    """The released run's effective config with exactly one component removed."""
    kind, key, _suffix = ARMS[arm]
    run = released_run(action)
    effective = json.loads((run / "effective_config.json").read_text())
    adv = effective.setdefault("adv", {})
    critics = adv.setdefault("critics", {})

    if kind == "critic" and critics.get(key) is not True:
        raise SystemExit(
            f"{action}/{arm}: the released run had critics.{key}="
            f"{critics.get(key)}; there is nothing for this arm to remove"
        )
    if kind == "weight" and not float(adv.get(key, 0.0)) > 0.0:
        raise SystemExit(
            f"{action}/{arm}: the released run had {key}={adv.get(key)}; "
            f"there is nothing for this arm to remove"
        )
    # The gradient protection is the release's and stays on: it is A7's variable,
    # not this one's.  Asserted rather than assumed, because deriving this config
    # from the wrong base is the one mistake that would silently invalidate all
    # three arms.
    if adv.get("project_conflicts") is not True or adv.get("max_grad_ratio") != 0.5:
        raise SystemExit(
            f"{action}: base config has project_conflicts="
            f"{adv.get('project_conflicts')} max_grad_ratio={adv.get('max_grad_ratio')}; "
            f"expected the release's True/0.5 -- this looks like A7's config, not the release's"
        )

    if kind == "critic":
        before = {"critics": dict(critics)}
        critics[key] = False
        after = {"critics": dict(critics)}
    else:
        before = {key: adv.get(key)}
        adv[key] = 0.0
        after = {key: adv[key]}
    adv["enabled"] = True

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(effective, sort_keys=True))
    return {
        "action": action,
        "arm": arm,
        "removed": key,
        "released_run": str(run),
        "epochs": int(effective.get("train", {}).get("epochs", 0)),
        "batch_size": int(effective.get("train", {}).get("batch_size", 0)),
        "before": before,
        "after": after,
        "gradient_protection": {"project_conflicts": adv.get("project_conflicts"),
                                "max_grad_ratio": adv.get("max_grad_ratio"),
                                "g_weight": adv.get("g_weight")},
        "config": str(output),
    }


def merge_plan(path: Path, records: list[dict]) -> list[dict]:
    """Merge into the plan rather than replacing it.

    A7's driver rewrote its plan from `--actions` alone, so re-running it for one
    action silently dropped the other -- and the downstream pipeline, which reads
    the plan to decide whether an arm has finished, then waited forever on an arm
    that had already trained.  Keyed by (action, arm) so a re-run updates its own
    entry and leaves every other one alone.
    """
    existing = json.loads(path.read_text()) if path.is_file() else []
    by_key = {(r["action"], r.get("arm")): r for r in existing}
    for record in records:
        by_key[(record["action"], record["arm"])] = record
    merged = sorted(by_key.values(), key=lambda r: (r.get("arm", ""), r["action"]))
    path.write_text(json.dumps(merged, indent=2, sort_keys=True))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--actions", nargs="+", default=["scroll", "swipe"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override; default is the released run's own count")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke only: cap steps per epoch")
    parser.add_argument("--tag", default="",
                        help="suffix the run name; use for smokes so their "
                             "checkpoints cannot be resumed by the real run")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the configs and print the commands, train nothing")
    args = parser.parse_args()
    if args.max_steps is not None and not args.tag:
        raise SystemExit("--max-steps is a smoke; pass --tag so it writes to its own run")

    for arm in args.arms:
        _kind, _key, suffix = ARMS[arm]
        out_root = ROOT / suffix
        out_root.mkdir(parents=True, exist_ok=True)

        records = []
        for action in args.actions:
            record = build_config(action, arm, out_root / f"config_{action}.yaml")
            records.append(record)
            print(f"{arm} {action}: {record['before']} -> {record['after']}"
                  f"  ({record['epochs']} epochs, from {Path(record['released_run']).name})",
                  flush=True)
        merge_plan(out_root / "plan.json", records)

        for record in records:
            action = record["action"]
            command = [
                PYTHON, "-m", "final_gen.train", "train",
                "--config", record["config"],
                "--action", action,
                "--method", "diffusion",
                "--protocol", "fewshot_adv",
                "--adv", "true",
                "--run-name", f"{action}_{suffix}{args.tag}",
            ]
            if args.epochs is not None:
                command += ["--epochs", str(args.epochs)]
            if args.max_steps is not None:
                command += ["--max-steps", str(args.max_steps)]
            print("  " + " ".join(command), flush=True)
            if args.dry_run:
                continue
            log = out_root / f"train_{action}.log"
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
            with log.open("a") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                        env=environment, cwd=str(FINAL))
            print(f"  {arm} {action}: exit {result.returncode}", flush=True)


if __name__ == "__main__":
    main()
