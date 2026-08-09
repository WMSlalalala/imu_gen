#!/usr/bin/env python3
"""Pin the generator run behind each released checkpoint, by name.

WHY THIS EXISTS
---------------
The upstream layout is `runs/<action>/diffusion/<protocol>/<run-name>/`, and the
sampler used to pick a checkpoint by taking the **most recently modified** run
under that protocol.  That works only as long as nobody ever trains again.

They did.  The A7 ablation retrains under `fewshot_adv` -- it has to, the
trainer only accepts the four protocol names -- so the moment it starts, its own
run becomes the newest and every subsequent `fewshot_adv` sample (the k_refs
sweep, among others) would be drawn from the ablation instead of from the
release.  No error, no warning: just an ablation silently measuring itself.

This was caught when an eight-epoch smoke run displaced the released scroll
checkpoint.  Nothing had sampled from it yet -- the caches on disk were verified
against their `ablation.json` and all named the right runs -- but the k_refs
caches were queued behind it and would have.

So the mapping is frozen here, by name, and resolved exactly.  A run that is not
in this registry can never be selected, however new it is; a registry entry that
has gone missing is an error rather than a silent fallback.

REGENERATING
------------
Only when the release itself is rebuilt from new training runs:

    python released_generators.py --refresh

That rewrites the JSON from whatever is newest, which is the old behaviour --
so do it deliberately, and check the diff.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

FINAL = Path("/mnt/share/mwang49/real-human/imu_gen/final")
REGISTRY = Path(__file__).resolve().parent / "released_generators.json"
ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
PROTOCOLS = ("noshot", "noshot_adv", "fewshot", "fewshot_adv")


def _runs_under(action: str, protocol: str) -> list:
    root = FINAL / "runs" / action / "diffusion" / protocol
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _checkpoint_in(run: Path) -> Path | None:
    checkpoints = sorted(
        (run / "checkpoints").glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    preferred = [c for c in checkpoints if c.name.startswith("best_")]
    if preferred:
        return preferred[0]
    return checkpoints[0] if checkpoints else None


@lru_cache(maxsize=1)
def registry() -> dict:
    if not REGISTRY.is_file():
        raise SystemExit(
            f"{REGISTRY.name} is missing. Run `python released_generators.py "
            "--refresh` once, from a tree where the newest run under each "
            "protocol really is the released one, and commit the result."
        )
    return json.loads(REGISTRY.read_text())["runs"]


def resolve(action: str, protocol: str) -> tuple:
    """Return (run directory, checkpoint) for the pinned release run.

    Raises rather than falling back: a missing pin means the tree no longer
    matches the release, and quietly substituting another run is exactly the
    failure this module exists to prevent.
    """

    pinned = registry().get(action, {}).get(protocol)
    if pinned is None:
        raise SystemExit(f"no released run pinned for {action}/{protocol}")
    run = FINAL / "runs" / action / "diffusion" / protocol / pinned["run"]
    if not run.is_dir():
        raise SystemExit(
            f"pinned run {pinned['run']} for {action}/{protocol} is gone from the "
            "tree; the registry and the runs directory disagree"
        )
    # The checkpoint is pinned by filename too.  Picking it by "prefer best_*"
    # chose a different file than the release for scroll and pinch, which both
    # sampled from `last.pt` -- a generator ablation anchored on the wrong
    # weights measures the wrong thing just as surely as one anchored on the
    # wrong run.
    checkpoint = run / "checkpoints" / pinned["checkpoint"]
    if not checkpoint.is_file():
        raise SystemExit(
            f"pinned checkpoint {pinned['checkpoint']} is missing from "
            f"{run.name}; the registry and the runs directory disagree"
        )
    return run, checkpoint


def available(action: str, protocol: str) -> bool:
    return registry().get(action, {}).get(protocol) is not None


# The cache the release was actually composed from.  Every sample in it records,
# in its own `metadata_json`, the run directory and checkpoint file that produced
# it -- which makes it the authority on what the release used, ahead of anything
# inferred from the runs tree.
RELEASE_CACHE = Path(
    "/mnt/share/mwang49/real-human/imu_gen/final/"
    "android_duration_time_fixed_20260720/user_cache_eval_200"
)


def _truth_from_release_cache(action: str) -> tuple | None:
    """(run name, checkpoint name) as recorded by the release's own cache."""

    import json as _json

    import numpy as _np

    for split in ("train", "val", "test"):
        sample = RELEASE_CACHE / f"user_000/{action}/{split}/sample_0000.npz"
        if not sample.is_file():
            continue
        with _np.load(sample, allow_pickle=False) as archive:
            meta = _json.loads(str(_np.asarray(archive["metadata_json"]).item()))
        run = str(meta.get("run_dir", "")).rstrip("/").split("/")[-1]
        checkpoint = str(meta.get("checkpoint", "")).split("/")[-1]
        if run and checkpoint:
            return run, checkpoint
    return None


def refresh() -> dict:
    """Rebuild the registry, preferring the release's own record over mtime.

    Taking the newest run was wrong in two ways at once, and both were measured:
    it named a different run than the release for tap and swipe, and it preferred
    a `best_*.pt` checkpoint where the release had used `last.pt` for scroll and
    pinch.  Only keystroke came out right.  The swipe mismatch was not cosmetic --
    the two runs differ in the feature-match weights, the learning rate and the
    epoch count -- so an ablation configured from it would have confounded the
    thing it was meant to isolate.

    For `fewshot_adv` the answer is read from the release cache, which records
    what actually produced each sample.  The other three protocols never entered
    the release, so there is nothing authoritative to read: those fall back to
    the newest run and are marked as such, since they only ever serve as
    ablation arms compared against each other.
    """

    runs: dict = {}
    for action in ACTIONS:
        runs[action] = {}
        for protocol in PROTOCOLS:
            if protocol == "fewshot_adv":
                truth = _truth_from_release_cache(action)
                if truth is not None:
                    runs[action][protocol] = {
                        "run": truth[0],
                        "checkpoint": truth[1],
                        "source": "release cache metadata_json",
                    }
                    continue
            chosen = None
            for run in _runs_under(action, protocol):
                checkpoint = _checkpoint_in(run)
                if checkpoint is not None:
                    chosen = {"run": run.name, "checkpoint": checkpoint.name,
                              "source": "newest run (never entered the release)"}
                    break
            if chosen is not None:
                runs[action][protocol] = chosen
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="rewrite the registry from whatever is newest")
    parser.add_argument("--check", action="store_true",
                        help="resolve every pin and report")
    args = parser.parse_args()

    if args.refresh:
        runs = refresh()
        REGISTRY.write_text(json.dumps(
            {
                "note": (
                    "The generator run behind each released checkpoint, pinned by "
                    "name. Selecting by mtime instead would let any later training "
                    "run -- including the A7 ablation, which must train under "
                    "fewshot_adv -- silently replace the release."
                ),
                "runs": runs,
            },
            indent=2, sort_keys=True,
        ))
        print(f"wrote {REGISTRY}")
        for action, protocols in sorted(runs.items()):
            for protocol, name in sorted(protocols.items()):
                print(f"  {action:10s} {protocol:12s} {name}")
        return

    for action in ACTIONS:
        for protocol in PROTOCOLS:
            if not available(action, protocol):
                print(f"  {action:10s} {protocol:12s} (none)")
                continue
            run, checkpoint = resolve(action, protocol)
            print(f"  {action:10s} {protocol:12s} {run.name}  <- {checkpoint.name[:52]}")


if __name__ == "__main__":
    main()
