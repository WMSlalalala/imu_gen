#!/usr/bin/env python3
"""Regenerate the inertial cache under a different protocol, for an ablation.

The released cache was drawn with the `fewshot_adv` protocol: five reference
windows of the victim, adversarial fine-tuning on.  Four protocols were trained
and all four sets of checkpoints exist, so the factorial

                     adversarial off   adversarial on
    k_refs = 0          noshot            noshot_adv
    k_refs = 5          fewshot           fewshot_adv   <- released

costs sampling and no training at all.  This driver draws a cache under any of
them, plus any reference count the trained encoder will accept.

NOTHING IN THE GENERATOR IS MODIFIED.  `AndroidIMUDiffusionLayer` already takes
`protocol` and `action_specs` as constructor arguments; the released pipeline
simply never passes them.  This script sets both and then hands control to the
authors' own `generate_user_cache_full.main`, so the request drawing, the
duration policy, the window contract and the cache layout are all theirs.

ONE THING TO DISCLOSE WHEN A REFERENCE COUNT IS OVERRIDDEN.  Every checkpoint
was trained at the reference count its own protocol names -- five for the
fewshot families, zero for the noshot ones.  Sampling `fewshot_adv` at one or
three or eight references therefore measures how the trained reference encoder
generalises across reference counts, which is a different question from how a
model trained for one reference would behave.  The distinction is recorded in
the cache's own metadata and has to survive into the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FINAL_ROOT = Path("/mnt/share/mwang49/real-human/imu_gen/final")
ANDROID_ROOT = FINAL_ROOT / "android_physical_layer_20260709"
UPSTREAM = Path("/mnt/share/mwang49/data7/code/direct100k/upstream_carrier_imu")


def newest_checkpoint(action: str, protocol: str) -> tuple[Path, Path]:
    """Locate the released run of one protocol and its best checkpoint.

    This used to take whichever run under the protocol was most recently
    modified.  That is safe only while nobody trains again -- and the A7
    ablation must retrain under `fewshot_adv`, because the trainer accepts only
    the four protocol names.  Its run would then have become the newest, and
    every later `fewshot_adv` draw (the whole k_refs sweep) would have sampled
    the ablation instead of the release, with no error to notice.

    The run is now pinned by name in `released_generators.json`.  A run that is
    not in that registry can never be selected, however new it is.
    """

    from released_generators import resolve

    return resolve(action, protocol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True,
                        choices=("noshot", "noshot_adv", "fewshot", "fewshot_adv"))
    parser.add_argument("--k-refs", type=int, default=None,
                        help="override the protocol's reference count; disclosed in metadata")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--actions", nargs="+", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--samples-per-user-action", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help=("sample from this run instead of the pinned release "
                              "run; required for A7, whose generator is retrained "
                              "and therefore is not in the released registry"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="checkpoint inside --run-dir; defaults to its newest best_*")
    args = parser.parse_args()

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(ANDROID_ROOT))
    sys.path.insert(0, str(UPSTREAM / "scripts"))

    from android_imu_layer import diffusion_generator as gen

    specs = {}
    resolved = {}
    unavailable = []
    actions = args.actions or list(gen.ACTION_SPECS)
    for action in list(actions):
        # Not every protocol was trained for every action -- the adversarial-off
        # `fewshot` family has no keystroke run upstream.  An action without a
        # checkpoint is dropped and recorded, never quietly filled from another
        # protocol, because that would make the cell measure a different thing
        # than the one it is labelled with.
        if args.run_dir is not None:
            # An explicit run, named rather than inferred.  Selecting by
            # recency is what let an ablation displace the release once; an
            # ablation that has to sample from its own retrained generator says
            # so on the command line instead.
            run = args.run_dir
            if args.checkpoint is not None:
                checkpoint = args.checkpoint
            else:
                candidates = sorted((run / "checkpoints").glob("best_*.pt"),
                                    key=lambda p: p.stat().st_mtime, reverse=True)
                if not candidates:
                    unavailable.append(action)
                    actions.remove(action)
                    continue
                checkpoint = candidates[0]
        else:
            try:
                run, checkpoint = newest_checkpoint(action, args.protocol)
            except SystemExit:
                unavailable.append(action)
                actions.remove(action)
                continue
        spec = dict(gen.ACTION_SPECS[action])
        spec["run_dir"] = str(run)
        spec["checkpoint"] = str(checkpoint)
        if args.sample_steps is not None:
            spec["sample_steps"] = int(args.sample_steps)
        specs[action] = spec
        resolved[action] = {"run": run.name, "checkpoint": checkpoint.name}
    for action in gen.ACTION_SPECS:
        specs.setdefault(action, gen.ACTION_SPECS[action])

    # The layer reads this module-level table when no action_specs is passed,
    # and the authors' cache script does not pass one.  Pointing it at the
    # ablation's checkpoints is the whole change.
    gen.ACTION_SPECS = specs

    original_init = gen.AndroidIMUDiffusionLayer.__init__

    def patched(self, *positional, **keyword):
        keyword["protocol"] = args.protocol
        keyword.setdefault("action_specs", specs)
        original_init(self, *positional, **keyword)
        if args.k_refs is None:
            return

        # The reference count has to be changed BEFORE the runtime is built,
        # not after.  `_get_runtime` reads it out of the protocol dict
        # (`k_refs = int(proto.get("k_refs", 0)) if proto.get("fewshot") else 0`)
        # and then constructs `UserRefBank(data, users, k_refs, seed)` with it --
        # so by the time a `_Runtime` object exists, the bank has already been
        # drawn at the old count and rewriting the attribute changes nothing.
        #
        # An earlier version of this patch did exactly that, and worse: it
        # iterated `self._runtimes`, which does not exist (the attribute is
        # `self._runtime`), and did so inside `__init__`, when no runtime has
        # been built yet.  It therefore did nothing at all, in silence, and the
        # first evidence was the cache script refusing a sample whose
        # ref_count was still 5 under a run declaring k_refs=1.
        #
        # Wrapping `protocol_cfg` puts the override where the value is actually
        # read, so the bank, the sampler and the recorded metadata all agree.
        original_protocol_cfg = self._fg["protocol_cfg"]

        def protocol_cfg(cfg, protocol):
            proto = dict(original_protocol_cfg(cfg, protocol))
            proto["k_refs"] = int(args.k_refs)
            # k_refs is only consulted when the protocol calls itself few-shot.
            proto["fewshot"] = int(args.k_refs) > 0
            return proto

        self._fg["protocol_cfg"] = protocol_cfg

    gen.AndroidIMUDiffusionLayer.__init__ = patched

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # The cache script asserts ref_count == 5 on every sample.  That guards a
    # property of the *released* cache -- that it is five-shot -- not a property
    # of the generator, and it is exactly what an ablation over the reference
    # count has to vary.  The assertion is retargeted at the count this run
    # declares, so a wrong count is still caught; nothing else about the check
    # changes, and the orientation invariant beside it is left alone.
    expected_refs = args.k_refs
    if expected_refs is None:
        expected_refs = 0 if args.protocol.startswith("noshot") else 5
    (args.out_dir / "ablation.json").write_text(
        json.dumps(
            {
                "protocol": args.protocol,
                "k_refs_override": args.k_refs,
                "k_refs_note": (
                    None
                    if args.k_refs is None
                    else "checkpoints were trained at their protocol's own "
                         "reference count; sampling at a different count measures "
                         "how the reference encoder generalises, not a model "
                         "trained for that count"
                ),
                "sample_steps_override": args.sample_steps,
                "checkpoints": resolved,
                "actions_without_checkpoint": unavailable,
                "generator_modified": False,
            },
            indent=2,
            sort_keys=True,
        )
    )

    import generate_user_cache_full as cache

    source = (UPSTREAM / "scripts" / "generate_user_cache_full.py").read_text()

    # The cache script hardcodes "this cache is five-shot" in TWO places, and
    # both have to move together.  Retargeting only the first one is a bug I
    # shipped and then watched cost real GPU time: the write path accepted a
    # zero-reference sample, the resume path then judged that same sample
    # invalid, quarantined it and drew it again -- so a restarted noshot run
    # redid every sample it had already finished, forever, while reporting
    # "skipped: 0".  9,351 good samples were moved out of the tree before this
    # was caught.
    #
    #   1. the write-time invariant, checked as each sample is produced
    #   2. `existing_sample_is_valid`, checked as each sample is resumed
    #
    # Both are retargeted to the reference count this run declares, so a wrong
    # count is still caught in both directions; nothing else about either check
    # changes, and the orientation and schema invariants beside them are left
    # exactly as the authors wrote them.
    rewrites = [
        (
            "if ref_count != 5 or used_refs.shape != (5,) or np.any(used_refs < 0):",
            f"if ref_count != {expected_refs} or used_refs.shape != ({expected_refs},) "
            f"or np.any(used_refs < 0):",
        ),
        (
            'if int(meta.get("ref_count", -1)) != 5:',
            f'if int(meta.get("ref_count", -1)) != {expected_refs}:',
        ),
        (
            "if refs.shape != (5,) or np.any(refs < 0) or len(np.unique(refs)) != 5:",
            f"if refs.shape != ({expected_refs},) or np.any(refs < 0) "
            f"or len(np.unique(refs)) != {expected_refs}:",
        ),
    ]
    module_source = source
    for original, replacement in rewrites:
        if original not in module_source:
            raise SystemExit(
                "a five-shot invariant in the cache script has moved:\n  "
                f"{original}\nre-read the script before retargeting it"
            )
        module_source = module_source.replace(original, replacement)
    namespace = cache.__dict__
    exec(compile(module_source, cache.__file__, "exec"), namespace)  # noqa: S102
    cache_main = namespace["main"]

    sys.argv = [
        "generate_user_cache_full.py",
        "--out-dir", str(args.out_dir),
        "--coverage-mode", "random",
        "--samples-per-user-action", str(args.samples_per_user_action),
        "--samples-per-active-len", "2",
        "--num-shards", str(args.num_shards),
        "--shard-index", str(args.shard_index),
    ]
    if actions:
        sys.argv += ["--actions", *actions]
    cache_main()


if __name__ == "__main__":
    main()
