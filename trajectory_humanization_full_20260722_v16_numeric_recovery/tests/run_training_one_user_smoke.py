#!/usr/bin/env python3
"""Real one-user, five-action loader/collator/model smoke (not formal training)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory.data import FewShotExample, collate_fewshot_trajectories
from trajectory.model import TrajectoryDiffusion
from training.corpus import ACTIONS, FORMAL_SPLIT_PATH, NumericTrajectoryCorpus, SplitDefinition, atomic_json_dump


def main() -> int:
    torch.manual_seed(42)
    torch.set_num_threads(1)
    # This smoke was produced by the current strict extractor, including the
    # real OneFingerTouch two-endpoint fallback used when TouchEvent is sparse.
    source = ROOT / "results" / "smoke_fallback_user100669"
    split = SplitDefinition.load(FORMAL_SPLIT_PATH, require_pinned_hash=True)
    results = {}
    for action in ACTIONS:
        started = time.time()
        corpus = NumericTrajectoryCorpus(
            source / ("hmog_trajectory_%s.npz" % action), split, expected_action=action
        )
        corpus_audit = corpus.audit(require_all_users=False, validate_every_event=True)
        user_indices = corpus.indices_for_user("train", 0)
        if user_indices.size < 6:
            raise ValueError("one-user smoke needs six events for %s" % action)
        refs = [corpus.canonical_sample(int(index)) for index in user_indices[:5]]
        target = corpus.canonical_sample(int(user_indices[5]))
        batch = collate_fewshot_trajectories([FewShotExample(target, refs)])
        batch.validate(require_references=True)
        model = TrajectoryDiffusion(
            action,
            diffusion_steps=4,
            base_channels=16,
            cond_dim=32,
            time_dim=16,
            n_blocks=1,
            dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        output = model.training_loss(batch)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if not torch.isfinite(output["loss"]):
            raise FloatingPointError(action)
        results[action] = {
            "source_events": len(corpus),
            "all_source_events_restored": corpus_audit["full_event_validation"]["events"] == len(corpus),
            "all_source_pointer_streams_restored": corpus_audit["full_event_validation"]["pointer_streams"],
            "all_source_keys_restored": corpus_audit["full_event_validation"]["keys"],
            "target_sample_id": target.sample_id,
            "reference_sample_ids": [ref.sample_id for ref in refs],
            "unique_references": len(set(ref.sample_id for ref in refs)) == 5,
            "target_excluded": target.sample_id not in {ref.sample_id for ref in refs},
            "n_keys": target.n_keys,
            "n_letters": target.n_letters,
            "pointer_start_offset_ms": target.pointer_start_offset_ms.tolist(),
            "pointer_end_offset_ms": target.pointer_end_offset_ms.tolist(),
            "target_timeline_lengths": [int(values.shape[0]) for values in target.pointer_features],
            "loss": float(output["loss"].item()),
            "elapsed_seconds": time.time() - started,
            "passed": True,
        }
    report = {
        "scope": "one real user, all five actions, one optimizer step each",
        "formal_training": False,
        "reason_not_formal": "one-user extraction cannot represent fixed 70/10/20 users",
        "source": str(source),
        "actions": results,
        "passed": all(value["passed"] for value in results.values()),
    }
    output = ROOT / "results" / "training_one_user_smoke.json"
    atomic_json_dump(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
