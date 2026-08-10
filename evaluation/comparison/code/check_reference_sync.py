#!/usr/bin/env python3
"""Do the two channels condition on the same five real recordings?

The threat model says the attacker holds five real events of the victim.  Two
generators run per fake event -- one writes the touch trajectory, one writes the
inertial window -- and each picks its own five references.  If those two sets are
not the same five, the attacker has in fact consumed up to ten recordings and the
"five-shot" claim is understated.

The two sides index into different arrays, so neither set of indices can be
compared directly.  Both are mapped onto HMOG event ids first:

  IMU side        `used_ref_indices` in the sample's `metadata_json` are row
                  numbers in `processed_xy4_20260702/hmog_<action>.npz`; that
                  file carries `event_id` per row.
  trajectory side `material_manifest.jsonl` records a `source_cluster_id` per
                  shot; `genuine_bindings.jsonl` maps that to `source_event_id`.

Reproduces the IMU selection rule as a check before comparing: the bank is
`np.random.default_rng(345 + user*1009).permutation(rows)[:5]`, so the reproduced
indices must equal the ones the cache recorded.  If that assertion fails the
comparison below is meaningless and the script stops.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROCESSED = Path("/mnt/share/mwang49/real-human/imu_gen/final/data/processed_xy4_20260702")
MATERIAL = Path("/mnt/share/mwang49/data7/results/direct100k/fiveshot_material/material_manifest.jsonl")
BINDINGS = Path("/mnt/share/mwang49/data7/results/direct100k/genuine_bindings_v1/genuine_bindings.jsonl")
CACHE = Path("/mnt/share/mwang49/real-human/imu_gen/final/android_duration_time_fixed_20260720/user_cache_eval_200")
KEYSTROKE_PROV = Path("/mnt/share/mwang49/data7/direct100k_final/datasets/keystroke/provenance.jsonl")
ACTIONS = ("tap", "scroll", "swipe", "pinch", "keystroke")
REF_BANK_SEED = 345          # 42 + 303, written into the generator as EXPECTED_REF_BANK_SEED
SHOTS = 5


def imu_refs(action: str, user: int, uid: np.ndarray) -> list[int]:
    rows = np.where(uid == user)[0]
    rng = np.random.default_rng(REF_BANK_SEED + user * 1009)
    return sorted(rng.permutation(rows)[:SHOTS].tolist())


def verify_rule(action: str, uid: np.ndarray) -> int:
    """The reproduced indices must match what the cache recorded."""
    checked = 0
    for sample in sorted(CACHE.glob(f"user_*/{action}/*/sample_0000.npz"))[:5]:
        meta = json.loads(str(np.load(sample, allow_pickle=True)["metadata_json"]))
        if meta.get("ref_count") != SHOTS:
            continue
        if int(meta.get("ref_bank_seed", -1)) != REF_BANK_SEED:
            raise SystemExit(f"{sample}: ref_bank_seed is {meta.get('ref_bank_seed')}, "
                             f"expected {REF_BANK_SEED}")
        user = int(sample.parent.parent.parent.name.split("_")[1])
        if sorted(meta["used_ref_indices"]) != imu_refs(action, user, uid):
            raise SystemExit(f"{sample}: reproduced indices do not match the recorded "
                             f"ones -- the selection rule assumed here is wrong")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cluster_to_event = {}
    for line in BINDINGS.open():
        row = json.loads(line)
        cluster_to_event[row["source_cluster_id"]] = int(row["source_event_id"])

    material = defaultdict(list)
    unmapped = 0
    for line in MATERIAL.open():
        row = json.loads(line)
        event = cluster_to_event.get(row["source_cluster_id"])
        if event is None:
            unmapped += 1
            continue
        material[(row["user_id"], row["action"])].append(event)
    if unmapped:
        raise SystemExit(f"{unmapped} material shots have no binding; mapping is incomplete")

    report = {"shots_per_side": SHOTS, "ref_bank_seed": REF_BANK_SEED,
              "material_shots": sum(len(v) for v in material.values()), "actions": {}}

    for action in ACTIONS:
        z = np.load(PROCESSED / f"hmog_{action}.npz", allow_pickle=True)
        uid, event_id = z["user_id"], z["event_id"]
        verified = verify_rule(action, uid)

        histogram, unions = Counter(), []
        for user in range(100):
            shots = material.get((f"hmog_u{user:03d}", action))
            rows = np.where(uid == user)[0]
            if not shots or len(shots) != SHOTS or len(rows) == 0:
                continue
            theirs = {int(event_id[i]) for i in imu_refs(action, user, uid)}
            histogram[len(theirs & set(shots))] += 1
            unions.append(len(theirs | set(shots)))
        groups = sum(histogram.values())
        report["actions"][action] = {
            "groups": groups,
            "rule_verified_on_samples": verified,
            "intersection_histogram": dict(sorted(histogram.items())),
            "fully_disjoint": histogram.get(0, 0),
            "mean_intersection": sum(k * v for k, v in histogram.items()) / groups,
            "mean_union": sum(unions) / len(unions),
            "union_is_ten": sum(1 for u in unions if u == 2 * SHOTS),
        }

    # keystroke is the exception, and it is settled in the published artefacts
    # rather than in the diffusion cache: its inertial channel is written by the
    # analytic adapter, which is handed the same five events the touch side used.
    total = same = 0
    for line in KEYSTROKE_PROV.open():
        row = json.loads(line)
        if row.get("label") != 1:
            continue
        donor = row.get("donor") or {}
        touch, inertial = donor.get("material_source_event_ids"), (donor.get("imu") or {}).get("source_event_ids")
        if touch is None or inertial is None:
            continue
        total += 1
        same += list(touch) == list(inertial)
    report["keystroke_published"] = {
        "fake_events": total, "same_five_both_channels": same,
        "note": "the diffusion cache's keystroke references are not used; the "
                "published keystroke IMU comes from the analytic adapter"}

    gestures = [a for a in ACTIONS if a != "keystroke"]
    report["headline"] = {
        "gesture_groups": sum(report["actions"][a]["groups"] for a in gestures),
        "gesture_fully_disjoint": sum(report["actions"][a]["fully_disjoint"] for a in gestures),
        "recordings_consumed_per_victim_action": {
            "four_gestures": "up to 10 (five per channel, drawn independently)",
            "keystroke": "5 (both channels share the same five)"},
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
