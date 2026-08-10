#!/usr/bin/env python
"""Extract test-split provenance summary per bundle. Read-only on /mnt/share."""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor

D = "/mnt/share/mwang49/data7/results/direct100k"
OUT = os.path.dirname(os.path.abspath(__file__))


def one(bundle):
    p = os.path.join(D, bundle, "provenance.jsonl")
    rows = []
    with open(p) as fh:
        for line in fh:
            if '"split": "test"' not in line:
                continue
            d = json.loads(line)
            if d.get("split") != "test":
                continue
            dn = d.get("donor") or {}
            tb = dn.get("target_binding") or {}
            rows.append(dict(
                event_id=d.get("event_id"), user_id=d.get("user_id"), action=d.get("action"),
                label=d.get("label"),
                out_imu_sha=d.get("output_imu_sha256"), out_traj_sha=d.get("output_trajectory_sha256"),
                out_pair=d.get("output_cross_modal_pair_id"),
                in_dur=d.get("input_duration_ms"), tgt_dur=d.get("target_duration_ms"),
                in_samples=d.get("input_samples"), out_samples=d.get("output_samples"),
                tgt_samples=d.get("target_samples"),
                src_cluster=d.get("source_cluster_id"),
                role=dn.get("role"), gmode=dn.get("generation_mode"),
                mat_cluster=dn.get("source_material_cluster_id"),
                shot=dn.get("source_material_shot_ordinal"),
                tmpl_sha=dn.get("source_template_sha256"),
                gen_raw_dur=dn.get("generated_raw_duration_ms"),
                req_raw_dur=dn.get("requested_raw_duration_ms"),
                gen_rows=dn.get("generated_raw_row_count"),
                tb_dur=tb.get("duration_ms"), tb_dur_src=tb.get("duration_source"),
                tb_plan=tb.get("bound_event_plan_sha256"),
                tb_raw_dur=tb.get("replay_raw_duration_ms"),
                tb_traj_idx=tb.get("trajectory_archive_index"),
                model_used=dn.get("model_used"), human_replay=dn.get("human_replay"),
                seed=dn.get("seed") or d.get("seed"),
                rebuild=d.get("rebuild_method"),
            ))
    with open(os.path.join(OUT, f"prov_{bundle}.json"), "w") as fh:
        json.dump(rows, fh)
    return bundle, len(rows)


if __name__ == "__main__":
    bundles = ["replay_dataset_full", "replay_dataset_v3", "replay_dataset_v8",
               "replay_dataset_v10", "replay_dataset_v15"]
    with ProcessPoolExecutor(max_workers=5) as ex:
        for b, n in ex.map(one, bundles):
            print(b, n, flush=True)
