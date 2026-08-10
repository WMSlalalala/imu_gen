#!/usr/bin/env python
"""Build read-only caches for the adaptive-attacker / duration-confound experiment.

Writes everything under OUT (scratchpad). Reads only from /mnt/share (read-only).

Cache contents
--------------
events_test.npz   : per-event records for the 20 TEST users
                    (event_id, user_id, session_id, action, label, duration, features)
events_dev.npz    : per-event records for the 10 DEVELOPMENT users
                    (action, label, duration) -- used to FIT the duration LLR (rule F1)
scores.npz        : per (modality, detector, action) per-event scores aligned to
                    the events_test ordering, plus the frozen per-event thresholds.
"""
import json
import os
import glob
import numpy as np

SHARDS = '/mnt/share/mwang49/data7/results/direct100k/replay_dataset_full/shards'
CELLS = '/mnt/share/mwang49/data7/results/direct100k/detectors_90cell/cells'
OUT = os.path.dirname(os.path.abspath(__file__))

ACTIONS = ['tap', 'scroll', 'swipe', 'pinch', 'keystroke']
MODALITIES = ['imu_only', 'trajectory_xytime', 'imu_trajectory_xytime']
DETECTORS = ['authconformer', 'behaveformer_stdat', 'hmog_style_rf',
             'hmog_style_svm', 'paper_svm', 'paper_xgboost']

# ---------------------------------------------------------------- features
# A2 (self-consistency) feature space. Model-free summary statistics computable
# by the attacker from its OWN generated pool with no detector and no labels.
IMU_CH = [0, 1, 2, 3, 4, 5]
TRJ_CH = [0, 1, 2, 4, 5, 6]   # x, y, pressure-like, size-like, dx, dy (ch7=elapsed, ch8=const)


def event_features(imu, trj, dur):
    """imu:(n,6) trj:(n,9) -> 1-D float64 feature vector."""
    f = []
    n = imu.shape[0]
    for c in IMU_CH:
        v = imu[:, c].astype(np.float64)
        d = np.diff(v) if n > 1 else np.zeros(1)
        f += [v.mean(), v.std(), v.min(), v.max(),
              np.abs(d).mean(), np.sqrt((v ** 2).mean())]
    for c in TRJ_CH:
        v = trj[:, c].astype(np.float64)
        d = np.diff(v) if n > 1 else np.zeros(1)
        f += [v.mean(), v.std(), v.min(), v.max(), np.abs(d).mean()]
    # geometry / timing
    x = trj[:, 0].astype(np.float64)
    y = trj[:, 1].astype(np.float64)
    if n > 1:
        seg = np.hypot(np.diff(x), np.diff(y))
        path = seg.sum()
    else:
        path = 0.0
    disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    f += [path, disp, path / (disp + 1e-6), float(n), float(dur),
          float(n) / (dur + 1e-6), path / (dur + 1e-6)]
    return np.asarray(f, dtype=np.float64)


def load_shard(path, want_features):
    z = np.load(path, allow_pickle=True)
    off = z['offsets']
    trj = z['trajectory_flat']
    imu = z['imu_flat']
    ne = len(off) - 1
    dur = trj[off[1:] - 1, 7].astype(np.float64)
    # invariant check: the first row of every event is elapsed 0.0
    assert np.abs(trj[off[:-1], 7]).max() == 0.0, path
    feats = None
    if want_features:
        feats = np.zeros((ne, N_FEAT), dtype=np.float64)
        for i in range(ne):
            a, b = off[i], off[i + 1]
            feats[i] = event_features(imu[a:b], trj[a:b], dur[i])
    return dict(event_id=z['event_id'].astype(str), user_id=z['user_id'].astype(str),
                session_id=z['session_id'].astype(str), action=z['action'].astype(str),
                label=z['label'].astype(np.int64), duration=dur, features=feats,
                split=str(z['split']))


# probe feature dimension
_z = np.load(os.path.join(SHARDS, 'hmog_u006.npz'), allow_pickle=True)
_off = _z['offsets']
N_FEAT = len(event_features(_z['imu_flat'][_off[0]:_off[1]],
                            _z['trajectory_flat'][_off[0]:_off[1]],
                            float(_z['trajectory_flat'][_off[1] - 1, 7])))
del _z


def main():
    paths = sorted(glob.glob(os.path.join(SHARDS, '*.npz')))
    test, dev = [], []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        sp = str(z['split'])
        del z
        if sp == 'test':
            test.append(p)
        elif sp == 'development':
            dev.append(p)
    print('test shards', len(test), 'dev shards', len(dev), 'n_feat', N_FEAT, flush=True)

    def concat(paths, want_features):
        recs = [load_shard(p, want_features) for p in paths]
        out = {}
        for k in ['event_id', 'user_id', 'session_id', 'action', 'label', 'duration']:
            out[k] = np.concatenate([r[k] for r in recs])
        if want_features:
            out['features'] = np.concatenate([r['features'] for r in recs])
        return out

    te = concat(test, True)
    de = concat(dev, False)
    print('test events', len(te['event_id']), 'dev events', len(de['event_id']), flush=True)
    np.savez_compressed(os.path.join(OUT, 'events_test.npz'), **te)
    np.savez_compressed(os.path.join(OUT, 'events_dev.npz'),
                        action=de['action'], label=de['label'], duration=de['duration'],
                        user_id=de['user_id'])

    # ------------------------------------------------------------- scores
    idx = {e: i for i, e in enumerate(te['event_id'])}
    n = len(te['event_id'])
    score_arrays = {}
    thresholds = {}
    coverage = {}
    for mod in MODALITIES:
        for det in DETECTORS:
            s = np.full(n, np.nan)
            got = 0
            for act in ACTIONS:
                cell = f'{act}__{mod}__{det}'
                fp = os.path.join(CELLS, cell, 'test_scores.jsonl')
                th = json.load(open(os.path.join(CELLS, cell, 'thresholds.json')))
                assert th['score_direction'] == 'larger_is_more_fake', cell
                thresholds[cell] = dict(frr5=float(th['frr5']), eer=float(th['eer']),
                                        selection_split=th['selection_split'],
                                        target_frr=float(th['target_frr']))
                miss = 0
                with open(fp) as fh:
                    for line in fh:
                        r = json.loads(line)
                        j = idx.get(r['event_id'])
                        if j is None:
                            miss += 1
                            continue
                        s[j] = r['fake_high_score']
                        got += 1
                coverage[cell] = dict(rows_unmatched=miss)
            score_arrays[f'{mod}|{det}'] = s
            print(mod, det, 'scored events', got, 'of', n, flush=True)
    np.savez_compressed(os.path.join(OUT, 'scores.npz'), **score_arrays)
    json.dump(dict(thresholds=thresholds, coverage=coverage, n_feat=N_FEAT),
              open(os.path.join(OUT, 'score_meta.json'), 'w'), indent=1)
    print('done')


if __name__ == '__main__':
    main()
