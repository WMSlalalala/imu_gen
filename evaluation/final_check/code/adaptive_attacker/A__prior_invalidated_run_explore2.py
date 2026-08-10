import numpy as np, gzip, json, collections, os
BUNDLES={'full':'replay_dataset_full','v3':'replay_dataset_v3','v8':'replay_dataset_v8','v10':'replay_dataset_v10','v15':'replay_dataset_v15'}
ROOT='/mnt/share/mwang49/data7/results/direct100k'
CELLS='/home/mwang49/new/data7/data7_final_monitor_metrics_v1/USENIX8.25/code/dataset_test/results/cells'
users=['hmog_u006','hmog_u008']
def dur_map(bundle, u):
    z=np.load(f'{ROOT}/{bundle}/shards/{u}.npz', allow_pickle=True)
    off=z['offsets']; tr=z['trajectory_flat']; eid=z['event_id']; act=z['action']; lab=z['label']
    out={}
    first_nonzero=0
    for i in range(len(eid)):
        a,b=off[i],off[i+1]
        d=float(tr[b-1,7]); f=float(tr[a,7])
        if f!=0.0: first_nonzero+=1
        out[str(eid[i])]=(d, str(act[i]), int(lab[i]), b-a)
    return out, first_nonzero, len(eid)

for bn,b in BUNDLES.items():
    m,fnz,n=dur_map(b,'hmog_u006')
    print(bn, 'events', n, 'first-row-nonzero', fnz)

# overlap with scroll cell (v15)
rows=[json.loads(l) for l in gzip.open(f'{CELLS}/scroll__imu_only__behaveformer_stdat/test_scores.jsonl.gz','rt')]
r6=[r for r in rows if r['user_id']=='hmog_u006']
for bn,b in BUNDLES.items():
    m,_,_=dur_map(b,'hmog_u006')
    hit=sum(1 for r in r6 if r['event_id'] in m)
    hg=sum(1 for r in r6 if r['label']==0 and r['event_id'] in m)
    hf=sum(1 for r in r6 if r['label']==1 and r['event_id'] in m)
    print('scroll cell u006', bn, 'hit', hit, '/', len(r6), 'gen', hg, 'fake', hf)
