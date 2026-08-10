import numpy as np, collections
ROOT='/mnt/share/mwang49/data7/results/direct100k'
BUNDLES=['replay_dataset_full','replay_dataset_v3','replay_dataset_v8','replay_dataset_v10','replay_dataset_v15']
def dur_map(bundle,u):
    z=np.load(f'{ROOT}/{bundle}/shards/{u}.npz', allow_pickle=True)
    off=z['offsets']; tr=z['trajectory_flat']; eid=z['event_id']; act=z['action']; lab=z['label']
    d=tr[off[1:]-1,7].astype(np.float64)
    return dict(zip([str(x) for x in eid], zip(d,[str(a) for a in act],lab.tolist())))
u='hmog_u006'
maps={b:dur_map(b,u) for b in BUNDLES}
base=maps['replay_dataset_full']
for b in BUNDLES[1:]:
    m=maps[b]
    diffs=collections.Counter()
    maxd=0
    for k,(d,a,l) in base.items():
        d2=m[k][0]
        if abs(d-d2)>1e-9:
            diffs[(a,l)]+=1; maxd=max(maxd,abs(d-d2))
    print(b, 'ndiff', sum(diffs.values()), dict(diffs), 'maxabs', maxd)
# duration stats by action/label for full
for b in ['replay_dataset_full','replay_dataset_v15']:
    m=maps[b]
    agg=collections.defaultdict(list)
    for k,(d,a,l) in m.items(): agg[(a,l)].append(d)
    print('==',b)
    for k in sorted(agg): 
        v=np.array(agg[k]); print(' ',k,len(v),'med %.4f p5 %.4f p95 %.4f max %.4f'%(np.median(v),np.percentile(v,5),np.percentile(v,95),v.max()))
