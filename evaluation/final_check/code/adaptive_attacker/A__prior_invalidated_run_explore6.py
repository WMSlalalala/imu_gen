import numpy as np, collections, glob, os, json
ROOT='/mnt/share/mwang49/data7/results/direct100k'
BUNDLES=['replay_dataset_full','replay_dataset_v3','replay_dataset_v8','replay_dataset_v10','replay_dataset_v15']
def shard(bundle,u):
    z=np.load(f'{ROOT}/{bundle}/shards/{u}.npz', allow_pickle=True)
    off=z['offsets']; tr=z['trajectory_flat']
    dur=tr[off[1:]-1,7].astype(np.float64)
    first=tr[off[:-1],7].astype(np.float64)
    return dict(event_id=z['event_id'].astype(str), action=z['action'].astype(str),
                label=z['label'], dur=dur, first=first, split=str(z['split']), user=z['user_id'].astype(str))
# find test+dev users
users={}
for f in sorted(glob.glob(f'{ROOT}/replay_dataset_full/shards/*.npz')):
    u=os.path.basename(f)[:-4]
    z=np.load(f, allow_pickle=True)
    users[u]=str(z['split'])
test=[u for u,s in users.items() if s=='test']; dev=[u for u,s in users.items() if s=='development']
print('test',len(test),'dev',len(dev), dev)
diff=collections.Counter(); nz=0; ntot=0
for u in test+dev:
    base=shard('replay_dataset_full',u)
    ntot+=len(base['dur']); nz+= int((base['first']!=0.0).sum())
    for b in BUNDLES[1:]:
        m=shard(b,u)
        assert list(m['event_id'])==list(base['event_id'])
        d=np.abs(m['dur']-base['dur'])>1e-9
        if d.any():
            for a,l in zip(base['action'][d], base['label'][d]): diff[(b,a,int(l))]+=1
print('first-row nonzero count', nz, '/', ntot)
print('cross-bundle duration diffs:', dict(diff))
