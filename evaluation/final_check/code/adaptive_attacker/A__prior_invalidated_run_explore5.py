import json,gzip,numpy as np,collections,os
CM=json.load(open('/mnt/share/mwang49/data7/code/baselines/release_cell_map.json'))
cells=CM['cells']
print('n cells in map', len(cells))
far5=[]
keys=set()
for name,info in cells.items():
    rows=[json.loads(l) for l in gzip.open(info['scores'],'rt')]
    keys.update(rows[0].keys())
    s=np.array([r['fake_high_score'] for r in rows]); lab=np.array([r['label'] for r in rows])
    thr=info['frr5']
    far=float((s[lab==1]<thr).mean()); frr=float((s[lab==0]>=thr).mean())
    far5.append(far)
    if name in ('tap__imu_only__paper_xgboost','scroll__imu_only__behaveformer_stdat'):
        print(name,'FAR@frr5',round(far,4),'FRR@frr5',round(frr,4))
print('mean FAR@frr5 over 90', round(float(np.mean(far5)),6))
print('row keys', keys)
# bundles used
print(collections.Counter(v['bundle'] for v in cells.values()))
