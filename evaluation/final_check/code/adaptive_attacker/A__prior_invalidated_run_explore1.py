import numpy as np, gzip, json, collections, os
CELLS='/home/mwang49/new/data7/data7_final_monitor_metrics_v1/USENIX8.25/code/dataset_test/results/cells'
def load_scores(cell):
    rows=[]
    for line in gzip.open(f'{CELLS}/{cell}/test_scores.jsonl.gz','rt'):
        rows.append(json.loads(line))
    return rows
# test users
rows=load_scores('scroll__imu_only__behaveformer_stdat')
users=sorted(set(r['user_id'] for r in rows))
print('users', len(users), users)
fake=[r for r in rows if r['label']==1]
print('fake sess ids sample', sorted(set(r['session_id'] for r in fake))[:5], len(set(r['session_id'] for r in fake)))
print('fake event id sample', fake[0])
# per user fake count
c=collections.Counter(r['user_id'] for r in fake)
print('fake per user', sorted(c.values())[:5], sorted(c.values())[-5:])
cg=collections.Counter(r['user_id'] for r in rows if r['label']==0)
print('genuine per user', sorted(cg.values()))
