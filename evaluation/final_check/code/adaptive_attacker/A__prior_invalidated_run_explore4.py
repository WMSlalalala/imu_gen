import json, gzip, collections, numpy as np
CELLS='/home/mwang49/new/data7/data7_final_monitor_metrics_v1/USENIX8.25/code/dataset_test/results/cells'
ACTS=['tap','scroll','swipe','pinch','keystroke']
# genuine event sets per action, from one modality/detector; check identity across all 18
sets={}
for a in ACTS:
    per=[]
    for m in ['imu_only','trajectory_xytime','imu_trajectory_xytime']:
        for d in ['authconformer','behaveformer_stdat','hmog_style_rf','hmog_style_svm','paper_svm','paper_xgboost']:
            rows=[json.loads(l) for l in gzip.open(f'{CELLS}/{a}__{m}__{d}/test_scores.jsonl.gz','rt')]
            g=frozenset(r['event_id'] for r in rows if r['label']==0)
            f=frozenset(r['event_id'] for r in rows if r['label']==1)
            per.append((g,f))
    print(a, 'genuine identical across 18:', len(set(x[0] for x in per))==1, 'n=',len(per[0][0]),
          '| fake identical:', len(set(x[1] for x in per))==1, 'n=',len(per[0][1]))
    sets[a]=per[0]
# sessions
gen_rows=[]
for a in ACTS:
    rows=[json.loads(l) for l in gzip.open(f'{CELLS}/{a}__imu_only__paper_svm/test_scores.jsonl.gz','rt')]
    gen_rows += [r for r in rows if r['label']==0]
print('total scored genuine', len(gen_rows))
sess=collections.defaultdict(collections.Counter)
for r in gen_rows: sess[r['session_id']][r['action']]+=1
print('n scored sessions', len(sess))
reg=collections.Counter()
lens=[]
for s,c in sess.items():
    has_k = c['keystroke']>0
    has_t = sum(c[a] for a in ['tap','scroll','swipe','pinch'])>0
    reg[('keystroke' if has_k else '')+('touch' if has_t else '')]+=1
    lens.append(sum(c.values()))
print(reg)
print('len stats', np.min(lens), np.percentile(lens,[5,50,95]), np.max(lens), np.sum(lens))
# per-user tap genuine counts
tc=collections.Counter(r['user_id'] for r in gen_rows if r['action']=='tap')
print('tap per user', sorted(tc.items()))
ac=collections.Counter(r['action'] for r in gen_rows)
print('per action', ac)
