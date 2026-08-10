#!/usr/bin/env python
"""Build agent B's index once and pickle it for the review harness."""
import os, sys, pickle, time
sys.path.insert(0, "/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/B")
os.chdir("/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/B")
import adaptive_b as AB

t0 = time.time()
cm, C, B, bd = AB.load_all()
IDX = AB.build_index(cm, C, B, bd)
REG = AB.make_regimes(IDX)
print("built", time.time() - t0, flush=True)
OUT = "/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/adaptive/RV"
with open(os.path.join(OUT, "idx.pkl"), "wb") as fh:
    pickle.dump(dict(IDX=IDX, REG=REG, cm=cm), fh, protocol=4)
print("pickled", time.time() - t0)
