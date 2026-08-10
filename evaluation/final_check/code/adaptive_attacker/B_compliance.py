import json, os
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "out")
S = json.load(open(os.path.join(OUT, "summary.json")))
cov = S["coverage"]
c = {
 "experiment": "adaptive_attacker_curve_and_duration_confound_agentB",
 "rules_satisfied": {
  "F1": "No threshold or model choice used test data for selection. Per-event thresholds are the frozen dev-selected frr5 cuts in each cell's thresholds.json (selection_split=development). The duration LLR is fitted on the 10 DEVELOPMENT users only. Session operating points use the F2 user-disjoint split of the test users, recorded below.",
  "F2": "No per-event dev scores exist, so the score LLR bins and the session null are fitted on a user-disjoint half of the 20 test users (fold = index parity of the sorted user id), evaluated on the other half, folds swapped and pooled. The all-20 oracle-calibration variant is reported beside every operating point as caught_oracle_calibration / pooled_frr_oracle_calibration.",
  "F3": "Fake sessions mirror the genuine skeleton exactly: same user, same length, same action multiset, same slot order, same detector, same threshold. The only variable is the attacker's selection rule and rejection ratio r.",
  "F4": "No new material is consumed. The experiment only reorders and subsamples the already published five-shot fake pool (200 events per user-action) and reads already published per-event scores.",
  "F5": "A3 is oracle selection and is labelled 'oracle, upper bound, not a threat model' at every appearance; it is never the headline. A1 and A2 use only quantities the attacker holds. The per-victim MINIMUM over surrogates is also oracle surrogate choice and is labelled as an upper bound; the reported A1 headline is the surrogate ENSEMBLE (fixed rule, no victim access) and the median over surrogates.",
  "F6": "No model was loaded, retrained or refit. Only frozen score files and frozen threshold files were read.",
  "F7": "Every cell is scored only on its own modality; cross-modality surrogates are used only as the attacker's own ranking signal, never as a victim score, and are reported separately.",
  "F8": "Same-mechanics control C0_genuine_reassembled: genuine events drawn from the victim's own genuine pool are put through the IDENTICAL session-assembly, mirroring the same skeletons. Its alarm rate is reported next to the attack arms.",
  "F9": "All intervals are user-clustered bootstraps over the 20 test users, 2000 replicates, with one shared resampled user multiset per replicate applied to every cell. Arm comparisons are reported as paired differences (caught minus session FRR) with an interval on the difference.",
  "F10": "B0 count-of-caught-events, B1 mean session score and B2 duration-only LLR appear at the same operating point under the same selection rule in summary.baselines.",
  "F11": "Every rate names its denominator. caught and session FRR are per SESSION; per_event_FAR_at_dev_frr5 is per EVENT and is labelled as such.",
  "F12": "The session FRR required to reach caught 50/80/95 percent is reported per rule and r (price_frr_for_caught_*).",
  "F13": "preregistration.json was written before any number in out/ existed and is embedded in summary.preregistration.",
  "F14": "The pacing_replay branch of the duration consequence analysis is labelled TRUE BY CONSTRUCTION at the point where the number appears.",
  "F15": "coverage.json records dropped sessions, scored-slot fractions, and the forced-reuse (pool shortfall) count at every rejection ratio."
 },
 "threshold_provenance": {
  "per_event_tau": "cells/<cell>/thresholds.json field frr5, selection_split=development, frozen before test.",
  "session_operating_point": "empirical p-value of the session statistic against the CALIBRATION-half genuine session distribution; alarm iff p<=alpha; alpha in {0.05, 0.01, alpha90}. alpha90 is chosen on calibration-half genuine sessions only.",
  "duration_llr": "41-quantile LLR fitted on the 10 development users of the same bundle."
 },
 "parity_table": {
  "users": "identical (20 test users) in every arm",
  "sessions": cov["regimes"],
  "detector": "frozen released cell, identical across arms",
  "threshold": "identical across arms within a cell",
  "variable_under_test": "attacker selection rule and rejection ratio r"
 },
 "preregistered_outcomes": S["preregistration"]["preregistered_outcomes"],
 "reductions_in_scope_declared": [
  "Sessions that mix keystroke with touch actions in the full binding inventory (27 of 477 test sessions) are dropped; 3 further sessions carry no scored slot. Both counts are in coverage.json.",
  "The A2 self-consistency feature vector is a fixed 5-statistic-per-channel summary (mean, sd, p5, p50, p95) of the modality's own channels, not a learned representation. Two A2 variants (pool-centroid distance and 10-NN pool density) plus a reversed-direction diagnostic are reported.",
  "The price curve is evaluated on a 240-point alpha grid, not on every achievable p-value level.",
  "Bootstrap replicates resample users only; the LLR fit and the null are not refitted inside the bootstrap."
 ]
}
json.dump(c, open(os.path.join(OUT, "compliance.json"), "w"), indent=1)
print("compliance written")
