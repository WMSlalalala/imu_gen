# Reconciliation: session-level aggregation, implementation A vs implementation B

Adjudicated with a **third, independent referee implementation** (`C/referee.py`), written from the spec again with a different seed scheme (900001+7r, R=20), both candidate user splits plus 200 random balanced 10/10 splits, both price-aggregation conventions, and a brute-force validation of the session cut rule. CPU only, score-space arithmetic on the frozen JSONL files. Nothing under `/mnt/share` was written.

## Verdict

**The two implementations agree.** Every mechanically substantive choice is identical in A and B, and the referee reproduces each of them exactly under its own reporting convention. There is no measurement error in either. All 52 flagged differences trace to exactly three *reporting* choices:

| # | Reporting choice | A | B | Who is right |
|---|---|---|---|---|
| 1 | Which fixed user-disjoint split is primary | sorted ids, alternating parity | sorted ids, first-10 vs last-10 | **Neither is wrong**; split-to-split SD is ~0.004, both sit inside it. Reconciled value = marginal over 200 random balanced splits. |
| 2 | Breakout rows (per-modality / per-detector) | **bootstrap mean** | **point estimate** | **B.** A's breakout rows are bootstrap means and disagree with A's own point estimates by up to 0.028, and are inconsistent with A's own ALL row. |
| 3 | How the price of detection is aggregated over 18 cells | FA where the **macro-averaged curve** crosses the target | **mean of the per-cell prices** | **B** for the headline (a defender operates one cell, so the average price over deployments is the operational quantity). A's per-cell prices are correct and match B's; only A's aggregate is a different quantity. |

Everything else -- threshold field (`frr5`), accept rule (`caught iff score >= threshold`), tie handling at the session cut, within-session sampling without replacement, the count-rule integer boundary, AUC, the event-level premise -- agrees to within draw noise. The referee brute-force-verified that the session cut both agents use is exactly the caught-maximising cut subject to calibration FRR <= q, for all 5 statistics x 2 targets x 2 folds on 3 spanning cells (assert passed, tolerance 1e-12).

## 1. Verified identical (no disagreement)

| Quantity | A | B | Referee |
|---|---|---|---|
| Event FAR@frr5, mean over 90 cells | 0.774575 | 0.774575 | 0.774575 |
| Event FRR@frr5, mean over 90 cells | 0.052332 | 0.052332 | 0.052332 |
| Genuine sessions / users | 474 / 20 | 474 / 20 | 474 / 20 |
| Session length min/median/mean/max | 1 / 14 / 18.97 / 76 | 1 / 14 / 18.97 / 76 | 1 / 14 / 18.97 / 76 |
| Mirror shortfalls | 0 | 0 | 0 (audited all 20 draws) |

Session ROC AUC (macro over 18 cells, mean over draws) -- **all three agree to <= 0.0008**:

| statistic | A | B | referee |
|---|---|---|---|
| S1_COUNT | 0.7711 | 0.7714 | 0.7709 |
| S2_MEAN | 0.8493 | 0.8497 | 0.8494 |
| S3_MAX | 0.7626 | 0.7633 | 0.7629 |
| S4_TRIMMED | 0.8110 | 0.8118 | 0.8106 |
| S5_LOGODDS | 0.7837 | 0.7834 | 0.7836 |

S1 integer count rule -- **all three agree**; both agents and the referee select k=5 under a 5% macro session-FA budget and k=2 by Youden J:

| k | session FA (A) | session FA (B) | session FA (ref) | caught (A) | caught (B) | caught (ref) |
|---|---|---|---|---|---|---|
| 1 | 0.4660 | 0.4660 | 0.4660 | 0.8166 | 0.8185 | 0.8165 |
| 2 | 0.2331 | 0.2331 | 0.2331 | 0.6580 | 0.6571 | 0.6572 |
| 3 | 0.1195 | 0.1195 | 0.1195 | 0.5359 | 0.5340 | 0.5359 |
| 4 | 0.0665 | 0.0665 | 0.0665 | 0.4419 | 0.4407 | 0.4419 |
| 5 | 0.0384 | 0.0384 | 0.0384 | 0.3661 | 0.3665 | 0.3661 |
| 6 | 0.0244 | 0.0244 | 0.0244 | 0.3051 | 0.3036 | 0.3042 |
| 7 | 0.0124 | 0.0124 | 0.0124 | 0.2529 | 0.2519 | 0.2519 |
| 8 | 0.0069 | 0.0069 | 0.0069 | 0.2096 | 0.2090 | 0.2079 |
| 9 | 0.0039 | 0.0039 | 0.0039 | 0.1729 | 0.1719 | 0.1712 |
| 10 | 0.0027 | 0.0027 | 0.0027 | 0.1418 | 0.1408 | 0.1402 |

## 2. Disagreements > 0.005, with cause

### 2a. Caught-rate at session FRR = 5% and 1% (cause: user split)

The referee ran BOTH splits. It reproduces A under A's split to <= 0.0015 and B under B's split to <= 0.0025 (residual = the third seed scheme for the R=20 mirrored draw), which proves the pipelines are identical and the split is the only cause.

| stat | A (alt split) | ref under A's split | B (first10 split) | ref under B's split | **reconciled (split-marginal)** | split SD |
|---|---|---|---|---|---|---|
| S1_COUNT @5% | 0.3659 | 0.3660 | 0.3681 | 0.3695 | **0.3651** | 0.0042 |
| S2_MEAN @5% | 0.4891 | 0.4889 | 0.4889 | 0.4888 | **0.4881** | 0.0043 |
| S3_MAX @5% | 0.3997 | 0.4003 | 0.4105 | 0.4108 | **0.4041** | 0.0034 |
| S4_TRIMMED @5% | 0.4896 | 0.4897 | 0.4947 | 0.4946 | **0.4948** | 0.0041 |
| S5_LOGODDS @5% | 0.4021 | 0.4034 | 0.3977 | 0.3991 | **0.4005** | 0.0029 |
| S1_COUNT @1% | 0.2325 | 0.2310 | 0.2383 | 0.2375 | **0.2345** | 0.0040 |
| S2_MEAN @1% | 0.2505 | 0.2514 | 0.2561 | 0.2577 | **0.2562** | 0.0069 |
| S3_MAX @1% | 0.2585 | 0.2582 | 0.2511 | 0.2515 | **0.2533** | 0.0045 |
| S4_TRIMMED @1% | 0.2427 | 0.2429 | 0.2518 | 0.2543 | **0.2533** | 0.0065 |
| S5_LOGODDS @1% | 0.2647 | 0.2642 | 0.2739 | 0.2749 | **0.2703** | 0.0038 |

Verdict: **neither is wrong**. Both splits are legitimate fixed, documented, user-disjoint halves. Over 200 random balanced 10/10 user splits the macro caught-rate has SD 0.003-0.007 and every A-vs-B gap here (max 0.011) lies inside the 95% split band. The reconciled value is the split-marginal mean, which removes the arbitrary choice.

### 2b. Per-modality and per-detector breakouts (cause: bootstrap mean vs point estimate, stacked on the split)

A's breakout rows report `bootstrap[...]['mean']`; B's report the point estimate. A's own point estimates are recoverable from A's JSON and match the referee under A's split:

| stat / group | A reported (boot mean) | A's own point est. | B reported (point) | ref, A-split | ref, B-split | **reconciled** |
|---|---|---|---|---|---|---|
| S2_MEAN / hmog_style_svm | 0.3813 | 0.3761 | 0.3964 | 0.3793 | 0.3957 | **0.3785** |
| S2_MEAN / hmog_style_rf | 0.5277 | 0.5277 | 0.5312 | 0.5275 | 0.5305 | **0.5302** |
| S2_MEAN / paper_svm | 0.3331 | 0.3224 | 0.3177 | 0.3194 | 0.3169 | **0.3242** |
| S2_MEAN / paper_xgboost | 0.7066 | 0.7108 | 0.7013 | 0.7100 | 0.6993 | **0.7006** |
| S2_MEAN / behaveformer_stdat | 0.4077 | 0.4045 | 0.4074 | 0.4047 | 0.4102 | **0.4007** |
| S2_MEAN / authconformer | 0.5970 | 0.5928 | 0.5791 | 0.5923 | 0.5802 | **0.5944** |
| S4_TRIMMED / hmog_style_svm | 0.3801 | 0.3778 | 0.3943 | 0.3823 | 0.3962 | **0.3825** |
| S4_TRIMMED / hmog_style_rf | 0.5576 | 0.5531 | 0.5618 | 0.5539 | 0.5595 | **0.5630** |
| S4_TRIMMED / paper_svm | 0.3311 | 0.3034 | 0.3278 | 0.3007 | 0.3257 | **0.3229** |
| S4_TRIMMED / paper_xgboost | 0.6975 | 0.6968 | 0.7104 | 0.6996 | 0.7111 | **0.7105** |
| S4_TRIMMED / behaveformer_stdat | 0.4097 | 0.4132 | 0.3981 | 0.4123 | 0.3998 | **0.4104** |
| S4_TRIMMED / authconformer | 0.5805 | 0.5933 | 0.5755 | 0.5892 | 0.5757 | **0.5794** |


Reconciled breakout values are split-marginal, and a breakout averages only 3 cells, so it carries more split noise than the ALL row: measured SD across the 200 splits is 0.0043 for the ALL row but 0.005-0.013 per detector (largest: paper_svm S2 0.0130, paper_svm S4 0.0131). Every A-vs-B breakout gap listed here is within ~1.5 of that SD. **Treat detector-to-detector differences below ~0.02 as unresolved.**

Verdict: **B is right.** A's largest self-inconsistency is paper_svm / S4_TRIMMED, where A reported 0.3311 while A's own per-cell numbers average to 0.3034 -- a 2.8 pp bootstrap bias. With 20 unequal-sized user clusters the bootstrap mean is not an unbiased estimate of the caught rate and should not be reported as the caught rate. Note this does not touch A's headline ALL row, which is a point estimate.

### 2c. Price of detection (cause: aggregation convention)

This is the largest numerical gap in the whole comparison (up to 0.21 for S1) and it is **not an error in either sweep**. Proof: A's own per-cell prices, averaged over the 18 cells, reproduce B's aggregate.

| stat | target | A headline (macro-curve) | A's per-cell prices, averaged | B headline (mean per cell) | referee, mean per cell | referee, macro-curve |
|---|---|---|---|---|---|---|
| S1_COUNT | 50% | 0.1048 | 0.1363 | 0.1409 | 0.1406 | 0.1001 |
| S1_COUNT | 80% | 0.4440 | 0.6443 | 0.6545 | 0.6746 | 0.4440 |
| S1_COUNT | 95% | 1.0000 | 0.9687 | 0.9703 | 0.9687 | 1.0000 |
| S2_MEAN | 50% | 0.0587 | 0.0734 | 0.0739 | 0.0723 | 0.0556 |
| S2_MEAN | 80% | 0.2693 | 0.2756 | 0.2757 | 0.2723 | 0.2693 |
| S2_MEAN | 95% | 0.6967 | 0.5902 | 0.5860 | 0.5918 | 0.6954 |
| S3_MAX | 50% | 0.1023 | 0.1109 | 0.1078 | 0.1101 | 0.1008 |
| S3_MAX | 80% | 0.5225 | 0.4940 | 0.4921 | 0.4920 | 0.5246 |
| S3_MAX | 95% | 0.8480 | 0.7889 | 0.7891 | 0.7901 | 0.8438 |
| S4_TRIMMED | 50% | 0.0560 | 0.0715 | 0.0718 | 0.0704 | 0.0545 |
| S4_TRIMMED | 80% | 0.4051 | 0.3819 | 0.3807 | 0.3810 | 0.4051 |
| S4_TRIMMED | 95% | 0.7979 | 0.7126 | 0.7082 | 0.7139 | 0.7949 |
| S5_LOGODDS | 50% | 0.1133 | 0.1634 | 0.1640 | 0.1637 | 0.1079 |
| S5_LOGODDS | 80% | 0.4468 | 0.3987 | 0.4006 | 0.3979 | 0.4453 |
| S5_LOGODDS | 95% | 0.7409 | 0.5973 | 0.5966 | 0.5966 | 0.7365 |

Verdict: **B's convention is the right headline.** A defender deploys one (modality, detector) cell, so 'the false-alarm rate needed for 80% detection' is a per-cell quantity and the honest aggregate is its mean over deployments. A's macro-curve number asks a different question -- at what average FA does the average caught-rate cross the target -- which lets strong cells subsidise weak ones and is achievable by no single deployment. It is a legitimate secondary reading and is retained as such.

Precision caveat on S1: even under one convention the S1 price is unstable (A 0.6443, B 0.6545, referee 0.6746 at the 80% target) because an integer statistic makes each per-cell price a large jump. **Quote the S1 price to 1 decimal place at most: ~0.14 / ~0.65-0.67 / ~0.97.** The per-cell distribution is the honest summary: at 80% caught, S1's per-cell price has median 0.733 and range [0.139, 1.000].

### 2d. Calibration gap (cause: the same user split)

| stat | A @5% | B @5% | **reconciled @5%** | A @1% | B @1% | **reconciled @1%** |
|---|---|---|---|---|---|---|
| S1_COUNT | -0.0034 | -0.0070 | **-0.0026** | -0.0062 | -0.0126 | **-0.0095** |
| S2_MEAN | -0.0025 | -0.0035 | **-0.0017** | -0.0066 | -0.0131 | **-0.0116** |
| S3_MAX | +0.0039 | -0.0078 | **-0.0006** | -0.0145 | -0.0063 | **-0.0090** |
| S4_TRIMMED | +0.0087 | +0.0019 | **+0.0036** | -0.0147 | -0.0253 | **-0.0241** |
| S5_LOGODDS | -0.0015 | +0.0028 | **+0.0012** | -0.0018 | -0.0115 | **-0.0072** |

Both agents reached the same conclusion for the same reason and the referee confirms it: the gap is under 1 pp at a 5% budget, under 2.6 pp at a 1% budget, and **of inconsistent sign**, because the session threshold is a single empirical quantile of 474 genuine session statistics rather than a fitted model, and because the honest arm lands at a slightly different realised FA than the oracle arm. The pessimistic conclusion is not an artefact of holding users out.

## 3. FINAL RECONCILED NUMBERS

Release: repo-published cells (`.../USENIX8.25/code/dataset_test/results/cells`), the only tree where the task's premise reproduces (**event FAR@frr5 = 0.774575**, FRR@frr5 = 0.052332). The `/mnt/share/.../detectors_90cell/cells` path named in the DATA section is the r1 pre-fix baseline at event FAR 0.444733 -- both agents independently detected this and both ran both trees; that judgement is confirmed.

474 genuine sessions, 20 users, 18 (modality, detector) cells, R=20 mirrored fake-session draws, user-disjoint session-threshold calibration, split-marginal over 200 balanced 10/10 user splits. CI = user-cluster bootstrap, 2000 replicates, union of A's and B's intervals (the two designs differ slightly; the union is the conservative reading).

### Caught-rate at a 5% session false-alarm budget

| statistic | caught | 95% CI | realised session FA | oracle caught | session AUC |
|---|---|---|---|---|---|
| S1_COUNT | **0.3651** | [0.328, 0.418] | 0.0398 | 0.3625 | 0.7709 |
| S2_MEAN | **0.4881** | [0.427, 0.563] | 0.0521 | 0.4864 | 0.8494 |
| S3_MAX | **0.4041** | [0.373, 0.459] | 0.0513 | 0.4035 | 0.7629 |
| S4_TRIMMED | **0.4948** | [0.429, 0.561] | 0.0517 | 0.4984 | 0.8106 |
| S5_LOGODDS | **0.4005** | [0.368, 0.442] | 0.0518 | 0.4017 | 0.7836 |

### Caught-rate at a 1% session false-alarm budget

| statistic | caught | 95% CI | oracle caught |
|---|---|---|---|
| S1_COUNT | **0.2345** | [0.189, 0.282] | 0.2250 |
| S2_MEAN | **0.2562** | [0.167, 0.354] | 0.2446 |
| S3_MAX | **0.2533** | [0.198, 0.299] | 0.2443 |
| S4_TRIMMED | **0.2533** | [0.188, 0.350] | 0.2292 |
| S5_LOGODDS | **0.2703** | [0.216, 0.318] | 0.2631 |

### Price of detection (session FA the defender must accept)

Primary = mean over the 18 cells of each cell's own price. Median and range across cells given because the distribution is extremely skewed.

| statistic | 50% caught | 80% caught | 95% caught | 80%: median [min, max] across cells |
|---|---|---|---|---|
| S1_COUNT | 0.141 | 0.675 | 0.969 | 0.733 [0.139, 1.000] |
| S2_MEAN | 0.072 | 0.272 | 0.592 | 0.274 [0.023, 0.635] |
| S3_MAX | 0.110 | 0.492 | 0.790 | 0.498 [0.086, 0.751] |
| S4_TRIMMED | 0.070 | 0.381 | 0.714 | 0.376 [0.038, 0.717] |
| S5_LOGODDS | 0.164 | 0.398 | 0.597 | 0.353 [0.040, 0.698] |

Secondary (macro-averaged-curve convention, A's reading), for reference: S1_COUNT 0.100/0.444/1.000; S2_MEAN 0.056/0.269/0.695; S3_MAX 0.101/0.525/0.844; S4_TRIMMED 0.054/0.405/0.795; S5_LOGODDS 0.108/0.445/0.737.

### The deployable one-line count rule

Flag the session iff at least k of its events score at or above that event's own (action, modality, detector) dev `frr5` cut.

| k | macro session FA | macro caught |
|---|---|---|
| 1 | 0.4660 | 0.8165 |
| 2 | 0.2331 | 0.6572 |
| 3 | 0.1195 | 0.5359 |
| 4 | 0.0665 | 0.4419 |
| 5 | 0.0384 | 0.3661 |
| 6 | 0.0244 | 0.3042 |
| 7 | 0.0124 | 0.2519 |
| 8 | 0.0069 | 0.2079 |
| 9 | 0.0039 | 0.1712 |
| 10 | 0.0027 | 0.1402 |

- Best integer k under a 5% macro session-FA budget: **k = 5** -> FA 0.0384, caught 0.3661.
- Youden-optimal integer: **k = 2** -> FA 0.2331, caught 0.6572, J = 0.4241.
- With a properly calibrated (non-integer) user-disjoint threshold the same statistic gives caught 0.3651 at 5% and 0.2345 at 1%, so the integer restriction costs essentially nothing: S1 is simply a weak session statistic.

### Per-modality (caught @5% session FA)

| statistic | trajectory_xytime | imu_only | imu_trajectory_xytime |
|---|---|---|---|
| S1_COUNT | 0.3899 | 0.2562 | 0.4491 |
| S2_MEAN | 0.4841 | 0.3383 | 0.6419 |
| S3_MAX | 0.4202 | 0.3413 | 0.4509 |
| S4_TRIMMED | 0.5141 | 0.3697 | 0.6005 |
| S5_LOGODDS | 0.4304 | 0.2798 | 0.4915 |

Ordering is stable across every statistic and both operating points: joint imu+trajectory aggregates best, plain trajectory second, imu_only worst -- the exact inverse of the event-level FAR ordering (imu_only 0.8350, trajectory 0.7775, joint 0.7113). Both agents reported this and it survives reconciliation.

### Per-detector (caught @5% session FA)

| statistic | hmog_style_svm | hmog_style_rf | paper_svm | paper_xgboost | behaveformer_stdat | authconformer |
|---|---|---|---|---|---|---|
| S1_COUNT | 0.2899 | 0.3783 | 0.2628 | 0.4659 | 0.3045 | 0.4890 |
| S2_MEAN | 0.3785 | 0.5302 | 0.3242 | 0.7006 | 0.4007 | 0.5944 |
| S3_MAX | 0.2608 | 0.4561 | 0.2370 | 0.6192 | 0.3730 | 0.4785 |
| S4_TRIMMED | 0.3825 | 0.5630 | 0.3229 | 0.7105 | 0.4104 | 0.5794 |
| S5_LOGODDS | 0.1756 | 0.3916 | 0.1609 | 0.5341 | 0.4892 | 0.6518 |

Extremes at S2_MEAN: worst cell imu_only|authconformer 0.1807, best cell imu_trajectory_xytime|authconformer 0.8632.

### Secondary release (/mnt r1 pre-fix baseline, event FAR 0.4447) -- read only

| statistic | caught @5% | caught @1% | AUC | price 50/80/95 (mean per cell) |
|---|---|---|---|---|
| S1_COUNT | 0.6063 | 0.4714 | 0.8778 | 0.062 / 0.320 / 0.577 |
| S2_MEAN | 0.7678 | 0.6728 | 0.9350 | 0.030 / 0.116 / 0.275 |
| S3_MAX | 0.7077 | 0.5897 | 0.8884 | 0.057 / 0.230 / 0.385 |
| S4_TRIMMED | 0.7507 | 0.6680 | 0.9251 | 0.038 / 0.142 / 0.300 |
| S5_LOGODDS | 0.6939 | 0.6230 | 0.8988 | 0.071 / 0.185 / 0.330 |

Count rule on /mnt: best k under 5% FA = 5 (FA 0.0312, caught 0.5953); Youden k = 3, J = 0.6336. Both agents' /mnt numbers agree with each other and with the referee to <= 0.004.

## 4. Shared caveats both agents raised, both correct

- **S5_LOGODDS is ill-posed for two detectors.** hmog_style_svm and paper_svm emit raw SVM decision values (range -10.88 to +18.67); the specified clip to (0,1) saturates about half their events to +/-13.8. A reports 0.5044 / 0.4862 as the unweighted mean of per-cell out-of-unit fractions; B reports 0.5076 / 0.4894 as the event-pooled fraction. Both are correct under their own definition -- the tiny gap is cell-size weighting, verified. Read any S5 number for an SVM cell as a signed saturation count, not a log-odds sum.
- **Fake session structure is imposed, not observed.** Released fake events carry no session id (20 buckets, 1000 events each, 200 per user-action). Both agents mirrored genuine sessions (same user, same length, same action multiset, no within-session repeats, zero shortfalls -- independently audited by the referee on all 20 draws). Independent draws make fake session means concentrate more tightly than genuine ones, so the mean-type statistics flatter the defender: these caught-rates are an optimistic ceiling on session aggregation.
- **Two session regimes.** 145 of 474 sessions are keystroke-only, keystroke never co-occurs with scroll or pinch, and no session contains all five actions. A single pooled session threshold averages two structurally different populations.
- **S2/S3/S4/S5 mix score scales across actions within a session** (each action is a separately trained model). Only S1_COUNT is scale-free, since it applies each event's own action threshold.
- **20 users only.** All CIs are user-cluster bootstraps over 20 clusters and are wide; the detector ranking is suggestive, not resolved.

## 5. Confidence

**High** on the headline and on every number in section 3. Two independent implementations plus a third referee agree on the mechanics; the residual differences are reporting conventions that I identified in code, reproduced deliberately, and resolved. **Medium** on the S1 price of detection (integer statistic, implementation-to-implementation spread ~0.03 at the 80% target) and on the per-detector ordering (overlapping CIs). The scientific conclusion is unchanged by every disagreement found: at a 5% session false-alarm budget the best statistic catches under half of attacked sessions and the deployable count rule catches about a third; 80% detection costs roughly a 27% session false-alarm rate at best.

---

Files: `reconciled.json` (128 compared quantities, 52 flagged, each with cause), `C/referee.py` (referee implementation), `C/referee_repo.json`, `C/referee_mnt.json`, `C/referee_repo.log`, `C/referee_mnt.log`.
