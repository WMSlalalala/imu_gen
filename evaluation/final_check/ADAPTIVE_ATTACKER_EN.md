# Reconciliation: adaptive attacker + duration confound, session-level defence

Two agents implemented the same measurement over the published 90-cell release. This
document reconciles them. Every disputed number was **reproduced from a third,
independent referee implementation** written for this reconciliation, so that causes are
established by code and arithmetic rather than by inspection.

Referee code and outputs live in `adaptive/R/`
(`referee.py`, `run_referee.py`, `partb_referee.py`,
`referee_partA.json`, `referee_partB.json`, `referee_sessions.json`).

Plumbing, checked by the referee against the release files directly:
FAR at the frozen `frr5` cut over all 90 released cells = **0.774575** (matches
`release_cell_map.json`'s `far5_mean_release`); `tap__imu_only__paper_xgboost` =
**0.85425** against its decoy `summary.json` far of 0.3817. Accept rule verified:
ACCEPTED iff `score < frr5`, `score_direction = larger_is_more_fake` for all 90 cells.

---

## 1. The one number that matters most

> **Smallest rejection ratio r at which the knowledge-free attacker A2 drives session
> caught-rate below the genuine session false-alarm rate.**

**There is no such r for r ≤ 20. A2 never collapses the defence.**

The two agents already agreed on this. The referee confirms it and shows the margin is
not close. Touch regime, three-way-disjoint calibration, genuine session FRR **0.0442**:

| arm | r=2 | r=5 | r=10 | r=20 |
|---|---|---|---|---|
| A0 uniform (no adaptation) | 0.6594 | – | – | – |
| A2 centroid, A's 54-d features | 0.6285 | 0.6372 | 0.6518 | 0.6482 |
| A2 centroid, B's modality-matched features | 0.6465 | 0.6661 | 0.6846 | 0.6847 |
| A2 10-NN, A's features | 0.6308 | 0.6255 | **0.5956** | 0.5957 |
| A2 10-NN, B's features | 0.6550 | 0.6541 | 0.6437 | 0.6397 |

The lowest A2 caught-rate anywhere in the touch regime is **0.5956**, i.e. **13.5×** the
false-alarm rate it would have to fall below. Keystroke: lowest A2 value 0.2547 against a
false-alarm rate of 0.0709, **3.6×**. This holds under all three calibration
constructions, both feature specifications, both variants, both regimes, and at both
α = 0.05 and α = 0.01.

Per cell: **0 of 18** touch cells collapse under A2-centroid at any r under any
construction; A2-knn collapses 0–2 of 18 depending on feature set. In keystroke 2–6 of 18
"collapse", but **2–3 of those were already at or below chance under A0 with no
adaptation at all** — broken detectors, not a working attack.

r cannot be pushed past 20 on the published pool: at r=20 the filtered pool is 10 events
while one session needs up to 28 events of a single action, already forcing 16.1% (touch)
/ 26.1% (keystroke) within-session reuse. **r=10 is the largest clean ratio.**

---

## 2. Root cause of essentially every Part A disagreement

Both agents used the same two-fold user-disjoint calibration (fold = parity of position
in the sorted test-user list), the same 41-quantile score LLR, the same tail p-value
`(#null ≥ stat + 1)/(n+1)`, and the same alarm rule `p ≤ α`. They differ in **one thing**:
how the calibration null is coupled to the LLR table.

* **Agent A** — for fold *f*: fit the LLR on fold-*f* users, build the null from fold-*f*
  genuine **sessions scored with that same table**, evaluate fold-(1−*f*) sessions.
  The null is *in-sample* for the table; the evaluated sessions are *out-of-sample*.
  The null sits low, so held-out genuine sessions over-alarm.
* **Agent B** — every session is scored with the *other* fold's table; the null for a
  fold-*s* session is the fold-(1−*s*) sessions scored with the fold-*s* table. Both the
  null and the evaluated statistic are out-of-sample for their own table, so they are
  exchangeable.
* **Referee** — three user groups by index mod 3; for evaluation group *e* the table is
  fitted on group (*e*+1)%3 and the null comes from group (*e*+2)%3 scored with that same
  table. Fit / null / evaluation users pairwise disjoint. Unbiased reference.

Realised genuine session FRR at a **nominal 5%** target, touch:

| construction | realised FRR | A0 caught |
|---|---|---|
| Agent A (in-sample null) | **0.0740** | 0.7446 |
| Agent B (cross-fitted null) | **0.0501** | 0.6706 |
| Referee (three-way disjoint) | **0.0442** | 0.6594 |

**Agent B is right. Agent A's cut runs about 1.5× hotter than nominal.** A's caught-rates
are not wrong — they are correct numbers at an operating point that is not the 5% one A
calls it.

### The proof was visible before any recomputation

Every session statistic that does **not** depend on the fold-fitted LLR table agrees
between A and B to ~0.001, at *identical* realised FRRs. Every statistic that does depend
on it disagrees by 0.04–0.08.

| statistic | A caught / FRR | B caught / FRR | agrees? |
|---|---|---|---|
| B0 count-of-caught | 0.4199 / 0.0326 | 0.4211 / 0.0326 | ✅ |
| B1 mean session score | 0.6132 / 0.0465 | 0.6121 / 0.0465 | ✅ |
| B2 duration-only LLR (dev-fitted) | 0.1734 / 0.0382 | 0.1724 / 0.0382 | ✅ |
| llr_mean | 0.6669 / 0.0676 | 0.5838 / 0.0506 | ❌ |
| **llr_sum** | **0.7440 / 0.0740** | **0.6689 / 0.0501** | ❌ |

### Referee reproduces both agents from one codebase

Switching only the null construction:

| quantity | referee, A-style | agent A | referee, B-style | agent B |
|---|---|---|---|---|
| touch A0 caught | 0.7446 | 0.7440 | 0.6706 | 0.6689 |
| touch genuine FRR | 0.0740 | 0.0740 | 0.0501 | 0.0501 |
| A2 centroid r=10 | 0.7340 | 0.7339 | 0.6976 | 0.6978 |
| A1 same-modality r=10 | 0.3431 | 0.3429 | 0.2994 | 0.2992 |
| A1 ensemble-5 r=20 | 0.0890 | – | 0.0705 | 0.0708 |
| A3 oracle r=10 | 0.0015 | 0.0014 | 0.0010 | 0.0009 |

---

## 3. Side-by-side headline table, disagreements > 0.02 flagged

| quantity | A | B | Δ | cause | correct |
|---|---|---|---|---|---|
| touch A0 caught | 0.7440 @ FRR 0.0740 | 0.6689 @ FRR 0.0501 | **0.075** | null construction | **B** → 0.6594 @ 0.0442 |
| touch A2 centroid r=10 | 0.7339 | 0.6978 | **0.036** | null construction (+0.07) partly offset by A2 feature spec (−0.03) | both; 0.6518 (A feats) / 0.6846 (B feats) @ 0.0442 |
| touch A2 knn r=20 | 0.6613 | 0.6485 | 0.013 | same | – |
| touch A1 single same-modality r=10 (90 pairings) | 0.3429 | 0.2992 | **0.044** | null construction only; surrogate sets identical | **B** → 0.2908 |
| touch A1 cross-modality r=10 | 0.5119 (n=180) | 0.4201 (n=216) | **0.092** | **different surrogate definition**: A requires a different detector family, B allows any non-victim cell | **B**'s definition |
| touch A1 **ensemble of 5** | *arm absent* | 0.1028 (r=10) / 0.0708 (r=20) | – | design difference | **B**; referee 0.0981 / 0.0673 |
| keystroke A0 caught | 0.3243 @ 0.0867 (134 sess) | 0.2473 @ 0.0568 (133 sess) | **0.077** | null construction + session denominator | referee 0.2693 @ 0.0709 on 130 sessions |
| touch A3 oracle r=5 | 0.0102 | 0.0073 | 0.003 | null construction | either |
| **duration floor**, touch @5% | 0.1748 @ 0.0382 | 0.1758 @ 0.0382 | 0.001 | – | **agreed**, referee 0.1758 @ 0.0382 |
| duration after prescribed control | 0.1823 | 0.1949 | 0.013 | seeds | agreed: the control does nothing |
| **duration after quantile map** | 0.0689 @ 0.0446 | 0.0250 @ 0.0382 | **0.044** | **different transform** (see §5) | **B** |
| timing-detector duration share | "+0.004 borrowed, premise refuted" | "+0.144 excess, 80–100% duration" | **0.15** | **different detectors** (see §6) | both, for different questions |
| per-event FAR, touch A0 | 0.7622 | 0.7487 | 0.014 | action-unweighted vs slot-weighted mean | both; referee 0.7622 / 0.7478 |
| price: FRR for caught 50% | 0.048 | 0.0207 | – | A's 20-point α grid overshoots | **B** |

---

## 4. Sessions: neither agent's denominator is right

The two genuine session inventories disagree about session composition.

* The frozen session-assembler artefact (`sessions_genuine.jsonl`) is **missing 21 test
  sessions entirely**. A fell back to the scored-event composition for those and kept
  **4** sessions that the binding inventory shows as keystroke **+ swipe**.
* The test-split binding inventory covers every session but its per-session event list is
  a **subset** for some sessions. B therefore kept **3** sessions the artefact shows as
  mixed — e.g. `hmog_u014_s07`: binding says 15 keystroke, artefact says 35 keystroke + 2
  swipe.

| | A | B | referee (union rule) |
|---|---|---|---|
| touch sessions | 314 | 314 | **314** (identical set) |
| keystroke sessions | 134 | 133 | **130** |
| keystroke scored slots | 1638 | 1613 | **1577** |
| mixed sessions dropped | 26 | 27 | **30** |

**Conservative rule: drop a session if *either* inventory shows keystroke mixed with
touch.** The touch regime is unaffected, so every touch headline stands as measured.

Scored-slot fraction (F11) is likewise inventory-dependent: touch 6994 scored out of
39 473 (binding, 17.7%) or 43 383 (per-session maximum of the two, 16.1%). Report the
range: **between one slot in six and one in five carries a score**.

---

## 5. Quantile mapping: A's diagnosis, B's repair

Both agents' variants were reproduced by the referee (release bundles, touch, 5% cut):

| transform | referee caught / FRR | agent's own number |
|---|---|---|
| raw (no fix) | 0.1758 / 0.0382 | A 0.1748, B 0.1758 |
| reviewer's clip-at-cap + jitter control | 0.1903 / 0.0382 | A 0.1823, B 0.1949 |
| A variant 1: interpolating map, fake only | **0.5949** / 0.0414 | A 0.5838 |
| A variant 2: deterministic mid-rank map, fake only | **0.4360** / 0.0414 | A 0.4430 |
| A variant 3: dequantise **both arms** + interpolating map | 0.0691 / 0.0414 | A 0.0689 |
| **B: randomised mid-rank PIT, fake only, fixed detector** | **0.0226** / 0.0382 | B 0.0250 |
| **B: same, defender refits the LLR** | **0.0325** / 0.0446 | B 0.0363 |

**A's diagnosis is right and important.** Durations are massively tied: the 10
development users' genuine `tap` durations are 601 events on **17 distinct values**
(largest tie block 110); 99.8% of tap values sit in a tie, 94–97% for scroll/swipe/pinch.
A deterministic mid-rank map sends every tied value to one target and cannot reproduce the
marginal — hence 0.436. An interpolating map leaves the duration grid and manufactures a
brand-new tell — hence 0.595.

**B's repair is the right one.** A randomised tie-break inside the rank spreads tied
values across the corresponding target range while landing on *observed* genuine values,
so mapped fakes stay on the same grid as genuine (on-grid fraction 0.596 mapped vs 0.590
genuine). It drives the duration channel to **at or below its own false-alarm rate**, and
survives the defender refitting.

**A's variant 3 is not an attack.** It adds jitter to the **genuine** arm, which the
attacker cannot do; it hides the grid tell by destroying the grid on both sides. Its
residual excess (+0.028) is an artefact of that, not evidence that a residual leak
survives a real quantile map.

Verdict on the acceptance test: **quantile mapping works.** 0.1758 → 0.023 (fixed
detector) / 0.033 (refit detector).

---

## 6. The timing / duration-share question: two different detectors

This is not an arithmetic disagreement; it is a definitional one, established by reading
the code.

* **A's timing statistic reads only the onset sequence**: inter-onset interval
  `IOI[i] = duration[i−1] + gap[i]`. Duration enters only through its contribution to an
  interval whose variance is dominated by the human gap (gap p50 ≈ 7.5 s touch, 25 s
  keystroke, against a 10–70 ms duration shift). A measures a borrowed caught-rate of
  **+0.004** and concludes the confound does not survive being folded into onsets.
* **B's timing statistic is a per-event duration LLR + a per-event gap LLR + a
  session-span LLR** — it consumes duration **directly, as its own term**. It measures an
  excess of +0.144 to +0.151, of which 80–100% is duration.

Both are correct answers to different questions, and neither may be stated without its
construction. For the question actually asked — *can a session feature that touches
onsets silently import the carrier-padding artefact?* — **A is right and the answer is
essentially no.** B's number answers *can a detector that reads duration explicitly
import it*, which is near-tautological in exactly the way B labels its own pacing-replay
ablation, and should carry the same F14 label. B also notes its timing statistic is badly
miscalibrated across users (realised FRR 0.10 touch / 0.28 keystroke against a 0.05
target), so only its excesses are readable.

---

## 7. What both agents agree on, and it holds

* FAR@frr5 = **0.774575** over the 90 released cells; `tap__imu_only__paper_xgboost` =
  **0.85425**. Three independent reproductions.
* Touch: **314 sessions, 6994 scored slots**, identical set.
* Duration bulk statistics on the release bundles, identical to three decimals in A, B and
  the referee (tap gen 0.040/0.070/0.140/0.180 vs fake 0.040/0.080/0.130/0.180; scroll v15
  gen 0.110/0.440/1.770/3.350 vs fake 0.180/0.460/1.370/1.780; swipe gen
  0.119/0.380/1.451/3.080 vs fake 0.090/0.350/1.480/1.560; pinch gen 0.180/0.430/0.930/1.570
  vs fake 0.090/0.400/0.980/1.050; keystroke gen 0.600/4.200/20.483/65.800 vs fake
  0.580/4.400/21.617/52.970).
* **Only scroll/fake differs between `replay_dataset_full` and `replay_dataset_v15`**
  (5944 of 6000 events). tap, swipe, pinch and keystroke durations are byte-identical
  across all release bundles. The reviewer's scroll-cap story (fake max 1.68 s, median
  0.370 vs genuine 0.440, 5.6% on the cap) is a `replay_dataset_full` fact; **every
  released scroll cell uses v15**, where fake max is 1.78 s, 0.75% sit on the cap and the
  median shift **reverses** (fake 0.460 vs genuine 0.440). This is why the duration floor
  is 0.176 and not the reviewer's 0.389.
* The reviewer's prescribed clip+jitter control is **ineffective** (0.176 → 0.18–0.19),
  because only 0.1–4.3% of fakes sit on the cap while the whole body of the distribution
  is shifted.
* A3 (oracle on the victim's own score) collapses the defence — touch r=5, per-event FAR
  1.0000 at every r ≥ 5 — and is labelled an upper bound and never a headline by both
  agents (F5 satisfied).
* Pool shortfall: r ≤ 5 needs no reuse; r=10 forces 0.74% (touch) / 2.36% (keystroke) of
  slot draws to reuse; r=20 forces 16.14% / 26.10%. Agreed to four decimals.

---

## 8. Corrected headline

**Basis**: referee recomputation, three-way-disjoint calibration, `llr_sum`, 20 mirrored
draws, union-clean session set (314 touch / 130 keystroke), 18 cells. Agent B's
construction lands within 0.01–0.02 of these and carries the user-clustered intervals.

**Touch**, genuine session FRR **0.0442** at the 5% target:

| attacker | r=2 | r=5 | r=10 | r=20 |
|---|---|---|---|---|
| A0 uniform | 0.6594 (r=1) | | | |
| A2 knowledge-free (no PAD at all) | 0.6285–0.6550 | 0.6255–0.6661 | 0.5956–0.6846 | 0.5957–0.6847 |
| A1 one blindly-chosen same-modality surrogate | 0.4204 | 0.3149 | 0.2908 | 0.2757 |
| A1 **ensemble of 5 same-modality surrogates** | 0.3283 | 0.1520 | **0.0981** | **0.0673** |
| A3 ORACLE — *upper bound, not a threat model* | 0.1103 | 0.0063 | 0.0008 | 0.0003 |

Per-event FAR of the selected events rises the whole time: 0.748 (A0, slot-weighted) →
0.953 (ensemble r=10) → 1.000 (oracle r≥5). Duration-only baseline B2: 0.1758 at FRR
0.0382, falling to 0.0226 once the attacker quantile-maps its durations.

**Keystroke**, genuine session FRR 0.0709: A0 0.2693, A2 centroid r=20 0.2871, A1 single
r=20 0.2022, A1 ensemble-5 r=20 0.1479, A3 oracle r=20 0.0388. Three of 18 cells are
already at or below chance under A0. The 1% operating point does not exist in this regime
(65–67 calibration sessions per fold ⇒ smallest attainable tail probability ≈ 0.015), so
every 1% keystroke figure in both agents' outputs is 0.0000 by construction.

### Collapse points

| rule | collapse point (macro mean ≤ genuine FRR) |
|---|---|
| A2 centroid (free, no PAD) | **NEVER** for r ≤ 20, either regime, either α, either feature set, all three calibrations |
| A2 10-NN | **NEVER** at macro level; 0–2 of 18 touch cells, 3–6 of 18 keystroke cells (2–3 already broken under A0) |
| A1 one blind surrogate | **NEVER** at macro level (0.2757 vs 0.0442 at r=20, 6.2×); 29 of 90 pairings collapse individually |
| A1 ensemble of 5 | not a strict macro collapse at r ≤ 20 (0.0673 vs 0.0442), but **statistically indistinguishable from chance at r=20** (B's paired difference +0.0207, CI [−0.004, +0.045]); 13 of 18 touch cells collapse individually. Clean operating point r=10: 0.098 vs 0.044 |
| A3 ORACLE | touch r=5, keystroke r=10 — **not a threat model** |

**One line.** The knowledge-free attacker never collapses it. A single blindly-chosen
surrogate never collapses the aggregate but collapses about a third of individual
pairings. An ensemble of five surrogates — needing no guess about which one transfers —
takes it to the edge of chance at r=10–20. The oracle collapses it at r=5 and is not a
threat model. And whatever survives an adaptive attacker is the duration-only baseline,
whose power is carrier padding that a per-action randomised-rank quantile map removes
entirely.

---

## 9. Confidence and what was not done

**High** on Part A and the duration floor: the referee reproduced *both* agents' Part A
numbers to within 0.002 from one codebase by switching only the null construction, and all
four disputed quantile-map variants to within 0.011. The A2 verdict is invariant to
calibration, features, variant and regime, with a 13.5× margin in touch. The duration
floor agrees across three implementations to 0.001 at an identical realised FRR.

**Medium** on keystroke (the session inventory is genuinely ambiguous at 130/133/134, the
1% point does not exist, and 3 of 18 cells are already broken) and on the
timing-consequence question (arbitrated from the code, not re-run).

**Not done**: no confidence intervals were recomputed for the referee's own point
estimates (the agents' intervals, which agree that the A2 intervals are separated from the
false-alarm intervals by an enormous margin, are cited instead); the timing analysis was
not re-run; no Part C quantity was recomputed — the A-vs-B differences there follow
mechanically from the operating-point difference and from A's 6 versus B's 12
sessions/day, and neither cadence figure is quotable without that parameter.

No model was loaded, no GPU used, nothing under `/mnt/share` written, no git operation
performed.
