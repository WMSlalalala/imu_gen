#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read out/partA|B|C.json and emit out/SUMMARY.json + a readable text digest."""
import json, os, collections, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
A = json.load(open(f'{OUT}/partA.json'))
B = json.load(open(f'{OUT}/partB.json'))
C = json.load(open(f'{OUT}/partC.json'))
S = {}
L = []


def p(s=''):
    L.append(str(s))
    print(s)


REVIEWER_CELL = 'imu_only/behaveformer_stdat'
STAT = 'scoreLLR_sum'
Q = 'target_frr_0.05'
Q1 = 'target_frr_0.01'

by = {(b['combo'], b['regime']): b for b in A}
p('=== coverage ===')
for reg in ('touch', 'keystroke'):
    b = by[(REVIEWER_CELL, reg)]
    p('%s: %d sessions, %d users, mean len %.1f, scored slots %d'
      % (reg, b['n_sessions'], b['n_users'], b['mean_session_len'], b['total_scored_slots']))
S['coverage'] = {reg: dict(n_sessions=by[(REVIEWER_CELL, reg)]['n_sessions'],
                           n_users=by[(REVIEWER_CELL, reg)]['n_users'],
                           mean_len=by[(REVIEWER_CELL, reg)]['mean_session_len'],
                           scored_slots=by[(REVIEWER_CELL, reg)]['total_scored_slots'])
                 for reg in ('touch', 'keystroke')}


def cfgs(combo, regime):
    return by[(combo, regime)]['configs']


# ---------------------------------------------------------------- adaptive curve
def collect(regime, mode='SPLIT', stat=STAT, q=Q):
    """rows: (combo, rule, surrogate, r) -> dict of metrics"""
    rows = []
    for b in A:
        if b['regime'] != regime:
            continue
        for c in b['configs']:
            d = c[mode][stat][q]
            rows.append(dict(combo=b['combo'], modality=b['modality'], detector=b['detector'],
                             rule=c['rule'], sur=c['surrogate'], sur_det=c['surrogate_detector'],
                             sur_mod=c['surrogate_modality'], r=c['r'],
                             caught=d['caught'], lo=d['caught_ci'][0], hi=d['caught_ci'][1],
                             frr=d['realized_session_FRR'],
                             far_ev=c['per_event_FAR_submitted'],
                             pool_far=c['per_event_FAR_filtered_pool'],
                             relaxed=c['relaxed_groups'], ngroups=c['n_groups'],
                             eff_r=c['effective_r_mean'], eff_r_min=c['effective_r_min'],
                             price50=c[mode][stat]['price']['FAR_for_caught_50'],
                             price80=c[mode][stat]['price']['FAR_for_caught_80'],
                             price95=c[mode][stat]['price']['FAR_for_caught_95'],
                             roc5=c[mode][stat]['roc_reading']['caught_at_realized_FRR_0.05'],
                             roc1=c[mode][stat]['roc_reading']['caught_at_realized_FRR_0.01']))
    return rows


S['adaptive'] = {}
for regime in ('touch', 'keystroke'):
    rows = collect(regime)
    p('\n=== PART A adaptive attacker curve, regime=%s, %s, SPLIT calibration, target session FRR 5%% ==='
      % (regime, STAT))
    p('rule                         r   caught(median over 18 cells)  [IQR]        realizedFRR  per-event FAR  cells below FRR')
    tab = {}
    for rule in ('A0', 'A1', 'A2c', 'A2k', 'A2c_rev', 'A3'):
        for r in (1, 2, 5, 10, 20):
            sel = [x for x in rows if x['rule'] == rule and x['r'] == r]
            if not sel:
                continue
            if rule == 'A1':
                # per-cell mean over surrogates first, then median over cells
                percell = collections.defaultdict(list)
                for x in sel:
                    percell[x['combo']].append(x['caught'])
                cv = np.array([np.mean(v) for v in percell.values()])
                fv = np.array([np.mean([y['far_ev'] for y in sel if y['combo'] == k])
                               for k in percell])
                frr = np.array([np.mean([y['frr'] for y in sel if y['combo'] == k])
                                for k in percell])
                below = np.mean(cv < frr)
            else:
                cv = np.array([x['caught'] for x in sel])
                fv = np.array([x['far_ev'] for x in sel])
                frr = np.array([x['frr'] for x in sel])
                below = np.mean(cv < frr)
            tab[(rule, r)] = dict(caught_median=float(np.median(cv)),
                                  caught_q25=float(np.percentile(cv, 25)),
                                  caught_q75=float(np.percentile(cv, 75)),
                                  caught_min=float(cv.min()), caught_max=float(cv.max()),
                                  realizedFRR_median=float(np.median(frr)),
                                  per_event_FAR_median=float(np.median(fv)),
                                  frac_cells_caught_below_FRR=float(below),
                                  n_cells=len(cv))
            p('%-10s %-16s %2d  %.3f  [%.3f,%.3f]  min %.3f max %.3f   %.3f      %.3f        %.2f'
              % (rule, '', r, np.median(cv), np.percentile(cv, 25), np.percentile(cv, 75),
                 cv.min(), cv.max(), np.median(frr), np.median(fv), below))
    S['adaptive'][regime] = {'%s|r=%d' % k: v for k, v in tab.items()}

    # reviewer cell detail
    p('\n--- reviewer cell %s, regime=%s (SPLIT, target 5%% / 1%%) ---' % (REVIEWER_CELL, regime))
    det = {}
    for c in cfgs(REVIEWER_CELL, regime):
        k = '%s|%s|r=%d' % (c['rule'], c['surrogate'] or '-', c['r'])
        d5 = c['SPLIT'][STAT][Q]; d1 = c['SPLIT'][STAT][Q1]
        det[k] = dict(caught5=d5['caught'], ci5=d5['caught_ci'], frr5=d5['realized_session_FRR'],
                      caught1=d1['caught'], ci1=d1['caught_ci'], frr1=d1['realized_session_FRR'],
                      per_event_FAR=c['per_event_FAR_submitted'],
                      price50=c['SPLIT'][STAT]['price']['FAR_for_caught_50'],
                      price80=c['SPLIT'][STAT]['price']['FAR_for_caught_80'],
                      price95=c['SPLIT'][STAT]['price']['FAR_for_caught_95'],
                      roc5=c['SPLIT'][STAT]['roc_reading']['caught_at_realized_FRR_0.05'],
                      roc1=c['SPLIT'][STAT]['roc_reading']['caught_at_realized_FRR_0.01'])
        if c['rule'] in ('A0', 'A2c', 'A2k', 'A2c_rev', 'A3'):
            p('%-28s caught@5%%=%.3f [%.3f,%.3f] (realized FRR %.3f)  caught@1%%=%.3f  FARev=%.3f  price50=%s'
              % (k, d5['caught'], d5['caught_ci'][0], d5['caught_ci'][1],
                 d5['realized_session_FRR'], d1['caught'], c['per_event_FAR_submitted'],
                 ('%.3f' % c['SPLIT'][STAT]['price']['FAR_for_caught_50'])
                 if c['SPLIT'][STAT]['price']['FAR_for_caught_50'] is not None else 'never'))
    S.setdefault('reviewer_cell', {})[regime] = det

    # ---- collapse point
    p('\n--- collapse point (smallest r with caught < realized genuine session FRR), regime=%s ---' % regime)
    cp = {}
    for rule in ('A1', 'A2c', 'A2k', 'A2c_rev', 'A3'):
        percell = {}
        for b in A:
            if b['regime'] != regime:
                continue
            best = None
            for r in (2, 5, 10, 20):
                sel = [c for c in b['configs'] if c['rule'] == rule and c['r'] == r]
                if not sel:
                    continue
                if rule == 'A1':
                    ct = np.mean([c['SPLIT'][STAT][Q]['caught'] for c in sel])
                    fr = np.mean([c['SPLIT'][STAT][Q]['realized_session_FRR'] for c in sel])
                else:
                    ct = sel[0]['SPLIT'][STAT][Q]['caught']
                    fr = sel[0]['SPLIT'][STAT][Q]['realized_session_FRR']
                if ct < fr and best is None:
                    best = r
            percell[b['combo']] = best
        got = [v for v in percell.values() if v is not None]
        cp[rule] = dict(per_cell=percell, n_cells_collapsing=len(got), n_cells=len(percell),
                        median_r=(float(np.median(got)) if got else None))
        p('%-8s collapses in %2d/%2d cells; median collapse r = %s'
          % (rule, len(got), len(percell), cp[rule]['median_r']))
    S.setdefault('collapse_point', {})[regime] = cp

    # ---- surrogate spread
    p('\n--- A1 surrogate spread at r=10, regime=%s (caught, median over 18 victim cells) ---' % regime)
    spread = {}
    surs = sorted(set(x['sur'] for x in rows if x['rule'] == 'A1'))
    for s_ in surs:
        sel = [x for x in rows if x['rule'] == 'A1' and x['r'] == 10 and x['sur'] == s_]
        cv = np.array([x['caught'] for x in sel])
        spread[s_] = dict(caught_median=float(np.median(cv)), caught_min=float(cv.min()),
                          caught_max=float(cv.max()), n_victims=len(cv))
    for s_ in sorted(spread, key=lambda k: spread[k]['caught_median']):
        p('  %-36s caught median %.3f  [%.3f, %.3f]  over %d victims'
          % (s_, spread[s_]['caught_median'], spread[s_]['caught_min'],
             spread[s_]['caught_max'], spread[s_]['n_victims']))
    S.setdefault('surrogate_spread', {})[regime] = spread
    # same-modality vs cross-modality
    for tag, f in (('same_modality_surrogate', lambda x: x['sur_mod'] == x['combo'].split('/')[0]),
                   ('cross_modality_surrogate', lambda x: x['sur_mod'] != x['combo'].split('/')[0])):
        sel = [x for x in rows if x['rule'] == 'A1' and x['r'] == 10 and f(x)]
        cv = np.array([x['caught'] for x in sel])
        p('  %s: caught median %.3f (n=%d pairings)' % (tag, np.median(cv), len(cv)))
        S['surrogate_spread'][regime][tag] = dict(caught_median=float(np.median(cv)), n=len(cv))

# ---------------------------------------------------------------- cost / shortfall
p('\n=== cost & pool-sufficiency accounting (F15) ===')
cost = {}
for regime in ('touch', 'keystroke'):
    b = by[(REVIEWER_CELL, regime)]
    for c in b['configs']:
        if c['rule'] != 'A2c':
            continue
        cost['%s|r=%d' % (regime, c['r'])] = dict(
            K_available=c['K_available'], groups=c['n_groups'],
            relaxed_groups=c['relaxed_groups'],
            frac_relaxed=c['relaxed_groups'] / float(c['n_groups']),
            effective_r_mean=c['effective_r_mean'], effective_r_min=c['effective_r_min'],
            generated_per_submitted=c['generated_per_submitted'],
            n_submitted_events=c['n_submitted_events'])
        p('%-10s r=%2d  K=%3d  shortfall groups %3d/%3d (%.1f%%)  effective r mean %.1f min %.1f'
          % (regime, c['r'], c['K_available'], c['relaxed_groups'], c['n_groups'],
             100.0 * c['relaxed_groups'] / c['n_groups'], c['effective_r_mean'],
             c['effective_r_min']))
S['cost'] = cost

# ---------------------------------------------------------------- baselines
p('\n=== BASELINES (F10) at target session FRR 5%%, SPLIT, A0 uniform and A2c r=10 ===')
bl = {}
for regime in ('touch', 'keystroke'):
    for rule, r in (('A0', 1), ('A2c', 10), ('A3', 10)):
        for stn in ('scoreLLR_sum', 'scoreLLR_mean', 'B0_count', 'B1_meanscore', 'B1z_meanscore_z'):
            cv, fv = [], []
            for b in A:
                if b['regime'] != regime:
                    continue
                c = [x for x in b['configs'] if x['rule'] == rule and x['r'] == r][0]
                cv.append(c['SPLIT'][stn][Q]['caught'])
                fv.append(c['SPLIT'][stn][Q]['realized_session_FRR'])
            bl['%s|%s r=%d|%s' % (regime, rule, r, stn)] = dict(
                caught_median=float(np.median(cv)), caught_min=float(np.min(cv)),
                caught_max=float(np.max(cv)), realizedFRR_median=float(np.median(fv)))
            p('%-10s %-8s r=%-2d %-20s caught median %.3f  (min %.3f max %.3f)  realizedFRR %.3f'
              % (regime, rule, r, stn, np.median(cv), np.min(cv), np.max(cv), np.median(fv)))
# B2 duration-only
for regime in ('touch', 'keystroke'):
    d = B['duration_only_session_LLR'][regime]['raw']['SPLIT']
    bl['%s|B2_duration_only' % regime] = dict(
        caught5=d[Q]['caught'], ci5=d[Q]['caught_ci'], frr5=d[Q]['realized_session_FRR'],
        caught1=d[Q1]['caught'], ci1=d[Q1]['caught_ci'], frr1=d[Q1]['realized_session_FRR'])
    p('%-10s B2_duration_only        caught@target5%% %.3f [%.3f,%.3f] (realized FRR %.3f)  caught@1%% %.3f'
      % (regime, d[Q]['caught'], d[Q]['caught_ci'][0], d[Q]['caught_ci'][1],
         d[Q]['realized_session_FRR'], d[Q1]['caught']))
S['baselines'] = bl

# ---------------------------------------------------------------- part B
p('\n=== PART B: duration-only session LLR ===')
pb = {}
for regime in ('touch', 'keystroke'):
    for mname, mm in B['duration_only_session_LLR'][regime].items():
        for mode in ('SPLIT', 'ORACLE'):
            d = mm[mode]
            pb['%s|%s|%s' % (regime, mname, mode)] = dict(
                caught5=d[Q]['caught'], ci5=d[Q]['caught_ci'],
                frr5=d[Q]['realized_session_FRR'],
                caught1=d[Q1]['caught'], ci1=d[Q1]['caught_ci'],
                frr1=d[Q1]['realized_session_FRR'],
                roc_caught_at_realized_5=d['roc_reading']['caught_at_realized_FRR_0.05'],
                roc_caught_at_realized_1=d['roc_reading']['caught_at_realized_FRR_0.01'],
                price50=d['price']['FAR_for_caught_50'], price80=d['price']['FAR_for_caught_80'],
                price95=d['price']['FAR_for_caught_95'])
            p('%-10s %-42s %-7s caught@5%%=%.3f [%.3f,%.3f] realizedFRR=%.3f | caught@1%%=%.3f | ROC caught@realFRR5%%=%s'
              % (regime, mname, mode, d[Q]['caught'], d[Q]['caught_ci'][0], d[Q]['caught_ci'][1],
                 d[Q]['realized_session_FRR'], d[Q1]['caught'],
                 ('%.3f' % d['roc_reading']['caught_at_realized_FRR_0.05'])
                 if d['roc_reading']['caught_at_realized_FRR_0.05'] is not None else 'na'))
S['partB_duration'] = pb

p('\n--- per-action duration bulk statistics (TEST split) ---')
bs = {}
for k, v in B['bulk_duration_stats'].items():
    if not k.endswith('/test'):
        continue
    a = k.split('/')[0]
    bs[a] = v
    p('%-10s genuine p5/p50/p95/max = %.3f/%.3f/%.3f/%.3f   fake = %.3f/%.3f/%.3f/%.3f   '
      'cap=%.3f  frac fake on cap=%.4f  frac genuine above cap=%.4f  median shift=%+.3f s'
      % (a, v['genuine']['p5'], v['genuine']['p50'], v['genuine']['p95'], v['genuine']['max'],
         v['fake']['p5'], v['fake']['p50'], v['fake']['p95'], v['fake']['max'],
         v['fake_cap'], v['frac_fake_exactly_on_cap'], v['frac_genuine_above_fake_cap'],
         v['median_shift_s']))
S['duration_bulk_test'] = bs

p('\n--- timing-feature consequence analysis ---')
tc = {}
for regime in ('touch', 'keystroke'):
    if regime not in B['timing_consequence']:
        continue
    blk = B['timing_consequence'][regime]
    for name in ('full_timing', 'ablated_duration'):
        d = blk[name]['SPLIT']
        tc['%s|%s' % (regime, name)] = dict(caught5=d[Q]['caught'], ci5=d[Q]['caught_ci'],
                                            frr5=d[Q]['realized_session_FRR'],
                                            caught1=d[Q1]['caught'],
                                            frr1=d[Q1]['realized_session_FRR'])
        p('%-10s %-18s caught@target5%%=%.3f [%.3f,%.3f] realizedFRR=%.3f  caught@1%%=%.3f'
          % (regime, name, d[Q]['caught'], d[Q]['caught_ci'][0], d[Q]['caught_ci'][1],
             d[Q]['realized_session_FRR'], d[Q1]['caught']))
    f = tc['%s|full_timing' % regime]['caught5']; ab = tc['%s|ablated_duration' % regime]['caught5']
    fr = tc['%s|full_timing' % regime]['frr5']
    share = (f - ab) / max(f - fr, 1e-9)
    tc['%s|duration_share_of_timing_caught' % regime] = float(share)
    p('  -> duration share of the timing-feature session result: (%.3f - %.3f)/(%.3f - %.3f) = %.3f'
      % (f, ab, f, fr, share))
S['timing_consequence'] = tc
S['timing_gap_process'] = B['timing_consequence']['gap_process']

# ---------------------------------------------------------------- part C
p('\n=== PART C: who pays the false alarms ===')
pts = C['points']
sc = [x for x in pts if x['tag'].startswith('A|') and '|scoreLLR_sum|SPLIT|target_frr_0.05' in x['tag']]
arr_med = np.array([x['median'] for x in sc]); arr_p90 = np.array([x['p90'] for x in sc])
arr_max = np.array([x['max'] for x in sc]); arr_min = np.array([x['min'] for x in sc])
arr_pool = np.array([x['pooled'] for x in sc])
p('score-LLR-sum, SPLIT, target 5%%: over %d (cell,regime) points -- pooled FRR median %.3f; '
  'per-user min median %.4f, median median %.3f, p90 median %.3f, max median %.3f'
  % (len(sc), np.median(arr_pool), np.median(arr_min), np.median(arr_med),
     np.median(arr_p90), np.median(arr_max)))
ratios = [x['ratio_max_over_min'] for x in sc if x['ratio_max_over_min']]
p('  max/min per-user FRR ratio: median %.1f, max %.1f (n=%d finite)'
  % (np.median(ratios), np.max(ratios), len(ratios)))
bbp90 = [x['beta_binomial']['p90'] for x in sc if 'p90' in x.get('beta_binomial', {})]
p('  fitted beta-binomial p90 of per-user FRR: median %.3f' % np.median(bbp90))
p('  lockouts/day (sessions/day=%g): median user %.3f, p90 user %.3f'
  % (C['sessions_per_day'], np.median([x['lockouts_per_day_median_user'] for x in sc]),
     np.median([x['lockouts_per_day_p90_user'] for x in sc])))
S['partC_scoreLLR'] = dict(
    n_points=len(sc), pooled_FRR_median=float(np.median(arr_pool)),
    per_user_min_median=float(np.median(arr_min)), per_user_median_median=float(np.median(arr_med)),
    per_user_p90_median=float(np.median(arr_p90)), per_user_max_median=float(np.median(arr_max)),
    ratio_median=float(np.median(ratios)), ratio_max=float(np.max(ratios)),
    betabinom_p90_median=float(np.median(bbp90)),
    lockouts_per_day_median_user=float(np.median([x['lockouts_per_day_median_user'] for x in sc])),
    lockouts_per_day_p90_user=float(np.median([x['lockouts_per_day_p90_user'] for x in sc])),
    sessions_per_day=C['sessions_per_day'])

ut = C['user_tail_operating_point']
p('\n--- alternative operating point: cut s.t. >=90%% of calibration users have session FRR <= 1%% ---')
for regime in ('touch', 'keystroke'):
    sel = [x for x in ut if x['regime'] == regime]
    a5 = np.array([x['pooled_5pct']['caught'] for x in sel])
    a5f = np.array([x['pooled_5pct']['pooled_FRR'] for x in sel])
    at = np.array([x['user_tail_90pct_at_1pct']['caught'] for x in sel])
    atf = np.array([x['user_tail_90pct_at_1pct']['pooled_FRR'] for x in sel])
    frac = np.array([x['pooled_5pct']['frac_users_under_1pct'] for x in sel])
    fract = np.array([x['user_tail_90pct_at_1pct']['frac_users_under_1pct'] for x in sel])
    p('%-10s pooled-5%% cut: caught %.3f (pooled FRR %.3f, %.0f%% of eval users under 1%%) | '
      'user-tail cut: caught %.3f (pooled FRR %.3f, %.0f%% under 1%%) -> caught cost %.3f'
      % (regime, np.median(a5), np.median(a5f), 100 * np.median(frac),
         np.median(at), np.median(atf), 100 * np.median(fract), np.median(a5) - np.median(at)))
    S.setdefault('user_tail', {})[regime] = dict(
        pooled5_caught_median=float(np.median(a5)), pooled5_FRR_median=float(np.median(a5f)),
        pooled5_frac_users_under_1pct_median=float(np.median(frac)),
        tail_caught_median=float(np.median(at)), tail_FRR_median=float(np.median(atf)),
        tail_frac_users_under_1pct_median=float(np.median(fract)),
        caught_cost_median=float(np.median(a5) - np.median(at)),
        lockouts_per_day_median_user_pooled5=float(np.median(
            [x['pooled_5pct']['per_user_median'] for x in sel]) * C['sessions_per_day']),
        lockouts_per_day_p90_user_pooled5=float(np.median(
            [x['pooled_5pct']['per_user_p90'] for x in sel]) * C['sessions_per_day']),
        lockouts_per_day_median_user_tail=float(np.median(
            [x['user_tail_90pct_at_1pct']['per_user_median'] for x in sel]) * C['sessions_per_day']),
        lockouts_per_day_p90_user_tail=float(np.median(
            [x['user_tail_90pct_at_1pct']['per_user_p90'] for x in sel]) * C['sessions_per_day']))

# ---------------------------------------------------------------- split vs oracle gap
p('\n=== calibration optimism: SPLIT vs ORACLE (F2) ===')
for regime in ('touch', 'keystroke'):
    s_, o_, sf, of = [], [], [], []
    for b in A:
        if b['regime'] != regime:
            continue
        c = [x for x in b['configs'] if x['rule'] == 'A0'][0]
        s_.append(c['SPLIT'][STAT][Q]['caught']); sf.append(c['SPLIT'][STAT][Q]['realized_session_FRR'])
        o_.append(c['ORACLE'][STAT][Q]['caught']); of.append(c['ORACLE'][STAT][Q]['realized_session_FRR'])
    p('%-10s A0: SPLIT caught %.3f @ realized FRR %.3f  |  ORACLE caught %.3f @ realized FRR %.3f'
      % (regime, np.median(s_), np.median(sf), np.median(o_), np.median(of)))
    S.setdefault('split_vs_oracle', {})[regime] = dict(
        split_caught=float(np.median(s_)), split_FRR=float(np.median(sf)),
        oracle_caught=float(np.median(o_)), oracle_FRR=float(np.median(of)))

json.dump(S, open(f'{OUT}/SUMMARY.json', 'w'), indent=1)
open(f'{OUT}/SUMMARY.txt', 'w').write('\n'.join(L))
print('\nwrote SUMMARY.json / SUMMARY.txt')
