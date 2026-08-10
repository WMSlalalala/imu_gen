"""Merge the half-attack SCORING result into the pre-registered receipt.

The build-phase receipt (committed at 76657ed, before any arm was scored) is
read and extended.  `preregistered_outcomes` is carried through BYTE FOR BYTE:
fairness rule F13 is only satisfied if the interpretation that is in the file
now is the one that was in the file before the numbers existed.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

SC = Path('/tmp/claude-473016/-home-mwang49-new-data7-data7-final-monitor-metrics-v1'
          '/e1b42475-b309-42ae-b7f3-314c50fb68d8/scratchpad/halfattack')
DEST = Path('/mnt/share/mwang49/real-human/imu_gen/final/evaluation/final_check')
OUT = DEST / 'scores' / 'half_attack_joint.json'
ACTIONS = ['tap', 'scroll', 'swipe', 'pinch', 'keystroke']
CLASSICAL = ['hmog_style_svm', 'hmog_style_rf', 'paper_svm', 'paper_xgboost']
DEEP = ['behaveformer_stdat', 'authconformer']
WINDOW = {'tap': 16, 'scroll': 208, 'swipe': 176, 'pinch': 100, 'keystroke': 512}


def fmt(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def arm_block(entry):
    return {'far_at_frr5': fmt(entry['far_at_frr5']),
            'ci95_user_clustered': [fmt(v) for v in entry['ci95']],
            'bootstrap_sd': fmt(entry['boot_sd'], 5)}


TAUT = ('NEAR-TAUTOLOGICAL (F14). For this action the released fake touch IS the '
        'victim own held recording transported to the carrier endpoints -- '
        'rebuilding it from the five reproduces the released trajectory to '
        'float32 ulp -- so row 4 is the release arm re-derived with a re-drawn '
        'shot and a null here is guaranteed by construction, not measured.')
NO_S2 = ('NOT A VALID COMPARISON. For this action the release second stage cannot '
         'be reconstructed (pinch needs live two-pointer endpoint geometry a fake '
         'donor does not carry; keystroke touch is composed from key anchors, not '
         'transported), so S2 is the identity in rows 3/4/5 and those rows carry an '
         'uncontrolled screen-position treatment relative to rows 1/2. Rows 3 and 4 '
         'for this action are read ONLY against row 5.')
MIXED = ('MIXED VALIDITY. This aggregate spans pinch and keystroke, whose rows 3/4 '
         'are not entitled to a comparison against rows 1/2 (S2 is the identity '
         'there). Use the tap_scroll_swipe_12_cells group for any row1/row2 '
         'comparison.')
CONF = ('DURATION-CONFOUNDED. The build phase measured that these two arms do not '
        'share a T/duration distribution; read the duration-standardised table '
        'alongside.')
CONFOUNDED_PAIRS = {'row3_minus_row2', 'row4_minus_row5', 'row5_minus_row2',
                    'row2_minus_row1', 'row3_minus_row1'}
MATCHED_PAIRS = {'row3_minus_row5': 'MATCHED event for event on T and duration '
                                    '(rows 3 and 5 share the carrier).',
                 'row4_minus_row2': 'MATCHED as multisets on T and duration '
                                    '(row 4 carriers are the release fake pool).'}


def labels_for(group, pair):
    out = []
    action = group.split(':')[1] if group.startswith('action:') else None
    touches_12 = pair.endswith('_minus_row2') or pair.endswith('_minus_row1')
    if touches_12:
        if action in ('pinch', 'keystroke'):
            out.append(NO_S2)
        elif group in ('all_20_cells',) or group.startswith('detector:'):
            out.append(MIXED)
    if pair == 'row4_minus_row2' and action in ('tap', 'scroll'):
        out.append(TAUT)
    if pair == 'row4_minus_row2' and (group in ('all_20_cells',
                                                'tap_scroll_swipe_12_cells')
                                      or group.startswith('detector:')):
        out.append('Contains tap and scroll, where this difference is '
                   'near-tautological (F14).')
    if pair in CONFOUNDED_PAIRS:
        out.append(CONF)
    if pair in MATCHED_PAIRS:
        out.append(MATCHED_PAIRS[pair])
    return out


def diff_block(d, group=None, pair=None):
    b = {'difference': fmt(d['difference']),
         'ci95_on_the_difference': [fmt(v) for v in d['ci95']],
         'excludes_zero': d['excludes_zero']}
    if group is not None:
        lab = labels_for(group, pair)
        if lab:
            b['labels'] = lab
    return b


def classify(D, per_action_D):
    """Mechanical read of the pre-registered decision rules, plus the check that
    the branch's stated CONCLUSION is not contradicted by the same table."""
    d = D['all_20_cells']
    below = lambda x: x['difference'] < 0 and x['ci95'][1] < 0
    above = lambda x: x['difference'] > 0 and x['ci95'][0] > 0
    indist = lambda x: x['ci95'][0] <= 0 <= x['ci95'][1]
    v = {
        'row3_below_row5': bool(below(d['row3_minus_row5'])),
        'row4_below_row5': bool(below(d['row4_minus_row5'])),
        'row3_indistinguishable_from_row5': bool(indist(d['row3_minus_row5'])),
        'row4_indistinguishable_from_row5': bool(indist(d['row4_minus_row5'])),
        'row5_close_to_row2': bool(indist(d['row5_minus_row2'])),
        'row5_ACCEPTED_MORE_than_row2': bool(above(d['row5_minus_row2'])),
        'row3_ACCEPTED_MORE_than_row2': bool(above(d['row3_minus_row2'])),
        'row4_ACCEPTED_MORE_than_row2': bool(above(d['row4_minus_row2'])),
    }
    v['row3_below_row5_per_action'] = {
        a: bool(below(per_action_D[f'action:{a}']['row3_minus_row5']))
        for a in ACTIONS}
    v['row4_below_row5_per_action'] = {
        a: bool(below(per_action_D[f'action:{a}']['row4_minus_row5']))
        for a in ACTIONS}
    if v['row3_below_row5'] and v['row4_below_row5']:
        v['branch_literally_matched'] = 'claim_holds'
    elif v['row3_indistinguishable_from_row5'] and \
            v['row4_indistinguishable_from_row5']:
        v['branch_literally_matched'] = 'claim_fails'
    else:
        v['branch_literally_matched'] = 'inconclusive_or_mixed'

    contradicted = (v['branch_literally_matched'] == 'claim_holds'
                    and (v['row3_ACCEPTED_MORE_than_row2']
                         or v['row4_ACCEPTED_MORE_than_row2']))
    v['branch_conclusion_contradicted_by_the_same_table'] = bool(contradicted)
    v['branch_reported'] = ('inconclusive -- pattern not enumerated by the '
                            'pre-registration' if contradicted
                            else v['branch_literally_matched'])
    v['why'] = (
        'The literal antecedent of the claim_holds branch (rows 3 and 4 both '
        'markedly below row 5, paired intervals excluding 0) IS satisfied at the '
        '20-cell aggregate. Its stated conclusion -- "half attacks are caught on '
        'their merits, and synthesising both channels is necessary" -- is '
        'contradicted by the same table: both half attacks are accepted AT LEAST '
        'AS OFTEN as the release full dual-channel attack (row3 - row2 and '
        'row4 - row2 are both positive with intervals excluding 0). The '
        'cross_pairing_undetectable branch does not apply either, because row 5 '
        'is not close to row 2: it is markedly ABOVE it, in the opposite '
        'direction to the one that branch anticipated. The observed pattern is '
        'therefore the "anything else" case and is labelled inconclusive for the '
        'intended claim, while the substantive reading -- rows 3 and 4 measured '
        'against the release, row 2 -- REFUTES it.'
    ) if contradicted else 'branch antecedent and conclusion are consistent.'
    return v


def q(x):
    return f"{float(x):.3f}"


def plain_statement(S, D, verdict):
    """A readable rendering of the numbers, composed FROM the numbers."""
    g20, g12 = S['all_20_cells'], S['tap_scroll_swipe_12_cells']
    d20 = D['all_20_cells']
    parts = [
        "Over the 20 classical joint cells, mean of the per-cell FAR at the "
        f"development-selected FRR=5% cut: row 2 (the release, both channels "
        f"synthesised) {q(g20['row2']['far_at_frr5'])}, row 3 (fake touch + real "
        f"five-shot IMU) {q(g20['row3']['far_at_frr5'])}, row 4 (real five-shot "
        f"touch + fake IMU) {q(g20['row4']['far_at_frr5'])}, row 5 (the "
        f"same-mechanics genuine-only mismatch control) "
        f"{q(g20['row5']['far_at_frr5'])}. Row 1, the genuine arm, is accepted at "
        f"{q(g20['row1']['far_at_frr5'])} (that is 1 - FRR on test, not a FAR).",
        "The comparison the experiment turns on, row 3 minus row 5, is "
        f"{q(d20['row3_minus_row5']['difference'])} with a user-clustered 95% "
        f"interval of [{q(d20['row3_minus_row5']['ci95'][0])}, "
        f"{q(d20['row3_minus_row5']['ci95'][1])}]; row 4 minus row 5 is "
        f"{q(d20['row4_minus_row5']['difference'])} "
        f"[{q(d20['row4_minus_row5']['ci95'][0])}, "
        f"{q(d20['row4_minus_row5']['ci95'][1])}]; row 5 minus row 2 is "
        f"{q(d20['row5_minus_row2']['difference'])} "
        f"[{q(d20['row5_minus_row2']['ci95'][0])}, "
        f"{q(d20['row5_minus_row2']['ci95'][1])}].",
        "Restricted to the 12 tap/scroll/swipe cells, the only ones whose rows "
        "3/4/5 carry the release's own second-stage placement and are therefore "
        f"entitled to a comparison against rows 1/2: row 2 "
        f"{q(g12['row2']['far_at_frr5'])}, row 3 {q(g12['row3']['far_at_frr5'])}, "
        f"row 4 {q(g12['row4']['far_at_frr5'])}, row 5 "
        f"{q(g12['row5']['far_at_frr5'])}.",
        "Ordering of the arms, most accepted first: row 1 genuine "
        f"{q(g20['row1']['far_at_frr5'])} > row 5 mismatch control "
        f"{q(g20['row5']['far_at_frr5'])} > row 3 "
        f"{q(g20['row3']['far_at_frr5'])} > row 4 "
        f"{q(g20['row4']['far_at_frr5'])} > row 2 the release "
        f"{q(g20['row2']['far_at_frr5'])}. The mismatch control is the most "
        "accepted arm of the four under test. Cross-pairing two genuine "
        "recordings of the same victim is not caught -- it is accepted MORE "
        "often than the release's own fully synthesised event.",
        "Consequence for the intended claim: the joint detector does NOT catch a "
        "half attack. Row 3 (fake touch, real five-shot inertial window) is "
        f"accepted {q(D['all_20_cells']['row3_minus_row2']['difference'])} MORE "
        "often than the release arm that synthesises both channels, and row 4 "
        f"(real five-shot touch, fake inertial window) "
        f"{q(D['all_20_cells']['row4_minus_row2']['difference'])} more often. "
        "Synthesising both channels is not shown to be necessary by this "
        "experiment; on these 20 cells it is shown to be counter-productive for "
        "row 3.",
        "What row 5 settles for the standing audit item: the joint modality's "
        "comparative weakness for the attack is NOT a cross-channel content "
        "consistency check. A detector that read content consistency would "
        "reject a mismatched pair of two genuine recordings; these cells accept "
        f"it at {q(g20['row5']['far_at_frr5'])}, above the release and only "
        f"{q(g20['row1']['far_at_frr5'] - g20['row5']['far_at_frr5'])} below the "
        "genuine arm.",
        "Pre-registered branch whose literal antecedent is matched: "
        f"{verdict['branch_literally_matched']}. Branch reported after checking "
        f"the branch's own conclusion against the same table: "
        f"{verdict['branch_reported']}.",
    ]
    return parts


def main():
    doc = json.loads(OUT.read_text())
    res = json.loads((SC / 'score_result.json').read_text())
    prereg = doc['preregistered_outcomes']          # carried through verbatim

    S, D = res['summary'], res['paired_differences']
    v20 = classify(D, D)

    per_action = {a: {'far_by_arm': {k: arm_block(S[f'action:{a}'][k])
                                     for k in S[f'action:{a}']},
                      'paired_differences': {k: diff_block(v, f'action:{a}', k)
                                             for k, v in D[f'action:{a}'].items()}}
                  for a in ACTIONS}
    per_detector = {d: {'far_by_arm': {k: arm_block(S[f'detector:{d}'][k])
                                       for k in S[f'detector:{d}']},
                        'paired_differences': {
                            k: diff_block(v, f'detector:{d}', k)
                            for k, v in D[f'detector:{d}'].items()}}
                    for d in CLASSICAL}

    per_cell = {}
    for name, e in res['per_cell'].items():
        per_cell[name] = {
            'threshold_frr5': e['threshold_frr5'],
            **{arm: {'far_at_frr5': fmt(e[arm]['far_at_frr5']),
                     'ci95': [fmt(x) for x in e[arm]['ci95']],
                     'events': e[arm]['events'],
                     'mean_score': fmt(e[arm]['mean_score'], 5)}
               for arm in ('row1', 'row2', 'row3', 'row4', 'row5', 'row5_unmatched')},
            'paired_differences': {
                k: diff_block(v, f"action:{name.split('__')[0]}", k)
                for k, v in e['paired_differences'].items()},
        }

    spread = {}
    for g in ('all_20_cells', 'tap_scroll_swipe_12_cells',
              *[f'action:{a}' for a in ACTIONS]):
        spread[g] = {arm: {'sd_over_R_repetitions': fmt(s['sd'], 5),
                           'range': fmt(s['range'], 5),
                           'mean': fmt(s['mean']),
                           'per_repetition_far': [fmt(x) for x in
                                                  s['per_repetition_far']]}
                     for arm, s in res['construction_spread_over_repetitions'][g].items()}

    per_action_events = 20 * 200 * 4 * res['repetitions']
    volume = sum(per_action_events * WINDOW[a] * 15 for a in ACTIONS)
    deep = {
        'cells_deferred': [f'{a}__imu_trajectory_xytime__{d}'
                           for a in ACTIONS for d in DEEP],
        'arms_covered': ['row3', 'row4', 'row5', 'row5_unmatched'],
        'events_built_per_action': per_action_events,
        'events_total': per_action_events * len(ACTIONS),
        'forward_passes': per_action_events * len(ACTIONS) * len(DEEP),
        'padded_input_elements_float32': volume * len(DEEP),
        'padded_input_bytes_float32': volume * len(DEEP) * 4,
        'per_action_window_samples': WINDOW,
        'keystroke_share_of_padded_volume': round(
            per_action_events * WINDOW['keystroke'] * 15 / volume, 4),
        'classical_cost_measured_here': {
            'wall_clock_seconds_for_the_same_events_on_4_cpu_workers':
                round(res['build_and_score_seconds'], 1),
            'events_built_and_scored': int(sum(
                res['events_per_arm'][a] for a in
                ('row3', 'row4', 'row5', 'row5_unmatched'))),
            'note': 'classical only; the deep cells re-use the SAME built events, '
                    'so only the scoring half of this cost would be replaced by '
                    'GPU forward passes.'},
        'wall_clock': 'NOT PROJECTED -- a GPU throughput benchmark is required and '
                      'this session is under a no-GPU constraint. The counts above '
                      'are computed; any hours figure would be invented.',
    }

    doc['status'] = 'COMPLETE -- three new arms scored on the 20 classical joint cells'
    doc['written_before_any_arm_was_scored_on_the_grid'] = False
    doc['preregistered_outcomes'] = prereg
    doc['scope']['deep_cells_gpu_deferred'] = deep
    doc['results'] = {
        'metric': 'FAR at the development-selected FRR=5 percent cut (the frr5 '
                  'scalar of each cell thresholds.json). Accepted iff score < '
                  'threshold; score_direction is larger_is_more_fake in all 20 '
                  'cells and is asserted at load time.',
        'aggregation': 'a group number is the MEAN OVER CELLS of the per-cell FAR, '
                       'which is the aggregation the published main table uses '
                       '(reproduced: mean over the 90 released cells = 0.774575, '
                       'joint modality 30 cells = 0.711267, trajectory 0.777475, '
                       'inertial 0.834983).',
        'pipeline_validation': {
            'what': 'the release own published test_scores re-scored with the same '
                    'extract_event_features + classical_scores path used for every '
                    'new arm, on ALL 20 cells (the spec asks for at least 3)',
            'cells_checked': len(res['reproduce']),
            'worst_max_abs_diff': res['reproduce_worst_max_abs_diff'],
            'exact_bitwise_fraction_min': min(r['exact_bitwise_fraction']
                                              for r in res['reproduce']),
            'published_events_not_found_in_shards': sum(
                r['published_not_found_in_shards'] for r in res['reproduce']),
            'shard_events_not_in_published': sum(
                r['shard_events_not_in_published'] for r in res['reproduce']),
            'per_cell': res['reproduce'],
        },
        'events_scored': res['events_per_arm'],
        'events_scored_per_action': res['events_per_arm_per_action'],
        'build_rejects_F15': res['build_rejects'],
        'bootstrap': res['bootstrap'],
        'far_by_arm': {g: {k: arm_block(v) for k, v in S[g].items()}
                       for g in ('all_20_cells', 'tap_scroll_swipe_12_cells')},
        'paired_differences': {g: {k: diff_block(v, g, k) for k, v in D[g].items()}
                               for g in ('all_20_cells', 'tap_scroll_swipe_12_cells')},
        'per_action': per_action,
        'per_detector': per_detector,
        'per_cell': per_cell,
        'construction_spread_over_R_repetitions': spread,
        'duration_standardised_to_the_release_duration_distribution':
            res['duration_standardised'],
        'duration_standardisation_effect': {
            'what': 'how far any paired difference moves when the duration '
                    'distribution of every arm is standardised to the release '
                    'arm own duration deciles per action',
            'max_abs_move_on_any_difference_any_group': fmt(max(
                abs(res['duration_standardised'][g][k] - D[g][k]['difference'])
                for g in res['duration_standardised']
                for k in ('row3_minus_row2', 'row4_minus_row5', 'row5_minus_row2',
                          'row3_minus_row5', 'row4_minus_row2')), 5),
            'max_abs_move_at_the_20_cell_aggregate': fmt(max(
                abs(res['duration_standardised']['all_20_cells'][k]
                    - D['all_20_cells'][k]['difference'])
                for k in ('row3_minus_row2', 'row4_minus_row5', 'row5_minus_row2',
                          'row3_minus_row5', 'row4_minus_row2')), 5),
            'reading': 'the three duration-confounded differences survive '
                       'standardisation essentially unchanged -- at the 20-cell '
                       'aggregate they move by less than a fifth of the width of '
                       'their own interval -- so duration is not what is driving '
                       'them.'},
        'construction_spread_vs_interval': {
            'max_sd_over_R_repetitions_any_arm_any_group': fmt(max(
                s['sd'] for g in res['construction_spread_over_repetitions']
                for s in res['construction_spread_over_repetitions'][g].values()),
                5),
            'min_bootstrap_ci_halfwidth_20_cell_arms': fmt(min(
                (S['all_20_cells'][a]['ci95'][1]
                 - S['all_20_cells'][a]['ci95'][0]) / 2
                for a in ('row2', 'row3', 'row4', 'row5')), 5),
            'reading': 'the spread induced by re-drawing which of the five shots '
                       'is used is an order of magnitude smaller than the '
                       'user-clustered interval, so the construction is not what '
                       'limits the precision -- the 20 users are.'},
        'row2_crosscheck': {
            'what': 'the release arm FAR recomputed here, per cell, against the '
                    'FAR computed straight off the published test_scores',
            'max_abs_diff': max(
                abs(res['per_cell'][r_['cell']]['row2']['far_at_frr5']
                    - r_['published_far5_label1'])
                for r_ in res['reproduce'])},
        'outcome_against_preregistration': v20,
        'plain_statement': plain_statement(S, D, v20),
        'frozen_detectors_unchanged_F6': res['checksums_unchanged'],
        'caveats_that_bound_this_result': [
            'Row 5 is OUT of the five-shot budget by design (F4): it draws its '
            'touch template from the victim full genuine pool, up to 84 distinct '
            'recordings per victim-action-repetition, against exactly 5 for rows '
            '3 and 4. It is a control on pairing mechanics, not an attack, and '
            'must never appear inside a five-shot table.',
            'For pinch and keystroke the release second stage could not be '
            'reconstructed, so S2 is the identity in rows 3, 4 and 5 alike. Rows '
            '3 and 4 for those two actions are valid only against row 5.',
            'For tap and scroll row4 - row2 is near-tautological (F14): the '
            'released fake touch for those actions IS the victim own held '
            'recording transported to the carrier endpoints.',
            'row3 - row5 is not uniform across actions: it is negative for '
            'swipe, pinch and keystroke and NULL for tap and scroll. The '
            '20-cell aggregate hides that split.',
            'A residual resample-ratio gap between row 3 and row 5 remains for '
            'swipe (mean log2 0.099 vs 0.052) and pinch (0.178 vs 0.124); those '
            'are exactly the two actions with the largest row3 - row5 gap, so '
            'part of that gap may be resampling rather than content. The '
            'row5_unmatched diagnostic bounds the effect of the length match at '
            'row5 - row5_unmatched = ' + str(round(
                D['all_20_cells']['row5_minus_row5_unmatched']['difference'], 4)),
            'These are event-level rates on the 20 classical joint cells only. '
            'The 10 deep joint cells are GPU-deferred and could order the arms '
            'differently.',
        ],
        'what_this_settles_for_the_standing_audit_item': (
            'The joint modality deficit is NOT a cross-channel content '
            'consistency check. The dual-channel control that did not previously '
            'exist now exists and shows that a mismatched pair of two genuine '
            'recordings of the same victim -- maximally content-inconsistent, '
            'zero synthetic material -- is the MOST accepted of the four arms '
            'under test, above the release fake and only ' + str(round(
                S['all_20_cells']['row1']['far_at_frr5']
                - S['all_20_cells']['row5']['far_at_frr5'], 3)) +
            ' below the genuine arm. Whatever these joint cells are reading, it '
            'is a property of the synthesised material itself, not the agreement '
            'between the two channels.'),
    }
    doc.pop('results_note', None)
    fc = doc['frozen_detector_checksums']
    fc['scoring_phase_before'] = res['checksums_before']
    fc['scoring_phase_after'] = res['checksums_after']
    fc['scoring_phase_unchanged'] = res['checksums_unchanged']
    fc['sha256_identical_across_build_phase_and_scoring_phase'] = all(
        fc['before'][k]['model.joblib'] == res['checksums_after'][k]['model.joblib']
        and fc['before'][k]['thresholds.json']
        == res['checksums_after'][k]['thresholds.json']
        for k in res['checksums_after'])
    fc['note'] = ('the build-phase blocks carry extra bookkeeping keys (file '
                  'size, frr5, bundle); the sha256 fields are what is compared '
                  'and they are identical in all four snapshots.')
    doc['compliance']['F2'] = (
        'NOT APPLICABLE and no new calibration was performed. The operating point '
        'is the released cell own frr5 scalar, whose thresholds.json records '
        'selection_split=development; it is the identical threshold the published '
        'main table uses. No threshold was re-derived from test scores here, so '
        'there is no split-versus-oracle gap to show for this experiment.')
    doc['compliance']['F14'] = (
        'the two known degeneracies are labelled AT the number: every '
        'row4_minus_row2 difference for tap and scroll carries the '
        'near-tautology label, and every rows-3/4 against rows-1/2 difference '
        'for pinch and keystroke carries the not-a-valid-comparison label. '
        'Negative and null results are reported as they fell.')
    doc['compliance']['F6'] = (
        'detectors read-only; sha256 of all 20 model.joblib and all 20 '
        'thresholds.json taken immediately before and immediately after the '
        'scoring run: unchanged = ' + str(res['checksums_unchanged']))
    doc['compliance']['F9'] = (
        'every interval is a user-clustered bootstrap: the 20 test users are '
        'resampled as clusters, 2000 replicates, ONE shared user multiset per '
        'replicate across all 20 cells and all six arms, so every paired '
        'difference is paired within the replicate and carries its own interval.')
    doc['scratch_and_code'] = {
        'builder': str(DEST / 'code' / 'half_attack' / 'build_arms.py'),
        'scorer': str(DEST / 'code' / 'half_attack' / 'score_arms.py'),
        'scratch': str(SC),
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print('wrote', OUT, OUT.stat().st_size)
    print(json.dumps(v20, indent=1))


if __name__ == '__main__':
    main()
