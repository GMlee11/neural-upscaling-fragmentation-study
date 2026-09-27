"""Independent analysis of the predeclared E002 complete-service comparisons.

Practical gates are not significance tests, novelty proofs or acceptance odds.
The caller must first authenticate the frozen schedule, masks and retained data.
All means weight masks equally; repetitions are not new independent geometries.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache
from statistics import mean
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from src.report.paper3_strengthening_development import STAGES, WIDTHS

ACCEPTED_BUNDLE_SHA = '6985b7171d109228ff8ed55584b23b0122f045d80f147dbfbb3c1a81d2ec5abc'
ACCEPTED_SOURCE_SHA = '0364767473f4b073f6b453057f4a13f42c8009a67d8de529b3d166f8e3ef7a95'
POLICIES = ('exact', 'row_spans', 'grouped', 'per_row', 'affine_partition', 'lookup_partition')
CONFIGURATIONS = tuple(f'{p}:{m}' for p in POLICIES for m in ('whole', 'selected'))
PRACTICAL_GATES = {
    'chooser_mean_gain_fraction': .05, 'chooser_positive_each_session': True,
    'maximum_pooled_mask_regression_fraction': .10, 'primary_geometry_policy': 'row_spans',
    'geometry_minimum_ms': 2., 'geometry_fraction_whole_compact': .05,
    'geometry_minimum_families_per_graph': 2,
}
BOOTSTRAP_CONTRACT = {
    'draws': 5000, 'seed': 20260925, 'percentiles': [2.5, 97.5],
    'unit': 'matched pair within family; preserve both members/sessions/repeats',
    'best_fixed_reselected_each_draw': True,
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('invalid nonnegative timing')
    return float(value)


def _left_sum(values):
    """Match the frozen v3 binary64 left fold, not version-dependent sum()."""
    result = 0.
    for value in values:
        result = result + float(value)
    return result


def frozen_expectations(profile, rows):
    """Independent finite-row reconstruction of frozen jobs and candidate scores.

    Does not import the timed planner, execute bundle code, read measured costs,
    or search held-out outcomes. Tuples are (x, y, width, ordered commits).
    The score covers calibrated job components, not fallback/complete service.
    """
    rows = tuple(rows)
    if (len(rows) != 6 or any(not isinstance(r, str) or len(r) != 10 or
                            set(r)-{'0', '1'} for r in rows) or
            not any('1' in r for r in rows)):
        raise ValueError('invalid nonempty frozen mask')
    encoded = json.dumps(profile, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return _frozen_expectations(encoded, rows)


@lru_cache(maxsize=256)
def _frozen_expectations(encoded, rows):
    profile = json.loads(encoded)
    if set(profile) != {'whole', 'selected'}:
        raise ValueError('missing reconstruction calibration')
    for mode in profile.values():
        if set(mode['costs']) != {f'{kind}:{width}' for kind in ('top', 'middle', 'bottom') for width in WIDTHS}:
            raise ValueError('incomplete frozen shape profile')
        if len(mode['coefficients']) != 3:
            raise ValueError('invalid affine coefficient count')
        for value in mode['coefficients']:
            _number(value)
        for value in mode['costs'].values():
            _number(value['base_ms']); _number(value['commit_tile_ms'])

    def row_kind(row):
        return 'top' if row == 0 else 'bottom' if row == 5 else 'middle'

    def cost(job, mode, affine=False):
        x, y, width, commits = job
        if affine:
            launch, per_width, per_commit = map(float, profile[mode]['coefficients'])
            return launch + per_width*(width/64) + per_commit*len(commits)
        entry = profile[mode]['costs'][f'{row_kind(y//60)}:{width}']
        return float(entry['base_ms']) + float(entry['commit_tile_ms'])*len(commits)

    def spanning(row, columns, width=None):
        left, right = columns[0]*64, (columns[-1]+1)*64
        width = width or next(w for w in WIDTHS if w >= right-left)
        return (max(min(left, 640-width), right-width, 0), row*60, width,
                tuple(row*10+c for c in columns))

    def basics(row, columns):
        exact = tuple((c*64, row*60, 64, (row*10+c,)) for c in columns)
        span = (spanning(row, columns),)
        runs, start = [], 0
        for end in range(1, len(columns)+1):
            if end == len(columns) or columns[end] != columns[end-1]+1:
                runs.append(spanning(row, columns[start:end])); start = end
        return exact, span, tuple(runs)

    def partition(row, columns, mode, affine):
        # Ordinary additive interval DP with the frozen deterministic tie key.
        states = [(0., ())] + [(float('inf'), ()) for _ in columns]
        def key(item):
            total, jobs = item
            return total, len(jobs), tuple((j[2], j[0], j[3]) for j in jobs)
        for end in range(1, len(columns)+1):
            for begin in range(end):
                selected = columns[begin:end]
                extent = (selected[-1]-selected[0]+1)*64
                for width in WIDTHS:
                    if width >= extent:
                        job = spanning(row, selected, width)
                        proposed = states[begin][0]+cost(job, mode, affine), states[begin][1]+(job,)
                        if key(proposed) < key(states[end]):
                            states[end] = proposed
        return states[-1][1]

    result = {}
    for config in CONFIGURATIONS:
        policy, mode = config.split(':')
        jobs, row_scores = [], []
        for row, bits in enumerate(rows):
            columns = [i for i, bit in enumerate(bits) if bit == '1']
            if not columns:
                row_scores.append(0.); continue
            options = basics(row, columns)
            if policy in POLICIES[:3]:
                selected = options[POLICIES.index(policy)]
            elif policy == 'per_row':
                selected = min(enumerate(options),
                               key=lambda pair: (_left_sum(cost(j, mode) for j in pair[1]), pair[0]))[1]
            else:
                selected = partition(row, columns, mode, policy == 'affine_partition')
            jobs.extend(selected)
            row_scores.append(_left_sum(cost(j, mode) for j in selected))
        result[config] = {'jobs': tuple(jobs), 'score_ms': _left_sum(row_scores)}
    winner = min(CONFIGURATIONS, key=lambda c: result[c]['score_ms'])
    return result, winner


def analyze(records, masks, fixed_configurations, chooser='chooser',
            row_whole='row_spans:whole', row_selected='row_spans:selected'):
    """Analyze one graph, two sessions and two repetitions per configuration.

    Mask metadata: mask_id, pair_id, family, member (compact/fragmented).
    Record keys: session, repeat, mask_id, configuration, service_ns,
    normalization_ns. Endpoint semantics are established by the frozen runner,
    not inferred from timing values here.
    """
    metadata = {m['mask_id']: m for m in masks}
    if len(metadata) != len(masks) or len(masks) != 16:
        raise ValueError('expected sixteen unique frozen masks')
    configs = tuple(fixed_configurations)
    if len(configs) != 12 or len(set(configs)) != 12 or chooser in configs:
        raise ValueError('expected twelve fixed configurations plus chooser')
    if row_whole not in configs or row_selected not in configs:
        raise ValueError('missing primary row-span controls')
    pairs = {}
    families = {}
    for mask in masks:
        member = mask['member']
        if member not in ('compact', 'fragmented'):
            raise ValueError('unknown matched-pair member')
        pair = pairs.setdefault(mask['pair_id'], {})
        if member in pair:
            raise ValueError('duplicate matched-pair member')
        pair[member] = mask['mask_id']
        family = families.setdefault(mask['pair_id'], mask['family'])
        if family != mask['family']:
            raise ValueError('pair family mismatch')
    if len(pairs) != 8 or any(set(p) != {'compact', 'fragmented'} for p in pairs.values()):
        raise ValueError('expected eight complete matched pairs')
    family_ids = sorted(set(families.values()))
    if len(family_ids) != 4 or any(list(families.values()).count(f) != 2 for f in family_ids):
        raise ValueError('expected four families with two pairs each')
    actual = {}
    for record in records:
        key = (record['session'], record['repeat'], record['mask_id'], record['configuration'])
        if key[0] not in (0, 1) or key[1] not in (0, 1) or key[2] not in metadata or key[3] not in (*configs, chooser) or key in actual:
            raise ValueError('unexpected or duplicate evaluation row')
        service = _number(record['service_ns']) / 1e6
        normalization = _number(record['normalization_ns']) / 1e6
        if service <= 0 or normalization > service:
            raise ValueError('invalid service/component boundary')
        actual[key] = {'service': service, 'normalization': normalization}
    if len(actual) != 2*2*16*13:
        raise ValueError('incomplete evaluation matrix')

    def value(config, mask, sessions=(0, 1), metric='service'):
        return mean(actual[s, r, mask, config][metric] for s in sessions for r in (0, 1))

    def aggregate(config, sessions=(0, 1)):
        return mean(value(config, mask, sessions) for mask in metadata)

    means = {c: aggregate(c) for c in (*configs, chooser)}
    # Frozen configuration order breaks exact ties, not observed casewise wins.
    best = min(configs, key=means.get)
    reduction = (means[best]-means[chooser])/means[best]*100
    per_mask = {mask: {
        'chooser_ms': value(chooser, mask), 'global_best_fixed_ms': value(best, mask),
        'reduction_percent': (value(best, mask)-value(chooser, mask))/value(best, mask)*100,
    } for mask in metadata}
    sessions = {}
    for session in (0, 1):
        sm = {c: aggregate(c, (session,)) for c in (*configs, chooser)}
        sb = min(configs, key=sm.get)
        sessions[str(session)] = {
            'means_ms': sm, 'best_fixed': sb,
            'chooser_reduction_percent': (sm[sb]-sm[chooser])/sm[sb]*100,
        }
    worst = min(v['reduction_percent'] for v in per_mask.values())
    deployment_pass = reduction >= 5 and worst >= -10 and all(
        s['chooser_reduction_percent'] > 0 for s in sessions.values())
    comparisons = {}
    for config in (*configs, chooser):
        comparisons[config] = {
            'mean_service_ms': means[config],
            'reduction_vs_best_fixed_percent': (means[best]-means[config])/means[best]*100,
            'mask_regressions_vs_best_fixed': sum(value(config, m) > value(best, m) for m in metadata),
            'worst_mask_reduction_percent': min((value(best, m)-value(config, m))/value(best, m)*100 for m in metadata),
        }

    # Preserve compact/fragmented pairing and family balance. Two pairs/family
    # are little support: this is sensitivity analysis, not population coverage.
    rng = random.Random(20260925)
    pooled_pair_means = {p: {c: mean(value(c, mask) for mask in pair.values())
                            for c in (*configs, chooser)} for p, pair in pairs.items()}
    by_family = [[p for p in pairs if families[p] == family] for family in family_ids]
    bootstrap = []
    for _ in range(5000):
        draw = [rng.choice(family) for family in by_family for _ in range(2)]
        bm = {c: mean(pooled_pair_means[p][c] for p in draw) for c in (*configs, chooser)}
        base = min(bm[c] for c in configs)
        bootstrap.append((base-bm[chooser])/base*100)
    lo, hi = map(float, np.percentile(bootstrap, (2.5, 97.5)))

    geometry = {}
    for family in family_ids:
        ids = [p for p in pairs if families[p] == family]
        by_session = {}
        for session in (0, 1):
            by_pair = {}
            for ident in ids:
                compact, fragmented = pairs[ident]['compact'], pairs[ident]['fragmented']
                components = {}
                for metric in ('service', 'normalization'):
                    penalties = {mode: value(config, fragmented, (session,), metric)-value(config, compact, (session,), metric)
                                 for mode, config in (('whole', row_whole), ('selected', row_selected))}
                    components[metric] = dict(penalties, intervention_ms=penalties['whole']-penalties['selected'])
                by_pair[ident] = components
            by_session[str(session)] = {
                'pairs': by_pair,
                'service_intervention_ms': mean(p['service']['intervention_ms'] for p in by_pair.values()),
                'normalization_intervention_ms': mean(p['normalization']['intervention_ms'] for p in by_pair.values()),
            }
        effect = mean(s['service_intervention_ms'] for s in by_session.values())
        compact_mean = mean(value(row_whole, pairs[p]['compact']) for p in ids)
        threshold = max(2., .05*compact_mean)
        effects = [s['service_intervention_ms'] for s in by_session.values()]
        same_sign = all(e > 0 for e in effects) or all(e < 0 for e in effects)
        passed = abs(effect) >= threshold and same_sign
        geometry[family] = {
            'sessions': by_session, 'pooled_service_intervention_ms': effect,
            'whole_compact_mean_ms': compact_mean, 'practical_threshold_ms': threshold,
            'same_sign_in_both_sessions': same_sign, 'magnitude_gate_pass': passed,
            'supports_penalty_reduction': passed and effect > 0,
        }
    return {
        'status': 'complete_practical_analysis', 'evaluation_rows': len(actual),
        'interpretation': 'Preregistered practical thresholds, not significance or novelty guarantees; two sessions and eight matched pairs are limited evidence.',
        'best_fixed': best, 'configurations': comparisons, 'sessions': sessions,
        'chooser': {'mean_reduction_percent': reduction, 'worst_mask_reduction_percent': worst,
                    'per_mask': per_mask, 'practical_gate_pass': deployment_pass,
                    'bootstrap_sensitivity': {
                        'draws': 5000, 'seed': 20260925, 'percentiles': [2.5, 97.5],
                        'reduction_interval_percent': [lo, hi],
                        'interpretation': 'Descriptive stratified paired-mask bootstrap with best-fixed reselection; only two pairs per family, not validated population coverage or session/device generalization.',
                    },
                    'positive_sensitivity_interval': lo > 0},
        'primary_geometry_policy': 'row_spans', 'geometry_families': geometry,
        'geometry_magnitude_gate_pass': sum(v['magnitude_gate_pass'] for v in geometry.values()) >= 2,
        'geometry_penalty_reduction_gate_pass': sum(v['supports_penalty_reduction'] for v in geometry.values()) >= 2,
    }


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _recorded_controls(events):
    """Authenticate saved two-sweep snapshots, not continuous NPU occupancy.

    The runner checks its backend PID live but does not separately persist that
    PID. Replay checks the stable recorded PID/starttime identities and the
    live runner's empty foreign-owner field, not an independent worker identity.
    """
    helper = 'c86a1ae79a72a6955b8d0dc15c75e36e2fcbe2d332b981f4d7dde69022ecb4a7'
    baseline = None
    active_owners = None
    temperatures, monotonic = [], []
    def check(event, stage, active):
        nonlocal baseline, active_owners
        if event.get('state') != 'control' or event.get('stage') != stage:
            raise ValueError('missing or reordered point-in-time control')
        snap = event['snapshot']; capture = snap['dual_node_capture']; proof = snap['npu_ownership_proof']
        if (capture['helper_sha256'] != helper or proof['helper_sha256'] != helper or
                capture['returncode'] != 0 or capture['parse_error'] is not None or capture['stderr'] or
                _sha(capture['stdout'].encode()) != capture['stdout_sha256']):
            raise ValueError('dual-node helper identity/capture failure')
        coverage = json.loads(capture['stdout'])
        owners = proof['owners']
        if (proof['coverage'] != coverage or coverage.get('schema') != 1 or
                coverage.get('scope') != 'RK3576_DUAL_NODE_NPU_OWNERS_ONLY' or
                coverage.get('complete') is not True or coverage.get('errors') != [] or
                coverage.get('covered_devices') != [[226, 1], [226, 129]] or
                coverage.get('sweeps') != [owners, owners] or
                _sha(json.dumps(owners, sort_keys=True, separators=(',', ':')).encode()) != coverage.get('owner_vector_sha256') or
                snap.get('npu_ownership_complete') is not True):
            raise ValueError('incomplete or inconsistent dual-node proof')
        if any(not isinstance(row, dict) or type(row.get('pid')) is not int or row['pid'] <= 0 or
               type(row.get('starttime')) is not int or row['starttime'] < 0 or
               row.get('device') not in ([226, 1], [226, 129]) for row in owners):
            raise ValueError('malformed recorded owner identity')
        identities = {(row['pid'], row['starttime']) for row in owners}
        if active:
            if not identities:
                raise ValueError('loaded backend has no recorded NPU owner')
            if active_owners is None:
                active_owners = identities
            elif identities != active_owners:
                raise ValueError('recorded owner identity changed')
        elif identities:
            raise ValueError('unexpected owner outside backend lifetime')
        invariants = snap['invariants']
        if invariants.get('other_npu_owners') != []:
            raise ValueError('recorded foreign NPU owner')
        if baseline is None:
            baseline = invariants
        elif invariants != baseline:
            raise ValueError('recorded control invariants drifted')
        temperature = _number(snap['temperature_c'])
        stamp = _number(snap['monotonic_ns'])
        if temperature >= 80 or (monotonic and stamp <= monotonic[-1]):
            raise ValueError('recorded thermal/clock control failure')
        temperatures.append(temperature); monotonic.append(stamp)

    index = 1
    while index < len(events) and events[index].get('state') == 'control' and events[index].get('stage') == 'start':
        check(events[index], 'start', False); index += 1
    if not temperatures or temperatures[-1] > 60:
        raise ValueError('missing or too-hot accepted start')
    if events[index].get('state') != 'loaded':
        raise ValueError('missing backend load boundary')
    index += 1; check(events[index], 'loaded', True); index += 1
    if events[index].get('state') != 'preparation':
        raise ValueError('preparation is outside its frozen boundary')
    index += 1; check(events[index], 'prepared', True); index += 1
    for kind in ['reference']+['qualification']*48+['evaluation']*416:
        check(events[index], 'before_'+kind, True); index += 1
        if events[index].get('state') != 'sample' or events[index]['request'].get('kind') != kind:
            raise ValueError('sample is not bracketed by its controls')
        index += 1; check(events[index], 'after_'+kind, True); index += 1
    check(events[index], 'closed', False); index += 1
    if index != len(events)-1 or events[index].get('state') != 'complete':
        raise ValueError('unexpected events outside recorded control sequence')
    return {'snapshots': len(temperatures), 'minimum_temperature_c': min(temperatures),
            'maximum_temperature_c': max(temperatures),
            'recorded_pid_starttime': sorted(active_owners),
            'interpretation': 'Two point-in-time helper sweeps before/after each sample; not continuous occupancy or independently persisted backend PID identity.'}


def authenticate_run(directory, bundle, expected_manifest_sha=None):
    """Check one 465-sample cell/session against its locally retained source.

    This authenticates saved records, not continuous board ownership or a new
    numerical inference. The separate retained-pixel replay is also required.
    """
    directory, bundle = Path(directory).resolve(), Path(bundle).resolve()
    manifest_raw = (bundle/'source_manifest.json').read_bytes()
    if expected_manifest_sha is not None and _sha(manifest_raw) != expected_manifest_sha:
        raise ValueError('source is not the accepted frozen manifest')
    manifest = json.loads(manifest_raw)
    for name, pin in manifest['files'].items():
        target = bundle/name
        if target.is_symlink() or not target.resolve().is_relative_to(bundle) or _sha(target.read_bytes()) != pin:
            raise ValueError('frozen bundle file changed or escaped root')
    frozen_raw = (bundle/'frozen.json').read_bytes()
    method_raw = (bundle/'method_freeze.json').read_bytes()
    frozen = json.loads(frozen_raw)
    method = json.loads(method_raw)
    if frozen.get('method_freeze_sha256') != _sha(method_raw):
        raise ValueError('frozen requests do not bind this method')
    if method.get('practical_gates') != PRACTICAL_GATES or method.get('candidate_order') != list(CONFIGURATIONS):
        raise ValueError('analysis gates or candidate tie order differ from freeze')
    if any(method.get('bootstrap', {}).get(k) != v for k, v in BOOTSTRAP_CONTRACT.items()):
        raise ValueError('bootstrap procedure differs from freeze')
    if any(method.get(key) != frozen.get(key) for key in ('profiles', 'compiled', 'materialization_models')):
        raise ValueError('method/profile/compiled-cost binding differs')
    raw = (directory/'journal.jsonl').read_bytes()
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not events or events[0].get('state') != 'running' or events[-1].get('state') != 'complete' or any(e.get('state') == 'failed' for e in events):
        raise ValueError('unfinished or failed physical cell')
    start, end = events[0], events[-1]
    if (start['source_manifest_sha'], start['frozen_sha256'], start['method_freeze_sha256']) != (_sha(manifest_raw), _sha(frozen_raw), _sha(method_raw)):
        raise ValueError('run/source/protocol binding differs')
    if start['cell'] not in ('width24', 'fp16') or start['session'] not in (0, 1):
        raise ValueError('unknown graph/session')
    if start.get('accepted_bundle_sha') != ACCEPTED_BUNDLE_SHA:
        raise ValueError('accepted model/runtime bundle differs')
    if end.get('confirmation') is not True or (end.get('observations'), end.get('qualifications'), end.get('references')) != (416, 48, 1):
        raise ValueError('wrong terminal sample counts')
    try:
        control_audit = _recorded_controls(events)
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError('malformed or truncated recorded control sequence') from error
    prepared = [e for e in events if e.get('state') == 'preparation']
    if len(prepared) != 1:
        raise ValueError('missing/repeated table preparation')
    prepared = prepared[0]
    expected_tables = frozen['compiled'][start['cell']]
    profile = frozen['profiles'][start['cell']]
    if prepared['profile_sha256'] != _sha(json.dumps(profile, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()):
        raise ValueError('frozen cost profile differs')
    if prepared['score_sha256'] != expected_tables['score_sha256']:
        raise ValueError('chooser score table differs')
    if prepared['serialized_score_bytes'] != expected_tables['serialized_score_bytes']:
        raise ValueError('serialized score storage differs')
    preparation_ns = _number(prepared['preparation_ns'])
    if set(prepared['tables']) != set(expected_tables['tables']):
        raise ValueError('planner table set differs')
    for name, table in prepared['tables'].items():
        for key in ('cost_sha256', 'table_sha256', 'serialized_table_bytes'):
            if table[key] != expected_tables['tables'][name][key]:
                raise ValueError('planner table differs')
    if sum(_number(table['preparation_ns']) for table in prepared['tables'].values()) > preparation_ns:
        raise ValueError('nested planner preparation exceeds complete preparation')
    # The runner uses the accepted canonical JSON digest routine; explicit
    # profile/table contents are also source-pinned by frozen.json above.
    schedule = [req for req in frozen['schedule'] if (req['cell'], req['session']) == (start['cell'], start['session'])]
    samples = [e for e in events if e.get('state') == 'sample']
    expected_counts = {'reference': 1, 'qualification': 48, 'evaluation': 416}
    if len(samples) != 465 or any(sum(e['request']['kind'] == kind for e in samples) != count for kind, count in expected_counts.items()):
        raise ValueError('wrong retained sample coverage')
    evaluations = [e for e in samples if e['request']['kind'] == 'evaluation']
    if [e['request'] for e in evaluations] != schedule:
        raise ValueError('evaluation order or frozen requests differ')
    if any(e['request'].get('session') != start['session'] or e['request'].get('cell') != start['cell'] for e in samples):
        raise ValueError('sample belongs to another cell/session')
    if [e['request']['kind'] for e in samples] != ['reference']+['qualification']*48+['evaluation']*416:
        raise ValueError('qualification/evaluation order changed')
    qualification = samples[1:49]
    for index, (y, width, right) in enumerate(itertools.product((0, 300), WIDTHS, (False, True))):
        whole, selected = qualification[2*index:2*index+2]
        job = {'x': 640-width if right else 0, 'y': y, 'width': width,
               'commits': [y//60*10+(9 if right else 0)]}
        if (whole['request']['job'] != job or selected['request']['job'] != job or
            whole['request']['reconstruction'] != 'whole' or selected['request']['reconstruction'] != 'selected' or
            whole['array_sha256'] != selected['array_sha256']):
            raise ValueError('qualification edge coverage or byte agreement differs')
    verified = {}
    for sample in samples:
        if sample['metrics'].get('passed') is not True:
            raise ValueError('recorded numerical failure')
        name = sample['path']
        if Path(name).name != name or not name.startswith('raw_') or not name.endswith('.npz'):
            raise ValueError('invalid array name')
        path = directory/name
        if name not in verified:
            if path.is_symlink() or path.resolve().parent != directory:
                raise ValueError('array escaped run directory')
            payload = path.read_bytes()
            with np.load(path, allow_pickle=False) as data:
                array = data['output']
            if array.shape != (720, 1280, 3) or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
                raise ValueError('invalid retained output')
            verified[name] = (_sha(payload), _sha(np.ascontiguousarray(array).tobytes()))
        if verified[name] != (sample['sha256'], sample['array_sha256']):
            raise ValueError('retained output hash differs')
    index = {}
    normalized = []
    for sample in evaluations:
        req, met = sample['request'], sample['metrics']
        if sum(_number(met[k]) for k in STAGES) > _number(met['service_ns']):
            raise ValueError('timed stages exceed complete service')
        if _number(met['decision_ns']) > _number(met['planning_ns']):
            raise ValueError('decision time not included in planning')
        jobs = met['per_job']
        expected_commits = [r*10+c for r, row in enumerate(req['rows']) for c, bit in enumerate(row) if bit == '1']
        if sorted(i for j in jobs for i in j['commits']) != expected_commits or len(jobs) != met['calls']:
            raise ValueError('changed/duplicated selected-output commits')
        chosen = met['chosen_policy']+':'+met['chosen_reconstruction']
        if req['configuration'] != 'frozen_choice' and req['configuration'] != chosen:
            raise ValueError('fixed configuration silently changed')
        expected, winner = frozen_expectations(profile, req['rows'])
        if chosen not in expected:
            raise ValueError('unknown recorded execution configuration')
        signature = tuple((j['x'], j['y'], j['width'], tuple(j['commits'])) for j in jobs)
        if signature != expected[chosen]['jobs']:
            raise ValueError('recorded jobs differ from frozen planner semantics')
        if req['configuration'] == 'frozen_choice':
            if chosen != winner or _number(met['frozen_calibrated_job_cost_ms']) != expected[winner]['score_ms']:
                raise ValueError('chooser differs from frozen score argmin/tie-break')
        elif met.get('frozen_calibrated_job_cost_ms') is not None:
            raise ValueError('fixed configuration unexpectedly reported a chooser score')
        index[req['repeat'], req['mask_id'], req['configuration']] = sample
        normalized.append({key: req[key] for key in ('session', 'repeat', 'mask_id', 'configuration')} |
                          {key: met[key] for key in ('service_ns', 'normalization_ns')})
    def signature(sample):
        return [(j['x'], j['y'], j['width'], tuple(j['commits'])) for j in sample['metrics']['per_job']]
    fixed_pairs = 0
    for key, sample in index.items():
        repeat, ident, config = key
        if config in ('exact:whole', 'row_spans:whole', 'grouped:whole'):
            paired = index[repeat, ident, config.replace(':whole', ':selected')]
            if signature(sample) != signature(paired) or sample['array_sha256'] != paired['array_sha256']:
                raise ValueError('fixed-policy reconstruction changed jobs/bytes')
            fixed_pairs += 1
        if config == 'frozen_choice':
            met = sample['metrics']
            chosen = met['chosen_policy']+':'+met['chosen_reconstruction']
            paired = index[repeat, ident, chosen]
            if signature(sample) != signature(paired) or sample['array_sha256'] != paired['array_sha256']:
                raise ValueError('chooser differs from stated fixed configuration')
    return {
        'cell': start['cell'], 'session': start['session'], 'journal_sha256': _sha(raw),
        'source_manifest_sha256': _sha(manifest_raw), 'frozen_sha256': _sha(frozen_raw),
        'unique_raw_arrays': len(verified), 'fixed_reconstruction_pairs': fixed_pairs,
        'preparation': prepared, 'recorded_control_audit': control_audit, 'records': normalized,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('refusing to replace an earlier report')
    runs = [authenticate_run(path, args.bundle, expected_manifest_sha=ACCEPTED_SOURCE_SHA) for path in args.run]
    if len(runs) != 4 or {(r['cell'], r['session']) for r in runs} != {(c, s) for c in ('width24', 'fp16') for s in (0, 1)}:
        raise ValueError('need exactly two sessions for each graph')
    frozen = json.loads((args.bundle/'frozen.json').read_bytes())
    configs = CONFIGURATIONS
    results = {cell: analyze([row for run in runs if run['cell'] == cell for row in run['records']],
                            frozen['masks'], configs, chooser='frozen_choice') for cell in ('width24', 'fp16')}
    report = {'status': 'complete_saved_record_audit', 'graphs': results,
              'runs': [{k: v for k, v in run.items() if k != 'records'} for run in runs],
              'limits': ['Numerical replay against retained references is separate.',
                         'Saved-record audit does not independently authenticate continuous historical board ownership.',
                         'No significance, novelty or journal-acceptance probability is inferred.']}
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'graphs': {
        c: {'chooser_gate': r['chooser']['practical_gate_pass'],
            'geometry_gate': r['geometry_magnitude_gate_pass']} for c, r in results.items()}}))


if __name__ == '__main__':
    main()
