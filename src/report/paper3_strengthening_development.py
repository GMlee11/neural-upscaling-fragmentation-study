"""Independent replay of the bounded E001 engineering screen, not confirmation.

Checks complete schedules and retained arrays. This does not rerun inference,
authenticate a live board, independently regenerate fallback, or infer novelty.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np

from src.experiments.paper3_dispatch_successor_v1.core import WIDTHS, development_masks

POLICIES = ('exact', 'row_spans', 'grouped', 'per_row', 'affine_partition', 'lookup_partition')
MODES = ('whole', 'selected')
STAGES = ('planning_ns', 'fallback_ns', 'extraction_ns', 'inference_ns', 'normalization_ns', 'scatter_ns')


def masks():
    return {m['mask_id']: m for m in development_masks()
            if m['selected_tiles'] in (12, 36) and m['mask_id'].endswith('_0')}


def paired_summary(candidate, baseline):
    if not candidate or len(candidate) != len(baseline):
        raise ValueError('incomplete paired comparison')
    if any(not math.isfinite(v) or v <= 0 for v in candidate + baseline):
        raise ValueError('invalid service times')
    reductions = [(b-c)/b*100 for c, b in zip(candidate, baseline)]
    return {'pairs': len(candidate), 'mean_service_reduction_percent': (mean(baseline)-mean(candidate))/mean(baseline)*100,
            'median_paired_reduction_percent': median(reductions),
            'regressions': sum(c > b for c, b in zip(candidate, baseline)),
            'worst_paired_reduction_percent': min(reductions)}


def summarize_development(records):
    expected = masks()
    keys = list(itertools.product(range(3), sorted(expected), POLICIES, MODES))
    actual = {}
    positions = {}
    for position, record in enumerate(records):
        req, met = record['request'], record['metrics']
        key = (req['repeat'], req['mask_id'], req['policy'], req['reconstruction'])
        if key not in keys or key in actual or tuple(req['rows']) != tuple(expected[req['mask_id']]['rows']):
            raise ValueError('unexpected, duplicate or changed development request')
        if met.get('passed') is not True:
            raise ValueError('failed numerical sample')
        values = [met[k] for k in ('service_ns',) + STAGES]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values) or values[0] <= 0:
            raise ValueError('invalid timing')
        if sum(values[1:]) > values[0]:
            raise ValueError('stage times exceed complete service')
        actual[key] = record
        positions[key] = position
    if set(actual) != set(keys):
        raise ValueError('incomplete development schedule')
    pairs = list(itertools.product(range(3), sorted(expected)))
    values = {(p, m): [actual[r, ident, p, m]['metrics']['service_ns']/1e6 for r, ident in pairs]
              for p, m in itertools.product(POLICIES, MODES)}
    result = {}
    for mode in MODES:
        fixed = min(('exact', 'row_spans', 'grouped'), key=lambda p: mean(values[p, mode]))
        policies = {}
        for policy in POLICIES:
            rows = [actual[r, ident, policy, mode]['metrics'] for r, ident in pairs]
            policies[policy] = {'mean_service_ms': mean(values[policy, mode]),
                'median_service_ms': median(values[policy, mode]),
                'mean_stages_ms': {k.removesuffix('_ns'): mean(m[k] for m in rows)/1e6 for k in STAGES},
                'vs_fixed': {base: paired_summary(values[policy, mode], values[base, mode]) for base in ('exact', 'row_spans', 'grouped')},
                'vs_best_fixed_posthoc': paired_summary(values[policy, mode], values[fixed, mode]),
                'per_mask': {ident: {
                    'family': expected[ident]['family'],
                    'mean_service_ms': mean(actual[r, ident, policy, mode]['metrics']['service_ns']/1e6 for r in range(3)),
                    'vs_best_fixed_posthoc': paired_summary(
                        [actual[r, ident, policy, mode]['metrics']['service_ns']/1e6 for r in range(3)],
                        [actual[r, ident, fixed, mode]['metrics']['service_ns']/1e6 for r in range(3)]),
                    'job_width_histogram': dict(Counter(str(j['width']) for r in range(3)
                        for j in actual[r, ident, policy, mode]['metrics'].get('per_job', []))),
                } for ident in sorted(expected)}}
            policies[policy]['recorded_numerical_errors'] = {
                name: max((m[name] for m in rows if name in m), default=None)
                for name in ('max_tile_mae', 'selected_full_reference_mae')}
        result[mode] = {'best_fixed_posthoc': fixed, 'policies': policies}
    result['selected_vs_whole'] = {p: paired_summary(values[p, 'selected'], values[p, 'whole']) for p in POLICIES}
    result['selected_vs_whole_interpretation'] = 'Fixed policies have identical jobs; fitted policies may change jobs because each reconstruction mode has separate calibration.'
    result['reconstruction_pair_order'] = {p: {
        'whole_first': sum(positions[r, ident, p, 'whole'] < positions[r, ident, p, 'selected'] for r, ident in pairs),
        'selected_first': sum(positions[r, ident, p, 'selected'] < positions[r, ident, p, 'whole'] for r, ident in pairs),
    } for p in POLICIES}
    result['fixed_policy_reconstruction_hash_agreement'] = {p: {
        'pairs': len(pairs),
        'equal': sum(actual[r, ident, p, 'whole'].get('array_sha256') == actual[r, ident, p, 'selected'].get('array_sha256')
                     and 'array_sha256' in actual[r, ident, p, 'whole'] for r, ident in pairs),
    } for p in ('exact', 'row_spans', 'grouped')}
    result['order_limit'] = 'Cyclic rotation does not fully counterbalance reconstruction order; mode differences may include order/cache/governor effects. Do not treat this development screen as isolated causal confirmation.'
    rankings = {}
    for ident in sorted(expected):
        means = {mode: {p: mean(actual[r, ident, p, mode]['metrics']['service_ns']/1e6 for r in range(3))
                        for p in ('exact', 'row_spans', 'grouped')} for mode in MODES}
        best = {mode: min(means[mode], key=means[mode].get) for mode in MODES}
        rankings[ident] = {'mean_service_ms': means, 'best_fixed_posthoc': best,
                          'best_changed': best['whole'] != best['selected']}
    result['per_mask_fixed_rankings'] = rankings
    result['ranking_interpretation'] = 'Descriptive three-repetition means on opened development requests; a reversal may reflect noise and requires fresh validation before a stable-boundary claim.'
    return result


def audit(directory):
    directory = Path(directory).resolve()
    journal = directory/'journal.jsonl'
    data = journal.read_bytes()
    events = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not events or events[0].get('state') != 'running' or events[-1].get('state') != 'complete' or any(e.get('state') == 'failed' for e in events):
        raise ValueError('not a completed engineering attempt')
    if events[-1].get('confirmation') is not False:
        raise ValueError('wrong experimental status')
    records = [e for e in events if e.get('state') == 'sample']
    counts = Counter(e['request']['kind'] for e in records)
    if counts != {'reference': 1, 'qualification': 48, 'profile': 108, 'development': 288}:
        raise ValueError('incomplete or unexpected sample counts')
    if events[-1]['observations'] != 288 or events[-1]['profiles'] != 108:
        raise ValueError('terminal counts differ')
    if sum(e.get('state') == 'profile_frozen' for e in events) != 1:
        raise ValueError('missing or repeated profile freeze')
    freeze_index = next(i for i, e in enumerate(events) if e.get('state') == 'profile_frozen')
    frozen = events[freeze_index]['profile']
    if set(frozen) != set(MODES) or events[freeze_index]['records'] != 108:
        raise ValueError('wrong frozen profile modes/count')
    if any(e.get('state') == 'sample' and
           ((e['request']['kind'] == 'development' and i < freeze_index) or
            (e['request']['kind'] != 'development' and i > freeze_index))
           for i, e in enumerate(events)):
        raise ValueError('calibration/development order changed')
    expected_profiles = set(itertools.product(range(3), (0, 120, 300), WIDTHS, MODES))
    actual_profiles = set()
    for record in records:
        req = record['request']
        if req['kind'] == 'development':
            mode = req['reconstruction']
            if any(req.get(k) != frozen[mode][k] for k in ('costs', 'coefficients')):
                raise ValueError('development costs differ from frozen calibration')
            jobs = record['metrics'].get('per_job')
            if not isinstance(jobs, list) or len(jobs) != record['metrics'].get('calls'):
                raise ValueError('missing job decomposition')
            selected = sorted(i for j in jobs for i in j['commits'])
            expected_selected = [r*10+c for r, row in enumerate(req['rows']) for c, bit in enumerate(row) if bit == '1']
            if selected != expected_selected:
                raise ValueError('job commits change or duplicate selected outputs')
        if req['kind'] != 'profile':
            continue
        job = req['job']
        key = (req['repeat'], job['y'], job['width'], req['reconstruction'])
        expected_commits = [job['y']//60*10+i for i in range(job['width']//64)]
        if key not in expected_profiles or key in actual_profiles or job['x'] != 0 or job['commits'] != expected_commits:
            raise ValueError('profile identities or job geometry changed')
        actual_profiles.add(key)
    if actual_profiles != expected_profiles:
        raise ValueError('profile coverage incomplete')
    reference = None
    verified = {}
    for record in records:
        if record['metrics'].get('passed') is not True:
            raise ValueError('failed sample in completed attempt')
        name = record['path']
        if Path(name).name != name or not name.startswith('raw_') or not name.endswith('.npz'):
            raise ValueError('invalid raw array path')
        path = directory/name
        if name not in verified:
            if path.is_symlink() or path.resolve().parent != directory:
                raise ValueError('raw array leaves owned directory')
            raw = path.read_bytes()
            with np.load(path, allow_pickle=False) as content:
                output = content['output']
            if output.shape != (720, 1280, 3) or not np.issubdtype(output.dtype, np.floating) or not np.isfinite(output).all():
                raise ValueError('invalid raw output')
            verified[name] = (hashlib.sha256(raw).hexdigest(), hashlib.sha256(np.ascontiguousarray(output).tobytes()).hexdigest())
        if verified[name] != (record['sha256'], record['array_sha256']):
            raise ValueError('raw array hash mismatch')
        if record['request']['kind'] == 'reference':
            reference = record['array_sha256']
    qualification = [e for e in records if e['request']['kind'] == 'qualification']
    expected_qualification = list(itertools.product((0, 300), WIDTHS, (False, True)))
    for i in range(0, len(qualification), 2):
        left, right = qualification[i:i+2]
        if left['request']['job'] != right['request']['job'] or left['request']['reconstruction'] != 'whole' or right['request']['reconstruction'] != 'selected' or left['array_sha256'] != right['array_sha256']:
            raise ValueError('qualification reconstruction pair differs')
        y, width, at_right = expected_qualification[i//2]
        job = {'x': 640-width if at_right else 0, 'y': y, 'width': width,
               'commits': [y//60*10+(9 if at_right else 0)]}
        if left['request']['job'] != job:
            raise ValueError('qualification edge coverage changed')
    return {'status': 'complete_development_replay', 'confirmation': False, 'cell': events[0]['cell'],
            'journal_sha256': hashlib.sha256(data).hexdigest(), 'source_manifest_sha256': events[0]['source_manifest_sha'],
            'counts': dict(counts), 'unique_raw_arrays': len(verified), 'reference_array_sha256': reference,
            'limits': ['Development masks already opened; no held-out or cross-session inference.',
                       'Reconstruction order is incompletely counterbalanced; descriptive mode contrasts need fresh validation.',
                       'Verifies retained array hashes and shapes, not a fresh model/fallback numerical evaluation.',
                       'Does not independently authenticate historical board-control snapshots.',
                       'No statistical-significance or novel-algorithm claim.'],
            'development': summarize_development([e for e in records if e['request']['kind'] == 'development'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(args.directory), indent=2, sort_keys=True, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(report)
    print(report)


if __name__ == '__main__':
    main()
