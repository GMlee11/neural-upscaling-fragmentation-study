"""Prospective target-unfitted transfer: no target timing reader or fit function."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean

from src.experiments.paper3_dispatch_successor_v1.core import (
    FEATURES, POLICIES, choose, development_masks, digest, schedule, summarize,
)

DONOR_SHA = 'c9bfae357925d64dd9e566b8ed0e52fc738bb36e1910b94f6ae4d346f73e0107'
MASK_SEED = 20260924031
ORDER_SEED = 20260924032
COEFFICIENTS = (25.058020696584993, 1.7210469904490526,
                544.3142267347189, 8.148333720198874e-05, 0.0)


def donor(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != DONOR_SHA:
        raise ValueError('donor identity changed; target fitting is forbidden')
    value = json.loads(raw)
    if tuple(value['coefficients']['coefficients']) != COEFFICIENTS:
        raise ValueError('donor coefficient mismatch')
    if value['samples_observed'] != 240 or value['terminal']['status'] != 'complete':
        raise ValueError('incomplete donor')
    rows = value['calibration_rows']
    if len(rows) != 240 or any(r['split'] != 'calibration' or r['cell'] != 'width24' for r in rows):
        raise ValueError('not the original development calibration')
    return {'source_report_sha256': DONOR_SHA, 'coefficients': list(COEFFICIENTS),
            'features': list(FEATURES), 'calibration_requests': 240,
            'calibration_service_ms': sum(r['service_ms'] for r in rows),
            'additional_target_calibration_requests': 0}


def fresh_plan(excluded):
    excluded = {tuple(m) for m in excluded}
    masks = [dict(m, mask_id=m['mask_id'].replace('dev_', 'transfer_', 1),
                  split='fresh_transfer')
             for m in development_masks(seed=MASK_SEED, excluded=excluded)]
    by_id = {m['mask_id']: m for m in masks}
    requests = [dict(r, rows=list(by_id[r['mask_id']]['rows']), split='fresh_transfer')
                for r in schedule(list(by_id), repeats=8, seed=ORDER_SEED)]
    if excluded & {tuple(m['rows']) for m in masks}:
        raise ValueError('old geometry leaked')
    return {'masks': masks, 'requests': requests, 'mask_seed': MASK_SEED,
            'order_seed': ORDER_SEED, 'excluded_unique_geometries': len(excluded),
            'target_calibration_allowed': False,
            'claim': 'fresh geometries on previously studied cells; target-unfitted rule'}


def decision(rows, policy, halo, tensor):
    return choose(rows, policy, halo, COEFFICIENTS, **tensor)


def analyze(records, requests, calibration_service_ms):
    """No selection of good masks. Reject partial schedules and keep all deltas."""
    result = summarize(records, requests)
    fixed = result['best_fixed_posthoc']
    policy = result['policies']['calibrated']
    saved = result['policies'][fixed]['mean_service_ms'] - policy['mean_service_ms']
    by_group = {}
    for row in records:
        by_group.setdefault((row['block'], row['mask_id']), {})[row['policy']] = row
    mismatch = []
    for group in by_group.values():
        best = min(POLICIES[:2], key=lambda p: group[p]['service_ms'])
        observed_mode = 'exact' if best == 'always_exact' else 'row_spans'
        chosen = group['calibrated']['chosen_mode']
        if chosen not in ('exact', 'row_spans'):
            raise ValueError('invalid chosen mode')
        fixed_chosen = 'always_exact' if chosen == 'exact' else 'always_span'
        mismatch.append({'disagrees_with_observed_minimum': chosen != observed_mode,
                         'observed_choice_penalty_ms': group[fixed_chosen]['service_ms'] - group[best]['service_ms']})
    result.update(rule='fixed RK3576 width24 donor; zero target fitting',
                  calibration_service_ms=calibration_service_ms,
                  calibration_break_even_requests=(calibration_service_ms / saved if saved > 0 else None),
                  lower_mean_than_both_fixed=saved > 0,
                  mean_decision_ms=mean(r['policy_ms'] for r in records if r['policy'] == 'calibrated'),
                  observed_choice_mismatches=sum(r['disagrees_with_observed_minimum'] for r in mismatch),
                  observed_choice_penalties=mismatch,
                  warning='paired fixed minimum is noisy, not ground truth; no per-request guarantee')
    return result
