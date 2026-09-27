"""Offline mask-preserving dispatch candidate. No RKNN import or physical action."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from statistics import mean

WIDTHS = (64, 128, 256, 512, 576, 640)
POLICIES = ('always_exact', 'always_span', 'geometry', 'calibrated')
FAMILIES = ('compact', 'scattered', 'clustered', 'mixed')
FEATURES = ('intercept', 'launches', 'core_megapixels', 'tensor_mebibytes', 'commit_megapixels')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_mask(rows):
    rows = tuple(rows)
    if len(rows) != 6 or any(not isinstance(r, str) or len(r) != 10 or set(r) - {'0', '1'} for r in rows):
        raise ValueError('expected six binary row strings of length ten')
    if not any('1' in r for r in rows):
        raise ValueError('empty masks excluded from this study')
    return rows


@dataclass(frozen=True)
class Job:
    x: int
    y: int
    width: int
    commits: tuple[int, ...]


def plan(rows, mode, widths=WIDTHS):
    """Match choose_row_span: smallest accepted width and boundary-clamped origin."""
    rows = validate_mask(rows)
    if mode not in ('exact', 'row_spans'):
        raise ValueError('mode is not an existing execution organization')
    if 64 not in widths:
        raise ValueError('exact shape unavailable')
    jobs = []
    for r, row in enumerate(rows):
        columns = [c for c, bit in enumerate(row) if bit == '1']
        if not columns:
            continue
        if mode == 'exact':
            jobs.extend(Job(c * 64, r * 60, 64, (r * 10 + c,)) for c in columns)
        else:
            left, right = columns[0] * 64, (columns[-1] + 1) * 64
            width = next((w for w in sorted(widths) if w >= right - left), None)
            if width is None:
                raise ValueError('catalog cannot cover span; no compiler fallback')
            x = max(min(left, 640 - width), right - width, 0)
            jobs.append(Job(x, r * 60, width, tuple(r * 10 + c for c in columns)))
    selected = tuple(r * 10 + c for r, row in enumerate(rows) for c, b in enumerate(row) if b == '1')
    if tuple(i for j in jobs for i in j.commits) != selected:
        raise ValueError('selected output mask changed')
    return tuple(jobs)


def features(rows, mode, halo, *, input_itemsize, output_itemsize, boundary_mode, widths=WIDTHS, shape_elements=None):
    if halo not in (3, 7) or input_itemsize <= 0 or output_itemsize <= 0:
        raise ValueError('invalid frozen tensor contract')
    if boundary_mode not in ('edge', 'zero'):
        raise ValueError('explicit accepted boundary semantics required')
    jobs = plan(rows, mode, widths)
    tensor = 0
    for j in jobs:
        key = f'core_{j.width:04d}x0060'
        if boundary_mode == 'zero' and j.width == 640:
            key = 'full_width_top' if j.y == 0 else 'full_width_bottom' if j.y == 300 else 'full_width_middle'
        if shape_elements is None or key not in shape_elements:
            raise ValueError('actual accepted graph tensor-shape metadata required')
        inp, out = shape_elements[key]
        if inp <= 0 or out <= 0:
            raise ValueError('invalid graph tensor extent')
        tensor += inp * input_itemsize + out * output_itemsize
    commit = sum(len(j.commits) for j in jobs) * 64 * 60 * 4
    return (1., float(len(jobs)), sum(j.width * 60 for j in jobs) / 1e6,
            tensor / 2**20, commit / 1e6)


def catalog_tensor_elements(entries):
    """Normalize accepted catalog extents; dtype byte widths remain a live gate."""
    result = {}
    for row in entries:
        variant = row.get('variant')
        key = variant or f"core_{row['core_width']:04d}x{row['core_height']:04d}"
        onnx = row.get('onnx', row)
        inp = row.get('input_shape_nhwc', onnx.get('input_shape_nchw'))
        out = row.get('output_shape_nchw', onnx.get('output_shape_nchw'))
        if not inp or not out or any(type(x) is not int or x <= 0 for x in list(inp)+list(out)) or key in result:
            raise ValueError('missing, invalid or duplicate tensor extent')
        result[key] = (math.prod(inp), math.prod(out))
    return result


def choose(rows, policy, halo, coefficients=None, **tensor):
    if policy == 'always_exact':
        return 'exact'
    if policy == 'always_span':
        return 'row_spans'
    if policy == 'geometry':
        # Fixed simple baseline: >=2x launch reduction with <=25% core padding.
        rows=validate_mask(rows)
        selected=sum(r.count('1') for r in rows)
        active=0
        padded=0
        for row in rows:
            if '1' not in row: continue
            active+=1
            extent=(row.rfind('1')-row.find('1')+1)*64
            width=next((w for w in tensor.get('widths',WIDTHS) if w>=extent),None)
            if width is None: raise ValueError('catalog cannot cover heuristic span')
            padded+=width//64
        return 'row_spans' if active <= selected / 2 and padded <= selected * 1.25 else 'exact'
    exact, span = (features(rows, m, halo, **tensor) for m in ('exact', 'row_spans'))
    if policy != 'calibrated' or coefficients is None or len(coefficients) != len(FEATURES):
        raise ValueError('missing calibrated model or unknown policy')
    if any(not math.isfinite(c) or c < 0 for c in coefficients):
        raise ValueError('invalid coefficients')
    costs = [sum(c * x for c, x in zip(coefficients, f)) for f in (exact, span)]
    return 'row_spans' if costs[1] < costs[0] else 'exact'


def fit_cost(records, cell):
    """Fixed nonnegative ridge fit; per-cell calibration ONLY, never held-out."""
    import numpy as np
    if len(records) < 10 or any(r['split'] != 'calibration' or r['cell'] != cell for r in records):
        raise ValueError('requires per-cell calibration records')
    x = np.asarray([r['features'] for r in records], dtype=float)
    y = np.asarray([r['service_ms'] for r in records], dtype=float)
    if x.shape != (len(records), len(FEATURES)) or not np.isfinite(x).all() or not np.isfinite(y).all() or (x < 0).any() or (y <= 0).any():
        raise ValueError('invalid calibration rows')
    scale = np.maximum(np.sqrt(np.mean(x*x, axis=0)), 1e-12)
    z, beta = x / scale, np.zeros(len(FEATURES))
    # Fixed iterations/regularizer, no held-out search or fitted causal claims.
    for _ in range(2000):
        for j in range(len(beta)):
            residual = y - z @ beta + z[:, j] * beta[j]
            beta[j] = max(0., float(z[:, j] @ residual) / (float(z[:, j] @ z[:, j]) + 1e-6))
    return {'cell': cell, 'training_split': 'calibration', 'features': FEATURES,
            'coefficients': (beta / scale).tolist(), 'calibration_sha256': digest(records),
            'interpretation': 'predictive correlated features; not causal coefficients'}


def development_masks(seed=2026092001, per_stratum=2, excluded=()):
    """Synthetic development only. No trace reader, held-out generator, or backend."""
    rng, used, result = random.Random(seed), {tuple(x) for x in excluded}, []
    for family in FAMILIES:
        for count in (6, 12, 24, 36, 48):
            for replicate in range(per_stratum):
                for attempt in range(1000):
                    start = rng.randrange(60)
                    # Connected spatial region, no row-major frame wrap.
                    compact = [start]
                    while len(compact) < count:
                        frontier = sorted({j for i in compact for j in (i-10,i+10,i-1,i+1)
                                           if 0 <= j < 60 and abs(i//10-j//10)+abs(i%10-j%10)==1
                                           and j not in compact})
                        distance = lambda j: abs(j//10-start//10)+abs(j%10-start%10)
                        closest = min(map(distance, frontier))
                        compact.append(rng.choice([j for j in frontier if distance(j)==closest]))
                    if family == 'compact':
                        selected = compact
                    elif family == 'scattered':
                        selected = rng.sample(range(60), count)
                    elif family == 'clustered':
                        centers = rng.sample(range(60), 3)
                        jitter = {i: rng.random() for i in range(60)}
                        selected = sorted(range(60), key=lambda i: (min(abs(i//10-c//10)+abs(i%10-c%10) for c in centers), jitter[i]))[:count]
                    else:
                        first = compact[:count//2]
                        selected = first + rng.sample([i for i in range(60) if i not in first], count-len(first))
                    rows = tuple(''.join('1' if r*10+c in selected else '0' for c in range(10)) for r in range(6))
                    if rows not in used:
                        break
                else:
                    raise ValueError('unique mask budget exhausted')
                used.add(rows)
                result.append({'mask_id': f'dev_{family}_{count}_{replicate}', 'family': family,
                               'selected_tiles': count, 'rows': rows, 'split': 'development'})
    return result


def schedule(mask_ids, repeats=8, seed=2026092002):
    if not mask_ids or len(set(mask_ids)) != len(mask_ids) or repeats <= 0 or repeats % 4:
        raise ValueError('unique masks and positive multiple-of-four repeats required')
    rng, output = random.Random(seed), []
    base = list(mask_ids)
    for block in range(repeats):
        order = base.copy()
        rng.shuffle(order)
        for mask in order:
            offset = (block + base.index(mask)) % 4
            for p in POLICIES[offset:] + POLICIES[:offset]:
                output.append({'order': len(output), 'block': block, 'mask_id': mask, 'policy': p})
    return output


def summarize(records, expected_schedule, deadline_ms=1000/30):
    """Complete-service policy comparison. Reject missing/extra/nonfinite rows."""
    keys = ('order', 'block', 'mask_id', 'policy')
    identities = [(r['block'],r['mask_id'],r['policy']) for r in expected_schedule]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate expected sample identity')
    if len(records) != len(expected_schedule) or any(any(r[k] != s[k] for k in keys) for r, s in zip(records, expected_schedule)):
        raise ValueError('incomplete or changed schedule: retain failure, no success summary')
    if any(not math.isfinite(r['service_ms']) or r['service_ms'] <= 0 for r in records):
        raise ValueError('nonfinite/nonpositive service')
    if any(r.get('status') != 'complete' or r.get('output_verified') is not True for r in records):
        raise ValueError('failure or output-integrity gate unresolved')
    paired = {}
    for r in records:
        paired.setdefault((r['block'], r['mask_id']), {})[r['policy']] = r['service_ms']
    if any(set(v) != set(POLICIES) for v in paired.values()):
        raise ValueError('missing policy contrast')
    fixed = min(POLICIES[:2], key=lambda p: mean(v[p] for v in paired.values()))
    result = {}
    for p in POLICIES:
        vals = [v[p] for v in paired.values()]
        deltas = [v[p]-v[fixed] for v in paired.values()]
        result[p] = {'n': len(vals), 'mean_service_ms': mean(vals), 'deadline_successes': sum(v <= deadline_ms for v in vals),
                     'mean_delta_best_fixed_ms': mean(deltas), 'regressions_vs_best_fixed': sum(d > 0 for d in deltas),
                     'worst_regression_ms': max(0., max(deltas)),
                     'mean_signed_oracle_gap_ms': mean(v[p]-min(v['always_exact'], v['always_span']) for v in paired.values()),
                     'mean_positive_oracle_regret_ms': mean(max(0., v[p]-min(v['always_exact'], v['always_span'])) for v in paired.values()),
                     'paired_deltas_ms': deltas}
    return {'best_fixed_posthoc': fixed, 'policies': result,
            'oracle': 'paired observed fixed-mode minimum; posthoc noise-sensitive reference, never deployed'}


def analyze_capture(records, expected_schedule):
    """Keep incomplete/failed evidence, without a successful-policy summary."""
    try:
        summary = summarize(records, expected_schedule)
    except (KeyError, TypeError, ValueError) as error:
        return {'status': 'incomplete_or_invalid', 'error': str(error),
                'retained_records': records, 'expected_count': len(expected_schedule)}
    return {'status': 'complete', 'retained_records': records, 'summary': summary}
