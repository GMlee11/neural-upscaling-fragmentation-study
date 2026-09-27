"""Finite horizontal partition baseline (standard dynamic programming).

Optimality is only for supplied additive costs and the restricted row action
space. It is NOT actual-latency optimality, arbitrary 2D packing optimality or a
new algorithm. Calls can compute overlapping context; commits are disjoint.
"""
from __future__ import annotations

import math
from typing import Callable

from src.experiments.paper3_dispatch_successor_v1.core import Job, WIDTHS, plan, validate_mask


def grouped_rows(rows):
    """Ordinary contiguous-run grouping, same baseline as packing E002."""
    rows = validate_mask(rows)
    jobs = []
    for r, bits in enumerate(rows):
        c = 0
        while c < 10:
            if bits[c] == '0':
                c += 1
                continue
            left = c
            while c < 10 and bits[c] == '1':
                c += 1
            width = next(w for w in WIDTHS if w >= (c-left)*64)
            jobs.append(Job(min(left*64,640-width),r*60,width,tuple(r*10+i for i in range(left,c))))
    return tuple(jobs)


def finite_cost(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('cost must be finite and nonnegative')
    return value


def stable_sum(values):
    """Explicit binary64 left fold, identical across Python3.11/3.12 sum changes.

    Matches the board interpreter's original accumulation, not new fitting.
    """
    total=0.
    for value in values:total=total+float(value)
    return total


def row_class(y):
    if y not in (0, 60, 120, 180, 240, 300):
        raise ValueError('invalid row origin')
    return 'top' if y == 0 else 'bottom' if y == 300 else 'middle'


def candidate(row, columns, i, j, width):
    left, right = columns[i] * 64, (columns[j] + 1) * 64
    if width < right-left or width > 640 or width % 64:
        raise ValueError('invalid covering shape')
    x = max(min(left, 640-width), right-width, 0)
    return Job(x, row*60, width, tuple(row*10+c for c in columns[i:j+1]))


def partition(rows, cost: Callable[[Job], float], widths=WIDTHS):
    rows = validate_mask(rows)
    widths = tuple(sorted(set(widths)))
    if not widths or 64 not in widths or any(type(w) is not int or w <= 0 or w > 640 or w % 64 for w in widths):
        raise ValueError('invalid accepted width catalog')
    result = []
    for row, bits in enumerate(rows):
        columns = [c for c, b in enumerate(bits) if b == '1']
        n = len(columns)
        best = [(float('inf'), ()) for _ in range(n+1)]
        best[0] = (0., ())
        for end in range(1, n+1):
            for begin in range(end):
                extent = (columns[end-1]-columns[begin]+1)*64
                for width in widths:
                    if width < extent:
                        continue
                    job = candidate(row, columns, begin, end-1, width)
                    value = best[begin][0]+finite_cost(cost(job))
                    jobs = best[begin][1]+(job,)
                    # Deterministic tie-breaking, without performance tuning.
                    key = (value, len(jobs), tuple((j.width, j.x, j.commits) for j in jobs))
                    old = best[end]
                    oldkey = (old[0], len(old[1]), tuple((j.width, j.x, j.commits) for j in old[1]))
                    if key < oldkey:
                        best[end] = (value, jobs)
        result.extend(best[n][1])
    return verify_jobs(rows, result)


def verify_jobs(rows, jobs):
    rows = validate_mask(rows)
    expected = [r*10+c for r, bits in enumerate(rows) for c, b in enumerate(bits) if b == '1']
    actual = [i for j in jobs for i in j.commits]
    if sorted(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError('changed or duplicated output mask')
    for j in jobs:
        if j.width not in WIDTHS or j.y not in range(0,360,60) or j.x < 0 or j.x+j.width > 640:
            raise ValueError('invalid job shape/origin')
        if not j.commits or any(i//10 != j.y//60 or i%10*64 < j.x or (i%10+1)*64 > j.x+j.width for i in j.commits):
            raise ValueError('commit outside job')
    return tuple(jobs)


def per_row(rows, cost):
    rows = validate_mask(rows)
    output = []
    for r, bits in enumerate(rows):
        if '1' not in bits:
            continue
        one = tuple(bits if n == r else '0'*10 for n in range(6))
        options = (plan(one,'exact'), plan(one,'row_spans'), grouped_rows(one))
        output.extend(min(enumerate(options), key=lambda p:(stable_sum(finite_cost(cost(j)) for j in p[1]),p[0]))[1])
    return verify_jobs(rows, output)


def lookup_cost(profile):
    def cost(job):
        row = profile[f'{row_class(job.y)}:{job.width}']
        return finite_cost(row['base_ms']) + finite_cost(row['commit_tile_ms'])*len(job.commits)
    return cost


def affine_cost(coefficients):
    if len(coefficients) != 3:
        raise ValueError('three affine coefficients required')
    launch, tile_width, commit = map(finite_cost, coefficients)
    return lambda j: launch+tile_width*(j.width/64)+commit*len(j.commits)
