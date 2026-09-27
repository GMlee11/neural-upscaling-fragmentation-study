"""Injectable engineering service, deliberately without any device loader.

The future backend must accept the full declared input canvas and return the
full NCHW subpixel tensor. Boundary calls must implement the accepted ORIGINAL
exact boundary semantics. Both interfaces require physical qualification.
"""
import time

import numpy as np

from .core import compose, materialize, unpack_subpixel, validate_packing, validate_mask


def execute(image, rows, planner, fallback, packed_infer, boundary_exact,
            *, clock=time.perf_counter_ns):
    """Measure planning through returned composed frame, including host work.

    planner(rows) -> PackingPlan
    fallback(image) -> full 720p float RGB baseline
    packed_infer(canvas, input_hwc) -> full float NCHW subpixel residual
    boundary_exact(image, tile, halo) -> float (12,60,64) residual

    Loading/compilation/profile calibration are NOT inside this function and
    must be reported separately with a common declared residency policy.
    Component totals intentionally need not sum to service: Python iteration,
    validation and bookkeeping are included in service, not silently omitted.
    """
    start = clock()
    p = validate_packing(planner(rows))
    if p.rows != validate_mask(rows):
        raise ValueError('planner changed the requested mask')
    planned = clock()
    baseline = fallback(image)
    fallback_done = clock()
    residuals = {}
    stages = {'materialize_ns': 0, 'packed_inference_ns': 0,
              'unpack_ns': 0, 'boundary_exact_ns': 0}
    for canvas in p.canvases:
        t = clock()
        value = materialize(image, canvas, p.halo)
        t1 = clock()
        output = packed_infer(canvas, value)
        t2 = clock()
        unpacked = unpack_subpixel(output, canvas, p.halo)
        if set(unpacked) & set(residuals):
            raise ValueError('duplicate packed output')
        residuals.update(unpacked)
        t3 = clock()
        stages['materialize_ns'] += t1 - t
        stages['packed_inference_ns'] += t2 - t1
        stages['unpack_ns'] += t3 - t2
    for tile in p.exact_boundary:
        t = clock()
        value = boundary_exact(image, tile, p.halo)
        if not isinstance(value, np.ndarray) or value.shape != (12, 60, 64) or not np.isfinite(value).all():
            raise ValueError('invalid exact boundary output')
        residuals[tile] = value
        stages['boundary_exact_ns'] += clock() - t
    reconstruct_start = clock()
    output = compose(baseline, p, residuals)
    end = clock()
    return output, dict(stages, planning_ns=planned-start,
                        fallback_ns=fallback_done-planned,
                        reconstruction_ns=end-reconstruct_start,
                        service_ns=end-start, packed_calls=len(p.canvases),
                        exact_boundary_calls=len(p.exact_boundary))
