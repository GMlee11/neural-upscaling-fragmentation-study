"""Mask-preserving conventional controls, not a proposed novel packing algorithm.

Only interior tiles are packed. Boundary tiles retain the accepted exact path:
input-level zero padding is not equivalent to per-layer image-edge padding.
Canvas sizes are INPUT tensor dimensions, not selected-core dimensions.
No compiler, runtime, board access, or evaluation-result fitting occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from src.experiments.paper3_dispatch_successor_v1.core import Job, WIDTHS, plan, validate_mask


@dataclass(frozen=True)
class Placement:
    tile: int
    x: int
    y: int


@dataclass(frozen=True)
class Canvas:
    width: int
    height: int
    placements: tuple[Placement, ...]


@dataclass(frozen=True)
class PackingPlan:
    rows: tuple[str, ...]
    halo: int
    canvases: tuple[Canvas, ...]
    exact_boundary: tuple[int, ...]


def selected(rows):
    return tuple(r * 10 + c for r, row in enumerate(validate_mask(rows))
                 for c, bit in enumerate(row) if bit == '1')


def interior(tile, halo):
    r, c = divmod(tile, 10)
    return c * 64 >= halo and r * 60 >= halo and (c + 1) * 64 + halo <= 640 and (r + 1) * 60 + halo <= 360


def validate_packing(p):
    if p.halo not in (3, 7):
        raise ValueError('only the two declared receptive-field radii are supported')
    seen = list(p.exact_boundary)
    if any(interior(i, p.halo) for i in p.exact_boundary):
        raise ValueError('interior tile incorrectly labeled a boundary fallback')
    w, h = 64 + 2 * p.halo, 60 + 2 * p.halo
    for canvas in p.canvases:
        if not canvas.placements or canvas.width < w or canvas.height < h:
            raise ValueError('empty or undersized canvas')
        occupied = []
        for q in canvas.placements:
            if not interior(q.tile, p.halo) or q.x < 0 or q.y < 0 or q.x + w > canvas.width or q.y + h > canvas.height:
                raise ValueError('invalid source boundary or packed footprint')
            if any(q.x < x + w and x < q.x + w and q.y < y + h and y < q.y + h for x, y in occupied):
                raise ValueError('context footprints overlap')
            occupied.append((q.x, q.y))
            seen.append(q.tile)
    if sorted(seen) != list(selected(p.rows)):
        raise ValueError('missing, duplicated, or unrequested output tile')
    return p


def pack(rows, halo, canvas_shape, *, reverse=False):
    """Standard equal-rectangle shelf packing; no rotation, resizing or dropping.

    reverse is an engineering position-stress control, not a timed policy knob.
    All selected true-image-edge tiles remain explicit exact-path fallbacks.
    """
    rows = validate_mask(rows)
    if halo not in (3, 7):
        raise ValueError('unsupported halo')
    width, height = canvas_shape
    if type(width) is not int or type(height) is not int:
        raise ValueError('integer input tensor dimensions required')
    w, h = 64 + 2 * halo, 60 + 2 * halo
    columns, nrows = width // w, height // h
    capacity = columns * nrows
    if columns < 1 or nrows < 1:
        raise ValueError('canvas cannot contain one complete tile context')
    tiles = selected(rows)
    boundary = tuple(i for i in tiles if not interior(i, halo))
    inside = [i for i in tiles if interior(i, halo)]
    if reverse:
        inside.reverse()
    canvases = []
    for start in range(0, len(inside), capacity):
        placements = tuple(Placement(tile, (slot % columns) * w, (slot // columns) * h)
                           for slot, tile in enumerate(inside[start:start + capacity]))
        canvases.append(Canvas(width, height, placements))
    return validate_packing(PackingPlan(rows, halo, tuple(canvases), boundary))


def materialize(image, canvas, halo):
    """Copy complete source context into a zero-initialized HWC input canvas."""
    if image.shape != (360, 640, 3) or halo not in (3, 7):
        raise ValueError('expected fixed RGB source and supported halo')
    # Reuse the complete-plan validator even when called directly.
    ids = {q.tile for q in canvas.placements}
    rows = tuple(''.join('1' if r * 10 + c in ids else '0' for c in range(10)) for r in range(6))
    validate_packing(PackingPlan(rows, halo, (canvas,), ()))
    value = np.zeros((canvas.height, canvas.width, 3), dtype=image.dtype)
    for q in canvas.placements:
        r, c = divmod(q.tile, 10)
        patch = image[r * 60 - halo:(r + 1) * 60 + halo,
                      c * 64 - halo:(c + 1) * 64 + halo]
        value[q.y:q.y + patch.shape[0], q.x:q.x + patch.shape[1]] = patch
    return value


def unpack_subpixel(value, canvas, halo):
    """Extract 12-channel LR residuals; PixelShuffle/composition is downstream."""
    if value.shape != (1, 12, canvas.height, canvas.width) or not np.isfinite(value).all():
        raise ValueError('invalid full-canvas subpixel residual tensor')
    result = {}
    for q in canvas.placements:
        if q.tile in result:
            raise ValueError('duplicate scatter destination')
        result[q.tile] = np.ascontiguousarray(value[0, :, q.y + halo:q.y + halo + 60,
                                                       q.x + halo:q.x + halo + 64])
    return result


def compose(baseline, p, tile_residuals):
    """Commit selected pixels only; fallback array supplied by accepted pipeline.

    Caller must supply boundary-tile residuals from its qualified exact adapter.
    This function deliberately cannot approximate missing boundary results.
    """
    validate_packing(p)
    if baseline.shape != (720, 1280, 3) or not np.isfinite(baseline).all():
        raise ValueError('invalid full-frame fallback')
    if set(tile_residuals) != set(selected(p.rows)):
        raise ValueError('complete exactly-once selected residuals required')
    output = np.clip(baseline, 0., 1.)
    for tile, residual in tile_residuals.items():
        if residual.shape != (12, 60, 64) or not np.isfinite(residual).all():
            raise ValueError('invalid selected residual')
        # PyTorch PixelShuffle channel order: channel, subpixel-y, subpixel-x.
        pixels = residual.reshape(3, 2, 2, 60, 64).transpose(3, 1, 4, 2, 0).reshape(120, 128, 3)
        r, c = divmod(tile, 10)
        area = output[r * 120:(r + 1) * 120, c * 128:(c + 1) * 128]
        base = baseline[r * 120:(r + 1) * 120, c * 128:(c + 1) * 128]
        area[:] = np.clip(base + pixels, 0., 1.)
    return output


def grouped_rows(rows):
    """Conventional contiguous-run grouping with existing accepted row widths."""
    rows = validate_mask(rows)
    jobs = []
    for r, row in enumerate(rows):
        c = 0
        while c < 10:
            if row[c] == '0':
                c += 1
                continue
            start = c
            while c < 10 and row[c] == '1':
                c += 1
            width = next(w for w in WIDTHS if w >= (c - start) * 64)
            jobs.append(Job(min(start * 64, 640 - width), r * 60, width,
                            tuple(r * 10 + j for j in range(start, c))))
    return tuple(jobs)


def _cost(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('cost must be finite, nonnegative development evidence')
    return value


def per_row_choice(rows, estimate: Callable):
    """Choose exact/span/grouped independently per row using DEVELOPMENT cost.

    estimate must include the complete row service, not merely NPU duration.
    A full-request timing comparison still must include planning/shared setup.
    """
    rows = validate_mask(rows)
    jobs = []
    for r, row in enumerate(rows):
        if '1' not in row:
            continue
        one = tuple(row if i == r else '0' * 10 for i in range(6))
        candidates = (plan(one, 'exact'), plan(one, 'row_spans'), grouped_rows(one))
        jobs.extend(min(enumerate(candidates), key=lambda item: (_cost(estimate(item[1])), item[0]))[1])
    return tuple(jobs)


def profiled_pack(rows, halo, canvas_shapes, estimate: Callable):
    """Choose a catalog shelf plan with a frozen DEVELOPMENT-only service model.

    estimate receives the whole plan including edge fallbacks. Provenance and
    coefficient freeze are experiment-runner gates, not inferred by this API.
    """
    candidates = [pack(rows, halo, shape) for shape in canvas_shapes]
    if not candidates:
        raise ValueError('empty accepted shape catalog')
    return min(enumerate(candidates), key=lambda item: (_cost(estimate(item[1])), item[0]))[1]


def footprint(p):
    """Geometry accounting only: tensor element counts are NOT DRAM traffic."""
    validate_packing(p)
    patch_area = (64 + 2 * p.halo) * (60 + 2 * p.halo)
    area = sum(c.width * c.height for c in p.canvases)
    return {
        'selected_tiles': len(selected(p.rows)),
        'packed_tiles': sum(len(c.placements) for c in p.canvases),
        'boundary_exact_tiles': len(p.exact_boundary),
        'launches': len(p.canvases) + len(p.exact_boundary),
        'packed_canvas_pixels': area,
        'packed_context_pixels_copied': sum(len(c.placements) for c in p.canvases) * patch_area,
        'input_elements_including_exact': 3 * (area + len(p.exact_boundary) * patch_area),
        'output_elements_including_exact': 12 * (area + len(p.exact_boundary) * patch_area),
        'physical_service_time_measured': False,
    }
