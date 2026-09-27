"""Zero-padded sparse execution matching full-frame convolution boundaries."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import numpy as np

from src.deployment.benchmark_rk3576_sparse import (
    SparseFrameWorkList,
    build_dispatch_groups,
    normalize_subpixel_residual,
    resize_bicubic,
)


def extract_zero_halo_patch(lr_rgb: np.ndarray, descriptor: Any, *, halo: int) -> tuple[np.ndarray, int, int]:
    height, width = lr_rgb.shape[:2]
    desired_x0, desired_y0 = descriptor.core_x - halo, descriptor.core_y - halo
    desired_x1 = descriptor.core_x + descriptor.core_width + halo
    desired_y1 = descriptor.core_y + descriptor.core_height + halo
    x0, y0, x1, y1 = max(desired_x0, 0), max(desired_y0, 0), min(desired_x1, width), min(desired_y1, height)
    if (descriptor.input_x, descriptor.input_y, descriptor.input_width, descriptor.input_height) != (x0, y0, x1 - x0, y1 - y0):
        raise ValueError("work descriptor input rectangle does not match zero halo")
    crop = lr_rgb[y0:y1, x0:x1]
    if descriptor.core_x == 0 and descriptor.core_x + descriptor.core_width == width:
        patch = np.ascontiguousarray(crop, dtype=np.uint8)
        core_y = descriptor.core_y - y0
        return patch, 0, core_y
    patch = np.zeros((descriptor.core_height + 2 * halo, descriptor.core_width + 2 * halo, 3), dtype=np.uint8)
    # Align a clipped global boundary with the graph boundary so the graph's
    # own per-layer zero padding, rather than activations computed in an
    # artificial padded image domain, defines that edge.  If a 640-wide core
    # touches both horizontal boundaries, left alignment is deterministic;
    # the offline gate below decides whether the unrepresented opposite-edge
    # condition is acceptable (it currently must pass the frozen limits).
    insert_x = patch.shape[1] - crop.shape[1] if descriptor.core_x + descriptor.core_width == width and descriptor.core_x != 0 else 0
    insert_y = patch.shape[0] - crop.shape[0] if descriptor.core_y + descriptor.core_height == height and descriptor.core_y != 0 else 0
    patch[insert_y:insert_y + crop.shape[0], insert_x:insert_x + crop.shape[1]] = crop
    core_x = insert_x + descriptor.core_x - x0
    core_y = insert_y + descriptor.core_y - y0
    expected = (descriptor.core_height + 2 * halo, descriptor.core_width + 2 * halo, 3)
    if patch.shape != expected:
        raise RuntimeError("zero-padded halo patch shape changed")
    return np.ascontiguousarray(patch, dtype=np.uint8), core_x, core_y


def run_sparse_frame_zero_padded(lr_rgb: np.ndarray, worklist: SparseFrameWorkList, backend: Any, *, halo: int, dispatch_mode: str, supported_shapes: frozenset[tuple[int, int]]) -> tuple[np.ndarray, dict[str, Any]]:
    frame_start = time.perf_counter_ns()
    planning_start = time.perf_counter_ns()
    groups = build_dispatch_groups(worklist, mode=dispatch_mode, halo=halo, supported_shapes=supported_shapes)
    planning_ms = (time.perf_counter_ns() - planning_start) / 1e6
    base_start = time.perf_counter_ns()
    baseline = resize_bicubic(lr_rgb)
    base_ms = (time.perf_counter_ns() - base_start) / 1e6
    output = np.clip(baseline, 0.0, 1.0)
    extraction_ms = inference_ms = reconstruction_ms = 0.0
    input_bytes = output_bytes = selected_pixels = dispatched_pixels = 0
    shapes: Counter[str] = Counter()
    for group in groups:
        descriptor = group.inference_region
        start = time.perf_counter_ns(); patch, core_offset_x, core_offset_y = extract_zero_halo_patch(lr_rgb, descriptor, halo=halo); extraction_ms += (time.perf_counter_ns() - start) / 1e6
        shape = (descriptor.core_width, descriptor.core_height)
        boundary = None
        if descriptor.core_x == 0 and descriptor.core_x + descriptor.core_width == lr_rgb.shape[1]:
            boundary = "full_width_top" if descriptor.core_y == 0 else "full_width_bottom" if descriptor.core_y + descriptor.core_height == lr_rgb.shape[0] else "full_width_middle"
        start = time.perf_counter_ns()
        raw = backend.infer_boundary(boundary, shape, patch) if boundary is not None else backend.infer(shape, patch)
        inference_ms += (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        residual = normalize_subpixel_residual(raw, input_width=patch.shape[1], input_height=patch.shape[0], quantization_scale=None, quantization_zero_point=None)
        region = (slice(descriptor.core_y * 2, (descriptor.core_y + descriptor.core_height) * 2), slice(descriptor.core_x * 2, (descriptor.core_x + descriptor.core_width) * 2))
        core = residual[core_offset_y * 2:core_offset_y * 2 + descriptor.core_height * 2, core_offset_x * 2:core_offset_x * 2 + descriptor.core_width * 2]
        commit = np.zeros((descriptor.core_height * 2, descriptor.core_width * 2), dtype=bool)
        for tile in group.commit_tiles:
            x, y = (tile.core_x - descriptor.core_x) * 2, (tile.core_y - descriptor.core_y) * 2
            commit[y:y + tile.core_height * 2, x:x + tile.core_width * 2] = True
            selected_pixels += tile.core_width * tile.core_height
        corrected = np.clip(baseline[region] + core, 0.0, 1.0)
        np.copyto(output[region], corrected, where=np.broadcast_to(commit[:, :, None], corrected.shape))
        reconstruction_ms += (time.perf_counter_ns() - start) / 1e6
        input_bytes += patch.nbytes; output_bytes += np.asarray(raw).nbytes; dispatched_pixels += descriptor.core_width * descriptor.core_height
        shapes[f"{shape[0]}x{shape[1]}"] += 1
    return output, {
        "dispatch_mode": dispatch_mode, "dispatch_planning_ms": planning_ms, "bicubic_ms": base_ms,
        "halo_extraction_ms": extraction_ms, "npu_ms": inference_ms, "reconstruction_ms": reconstruction_ms,
        "compute_pipeline_ms": (time.perf_counter_ns() - frame_start) / 1e6, "npu_calls": len(groups),
        "npu_input_bytes": int(input_bytes), "npu_output_bytes": int(output_bytes), "shape_calls": dict(shapes),
        "selected_core_pixels": selected_pixels, "dispatched_core_pixels": dispatched_pixels,
    }
