"""Replay the frozen sparse-refinement contract on the physical RK3576.

This benchmark is intentionally narrower than the interactive Godot runtime.
It measures the deployable selector, exact binary work-list construction, fixed
shape RKNN calls, halo handling, residual reconstruction, and output quality on
the 300-frame internal replay. It does not claim live renderer integration,
zero-copy transport, independent generalization, or energy without those
separate measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deployment.board_probe import (  # noqa: E402
    collect_board_facts,
    read_thermal_zones,
    write_json_atomic,
)
from src.runtime.sparse_refinement import (  # noqa: E402
    TILE_FEATURE_NAMES,
    FrozenSparseTileSelector,
    SparseFrameWorkList,
    SparseWorkDescriptor,
    build_sparse_work_list,
)


@dataclass(frozen=True)
class TimingSummary:
    """Distribution and serial-equivalent FPS for one measured stage."""

    average_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    samples: int


class SparseShapeBackend(Protocol):
    """Minimal fixed-shape inference interface used by replay and tests."""

    supported_shapes: frozenset[tuple[int, int]]

    def infer(self, core_shape: tuple[int, int], input_rgb: np.ndarray) -> np.ndarray:
        """Return a 12-channel residual tensor for one halo-padded LR crop."""

    def warmup(self, iterations: int) -> None:
        """Warm every static graph outside the measured frame loop."""

    def close(self) -> None:
        """Release every model context owned by the backend."""


@dataclass(frozen=True)
class SparseDispatchGroup:
    """One physical NPU call plus the selected tiles it may commit."""

    inference_region: SparseWorkDescriptor
    commit_tiles: tuple[SparseWorkDescriptor, ...]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of one replay or model artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject malformed roots."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read deterministic nonempty JSON object rows."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL artifact contains no rows: {path}")
    return rows


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return one linearly interpolated percentile without SciPy."""

    if not values:
        raise ValueError("timing summary requires at least one sample")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def summarize_times(values_ms: Sequence[float]) -> TimingSummary:
    """Summarize one nonempty millisecond timing series."""

    average = statistics.fmean(values_ms)
    return TimingSummary(
        average_ms=average,
        median_ms=float(statistics.median(values_ms)),
        p95_ms=percentile(values_ms, 0.95),
        p99_ms=percentile(values_ms, 0.99),
        fps=1000.0 / average,
        samples=len(values_ms),
    )


def normalize_subpixel_residual(
    output: np.ndarray,
    *,
    input_width: int,
    input_height: int,
    quantization_scale: float | None,
    quantization_zero_point: int | None,
) -> np.ndarray:
    """PixelShuffle one NCHW/NHWC 12-channel residual into HWC RGB."""

    array = np.asarray(output)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.shape == (input_height, input_width, 12):
        array = np.transpose(array, (2, 0, 1))
    if array.shape != (12, input_height, input_width):
        raise ValueError(f"unexpected sparse NPU output shape: {array.shape}")
    if np.issubdtype(array.dtype, np.integer):
        if quantization_scale is None or quantization_zero_point is None:
            raise ValueError("integer sparse output requires RKNN quantization attributes")
        array = (
            array.astype(np.float32) - quantization_zero_point
        ) * quantization_scale
    array = array.astype(np.float32, copy=False)
    shuffled = (
        array.reshape(3, 2, 2, input_height, input_width)
        .transpose(0, 3, 1, 4, 2)
        .reshape(3, input_height * 2, input_width * 2)
    )
    # PixelShuffle's transpose produces a strided view. Materializing HWC once
    # is substantially cheaper on ARM than repeatedly clipping small tiles from
    # that strided view during residual composition.
    return np.ascontiguousarray(np.transpose(shuffled, (1, 2, 0)))


# Boundary descriptors contain a clipped source rectangle. Reconstruct the
# compile-time core+halo tensor with edge replication so every core pixel sees
# exactly the same receptive field convention as full-frame ESPCN inference.
def extract_halo_patch(
    lr_rgb: np.ndarray,
    descriptor: SparseWorkDescriptor,
    *,
    halo: int,
) -> np.ndarray:
    """Return one uint8 HWC patch with a complete fixed convolution halo."""

    frame_height, frame_width = lr_rgb.shape[:2]
    desired_x0 = descriptor.core_x - halo
    desired_y0 = descriptor.core_y - halo
    desired_x1 = descriptor.core_x + descriptor.core_width + halo
    desired_y1 = descriptor.core_y + descriptor.core_height + halo
    clipped_x0 = max(desired_x0, 0)
    clipped_y0 = max(desired_y0, 0)
    clipped_x1 = min(desired_x1, frame_width)
    clipped_y1 = min(desired_y1, frame_height)
    expected_descriptor = (
        clipped_x0,
        clipped_y0,
        clipped_x1 - clipped_x0,
        clipped_y1 - clipped_y0,
    )
    actual_descriptor = (
        descriptor.input_x,
        descriptor.input_y,
        descriptor.input_width,
        descriptor.input_height,
    )
    if actual_descriptor != expected_descriptor:
        raise ValueError("work descriptor input rectangle does not match its halo")
    crop = lr_rgb[clipped_y0:clipped_y1, clipped_x0:clipped_x1]
    padding = (
        (clipped_y0 - desired_y0, desired_y1 - clipped_y1),
        (clipped_x0 - desired_x0, desired_x1 - clipped_x1),
        (0, 0),
    )
    patch = np.pad(crop, padding, mode="edge")
    expected_shape = (
        descriptor.core_height + 2 * halo,
        descriptor.core_width + 2 * halo,
        3,
    )
    if patch.shape != expected_shape:
        raise RuntimeError(f"halo patch has the wrong shape: {patch.shape}")
    return np.ascontiguousarray(patch, dtype=np.uint8)


# Import OpenCV before warmup and cache the module. Importing it lazily inside
# the first measured resize previously charged several seconds of one-time
# shared-library startup to frame zero and distorted the arithmetic mean.
@lru_cache(maxsize=1)
def load_cv2() -> Any:
    """Return the cached OpenCV module used by bicubic reconstruction."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for sparse residual composition") from error
    return cv2


# Construct a protocol-valid descriptor for an execution rectangle. Its input
# fields describe the clipped source rectangle; extract_halo_patch then restores
# the complete compile-time tensor using edge replication.
def make_dispatch_descriptor(
    *,
    frame_id: int,
    frame_width: int,
    frame_height: int,
    core_x: int,
    core_y: int,
    core_width: int,
    core_height: int,
    halo: int,
    deadline_us: int,
    priority: float,
) -> SparseWorkDescriptor:
    """Create one checked inference descriptor for a coalesced rectangle."""

    input_x = max(core_x - halo, 0)
    input_y = max(core_y - halo, 0)
    input_x1 = min(core_x + core_width + halo, frame_width)
    input_y1 = min(core_y + core_height + halo, frame_height)
    return SparseWorkDescriptor(
        frame_id=frame_id,
        core_x=core_x,
        core_y=core_y,
        core_width=core_width,
        core_height=core_height,
        input_x=input_x,
        input_y=input_y,
        input_width=input_x1 - input_x,
        input_height=input_y1 - input_y,
        deadline_us=deadline_us,
        priority=priority,
        flags=1,
    )


# Recover the selected 64x60 macro-tiles from the frozen compact rectangles.
# Coalesced execution is allowed to compute extra pixels, but only these tiles
# may overwrite the immutable bicubic fallback.
def expand_selected_tiles(
    worklist: SparseFrameWorkList,
    *,
    halo: int,
) -> tuple[SparseWorkDescriptor, ...]:
    """Expand nonoverlapping frozen rectangles into their selected tile cells."""

    tiles: list[SparseWorkDescriptor] = []
    occupied: set[tuple[int, int]] = set()
    for descriptor in worklist.descriptors:
        if descriptor.core_x % worklist.tile_width != 0:
            raise ValueError("descriptor x coordinate is not tile aligned")
        if descriptor.core_y % worklist.tile_height != 0:
            raise ValueError("descriptor y coordinate is not tile aligned")
        if descriptor.core_width % worklist.tile_width != 0:
            raise ValueError("descriptor width is not a whole number of tiles")
        if descriptor.core_height % worklist.tile_height != 0:
            raise ValueError("descriptor height is not a whole number of tiles")
        for core_y in range(
            descriptor.core_y,
            descriptor.core_y + descriptor.core_height,
            worklist.tile_height,
        ):
            for core_x in range(
                descriptor.core_x,
                descriptor.core_x + descriptor.core_width,
                worklist.tile_width,
            ):
                coordinate = (core_x, core_y)
                if coordinate in occupied:
                    raise ValueError("frozen work-list descriptors overlap")
                occupied.add(coordinate)
                tiles.append(
                    make_dispatch_descriptor(
                        frame_id=worklist.frame_id,
                        frame_width=worklist.frame_width,
                        frame_height=worklist.frame_height,
                        core_x=core_x,
                        core_y=core_y,
                        core_width=worklist.tile_width,
                        core_height=worklist.tile_height,
                        halo=halo,
                        deadline_us=descriptor.deadline_us,
                        priority=descriptor.priority,
                    )
                )
    return tuple(sorted(tiles, key=lambda tile: (tile.core_y, tile.core_x)))


# Select the smallest compiled width that covers a row's selected span. The
# rectangle can include unselected gaps, but commit_tiles prevents those extra
# predictions from changing the rendered frame.
def choose_row_span(
    selected_tiles: Sequence[SparseWorkDescriptor],
    *,
    frame_width: int,
    supported_shapes: frozenset[tuple[int, int]],
) -> tuple[int, int]:
    """Return a supported x coordinate and width covering one selected row."""

    left = min(tile.core_x for tile in selected_tiles)
    right = max(tile.core_x + tile.core_width for tile in selected_tiles)
    widths = sorted(width for width, height in supported_shapes if height == 60)
    width = next((candidate for candidate in widths if candidate >= right - left), None)
    if width is None:
        raise ValueError("no compiled 60-pixel-high shape covers the selected row span")
    core_x = min(left, frame_width - width)
    core_x = max(core_x, right - width, 0)
    if core_x > left or core_x + width < right:
        raise RuntimeError("row-span placement does not cover every selected tile")
    return core_x, width


# Translate one immutable selector decision into alternative physical dispatch
# plans. This is a systems ablation: it changes NPU job granularity only and
# never changes which output tiles may receive a neural correction.
def build_dispatch_groups(
    worklist: SparseFrameWorkList,
    *,
    mode: str,
    halo: int,
    supported_shapes: frozenset[tuple[int, int]],
) -> tuple[SparseDispatchGroup, ...]:
    """Build exact, row-span, or half-frame NPU calls for one frozen packet."""

    selected_tiles = expand_selected_tiles(worklist, halo=halo)
    if mode == "exact":
        groups: list[SparseDispatchGroup] = []
        for descriptor in worklist.descriptors:
            commits = tuple(
                tile
                for tile in selected_tiles
                if descriptor.core_x <= tile.core_x
                and tile.core_x + tile.core_width
                <= descriptor.core_x + descriptor.core_width
                and descriptor.core_y <= tile.core_y
                and tile.core_y + tile.core_height
                <= descriptor.core_y + descriptor.core_height
            )
            groups.append(SparseDispatchGroup(descriptor, commits))
        return tuple(groups)

    if mode == "row_spans":
        groups = []
        for core_y in range(0, worklist.frame_height, worklist.tile_height):
            commits = tuple(tile for tile in selected_tiles if tile.core_y == core_y)
            if not commits:
                continue
            core_x, core_width = choose_row_span(
                commits,
                frame_width=worklist.frame_width,
                supported_shapes=supported_shapes,
            )
            inference = make_dispatch_descriptor(
                frame_id=worklist.frame_id,
                frame_width=worklist.frame_width,
                frame_height=worklist.frame_height,
                core_x=core_x,
                core_y=core_y,
                core_width=core_width,
                core_height=worklist.tile_height,
                halo=halo,
                deadline_us=min(tile.deadline_us for tile in commits),
                priority=max(tile.priority for tile in commits),
            )
            groups.append(SparseDispatchGroup(inference, commits))
        return tuple(groups)

    if mode == "half_frames":
        half_height = worklist.frame_height // 2
        if (worklist.frame_width, half_height) not in supported_shapes:
            raise ValueError("half-frame dispatch requires a compiled 640x180 shape")
        groups = []
        for core_y in (0, half_height):
            commits = tuple(
                tile
                for tile in selected_tiles
                if core_y <= tile.core_y < core_y + half_height
            )
            if not commits:
                continue
            inference = make_dispatch_descriptor(
                frame_id=worklist.frame_id,
                frame_width=worklist.frame_width,
                frame_height=worklist.frame_height,
                core_x=0,
                core_y=core_y,
                core_width=worklist.frame_width,
                core_height=half_height,
                halo=halo,
                deadline_us=min(tile.deadline_us for tile in commits),
                priority=max(tile.priority for tile in commits),
            )
            groups.append(SparseDispatchGroup(inference, commits))
        return tuple(groups)

    raise ValueError(f"unsupported sparse dispatch mode: {mode}")


class RKNNShapeFamilyBackend:
    """Load the accepted fixed-shape RKNN family on one selected NPU core."""

    def __init__(
        self,
        catalog_path: Path,
        core: str,
        output_scale: float | None,
        output_zero_point: int | None,
    ) -> None:
        """Verify the catalog, then initialize one RKNN Lite context per shape."""

        from rknnlite.api import RKNNLite

        catalog = read_json_object(catalog_path)
        if catalog.get("status") != "board_ready":
            raise ValueError("sparse RKNN catalog is not finalized for board use")
        core_masks = {
            "auto": RKNNLite.NPU_CORE_AUTO,
            "0": RKNNLite.NPU_CORE_0,
            "1": RKNNLite.NPU_CORE_1,
            "both": RKNNLite.NPU_CORE_0_1,
        }
        self.runtimes: dict[tuple[int, int], Any] = {}
        self.input_shapes: dict[tuple[int, int], tuple[int, int]] = {}
        self.supported_shapes: frozenset[tuple[int, int]] = frozenset()
        self.output_scale = output_scale
        self.output_zero_point = output_zero_point
        try:
            for entry in catalog["shapes"]:
                core_shape = (int(entry["core_width"]), int(entry["core_height"]))
                model_path = REPO_ROOT / str(entry["rknn"]["path"])
                if not model_path.is_file() or sha256_file(model_path) != entry["rknn"]["sha256"]:
                    raise ValueError(f"sparse RKNN identity mismatch: {model_path}")
                runtime = RKNNLite(verbose=False)
                if runtime.load_rknn(str(model_path)) != 0:
                    raise RuntimeError(f"failed to load sparse RKNN model: {model_path}")
                if runtime.init_runtime(core_mask=core_masks[core]) != 0:
                    runtime.release()
                    raise RuntimeError(f"failed to initialize sparse RKNN shape {core_shape}")
                self.runtimes[core_shape] = runtime
                self.input_shapes[core_shape] = (
                    int(entry["input_width"]),
                    int(entry["input_height"]),
                )
            self.supported_shapes = frozenset(self.runtimes)
        except Exception:
            self.close()
            raise

    def infer(self, core_shape: tuple[int, int], input_rgb: np.ndarray) -> np.ndarray:
        """Execute one static sparse graph and return its raw residual tensor."""

        if core_shape not in self.runtimes:
            raise ValueError(f"unsupported sparse RKNN core shape: {core_shape}")
        expected_width, expected_height = self.input_shapes[core_shape]
        if input_rgb.shape != (expected_height, expected_width, 3):
            raise ValueError(
                f"sparse RKNN {core_shape} expected {expected_width}x{expected_height}, "
                f"received {input_rgb.shape}"
            )
        outputs = self.runtimes[core_shape].inference(
            inputs=[input_rgb[None, ...]],
            data_format=["nhwc"],
        )
        if not outputs:
            raise RuntimeError("sparse RKNN inference returned no output")
        return np.asarray(outputs[0])

    # Warm every graph because RKNN performs lazy setup on its first inference.
    # Charging that setup to whichever shape appears first in a scene would make
    # dispatch-mode comparisons depend on frame order rather than architecture.
    def warmup(self, iterations: int) -> None:
        """Execute zero-valued inputs for every loaded static shape."""

        if iterations < 0:
            raise ValueError("warmup iterations must be nonnegative")
        for core_shape in sorted(self.runtimes):
            input_width, input_height = self.input_shapes[core_shape]
            sample = np.zeros((input_height, input_width, 3), dtype=np.uint8)
            for _ in range(iterations):
                self.infer(core_shape, sample)

    def close(self) -> None:
        """Release every initialized static-shape context."""

        for runtime in self.runtimes.values():
            runtime.release()
        self.runtimes.clear()


def build_worklist_from_replay(
    selector: FrozenSparseTileSelector,
    selector_artifact: Mapping[str, Any],
    row: Mapping[str, Any],
) -> SparseFrameWorkList:
    """Execute selector arithmetic and recreate one frozen binary packet."""

    feature_rows = np.asarray(row.get("tile_feature_vectors"), dtype=np.float32)
    if feature_rows.shape != (60, len(TILE_FEATURE_NAMES)):
        raise ValueError(f"replay tile features have the wrong shape: {feature_rows.shape}")
    probabilities = np.asarray(
        [selector.predict_probability(features) for features in feature_rows],
        dtype=np.float64,
    ).reshape(6, 10)
    selected = probabilities >= selector.selection_threshold
    deployment = selector_artifact["deployment"]
    macro_tiles = selector_artifact["macro_tiles"]
    supported = frozenset(
        (int(shape[0]), int(shape[1]))
        for shape in selector_artifact["supported_core_shapes"]
    )
    result = build_sparse_work_list(
        selected,
        probabilities,
        frame_id=int(row["frame_index"]),
        frame_width=640,
        frame_height=360,
        tile_width=int(macro_tiles["lr_tile_width"]),
        tile_height=int(macro_tiles["lr_tile_height"]),
        halo=int(macro_tiles["lr_halo_pixels"]),
        deadline_us=int(deployment["frame_deadline_us"]),
        elapsed_us=0,
        safety_margin_us=int(deployment["scheduler_safety_margin_us"]),
        full_frame_compute_us=float(deployment["full_frame_npu_driver_us"]),
        queue_depth=0,
        maximum_queue_depth=int(deployment["maximum_queue_depth"]),
        temperature_c=None,
        thermal_soft_limit_c=float(deployment["thermal_soft_limit_c"]),
        thermal_hard_limit_c=float(deployment["thermal_hard_limit_c"]),
        supported_core_shapes=supported,
    )
    return result.work_list


def resize_bicubic(lr_rgb: np.ndarray) -> np.ndarray:
    """Return the model-matched 2x OpenCV bicubic base in float RGB."""

    cv2 = load_cv2()
    return cv2.resize(
        lr_rgb.astype(np.float32) / 255.0,
        (lr_rgb.shape[1] * 2, lr_rgb.shape[0] * 2),
        interpolation=cv2.INTER_CUBIC,
    )


def psnr_from_float(output: np.ndarray, reference_rgb: np.ndarray) -> float:
    """Return RGB PSNR for a clipped float output and uint8 reference."""

    error = output.astype(np.float64) - reference_rgb.astype(np.float64) / 255.0
    mse = float(np.mean(error * error))
    return math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


# Execute every descriptor against the immutable bicubic base. Regions never
# overlap after compaction, so descriptor order affects deadlines but not final
# pixels; this is checked by exact packet replay before inference begins.
def run_sparse_frame(
    lr_rgb: np.ndarray,
    worklist: SparseFrameWorkList,
    backend: SparseShapeBackend,
    *,
    halo: int,
    output_scale: float | None,
    output_zero_point: int | None,
    dispatch_mode: str = "exact",
    supported_shapes: frozenset[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one sparse frame and return output plus stage-level measurements."""

    frame_start = time.perf_counter_ns()
    if supported_shapes is None:
        supported_shapes = getattr(
            backend,
            "supported_shapes",
            frozenset(
                (descriptor.core_width, descriptor.core_height)
                for descriptor in worklist.descriptors
            ),
        )
    planning_start = time.perf_counter_ns()
    dispatch_groups = build_dispatch_groups(
        worklist,
        mode=dispatch_mode,
        halo=halo,
        supported_shapes=supported_shapes,
    )
    planning_ms = (time.perf_counter_ns() - planning_start) / 1_000_000.0
    baseline_start = time.perf_counter_ns()
    baseline = resize_bicubic(lr_rgb)
    baseline_ms = (time.perf_counter_ns() - baseline_start) / 1_000_000.0
    # OpenCV's cubic kernel can overshoot the legal RGB range near sharp
    # transitions. Clamp the immutable fallback everywhere so selected and
    # unselected regions share the same output contract at their boundary.
    output = np.clip(baseline, 0.0, 1.0)
    extraction_ms = 0.0
    npu_ms = 0.0
    reconstruction_ms = 0.0
    input_bytes = 0
    output_bytes = 0
    shapes: Counter[str] = Counter()
    shape_npu_ms: Counter[str] = Counter()
    selected_core_pixels = 0
    dispatched_core_pixels = 0
    for group in dispatch_groups:
        descriptor = group.inference_region
        extraction_start = time.perf_counter_ns()
        patch = extract_halo_patch(lr_rgb, descriptor, halo=halo)
        extraction_ms += (time.perf_counter_ns() - extraction_start) / 1_000_000.0
        core_shape = (descriptor.core_width, descriptor.core_height)
        shape_name = f"{core_shape[0]}x{core_shape[1]}"
        npu_start = time.perf_counter_ns()
        raw_residual = backend.infer(core_shape, patch)
        call_ms = (time.perf_counter_ns() - npu_start) / 1_000_000.0
        npu_ms += call_ms
        shape_npu_ms[shape_name] += call_ms
        reconstruct_start = time.perf_counter_ns()
        residual = normalize_subpixel_residual(
            raw_residual,
            input_width=patch.shape[1],
            input_height=patch.shape[0],
            quantization_scale=output_scale,
            quantization_zero_point=output_zero_point,
        )
        crop_x = halo * 2
        crop_y = halo * 2
        output_x = descriptor.core_x * 2
        output_y = descriptor.core_y * 2
        output_width = descriptor.core_width * 2
        output_height = descriptor.core_height * 2
        core_residual = residual[
            crop_y : crop_y + output_height,
            crop_x : crop_x + output_width,
        ]
        region = (
            slice(output_y, output_y + output_height),
            slice(output_x, output_x + output_width),
        )
        commit_mask = np.zeros((output_height, output_width), dtype=bool)
        for commit_tile in group.commit_tiles:
            local_x = (commit_tile.core_x - descriptor.core_x) * 2
            local_y = (commit_tile.core_y - descriptor.core_y) * 2
            commit_mask[
                local_y : local_y + commit_tile.core_height * 2,
                local_x : local_x + commit_tile.core_width * 2,
            ] = True
            selected_core_pixels += commit_tile.core_width * commit_tile.core_height
        corrected = np.clip(baseline[region] + core_residual, 0.0, 1.0)
        if bool(np.all(commit_mask)):
            output[region] = corrected
        else:
            np.copyto(
                output[region],
                corrected,
                where=np.broadcast_to(commit_mask[:, :, None], corrected.shape),
            )
        reconstruction_ms += (time.perf_counter_ns() - reconstruct_start) / 1_000_000.0
        input_bytes += int(patch.nbytes)
        output_bytes += int(np.asarray(raw_residual).nbytes)
        dispatched_core_pixels += descriptor.core_width * descriptor.core_height
        shapes[shape_name] += 1
    compute_ms = (time.perf_counter_ns() - frame_start) / 1_000_000.0
    return output, {
        "dispatch_mode": dispatch_mode,
        "dispatch_planning_ms": planning_ms,
        "bicubic_ms": baseline_ms,
        "halo_extraction_ms": extraction_ms,
        "npu_ms": npu_ms,
        "reconstruction_ms": reconstruction_ms,
        "compute_pipeline_ms": compute_ms,
        "npu_calls": len(dispatch_groups),
        "npu_input_bytes": input_bytes,
        "npu_output_bytes": output_bytes,
        "shape_calls": dict(shapes),
        "shape_npu_ms": dict(shape_npu_ms),
        "selected_core_pixels": selected_core_pixels,
        "dispatched_core_pixels": dispatched_core_pixels,
    }


# Preflight every replay row without loading RKNN Lite. This is deliberately
# exhaustive: a single missing frame, stale feature row, changed packet, or
# unsupported compacted shape should block board transfer rather than fail in
# the middle of a 300-frame physical run.
def validate_replay_package(
    selector_path: Path,
    selector_report_path: Path,
    catalog_path: Path,
    replay_path: Path,
) -> dict[str, Any]:
    from PIL import Image
    """Verify all board-independent identities and geometry in the replay."""

    selector_artifact = read_json_object(selector_path)
    selector_report = read_json_object(selector_report_path)
    catalog = read_json_object(catalog_path)
    if catalog.get("status") != "board_ready":
        raise ValueError("sparse RKNN catalog is not board-ready")
    if catalog.get("selector", {}).get("sha256") != sha256_file(selector_path):
        raise ValueError("catalog and selector hashes disagree")
    if not selector_report.get("decision", {}).get("offline_checks_passed"):
        raise ValueError("selector report did not pass offline checks")
    selector = FrozenSparseTileSelector(selector_artifact)
    rows = read_jsonl(replay_path)
    if len(rows) != 300:
        raise ValueError(f"physical sparse replay requires 300 frames, found {len(rows)}")
    supported_shapes = {
        (int(entry["core_width"]), int(entry["core_height"]))
        for entry in catalog["shapes"]
    }
    expected_shapes = {
        (int(shape[0]), int(shape[1]))
        for shape in selector_artifact["supported_core_shapes"]
    }
    if supported_shapes != expected_shapes:
        raise ValueError("selector and RKNN shape dictionaries disagree")
    packet_bytes = 0
    descriptor_count = 0
    shape_counts: Counter[str] = Counter()
    sequences: set[str] = set()
    engines: set[str] = set()
    for row_index, row in enumerate(rows):
        if int(row["frame_index"]) != row_index:
            raise ValueError("replay frame indices are not contiguous")
        worklist = build_worklist_from_replay(selector, selector_artifact, row)
        packet_path = REPO_ROOT / str(row["worklist_path"])
        packet = packet_path.read_bytes()
        if sha256_file(packet_path) != row["worklist_sha256"]:
            raise ValueError(f"replay work-list hash mismatch: {packet_path}")
        if worklist.pack() != packet:
            raise RuntimeError(f"portable selector changed frozen packet at row {row_index}")
        decoded = SparseFrameWorkList.unpack(packet)
        for descriptor in decoded.descriptors:
            shape = (descriptor.core_width, descriptor.core_height)
            if shape not in supported_shapes:
                raise ValueError(f"work descriptor has no compiled RKNN shape: {shape}")
            shape_counts[f"{shape[0]}x{shape[1]}"] += 1
        for path_field, expected_size in (("lr_path", (640, 360)), ("hr_path", (1280, 720))):
            image_path = REPO_ROOT / str(row[path_field])
            with Image.open(image_path) as image:
                if image.size != expected_size:
                    raise ValueError(
                        f"{path_field} has size {image.size}, expected {expected_size}: {image_path}"
                    )
        packet_bytes += len(packet)
        descriptor_count += len(decoded.descriptors)
        sequences.add(str(row["sequence_id"]))
        engines.add(str(row["engine_id"]))
    return {
        "schema_version": 1,
        "experiment": "sparse_refinement_v2_board_payload_preflight",
        "physical_rknn_execution_measured": False,
        "all_checks_passed": True,
        "frames": len(rows),
        "sequences": len(sequences),
        "engines": len(engines),
        "worklist_packets": len(rows),
        "descriptor_count": descriptor_count,
        "packet_bytes": packet_bytes,
        "shape_counts": dict(sorted(shape_counts.items())),
        "selector_sha256": sha256_file(selector_path),
        "selector_report_sha256": sha256_file(selector_report_path),
        "catalog_sha256": sha256_file(catalog_path),
        "replay_sha256": sha256_file(replay_path),
        "next_boundary": "physical_rk3576_execution",
    }


def run_benchmark(args: argparse.Namespace, backend: SparseShapeBackend | None = None) -> dict[str, Any]:
    from PIL import Image
    """Run the full deterministic board replay and return its result artifact."""

    selector_path = args.selector.resolve()
    report_path = args.selector_report.resolve()
    catalog_path = args.catalog.resolve()
    replay_path = args.replay.resolve()
    selector_artifact = read_json_object(selector_path)
    selector_report = read_json_object(report_path)
    catalog = read_json_object(catalog_path)
    if catalog.get("status") != "board_ready":
        raise ValueError("sparse RKNN catalog is not board-ready")
    if catalog.get("selector", {}).get("sha256") != sha256_file(selector_path):
        raise ValueError("catalog and selector hashes disagree")
    if not selector_report.get("decision", {}).get("offline_checks_passed"):
        raise ValueError("selector report did not pass offline checks")
    rows = read_jsonl(replay_path)
    if len(rows) != 300:
        raise ValueError(f"physical sparse replay requires 300 frames, found {len(rows)}")
    selector = FrozenSparseTileSelector(selector_artifact)
    halo = int(selector_artifact["macro_tiles"]["lr_halo_pixels"])
    owns_backend = backend is None
    if backend is None:
        backend = RKNNShapeFamilyBackend(
            catalog_path,
            args.npu_core,
            args.output_scale,
            args.output_zero_point,
        )
    supported_shapes = getattr(
        backend,
        "supported_shapes",
        frozenset(
            (int(entry["core_width"]), int(entry["core_height"]))
            for entry in catalog["shapes"]
        ),
    )

    selector_times: list[float] = []
    contract_times: list[float] = []
    decode_times: list[float] = []
    planning_times: list[float] = []
    extraction_times: list[float] = []
    npu_times: list[float] = []
    reconstruction_times: list[float] = []
    bicubic_times: list[float] = []
    compute_times: list[float] = []
    online_times: list[float] = []
    service_times: list[float] = []
    sparse_psnr: list[float] = []
    bicubic_psnr: list[float] = []
    frame_hashes: list[str] = []
    shape_calls: Counter[str] = Counter()
    npu_calls = 0
    npu_input_bytes = 0
    npu_output_bytes = 0
    selected_core_pixels = 0
    dispatched_core_pixels = 0
    shape_npu_ms: Counter[str] = Counter()

    # Warm OpenCV and every static RKNN shape before reading thermal baselines
    # or starting frame timers. This matches the established 20-warmup thesis
    # convention while preventing first-use library setup from polluting frame 0.
    first_lr_path = REPO_ROOT / str(rows[0]["lr_path"])
    with Image.open(first_lr_path) as image:
        warmup_lr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    resize_bicubic(warmup_lr)
    warmup = getattr(backend, "warmup", None)
    if warmup is not None:
        warmup(args.warmup_iterations)

    start_thermal = read_thermal_zones()
    try:
        for row_index, row in enumerate(rows):
            service_start = time.perf_counter_ns()
            selector_start = time.perf_counter_ns()
            worklist = build_worklist_from_replay(selector, selector_artifact, row)
            selector_ms = (time.perf_counter_ns() - selector_start) / 1_000_000.0
            contract_start = time.perf_counter_ns()
            packet_path = REPO_ROOT / str(row["worklist_path"])
            expected_packet = packet_path.read_bytes()
            if sha256_file(packet_path) != row["worklist_sha256"]:
                raise ValueError(f"replay work-list hash mismatch: {packet_path}")
            if worklist.pack() != expected_packet:
                raise RuntimeError(f"board selector changed frozen work-list at row {row_index}")
            decoded = SparseFrameWorkList.unpack(expected_packet)
            contract_ms = (time.perf_counter_ns() - contract_start) / 1_000_000.0
            lr_path = REPO_ROOT / str(row["lr_path"])
            hr_path = REPO_ROOT / str(row["hr_path"])
            decode_start = time.perf_counter_ns()
            with Image.open(lr_path) as image:
                lr_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            with Image.open(hr_path) as image:
                hr_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            decode_ms = (time.perf_counter_ns() - decode_start) / 1_000_000.0
            output, frame_stats = run_sparse_frame(
                lr_rgb,
                decoded,
                backend,
                halo=halo,
                output_scale=args.output_scale,
                output_zero_point=args.output_zero_point,
                dispatch_mode=args.dispatch_mode,
                supported_shapes=supported_shapes,
            )
            service_ms = (time.perf_counter_ns() - service_start) / 1_000_000.0
            online_ms = selector_ms + float(frame_stats["compute_pipeline_ms"])
            selector_times.append(selector_ms)
            contract_times.append(contract_ms)
            decode_times.append(decode_ms)
            planning_times.append(float(frame_stats["dispatch_planning_ms"]))
            extraction_times.append(float(frame_stats["halo_extraction_ms"]))
            npu_times.append(float(frame_stats["npu_ms"]))
            reconstruction_times.append(float(frame_stats["reconstruction_ms"]))
            bicubic_times.append(float(frame_stats["bicubic_ms"]))
            compute_times.append(float(frame_stats["compute_pipeline_ms"]))
            online_times.append(online_ms)
            service_times.append(service_ms)
            sparse_psnr.append(psnr_from_float(output, hr_rgb))
            bicubic = resize_bicubic(lr_rgb)
            bicubic_psnr.append(psnr_from_float(np.clip(bicubic, 0.0, 1.0), hr_rgb))
            output_u8 = np.rint(np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)
            frame_hashes.append(hashlib.sha256(output_u8.tobytes()).hexdigest())
            npu_calls += int(frame_stats["npu_calls"])
            npu_input_bytes += int(frame_stats["npu_input_bytes"])
            npu_output_bytes += int(frame_stats["npu_output_bytes"])
            selected_core_pixels += int(frame_stats["selected_core_pixels"])
            dispatched_core_pixels += int(frame_stats["dispatched_core_pixels"])
            shape_calls.update(frame_stats["shape_calls"])
            shape_npu_ms.update(frame_stats["shape_npu_ms"])
            if args.save_outputs_dir is not None and row_index < args.save_output_count:
                args.save_outputs_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(output_u8, mode="RGB").save(
                    args.save_outputs_dir / f"sparse_{row_index:04d}.png"
                )
    finally:
        if owns_backend:
            backend.close()
    end_thermal = read_thermal_zones()
    service = summarize_times(service_times)
    online = summarize_times(online_times)
    deadline_ms = float(selector_artifact["deployment"]["frame_deadline_us"]) / 1000.0
    offline_misses = sum(value > deadline_ms for value in service_times)
    online_misses = sum(value > deadline_ms for value in online_times)
    shape_profiles = {
        shape: {
            "calls": int(shape_calls[shape]),
            "total_ms": float(shape_npu_ms[shape]),
            "average_ms_per_call": float(shape_npu_ms[shape] / shape_calls[shape]),
        }
        for shape in sorted(shape_calls)
    }
    report = {
        "schema_version": 2,
        "experiment": "rk3576_sparse_refinement_v2_physical_dispatch_ablation",
        "board": collect_board_facts(),
        "evidence_boundary": {
            "physical_rknn_execution_measured": True,
            "live_godot_integration_measured": False,
            "independent_quality_claim_authorized": False,
            "power_or_energy_measured": False,
        },
        "artifacts": {
            "selector": {"path": str(args.selector), "sha256": sha256_file(selector_path)},
            "selector_report": {"path": str(args.selector_report), "sha256": sha256_file(report_path)},
            "catalog": {"path": str(args.catalog), "sha256": sha256_file(catalog_path)},
            "replay": {"path": str(args.replay), "sha256": sha256_file(replay_path)},
        },
        "workload": {
            "frames": len(rows),
            "input_resolution": [640, 360],
            "output_resolution": [1280, 720],
            "npu_core": args.npu_core,
            "dispatch_mode": args.dispatch_mode,
            "warmup_iterations_per_shape": args.warmup_iterations,
            "npu_calls": npu_calls,
            "npu_calls_per_frame": npu_calls / len(rows),
            "shape_calls": dict(sorted(shape_calls.items())),
            "shape_npu_profiles": shape_profiles,
            "npu_input_bytes": npu_input_bytes,
            "npu_output_bytes": npu_output_bytes,
            "selected_core_pixels": selected_core_pixels,
            "dispatched_core_pixels": dispatched_core_pixels,
            "extra_compute_ratio": (
                dispatched_core_pixels / selected_core_pixels - 1.0
                if selected_core_pixels
                else 0.0
            ),
        },
        "timing": {
            "selector_and_exact_worklist": asdict(summarize_times(selector_times)),
            "frozen_packet_validation": asdict(summarize_times(contract_times)),
            "png_input_and_reference_decode": asdict(summarize_times(decode_times)),
            "dispatch_planning": asdict(summarize_times(planning_times)),
            "bicubic_base": asdict(summarize_times(bicubic_times)),
            "halo_extraction": asdict(summarize_times(extraction_times)),
            "npu_calls": asdict(summarize_times(npu_times)),
            "residual_reconstruction": asdict(summarize_times(reconstruction_times)),
            "compute_pipeline_excluding_selector": asdict(summarize_times(compute_times)),
            "estimated_live_in_memory_pipeline": asdict(online),
            "complete_offline_service": asdict(service),
            "deadline_ms": deadline_ms,
            "estimated_live_deadline_misses": online_misses,
            "estimated_live_deadline_miss_rate": online_misses / len(rows),
            "offline_deadline_misses": offline_misses,
            "offline_deadline_miss_rate": offline_misses / len(rows),
        },
        "quality": {
            "sparse_mean_psnr_db": statistics.fmean(sparse_psnr),
            "bicubic_mean_psnr_db": statistics.fmean(bicubic_psnr),
            "sparse_gain_vs_bicubic_db": statistics.fmean(sparse_psnr) - statistics.fmean(bicubic_psnr),
            "output_hashes": frame_hashes,
        },
        "thermal": {"start": start_thermal, "end": end_thermal},
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the physical sparse RK3576 replay command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--selector-report", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npu-core", choices=("auto", "0", "1", "both"), default="auto")
    parser.add_argument(
        "--dispatch-mode",
        choices=("exact", "row_spans", "half_frames"),
        default="exact",
        help="Physical NPU grouping applied after recreating the frozen selector packet.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=2,
        help="Unmeasured inference warmups performed for every static RKNN shape.",
    )
    parser.add_argument("--output-scale", type=float)
    parser.add_argument("--output-zero-point", type=int)
    parser.add_argument("--save-outputs-dir", type=Path)
    parser.add_argument("--save-output-count", type=int, default=6)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Verify the entire board payload without loading RKNN Lite.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from PIL import Image  # noqa: F401 - required only by the historical CLI path
    """Run one physical sparse replay and save its checksummed result."""

    args = build_parser().parse_args(argv)
    if args.save_output_count < 0:
        print("error: --save-output-count must be nonnegative", file=sys.stderr)
        return 1
    if args.warmup_iterations < 0:
        print("error: --warmup-iterations must be nonnegative", file=sys.stderr)
        return 1
    if args.save_outputs_dir is not None:
        args.save_outputs_dir = args.save_outputs_dir.resolve()
    try:
        if args.validate_only:
            report = validate_replay_package(
                args.selector.resolve(),
                args.selector_report.resolve(),
                args.catalog.resolve(),
                args.replay.resolve(),
            )
        else:
            report = run_benchmark(args)
        write_json_atomic(args.output.resolve(), report, args.overwrite)
    except (FileExistsError, FileNotFoundError, ImportError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.validate_only:
        print(
            "RK3576_SPARSE_PREFLIGHT_PASS "
            f"frames={report['frames']} "
            f"descriptors={report['descriptor_count']} "
            f"next={report['next_boundary']}"
        )
        return 0
    timing = report["timing"]["estimated_live_in_memory_pipeline"]
    print(
        "RK3576_SPARSE_RESULT "
        f"mode={report['workload']['dispatch_mode']} "
        f"fps={timing['fps']:.3f} "
        f"p95_ms={timing['p95_ms']:.3f} "
        f"deadline_misses={report['timing']['estimated_live_deadline_misses']} "
        f"gain_vs_bicubic_db={report['quality']['sparse_gain_vs_bicubic_db']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
