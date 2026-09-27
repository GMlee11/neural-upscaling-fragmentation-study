"""Independent synthetic fixture, frozen masks, and worklists for v2 runners."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.deployment.benchmark_rk3576_sparse import SparseFrameWorkList, make_dispatch_descriptor


@dataclass(frozen=True)
class FrozenMask:
    mask_id: str
    pair_id: str
    geometry: str
    density_percent: int
    selected_tiles: int
    active_rows: int
    horizontal_runs: int
    row_span_coverage_tiles: int
    mask_hex: str
    rows: tuple[str, ...]
    grid: np.ndarray


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def generate_coordinate_fixture(contract: Mapping[str, Any]) -> np.ndarray:
    spec = contract["input_fixture"]
    height, width = int(spec["height"]), int(spec["width"])
    y, x = np.indices((height, width), dtype=np.int64)
    fixture = np.empty((height, width, 3), dtype=np.uint8)
    fixture[..., 0] = (3 * x + 5 * y + 17) % 256
    fixture[..., 1] = (7 * x + 11 * y + 29) % 256
    fixture[..., 2] = (13 * x + 2 * y + 47) % 256
    if fixture.nbytes != int(spec["byte_length"]) or hashlib.sha256(fixture.tobytes()).hexdigest() != spec["raw_bytes_sha256"]:
        raise ValueError("synthetic coordinate fixture identity changed")
    return fixture


def generate_progressive_fixture() -> np.ndarray:
    """Return the independently frozen Progressive A+B+C coordinate input."""

    y, x = np.indices((360, 640), dtype=np.uint32)
    fixture = np.stack(
        ((5 * x + 3 * y + 71) % 256, (11 * x + 7 * y + 43) % 256, (2 * x + 13 * y + 19) % 256),
        axis=-1,
    ).astype(np.uint8)
    expected = "88854a9e2b5b670aa89ab09548b4d11b43a4f1c9c628b4f21a3f16177fe3a66f"
    if fixture.nbytes != 691200 or hashlib.sha256(fixture.tobytes(order="C")).hexdigest() != expected:
        raise ValueError("progressive synthetic fixture identity changed")
    return fixture


def _runs(row: str) -> int:
    return sum(bit == "1" and (index == 0 or row[index - 1] == "0") for index, bit in enumerate(row))


def _span(row: str) -> int:
    positions = [index for index, bit in enumerate(row) if bit == "1"]
    if not positions:
        raise ValueError("every frozen row must be active")
    return positions[-1] - positions[0] + 1


def _integer(rows: Sequence[str]) -> int:
    value = 0
    for row_index, row in enumerate(rows):
        for column_index, bit in enumerate(row):
            if bit == "1":
                value |= 1 << (row_index * 10 + column_index)
    return value


def load_frozen_masks(contract: Mapping[str, Any]) -> tuple[FrozenMask, ...]:
    spec = contract["mask_contract"]
    sources = spec["masks"]
    if len(sources) != 6 or int(spec["mask_count"]) != 6:
        raise ValueError("exactly six masks required")
    rows_by_id = {str(row["id"]): row["rows"] for row in sources}
    if _canonical_sha256(rows_by_id) != str(spec["canonical_rows_sha256"]):
        raise ValueError("frozen mask-row hash changed")
    masks = []
    for source in sources:
        rows = tuple(str(row) for row in source["rows"])
        if len(rows) != 6 or any(len(row) != 10 or set(row) - {"0", "1"} for row in rows):
            raise ValueError("invalid frozen mask")
        selected = sum(row.count("1") for row in rows)
        runs = sum(_runs(row) for row in rows)
        coverage = sum(_span(row) for row in rows)
        if (selected != int(source["selected_tiles"]) or runs != int(source["horizontal_runs"])
                or coverage != int(source["row_span_coverage_tiles"])
                or _integer(rows) != int(str(source["mask_hex"]), 16)):
            raise ValueError("frozen mask accounting changed")
        grid = np.asarray([[bit == "1" for bit in row] for row in rows], dtype=bool)
        masks.append(FrozenMask(str(source["id"]), str(source["pair_id"]), str(source["geometry"]),
            int(source["density_percent"]), selected, int(np.count_nonzero(np.any(grid, axis=1))),
            runs, coverage, str(source["mask_hex"]), rows, grid))
    by_pair: dict[str, list[FrozenMask]] = {}
    for mask in masks:
        by_pair.setdefault(mask.pair_id, []).append(mask)
    if len(by_pair) != 3 or any(len(pair) != 2 or len({m.selected_tiles for m in pair}) != 1
            or any(m.active_rows != 6 for m in pair) for pair in by_pair.values()):
        raise ValueError("frozen compact/fragmented pairing changed")
    return tuple(masks)


def build_exact_worklist(mask: FrozenMask, *, frame_id: int, contract: Mapping[str, Any]) -> SparseFrameWorkList:
    workload = contract["workload"]
    tile_width, tile_height = map(int, workload["macro_tile_lr"])
    frame_width, frame_height = map(int, workload["input_resolution"])
    halo = int(workload["halo_pixels"])
    deadline_us = int(round(float(workload["deadline_ms"]) * 1000.0))
    descriptors = []
    for tile_index in np.flatnonzero(mask.grid.reshape(-1)):
        row, column = divmod(int(tile_index), int(workload["grid_columns"]))
        descriptors.append(make_dispatch_descriptor(frame_id=frame_id, frame_width=frame_width, frame_height=frame_height,
            core_x=column * tile_width, core_y=row * tile_height, core_width=tile_width, core_height=tile_height,
            halo=halo, deadline_us=deadline_us, priority=1.0))
    return SparseFrameWorkList(frame_id=frame_id, frame_width=frame_width, frame_height=frame_height,
        tile_width=tile_width, tile_height=tile_height, deadline_us=deadline_us, descriptors=tuple(descriptors))
