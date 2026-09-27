"""Run the preregistered Paper 3 RK3576 fragmentation-pair experiment.

The module has deliberately narrow inputs: one frozen YAML contract, the
accepted fixed-shape RKNN catalog, and the contract's single raw-output path.
It creates the synthetic coordinate fixture and six masks locally and never
accepts a dataset, trace, policy, routing, checkpoint replacement, or model
conversion argument.  The checked-in contract remains ``ACCESS_REQUIRED``;
therefore the command must stop before RKNN initialization until an
independently reviewed contract changes every physical-access pin.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.benchmark_rk3576_sparse import (  # noqa: E402
    RKNNShapeFamilyBackend,
    SparseShapeBackend,
    build_dispatch_groups,
    make_dispatch_descriptor,
    resize_bicubic,
    run_sparse_frame,
    sha256_file,
)
from src.deployment.board_probe import collect_board_facts, read_optional_text, read_thermal_zones  # noqa: E402
from src.runtime.sparse_refinement import SparseFrameWorkList  # noqa: E402


EXPERIMENT_ID = "paper3_rk3576_fragmentation_pairs_v1"
PENDING_STATUS = "ACCESS_REQUIRED"
AUTHORIZED_STATUS = "AUTHORIZED_FOR_ACCESS"
PINNED_BOARD_IDENTITY = "928996c84e65dc3f119cd65f232d2ad6b4d4acd12a0d76ed7b4ecdec016b199c"
PINNED_RKNN_RUNTIME_LIBRARY_SHA256 = "cf6ea624489225ddc9d334b370c64083b0fd9d6a5c9166eed47d999cc85a806b"
CONTRACT_AUTHORIZATION_FIELDS = (
    "status",
    "authorization_boundary.physical_rk3576_authorized",
    "authorization_boundary.result_files_authorized",
    "authorization_boundary.required_next_state",
    "device_runtime_controls.board_identity",
    "device_runtime_controls.rknn_runtime_library_sha256",
    "device_runtime_controls.accepted_repository_commit",
    "expected_outputs_after_future_authorization.current_phase_outputs_must_not_exist",
)
PENDING_AUTHORIZATION_VALUES = (
    PENDING_STATUS,
    False,
    False,
    "explicit_orchestrator_access_authorization",
    "TO_BE_PINNED_BEFORE_ACCESS",
    "TO_BE_PINNED_BEFORE_ACCESS",
    "TO_BE_PINNED_BEFORE_ACCESS",
    True,
)
PENDING_OFFLINE_GATES = (
    "contract_status_is_ACCESS_REQUIRED",
    "physical_rk3576_authorized_is_false",
    "result_files_authorized_is_false",
    "board_identity_not_pinned",
    "rknn_runtime_library_sha256_not_pinned",
    "accepted_repository_commit_not_pinned",
    "ssh_authentication_not_available",
    "exclusive_session_ac_power_and_reviewer_signoff_not_attested",
)
AUTHORIZED_OFFLINE_GATES = (
    "ssh_authentication_not_available",
    "exclusive_session_ac_power_and_reviewer_signoff_not_attested",
)
DEFAULT_CONTRACT = ROOT / "configs/paper3_rk3576_fragmentation_pairs_v1.yaml"
DEFAULT_CATALOG = ROOT / "results/models/sparse_refinement_rknn_v2/catalog.json"
MODEL_MACS_PER_LR_POSITION = 8_424
FORBIDDEN_PATH_TOKENS = (
    "sealed_trace",
    "trace_manifest",
    "observation_log",
    "selector",
    "routing",
    "retun",
)
IMPLEMENTATION_PATHS = (
    Path("configs/paper3_rk3576_fragmentation_pairs_v1.yaml"),
    Path("docs/paper3_rk3576_fragmentation_pairs_v1.md"),
    Path("scripts/rk3576/run_paper3_rk3576_fragmentation_pairs_v1.sh"),
    Path("src/deployment/benchmark_rk3576_fragmentation_pairs.py"),
    Path("src/deployment/prepare_paper3_rk3576_fragmentation_pairs.py"),
    Path("src/report/paper3_rk3576_fragmentation_pairs_v1.py"),
    Path("src/report/paper3_rk3576_fragmentation_pairs_manifest.py"),
    Path("tests/test_paper3_rk3576_fragmentation_pairs_contract.py"),
    Path("tests/test_benchmark_rk3576_fragmentation_pairs.py"),
    Path("tests/test_paper3_rk3576_fragmentation_pairs_analysis.py"),
    Path("tests/test_paper3_rk3576_fragmentation_pairs_manifest.py"),
    Path("src/deployment/benchmark_rk3576_sparse.py"),
    Path("src/deployment/board_probe.py"),
    Path("src/runtime/sparse_refinement.py"),
)


class AccessBlocked(RuntimeError):
    """Raised before backend creation when a frozen access gate is unresolved."""


class PhysicalStop(RuntimeError):
    """Raised after output ownership when a frozen physical stop rule fires."""


@dataclass(frozen=True)
class FrozenMask:
    """One validated synthetic 6x10 mask from the preregistration."""

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


def canonical_sha256(value: object) -> str:
    """Hash one JSON value using the preregistration's canonical encoding."""

    def encode_date(candidate: object) -> str:
        if isinstance(candidate, date):
            return candidate.isoformat()
        raise TypeError(f"unsupported canonical JSON value: {type(candidate).__name__}")

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=encode_date,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dotted_value(value: Mapping[str, Any], dotted: str) -> Any:
    """Read one required dotted field from a nested contract mapping."""

    target: Any = value
    for part in dotted.split("."):
        if not isinstance(target, Mapping) or part not in target:
            raise AccessBlocked(f"authorization field is absent from contract: {dotted}")
        target = target[part]
    return target


def contract_authorization_values(contract: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the exact eight-field authorization tuple after policy validation."""

    try:
        configured = tuple(
            str(field)
            for field in contract["device_runtime_controls"]["repository_authorization_policy"]
            ["contract_authorization_fields"]
        )
    except (KeyError, TypeError) as error:
        raise AccessBlocked("contract authorization-field policy is missing") from error
    if configured != CONTRACT_AUTHORIZATION_FIELDS:
        raise AccessBlocked("contract authorization fields differ from the exact eight-field allowlist")
    return tuple(_dotted_value(contract, dotted) for dotted in CONTRACT_AUTHORIZATION_FIELDS)


def contract_authorization_phase(contract: Mapping[str, Any]) -> str:
    """Accept only one complete pending tuple or one complete authorized tuple."""

    values = contract_authorization_values(contract)
    if values == PENDING_AUTHORIZATION_VALUES:
        return PENDING_STATUS
    authorized_prefix = (
        AUTHORIZED_STATUS,
        True,
        True,
        "authorized_physical_capture",
        PINNED_BOARD_IDENTITY,
        PINNED_RKNN_RUNTIME_LIBRARY_SHA256,
    )
    baseline = values[6]
    if (
        values[:6] == authorized_prefix
        and isinstance(baseline, str)
        and re.fullmatch(r"[0-9a-f]{40}", baseline) is not None
        and values[7] is False
    ):
        return AUTHORIZED_STATUS
    raise AccessBlocked("contract authorization fields are a mixed, partial, or unknown phase")


def offline_unresolved_gates(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Return only gates that remain unresolved in the contract's coherent phase."""

    phase = contract_authorization_phase(contract)
    return PENDING_OFFLINE_GATES if phase == PENDING_STATUS else AUTHORIZED_OFFLINE_GATES


def load_contract(path: Path) -> dict[str, Any]:
    """Load one YAML object and require the exact Paper 3 experiment ID."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("fragmentation-pair contract root must be a mapping")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected experiment_id in fragmentation-pair contract")
    return value


def generate_coordinate_fixture(contract: Mapping[str, Any]) -> np.ndarray:
    """Generate the only authorized input pixels without reading any dataset."""

    spec = contract["input_fixture"]
    height = int(spec["height"])
    width = int(spec["width"])
    y, x = np.indices((height, width), dtype=np.int64)
    fixture = np.empty((height, width, 3), dtype=np.uint8)
    fixture[..., 0] = (3 * x + 5 * y + 17) % 256
    fixture[..., 1] = (7 * x + 11 * y + 29) % 256
    fixture[..., 2] = (13 * x + 2 * y + 47) % 256
    if fixture.nbytes != int(spec["byte_length"]):
        raise ValueError("synthetic coordinate fixture byte length changed")
    observed = hashlib.sha256(fixture.tobytes(order="C")).hexdigest()
    if observed != str(spec["raw_bytes_sha256"]):
        raise ValueError("synthetic coordinate fixture hash changed")
    return fixture


def _count_runs(row: str) -> int:
    return sum(bit == "1" and (index == 0 or row[index - 1] == "0") for index, bit in enumerate(row))


def _row_span(row: str) -> int:
    selected = [index for index, bit in enumerate(row) if bit == "1"]
    if not selected:
        raise ValueError("every frozen mask row must be active")
    return selected[-1] - selected[0] + 1


def _mask_integer(rows: Sequence[str]) -> int:
    value = 0
    for row_index, row in enumerate(rows):
        for column_index, bit in enumerate(row):
            if bit == "1":
                value |= 1 << (row_index * 10 + column_index)
    return value


def load_frozen_masks(contract: Mapping[str, Any]) -> tuple[FrozenMask, ...]:
    """Validate and materialize exactly the six preregistered synthetic masks."""

    spec = contract["mask_contract"]
    source_rows = spec["masks"]
    if len(source_rows) != 6 or int(spec["mask_count"]) != 6:
        raise ValueError("fragmentation study requires exactly six masks")
    canonical_rows = {str(row["id"]): row["rows"] for row in source_rows}
    if canonical_sha256(canonical_rows) != str(spec["canonical_rows_sha256"]):
        raise ValueError("frozen canonical mask-row hash changed")

    masks: list[FrozenMask] = []
    for source in source_rows:
        rows = tuple(str(row) for row in source["rows"])
        if len(rows) != 6 or any(len(row) != 10 or set(row) - {"0", "1"} for row in rows):
            raise ValueError(f"invalid frozen mask rows: {source['id']}")
        selected = sum(row.count("1") for row in rows)
        runs = sum(_count_runs(row) for row in rows)
        coverage = sum(_row_span(row) for row in rows)
        if selected != int(source["selected_tiles"]):
            raise ValueError(f"selected-tile mismatch: {source['id']}")
        if runs != int(source["horizontal_runs"]):
            raise ValueError(f"horizontal-run mismatch: {source['id']}")
        if coverage != int(source["row_span_coverage_tiles"]):
            raise ValueError(f"row-span coverage mismatch: {source['id']}")
        if _mask_integer(rows) != int(str(source["mask_hex"]), 16):
            raise ValueError(f"mask hexadecimal encoding mismatch: {source['id']}")
        grid = np.asarray([[bit == "1" for bit in row] for row in rows], dtype=bool)
        masks.append(
            FrozenMask(
                mask_id=str(source["id"]),
                pair_id=str(source["pair_id"]),
                geometry=str(source["geometry"]),
                density_percent=int(source["density_percent"]),
                selected_tiles=selected,
                active_rows=int(np.count_nonzero(np.any(grid, axis=1))),
                horizontal_runs=runs,
                row_span_coverage_tiles=coverage,
                mask_hex=str(source["mask_hex"]),
                rows=rows,
                grid=grid,
            )
        )

    by_pair: dict[str, list[FrozenMask]] = {}
    for mask in masks:
        by_pair.setdefault(mask.pair_id, []).append(mask)
    if len(by_pair) != 3 or any(len(pair) != 2 for pair in by_pair.values()):
        raise ValueError("six masks must form three compact/fragmented pairs")
    for pair in by_pair.values():
        if len({mask.selected_tiles for mask in pair}) != 1:
            raise ValueError("pair members changed selected-output count")
        if any(mask.active_rows != 6 for mask in pair):
            raise ValueError("every pair member must keep six active rows")
    return tuple(masks)


def build_schedule(contract: Mapping[str, Any], masks: Sequence[FrozenMask]) -> list[dict[str, Any]]:
    """Materialize the frozen five-block, adjacent-pair, 60-entry schedule."""

    design = contract["execution_design"]
    modes = tuple(str(mode) for mode in design["dispatch_modes"])
    if modes != ("exact", "row_spans"):
        raise ValueError("dispatch-mode order changed")
    if int(design["blocks"]) != 5 or int(design["repetitions_per_mask_mode"]) != 5:
        raise ValueError("five-block repetition contract changed")
    by_pair: dict[str, dict[str, FrozenMask]] = {}
    for mask in masks:
        by_pair.setdefault(mask.pair_id, {})[mask.geometry] = mask
    pair_ids = sorted(by_pair, key=lambda value: int(value.rsplit("_", 1)[1]))
    rng = random.Random(int(design["randomization_seed"]))
    schedule: list[dict[str, Any]] = []
    order = 0
    for block_index in range(1, int(design["blocks"]) + 1):
        cells = [(pair_id, mode) for pair_id in pair_ids for mode in modes]
        rng.shuffle(cells)
        for pair_id, mode in cells:
            pair = by_pair[pair_id]
            compact = pair["compact"]
            fragmented = pair["maximally_fragmented"]
            oriented = (compact, fragmented) if rng.getrandbits(1) == 0 else (fragmented, compact)
            for mask in oriented:
                order += 1
                schedule.append(
                    {
                        "repeat_index": block_index,
                        "block_index": block_index,
                        "execution_order": order,
                        "pair_id": pair_id,
                        "dispatch_mode": mode,
                        "mask_id": mask.mask_id,
                        "geometry": mask.geometry,
                        "density_percent": mask.density_percent,
                    }
                )
    expected_samples = int(design["measured_samples"])
    if expected_samples <= 0:
        raise ValueError("frozen measured-sample count must be positive")
    if len(schedule) != expected_samples:
        raise ValueError("materialized schedule is not the frozen 60 samples")
    for index in range(0, len(schedule), 2):
        left, right = schedule[index : index + 2]
        key_fields = ("repeat_index", "pair_id", "dispatch_mode")
        if any(left[field] != right[field] for field in key_fields):
            raise RuntimeError("pair members are not adjacent in the deterministic schedule")
        if {left["geometry"], right["geometry"]} != {"compact", "maximally_fragmented"}:
            raise RuntimeError("schedule cell lost one pair orientation")
    return schedule


def build_exact_worklist(mask: FrozenMask, *, frame_id: int, contract: Mapping[str, Any]) -> SparseFrameWorkList:
    """Create one accepted 64x60 descriptor per selected mask cell."""

    workload = contract["workload"]
    tile_width, tile_height = map(int, workload["macro_tile_lr"])
    frame_width, frame_height = map(int, workload["input_resolution"])
    halo = int(workload["halo_pixels"])
    deadline_us = int(round(float(workload["deadline_ms"]) * 1000.0))
    descriptors = []
    for tile_index in np.flatnonzero(mask.grid.reshape(-1)):
        tile_row, tile_column = divmod(int(tile_index), int(workload["grid_columns"]))
        descriptors.append(
            make_dispatch_descriptor(
                frame_id=frame_id,
                frame_width=frame_width,
                frame_height=frame_height,
                core_x=tile_column * tile_width,
                core_y=tile_row * tile_height,
                core_width=tile_width,
                core_height=tile_height,
                halo=halo,
                deadline_us=deadline_us,
                priority=1.0,
            )
        )
    return SparseFrameWorkList(
        frame_id=frame_id,
        frame_width=frame_width,
        frame_height=frame_height,
        tile_width=tile_width,
        tile_height=tile_height,
        deadline_us=deadline_us,
        descriptors=tuple(descriptors),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _reject_forbidden_path(path: Path, label: str) -> None:
    normalized = path.as_posix().lower()
    if any(token in normalized for token in FORBIDDEN_PATH_TOKENS):
        raise ValueError(f"{label} resolves through a prohibited path class")


def validate_offline_dependencies(
    contract_path: Path,
    catalog_path: Path,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Hash the contract, checkpoint, catalog, and every accepted RKNN graph."""

    _reject_forbidden_path(contract_path, "contract")
    _reject_forbidden_path(catalog_path, "catalog")
    contract = load_contract(contract_path)
    catalog = _read_json(catalog_path)
    workload = contract["workload"]
    if sha256_file(catalog_path) != str(workload["rknn_catalog_file_sha256"]):
        raise ValueError("accepted RKNN catalog file hash mismatch")
    if catalog.get("status") != "board_ready" or catalog.get("target_platform") != "rk3576":
        raise ValueError("accepted RKNN catalog is not board-ready for RK3576")
    if catalog["checkpoint"]["sha256"] != str(workload["checkpoint_sha256"]):
        raise ValueError("catalog and preregistration checkpoint identities differ")
    checkpoint_path = root / str(workload["checkpoint_path"])
    _reject_forbidden_path(checkpoint_path, "checkpoint")
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != str(workload["checkpoint_sha256"]):
        raise ValueError("frozen checkpoint hash mismatch")

    graphs: list[dict[str, Any]] = []
    shapes: set[tuple[int, int]] = set()
    for entry in catalog["shapes"]:
        shape = (int(entry["core_width"]), int(entry["core_height"]))
        graph_path = root / str(entry["rknn"]["path"])
        _reject_forbidden_path(graph_path, "RKNN graph")
        observed = sha256_file(graph_path)
        if observed != str(entry["rknn"]["sha256"]):
            raise ValueError(f"accepted RKNN graph hash mismatch: {graph_path}")
        if str(entry["rknn"]["toolkit_version"]) != str(workload["rknn_toolkit_compiler_version"]):
            raise ValueError("RKNN compiler version differs from the frozen contract")
        shapes.add(shape)
        graphs.append(
            {
                "shape": list(shape),
                "path": graph_path.relative_to(root).as_posix(),
                "bytes": graph_path.stat().st_size,
                "sha256": observed,
            }
        )
    required_shapes = {(64, 60), (128, 60), (256, 60), (512, 60), (640, 60)}
    if not required_shapes.issubset(shapes):
        raise ValueError("catalog lacks a static shape required by the six masks")
    return {
        "contract": {
            "path": contract_path.relative_to(root).as_posix(),
            "bytes": contract_path.stat().st_size,
            "sha256": sha256_file(contract_path),
        },
        "catalog": {
            "path": catalog_path.relative_to(root).as_posix(),
            "bytes": catalog_path.stat().st_size,
            "sha256": sha256_file(catalog_path),
        },
        "checkpoint": {
            "path": checkpoint_path.relative_to(root).as_posix(),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
        "rknn_graphs": graphs,
        "supported_shapes": [list(shape) for shape in sorted(shapes)],
    }


def build_offline_preflight(contract_path: Path, catalog_path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Build the deterministic, non-authorizing preflight content."""

    contract = load_contract(contract_path)
    phase = contract_authorization_phase(contract)
    fixture = generate_coordinate_fixture(contract)
    masks = load_frozen_masks(contract)
    schedule = build_schedule(contract, masks)
    dependencies = validate_offline_dependencies(contract_path, catalog_path, root=root)
    implementation = []
    for relative in IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Paper 3 implementation dependency is missing: {path}")
        implementation.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    dependencies["implementation"] = implementation
    mask_rows = {mask.mask_id: list(mask.rows) for mask in masks}
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "record_kind": "offline_preflight_candidate",
        "contract_status": str(contract["status"]),
        "physical_execution_measured": False,
        "warmups_executed": 0,
        "measured_samples": 0,
        "authorized_for_samples": False,
        "stimuli": {
            "fixture": {
                "id": str(contract["input_fixture"]["id"]),
                "bytes": fixture.nbytes,
                "sha256": hashlib.sha256(fixture.tobytes(order="C")).hexdigest(),
            },
            "mask_count": len(masks),
            "mask_rows": mask_rows,
            "canonical_rows_sha256": canonical_sha256(mask_rows),
        },
        "schedule": {
            "entries": schedule,
            "entry_count": len(schedule),
            "sha256": canonical_sha256(schedule),
        },
        "dependencies": dependencies,
        "board_probe": {
            "status": "SSH_AUTHENTICATION_REQUIRED",
            "target": "radxa@192.168.1.84",
            "identity": None,
            "runtime_library_sha256": None,
            "cpu_governors": None,
            "npu_frequencies": None,
            "thermals": None,
            "storage": None,
        },
        "unresolved_gates": list(offline_unresolved_gates(contract)),
        "next_action": (
            "independent_review_then_explicit_contract_authorization"
            if phase == PENDING_STATUS
            else "complete_physical_runtime_gates_then_single_capture"
        ),
    }


def _git_head(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


GitRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _default_git_runner(root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run one read-only Git query inside the candidate checkout."""

    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "-C",
            str(root),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )


def _required_git_output(run_git: GitRunner, arguments: Sequence[str], label: str) -> str:
    result = run_git(arguments)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise AccessBlocked(f"repository {label} query failed: {detail}")
    value = result.stdout.strip()
    if not value:
        raise AccessBlocked(f"repository {label} query returned no identity")
    return value


def _validated_relative_paths(values: Sequence[Any], label: str) -> tuple[str, ...]:
    paths = tuple(str(value).replace("\\", "/") for value in values)
    if not paths or len(paths) != len(set(paths)):
        raise AccessBlocked(f"repository {label} must be a nonempty unique path list")
    for value in paths:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value.startswith("/"):
            raise AccessBlocked(f"repository {label} contains an unsafe path: {value}")
    return paths


def _without_dotted_fields(value: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """Remove explicitly authorized scalar fields from a contract copy."""

    projected = copy.deepcopy(dict(value))
    for dotted in fields:
        parts = dotted.split(".")
        target: Any = projected
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise AccessBlocked(f"authorization field is absent from contract: {dotted}")
            target = target[part]
        if not isinstance(target, dict) or parts[-1] not in target:
            raise AccessBlocked(f"authorization field is absent from contract: {dotted}")
        del target[parts[-1]]
    return projected


def contract_invariant_sha256(contract: Mapping[str, Any]) -> str:
    """Hash every contract field except the exact authorization allowlist."""

    return canonical_sha256(_without_dotted_fields(contract, CONTRACT_AUTHORIZATION_FIELDS))


def verify_repository_baseline(
    contract: Mapping[str, Any],
    *,
    root: Path = ROOT,
    run_git: GitRunner | None = None,
) -> dict[str, Any]:
    """Verify a clean metadata-only descendant of the implementation baseline."""

    if contract_authorization_phase(contract) != AUTHORIZED_STATUS:
        raise AccessBlocked("repository verification requires a coherent authorized contract")
    controls = contract["device_runtime_controls"]
    baseline = str(controls["accepted_repository_commit"])
    if str(controls.get("accepted_repository_commit_role")) != "accepted_implementation_baseline":
        raise AccessBlocked("accepted_repository_commit role is not the implementation baseline")
    if baseline.startswith("TO_BE_PINNED"):
        raise AccessBlocked("accepted_repository_commit remains unresolved")
    if re.fullmatch(r"[0-9a-f]{40,64}", baseline) is None:
        raise AccessBlocked("accepted_repository_commit must be a full hexadecimal commit identity")
    policy = controls["repository_authorization_policy"]
    if policy.get("require_clean_checkout") is not True:
        raise AccessBlocked("repository policy must require a clean checkout")
    if policy.get("baseline_must_be_ancestor_of_authorized_checkout") is not True:
        raise AccessBlocked("repository policy must require baseline ancestry")
    metadata_paths = _validated_relative_paths(
        policy["authorization_metadata_paths"], "authorization metadata paths"
    )
    executable_paths = _validated_relative_paths(
        policy["executable_baseline_paths"], "executable baseline paths"
    )
    authorization_fields = tuple(str(value) for value in policy["contract_authorization_fields"])
    if authorization_fields != CONTRACT_AUTHORIZATION_FIELDS:
        raise AccessBlocked("contract authorization fields differ from the exact eight-field allowlist")

    if run_git is None:
        run_git = lambda arguments: _default_git_runner(root, arguments)
    status = run_git(("status", "--porcelain=v1", "--untracked-files=all"))
    if status.returncode != 0:
        raise AccessBlocked(f"repository cleanliness query failed: {status.stderr.strip()}")
    if status.stdout.strip():
        raise AccessBlocked("repository checkout is dirty")
    head = _required_git_output(run_git, ("rev-parse", "HEAD"), "HEAD")
    resolved_baseline = _required_git_output(
        run_git, ("rev-parse", "--verify", f"{baseline}^{{commit}}"), "baseline"
    )
    if resolved_baseline != baseline:
        raise AccessBlocked("accepted_repository_commit did not resolve to its exact full identity")
    ancestry = run_git(("merge-base", "--is-ancestor", baseline, head))
    if ancestry.returncode != 0:
        raise AccessBlocked("accepted implementation baseline is not an ancestor of HEAD")
    changed_result = run_git(("diff", "--name-only", "--diff-filter=ACDMRTUXB", f"{baseline}..{head}"))
    if changed_result.returncode != 0:
        raise AccessBlocked(f"repository descendant-diff query failed: {changed_result.stderr.strip()}")
    changed_paths = tuple(line.strip().replace("\\", "/") for line in changed_result.stdout.splitlines() if line.strip())
    unexpected = sorted(set(changed_paths) - set(metadata_paths))
    if unexpected:
        raise AccessBlocked(f"descendant changes non-authorization paths: {unexpected}")

    contract_relative = "configs/paper3_rk3576_fragmentation_pairs_v1.yaml"
    if contract_relative not in metadata_paths:
        raise AccessBlocked("contract path is not explicitly authorized metadata")
    baseline_text = _required_git_output(
        run_git, ("show", f"{baseline}:{contract_relative}"), "baseline contract"
    )
    baseline_contract = yaml.safe_load(baseline_text)
    if not isinstance(baseline_contract, dict):
        raise AccessBlocked("baseline contract is not a mapping")
    if contract_authorization_phase(baseline_contract) != PENDING_STATUS:
        raise AccessBlocked("accepted implementation baseline contract is not coherently pending")
    if contract_invariant_sha256(contract) != contract_invariant_sha256(baseline_contract):
        raise AccessBlocked("contract drift exceeds the explicitly authorized metadata fields")

    blob_records = []
    for relative in executable_paths:
        current_path = root / relative
        if not current_path.is_file():
            raise AccessBlocked(f"executable baseline path is missing: {relative}")
        baseline_blob = _required_git_output(
            run_git, ("rev-parse", f"{baseline}:{relative}"), f"baseline blob {relative}"
        )
        current_blob = _required_git_output(
            run_git, ("hash-object", "--", relative), f"current blob {relative}"
        )
        if current_blob != baseline_blob:
            raise AccessBlocked(f"executable drift from accepted baseline: {relative}")
        blob_records.append({"path": relative, "git_blob": current_blob})
    return {
        "accepted_implementation_baseline": baseline,
        "authorized_checkout_head": head,
        "clean_checkout": True,
        "baseline_is_ancestor": True,
        "changed_authorization_metadata_paths": list(changed_paths),
        "contract_authorization_fields": list(authorization_fields),
        "verified_executable_blobs": blob_records,
    }


def _runtime_library() -> dict[str, Any]:
    candidates = []
    for root in (Path("/usr/lib"), Path("/usr/local/lib"), Path("/lib")):
        if root.exists():
            candidates.extend(root.glob("**/librknnrt.so*"))
    files = sorted({path.resolve() for path in candidates if path.is_file()})
    if not files:
        return {"path": None, "sha256": None}
    path = files[0]
    return {"path": str(path), "sha256": sha256_file(path)}


def _read_control_snapshot() -> dict[str, Any]:
    governors = {
        str(path): read_optional_text(path)
        for path in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor"))
    }
    npu: dict[str, str | None] = {}
    for device in sorted(Path("/sys/class/devfreq").glob("*")):
        if "npu" not in device.name.lower() and "rknpu" not in device.name.lower():
            continue
        for field in ("governor", "cur_freq", "min_freq", "max_freq"):
            npu[f"{device}/{field}"] = read_optional_text(device / field)
    return {"cpu_governors": governors, "npu_frequency_controls": npu}


def _max_temperature(zones: Sequence[Mapping[str, Any]]) -> float | None:
    values = [float(row["temperature_c"]) for row in zones if row.get("temperature_c") is not None]
    return max(values) if values else None


def collect_physical_controls(output_parent: Path) -> dict[str, Any]:
    """Read identity and timing controls without changing board state."""

    facts = collect_board_facts()
    usage = shutil.disk_usage(output_parent if output_parent.exists() else output_parent.parent)
    compatible = facts["platform"].get("device_tree_compatible")
    model = read_optional_text(Path("/proc/device-tree/model"))
    identity_payload = {
        "hostname": socket.gethostname(),
        "model": model,
        "compatible": compatible,
        "machine": platform.machine(),
    }
    thermals = read_thermal_zones()
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "board": facts,
        "board_identity": identity_payload,
        "board_identity_sha256": canonical_sha256(identity_payload),
        "runtime_library": _runtime_library(),
        "repository_commit": _git_head(ROOT),
        "controls": _read_control_snapshot(),
        "thermals": thermals,
        "maximum_temperature_c": _max_temperature(thermals),
        "storage": {
            "path": str(output_parent),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
    }


def enforce_physical_authorization(
    contract: Mapping[str, Any],
    *,
    require_status: str,
    attest_exclusive: bool,
    attest_ac_power: bool,
    attest_no_other_workloads: bool,
    reviewer_signoff: str | None,
) -> None:
    """Refuse backend construction until every human access gate is explicit."""

    if require_status != AUTHORIZED_STATUS:
        raise AccessBlocked(f"--require-status must be {AUTHORIZED_STATUS}")
    if str(contract["status"]) != AUTHORIZED_STATUS:
        raise AccessBlocked(f"contract status is {contract['status']}; expected {AUTHORIZED_STATUS}")
    boundary = contract["authorization_boundary"]
    if not bool(boundary["physical_rk3576_authorized"]):
        raise AccessBlocked("physical_rk3576_authorized is false")
    if not bool(boundary["result_files_authorized"]):
        raise AccessBlocked("result_files_authorized is false")
    if bool(boundary["mask_generation_from_sealed_traces_authorized"]):
        raise AccessBlocked("trace-derived mask generation must remain unauthorized")
    if bool(boundary["sealed_trace_evaluation_authorized"]):
        raise AccessBlocked("sealed-trace evaluation must remain unauthorized")
    if bool(boundary["model_or_routing_retuning_authorized"]):
        raise AccessBlocked("model/routing retuning must remain unauthorized")
    if not (attest_exclusive and attest_ac_power and attest_no_other_workloads):
        raise AccessBlocked("exclusive board, AC power, and idle-workload attestations are required")
    if not reviewer_signoff or not reviewer_signoff.strip():
        raise AccessBlocked("independent reviewer signoff is required")
    if bool(contract["expected_outputs_after_future_authorization"]["current_phase_outputs_must_not_exist"]):
        raise AccessBlocked("current_phase_outputs_must_not_exist remains true")


def enforce_device_pins(contract: Mapping[str, Any], controls: Mapping[str, Any]) -> None:
    """Compare the read-only board probe with all preregistered pins."""

    expected = contract["device_runtime_controls"]
    for field in ("board_identity", "rknn_runtime_library_sha256"):
        if str(expected[field]).startswith("TO_BE_PINNED"):
            raise AccessBlocked(f"{field} remains unresolved")
    platform_facts = controls["board"]["platform"]
    if str(platform_facts["machine"]) != str(expected["required_machine"]):
        raise AccessBlocked("board machine identity mismatch")
    if str(platform_facts["release"]) != str(expected["expected_kernel_release"]):
        raise AccessBlocked("board kernel release mismatch")
    if str(platform_facts["python"]) != str(expected["expected_python"]):
        raise AccessBlocked("board Python version mismatch")
    if str(expected["expected_os"]) not in str(platform_facts.get("os_release") or ""):
        raise AccessBlocked("board operating-system identity mismatch")
    compatible = str(platform_facts.get("device_tree_compatible") or "")
    if str(expected["required_device_tree_compatible_contains"]) not in compatible:
        raise AccessBlocked("device-tree compatibility does not identify RK3576")
    if str(controls["board_identity_sha256"]) != str(expected["board_identity"]):
        raise AccessBlocked("board identity hash mismatch")
    if str(controls["runtime_library"]["sha256"]) != str(expected["rknn_runtime_library_sha256"]):
        raise AccessBlocked("RKNN runtime library hash mismatch")
    snapshot = controls["controls"]
    if not snapshot["cpu_governors"] or not snapshot["npu_frequency_controls"]:
        raise AccessBlocked("CPU governor or NPU frequency controls are not observable")
    maximum = controls.get("maximum_temperature_c")
    if maximum is None or float(maximum) > float(expected["start_temperature_max_c"]):
        raise AccessBlocked("board temperature is unavailable or above the block-start maximum")


def expected_output_path(contract: Mapping[str, Any], *, root: Path = ROOT) -> Path:
    return root / str(contract["expected_outputs_after_future_authorization"]["raw_capture"])


def enforce_output_boundary(output: Path, contract: Mapping[str, Any], *, root: Path = ROOT) -> None:
    """Require the exact raw path and a nonexistent or empty dedicated root."""

    expected = expected_output_path(contract, root=root).resolve()
    resolved = output.resolve()
    _reject_forbidden_path(resolved, "output")
    if resolved != expected:
        raise ValueError(f"raw output must be the preregistered path: {expected}")
    if resolved.exists():
        raise FileExistsError(f"raw output already exists: {resolved}")
    parent = resolved.parent
    if parent.exists() and any(parent.iterdir()):
        raise FileExistsError(f"raw output directory is nonempty: {parent}")


def _atomic_snapshot(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _claim_output(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _selected_pixels(array: np.ndarray, mask: FrozenMask, contract: Mapping[str, Any]) -> np.ndarray:
    tile_width, tile_height = map(int, contract["workload"]["macro_tile_lr"])
    parts = []
    for tile_index in np.flatnonzero(mask.grid.reshape(-1)):
        row, column = divmod(int(tile_index), 10)
        y0 = row * tile_height * 2
        x0 = column * tile_width * 2
        parts.append(array[y0 : y0 + tile_height * 2, x0 : x0 + tile_width * 2])
    return np.concatenate([part.reshape(-1, 3) for part in parts], axis=0)


def _selected_mask_hr(mask: FrozenMask, contract: Mapping[str, Any]) -> np.ndarray:
    width, height = map(int, contract["workload"]["output_resolution"])
    tile_width, tile_height = map(int, contract["workload"]["macro_tile_lr"])
    selected = np.zeros((height, width), dtype=bool)
    for tile_index in np.flatnonzero(mask.grid.reshape(-1)):
        row, column = divmod(int(tile_index), 10)
        selected[
            row * tile_height * 2 : (row + 1) * tile_height * 2,
            column * tile_width * 2 : (column + 1) * tile_width * 2,
        ] = True
    return selected


def _psnr(left: np.ndarray, right: np.ndarray) -> float:
    mse = float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2))
    return math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def _structural_metrics(
    worklist: SparseFrameWorkList,
    mode: str,
    supported_shapes: frozenset[tuple[int, int]],
    halo: int,
) -> tuple[int, float]:
    groups = build_dispatch_groups(
        worklist,
        mode=mode,
        halo=halo,
        supported_shapes=supported_shapes,
    )
    macs = sum(
        (group.inference_region.core_width + 2 * halo)
        * (group.inference_region.core_height + 2 * halo)
        * MODEL_MACS_PER_LR_POSITION
        for group in groups
    )
    return len(groups), macs / 1.0e9


def run_authorized_experiment(
    args: argparse.Namespace,
    *,
    backend: SparseShapeBackend | None = None,
) -> dict[str, Any]:
    """Run only after the independently pinned contract authorizes access."""

    contract_path = args.contract.resolve()
    catalog_path = args.catalog.resolve()
    output_path = args.output.resolve()
    contract = load_contract(contract_path)
    enforce_physical_authorization(
        contract,
        require_status=args.require_status,
        attest_exclusive=args.confirm_exclusive_board,
        attest_ac_power=args.confirm_ac_power,
        attest_no_other_workloads=args.confirm_no_other_workloads,
        reviewer_signoff=args.reviewer_signoff,
    )
    repository_verification = verify_repository_baseline(contract)
    enforce_output_boundary(output_path, contract)
    offline = build_offline_preflight(contract_path, catalog_path)
    controls = collect_physical_controls(output_path.parent)
    enforce_device_pins(contract, controls)
    fixture = generate_coordinate_fixture(contract)
    masks = {mask.mask_id: mask for mask in load_frozen_masks(contract)}
    schedule = list(offline["schedule"]["entries"])
    supported_shapes = frozenset(tuple(row) for row in offline["dependencies"]["supported_shapes"])
    halo = int(contract["workload"]["halo_pixels"])
    deadline_ms = float(contract["workload"]["deadline_ms"])
    thermal_rules = contract["device_runtime_controls"]
    baseline = np.clip(resize_bicubic(fixture), 0.0, 1.0)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "physical_running",
        "all_gates_passed": False,
        "physical_execution_measured": False,
        "contract_sha256": sha256_file(contract_path),
        "offline_preflight": offline,
        "repository_baseline_verification": repository_verification,
        "physical_controls_start": controls,
        "attestations": {
            "exclusive_board": args.confirm_exclusive_board,
            "ac_power": args.confirm_ac_power,
            "no_other_workloads": args.confirm_no_other_workloads,
            "reviewer_signoff": args.reviewer_signoff,
        },
        "records": [],
        "cross_mode_integrity": [],
        "failure": None,
    }
    _claim_output(output_path, payload)
    owns_backend = backend is None
    selected_outputs: dict[tuple[int, str, str], np.ndarray] = {}
    selected_bicubic_psnr: dict[tuple[int, str, str], float] = {}
    initial_snapshot = controls["controls"]
    try:
        if backend is None:
            backend = RKNNShapeFamilyBackend(
                catalog_path,
                str(contract["workload"]["npu_core"]),
                None,
                None,
            )
        if frozenset(backend.supported_shapes) != supported_shapes:
            raise PhysicalStop("loaded RKNN shape family differs from the accepted catalog")
        backend.warmup(int(contract["execution_design"]["warmup_iterations_per_shape"]))
        active_block = None
        for entry in schedule:
            block_index = int(entry["block_index"])
            if block_index != active_block:
                zones = read_thermal_zones()
                maximum = _max_temperature(zones)
                if maximum is None or maximum > float(thermal_rules["cooldown_to_c_before_each_block"]):
                    raise PhysicalStop("block start temperature is unavailable or above the cooldown gate")
                if _read_control_snapshot() != initial_snapshot:
                    raise PhysicalStop("CPU governor or NPU frequency changed before a block")
                active_block = block_index
            mask = masks[str(entry["mask_id"])]
            worklist = build_exact_worklist(mask, frame_id=int(entry["execution_order"]), contract=contract)
            thermal_start = read_thermal_zones()
            output, stats = run_sparse_frame(
                fixture,
                worklist,
                backend,
                halo=halo,
                output_scale=None,
                output_zero_point=None,
                dispatch_mode=str(entry["dispatch_mode"]),
                supported_shapes=backend.supported_shapes,
            )
            thermal_end = read_thermal_zones()
            start_c = _max_temperature(thermal_start)
            end_c = _max_temperature(thermal_end)
            selected_float = _selected_pixels(output, mask, contract)
            selected_base = _selected_pixels(baseline, mask, contract)
            selected_u8 = np.rint(np.clip(selected_float, 0.0, 1.0) * 255.0).astype(np.uint8)
            selected_hr_mask = _selected_mask_hr(mask, contract)
            unselected_changed = not np.array_equal(output[~selected_hr_mask], baseline[~selected_hr_mask])
            npu_calls, executed_gmac = _structural_metrics(
                worklist,
                str(entry["dispatch_mode"]),
                backend.supported_shapes,
                halo,
            )
            committed = mask.selected_tiles * 64 * 60 * 4
            tensor_bytes = int(stats["npu_input_bytes"]) + int(stats["npu_output_bytes"])
            service_ms = float(stats["compute_pipeline_ms"])
            record = {
                **entry,
                "selected_tiles": mask.selected_tiles,
                "active_rows": mask.active_rows,
                "horizontal_runs": mask.horizontal_runs,
                "row_span_coverage_tiles": mask.row_span_coverage_tiles,
                "npu_calls": npu_calls,
                "executed_gmac": executed_gmac,
                "input_tensor_bytes": int(stats["npu_input_bytes"]),
                "output_tensor_bytes": int(stats["npu_output_bytes"]),
                "tensor_bytes_per_committed_hr_pixel": tensor_bytes / committed,
                "extraction_ms": float(stats["halo_extraction_ms"]),
                "planning_ms": float(stats["dispatch_planning_ms"]),
                "npu_inference_ms": float(stats["npu_ms"]),
                "reconstruction_ms": float(stats["reconstruction_ms"]),
                "service_ms": service_ms,
                "deadline_overrun_ms": max(0.0, service_ms - deadline_ms),
                "strict_deadline_success": service_ms <= deadline_ms,
                "committed_hr_pixels": committed,
                "selected_output_checksum": hashlib.sha256(selected_u8.tobytes(order="C")).hexdigest(),
                "selected_bicubic_psnr_db": _psnr(selected_float, selected_base),
                "unselected_output_commit": unselected_changed,
                "thermal_start_c": start_c,
                "thermal_end_c": end_c,
            }
            required_timings = (
                record["extraction_ms"],
                record["planning_ms"],
                record["npu_inference_ms"],
                record["reconstruction_ms"],
                record["service_ms"],
            )
            payload["records"].append(record)
            payload["physical_execution_measured"] = True
            _atomic_snapshot(output_path, payload)
            if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in required_timings):
                raise PhysicalStop("missing, nonfinite, or nonpositive timing")
            if unselected_changed:
                raise PhysicalStop("unselected output was committed")
            if end_c is None or end_c >= float(thermal_rules["abort_temperature_c"]):
                raise PhysicalStop("thermal abort threshold reached")
            if _read_control_snapshot() != initial_snapshot:
                raise PhysicalStop("CPU governor or NPU frequency changed during sampling")
            key = (int(entry["repeat_index"]), mask.mask_id, str(entry["dispatch_mode"]))
            selected_outputs[key] = selected_float.copy()
            selected_bicubic_psnr[key] = float(record["selected_bicubic_psnr_db"])

            if len(payload["records"]) % 12 == 0:
                repeat_index = int(entry["repeat_index"])
                for mask_id in sorted(masks):
                    exact = selected_outputs[(repeat_index, mask_id, "exact")]
                    spans = selected_outputs[(repeat_index, mask_id, "row_spans")]
                    comparison = {
                        "repeat_index": repeat_index,
                        "mask_id": mask_id,
                        "mean_abs_float": float(np.mean(np.abs(exact - spans))),
                        "psnr_delta_db": abs(
                            selected_bicubic_psnr[(repeat_index, mask_id, "exact")]
                            - selected_bicubic_psnr[(repeat_index, mask_id, "row_spans")]
                        ),
                    }
                    payload["cross_mode_integrity"].append(comparison)
                    gates = contract["frozen_gates"]["integrity"]
                    if comparison["mean_abs_float"] > float(gates["max_cross_mode_mean_abs_float"]):
                        raise PhysicalStop("cross-mode mean-absolute integrity gate failed")
                    if comparison["psnr_delta_db"] > float(gates["max_cross_mode_psnr_delta_db"]):
                        raise PhysicalStop("cross-mode PSNR-delta integrity gate failed")
                _atomic_snapshot(output_path, payload)

        if len(payload["records"]) != int(contract["execution_design"]["measured_samples"]):
            raise PhysicalStop("measured sample count differs from the frozen schedule")
        payload["status"] = "physical_complete_pending_offline_analysis"
        payload["physical_controls_end"] = collect_physical_controls(output_path.parent)
        _atomic_snapshot(output_path, payload)
        return payload
    except Exception as error:
        payload["status"] = "physical_failed_preserved"
        payload["failure"] = {"type": type(error).__name__, "message": str(error)}
        _atomic_snapshot(output_path, payload)
        raise
    finally:
        if owns_backend and backend is not None:
            backend.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-status", required=True)
    parser.add_argument("--confirm-exclusive-board", action="store_true")
    parser.add_argument("--confirm-ac-power", action="store_true")
    parser.add_argument("--confirm-no-other-workloads", action="store_true")
    parser.add_argument("--reviewer-signoff")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_authorized_experiment(args)
    except (AccessBlocked, FileExistsError, FileNotFoundError, KeyError, OSError, PhysicalStop, RuntimeError, ValueError) as error:
        print(f"PAPER3_RK3576_FRAGMENTATION_BLOCKED: {error}", file=sys.stderr)
        return 1
    print(f"PAPER3_RK3576_FRAGMENTATION_CAPTURE_COMPLETE samples={len(result['records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
