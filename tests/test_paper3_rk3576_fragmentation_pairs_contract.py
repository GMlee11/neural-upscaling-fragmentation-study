"""Offline contract tests for the unexecuted Paper 3 RK3576 pair study."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "paper3_rk3576_fragmentation_pairs_v1.yaml"
PREFLIGHT = ROOT / "results" / "preflight" / "paper3_rk3576_fragmentation_pairs_v1.json"
AUTHORIZATION_FIELDS = (
    "status",
    "authorization_boundary.physical_rk3576_authorized",
    "authorization_boundary.result_files_authorized",
    "authorization_boundary.required_next_state",
    "device_runtime_controls.board_identity",
    "device_runtime_controls.rknn_runtime_library_sha256",
    "device_runtime_controls.accepted_repository_commit",
    "expected_outputs_after_future_authorization.current_phase_outputs_must_not_exist",
)
PENDING_VALUES = (
    "ACCESS_REQUIRED",
    False,
    False,
    "explicit_orchestrator_access_authorization",
    "TO_BE_PINNED_BEFORE_ACCESS",
    "TO_BE_PINNED_BEFORE_ACCESS",
    "TO_BE_PINNED_BEFORE_ACCESS",
    True,
)
PINNED_BOARD_IDENTITY = "928996c84e65dc3f119cd65f232d2ad6b4d4acd12a0d76ed7b4ecdec016b199c"
PINNED_RKNN_RUNTIME_LIBRARY_SHA256 = "cf6ea624489225ddc9d334b370c64083b0fd9d6a5c9166eed47d999cc85a806b"
PENDING_GATES = [
    "contract_status_is_ACCESS_REQUIRED",
    "physical_rk3576_authorized_is_false",
    "result_files_authorized_is_false",
    "board_identity_not_pinned",
    "rknn_runtime_library_sha256_not_pinned",
    "accepted_repository_commit_not_pinned",
    "ssh_authentication_not_available",
    "exclusive_session_ac_power_and_reviewer_signoff_not_attested",
]
AUTHORIZED_GATES = [
    "ssh_authentication_not_available",
    "exclusive_session_ac_power_and_reviewer_signoff_not_attested",
]
CONTRACT_INVARIANT_SHA256 = "a1e655deef1af5dbe81bc824046e65e26c056988bea9c1057bc9afb2a59e5d04"


def load_contract() -> dict:
    """Load the preregistration without importing a board-facing runner."""

    value = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def dotted_value(value: dict, dotted: str) -> object:
    target: object = value
    for part in dotted.split("."):
        assert isinstance(target, dict) and part in target
        target = target[part]
    return target


def set_dotted_value(value: dict, dotted: str, replacement: object) -> None:
    parts = dotted.split(".")
    target = value
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = replacement


def authorization_values(contract: dict) -> tuple[object, ...]:
    return tuple(dotted_value(contract, field) for field in AUTHORIZATION_FIELDS)


def authorization_phase(contract: dict) -> str | None:
    values = authorization_values(contract)
    if values == PENDING_VALUES:
        return "ACCESS_REQUIRED"
    if (
        values[:6]
        == (
            "AUTHORIZED_FOR_ACCESS",
            True,
            True,
            "authorized_physical_capture",
            PINNED_BOARD_IDENTITY,
            PINNED_RKNN_RUNTIME_LIBRARY_SHA256,
        )
        and isinstance(values[6], str)
        and re.fullmatch(r"[0-9a-f]{40}", values[6]) is not None
        and values[7] is False
    ):
        return "AUTHORIZED_FOR_ACCESS"
    return None


def phase_contract(contract: dict, phase: str, *, baseline: str = "a" * 40) -> dict:
    result = copy.deepcopy(contract)
    values = PENDING_VALUES
    if phase == "AUTHORIZED_FOR_ACCESS":
        values = (
            phase,
            True,
            True,
            "authorized_physical_capture",
            PINNED_BOARD_IDENTITY,
            PINNED_RKNN_RUNTIME_LIBRARY_SHA256,
            baseline,
            False,
        )
    for dotted, replacement in zip(AUTHORIZATION_FIELDS, values, strict=True):
        set_dotted_value(result, dotted, replacement)
    return result


def invariant_sha256(contract: dict) -> str:
    projected = copy.deepcopy(contract)
    for dotted in AUTHORIZATION_FIELDS:
        parts = dotted.split(".")
        target = projected
        for part in parts[:-1]:
            target = target[part]
        del target[parts[-1]]

    def encode_date(candidate: object) -> str:
        if isinstance(candidate, date):
            return candidate.isoformat()
        raise TypeError(type(candidate).__name__)

    encoded = json.dumps(projected, sort_keys=True, separators=(",", ":"), default=encode_date).encode()
    return hashlib.sha256(encoded).hexdigest()


def count_runs(row: str) -> int:
    """Count left-to-right zero-to-one transitions in one frozen row."""

    return sum(bit == "1" and (index == 0 or row[index - 1] == "0") for index, bit in enumerate(row))


def row_span_coverage(row: str) -> int:
    """Return inclusive leftmost-to-rightmost selected coverage."""

    selected = [index for index, bit in enumerate(row) if bit == "1"]
    assert selected
    return selected[-1] - selected[0] + 1


def mask_integer(rows: list[str]) -> int:
    """Encode the declared row-major bit-index convention."""

    value = 0
    for row_index, row in enumerate(rows):
        for column_index, bit in enumerate(row):
            if bit == "1":
                value |= 1 << (row_index * 10 + column_index)
    return value


def test_contract_authorization_phase_and_checked_in_preflight_are_coherent() -> None:
    """Accept either complete phase, freeze all other fields, and keep preflight offline."""

    contract = load_contract()
    authorization = contract["authorization_boundary"]
    pending = phase_contract(contract, "ACCESS_REQUIRED")
    authorized = phase_contract(contract, "AUTHORIZED_FOR_ACCESS")

    assert contract["experiment_id"] == "paper3_rk3576_fragmentation_pairs_v1"
    assert contract["publication_decision"] == "GO_REQUIRED_FOR_FULL_PAPER"
    assert contract["sealed_before_physical_access"] is True
    assert authorization["mask_generation_from_sealed_traces_authorized"] is False
    assert authorization["sealed_trace_evaluation_authorized"] is False
    assert authorization_phase(contract) in {"ACCESS_REQUIRED", "AUTHORIZED_FOR_ACCESS"}
    assert authorization_values(pending) == PENDING_VALUES
    assert authorization_phase(pending) == "ACCESS_REQUIRED"
    assert authorization_phase(authorized) == "AUTHORIZED_FOR_ACCESS"
    assert invariant_sha256(contract) == invariant_sha256(pending) == invariant_sha256(authorized)
    assert invariant_sha256(contract) == CONTRACT_INVARIANT_SHA256

    authorized_values = authorization_values(authorized)
    for index, (dotted, replacement) in enumerate(zip(AUTHORIZATION_FIELDS, authorized_values, strict=True)):
        partial = copy.deepcopy(pending)
        set_dotted_value(partial, dotted, replacement)
        assert authorization_phase(partial) is None, f"partial transition accepted at index {index}: {dotted}"

    expected = contract["expected_outputs_after_future_authorization"]
    physical_paths = {
        name: relative_path
        for name, relative_path in expected.items()
        if name != "current_phase_outputs_must_not_exist"
    }
    assert len(physical_paths) == 5
    phase = authorization_phase(contract)
    outputs_must_not_exist = bool(expected["current_phase_outputs_must_not_exist"])
    assert outputs_must_not_exist is (phase == "ACCESS_REQUIRED")
    resolved_outputs = {
        name: ROOT / relative_path for name, relative_path in physical_paths.items()
    }
    if outputs_must_not_exist:
        for name, path in resolved_outputs.items():
            assert not path.exists(), f"pre-capture output exists: {name}={path}"
    else:
        assert phase == "AUTHORIZED_FOR_ACCESS"
        for name, path in resolved_outputs.items():
            assert path.is_file(), f"authorized output is missing: {name}={path}"

        evidence_manifest = json.loads(
            resolved_outputs["evidence_manifest"].read_text(encoding="utf-8")
        )
        assert evidence_manifest["experiment_id"] == contract["experiment_id"]
        assert evidence_manifest["status"] == "evidence_complete"
        assert evidence_manifest["physical_execution_measured"] is True
        assert evidence_manifest["claim_boundary"] == contract["claim_boundary"]
        assert evidence_manifest["all_primary_gates_passed"] is True
        evidence_records = {
            row["path"]: row for row in evidence_manifest["evidence_files"]
        }
        evidence_outputs = {
            relative_path
            for name, relative_path in physical_paths.items()
            if name != "evidence_manifest"
        }
        assert set(evidence_records) == evidence_outputs
        for relative_path in evidence_outputs:
            path = ROOT / relative_path
            record = evidence_records[relative_path]
            assert record["bytes"] == path.stat().st_size
            assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

        raw_capture = json.loads(resolved_outputs["raw_capture"].read_text(encoding="utf-8"))
        assert raw_capture["status"] == "physical_complete_pending_offline_analysis"
        assert raw_capture["failure"] is None
        assert len(raw_capture["records"]) == contract["execution_design"]["measured_samples"] == 60

    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    expected_gates = PENDING_GATES if contract["status"] == "ACCESS_REQUIRED" else AUTHORIZED_GATES
    assert preflight["contract_status"] == contract["status"]
    assert preflight["dependencies"]["contract"]["sha256"] == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    assert preflight["unresolved_gates"] == expected_gates
    assert preflight["authorized_for_samples"] is False
    assert preflight["physical_execution_measured"] is False
    assert preflight["warmups_executed"] == 0
    assert preflight["measured_samples"] == 0


def test_six_masks_form_three_density_matched_geometry_pairs() -> None:
    """Hold count and active rows fixed while maximizing run contrast."""

    contract = load_contract()
    mask_contract = contract["mask_contract"]
    masks = mask_contract["masks"]
    by_pair: dict[str, list[dict]] = {}

    assert mask_contract["mask_count"] == 6
    assert mask_contract["density_pair_count"] == 3
    assert mask_contract["mode_stratified_pair_cells"] == 6
    assert len(masks) == 6

    canonical_rows = {mask["id"]: mask["rows"] for mask in masks}
    canonical = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == mask_contract["canonical_rows_sha256"]

    for mask in masks:
        rows = mask["rows"]
        assert len(rows) == 6
        assert all(len(row) == 10 and set(row) <= {"0", "1"} for row in rows)
        assert all("1" in row for row in rows)
        selected = sum(row.count("1") for row in rows)
        runs = sum(count_runs(row) for row in rows)
        coverage = sum(row_span_coverage(row) for row in rows)
        assert selected == mask["selected_tiles"]
        assert selected == 60 * mask["density_percent"] // 100
        assert all(row.count("1") == mask["selected_per_row"] for row in rows)
        assert runs == mask["horizontal_runs"]
        assert coverage == mask["row_span_coverage_tiles"]
        assert int(mask["mask_hex"], 16) == mask_integer(rows)
        by_pair.setdefault(mask["pair_id"], []).append(mask)

    assert set(by_pair) == {"density_20", "density_40", "density_60"}
    for pair in by_pair.values():
        assert len(pair) == 2
        compact = next(mask for mask in pair if mask["geometry"] == "compact")
        fragmented = next(mask for mask in pair if mask["geometry"] == "maximally_fragmented")
        assert compact["selected_tiles"] == fragmented["selected_tiles"]
        assert compact["density_percent"] == fragmented["density_percent"]
        assert compact["horizontal_runs"] == 6
        per_row_selected = fragmented["selected_per_row"]
        maximum_runs_per_row = min(per_row_selected, 10 - per_row_selected + 1)
        assert fragmented["horizontal_runs"] == 6 * maximum_runs_per_row
        assert fragmented["row_span_coverage_tiles"] == 60
        assert compact["row_span_coverage_tiles"] == compact["selected_tiles"]


def test_execution_matrix_and_frozen_decision_rules_are_complete() -> None:
    """Require both modes, five repeats, sixty samples, and adverse rules."""

    contract = load_contract()
    design = contract["execution_design"]
    gates = contract["frozen_gates"]

    assert design["dispatch_modes"] == ["exact", "row_spans"]
    assert design["repetitions_per_mask_mode"] == 5
    assert design["warmup_iterations_per_shape"] == 2
    assert design["measured_samples"] == 6 * 2 * 5 == 60
    assert design["blocks"] == 5
    assert design["samples_per_block"] == 12
    assert design["outlier_removal_authorized"] is False
    assert design["replacement_samples_authorized"] is False
    assert contract["analysis"]["regression_authorized"] is False
    assert contract["analysis"]["p_value_claim_authorized"] is False

    assert gates["integrity"]["observed_measured_samples_must_equal"] == 60
    assert gates["structural_padding"]["row_span_fragmented_executed_gmac_gt_compact_at_all_densities"] is True
    assert gates["structural_padding"]["exact_mode_executed_gmac_and_tensor_bytes_equal_within_pair"] is True
    assert gates["physical_service"]["row_span_positive_paired_service_delta_at_density_count"] == 3
    assert gates["physical_service"]["row_span_minimum_density_pairs_with_median_penalty_ge_5_percent"] == 2
    assert gates["exact_negative_control"]["max_absolute_median_paired_service_delta_percent_each_density"] == 5.0


def test_input_and_runtime_controls_are_frozen_without_trace_dependencies() -> None:
    """Use deterministic pixels and require the exact authorization allowlist."""

    contract = load_contract()
    fixture = contract["input_fixture"]
    runtime = contract["device_runtime_controls"]

    pixels = bytearray()
    for y in range(fixture["height"]):
        for x in range(fixture["width"]):
            pixels.extend(
                (
                    (3 * x + 5 * y + 17) % 256,
                    (7 * x + 11 * y + 29) % 256,
                    (13 * x + 2 * y + 47) % 256,
                )
            )
    assert len(pixels) == fixture["byte_length"]
    assert hashlib.sha256(pixels).hexdigest() == fixture["raw_bytes_sha256"]
    assert fixture["origin"] == "synthetic_and_independent_of_all_sealed_traces"
    assert fixture["quality_claim_authorized"] is False
    assert "trace_manifest" not in contract["workload"]
    assert "selector" not in contract["workload"]

    assert authorization_phase(contract) in {"ACCESS_REQUIRED", "AUTHORIZED_FOR_ACCESS"}
    assert runtime["accepted_repository_commit_role"] == "accepted_implementation_baseline"
    policy = runtime["repository_authorization_policy"]
    assert policy["require_clean_checkout"] is True
    assert policy["baseline_must_be_ancestor_of_authorized_checkout"] is True
    assert policy["authorization_metadata_paths"] == [
        "configs/paper3_rk3576_fragmentation_pairs_v1.yaml",
        "docs/paper3_rk3576_fragmentation_pairs_v1.md",
        "results/preflight/paper3_rk3576_fragmentation_pairs_v1.json",
    ]
    assert tuple(policy["contract_authorization_fields"]) == AUTHORIZATION_FIELDS
    assert set(policy["executable_baseline_paths"]) == {
        "scripts/rk3576/run_paper3_rk3576_fragmentation_pairs_v1.sh",
        "src/deployment/benchmark_rk3576_fragmentation_pairs.py",
        "src/deployment/prepare_paper3_rk3576_fragmentation_pairs.py",
        "src/report/paper3_rk3576_fragmentation_pairs_v1.py",
        "src/report/paper3_rk3576_fragmentation_pairs_manifest.py",
        "tests/test_paper3_rk3576_fragmentation_pairs_contract.py",
        "tests/test_benchmark_rk3576_fragmentation_pairs.py",
        "tests/test_paper3_rk3576_fragmentation_pairs_analysis.py",
        "tests/test_paper3_rk3576_fragmentation_pairs_manifest.py",
        "src/deployment/benchmark_rk3576_sparse.py",
        "src/deployment/board_probe.py",
        "src/runtime/sparse_refinement.py",
    }
    assert runtime["start_temperature_max_c"] < runtime["abort_temperature_c"]


def test_claim_boundary_rejects_generalization_and_superiority() -> None:
    """A passing pair study remains one-model, one-device evidence."""

    claims = load_contract()["claim_boundary"]

    assert claims["selector_generalization_claimed"] is False
    assert claims["natural_mask_prevalence_claimed"] is False
    assert claims["model_family_generalization_claimed"] is False
    assert claims["physical_energy_claimed"] is False
    assert claims["rk3576_720p_realtime_claimed"] is False
    assert claims["platform_superiority_claimed"] is False
    assert claims["cross_platform_bit_identity_claimed"] is False
