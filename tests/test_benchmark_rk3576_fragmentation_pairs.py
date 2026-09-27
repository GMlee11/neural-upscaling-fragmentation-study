"""Offline tests for the blocked Paper 3 RK3576 physical runner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

import src.deployment.benchmark_rk3576_fragmentation_pairs as fragmentation_runner
from src.deployment.benchmark_rk3576_fragmentation_pairs import (
    AUTHORIZED_OFFLINE_GATES,
    AUTHORIZED_STATUS,
    CONTRACT_AUTHORIZATION_FIELDS,
    DEFAULT_CATALOG,
    DEFAULT_CONTRACT,
    PENDING_AUTHORIZATION_VALUES,
    PENDING_OFFLINE_GATES,
    PENDING_STATUS,
    PINNED_BOARD_IDENTITY,
    PINNED_RKNN_RUNTIME_LIBRARY_SHA256,
    AccessBlocked,
    build_offline_preflight,
    build_parser,
    build_schedule,
    contract_authorization_phase,
    contract_authorization_values,
    contract_invariant_sha256,
    enforce_output_boundary,
    generate_coordinate_fixture,
    load_contract,
    load_frozen_masks,
    run_authorized_experiment,
    verify_repository_baseline,
)
from src.deployment.prepare_paper3_rk3576_fragmentation_pairs import write_exclusive


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "a" * 40
HEAD = "b" * 40
CONTRACT_INVARIANT_SHA256 = "a1e655deef1af5dbe81bc824046e65e26c056988bea9c1057bc9afb2a59e5d04"


class FakeGit:
    """Return controlled read-only Git answers without creating a repository."""

    def __init__(
        self,
        baseline_contract: dict,
        *,
        dirty: bool = False,
        descendant: bool = True,
        drift_path: str | None = None,
        changed_paths: tuple[str, ...] | None = None,
    ) -> None:
        self.baseline_contract = baseline_contract
        self.dirty = dirty
        self.descendant = descendant
        self.drift_path = drift_path
        self.changed_paths = changed_paths or (
            "configs/paper3_rk3576_fragmentation_pairs_v1.yaml",
            "docs/paper3_rk3576_fragmentation_pairs_v1.md",
            "results/preflight/paper3_rk3576_fragmentation_pairs_v1.json",
        )
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, arguments: tuple[str, ...] | list[str]) -> subprocess.CompletedProcess[str]:
        args = tuple(arguments)
        self.calls.append(args)
        if args[0] == "status":
            return subprocess.CompletedProcess(args, 0, " M dirty.py\n" if self.dirty else "", "")
        if args == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(args, 0, HEAD + "\n", "")
        if args[:2] == ("rev-parse", "--verify"):
            return subprocess.CompletedProcess(args, 0, BASELINE + "\n", "")
        if args[0] == "merge-base":
            return subprocess.CompletedProcess(args, 0 if self.descendant else 1, "", "")
        if args[0] == "diff":
            return subprocess.CompletedProcess(args, 0, "\n".join(self.changed_paths) + "\n", "")
        if args[0] == "show":
            return subprocess.CompletedProcess(
                args,
                0,
                yaml.safe_dump(self.baseline_contract, sort_keys=False),
                "",
            )
        if args[0] == "rev-parse" and ":" in args[1]:
            path = args[1].split(":", 1)[1]
            return subprocess.CompletedProcess(args, 0, f"blob-{path}\n", "")
        if args[:2] == ("hash-object", "--"):
            path = args[2]
            blob = f"blob-{path}"
            if path == self.drift_path:
                blob = "drifted-blob"
            return subprocess.CompletedProcess(args, 0, blob + "\n", "")
        raise AssertionError(f"unexpected Git query: {args}")


def set_dotted_value(value: dict, dotted: str, replacement: object) -> None:
    parts = dotted.split(".")
    target = value
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = replacement


def pending_contract() -> dict:
    contract = copy.deepcopy(load_contract(DEFAULT_CONTRACT))
    for dotted, replacement in zip(CONTRACT_AUTHORIZATION_FIELDS, PENDING_AUTHORIZATION_VALUES, strict=True):
        set_dotted_value(contract, dotted, replacement)
    return contract


def authorized_descendant_contract() -> tuple[dict, dict]:
    baseline = pending_contract()
    current = copy.deepcopy(baseline)
    current["status"] = AUTHORIZED_STATUS
    current["authorization_boundary"]["physical_rk3576_authorized"] = True
    current["authorization_boundary"]["result_files_authorized"] = True
    current["authorization_boundary"]["required_next_state"] = "authorized_physical_capture"
    current["device_runtime_controls"]["board_identity"] = PINNED_BOARD_IDENTITY
    current["device_runtime_controls"]["rknn_runtime_library_sha256"] = PINNED_RKNN_RUNTIME_LIBRARY_SHA256
    current["device_runtime_controls"]["accepted_repository_commit"] = BASELINE
    current["expected_outputs_after_future_authorization"]["current_phase_outputs_must_not_exist"] = False
    return baseline, current


def test_fixture_masks_and_schedule_are_synthetic_frozen_and_complete() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    fixture = generate_coordinate_fixture(contract)
    masks = load_frozen_masks(contract)
    schedule = build_schedule(contract, masks)

    assert fixture.shape == (360, 640, 3)
    assert fixture.dtype.name == "uint8"
    assert hashlib.sha256(fixture.tobytes()).hexdigest() == contract["input_fixture"]["raw_bytes_sha256"]
    assert len(masks) == 6
    assert {mask.density_percent for mask in masks} == {20, 40, 60}
    assert all(mask.active_rows == 6 for mask in masks)
    assert len(schedule) == 60
    assert [row["execution_order"] for row in schedule] == list(range(1, 61))
    assert {row["dispatch_mode"] for row in schedule} == {"exact", "row_spans"}
    for index in range(0, 60, 2):
        pair = schedule[index : index + 2]
        assert len({row["repeat_index"] for row in pair}) == 1
        assert len({row["pair_id"] for row in pair}) == 1
        assert len({row["dispatch_mode"] for row in pair}) == 1
        assert {row["geometry"] for row in pair} == {"compact", "maximally_fragmented"}
    assert schedule == build_schedule(contract, masks)


def test_schedule_rejects_nonpositive_or_mismatched_frozen_count() -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    masks = load_frozen_masks(contract)

    nonpositive = copy.deepcopy(contract)
    nonpositive["execution_design"]["measured_samples"] = 0
    with pytest.raises(ValueError, match="must be positive"):
        build_schedule(nonpositive, masks)

    mismatched = copy.deepcopy(contract)
    mismatched["execution_design"]["measured_samples"] = 59
    with pytest.raises(ValueError, match="frozen 60 samples"):
        build_schedule(mismatched, masks)


def test_offline_preflight_hashes_every_accepted_dependency_without_authorizing_samples(monkeypatch) -> None:
    current = load_contract(DEFAULT_CONTRACT)
    current_phase = contract_authorization_phase(current)
    expected_gates = (
        PENDING_OFFLINE_GATES if current_phase == PENDING_STATUS else AUTHORIZED_OFFLINE_GATES
    )
    record = build_offline_preflight(DEFAULT_CONTRACT, DEFAULT_CATALOG)

    assert record["contract_status"] == current_phase
    assert record["authorized_for_samples"] is False
    assert record["physical_execution_measured"] is False
    assert record["warmups_executed"] == 0
    assert record["measured_samples"] == 0
    assert record["schedule"]["entry_count"] == 60
    assert len(record["dependencies"]["rknn_graphs"]) == 8
    implementation_paths = {
        row["path"] for row in record["dependencies"]["implementation"]
    }
    executable_paths = set(
        load_contract(DEFAULT_CONTRACT)["device_runtime_controls"]
        ["repository_authorization_policy"]["executable_baseline_paths"]
    )
    assert executable_paths <= implementation_paths
    assert record["unresolved_gates"] == list(expected_gates)
    assert all(row["sha256"] for row in record["dependencies"]["rknn_graphs"])

    _, authorized = authorized_descendant_contract()
    monkeypatch.setattr(fragmentation_runner, "load_contract", lambda _path: copy.deepcopy(authorized))
    authorized_record = build_offline_preflight(DEFAULT_CONTRACT, DEFAULT_CATALOG)
    assert authorized_record["contract_status"] == AUTHORIZED_STATUS
    assert authorized_record["unresolved_gates"] == list(AUTHORIZED_OFFLINE_GATES)
    assert authorized_record["authorized_for_samples"] is False
    assert authorized_record["physical_execution_measured"] is False
    assert authorized_record["warmups_executed"] == 0
    assert authorized_record["measured_samples"] == 0


def test_explicit_pending_contract_blocks_before_output_or_backend(tmp_path: Path) -> None:
    contract_path = tmp_path / "pending_contract.yaml"
    contract_path.write_text(yaml.safe_dump(pending_contract(), sort_keys=False), encoding="utf-8")
    output = tmp_path / "must_not_exist.json"
    args = argparse.Namespace(
        contract=contract_path,
        catalog=DEFAULT_CATALOG,
        output=output,
        require_status=AUTHORIZED_STATUS,
        confirm_exclusive_board=True,
        confirm_ac_power=True,
        confirm_no_other_workloads=True,
        reviewer_signoff="offline-test",
    )

    with pytest.raises(AccessBlocked, match="contract status is ACCESS_REQUIRED"):
        run_authorized_experiment(args)
    assert not output.exists()
    assert not output.parent.joinpath(output.name + ".tmp").exists()


def test_output_boundary_refuses_noncanonical_existing_and_nonempty_roots(tmp_path: Path) -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    expected = tmp_path / contract["expected_outputs_after_future_authorization"]["raw_capture"]

    with pytest.raises(ValueError, match="preregistered path"):
        enforce_output_boundary(tmp_path / "other.json", contract, root=tmp_path)
    expected.parent.mkdir(parents=True)
    (expected.parent / "contaminant.txt").write_text("blocked", encoding="utf-8")
    with pytest.raises(FileExistsError, match="nonempty"):
        enforce_output_boundary(expected, contract, root=tmp_path)
    (expected.parent / "contaminant.txt").unlink()
    expected.write_text("already owned", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        enforce_output_boundary(expected, contract, root=tmp_path)


def test_runner_exposes_no_dataset_or_retuning_input_surface() -> None:
    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    assert options == {
        "-h",
        "--help",
        "--contract",
        "--catalog",
        "--output",
        "--require-status",
        "--confirm-exclusive-board",
        "--confirm-ac-power",
        "--confirm-no-other-workloads",
        "--reviewer-signoff",
    }


def test_clean_metadata_only_descendant_with_baseline_blobs_is_allowed() -> None:
    baseline, current = authorized_descendant_contract()
    fake = FakeGit(baseline)

    assert contract_authorization_phase(baseline) == PENDING_STATUS
    assert contract_authorization_phase(current) == AUTHORIZED_STATUS
    assert contract_invariant_sha256(baseline) == contract_invariant_sha256(current)
    assert contract_invariant_sha256(baseline) == CONTRACT_INVARIANT_SHA256
    authorized_values = contract_authorization_values(current)
    for dotted, replacement in zip(CONTRACT_AUTHORIZATION_FIELDS, authorized_values, strict=True):
        partial = copy.deepcopy(baseline)
        set_dotted_value(partial, dotted, replacement)
        with pytest.raises(AccessBlocked, match="mixed, partial, or unknown phase"):
            contract_authorization_phase(partial)
    altered_allowlist = copy.deepcopy(current)
    altered_allowlist["device_runtime_controls"]["repository_authorization_policy"][
        "contract_authorization_fields"
    ] = list(reversed(CONTRACT_AUTHORIZATION_FIELDS))
    with pytest.raises(AccessBlocked, match="exact eight-field allowlist"):
        contract_authorization_phase(altered_allowlist)

    result = verify_repository_baseline(current, root=ROOT, run_git=fake)

    assert result["accepted_implementation_baseline"] == BASELINE
    assert result["authorized_checkout_head"] == HEAD
    assert result["clean_checkout"] is True
    assert len(result["verified_executable_blobs"]) == len(
        current["device_runtime_controls"]["repository_authorization_policy"]["executable_baseline_paths"]
    )
    assert {call[0] for call in fake.calls} <= {
        "status",
        "rev-parse",
        "merge-base",
        "diff",
        "show",
        "hash-object",
    }


def test_dirty_checkout_is_rejected_before_blob_verification() -> None:
    baseline, current = authorized_descendant_contract()
    fake = FakeGit(baseline, dirty=True)

    with pytest.raises(AccessBlocked, match="checkout is dirty"):
        verify_repository_baseline(current, root=ROOT, run_git=fake)
    assert fake.calls == [("status", "--porcelain=v1", "--untracked-files=all")]


def test_non_descendant_authorization_checkout_is_rejected() -> None:
    baseline, current = authorized_descendant_contract()
    fake = FakeGit(baseline, descendant=False)

    with pytest.raises(AccessBlocked, match="not an ancestor"):
        verify_repository_baseline(current, root=ROOT, run_git=fake)


def test_executable_blob_drift_is_rejected_even_when_diff_claims_metadata_only() -> None:
    baseline, current = authorized_descendant_contract()
    runner_path = "src/deployment/benchmark_rk3576_fragmentation_pairs.py"
    fake = FakeGit(baseline, drift_path=runner_path)

    with pytest.raises(AccessBlocked, match=f"executable drift.*{runner_path}"):
        verify_repository_baseline(current, root=ROOT, run_git=fake)


def test_contract_threshold_drift_is_not_authorization_metadata() -> None:
    baseline, current = authorized_descendant_contract()
    current["frozen_gates"]["integrity"]["max_cross_mode_mean_abs_float"] = 0.5
    fake = FakeGit(baseline)

    with pytest.raises(AccessBlocked, match="contract drift exceeds"):
        verify_repository_baseline(current, root=ROOT, run_git=fake)


def test_preflight_record_writer_is_exclusive(tmp_path: Path) -> None:
    output = tmp_path / "preflight.json"
    write_exclusive(output, {"authorized_for_samples": False})
    with pytest.raises(FileExistsError):
        write_exclusive(output, {"authorized_for_samples": True})
