"""Evidence-boundary tests for the Paper 3 fragmentation manifest builder."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from src.deployment.benchmark_rk3576_fragmentation_pairs import (
    AUTHORIZED_STATUS,
    CONTRACT_AUTHORIZATION_FIELDS,
    PENDING_AUTHORIZATION_VALUES,
    PENDING_STATUS,
    PINNED_BOARD_IDENTITY,
    PINNED_RKNN_RUNTIME_LIBRARY_SHA256,
    contract_authorization_phase,
    contract_invariant_sha256,
)
from src.report.paper3_rk3576_fragmentation_pairs_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = yaml.safe_load(
    (ROOT / "configs/paper3_rk3576_fragmentation_pairs_v1.yaml").read_text(encoding="utf-8")
)
BASELINE = "a" * 40


def set_dotted_value(value: dict, dotted: str, replacement: object) -> None:
    parts = dotted.split(".")
    target = value
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = replacement


def pending_contract() -> dict:
    contract = copy.deepcopy(CONTRACT)
    for dotted, replacement in zip(
        CONTRACT_AUTHORIZATION_FIELDS, PENDING_AUTHORIZATION_VALUES, strict=True
    ):
        set_dotted_value(contract, dotted, replacement)
    return contract


def authorized_result_contract() -> dict:
    pending = pending_contract()
    contract = copy.deepcopy(pending)
    authorized_values = (
        AUTHORIZED_STATUS,
        True,
        True,
        "authorized_physical_capture",
        PINNED_BOARD_IDENTITY,
        PINNED_RKNN_RUNTIME_LIBRARY_SHA256,
        BASELINE,
        False,
    )
    for dotted, replacement in zip(
        CONTRACT_AUTHORIZATION_FIELDS, authorized_values, strict=True
    ):
        set_dotted_value(contract, dotted, replacement)
    assert contract_invariant_sha256(contract) == contract_invariant_sha256(pending)
    return contract


def write_result(root: Path, relative: str, content: str) -> None:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def test_pending_and_authorized_contracts_reach_phase_appropriate_manifest_boundaries(
    tmp_path: Path,
) -> None:
    pending = pending_contract()
    authorized = authorized_result_contract()

    assert contract_authorization_phase(pending) == PENDING_STATUS
    assert contract_authorization_phase(authorized) == AUTHORIZED_STATUS
    with pytest.raises(ValueError, match="still forbids"):
        build_manifest(pending, root=tmp_path, control_paths=())
    with pytest.raises(FileNotFoundError):
        build_manifest(authorized, root=tmp_path, control_paths=())


def test_manifest_hashes_only_explicit_control_and_result_files(tmp_path: Path) -> None:
    contract = authorized_result_contract()
    expected = contract["expected_outputs_after_future_authorization"]
    controls = (Path("control_a.py"), Path("control_b.yaml"))
    for path in controls:
        destination = tmp_path / path
        destination.write_text(f"content:{path}\n", encoding="utf-8")
    write_result(tmp_path, expected["raw_capture"], '{"status":"physical_complete"}\n')
    write_result(
        tmp_path,
        expected["analysis_json"],
        json.dumps(
            {
                "experiment_id": "paper3_rk3576_fragmentation_pairs_v1",
                "status": "analysis_complete",
                "disposition": "PRIMARY_GATES_PASS",
                "all_primary_gates_passed": True,
            }
        )
        + "\n",
    )
    write_result(tmp_path, expected["analysis_markdown"], "# analysis\n")
    write_result(tmp_path, expected["latex_candidate"], "% analysis\n")

    manifest = build_manifest(contract, root=tmp_path, control_paths=controls)

    assert manifest["status"] == "evidence_complete"
    assert len(manifest["evidence_files"]) == 4
    assert len(manifest["control_files"]) == 2
    assert all(
        len(row["sha256"]) == 64
        for row in manifest["evidence_files"] + manifest["control_files"]
    )
    assert {row["path"] for row in manifest["evidence_files"]} == {
        expected["raw_capture"],
        expected["analysis_json"],
        expected["analysis_markdown"],
        expected["latex_candidate"],
    }


def test_manifest_refuses_missing_expected_result(tmp_path: Path) -> None:
    contract = authorized_result_contract()
    with pytest.raises(FileNotFoundError):
        build_manifest(contract, root=tmp_path, control_paths=())
