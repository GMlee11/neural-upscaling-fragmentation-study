"""Tests for Paper 3's independent NPU-only summary."""

from __future__ import annotations

import json
from pathlib import Path

from src.report.paper3_npu_fragmentation_summary_v1 import analyze, load_json


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_INPUT = ROOT / "results/tables/codesign_matched_masks_rk3576_v1_public.json"
PRIVATE_INPUT = ROOT / "results/tables/codesign_matched_masks_rk3576_v1.json"
INPUT = PUBLIC_INPUT if PUBLIC_INPUT.is_file() else PRIVATE_INPUT


def test_summary_reproduces_frozen_within_backend_tradeoff() -> None:
    report = analyze(load_json(INPUT))
    aggregate = report["aggregate"]
    assert report["patterns"] == 15
    assert abs(aggregate["row_span_call_reduction_percent"] - 80.3333333333) < 1e-9
    assert abs(aggregate["row_span_dispatched_core_increase_percent"] - 34.3333333333) < 1e-9
    assert abs(aggregate["row_span_tensor_byte_increase_percent"] - 24.5047619048) < 1e-9
    assert abs(aggregate["row_span_service_reduction_percent"] - 10.9578880151) < 1e-9


def test_summary_has_no_external_paper_or_fpga_dependency() -> None:
    report = analyze(load_json(INPUT))
    compact = json.dumps(report, sort_keys=True).lower()
    for forbidden in (
        "paper 1",
        "paper 2",
        "paper1",
        "paper2",
        "fpga",
        "zybo",
        "exact-closure",
        "exact_closure",
    ):
        assert forbidden not in compact
    assert report["claim_boundary"] == {
        "cross_platform_performance_claimed": False,
        "external_accelerator_denominator_used": False,
        "independent_submission_artifact": True,
        "work_accounting": "selected_masks_and_dispatched_rknn_tensor_shapes",
    }


def test_every_pair_is_self_consistent() -> None:
    report = analyze(load_json(INPUT))
    assert [row["pattern_id"] for row in report["pairs"]] == list(range(1, 16))
    for row in report["pairs"]:
        assert row["selected_tiles"] > 0
        assert row["row_span_calls"] == row["active_rows"]
        assert row["exact_calls"] == row["selected_tiles"]
        assert row["row_span_dispatched_core_pixels"] >= row["selected_core_pixels"]
