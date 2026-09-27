"""Frozen-decision tests for the Paper 3 fragmentation-pair analyzer."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from src.report.paper3_rk3576_fragmentation_pairs_v1 import (
    analyze_capture,
    render_latex,
    render_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = yaml.safe_load(
    (ROOT / "configs/paper3_rk3576_fragmentation_pairs_v1.yaml").read_text(encoding="utf-8")
)


def synthetic_complete_raw() -> dict:
    records = []
    order = 0
    for repeat in range(1, 6):
        for density, selected in ((20, 12), (40, 24), (60, 36)):
            for mode in ("exact", "row_spans"):
                for geometry in ("compact", "maximally_fragmented"):
                    order += 1
                    fragmented = geometry == "maximally_fragmented"
                    if mode == "exact":
                        service = 10.1 if fragmented else 10.0
                        gmac = selected * 0.04
                        bytes_per_pixel = 6.0
                        calls = selected
                    else:
                        service = 11.0 if fragmented else 10.0
                        gmac = density / 100.0 + (0.5 if fragmented else 0.0)
                        bytes_per_pixel = density / 10.0 + (4.0 if fragmented else 0.0)
                        calls = 6
                    committed = selected * 64 * 60 * 4
                    records.append(
                        {
                            "repeat_index": repeat,
                            "block_index": repeat,
                            "execution_order": order,
                            "mask_id": f"p{density}_{'fragmented' if fragmented else 'compact'}",
                            "pair_id": f"density_{density}",
                            "geometry": geometry,
                            "density_percent": density,
                            "selected_tiles": selected,
                            "active_rows": 6,
                            "horizontal_runs": 12 if fragmented else 6,
                            "row_span_coverage_tiles": 60 if fragmented else selected,
                            "dispatch_mode": mode,
                            "npu_calls": calls,
                            "executed_gmac": gmac,
                            "input_tensor_bytes": 1_000,
                            "output_tensor_bytes": 2_000,
                            "tensor_bytes_per_committed_hr_pixel": bytes_per_pixel,
                            "extraction_ms": 1.0,
                            "planning_ms": 0.1,
                            "npu_inference_ms": 4.0,
                            "reconstruction_ms": 2.0,
                            "service_ms": service,
                            "deadline_overrun_ms": 0.0,
                            "strict_deadline_success": True,
                            "committed_hr_pixels": committed,
                            "selected_output_checksum": "0" * 64,
                            "thermal_start_c": 40.0,
                            "thermal_end_c": 42.0,
                            "unselected_output_commit": False,
                        }
                    )
    return {
        "schema_version": 1,
        "experiment_id": "paper3_rk3576_fragmentation_pairs_v1",
        "status": "physical_complete_pending_offline_analysis",
        "records": records,
        "cross_mode_integrity": [
            {
                "repeat_index": repeat,
                "mask_id": mask_id,
                "mean_abs_float": 0.0005,
                "psnr_delta_db": 0.01,
            }
            for repeat in range(1, 6)
            for mask_id in (
                "p20_compact",
                "p20_fragmented",
                "p40_compact",
                "p40_fragmented",
                "p60_compact",
                "p60_fragmented",
            )
        ],
    }


def test_primary_gates_pass_for_preregistered_positive_control() -> None:
    analysis = analyze_capture(synthetic_complete_raw(), CONTRACT)

    assert analysis["integrity"]["all_passed"] is True
    assert analysis["all_primary_gates_passed"] is True
    assert analysis["disposition"] == "PRIMARY_GATES_PASS"
    assert all(analysis["causal_checks"].values())
    assert analysis["mode_interaction_percentage_points"] == pytest.approx(
        {"20": 9.0, "40": 9.0, "60": 9.0}
    )
    assert analysis["statistical_scope"]["p_value_claimed"] is False
    assert analysis["descriptive_sensitivity"] == {
        "row_span_positive_pairs": 15,
        "row_span_total_pairs": 15,
        "both_pair_orientations_present_at_each_density_in_row_span_mode": False,
        "thermal_start_min_c": 40.0,
        "thermal_start_max_c": 40.0,
        "thermal_end_min_c": 42.0,
        "thermal_end_max_c": 42.0,
        "maximum_paired_start_temperature_difference_c": 0.0,
        "timestamps_available": False,
        "physical_capture_sessions_observed": 1,
        "order_or_temperature_causality_claimed": False,
        "uncontrolled": [
            "sample order was frozen rather than randomized after capture",
            "temperature was observed rather than independently manipulated",
            "the records contain no per-sample wall-clock timestamps",
            "all samples come from one physical capture session",
        ],
    }
    assert "No p-value" in render_markdown(analysis)
    assert "Density" in render_latex(analysis)


def test_checked_in_capture_reports_frozen_repeat_order_and_temperature_descriptively() -> None:
    raw_path = (
        ROOT
        / "results/rk3576/paper3_fragmentation_pairs_v1/"
        "paper3_rk3576_fragmentation_pairs_v1_raw.json"
    )
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    analysis = analyze_capture(raw, CONTRACT)

    sensitivity = analysis["descriptive_sensitivity"]
    assert sensitivity["row_span_positive_pairs"] == 15
    assert sensitivity["row_span_total_pairs"] == 15
    assert sensitivity["both_pair_orientations_present_at_each_density_in_row_span_mode"] is True
    assert sensitivity["thermal_start_min_c"] == pytest.approx(45.307)
    assert sensitivity["thermal_start_max_c"] == pytest.approx(50.846)
    assert sensitivity["maximum_paired_start_temperature_difference_c"] == pytest.approx(1.846)
    assert sensitivity["timestamps_available"] is False
    assert sensitivity["order_or_temperature_causality_claimed"] is False

    summaries = {
        (row["density_percent"], row["dispatch_mode"]): row
        for row in analysis["pair_summaries"]
    }
    assert summaries[(20, "row_spans")]["fragmented_executed_second_count"] == 3
    assert summaries[(40, "row_spans")]["fragmented_executed_second_count"] == 4
    assert summaries[(60, "row_spans")]["fragmented_executed_second_count"] == 1
    assert summaries[(20, "row_spans")]["minimum_service_penalty_percent"] == pytest.approx(187.165, abs=0.001)
    assert summaries[(20, "row_spans")]["maximum_service_penalty_percent"] == pytest.approx(231.914, abs=0.001)


def test_exact_mode_timing_failure_blocks_geometry_causal_claim() -> None:
    raw = synthetic_complete_raw()
    for record in raw["records"]:
        if record["dispatch_mode"] == "exact" and record["geometry"] == "maximally_fragmented":
            record["service_ms"] = 11.0
    analysis = analyze_capture(raw, CONTRACT)

    assert analysis["integrity"]["all_passed"] is True
    assert analysis["causal_checks"]["exact_service_negative_control"] is False
    assert analysis["disposition"] == "NO_CAUSAL_TIMING_CLAIM_NEGATIVE_CONTROL_FAILED"


def test_integrity_failure_makes_capture_unusable() -> None:
    raw = synthetic_complete_raw()
    raw["records"][0]["unselected_output_commit"] = True
    analysis = analyze_capture(raw, CONTRACT)

    assert analysis["integrity"]["all_passed"] is False
    assert analysis["disposition"] == "UNUSABLE_INTEGRITY_FAILURE"


def test_missing_or_duplicate_sample_is_rejected_by_integrity_matrix() -> None:
    raw = synthetic_complete_raw()
    raw["records"][-1] = copy.deepcopy(raw["records"][0])
    with pytest.raises(ValueError, match="duplicate geometry"):
        analyze_capture(raw, CONTRACT)
