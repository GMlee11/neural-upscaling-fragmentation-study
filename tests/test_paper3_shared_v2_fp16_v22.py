"""Frozen offline gates for the Paper 3 V22 second-graph cell."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
import pytest


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v22_protocol_binds_preexisting_candidate_before_target_output() -> None:
    protocol = yaml.safe_load((ROOT / "configs/paper3_shared_v2_fp16_fragmentation_v22.yaml").read_text(encoding="utf-8"))
    candidate = protocol["candidate_selection"]
    assert protocol["status"] == "FROZEN_OFFLINE_PREPARATION_AUTHORIZED_NO_TARGET_OUTPUT_OPENED"
    assert candidate["target_results_used_for_selection"] is False
    assert candidate["alternate_candidate_after_target_output"] == "forbidden"
    checkpoint = ROOT / candidate["checkpoint_path"]
    training_config = ROOT / candidate["training_config_path"]
    if checkpoint.is_file() and training_config.is_file():
        assert sha256(checkpoint) == candidate["checkpoint_sha256"]
        assert sha256(training_config) == candidate["training_config_sha256"]
    else:
        assert candidate["checkpoint_sha256"] == "f78730284e8199ca93348a5486f059eb7b2bd6e6b6389c96f135ff0daec7ef5a"
        assert len(candidate["training_config_sha256"]) == 64
    assert protocol["graph"] == {
        "architecture": "shared_progressive_espcn_v2", "exit_mode": "direct", "stage": "C",
        "feature_channels": 6, "blocks_per_stage": [1, 1, 3], "convolutions": 7,
        "residual_feature_additions": 4, "lr_receptive_field_halo_pixels": 7,
        "precision": "fp16", "target": "rk3576", "compiler": "rknn-toolkit2-2.3.2",
    }


def test_v22_onnx_and_offline_reference_are_complete() -> None:
    model_root = ROOT / "results/models/paper3_shared_v2_fp16_sparse_v22"
    if not model_root.is_dir():
        pytest.skip("minimal reviewer package excludes ONNX graphs and FP32 tensor payload")
    catalog = json.loads((model_root / "catalog.json").read_text(encoding="utf-8"))
    graphs = catalog["base_graphs"] + catalog["boundary_graphs"]
    assert len(catalog["base_graphs"]) == 8
    assert [row["variant"] for row in catalog["boundary_graphs"]] == [
        "full_width_top", "full_width_middle", "full_width_bottom"]
    assert len(graphs) == 11
    for row in graphs:
        assert sha256(ROOT / row["path"]) == row["sha256"]
        assert row["operators"] == {"Conv": 7, "Relu": 6, "Add": 4}
        assert row["maximum_pytorch_onnx_error"] <= 2e-6

    reference_path = ROOT / "results/manifests/paper3_shared_v2_fp16_offline_reference_v22.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    payload = ROOT / reference["payload"]["path"]
    assert reference["status"] == "OFFLINE_FP32_REFERENCE_PASS_RKNN_QUALIFICATION_PENDING"
    assert len(reference["records"]) == 6
    assert sha256(payload) == reference["payload"]["sha256"]
    for row in reference["records"]:
        assert max(row["selected_output_mae"].values()) <= .001
        assert row["cross_mode_psnr_delta_db"] <= .02
        assert row["maximum_unselected_output_change"] == 0


def test_v22_schedule_is_complete_paired_and_counterbalanced() -> None:
    schedule = json.loads((ROOT / "results/manifests/paper3_shared_v2_fp16_schedule_v22.json").read_text(encoding="utf-8"))
    assert len(schedule) == 120
    for offset in range(0, len(schedule), 2):
        left, right = schedule[offset:offset + 2]
        assert (left["block_index"], left["pair_id"], left["dispatch_mode"]) == (
            right["block_index"], right["pair_id"], right["dispatch_mode"])
        assert {left["geometry"], right["geometry"]} == {"compact", "maximally_fragmented"}
    for pair_id in ("density_20", "density_40", "density_60"):
        for mode in ("exact", "row_spans"):
            first = [schedule[index]["geometry"] for index in range(0, 120, 2)
                if schedule[index]["pair_id"] == pair_id and schedule[index]["dispatch_mode"] == mode]
            assert first.count("compact") == first.count("maximally_fragmented") == 5


def test_v22_rk3576_qualification_passes_every_frozen_numerical_gate() -> None:
    path = ROOT / "results/manifests/paper3_shared_v2_fp16_rk3576_qualification_v22.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "V22_RK3576_NUMERICAL_QUALIFICATION_PASS"
    assert result["graph_count"] == 11
    assert result["mask_mode_count"] == len(result["records"]) == 12
    assert result["physical_execution_measured"] is False
    assert max(row["selected_output_mae"] for row in result["records"]) <= .001
    assert max(row["cross_mode_psnr_delta_db"] for row in result["records"]) <= .02
    assert max(row["maximum_unselected_output_change"] for row in result["records"]) == 0
    assert len({row["mask_id"] for row in result["records"]}) == 6


def test_v22_capture_and_analysis_preserve_all_samples_and_replication_result() -> None:
    capture = json.loads((ROOT / "results/manifests/paper3_shared_v2_fp16_rk3576_capture_v22.json").read_text(encoding="utf-8"))
    analysis = json.loads((ROOT / "results/manifests/paper3_shared_v2_fp16_rk3576_analysis_v22.json").read_text(encoding="utf-8"))
    assert capture["status"] == "V22_RK3576_PHYSICAL_CAPTURE_COMPLETE"
    assert capture["sample_count"] == len(capture["samples"]) == 120
    assert capture["warmups_per_graph"] == 2
    assert len(capture["blocks"]) == 10
    assert max(row["service_ms"] for row in capture["samples"]) > 2300
    assert analysis["status"] == "V22_RK3576_OFFLINE_ANALYSIS_COMPLETE"
    assert analysis["paired_observation_count"] == 60
    assert analysis["replication"] == {
        "exact_control_within_5_percent_all_densities": True,
        "row_span_penalty_at_least_5_percent_density_count": 3,
        "row_span_penalty_positive_all_densities": True,
    }
    expected = {
        (20, "exact"): -0.125,
        (20, "row_spans"): 156.857,
        (40, "exact"): 0.906,
        (40, "row_spans"): 73.339,
        (60, "exact"): 0.882,
        (60, "row_spans"): 12.252,
    }
    for row in analysis["summaries"]:
        assert round(row["median_service_penalty_percent"], 3) == expected[(row["density_percent"], row["dispatch_mode"])]
