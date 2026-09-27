"""Analyze the preregistered Paper 3 fragmentation-pair raw capture."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.benchmark_rk3576_sparse import sha256_file  # noqa: E402


EXPERIMENT_ID = "paper3_rk3576_fragmentation_pairs_v1"
DEFAULT_CONTRACT = ROOT / "configs/paper3_rk3576_fragmentation_pairs_v1.yaml"


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping root: {path}")
    return value


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _fractional(fragmented: float, compact: float) -> float:
    if compact <= 0.0:
        raise ValueError("paired denominator must be positive")
    return (fragmented - compact) / compact


def analyze_capture(raw: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only the frozen integrity and causal decision rules."""

    if raw.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("raw capture has the wrong experiment_id")
    if raw.get("status") != "physical_complete_pending_offline_analysis":
        raise ValueError("raw capture is incomplete or failed")
    records = list(raw.get("records", []))
    expected_samples = int(contract["frozen_gates"]["integrity"]["observed_measured_samples_must_equal"])
    required_fields = set(contract["metrics_per_sample"]) | {
        "block_index",
        "unselected_output_commit",
    }
    checks: dict[str, bool] = {}
    checks["sample_count"] = len(records) == expected_samples
    checks["required_metrics_present"] = all(required_fields.issubset(record) for record in records)
    timing_fields = ("extraction_ms", "planning_ms", "npu_inference_ms", "reconstruction_ms", "service_ms")
    checks["positive_finite_timings"] = all(
        _finite_positive(record.get(field)) for record in records for field in timing_fields
    )
    checks["no_unselected_commit"] = all(record.get("unselected_output_commit") is False for record in records)
    observed_keys = [
        (int(record["repeat_index"]), str(record["mask_id"]), str(record["dispatch_mode"]))
        for record in records
        if {"repeat_index", "mask_id", "dispatch_mode"}.issubset(record)
    ]
    checks["complete_unique_matrix"] = len(observed_keys) == expected_samples and len(set(observed_keys)) == expected_samples

    by_cell: dict[tuple[int, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (int(record["repeat_index"]), str(record["pair_id"]), str(record["dispatch_mode"]))
        geometry = str(record["geometry"])
        if geometry in by_cell[key]:
            raise ValueError(f"duplicate geometry in paired cell: {key}")
        by_cell[key][geometry] = record
    checks["complete_pair_cells"] = len(by_cell) == 30 and all(
        set(cell) == {"compact", "maximally_fragmented"} for cell in by_cell.values()
    )

    pair_counts_match = True
    for cell in by_cell.values():
        if set(cell) != {"compact", "maximally_fragmented"}:
            pair_counts_match = False
            continue
        compact = cell["compact"]
        fragmented = cell["maximally_fragmented"]
        pair_counts_match &= int(compact["selected_tiles"]) == int(fragmented["selected_tiles"])
        pair_counts_match &= int(compact["committed_hr_pixels"]) == int(fragmented["committed_hr_pixels"])
        pair_counts_match &= int(compact["active_rows"]) == int(fragmented["active_rows"]) == 6
    checks["paired_counts_and_commits_match"] = bool(pair_counts_match)

    integrity_rows = list(raw.get("cross_mode_integrity", []))
    integrity_gate = contract["frozen_gates"]["integrity"]
    checks["cross_mode_matrix"] = len(integrity_rows) == 30
    max_mean_abs = max((float(row["mean_abs_float"]) for row in integrity_rows), default=math.inf)
    max_psnr_delta = max((float(row["psnr_delta_db"]) for row in integrity_rows), default=math.inf)
    checks["cross_mode_mean_abs"] = max_mean_abs <= float(integrity_gate["max_cross_mode_mean_abs_float"])
    checks["cross_mode_psnr_delta"] = max_psnr_delta <= float(integrity_gate["max_cross_mode_psnr_delta_db"])
    integrity_passed = all(checks.values())

    density_modes: dict[
        tuple[int, str],
        list[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
    ] = defaultdict(list)
    for (repeat, _, mode), cell in by_cell.items():
        if set(cell) != {"compact", "maximally_fragmented"}:
            continue
        compact = cell["compact"]
        fragmented = cell["maximally_fragmented"]
        density = int(compact["density_percent"])
        if density != int(fragmented["density_percent"]):
            raise ValueError("paired density mismatch")
        density_modes[(density, mode)].append((repeat, compact, fragmented))

    pair_summaries: list[dict[str, Any]] = []
    for density in (20, 40, 60):
        for mode in ("exact", "row_spans"):
            pairs = sorted(density_modes[(density, mode)], key=lambda item: item[0])
            if len(pairs) != 5:
                raise ValueError(f"expected five paired repeats for {density}/{mode}")
            service_deltas = [
                _fractional(float(fragmented["service_ms"]), float(compact["service_ms"]))
                for _, compact, fragmented in pairs
            ]
            repeat_details = [
                {
                    "repeat_index": repeat,
                    "service_penalty_percent": 100.0
                    * _fractional(float(fragmented["service_ms"]), float(compact["service_ms"])),
                    "fragmented_executed_second": int(fragmented["execution_order"])
                    > int(compact["execution_order"]),
                    "compact_thermal_start_c": float(compact["thermal_start_c"]),
                    "fragmented_thermal_start_c": float(fragmented["thermal_start_c"]),
                    "paired_start_temperature_difference_c": abs(
                        float(fragmented["thermal_start_c"]) - float(compact["thermal_start_c"])
                    ),
                }
                for repeat, compact, fragmented in pairs
            ]
            pair_summaries.append(
                {
                    "density_percent": density,
                    "dispatch_mode": mode,
                    "paired_repeats": 5,
                    "service_fractional_differences": service_deltas,
                    "paired_repeat_details": repeat_details,
                    "minimum_service_penalty_percent": 100.0 * min(service_deltas),
                    "median_service_penalty_percent": 100.0 * statistics.median(service_deltas),
                    "maximum_service_penalty_percent": 100.0 * max(service_deltas),
                    "fragmented_executed_second_count": sum(
                        detail["fragmented_executed_second"] for detail in repeat_details
                    ),
                    "maximum_paired_start_temperature_difference_c": max(
                        detail["paired_start_temperature_difference_c"] for detail in repeat_details
                    ),
                    "compact_executed_gmac": statistics.median(
                        float(left["executed_gmac"]) for _, left, _ in pairs
                    ),
                    "fragmented_executed_gmac": statistics.median(
                        float(right["executed_gmac"]) for _, _, right in pairs
                    ),
                    "compact_tensor_bytes_per_pixel": statistics.median(
                        float(left["tensor_bytes_per_committed_hr_pixel"]) for _, left, _ in pairs
                    ),
                    "fragmented_tensor_bytes_per_pixel": statistics.median(
                        float(right["tensor_bytes_per_committed_hr_pixel"]) for _, _, right in pairs
                    ),
                    "compact_npu_calls": statistics.median(
                        float(left["npu_calls"]) for _, left, _ in pairs
                    ),
                    "fragmented_npu_calls": statistics.median(
                        float(right["npu_calls"]) for _, _, right in pairs
                    ),
                }
            )
    summary_index = {(row["density_percent"], row["dispatch_mode"]): row for row in pair_summaries}

    row_span_structural = all(
        summary_index[(density, "row_spans")]["fragmented_executed_gmac"]
        > summary_index[(density, "row_spans")]["compact_executed_gmac"]
        and summary_index[(density, "row_spans")]["fragmented_tensor_bytes_per_pixel"]
        > summary_index[(density, "row_spans")]["compact_tensor_bytes_per_pixel"]
        for density in (20, 40, 60)
    )
    exact_structural_equal = all(
        math.isclose(
            summary_index[(density, "exact")]["fragmented_executed_gmac"],
            summary_index[(density, "exact")]["compact_executed_gmac"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            summary_index[(density, "exact")]["fragmented_tensor_bytes_per_pixel"],
            summary_index[(density, "exact")]["compact_tensor_bytes_per_pixel"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for density in (20, 40, 60)
    )
    row_penalties = [summary_index[(density, "row_spans")]["median_service_penalty_percent"] for density in (20, 40, 60)]
    exact_penalties = [summary_index[(density, "exact")]["median_service_penalty_percent"] for density in (20, 40, 60)]
    interactions = [row - exact for row, exact in zip(row_penalties, exact_penalties)]
    gates = contract["frozen_gates"]
    causal_checks = {
        "row_span_structural_padding_all_densities": row_span_structural,
        "exact_structural_negative_control": exact_structural_equal,
        "row_span_positive_service_all_densities": sum(value > 0.0 for value in row_penalties)
        == int(gates["physical_service"]["row_span_positive_paired_service_delta_at_density_count"]),
        "row_span_service_penalty_ge_5_percent_at_two_densities": sum(value >= 5.0 for value in row_penalties)
        >= int(gates["physical_service"]["row_span_minimum_density_pairs_with_median_penalty_ge_5_percent"]),
        "exact_service_negative_control": all(
            abs(value) <= float(gates["exact_negative_control"]["max_absolute_median_paired_service_delta_percent_each_density"])
            for value in exact_penalties
        ),
    }
    interaction_checks = {
        "positive_all_densities": sum(value > 0.0 for value in interactions)
        == int(gates["mode_interaction"]["positive_interaction_at_density_count"]),
        "ge_5_percentage_points_at_two_densities": sum(
            value >= float(gates["mode_interaction"]["minimum_interaction_percentage_points"])
            for value in interactions
        )
        >= int(gates["mode_interaction"]["minimum_density_pairs_with_interaction_ge_percentage_points"]),
    }
    primary_passed = integrity_passed and all(causal_checks.values())
    if not integrity_passed:
        disposition = "UNUSABLE_INTEGRITY_FAILURE"
        claim = "No geometry-causal result is authorized."
    elif not causal_checks["exact_service_negative_control"]:
        disposition = "NO_CAUSAL_TIMING_CLAIM_NEGATIVE_CONTROL_FAILED"
        claim = "Only deterministic structural accounting may be reported."
    elif row_span_structural and not (
        causal_checks["row_span_positive_service_all_densities"]
        and causal_checks["row_span_service_penalty_ge_5_percent_at_two_densities"]
    ):
        disposition = "ADVERSE_PHYSICAL_SERVICE_RESULT"
        claim = "Report padding and traffic causality; do not claim a physical service penalty."
    elif primary_passed:
        disposition = "PRIMARY_GATES_PASS"
        claim = str(contract["claim_boundary"]["causal_claim_if_all_primary_gates_pass"])
    else:
        disposition = "PRIMARY_GATES_FAIL"
        claim = "Report the adverse result within the preregistered one-device scope."

    row_span_summaries = [summary_index[(density, "row_spans")] for density in (20, 40, 60)]
    thermal_start_values = [float(record["thermal_start_c"]) for record in records]
    thermal_end_values = [float(record["thermal_end_c"]) for record in records]
    descriptive_sensitivity = {
        "row_span_positive_pairs": sum(
            detail["service_penalty_percent"] > 0.0
            for row in row_span_summaries
            for detail in row["paired_repeat_details"]
        ),
        "row_span_total_pairs": 15,
        "both_pair_orientations_present_at_each_density_in_row_span_mode": all(
            0 < row["fragmented_executed_second_count"] < row["paired_repeats"]
            for row in row_span_summaries
        ),
        "thermal_start_min_c": min(thermal_start_values),
        "thermal_start_max_c": max(thermal_start_values),
        "thermal_end_min_c": min(thermal_end_values),
        "thermal_end_max_c": max(thermal_end_values),
        "maximum_paired_start_temperature_difference_c": max(
            row["maximum_paired_start_temperature_difference_c"] for row in pair_summaries
        ),
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

    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "analysis_complete",
        "raw_capture_sha256": raw.get("self_sha256"),
        "integrity": {
            "all_passed": integrity_passed,
            "checks": checks,
            "max_cross_mode_mean_abs_float": max_mean_abs,
            "max_cross_mode_psnr_delta_db": max_psnr_delta,
        },
        "pair_summaries": pair_summaries,
        "descriptive_sensitivity": descriptive_sensitivity,
        "mode_interaction_percentage_points": {
            str(density): value for density, value in zip((20, 40, 60), interactions)
        },
        "causal_checks": causal_checks,
        "secondary_interaction_checks": interaction_checks,
        "all_primary_gates_passed": primary_passed,
        "disposition": disposition,
        "authorized_claim": claim,
        "statistical_scope": {
            "paired_statistic": contract["analysis"]["paired_statistic"],
            "confidence_interval_claimed": False,
            "p_value_claimed": False,
            "regression_used": False,
            "deadline_descriptive_only": True,
        },
        "nonclaims": {
            key: value
            for key, value in contract["claim_boundary"].items()
            if key.endswith("_claimed")
        },
    }


def render_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Paper 3 RK3576 Fragmentation Pairs V1",
        "",
        f"Disposition: **{analysis['disposition']}**",
        "",
        "| Density | Mode | Compact GMAC | Fragmented GMAC | Compact B/pix | Fragmented B/pix | Service penalty min / median / max | Fragmented second | Max paired start-temperature gap |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["pair_summaries"]:
        lines.append(
            f"| {row['density_percent']}% | {row['dispatch_mode']} | "
            f"{row['compact_executed_gmac']:.6f} | {row['fragmented_executed_gmac']:.6f} | "
            f"{row['compact_tensor_bytes_per_pixel']:.4f} | {row['fragmented_tensor_bytes_per_pixel']:.4f} | "
            f"{row['minimum_service_penalty_percent']:.3f}% / "
            f"{row['median_service_penalty_percent']:.3f}% / "
            f"{row['maximum_service_penalty_percent']:.3f}% | "
            f"{row['fragmented_executed_second_count']}/5 | "
            f"{row['maximum_paired_start_temperature_difference_c']:.3f} C |"
        )
    sensitivity = analysis["descriptive_sensitivity"]
    order_sentence = (
        "Both compact-first and fragmented-first orders occur at every density. "
        if sensitivity["both_pair_orientations_present_at_each_density_in_row_span_mode"]
        else "The records do not contain both pair orientations at every density. "
    )
    lines.extend(
        [
            "",
            "## Paired repeat distributions",
            "",
            "| Density | Mode | Repeat 1 | Repeat 2 | Repeat 3 | Repeat 4 | Repeat 5 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["pair_summaries"]:
        deltas = " | ".join(
            f"{detail['service_penalty_percent']:.3f}%"
            for detail in row["paired_repeat_details"]
        )
        lines.append(f"| {row['density_percent']}% | {row['dispatch_mode']} | {deltas} |")
    lines.extend(
        [
            "",
            "## Descriptive order and temperature audit",
            "",
            f"All {sensitivity['row_span_positive_pairs']}/{sensitivity['row_span_total_pairs']} "
            "row-span paired service deltas are positive. "
            + order_sentence
            + "Start temperature spans "
            f"{sensitivity['thermal_start_min_c']:.3f}--{sensitivity['thermal_start_max_c']:.3f} C; "
            "the maximum within-pair start-temperature difference is "
            f"{sensitivity['maximum_paired_start_temperature_difference_c']:.3f} C.",
            "",
            "This audit is descriptive only: the capture has no per-sample timestamps, "
            "temperature was not independently manipulated, order was frozen, and all "
            "records come from one physical session.",
            "",
            f"Authorized interpretation: {analysis['authorized_claim']}",
            "",
            "No p-value, confidence interval, regression, selector-generalization, energy, "
            "720p-real-time, platform-superiority, or cross-platform bit-identity claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def render_latex(analysis: Mapping[str, Any]) -> str:
    lines = [
        "% Generated only from paper3_rk3576_fragmentation_pairs_v1 raw capture.",
        r"\begin{tabular}{rlrrrrrrrr}",
        r"Density & Mode & Median & R1 & R2 & R3 & R4 & R5 & Frag. second & Max $\Delta T_0$ (C) \\",
        r"\hline",
    ]
    index = {(row["density_percent"], row["dispatch_mode"]): row for row in analysis["pair_summaries"]}
    for density in (20, 40, 60):
        for mode, label in (("exact", "Exact"), ("row_spans", "Row spans")):
            row = index[(density, mode)]
            deltas = " & ".join(
                f"{detail['service_penalty_percent']:.3f}"
                for detail in row["paired_repeat_details"]
            )
            lines.append(
                f"{density} & {label} & {row['median_service_penalty_percent']:.3f} & "
                f"{deltas} & {row['fragmented_executed_second_count']}/5 & "
                f"{row['maximum_paired_start_temperature_difference_c']:.3f} \\\\"
            )
    lines.extend([r"\end{tabular}", ""])
    return "\n".join(lines)


def _write_exclusive(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--latex-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = (args.json_output.resolve(), args.markdown_output.resolve(), args.latex_output.resolve())
    try:
        if any(path.exists() for path in outputs):
            raise FileExistsError("one or more analysis outputs already exist")
        raw_path = args.raw.resolve()
        raw = _load_mapping(raw_path)
        raw["self_sha256"] = sha256_file(raw_path)
        contract = _load_mapping(args.contract.resolve())
        analysis = analyze_capture(raw, contract)
        _write_exclusive(outputs[0], json.dumps(analysis, indent=2, sort_keys=True) + "\n")
        _write_exclusive(outputs[1], render_markdown(analysis))
        _write_exclusive(outputs[2], render_latex(analysis))
    except (FileExistsError, FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"PAPER3_FRAGMENTATION_ANALYSIS_BLOCKED: {error}", file=sys.stderr)
        return 1
    print(f"PAPER3_FRAGMENTATION_ANALYSIS_COMPLETE disposition={analysis['disposition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
