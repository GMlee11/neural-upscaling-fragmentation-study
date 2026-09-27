"""Build Paper 3's self-contained RKNN fragmentation summary.

The report uses only Paper 3 physical RKNN measurements.  It deliberately
avoids FPGA counters, exact-closure projections, companion-paper results, and
cross-platform performance comparisons.  Every work metric is derived from
the selected mask and the static graph shapes actually dispatched by RKNN.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/tables/codesign_matched_masks_rk3576_v1.json"
DEFAULT_JSON = ROOT / "results/tables/paper3_npu_fragmentation_summary_v1.json"
DEFAULT_MARKDOWN = ROOT / "results/tables/paper3_npu_fragmentation_summary_v1.md"
MASK_COLUMNS = 10
MASK_ROWS = 6
FRAME_DEADLINE_MS = 1000.0 / 30.0


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def decode_mask(mask_hex: str) -> dict[str, int]:
    """Decode selected tiles, active rows, and horizontal runs."""

    mask = int(mask_hex, 16)
    if mask < 0 or mask >= (1 << (MASK_COLUMNS * MASK_ROWS)):
        raise ValueError("mask is outside the frozen 10x6 grid")
    selected = 0
    active_rows = 0
    horizontal_runs = 0
    for row_index in range(MASK_ROWS):
        row_bits = [
            bool(mask & (1 << (row_index * MASK_COLUMNS + column)))
            for column in range(MASK_COLUMNS)
        ]
        selected += sum(row_bits)
        active_rows += int(any(row_bits))
        horizontal_runs += sum(
            bit and (column == 0 or not row_bits[column - 1])
            for column, bit in enumerate(row_bits)
        )
    return {
        "selected_tiles": selected,
        "active_rows": active_rows,
        "horizontal_runs": horizontal_runs,
    }


def _index_rows(payload: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    rows = payload.get("pattern_rows")
    if not isinstance(rows, list):
        raise ValueError("physical report has no pattern_rows list")
    index: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("pattern row must be an object")
        key = (int(row["pattern_id"]), str(row["dispatch_mode"]))
        if key in index:
            raise ValueError(f"duplicate physical row: {key}")
        index[key] = row
    expected = {
        (pattern_id, mode)
        for pattern_id in range(1, 16)
        for mode in ("exact", "row_spans")
    }
    if set(index) != expected:
        raise ValueError("physical report must contain 15 masks in both modes")
    return index


def analyze(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a Paper-3-owned, NPU-only fragmentation report."""

    if not bool(payload.get("all_gates_passed")):
        raise ValueError("physical RKNN report did not pass all gates")
    index = _index_rows(payload)
    pairs: list[dict[str, Any]] = []
    for pattern_id in range(1, 16):
        exact = index[(pattern_id, "exact")]
        spans = index[(pattern_id, "row_spans")]
        if str(exact["mask_hex"]) != str(spans["mask_hex"]):
            raise ValueError(f"mask identity differs for pattern {pattern_id}")
        geometry = decode_mask(str(exact["mask_hex"]))
        selected_pixels = int(exact["selected_core_pixels"])
        if selected_pixels != int(spans["selected_core_pixels"]):
            raise ValueError(f"selected work differs for pattern {pattern_id}")
        if geometry["selected_tiles"] != int(exact["selected_tiles"]):
            raise ValueError(f"decoded mask differs for pattern {pattern_id}")
        exact_dispatched = int(exact["dispatched_core_pixels"])
        span_dispatched = int(spans["dispatched_core_pixels"])
        exact_bytes = int(exact["npu_input_bytes"]) + int(exact["npu_output_bytes"])
        span_bytes = int(spans["npu_input_bytes"]) + int(spans["npu_output_bytes"])
        exact_service = float(exact["timing"]["service_ms"]["average_ms"])
        span_service = float(spans["timing"]["service_ms"]["average_ms"])
        pairs.append(
            {
                "pattern_id": pattern_id,
                "sequence_id": str(exact["sequence_id"]),
                "mask_hex": str(exact["mask_hex"]),
                **geometry,
                "selected_core_pixels": selected_pixels,
                "exact_calls": int(exact["npu_calls"]),
                "row_span_calls": int(spans["npu_calls"]),
                "exact_dispatched_core_pixels": exact_dispatched,
                "row_span_dispatched_core_pixels": span_dispatched,
                "row_span_core_padding_percent": 100.0
                * (span_dispatched - selected_pixels)
                / selected_pixels,
                "exact_tensor_bytes": exact_bytes,
                "row_span_tensor_bytes": span_bytes,
                "exact_service_ms": exact_service,
                "row_span_service_ms": span_service,
                "row_span_service_change_percent": 100.0
                * (span_service / exact_service - 1.0),
                "exact_mean_meets_33p33ms": exact_service <= FRAME_DEADLINE_MS,
                "row_span_mean_meets_33p33ms": span_service <= FRAME_DEADLINE_MS,
            }
        )

    exact_calls = sum(row["exact_calls"] for row in pairs)
    span_calls = sum(row["row_span_calls"] for row in pairs)
    exact_dispatched = sum(row["exact_dispatched_core_pixels"] for row in pairs)
    span_dispatched = sum(row["row_span_dispatched_core_pixels"] for row in pairs)
    exact_bytes = sum(row["exact_tensor_bytes"] for row in pairs)
    span_bytes = sum(row["row_span_tensor_bytes"] for row in pairs)
    exact_service = statistics.fmean(row["exact_service_ms"] for row in pairs)
    span_service = statistics.fmean(row["row_span_service_ms"] for row in pairs)
    return {
        "schema_version": 1,
        "artifact": "paper3_npu_fragmentation_summary_v1",
        "scope": "paper3_physical_rknn_measurements_only",
        "claim_boundary": {
            "cross_platform_performance_claimed": False,
            "external_accelerator_denominator_used": False,
            "independent_submission_artifact": True,
            "work_accounting": "selected_masks_and_dispatched_rknn_tensor_shapes",
        },
        "patterns": len(pairs),
        "pairs": pairs,
        "aggregate": {
            "average_exact_calls": exact_calls / len(pairs),
            "average_row_span_calls": span_calls / len(pairs),
            "row_span_call_reduction_percent": 100.0
            * (1.0 - span_calls / exact_calls),
            "row_span_dispatched_core_increase_percent": 100.0
            * (span_dispatched / exact_dispatched - 1.0),
            "row_span_tensor_byte_increase_percent": 100.0
            * (span_bytes / exact_bytes - 1.0),
            "average_exact_service_ms": exact_service,
            "average_row_span_service_ms": span_service,
            "row_span_service_reduction_percent": 100.0
            * (1.0 - span_service / exact_service),
            "exact_mean_deadline_success_percent": 100.0
            * sum(row["exact_mean_meets_33p33ms"] for row in pairs)
            / len(pairs),
            "row_span_mean_deadline_success_percent": 100.0
            * sum(row["row_span_mean_meets_33p33ms"] for row in pairs)
            / len(pairs),
        },
        "interpretation": (
            "Across the frozen 15-mask RK3576 replay, row-span dispatch reduces "
            "static launches and mean service time while increasing dispatched core "
            "area and tensor traffic. These are within-backend organization effects."
        ),
    }


def markdown(report: Mapping[str, Any]) -> str:
    """Render a compact reviewer-facing summary."""

    aggregate = report["aggregate"]
    return "\n".join(
        [
            "# Paper 3 Self-Contained RKNN Fragmentation Summary",
            "",
            "This report uses only Paper 3 physical RKNN measurements and the static graph shapes actually dispatched.",
            "It imports no FPGA counters, companion-paper results, or external accelerator denominator.",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Masks | {report['patterns']} |",
            f"| Average exact launches | {aggregate['average_exact_calls']:.3f} |",
            f"| Average row-span launches | {aggregate['average_row_span_calls']:.3f} |",
            f"| Row-span launch reduction | {aggregate['row_span_call_reduction_percent']:.2f}% |",
            f"| Row-span dispatched-core increase | {aggregate['row_span_dispatched_core_increase_percent']:.2f}% |",
            f"| Row-span tensor-byte increase | {aggregate['row_span_tensor_byte_increase_percent']:.2f}% |",
            f"| Row-span mean-service reduction | {aggregate['row_span_service_reduction_percent']:.2f}% |",
            f"| Exact mean-deadline success | {aggregate['exact_mean_deadline_success_percent']:.1f}% |",
            f"| Row-span mean-deadline success | {aggregate['row_span_mean_deadline_success_percent']:.1f}% |",
            "",
            str(report["interpretation"]),
            "",
        ]
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze(load_json(args.input.resolve()))
    write_json(args.json_output.resolve(), report)
    args.markdown_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.resolve().write_text(
        markdown(report), encoding="utf-8", newline="\n"
    )
    print(
        "PAPER3_NPU_FRAGMENTATION_SUMMARY_PASS "
        f"patterns={report['patterns']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
