"""Offline replay of audited dispatch rows; does not revalidate omitted tensors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
INPUT = "results/manifests/paper3_dispatch_public_v1.json"
POLICIES = ("always_exact", "always_span", "geometry", "calibrated")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def features(rows, mode, tensor):
    """Independent geometry accounting, not imported from experiment code."""
    calls = pixels = moved = selected = 0
    for y, row in enumerate(rows):
        columns = [i for i, bit in enumerate(row) if bit == "1"]
        selected += len(columns)
        if not columns:
            continue
        widths = [64] * len(columns) if mode == "exact" else [next(
            w for w in (64, 128, 256, 512, 576, 640) if w >= (columns[-1] - columns[0] + 1) * 64)]
        for width in widths:
            key = f"core_{width:04d}x0060"
            if width == 640 and tensor["boundary_mode"] == "zero":
                key = "full_width_top" if y == 0 else "full_width_bottom" if y == 5 else "full_width_middle"
            inp, out = tensor["shape_elements"][key]
            moved += inp * tensor["input_itemsize"] + out * tensor["output_itemsize"]
            calls += 1
            pixels += width * 60
    return [1., float(calls), pixels / 1e6, moved / 2**20, selected * 64 * 60 * 4 / 1e6]


def summarize(payload):
    require(payload["schema_version"] == 1 and payload["artifact"] == "paper3_dispatch_public_v1", "schema")
    require(payload["excluded_campaign"]["performance_summary_withheld"] is True, "failed campaign pooled")
    plan = payload["plan"]
    result = []
    require([(c["device"], c["graph"]) for c in payload["cells"]] == [
        ("rk3576", "width24"), ("rk3576", "fp16"), ("rk3566", "width24"), ("rk3566", "fp16")], "cell coverage")
    for cell in payload["cells"]:
        pins, samples, audited = cell["pins"], cell["samples"], cell["full_raw_audit"]
        replication = cell["device"] == "rk3566" or cell["graph"] == "fp16"
        require(audited["requests"] == len(samples) == 1280, "complete sample count")
        require(audited["previously_opened_geometry_replication"] == replication, "replication label")
        require([s["journal"]["sha256"] for s in cell["sessions"]] == audited["journals_sha256"], "journal provenance")
        pairs, choices, identities, overheads = {}, {p: {} for p in POLICIES}, {}, {p: [] for p in POLICIES}
        coefficients = pins["coefficients"]
        require(len(coefficients) == 5 and all(math.isfinite(c) and c >= 0 for c in coefficients), "coefficients")
        for i, sample in enumerate(samples):
            request = sample["request"]
            expected = dict(plan["requests"][i])
            if replication:
                expected["split"] = "previously_opened_replication"
            require(request == expected, "schedule/mask mismatch")
            require(sample["session"] == i // 640 and sample["index"] == i % 640, "sample ordering")
            policy, rows = request["policy"], request["rows"]
            exact, span = [features(rows, m, pins["tensor_contract"]) for m in ("exact", "row_spans")]
            if policy == "always_exact":
                selected_mode = "exact"
            elif policy == "always_span":
                selected_mode = "row_spans"
            elif policy == "geometry":
                count = sum(r.count("1") for r in rows)
                selected_mode = "row_spans" if span[1] <= count / 2 and round(span[2] * 1e6 / 3840) <= count * 1.25 else "exact"
            else:
                selected_mode = "row_spans" if sum(c*x for c,x in zip(coefficients, span)) < sum(c*x for c,x in zip(coefficients, exact)) else "exact"
            require(sample["chosen_mode"] == selected_mode, "policy decision mismatch")
            elapsed = (sample["service_end_ns"] - sample["service_start_ns"]) / 1e6
            require(math.isfinite(elapsed) and elapsed > 0 and elapsed == sample["service_ms"], "service timing")
            require(all(math.isfinite(t) and t < 80 for t in sample["temperatures_c"]), "temperature gate")
            require(sample["invariants_sha256"] == [digest(pins["qualified_control_invariants"])] * 2, "control identity")
            key = (request["block"], request["mask_id"])
            pair = pairs.setdefault(key, {})
            require(policy not in pair, "duplicate policy")
            pair[policy] = elapsed
            identity_key = (request["mask_id"], selected_mode)
            previous = identities.setdefault(identity_key, sample["raw_identity"])
            require(previous == sample["raw_identity"], "repeated-output identity mismatch")
            choices[policy][request["mask_id"]] = selected_mode
            overheads[policy].append(sample["policy_ms"])
        require(len(pairs) == 320 and all(set(p) == set(POLICIES) for p in pairs.values()), "pair completeness")
        means = {p: mean(v[p] for v in pairs.values()) for p in POLICIES}
        fixed = min(POLICIES[:2], key=means.get)
        summaries = {}
        for policy in POLICIES:
            deltas = [v[policy] - v[fixed] for v in pairs.values()]
            summaries[policy] = {"mean_service_ms": means[policy], "delta_best_fixed_ms": mean(deltas),
                "regressions_vs_best_fixed": sum(d > 0 for d in deltas), "maximum_regression_ms": max(0., max(deltas)),
                "deadline_successes_33p33ms": sum(v[policy] <= 1000 / 30 for v in pairs.values()),
                "session_means_ms": [mean(v[policy] for (block, _), v in pairs.items() if block // 4 == s) for s in (0, 1)]}
        require(summaries == audited["policies"], "raw-audit/per-request summary mismatch")
        result.append({"device": cell["device"], "graph": cell["graph"], "requests": 1280,
            "previously_opened_geometry_replication": replication, "best_fixed_posthoc": fixed,
            "calibrated_reduction_percent": 100 * (means[fixed] - means["calibrated"]) / means[fixed],
            "policies": summaries, "policy_mean_overhead_ms": {p: mean(v) for p,v in overheads.items()},
            "mode_choices_over_40_masks": {p: {m: sum(v == m for v in choices[p].values()) for m in ("exact", "row_spans")} for p in POLICIES},
            "maximum_cross_mode_mae": audited["maximum_cross_mode_mae"],
            "maximum_reference_mae": audited["maximum_reference_mae"],
            "maximum_psnr_delta_db": audited["maximum_psnr_delta_db"]})
    return {"schema_version": 1, "status": "COMPACT_TIMING_AND_POLICY_REPLAY_PASS", "cells": result,
            "scope": "Four named graph/device cells; per-target calibration; numerical/raw-control audit outcomes are archived, not re-executed by compact replay.",
            "regression_interpretation": "Paired timing differences include run variability and policy overhead, not wrong-choice counts.",
            "failed_fp16_v1_excluded": True}


def latex(summary):
    rows = [r"\begin{tabular}{llrrrrr}", r"\hline",
        r"Device & Graph & Exact & Row-span & Geometry & Calibrated & Reduction \\", r"\hline"]
    for c in summary["cells"]:
        means = [c["policies"][p]["mean_service_ms"] for p in POLICIES]
        graph = "Width-24 INT8" if c["graph"] == "width24" else "Seven-conv. FP16"
        rows.append(f"{c['device'].upper()} & {graph} & " + " & ".join(f"{v:.2f}" for v in means) + f" & {c['calibrated_reduction_percent']:.2f}\\% " + r"\\")
    rows += [r"\hline", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / INPUT)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--latex-output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(json.loads(args.input.read_bytes()))
    for path, text in ((args.json_output, json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n"), (args.latex_output, latex(summary))):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    print("PAPER3_DISPATCH_COMPACT_REPLAY_PASS")


if __name__ == "__main__":
    main()
