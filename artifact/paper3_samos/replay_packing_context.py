"""Replay the earlier RK3566 packing screen's ranks, not omitted raw pixels.

The context root preserves original relative paths (output/, results/, docs/).
No board, model, tensor, fitting, or current RK3576 confirmation is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from statistics import mean

SOURCE_SHA = "7fc28c4228fa3eb84d28f6430f5ba5f37ae74f166561b641b5c8e8b47eeb7fed"
REPORT_SHA = "2f26e4719fb78bd2bbf769ff69878bc3380dc76b8e77c6f3a819540699d5fad0"
SHAPES = ((64, 60), (128, 60), (256, 60), (512, 60), (576, 60), (640, 60), (640, 120), (640, 180))
POLICIES = ("exact", "row_spans", "grouped")+tuple(f"pack_{w}x{h}" for w, h in SHAPES)
FAMILIES = ("compact", "scattered", "clustered", "mixed")
SOURCE_FILES = {"E002_headroom_plan.md", "headroom_rk3566.py", "qualify_rk3566.py",
                "packing/__init__.py", "packing/core.py", "packing/service.py", "packing/jobs_service.py"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or ":" in relative or "\\" in relative:
        raise ValueError("unsafe context path")
    root = Path(root).resolve()
    candidate = root.joinpath(*rel.parts)
    if not candidate.resolve().is_relative_to(root) or candidate.is_symlink():
        raise ValueError("context path escapes root")
    return candidate


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid recorded nonnegative number")
    return value


def analyze_journal(raw, audit, cell):
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if (cell not in ("width24", "fp16") or not events or events[0].get("state") != "running" or
            events[0].get("cell") != cell or events[0].get("purpose") != "headroom_development_only" or
            events[0].get("source_manifest_sha") != SOURCE_SHA or
            events[-1].get("state") != "complete" or events[-1].get("observations") != 88 or
            events[-1].get("confirmation") is not False):
        raise ValueError("not the completed bounded packing screen")
    if (audit.get("terminal") != "complete" or audit.get("observations") != 88 or
            audit.get("engineering_only") is not True or audit.get("journal_sha256") != sha(raw)):
        raise ValueError("private audit is incomplete or journal hash differs")
    masks = events[0]["masks"]
    expected_ids = [f"dev_{family}_{count}_0" for family in FAMILIES for count in (12, 36)]
    if [m["mask_id"] for m in masks] != expected_ids:
        raise ValueError("eight development-mask identities/order differ")
    for mask, (family, count) in zip(masks, ((f, n) for f in FAMILIES for n in (12, 36)), strict=True):
        rows = mask["rows"]
        if (len(rows) != 6 or any(not isinstance(row, str) or len(row) != 10 or set(row)-{"0", "1"} for row in rows)
                or sum(row.count("1") for row in rows) != count or mask.get("selected_tiles") != count or
                mask.get("family") != family or mask.get("split") != "development"):
            raise ValueError("changed development-mask geometry")
    index, baseline, temperatures, unique_arrays = 1, None, [], {}
    def take(state):
        nonlocal index
        if index >= len(events) or events[index].get("state") != state:
            raise ValueError("incomplete or reordered engineering journal")
        event = events[index]; index += 1
        return event
    def control(stage):
        nonlocal baseline
        event = take("control")
        snap = event["snapshot"]
        if event["stage"] != stage or snap["invariants"].get("other_npu_owners") != []:
            raise ValueError("recorded control stage/owner differs")
        baseline = snap["invariants"] if baseline is None else baseline
        if snap["invariants"] != baseline or number(snap["temperature_c"]) >= 80:
            raise ValueError("recorded control invariant or temperature differs")
        compatible = snap["target_snapshot"]["invariants"]["compatible"]
        if "rockchip,rk3566" not in compatible:
            raise ValueError("packing screen was not recorded on RK3566")
        temperatures.append(snap["temperature_c"])
    def tensor_record(event):
        # Validate recorded identities only. Files themselves are NOT loaded.
        a, compressed = event["array_sha256"], event["sha256"]
        if any(not isinstance(h, str) or len(h) != 64 or set(h)-set("0123456789abcdef") for h in (a, compressed)):
            raise ValueError("invalid omitted-tensor digest")
        if event["path"] != "raw_"+a+".npz" or event["metrics"].get("passed") is not True:
            raise ValueError("recorded tensor identity or pass state differs")
        if event["path"] in unique_arrays and unique_arrays[event["path"]] != (compressed, a):
            raise ValueError("inconsistent omitted-tensor identity")
        unique_arrays[event["path"]] = compressed, a
    control("start")
    if temperatures[-1] > 60:
        raise ValueError("recorded start exceeded its threshold")
    loaded = take("backend_loaded")
    number(loaded["initialization_warmup_s"])
    control("loaded")
    rows = []
    policy_records = [{"policy": p} for p in POLICIES[:3]] + [
        {"policy": f"pack_{w}x{h}", "shape": [w, h]} for w, h in SHAPES]
    for k, mask in enumerate(masks):
        reference = take("reference")
        if reference["request"] != dict(mask, policy="reference"):
            raise ValueError("reference does not match mask")
        tensor_record(reference)
        for option in policy_records[k:]+policy_records[:k]:
            control("before_"+str(len(rows)))
            event = take("output_retained")
            if event["index"] != len(rows) or event["request"] != dict(mask, **option):
                raise ValueError("policy/mask schedule differs")
            tensor_record(event)
            metrics = event["metrics"]
            integrity = metrics["integrity"]
            service = number(metrics["service_ns"])/1e6
            mae = number(integrity["selected_mae"])
            if service <= 0 or mae > .001 or integrity.get("unselected_unchanged") is not True:
                raise ValueError("invalid recorded timing or numerical check")
            rows.append({"mask_id": mask["mask_id"], "policy": option["policy"],
                         "service_ms": service, "passed": True, "selected_mae": mae})
            control("after_"+str(len(rows)-1))
    control("closed")
    take("complete")
    if index != len(events) or len(rows) != 88 or len(audit["records"]) != 88:
        raise ValueError("extra events or missing records")
    for actual, recorded in zip(rows, audit["records"], strict=True):
        if any(actual[k] != recorded[k] for k in ("mask_id", "policy", "service_ms", "passed")) or abs(actual["selected_mae"]-recorded["selected_mae"]) > 1e-15:
            raise ValueError("private audit record differs from exact journal")
    if len(unique_arrays) != audit["raw_files_verified"]:
        raise ValueError("private audit raw-file count differs")
    lookup = {(r["mask_id"], r["policy"]): r["service_ms"] for r in rows}
    if set(lookup) != {(m, p) for m in expected_ids for p in POLICIES}:
        raise ValueError("incomplete eight-by-eleven matrix")
    means = {p: mean(lookup[m, p] for m in sorted(expected_ids)) for p in sorted(POLICIES)}
    def oracle(options):
        return mean(min(lookup[m, p] for p in options) for m in sorted(expected_ids))
    summary = {"means_ms": means, "packing_oracle_mean_ms": oracle(POLICIES[3:]),
               "conventional_oracle_mean_ms": oracle(POLICIES[:3]),
               "all_policy_oracle_mean_ms": oracle(POLICIES),
               "group_or_packing_oracle_mean_ms": oracle(("grouped",)+POLICIES[3:]),
               "packing_oracle_wins_against_grouped": sum(min(lookup[m, p] for p in POLICIES[3:]) < lookup[m, "grouped"] for m in expected_ids),
               "best_fixed_conventional": min(POLICIES[:3], key=means.get)}
    if any(summary[k] != audit[k] for k in summary):
        raise ValueError("recomputed all-policy summary differs from historical audit")
    best = min(POLICIES, key=means.get)
    if best != "grouped":
        raise ValueError("historical grouped-minimum statement does not match records")
    return {"cell": cell, "board": "RK3566", "observations": 88, **summary,
            "best_fixed_of_all_eleven": best, "best_fixed_packing": min(POLICIES[3:], key=means.get),
            "per_mask_ms": {m: {p: lookup[m, p] for p in POLICIES} for m in expected_ids},
            "journal_sha256": sha(raw), "omitted_raw_files": len(unique_arrays),
            "initialization_warmup_s": loaded["initialization_warmup_s"],
            "temperature_range_c": [min(temperatures), max(temperatures)]}


def replay(root):
    source = path(root, "output/paper3_packing_v1/E002_source")
    manifest_raw = path(source, "source_manifest.json").read_bytes()
    if sha(manifest_raw) != SOURCE_SHA:
        raise ValueError("packing source manifest changed")
    manifest = json.loads(manifest_raw)
    if set(manifest["files"]) != SOURCE_FILES or manifest.get("purpose") != "headroom_development_E002":
        raise ValueError("unexpected frozen source closure")
    for relative, digest in manifest["files"].items():
        if sha(path(source, relative).read_bytes()) != digest:
            raise ValueError("packing source changed")
    if sha(path(root, "docs/paper3_packing_v1/E002_result.md").read_bytes()) != REPORT_SHA:
        raise ValueError("historical outcome report changed")
    graphs = {}
    for cell in ("width24", "fp16"):
        audit_raw = path(root, f"results/paper3_packing_v1/E002_{cell}_audit.json").read_bytes()
        raw = path(root, f"output/paper3_packing_v1/E002_{cell}/journal.jsonl").read_bytes()
        graphs[cell] = analyze_journal(raw, json.loads(audit_raw), cell)
        graphs[cell]["private_audit_sha256"] = sha(audit_raw)
    return {"status": "packing_context_rank_replay_passed", "raw_pixels_revalidated": False,
            "source_manifest_sha256": SOURCE_SHA, "historical_report_sha256": REPORT_SHA, "graphs": graphs,
            "limits": ["Earlier RK3566 engineering screen, not the current RK3576 confirmation.",
                       "Eight already-open masks; one observation per organization/mask; no stable ranking or confidence claim.",
                       "Conservative true-image-boundary tiles use separate exact calls; not all possible packing implementations.",
                       "All eleven means and per-mask results are retained; hindsight oracles are not deployable policies.",
                       "Raw tensors omitted: numerical pass fields and prior audit hashes are provenance, not pixel revalidation.",
                       "Recorded controls are not independent historical or continuous-occupancy attestation.",
                       "The unchanged historical report predates this contextual artifact inclusion; it is not newly confirmed evidence."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.root)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "raw_pixels_revalidated": False,
                      "graphs": {c: {"observations": g["observations"], "means_ms": g["means_ms"],
                                     "best_fixed": g["best_fixed_of_all_eleven"]} for c, g in result["graphs"].items()}}))


if __name__ == "__main__":
    main()
