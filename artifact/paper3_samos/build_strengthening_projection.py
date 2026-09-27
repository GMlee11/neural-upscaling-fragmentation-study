"""Build/replay the compact E002 journal artifact; never run inference.

Exact journals, frozen source and completed private-audit reports are retained.
Timing, scheduling, recorded controls and frozen decisions are replayable here.
Raw output tensors and neural models are deliberately absent: retained-pixel
correctness is an author-reported, hash-bound private audit, NOT recomputed by
this compact replay. A SHA-256 binding is not independent historical attestation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import shutil

from src.report import paper3_strengthening_confirmation as confirmation
from src.report import paper3_strengthening_predictions as predictions

SCHEMA = "paper3_strengthening_compact_v1"
CELLS = {(c, s) for c in ("width24", "fp16") for s in (0, 1)}
LIMITS = [
    "Replays complete timing/control journals, frozen requests, decisions, and practical gates without a board or models.",
    "Raw output tensors and neural models are omitted; recorded tensor hashes and private numerical-audit reports are retained, not independently recomputed here.",
    "Recorded two-sweep controls are point-in-time evidence, not continuous NPU ownership or independent historical authentication.",
    "Hashes detect changes relative to this package; they do not prove historical timestamps, preregistration, or truthful execution.",
    "No inference of statistical significance, novelty, cross-vendor generality, or acceptance probability.",
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def child(root, name):
    """Reject path traversal, absolute names, Windows drive names and symlinks."""
    name = str(name)
    rel = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or rel.is_absolute() or ".." in rel.parts:
        raise ValueError("unsafe package path")
    root = Path(root).resolve()
    target = root.joinpath(*rel.parts)
    if not target.resolve().is_relative_to(root) or any(p.is_symlink() for p in (target, *target.parents) if p != root and p.is_relative_to(root)):
        raise ValueError("path escapes package or contains symlink")
    return target


def require_hash(value):
    if not isinstance(value, str) or len(value) != 64 or set(value)-set("0123456789abcdef"):
        raise ValueError("invalid SHA-256 digest")
    return value


def source_bundle(bundle, expected=confirmation.ACCEPTED_SOURCE_SHA):
    bundle = Path(bundle).resolve()
    manifest_raw = child(bundle, "source_manifest.json").read_bytes()
    if expected is not None and sha(manifest_raw) != expected:
        raise ValueError("source is not accepted frozen manifest")
    manifest = json.loads(manifest_raw)
    if not {"frozen.json", "method_freeze.json"}.issubset(manifest["files"]):
        raise ValueError("source manifest omits method or requests")
    for name, digest in manifest["files"].items():
        if sha(child(bundle, name).read_bytes()) != require_hash(digest):
            raise ValueError("frozen source file changed")
    frozen_raw = child(bundle, "frozen.json").read_bytes()
    method_raw = child(bundle, "method_freeze.json").read_bytes()
    frozen, method = json.loads(frozen_raw), json.loads(method_raw)
    if frozen.get("method_freeze_sha256") != sha(method_raw):
        raise ValueError("method/request binding differs")
    if method.get("practical_gates") != confirmation.PRACTICAL_GATES or method.get("candidate_order") != list(confirmation.CONFIGURATIONS):
        raise ValueError("practical gates or tie order differ")
    if any(method.get("bootstrap", {}).get(k) != v for k, v in confirmation.BOOTSTRAP_CONTRACT.items()):
        raise ValueError("bootstrap procedure differs")
    if any(method.get(k) != frozen.get(k) for k in ("profiles", "compiled", "materialization_models")):
        raise ValueError("frozen cost/profile binding differs")
    return frozen, method, {"source_manifest_sha256": sha(manifest_raw),
                            "frozen_sha256": sha(frozen_raw), "method_freeze_sha256": sha(method_raw)}


def journal_metadata(raw, frozen, pins):
    """Authenticate journal metadata only; never substitute for pixel replay."""
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not events or events[0].get("state") != "running" or events[-1].get("state") != "complete" or any(e.get("state") == "failed" for e in events):
        raise ValueError("unfinished or failed physical cell")
    start, end = events[0], events[-1]
    ident = start["cell"], start["session"]
    if ident not in CELLS or start.get("accepted_bundle_sha") != confirmation.ACCEPTED_BUNDLE_SHA:
        raise ValueError("unknown graph/session or accepted model bundle")
    if any(start.get(k) != pins[p] for k, p in (("source_manifest_sha", "source_manifest_sha256"),
            ("frozen_sha256", "frozen_sha256"), ("method_freeze_sha256", "method_freeze_sha256"))):
        raise ValueError("journal/source binding differs")
    if end.get("confirmation") is not True or tuple(end.get(k) for k in ("observations", "qualifications", "references")) != (416, 48, 1):
        raise ValueError("wrong terminal counts")
    try:
        controls = confirmation._recorded_controls(events)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("malformed recorded controls") from exc
    prepared = next(e for e in events if e["state"] == "preparation")
    profile, tables = frozen["profiles"][ident[0]], frozen["compiled"][ident[0]]
    if prepared["profile_sha256"] != sha(json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()):
        raise ValueError("profile digest differs")
    if any(prepared[k] != tables[k] for k in ("score_sha256", "serialized_score_bytes")) or set(prepared["tables"]) != set(tables["tables"]):
        raise ValueError("compiled table identity differs")
    for name, table in prepared["tables"].items():
        if any(table[k] != tables["tables"][name][k] for k in ("cost_sha256", "table_sha256", "serialized_table_bytes")):
            raise ValueError("compiled planner digest differs")
    if sum(confirmation._number(t["preparation_ns"]) for t in prepared["tables"].values()) > confirmation._number(prepared["preparation_ns"]):
        raise ValueError("nested preparation exceeds complete preparation")
    samples = [e for e in events if e["state"] == "sample"]
    expected_kinds = ["reference"]+["qualification"]*48+["evaluation"]*416
    if [e["request"]["kind"] for e in samples] != expected_kinds:
        raise ValueError("incomplete or reordered sample matrix")
    if any((e["request"].get("cell"), e["request"].get("session")) != ident for e in samples):
        raise ValueError("sample cell/session differs")
    evaluations = samples[49:]
    if [e["request"] for e in evaluations] != [r for r in frozen["schedule"] if (r["cell"], r["session"]) == ident]:
        raise ValueError("frozen evaluation schedule differs")
    references = {}
    for sample in samples:
        if sample["metrics"].get("passed") is not True:
            raise ValueError("recorded numerical failure")
        name = sample["path"]
        if Path(name).name != name or "/" in name or "\\" in name or not name.startswith("raw_") or not name.endswith(".npz"):
            raise ValueError("invalid omitted tensor name")
        pair = require_hash(sample["sha256"]), require_hash(sample["array_sha256"])
        if name in references and references[name] != pair:
            raise ValueError("inconsistent omitted tensor digest")
        references[name] = pair
    for i, (y, width, right) in enumerate(itertools.product((0, 300), confirmation.WIDTHS, (False, True))):
        a, b = samples[1+2*i:3+2*i]
        job = {"x": 640-width if right else 0, "y": y, "width": width, "commits": [y//60*10+(9 if right else 0)]}
        if a["request"]["job"] != job or b["request"]["job"] != job or a["request"]["reconstruction"] != "whole" or b["request"]["reconstruction"] != "selected" or a["array_sha256"] != b["array_sha256"]:
            raise ValueError("qualification edge coverage or recorded digest agreement differs")
    normalized, indexed = [], {}
    def signature(sample):
        return tuple((j["x"], j["y"], j["width"], tuple(j["commits"])) for j in sample["metrics"]["per_job"])
    for sample in evaluations:
        req, met = sample["request"], sample["metrics"]
        if sum(confirmation._number(met[k]) for k in confirmation.STAGES) > confirmation._number(met["service_ns"]) or confirmation._number(met["decision_ns"]) > confirmation._number(met["planning_ns"]):
            raise ValueError("invalid complete service boundaries")
        expected, winner = confirmation.frozen_expectations(profile, req["rows"])
        chosen = met["chosen_policy"]+":"+met["chosen_reconstruction"]
        if chosen not in expected or signature(sample) != expected[chosen]["jobs"] or len(met["per_job"]) != met["calls"]:
            raise ValueError("jobs or calls differ from frozen planner")
        if req["configuration"] == "frozen_choice":
            if chosen != winner or confirmation._number(met["frozen_calibrated_job_cost_ms"]) != expected[winner]["score_ms"]:
                raise ValueError("frozen chooser score or tie rule differs")
        elif chosen != req["configuration"] or met.get("frozen_calibrated_job_cost_ms") is not None:
            raise ValueError("fixed configuration changed")
        key = req["repeat"], req["mask_id"], req["configuration"]
        if key in indexed:
            raise ValueError("duplicate evaluation identity")
        indexed[key] = sample
        normalized.append({k: req[k] for k in ("session", "repeat", "mask_id", "configuration")} |
                          {k: met[k] for k in ("service_ns", "normalization_ns")})
    fixed_pairs = 0
    for (repeat, mask, config), sample in indexed.items():
        other = None
        if config in ("exact:whole", "row_spans:whole", "grouped:whole"):
            other = config.replace(":whole", ":selected"); fixed_pairs += 1
        elif config == "frozen_choice":
            other = sample["metrics"]["chosen_policy"]+":"+sample["metrics"]["chosen_reconstruction"]
        if other:
            paired = indexed[repeat, mask, other]
            if signature(sample) != signature(paired) or sample["array_sha256"] != paired["array_sha256"]:
                raise ValueError("paired jobs or recorded output digests differ")
    return {"cell": ident[0], "session": ident[1], "journal_sha256": sha(raw), **pins,
            "recorded_control_audit": controls, "preparation": prepared,
            "omitted_unique_raw_arrays": len(references), "fixed_reconstruction_pairs": fixed_pairs,
            "records": normalized, "samples": samples}


def bind_private_reports(audit, numerical, run):
    """Reports are retained provenance, not a replacement for omitted tensors."""
    if audit.get("status") != "complete_saved_record_audit":
        raise ValueError("completed private saved-record audit required")
    matches = [r for r in audit["runs"] if (r["cell"], r["session"]) == (run["cell"], run["session"])]
    if len(matches) != 1:
        raise ValueError("private audit cell coverage differs")
    report = matches[0]
    for key in ("journal_sha256", "source_manifest_sha256", "frozen_sha256", "fixed_reconstruction_pairs", "preparation", "recorded_control_audit"):
        if encode(report[key]) != encode(run[key]):
            raise ValueError("private saved-record audit binding differs")
    if report["unique_raw_arrays"] != run["omitted_unique_raw_arrays"]:
        raise ValueError("private raw-array count differs")
    if (numerical.get("status") != "retained_pixel_replay_passed" or numerical.get("confirmation") is not True or
            numerical.get("journal_sha256") != run["journal_sha256"] or numerical.get("cell") != run["cell"] or
            numerical.get("session") != run["session"] or numerical.get("source_manifest_sha256") != run["source_manifest_sha256"] or
            numerical.get("sample_count") != 465 or numerical.get("fixed_job_reconstruction_pairs") != 96 or
            numerical.get("fixed_job_reconstruction_pairs_byte_identical") is not True or
            numerical.get("independent_fp16_reference_used") is not (run["cell"] == "fp16") or
            numerical.get("full_reference_file_sha256") != (predictions.FULL_REFERENCE_SHA if run["cell"] == "fp16" else None)):
        raise ValueError("completed private numerical-audit binding differs")


def replay(root, expected_source=confirmation.ACCEPTED_SOURCE_SHA):
    root = Path(root).resolve()
    package = json.loads(child(root, "projection.json").read_bytes())
    if package.get("schema") != SCHEMA or package.get("limits") != LIMITS or package.get("raw_tensors_included") is not False:
        raise ValueError("unknown projection or overstated verification boundary")
    allowed = set(package["files"]) | {"projection.json", "expected_replay.json"}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if not actual.issubset(allowed) or not set(package["files"]).issubset(actual):
        raise ValueError("projection inventory is incomplete or contains unlisted files")
    for name, digest in package["files"].items():
        if sha(child(root, name).read_bytes()) != require_hash(digest):
            raise ValueError("compact package file changed")
    frozen, method, pins = source_bundle(child(root, "source"), expected_source)
    audit = json.loads(child(root, "private_audits/saved_record_audit.json").read_bytes())
    runs, samples = [], {c: [] for c in ("width24", "fp16")}
    for cell, session in sorted(CELLS):
        ident = f"{cell}_session{session}"
        run = journal_metadata(child(root, f"journals/{ident}.jsonl").read_bytes(), frozen, pins)
        if (run["cell"], run["session"]) != (cell, session):
            raise ValueError("package run identity differs")
        numerical = json.loads(child(root, f"private_audits/{ident}_numerical.json").read_bytes())
        bind_private_reports(audit, numerical, run)
        samples[cell].extend(run.pop("samples")); runs.append(run)
    if len(audit["runs"]) != 4:
        raise ValueError("private audit has unexpected cells")
    graphs, models = {}, {}
    for cell in samples:
        graphs[cell] = confirmation.analyze([r for run in runs if run["cell"] == cell for r in run["records"]], frozen["masks"], confirmation.CONFIGURATIONS, chooser="frozen_choice")
        if graphs[cell] != audit["graphs"][cell]:
            raise ValueError("replayed practical gates differ from private saved-record audit")
        raw = child(root, f"calibration/{cell}.jsonl").read_bytes()
        if sha(raw) != method["calibration_bindings"][cell]["journal_sha256"]:
            raise ValueError("original calibration journal differs")
        calibration = [e for e in map(json.loads, raw.splitlines()) if e["state"] == "sample" and e["request"]["kind"] in ("qualification", "profile")]
        if len(calibration) != 156 or any(e["metrics"].get("passed") is not True for e in calibration):
            raise ValueError("incomplete or failed original calibration")
        measured = predictions.total(e["metrics"]["service_ns"] for e in calibration)/1e6
        models[cell] = predictions.analyze(samples[cell], frozen["profiles"][cell], frozen["materialization_models"][cell], [r["preparation"]["preparation_ns"]/1e6 for r in runs if r["cell"] == cell], measured)
    return {"status": "compact_journal_replay_passed", "raw_tensor_numerical_replay_performed": False,
            "private_numerical_reports_hash_bound": True, "limits": LIMITS,
            "graphs": graphs, "prediction_diagnostics": models,
            "runs": [{k: v for k, v in r.items() if k != "records"} for r in runs]}


def build(bundle, run_directories, audit_path, numerical_paths, calibration_directories, output, retained_files=()):
    """Create a NEW stage only, after all four private numerical audits pass."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to replace an existing projection")
    frozen, method, pins = source_bundle(bundle)
    audit = json.loads(Path(audit_path).read_bytes())
    numerical = {json.loads(Path(p).read_bytes())["journal_sha256"]: Path(p) for p in numerical_paths}
    if len(numerical_paths) != 4 or len(numerical) != 4 or len(run_directories) != 4:
        raise ValueError("exactly four completed physical and numerical audit cells required")
    inputs = {"private_audits/saved_record_audit.json": Path(audit_path)}
    manifest = json.loads((Path(bundle)/"source_manifest.json").read_bytes())
    inputs.update({"source/"+name: child(bundle, name) for name in (*manifest["files"], "source_manifest.json")})
    found = set()
    for directory in run_directories:
        path = Path(directory)/"journal.jsonl"
        run = journal_metadata(path.read_bytes(), frozen, pins)
        key = run["cell"], run["session"]
        if key in found or run["journal_sha256"] not in numerical:
            raise ValueError("duplicate run or absent numerical report")
        found.add(key)
        report = numerical[run["journal_sha256"]]
        bind_private_reports(audit, json.loads(report.read_bytes()), run)
        ident = f"{key[0]}_session{key[1]}"
        inputs[f"journals/{ident}.jsonl"] = path
        inputs[f"private_audits/{ident}_numerical.json"] = report
    if found != CELLS:
        raise ValueError("four physical cells are not complete")
    calibration = {sha((Path(p)/"journal.jsonl").read_bytes()): Path(p)/"journal.jsonl" for p in calibration_directories}
    if len(calibration_directories) != 2:
        raise ValueError("two original development calibration journals required")
    for cell in ("width24", "fp16"):
        inputs[f"calibration/{cell}.jsonl"] = calibration[method["calibration_bindings"][cell]["journal_sha256"]]
    # Explicit append-only lineage and zero-sample engineering failures can be
    # retained intact without pooling them into confirmatory measurements.
    # Arguments are logical relative names paired with already inspected files.
    for name, path in retained_files:
        key = "retained_context/"+name
        child(output, key)
        if key in inputs:
            raise ValueError("duplicate retained context name")
        inputs[key] = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    for name, path in inputs.items():
        target = child(output, name); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    package = {"schema": SCHEMA, "raw_tensors_included": False, "limits": LIMITS,
               "files": {name: sha(child(output, name).read_bytes()) for name in sorted(inputs)}}
    (output/"projection.json").write_bytes(encode(package))
    result = replay(output)
    (output/"expected_replay.json").write_bytes(encode(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build")
    make.add_argument("--bundle", type=Path, required=True)
    make.add_argument("--run", type=Path, action="append", required=True)
    make.add_argument("--audit", type=Path, required=True)
    make.add_argument("--numerical", type=Path, action="append", required=True)
    make.add_argument("--calibration", type=Path, action="append", required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--retained-file", action="append", default=[], metavar="NAME=PATH")
    check = commands.add_parser("replay")
    check.add_argument("root", type=Path)
    check.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        retained = [tuple(item.split("=", 1)) for item in args.retained_file]
        if any(len(pair) != 2 for pair in retained):
            parser.error("--retained-file requires NAME=PATH")
        result = build(args.bundle, args.run, args.audit, args.numerical, args.calibration, args.output, retained)
    else:
        result = replay(args.root)
        if (args.root/"expected_replay.json").read_bytes() != encode(result):
            raise ValueError("replayed output differs from packaged expected output")
        if args.output:
            with args.output.open("xb") as stream:
                stream.write(encode(result))
    print(json.dumps({"status": result["status"], "numerical_replay_performed": False}))


if __name__ == "__main__":
    main()
