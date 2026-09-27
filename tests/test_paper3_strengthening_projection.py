"""Synthetic-only compact replay tests; never inspect held-out physical results."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from artifact.paper3_samos.build_strengthening_projection import (
    CELLS, LIMITS, SCHEMA, bind_private_reports, child, encode, journal_metadata,
    replay, require_hash, sha, source_bundle,
)
from src.report import paper3_strengthening_confirmation as confirmation
from src.report import paper3_strengthening_predictions as predictions
from test_paper3_strengthening_confirmation import authenticated_fixture


def setup_fixture(tmp_path):
    bundle, run, events = authenticated_fixture(tmp_path)
    frozen, method, pins = source_bundle(bundle, expected=None)
    raw = b"".join(encode(e).replace(b"\n", b"")+b"\n" for e in events)
    return bundle, run, events, frozen, pins, raw


def test_tensor_free_replay_preserves_scope(tmp_path):
    bundle, run, events, frozen, pins, raw = setup_fixture(tmp_path)
    # Crucially, no array path/root is passed to metadata replay. Its absence
    # cannot be mistaken for bytewise numerical verification.
    result = journal_metadata(raw, frozen, pins)
    assert result["omitted_unique_raw_arrays"] == 1
    assert result["fixed_reconstruction_pairs"] == 96
    assert len(result["records"]) == 416
    assert result["recorded_control_audit"]["snapshots"] == 934
    assert "not independently recomputed" in " ".join(LIMITS)


@pytest.mark.parametrize("mutation", ["order", "source", "profile", "jobs", "winner", "score",
                                    "timer", "array_hash", "array_conflict", "qualification",
                                    "control", "temperature", "terminal", "failure"])
def test_compact_replay_rejects_metadata_drift(tmp_path, mutation):
    bundle, run, events, frozen, pins, raw = setup_fixture(tmp_path)
    evaluations = [e for e in events if e.get("request", {}).get("kind") == "evaluation"]
    choice = next(e for e in evaluations if e["request"]["configuration"] == "frozen_choice")
    if mutation == "order":
        a, b = events.index(evaluations[0]), events.index(evaluations[1])
        events[a], events[b] = events[b], events[a]
    elif mutation == "source":
        events[0]["frozen_sha256"] = "0"*64
    elif mutation == "profile":
        next(e for e in events if e["state"] == "preparation")["profile_sha256"] = "0"*64
    elif mutation == "jobs":
        evaluations[0]["metrics"]["per_job"][0]["width"] = 128
    elif mutation == "winner":
        choice["metrics"]["chosen_reconstruction"] = "selected"
    elif mutation == "score":
        choice["metrics"]["frozen_calibrated_job_cost_ms"] = 3.1
    elif mutation == "timer":
        evaluations[0]["metrics"]["planning_ns"] = 1e12
    elif mutation == "array_hash":
        evaluations[0]["array_sha256"] = "not-a-hash"
    elif mutation == "array_conflict":
        evaluations[0]["array_sha256"] = "0"*64
    elif mutation == "qualification":
        next(e for e in events if e.get("request", {}).get("kind") == "qualification")["request"]["job"]["width"] = 65
    elif mutation == "control":
        next(e for e in events if e.get("stage") == "after_evaluation")["snapshot"]["dual_node_capture"]["stdout"] += " "
    elif mutation == "temperature":
        next(e for e in events if e.get("stage") == "after_evaluation")["snapshot"]["temperature_c"] = 80
    elif mutation == "terminal":
        events[-1]["observations"] = 415
    else:
        evaluations[0]["metrics"]["passed"] = False
    raw = b"\n".join(json.dumps(e).encode() for e in events)
    with pytest.raises(ValueError):
        journal_metadata(raw, frozen, pins)


def bound_reports(run):
    saved = {k: v for k, v in run.items() if k not in ("records", "samples", "omitted_unique_raw_arrays")}
    saved["unique_raw_arrays"] = run["omitted_unique_raw_arrays"]
    audit = json.loads(encode({"status": "complete_saved_record_audit", "runs": [saved]}))
    numerical = {"status": "retained_pixel_replay_passed", "confirmation": True,
                 "journal_sha256": run["journal_sha256"], "cell": run["cell"],
                 "session": run["session"], "source_manifest_sha256": run["source_manifest_sha256"],
                 "sample_count": 465, "fixed_job_reconstruction_pairs": 96,
                 "fixed_job_reconstruction_pairs_byte_identical": True,
                 "independent_fp16_reference_used": False, "full_reference_file_sha256": None}
    return audit, numerical


def test_binds_private_audits_without_claiming_pixel_replay(tmp_path):
    _, _, _, frozen, pins, raw = setup_fixture(tmp_path)
    run = journal_metadata(raw, frozen, pins)
    audit, numerical = bound_reports(run)
    bind_private_reports(audit, numerical, run)
    assert "raw_tensor_numerical_replay_performed" not in run
    numerical["journal_sha256"] = "0"*64
    with pytest.raises(ValueError, match="numerical-audit binding"):
        bind_private_reports(audit, numerical, run)


@pytest.mark.parametrize("field", ["status", "confirmation", "sample_count", "fixed_job_reconstruction_pairs_byte_identical", "independent_fp16_reference_used", "session", "source_manifest_sha256"])
def test_rejects_incomplete_or_misbound_numerical_report(tmp_path, field):
    _, _, _, frozen, pins, raw = setup_fixture(tmp_path)
    run = journal_metadata(raw, frozen, pins)
    audit, numerical = bound_reports(run)
    numerical[field] = None
    with pytest.raises(ValueError):
        bind_private_reports(audit, numerical, run)


@pytest.mark.parametrize("name", ["../outside", "C:/outside", "/outside", "a\\b", "a/../../outside"])
def test_rejects_unsafe_context_paths(tmp_path, name):
    with pytest.raises(ValueError):
        child(tmp_path, name)


def test_source_bundle_checks_files_and_method(tmp_path):
    bundle, _, _, _, _, _ = setup_fixture(tmp_path)
    (bundle/"dummy.py").write_text("# changed fixture\n")
    with pytest.raises(ValueError, match="source file changed"):
        source_bundle(bundle, expected=None)


def test_rejects_non_digest():
    with pytest.raises(ValueError):
        require_hash("" )


def complete_projection_fixture(tmp_path):
    """Four graph/session fixture, synthetic costs only, no physical outcomes."""
    fixture = tmp_path/"fixture"
    fixture.mkdir()
    bundle, _, original, frozen, _, _ = setup_fixture(fixture)
    root = tmp_path/"projection"
    root.mkdir()
    method = json.loads((bundle/"method_freeze.json").read_bytes())
    method["profiles"]["fp16"] = copy.deepcopy(method["profiles"]["width24"])
    method["compiled"]["fp16"] = copy.deepcopy(method["compiled"]["width24"])
    method["materialization_models"] = {c: {"call_ms": .1, "million_elements_ms": .01} for c in ("width24", "fp16")}
    method["calibration_bindings"] = {}
    def put(name, raw):
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    calibration = b"".join(json.dumps({"state": "sample", "request": {"kind": "profile"}, "metrics": {"passed": True, "service_ns": 1e6}}).encode()+b"\n" for _ in range(156))
    for cell in ("width24", "fp16"):
        put(f"calibration/{cell}.jsonl", calibration)
        method["calibration_bindings"][cell] = {"journal_sha256": sha(calibration)}
    for key in ("profiles", "compiled", "materialization_models"):
        frozen[key] = copy.deepcopy(method[key])
    base_schedule = frozen["schedule"]
    frozen["schedule"] = [dict(req, cell=c, session=s) for c, s in sorted(CELLS) for req in base_schedule]
    frozen["method_freeze_sha256"] = sha(encode(method))
    put("source/method_freeze.json", encode(method))
    put("source/frozen.json", encode(frozen))
    manifest = {"files": {n: sha((root/"source"/n).read_bytes()) for n in ("method_freeze.json", "frozen.json")}}
    put("source/source_manifest.json", encode(manifest))
    _, _, pins = source_bundle(root/"source", expected=None)
    runs = []
    for cell, session in sorted(CELLS):
        events = copy.deepcopy(original)
        events[0].update(cell=cell, session=session, source_manifest_sha=pins["source_manifest_sha256"],
                         frozen_sha256=pins["frozen_sha256"], method_freeze_sha256=pins["method_freeze_sha256"])
        for event in events:
            if event["state"] == "sample":
                event["request"].update(cell=cell, session=session)
                for job in event["metrics"].get("per_job", []):
                    job["output_elements"] = 64*60*12
                    for key in ("extraction_ns", "inference_ns", "normalization_ns", "scatter_ns"):
                        job[key] = event["metrics"][key]
                    job["job_ns"] = sum(job[k] for k in ("extraction_ns", "inference_ns", "normalization_ns", "scatter_ns"))
        raw = b"".join(json.dumps(e).encode()+b"\n" for e in events)
        run = journal_metadata(raw, frozen, pins)
        runs.append(run)
        audit, numerical = bound_reports(run)
        numerical["independent_fp16_reference_used"] = cell == "fp16"
        numerical["full_reference_file_sha256"] = predictions.FULL_REFERENCE_SHA if cell == "fp16" else None
        put(f"journals/{cell}_session{session}.jsonl", raw)
        put(f"private_audits/{cell}_session{session}_numerical.json", encode(numerical))
    graphs = {cell: confirmation.analyze([r for run in runs if run["cell"] == cell for r in run["records"]], frozen["masks"], confirmation.CONFIGURATIONS, chooser="frozen_choice") for cell in ("width24", "fp16")}
    audit = {"status": "complete_saved_record_audit", "graphs": graphs,
             "runs": [bound_reports(run)[0]["runs"][0] for run in runs]}
    put("private_audits/saved_record_audit.json", encode(audit))
    package = {"schema": SCHEMA, "limits": LIMITS, "raw_tensors_included": False,
               "files": {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in root.rglob("*") if p.is_file()}}
    put("projection.json", encode(package))
    return root


def test_complete_four_cell_replay_and_predictions(tmp_path):
    root = complete_projection_fixture(tmp_path)
    result = replay(root, expected_source=None)
    assert result["status"] == "compact_journal_replay_passed"
    assert result["raw_tensor_numerical_replay_performed"] is False
    assert result["private_numerical_reports_hash_bound"] is True
    assert len(result["runs"]) == 4
    assert all(r["evaluation_rows"] == 832 for r in result["graphs"].values())
    assert all(len(r["observations"]) == 832 for r in result["prediction_diagnostics"].values())
    assert all(r["calibration_measured_service_ms"] == 156 for r in result["prediction_diagnostics"].values())
    assert not list(root.rglob("*.npz"))


def test_projection_rejects_changed_journal_before_analysis(tmp_path):
    root = complete_projection_fixture(tmp_path)
    path = root/"journals/width24_session0.jsonl"
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="file changed"):
        replay(root, expected_source=None)


def test_projection_rejects_unlisted_payload(tmp_path):
    root = complete_projection_fixture(tmp_path)
    (root/"unlisted_model.rknn").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="unlisted files"):
        replay(root, expected_source=None)


def test_minimal_source_closure_replays_away_from_repository(tmp_path):
    projection = complete_projection_fixture(tmp_path)
    isolated = tmp_path/"isolated_source"
    repository = Path(__file__).resolve().parents[1]
    files = (
        "artifact/paper3_samos/build_strengthening_projection.py",
        "src/report/paper3_strengthening_confirmation.py",
        "src/report/paper3_strengthening_development.py",
        "src/report/paper3_strengthening_predictions.py",
        "src/experiments/paper3_dispatch_successor_v1/core.py",
    )
    for relative in files:
        target = isolated/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository/relative, target)
    before = {p.relative_to(projection).as_posix(): sha(p.read_bytes()) for p in projection.rglob("*") if p.is_file()}
    code = ("from pathlib import Path; from artifact.paper3_samos.build_strengthening_projection import replay; "
            "result=replay(Path("+repr(str(projection))+"), expected_source=None); "
            "assert result['status']=='compact_journal_replay_passed'; print('ISOLATED_SYNTHETIC_REPLAY_PASS')")
    environment = dict(os.environ, PYTHONPATH=str(isolated), PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.run([sys.executable, "-B", "-c", code], cwd=isolated, env=environment,
                             capture_output=True, text=True, timeout=60, check=True)
    assert "ISOLATED_SYNTHETIC_REPLAY_PASS" in process.stdout
    after = {p.relative_to(projection).as_posix(): sha(p.read_bytes()) for p in projection.rglob("*") if p.is_file()}
    assert before == after
