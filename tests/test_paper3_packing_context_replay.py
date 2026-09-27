"""Replay/mutate completed historical packing records, never live confirmation."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from artifact.paper3_samos.replay_packing_context import (
    POLICIES, SOURCE_FILES, analyze_journal, path, replay, sha,
)


def context_root():
    root = Path(__file__).resolve().parents[1]
    packaged = root/"results/paper3_strengthening_v2/public_v1/retained_context/packing"
    return packaged if packaged.is_dir() else root


def records(cell="width24"):
    root = context_root()
    raw = (root/f"output/paper3_packing_v1/E002_{cell}/journal.jsonl").read_bytes()
    audit = json.loads((root/f"results/paper3_packing_v1/E002_{cell}_audit.json").read_bytes())
    return raw, audit


def test_all_observed_policies_and_masks_are_retained():
    result = replay(context_root())
    assert result["status"] == "packing_context_rank_replay_passed"
    assert result["raw_pixels_revalidated"] is False
    for cell, expected in (("width24", 137.597830375), ("fp16", 169.1337915)):
        graph = result["graphs"][cell]
        assert graph["board"] == "RK3566"
        assert graph["observations"] == 88
        assert graph["best_fixed_of_all_eleven"] == "grouped"
        assert graph["means_ms"]["grouped"] == expected
        assert set(graph["means_ms"]) == set(POLICIES)
        assert len(graph["per_mask_ms"]) == 8
        assert all(set(row) == set(POLICIES) for row in graph["per_mask_ms"].values())
    assert result["graphs"]["width24"]["packing_oracle_wins_against_grouped"] == 3
    assert result["graphs"]["fp16"]["packing_oracle_wins_against_grouped"] == 4


@pytest.mark.parametrize("mutation", ["hash", "incomplete", "confirmed", "wrong_board", "source", "missing",
    "duplicate", "policy", "geometry", "temperature", "pass", "mae", "fallback", "nan", "audit_mean", "audit_row"])
def test_rejects_changed_coverage_controls_or_claims(mutation):
    raw, audit = records()
    events = [json.loads(line) for line in raw.splitlines()]
    output = next(e for e in events if e["state"] == "output_retained")
    if mutation == "hash":
        audit["journal_sha256"] = "0"*64
        with pytest.raises(ValueError):
            analyze_journal(raw, audit, "width24")
        return
    if mutation == "incomplete":
        events[-1]["observations"] = 87
    elif mutation == "confirmed":
        events[-1]["confirmation"] = True
    elif mutation == "wrong_board":
        next(e for e in events if e["state"] == "control")["snapshot"]["target_snapshot"]["invariants"]["compatible"] = ["rockchip,rk3576"]
    elif mutation == "source":
        events[0]["source_manifest_sha"] = "0"*64
    elif mutation == "missing":
        events.remove(output)
    elif mutation == "duplicate":
        events.insert(events.index(output), copy.deepcopy(output))
    elif mutation == "policy":
        output["request"]["policy"] = "invented"
    elif mutation == "geometry":
        output["request"]["rows"][0] = "1111111111"
    elif mutation == "temperature":
        next(e for e in events if e["state"] == "control")["snapshot"]["temperature_c"] = 80
    elif mutation == "pass":
        output["metrics"]["passed"] = False
    elif mutation == "mae":
        output["metrics"]["integrity"]["selected_mae"] = .002
    elif mutation == "fallback":
        output["metrics"]["integrity"]["unselected_unchanged"] = False
    elif mutation == "nan":
        output["metrics"]["service_ns"] = float("nan")
    elif mutation == "audit_mean":
        audit["means_ms"]["grouped"] = 0
    else:
        audit["records"][0]["service_ms"] = 0
    raw = b"".join(json.dumps(e).encode()+b"\n" for e in events)
    audit["journal_sha256"] = sha(raw)
    with pytest.raises(ValueError):
        analyze_journal(raw, audit, "width24")


def copy_context(destination):
    files = ["output/paper3_packing_v1/E002_source/source_manifest.json",
             "docs/paper3_packing_v1/E002_result.md"]
    files += ["output/paper3_packing_v1/E002_source/"+name for name in SOURCE_FILES]
    for cell in ("width24", "fp16"):
        files.extend([f"output/paper3_packing_v1/E002_{cell}/journal.jsonl",
                      f"results/paper3_packing_v1/E002_{cell}_audit.json"])
    for relative in files:
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(context_root()/relative, target)


def test_arbitrary_compact_context_root_without_raw_arrays(tmp_path):
    copy_context(tmp_path)
    before = {p.relative_to(tmp_path).as_posix(): sha(p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()}
    assert replay(tmp_path)["raw_pixels_revalidated"] is False
    assert not list(tmp_path.rglob("*.npz"))
    assert before == {p.relative_to(tmp_path).as_posix(): sha(p.read_bytes()) for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("relative", ["output/paper3_packing_v1/E002_source/packing/core.py",
                                    "docs/paper3_packing_v1/E002_result.md"])
def test_rejects_modified_source_or_historical_report(tmp_path, relative):
    copy_context(tmp_path)
    candidate = tmp_path/relative
    candidate.write_bytes(candidate.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="changed"):
        replay(tmp_path)


def test_rejects_escaping_source_name(tmp_path):
    with pytest.raises(ValueError):
        path(tmp_path, "../outside")
