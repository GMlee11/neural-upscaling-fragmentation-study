"""Compact reviewer replay rejects changed timing, decisions and provenance."""
import copy
import json
from pathlib import Path

import pytest

from src.report.paper3_dispatch_publication_v1 import INPUT, latex, summarize

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def payload():
    return json.loads((ROOT / INPUT).read_bytes())


def test_frozen_replay(payload):
    report = summarize(payload)
    expected = json.loads((ROOT / "results/tables/paper3_dispatch_publication_v1.json").read_bytes())
    assert report == expected
    assert latex(report) == (ROOT / "docs/paper/generated/paper3_dispatch_publication_v1.tex").read_text()
    assert sum(c["requests"] for c in report["cells"]) == 5120
    assert report["cells"][1]["policies"]["calibrated"]["session_means_ms"][0] > report["cells"][1]["policies"]["always_span"]["session_means_ms"][0]
    assert all(c["policies"]["calibrated"]["regressions_vs_best_fixed"] > 0 for c in report["cells"])
    assert report["failed_fp16_v1_excluded"]


@pytest.mark.parametrize("change", ["time", "mode", "schedule", "control", "count", "audit", "repeat_label", "journal", "coefficient"])
def test_rejects_mutation(payload, change):
    value = copy.deepcopy(payload)
    cell, row = value["cells"][0], value["cells"][0]["samples"][0]
    if change == "time": row["service_end_ns"] += 1
    elif change == "mode": row["chosen_mode"] = "invalid"
    elif change == "schedule": row["request"]["block"] = 8
    elif change == "control": row["invariants_sha256"][0] = "0" * 64
    elif change == "count": cell["samples"].pop()
    elif change == "audit": cell["full_raw_audit"]["policies"]["calibrated"]["mean_service_ms"] += 1
    elif change == "repeat_label": cell["full_raw_audit"]["previously_opened_geometry_replication"] = True
    elif change == "journal": cell["sessions"][0]["journal"]["sha256"] = "0" * 64
    elif change == "coefficient": cell["pins"]["coefficients"][0] = -1
    with pytest.raises(ValueError):
        summarize(value)
