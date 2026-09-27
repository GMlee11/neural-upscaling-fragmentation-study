"""Build the narrow evidence manifest for Paper 3 fragmentation pairs V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.benchmark_rk3576_sparse import sha256_file  # noqa: E402


EXPERIMENT_ID = "paper3_rk3576_fragmentation_pairs_v1"
DEFAULT_CONTRACT = ROOT / "configs/paper3_rk3576_fragmentation_pairs_v1.yaml"
CONTROL_PATHS = (
    Path("configs/paper3_rk3576_fragmentation_pairs_v1.yaml"),
    Path("docs/paper3_rk3576_fragmentation_pairs_v1.md"),
    Path("scripts/rk3576/run_paper3_rk3576_fragmentation_pairs_v1.sh"),
    Path("src/deployment/benchmark_rk3576_fragmentation_pairs.py"),
    Path("src/deployment/prepare_paper3_rk3576_fragmentation_pairs.py"),
    Path("src/report/paper3_rk3576_fragmentation_pairs_v1.py"),
    Path("src/report/paper3_rk3576_fragmentation_pairs_manifest.py"),
    Path("tests/test_paper3_rk3576_fragmentation_pairs_contract.py"),
    Path("tests/test_benchmark_rk3576_fragmentation_pairs.py"),
    Path("tests/test_paper3_rk3576_fragmentation_pairs_analysis.py"),
    Path("tests/test_paper3_rk3576_fragmentation_pairs_manifest.py"),
)


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": resolved.relative_to(root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def build_manifest(
    contract: Mapping[str, Any],
    *,
    root: Path,
    control_paths: Sequence[Path] = CONTROL_PATHS,
) -> dict[str, Any]:
    """Hash only the preregistration, P3C implementation, and expected outputs."""

    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected fragmentation-pair experiment_id")
    expected = contract["expected_outputs_after_future_authorization"]
    if bool(expected["current_phase_outputs_must_not_exist"]):
        raise ValueError("contract still forbids result-manifest creation")
    evidence_names = ("raw_capture", "analysis_json", "analysis_markdown", "latex_candidate")
    evidence = [_file_record(root, Path(str(expected[name]))) for name in evidence_names]
    analysis_path = root / str(expected["analysis_json"])
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("experiment_id") != EXPERIMENT_ID or analysis.get("status") != "analysis_complete":
        raise ValueError("analysis artifact is incomplete or belongs to another experiment")
    controls = [_file_record(root, path) for path in control_paths]
    return {
        "schema_version": 1,
        "manifest_id": "paper3_rk3576_fragmentation_pairs_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "evidence_complete",
        "physical_execution_measured": True,
        "evidence_files": evidence,
        "control_files": controls,
        "claim_boundary": contract["claim_boundary"],
        "analysis_disposition": analysis["disposition"],
        "all_primary_gates_passed": bool(analysis["all_primary_gates_passed"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            raise ValueError("contract root must be a mapping")
        output = args.output or ROOT / str(
            contract["expected_outputs_after_future_authorization"]["evidence_manifest"]
        )
        output = output.resolve()
        if output.exists():
            raise FileExistsError(f"manifest output already exists: {output}")
        manifest = build_manifest(contract, root=ROOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except (FileExistsError, FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"PAPER3_FRAGMENTATION_MANIFEST_BLOCKED: {error}", file=sys.stderr)
        return 1
    print(f"PAPER3_FRAGMENTATION_MANIFEST_COMPLETE files={len(manifest['evidence_files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
