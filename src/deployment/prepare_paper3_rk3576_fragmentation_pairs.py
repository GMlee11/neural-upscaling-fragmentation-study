"""Generate a non-authorizing offline preflight record for Paper 3 P3C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.benchmark_rk3576_fragmentation_pairs import (  # noqa: E402
    DEFAULT_CATALOG,
    DEFAULT_CONTRACT,
    build_offline_preflight,
)


DEFAULT_OUTPUT = ROOT / "results/preflight/paper3_rk3576_fragmentation_pairs_v1.json"


def write_exclusive(path: Path, payload: object) -> None:
    """Write a preflight candidate once without replacing prior review state."""

    if path.exists():
        raise FileExistsError(f"preflight record already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_offline_preflight(args.contract.resolve(), args.catalog.resolve())
        write_exclusive(args.output.resolve(), payload)
    except (FileExistsError, FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"PAPER3_FRAGMENTATION_PREFLIGHT_BLOCKED: {error}", file=sys.stderr)
        return 1
    print(
        "PAPER3_FRAGMENTATION_PREFLIGHT_READY "
        f"status={payload['contract_status']} schedule={payload['schedule']['entry_count']} "
        f"authorized={str(payload['authorized_for_samples']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
