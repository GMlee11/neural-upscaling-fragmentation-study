"""Qualify and measure the instrumentation-corrected Paper 3 V24 cell."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment import run_paper3_shared_v2_fp16_rk3566_v23 as previous
from src.deployment import run_paper3_shared_v2_fp16_v22 as base


ACTION_ID = "paper3_shared_v2_fp16_rk3566_replication_v24"
CONTROL_NODE = Path("/dev/dri/card1")
RENDER_ALIAS = Path("/dev/dri/renderD129")
RKNPU_DRIVER = Path("/sys/bus/platform/drivers/RKNPU")


class V24RuntimeError(RuntimeError):
    """A frozen V24 target-side gate failed."""


def driver_for(node: Path) -> Path:
    return Path(f"/sys/class/drm/{node.name}/device/driver").resolve()


def board_facts() -> dict[str, Any]:
    facts = previous.board_facts()
    if not CONTROL_NODE.exists() or not RENDER_ALIAS.exists():
        raise V24RuntimeError("required RK3566 RKNPU DRM nodes are absent")
    control_driver = driver_for(CONTROL_NODE)
    render_driver = driver_for(RENDER_ALIAS)
    if control_driver != RKNPU_DRIVER or render_driver != RKNPU_DRIVER:
        raise V24RuntimeError(
            f"DRM nodes are not both bound to RKNPU: {control_driver}, {render_driver}"
        )
    facts["rknpu_control_node"] = str(CONTROL_NODE)
    facts["rknpu_render_alias"] = str(RENDER_ALIAS)
    facts["rknpu_control_driver"] = str(control_driver)
    facts["rknpu_render_driver"] = str(render_driver)
    return facts


def load_contract(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise V24RuntimeError("contract is not a mapping")
    if protocol.get("schema_version") != 24 or protocol.get("experiment_id") != (
        "paper3_shared_v2_fp16_rk3566_fragmentation_v24"
    ):
        raise V24RuntimeError("unexpected V24 protocol identity")
    return previous.load_contract(path)


class RKNNBackend(previous.RKNNBackend):
    """Use the unchanged V23 catalog while authenticating its actual control FD."""

    @staticmethod
    def _infer(record: tuple[Any, Mapping[str, Any]], value: Any) -> Any:
        runtime, row = record
        if list(value.shape) != list(row["input_shape_nhwc"]):
            raise V24RuntimeError(f"input shape changed for {row['graph_id']}: {value.shape}")
        before = previous.fd_targets()
        outputs = runtime.inference(inputs=[value[None, ...]], data_format=["nhwc"])
        after = previous.fd_targets()
        observed = sorted(
            {
                target
                for target in (*before.values(), *after.values())
                if target == str(CONTROL_NODE)
            }
        )
        if observed != [str(CONTROL_NODE)]:
            raise V24RuntimeError("inference retained no authenticated RKNPU control-node FD")
        output = __import__("numpy").asarray(outputs[0]) if outputs else None
        if output is None or list(output.shape) != list(row["output_shape_nchw"]):
            raise V24RuntimeError(f"output shape changed for {row['graph_id']}")
        return output, {
            "control_nodes": observed,
            "control_driver": str(driver_for(CONTROL_NODE)),
            "fd_count_before": len(before),
            "fd_count_after": len(after),
        }


def write_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(previous.canonical_json(value))
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "measure", "analyze"))
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    start = None
    backend = None
    try:
        if args.output.exists():
            raise FileExistsError(f"terminal output exists: {args.output}")
        if args.command == "analyze":
            value = base.analyze(json.loads(args.capture.read_text(encoding="utf-8")))
            value["status"] = "V24_RK3566_OFFLINE_ANALYSIS_COMPLETE"
        else:
            start = board_facts()
            if args.command == "measure":
                if args.qualification is None or not args.qualification.is_file():
                    raise V24RuntimeError("measurement requires the separate qualification manifest")
                qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
                if qualification.get("status") != "V24_RK3566_NUMERICAL_QUALIFICATION_PASS":
                    raise V24RuntimeError("measurement is blocked until V24 qualification passes")
            contract = load_contract(args.contract)
            backend = RKNNBackend(args.catalog, args.package_root)
            value = (
                base.qualify(backend, contract, args.reference)
                if args.command == "qualify"
                else base.measure(backend, contract, args.schedule)
            )
            if not backend.npu_usage_evidence:
                raise V24RuntimeError("no RKNPU control-node evidence was retained")
            value["status"] = (
                "V24_RK3566_NUMERICAL_QUALIFICATION_PASS"
                if args.command == "qualify"
                else "V24_RK3566_PHYSICAL_CAPTURE_COMPLETE"
            )
            value["npu_usage_evidence"] = backend.npu_usage_evidence[:16]
            value["board_start"] = start
            value["board_end"] = board_facts()
            value["graphs"] = backend.graphs
        value.update(
            {
                "schema_version": 24,
                "experiment_id": ACTION_ID,
                "physical_execution_measured": args.command == "measure",
            }
        )
        write_atomic(args.output, value)
        print(value["status"])
        return 0
    except BaseException as error:
        failure = {
            "schema_version": 24,
            "experiment_id": ACTION_ID,
            "status": f"V24_{args.command.upper()}_TERMINAL_FAILURE",
            "failure": {"type": type(error).__name__, "message": str(error)},
            "board_start": start,
            "physical_execution_measured": args.command == "measure",
        }
        if not args.output.exists():
            write_atomic(args.output, failure)
        print(failure["status"] + ": " + str(error))
        return 2
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
