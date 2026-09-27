"""Qualify and measure the frozen Paper 3 V23 catalog on RK3566."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment import run_paper3_shared_v2_fp16_v22 as base  # noqa: E402


ACTION_ID = "paper3_shared_v2_fp16_rk3566_replication_v23"
TARGET = "rk3566"


class V23RuntimeError(RuntimeError):
    """A frozen RK3566 target-side gate failed."""


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")


def write_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json(value))
    os.replace(temporary, path)


def load_contract(path: Path) -> dict[str, Any]:
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise V23RuntimeError("contract is not a mapping")
    anchor_relative = protocol.get("workload", {}).get("anchor_mask_contract")
    if anchor_relative != "configs/paper3_rk3576_fragmentation_pairs_v1.yaml":
        raise V23RuntimeError("unexpected anchor contract")
    if protocol.get("frozen_source", {}).get("mask_rows_sha256") != base.MASK_ROWS_SHA256:
        raise V23RuntimeError("mask identity changed")
    anchor_path = ROOT / str(anchor_relative)
    anchor = yaml.safe_load(anchor_path.read_text(encoding="utf-8"))
    if not isinstance(anchor, dict) or anchor.get("mask_contract", {}).get("canonical_rows_sha256") != base.MASK_ROWS_SHA256:
        raise V23RuntimeError("resolved anchor mask contract identity changed")
    anchor["workload"]["halo_pixels"] = base.HALO
    return anchor


def board_facts() -> dict[str, Any]:
    facts = base.board_facts()
    compatible = str(facts.get("compatible", ""))
    if "rk3566" not in compatible.lower() and "rk3568" not in compatible.lower():
        raise V23RuntimeError(f"unexpected RK3566-compatible device tree: {compatible}")
    device = Path("/dev/dri/renderD129")
    driver = Path("/sys/bus/platform/drivers/RKNPU")
    if not device.exists() or not driver.exists():
        raise V23RuntimeError("RK3566 RKNPU DRM device/driver binding is absent")
    facts.update({"target": TARGET, "npu_device": str(device), "npu_driver_binding": str(driver),
                  "npu_device_stat": {"mode": device.stat().st_mode & 0o777,
                                       "major": os.major(device.stat().st_rdev),
                                       "minor": os.minor(device.stat().st_rdev)}})
    return facts


def fd_targets() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in Path("/proc/self/fd").glob("*"):
        try:
            result[path.name] = os.readlink(path)
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return result


class RKNNBackend:
    """Load exactly the eleven hashes from the single frozen RK3566 catalog."""

    def __init__(self, catalog_path: Path, package_root: Path) -> None:
        from rknnlite.api import RKNNLite

        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if (catalog.get("schema_version") != 23 or catalog.get("experiment_id") != ACTION_ID
                or catalog.get("job_count") != 11 or catalog.get("status") != "OFFLINE_FP16_COMPILED_RUNTIME_QUALIFICATION_PENDING"
                or catalog.get("target") != TARGET or catalog.get("precision") != "fp16"):
            raise V23RuntimeError("compiled catalog is not the frozen V23 RK3566 catalog")
        self.base: dict[tuple[int, int], tuple[Any, Mapping[str, Any]]] = {}
        self.boundary: dict[str, tuple[Any, Mapping[str, Any]]] = {}
        self.graphs = []
        self.npu_usage_evidence: list[dict[str, Any]] = []
        try:
            for row in catalog["jobs"]:
                filename = Path(str(row["path"])).name
                path = package_root / str(row["graph_id"]) / filename
                if path.stat().st_size != int(row["bytes"]) or base.sha256_file(path) != row["sha256"]:
                    raise V23RuntimeError(f"compiled graph identity changed: {row['graph_id']}")
                runtime = RKNNLite(verbose=False)
                if runtime.load_rknn(str(path)) != 0:
                    runtime.release()
                    raise V23RuntimeError(f"graph load failed: {row['graph_id']}")
                # RK3566 has no RK3576 dual-core mask contract. Omitting core_mask
                # is intentional and is bound by the V23 protocol.
                if runtime.init_runtime() != 0:
                    runtime.release()
                    raise V23RuntimeError(f"runtime init failed: {row['graph_id']}")
                record = (runtime, row)
                if row["variant"] is None:
                    self.base[(int(row["core_width"]), int(row["core_height"]))] = record
                else:
                    self.boundary[str(row["variant"])] = record
                self.graphs.append({"graph_id": row["graph_id"], "bytes": row["bytes"],
                                    "sha256": row["sha256"], "input_shape_nhwc": row["input_shape_nhwc"],
                                    "output_shape_nchw": row["output_shape_nchw"]})
            if len(self.base) != 8 or len(self.boundary) != 3:
                raise V23RuntimeError("eleven-graph base/boundary partition changed")
            self.supported_shapes = frozenset(self.base)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _infer(record: tuple[Any, Mapping[str, Any]], value: Any) -> Any:
        runtime, row = record
        if list(value.shape) != list(row["input_shape_nhwc"]):
            raise V23RuntimeError(f"input shape changed for {row['graph_id']}: {value.shape}")
        before = fd_targets()
        outputs = runtime.inference(inputs=[value[None, ...]], data_format=["nhwc"])
        after = fd_targets()
        observed = sorted({target for target in (*before.values(), *after.values())
                           if "renderD129" in target})
        if not observed:
            raise V23RuntimeError("inference produced no authenticated RKNN/RKNPU render-node evidence")
        output = __import__("numpy").asarray(outputs[0]) if outputs else None
        if output is None or list(output.shape) != list(row["output_shape_nchw"]):
            raise V23RuntimeError(f"output shape changed for {row['graph_id']}")
        # The caller attaches this evidence to the backend after successful use.
        return output, {"render_nodes": observed, "fd_count_before": len(before), "fd_count_after": len(after)}

    def infer(self, shape: tuple[int, int], value: Any) -> Any:
        output, evidence = self._infer(self.base[shape], value)
        self.npu_usage_evidence.append(evidence)
        return output

    def infer_boundary(self, variant: str, shape: tuple[int, int], value: Any) -> Any:
        if shape != (640, 60):
            raise V23RuntimeError("boundary core shape changed")
        output, evidence = self._infer(self.boundary[variant], value)
        self.npu_usage_evidence.append(evidence)
        return output

    def warmup(self, iterations: int) -> None:
        if iterations != 2:
            raise V23RuntimeError("exactly two warmups per graph required")
        records = [self.base[key] for key in sorted(self.base)] + [self.boundary[key] for key in sorted(self.boundary)]
        import numpy as np
        for runtime_record in records:
            sample = np.zeros(tuple(runtime_record[1]["input_shape_nhwc"]), dtype=np.uint8)
            for _ in range(iterations):
                self._infer(runtime_record, sample)

    def close(self) -> None:
        for runtime, _ in list(getattr(self, "base", {}).values()) + list(getattr(self, "boundary", {}).values()):
            runtime.release()
        getattr(self, "base", {}).clear()
        getattr(self, "boundary", {}).clear()


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
            value["status"] = "V23_RK3566_OFFLINE_ANALYSIS_COMPLETE"
        else:
            start = board_facts()
            if args.command == "measure":
                if args.qualification is None or not args.qualification.is_file():
                    raise V23RuntimeError("measurement requires the separate qualification manifest")
                qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
                if qualification.get("status") != "V23_RK3566_NUMERICAL_QUALIFICATION_PASS":
                    raise V23RuntimeError("measurement is blocked until RK3566 qualification passes")
            contract = load_contract(args.contract)
            backend = RKNNBackend(args.catalog, args.package_root)
            value = base.qualify(backend, contract, args.reference) if args.command == "qualify" else base.measure(backend, contract, args.schedule)
            if not backend.npu_usage_evidence:
                raise V23RuntimeError("no NPU usage evidence was retained")
            value["status"] = ("V23_RK3566_NUMERICAL_QUALIFICATION_PASS" if args.command == "qualify"
                                else "V23_RK3566_PHYSICAL_CAPTURE_COMPLETE")
            value["npu_usage_evidence"] = backend.npu_usage_evidence[:16]
            value["board_start"] = start
            value["board_end"] = board_facts()
            value["graphs"] = backend.graphs
        value.update({"schema_version": 23, "experiment_id": ACTION_ID,
                      "physical_execution_measured": args.command == "measure"})
        write_atomic(args.output, value)
        print(value["status"])
        return 0
    except BaseException as error:
        failure = {"schema_version": 23, "experiment_id": ACTION_ID,
                   "status": f"V23_{args.command.upper()}_TERMINAL_FAILURE",
                   "failure": {"type": type(error).__name__, "message": str(error)},
                   "board_start": start, "physical_execution_measured": args.command == "measure"}
        if not args.output.exists():
            write_atomic(args.output, failure)
        print(failure["status"] + ": " + str(error))
        return 2
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
