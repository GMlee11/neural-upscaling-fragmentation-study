"""Qualify and measure the frozen Paper 3 V22 graph on RK3576."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.deployment.benchmark_rk3576_sparse import resize_bicubic
from src.deployment.paper3_fragmentation_fixture_v2 import (
    build_exact_worklist,
    generate_progressive_fixture,
    load_frozen_masks,
)
from src.deployment.paper3_progressive_zero_padding_v2 import run_sparse_frame_zero_padded


ACTION_ID = "paper3_shared_v2_fp16_rk3576_fragmentation_v22"
HALO = 7
CHECKPOINT_SHA256 = "f78730284e8199ca93348a5486f059eb7b2bd6e6b6389c96f135ff0daec7ef5a"
FIXTURE_SHA256 = "88854a9e2b5b670aa89ab09548b4d11b43a4f1c9c628b4f21a3f16177fe3a66f"
MASK_ROWS_SHA256 = "b605fd865109eb943e8a7dfb60ed77b469d9c7190bac91ee8a81d66cb86992c0"


class V22RuntimeError(RuntimeError):
    """A frozen target-side gate failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")


def write_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json(value)); os.replace(temporary, path)


def temperature_c() -> float | None:
    values = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            values.append(float(path.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(values) if values else None


def board_facts() -> dict[str, Any]:
    compatible = Path("/proc/device-tree/compatible").read_bytes().replace(b"\x00", b",").decode("ascii", "replace")
    try:
        runtime_version = importlib.metadata.version("rknn-toolkit-lite2")
    except importlib.metadata.PackageNotFoundError:
        runtime_version = None
    return {"hostname": platform.node(), "machine": platform.machine(), "kernel": platform.release(),
        "compatible": compatible, "python": platform.python_version(), "rknn_toolkit_lite2": runtime_version,
        "temperature_c": temperature_c()}


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["mask_contract"]["canonical_rows_sha256"] != MASK_ROWS_SHA256:
        raise V22RuntimeError("mask contract identity changed")
    value["workload"]["halo_pixels"] = HALO
    return value


class RKNNBackend:
    """Load exactly the eleven hashes produced by the frozen V22 build."""

    def __init__(self, catalog_path: Path, package_root: Path) -> None:
        from rknnlite.api import RKNNLite

        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if (catalog.get("schema_version") != 22 or catalog.get("job_count") != 11
                or catalog.get("status") != "OFFLINE_FP16_COMPILED_RUNTIME_QUALIFICATION_PENDING"
                or catalog.get("target") != "rk3576" or catalog.get("precision") != "fp16"):
            raise V22RuntimeError("compiled catalog is not the frozen V22 catalog")
        self.base: dict[tuple[int, int], tuple[Any, Mapping[str, Any]]] = {}
        self.boundary: dict[str, tuple[Any, Mapping[str, Any]]] = {}
        self.graphs = []
        try:
            for row in catalog["jobs"]:
                filename = Path(str(row["path"])).name
                path = package_root / str(row["graph_id"]) / filename
                if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
                    raise V22RuntimeError(f"compiled graph identity changed: {row['graph_id']}")
                runtime = RKNNLite(verbose=False)
                if runtime.load_rknn(str(path)) != 0:
                    runtime.release(); raise V22RuntimeError(f"graph load failed: {row['graph_id']}")
                if runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1) != 0:
                    runtime.release(); raise V22RuntimeError(f"runtime init failed: {row['graph_id']}")
                record = (runtime, row)
                if row["variant"] is None:
                    self.base[(int(row["core_width"]), int(row["core_height"]))] = record
                else:
                    self.boundary[str(row["variant"])] = record
                self.graphs.append({"graph_id": row["graph_id"], "bytes": row["bytes"], "sha256": row["sha256"],
                    "input_shape_nhwc": row["input_shape_nhwc"], "output_shape_nchw": row["output_shape_nchw"]})
            if len(self.base) != 8 or len(self.boundary) != 3:
                raise V22RuntimeError("eleven-graph base/boundary partition changed")
            self.supported_shapes = frozenset(self.base)
        except BaseException:
            self.close(); raise

    @staticmethod
    def _infer(record: tuple[Any, Mapping[str, Any]], value: np.ndarray) -> np.ndarray:
        runtime, row = record
        if list(value.shape) != list(row["input_shape_nhwc"]):
            raise V22RuntimeError(f"input shape changed for {row['graph_id']}: {value.shape}")
        outputs = runtime.inference(inputs=[value[None, ...]], data_format=["nhwc"])
        if not outputs:
            raise V22RuntimeError(f"no output for {row['graph_id']}")
        output = np.asarray(outputs[0])
        if list(output.shape) != list(row["output_shape_nchw"]):
            raise V22RuntimeError(f"output shape changed for {row['graph_id']}: {output.shape}")
        return output

    def infer(self, shape: tuple[int, int], value: np.ndarray) -> np.ndarray:
        return self._infer(self.base[shape], value)

    def infer_boundary(self, variant: str, shape: tuple[int, int], value: np.ndarray) -> np.ndarray:
        if shape != (640, 60):
            raise V22RuntimeError("boundary core shape changed")
        return self._infer(self.boundary[variant], value)

    def warmup(self, iterations: int) -> None:
        if iterations != 2:
            raise V22RuntimeError("exactly two warmups per graph required")
        records = [self.base[key] for key in sorted(self.base)] + [self.boundary[key] for key in sorted(self.boundary)]
        for record in records:
            sample = np.zeros(tuple(record[1]["input_shape_nhwc"]), dtype=np.uint8)
            for _ in range(iterations):
                self._infer(record, sample)

    def close(self) -> None:
        for runtime, _ in list(getattr(self, "base", {}).values()) + list(getattr(self, "boundary", {}).values()):
            runtime.release()
        getattr(self, "base", {}).clear(); getattr(self, "boundary", {}).clear()


def selected(array: np.ndarray, mask: Any) -> np.ndarray:
    values = []
    for index in np.flatnonzero(mask.grid.reshape(-1)):
        row, column = divmod(int(index), 10)
        values.append(array[row * 120:(row + 1) * 120, column * 128:(column + 1) * 128].reshape(-1, 3))
    return np.ascontiguousarray(np.concatenate(values)).astype("<f4", copy=False)


def selected_mask(mask: Any) -> np.ndarray:
    value = np.zeros((720, 1280), dtype=bool)
    for index in np.flatnonzero(mask.grid.reshape(-1)):
        row, column = divmod(int(index), 10)
        value[row * 120:(row + 1) * 120, column * 128:(column + 1) * 128] = True
    return value


def psnr(left: np.ndarray, right: np.ndarray) -> float:
    delta = left.astype(np.float64) - right.astype(np.float64)
    mse = float(np.mean(delta * delta))
    return math.inf if mse == 0 else 10 * math.log10(1 / mse)


def qualify(backend: RKNNBackend, contract: Mapping[str, Any], reference_path: Path) -> dict[str, Any]:
    fixture = generate_progressive_fixture()
    if hashlib.sha256(fixture.tobytes()).hexdigest() != FIXTURE_SHA256:
        raise V22RuntimeError("fixture identity changed")
    baseline = np.clip(resize_bicubic(fixture), 0, 1); references = np.load(reference_path, allow_pickle=False)
    records = []
    try:
        for mask in load_frozen_masks(contract):
            outputs = {}
            for mode in ("exact", "row_spans"):
                checksums = []
                for repeat in range(2):
                    output, _ = run_sparse_frame_zero_padded(fixture,
                        build_exact_worklist(mask, frame_id=repeat + 1, contract=contract), backend,
                        halo=HALO, dispatch_mode=mode, supported_shapes=backend.supported_shapes)
                    checksums.append(hashlib.sha256(selected(output, mask).tobytes()).hexdigest())
                    if repeat == 0: outputs[mode] = output
                if len(set(checksums)) != 1:
                    raise V22RuntimeError(f"repeat instability: {mask.mask_id}/{mode}")
                expected = references[f"{mask.mask_id}__{mode}"]
                mae = float(np.mean(np.abs(selected(outputs[mode], mask).astype(np.float64) - expected.astype(np.float64))))
                change = float(np.max(np.abs(outputs[mode][~selected_mask(mask)] - baseline[~selected_mask(mask)])))
                if not math.isfinite(mae) or mae > .001 or change != 0:
                    raise V22RuntimeError(f"numerical gate failed: {mask.mask_id}/{mode} mae={mae} unselected={change}")
                records.append({"mask_id": mask.mask_id, "pair_id": mask.pair_id, "geometry": mask.geometry,
                    "density_percent": mask.density_percent, "mode": mode, "selected_output_mae": mae,
                    "maximum_unselected_output_change": change, "repeated_selected_sha256": checksums[0]})
            exact_psnr = psnr(selected(outputs["exact"], mask), references[f"{mask.mask_id}__exact"])
            rows_psnr = psnr(selected(outputs["row_spans"], mask), references[f"{mask.mask_id}__row_spans"])
            delta = 0 if math.isinf(exact_psnr) and math.isinf(rows_psnr) else abs(exact_psnr - rows_psnr)
            if not math.isfinite(delta) or delta > .02:
                raise V22RuntimeError(f"cross-mode PSNR gate failed: {mask.mask_id} delta={delta}")
            for row in records[-2:]: row["cross_mode_psnr_delta_db"] = delta
    finally:
        references.close()
    return {"status": "V22_RK3576_NUMERICAL_QUALIFICATION_PASS", "graph_count": 11,
        "mask_mode_count": 12, "records": records}


def measure(backend: RKNNBackend, contract: Mapping[str, Any], schedule_path: Path) -> dict[str, Any]:
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if not isinstance(schedule, list) or len(schedule) != 120:
        raise V22RuntimeError("frozen schedule must contain 120 rows")
    masks = {mask.mask_id: mask for mask in load_frozen_masks(contract)}
    fixture = generate_progressive_fixture(); backend.warmup(2)
    samples = []; blocks = []; current_block = 0
    for row in schedule:
        block = int(row["block_index"])
        if block != current_block:
            current_block = block; blocks.append({"block_index": block, "start_temperature_c": temperature_c()})
        mask = masks[row["mask_id"]]
        output, metrics = run_sparse_frame_zero_padded(fixture,
            build_exact_worklist(mask, frame_id=int(row["execution_order"]), contract=contract), backend,
            halo=HALO, dispatch_mode=row["dispatch_mode"], supported_shapes=backend.supported_shapes)
        samples.append({**row, "npu_calls": int(metrics["npu_calls"]),
            "planning_ms": float(metrics["dispatch_planning_ms"]), "extraction_ms": float(metrics["halo_extraction_ms"]),
            "npu_inference_ms": float(metrics["npu_ms"]), "reconstruction_ms": float(metrics["reconstruction_ms"]),
            "service_ms": float(metrics["compute_pipeline_ms"]), "input_tensor_bytes": int(metrics["npu_input_bytes"]),
            "output_tensor_bytes": int(metrics["npu_output_bytes"]), "selected_core_pixels": int(metrics["selected_core_pixels"]),
            "dispatched_core_pixels": int(metrics["dispatched_core_pixels"]),
            "selected_output_sha256": hashlib.sha256(selected(output, mask).tobytes()).hexdigest()})
        if len(samples) % 12 == 0: blocks[-1]["end_temperature_c"] = temperature_c()
    return {"status": "V22_RK3576_PHYSICAL_CAPTURE_COMPLETE", "warmups_per_graph": 2,
        "sample_count": len(samples), "blocks": blocks, "samples": samples}


def analyze(capture: Mapping[str, Any]) -> dict[str, Any]:
    samples = list(capture["samples"]); pairs = []
    for offset in range(0, len(samples), 2):
        left, right = samples[offset:offset + 2]
        if (left["block_index"], left["pair_id"], left["dispatch_mode"]) != (right["block_index"], right["pair_id"], right["dispatch_mode"]):
            raise V22RuntimeError("measured pair adjacency changed")
        values = {left["geometry"]: left, right["geometry"]: right}
        compact, fragmented = values["compact"], values["maximally_fragmented"]
        pairs.append({"block_index": left["block_index"], "pair_id": left["pair_id"],
            "density_percent": left["density_percent"], "dispatch_mode": left["dispatch_mode"],
            "service_penalty_percent": 100 * (fragmented["service_ms"] - compact["service_ms"]) / compact["service_ms"],
            "npu_penalty_percent": 100 * (fragmented["npu_inference_ms"] - compact["npu_inference_ms"]) / compact["npu_inference_ms"],
            "compact_service_ms": compact["service_ms"], "fragmented_service_ms": fragmented["service_ms"],
            "compact_input_bytes": compact["input_tensor_bytes"], "fragmented_input_bytes": fragmented["input_tensor_bytes"],
            "compact_output_bytes": compact["output_tensor_bytes"], "fragmented_output_bytes": fragmented["output_tensor_bytes"]})
    summaries = []
    for density in (20, 40, 60):
        for mode in ("exact", "row_spans"):
            rows = [row for row in pairs if row["density_percent"] == density and row["dispatch_mode"] == mode]
            summaries.append({"density_percent": density, "dispatch_mode": mode, "paired_blocks": len(rows),
                "median_service_penalty_percent": statistics.median(row["service_penalty_percent"] for row in rows),
                "median_npu_penalty_percent": statistics.median(row["npu_penalty_percent"] for row in rows),
                "service_penalties_percent": [row["service_penalty_percent"] for row in rows],
                "input_bytes_equal_within_pair": all(row["compact_input_bytes"] == row["fragmented_input_bytes"] for row in rows),
                "median_input_byte_inflation_percent": statistics.median(100 * (row["fragmented_input_bytes"] - row["compact_input_bytes"]) / row["compact_input_bytes"] for row in rows)})
    by_key = {(row["density_percent"], row["dispatch_mode"]): row for row in summaries}
    interactions = [{"density_percent": density,
        "service_interaction_percentage_points": by_key[(density, "row_spans")]["median_service_penalty_percent"] - by_key[(density, "exact")]["median_service_penalty_percent"]}
        for density in (20, 40, 60)]
    return {"status": "V22_RK3576_OFFLINE_ANALYSIS_COMPLETE", "paired_observation_count": len(pairs),
        "summaries": summaries, "interactions": interactions,
        "replication": {"row_span_penalty_positive_all_densities": all(by_key[(d, "row_spans")]["median_service_penalty_percent"] > 0 for d in (20, 40, 60)),
            "row_span_penalty_at_least_5_percent_density_count": sum(by_key[(d, "row_spans")]["median_service_penalty_percent"] >= 5 for d in (20, 40, 60)),
            "exact_control_within_5_percent_all_densities": all(abs(by_key[(d, "exact")]["median_service_penalty_percent"]) <= 5 for d in (20, 40, 60))}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("qualify", "measure", "analyze"))
    parser.add_argument("--package-root", type=Path); parser.add_argument("--catalog", type=Path); parser.add_argument("--contract", type=Path)
    parser.add_argument("--reference", type=Path); parser.add_argument("--schedule", type=Path); parser.add_argument("--capture", type=Path); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); start = board_facts() if args.command != "analyze" else None; backend = None
    try:
        if args.output.exists(): raise FileExistsError(f"terminal output exists: {args.output}")
        if args.command == "analyze": value = analyze(json.loads(args.capture.read_text(encoding="utf-8")))
        else:
            contract = load_contract(args.contract); backend = RKNNBackend(args.catalog, args.package_root)
            value = qualify(backend, contract, args.reference) if args.command == "qualify" else measure(backend, contract, args.schedule)
            value["board_start"] = start; value["board_end"] = board_facts(); value["graphs"] = backend.graphs
        value.update({"schema_version": 22, "experiment_id": ACTION_ID, "checkpoint_sha256": CHECKPOINT_SHA256,
            "physical_execution_measured": args.command == "measure"}); write_atomic(args.output, value)
        print(value["status"]); return 0
    except BaseException as error:
        failure = {"schema_version": 22, "experiment_id": ACTION_ID, "status": f"V22_{args.command.upper()}_TERMINAL_FAILURE",
            "failure": {"type": type(error).__name__, "message": str(error)}, "board_start": start,
            "physical_execution_measured": args.command == "measure"}
        if not args.output.exists(): write_atomic(args.output, failure)
        print(failure["status"] + ": " + str(error)); return 2
    finally:
        if backend is not None: backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
