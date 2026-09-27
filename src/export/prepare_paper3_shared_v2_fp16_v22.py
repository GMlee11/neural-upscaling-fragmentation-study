"""Prepare the frozen Paper 3 shared-feature FP16 second-graph cell."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
import math
import random
import shutil
import sys
import warnings
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnx
import onnxruntime
import torch
try:
    import yaml
except ModuleNotFoundError:  # The isolated RKNN compiler path does not need YAML.
    yaml = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.convert_rknn import convert_rknn  # noqa: E402
from src.deployment.paper3_fragmentation_fixture_v2 import (  # noqa: E402
    build_exact_worklist,
    generate_progressive_fixture,
    load_frozen_masks,
)
from src.deployment.paper3_progressive_zero_padding_v2 import (  # noqa: E402
    run_sparse_frame_zero_padded,
)
from src.deployment.benchmark_rk3576_sparse import (  # noqa: E402
    normalize_subpixel_residual,
    resize_bicubic,
)
from src.models.progressive_espcn_v2 import SharedProgressiveESPCNV2  # noqa: E402


CHECKPOINT_SHA256 = "f78730284e8199ca93348a5486f059eb7b2bd6e6b6389c96f135ff0daec7ef5a"
MASK_ROWS_SHA256 = "b605fd865109eb943e8a7dfb60ed77b469d9c7190bac91ee8a81d66cb86992c0"
FIXTURE_SHA256 = "88854a9e2b5b670aa89ab09548b4d11b43a4f1c9c628b4f21a3f16177fe3a66f"
HALO = 7
BASE_SHAPES = ((64, 60), (640, 60), (576, 60), (256, 60), (640, 120), (640, 180), (512, 60), (128, 60))
BOUNDARY_SHAPES = {
    "full_width_top": (640, 67),
    "full_width_middle": (640, 74),
    "full_width_bottom": (640, 67),
}


class V22GateError(RuntimeError):
    """A frozen preparation or qualification prerequisite failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")


def portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def protocol_contract(anchor_path: Path) -> dict[str, Any]:
    if yaml is None:
        raise V22GateError("PyYAML is required for reference and schedule preparation")
    value = yaml.safe_load(anchor_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V22GateError("anchor contract root is not a mapping")
    value = copy.deepcopy(value)
    if value["mask_contract"]["canonical_rows_sha256"] != MASK_ROWS_SHA256:
        raise V22GateError("mask identity changed")
    value["workload"]["halo_pixels"] = HALO
    return value


class SharedV2StageCPrefix(torch.nn.Module):
    """Return the direct Stage-C LR subpixel residual before PixelShuffle."""

    def __init__(self, model: SharedProgressiveESPCNV2) -> None:
        super().__init__()
        if model.exit_mode != "direct" or tuple(model.blocks_per_stage) != (1, 1, 3):
            raise V22GateError("checkpoint topology differs from the frozen candidate")
        self.stem = model.stem
        self.stage_blocks = model.stage_blocks
        self.head = model.exit_heads[2]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.stem(value))
        for stage in self.stage_blocks:
            for block in stage:
                features = block(features)
        return self.head(features)


def load_prefix(checkpoint_path: Path) -> SharedV2StageCPrefix:
    from src.train.train_progressive_espcn_v2 import load_progressive_v2_checkpoint

    if sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
        raise V22GateError("checkpoint identity changed")
    model, _ = load_progressive_v2_checkpoint(checkpoint_path, torch.device("cpu"))
    return SharedV2StageCPrefix(model.eval()).eval()


def export_one(module: torch.nn.Module, path: Path, width: int, height: int, seed: int) -> dict[str, Any]:
    shape = (1, 3, height, width)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(module, torch.zeros(shape), path, input_names=["lr_input"],
            output_names=["subpixel_residual"], opset_version=17, dynamo=False,
            external_data=False, export_params=True, do_constant_folding=True)
    graph = onnx.load(str(path)); onnx.checker.check_model(graph)
    operators = [node.op_type for node in graph.graph.node]
    if operators.count("Conv") != 7 or operators.count("Relu") != 6 or operators.count("Add") != 4:
        raise V22GateError(f"unexpected operator topology: {operators}")
    sample = np.random.default_rng(seed).random(shape, dtype=np.float32)
    with torch.inference_mode():
        expected = module(torch.from_numpy(sample)).numpy()
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
    maximum = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    if tuple(actual.shape) != (1, 12, height, width) or not np.allclose(actual, expected, rtol=1e-5, atol=2e-6):
        raise V22GateError(f"PyTorch/ONNX agreement failed for {width}x{height}: {maximum}")
    return {"path": portable(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "input_shape_nchw": list(shape), "output_shape_nchw": [1, 12, height, width],
        "operators": {"Conv": 7, "Relu": 6, "Add": 4}, "maximum_pytorch_onnx_error": maximum}


def export_catalog(checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary output: {temporary}")
    module = load_prefix(checkpoint_path)
    temporary.mkdir(parents=True)
    try:
        base = []
        for index, (width, height) in enumerate(BASE_SHAPES):
            input_width, input_height = width + 2 * HALO, height + 2 * HALO
            relative = Path("base") / f"core_{width:04d}x{height:04d}.onnx"
            record = export_one(module, temporary / relative, input_width, input_height, 2026082400 + index)
            record["path"] = portable(output_dir / relative)
            base.append({"graph_id": f"core_{width:04d}x{height:04d}", "core_width": width,
                "core_height": height, "halo": HALO, "variant": None, **record})
        boundary = []
        for index, (variant, (width, height)) in enumerate(BOUNDARY_SHAPES.items()):
            relative = Path("boundary") / f"{variant}.onnx"
            record = export_one(module, temporary / relative, width, height, 2026082500 + index)
            record["path"] = portable(output_dir / relative)
            boundary.append({"graph_id": variant, "core_width": 640, "core_height": 60,
                "halo": HALO, "variant": variant, **record})
        catalog = {"schema_version": 22, "artifact": "paper3_shared_v2_fp16_onnx_catalog_v22",
            "status": "OFFLINE_ONNX_AND_FP32_REFERENCE_READY_RKNN_COMPILATION_PENDING",
            "checkpoint": {"path": portable(checkpoint_path), "sha256": CHECKPOINT_SHA256},
            "graph": {"architecture": "shared_progressive_espcn_v2", "exit_mode": "direct",
                "stage": "C", "convolutions": 7, "residual_feature_additions": 4, "halo": HALO,
                "precision": "fp16"}, "base_graphs": base, "boundary_graphs": boundary,
            "physical_execution_measured": False}
        (temporary / "catalog.json").write_bytes(canonical_json(catalog))
        temporary.replace(output_dir)
        return catalog
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


class OnnxBackend:
    def __init__(self, catalog_path: Path) -> None:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.base: dict[tuple[int, int], onnxruntime.InferenceSession] = {}
        self.boundary: dict[str, onnxruntime.InferenceSession] = {}
        for row in catalog["base_graphs"] + catalog["boundary_graphs"]:
            path = ROOT / row["path"]
            if sha256_file(path) != row["sha256"]:
                raise V22GateError("ONNX catalog identity changed")
            session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            if row["variant"] is None:
                self.base[(int(row["core_width"]), int(row["core_height"]))] = session
            else:
                self.boundary[str(row["variant"])] = session
        self.supported_shapes = frozenset(self.base)

    @staticmethod
    def _run(session: onnxruntime.InferenceSession, value: np.ndarray) -> np.ndarray:
        tensor = np.transpose(value.astype(np.float32) / 255.0, (2, 0, 1))[None]
        return np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0])

    def infer(self, shape: tuple[int, int], value: np.ndarray) -> np.ndarray:
        return self._run(self.base[shape], value)

    def infer_boundary(self, variant: str, shape: tuple[int, int], value: np.ndarray) -> np.ndarray:
        if shape != (640, 60):
            raise V22GateError("unexpected boundary core shape")
        return self._run(self.boundary[variant], value)

    def close(self) -> None:
        self.base.clear(); self.boundary.clear()


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
    error = left.astype(np.float64) - right.astype(np.float64)
    mse = float(np.mean(error * error))
    return math.inf if mse == 0 else 10 * math.log10(1 / mse)


def deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO(); np.lib.format.write_array(payload, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100444 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def full_reference(checkpoint_path: Path, fixture: np.ndarray) -> np.ndarray:
    module = load_prefix(checkpoint_path)
    value = np.transpose(fixture.astype(np.float32) / 255.0, (2, 0, 1))[None]
    with torch.inference_mode():
        raw = module(torch.from_numpy(value)).numpy()
    residual = normalize_subpixel_residual(raw, input_width=640, input_height=360,
        quantization_scale=None, quantization_zero_point=None)
    return np.clip(resize_bicubic(fixture) + residual, 0, 1)


def generate_reference(catalog_path: Path, checkpoint_path: Path, anchor_path: Path) -> tuple[dict[str, Any], bytes]:
    contract = protocol_contract(anchor_path); masks = load_frozen_masks(contract)
    fixture = generate_progressive_fixture()
    if hashlib.sha256(fixture.tobytes()).hexdigest() != FIXTURE_SHA256:
        raise V22GateError("fixture identity changed")
    full = full_reference(checkpoint_path, fixture); baseline = np.clip(resize_bicubic(fixture), 0, 1)
    backend = OnnxBackend(catalog_path); arrays: dict[str, np.ndarray] = {}; records = []
    try:
        for frame_id, mask in enumerate(masks, 1):
            worklist = build_exact_worklist(mask, frame_id=frame_id, contract=contract)
            outputs = {}
            for mode in ("exact", "row_spans"):
                output, metrics = run_sparse_frame_zero_padded(fixture, worklist, backend, halo=HALO,
                    dispatch_mode=mode, supported_shapes=backend.supported_shapes)
                outputs[mode] = output; arrays[f"{mask.mask_id}__{mode}"] = selected(output, mask)
            truth = selected(full, mask); exact = arrays[f"{mask.mask_id}__exact"]; rows = arrays[f"{mask.mask_id}__row_spans"]
            exact_mae = float(np.mean(np.abs(exact.astype(np.float64) - truth.astype(np.float64))))
            rows_mae = float(np.mean(np.abs(rows.astype(np.float64) - truth.astype(np.float64))))
            exact_psnr, rows_psnr = psnr(exact, truth), psnr(rows, truth)
            delta = 0.0 if math.isinf(exact_psnr) and math.isinf(rows_psnr) else abs(exact_psnr - rows_psnr)
            unselected = max(float(np.max(np.abs(outputs[mode][~selected_mask(mask)] - baseline[~selected_mask(mask)]))) for mode in outputs)
            if exact_mae > .001 or rows_mae > .001 or delta > .02 or unselected != 0:
                raise V22GateError(f"offline integrity failed for {mask.mask_id}: {exact_mae}, {rows_mae}, {delta}, {unselected}")
            records.append({"mask_id": mask.mask_id, "pair_id": mask.pair_id, "geometry": mask.geometry,
                "density_percent": mask.density_percent, "selected_output_mae": {"exact": exact_mae, "row_spans": rows_mae},
                "selected_output_psnr_db": {"exact": exact_psnr, "row_spans": rows_psnr},
                "cross_mode_psnr_delta_db": delta, "maximum_unselected_output_change": unselected})
    finally:
        backend.close()
    payload = deterministic_npz(arrays)
    result = {"schema_version": 22, "artifact": "paper3_shared_v2_fp16_offline_reference_v22",
        "status": "OFFLINE_FP32_REFERENCE_PASS_RKNN_QUALIFICATION_PENDING", "checkpoint_sha256": CHECKPOINT_SHA256,
        "catalog_sha256": sha256_file(catalog_path), "fixture_sha256": FIXTURE_SHA256,
        "mask_rows_sha256": MASK_ROWS_SHA256, "halo": HALO,
        "payload": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "array_count": len(arrays)},
        "thresholds": {"selected_output_mae_max": .001, "cross_mode_psnr_delta_db_max": .02,
            "unselected_output_change": 0.0}, "records": records, "physical_execution_measured": False}
    return result, payload


def build_schedule(anchor_path: Path) -> list[dict[str, Any]]:
    contract = protocol_contract(anchor_path); masks = load_frozen_masks(contract)
    pairs: dict[str, dict[str, Any]] = {}
    for mask in masks:
        pairs.setdefault(mask.pair_id, {})[mask.geometry] = mask
    pair_ids = sorted(pairs, key=lambda value: int(value.rsplit("_", 1)[1]))
    randomizer = random.Random(357622); rows = []
    for block in range(1, 11):
        cells = [(index, pair, mode_index, mode) for index, pair in enumerate(pair_ids)
            for mode_index, mode in enumerate(("exact", "row_spans"))]
        randomizer.shuffle(cells)
        for pair_index, pair_id, mode_index, mode in cells:
            compact_first = (block + pair_index + mode_index) % 2 == 0
            order = (pairs[pair_id]["compact"], pairs[pair_id]["maximally_fragmented"])
            if not compact_first: order = tuple(reversed(order))
            for mask in order:
                rows.append({"block_index": block, "execution_order": len(rows) + 1, "pair_id": pair_id,
                    "dispatch_mode": mode, "mask_id": mask.mask_id, "geometry": mask.geometry,
                    "density_percent": mask.density_percent})
    if len(rows) != 120:
        raise V22GateError("schedule count changed")
    return rows


def compile_fp16(catalog_path: Path, output_root: Path, calibration_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"compilation root exists: {output_root}")
    if importlib.metadata.version("rknn-toolkit2") != "2.3.2":
        raise V22GateError("exact rknn-toolkit2 2.3.2 is required")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")); output_root.mkdir(parents=True)
    jobs = []
    try:
        for row in catalog["base_graphs"] + catalog["boundary_graphs"]:
            onnx_path = ROOT / row["path"]
            destination = output_root / row["graph_id"]
            manifest = convert_rknn(onnx_path, calibration_root, destination,
                graph="subpixel_residual", precision="fp16", artifact_name=f"paper3_shared_v2_v22_{row['graph_id']}",
                overwrite=False, verbose=False, input_width=int(row["input_shape_nchw"][3]),
                input_height=int(row["input_shape_nchw"][2]), target_platform="rk3576")
            jobs.append({"graph_id": row["graph_id"], "variant": row["variant"],
                "core_width": row["core_width"], "core_height": row["core_height"],
                "input_shape_nhwc": [int(row["input_shape_nchw"][2]), int(row["input_shape_nchw"][3]), 3],
                "output_shape_nchw": row["output_shape_nchw"], "onnx_sha256": row["sha256"],
                "path": str((destination / manifest["rknn"]["path"]).resolve()),
                "bytes": manifest["rknn"]["size_bytes"], "sha256": manifest["rknn"]["sha256"]})
        result = {"schema_version": 22, "artifact": "paper3_shared_v2_fp16_rknn_catalog_v22",
            "status": "OFFLINE_FP16_COMPILED_RUNTIME_QUALIFICATION_PENDING", "target": "rk3576",
            "precision": "fp16", "compiler": {"name": "rknn-toolkit2", "version": "2.3.2"},
            "onnx_catalog_sha256": sha256_file(catalog_path), "job_count": len(jobs), "jobs": jobs,
            "physical_execution_measured": False}
        if len(jobs) != 11:
            raise V22GateError("exactly eleven compiled graphs required")
        (output_root / "catalog.json").write_bytes(canonical_json(result)); return result
    except BaseException:
        (output_root / "FAILED.json").write_bytes(canonical_json({"status": "V22_COMPILATION_FAILED", "completed_jobs": len(jobs)}))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export"); export.add_argument("--checkpoint", type=Path, required=True); export.add_argument("--output-dir", type=Path, required=True)
    reference = sub.add_parser("reference"); reference.add_argument("--catalog", type=Path, required=True); reference.add_argument("--checkpoint", type=Path, required=True); reference.add_argument("--anchor", type=Path, required=True); reference.add_argument("--json-output", type=Path, required=True); reference.add_argument("--payload-output", type=Path, required=True); reference.add_argument("--schedule-output", type=Path, required=True)
    compile_cmd = sub.add_parser("compile"); compile_cmd.add_argument("--catalog", type=Path, required=True); compile_cmd.add_argument("--output-root", type=Path, required=True); compile_cmd.add_argument("--calibration-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            value = export_catalog(args.checkpoint, args.output_dir); print(f"V22_ONNX_READY graphs={len(value['base_graphs']) + len(value['boundary_graphs'])}")
        elif args.command == "reference":
            if any(path.exists() for path in (args.json_output, args.payload_output, args.schedule_output)):
                raise FileExistsError("reference or schedule output already exists")
            value, payload = generate_reference(args.catalog, args.checkpoint, args.anchor)
            for path in (args.json_output, args.payload_output, args.schedule_output): path.parent.mkdir(parents=True, exist_ok=True)
            args.payload_output.write_bytes(payload); value["payload"]["path"] = portable(args.payload_output)
            args.json_output.write_bytes(canonical_json(value)); args.schedule_output.write_bytes(canonical_json(build_schedule(args.anchor)))
            print("V22_OFFLINE_REFERENCE_PASS masks=6 schedule=120")
        else:
            value = compile_fp16(args.catalog, args.output_root, args.calibration_root); print(f"V22_RKNN_FP16_READY graphs={value['job_count']}")
    except (FileNotFoundError, FileExistsError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
