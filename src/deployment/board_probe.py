"""Record reproducibility facts from the RK3576 deployment board."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence


# Files in sysfs and procfs vary across board images. Missing optional facts are
# represented as null instead of guessed, keeping provisional results honest.
def read_optional_text(path: Path) -> str | None:
    """Read a small system-information file or return null when unavailable."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except (OSError, UnicodeError):
        return None


# Linux thermal zones report millidegrees Celsius. Retain each named sensor so
# later reports can identify whether a value came from the SoC, GPU, or NPU.
def read_thermal_zones(sys_root: Path = Path("/sys")) -> list[dict[str, object]]:
    """Return available thermal-zone names and temperatures in Celsius."""
    zones: list[dict[str, object]] = []
    for zone_path in sorted((sys_root / "class/thermal").glob("thermal_zone*")):
        sensor_type = read_optional_text(zone_path / "type") or zone_path.name
        raw_temperature = read_optional_text(zone_path / "temp")
        try:
            temperature_c = float(raw_temperature) / 1000.0
        except (TypeError, ValueError):
            temperature_c = None
        zones.append({"name": sensor_type, "temperature_c": temperature_c})
    return zones


# Package versions are queried independently because CPU, GPU, and NPU
# runtimes are optional and may be installed in different board environments.
def package_version(distribution_names: Sequence[str]) -> str | None:
    """Return the first installed distribution version from a candidate list."""
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def default_command_runner(command: Sequence[str]) -> str | None:
    """Run a short diagnostic command and return stdout when it succeeds."""
    try:
        completed = subprocess.run(
            list(command), capture_output=True, check=False, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


# Gather facts without changing clocks, governors, cooling, or power settings.
# Those conditions remain user-controlled and are supplied as explicit labels.
def collect_board_facts(
    *,
    proc_root: Path = Path("/proc"),
    sys_root: Path = Path("/sys"),
    dev_root: Path = Path("/dev"),
    run_command: Callable[[Sequence[str]], str | None] = default_command_runner,
) -> dict[str, object]:
    """Collect board identity, runtimes, devices, thermal data, and GPU facts."""
    compatible = read_optional_text(proc_root / "device-tree/compatible")
    os_release = read_optional_text(Path("/etc/os-release"))
    device_candidates = ("rknpu", "dri/card0", "dri/renderD128")
    devices = [name for name in device_candidates if (dev_root / name).exists()]
    vulkan_summary = (
        run_command(("vulkaninfo", "--summary"))
        if shutil.which("vulkaninfo")
        else None
    )
    return {
        "schema_version": 1,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "device_tree_compatible": compatible,
            "os_release": os_release,
        },
        "runtime_versions": {
            "onnxruntime": package_version(("onnxruntime",)),
            "rknn_lite2": package_version(("rknn-toolkit-lite2", "rknn_toolkit_lite2")),
            "ncnn": package_version(("ncnn",)),
        },
        "devices": devices,
        "vulkan_summary": vulkan_summary,
        "thermal_zones": read_thermal_zones(sys_root),
    }


def write_json_atomic(output_path: Path, payload: object, overwrite: bool) -> None:
    """Atomically write one JSON artifact without silently replacing results."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    """Build the non-destructive board-probe command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Probe the current board and save a reproducibility record."""
    args = build_parser().parse_args(argv)
    try:
        facts = collect_board_facts()
        write_json_atomic(args.output.resolve(), facts, args.overwrite)
    except (FileExistsError, OSError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"board probe: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
