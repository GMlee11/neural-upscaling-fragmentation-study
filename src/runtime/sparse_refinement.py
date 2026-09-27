"""Portable renderer-guided sparse-refinement selection and work-list packing.

This runtime module deliberately has no image decoder, training framework, HR
reference, dataset identity, or scene identity interface. A renderer supplies
bounded macro-tile summaries, the frozen selector predicts neural utility, and
the scheduler emits compact halo-aware descriptors that an NPU or FPGA backend
can consume without rescanning the rendered image on the CPU.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


AA_SAMPLES = {
    "disabled": 0,
    "fxaa": 1,
    "msaa_2x": 2,
    "msaa_4x": 4,
    "msaa_8x": 8,
}
SEMANTIC_FIELDS = (
    "thin_geometry_fraction",
    "repeated_patterns_fraction",
    "foliage_fraction",
    "hud_text_fraction",
    "particles_fraction",
    "fences_fraction",
    "reflective_surfaces_fraction",
)
TILE_FEATURE_NAMES = (
    "tile_center_x_normalized",
    "tile_center_y_normalized",
    "camera_fov_normalized",
    "camera_pitch_normalized",
    "antialiasing_samples_normalized",
    "camera_translation_delta_normalized",
    "camera_rotation_delta_normalized",
    "camera_motion_available",
    *SEMANTIC_FIELDS,
    "dynamic_object_fraction",
    "disocclusion_fraction",
    "dynamic_object_available",
    "disocclusion_available",
    "luma_mean",
    "luma_std",
    "gradient_mean",
    "gradient_p90",
    "edge_fraction",
    "temporal_change_mean",
    "temporal_change_p90",
    "temporal_changed_fraction",
    "temporal_summary_available",
)

WORKLIST_MAGIC = b"SRWL"
WORKLIST_VERSION = 1
WORKLIST_HEADER = struct.Struct("<4sHHIHHHHHHII")
WORK_DESCRIPTOR = struct.Struct("<IHHHHHHHHIfHH")
WORKLIST_HEADER_BYTES = WORKLIST_HEADER.size
WORK_DESCRIPTOR_BYTES = WORK_DESCRIPTOR.size
WORK_FLAG_PADDING_REQUIRED = 1 << 0


# Validate one optional normalized renderer value while preserving missingness.
def _optional_fraction(value: object, name: str) -> float | None:
    """Return a finite normalized fraction or ``None``."""

    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be null or a fraction in [0, 1]")
    return result


@dataclass(frozen=True)
class RendererTileSummary:
    """One identity-free macro-tile summary produced before NPU dispatch."""

    tile_center_x_normalized: float
    tile_center_y_normalized: float
    camera_fov_degrees: float
    camera_pitch_radians: float
    antialiasing_mode: str
    camera_translation_delta: float | None
    camera_rotation_delta_radians: float | None
    thin_geometry_fraction: float = 0.0
    repeated_patterns_fraction: float = 0.0
    foliage_fraction: float = 0.0
    hud_text_fraction: float = 0.0
    particles_fraction: float = 0.0
    fences_fraction: float = 0.0
    reflective_surfaces_fraction: float = 0.0
    dynamic_object_fraction: float | None = None
    disocclusion_fraction: float | None = None
    luma_mean: float = 0.0
    luma_std: float = 0.0
    gradient_mean: float = 0.0
    gradient_p90: float = 0.0
    edge_fraction: float = 0.0
    temporal_change_mean: float | None = None
    temporal_change_p90: float | None = None
    temporal_changed_fraction: float | None = None

    # Keep every deployable feature finite, bounded, and independent of identity.
    def validate(self) -> None:
        """Reject malformed renderer summaries before selector execution."""

        fractions = (
            "tile_center_x_normalized",
            "tile_center_y_normalized",
            *SEMANTIC_FIELDS,
            "luma_mean",
            "luma_std",
            "gradient_mean",
            "gradient_p90",
            "edge_fraction",
        )
        for name in fractions:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be a fraction in [0, 1]")
        for name in (
            "dynamic_object_fraction",
            "disocclusion_fraction",
            "temporal_change_mean",
            "temporal_change_p90",
            "temporal_changed_fraction",
        ):
            _optional_fraction(getattr(self, name), name)
        if not math.isfinite(self.camera_fov_degrees) or not 1.0 <= self.camera_fov_degrees <= 179.0:
            raise ValueError("camera FOV must be finite and in [1, 179] degrees")
        if not math.isfinite(self.camera_pitch_radians) or abs(self.camera_pitch_radians) > math.pi:
            raise ValueError("camera pitch must be finite and within one rotation")
        if self.antialiasing_mode not in AA_SAMPLES:
            raise ValueError(f"unsupported antialiasing mode: {self.antialiasing_mode}")
        for name in ("camera_translation_delta", "camera_rotation_delta_radians"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be null or finite and nonnegative")

    # Convert missing measurements into value/availability pairs so missing is
    # never confused with a renderer that measured a real zero.
    def feature_vector(self) -> np.ndarray:
        """Return the frozen selector feature order as float32."""

        self.validate()
        motion_available = self.camera_translation_delta is not None and self.camera_rotation_delta_radians is not None
        temporal_available = (
            self.temporal_change_mean is not None
            and self.temporal_change_p90 is not None
            and self.temporal_changed_fraction is not None
        )
        values = np.asarray(
            [
                self.tile_center_x_normalized,
                self.tile_center_y_normalized,
                self.camera_fov_degrees / 180.0,
                abs(self.camera_pitch_radians) / math.pi,
                AA_SAMPLES[self.antialiasing_mode] / 8.0,
                min(float(self.camera_translation_delta or 0.0), 1.0),
                min(float(self.camera_rotation_delta_radians or 0.0) / math.pi, 1.0),
                float(motion_available),
                *(float(getattr(self, name)) for name in SEMANTIC_FIELDS),
                float(self.dynamic_object_fraction or 0.0),
                float(self.disocclusion_fraction or 0.0),
                float(self.dynamic_object_fraction is not None),
                float(self.disocclusion_fraction is not None),
                self.luma_mean,
                self.luma_std,
                self.gradient_mean,
                self.gradient_p90,
                self.edge_fraction,
                float(self.temporal_change_mean or 0.0),
                float(self.temporal_change_p90 or 0.0),
                float(self.temporal_changed_fraction or 0.0),
                float(temporal_available),
            ],
            dtype=np.float32,
        )
        if values.shape != (len(TILE_FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError("renderer tile summary produced an invalid feature vector")
        return values

    # Mapping construction is strict because accidentally feeding a scene ID or
    # an HR-derived target would invalidate the deployable-policy evidence.
    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "RendererTileSummary":
        """Parse an identity-free renderer protocol mapping."""

        forbidden = {
            "pair_id",
            "scene_id",
            "scene_theme",
            "scene_variant",
            "sequence_id",
            "source_id",
            "engine_id",
            "hr_pixels",
            "hr_path",
            "oracle_error",
            "oracle_label",
            "neural_utility",
        }
        leaked = sorted(forbidden.intersection(str(key) for key in payload))
        if leaked:
            raise ValueError(f"renderer tile summary contains forbidden input: {leaked}")
        required = {
            "tile_center_x_normalized",
            "tile_center_y_normalized",
            "camera_fov_degrees",
            "camera_pitch_radians",
            "antialiasing_mode",
        }
        missing = sorted(required.difference(str(key) for key in payload))
        if missing:
            raise ValueError(f"renderer tile summary is missing required fields: {missing}")
        result = cls(
            tile_center_x_normalized=float(payload["tile_center_x_normalized"]),
            tile_center_y_normalized=float(payload["tile_center_y_normalized"]),
            camera_fov_degrees=float(payload["camera_fov_degrees"]),
            camera_pitch_radians=float(payload["camera_pitch_radians"]),
            antialiasing_mode=str(payload["antialiasing_mode"]),
            camera_translation_delta=(
                None if payload.get("camera_translation_delta") is None else float(payload["camera_translation_delta"])
            ),
            camera_rotation_delta_radians=(
                None
                if payload.get("camera_rotation_delta_radians") is None
                else float(payload["camera_rotation_delta_radians"])
            ),
            **{
                name: float(payload.get(name, 0.0))
                for name in (
                    *SEMANTIC_FIELDS,
                    "luma_mean",
                    "luma_std",
                    "gradient_mean",
                    "gradient_p90",
                    "edge_fraction",
                )
            },
            **{
                name: _optional_fraction(payload.get(name), name)
                for name in (
                    "dynamic_object_fraction",
                    "disocclusion_fraction",
                    "temporal_change_mean",
                    "temporal_change_p90",
                    "temporal_changed_fraction",
                )
            },
        )
        result.validate()
        return result


@dataclass(frozen=True)
class SparseTilePrediction:
    """One neural-admission prediction made without HR information."""

    neural_probability: float
    selected: bool


class FrozenSparseTileSelector:
    """Execute a portable linear, decision-tree, or tiny-MLP tile selector."""

    def __init__(self, artifact: Mapping[str, object]) -> None:
        """Validate the frozen selector artifact and feature normalization."""

        if artifact.get("artifact") != "sparse_refinement_selector_v1":
            raise ValueError("artifact is not a sparse-refinement selector")
        if tuple(artifact.get("feature_names", ())) != TILE_FEATURE_NAMES:
            raise ValueError("sparse selector feature order changed")
        self.model_type = str(artifact.get("model_type", ""))
        if self.model_type not in {"linear_heuristic", "decision_tree", "tiny_mlp", "random_forest"}:
            raise ValueError(f"unsupported sparse selector model: {self.model_type}")
        self.feature_mean = np.asarray(artifact.get("feature_mean"), dtype=np.float64)
        self.feature_std = np.asarray(artifact.get("feature_std"), dtype=np.float64)
        if (
            self.feature_mean.shape != (len(TILE_FEATURE_NAMES),)
            or self.feature_std.shape != (len(TILE_FEATURE_NAMES),)
            or not np.isfinite(self.feature_mean).all()
            or not np.isfinite(self.feature_std).all()
            or np.any(self.feature_std <= 0.0)
        ):
            raise ValueError("sparse selector normalization is invalid")
        self.selection_threshold = float(artifact.get("selection_threshold", math.nan))
        if not math.isfinite(self.selection_threshold) or not 0.0 <= self.selection_threshold <= 1.0:
            raise ValueError("sparse selector threshold must be in [0, 1]")
        model = artifact.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("sparse selector model payload must be a mapping")
        self.model = model
        self.support_guard = self._validate_support_guard(artifact.get("support_guard"))

    # V3 optionally restricts classical fallback to feature regions represented
    # by development data. Older artifacts omit this payload and retain their
    # exact threshold-only behavior.
    def _validate_support_guard(self, payload: object) -> Mapping[str, object] | None:
        """Validate and return the optional conservative support envelope."""

        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise ValueError("support guard must be a mapping")
        method = payload.get("method")
        if method not in {"tree_leaf_feature_envelope", "global_feature_envelope"}:
            raise ValueError("unsupported sparse selector support guard")
        if method == "tree_leaf_feature_envelope" and self.model_type != "decision_tree":
            raise ValueError("leaf support guard requires a decision-tree selector")
        if payload.get("unsupported_action") != "always_neural":
            raise ValueError("unsupported tiles must fall back to neural execution")
        indices = tuple(int(value) for value in payload.get("feature_indices", ()))
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("support guard feature indices are empty or duplicated")
        if any(index < 0 or index >= len(TILE_FEATURE_NAMES) for index in indices):
            raise ValueError("support guard feature index is out of range")
        if method == "global_feature_envelope":
            lower = np.asarray(payload.get("lower"), dtype=np.float64)
            upper = np.asarray(payload.get("upper"), dtype=np.float64)
            if (
                lower.shape != (len(indices),)
                or upper.shape != lower.shape
                or not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
                or np.any(lower > upper)
                or int(payload.get("sample_count", 0)) < int(payload.get("minimum_samples", 0))
            ):
                raise ValueError("global support guard bounds are invalid")
            return payload
        leaves = payload.get("leaves")
        if not isinstance(leaves, Mapping) or not leaves:
            raise ValueError("support guard contains no leaf envelopes")
        for leaf_index, envelope in leaves.items():
            if not str(leaf_index).isdigit() or not isinstance(envelope, Mapping):
                raise ValueError("support guard leaf envelope is malformed")
            lower = np.asarray(envelope.get("lower"), dtype=np.float64)
            upper = np.asarray(envelope.get("upper"), dtype=np.float64)
            if (
                lower.shape != (len(indices),)
                or upper.shape != lower.shape
                or not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
                or np.any(lower > upper)
                or int(envelope.get("sample_count", 0)) < 0
            ):
                raise ValueError("support guard leaf bounds are invalid")
        return payload

    # Stable sigmoid avoids overflow from compact fixed artifacts with large
    # standardized logits.
    @staticmethod
    def _sigmoid(value: float) -> float:
        """Return a numerically stable scalar logistic probability."""

        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    # Execute only simple arithmetic and branches so this same artifact can be
    # reimplemented in C++ without a machine-learning runtime dependency.
    def predict_probability(self, features: Sequence[float] | np.ndarray) -> float:
        """Return the predicted probability that neural refinement is useful."""

        values = np.asarray(features, dtype=np.float64)
        if values.shape != (len(TILE_FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError("sparse selector feature row has an invalid shape or value")
        standardized = (values - self.feature_mean) / self.feature_std
        if self.model_type == "linear_heuristic":
            weights = np.asarray(self.model.get("weights"), dtype=np.float64)
            if weights.shape != standardized.shape:
                raise ValueError("linear sparse selector weights have the wrong shape")
            return self._sigmoid(float(np.dot(weights, standardized)) + float(self.model["bias"]))
        if self.model_type == "decision_tree":
            probability, _ = self._tree_probability_and_leaf(standardized)
            return probability
        if self.model_type == "random_forest":
            trees = self.model.get("trees")
            if not isinstance(trees, list) or not trees:
                raise ValueError("sparse random forest contains no trees")
            probabilities = [
                self._nodes_probability_and_leaf(tree.get("nodes"), standardized)[0]
                for tree in trees
                if isinstance(tree, Mapping)
            ]
            if len(probabilities) != len(trees):
                raise ValueError("sparse random forest tree is malformed")
            return float(np.mean(probabilities))
        value = standardized
        layers = self.model.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("sparse tiny MLP contains no layers")
        for index, layer in enumerate(layers):
            value = np.asarray(layer["weight"], dtype=np.float64) @ value + np.asarray(
                layer["bias"], dtype=np.float64
            )
            if index + 1 < len(layers):
                value = np.maximum(value, 0.0)
        if value.size != 1:
            raise ValueError("sparse tiny MLP must produce one logit")
        return self._sigmoid(float(value.reshape(-1)[0]))

    # Returning the reached leaf lets the support guard reuse the exact tree
    # traversal instead of maintaining a second, potentially drifting model.
    def _tree_probability_and_leaf(self, standardized: np.ndarray) -> tuple[float, int]:
        """Return a decision-tree probability and reached leaf index."""

        return self._nodes_probability_and_leaf(self.model.get("nodes"), standardized)

    # Forest and single-tree execution use the same branch contract. Sharing
    # this traversal prevents subtle probability differences between models.
    @staticmethod
    def _nodes_probability_and_leaf(
        nodes: object, standardized: np.ndarray
    ) -> tuple[float, int]:
        """Return probability and leaf index for one serialized node list."""

        if not isinstance(nodes, list) or not nodes:
            raise ValueError("sparse decision tree contains no nodes")
        index = 0
        while True:
            node = nodes[index]
            if bool(node["leaf"]):
                probability = min(max(float(node["positive_probability"]), 0.0), 1.0)
                return probability, index
            index = int(
                node["left"]
                if standardized[int(node["feature_index"])] <= float(node["threshold"])
                else node["right"]
            )

    # A tile is supported only when its reached leaf has enough development
    # examples and every guarded standardized feature lies inside that leaf's
    # frozen envelope. Unknown inputs deliberately use the safer neural path.
    def _is_supported(self, standardized: np.ndarray, leaf_index: int) -> bool:
        """Return whether one feature row lies inside the frozen support set."""

        if self.support_guard is None:
            return True
        if self.support_guard["method"] == "global_feature_envelope":
            indices = np.asarray(self.support_guard["feature_indices"], dtype=np.int64)
            values = standardized[indices]
            lower = np.asarray(self.support_guard["lower"], dtype=np.float64)
            upper = np.asarray(self.support_guard["upper"], dtype=np.float64)
            return bool(np.all(values >= lower) and np.all(values <= upper))
        envelope = self.support_guard["leaves"].get(str(leaf_index))
        if not isinstance(envelope, Mapping):
            return False
        if int(envelope["sample_count"]) < int(self.support_guard["minimum_leaf_samples"]):
            return False
        indices = np.asarray(self.support_guard["feature_indices"], dtype=np.int64)
        lower = np.asarray(envelope["lower"], dtype=np.float64)
        upper = np.asarray(envelope["upper"], dtype=np.float64)
        values = standardized[indices]
        return bool(np.all(values >= lower) and np.all(values <= upper))

    # This feature-vector entry point is shared by offline qualification and
    # renderer runtimes that have already produced the ordered feature row.
    def predict_features(self, features: Sequence[float] | np.ndarray) -> SparseTilePrediction:
        """Return one guarded prediction from an ordered feature vector."""

        values = np.asarray(features, dtype=np.float64)
        probability = self.predict_probability(values)
        selected = probability >= self.selection_threshold
        if self.support_guard is not None:
            standardized = (values - self.feature_mean) / self.feature_std
            leaf_index = (
                self._tree_probability_and_leaf(standardized)[1]
                if self.support_guard["method"] == "tree_leaf_feature_envelope"
                else -1
            )
            selected = selected or not self._is_supported(standardized, leaf_index)
        return SparseTilePrediction(neural_probability=probability, selected=selected)

    # Thresholding is kept outside model arithmetic so validation can freeze one
    # auditable operating point without retraining the predictor.
    def predict(self, summary: RendererTileSummary) -> SparseTilePrediction:
        """Return one probability and its frozen admission decision."""

        return self.predict_features(summary.feature_vector())


# Load a JSON artifact through the same strict checks used by live inference.
def load_sparse_tile_selector(path: Path) -> FrozenSparseTileSelector:
    """Load one portable sparse selector artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("sparse selector artifact root must be a mapping")
    return FrozenSparseTileSelector(payload)


@dataclass(frozen=True)
class SparseWorkDescriptor:
    """One compact LR core rectangle plus its padded neural input region."""

    frame_id: int
    core_x: int
    core_y: int
    core_width: int
    core_height: int
    input_x: int
    input_y: int
    input_width: int
    input_height: int
    deadline_us: int
    priority: float
    flags: int = 0

    # A fixed 32-byte layout gives C++, RTL, and Python one checksummed contract.
    def pack(self) -> bytes:
        """Serialize this descriptor in little-endian protocol order."""

        integers = (
            self.frame_id,
            self.core_x,
            self.core_y,
            self.core_width,
            self.core_height,
            self.input_x,
            self.input_y,
            self.input_width,
            self.input_height,
            self.deadline_us,
            self.flags,
        )
        if any(value < 0 for value in integers):
            raise ValueError("work descriptor integers must be nonnegative")
        if self.frame_id > 0xFFFFFFFF or self.deadline_us > 0xFFFFFFFF:
            raise ValueError("work descriptor 32-bit field overflow")
        if any(value > 0xFFFF for value in integers[1:9]) or self.flags > 0xFFFF:
            raise ValueError("work descriptor 16-bit field overflow")
        if not math.isfinite(self.priority):
            raise ValueError("work descriptor priority must be finite")
        return WORK_DESCRIPTOR.pack(
            self.frame_id,
            self.core_x,
            self.core_y,
            self.core_width,
            self.core_height,
            self.input_x,
            self.input_y,
            self.input_width,
            self.input_height,
            self.deadline_us,
            self.priority,
            self.flags,
            0,
        )

    # Decode only exact descriptor-sized payloads so packet corruption cannot
    # silently shift fields into later work items.
    @classmethod
    def unpack(cls, payload: bytes) -> "SparseWorkDescriptor":
        """Deserialize one exact 32-byte work descriptor."""

        if len(payload) != WORK_DESCRIPTOR_BYTES:
            raise ValueError("sparse work descriptor has the wrong byte count")
        values = WORK_DESCRIPTOR.unpack(payload)
        if values[-1] != 0:
            raise ValueError("sparse work descriptor reserved field is nonzero")
        return cls(*values[:-1])


@dataclass(frozen=True)
class SparseFrameWorkList:
    """One checksummed frame packet containing deadline-ordered work."""

    frame_id: int
    frame_width: int
    frame_height: int
    tile_width: int
    tile_height: int
    deadline_us: int
    descriptors: tuple[SparseWorkDescriptor, ...]
    flags: int = 0

    # Header CRC covers descriptor bytes; immutable geometry remains in the
    # header and is checked independently during decode.
    def pack(self) -> bytes:
        """Return one portable binary work-list packet."""

        payload = b"".join(descriptor.pack() for descriptor in self.descriptors)
        if len(self.descriptors) > 0xFFFF:
            raise ValueError("sparse work list has too many descriptors")
        header_values = (
            self.frame_width,
            self.frame_height,
            self.tile_width,
            self.tile_height,
            self.flags,
        )
        if any(value < 1 or value > 0xFFFF for value in header_values[:4]):
            raise ValueError("work-list geometry must fit positive uint16 fields")
        if self.flags < 0 or self.flags > 0xFFFF:
            raise ValueError("work-list flags must fit uint16")
        header = WORKLIST_HEADER.pack(
            WORKLIST_MAGIC,
            WORKLIST_VERSION,
            WORK_DESCRIPTOR_BYTES,
            self.frame_id,
            len(self.descriptors),
            self.frame_width,
            self.frame_height,
            self.tile_width,
            self.tile_height,
            self.flags,
            self.deadline_us,
            zlib.crc32(payload) & 0xFFFFFFFF,
        )
        return header + payload

    # Packet decoding validates magic, schema, exact length, CRC, frame IDs, and
    # rectangle containment before returning backend-visible work.
    @classmethod
    def unpack(cls, packet: bytes) -> "SparseFrameWorkList":
        """Validate and decode one sparse work-list packet."""

        if len(packet) < WORKLIST_HEADER_BYTES:
            raise ValueError("sparse work-list packet is shorter than its header")
        values = WORKLIST_HEADER.unpack(packet[:WORKLIST_HEADER_BYTES])
        magic, version, descriptor_bytes = values[:3]
        if magic != WORKLIST_MAGIC or version != WORKLIST_VERSION:
            raise ValueError("sparse work-list magic or version is invalid")
        if descriptor_bytes != WORK_DESCRIPTOR_BYTES:
            raise ValueError("sparse work-list descriptor size changed")
        frame_id, count, frame_width, frame_height, tile_width, tile_height, flags, deadline_us, expected_crc = values[3:]
        payload = packet[WORKLIST_HEADER_BYTES:]
        if len(payload) != count * WORK_DESCRIPTOR_BYTES:
            raise ValueError("sparse work-list payload length is invalid")
        if zlib.crc32(payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError("sparse work-list descriptor CRC mismatch")
        descriptors = tuple(
            SparseWorkDescriptor.unpack(payload[offset : offset + WORK_DESCRIPTOR_BYTES])
            for offset in range(0, len(payload), WORK_DESCRIPTOR_BYTES)
        )
        for descriptor in descriptors:
            if descriptor.frame_id != frame_id:
                raise ValueError("descriptor frame ID does not match work-list header")
            if (
                descriptor.core_x + descriptor.core_width > frame_width
                or descriptor.core_y + descriptor.core_height > frame_height
                or descriptor.input_x + descriptor.input_width > frame_width
                or descriptor.input_y + descriptor.input_height > frame_height
            ):
                raise ValueError("work descriptor rectangle exceeds frame bounds")
        return cls(
            frame_id=frame_id,
            frame_width=frame_width,
            frame_height=frame_height,
            tile_width=tile_width,
            tile_height=tile_height,
            deadline_us=deadline_us,
            descriptors=descriptors,
            flags=flags,
        )


# Track tile membership while merging identical horizontal runs down adjacent
# rows. Membership is needed to aggregate predicted utility per descriptor.
def compact_selected_tiles(
    selected: np.ndarray,
    *,
    tile_width: int,
    tile_height: int,
) -> tuple[tuple[tuple[int, int, int, int], tuple[int, ...]], ...]:
    """Merge a boolean tile grid into non-overlapping core rectangles."""

    mask = np.asarray(selected, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("selected tile mask must be a nonempty 2D array")
    if tile_width < 1 or tile_height < 1:
        raise ValueError("tile dimensions must be positive")
    rows, columns = mask.shape
    active: dict[tuple[int, int], tuple[tuple[int, int, int, int], tuple[int, ...]]] = {}
    finished: list[tuple[tuple[int, int, int, int], tuple[int, ...]]] = []
    for row in range(rows):
        runs: list[tuple[int, int]] = []
        column = 0
        while column < columns:
            if not bool(mask[row, column]):
                column += 1
                continue
            start = column
            while column < columns and bool(mask[row, column]):
                column += 1
            runs.append((start, column))
        next_active: dict[tuple[int, int], tuple[tuple[int, int, int, int], tuple[int, ...]]] = {}
        for run in runs:
            members = tuple(row * columns + value for value in range(run[0], run[1]))
            if run in active:
                rectangle, previous_members = active[run]
                next_active[run] = (
                    (rectangle[0], rectangle[1], rectangle[2], (row + 1) * tile_height),
                    previous_members + members,
                )
            else:
                next_active[run] = (
                    (
                        run[0] * tile_width,
                        row * tile_height,
                        run[1] * tile_width,
                        (row + 1) * tile_height,
                    ),
                    members,
                )
        finished.extend(value for key, value in active.items() if key not in next_active)
        active = next_active
    finished.extend(active.values())
    return tuple(finished)


# Unsupported merged shapes are split back into base macro-tiles. This makes a
# finite RKNN shape dictionary safe: missing classes reduce efficiency but can
# never corrupt coverage or force CPU neural fallback.
def constrain_regions_to_shapes(
    regions: Sequence[tuple[tuple[int, int, int, int], tuple[int, ...]]],
    *,
    grid_columns: int,
    tile_width: int,
    tile_height: int,
    supported_core_shapes: frozenset[tuple[int, int]] | None,
) -> tuple[tuple[tuple[int, int, int, int], tuple[int, ...]], ...]:
    """Split unsupported compacted rectangles into supported base tiles."""

    if supported_core_shapes is None:
        return tuple(regions)
    if (tile_width, tile_height) not in supported_core_shapes:
        raise ValueError("supported shape dictionary must include the base tile")
    constrained: list[tuple[tuple[int, int, int, int], tuple[int, ...]]] = []
    for rectangle, members in regions:
        width = rectangle[2] - rectangle[0]
        height = rectangle[3] - rectangle[1]
        if (width, height) in supported_core_shapes:
            constrained.append((rectangle, members))
            continue
        for member in members:
            row, column = divmod(member, grid_columns)
            constrained.append(
                (
                    (
                        column * tile_width,
                        row * tile_height,
                        (column + 1) * tile_width,
                        (row + 1) * tile_height,
                    ),
                    (member,),
                )
            )
    return tuple(constrained)


@dataclass(frozen=True)
class WorkListBuildResult:
    """A packed work list plus explicit scheduler rejection accounting."""

    work_list: SparseFrameWorkList
    admitted_tile_mask: np.ndarray
    candidate_tile_count: int
    deadline_rejected_tile_count: int
    queue_or_thermal_rejected_tile_count: int
    estimated_compute_us: float


# Build descriptors in descending predicted-utility order and admit only work
# that fits the remaining deadline budget. The GPU's classical base remains the
# commit-safe output for every rejected or unfinished region.
def build_sparse_work_list(
    selected: np.ndarray,
    neural_probabilities: np.ndarray,
    *,
    frame_id: int,
    frame_width: int,
    frame_height: int,
    tile_width: int,
    tile_height: int,
    halo: int,
    deadline_us: int,
    elapsed_us: int,
    safety_margin_us: int,
    full_frame_compute_us: float,
    queue_depth: int = 0,
    maximum_queue_depth: int = 2,
    temperature_c: float | None = None,
    thermal_soft_limit_c: float = 75.0,
    thermal_hard_limit_c: float = 85.0,
    supported_core_shapes: frozenset[tuple[int, int]] | None = None,
) -> WorkListBuildResult:
    """Create a deadline-, queue-, and temperature-aware sparse work list."""

    mask = np.asarray(selected, dtype=bool)
    probabilities = np.asarray(neural_probabilities, dtype=np.float64)
    if mask.shape != probabilities.shape or mask.ndim != 2:
        raise ValueError("selection mask and probability grid must share a 2D shape")
    if frame_width != mask.shape[1] * tile_width or frame_height != mask.shape[0] * tile_height:
        raise ValueError("tile grid does not exactly cover the LR frame")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("neural probabilities must be finite values in [0, 1]")
    if halo < 0 or full_frame_compute_us <= 0.0:
        raise ValueError("halo must be nonnegative and compute estimate positive")
    candidate_count = int(mask.sum())
    empty_mask = np.zeros_like(mask, dtype=bool)
    blocked = queue_depth >= maximum_queue_depth or (
        temperature_c is not None and temperature_c >= thermal_hard_limit_c
    )
    if blocked:
        work_list = SparseFrameWorkList(
            frame_id,
            frame_width,
            frame_height,
            tile_width,
            tile_height,
            deadline_us,
            (),
        )
        return WorkListBuildResult(work_list, empty_mask, candidate_count, 0, candidate_count, 0.0)

    available_us = max(deadline_us - elapsed_us - safety_margin_us, 0)
    if temperature_c is not None and temperature_c > thermal_soft_limit_c:
        span = max(thermal_hard_limit_c - thermal_soft_limit_c, 1e-6)
        available_us = int(available_us * max((thermal_hard_limit_c - temperature_c) / span, 0.0))
    regions = compact_selected_tiles(mask, tile_width=tile_width, tile_height=tile_height)
    regions = constrain_regions_to_shapes(
        regions,
        grid_columns=mask.shape[1],
        tile_width=tile_width,
        tile_height=tile_height,
        supported_core_shapes=supported_core_shapes,
    )
    ranked: list[tuple[float, tuple[int, int, int, int], tuple[int, ...], float]] = []
    frame_area = frame_width * frame_height
    for rectangle, members in regions:
        x0, y0, x1, y1 = rectangle
        input_x = max(0, x0 - halo)
        input_y = max(0, y0 - halo)
        input_x1 = min(frame_width, x1 + halo)
        input_y1 = min(frame_height, y1 + halo)
        # Fixed-shape RKNN models consume core+halo dimensions even at frame
        # boundaries. Missing edge samples are replicated by the backend, so
        # include those padded pixels in the compute budget rather than using a
        # deceptively cheap clipped-area estimate.
        area = (x1 - x0 + 2 * halo) * (y1 - y0 + 2 * halo)
        estimate_us = full_frame_compute_us * area / frame_area
        # Quantize to the protocol's float32 representation before retaining the
        # descriptor so an encode/decode round trip is exactly reproducible.
        priority = float(np.float32(np.mean(probabilities.reshape(-1)[list(members)])))
        ranked.append((priority / max(estimate_us, 1e-9), rectangle, members, estimate_us))
    ranked.sort(key=lambda value: (-value[0], value[1]))

    admitted_mask = np.zeros_like(mask, dtype=bool)
    descriptors: list[SparseWorkDescriptor] = []
    used_us = 0.0
    rejected_tiles = 0
    for _, rectangle, members, estimate_us in ranked:
        if used_us + estimate_us > available_us:
            rejected_tiles += len(members)
            continue
        x0, y0, x1, y1 = rectangle
        input_x = max(0, x0 - halo)
        input_y = max(0, y0 - halo)
        input_x1 = min(frame_width, x1 + halo)
        input_y1 = min(frame_height, y1 + halo)
        priority = float(np.float32(np.mean(probabilities.reshape(-1)[list(members)])))
        descriptors.append(
            SparseWorkDescriptor(
                frame_id=frame_id,
                core_x=x0,
                core_y=y0,
                core_width=x1 - x0,
                core_height=y1 - y0,
                input_x=input_x,
                input_y=input_y,
                input_width=input_x1 - input_x,
                input_height=input_y1 - input_y,
                deadline_us=deadline_us,
                priority=priority,
                flags=(
                    WORK_FLAG_PADDING_REQUIRED
                    if (input_x1 - input_x) != (x1 - x0 + 2 * halo)
                    or (input_y1 - input_y) != (y1 - y0 + 2 * halo)
                    else 0
                ),
            )
        )
        used_us += estimate_us
        for member in members:
            row, column = divmod(member, mask.shape[1])
            admitted_mask[row, column] = True
    work_list = SparseFrameWorkList(
        frame_id=frame_id,
        frame_width=frame_width,
        frame_height=frame_height,
        tile_width=tile_width,
        tile_height=tile_height,
        deadline_us=deadline_us,
        descriptors=tuple(descriptors),
    )
    return WorkListBuildResult(
        work_list=work_list,
        admitted_tile_mask=admitted_mask,
        candidate_tile_count=candidate_count,
        deadline_rejected_tile_count=rejected_tiles,
        queue_or_thermal_rejected_tile_count=0,
        estimated_compute_us=used_us,
    )
