"""Typed, immutable records shared by structural terrain repair stages."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class FlowOrientation(StrEnum):
    """Direction of increasing station distance relative to downstream flow."""

    FORWARD = "forward"
    REVERSE = "reverse"
    UNKNOWN = "unknown"


class WaterAbstention(StrEnum):
    """Fail-closed reasons emitted before a reach may alter terrain."""

    AMBIGUOUS_TOPOLOGY = "ambiguous_topology"
    DISCONTINUITY_UNOWNED = "discontinuity_unowned"
    ENDPOINT_UNSUPPORTED = "endpoint_unsupported"
    EPOCH_INCOMPATIBLE = "epoch_incompatible"
    MISSING_SPAN = "missing_span_over_limit"
    PROFILE_NONFINITE = "profile_nonfinite"
    RESOURCE_LIMIT = "resource_limit"
    ROBUST_FIT_FAILED = "robust_fit_failed"
    WATER_SUPPORT_INSUFFICIENT = "water_support_insufficient"
    BANK_INCONSISTENT = "bank_inconsistent"


class CorrectionClass(StrEnum):
    """Structural ownership labels; values are serialized by name, not ordinal."""

    RAW_DTM_UNCHANGED = "raw_dtm_unchanged"
    UNKNOWN_BATHYMETRY_ENVELOPE = "unknown_bathymetry_render_envelope"
    ABSTAINED = "abstained"


def _readonly_1d(values: np.ndarray, dtype: np.dtype, name: str) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class FlowProfile:
    """Qualified and fitted water elevations along one unbranched reach segment."""

    reach_id: str
    epoch_id: str
    station_m: np.ndarray
    observation_y: np.ndarray
    accepted: np.ndarray
    water_y: np.ndarray
    orientation: FlowOrientation
    longest_missing_span_m: float
    source_artifact_sha256: str

    def __post_init__(self) -> None:
        station = _readonly_1d(self.station_m, np.dtype(np.float64), "station_m")
        observation = _readonly_1d(
            self.observation_y, np.dtype(np.float64), "observation_y"
        )
        accepted = _readonly_1d(self.accepted, np.dtype(np.bool_), "accepted")
        water = _readonly_1d(self.water_y, np.dtype(np.float64), "water_y")
        size = station.size
        if size < 2 or any(array.size != size for array in (observation, accepted, water)):
            raise ValueError("flow-profile arrays must have the same length >= 2")
        if not np.isfinite(station).all() or not np.all(np.diff(station) > 0.0):
            raise ValueError("station_m must be finite and strictly increasing")
        if not np.isfinite(water).all():
            raise ValueError("accepted flow profile must contain finite water_y")
        if not np.isfinite(observation[accepted]).all():
            raise ValueError("accepted observations must be finite")
        if not self.reach_id or not self.epoch_id:
            raise ValueError("reach_id and epoch_id are required")
        if len(self.source_artifact_sha256) != 64:
            raise ValueError("source_artifact_sha256 must be a SHA-256 hex digest")
        try:
            bytes.fromhex(self.source_artifact_sha256)
        except ValueError as error:
            raise ValueError("source_artifact_sha256 must be a SHA-256 hex digest") from error
        object.__setattr__(self, "station_m", station)
        object.__setattr__(self, "observation_y", observation)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "water_y", water)


@dataclass(frozen=True)
class AbstainedReach:
    """A reach that is explicitly forbidden from modifying height or water products."""

    reach_id: str
    reason: WaterAbstention
    detail: str


@dataclass(frozen=True)
class WaterSamples:
    """Separated water/bed evaluation at paired world-coordinate samples."""

    wet: np.ndarray
    supported: np.ndarray
    water_y: np.ndarray
    bed_y: np.ndarray

    def __post_init__(self) -> None:
        wet = _readonly_1d(self.wet, np.dtype(np.bool_), "wet")
        supported = _readonly_1d(self.supported, np.dtype(np.bool_), "supported")
        water = _readonly_1d(self.water_y, np.dtype(np.float64), "water_y")
        bed = _readonly_1d(self.bed_y, np.dtype(np.float64), "bed_y")
        size = wet.size
        if any(array.size != size for array in (supported, water, bed)):
            raise ValueError("water-sample arrays must have the same length")
        if np.any(supported & ~wet):
            raise ValueError("supported water samples must be wet")
        if not np.isfinite(water[supported]).all() or not np.isfinite(bed[supported]).all():
            raise ValueError("supported water and bed heights must be finite")
        if np.isfinite(water[~supported]).any() or np.isfinite(bed[~supported]).any():
            raise ValueError("unsupported/dry water and bed heights must be NaN")
        object.__setattr__(self, "wet", wet)
        object.__setattr__(self, "supported", supported)
        object.__setattr__(self, "water_y", water)
        object.__setattr__(self, "bed_y", bed)


@dataclass(frozen=True)
class StructuralTile:
    """Absolute structural height plus explicit ownership/abstention masks."""

    height: np.ndarray
    valid: np.ndarray
    unknown_bathymetry: np.ndarray
    forbidden_morphology: np.ndarray

    def __post_init__(self) -> None:
        height = np.array(self.height, dtype=np.float64, copy=True)
        valid = np.array(self.valid, dtype=np.bool_, copy=True)
        unknown = np.array(self.unknown_bathymetry, dtype=np.bool_, copy=True)
        forbidden = np.array(self.forbidden_morphology, dtype=np.bool_, copy=True)
        if height.ndim != 2 or any(
            array.shape != height.shape for array in (valid, unknown, forbidden)
        ):
            raise ValueError("structural-tile arrays must be matching 2D rasters")
        if not np.isfinite(height[valid]).all() or np.isfinite(height[~valid]).any():
            raise ValueError("structural invalid samples must be NaN and valid samples finite")
        if np.any(unknown & (~valid | ~forbidden)):
            raise ValueError("unknown bathymetry must be valid and morphology-forbidden")
        for array in (height, valid, unknown, forbidden):
            array.flags.writeable = False
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "unknown_bathymetry", unknown)
        object.__setattr__(self, "forbidden_morphology", forbidden)
