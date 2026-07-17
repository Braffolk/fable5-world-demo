"""Whole-core+halo plurigaussian synthesis -> the 512x512 0.25 m core relief float.

Solves the two-field plurigaussian over a fixed work domain (core + 192 m halo) in world
coordinates, then crops the storage core last. Reuses ``peat_bog_network`` hydrology (flow
direction), masks (ETAK authority / open water) and prf (world-keyed innovations). Produces
the relief in the interface the existing packer/verifier consume (``core_relief_025m``).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyogrio.raw
import shapely
from scipy.ndimage import distance_transform_edt

from ..peat_bog_network import hydrology, masks
from . import anamorphosis, fields

# --- Fixed site geometry (the reselected pool-bearing raised bog, shared with the v4 pack) -
MIRE_MEMBER_ETAK_IDS = (4071135,)
MIRE_GEOMETRY_SHA256 = "0f85b612f992acf54d80cd5ee622738bb6d954dcf090840169190992ea0c36c7"
WHOLE_MIRE_BOUNDS = (539999.4959999993, 6425554.344999999, 541869.2299999967, 6430000.0929999985)
CORE_BBOX = (540224.0, 6429504.0, 540352.0, 6429632.0)
CORE_M = 128.0
OUTPUT_PITCH_M = 0.25
CORE_RES = 512  # 128 m / 0.25 m
HALO_MARGIN_M = 192.0  # >= 3 coarse correlation lengths of context around the core
MIRE_LAYER = "E_306_margala_a"


def _smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class SynthParams:
    """Frozen recipe parameters (recorded in the preregistration)."""

    fine_range_m: float = 0.7
    fine_nu: float = 2.0
    coarse_major_len_m: float = 60.0
    coarse_minor_len_m: float = 22.0
    coarse_weight_beta: float = 0.8
    ridge_gain: float = 0.25
    roughness_sigma_m: float = 0.006
    roughness_corner_m: float = 0.9
    roughness_slope_exp: float = 1.8
    amplitude: float = 1.0
    pool_bias_gain: float = 2.4
    pool_bias_radius_m: float = 12.0
    authority_taper_m: float = 3.0
    water_taper_m: float = 1.0
    core_edge_taper_m: float = 12.0
    canonical_seed_hex: str = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"


@dataclass
class SynthResult:
    relief_core: np.ndarray  # (512,512) float64, 0.25 m over CORE_BBOX
    height_work: np.ndarray  # raw relative height over the work grid (pre-envelope)
    relief_work: np.ndarray  # enveloped relief over the work grid
    latent_std: np.ndarray  # standardized latent over the work grid
    coarse: np.ndarray
    authority: np.ndarray
    open_water: np.ndarray
    core_slice: tuple[slice, slice]
    grid: fields.WorkGrid
    flow_angle_rad: float
    class_fractions: dict
    params: SynthParams
    inputs: dict = field(default_factory=dict)


def _work_bbox() -> tuple[int, int, int, int]:
    e0 = int(CORE_BBOX[0] - HALO_MARGIN_M)
    n0 = int(CORE_BBOX[1] - HALO_MARGIN_M)
    e1 = int(CORE_BBOX[2] + HALO_MARGIN_M)
    n1 = int(CORE_BBOX[3] + HALO_MARGIN_M)
    return e0, n0, e1, n1


def load_mire(etak: Path) -> shapely.Geometry:
    where = " OR ".join(f"etak_id = {int(v)}" for v in MIRE_MEMBER_ETAK_IDS)
    metadata, _fids, wkbs, values = pyogrio.raw.read(
        etak, layer=MIRE_LAYER, where=where, columns=["etak_id"], return_fids=True
    )
    if str(metadata.get("crs")) != "EPSG:3301" or wkbs is None:
        raise ValueError("dev mire geometry unavailable")
    if sorted(int(v) for v in values[0]) != sorted(int(v) for v in MIRE_MEMBER_ETAK_IDS):
        raise ValueError("dev mire membership changed")
    geometry = shapely.union_all([shapely.force_2d(shapely.from_wkb(bytes(w))) for w in wkbs])
    if hashlib.sha256(geometry.wkb).hexdigest() != MIRE_GEOMETRY_SHA256:
        raise ValueError("dev mire geometry changed")
    return geometry


def _flow_angle(dtm10: Path, work_bbox: tuple[int, int, int, int]) -> float:
    """Mean downslope flow angle at the core, atan2(flow_row, flow_col) on the row=S/col=E grid."""
    hydro = hydrology.compute(dtm10, WHOLE_MIRE_BOUNDS, work_bbox, 2.0)
    e0, n0, e1, n1 = work_bbox
    # Core center in the 2 m process grid.
    cc = int(round((CORE_BBOX[0] + CORE_M / 2 - e0) / 2.0))
    cr = int(round((n1 - (CORE_BBOX[1] + CORE_M / 2)) / 2.0))
    r0, r1 = max(0, cr - 8), cr + 8
    c0, c1 = max(0, cc - 8), cc + 8
    fr = float(np.mean(hydro.flow_row[r0:r1, c0:c1]))
    fc = float(np.mean(hydro.flow_col[r0:r1, c0:c1]))
    return float(np.arctan2(fr, fc))


def _white_cache(grid: fields.WorkGrid, cache_dir: Path | None) -> dict[str, np.ndarray]:
    channels = ("fine", "coarse", "rough")
    if cache_dir is not None:
        key = f"white_{grid.east0_idx}_{grid.north1_idx}_{grid.width}_{grid.height}.npz"
        path = cache_dir / key
        if path.exists():
            data = np.load(path)
            return {c: data[c] for c in channels}
    white = {c: fields.world_white(grid, c) for c in channels}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_dir / key, **white)
    return white


def synthesize(etak: Path, dtm10: Path, params: SynthParams,
               cache_dir: Path | None = None) -> SynthResult:
    work_bbox = _work_bbox()
    e0, n0, e1, n1 = work_bbox
    width = int(round((e1 - e0) / OUTPUT_PITCH_M))
    height = int(round((n1 - n0) / OUTPUT_PITCH_M))
    grid = fields.WorkGrid(east0=float(e0), north1=float(n1), width=width, height=height, pitch=OUTPUT_PITCH_M)

    mire = load_mire(etak)
    mask_set = masks.build(etak, mire, work_bbox, OUTPUT_PITCH_M)
    authority = mask_set.authority
    open_water = mask_set.open_water
    flow_angle = _flow_angle(dtm10, work_bbox)

    white = _white_cache(grid, cache_dir)
    fine = fields.fine_field(grid, white["fine"], range_m=params.fine_range_m, nu=params.fine_nu)
    coarse = fields.coarse_field(
        grid, white["coarse"], major_len_m=params.coarse_major_len_m,
        minor_len_m=params.coarse_minor_len_m, flow_angle_rad=flow_angle,
    )
    rough = fields.roughness_field(
        grid, white["rough"], corner_m=params.roughness_corner_m, slope_exp=params.roughness_slope_exp
    )

    # Pool-margin wetness bias: hollows grade into real ETAK pools; bias the latent downward
    # within a margin of open water. Distance in metres.
    dist_water = distance_transform_edt(~open_water) * OUTPUT_PITCH_M if open_water.any() else None
    bias = np.zeros_like(fine)
    if dist_water is not None:
        bias = params.pool_bias_gain * np.clip(1.0 - dist_water / params.pool_bias_radius_m, 0.0, 1.0)

    latent_raw = fine + params.coarse_weight_beta * coarse - bias
    mu = float(latent_raw[authority].mean())
    sd = float(latent_raw[authority].std())
    latent_std = (latent_raw - mu) / (sd if sd > 0 else 1.0)

    height_rel = anamorphosis.anamorphosis(
        latent_std, amplitude=params.amplitude, ridge_field=coarse, ridge_gain=params.ridge_gain
    )
    height_rel = height_rel + params.roughness_sigma_m * rough

    # DC-drift guard: demean over the full-relief authority interior (away from tapers/water).
    interior = authority & (~open_water)
    if dist_water is not None:
        interior = interior & (dist_water > params.pool_bias_radius_m)
    height_rel = height_rel - float(height_rel[interior].mean())

    # Envelope: taper to 0 at authority boundary, at open water, and at the core boundary.
    auth_dist = distance_transform_edt(authority) * OUTPUT_PITCH_M
    auth_taper = _smoothstep(auth_dist / params.authority_taper_m)
    water_taper = _smoothstep(dist_water / params.water_taper_m) if dist_water is not None else np.ones_like(fine)
    core_taper = _core_edge_taper(grid, work_bbox, params.core_edge_taper_m)
    relief_work = height_rel * auth_taper * water_taper * core_taper
    relief_work = np.where(authority, relief_work, 0.0)
    relief_work[open_water] = 0.0

    core_slice = _core_slice(grid, work_bbox)
    relief_core = relief_work[core_slice].astype(np.float64)

    # Plurigaussian class fractions over the core authority (compare to the Ilyasov target).
    core_auth = authority[core_slice]
    p_lo, p_hi = anamorphosis.class_percentile_bounds()
    from scipy.special import ndtr

    perc = ndtr(latent_std[core_slice])[core_auth]
    class_fractions = {
        "hollow": float((perc < p_lo).mean()),
        "lawn": float(((perc >= p_lo) & (perc < p_hi)).mean()),
        "hummock": float((perc >= p_hi).mean()),
        "p_lo": p_lo,
        "p_hi": p_hi,
    }

    return SynthResult(
        relief_core=relief_core, height_work=height_rel, relief_work=relief_work,
        latent_std=latent_std, coarse=coarse, authority=authority, open_water=open_water,
        core_slice=core_slice, grid=grid, flow_angle_rad=flow_angle,
        class_fractions=class_fractions, params=params,
        inputs={"etak": str(etak), "dtm10": str(dtm10), "work_bbox_en": list(work_bbox)},
    )


def _core_slice(grid: fields.WorkGrid, work_bbox: tuple[int, int, int, int]) -> tuple[slice, slice]:
    e0, _n0, _e1, n1 = work_bbox
    r0 = int(round((n1 - CORE_BBOX[3]) / OUTPUT_PITCH_M))
    r1 = int(round((n1 - CORE_BBOX[1]) / OUTPUT_PITCH_M))
    c0 = int(round((CORE_BBOX[0] - e0) / OUTPUT_PITCH_M))
    c1 = int(round((CORE_BBOX[2] - e0) / OUTPUT_PITCH_M))
    return slice(r0, r1), slice(c0, c1)


def _core_edge_taper(grid: fields.WorkGrid, work_bbox: tuple[int, int, int, int], taper_m: float) -> np.ndarray:
    """Envelope 1 in the core interior, cosine-fading to 0 at the core boundary; 0 outside core.

    The relief must be zero at the storage-core boundary so the overlay embeds seamlessly into
    the baseline outside the corrected core (this is the corrected-region edge taper, not an
    inter-chunk crossfade).
    """
    e0, _n0, _e1, n1 = work_bbox
    east = e0 + (np.arange(grid.width) + 0.5) * OUTPUT_PITCH_M
    north = n1 - (np.arange(grid.height) + 0.5) * OUTPUT_PITCH_M
    ce0, cn0, ce1, cn1 = CORE_BBOX
    d_e = np.minimum(east - ce0, ce1 - east)
    d_n = np.minimum(north - cn0, cn1 - north)
    dist = np.minimum(d_n[:, None], d_e[None, :])  # metres inside the core (negative outside)
    return _smoothstep(dist / taper_m)
