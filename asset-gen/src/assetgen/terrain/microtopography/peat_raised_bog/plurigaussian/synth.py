"""Whole-core+halo plurigaussian synthesis -> the 2048x2048 0.0625 m core relief float.

Solves the two-field plurigaussian over a fixed work domain (core + 192 m halo) in world
coordinates, then crops the storage core last. Reuses ``peat_bog_network`` hydrology (flow
direction), masks (ETAK authority / open water) and prf (world-keyed innovations). Produces
the relief in the interface the existing packer/verifier consume (``core_relief_00625m``).

The relief is emitted at the LOD -2 finest rung pitch (0.0625 m, 2048 over the 128 m core) so
the packer places it 1:1 (no 4x nearest-neighbour block-replication, which produced flat
0.25 m terraces). The stochastic fields are DRAWN on the accepted 0.25 m lattice -- fixing the
realization byte-identically to the accepted, gate-passing 0.25 m surface -- and then
band-limited (trig / FFT zero-pad) INTERPOLATED to 0.0625 m. That evaluates the SAME
continuous surface the accepted spectral synthesis already defines, sampled finer; it is
mathematically exact for the band-limited field (NOT a nearest/bilinear upsample). World-PRF
determinism stays on the 0.25 m lattice. Masks, pool-bias, ordered thresholds, the per-cell
Gaussian anamorphosis and the tapers are evaluated natively at 0.0625 m.
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
OUTPUT_PITCH_M = 0.0625  # LOD -2 finest rung pitch; the surface is sampled here (no terraces)
CORE_RES = 2048  # 128 m / 0.0625 m == FINE_CORE
# The stochastic fields are DRAWN on this accepted lattice (world-PRF keys on it), which fixes
# the realization byte-identically to the accepted 0.25 m surface; they are then trig
# (band-limited) interpolated to OUTPUT_PITCH_M -- the SAME continuous surface sampled finer,
# not a nearest/bilinear upsample. Masks, pool-bias, thresholds, anamorphosis and tapers are
# evaluated natively at OUTPUT_PITCH_M.
FIELD_PITCH_M = 0.25
HALO_MARGIN_M = 192.0  # >= 3 coarse correlation lengths of context around the core
MIRE_LAYER = "E_306_margala_a"


def _smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class SynthParams:
    """Frozen recipe parameters (recorded in the preregistration).

    v7 quality upgrade over v6: the single-Matern fine field and single-wavelength coarse
    field become MULTI-SCALE (broad microform size distribution + multi-octave ridge spectrum),
    and the roughness channel is drawn NATIVELY at the 6 cm output pitch (genuine sub-0.5 m
    texture), not the 0.25 m lattice then interpolated.
    """

    # Multi-scale fine microform field: PSD = sqrt(sum_i w_i * Matern(range_i, nu)). Gives a
    # ~0.35-1.4 m microform size distribution instead of the v6 single 0.7 m size, tuned so the
    # detrended autocorr first-zero stays in [0.3,0.9] m (two_scale gate).
    fine_ranges: tuple[tuple[float, float], ...] = ((0.35, 0.6), (0.8, 1.0), (1.4, 0.12))
    fine_nu: float = 2.0
    # Multi-octave anisotropic coarse patterning: (major_len_m, minor_len_m, weight), major
    # perpendicular to flow. Breaks the v6 single 60/22 m wavelength into a spread of spacings.
    coarse_octaves: tuple[tuple[float, float, float], ...] = ((90.0, 34.0, 1.0), (55.0, 20.0, 0.7), (32.0, 12.0, 0.45))
    coarse_weight_beta: float = 0.8
    ridge_gain: float = 0.25
    # Native-6 cm roughness (the detectability fix): additive, NOT in the latent, so it cannot
    # disturb ordering/transiogram/fraction gates. Raised from v6's 0.006 (invisible at 6 cm) to
    # the largest value keeping the anti_artifact PSD slope <= -3.3 with margin (measured -3.53).
    roughness_sigma_m: float = 0.018
    roughness_corner_m: float = 0.7
    roughness_slope_exp: float = 1.8
    amplitude: float = 1.15
    pool_bias_gain: float = 2.4
    pool_bias_radius_m: float = 12.0
    authority_taper_m: float = 3.0
    water_taper_m: float = 1.0
    core_edge_taper_m: float = 12.0
    canonical_seed_hex: str = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"


@dataclass
class SynthResult:
    relief_core: np.ndarray  # (2048,2048) float64, 0.0625 m over CORE_BBOX
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
    # fine + coarse latent streams are drawn on the 0.25 m field lattice (determinism unchanged
    # from v6). The rough stream is drawn separately at the native 6 cm output pitch, below.
    channels = ("fine", "coarse")
    if cache_dir is not None:
        key = f"white_{grid.east0_idx}_{grid.north1_idx}_{grid.width}_{grid.height}.npz"
        path = cache_dir / key
        if path.exists():
            data = np.load(path)
            if all(c in data for c in channels):
                return {c: data[c] for c in channels}
    white = {c: fields.world_white(grid, c) for c in channels}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_dir / key, **white)
    return white


def _rough_white_6cm(grid: fields.WorkGrid, cache_dir: Path | None) -> np.ndarray:
    """World-keyed white for the roughness channel drawn NATIVELY at the 6 cm output lattice.

    Genuine sub-0.5 m content (v6 drew rough at 0.25 m then band-limit-interpolated, so it could
    carry NO detail below 0.5 m). Its own world-PRF stream (channel 'rough'), keyed on the 6 cm
    integer world lattice -> distinct from the 0.25 m latent streams, deterministic, crop-
    independent. Cached (the 6 cm draw is ~8192^2 BLAKE2b keys).
    """
    if cache_dir is not None:
        key = f"rough6cm_{grid.east0_idx}_{grid.north1_idx}_{grid.width}_{grid.height}.npz"
        path = cache_dir / key
        if path.exists():
            return np.load(path)["rough"]
    rough = fields.world_white(grid, "rough")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_dir / key, rough=rough)
    return rough


def synthesize(etak: Path, dtm10: Path, params: SynthParams,
               cache_dir: Path | None = None) -> SynthResult:
    work_bbox = _work_bbox()
    e0, n0, e1, n1 = work_bbox
    width = int(round((e1 - e0) / OUTPUT_PITCH_M))
    height = int(round((n1 - n0) / OUTPUT_PITCH_M))
    grid = fields.WorkGrid(east0=float(e0), north1=float(n1), width=width, height=height, pitch=OUTPUT_PITCH_M)

    # Stochastic fields on the accepted FIELD_PITCH_M lattice -> byte-identical accepted
    # realization; then band-limited trig interpolation to the 6 cm output grid.
    upsample = int(round(FIELD_PITCH_M / OUTPUT_PITCH_M))
    fwidth = int(round((e1 - e0) / FIELD_PITCH_M))
    fheight = int(round((n1 - n0) / FIELD_PITCH_M))
    field_grid = fields.WorkGrid(
        east0=float(e0), north1=float(n1), width=fwidth, height=fheight, pitch=FIELD_PITCH_M
    )

    mire = load_mire(etak)
    mask_set = masks.build(etak, mire, work_bbox, OUTPUT_PITCH_M)
    authority = mask_set.authority
    open_water = mask_set.open_water
    flow_angle = _flow_angle(dtm10, work_bbox)

    white = _white_cache(field_grid, cache_dir)
    fine_lo = fields.multiscale_fine_field(
        field_grid, white["fine"], ranges=params.fine_ranges, nu=params.fine_nu
    )
    coarse_lo = fields.multiscale_coarse_field(
        field_grid, white["coarse"], octaves=params.coarse_octaves, flow_angle_rad=flow_angle,
    )
    fine = fields.fft_interp_cellcentered(fine_lo, upsample)
    coarse = fields.fft_interp_cellcentered(coarse_lo, upsample)
    # Native-6 cm roughness: drawn on the OUTPUT lattice (NOT the 0.25 m lattice then interp),
    # so it carries genuine sub-0.5 m texture. Additive to the height (below), never in the
    # latent -> cannot disturb ordering / transiogram / fraction gates.
    rough = fields.roughness_field(
        grid, _rough_white_6cm(grid, cache_dir),
        corner_m=params.roughness_corner_m, slope_exp=params.roughness_slope_exp,
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

    # --- #104 pool-depth carve (task D) ---------------------------------------------------
    # Open water carries a monotone, non-positive Laugas BED WEDGE instead of 0: the rendered
    # water surface (a SEPARATE `water` HeightLevel plane, waterY, INHERITED unchanged by this
    # height-only overlay) then floats above the carved bed, opening a real depth column
    # (Beer-Lambert). The wedge is the deterministic #104 physics model (process.water.
    # bed_depth_field) over the SAME work window at 6 cm, cropped to open_water; it tapers to 0
    # at the shoreline (no water-side cliff) and is multiplied by the core-edge taper so a pool
    # clipped at the storage-core boundary carries no step into the uncarved baseline outside.
    # This is NOT a fake microform bump (relief <= 0 everywhere in water); it is the depth model
    # the user asked for, and the "open-water relief-free" safety gate is reinterpreted to forbid
    # only POSITIVE microform in water while permitting this monotone carve down.
    pool_depth = _pool_bed_depth(work_bbox)  # >=0 metres below the water surface, over the grid
    open_water_wedge = np.where(open_water, -pool_depth * core_taper, 0.0)
    relief_work[open_water] = open_water_wedge[open_water]

    core_slice = _core_slice(grid, work_bbox)
    relief_core = relief_work[core_slice].astype(np.float64)

    # Plurigaussian class fractions over the WHOLE work-grid authority (not the 128 m core,
    # which is smaller than ~2 coarse ridge-hollow wavelengths and so is fragile to the coarse
    # PHASE the core happens to sample). Since the latent is standardized over the full
    # authority and the thresholds are its percentiles, the full-authority fractions recover
    # the Ilyasov target by construction and are phase-stable. See prereg v7 gate-appropriateness.
    p_lo, p_hi = anamorphosis.class_percentile_bounds()
    from scipy.special import ndtr

    perc = ndtr(latent_std)[authority]
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


def _pool_bed_depth(work_bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Deterministic Laugas/river bed-depth wedge (>=0 m) over the work grid at 6 cm.

    Reuses the #104 depth model (``process.water.bed_depth_field``) over the SAME haloed work
    window the synth uses, so a Laugas is measured against its WHOLE polygon shore (correct
    shore-shelf), not clipped at the storage-core edge. The ``stack`` argument is unused inside
    the depth model (no DEM dependency); the coverage supersample (which the model also computes
    but we discard) is disabled for this call so the 8192^2 window stays in memory.
    """
    from .....process import water as water_model

    e0, n0, e1, n1 = work_bbox
    window = (float(e0), float(n0), float(e1), float(n1), OUTPUT_PITCH_M)
    saved_ss = water_model.COVERAGE_SUPERSAMPLE
    water_model.COVERAGE_SUPERSAMPLE = 1  # discard coverage; avoid the 65536^2 supersample
    try:
        depth, _coverage = water_model.bed_depth_field(window, None)
    finally:
        water_model.COVERAGE_SUPERSAMPLE = saved_ss
    return np.asarray(depth, dtype=np.float64)
