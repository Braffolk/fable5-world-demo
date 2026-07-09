"""Multi-variable ecological suitability — what the local gradients make POSSIBLE.

A community label from a source polygon says what *tends* to grow there; whether it actually
can, and how densely, is cut by slope, soil wetness, soil texture, stoniness and fertility.
A fern won't hold a steep dry sandy knoll even inside a "herb-rich" stand; sphagnum needs it
wet. We turn those gradients into a 0..1 density multiplier so unsuitable ground thins or
clears instead of being uniformly filled.

Inputs are the already-cooked planes: height (LOD0, for slope), soil (soilType/texCore/
stoniness/boniteet). Everything is computed per 2 m cell.
"""
from __future__ import annotations

import tomllib
from functools import lru_cache

import numpy as np

from ..config import CONFIG_DIR

# community moisture/fertility tags -> target position on the two gradients (0..1)
MOISTURE = {"any": 0.5, "dry": 0.15, "fresh": 0.4, "moist": 0.65, "wet": 0.9}
FERTILITY = {"poor": 0.25, "moderate": 0.55, "rich": 0.85}

# texCore id (config/soil-texture.toml) -> (wetness, richness) of that texture
TEXTURE = {
    0: (0.45, 0.45),  # unknown -> neutral
    1: (0.20, 0.20), 2: (0.28, 0.30),          # l, pl (sand)
    3: (0.40, 0.45), 4: (0.48, 0.60), 5: (0.48, 0.60), 6: (0.50, 0.62), 7: (0.55, 0.60),  # sl, ls*
    8: (0.62, 0.65),                            # s (clay)
    9: (0.90, 0.35), 10: (0.82, 0.35), 11: (0.90, 0.38), 12: (0.95, 0.40),  # peat t/t1/t2/t3
    13: (0.90, 0.42), 14: (0.88, 0.38),         # th, tt
}


@lru_cache
def _wet_soil_ids() -> frozenset[int]:
    """Soil-type ids the legend marks wet (gley / gleistunud / turvastunud / soomuld / lammi)."""
    raw = tomllib.loads((CONFIG_DIR / "soil-types.toml").read_text())
    wet_markers = ("glei", "gleistunud", "turvastunud", "soomuld", "lammi", "veealune", "ranniku")
    return frozenset(
        spec["id"] for spec in raw["types"].values()
        if any(m in spec["name"].lower() for m in wet_markers)
    )


def slope_deg_from_height(height_lod0: np.ndarray, target_res: int) -> np.ndarray:
    """Slope in degrees at the 2 m grid, from the 1 m LOD0 height raster."""
    gy, gx = np.gradient(height_lod0.astype(np.float32), 1.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    step = max(1, height_lod0.shape[0] // target_res)
    return slope[::step, ::step][:target_res, :target_res]


def cell_wetness(soil_type: np.ndarray, tex_core: np.ndarray) -> np.ndarray:
    lut = np.full(256, 0.45, dtype=np.float32)
    for tid, (w, _) in TEXTURE.items():
        lut[tid] = w
    wet = lut[tex_core]
    gley = np.isin(soil_type, list(_wet_soil_ids()))
    return np.maximum(wet, np.where(gley, 0.78, 0.0)).astype(np.float32)


def cell_richness(boniteet: np.ndarray, tex_core: np.ndarray) -> np.ndarray:
    lut = np.full(256, 0.45, dtype=np.float32)
    for tid, (_, r) in TEXTURE.items():
        lut[tid] = r
    tex_rich = lut[tex_core]
    boni = np.clip(boniteet.astype(np.float32) / 55.0, 0, 1)
    return (0.65 * boni + 0.35 * tex_rich).astype(np.float32)


def understory_suitability(
    community: np.ndarray,
    target_wetness: np.ndarray,   # per-community target, indexed by community id -> value
    target_fertility: np.ndarray,
    slope_deg: np.ndarray,
    wetness: np.ndarray,
    richness: np.ndarray,
    stoniness: np.ndarray,
) -> np.ndarray:
    """0..1 density multiplier from gradient match + slope/stoniness penalties."""
    tw = target_wetness[community]
    tf = target_fertility[community]
    match = np.exp(-((wetness - tw) ** 2) / 0.09) * np.exp(-((richness - tf) ** 2) / 0.14)
    slope_pen = np.clip(1.0 - np.clip(slope_deg - 12.0, 0, None) / 45.0, 0.2, 1.0)
    stone_pen = np.clip(1.0 - stoniness.astype(np.float32) * 0.07, 0.4, 1.0)
    return (match * slope_pen * stone_pen).astype(np.float32)


@lru_cache
def community_targets() -> tuple[np.ndarray, np.ndarray]:
    from .understory import load_communities

    comm = load_communities()
    raw = tomllib.loads((CONFIG_DIR / "understory-communities.toml").read_text())["communities"]
    tw = np.full(256, 0.5, dtype=np.float32)
    tf = np.full(256, 0.5, dtype=np.float32)
    for spec in raw.values():
        tw[spec["id"]] = MOISTURE.get(spec["moisture"], 0.5)
        tf[spec["id"]] = FERTILITY.get(spec["fertility"], 0.5)
    return tw, tf


def debris_suitability(surface_class: np.ndarray, slope_deg: np.ndarray, stoniness: np.ndarray) -> np.ndarray:
    """Litter accumulates on gentle ground and washes off steep slopes; stone/scree classes
    (5,6) do the opposite — steep, thin, stony ground exposes more rock."""
    is_stone = np.isin(surface_class, [5, 6, 7])
    litter = np.clip(1.0 - np.clip(slope_deg - 10.0, 0, None) / 40.0, 0.25, 1.0)
    stone = np.clip(0.6 + np.clip(slope_deg - 8.0, 0, None) / 30.0 + stoniness.astype(np.float32) * 0.06, 0.3, 1.4)
    return np.where(is_stone, stone, litter).astype(np.float32)
