"""Per-tree derivation: nDSM canopy -> crown positions -> species from forest stands.

Method: variable-window local maxima on the 1 m nDSM (Popescu-style — window radius grows
with canopy height), keep peaks >= MIN_TREE_H that sit under an ETAK forest/shrub class,
assign species by point-in-stand join against Metsaregister (deterministic position hash
over the stand's composition), fallback to a landcover/conifer prior outside registry cover.

MINIMAL record — only what the DATA determines, plus one art-direction seed:
  x,z     u16  chunk-local position  (from nDSM crown detection)
  species u8   type                  (from Metsaregister stand; FULL ~21-id taxonomy)
  scale   u8   size = crownH/refH    (from nDSM height)
  variant u8   variation seed        (subspecies/form/sway; client expands cosmetics)
No y (client grounds the tree on the rendered terrain). 7 bytes/tree. Deterministic and
Morton-sorted so chunks are content-hash-stable.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from scipy.ndimage import maximum_filter

from ..config import CONFIG_DIR, DATA_IN
from .landcover import BIOME_TEXEL, load_rules, rasterize_classes
from .mosaic import RasterStack

MIN_TREE_H = 3.0
NDSM_TEXEL = 1.0
FOREST_CLASSES = {1, 2}  # forest, shrub (from landcover-classes.toml palette)

unmapped_species: dict[str, int] = {}


@dataclass(frozen=True)
class SpeciesMap:
    code_id: dict[str, int]  # Metsaregister code -> species id
    ref_height: dict[str, float]  # code -> ref height
    ref_by_id: dict[int, float]
    fallback_conifer: str
    fallback_broadleaf: str
    dictionary: dict[int, dict]  # id -> {code, latin, english, leaf, ref_height_m}


def load_species_map() -> SpeciesMap:
    raw = tomllib.loads((CONFIG_DIR / "species-map.toml").read_text())
    sp = raw["species"]
    code_id = {code: spec["id"] for code, spec in sp.items()}
    ref_height = {code: float(spec["ref_height_m"]) for code, spec in sp.items()}
    ref_by_id = {spec["id"]: float(spec["ref_height_m"]) for spec in sp.values()}
    dictionary = {
        spec["id"]: {
            "code": code, "latin": spec.get("latin", ""), "english": spec["english"],
            "leaf": spec["leaf"], "ref_height_m": spec["ref_height_m"],
        }
        for code, spec in sp.items()
    }
    fb = raw["fallback"]
    return SpeciesMap(code_id, ref_height, ref_by_id, fb["conifer"], fb["broadleaf"], dictionary)


def ndsm_stack() -> RasterStack:
    paths = sorted((DATA_IN / "ndsm_1m").glob("*_ndsm_1m.tif"))
    if not paths:
        raise FileNotFoundError("no nDSM rasters — run `assetgen fetch --only elevation`")
    return RasterStack(paths)


def _variable_window_maxima(h: np.ndarray) -> np.ndarray:
    """Boolean crown-peak mask. Window radius r = round(1.5 + 0.05*height), clamped 1..7.

    Applied in a few height bands so the filter footprint tracks canopy height (tall
    crowns need a wider exclusion window than saplings) without a per-pixel filter.
    """
    peaks = np.zeros(h.shape, dtype=bool)
    for lo, hi, r in [(3, 8, 1), (8, 14, 2), (14, 20, 3), (20, 26, 4), (26, 1e9, 5)]:
        band = (h >= lo) & (h < hi)
        if not band.any():
            continue
        local_max = maximum_filter(h, size=2 * r + 1, mode="nearest")
        peaks |= band & (h >= local_max - 1e-3)
    return peaks


def _load_stands(bbox_en) -> tuple[np.ndarray, np.ndarray, list]:
    """(shapely polygons, dominant-code array, composition list) from the WFS dump."""
    files = sorted((DATA_IN / "wfs" / "metsaregister").glob("*eraldis.*.ndjson"))
    geoms, codes, comps = [], [], []
    for f in files:
        for line in f.open():
            feat = json.loads(line)
            geom = feat.get("geometry")
            if not geom:
                continue
            geoms.append(shapely.geometry.shape(geom))
            codes.append(feat["properties"].get("peapuuliik_kood"))
            comps.append(feat["properties"])
    return np.array(geoms, dtype=object), np.array(codes, dtype=object), comps


def _pcg_hash(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Deterministic uint32 hash of integer coords (stable across runs/machines)."""
    h = (x.astype(np.uint32) * np.uint32(747796405)) ^ (y.astype(np.uint32) * np.uint32(2891336453))
    h = (h ^ (h >> 15)) * np.uint32(2246822519)
    h = (h ^ (h >> 13)) * np.uint32(3266489917)
    return h ^ (h >> 16)


@dataclass
class TreeColumns:
    x: np.ndarray  # u16 chunk-local (step chunk_m / 65535)
    z: np.ndarray
    species: np.ndarray  # u8 species id
    scale: np.ndarray  # u8, height/ref * 64
    variant: np.ndarray  # u8 variation seed

    def __len__(self) -> int:
        return len(self.x)


def derive_trees_for_chunk(
    chunk_bounds_en: tuple[float, float, float, float],
    ndsm: RasterStack,
    stands: tuple[np.ndarray, np.ndarray, list],
    smap: SpeciesMap,
    chunk_m: int,
) -> TreeColumns:
    e_min, n_min, e_max, n_max = chunk_bounds_en
    rows = cols = int(chunk_m / NDSM_TEXEL)
    h = np.nan_to_num(ndsm.read_window(e_min, n_min, e_max, n_max, NDSM_TEXEL), nan=0.0)

    # forest/shrub mask from land cover (2 m) upsampled to the 1 m nDSM grid
    lc_window = (e_min, n_min - BIOME_TEXEL, e_max + BIOME_TEXEL, n_max, BIOME_TEXEL)
    lc = rasterize_classes(load_rules(), lc_window)[:-1, :-1]  # drop apron row/col
    lc_full = np.repeat(np.repeat(lc, 2, axis=0), 2, axis=1)[:rows, :cols]
    forest = np.isin(lc_full, list(FOREST_CLASSES))

    peaks = _variable_window_maxima(h) & forest & (h >= MIN_TREE_H)
    ry, rx = np.nonzero(peaks)
    if ry.size == 0:
        z8, z16 = np.empty(0, np.uint8), np.empty(0, np.uint16)
        return TreeColumns(z16, z16, z8, z8, z8)

    heights = h[ry, rx]
    # world position of each crown (texel center); rows run north->south
    ex = e_min + (rx + 0.5) * NDSM_TEXEL
    nz = n_max - (ry + 0.5) * NDSM_TEXEL
    hash32 = _pcg_hash(rx, ry)

    species = _assign_species(ex, nz, hash32, stands, smap)
    ref = np.array([smap.ref_by_id.get(int(s), 24.0) for s in species])
    scale = np.clip(np.round(heights / ref * 64.0), 1, 255).astype(np.uint8)
    variant = (hash32 & 0xFF).astype(np.uint8)  # art-direction seed

    # chunk-local u16 quantization (x east from e_min, z south from n_max)
    q = 65535.0 / chunk_m
    xq = np.clip(np.round((ex - e_min) * q), 0, 65535).astype(np.uint16)
    zq = np.clip(np.round((n_max - nz) * q), 0, 65535).astype(np.uint16)

    order = np.argsort((zq.astype(np.uint32) << 16) | xq)  # row-major (proxy Morton)
    return TreeColumns(
        xq[order], zq[order], species[order].astype(np.uint8), scale[order], variant[order],
    )


def _assign_species(ex, nz, hash32, stands, smap: SpeciesMap) -> np.ndarray:
    geoms, codes, _ = stands
    out = np.full(ex.shape, -1, dtype=np.int16)
    if len(geoms):
        pts = shapely.points(ex, nz)
        tree = shapely.STRtree(geoms)
        hit_tree, hit_stand = tree.query(pts, predicate="within")
        # first stand wins per tree (deterministic: query order is stable)
        for ti, si in zip(hit_tree, hit_stand):
            if out[ti] != -1:
                continue
            out[ti] = _id_for_code(codes[si], smap)
    # fallback for trees outside any stand: pine/birch/spruce prior by hash
    missing = out == -1
    if missing.any():
        r = (hash32[missing] & 0xFFFF) / 65535.0
        out[missing] = np.where(
            r < 0.55, smap.code_id["MA"],
            np.where(r < 0.85, smap.code_id["KS"], smap.code_id["KU"]),
        )
    return out


def _id_for_code(code, smap: SpeciesMap) -> int:
    if code in smap.code_id:
        return smap.code_id[code]
    if code is not None:
        unmapped_species[str(code)] = unmapped_species.get(str(code), 0) + 1
    return smap.code_id[smap.fallback_conifer]
