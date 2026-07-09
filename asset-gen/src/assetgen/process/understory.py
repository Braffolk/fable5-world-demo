"""Understory community field: forest site type (kasvukohatüüp) + land cover -> per-cell
communityId + density. The client scatters ground flora from the community palette; we ship
only the field. Density (u8, 0..255 = fraction of the community's base plants/m^2) rises with
soil fertility (boniteet) and falls under very dense canopy or on bare classes.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio.features
import rasterio.transform
import shapely

from ..config import CONFIG_DIR, DATA_IN
from .landcover import BIOME_TEXEL

unmapped_site_types: dict[str, int] = {}


@dataclass(frozen=True)
class Communities:
    id_by_name: dict[str, int]
    base_density: dict[int, float]
    site_type: dict[str, str]
    land_cover: dict[int, str]
    dictionary: dict[int, dict]


@lru_cache
def load_communities() -> Communities:
    raw = tomllib.loads((CONFIG_DIR / "understory-communities.toml").read_text())
    comm = raw["communities"]
    id_by_name = {name: spec["id"] for name, spec in comm.items()}
    base_density = {spec["id"]: float(spec["base_density"]) for spec in comm.values()}
    dictionary = {
        spec["id"]: {"name": spec["name"], "moisture": spec["moisture"],
                     "base_density": spec["base_density"], "palette": spec["palette"]}
        for spec in comm.values()
    }
    land_cover = {int(k): v for k, v in raw["land_cover"].items()}
    return Communities(id_by_name, base_density, dict(raw["site_type"]), land_cover, dictionary)


def _stands_by_site_type(bbox_en) -> tuple[list, list]:
    files = sorted((DATA_IN / "wfs" / "metsaregister").glob("*eraldis.*.ndjson"))
    geoms, sites = [], []
    for f in files:
        for line in f.open():
            feat = json.loads(line)
            g = feat.get("geometry")
            if not g:
                continue
            geoms.append(shapely.geometry.shape(g))
            sites.append(feat["properties"].get("kasvukoht_kood"))
    return geoms, sites


def rasterize_understory(
    window_en: tuple[float, float, float, float, float],
    classid_plane: np.ndarray,
    vegdensity_plane: np.ndarray,
    boniteet_plane: np.ndarray,
) -> list[np.ndarray]:
    """(communityId u8, density u8). classid/vegdensity are the biome planes; boniteet the
    soil fertility plane — all at the same 2 m grid as this window."""
    comm = load_communities()
    e_min, n_min, e_max, n_max, t = window_en
    rows = round((n_max - n_min) / t)
    cols = round((e_max - e_min) / t)
    transform = rasterio.transform.from_origin(e_min, n_max, t, t)

    community = np.zeros((rows, cols), dtype=np.uint8)
    # 1) land-cover fallback everywhere
    for classid, name in comm.land_cover.items():
        community[classid_plane == classid] = comm.id_by_name[name]
    # 2) forest site-type polygons override where a stand exists
    geoms, sites = _stands_by_site_type((e_min, n_min, e_max, n_max))
    shapes = []
    for geom, site in zip(geoms, sites):
        if geom is None:
            continue
        name = comm.site_type.get(site)
        if name is None:
            if site is not None:
                unmapped_site_types[str(site)] = unmapped_site_types.get(str(site), 0) + 1
            name = "herb_mesic"
        shapes.append((geom, comm.id_by_name[name]))
    if shapes:
        site_raster = rasterio.features.rasterize(
            shapes, out_shape=(rows, cols), transform=transform, fill=255
        )
        has_site = site_raster != 255
        community[has_site] = site_raster[has_site].astype(np.uint8)

    # density: community base scaled to 0..255, lifted by fertility, trimmed under very dense
    # canopy (deep shade) and zeroed where the community is "none"
    base = np.array([comm.base_density.get(int(c), 0.0) for c in range(256)])
    dens = base[community] / max(base.max(), 1e-6)  # 0..1 by community richness
    fert = np.clip(boniteet_plane.astype(np.float32) / 60.0, 0.3, 1.3)  # boniteet ~0..60
    shade = 1.0 - 0.4 * np.clip(vegdensity_plane.astype(np.float32) / 255.0 - 0.6, 0, 0.4) / 0.4
    density = np.clip(dens * fert * shade * 255.0, 0, 255).astype(np.uint8)
    density[community == 0] = 0
    return [community, density]
