"""Real mapped boulders from ETAK E_101_kivi_p -> explicit instance records (like trees).

These ARE discrete, surveyed features (Üksik kivi = single boulder, Kivihunnik = boulder
pile), so exact positions are worth storing. Record: x,z u16 chunk-local, kind u8
(0 = single, 1 = pile), size u8 (from korgus, else nominal), variant u8. No y — client grounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from .etak_read import etak_gpkg, read_layer_window

LAYER = "E_101_kivi_p"


@dataclass
class BoulderColumns:
    x: np.ndarray
    z: np.ndarray
    kind: np.ndarray
    size: np.ndarray
    variant: np.ndarray

    def __len__(self) -> int:
        return len(self.x)


def _pcg(x, y):
    h = (x.astype(np.uint32) * np.uint32(747796405)) ^ (y.astype(np.uint32) * np.uint32(2891336453))
    h = (h ^ (h >> 15)) * np.uint32(2246822519)
    return (h ^ (h >> 13)) & np.uint32(0xFFFFFFFF)


def boulders_for_chunk(chunk_bounds_en, chunk_m: int) -> BoulderColumns:
    e_min, n_min, e_max, n_max = chunk_bounds_en
    geoms, cols = read_layer_window(
        etak_gpkg(), LAYER, (e_min, n_min, e_max, n_max), fields=["tyyp", "korgus"]
    )
    empty16, empty8 = np.empty(0, np.uint16), np.empty(0, np.uint8)
    if not len(geoms):
        return BoulderColumns(empty16, empty16, empty8, empty8, empty8)

    ex, nz, kind, size = [], [], [], []
    for i, g in enumerate(geoms):
        if g is None:
            continue
        pt = g if g.geom_type == "Point" else g.centroid
        ex.append(pt.x)
        nz.append(pt.y)
        kind.append(1 if str(cols["tyyp"][i]) == "20" else 0)  # 20=pile, 10=single
        korgus = cols["korgus"][i]
        h = float(korgus) if korgus not in (None, 0, "0") else (2.0 if kind[-1] else 1.0)
        size.append(int(np.clip(h * 40, 8, 255)))  # ~0.2..6 m mapped to u8
    ex = np.array(ex)
    nz = np.array(nz)
    q = 65535.0 / chunk_m
    xq = np.clip(np.round((ex - e_min) * q), 0, 65535).astype(np.uint16)
    zq = np.clip(np.round((n_max - nz) * q), 0, 65535).astype(np.uint16)
    variant = (_pcg((ex * 100).astype(np.int64), (nz * 100).astype(np.int64)) & 0xFF).astype(np.uint8)
    order = np.argsort((zq.astype(np.uint32) << 16) | xq)
    return BoulderColumns(
        xq[order], zq[order], np.array(kind, np.uint8)[order],
        np.array(size, np.uint8)[order], variant[order],
    )
