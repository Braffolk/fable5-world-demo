"""Exact signed-LOD closure plan for a corrected structural authority chunk."""
from __future__ import annotations

from dataclasses import dataclass

from ...config import GridConfig
from ...height_geom import (
    HeightChunkId,
    children_of,
    chunk_origin_en_units,
    footprint_units,
    parent_of,
)


@dataclass(frozen=True)
class StructuralRepairPlan:
    corrected_lod0: tuple[HeightChunkId, ...]
    authority_lod0: HeightChunkId
    review_parent: HeightChunkId
    lod1_reducer: tuple[HeightChunkId, ...]
    lod2_reducer: tuple[HeightChunkId, ...]
    authority_support: tuple[HeightChunkId, ...]
    published_lod2: tuple[HeightChunkId, ...]
    review_lod2_support: tuple[HeightChunkId, ...]

    def __post_init__(self) -> None:
        if (
            not self.corrected_lod0
            or len(set(self.corrected_lod0)) != len(self.corrected_lod0)
            or any(chunk.lod != 0 for chunk in self.corrected_lod0)
            or tuple(sorted(self.corrected_lod0)) != self.corrected_lod0
            or self.authority_lod0.lod != 0
            or self.review_parent.lod != -1
        ):
            raise ValueError("structural plan requires LOD0 authority and LOD-1 review parent")
        if parent_of(self.review_parent) != self.authority_lod0:
            raise ValueError("review parent does not belong to the authority LOD0 chunk")
        if self.authority_lod0 not in self.corrected_lod0:
            raise ValueError("review authority is absent from corrected LOD0 cores")
        expected_lod1 = tuple(
            sorted(
                {
                    child
                    for corrected in self.corrected_lod0
                    for child in _reducer_children(corrected)
                }
            )
        )
        expected_lod2 = tuple(
            sorted(
                {
                    child
                    for parent in expected_lod1
                    for child in _reducer_children(parent)
                }
            )
        )
        if self.lod1_reducer != expected_lod1:
            raise ValueError("LOD-1 reducer closure differs from corrected-core union")
        if self.lod2_reducer != expected_lod2:
            raise ValueError("LOD-2 reducer closure differs from corrected-core union")
        expected_authority = _one_tile_halo(self.lod2_reducer)
        if (
            len(set(self.authority_support)) != len(self.authority_support)
            or self.authority_support != expected_authority
        ):
            raise ValueError("authority support must be the exact one-tile reducer halo")
        if set(self.published_lod2) != set(children_of(self.review_parent)):
            raise ValueError("published LOD-2 set must be the review parent's exact children")
        if not set(self.published_lod2).issubset(self.lod2_reducer):
            raise ValueError("published LOD-2 chunks leave reducer support")
        if not set(self.review_lod2_support).issubset(self.lod2_reducer):
            raise ValueError("review LOD-2 apron support leaves reducer support")


def _reducer_children(parent: HeightChunkId) -> tuple[HeightChunkId, ...]:
    """Four-by-four children plus the east/south/southeast reduction apron."""
    core = children_of(parent)
    lod = parent.lod - 1
    cx0, cz0 = parent.cx * 4, parent.cz * 4
    support = tuple(
        [HeightChunkId(lod, cx0 + 4, cz0 + dz) for dz in range(4)]
        + [HeightChunkId(lod, cx0 + dx, cz0 + 4) for dx in range(4)]
        + [HeightChunkId(lod, cx0 + 4, cz0 + 4)]
    )
    return tuple(sorted((*core, *support)))


def _one_tile_halo(chunks: tuple[HeightChunkId, ...]) -> tuple[HeightChunkId, ...]:
    if not chunks or len({chunk.lod for chunk in chunks}) != 1:
        raise ValueError("authority halo requires one nonempty LOD")
    lod = chunks[0].lod
    cx0 = min(chunk.cx for chunk in chunks)
    cx1 = max(chunk.cx for chunk in chunks)
    cz0 = min(chunk.cz for chunk in chunks)
    cz1 = max(chunk.cz for chunk in chunks)
    expected_core = {
        HeightChunkId(lod, cx, cz)
        for cz in range(cz0, cz1 + 1)
        for cx in range(cx0, cx1 + 1)
    }
    if set(chunks) != expected_core:
        raise ValueError("authority halo requires a complete rectangular reducer set")
    return tuple(
        sorted(
            HeightChunkId(lod, cx, cz)
            for cz in range(cz0 - 1, cz1 + 2)
            for cx in range(cx0 - 1, cx1 + 2)
        )
    )


def plan_structural_repair(
    corrected_lod0: HeightChunkId | tuple[HeightChunkId, ...],
    review_parent: HeightChunkId,
) -> StructuralRepairPlan:
    """Derive complete decoded-child support without coordinate-specific constants."""
    corrected = (
        (corrected_lod0,)
        if isinstance(corrected_lod0, HeightChunkId)
        else tuple(sorted(corrected_lod0))
    )
    authority_lod0 = parent_of(review_parent)
    if (
        not corrected
        or any(chunk.lod != 0 for chunk in corrected)
        or review_parent.lod != -1
    ):
        raise ValueError("expected LOD0 authority and LOD-1 review parent")
    if authority_lod0 not in corrected:
        raise ValueError("review parent lies outside the corrected authority chunk")
    lod1 = tuple(
        sorted({child for chunk in corrected for child in _reducer_children(chunk)})
    )
    lod2 = tuple(sorted({child for parent in lod1 for child in _reducer_children(parent)}))
    authority_support = _one_tile_halo(lod2)
    published = tuple(sorted(children_of(review_parent)))
    review_support = tuple(sorted(set(_reducer_children(review_parent)) - set(published)))
    return StructuralRepairPlan(
        corrected_lod0=corrected,
        authority_lod0=authority_lod0,
        review_parent=review_parent,
        lod1_reducer=lod1,
        lod2_reducer=lod2,
        authority_support=authority_support,
        published_lod2=published,
        review_lod2_support=review_support,
    )


def chunk_set_bounds_en_units(
    grid: GridConfig, chunks: tuple[HeightChunkId, ...]
) -> tuple[int, int, int, int]:
    """Return half-open `(e_min,n_min,e_max,n_max)` bounds for one LOD chunk set."""
    if not chunks or len({chunk.lod for chunk in chunks}) != 1:
        raise ValueError("chunk bounds require a nonempty single-LOD set")
    origins = [chunk_origin_en_units(grid, chunk) for chunk in chunks]
    footprint = footprint_units(grid, chunks[0].lod)
    return (
        min(origin[0] for origin in origins),
        min(origin[1] - footprint for origin in origins),
        max(origin[0] + footprint for origin in origins),
        max(origin[1] for origin in origins),
    )
