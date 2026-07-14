"""Composition of typed structural overrides onto an observation baseline."""
from __future__ import annotations

from .baseline import BaselineTile
from .closure import close_mixed_water_authority
from .model import StructuralTile, WaterSamples


def compose_water_bed(
    baseline: BaselineTile,
    water: WaterSamples,
) -> StructuralTile:
    """Replace only qualified wet samples and preserve all dry observations.

    ``WaterSamples`` is flat so callers can evaluate Shapely geometry in bounded
    stripes. Its ordering must be row-major for the baseline tile. Mixed
    supported/abstained wet regions receive the fail-closed C0 closure before
    composition. ``forbidden_morphology & ~unknown_bathymetry`` identifies wet
    abstention in the result.
    """
    return close_mixed_water_authority(baseline, water).tile
