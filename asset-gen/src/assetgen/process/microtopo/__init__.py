"""Measured, deterministic cook-side microtopography."""

from .model import GroundSurface, PatchBank
from .api import load_exemplar_bank, prepare_lapinjarvi_bank, synthesize_residual
from .preprocess import (
    LAPINJARVI_ROLES,
    GroundExtractionConfig,
    build_residual_bank,
    extract_ground_surface,
    load_laz_xyz,
    prepare_lapinjarvi_laz,
    prepare_lapinjavi_laz,
)
from .synthesis import synthesize_measured

__all__ = [
    "GroundExtractionConfig",
    "GroundSurface",
    "LAPINJARVI_ROLES",
    "PatchBank",
    "build_residual_bank",
    "extract_ground_surface",
    "load_laz_xyz",
    "load_exemplar_bank",
    "prepare_lapinjarvi_bank",
    "prepare_lapinjarvi_laz",
    "prepare_lapinjavi_laz",
    "synthesize_measured",
    "synthesize_residual",
]
