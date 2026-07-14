"""Bounded raw-observation comparison between full and thinned Hovi geometry."""

from .compare import (
    THINNED_LAZ_BYTES,
    THINNED_LAZ_SHA256,
    ComparisonProduct,
    compare_hovi_full_vs_thinned,
)
from .model import (
    HOVI_RESOLUTIONS_M,
    CommonGridPlan,
    MultiscaleObservables,
    frozen_hovi_grid_plan,
)

__all__ = [
    "CommonGridPlan",
    "ComparisonProduct",
    "HOVI_RESOLUTIONS_M",
    "MultiscaleObservables",
    "THINNED_LAZ_BYTES",
    "THINNED_LAZ_SHA256",
    "compare_hovi_full_vs_thinned",
    "frozen_hovi_grid_plan",
]
