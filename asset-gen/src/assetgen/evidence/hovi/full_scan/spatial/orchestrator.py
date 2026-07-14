"""Compatibility exports for the spatial execution-selection boundary."""

from .selection import (
    DEFAULT_SPATIAL_OUTPUT_ROOT,
    SpatialExecutionSelection,
    select_spatial_execution,
    spatial_materialization_plan,
)

__all__ = [
    "DEFAULT_SPATIAL_OUTPUT_ROOT",
    "SpatialExecutionSelection",
    "select_spatial_execution",
    "spatial_materialization_plan",
]
