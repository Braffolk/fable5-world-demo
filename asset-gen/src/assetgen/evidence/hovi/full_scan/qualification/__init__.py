"""Fail-closed authority for bounded Hovi full-scan reader validation."""

from .authority import PointProbeAuthority, load_point_probe_authority

__all__ = ["PointProbeAuthority", "load_point_probe_authority"]
