"""Agriculture v2: operation-state cultivated roughness with real dual scales.

Isolated from the rejected v1 module (terrain/microtopography/agriculture); shares
only read-only repair/height scaffolding. Development float + gate + QA only.
"""
from .run import run

__all__ = ["run"]
