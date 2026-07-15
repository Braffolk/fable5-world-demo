"""Bound entry point for one Development-A amplification candidate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..morphodynamics.assemble import BoundDevelopmentA, load_bound_development_a
from .artifact import AmplificationArtifact, materialize
from .model import AmplificationResult
from .solver import solve


@dataclass(frozen=True)
class AmplificationRun:
    bound: BoundDevelopmentA
    result: AmplificationResult
    artifact: AmplificationArtifact


def run_development_a(
    *, recipe_path: Path, expected_recipe_sha256: str, artifact_root: Path
) -> AmplificationRun:
    bound = load_bound_development_a(
        recipe_path, expected_recipe_sha256=expected_recipe_sha256
    )
    result = solve(bound)
    artifact = materialize(bound, result, artifact_root)
    return AmplificationRun(bound=bound, result=result, artifact=artifact)
