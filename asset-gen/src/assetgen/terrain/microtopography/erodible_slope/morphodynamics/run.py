"""Strict recipe-only entry point for Development A morphodynamics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .artifact import MorphodynamicsArtifact, materialize_bound_result
from .assemble import BoundDevelopmentA, load_bound_development_a
from .solver import MorphodynamicsResult, solve_morphodynamics


@dataclass(frozen=True)
class BoundMorphodynamicsRun:
    bound: BoundDevelopmentA
    result: MorphodynamicsResult
    artifact: MorphodynamicsArtifact


def run_bound_development_a(
    *,
    recipe_path: Path,
    expected_recipe_sha256: str,
    workspace_root: Path,
    artifact_root: Path,
) -> BoundMorphodynamicsRun:
    """Load, solve, and immutably materialize only the caller-bound recipe."""
    bound = load_bound_development_a(
        recipe_path,
        expected_recipe_sha256=expected_recipe_sha256,
    )
    workspace = Path(workspace_root) / bound.semantic_sha256
    workspace.mkdir(parents=True, exist_ok=True)
    result = solve_morphodynamics(
        bound.authority,
        bound.process,
        bound.canonical_boundary,
        bound.control_boundary,
        bound.canonical_organization,
        bound.control_organization,
        workspace=workspace,
        config=bound.recipe.config,
    )
    artifact = materialize_bound_result(bound, result, artifact_root)
    return BoundMorphodynamicsRun(bound=bound, result=result, artifact=artifact)
