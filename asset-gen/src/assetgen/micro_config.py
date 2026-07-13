"""Strict configuration for the cook-side microtopography recipe."""
from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG_DIR, BaseConfig
from .height_geom import footprint_m, texel_m

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class MicroConfig:
    recipe_version: int
    seed: int
    fine_lod: int
    parent_lod: int
    fine_texel_m: float
    supertile_m: int
    halo_m: int
    base_manifest: str
    base_manifest_sha256: str
    authority_lod: int
    reducer: str
    conservative_base: str
    qscale_candidates_m: tuple[float, ...]
    selected_fine_qscale_m: float
    parent_qscale_m: float
    codec: str
    exemplar_manifest: str
    exemplar_top_k: int
    placement_accumulator: str
    source_coverage_min_fraction: float
    allow_latest: bool


def _require(raw: dict, section: str, key: str):
    try:
        return raw[section][key]
    except KeyError as exc:
        raise ValueError(f"micro config missing [{section}].{key}") from exc


def load_micro_config(base: BaseConfig, path: Path | None = None) -> MicroConfig:
    source = path or CONFIG_DIR / "microtopography.toml"
    raw = tomllib.loads(source.read_text())
    cfg = MicroConfig(
        recipe_version=int(_require(raw, "format", "recipe_version")),
        seed=int(_require(raw, "format", "seed")),
        fine_lod=int(_require(raw, "format", "fine_lod")),
        parent_lod=int(_require(raw, "format", "parent_lod")),
        fine_texel_m=float(_require(raw, "format", "fine_texel_m")),
        supertile_m=int(_require(raw, "format", "supertile_m")),
        halo_m=int(_require(raw, "format", "halo_m")),
        base_manifest=str(_require(raw, "base", "manifest")),
        base_manifest_sha256=str(_require(raw, "base", "manifest_sha256")),
        authority_lod=int(_require(raw, "projection", "authority_lod")),
        reducer=str(_require(raw, "projection", "reducer")),
        conservative_base=str(_require(raw, "projection", "base")),
        qscale_candidates_m=tuple(float(v) for v in _require(raw, "encoding", "qscale_candidates_m")),
        selected_fine_qscale_m=float(_require(raw, "encoding", "selected_fine_qscale_m")),
        parent_qscale_m=float(_require(raw, "encoding", "parent_qscale_m")),
        codec=str(_require(raw, "encoding", "codec")),
        exemplar_manifest=str(_require(raw, "exemplars", "manifest")),
        exemplar_top_k=int(_require(raw, "exemplars", "top_k")),
        placement_accumulator=str(_require(raw, "exemplars", "placement_accumulator")),
        source_coverage_min_fraction=float(_require(raw, "coverage", "source_coverage_min_fraction")),
        allow_latest=bool(_require(raw, "publish", "allow_latest")),
    )
    _validate_micro_config(base, raw, cfg)
    return cfg


def _validate_micro_config(base: BaseConfig, raw: dict, cfg: MicroConfig) -> None:
    if cfg.recipe_version != 1:
        raise ValueError(f"unsupported micro recipe version {cfg.recipe_version}")
    if (cfg.fine_lod, cfg.parent_lod, cfg.authority_lod) != (-2, -1, 0):
        raise ValueError("micro v1 requires physical LODs -2, -1 and authority LOD 0")
    if not math.isclose(cfg.fine_texel_m, texel_m(cfg.fine_lod), abs_tol=0.0):
        raise ValueError("configured fine texel differs from the frozen signed height lattice")
    if footprint_m(base.grid, cfg.fine_lod) != 128.0:
        raise ValueError("micro v1 requires 128 m fine chunks on the frozen 2048-sample grid")
    if cfg.supertile_m != 256 or cfg.halo_m != 32:
        raise ValueError("micro v1 requires 256 m supertiles with 32 m halos")
    if not _SHA256_RE.fullmatch(cfg.base_manifest_sha256):
        raise ValueError("base manifest SHA-256 must be 64 lowercase hex characters")
    if cfg.reducer != "box_mean_16" or cfg.conservative_base != "conservative_cubic_bspline_v1":
        raise ValueError("unsupported projection recipe")
    if cfg.codec != base.encode.codec:
        raise ValueError("micro codec must match the base release codec")
    if cfg.qscale_candidates_m != (0.001, 0.002, 0.005, 0.01):
        raise ValueError("Stage-1 qscale candidates must remain 1/2/5/10 mm")
    if cfg.selected_fine_qscale_m not in cfg.qscale_candidates_m:
        raise ValueError("selected fine qscale is not a declared candidate")
    if cfg.parent_qscale_m != 0.005:
        raise ValueError("micro v1 parent qscale must be 5 mm")
    if cfg.placement_accumulator != "float64" or cfg.exemplar_top_k < 1:
        raise ValueError("invalid exemplar placement policy")
    if not 0.0 < cfg.source_coverage_min_fraction <= 1.0:
        raise ValueError("source coverage threshold must be in (0,1]")
    required_flags = (
        "require_child_parent",
        "require_rectangular_lod_minus2",
        "require_rectangular_lod_minus1",
        "require_complete_children_inside_minus2_rect",
    )
    if any(raw.get("coverage", {}).get(name) is not True for name in required_flags):
        raise ValueError("all micro v1 coverage invariants must be enabled")
    if cfg.allow_latest:
        raise ValueError("micro cooks may not publish latest.json")
