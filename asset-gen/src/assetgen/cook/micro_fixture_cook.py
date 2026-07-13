"""Cook the Stage-1 geometric-retention fixture into LAC2 artifacts.

This is deliberately not the production morphology synthesizer.  It proves the
signed hierarchy, conservative base, quantization, streaming, and rendering path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

from ..config import BaseConfig
from ..height_geom import (
    HeightChunkId,
    HeroCoverage,
    chunk_origin_en_units,
    plan_hero,
    texel_m,
)
from ..micro_config import MicroConfig
from ..process.micro_fixture import (
    calibrated_fixture_residual,
    conservative_cell_correct,
    project_residual_zero_mean,
)
from .chunkio import ChunkMeta, read_chunk_v2, write_chunk_v2
from .encode import decode_quant16, encode_quant16_checked
from .micro_hierarchy import (
    assemble_parent_source_memmap,
    box_mean4_striped,
    dependency_merkle_root,
)
from .pinned_height import PinnedBaseHeight

FIXTURE_COOK_REV = 1
REFINEMENT = 16


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_path(root: Path, chunk: HeightChunkId) -> Path:
    return root / "height" / str(chunk.lod) / f"{chunk.cx}_{chunk.cz}.lac"


def _load_expectation(build_root: Path, build_digest: str) -> dict:
    blob = (build_root / "expectation.json").read_bytes()
    if hashlib.sha256(blob).hexdigest() != (build_root / "expectation.sha256").read_text().strip():
        raise ValueError("micro expectation digest mismatch")
    value = json.loads(blob)
    if value.get("recipeSha256") != build_digest:
        raise ValueError("micro expectation belongs to another recipe")
    return value


def _fit_global_base(
    reader: PinnedBaseHeight,
    base: BaseConfig,
    coverage: HeroCoverage,
    margin_cells: int = 4,
) -> RectBivariateSpline:
    first = coverage.published_fine[0]
    origin_e_u, origin_n_u = chunk_origin_en_units(base.grid, first)
    origin_e, origin_n = origin_e_u // 32, origin_n_u // 32
    domain_cells = 5 * 128
    e_min = origin_e - margin_cells
    n_max = origin_n + margin_cells
    side = domain_cells + 2 * margin_cells
    authority = reader.read_cells(e_min, n_max, side, side).astype(np.float64)
    east = e_min + np.arange(side, dtype=np.float64) + 0.5
    south = base.grid.anchor_n - n_max + np.arange(side, dtype=np.float64) + 0.5
    return RectBivariateSpline(south, east, authority, kx=3, ky=3, s=0.0)


def _fixture_chunk_surface(
    reader: PinnedBaseHeight,
    spline: RectBivariateSpline,
    base: BaseConfig,
    chunk: HeightChunkId,
) -> np.ndarray:
    if chunk.lod != -2:
        raise ValueError("fixture surface is defined only for LOD -2")
    origin_e_u, origin_n_u = chunk_origin_en_units(base.grid, chunk)
    origin_e, origin_n = origin_e_u / 32.0, origin_n_u / 32.0
    texel = texel_m(-2)
    fine_side = 129 * REFINEMENT
    east = origin_e + (np.arange(fine_side, dtype=np.float64) + 0.5) * texel
    south = base.grid.anchor_n - origin_n + (
        np.arange(fine_side, dtype=np.float64) + 0.5
    ) * texel
    smooth = spline(south, east, grid=True)
    authority = reader.read_cells(int(origin_e), int(origin_n), 129, 129)
    conservative = conservative_cell_correct(smooth, authority, REFINEMENT)
    e_grid, s_grid = np.meshgrid(east, south)
    n_grid = base.grid.anchor_n - s_grid
    residual = calibrated_fixture_residual(e_grid, n_grid)
    residual = project_residual_zero_mean(residual, REFINEMENT)
    surface = conservative + residual
    return np.ascontiguousarray(surface[:2049, :2049], dtype=np.float64)


def _write_height(
    path: Path,
    base: BaseConfig,
    chunk: HeightChunkId,
    values: np.ndarray,
    qscale: float,
    qoffset: float | None = None,
) -> dict:
    payload, qoffset, wire_qscale = encode_quant16_checked(
        base.encode, values, qscale, qoffset
    )
    decoded = decode_quant16(
        base.encode, payload, values.shape[0], qoffset, wire_qscale
    )
    error = float(np.max(np.abs(decoded.astype(np.float64) - values)))
    decoded_peak = np.float32(np.max(np.abs(decoded)))
    float32_allowance = 2.0 * abs(float(np.spacing(decoded_peak)))
    if error > wire_qscale * 0.5 + float32_allowance:
        raise AssertionError(f"checked quantization error {error} on {chunk}")
    origin_e_u, origin_n_u = chunk_origin_en_units(base.grid, chunk)
    meta = ChunkMeta(
        layer="height",
        lod=chunk.lod,
        enc=1,
        cx=chunk.cx,
        cz=chunk.cz,
        res=values.shape[0],
        count=0,
        origin_e=origin_e_u / 32.0,
        origin_n=origin_n_u / 32.0,
        qoffset=qoffset,
        qscale=wire_qscale,
    )
    write_chunk_v2(path, meta, payload)
    return {
        "key": [chunk.lod, chunk.cx, chunk.cz],
        "path": str(path),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
        "qoffset": qoffset,
        "qscale": wire_qscale,
        "maxRoundTripErrorM": error,
        "roundTripLimitM": wire_qscale * 0.5 + float32_allowance,
    }


def cook_micro_fixture(
    base: BaseConfig,
    micro: MicroConfig,
    build_digest: str,
    base_manifest_path: Path,
    base_out_root: Path,
    work_root: Path,
    *,
    log=print,
) -> Path:
    """Cook 16 publishable fine chunks, nine support chunks, and one parent."""
    build_root = work_root / "builds" / build_digest
    expectation = _load_expectation(build_root, build_digest)
    parent_key = expectation["parent"]
    coverage = plan_hero(int(parent_key[1]), int(parent_key[2]))
    if expectation["publishedFine"] != [[c.lod, c.cx, c.cz] for c in coverage.published_fine]:
        raise ValueError("expectation fine closure differs from geometry plan")
    if expectation["transientSupport"] != [[c.lod, c.cx, c.cz] for c in coverage.transient_support]:
        raise ValueError("expectation support closure differs from geometry plan")

    reader = PinnedBaseHeight(
        base_manifest_path,
        micro.base_manifest_sha256,
        base_out_root,
        base.encode,
    )
    spline = _fit_global_base(reader, base, coverage)
    first_e_u, first_n_u = chunk_origin_en_units(base.grid, coverage.published_fine[0])
    domain_authority = reader.read_cells(first_e_u // 32, first_n_u // 32, 640, 640)
    fine_qoffset = float(np.floor(float(domain_authority.min()) - 2.0))
    staged_root = build_root / "chunks"
    transient_root = build_root / "transient"
    evidence_root = build_root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict] = []
    all_fine = (*coverage.published_fine, *coverage.transient_support)
    published = set(coverage.published_fine)
    for index, chunk in enumerate(all_fine, start=1):
        surface = _fixture_chunk_surface(reader, spline, base, chunk)
        root = staged_root if chunk in published else transient_root
        artifact = _write_height(
            _chunk_path(root, chunk), base, chunk, surface,
            micro.selected_fine_qscale_m, fine_qoffset,
        )
        artifacts.append(artifact)
        log(f"fixture fine [{index}/25] {chunk}: {artifact['size'] / 1e6:.2f} MB")

    by_chunk = {
        HeightChunkId(*artifact["key"]): artifact for artifact in artifacts
    }

    def load_decoded(chunk: HeightChunkId) -> np.ndarray:
        meta, payload = read_chunk_v2(Path(by_chunk[chunk]["path"]))
        return decode_quant16(base.encode, payload, meta.res, meta.qoffset, meta.qscale)

    mosaic_path = build_root / "scratch" / "parent-source.f32"
    mosaic = assemble_parent_source_memmap(mosaic_path, coverage, load_decoded)
    parent_values = box_mean4_striped(mosaic)
    parent_artifact = _write_height(
        _chunk_path(staged_root, coverage.parent),
        base,
        coverage.parent,
        parent_values,
        micro.parent_qscale_m,
    )
    dependencies = [(chunk, by_chunk[chunk]["sha256"]) for chunk in all_fine]
    sidecar = {
        "format": 1,
        "fixture": "calibrated-retention-v1-not-production-synthesis",
        "recipeSha256": build_digest,
        "parent": parent_artifact,
        "children": artifacts,
        "dependencyMerkleRoot": dependency_merkle_root(dependencies),
    }
    sidecar_path = evidence_root / "fixture-cook.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=1, sort_keys=True) + "\n")
    log(f"fixture parent {coverage.parent}: {parent_artifact['size'] / 1e6:.2f} MB")
    return sidecar_path
