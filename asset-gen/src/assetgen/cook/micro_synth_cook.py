"""Cook the exemplar-driven production microtopography pilot into LAC2 artifacts.

The generator owns morphology only.  Conservative reconstruction, mean-null
projection, quantization, support closure, and parent derivation remain the same
packing boundary proven by the Stage-1 fixture cook.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

from ..config import ASSET_GEN_ROOT, BaseConfig
from ..height_geom import (
    HeightChunkId,
    HeroCoverage,
    chunk_origin_en_units,
    plan_hero,
    texel_m,
)
from ..micro_config import MicroConfig
from ..micro_recipe import _synthesis_environment_identity
from ..process.micro_masks import (
    CLIFF_CONTEXT_BUFFER_M,
    MESIC_BONITEET_RANGE,
    SUPPORTED_MINERAL_SOIL_TYPES,
    rasterize_micro_morphology_mask,
)
from ..process.micro_fixture import (
    conservative_cell_correct,
)
from ..process.microtopo import load_exemplar_bank, synthesize_residual
from ..process.microtopo.projection import (
    SmoothMeanNullProjector,
    build_smooth_mean_null_projector,
)
from .chunkio import ChunkMeta, read_chunk_v2, write_chunk_v2
from .encode import decode_quant16, encode_quant16_checked
from .micro_hierarchy import (
    assemble_parent_source_memmap,
    box_mean4_striped,
    dependency_merkle_root,
)
from .pinned_height import PinnedBaseHeight

SYNTHESIS_COOK_REV = 3
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
    expected_sha = (build_root / "expectation.sha256").read_text().strip()
    if hashlib.sha256(blob).hexdigest() != expected_sha:
        raise ValueError("micro expectation digest mismatch")
    value = json.loads(blob)
    if value.get("recipeSha256") != build_digest:
        raise ValueError("micro expectation belongs to another recipe")
    expected_environment = value.get("recipeInputs", {}).get("environment")
    if expected_environment != _synthesis_environment_identity(ASSET_GEN_ROOT):
        raise ValueError("micro expectation belongs to another numerical environment")
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


def _synth_chunk_residual(
    bank,
    base: BaseConfig,
    chunk: HeightChunkId,
    seed: int,
    top_k: int,
) -> tuple[np.ndarray, object]:
    if chunk.lod != -2:
        raise ValueError("production synthesis surface is defined only for LOD -2")
    origin_e_u, origin_n_u = chunk_origin_en_units(base.grid, chunk)
    origin_e, origin_n = origin_e_u / 32.0, origin_n_u / 32.0
    fine_side = 129 * REFINEMENT
    texel = texel_m(chunk.lod)
    east = origin_e + (np.arange(fine_side, dtype=np.float64) + 0.5) * texel
    north = origin_n - (np.arange(fine_side, dtype=np.float64) + 0.5) * texel
    residual = np.asarray(
        synthesize_residual(bank, east, north, seed, top_k=top_k), dtype=np.float64
    )
    if residual.shape != (fine_side, fine_side) or not np.isfinite(residual).all():
        raise ValueError(
            "microtopography synthesizer returned an invalid residual: "
            f"{residual.dtype} {residual.shape}"
        )
    morphology_mask = rasterize_micro_morphology_mask(east, north)
    return residual, morphology_mask


def _synth_chunk_surface(
    reader: PinnedBaseHeight,
    spline: RectBivariateSpline,
    bank,
    base: BaseConfig,
    chunk: HeightChunkId,
    seed: int,
    top_k: int,
    projector: SmoothMeanNullProjector,
    projection_row0: int,
    projection_col0: int,
    amplitude_limit_m: float,
    slope_limit: float,
) -> np.ndarray:
    origin_e_u, origin_n_u = chunk_origin_en_units(base.grid, chunk)
    origin_e, origin_n = origin_e_u / 32.0, origin_n_u / 32.0
    fine_side = 129 * REFINEMENT
    texel = texel_m(chunk.lod)
    east = origin_e + (np.arange(fine_side, dtype=np.float64) + 0.5) * texel
    north = origin_n - (np.arange(fine_side, dtype=np.float64) + 0.5) * texel
    south = base.grid.anchor_n - north

    smooth = spline(south, east, grid=True)
    authority = reader.read_cells(int(origin_e), int(origin_n), 129, 129)
    conservative = conservative_cell_correct(smooth, authority, REFINEMENT)
    residual, morphology_mask = _synth_chunk_residual(
        bank, base, chunk, seed, top_k
    )
    residual = projector.project_window(
        residual,
        morphology_mask.allowed,
        row0=projection_row0,
        col0=projection_col0,
    )
    if np.any(residual[~morphology_mask.allowed] != 0.0):
        raise AssertionError("micro residual is nonzero outside supported morphology")
    max_abs = float(np.max(np.abs(residual)))
    max_slope = float(max(
        np.max(np.abs(np.diff(residual, axis=0))),
        np.max(np.abs(np.diff(residual, axis=1))),
    ) / texel)
    if max_abs > amplitude_limit_m or max_slope > slope_limit:
        raise ValueError(
            "projected morphology exceeds the measured bank envelope: "
            f"amplitude={max_abs} slope={max_slope}"
        )
    surface = conservative + residual
    return np.ascontiguousarray(surface[:2049, :2049], dtype=np.float64)


def _measured_envelope(bank) -> tuple[float, float]:
    patches = np.asarray(bank.patches_m, dtype=np.float64)
    amplitude = float(np.max(np.abs(patches)))
    slopes = np.concatenate((
        np.abs(np.diff(patches, axis=1)).ravel(),
        np.abs(np.diff(patches, axis=2)).ravel(),
    )) / float(bank.texel_m)
    return amplitude, float(np.max(slopes))


def _unsafe_projection_cells(
    residual: np.ndarray,
    amplitude_limit_m: float,
    slope_limit: float,
    texel: float,
) -> np.ndarray:
    """Identify every authority cell touching an out-of-bank fine sample or edge."""
    rows = residual.shape[0] // REFINEMENT
    cols = residual.shape[1] // REFINEMENT
    unsafe = (
        np.abs(residual).reshape(rows, REFINEMENT, cols, REFINEMENT)
        > amplitude_limit_m
    ).any(axis=(1, 3))
    step_limit = slope_limit * texel
    for axis in (0, 1):
        edge_y, edge_x = np.nonzero(np.abs(np.diff(residual, axis=axis)) > step_limit)
        unsafe[edge_y // REFINEMENT, edge_x // REFINEMENT] = True
        if axis == 0:
            unsafe[(edge_y + 1) // REFINEMENT, edge_x // REFINEMENT] = True
        else:
            unsafe[edge_y // REFINEMENT, (edge_x + 1) // REFINEMENT] = True
    return unsafe


def _plan_global_projection(
    chunks: tuple[HeightChunkId, ...],
    bank,
    base: BaseConfig,
    seed: int,
    top_k: int,
) -> tuple[SmoothMeanNullProjector, dict[HeightChunkId, dict], int, int]:
    """Measure one 5x5 closure, including the shared east/south apron cell."""
    min_cx = min(chunk.cx for chunk in chunks)
    max_cx = max(chunk.cx for chunk in chunks)
    min_cz = min(chunk.cz for chunk in chunks)
    max_cz = max(chunk.cz for chunk in chunks)
    rows = (max_cz - min_cz + 1) * 128 + 1
    cols = (max_cx - min_cx + 1) * 128 + 1
    fully_soft = np.zeros((rows, cols), dtype=bool)
    filled = np.zeros((rows, cols), dtype=bool)
    evidence: dict[HeightChunkId, dict] = {}
    for chunk in chunks:
        residual, morphology_mask = _synth_chunk_residual(
            bank, base, chunk, seed, top_k
        )
        cell_soft = morphology_mask.allowed.reshape(
            129, REFINEMENT, 129, REFINEMENT
        ).all(axis=(1, 3))
        row0 = (chunk.cz - min_cz) * 128
        col0 = (chunk.cx - min_cx) * 128
        region = np.s_[row0 : row0 + 129, col0 : col0 + 129]
        overlap = filled[region]
        if np.any(overlap):
            if not np.array_equal(cell_soft[overlap], fully_soft[region][overlap]):
                raise AssertionError("projection mask differs across shared chunk cells")
        fully_soft[region] = np.where(overlap, fully_soft[region], cell_soft)
        filled[region] = True
        evidence[chunk] = morphology_mask.evidence()
    if not filled.all():
        raise AssertionError("global projection closure is incomplete")
    amplitude_limit_m, slope_limit = _measured_envelope(bank)

    def build_projector() -> SmoothMeanNullProjector:
        support = build_smooth_mean_null_projector(
            np.zeros((rows, cols), dtype=np.float64), fully_soft, factor=REFINEMENT
        )
        means = np.full((rows, cols), np.nan, dtype=np.float64)
        filled.fill(False)
        for chunk in chunks:
            residual, morphology_mask = _synth_chunk_residual(
                bank, base, chunk, seed, top_k
            )
            row0 = (chunk.cz - min_cz) * 128
            col0 = (chunk.cx - min_cx) * 128
            taper = support.taper_window(row0, col0, 129, 129)
            tapered = np.where(morphology_mask.allowed, residual * taper, 0.0)
            cell_means = tapered.reshape(
                129, REFINEMENT, 129, REFINEMENT
            ).mean(axis=(1, 3))
            region = np.s_[row0 : row0 + 129, col0 : col0 + 129]
            overlap = filled[region]
            if np.any(overlap) and not np.allclose(
                cell_means[overlap], means[region][overlap], rtol=0, atol=1e-15
            ):
                raise AssertionError("projection means differ across shared chunk cells")
            means[region] = np.where(overlap, means[region], cell_means)
            filled[region] = True
        if not filled.all() or not np.isfinite(means).all():
            raise AssertionError("global projection mean field is incomplete")
        return build_smooth_mean_null_projector(
            means, fully_soft, factor=REFINEMENT
        )

    # Projection can add a small smooth correction beyond a source patch's
    # extrema. Reject the responsible authority cells and rebuild instead of
    # clipping the residual or introducing an amplitude control.
    for _ in range(4):
        projector = build_projector()
        unsafe = np.zeros((rows, cols), dtype=bool)
        for chunk in chunks:
            residual, morphology_mask = _synth_chunk_residual(
                bank, base, chunk, seed, top_k
            )
            row0 = (chunk.cz - min_cz) * 128
            col0 = (chunk.cx - min_cx) * 128
            projected = projector.project_window(
                residual, morphology_mask.allowed, row0=row0, col0=col0
            )
            local_unsafe = _unsafe_projection_cells(
                projected,
                amplitude_limit_m,
                slope_limit,
                texel_m(chunk.lod),
            )
            region = np.s_[row0 : row0 + 129, col0 : col0 + 129]
            unsafe[region] |= local_unsafe
        unsafe &= fully_soft
        if not np.any(unsafe):
            return projector, evidence, min_cx, min_cz
        fully_soft[unsafe] = False
    raise ValueError("projection envelope did not converge after four fail-closed passes")


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
    limit = wire_qscale * 0.5 + float32_allowance
    if error > limit:
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
        "roundTripLimitM": limit,
    }


def _decoded_height(path: Path, base: BaseConfig) -> np.ndarray:
    meta, payload = read_chunk_v2(path)
    return decode_quant16(base.encode, payload, meta.res, meta.qoffset, meta.qscale)


def _max_decoded_seam(
    coverage: HeroCoverage,
    paths: dict[HeightChunkId, Path],
    base: BaseConfig,
) -> float:
    """Measure the 5x5 closure with at most two decoded chunks resident."""
    cx0 = coverage.parent.cx * 4
    cz0 = coverage.parent.cz * 4
    maximum = 0.0
    for dz in range(5):
        previous = _decoded_height(paths[HeightChunkId(-2, cx0, cz0 + dz)], base)
        for dx in range(1, 5):
            current = _decoded_height(
                paths[HeightChunkId(-2, cx0 + dx, cz0 + dz)], base
            )
            maximum = max(
                maximum,
                float(np.max(np.abs(previous[:, -1] - current[:, 0]))),
            )
            previous = current
    for dx in range(5):
        previous = _decoded_height(paths[HeightChunkId(-2, cx0 + dx, cz0)], base)
        for dz in range(1, 5):
            current = _decoded_height(
                paths[HeightChunkId(-2, cx0 + dx, cz0 + dz)], base
            )
            maximum = max(
                maximum,
                float(np.max(np.abs(previous[-1, :] - current[0, :]))),
            )
            previous = current
    return maximum


def cook_micro_synthesis(
    base: BaseConfig,
    micro: MicroConfig,
    build_digest: str,
    base_manifest_path: Path,
    base_out_root: Path,
    work_root: Path,
    exemplar_manifest_path: Path,
    *,
    log=print,
) -> Path:
    """Cook 16 published, nine support, and one derived-parent pilot chunks."""
    build_root = work_root / "builds" / build_digest
    expectation = _load_expectation(build_root, build_digest)
    if expectation.get("recipeKind") != "measured-synthesis-pilot":
        raise ValueError("production synthesis cook requires a measured-synthesis-pilot expectation")
    recipe_inputs = expectation.get("recipeInputs", {})
    if not isinstance(recipe_inputs.get("soilSourceSha256"), dict):
        raise ValueError("production synthesis recipe does not bind Mullakaart inputs")
    parent_key = expectation["parent"]
    coverage = plan_hero(int(parent_key[1]), int(parent_key[2]))
    if expectation["publishedFine"] != [
        [c.lod, c.cx, c.cz] for c in coverage.published_fine
    ]:
        raise ValueError("expectation fine closure differs from geometry plan")
    if expectation["transientSupport"] != [
        [c.lod, c.cx, c.cz] for c in coverage.transient_support
    ]:
        raise ValueError("expectation support closure differs from geometry plan")

    bank = load_exemplar_bank(exemplar_manifest_path)
    reader = PinnedBaseHeight(
        base_manifest_path,
        micro.base_manifest_sha256,
        base_out_root,
        base.encode,
    )
    spline = _fit_global_base(reader, base, coverage)
    first_e_u, first_n_u = chunk_origin_en_units(
        base.grid, coverage.published_fine[0]
    )
    domain_authority = reader.read_cells(
        first_e_u // 32, first_n_u // 32, 640, 640
    )
    fine_qoffset = float(np.floor(float(domain_authority.min()) - 2.0))

    staged_root = build_root / "chunks"
    transient_root = build_root / "transient"
    evidence_root = build_root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    all_fine = (*coverage.published_fine, *coverage.transient_support)
    published = set(coverage.published_fine)
    projector, mask_by_chunk, projection_min_cx, projection_min_cz = (
        _plan_global_projection(
            all_fine, bank, base, micro.seed, micro.exemplar_top_k
        )
    )
    amplitude_limit_m, slope_limit = _measured_envelope(bank)
    artifacts: list[dict] = []
    mask_evidence: list[dict] = []
    paths: dict[HeightChunkId, Path] = {}
    for index, chunk in enumerate(all_fine, start=1):
        surface = _synth_chunk_surface(
            reader,
            spline,
            bank,
            base,
            chunk,
            micro.seed,
            micro.exemplar_top_k,
            projector,
            (chunk.cz - projection_min_cz) * 128,
            (chunk.cx - projection_min_cx) * 128,
            amplitude_limit_m,
            slope_limit,
        )
        chunk_mask_evidence = mask_by_chunk[chunk]
        chunk_mask_evidence["key"] = [chunk.lod, chunk.cx, chunk.cz]
        mask_evidence.append(chunk_mask_evidence)
        root = staged_root if chunk in published else transient_root
        path = _chunk_path(root, chunk)
        artifact = _write_height(
            path,
            base,
            chunk,
            surface,
            micro.selected_fine_qscale_m,
            fine_qoffset,
        )
        artifacts.append(artifact)
        paths[chunk] = path
        log(f"synthesis fine [{index}/25] {chunk}: {artifact['size'] / 1e6:.2f} MB")

    by_chunk = {HeightChunkId(*item["key"]): item for item in artifacts}

    def load_decoded(chunk: HeightChunkId) -> np.ndarray:
        return _decoded_height(Path(by_chunk[chunk]["path"]), base)

    mosaic = assemble_parent_source_memmap(
        build_root / "scratch" / "synthesis-parent-source.f32",
        coverage,
        load_decoded,
    )
    parent_values = box_mean4_striped(mosaic)
    parent_artifact = _write_height(
        _chunk_path(staged_root, coverage.parent),
        base,
        coverage.parent,
        parent_values,
        micro.parent_qscale_m,
    )

    max_seam = _max_decoded_seam(coverage, paths, base)
    if max_seam != 0.0:
        raise AssertionError(f"decoded fine apron seam is {max_seam} m")
    max_round_trip = max(
        item["maxRoundTripErrorM"] for item in (*artifacts, parent_artifact)
    )
    dependencies = [(chunk, by_chunk[chunk]["sha256"]) for chunk in all_fine]
    evidence = {
        "format": 1,
        "cook": "exemplar-driven-production-pilot-v1",
        "cookRevision": SYNTHESIS_COOK_REV,
        "recipeSha256": build_digest,
        "seed": micro.seed,
        "exemplarTopK": micro.exemplar_top_k,
        "exemplarManifest": {
            "path": str(exemplar_manifest_path.resolve()),
            "sha256": _sha256_file(exemplar_manifest_path),
        },
        "maskIntegration": {
            "status": "etak-soil-fail-closed-fine-v2",
            "applied": True,
            "analogueStatus": "foreign-analogue-low-confidence",
            "semantics": [
                "exact-etak-forest",
                "known-unmodified-mineral-non-peat-soil",
                "productive-mesic-boniteet",
                "etak-slope-cliff-context-excluded",
                "open-water-excluded",
                "building-excluded",
                "paved-road-excluded",
            ],
            "policy": {
                "supportedMineralSoilTypeIds": sorted(SUPPORTED_MINERAL_SOIL_TYPES),
                "boniteetInclusive": list(MESIC_BONITEET_RANGE),
                "slopeCliffBufferM": CLIFF_CONTEXT_BUFFER_M,
                "textureUse": "exclusion-only-explicit-peat-or-unparseable",
                "geologyMatchClaimed": False,
                "genericFallback": False,
            },
            "sourceBindings": {
                "etakGpkgPath": recipe_inputs.get("etakGpkgPath"),
                "etakGpkgSha256": recipe_inputs.get("etakGpkgSha256"),
                "soilSourceSha256": recipe_inputs["soilSourceSha256"],
                "classificationConfigSha256": {
                    name: recipe_inputs.get("configSha256", {}).get(name)
                    for name in (
                        "landcover-classes.toml",
                        "soil-texture.toml",
                        "soil-types.toml",
                    )
                },
            },
            "unpavedRoadsSuppressed": False,
            "maxAbsRejectedMorphologyResidualM": 0.0,
            "chunks": mask_evidence,
        },
        "parent": parent_artifact,
        "children": artifacts,
        "dependencyMerkleRoot": dependency_merkle_root(dependencies),
        "verification": {
            "scope": ["quantization-round-trip", "decoded-apron-seams"],
            "maxRoundTripErrorM": max_round_trip,
            "maxDecodedSeamErrorM": max_seam,
        },
    }
    evidence_path = evidence_root / "micro-synthesis-cook.json"
    evidence_path.write_text(json.dumps(evidence, indent=1, sort_keys=True) + "\n")
    log(f"synthesis parent {coverage.parent}: {parent_artifact['size'] / 1e6:.2f} MB")
    return evidence_path
