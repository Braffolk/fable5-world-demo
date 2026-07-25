#!/usr/bin/env python3
"""Candidate-L slope-coordinate rate and allocation gate.

This consumes existing GCRP first-hit pages.  It performs no new BVH raycast,
packing experiment, runtime edit, or shader work.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


VERSION = "candidate-l-slope-rate-v1"
SOURCE_SIGMAS = (4, 16)
REFERENCE_Y = 0.49255
HORIZON_CUTOFF_DEGREES = 0.25
VD_COUNTS = (257, 513, 769)
AZIMUTH_CANDIDATES = (8, 16, 32, 64, 128, 256)
RADIAL_LAWS = ("uniform-elevation", "uniform-log-slope", "uniform-compact-slope")
VQ_SIDE_CHOICES = (32, 40, 44, 48, 56, 64, 72, 80)
L_CHOICES = (2, 3, 4, 5, 6, 7)
SCALE_CAP_BYTES = int(32.5 * 1024 * 1024)


@dataclass(frozen=True)
class Mesh:
    key: str
    vertex_count: int
    azimuth_count: int
    radial_count: int
    radial_law: str


def module_from(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_l_h_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    q = np.quantile(finite, (0.5, 0.9, 0.95, 0.99))
    return {
        "count": int(finite.size),
        "p50": float(q[0]),
        "p90": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
        "maximum": float(np.max(finite)),
    }


def mesh_candidates(vertex_count: int) -> list[Mesh]:
    remainder = vertex_count - 1
    result: list[Mesh] = []
    for azimuth_count in AZIMUTH_CANDIDATES:
        if remainder % azimuth_count:
            continue
        radial_count = remainder // azimuth_count
        if radial_count < 2:
            continue
        for law in RADIAL_LAWS:
            result.append(Mesh(
                key=f"vd{vertex_count}-a{azimuth_count}-r{radial_count}-{law}",
                vertex_count=vertex_count,
                azimuth_count=azimuth_count,
                radial_count=radial_count,
                radial_law=law,
            ))
    return result


def radial_nodes(mesh: Mesh) -> np.ndarray:
    rho_max = 1.0 / math.tan(math.radians(HORIZON_CUTOFF_DEGREES))
    u = np.arange(mesh.radial_count + 1, dtype=np.float64) / mesh.radial_count
    if mesh.radial_law == "uniform-elevation":
        elevation = 90.0 - (90.0 - HORIZON_CUTOFF_DEGREES) * u
        rho = 1.0 / np.tan(np.radians(elevation))
        rho[0] = 0.0
        rho[-1] = rho_max
        return rho
    if mesh.radial_law == "uniform-log-slope":
        return np.expm1(u * math.log1p(rho_max))
    if mesh.radial_law == "uniform-compact-slope":
        t_max = rho_max / (1.0 + rho_max)
        t = u * t_max
        return t / (1.0 - t)
    raise AssertionError(mesh.radial_law)


def local_cell(mesh: Mesh, elevation_degrees: float) -> dict[str, float | int]:
    rho = 0.0 if elevation_degrees >= 90.0 - 1e-9 else 1.0 / math.tan(math.radians(elevation_degrees))
    nodes = radial_nodes(mesh)
    cell = int(np.clip(np.searchsorted(nodes, rho, side="right") - 1, 0, mesh.radial_count - 1))
    inner = float(nodes[cell])
    outer = float(nodes[cell + 1])
    radial = outer - inner
    tangential = 2.0 * outer * math.sin(math.pi / mesh.azimuth_count)
    diameter = math.hypot(radial, tangential)
    anisotropy = max(radial, tangential) / max(1e-30, min(radial, tangential))
    return {
        "cell": cell,
        "rho": rho,
        "rhoInner": inner,
        "rhoOuter": outer,
        "radialDeltaS": radial,
        "tangentialDeltaS": tangential,
        "simplexDiameterDeltaS": diameter,
        "anisotropy": anisotropy,
    }


def build_residual_pages(hmod: Any, profile: Any) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    slices: list[dict[str, Any]] = []
    scale_rows: dict[int, list[dict[str, Any]]] = {sigma: [] for sigma in SOURCE_SIGMAS}
    for index, slice_data in enumerate(profile.slices):
        covered, depth, _ = hmod.decode_slice_base(profile, index)
        vertical = float(-slice_data.direction[1])
        elevation = math.degrees(math.asin(np.clip(vertical, 0.0, 1.0)))
        azimuth = math.degrees(math.atan2(float(slice_data.direction[2]), float(slice_data.direction[0]))) % 360.0
        height = profile.top_h - vertical * depth
        residual = np.where(covered, np.abs(height - REFERENCE_Y), np.nan).astype(np.float32)
        hit_summary = quantiles(residual[covered])
        base = {
            "slice": index,
            "elevationDegrees": elevation,
            "azimuthDegrees": azimuth,
            "coveredFraction": float(np.mean(covered)),
            "hitResidualMetres": hit_summary,
        }
        slices.append(base)
        for sigma in SOURCE_SIGMAS:
            size = sigma * 2 + 1
            cover_weight = ndimage.uniform_filter(covered.astype(np.float32), size=size, mode="wrap")
            residual_zero = np.where(covered, residual, 0.0)
            mean_numerator = ndimage.uniform_filter(residual_zero, size=size, mode="wrap")
            local_mean = np.divide(
                mean_numerator,
                cover_weight,
                out=np.full_like(mean_numerator, np.nan),
                where=cover_weight > 1e-8,
            )
            local_max = ndimage.maximum_filter(residual_zero, size=size, mode="wrap")
            support = cover_weight > 1e-8
            scale_rows[sigma].append({
                **base,
                "footprintCoverage": quantiles(cover_weight),
                "footprintMeanResidualMetres": quantiles(local_mean[support]),
                "footprintMaxResidualMetres": quantiles(local_max[support]),
            })
        print(f"[candidate-l-rate] reduced slice {index + 1}/{len(profile.slices)}", flush=True)
    return slices, scale_rows


def rate_tables(
    profile: Any,
    scale_rows: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    phase_radius = {
        sigma: sigma / profile.interior_width * max(profile.size_x, profile.size_z)
        for sigma in SOURCE_SIGMAS
    }
    mesh_rows: list[dict[str, Any]] = []
    anisotropy_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for vertex_count in VD_COUNTS:
        for mesh in mesh_candidates(vertex_count):
            per_mesh: list[dict[str, Any]] = []
            for sigma in SOURCE_SIGMAS:
                radius = phase_radius[sigma]
                for source in scale_rows[sigma]:
                    local = local_cell(mesh, source["elevationDegrees"])
                    residual = source["footprintMaxResidualMetres"]["p95"]
                    typical = source["hitResidualMetres"]["p95"]
                    row = {
                        "mesh": asdict(mesh),
                        "sigmaAt256": sigma,
                        "physicalRadiusMetres": radius,
                        "slice": source["slice"],
                        "elevationDegrees": source["elevationDegrees"],
                        "azimuthDegrees": source["azimuthDegrees"],
                        "hitResidualP95Metres": typical,
                        "footprintMaxResidualP95Metres": residual,
                        **local,
                        "typicalRate": typical * local["simplexDiameterDeltaS"] / radius,
                        "conservativeRate": residual * local["simplexDiameterDeltaS"] / radius,
                        "radialRate": residual * local["radialDeltaS"] / radius,
                        "tangentialRate": residual * local["tangentialDeltaS"] / radius,
                    }
                    mesh_rows.append(row)
                    per_mesh.append(row)
            for sigma in SOURCE_SIGMAS:
                sigma_rows = [row for row in per_mesh if row["sigmaAt256"] == sigma]
                for elevation in sorted(set(round(row["elevationDegrees"], 6) for row in sigma_rows)):
                    group = [row for row in sigma_rows if round(row["elevationDegrees"], 6) == elevation]
                    anisotropy_rows.append({
                        "meshKey": mesh.key,
                        "vertexCount": vertex_count,
                        "sigmaAt256": sigma,
                        "elevationDegrees": elevation,
                        "azimuthSamples": len(group),
                        "cellAnisotropy": group[0]["anisotropy"],
                        "conservativeRateAcrossAzimuth": quantiles(np.array([row["conservativeRate"] for row in group])),
                        "radialRateAcrossAzimuth": quantiles(np.array([row["radialRate"] for row in group])),
                        "tangentialRateAcrossAzimuth": quantiles(np.array([row["tangentialRate"] for row in group])),
                    })
            summary = {
                "mesh": asdict(mesh),
                "scales": {},
                "horizonFringe": {
                    "elevationRangeDegrees": [0.0, HORIZON_CUTOFF_DEGREES],
                    "slopeDiameter": "infinite",
                    "rate": "infinite for any nonzero residual",
                    "appearanceConvergence": "RED in Candidate K 0.05--0.5 degree heldout sequence",
                },
            }
            for sigma in SOURCE_SIGMAS:
                group = [row for row in per_mesh if row["sigmaAt256"] == sigma]
                summary["scales"][str(sigma)] = {
                    "physicalRadiusMetres": phase_radius[sigma],
                    "typicalRate": quantiles(np.array([row["typicalRate"] for row in group])),
                    "conservativeRate": quantiles(np.array([row["conservativeRate"] for row in group])),
                    "worstConservativeRate": max(row["conservativeRate"] for row in group),
                    "allMeasuredDirectionsBelowOne": all(row["conservativeRate"] <= 1.0 for row in group),
                }
            best_rows.append(summary)
    # One best mesh per V_D and scale; minimizing the measured worst rate.
    winners: list[dict[str, Any]] = []
    for vertex_count in VD_COUNTS:
        candidates = [row for row in best_rows if row["mesh"]["vertex_count"] == vertex_count]
        for sigma in SOURCE_SIGMAS:
            winner = min(candidates, key=lambda row: row["scales"][str(sigma)]["worstConservativeRate"])
            winners.append({
                "vertexCount": vertex_count,
                "sigmaAt256": sigma,
                "mesh": winner["mesh"],
                "scale": winner["scales"][str(sigma)],
                "horizonFringe": winner["horizonFringe"],
            })
    return mesh_rows, anisotropy_rows, winners


def memory_table(profile: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vd in VD_COUNTS:
        for side in VQ_SIDE_CHOICES:
            vq = side * side
            phase_spacing = max(profile.size_x, profile.size_z) / side
            for levels in L_CHOICES:
                byte_count = 4 * vq * vd * levels
                rows.append({
                    "V_D": vd,
                    "V_Q": vq,
                    "sideEquivalent": side,
                    "L": levels,
                    "bytes": byte_count,
                    "MiB": byte_count / 1048576,
                    "under32Point5MiB": byte_count <= SCALE_CAP_BYTES,
                    "phaseSpacingMetres": phase_spacing,
                    "sigma4RadiusMetres": 4 / profile.interior_width * max(profile.size_x, profile.size_z),
                    "sigma4NonBlurredBySpacing": phase_spacing <= 4 / profile.interior_width * max(profile.size_x, profile.size_z),
                })
    return rows


def write_qa(output: Path, winners: list[dict[str, Any]], memory: list[dict[str, Any]]) -> list[dict[str, str]]:
    qa = output / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1100, 520), (16, 20, 27))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "Candidate L: best measured R * Delta_s / r (GREEN <= 1)", fill=(245, 245, 245))
    colors = {4: (255, 105, 97), 16: (73, 190, 255)}
    for row_index, winner in enumerate(winners):
        y = 65 + row_index * 68
        rate = float(winner["scale"]["worstConservativeRate"])
        sigma = int(winner["sigmaAt256"])
        width = int(min(850, 850 * math.log10(max(1.0, rate)) / math.log10(180.0)))
        draw.text((24, y), f"VD={winner['vertexCount']:3d} sigma={sigma:2d}  {rate:8.2f}", fill=(230, 230, 230))
        draw.rectangle((245, y, 245 + width, y + 24), fill=colors[sigma])
    rate_path = qa / "001-best-rate.png"
    image.save(rate_path)

    frontier = Image.new("RGB", (1100, 560), (16, 20, 27))
    draw = ImageDraw.Draw(frontier)
    draw.text((24, 18), "Candidate L: non-blurred (V_Q >= 64^2) allocation frontier", fill=(245, 245, 245))
    y = 55
    draw.text((24, y), "V_D   maximum L at 64^2   MiB     highest under-cap phase side / L", fill=(190, 210, 230))
    for vd in VD_COUNTS:
        valid = [row for row in memory if row["V_D"] == vd and row["under32Point5MiB"] and row["sigma4NonBlurredBySpacing"]]
        at64 = [row for row in valid if row["sideEquivalent"] == 64]
        max_l = max((row["L"] for row in at64), default=0)
        selected = max(valid, key=lambda row: (row["sideEquivalent"], row["L"]))
        y += 72
        draw.text((24, y), f"{vd:3d}          {max_l:2d}          {max(row['MiB'] for row in at64):6.2f}       {selected['sideEquivalent']:2d}^2 / L={selected['L']}", fill=(235, 235, 235))
    draw.text((24, 360), "The best angular rate at V_D=769 is still 93.35 (sigma4) / 24.77 (sigma16).", fill=(255, 150, 120))
    draw.text((24, 400), "The compactified 0--0.25 degree fringe has infinite slope diameter and failed appearance convergence.", fill=(255, 150, 120))
    memory_path = qa / "002-memory-frontier.png"
    frontier.save(memory_path)
    images = []
    for path, interpretation in [
        (rate_path, "Best measured structured-mesh rate per direction budget and physical footprint scale."),
        (memory_path, "Allocation frontier after requiring phase spacing no coarser than the sigma-4 physical radius."),
    ]:
        images.append({"file": path.name, "sha256": sha256(path), "interpretation": interpretation})
    (qa / "index.json").write_text(json.dumps({"schema": "candidate-l-slope-rate-qa-v1", "images": images}, indent=2) + "\n")
    return images


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "src/assets/groundcover/calamagrostis-canescens.gcrp"
    h_path = root / "tools/groundcover-bake/analyze_candidate_h_angular_continuity.py"
    doc = root / "docs/tasks/2026-07-25/groundcover/archive-inactive/by-original-date/2026-07-24/GRASSPROFILE2-CANDIDATE-L-CONFORMING-4D-MEASURE-FE.md"
    k_report = root / "data/work/groundcover-candidate-k-angular-convergence/e3e0a4175b151b89/18fafa8984a3c8ac/report.json"
    hmod = module_from(h_path)
    profile = hmod.load_profile(source)
    slices, scale_rows = build_residual_pages(hmod, profile)
    mesh_rows, anisotropy_rows, winners = rate_tables(profile, scale_rows)
    memory = memory_table(profile)
    phase_radius = {str(s): s / profile.interior_width * max(profile.size_x, profile.size_z) for s in SOURCE_SIGMAS}
    feasible_nonblurred = [row for row in memory if row["under32Point5MiB"] and row["sigma4NonBlurredBySpacing"]]
    rate_green = any(
        row["scale"]["allMeasuredDirectionsBelowOne"]
        for row in winners
        if row["sigmaAt256"] == 4
    )
    verdict = "GREEN_RATE_PLAUSIBLE" if rate_green and feasible_nonblurred else "RED_RATE_OR_MEMORY"
    recipe = {
        "version": VERSION,
        "sourceSha256": sha256(source),
        "scriptSha256": sha256(Path(__file__)),
        "candidateLDocSha256": sha256(doc),
        "candidateKConvergenceSha256": sha256(k_report),
        "referenceY": REFERENCE_Y,
        "horizonCutoffDegrees": HORIZON_CUTOFF_DEGREES,
        "sourceSigmas": SOURCE_SIGMAS,
        "vertexCounts": VD_COUNTS,
        "azimuthCandidates": AZIMUTH_CANDIDATES,
        "radialLaws": RADIAL_LAWS,
        "scaleCapBytes": SCALE_CAP_BYTES,
    }
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    output = root / "data/work/groundcover-candidate-l-slope-rate" / recipe["sourceSha256"][:16] / recipe_hash[:16]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": VERSION,
        "verdict": verdict,
        "recipe": recipe,
        "source": {
            "topH": profile.top_h,
            "tileSizeX": profile.size_x,
            "tileSizeZ": profile.size_z,
            "phaseResolution": profile.interior_width,
            "physicalFilterRadiiMetres": phase_radius,
        },
        "firstHitResiduals": slices,
        "scaleResiduals": {str(key): value for key, value in scale_rows.items()},
        "rateWinners": winners,
        "allMeshDirectionRates": mesh_rows,
        "perElevationAzimuthAnisotropy": anisotropy_rows,
        "memory": memory,
        "nonBlurredMemoryChoices": feasible_nonblurred,
        "decision": {
            "rateGreenAtSigma4": rate_green,
            "nonBlurredMemoryChoiceExists": bool(feasible_nonblurred),
            "criterion": "at least one structured mesh has R_p95*Delta_s/r <= 1 in every measured direction at sigma4, and one allocation retains <= sigma4 phase spacing",
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_qa(output, winners, memory)
    print(json.dumps({
        "output": str(output),
        "verdict": verdict,
        "physicalFilterRadiiMetres": phase_radius,
        "rateWinners": winners,
        "nonBlurredMemoryChoiceCount": len(feasible_nonblurred),
    }, indent=2))


if __name__ == "__main__":
    main()
