"""Integration facade for preparing, binding, and sampling an exemplar bank."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from .model import GroundSurface, PatchBank
from .preprocess import LAPINJARVI_ROLES, build_residual_bank, prepare_lapinjarvi_laz
from .synthesis import synthesize_measured


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "laas-assetgen/1"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as dst:
            while block := response.read(8 * 1024 * 1024):
                dst.write(block)
            dst.flush()
            os.fsync(dst.fileno())
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _raw_path(config_path: Path, output_root: Path, raw_root: str, plot: dict) -> Path:
    if "path" in plot:
        path = Path(plot["path"])
        return path if path.is_absolute() else config_path.parent.parent / path
    configured = config_path.parent.parent / raw_root / f"{plot['id']}.laz"
    if configured.exists():
        return configured
    return output_root / "raw" / f"{plot['id']}.laz"


def _qa_for(config: dict, plot: dict) -> dict | None:
    qa = plot.get("qa")
    if qa is None:
        qa = config.get("groundQa", {}).get(plot["id"])
    return qa if isinstance(qa, dict) else None


def prepare_lapinjarvi_bank(source_config_path: Path, output_root: Path) -> Path:
    """Fetch declared plots, prepare reviewed surfaces, and write a bound manifest.

    Ground extraction products are saved even when approval is absent, but calibration
    stops until every calibration plot has explicit ``qa.status=approved`` plus reviewer
    and notes in the source config. This prevents automated filtering from silently
    becoming morphology truth.
    """
    source_config_path = Path(source_config_path)
    output_root = Path(output_root)
    config = json.loads(source_config_path.read_text())
    if int(config.get("format", 0)) != 1:
        raise ValueError("unsupported exemplar source config format")
    plots = list(config.get("plots", []))
    roles = {str(p.get("id")): str(p.get("role")) for p in plots}
    if roles != LAPINJARVI_ROLES:
        raise ValueError(f"plots/roles must be exactly {LAPINJARVI_ROLES}")
    missing_sizes = [str(p.get("id")) for p in plots if "expectedBytes" not in p]
    if missing_sizes:
        raise ValueError("source config lacks expectedBytes for: " + ", ".join(missing_sizes))
    output_root.mkdir(parents=True, exist_ok=True)
    derived = output_root / "derived-v1"
    derived.mkdir(parents=True, exist_ok=True)

    raw_sources: list[dict] = []
    surfaces: list[GroundSurface] = []
    missing_qa: list[str] = []
    for plot in plots:
        plot_id = str(plot["id"])
        raw_path = _raw_path(
            source_config_path, output_root, str(config.get("rawRoot", "")), plot
        )
        if not raw_path.exists():
            _download(str(plot["url"]), raw_path)
        expected_bytes = int(plot["expectedBytes"])
        if raw_path.stat().st_size != expected_bytes:
            raise ValueError(
                f"raw byte-size mismatch for {plot_id}: "
                f"{raw_path.stat().st_size} != {expected_bytes}"
            )
        raw_sha = _sha256(raw_path)
        expected = plot.get("sha256")
        if expected and raw_sha.lower() != str(expected).lower():
            raise ValueError(f"raw SHA-256 mismatch for {plot_id}")
        raw_sources.append({
            "id": plot_id,
            "role": str(plot["role"]),
            "path": os.path.relpath(raw_path, output_root),
            "sha256": raw_sha,
            "size": raw_path.stat().st_size,
        })

        surface_path = derived / f"{plot_id}-ground.npz"
        if surface_path.exists():
            surface = GroundSurface.load(surface_path)
            source_sha = surface.provenance.get("source", {}).get("sha256")
            if source_sha != raw_sha:
                raise ValueError(f"stale derived ground surface for {plot_id}")
        else:
            surface = prepare_lapinjarvi_laz(
                raw_path, source_id=plot_id, sha256=raw_sha
            )
        qa = _qa_for(config, plot)
        if qa and qa.get("status") == "approved":
            surface = surface.approved(str(qa.get("reviewer", "")), str(qa.get("notes", "")))
        else:
            missing_qa.append(plot_id)
        surface.save(surface_path)
        surfaces.append(surface)

    if missing_qa:
        raise RuntimeError(
            "ground surfaces written for review; add approved QA records for: "
            + ", ".join(missing_qa)
        )

    bank = build_residual_bank([s for s in surfaces if s.role == "calibration"])
    bank_path = derived / "residual-bank.npz"
    bank.save(bank_path)
    bank_sha = _sha256(bank_path)
    manifest_path = output_root / "manifest.json"
    manifest = {
        "format": 1,
        "bankId": str(config["bankId"]),
        "surfaceFamily": str(config["surfaceFamily"]),
        "analogueStatus": str(config["analogueStatus"]),
        "license": str(config["license"]),
        "citation": str(config["citation"]),
        "bankFile": os.path.relpath(bank_path, manifest_path.parent),
        "bankSha256": bank_sha,
        "rawSources": raw_sources,
        "holdoutSurface": os.path.relpath(
            derived / "k19-ground.npz", manifest_path.parent
        ),
        "bankProvenance": bank.provenance,
    }
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    )
    temporary.replace(manifest_path)
    return manifest_path


def load_exemplar_bank(manifest_path: Path) -> PatchBank:
    """Load a bank only after verifying the bytes bound by its manifest."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("format", 0)) != 1:
        raise ValueError("unsupported exemplar manifest format")
    bank_path = manifest_path.parent / str(manifest["bankFile"])
    actual = _sha256(bank_path)
    expected = str(manifest["bankSha256"])
    if actual.lower() != expected.lower():
        raise ValueError("exemplar bank SHA-256 mismatch")
    return PatchBank.load(bank_path)


def _uniform_axis(values: np.ndarray, texel_m: float, name: str) -> tuple[bool, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a non-empty finite 1D coordinate array")
    if values.size == 1:
        return False, float(values[0])
    delta = np.diff(values)
    reverse = bool(delta[0] < 0)
    expected = -texel_m if reverse else texel_m
    if not np.allclose(delta, expected, rtol=0, atol=1e-8):
        raise ValueError(f"{name} must be uniformly spaced at the bank texel size")
    return reverse, float(values[-1] if reverse else values[0])


def synthesize_residual(
    bank: PatchBank,
    east_1d: np.ndarray,
    north_1d: np.ndarray,
    seed: int,
    top_k: int = 12,
) -> np.ndarray:
    """Return float64 residuals on increasing or decreasing regular coordinate axes."""
    east = np.asarray(east_1d, dtype=np.float64)
    north = np.asarray(north_1d, dtype=np.float64)
    reverse_e, origin_e = _uniform_axis(east, bank.texel_m, "east_1d")
    reverse_n, origin_n = _uniform_axis(north, bank.texel_m, "north_1d")
    result = synthesize_measured(
        bank,
        origin_e_m=origin_e,
        origin_n_m=origin_n,
        shape=(north.size, east.size),
        seed=int(seed),
        top_k=int(top_k),
    )
    if reverse_n:
        result = result[::-1]
    if reverse_e:
        result = result[:, ::-1]
    return np.asarray(result, dtype=np.float64)
