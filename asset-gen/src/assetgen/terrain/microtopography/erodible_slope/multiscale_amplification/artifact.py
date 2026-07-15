"""Immutable float and visual artifact for multiscale amplification."""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..morphodynamics.assemble import BoundDevelopmentA
from ..morphodynamics.structural_base import (
    EAST_OUTPUT_SAMPLE_SLICE,
    OUTPUT_SAMPLE_SLICE,
    WEST_OUTPUT_SAMPLE_SLICE,
)
from .model import AmplificationResult


@dataclass(frozen=True)
class AmplificationArtifact:
    root: Path
    manifest_path: Path
    content_sha256: str
    qa_paths: tuple[Path, ...]


def _canonical(document: object) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def _identity(payload: bytes, path: str) -> dict[str, object]:
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write(root: Path, relative: str, payload: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _identity(payload, relative)


def _array_payload(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(
        output, np.asarray(values, dtype="<f8", order="C"), allow_pickle=False
    )
    return output.getvalue()


def _hillshade(values: np.ndarray, texel_m: float) -> Image.Image:
    south, east = np.gradient(values.astype(np.float64), texel_m)
    nx, ny, nz = -east, south, np.ones(values.shape, dtype=np.float64)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    shade = np.clip((-0.45 * nx - 0.55 * ny + 0.70 * nz) / norm, 0.0, 1.0)
    gray = (28 + 222 * shade).astype(np.uint8)
    return Image.fromarray(np.repeat(gray[..., None], 3, axis=2), mode="RGB")


def _signed(values: np.ndarray) -> Image.Image:
    finite = values[np.isfinite(values)]
    limit = max(float(np.quantile(np.abs(finite), 0.995)), 1e-9)
    scaled = np.clip(values / limit, -1.0, 1.0)
    rgb = np.stack(
        (
            240 - 115 * np.maximum(-scaled, 0.0),
            240 - 135 * np.abs(scaled),
            240 - 100 * np.maximum(scaled, 0.0),
        ),
        axis=-1,
    )
    return Image.fromarray(np.nan_to_num(rgb).astype(np.uint8), mode="RGB")


def _scalar(values: np.ndarray, *, logarithmic: bool = False) -> Image.Image:
    field = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if logarithmic:
        field = np.log1p(np.maximum(field, 0.0))
    low, high = np.quantile(field, (0.01, 0.995))
    high = high if high > low else low + 1.0
    value = np.clip((field - low) / (high - low), 0.0, 1.0)
    rgb = np.stack(
        (
            25 + 220 * value,
            42 + 175 * np.sin(np.pi * value),
            62 + 170 * (1.0 - value),
        ),
        axis=-1,
    )
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def _panel(title: str, image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = Image.new("RGB", (size[0], size[1] + 36), "#eee9dc")
    ImageDraw.Draw(result).text((8, 8), title, fill="#17241c")
    result.paste(image.resize(size, Image.Resampling.BILINEAR), (0, 36))
    return result


def _png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=9, optimize=False)
    return output.getvalue()


def _relief_png(result: AmplificationResult, window: tuple[slice, slice]) -> bytes:
    c0 = result.c0_node_m[window]
    c1 = result.c1_node_m[window]
    size = (640, 420)
    canvas = Image.new("RGB", (size[0] * 3, size[1] + 36), "#eee9dc")
    images = (
        _panel("accepted C0 hillshade", _hillshade(c0, 0.0625), size),
        _panel("multiscale C1 hillshade", _hillshade(c1, 0.0625), size),
        _panel("signed C1 - C0", _signed(c1 - c0), size),
    )
    for index, image in enumerate(images):
        canvas.paste(image, (index * size[0], 0))
    return _png(canvas)


def _process_png(result: AmplificationResult) -> bytes:
    window = OUTPUT_SAMPLE_SLICE
    fields = (
        ("log flow m3/s", result.flow_m3_s[window], True),
        ("eroded depth m", result.eroded_depth_m[window], False),
        ("deposited depth m", result.deposited_depth_m[window], False),
        ("thermal signed m", result.thermal_delta_m[window], False),
        ("total signed relief m", result.c1_node_m[window] - result.c0_node_m[window], False),
        ("hard active support", result.active_node[window].astype(np.float64), False),
    )
    size = (520, 320)
    canvas = Image.new("RGB", (size[0] * 3, (size[1] + 36) * 2), "#eee9dc")
    for index, (title, values, logarithmic) in enumerate(fields):
        image = _signed(values) if title.startswith("thermal") or title.startswith("total") else _scalar(values, logarithmic=logarithmic)
        canvas.paste(
            _panel(title, image, size),
            ((index % 3) * size[0], (index // 3) * (size[1] + 36)),
        )
    return _png(canvas)


def materialize(
    bound: BoundDevelopmentA,
    result: AmplificationResult,
    output_root: Path,
) -> AmplificationArtifact:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        records: list[dict[str, object]] = []
        windows = (
            ("stitched", OUTPUT_SAMPLE_SLICE),
            ("west", WEST_OUTPUT_SAMPLE_SLICE),
            ("east", EAST_OUTPUT_SAMPLE_SLICE),
        )
        for name, window in windows:
            for role, values in (
                ("c0", result.c0_node_m[window]),
                ("c1", result.c1_node_m[window]),
                ("delta", result.c1_node_m[window] - result.c0_node_m[window]),
            ):
                records.append(
                    _write(temporary, f"float/{name}-{role}.npy", _array_payload(values))
                )
        metrics = {
            "schema": "laas.erodible-slope-multiscale-amplification-metrics/1",
            "recipeSha256": bound.recipe.sha256,
            "boundSemanticSha256": bound.semantic_sha256,
            "maximumAbsReliefM": result.maximum_abs_relief_m,
            "protectedMaxAbsM": result.protected_max_abs_m,
            "volumeChangeM3": result.volume_change_m3,
            "scaleLedgers": [asdict(value) for value in result.ledgers],
        }
        records.append(_write(temporary, "metrics.json", _canonical(metrics)))
        qa_payloads = (
            ("qa/01_west_c0_c1_relief.png", _relief_png(result, WEST_OUTPUT_SAMPLE_SLICE)),
            ("qa/02_east_c0_c1_relief.png", _relief_png(result, EAST_OUTPUT_SAMPLE_SLICE)),
            ("qa/03_stitched_process_fields.png", _process_png(result)),
        )
        qa_records = []
        for relative, payload in qa_payloads:
            record = _write(temporary, relative, payload)
            qa_records.append(record)
            records.append(record)
        package_root = Path(__file__).resolve().parent
        implementation = []
        for path in sorted(package_root.glob("*.py")):
            payload = path.read_bytes()
            implementation.append(
                {
                    "path": path.relative_to(Path.cwd().parent).as_posix(),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        qa_index = {
            "schema": "laas.erodible-slope-multiscale-amplification-qa/1",
            "images": qa_records,
        }
        records.append(_write(temporary, "qa/index.json", _canonical(qa_index)))
        records.sort(key=lambda value: str(value["path"]))
        identity = {
            "schema": "laas.erodible-slope-multiscale-amplification-artifact/1",
            "recipeSha256": bound.recipe.sha256,
            "boundSemanticSha256": bound.semantic_sha256,
            "sourceRevision": bound.recipe.source_revision,
            "operatorSequence": ["erosion", "thermal", "deposition", "x2"],
            "scaleTexelMeters": [0.25, 0.125, 0.0625],
            "implementation": implementation,
            "artifacts": records,
        }
        content_sha256 = hashlib.sha256(_canonical(identity)).hexdigest()
        manifest = {**identity, "contentSha256": content_sha256}
        _write(temporary, "manifest.json", _canonical(manifest))
        destination = output_root / "sha256" / content_sha256
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(temporary)
        else:
            temporary.replace(destination)
        qa_paths = tuple(destination / str(row["path"]) for row in qa_records)
        return AmplificationArtifact(
            root=destination,
            manifest_path=destination / "manifest.json",
            content_sha256=content_sha256,
            qa_paths=qa_paths,
        )
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
