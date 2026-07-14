"""Decision-relevant PNGs for the cultivated agricultural R0 surface."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from ....cook.micro_hierarchy import box_mean_fixed

_CANVAS = 1120
_IMAGE = 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _quantile_scale(values: np.ndarray, symmetric: bool = False) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    if symmetric:
        bound = max(float(np.quantile(np.abs(finite), 0.995)), 1e-9)
        return np.clip(0.5 + 0.5 * finite / bound, 0.0, 1.0)
    low, high = np.quantile(finite, (0.005, 0.995))
    return np.clip((finite - low) / max(float(high - low), 1e-9), 0.0, 1.0)


def _earth(values: np.ndarray) -> Image.Image:
    t = _quantile_scale(values)
    rgb = np.empty((*t.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (42 + 177 * t).astype(np.uint8)
    rgb[..., 1] = (63 + 135 * t).astype(np.uint8)
    rgb[..., 2] = (40 + 104 * t).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _diverging(values: np.ndarray) -> Image.Image:
    t = _quantile_scale(values, symmetric=True)
    rgb = np.empty((*t.shape, 3), dtype=np.uint8)
    low = t <= 0.5
    q = np.where(low, t * 2.0, (t - 0.5) * 2.0)
    rgb[..., 0] = np.where(low, 29 + 202 * q, 231 - 76 * q).astype(np.uint8)
    rgb[..., 1] = np.where(low, 86 + 145 * q, 231 - 112 * q).astype(np.uint8)
    rgb[..., 2] = np.where(low, 151 + 80 * q, 231 - 177 * q).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def _hillshade(height: np.ndarray, texel_m: float) -> Image.Image:
    gy, gx = np.gradient(np.asarray(height, dtype=np.float64), texel_m)
    nx, ny, nz = -gx, -gy, np.ones_like(gx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    light = np.asarray((-0.45, -0.55, 0.704), dtype=np.float64)
    shade = np.clip((nx * light[0] + ny * light[1] + nz * light[2]) / norm, 0, 1)
    gray = (28 + 227 * shade).astype(np.uint8)
    return Image.fromarray(np.repeat(gray[..., None], 3, axis=2), "RGB")


def _owner(owner: np.ndarray) -> Image.Image:
    palette = np.asarray(
        [
            (78, 83, 65),
            (202, 166, 77),
            (130, 76, 48),
            (191, 125, 76),
            (47, 104, 134),
            (76, 146, 112),
        ],
        dtype=np.uint8,
    )
    return Image.fromarray(palette[np.asarray(owner, dtype=np.uint8)], "RGB")


def _resize(image: Image.Image, size: tuple[int, int] = (_IMAGE, _IMAGE)) -> Image.Image:
    return image.resize(size, Image.Resampling.BILINEAR)


def _document(
    title: str,
    caption: str,
    image: Image.Image,
    path: Path,
    legend: str | None = None,
) -> None:
    canvas = Image.new("RGB", (_CANVAS, 1180), "#f3efe5")
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 24), title, font=_font(29, True), fill="#172522")
    canvas.paste(_resize(image), (48, 82))
    draw.rectangle((47, 81, 1072, 1106), outline="#58645e", width=2)
    draw.text((48, 1120), caption, font=_font(16), fill="#394943")
    if legend:
        draw.text((48, 1148), legend, font=_font(14), fill="#6b3d2e")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=9)


def _scale_band_image(residual: np.ndarray) -> Image.Image:
    core = np.asarray(residual[:2048, :2048], dtype=np.float64)
    quarter = box_mean_fixed(core, factor=4)
    metre = box_mean_fixed(quarter, factor=4)
    metre_at_quarter = ndimage.zoom(
        metre, 4, order=1, mode="nearest", grid_mode=True, prefilter=False
    )
    quarter_at_fine = ndimage.zoom(
        quarter, 4, order=1, mode="nearest", grid_mode=True, prefilter=False
    )
    b1 = quarter - metre_at_quarter
    b2 = core - quarter_at_fine
    left = _diverging(b1).resize((500, 1000), Image.Resampling.BILINEAR)
    right = _diverging(b2).resize((500, 1000), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (1024, 1024), "#f3efe5")
    canvas.paste(left, (0, 24))
    canvas.paste(right, (524, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 2), "B1 diagnostic: 0.25-1.0 m", font=_font(15, True), fill="#172522")
    draw.text((536, 2), "B2 diagnostic: 0.125-0.25 m", font=_font(15, True), fill="#172522")
    return canvas


def _hydrology_image(
    flow_area: np.ndarray,
    incision: np.ndarray,
    deposition: np.ndarray,
) -> Image.Image:
    flow = np.log1p(np.asarray(flow_area, dtype=np.float64))
    panels = [
        _earth(flow).resize((334, 1000), Image.Resampling.BILINEAR),
        _diverging(incision).resize((334, 1000), Image.Resampling.BILINEAR),
        _diverging(deposition).resize((334, 1000), Image.Resampling.BILINEAR),
    ]
    canvas = Image.new("RGB", (1024, 1024), "#f3efe5")
    labels = ("flow area", "incision", "deposition")
    for index, panel in enumerate(panels):
        x = index * 345
        canvas.paste(panel, (x, 24))
        ImageDraw.Draw(canvas).text((x + 8, 2), labels[index], font=_font(15, True), fill="#172522")
    return canvas


def _ground_closeup(height: np.ndarray, texel_m: float) -> Image.Image:
    side = int(round(16.0 / texel_m))
    start = (height.shape[0] - side) // 2
    crop = np.asarray(height[start : start + side, start : start + side])
    return _hillshade(crop, texel_m)


def render_qa(
    qa_root: Path,
    *,
    recipe_sha256: str,
    source_hashes: dict[str, str],
    height: np.ndarray,
    base: np.ndarray,
    owner: np.ndarray,
    flow_area: np.ndarray,
    incision: np.ndarray,
    deposition: np.ndarray,
    conditioning_rgb: np.ndarray,
    texel_m: float,
) -> list[dict[str, Any]]:
    residual = np.asarray(height, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    definitions = [
        (
            "00_conditioning_orthophoto.png",
            "Bound 2025 RGB conditioning evidence",
            "Observed evidence supports paired tramline orientation and mild curvature; it does not identify management state or soil microgeometry.",
            Image.fromarray(np.asarray(conditioning_rgb, dtype=np.uint8), "RGB"),
            None,
        ),
        (
            "01_absolute_height.png",
            "Absolute 0.0625 m cultivated R0 master",
            "One joint absolute surface; color range is robustly normalized for inspection.",
            _earth(height),
            None,
        ),
        (
            "02_residual_from_low_frequency_base.png",
            "Typed-form residual from the low-frequency parcel base",
            "Brown is positive relief; blue is incision/compaction. This is not a zero-mean projection.",
            _diverging(residual),
            None,
        ),
        (
            "03_hillshade.png",
            "Ground-scale hillshade of the joint surface",
            "Inspect row continuity, rut shoulders, resolved clod silhouettes, and connected rills.",
            _hillshade(height, texel_m),
            None,
        ),
        (
            "04_form_ownership.png",
            "Dominant typed-form ownership",
            "Each sample is assigned to its largest absolute typed contribution.",
            _owner(owner),
            "base gray | rows ochre | tracks brown | clods orange | incision blue | deposition green",
        ),
        (
            "05_scale_band_decomposition.png",
            "Post-synthesis scale-band diagnostic",
            "Both bands are reductions of the same master, never independently stamped synthesis layers.",
            _scale_band_image(residual),
            None,
        ),
        (
            "06_flow_incision_deposition.png",
            "Flow-connected transport diagnostic",
            "The 0.25 m process network drives both incision and downstream sediment deposition.",
            _hydrology_image(flow_area, incision, deposition),
            None,
        ),
        (
            "07_ground_scale_closeup.png",
            "Ground-scale 16 m close-up at tile center",
            "Enlarged 0.0625 m master samples expose paired rut sections, row profiles, and resolved clod silhouettes hidden by the full-tile view.",
            _ground_closeup(height, texel_m),
            None,
        ),
    ]
    records: list[dict[str, Any]] = []
    for name, title, caption, image, legend in definitions:
        path = qa_root / name
        _document(title, caption, image, path, legend)
        with Image.open(path) as opened:
            dimensions = list(opened.size)
        records.append(
            {
                "path": name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "dimensions_xy": dimensions,
                "recipe_sha256": recipe_sha256,
                "source_hashes": source_hashes,
                "interpretation": caption,
            }
        )
    return records


def write_index(
    qa_root: Path,
    *,
    recipe_sha256: str,
    source_hashes: dict[str, str],
    metrics: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> Path:
    path = qa_root / "index.json"
    payload = {
        "schema_version": "laas.agriculture-cultivated-r0.qa/1",
        "status": "research_development_only",
        "recipe_sha256": recipe_sha256,
        "source_hashes": source_hashes,
        "metrics": metrics,
        "artifacts": artifacts,
        "claim_boundary": (
            "This development surface is not calibrated Estonia truth, not a production "
            "regime release, and not evidence for unknown parcel operations or moisture."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
