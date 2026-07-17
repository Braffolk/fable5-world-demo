"""Numbered QA PNG sheets for the raised-bog v3 network preview (research-only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import network

PAPER = (238, 235, 224)
INK = (28, 34, 30)
LABEL_RGB = {
    network.LAWN: (0.80, 0.82, 0.62),
    network.RIDGE: (0.46, 0.30, 0.16),
    network.HOLLOW: (0.34, 0.58, 0.55),
    network.POOL: (0.10, 0.28, 0.52),
}


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _labels_rgb(labels: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*labels.shape, 3), dtype=np.float64)
    for value, colour in LABEL_RGB.items():
        rgb[labels == value] = colour
    return rgb


def _hillshade(height: np.ndarray, pitch: float, limits: tuple[float, float]) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float64), pitch)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    light = np.sin(np.deg2rad(38.0)) * np.cos(slope) + np.cos(np.deg2rad(38.0)) * np.sin(slope) * np.cos(
        np.deg2rad(315.0) - aspect
    )
    light = np.clip((light + 0.2) / 1.2, 0.0, 1.0)
    low, high = limits
    elev = np.clip((height - low) / max(high - low, 1.0e-9), 0.0, 1.0)
    base = np.stack((0.36 + 0.30 * elev, 0.42 + 0.32 * elev, 0.30 + 0.20 * elev), axis=-1)
    return np.clip(base * (0.5 + 0.68 * light[..., None]), 0.0, 1.0)


def _signed(values: np.ndarray, limit: float) -> np.ndarray:
    scaled = np.clip(values / max(limit, 1.0e-9), -1.0, 1.0)
    neutral = np.asarray([0.94, 0.92, 0.84])
    pos = np.asarray([0.72, 0.16, 0.09])
    neg = np.asarray([0.06, 0.31, 0.61])
    return np.where(
        (scaled >= 0)[..., None],
        neutral + scaled[..., None] * (pos - neutral),
        neutral + (-scaled)[..., None] * (neg - neutral),
    )


def _panel(rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(np.asarray(np.clip(rgb, 0, 1) * 255, dtype=np.uint8), "RGB")
    scale = min(size[0] / image.width, size[1] / image.height)
    image = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.NEAREST,
    )
    canvas = Image.new("RGB", size, PAPER)
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _sheet(path: Path, title: str, note: str, panels: list[tuple[str, np.ndarray]]) -> None:
    size = (1800, 1200)
    image = Image.new("RGB", size, PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((34, 22), title, font=_font(26), fill=INK)
    draw.text((34, 58), note, font=_font(15), fill=(70, 76, 67))
    top = 96
    cols = 2 if len(panels) > 1 else 1
    rows = (len(panels) + cols - 1) // cols
    pw = (size[0] - 48) // cols
    ph = (size[1] - top - 28) // rows
    for i, (label, rgb) in enumerate(panels):
        r, c = divmod(i, cols)
        x, y = 24 + c * pw, top + r * ph
        draw.text((x + 6, y + 4), label, font=_font(16), fill=INK)
        image.paste(_panel(rgb, (pw - 14, ph - 34)), (x + 6, y + 30))
    image.save(path, optimize=True)


def _overlay_topology(labels: np.ndarray, topo: network.Topology) -> np.ndarray:
    rgb = _labels_rgb(labels)
    rgb[topo.skeleton] = (0.98, 0.95, 0.30)
    rgb[topo.terminations] = (0.95, 0.20, 0.20)
    rgb[topo.junctions] = (0.20, 0.95, 0.30)
    return rgb


def _coupling_map(labels: np.ndarray, radius_cells: float) -> np.ndarray:
    from scipy import ndimage

    rgb = _labels_rgb(labels)
    pool = labels == network.POOL
    ridge = labels == network.RIDGE
    ridge_near = ndimage.distance_transform_edt(~ridge) <= radius_cells
    rgb[pool & ridge_near] = (0.20, 0.85, 0.90)
    rgb[pool & ~ridge_near] = (0.90, 0.30, 0.65)
    return rgb


def write_all(
    root: Path,
    *,
    whole_labels: np.ndarray,
    whole_activator_norm: np.ndarray,
    whole_mire: np.ndarray,
    anisotropy: np.ndarray,
    output_labels: np.ndarray,
    topology: network.Topology,
    output_c0: np.ndarray,
    output_c1: np.ndarray,
    relief: np.ndarray,
    deviation_1m: np.ndarray,
    authority_1m: np.ndarray,
    open_water_1m: np.ndarray,
    coupling_radius_cells: float,
    gates: dict,
) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    limits = tuple(np.percentile(output_c0, [2, 98]))
    aniso_rgb = np.stack([anisotropy] * 3, axis=-1)
    whole_v = np.stack([whole_activator_norm] * 3, axis=-1)
    whole_org = _labels_rgb(whole_labels)
    whole_org[~whole_mire] = (0.12, 0.14, 0.12)

    _sheet(
        root / "01_whole_mire_organization.png",
        "01 / Whole-mire self-organized network (development mire etak-component-0004069028)",
        "Emergent activator field and typed forms over the whole connected mire. Research-only; not a target or transfer claim.",
        [("normalized activator (peat/vascular)", whole_v),
         ("typed forms: ridge/hollow/pool/lawn", whole_org),
         ("anisotropy strength |grad dome|/slope_ref", aniso_rgb),
         ("mire coverage", np.stack([whole_mire.astype(float)] * 3, -1))],
    )
    _sheet(
        root / "02_skeleton_junctions.png",
        "02 / String skeleton, junctions (green) and terminations (red)",
        f"branch={topology.branch_count} merge={topology.merge_count} terminations={topology.termination_count} over the output core+halo.",
        [("typed forms + skeleton overlay", _overlay_topology(output_labels, topology))],
    )
    _sheet(
        root / "03_pool_coupling.png",
        "03 / Pool-margin coupling map",
        f"cyan = pool perimeter within the trough-membership radius ({coupling_radius_cells*2:.0f} m) of a string; "
        f"magenta = uncoupled (isolated socket). coupling={gates['pool_coupling']['coupling_fraction']:.3f}.",
        [("pool coupling to strings", _coupling_map(output_labels, coupling_radius_cells))],
    )
    _sheet(
        root / "04_scale_bands.png",
        "04 / Carved relief and 2-16 m band",
        f"carried low-relief typed forms; band RMS={gates['amplitude_envelope']['band_2_to_16m_rms_m']:.4f} m within the Valgesoo carrier envelope.",
        [("signed carved relief (p98 scaled)", _signed(relief, float(np.percentile(np.abs(relief[relief != 0]), 98)) if (relief != 0).any() else 0.06)),
         ("C1 - C0 signed relief hillshade", _hillshade(output_c1, 0.25, limits))],
    )
    _sheet(
        root / "05_ground_closeup.png",
        "05 / Common-light C0 (raw carrier) vs C1 (carved), core closeup",
        "Same light and elevation scale. C1 shows recognizable string/hollow/pool relief above the raw 1 m ALS carrier.",
        [("C0 raw carrier 0.25 m", _hillshade(output_c0, 0.25, limits)),
         ("C1 carved network 0.25 m", _hillshade(output_c1, 0.25, limits))],
    )
    dev_rgb = _signed(deviation_1m, float(gates["parent_deviation"]["patterned_p99_m"]) or 0.06)
    mask_rgb = np.zeros((*authority_1m.shape, 3))
    mask_rgb[:] = (0.5, 0.2, 0.16)
    mask_rgb[authority_1m] = (0.7, 0.78, 0.4)
    mask_rgb[open_water_1m] = (0.1, 0.4, 0.7)
    _sheet(
        root / "06_masks_and_deviation.png",
        "06 / Masks and parent-deviation closure",
        f"deviation unpatterned p95={gates['parent_deviation']['unpatterned_p95_m']:.4f} m, patterned p99="
        f"{gates['parent_deviation']['patterned_p99_m']:.3f} m; hard/water residual exactly 0.",
        [("authority (green) / hard (red) / water (blue)", mask_rgb),
         ("C1 - carrier deviation (1 m, signed)", dev_rgb)],
    )

    images = []
    import hashlib

    for name in sorted(p.name for p in root.glob("*.png")):
        data = (root / name).read_bytes()
        images.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return images
