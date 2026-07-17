"""Common-light QA diagnostics for the cliff-anchor-organization reconstruction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT = ImageFont.load_default()
_BG = (244, 240, 229)


def _shade(height, pitch, vmin, vmax):
    gy, gx = np.gradient(height.astype(np.float64), pitch)
    nx, ny, nz = -gx, gy, np.ones_like(height)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    light = np.asarray([-0.62, 0.55, 0.56])
    lit = np.clip((nx * light[0] + ny * light[1] + nz * light[2]) / norm, -0.25, 1.0)
    lit = 0.42 + 0.58 * (lit + 0.25) / 1.25
    t = np.clip((height - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    low = np.asarray([58.0, 82.0, 47.0]); high = np.asarray([190.0, 178.0, 138.0])
    color = low[None, None, :] * (1.0 - t[..., None]) + high[None, None, :] * t[..., None]
    return Image.fromarray(np.clip(color * lit[..., None], 0, 255).astype(np.uint8), "RGB")


def _signed(value, limit):
    t = np.clip(value / max(limit, 1e-6), -1.0, 1.0)
    neutral = np.asarray([239.0, 236.0, 224.0]); cold = np.asarray([39.0, 92.0, 152.0]); warm = np.asarray([183.0, 53.0, 47.0])
    color = np.where((t < 0.0)[..., None], neutral + (-t[..., None]) * (cold - neutral), neutral + t[..., None] * (warm - neutral))
    return Image.fromarray(np.clip(color, 0, 255).astype(np.uint8), "RGB")


def _fit(image, size):
    return image.resize(size, Image.Resampling.LANCZOS)


def _panel(image, title, size=(760, 460)):
    canvas = Image.new("RGB", (size[0], size[1] + 34), _BG)
    canvas.paste(_fit(image, size), (0, 34))
    ImageDraw.Draw(canvas).text((10, 10), title, fill=(25, 29, 25), font=_FONT)
    return canvas


def _join(images, columns):
    rows = (len(images) + columns - 1) // columns
    w = max(i.width for i in images); h = max(i.height for i in images)
    result = Image.new("RGB", (columns * w, rows * h), _BG)
    for i, im in enumerate(images):
        result.paste(im, ((i % columns) * w, (i // columns) * h))
    return result


def write_qa(root: Path, ev, recon, anchor_residual, state, envelope, corridor, gates, config) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=False)
    c0 = ev.c0_m.astype(np.float64); pitch = ev.pitch_m
    residual = recon.residual_m
    after = c0 + residual
    vmin, vmax = np.percentile(c0[ev.active], [1, 99])
    limit = max(float(np.percentile(np.abs(residual[ev.active]), 99)), 0.05)

    # crop windows
    b = ev.bbox_en
    def win(n0, n1, e0, e1):
        r0 = int(round((b[3] - n1) / pitch)); r1 = int(round((b[3] - n0) / pitch)) + 1
        c0c = int(round((e0 - b[0]) / pitch)); c1c = int(round((e1 - b[0]) / pitch)) + 1
        return slice(r0, r1), slice(c0c, c1c)
    out = win(6444416.0, 6444544.0, 680448.0, 680704.0)

    # 01 whole-line organization: C0 vs reconstruction, common light
    p1 = root / "01-whole-line-organization.png"
    _join([
        _panel(_shade(c0, pitch, vmin, vmax), "Accepted corrected base C0, common light"),
        _panel(_shade(after, pitch, vmin, vmax), "Cliff-anchor-organization reconstruction, common light"),
    ], 2).save(p1)

    # 02 anchored-vs-interpolated state map
    H, W = c0.shape
    stmap = np.zeros((H, W, 3), dtype=np.uint8); stmap[:] = (225, 220, 205)
    stmap[recon.face_member > 0.5] = (196, 120, 60)
    stmap[recon.toe_member > 0.5] = (70, 120, 170)
    stmap[recon.shoulder_member > 0.5] = (120, 160, 90)
    stmap[ev.mapped_face] = (20, 22, 22)
    # state curves panel
    curve = Image.new("RGB", (900, 460), _BG); d = ImageDraw.Draw(curve)
    S = state.face_h; nS = len(S)
    d.text((10, 8), "Along-line profile state: face height (red), width x2 (blue), envelope x20 (green)", fill=(25, 29, 25), font=_FONT)
    def poly(arr, sc, off, col):
        pts = [(40 + i / max(nS - 1, 1) * 840, 440 - np.clip(arr[i] * sc + off, 0, 420)) for i in range(nS)]
        d.line(pts, fill=col, width=2)
    poly(state.face_h, 20, 20, (200, 60, 60))
    poly(state.width, 40, 20, (60, 90, 170))
    poly(envelope, 400, 20, (60, 150, 80))
    p2 = root / "02-anchored-interpolated-state.png"
    _join([_panel(Image.fromarray(stmap, "RGB"), "Form membership: orange=face, blue=toe, green=shoulder, black=mapped line"), _panel(curve, "Interpolated profile state along line")], 2).save(p2)

    # 03 junction / socket / chute placements
    gmap = _signed(recon.grammar_m, max(float(np.percentile(np.abs(recon.grammar_m[recon.grammar_m != 0]), 99)) if np.any(recon.grammar_m != 0) else 0.05, 0.03))
    p3 = root / "03-grammar-socket-chute-placements.png"
    _join([
        _panel(gmap, "Biala connective grammar delta (bench/socket/chute), warm=fill cold=cut"),
        _panel(_shade(after[out], pitch, vmin, vmax), "Reconstruction over packed output window, common light"),
    ], 2).save(p3)

    # 04 common-light closeups before/after
    close = win(6444430.0, 6444478.0, 680500.0, 680600.0)
    p4 = root / "04-common-light-closeups.png"
    _join([
        _panel(_shade(c0[close], pitch, vmin, vmax), "Before: C0 closeup"),
        _panel(_shade(after[close], pitch, vmin, vmax), "After: reconstruction closeup (same light)"),
    ], 2).save(p4)

    # 05 bands: fine band + deviation field
    fmap = _signed(recon.fine_m, max(float(np.percentile(np.abs(recon.fine_m[recon.fine_m != 0]), 99)) if np.any(recon.fine_m != 0) else 0.05, 0.03))
    p5 = root / "05-fine-band-and-deviation.png"
    _join([
        _panel(fmap, "Band-limited 0.5-1.25 m fine component (conditioned on face/toe)"),
        _panel(_signed(residual, limit), f"Total deviation from C0, +/-{limit:.3f} m"),
    ], 2).save(p5)

    # 06 masks + proximity + ordinary ground
    mask = np.zeros((H, W, 3), dtype=np.uint8); mask[:] = (225, 220, 205)
    mask[ev.active] = (65, 132, 96)
    mask[recon.ordinary_ground] = (150, 175, 120)
    mask[ev.hard_zero] = (181, 65, 44)
    mask[ev.mapped_face] = (20, 22, 22)
    prox = np.clip(recon.proximity, 0, 1)
    prox_img = Image.fromarray((np.stack([prox, prox, prox], axis=-1) * 255).astype(np.uint8), "RGB")
    p6 = root / "06-masks-proximity.png"
    _join([
        _panel(Image.fromarray(mask, "RGB"), "green=active, pale=ordinary ground (DC guard), red=hard, black=face"),
        _panel(prox_img, "ALS proximity weight (0 at measurements -> detail suppressed there)"),
    ], 2).save(p6)

    return {
        p1.name: "whole-line organization: C0 vs connected reconstruction under one light",
        p2.name: "form-type membership and interpolated along-line profile state",
        p3.name: "Biala connective grammar placements and packed-window reconstruction",
        p4.name: "before/after common-light closeups of the escarpment face",
        p5.name: "band-limited fine component and total deviation-from-C0 field",
        p6.name: "ownership masks, ordinary-ground DC guard, and ALS proximity weighting",
    }
