"""Coherent example-based amplification of mapped aeolian dune terrain."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

from ...repair.prolong import prolong_structural_4x
from .conditions import PilotConditions


SOURCE_PIXEL_M = 0.1
FINE_PIXEL_M = 0.25
PARENT_PIXEL_M = 1.0
DICTIONARY_PATCH_PARENT = 8
PLACEMENT_STRIDE_PARENT = 4
SOURCE_LOWPASS_CUTOFF_M = 1.0


@dataclass(frozen=True)
class SourceAtom:
    exemplar_index: int
    row: int
    col: int
    feature: np.ndarray
    parent_patch: np.ndarray
    residual_patch: np.ndarray
    gradient_rms: float


@dataclass(frozen=True)
class SynthesisResult:
    parent_core_m: np.ndarray
    structural_fine_m: np.ndarray
    residual_fine_m: np.ndarray
    absolute_fine_m: np.ndarray
    chosen_atom_index: np.ndarray
    amplitude_scale: np.ndarray
    source_rotation_degrees: tuple[float, float]
    source_residual_examples_m: np.ndarray
    metrics: dict


def _read_parent(path: Path, bbox: tuple[int, int, int, int]) -> np.ndarray:
    with rasterio.open(path) as source:
        if (
            str(source.crs) != "EPSG:3301"
            or source.transform.a != 1.0
            or source.transform.e != -1.0
            or source.nodata is None
        ):
            raise ValueError("pilot DTM contract changed")
        window = rasterio.windows.from_bounds(*bbox, source.transform)
        values = source.read(1, window=window, out_dtype="float64")
        invalid = ~np.isfinite(values) | (values == float(source.nodata))
        if invalid.any():
            raise ValueError("dune pilot support contains invalid DTM samples")
    expected = (bbox[3] - bbox[1], bbox[2] - bbox[0])
    if values.shape != expected:
        raise ValueError(f"unexpected DTM support shape: {values.shape} != {expected}")
    return values


def _fill_nan(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if np.all(valid):
        return values.astype(np.float64)
    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
    output = values.astype(np.float64, copy=True)
    output[~valid] = output[tuple(axis[~valid] for axis in indices)]
    return output


def _dominant_tangent_xy(height: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(height.astype(np.float64))
    covariance = np.asarray(
        [[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]],
        dtype=np.float64,
    )
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, -1]
    tangent = np.asarray((-normal[1], normal[0]), dtype=np.float64)
    if tangent[0] < 0.0:
        tangent *= -1.0
    return tangent / np.linalg.norm(tangent)


def _angle_degrees(vector_xy: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector_xy[1]), float(vector_xy[0])))


def _detrended_feature(patch: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(patch.shape, dtype=np.float64)
    design = np.stack((cols.ravel(), rows.ravel(), np.ones(patch.size)), axis=1)
    coefficients, *_ = np.linalg.lstsq(design, patch.ravel(), rcond=None)
    residual = patch - (
        coefficients[0] * cols + coefficients[1] * rows + coefficients[2]
    )
    gy, gx = np.gradient(residual)
    scale = max(float(np.sqrt(np.mean(gx * gx + gy * gy))), 1.0e-8)
    return np.concatenate(
        ((residual / scale).ravel(), (gx / scale).ravel(), (gy / scale).ravel())
    ).astype(np.float32)


def _gradient_rms(patch: np.ndarray) -> float:
    gy, gx = np.gradient(patch.astype(np.float64))
    return float(np.sqrt(np.mean(gx * gx + gy * gy)))


def _source_atoms(
    target_npz: Path,
    shore_tangent_en: tuple[float, float],
) -> tuple[list[SourceAtom], tuple[float, float], np.ndarray]:
    with np.load(target_npz) as source:
        patches = source["representative_patches_m"][:2].astype(np.float64)

    shore_xy = np.asarray(
        (shore_tangent_en[0], -shore_tangent_en[1]), dtype=np.float64
    )
    shore_xy /= np.linalg.norm(shore_xy)
    atoms: list[SourceAtom] = []
    rotations: list[float] = []
    examples: list[np.ndarray] = []
    sigma = SOURCE_LOWPASS_CUTOFF_M / (2.355 * SOURCE_PIXEL_M)
    for exemplar_index, patch in enumerate(patches):
        filled = _fill_nan(patch)
        supported = ndimage.gaussian_filter(filled, sigma=sigma, mode="reflect")
        source_tangent = _dominant_tangent_xy(supported)
        rotation = _angle_degrees(shore_xy) - _angle_degrees(source_tangent)
        aligned = ndimage.rotate(
            supported,
            rotation,
            reshape=False,
            order=3,
            mode="reflect",
            prefilter=True,
        )
        resampled = ndimage.zoom(
            aligned,
            SOURCE_PIXEL_M / FINE_PIXEL_M,
            order=3,
            mode="reflect",
            prefilter=True,
        )
        side = 96
        row0 = (resampled.shape[0] - side) // 2
        col0 = (resampled.shape[1] - side) // 2
        fine = resampled[row0 : row0 + side, col0 : col0 + side]
        parent = fine.reshape(24, 4, 24, 4).mean(axis=(1, 3))
        baseline = prolong_structural_4x(
            parent, parent_rows=(2, 22), parent_cols=(2, 22)
        )
        true_inner = fine[8:88, 8:88]
        residual = true_inner - baseline
        parent_inner = parent[2:22, 2:22]
        examples.append(residual)
        rotations.append(rotation)
        for row in range(0, 20 - DICTIONARY_PATCH_PARENT + 1):
            for col in range(0, 20 - DICTIONARY_PATCH_PARENT + 1):
                parent_patch = parent_inner[
                    row : row + DICTIONARY_PATCH_PARENT,
                    col : col + DICTIONARY_PATCH_PARENT,
                ]
                fine_row = row * 4
                fine_col = col * 4
                residual_patch = residual[
                    fine_row : fine_row + DICTIONARY_PATCH_PARENT * 4,
                    fine_col : fine_col + DICTIONARY_PATCH_PARENT * 4,
                ]
                atoms.append(
                    SourceAtom(
                        exemplar_index=exemplar_index,
                        row=row,
                        col=col,
                        feature=_detrended_feature(parent_patch),
                        parent_patch=parent_patch.astype(np.float32),
                        residual_patch=residual_patch.astype(np.float32),
                        gradient_rms=_gradient_rms(parent_patch),
                    )
                )
    return atoms, (float(rotations[0]), float(rotations[1])), np.stack(examples)


def _closed_residual(
    target_parent_support: np.ndarray,
    atoms: list[SourceAtom],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = target_parent_support.shape[0]
    if target_parent_support.shape != (136, 136):
        raise ValueError("dune support must be the frozen 136x136 parent solve")
    raw = np.zeros((side * 4, side * 4), dtype=np.float64)
    weights = np.zeros_like(raw)
    choice = np.full((side, side), -1, dtype=np.int32)
    scales = np.zeros((side, side), dtype=np.float32)
    dictionary = np.stack([atom.feature for atom in atoms]).astype(np.float64)
    window_parent = np.outer(np.hanning(DICTIONARY_PATCH_PARENT + 2)[1:-1], np.hanning(DICTIONARY_PATCH_PARENT + 2)[1:-1])
    window_fine = np.repeat(np.repeat(window_parent, 4, axis=0), 4, axis=1)
    for row in range(0, side - DICTIONARY_PATCH_PARENT + 1, PLACEMENT_STRIDE_PARENT):
        for col in range(0, side - DICTIONARY_PATCH_PARENT + 1, PLACEMENT_STRIDE_PARENT):
            target_patch = target_parent_support[
                row : row + DICTIONARY_PATCH_PARENT,
                col : col + DICTIONARY_PATCH_PARENT,
            ]
            feature = _detrended_feature(target_patch).astype(np.float64)
            distances = np.mean((dictionary - feature[None, :]) ** 2, axis=1)
            atom_index = int(np.argmin(distances))
            atom = atoms[atom_index]
            target_gradient = _gradient_rms(target_patch)
            amplitude = min(1.0, target_gradient / max(atom.gradient_rms, 1.0e-8))
            fine_row = row * 4
            fine_col = col * 4
            region = np.s_[
                fine_row : fine_row + DICTIONARY_PATCH_PARENT * 4,
                fine_col : fine_col + DICTIONARY_PATCH_PARENT * 4,
            ]
            raw[region] += atom.residual_patch * amplitude * window_fine
            weights[region] += window_fine
            center_row = row + DICTIONARY_PATCH_PARENT // 2
            center_col = col + DICTIONARY_PATCH_PARENT // 2
            choice[center_row, center_col] = atom_index
            scales[center_row, center_col] = amplitude
    if np.any(weights[8:-8, 8:-8] <= 0.0):
        raise ValueError("exemplar overlap-add left an uncovered interior sample")
    raw = np.divide(raw, weights, out=np.zeros_like(raw), where=weights > 0.0)

    block_means = raw.reshape(side, 4, side, 4).mean(axis=(1, 3))
    correction = prolong_structural_4x(
        block_means, parent_rows=(2, 134), parent_cols=(2, 134)
    )
    closed_inner = raw[8:-8, 8:-8] - correction
    core = closed_inner[8:520, 8:520]
    return core, choice[4:132, 4:132], scales[4:132, 4:132]


def synthesize(
    *,
    dtm_path: Path,
    target_npz: Path,
    conditions: PilotConditions,
) -> SynthesisResult:
    parent_support = _read_parent(dtm_path, conditions.support_bbox)
    parent_core = parent_support[4:132, 4:132]
    structural = prolong_structural_4x(
        parent_support, parent_rows=(4, 132), parent_cols=(4, 132)
    )
    atoms, rotations, source_examples = _source_atoms(
        target_npz, conditions.shore_tangent_en
    )
    residual, choices, scales = _closed_residual(parent_support, atoms)
    if not np.all(conditions.authority_fine):
        raise ValueError("pilot authority must remain complete for this FLOAT attempt")
    absolute = structural + residual
    recovered = absolute.reshape(128, 4, 128, 4).mean(axis=(1, 3))
    closure = recovered - parent_core
    structural_recovered = structural.reshape(128, 4, 128, 4).mean(axis=(1, 3))

    gy, gx = np.gradient(residual, FINE_PIXEL_M)
    gradient_covariance = np.asarray(
        [[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]],
        dtype=np.float64,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(gradient_covariance)
    normal_xy = eigenvectors[:, -1]
    tangent_xy = np.asarray((-normal_xy[1], normal_xy[0]))
    tangent_en = np.asarray((tangent_xy[0], -tangent_xy[1]))
    tangent_en /= np.linalg.norm(tangent_en)
    shore = np.asarray(conditions.shore_tangent_en)
    alignment = abs(float(np.dot(tangent_en, shore)))
    selected = choices[choices >= 0]
    selected_exemplars = np.asarray([atoms[int(value)].exemplar_index for value in selected])
    metrics = {
        "surface": {
            "parentShape": list(parent_core.shape),
            "fineShape": list(absolute.shape),
            "parentPixelM": PARENT_PIXEL_M,
            "finePixelM": FINE_PIXEL_M,
        },
        "dictionary": {
            "atoms": len(atoms),
            "patchParentSamples": DICTIONARY_PATCH_PARENT,
            "placementStrideParentSamples": PLACEMENT_STRIDE_PARENT,
            "sourceLowpassCutoffM": SOURCE_LOWPASS_CUTOFF_M,
            "selectedPlacements": int(selected.size),
            "selectedExemplarCounts": {
                str(index): int(np.count_nonzero(selected_exemplars == index))
                for index in range(2)
            },
            "amplitudeScaleP05P50P95": [
                float(value)
                for value in np.quantile(scales[scales > 0], (0.05, 0.5, 0.95))
            ],
            "sourceRotationDegrees": list(rotations),
        },
        "closure": {
            "maximumAbsParentMeanErrorM": float(np.max(np.abs(closure))),
            "rmsParentMeanErrorM": float(np.sqrt(np.mean(closure * closure))),
            "structuralMaximumAbsParentMeanErrorM": float(
                np.max(np.abs(structural_recovered - parent_core))
            ),
        },
        "residual": {
            "minimumM": float(np.min(residual)),
            "maximumM": float(np.max(residual)),
            "rmsM": float(np.sqrt(np.mean(residual * residual))),
            "absP95M": float(np.quantile(np.abs(residual), 0.95)),
            "absP99M": float(np.quantile(np.abs(residual), 0.99)),
            "ridgeTangentEn": [float(tangent_en[0]), float(tangent_en[1])],
            "shoreTangentAlignmentAbsDot": alignment,
            "gradientEigenvalues": [float(value) for value in eigenvalues],
        },
    }
    return SynthesisResult(
        parent_core_m=parent_core.astype(np.float32),
        structural_fine_m=structural.astype(np.float32),
        residual_fine_m=residual.astype(np.float32),
        absolute_fine_m=absolute.astype(np.float32),
        chosen_atom_index=choices,
        amplitude_scale=scales,
        source_rotation_degrees=rotations,
        source_residual_examples_m=source_examples.astype(np.float32),
        metrics=metrics,
    )
