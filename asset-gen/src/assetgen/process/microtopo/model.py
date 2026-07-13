"""Data models and portable serialization for measured microtopography."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


def _metadata_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class GroundSurface:
    """A TLS-derived ground raster plus evidence retained for fail-closed QA."""

    elevation_m: np.ndarray
    measured: np.ndarray
    confidence: np.ndarray
    origin_x_m: float
    origin_y_m: float
    texel_m: float
    source_id: str
    role: str
    provenance: dict[str, Any]
    qa_status: str = "unreviewed"

    def approved(self, reviewer: str, notes: str) -> "GroundSurface":
        if not reviewer.strip() or not notes.strip():
            raise ValueError("ground approval requires reviewer and notes")
        provenance = dict(self.provenance)
        provenance["qa"] = {"reviewer": reviewer, "notes": notes}
        return replace(self, provenance=provenance, qa_status="approved")

    def save(self, path: Path) -> None:
        meta = {
            "origin_x_m": self.origin_x_m,
            "origin_y_m": self.origin_y_m,
            "texel_m": self.texel_m,
            "source_id": self.source_id,
            "role": self.role,
            "qa_status": self.qa_status,
            "provenance": self.provenance,
        }
        np.savez_compressed(
            path,
            elevation_m=np.asarray(self.elevation_m, dtype=np.float32),
            measured=np.asarray(self.measured, dtype=np.uint8),
            confidence=np.asarray(self.confidence, dtype=np.float32),
            metadata=np.asarray(_metadata_json(meta)),
        )

    @classmethod
    def load(cls, path: Path) -> "GroundSurface":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            return cls(
                elevation_m=data["elevation_m"].astype(np.float64),
                measured=data["measured"].astype(bool),
                confidence=data["confidence"].astype(np.float64),
                origin_x_m=float(meta["origin_x_m"]),
                origin_y_m=float(meta["origin_y_m"]),
                texel_m=float(meta["texel_m"]),
                source_id=str(meta["source_id"]),
                role=str(meta["role"]),
                provenance=dict(meta["provenance"]),
                qa_status=str(meta["qa_status"]),
            )


@dataclass(frozen=True)
class PatchBank:
    """Measured high-frequency residual patches; synthesis never rescales them."""

    patches_m: np.ndarray
    quality: np.ndarray
    source_index: np.ndarray
    source_ids: tuple[str, ...]
    texel_m: float
    overlap_cells: int
    provenance: dict[str, Any]

    @property
    def patch_cells(self) -> int:
        return int(self.patches_m.shape[1])

    def validate(self) -> None:
        p = np.asarray(self.patches_m)
        if p.ndim != 3 or p.shape[0] == 0 or p.shape[1] != p.shape[2]:
            raise ValueError("patches_m must be non-empty [patch, y, x] square patches")
        if not np.isfinite(p).all():
            raise ValueError("patch bank contains non-finite values")
        if not 0 < self.overlap_cells < p.shape[1] // 2 + 1:
            raise ValueError("overlap_cells must be positive and no greater than half a patch")
        if np.asarray(self.quality).shape != (p.shape[0],):
            raise ValueError("quality length does not match patches")
        if np.asarray(self.source_index).shape != (p.shape[0],):
            raise ValueError("source_index length does not match patches")
        if np.any(self.source_index < 0) or np.any(self.source_index >= len(self.source_ids)):
            raise ValueError("invalid source index")
        if self.texel_m <= 0:
            raise ValueError("texel_m must be positive")

    def save(self, path: Path) -> None:
        self.validate()
        meta = {
            "source_ids": list(self.source_ids),
            "texel_m": self.texel_m,
            "overlap_cells": self.overlap_cells,
            "provenance": self.provenance,
        }
        np.savez_compressed(
            path,
            patches_m=np.asarray(self.patches_m, dtype=np.float32),
            quality=np.asarray(self.quality, dtype=np.float32),
            source_index=np.asarray(self.source_index, dtype=np.int32),
            metadata=np.asarray(_metadata_json(meta)),
        )

    @classmethod
    def load(cls, path: Path) -> "PatchBank":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            bank = cls(
                patches_m=data["patches_m"].astype(np.float64),
                quality=data["quality"].astype(np.float64),
                source_index=data["source_index"].astype(np.int64),
                source_ids=tuple(str(v) for v in meta["source_ids"]),
                texel_m=float(meta["texel_m"]),
                overlap_cells=int(meta["overlap_cells"]),
                provenance=dict(meta["provenance"]),
            )
        bank.validate()
        return bank
