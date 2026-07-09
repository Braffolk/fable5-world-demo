"""Configuration loading: base.toml (world grid, immutable) + per-AOI TOMLs."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ASSET_GEN_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ASSET_GEN_ROOT / "config"
DATA_IN = ASSET_GEN_ROOT / "data" / "in"
DATA_WORK = ASSET_GEN_ROOT / "data" / "work"
DATA_OUT = ASSET_GEN_ROOT / "data" / "out"


@dataclass(frozen=True)
class GridConfig:
    anchor_e: int
    anchor_n: int
    chunk_m: int
    lod_step: int
    lods: tuple[int, ...]
    chunk_res: int


@dataclass(frozen=True)
class EncodeConfig:
    codec: str
    height_qscale: float
    water_qscale: float
    zstd_level: int
    deflate_level: int


@dataclass(frozen=True)
class FetchConfig:
    user_agent: str
    min_interval_s: float
    max_retries: int


@dataclass(frozen=True)
class BaseConfig:
    grid: GridConfig
    encode: EncodeConfig
    fetch: FetchConfig
    attribution: dict[str, str]


@dataclass(frozen=True)
class AoiConfig:
    name: str
    layers: dict[str, bool]
    # Either an explicit centered box…
    center_e: int | None = None
    center_n: int | None = None
    half_m: int | None = None
    # …or the whole country (bbox derived from the sheet grid index at plan time).
    whole_country: bool = False

    def bbox_en(self) -> tuple[int, int, int, int] | None:
        """(min_e, min_n, max_e, max_n) or None when whole_country (resolved later)."""
        if self.whole_country:
            return None
        assert self.center_e is not None and self.center_n is not None and self.half_m is not None
        return (
            self.center_e - self.half_m,
            self.center_n - self.half_m,
            self.center_e + self.half_m,
            self.center_n + self.half_m,
        )


def load_base() -> BaseConfig:
    raw = tomllib.loads((CONFIG_DIR / "base.toml").read_text())
    g = raw["grid"]
    e = raw["encode"]
    f = raw["fetch"]
    return BaseConfig(
        grid=GridConfig(
            anchor_e=g["anchor_e"],
            anchor_n=g["anchor_n"],
            chunk_m=g["chunk_m"],
            lod_step=g["lod_step"],
            lods=tuple(g["lods"]),
            chunk_res=g["chunk_res"],
        ),
        encode=EncodeConfig(
            codec=e["codec"],
            height_qscale=e["height_qscale"],
            water_qscale=e["water_qscale"],
            zstd_level=e["zstd_level"],
            deflate_level=e["deflate_level"],
        ),
        fetch=FetchConfig(
            user_agent=f["user_agent"],
            min_interval_s=f["min_interval_s"],
            max_retries=f["max_retries"],
        ),
        attribution=dict(raw["attribution"]),
    )


def load_aoi(name: str) -> AoiConfig:
    path = CONFIG_DIR / f"aoi-{name}.toml"
    if not path.exists():
        available = sorted(p.stem.removeprefix("aoi-") for p in CONFIG_DIR.glob("aoi-*.toml"))
        raise FileNotFoundError(f"no AOI config {path.name}; available: {available}")
    raw = tomllib.loads(path.read_text())
    return AoiConfig(
        name=raw["name"],
        layers=dict(raw.get("layers", {})),
        center_e=raw.get("center_e"),
        center_n=raw.get("center_n"),
        half_m=raw.get("half_m"),
        whole_country=raw.get("whole_country", False),
    )
