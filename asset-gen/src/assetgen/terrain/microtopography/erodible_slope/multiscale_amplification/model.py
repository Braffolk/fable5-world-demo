"""Result contracts for the multiscale amplification candidate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaleLedger:
    texel_m: float
    active_nodes: int
    routed_nodes: int
    eroded_m3: float
    deposited_m3: float
    exported_m3: float
    trapped_m3: float
    thermal_m3: float
    sediment_error_m3: float
    thermal_error_m3: float


@dataclass(frozen=True)
class AmplificationResult:
    c0_node_m: np.ndarray
    c1_node_m: np.ndarray
    active_node: np.ndarray
    flow_m3_s: np.ndarray
    eroded_depth_m: np.ndarray
    deposited_depth_m: np.ndarray
    thermal_delta_m: np.ndarray
    ledgers: tuple[ScaleLedger, ...]
    maximum_abs_relief_m: float
    protected_max_abs_m: float
    volume_change_m3: float
