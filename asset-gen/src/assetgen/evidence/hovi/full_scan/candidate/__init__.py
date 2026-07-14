"""View-aware Hovi candidate inputs without a surface-selection policy."""

from .accumulators import (
    CellEvidenceLevel,
    ContinuityEvidence,
    MultiscaleShardAccumulator,
    MultiscaleShardEvidence,
    StratumCounts,
    VerticalSampleBatch,
    VerticalSampleGroup,
)
from .manifest import (
    RESOLUTIONS_M,
    PointShard,
    ScanTransform,
    VerifiedSpatialManifest,
    load_verified_spatial_manifest,
)
from .hypotheses import (
    CandidateSurfaceHypothesisGenerator,
    CellHypothesisSet,
    HypothesisConfig,
    ScanSheetSupport,
    SheetHypothesis,
)
from .records import ObservationBatch, SpatialObservationReader

__all__ = [
    "CellEvidenceLevel",
    "CellHypothesisSet",
    "CandidateSurfaceHypothesisGenerator",
    "ContinuityEvidence",
    "HypothesisConfig",
    "MultiscaleShardAccumulator",
    "MultiscaleShardEvidence",
    "ObservationBatch",
    "PointShard",
    "RESOLUTIONS_M",
    "ScanTransform",
    "ScanSheetSupport",
    "SheetHypothesis",
    "SpatialObservationReader",
    "StratumCounts",
    "VerifiedSpatialManifest",
    "VerticalSampleBatch",
    "VerticalSampleGroup",
    "load_verified_spatial_manifest",
]
