"""Materialize one bounded FLOAT candidate for ``sand.dune_aeolian``."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ....config import ASSET_GEN_ROOT, DATA_IN, DATA_WORK
from .conditions import DTM_SHEET, load_pilot_conditions
from .model import synthesize
from .qa import write_index, write_qa


REPOSITORY_ROOT = ASSET_GEN_ROOT.parent
TARGET = (
    DATA_WORK
    / "microtopography/mrzezyno/target/sha256/"
    "f6aa61f6ecd256d04175d9a9bf72809d488383e3695bd844eb388ff2bf829e9a/"
    "mrzezyno-morphology-target.npz"
)
DTM = DATA_IN / "dem_1m" / f"{DTM_SHEET}_dtm_1m.tif"
ETAK = DATA_IN / "etak/ETAK_EESTI_GPKG.gpkg"
GEOLOGY_200K = (
    REPOSITORY_ROOT
    / "docs/deep-research/microtopography-generation/library/data/egt/"
    "pinnakate-200k/q_avamus_a_200t.shp"
)
GEOLOGY_50K = (
    REPOSITORY_ROOT
    / "docs/deep-research/microtopography-generation/library/data/egt/"
    "pinnakate/Q_Avamus_a.shp"
)
OUTPUT_ROOT = DATA_WORK / "microtopography/sand-dune-aeolian/float/sha256"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha(path)}


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix(),
        **_identity(path),
    }


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")


def _write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def run(output_root: Path = OUTPUT_ROOT) -> Path:
    module_root = Path(__file__).parent
    sources = [_source(path) for path in (DTM, ETAK, GEOLOGY_200K, GEOLOGY_50K, TARGET)]
    implementation = {
        path.name: _sha(path)
        for path in sorted(module_root.glob("*.py"))
    }
    recipe = {
        "schemaVersion": "laas.sand-dune-aeolian-float-recipe/1",
        "status": "research_float_only",
        "regime": "sand.dune_aeolian",
        "sources": sources,
        "implementation": implementation,
        "method": {
            "name": "coast-aligned coherent exemplar amplification",
            "sourceRoles": {
                "MrzezynoHighReliefPatches": "shape dictionary and 1-8 m amplitude/correlation only",
                "MaaAmetDtm1m": "accepted parent landform position and >8 m organization",
                "EGT200k": "aeolian genesis and sand lithology authority",
                "EGT50k": "complete local surficial coverage corroboration; codes retained raw",
                "ETAK": "forest anchoring, shoreline orientation, hard/protected exclusions",
            },
            "subOneMeterInnovation": False,
            "runtimeSynthesis": False,
            "packedOutput": False,
            "worldCoordinateSolveCropLast": True,
            "parentMeanClosure": "smooth project prolong_structural_4x correction",
        },
    }
    build_id = hashlib.sha256(_canonical(recipe)).hexdigest()
    root = output_root / build_id
    manifest = root / "manifest.json"
    if manifest.exists():
        return manifest
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "recipe.json", _canonical(recipe))

    conditions = load_pilot_conditions(
        etak=ETAK,
        geology_200k=GEOLOGY_200K,
        geology_50k=GEOLOGY_50K,
    )
    result = synthesize(dtm_path=DTM, target_npz=TARGET, conditions=conditions)
    payload_path = root / "sand-dune-aeolian-float.npz"
    temporary = payload_path.with_suffix(".npz.part")
    with temporary.open("wb") as target:
        np.savez_compressed(
            target,
            parent_core_m=result.parent_core_m,
            structural_fine_m=result.structural_fine_m,
            residual_fine_m=result.residual_fine_m,
            absolute_fine_m=result.absolute_fine_m,
            authority_fine=conditions.authority_fine,
            hard_exclusion_fine=conditions.hard_exclusion_fine,
            protected_fine=conditions.protected_fine,
            chosen_atom_index=result.chosen_atom_index,
            amplitude_scale=result.amplitude_scale,
            source_residual_examples_m=result.source_residual_examples_m,
        )
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(payload_path)

    measurements = {
        "schemaVersion": "laas.sand-dune-aeolian-float-measurements/1",
        "buildId": build_id,
        "conditions": conditions.evidence,
        **result.metrics,
    }
    _write(root / "measurements.json", _canonical(measurements))
    qa_records = write_qa(root / "qa", result, conditions)
    write_index(root / "qa/index.json", qa_records, _identity)

    closure = result.metrics["closure"]["maximumAbsParentMeanErrorM"]
    amplitude = result.metrics["residual"]["absP95M"]
    alignment = result.metrics["residual"]["shoreTangentAlignmentAbsDot"]
    pack_worthy = closure <= 1.0e-9 and amplitude >= 0.01 and alignment >= 0.55
    verdict = {
        "schemaVersion": "laas.sand-dune-aeolian-float-verdict/1",
        "buildId": build_id,
        "verdict": "pack_worthy_research_candidate" if pack_worthy else "park",
        "scope": "mapped forest-anchored coastal aeolian sand only",
        "evidenceBoundary": (
            "Mrzezyno authorizes high-relief Baltic dune shapes and 1-8 m "
            "amplitude/correlation. It does not authorize beach/wet-dry/water "
            "morphology or independent detail below 1 m."
        ),
        "resumeConditionIfParked": (
            "Resume only with either an independently qualified Baltic high-relief "
            "dune exemplar that improves coherent form transfer, or a visual finding "
            "showing a specific repeat/grid/orientation failure in these QA panels."
        ),
        "automaticChecks": {
            "parentMeanClosurePass": closure <= 1.0e-9,
            "nontrivialGeometryPass": amplitude >= 0.01,
            "shoreOrientationPass": alignment >= 0.55,
        },
        "automaticChecksAreNotVisualAcceptance": True,
    }
    _write(root / "verdict.json", _canonical(verdict))
    document = {
        "schemaVersion": "laas.sand-dune-aeolian-float-manifest/1",
        "buildId": build_id,
        "status": "complete",
        "recipe": {"path": "recipe.json", **_identity(root / "recipe.json")},
        "payload": {"path": payload_path.name, **_identity(payload_path)},
        "measurements": {"path": "measurements.json", **_identity(root / "measurements.json")},
        "qaIndex": {"path": "qa/index.json", **_identity(root / "qa/index.json")},
        "verdict": {"path": "verdict.json", **_identity(root / "verdict.json")},
    }
    _write(manifest, _canonical(document))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    print(run(arguments.output_root))


if __name__ == "__main__":
    main()
