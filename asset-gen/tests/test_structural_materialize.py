import hashlib
import json
from pathlib import Path

import numpy as np

from assetgen.config import EncodeConfig, GridConfig
from assetgen.height_geom import HeightChunkId
from assetgen.terrain.repair.baseline import BaselineTile
from assetgen.terrain.repair.hierarchy import (
    FineReevaluationResult,
    FineSurfaceWindow,
    HierarchyLayout,
)
from assetgen.terrain.repair.authority_inventory import (
    CanonicalBaselineInput,
    StructuralAuthorityPair,
    StructuralAuthorityInventory,
)
from assetgen.terrain.repair.materialize import (
    decode_fine_ownership,
    materialize_structural_hierarchy,
)
from assetgen.terrain.repair.model import StructuralTile
from assetgen.terrain.repair.plan import plan_structural_repair
from assetgen.terrain.repair.storage import (
    encode_authority_manifest,
    write_baseline_tile,
    write_structural_tile,
)


GRID = GridConfig(
    anchor_e=0,
    anchor_n=0,
    chunk_m=2048,
    lod_step=4,
    lods=(0, 1, 2, 3, 4),
    chunk_res=2048,
)
ENCODE = EncodeConfig("deflate", 0.01, 0.01, 19, 1)


def _digest(role: str, chunk: HeightChunkId) -> str:
    return hashlib.sha256(f"{role}:{chunk.lod}:{chunk.cx}:{chunk.cz}".encode()).hexdigest()


def test_authority_inventory_checks_pairs_and_bounds_decoded_cache(tmp_path: Path) -> None:
    chunks = (HeightChunkId(-2, 2, 3), HeightChunkId(-2, 3, 3))
    valid = np.ones((512, 512), dtype=bool)
    empty = np.zeros((512, 512), dtype=bool)
    height = np.full((512, 512), 74.0, dtype=np.float64)
    artifacts = []
    for chunk in chunks:
        baseline = write_baseline_tile(tmp_path, chunk, BaselineTile(height, valid))
        artifacts.append(
            write_structural_tile(
                tmp_path,
                chunk,
                StructuralTile(height, valid, empty, empty),
                evidence_sha256="44" * 32,
                baseline_artifact=baseline,
            )
        )
    baseline_authority = {
        "release": {
            "manifestSha256": "66" * 32,
            "heightIndexPath": "index/height.bin",
            "heightIndexBytes": 123,
            "heightIndexSha256": "77" * 32,
            "reconstructionVersion": "test-pinned-baseline/1",
        },
        "tiles": [
            {
                "key": [chunk.lod, chunk.cx, chunk.cz],
                "closureHaloSamples": 4,
                "manifestSha256": "66" * 32,
                "reconstructionVersion": "test-pinned-baseline/1",
                "dependencyRootSha256": _digest("dependency-root", chunk),
                "sourceAuthoritySha256": _digest("source-authority", chunk),
                "maximumMeanErrorMeters": 0.0,
                "meanErrorLimitMeters": 1e-12,
                "dependencies": [
                    {
                        "chunk": [0, 0, 0],
                        "content_relative_path": "c/height/0/0_0.test.bin",
                        "bytes": 456,
                        "sha256": _digest("source", chunk),
                        "index_hash64": 1234,
                        "decoded_sha256": _digest("decoded", chunk),
                        "qoffset": 70.0,
                        "qscale": 0.01,
                        "flags": 1,
                    }
                ],
            }
            for chunk in chunks
        ],
    }
    manifest_payload = encode_authority_manifest(
        recipe_sha256="55" * 32,
        evidence_sha256="44" * 32,
        artifacts=tuple(artifacts),
        baseline_authority=baseline_authority,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_payload)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()

    inventory = StructuralAuthorityInventory(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        expected_chunks=chunks,
        cache_tiles=1,
    )
    first = inventory.load(chunks[0])
    assert first.authority_manifest_sha256 == manifest_sha
    assert first.chunk == chunks[0]
    canonical = inventory.load_canonical(chunks[0])
    assert canonical.base_manifest_sha256 == "66" * 32
    np.testing.assert_array_equal(canonical.tile.height, first.paired_baseline.height)
    assert inventory.load(chunks[1]).chunk == chunks[1]
    assert inventory.cached_tile_count == 1


def test_disk_materializer_builds_full_closure_but_stages_review_only(
    tmp_path: Path,
) -> None:
    layout = HierarchyLayout(authority_core_res=8)
    corrected = (HeightChunkId(0, 0, 0), HeightChunkId(0, 1, 0))
    review = HeightChunkId(-1, 0, 0)
    plan = plan_structural_repair(corrected, review)
    assert len(plan.lod2_reducer) == 777
    assert len(plan.lod1_reducer) == 45

    def load_pair(chunk: HeightChunkId) -> StructuralAuthorityPair:
        height = np.full((8, 8), 80.0, dtype=np.float64)
        valid = np.ones((8, 8), dtype=bool)
        empty = np.zeros((8, 8), dtype=bool)
        return StructuralAuthorityPair(
            chunk=chunk,
            structural=StructuralTile(height, valid, empty, empty),
            paired_baseline=BaselineTile(height, valid),
            structural_sha256=_digest("structural", chunk),
            paired_baseline_sha256=_digest("paired", chunk),
            authority_manifest_sha256="22" * 32,
        )

    def load_canonical(chunk: HeightChunkId) -> CanonicalBaselineInput:
        height = np.full((8, 8), 80.0, dtype=np.float64)
        return CanonicalBaselineInput(
            chunk,
            BaselineTile(height, np.ones((8, 8), dtype=bool)),
            _digest("canonical", chunk),
            "33" * 32,
        )

    def preserve(context: FineSurfaceWindow) -> FineReevaluationResult:
        baseline = context.baseline_height[context.core_slice]
        shape = baseline.shape
        empty = np.zeros(shape, dtype=bool)
        return FineReevaluationResult(baseline, empty, empty)

    recipe_sha = "11" * 32
    authority_sha = "22" * 32
    base_sha = "33" * 32
    result = materialize_structural_hierarchy(
        grid=GRID,
        encode=ENCODE,
        plan=plan,
        recipe_sha256=recipe_sha,
        authority_manifest_sha256=authority_sha,
        canonical_base_manifest_sha256=base_sha,
        load_pair=load_pair,
        load_canonical_baseline=load_canonical,
        reevaluate_fine_surface=preserve,
        ownership_closure={
            "version": "test-shared-closure/1",
            "authorityManifestSha256": authority_sha,
            "campaignContentSha256": "44" * 32,
            "campaignQualificationSha256": "55" * 32,
            "scientificClosureSha256": "77" * 32,
            "rasterizer": {"id": "test-rasterizer", "sourceSha256": "66" * 32},
        },
        output_root=tmp_path / "transaction",
        layout=layout,
        pair_cache_tiles=9,
        canonical_cache_tiles=9,
    )

    assert len(result.fine) == 777
    assert len(result.parents) == 45
    assert [item.chunk for item in result.corrected_lod0] == list(corrected)
    assert [item.chunk for item in result.published_overlay] == [
        *plan.published_lod2,
        review,
    ]
    assert len(result.published_overlay) == 17
    assert len(list((result.root / "overlay" / "height" / "-2").glob("*.lac2"))) == 16
    assert len(list((result.root / "overlay" / "height" / "-1").glob("*.lac2"))) == 1
    assert len(list((result.root / "corrected-base" / "height" / "0").glob("*.lac2"))) == 2

    transaction = json.loads(result.manifest_path.read_text())
    assert transaction["recipeSha256"] == recipe_sha
    assert transaction["authorityManifestSha256"] == authority_sha
    assert transaction["canonicalBaseManifestSha256"] == base_sha
    assert transaction["fineSharedQoffset"] == 78.0
    assert transaction["worldOverlap"]["matchedEdges"] > 0
    assert not transaction["boundedMemory"]["completeHierarchyHeldInMemory"]
    assert transaction["boundedMemory"]["decodedChunkCache"] == 3
    assert len(transaction["fine"]) == 777
    assert len(transaction["parents"]) == 45
    assert len(transaction["correctedLod0"]) == 2
    assert transaction["publishedOverlay"] == [
        [chunk.lod, chunk.cx, chunk.cz] for chunk in (*plan.published_lod2, review)
    ]

    first_fine = result.fine[0]
    sidecar = json.loads((result.root / first_fine.sidecar_relative_path).read_text())
    assert sidecar["recipeSha256"] == recipe_sha
    assert len(sidecar["dependencies"]) == 9
    assert sidecar["dependencyMerkleRoot"] == first_fine.dependency_merkle_root
    assert sidecar["ownership"]["format"] == "TOM1/1"
    assert sidecar["ownership"]["closureSha256"]
    ownership_path = result.root / sidecar["ownership"]["path"]
    assert hashlib.sha256(ownership_path.read_bytes()).hexdigest() == sidecar[
        "ownership"
    ]["containerSha256"]
    authority, abstained = decode_fine_ownership(ownership_path.read_bytes())
    assert authority.shape == (33, 33)
    assert not authority.any()
    assert not abstained.any()
    assert transaction["fine"][0]["ownership"] == sidecar["ownership"]
