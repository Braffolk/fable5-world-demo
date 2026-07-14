from assetgen.config import CONFIG_DIR
from assetgen.height_geom import HeightChunkId
from assetgen.pilots.taevaskoda_authority import (
    AUTHORITY_EVIDENCE_BINDING_VERSION,
    baseline_authority_identity,
    load_authority_config,
)
from assetgen.terrain.repair.storage import baseline_tile_contract


def test_frozen_authority_config_is_the_complete_canonical_closure() -> None:
    config = load_authority_config(
        CONFIG_DIR / "terrain-repair/taevaskoda-ahja-authority-stage1.json"
    )

    assert config.plan.authority_lod0 == HeightChunkId(0, 151, 93)
    assert config.plan.corrected_lod0 == (
        HeightChunkId(0, 151, 93),
        HeightChunkId(0, 152, 93),
    )
    assert config.plan.review_parent == HeightChunkId(-1, 607, 372)
    assert len(config.plan.lod1_reducer) == 45
    assert config.plan.lod1_reducer[0] == HeightChunkId(-1, 604, 372)
    assert config.plan.lod1_reducer[-1] == HeightChunkId(-1, 612, 376)
    assert len(config.plan.lod2_reducer) == 777
    assert config.plan.lod2_reducer[0] == HeightChunkId(-2, 2416, 1488)
    assert config.plan.lod2_reducer[-1] == HeightChunkId(-2, 2452, 1508)
    assert len(config.plan.authority_support) == 897
    assert config.plan.authority_support[0] == HeightChunkId(-2, 2415, 1487)
    assert config.plan.authority_support[-1] == HeightChunkId(-2, 2453, 1509)
    assert [path.name for path in config.dtm_sources] == [
        "54472_dtm_1m.tif",
        "54474_dtm_1m.tif",
        "54481_dtm_1m.tif",
        "54483_dtm_1m.tif",
    ]
    assert config.pinned_manifest.name == "manifest.json"
    assert config.pinned_content_root.name == "out"
    assert config.raw["pinnedBase"]["manifestSha256"] == (
        "708478a57c2118eaa618867e87cc74a35e20c615999b60d7e4177b595ef5495a"
    )
    assert config.raw["closure"]["lod2Reducer"]["boundsEn"] == [
        677888,
        6442368,
        682624,
        6445056,
    ]
    assert config.raw["closure"]["authoritySupport"]["boundsEn"] == [
        677760,
        6442240,
        682752,
        6445184,
    ]
    assert len(config.canonical_sha256) == 64


def test_baseline_authority_identity_requires_persisted_exact_artifacts() -> None:
    identity = baseline_authority_identity()

    assert identity["persistedArtifact"] is True
    assert identity["contract"] == baseline_tile_contract()
    assert identity["contract"]["role"] == "unmodified_structural_baseline_0.25m"
    assert identity["source"] == "pinned browser-decoded format-1 LOD0 height release"
    assert identity["release"]["heightIndexSha256"] == (
        "7fd5987a4414e260056e189e754a2b7bc653d9ff86e2fa05414d49ac458492a3"
    )
    assert identity["reconstruction"]["boundedDecodedChunkLru"] == 4


def test_authority_evidence_binding_keeps_qualification_and_content_distinct() -> None:
    assert AUTHORITY_EVIDENCE_BINDING_VERSION == "qualification-and-campaign-content/1"
