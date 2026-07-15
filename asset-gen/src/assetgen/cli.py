"""assetgen CLI — reproducible fetch + cook of Estonian geodata into streamable chunks.

Quickstart (fresh clone):
    cd asset-gen
    uv run assetgen plan  --aoi taevaskoja      # show what would be fetched
    uv run assetgen fetch --aoi taevaskoja      # populate data/in (idempotent, resumable)
    uv run assetgen cook  --aoi taevaskoja      # cook data/out chunks (later phases)
"""
from __future__ import annotations

from pathlib import Path

import click

from .config import CONFIG_DIR, DATA_IN, DATA_WORK, load_aoi, load_base
from .fetch.http import PoliteSession
from .grid import chunks_covering_bbox_en, snap_bbox_to_chunks_en
from .sheets import grid_union_bbox, load_sheet_grid


def _resolve_bbox(session, aoi, grids_dir):
    """AOI bbox in EPSG:3301, chunk-snapped; whole_country resolves from the sheet grid."""
    base = load_base()
    if aoi.whole_country:
        from .fetch.maaamet import ensure_sheet_grids

        ensure_sheet_grids(session, grids_dir)
        bbox = grid_union_bbox(load_sheet_grid(grids_dir, "10k"))
    else:
        bbox = aoi.bbox_en()
    return snap_bbox_to_chunks_en(base.grid, bbox, 0)


@click.group()
def main() -> None:
    pass


@main.command()
@click.option("--aoi", required=True, help="AOI config name (config/aoi-<name>.toml)")
def plan(aoi: str) -> None:
    """Print the fetch plan + chunk coverage for an AOI, with size estimates."""
    base, cfg = load_base(), load_aoi(aoi)
    session = PoliteSession(base.fetch)
    from .fetch.maaamet import ensure_sheet_grids, plan_country, plan_elevation

    grids_dir = ensure_sheet_grids(session)
    bbox = _resolve_bbox(session, cfg, grids_dir)
    click.echo(f"AOI {cfg.name}: chunk-snapped bbox E {bbox[0]}..{bbox[2]}, N {bbox[1]}..{bbox[3]}")
    for lod in base.grid.lods:
        n = len(chunks_covering_bbox_en(base.grid, bbox, lod))
        click.echo(f"  LOD{lod} ({base.grid.lod_step**lod} m/texel): {n} chunks")
    items = plan_elevation(session, bbox, grids_dir, want_chm=False)
    n_dem = sum(1 for i in items if "dem_1m" in str(i.dest))
    n_ndsm = sum(1 for i in items if "ndsm_1m" in str(i.dest))
    click.echo(f"  DTM 1m sheets: {n_dem} (~{n_dem * 73 / 1024:.1f} GB)")
    click.echo(f"  nDSM 1m sheets: {n_ndsm} (~{n_ndsm * 4 / 1024:.1f} GB)")
    click.echo(f"  + CHM sheets (searched at fetch time), country DTM_10m (~2 GB), ETAK GPKG, WFS layers")


@main.command()
@click.option("--aoi", required=True)
@click.option("--only", type=click.Choice(["elevation", "country", "etak", "wfs"]), default=None,
              help="Fetch just one source group (default: everything the AOI needs)")
def fetch(aoi: str, only: str | None) -> None:
    """Populate data/in for an AOI. Idempotent: re-runs skip verified files."""
    base, cfg = load_base(), load_aoi(aoi)
    session = PoliteSession(base.fetch)
    from .fetch.etak import fetch_etak
    from .fetch.maaamet import ensure_sheet_grids, plan_country, plan_elevation, run_fetch
    from .fetch.wfs import fetch_wfs_layer

    grids_dir = ensure_sheet_grids(session)
    bbox = _resolve_bbox(session, cfg, grids_dir)

    if only in (None, "country"):
        run_fetch(session, plan_country(session), log=click.echo)
    if only in (None, "elevation"):
        need_veg = cfg.layers.get("trees", False)
        items = plan_elevation(
            session, bbox, grids_dir, want_dem1m=True, want_ndsm=need_veg, want_chm=need_veg
        )
        click.echo(f"elevation: {len(items)} files")
        run_fetch(session, items, log=click.echo)
    if only in (None, "etak") and (cfg.layers.get("landcover") or cfg.layers.get("water")):
        fetch_etak(session, log=click.echo)
    if only in (None, "wfs") and cfg.layers.get("trees"):
        # Layer names verified against GetCapabilities at runtime; eraldis = forest stands.
        from .fetch.wfs import list_layers

        # only the polygon stand layer — eraldis_element is a geometry-less attribute
        # table and rejects bbox filters (HTTP 400)
        names = [n for n in list_layers(session, "metsaregister") if n.lower().endswith("eraldis")]
        for name in names:
            fetch_wfs_layer(session, "metsaregister", name, bbox, log=click.echo)
    click.echo("fetch complete.")


@main.command("country-floor")
@click.option("--workers", default=6, show_default=True)
@click.option("--min-lod", default=2, show_default=True,
              help="Coarsest-to-finest floor: cook this LOD and coarser over ALL Estonia")
def country_floor(workers: int, min_lod: int) -> None:
    """Cook the always-resident coarse height floor for the whole country from DTM_10m.

    Guarantees the renderer's no-hole backstop: every chunk (incl. sea) exists at the
    coarse LODs. Fetch the 10 m DTM first (`assetgen fetch --aoi estonia --only country`).
    """
    base = load_base()
    session = PoliteSession(base.fetch)
    from .cook.height_cook import cook_height
    from .fetch.maaamet import ensure_sheet_grids

    grids_dir = ensure_sheet_grids(session)
    bbox = _resolve_bbox(session, load_aoi("estonia"), grids_dir)
    lods = tuple(lod for lod in base.grid.lods if lod >= min_lod)
    click.echo(f"country floor: LODs {list(lods)} over E {bbox[0]}..{bbox[2]}, N {bbox[1]}..{bbox[3]}")
    cook_height(base, bbox, lods=lods, workers=workers, log=click.echo)
    click.echo("country floor complete.")


@main.command("country-floor-biome")
@click.option("--workers", default=6, show_default=True)
@click.option("--min-lod", default=2, show_default=True,
              help="Coarsest-to-finest floor: cook this LOD and coarser over ALL Estonia")
def country_floor_biome(workers: int, min_lod: int) -> None:
    """Cook the always-resident coarse biome floor for the whole country from ETAK landcover.

    Mirrors `country-floor` (height): beyond the 16 km pilot the reduced coarse biome rungs
    are absent, so far terrain renders as bare soil. This rasterizes the whole-country ETAK
    landcover directly at the coarse texel so far forests carry vegDensity (green up), not
    dirt. Fetch ETAK first (`assetgen fetch --aoi estonia --only etak`); matches height's
    country pyramid extent. Run AFTER the pilot cook so its finer rungs are gap-filled.
    """
    base = load_base()
    session = PoliteSession(base.fetch)
    from .cook.layers_cook import cook_biome_floor
    from .fetch.maaamet import ensure_sheet_grids

    grids_dir = ensure_sheet_grids(session)
    bbox = _resolve_bbox(session, load_aoi("estonia"), grids_dir)
    lods = tuple(lod for lod in base.grid.lods if lod >= min_lod)
    click.echo(f"country biome floor: LODs {list(lods)} over E {bbox[0]}..{bbox[2]}, N {bbox[1]}..{bbox[3]}")
    cook_biome_floor(base, bbox, lods=lods, workers=workers, log=click.echo)
    click.echo("country biome floor complete.")


@main.command()
@click.option("--aoi", required=True)
@click.option("--layer", "layers_opt", multiple=True,
              type=click.Choice(["height", "biome", "water", "waterbed", "canopy", "soil", "trees",
                                  "understory", "debris", "boulders"]),
              help="Cook only these layers (default: all the AOI enables)")
@click.option("--workers", default=6, show_default=True)
def cook(aoi: str, layers_opt: tuple[str, ...], workers: int) -> None:
    """Cook data/in sources into LAC1 chunks under data/work/chunks."""
    base, cfg = load_base(), load_aoi(aoi)
    session = PoliteSession(base.fetch)
    from .fetch.maaamet import ensure_sheet_grids

    bbox = _resolve_bbox(session, cfg, ensure_sheet_grids(session))
    wanted = set(layers_opt) or {
        {"landcover": "biome"}.get(k, k) for k, v in cfg.layers.items() if v
    }
    if "height" in wanted:
        from .cook.height_cook import cook_height

        cook_height(base, bbox, workers=workers, log=click.echo)
    if "biome" in wanted:
        from .cook.layers_cook import cook_biome

        cook_biome(base, bbox, log=click.echo)
    if "water" in wanted:
        from .cook.layers_cook import cook_water

        cook_water(base, bbox, log=click.echo)
    # waterbed is a POST-PASS over cooked height+water: carve the submerged bed (#104) and
    # emit the anti-aliased shore-coverage layer (#114). Not a default layer — request it
    # explicitly with `--layer waterbed` after height + water are cooked.
    if "waterbed" in wanted:
        from .cook.layers_cook import cook_waterbed

        cook_waterbed(base, bbox, log=click.echo)
    if "canopy" in wanted:
        from .cook.layers_cook import cook_canopy

        cook_canopy(base, bbox, log=click.echo)
    if "soil" in wanted:
        from .cook.layers_cook import cook_soil

        cook_soil(base, bbox, log=click.echo)
    if "trees" in wanted:
        from .cook.trees_cook import cook_trees

        cook_trees(base, bbox, log=click.echo)
    # understory/debris read the cooked biome+soil planes, so they run after them
    if "understory" in wanted:
        from .cook.layers_cook import cook_understory

        cook_understory(base, bbox, log=click.echo)
    if "debris" in wanted:
        from .cook.layers_cook import cook_debris

        cook_debris(base, bbox, log=click.echo)
    if "boulders" in wanted:
        from .cook.layers_cook import cook_boulders

        cook_boulders(base, bbox, log=click.echo)
    known = {"height", "biome", "water", "waterbed", "canopy", "soil", "trees", "understory", "debris", "boulders"}
    for name in sorted(wanted - known):
        click.echo(f"(layer {name}: cooker lands in a later phase)")
    click.echo("cook complete.")


@main.command("manifest")
@click.option("--cook-rev", default=1, show_default=True)
def manifest_cmd(cook_rev: int) -> None:
    """Disabled legacy publisher; use the recipe-addressed release transaction."""
    del cook_rev
    raise click.ClickException(
        "legacy manifest publication is disabled because it globs shared data/work/chunks; "
        "use release-plan, build --no-latest, then publish"
    )


@main.command("micro-plan")
@click.option("--build", "build_digest", required=True, help="64-hex recipe digest")
@click.option("--parent-cx", required=True, type=int, help="Signed height LOD -1 parent X")
@click.option("--parent-cz", required=True, type=int, help="Signed height LOD -1 parent Z")
@click.option("--base-manifest", required=True, type=click.Path(path_type=Path))
@click.option("--base-sha256", required=True, help="Full SHA-256 of --base-manifest")
@click.option("--base-out-root", required=True, type=click.Path(path_type=Path))
@click.option("--exemplar-manifest", type=click.Path(path_type=Path), default=None)
def micro_plan_cmd(
    build_digest: str,
    parent_cx: int,
    parent_cz: int,
    base_manifest: Path,
    base_sha256: str,
    base_out_root: Path,
    exemplar_manifest: Path | None,
) -> None:
    """Freeze the complete 16+1 Stage-1 coverage before any chunk is cooked."""
    from .release import create_micro_expectation

    path = create_micro_expectation(
        load_base(), build_digest, parent_cx, parent_cz,
        base_manifest, base_sha256, base_out_root=base_out_root,
        exemplar_manifest_path=exemplar_manifest,
    )
    click.echo(f"expectation: {path}")
    click.echo("published: 16 LOD -2 + 1 LOD -1; transient support: 9 LOD -2")


@main.command("evidence-fetch-als")
@click.option(
    "--selection",
    type=click.Path(path_type=Path),
    default=None,
    help="Strict ALS selection JSON; defaults to config/taevaskoda-als.json",
)
@click.option("--year", "years", multiple=True, type=int, help="Fetch only selected years")
def evidence_fetch_als_cmd(selection: Path | None, years: tuple[int, ...]) -> None:
    """Retain preregistered public ALS bytes with HTTP and license provenance."""
    from .fetch.als import fetch_als_selection

    path = fetch_als_selection(load_base(), selection, years=years, log=click.echo)
    click.echo(f"retention manifest: {path}")


@main.command("evidence-fetch-orthophoto-stage1")
def evidence_fetch_orthophoto_stage1_cmd() -> None:
    """Retain the exact Taevaskoda Stage-1 RGB/CIR orthophoto snapshot."""
    from .fetch.orthophoto import fetch_taevaskoda_stage1_orthophoto

    path = fetch_taevaskoda_stage1_orthophoto(load_base(), log=click.echo)
    click.echo(f"retention manifest: {path}")


@main.command("evidence-fetch-orthophoto-sheet")
@click.option("--sheet", required=True, help="Explicitly frozen Maa-amet sheet id.")
def evidence_fetch_orthophoto_sheet_cmd(sheet: str) -> None:
    """Retain current workbook-bound RGB/CIR for a frozen sheet."""
    from .fetch.orthophoto import fetch_frozen_orthophoto_sheet

    path = fetch_frozen_orthophoto_sheet(load_base(), sheet, log=click.echo)
    click.echo(f"retention manifest: {path}")


@main.command("evidence-fetch-hovi")
@click.option(
    "--through",
    type=click.Choice(("shared", "hy-spruce4-photos", "hy-spruce4-geometry")),
    default="hy-spruce4-geometry",
    show_default=True,
)
def evidence_fetch_hovi_cmd(through: str) -> None:
    """Retain the frozen shared and HY_SPRUCE4 Hovi evidence tranches."""
    from .fetch.hovi import fetch_hovi_selection

    path = fetch_hovi_selection(load_base(), through=through, log=click.echo)
    click.echo(f"retention manifest: {path}")


@main.command("evidence-condition-hovi")
@click.option("--retained", required=True, type=click.Path(path_type=Path))
@click.option("--plot", default="HY_SPRUCE4", show_default=True)
def evidence_condition_hovi_cmd(retained: Path, plot: str) -> None:
    """Materialize condition evidence for an unsealed Hovi development plot."""
    from .evidence.hovi.conditions import emit_hovi_conditions

    path = emit_hovi_conditions(retained, plot)
    click.echo(f"condition evidence: {path}")


@main.command("evidence-convert-hovi")
@click.option("--retained", required=True, type=click.Path(path_type=Path))
@click.option("--plot", default="HY_SPRUCE4", show_default=True)
def evidence_convert_hovi_cmd(retained: Path, plot: str) -> None:
    """Inventory multiscale raw observation support for a Hovi development plot."""
    from .evidence.hovi.convert import convert_hovi_plot

    path = convert_hovi_plot(retained, plot, log=click.echo)
    click.echo(f"artifact manifest: {path}")


@main.command("evidence-inventory-als")
@click.option(
    "--retained",
    type=click.Path(path_type=Path),
    default=None,
    help="Complete Taevaskoda retained.json; defaults to the frozen public evidence path",
)
def evidence_inventory_als_cmd(retained: Path | None) -> None:
    """Inventory retained ALS schemas and point semantics without raster conversion."""
    from .evidence.als_inventory import inventory_taevaskoda_als

    path = inventory_taevaskoda_als(retained, log=click.echo)
    click.echo(f"inventory: {path}")


@main.command("structural-stage1")
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    default=CONFIG_DIR / "terrain-repair/taevaskoda-ahja-authority-stage1.json",
    show_default=True,
)
@click.option("--work-root", type=click.Path(path_type=Path), default=DATA_WORK,
              show_default=True)
def structural_stage1_cmd(config: Path, work_root: Path) -> None:
    """Resume the immutable Ahja structural-repair transaction through preview."""
    from .terrain.repair.stage1_transaction import run_stage1_transaction

    result = run_stage1_transaction(config, work_root=work_root)
    click.echo(f"verified preview transaction: {result.manifest_path}")
    click.echo(f"transaction sha256: {result.transaction_sha256}")
    click.echo(f"release ready: {'yes' if result.release_ready else 'no'}")


@main.command("micro-recipe")
@click.option("--parent-cx", required=True, type=int)
@click.option("--parent-cz", required=True, type=int)
@click.option("--base-manifest", required=True, type=click.Path(path_type=Path))
@click.option("--exemplar-manifest", type=click.Path(path_type=Path), default=None)
def micro_recipe_cmd(
    parent_cx: int,
    parent_cz: int,
    base_manifest: Path,
    exemplar_manifest: Path | None,
) -> None:
    """Print a content-derived fixture or measured-synthesis recipe digest."""
    from .micro_recipe import derive_micro_fixture_recipe, derive_micro_synthesis_recipe

    if exemplar_manifest is None:
        digest, _ = derive_micro_fixture_recipe(parent_cx, parent_cz, base_manifest)
    else:
        digest, _ = derive_micro_synthesis_recipe(
            parent_cx, parent_cz, base_manifest, exemplar_manifest
        )
    click.echo(digest)


@main.command("micro-exemplars-prepare")
@click.option("--source-config", required=True, type=click.Path(path_type=Path))
@click.option("--output-root", required=True, type=click.Path(path_type=Path))
def micro_exemplars_prepare_cmd(source_config: Path, output_root: Path) -> None:
    """Prepare the reviewed TLS surfaces and their content-bound patch bank."""
    from .process.microtopo import prepare_lapinjarvi_bank

    path = prepare_lapinjarvi_bank(source_config, output_root)
    click.echo(f"exemplar manifest: {path}")


@main.command("micro-fixture-cook")
@click.option("--build", "build_digest", required=True, help="64-hex recipe digest")
@click.option("--base-manifest", required=True, type=click.Path(path_type=Path))
@click.option("--base-out-root", required=True, type=click.Path(path_type=Path))
def micro_fixture_cook_cmd(
    build_digest: str,
    base_manifest: Path,
    base_out_root: Path,
) -> None:
    """Cook the calibrated Stage-1 retention fixture, not production morphology."""
    from .cook.micro_fixture_cook import cook_micro_fixture
    from .micro_config import load_micro_config

    base = load_base()
    path = cook_micro_fixture(
        base,
        load_micro_config(base),
        build_digest,
        base_manifest,
        base_out_root,
        DATA_WORK,
        log=click.echo,
    )
    click.echo(f"fixture evidence: {path}")


@main.command("micro-synthesis-cook")
@click.option("--build", "build_digest", required=True, help="64-hex recipe digest")
@click.option("--base-manifest", required=True, type=click.Path(path_type=Path))
@click.option("--base-out-root", required=True, type=click.Path(path_type=Path))
@click.option("--exemplar-manifest", required=True, type=click.Path(path_type=Path))
def micro_synthesis_cook_cmd(
    build_digest: str,
    base_manifest: Path,
    base_out_root: Path,
    exemplar_manifest: Path,
) -> None:
    """Cook the measured-exemplar pilot into immutable preview artifacts."""
    from .cook.micro_synth_cook import cook_micro_synthesis
    from .micro_config import load_micro_config

    base = load_base()
    path = cook_micro_synthesis(
        base,
        load_micro_config(base),
        build_digest,
        base_manifest,
        base_out_root,
        DATA_WORK,
        exemplar_manifest,
        log=click.echo,
    )
    click.echo(f"synthesis evidence: {path}")


@main.command("micro-agriculture-r0")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=CONFIG_DIR / "microtopography/agriculture-cultivated-r0.json",
    show_default=True,
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=DATA_WORK / "microtopography/agriculture/sha256",
    show_default=True,
)
def micro_agriculture_r0_cmd(config_path: Path, output_root: Path) -> None:
    """Materialize one research-only cultivated 128 m development surface."""
    from .terrain.microtopography.agriculture import build_cultivated_r0

    path = build_cultivated_r0(config_path, output_root=output_root)
    click.echo(f"agriculture R0 manifest: {path}")


@main.command("condition-soil-window")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Named Mullastikukaart window selection JSON.",
)
@click.option(
    "--bbox",
    type=float,
    nargs=4,
    default=None,
    metavar="MIN_E MIN_N MAX_E MAX_N",
    help="Direct EPSG:3301 window; mutually exclusive with --config.",
)
@click.option(
    "--name",
    default="direct-bbox",
    show_default=True,
    help="Artifact slug used with --bbox.",
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=DATA_WORK / "terrain/conditions/soil/mullastikukaart/sha256",
    show_default=True,
)
def condition_soil_window_cmd(
    config_path: Path | None,
    bbox: tuple[float, float, float, float] | None,
    name: str,
    output_root: Path,
) -> None:
    """Snapshot complete Mullastikukaart polygons intersecting one window."""
    from .terrain.conditions.soil import extract_soil_window, load_window_selection

    if (config_path is None) == (bbox is None):
        raise click.UsageError("provide exactly one of --config or --bbox")
    if config_path is not None:
        selected_name, selected_bbox, selection_source = load_window_selection(
            config_path
        )
    else:
        assert bbox is not None
        selected_name = name
        selected_bbox = bbox
        selection_source = {"kind": "direct_cli_bbox"}
    path = extract_soil_window(
        selected_bbox,
        name=selected_name,
        selection_source=selection_source,
        output_root=output_root,
    )
    click.echo(f"soil condition window: {path}")


@main.command("condition-soil-profile-inventory")
@click.option(
    "--output-root",
    type=click.Path(path_type=Path),
    default=DATA_WORK
    / "terrain/conditions/soil/mullastikukaart/profile-inventory/sha256",
    show_default=True,
)
def condition_soil_profile_inventory_cmd(output_root: Path) -> None:
    """Inventory national support for the official soil-profile grammar."""
    from .terrain.conditions.soil import inventory_profile_coverage

    path = inventory_profile_coverage(output_root)
    click.echo(f"soil profile coverage inventory: {path}")


@main.command("micro-fixture-verify")
@click.option("--build", "build_digest", required=True, help="64-hex recipe digest")
@click.option("--base-manifest", required=True, type=click.Path(path_type=Path))
@click.option("--base-out-root", required=True, type=click.Path(path_type=Path))
def micro_fixture_verify_cmd(
    build_digest: str,
    base_manifest: Path,
    base_out_root: Path,
) -> None:
    """Independently reopen and verify the complete Stage-1 fixture closure."""
    from .micro_verify import verify_micro_fixture

    path = verify_micro_fixture(
        build_digest, base_manifest, base_out_root, DATA_WORK
    )
    click.echo(f"micro verification: {path}")


@main.command("release-plan")
@click.option("--build", "build_digest", required=True, help="64-hex recipe digest")
@click.option("--cook-rev", default=1, show_default=True)
@click.option("--format", "manifest_format", type=click.Choice(["1", "2"]), default="1",
              show_default=True, help="Release manifest/index format")
@click.option("--micro-parent-cx", type=int, default=None, help="Declared LOD -1 proof parent X")
@click.option("--micro-parent-cz", type=int, default=None, help="Declared LOD -1 proof parent Z")
@click.option("--base-manifest", type=click.Path(path_type=Path), default=None)
@click.option("--base-sha256", default=None, help="Full SHA-256 of --base-manifest")
@click.option("--base-out-root", type=click.Path(path_type=Path), default=None,
              help="Release root containing the base manifest's c/ objects")
def release_plan_cmd(
    build_digest: str,
    cook_rev: int,
    manifest_format: str,
    micro_parent_cx: int | None,
    micro_parent_cz: int | None,
    base_manifest: Path | None,
    base_sha256: str | None,
    base_out_root: Path | None,
) -> None:
    """Freeze the exact chunks under data/work/builds/<digest>/chunks."""
    from .release import create_build_plan

    kwargs = {}
    if base_out_root is not None:
        kwargs["base_out_root"] = base_out_root
    if (micro_parent_cx is None) != (micro_parent_cz is None):
        raise click.ClickException("--micro-parent-cx and --micro-parent-cz must be provided together")
    micro_parent = (
        (micro_parent_cx, micro_parent_cz)
        if micro_parent_cx is not None and micro_parent_cz is not None
        else None
    )
    path = create_build_plan(
        load_base(), build_digest, cook_rev,
        base_manifest_path=base_manifest, base_manifest_sha256=base_sha256,
        manifest_format=int(manifest_format), micro_parent=micro_parent, **kwargs,
    )
    click.echo(f"plan: {path}")


@main.command("build")
@click.option("--build", "build_digest", required=True, help="64-hex recipe digest")
@click.option("--no-latest", is_flag=True, help="Required: build an immutable preview only")
def build_cmd(build_digest: str, no_latest: bool) -> None:
    """Verify one frozen build and materialize a non-latest preview release."""
    if not no_latest:
        raise click.ClickException("--no-latest is required; only publish may update latest.json")
    from .release import materialize_preview

    path = materialize_preview(build_digest)
    click.echo(f"preview manifest: {path}")


@main.command("publish")
@click.option("--build", "build_digest", required=True, help="64-hex recipe digest")
def publish_cmd(build_digest: str) -> None:
    """Publish a COMPLETE verified build, updating latest.json as the final write."""
    from .release import publish_build

    path = publish_build(build_digest)
    click.echo(f"published manifest: {path}")


@main.command()
@click.option("--aoi", required=True)
@click.option("--composite", is_flag=True, help="Also render the fused all-layers debug image")
@click.option("--composite-scale", default=1, show_default=True, help="meters/pixel for --composite")
def preview(aoi: str, composite: bool, composite_scale: int) -> None:
    """Render QA PNGs (hillshade etc.) into data/out/preview."""
    base, cfg = load_base(), load_aoi(aoi)
    session = PoliteSession(base.fetch)
    from .fetch.maaamet import ensure_sheet_grids
    from .preview import (
        debug_composite,
        preview_ground,
        preview_height,
        preview_layers,
        preview_trees,
    )

    bbox = _resolve_bbox(session, cfg, ensure_sheet_grids(session))
    for fn in (preview_height, preview_layers, preview_trees, preview_ground):
        for p in fn(base, cfg.name, bbox, log=click.echo):
            click.echo(p)
    if composite:
        for p in debug_composite(base, cfg.name, bbox, composite_scale, log=click.echo):
            click.echo(p)


@main.command()
@click.option("--aoi", required=True)
def verify(aoi: str) -> None:
    """Coverage + decode round-trip + seam checks on cooked chunks."""
    base, cfg = load_base(), load_aoi(aoi)
    session = PoliteSession(base.fetch)
    from .fetch.maaamet import ensure_sheet_grids
    from .verify import verify_height

    bbox = _resolve_bbox(session, cfg, ensure_sheet_grids(session))
    ok = verify_height(base, bbox, log=click.echo)
    if not ok:
        raise click.ClickException("verification FAILED")
    click.echo("verification passed.")


@main.command()
@click.argument("what", type=click.Choice(["etak", "wfs-metsaregister", "wfs-eelis"]))
def inspect(what: str) -> None:
    """Dump source vocabularies (ETAK layer/class codes, WFS layer names) for mapping tables."""
    base = load_base()
    session = PoliteSession(base.fetch)
    if what == "etak":
        import pyogrio

        gpkgs = sorted((DATA_IN / "etak").glob("**/*.gpkg"))
        if not gpkgs:
            raise click.ClickException("run `assetgen fetch --only etak` first")
        for name in pyogrio.list_layers(gpkgs[0])[:, 0]:
            click.echo(name)
    else:
        from .fetch.wfs import list_layers

        for name in list_layers(session, what.removeprefix("wfs-")):
            click.echo(name)


if __name__ == "__main__":
    main()
