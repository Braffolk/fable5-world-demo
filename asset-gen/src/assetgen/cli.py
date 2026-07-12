"""assetgen CLI — reproducible fetch + cook of Estonian geodata into streamable chunks.

Quickstart (fresh clone):
    cd asset-gen
    uv run assetgen plan  --aoi taevaskoja      # show what would be fetched
    uv run assetgen fetch --aoi taevaskoja      # populate data/in (idempotent, resumable)
    uv run assetgen cook  --aoi taevaskoja      # cook data/out chunks (later phases)
"""
from __future__ import annotations

import click

from .config import DATA_IN, load_aoi, load_base
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
              type=click.Choice(["height", "biome", "water", "canopy", "soil", "trees",
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
    known = {"height", "biome", "water", "canopy", "soil", "trees", "understory", "debris", "boulders"}
    for name in sorted(wanted - known):
        click.echo(f"(layer {name}: cooker lands in a later phase)")
    click.echo("cook complete.")


@main.command("manifest")
@click.option("--cook-rev", default=1, show_default=True)
def manifest_cmd(cook_rev: int) -> None:
    """Build the content-addressed release tree (data/out) from cooked chunks."""
    from .manifest import build_release

    path = build_release(load_base(), cook_rev, log=click.echo)
    click.echo(f"manifest: {path}")


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
