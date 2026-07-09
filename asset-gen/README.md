# asset-gen — Estonia 1:1 world data pipeline

Fetches open Estonian geodata (Maa-amet elevation + ETAK topography + Metsaregister forest
stands) and cooks it into chunked, quantized, S3-streamable game assets for the meter-precision
Estonia world (see `docs/world/v2.md` for the data catalogue and
`docs/world/streaming-integration-plan.md` for how the renderer consumes this).

Nature-only scope by design: **no buildings, no roads, no orthophotos** — terrain materials
derive from the soil map (Mullastikukaart) + slope/height, not photos.

## Quickstart (fresh clone)

```sh
cd asset-gen
uv run assetgen plan  --aoi taevaskoja   # show fetch plan + chunk coverage
uv run assetgen fetch --aoi taevaskoja   # populate data/in (~5 GB, idempotent/resumable)
uv run assetgen cook  --aoi taevaskoja   # cook data/out chunks + manifest
uv run assetgen verify --aoi taevaskoja  # round-trip + coverage + sanity checks
uv run assetgen preview --aoi taevaskoja # QA PNGs in data/out/preview/
```

Full country = the same commands with `--aoi estonia` (one config flip; ~170 GB of 1 m DTM —
the country-wide coarse floor cooks from the 2 GB 10 m DTM instead, see `cook --layer country-floor`).

- `data/in` — raw downloads, mirrored by source; every file has a `.sha256` sidecar; re-running
  `fetch` skips verified files (that is the reproducibility contract — no bulk data in git).
- `data/work` — intermediates + idempotence sidecars.
- `data/out` — the S3-sync tree (immutable content-hash-named chunks + manifest) + `preview/`.

Coordinate/format contract (anchor, chunk grid, LAC1 container, layer encodings) is defined in
`config/base.toml` + `src/assetgen/grid.py` and frozen into the shipped `manifest.json`.

## Data sources & attribution

| Source | Licence | Attribution |
|---|---|---|
| Kõrgusandmed (DTM/nDSM/CHM), ETAK, Mullastikukaart | CC BY 4.0 equivalent | "Eesti topograafia andmekogu / kõrgusandmed, Maa- ja Ruumiamet" |
| Metsaregister (forest stands) | CC BY 4.0 | "Metsaregister, Keskkonnaagentuur" |
| EELIS (habitats/bogs) | CC BY 4.0 | "EELIS (Eesti Looduse Infosüsteem), Keskkonnaagentuur" |

All sources EPSG:3301 (L-EST97), heights EH2000. Fetchers are rate-limited (≥0.5 s/host) —
please keep them polite.
