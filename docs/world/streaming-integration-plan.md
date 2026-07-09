# Streaming Estonia — renderer integration plan

**Status:** design only. This document is written by the asset-gen side for the renderer team.
It describes how the existing WebGPU nanite renderer evolves from its boot-static, procedurally
generated 4096 m world into a chunked, streamed, real-Estonia world fed by the cooked assets in
`asset-gen/` (S3/CloudFront). **No `src/` code is changed by this document.**

The cooked-asset contract it depends on is already implemented and validated on the Taevaskoja
pilot (see `asset-gen/README.md`). This plan tells the renderer how to consume it.

---

## 0. The two things the renderer stores per object (read this first)

There is a persistent misconception risk, so it is stated up front. The assets contain **no
geometry**. Two data kinds only:

- **Field layers** (height, water, land-cover/biome, soil) — small quantized raster tiles, one
  number per ground cell. A heightmap *is* the terrain shape; it cannot be an object list. These
  are what "texels" refers to.
- **Instance layers** (trees now; plants, debris next) — flat lists of tiny records: **position +
  type + size + a 1-byte variation seed.** A tree is 7 bytes. The browser grows the actual mesh
  procedurally from `type + seed`; the asset never carries a vertex.

Everything below preserves that: the renderer keeps generating geometry the way it does today; only
the *placement and field data* switch from procedural synthesis to fetched data.

---

## 1. Premise audit (which "givens" are real)

| Filed as fixed | Verdict | Consequence |
|---|---|---|
| `GeometryRegistry.build()` freezes all mega-buffer caps at boot | **Keep** — caps are *residency* caps, not *content* caps. The streamed resident set is bounded by ring geometry, so boot-sized caps are correct and keep per-frame scratch (`projVertBuf`, MID/HW queues) bounded — the p95 guarantee. Do NOT unfreeze the registry. |
| `instanceCount` baked into the frame graph (`buildNaniteFrame`) | **Keep as a ceiling** — dispatch over the full pool every frame; evicted instances park their sphere at 1e9 and die in `kInstCull`'s first frustum test (the terrain tile pool already proves this is cheap). No frame-graph rebuild ever. |
| Single-origin float32 world coords, `WORLD_SIZE=4096` centered | **Change** — at L-EST97 magnitudes f32 ulp is 3–6 cm. Fix = anchor-relative coords + rare integer rebase (§2). `WORLD_SIZE` is reinterpreted as "resident detail window," not "the world." |
| The 4096² `Heightfield` *is* the world | **Change** — it becomes a sliding window over a national field, plus coarse sibling windows replacing the analytic `FAR_RADIUS` vista. All consumers ride through `Heightfield.uvFromWorld`/`sample*`, so it is one seam to re-point. |

The existing **terrain tile pool** (`GeometryRegistry.reserveTilePool` / `attachHeightDagTile` /
`evictHeightDagTile` + `TerrainStreamer`) is the working template for every streamer below: fixed
slot pool, in-place overwrite, evict-by-parking, single-flight async builds, coarse resident
backstop. Nothing here invents a new streaming model; it clones that one three times.

---

## 2. Coordinates: anchor + integer rebase

- **Manifest is the datum.** `anchor = { e: 368640, n: 6635520 }` (L-EST97, chunk-aligned).
  `gameX = E − anchor.e`, `gameZ = anchor.n − N`, `gameY = EH2000 m`. CPU keeps float64 globals;
  the GPU only ever sees anchor-relative f32.
- **Chunk payloads are chunk-local.** Positions are relative to the chunk corner (trees: u16 within
  the 2048 m chunk). Rebased to anchor-relative at attach (one add during the staging copy).
- **Rebase event:** when `|cam − sessionAnchor| > 24 km`, pick a new anchor on the chunk grid. The
  delta is an exact integer number of meters and every resident coord is < 32 km, so `pos += delta`
  is bit-exact in f32 — no drift, no cracks. One compute pass over instance/cluster/dag/mesh xz
  buffers + CPU mirrors; ~1–3 ms GPU, once per ~20 km travelled, scheduled on a low-motion frame.
- Everything already camera-follows (terrain clipmap, water clipmap, GroundRing grass), so no
  per-frame origin work is added.

---

## 3. Terrain: one resident field → a paged national field

**Generalizes verbatim (keep):** `reserveTilePool`/`allocTileSlot`/`attachHeightDagTile`/
`evictHeightDagTile`, the whole `TerrainStreamer` state machine, `TerrainClipmap.clipmapTiles`
(pure, world-size-independent), `buildTerrainTile` + `DagWorkerPool` + per-tile `DagCache`,
`BuildHeightGrid` skirts/hierarchy. The clipmap stride table already matches the asset LOD table
1:1 — **LOD k texel = lod_step^k m = {1, 4, 16, 64} m** — so a clipmap ring at level k feeds directly
from LOD-k chunks.

**Replace:**
- `Heightfield.generate()` (synthesis→erosion→hydrology→classification) → `Heightfield.fromChunks()`:
  upload decoded height/biome/water/soil rasters instead of synthesizing; keep `rebuildDerivedMaps`
  for normals/slope. The erosion/hydrology GPU passes retire from the runtime path.
- The single `heightTex`/`height` buffer → a **windowed set**: finest window at today's shape
  (4096² r32float, 1 m, ±2048 m around camera) + coarse siblings (4 m ±8 km, 16 m ±32 km). The
  16 m window replaces the analytic `FAR_RADIUS=14000` vista entirely (data now exists that far).
- `TileBuildDeps.heights` (one big Float32Array) → a `HeightPager` that owns decoded CPU height
  tiles (LRU by bytes) and serves both `buildTerrainTile` subsampling and `heightAtCpu`/`waterYAtCpu`
  (camera clamp, spawn). Signature-compatible facade so `TerrainScene`/`FlyCamera` don't change.
- Per-slot mesh `originX/originZ/cellSize` become per-attach (the 18-word mesh record already has
  the fields); `gridVerts` texel coords become window-local via the `windowOrigin` uniform, so
  resident tiles need no rewrite on window scroll (the key p95 decision).

**Soil (`asset-gen` soil layer)** is a new field consumer: 2 m raster of soilType/texCore/
texSkeleton/stoniness/boniteet. It drives **terrain materials** (the user's requirement: materials
from the Mullastikukaart soil map, modulated by slope/height which the renderer already has from
`normalTex`). Add a `soilTex` window alongside `biomeTex`; the terrain material shader samples it.

---

## 4. Vegetation streaming (the missing pool — new, cloned from the tile pool)

`reserveInstancePool(layers)` (pre-build, like `reserveTilePool`): partition each scatter layer's
existing cap into C chunk-slots × perSlotCap. `attachVegChunk(slot, decoded)` writes the instance
records into the slot's fixed range + `instanceMesh[]` (mixed species per slot is legal — instance
contiguity is a bind-time cursor rule, not a cull-time invariant; verify `instFirst`/`instCount`
has no runtime consumer before relying on it) and parks the tail. `evictVegChunk(slot)` parks +
frees the slot. `VegStreamer` = a `TerrainStreamer` clone over a chunk ring.

**Decode of a tree chunk:** read the 7-byte records; expand to the renderer's two-vec4 instance
form (A = xyz+scale, B = yaw/lean/idF). Position: chunk-local u16 → world; **y is sampled from the
resident height window at attach** (asset omits y so trees always sit on the rendered terrain).
yaw/lean/variant: hash `(cx,cz,x,z)` + the stored `variant` byte → the same cosmetic values every
run. species id → archetype: the asset ships the **full ~21-species Estonian taxonomy** in
`manifest.speciesMap`; the renderer maps ids → its procedural archetypes (today ~6; grow toward the
full set — the data is the source of truth, the renderer catches up).

**Archetypes stay boot-built.** Crown geometry, voxelizations, crown-LOD DAGs are per-species
(bounded, not world-bounded) → keep in `BootCache` exactly as today. Optional later: serve the
`PackedVox`/`PackedDagBuilds` blobs from CDN as a cold-boot accelerator (byte-identical to local
builds — the BootCache determinism gate transfers directly).

**FarTiles** become a streamed far ring with a LOD pyramid (64/256/1024 m rungs cooked offline into
`PackedFarTile` blobs, or client-splatted from the streamed tree lists first). `reserveFarTilePool`
+ `attachFarTile` mirror the terrain pool; `PackedFarTile` is already the wire format.

---

## 4b. Scatter-guidance fields (understory, debris) — NOT paint-by-numbers

`understory` (communityId + density) and `debris` (surfaceClass + density) are **coarse scatter
guidance**, like a biome/splat map — never a 1:1 stamp. The cook already: (a) cut density by a
multi-variable ecological suitability model (slope from the heightmap, soil wetness/texture/
stoniness/fertility) so a community only appears where the gradients allow it; and (b) domain-warped
the source-polygon boundaries with globally-continuous noise and noise-modulated the density, so no
cadastral polygon edges survive and the field is seamless across chunk borders.

The client must still: **jitter each instance** (position/rotation/scale from a position hash) and
**blend palettes across the class-field neighborhood** so ecotones are fuzzy, not hard. Treating a
cell's class as a hard mask would reintroduce seams the data worked to remove. Density byte /255 ×
the community/class `base_density` (from `understoryMap`/`debrisMap`) = plants-or-pieces per m². The
nanite pipeline expands this into arbitrarily dense stones/plants/litter — the field is intentionally
tiny (2 planes @ 2 m); the density is in the renderer, not the asset.

## 5. Biome / water / soil rasters

Same windowing as height (every consumer samples through `Heightfield` helpers):
- `biomeTex` (rgba8) ← asset `biome` layer (classId + vegDensity). Map the asset's land-cover
  palette → the renderer biome enum; snow/rockExposure are client-filled (asset omits them).
- `waterY` + `fieldsTex` ← asset `water` layer. **Wet texel = surface elevation; quantized value 0 =
  DRY → renderer substitutes (own bed height − 2.0 m).** This preserves `WaterSurface`'s
  dive-under-terrain trick and `waterYAtCpu` guard unchanged. Absent water chunk = all dry.
- `soilTex` ← asset `soil` layer (new window; §3).

`WaterSurface` (camera-following clipmap) needs zero structural change — only its `sampleWaterY`
window transform.

---

## 6. Network seam

New `ChunkCache` (IndexedDB `laas-chunks`), sibling to `BootCache` (which stays for species
artifacts). Key = content hash from the manifest; value = decoded GPU-ready typed arrays; LRU by
byte budget (~1–2 GB). Flow: `desire → ChunkCache.get(hash) → miss: fetch(CloudFront) → worker
decode (DecompressionStream + dequantize) → put → attach`. A `ChunkWorker` pool (2–3) does
fetch+decode off-thread; the main thread only does budgeted `writeBuffer` staging (≤2 attaches,
≤4–8 MB/frame). Prefetch = concentric rings weighted by velocity direction; coarse layers prefetch
at 2× radius (the always-resident backstop). Offline/failure: coarse-first ordering means a fine
fetch failure still renders the coarse tile — never a hole; full offline renders whatever is cached.

---

## 7. Phasing (honest scope + the p95 story per phase)

| Phase | Delivers | Renderer change | p95 risk |
|---|---|---|---|
| **P0 drop-in** | one 4096 m Estonian region, boot-static | replace `Heightfield.generate` synthesis with `fromChunks`; bind fetched instance lists via existing `bindInstances`; fartiles from fetched placements at boot | ~zero — same buffers/counts, boot-time only. **This is the data-pipeline validation gate — ship first.** |
| **P1 anchor** | anchor-relative coords + multi-region teleport (full reload between regions) | float64 globals, manifest datum, region select | zero (reload is a boot) |
| **P2 terrain paging** | windowed height/biome/water/soil (1/4/16 m), `HeightPager`, national `TerrainStreamer`, vista retired | **highest risk** — window-scroll seams, texel off-by-ones; mitigate with `settleAt` headless probes | scroll strips ≤ budget/frame; +~24 resident slots (tiny cull cost) |
| **P3 vegetation pool** | `reserveInstancePool`/`attachVegChunk`/`VegStreamer`; scatter retired for streamed layers | new pool cloned from tile pool | attaches ~300 KB; parked-instance cull measured before merge |
| **P4 far-field** | far-tile slot pools + ring streamer; offline `PackedFarTile` rungs | new far pool | brick writes budgeted; resident far count held at today's order |
| **P5 national roam** | rebase compute passes, prefetch cone, offline/failure polish | §2 rebase | rebase is the one spike — bounded, rare, schedulable |

---

## 8. Requirements this puts back on the asset pipeline (already satisfied)

1. Chunk grid 2048 m, corner-aligned to the anchor; power-of-two, multiple of the 64 m fartile. ✓
2. Height per chunk per LOD ∈ {1,4,16,64} m, **2049² incl. the shared far edge/apron** so window
   seams never read across a fetch boundary. ✓ (`grid.chunk_raster_window_en`)
3. Coarse LODs (16/64 m) exist for **every** chunk incl. sea (constant tiles) — the no-hole
   backstop. ✓ (`assetgen country-floor`)
4. Water `waterY` with the **q==0 = dry** convention; absent chunk = all dry. ✓
5. Trees: chunk-local SoA, **no y** (client grounds), full species taxonomy in `speciesMap`, 1
   art-seed byte. ✓
6. Determinism + versioning: immutable content-hashed chunks; `latest.json` flips releases
   atomically; any encoding/enum change bumps the manifest hash. ✓ (`asset-gen/S3.md`)
7. Decode: default `deflate` (native `DecompressionStream`), `zstd` config switch. ✓

---

## 9. Nature-only scope (user law)

No buildings, no roads (an optional dormant road-mask plane may exist for a future "cover up roads"
pass), no orthophotos anywhere — materials are derived from soil + slope + land cover, never photos.
Cities may become no-go areas. The renderer should treat absent building/road data as intentional.
