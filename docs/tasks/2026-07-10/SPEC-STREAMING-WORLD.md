# SPEC — Generic World Source + Streamed Estonia World (V2, approved 2026-07-10)

Produced by spec→criticise→spec→criticise loop (V1 → engine-reality critic A + architecture
critic B → V2 → round-2 verification critic, findings folded in during execution).
Approved with the plan (~/.claude/plans/misty-crafting-simon.md). Critic finding ids
(A1-A15, F1-F15) are marked where resolved.

## User laws binding this arc (embed in EVERY implementer prompt — agents don't inherit CLAUDE.md)

1. Rocks are MESHES — no rock sub-renderer / runtime SDF raymarch / per-frame deformation.
2. NO subsystem gets dropped for the streamed world — one-shot world bakes become
   camera-following windows, full quality.
3. The reimplementation is THE default scene=world path — generated world runs the SAME
   windowed implementations via its WorldSource; one-shot variants EXCISED as ports land;
   subsystem code never knows which source feeds it.
4. "Engine reality" is never a blocker — "engine can't do X" outputs a redesign of the
   blocking src/ part, never a scope cut (only WebGPU/hardware limits are real).
5. Streaming machinery lives on WORKERS — ring diffs, priorities, splats, bakes; main
   thread = budgeted uploads of ready transferables.
6. Data volume is never the problem — demand logic is. Fetch only what proximity
   immediately requires; far areas only EVER receive coarse data. Simple solutions
   (more pyramid rungs, not special packs).

Plus: no VRAM hogs (ceilings stated, throw-loud); p95 only; no noise/dither tricks; no
systemic knobs; excision law (dead code goes completely incl. comments); go-up-a-level
premise audit on every negative result (CLAUDE.md verbatim block).

## 0. What V2 changed vs V1 (for reviewers)

1. Payloads are decoded f32, not wire-semantic (F1/F7). Lac1 encode/decode lives only in
   RemoteWorldSource + the S1 validator; the validator round-trips *generated* chunks
   through a new Lac1 **encoder** for stronger quantization coverage.
2. `TerrainField` replaces the Heightfield as the scene handle — the full streamed
   terrain field set with a disposition table for **every** hf-coupled consumer (A1/F3).
   No ablations; every port becomes the one default path, one-shot excised same slice.
3. The streaming brain is a worker (law 5): ring diffs, priorities, fetch, decode,
   splat/emit/pyramid, tile DAG bakes, caches, decoded LRU in `StreamBrain.worker`;
   main thread = mailbox draining transferables under ONE token bucket (F13). The
   fartile emit main-thread problem (A3) resolved by a named refactor (FarTilesCore).
4. Boot-to-horizon = more pyramid rungs (user directive): asset-gen adds height LOD4
   (256 m texel, 524 km footprint — one chunk covers Estonia; LOD5 only if measured),
   per-LOD relaxed qscale at coarse rungs. Same format/lods[]/runtime path (A6).
5. Height residency = the plane windows themselves (demand law): a height chunk of LOD k
   is fetched iff it overlaps level k's window (2048·4^k m around the camera). Kills
   V1's L0-window-vs-ring geometry error (A7) structurally.
6. Instance pool redesigned (A4), brick pool vs mirrors resolved (A2), budgets NET
   (F5/F4/A9), re-coarsening/hysteresis/rebase/teleport explicit (F8/A13), stages with
   excision checklists (F2/F6/F9/F11).

## 1. Premise notes (V1 verdicts that stand)

- Octree REJECTED: 2.5D heightfield world; the data is already a ×4 quadtree; residency
  = per-LOD camera-centered ring/window arithmetic over (lod,cx,cz). No tree structure.
- UE5-style geometry page streaming REJECTED: geometry is a closed boot-built library
  (species × variants × LOD rings + crown DAGs, BootCache'd); world content = INSTANCES
  of library entries + terrain tiles + merged-foliage bricks + textures. Library
  mega-buffers stay build-once/all-resident. (UE5 refs: docs/perf-runs/Nanite-UE5-shaders/,
  docs/legacy/deep-review/19-ue5-and-prior-art.md — page streaming solves
  unique-geometry-per-object worlds, which we don't have.)
- StreamOrigin MANDATORY: Estonia coords are 311-700 km from the L-EST97 anchor → f32
  ULP 3.1-6.3 cm → vertex/depth jitter. All pooled GPU positions relative to a
  LOD0-grid-snapped origin near the camera; rebase is a rare bounded event; the
  generated world runs origin (0,0) bit-identically.

## 2. WorldSource abstraction (F1, F7, A5, A11)

Layout (F12): `src/world/source/` = pure data, node-testable, no GPU deps:
`WorldSource.ts`, `GeneratedWorldSource.ts`, `RemoteWorldSource.ts`, `Lac1.ts`
(header/index/records + **encode** and decode), `Lac1Decode.worker.ts`.
ALL residency machinery in `src/nanite/world/`: `StreamBrain.worker.ts`,
`TerrainField.ts`, `ChunkContent.ts`, `PlaceholderLib.ts`, + existing
registry/streamer/pools.

```ts
export interface ChunkKey { lod: number; cx: number; cz: number }

export type ChunkPayload =
  | { kind: 'height'; res: number; heights: Float32Array }   // meters, absolute, incl. apron row/col
  | { kind: 'planes'; res: number; planes: Uint8Array[] }
  | { kind: 'records'; count: number;
      cols: { x: Float32Array; z: Float32Array;              // chunk-local meters
              species: Uint8Array; scale: Float32Array; variant: Uint8Array;
              // OPTIONAL exactness columns — provided when the source knows them:
              y?: Float32Array; yaw?: Float32Array; leanX?: Float32Array; leanZ?: Float32Array } };

export interface WorldSource {
  open(progress?): Promise<WorldManifest>;   // grid, lods[], layers meta, coverage(), dictionaries
  fetch(layer: LayerName, key: ChunkKey, signal?: AbortSignal): Promise<ChunkPayload | null>;
  close(): void;
}
```

- **Decoded f32 everywhere.** Quantization/delta/deflate are LAC1 wire concerns, existing
  only inside RemoteWorldSource (worker-side) and Lac1.ts. Quantize-once happens at
  exactly one place per GPU plane, nowhere else.
- **Exactness columns (A5):** GeneratedWorldSource passes scatter's exact y/yaw/leanX/leanZ
  through — generated instance words byte-identical to today's (makes the S2 gate real).
  RemoteWorldSource omits them; ChunkContent derives-if-absent with ONE shared rule:
  y = height sample at (x,z); yaw = pcg hash(cx,cz,xq,zq); lean = slope-normal·0.18 +
  hash jitter — the same formula as Scatter.ts:467-468, so Estonia trees lean on slopes
  exactly like generated ones. One consumer, one derivation rule, no second path.
- **Lattice honesty (A11):** the generated wire window is defined on the generated
  field's native texel-centered lattice; GeneratedWorldSource.fetch('height') serves f32
  windows of cpuHeights directly. The S1 validator round-trips these through Lac1
  encode→decode with ε = qscale/2. The S4 "identical output" gate applies to the decoded
  f32 path (bit-exact by construction).
- `GeneratedWorldSource.open()` runs Heightfield.generate + runScatter once (BootCache
  untouched), bins results into chunk-keyed views, then **releases the Heightfield GPU
  set post-fill** (F5; see NET budget).
- `coverage(layer, key) → ChunkRef | null` = authoritative absence (dry water, empty
  boulders, outside-Estonia are one mechanism). RemoteWorldSource.open() = latest.json
  (cache:'no-cache'; prefer S3 endpoint in dev — braffolk.com CDN caches longer) →
  manifest → all layer indexes into Map<packed(lod,cx,cz), ChunkRef>.
- Decoded-chunk LRU lives in the brain, 96 MB, nothing pinned (plane windows are the
  persistent store; decoded chunks discardable after plane-fill + overlapping tile bakes).

## 3. TerrainField — the streamed terrain field set (A1, A7, F3, F11; laws 2-3)

`TerrainField` is the new scene handle threaded where `hf` goes today. It owns:

- **Height plane pyramid** — r32float 2048² per level, toroidal, levels = data LODs 0..4
  (5 levels × 16.78 MB = **83.9 MB**). Windows: 2 / 8.2 / 33 / 131 / 524 km. **L4 static
  for Estonia** (covers the country; never scrolls). r32float keeps today's exact idiom —
  heightTex is already r32float/NearestFilter/textureLoad-only (Heightfield.ts:85,120-126)
  — so consumers keep their sampling shape; no TSL uint-mistype bug (documented at
  NaniteGrass ~:143), no r16float 0.25 m ULP steps.
- **Derived normals: none stored.** Terrain normal + slope = 4 height-plane taps
  (central differences) at the pixel's matched level, in-shader — exactly what the
  normal bake does today, moved into the resolve. **Retires normalTex (4096² rgba16f =
  134 MB).** Pre-approved fallback if the S3b p95 gate fails: stored rg8 oct plane per
  level (+21 MB) — measured first (fresh-profiling law).
- **Biome/canopy planes** — rgba8 1024² × 4 levels (class, vegDensity, canopyHeight,
  cover), 16.8 MB, filled from biome+canopy layers (asset-gen asks); at L0 canopy comes
  from resident tree records (F10).
- **Surface-fields planes (round-2 F-1 — MANDATORY)** — rgba8, L0/L1 windows only
  (near-consumers): moisture, riverDepth/flow, snow, rockExposure. Per-frame consumers
  today: fieldsTex in terrain shading (NaniteResolve.ts:547-548 → TerrainMaterial.ts:
  136-137), raster displacement (NaniteFetch.ts:105,110-111), grass density
  (NaniteGrass.ts:308), Froxels.ts:137; biomeTex snow/rockExposure (NaniteFetch.ts:104,
  Particles.ts:87, ProbeGI.ts:169). Generated source fills from its fieldsTex/biomeTex
  binning inside open(); Estonia derives snow/rockExposure IN-SHADER from
  slope+height+class (the BiomeSnow.ts formulas — pure math on data TerrainField has)
  and moisture/river from the water/soil layers. Budget ~+20-40 MB; NET stays deeply
  negative.
- **Water planes (round-2 F-1 — MANDATORY)** — waterY window (r16f, camera-centered,
  ~sim-res texel) + coarse waterYFar level + L0 CPU mirror for walk mode. Consumers:
  WaterMaterial.ts:123, Caustics.ts:196, grass water-gate (NaniteGrass.ts:244,312),
  cpuWaterY walk-mode (Bookmarks.ts:50,121). Generated fills from hf.waterY; Estonia
  from the water layer (LOD0 exists; water LOD1 ask is BLOCKING for far water at S9).
- **CPU mirrors — near levels only** (demand law): height L0+L1 f32 mirrors (33.6 MB RAM)
  serve walk probe, spawn, instance grounding. Coarse levels GPU-only; tile DAG bakes
  read decoded chunks IN THE BRAIN WORKER, never main-thread mirrors.
- **Samplers**: TSL `fieldHeight(wxz, level)` / `fieldHeightFinest(wxz)` (level-select
  ALU hoisted per draw class where the band is known — A15 mitigation: hf tile mesh
  records already carry per-tile origin/level ctx words, GeometryRegistry.ts:29-31);
  CPU `heightAt(wx,wz)`.
- **Scroll↔residency transaction rule (A7)**: demote-before-scroll, promote-after-fill —
  a level-k window region may be overwritten only after tiles reading it are demoted to
  level k+1 (always resident — the backstop invariant), and promoted tiles attach only
  after their texels land. Terrain verts fetch height live at raster (NaniteFetch.ts:416),
  so this ordering is load-bearing; debug assert.

**Consumer disposition table** — every row becomes THE default scene=world path; the
one-shot variant is excised in the named stage; NO ablations (laws 2-3):

| Consumer (anchor) | Today | Streamed-world implementation | Stage |
|---|---|---|---|
| Raster kernels (NaniteFetch.ts:416) | heightTex textureLoad | height plane, per-tile level ctx | S3b |
| Resolve terrain (NaniteResolve.ts:938,1176,1288) | heightTex + normalTex | height plane + in-shader normal/slope | S3b |
| Grass guide bake (NaniteGrass.ts:250-274) | heightTex 4-tap | height plane L0 4-tap | S3b |
| Shadow clip raster (NaniteShadowClip.ts) | heightTex | height plane | S3b |
| Water (TerrainScene.ts:408), Froxels (:337), Particles (:328), Caustics (:186-189) | hf.height buffer via sampleHeight | fieldHeight (local-range effects; direct migration) | S3a |
| Scatter (generated) | hf.height | unchanged — runs INSIDE GeneratedWorldSource.open(), source-internal | S2 |
| groundProbe / findWalkSpawn (TerrainScene.ts:481-503) | hf.heightAtCpu | TerrainField.heightAt (L0/L1 mirrors) | S3a |
| ProbeGI (TerrainScene.ts:171-178; 256²×6 probes over 4096 m) | one-shot world bake | **toroidal probe window**: same 256²×6 grid + 16 m spacing, 4096 m camera-centered window, incremental row/col re-bakes on half-stride crossings, budgeted off-frame; same VRAM | S4 |
| FarShadow (FarShadow.ts:12,89-90) | one world-spanning height-vis map | **~16 km window** @ L2 heights, re-bake rows on stride crossing + sun change (sun static in-session); beyond 16 km shadows are sub-pixel — demand logic, not a drop | S4 |
| Canopy map (buildCanopyMap(scatter.trees) TerrainScene.ts:159; consumers Scatter.ts:361, Froxels.ts:136, wind, GI) | one-shot world texture | **canopy window** from resident tree records (near) + biome-plane canopy channel (far) | S4 |
| Wind hf.noiseA | periodic noise bake | unchanged — position-independent, not height-coupled | — |
| Understory/extras streams (F2) | ScatterResult direct | first-class generated-source record layers through ChunkContent; Estonia derives from guidance planes | S2/S9 |

## 4. StreamBrain — worker-first streaming machinery (law 5; A3, F13)

`StreamBrain.worker.ts` owns: manifest + layer indexes; per-layer ring/window diffs
(camera pose posted ~10 Hz + on teleport); priority queue (coarsest-first at boot,
finest-relevant-first roaming, velocity prefetch at idle); fetch + Lac1 decode (2 decode
sub-workers); **fartile splat + emit + pyramid**; terrain tile DAG bakes (DagWorker pool
folded under the brain); IndexedDB stream caches; the decoded-chunk LRU. It emits
**attach packets** — transferable typed arrays + target descriptors (pool slot / plane
region / instance block).

Main thread does only what cannot leave it: GPU queue writes against the render device,
registry bookkeeping (JS object graph), evict final commit. All uploads drain through
**ONE token bucket** (F13): ≤ 2 MB or ≤ 1.5 ms per frame across plane fills, tile
attaches, instance/brick writes. (Bucket is conservative: writeBuffer reality
~0.2-0.5 ms per 2 MB.)

**FarTilesCore refactor (A3 — the named fix):** extract `FarTilesCore.ts` — dense-emit +
buildVoxelPyramid + pack made three-free (precedent: VoxelBrickCore, 07-04). Library
crown pyramids are already packed words; they transfer to the brain once at boot. The
entire per-chunk fartile job (plan → splat → emit → pyramid → pack) runs worker-side;
the measured 4.4 ms/tile main-thread emit moves off-frame entirely. Main thread receives
packed brick words + head records and writes them under the bucket.

## 5. Residency pools (A2, A4, A8, A12, A14)

**Instance pool (A4):**
- **Pre-bind to capacity** at build(): pool region bound with parked instances (scale 0,
  off-world) so registry.instanceCount — which sizes the frozen cull dispatch
  (NaniteCull.ts:248-250) — equals pool capacity. Parked ≈ free (early-outs :988,997).
- **No per-block origins** (V1's alias-mesh/word7 hole dissolved): A-words store
  positions **relative to StreamOrigin**; blocks keep small CPU mirrors (~2 MB).
  **Rebase** (camera > 8 km from origin, rare) rewrites all resident pooled A-words from
  mirrors — ≤ 2 MB writeBuffer split across ≤ 2 frames — including fartile head identity
  instances (A13). f32 vs origin ≤ 16 km ⇒ sub-mm.
- **Evict = per-instance word rewrites**: park (scale 0) via block mirror, free-list at
  block granularity; NEVER zero shared species meshes' clusterCount (A4-iii).
- **Instance band is its own ring (A12)**: trees-as-instances resident within ~300 m ⇒
  2×2 LOD0 chunks + hysteresis; capacity 4 blocks × 8k + margin = 40k, ceiling at boot.

**Brick pool (A2):** pooled fartile bricks live in the EXISTING brick mega-buffer as a
reserved pool region; writes go **writeBuffer-direct** to the backing GPUBuffer at pool
offsets (verify the r184 backend accessor at S8 — if the accessor differs from
`renderer.backend.get(attr).buffer`, that's a small backend helper; engine-reality law).
Library-region mirrors keep being released (releaseImmutableMirrors untouched); the pool
region has NO CPU mirror — packed words arrive transferable, get written, get dropped.
No new shader binding; per-cluster absolute brickBase (cluster word6) makes in-place
block overwrite addressable. Ceiling: **calibrate against the pilot before freezing**
(A8: est. 105-290 MB for 9 chunks); overflow → coarser ftCell for that chunk, logged.
**Replaces today's ~288 MB all-resident bricks on BOTH sources — net VRAM cut even at
the ceiling.**

**Terrain tile pool:** slot mechanics unchanged; multi-LOD ring plan (strides ×4^k,
signed coords); **hierDepth throw-loud assert at attach** (A14 — today _maxDagDepth
grows silently past the BFS pass count, GeometryRegistry.ts:2020).

**Re-coarsening + hysteresis (F8):** demotion ordering — the covering coarser tile is
always resident (backstop invariant): confirm coarse live → evict fine → then the plane
may scroll (§3 rule). Per-class exit distance = enter distance + ½ chunk. Same shape for
instances→fartiles (fartiles bake ALL trees, so instance evict never uncovers) and
fartiles→canopy-displaced terrain (~3 km+).

**Teleport (A13):** brain clears in-flight, re-seeds coarsest-first at the new pose; far
shell (L4/L3, country-resident) renders immediately; shadow clipmap and GI window do
full staggered refills; StreamOrigin snaps immediately.

## 6. Budgets — NET (F5, F4, A9; both sources; ceilings logged at boot, throw-loud)

| | Allocated | Retired | NET |
|---|---|---|---|
| VRAM | height planes 84 + biome/canopy 17 + instance pool 2 + brick pool ≤160 (calibrated) | heightTex −67 · normalTex −134 · hf.height −67 (+ erosion scratch) · boot fartile bricks −288 | **≈ −290 MB vs today** |
| RAM (main) | L0/L1 mirrors 33.6 + instance mirrors 2 + biome ~4 | cpuHeights −67 (post-S3a) | ≤ 48 MB new |
| RAM (brain worker) | decoded LRU 96 + in-flight ~25 + indexes 2 | — | ≤ 128 MB |

(The bark spec independently retires a further ~535 MB of dead bark maps — context, not
double-counted.) IndexedDB (A10): one `laas-stream` DB (tile DAGs, fartile packs) keyed
content-hash + manifest-hash, lastAccess LRU with 1.5 GB budget,
navigator.storage.estimate() guard at open, stale-manifest sweep. BootCache (library
builds) untouched.

**Boot (A6 + LOD4/5 directive):** LOD4 height (~sub-MB with relaxed qscale) + biome L4 →
whole-country far shell in one small fetch, < 1 s at any sane bandwidth; near budget:
terrain ≤ 3 s, full near content ≤ 10 s cold @ 100 Mbps (F14, gated S11). hooks.ready
fires at far-shell + near-terrain-resident.

## 7. Content build, placeholders (F10, F15)

Derive-if-absent columns (§2); understory/extras are first-class generated-source layers
(F2); EVERY dictionary resolver — species, biome class, soil, understory community,
debris surface, water, far-material — has a black/magenta checker default + one boot
summary line (F15). Placeholder tree = checkered cone-on-cylinder through the normal
TreeBuilder→DAG path at boot; placeholder rock = checkered icosahedron. S6 far-material
fallback (F10): until asset-gen biome/canopy pyramids land, far shading = elevation+slope
tint ramp explicitly styled as a placeholder material. Near-canopy on Estonia derives
from tree records. Water v1: q>0 texels → flat patches at dequantized level, near ring
only; far water = biome tint (full river/lake treatment is its own later arc).
SpeciesMap: manifest ids (21) → VegLibrary pools (6: MA→PINE, KU→SPRUCE, KS/HB→BIRCH,
broadleaf→BEECH, SNAG→SNAG, …); unmapped → placeholder, logged once.

## 8. asset-gen change requests → see ASSETGEN-ASKS.md (same directory)

1. Height LOD4 (+LOD5 if measured; relaxed per-LOD qscale) — BLOCKING far shell.
2. Biome pyramid LODs 1-4 — BLOCKING far material.
3. Canopy layer LODs 1-4 — BLOCKING far forests.
4. Species histogram sidecar; water LOD1 — nice-to-have.

## 9. Stages S1-S11

Every slice: `npx tsc --noEmit` + worst-pose smoke (`?cam=-582.1,302.4,1006.1,2.5692,-0.0077`)
+ SSIM/visual + p95 interleaved A/B; generated world green at EVERY stage THROUGH THE NEW
PATH; excision checklist per slice; one commit per slice.

- **S1** Lac1 codec (decode + encode) + RemoteWorldSource + `tools/probe-lac1.ts`
  (real-pilot decode asserts + generated round-trip ε=qscale/2).
- **S2** WorldSource + GeneratedWorldSource (exactness columns) + ChunkContent placement.
  EXCISE scatter-direct instance wiring. Gate: exact counts + byte-identical instance
  words (generated) + SSIM.
- **S3a** TerrainField construction (static fill on generated; forest/gallery get a
  single-level TerrainField — the fake-hf hack dies, F11) + cold consumers.
- **S3b** hot shaders ONE PER SLICE: raster fetch → grass → shadow clip → resolve
  (+in-shader normals). Per-slice gates: register count (56 cliff), p95, visual parity.
  EXCISE at end: hf.height reads, normalTex, external heightTex reads; generated world
  releases the Heightfield GPU set post-fill.
- **S4** subsystem windows: ProbeGI toroidal · FarShadow window · canopy window — each
  becomes the one default path, one-shot excised same slice.
- **S5** StreamBrain worker + multi-LOD tile streaming + StreamOrigin/rebase + token
  bucket + streaming HUD counters + hierDepth assert. Gate: generated fly-loop
  spike-free (p95 ≤ static+10%, no frame > 2× median).
- **S6 = ESTONIA MILESTONE**: `?src=estonia` + far-view infra (camera far plane,
  fog/aerial ranges, sky horizon) + placeholder far material. Gate: whole-country
  horizon, far shell < 3 s.
- **S7** instance pool + streamed trees/boulders + SpeciesMap + all-resolver placeholders.
- **S8** FarTilesCore (three-free) + brick pool writeBuffer-direct + runtime fartiles
  BOTH sources; EXCISE boot fartile build. Gate: mid-band + first-visit keep-up (25 m/s
  sustained) + VRAM ledger vs §6.
- **S9** understory/debris from planes + water v1 + rocks-ETAK onto real records.
- **S10** WORLD_SIZE retirement audit (69 refs / 18 files) + teleport path + IndexedDB
  quota LRU.
- **S11** hardening: prefetch, rebase test (staggered shadows), budget audit, cold/warm
  fly p95 proof + time-to-detail budget.

## 9a. S1 FORMAT DISCOVERIES (verified against real bytes; binding for all consumers)

- **Water dry texels decode to NaN** in the kind:'height' payload (q=0 = dry; NaN keeps
  dry distinguishable from a real surface at qoffset). Consumers must NaN-check.
- **Height nodata**: cooker encodes NaN source texels as q=0 → decodes to qoffset
  (= floor(min−1); the country LOD4 chunk bottoms at exactly −11 m Baltic). Do NOT read
  those texels as bathymetry.
- **Header qscale does double duty and is f32**: enc1 dequant step AND enc3 x/z record
  dequant step (= footprint/65535). Expectation math must Math.fround(qscale) first.
- **CRC is over the COMPRESSED payload** (zlib.crc32), checked before inflate.
- Boulders' scale decodes to METERS (×0.4 step) vs trees' unitless ×1/64 — two
  semantics, documented at TREE_SCALE_Q/BOULDER_SIZE_STEP_M in Lac1.ts (S7 consumer).
- Height lods are NOW [0..4] (AG1 landed LOD4, qscale 0.25/1.0 at LOD3/4); coarse res
  still 2049. RemoteWorldSource.open() skips manifest layers the codec doesn't know —
  canopy streams automatically once its LAC1 layer id (8) is added to LAC1_LAYER_IDS.
- ⚠️ **Pilot boulder data looks defaulted** (all 3 records in 148_90: kind=1, size=80 →
  32 m!) — raw-bytes confirmed, not a decode bug. asset-gen boulders cook needs an
  audit (queued AG track).

## 9b. ROUND-2 VERIFICATION FIXES (binding; override anything above where they conflict)

- **F-2 brick ceiling**: distance-graded ftCell is the DESIGN, not an overflow fallback —
  brick cell size steps with chunk ring distance (demand law applied to bricks). Today's
  ledger: 8M bricks = 288 MB over 16.8 km²; a 9-chunk residency (37.7 km²) at flat
  density would be ~2× that, so grading is load-bearing. NET VRAM is a calibrated RANGE
  until the pilot measurement; one ceiling number, stated in §6 after calibration.
- **F-3 rebase list additions**: terrain-tile mesh records carry ABSOLUTE origins today
  (GeometryRegistry.ts:16; TerrainStreamer.ts:116-117 writes them) → they become
  StreamOrigin-relative and rebase rewrites them via the live rewriteMeshRecord path
  (GeometryRegistry.ts:1291) — hundreds × 72 B, trivial. Rebase also invalidates ALL
  shadow-clip levels (cached depth rastered under old origin); the staggered refill
  rides the existing per-level budget machinery (NaniteShadowClip.ts:798-811, snap-delta
  caching :920-935). Fartile head count correction: 2048 m chunk = 32×32 = 1024 tiles
  (~9.2k heads across 9 chunks ≈ 0.29 MB rewrite, not "~500 tiles/chunk").
- **F-4 instance overwrite API**: bindInstances is append-only (GeometryRegistry.ts:
  1309-1332) — S7 adds a small registry method rewriting instance A/B + instMesh words
  through the live instArr/instMeshArr mirrors (NOT released, :1495-1496) + pushRange.
  ~15 lines, named: `rewriteInstanceBlock`.
- **F-5 brick-buffer invariant**: after releaseImmutableMirrors, add a throw-guard in
  appendBricks (mirrorsReleased ⇒ throw) and the invariant "no needsUpdate on
  voxelBricksAttr after release" — three's upload path for that buffer is dead by design
  post-boot; pool writes are exclusively writeBuffer-direct. Backend accessor CONFIRMED:
  renderer.backend.get(attr).buffer (precedent GpuProfiler.ts:74-78).
- **F-6 S6 far forests (decided autonomously per user's no-questions directive)**: S6
  includes canopy-displaced far terrain (canopy-height displacement + cover-darkened
  tint from the biome/canopy planes) so the Estonia milestone shows a FORESTED horizon,
  not a treeless country. Depends on AG2's canopy pyramid; the placeholder tint ramp
  remains the fallback if AG2 slips.
- **F-7 L4 anchoring**: AOI = 380.9 × 266.2 km (union of 2357 sheets, chunk-snapped E
  364544→745472, LOD0 cx −2…183). L4's 524 km window covers it ONLY centered near the
  AOI centroid — either center L4 there (it never scrolls) or the LOD4 cook covers the
  snapped union incl. negative cx. State in TerrainField init.
- **F-8 ordering enforcement mechanism**: all GPU mutations execute main-thread; the
  token bucket drains the brain's packet mailbox strictly FIFO (never reorder) — that IS
  the demote→scroll→promote transaction. Slot alloc stays main-thread (registry state)
  with alloc-failure feedback to the brain.
- **F-9 corrections**: f32 ulp at 16 km ≈ 2 mm (not sub-mm — still fine). NaniteFetch
  raster kernels DO read normalTex today (:103, slope in the <85 m disp branch) — the
  4-tap swap adds ~3 taps in a 56-reg kernel; covered by the S3b per-slice register
  gate, not free. Scatter lean anchor = Scatter.ts:469-470; grass guide = NaniteGrass.ts
  :243-312. Boot LOD4 "~sub-MB" is optimistic — the <1 s gate absorbs it. S2 bridge:
  ChunkContent feeds the boot species-stream bindInstances at WorldRegistry.ts:1111-1126
  until S7's pool exists. ForestScene's real generated Heightfield (:493-495) ports to a
  single-level TerrainField; its grass ALSO needs the F-1 fields/water planes.

## 10. Rejected alternatives

Octree · UE5 geometry page streaming · wire-semantic payloads (encoder-validator is
stronger; planes re-quantize once at fill anyway) · per-block origin words / alias mesh
records (cluster word7 collision + kTraverse 10-binding ceiling; mirror-rewrite rebase
is strictly simpler) · stored normal plane (fallback only, measured first) ·
r16uint/r16float planes (TSL uint bug / 0.25 m steps) · special coarse-boot pack
(LOD4/5 rungs supersede) · separate pooled-brick GPU buffer (binding ceiling;
writeBuffer-direct into the existing region) · pinned-LOD3 LRU (plane windows are the
persistent store) · impostor/billboard far forests (geometric-only law) · growable
mega-buffers (realloc stalls; loud fixed pools) · per-chunk resident height textures
(O(rings) VRAM + re-plumb; clipmap is O(1), texel-exact to the data pyramid).

## 11. Decisions locked at plan approval (user defaults)

1. Height planes: r32float, 84 MB ceiling.
2. Rebase shadows: staggered ~4-frame clipmap re-raster.
3. Design speed: sustained 25 m/s + graceful catch-up at 100 m/s scroll-boost.
