# Bog vegetation arc — DETAILED PLAN (2026-07-19)

> Written pre-compaction for FRESH-EYES review + execution. This is self-contained: a
> post-compact orchestrator with only the conversation summary should be able to (a) sanity-check
> the judgment calls in "§8 REVIEW FIRST", then (b) execute §5–§7 through subagents.

## 0. USER MANDATE (verbatim law — put in EVERY subagent prompt)
"extremely realistic plants are absolutely critical. no fucking 'good enough' bullshit that you
keep doing. we never serve 'good enough'." → Every mesh grounded in REAL reference (photos +
botanical growth-form) FIRST, never guess shapes ([[ground-visual-forms-in-real-reference]]).
Judge each mesh at GROUND LEVEL against real reference before accepting; iterate, don't ship
"good enough". Realism is the acceptance gate, not gate-passing.

## 1. CONTEXT (what this is)
Research preview of a raised-bog (raba) microtopography patch, mire `etak-component-0004071135`,
spliced into the full Estonia base for hierarchy closure. Current state (all committed this
session): real 6 cm relief geometry renders (no holes — LOD0-hole bug fixed 30d2482), water
pools have depth + smooth shoreline, caustics de-blocked, and a PREVIEW CLIP renders ONLY the
cooked core with void beyond (`?previewclip=1` default; `=0` shows full map). See memory
`microtopography-orchestration-arc` for the full commit chain.
- Preview manifest: `asset-gen/data/out/builds/d07a78c58a4cc69e516f8e7bee57efae700c24040f00ca818bfc469f068d808a/m/f38a967ac83e1e4f/manifest.json`.
- Data server: `PORT=8790 LAAS_PREVIEW_MANIFEST=builds/d07a78c58a4cc69e516f8e7bee57efae700c24040f00ca818bfc469f068d808a/m/f38a967ac83e1e4f/manifest.json node tools/serve-data.mjs` (restart if down). Vite: `:5199`.
- Ground URL: `http://localhost:5199/?src=estonia&scene=world&grass=0&x=171672&z=205940&alt=1.6&yaw=2.4&pitch=-0.12&dataurl=http://localhost:8790` (grass=0 for mesh; drop it for veg; `&previewclip=1` clip).
- Boot/verify: `npx tsx tools/boot-smoke.ts --url "<url>" --out <png> --settle 80`. Env note:
  ~half of boots wedge in the veg "clusterizing opaque pools" phase (pre-existing) — retry boots.

## 2. DATA FINDINGS (verified, Fable a064fd1e — cite these)
- Mire land cover: ETAK `E_306_margala_a` `tyyp=20` "Raba", `puis=Ei` ("not predominantly wooded"
  ≠ treeless). Cooked biome.classId over core = 8 "bog" (3678/4096) + 11 "water_still" pools
  (418). Soil = peat (62). Config: `asset-gen/config/landcover-classes.toml:62-67`.
- TREES assigned = **NONE** (0 in core; nearest 548 m). Cause: `asset-gen/src/assetgen/process/
  trees.py:135-136` keeps nDSM peaks only under a forest/shrub landcover mask; bog (8) excluded.
  DATA GAP — real raba has sparse stunted bog-pine + margin birch. Pine/Birch meshes EXIST but as
  normal FOREST forms, not stunted bog forms.
- PLANTS assigned = understory community 5 "raised-bog dwarf" (`asset-gen/config/
  understory-communities.toml` `[communities.bog] id=5`): `sphagnum:6, cottongrass:2,
  labrador_tea:2, bog_cranberry:2, cloudberry:1, heather:1`. Present over 3666/4096 core texels
  (sparse, ~2/255). `[land_cover] 8="bog"` maps class→community.
- IMPL gaps (`src/nanite/world/ScatterMap.ts:34-42`): sphagnum + cottongrass = SKIP (no mesh,
  tint/grass only); the 4 dwarf shrubs all collapse to ONE generic `VegClass.BushPink`.
- Renderer veg builders: `src/vegetation/` (`Understory.ts:67 'bushPink'`, `VegLibrary.ts:472`,
  buildShrub/buildFern/age-forms), voxel crown builders, `src/render/TerrainMaterial.ts:561-571`
  (sphagnum peat tint). Rocks = MESHES on an instanced-scatter path (find it: grep rock/boulder
  scatter). Grass = raymarched lane (wind-driven, the cost).

## 3. ECOLOGY TARGET (real Estonian raba — the realism reference)
- Trees: scattered **stunted bog-form Scots pine** (Pinus sylvestris f. litwinowii/willkommii —
  short, gnarled, flat/candelabra crown, 1–4 m, sometimes multi-stem) across the open expanse;
  **downy birch** (Betula pubescens — scrubby, small, multi-stem) on margins/hummocks.
- Dwarf shrubs on hummocks: heather (Calluna), Labrador tea (Rhododendron tomentosum/Ledum), bog
  rosemary (Andromeda polifolia), crowberry (Empetrum), bog bilberry; creeping cranberry
  (Vaccinium oxycoccos). Herb: cloudberry (Rubus chamaemorus — palmate leaves + amber berry, NOT
  a shrub). Sedge: cotton-grass (Eriophorum vaginatum tussocks + E. angustifolium) with white
  cotton seed-heads. Carpet: Sphagnum mosses (rusty-red → green) + Cladonia lichen on hummocks.
- Placement is MICROTOPOGRAPHY-driven: hummocks (dry) = shrubs/lichen/pine; lawns = sphagnum +
  cotton-grass; hollows/pool margins = wet sphagnum, sundew. Use the cooked relief/wetness fields
  for placement, not uniform scatter.

## 4. DECISIONS
- **Cotton-grass + all dwarf shrubs** → the USUAL BUSH/FLOWER scatter path (same as bushes and
  flowers: buildShrub / flower mesh via `ScatterMap` + `Understory`). Cotton head is part of that
  scatter mesh. (User 07-19.)
- **Sphagnum moss** → PARKED, own research path. It's a CONTINUOUS CARPET (millions of tiny
  elements), so per-cushion instances via the rock path would blow instance counts — the whole
  concern. Deep-research IN FLIGHT (workflow `wf_d7d8a471-3b9`, UE5 5.8 / Nanite Foliage: how to
  render dense low ground cover at SANE instance counts — imposters, aggregate/merged geo, shell/
  parallax height layers, density fade, PCG). Going-in hypothesis (let evidence decide): a
  terrain-surface SHELL/parallax moss layer (zero instances) for the carpet + a SPARSE scatter of
  real cushion/hummock meshes via the rock path for near silhouette, density-faded/imposter'd at
  distance. AFTER research: assess our own paths (rock scatter / voxel / grass shell / terrain
  material) against the technique menu + our instance budget (session baseline nanite.inst ~1.8 M
  — a moss carpet must NOT scale per-cushion), then decide + implement.
- **Trees** → integrate COOK-SIDE (place sparse bog-pine across expanse + downy birch on margins
  at real raba frequency), with BOG-FORM meshes (stunted, not forest form).
- **Dwarf shrubs** → realistic PER-SPECIES geometry, replacing the single BushPink collapse for
  bog tokens.

## 5. MESH SUBAGENTS (Fable, one per mesh; agent cap = 2 → batch in pairs). EACH brief embeds:
§0 mandate verbatim; research REAL reference FIRST; the exact veg-builder patterns to match
(`src/vegetation/` buildShrub/buildFern/age-forms + `VegLibrary.ts` + `Understory.ts` + the
voxel/crown builders + `ScatterMap.ts:34-42` token→VegClass wiring — study how an existing
species is built end-to-end and mirror it); which PATH (bush/flower for cotton-grass+shrubs;
cook-side tree for pine/birch); ground-level QA in real WebGPU (grass=0 + on); tsc clean;
generated world byte-identical; do NOT git commit (orchestrator commits). Meshes:
1. **Stunted bog Scots pine** (bog-form): short gnarled sparse-crown 1–4 m; a bog age-form/species
   variant so cook-side placement can request it (not the tall forest pine).
2. **Downy birch bog/margin form**: scrubby small multi-stem.
3. **Cotton-grass** (Eriophorum): bush/flower path scatter mesh — sedge tuft + white cotton head.
4. **Dwarf-shrub set** (bush/flower path): per-species Calluna / Ledum(Labrador tea) / Andromeda /
   creeping cranberry + cloudberry herb — replace the BushPink collapse for bog tokens; wire each
   token in `ScatterMap.ts`.
5. **Sphagnum** — DO NOT START until §4 research resolves the approach.

## 6. COOK-SIDE PLACEMENT (asset-gen; after meshes exist)
- `trees.py`: allow bog class to receive TREES — sparse stunted pine across the raba expanse +
  downy birch biased to margins (near non-bog / pool edges). Frequency from real raba tree cover
  (research the density; raba is LOW/scattered, not forest) + any nDSM signal. Tag species/age so
  the bog-form meshes are used. Keep the fix scoped (don't regurgitate forest trees onto bog).
- Understory: ensure community 5 renders sphagnum-cover + cotton-grass + DISTINCT shrubs at
  correct density and MICROTOPOGRAPHY-driven placement (hummock vs lawn vs pool-margin via the
  cooked relief/wetness fields). Raise density from the current ~2/255 toward a believable cover.

## 7. 256 m × 256 m TEST MAP (deliverable the user asked for)
A new cooked preview region, 256 m, in the SAME mire, where all these plants naturally appear, so
the user can walk it and inspect trees + shrubs + cotton-grass + (eventually) sphagnum at natural
placement. Expand the current 128 m core to a 256 m bbox in `etak-component-0004071135` (pick a
sub-region containing hummocks + pools + margin so all communities show), reuse the preview packer
(`network_preview.py`) + the preview-clip (void beyond). Cook-side task; verify with ground-level
fly (grass off + on). Report manifest sha + exact-position URL.

## 8. REVIEW FIRST (fresh eyes — sanity-check these BEFORE executing)
1. Is the bush/flower path actually right for cotton-grass (a ~0.5–1 m sedge tuft) AND for the
   dwarf shrubs, or does cotton-grass want the grass lane after all? (User said bush/flower path —
   confirm the path handles both cleanly by reading it.)
2. Sphagnum approach — WAIT for the deep-research (`wf_d7d8a471-3b9`) report; then do the
   our-paths assessment (rock scatter vs voxel vs surface shell vs terrain material) + instance
   budget before deciding. Do NOT default to per-cushion instances.
3. Tree placement frequency — verify a realistic raba tree DENSITY from a source before coding
   trees.py; too many trees = not a raba, too few = the current empty gap.
4. Bog-form pine/birch — the meshes "exist" but as forest forms; confirm whether to add age-form
   variants vs new species entries (check the age-form system used for larch/oak per memory).
5. Batch order: meshes (pine+birch pair, then cotton-grass+shrubs pair) → cook-side trees +
   understory density → 256 m test map → ground QA. Sphagnum slots in after its research.
6. Do the mesh subagents need the data server running for QA? Yes (:8790). Restart if down (§1).

## 9. QUALITY GATE
NO "good enough". Each mesh judged at ground level against real reference photos before accept;
iterate. The 256 m test map is the final inspectable artifact for the user's verdict.
