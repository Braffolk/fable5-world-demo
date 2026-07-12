# Estonia streaming world — live-fly review plan & tracker

**Living tracker** for the phase after the S1–S9 streaming milestone, driven by the user's live-fly
review of `?src=estonia` on :5180. Source specs (design, still valid): `SPEC-STREAMING-WORLD.md`,
`SPEC-BARK.md`, `SPEC-ROCKS.md`, `ASSETGEN-ASKS.md` (all in `docs/tasks/2026-07-10/`) and
`docs/world/streaming-integration-plan.md`. This doc tracks the *current* task queue + status so the
plan lives in the repo, not only in session memory.

**Status snapshot (2026-07-12):** HEAD `aaf6007`; data manifest `m/fffd3771349f27c7` (cook_rev 4:
whole-country coarse biome + carved riverbed depth + α coverage). **asset-gen cook source now committed**
(was uncommitted debt): `edfbc21` (#108 biome floor cook), `ec10541` (#104/#114 water carve + coverage) —
asset-gen edits now get normal per-task commit handling. Canonical test URL (hard-reload after a manifest change):
`http://localhost:5180/?scene=world&src=estonia&dataurl=http://localhost:8787`. Generated-world
determinism baseline: scatter `veg.trees 188724 / under 495241 / extras 25131 / stones 453230`
(scatter is INVARIANT — always assert these); `nanite.inst 1611536` (rebaselined: 1320241 → 1323569
at #109 fartile pool parks one inst/slot → 1611536 at #105 each shrub binds a co-located leaf inst);
`nanite.meshes 11681`.

Laws (binding on every slice; also `docs/tasks/2026-07-10/` specs + memory): never fake data
(esp. water depth, landcover); geometric only — no billboards/noise/dither/systemic-knob tricks;
no VRAM hogs (state ceilings); one commit per task (explicit file list, never `asset-gen/` or the
two never-commit docs); the reimplementation is THE default `scene=world` path; audit every agent's
PASS before commit; go up a level on any negative result.

---

## Phase L — live-fly visual fixes (the user's review complaints)

| # | Task | Status | Commit / note |
|---|---|---|---|
| — | Spawn-void: one-direction terrain missing (tagged-union cull misread of mesh word 8) | ✅ DONE | `e2140b4` (user-confirmed) |
| 102 | Far-shell integration: country-floor HEIGHT wired, triangular wedge dissolved | ✅ DONE | `c4bbd20` + manifest `d6014997` |
| 103 | Stones/boulders invisible | ✅ NOT-A-BUG | data-honest sparse + 140 m fly spawn |
| 107 | Default spawn → Taevaskoja river cliffs (game 311123, 190723) | ✅ DONE | `37b975b` (user-confirmed pos) |
| 108 | Far terrain = dirt beyond pilot → whole-country coarse biome material | ✅ DONE | `c1b9538` (runtime) + asset-gen cook_rev 3. Runtime: biome/canopy country-floor (generalized `ensureFloorCoversBox`) + un-gate dead `inp.far`. Asset-gen: whole-country coarse vegDensity from ETAK landcover. Canopy HEIGHT stays honestly pilot-only (no whole-country CHM). |
| 109 | Far voxel-tree fartiles have HOLES | ✅ DONE | `fe6ae0b`. Slot-pool exhaustion (grade never coarsens tileSize → far cells cost 64 slots each; ceiling 8000 < ring's ~10.8k) + slot leak. Fix: ring-arithmetic `ftSlots 11328` + `clusterCap 64→44` (peak 38) + reserve-whole-cell-or-defer. Net −0.76 MB. Steady-state holes gone (user-confirmed); **transient in-motion holes at fly-speed remain** → the deeper `tileSizes[grade]` ladder (surfaced, not scheduled). |
| 105 | Understory SHRUBS render as leafless STEMS → real foliage meshes | ✅ DONE | `cdc3146`. `buildShrub` called `buildTree` without `foliageMode:'mesh'` → crown never built (specs existed, unconsumed; card layer deleted in S8). Fix: shrubs get the same co-located leaf head trees use (scoped to understory, voxel crown trees-only). Cost: nanite.inst +288k (co-located leaf per shrub), +14 MB (0.77%), ~-3% fps at dense-forest worst pose. Debris "sticks" are correct, not this. |
| 113 | Understory FERNS + FLOWERS absent (no pool since S8) → restore | ✅ DONE | `d584227`. Were scattered (in veg.under) but DROPPED (classPolicy null cls 11-14, no head). Fix: FOLIAGE_CLASSES + leaf-primary pool-walk branch (same crown path as tree/shrub, gated by leafOn) + new 3-D bipinnate `buildFern` (ostrich/lady-fern reference) + 4 fern/12 flower pools + Estonia ScatterMap tokens. nanite.inst +207k (dropped understory now binds), +9 MB (shared geo). Streamed Estonia verified by orchestrator (uband 5606→8021). FLOWERS render but muted single-tint (richer bloom = follow-up); fern LOD is single-level (ladder = optimization lever). |
| 110 | Tree SIZING: uniform scale-multiply looks weird → age/height-dependent variants | ✅ DONE | `1b77e1a`. Root cause: `instTransformPoint` A.w uniform multiply → big tree = enlarged small tree. Fix (AgeForm.ts): scale SELECTS an age-form on the existing 4 slots (ZERO count increase); Skeleton ontogeny (trees only) = crown-base lift/self-prune, broaden, stouten. Grounded in real dendrology (research spec in agent report). nanite.inst/meshes UNCHANGED; **nanite.mb −308 MB / −17%** (young forms sparser). Shrubs/gallery byte-identical (`ageForm=false`). |
| 114 | Water/shoreline BORDERS blocky "big squares" → smooth (NEAR) | ✅ DONE | `2e12a41` (+ cook α). `watercover` α plane streamed (Lac1/StreamProtocol/PlaneFill/StreamBrainCore/TerrainField) + WaterMaterial near `wet=cov>0.5` + feather opacity. Generated no-op (compile-time hasWaterCoverage branch). +4 MB Estonia. NEAR shoreline smoothed. |
| 115 | Smooth FAR water edges (waterFar min-reduce) — follow-up to #114 | ✅ DONE | `4523438`. `waterCoverFar` = ×8 MEAN-reduce of the watercover window (new PlaneKind, 128² rgba8 = 64 KB), mirrors `waterFar`←`water`; far levels gate `wet=covFar>0.5` + feather via `fieldWaterCoverageFar`; `waterFar` reduce flips MIN→MAX **keyed on `plan.waterCover`** (MAX dilates ≤16 m, cov-gate renders a strict subset ⇒ every far fragment has a valid surface). Generated BIT-IDENTICAL by construction (no watercover ⇒ waterCoverFar null, reduce stays MIN, far shader branch never compiled) — verified 2 boots. A/B vs `?watercover=0` = old blocky polygon. **Deferred (live-fly watch):** deep-far-field (>1 km) coverage clamp-extends past its ~1 km window; after long flight water(512 m)/watercover(256 m) windows snap on different grids ⇒ far-shore gate offset ≤1 snap — clean fix (far-plane origin commits) changes shipped `waterFar` behavior ⇒ own slice. |
| 104 | Water bathymetry: bed coplanar w/ terrain (z-fight); needs REAL depth | ✅ DONE (depth) | **option C**, cooked cook_rev 4: riverbed carved INTO the height layer (`height − depth`), depth = physics `d_max=k·W^0.8·1.5` from REAL ETAK width/type + REAL LiDAR banks (parabolic section), Ahja 0.36–2.5 m. ZERO runtime code (terrain renders carved height → water surface above bed → z-fight gone, Beer-Lambert falls out). Verified: river renders with depth+reflections, dry land byte-identical, scatter untouched. Dynamic no-flicker = user fly-gate. Sea/big-lake real bathymetry = deferred (external data). | (physics + real banks: ETAK width/type hydraulic-geometry + LiDAR bank wedge for ungauged rivers/lakes; real gridded bathymetry — Maa-amet HIS / EMODnet / BSBD sea, TLU/KAUR lakes — where it exists; NEVER fake). Research done. Pre-cook groundwork: confirm Maa-amet download packaging + datum offset, KAUR lake-raster download path, pin EE width→depth coefficients. Split: asset-gen depth-layer cook + runtime submerged-bed draw. |

**Deferred observation (user, "dont focus rn"):** far material "might still be some oddly long dirt
patches." Likely legitimate agricultural fields (tan/brown = correct) OR ties to #106 coarse landcover
polygons OR a field-class mis-colored as dirt. Revisit when reaching landcover (#106) — sample the far
biome class at a patch to classify (field/fallow = correct vs none/dirt = bug).

---

## Phase R — richness / fidelity (express more of the cooked data)

The renderer currently expresses far less than asset-gen distinguishes. These land after Phase L.

| # | Task | Status | Gap |
|---|---|---|---|
| 111 | Terrain MATERIALS → classId palette (soil deferred) | ✅ DONE (classId) | `ca21e52`. classId (already streamed) → distinct materials: peat/bog dark wet, arable field earthier, barren/sand dry — co-gated by real fields, compile-gated OFF for generated (Biome enum overlaps ETAK ids — landmine caught). Shader-only, 0 VRAM. SOIL taxonomy wiring deferred → #116. |
| 116 | Wire cooked SOIL taxonomy (soilType/stoniness/boniteet) — follow-up to #111 | ✅ DONE | `aaf6007`. Cooked soil (5 u8 planes, pilot LOD0) → one 1024² rgba8 camera-window plane: texCore→mineral tint, stoniness→geometric speckle, boniteet→litter richness, texSkeleton→hue; soilType DROPPED (redundant w/ #111 classId). One compile-gated `if(hasSoil)` block on the composite colour BEFORE classId overrides (classId wins on top, no double-count). Plumbing mirrors watercover — watercover scroll helpers GENERALIZED to `scrollU8`/`refillU8` (afterScroll hook = watercover's far reduce, soil none; no copy-paste twin). Fixed `StreamBrainClient.buildInit` layer-list omitting `'soil'`. Generated BIT-IDENTICAL (`hasSoil=false` ⇒ block never built) — verified 2 boots; water NOT regressed by the scroll refactor (0 page/val errors across a 600 m scroll). VRAM +4 MiB Estonia / 0 generated. **Deferred (user eye):** mineral tint constants art-tunable vs real soil photos (?soil=0 A/B, best over open arable); soilType genesis-hue = +4 MB 2nd plane (decision); soil pilot-only (no country floor). |
| 112 | Tree SPECIES meshes → distinct forms | ✅ DONE (larch+oak) | `44cf5bc`. Added Larch (VegClass 6) + Oak (7) in free tree slots, research-grounded (larch open airy drooping conifer / oak broad stout dome), one buildTree path (inherit #110 age-forms), routed both worlds (Scatter treeK + SpeciesMap LH/TA). Reuse existing bark → 0 new bark VRAM. Right-sized after an over-dense pass (nanite.mb 2070→1839, larch/oak at established per-species rate; ~+20 MB net session). REMAINING (surfaced): aspen/willow/alder/rowan/ash/maple/… still fold to beech/birch — needs an enum renumber (slots 6,7 full). |
| 106 | Landcover + understory follow coarse multi-km polygons → fine granularity | ⏳ QUEUED | asset-gen R&D: species/density SMOOTH fields interpolated from granular data (noise MAY be a component of a real field, never a randomise-shit "solution"). |

---

## Phase I — infra / hardening (after content)

| # | Task | Status |
|---|---|---|
| 88 | S10: `WORLD_SIZE` retirement audit (69 refs/18 files) + teleport path + IndexedDB quota LRU | ⏳ QUEUED |
| 89 | S11: hardening — prefetch, rebase test (staggered shadows), budget audit, cold/warm fly-through p95 proof | ⏳ QUEUED |

## Parallel tracks (independent; slot when a src/asset-gen lane is free)

| # | Task | Status |
|---|---|---|
| 90 | Bark rework B1–B4 (BarkField → displacement → POM → excision) — see `SPEC-BARK.md` | ⏳ QUEUED |
| 95 | Rocks R2–R4 (shading + gallery user-gate → ETAK fixture → excision) — see `SPEC-ROCKS.md` | ⏳ QUEUED |

---

## Latent / surfaced-not-scheduled
- `tileSizes[grade]` ladder for fartiles (dissolves transient in-motion holes + realizes "far cells
  cheaper") — bigger change, risks seam/T-junction holes + determinism re-baseline. Surfaced at #109.
- Far-envelope check (`NaniteCull.ts` ~:1001) uses `instDist=|camPos|` for identity terrain — harmless
  (terrain lodDist=0), fix if terrain ever gets a draw distance. Surfaced at the spawn-void fix.
