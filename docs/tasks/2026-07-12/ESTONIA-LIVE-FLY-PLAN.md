# Estonia streaming world — live-fly review plan & tracker

**Living tracker** for the phase after the S1–S9 streaming milestone, driven by the user's live-fly
review of `?src=estonia` on :5180. Source specs (design, still valid): `SPEC-STREAMING-WORLD.md`,
`SPEC-BARK.md`, `SPEC-ROCKS.md`, `ASSETGEN-ASKS.md` (all in `docs/tasks/2026-07-10/`) and
`docs/world/streaming-integration-plan.md`. This doc tracks the *current* task queue + status so the
plan lives in the repo, not only in session memory.

**Status snapshot (2026-07-12):** HEAD `fe6ae0b`; data manifest `m/843ed7c85cc1370b` (cook_rev 3,
whole-country coarse biome). Canonical test URL (hard-reload after a manifest change):
`http://localhost:5180/?scene=world&src=estonia&dataurl=http://localhost:8787`. Generated-world
determinism baseline: scatter `veg.trees 188724 / under 495241 / extras 25131 / stones 453230`;
`nanite.inst 1323569` (rebaselined from 1320241 at #109 — the fartile pool parks one instance/slot).

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
| 105 | Understory shrubs/ferns render as leafless STEMS → real foliage meshes | 🔄 IN FLIGHT | give shrubs/ferns geometric leaf/frond crowns (missing-leaves TODO). Debris "sticks" are correct, not this. |
| 110 | Tree SIZING: uniform scale-multiply looks weird → age/height-dependent variants | ⏳ QUEUED (next) | replace raw scale multiply with age-stage variant forms (young slender/low-variance → old tall/high-variance) + bounded jitter; keep variant/instance count ~similar (slight increase, NOT a variant explosion). asset-gen already carries per-tree scale + variant. Visual-correctness bug → ahead of Phase R. **RESEARCH-FIRST (user law):** agent researches REAL reference (pics + forestry growth-form per species/age) and writes a reference spec BEFORE params — never guess silhouettes. |
| 104 | Water bathymetry: bed coplanar w/ terrain (z-fight); needs REAL depth | ⏳ QUEUED | **USER DECISION = option C** (physics + real banks: ETAK width/type hydraulic-geometry + LiDAR bank wedge for ungauged rivers/lakes; real gridded bathymetry — Maa-amet HIS / EMODnet / BSBD sea, TLU/KAUR lakes — where it exists; NEVER fake). Research done. Pre-cook groundwork: confirm Maa-amet download packaging + datum offset, KAUR lake-raster download path, pin EE width→depth coefficients. Split: asset-gen depth-layer cook + runtime submerged-bed draw. |

**Deferred observation (user, "dont focus rn"):** far material "might still be some oddly long dirt
patches." Likely legitimate agricultural fields (tan/brown = correct) OR ties to #106 coarse landcover
polygons OR a field-class mis-colored as dirt. Revisit when reaching landcover (#106) — sample the far
biome class at a patch to classify (field/fallow = correct vs none/dirt = bug).

---

## Phase R — richness / fidelity (express more of the cooked data)

The renderer currently expresses far less than asset-gen distinguishes. These land after Phase L.

| # | Task | Status | Gap |
|---|---|---|---|
| 111 | Terrain MATERIALS → full palette + wire soil | ⏳ QUEUED | asset-gen has ~10 discrete ETAK landcover classes + a soil taxonomy (soilType/stoniness/boniteet); shader blends only ~6 materials from continuous fields, discrete classes collapse to grass/forest/soil, and the SOIL layer is UNWIRED (hardcoded constant). Map classes → distinct materials + wire soil. Near #106. |
| 112 | Tree SPECIES meshes → distinct forms | ⏳ QUEUED | asset-gen cooks the full ~21-species Estonian taxonomy; `SpeciesMap` folds all onto 6 pools (Spruce/Pine/Beech/Birch/KarstGnarl/Snag) by leaf class. id+variant preserved in the stream. Add distinct TreeBuilder forms (larch, oak, aspen, alder, willow, rowan). **RESEARCH-FIRST (user law):** research REAL per-species reference (pics + growth-form/silhouette) before params — never guess. Alongside bark #90. |
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
