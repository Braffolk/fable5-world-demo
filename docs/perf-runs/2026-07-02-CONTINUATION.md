# CONTINUATION HANDOFF — nanite/voxel 60fps mission (write date 2026-07-02 ~03:20, post-compaction entry point)

**READ THIS FIRST after compaction.** You are mid-mission: forest 200k trees @ dpr1.5
(2268×1473), target = locked 60 fps (16.6 ms) moving camera with NO visual quality loss.
You (Fable 5) have been iterating autonomously all night with the user dropping in. This
file + `foliage-is-the-bottleneck-2026-07.md` (auto-memory) + this doc's sibling
`2026-07-02-fable5-review-and-costmap.md` (§1-9) carry everything.

## 0. Where things stand RIGHT NOW

- Branch `nanite-raster`, 19 commits today, HEAD ≈ `b4216c9` (+1 pending lighting commit,
  see §3). Working tree may hold the lit-v4 ambient fix — check `git status`.
- **Perf trajectory (200k, eye/oblique/aerial isolated gpuWall):** day start ~44/59/38 →
  current ~33-36/39-45/15-18 (WARM machine; rested is ~15-25% better). User-confirmed:
  **stutters GONE**, HUD 96 fps aerial / 46-62 oblique-ish, "significantly faster".
- **NOT at target yet.** Best rested full-config: eye 40.2 (pre-fartiles) → fartiles took
  oblique/aerial way down; a RESTED post-everything milestone was never captured (the
  machine has been hot for hours). **First action next session: long-rested milestone —
  `CONFIG=default LABEL=final TREES=200000 TICKS=600 COOLDOWN_S=60 npx tsx
  tools/probe-fresh-stutter.ts` — live p50/p95 vs 16.6 is the verdict metric.**

## 1. THE GOVERNING MEASUREMENT FACTS (do not re-derive)

- **Pixel-scaling law** (dpr sweep 0.75/1.0/1.5): gpuWall ≈ fixed + k·Mpx — eye 13+6.0/Mpx,
  oblique 12+8.1/Mpx, aerial 5+3.9/Mpx → **60-73% of the frame is per-PIXEL cost at retina**.
  User: 120 fps at a ~400×300 window (their observation, confirmed).
- **Live ≈ 0.6-0.7 × isolated** (GPU pipelining) — e.g. noleaves isolated 15.8 → live 8.3.
  Locked-60 live needs isolated ≈ 24-26 at the worst pose.
- **USER DIRECTIVES:** (a) dpr-lowering = "unacceptable levels of blurry" — internal-res
  reduction is DEAD without a real TAA upscaler; (b) **1080p60 + naive upscale is the
  sanctioned LAST RESORT** only when every quality-neutral option is exhausted (per the
  law, ~2.07Mpx ≈ 25 isolated ≈ 16-17 live at eye — nearly locks already); (c) fidelity
  bar is high: "voxels small enough to look like trees" — no giant squares ever.
- **Thermal discipline:** machine drifts 10-15%+ slower over a probe session (chassis
  heat). ONLY same-session ordered ratios are honest; run candidates AFTER baselines
  (bias against the candidate); pose-only runs (TICKS=0, COOLDOWN_S=45-60); long rests
  before milestones. `pmset -g therm` shows nothing useful.
- **ALWAYS screenshot-gate.** The `?vcompact` "3× win" was an EMPTY SCENE (silent pipeline
  death). Probe saves per-pose shots automatically.

## 2. APPARATUS (tools/probe-fresh-stutter.ts — built today, use it)

`CONFIG=default|noleaves EXTRA=k=v,k2=v2 LABEL=name TREES=n DPR=x TICKS=n(0=skip live)
COOLDOWN_S=s FRAMES=n WARMUP=n npx tsx tools/probe-fresh-stutter.ts`
- Boots canonical forest headless (Playwright channel:'chromium'; `globalThis.__name` shim
  for the tsx/esbuild trap), live rAF moving capture (TICKS>0), isolated
  `__laas.measureFrames` at eye/oblique/aerial, per-pose screenshots + JSON to the session
  scratchpad (`fresh-<LABEL>.json`, `shots/<LABEL>-<pose>.png`).
- Boot ~60-105 s (incl. fartiles tile build 36-42 s — optimization backlog!).
- GPU is STRICTLY SERIAL: one probe at a time; NEVER edit src/ while a probe runs (vite
  HMR reloads the page mid-measure; vite now ignores docs/tools/.claude — safe to write
  those anytime). Dev server: localhost:5173, watcher fixed (`ad248bf` — was 50-60% CPU).
- `| tail -N` on the probe buffers output until exit — read the JSON for full data.
- Debug: `?nandbg=albedo|flat|shadow`, `?nanbark=const`, `tools/vcdebug.mjs` (console/
  validation error capture — how the vcompact 11-buffer death was found).

## 3. EVERYTHING SHIPPED TODAY (all default-ON unless noted; flags revert)

Wave 1 `c24ba72`: **?leaflodk** aggregate leaf ladder scale (default now 0.4 via `502662c`);
**errorK** voxel ladder 3→2→**1** (`cace25b`); **?voxnear** mesh→voxel handoff 35→60→**45**
(`502662c`); **?voxbn** per-brick normal+albedo shading (payload bits 21-27, QVOX_CAP=2^21);
impostor-bake skip at forest boot (~7 s).
Wave 2 `6c1fd51`+`597d1cf`: **?voxcell** per-pixel ray→brick raster — slab test (rotated-cube
silhouettes) + ≤6-step DDA into the 4³ occupancy mask (exhaustion=conservative hit), exact
per-pixel depth; per-cluster LINEAR-in-ndc ray/clip bases (~6 FMA/px); front-slab-key
overdraw early-out; gates: non-straddler + area>?voxcellmin(64px²). `?voxtaucap` τ_eff clamp
for voxel clusters — **default 12** (`1a944e0`; per-TILE descent made it cheap; it was
ruinous per-tree at 8: aerial 36→81 ms @4k).
Wave 3 `d636353`: **?fartiles** cross-instance aggregation — 64 m tiles of trees beyond
`?aggdist=140` merged into ONE voxel head each (FarTiles.ts: splat coarse crown bricks +
trunk columns into `?ftcell=0.75` m cell grids → existing buildVoxelPyramid/appendVoxelCrown;
identity instance per tile; per-tree heads end at aggDist; tile nearDist=aggDist−46 overlap).
200k: 812 tiles / ~4M bricks / ~140 MB (**256 MB max-buffer cliff — 0.5 m cells overflow**).
Fixed the instMinPx ~300 m forest-edge deletion (forest now runs to the horizon).
`1a944e0`: tile-boundary bucketing (crown reach) + VOLUME splat (kills holes + the
"wireframe" sampling-beat stripes). `d49ffc3`: **gap-preserving masks** — per-cell coverage
threshold OCC_COVER=0.22 → inter-crown gaps survive coarse levels → voxcell carves far
bricks into tree-shaped clumps ("trees not cubes").
Lighting `58b3ad8`+`b4216c9`+(pending lit-v4): mesh-leaf far cheap path (?resfar·0.6 gate);
**?resfar=60** bark detail gate (moss fbm + normal-map near-only, ~neutral); vox **WRAP
LIGHTING** N·L·0.5+0.5 ×0.9 with floor (0.18→0.25) — fixed whole-tree yaw-normal darkness;
vox BACKLIGHT (translucent forward-scatter, was leaf-only); vox ambient hemisphere up-bias
(ambUp.max(0.6)) — the "still some dark trees" residual (ambUp mapped down-mean-normals to
the 3× darker ground ambient). Albedo exonerated via ?nandbg=albedo.
Also `ad248bf` vite watcher fix (was 50-60% constant CPU = the historical "background load").

## 4. OPEN ISSUES — ranked, with diagnosis state

1. **Rested final milestone missing** (§0) — run FIRST, it's the honest verdict.
2. **Dark trees residual** — lit-v4 (ambient up-bias + floor 0.25) validating at write
   time; if user still sees dark trees, next suspects: per-brick baked NORMAL outliers
   (bake-side clamp: reject strongly-down normals at splat/bake), or species albedo ×
   backlight interplay. Debug via ?nandbg=albedo (colors) vs lit (lighting).
3. **Eye p95 ≫ med spikes** (e.g. 49 vs 23 isolated) — uninvestigated. Suspect: first
   frames after settle (cold HZB → occlusion off → giant frame) leaking into the sample;
   or periodic voxOccPyr rebuild. Check per-frame arrays in the probe JSON.
4. **Tile build 36-42 s of boot** (sync main-thread in ForestScene). Fix: Web Worker or
   IndexedDB cache keyed on (seed, trees, ftcell, aggdist, splat-code-hash) — the existing
   DagWorkerPool/DagCache patterns are precedent (WorldRegistry uses them for terrain).
5. **Far-skyline "trunk antennas"** — 1-cell trunk columns union up the pyramid into thin
   tall bricks poking above the far canopy. Fix: don't splat trunk cells above ~60% of
   crownMinY into coarse levels, or weight trunk coverage lower (they're sub-OCC_COVER
   then), or clip tile grid Y at splat time per-column.
6. **THE REMAINING PERF AXIS (per the pixel law): per-pixel cost.** In priority order:
   a. **Shade-binning resolve** (UE5's decode-once): NEAR leaf/bark pixels still pay
      makeCtx (incl. gust TEXTURE samples) + 3× fetchWorldVert + 3× readVertex PER PIXEL.
      Design: bin pixels by (meshId»class) or (instId,ci) in a small compute pre-pass,
      then shade coherently / cache ctx per tile. Big build; biggest single win left.
   b. **Election single-word experiment (F1)**: current = relaxed load + atomicMax
      (visPayloadV) + atomicStore (visBV) per winning fragment. UE5 = one 64-bit atomic.
      Experiment: pack everything in 32 bits for a variant (depth16 | payload16-via-
      indirection?) or measure a load-elide variant — never isolated.
   c. **vcache re-measure** — BLOCKED on freeing a storage-buffer binding (world1 is at
      the 10-buffer Metal cliff; binding vcompact was the 11th → EMPTY SCENE, diagnosed
      in `6193ca6`). Fold vcompact data into the hwQueue tail (scar counters precedent,
      NaniteRaster ~338-368) — barrier-order fix already landed.
   d. Rect fast path (F2, NaniteRaster ~831): tris ≤4px skip the 3-divide row solve.
7. **leaflodk pose-trade note:** 0.25→0.15 measured eye −3.3 but oblique +6.8 (grown
   leaves leave the cheap-tiny raster regime). Don't push K down blindly; a growMax cap
   in BuildAggregateDag is the smarter knob if the near band needs more cutting.
8. **Backlog small:** naniteleaf is a NO-OP flag in forest (docs trap); registry keeps
   CPU staging arrays forever (~memory floor); meter() readback churn; two-pass occlusion
   (disocclusion holes on motion — correctness, scaffolding exists in NaniteCull).

## 5. KEY CODE MAP (today's files)

- `src/nanite/FarTiles.ts` (NEW): tile splat/pyramid/append. OCC_COVER, EMPTY_BRICK,
  volume splatBox, per-cell cellW, trunk columns, reach bucketing.
- `src/nanite/NaniteVoxelRaster.ts`: voxCell ray path (per-cluster bases ~line 1000+,
  per-brick records in Phase A store ~line 840, per-pixel slab/DDA ~1120-1330), voxBn
  payload, wgSetF, voxCellMinArea=64.
- `src/nanite/NaniteCull.ts`: voxTauCap (default 12) clamp in makeTraverse (~line 824).
- `src/nanite/NaniteResolve.ts`: vox wrap lighting + floor + ambient up-bias (~836-860 +
  ambFloor block ~905), vox backlight (blGate/blSrc ~930), far-leaf cheap path (~670-720),
  ?resfar bark gates (~500-560), voxIdx 21-bit mask + brickSel decode (~380, ~770).
- `src/nanite/VoxelizeCrown.ts`: VOXLOD_CFG errorK=1; buildVoxelPyramid exported (tiles
  reuse it); downsampleBrickGrid (union → re-binned occupancy; gap survival comes from
  the tile-side coverage threshold, NOT here).
- `src/debug/ForestScene.ts`: all forest defaults + fartiles wiring (build pre-reg.build,
  append post-build), voxnear=45, leaflodk=0.4 defaults.
- `src/nanite/NaniteVertexCache.ts`: bitrot documented (11-buffer cliff).
- `vite.config.ts`: watcher ignores (never revert — 50-60% CPU).

## 6. NEXT-SESSION RECIPE

1. Rested milestone (§0). Judge live p50/p95 vs 16.6.
2. If user reports remaining dark trees → §4.2.
3. Then §4.6a (shade-binning resolve) as the big rock; 6b/6c as bounded experiments.
4. Tile-build worker/cache (§4.4) — big UX win, zero risk.
5. Re-eyeball with the user; keep flags for every change; commit per gated win.
