# FarTiles (horizon band, 140 m+) deep review (2026-07-02)

Scope: `src/nanite/FarTiles.ts`, `FarTilesSplat.ts`, `FarTiles.worker.ts`, tile entry points in
`VoxelizeCrown.ts` (buildVoxelPyramid / appendVoxelCrown), wiring in `src/debug/ForestScene.ts`,
and how `NaniteCull.ts` / `NaniteVoxelRaster.ts` treat tile clusters at runtime.

## Premise audit

1. **"voxOccPyr is mesh-only ⇒ tiles get no occlusion vs the canopy" is INCOMPLETE.** Two
   occlusion systems act on tiles, at different granularities and freshness:
   - **Cluster-level, prev-frame, FULL-scene:** the traverse's emit-time `sphereOccluded`
     (NaniteCull.ts:919-925) reads the HZB, and the HZB is built AFTER `world1()` which
     *includes* `dispatchVoxel` (NaniteRaster.ts:1418-1424, NaniteFrame.ts:481) — so last
     frame's HZB **contains vox depth** (per-tree crowns AND tiles). Tile clusters fully
     behind yesterday's canopy are already dropped at emit.
   - **Brick-level, same-frame, MESH-only:** voxOccPyr is rebuilt from visPayloadV *before*
     the vox scatter (NaniteVoxelRaster.ts:389-392), i.e. mesh-only; the ?voxbocc per-brick
     test (NaniteVoxelRaster.ts:838+) therefore cannot see vox-behind-vox.
   The real gap is **granularity**: cluster spheres of 64 m-tile blocks are huge and the
   min-pool keeps a block if ANY covered texel is see-through, so the prev-frame cluster cull
   is weak exactly where the per-brick cull is strong — and the per-brick cull is blind to
   the per-tree vox canopy that hides most of the oblique tile field. That asymmetry is
   consistent with voxbocc's pose split (eye −17.2 where occluders are MESH leaves 0-45 m;
   oblique only −6.0 where occluders are per-tree VOX 45-140 m).
2. **"Aerial 659 clusters / ~5.5 ms ⇒ tiles cheap per pixel" — true but pose-flattered.**
   At aerial ([0,150,0]) every tree is ≥150 m ⇒ per-tree heads (maxDist=aggDist=140,
   ForestScene.ts:374-376) all drop ⇒ **aerial foliage is 100 % tiles**. The ~5.4 ms for a
   full 3.34 Mpx canopy (≈1.6 ms/Mpx incl. election+decode) is genuinely cheap, BUT aerial is
   also structurally the easy pose: top-down = single-layer canopy (minimal brick stacking),
   post ≈ 0 there (ablate), and bark is clipped at 140 m (ForestScene.ts:388) so trunk load is
   tiny (44.5 k visTris). Do not extrapolate the aerial per-pixel rate to oblique grazing rays,
   where the same tile field stacks 5-20 bricks deep per ray.
3. **noleaves is a biased base for foliage subtraction.** noleaves disables fartiles
   (ForestScene.ts:312 `!noLeaves`) so bark keeps maxDist 2000 → oblique visTris 2.11 M vs
   default 1.27 M. "Foliage = 37.2−15.4" overstates base / understates foliage by ~1-2 ms.
4. **Metric check — the cut is NOT over-refining tiles.** With τ_eff clamped to 12 px for
   voxel clusters beyond ~42 m (NaniteCull.ts:852-859; lodWarp params NaniteFrame.ts:157-177:
   τ=3, simband=6, lodnear=4, lodpow=0.6 → warp crosses 12 px at ~42 m), emitted tile bricks
   subtend 12-45 px and their 4³ cells 3-11 px. Nothing sub-pixel is being refined-to; the
   coarser direction is quality-trading by ruling.
5. **Fact-pack wording:** OCC_COVER=0.22 is a per-CELL coverage threshold inside the splat
   result pass (FarTilesSplat.ts:205-219), not a separate GPU post-pass. Correctly understood.

## How it works today

**Build (boot).**
- Plan (FarTiles.ts:91-198): world extent from the instance streams; 64 m tiles
  (ForestScene.ts:346), grid per tile = `cellsXZ = ceil(64/0.75/4)*4 = 88`, `cellsY =
  ceil(48/0.75/4)*4 = 64` cells (FarTiles.ts:114-115) → 22×16×22 = 7 744 brick slots.
  Species source = the per-tree crown pyramid level whose brick world ≈ ftcell 0.75 m
  (ForestScene.ts:323-333; grid-180 crowns pick ~L1, 0.53 m bricks). Instances are bucketed
  into EVERY tile their crown can reach (reach = maxR·maxS·1.45, FarTiles.ts:153-189) so
  boundary crowns splat into both tiles (hole-free by duplication).
- Splat (FarTilesSplat.ts:79-224, worker-fanned ≤8× via FarTiles.worker.ts; 6.3 s + 3.6 s
  main-thread emit at 200 k): per instance, volume-splat each source brick with per-cell
  box-overlap fractions (:159-195) accumulating per-brick albedo/normal/weight and per-CELL
  coverage; plus a trunk column of full-coverage cells from ground to `crownMinY·s`
  (:196-202). Then the gap-preserving pass: a cell contributes an occupancy bit only if
  coverage ≥ OCC_COVER=0.22 (:205-219).
- Emit (FarTiles.ts:201-283): bricks with `w ≤ 0.15` or zero occupancy bits are dropped
  (:220); survivors get mean albedo/normal, density=min(1,w/8), then the tile grid runs the
  SAME `buildVoxelPyramid` as a single crown (:246-251).
- Pyramid (VoxelizeCrown.ts:949-1018): ownError(L)=curCell·4·0.5·errorK (=0 at L0, :988);
  K_FLOOR=3 stops coarsening at grid 3×2×3 (:1006-1007) → tiles get **4 levels**:
  L0 3 m bricks (err 0), L1 6 m (err 3), L2 12 m (err 6), L3 24 m (err 12, roots).
- Append (FarTiles.ts:361-381): one voxel head per tile, ONE identity instance at the tile
  center, `nearDist = max(10, aggDist−46)` = 94 m (ForestScene.ts:390), maxDist 100 000.
  Per-tree voxel heads get maxDist=aggDist=140 (ForestScene.ts:374-376) and bark too (:388).
  **Boot totals (console, fresh-voxbocc.json): 812 tiles, 4 033 553 bricks = 138.5 MB (54 %
  of the 256 MB cliff), 57 435 clusters → 4 967 bricks / 70.7 clusters per tile, 70.2
  bricks/cluster.**

**Runtime (per frame).**
- Seed: tile instance passes the nearDist gate on center distance (NaniteCull.ts:786-787)
  and instance frustum (:790); instMinPx never fires (tile spheres are huge).
- Cut: pOwn = projK·ownError/den ≤ τ_eff, projK = cot(27.5°)·1473/2 = **1 414.8**, τ_eff=12
  for vox ⇒ emit bands: **L0 owns 94-354 m, L1 354-707 m, L2 707-1 415 m** (forest corner is
  ~1 267 m, so L3 emits ~never — it exists as root scaffolding). Emitted brick screen sizes:
  45→12 px (L0 across its band), 24→12 px (L1, L2). Emit-time culls: frustum, prev-frame
  full-scene HZB (:919-925). Descent enqueues children with no occlusion test (:935-947) —
  shallow (≤3 hops) for tiles.
- Raster: tile clusters ride the common vox scatter — per-block occlusion vs same-frame
  mesh-only voxOccPyr (NaniteVoxelRaster.ts:588-673), ?voxbocc per-brick vs the same pyramid
  (:838+), Phase A 1 lane/brick records, Phase B footprint spread with front-slab overdraw
  early-out, voxcell DDA carve for footprints > 64 px².

**Measured cluster shares (fresh-voxbocc / fresh-aggdist60 counters):** oblique 9 185 vox
clusters of which tiles ≈ 4.2 k (aggdist60's 4 429 ≈ pure-tile field); eye 5 935 / tiles
≈ 2.0 k; aerial 659 / tiles = ALL.

## Work model

Per pose: `voxCost ≈ Σ_clusters (WG fixed + activeBricks·phaseA) + Σ_bricks(pass cull)
footprintPx·electionAttempt + wonPx·decode`.

- **Aerial (calibrates visible fill):** 659 clusters ≈ 46 k bricks paint 3.34 Mpx single-layer
  → foliage ≈ 5.4 ms ⇒ ~**1.6 ms/Mpx** end-to-end for won pixels. Explains the aerial number
  entirely; per-cluster fixed cost is negligible at this count.
- **Oblique:** visible tile strip = canopy above 140 m = elevations −4.1°..0° of a 55° fov
  ≈ 7.5 % of screen ≈ 0.25 Mpx → **visible tile fill ≈ 0.4-0.8 ms**. The other ~4.2 k tile
  clusters × ~70 bricks ≈ 290 k Phase-A bricks, most of whose footprints are behind the
  45-140 m per-tree vox canopy — invisible to the mesh-only per-brick cull, killed only by
  per-pixel election reads. Subtracting base (~14-15) and post (~5-6) from 37.2 leaves
  ~16-17 ms vox at oblique; the aggdist60 split (−10.7 for the 60-140 ring, pre-voxbocc)
  puts per-tree at ~10-11 of it ⇒ **tiles at oblique ≈ 4-6 ms total, of which ~3-5 ms never
  reaches a visible pixel** (estimate — see probe 1 to pin it).
- **Eye:** tiles ≈ 2 k clusters mostly behind the 0-45 m MESH canopy ⇒ voxbocc already kills
  their bricks (part of the −17.2). Residual tile cost at eye ≈ 1-2 ms.
- What it explains: aerial's cheapness (pose structure + genuinely cheap fill), the voxbocc
  pose asymmetry (occluder class: mesh at eye, vox at oblique), aggdist60's aerial no-op
  (aerial was already all-tiles).

## Waste inventory

1. **Tile bricks behind the per-tree vox canopy at oblique (~3-5 ms, the area's whale).**
   Same-frame per-brick cull is mesh-only (NaniteVoxelRaster.ts:389-392); prev-frame
   cluster-level HZB is too coarse for 64 m-tile blocks (any gap ⇒ keep). ~290 k Phase-A
   bricks at oblique for ~0.25 Mpx of visible strip.
2. **94-140 m double-representation band (~0.5-1.5 ms eye+oblique + a quality hazard).** A
   tile seeded at center-dist ≥94 m renders ALL its clusters, including content at 49-140 m
   that per-tree heads (≤140 m) fully cover. Both reps model the same foliage surface within
   ~1 m ⇒ the 3 m tile brick sometimes WINS the election against the finer ~2 m per-tree
   brick — coarse-rep bleed into exactly the mid-band the user already flags as too coarse.
   ~2-5 tiles/pose in the annulus ≈ 150-350 clusters ≈ 10-25 k big (30-45 px) bricks.
3. **Phase-A lane idling:** 70.2 bricks/cluster avg vs 128 lanes ⇒ ~45 % idle lanes in
   Phase A (Phase B uses all lanes). Minor; partition fill is a build-time knob.
4. **Descent enqueues without occlusion (NaniteCull.ts:935-947):** ≤3 extra hops per tile;
   frontier-only cost, negligible.
5. **VRAM, not ms:** 138.5 MB tile bricks (97 % of the brick buffer) — one trees-doubling or
   ftcell=0.5 crosses the 256 MB cliff. L0 (~3.8 k bricks/tile) is only ever EMITTED for
   tiles within 354 m (~90 of 812 tiles at any pose) but must exist everywhere because the
   camera roams. Streaming/lazy L0 is a boot/VRAM lever, not a frame lever.
6. **Boot:** 6.3 s worker splat + 3.6 s main-thread emit/pyramid at every boot; content is
   deterministic in (seed, trees, ftcell, aggdist, splat-hash) — cacheable.

**Artifact check (shots, scratchpad crops/skyline-*.png from voxbocc-oblique.png):** floaters
DO exist at 140 m+ today — small detached brick specks along the skyline and horizon-slab
edges (≈7 px at 140 m vs the 17 px monsters at aggdist60; low-w fringe cells crossing
OCC_COVER). Thin "antenna" lattices poke above the far canopy — likely coarse BARK mesh of
100-140 m trees (bark still lives ≤140 m; a 40 m+ tree at 140 m crests the far canopy line),
NOT tile trunk columns: the splat's trunk top is capped at ~2 m by the `crownMinY = 2`
initializer (ForestScene.ts:336) — which itself is suspicious (species whose crown base is
higher get a floating-crown gap over a 2 m stub; invisible at grazing angles today). Probe 3
discriminates.

## Levers

Values = expected isolated-gpuWall reduction (ms) eye/oblique/aerial at the canonical poses.

### L1 — fartiles:tile-wave-occlusion (identical, M, the area's main lever)
Split the vox scatter into TWO head-class waves: per-tree vox clusters first, then ONE
voxOccPyr rebuild (now containing mesh + per-tree vox), then tile clusters — whose per-brick
?voxbocc test now sees the 45-140 m canopy that actually hides them at oblique. Mechanism is
the proven conservative idiom (min-pool, keep-on-tie, straddler-exempt) ⇒ quality-identical
by construction. NOT the refuted F2B: no bucket build, no 16-dispatch chain — one extra
pyramid rebuild (~0.3-0.5 ms) + one dispatch split; tile heads are registered last so the
gate is a meshId-range uniform in the scatter guard. The prior "waves ≈ dead" verdict was
measured at BLOCK granularity before voxbocc existed and under F2B's +6 ms chain; per-brick
granularity is what made voxbocc's eye −17.2 possible, so the verdict does not bound this.
Expected: eye 0.5 / oblique 2.0 / aerial 0.7 (aerial: tiles are the only vox, self-occlusion
is top-down-minimal; the rebuild may eat most of it). Risks: rebuild bubble nets ~0 at
eye/aerial; two half-empty dispatches; must keep bocc's straddler exemption for tile bricks
crossing the near plane. Discriminator: probe 4 below; also probe 1 bounds the ceiling first.

### L2 — fartiles:overlap-near-gate (improving, S)
Per-cluster near gate at emit for tile heads: drop a tile cluster when
`dist(cam, s.center) + s.radius < aggDist − reachMargin` (reachMargin ≥ max species reach
≈ 16 m ⇒ every tree contributing to the cluster has instDist < aggDist ⇒ per-tree rep is
live ⇒ provably hole-free). Kills waste #2 AND the coarse-brick bleed-through in the
94-140 m band (strictly a quality improvement: the finer per-tree rep, already rendered,
shows instead; not pixel-identical in an A/B flip, so classed improving, not identical).
Expected: eye 0.5 / oblique 0.6 / aerial 0. Risks: margin must exceed true reach (audit
reachOf against scale outliers); needs a tile-head discriminator at emit (nearDist word is
already distinct: 94 vs 45 — or a mesh flag bit).

### L3 — fartiles:hzb-feedback-damp (identical, M, LOW confidence — investigate first)
The oblique/aerial bimodality (runs of ~4 alternating ±4-9 ms; TAA refuted) now has a
candidate mechanism in this area: the emit-time HZB *includes* vox from the previous frame
(premise-audit #1) ⇒ a cross-frame relaxation oscillator (thick canopy elected → next frame
over-culls tile/vox clusters → thinner canopy → under-culls → …), strongest on the huge
horizon-strip tile blocks that sit exactly at the cull margin. If probe 2 (?occl=0) makes
the bimodality vanish, damp the oscillation (e.g. union of two frames' HZB for the
sphereOccluded test = strictly more conservative ⇒ identical). Value = upper-mode removal:
eye 0 / oblique ~2 (p95/mode, not med) / aerial ~2. Risks: theory may be wrong (GI probe
cycle is the rival suspect); union-HZB costs a small pool pass.

### L4 — fartiles:skyline-floater-cull (improving, S, ~0 ms)
Kill the low-w floaters at build: after the OCC_COVER pass, drop cells with zero occupied
face-neighbours (within-brick + brick-adjacent) whose coverage < 2×OCC_COVER, and/or raise
the per-brick prune (FarTiles.ts:220) from `w ≤ 0.15` to a rank-2-style "≥2 occupied cells
or w ≥ 0.5". Pure quality (skyline specks); perf ≈ 0. Risk: over-thinning wispy crown tops —
gate on a skyline shot A/B.

### L5 — fartiles:boot-cache (identical, M, 0 ms/frame)
IndexedDB-cache the splatted tile grids keyed on (seed, trees, ftcell, aggdist, splat-code
hash) — the DagCache precedent. Saves ~10 s of every boot (user-facing + probe throughput).
Also fix the `crownMinY = 2` cap (ForestScene.ts:336 — initializer acts as a CAP on the
trunk-column top, not a floor) while in there; latent floating-crown gap.

### REJECTED-BY-POLICY (quality-trading — listed for completeness only)
- ftcell 0.75 → 1.0 (bigger cells everywhere beyond 94 m).
- aggdist < 140 (measured −10 ms eye+oblique at 60; look explicitly rejected by user).
- Raising voxTauCap / errorK for tile heads (coarser-than-τ far bricks).
- Dropping tile L0 (min level L1 = 6 m bricks in the 94-354 m band).

## What UE5/prior art does here

UE5's equivalent band is HLOD/Nanite-instanced imposters and (5.4+) Nanite Assemblies: far
aggregates are PRE-BAKED merged proxies — same move as FarTiles — but (a) their runtime cull
is a two-phase occlusion test (prev-frame HZB then re-test after a fresh depth build), which
is exactly the vox-inclusive second wave of L1; (b) their aggregate proxies are built with
interior deletion (only the shell survives, like our shellCoarseBricks — which `voxlodshell`
leaves OFF by default: VoxelizeCrown.ts:1014, worth a probe on tile pyramids specifically);
(c) they stream HLOD tiles by distance rather than keeping all LODs resident — the 138 MB /
256 MB pressure has a standard answer. The prior-art agent should confirm (b)/(c) details.

## Open questions + proposed serial probes (GPU-serialized, run outside this review)

1. **Pin the tile-pipeline cost per pose** (needs an S-effort `?ftskip=1` scatter guard that
   drops clusters with meshId in the tile range — attribution instrument, like noleaves):
   `CONFIG=default EXTRA=ftskip=1 LABEL=ftskip TREES=200000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   Expectation if the model above is right: oblique −4..6, eye −1..2, aerial −5..6 (sky).
2. **Bimodality vs cull-occlusion feedback** (no build needed):
   `CONFIG=default EXTRA=occl=0 LABEL=occl0 TREES=200000 TICKS=0 FRAMES=120 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   Read the per-frame array: runs-of-4 alternation gone (med may rise) ⇒ L3 confirmed.
3. **Antenna + floater attribution** (no build; also the fartiles-off reference):
   `CONFIG=default EXTRA=fartiles=0 LABEL=noft TREES=200000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   Compare skyline crops vs shots/voxbocc-oblique.png: antennas persist ⇒ bark-at-≤140 m,
   not tiles. (Perf numbers are confounded: bark+pertree extend to 2000 m, instMinPx pops
   trees at ~257 m.)
4. **L1 A/B** (after the M build): 
   `CONFIG=default EXTRA=voxtilewave=1 LABEL=tilewave TREES=200000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   Gate: oblique med −1.5 or better at pixel-identical shots.
5. **L2 A/B** (after the S build):
   `CONFIG=default EXTRA=ftneargate=1 LABEL=ftneargate TREES=200000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   Gate: no holes at the 140 m ring in eye/oblique shots; any positive delta ships (it also
   removes the coarse-bleed hazard).
6. **Shell the tile pyramids** (config-only probe; shellCoarseBricks exists but defaults off):
   `CONFIG=default EXTRA=voxlodshell=1 LABEL=ftshell TREES=200000 TICKS=0 COOLDOWN_S=45 npx tsx tools/probe-fresh-stutter.ts`
   Interior L1/L2 tile bricks add depth-only overdraw at the horizon; shelling is
   silhouette-identical by construction (VoxelizeCrown.ts:844-887). Expect small oblique win;
   verify no pyramid-tree pathologies (the empty-vote fallback path, VoxelizeCrown.ts:1057+).
