# 2026-07-02 (day session) — rested milestone, attribution, bead normals, vox waves

Post-compaction continuation of the 60 fps mission. Entry context: `2026-07-02-CONTINUATION.md`.
User note added this session: far crowns still read BLOCKY — "maybe round normals could help,
unless that kills perf". User also asked to RE-CHECK the backlog logic (late-context plans).

## 1. Rested final milestone (fresh-final-rested.json, machine rested overnight)

- LIVE moving 600 ticks: avg 27.25 p50 25.0 p95 33.4 max 34.3 — **0 frames >100 ms**
  (stutter fix HOLDS), but 584/600 frames >16.7 ⇒ NOT at target. Live sits at 2-4
  vsync periods @120 Hz (25.0 = 3 quanta).
- ISOLATED med/p95: eye 31.2/35.3, oblique 38.8/47.4, aerial 14.6/19.9.
- Gap to locked-60 (needs worst-pose isolated ≈25): oblique −14 ms (−36%), eye −6 ms (−20%).

### Bimodality discovery (per-frame arrays)
- Oblique alternates RUNS of ~4 frames: ~34-39 vs ~43-47.4. Aerial: floor cluster 8.7-11.9
  vs 14-22.8 (2.3× swing!). Eye varies ±3 with no clean split.
- If the periodic cost vanished: aerial med ~10, oblique ~35-36.
- NOT resolve lighting (persists under nandbg=flat). NOT shadow cadence at a static pose
  (shadow-clip levels re-raster only on texel-boundary crossings; static camera ⇒ VP
  bit-identical ⇒ cached). Suspects remaining: TRAA jitter (32-point Halton; jittered VP
  → borderline HZB/cut flicker on BIG far-tile clusters), GI probe updates, voxOccPyr /
  pyramid interactions. UNRESOLVED — check `?traa=0`-equivalent + GI flags next.

## 2. Attribution A/Bs (ordered same-session; candidates later ⇒ thermal bias AGAINST them)

| run | eye | oblique | aerial | verdict |
|---|---|---|---|---|
| default (rested) | 31.2 | 38.8 | 14.6 | baseline |
| `nandbg=flat` (no lighting/GI/backlight) | 29.9 | 39.3 | 17.6 | **lighting ≈ FREE** |
| `nanshadow=0` (NO shadow system at all) | 32.6 | 37.0 | 15.6 | **shadows ≈ FREE (0-2ms)** |
| noleaves (historic, cooler) | 14-17 | 12-14 | 10-13 | foliage = eye +15, obl +25 |

**CONCLUSION (recheck of the backlog's §4.6a):** the "shade-binning resolve" premise was
HALF WRONG — the lighting/shadow/GI per-pixel chain costs ~nothing. The frame =
base (~15) + foliage RASTER COVERAGE + foliage MATERIAL DECODE. Note `nandbg=flat` does
NOT test the decode half (albedo path still runs makeCtx + 3× vertex fetch for near
leaves) — that ceiling is measured by the new `?leafcheap=all` lever (pending).

## 3. Far-band blockiness diagnosis (crop of final-rested-oblique)

Zoomed crop shows the blocky look = **piecewise-constant shading per brick**: large flat
same-green quads (one baked normal + one albedo each) + a handful of distinct greens tiling
the far field like camouflage. Silhouettes are fine (voxcell carving works); it's the FLAT
FILL that reads as blocks. ⇒ the user's "round normals" instinct is correct and cheap.

## 4. Shipped this session (flags; all typecheck-clean)

1. **`?voxbead=k`** (default 0.6, NaniteResolve): per-pixel "round normal" — bend the vox
   shading normal toward normalize(wp − brickCenterWorld) (brick center = words 5-7,
   transformed by `instTransformPoint`; far tiles ride identity instances ⇒ pass-through).
   Ball-like within-brick gradients through the existing wrap+ambient. ~3 loads + ~20 ALU,
   vox pixels only. The classic SpeedTree crown-normal trick at brick scale.
2. **`?voxjit=a`** (default 0.12, NaniteResolve): world-anchored per-cell VALUE jitter on
   vox albedo — hash of floor(wp·1.4) (~0.7 m cells, motion-stable, no payload bits).
   Breaks the few-greens tiling.
3. **`?leafcheap=all`** (attribution lever, NaniteResolve): route EVERY mesh-leaf pixel
   through the ?resfar cheap path (tint × quad normal, NO makeCtx/gust/3-vert interp).
   Measured delta vs default = the exact ceiling of ANY leaf-decode optimization
   (incl. shade-binning) with a real candidate look.
4. **`?voxwaves=N`** (with `?voxf2b=1`, NaniteVoxelRaster.dispatchVoxel): chunk the K=16
   F2B bucket dispatches into N near→far WAVES with a voxOccPyr REBUILD between waves.
   This is the fix for the F2B postmortem's own "per-block-cull pre-seed is never
   realized" — the pyramid is mesh-only when the vox scatter runs (built from
   visPayloadV BEFORE dispatchVoxel), so vox-behind-vox (the DOMINANT oblique occlusion)
   was invisible to the block cull. Wave w+1 now tests against the canopy wave w elected.
   ZERO shader changes — the block cull picks up the refreshed pyramid automatically.
5. **`?voxbocc=1`** (NaniteVoxelRaster Phase A): PER-BRICK occlusion — each brick's own
   clamped bbox + front-slab key vs voxOccPyr (exact conservative idiom of the block
   test: min-pool + |0xff keep-on-tie; straddlers exempt). Kills individually-occluded
   bricks inside partially-visible blocks; pairs with voxwaves for brick-level
   vox-behind-vox.

6. **FarTiles WORKER SPLAT** (boot 36-42 s → target ~6 s): the per-tile splat +
   OCC_COVER post-pass extracted THREE-FREE to `FarTilesSplat.ts` (behavior-exact port,
   same accumulation order ⇒ bit-identical), fanned across ≤8 module Workers
   (`FarTiles.worker.ts`, typed-array transfer both ways); dense-emit + pyramid + prep
   stay on main (VoxelizeCrown drags three imports). `buildFarTilesAsync` in FarTiles.ts
   (deterministic tile order = sync path; sync fallback on worker error); ForestScene
   awaits it. Also fixes the USER-reported slow boot for interactive use.
7. **`?ablate=clouds,ao,taa,bloom,bounce`** (pre-existing, rediscovered): the post
   stack (volumetric cloud march, AO, GI bounce, TRAA) runs inside every gpuWall ever
   measured and was NEVER attributed. The pixel-law "fixed" terms (eye 13/obl 12/aer 5)
   may largely be post. `ablate=taa` also freezes the Halton jitter ⇒ discriminates the
   TRAA-jitter cut-flicker theory of the bimodality (visTris swings 4.0-4.9M live).

## 5. Pending measurements (thermal-ordered, all TICKS=0 COOLDOWN_S=45)

1. `LABEL=bead-v1` default config (bead+jit now default-ON; worker splat in) — gates:
   perf-neutrality, far-band screenshot (within-brick gradients? tiling broken?), boot
   time (~102 s → ~65 s expected), fartiles console log shows worker/emit split +
   IDENTICAL brick/cluster counts vs the last sync boot (bit-exactness check).
2. `EXTRA=ablate=clouds,ao,bounce,bloom,taa` — post-stack total (then singles as needed);
   `ablate=taa` alone for the bimodality/jitter theory.
3. `EXTRA=leafcheap=all` — the leaf-decode ceiling (eye pose is the money number).
4. `EXTRA=voxf2b=1` — F2B barrier-cost control at 200k forest scale (the 2026-06-26
   net-loss was measured at a 4k single-tree pose = pathological occupancy).
5. `EXTRA=voxf2b=1,voxwaves=4` — the vox-behind-vox occlusion gain (oblique is the money
   number). Then voxwaves=2 / voxbocc=1 permutations.

## 6. Notes / hazards

- **NEVER edit src/ while a probe is in flight** — vite full-reloads the connected page
  and the probe dies with "Execution context was destroyed" (killed ablate run #3,
  2026-07-02 04:20). Edits and probes strictly serialized.
- Probe EXTRA can't carry commas; PostStack ?ablate now splits on [,\s+]+ because
  laasUrl %2B-encodes '+' (two wasted runs before the parser fix).
- **bead-v1 → v2 (user: "sharp cones, not blobs")**: v1's per-brick radial field
  (wp−brickCenter) made every brick a bright-tip/dark-flank facet — high-frequency
  contrast, serrated-cone look, AO crevices amplifying. v2 = LOW-frequency crown field:
  horizontal radial from tree origin (A.xyz) + 0.55 up-tilt, 25% residual within-brick
  term. GTAO is exonerated as the CAUSE (it derives normals FROM DEPTH — Gtao.ts, so
  the bent shading normal never feeds AO), it only outlines brick cubes.

- Anthropic classifier outage stalled Bash mid-session; probes were queued behind
  read-only prep. All edits above landed with tsc clean.
- voxwaves adds pyramid-chain dispatches (14 levels, serialized) per extra wave — the
  bubble is the cost to beat; K=16 stays ONE histogram (bucket build is unchanged).
- Aerial/oblique BIMODALITY (§1) is worth its own hunt after the wave experiments — it's
  ~4-9 ms of periodic cost on the two worst poses.
