# Foliage SW-Raster Perf Portfolio — Prior-Art Synthesis

Goal: ~20fps → solid 60fps (p0.05 worst-5% frame ≤ 16.6ms) on the WebGPU foliage
software rasterizer, **zero perceptible quality loss**. Synthesized from 11 deep
source briefs (Scthe/nanite-webgpu, Laine&Karras CudaRaster, CuRast, FreePipe,
LucidRaster, Schütz point-raster, Nanite SIGGRAPH 2021 deep-dive, StarsX/ComputeRaster,
paraLLEl-GS, Tellusim Compute-vs-HW, Granite mesh, Bevy virtual-geometry).

## Cost map being attacked (measured, GPU-bound, worst view, `nanRasterWorld1` ≈ 23ms)
- **60% (~14ms) PER-PIXEL COVERAGE LOOP** — the dominant driver. Per covered fragment:
  3 integer edge tests + barycentric-z + relaxed `atomicLoad` of the election word +
  compare; `atomicMax`+store fires only on a new front (losers early-out → NOT
  atomic-contention bound). Driven by ~12–20× overdraw (~36M visible tris over ~3M px)
  on holey opaque foliage.
- **40% (~9ms) PER-TRIANGLE TRANSFORM + SETUP + LAUNCH** — one workgroup per visible
  cluster (~154k clusters, ~235 tris each), 3× `fetchWorldVert` (~4.7× redundant
  vertex re-fetch, ~48 wind texture taps), edge setup.

## Ground-truth of OUR code (verified this session, `NaniteRaster.ts` / `NaniteVertexCache.ts`)
- Election: `world1` mode = 24b depthKey | 8b tiebreak `atomicMax` into `visPayloadV`;
  the unique winner `atomicStore`s the full 25-bit id into one side buffer `visBV`
  (NaniteRaster.ts:762–777). **No 64-bit atomics anywhere; our split-buffer election is
  ahead of every 32-bit-atomic scheme in the literature.**
- Inner loop (NaniteRaster.ts:702–784): incremental integer edge funcs (`cw += sx`),
  per-pixel `cw≥0 ×3`, depth = `(uw·ndc.z)·rcpArea` (multiply by reciprocal — already
  NOT a per-pixel divide), then relaxed `aLoadU` → compare → conditional elect.
- Scanline x-span: **shipped** (UE5-gap win 1, NaniteRaster.ts:669–701).
- `wgcache` (per-cluster `makeCtx`/wind broadcast through workgroup shared mem): **ON by
  default** — so the per-cluster CONTEXT (incl. trunk-gust texture samples) is already
  deduped to once-per-cluster.
- `vcompact` (cooperative per-UNIQUE-vertex transform into workgroup shared mem):
  **EXISTS but OFF by default** — measured marginal because our redundancy factor R is
  only ~4.7 and the barrier taxes the cheap-terrain majority (NaniteVertexCache.ts:12–23).
  **This materially changes the priority of the "shared-vertex transform" idea that many
  sources converge on — we already built it and it was a wash without cost-aware gating.**
- **No between-samples / sample-miss tiny-triangle cull** exists (verified: bbox is
  computed at NaniteRaster.ts:561–564 but nothing rejects a tri whose bbox contains no
  pixel center before the scanline; only `area2>0` + `validBB` guard).

---

# QUICK WINS (small tweak → medium, do these first)

### Q1. Between-samples / sample-miss tiny-triangle cull (pre-scanline)
- **Driver:** BOTH — removes the triangle's entire per-pixel loop (60%) AND short-circuits
  its setup tail (40%).
- **Mechanism:** after the integer bbox, before edge setup, reject any triangle whose
  snapped bbox lies strictly between sample centers on x OR y (covers no pixel center →
  zero fragments). A handful of integer comparisons on values we already have (`xi*`,
  `yi*` at NaniteRaster.ts:552–564).
- **Provenance:** STRONG CONVERGENCE — CudaRaster "between samples" (Table 3, 3.3×
  downstream tri cut on San Miguel vegetation, our exact regime), CuRast sample-miss cull
  (paper Table 4: Zorah 102→74ms, 140→99ms = −27/−29% of whole frame), Granite micro-poly
  subpixel-bbox reject (headline 31M→1.2M prims), LucidRaster between-samples cull. Four
  independent sources, two with hard numbers on micro-poly workloads.
- **No-64b + WGSL plan:** pure integer test in the per-triangle prologue. For each axis:
  the bbox covers a center iff `floor((bboxMin*256 .. bboxMax*256) crosses a (k*256+128)
  grid line)`. Concretely test whether `ceil((min-128)/256) <= floor((max-128)/256)` per
  axis using the existing `xi0..xi2`/`yi0..yi2` (already in 1/256 units). Reject if either
  axis has no covered center. Independent of the election; no atomics.
- **Magnitude (honest):** CudaRaster/CuRast saw −27 to −40% on dense micro-poly. BUT our
  cluster cut already emits ≤1px tris and we already cull degenerate/snapped slivers
  (`area2>0`), so our true-zero-coverage fraction is **unknown and likely smaller** —
  treat published numbers as an upper bound. Even a 10–20% cut of the 36M-tri input is a
  large swing on both drivers. Must MEASURE.
- **Effort:** small (≤10 WGSL lines in the prologue).
- **Quality risk:** ZERO if the test is strictly conservative against OUR exact top-left
  fill convention — a tri rejected here covers no sample under the current coverage rule,
  so it would never have won a pixel. Risk is an off-by-one that drops a tri that DOES
  touch a center → 1px cracks/holes. **Match the `tlBias` ownership rule exactly.**
- **Validate:** pixel-diff a captured frame cull-on vs cull-off (must be bit-identical);
  count rejected tris via the existing `auditV` counter; then rdbg stage-split timing.

### Q2. Hoist any per-fragment constant into the per-triangle prologue (sample bias / register shape)
- **Driver:** 60% per-pixel loop (per-fragment ALU + register pressure → occupancy).
- **Mechanism:** ensure NOTHING inside the `loopI('sx',…)` body recomputes a per-triangle
  constant, and the half-pixel sample bias is baked into the snapped vertex once (we snap
  with `+128` = +0.5px at the pixel-center `pcx/pcy` setup, NaniteRaster.ts:614–615 — verify
  it is not also re-applied per pixel). Minimize live values across the inner loop to raise
  resident SIMD-group count on Apple GPUs (register-bound occupancy).
- **Provenance:** STRONG CONVERGENCE — CuRast measured a SINGLE hoisted half-pixel offset
  at ~40% on Zorah (74→102ms; pure codegen, same image) and a 55→48 register drop raising
  occupancy; Tellusim bakes the bias into the vertex once; Nanite "inner loop does barely
  anything, a couple ALU + the atomic". Three sources, one with a hard ~40% number.
- **No-64b + WGSL plan:** audit `loopI('sx')` body (NaniteRaster.ts:702–784). The current
  body recomputes `uw0/uw1/uw2 = cw - bias` and does 3 multiplies for `cz` per pixel — see
  Q3 to fold those into an incremental add. WGSL gives no register-count control, but the
  lever (fewer inner-loop live values) is testable on Metal occupancy.
- **Magnitude:** potentially large and free per CuRast's ~40%, but our loop is already
  fairly tight; realistic 5–15% on the loop. MEASURE.
- **Effort:** small (audit + a few hoists).
- **Quality risk:** ZERO (codegen/register shape only; same image).
- **Validate:** rdbg stage-split (rdbg3−rdbg2 = the loop); pixel-diff for identity.

### Q3. Fold barycentric-z into a pure incremental add (kill the per-pixel multiply)
- **Driver:** 60% per-pixel loop.
- **Mechanism:** `cz` is linear in screen (x,y) once the perspective weights are fixed, so
  replace the per-pixel `(uw0·ndc0.z + uw1·ndc1.z + uw2·ndc2.z)·rcpArea` (3 mul + 2 add +
  the `uw=cw-bias` subtractions) with `z += zStepX` in the x-loop and `zRow += zStepY` per
  row, with `zStepX = (sx0·ndc0.z + sx1·ndc1.z + sx2·ndc2.z)·rcpArea` precomputed once per
  triangle. Exactly Nanite's `ZX += GradZ.x` scheme.
- **Provenance:** STRONG CONVERGENCE — Nanite deep-dive (`ZX+=GradZ.x`, depth gradient set
  up once), CudaRaster plane-eq pleqs, CuRast incremental DDA, Tellusim, ComputeRaster,
  Bevy. SIX sources all use incremental depth; we currently recompute per pixel.
- **No-64b + WGSL plan:** add `zRow`/`zX` f32 accumulators alongside `cw0..2`; advance in
  the same add-step as the edge funcs. Must reproduce the exact unbiased-weight depth
  (NaniteRaster.ts:709–728 carefully uses UNBIASED weights summing to `area2` for exact
  cz — the incremental form must match to the bit to keep depth-test parity across passes).
- **Magnitude:** removes ~3 mul + ~3 sub per fragment over 36M fragments. Modest but real
  (a few % of the 60% loop). Stacks with Q2.
- **Effort:** small-medium (must preserve bit-exact depth; the bias/unbiased subtlety is a
  trap — see the N4-C0 water-depth note in code).
- **Quality risk:** ZERO if bit-exact; **borderline** if f32 accumulation drift across a
  long scanline changes a depth bit vs the per-pixel recompute → could desync HZB/shadow
  reconstruction. Validate hard.
- **Validate:** parity-check `cz` bits incremental vs recompute across worst-case spans;
  pixel-diff + depth-buffer diff.

### Q4. Resolve-pass early-discard on clear value (completeness check)
- **Driver:** other (resolve / shading side of the 40%).
- **Mechanism:** in the fullscreen resolve, discard immediately when the election word is
  still the clear value (background) before any unpack/shade work.
- **Provenance:** Bevy ("doubled resolve pass perf"); Nanite material-depth gating.
- **No-64b + WGSL plan:** our clear is 0; `if (election == 0u) { discard; }` at the top of
  the resolve fragment. We already gate `buildTerrainShading` behind `If(isT)` (commit
  8b2a256) — this is a COMPLETENESS check: ensure the ENTIRE resolve fragment (incl. depth
  write) early-outs on clear, not just the terrain sub-branch.
- **Magnitude:** small (resolve is not the dominant pass), but free.
- **Effort:** small.
- **Quality risk:** ZERO.
- **Validate:** confirm background pixels take the discard; frame-time on the resolve pass.

---

# BIG BETS (medium → big refactor, higher ceiling)

### B1. Sort-middle TILED rasterizer with WORKGROUP-MEMORY depth election ★ highest ceiling
- **Driver:** the DOMINANT 60% per-pixel loop, structurally.
- **Mechanism:** bin clusters/triangles to screen tiles (e.g. 8×8 or 16×16 px); each tile
  owned by exactly one workgroup that holds its depth+id buffer in `var<workgroup>` shared
  memory and resolves visibility privately. The per-fragment global relaxed `atomicLoad` +
  compare (one global round-trip per ~36M-fragment, 12–20× overdraw — literally the 60%
  cost) becomes an on-chip shared read-compare. Coverage computed once per (tri,tile).
- **Provenance:** OVERWHELMING CONVERGENCE — this is what EVERY high-perf SW rasterizer in
  the set does and we do NOT: CudaRaster (sort-middle bin→coarse→fine, per-8×8-tile warp,
  shared-mem depth), CuRast (size-binned), LucidRaster (bin→block-row→block→half-block),
  ComputeRaster (bin→tile→pixel), paraLLEl-GS (one subgroup owns a tile, depth in
  registers), FreePipe's MEASURED failure (CudaRaster Table 1: per-triangle scatter is
  53.8×–107× slower than sort-middle on San Miguel vegetation — the strongest single
  prior-art signal that our 60% bottleneck is STRUCTURAL to the scatter model). **See
  cross-cutting insight #1.**
- **No-64b + WGSL plan:** THE no-64b problem disappears — each pixel is owned by one
  workgroup, so depth election is workgroup-scoped: keep our packed (24b depth | 8b
  tiebreak) word but as `var<workgroup> atomic<u32>` resolved by workgroup `atomicMax`
  (WGSL supports 32-bit workgroup atomics); winner writes the id to a workgroup-local side
  array; flush winners to global `visBV`/`visPayloadV` at tile completion. 8×8 tile =
  64px × (u32 depth + u32 id) = 512B shared; 16×16 = 2KB — well within limits. Needs a
  binning compute pass to build per-tile cluster/tri lists (32-bit `atomicAdd` queue +
  `dispatchWorkgroupsIndirect`; ComputeRaster/Bevy two-ended buffer for layout).
- **Magnitude:** the highest ceiling in the portfolio — credibly the path to 2×, since it
  removes the global per-fragment traffic that IS the 60% cost. Honest caveat (CudaRaster
  §6.2): a fixed per-tri cost through binning even for 1px tris — must pair with Q1
  (between-samples cull) + large batches to amortize. At 36M tris a per-32×32-tile tri
  list is overdraw-sized; bin at CLUSTER granularity (clusters span tiles) or use the
  ComputeRaster small-tri bypass (bin tiny tris straight to one fine-tile bucket, skip
  coarse).
- **Effort:** BIG refactor of `nanRasterWorld1` (multi-week, real risk).
- **Quality risk:** ZERO in principle — a reorganization of WHERE depth lives, identical
  visibility. Risk is in watertight edges + the depth tiebreak staying bit-identical at
  tile seams (our integer scanline core already nails watertightness).
- **Validate:** prototype on one view; pixel-diff vs current raster (must be identical);
  profile global-atomic traffic before/after via WebGPU-Inspector capture.

### B2. Per-tile EXACT hierarchical-Z kill (zmin-vs-tile-zmax), live in-pass
- **Driver:** 60% per-pixel loop (removes tris before their fragment loop).
- **Mechanism:** each tri carries a conservative `zmin`; each tile tracks the EXACT `zmax`
  of fragments already painted IN THAT TILE (workgroup reduction, refreshed on overwrite);
  skip a tri wholesale if `zmin ≥ tileZmax` before coverage.
- **Provenance:** CudaRaster (per-8×8-tile exact zmax, 15–37% tris killed), FreePipe-remedy
  framing, LucidRaster front-to-back-vs-live-buffer. **CRITICAL distinction from our
  refuted dead-end:** our refuted HZB was COARSE/cluster-level conservative vs a static
  max-Z pyramid (~0% on holey foliage because a gap pins max-Z to far). This is a
  per-TILE EXACT zmax of fragments ALREADY PAINTED, updated LIVE during the same pass —
  a genuinely different, unmeasured instrument. It pays off precisely where a tile is
  fully painted by near leaves (the dense crown interior), which is where overdraw lives.
- **No-64b + WGSL plan:** ONLY meaningful on top of B1 (needs a per-tile owner so an exact
  live zmax exists). `var<workgroup>` zmax + workgroup reduction; 32-bit, no 64b.
- **Magnitude:** CudaRaster's 15–37% tri-kill is on top of B1's gains; for our 12–20×
  overdraw the dense interior is exactly the case it targets — potentially large, but
  UNMEASURED on holey foliage (the open question: do enough tiles fully paint before the
  back-leaves arrive, given front-to-back is not guaranteed without sorting?).
- **Effort:** medium (rides on B1).
- **Quality risk:** ZERO — `zmin` is a true conservative per-tri lower bound; exactly
  conservative, no popping. NOT a quantile test. (Contrast: a quantile/non-conservative
  occlusion WOULD pop back-leaves and is forbidden.)
- **Validate:** pixel-diff (identical); measure tri-kill rate via audit counter.

### B3. Front-to-back tile ordering against the LIVE election buffer (opaque early-out)
- **Driver:** 60% per-pixel loop (overdraw reduction).
- **Mechanism:** within a tile, order clusters by representative nearest depth and raster
  near→far; a covered opaque pixel that already has a nearer election winner skips. For
  opaque, first-opaque-wins is exactly equivalent to a depth buffer — a leaf GAP simply
  emits no terminating fragment, so back-leaves through gaps still appear (NOT the
  forbidden quantile popping).
- **Provenance:** CuRast translucent pipeline (tile-bin + coarse-depth sort + bounded
  front-to-back with first-opaque early-out — proven equivalent to a depth buffer for
  opaque), LucidRaster front-to-back-vs-live-buffer. Two sources.
- **No-64b + WGSL plan:** rides on B1's per-tile owner; the early-out tests the
  `var<workgroup>` election word (relaxed read), 32-bit only. Needs a per-tile cluster
  sort by nearest depth — a coarse depth-bucket count-sort suffices (approximate
  front-to-back is enough; correctness comes from the exact per-pixel election).
- **Magnitude:** cuts the ~12–20× overdraw toward ~1× in the limit — potentially the
  single largest fragment-count reduction. But bookkeeping (sort + per-tile ordering) may
  exceed savings on ~1px tris; MEASURE-FIRST.
- **Effort:** big (rides on B1 + a per-tile sort).
- **Quality risk:** ZERO if tested against ACTUAL per-pixel election depths (never a
  quantile). Flag: the temptation to approximate "behind" with a per-tile representative
  depth would pop back-leaves — forbidden. The reject MUST be per-pixel.
- **Validate:** pixel-diff vs current (identical); overdraw-factor measurement.

### B4. Leaf-crown / billboard impostors for the far field (visibility-buffer injection)
- **Driver:** BOTH 60% and 40% for the far field (where the bulk of the ~12–20× overdraw
  lives — distant crowns).
- **Mechanism:** replace whole distant foliage instances with a camera-facing textured
  quad (pre-baked octahedral views + baked albedo/normal/depth), injected DIRECTLY into
  the visbuffer, bypassing the cluster raster. Collapses thousands of 1px overdrawn tris
  to a few textured fragments.
- **Provenance:** Scthe/nanite-webgpu (billboard impostors — "the ONLY thing that
  materially cut dense-overdraw cost"), Nanite deep-dive (8:8 depth:triID octahedral
  impostor atlas injected into the visbuffer). The ONLY workload-SHAPING lever multiple
  sources found that beats dense overdraw rather than shaving per-fragment cost.
- **No-64b + WGSL plan:** the impostor injects a packed depth:id word — adapt to write OUR
  32-bit election word (depth high bits + id), slotting into the existing `atomicMax`
  election with no 64b dependency. Needs an atlas bake pass + a per-instance distance/area
  test to swap mesh→impostor + an inject compute pass.
- **Magnitude:** large for the far field if a big fraction of instances are distant
  crowns; the far field is where overdraw concentrates.
- **Effort:** big (atlas bake + inject pass + swap logic).
- **Quality risk:** BORDERLINE — impostor pop/parallax. Karis notes noticeable pop on
  repeated neighbors (dither+TAA hides most); Scthe's impostors don't handle up/down
  views. For canopy foliage viewed from a band of angles it's plausible, but this is the
  one big bet that is NOT inherently zero-loss — it MUST be validated against the
  zero-perceptible-loss bar (view count + cross-fade + distance threshold). Quality-budgeted.
- **Validate:** A/B perceptual review at the distance band; measure pop under camera
  motion with TAA; tune view-count until imperceptible, THEN measure perf.

### B5. Persistent-thread work-queue raster (replace ~154k one-workgroup-per-cluster dispatches)
- **Driver:** 40% transform+launch (launch model + long-tail load balance).
- **Mechanism:** dispatch a fixed modest pool of workgroups (~few hundred, tuned to
  saturate the M-series GPU); each loops `i = atomicAdd(&cursor, BATCH)` over the global
  cluster list until drained, instead of one workgroup per cluster.
- **Provenance:** STRONG CONVERGENCE — CudaRaster (16 persistent CTAs, big-batch atomic
  intake), CuRast (occupancy-bound grid + atomicAdd batches), LucidRaster (`loadNextBin`),
  ComputeRaster (GPU-sized indirect dispatch). Four sources.
- **No-64b + WGSL plan:** WebGPU has no `grid.sync`, but the persistent pattern works
  inside one dispatch with a storage-buffer 32-bit `atomicAdd` cursor and NO
  inter-workgroup wait (no deadlock risk). Counter init in a tiny separate pass. Replaces
  the indirect per-cluster dispatch.
- **Magnitude:** HONEST — our clusters are uniform (235/256 filled) so load-balance upside
  is modest; the upside is whether ~154k workgroup LAUNCHES actually cost (Apple driver
  dispatch overhead). The brief already flags submit/BFS-pass batching as MEASURED
  marginal, but this is a DIFFERENT lever (the launch MODEL, not submit batching) and is
  untested. MEASURE a ~512-wg persistent pool vs the current dispatch before investing.
- **Effort:** medium.
- **Quality risk:** ZERO (pure scheduling, identical fragments).
- **Validate:** swap the dispatch, pixel-diff (identical), compare frame time + the
  launch-floor rdbg=4 timing.

### B6. Cost-aware-gated cooperative vertex transform (revive `vcompact` with gating)
- **Driver:** 40% transform+setup (the ~4.7× redundant `fetchWorldVert` + ~48 wind taps).
- **Mechanism:** transform each cluster's UNIQUE vertex (incl. wind animation) ONCE into
  `var<workgroup>` shared memory (phase A: 1 thread/vertex), barrier, then per-triangle
  threads read 3 corners from shared (phase B). **We already built this (`vcompact`,
  NaniteVertexCache.ts) and it was MEASURED MARGINAL** because R≈4.7 barely clears the
  barrier cost and the cheap-terrain majority is taxed. The revival is the COST-AWARE GATE
  the code's own note calls for (NaniteVertexCache.ts:20–23): the cull binning step
  classifies clusters by per-vertex cost and pays the barrier ONLY on expensive
  (high-wind-tap, animated) foliage clusters.
- **Provenance:** OVERWHELMING CONVERGENCE — Nanite deep-dive (the ONE place Nanite does
  less work than us; phase-1 per-vertex transform to groupshared), CuRast (names this as
  the Karis gap it lacks), Schütz, Tellusim, ComputeRaster vertex-once, Bevy
  `viewport_vertices[256]`, Granite split vertex/attribute shading. SEVEN sources. **But
  see cross-cutting insight #2: we already tested the ungated version and it was a wash —
  so the transferable part is the GATE, not the technique.**
- **No-64b + WGSL plan:** already implemented; add a per-cluster cost class from the build
  (wind-tap count / animation flag) into the cull output, and gate the `prime()` barrier
  path on it. Orthogonal to the election.
- **Magnitude:** bounded by R≈4.7 on the expensive clusters only — modest, and only on the
  foliage subset. Honest: this is NOT the 40% silver bullet the briefs imply, because
  `wgcache` already deduped the per-cluster CONTEXT (the expensive trunk-gust samples) to
  once-per-cluster; the residual is the per-VERTEX wind displacement, which is cheaper.
- **Effort:** medium (the gate + build-side cost class).
- **Quality risk:** ZERO (bit-identical transform, verified A/B in code).
- **Validate:** rdbg stage-split on a wind-heavy forest view with the gate on; confirm
  neutral-or-better on the terrain-heavy vista (the regression case).

---

# CROSS-CUTTING INSIGHTS

1. **Every high-performance SW rasterizer in the set is sort-middle TILED (bin→coarse→fine
   with on-chip per-tile depth); we are the only one using one-workgroup-per-cluster
   scatter.** CudaRaster's Table 1 MEASURES our architecture (FreePipe) at 53.8×–107×
   slower than sort-middle on San Miguel vegetation (≈ our holey foliage) vs a TIE on
   large-triangle Buddha. This is the strongest signal in the prior art that our 60%
   bottleneck is STRUCTURAL to the scatter model, not a micro-op to shave. B1 is the
   architectural answer; B2/B3 ride on it. The two WebGPU peers (Scthe, Bevy) did NOT find
   a magic inner-loop trick under our exact constraint — they shaped the workload down
   (impostors) instead. The tiling lever comes from the CUDA/Vulkan lineage, not the
   WebGPU peers.

2. **The "shared-memory per-cluster vertex transform" that SEVEN sources converge on, we
   already built and measured marginal.** `vcompact` exists and is OFF by default because
   R≈4.7 doesn't clear the barrier and `wgcache` already deduped the expensive per-cluster
   context. Convergence across sources is normally strong signal — here it would have
   MISLED us into re-implementing a wash. The real residual lever is the COST-AWARE GATE
   (B6), not the technique. This is the portfolio's clearest "convergence ≠ win for us"
   finding.

3. **The packed-32b-atomic election is the historically validated, correct no-64b answer
   and ours is AHEAD of the literature.** FreePipe (origin), Tellusim (wished-for
   `imageAtomicPayloadMax` — we emulate it), Scthe (16b depth = "tons of artifacts" — we
   use 24b + side-store), Bevy (needs r64uint, panics without it). Our 24b-depth | 8b-
   tiebreak `atomicMax` + winner-plain-store-25b-id is strictly better than every 32-bit
   scheme. Do NOT spend effort "fixing" the election. KEEP the 8-bit tiebreak (FreePipe
   documents the non-deterministic equal-depth race it prevents — real correctness work).
   Do NOT adopt FreePipe's dual-32b-atomic split (it franksteins the id payload).

4. **Coverage-traversal micro-optimization is low-EV at ~1px (stop digging there).** CuRast
   TRIED Pineda/in-triangle traversal and measured a LOSS on pixel-sized tris ("avoiding
   wasted work is more expensive than the cheap wasted work"). This confirms our shipped
   scanline x-span is at the favorable cheap end and warns against pushing it toward exact
   per-fragment edge-clipping (our reverted depth-DDA aligns). The profitable direction is
   moving the cull EARLIER+CHEAPER (Q1), which they DID find profitable — not deeper
   per-fragment traversal.

5. **The cheapest fragment is one never rasterized.** The two genuinely workload-shaping
   levers (Q1 between-samples cull; B4 far-field impostors) and the overdraw-removal levers
   (B2 live tile-zmax; B3 front-to-back) attack fragment COUNT, not per-fragment cost. Given
   the 60% loop is overdraw-VOLUME bound (Nanite's own slides confirm leaves/grass are its
   worst case and the HZB is "almost useless" on holey surfaces), count-reduction is the
   high-leverage axis. Inner-loop ALU shaving (Q2/Q3) is real but bounded.

---

# CONFIRMED WALLS (prior art CONFIRMS these are dead — do NOT resurface)

1. **Conservative occlusion / static max-Z HZB ≈ 0% on holey foliage.** Independently
   confirmed by Scthe ("Swiss cheese theory", README:111), Nanite deep-dive (HZB max-op
   "almost useless" on aggregates), Schütz, Bevy (min-reduction pyramid, same Swiss-cheese
   failure), Granite (two-phase HiZ), CuRast/CudaRaster (do none, still win). A gap pins
   max-Z to the far plane. NOTE: B2's per-tile EXACT LIVE zmax is a DIFFERENT instrument —
   not this wall.

2. **Per-triangle occlusion culling is not worth it.** Karis: re-evaluation + re-raster
   would nullify savings, plus thread-per-tri divergence. Confirmed.

3. **SW atomic contention is NOT the bottleneck.** CuRast (billions of tris, no contention
   problem), CudaRaster (removes global atomics by tiling — the cost is the global
   round-trip+compare, not contention), Nanite (atomic only bottlenecks LARGE tris, not
   ~1px), Schütz, Tellusim, ComputeRaster (the contention-eliminating mutex path is the
   SLOW one). Our relaxed-load-then-elect is already ahead. Subgroup atomic-collapse
   (Schütz T2) collapses an op we mostly don't execute — skip.

4. **Hardware early-Z / HW raster of sub-pixel tris wastes ~4× on mandatory 2×2 quads.**
   CudaRaster §3.4/§6.1, Nanite (the reason SW raster exists), Bevy (keeps sub-pixel in
   SW), Tellusim (compute beats HW 1.68× on M1). The SW choice is correct. (Routing only
   LARGE-projecting meshlets to HW for early-Z — Scthe T4 — applies only if our foliage
   ever projects large near camera; verify before investing, likely a non-lever for ≤1px.)

5. **Sub-pixel LOD over-render is already past the literature.** Our cluster cut emits
   ≤1px screen error; CudaRaster culls sub-sample tris, CuRast has no LOD. Nothing to add;
   pushing adaptive precision past the sub-pixel threshold = forbidden quality loss.

6. **Submit/BFS-pass batching is marginal** (already measured). B5's persistent-thread
   work-queue is a DIFFERENT lever (launch model) and is untested — not this wall.
