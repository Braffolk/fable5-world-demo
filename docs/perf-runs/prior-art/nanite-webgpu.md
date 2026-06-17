# Prior-art brief: `Scthe/nanite-webgpu`

**Source:** https://github.com/Scthe/nanite-webgpu (MIT). Shallow-cloned to
`docs/perf-runs/prior-art/sources/nanite-webgpu`. Author: Marcin Matzke (Scthe);
companion blog https://www.sctheblog.com/blog/hair-software-rasterize/ .

**Why it is the single most directly-comparable source:** it is a full Nanite
implementation in WebGPU/WGSL (Deno + Chrome), under the EXACT same constraint as
us — no `atomic<u64>`. It has a meshlet LOD DAG, GPU-driven multi-step culling
(per-instance + per-meshlet frustum + occlusion), a compute software rasterizer, a
hardware rasterizer for larger meshlets, billboard impostors for distant objects,
and a single-frame HZB depth pyramid for occlusion. It is a demo/education repo
(author explicitly says "a lot of code can be optimized"), so treat it as an
architecture reference, NOT a tuned perf target.

---

## 1. Raster architecture (as built)

Pass order (`src/renderer.ts:204` `cmdDrawNanite_GPU`):
1. `CullInstancesPass` → list of visible instances (indirect).
2. `CullMeshletsPass` (`cullMeshletsPass.wgsl.ts`) — thread-per-meshlet; frustum +
   occlusion (HZB) + Nanite LOD error test. For each surviving meshlet it computes
   the meshlet's screen-space AABB and **routes it to SW or HW** by area threshold
   (`registerDraw`, line 228): `pixelSpan.x*pixelSpan.y < softwareRasterizerThreshold`
   (default `1360` px², i.e. ~37×37 px — `constants.ts:176`). SW entries are appended
   to the BACK of one shared `drawnMeshletsList`, HW entries to the front
   (`drawnMeshletsBuffer.ts:63-84`). Counts drive indirect dispatch/draw args.
3. `RasterizeHwPass` — ordinary vertex+fragment pipeline for the big meshlets,
   writes HDR color + hardware depth (gets free early-Z / 2×2 quads).
4. `RasterizeSwPass` — the compute SW rasterizer (the hot analogue to ours).
5. `NaniteBillboardPass` — octahedral-ish 12-view billboard impostors for whole
   objects below `impostors.billboardThreshold` (4000 px²) — replaces far overdraw.
6. `RasterizeCombine` — fullscreen pass: decodes the SW result buffer, writes
   `frag_depth` + shaded color, depth-tests against the HW depth so HW/SW compose.
7. Ground, then `DepthPyramidPass` builds next-frame HZB (`max` reduction).

### The SW rasterizer kernel (`rasterizeSw/rasterizeSwPass.wgsl.ts`)
- **Work distribution:** indirect `dispatchWorkgroups(X=ceil(128/32)=4, Y=#SW-meshlets, Z=1)`,
  `@workgroup_size(32,1,1)` (`rasterizeSwPass.wgsl.ts:87`, dispatch at
  `naniteBuffers/index.ts:152`). `global_id.x` = triangle index within a meshlet
  (0..127); `global_id.y` selects the SW meshlet. So effectively **one workgroup-row
  per meshlet, one thread per triangle** — the same family as our one-workgroup-per-cluster.
  An inner `for` loop handles the case where #meshlets exceeds the 32768 Y-dispatch cap.
- **Per-triangle:** fetch 3 indices + 3 positions + 3 normals, `mvp*pos`, perspective
  divide, NDC→pixels (`rasterize()`, line 131). Backface cull via signed area
  (line 163). Compute screen AABB, clamp to viewport.
- **Per-fragment (the coverage loop, line 202-237):** classic incremental
  edge-function scanline — `EdgeC{A,B,C}` set up once (line 242), then add `A` per x
  and `B` per y (Pineda/fgiesen "optimizing the basic rasterizer"). Inside the
  `CX0>=0 && CX1>=0 && CX2>=0` test: barycentric interpolate depth + normal, pack,
  and `storeResult` → **`atomicMax` on one `array<atomic<u32>>`** (line 315).
- **NO tiling, NO binning, NO coarse/fine hierarchy, NO subgroup/quad packing, NO
  persistent threads, NO work queue, NO hierarchical-Z coverage in the SW raster.**
  Verified by exhaustive grep across `src/passes/`. The author leans entirely on
  culling + HW/SW/impostor routing to keep SW triangle/overdraw counts low rather
  than making the SW loop itself overdraw-cheaper.

### The visibility/depth election — the no-64-bit-atomic adaptation
This is the headline comparison. README §"Why untextured" and `rasterizeSwPass.wgsl.ts:260`
`createPayload`:
- One `u32` per pixel = **`(depth:u16 << 16) | (octNormal.x:u8 << 8) | octNormal.y:u8`**.
  Depth is `1.0 - ndcDepth` so **nearer = larger**, enabling `atomicMax` (buffer clears
  to 0, can't use min). The election word IS the payload — there is **no separate
  id-store**; the winning thread's `atomicMax` simultaneously commits depth AND the
  shaded attribute (normal). `RasterizeCombine` later unpacks depth+normal and shades.
- Author explicitly documents the design space (README:31): *"PPLL, or something with
  tiles, or double rasterization (1st pass writes depth, 2nd does `compareExchange()`)…
  the 32-bit limitation is only in WebGPU, so I chose to stick to UE5's solution"* —
  i.e. he picked the single-atomicMax pack and ate the precision loss (16-bit depth
  → "tons of z-fighting/leaks", README:74) rather than the slower exact methods.

---

## 2. Most transferable techniques (ranked)

### DIRECT (portable to our no-64b WGSL today)

**T1 — Pack the shaded ATTRIBUTE into the election word; drop the side-buffer id store.**
*This is the biggest concrete divergence from our pipeline.* We currently do
`(24b depth | 8b tiebreak)` atomicMax, then the WINNER plain-stores a 25b cluster/tri
id into a SECOND buffer; a later pass re-fetches the triangle and re-derives attributes.
nanite-webgpu instead packs `(16b depth | 16b oct-normal)` so the single `atomicMax`
commits everything — **no second store, no resolve-time re-fetch/re-transform of the
winning triangle.** Mechanism: the atomicMax word carries the answer.
- *Adapt:* our resolve already re-runs `buildTerrainShading` for the winner (gated
  behind `If(isT)` per commit 8b2a256). If we could pack our actual shading inputs
  (e.g. normal + a small material/wind key, or a packed albedo) into the low bits of
  the election word, the resolve pass stops re-fetching/re-transforming the winning
  triangle entirely. BUT: our budget is "ZERO perceptible quality loss", and the
  source itself warns 16-bit depth leaks badly (README:74). So the honest port is a
  **bit-budget study**, not a copy: how few depth bits do WE need given our depth
  range, and can the freed bits carry enough shading to skip the resolve re-fetch?
  This attacks the 40% transform/launch side indirectly (work moves out of resolve) —
  it does NOT cut the 60% coverage loop. Quality-preserving only if the depth-bit
  reduction is provably imperceptible at our depth range. Effort: medium.

**T2 — Incremental edge functions with the `B*y+C` term hoisted out of the x-loop.**
`edgeC()` precomputes `A,B,C` once per triangle; the row loop adds `B`, the column
loop adds `A` (`rasterizeSwPass.wgsl.ts:189-237`). Per covered pixel the inner test is
3 adds + 3 compares — no per-pixel mul. We already do "3 integer edge tests"; confirm
ours is the incremental add form (not recomputing `edgeFunction` per pixel as in the
commented-out lines 211-213). If ours still does any per-pixel multiply in the
barycentric-z, fold the depth interpolation into the same incremental scheme (depth is
linear in screen-x/y after the perspective divide is folded into the bary weights;
nanite-webgpu recomputes `C0/C1/C2` divides per pixel at line 218 — that is the part to
beat, not copy). Direct, quality-preserving, small. Effort: low.

**T3 — Tight AABB + scanline x-span clamp (already shipped by us).**
Lines 169-174 clamp the triangle bbox to the viewport and iterate only `[min,max)`.
This is exactly our shipped commit 19eb834 "SW raster scanline x-span". CONFIRMS our
win; no new lever. (Flag: source does NOT do per-row exact x-span entry/exit — it
still iterates the full bbox row and tests the edge sign per pixel, which our scanline
commit already improved on. So we are AHEAD of this source here.)

**T4 — Area-threshold HW/SW routing per meshlet (quality-preserving overdraw escape).**
`registerDraw` (line 228) sends meshlets whose screen AABB > ~1360 px² to the HARDWARE
rasterizer. The insight for us: the SW path is only a win for sub-pixel-triangle
meshlets; once a meshlet covers enough screen area, HW raster (with its free early-Z
depth-reject and 2×2-quad amortization) is cheaper *despite* the 4× quad waste, because
HW early-Z kills overdraw that our SW loop pays for in full. Our cost map says HW raster
of sub-pixel tris "wastes 4× on mandatory quads" — true for tiny tris — but this source
draws the OPPOSITE conclusion for *larger* meshlets, and routes by area. Mechanism: HW
early-Z is the overdraw weapon the SW path lacks. Indirect/big-refactor lever:
**if any of our foliage meshlets ever project large (near camera), routing those to a
HW pass with real early-Z could cut their overdraw cost that the SW atomicMax loop pays
in full.** Caveat: our refuted dead-ends note HW early-Z is "structurally absent from
the SW raster" — this technique is precisely about NOT using SW for those meshlets.
Effort: high (adds a second raster path + composite). Quality-preserving (same image).

### INDIRECT (adapt-the-principle; bigger lifts)

**T5 — Billboard impostors to delete far-field overdraw entirely.**
The whole `NaniteBillboardPass` exists because thousands of 1-px software triangles in
the far field are the dominant cost, and replacing a whole distant object with a single
camera-facing textured quad (12 pre-baked views, dither-blended, with baked normals for
runtime shading) is vastly cheaper (`constants.ts:142`, README:115 "most of the
instances are rendered as impostors"). For OUR holey foliage this is the most relevant
*structural* idea against the 60% coverage loop: **the cheapest covered fragment is the
one you never rasterize.** A leaf-crown billboard impostor (a few quads with baked
albedo+normal+alpha) for distant foliage instances would collapse the ~12–20× overdraw
on far crowns to a handful of textured fragments. The quality bar is the catch: impostor
pop/parallax must be below perceptible — the source dithers between 12 views and admits
it "does not handle up/down views". For foliage this is plausible (canopy mostly viewed
from a band of angles) and is a known UE5 Nanite technique (Karis 2021 talk, integrated
into the visbuffer). Effort: high. Quality risk: managed via view-count + cross-fade;
must be validated against the zero-perceptible-loss bar.

**T6 — Single-frame HZB occlusion as a *cheap approximate* cull (CONFIRMS our dead-end,
with one twist).** `cullOcclusion.wgsl.ts` samples a `max`-reduced depth pyramid from
the PREVIOUS frame at the mip matching the meshlet's pixel-span, and culls if the
meshlet's closest point is behind it. This is conservative and the README explicitly
says it does NOT help on the "holey" Jinx scene (gaps pin the pyramid to far depth,
README:111 "Swiss cheese theory") — **CONFIRMING our refuted conservative-occlusion
dead-end on holey foliage.** The twist worth noting: the source culls at the MESHLET
(cluster) granularity using the meshlet AABB vs an HZB mip, not per-triangle, and it is
single-pass (prev-frame HZB, no two-pass re-test). We already concluded conservative
occlusion is ~0% on foliage; this source independently reaches the same conclusion. No
new lever; do not resurface.

**T7 — Depth pyramid via `max` reduction / `textureGather`, float-filterable.**
`depthPyramidPass.wgsl.ts:46` does a 2×2 `max` per level (and notes `textureGather`
could replace 4 loads). Minor; only relevant if our HZB build shows up in a profile.
Effort: low; not on our critical path per our cost map.

---

## 3. What the source does NOT solve (honest gaps for our mission)

- **It does NOT attack the per-fragment coverage loop at all.** No tiled/binned
  rasterizer, no coarse→fine hierarchical coverage, no quad/SIMD fragment packing, no
  hierarchical-Z inside the SW raster, no early-out beyond the per-pixel edge sign and
  the bbox clamp. For our DOMINANT 60% coverage cost, this repo offers **no direct
  algorithmic win** — only the structural escapes T4 (route to HW) and T5 (impostors).
  Takeaway: a WebGPU peer under our exact constraint simply did NOT find a magic SW
  coverage-loop trick; they engineered the *workload* down instead. That is a strong
  signal that our biggest realistic wins are workload-shaping (impostors/HW routing /
  payload-pack-to-skip-resolve), not micro-opt of the inner loop.
- **Work distribution is the same one-workgroup(-row)-per-meshlet, thread-per-triangle
  scheme we already use** — no persistent-thread/work-queue improvement to borrow. The
  only aggregation idea is a *TODO* comment in `registerDraw` (line 234) referencing
  Wihlidal 2015 ballot-based atomic aggregation — for the CULL counter, not the raster.
- **The 64-bit-atomic story is the same trick, used MORE aggressively than us:** they
  pack the *attribute* into the election word (T1) and accept 16-bit depth precision
  loss. We currently keep 24-bit depth and a side id-store for quality. The portable
  idea is the *direction* (move work out of resolve by packing payload), bounded by our
  quality bar — NOT a copy of their 16-bit depth.

## 4. Bottom line for our 20→60fps mission
- **Best new lever from this source: T5 (foliage impostors) + T4 (HW routing for any
  large meshlets)** — both shrink the *workload* feeding the 60% coverage loop, which
  is the only thing this peer found that actually moves dense-overdraw cost, and both
  can be made image-equivalent within tolerance.
- **T1 (pack payload, skip resolve re-fetch)** is a real, portable cut to the 40%
  transform/launch+resolve side, gated by a depth-bit-budget quality study.
- T2/T3/T6/T7 mostly CONFIRM what we already ship or already refuted. No tiled/binned
  rasterizer or hierarchical coverage exists here to port — that idea must come from a
  different source (CUDA software rasterizers / cudaraster / Laine-Karras, UE5
  NaniteRasterizer.ush, or a tiled-deferred SW raster paper).
