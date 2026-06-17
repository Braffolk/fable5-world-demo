# Prior-art brief: jms55 "Virtual Geometry in Bevy" (0.15 / 0.16) + Bevy meshlet source

**Source.** Blog: https://jms55.github.io/posts/2024-11-14-virtual-geometry-bevy-0-15/
(primary), with follow-ups https://jms55.github.io/posts/2025-03-27-virtual-geometry-bevy-0-16/
and PR https://github.com/bevyengine/bevy/pull/17765 (texture atomics). Author jms55
is the Bevy meshlet/virtual-geometry maintainer. The blog is the most detailed
public WebGPU-adjacent Nanite clone narrative; the *actual hot code* lives in Bevy's
`crates/bevy_pbr/src/meshlet/`.

**Fetched.** WebFetch on all three URLs (cached, prose extracted). Crucially I also
shallow+sparse cloned the real source (`git clone --depth 1 --filter=blob:none
--sparse` then `sparse-checkout set crates/bevy_pbr/src/meshlet`) at Bevy commit
`f2f8584` (2026-06-17) so all citations below are to the CURRENT real shaders, not
the blog's simplified snippets. Key shaders copied to
`docs/perf-runs/prior-art/sources/posts/bevy-meshlet-shaders/`. WebFetch on the 0.16
post and PR #17765 each MISSED the fallback detail (model summarized "no 32-bit
fallback"); reading the source resolved it precisely (see §3).

**One-line verdict.** Bevy's SW raster is the closest living relative of ours
(compute, one-workgroup-per-cluster, atomicMax election, aimed at 1px triangles). It
gives us ONE genuinely new and directly-portable inner-loop lever — the **per-row
exact x-span (analytic scanline entry/exit) gated by a subgroup-coherent width test**
— plus several structural confirmations. BUT on the headline question (no-64-bit
atomics) Bevy is the OPPOSITE of a solution: it **requires 64-bit atomics and simply
PANICS on M1/A16**; its `r32uint` path is depth-ONLY (a prepass), not a depth+id
election. So Bevy confirms our constraint is real and confirms our single-32b-atomicMax
pack is a design THEY do not have — there is no fallback to copy here.

---

## 1. Raster architecture (as built, at commit f2f8584)

GPU-driven, two-pass occlusion, visibility-buffer. Pass order (`visibility_buffer_raster_node.rs`):
1. `clear_visibility_buffer` — compute shader zeroes the storage texture (replaces
   wgpu's "insanely expensive" buffer→image clear polyfill; PR #17765).
2. Instance cull → BVH cull → **cluster cull** (`cull_clusters.wgsl`). The cluster
   cull does frustum + LOD-error + HZB-occlusion, and for survivors computes the
   screen AABB and **routes SW vs HW** + appends to a shared two-ended buffer (§T3).
3. `remap_1d_to_2d_dispatch` — if SW cluster count exceeds the 1-D workgroup limit,
   reshape the indirect dispatch to a 2-D `ceil(sqrt(n))×ceil(sqrt(n))` grid
   (`remap_1d_to_2d_dispatch.wgsl:18-23`). Relevant to us: we have ~154k clusters; a
   1-D dispatch can blow the 65535 cap, so this reshape is a portable correctness/perf
   detail.
4. `visibility_buffer_software_raster` — the hot compute kernel (one workgroup per
   cluster). THE analogue to our `nanRasterWorld1`.
5. `visibility_buffer_hardware_raster` — ordinary vertex+fragment for large clusters,
   but it *still does `textureAtomicMax`* in the fragment shader rather than using a
   render-target/HW-depth, "to keep things simple and reuse the same bind group …
   between the rasterization passes" (0.15 post §5). So HW path here does NOT buy
   free early-Z — important, see §4.
6. Second cull pass (re-tests this-frame HZB) → second SW + HW raster.
7. `resolve_render_targets` — fullscreen passes that unpack the visbuffer into a real
   depth target and a material-id depth target, with the **`discard` on zero-depth**
   trick (§T2).
8. Material/shade pass re-derives attributes from the winning `packed_ids`
   (`visibility_buffer_resolve.wgsl`), incl. analytic barycentrics + ddx/ddy.
9. Depth-pyramid (HZB) build for next frame.

### The SW rasterizer kernel — `visibility_buffer_software_raster.wgsl`
- **Work distribution (same family as ours):** `@workgroup_size(128,1,1)`, **one
  workgroup per cluster** (`:29`). Phase 1: each thread projects 1–2 vertices into
  `var<workgroup> viewport_vertices: array<vec3f,256>` (`:26,:52-66`) then
  `workgroupBarrier()`. Phase 2: **one thread per triangle** (`:70`). So shared-memory
  vertex projection amortizes the transform across the workgroup — directly relevant
  to our 40% transform/launch + "~5× redundant vertex re-fetch" (§T5).
- **Per-triangle setup (`:80-107`):** backface via signed `edge_function` area;
  gradient setup `w_x,w_y` (the incremental edge increments) and `z_x,z_y` (linear
  depth gradients, with `vertices_z` pre-divided by `triangle_double_area` at `:86`).
  Screen AABB floor/ceil + viewport clamp (`:91-98`). Initial edge values `w_row`
  evaluated once at the bbox min pixel center (`:101-106`); `z_row` = `dot(vertices_z,
  w_row)`. **Depth is fully incremental: `z += z_x` per pixel, `z_row += z_y` per row
  — no per-pixel divide or mul in the inner loop.**
- **The inner loop — TWO variants, chosen by a SUBGROUP-COHERENT width test (`:110`):**
  `if subgroupAny(max_x - min_x > 4.0)`.
  - **Scanline variant (wide, `:111-144`):** precompute per-edge reciprocals
    `inverse_edge_012 = 1/(-w_x)` ONCE per triangle (`:114`), then per row compute the
    analytic x-entry/x-exit of the triangle on that scanline via `cross_x = w_row *
    inverse_edge_012` and select min/max by edge orientation (`:117-127`). The inner
    `for x` loop then runs only `[x0,x1]` — i.e. **only pixels actually inside the
    triangle's span on that row**, not the whole bbox row.
  - **Bbox variant (narrow ≤4px, `:146-165`):** iterate the full bbox row, test
    `min3(w[0],w[1],w[2]) >= 0.0` per pixel.
  - BOTH still keep a per-pixel `min3(w) >= 0` test even in the scanline path
    (`:132`), with the author's comment "*this shouldn't be needed, but there's bugs
    without it*". So the scanline still pays a 3-compare guard; the win is iterating
    fewer x's, not removing the test.
- **The write (`:169-177`):** `write_visibility_buffer_pixel` bitcasts the
  interpolated `z` to u32, packs `(u64(depth)<<32)|u64(packed_ids)` and
  `textureAtomicMax`. `packed_ids = (cluster_id<<7)|triangle_id` (`:77`) → 7 bits of
  triangle id (≤128 tris/meshlet), 25 bits cluster id.
- **NO tiling, NO binning, NO coarse→fine hierarchy, NO per-pixel hierarchical-Z, NO
  persistent-thread/work-queue, NO quad packing.** Same as the other WebGPU peer:
  Bevy attacks the *workload* (cull + LOD + HW/SW split) and the *iteration extent*
  (scanline span), not the per-fragment election cost.

### The visibility/depth election — and the no-64-bit-atomic story (the headline)
- **Bevy uses TRUE 64-bit atomics on a storage TEXTURE.** `meshlet_bindings.wgsl:199`:
  `var meshlet_visibility_buffer: texture_storage_2d<r64uint, atomic>` when
  `MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT` is defined; election is
  `textureAtomicMax(buf, xy, (u64(depth)<<32)|u64(packed_ids))`
  (`visibility_buffer_software_raster.wgsl:170-177`). Upper 32b = `bitcast<u32>(z)`
  (raw f32 depth bits; reverse-Z so larger=nearer), lower 32b = full id. **The single
  atomicMax commits BOTH depth and the full 25b+7b id with zero precision loss** — the
  exact thing our no-64b constraint forbids.
- **The `r32uint` path is NOT a depth+id fallback — it is depth-ONLY.** When
  `..._OUTPUT` is *undefined* (the depth-prepass / shadow path, `:173-175` and
  `meshlet_bindings.wgsl:200-202`), `let visibility = depth;` and it `atomicMax`es
  raw f32 depth bits only — there is no id, because the prepass doesn't need one.
- **On hardware WITHOUT 64-bit texture atomics, Bevy does not degrade — it PANICS.**
  PR #17765 discussion: `TEXTURE_INT64_ATOMIC` is M2-Macs-or-newer / A17-or-newer;
  "meshlets panics on startup" on M1; jms55: "*you need an M2 GPU or newer.*" There is
  **no documented 32-bit depth+id election fallback in Bevy.**
- **Therefore for OUR mission this source is a CONFIRMATION, not a recipe:** it proves
  our constraint is real on exactly our class of hardware (Apple M-series M1/A16), and
  it shows that the most-developed WebGPU-adjacent clone chose to *require* the 64-bit
  feature rather than solve the 32-bit case. **Our (24b depth | 8b tiebreak)
  single-atomicMax + side-id-store is a more portable design than Bevy's**, so there
  is nothing to port on the election itself — only to note Bevy's freedom from the
  resolve-time re-fetch that 64-bit packing buys (which we cannot have without 64b).

---

## 2. Most transferable techniques (ranked)

### DIRECT (portable to our no-64b WGSL today)

**T1 — Per-row analytic x-span (true scanline entry/exit), gated by a
SUBGROUP-COHERENT width test. [the one genuinely new inner-loop lever]**
`visibility_buffer_software_raster.wgsl:110-144`. For triangles wide enough to matter,
precompute edge reciprocals once (`inverse_edge_012 = 1/(-w_x)`, `:114`) and per
scanline compute the exact `[x0,x1]` where all three edges are ≥0 via `cross_x =
w_row * inverse_edge_012` + orientation `select` (`:117-127`), iterating ONLY interior
pixels. This is strictly more than our shipped commit 19eb834 "scanline x-span":
ours skips outside-bbox pixels but (per our cost map) still tests the edge sign per
pixel inside the bbox row; Bevy *computes the analytic entry/exit* so the loop bound
itself excludes most outside-triangle pixels. CAVEAT/HONESTY: our triangles are ~1px,
and Bevy ONLY takes this path when `subgroupAny(max_x-min_x > 4.0)` — i.e. it
deliberately does NOT use analytic scanline for sub-4px tris (the reciprocal+select
overhead loses on tiny tris). So the *direct* port is small for us UNLESS we have a
near-camera band of wider foliage tris; the *transferable principle* is the
**subgroup-coherent branch**: pick bbox-vs-scanline once per subgroup with
`subgroupAny`, so the whole subgroup runs one path and avoids divergence. WGSL has
`subgroupAny` (subgroups are available in WebGPU behind the `subgroups` feature on
Metal). Mechanism: removes per-thread control-flow divergence in the inner loop +
shrinks x-iteration for the wide minority. Attacks the 60% coverage loop. Quality-
preserving (same covered set). Effort: low-medium. *Verify first* whether our foliage
ever produces >4px-wide tris near camera; if it is 100% sub-pixel, this lever is ~0
for us and we should say so.

**T2 — `discard` zero-depth fragments in the resolve pass → 2× resolve.**
`resolve_render_targets.wgsl:22` `if depth == 0u { discard; }` and `:34` for material
depth. jms55: this "*doubled the performance of the resolve depth/material depth
passes*" (0.16 post / PR #17765). Background pixels (visbuffer never written) early-out
of the fullscreen resolve instead of running the full unpack+shade. This is the exact
"discard background pixels in resolve" trick the mission flagged. For us: our resolve
already gates `buildTerrainShading` behind `If(isT)` (commit 8b2a256) — that IS this
idea for the terrain branch. The portable extension: ensure the *entire* resolve
fragment (depth write + any per-pixel work) early-discards when the election word is
still the clear value, not just the shading sub-branch. Attacks the resolve side of
the 40%. Direct, quality-preserving, low effort. Likely already-mostly-shipped — flag
as CONFIRM + a small completeness check.

**T3 — Two-ended shared cluster buffer (SW appends left, HW appends right) — avoids a
separate sort/allocation and a second dispatch-setup.**
`cull_clusters.wgsl:81-92`: `if sw_raster { slot = atomicAdd(&sw_args.x,1)+prev[0] }
else { slot = rightmost_slot - (atomicAdd(&hw_args.instance_count,1)+prev[1]) };
meshlet_raster_clusters[slot] = ...`. One buffer, two monotonic cursors growing toward
each other; the SW dispatch reads `[0..sw_count)` and the HW draw reads from the right
end (`visibility_buffer_hardware_raster.wgsl:31-32`). 0.15 post §2 verbatim: "*Software
rasterized clusters will be added starting from the left side of the buffer, while
hardware rasterized clusters will be added from the right side.*" For us: IF we ever
split work into ≥2 raster queues (e.g. a sub-pixel SW queue vs a wider-tri queue, or
SW vs a future HW path), this is the clean way to bucket in ONE pass with no sort and
no second compaction dispatch — directly relevant to cutting the 40% launch/setup.
Direct, portable (plain `atomicAdd` on u32). Effort: low (data-structure change).
Quality-neutral. Only valuable once we actually have two buckets — today we have one.

**T4 — Shared-memory vertex projection: project each meshlet vertex ONCE into
`workgroup` memory, then one-thread-per-triangle reads 3 cached verts.**
`visibility_buffer_software_raster.wgsl:26,52-67,74-76`. Each of the 128 threads
projects 1–2 vertices into `viewport_vertices[256]`, barrier, then triangle threads
index that array — so a vertex shared by N triangles is transformed ONCE, not N times.
This DIRECTLY attacks our cost map's "*~5× redundant vertex re-fetch*" inside the 40%
transform/launch cost: we currently `fetchWorldVert` 3× per triangle with ~48 wind
texture taps, re-fetching shared verts ~5×. Porting the workgroup-shared projection
(transform+wind-animate each unique meshlet vertex once into shared memory, then have
triangle threads read it) could cut the per-cluster transform + the expensive wind
taps by ~5×. Mechanism: dedupe transform via `var<workgroup>` + `workgroupBarrier`.
This is the single most promising lever from this source for our 40% slice. CAVEAT:
needs enough shared memory (256 vec3 = 3KB, fine) and our wind animation must be
expressible as a pure per-vertex function (it is — it's a per-vertex displacement).
Direct, quality-preserving, **medium effort, high expected payoff** — recommend
prioritizing.

### INDIRECT (adapt-the-principle)

**T5 — Area-threshold SW/HW routing (`< 64px on both axes` → SW, else HW; near-plane →
HW).** `cull_clusters.wgsl:76-79` (`all(aabb_size <= vec2(64.0))`) and the near-plane
guard reused from occlusion. Same idea as the other WebGPU peer (nanite-webgpu T4) but
with a much larger SW threshold (64×64 vs ~37×37) AND, notably, **Bevy's HW path still
uses atomicMax not HW depth** — so Bevy's HW routing is NOT about gaining free early-Z;
it's about (a) avoiding the SW near-plane homogeneous-divide singularity and (b)
amortizing big triangles over the fixed-function rasterizer. For us this REFINES the
nanite-webgpu T4 takeaway: the win of an HW path is the near-plane robustness +
fixed-function efficiency on large tris, and if you want the *early-Z overdraw weapon*
you must bind a real depth target (which Bevy chose NOT to do). Our refuted dead-end
says "HW raster of sub-pixel tris wastes 4× on quads" — true, and Bevy agrees by
sending only >64px clusters to HW. Indirect lever: a near-plane / large-cluster HW
escape hatch, only worth it if our foliage projects large near camera. Effort: high.
Quality-preserving.

**T6 — Two-pass occlusion with prev-frame-then-this-frame HZB at MESHLET granularity;
`firstLeadingBit`-selected HZB mip; 4×4 min-gather.** `cull_clusters.wgsl:49-61`
(occluded clusters push children to a second queue) + `meshlet_cull_shared.wgsl:146-170`
(`occlusion_cull_screen_aabb`: pick mip via `firstLeadingBit(max_size)`, sample a 4×4
block and take the min, cull if `aabb.max.z <= curr_depth`). This is conservative
(min-reduction HZB, tests fully-behind). Per our refuted dead-ends, conservative
occlusion is ~0% on holey foliage (a gap pins the pyramid to far depth). Bevy's HZB is
a `min` (reverse-Z "closest") pyramid and the test is conservative, so it **inherits
the same Swiss-cheese failure on holey crowns** — CONFIRMS our dead-end. Do NOT
resurface. One portable micro-detail if our HZB ever matters: the
`firstLeadingBit`-based mip pick + 4×4 min-gather (`:159-168`) is a tidy conservative
sampler; not on our critical path.

**T7 — Analytic barycentrics + ddx/ddy in the resolve (no GPU quad derivatives), from
The-Forge.** `visibility_buffer_resolve.wgsl:41-83` (`compute_partial_derivatives`).
Because the visbuffer resolve runs in a fullscreen pass where you only have the winning
triangle's 3 clip-space verts, Bevy reconstructs perspective-correct barycentrics AND
texture-gradient ddx/ddy analytically from the triangle, instead of relying on
hardware 2×2-quad derivatives. For us this is the standard visbuffer-resolve technique;
relevant only as a reference if/when we add textured foliage materials needing correct
mip selection in resolve. Not a perf lever for the 60%/40% today. Effort: medium if
adopted. Quality-relevant (correct mipping), not perf.

---

## 3. What this source does NOT solve (honest gaps)

- **It does NOT solve the no-64-bit-atomic election** — it *requires* 64-bit texture
  atomics and PANICS on M1/A16 (PR #17765). Its `r32uint` path is depth-only prepass,
  NOT a depth+id fallback. There is nothing here to port for our central constraint;
  our single-32b-atomicMax-plus-side-store is already the more portable answer. This is
  the most important honest finding: a leading WebGPU-adjacent clone looked at our
  exact hardware and chose to drop it rather than solve it.
- **No tiled / binned / coarse→fine rasterizer, no hierarchical-Z coverage, no quad/
  SIMD fragment packing, no persistent-thread work queue inside the raster.** Same as
  the other WebGPU peer. For our DOMINANT 60% per-fragment loop the only inner-loop
  win here is T1 (analytic x-span + subgroup-coherent branch), and that only helps the
  >4px-wide minority. The tiled/binned/hierarchical-coverage idea must come from a
  different source (CUDA cudaraster / Laine-Karras, FreePipe, ComputeRaster, or a
  tiled-deferred SW-raster paper) — NOT from Bevy.
- **Work distribution is the same one-workgroup-per-cluster, thread-per-triangle
  scheme we already use.** No aggregation/persistent-thread improvement to borrow. The
  only adjacent ideas are the 1D→2D dispatch reshape (`remap_1d_to_2d_dispatch.wgsl`,
  correctness for >65535 clusters) and the two-ended bucket buffer (T3).
- **HW path forgoes early-Z** (uses atomicMax in fragment, hardware_raster.wgsl:64),
  so it does NOT demonstrate the early-Z overdraw cure — it is not a counterexample to
  our refuted "HW early-Z absent from SW raster" note.

## 4. Bottom line for our 20→60fps mission
- **Best new lever: T4 (workgroup-shared vertex projection)** — directly cuts our 40%
  transform/launch by killing the ~5× redundant vertex re-fetch + the ~48 wind taps,
  amortizing each unique meshlet vertex once across the workgroup. Medium effort, high
  payoff, quality-neutral. Prioritize.
- **T1 (analytic x-span + `subgroupAny` coherent branch)** is the only inner-loop
  (60%) lever, but gated to >4px tris; measure whether our foliage ever projects that
  wide near camera before investing. The subgroup-coherent branch idea is portable
  regardless.
- **T2 (resolve zero-depth discard)** and **T3 (two-ended cluster buffer)** are small,
  portable, partly already-shipped (T2) — adopt T3 only once we have ≥2 raster buckets.
- **T5/T6/T7 mostly CONFIRM** our existing decisions (HW-routing tradeoff,
  conservative-occlusion dead-end on holey foliage, standard visbuffer resolve) and
  add no new perf lever.
- **The election (no-64b atomics): nothing to port.** Bevy needs 64b and skips our
  hardware; our 32b pack is the more portable design. Treat as constraint-confirmation.
