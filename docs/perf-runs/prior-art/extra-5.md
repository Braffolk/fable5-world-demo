# extra-5 — "Modernizing Granite's mesh rendering" (Hans-Kristian Arntzen / themaister, 2024-01-17)

Source: https://themaister.net/blog/2024/01/17/modernizing-granites-mesh-rendering/
Local copies:
- `docs/perf-runs/prior-art/sources/extra-5-granite-mesh.html` (raw)
- `docs/perf-runs/prior-art/sources/extra-5-granite-mesh.txt` (stripped text, line-cited below)

## What this source IS (and is NOT)

This is a long engineering blog about rewriting Granite's renderer around **GPU-driven mesh-shader / MultiDrawIndirect (MDI) culling** with a custom GPU-decompressed meshlet format. It is a *front-end geometry submission* pipeline that feeds the **hardware rasterizer**.

It is **NOT a software rasterizer.** There is no per-fragment coverage loop, no compute-raster depth election, no visibility-buffer write in this post. I verified Granite's own OVERVIEW.md documents only hardware graphics pipelines (no compute raster / visbuffer / atomic-depth module). The post mentions a SW rasterizer only *by reference to Nanite* — and crucially confirms our central constraint:

> "Nanite relies heavily on rendering primitive IDs to a visibility buffer... In the primary compute software rasterizer, this becomes a **64-bit atomic**, and in the mesh shader fallback, a single primitive ID is exported to fragment stage as a per-primitive varying, where fragment shader just does the atomic." (txt:9)

So for our no-64-bit-atomic constraint this source offers **zero direct help on the depth ELECTION** (it sidesteps the visbuffer entirely by using HW raster + flat/per-primitive varyings). Its value is entirely on the **front-end: killing primitives before they ever reach a rasterizer** — which maps onto our **40% per-triangle transform+setup+launch stage**, and *indirectly* onto the **60% per-pixel loop** (every triangle culled pre-raster is overdraw that never happens).

Be honest about fit: our profile says triangles are TINY (~1px) and clusters are WELL-FILLED (235/256), and our overdraw is on *genuinely visible* holey-foliage leaf cards. Granite's headline 31M→1.2M (96%) win is on a *pathologically* micro-poly scene where most prims have **zero coverage** (land between pixel grid lines). How much of our 36M visible tris are true zero-coverage vs. truly-covered-but-overlapping is the open question that gates how much of this transfers. The micro-poly test only removes the *zero-coverage* fraction. It is still likely a real win and is cheap to try.

## Raster architecture (front-end pipeline)

1. Offline-bake meshes into **256 vert/prim meshlets**, internally grouped as **8×32 "sublets"** so the runtime can specialize to 32-wide subgroups (txt:32-36). Compressed payload (base+delta, tightly-packed bits) decoded inline on GPU (txt:106-243).
2. A compute "task" stage does hierarchical AABB + frustum + back-face-cone culling and emits an MDI / `vkCmdDrawMeshTasksIndirectCountEXT` buffer (txt:244-245, 763-802).
3. Per-meshlet, in the mesh shader (or compute), do **per-primitive culling** entirely in-subgroup: clip-code early-out → 2D-cross back-face → **micro-poly subpixel-bbox reject** → **subgroupBallot compaction** of survivors → SetMeshOutputCounts (txt:337-514).
4. **Two-phase HiZ occlusion culling** with a 1-bit-per-meshlet visibility bitmask carried across frames (txt:804-823). [NOTE: our roadmap already refuted conservative HiZ occlusion on holey foliage — see "dead confirmations" below.]

Reported on RTX 3070, 13³-instanced Niagara mesh, 63.6M tris: vkCmdDrawIndexed no-cull 5.5ms → frustum 4.3 → MDI+cone 3.9 → +back-face 3.3 → **+micro-poly 1.9** → (stats off) 1.65 → (NV barycentric trick) 1.0ms (txt:276-553).

## Transferable techniques

### T1 (INDIRECT, HIGH VALUE) — Micro-poly subpixel-bbox rejection: cull triangles whose snapped bbox lands between pixel grid lines
The single biggest front-end win in the post (3.3ms→1.9ms; 31M→1.2M prims, txt:461-492). After transforming each vertex to **fixed-point window coords** (8 subpixel bits), reject a triangle if the floor() of its min and max corner land in the *same* pixel cell on BOTH axes:
```glsl
const int SUBPIXEL_BITS = 8;
vec2 lo = floor(ldexp(min(min(a,b),c), ivec2(-SUBPIXEL_BITS)));
vec2 hi = floor(ldexp(max(max(a,b),c), ivec2(-SUBPIXEL_BITS)));
active = all(notEqual(lo, hi));   // survives only if bbox straddles a pixel boundary
```
The viewport transform bakes in a 0.5px sample-center shift and a 1-subpixel nudge for top-left fill rule so `[1.0,2.0]` survives but `[1.0+1/256, 2.0]` does not (txt:476-489).

Why it maps to US: this directly attacks the **40% transform+setup+launch**. Each of our ~36M visible tris already gets transformed; if a meaningful fraction are true zero-coverage micro-polys, this kills them *before* the per-pixel coverage loop spins up edge setup + the per-fragment loop — also shaving the **60% loop**. Our edge functions already produce the data needed (we snap to a sub-pixel grid for the integer edge tests), so the marginal cost is ~6 floors + 4 compares per triangle.

Adapt to WGSL / no-64-bit-atomic: **fully portable, atomics irrelevant** (this is pure per-triangle ALU, no atomic, no subgroup). Reuse our existing fixed-point window coords; if we snap with N subpixel bits, use `floor(ldexp(...))` or integer `>> N`. Watch our top-left vs bottom-left fill convention and Y-flip (post warns Vulkan vs D3D differ, txt:378; WebGPU matches D3D Y-down NDC-to-framebuffer — match whatever our existing coverage test uses so the cull is exactly conservative w.r.t. the real test). CRITICAL for our "zero quality loss" bar: the test must be **conservative against our actual rasterizer's coverage rule** — if our coverage test could ever light a pixel for a bbox that floors equal, this would drop it. Validate by A/B pixel-diffing a frame with the cull on/off; if identical, ship.

### T2 (INDIRECT/DIRECT, MEDIUM) — FP32-exact 2D-cross back-face cull with magnitude-guarded GE/GT
Back-face cull via 2D cross product of window-space edges, done in FP32 (not int64), with an exactness guard (txt:427-457):
```glsl
precise float pos_area = ab.y*ac.x;
precise float neg_area = ab.x*ac.y;
if (abs(pos_area) < 16777216.0) active = pos_area >  neg_area; // exact below 2^24
else                           active = pos_area >= neg_area; // conservative above
```
For foliage we likely render **double-sided** leaf cards (holey crowns), so a hard back-face cull would change the image and is forbidden. Value to us is the *technique*, not the cull: it shows how to get **exactly-rounded** signed-area / orientation in FP32 by guarding against the 2^24 mantissa limit, instead of paying int64. We can reuse this guard for any orientation/winding/edge-sign decision in our setup where we want determinism without int64. Mark the cull itself **not applicable** unless a material is known single-sided; mark the FP32-exactness trick **portable & useful**.

### T3 (DIRECT, MEDIUM) — In-subgroup ballot compaction of surviving primitives (and active-vertex mask)
After per-prim cull, compact survivors with a ballot + exclusive popcount, and dedup the live vertices with an OR-reduced bitmask (txt:499-514):
```glsl
uvec4 prim_ballot = subgroupBallot(is_active_prim);
shared_active_prim_offset      = subgroupBallotExclusiveBitCount(prim_ballot); // packed write index
shared_active_prim_count_total = subgroupBallotBitCount(prim_ballot);
uint vert_mask = is_active_prim ? (1u<<prim.x)|(1u<<prim.y)|(1u<<prim.z) : 0u;
shared_active_vert_mask = subgroupOr(vert_mask);                 // unique live verts
shared_active_vert_count_total = bitCount(shared_active_vert_mask);
```
Two ideas here: (a) **compaction** so downstream work iterates only survivors (no idle lanes); (b) the **active-vertex mask** so you transform/decode each unique vertex once.

Adapt to WGSL: **portable.** WebGPU exposes the `subgroups` feature (`enable subgroups;`) with `subgroupBallot` (returns `vec4<u32>`), `subgroupOr`, `subgroupAdd`/`subgroupExclusiveAdd`, `subgroupShuffle`, `subgroupElect`, available on M-series/Metal. There is no single `subgroupBallotExclusiveBitCount` builtin in WGSL — synthesize it one of two ways:
- `let idx = subgroupExclusiveAdd(select(0u, 1u, is_active));` (cheap, exact prefix index), or
- mask the ballot by lanes-below-self and popcount: build a below-mask from `subgroup_invocation_id` and `countOneBits(ballot & below_mask)` summed across the (up to 4) `vec4<u32>` words.
`subgroupBallotBitCount` = sum of `countOneBits()` over the 4 ballot words. NOTE on our actual bottleneck: T3 helps **idle-lane / partial-fill**, which our cost map says is NOT a lever (clusters are 235/256 full). So T3's compaction is mostly redundant for us *unless* paired with T1 — i.e. after micro-poly cull, a cluster that was 235/256 full may drop to e.g. 40/256 survivors, and *then* compaction + active-vertex-mask avoids re-running our 5×-redundant `fetchWorldVert` (~48 wind-animation texture taps/tri) on dead triangles. So the real lever is **T1→(T3 active-vertex mask)** combined: cull micro-polys, then transform/wind-sample only the unique vertices feeding survivors. That attacks the 40% transform stage's redundant-fetch problem head-on.

### T4 (INDIRECT, MEDIUM) — Transform/shade only the UNIQUE, SURVIVING vertices ("split vertex/attribute shading")
RADV/NGG-style: after compaction, route unique live vertex IDs through groupshared and shade each exactly once (txt:636-664):
```glsl
if (meshlet_lane_has_active_vert()) { shared_attr_index[meshlet_compacted_vertex_output()] = vert_id; }
barrier();
if (gl_LocalInvocationIndex < shared_active_vert_count_total) { /* load+transform that one vertex */ }
```
> "Only computing visible attributes is a very common optimization in GPUs in general." (txt:664)

Maps to our **40% stage's "~5× redundant vertex re-fetch, ~48 texture taps for wind animation"** — the single most concrete overlap with our cost map. Our one-workgroup-per-cluster already has the locality; the missing piece is: (1) build the unique-vertex set for the cluster (the OR-mask from T3, or a dedup over the 256 indices), (2) run `fetchWorldVert` + wind animation **once per unique vertex** into shared/registers, (3) have the per-triangle edge setup read transformed positions from shared. Even *without* the micro-poly cull this removes the ~5× re-fetch. Fully portable (shared memory + barriers + optional subgroup ops; no atomics, no 64-bit).

### T5 (INDIRECT, LOW/INFO) — Vertices-as-barycentrics deferred attribute fetch
NVIDIA-specific: output only vertex IDs + barycentrics, refetch/interpolate attributes in the fragment stage (1.0ms, txt:516-555). Author rejects it for his use case: "Moves a ton of extra work to fragment stage... I'm not aiming for Nanite-style micro-poly hell" (txt:558-560).
For US this is anti-aligned: we ARE the micro-poly/overdraw case, so moving work per-fragment (where we're 60% bound) is the wrong direction. Logged as a **dead-end confirmation** for our profile, not a lever. (It also relies on a HW fragment stage + `fragment_shader_barycentrics`, which our compute SW raster doesn't have.)

### T6 (INFO) — Clip-code early accept/reject before any area math
Cheap per-vertex clip codes (X/Y/W planes, an INACCURATE guard-band flag), shuffled to triangle vertices; `and_code` rejects fully-outside, `or_code` of NEGATIVE_W/INACCURATE force-accepts (avoids divide-by-near-zero) before back-face/micro-poly (txt:342-422). Portable per-triangle ALU; useful hygiene for our setup to skip area/coverage math on trivially off-screen or near-plane-clipped tris. Minor for us since the cluster cut already frustum-culls coarsely; only helps the boundary clusters.

## Dead-end CONFIRMATIONS (source agrees with our already-refuted list)
- **64-bit atomic visbuffer is the Nanite default** (txt:9) — confirms our central constraint; source offers no no-64b adaptation (it avoids the visbuffer).
- **Two-phase HiZ occlusion culling** (txt:804-823) is the source's occlusion answer — our roadmap already refuted conservative HiZ on holey foliage (gaps pin max-Z to far plane). Source is conservative HiZ, so it would be ~0% for us too; do NOT resurface.
- **Per-fragment deferred attribute fetch (T5)** confirms moving work to the per-fragment stage is bad when micro-poly/overdraw dominated — aligns with our 60% being the per-pixel loop.

## Net recommendation for our system
The portfolio worth prototyping, in order:
1. **T1 micro-poly subpixel-bbox reject** — cheapest, attacks 40% (and via reduced overdraw, 60%); MUST be made conservative against our exact coverage rule and pixel-diff-verified for zero quality loss. Win size depends entirely on what fraction of our 36M visible tris are true zero-coverage.
2. **T4 + T3 active-vertex mask** — transform/wind-sample each unique cluster vertex once instead of ~5×; directly named in our cost map as the 40%-stage waste. Portable, no atomics.
3. **T2 FP32-exact-area guard** — adopt for deterministic winding/edge math without int64, regardless of whether we cull.
None of these touch the depth ELECTION; our packed-24b-depth|8b-id atomicMax stays as-is. The source contributes nothing new to the no-64-bit-atomic election problem.
