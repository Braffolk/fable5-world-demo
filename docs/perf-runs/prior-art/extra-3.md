# Prior-art brief: extra-3 — Tellusim "Compute versus Hardware" (compute SW raster on Apple M1/Metal)

- **slug**: extra-3
- **kind**: post (engineering blog + verbatim shader source + cross-vendor benchmark table)
- **source**: Tellusim Technologies, *Compute versus Hardware* (the "compute-raster" post), Sep 13 2021.
- **refs**: https://tellusim.com/compute-raster/ · binaries: TellusimDrawMeshlet.zip (Windows only)
- **fetched**: `curl -sL https://tellusim.com/compute-raster/` → `/tmp/tellusim-compute-raster.html` (31 KB, HTTP 200, OK). WebFetch gave the prose + table + numbers cleanly, but its markdown pass mangled the GLSL (the `<` / `>=` operators in the for-loop conditions and the `if(p0.z < 0…)` clip are literal `<` chars in the page's `<pre>`, which both WebFetch's converter and a naive regex strip/truncate). Recovered the EXACT shader by byte-slicing the raw HTML around `atomicMax`, `void main()`, and `void rasterize` (python `s.find(...)`) — the load-bearing inner loop, the Metal buffer-vs-image `#if CLAY_MTL` split, the 1/256 subpixel snap, and the full transform/index-unpack are all quoted verbatim below. No repo (closed-source Clay engine; only Windows demo binaries shipped). Single page = whole source; nothing else to fetch.

---

## What it is

A 2021 micro-benchmark + minimal reference shader from the Tellusim/Clay engine that pits **compute-shader software rasterization against hardware** (single DIP, Mesh Shaders, MultiDrawIndirect) across Nvidia/AMD/Intel/Qualcomm/**Apple** GPUs, rendering 498 990 meshlets (64 verts / 128 tris each) **depth-only**. It is short and not a full Nanite, but it is THE most directly-relevant public data point for our exact hardware/constraint class, for three reasons the mission already flagged:

1. **It runs on Apple M1 (Metal) and reports the numbers** — Compute **2.30 B tri/s** vs Single-DIP HW **1.37 B** vs Mesh-Shader **739 M**. Compute beats the best HW path by **1.68×** for tiny triangles on our exact GPU family. (A14/iOS: 1.02 B compute vs 666 M HW = 1.53×.) This is the empirical justification that SW raster of sub-pixel tris is the right architecture on M-series — the very premise of our system.
2. **It hits, and documents, our exact constraint**: "We will limit software rasterization to **depth-only mode because 64-bit atomics are not available on Mobile devices and Metal**." Its election is a single 32-bit `atomicMax(floatBitsToUint(z))` — **literally the depth half of our 24b|8b packed `atomicMax`**. It then states the missing-feature wishlist (`imageAtomicPayloadMax`) that is exactly the gap our team closed.
3. **It names the Metal "buffer-from-texture" workaround** for Metal's missing *texture* atomics: use a `std430` buffer + buffer `atomicMax` instead of an `imageAtomicMax` on an `r32ui` image.

**Blunt caveat up front:** this is a *throughput* micro-bench on uniform, NON-overdrawn meshlets ("without any culling except back-face"), depth-only, with NO payload, NO occlusion, NO LOD, NO binning, and an explicitly **"unoptimized"** scanline. So it CONFIRMS our architecture and hardware bet and pins down the no-64-bit-atomic election, but it offers **no new coverage-loop or work-distribution optimization** — its raster is strictly *simpler* than ours (and in two ways measurably *behind* what we already ship). Its value is confirmation + the Metal storage-format detail + one micro-codegen idea, not a big refactor.

---

## Raster architecture (verbatim)

One compute group **per meshlet**, one triangle **per thread**, shared-memory vertex+index staging, per-triangle **bbox scanline** with incremental barycentric DDA, single 32-bit `atomicMax` depth election. No tiles, no bins, no hierarchical Z, no coarse/fine split, no persistent threads.

**(a) Distribution + cooperative load (`void main`)** — group = meshlet, thread 0 reads meshlet header into shared, all threads cooperatively transform verts to shared, unpack indices to shared, then `if(local_id < num_primitives)` each thread rasterizes one triangle:
```glsl
layout(local_size_x = GROUP_SIZE) in;
shared vec3 positions[NUM_VERTICES];
shared uint indices[NUM_PRIMITIVES * 3u];
// thread 0 loads meshlet header (num_primitives/vertices, base_index/vertex)
// then ALL threads cooperatively transform vertices into shared 'positions':
position = vec4(dot(row_0,position), dot(row_1,position), dot(row_2,position), 1.0f); // instance xform
position = projection * (modelview * position);                                       // to clip
positions[index] = vec3(round((position.xy*(0.5f/position.w)+0.5f)*surface_size*256.0f)/256.0f - 0.5f,
                        position.z/position.w);                                       // 1/256 SNAP + persp divide
memoryBarrierShared(); barrier();
// rasterize triangles
[[branch]] if(local_id < num_primitives) {
  uint index = local_id*3u;
  rasterize(positions[indices[index+0u]], positions[indices[index+1u]], positions[indices[index+2u]]);
}
```
Note `round(... * 256.0f)/256.0f - 0.5f`: a **1/256-pixel fixed-point snap + half-pixel sample bias folded into the vertex** — identical in intent to our `1/256` grid snap (`NaniteRaster.ts:15-16`).

**(b) Per-triangle setup + cull (`void rasterize`, head):**
```glsl
[[branch]] if(p0.z < 0.0f || p1.z < 0.0f || p2.z < 0.0f) return;   // near-plane: DROP (no clip!)
vec3 p10 = p1-p0; vec3 p20 = p2-p0;
float det = p20.x*p10.y - p20.y*p10.x;
[[branch]] if(det >= 0.0f) return;                                  // back-face
vec2 min_p = floor(min(min(p0.xy,p1.xy),p2.xy));
vec2 max_p = ceil (max(max(p0.xy,p1.xy),p2.xy));
[[branch]] if(max_p.x < 0.0f || max_p.y < 0.0f || min_p.x >= surface_size.x || min_p.y >= surface_size.x) return;
min_p = clamp(min_p, vec2(0.0f), surface_size-1.0f);
max_p = clamp(max_p, vec2(0.0f), surface_size-1.0f);
vec2 texcoord_dx = vec2(-p20.y, p10.y)/det;   // d(bary)/dx, computed ONCE
vec2 texcoord_dy = vec2( p20.x,-p10.x)/det;   // d(bary)/dy
vec2 texcoord_x = texcoord_dx*(min_p.x-p0.x);
vec2 texcoord_y = texcoord_dy*(min_p.y-p0.y);
```

**(c) The inner coverage + DEPTH ELECTION loop (verbatim — the load-bearing part):**
```glsl
for(float y = min_p.y; y <= max_p.y; y += 1.0f) {
  vec2 texcoord = texcoord_x + texcoord_y;                      // incremental DDA, no per-pixel cross-products
  for(float x = min_p.x; x <= max_p.x; x += 1.0f) {
    if(texcoord.x >= 0.0f && texcoord.y >= 0.0f && texcoord.x + texcoord.y <= 1.0f) {  // 3-test coverage
      float z = p10.z*texcoord.x + p20.z*texcoord.y + p0.z;     // barycentric z
      #if CLAY_MTL
        uint index = uint(surface_stride*y + x);
        atomicMax(out_surface[index], floatBitsToUint(z));       // Metal: BUFFER atomic
      #else
        imageAtomicMax(out_surface, ivec2(vec2(x,y)), floatBitsToUint(z));  // image atomic
      #endif
    }
    texcoord += texcoord_dx;
  }
  texcoord_y += texcoord_dy;
}
```
The Metal vs non-Metal storage split:
```glsl
#if CLAY_MTL
  layout(std430, binding = 5) buffer surface_buffer { uint out_surface[]; };   // buffer-from-texture
#else
  layout(binding = 0, set = 1, r32ui) uniform uimage2D out_surface;
#endif
```

So: **bbox-scanline brute force over the AABB** (NOT row-clipped to the triangle span), incremental barycentric (`texcoord += texcoord_dx`), 3-compare coverage, one barycentric `z`, single 32-bit `atomicMax` of `floatBitsToUint(z)`. That is the entire raster.

---

## The election: no-64-bit-atomics — what it confirms, where WE are ahead

This source is the clearest public statement of OUR central constraint, so it's worth being precise about what it does and does NOT give us.

- **What it does (the depth half, exactly ours):** `atomicMax(out_surface[index], floatBitsToUint(z))`. For an IEEE-754 float with `z >= 0`, the bit pattern is monotonic in the value, so `atomicMax` on the bit pattern = "keep the FARther z" (it's a depth-only Z-fight where larger-z wins; we invert so nearer wins, same trick). This is **byte-for-byte the principle of our depth key** (`bcF2U`/`depthKey24`, `NaniteRaster.ts:731,762`): convert float depth to an order-preserving uint and run ONE 32-bit atomic. The post proves this is the standard, performant M1/Metal answer — it validates our depth election is on the documented happy path, not a hack.
- **What it does NOT do — and what WE solved:** it carries **no payload/id at all**. Author's own words: *"we are limited to 32-bit payload data and must perform a redundant triangle intersection,"* and the explicit feature request:
  ```glsl
  uint imageAtomicPayloadMax(gimage2D atomic_image, gimage2D payload_image,
                             ivec2 P, uint atomic_data, gvec4 payload_data);  // "would change everything"
  ```
  i.e. they wanted a HW atomic that elects on depth AND writes a payload in one op (what 64-bit InterlockedMax gives a visbuffer). **Our 24b-depth|8b-tiebreak `atomicMax` + winner-plain-stores-25b-id-into-visBV (`NaniteRaster.ts:766-776`) is exactly the software emulation of their wished-for `imageAtomicPayloadMax`.** We are a step ahead of this source: it stops at depth-only because it lacked the trick; we built the trick. The "redundant triangle intersection" they mention as the fallback (re-rasterize in a 2nd pass to recover the winning triangle's attributes) is the FreePipe/2-pass family — strictly worse than our single-pass election (see extra-0 brief for why the 2-buffer color split is unsafe for an id payload).
- **CONFIRMS the constraint is real and engine-wide:** "64-bit atomics are not available on Mobile devices and Metal" from a vendor that ships a cross-API engine on M1 — independent corroboration that our no-64-bit design premise is correct, not a self-imposed limit.

**Net on election: zero new election idea — but the strongest possible confirmation** that (i) order-preserving-uint + single 32-bit `atomicMax` is THE depth answer on M1, and (ii) the payload-election we built is precisely the missing primitive this 2021 state-of-the-art could only wish for.

---

## Transferable techniques (ranked by value to us)

### 1. [DIRECT — config/storage, possibly already-satisfied] Metal "buffer-from-texture": elect into a `std430` storage buffer, not an `r32ui` image
The `#if CLAY_MTL` split exists because **Metal has no atomics on textures** — so on Metal you must back the visbuffer with a plain buffer and use buffer `atomicMax(out_surface[index], …)`, indexing `index = surface_stride*y + x`.
- **Mechanism / relevance:** in WebGPU there is no `imageAtomic*` at all — atomics are only legal on `atomic<u32>` in a `storage` buffer. So our visbuffer is **already forced down Tellusim's Metal path**: `visPayloadV`/`visBV`/`visDepthV` are storage buffers indexed by `px = y*W + x` (`NaniteRaster.ts:730,766-775`). This source **confirms our storage-buffer choice is the correct (indeed the only) Metal-class answer**, and that on M1 it is not a perf compromise — their 2.30 B tri/s is on exactly this buffer-atomic path. Honest: this is a confirmation, not a change. The only actionable nuance is `surface_stride` — they store an explicit row stride (`uint(surface_stride*y + x)`) rather than `y*width+x`; if our `W` (`cam.uW`) is ever a non-power-of-two that the driver would prefer padded, a padded stride can improve store locality. Low expected value; flag only.
- **No-64b adaptation:** n/a (this IS the 32-bit path).
- **effort:** none/verify-only.

### 2. [INDIRECT — micro-codegen] Fold the half-pixel sample bias and the 1/256 snap INTO the vertex once, never into the per-pixel math
Their snap line does `round(ndc.xy * size * 256)/256 - 0.5` **per vertex, once**, so the inner loop carries no sample-offset constant at all — coverage is the bare `texcoord.x>=0 && texcoord.y>=0 && x+y<=1` and depth is the bare barycentric. This is the same lesson extra-0 measured as a ~40% codegen cliff (sample-offset-in-vertex vs sample-offset-in-loop).
- **Mechanism / what it attacks:** our DOMINANT 60% per-pixel loop and per-triangle 40% — any constant that lives in the inner loop costs registers/ALU 36M×/frame. Tellusim's structure is the "right" shape: bias folded into the snapped vertex, only incremental adds inside.
- **WGSL adaptation:** we ALREADY snap to 1/256 with a top-left rule and carry incremental integer edge accumulators (`ex/ey = dE per +1 UNIT`, `NaniteRaster.ts:591-601,634`), and we compute `rcpArea`/edge coeffs once per triangle. So we already implement this principle. The audit to run: confirm the half-pixel sample bias is baked into the snapped screen-space vertex (like their `- 0.5f`) and NOT re-applied per pixel, and that nothing inside the `loopI('sx',…)` body (`:702-781`) is a per-triangle constant recomputed per fragment. Same-image, register-only.
- **No-64b adaptation:** orthogonal to the election.
- **effort:** low (audit + hoist); confirms a direction extra-0 already quantified.

### 3. [INDIRECT — confirms our wgcache, do NOT re-derive] Cooperative per-meshlet vertex transform in shared memory (transform each unique vertex ONCE)
Their `main()` transforms each meshlet vertex exactly once into `shared vec3 positions[NUM_VERTICES]` (all threads cooperate, `barrier()`), then per-triangle threads READ the 3 transformed corners from shared memory. No vertex is transformed more than once even though it's shared by many triangles.
- **Mechanism / what it attacks:** our 40% transform+launch — the "~5× redundant vertex re-fetch + ~48 wind texture taps per triangle" the mission calls out. This is the exact structural fix.
- **Status in OUR code — already shipped, partially:** `NaniteRaster.ts:241-248` (PERF-3) already broadcasts the per-cluster `makeCtx` (incl. the trunk-wind gust texture samples) via `workgroupArray` shared memory (the `wgcache` path, A/B-validated bit-identical), and `NaniteVertexCache.ts` exists for "the cooperative vertex-transform cache." So Tellusim **confirms** the architecture we already adopted to kill the 128×/cluster makeCtx redundancy. The remaining gap to check against Tellusim's cleaner model: do we transform each **unique vertex** once into shared `positions[]` (Tellusim) — or do we still call `fetchWorldVert` 3× per triangle and only share the *context*, leaving the 3× per-corner vertex fetch+transform in place? The mission's "5× redundant vertex re-fetch" wording suggests the per-corner fetch is still redundant. If so, the Tellusim/extra-0 pattern (phase A: lanes transform the cluster's UNIQUE verts to `var<workgroup>`; barrier; phase B: per-triangle lanes read 3 shared slots) is the unrealized win — see extra-0 #7, which quantifies this as the biggest transform-side lever. Tellusim is the second independent source showing this is the standard shape.
- **No-64b adaptation:** orthogonal.
- **WGSL adaptation:** `var<workgroup>` + `workgroupBarrier()` (we already use both). Needs the cluster's unique-vertex list (a meshlet build provides it) and a per-triangle local index→shared-slot map (their `indices[]` unpack does exactly this: 1-byte local indices into the shared `positions[]`).
- **effort:** medium (only the vertex half, if not already done; the context half is shipped).

### 4. [INDIRECT — measured non-win for US, confirms keep-it-cheap] Brute bbox scanline (NO row-clip / NO binning / NO hierarchy) is "fast enough" — but we already beat it
Tellusim's inner loop walks the FULL clamped AABB and only the 3-compare coverage test rejects out-of-triangle pixels — no per-row x-span, no tile binning, no hierarchical coverage. It's labelled "unoptimized" yet still wins HW by 1.68× on M1.
- **Relevance:** this is a **caution-confirmation**, the same signal as extra-0 #2. A vendor benchmarking M1 left the raster as bare AABB-scanline and still beat hardware — i.e. for tiny meshlet tris the per-fragment work is cheap enough that elaborate coverage machinery isn't what wins the micro-bench. BUT: that micro-bench has **no overdraw** (uniform meshlets, depth-only, no holey foliage). Our regime is 12–20× overdraw on holey leaf cards, where wasted fragments DO dominate — which is exactly why our shipped **scanline x-span** (commit 19eb834, `NaniteRaster.ts:669-707`, solve each edge for its crossing x, superset span, skip the ~half-AABB provably outside) is a real win that Tellusim's bench wouldn't have surfaced. So: Tellusim does NOT contradict our x-span; it just operates in a regime that doesn't reward it. Do not interpret "Tellusim left it as bare AABB" as license to revert our x-span.
- **DEAD-END flags it touches:** confirms (with extra-0) that there is **no tiled/binned/hierarchical-coverage trick hiding in this source** — it has none. If the mission's tiled-rasterization hope is to be satisfied, it will NOT come from here.
- **effort:** n/a (stop-digging signal).

### 5. [DIRECT but for THEM, not us — they LACK clip; we have it] Near-plane handling: they DROP, we route to HW
Tellusim's near-plane policy is `if(p0.z<0||p1.z<0||p2.z<0) return;` — they simply **discard** any triangle crossing the near plane (acceptable for a throughput bench; a visible quality bug in a real renderer). Our system instead routes near-crossing tris to the HW vertex-pulling path (`NaniteRaster.ts:519-522`, `nearOK`/HW_CAP), which clips correctly.
- **Relevance:** purely a confirmation that our near-plane routing is a *quality* feature this source skipped; nothing to port (we are ahead). Listed so the reader doesn't mistake their terse clip for a technique.

---

## Benchmark table (verbatim — 498 990 meshlets of 64 verts/128 tris, depth-only, back-face only)

| GPU | Single DIP | Mesh Shader | MDI/ICB/Loop | **Compute** | Compute vs best-HW |
|---|---|---|---|---|---|
| GeForce 2080 Ti | 12.05 B | 12.57 B | 12.63 B | **17.26 B** | 1.37× |
| GeForce 1060 M | 3.86 B | — | 3.90 B | **4.55 B** | 1.17× |
| Radeon 6700 XT | 14.73 B | 4.38 B | 3.63 B | **16.74 B** | 1.14× |
| Radeon RX 5600M | 4.87 B | — | 1.11 B | **7.57 B** | 1.55× |
| Radeon Vega 56 (macOS) | 2.40 B | — | 796 M | **3.17 B** | 1.32× |
| **Apple M1 (macOS)** | **1.37 B** | **739 M** | — | **2.30 B** | **1.68×** |
| Apple A14 (iOS) | 666.1 M | — | 475 M | **1.02 B** | 1.53× |
| Intel UHD | 680 M | — | 396.5 M | **556.1 M** | 0.82× (HW wins) |
| Adreno 660 (Android) | 565.2 M | — | 31.17 M | **497.3 M** | 0.88× (HW wins) |

Numbers are processed triangles/sec. Author conclusions: "Best Mesh Shaders / MDI are slower than Compute-based rasterization"; "MultiDrawIndirect doesn't work well on mobile because of the tile-based rendering"; "Single shader type is better than 14 dedicated shader types." **On M1, compute SW raster beats the best HW path 1.68× for tiny tris** — the empirical foundation of our SW-raster bet. (Caveat for our planning: this is a no-overdraw, depth-only, no-payload micro-bench; it sets an upper bound on raw raster throughput, NOT a model of our overdrawn/payload/holey-foliage workload.)

---

## Dead-end confirmations (flag, do not resurface)

- **Tiled / binned / coarse-to-fine / hierarchical coverage:** NONE present in this source. It is pure per-triangle AABB scanline. If we want binning prior art, this is not it.
- **Hardware early-Z / HW raster of micro-tris:** the whole post is evidence AGAINST relying on HW for sub-pixel tris (HW loses 1.68× on M1) — consistent with our "HW wastes ~4× on 2×2 quads" rationale and the existence of our SW path.
- **Occlusion / HZB:** none ("without any culling except back-face") — does not bear on our finding that conservative occlusion is ~0% on holey foliage; neither confirms nor refutes.
- **Atomic contention:** they run a single global `atomicMax`/pixel at billions of tris with no reported contention issue — consistent with our refutation that SW-atomic contention is not our bottleneck.
- **Sub-pixel LOD over-render:** n/a (no LOD in this bench).

## Bottom line for our 20→60fps goal

extra-3 is a **confirmation source, not a new-technique source**. Its real worth: (1) it pins down, from a vendor measuring M1/Metal directly, that **single 32-bit `atomicMax(order-preserving-uint(z))` is THE depth election on our hardware** — our depth key is on the documented happy path; (2) it states the no-64-bit-atomic limitation explicitly and frames the **`imageAtomicPayloadMax` we'd need** — which our 24b|8b `atomicMax` + winner-side-store already *emulates*, putting us a step ahead of this 2021 state of the art; (3) it confirms the **Metal buffer-backed visbuffer** (forced on us by WebGPU anyway) is the correct, performant atomic target on M1, not a compromise; (4) it independently corroborates the **cooperative per-meshlet vertex-transform-into-shared-memory** shape behind our `wgcache`/`NaniteVertexCache` — and, read against extra-0 #7, points at the one possibly-unrealized piece: transforming each cluster's UNIQUE vertices ONCE into `var<workgroup>` and having per-triangle lanes read the 3 shared corners, instead of re-fetching+transforming 3 corners per triangle (the "5× re-fetch + 48 wind taps" 40% cost). No coverage-loop, binning, or election idea here surpasses what we ship; the source's chief gift is high-confidence validation of our hardest design choices.
