# Prior-art brief — paraLLEl-GS (PS2 GS emulation in Vulkan compute)

- **slug:** extra-2 · **kind:** post
- **ref:** https://themaister.net/blog/2024/07/03/playstation-2-gs-emulation-the-final-frontier-of-vulkan-compute-emulation/
- **fetched:** `curl` of the page HTML → `docs/perf-runs/prior-art/sources/posts/parallel-gs-maister.html`
  (110 KB), then a local Python HTML→text strip → `…/posts/parallel-gs-maister.txt` (914 lines).
  WebFetch's small-model summarizer was used first but **hallucinated** struct fields and an
  "avoids atomics by … ordered subgroup ops" depth story that is NOT in the post; all quotes
  below are taken from the locally-saved primary text (section headings + line refs into the .txt).
  The post is by Hans-Kristian Arntzen ("Themaister", author of Granite / paraLLEl-RDP / Fossilize),
  describing **paraLLEl-GS**, a from-scratch compute-shader software rasterizer that emulates the
  PlayStation 2 Graphics Synthesizer.

---

## What it is, and why the architecture transfers (and where it does NOT)

paraLLEl-GS is a **tile-based deferred software rasterizer in Vulkan compute** ("Like paraLLEl-RDP,
paraLLEl-GS is a tile-based renderer" — §Triangle setup and binning, .txt:455). It rasterizes 2D/3D
primitives the PS2 GS emits, at high internal super-sample rates, with 100% bit-ish-accurate
blending. It is the most detailed public post-2020 description of a **compute-binned, subgroup-tiled
ubershader rasterizer**.

The single most important architectural fact — and the one that reframes the whole "no 64-bit
atomics" question for us — is that **paraLLEl-GS has NO depth atomics at all, and the post never even
discusses an atomicMax depth election.** Instead:

- The screen is cut into **NxN coarse blocks (usually 32×32)**, and **"One subgroup is responsible
  for iterating through all primitives in a block."** (§Binning, .txt:574).
- A block's **pixel state lives in registers** for the lifetime of the primitive loop — color,
  depth, coverage, and a *pending shade request* — and is only written to the SSBO **once, at tile
  completion**: *"Since we're an uber-shader, all pixels are 'on-chip', i.e. in registers"*
  (§Deferred on-tile shading, .txt:639); *"When everything is done, the resulting framebuffer color
  and depth is written out to SSBO. GPU bandwidth is kept to a minimum, just like a normal TBDR
  renderer."* (.txt:696).
- Because **one block is owned by exactly one subgroup and primitives are consumed in submission
  order inside that subgroup's loop, the depth test/write is just register reads/writes — serialized
  by construction, no atomic needed, and blend ordering is preserved for free.** This is the PS2's
  fundamental requirement (programmable blending, alpha test, dest-alpha) and it is *why* the loop is
  serialized per pixel.

That is the deep contrast with us. **Our** election is atomic because **multiple workgroups
(one-per-cluster) write the same screen pixel concurrently** — we have no per-pixel owner. paraLLEl-GS
sidesteps 64-bit atomics not with a clever pack but by **changing the work decomposition to
screen-tile ownership** so depth is plain register state. That is the big, structural lever this
source argues for; the small levers below are extractable without going all the way there.

**Honest caveat on the headline "overdraw" match:** PS2 fill-rate is high but the *primitive* counts
are tiny by our standards — *"between 10k and 30k primitives for the 'main' render pass"* (.txt:572),
capped at **64k total** by u16 indices. We push **~36M visible triangles**. paraLLEl-GS's binning
assumes a few-×10k-prim world where a 32×32 block's prim list is short; our per-tile lists would be
enormous. So the *binning data structure* is an indirect inspiration, not a drop-in. The *lazy
shading* and *scalarized resolve* ideas, however, transfer on principle regardless of scale.

---

## Raster architecture (the pipeline)

1. **Triangle setup + binning** (stateless, batched across passes — .txt:576). Inputs are three SoA
   arrays: `VertexPosition {ivec2 pos; float z;}` (.txt:460), `VertexAttribute {vec2 st; float q;
   uint rgba; float fog; u16vec2 uv;}` (.txt:471), `PrimitiveAttribute {i16vec4 bb; uint state; uint
   tex; uint tex2; uint alpha; uint fbmsk; uint fogcol;}` (.txt:484). Note the deliberate split:
   `state`/`tex` carry **scalarization keys** (texture state "which should be scalarized. Affects
   code paths", .txt:489) separate from `tex2` ("Does not affect code paths", .txt:493).
2. **Coarse binning:** every NxN (32×32) block gets a list of **u16 primitive indices** (.txt:572).
3. **Ubershader (the hot kernel):** one subgroup per block, **fine-binning inside the shader** to a
   subgroup-sized fine tile, then a serialized per-pixel coverage→lazy-shade loop, then a single
   write-out.

### Coverage + barycentric (.txt:501–524)
A standard **edge-function / barycentric** rasterizer ("heavily inspired by fgiesen … based on
*A Parallel Algorithm for Polygon Rasterization* (Pineda, 1988)"). Coverage is three integer dot
products plus a fixed-point reciprocal-area to recover I/J barycentrics:

```glsl
bool evaluate_coverage_single(PrimitiveSetup setup, bool parallelogram,
    ivec2 parallelogram_offset, ivec2 coord, inout float i, inout float j) {
  int a = idot3(setup.a, coord);
  int b = idot3(setup.b, coord);
  int c = idot3(setup.c, coord);
  precise float i_result = float(b) * setup.inv_area + setup.error_i;
  precise float j_result = float(c) * setup.inv_area + setup.error_j;
  ...
}
```
Edge precision is engineered to **fit 32-bit integer math** even over the GS [-4k,+4k] range by
"downsampling the edge equations" with `error_i/error_j` tie-break terms (.txt:524) — exactly the
trick our scanline core already uses (i32 edge terms < 2^26, 8 subpixel bits).

### Depth — early-Z with late-Z demotion (.txt:618–635)
Depth is **register state inside the subgroup**, not an atomic. Early-Z is used, but a *pending*
(not-yet-resolved) shade with a Z-write in flight forces a **late-Z demotion** so the in-flight
write can be observed:
```glsl
bool pending_z_write_can_affect_result =
    (pixel.request.z_test || !pixel.request.z_write) && pending_shade_request.z_write;
if (pending_z_write_can_affect_result) {
  pixel.opaque = false; // demote to late-Z; can't discard earlier pixels
}
```

---

## The transferable techniques

### DIRECT-ish ports

#### T1 — Lazy pixel shading: an opaque fragment KILLS the pending shade request (overdraw cull)
*"after rasterization, if a pixel is considered opaque, it will simply replace the shading request
that exists for that framebuffer coordinate. It won't be visible at all anyway."* (.txt:639) and
*"If our pixel remains opaque, we can just kill the pending pixel shade request."* (.txt:645):
```glsl
if (pixel.request.coverage > 0) {
  need_flush = !pixel.opaque && pending_shade_request.coverage > 0;
  if (!need_flush) {                                   // no hazard ⇒ overwrite
    set_pending_shade_request(pixel.request, shade_primitive_index);
    pixel.request.coverage = 0;
    pixel.request.z_write = false;
  }
}
```
This is the direct analog of our **12–20× overdraw on holey opaque foliage**: most covered fragments
are doomed. paraLLEl-GS only *records* a shade request per pixel and overwrites it when a nearer
opaque fragment lands; the actual texturing/shading runs at most once per pixel. **Mechanism:** decouple
*coverage+depth election* (cheap) from *shading* (expensive) and let later opaque winners erase
earlier pending shades before they ever shade. **Why it matters here:** we already split election from
shading (the resolve pass shades only the election winner per pixel), so the *over-shade* problem is
mostly already solved on our side. The residual it attacks for us is the **per-fragment election
traffic**, not shading — see the adaptation note.

#### T2 — Scalarized waterfall resolve (kill divergent branches across state)
The resolve loop forces **uniform control flow per subgroup** so all the branchy state (alpha-test
mode, blend mode, filter mode, texture index) is hit *scalar*, not per-lane-divergent (.txt:665–694):
```glsl
if (subgroupAny(need_flush)) { shade_resolve(); ... }   // resolve the whole subgroup at once
while (subgroupAny(has_work)) {                          // waterfall
  if (has_work) {
    uint state_index = subgroupBroadcastFirst(pending_shade_request.state);
    uint tex         = subgroupBroadcastFirst(prim_tex);
    if (state_index == pending_shade_request.state && prim_tex == tex) {
      has_work = false;
      shade_resolve(pending_primitive_index, state_index, tex);
    }
  }
}
```
*"This scalarization ensures that all branches on … alpha test mode, blend modes, etc, are purely
scalar, and GPUs like that."* (.txt:694). **Mechanism:** group lanes by a shared state key via
`subgroupBroadcastFirst` and iterate; each pass shades the lanes whose state matches the broadcast
leader, guaranteeing every state-dependent branch is subgroup-uniform.

#### T3 — Subgroup-ballot primitive compaction in the binning inner loop
Fine-binning pulls `SubgroupSize` primitives at once, ballots which ones hit the tile, then
**iterates only the set bits** via `subgroupBallotFindLSB` + a clear-lowest-bit trick + `subgroupShuffle`
to fetch the chosen primitive index (.txt:588–616):
```glsl
uvec4 work_ballot = subgroupBallot(binned_to_tile);
uint bit = subgroupBallotFindLSB(work_ballot);
... work_ballot.x &= work_ballot.x - 1; ...
shade_primitive_index = subgroupShuffle(bin_primitive_index, bit);
```
**Mechanism:** turn a divergent "is this lane's primitive in my tile?" into a dense iteration over
only the covering primitives — no idle lanes walking dead prims.

### INDIRECT (adapt-the-principle)

#### T4 — Two-level binning with subgroup-width-sized fine tiles (the work-distribution lever)
Coarse **32×32 block → one subgroup**, dynamically resized "depending on how heavy the geometry load
is" (.txt:574); fine tile **4×4 / 4×8 / 8×8 for subgroup size 16 / 32 / 64** (.txt:586). This is the
canonical *bin → coarse → fine* pipeline the mission asks about, and it is the structural alternative
to our **one-workgroup-per-cluster** decomposition. **Principle:** give each screen tile a single
owner subgroup so (a) per-pixel depth is register state (no atomics), (b) overdraw within a tile is
resolved locally and lazily, (c) triangle setup is amortized because a tile's covering prims are
streamed through shared edge math.

#### T5 — Pixel state on-chip; one SSBO write at tile completion (bandwidth/atomic-traffic lever)
*"all pixels are 'on-chip', i.e. in registers"* … *"the resulting framebuffer color and depth is
written out to SSBO"* only at the end (.txt:639,696). **Principle:** the per-fragment atomic/global
traffic that dominates us (a relaxed `atomicLoad` of the election word for *every* covered fragment)
becomes a *register* compare when one owner serializes the tile's pixels — global memory is touched
once per pixel, not once per fragment.

#### T6 — State split for scalarization, and "separate the rarely-touched attributes"
`PrimitiveAttribute` puts the **code-path-affecting** state (`tex`) apart from the non-affecting
`tex2` (.txt:489–494), and `TransformedAttributes` (full interpolants) is a **separate buffer "only
read if we actually end up shading … so it's important to keep this separate to avoid polluting
caches with too much garbage."** (.txt:528). **Principle:** keep the coverage/election hot loop
touching only the minimal per-prim words; defer the fat shading attributes (for us: the wind-anim
vertex re-fetch / 48 texture taps) to the resolve stage that runs once per surviving pixel.

---

## Adaptation to our no-64-bit-atomic WGSL system

**The honest framing:** paraLLEl-GS does NOT solve "depth|payload in one 32-bit atomicMax" — it
*avoids the question* by owning each screen tile with one subgroup, making depth plain register
state. Our current design (one-workgroup-per-cluster, concurrent writers, 24b-depth|8b-tiebreak
atomicMax election with a side-buffer id plain-store on the winner) is already a perfectly good
*no-64-bit-atomic* answer for a scatter-style rasterizer. The source's value is **not** a better
election; it's a **different decomposition** that would eliminate the per-fragment atomic traffic
entirely.

- **T1 lazy-shade — PARTIALLY ALREADY DONE; the residual is the per-fragment election compare, not
  shading.** Our resolve already shades only the per-pixel election winner, so we don't over-*shade*.
  What lazy-shade additionally buys (within a tile owner) is skipping the per-fragment *election*
  work for doomed fragments — but that only materializes under T4/T5 (tile ownership). As a
  standalone tweak it is **not portable**: with concurrent cluster workgroups there is no single
  "pending request" to kill. Flag: this is the one technique the brief in our mission ("lazy shading
  is a direct analog") most over-promises for us — we already reap its shading half.

- **T2 scalarized waterfall — INDIRECT, applies to the RESOLVE pass, not the raster.** Our hot loop
  is `nanRasterWorld1`, but the *shading* (material/wind branches) lives in `NaniteResolve`. If the
  resolve has divergent per-pixel material/branch selection, a `subgroupBroadcastFirst`-keyed
  waterfall over the cluster/material id would scalarize those branches. Portable: WGSL has
  `subgroupBroadcastFirst`, `subgroupBallot`, `subgroupAny` (behind the subgroups feature, available
  on Metal). Worth checking whether our resolve already runs material-uniform per workgroup; if so,
  zero gain. **Effort: medium. Attacks: shading divergence, not the 60% coverage loop directly.**

- **T3 ballot compaction — NOT APPLICABLE to our current decomposition.** It compacts a per-tile
  primitive-bin iteration; we have no fine-tile prim bin (we have one-cluster-per-workgroup, clusters
  are already well-filled at 235/256 tris). Only relevant if we adopt T4. Flag: our clusters are NOT
  idle-lane-bound, so even under T4 the win is small.

- **T4 two-level binning + tile-owner subgroups — THE BIG STRUCTURAL LEVER, but a major refactor and
  scale-mismatched.** This is the genuine attack on BOTH our costs: it removes per-fragment atomics
  (depth→registers) AND amortizes transform/setup (a tile streams its prims through shared edge
  math). BUT: (a) our **36M visible tris vs PS2's ≤64k** means per-32×32-tile prim lists are huge —
  a binning pass that lists every triangle per tile is itself an overdraw-sized write. A *cluster*-
  granular coarse bin (bin clusters, not triangles, to tiles) is the only tractable form, and
  clusters span many tiles. (b) Register pressure: holding color+depth+coverage+pending-request for
  a whole 32×32 tile (1024 px) in one subgroup is impossible; paraLLEl-GS holds only a fine tile
  (≤64 px) per subgroup and loops coarse-blocks of fine-tiles. (c) It only removes atomics if a tile
  has a single owner — which forbids the cross-cluster concurrency we rely on. **Verdict: adapt the
  PRINCIPLE (screen-tiled, register-resident depth, lazy local overdraw resolution) only if we are
  willing to rebuild the raster as a tile-binned compute pipeline. This is a multi-week refactor with
  real risk, not a tweak. It is, however, exactly the "tiled bin→coarse→fine that minimizes wasted
  per-fragment coverage work" the mission asked to find, and it is the only thing here that plausibly
  reaches the 2× target at the same image. Effort: very high. Attacks: BOTH 60% coverage loop (atomic
  traffic→registers) AND 40% transform/setup (per-tile amortization). Quality-preserving: yes (it's a
  reorganization, not an approximation) — modulo getting watertight edges + the depth tie-break bit-
  identical, which our scanline core already nails.**

- **T5 on-chip pixel state — same as T4, it IS the payoff of T4.** Not separable. Under our current
  scatter raster it is not portable (no per-pixel owner).

- **T6 hot/cold attribute split — DIRECT, low-risk, AND aimed at our 40% transform cost.** Our
  per-triangle path does `3× fetchWorldVert` with ~5× redundant re-fetch and ~48 wind-animation
  texture taps *inside the raster kernel*. paraLLEl-GS's discipline — keep `TransformedAttributes`
  (the fat interpolants) in a separate buffer *"only read if we actually end up shading"* — argues
  for moving the wind-animation / fat vertex work OUT of `nanRasterWorld1` into a **pre-transform
  pass keyed by cluster** (compute the animated world verts once per unique vertex, write to a
  buffer, then the raster kernel just reads positions). This kills the ~5× redundant re-fetch and the
  in-raster texture taps. **Caveat:** check whether a per-cluster pre-transform already exists; the
  redundancy note suggests it does not. **Effort: medium. Attacks: the 40% transform/setup directly.
  Quality-preserving: yes (identical math, computed once).** This is the highest-value *small*
  takeaway from this source.

### Dead-end confirmations (source touches, does NOT revive)
- **No occlusion-culling claim** — paraLLEl-GS relies on its lazy-shade overwrite, not HZB occlusion;
  consistent with our finding that conservative HZB occlusion is ~0% on holey foliage. (No quote
  contradicts our refutation.)
- **No sub-pixel-LOD / micro-poly story relevant to us** — PS2 prims are *not* sub-pixel; the
  source's micro-precision work (UV epsilon snapping, .txt:544–567) is about NEAREST-filter texel
  alignment, irrelevant to our 1px-triangle overdraw.
- **Atomic-contention is a non-topic** — paraLLEl-GS has no depth atomics; it neither confirms nor
  refutes our (already-refuted) atomic-contention thesis, it simply shows the alternative
  decomposition that has no atomics to contend on.

---

## Bottom line for our 20→60fps hunt
1. **T6 (hot/cold attribute split → move wind-anim/vertex work to a per-cluster pre-transform pass)**
   — best *small* win available here, directly cuts the 40% transform/setup, bit-identical.
2. **T4/T5 (screen-tiled, register-resident, lazy-overdraw raster)** — the only idea here that
   credibly attacks the dominant 60% per-fragment coverage/election loop at the same image, by
   replacing per-fragment global atomics with per-pixel register state. But it is a ground-up raster
   refactor, scale-mismatched to 36M tris (needs cluster-granular binning + fine-tile register
   budgeting), and only pays if we accept single-owner tiles. High risk, high ceiling.
3. **T2 (scalarized waterfall resolve)** — opportunistic, only if our resolve has divergent material
   branches.
4. Lazy-shade's *shading* half (T1) we already have; do not double-count it.
