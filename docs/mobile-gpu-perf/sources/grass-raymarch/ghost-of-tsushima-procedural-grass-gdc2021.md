# Procedural Grass in Ghost of Tsushima (GDC 2021, Eric Wohllaib / Sucker Punch)

Source: https://archive.thedatadungeon.com/ghost_of_tsushima_2020/documents/gdc_2021/gdc_2021_procedural_grass_in_got.pdf
Presenter: @ericwoh (Sucker Punch). Reference cited: Outerra 2012 procedural grass blog.

## THE LOAD-BEARING FACT for our arc
GoT renders grass as **GPU-generated real geometry per blade** (compute → indirect draw of
tri-strip blades), **NOT a per-pixel raymarch**. The whole field of ~millions of blades is built
and culled in two compute passes, then drawn as instanced Bezier-curve tri-strips. This is the
opposite lane from our kGrassRay per-pixel analytic march. On console this hits their frame budget
(the widely-cited ~1M blades in ~a couple ms is this pipeline: cheap per-blade compute + tiny
7–15-vertex strips + aggressive culling, so cost scales with VISIBLE blades not screen pixels).

## Overview (5 sections)
Compute shader · Data pipeline · Vertex shader · Pixel shader · Miscellaneous.

## Compute shader — PLACING a grass blade
Grass lives on a **tile grid**; each compute thread ("lane") owns one potential blade slot.
Per lane:
- Turn **lane ID → position on tile grid, + jitter** (so it's not a regular lattice).
- **Distance culling and frustum culling.**
- **Determine grass type and height from a texture at the position** (single texture read drives
  type/height — density/appearance is texture-authored, not per-instance data).
- **Drop lanes that don't have grass** (density/coverage cull — empty slots cost nothing downstream).
- **Occlusion culling** (HZB-style — don't emit hidden blades).

Only surviving lanes append a blade to the instance-data buffer. This is the key perf property:
**work compacts to visible, on-screen, non-occluded blades before any vertex work happens.**

## Compute shader — GENERATING a grass blade (per-blade instance data)
Stored per surviving blade:
- Position (3 floats)
- Facing (2 floats)
- Wind strength at position
- Per-blade Hash (drives all procedural variation)
- Grass Type
- Clump facing (2 floats), Clump color   ← blades belong to CLUMPS for coherent facing/color
- Height, Width, Tilt, Bend, Side Curve

**Clumping:** each sample point is assigned to a nearby clump center (Voronoi-ish; the deck shows
sample point → nearest clump-center vectors on a jittered grid). Clump facing + clump color give
the grass its natural tufted, art-directable look instead of uniform noise.

## Data pipeline (2 compute dispatches + indirect draw)
```
compute 1  ──> blade count ──> compute 2 ──> indirect draw args
    │                                              │
    └────────> instance data ───────────────────> vertex ──> pixel
```
- **compute 1**: place + cull, writes surviving blades to instance-data buffer and a blade count.
- **compute 2**: reads blade count, writes **indirect draw args**.
- **vertex/pixel**: indirect-drawn instanced blades read instance data.
- Multiple instance-data buffers are double/triple-buffered so compute1/compute2/vertex/pixel of
  successive tiles/frames **overlap on the GPU** (pipelining diagram) — hides latency.

## Vertex shader — the blade geometry
- **Two LODs: High = 15 vertices, Low = 7 vertices** (a blade is a thin tapered tri-strip).
- LOD picked per-blade; deck shows the 4-subtile→1 merge that halves vertex density with distance.
- **Blade shape = a cubic Bezier curve** up its length:
  - Position along blade easy to evaluate; **derivative easy to evaluate → gives the normal.**
  - Moving control points changes blade shape — used for **animation (wind)** and **appearance variety**.
  - The **midpoint offset is "controlled by bend"** (single bend param bows the blade).
- Per vertex: evaluate Bezier for position; get normal orthogonal to facing; **step vertex in the
  width direction**; evaluate Bezier-derivative curve for the tangent/normal.
- **Rounded vs flat normals:** they bend the across-width normal so a flat 2-tri blade shades like a
  rounded cylinder (cheap normal trick, big lighting win — no extra geometry).

## Pixel shader (cheap — outputs to G-buffer)
- Output material data to G-buffers (deferred).
- **Gloss:** 1D texture (along blade length).
- **Diffuse:** two textures — a **1D texture for the vein** + a **2D texture for color and alternate
  colors** (clump color tints here).
- **Translucency:** constant value.
- **AO:** constant value.
So the pixel/material cost per blade is tiny: 1D + small 2D fetch, constants for the rest.

## Miscellaneous / results
- Handles distinct grass types (green field, blue/rice, white susuki/pampas plumes, ferns) all from
  the same pipeline via the type texture + clump params.
- Wind = animate Bezier control points by per-position wind strength.
- Shadow: blades cast/receive; deck shows depth/shadow renders of the field.

## Future work
- Better LOD-transition (the 8→4 sample merge, reduce popping).

---

## MAP TO OUR kGrassRay (the strategic question)
Our grass is **+14.5ms per-pixel analytic raycast** (screen-scaled). GoT is the **per-blade
generate** lane:
1. **Cost model is inverted.** GoT cost ∝ visible blades (compacted by 4 cull stages BEFORE vertex
   work); ours ∝ covered pixels × march steps. On a filled screen ours pays for every pixel every
   frame regardless of how little grass detail survives. **This is the "wrong lane for mobile" signal**
   — a TBDR mobile GPU hates long dependent per-pixel loops far more than it hates many tiny cheap
   tri-strips with early cull.
2. **Culling happens up front, once, in compute** — not per-pixel per-march-step. Port: a compute
   place+cull pass emitting an indirect blade draw would move grass off the per-pixel loop entirely.
3. **Bezier blade = 7–15 verts, analytic normal from derivative** — no raymarch, no dependent
   texture fetches in the inner loop. Material is 1D + tiny 2D fetch.
4. **Clumping (Voronoi clump centers → clump facing/color)** is how they get lushness cheaply — the
   look our GrassLushnessLaw demands, without per-blade authoring.
5. **Texture-driven density/type/height** = one fetch to place, then drop-empty-lanes. Empty ground
   costs ~nothing; our march pays for empty ground too.

**Verdict seed:** GoT is strong evidence the per-blade-generate + indirect-draw lane is the
mobile-correct one, and our per-pixel march is the lane to reconsider — OR to make screen-constant
(Babylon corridor/analytic) + step-reduced (Cloudscapes jitter+TAA) if we stay in raymarch.
