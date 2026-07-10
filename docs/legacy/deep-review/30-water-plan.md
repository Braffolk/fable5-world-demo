# 30 — Water in the nanite path (design + budget + integration plan)

Scope: WATER as a first-class citizen of the software-Nanite pipeline (nanite-raster @ 6a93dfb) at
zero quality sacrifice and inside the locked-60 whole-frame law (underlying live p95 ≤ ~15.5 ms per
`90-premise-audit.md` §2.2). This is an artifact doc for later evaluation — design, budget and
integration plan; no code is written by this doc. Every claim about our code carries `file:line`;
prior art carries sources. Companion area docs: `11-cull-and-hzb.md`, `16-resolve-and-post.md`,
`90-premise-audit.md`.

## 0. Premise audit (go up a level before designing)

The reflex framing — "water must join the vis-buffer election like everything else" — is the wrong
level, and both our engine and UE5 prove it:

- **The election is structurally single-layer opaque.** One 32-bit word per pixel,
  `depthKey24<<8 | id8` guarded atomicMax (NaniteRaster.ts:335-337, 952-975), one winner, one depth.
  Water is a *transmissive* surface: it needs the fully shaded scene **behind** it, which only
  exists after the resolve runs. Electing water one level down fights the architecture instead of
  using it.
- **UE5 reached the same conclusion.** Nanite rasters opaque/masked; water is Single Layer Water —
  **[VERIFIED-FIX]** exact doc wording: "The fully lit scene and depth are used as input to the
  Single Layer Water pass", which "runs after the base pass and deferred lighting, but before
  regular translucency rendering" ([Epic SLW docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/single-layer-water-shading-model-in-unreal-engine), fetched 2026-07-02).
  Where UE5 does raster translucent Nanite clusters it forces them into HW-only bins — our local UE
  source copy states it verbatim: "this transcode only operates on Nanite translucent clusters,
  which only support HW raster (currently)" and "software is currently unsupported for translucent
  bins" (docs/perf-runs/Nanite-UE5-shaders/NaniteTranslucency.usf:22-23, :33 **[VERIFIED-FIX]**
  line numbers corrected against the local file). UE's own water mesh is
  not Nanite at all; SLW and Nanite interact only through depth.
- **The biggest hidden premise: this system is not greenfield.** The repo already ships a complete,
  tuned single-layer water tier (clipmap sheet + SLW-shaped material + analytic caustics,
  `src/world/WaterSurface.ts`, `src/render/WaterMaterial.ts`, `src/render/Caustics.ts`) that reads
  only screen depth, screen color and resident hydrology buffers — i.e. it is *already
  deferred-compatible by construction*. It is excluded from the nanite slate by a single boot gate
  (`DISABLE_OLD_GEOMETRY`, TerrainScene.ts:64, water gated at :180), not by any architectural
  incompatibility. The nanite raster's sub-pixel depth-bias bug was in fact *found by this water
  depth-testing against the vis-buffer depth* at N4-C0 (NaniteRaster.ts:884-891) — the depth interop
  has already worked once.
- **Scene split caveat.** The canonical perf scene (forest, 200k trees) has no hydrology and no
  water; water lives in the world scene, which also carries shadows/GI/clouds the forest scene lacks
  (`16-resolve-and-post.md` P2, Consequence C). Water's budget is therefore a line item in the
  world-parity reserve, not a subtraction from the forest pools.

Conclusion: the task is not "build water for nanite" — it is (a) re-admit the shipped tier over the
nanite resolve output, (b) port its two three.js dependencies (shadow receive, foam lighting) to the
nanite equivalents, (c) bound its cost with the UE SLW cost machinery, and (d) extend it upward
(lake/ocean waves) on the same skeleton.

## 1. TL;DR + chosen architecture

**Chosen: "SLW-over-resolve" — the shipped forward water layer composited on the nanite resolve
output, plus the UE Single Layer Water cost machinery, plus an FFT wave tier for large water.**

Frame shape (all existing slots, nothing reordered): cull → SW/HW/vox raster → HZB → shadow clipmap
→ half-res shadow → scene pass [resolve tri (−1000, NaniteResolve.ts:1085-1090) → resolve vox
(−999, :1097-1105) → **water sheets (transparent, drawn last by three's sort)**] → post chain
(NaniteFrame.ts:440-496, PostStack.ts:133-165). The resolve writes real depth via `depthNode`
(NaniteResolve.ts:1071-1078), so the water material's `viewportDepthTexture` reads nanite geometry
and its `viewportSharedTexture` reads the fully-lit opaque frame — byte-for-byte the SLW contract.

Three tiers on one skeleton (§2): **puddle/wetness** inside the resolve terrain branch (already
half-built: wet fringe + biofilm + caustic tint, NaniteResolve.ts:458-468), **river/lake** = the
shipped clipmap tier, **ocean/large-lake** = staggered 256² FFT cascades displacing the same clipmap
verts, whose Jacobian unifies foam + caustics with the surface.

**Runners-up and why not (as the ship vehicle — none are "impossible", each is staged or scoped):**

| Candidate | What it is | Why not first |
|---|---|---|
| **B — water election channel in the vis-buffer** | Raster the sheet in the SW raster into a small water-only atomicMax channel; shade in a second mini-resolve after the opaque resolve (mirrors UE pass order). Kills the transparent-pass overdraw + snapshot. **[VERIFIED-FIX]** The water channel must stay OUT of the HZB source: the HZB feeds next-frame occlusion cull (NaniteFrame.ts:481 "this frame's depth → next frame's occluder"), and letting water depth occlude would cull exactly the underwater/behind-surface geometry the refraction and SSR reads need shaded — the HZB keeps building from the opaque election (visPayloadV) only. B's wins are snapshot-delete + overdraw kill, not cull feedback. | Refraction needs post-resolve scene color, so it needs its own small resolve pass *anyway*; id headroom is the vox namespace bits 28-30 only (payload budget: NaniteRaster.ts:9-13 — 1 spare bit; NaniteVoxelRaster.ts:80-85, :264). Correct order: measure A's residual forward cost first; B is the engine-native endgame (staged as W7, §6) if that residual is ≥ ~0.5 ms. |
| **UE translucent-bin transplant** (HW raster VS/PS over cluster data) | UE's general Nanite translucency path (NaniteTranslucency.usf). | Solves arbitrary *sorted* translucency, which single-layer water does not need — **[VERIFIED-FIX]** UE's own water doesn't use it either: SLW is its own pass "after the base pass and deferred lighting" per the [Epic SLW docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/single-layer-water-shading-model-in-unreal-engine) (the previously-cited Vermilion post covers wet-material water tables, not SLW pass structure — citation moved to Tier P where it belongs). |
| **Planar reflection pass** | Mirrored re-run of cull+raster+resolve for hero lakes (slot reserved, WaterMaterial.ts:12-13). | **[VERIFIED-FIX]** Canonically heavy by mechanism, not by a borrowed number: it is a second scene render from the mirrored view — in OUR pipeline a second cull+SW/HW raster+resolve run, i.e. a second base-raster class cost (~10 ms at the current base, 90-premise-audit §2.4) even before reduced-res levers ([Epic planar reflections](https://dev.epicgames.com/documentation/unreal-engine/planar-reflections-in-unreal-engine); cost-tradeoff discussion: [gamedev.net thread](https://gamedev.net/forums/topic/703353-screen-space-reflections-faster-than-projective-reflections/5410738/) — the "+23 ms" figure previously quoted could not be re-verified and is withdrawn). The quality bar is met without it (§5.1); it stays a ladder rung for an art-directed hero shot at aggressive errorK, entered only with a measured budget. |
| **OIT / depth peeling** | General transparency machinery. | The clipmap guarantees exactly one water layer per pixel (inner-rect cutouts, WaterMaterial.ts:113-124) — the whole point of SLW is that this property makes blending machinery unnecessary. |

Existence proofs at our hardware class: UE5 SLW ships on everything down to mobile scalability
("an alternative render path... skips some of the features of Single Layer Water in favor of
performance", [Epic SLW docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/single-layer-water-shading-model-in-unreal-engine));
**[VERIFIED-FIX]** Sea of Thieves ran full FFT ocean ("The Technical Art of Sea of Thieves",
[SIGGRAPH 2018 Talks, DOI 10.1145/3214745.3214820](https://dl.acm.org/doi/10.1145/3214745.3214820))
and officially supports 540p/30 play on Intel Iris integrated laptop GPUs — the "Cursed" tier of
its published spec ladder ([Rare PC system requirements](https://support.seaofthieves.com/articles/360000340928-PC-System-Requirements)); Enshrouded ships dynamically
simulated water over voxel terrain in a custom engine on low-end HW, with the same factoring we
have — sim field → extracted surface sheet → single-layer shade ([GPC 2025 talk, Mantler & Koenen](https://graphicsprogrammingconference.com/archive/2025/),
[Wake of the Water update](https://enshrouded.com/en-US/news/enshrouded-wake-of-the-water-update)).

## 2. Tiered design — puddle / river / lake / ocean

All tiers share: the hydrology fields (`waterY` f32 sim-res buffer + min-reduced `waterYFar`,
Heightfield.ts:68-73, :164-171; `flow.flowDir`), one clipmap geometry + material family (6 levels,
cells 1.5→48 m, outermost ±3.07 km, one shared grid geometry, per-level uniforms only,
WaterSurface.ts:27-28, :79-98), one absorption model (per-channel SIGMA, WaterMaterial.ts:75), one
caustic bake (512² compute, ~0.05 ms, Caustics.ts:10, :103-174), one debug ladder
(`?waterdbg=1..8` incl. depth forensics, WaterMaterial.ts:341-383).

### Tier P — puddles / surface wetness (resolve-native, no sheet)
Puddles are sub-texel for the hydrology sim grid; they belong to the *material*, not the mesh —
exactly UE's answer for wet Nanite geometry (material-level water tables, [Vermilion](https://blog.rime.red/unreal-engine-5-nanite-and-water/)).
Mechanism: extend the resolve terrain branch's existing wet-fringe/biofilm/caustic block
(NaniteResolve.ts:458-468) with a wetness mask (rain state × concavity from the fields texture) that
darkens albedo and adds a flattened-normal fresnel sky-LUT term. Cost class: ALU-only in the tri
pass — the measured-free regime, obeying the register rule that texture-heavy subgraphs must stay
build-time-guarded per pass (NaniteResolve.ts:439-446). **Cost ≈ 0; no new pass, no new binding.**

### Tier R — rivers / streams (the shipped tier, unchanged math)
- Sheet: clipmap verts sample `sampleWaterY` (near) / `sampleWaterYFar` (levels with cell ≥ 12 m,
  WaterSurface.ts:86, WaterMaterial.ts:109-111; Heightfield.ts:397-411). Dry cells sit ~2 m under
  the bed and lose the z-test (WaterSurface.ts:11-14) — dry area costs vertex work + early-z only.
- Ripples: two-phase flowmap over pre-derived fbm gradients — one fetch per layer
  (WaterMaterial.ts:133-152); flow speed from the bilinear flow field (:126-131), breeze fallback in
  still water (:139).
- Refraction: ripple-driven UV shrinking with distance, depth-validated against leaks (:161-178);
  Beer–Lambert per-channel absorption over ray thickness + sky-tracked turbidity in-scatter
  (:179-193).
- Reflection: 18-step IGN-jittered SSR against scene depth — **[VERIFIED-FIX]** the 28 m clamp is
  the per-STEP length (`stepLen = clamp(dist·0.09, 0.25, 28)`, WaterMaterial.ts:206), so the march
  reaches ~500 m at grazing lake distances while cost stays bounded by the fixed 18 taps — plus
  screen-edge fade, and a
  **crowned-horizon geometric fallback** — 4 log-spaced heightfield taps raised by the canopy map,
  probe-GI radiance when occluded, sky LUT when clear (:200-269). Fresnel on the flattened normal
  (:271-277).
- Foam: shore + rapids-keyed-on-drop, variance-renormalized two-phase advection (:279-309);
  shoreline feather + dive-ramp fade (:321-334).

### Tier L — lakes (same sheet, upgraded reflection rung + optional wind waves)
Lakes are the flow≈0 case of Tier R (breeze ripples). Two upgrades, both cost-gated:
1. **Reflection proxy upgrade** (unique to us): the crowned-horizon march already gives a geometric
   reflection proxy at 4 taps (WaterMaterial.ts:248-257) — extend with 2–3 more ranges and a
   far-tile/impostor color sample so grazing lake reflections carry tree-line *color*, not just
   occlusion. Stability better than SSR (no screen-space misses), cost a few taps — a mid rung
   between SSR and planar that UE's ladder does not have.
2. **Wind-wave normals** from Tier O's FFT (normal-only, no displacement) when the lake is large
   enough to read as flat.
Known far-rim artifact of the min-reduced field on large lakes is pre-diagnosed with alternatives
recorded (Heightfield.ts:371-381) — owned by the per-water-body far field / hero-lake ladder rung,
not by this plan's critical path.

### Tier O — ocean / very large water (Candidate C)
- **Sim**: 256² FFT, 2–4 cascades at golden-ratio spacings, TMA/JONSWAP spectrum, staggered **one
  cascade update per frame** (GodotOceanWaves policy, [repo](https://github.com/2Retr0/GodotOceanWaves));
  outputs displacement + derivatives + Jacobian. Sizing anchors: **[VERIFIED-FIX]** NVIDIA's DX11
  compute-FFT ocean slides state "multiple 512×512 transforms can be performed in trivial time on
  high-end GPUs" and "FFT takes trivial time to complete on most GPUs" — 2010-era discrete HW
  ([NVIDIA OceanCS](https://developer.download.nvidia.com/assets/gamedev/files/sdk/11/OceanCS_Slides.pdf), quotes re-verified from the PDF 2026-07-02);
  practitioner sweet spot 256²/512², up to 4 cascades ([R. Ryan, Ocean Rendering Pt 1](https://rtryan98.github.io/2025/10/04/ocean-rendering-part-1.html));
  Gerstner (8–16 waves, vertex-stage) is the zero-new-pass fallback and is cheaper at low wave
  counts ([BTH thesis, FFT vs Gerstner](https://bth.diva-portal.org/smash/get/diva2:1778248/FULLTEXT02.pdf)).
  **Estimate ~0.1–0.3 ms amortized on M1 Max — labeled estimate, probe-gated (§6 W6).**
- **Mesh**: zero new geometry — displacement applies to the existing clipmap verts near, normal-only
  on mid levels, normals-only far skirt. This mirrors UE's quadtree-LOD + separate cheap
  "far distance mesh" ([Epic water meshing](https://dev.epicgames.com/documentation/en-us/unreal-engine/water-meshing-system-and-surface-rendering-in-unreal-engine))
  with our clipmap playing the quadtree's role (equal cost shape: concentric density halving).
- **Coherence win**: the FFT Jacobian replaces both the foam pattern and the caustic bake's wave
  source, so surface, foam and caustics share one wave field. Our caustic bake already computes
  exactly the inverse-Jacobian quantity from an analytic lattice (Caustics.ts:4-11, kernel
  :134-166). Doc correction while here: the header says "7 integer-lattice sine waves"
  (Caustics.ts:5) but the WAVES table holds 9 entries (:83-93).
- Rivers keep hydrology-flow ripples untouched — cheaper than any spectrum and flow-aware.

## 3. Integration into OUR pipeline (file:line anchors)

### 3.1 Vis-buffer / resolve seam
Water never enters the election. It is added to `engine.scene` exactly like the resolve meshes are
(NaniteFrame.ts:288, :294) and renders inside the same scene pass (`pass(scene, camera)`,
PostStack.ts:133) *after* both resolve triangles, because three sorts transparents after opaques and
the water material is `transparent = true, depthWrite = true` (WaterMaterial.ts:103-105). Inputs:
- `viewportDepthTexture` = scene-pass depth = the resolve's reconstructed 24-bit-key depth
  (NaniteResolve.ts:1071-1078). The water's depth epsilons (leak test +0.02 m, WaterMaterial.ts:168;
  SSR hit window :227) were tuned against this exact buffer family — the N4-C0 depth-bias fix was
  driven by water z-testing this buffer (NaniteRaster.ts:884-891).
- `viewportSharedTexture` = three's scene-color snapshot of everything drawn so far = the fully lit
  opaque nanite frame (refraction :184, SSR hit color :267).
The only code change for bring-up is the boot gate: WaterSurface is currently constructed only under
`?oldgeo=1` (TerrainScene.ts:64, :179-188); in the nanite slate it must be constructed
unconditionally (minus `?ablate=water`). The caustics context already IS active in the nanite slate
(TerrainScene.ts:124-131) and already feeds the resolve terrain branch (NaniteResolve.ts:458-468).

**Binding budget**: the water material is its own render pipeline — its fragment storage buffers
(waterY, waterYFar, flowDir, height, GI probes ≈ 5) do not count against the resolve passes' Metal
≤10 ceiling (tri pass at 8, vox at 7, NaniteResolve.ts:305-319). Water adds **zero** bindings to any
existing pass.

### 3.2 Shadow system (the one real port)
The legacy material receives sun via three's CSM, whose maps are empty in the nanite slate (all
casters go through the nanite shadow path; the resolve documents keep≡1 against empty maps,
NaniteResolve.ts:277-291). Port: multiply the water's sun-lit terms — foam diffuse (colorNode,
WaterMaterial.ts:312) and the PBR sun glint — by the nanite shadow factor, exactly the expressions
the resolve already evaluates per pixel: `world.shadowHalf.upsample(wp, camDist)` or
`world.naniteShadow.shadowFactor(wp, n)` (NaniteResolve.ts:922-955, esp. :933-935). Both are plain
TSL over storage textures, usable in any node material. The shadow clipmap itself needs zero
changes: water casts nothing (`castShadow = false`, WaterSurface.ts:90) and the clipmap's hollow
rings / min-px cull / VP-equality caching are receiver-agnostic (NaniteShadowClip.ts:1-36, :116,
:136, :223). The emissive reflection path already bypasses three lighting (sky LUT + probe GI,
WaterMaterial.ts:236-263), so after this port the only three-lit term left is foam — either keep the
scene's directional light for it (cheap, correct: sun exists in the world scene) or fold it into the
resolve's manual sun+ambient formula (NaniteResolve.ts:899-1002) for exact parity.

### 3.3 TRAA
- Water writes depth (WaterMaterial.ts:105) ⇒ TRAA's analytic camera reprojection — which is
  computed from each pixel's own depth and injected through the `velocityNode.load()` seam
  (PostStack.ts:520-536) — is exact for the surface under camera motion, including translation
  parallax. Content refracted through the surface reprojects at the water's depth; the refraction
  offset shrinks with distance (WaterMaterial.ts:161) so the error is bounded and sub-pixel far.
- Ripple/foam self-motion falls to variance clipping by documented contract ("Object self-motion
  (wind sway, water) isn't captured and falls back to variance clipping", PostStack.ts:509-510).
  Upgrade path (W4): a water branch in `velReproject` adding flow-field NDC velocity — `flowDir` is
  a resident buffer and the seam is exactly one function (PostStack.ts:520-533). **[VERIFIED-FIX]**
  Concrete water-pixel classifier (previously unstated): `velReproject` already reconstructs the
  pixel's world position from its own depth (`posW`, PostStack.ts:527); a pixel is water when
  `|posW.y − sampleWaterY(posW.xz)| < ε` (ε ≈ 0.15 m, both `waterY` and `flowDir` are resident sim-res
  buffers, Heightfield.ts:68, :397-401) — two extra storage-buffer reads in the post pipeline, which
  has its own binding budget separate from the resolve's ≤10 ceiling. This is *real*
  velocity, not UE-style "responsive AA" history nuking — no quality trade.
- SSR and TRAA are co-designed: the march is IGN-jittered (WaterMaterial.ts:207) and converges under
  TRAA accumulation, same policy as UE 5.6's downsampled+denoised water SSR
  ([Tom Looman, UE 5.6 highlights](https://tomlooman.com/unreal-engine-5-6-performance-highlights/)).
  Invariant to keep: never zero TRAA history on water pixels.
- The nanite raster mirrors the TRAA jitter (`traaNode._jitterIndex`, PostStack.ts:60-62, :540-543)
  and water renders with the same jittered camera in the same pass ⇒ no water-vs-geometry jitter
  mismatch by construction.

### 3.4 Post stack
- **GTAO**: from-depth, half-res, 1.6 m radius (PostStack.ts:192-207). Open water is near-planar ⇒
  AO≈1; shoreline creases darken mildly (plausible). The bilateral upsample's depth-gap weighting
  (PostStack.ts:400, gated fallback :405-411) already rejects cross-surface taps — no mask needed.
- **Contact shadows**: SSCS marches scene depth toward the sun within 240 m (PostStack.ts:424-470).
  Once water depth is in the buffer, reeds/banks/rocks gain contact darkening on the water for free.
- **Aerial/froxels/clouds**: applied from depth after the scene pass (PostStack.ts:264-285); water
  pixels carry water depth ⇒ haze integrates over the correct distance; the cloud near-solid
  upsample gate works unchanged (PostStack.ts:338-350).

### 3.5 Terrain / streaming
Zero coupling to TerrainClipmap/TerrainStreamer: the water clipmap is its own camera-snapped 6-sheet
grid (WaterSurface.ts:101-113) over resident hydrology buffers — nothing streams. Shorelines emerge
from the bilinear wet/dry crossing + material feather (WaterSurface.ts:12-14,
WaterMaterial.ts:321-334). Beyond the streamed terrain rings the resolve depth is coarse — the
min-reduced far water field is designed for exactly that regime (Heightfield.ts:357-394).

### 3.6 Underwater
Camera-submerged state is already computable CPU-side (`hf.waterYAtCpu`, Heightfield.ts:214-216;
rig probe TerrainScene.ts:442-446). Underwater = post-pool work, built only when submerged:
(a) full-screen distance fog with the SAME SIGMA constants (WaterMaterial.ts:75); (b) waterline mask
from the sheet's depth for half-submerged frames; (c) submerged terrain/rocks already carry caustics
through the standard lit path (Caustics.ts:13-17, applyCaustics :276-283; nanite terrain branch
NaniteResolve.ts:458-468) — **[VERIFIED-FIX]** at parity with UE's approach, where underwater is
likewise a camera post-process ("Water Bodies include their own underwater post-processing that
happens when the camera moves below the water's surface",
[Epic water bodies](https://dev.epicgames.com/documentation/en-us/unreal-engine/water-body-actors-in-unreal-engine));
(d) Snell's window: above the critical angle, reflect into the sky LUT total-internal-reflection
term. Est. ≤ 0.3 ms, only-when-submerged.

## 4. Frame-budget ledger

**Forest headroom assumption (stated per the deep-review master plan):** this ledger assumes the
plan's per-pose isolated targets land — **oblique ≤ 21, eye ≤ 18, aerial ≤ 16 ms med gpuWall**
(mid-band r = 0.75 ⇒ underlying live p95 ≈ 15.5 ms; `90-premise-audit.md` §2.2-2.3) — via the
foliage/base/post levers owned by the other readers. Water is additive on the *world-parity* frame,
which `16-resolve-and-post.md` (P2, Consequence C) already flags as carrying an unmeasured +2–6 ms
class for shadows/GI/clouds relative to the forest measurement scene.

**[VERIFIED-FIX] Why the sum actually closes — the coverage-displacement mechanism** (the previous
draft asked for a 2.5 ms reserve, which does NOT survive the r = 0.75 arithmetic at the oblique
target: 21 + 2.5 = 23.5 iso ⇒ ≈ 17.6 ms live > 16.7; that framing is replaced):

1. **Wet pixels are never foliage pixels.** The scatter hard-excludes trees from open/standing
   water, river channels and the lake shelf (Scatter.ts:389-395; site fields from
   `sampleWaterYNearest`, Heightfield.ts:412-419). A lake band on screen displaces exactly the
   foliage coverage that makes oblique the worst pose.
2. **The displaced cost is measured and large; water's replacement cost is small.** Oblique is
   slope-dominated at **8.1 ms/Mpx** of covered foliage (pixel-scaling law, `90-premise-audit.md`
   §2.4). Water's worst-case shading is ≤ ~2 ms for a FULL-screen lake (table below) ⇒
   ≤ ~0.6 ms/Mpx — ~13× cheaper per covered Mpx. Both terms scale with the SAME variable (wet
   screen coverage W, in Mpx), so at any pose:
   `total ≤ forest_target − 8.1·W + 0.6·W + fixed ≤ forest_target + fixed`.
   The coverage term is *self-funding*; big lakes make the frame CHEAPER than the forest target,
   not more expensive. (Base/terrain raster is triangle-emit-bound, not coverage-bound —
   `base-raster-is-the-bottleneck` — so the submerged terrain it still rasters is already inside
   the forest target's base pool.)
3. **The only true additive is the fixed adders** — paid whenever any wet cell is in the frustum,
   independent of W: caustics bake 0.05 + clipmap vertex work ~0.1 + scene-color snapshot 0.2–0.4 +
   depth copy ~0.1 + FFT ~0.1–0.3 (Tier O active only) ≈ **0.5–1.0 ms isolated ⇒ ~0.4–0.75 ms
   live at r = 0.75**. THIS is what the master plan must reserve.

**Decision this doc forces at plan level: the master budget must carve a named world-parity
reserve; water's share of it is the FIXED adders only — ≤ 1.0 ms isolated hard cap** (skip-draw
deletes it on dry frames; W7 structurally deletes the snapshot, its largest term, if W1 measures
it high). The wet-coverage shading needs no reserve — it is displacement-funded per (2). Grass
(sibling plan) budgets separately.

Line items at 2268×1473 (estimates labeled; W0/W1 probes in §6 pin them):

| Item | Cost | Bound mechanism |
|---|---|---|
| Caustics bake (compute, per frame) | ~0.05 ms (documented, Caustics.ts:10) | fixed 512² kernel, zero fetches |
| Clipmap vertex work (6 × ~32k tris ≈ 196k tris) | ~0.1 ms est. | fixed geometry; dry fragments die at early-z (WaterSurface.ts:11-14) |
| `viewportSharedTexture` scene-color snapshot + `viewportDepthTexture` depth copy **[VERIFIED-FIX]** (the depth copy was previously uncounted) | 0.2–0.4 + ~0.1 ms est. | skip the water draw entirely when no wet cell intersects the frustum (CPU test vs the resident `waterY` mirror, Heightfield.ts:81-82) |
| Wet-pixel shading **[VERIFIED-FIX]** (honest fetch inventory): ripples 4 fetches + refraction 2–3 + foam 4 + SSR ≤ 18 depth taps + 1 color + horizon 4 ranges × 2 (height + canopy) = 8 + ~6 `waterY` bilerps (rapids drop + ramp feather, WaterMaterial.ts:305-307, :322-327) | ∝ wet coverage: 0.3–1.0 ms typical river/lake pose; ≤ ~2 ms full-screen lake | half-res refraction/SSR reads (UE `r.Water.SingleLayer.DownSampleFactor` analog — cvar list verified at [Tom Looman FFW profiling](https://tomlooman.com/unreal-engine-optimization-farfarwest/)), SSR step/cap flags, stripped far-level material (levels ≥ 3: sky+absorption only — the UE "far distance mesh material" analog) |
| Underwater post state | ≤ 0.3 ms est., only-when-submerged | built on submersion only |
| FFT tier (when Tier O active) | 0.1–0.3 ms est. amortized | one cascade update/frame; cascade count ladder 4→2→Gerstner |
| **Total** **[VERIFIED-FIX]** | **NET additive over the forest targets ≈ 0.5–1.0 ms (fixed adders; wet-coverage shading is displacement-funded, see above). Gross water work ≈ 0.6–1.5 ms typical, ≤ ~2.9 ms full-screen-lake pose — a pose whose displaced foliage repays ≥ 8 ms.** | every lever is established SLW practice at its listed operating point |

**[VERIFIED-FIX]** Per-pose picture with the arithmetic carried all the way to live ms
(net water add = fixed adders + any shading NOT covered by displacement; live = iso × r 0.75;
budget law: live ≤ 16.7, plan slack target 15.5 per `90-premise-audit.md` §2.2):

| pose | forest target (iso med) | net water add (worst, iso) | iso sum | live @ r=0.75 | inside 16.7? |
|---|---|---|---|---|---|
| eye, river on screen | ≤ 18 | +2.0 (fixed 1.0 + band shading 1.0, **zero** displacement credit claimed at eye — foliage pool there is only ~2.1) | 20.0 | 15.0 | yes, 1.7 ms slack |
| oblique vista w/ lake | ≤ 21 | +1.0 (fixed only; lake shading is displacement-funded at 8.1 vs 0.6 ms/Mpx — a big lake is NET-NEGATIVE vs the all-forest worst pose) | 22.0 | 16.5 | yes vs the 16.7 edge, but ABOVE the 15.5 slack line — this is exactly the pose the named ≤1.0 ms reserve must be carved for; with it funded (oblique 20 at water poses) live = 15.75 |
| aerial | ≤ 16 | +0.9 (fixed + few stripped far pixels) | 16.9 | 12.7 | yes, 4.0 ms slack |
| underwater | n/a (near field dominates; surface shading collapses — screen mostly below the sheet) | +0.5 (post state, only-when-submerged) | — | — | yes |

Honest statements the ledger carries: (1) the shipped tier's live cost has **never been measured**
— `?ablate=water` exists (TerrainScene.ts:180) and W0 pins it before anything is sized finer;
(2) the snapshot cost estimate is the softest number and is what Candidate B (W7) deletes if it
measures high; (3) nothing here moves the eye/oblique forest gap — that is owned by the
foliage/base workstreams; (4) **[VERIFIED-FIX]** the 8.1 ms/Mpx displacement credit is a
forest-scene measurement (`90-premise-audit.md` §2.4) — W0/W1 must re-pin the foliage slope in the
WORLD scene (shadows/GI/clouds raise per-pixel foliage cost, which makes the credit LARGER, so the
forest number is the conservative side — but it is still an inference until probed).

## 5. Quality bar — mechanism per feature, artifact-free under TRAA

1. **Reflections.** Ladder: sky LUT (free, WaterMaterial.ts:236-237) → probe GI for occluded rays
   (:258-260) → crowned-horizon geometric proxy (terrain+canopy, 4 taps, :248-257 — deterministic,
   so SSR misses fall back without flicker) → 18-step IGN-jittered SSR (:200-234) converged by TRAA
   accumulation (jitter decorrelated per pixel; history valid on water because water writes depth
   and reprojection is analytic, §3.3) → [reserved hero rung: planar]. Anti-artifact mechanisms:
   screen-edge fade on hits (:265-266), ripple-jittered fallback blend against banding (:262-263),
   wide smoothstep knee on the horizon test against razor bands (:256).
2. **Refraction.** Depth-validated UV — a sample landing on geometry *in front of* the surface falls
   back to the straight UV (:168-178), so foreground objects never smear into the refraction.
   Offset ∝ 1/distance (:161) keeps the TRAA reprojection error of refracted content sub-pixel at
   range. The half-res ladder blurs only *under* the surface — the trade UE ships as default
   ([Epic SLW docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/single-layer-water-shading-model-in-unreal-engine)).
3. **Caustics.** Physically-defined intensity (inverse Jacobian of the surface→bed projection,
   closed-form Hessian, real dispersion — Caustics.ts:4-11, :134-166), multiplied into ALBEDO so
   sun/shadow/GI scale it for free (:13-17, :276-283) and TRAA sees it as stable surface detail,
   not a translucent overlay. Depth-parallax entry point + mip-bias defocus + focal ramp
   (:211-213, :241-254). Top of the non-RT quality ladder already ([Yuksel heightfield caustics](https://www.cemyuksel.com/research/heightfield_caustics/),
   [GPU Gems ch. 2](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-2-rendering-water-caustics)); Tier O upgrades its wave source to the FFT slice for surface/foam/caustic coherence.
4. **Underwater.** Same-sigma fog ⇒ the surface→underwater transition conserves color; waterline
   mask from real sheet depth ⇒ no half-submerged seam; Snell window from the same sky LUT ⇒ no
   new lighting model. All post-pool, after TRAA input assembly, so no history pollution.
5. **Foam/ripple motion.** Today: high-frequency detail tolerated by variance clipping
   (PostStack.ts:509-510). W4 adds true flow velocity at the `velReproject` seam (:520-533) so fast
   rivers and FFT swell reproject correctly instead of relying on clipping — removing the one
   ghosting/softening mechanism the current contract permits.
6. **Shadow receive.** PCSS from the shadow clipmap at the water's wp (§3.2) — foam in cliff shade
   goes dim exactly like every other receiver (NaniteResolve.ts:922-955).

## 6. Staged build plan (each stage independently shippable + measurable)

Quality classes and gates per `90-premise-audit.md` §4 (classify BEFORE measuring; shots archived).

- **W0 — measure the legacy tier (no code).** World scene `?oldgeo=1`, A/B `?ablate=water`
  (TerrainScene.ts:180) at 2268×1473, river/lake/aerial poses, `probe-fresh-stutter` style. Gate:
  a real number for today's water cost (never measured). Class: n/a (measurement).
- **W1 — nanite bring-up.** Construct WaterSurface in the nanite slate (drop the
  `DISABLE_OLD_GEOMETRY` gate for water at TerrainScene.ts:180; CausticsBake already runs,
  :124-131). Verify with `?waterdbg=7` depth forensics over resolve depth (WaterMaterial.ts:341-383)
  + screenshots at shoreline/grazing poses. Measure `?ablate=water` A/B on the nanite frame.
  ~0 new shader code. Class Q (new visible feature) → user sign-off on shots.
- **W2 — shadow receive port.** Foam/spec sun terms × `shadowHalf.upsample`/`shadowFactor` (§3.2).
  Gate: side-by-side crops (lit vs cliff-shade water), Class Q sign-off.
- **W3 — SLW cost machinery.** (a) skip-draw when no wet cells in frustum; (b) `?wrefr=half`
  half-res refraction reads; (c) `?wssr=N,half` SSR step/resolution ladder; (d) stripped far-level
  material for levels ≥ 3 (sky reflection + absorption only). Gates: per-flag gpuWall A/B ≥ noise
  floor + shotdiff on near poses (far strip is Class R → user sign-off with shots; UE precedent:
  `r.water.SingleLayer.*` scalability ladder, [Tom Looman FFW profiling](https://tomlooman.com/unreal-engine-optimization-farfarwest/)).
- **W4 — water velocity for TRAA.** Flow-field NDC velocity branch in `velReproject`
  (PostStack.ts:520-533), gated `?wvel`. Gate: moving-river crops/video A/B (ghost trail gone),
  Class Q.
- **W5 — underwater state.** Submersion-gated post fog + waterline mask + Snell window (§3.6).
  Gate: submerge/emerge capture sequence; cost probe only-when-submerged. Class Q.
- **W6 — Tier O (FFT).** 256², 2 cascades, staggered; displacement near / normals mid / skirt far;
  Jacobian → foam + caustic source. Gates: sim-cost probe (target ≤ 0.3 ms amortized), open-water
  shots, tiling check at golden-ratio spacings. Class Q.
- **W7 — Candidate B (conditional endgame).** Only if W1/W3 measure the forward pass's residual
  (snapshot + transparent overdraw) ≥ ~0.5 ms: water-only election channel + mini-resolve after the
  opaque resolve (§1 table). **[VERIFIED-FIX]** Invariant: the water channel is excluded from the
  HZB build — occluding with water depth would cull the refraction/SSR source geometry behind the
  surface (§1 table, row B). Gate: shotdiff ≈ 0 vs W3 output + gpuWall delta. Class I target.

Each stage lands behind a URL flag, is measurable in isolation, and no later stage blocks an earlier
one from shipping.

## 7. Risks + mitigations (mechanisms, not adjectives)

| Risk | Mechanism that bounds it |
|---|---|
| `viewportSharedTexture` snapshot costs more than estimated, or triggers per-material copies across 6 level materials | **[VERIFIED-FIX]** Mechanism named: three's `ViewportSharedTextureNode` renders into a single SHARED framebuffer target across all materials referencing it (that is the "shared" in the name — three.js `src/nodes/display/ViewportSharedTextureNode.js`), and the shipped 6-level tier already exercised exactly this configuration in the old pipeline; W0/W1 measure it directly; skip-draw gate removes it when no water is visible; W7 deletes it structurally. |
| Scene-color snapshot ordering vs the vox resolve (−999) | Water draws in the transparent phase, strictly after both opaque resolve meshes (renderOrder −1000/−999, NaniteResolve.ts:1085-1105) — the snapshot by construction contains both. Verified visually at W1 (voxel crowns must appear in refraction/SSR). |
| 24-bit-key depth quantization vs water's depth epsilons (leak test 0.02 m, SSR hit window) | The resolve's written depth and the election key are the same quantity (NaniteResolve.ts:1071-1078); banding is sub-pixel by the depthKey24 design (256× finer than 16-bit, NaniteRaster.ts:335-341); `?waterdbg=7/8` forensics exist to re-verify (WaterMaterial.ts:341-383); precedent: this exact interop drove the N4-C0 fix (NaniteRaster.ts:884-891). |
| Foam lighting parity (three-lit colorNode in a manually-lit slate) | Two closed options (§3.2): keep the scene's directional light for foam, or fold foam into the resolve's sun formula — both expressed in existing TSL; A/B by shots. |
| TRAA ghosting on fast water / FFT swell | Variance clipping already bounds it (PostStack.ts:509-510); W4 replaces the fallback with true flow velocity at the documented seam (:520-536). |
| GTAO/bilateral darkening at the waterline | Depth-gap tap rejection already in the bilateral (PostStack.ts:400, :405-411); if a band ever shows, the AO distance-fade + water's near-planar depth make a water-pixel AO clamp a one-line, shot-gated tweak. |
| Large-lake far-rim band from the min-reduced field | Pre-diagnosed with rejected alternatives recorded (Heightfield.ts:371-381); owned by the per-water-body far field / hero-lake rung; near levels already fade the dive ramp (WaterMaterial.ts:321-334). |
| Aerial jitter-linked frame bimodality interacting with a new transparent pass | Water adds no cull/HZB feedback (it never enters the election or the pyramids); the bimodality workstream is `11-cull-and-hzb.md` L6 — decoupled by construction. |
| World-parity budget squeeze (shadows/GI/clouds + water together) | Surfaced, not parked: §4 forces the named reserve decision at master-plan level. **[VERIFIED-FIX]** Water's reserve ask is the FIXED adders only (≤ 1.0 ms iso hard cap); its coverage-proportional shading is displacement-funded (8.1 vs ~0.6 ms/Mpx, §4) — so water cannot compound the shadows/GI/clouds squeeze beyond that cap, and each W3 rung has UE-shipped precedent at its operating point. |

### Sources
**[VERIFIED-FIX]** All external citations below were re-fetched or search-verified on 2026-07-02
(skeptic pass): Epic SLW/meshing/bodies/planar, Tom Looman ×2, SoT SIGGRAPH DOI + Rare min-spec,
GodotOceanWaves (staggered one-cascade-per-frame quote confirmed), rtryan98 (256²/512² sweet spot +
≤4 cascades confirmed), BTH thesis (diva2:1778248, "Gerstner waves is still a lot faster"
confirmed), NVIDIA OceanCS PDF (timing quotes extracted), GPC 2025 Enshrouded talk (title/speakers
confirmed), Enshrouded update page, Vermilion (scope corrected to water tables). The one
unverifiable number (+23 ms planar) was withdrawn in §1.

[Epic — Single Layer Water](https://dev.epicgames.com/documentation/en-us/unreal-engine/single-layer-water-shading-model-in-unreal-engine) ·
[Epic — Water meshing](https://dev.epicgames.com/documentation/en-us/unreal-engine/water-meshing-system-and-surface-rendering-in-unreal-engine) ·
[Epic — Water bodies](https://dev.epicgames.com/documentation/en-us/unreal-engine/water-body-actors-in-unreal-engine) ·
[Epic — Planar reflections](https://dev.epicgames.com/documentation/unreal-engine/planar-reflections-in-unreal-engine) ·
[Tom Looman — UE 5.7 FFW profiling](https://tomlooman.com/unreal-engine-optimization-farfarwest/) ·
[Tom Looman — UE 5.6 highlights](https://tomlooman.com/unreal-engine-5-6-performance-highlights/) ·
[Vermilion — UE5 Nanite and Water](https://blog.rime.red/unreal-engine-5-nanite-and-water/) ·
[BTH thesis — FFT vs Gerstner](https://bth.diva-portal.org/smash/get/diva2:1778248/FULLTEXT02.pdf) ·
[R. Ryan — Ocean Rendering Pt 1](https://rtryan98.github.io/2025/10/04/ocean-rendering-part-1.html) ·
[GodotOceanWaves](https://github.com/2Retr0/GodotOceanWaves) ·
[NVIDIA OceanCS slides](https://developer.download.nvidia.com/assets/gamedev/files/sdk/11/OceanCS_Slides.pdf) ·
[Sea of Thieves — SIGGRAPH 2018](https://dl.acm.org/doi/10.1145/3214745.3214820) ·
[Rare — SoT PC system requirements](https://support.seaofthieves.com/articles/360000340928-PC-System-Requirements) ·
[GPC 2025 — Enshrouded water talk](https://graphicsprogrammingconference.com/archive/2025/) ·
[Enshrouded — Wake of the Water](https://enshrouded.com/en-US/news/enshrouded-wake-of-the-water-update) ·
[GPU Gems ch. 2 — water caustics](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-2-rendering-water-caustics) ·
[Yuksel — heightfield caustics](https://www.cemyuksel.com/research/heightfield_caustics/) ·
[Renou — real-time caustics](https://medium.com/@martinRenou/real-time-rendering-of-water-caustics-59cda1d74aa) ·
[gamedev.net — SSR vs planar cost discussion](https://gamedev.net/forums/topic/703353-screen-space-reflections-faster-than-projective-reflections/5410738/) ·
[Intel GameTechDev TAA](https://github.com/GameTechDev/TAA) ·
[Wikipedia — Temporal anti-aliasing](https://en.wikipedia.org/wiki/Temporal_anti-aliasing) ·
local UE5 source: docs/perf-runs/Nanite-UE5-shaders/NaniteTranslucency.usf, NaniteTranslucencyFactory.ush, NaniteVertexDeformation.ush.
