# PROJECT LAAS — STATUS (source of truth)

> **Rehydration protocol** (for an agent resuming with no context): read this file fully, then
> `PROJECT_LAAS_v2.md` (the spec — binding), then `docs/THREE-NOTES.md` (API gotchas for the
> pinned three.js), then the **Current focus** section below. Reference images: `reference/`.
> Never re-plan from scratch; continue from "Next actions". Update this file after every
> meaningful step. Commit per milestone with descriptive messages.

## Mission (1 paragraph)

Fully procedural 4×4 km open world in the browser. WebGPU only (three.js WebGPURenderer + TSL +
raw WGSL compute), TypeScript strict, zero `any`, zero external assets, deterministic by
`?seed=N`. Visual bar: the four UE5-class reference images in `reference/` (noon forest ravine
w/ cobbled streambed; gully close-up; karst tower forest in haze; golden-hour serrated alpine
vista w/ snow + cloud sea below summits — "Witcher" frame). 8 gated phases; verification by
Playwright screenshots compared against references; `DELTA.md` loop each phase. Must ALSO be
smooth + explorable interactively by the user (fly camera, ToD control, bookmarks) — user
feedback comes in chat; the two-frame test is the agent-side acceptance only.

## Hard rules digest (full text = spec §)

- No black/gray shadows (Pillar B); no bare terrain within 10 m (Pillar A/§9); no cloned trees;
  no smooth silhouettes on hero rock/tree; no fog-as-cover; no `MeshBasicMaterial`; no CPU
  per-instance updates; no one-file architecture; never ask user to lower the bar.
- Floors (§2): ≥5M tris forest hero / ≥3M vista post-culling; 4096² heightfield sim; erosion
  ≥500 iters @ ≥2048²; ≥6 tree species w/ per-instance uniqueness; hero tree ≥100k tris; hero
  rock ≥200k tris; grass ≥800k blades; debris ≥80k; particles ≥100k; visible range ≥4 km;
  ≥5 biomes incl. alpine snow; probes ≥24×24×6 per chunk; CSM 4×≥2048² + PCSS + contact
  shadows; raymarched 2-layer clouds + cloud shadows; Hillaire atmosphere; 60 fps @ 1440p
  target hardware class.
- Infeasible item → nearest feasible alternative + entry in `DEVIATIONS.md`. A closed phase has
  zero TODOs in its code.

## Verified environment facts

- macOS 26.4 (Darwin 25.4.0), Apple **M1 Max 32-core GPU**, Metal 4, 3456×2234 display.
- Node v22.12.0, npm 10.9.0. Git repo initialized on `main` (no remote).
- three.js pinned: **0.184.0** (latest on npm as of 2026-06-10). VERIFY APIs against
  `node_modules/three/` source before use — do not trust memory for TSL surface.
- **Playwright WebGPU recipe (SOLVED)**: `chromium.launch({ headless: true, channel: 'chromium' })`
  → apple/metal-3 adapter. Two traps: (1) WebGPU only exists on secure contexts — probe on
  http://localhost:5173, never about:blank; (2) default Playwright headless = GPU-less
  "headless shell"; `channel:'chromium'` selects full Chromium new-headless. Cached in
  `.cache/webgpu-flags.json` by tools/launch.ts.
- Dev server: `npm run dev` (background, port 5173 strict). Shots:
  `npx tsx tools/shoot.ts --scene X --cam "..." --out shots/x.png [--hud 1] [--stats f.json]`.
  Compare: `npx tsx tools/compare.ts --a ours.png --b reference/sceneN.png --out cmp.png`.
  Pixel sampling (shadow-color test): `--sample img.png --px "x,y;x,y"`.
- Sanity scene measured (1080p, M1 Max): 3.1M tris @ 117 fps, render 7.2 ms — lots of headroom.

## Phase checklist

- [x] **Phase 0** — DONE 2026-06-10. Scaffold, WebGPU init + fail-loud diagnostics, HUD, fly
      camera, params, Playwright shot harness (headless WebGPU working), compare tool. Gate
      passed: `shots/phase-0/cmp_sanity_vs_scene1.png`. Proven: compute→storage→instanced draw,
      compute→StorageTexture→sampling, TSL vertex displacement, CPU procedural geometry,
      GPU timestamps, deterministic seeding.
- [x] **Phase 1** — DONE 2026-06-11. 4096² synth (macro layout: NE massif/valley/karst/lake w/
      outlet), pipe erosion 640 it @2048 (hardness-aware thermal), multigrid lake fill, particle
      flow accumulation → carved rivers, moisture, biome+snow classify (coarse-slope hold +
      couloirs + ledges + dither), CDLOD instanced tiles + far shell w/ analytic normals +
      far-detail normal synthesis, PBR splat material (strata/iron bands/lichen/macro variation/
      wet darkening/snow), erosion split view, ground-clamped camera (`x/z/alt/yaw`), CPU height
      readback. Gates passed; see docs/DELTA.md Phase 1. Artifacts: shots/phase-1/.
- [x] **Phase 2** — DONE 2026-06-11. Hillaire LUT atmosphere + aerial perspective (post-pass
      camera-uniform bug fixed — explicit uCamPos/uProjInv/uCamWorld); GPU auto-exposure
      (key 0.125); hemisphere ambient (IBL env path dead → Phase 3 probes); CSM×4 + PCSS +
      screen-space contact shadows (12-step depth march, near-field, floored); volumetric
      clouds (half-res RTT march, baked weather, cloud sea below summits + cloud shadow map);
      TRAA, GTAO (depth-derived normals, distance-faded), bloom, per-ToD grade (strong
      teal-orange golden split). Gates PASSED: golden vista vs Witcher (DELTA.md Phase 2,
      ~70% of ref without vegetation), shadow-color test (chroma 18.3/255, no gray).
      Artifacts: shots/phase-2/. Known debts → DELTA items 1,4,7–10.
- [x] **Phase 3** — DONE 2026-06-11 (vegetation-dependent parts deferred w/ DEVIATIONS).
      Irradiance probe field: 256×256×6 TERRAIN-RELATIVE layers (1.5–105 m above ground),
      heightfield ray-march gather (16 dirs × 16 steps, sun horizon test + albedo proxy +
      sky LUT misses), SH-L1 → 3×rgba16f 3D textures, time-sliced 3072/frame (~2 s refresh,
      invalidate() fast-converge on ToD jumps), injected via IrradianceNode (setupLightMap
      patch); hemisphere dimmed to 0.15× floor. GTAO: depth-derived normals + distance fade
      + luminance-masked 'indirect-only' approx (DEVIATIONS D-1). Screen-space bounce +
      foliage translucency → Phase 4 (D-2). Probe density vs spec floor → D-3.
      VERIFIED: no-black-shadows at golden hour (darkest-20 lum 61.8, chroma 20.1 — AgX-toe
      desat fixed); ?view=probes ambient-only debug view; +3 ms GPU. Forest-interior gate
      re-judged after Phase 4 (no forest exists). Artifacts: shots/phase-3/.
- [x] **Phase 4** — DONE 2026-06-11. Growth grammar (tropisms, whorl/spiral/PLANAR phyllotaxis,
      crown envelopes, light-competition asym, per-instance lean/age/bias = D5); 6 tree species
      (spruce/pine/beech/birch/karst-gnarl/snag) + 3 shrubs (incl. PINK FLOWERING) + fern + 4
      flowers; foliage CLUSTER-CARD pipeline (real leaf/needle meshes captured to per-species
      2×2 atlases — the ez-tree look, zero assets) + hero HYBRID mode (cards + real-mesh
      foliage; hero spruce 1.18M / beech 1.26M tris); bark synthesis 6 recipes (2048² compute,
      albedo/cavity + normal/rough/height, aoNode wired = D-1); rocks (welded icosphere +
      strata ledges + fracture cuts; hero 327k craggy, cliffFace preset, wall, cobbles); grass
      (clumped instanced blades, 260k shown), debris kit (cobbles/pebbles/twigs/chips/litter
      reusing leaf atlas), deadfall (logs ×3 decay + stumps + shelf/cap fungi), dressing
      (moss/lichen/streaks by upness+cavity, hanging vines, ledge ferns, litter ring); foliage
      translucency + SS bounce (D-2), octahedral impostor capture 8×8 albedo+normal+depth +
      relit preview (runtime → D-4/Phase 5). Gate: gallery sheet shots/phase-4/ + macro-meso-
      micro audit in DELTA.md (top-3 deltas fixed: foliage hue variance, log moss, blossoms).
      Forest-interior gate re-judge happens after Phase-5 assembly (no forest yet).
- [x] **Phase 5** — DONE 2026-06-11. GPU scatter (162k trees/467k under/451k stones), per-frame
      instance cull (frustum + terrain-march occlusion + ring classify) → compact indirect draws,
      LOD rings hero≤26/R1≤150/R2≤460/octahedral impostors (dithered crossfades, vegViewPos),
      PER-CASCADE shadow caster culling + fitted crown shadow proxies (world-anchored dither,
      impostor-band casters to 1.1 km) + world-metric PCSS, canopy-aware chromatic probe GI
      (green crown slab + glow), grass/debris probe GI + shade coloring, terrain micro-
      displacement (fbm+ridged creases, biome/gravel-gated, shared DISP table, ?dispdbg=1),
      gorge wall greening, river boulder affinity, grass 5/3-blade clumps + 3-plane tufts +
      near scruff floor. GATE PASSED: floors hero 19.5M / vista 6.8M veg tris / grass ~1.0M
      blades (shots/phase-5/floor-*), repetition strip clean (strip-1..5), DELTA Phase-5
      top-10 logged + top-3 fixed, DEVIATIONS D-5. Shadow regression user-reported and FIXED
      (blobby/flicker/circle — see gotchas). Carried: geometric wall plants, moss volume geo,
      noon-dapple gap-framing re-judge, perf 50–151 ms GPU veg-heavy (Phase 7).
- [ ] **Phase 6** — stream water (refraction/caustics/foam/wet margins), lakes (planar refl),
      hierarchical wind, froxel volumetrics (canopy shafts, valley fog), GPU particles
      (snow/pollen/leaves). Gate: streambed close-up vs scene1/2.
- [ ] **Phase 7** — perf pass (60fps@1440p / reduced preset), HUD full (per-pass GPU timings),
      9 bookmarks, 90s flythrough, full battery, final two-frame test, self-score rubric.
- [ ] **Tier 3** — only after battery passes (see spec §11).

## Current focus

**Phase 2 — atmosphere, shadows, clouds, post** + USER FEEDBACK BATCH 1 (2026-06-11).

User feedback (all four addressed, commits e939266/575b621/next):
1. PERF "~40fps before objects": root-caused via new `?ablate=` + `--gpusample` median
   harness → terrain splat material was ~52 ms of a 73.5 ms GPU frame (35 live noise
   evals/px). Fixed: `NoiseBake.ts` baked value/fbm/ridged + PRE-DERIVED GRADIENT
   textures; GTAO samples 16→8 (defaults cost ~50 ms on vistas); clouds half-res RTT +
   baked weather; 3D-distance quadtree split; castShadowPositionNode (nearest, no morph);
   CSM maxFar 3200. NOW: 19–23 ms GPU @1080p all views (was 73–134). Phase 7 finishes
   (vsync-real fps; spikes re-check on live flythrough).
2. EROSION "sharp diagonal/straight 1-cell trenches, predictable lake patterns": particle
   trace was D8 (8-direction snap) → continuous bilinear-gradient descent w/ inertia;
   strength field blurred before carve (channels have width); carve faded inside lakes;
   particles STOP on filled flats (ε-tilt alignment printed parallel lines) and in lakes;
   hardness-aware talus relax (26 it) post-carve rounds trench walls, towers protected;
   trench enforcement got V-profile (was rectangular select) + fine meander warp octave
   (61 m / ±16 m) so spline trenches aren't ruler-straight; kettle ponds render dark
   (were gravel-gray dots). VERIFIED shots/wip/fix-round2-*.png.
3. LOD "center always high detail": VERIFIED FALSE for the quadtree (live setPose test:
   rings follow camera; `?view=lod` debug added). Real causes user saw: far shell beyond
   world edge + coarse cliffs (see 4). 3D split distance stops altitude over-refine.
4. MESHING "stretched verts on slopes": skirted patches (PlaneGeometry +2 ring, clamp +
   drop in shader → crack-proof) + error-biased splits (height-range mip pyramid; rough
   tiles split earlier and down to 32 m → 0.5 m quads on cliff close-ups). Snow dither
   gated near boundary (white speckle on rock fixed).

Phase 2 items: 1–5 BUILT as before (atmosphere LUTs, SunSky, CSM+PCSS, clouds, post).
CLOUDS NOW VISIBLE AND CORRECT — root causes were (a) quad-pass camera uniforms
(cameraPosition/WorldMatrix/ProjectionMatrixInverse are the POST QUAD camera inside
RenderPipeline.outputNode → explicit uCamPos/uProjInv/uCamWorld uniforms now) and
(b) depth convention is CLASSIC here (sky d=1.0, not reversed) → isSky + maxD fixed.
Aerial perspective only became truly distance-correct with the same fix.
`?cloudview=1..9` probe ladder kept (tone mapping auto-off when probing).

PHASE 2 CLOSED 2026-06-11 (see checklist + DELTA.md). All listed items landed: cloud art
pass (contrast-stretched weather, isotropic phase floor, base-darkened ambient, default
cov 0.62), contact shadows (?ablate=contact to A/B), black facets root-caused to GTAO
(NOT PCSS — depth-derived normals fixed it), gate + shadow-color test PASSED.

**Phase 5 — BUILT, gate pending.** The world is planted end-to-end:
- `Scatter.ts`: boot GPU clustered-Poisson (162k trees / 467k understory /
  7.4k extras at seed 1), per-class density fns (biome/slope/treeline/moisture/
  snow/rockExp/water), ecotone warp, parent-clump field doubling as canopy
  proxy for understory (ferns under crowns, flowers in gaps, pink shrubs at
  clump EDGES). pcg2d integer hash (pure expression — usable in materials).
  + `buildCanopyMap`: crowns splatted to a 1024² coverage field; attenuates
  probe ambient under canopy (terrain ×0.55, veg ×0.4) = forest interiors no
  longer sky-bright (user "washed out" + shadow-visibility fix).
- `VegLibrary.ts`: K=4 variants/species; R1/R2 ring geoms from the SAME
  skeleton (no-pop LODs); ring diet in TreeBuilder (bark stops below anchor
  level; cards thin+enlarge ≈ sqrt(stride)) → R1 avg 8.4k tris, R2 1.8k.
  Impostor capture per species.
- `Forests.ts`: per-frame clear→cull→indirect computes. Cull = per-class
  dist bound + 6-plane frustum + terrain-occlusion march (camera→crown-top
  against height buffer) + ring classify w/ overlap bands → atomic append
  into per-(pool,ring) compact regions → `geometry.setIndirect` draws (one
  shared IndirectStorageBufferAttribute, byte offsets). Rings: R1 cards
  ≤150 m → R2 ≤460 m → octahedral impostors (D-4 runtime: 4-tile hemi-oct
  bilinear blend, relit normals, per-instance yaw/tint) — IGN-dithered
  crossfades. Tree rings 1+2 cast shadows; terrain casts via `ShadowProxy`
  (512² grid; CDLOD castShadow=false; saved ~54 ms).
- `GroundRing.ts`: toroidal-clipmap grass (3072², 136 slots/m², 4/2-blade
  CLUMP geoms near/mid + tuft cross far; ≈520k blades visible at meadow
  framings) + debris ring (cobble/pebble/twig/chip/litter; streambed
  override density — beds read cobbled). `CanopyShell.ts`: far forests as a
  lit lumpy aggregate beyond 620 m.
- Veg materials: GI-patched (IrradianceNode), canopy-attenuated, per-instance
  tint, vec4-alpha shadow contract + maskShadowNode cutouts,
  castShadowPositionNode, instance NORMAL rotation (normalLocal.assign).

## Next actions (always keep current)

- **PHASE 6 — water, wind, volumetrics, particles (task #7 in_progress).**
  (1) STREAM WATER v1 LANDED (commit pending tag): WaterSurface 6-level
  clipmap (shared 128² grid, world-pos vertex from waterY f32 buffer,
  inner-rect fragment cutout, min-reduced far field, world-bounds mask) +
  WaterMaterial (viewportSharedTexture refraction w/ depth-validated uv,
  Beer–Lambert absorption, SSR reflections w/ canopy-attenuated sky
  fallback, two-phase flowmap ripples from fbm gradients, slope-keyed
  rapid foam + shore foam, depth-feathered shoreline). Hydrology:
  waterYRaw in carve kernel (ponds at FILL level W — flat; rivers at
  bed+gated depth; slope-gated: steep reaches dry), SPLIT thresholds
  (RIVER_T 14 carve/moisture unchanged; WATER_T +220 visible water only),
  flowDir = dir × speed (0 in lakes). User-confirmed fixed LIVE: water
  spikes, far half-flooded mountains, border sheets, foam stripes, foam
  loop seam (detail octave moved inside the two-phase blend + variance
  renorm, commit "FIX: foam loop seam"), flow motion "looks good now".
  REMAINING for stream-water close: caustics on submerged beds, gate
  shots vs scene1/2 (streambed close-up), underwater camera guard maybe,
  WATER_T fine-tune if user flags. Deadfall winding/caps fixed+confirmed
  (1a80f86). THEN: lakes planar reflection → hierarchical wind (user is
  waiting on animated trees) → froxels → particles.
  (2) LAKES: planar reflection at LAKE_LEVEL (one clip-plane pass) +
  shore blend.
  (3) HIERARCHICAL WIND: trunk sway (low-freq) + branch/card flutter
  (per-instance phase from slot hash) + grass bend (GroundRing positionNode)
  — one global wind uniform field (dir + gust fbm), amplitude by exposure
  (canopy map). (4) FROXEL VOLUMETRICS: canopy light shafts + valley fog
  (froxel grid, sun shadow-map gated, probe-ambient in-scatter). (5) GPU
  PARTICLES ≥100k: pollen/leaves/snow by biome+season-of-day, wind-advected,
  simple compute integrate + instanced quads. (6) Carried Ph-2: 2nd cloud
  layer + god rays (froxels may cover). Gate: streambed close-up vs
  scene1/scene2 + golden-hour wind motion check + particle density floor.
- Phase 5 carried debts (do NOT re-open the phase; fold into 6/7 where
  natural): geometric wall plants (wall-veg scatter class), moss volume
  geometry, per-cluster LOD for hero rocks/trunks if close-ups demand,
  noon-dapple re-judge at a canopy GAP framing (crown proxies landed —
  expect pools now), debris-floor interpretation (caps 220k; visible max
  measured ~19k at bed framings — revisit at Phase-6 streambed gate),
  distant-forest felt at vistas (impostor/shell uniformity).
- Perf watch (Phase 7 owns): 50–151 ms GPU at veg/grass-heavy 1080p views
  (grass tuft band + caster draws added); timestamp-query overflow warning
  (HUD GPU timings undercount — cosmetic; resolveTimestampsAsync batching).
- PENDING USER CONFIRM: shadow flicker gone live (world-anchored dither) —
  confirmed fixed for blobs/circle; flicker check outstanding.

## Key decisions log

- **D1** Pin three@0.184.0; mitigation for API drift: read installed source, keep notes in
  docs/THREE-NOTES.md. Downgrade to 0.180.x only if 0.184 breaks something structural.
- **D2** Tracking: STATUS.md (this file) = source of truth; harness task list mirrors phases
  (tasks #1–#8 = phases 0–7); git commit per milestone. DELTA.md / DEVIATIONS.md per spec.
- **D3** World macro-layout is code-guided for art direction (composed, per Pillar E): main
  glacial U-valley NE→SW with river → lake in SW low corner; serrated alpine massif N/NE
  (Witcher frame); tower-karst forest ravine biome center-S (scene1/3); meadows + rolling
  forest between; wetland margin at lake. Detail fully procedural + seed-driven.
- **D4** Verification screenshots: prefer headless Playwright Chromium with WebGPU/Metal flags;
  fall back to headed if headless adapter unavailable. (Resolved Phase 0 → record flags above.)
- **D5** Per-instance tree uniqueness strategy: K structural variants per species per LOD ring
  + continuous per-instance GPU deformation (lean/droop/crown asymmetry/age/hue) + bespoke
  unique meshes for nearest hero trees (background-generated, cached). Document in DEVIATIONS.
- **D6** Erosion default 2048² active grid (spec floor) on 4096² synth field; `?quality=ultra`
  runs 4096². Decide final default by measured load time (~budget ≤15 s gen).

## Architecture map (planned; update as built)

```
src/core/      Engine, Diagnostics, Params, Seed, Profiler, Quality presets
src/gpu/       passes/ (Heightfield, Erosion, Flow, Biome, Scatter, Cull, Probes, Clouds,
               Froxel, Wind, Particles, TexSynth), HiZ, indirect helpers, noise lib (TSL+WGSL)
src/world/     Heightfield(owner of terrain textures), TerrainTiles(quadtree+meshlets),
               Streaming, Biomes, Rivers, Lakes, Snow
src/vegetation/ TreeBuilder + species/, RockBuilder, GrassSystem, Shrubs, Flowers, Ferns,
               Debris, Deadfall, Dressing, Impostors
src/render/    Materials (terrain/bark/foliage/rock/water TSL), ShadowSetup(CSM+PCSS+contact),
               GIProbes, PostStack (TAA/GTAO/bloom/grade/DoF), AutoExposure
src/sky/       AtmosphereLUTs, SkyModel, SunIBL, Clouds
src/debug/     HUD, Scenes (gallery/terrain/...), Bookmarks, Flythrough, Compare overlay
tools/         shoot.ts, compare.ts, battery.ts (Playwright verification battery)
shots/         screenshot output (gitignored except curated phase closes → shots/phase-N/)
docs/          THREE-NOTES.md (API gotchas), DELTA.md, DEVIATIONS.md, COLOR-SCRIPT.md
```

## Reference image analysis (art targets)

- `scene1.png` 1920×1080-class, noon ravine: cobbled dry streambed w/ trickle, rounded mossy
  boulders, dark cliff overhangs framing top corners, lush karst towers midground, luminous
  white-blue haze bg. Shadows: blue-gray on rock, green-filled in foliage. Value structure:
  dark frame → lit mid → bright bg.
- `scene2.png` gully close-up: deadfall logs across cobbles, deep-green mossy overhang (shadowed
  but COLORFUL), sunlit tower behind.
- `scene3.png` karst forest vista: dozens of vegetated rock towers receding through 4+ haze
  layers; canopy sea between towers; soft broken-cloud toplight.
- `02_Silver_Demo_Wallpaper...png` (Witcher IV, 3840×2160): golden hour alpine; dark foreground
  outcrop + figure (silhouette framing); serrated rust-red peaks w/ slope-correct snow; conifer
  slopes down to huge hazy valley; cloud sea BELOW summits wrapping ridges; god rays from
  upper-left sun; teal-orange split (warm rock/lit conifers vs cool snow shadows/valley haze);
  scattered dead snags on right slope.
- Implied landforms: serrated ridged massif + vertical-walled tower karst + glacial valley.
  Terrain synthesis needs an explicit tower/mesa formation term, not just ridged fBm.

## Phase 1 progress snapshot (2026-06-10)

Done: synthesis (macro layout + karst towers + anisotropic ridges), pipe-model erosion
(hardness-aware thermal), multigrid lake fill, particle flow accumulation, river carve +
channel enforcement, lake w/ outlet, moisture; debug hillshade preview + `?view=hydro`.
Remaining for phase close: TerrainTiles (CDLOD quadtree + far shell), real PBR terrain
material (triplanar/splats/snow/macro variation), biome+snow classify pass, `?scene=terrain`
split view, ground-clamped camera helper, silhouette/tiling gate + DELTA.md.

## Gotchas / lessons learned (append-only)

- WebGPU secure-context + headless-shell traps → see "Verified environment facts".
- TSL `.assign()/.addAssign()/.toVar()` require an active stack (inside `Fn()`); material node
  graphs are NOT inside Fn → shared TSL helpers must be pure expression builders (NoiseTSL is).
- @types/three 0.184 types nodes generically: use `Node<'vec3'>` aliases from `src/gpu/TSLTypes.ts`
  (`NF/NV2/NV3/NV4…`); bare `Node` has no operators/swizzles.
- `three` and `three/webgpu` both re-export from `three.core.js` — safe to mix imports.
- `StorageTexture` defaults rgba8unorm + `mipmapsAutoUpdate=true` (auto mips after compute
  writes when generateMipmaps). For float data set `.type = FloatType` etc.
- Verify cast shadows w/ custom `positionNode` on instanced meshes when real shadows land
  (Phase 2) — sanity scene shadows looked absent; may need `material.shadowPositionNode`.
- Compute storage-buffer limit: default 8 per stage — request more via
  `requiredLimits` (done in Engine; adapter max here = 10) AND keep kernels lean.
- TSL atomics: `instancedArray(n,'uint').toAtomic()`; then ALL access via
  atomicStore/atomicAdd/atomicLoad; `float(atomicLoad(...) as unknown as NU)` for reads
  (AtomicFunctionNode lacks value-typed methods in @types).
- mx_noise/mx_fractal outputs are SIGNED — remap explicitly or lowlands sink below
  lake level ("puddle plague").
- Relaxation-style fills propagate ~1 cell/iter: ALWAYS multigrid them.
- A lake without an outlet river floods its valley to the spill saddle.
- Endless-loop debug rule: when iterating visual passes "with no effect", first verify the
  served code changed (curl the module), THEN check upstream state assumptions.
- Per-component Rng streams (seed.rng('x')): adding draws must never re-roll other systems.
- 1D dispatch >65535 workgroups: three auto-splits to 2D and instanceIndex stays linear —
  but pad-guard every kernel (`If(i >= N) Return()`).
- RenderPipeline.outputNode runs on a QUAD camera: `cameraPosition`/`cameraWorldMatrix`/
  `cameraProjectionMatrixInverse` resolve to THAT camera (silently wrong values, no error).
  Pass scene-camera uniforms explicitly (this is why three's GTAO/TRAA take `camera`).
- Depth here is CLASSIC convention (sky/clear = 1.0). Verify per pass — don't assume
  reversed-z. Probe in-shader (paint values) rather than reasoning from docs.
- Tooling traps: vite fsevents misses tool-driven writes → `server.watch.usePolling` in
  vite.config; esbuild strips comments from served TS → grep served code for IDENTIFIERS
  only; numeric literals get rewritten (1000 → 1e3).
- `fps` in headless ≠ GPU throughput (CPU submits ahead). Use gpuPasses timestamps,
  median over many samples (`tools/shoot.ts --gpusample N`), plus `?ablate=` attribution.
- GTAONode defaults (16 samples) cost ~50 ms on 1080p terrain vistas; resolutionScale 0.5
  produced row-streak artifacts — keep full res, 8 samples.
- Filled-DEM flats have a UNIFORM ε-tilt: particles crossing them all align to it and
  print parallel straight lines. Stop particles below ~2× the ε slope (and in lakes).
- device.onuncapturederror is wired in Engine — silent black frames usually mean a
  LOGIC bug (wrong uniforms), not a validation error.
- WebGPU `readRenderTargetPixelsAsync` rows are TOP-left origin — flipRows()
  before building DataTextures or every capture is v-flipped (was invisible on
  near-symmetric sprays, obvious on trees).
- Capture scenes MUST use DoubleSide materials — leaf blades facing away from
  the ortho camera get backface-culled and the atlas comes out empty (bit the
  broadleaf tiles; conifer needles survived by accident of normal tilt).
- Real-geometry needles at true scale are sub-pixel at review distance — they
  vanish under TRAA. The ez-tree lesson: lushness = BIG captured cluster cards
  (one card = a whole painted spray); real needle geometry is for the hero ring
  where pixels exist. Hybrid (cards + mesh) wins close-up.
- Tree structure realism (user feedback): foliage must sit on a FINE twig level
  (planar two-sided branchlet lattices for conifer boughs / distichous beech
  twigs), never directly on primaries — `planar` LevelParams flag.
- Auto-exposure note again for assets: albedo tweaks barely move the frame;
  judge materials by RELATIVE contrast (bark vs foliage vs ground).
- 8-bit capture of dark albedos bands — sqrt-encode at write, square at sample
  (foliage atlases, bark, impostors all do this).
- Broken-trunk taper: trunk points span only the kept length — taper must use
  t×brokenTop or the break ends in a spike and the jagged cap never triggers
  (also: don't double-cull children above a break that's already shortened).
- TSL toVar/assign (incl. inside helper fns like a hash!) need a Fn() stack —
  material node graphs DON'T have one. Shared helpers must be PURE expression
  chains (pcg2d was rewritten for this).
- WGSL buffer indices must be i32/u32: a float select-chain `.toInt()` can
  still emit an f32 var as index — use int(0).toVar() + If-assigns.
- sim-res hydrology vs full-res height: W−h and riverDepth comparisons need
  generous thresholds (≥0.25 m) or interpolation mismatch flags whole
  floodplains as "under water" (silently deleted 53k trees + all grass there).
- three shadow contract for custom materials: shadow alpha = colorNode.a ×
  alphaTest copy — vec3 colorNodes silently discard ALL caster fragments.
  Pin vec4(rgb,1) + maskShadowNode for alpha-tested cutouts. Instanced
  positionNode ALSO needs castShadowPositionNode set explicitly.
- Custom instancing must rotate normals: assign normalLocal inside the
  positionNode Fn (three's own InstanceNode mechanism). "Quasi-radial normals
  don't need rotation" is wrong — per-fragment lighting flips sides.
- frontFacing-based debugging on DoubleSide cards is ambiguous (rolled quads
  show both faces) — verify winding on closed tubes or single-sided geo only.
- FlyCamera owns camera orientation: scenes can't lookAt; pass spawn pose via
  hooks.initialPose (applied after the rig exists). ?pitch= now works.
- Indirect-draw stack that works on three 0.184/WebGPU: Mesh (not
  InstancedMesh) + geometry.setIndirect(attr, byteOffset) + instanceIndex
  reads via compact list; counts written by compute into the SAME
  IndirectStorageBufferAttribute via storage(); frustumCulled=false.
- CSMShadowNode (three 0.184): cascade shadows CLONE light.shadow — set
  sun.shadow.camera.near/far EXPLICITLY (defaults near .5/far 500 <
  lightMargin → empty maps, no errors). Lazy _init samples the projection
  at first material build (TRAA jitter/boot transients → NaN extents cached
  forever); apps must call updateFrustums() after camera changes — we
  refresh jitter-stripped + verify finite + resize hook (ShadowSetup).
- Shadow-debug traps that burned hours: (1) judge shadow PRESENCE only with
  the sun positioned so shadows fall TOWARD the camera (they hide behind
  casters otherwise — false "doesn't cast" reads); (2) FlyCamera owns
  orientation — debug scenes MUST set hooks.initialPose or every shot frames
  the wrong spot; forward = (−sin yaw, 0, −cos yaw); (3) headless static
  shots ≠ user's interactive session (DPR 1.5, window resizes, continuous
  motion, TRAA history) — verify BOTH before declaring lighting fixed;
  (4) ablate evidence goes STALE after upstream fixes — re-run the matrix.
- vdata trick for artifact triage: ?clsdbg=1 flat-colors every veg class
  (hue = cls·47°) — identified "dark slabs" as beech cards in minutes after
  hours of wrong guesses (they were SPECULAR-washed cards: one flat normal
  per card ⇒ uniform silver sheen at glancing sun; foliage cards must be
  near-diffuse, roughness .92).
- **TSL `cameraPosition` is PER-PASS** — in the shadow pass it's the cascade
  shadow camera (~lightMargin away from everything). ANY camera-distance
  logic that discards/collapses geometry (LOD fades, distance culls,
  billboard shrink) silently deletes those casters from EVERY cascade map
  while the main view stays perfect ("vegetation casts no shadows" bug —
  weeks of misdirected CSM debugging). Route fade distances through an
  explicit main-camera uniform (vegViewPos in VegInstance).
- maskNode vs maskShadowNode (three 0.184): maskNode discards in the MAIN
  pass; the shadow pass uses maskShadowNode ?? maskNode. Dither-fades belong
  in maskNode with maskShadowNode pinned (cutout or bool(true)) — if both
  rings of an LOD crossfade dither the SHADOW pass with the same IGN,
  correlated texel holes thin the shadow exactly at every ring band.
- Differential debugging beats layer-bisection when a system "half works":
  the user's "terrain casts, vegetation doesn't" + "stones cast, trees
  don't" observations localized in minutes what ablate-matrix bisection
  (filter/post/GI/material/cascades) couldn't — ask WHICH objects differ,
  not WHICH pipeline stage.
- Shadow-proxy lessons (user-reported "small objects, massive flickery
  shadows in a circle"): (1) proxy dims must FIT the pool's real geometry
  (class-max cull bounds oversize small variants ~2×); (2) NEVER dither
  shadow casters with screen-space IGN — cascade boxes refit every frame
  so the pattern swims = flicker; anchor dither in WORLD space
  (hash12(positionWorld)); (3) texel-metric PCSS penumbra caps are
  cascade-relative — 14 texels = 28 cm near, 21 m far; convert blur to
  WORLD meters via reference('left/right/near/far', shadow.camera);
  (4) any caster-reach cutoff by camera distance prints a visible CIRCLE
  on the ground from altitude — fade casters out (impostor-band proxies
  to 1.1 km), never hard-stop them.
- An "identical render" after a lighting change usually means auto-exposure
  re-normalized it away: judge lighting work by ablate A/B DIFFS and the
  ?view=probes ambient view, not by absolute frame brightness.
- MeshGrower enforces NO winding convention — every generator owns its own.
  Tube basis (N, B=T×N) needs base-ring-first quads (a[k], a[k+1], b[k+1],
  b[k]) for outward fronts; an x/z lathe param (cos a, ·, sin a) is LEFT-
  handed → the MIRROR order; caps advancing along −T flip handedness again.
  DoubleSide masks reversed winding silently (bark "insurance" hid the tube
  bug for two phases) — FrontSide materials (deadwood/mushroom/rock) expose
  it. User-reported on logs/stumps/branches; fixed at source 1a80f86.
  Also: tubes have no ring-0 cap — fine attached to a parent, an OPEN HOLE
  on free-lying deadfall (capBase opt). Verify new closed geometry with
  ?facedbg=1 (front green / back red) before shipping it.
- flowStrength is a SHARED driver (carve depth, moisture, splat beds, veg
  gates, boulder affinity). NEVER retune its threshold for rendering — the
  whole world re-layouts (rivers move, forests shift). Split thresholds:
  RIVER_T = terrain texture, WATER_T = visible water (FlowRivers).
- Pond/lake water surface must be the FILL LEVEL W (flat per pond, meets
  terrain at the true shoreline). bed + blurred(depth) builds 30 m faceted
  water towers wherever deep pots abut high ground (blur smears depth onto
  ridge cells). Dry cells in the render field sink below the 3×3
  NEIGHBORHOOD-MIN bed (own-bed−2 still stands above channel water on tall
  banks = water walls). Wet cells get 2 smoothing iterations (wet-masked)
  or cascades render as 2 m staircase shards.
- Water clipmap traps: (a) far levels MUST sample a min-reduced field —
  coarse verts on the full field stretch one wet texel across a 48 m cell
  ("mountains half under water" from afar, gone up close); (b) clamp-to-
  border sampling extends any wet border texel into an infinite off-world
  sheet — hard world-bounds mask in the material; (c) animated foam must
  advect with the TWO-PHASE flowmap like the normals — linear time
  advection slides thresholded fbm level sets into hard white stripes.
