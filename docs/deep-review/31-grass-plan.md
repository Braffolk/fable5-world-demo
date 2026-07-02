# 31 — GRASS / GROUND-COVER PLAN (nanite path, zero quality / zero perf sacrifice)

Artifact doc (design + budget + integration; no code yet). Companion to the water plan (30-water-plan.md);
same structure. Every claim about our code cites `file:line` at nanite-raster HEAD `6a93dfb`; prior art is
cited with sources. Canonical measurement config: 200k trees @ 2268×1473 (dpr 1.5), poses eye/oblique/aerial,
gpuWall med — current isolated ≈ 18.9 / 37.2 / 16.5 ms, live p50 25.0 / p95 33.4
(docs/perf-runs/2026-07-02-attribution-and-waves.md). Frame law: locked-60 LIVE, whole frame p95 ≤ 16.7 ms.

---

## 0. TL;DR + chosen architecture

**Grass is the workload the vis-buffer election was built for, and the repo already ships a GoT-class
per-blade grass system in the wrong lane.** `GroundRing` is a complete, art-tuned, compute-culled toroidal
blade system (≥800k blades, 4-rung LOD ladder, coverage-conserving thinning, terrain-splat handoff —
GroundRing.ts:1-17, 90-125) that today renders as HW three.js draws with a depth-prepass twin to fight 2-8×
overdraw (GroundRing.ts:744-757, VegPrepass.ts:1-22). The nanite frame makes that entire overdraw machinery
obsolete: the election is a free exact depth-prepass (one 32-bit atomicMax per fragment, one resolve shade
per pixel — NaniteRaster.ts:952-975, NaniteResolve.ts:1084-1090), and the architecture has grass slots
**already reserved**: `MATERIAL_CLASS.grass = 5` (GeometryRegistry.ts:141-154), `TRANSFORM_CHANNEL.grass = 3`
(GeometryRegistry.ts:157-163), a resolve fall-through currently painting flat gray (NaniteResolve.ts:872-879),
and a two-sided cluster raster built for exactly this geometry (GeometryRegistry.ts:174-177,
NaniteRaster.ts:105-126).

**Chosen architecture — two tiers, mirroring the forest's own mesh→aggregate→splat shape:**

- **Tier 1 (0 → ~150 m): grass PATCHES as matClass-5 nanite instances.** A small library of pre-built
  grass-patch meshes (blade clumps at L0 → card tufts at L2/L3) registered with a real aggregate DAG
  (the leaf-crown builder precedent: remove-whole-blades + grow-survivors, BuildAggregateDag.ts:1-31,
  registered like the leaf head with `transformChannel`/`twoSided`/`aggregate`, GeometryRegistry.ts:1102-1105).
  Patches ride the normal instance buffers, hier BFS cull, HZB occlusion, SW/HW raster, and the single
  resolve — **zero new raster machinery**. LOD is the error-driven DAG cut, not hand-tuned dither bands.
- **Tier 2 (~150/265 m → horizon): the terrain splat, which is already shipped and already color-matched** to
  the blade palette, including the directional backlit sheen that keeps far swards alive
  (TerrainMaterial.ts:202-213, 266-284). Work item: make sure the same grass-field term lives in the nanite
  resolve's terrain branch (`buildTerrainShading` inputs at NaniteResolve.ts:447-456) so the handoff is
  tonally invisible. Zero triangles, zero marginal cost.
- **Wind** stays the single shared advected-fbm field (Wind.ts:44-104) via a new `grass` block in
  NaniteFetch's channel switch (trunk precedent NaniteFetch.ts:202-224, leaf NaniteFetch.ts:232-252), so the
  raster and the resolve deform identically by construction (shared `makeFetch` — NaniteResolve.ts:219-227).
- **Interaction (trample)** is the Ghost-of-Tsushima displacement buffer: a ~128² camera-following ping-pong
  texture, capsule-stamped + damped-spring relaxed each frame, read by the grass channel. ≤0.05 ms.
- **The shipped GroundRing/VegPrepass lane is the A/B control and the fallback** — it composites correctly
  against resolve-written depth today (NaniteResolve.ts:1071-1078, renderOrder −1000 at 1088) — and is
  deleted from the nanite build only when every gate in §7 passes. "Zero sacrifice" is guaranteed by
  construction: the old lane keeps running until the new one beats it on BOTH gpuWall and shotdiff.

Prior art says this exact shape ships at 60 fps on far weaker hardware: Ghost of Tsushima renders full-screen
per-blade fields in a 33 ms PS4 frame with compute generation + 2-level blade LOD + terrain-texture far field
([GDC 2021, Procedural Grass in Ghost of Tsushima](https://gdcvault.com/play/1027033/Advanced-Graphics-Summit-Procedural-Grass),
[slides](https://archive.thedatadungeon.com/ghost_of_tsushima_2020/documents/gdc_2021/gdc_2021_procedural_grass_in_got.pdf));
Horizon Zero Dawn places all ground cover procedurally on GPU at runtime with 3 dithered LODs
([Guerrilla, GPU-Based Procedural Placement](https://www.guerrilla-games.com/read/gpu-based-procedural-placement-in-horizon-zero-dawn));
UE5.6/5.7 Nanite Foliage converges on aggregates-then-voxels-then-material for sub-pixel vegetation and ships
dense ray-traced forest + ground cover at 60 fps on PS5
([Epic docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-foliage),
[PC Gamer on the Witcher 4 demo](https://www.pcgamer.com/hardware/the-witcher-4s-leafy-glory-is-all-down-to-epics-nanite-foliage-and-largely-so-is-the-fact-it-can-run-ray-traced-at-60-fps-on-a-lowly-ps5/)).

---

## 1. THE DISTANCE LADDER (centerpiece)

### 1.1 The disease this ladder must never contract

Our SW raster's measured failure mode is **sub-pixel emit**: per-triangle setup with <1 px coverage is pure
waste, and the trunk DAG's failure to coarsen produced 46M tris with ~97% sub-pixel — the base-raster
bottleneck finding (docs/perf-runs/2026-06-26-base-raster-review-handoff.md; memory
`base-raster-is-the-bottleneck`). A raw grass blade is 14 mm wide (`W = 0.014`, GroundCover.ts:37). Naïvely
instanced to distance, blades re-create the trunk disease at ~25 m. The ladder's governing law is therefore:

> **LAW: blade work never exceeds pixel work.** At every rung, the representation's smallest emitted triangle
> must project ≥ ~1 px in its *narrow* dimension across the rung's entire distance band. Enforced two ways at
> once: (a) **population thinning conserved by widening** — survivors widen by 1/√thin so screen coverage is
> constant while emit count falls (`grassThin` + widen, GroundRing.ts:106-110, 791-795); (b) **the DAG cut**
> — each patch DAG level is authored so its selection band (projK·ownError ≤ τ, the same flat cut every
> aggregate rides — GeometryRegistry.ts:170-173, BuildAggregateDag.ts:12-16) begins before its triangles
> shrink below ~1.5 px. A level that would emit sub-pixel triangles is *unreachable by construction*.

This is the same conservation every shipped system uses: GoT thins 4:1 per coarser tile ring and ends in "a
single texture on terrain"; HZD dithers 3 LODs; UE5.7 collapses clusters to aggregate voxels. Nobody rasters
per-blade geometry at distance ([GDC 2021 GoT](https://gdcvault.com/play/1027033/Advanced-Graphics-Summit-Procedural-Grass);
[Guerrilla GDC 2017](https://gdcvault.com/play/1024700/GPU-Based-Run-Time-Procedural);
[Epic Nanite Foliage](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-foliage)).

### 1.2 The rungs

Projected sizes below use the canonical frame (2268×1473; focal ≈ 1600 px ⇒ px-per-meter ≈ 1600/d). The
world widths are the shipped, art-approved ones — the ladder keeps the tuned look and only changes the lane.

| Rung | Band | Representation (per patch-DAG level) | Emit unit | World width (incl. widen) | Projected width at band far edge | Source geometry |
|---|---|---|---|---|---|---|
| **R0 near blades** | 0–30 m | L0: 5-blade × 4-seg clumps, rounded-cross-section normals | 35 tri/clump | 16 mm (×1.15) | ~0.9 px @30 m | GroundCover.ts:32-71 (±38° edge-normal tilt 39-52), bladeClump GroundRing.ts:207-253 |
| **R1 mid blades** | 30–70 m | L1: 3-blade × 2-seg clumps, ~⅓ population, 1/√thin widened | 9 tri/clump | 16–33 mm (widen ≤4, GroundRing.ts:793-795) | ~0.7 px width but 5–11 px height @70 m (coverage ≥3 px) | grassBladeGeometry(2), GroundRing.ts:667 |
| **R2 tufts** | 70–150 m | L2: 3-crossed-card tufts, ~⅛ population | 6 tri/tuft | 0.12 m span | ~1.3 px @150 m | tuftGeometry GroundRing.ts:255-291 |
| **R3 super-tufts** | 150–265 m | L3: single wide tuft card(s), coarse population (~2/m² grid precedent) | 2–6 tri | 0.6–1.3 m span | ~4–8 px @265 m | tuftGeometry(0.21) + far widen, GroundRing.ts:696-698, 793-794 |
| **R4 splat** | 265 m → horizon | terrain material response: blade-palette-matched field color + patch dryness + directional backlit sheen | **0 tri** | — | — | TerrainMaterial.ts:202-213 (color match), 266-284 (sheen); resolve terrain branch NaniteResolve.ts:446-472 |

Band edges are the *shipped* constants (G_NEAR=30 / G_MID=70 — GroundRing.ts:93-94; FAR_R0=150 / FAR_R=265 —
GroundRing.ts:121-124) recast as DAG-cut error thresholds: author `ownError_k ≈ half the rung's world width`
so the flat cut selects L_{k+1} exactly where L_k's width crosses ~1.5 px. The cut replaces the shipped
three-band complementary-IGN dither crossfade (GroundRing.ts:151-187, cull double-append 495-506) with a
continuous error-driven transition; if a visible pop ever survives the DAG's sibling-exactness, the same IGN
dither can be re-applied *inside the raster* at level boundaries — but the leaf-crown aggregate DAG's
crack-free cut machinery (bit-exact sibling pairs, BuildAggregateDag.ts:12-19) is the reason to expect none.

**Why patch-DAG and not per-blade emit:** one patch (≈4×4 m) is one instance in the hier BFS; its DAG cut
picks ONE level per cluster; cluster count per patch falls with distance². The cull's per-instance work is
~4.7k patches worst case (all-meadow ±155 m ring: π·155²/16 m²) — noise next to 200k trees. An escalation
lane exists if measurement ever demands it (§8 R6): a third work-queue class that regenerates blades
procedurally in-register from `pcg(worldCell)` with screen-clamped ribbon widths — the voxel raster proves
the pipeline accepts a second emit class cleanly (`qVoxRasterRO`, NaniteRaster.ts:204-215) — but the patch
route needs no new kernels and is the plan of record.

### 1.3 Emit budget proof (worst pose: eye-level open meadow)

Worst-case R0 population: π·30² m² × ~90 slots/m² × density ≤1 ≈ 254k clumps × 35 tri ≈ **8.9M tri**, every
one ≥1 px wide and mostly tens of px tall — i.e. *fragments*, not setup waste, dominate (the measured world1
profile: per-pixel coverage/election loop ≈90% — NaniteRaster.ts:264-271). R1: ~π·(70²−30²)×30/m²×9 ≈ 3.4M
tri. R2/R3 are card counts in the tens of thousands. Fragment cost = screen coverage × overdraw; the election
absorbs overdraw at 1-2 u32 atomics per fragment with an early-loss guard (NaniteRaster.ts:962-975) plus the
loss-exact `?f2b` pre-read that skips depth work for losing fragments (NaniteRaster.ts:979-989). The shipped
system survives the same population on the *slower* lane (full MeshStandard shading per overdrawn fragment,
2-8×, VegPrepass.ts:4-8) — the election lane strictly reduces per-fragment work. Hard caps ride the existing
compact-region pattern (GRASS_CAPS 524k/1M/1.8M — GroundRing.ts:97; indirect clamp GroundRing.ts:762-771)
recast as per-band patch-count caps in the cull.

---

## 2. Density / coverage spec — what "zero quality sacrifice" means, checkably

The new lane must reproduce, verbatim, every tuned behavior of the shipped system (shotdiff-gated per §7),
and adds capabilities the old lane cannot have. The checklist:

**Parity (each item maps to shipped code that defines "correct"):**
1. Per-blade geometry with rounded-cross-section normals to 30 m; 5-blade clump overlap ("lush" reads from
   blade overlap, not density — GroundRing.ts:207-212; GoT-derived ±38° normal trick GroundCover.ts:39-52).
2. Scruff floor: nothing within ~12 m is bald (density max-floor, GroundRing.ts:468-473).
3. Biome/water/canopy/slope density law: bank margins above the ACTUAL water surface (not the carve apron —
   GroundRing.ts:446-461), canopy thinning ×(1−0.45·canopy), snow/steep hard gates (GroundRing.ts:462-476).
4. Coverage conserved to 155 m+ by thin×widen (GroundRing.ts:100-110); silhouette carried to 265 m by
   super-tufts (GroundRing.ts:118-125); color-matched splat beyond (TerrainMaterial.ts:202-213).
5. Shading: base→tip albedo ramp, patch-scale (~1.6 m) dryness drifts, shade-grown darkening under canopy,
   tip-weighted translucent backlight, tip-AO (GroundRing.ts:882-911; grassTranslucency VegMaterials.ts:62-64),
   and the terrain-normal blend so swards light like their hillside, hardening with distance
   (GroundRing.ts:849-880).
6. Wind: cantilever tip² bend + lean² sward-flattening + distance-faded shimmer, riding the SAME gust field
   as the trees so meadow waves line up with canopy surges (GroundRing.ts:810-842, Wind.ts:28-29, 74-104).
7. Far-field directional sheen on the splat (TerrainMaterial.ts:266-284).

**Upgrades (new capabilities, impossible in the HW lane):**
8. HZB occlusion — meadow behind ridges/dense stands never rasters (the ring only frustum-tests,
   GroundRing.ts:394-403, 487; the hier cull occlusion-tests everything it feeds — NaniteFrame.ts:441-444).
9. Exactly one shade per covered pixel with unified depth — grass participates correctly in TRAA reprojection,
   GTAO, SSCS contact shadows, and water's depth-tested refraction/SSR (PostStack.ts:520-536, 196-207,
   424-470; water depth interop proven at N4-C0 — NaniteRaster.ts:884-891).
10. Blade-on-ground contact shadows for free once grass depth is in the vis buffer (SSCS marches scene depth
    ≤240 m toward the sun — PostStack.ts:424-470).
11. Trample/interaction field (§6) — the shipped system has none.
12. The @invariant depth-prepass twin and its Metal invariance hazard are deleted outright
    (VegPrepass.ts:28-56).

---

## 3. Integration anchors (all verified at HEAD)

| Seam | Anchor | What the grass work does there |
|---|---|---|
| Election | 24-bit depth key << 8 \| id8 atomicMax, winner side-stores 25-bit id (NaniteRaster.ts:327-336, 952-975) | Nothing new — grass tris carry the standard payload (itemIdx 23b \| localTri 8b, NaniteRaster.ts:9-13). No id-space cost, no new namespace (bit31 stays voxel — NaniteVoxelRaster.ts:79-85). |
| Two-sided raster | MESH_FLAG_TWO_SIDED + orientForRaster re-winds back-faces in place (GeometryRegistry.ts:174-177, NaniteRaster.ts:105-126) | Register patches two-sided; each blade/card triangle exists ONCE. |
| SW/HW routing | 16 px bbox limit routes big/near tris to the HW vertex-pull queue, HW_CAP 2M sized by the 876k-tri leaf finding (NaniteRaster.ts:94-103, 1002-1009) | Tall near blades overflow to HW exactly like leaf needles today; measure hwTris (§7 S3 gate). |
| Registry | `registerMesh(src, 'grass', { transformChannel:'grass', twoSided:true, aggregate:true })` — the leaf-head call shape (GeometryRegistry.ts:216-223, 1102-1105, 1260) | Patch library registration; tip ramp/hue baked into vdata like leaf/rock (unpackVdata consumers NaniteResolve.ts:494-497, 713-716). |
| Resolve branch budget | Two Discard-partitioned passes because of Metal's ≤10 storage buffers/stage; tri pass = 8 bindings, vox = 7 (NaniteResolve.ts:305-319). The one measured way resolve stops being free: demote-forcing texture subgraphs collapsing occupancy (37.5 ms vox cliff — NaniteResolve.ts:439-445) | Grass = matClass 5 branch in the TRI pass replacing the gray fall-through (NaniteResolve.ts:872-879): ALU-only + ONE explicit-LOD normalTex tap (the ring already samples it lod-0 — GroundRing.ts:859-865), zero new storage buffers ⇒ stays in the measured-free regime (lighting+shadows+decode ≈ FREE — docs/perf-runs/2026-07-02-attribution-and-waves.md). Cheap-path law beyond ~30 m: single-vertex normal, no 3-vert interp (the ?resfar pattern, NaniteResolve.ts:737-754). Backlight joins the shared `lit` block exactly as leaf/vox do (NaniteResolve.ts:1003-1020). |
| Wind | One field: gustAt/gustLagAt/windExposure (Wind.ts:74-104); grass response terms (GroundRing.ts:810-842, Wind.ts:28-29) | New `TRANSFORM_CHANNEL.grass` block in makeCtx beside trunk (NaniteFetch.ts:202-224) and leaf (232-252); per-instance field taps once per cluster via wgcache — extending TrunkWindFields MUST extend the shF broadcast (explicit note at NaniteRaster.ts:446-449). Raster⇄resolve bit-identity is automatic: both consume the same makeFetch (NaniteResolve.ts:219-227, NaniteRaster.ts:224-226). Never TSL cameraPosition for distances — the cascade-camera trap (VegInstance.ts:40-53); use the wind camPos uniform (NaniteFetch.ts:44-48). |
| Placement data | The cull fields the ring samples today: biomeTex/fieldsTex/normalTex, sampleHeight, sampleWaterYNearest, canopyAt (GroundRing.ts:432-453) — all resident, no streaming | Patch instance generation samples the same fields; residency = the toroidal wrap at patch granularity (slot ↔ nearest congruent world cell, params re-derived from pcg(worldCell) — GroundRing.ts:5-11, 139-149): zero uploads, instance-buffer rewrite only on re-centering. |
| Shadows — receive | Resolve PCSS at reconstructed wp / half-res upsample (NaniteResolve.ts:922-955) | Automatic for matClass 5. |
| Shadows — cast | Clipmap: hollow rings, per-level minPx cull, exact-VP cache (NaniteShadowClip.ts:17-26, 113-118, 133-136); per-head maxDist exists (WorldRegistry.ts:484-485) | Patches cast into the FINEST level only (shadow maxDist ≈ E₀); beyond it grass shadow is sub-texel by construction. Contact range owned by SSCS (PostStack.ts:424-470). |
| Frame order | cull → world1 (owns vis clear) → HZB → shadow clips → shadowHalf → post.render() scene pass (resolve −1000, vox −999) → post chain (NaniteFrame.ts:440-495, NaniteResolve.ts:1088-1102, PostStack.ts:133-165) | Grass rides world1 + the tri resolve pass; no new passes, no order changes. Water (30-water-plan) draws after resolve depth and therefore correctly over/behind grass automatically. |
| TRAA | Analytic camera reprojection from depth; self-motion → variance clipping (PostStack.ts:493-536) | Same contract leaves have; blade flutter distance-fades (Wind.ts:202-204, GroundRing.ts:829-838) so far grass never feeds shimmer. |
| Far-field handoff | FarTiles handoff-matching discipline (brick size matched at the boundary — FarTiles.ts:13-16) | The R3→R4 handoff must match the splat's patch-dryness field (already shared: the splat uses the same ~1.6 m patch noise — TerrainMaterial.ts:206-213 vs GroundRing.ts:787-788). |

---

## 4. Frame-budget ledger (per pose, inside locked-60 with the forest)

Grass line items, gross, at canonical config. "Reclaim" = deleting the legacy lane (GroundRing draws +
depth-prepass twin + its per-fragment MeshStandard shading + three.js draw overhead) from the nanite build —
a strictly negative line in any integrated comparison.

| Line item | Eye meadow (worst) | Eye forest-interior | Oblique | Aerial | Bound mechanism |
|---|---|---|---|---|---|
| Cull: ≤~4.7k patch instances → clusters after DAG cut | 0.1–0.3 ms | <0.1 | <0.1 | <0.1 | cluster count ∝ 1/d²; HZB kills occluded meadow; hard patch caps (§1.3) |
| Raster fragments (coverage × overdraw through the election) | **1.5–3.0 ms** | 0.2–0.5 (canopy occludes floor) | 0.3–0.8 | ≤0.3 | ≥1 px law (§1.1); early-loss guard + f2b (NaniteRaster.ts:962-989); thin×widen conservation |
| Resolve grass branch | ~0 | ~0 | ~0 | ~0 | ALU-only + 1 explicit-LOD tap, 0 new buffers — the measured-free regime (NaniteResolve.ts:439-445) |
| Shadow cast (finest clip only) | 0.2–0.5 ms | ~0.1 | ~0.2 | ~0.1 | ring cull + minPx + VP-equality cache (NaniteShadowClip.ts:113-118, 133-136) |
| Interaction field (§6) | ≤0.05 ms | ≤0.05 | ≤0.05 | ≤0.05 | fixed 128² stamp+relax kernel |
| Legacy reclaim (GroundRing + VegPrepass out of nanite build) | **negative** | negative | negative | negative | — |
| **Net grass envelope** | **≈ 2.0–3.5 ms cap** | **≈ 0.3–0.6 ms** | **≈ 0.5–1.0 ms** | **≈ 0.3–0.5 ms** | every lever individually cappable: thin curve, band edges (DAG τ), L2 promotion distance, shadow maxDist, patch caps |

Context lines: forest isolated med 18.9 / 37.2 / 16.5 (eye/oblique/aerial); post pool ≈ 6 ms; water
(30-water-plan) +0.6–1.5 typical. Honest statement the evaluation must hold us to: grass as specified costs
**~2-3.5 ms in a worst eye-level open-meadow pose and well under 1 ms in forest/aerial poses**, does not move
the existing eye/oblique gap (owned by the foliage-coverage workstream), and every line has a named dial. The
16.7 ms law is met by the whole-frame program (foliage reduction + these budgets), and grass structurally
*replaces* pixels it covers (meadow pixels stop being terrain-branch pixels — one election, one shade) rather
than stacking a second full-screen cost on top, unlike the legacy lane which re-shades over the resolve's
output today. GoT's full-meadow frames on PS4 ([GDC 2021](https://gdcvault.com/play/1027033/Advanced-Graphics-Summit-Procedural-Grass))
and UE5's PS5 Witcher-4 ground cover ([PC Gamer](https://www.pcgamer.com/hardware/the-witcher-4s-leafy-glory-is-all-down-to-epics-nanite-foliage-and-largely-so-is-the-fact-it-can-run-ray-traced-at-60-fps-on-a-lowly-ps5/))
bound this envelope from above on weaker silicon.

---

## 5. Resolve grass branch — exact spec

Replace the matClass-5 fall-through (currently `palette` gray — NaniteResolve.ts:872-879) with a branch cloned
from the leaf branch (NaniteResolve.ts:692-758) minus everything expensive:

- **Inputs already bound** in the tri pass: verts/indices/instances/clusters/meshes (NaniteResolve.ts:312-313).
  Species/field tint via mesh word 7 like leaf (NaniteResolve.ts:697-702); tip parameter + per-blade hue from
  baked vdata (unpackVdata idiom, NaniteResolve.ts:713-716).
- Albedo = fresh/dry tip ramps × ~1.6 m patch dryness × canopy shade-darkening — a port of
  GroundRing.ts:882-905 (canopyAt is already available in the resolve — NaniteResolve.ts:980-981).
- Normal = blade normal pulled toward terrain normal, harder with distance (GroundRing.ts:849-880): ONE
  explicit-lod normalTex tap (`texture(..., 0)`, the ring's own idiom GroundRing.ts:859-865 — no implicit
  derivatives, so no demote pressure per the vox-cliff rule NaniteResolve.ts:439-445).
- Backlight = grassTranslucency's tip-weighted forward scatter (VegMaterials.ts:53-64) added in the shared
  `lit` block via the existing leaf/vox gate pattern (NaniteResolve.ts:1003-1020).
- Tip-AO fold (GroundRing.ts:909-911) multiplies the indirect term like rock/bark ao (NaniteResolve.ts:891).
- Distance law: beyond ~30 m take the cheap path — single-vertex normal, no makeCtx, no 3-vert interp
  (NaniteResolve.ts:737-754 pattern). Sub-pixel-width blades at range shade indistinguishably from it.
- **Zero new storage buffers, zero implicit-derivative samples** ⇒ the branch lands in the tri pass at ≈ zero
  marginal cost (§3 resolve row). Debris (class 6) later fits the same slot the same way; the NEXT genuinely
  buffer-hungry class opens a third Discard partition instead of pushing tri-pass registers
  (NaniteResolve.ts:305-319 precedent).

---

## 6. Interaction plan — wind + trample

**Wind (one field, N consumers — the codebase law).** The global gust field stays the single source: two
advected fbm octaves (85 m fronts @10.5 m/s + 17 m detail — Wind.ts:74-82), canopy shelter (Wind.ts:100-104).
Grass response = the shipped model verbatim: cantilever bend ∝ tip², lean² "strong wind flattens the sward",
per-blade shimmer faded by ~120 m (GroundRing.ts:810-842; leaf-flutter fade precedent Wind.ts:202-204). GoT
bends Bézier control points with a 2D gust field the same way; UE5.7's Nanite Foliage explicitly replaces
per-vertex WPO with cheap shared animation for exactly our reasons (culling bounds + cost)
([GoT GDC 2021](https://gdcvault.com/play/1027033/Advanced-Graphics-Summit-Procedural-Grass);
[Epic Nanite Foliage](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-foliage)).
Implementation: the `grass` case in makeCtx computes per-instance amp/phase once per cluster (field taps in
the wgcache broadcast — extend shF per the NaniteRaster.ts:446-449 note); fetchWorldVert applies the
per-vertex tip² profile. The resolve inherits it through the shared fetch (NaniteResolve.ts:219-227) — the
same contract that keeps trunk wind bit-identical across raster and resolve (NaniteRaster.ts:224-226).

**Trample (the GoT displacement buffer, ported).** A camera-following ~128² RG(bend dir)+B(strength)
ping-pong storage texture over ±20 m. Per frame, one tiny compute pass: stamp capsules (player, deer, boar),
then relax toward rest with a damped spring so blades overshoot and settle instead of snapping back, and
trails persist and decay ([PlayStation Blog on GoT's grass displacement](https://blog.playstation.com/2021/01/12/how-stunning-visual-effects-bring-ghost-of-tsushima-to-life/)).
Consumers: the grass channel adds the sampled bend to the cantilever term pre-projection (one extra texture
tap in the per-cluster wgcache slot for the patch center + a cheap per-vertex refinement for near patches);
the legacy lane (while it lives) can read the same texture in `grassMaterial`'s wind block
(GroundRing.ts:813-842) so the A/B stays fair. Budget ≤0.05 ms (fixed 128² kernel — same class as the
caustics bake, Caustics.ts:10). The render-below-camera alternative costs a raster pass we don't have; capsule
stamping is cheaper and is what shipped ([hexaquo grass series survey](https://hexaquo.at/pages/grass-rendering-series-part-2-full-geometry-grass-in-godot/)).

---

## 7. Staged build plan with probe gates

Gate discipline (memory `verify-user-observable-output` + `nanite-perf-canonical-config-and-baseline`): every
stage gates on (a) the standard isolated probe at canonical 200k @ 2268×1473, eye/oblique/aerial, ms→p0.95,
AND (b) the user-observable check named below — never machinery reports. A band/stage migrates only when it
wins BOTH gpuWall and visual parity vs the live GroundRing control.

- **S0 — resolve branch first** (`?grassreg`): register a debug grass patch, implement the matClass-5 branch
  (§5). GATE: attribution probe shows resolve delta ≈ 0 (the free-regime claim proven before any raster load
  exists); patch shades with ramp/translucency/terrain-normal blend in a screenshot.
- **S1 — grass transform channel** in NaniteFetch + wgcache slots (§6). GATE: `?audit=1` orphan count 0
  (raster⇄resolve agreement tooling — NaniteRaster.ts:147-149); wind A/B (`?nanwind=0` analog) shows motion.
- **S2 — patch library + aggregate DAG** (BuildAggregateDag remove+grow over blades; L0→L3 per §1.2 with the
  authoring-law table checked numerically at build time: assert every level's min triangle width ≥1.5 px at
  its selection distance). GATE: single-patch shotdiff vs a GroundRing patch at 5/15/30/60/120 m ≤ noise;
  emitted-tri telemetry confirms no rung emits sub-pixel width (scar-style counter precedent,
  NaniteRaster.ts:317-322).
- **S3 — toroidal patch residency + density-law port** (§2 items 2-4); delete GroundRing+VegPrepass from the
  nanite build behind a flag. GATE: net gpuWall vs the GroundRing control ≤ +2.5 ms eye-meadow / ≤ +0.5 ms
  forest-interior & aerial *gross*, and ≤ control *net* (reclaim counted); hwTris within HW_CAP headroom
  (NaniteRaster.ts:94-101); USER-OBSERVABLE: continuous sward to horizon, no rings, no bald near field,
  meadow occluded behind ridges (HZB win visible in the isolated probe).
- **S4 — far handoff**: verify/port the grass-field + sheen terms in the nanite terrain branch
  (NaniteResolve.ts:446-456 inputs; TerrainMaterial.ts:202-213, 266-284). GATE: shotdiff band straddling the
  R3→R4 edge is tonally invisible (band-mean ΔE below the FarTiles handoff precedent — FarTiles.ts:13-16).
- **S5 — shadows**: cast into finest clip level with shadow maxDist ≈ E₀. GATE: shadow probe ≤ +0.5 ms;
  static-frame VP cache still hits (NaniteShadowClip.ts:113-118); SSCS blade contact visible in a macro shot.
- **S6 — trample field** (§6). GATE: ≤0.05 ms; walk-through leaves persistent decaying trails on video probe.
- **S7 (contingency, only if S3's near band measures poorly)**: procedural near-band generator as a third
  queue class with screen-clamped ribbon widths (§1.2 escalation; voxel-queue precedent
  NaniteRaster.ts:204-215). Not scheduled — the patch route is expected to hold.

---

## 8. Risks + mitigations

| # | Risk | Evidence anchor | Mitigation |
|---|---|---|---|
| R1 | SW/HW routing mix at eye level — tall near blades exceed the 16 px SW box and flood the HW queue | NaniteRaster.ts:94-103; leaf precedent 876k hwTris handled | Measure hwTris at S3; HW_CAP 2M has headroom; if flooded, split L0 blade strips into shorter segments (more, smaller tris — still ≥1 px by the law) |
| R2 | Election atomic contention under meadow overdraw | The guarded-max already carries 876k-tri leaf loads (NaniteRaster.ts:94-101, 962-975) | Early-loss guard is default; `?f2b` loss-exact pre-read (979-989); thin×widen bounds fragment count |
| R3 | Resolve occupancy regression from the grass branch | The 37.5 ms vox cliff (NaniteResolve.ts:439-445) | Branch is ALU-only + explicit-LOD single tap by spec (§5); S0 gates it before any load exists |
| R4 | LOD pop replacing the tuned dither crossfades | GroundRing.ts:151-187 double-append machinery being retired | Aggregate DAG's crack-free bit-exact sibling cut (BuildAggregateDag.ts:12-19); fallback: IGN dither at cut boundaries inside the raster; S2 shotdiff ladder gates it |
| R5 | TRAA shimmer from blade self-motion | PostStack.ts:509-511 (self-motion → variance clipping) | Flutter distance-fade (GroundRing.ts:829-838, Wind.ts:202-204); same contract leaves ship with today |
| R6 | Patch-DAG memory / instance-buffer rewrite cost on re-centering | Toroidal re-derivation is zero-upload for params (GroundRing.ts:5-11); patch instances are ~4.7k slots | Rewrite only moved slots (clipmap diff); if measured hot, adopt the TerrainStreamer slot-pool diff (TerrainStreamer.ts:371-467) or escalate to S7's in-register generator |
| R7 | Far handoff tone drift (R3→R4) | The splat is palette-matched but the DAG look at 150-265 m is new | Shared patch-dryness noise (TerrainMaterial.ts:206-213 ↔ GroundRing.ts:787-788); S4 ΔE gate |
| R8 | Stale-HZB cut inflation on fast camera moves inflating grass clusters | Known moving-camera behavior (memory `base-raster-is-the-bottleneck`) | Same exposure as all geometry; patch caps bound the blast radius; benefits from any global stale-HZB fix |
| R9 | Grass casters bloating the fine shadow clip | NaniteShadowClip.ts:133-136 minPx | Shadow maxDist ≈ E₀ (finest ring only); blades are sub-texel beyond by construction; S5 gate |

---

## 9. Sources

- [GDC 2021 — Procedural Grass in Ghost of Tsushima (Eric Wohllaib, Sucker Punch)](https://gdcvault.com/play/1027033/Advanced-Graphics-Summit-Procedural-Grass) · [slides PDF](https://archive.thedatadungeon.com/ghost_of_tsushima_2020/documents/gdc_2021/gdc_2021_procedural_grass_in_got.pdf) · [PS Blog — GoT grass displacement/springback](https://blog.playstation.com/2021/01/12/how-stunning-visual-effects-bring-ghost-of-tsushima-to-life/)
- [Guerrilla — GPU-Based Procedural Placement in Horizon Zero Dawn](https://www.guerrilla-games.com/read/gpu-based-procedural-placement-in-horizon-zero-dawn) · [GDC 2017 talk](https://gdcvault.com/play/1024700/GPU-Based-Run-Time-Procedural) · [GDC 2018 — Between Tech and Art: The Vegetation of HZD](https://www.gdcvault.com/play/1025530/Between-Tech-and-The)
- [Epic — Nanite Foliage documentation (assemblies / voxels / skinning)](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-foliage) · [PC Gamer — Witcher 4 demo, Nanite Foliage at 60 fps on PS5](https://www.pcgamer.com/hardware/the-witcher-4s-leafy-glory-is-all-down-to-epics-nanite-foliage-and-largely-so-is-the-fact-it-can-run-ray-traced-at-60-fps-on-a-lowly-ps5/) · [UE forum — Landscape Grass vs Nanite overdraw guidance](https://forums.unrealengine.com/t/landscape-grass-nanite/237794)
- [vkguide — Project Ascendant: clutter in the unified GPU-driven cull/draw fabric](https://vkguide.dev/docs/ascendant/ascendant_geometry/) · [hexaquo — full-geometry grass series (interaction alternatives)](https://hexaquo.at/pages/grass-rendering-series-part-2-full-geometry-grass-in-godot/) · [GodotGrass — GoT-style open reimplementation](https://github.com/2Retr0/GodotGrass)
- In-repo: docs/perf-runs/2026-07-02-attribution-and-waves.md (resolve ≈ free; pose costs) · docs/perf-runs/2026-06-26-base-raster-review-handoff.md (sub-pixel emit disease) · 30-water-plan.md (single-layer-water companion; shared depth story)
