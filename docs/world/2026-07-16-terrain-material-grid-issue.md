# Terrain material sub-metre grid artifact

Date: 2026-07-16

Status: unresolved; the runtime material-detail work is uncommitted and awaiting visual acceptance.

## Scope and constraints

This record covers the visible terrain-material artifact in the Estonia world runtime. It does not cover terrain asset generation or the content of the packed terrain-height rungs.

The work has been constrained as follows:

- Do not change anything in `asset-gen`.
- Do not resize `noiseBakeA` or `noiseBakeB`.
- Do not resize the streamed biome, surface-fields, or soil planes.
- Keep runtime, GPU-memory, and texture-fetch impact small.
- Preserve material detail through material overlaps, not only inside single-material regions.
- Concentrate extra close-range work near the camera; distant terrain does not need the same centimetre-scale material work.
- Visual acceptance is performed by the user.

## Visible defect

The terrain surface contains an axis-aligned grid of rectangular or square colour regions. The user estimates the visible regions at approximately 10–20 cm across in the inspected close views.

The defect has the following observed appearance:

- Green, tan, gray, and dark terrain colours form connected rectangular strips, blocks, corners, and stepped outlines.
- Some boundaries are softly feathered over a short distance, but the overall shapes remain aligned to a square grid.
- Adjacent blocks can read as different terrain materials rather than as continuous variation within one material.
- The grid is visible across mixed grass, exposed ground, and rock-like colour regions.
- The grid remains visible when the underlying fine texture becomes sharper.
- The underlying terrain mesh resolves finer geometric structure than the visible material grid.
- The grid is especially clear in a close or top-down view of the raw terrain albedo.
- The same grid pattern remains visually stable while finer terrain-height LODs arrive. The height and silhouette gain detail, while the material grid does not visibly change.

The original wide ground-level view showed the grid as coarse rectangular tan and dark patches over a more detailed green terrain surface. Later close views showed the same pattern as explicit block-shaped material regions.

## Debug-buffer observations

### Raw albedo

`nandbg=albedo` displays the grid strongly. In this view, lighting is removed and the rectangular green/tan/dark regions remain visible in the terrain colour itself.

### Normal output

`nandbg=normal` did not show an equivalently obvious grid to the user. The normal view contained continuous normal variation, while the pronounced material-colour rectangles seen in albedo were not visually reproduced at the same strength.

### Height LOD transition

The user observed the terrain before and after the detailed height LODs arrived. The terrain geometry became more detailed. The visible material-grid pattern remained consistent through that transition.

## Runtime detail change attempted

The original cosmetic micro carrier used a nominal 0.19 m feature scale. A runtime-only 4× detail change was added in `src/render/TerrainMaterial.ts`, using a nominal 0.0475 m feature scale.

That change included:

- A 4× finer scalar material-detail carrier.
- A 4× finer cosmetic normal-detail carrier.
- Fine variation applied to the final terrain colour so single-material interiors receive the added detail.
- Fine variation applied inside existing rock, scree, grass, forest, and categorical-material overlaps.
- Endpoint-preserving overlap modulation so the added detail does not extend a material outside its existing support.
- Distance gating for the fine scalar and normal texture reads.
- A reduced fine-normal height coefficient paired with the increased carrier frequency.
- No texture-dimension changes.
- No new texture resources or bindings.
- No `asset-gen` changes.

Observed result:

- The underlying terrain textures looked more detailed to the user.
- The rectangular material-grid pattern remained visible.
- The material-grid complaint was not resolved.

## Close-range transition adjustment attempted

A temporary close-range widening of material slope-transition thresholds was tried to soften material changes near the camera.

Observed result:

- The user reported zero visible improvement in the grid pattern.
- The temporary adjustment was fully reverted.

## Fine-detail distance-frame adjustment attempted

The fine-detail branch and amplitude fade originally used full three-dimensional camera-to-surface distance. A temporary correction changed the fine albedo carrier, overlap modulation, and fine normal carrier to a planar XZ distance with the same 32–40 m band used by the finest packed terrain LOD.

Observed result:

- The user reported no visible change in the grid pattern.
- The planar-distance correction was reverted.

## Material-component ablations attempted

All ablations below were inspected in the raw terrain albedo view.

### Soil modulation disabled

The terrain was inspected with `soil=0`, disabling the runtime soil-pedology modulation block.

Observed result:

- The grid remained visible.
- The user reported the same problem.

### Categorical land-cover overlay disabled

A scoped `landcover=0` diagnostic was added in `src/nanite/shade/NaniteResolve.ts`. It disables the categorical land-cover material-overlay block.

Observed result:

- The grid remained visible.
- Combining this with `soil=0` did not remove or visibly change the grid pattern.

### Scalar meso carrier neutralized

A scoped `meso=0` diagnostic was added. It replaces the scalar 1.45 m meso carrier with its neutral midpoint while leaving material selection and cosmetic normal detail enabled.

Observed result:

- The user reported absolutely no visible change.
- The grid remained visible.

## Current source state

The current uncommitted terrain-material work is limited to these runtime files:

- `src/render/TerrainMaterial.ts`
- `src/nanite/shade/NaniteResolve.ts`

The current state retains:

- The 4× finer runtime material-detail implementation.
- The `landcover=0` categorical-overlay diagnostic.
- The `meso=0` scalar-carrier diagnostic.

The current state does not retain:

- The temporary close-range slope-transition adjustment.
- The temporary planar-distance correction.

No attempted change before 2026-07-16 removed or visibly reduced the rectangular material-grid artifact.

## SUPERSEDED diagnosis (2026-07-16 — wrong, kept for the record)

A first diagnosis blamed the biome plane's 2 m texel lattice (vegDensity/canopy
through `biomeAt`) and dismissed the user's 7–20 cm size estimate as a scale
misread. The user's estimate was right. The decisive facts it could not explain:
the blocks keep the SAME world size at every camera distance and across
height-LOD arrival, at ~7–20 cm. The 2 m landcover quantization is real but is a
separate, coarser artifact; the warp section below now targets only that.

## ROOT CAUSE (2026-07-16, final — measured and RENDERED from the cooked bytes)

The grid is IN THE COOKED HEIGHT DATA. A pure-numpy hillshade of a decoded
32 m × 32 m patch of L0 chunk `2430_1490` (no engine code whatsoever) shows the
artifact directly: sharp axis-aligned step-seam LINES on dyadic 4/8/16-texel
boundaries (25 cm / 50 cm / 1 m), with per-region density — the fingerprint of
a stepped multi-resolution base assembly — over otherwise organic fibrous
micro-detail. The user-visible "fuzzy ~10 cm grid" is these seams (smallest
cells + shadowed seam widths ≈ 10–25 cm), and the coarse "stepped outlines" are
the same hierarchy's 1 m tier. The synth detail itself is NOT the problem
(smooth, lag-1 autocorrelation 0.76); the stepped assembly/projection is
(`asset-gen/src/assetgen/process/micro_synth.py`: "the authoritative (or
stepped-upsampled) 1 m source window"; "caller adds to upsampled base then
applies the hard block-mean projection for exact 1 m consistency"). Measured on
decoded chunks (pure Python, no renderer):

- Every 1 m block is a noisy plateau: within-1m-block std 3.2 cm vs 3.9 m
  global; height deltas at 1 m block boundaries are 5–20 cm vs 3.7 cm interior
  (1.4×–4.8×, both axes, every chunk sampled; worst on relief). Secondary
  harmonics at 50 cm (2.5×) and 25 cm (1.7×).
- Headless captures at the spawn show the terrain surface as literal 1 m cubes
  with checkered tops; `?grass=0` is identical (not the grass lane), and the
  state persists through 3600 settle frames (not a streaming transient).
- The renderer draws the bytes honestly (lattice honesty A11). All downstream
  "grid" appearances are this data lattice expressed through shading: slope
  spikes at plateau walls flip the material class windows (the tan/dark
  rectangles), lighting/shadows draw the fuzzy value grid on uniform grass, and
  the walls remain at every camera distance and every mesh LOD — which is why
  NO runtime change (soil/landcover/meso ablations, 4× carriers, control-plane
  warp, class-slope rework) could ever affect it.

FIX (asset-gen lane — the micro-synth cook): the seams appear at EVERY dyadic
assembly boundary (25 cm / 50 cm / 1 m), so every stepped operation in the
base-assembly chain must become smooth: upsample the coarse base with a smooth
interpolant (bicubic/bilinear), interpolate across any multi-resolution tile
joins, and enforce the source-consistency constraint with a smoothly
distributed correction at every hierarchy level (a per-block CONSTANT
correction re-prints the lattice at that block size). Re-cook; no runtime change needed
for this artifact. Cook gate (reproduce in ~20 lines of numpy): decode an L0
height chunk (LAC1 enc1: 56 B header `<4sBBBBiiHxxIddffII`, zlib payload,
undelta2d — column 0 cumsum down then rows cumsum right mod 2^16, ×qscale
+qoffset) and compare mean |Δh| across 16-texel (1 m) boundaries vs interior
texel pairs: the lattice is gone when the ratio ≈ 1 (currently 1.4–4.8).

Separate, milder, real artifacts that are NOT this bug: the 2 m
landcover/vegDensity texel steps in material selection, and mid-stream
coarse-fill replication (cubes until fine chunks land).

## SUPERSEDED (2026-07-16): micro-lattice slope through the class windows

The grid is the height L0 texel lattice of the new micro cook, printed through
the slope-thresholded material class weights.

- The live client resolves `latest.json` → manifest `m/b7afac38c167af79`
  (cook_rev 500), whose height L0 is a 6.25 cm lattice (`chunkMeters 128 /
  chunkRes 2048`, format 1, lods 0/1/2 at 6.25 cm / 25 cm / 1 m).
- Decoded chunks at/near the spawn (L0 `2430_1490`, `2434_1490`, `2430_1494`)
  measure: adjacent-texel height delta mean 4.4 cm; per-texel CD slope mean
  0.52–0.74; **median adjacent-texel slope jump 0.25–0.28 — the full width of
  the grassW window (0.22–0.5)**; 51–56 % of adjacent texel pairs jump more
  than 25 % of that window. At L1 (25 cm) the jump p50 is 0.067 (15 % flips);
  at L2 (1 m) 0.011 (1 % flips) — the smooth field the thresholds were tuned on
  (the previous cook_rev 4 had a 1 m L0).
- Heights are quantized at 1 cm (`qscale`), so slope steps in units of 0.08 —
  29 % of the grassW window per least-significant bit: part of the flipping is
  compression noise, not relief.
- `fieldNormalSlopeHot` reads slope via `cdTaps` — a nearest-texel snapped
  stencil, piecewise-CONSTANT per texel — so the class flips print as
  axis-aligned 6.25 cm rectangles that clump into the observed 7–20 cm strips.
- This explains every observation: fixed world size at all distances (the L0
  window spans ±64 m of the camera — every close inspection sits on one
  lattice); unchanged by height-LOD arrival (the lattice belongs to the plane
  window, not the fills, and all rungs carry the same rough source); albedo-only
  (class weights read the RAW plane slope, while the displayed normal buries its
  per-texel facets under cosmetic bump detail); immune to `soil=0` /
  `landcover=0` / `meso=0` / 4× micro carriers / threshold widening (none
  touched slope; widening a window cannot un-quantize its input); "mesh resolves
  finer structure than the grid" (a 4 cm ledge is subtle geometry but a full
  material flip).

## Two-scale class slope (2026-07-16, REVERTED — user observed zero effect)

Implemented and fully reverted the same day. Zero visible effect, correctly so:
the grid is in the height bytes (root cause above), not in how slope is
sampled. The slope statistics measured in the superseded section are real and
may still merit a class-slope revisit AFTER the cook is fixed (the re-cooked
data changes those statistics). Original design for the record:

Runtime-only, in `src/nanite/world/TerrainField.ts` + `src/render/TerrainMaterial.ts`,
compile-gated on `heightLevels[0].texel < 0.5` (fine-lattice cooks only — the
generated world and 1 m cooks compile their exact previous graphs):

- `planeGradientSmooth`: C0-continuous height gradient (2×2 cell mean-gradients
  bilinearly interpolated at the continuous position, 3×3 stencil / 9 loads) —
  never piecewise-constant per texel, so thresholds cannot print the lattice.
- `fieldClassSlopeHot`: material-SELECTION slope from the first ≥0.5 m height
  level (rev 500 → the 1 m L2; measured jump p50 0.011 = invisible through the
  windows). Consumed by rockW/screeW/grassW/forestW/riverW, pondK, the soil
  `exposed` gate, the landcover fieldW/peatW slope gates and wallK. Restores the
  intended class balance (with micro slope, HALF of all ground crossed the grass
  window — the terrain read as barren mottle).
- `fieldReliefSlopeHot` + fine-relief exposure: the sub-metre material variation
  the micro cook is FOR, form-driven instead of lattice-driven — the top decile
  of the C0 fine gradient (real ledges/boulder faces: knees 0.85–1.35 bare,
  1.35–2.1 stone, ≥10× the quantization noise) thins grassW/forestW (bare
  soil/litter shows) and joins rockW (stone faces). Evaluated inside 60 m
  (sub-texel on screen beyond; fades before the L0→L1 window hand-off).
- The centimetre relief keeps SHADING through the unchanged ns/baseNormal and
  displacement/bump gates — it just no longer flips material identity per texel.
- Cost: +9 r32f loads always (class slope on the small 1 m plane) and +9 within
  60 m (fine relief); no VRAM, no new bindings, no URL knobs.

Predicted outcome: the axis-aligned grid is gone at every distance and LOD by
construction (no texel-quantized signal reaches albedo); grass/soil/rock class
boundaries follow metre-scale form; sub-metre bare/stone patches follow actual
micro-relief. The density knees of the fine exposure may need one taste pass.

## Control-plane de-grid warp (2026-07-16, REVERTED)

Written under the first superseded diagnosis; implemented and fully reverted
the same day (zero effect on the reported grid — root cause above). The 2 m
landcover/soil texel quantization it targeted is a real, separate, coarser
artifact; the technique remains a candidate for that issue later. Original
design for the record (`src/render/TerrainMaterial.ts` + `src/nanite/shade/NaniteResolve.ts`):

- One warped control coordinate `wxzB = wxz + ±0.75 m` world-anchored signed
  value-noise vector (~5.3 m features, 2 baked-noise fetches via the existing
  precision-safe `noiseUv` helpers), applied to the CONTROL taps only —
  `biomeAt`, `fieldsAt`, `biomeClassAt`, `soilAt` — so texel boundaries meander
  organically instead of tracing the lattice. The 2 m content variation stays
  (real ETAK/CHM data); only its boundary shape de-axis-aligns.
- Geometry-paired taps stay on the true `wxz`: height/normal/slope, and waterY
  (riverDepth = waterY − real surface height; warping one side would fabricate
  depth).
- ±0.75 m ≈ 0.37 texel — within the 2 m raster's own positional honesty; the
  endpoint-preserving property holds (a resample of the same field cannot leak a
  material beyond ~1 texel of its data support).
- `?bwarp=<m>` overrides the amplitude; `?bwarp=0` disables (exact old
  coordinate) — the live A/B also confirms the diagnosis, since the warp touches
  only the biome/control taps.
- Cooked sources only (rides `hasCanopy`): the generated world compiles
  `wxzB ≡ wxz` — bit-identical graph.
- Known limit: the fixed 0.75 m amplitude de-grids the near 2 m level only.
  Mid-field 8 m L1 texels keep their softer 8 m bilinear ramps. The
  durable other half is cook-side (fractional-coverage / finer landcover raster,
  the #114 α template) = open task #106, asset-gen lane; the warp composes with
  it rather than replacing it.
