/**
 * Terrain shading — near tiles AND the streamed far field (one path, distance-
 * gated). Reconstructed-pixel shading (NaniteResolve) samples the TerrainField
 * plane pyramid at any distance; the coarsest biome/height levels are country
 * floors (PlaneFill.ensureFloorCoversBox), so far terrain reads real fields.
 *
 * Splat classes are derived from CONTINUOUS fields (slope, snow, moisture,
 * rock exposure, zone masks) so everything filters cleanly; the quantized
 * biome id channel is only for scatter passes (read with textureLoad there).
 *
 * Macro–meso–micro law: every class gets a 2–50 m macro variation layer, a
 * ~1.5 m meso albedo/normal band, and a ~0.2 m micro normal band. Snow edges
 * are hash-dithered. Wet margins darken. Distant tiles re-amplify ridged noise
 * in the normal domain (distance-gated) so far mountains stay serrated (Pillar D).
 *
 * PERF: all repeated noise comes from the baked NoiseBake textures (was ~35
 * live noise evaluations per pixel ≈ 52 ms/frame; now ~14 filtered fetches).
 * Gradient channels are pre-derived, so bump/ridge detail is one fetch
 * instead of four finite-difference evaluations.
 */

import type { StorageTexture } from 'three/webgpu';
import {
  If,
  cameraPosition,
  clamp,
  float,
  mix,
  positionWorld,
  smoothstep,
  texture,
  transformNormalToView,
  vec2,
  vec3,
} from 'three/tsl';
import type { NF, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import type { TerrainField } from '../nanite/world/TerrainField';
import { hash12 } from '../gpu/noise/NoiseTSL';
import {
  PERIOD_FBM,
  PERIOD_RID,
  PERIOD_VAL,
} from '../gpu/passes/NoiseBake';
import { sunU } from './VegMaterials';
import { zoneMasks, type MacroParams } from '../world/MacroMap';
import { LAKE_LEVEL } from '../world/WorldConst';

export interface TerrainShadingInputs {
  /** the TerrainField plane set — THE terrain data source. Normal+slope =
   *  height-plane central differences (the retired normalTex's bake stencil,
   *  in-shader), snow/rockExposure/moisture/flow = ONE filtered fields-plane
   *  tap + vegDensity from the biome plane, riverDepth = DERIVED waterY − h
   *  (spec §3: not stored). */
  field: TerrainField;
  /** baked tileable noise (NoiseBake channel map) */
  noiseA: StorageTexture;
  noiseB: StorageTexture;
  mp: MacroParams;
  /** the biome plane carries the merged far-forest canopy in channels 2/3
   *  (Estonia) rather than snow/rockExposure (the generated world) — gates the
   *  canopy tint so the generated look is bit-identical. */
  hasCanopy: boolean;
  /** Diagnostic gate for the scalar 1.45 m meso carrier. False substitutes its
   *  neutral midpoint while leaving material selection and normal detail intact. */
  meso: boolean;
  /** the biome plane's classId channel (channel 0) carries ETAK land-cover ids
   *  (the cooked Estonia source) rather than the generated world's Biome enum —
   *  gates the classId→material block so the generated graph is compile-time
   *  BIT-IDENTICAL. Same cooked-source discriminator as hasCanopy (both ride the
   *  canopy layer's presence), named for its own meaning at the use site. */
  landcover: boolean;
  /** the soil plane (#116) carries the cooked Mullastikukaart substrate
   *  [texCore, stoniness, boniteet, texSkeleton] — true iff the source has a soil
   *  layer (Estonia). The generated world has no soil layer, so its graph compiles
   *  WITHOUT any soil node ⇒ BIT-IDENTICAL. Same cooked-source discriminator family
   *  as `landcover`/`hasCanopy`. */
  hasSoil: boolean;
  /** the geology plane (30a16ef: EGT categorical priors [bedrockFamily,
   *  surficialFamily, processFamily, coverageFlags] on the 2 m lattice, ids
   *  nearest-only) — true iff the source cooked one. Old manifests and the
   *  generated world have none, so their graphs compile WITHOUT any geology
   *  node ⇒ BIT-IDENTICAL. Same discriminator family as hasSoil/hasCanopy. */
  hasGeology: boolean;
  /**
   * surface context override (N4 nanite resolve): explicit world position +
   * camera position instead of the vertex-pipeline TSL singletons. The old
   * forward materials omit this and get `positionWorld`/`cameraPosition` —
   * bit-identical graphs by construction. Every other input above is already
   * world-space-parametric (textures fetched by world uv derived from wp).
   */
  surf?: {
    wp: NV3;
    camPos: NV3;
    /** Precision-safe normalized world coordinate for baked material noise.
     *  Streamed resolve supplies anchor-relative position plus a CPU-f64 world
     *  phase; forward/generated materials omit it and retain the old path. */
    noiseCoord?: (periodM: number) => NV3;
  };
}

export interface TerrainShading {
  colorNode: NV3;
  normalNode: NV3;
  roughnessNode: NF;
  /** final shading normal in WORLD space (for probe irradiance) */
  worldNormalNode: NV3;
}

/**
 * Micro-displacement constants — SHARED by the TerrainTiles vertex stage
 * (geometry) and the fragment normal counterpart below. fbm(2.6 m) rolls +
 * val(0.9 m) breakup + ridged(1.15 m) creases (rock-weighted); amplitude
 * fades out 45→85 m and is gated by slope/rockExposure so grass meadows
 * stay smooth under their blade carpet (veg sits on the undisplaced field).
 */
export const DISP = {
  base: 0.15,
  rock: 0.55,
  gravel: 0.3,
  fade0: 45,
  fade1: 85,
  sF1: 2.6,
  sF2: 0.9,
  sRid: 1.15,
  wF1: 0.55,
  wF2: 0.33,
  wRid: 0.62,
  ridBase: 0.25,
  slopeKnee0: 0.45,
  slopeKnee1: 0.95,
} as const;

/** Finest cosmetic material carrier. The packed terrain can resolve 0.0625 m
 *  geometry, so the old 0.19 m carrier visibly formed a coarser grid over the
 *  surface. Keep this slightly finer than the mesh and fade it with the LOD -2
 *  geometry band; macro/meso material structure remains unchanged. */
const MATERIAL_DETAIL_M = 0.19 / 4;
const MATERIAL_DETAIL_BLEND = 0.12;

export function buildTerrainShading(inp: TerrainShadingInputs): TerrainShading {
  const wp = inp.surf?.wp ?? (positionWorld as unknown as NV3);
  const camPos = inp.surf?.camPos ?? (cameraPosition as unknown as NV3);
  const wxz = wp.xz;
  const h = wp.y;
  const camDist = wp.sub(camPos).length();

  /** 1D band noise [0,1] along an arbitrary phase axis */
  const band = (phase: NF, lane: NF): NF =>
    texture(inp.noiseA, vec2(phase, lane).div(PERIOD_VAL)).x;

  // --- baked-noise coordinates (hoisted above the field taps — the control-
  // plane warp and relief lace below sample them) -------------------------------
  // Streamed scalar carriers use one continuous oblique projection. XZ remains
  // unchanged on a level surface (Y contributes only a constant translation),
  // while the unit-length Y shear gives vertical faces real surface variation.
  // Unlike dominant-axis box projection this has no orientation-switch seams;
  // unlike triplanar it retains exactly one texture fetch. Generated/forward
  // materials omit noiseCoord and compile their original XZ graph unchanged.
  const shearX = 0.754877666;
  const shearZ = 0.655865425;
  const noise3 = (periodM: number): NV3 =>
    inp.surf?.noiseCoord?.(periodM) ?? (wp.div(periodM) as unknown as NV3);
  const noiseUv = (periodM: number): NV2 => {
    const p = noise3(periodM);
    if (!inp.surf?.noiseCoord) return p.xz;
    return vec2(p.x.add(p.y.mul(shearX)), p.z.add(p.y.mul(shearZ))) as unknown as NV2;
  };
  /** value noise [0,1] at world feature scale `s` m */
  const val = (s: number, ox = 0, oz = 0): NF =>
    texture(inp.noiseA, noiseUv(s * PERIOD_VAL).add(vec2(ox, oz))).x;
  /** signed value noise [-1,1] */
  const valS = (s: number, ox = 0, oz = 0): NF => val(s, ox, oz).mul(2).sub(1);
  /** fbm-3 [0,1] */
  const fbmV = (s: number, ox = 0, oz = 0): NF =>
    texture(inp.noiseA, noiseUv(s * PERIOD_FBM).add(vec2(ox, oz))).y;

  // Control-plane de-grid (2026-07-16 grid issue, 2 m tier): the cooked biome/
  // soil planes are 2 m class rasters (ETAK/CHM); bilinear feathers their texel
  // steps by ≤1 texel and the canopy/landcover smoothsteps re-sharpen that into
  // visible 2 m blocks (the "blocky darkening"). Meander the CONTROL-plane
  // sample coordinate with a world-anchored signed offset (±0.75 m, ~5 m
  // features — within the 2 m raster's positional honesty, so no class leaks
  // beyond ~1 texel of its data support) so texel boundaries stop tracing the
  // lattice. Geometry-paired taps (height/normal/slope, waterY — riverDepth is
  // waterY − real surface h) stay on the true wxz. Cooked sources only: without
  // hasCanopy wxzB ≡ wxz ⇒ the generated graph is compile-time BIT-IDENTICAL.
  const wxzB = inp.hasCanopy
    ? (wxz.add(
        vec2(valS(5.3, 0.19, 0.67), valS(5.3, 0.83, 0.41)).mul(0.75),
      ) as unknown as NV2)
    : wxz;

  // terrain field context: the TerrainField planes (S3b). Fine-lattice cooks
  // take the C0 smooth normal/slope (per-texel-constant CD normals render as
  // faint 6–12 cm facet tiles under the sun term); coarse sources compile the
  // exact nearest-CD graph (bit-identical).
  const field = inp.field;
  const fineLatticeNs = (field.heightLevels[0] as { texel: number }).texel < 0.5;
  const ns: NV4 = fineLatticeNs
    ? field.fieldNormalSlopeSmoothHot(wxz)
    : field.fieldNormalSlopeHot(wxz);
  const fld = field.fieldsAt(wxzB);
  const bio = field.biomeAt(wxzB);
  const snowRaw = fld.z as unknown as NF;
  const vegRaw = bio.y as unknown as NF;
  const rockRaw = fld.w as unknown as NF;
  const moistRaw = fld.x as unknown as NF;
  const flowRaw = fld.y as unknown as NF;
  // riverDepth DERIVED (spec §3): waterY plane − this pixel's surface height
  // (h = the rendered surface — self-consistent per pixel). BILINEAR waterY:
  // dry texels hold the bed−2 sentinel, a nearest tap zeroes whole sim-texel
  // patches of submerged bed; max(0) absorbs the sentinel and the no-water −1e4.
  const riverRaw = (field.water ? field.fieldWaterY(wxz) : float(-1e4)).sub(h).max(0) as unknown as NF;
  /** standing-water magnitude for the silt-bed term (see pondK): the retired
   *  fieldsTex.z was NOT the water column — it was the hydrology CARVE metric
   *  (FlowRivers carveK: depth·0.45+0.12, lakes = fill depth), and pondK's
   *  1.1–2.6 m knees are calibrated to IT (the real column rarely tops 1.5 m).
   *  Reconstruct it from the flowStrength PLANE (lakes carry 1 ⇒ 3.5; rivers
   *  match FlowRivers' sB^1.35·7.5·0.45+0.12 law), gated on being actually
   *  submerged — carve gullies that render dry stay silt-free (legacy rdGate). */
  const pondDepth = flowRaw
    .pow(1.35)
    .mul(3.375)
    .add(0.12)
    .mul(smoothstep(0.02, 0.1, riverRaw)) as unknown as NF;
  const baseNormal = ns.xyz.normalize().toVar();
  const slope = ns.w.toVar();

  // ---------- class-selection slope (fine-lattice cooks) -----------------------
  // The material BLEND field must be continuous: nearest-texel CD slope is
  // piecewise-constant per height texel (6.25 cm on the micro cook), so class
  // weights could only change in texel-sized blocks over full-resolution
  // carriers — material transitions rendered as ~10 cm "pixels" (2026-07-16
  // grid issue). Classes read the ≥0.5 m field through the C0 smooth gradient
  // (the scale their windows were tuned on; measured adjacent-texel jump p50
  // 0.011 there vs 0.27 at 6.25 cm), and the centimetre relief re-enters
  // SELECTION through the continuous fine-relief exposure below — form-driven,
  // never lattice-quantized. Shading normals (ns/baseNormal) are unchanged.
  // Coarse-lattice sources (generated world, 1 m cooks) compile classSlope ≡
  // slope ⇒ bit-identical graphs.
  const fineLattice = (field.heightLevels[0] as { texel: number }).texel < 0.5;
  if (fineLattice)
    // eslint-disable-next-line no-console
    console.log(
      '[laas] terrain class-slope: fine-lattice gate ACTIVE (continuous selection slope + relief exposure)',
    );
  const classSlope = fineLattice ? (field.fieldClassSlopeHot(wxz).toVar() as unknown as NF) : slope;
  // Fine-relief LACEWORK: the intricate broken-ground dirt/stone marbling the
  // old per-texel selection noise used to produce, rebuilt from CONTINUOUS
  // ingredients only — the C0 fine gradient (real ledges, boulder faces, root
  // steps) MULTIPLIED by a smooth fbm carrier, so the exposure pattern is dense
  // and organic but mathematically cannot print the texel lattice. reliefBare
  // thins grassW/forestW (the soil/litter beneath shows through in lace
  // filaments); reliefRock joins rockW on the strongest forms. Density lives in
  // the two smoothstep knee pairs below. Evaluated only inside 60 m: sub-texel
  // on screen beyond, and the fade keeps the term clear of the finest window's
  // L0→L1 hand-off.
  let reliefBare: NF | null = null;
  let reliefRock: NF | null = null;
  if (fineLattice) {
    const relief = float(0).toVar();
    If(camDist.lessThan(60), () => {
      relief.assign(field.fieldReliefSlopeHot(wxz));
    });
    const reliefNear = smoothstep(60, 45, camDist);
    // lace = multi-scale smooth carriers (0.85 m fbm + 0.29 m val) MODULATED by
    // relief. The carrier base gives the dense broken-ground marbling on FLAT
    // terrain too (the old noise-driven look, from continuous ingredients); the
    // relief factor concentrates it where the ground is genuinely broken.
    // Density knobs = the two knee pairs below.
    const lace = fbmV(0.85, 0.41, 0.23)
      .mul(0.65)
      .add(val(0.29, 0.53, 0.11).mul(0.35))
      .mul(relief.mul(1.1).add(0.45)) as unknown as NF;
    reliefBare = smoothstep(0.3, 0.72, lace).mul(reliefNear) as unknown as NF;
    reliefRock = smoothstep(0.72, 1.15, lace).mul(reliefNear) as unknown as NF;
  }

  // --- baked-noise gradient helpers --------------------------------------------
  const liftNoiseGradient = (g: NV2): NV3 => {
    if (!inp.surf?.noiseCoord) return vec3(g.x, 0, g.y) as unknown as NV3;
    return vec3(g.x, g.x.mul(shearX).add(g.y.mul(shearZ)), g.y) as unknown as NV3;
  };
  const tangentNoiseGradient = (g: NV3): NV3 => {
    if (!inp.surf?.noiseCoord) return g;
    return g.sub(baseNormal.mul(baseNormal.dot(g))) as unknown as NV3;
  };
  /** projected fbm/ridged gradients for cosmetic normal detail */
  const fbmGW = (s: number, ox = 0, oz = 0): NV3 =>
    liftNoiseGradient(texture(inp.noiseA, noiseUv(s * PERIOD_FBM).add(vec2(ox, oz))).zw.div(s));
  const ridGW = (s: number): NV3 =>
    liftNoiseGradient(texture(inp.noiseB, noiseUv(s * PERIOD_RID)).xy.div(s));
  /** XZ gradients remain paired with the actual heightfield displacement. */
  const fbmGXz = (s: number): NV2 =>
    texture(inp.noiseA, noise3(s * PERIOD_FBM).xz).zw.div(s) as unknown as NV2;
  const ridGXz = (s: number): NV2 =>
    texture(inp.noiseB, noise3(s * PERIOD_RID).xz).xy.div(s) as unknown as NV2;
  // the TerrainField plane taps ARE the field at any distance (the coarsest
  // biome/height levels are country floors) — consumed directly, near and far.
  const snowField = snowRaw;
  const vegDensity = vegRaw;
  const rockExposure = rockRaw;
  const moisture = moistRaw;
  const flowStrength = flowRaw;
  const riverDepth = riverRaw;
  const zm = zoneMasks(wxz, inp.mp);

  // ---------- geology priors (EGT, optional plane — 30a16ef) --------------------
  // [bedrockFamily, surficialFamily, processFamily, coverageFlags] on the 2 m
  // condition lattice. CATEGORY IDENTITY is one NEAREST tap — ids are never
  // interpolated; the second bilinear tap is ONLY a boundary-confidence signal
  // (|linear − nearest| ≈ edge proximity, the landcover classInterior idiom),
  // so every derived appearance parameter fades to 0 BEFORE the nearest id can
  // change and no straight polygon seam can print. Both taps ride wxzB, so the
  // fades meander with the other control planes (world-stable, TAA-stable).
  // The values are coarse regional PRIORS (1:200k map evidence): they re-tint
  // existing palettes and bias existing weights; they never place bedding,
  // cliffs, or polygon-edge geometry. Per-family KNOWN bits × the authoritative
  // bit gate every term — missing evidence ⇒ weight 0 ⇒ exactly the current
  // material. The whole block sits inside `if (inp.hasGeology)` ⇒ old manifests
  // and the generated world compile BIT-IDENTICAL graphs. Cost when present:
  // 5 rgba8 samples (1 nearest + the 4-load toroidal bilinear) + ~70 ALU of
  // windows/mixes whose temporaries die at the composite — no cross-branch
  // liveness, no storage buffers, one texture binding (the plane itself).
  let geo: {
    sandstone: NF;
    carbonate: NF;
    sand: NF;
    till: NF;
    gravel: NF;
    peat: NF;
    aeolian: NF;
    expose: NF;
  } | null = null;
  if (inp.hasGeology) {
    const gN = field.geologyAt(wxzB);
    const gL = field.geologyLinearAt(wxzB);
    const bed = (gN.x as unknown as NF).mul(255);
    const surf = (gN.y as unknown as NF).mul(255);
    const proc = (gN.z as unknown as NF).mul(255);
    const flags = (gN.w as unknown as NF).mul(255).add(0.5).floor();
    const bit = (i: number): NF => flags.div(1 << i).floor().mod(2) as unknown as NF;
    // boundary confidence: any channel's bilinear value pulling away from the
    // nearest byte means a category edge inside the 2×2 — fade before it
    const dMax = gL.x
      .sub(gN.x)
      .abs()
      .max(gL.y.sub(gN.y).abs())
      .max(gL.z.sub(gN.z).abs())
      .max(gL.w.sub(gN.w).abs())
      .mul(255);
    const interior = smoothstep(0.55, 0.12, dMax);
    const auth = bit(0).mul(interior) as unknown as NF;
    /** unit window at categorical id k (nearest ids are exact bytes) */
    const idW = (v: NF, k: number): NF =>
      smoothstep(k - 0.45, k - 0.05, v).mul(smoothstep(k + 0.45, k + 0.05, v)) as unknown as NF;
    const bedK = bit(1).mul(auth) as unknown as NF;
    const surfK = bit(2).mul(auth) as unknown as NF;
    const procK = bit(3).mul(auth) as unknown as NF;
    geo = {
      sandstone: idW(bed, 1).mul(bedK) as unknown as NF,
      carbonate: idW(bed, 2).mul(bedK) as unknown as NF,
      sand: idW(surf, 1).mul(surfK) as unknown as NF,
      till: idW(surf, 2).mul(surfK) as unknown as NF,
      gravel: idW(surf, 3).mul(surfK) as unknown as NF,
      peat: idW(surf, 4).mul(surfK) as unknown as NF,
      // aeolian refines SAND toward clean dune pale; the other process families
      // either duplicate a surficial family (peat-forming, glaciofluvial gravel)
      // or belong to systems handled elsewhere (water, anthropogenic → landcover)
      aeolian: idW(proc, 10).mul(procK) as unknown as NF,
      // mapped bedrock exposure (flag bit 4): an exposure PRIOR, applied only
      // where slope already suggests rock (see the rockW bias below)
      expose: bit(4).mul(auth) as unknown as NF,
    };
  }

  // ---------- macro variation (2–50 m breakup — tiling killer) ----------------
  const macroA = val(43.7);
  const macroB = val(11.3, 0.37, 0.61);

  // ---------- meso/micro detail noise ------------------------------------------
  const meso: NF = inp.meso ? fbmV(1.45) : float(0.5);
  // Explicit LOD keeps this sample legal inside a non-uniform distance branch
  // (the bake has no mip chain). Past the fine-geometry band, 0.5 is the neutral
  // value and the texture read is skipped entirely.
  const micro = float(0.5).toVar();
  If(camDist.lessThan(40), () => {
    micro.assign(
      texture(
        inp.noiseA,
        noiseUv(MATERIAL_DETAIL_M * PERIOD_VAL).add(vec2(0.71, 0.13)),
        0,
      ).x,
    );
  });
  const microSigned = micro.mul(2).sub(1) as unknown as NF;
  // Match the packed 6.25 cm rung's 32 m full-detail / 40 m morph-out band.
  // This is an amplitude gate, not a sampling branch, so derivative legality and
  // quad execution stay unchanged.
  const materialDetailNear = smoothstep(40, 32, camDist);
  /** Add fine structure only inside an existing overlap. The 4*w*(1-w)
   *  envelope is exactly zero at both endpoints, so a material cannot leak
   *  outside the support supplied by slope/biome/field data. */
  const refineOverlap = (w: NF, strength = MATERIAL_DETAIL_BLEND): NF =>
    w
      .add(
        microSigned
          .mul(materialDetailNear)
          .mul(w.mul(w.oneMinus()))
          .mul(strength * 4),
      )
      .clamp(0, 1) as unknown as NF;
  // ---------- class palettes ----------------------------------------------------
  // rock: subtle strata banding; warm rust in the alpine zone, pale gray in
  // karst. Low contrast + heavy phase warp so it reads as geology, not zebra.
  const strataPhase = h
    .mul(0.028)
    .add(valS(74, 0.11, 0.83).mul(3.6))
    .add(valS(540, 0.43, 0.29).mul(2.4))
    .add(valS(27, 0.91, 0.07).mul(1.3)); // fine jitter fragments the bands
  const strata = band(strataPhase, valS(610, 0.67, 0.41).mul(1.7).add(31.7))
    .mul(0.36)
    .add(0.3); // compress contrast — long smooth walls turn 'layer cake' fast
  // reference peaks are DARK: gray-blue mass with rust faces catching light —
  // pale palettes washed the whole massif into cream at golden hour
  const alpRock = mix(vec3(0.16, 0.135, 0.125), vec3(0.38, 0.26, 0.18), strata);
  const karstRock = mix(vec3(0.3, 0.3, 0.29), vec3(0.5, 0.48, 0.44), strata);
  const genericRock = mix(vec3(0.26, 0.245, 0.225), vec3(0.42, 0.39, 0.35), strata);
  let rockCol = mix(genericRock, karstRock, zm.tKarst);
  rockCol = mix(rockCol, alpRock, zm.tAlp.mul(0.85));
  // iron-oxide bands: dark rust layers at noise-chosen elevations (refs show
  // strong hue layering on alpine faces)
  const ironPhase = band(h.mul(0.011), valS(800, 0.07, 0.93).mul(1.3).add(57.3));
  const ironBand = smoothstep(0.45, 0.62, ironPhase).mul(smoothstep(0.85, 0.62, ironPhase));
  rockCol = mix(rockCol, vec3(0.3, 0.18, 0.12), ironBand.mul(zm.tAlp.mul(0.6).add(0.12)));
  // lichen/weathering: dark macro splotches on long-exposed faces
  const lichen = smoothstep(0.6, 0.85, val(23.7, 0.53, 0.27));
  rockCol = mix(rockCol, rockCol.mul(0.62), lichen.mul(0.5));
  if (geo) {
    // bedrock family → rock PALETTE prior (identity of exposed rock only —
    // exposure itself still comes from slope/relief/rockExposure).
    // SANDSTONE (Estonian Devonian reference — Taevaskoja/Härma walls):
    // near-HORIZONTAL strata whose color FAMILY drifts with depth — cream/
    // buff, salmon-pink, red-ochre, thin maroon seams — bands 0.3–2 m with
    // only mild lateral waviness. The generic strataPhase's heavy lateral
    // warp (±7.3 phase units, tuned to fragment kilometre-scale generated
    // massifs) exceeds a 25 m outcrop's WHOLE elevation phase and reads as
    // camo mottle, so sandstone gets its own h-dominant signal: ~1.25 m
    // bands (±0.4 m waviness), an ~18 m member-family drift sliding the
    // palette pale→red with depth, and occasional thin maroon seams. Still
    // cosmetic noise banding, not asserted bedding.
    // Superposed strata signals (ref: Härma/Taevaskoja walls) — the variety
    // lives in three independent axes, all h-dominant and value-noise
    // IRREGULAR (features vary ±50% around their nominal size):
    //   hueSig  ~5 m  color members through a 4-stop ramp (cream → ochre-
    //                 salmon → brick → dusty violet)
    //   thick   ~1.8 m broad value banding (midrange-expanded)
    //   lam     ~0.4 m thin laminae, arriving in PACKETS (bundle ~8 m) with
    //                 quiet washes between — sedimentary bundling
    const ssLane = valS(310, 0.77, 0.13).mul(1.1).add(17.3);
    // Per-band lateral pinch-and-swell: these are cross-bedded FLUVIAL sheets —
    // individual beds wander independently, so each selector gets its OWN
    // decorrelated warp (one shared warp made every boundary undulate in
    // parallel, the "too perfectly bandy" read).
    const ssWav1 = valS(9.7, 0.31, 0.57).mul(0.35);
    const ssWav2 = valS(7.3, 0.63, 0.29).mul(0.3);
    const ssWav3 = valS(11.9, 0.17, 0.91).mul(0.4);
    // "up top" datum: the Devonian beds are subhorizontal (dips <1°), so within
    // a valley ABSOLUTE elevation ≈ stratigraphic depth — free (h). The member
    // ceiling drifts ±12 m regionally via one slow tap. deepBias fades the cold
    // member out above the ceiling; deepBias2 (fully on ~16 m below it) makes
    // the violet washes BROADER and more common in the deepest sections (ref:
    // the bottom-third violet wash).
    const deepCeil = valS(2900, 0.19, 0.83).mul(12).add(54);
    const deepBias = smoothstep(deepCeil.add(8), deepCeil.sub(6), h);
    const deepBias2 = smoothstep(deepCeil.sub(4), deepCeil.sub(16), h);
    // ALL BANDS, NO BASE (ref): every elevation is a COMMITTED band color.
    // E() expands value noise's midpoint-heavy distribution onto plateaus, so
    // each band holds one solid color with a narrow soft edge. s1 (~2.4 m)
    // switches warm↔deep family (its raised lower knee keeps grazing crossings
    // warm — no sliver-thin deep bands); s2/s3 (~0.7 m) pick within the family.
    const E = (x: NF): NF => smoothstep(0.38, 0.62, x) as unknown as NF;
    const s1 = smoothstep(0.4, 0.66, band(h.mul(0.42).add(ssWav1), ssLane.add(11.1)));
    const s2 = E(band(h.mul(1.5).add(ssWav2.mul(1.5)), ssLane.add(29.3)));
    // violet: window shifted by depth (thicker/commoner at the bottom), gated
    // to WELL-committed deep bands only — a thin deep sliver renders brick,
    // never a thin blue line (the immersion breaker); thin warm laminae stay.
    const s3 = E(
      band(h.mul(1.5).add(ssWav3.mul(1.5)), ssLane.add(53.9)).add(deepBias2.mul(0.14)),
    )
      .mul(deepBias)
      .mul(smoothstep(0.55, 0.85, s1));
    const lam = band(h.mul(2.4).add(ssWav2.mul(2)), ssLane.add(47.9));
    const bundle = smoothstep(0.42, 0.72, band(h.mul(0.13), ssLane.add(71.3)));
    const warmPair = mix(vec3(0.82, 0.75, 0.62), vec3(0.76, 0.5, 0.36), s2);
    // violet stop: dusty, low-chroma, a touch darker (ref) — not vivid purple
    const deepPair = mix(vec3(0.58, 0.32, 0.24), vec3(0.44, 0.36, 0.39), s3);
    let sandstoneR = mix(
      warmPair,
      deepPair,
      s1.mul(deepBias.mul(0.45).add(0.55)),
    ) as unknown as NV3;
    // thin laminae in packets — soft value shimmer only. (A discrete dark
    // "seam line" term was tried and cut: sub-decimetre hard lines read as
    // rendering artifacts at wall scale, not geology.)
    sandstoneR = sandstoneR.mul(lam.sub(0.5).mul(0.22).mul(bundle).add(1)) as unknown as NV3;
    // CARBONATE: pale limestone gray on the generic strata band.
    const carbonateR = mix(vec3(0.35, 0.34, 0.31), vec3(0.57, 0.55, 0.49), strata);
    rockCol = mix(rockCol, sandstoneR, geo.sandstone.mul(0.9));
    rockCol = mix(rockCol, carbonateR, geo.carbonate.mul(0.85));
  }
  // cavity dirt: concave-ish micro band darkening
  rockCol = rockCol.mul(meso.mul(0.22).add(0.89)).mul(micro.mul(0.1).add(0.95));

  const scree = vec3(0.36, 0.345, 0.325).mul(meso.mul(0.35).add(0.78));
  let soil = mix(vec3(0.155, 0.12, 0.085), vec3(0.24, 0.195, 0.135), meso).mul(
    micro.mul(0.2).add(0.9),
  ) as unknown as NV3;
  if (geo) {
    // surficial family → bare-ground mineral tints (soil feeds the composite
    // base AND litter, so one chain covers every place bare ground shows —
    // including the relief lace patches). Targets keep the meso variation so
    // tinted regions don't flatten; the id windows are mutually exclusive.
    const gm = meso.mul(0.3).add(0.85);
    const sandT = mix(vec3(0.42, 0.35, 0.22), vec3(0.5, 0.44, 0.31), geo.aeolian).mul(gm);
    soil = mix(soil, sandT, geo.sand.mul(0.55)) as unknown as NV3;
    soil = mix(soil, vec3(0.235, 0.2, 0.155).mul(gm), geo.till.mul(0.45)) as unknown as NV3;
    soil = mix(soil, vec3(0.38, 0.36, 0.33).mul(gm), geo.gravel.mul(0.5)) as unknown as NV3;
    soil = mix(soil, vec3(0.075, 0.058, 0.042).mul(gm), geo.peat.mul(0.6)) as unknown as NV3;
  }
  // grass field color = the FINAL grass LOD: matched to the blade-ring
  // palette (screen-average of the blade ramps) with the SAME ~1.6 m patch
  // dryness, so the geometric grass dissolves into this instead of ending
  // at a visible ring edge ("empty terrain" feedback)
  const patchN = val(1.6, 0.23, 0.77);
  const grassG = mix(vec3(0.036, 0.094, 0.019), vec3(0.06, 0.13, 0.028), macroA);
  const grassDry = vec3(0.15, 0.122, 0.052);
  const grassCol = mix(
    grassG,
    grassDry,
    smoothstep(0.6, 0.92, patchN.mul(0.55).add(macroB.mul(0.45))),
  ).mul(meso.mul(0.25).add(0.85));
  // forest floor: litter brown blended w/ moss by moisture
  const litter = mix(soil, vec3(0.18, 0.15, 0.095), meso);
  const mossy = vec3(0.11, 0.185, 0.065);
  const forestFloor = mix(litter, mossy, smoothstep(0.45, 0.8, moisture).mul(0.7));
  // gravel/cobble tint in stream channels
  const gravel = mix(vec3(0.34, 0.33, 0.31), vec3(0.47, 0.45, 0.43), micro);
  const snowCol = mix(vec3(0.86, 0.88, 0.94), vec3(0.93, 0.95, 0.99), macroA).mul(
    meso.mul(0.08).add(0.95),
  );

  // ---------- class weights ------------------------------------------------------
  const rockW = refineOverlap(
    smoothstep(0.62, 1.15, classSlope).max(rockExposure.mul(0.85)) as unknown as NF,
  ).toVar();
  if (reliefRock) rockW.assign(rockW.max(reliefRock.mul(0.9)));
  if (geo) {
    // mapped bedrock-exposure polygons (flag bit 4) BIAS exposure where slope
    // already suggests rock — a prior on the weight, never a placed cliff.
    rockW.assign(rockW.max(smoothstep(0.3, 0.85, classSlope).mul(geo.expose).mul(0.55)));
  }
  const screeW = refineOverlap(
    smoothstep(0.42, 0.62, classSlope)
      .mul(smoothstep(1.15, 0.7, classSlope))
      .mul(smoothstep(380, 700, h))
      .mul(rockW.oneMinus()) as unknown as NF,
    0.1,
  );
  let grassW = refineOverlap(
    smoothstep(0.5, 0.22, classSlope)
      .mul(vegDensity)
      .mul(zm.tKarst.mul(0.5).oneMinus())
      .mul(rockW.oneMinus()) as unknown as NF,
  );
  if (reliefBare) grassW = grassW.mul(reliefBare.mul(0.85).oneMinus()) as unknown as NF;
  let forestW = refineOverlap(
    vegDensity
      .mul(smoothstep(0.9, 0.45, classSlope))
      .mul(smoothstep(0.25, 0.6, moisture.add(zm.tKarst.mul(0.3))))
      .mul(rockW.oneMinus()) as unknown as NF,
    0.08,
  );
  if (reliefBare) forestW = forestW.mul(reliefBare.mul(0.5).oneMinus()) as unknown as NF;
  // gravel only for REAL channels on open ground: weak-flow rills under
  // grass painted pale streaks down every meadow hillside — those should
  // darken via moisture instead
  const riverW = smoothstep(0.3, 0.68, flowStrength)
    .mul(smoothstep(0.45, 0.2, classSlope))
    .mul(grassW.mul(0.75).oneMinus());

  // snow with hash-dithered edge (reads as crisp organic boundary, not
  // gradient). Dither only near the boundary — ungated it sprinkled white
  // pixels over bare rock wherever snowField hovered above zero.
  const ditherGate = smoothstep(0.06, 0.22, snowField).mul(smoothstep(0.95, 0.6, snowField));
  const dither = hash12(wxz.mul(7.31)).sub(0.5).mul(0.34).mul(ditherGate);
  const snowW = smoothstep(0.16, 0.5, snowField.add(dither)).toVar();

  // ---------- composite -----------------------------------------------------------
  // standing-water beds (kettle ponds, lake): fine dark silt, not gravel —
  // the real Phase-6 water surface + Beer–Lambert absorption sit above this
  const pondK = smoothstep(1.1, 2.6, pondDepth).mul(smoothstep(0.3, 0.12, classSlope));
  let col: NV3 = soil;
  col = mix(col, grassCol, grassW);
  col = mix(col, forestFloor, forestW);
  col = mix(col, scree, screeW);
  col = mix(col, rockCol, rockW);

  // ---------- soil pedology (Estonia Mullastikukaart, #116) ---------------------
  // The cooked soil plane [texCore, stoniness, boniteet, texSkeleton] modulates the
  // MINERAL SUBSTRATE that land-cover (classId, below) is blind to — a forest / grass /
  // field texel each sit on some soil texture. DATA-DRIVEN and geometric: texCore sets a
  // mineral tint (sand pale-warm → loam neutral → clay red-brown → peat dark), stoniness
  // sets the AMPLITUDE of a value-noise pebble speckle (the noise is the carrier, the
  // stoniness class is the field — never a stand-in for missing data), boniteet enriches
  // the ground flora, texSkeleton tints the speckle by lithology. Applied to the base
  // composite BEFORE the classId overrides (which mix() fully on top ⇒ near-zero
  // double-count). Everything sits INSIDE `if (inp.hasSoil)` so the generated world (no
  // soil layer ⇒ hasSoil=false) never constructs a single soil node ⇒ BIT-IDENTICAL.
  if (inp.hasSoil) {
    const soilS = field.soilAt(wxzB); // [texCore, stoniness, boniteet, texSkeleton] byte/255
    const texCore = (soilS.x as unknown as NF).mul(255); // 0..14 (255 unparseable, 0 no-data)
    const stony = (soilS.y as unknown as NF).mul(255).div(6).clamp(0, 1); // 0..6 → [0,1]
    const boni = (soilS.z as unknown as NF).mul(255).div(100).clamp(0, 1); // 0..100 → [0,1]
    const texSkel = (soilS.w as unknown as NF).mul(255); // 0..30 (255 unparseable)
    // valid mineral texel: id in [1,14]; the descending knee fades it to 0 across the
    // 255-unparseable bilinear blur (a blend toward 255 leaves the window fast) and 0
    // no-data reads as invalid too ⇒ neither sentinel tints.
    const texValid = smoothstep(0.5, 1.0, texCore).mul(smoothstep(15.5, 14.0, texCore));
    // exposed bare ground — low veg, not rock, gentle slope: soil texture only reads where
    // the grass/forest carpet doesn't hide it (so meadows/woods aren't speckled or tinted).
    const exposed = smoothstep(0.55, 0.2, vegDensity).mul(rockW.oneMinus()).mul(smoothstep(0.6, 0.25, classSlope));

    // texCore → mineral tint ramp. The id ordering IS a physical gradient (§2), so a
    // bilinear tap stays within-family — treated as a near-continuous mineral index.
    const mineral = mix(vec3(1.14, 1.06, 0.9), vec3(0.86, 0.72, 0.58), smoothstep(3.0, 8.0, texCore)); // sand→clay
    const mineralP = mix(mineral, vec3(0.52, 0.44, 0.34), smoothstep(8.5, 12.0, texCore)); // →peat dark-brown
    col = col.mul(mix(vec3(1), mineralP, texValid.mul(exposed))) as NV3;

    // boniteet → ground-flora richness: fertile soil greens/darkens the vegetated ground,
    // poor soil pales it. Continuous (bilinear-safe), gated to the veg coverage it drives.
    const vegK = grassW.max(forestW);
    const rich = mix(vec3(0.9, 0.95, 0.82), vec3(0.72, 1.05, 0.66), smoothstep(0.15, 0.7, boni));
    col = col.mul(mix(vec3(1), rich, vegK.mul(0.5))) as NV3;

    // stoniness → geometric micro-speckle: value noise is the CARRIER, stoniness sets the
    // amplitude; scattered pebbles/gravel on exposed soil only. texSkeleton shifts the hue
    // toward pale carbonate gray on rähk (6-10) — low-weight, tolerant of categorical blur.
    const speck = val(0.13, 0.29, 0.83).sub(0.5); // signed micro speckle
    const speckAmp = stony.mul(exposed).mul(0.4);
    const speckHue = mix(vec3(1), vec3(0.92, 0.92, 0.97), smoothstep(5.5, 10.5, texSkel));
    col = col.mul(speck.mul(speckAmp).add(1)).mul(mix(vec3(1), speckHue, speckAmp)) as NV3;
  }

  // ---------- land-cover classes (Estonia ETAK classId) -------------------------
  // The streamed biome plane's channel 0 is the ETAK land-cover class id
  // (landcover-classes.toml: forest 1, shrub 2, grassland 3, barren 4, sand 5,
  // field 6, yard 7, bog 8, fen 9, peatfield 10, water 11–13). The continuous
  // fields (vegDensity/slope) already carry forest/meadow; classId adds the
  // CATEGORICAL reads those fields cannot infer — peat bog / sedge fen / cut
  // peatfield (all dark & wet, not bright grass), tilled arable field, and bare
  // barren/sand. Composited ON TOP of the field result, feathered by the bilinear
  // class channel and CO-GATED by the real fields (slope/veg) so the intermediate
  // ids a bilinear sweep crosses between two distant classes don't flash a wrong
  // material. Estonia carries NO fields layer ⇒ moisture/snow/rock are 0 here, so
  // classId is the ONLY signal that a flat green texel is a bog. Gated to
  // `landcover` (the cooked ETAK source): the generated world packs the Biome enum
  // in classId instead, so its graph is compile-time BIT-IDENTICAL.
  if (inp.landcover) {
    const cid = field.biomeClassAt(wxzB).mul(255);
    // Identity comes from a real nearest source texel; the filtered channel is
    // used only as an edge-confidence signal. Every categorical overlay fades
    // to zero before the nearest id changes, preventing both false intermediate
    // materials and a hard 2 m square step at the class boundary.
    const cidLinear = (bio.x as unknown as NF).mul(255);
    const classInterior = refineOverlap(
      smoothstep(0.55, 0.12, cidLinear.sub(cid).abs()) as unknown as NF,
      0.08,
    );
    const notRock = rockW.oneMinus();
    // bare open ground — barren(4) → warm sand(5). Only where the veg field agrees
    // (suppresses the barren/sand ids a forest↔field sweep crosses in the seam).
    const bareW = smoothstep(3.55, 4.0, cid)
      .mul(smoothstep(5.5, 5.05, cid))
      .mul(smoothstep(0.45, 0.15, vegDensity))
      .mul(notRock)
      .mul(classInterior)
      .mul(0.9);
    const bareCol = mix(vec3(0.3, 0.27, 0.215), vec3(0.47, 0.41, 0.3), smoothstep(4.5, 5.0, cid))
      .mul(meso.mul(0.24).add(0.86))
      .mul(micro.mul(0.14).add(0.92));
    // arable field(6): tilled earthy tone — warmer and more uniform than the
    // natural meadow green. Flat ground only.
    const fieldW = smoothstep(5.55, 6.0, cid)
      .mul(smoothstep(6.65, 6.15, cid))
      .mul(smoothstep(0.5, 0.2, classSlope))
      .mul(notRock)
      .mul(classInterior)
      .mul(0.72);
    const fieldCol = mix(vec3(0.185, 0.14, 0.09), vec3(0.15, 0.14, 0.078), macroB).mul(
      meso.mul(0.2).add(0.88),
    );
    // peat wetland — bog(8)/fen(9)/peatfield(10): dark wet moss/peat, NOT grass —
    // the highest-impact class (these otherwise read as generic meadow). bog rusty
    // sphagnum → fen olive sedge → cut peatfield near-black bare peat. Flat only.
    const peatW = smoothstep(7.55, 8.1, cid)
      .mul(smoothstep(10.75, 10.3, cid))
      .mul(smoothstep(0.4, 0.15, classSlope))
      .mul(notRock)
      .mul(snowW.oneMinus())
      .mul(classInterior)
      .mul(0.92);
    const bogFen = mix(vec3(0.115, 0.086, 0.052), vec3(0.086, 0.1, 0.056), smoothstep(8.0, 9.2, cid));
    const peatCol = mix(bogFen, vec3(0.046, 0.039, 0.031), smoothstep(9.4, 10.1, cid)).mul(
      meso.mul(0.18).add(0.88),
    );
    col = mix(col, bareCol, bareW) as NV3;
    col = mix(col, fieldCol, fieldW) as NV3;
    col = mix(col, peatCol, peatW) as NV3;
  }

  col = mix(col, gravel, riverW.mul(0.85).mul(pondK.oneMinus()));
  col = mix(col, vec3(0.055, 0.052, 0.038), pondK);
  col = mix(col, snowCol, snowW);

  // feedback 2.8 (splat half): a real grass field is DIRECTIONAL — forward
  // scatter through backlit blades brightens and warms it toward the sun at
  // grazing view angles. Distance-gated: near meadows have actual blades
  // (g0–g3); this gives the 200 m+ sward the same directional life so the
  // far layers dissolve into a live field, not flat paint.
  {
    const vDir = wp.sub(camPos).normalize();
    const sunD = vec3(sunU.dir as unknown as NV3).normalize();
    const toSun = vDir.dot(sunD).max(0);
    const grazing = float(1).sub(baseNormal.dot(vDir.negate()).abs()).pow2();
    const sheenK = grassW
      .mul(snowW.oneMinus())
      .mul(toSun.pow(3))
      .mul(grazing)
      .mul(smoothstep(0.05, 0.22, sunD.y))
      .mul(smoothstep(60, 220, camDist))
      .mul(0.55);
    col = col.add(vec3(0.085, 0.1, 0.032).mul(sheenK)) as NV3;
  }

  // gorge/ravine wall vegetation (scene1: ravine walls are NOT bare — they
  // carry moss bands, hanging greens and ledge clumps). Steep faces in damp
  // valleys grow green in noise pockets: fbm bands read as hanging veg,
  // value-noise pockets as ledge clumps. Karst gorges get the most.
  const wallK = smoothstep(0.62, 1.0, classSlope)
    .mul(smoothstep(0.12, 0.42, moisture.add(riverDepth.mul(2))))
    .mul(smoothstep(1350, 700, h))
    .mul(snowW.oneMinus())
    .mul(zm.tKarst.mul(0.45).add(0.55));
  const wallBands = smoothstep(0.38, 0.72, fbmV(7.3, 0.13, 0.49));
  const ledgePock = smoothstep(0.45, 0.78, val(2.9, 0.61, 0.07));
  const wallVeg = wallK
    .mul(wallBands.mul(0.85).add(ledgePock.mul(0.6)))
    .clamp(0, 0.92);
  const wallGreen = mix(vec3(0.07, 0.115, 0.04), vec3(0.105, 0.165, 0.05), macroA);
  col = mix(col, wallGreen, wallVeg);

  // wet darkening: river margins, lake shores, marshes
  const shoreWet = smoothstep(LAKE_LEVEL + 2.5, LAKE_LEVEL + 0.3, h);
  const wet = clamp(
    smoothstep(0.55, 0.95, moisture).mul(0.5).add(riverDepth.mul(2)).add(shoreWet.mul(0.6)),
    0,
    0.75,
  ).mul(snowW.oneMinus());
  col = col.mul(wet.mul(0.55).oneMinus());

  // far-forest canopy masses: where the cooked canopy layer reports cover (CHM
  // ≥ 2 m), the terrain shows treetops — not the ground material under them — as
  // a darker, cooler, richer green than open field. cover drives the blend;
  // canopy height deepens the shade (tall boreal spruce/pine read darkest). ETAK
  // land-cover carries NO conifer/deciduous split (its filtered classId is
  // scatter-only), so the tint leans on cover+height, not species. Gated to
  // hasCanopy (source carries canopy in biome ch 2/3) ⇒ the generated world
  // (snow/rockExposure there instead) is BIT-IDENTICAL. Distance-selects itself:
  // near Estonia reads biome L0 (no canopy lod ⇒ cover 0, no tint), the mid/far
  // levels carry the merged canopy. Water/sea never tint: the CHM has no canopy
  // over water ⇒ cover 0 there.
  if (inp.hasCanopy) {
    const cover = bio.w as unknown as NF;
    const canopyH = (bio.z as unknown as NF).mul(255); // heightM, m
    const forestK = smoothstep(0.12, 0.62, cover)
      .mul(smoothstep(1, 6, canopyH))
      // The tint's DOMAIN made explicit — it was implicit via biome-LOD
      // residency ("coarse level answered ⇒ far"), which leaked treetop green
      // onto NEAR terrain during stream transients. Distant forest only: fade
      // in where individual rendered trees genuinely thin out…
      .mul(smoothstep(300, 800, camDist))
      // …and downward-visible ground only: a cliff FACE under a forested brow
      // is rock, not treetops (the "green cliffs" report).
      .mul(smoothstep(0.9, 0.5, classSlope));
    const canopyCol = mix(
      vec3(0.038, 0.066, 0.03),
      vec3(0.02, 0.043, 0.024),
      smoothstep(6, 26, canopyH),
    );
    col = mix(col, canopyCol, forestK.mul(0.9)) as NV3;
  }

  // One shared fine carrier reaches every final material interior, including
  // grass, forest floor, snow and canopy, which previously stopped at meso
  // scales. Rock receives more breakup; snow/wet surfaces remain restrained.
  const materialDetailAmp = mix(float(0.045), float(0.09), rockW)
    .mul(snowW.mul(0.7).oneMinus())
    .mul(wet.mul(0.35).oneMinus())
    .mul(materialDetailNear);
  col = col.mul(microSigned.mul(materialDetailAmp).add(1)) as NV3;

  // ---------- normal perturbation ---------------------------------------------------
  // far-detail synthesis (Pillar D): serrated normal-domain detail keeps
  // mid/far ridges craggy where geometric density has LOD'd out. DISTANCE-gated.
  const farK = smoothstep(900, 2600, camDist);
  // pre-baked ridged gradient at 310 m features; ×44 ≈ the old ±22 m
  // finite-difference amplitude (×2: baked noise is [0,1], mx was [-1,1])
  const rg = ridGW(310).mul(44 * 2);
  // crag synthesis belongs to ROCK faces — on smooth vegetated hills the
  // ridged gradient field printed parallel pale corrugation streaks
  const farAmp = smoothstep(0.5, 1.1, slope)
    .mul(0.4)
    .add(smoothstep(0.32, 0.7, slope).mul(0.08))
    .mul(farK);
  // never let detail flip the surface away from the sky
  const perturbed = baseNormal.add(tangentNoiseGradient(rg).mul(farAmp));
  let nrm: NV3 = vec3(perturbed.x, perturbed.y.max(0.1), perturbed.z).normalize();

  // near/mid detail (both terms self-fade to nothing with distance — far tiles pay
  // no analytic bump/displacement): scoped block to keep its temporaries local.
  {
    // meso + micro analytic bumps near camera, stronger on rock — baked fbm
    // gradients at two scales (×2e ≈ old FD amplitudes, ×2 range factor)
    const b1 = fbmGW(1.45).mul(1.8 * 2);
    // 4x frequency with 1/4 the height coefficient preserves approximately the
    // old RMS normal strength instead of turning finer detail into steeper noise.
    const b2 = vec3(0).toVar();
    If(camDist.lessThan(40), () => {
      b2.assign(
        liftNoiseGradient(
          texture(
            inp.noiseA,
            noiseUv(MATERIAL_DETAIL_M * PERIOD_FBM).add(vec2(0.31, 0.77)),
            0,
          ).zw.div(MATERIAL_DETAIL_M),
        ).mul(0.06 * 2),
      );
    });
    const bumpAmp = mix(float(0.25), float(0.85), rockW)
      .mul(snowW.mul(0.7).oneMinus())
      .mul(farK.oneMinus());
    nrm = nrm
      .add(
        tangentNoiseGradient(
          b1.mul(0.7).add(b2.mul(0.45).mul(materialDetailNear)) as unknown as NV3,
        ).mul(bumpAmp),
      )
      .normalize();

    // geometric micro-displacement counterpart (TerrainTiles vertex): the
    // silhouette now has fbm/ridged relief — light it with the analytic
    // height-gradient normal (−∂h/∂x, 0, −∂h/∂z), same amplitudes + fade,
    // or the displaced surface shades as if it were still flat. Same gating
    // curve as the vertex stage (NOT rockW — different knees).
    const rockKd = smoothstep(DISP.slopeKnee0, DISP.slopeKnee1, slope).max(
      rockExposure.mul(0.85),
    );
    // gravel banks/streambeds are lumpy even on gentle slopes
    const gravelKd = smoothstep(0.32, 0.7, flowStrength)
      .max(smoothstep(0.02, 0.2, riverDepth))
      .mul(float(DISP.gravel));
    const dispAmpF = mix(float(DISP.base), float(DISP.rock), rockKd)
      .max(gravelKd)
      .mul(snowW.mul(0.75).oneMinus())
      .mul(
        clamp(float(DISP.fade1).sub(camDist).div(DISP.fade1 - DISP.fade0), 0, 1),
      );
    const gF = fbmGXz(DISP.sF1).mul(2 * DISP.wF1);
    const gR = ridGXz(DISP.sRid).mul(
      rockKd.mul(1 - DISP.ridBase).add(DISP.ridBase).mul(DISP.wRid),
    );
    const gSum = gF.add(gR).mul(dispAmpF);
    nrm = nrm.add(vec3(gSum.x.negate(), 0, gSum.y.negate())).normalize();
  }

  // ---------- roughness ---------------------------------------------------------------
  const rough = mix(float(0.94), float(0.8), rockW)
    .sub(snowW.mul(0.32))
    .sub(wet.mul(0.45))
    .clamp(0.25, 1);

  return {
    colorNode: col,
    normalNode: transformNormalToView(nrm),
    roughnessNode: rough,
    worldNormalNode: nrm,
  };
}
