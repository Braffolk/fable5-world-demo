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
 * NO-FAKE-DETAIL LAW (2026-07-19): every runtime procedural relief/shading
 * term (analytic fbm/ridged normal bumps, the micro-displacement fragment
 * counterpart, far ridged normal re-amplification) and every albedo term whose
 * spatial structure came from position-noise rather than a REAL cooked field
 * is REMOVED. The lighting normal is the real cooked height gradient only
 * (XZ-amplified for readability — stronger light on real geometry, not added
 * noise). Remaining noise taps are (a) h-driven strata banding (real elevation
 * is the axis), (b) the ±0.75 m control-plane de-grid meander (sampling
 * honesty for 2 m class rasters, not detail). PERF: net removal of ~8 noise
 * fetches + gradient ALU per fragment vs the pre-law shader.
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
import { PERIOD_VAL } from '../gpu/passes/NoiseBake';
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
  mp: MacroParams;
  /** the biome plane carries the merged far-forest canopy in channels 2/3
   *  (Estonia) rather than snow/rockExposure (the generated world) — gates the
   *  canopy tint so the generated look is bit-identical. */
  hasCanopy: boolean;
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

/** LIGHTING-normal gradient amplification for the REAL cooked relief. The C0
 *  smooth field normal tilts only ~11° at a genuine 0.2 slope, so the real
 *  6 cm hummock/hollow relief lights weakly; scaling the REAL XZ gradient
 *  steepens the light response of geometry that actually exists — legal
 *  (stronger light on real geometry), unlike the removed noise bumps. Applied
 *  to the lighting normal ONLY — baseNormal (class selection, grazing sheen)
 *  stays exact. Fades with distance for free: the field pyramid's coarser
 *  levels carry no fine gradient. */
const RELIEF_LIGHT_K = 2.0;

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
  // Fine-relief exposure: bare-ground / rock exposure where the REAL cooked
  // micro-forms are genuinely broken — driven by the C0 fine gradient
  // (fieldReliefSlopeHot: real ledges, boulder faces, root steps) ONLY. The
  // former fbm/val "lace" carrier was procedural-noise albedo structure (it
  // marbled FLAT ground) and is removed under the no-fake-detail law; the
  // knees below are the old carrier knees folded at its 0.5 mean. Evaluated
  // only inside 60 m: sub-texel on screen beyond, and the fade keeps the term
  // clear of the finest window's L0→L1 hand-off.
  let reliefBare: NF | null = null;
  let reliefRock: NF | null = null;
  if (fineLattice) {
    const relief = float(0).toVar();
    If(camDist.lessThan(60), () => {
      relief.assign(field.fieldReliefSlopeHot(wxz));
    });
    const reliefNear = smoothstep(60, 45, camDist);
    reliefBare = smoothstep(0.14, 0.9, relief).mul(reliefNear) as unknown as NF;
    reliefRock = smoothstep(0.9, 1.7, relief).mul(reliefNear) as unknown as NF;
  }

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

  // Flat palette constants below = the old noise-mottled palettes folded at
  // their carriers' neutral mean (no-fake-detail law: color variation must come
  // from REAL cooked fields — classId, soil, geology, moisture — not noise).
  const scree = vec3(0.344, 0.33, 0.31);
  let soil = vec3(0.198, 0.158, 0.11) as unknown as NV3;
  if (geo) {
    // surficial family → bare-ground mineral tints (soil feeds the composite
    // base AND litter, so one chain covers every place bare ground shows —
    // including the relief exposure patches). The id windows are mutually
    // exclusive.
    const sandT = mix(vec3(0.42, 0.35, 0.22), vec3(0.5, 0.44, 0.31), geo.aeolian);
    soil = mix(soil, sandT, geo.sand.mul(0.55)) as unknown as NV3;
    soil = mix(soil, vec3(0.235, 0.2, 0.155), geo.till.mul(0.45)) as unknown as NV3;
    soil = mix(soil, vec3(0.38, 0.36, 0.33), geo.gravel.mul(0.5)) as unknown as NV3;
    soil = mix(soil, vec3(0.075, 0.058, 0.042), geo.peat.mul(0.6)) as unknown as NV3;
  }
  // grass field color = the FINAL grass LOD: matched to the blade-ring palette
  // (screen-average of the blade ramps) so the geometric grass dissolves into
  // this instead of ending at a visible ring edge ("empty terrain" feedback)
  const grassCol = vec3(0.047, 0.109, 0.023);
  // forest floor: litter brown blended w/ moss by moisture
  const litter = mix(soil, vec3(0.18, 0.15, 0.095), 0.5);
  const mossy = vec3(0.11, 0.185, 0.065);
  const forestFloor = mix(litter, mossy, smoothstep(0.45, 0.8, moisture).mul(0.7));
  // gravel/cobble tint in stream channels
  const gravel = vec3(0.405, 0.39, 0.37);
  const snowCol = vec3(0.886, 0.906, 0.955);

  // ---------- class weights ------------------------------------------------------
  const rockW = (
    smoothstep(0.62, 1.15, classSlope).max(rockExposure.mul(0.85)) as unknown as NF
  ).toVar();
  if (reliefRock) rockW.assign(rockW.max(reliefRock.mul(0.9)));
  if (geo) {
    // mapped bedrock-exposure polygons (flag bit 4) BIAS exposure where slope
    // already suggests rock — a prior on the weight, never a placed cliff.
    rockW.assign(rockW.max(smoothstep(0.3, 0.85, classSlope).mul(geo.expose).mul(0.55)));
  }
  const screeW = smoothstep(0.42, 0.62, classSlope)
    .mul(smoothstep(1.15, 0.7, classSlope))
    .mul(smoothstep(380, 700, h))
    .mul(rockW.oneMinus()) as unknown as NF;
  let grassW = smoothstep(0.5, 0.22, classSlope)
    .mul(vegDensity)
    .mul(zm.tKarst.mul(0.5).oneMinus())
    .mul(rockW.oneMinus()) as unknown as NF;
  if (reliefBare) grassW = grassW.mul(reliefBare.mul(0.85).oneMinus()) as unknown as NF;
  let forestW = vegDensity
    .mul(smoothstep(0.9, 0.45, classSlope))
    .mul(smoothstep(0.25, 0.6, moisture.add(zm.tKarst.mul(0.3))))
    .mul(rockW.oneMinus()) as unknown as NF;
  if (reliefBare) forestW = forestW.mul(reliefBare.mul(0.5).oneMinus()) as unknown as NF;
  // gravel only for REAL channels on open ground: weak-flow rills under
  // grass painted pale streaks down every meadow hillside — those should
  // darken via moisture instead
  const riverW = smoothstep(0.3, 0.68, flowStrength)
    .mul(smoothstep(0.45, 0.2, classSlope))
    .mul(grassW.mul(0.75).oneMinus());

  // snow edge from the real snow field only (the former hash dither was
  // procedural edge noise — removed under the no-fake-detail law)
  const snowW = smoothstep(0.16, 0.5, snowField).toVar();

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
  // field texel each sit on some soil texture. DATA-DRIVEN: texCore sets a mineral tint
  // (sand pale-warm → loam neutral → clay red-brown → peat dark), boniteet enriches the
  // ground flora. (The former stoniness-scaled value-noise pebble speckle is removed:
  // its visible pattern was the noise carrier, not cooked data — no-fake-detail law.)
  // Applied to the base composite BEFORE the classId overrides (which mix() fully on
  // top ⇒ near-zero double-count). Everything sits INSIDE `if (inp.hasSoil)` so the
  // generated world (no soil layer ⇒ hasSoil=false) never constructs a single soil
  // node ⇒ BIT-IDENTICAL.
  if (inp.hasSoil) {
    const soilS = field.soilAt(wxzB); // [texCore, stoniness, boniteet, texSkeleton] byte/255
    const texCore = (soilS.x as unknown as NF).mul(255); // 0..14 (255 unparseable, 0 no-data)
    const boni = (soilS.z as unknown as NF).mul(255).div(100).clamp(0, 1); // 0..100 → [0,1]
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
    const classInterior = smoothstep(0.55, 0.12, cidLinear.sub(cid).abs()) as unknown as NF;
    const notRock = rockW.oneMinus();
    // bare open ground — barren(4) → warm sand(5). Only where the veg field agrees
    // (suppresses the barren/sand ids a forest↔field sweep crosses in the seam).
    const bareW = smoothstep(3.55, 4.0, cid)
      .mul(smoothstep(5.5, 5.05, cid))
      .mul(smoothstep(0.45, 0.15, vegDensity))
      .mul(notRock)
      .mul(classInterior)
      .mul(0.9);
    const bareCol = mix(vec3(0.291, 0.262, 0.209), vec3(0.456, 0.398, 0.291), smoothstep(4.5, 5.0, cid));
    // arable field(6): tilled earthy tone — warmer and more uniform than the
    // natural meadow green. Flat ground only.
    const fieldW = smoothstep(5.55, 6.0, cid)
      .mul(smoothstep(6.65, 6.15, cid))
      .mul(smoothstep(0.5, 0.2, classSlope))
      .mul(notRock)
      .mul(classInterior)
      .mul(0.72);
    const fieldCol = vec3(0.164, 0.137, 0.082);
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
    const bogFen = mix(vec3(0.112, 0.083, 0.05), vec3(0.083, 0.097, 0.054), smoothstep(8.0, 9.2, cid));
    const peatCol = mix(bogFen, vec3(0.045, 0.038, 0.03), smoothstep(9.4, 10.1, cid));
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

  // ---------- lighting normal --------------------------------------------------
  // REAL cooked relief ONLY. Every procedural normal term (far ridged
  // re-amplification, fbm meso/micro bumps, the micro-displacement gradient
  // counterpart) is removed — the law: relief/shading detail comes from real
  // geometry. The XZ gradient of the real surface is amplified for the
  // LIGHTING normal so the genuine 6 cm relief reads (see RELIEF_LIGHT_K);
  // it fades with distance as the field pyramid coarsens, exactly like the
  // mesh. PERF: −6 noise fetches + gradient/projection ALU per fragment.
  const nrm: NV3 = vec3(
    baseNormal.x.mul(RELIEF_LIGHT_K),
    baseNormal.y,
    baseNormal.z.mul(RELIEF_LIGHT_K),
  ).normalize();

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
