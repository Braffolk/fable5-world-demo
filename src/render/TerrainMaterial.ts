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
  /**
   * surface context override (N4 nanite resolve): explicit world position +
   * camera position instead of the vertex-pipeline TSL singletons. The old
   * forward materials omit this and get `positionWorld`/`cameraPosition` —
   * bit-identical graphs by construction. Every other input above is already
   * world-space-parametric (textures fetched by world uv derived from wp).
   */
  surf?: { wp: NV3; camPos: NV3 };
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

export function buildTerrainShading(inp: TerrainShadingInputs): TerrainShading {
  const wp = inp.surf?.wp ?? (positionWorld as unknown as NV3);
  const camPos = inp.surf?.camPos ?? (cameraPosition as unknown as NV3);
  const wxz = wp.xz;
  const h = wp.y;

  // --- baked-noise helpers (uv = world / (scale · channel period)) -----------
  /** value noise [0,1] at world feature scale `s` m */
  const val = (s: number, ox = 0, oz = 0): NF =>
    texture(inp.noiseA, wxz.div(s * PERIOD_VAL).add(vec2(ox, oz))).x;
  /** signed value noise [-1,1] */
  const valS = (s: number, ox = 0, oz = 0): NF => val(s, ox, oz).mul(2).sub(1);
  /** fbm-3 [0,1] */
  const fbmV = (s: number, ox = 0, oz = 0): NF =>
    texture(inp.noiseA, wxz.div(s * PERIOD_FBM).add(vec2(ox, oz))).y;
  /** fbm-3 gradient (d/dx, d/dz in world units at feature scale s) */
  const fbmG = (s: number, ox = 0, oz = 0): NV2 =>
    texture(inp.noiseA, wxz.div(s * PERIOD_FBM).add(vec2(ox, oz))).zw.div(s);
  /** ridged-3 gradient (world units at feature scale s) */
  const ridG = (s: number): NV2 =>
    texture(inp.noiseB, wxz.div(s * PERIOD_RID)).xy.div(s);
  /** 1D band noise [0,1] along an arbitrary phase axis */
  const band = (phase: NF, lane: NF): NF =>
    texture(inp.noiseA, vec2(phase, lane).div(PERIOD_VAL)).x;

  // terrain field context: the TerrainField planes (S3b).
  const field = inp.field;
  const ns: NV4 = field.fieldNormalSlope(wxz);
  const fld = field.fieldsAt(wxz);
  const bio = field.biomeAt(wxz);
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
  // the TerrainField plane taps ARE the field at any distance (the coarsest
  // biome/height levels are country floors) — consumed directly, near and far.
  const snowField = snowRaw;
  const vegDensity = vegRaw;
  const rockExposure = rockRaw;
  const moisture = moistRaw;
  const flowStrength = flowRaw;
  const riverDepth = riverRaw;
  const zm = zoneMasks(wxz, inp.mp);

  // ---------- macro variation (2–50 m breakup — tiling killer) ----------------
  const macroA = val(43.7);
  const macroB = val(11.3, 0.37, 0.61);
  const macroMix = macroA.mul(0.65).add(macroB.mul(0.35));
  const macroTint = macroMix.sub(0.5).mul(0.16); // ±8% value shift

  // ---------- meso/micro detail noise ------------------------------------------
  const meso = fbmV(1.45);
  const micro = val(0.19, 0.71, 0.13);

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
  // cavity dirt: concave-ish micro band darkening
  rockCol = rockCol.mul(meso.mul(0.22).add(0.89)).mul(micro.mul(0.1).add(0.95));

  const scree = vec3(0.36, 0.345, 0.325).mul(meso.mul(0.35).add(0.78));
  const soil = mix(vec3(0.155, 0.12, 0.085), vec3(0.24, 0.195, 0.135), meso).mul(
    micro.mul(0.2).add(0.9),
  );
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
  const rockW = smoothstep(0.62, 1.15, slope).max(rockExposure.mul(0.85)).toVar();
  const screeW = smoothstep(0.42, 0.62, slope)
    .mul(smoothstep(1.15, 0.7, slope))
    .mul(smoothstep(380, 700, h))
    .mul(rockW.oneMinus());
  const grassW = smoothstep(0.5, 0.22, slope)
    .mul(vegDensity)
    .mul(zm.tKarst.mul(0.5).oneMinus())
    .mul(rockW.oneMinus());
  const forestW = vegDensity
    .mul(smoothstep(0.9, 0.45, slope))
    .mul(smoothstep(0.25, 0.6, moisture.add(zm.tKarst.mul(0.3))))
    .mul(rockW.oneMinus());
  // gravel only for REAL channels on open ground: weak-flow rills under
  // grass painted pale streaks down every meadow hillside — those should
  // darken via moisture instead
  const riverW = smoothstep(0.3, 0.68, flowStrength)
    .mul(smoothstep(0.45, 0.2, slope))
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
  const pondK = smoothstep(1.1, 2.6, pondDepth).mul(smoothstep(0.3, 0.12, slope));
  let col: NV3 = soil;
  col = mix(col, grassCol, grassW);
  col = mix(col, forestFloor, forestW);
  col = mix(col, scree, screeW);
  col = mix(col, rockCol, rockW);
  col = mix(col, gravel, riverW.mul(0.85).mul(pondK.oneMinus()));
  col = mix(col, vec3(0.055, 0.052, 0.038), pondK);
  col = mix(col, snowCol, snowW);
  col = col.mul(macroTint.add(1));

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
      .mul(smoothstep(60, 220, wp.sub(camPos).length()))
      .mul(0.55);
    col = col.add(vec3(0.085, 0.1, 0.032).mul(sheenK)) as NV3;
  }

  // gorge/ravine wall vegetation (scene1: ravine walls are NOT bare — they
  // carry moss bands, hanging greens and ledge clumps). Steep faces in damp
  // valleys grow green in noise pockets: fbm bands read as hanging veg,
  // value-noise pockets as ledge clumps. Karst gorges get the most.
  const wallK = smoothstep(0.62, 1.0, slope)
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
    const forestK = smoothstep(0.12, 0.62, cover).mul(smoothstep(1, 6, canopyH));
    const canopyCol = mix(
      vec3(0.038, 0.066, 0.03),
      vec3(0.02, 0.043, 0.024),
      smoothstep(6, 26, canopyH),
    );
    col = mix(col, canopyCol, forestK.mul(0.9)) as NV3;
  }

  // ---------- normal perturbation ---------------------------------------------------
  // far-detail synthesis (Pillar D): serrated normal-domain detail keeps
  // mid/far ridges craggy where geometric density has LOD'd out. DISTANCE-gated.
  const camDist = wp.sub(camPos).length();
  const farK = smoothstep(900, 2600, camDist);
  // pre-baked ridged gradient at 310 m features; ×44 ≈ the old ±22 m
  // finite-difference amplitude (×2: baked noise is [0,1], mx was [-1,1])
  const rg = ridG(310).mul(44 * 2);
  // crag synthesis belongs to ROCK faces — on smooth vegetated hills the
  // ridged gradient field printed parallel pale corrugation streaks
  const farAmp = smoothstep(0.5, 1.1, slope)
    .mul(0.4)
    .add(smoothstep(0.32, 0.7, slope).mul(0.08))
    .mul(farK);
  // never let detail flip the surface away from the sky
  const perturbed = baseNormal.add(vec3(rg.x, 0, rg.y).mul(farAmp));
  let nrm: NV3 = vec3(perturbed.x, perturbed.y.max(0.1), perturbed.z).normalize();

  // near/mid detail (both terms self-fade to nothing with distance — far tiles pay
  // no analytic bump/displacement): scoped block to keep its temporaries local.
  {
    // meso + micro analytic bumps near camera, stronger on rock — baked fbm
    // gradients at two scales (×2e ≈ old FD amplitudes, ×2 range factor)
    const b1 = fbmG(1.45).mul(1.8 * 2);
    const b2 = fbmG(0.19, 0.31, 0.77).mul(0.24 * 2);
    const bumpAmp = mix(float(0.25), float(0.85), rockW)
      .mul(snowW.mul(0.7).oneMinus())
      .mul(farK.oneMinus());
    nrm = nrm
      .add(
        vec3(
          b1.x.mul(0.7).add(b2.x.mul(0.45)),
          0,
          b1.y.mul(0.7).add(b2.y.mul(0.45)),
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
    const gF = fbmG(DISP.sF1).mul(2 * DISP.wF1);
    const gR = ridG(DISP.sRid).mul(
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
