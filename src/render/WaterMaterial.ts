/**
 * Stream/lake water shading (Phase 6). One material per clipmap level of the
 * WaterSurface mesh (levels differ only in uniforms — pipeline is shared).
 *
 * Composition (all scene-linear, post stack tonemaps later):
 *   emissive = (1−foam) · mix(refraction, skyReflection, fresnel)
 *     refraction: viewportSharedTexture sampled at a ripple-refracted uv,
 *       depth-validated (samples landing on geometry IN FRONT of the water
 *       fall back to the straight uv), Beer–Lambert absorbed by the water
 *       column thickness from viewportDepthTexture, plus turbidity
 *       in-scatter tied to the sky so it tracks time-of-day.
 *     reflection: sky-view LUT along the reflected ray (streams reflect sky;
 *       lakes upgrade to a planar pass later in Phase 6).
 *   diffuse (colorNode) = foam albedo — lit by sun/CSM/GI like any surface,
 *     so foam in cliff shade goes properly dim.
 *   PBR spec from the scene sun (roughness ~0.05 + ripple normals) supplies
 *   glints with cast shadows; the emissive reflection is sky-dome only, so
 *   nothing double-counts.
 *
 * Ripples: two fbm-gradient layers advected along the hydrology flow field
 * with the classic two-phase flowmap blend (no sliding-texture artifact).
 * |flowDir| encodes speed and is ZERO in lakes — they fall back to a faint
 * breeze ripple. Normals come from NoiseBake's pre-derived d(fbm)/dxz, so a
 * ripple layer costs one texture fetch.
 */

import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  Loop,
  abs,
  cameraFar,
  cameraNear,
  cameraPosition,
  cameraProjectionMatrix,
  cameraViewMatrix,
  clamp,
  dot,
  exp,
  float,
  fract,
  normalize,
  getScreenPosition,
  interleavedGradientNoise,
  mix,
  perspectiveDepthToViewZ,
  viewZToPerspectiveDepth,
  positionLocal,
  positionView,
  positionWorld,
  reflect,
  screenCoordinate,
  screenUV,
  smoothstep,
  texture,
  time,
  transformNormalToView,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { DepthTexture, Texture } from 'three';
import type { StorageTexture } from 'three/webgpu';
import { PERIOD_FBM } from '../gpu/passes/NoiseBake';
import { bilerpVec2Buffer } from '../gpu/BufferSample';
import { sunU } from './VegMaterials';
import { canopyAt } from '../gpu/passes/Scatter';
import type { ProbeGI } from '../gpu/passes/ProbeGI';
import type { NB, NF, NI, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import type { Atmosphere } from '../sky/Atmosphere';
import type { TerrainField } from '../nanite/world/TerrainField';
import type { Heightfield } from '../world/Heightfield';

/** clear alpine water: absorption per meter (r dies first → teal depths) */
const SIGMA = { r: 0.42, g: 0.135, b: 0.095 };

/** flowmap cycles/s — shared by ripples, foam and the caustic advection */
export const FLOW_CYC = 0.45;

/** W2: composed sun visibility (clipmap PCSS × cloud × far-shadow) at a world pos */
export type WaterSunVis = (wp: NV3, n: NV3) => NF;

export interface WaterLevelHandles {
  /** snapped world origin of this clipmap level (uniform, updated per frame) */
  origin: NV2;
  /** world rect (minX, minZ, maxX, maxZ) of the next-finer level — discarded here */
  innerRect: NV4;
  /** cell size in meters (compile-time constant per level) */
  cell: number;
  /** coarse level → sample the min-reduced far field (narrow channels
   *  vanish at distance instead of stretching across whole cells) */
  far: boolean;
}

export function waterMaterial(
  // hf carries ONLY the hydrology flow field + baked ripple noise (ripple/foam
  // advection, caustic drift). The water SURFACE (waterY/waterYFar) reads the
  // TerrainField water plane since the S9 water port — the one path both sources
  // ride, so Estonia's streamed water renders through this same material; terrain
  // HEIGHT reads live on the TerrainField planes since S3a.
  hf: Heightfield,
  field: TerrainField,
  atm: Atmosphere,
  canopyTex: StorageTexture | null,
  gi: ProbeGI | null,
  lvl: WaterLevelHandles,
  // once-per-frame scene snapshots owned by WaterSurface (NOT three's
  // viewportSharedTexture/viewportDepthTexture — those dedupe per node
  // INSTANCE and fired 12 color + 18 depth copies/frame across the 6 levels).
  // sunVis (W2): present in the severed-CSM nanite slate — switches foam/glint
  // to manual shadowed lighting (lights = false).
  snap: { color: Texture; depth: DepthTexture; sunVis?: WaterSunVis },
): MeshStandardNodeMaterial {
  const flow = hf.flow;
  const noiseA = hf.noiseA;
  if (!flow || !noiseA) throw new Error('waterMaterial needs hydrology + baked noise');
  const sceneDepthAt = (uv: NV2): NF => (texture(snap.depth, uv) as unknown as NV4).x;
  const sceneColorAt = (uv: NV2): NV3 => (texture(snap.color, uv) as unknown as NV4).rgb;

  const mat = new MeshStandardNodeMaterial();
  mat.name = 'waterSurface';
  mat.transparent = true;
  mat.depthWrite = true;
  mat.metalness = 0;

  // ---- vertex: clipmap grid (cell units) → world water surface ----------------
  // waterY from the TerrainField water plane (S9 port): generated fills it from
  // hf.cpuWaterY (bit-identical addressing to the retired hf.sampleWaterY tap —
  // same texel-centered sim-res lattice), Estonia from its streamed water layer.
  // coverOn = a watercover α plane exists (Estonia) and ?watercover != 0. false on
  // the generated world (hasWaterCoverage=false) ⇒ every #GAP/coverage gate below
  // compiles the pre-coverage graph VERBATIM (bit-identical generated water).
  const coverOn = field.hasWaterCoverage && new URLSearchParams(window.location.search).get('watercover') !== '0';
  // #GAP wetSurf = Estonia's NEAR water surface. Its dry cells hold the −1e4 dry
  // sentinel, so a plain bilinear near a shore blends the real water level with −1e4
  // and PLUNGES into a pit (the shore gap). Sample WET-PREFERRING (fieldWaterYWet
  // masks the sentinel corners out of the 2×2 — the near analog of the #115 far
  // maxReduce) so the surface stays FLAT at the true level up to the α shoreline,
  // then clamp the degenerate all-dry vertex to just under the bed so a shore-
  // crossing triangle MEETS the bank instead of diving. Generated ⇒ false ⇒ the
  // exact old fieldWaterY tap + unclamped position.
  const wetSurf = coverOn && !lvl.far;
  const sampleY = (q: NV2): NF =>
    lvl.far ? field.fieldWaterYFar(q) : wetSurf ? field.fieldWaterYWet(q) : field.fieldWaterY(q);
  const wxz = lvl.origin.add(positionLocal.xz.mul(lvl.cell));
  const surfY = wetSurf ? sampleY(wxz).max(field.fieldHeightFinest(wxz).sub(0.5)) : sampleY(wxz);
  mat.positionNode = vec3(wxz.x, surfY, wxz.y);

  // ---- inner-level cutout + hard world bounds ----------------------------------
  // Outside the source's coverage box the field samples clamp to the border texel
  // — a wet border cell would extend an infinite water band into the far shell. The
  // box IS the world extent for BOTH sources: ±WORLD_HALF on the generated world
  // (so this is bit-identical to the old WORLD_HALF clamp), the whole Estonia AOI on
  // the streamed world (the old WORLD_HALF clamp masked ALL of Estonia's ~311 km
  // coords → water never rendered; that was the second half of the S9 water bug).
  const p = positionWorld.xz;
  const r = lvl.innerRect;
  const insideInner = p.x
    .greaterThan(r.x)
    .and(p.y.greaterThan(r.y))
    .and(p.x.lessThan(r.z))
    .and(p.y.lessThan(r.w));
  const cb = field.coverageBox;
  const inWorld = p.x
    .greaterThan(cb.minX + 4)
    .and(p.x.lessThan(cb.maxX - 4))
    .and(p.y.greaterThan(cb.minZ + 4))
    .and(p.y.lessThan(cb.maxZ - 4));
  // WORLD-SPACE wetness guard (?watermask, default on). Dry cells encode the sheet
  // at neighbourhood-min bed − 2 m (buildWaterY); it normally loses the hardware
  // depth test / thick-based opacity to the terrain 2 m above. At long range the
  // 0.3→30000 m depth buffer cannot resolve that 2 m (NDC-z compression), so the
  // dive stops hiding the sheet and water speckles onto dry hills. Test the real
  // signal: keep the fragment only where the surface sits above the terrain bed.
  // Strict no-op near camera (where depth already hid dry land); −0.75 m margin
  // absorbs neighbourhood-min vs local-bed at the ~2 m sim texel; shoreline crosses
  // 0 exactly at the wet→dry bilinear edge so the opacity feather still finishes it.
  const wetGuard = new URLSearchParams(window.location.search).get('watermask') !== '0';
  // #114/#115 SHORELINE: the cooked anti-aliased coverage α (Estonia's watercover plane)
  // makes the wet edge track α's 0.5 iso-contour — bilinear of a FRACTION resolves the
  // sub-texel shore the binary water mask could only quantize to grid squares. The NEAR
  // levels (cell < 12 m) read the LOD0 α (fieldWaterCoverage); the FAR levels (#115) read
  // the ×8 mean-reduced α (fieldWaterCoverageFar) with a MAX-reduced wet surface, so the
  // far shore stops quantizing to blocky 16 m squares (was: the min-reduced bed dive). A
  // source with NO watercover plane (the generated world) has hasWaterCoverage=false ⇒ the
  // OLD binary dive guard + old opacity graph compile VERBATIM (bit-identical generated water).
  // ?watercover=0 forces the old binary edge (an A/B toggle beside ?watermask —
  // default on where a coverage plane exists). No-op on the generated world.
  // (coverOn is hoisted above the vertex block — the #GAP wetSurf gate needs it.)
  let wet: NB;
  let coverFeather: NF | null = null;
  if (coverOn && !lvl.far) {
    const cov = field.fieldWaterCoverage(p);
    wet = cov.greaterThan(0.5) as unknown as NB;
    coverFeather = smoothstep(0.35, 0.65, cov) as unknown as NF;
  } else if (coverOn && lvl.far) {
    const covFar = field.fieldWaterCoverageFar(p);
    wet = covFar.greaterThan(0.5) as unknown as NB;
    coverFeather = smoothstep(0.35, 0.65, covFar) as unknown as NF;
  } else {
    const bedH = lvl.far ? field.fieldHeightFinestNearest(p) : field.fieldHeightFinest(p);
    wet = positionWorld.y.greaterThan(bedH.sub(0.75)) as unknown as NB;
  }
  mat.maskNode = wetGuard ? insideInner.not().and(inWorld).and(wet) : insideInner.not().and(inWorld);

  // ---- flow field --------------------------------------------------------------
  const simRes = hf.simRes;
  const g = clamp(positionWorld.xz.div(4096).add(0.5), 0, 1).mul(simRes).sub(0.5);
  const flowV = bilerpVec2Buffer(flow.flowDir, simRes, g);
  const spd = flowV.length();
  const fdir = flowV.div(spd.max(1e-4));

  // ---- ripple normal: two-phase flowmap over fbm gradients ---------------------
  const CYC = 0.45; // flowmap cycles/s
  const ph1 = fract(time.mul(CYC));
  const ph2 = fract(time.mul(CYC).add(0.5));
  const w2 = abs(ph1.sub(0.5)).mul(2);
  // advection velocity (m/s): rivers stream, lakes get a faint breeze drift
  const vel = fdir.mul(spd.mul(1.9)).add(vec2(0.045, 0.03));
  const gradAt = (s: number, off: NV2): NV2 =>
    (texture(noiseA, positionWorld.xz.sub(off).div(s * PERIOD_FBM)) as unknown as NV4).zw.div(s);
  const offA = vel.mul(ph1.div(CYC));
  const offB = vel.mul(ph2.div(CYC)).add(vec2(3.71, 1.13));
  const layer = (off: NV2): NV2 => gradAt(0.9, off).add(gradAt(3.4, off.mul(0.62)).mul(0.5));
  const grad = mix(layer(offA), layer(offB), w2);
  // baked fbm gradients are ±(3..10)/m at these scales — the old amp
  // (0.018+0.085·spd) tilted normals 8–30° everywhere, saturating fresnel
  // to ~1 and turning every stream into a sky mirror ("white sheet")
  const rippleAmp = float(0.007).add(spd.mul(0.028));
  const slope = grad.mul(rippleAmp);
  const n = vec3(slope.x.negate(), 1, slope.y.negate()).normalize();
  mat.normalNode = transformNormalToView(n);

  // ---- view / depth ------------------------------------------------------------
  const toCam = cameraPosition.sub(positionWorld);
  const dist = toCam.length();
  const viewDir = toCam.div(dist.max(1e-4));
  const fragZ = positionView.z; // negative

  // refraction uv: ripple-driven, shrinking with distance, depth-validated
  const refrK = clamp(float(9).div(dist.max(1)), 0.04, 1).mul(0.055);
  const ruv = screenUV.add(n.xz.mul(refrK));
  const zR = perspectiveDepthToViewZ(sceneDepthAt(ruv), cameraNear, cameraFar);
  const leaked = zR.greaterThan(fragZ.add(0.02)); // refr sample in FRONT of water
  const uvF = mix(ruv, screenUV, leaked.select(float(1), float(0)));
  const zScene = mix(
    zR,
    perspectiveDepthToViewZ(sceneDepthAt(screenUV as unknown as NV2), cameraNear, cameraFar),
    leaked.select(float(1), float(0)),
  );
  const thick = fragZ.sub(zScene).max(0); // meters of water along the ray
  // vertical water column under this fragment (foam/shore feather)
  const vDepth = thick.mul(viewDir.y.abs().max(0.06));

  // ---- transmitted light --------------------------------------------------------
  const sceneCol = sceneColorAt(uvF as unknown as NV2);
  const absorb = thick.mul(1.25);
  const T = vec3(
    exp(absorb.mul(-SIGMA.r)),
    exp(absorb.mul(-SIGMA.g)),
    exp(absorb.mul(-SIGMA.b)),
  );
  // turbidity in-scatter follows the zenith sky → tracks time-of-day
  const inscat = atm.skyColor(vec3(0, 1, 0)).mul(vec3(0.013, 0.036, 0.032));
  const refr = sceneCol.mul(T).add(inscat.mul(vec3(1, 1, 1).sub(T)));

  // ---- reflection: screen-space march with sky fallback ---------------------------
  // Streams at grazing angles must reflect the far bank / trees (dark), not
  // bright horizon haze — sky-only reflection read as a white sheet. March
  // the opaque depth buffer along the reflected ray; misses fall back to
  // the sky-view LUT.
  const rdir = reflect(viewDir.negate(), vec3(n.x.mul(0.55), n.y, n.z.mul(0.55)).normalize());
  const reflection = Fn((): NV3 => {
    const dirV = cameraViewMatrix.mul(vec4(rdir, 0)).xyz;
    // far cap 28 m: a grazing lake reflects its far tree line — with a
    // 12 m cap the march died ~200 m short and the whole far band fell to
    // the FLAT probe fallback (read as a dark slab hovering on the lake)
    const stepLen = clamp(dist.mul(0.09), 0.25, 28);
    const jitter = interleavedGradientNoise(screenCoordinate.xy);
    const hit = float(0).toVar();
    const hitUv = vec2(0, 0).toVar();
    Loop(18, ({ i }: { readonly i: NI }) => {
      const t = float(i).add(jitter).mul(stepLen);
      const pV = positionView.add(dirV.mul(t));
      const uvS = getScreenPosition(pV, cameraProjectionMatrix) as unknown as NV2;
      If(
        uvS.x.lessThan(0).or(uvS.x.greaterThan(1)).or(uvS.y.lessThan(0)).or(uvS.y.greaterThan(1)),
        () => {
          Break();
        },
      );
      const zS = perspectiveDepthToViewZ(sceneDepthAt(uvS), cameraNear, cameraFar);
      // hit: scene surface just in front of the ray point (viewZ is negative)
      If(
        zS.greaterThan(pV.z.add(0.06)).and(zS.lessThan(pV.z.add(stepLen.mul(2.6).add(0.7)))),
        () => {
          hit.assign(1);
          hitUv.assign(uvS);
          Break();
        },
      );
    });
    // sky fallback (horizon-clamped so the LUT never samples below ground)
    const rdirUp = vec3(rdir.x, rdir.y.max(0.035), rdir.z).normalize();
    const sky = atm.skyColor(rdirUp);
    // CROWNED-HORIZON occlusion: when the SSR march misses, "sky" is only
    // correct if the reflected ray clears terrain AND tree crowns. March
    // the height field at log-spaced ranges with the canopy map raising
    // the tested horizon by crown height — one test covers both regimes:
    // steep gorge-stream rays get caught by overhead crowns (dark wall/
    // canopy mirror, scene1), grazing lake rays clear the far tree line
    // into open sky (a blanket canopy multiply here used to crush the
    // far-lake band to black). Occluded rays fall back to the probe field
    // toward the ray — it already encodes wall/canopy brightness.
    const horizonVis = float(1).toVar();
    for (const dRay of [9, 24, 65, 180]) {
      const q = positionWorld.xz.add(rdir.xz.mul(dRay));
      const rayY = positionWorld.y.add(rdir.y.mul(dRay));
      let hQ = field.fieldHeightFinestNearest(q);
      if (canopyTex) {
        hQ = hQ.add(canopyAt(canopyTex, q).mul(16)) as NF;
      }
      // wide knee — a hard threshold printed razor-edged reflection bands
      horizonVis.mulAssign(smoothstep(-16, 7, rayY.sub(hQ)));
    }
    const wallAmb = gi
      ? (gi.irradiance(positionWorld, rdir).mul(0.65) as unknown as NV3)
      : (sky.mul(0.18) as unknown as NV3);
    // ripple-jittered blend breaks the residual banding at the transition
    const vJit = n.x.add(n.z).mul(0.18);
    const fallback = mix(wallAmb, sky as unknown as NV3, horizonVis.add(vJit).clamp(0, 1));
    // fade SSR toward the screen border so hits don't pop at the edge
    const e = hitUv.sub(0.5).abs().mul(2);
    const edgeFade = smoothstep(1.0, 0.82, e.x.max(e.y));
    const scene = sceneColorAt(hitUv as unknown as NV2);
    return mix(fallback, scene, hit.mul(edgeFade));
  })();
  const skyRefl = reflection as unknown as NV3;
  // fresnel on a FLATTENED normal (standard water practice): per-pixel
  // ripple tilt makes (1−cosθ)^5 explode at any view angle — reflectance
  // weight should follow the mean surface, the ripples only shape WHAT is
  // reflected (rdir above keeps the full normal)
  const nFres = vec3(n.x.mul(0.3), n.y, n.z.mul(0.3)).normalize();
  const cosT = clamp(viewDir.dot(nFres), 0.0, 1.0);
  const fres = float(0.02).add(float(0.98).mul(cosT.oneMinus().pow(5)));

  // ---- foam ----------------------------------------------------------------------
  // Two-phase advection like the ripple normals — a linearly time-advected
  // pattern slides coherently and its thresholded fbm level sets read as
  // sharp white stripes (user-reported). Two decorrelated scales multiply
  // into clumpy patches instead of bands.
  const foamUv = (off: NV2, s: number): NV2 => positionWorld.xz.sub(off).div(s * PERIOD_FBM);
  // EVERY octave must live inside the two-phase blend: phase A's offset
  // snaps to zero at its cycle wrap, and the blend only hides that for
  // terms weighted by w2 — a detail octave pinned to offA alone snapped
  // visibly once per cycle (user-reported "sharp stop in the loop").
  const fA = (texture(noiseA, foamUv(offA, 0.55)) as unknown as NV4).y;
  const fB = (texture(noiseA, foamUv(offB.mul(1.13), 0.55)) as unknown as NV4).y;
  const dA = (texture(noiseA, foamUv(offA.mul(0.6), 0.21)) as unknown as NV4).y;
  const dB = (texture(noiseA, foamUv(offB.mul(0.71), 0.21)) as unknown as NV4).y;
  // renormalize the crossfade variance — averaging two uncorrelated fields
  // flattens the pattern at blend midpoints and thresholded coverage pulses
  const varNorm = w2.mul(w2).add(w2.oneMinus().mul(w2.oneMinus())).sqrt();
  const fblend = mix(fA, fB, w2).sub(0.5).div(varNorm).add(0.5);
  const fDetail = mix(dA, dB, w2).sub(0.5).div(varNorm).add(0.5);
  const foamPat = smoothstep(0.42, 0.85, fblend.mul(0.62).add(fDetail.mul(0.38)));
  const shoreFoam = smoothstep(0.16, 0.03, vDepth).mul(0.42);
  // rapids key on the DROP of the water surface along flow (a large calm
  // river has high strength but no whitewater — slope is what froths).
  // Window starts at ~3% grade: a 1.5% start blanketed every gorge reach
  // in white (real streams run clear on smooth grades and froth at STEPS,
  // which survive in the smoothed field as locally steeper sub-reaches).
  const drop = sampleY(positionWorld.xz)
    .sub(sampleY(positionWorld.xz.add(fdir.mul(3))))
    .div(3);
  const rapidFoam = smoothstep(0.09, 0.24, drop).mul(smoothstep(0.18, 0.55, spd)).mul(0.8);
  const foam = clamp(shoreFoam.add(rapidFoam), 0, 1).mul(foamPat).clamp(0, 0.68) as NF;

  // ---- compose --------------------------------------------------------------------
  const foamAlb = vec3(0.74, 0.76, 0.74);
  const rough = mix(float(0.05), float(0.55), foam);
  if (snap.sunVis) {
    // W2 (severed-CSM slate): three's directional light carries no shadow here, so
    // foam diffuse + the sun glint go MANUAL, gated by the composed nanite sun
    // visibility (clipmap PCSS × cloud × far-shadow) at the water's own wp — same
    // energy convention as the resolve (irradiance = NdotL·sunColor, ÷π once).
    mat.lights = false;
    mat.colorNode = vec3(0);
    mat.emissiveNode = Fn(() => {
      const sunDir = normalize(vec3(sunU.dir)) as unknown as NV3;
      const sunCol = (sunU.color as unknown as NV3).mul(float(sunU.intensity)) as unknown as NV3;
      const ndl = clamp(dot(n, sunDir), 0, 1);
      const amb = gi
        ? (gi.irradiance(positionWorld, vec3(0, 1, 0)) as unknown as NV3)
        : (atm.skyColor(vec3(0, 1, 0)).mul(0.35) as unknown as NV3);
      // sun glint: GGX D × Schlick F × Kelemen V on the full ripple normal — the
      // manual replacement for the MeshStandard directional specular (lights=false).
      // Roughness rides the same foam mix, so foam kills the glint exactly as before.
      const hv = normalize(sunDir.add(viewDir));
      const ndh = clamp(dot(n, hv), 0, 1);
      const ldh = clamp(dot(sunDir as unknown as NV3, hv), 0, 1);
      const aG = rough.mul(rough);
      const a2 = aG.mul(aG);
      const dDen = ndh.mul(ndh).mul(a2.sub(1)).add(1);
      const ggxD = a2.div(dDen.mul(dDen).mul(Math.PI));
      const glintF = float(0.02).add(float(0.98).mul(ldh.oneMinus().pow(5)));
      const kelemenV = float(0.25).div(ldh.mul(ldh).max(1e-3));
      const specRaw = ggxD.mul(glintF).mul(kelemenV).mul(ndl);
      // PCSS is the expensive term (6 blocker + 9 PCF taps) and sunVis only
      // feeds foam + glint — skip it where both are invisible (most of any
      // open lake). Real branch, not select(): measured +0.7-2 ms at full-lake
      // coverage when evaluated unconditionally (2026-07-03).
      const sv = float(1).toVar();
      If(foam.greaterThan(0.004).or(specRaw.greaterThan(0.002)), () => {
        sv.assign(snap.sunVis!(positionWorld as unknown as NV3, nFres as unknown as NV3));
      });
      const foamLit = foamAlb.mul(sunCol.mul(ndl.mul(sv)).add(amb)).mul(1 / Math.PI);
      return mix(refr, skyRefl, fres)
        .mul(foam.oneMinus())
        .add(foamLit.mul(foam))
        .add(sunCol.mul(specRaw).mul(sv));
    })() as unknown as typeof mat.emissiveNode;
  } else {
    mat.colorNode = foamAlb.mul(foam);
    mat.emissiveNode = mix(refr, skyRefl, fres).mul(foam.oneMinus());
  }
  mat.roughnessNode = rough;
  // shoreline feather: mm-deep water fades out over the bed. ALSO fade
  // steep surface RAMPS: the field dives ~2 m to the dry sentinel past
  // every shoreline — across a FLAT far beach seen edge-on that dive
  // renders as a thick dark band hugging the shore (twin-lake artifact).
  // Hydrology gates real water to gentle slopes (rdGate), so any render
  // slope ≥ ~30° is a dive, never water.
  const eS = 2.0;
  const gWx = sampleY(positionWorld.xz.add(vec2(eS, 0)))
    .sub(sampleY(positionWorld.xz.sub(vec2(eS, 0))))
    .div(2 * eS);
  const gWz = sampleY(positionWorld.xz.add(vec2(0, eS)))
    .sub(sampleY(positionWorld.xz.sub(vec2(0, eS))))
    .div(2 * eS);
  // far levels: rampK=1. The generated world's far surface = PlaneFill.minReduce
  // of the near plane, whose shore dip IS the wet gate (fading it would expose the
  // dark silt bed as a rim band); the #115 Estonia far-coverage path instead
  // feathers the shore via coverFeather (from covFar) below. Near levels fade the
  // steep field dive.
  // #GAP Estonia near (wetSurf): DISABLE the dive-fade — the wet-preferring surface
  // no longer dives at the shore, so fading on |∇surfaceY| would only erase the real
  // shoreline; coverFeather (from the α) owns the shore opacity there. Far ⇒ 1
  // (unchanged). Generated near ⇒ the exact old smoothstep dive-fade (bit-identical).
  const rampK =
    lvl.far || wetSurf ? float(1) : smoothstep(0.55, 0.3, vec2(gWx, gWz).length());
  // #114/#115: feather the very shore by the coverage fraction (near AND far coverage
  // levels). Null on the generated path ⇒ the EXACT old opacity graph (bit-identical
  // generated water).
  const opacityBase = smoothstep(0.004, 0.05, vDepth).mul(rampK);
  mat.opacityNode = (coverFeather ? opacityBase.mul(coverFeather) : opacityBase).mul(0.985);

  // ?waterdbg=N — component probe ladder (1 foam, 2 fresnel, 3 refraction,
  // 4 reflection, 5 column thickness, 6 SSR hit/horizon mix, 7 depth-test
  // forensics: R = stored scene depth (far-stretched), G = own raster depth,
  // B = 1 where water LOSES the depth test — depthTest disabled so the paint
  // lands even where the surface is z-rejected)
  const dbg = Number(new URLSearchParams(window.location.search).get('waterdbg') ?? '0');
  if (dbg > 0) {
    const storedD = sceneDepthAt(screenUV as unknown as NV2);
    const ownD = viewZToPerspectiveDepth(fragZ, cameraNear, cameraFar);
    const paint =
      dbg === 1
        ? vec3(foam)
        : dbg === 2
          ? vec3(fres)
          : dbg === 3
            ? refr
            : dbg === 4
              ? skyRefl
              : dbg === 5
                ? vec3(thick.mul(0.25), vDepth.mul(0.25), 0)
                : dbg === 7
                  ? vec3(
                      ownD.sub(storedD).mul(5000).clamp(0, 1), // R: water LOSES by
                      storedD.sub(ownD).mul(5000).clamp(0, 1), // G: water WINS by
                      storedD.lessThan(1e-6).select(float(1), float(0)), // B: stored ≈ 0
                    )
                  : dbg === 8
                    ? // numeric: own/stored as view DISTANCE / 2000 (decode from
                      // a raw screenshot; needs skyveldbg=raw for NoToneMapping)
                      vec3(
                        fragZ.negate().div(2000).clamp(0, 1),
                        float(0.3)
                          .div(float(1).sub(storedD.mul(float(29999.7).div(30000))))
                          .div(2000)
                          .clamp(0, 1),
                        0,
                      )
                    : (skyRefl as NV3);
    mat.colorNode = vec3(0);
    mat.emissiveNode = paint;
    mat.opacityNode = float(1);
    if (dbg === 7 || dbg === 8) {
      // pure scene-depth read: no z-reject AND no self-writes (six clipmap
      // sheets otherwise occlude each other into the copy)
      mat.depthTest = false;
      mat.depthWrite = false;
    }
  }

  return mat;
}
