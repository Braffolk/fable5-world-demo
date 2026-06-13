/**
 * Material resolve (N4) — the fullscreen-triangle mesh that turns the vis
 * buffer into shaded pixels INSIDE the main scene pass (D-N18): renders first
 * (renderOrder −1000), writes the raster's REAL f32 depth via depthNode, and
 * Discards uncovered pixels so the cleared depth survives for the sky.
 *
 * ARCHITECTURE (reset after the C1 fragility — D-N20): a CLIP-SPACE fullscreen
 * triangle + a plain NodeMaterial with a fragmentNode. This is the C0 path
 * that provably rendered near-camera terrain (the `?nanitedbg=flat` view).
 * The earlier camera-glued near-plane triangle + MeshPhysicalNodeMaterial
 * failed to compile its lighting (missing geometry normal) and fell back to a
 * material that ignored every node — leaving the near terrain transparent.
 *
 * Shading is computed in the fragment from the reconstructed surface
 * (manual lighting, D-N17): TERRAIN runs buildTerrainShading on the
 * reconstructed world position, then sun + sky-ambient (+ probe GI). CSM
 * receive + exact IBL parity are layered in next; the immediate contract is
 * that the terrain is THERE (opaque, correct albedo), not transparent.
 *
 * Bisects (URL-gated): ?nandbg=flat (matClass palette, no shading),
 * ?nandbg=albedo|normal|cov, ?nandepth=0 (depth write off).
 */

import { Mesh, Sphere, Vector3 } from 'three';
import { BufferGeometry, Float32BufferAttribute } from 'three';
import type { Texture } from 'three';
import { NodeMaterial, type StorageTexture } from 'three/webgpu';
import {
  Discard,
  Fn,
  If,
  cameraProjectionMatrixInverse,
  cameraWorldMatrix,
  dot,
  float,
  getViewPosition,
  max,
  mix,
  nodeObject,
  normalize,
  positionGeometry,
  screenCoordinate,
  screenUV,
  smoothstep,
  uint,
  vec3,
  vec4,
} from 'three/tsl';
import type { NF, NV3, NV4 } from '../gpu/TSLTypes';
import type { CSMShadowNode } from 'three/addons/csm/CSMShadowNode.js';
import { causticContext, causticDepth, causticTint } from '../render/Caustics';
import { buildTerrainShading } from '../render/TerrainMaterial';
import { sunU } from '../render/VegMaterials';
import { canopyAt } from '../gpu/passes/Scatter';
import type { ProbeGI } from '../gpu/passes/ProbeGI';
import type { Heightfield } from '../world/Heightfield';
import { MESH_WORDS } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import type { NaniteCam } from './NaniteCommon';
import type { NaniteVisBuffers } from './NaniteRaster';
import { bcU2F, elemU } from './Tsl';
import type { BufOf, UV2 } from './Tsl';

export interface NaniteResolveHandles {
  /** add to engine.scene; renderOrder −1000, castShadow off */
  mesh: Mesh;
}

export interface ResolveWorld {
  hf: Heightfield;
  gi: ProbeGI | null;
  canopyTex: StorageTexture | null;
  /** sun CSM cascades (D-N17 shadow receive) — sampled at the reconstructed
   *  world position via receivedShadowPositionNode; null = no sun shadows.
   *  At runtime this is the CachedCsmShadowNode (default), whose
   *  shadowPositionWorld-based cascade-select is what makes per-pixel
   *  reconstructed positions select the right cascade (?shadowcache=0 falls
   *  back to the base positionView.z select — a debug-only A/B). */
  csm: CSMShadowNode | null;
}

export function buildNaniteResolve(
  gpu: RegistryGpu,
  heightTex: Texture,
  cam: NaniteCam,
  cull: { qRasterRO: BufOf<UV2> },
  vis: NaniteVisBuffers,
  world: ResolveWorld,
): NaniteResolveHandles {
  void heightTex; // raster-side fetch uses it; the resolve reconstructs wp
  const hf = world.hf;
  if (!hf.biomeTex || !hf.fieldsTex || !hf.noiseA || !hf.noiseB) {
    throw new Error('NaniteResolve: heightfield derived maps missing (boot order)');
  }
  const q = new URLSearchParams(window.location.search);
  const nandepth = q.get('nandepth');
  const nandbg = q.get('nandbg');
  // ?nanshadow=0 — bisect: drop the CSM sun-shadow receive (A/B the term)
  const shadowsOn = world.csm !== null && q.get('nanshadow') !== '0';

  // CLIP-SPACE fullscreen triangle (covers ndc [-1,1]² via (-1,-1),(3,-1),(-1,3))
  const geometry = new BufferGeometry();
  geometry.setAttribute(
    'position',
    new Float32BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
  );
  geometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);

  const mat = new NodeMaterial();
  mat.vertexNode = vec4(positionGeometry.xy, 0, 1) as unknown as typeof mat.vertexNode;

  // D-N17 shadow receive: the CSM cascade-select + sampling read
  // `shadowPositionWorld`, which ShadowBaseNode sources from
  // material.receivedShadowPositionNode. The fullscreen triangle's
  // positionWorld is the clip-space vertex (useless), so supply the
  // per-pixel RECONSTRUCTED world position — self-contained like depthNode,
  // not a closure var, so it builds inside the shadow subgraph cleanly.
  if (shadowsOn) {
    (mat as unknown as { receivedShadowPositionNode?: unknown }).receivedShadowPositionNode = Fn(
      () => {
        const fy = float(cam.uH).sub(screenCoordinate.y);
        const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
        const zDev = bcU2F(elemU(vis.depthV.ro, pixelIndex));
        const wpv = getViewPosition(
          screenUV,
          zDev,
          cameraProjectionMatrixInverse,
        ) as unknown as NV3;
        return (
          (cameraWorldMatrix as unknown as { mul(v: NV4): NV4 }).mul(
            (vec4 as unknown as (a: NV3, b: number) => NV4)(wpv, 1),
          ) as unknown as NV4
        ).xyz;
      },
    )();
  }

  mat.fragmentNode = Fn(() => {
    // vis-buffer fetch (bottom-up rows: raster writes y·W+x with y bottom-up,
    // screenCoordinate is top-down, so flip)
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    const dRaw = elemU(vis.depthV.ro, pixelIndex).toVar();
    const pRaw = elemU(vis.payloadV.ro, pixelIndex).toVar();
    If(dRaw.equal(uint(0xffffffff)), () => {
      Discard();
    });
    If(pRaw.equal(uint(0xffffffff)), () => {
      Discard();
    });

    // reconstruct world position from the stored depth (post-chain's pattern)
    const zDev = bcU2F(dRaw);
    const wpv = getViewPosition(screenUV, zDev, cameraProjectionMatrixInverse) as unknown as NV3;
    const wp = (
      (cameraWorldMatrix as unknown as { mul(v: NV4): NV4 }).mul(
        (vec4 as unknown as (a: NV3, b: number) => NV4)(wpv, 1),
      ) as unknown as NV4
    ).xyz.toVar() as unknown as NV3;

    // payload → mesh → matClass
    const itemIdx = pRaw.shiftRight(uint(7));
    const item = cull.qRasterRO.element(itemIdx.add(uint(1)));
    const ci = item.y;
    const meshId = elemU(gpu.clusters, ci.mul(uint(8)).add(uint(7))).shiftRight(uint(16));
    const matClass = elemU(gpu.meshes, meshId.mul(uint(MESH_WORDS)).add(uint(6)))
      .shiftRight(uint(8))
      .bitAnd(uint(0xff))
      .toVar();
    const isT = matClass.equal(uint(0));

    // ---- TERRAIN shading on the reconstructed surface --------------------
    const camPos = vec3(cam.camPos) as unknown as NV3;
    const shading = buildTerrainShading({
      normalTex: hf.normalTex,
      biomeTex: hf.biomeTex as StorageTexture,
      fieldsTex: hf.fieldsTex as StorageTexture,
      noiseA: hf.noiseA as StorageTexture,
      noiseB: hf.noiseB as StorageTexture,
      mp: hf.mp,
      far: false,
      surf: { wp, camPos },
    });
    let terrainCol: NV3 = shading.colorNode;
    let terrainRough: NF = shading.roughnessNode;
    const cctx = causticContext();
    if (cctx) {
      const d = causticDepth(wp);
      const fringe = smoothstep(-0.45, -0.04, d);
      const caust = causticTint(wp, d);
      const biofilm = smoothstep(0.04, 0.5, d);
      let wetCol = terrainCol
        .mul(fringe.mul(0.38).oneMinus())
        .mul(biofilm.mul(0.42).oneMinus()) as unknown as NV3;
      wetCol = mix(wetCol, wetCol.mul(vec3(0.72, 0.86, 0.55)), biofilm.mul(0.65)) as unknown as NV3;
      terrainCol = wetCol.mul(caust.mul(1.7).add(1)) as unknown as NV3;
      terrainRough = shading.roughnessNode.sub(fringe.mul(0.42)).clamp(0.18, 1) as unknown as NF;
    }
    void terrainRough;

    // un-ported classes (rock/bark/deadwood): flat palette
    const isR = matClass.equal(uint(1));
    const isB = matClass.equal(uint(2));
    const isD = matClass.equal(uint(3));
    const palette = isR.select(
      vec3(0.42, 0.41, 0.4),
      isB.select(vec3(0.36, 0.27, 0.19), isD.select(vec3(0.33, 0.28, 0.22), vec3(0.35, 0.33, 0.3))),
    ) as unknown as NV3;
    const albedo = isT.select(terrainCol, palette).toVar() as unknown as NV3;
    const wNormal = isT
      .select(shading.worldNormalNode, vec3(0, 1, 0))
      .toVar() as unknown as NV3;

    // ---- MANUAL lighting (D-N17): sun lambert × CSM shadow + sky ambient +
    // probe GI. The CSM node (proven on the old path) is referenced as a
    // multiplicative factor exactly like AnalyticLightNode does
    // (colorNode.mul(shadowNode)); it carries OUR pcssFilter + the cloud
    // gate, sampling at receivedShadowPositionNode set above. Exact IBL
    // parity for the ambient is the remaining N4-C1 term.
    const sunDir = normalize(vec3(sunU.dir)) as unknown as NV3;
    const nDotL = max(dot(wNormal, sunDir), 0) as unknown as NF;
    const sunCol = (sunU.color as unknown as NV3).mul(float(sunU.intensity)) as unknown as NV3;
    let direct: NF = nDotL;
    if (shadowsOn && world.csm) {
      const sf = (nodeObject(world.csm) as unknown as NV4).x.clamp(0, 1).toVar() as unknown as NF;
      direct = nDotL.mul(sf) as unknown as NF;
    }
    // ENERGY-CORRECT lighting (D-N22, user choice — NOT pixel-parity with the
    // old terrain). Uses three's BRDF energy exactly: BRDF_Lambert = albedo/π
    // on BOTH the direct sun term (irradiance = NdotL·sunColor, sunColor =
    // color·intensity, no π — three src 303/600/624) AND the indirect probe
    // irradiance (the old path's IrradianceNode → context.irradiance → ×albedo/π
    // — line 713). The probe field is the sole sky-diffuse ambient here
    // (occlusion-aware, ray-marches the atmosphere). DELIBERATE DIVERGENCE from
    // the old terrain, which ALSO adds a full env-IBL skylight term
    // (scene.environment, intensity 1.0) on top of the probe — making it
    // brighter; we do NOT replicate that (it double-counts the unoccluded sky).
    // So nanite terrain is dimmer than old by design; parity was abandoned.
    // Accumulate radiance, divide once by π.
    let radiance: NV3 = sunCol.mul(direct) as unknown as NV3;
    if (world.gi) {
      let irr = world.gi.irradiance(wp, shading.worldNormalNode) as unknown as NV3;
      if (world.canopyTex) {
        irr = irr.mul(canopyAt(world.canopyTex, wp.xz).mul(0.18).oneMinus()) as unknown as NV3;
      }
      radiance = radiance.add(irr) as unknown as NV3;
    }
    let lit: NV3 = albedo.mul(radiance).mul(float(1 / Math.PI)) as unknown as NV3;

    // ---- debug overrides ------------------------------------------------------
    if (nandbg === 'flat') return vec4(albedo, 1);
    if (nandbg === 'albedo') return vec4(albedo, 1);
    if (nandbg === 'normal') return vec4(wNormal.mul(0.5).add(0.5), 1);
    if (nandbg === 'cov') return vec4(1, 0, 0, 1); // every covered pixel red
    return vec4(lit, 1);
  })() as unknown as typeof mat.fragmentNode;

  mat.depthNode = Fn(() => {
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    return bcU2F(elemU(vis.depthV.ro, pixelIndex)) as unknown as NF;
  })() as unknown as typeof mat.depthNode;
  mat.depthTest = false;
  mat.depthWrite = nandepth !== '0';
  mat.fog = false;
  mat.lights = false;

  const mesh = new Mesh(geometry, mat);
  mesh.name = 'naniteResolve';
  mesh.frustumCulled = false;
  mesh.renderOrder = -1000;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return { mesh };
}
