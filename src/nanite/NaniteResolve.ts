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
  cross,
  dot,
  float,
  getViewPosition,
  int,
  max,
  mix,
  nodeObject,
  normalize,
  positionGeometry,
  screenCoordinate,
  screenUV,
  smoothstep,
  texture,
  uint,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { NF, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import type { CSMShadowNode } from 'three/addons/csm/CSMShadowNode.js';
import { causticContext, causticDepth, causticTint } from '../render/Caustics';
import { buildTerrainShading } from '../render/TerrainMaterial';
import { sunU } from '../render/VegMaterials';
import { canopyAt } from '../gpu/passes/Scatter';
import { BARK_RES } from '../gpu/passes/BarkSynth';
import { fbm3, valueNoise3 } from '../gpu/noise/NoiseTSL';
import type { ProbeGI } from '../gpu/passes/ProbeGI';
import type { Heightfield } from '../world/Heightfield';
import { MESH_WORDS, readVertex } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import { makeFetch } from './NaniteFetch';
import { instRotateDir, type NaniteCam } from './NaniteCommon';
import type { NaniteVisBuffers } from './NaniteRaster';
import { bcU2F, elemU, toF } from './Tsl';
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
  /** bark/deadwood texture-array (texA albedo+cavity, texB normal+rough+height);
   *  sampled at the per-mesh layer slice (mesh word 7). null = bark unported. */
  barkTexA: Texture | null;
  barkTexB: Texture | null;
}

/** TextureNode sample-config chain (depth = array slice, grad = explicit deriv) */
interface TexSample {
  depth(d: unknown): TexSample;
  grad(a: unknown, b: unknown): TexSample;
}

/** hue jitter (port of VegMaterials.hueShift): warm/cool tint by vdata.x */
function hueShift(base: NV3, hue: NF, amount: number): NV3 {
  const k = hue.mul(amount);
  const warm = vec3(1.18, 1.0, 0.55);
  const cool = vec3(0.7, 0.95, 1.25);
  return base
    .mul(warm)
    .mul(k.clamp(0, 1))
    .add(base.mul(cool).mul(k.negate().clamp(0, 1)))
    .add(base.mul(float(1).sub(k.abs()))) as unknown as NV3;
}

/** 3D barycentric of p inside triangle (a,b,c) — perspective-correct because
 *  p is the real reconstructed world point on the rasterized surface (not a
 *  screen-interpolated value). Ericson's method. */
function baryWeights(p: NV3, a: NV3, b: NV3, c: NV3): NV3 {
  const v0 = b.sub(a);
  const v1 = c.sub(a);
  const v2 = p.sub(a);
  const d00 = dot(v0, v0);
  const d01 = dot(v0, v1);
  const d11 = dot(v1, v1);
  const d20 = dot(v2, v0);
  const d21 = dot(v2, v1);
  const denom = d00.mul(d11).sub(d01.mul(d01)).max(float(1e-12)) as unknown as NF;
  const v = d11.mul(d20).sub(d01.mul(d21)).div(denom) as unknown as NF;
  const w = d00.mul(d21).sub(d01.mul(d20)).div(denom) as unknown as NF;
  const u = float(1).sub(v).sub(w);
  return vec3(u, v, w) as unknown as NV3;
}

/** registry vdata word (4×u8 unorm, WorldRegistry.geometryToSource) → vec4 */
function unpackVdata(packed: NU): NV4 {
  return vec4(
    toF(packed.bitAnd(uint(0xff))),
    toF(packed.shiftRight(uint(8)).bitAnd(uint(0xff))),
    toF(packed.shiftRight(uint(16)).bitAnd(uint(0xff))),
    toF(packed.shiftRight(uint(24)).bitAnd(uint(0xff))),
  ).div(255) as unknown as NV4;
}

/** ROCK material (port of VegMaterials.rockMaterial, defaults) — strata banding
 *  from vdata.y, lichen via vdata.z, AO/moss via vdata.w, on world-space noise.
 *  Returns albedo (incl. the colorNode AO darkening) + ao (the aoNode, applied
 *  to indirect only by the caller). Roughness omitted — terrain is matte-lit. */
function rockShade(d: NV4, wp: NV3, nrm: NV3): { albedo: NV3; ao: NF } {
  const strataT = d.y;
  const upness = nrm.y.max(0);
  const bandTint = valueNoise3(vec3(float(0), strataT.mul(7.3), float(0)).add(wp.mul(0.02)));
  const grain = fbm3(wp.mul(2.1), 3).mul(0.5).add(0.5);
  const tr = 0.285;
  const tg = 0.255;
  const tb = 0.215;
  let albedo = mix(
    vec3(tr * 0.42, tg * 0.44, tb * 0.55),
    vec3(tr, tg, tb),
    bandTint.mul(0.55).add(grain.mul(0.45)).clamp(0, 1),
  ) as unknown as NV3;
  const lich = smoothstep(0.62, 0.78, valueNoise3(wp.mul(3.7))).mul(d.z.mul(0.7).add(0.3));
  albedo = mix(albedo, vec3(0.16, 0.175, 0.14), lich.mul(0.55)) as unknown as NV3;
  albedo = mix(albedo, vec3(0.17, 0.15, 0.12), upness.pow(2).mul(0.3)) as unknown as NV3;
  const steep = float(1).sub(upness);
  const streakN = valueNoise3(vec3(wp.x.mul(2.6), wp.y.mul(0.22), wp.z.mul(2.6)));
  const streak = smoothstep(0.55, 0.82, streakN).mul(smoothstep(0.45, 0.8, steep)).mul(0.55);
  albedo = mix(albedo, albedo.mul(vec3(0.5, 0.46, 0.4)), streak) as unknown as NV3;
  // moss (default amount 0.25 → ×0.5 gate)
  const mossN = smoothstep(0.45, 0.75, fbm3(wp.mul(1.7), 3).mul(0.5).add(0.5));
  const moss = smoothstep(0.45, 0.85, upness).mul(mossN).mul(d.w).mul(0.5).clamp(0, 1);
  albedo = mix(albedo, vec3(0.045, 0.085, 0.03), moss) as unknown as NV3;
  albedo = albedo.mul(d.w.mul(0.35).add(0.65)) as unknown as NV3; // colorNode AO darkening
  return { albedo, ao: d.w as unknown as NF };
}

export function buildNaniteResolve(
  gpu: RegistryGpu,
  heightTex: Texture,
  cam: NaniteCam,
  cull: { qRasterRO: BufOf<UV2> },
  vis: NaniteVisBuffers,
  world: ResolveWorld,
): NaniteResolveHandles {
  const hf = world.hf;
  if (!hf.biomeTex || !hf.fieldsTex || !hf.noiseA || !hf.noiseB) {
    throw new Error('NaniteResolve: heightfield derived maps missing (boot order)');
  }
  const q = new URLSearchParams(window.location.search);
  // ROCK (and future explicit-mesh classes) need per-vertex attributes →
  // re-fetch the cluster triangle. Terrain reconstructs wp from depth and
  // never touches this. ?nanwind=0 A/Bs the trunk wind — MUST match the raster's
  // makeFetch (both read this flag) so their windy positions stay bit-identical.
  const windOn = q.get('nanwind') !== '0';
  const fetch = makeFetch(gpu, heightTex, undefined, windOn ? { camPos: cam.camPos } : undefined);
  const nandepth = q.get('nandepth');
  const nandbg = q.get('nandbg');
  // ?nanbark= bisect: const (flat brown) | lN (force mip N) | grad (anisotropic
  // ray-plane derivatives — known NaN on near trunks, default is analytic LOD)
  const nanbark = q.get('nanbark');
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

    // ---- ROCK shading (N4-C2): re-fetch the cluster triangle, barycentric-
    // interpolate vdata + normal at the reconstructed surface point, run the
    // ported rockMaterial. Gated on isR so terrain (heightfield clusters, no
    // explicit verts) never enters the explicit-mesh fetch.
    const isR = matClass.equal(uint(1)).toVar();
    const rockCol = vec3(0.3).toVar() as unknown as NV3;
    const rockNrm = vec3(0, 1, 0).toVar() as unknown as NV3;
    const rockAo = float(1).toVar() as unknown as NF;
    If(isR, () => {
      const instId = item.x;
      const localTri = pRaw.bitAnd(uint(127));
      const ctx = fetch.makeCtx(instId, ci);
      const w0 = fetch.fetchWorldVert(ctx, localTri, 0);
      const w1 = fetch.fetchWorldVert(ctx, localTri, 1);
      const w2 = fetch.fetchWorldVert(ctx, localTri, 2);
      const bw = baryWeights(wp, w0, w1, w2);
      const tb = ctx.triStart.add(localTri).mul(uint(3));
      const a = readVertex(gpu.verts, elemU(gpu.indices, tb));
      const b = readVertex(gpu.verts, elemU(gpu.indices, tb.add(uint(1))));
      const c = readVertex(gpu.verts, elemU(gpu.indices, tb.add(uint(2))));
      const dv = unpackVdata(a.vdata)
        .mul(bw.x)
        .add(unpackVdata(b.vdata).mul(bw.y))
        .add(unpackVdata(c.vdata).mul(bw.z)) as unknown as NV4;
      const nrm = normalize(
        instRotateDir(ctx.yawSc, a.nrm)
          .mul(bw.x)
          .add(instRotateDir(ctx.yawSc, b.nrm).mul(bw.y))
          .add(instRotateDir(ctx.yawSc, c.nrm).mul(bw.z)),
      ) as unknown as NV3;
      const rk = rockShade(dv, wp, nrm);
      rockCol.assign(rk.albedo);
      rockNrm.assign(nrm);
      rockAo.assign(rk.ao);
    });

    // ---- BARK + DEADWOOD shading (N4-C3): textured trunks/snags. Same
    // explicit-mesh fetch as rock, plus per-vertex UV interpolation, a tangent
    // frame from the triangle edges for the normal map, analytic UV gradients
    // (neighbour-pixel rays intersected with THIS triangle's plane → no
    // silhouette mip spike), and the bark texture-ARRAY sampled at the per-mesh
    // layer slice (mesh word 7). Diffuse-only (roughness unused, like terrain/
    // rock). Trunk WIND rides in via fetchWorldVert at the C3 second commit.
    const isB = matClass.equal(uint(2)).toVar();
    const isD = matClass.equal(uint(3)).toVar();
    const isBD = isB.or(isD).toVar();
    const barkCol = vec3(0.3).toVar() as unknown as NV3;
    const barkNrm = vec3(0, 1, 0).toVar() as unknown as NV3;
    const barkAo = float(1).toVar() as unknown as NF;
    if (world.barkTexA && world.barkTexB) {
      const barkTexA = world.barkTexA;
      const barkTexB = world.barkTexB;
      If(isBD, () => {
        const instId = item.x;
        const localTri = pRaw.bitAnd(uint(127));
        const ctx = fetch.makeCtx(instId, ci);
        const w0 = fetch.fetchWorldVert(ctx, localTri, 0);
        const w1 = fetch.fetchWorldVert(ctx, localTri, 1);
        const w2 = fetch.fetchWorldVert(ctx, localTri, 2);
        const bw = baryWeights(wp, w0, w1, w2);
        const tb = ctx.triStart.add(localTri).mul(uint(3));
        const va = readVertex(gpu.verts, elemU(gpu.indices, tb));
        const vb = readVertex(gpu.verts, elemU(gpu.indices, tb.add(uint(1))));
        const vc = readVertex(gpu.verts, elemU(gpu.indices, tb.add(uint(2))));
        const uvv = va.uv
          .mul(bw.x)
          .add(vb.uv.mul(bw.y))
          .add(vc.uv.mul(bw.z)) as unknown as NV2;
        const dv = unpackVdata(va.vdata)
          .mul(bw.x)
          .add(unpackVdata(vb.vdata).mul(bw.y))
          .add(unpackVdata(vc.vdata).mul(bw.z)) as unknown as NV4;
        const gnrm = normalize(
          instRotateDir(ctx.yawSc, va.nrm)
            .mul(bw.x)
            .add(instRotateDir(ctx.yawSc, vb.nrm).mul(bw.y))
            .add(instRotateDir(ctx.yawSc, vc.nrm).mul(bw.z)),
        ) as unknown as NV3;

        // tangent frame (world-space): T along +U, Gram-Schmidt vs the normal,
        // Bi = N×T. Edge/uv-delta solve (Lengyel) — bark texB.xy perturbs it.
        const e1 = w1.sub(w0);
        const e2 = w2.sub(w0);
        const du1 = vb.uv.x.sub(va.uv.x);
        const dq1 = vb.uv.y.sub(va.uv.y);
        const du2 = vc.uv.x.sub(va.uv.x);
        const dq2 = vc.uv.y.sub(va.uv.y);
        const r = float(1).div(du1.mul(dq2).sub(du2.mul(dq1)).add(1e-8));
        const Traw = e1.mul(dq2).sub(e2.mul(dq1)).mul(r) as unknown as NV3;
        const Braw = e2.mul(du1).sub(e1.mul(du2)).mul(r) as unknown as NV3;
        const T = normalize(Traw.sub(gnrm.mul(dot(gnrm, Traw)))) as unknown as NV3;
        const Bi = normalize(cross(gnrm, T)) as unknown as NV3;

        // analytic mip LOD (NaN-proof, isotropic): world size of one screen
        // pixel at the surface vs world size of one bark texel. |Traw|/|Braw| =
        // world metres per uv unit; bark tiles once per uv unit (BARK_RES texels/
        // unit). Conservative axis (min world-per-texel) to anti-alias. The
        // hardware auto-mip is unusable here (uv is computed in non-uniform
        // control flow → undefined derivatives); anisotropic .grad() is future
        // work (?nanbark=grad — the ray-plane neighbour path NaNs on near trunks).
        const C = vec3(cam.camPos) as unknown as NV3;
        const dist = wp.sub(C).length();
        const pixWorld = dist.mul(2).div(float(cam.cotHalfFov).mul(float(cam.uH)));
        const wPerTexel = Traw.length().min(Braw.length()).div(BARK_RES).max(1e-6);
        const lod = pixWorld.div(wPerTexel).max(1e-4).log2().max(0);
        const planeN = normalize(cross(e1, e2)) as unknown as NV3;
        const rayDir = (suv: NV2): NV3 => {
          const vpN = getViewPosition(suv, zDev, cameraProjectionMatrixInverse) as unknown as NV3;
          const wd = (
            (cameraWorldMatrix as unknown as { mul(v: NV4): NV4 }).mul(
              (vec4 as unknown as (a: NV3, b: number) => NV4)(vpN, 0),
            ) as unknown as NV4
          ).xyz as unknown as NV3;
          return normalize(wd) as unknown as NV3;
        };
        const uvAt = (dir: NV3): NV2 => {
          const tt = dot(planeN, w0.sub(C)).div(dot(planeN, dir).add(1e-8));
          const bh = baryWeights(C.add(dir.mul(tt)) as unknown as NV3, w0, w1, w2);
          return va.uv.mul(bh.x).add(vb.uv.mul(bh.y)).add(vc.uv.mul(bh.z)) as unknown as NV2;
        };

        const layer = int(fetch.meshWord(ctx.meshId, 7).bitAnd(uint(0xff)));
        // ?nanbark=const — flat brown, no texture (fetch/branch sanity)
        if (nanbark === 'const') {
          barkCol.assign(vec3(0.4, 0.25, 0.13) as unknown as NV3);
          barkNrm.assign(gnrm);
          barkAo.assign(float(1) as unknown as NF);
          return;
        }
        // ?nanbark=lN — force mip level N (inspect the generated chain)
        const lvlMatch = nanbark ? /^l(\d+)$/.exec(nanbark) : null;
        const sample = (t: Texture): NV4 => {
          const base = texture(t, uvv as never) as unknown as TexSample;
          if (lvlMatch)
            return (base.depth(layer) as unknown as { level(n: number): NV4 }).level(
              Number(lvlMatch[1]),
            );
          if (nanbark === 'grad') {
            const dUVdx = uvAt(
              rayDir(screenUV.add(vec2(float(1).div(cam.uW), 0)) as unknown as NV2),
            ).sub(uvv) as unknown as NV2;
            const dUVdy = uvAt(
              rayDir(screenUV.add(vec2(0, float(1).div(cam.uH))) as unknown as NV2),
            ).sub(uvv) as unknown as NV2;
            return base.depth(layer).grad(dUVdx, dUVdy) as unknown as NV4;
          }
          // DEFAULT: analytic isotropic mip LOD (NaN-proof)
          return (base.depth(layer) as unknown as { level(n: unknown): NV4 }).level(lod);
        };
        const tA = sample(barkTexA);
        const tB = sample(barkTexB);

        // albedo: sqrt-decoded texture, bark (hue+cavity) vs deadwood (dim+moss+rot)
        const tex = tA.rgb.mul(tA.rgb) as unknown as NV3;
        const barkAlb = hueShift(tex, dv.x, 0.14).mul(dv.w.mul(0.45).add(0.55)) as unknown as NV3;
        // deadwood dim (logDim, representative — energy-correct, not per-pool)
        let deadAlb = tex.mul(vec3(0.6, 0.52, 0.44)) as unknown as NV3;
        const mossN = smoothstep(0.24, 0.58, fbm3(wp.mul(2.6), 3).mul(0.5).add(0.5));
        const moss = smoothstep(0.05, 0.65, gnrm.y).mul(dv.z).mul(mossN).clamp(0, 1);
        deadAlb = mix(deadAlb, vec3(0.05, 0.1, 0.032), moss) as unknown as NV3;
        deadAlb = deadAlb.mul(float(1).sub(dv.z.mul(0.25))) as unknown as NV3; // rot
        deadAlb = hueShift(deadAlb, dv.x, 0.1) as unknown as NV3;

        // tangent-space normal map (three normalMap: n = tex·2−1, z kept = 1)
        const pert = normalize(
          T.mul(tB.x.mul(2).sub(1))
            .add(Bi.mul(tB.y.mul(2).sub(1)))
            .add(gnrm),
        ) as unknown as NV3;

        barkCol.assign(isD.select(deadAlb, barkAlb));
        barkNrm.assign(pert);
        barkAo.assign(tA.w as unknown as NF);
      });
    }

    // unported explicit classes (leaf/grass/debris — N9/N10) keep a flat gray
    const palette = vec3(0.35, 0.33, 0.3) as unknown as NV3;
    const albedo = isT
      .select(terrainCol, isR.select(rockCol, isBD.select(barkCol, palette)))
      .toVar() as unknown as NV3;
    const wNormal = isT
      .select(shading.worldNormalNode, isR.select(rockNrm, isBD.select(barkNrm, vec3(0, 1, 0))))
      .toVar() as unknown as NV3;
    // aoNode (rock + bark cavity): applied to indirect only — 1 elsewhere
    const ao = isR.select(rockAo, isBD.select(barkAo, float(1))) as unknown as NF;

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
      // F9: read ground height from heightTex (TEXTURE — plentiful) so the
      // resolve does not bind the height STORAGE buffer (10-buffer/stage cap)
      const groundY = (
        texture(hf.heightTex, hf.uvFromWorld(wp.xz)) as unknown as NV4
      ).x as unknown as NF;
      let irr = world.gi.irradiance(wp, wNormal, 2.0, groundY) as unknown as NV3;
      if (world.canopyTex) {
        irr = irr.mul(canopyAt(world.canopyTex, wp.xz).mul(0.18).oneMinus()) as unknown as NV3;
      }
      radiance = radiance.add(irr.mul(ao)) as unknown as NV3;
    }
    let lit: NV3 = albedo.mul(radiance).mul(float(1 / Math.PI)) as unknown as NV3;

    // ---- debug overrides ------------------------------------------------------
    if (nandbg === 'flat') return vec4(albedo, 1);
    if (nandbg === 'albedo') return vec4(albedo, 1);
    if (nandbg === 'normal') return vec4(wNormal.mul(0.5).add(0.5), 1);
    if (nandbg === 'cov') return vec4(1, 0, 0, 1); // every covered pixel red
    if (nandbg === 'cls')
      // matClass tint: terrain green / rock red / bark blue / deadwood cyan /
      // other (leaf/grass/debris) magenta
      return vec4(
        isT.select(
          vec3(0.1, 0.6, 0.1),
          isR.select(
            vec3(0.95, 0.1, 0.1),
            isB.select(
              vec3(0.1, 0.1, 0.95),
              isD.select(vec3(0.1, 0.7, 0.8), vec3(0.8, 0.1, 0.8)),
            ),
          ),
        ),
        1,
      ) as unknown as NV4;
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
