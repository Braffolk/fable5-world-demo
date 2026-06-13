/**
 * Material resolve (N4) — the fullscreen-triangle mesh that turns the vis
 * buffer into shaded pixels INSIDE the main scene pass (D-N18): renders first
 * (renderOrder −1000), writes the raster's REAL f32 depth via depthNode, and
 * mask-discards uncovered pixels so the cleared depth survives for sky.
 *
 * C1: the resolve is a **MeshPhysicalNodeMaterial riding three's standard
 * lighting pipeline** — the SAME code path that lights the old forward
 * materials (sun + CSM + PCSS, hemisphere, IBL environment, lightmap-slot
 * probe GI) — with the per-fragment surface reconstructed from the vis
 * buffer instead of vertex varyings:
 *  - world position: unproject (pixel-center ndc, stored z) through invVp —
 *    exact for the rastered surface, no barycentrics needed for terrain
 *    (every terrain shading input derives from wp + world-space textures);
 *  - `receivedShadowPositionNode` = wp feeds the shadow system the true
 *    position (CSM cascade-select reads it via the CsmCached override —
 *    `positionView` is a near-plane point under a fullscreen vertexNode,
 *    D-N17/D-N18);
 *  - view direction (specular/env) comes from three's clip-space
 *    reconstruction — direction-correct for a fullscreen triangle;
 *  - TERRAIN shades through the SAME buildTerrainShading graph as the tiles
 *    (parameterized `surf`), plus the tiles' capillary-wet/caustic block.
 *
 * Classes without a landed port (rock/bark/deadwood until C2/C3) fall back
 * to flat palette albedo with an up normal — they are not in the default
 * migration set (D-N19) and only appear under ?naniteclasses overrides.
 *
 * Bisects: ?nandepth=0|half (depth write off / constant), ?nandbg=dist
 * (paint fetched depth as distance/2000 in R).
 */

import { Mesh, Sphere, Vector3 } from 'three';
import { BufferGeometry, Float32BufferAttribute } from 'three';
import type { Texture } from 'three';
import { IrradianceNode, MeshPhysicalNodeMaterial, type StorageTexture } from 'three/webgpu';
import {
  Fn,
  If as IfTsl,
  cameraProjectionMatrixInverse,
  cameraViewMatrix,
  cameraWorldMatrix,
  float,
  getViewPosition,
  screenCoordinate,
  screenUV,
  uint,
  vec3,
  vec4,
} from 'three/tsl';
import type { NB, NF, NU, NV3, NV4 } from '../gpu/TSLTypes';
import { causticContext, causticDepth, causticTint } from '../render/Caustics';
import { buildTerrainShading } from '../render/TerrainMaterial';
import { canopyAt } from '../gpu/passes/Scatter';
import type { ProbeGI } from '../gpu/passes/ProbeGI';
import type { Heightfield } from '../world/Heightfield';
import { MESH_WORDS } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import type { NaniteCam } from './NaniteCommon';
import { makeFetch } from './NaniteFetch';
import type { NaniteVisBuffers } from './NaniteRaster';
import { bcU2F, elemU } from './Tsl';
import type { BufOf, UV2 } from './Tsl';
import { mix, smoothstep } from 'three/tsl';

export interface NaniteResolveHandles {
  /** add to engine.scene; renderOrder −1000, castShadow off */
  mesh: Mesh;
  /** per-frame: glue the triangle to the camera (matrixWorld) and refresh
   *  the near-plane corner attribute when fov/aspect/near change */
  syncCamera(camera: {
    matrixWorld: { elements: number[] };
    fov: number;
    aspect: number;
    near: number;
  }): void;
}

export interface ResolveWorld {
  hf: Heightfield;
  gi: ProbeGI | null;
  canopyTex: StorageTexture | null;
}

export function buildNaniteResolve(
  gpu: RegistryGpu,
  heightTex: Texture,
  cam: NaniteCam,
  cull: { qRasterRO: BufOf<UV2> },
  vis: NaniteVisBuffers,
  world: ResolveWorld,
  /** ?nanprobe=1 — fragment writes wp at the center pixel into slots 28..30
   *  of the probe buffer (exact material-side readback, no image transforms) */
  dbg?: { buf: { element(i: NU | number): { assign(v: unknown): void } } },
): NaniteResolveHandles {
  const hf = world.hf;
  if (!hf.biomeTex || !hf.fieldsTex || !hf.noiseA || !hf.noiseB) {
    throw new Error('NaniteResolve: heightfield derived maps missing (boot order)');
  }
  const nandepth = new URLSearchParams(window.location.search).get('nandepth');
  const nandbg = new URLSearchParams(window.location.search).get('nandbg');

  // CAMERA-GLUED NEAR-PLANE TRIANGLE (no vertexNode): the mesh's matrixWorld
  // tracks the camera and the three vertices sit at view-space near-plane
  // corners spanning ndc (−1,−1)(3,−1)(−1,3) — the DEFAULT MVP renders it
  // exactly fullscreen, and every standard varying carries the geometrically
  // right thing per pixel: positionView = the near-plane point ALONG THE RAY
  // (so positionViewDirection = the exact view ray → GGX specular + IBL env
  // sample the same directions the old forward materials see), positionWorld
  // = that point in world. A vertexNode fullscreen triangle instead leaves
  // positionViewDirection as an interpolated varying of garbage local
  // positions (Position.js builds it in the VERTEX stage — the fragment
  // clip-reconstruction branch never fires for it): the terrain lost its
  // whole glancing-sun specular sheen (39% unshadowed diff at bm4).
  const geometry = new BufferGeometry();
  const cornerArr = new Float32Array(9);
  geometry.setAttribute('position', new Float32BufferAttribute(cornerArr, 3));
  // +Y normals: the shadow system biases its sample by GEOMETRY normalWorld ·
  // normalBias (ShadowNode.js:512 — not the material normalNode). The old
  // terrain tiles' geometry normals are flat +Y (rotated PlaneGeometry), so
  // a +Y quad normal reproduces their exact shadow sample position
  // wp + (0,1,0)·normalBias. Also silences the missing-attribute warning.
  geometry.setAttribute(
    'normal',
    new Float32BufferAttribute(new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0]), 3),
  );
  geometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);
  let lastFov = 0;
  let lastAspect = 0;
  let lastNear = 0;

  // ---- shared per-fragment reconstruction (one set of node instances — every
  // material slot referencing them emits the expressions once) ---------------
  const fy = float(cam.uH).sub(screenCoordinate.y);
  const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
  const dRaw = elemU(vis.depthV.ro, pixelIndex).toVar();
  const pRaw = elemU(vis.payloadV.ro, pixelIndex).toVar();
  const covered = dRaw
    .notEqual(uint(0xffffffff))
    .and(pRaw.notEqual(uint(0xffffffff))) as unknown as NB;
  const zDev = bcU2F(dRaw);
  // world position via three's OWN in-material reconstruction (the post
  // chain's exact pattern, THREE-NOTES: getViewPosition flips v internally):
  // standard camera nodes — no custom uniforms, no stage ambiguity. The
  // hand-rolled invVp·ndc unprojection was kernel-exact but evaluated wrong
  // in material slot graphs.
  const wpv = getViewPosition(screenUV, zDev, cameraProjectionMatrixInverse) as unknown as NV3;
  const wp = (
    (cameraWorldMatrix as unknown as { mul(v: NV4): NV4 }).mul(
      (vec4 as unknown as (a: NV3, b: number) => NV4)(wpv, 1),
    ) as unknown as NV4
  ).xyz as unknown as NV3;

  // payload → mesh → matClass (cluster word 7 high half = meshId; mesh word 6
  // bits 8–15 = matClass)
  const itemIdx = pRaw.shiftRight(uint(7));
  const localTri = pRaw.bitAnd(uint(127));
  const item = cull.qRasterRO.element(itemIdx.add(uint(1)));
  const instId = item.x;
  const ci = item.y;
  const meshId = elemU(gpu.clusters, ci.mul(uint(8)).add(uint(7))).shiftRight(uint(16));
  const matClass = elemU(gpu.meshes, meshId.mul(uint(MESH_WORDS)).add(uint(6)))
    .shiftRight(uint(8))
    .bitAnd(uint(0xff))
    .toVar();

  // ---- TERRAIN shading: the tiles' graph, parameterized on wp --------------
  const camPos = vec3(cam.camPos) as unknown as NV3;
  const shading = buildTerrainShading({
    normalTex: hf.normalTex,
    biomeTex: hf.biomeTex,
    fieldsTex: hf.fieldsTex,
    noiseA: hf.noiseA,
    noiseB: hf.noiseB,
    mp: hf.mp,
    far: false,
    surf: { wp, camPos },
  });
  // capillary-wet band + caustics on submerged beds (TerrainTiles port,
  // parameterized on wp)
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

  // ---- un-ported classes: flat palette + up normal (D-N19: not in the
  // default migration set; visible only under ?naniteclasses overrides) ------
  const isT = matClass.equal(uint(0));
  const isR = matClass.equal(uint(1));
  const isB = matClass.equal(uint(2));
  const isD = matClass.equal(uint(3));
  const palette = isR
    .select(
      vec3(0.42, 0.41, 0.4),
      isB.select(vec3(0.36, 0.27, 0.19), isD.select(vec3(0.33, 0.28, 0.22), vec3(0.35, 0.33, 0.3))),
    )
    .toVar() as unknown as NV3;
  // view-rotate WORLD normals explicitly: the camera-glued mesh has
  // modelView = identity, so transformNormalToView (modelNormalViewMatrix)
  // would pass world normals through as view-space — fed the lighting
  // world-space normals and turned the meadow into an ice mirror
  const viewRot = (n: NV3): NV3 =>
    (cameraViewMatrix as unknown as { mul(v: NV4): NV4 })
      .mul((vec4 as unknown as (a: NV3, b: number) => NV4)(n, 0))
      .xyz.normalize() as unknown as NV3;
  const terrainNView = viewRot(shading.worldNormalNode);
  const upView = viewRot(vec3(0, 1, 0) as unknown as NV3);

  // ---- material — three's standard lit pipeline ------------------------------
  const mat = new MeshPhysicalNodeMaterial();
  // ?nandbg=nospec — kill the specular term (glare bisect)
  mat.specularIntensity = nandbg === 'nospec' ? 0 : 0.35; // terrain; per-class node at C2+
  mat.metalness = 0;
  mat.maskNode = covered as unknown as typeof mat.maskNode;
  if (nandbg === 'wperr') {
    // |wp − payload corner| — wp validity vs the owning triangle (≤~2 m ok);
    // If()/toVar in material graphs need an Fn stack (N0 gotcha)
    mat.lights = false;
    mat.colorNode = Fn(() => {
      const fetch = makeFetch(gpu, heightTex);
      const ctx = fetch.makeCtx(instId as unknown as NU, ci as unknown as NU);
      const w0 = fetch.fetchWorldVert(ctx, localTri as unknown as NU, 0);
      const err = wp.sub(w0).length();
      return vec4(
        err.div(10).clamp(0, 1),
        err.div(100).clamp(0, 1),
        err.div(1000).clamp(0, 1),
        1,
      );
    })() as unknown as typeof mat.colorNode;
  } else if (nandbg === 'wp') {
    // wp forensics: R=fract(x/100) G=fract(z/100) B=y/500
    mat.lights = false;
    mat.colorNode = vec4(
      wp.x.add(2048).div(4096).clamp(0, 1),
      wp.z.add(2048).div(4096).clamp(0, 1),
      wp.y.div(500).clamp(0, 1),
      1,
    ) as unknown as typeof mat.colorNode;
  } else if (nandbg === 'albedo') {
    // raw albedo, unlit (splat bisect)
    mat.lights = false;
    mat.colorNode = isT.select(terrainCol, palette) as unknown as typeof mat.colorNode;
  } else if (nandbg === 'dist') {
    // numeric forensics: fetched depth as view-distance/2000 in R (raw out)
    const dist = float(0.3).div(float(1).sub(zDev.mul(float(29999.7).div(30000))));
    mat.colorNode = vec4(dist.div(2000).clamp(0, 1), 0, 0, 1) as unknown as typeof mat.colorNode;
  } else {
    const baseCol = isT.select(terrainCol, palette);
    mat.colorNode = (dbg
      ? Fn(() => {
          const isCenter = uint(screenCoordinate.x)
            .equal(uint(Math.floor(0)))
            .and(uint(screenCoordinate.y).equal(uint(0)));
          void isCenter;
          // write wp at the exact center pixel (cheap compare per fragment)
          const cx = uint(float(cam.uW).div(2));
          const cy = uint(float(cam.uH).div(2));
          IfTsl(
            uint(screenCoordinate.x).equal(cx).and(uint(screenCoordinate.y).equal(cy)),
            () => {
              dbg.buf.element(28).assign(wp.x);
              dbg.buf.element(29).assign(wp.y);
              dbg.buf.element(30).assign(wp.z);
            },
          );
          return baseCol;
        })()
      : baseCol) as unknown as typeof mat.colorNode;
  }
  mat.normalNode = isT.select(terrainNView, upView) as unknown as typeof mat.normalNode;
  mat.roughnessNode = isT.select(terrainRough, float(0.9)) as unknown as typeof mat.roughnessNode;
  mat.metalnessNode = float(0) as unknown as typeof mat.metalnessNode;
  // the TRUE per-pixel world position for the shadow system (cascade select +
  // shadow coords) — positionWorld/positionView are fullscreen-triangle
  // varyings here (D-N17); ShadowBaseNode assigns this into shadowPositionWorld
  (mat as unknown as { receivedShadowPositionNode: unknown }).receivedShadowPositionNode = wp;
  // probe-GI through the lightmap slot — the same injection the tiles use
  // (canopy residual 0.18, lift default 2 m)
  if (world.gi) {
    const gi = world.gi;
    let irr = gi.irradiance(wp, shading.worldNormalNode);
    if (world.canopyTex) {
      irr = irr.mul(canopyAt(world.canopyTex, wp.xz).mul(0.18).oneMinus()) as typeof irr;
    }
    (mat as unknown as { setupLightMap: () => unknown }).setupLightMap = () =>
      new IrradianceNode(irr as unknown as ConstructorParameters<typeof IrradianceNode>[0]);
  }
  mat.depthNode = Fn(() => {
    // ?nandepth=half — frag_depth liveness probe (erases everything > ~0.6 m)
    if (nandepth === 'half') return float(0.5);
    return bcU2F(dRaw) as unknown as NF;
  })() as unknown as typeof mat.depthNode;
  // always-pass + write (backend: depthTest=false → compare 'always');
  // cleared depth survives only where the mask discards
  mat.depthTest = false;
  mat.depthWrite = nandepth !== '0';

  const mesh = new Mesh(geometry, mat);
  mesh.name = 'naniteResolve';
  mesh.frustumCulled = false;
  mesh.renderOrder = -1000;
  mesh.castShadow = false;
  mesh.receiveShadow = true;
  mesh.matrixAutoUpdate = false;

  const syncCamera: NaniteResolveHandles['syncCamera'] = (camera) => {
    mesh.matrix.fromArray(camera.matrixWorld.elements);
    mesh.matrixWorld.fromArray(camera.matrixWorld.elements);
    if (camera.fov !== lastFov || camera.aspect !== lastAspect || camera.near !== lastNear) {
      lastFov = camera.fov;
      lastAspect = camera.aspect;
      lastNear = camera.near;
      // view-space near-plane corners for ndc (−1,−1), (3,−1), (−1,3); a hair
      // beyond the near plane so the corners never clip
      const zn = camera.near * 1.0001;
      const ty = Math.tan(((camera.fov * Math.PI) / 180) / 2) * zn;
      const tx = ty * camera.aspect;
      const c = [-tx, -ty, 3 * tx, -ty, -tx, 3 * ty];
      for (let i = 0; i < 3; i++) {
        cornerArr[i * 3] = c[i * 2] as number;
        cornerArr[i * 3 + 1] = c[i * 2 + 1] as number;
        cornerArr[i * 3 + 2] = -zn;
      }
      const attr = geometry.getAttribute('position');
      (attr as unknown as { needsUpdate: boolean }).needsUpdate = true;
    }
  };

  return { mesh, syncCamera };
}

// referenced only for parity with the raster-side fetch (kept import-stable)
export type { NU };
