/**
 * N5 — cluster-driven CSM shadow casters (D-N26).
 *   C0: the per-cascade CULL.
 *   C1: the per-cascade HW vertex-pulling CASTER mesh (this file).
 *
 * One nanite cull chain per CSM cascade, fed the cascade's ORTHO frustum (not
 * the camera frustum): an off-screen / ridge-hidden caster still casts (F5), so
 * shadow casters are culled by the light view, never the camera view. Reuses
 * buildNaniteCull with:
 *   - the cascade ortho frustum planes (refreshed one frame stale from
 *     csm.lights[c].shadow.camera, exactly like Forests.planesCsmU — the
 *     CsmCached lightMargin slack swallows the lag),
 *   - sphereOccluded = null (casters are never HZB-occluded),
 *   - coneCull = false (camera-relative cone backface is wrong for a light view),
 *   - camPos = the MAIN camera position so LOD selection MATCHES the visible
 *     geometry's LOD (a caster at a coarser LOD than its lit surface peter-pans).
 *
 * C1 caster (D-N26): per cascade, a vertex-pulling NodeMaterial Mesh on layer
 * 2+c (castShadow=true, frustumCulled=false) added to engine.scene. three's CSM
 * renders ONLY layer 2+c into cascade c (csm.lights[c].shadow.camera.layers
 * .enable(2+c) below), so each caster fills exactly its own cascade depth map.
 *
 * The integration wall (verified against three 0.184 source): the shadow pass
 * does NOT use a mesh's vertexNode — it swaps in a shared depth override
 * material and reads ONLY material.castShadowPositionNode (Renderer
 * ._getShadowNodes → positionLocal). That node returns LOCAL space, three then
 * applies modelViewProjection = cameraProjectionMatrix · cameraViewMatrix ·
 * modelWorldMatrix. During cascade c's shadow render the active camera IS the
 * cascade ortho light camera, so cameraView/Projection ARE the cascade light VP.
 * We force the caster mesh's matrixWorld to identity ⇒ modelWorldMatrix = I ⇒
 * the world position fetchWorldVert returns lands straight in light clip space —
 * no hand-rolled vp uniform. side = DoubleSide so terrain's single-sided up-faces
 * still cast (the default front→back shadow flip would cull them).
 *
 * The caster's makeFetch is built with the SAME (gpu, heightTex, disp, wind) the
 * CAMERA raster uses (NaniteFrame), so caster geometry is bit-identical to the
 * rendered geometry incl. trunk wind + terrain micro-displacement — shadows
 * attach to their surfaces with no peter-pan / detach.
 *
 * Draw: the cull produces a visible-cluster list (cull.qRasterRO; slot 0 =
 * (count, _), items at 1.. = (instId, ci)). A tiny per-cascade kCasterArgs
 * kernel converts that count into a non-indexed indirect DRAW (vertexCount =
 * count · MAX_CLUSTER_TRIS · 3, the same over-draw-then-degenerate stride the SW
 * raster dispatches). The caster's castShadowPositionNode decodes
 * vertexIndex → (cluster, localTri, corner) → fetchWorldVert; padding triangles
 * (localTri ≥ triCount) collapse all three corners to vec3(0) ⇒ zero-area.
 *
 * NOTE (C1 scope): the cull still runs every frame for all cascades (C0). three
 * only actually RENDERS a cascade — hence draws its caster — on that cascade's
 * CsmCached refresh tick (frozen cascades reuse the cached map), so the heavy
 * caster draw is already cadence-amortised; gating the CULL itself to the
 * refresh tick (and dropping the now-unused reject buffers) is C3.
 */

import {
  BufferGeometry,
  DoubleSide,
  Float32BufferAttribute,
  Frustum,
  Matrix4,
  Mesh,
  Sphere,
  Vector3,
} from 'three';
import type { PerspectiveCamera, Texture } from 'three';
import {
  IndirectStorageBufferAttribute,
  NodeMaterial,
  StorageBufferAttribute,
  type Renderer,
} from 'three/webgpu';
import {
  Fn,
  If,
  cameraProjectionMatrix,
  cameraViewMatrix,
  uint,
  vec3,
  vec4,
  vertexIndex,
} from 'three/tsl';
import type { NU, NV3 } from '../gpu/TSLTypes';
import { MAX_CLUSTER_TRIS, type RegistryGpu } from './GeometryRegistry';
import { makeNaniteCam, type NaniteCam } from './NaniteCommon';
import { buildNaniteCull, type NaniteCullChain } from './NaniteCull';
import { makeFetch, type TerrainDisp, type TrunkWindOpt } from './NaniteFetch';
import { dispatch, sU32Views } from './Tsl';

/** CSM default cascade count (csmcasc can lower it — we guard per-cascade) */
export const SHADOW_CASCADES = 4;

/** verts emitted per visible cluster (over-draw; padding tris degenerate) */
const CASTER_VERTS_PER_CLUSTER = MAX_CLUSTER_TRIS * 3;

/** the per-cascade light camera fields we read (ortho cascade camera) */
interface CascadeLightCam {
  projectionMatrix: Matrix4;
  matrixWorldInverse: Matrix4;
  layers: { enable(ch: number): void };
  left?: number;
}

interface CascadeCull {
  cam: NaniteCam;
  cull: NaniteCullChain;
  /** indirect DRAW args (vertexCount, instanceCount, firstVertex, firstInstance) */
  kCasterArgs: unknown;
  /** the layer-2+c caster mesh (added to engine.scene by NaniteFrame) */
  mesh: Mesh;
  /** last-read visible-cluster count (HUD) */
  count: number;
}

export interface NaniteShadow {
  /** refresh per-cascade planes from the CSM lights, run each cascade cull, and
   *  write each caster's indirect draw args. Call BEFORE post.render() (the
   *  cascade cameras hold last frame's fit — one frame stale, hidden in
   *  lightMargin like Forests). No-op until CSM inits. */
  update(renderer: Renderer, csm: object | null, mainCamera: PerspectiveCamera): void;
  /** per-cascade visible-cluster counts (HUD; -1 if a cascade never ran) */
  readCounts(renderer: Renderer): Promise<number[]>;
  cascades: number;
  /** the per-cascade caster meshes — add to engine.scene so three's CSM
   *  shadow render includes them (one per cascade, layer 2+c). */
  casterMeshes: Mesh[];
}

export function buildNaniteShadow(
  gpu: RegistryGpu,
  instanceCount: number,
  heightTex: Texture,
  /** terrain micro-displacement — MUST match the camera raster's makeFetch */
  disp?: TerrainDisp,
  /** trunk wind — MUST match the camera raster's makeFetch */
  wind?: TrunkWindOpt,
): NaniteShadow {
  // ONE shared fetch (closes over gpu/heightTex/disp/wind, NOT the per-cascade
  // qRaster) — identical inputs to the camera raster ⇒ bit-identical geometry.
  const { makeCtx, fetchWorldVert } = makeFetch(gpu, heightTex, disp, wind);

  // ?nancasterdbg=1 — render cascade 0's caster in the MAIN camera pass (bright,
  // depthTest off) so the vertex-pulled geometry is directly visible: isolates
  // "is the world decode correct" from "is the shadow light VP correct".
  const dbgCaster = new URLSearchParams(window.location.search).get('nancasterdbg') === '1';

  // vertex-pulling world-position decode: vertexIndex → (cluster, localTri,
  // corner) → world. Padding tris (localTri ≥ triCount) collapse all three
  // corners to vec3(0) ⇒ zero-area. Returned as a fresh node per call so it can
  // feed both castShadowPositionNode and (dbg) a main-pass vertexNode.
  const worldPos = (cull: NaniteCullChain): NV3 =>
    Fn(() => {
      const triGlobal = vertexIndex.div(3) as unknown as NU;
      const corner = vertexIndex.mod(3) as unknown as NU;
      const itemIdx = triGlobal.shiftRight(uint(7)).toVar(); // / 128
      const localTri = triGlobal.bitAnd(uint(127)).toVar(); // % 128
      const item = cull.qRasterRO.element(itemIdx.add(uint(1)));
      const instId = item.x.toVar();
      const ci = item.y.toVar();
      const ctx = makeCtx(instId, ci);
      const out = vec3(0).toVar();
      If(localTri.lessThan(ctx.triCount), () => {
        const w0 = fetchWorldVert(ctx, localTri, 0);
        const w1 = fetchWorldVert(ctx, localTri, 1);
        const w2 = fetchWorldVert(ctx, localTri, 2);
        const world = corner
          .equal(uint(1))
          .select(w1, corner.equal(uint(2)).select(w2, w0)) as unknown as NV3;
        out.assign(world);
      });
      return out;
    })() as unknown as NV3;

  const cascades: CascadeCull[] = [];
  for (let c = 0; c < SHADOW_CASCADES; c++) {
    // size is irrelevant to the cull (it reads only planes + camPos); the
    // shadow map is 2048² but the cull never touches width/height.
    const cam = makeNaniteCam(2048, 2048);
    const cull = buildNaniteCull(gpu, instanceCount, cam, null, { coneCull: false });

    // indirect DRAW args, filled from the visible-cluster count by kCasterArgs
    const drawAttr = new IndirectStorageBufferAttribute(new Uint32Array(4), 4);
    const drawBuf = sU32Views(drawAttr as unknown as StorageBufferAttribute, 4).rw;
    const kCasterArgs = Fn(() => {
      const n = cull.qRasterRO.element(0).x.toVar();
      drawBuf.element(0).assign(n.mul(uint(CASTER_VERTS_PER_CLUSTER)));
      drawBuf.element(1).assign(uint(1));
      drawBuf.element(2).assign(uint(0));
      drawBuf.element(3).assign(uint(0));
    })().compute(1, [1]);
    (kCasterArgs as unknown as { setName(n: string): unknown }).setName(`nanCasterArgs${c}`);

    // vertex-pulling caster: world position injected via castShadowPositionNode
    // (LOCAL space; identity matrixWorld ⇒ LOCAL == WORLD), so three's
    // light-camera MVP projects it during cascade c's shadow render.
    const mat = new NodeMaterial();
    mat.name = `nanCaster${c}`;
    // DoubleSide so the override's shadow side is DoubleSide (front→back flip
    // would cull terrain's single up-faces ⇒ no terrain shadow). Depth-only:
    // the shared shadow override owns depth; colorWrite/lights are moot (the
    // caster never renders in the main pass — layer-gated off camera layer 0).
    mat.side = DoubleSide;
    mat.colorWrite = false;
    mat.fog = false;
    mat.lights = false;
    // CRITICAL: base NodeMaterial leaves `map` UNDEFINED, but three's shadow
    // setup gates on `material.map !== null` (Renderer._getShadowNodes) — so
    // undefined makes it try reference('map','texture',material) on a missing
    // texture → "texture() expects a valid Texture" → the shadow override fails
    // to build → no shadows. MeshStandardNodeMaterial sets map=null; we must too.
    (mat as unknown as { map: unknown }).map = null;
    (mat as unknown as { castShadowPositionNode: unknown }).castShadowPositionNode = worldPos(cull);

    const geo = new BufferGeometry();
    geo.setAttribute('position', new Float32BufferAttribute(new Float32Array(3), 3));
    geo.setIndirect(drawAttr, 0);
    geo.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);

    const mesh = new Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = false;
    mesh.frustumCulled = false;
    // pin matrixWorld = identity: castShadowPositionNode is LOCAL space and three
    // does modelViewMatrix · positionLocal — only correct when modelWorld = I.
    mesh.matrixAutoUpdate = false;
    mesh.matrixWorldAutoUpdate = false;
    mesh.matrixWorld.identity();
    mesh.layers.set(2 + c); // ONLY cascade c's shadow camera renders this caster

    if (dbgCaster && c === 0) {
      // visualize cascade 0's vertex-pulled geometry in the MAIN camera pass:
      // emissive green, depthTest off (always on top). If this shows the near
      // terrain/trunks correctly, the decode is good and the bug is shadow-side.
      mat.colorWrite = true;
      mat.depthTest = false;
      mat.depthWrite = false;
      mat.vertexNode = Fn(() =>
        cameraProjectionMatrix.mul(cameraViewMatrix.mul(vec4(worldPos(cull), 1))),
      )() as unknown as typeof mat.vertexNode;
      mat.fragmentNode = Fn(() => vec4(0, 1, 0, 1))() as unknown as typeof mat.fragmentNode;
      mesh.layers.enable(0); // also render in the main camera pass
    }

    cascades.push({ cam, cull, kCasterArgs, mesh, count: -1 });
  }

  const cascM = new Matrix4();
  const cascFrustum = new Frustum();

  const update = (
    renderer: Renderer,
    csm: object | null,
    mainCamera: PerspectiveCamera,
  ): void => {
    const lights = (csm as { lights?: { shadow?: { camera?: CascadeLightCam } }[] } | null)
      ?.lights;
    if (!lights) return;
    for (let c = 0; c < SHADOW_CASCADES; c++) {
      const lcam = lights[c]?.shadow?.camera;
      // CSM inits its lwLights lazily; left is finite once the fit ran
      if (!lcam || !Number.isFinite(lcam.left ?? NaN)) continue;
      lcam.layers.enable(2 + c); // caster siblings on this layer only
      cascM.multiplyMatrices(lcam.projectionMatrix, lcam.matrixWorldInverse);
      cascFrustum.setFromProjectionMatrix(cascM);
      const cam = cascades[c]!.cam;
      for (let p = 0; p < 6; p++) {
        const pl = cascFrustum.planes[p];
        if (pl) cam.planes.array[p]?.set(pl.normal.x, pl.normal.y, pl.normal.z, pl.constant);
      }
      cam.camPos.value.copy(mainCamera.position); // LOD by the camera, not the light
      cascades[c]!.cull.runPhase1(renderer); // no occlusion → no phase 2
      dispatch(renderer, cascades[c]!.kCasterArgs); // visible count → draw args
    }
  };

  const readCounts = async (renderer: Renderer): Promise<number[]> => {
    const out = await Promise.all(
      cascades.map(async (cc) => {
        const counts = await cc.cull.readCounts(renderer);
        cc.count = counts.visClusters;
        return counts.visClusters;
      }),
    );
    return out;
  };

  return {
    update,
    readCounts,
    cascades: SHADOW_CASCADES,
    casterMeshes: cascades.map((cc) => cc.mesh),
  };
}
