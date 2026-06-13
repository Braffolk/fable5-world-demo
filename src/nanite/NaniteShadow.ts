/**
 * N5 — cluster-driven CSM shadow casters (D-N26). C0: the per-cascade CULL.
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
 * C0 produces per-cascade visible-cluster lists + indirect args and a HUD count;
 * no raster yet (C1 adds the HW caster mesh that draws these into the cascades).
 *
 * Each cascade camera is pinned to layer 2+c here too (cam.layers.enable) so C1's
 * caster siblings on that layer render into ONLY that cascade — the Forests hook.
 */

import { Frustum, Matrix4 } from 'three';
import type { PerspectiveCamera } from 'three';
import type { Renderer } from 'three/webgpu';
import type { RegistryGpu } from './GeometryRegistry';
import { makeNaniteCam, type NaniteCam } from './NaniteCommon';
import { buildNaniteCull, type NaniteCullChain } from './NaniteCull';

/** CSM default cascade count (csmcasc can lower it — we guard per-cascade) */
export const SHADOW_CASCADES = 4;

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
  /** last-read visible-cluster count (HUD) */
  count: number;
}

export interface NaniteShadow {
  /** refresh per-cascade planes from the CSM lights + run each cascade cull.
   *  Call BEFORE post.render() (the cascade cameras hold last frame's fit — one
   *  frame stale, hidden in lightMargin like Forests). No-op until CSM inits. */
  update(renderer: Renderer, csm: object | null, mainCamera: PerspectiveCamera): void;
  /** per-cascade visible-cluster counts (HUD; -1 if a cascade never ran) */
  readCounts(renderer: Renderer): Promise<number[]>;
  cascades: number;
}

export function buildNaniteShadow(gpu: RegistryGpu, instanceCount: number): NaniteShadow {
  const cascades: CascadeCull[] = [];
  for (let c = 0; c < SHADOW_CASCADES; c++) {
    // size is irrelevant to the cull (it reads only planes + camPos); the
    // shadow map is 2048² but the cull never touches width/height.
    const cam = makeNaniteCam(2048, 2048);
    const cull = buildNaniteCull(gpu, instanceCount, cam, null, { coneCull: false });
    cascades.push({ cam, cull, count: -1 });
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
      lcam.layers.enable(2 + c); // C1: caster siblings on this layer only
      cascM.multiplyMatrices(lcam.projectionMatrix, lcam.matrixWorldInverse);
      cascFrustum.setFromProjectionMatrix(cascM);
      const cam = cascades[c]!.cam;
      for (let p = 0; p < 6; p++) {
        const pl = cascFrustum.planes[p];
        if (pl) cam.planes.array[p]?.set(pl.normal.x, pl.normal.y, pl.normal.z, pl.constant);
      }
      cam.camPos.value.copy(mainCamera.position); // LOD by the camera, not the light
      cascades[c]!.cull.runPhase1(renderer); // no occlusion → no phase 2
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

  return { update, readCounts, cascades: SHADOW_CASCADES };
}
