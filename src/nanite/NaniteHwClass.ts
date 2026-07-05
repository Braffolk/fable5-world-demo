/**
 * ?clhw — per-CLUSTER SW/HW raster classifier, SHARED by kHwPartition (NaniteCull) and the
 * SW world1 kernel early-out (NaniteRaster). Returns TRUE when the cluster's triangles
 * project large enough that the HARDWARE rasterizer should draw the whole cluster and the
 * SW compute raster should skip it.
 *
 * Using ONE helper in both sites is the correctness keystone: the SW-skip and the HW-cluster
 * append make a BIT-IDENTICAL decision (same reads, same ops), so they can never disagree —
 * every triangle cluster is drawn by EXACTLY ONE path (no holes, no double-raster) WITHOUT
 * mutating qRaster (the resolve / soup / voxel readers stay untouched). Misclassification, if
 * the projection is imperfect, is therefore perf-only: a too-large cluster on SW is still
 * i32-safe up to ~128 px, and a too-small cluster on HW just draws correctly.
 *
 * projTri ≈ projK · (clusterRadius / sqrt(triCount)) / nearestDepth. A cluster crossing the
 * near plane has nearestDepth → 0 ⇒ projTri → ∞ ⇒ HW, so the near-plane case is subsumed (no
 * separate test). Sphere math mirrors the cut's instWorldSphere exactly.
 * See docs/mobile-gpu-perf/SW-HW-CLUSTER-AUDIT.md.
 */
import { If, float, max, sqrt, uint, vec3 } from 'three/tsl';
import type { NB, NF, NU, NV3, NV4 } from '../gpu/TSLTypes';
import { MESH_WORDS, readCluster } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import { instSphereRadius, instTransformPoint, instYaw } from './NaniteCommon';
import { bcU2F, elemU } from './Tsl';

/** TRUE ⇒ this cluster's tris are big enough on screen that the HW rasterizer should draw the
 *  whole cluster (SW skips it). `swmaxCl` = the SW/HW crossover in pixels (?clhwmax, default 16). */
export function clusterHwClass(
  gpu: RegistryGpu,
  camPos: NV3,
  projK: NF,
  instId: NU,
  ci: NU,
  swmaxCl: number,
): NB {
  const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
  const B = gpu.instances.element(instId.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
  const c = readCluster(gpu.clusters, ci);
  const isHF = c.flags.bitAnd(uint(1)).notEqual(uint(0));
  const swayPad = bcU2F(elemU(gpu.meshes, c.meshId.mul(uint(MESH_WORDS)).add(uint(11))));
  const yawSc = instYaw(B);
  // world-space cluster bounding sphere (mirrors NaniteCull.instWorldSphere)
  const centerW = vec3(0).toVar();
  const radiusW = float(0).toVar();
  If(isHF, () => {
    centerW.assign(c.sphere.xyz);
    radiusW.assign(c.sphere.w);
  }).Else(() => {
    centerW.assign(instTransformPoint(A, B, yawSc, c.sphere.xyz as unknown as NV3));
    radiusW.assign(instSphereRadius(A, B, c.sphere.w, swayPad));
  });
  // nearest surface depth (near-plane crossing ⇒ →0 ⇒ projTri huge ⇒ HW)
  const nearestDepth = camPos
    .sub(centerW as unknown as NV3)
    .length()
    .sub(radiusW)
    .max(float(1e-3)) as unknown as NF;
  // representative tri world size = sphere radius / sqrt(triCount)
  const triWorld = radiusW.div(sqrt(max(float(1), float(c.triCount)))) as unknown as NF;
  const projTri = projK.mul(triWorld).div(nearestDepth) as unknown as NF;
  return projTri.greaterThan(float(swmaxCl)) as unknown as NB;
}
