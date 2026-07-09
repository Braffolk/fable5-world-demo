/**
 * ?clhw — per-CLUSTER SW/HW raster classifier, SHARED by kHwPartition (NaniteCull), the
 * ClusterCtx slot-11 write (the world1 SW-skip broadcast), the inline world1 classify
 * fallbacks (NaniteRaster, non-default builds) and the ?nandbg=clhw tint (NaniteResolve).
 * Returns TRUE when the cluster's triangles project large enough that the HARDWARE
 * rasterizer should draw the whole cluster and the SW compute raster should skip it.
 *
 * Using ONE helper in all sites is the correctness keystone: the SW-skip and the HW-cluster
 * append make a BIT-IDENTICAL decision (same reads, same ops), so they can never disagree —
 * every triangle cluster is drawn by EXACTLY ONE path (no holes, no double-raster) WITHOUT
 * mutating qRaster (the resolve / soup / voxel readers stay untouched). Misclassification, if
 * the projection is imperfect, is therefore perf-only: a too-large cluster on SW is still
 * i32-safe up to ~128 px, and a too-small cluster on HW just draws correctly.
 *
 * projTri ≈ projK · (clusterRadius / sqrt(triCount)) / nearestDepth.
 *
 * MESH HW-eligibility carries two EXTRA gates (HW vertex-prepass, 2026-07-09) because the
 * mesh `_clE` HW draw reads PRE-PROJECTED verts from projVertBuf (no real clip, no fallback):
 *   (a) NEAR gate — the cluster must be provably CLEAR of the near plane. Metric = VIEW-Z
 *       (not Euclidean): min view-z over the padded sphere = clipW(center) − radiusPadded,
 *       where clipW = (vp·(center,1)).w ≡ dot(camForward, center − camPos) for a rigid view +
 *       standard perspective — the EXACT w projectVert guards on. radiusPadded already
 *       includes swayPad (instSphereRadius), so wind cannot escape the bound. IMPLIED BOUND:
 *       with view-z − radiusPadded > NEAR_MARGIN and the frustum cull passed, every vertex of
 *       the cluster satisfies projectVert's ok-test (w ≥ NEAR_MARGIN ≫ 1e-4, and |screen·256|
 *       < 1e8 since |ndc| stays bounded by the frustum-overlapping sphere at w ≥ NEAR_MARGIN)
 *       ⇒ its projVertBuf records are real values, NEVER NEAR_SENTINEL — the `_clE` reader
 *       needs no sentinel branch. A near-crossing mesh cluster stays SW, where the per-tri
 *       classifier routes NEAR_SENTINEL tris to the hwQueue SOUP draw with real clip.
 *   (b) COVERAGE gate — vcompact[ci·2+1] > 0 (the cluster's verts span a dense window) ⇒ the
 *       projection pre-pass takes its per-UNIQUE-VERT path, so projecting mesh HW clusters
 *       cannot re-balloon nanProjectVerts (the reverted Phase-2 failure mode). Non-covered
 *       mesh clusters stay SW.
 * TERRAIN (isHF) keeps its original size-only eligibility UNCHANGED — its `_clT` draw keeps
 * the full compute-fetch vertex path (real clip; it never reads projVertBuf), and its old
 * near-plane subsumption (nearestDepth → 0 ⇒ projTri → ∞ ⇒ HW) still holds.
 * Sphere math mirrors the cut's instWorldSphere exactly.
 * See docs/mobile-gpu-perf/SW-HW-CLUSTER-AUDIT.md.
 */
import { If, float, max, sqrt, uint, vec3, vec4 } from 'three/tsl';
import type { NB, NF, NU, NV3, NV4 } from '../gpu/TSLTypes';
import { MESH_WORDS, readCluster } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import type { NaniteCam } from './NaniteCommon';
import { instSphereRadius, instTransformPoint, instYaw } from './NaniteCommon';
import { bcU2F, elemU } from './Tsl';

// Mesh near gate margin = max(4·nearPlane, 0.3) with the Engine camera near = 0.3 ⇒ 1.2.
// Every mesh HW vertex then has clip-w ≥ NEAR_MARGIN — the bound the `_clE` projVertBuf
// reader relies on (see header (a)).
const NEAR_MARGIN = Math.max(4 * 0.3, 0.3);

/** ?hwproj=1 — the HW vertex-prepass flagship (the `_clE` projVertBuf reader + the mesh
 *  near/coverage routing gates + the narrowed Project.ts skip). OPT-IN (default OFF,
 *  2026-07-09 23:xx): with the gates active the SW-skip (ClusterCtx slot-11) and the HW
 *  partition must agree on a CAMERA-DEPENDENT near test evaluated in two different kernels —
 *  any per-frame skew between them (uniform update timing / jitter) makes a marginal cluster
 *  SW-skipped but not HW-drawn ⇒ large tris flicker out near trunks (user-observed at melee
 *  range, round-2 bisect). The old size-only rule has the same theoretical skew but its
 *  disagreement zone is distant sub-pixel clusters. Re-enable only with a consistency fix
 *  (single-kernel decision or slot-11-as-the-only-source-of-truth for routing). */
export const HWPROJ =
  typeof location !== 'undefined' &&
  new URLSearchParams(location.search).get('hwproj') === '1';

/** TRUE ⇒ this cluster's tris are big enough on screen that the HW rasterizer should draw the
 *  whole cluster (SW skips it). `swmaxCl` = the SW/HW crossover in pixels (CLHW_MAX).
 *  Mesh clusters additionally require the near + vcompact-coverage gates (header (a)/(b)). */
export function clusterHwClass(
  gpu: RegistryGpu,
  cam: NaniteCam,
  projK: NF,
  instId: NU,
  ci: NU,
  swmaxCl: number,
): NB {
  const camPos = cam.camPos as unknown as NV3;
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
  // nearest surface depth (near-plane crossing ⇒ →0 ⇒ projTri huge ⇒ HW — terrain only;
  // the mesh near gate below overrides that route with the SW path)
  const nearestDepth = camPos
    .sub(centerW as unknown as NV3)
    .length()
    .sub(radiusW)
    .max(float(1e-3)) as unknown as NF;
  // representative tri world size = sphere radius / sqrt(triCount)
  const triWorld = radiusW.div(sqrt(max(float(1), float(c.triCount)))) as unknown as NF;
  const projTri = projK.mul(triWorld).div(nearestDepth) as unknown as NF;
  const bigEnough = projTri.greaterThan(float(swmaxCl)) as unknown as NB;
  // mesh gate (a): min view-z over the padded sphere via clip-w (see header). Perspective vp
  // row 4 = (0,0,−1,0)·view ⇒ w(p) = dot(camForward, p − camPos) exactly, unit scale; w is
  // affine in p ⇒ min over the sphere = w(center) − radiusW.
  const wC = (
    cam.vp.mul(vec4(centerW as unknown as NV3, 1)) as unknown as NV4
  ).w as unknown as NF;
  const nearClear = wC.sub(radiusW).greaterThan(float(NEAR_MARGIN)) as unknown as NB;
  // mesh gate (b): vcompact coverage — vcCount > 0 ⇒ the projection pre-pass dedups this
  // cluster (per-unique-vert path), so its projVertBuf records exist AND stay cheap.
  const covered = elemU(gpu.vcompact, ci.mul(uint(2)).add(uint(1))).greaterThan(
    uint(0),
  ) as unknown as NB;
  if (!HWPROJ) return bigEnough; // pre-flagship rule: size only (see HWPROJ note)
  return bigEnough.and(
    (isHF as unknown as NB).or(nearClear.and(covered)),
  ) as unknown as NB;
}
