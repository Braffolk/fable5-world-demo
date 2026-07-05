/**
 * S3-perf — SHARED cross-level cull for the shadow CLIPMAP (the brute-deletion
 * unlock; D-N29, LOG bw/bx).
 *
 * The clipmap's L concentric ortho levels all select the SAME geometric LOD cut:
 * the cut is `project(ownError) ≤ τ` with projK (constant ortho), cam.camPos (MAIN
 * camera) and τ IDENTICAL across levels — only the FRUSTUM (each level's ortho box)
 * and the HOLLOW (innerReject ring) differ. So running a full hier BFS per level (as
 * the first SHADOW-HIER cut did) re-walks the SAME DAG L times. This module walks it
 * ONCE into a shared CUT list, then runs a cheap single-pass FILTER per level that
 * applies only that level's frustum + hollow and re-fills a shared raster queue:
 *
 *   runSharedCut : one buildNaniteCull(hier) traverse over the OUTER box (covers all
 *                  levels; no hollow) → cut = the cluster list at the LOD cut, plus a
 *                  one-shot filterArgs (dispatch sized to the cut count, reused by all
 *                  level filters this frame).
 *   runLevelFilter(k): clear the shared queue counter → kFilter_k (read cut, recompute
 *                  each cluster's world sphere, test level-k frustum + level-k hollow,
 *                  append survivors) → kRasterArgs_k (queue[0] = count, raster dispatch,
 *                  countBuf[k] = count). The level's raster then consumes the shared
 *                  queue (levels are processed sequentially, reusing one vis buffer +
 *                  one queue — the clipmap already serializes them).
 *
 * Cost: 1 traverse + L single-pass filters, vs the per-level path's L traverses, vs
 * brute's L inst-cull→chunk→cluster-cull chains. The traverse (the expensive 18-pass
 * BFS) is paid ONCE; the filters are flat frustum+hollow tests over the cut. The cut
 * only runs when ≥1 level re-rasters (the caller gates it on the R1 cadence), so a
 * static camera stays ~0 cost.
 *
 * CAVEAT (the shared-cut precondition): valid ONLY while every level shares one τ (the
 * clipmap default). A future per-level τ (S4-on-clipmap: coarsen far levels' GEOMETRY)
 * would give each level a DIFFERENT cut ⇒ the shared traverse breaks. The cascades
 * (NaniteShadow) already differ per cascade (CASCADE_TAU_MUL) and keep per-cascade hier.
 */

import { IndirectStorageBufferAttribute, StorageBufferAttribute } from 'three/webgpu';
import type { Renderer } from 'three/webgpu';
import {
  Fn,
  If,
  Loop,
  abs,
  atomicAdd,
  atomicMax,
  atomicStore,
  dot,
  float,
  instanceIndex,
  int,
  uint,
  vec3,
  vec4,
} from 'three/tsl';
import type { NB, NF, NU, NV3, NV4 } from '../gpu/TSLTypes';
import { MESH_WORDS, readCluster } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import { buildNaniteCull, type NaniteCullChain } from './NaniteCull';
import {
  DISPATCH_ROW,
  instSphereRadius,
  instTransformPoint,
  instYaw,
  queueCapParam,
  registerQueueHw,
  type NaniteCam,
} from './NaniteCommon';
import {
  aLoadU,
  bcU2F,
  dispatch,
  dispatchIndirect,
  elemU,
  elemUW,
  localX,
  maxU,
  minU,
  readBuffer,
  returnIf,
  sU32Views,
  setIndirectDispatch,
  sUvec2,
  uv2,
  wgLinear,
  type UniformArrV4,
  type UniformF,
} from './Tsl';
import type { BufOf, UV2 } from './Tsl';

interface ComputeKernel {
  setName(name: string): unknown;
}

/** the shared raster queue every clipmap level's raster consumes (sequential reuse) */
export interface ClipLevelQueue {
  /** the queue's item capacity (?qshcap) — consumers clamp counts against THIS,
   *  not the camera's QRASTER_CAP (which may now be smaller than the shadow cap) */
  cap: number;
  qRasterRO: BufOf<UV2>;
  rasterDispatchAttr: IndirectStorageBufferAttribute;
  rasterDispatch2Attr: IndirectStorageBufferAttribute;
  rasterDispatchFullAttr: IndirectStorageBufferAttribute;
}

export interface ClipCull {
  /** the shared cut producer (one hier BFS over the outer box) */
  shared: NaniteCullChain;
  /** shared raster queue — pass to every level's buildNaniteRaster */
  queue: ClipLevelQueue;
  /** the ONE traverse + filterArgs; call once/frame when ≥1 level re-rasters */
  runSharedCut(renderer: Renderer): void;
  /** CAMERA||SHADOW OVERLAP (item 4): the shared-cut cull's ordered batch (the hier BFS
   *  + kFilterArgs), as ONE list, so a caller can CONCATENATE it with the camera cull's
   *  phase1Batch() into a SINGLE submit and let Dawn overlap the two (they write DISJOINT
   *  buffers: this cull owns fresh counters/qRaster/qFrontier, sphereOccluded=null ⇒ no
   *  HZB dep). Same internal order/RAW as runSharedCut. The caller dispatches the combined
   *  batch INSTEAD of runSharedCut (do not call both). */
  sharedCutBatch(): readonly unknown[];
  /** clear queue → filter the cut by level k's frustum+hollow → raster args */
  runLevelFilter(renderer: Renderer, level: number): void;
  /** P9 (submit coalescing): the SAME three kernels runLevelFilter dispatches, as
   *  an ordered batch list (the filter pre-tagged with its indirect args) so the
   *  caller can fold them into ONE dispatchBatchMixed submit with the rest of the
   *  level's compute chain. */
  levelFilterBatch(level: number): readonly unknown[];
  /** cut size + per-level survivor counts (HUD) */
  readCounts(renderer: Renderer): Promise<{ cut: number; perLevel: number[] }>;
}

export function buildClipCull(
  gpu: RegistryGpu,
  instanceCount: number,
  /** the cut cam: projK (constant ortho) + camPos (MAIN camera) + the OUTER box
   *  frustum + τ — set each frame by the caller before runSharedCut */
  cutCam: NaniteCam,
  /** per-level cams (each level's ortho VP + frustum planes, set each frame) */
  levelCams: NaniteCam[],
  /** per-level hollow uniform (1/E_k for k≥1, 0 for level 0) */
  innerRejects: UniformF[],
  opts: {
    minPx: UniformF;
    frontierCap: number;
    hierDepth?: number;
    /** P5 TOROIDAL STRIPS (shadow arc 2026-07-03): per-level active-rect uniforms —
     *  4 rects in the level's WINDOW UV space (x0,y0,x1,y1; empty = x1<=x0). When
     *  present, the level filter ALSO drops clusters whose uv-box misses every
     *  active rect, so the raster only touches the newly-exposed texel strips
     *  (the strip-scoped kCopy publishes exactly those texels). */
    strips?: UniformArrV4[];
    /** per-level ortho half-extents E_k (static) — bakes radius→uv scale into the
     *  strip test as a compile-time constant. Required when strips is set. */
    levelHalves?: number[];
    /** P10: ring-snapped LOD distance for the shared cut (see NaniteCull) */
    lodRingSnap?: number;
    /** P11 caster-LOD warp (fly-through spike fix 2026-07-04): τ + the lodWarp
     *  params for the SHARED cut. One distance-based warp is legal here — the
     *  cut stays IDENTICAL across levels (unlike a per-level τ, which would
     *  break the shared traverse). Casters coarsen with camera distance the
     *  same way the main view does (which never reached the shadow path). */
    tau?: UniformF;
    lodNear?: UniformF;
    lodPow?: UniformF;
    simBandD?: UniformF;
    /** shadow-only voxel-τ coarsen (>1) → the shvox2 caster reads fewer/bigger bricks
     *  (the "less detailed crown in the shadow"). Threaded to the shared cut; voxel
     *  matClass only, so trunk casters are unaffected. */
    voxCoarsen?: number;
    /** shadow-only: seed voxel roots at ALL distances so the voxel crown casts shadows
     *  in the near band (<60 m) too — the leaf mesh is castShadows:false. */
    seedVoxAllDist?: boolean;
  },
): ClipCull {
  const LEVELS = levelCams.length;

  // SHADOW queue cap (?qshcap; memory sizing 2026-07-04): the shared cut is
  // ring-bounded (the outer clipmap box / active strip union), far below the
  // camera chain's far-field demand — sizing its qRaster AND the per-level qLevel
  // twin at the old 8M QRASTER_CAP wasted 2×67 MB GPU (+ the permanent CPU
  // mirrors). MEASURED demand (2026-07-04 world @dpr2): cut HW 89 k on a 12 s
  // 8 m/s walk (strip re-rasters), 54 k on the boot FULL all-level re-raster,
  // 0 when still (strips cached); qLevel HW 33 k. Default 2^18 = 262,144 (~2.9×
  // the worst case) for BOTH queues (the filter only ever selects a subset of
  // the cut, and a full-invalidate level filter can approach the whole cut).
  // Overflow = clusters dropped this frame (missing casters) + readCounts flag.
  const shCap = queueCapParam('qshcap', 262_144, 4_096, 8_388_608);

  // The shared cut: a normal hier cull over cutCam, NO hollow (innerReject default
  // 0), shared minPx. Its qRaster IS the cut list (instId, ci) at the LOD cut within
  // the outer box. runPhase1 also writes qRaster[0] = (count, 0) — the filter's size.
  const shared = buildNaniteCull(gpu, instanceCount, cutCam, null, {
    coneCull: false,
    minPx: opts.minPx,
    frontierCap: opts.frontierCap,
    hierDepth: opts.hierDepth, // item 6: BFS passes = measured max DAG depth (no holes, sheds tail)
    lodRingSnap: opts.lodRingSnap, // P10: ring-snapped shadow LOD (no strip LOD-age)
    tau: opts.tau, // P11: caster-LOD τ + distance warp (shared-cut-legal)
    lodNear: opts.lodNear,
    lodPow: opts.lodPow,
    simBandD: opts.simBandD,
    voxCoarsen: opts.voxCoarsen, // shadow "less detailed" crown — coarsen voxel clusters
    seedVoxAllDist: opts.seedVoxAllDist, // voxel casts the crown shadow in the near band too
    qRasterCap: shCap, // the cut queue rides the shadow cap, not the camera's
    // this chain NEVER runs the vox fan-out (only the camera cull calls
    // runVoxFanout) — minimum-size its qVoxRaster instead of 16.8 MB dead weight
    qVoxCap: 64,
    label: 'shadowCut',
  });
  const cutRO = shared.qRasterRO;

  // shared raster queue (one, refilled per level — levels raster sequentially)
  const qLevelAttr = new StorageBufferAttribute(new Uint32Array((shCap + 1) * 2), 2);
  qLevelAttr.name = 'nanShadowQLevel';
  const qLevel = sUvec2(qLevelAttr, shCap + 1);
  // append counter (slot 0) — cleared before each level filter
  const countAttr = new StorageBufferAttribute(new Uint32Array(1), 1);
  countAttr.name = 'nanShadowClipCount';
  const countV = sU32Views(countAttr, 1);
  // per-level survivor counts for the HUD (persist across the frame, indexed by level)
  const perLevelAttr = new StorageBufferAttribute(new Uint32Array(LEVELS), 1);
  perLevelAttr.name = 'nanShadowPerLevel';
  const perLevelV = sU32Views(perLevelAttr, LEVELS);

  const rasterDispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  rasterDispatchAttr.name = 'nanShadowRasterDispatch';
  const rasterDispatch = sU32Views(rasterDispatchAttr as unknown as StorageBufferAttribute, 3).rw;
  const rasterDispatch2Attr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  rasterDispatch2Attr.name = 'nanShadowRasterDispatch2';
  const rasterDispatch2 = sU32Views(rasterDispatch2Attr as unknown as StorageBufferAttribute, 3).rw;
  const rasterDispatchFullAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  rasterDispatchFullAttr.name = 'nanShadowRasterDispatchFull';
  const rasterDispatchFull = sU32Views(
    rasterDispatchFullAttr as unknown as StorageBufferAttribute,
    3,
  ).rw;
  const filterDispatchAttr = new IndirectStorageBufferAttribute(new Uint32Array(3), 3);
  filterDispatchAttr.name = 'nanShadowFilterDispatch';
  const filterDispatch = sU32Views(filterDispatchAttr as unknown as StorageBufferAttribute, 3).rw;

  const split2D = (args: ReturnType<typeof sU32Views>['rw'], n: NU): void => {
    const rows = n.add(uint(DISPATCH_ROW - 1)).div(uint(DISPATCH_ROW));
    elemUW(args, 0).assign(minU(n, uint(DISPATCH_ROW)));
    elemUW(args, 1).assign(maxU(rows, uint(1)));
    elemUW(args, 2).assign(uint(1));
  };

  // ---- world-sphere + frustum helpers (mirror NaniteCull's locals) -----------------
  const frustumVisible = (planes: NaniteCam['planes'], center: NV3, radius: NF): NF => {
    const visible = float(1).toVar();
    Loop(6, ({ i: pi }) => {
      const plane = planes.element(pi);
      const d = dot(plane.xyz, center).add(plane.w) as unknown as NF;
      If(d.lessThan(radius.negate()), () => {
        visible.assign(0);
      });
    });
    return visible as unknown as NF;
  };
  const instWorldSphere = (
    A: NV4,
    B: NV4,
    isHF: NB,
    localSphere: NV4,
    swayPad: NF,
  ): { center: NV3; radius: NF } => {
    const yawSc = instYaw(B);
    const centerW = vec3(0).toVar();
    const radiusW = float(0).toVar();
    If(isHF, () => {
      centerW.assign(localSphere.xyz);
      radiusW.assign(localSphere.w);
    }).Else(() => {
      centerW.assign(instTransformPoint(A, B, yawSc, localSphere.xyz as unknown as NV3));
      radiusW.assign(instSphereRadius(A, B, localSphere.w, swayPad));
    });
    return { center: centerW as unknown as NV3, radius: radiusW as unknown as NF };
  };

  // ---- queue high-water diag (memory sizing): raw qLevel append maxima -------------
  // [0] = max RAW per-level survivor count (pre-clamp) across levels since reset.
  const hwAttr = new StorageBufferAttribute(new Uint32Array(1), 1);
  hwAttr.name = 'nanShadowClipHwm';
  const hwV = sU32Views(hwAttr, 1).atomic;
  const kHwReset = Fn(() => {
    If(instanceIndex.equal(uint(0)), () => {
      atomicStore(hwV.element(0), uint(0));
    });
  })().compute(1, [1]);
  (kHwReset as unknown as ComputeKernel).setName('nanClipQHwReset');
  registerQueueHw({
    label: 'shadowQLevel',
    caps: { qLevel: shCap },
    read: async (renderer: unknown): Promise<Record<string, number>> => {
      const u = new Uint32Array(await readBuffer(renderer as Renderer, hwAttr, 0, 4));
      return { qLevel: u[0] ?? 0 };
    },
    reset: (renderer: unknown): void => {
      dispatch(renderer as Renderer, kHwReset);
    },
  });

  // ---- filterArgs: size the (shared) filter dispatch from the cut count -------------
  const kFilterArgs = Fn(() => {
    const n = minU(cutRO.element(0).x, uint(shCap));
    split2D(filterDispatch, n.add(uint(63)).div(uint(64)));
  })().compute(1, [1]);
  (kFilterArgs as unknown as ComputeKernel).setName('nanClipFilterArgs');

  // ---- per-level filter: cut → (frustum_k + hollow_k) → shared queue ----------------
  const filters: unknown[] = [];
  const rasterArgsKernels: unknown[] = [];
  const clears: unknown[] = [];
  for (let k = 0; k < LEVELS; k++) {
    const cam = levelCams[k]!;
    const innerInvHalf = innerRejects[k]!;

    const kClear = Fn(() => {
      If(instanceIndex.equal(uint(0)), () => {
        atomicStore(countV.atomic.element(0), uint(0));
      });
    })().compute(1, [1]);
    (kClear as unknown as ComputeKernel).setName(`nanClipClear${k}`);
    clears.push(kClear);

    const kFilter = Fn(() => {
      const tid = wgLinear(DISPATCH_ROW).mul(uint(64)).add(localX()).toVar();
      returnIf(tid.greaterThanEqual(minU(cutRO.element(0).x, uint(shCap))));
      const item = cutRO.element(tid.add(uint(1)));
      const instId = item.x.toVar();
      const ci = item.y.toVar();
      const c = readCluster(gpu.clusters, ci);
      const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
      const B = gpu.instances
        .element(instId.mul(uint(2)).add(uint(1)))
        .toVar() as unknown as NV4;
      const isHF = c.flags.bitAnd(uint(1)).notEqual(uint(0)).toVar();
      const swayPad = bcU2F(elemU(gpu.meshes, c.meshId.mul(uint(MESH_WORDS)).add(uint(11))));
      const s = instWorldSphere(A, B, isHF as unknown as NB, c.sphere, swayPad);
      const visible = frustumVisible(cam.planes, s.center, s.radius).toVar();
      // hollow: drop clusters wholly inside the next-finer level's [±0.5] box
      If(visible.greaterThan(0.5).and(innerInvHalf.greaterThan(0)), () => {
        const clip = cam.vp.mul(vec4(s.center, 1)) as unknown as NV4;
        const rClip = s.radius.mul(innerInvHalf);
        If(
          abs(clip.x).add(rClip).lessThan(0.5).and(abs(clip.y).add(rClip).lessThan(0.5)),
          () => {
            visible.assign(0);
          },
        );
      });
      // P5 strips: drop clusters whose uv-box misses every active rect — the level
      // only publishes strip texels this frame, so anything else is dead raster work.
      // Ortho ⇒ clip.w ≡ 1, ndc = clip; uv = ndc·0.5+0.5; radius→uv = r/(2·E_k).
      if (opts.strips) {
        const rects = opts.strips[k]!;
        const rUvK = 1 / (2 * (opts.levelHalves?.[k] ?? 1));
        If(visible.greaterThan(0.5), () => {
          const clip = cam.vp.mul(vec4(s.center, 1)) as unknown as NV4;
          const ux = clip.x.mul(0.5).add(0.5);
          const uy = clip.y.mul(0.5).add(0.5);
          const rUv = s.radius.mul(rUvK);
          const hit = float(0).toVar();
          for (let r = 0; r < 4; r++) {
            const rect = rects.element(int(r)) as unknown as NV4;
            If(
              ux
                .add(rUv)
                .greaterThanEqual(rect.x)
                .and(ux.sub(rUv).lessThan(rect.z))
                .and(uy.add(rUv).greaterThanEqual(rect.y))
                .and(uy.sub(rUv).lessThan(rect.w)),
              () => {
                hit.assign(1);
              },
            );
          }
          If(hit.equal(0), () => {
            visible.assign(0);
          });
        });
      }
      If(visible.greaterThan(0.5), () => {
        const slot = atomicAdd(countV.atomic.element(0), uint(1)) as unknown as NU;
        If(slot.lessThan(uint(shCap)), () => {
          qLevel.rw.element(slot.add(uint(1))).assign(uv2(instId, ci));
        });
      });
    })().compute(shCap, [64]);
    (kFilter as unknown as ComputeKernel).setName(`nanClipFilter${k}`);
    filters.push(kFilter);

    const kRasterArgs = Fn(() => {
      // queue high-water diag: RAW survivor count before the cap clamp
      atomicMax(hwV.element(0), aLoadU(countV.atomic.element(0)));
      const n = minU(aLoadU(countV.atomic.element(0)), uint(shCap));
      qLevel.rw.element(0).assign(uv2(n, 0));
      elemUW(perLevelV.rw, uint(k)).assign(n);
      split2D(rasterDispatch, n);
      // depth-only shadow raster uses rasterDispatchAttr (phase-1); keep full/2 in
      // sync so a shared raster's payload/combined path (unused here) stays valid.
      split2D(rasterDispatchFull, n);
      split2D(rasterDispatch2, uint(0));
    })().compute(1, [1]);
    (kRasterArgs as unknown as ComputeKernel).setName(`nanClipRasterArgs${k}`);
    rasterArgsKernels.push(kRasterArgs);
  }

  const runSharedCut = (renderer: Renderer): void => {
    shared.runPhase1(renderer); // the ONE hier BFS → cut in qRasterRO
    dispatch(renderer, kFilterArgs as never); // size the per-level filter dispatch
  };

  // item 4: the shared-cut batch = the camera-disjoint hier BFS + kFilterArgs, as ONE
  // ordered list. RAW inside: shared.phase1Batch() ends with its kRasterArgs (writes the
  // cut's qRaster[0]); kFilterArgs reads cutRO.element(0).x (the cut count) → sizes the
  // per-level filter dispatch. Concatenated after the camera cull's phase1Batch() the two
  // halves touch no common writable buffer, so Dawn may overlap them in the one submit.
  const sharedCutBatch = (): readonly unknown[] => [...shared.phase1Batch(), kFilterArgs];

  const runLevelFilter = (renderer: Renderer, level: number): void => {
    dispatch(renderer, clears[level] as never);
    dispatchIndirect(renderer, filters[level] as never, filterDispatchAttr);
    dispatch(renderer, rasterArgsKernels[level] as never);
  };

  // P9: pre-tag each level filter with its indirect args once (idempotent) and
  // hand back the ordered [clear, filter, rasterArgs] list for batched submits.
  for (const f of filters) setIndirectDispatch(f, filterDispatchAttr);
  const levelFilterBatch = (level: number): readonly unknown[] => [
    clears[level],
    filters[level],
    rasterArgsKernels[level],
  ];

  const readCounts = async (
    renderer: Renderer,
  ): Promise<{ cut: number; perLevel: number[] }> => {
    const [head, lv] = await Promise.all([
      readBuffer(renderer, shared.qRasterAttr, 0, 8),
      readBuffer(renderer, perLevelAttr, 0, LEVELS * 4),
    ]);
    const cut = new Uint32Array(head)[0] ?? 0;
    const perLevel = Array.from(new Uint32Array(lv));
    return { cut, perLevel };
  };

  return {
    shared,
    queue: { cap: shCap, qRasterRO: qLevel.ro, rasterDispatchAttr, rasterDispatch2Attr, rasterDispatchFullAttr },
    runSharedCut,
    sharedCutBatch,
    runLevelFilter,
    levelFilterBatch,
    readCounts,
  };
}
