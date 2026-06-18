/**
 * N2 vis-buffer raster on REGISTRY buffers — the spike pipeline (SpikeRaster)
 * ported onto GeometryRegistry mega-buffers + the world heightfield texture,
 * consuming the N2 cull chain's work queue. Option C only (D-N5): pass 1
 * atomicMin(f32-depth-bits), pass 2 equality payload store; near-crossing and
 * oversized triangles route to the HW queue whose fragment stage writes the
 * SAME buffers (depthWrite off — one resolve, one convention).
 *
 * Payload = (workItemIdx << CLUSTER_TRI_BITS) | localTri — qRaster doubles as the
 * visible-cluster list, so the resolve recovers (instance, cluster) in one
 * indirection. CLUSTER_TRI_BITS = log2(MAX_CLUSTER_TRIS) (7 @128, 8 @256 — the
 * ?clustertris A/B); itemIdx has 23 bits (QRASTER_CAP=2^23) ⇒ 23+8 = 31/32 at the
 * 256 cap, 1 spare.
 *
 * Scanline core (N3a): FIXED-POINT integer edge functions — verts snapped to
 * a 1/256-px grid (8 subpixel bits, HW convention), coverage + top-left rule
 * in exact i32 math (watertight; replaces the example's float −1e-5 bias).
 * Flat resolve = matClass palette or cluster-hash tint (the deferred N1
 * checkpoint), face-normal lambert, real f32 depth out.
 */

import { DoubleSide, Mesh, Scene, Vector3 } from 'three';
import { BufferGeometry, Float32BufferAttribute, RenderTarget, Sphere } from 'three';
import type { PerspectiveCamera, Texture } from 'three';
import {
  IndirectStorageBufferAttribute,
  NodeMaterial,
  StorageBufferAttribute,
  type Renderer,
} from 'three/webgpu';
import {
  Discard,
  Fn,
  If,
  atomicAdd,
  atomicMax,
  atomicMin,
  atomicStore,
  cross,
  dot,
  float,
  instanceIndex,
  max,
  normalize,
  positionGeometry,
  screenCoordinate,
  uint,
  varyingProperty,
  vec2,
  vec3,
  vec4,
  vertexIndex,
  workgroupArray,
  workgroupBarrier,
} from 'three/tsl';
import type { NB, NF, NI, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import { markFragmentWritable } from '../render/ThreePatches';
import {
  CLUSTER_TRI_BITS,
  CLUSTER_TRI_MASK,
  CLUSTER_WORDS,
  MAX_CLUSTER_TRIS,
  MESH_WORDS,
} from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import { DISPATCH_ROW, QRASTER_CAP, hashColor, instYaw, type NaniteCam } from './NaniteCommon';
import { makeFetch, type TerrainDisp, type TrunkWindOpt, type VertCtx } from './NaniteFetch';
import { makeVertexCache } from './NaniteVertexCache';
import { buildTileRaster } from './NaniteTileRaster';
import {
  aLoadU,
  bcF2U,
  bcU2F,
  dispatch,
  dispatchIndirect,
  elemU,
  localX,
  loopI,
  maxI,
  minI,
  minU,
  readBuffer,
  returnIf,
  sU32Views,
  toF,
  toI,
  wgLinear,
} from './Tsl';
import type { BufOf, UV2 } from './Tsl';

// N9-C0: leaf needles are long, thin, near-camera tris — their bbox exceeds the
// SW raster's 16 px i32-safe limit, so a dense crown routes hundreds of thousands
// of them to the HW vertex-pulling path. The old 262k cap overflowed (clamp →
// dropped tris → black holes in foliage; measured 876k hwTris in a dense stand).
// Sized for that load; the queue only costs memory + the HW pass only pays for the
// tris actually present. The real reduction (halving the dup, shedding far crowns)
// is the two-sided raster + the aggregate DAG (N9-C2).
const HW_CAP = 2_097_152;
const MAX_RASTER_SIZE = 16;
const NEAR_EPS = 1e-4;

/**
 * N9-C2 two-sided raster orientation (D-N43 Stage 0). The integer scanline core
 * below is built for POSITIVE-area (CCW) triangles — `area2`, `rcpArea`, the edge
 * walk and the top-left coverage rule all assume it. A single-sided cluster keeps
 * the classic back-face cull (accept front-faces, areaNdc > 0). A TWO-SIDED cluster
 * (leaf crowns) instead RE-WINDS a back-face to CCW in place — swap v1↔v2, which
 * negates the signed area (edgeFn = cross(v1−v0, v2−v0)) — so the SAME core rasters
 * whichever side faces the camera, exactly once. This replaces the leaf geometry's
 * reversed-winding triangle duplicate: half the leaf triangles, half the leaf
 * clusters, identical pixels (the resolve flips the leaf normal camera-ward, so
 * shading is unaffected by which winding rastered). `ndc1`/`ndc2` are the caller's
 * toVar()'d apex-relative corners, mutated in place; ndc0 (the shared apex) is
 * unchanged. Returns the accept gate — a degenerate area==0 still falls out at the
 * downstream integer `area2 > 0` test.
 */
export function orientForRaster(ndc1: NV3, ndc2: NV3, areaNdc: NF, twoSided: NB): NB {
  const flip = twoSided.and(areaNdc.lessThan(0)).toVar();
  const keep1 = vec3(ndc1).toVar(); // snapshot v1 before the in-place swap
  ndc1.assign(flip.select(ndc2, ndc1));
  ndc2.assign(flip.select(keep1, ndc2));
  return twoSided.select(areaNdc.notEqual(0), areaNdc.greaterThan(0)) as unknown as NB;
}

interface ComputeKernel {
  setName(name: string): unknown;
}

export interface NaniteRasterHandles {
  /** fullscreen resolve mesh in its own scene — render with the main camera */
  resolveScene: Scene;
  /** clear vis buffers + hw queue (every frame, before any pass) */
  clearVis(renderer: Renderer): void;
  /** SW depth-only pass (shadow cascades/clipmap; atomicMin Z) */
  depth1(renderer: Renderer): void;
  /** HW big/near-tri depth render (re-runs hwArgs; full-queue redraw is
   *  idempotent atomicMin) */
  hwDepth(renderer: Renderer, camera: PerspectiveCamera): void;
  /** NaniteView debug single-pass packed Z+id (idLo/idHi split) — SW + HW in one go */
  combined(renderer: Renderer, camera: PerspectiveCamera): void;
  /** PERF-VB4 WORLD single pass: 24-bit Z election + full-id side buffer (visBV).
   *  `?tileproto=1` routes the SW election through the B1-PROTO sort-middle tiled raster
   *  (D-N46) instead of the per-cluster scatter kernel; the HW pass is identical. */
  world1(renderer: Renderer, camera: PerspectiveCamera): void;
  /** B1-PROTO tile stats [overflow, nearcross, binEntries, hwEnqueues] (zeros if off) */
  readTileStat(renderer: Renderer): Promise<Uint32Array>;
  readHwCount(renderer: Renderer): Promise<number>;
  /** count covered/orphan pixels (NaniteView ?audit=1) */
  audit(renderer: Renderer): void;
  readAudit(renderer: Renderer): Promise<{ orphans: number; covered: number }>;
}

/** Option C vis buffers — created OUTSIDE the raster so the HZB (which the
 *  cull consumes) can view the depth buffer without a builder cycle */
export interface NaniteVisBuffers {
  depthAttr: StorageBufferAttribute;
  depthV: ReturnType<typeof sU32Views>;
  payloadAttr: StorageBufferAttribute;
  payloadV: ReturnType<typeof sU32Views>;
  /** PACKED-mode 2nd id buffer (idHi). visA(payloadV) holds (dk16<<16|idLo16), visB
   *  holds (dk16<<16|idHi16); both atomicMax'd so the nearest fragment wins BOTH and
   *  the depth+id can't desync (the race-free fix for the payload "branch-through-trunk"). */
  visBAttr: StorageBufferAttribute;
  visBV: ReturnType<typeof sU32Views>;
}

export function makeVisBuffers(pixelCount: number): NaniteVisBuffers {
  const depthAttr = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
  markFragmentWritable(depthAttr);
  const payloadAttr = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
  markFragmentWritable(payloadAttr);
  const visBAttr = new StorageBufferAttribute(new Uint32Array(pixelCount), 1);
  markFragmentWritable(visBAttr);
  return {
    depthAttr,
    depthV: sU32Views(depthAttr, pixelCount),
    payloadAttr,
    payloadV: sU32Views(payloadAttr, pixelCount),
    visBAttr,
    visBV: sU32Views(visBAttr, pixelCount),
  };
}

export function buildNaniteRaster(
  gpu: RegistryGpu,
  heightTex: Texture,
  cam: NaniteCam,
  cull: {
    qRasterRO: BufOf<UV2>;
    rasterDispatchAttr: IndirectStorageBufferAttribute;
    rasterDispatch2Attr: IndirectStorageBufferAttribute;
    rasterDispatchFullAttr: IndirectStorageBufferAttribute;
  },
  vis: NaniteVisBuffers,
  tint: 'flat' | 'cluster' | 'lod',
  /** false (?shade=0): pure matClass color, no lambert — the parity gate's
   *  shading-free mode (coverage/structure compare only) */
  shade = true,
  /** terrain micro-displacement (frame mode; dbg views omit — NaniteFetch) */
  disp?: TerrainDisp,
  /** trunk wind (frame mode) — MUST match the resolve's makeFetch so the
   *  rastered geometry and the resolve's barycentric corners agree */
  wind?: TrunkWindOpt,
  /** PACKED vis-buffer (race-free combined path): the combined kernels write the id
   *  split across visA(payloadV)+visB via depth-keyed atomicMax, and the resolve decodes
   *  it from those two. Only valid with the hier `combined()` pass. Default false keeps
   *  the legacy depthV/payloadV (two-pass + brute) encoding for the world/shadow paths. */
  packed = false,
  /** WORLD single-pass (?vb=single / PERF-VB4 D-N45): ONE SW+HW pass writes EXACT f32
   *  depth (visDepthV via atomicMin — so the HZB, shadows, and the resolve's wp
   *  reconstruction stay byte-identical + exact) AND a COHERENT id (depth16-keyed
   *  atomicMax election into visPayloadV, no branch-through-trunk) whose winner plain-
   *  stores the FULL 25-bit id into ONE side buffer (visBV). Drops the 2nd (payload)
   *  raster pass. DEVIATES from D-N45's "recompute depth in the resolve": keeping the
   *  exact depthV is cheaper (folds into pass-1's atomicMin) and leaves every depth
   *  consumer unchanged. The residual is the same artifact class — at a rare depth16
   *  tie the id can pick the bucket-neighbour (a valid wrong-material pixel, never a
   *  torn/garbage fetch). Differs from `packed` (NaniteView's 16-bit-depth combined). */
  singlePass = false,
): NaniteRasterHandles {
  const { width, height } = cam;
  // single-pass clears the id buffers like `packed` (election anchor → 0, side id →
  // 0) but keeps the exact-depth sentinel (depthV → 0xffffffff, atomicMin).
  const packedClear = packed || singlePass;
  const pixelCount = width * height;
  // PERF-3 raster ablation (?rdbg): BUILD-TIME gate — 0 / absent emits the kernel
  // UNCHANGED (production pristine, zero instrumentation). Two independent stage
  // splits live behind it, dispatched by which kernel actually runs in a config:
  //
  //  • DEPTH kernel (shadow-only): rdbg 1/2/3 attribute nanRasterDepth's per-triangle
  //    setup (resolution scaling proved it ~70% setup-bound) — the three depth sinks.
  //
  //  • WORLD1 kernel (the full-pipe camera raster, the PERF LEVER #1 measure-first
  //    gate): rdbg 4→1→2→3 are nested STOP points for the launch-vs-setup-vs-pixel
  //    split — 4 = raw launch (before makeCtx), 1 = +makeCtx/wgcache FLOOR, 2 = +the
  //    per-triangle edge-setup (skip per-pixel), 3 = full. See the four world1 sinks.
  //    FINDING (2026-06-16 LEVER #1 run): at the forest operating point the per-PIXEL
  //    coverage/election loop is ≈90% of world1; launch+makeCtx+edge-setup together
  //    are < the 8.3 ms vsync floor even at 3.3M clusters / 2× res, and pure launch is
  //    < 2.5 ns/wg (≤ ~1.2 ms at 477k cl) ⇒ triangle-granular re-dispatch (#1) is
  //    bounded-marginal and was NOT implemented. CAVEAT for re-measuring: the gutted
  //    variants run at the 120 fps rAF cap, where the c.nanRasterWorld1 GPU timestamp
  //    reads a BOGUS constant ~15.2 ms (cross-frame pipelining span, not active GPU
  //    time) — only GPU-BOUND (frameMs ≫ 8.3) readings of these variants are valid.
  const rdbg = Number(new URLSearchParams(window.location.search).get('rdbg') ?? '0');
  // PERF-3 (LOG az/ba): one workgroup == one cluster (128 threads share instId/ci),
  // but makeCtx ran PER THREAD (128×/cluster, incl. the trunk gust texture samples) =
  // 0.46 ms / 16% of nanRasterDepth, 100% redundant. Compute it ONCE (thread 0) and
  // broadcast via workgroup shared memory. DEFAULT ON (A/B-validated bit-identical —
  // f32 round-trip is exact, so the raster still agrees with the resolve's own
  // makeCtx); `?wgcache=0` opts out. Applies to the camera raster AND every shadow-
  // clipmap level + the HW vertex stage (all share buildNaniteRaster).
  const wgcache = new URLSearchParams(window.location.search).get('wgcache') !== '0';
  // ?noguard=1 — DROP the relaxed-load election guard (always atomicMax) = NAIVE FreePipe
  // scatter. Measurement only: quantifies how much of the sort-middle win world1 already
  // captures (the paper's 16–107× is vs naive FreePipe, not our guarded scatter).
  const noguard = new URLSearchParams(window.location.search).get('noguard') === '1';
  // ?f2b=1 — FRONT-TO-BACK per-pixel early-out in the world1 SCATTER raster (B3, no storage
  // tax — the tile study proved storage-tiling is bandwidth-bound at our tri count). Reads the
  // pixel's current winner BEFORE the z-interp; if the tri's nearest-possible key can't beat it
  // the fragment would lose → skip the dominant depth work. cand ≤ nearKey ⇒ loss-exact.
  const f2b = new URLSearchParams(window.location.search).get('f2b') === '1';
  // B1-PROTO (D-N46): route the WORLD single-pass SW election through the sort-middle
  // TILED raster (NaniteTileRaster) instead of the per-cluster scatter kernel. Gate only —
  // the world1 kernel below stays built + byte-pristine; only the dispatch path swaps.
  const tileproto =
    singlePass && new URLSearchParams(window.location.search).get('tileproto') === '1';
  const qRasterRO = cull.qRasterRO;
  const visDepthV = vis.depthV;
  const visPayloadV = vis.payloadV;
  const visBV = vis.visBV;
  // depth → 16-bit key, INVERTED so nearer (smaller cz) ⇒ LARGER key ⇒ wins atomicMax.
  const depthKey16 = (cz: NF): NU =>
    uint(float(1).sub(cz).mul(65535).clamp(0, 65535)) as unknown as NU;
  // PERF-VB4 world1: a 24-BIT key (8-bit id tiebreak below it). The full id is in the
  // side buffer, so the election word only needs depth + a coarse tiebreak — 24 bits of
  // depth (256× finer than the combined path's 16) makes the wp-reconstruction banding
  // sub-pixel. The resolve/shadowHalf decode cz = 1 − (key>>8)/16777215; the HZB reads
  // the top 16 bits (key>>16) which still decode as a valid coarse occluder.
  const depthKey24 = (cz: NF): NU =>
    uint(float(1).sub(cz).mul(16777215).clamp(0, 16777215)) as unknown as NU;

  // hwQueue: [0] = atomic count, then (payload, instId) pairs
  const hwQueueAttr = new StorageBufferAttribute(new Uint32Array(1 + HW_CAP * 2), 1);
  const hwQueueV = sU32Views(hwQueueAttr, 1 + HW_CAP * 2);
  const hwDrawAttr = new IndirectStorageBufferAttribute(new Uint32Array(4), 4);
  const hwDrawBuf = sU32Views(hwDrawAttr as unknown as StorageBufferAttribute, 4).rw;

  // consistency audit (?audit=1): [0] = orphans (depth written but payload
  // never matched it — ANY pass disagreement, SW or HW, shows up here),
  // [1] = covered pixels
  const auditAttr = new StorageBufferAttribute(new Uint32Array(4), 1);
  const auditV = sU32Views(auditAttr, 4);

  // ---- shared fetch helpers (NaniteFetch.ts — also the resolve's decode) ----------
  const nfetch = makeFetch(gpu, heightTex, disp, wind);
  const { makeCtx, fetchWorldVert } = nfetch;
  // PERF-3 win #2 — the cooperative vertex-transform cache lives in its own module
  // (default OFF, ?vcompact=1; measured marginal/conditional — see NaniteVertexCache).
  const vcache = makeVertexCache(gpu, nfetch);

  const edgeFn = (a: NV2, b: NV2, p: NV2): NF =>
    p.y.sub(a.y).mul(b.x.sub(a.x)).sub(p.x.sub(a.x).mul(b.y.sub(a.y))) as unknown as NF;

  // wgcache broadcast (the per-cluster makeCtx dedup) extracted as a reusable closure so the
  // B1-PROTO setup pass gets the SAME once-per-cluster ctx (per-THREAD makeCtx is ~256× the
  // wind-gust texture samples — the documented redundancy world1 fixes). Must be called with
  // a UNIFORM-across-the-workgroup (instId,ci) so all threads reach the barrier. Same field
  // serialization as world1's inline block above.
  const primeCtx = (instId2: NU, ci2: NU, localTri2: NU): VertCtx => {
    if (!wgcache) return makeCtx(instId2, ci2);
    const shU = workgroupArray('uint', 10);
    const shF = workgroupArray('float', 21);
    const setU = (i: number, v: NU): void =>
      void (shU.element(uint(i)) as unknown as { assign(x: NU): unknown }).assign(v);
    const setF = (i: number, v: NF): void =>
      void (shF.element(uint(i)) as unknown as { assign(x: NF): unknown }).assign(v);
    const getU = (i: number): NU => shU.element(uint(i)) as unknown as NU;
    const getF = (i: number): NF => shF.element(uint(i)) as unknown as NF;
    If(localTri2.equal(uint(0)), () => {
      const c = makeCtx(instId2, ci2);
      setU(0, b2u(c.isHF));
      setU(1, b2u(c.isDAG));
      setU(2, c.triStart);
      setU(3, c.triCount);
      setU(4, c.meshId);
      setU(5, c.channel);
      setU(6, c.gx);
      setU(7, c.gz);
      setU(8, c.qxw);
      setU(9, b2u(c.twoSided));
      setF(0, c.A.x as unknown as NF);
      setF(1, c.A.y as unknown as NF);
      setF(2, c.A.z as unknown as NF);
      setF(3, c.A.w as unknown as NF);
      setF(4, c.B.x as unknown as NF);
      setF(5, c.B.y as unknown as NF);
      setF(6, c.B.z as unknown as NF);
      setF(7, c.B.w as unknown as NF);
      setF(8, c.oX);
      setF(9, c.oZ);
      setF(10, c.cell);
      if (wind) {
        const w = c.wind as NonNullable<VertCtx['wind']>;
        setF(11, w.h0);
        setF(12, w.dirX);
        setF(13, w.dirY);
        setF(14, w.leanBase);
        setF(15, w.swayABase);
        setF(16, w.swayPhase);
        setF(17, w.ph);
        setF(18, w.branchBase);
        setF(19, w.flutBase);
        setF(20, w.swayXPhase);
      }
    });
    workgroupBarrier();
    const cB = vec4(getF(4).toVar(), getF(5).toVar(), getF(6).toVar(), getF(7).toVar()) as unknown as NV4;
    return {
      isHF: getU(0).toVar().equal(uint(1)),
      isDAG: getU(1).toVar().equal(uint(1)),
      A: vec4(getF(0).toVar(), getF(1).toVar(), getF(2).toVar(), getF(3).toVar()) as unknown as NV4,
      B: cB,
      yawSc: instYaw(cB),
      triStart: getU(2).toVar(),
      triCount: getU(3).toVar(),
      meshId: getU(4).toVar(),
      channel: getU(5).toVar(),
      twoSided: getU(9).toVar().equal(uint(1)),
      wind: wind
        ? {
            h0: getF(11).toVar(),
            dirX: getF(12).toVar(),
            dirY: getF(13).toVar(),
            leanBase: getF(14).toVar(),
            swayABase: getF(15).toVar(),
            swayPhase: getF(16).toVar(),
            ph: getF(17).toVar(),
            branchBase: getF(18).toVar(),
            flutBase: getF(19).toVar(),
            swayXPhase: getF(20).toVar(),
          }
        : null,
      gx: getU(6).toVar(),
      gz: getU(7).toVar(),
      qxw: getU(8).toVar(),
      oX: getF(8).toVar(),
      oZ: getF(9).toVar(),
      cell: getF(10).toVar(),
    } as unknown as VertCtx;
  };

  // B1-PROTO tiled raster (D-N46) — built only under ?tileproto; reuses the SAME makeCtx /
  // fetchWorldVert / depthKey24 / edgeFn / election buffers so the tiled candidates are
  // bit-identical to world1's. world1() routes the SW dispatch to it; HW pass unchanged.
  const tile = tileproto
    ? buildTileRaster({
        cam,
        primeCtx,
        vcache,
        depthKey24,
        edgeFn,
        visPayloadV,
        visBV,
        hwQueueV,
        HW_CAP,
        MAX_RASTER_SIZE,
        qRasterRO,
        rasterDispatchFullAttr: cull.rasterDispatchFullAttr,
        width,
        height,
        pixelCount,
      })
    : null;

  // ?wgcache bool→uint for packing isHF/isDAG into the shared-memory uint array
  const b2u = (b: NB): NU =>
    (b as unknown as { select(a: NU, c: NU): NU }).select(uint(1), uint(0));

  // ---- kVisClear ------------------------------------------------------------------
  const kVisClear = Fn(() => {
    If(instanceIndex.lessThan(uint(pixelCount)), () => {
      atomicStore(visDepthV.atomic.element(instanceIndex), uint(0xffffffff));
      // packed/single-pass: payload(+visB) are atomicMax/side targets ⇒ clear to 0 (the
      // smallest, "no fragment"). legacy: payload keeps the 0xffffffff orphan sentinel.
      atomicStore(visPayloadV.atomic.element(instanceIndex), uint(packedClear ? 0 : 0xffffffff));
      if (packedClear) atomicStore(visBV.atomic.element(instanceIndex), uint(0));
    });
    If(instanceIndex.equal(uint(0)), () => {
      atomicStore(hwQueueV.atomic.element(0), uint(0));
      atomicStore(auditV.atomic.element(0), uint(0));
      atomicStore(auditV.atomic.element(1), uint(0));
    });
  })().compute(pixelCount, [256]);
  (kVisClear as unknown as ComputeKernel).setName('nanVisClear');

  // ---- SW raster kernels (fixed-point integer scanline) ------------------------------
  // mode 'depth' = atomicMin Z only (the SHADOW depth-only path). 'combined' = single-pass
  // packed Z+id for the NaniteView debug view (16-bit depth split idLo/idHi). 'world1' =
  // the WORLD single pass (PERF-VB4): a 24-bit depth election (visPayloadV) whose winner
  // stores the full 25-bit id into the side buffer visBV; the resolve reconstructs depth
  // from the election key. No exact depthV (a 3rd hot-loop atomic buffer = a 3× cliff).
  const rasterKernel = (mode: 'depth' | 'combined' | 'world1'): unknown => {
    const kn = Fn(() => {
      const itemIdx = wgLinear(DISPATCH_ROW).toVar();
      const localTri = localX().toVar();
      const itemCount = qRasterRO.element(0).x;
      returnIf(itemIdx.greaterThanEqual(itemCount));
      const item = qRasterRO.element(itemIdx.add(uint(1)));
      const instId = item.x.toVar();
      const ci = item.y.toVar();
      if (mode === 'world1' && rdbg === 4) {
        // RAW LAUNCH floor (?rdbg=4): return BEFORE makeCtx + the wgcache broadcast —
        // isolates the pure workgroup-launch cost of the QRASTER_CAP×128 grid (one
        // wg/cluster) from makeCtx. (rdbg1 − rdbg4) = makeCtx + the wgcache barrier;
        // rdbg4 = the launch overhead the triangle-granular re-shape directly attacks
        // (it dispatches a flat ~visTris grid instead of one near-empty wg/cluster).
        // Thread-0-only sink (1 atomic/wg) consumes item so the work-item read stays.
        If(localTri.equal(uint(0)), () => {
          atomicMin(visDepthV.atomic.element(instId.add(ci).mod(uint(pixelCount))), instId);
        });
        returnIf(itemCount.greaterThanEqual(uint(0)));
      }
      let ctx: VertCtx;
      if (wgcache) {
        // Compute makeCtx ONCE (thread 0), broadcast through workgroup shared
        // memory: 9 uint + ≤20 float fields. yawSc is recomputed from the cached B
        // (cheap, deterministic). The f32 round-trip is exact, so the cached ctx is
        // bit-identical to a per-thread makeCtx ⇒ the raster still agrees with the
        // resolve. The only prior early-out (itemIdx ≥ itemCount) is UNIFORM across
        // the workgroup, so every live thread reaches the barrier (no deadlock).
        // NOTE: TrunkWindFields is serialized field-by-field here — adding a wind
        // field (e.g. swayXPhase, slot 20) MUST extend shF + both halves below.
        const shU = workgroupArray('uint', 10);
        const shF = workgroupArray('float', 21);
        // .element() is typed as a bare Node here — cast to the fluent TSL types
        const setU = (i: number, v: NU): void =>
          void (shU.element(uint(i)) as unknown as { assign(x: NU): unknown }).assign(v);
        const setF = (i: number, v: NF): void =>
          void (shF.element(uint(i)) as unknown as { assign(x: NF): unknown }).assign(v);
        const getU = (i: number): NU => shU.element(uint(i)) as unknown as NU;
        const getF = (i: number): NF => shF.element(uint(i)) as unknown as NF;
        If(localTri.equal(uint(0)), () => {
          const c = makeCtx(instId, ci);
          setU(0, b2u(c.isHF));
          setU(1, b2u(c.isDAG));
          setU(2, c.triStart);
          setU(3, c.triCount);
          setU(4, c.meshId);
          setU(5, c.channel);
          setU(6, c.gx);
          setU(7, c.gz);
          setU(8, c.qxw);
          setU(9, b2u(c.twoSided)); // N9-C2 two-sided flag (slot 9)
          setF(0, c.A.x as unknown as NF);
          setF(1, c.A.y as unknown as NF);
          setF(2, c.A.z as unknown as NF);
          setF(3, c.A.w as unknown as NF);
          setF(4, c.B.x as unknown as NF);
          setF(5, c.B.y as unknown as NF);
          setF(6, c.B.z as unknown as NF);
          setF(7, c.B.w as unknown as NF);
          setF(8, c.oX);
          setF(9, c.oZ);
          setF(10, c.cell);
          if (wind) {
            const w = c.wind as NonNullable<VertCtx['wind']>;
            setF(11, w.h0);
            setF(12, w.dirX);
            setF(13, w.dirY);
            setF(14, w.leanBase);
            setF(15, w.swayABase);
            setF(16, w.swayPhase); // hoisted sin (was natW)
            setF(17, w.ph);
            setF(18, w.branchBase);
            setF(19, w.flutBase); // N9-C0 leaf flutter
            setF(20, w.swayXPhase); // hoisted cross-sway sin
          }
        });
        workgroupBarrier();
        const cB = vec4(getF(4).toVar(), getF(5).toVar(), getF(6).toVar(), getF(7).toVar()) as unknown as NV4;
        ctx = {
          isHF: getU(0).toVar().equal(uint(1)),
          isDAG: getU(1).toVar().equal(uint(1)),
          A: vec4(getF(0).toVar(), getF(1).toVar(), getF(2).toVar(), getF(3).toVar()) as unknown as NV4,
          B: cB,
          yawSc: instYaw(cB),
          triStart: getU(2).toVar(),
          triCount: getU(3).toVar(),
          meshId: getU(4).toVar(),
          channel: getU(5).toVar(),
          twoSided: getU(9).toVar().equal(uint(1)),
          wind: wind
            ? {
                h0: getF(11).toVar(),
                dirX: getF(12).toVar(),
                dirY: getF(13).toVar(),
                leanBase: getF(14).toVar(),
                swayABase: getF(15).toVar(),
                swayPhase: getF(16).toVar(),
                ph: getF(17).toVar(),
                branchBase: getF(18).toVar(),
                flutBase: getF(19).toVar(), // N9-C0 leaf flutter
                swayXPhase: getF(20).toVar(),
              }
            : null,
          gx: getU(6).toVar(),
          gz: getU(7).toVar(),
          qxw: getU(8).toVar(),
          oX: getF(8).toVar(),
          oZ: getF(9).toVar(),
          cell: getF(10).toVar(),
        } as unknown as VertCtx;
      } else {
        ctx = makeCtx(instId, ci);
      }

      if (mode === 'depth' && rdbg === 3) {
        // ?rdbg=3 — stop right after makeCtx (the per-(instance,cluster) decode:
        // cluster/mesh/instance reads + the trunk-wind gust TEXTURE samples that
        // run once per THREAD = 128×/cluster). The sink folds the wind scalars so
        // those gust samples can't be DCE'd. (rdbg1 − rdbg3) = the 3 fetchWorldVert.
        const wSink = ctx.wind
          ? ctx.wind.leanBase.add(ctx.wind.swayPhase).add(ctx.wind.branchBase)
          : float(0);
        const sinkV = ctx.A.x.add(toF(ctx.triCount)).add(wSink).clamp(0, 1);
        const sinkPx = itemIdx.mul(uint(2654435761)).add(localTri).mod(uint(pixelCount));
        atomicMin(visDepthV.atomic.element(sinkPx), bcF2U(sinkV as unknown as NF));
        returnIf(itemCount.greaterThanEqual(uint(0)));
      }

      // PERF LEVER #1 STAGE-SPLIT (?rdbg on the WORLD single-pass kernel) — the
      // measure-first STOP gate that sizes the launch+ctx FLOOR (killable by the
      // triangle-granular re-shape) vs the per-triangle edge-setup (NOT killable —
      // same triangles either way) vs the per-pixel coverage loop. Build-time gate,
      // production pristine when rdbg==0. Numbering is world1-LOCAL (independent of
      // the depth kernel's 1/2/3):
      //   rdbg=1 → STOP right after makeCtx/wgcache broadcast = launch+ctx FLOOR
      //            (workgroup launch + barrier + the per-cluster makeCtx broadcast;
      //            skips ALL per-triangle work). The per-cluster slice of this floor
      //            is the ~4.3 ms "fixed term" candidate the re-shape can reclaim.
      //   rdbg=2 → ALSO do the per-triangle work (3× fetchWorldVert + vp transform +
      //            ndc + fixed-point snap + bbox + edge setup) but SKIP the per-pixel
      //            coverage loop + atomics (sink at the loop entry — see below).
      //   rdbg=3 → full kernel (== rdbg unset for world1).
      // (rdbg2 − rdbg1) = per-triangle edge-setup; (rdbg3 − rdbg2) = per-pixel loop.
      if (mode === 'world1' && rdbg === 1) {
        // launch+ctx FLOOR: consume the broadcast ctx so makeCtx + the wgcache
        // barrier survive DCE, then unconditionally return BEFORE any corner fetch
        // or triangle work. CRITICAL — the sink fires only on thread 0 (ONE atomic
        // per workgroup, not 128): the idle lanes (localTri ≥ triCount) do NOTHING
        // in the REAL kernel, so making all 128 lanes atomicMin here would inject
        // ~61M atomics of artificial contention that the production floor never
        // pays (the mirage trap). One atomic/workgroup matches the real coverage
        // floor while still pinning makeCtx + the wgcache barrier (all 128 lanes
        // reach the barrier, so the launch+barrier cost is fully timed). The sink
        // folds A.x + triCount + wind so the per-cluster gust samples can't be DCE'd.
        If(localTri.equal(uint(0)), () => {
          const wSink = ctx.wind
            ? ctx.wind.leanBase.add(ctx.wind.swayPhase).add(ctx.wind.branchBase)
            : float(0);
          const sinkV = ctx.A.x.add(toF(ctx.triCount)).add(wSink).clamp(0, 1);
          const sinkPx = itemIdx.mod(uint(pixelCount));
          atomicMin(visDepthV.atomic.element(sinkPx), bcF2U(sinkV as unknown as NF));
        });
        returnIf(itemCount.greaterThanEqual(uint(0)));
      }

      // PERF-3 win #2 — cooperative vertex-transform cache (own module; default off,
      // measured marginal). prime() emits the populate + barrier (or nothing when off)
      // and returns the corner fetcher; all 128 threads are live here (uniform early-out).
      const corner = vcache.prime(ctx, ci, localTri);

      If(localTri.lessThan(ctx.triCount), () => {
        const w0 = corner(localTri, 0);
        const w1 = corner(localTri, 1);
        const w2 = corner(localTri, 2);

        const p0 = cam.vp.mul(vec4(w0, 1)).toVar();
        const p1 = cam.vp.mul(vec4(w1, 1)).toVar();
        const p2 = cam.vp.mul(vec4(w2, 1)).toVar();

        if (mode === 'depth' && rdbg === 1) {
          // ?rdbg=1 — stop right after the 3 vertex fetch+transforms. The sink
          // atomicMin consumes ALL THREE clip positions BEFORE the early-out, so
          // the compiler cannot sink the fetchWorldVert/transform work past the
          // return (it would be a side-effecting use). returnIf's guard is a
          // runtime-true compare (uint ≥ 0) ⇒ always returns here, yet the
          // downstream stays REACHABLE (no unreachable-code error). Isolates:
          // work-item fetch + ctx + 3× fetchWorldVert + 3× vp transform.
          const sinkV = p0.z
            .div(p0.w.max(NEAR_EPS))
            .add(p1.z.div(p1.w.max(NEAR_EPS)))
            .add(p2.z.div(p2.w.max(NEAR_EPS)))
            .mul(1 / 3)
            .clamp(0, 1);
          const sinkPx = itemIdx.mul(uint(2654435761)).add(localTri).mod(uint(pixelCount));
          atomicMin(visDepthV.atomic.element(sinkPx), bcF2U(sinkV as unknown as NF));
          returnIf(itemCount.greaterThanEqual(uint(0)));
        }

        const payload = itemIdx.shiftLeft(uint(CLUSTER_TRI_BITS)).bitOr(localTri).toVar();

        const nearOK = p0.w
          .greaterThan(NEAR_EPS)
          .and(p1.w.greaterThan(NEAR_EPS))
          .and(p2.w.greaterThan(NEAR_EPS));

        If(nearOK.not(), () => {
          // near-plane crossing → HW path clips it (never drop, F10c)
          const slot = atomicAdd(hwQueueV.atomic.element(0), uint(1)) as unknown as NU;
          If(slot.lessThan(uint(HW_CAP)), () => {
            const base = slot.mul(uint(2)).add(uint(1));
            atomicStore(hwQueueV.atomic.element(base), payload);
            atomicStore(hwQueueV.atomic.element(base.add(uint(1))), instId);
          });
        }).Else(() => {
          const ndc0 = p0.xyz.div(p0.w).toVar();
          const ndc1 = p1.xyz.div(p1.w).toVar();
          const ndc2 = p2.xyz.div(p2.w).toVar();

          const areaNdc = edgeFn(
            ndc0.xy as unknown as NV2,
            ndc1.xy as unknown as NV2,
            ndc2.xy as unknown as NV2,
          );
          // N9-C2: front-faces pass; a two-sided (leaf) back-face is re-wound to CCW
          // in place so the positive-area core below rasters it once (orientForRaster).
          const accept = orientForRaster(ndc1 as unknown as NV3, ndc2 as unknown as NV3, areaNdc as unknown as NF, ctx.twoSided);
          If(accept, () => {
            const W = float(cam.uW);
            const H = float(cam.uH);
            const s0 = ndc0.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
            const s1 = ndc1.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
            const s2 = ndc2.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
            // f2b: the tri's NEAREST-possible election key (cand ≤ nearKey at every pixel), used
            // by the per-pixel early-out in the scanline to skip would-lose fragments' z-interp.
            const nearKey =
              mode === 'world1' && f2b
                ? depthKey24(ndc0.z.min(ndc1.z).min(ndc2.z) as unknown as NF)
                    .shiftLeft(uint(8))
                    .bitOr(uint(0xff))
                    .toVar()
                : null;

            // FIXED-POINT snap (N3a): 1/256-px integer grid — 8 subpixel bits,
            // the D3D HW convention. All coverage below is exact i32 math:
            // watertight at shared edges and bit-identical between the depth
            // and payload kernels by construction. WGSL f32→i32 SATURATES, so
            // far off-screen verts read as a huge extent → HW route.
            const xi0 = toI(s0.x.mul(256).round()).toVar();
            const yi0 = toI(s0.y.mul(256).round()).toVar();
            const xi1 = toI(s1.x.mul(256).round()).toVar();
            const yi1 = toI(s1.y.mul(256).round()).toVar();
            const xi2 = toI(s2.x.mul(256).round()).toVar();
            const yi2 = toI(s2.y.mul(256).round()).toVar();

            // whole-pixel bbox (trunc-div ≠ floor only below 0, where start
            // clamps to 0 / validBB rejects — harmless)
            const bbMinX = minI(xi0, minI(xi1, xi2)).div(toI(256)).toVar();
            const bbMaxX = maxI(xi0, maxI(xi1, xi2)).div(toI(256)).toVar();
            const bbMinY = minI(yi0, minI(yi1, yi2)).div(toI(256)).toVar();
            const bbMaxY = maxI(yi0, maxI(yi1, yi2)).div(toI(256)).toVar();

            // SW only when the UNCLAMPED extent is small (the float core sized
            // the clamped box — a screen-spanning tri with a 16-px on-screen
            // sliver ran SW with unbounded edge terms; now it routes HW).
            // Bound: deltas ≤ ~17 px ⇒ every edge term < 2^26, i32-safe.
            const smallEnough = bbMaxX
              .sub(bbMinX)
              .lessThanEqual(toI(MAX_RASTER_SIZE))
              .and(bbMaxY.sub(bbMinY).lessThanEqual(toI(MAX_RASTER_SIZE)));

            const startX = maxI(toI(0), bbMinX).toVar();
            const endX = minI(toI(width - 1), bbMaxX).toVar();
            const startY = maxI(toI(0), bbMinY).toVar();
            const endY = minI(toI(height - 1), bbMaxY).toVar();
            const validBB = startX.lessThanEqual(endX).and(startY.lessThanEqual(endY));

            If((smallEnough as unknown as { and: (o: unknown) => NB }).and(validBB), () => {
              // integer twice-area; snapping can collapse/flip a sub-1/256-px
              // sliver → skip (also guards the reciprocal — the float core
              // divided blindly)
              const area2 = yi2
                .sub(yi0)
                .mul(xi1.sub(xi0))
                .sub(xi2.sub(xi0).mul(yi1.sub(yi0)))
                .toVar();
              // SAMPLE-MISS cull (CuRast): skip a small tri whose snapped extent contains NO pixel
              // CENTRE (k·256+128 grid) on x OR y ⇒ it covers no sample ⇒ the scanline rasters
              // nothing. Conservative + exact (the scanline samples at the same centres) ⇒ zero
              // visible quality loss. Float ceil handles negative/off-screen xiMin. (Kept as
              // leverage: ~neutral on world1's already-≤1px cut, a building block / helps the tiled.)
              const xiMin = minI(xi0, minI(xi1, xi2));
              const xiMax = maxI(xi0, maxI(xi1, xi2));
              const yiMin = minI(yi0, minI(yi1, yi2));
              const yiMax = maxI(yi0, maxI(yi1, yi2));
              const firstCx = toF(xiMin.sub(toI(128)) as unknown as NI).div(256).ceil().mul(256).add(128);
              const firstCy = toF(yiMin.sub(toI(128)) as unknown as NI).div(256).ceil().mul(256).add(128);
              const coversSample = firstCx
                .lessThanEqual(toF(xiMax as unknown as NI))
                .and(firstCy.lessThanEqual(toF(yiMax as unknown as NI)));
              If(area2.greaterThan(toI(0)).and(coversSample), () => {
                // edge i is opposite vertex i; ex/ey = dE per +1 UNIT (1/256 px)
                const ex0 = yi1.sub(yi2).toVar();
                const ey0 = xi2.sub(xi1).toVar();
                const ex1 = yi2.sub(yi0).toVar();
                const ey1 = xi0.sub(xi2).toVar();
                const ex2 = yi0.sub(yi1).toVar();
                const ey2 = xi1.sub(xi0).toVar();

                // top-left rule, same orientation convention as the float core:
                // the boundary E == 0 is owned iff dE/dx < 0, or dE/dx == 0 and
                // dE/dy > 0; the -1 bias turns ≥0 into >0 on unowned edges —
                // exact and scale-free (the -1e-5 float bias competed with ulp
                // at edge-term scale ~1e6)
                const tlBias = (ex: NI, ey: NI): NI =>
                  ex
                    .lessThan(toI(0))
                    .or(ex.equal(toI(0)).and(ey.greaterThan(toI(0))))
                    .select(toI(0), toI(-1)) as unknown as NI;
                const bias0 = tlBias(ex0 as unknown as NI, ey0 as unknown as NI);
                const bias1 = tlBias(ex1 as unknown as NI, ey1 as unknown as NI);
                const bias2 = tlBias(ex2 as unknown as NI, ey2 as unknown as NI);

                // E at the center of (startX, startY), in units²
                const pcx = startX.mul(toI(256)).add(toI(128)).toVar();
                const pcy = startY.mul(toI(256)).add(toI(128)).toVar();
                const rw0 = pcy
                  .sub(yi1)
                  .mul(xi2.sub(xi1))
                  .sub(pcx.sub(xi1).mul(yi2.sub(yi1)))
                  .add(bias0)
                  .toVar();
                const rw1 = pcy
                  .sub(yi2)
                  .mul(xi0.sub(xi2))
                  .sub(pcx.sub(xi2).mul(yi0.sub(yi2)))
                  .add(bias1)
                  .toVar();
                const rw2 = pcy
                  .sub(yi0)
                  .mul(xi1.sub(xi0))
                  .sub(pcx.sub(xi0).mul(yi1.sub(yi0)))
                  .add(bias2)
                  .toVar();
                // per-PIXEL steps = ex/ey × 256 units
                const sx0 = ex0.mul(toI(256)).toVar();
                const sx1 = ex1.mul(toI(256)).toVar();
                const sx2 = ex2.mul(toI(256)).toVar();
                const sy0 = ey0.mul(toI(256)).toVar();
                const sy1 = ey1.mul(toI(256)).toVar();
                const sy2 = ey2.mul(toI(256)).toVar();
                const rcpArea = float(1).div(toF(area2 as unknown as NI)).toVar();

                if ((mode === 'depth' || mode === 'world1') && rdbg === 2) {
                  // ?rdbg=2 — stop right before the scanline loop. The sink folds
                  // the edge-setup vars (rcpArea, rw0..2, the per-pixel steps) so
                  // none are sunk past / DCE'd, forcing the full per-triangle
                  // setup to be timed. (rdbg2 − rdbg1) = near/backface + ndc +
                  // fixed-point snap + bbox + edge setup + the 3× fetchWorldVert
                  // (world1's rdbg=1 stops BEFORE any corner fetch); (rdbg3 − rdbg2)
                  // = the per-pixel loop (coverage + depth interp + the election
                  // atomicMax/atomicStore). Shared by the depth + world1 kernels —
                  // only the rdbg-gated CONFIG that's actually dispatched matters.
                  const sinkV = rcpArea
                    .add(toF(rw0 as unknown as NI))
                    .add(toF(rw1 as unknown as NI))
                    .add(toF(rw2 as unknown as NI))
                    .add(toF(sx0 as unknown as NI))
                    .add(toF(sy0 as unknown as NI))
                    .clamp(0, 1);
                  const sinkPx = uint(startY).mul(uint(cam.uW)).add(uint(startX));
                  atomicMin(visDepthV.atomic.element(sinkPx), bcF2U(sinkV as unknown as NF));
                  returnIf(itemCount.greaterThanEqual(uint(0)));
                }

                loopI('sy', startY as unknown as NI, endY as unknown as NI, (y) => {
                  const cw0 = rw0.toVar();
                  const cw1 = rw1.toVar();
                  const cw2 = rw2.toVar();
                  // PERF (UE5-gap win 1) — SCANLINE x-span. The edge value at column x is
                  // cw_i(x) = cw_i + (x−startX)·sx_i; the row is covered where all 3 are ≥0.
                  // Solve each edge for the crossing x = startX − cw_i/sx_i (÷-by-0 guarded),
                  // floor/ceil + ±1 pad so the span is a guaranteed SUPERSET (f32-safe). The
                  // EXACT per-pixel cw≥0 test below still decides coverage — this only fast-
                  // skips the ~half of the AABB that is provably outside the triangle. Bit-
                  // identical (skipped pixels all fail the test).
                  const den0 = sx0.equal(toI(0)).select(toI(1), sx0 as unknown as NI) as unknown as NI;
                  const den1 = sx1.equal(toI(0)).select(toI(1), sx1 as unknown as NI) as unknown as NI;
                  const den2 = sx2.equal(toI(0)).select(toI(1), sx2 as unknown as NI) as unknown as NI;
                  const xc0 = toF(startX).sub(toF(cw0 as unknown as NI).div(toF(den0)));
                  const xc1 = toF(startX).sub(toF(cw1 as unknown as NI).div(toF(den1)));
                  const xc2 = toF(startX).sub(toF(cw2 as unknown as NI).div(toF(den2)));
                  const lo0 = sx0.greaterThan(toI(0)).select(toI(xc0.floor().sub(float(1))), startX) as unknown as NI;
                  const lo1 = sx1.greaterThan(toI(0)).select(toI(xc1.floor().sub(float(1))), startX) as unknown as NI;
                  const lo2 = sx2.greaterThan(toI(0)).select(toI(xc2.floor().sub(float(1))), startX) as unknown as NI;
                  const hi0 = sx0.lessThan(toI(0)).select(toI(xc0.ceil().add(float(1))), endX) as unknown as NI;
                  const hi1 = sx1.lessThan(toI(0)).select(toI(xc1.ceil().add(float(1))), endX) as unknown as NI;
                  const hi2 = sx2.lessThan(toI(0)).select(toI(xc2.ceil().add(float(1))), endX) as unknown as NI;
                  // a zero-slope edge that is already negative ⇒ the whole row is empty
                  const emptyRow = sx0
                    .equal(toI(0))
                    .and(cw0.lessThan(toI(0)))
                    .or(sx1.equal(toI(0)).and(cw1.lessThan(toI(0))))
                    .or(sx2.equal(toI(0)).and(cw2.lessThan(toI(0))));
                  const xLo = maxI(maxI(maxI(startX, lo0), lo1), lo2).toVar();
                  const xHi = minI(minI(minI(endX, hi0), hi1), hi2).toVar();
                  xHi.assign(emptyRow.select(xLo.sub(toI(1)), xHi) as unknown as NI);
                  // advance the incremental edge values from startX to xLo
                  const dxL = xLo.sub(startX).toVar();
                  cw0.addAssign(dxL.mul(sx0));
                  cw1.addAssign(dxL.mul(sx1));
                  cw2.addAssign(dxL.mul(sx2));
                  loopI('sx', xLo as unknown as NI, xHi as unknown as NI, (x) => {
                    If(
                      cw0
                        .greaterThanEqual(toI(0))
                        .and(cw1.greaterThanEqual(toI(0)))
                        .and(cw2.greaterThanEqual(toI(0))),
                      () => {
                        // depth from the UNBIASED integer weights: the top-left
                        // −1 biases belong to COVERAGE only. Folding them into
                        // the weights divides by area2 while the weights sum to
                        // area2−(1..3) — a RELATIVE error of ~bias/area2 that is
                        // ulp-level on big triangles but ~5e-4 on sub-pixel far
                        // slivers (area2 ~10³ units²) ⇒ far depth biased NEARER
                        // by hundreds of meters. Self-consistent across passes
                        // (audit/parity blind) — found at N4-C0 when WATER
                        // depth-tested against the buffer. Unbiased weights sum
                        // to area2 exactly (integer identity): cz is exact, and
                        // both passes still compute identical bits.
                        const emitFrag = (): void => {
                        const uw0 = cw0.sub(bias0).toVar();
                        const uw1 = cw1.sub(bias1).toVar();
                        const uw2 = cw2.sub(bias2).toVar();
                        const cz = toF(uw0 as unknown as NI)
                          .mul(ndc0.z)
                          .add(toF(uw1 as unknown as NI).mul(ndc1.z))
                          .add(toF(uw2 as unknown as NI).mul(ndc2.z))
                          .mul(rcpArea)
                          .toVar();
                        If(cz.greaterThanEqual(0).and(cz.lessThanEqual(1)), () => {
                          const px = uint(y).mul(uint(cam.uW)).add(uint(x));
                          const bits = bcF2U(cz as unknown as NF);
                          if (mode === 'depth') {
                            const cur = aLoadU(visDepthV.atomic.element(px));
                            If(bits.lessThan(cur), () => {
                              atomicMin(visDepthV.atomic.element(px), bits);
                            });
                          } else if (mode === 'combined') {
                            // RACE-FREE packed vis buffer, NO separate depthV (stays under
                            // the storage-buffer ceiling): id packed as (depthKey16<<16 |
                            // idPart) into visA(payload)+visB via atomicMax. Depth lives in
                            // the high bits ⇒ the nearest fragment wins BOTH buffers ⇒
                            // depth+id can NEVER desync (kills "branch through trunk" for
                            // any separated depths; only truly equal-Z coplanar tris can
                            // still hybridise — rare). The HZB reads visA's key (packed).
                            const myDk = depthKey16(cz as unknown as NF).toVar();
                            const stored = aLoadU(visPayloadV.atomic.element(px));
                            If(myDk.greaterThan(stored.shiftRight(uint(16))), () => {
                              const dk = myDk.shiftLeft(uint(16));
                              atomicMax(visPayloadV.atomic.element(px), dk.bitOr(payload.bitAnd(uint(0xffff))));
                              atomicMax(visBV.atomic.element(px), dk.bitOr(payload.shiftRight(uint(16)).bitAnd(uint(0xffff))));
                            });
                          } else if (mode === 'world1') {
                            // PERF-VB4 single pass: EXACT f32 depth (atomicMin → HZB +
                            // shadows + the resolve's wp reconstruction stay exact and
                            // unchanged) AND a coherent id election. Depth16 in the high
                            // bits of an atomicMax election (visPayloadV) ⇒ the nearest
                            // fragment wins with no branch-through-trunk; the election
                            // WINNER plain-stores the FULL 25-bit id into ONE side buffer
                            // (visBV). A depth16 tie degrades to a valid-but-maybe-wrong
                            // cluster (sparse wrong-material speckle), never a torn/garbage
                            // id (the franksteining the idLo/idHi split would cause).
                            const cand = depthKey24(cz as unknown as NF)
                              .shiftLeft(uint(8))
                              .bitOr(payload.bitAnd(uint(0xff)))
                              .toVar();
                            if (noguard) {
                              // NAIVE FreePipe: unconditional atomicMax per fragment.
                              const wonE = atomicMax(visPayloadV.atomic.element(px), cand) as unknown as NU;
                              If(cand.greaterThan(wonE), () => {
                                atomicStore(visBV.atomic.element(px), payload);
                              });
                            } else {
                            const prevE = aLoadU(visPayloadV.atomic.element(px));
                            If(cand.greaterThan(prevE), () => {
                              const wonE = atomicMax(visPayloadV.atomic.element(px), cand) as unknown as NU;
                              If(cand.greaterThan(wonE), () => {
                                // election WINNER stores the FULL 25-bit id into the single side
                                // buffer (no franksteining). NO depthV write: a 3rd atomic storage
                                // buffer in this kernel is the 3× cliff (15-17 ms) AND breaks its
                                // writes — so the resolve takes the 16-bit depth from the election
                                // key (visPayloadV high bits) instead of an exact depthV.
                                atomicStore(visBV.atomic.element(px), payload);
                              });
                            });
                            }
                          }
                        });
                        };
                        // f2b (world1 only): read the pixel's current winner BEFORE the z-interp;
                        // if the tri's nearest-possible key can't beat it, the fragment would lose
                        // → skip emitFrag's z-interp + election. cand ≤ nearKey ⇒ loss-exact (zero
                        // quality loss). With near→far order, occluded overdraw skips the depth work.
                        if (mode === 'world1' && f2b && nearKey) {
                          const pxg = uint(y).mul(uint(cam.uW)).add(uint(x)).toVar();
                          const prevEg = aLoadU(visPayloadV.atomic.element(pxg)).toVar();
                          If(nearKey.greaterThan(prevEg), emitFrag);
                        } else {
                          emitFrag();
                        }
                      },
                    );
                    cw0.addAssign(sx0);
                    cw1.addAssign(sx1);
                    cw2.addAssign(sx2);
                  });
                  rw0.addAssign(sy0);
                  rw1.addAssign(sy1);
                  rw2.addAssign(sy2);
                });
              });
            }).Else(() => {
              If(validBB, () => {
                // big triangle → HW queue
                const slot = atomicAdd(hwQueueV.atomic.element(0), uint(1)) as unknown as NU;
                If(slot.lessThan(uint(HW_CAP)), () => {
                  const base = slot.mul(uint(2)).add(uint(1));
                  atomicStore(hwQueueV.atomic.element(base), payload);
                  atomicStore(hwQueueV.atomic.element(base.add(uint(1))), instId);
                });
              });
            });
          });
        });
      });
    })().compute(QRASTER_CAP * MAX_CLUSTER_TRIS, [MAX_CLUSTER_TRIS]);
    return kn;
  };

  // SHADOW depth-only pass (atomicMin Z into the shadow vis buffer).
  const kRasterDepth = rasterKernel('depth');
  (kRasterDepth as ComputeKernel).setName('nanRasterDepth');
  // NaniteView debug single-pass (packed 16-bit Z + idLo/idHi split).
  const kRasterCombined = rasterKernel('combined');
  (kRasterCombined as ComputeKernel).setName('nanRasterCombined');
  // PERF-VB4 WORLD single pass: 24-bit Z election + full-id side buffer (visBV).
  const kRasterWorld1 = rasterKernel('world1');
  (kRasterWorld1 as ComputeKernel).setName('nanRasterWorld1');

  // ---- kHwArgs ----------------------------------------------------------------------
  const kHwArgs = Fn(() => {
    const n = minU(aLoadU(hwQueueV.atomic.element(0)), uint(HW_CAP));
    hwDrawBuf.element(0).assign(n.mul(uint(3)));
    hwDrawBuf.element(1).assign(uint(1));
    hwDrawBuf.element(2).assign(uint(0));
    hwDrawBuf.element(3).assign(uint(0));
  })().compute(1, [1]);
  (kHwArgs as unknown as ComputeKernel).setName('nanHwArgs');

  // ---- kAudit (?audit=1) — run AFTER the payload passes: a covered pixel
  // whose payload is still the clear sentinel means no pass-2 writer ever
  // reproduced the stored depth (raster pass inconsistency). Gate: 0.
  const kAudit = Fn(() => {
    returnIf(instanceIndex.greaterThanEqual(uint(pixelCount)));
    const d = elemU(visDepthV.ro, instanceIndex);
    If(d.notEqual(uint(0xffffffff)), () => {
      atomicAdd(auditV.atomic.element(1), uint(1));
      const p = elemU(visPayloadV.ro, instanceIndex);
      If(p.equal(uint(0xffffffff)), () => {
        atomicAdd(auditV.atomic.element(0), uint(1));
      });
    });
  })().compute(pixelCount, [256]);
  (kAudit as unknown as ComputeKernel).setName('nanAudit');

  // ---- HW big/near-triangle passes (vertex pulling; fragment writes vis bufs) --------
  const hwGeometry = new BufferGeometry();
  hwGeometry.setAttribute('position', new Float32BufferAttribute(new Float32Array(3), 3));
  hwGeometry.setIndirect(hwDrawAttr, 0);
  hwGeometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);

  const buildHwMaterial = (pass: 'depth' | 'combined' | 'world1'): NodeMaterial => {
    const mat = new NodeMaterial();
    const vPayLo = varyingProperty('float', `nanPayLo_${pass}`) as unknown as NF;
    const vPayHi = varyingProperty('float', `nanPayHi_${pass}`) as unknown as NF;
    const vZ = varyingProperty('float', `nanZ_${pass}`) as unknown as NF;
    const vW = varyingProperty('float', `nanW_${pass}`) as unknown as NF;

    mat.vertexNode = Fn(() => {
      const triIndex = vertexIndex.div(3) as unknown as NU;
      const corner = vertexIndex.mod(3) as unknown as NU;
      const base = triIndex.mul(uint(2)).add(uint(1));
      const payload = elemU(hwQueueV.ro, base).toVar();
      const instId = elemU(hwQueueV.ro, base.add(uint(1))).toVar();
      const itemIdx = payload.shiftRight(uint(CLUSTER_TRI_BITS));
      const localTri = payload.bitAnd(uint(CLUSTER_TRI_MASK));
      const item = qRasterRO.element(itemIdx.add(uint(1)));
      const ci = item.y.toVar();
      const ctx = makeCtx(instId, ci);

      const w0 = fetchWorldVert(ctx, localTri, 0);
      const w1 = fetchWorldVert(ctx, localTri, 1);
      const w2 = fetchWorldVert(ctx, localTri, 2);
      const world = corner
        .equal(uint(1))
        .select(w1, corner.equal(uint(2)).select(w2, w0)) as unknown as NV3;
      const clip = cam.vp.mul(vec4(world, 1)).toVar();

      (vPayLo as unknown as { assign: (v: unknown) => void }).assign(toF(payload.bitAnd(uint(0xffff))));
      (vPayHi as unknown as { assign: (v: unknown) => void }).assign(toF(payload.shiftRight(uint(16))));
      (vZ as unknown as { assign: (v: unknown) => void }).assign(clip.z);
      (vW as unknown as { assign: (v: unknown) => void }).assign(clip.w);
      return clip;
    })() as unknown as typeof mat.vertexNode;

    mat.fragmentNode = Fn(() => {
      const z = vZ.div(vW).toVar();
      const pay = uint(vPayLo.round()).bitOr(uint(vPayHi.round()).shiftLeft(uint(16))).toVar();
      const fy = float(cam.uH).sub(screenCoordinate.y);
      const px = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
      If(z.greaterThanEqual(0).and(z.lessThanEqual(1)), () => {
        const bits = bcF2U(z as unknown as NF);
        if (pass === 'depth') {
          atomicMin(visDepthV.atomic.element(px), bits);
        } else if (pass === 'combined') {
          // RACE-FREE packed (mirrors the SW combined path): id packed by depth-keyed
          // atomicMax into visA(payload)+visB, no separate depthV.
          const myDk = depthKey16(z as unknown as NF).toVar();
          const stored = aLoadU(visPayloadV.atomic.element(px));
          If(myDk.greaterThan(stored.shiftRight(uint(16))), () => {
            const dk = myDk.shiftLeft(uint(16));
            atomicMax(visPayloadV.atomic.element(px), dk.bitOr(pay.bitAnd(uint(0xffff))));
            atomicMax(visBV.atomic.element(px), dk.bitOr(pay.shiftRight(uint(16)).bitAnd(uint(0xffff))));
          });
        } else if (pass === 'world1') {
          // PERF-VB4 single-pass WORLD (mirrors the SW world1 path): a depth16-keyed
          // atomicMax election whose WINNER write-stores the full id + its exact depth.
          const cand = depthKey24(z as unknown as NF)
            .shiftLeft(uint(8))
            .bitOr(pay.bitAnd(uint(0xff)))
            .toVar();
          // Mirror the SW world1 election's relaxed-load guard (the SW path at the
          // .Else()-branch above has had this; the HW vertex-pull path was missing
          // it — it RMW-contended visPayloadV on EVERY covered fragment). Bit-identical:
          // atomicMax is monotone, so skipping it when cand<=prevE leaves the buffer
          // value unchanged, and the store is already gated by cand>wonE which is false
          // whenever cand<=prevE (the buffer only grows). The only effect is dropping
          // the RMW for fragments behind the current front — the bulk of HW overdraw.
          const prevE = aLoadU(visPayloadV.atomic.element(px));
          If(cand.greaterThan(prevE), () => {
            const wonE = atomicMax(visPayloadV.atomic.element(px), cand) as unknown as NU;
            If(cand.greaterThan(wonE), () => {
              atomicStore(visBV.atomic.element(px), pay);
            });
          });
        }
      });
      return vec4(0, 0, 0, 0);
    })() as unknown as typeof mat.fragmentNode;
    mat.depthTest = false;
    mat.depthWrite = false;
    mat.colorWrite = false;
    mat.fog = false;
    mat.lights = false;
    // N9-C2: don't HW back-face-cull. Two-sided leaf tris that cross the near plane
    // or oversize the i32 SW path land here with a single winding (the reversed dup
    // is gone), so culling would hole them. Safe for opaque solids: the nearer face
    // wins atomicMin and only the depth-match writes payload, so back-faces never win.
    mat.side = DoubleSide;
    return mat;
  };

  const hwDepthMat = buildHwMaterial('depth');
  const hwCombinedMat = buildHwMaterial('combined');
  const hwWorld1Mat = buildHwMaterial('world1');
  const hwScene = new Scene();
  const hwMesh = new Mesh(hwGeometry, hwDepthMat);
  hwMesh.frustumCulled = false;
  hwScene.add(hwMesh);
  const hwRT = new RenderTarget(width, height, { depthBuffer: false });
  hwRT.texture.name = 'nanHwPass';

  // ---- flat resolve -----------------------------------------------------------------
  const resolveGeometry = new BufferGeometry();
  resolveGeometry.setAttribute(
    'position',
    new Float32BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
  );
  resolveGeometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);

  const resolveMat = new NodeMaterial();
  resolveMat.vertexNode = vec4(positionGeometry.xy, 0, 1) as unknown as typeof resolveMat.vertexNode;
  resolveMat.fragmentNode = Fn(() => {
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    let itemIdx: NU;
    let localTri: NU;
    if (packed) {
      // PACKED (combined): id is split across visA(payload)+visB, depth in the high
      // bits. visA==0 ⇒ atomicMax never ran ⇒ empty pixel. (depthV is NOT read here —
      // keeps the resolve at 2 vis buffers under the 10-storage Metal ceiling.)
      const aRaw = elemU(visPayloadV.ro, pixelIndex);
      const bRaw = elemU(visBV.ro, pixelIndex);
      If(aRaw.equal(uint(0)), () => {
        Discard();
      });
      const id = bRaw.bitAnd(uint(0xffff)).shiftLeft(uint(16)).bitOr(aRaw.bitAnd(uint(0xffff)));
      itemIdx = id.shiftRight(uint(CLUSTER_TRI_BITS)) as unknown as NU;
      localTri = id.bitAnd(uint(CLUSTER_TRI_MASK)) as unknown as NU;
    } else {
      const dRaw = elemU(visDepthV.ro, pixelIndex);
      const pRaw = elemU(visPayloadV.ro, pixelIndex);
      If(dRaw.equal(uint(0xffffffff)), () => {
        Discard();
      });
      // orphan (pass-2 never matched pass-1's depth): background, not a garbage
      // payload decode — black-pixel probes then see it as a hole
      If(pRaw.equal(uint(0xffffffff)), () => {
        Discard();
      });
      itemIdx = pRaw.shiftRight(uint(CLUSTER_TRI_BITS)) as unknown as NU;
      localTri = pRaw.bitAnd(uint(CLUSTER_TRI_MASK)) as unknown as NU;
    }
    const item = qRasterRO.element(itemIdx.add(uint(1)));
    const instId = item.x.toVar();
    const ci = item.y.toVar();
    const ctx = makeCtx(instId, ci);

    let lambert: NF;
    if (shade) {
      const w0 = fetchWorldVert(ctx, localTri, 0);
      const w1 = fetchWorldVert(ctx, localTri, 1);
      const w2 = fetchWorldVert(ctx, localTri, 2);
      const faceN = normalize(
        cross(w1.sub(w0) as unknown as NV3, w2.sub(w0) as unknown as NV3),
      ) as unknown as NV3;
      const L = normalize(vec3(0.55, 0.8, 0.25)) as unknown as NV3;
      lambert = max(dot(faceN, L), 0).mul(0.85).add(0.18) as unknown as NF;
    } else {
      lambert = float(1) as unknown as NF;
    }

    // matClass palette (flat) — terrain/rock/bark/deadwood/leaf/grass/debris
    const matClass = elemU(gpu.meshes, ctx.meshId.mul(uint(MESH_WORDS)).add(uint(6)))
      .shiftRight(uint(8))
      .bitAnd(uint(0xff))
      .toVar();
    const isT = matClass.equal(uint(0));
    const isR = matClass.equal(uint(1));
    const isB = matClass.equal(uint(2));
    const isD = matClass.equal(uint(3));
    const albedo = isT
      .select(
        vec3(0.3, 0.36, 0.22),
        isR.select(
          vec3(0.42, 0.41, 0.4),
          isB.select(vec3(0.36, 0.27, 0.19), isD.select(vec3(0.33, 0.28, 0.22), vec3(0.35, 0.33, 0.3))),
        ),
      )
      .toVar();
    let col: NV3;
    if (tint === 'cluster') {
      col = hashColor(ci).mul(lambert) as unknown as NV3;
    } else if (tint === 'lod') {
      // color by the cluster's LOD LEVEL (packed in cluster word7 bits 10-15) — a
      // per-level hue so the cut's distance falloff reads as colour bands (fine→coarse).
      // The hue is a NON-wrapping red→violet sweep: level/MAXLVL maps to t∈[0,4.8] rad
      // (< 2π), so the rainbow never completes a cycle and high levels can't loop back
      // to salmon (level 0). Leaf DAGs reach ~14 levels; MAXLVL=16 keeps them in range.
      const lvl = elemU(gpu.clusters, ci.mul(uint(CLUSTER_WORDS)).add(uint(7)))
        .shiftRight(uint(10))
        .bitAnd(uint(0x3f));
      const t = toF(lvl).div(16).min(1).mul(4.8);
      const c3 = vec3(
        t.cos().mul(0.5).add(0.5),
        t.add(2.094).cos().mul(0.5).add(0.5),
        t.add(4.188).cos().mul(0.5).add(0.5),
      ) as unknown as NV3;
      col = c3.mul(lambert) as unknown as NV3;
    } else {
      col = (albedo as unknown as NV3).mul(lambert) as unknown as NV3;
    }
    return vec4(col, 1);
  })() as unknown as typeof resolveMat.fragmentNode;
  resolveMat.depthNode = Fn(() => {
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    if (packed) {
      // reconstruct cz from visA's 16-bit depth key (dk = (1−cz)·65535). Approximate
      // (16-bit) — exact enough for the isolated debug view; depthV stays out of the
      // resolve's binding set. (A future world-mode packed path can recompute exact
      // cz from the reconstructed triangle instead.)
      const dk = toF(elemU(visPayloadV.ro, pixelIndex).shiftRight(uint(16)));
      return float(1).sub(dk.div(65535)) as unknown as typeof resolveMat.depthNode;
    }
    return bcU2F(elemU(visDepthV.ro, pixelIndex));
  })() as unknown as typeof resolveMat.depthNode;
  resolveMat.depthTest = false;
  resolveMat.depthWrite = true;
  resolveMat.fog = false;
  resolveMat.lights = false;

  const resolveMesh = new Mesh(resolveGeometry, resolveMat);
  resolveMesh.frustumCulled = false;
  const resolveScene = new Scene();
  resolveScene.add(resolveMesh);

  // ---- per-frame passes --------------------------------------------------------------
  const clearVis = (renderer: Renderer): void => {
    dispatch(renderer, kVisClear);
  };
  const depth1 = (renderer: Renderer): void => {
    dispatchIndirect(renderer, kRasterDepth, cull.rasterDispatchAttr);
  };
  const hwRender = (renderer: Renderer, camera: PerspectiveCamera, mat: NodeMaterial): void => {
    const prevRT = renderer.getRenderTarget();
    renderer.setRenderTarget(hwRT);
    hwMesh.material = mat;
    renderer.render(hwScene, camera);
    renderer.setRenderTarget(prevRT);
  };
  const hwDepth = (renderer: Renderer, camera: PerspectiveCamera): void => {
    dispatch(renderer, kHwArgs);
    hwRender(renderer, camera, hwDepthMat);
  };
  // single-pass Z+payload over the full set (hier path): replaces depth1 + the late
  // hwDepth + payload re-raster with ONE SW pass + ONE HW pass.
  const combined = (renderer: Renderer, camera: PerspectiveCamera): void => {
    dispatchIndirect(renderer, kRasterCombined, cull.rasterDispatchFullAttr);
    dispatch(renderer, kHwArgs); // SW pass filled hwQueue → build the indirect draw args
    hwRender(renderer, camera, hwCombinedMat);
  };
  // PERF-VB4 (D-N45) WORLD single pass over the full set (replaced depth1 → hwDepth →
  // payload). ONE SW + ONE HW pass: a 24-bit depth election (visPayloadV) whose winner
  // stores the full id into visBV; the resolve/shadows/HZB read depth from the election
  // key (no exact depthV — a 3rd hot-loop atomic buffer was measured a 3× cliff).
  const world1 = (renderer: Renderer, camera: PerspectiveCamera): void => {
    if (tile) {
      // B1-PROTO: clearBins → bin clusters → tiled SW election (workgroup-memory, flush
      // to global). The HW pass is identical (the tiled kernel fills hwQueue the same way).
      tile.dispatchTiled(renderer);
    } else {
      dispatchIndirect(renderer, kRasterWorld1, cull.rasterDispatchFullAttr);
    }
    dispatch(renderer, kHwArgs); // SW pass filled hwQueue → build the indirect draw args
    hwRender(renderer, camera, hwWorld1Mat);
  };

  const readHwCount = async (renderer: Renderer): Promise<number> => {
    const buf = await readBuffer(renderer, hwQueueAttr, 0, 4);
    return new Uint32Array(buf)[0] ?? 0;
  };
  const audit = (renderer: Renderer): void => {
    dispatch(renderer, kAudit);
  };
  const readAudit = async (renderer: Renderer): Promise<{ orphans: number; covered: number }> => {
    const buf = await readBuffer(renderer, auditAttr, 0, 8);
    const u = new Uint32Array(buf);
    return { orphans: u[0] ?? 0, covered: u[1] ?? 0 };
  };

  const readTileStat = async (renderer: Renderer): Promise<Uint32Array> =>
    tile ? tile.readStat(renderer) : new Uint32Array(4);

  return {
    resolveScene,
    clearVis,
    depth1,
    hwDepth,
    combined,
    world1,
    readTileStat,
    readHwCount,
    audit,
    readAudit,
  };
}
