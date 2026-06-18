/**
 * B1-PROTO — SORT-MIDDLE TILED raster (SPEC D-N46). Gated `?tileproto=1`; `world1` stays pristine.
 *
 * STATUS (2026-06-18): CORRECT — renders bit-equivalently to `world1` and tracks its occlusion cut
 * exactly (both oscillate the same ~42k on the dense forest = the PRE-EXISTING occlusion-feedback
 * instability, NOT a tiled bug). The tiled-specific cluster FLICKER was a real bug, found by the
 * `find-tiled-flicker` workflow + fixed: the bounded-batching xtri local index used
 * `batchBase·MAX_CLUSTER_TRIS` (255) while the payload packs with stride `1<<CLUSTER_TRI_BITS`
 * (256) — an OOB write past the xtri region into the HW queue, stomping near/big-leaf HW tris →
 * HZB holes → cut oscillation. UNIQUELY a 256-cap defect (commit 5092074). Fixed: index by
 * TRI_STRIDE = 1<<CLUSTER_TRI_BITS everywhere (sizing + write + read).
 *
 * PERF vs scatter: OPEN — an earlier "≈2.8× slower" reading was an UNOPTIMIZED impl (naive 7-word
 * round-trip + a 20-wave dispatch) at a degenerate cam, NOT the proper-impl ceiling; re-measure
 * honestly (measureActiveGpu, worst cam) with the fix in. Atomics-free still holds (the on-chip
 * election isn't the lever); the front-to-back early-out (`?f2b=1` in world1) is also a real lever.
 *
 * Goal: match `world1` (zero quality loss) and beat it on the dense foliage worst case.
 *
 * Architecture (CudaRaster / Laine&Karras 2011 — the canonical design for ~1px-tri
 * vegetation; brief: docs/perf-runs/prior-art/laine2011-cudaraster.md, T5/T1/T7):
 *
 *   kSetup (TriangleSetup, 1 thread : 1 triangle, transform ONCE):
 *     makeCtx + 3× fetchWorldVert + vp transform + ndc + back-face/two-sided + fixed-point
 *     snap — IDENTICAL to world1 up to the snap. Then EMIT instead of raster:
 *       • near-plane-crossing OR big (bbox > MAX_RASTER_SIZE) → HW queue (once, directly —
 *         each triangle is processed exactly once here, so no anchor/home-tile dedup).
 *       • small → append its screen-space SETUP (6 snapped coords + 3 ndc.z + payload) to a
 *         flat xtri buffer via one atomic cursor, then scatter the xtri index into every
 *         16×16 tile its bbox overlaps.
 *     THE FIX vs the 188 ms cluster-rebin: geometry is transformed ONCE here, never again.
 *
 *   kRasterTiled (FineRaster, 1 workgroup : 1 tile, election on-chip):
 *     reads each binned triangle's pre-computed setup from xtri (NO makeCtx, NO transform,
 *     NO geometry fetch — T4), redoes only the cheap INTEGER edge setup, runs world1's exact
 *     integer scanline CLIPPED to the tile, and elects into a `var<workgroup>
 *     array<atomic<u32>>` depth|id word. The global per-fragment `atomicLoad`+compare (the
 *     60% cost) becomes an on-chip shared read; the winner's global atomic collapses to one
 *     plain store per tile pixel at flush.
 *
 * IDENTITY: the election is order-independent `atomicMax` on world1's verbatim packed
 * `(depthKey24<<8 | id8)` word; coverage/depth math is world1's integer core unchanged;
 * binning is a conservative superset (sphere screen-bbox, nearest-point radius). So the same
 * fragment wins each pixel regardless of tiling. The id side-store keeps world1's SAME benign
 * equal-depth race. Bar: no holes / no new speckle (per the perf methodology — bit-identity
 * isn't screenshot-provable given wind + the accepted race).
 *
 * PROTO SIMPLIFICATIONS (perf/scale, not correctness): fixed per-tile capacity + overflow
 * tripwire (vs compacted prefix-sum); xtri sized for the world view (forest 36M needs tile
 * BATCHING — CudaRaster §6.2); kSetup has no wgcache yet (makeCtx per triangle); thread=
 * triangle scanline in the tile (not yet T2 coverage-mask / T6 density-flatten). These are
 * the optimization ladder, applied after the base is correct + measured.
 */
import { IndirectStorageBufferAttribute, StorageBufferAttribute } from 'three/webgpu';
import type { Renderer } from 'three/webgpu';
import {
  Fn,
  If,
  atomicAdd,
  atomicMax,
  atomicStore,
  float,
  instanceIndex,
  int,
  uint,
  uniform,
  vec2,
  vec4,
} from 'three/tsl';
import type { NB, NF, NI, NU, NV2, NV3 } from '../gpu/TSLTypes';
import { CLUSTER_TRI_BITS, MAX_CLUSTER_TRIS } from './GeometryRegistry';
import { DISPATCH_ROW } from './NaniteCommon';
import type { NaniteCam } from './NaniteCommon';
import type { VertCtx } from './NaniteFetch';
import type { VertexCache } from './NaniteVertexCache';
import { orientForRaster } from './NaniteRaster';
import {
  aLoadU,
  bcF2U,
  bcU2F,
  dispatch,
  elemU,
  localX,
  loopI,
  loopU,
  maxI,
  minI,
  minU,
  readBuffer,
  sU32Views,
  toF,
  toI,
  wgLinear,
} from './Tsl';
import type { AtomicBuf, BufOf, UV2 } from './Tsl';

const TILE = 16; // px per tile edge
const WG = 64; // raster workgroup threads (thread-per-triangle over the tile's binned list)
const TILE_CAP = 8192; // max triangles binned per tile PER BATCH (overflow → tripwire stat)
// xtri entry (7 words, LOSSLESS pack — tris are ≤MAX_RASTER_SIZE px per instminpx/minpx, so
// verts fit 16-bit offsets from the bbox-min): w0 = (bbMinX+BIAS)<<16 | (bbMinY+BIAS); w1..3
// = (dx<<16 | dy) per vert in 1/256 units relative to bbMin·256; w4..6 = ndc.z (f32 bits).
// 36→28 bytes/tri. payload (itemIdx·256 + localTri) IS the xtri index — no append cursor.
const XTRI_WORDS = 7;
const COORD_BIAS = 256; // keeps bbMin (∈[-16,~2592]) non-negative for 16-bit packing
// BOUNDED BATCHING (device-portable): the cut is processed in N_BATCHES waves of BATCH_CLUSTERS
// clusters each, so the transformed-triangle (xtri) buffer is FIXED-SIZE regardless of cut size
// — fits any GPU, not just unified-memory Macs. Each wave elects into the PERSISTENT global
// visBuffer (atomicMax is FREE per guard==naive). xtri = BATCH·256·7·4 ≈ 57 MB. batchBase (a
// uniform) offsets the wave's clusters so the local xtri index = payload − batchBase·256.
const BATCH_CLUSTERS = 32_000;
// max cut clusters covered = N_BATCHES × BATCH_CLUSTERS. The forest cut OVER-EMITS (HZB occlusion
// ~0% on holey foliage): measured up to 388k, and it WOBBLES frame-to-frame, so any cut beyond
// the cap drops a varying set ⇒ FLICKER. Cover the worst with headroom (640k ⇒ 20 waves). Clusters
// past this are still dropped, but the cull's cut never approaches it ⇒ no in-view flicker.
const MAX_CUT_CLUSTERS = 640_000;
const N_BATCHES = Math.ceil(MAX_CUT_CLUSTERS / BATCH_CLUSTERS);
const NEAR_EPS = 1e-4;

export interface TileRasterDeps {
  cam: NaniteCam;
  /** wgcache'd per-cluster ctx (once-per-cluster makeCtx broadcast). Call uniformly across
   *  the workgroup (itemIdx identical for all threads) so all reach the broadcast barrier. */
  primeCtx: (instId: NU, ci: NU, localTri: NU) => VertCtx;
  /** vertex-transform-once-per-cluster cache (Bevy/vcompact) — measured 1.8× on the
   *  wind-heavy forest. prime() once per cluster (all threads live) → corner fetcher. */
  vcache: VertexCache;
  depthKey24: (cz: NF) => NU;
  edgeFn: (a: NV2, b: NV2, p: NV2) => NF;
  visPayloadV: { atomic: AtomicBuf };
  visBV: { atomic: AtomicBuf };
  hwQueueV: { atomic: AtomicBuf };
  HW_CAP: number;
  MAX_RASTER_SIZE: number;
  qRasterRO: BufOf<UV2>;
  /** indirect dispatch args sized to the actual cut (world1 uses this) — kSetup shares it
   *  so it launches ~count workgroups, not the full QRASTER_CAP×MAX_CLUSTER_TRIS grid. */
  rasterDispatchFullAttr: IndirectStorageBufferAttribute;
  width: number;
  height: number;
  pixelCount: number;
}

export interface TileRasterHandles {
  dispatchTiled: (renderer: Renderer) => void;
  /** [overflowTri, droppedXtri, binEntries, smallTris] — overflow/dropped MUST be 0 */
  readStat: (renderer: Renderer) => Promise<Uint32Array>;
  tilesX: number;
  tilesY: number;
  nTiles: number;
}

export function buildTileRaster(deps: TileRasterDeps): TileRasterHandles {
  const {
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
    width,
    height,
  } = deps;

  const tilesX = Math.ceil(width / TILE);
  const tilesY = Math.ceil(height / TILE);
  const nTiles = tilesX * tilesY;

  // ---- buffers -----------------------------------------------------------------------
  // Split by ACCESS TYPE (the 34 ms lesson): bulk setup/list writes have a UNIQUE owner per
  // slot → PLAIN stores (cheap); only the cursor/counts/stats truly contend → atomicAdd.
  // Mixing atomic+rw views of one buffer in a kernel is illegal aliasing, AND atomicStore
  // for the ~120M unique-owner writes WAS the 34 ms — so two buffers, one per access type.
  //
  // atomicBuf (atomicAdd targets — the only true contention): [0,nTiles) per-tile counts ·
  //   [STAT_BASE+0..1] overflow stats · [HW_CURSOR] HW-queue append cursor.
  const STAT_BASE = nTiles;
  const HW_CURSOR = nTiles + 2;
  const atomicWords = nTiles + 3;
  const atomicBufAttr = new StorageBufferAttribute(new Uint32Array(atomicWords), 1);
  const atomicBuf = sU32Views(atomicBufAttr, atomicWords);
  // dataBuf (PLAIN stores, unique owner per slot): [tile·TILE_CAP + slot] tile lists ·
  //   [XENTRY + payload·XTRI_WORDS] xtri entry · [HW_BASE + slot·2] HW (payload,instId) pairs.
  //   HW routing lives HERE (not hwQueueV) so the setup binds 10 storage buffers, not 11
  //   (vcompact's gpu.vcompact needs the slot); a tiny copy pass moves it to hwQueueV.
  const XENTRY = nTiles * TILE_CAP;
  // xtri sized to ONE batch (not the whole cut) — this is the device-portable bound.
  // payload packs localTri with stride 2^CLUSTER_TRI_BITS (line below), NOT MAX_CLUSTER_TRIS —
  // at the 256 cap those DIFFER (256 vs 255). The xtri local index MUST use the payload stride
  // or it runs past this region into the HW queue (OOB → stomped near/big tris → HZB holes →
  // occlusion-cut oscillation; the bug commit 5092074's 128→256 cap exposed).
  const TRI_STRIDE = 1 << CLUSTER_TRI_BITS;
  const TRI_SLOTS = BATCH_CLUSTERS * TRI_STRIDE;
  const HW_BASE = XENTRY + TRI_SLOTS * XTRI_WORDS;
  const dataWords = HW_BASE + HW_CAP * 2;
  const dataBufAttr = new StorageBufferAttribute(new Uint32Array(dataWords), 1);
  const dataBuf = sU32Views(dataBufAttr, dataWords);

  // per-wave cluster base: itemIdx = batchBase + workgroup; local xtri index = payload−base·256.
  const batchBaseU = uniform(0, 'uint' as 'float'); // settable handle (.value) per batch
  const batchBase = batchBaseU as unknown as NU; // uint-typed node for TSL math

  // ---- kClearBins: full clear (counts + stats + HW cursor), once before the batch loop ----
  const kClearBins = Fn(() => {
    const i = instanceIndex.toVar();
    If(i.lessThan(uint(atomicWords)), () => {
      atomicStore(atomicBuf.atomic.element(i), uint(0)); // counts + cursor + stats
    });
  })().compute(atomicWords, [256]);
  (kClearBins as { setName(n: string): unknown }).setName('nanTileClearBins');

  // ---- kClearCounts: clear ONLY the per-tile counts [0,nTiles), per batch (HW cursor + the
  // already-appended HW survive so big tris accumulate across waves; the global visBuffer too).
  const kClearCounts = Fn(() => {
    const i = instanceIndex.toVar();
    If(i.lessThan(uint(nTiles)), () => {
      atomicStore(atomicBuf.atomic.element(i), uint(0));
    });
  })().compute(nTiles, [256]);
  (kClearCounts as { setName(n: string): unknown }).setName('nanTileClearCounts');

  // ---- kSetup (TriangleSetup, transform ONCE) ----------------------------------------
  const kSetup = Fn(() => {
    // wave-local workgroup → global cluster index. Dispatched BATCH_CLUSTERS workgroups/wave.
    const itemIdx = batchBase.add(wgLinear(DISPATCH_ROW)).toVar();
    const localTri = localX().toVar();
    const itemCount = qRasterRO.element(0).x;
    If(itemIdx.lessThan(itemCount), () => {
      const item = qRasterRO.element(itemIdx.add(uint(1)));
      const instId = item.x.toVar();
      const ci = item.y.toVar();
      const ctx = primeCtx(instId, ci, localTri); // wgcache: makeCtx once per cluster
      // vertex-once-per-cluster (Bevy/vcompact): transform each unique vert ONCE into shared,
      // then triangles read corners from there. Must be primed with all threads live (uniform).
      const corner = vcache.prime(ctx, ci, localTri);
      If(localTri.lessThan(ctx.triCount), () => {
        const w0 = corner(localTri, 0);
        const w1 = corner(localTri, 1);
        const w2 = corner(localTri, 2);
        const p0 = cam.vp.mul(vec4(w0, 1)).toVar();
        const p1 = cam.vp.mul(vec4(w1, 1)).toVar();
        const p2 = cam.vp.mul(vec4(w2, 1)).toVar();
        const payload = itemIdx.shiftLeft(uint(CLUSTER_TRI_BITS)).bitOr(localTri).toVar();
        const nearOK = p0.w
          .greaterThan(float(NEAR_EPS))
          .and(p1.w.greaterThan(float(NEAR_EPS)))
          .and(p2.w.greaterThan(float(NEAR_EPS)));
        const enqueueHw = (): void => {
          // append (payload, instId) into dataBuf's HW region (cursor in atomicBuf). kCopyHw
          // moves it to hwQueueV afterwards — keeps the setup off hwQueueV (10-buffer limit).
          const slot = atomicAdd(atomicBuf.atomic.element(uint(HW_CURSOR)), uint(1)) as unknown as NU;
          If(slot.lessThan(uint(HW_CAP)), () => {
            const base = uint(HW_BASE).add(slot.mul(uint(2)));
            (dataBuf.rw.element(base) as unknown as { assign(x: NU): void }).assign(payload);
            (dataBuf.rw.element(base.add(uint(1))) as unknown as { assign(x: NU): void }).assign(instId);
          });
        };
        If(nearOK.not(), () => {
          enqueueHw();
        }).Else(() => {
          const ndc0 = p0.xyz.div(p0.w).toVar();
          const ndc1 = p1.xyz.div(p1.w).toVar();
          const ndc2 = p2.xyz.div(p2.w).toVar();
          const areaNdc = edgeFn(
            ndc0.xy as unknown as NV2,
            ndc1.xy as unknown as NV2,
            ndc2.xy as unknown as NV2,
          );
          const accept = orientForRaster(
            ndc1 as unknown as NV3,
            ndc2 as unknown as NV3,
            areaNdc as unknown as NF,
            ctx.twoSided,
          );
          If(accept, () => {
            const W = float(cam.uW);
            const H = float(cam.uH);
            const s0 = ndc0.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
            const s1 = ndc1.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
            const s2 = ndc2.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
            const xi0 = toI(s0.x.mul(256).round()).toVar();
            const yi0 = toI(s0.y.mul(256).round()).toVar();
            const xi1 = toI(s1.x.mul(256).round()).toVar();
            const yi1 = toI(s1.y.mul(256).round()).toVar();
            const xi2 = toI(s2.x.mul(256).round()).toVar();
            const yi2 = toI(s2.y.mul(256).round()).toVar();
            const bbMinX = minI(xi0, minI(xi1, xi2)).div(toI(256)).toVar();
            const bbMaxX = maxI(xi0, maxI(xi1, xi2)).div(toI(256)).toVar();
            const bbMinY = minI(yi0, minI(yi1, yi2)).div(toI(256)).toVar();
            const bbMaxY = maxI(yi0, maxI(yi1, yi2)).div(toI(256)).toVar();
            const smallEnough = bbMaxX
              .sub(bbMinX)
              .lessThanEqual(toI(MAX_RASTER_SIZE))
              .and(bbMaxY.sub(bbMinY).lessThanEqual(toI(MAX_RASTER_SIZE)));
            const startX = maxI(toI(0), bbMinX).toVar();
            const endX = minI(toI(width - 1), bbMaxX).toVar();
            const startY = maxI(toI(0), bbMinY).toVar();
            const endY = minI(toI(height - 1), bbMaxY).toVar();
            const validBB = startX.lessThanEqual(endX).and(startY.lessThanEqual(endY));
            // area2 > 0: snapping can collapse/flip a sub-1/256-px sliver → it produces NO
            // fragments in the raster (which gates on the same test). Cull it HERE so its
            // whole emit (9 stores + bin scatter) + raster visit is skipped. A degenerate
            // SMALL tri is dropped (NOT routed to HW — matches world1).
            const area2 = yi2.sub(yi0).mul(xi1.sub(xi0)).sub(xi2.sub(xi0).mul(yi1.sub(yi0)));
            // SAMPLE-MISS cull (CuRast :190-196, 27-40%): a tri whose snapped extent covers no
            // pixel CENTRE (k·256+128 grid) on x OR y produces zero fragments → skip before any
            // emit/raster. Conservative + exact (the scanline samples at the same centres).
            const xiMin = minI(xi0, minI(xi1, xi2));
            const xiMax = maxI(xi0, maxI(xi1, xi2));
            const yiMin = minI(yi0, minI(yi1, yi2));
            const yiMax = maxI(yi0, maxI(yi1, yi2));
            const firstCx = toF(xiMin.sub(toI(128)) as unknown as NI).div(256).ceil().mul(256).add(128);
            const firstCy = toF(yiMin.sub(toI(128)) as unknown as NI).div(256).ceil().mul(256).add(128);
            const coversSample = firstCx
              .lessThanEqual(toF(xiMax as unknown as NI))
              .and(firstCy.lessThanEqual(toF(yiMax as unknown as NI)));
            If((smallEnough as unknown as { and: (o: unknown) => NB }).and(validBB), () => {
             If(area2.greaterThan(toI(0)).and(coversSample), () => {
              // SMALL triangle: xtri slot = batch-LOCAL payload (payload − batchBase·256, a
              // UNIQUE natural index within the wave) → plain store, NO append cursor. Then
              // scatter the GLOBAL payload into each overlapped tile's list (the resolve id).
              If(itemIdx.sub(batchBase).lessThan(uint(BATCH_CLUSTERS)), () => {
                const localPay = payload.sub(batchBase.shiftLeft(uint(CLUSTER_TRI_BITS)));
                const b = uint(XENTRY).add(localPay.mul(uint(XTRI_WORDS)));
                const put = (o: number, v: NU): void =>
                  void (dataBuf.rw.element(b.add(uint(o))) as unknown as { assign(x: NU): void }).assign(v);
                // LOSSLESS 16-bit pack: bbMin = floor(min(xi)/256) px; each vert = (xi −
                // bbMin·256) ∈ [0, ~17·256) fits u16. reconstruct exact 1/256 coords on read.
                const u16 = (i: NI): NU => uint(i as unknown as never).bitAnd(uint(0xffff)) as unknown as NU;
                const pack2 = (hi: NI, lo: NI): NU => u16(hi).shiftLeft(uint(16)).bitOr(u16(lo));
                const bbX = toI(toF(xiMin as unknown as NI).div(256).floor()).toVar();
                const bbY = toI(toF(yiMin as unknown as NI).div(256).floor()).toVar();
                put(0, pack2(bbX.add(toI(COORD_BIAS)) as unknown as NI, bbY.add(toI(COORD_BIAS)) as unknown as NI));
                put(1, pack2(xi0.sub(bbX.mul(toI(256))) as unknown as NI, yi0.sub(bbY.mul(toI(256))) as unknown as NI));
                put(2, pack2(xi1.sub(bbX.mul(toI(256))) as unknown as NI, yi1.sub(bbY.mul(toI(256))) as unknown as NI));
                put(3, pack2(xi2.sub(bbX.mul(toI(256))) as unknown as NI, yi2.sub(bbY.mul(toI(256))) as unknown as NI));
                put(4, bcF2U(ndc0.z as unknown as NF));
                put(5, bcF2U(ndc1.z as unknown as NF));
                put(6, bcF2U(ndc2.z as unknown as NF));
                // scatter payload to tiles the bbox overlaps (clamped to screen → tiles)
                const tMinX = maxI(toI(0), startX.div(toI(TILE)));
                const tMaxX = minI(toI(tilesX - 1), endX.div(toI(TILE)));
                const tMinY = maxI(toI(0), startY.div(toI(TILE)));
                const tMaxY = minI(toI(tilesY - 1), endY.div(toI(TILE)));
                loopI('ty', tMinY as unknown as NI, tMaxY as unknown as NI, (ty) => {
                  loopI('tx', tMinX as unknown as NI, tMaxX as unknown as NI, (tx) => {
                    const tileIdx = ty.mul(toI(tilesX)).add(tx);
                    const slot = atomicAdd(
                      atomicBuf.atomic.element(uint(tileIdx)),
                      uint(1),
                    ) as unknown as NU;
                    If(slot.lessThan(uint(TILE_CAP)), () => {
                      const base = uint(tileIdx).mul(uint(TILE_CAP)).add(slot);
                      (dataBuf.rw.element(base) as unknown as { assign(x: NU): void }).assign(payload);
                    }).Else(() => {
                      atomicAdd(atomicBuf.atomic.element(uint(STAT_BASE)), uint(1)); // tile ovf
                    });
                  });
                });
              }).Else(() => {
                atomicAdd(atomicBuf.atomic.element(uint(STAT_BASE + 1)), uint(1)); // cluster ovf
              });
             }); // close If(area2 > 0) — a degenerate SMALL tri is dropped (not HW)
            }).Else(() => {
              If(validBB, () => {
                enqueueHw(); // big triangle → HW
              });
            });
          });
        });
      });
    });
  })().compute(BATCH_CLUSTERS * MAX_CLUSTER_TRIS, [MAX_CLUSTER_TRIS]);
  (kSetup as { setName(n: string): unknown }).setName('nanTileSetup');

  // ---- kRasterTiled (FineRaster, election into GLOBAL visBuffer) ----------------------
  // One workgroup per tile; thread owns binned triangles [local, local+WG, …]. Elects into
  // the SAME global visPayloadV/visBV as world1 — atomicMax is FREE (proven: guarded==naive),
  // so the on-chip shared election + flush bought nothing and blocked bounded batching; this
  // is bit-identical to world1's election. The tiling's win is B3 (front-to-back per-pixel
  // early-out below), NOT the atomic. Across bounded batches the global buffer accumulates.
  const kRasterTiled = Fn(() => {
    const tileIdx = wgLinear(DISPATCH_ROW).toVar();
    const local = localX().toVar();
    const tileX = tileIdx.mod(uint(tilesX));
    const tileY = tileIdx.div(uint(tilesX));
    const px0 = tileX.mul(uint(TILE)).toVar();
    const py0 = tileY.mul(uint(TILE)).toVar();

    const count = minU(elemU(atomicBuf.ro, tileIdx), uint(TILE_CAP));
    const tileX0 = int(px0) as unknown as NI;
    const tileY0 = int(py0) as unknown as NI;
    const tileX1 = minI(tileX0.add(toI(TILE - 1)), toI(width - 1));
    const tileY1 = minI(tileY0.add(toI(TILE - 1)), toI(height - 1));

    loopU(uint(0), uint(Math.ceil(TILE_CAP / WG)), (r) => {
      const slot = local.add(r.mul(uint(WG))).toVar();
      If(slot.lessThan(count), () => {
        const payload = elemU(dataBuf.ro, tileIdx.mul(uint(TILE_CAP)).add(slot)).toVar();
        const b = uint(XENTRY).add(payload.sub(batchBase.shiftLeft(uint(CLUSTER_TRI_BITS))).mul(uint(XTRI_WORDS)));
        // unpack the 7-word LOSSLESS layout (see kSetup): bbMin px + 3 (dx|dy) verts + 3 z.
        const w0 = elemU(dataBuf.ro, b).toVar();
        const bbX = (int(w0.shiftRight(uint(16))) as unknown as NI).sub(toI(COORD_BIAS)).toVar();
        const bbY = (int(w0.bitAnd(uint(0xffff))) as unknown as NI).sub(toI(COORD_BIAS)).toVar();
        const ux = (w: NU): NI => int(w.shiftRight(uint(16))) as unknown as NI;
        const uy = (w: NU): NI => int(w.bitAnd(uint(0xffff))) as unknown as NI;
        const w1 = elemU(dataBuf.ro, b.add(uint(1))).toVar();
        const w2 = elemU(dataBuf.ro, b.add(uint(2))).toVar();
        const w3 = elemU(dataBuf.ro, b.add(uint(3))).toVar();
        const xi0 = bbX.mul(toI(256)).add(ux(w1)).toVar();
        const yi0 = bbY.mul(toI(256)).add(uy(w1)).toVar();
        const xi1 = bbX.mul(toI(256)).add(ux(w2)).toVar();
        const yi1 = bbY.mul(toI(256)).add(uy(w2)).toVar();
        const xi2 = bbX.mul(toI(256)).add(ux(w3)).toVar();
        const yi2 = bbY.mul(toI(256)).add(uy(w3)).toVar();
        const z0 = bcU2F(elemU(dataBuf.ro, b.add(uint(4)))).toVar();
        const z1 = bcU2F(elemU(dataBuf.ro, b.add(uint(5)))).toVar();
        const z2 = bcU2F(elemU(dataBuf.ro, b.add(uint(6)))).toVar();
        const nearKey = depthKey24(z0.min(z1).min(z2) as unknown as NF).shiftLeft(uint(8)).bitOr(uint(0xff));

        const startX = maxI(maxI(toI(0), minI(xi0, minI(xi1, xi2)).div(toI(256))), tileX0).toVar();
        const endX = minI(minI(toI(width - 1), maxI(xi0, maxI(xi1, xi2)).div(toI(256))), tileX1).toVar();
        const startY = maxI(maxI(toI(0), minI(yi0, minI(yi1, yi2)).div(toI(256))), tileY0).toVar();
        const endY = minI(minI(toI(height - 1), maxI(yi0, maxI(yi1, yi2)).div(toI(256))), tileY1).toVar();
        const validBB = startX.lessThanEqual(endX).and(startY.lessThanEqual(endY));
        If(validBB, () => {
          const area2 = yi2.sub(yi0).mul(xi1.sub(xi0)).sub(xi2.sub(xi0).mul(yi1.sub(yi0))).toVar();
          If(area2.greaterThan(toI(0)), () => {
            const ex0 = yi1.sub(yi2).toVar();
            const ey0 = xi2.sub(xi1).toVar();
            const ex1 = yi2.sub(yi0).toVar();
            const ey1 = xi0.sub(xi2).toVar();
            const ex2 = yi0.sub(yi1).toVar();
            const ey2 = xi1.sub(xi0).toVar();
            const tlBias = (ex: NI, ey: NI): NI =>
              ex
                .lessThan(toI(0))
                .or(ex.equal(toI(0)).and(ey.greaterThan(toI(0))))
                .select(toI(0), toI(-1)) as unknown as NI;
            const bias0 = tlBias(ex0 as unknown as NI, ey0 as unknown as NI);
            const bias1 = tlBias(ex1 as unknown as NI, ey1 as unknown as NI);
            const bias2 = tlBias(ex2 as unknown as NI, ey2 as unknown as NI);
            const pcx = startX.mul(toI(256)).add(toI(128)).toVar();
            const pcy = startY.mul(toI(256)).add(toI(128)).toVar();
            const rw0 = pcy.sub(yi1).mul(xi2.sub(xi1)).sub(pcx.sub(xi1).mul(yi2.sub(yi1))).add(bias0).toVar();
            const rw1 = pcy.sub(yi2).mul(xi0.sub(xi2)).sub(pcx.sub(xi2).mul(yi0.sub(yi2))).add(bias1).toVar();
            const rw2 = pcy.sub(yi0).mul(xi1.sub(xi0)).sub(pcx.sub(xi0).mul(yi1.sub(yi0))).add(bias2).toVar();
            const sx0 = ex0.mul(toI(256)).toVar();
            const sx1 = ex1.mul(toI(256)).toVar();
            const sx2 = ex2.mul(toI(256)).toVar();
            const sy0 = ey0.mul(toI(256)).toVar();
            const sy1 = ey1.mul(toI(256)).toVar();
            const sy2 = ey2.mul(toI(256)).toVar();
            const rcpArea = float(1).div(toF(area2 as unknown as NI)).toVar();
            loopI('sy', startY as unknown as NI, endY as unknown as NI, (y) => {
              const cw0 = rw0.toVar();
              const cw1 = rw1.toVar();
              const cw2 = rw2.toVar();
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
              const emptyRow = sx0
                .equal(toI(0))
                .and(cw0.lessThan(toI(0)))
                .or(sx1.equal(toI(0)).and(cw1.lessThan(toI(0))))
                .or(sx2.equal(toI(0)).and(cw2.lessThan(toI(0))));
              const xLo = maxI(maxI(maxI(startX, lo0), lo1), lo2).toVar();
              const xHi = minI(minI(minI(endX, hi0), hi1), hi2).toVar();
              xHi.assign(emptyRow.select(xLo.sub(toI(1)), xHi) as unknown as NI);
              const dxL = xLo.sub(startX).toVar();
              cw0.addAssign(dxL.mul(sx0));
              cw1.addAssign(dxL.mul(sx1));
              cw2.addAssign(dxL.mul(sx2));
              loopI('sx', xLo as unknown as NI, xHi as unknown as NI, (x) => {
                If(
                  cw0.greaterThanEqual(toI(0)).and(cw1.greaterThanEqual(toI(0))).and(cw2.greaterThanEqual(toI(0))),
                  () => {
                    // FRONT-TO-BACK per-pixel early-out (B3, EXACT, no all-pairs cost). nearKey =
                    // the tri's NEAREST-possible election key; cand ≤ nearKey at EVERY pixel. So
                    // if nearKey can't beat this pixel's current winner, this fragment would lose
                    // → skip the dominant z-interp + depthKey + election entirely. Read the winner
                    // BEFORE the z-interp. With near→far tile order, occluded fragments (the 11-19
                    // of ~20 overdraw) skip the depth work. Loss-exact ⇒ zero quality loss.
                    const px = uint(y).mul(uint(width)).add(uint(x)).toVar();
                    const prevE = aLoadU(visPayloadV.atomic.element(px)).toVar();
                    If(nearKey.greaterThan(prevE), () => {
                      const uw0 = cw0.sub(bias0).toVar();
                      const uw1 = cw1.sub(bias1).toVar();
                      const uw2 = cw2.sub(bias2).toVar();
                      const cz = toF(uw0 as unknown as NI)
                        .mul(z0)
                        .add(toF(uw1 as unknown as NI).mul(z1))
                        .add(toF(uw2 as unknown as NI).mul(z2))
                        .mul(rcpArea)
                        .toVar();
                      If(cz.greaterThanEqual(0).and(cz.lessThanEqual(1)), () => {
                        const cand = depthKey24(cz as unknown as NF)
                          .shiftLeft(uint(8))
                          .bitOr(payload.bitAnd(uint(0xff)))
                          .toVar();
                        If(cand.greaterThan(prevE), () => {
                          const wonE = atomicMax(visPayloadV.atomic.element(px), cand) as unknown as NU;
                          If(cand.greaterThan(wonE), () => {
                            atomicStore(visBV.atomic.element(px), payload);
                          });
                        });
                      });
                    });
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
        });
      });
    });
  })().compute(nTiles * WG, [WG]);
  (kRasterTiled as { setName(n: string): unknown }).setName('nanRasterTiled');

  // ---- kCopyHw: move the setup's HW candidates (dataBuf HW region) → hwQueueV -----------
  const kCopyHw = Fn(() => {
    const i = instanceIndex.toVar();
    const n = minU(elemU(atomicBuf.ro, uint(HW_CURSOR)), uint(HW_CAP));
    If(i.equal(uint(0)), () => {
      atomicStore(hwQueueV.atomic.element(uint(0)), n); // the indirect draw count
    });
    If(i.lessThan(n), () => {
      const src = uint(HW_BASE).add(i.mul(uint(2)));
      const dst = i.mul(uint(2)).add(uint(1));
      atomicStore(hwQueueV.atomic.element(dst), elemU(dataBuf.ro, src));
      atomicStore(hwQueueV.atomic.element(dst.add(uint(1))), elemU(dataBuf.ro, src.add(uint(1))));
    });
  })().compute(HW_CAP, [256]);
  (kCopyHw as { setName(n: string): unknown }).setName('nanTileCopyHw');

  const dispatchTiled = (renderer: Renderer): void => {
    // BOUNDED BATCHING: process the cut in N_BATCHES waves of BATCH_CLUSTERS clusters, each
    // into the FIXED-SIZE xtri buffer → device-portable regardless of cut size. The global
    // visBuffer (cleared once by NaniteFrame before this) accumulates across waves; HW (big
    // tris) appends across waves (HW cursor survives kClearCounts) → one kCopyHw at the end.
    dispatch(renderer, kClearBins); // full clear once (counts + stats + HW cursor)
    for (let i = 0; i < N_BATCHES; i++) {
      batchBaseU.value = i * BATCH_CLUSTERS;
      if (i > 0) dispatch(renderer, kClearCounts); // counts only; HW + visBuffer persist
      dispatch(renderer, kSetup); // BATCH_CLUSTERS workgroups, clusters [base, base+BATCH)
      dispatch(renderer, kRasterTiled); // elect this wave's bins into the global visBuffer
    }
    dispatch(renderer, kCopyHw); // accumulated HW region → hwQueueV (kHwArgs/hwRender)
  };

  const readStat = async (renderer: Renderer): Promise<Uint32Array> => {
    const buf = await readBuffer(renderer, atomicBufAttr, STAT_BASE * 4, 16);
    return new Uint32Array(buf);
  };

  return { dispatchTiled, readStat, tilesX, tilesY, nTiles };
}
