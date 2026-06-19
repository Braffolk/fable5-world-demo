/**
 * NaniteVoxelRaster — the voxel-brick raster. TWO interchangeable backends behind
 * ?voxraster= (the SCAR fix; spec §6.0):
 *
 *   ?voxraster=scatter (DEFAULT — kVoxScatter, ONE kernel): each voxel BLOCK directly
 *     atomicMax-elects its footprint into the GLOBAL visPayloadV/visBV — EXACTLY like
 *     world1 rasterizes triangles (NaniteRaster.ts:937-959). NO per-tile bin, NO K
 *     near→far passes, NO on-chip wgElect/flush-merge. The rigorous sweep proved the
 *     bin's SETUP FLOOR — not the leaf-overdraw it replaces — is what net-LOSES at
 *     every config (the SCAR §6.0): it is the SAME bin/K-pass mechanism that was
 *     REFUTED + DELETED for triangles (eecf046). For small opaque sub-pixel blocks the
 *     per-pixel overlap is low, so direct scatter is cheap and the bin's early-skip is
 *     unnecessary overhead. The atomicMax handles depth (nearest block wins, order-free
 *     + loss-exact, exactly as world1's unordered triangle scatter). A coarse PER-BLOCK
 *     OCCLUSION CULL (?voxoccl=1, one global depth read at the projected centre) replaces
 *     the bin's per-pixel early-skip for the deep-canopy occluded blocks — one read per
 *     block, not per footprint pixel.
 *
 *   ?voxraster=bin (A/B FALLBACK — kVoxBin + kRasterVox): the DEPTH-BUCKETED two-kernel
 *     binned path below, kept for comparison. PORTED from the deleted sort-middle TILED
 *     triangle prototype (`git show dfe6518:src/nanite/NaniteTileRaster.ts`, commits
 *     b037f4d/43e4aa1/fed6771) with BRICKS (block clusters) as the binned primitive
 *     instead of triangles. Two kernels, mirroring the tileproto kSetup→kRasterTiled
 *     split, with buffers split by ACCESS TYPE (the 34 ms lesson):
 *
 *   kVoxBin (§6.2 — one thread per VOXEL CLUSTER work-item from qVoxRaster):
 *     project the cluster's BLOCK AABB through cam.vp → screen bbox + nearest NDC z →
 *     bucket = top KBITS of depthKey24(nearestZ) → scatter a 1-word bucket-tagged entry
 *     (payload = the qVoxRaster item index) into every 16×16 tile the bbox overlaps,
 *     via an atomicAdd cursor into a flat per-tile list. atomicBuf(counts)/dataBuf(list).
 *
 *   kRasterVox (§6.3 — one workgroup per 16×16 tile, on-chip election):
 *     PRE-SEED wgElect from the GLOBAL visPayloadV (so near-field TRIANGLE winners
 *     occlude far bricks across the tri/voxel tier boundary). Then K near→far passes
 *     (bk = 7..0) over the single flat list, processing only entries with eBucket==bk.
 *     Per covered pixel: read prevE BEFORE the per-pixel z-interp; if the block's
 *     nearestZ key can't beat prevE → SKIP the whole election (occlusion skip). Else
 *     derive voxCz (AABB-slab front depth) + elect into the on-chip wgElect/wgId via
 *     atomicMax. After all passes, FLUSH-MERGE wgElect/wgId into global visPayloadV/
 *     visBV via atomicMax (merge, never overwrite — preserves nearer near-field tris).
 *
 * SPATIAL UNIT = the CLUSTER (≤128-brick BLOCK), not the individual brick (§A1 blocking
 * decision): the brick record has NO per-brick position, and the cluster sphere (word0-3)
 * bounds the whole block AABB. So kVoxBin projects the block, and the election id is the
 * qVoxRaster item index (brick range recovered in the resolve via the cluster's word6/7).
 * A block covers ONE or SEVERAL pixels — coarse, NOT a 1-write/pixel model (§6.1).
 *
 * BIT BUDGET (§4.5): the election WORD is `depthKey24(24) | id8`; the full id is plain-
 * stored into visBV by the winner. The voxel id carries bit31=1 (the only bit free at
 * both the 128- and 256-tri caps) + the qVoxRaster item index in the low ≤30 bits.
 *
 * ≤10 STORAGE BUFFERS per kernel (the hard constraint): kVoxBin drops tri-only
 * verts/indices/hfVerts and binds clusters/meshes/instances/voxelBricks/qVoxRaster +
 * atomicBuf/dataBuf = 7; kRasterVox binds voxelBricks/qVoxRaster/atomicBuf/dataBuf/
 * visPayloadV/visBV = 6 (split atomicBuf/dataBuf like the tileproto).
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
  vec3,
  vec4,
  workgroupArray,
  workgroupBarrier,
} from 'three/tsl';
import type { NF, NI, NU, NV3, NV4 } from '../gpu/TSLTypes';
import { CLUSTER_WORDS } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import { BRICK_ALBEDO, BRICK_HALF, BRICK_POS_X, BRICK_WORDS, MAX_BRICKS_PER_CLUSTER } from './VoxelBrick';
import { DISPATCH_ROW, QVOX_CAP } from './NaniteCommon';
import type { NaniteCam } from './NaniteCommon';
import { instTransformPoint, instYaw, instSphereRadius } from './NaniteCommon';
import {
  aLoadU,
  bcU2F,
  dispatch,
  dispatchIndirect,
  elemU,
  elemUW,
  loopI,
  loopU,
  localX,
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

const TILE = 16; // px per tile edge (matches the tileproto)
const WG = 64; // raster workgroup threads
// FLAT per-tile list cap (the tileproto's original, fine overflow behaviour: a dropped
// block is simply not rasterized — NOT mis-elected). A voxel band has FAR fewer
// primitives per tile than the 388k-cluster triangle cut, so this is generous.
const FLAT_TILE_CAP = 4096;
// FRONT-TO-BACK depth buckets (§6.2, loss-exact). entry = payload | (bucket<<BUCKET_SHIFT);
// bucket = top KBITS of depthKey24(nearest NDC z). LARGER depthKey = NEARER = HIGHER bucket.
const K_BUCKETS = 8;
const KBITS = 3; // log2(K_BUCKETS) — top KBITS of the 24-bit depthKey select the bucket
const BUCKET_SHIFT = 28; // bucket field bit position; low 28 bits = payload
const PAYLOAD_MASK = (1 << BUCKET_SHIFT) - 1;
const NEAR_EPS = 1e-4;
// voxel namespace bit (§4.5): the only bit free at BOTH the 128- and 256-tri caps.
const VOX_BIT = 0x80000000;
// DENSITY-MODULATED COVERAGE (§3 Risk #1 / §4.3 word4.A) — the bloat-to-blob fix.
// A coarse low-density block (conifer crown density ~0.16-0.26) must paint only a
// ~density FRACTION of its projected footprint so (1-density) sees THROUGH to the
// bricks/background behind ⇒ the band reads as sparse FOLIAGE, not a solid slab.
// Each kept pixel stays OPAQUE + depth-correct (the election is unchanged); we just
// DROP (1-density) of the footprint pixels via a STABLE per-(pixel,block) dither.
//   COVER_FLOOR — a block never vanishes (keeps a minimum see-through-but-present
//                 coverage even for the sparsest brick), so silhouettes survive.
//   COVER_CEIL  — even a dense block stays a touch see-through (real leaves are never
//                 a perfect wall), which also bounds the added overdraw from the win.
const COVER_FLOOR = 0.1;
const COVER_CEIL = 0.92;

// PER-THREAD FOOTPRINT CAP (the close-up pathology fix, RC2 / missed-cause-6).
// One-thread-per-brick still makes each lane O(footprint-area). A single brick
// straddling the near plane (forcevox=all → nearDist=0, camera in the crown) can
// project to extreme screen coords and clamp its bbox to the FULL framebuffer
// (2268×1473 ≈ 3.3M px) ⇒ ONE lane loops the whole screen ⇒ the ~500 ms whole-OS
// stall. UE's invariant: every SW-rasterized primitive is provably small. We apply
// it to the RASTER itself: clamp each thread's footprint to a hard pixel-extent box
// (BRICK_MAX_EXT per side) AROUND the projected brick CENTRE. A correct foliage brick
// at any sane on-screen size is a few px; a brick that projects larger than this is
// either degenerate (near-plane straddle) or a debug-only (forcevox) up-close voxel
// that should be a triangle, not a screen-filling slab. Clamping the FOOTPRINT (not
// the election) keeps every painted pixel depth-correct + loss-exact within the box;
// it only DROPS the pathological skirt that no real foliage brick has. The cap is
// generous (a real near brick is ≪ this) so it never clips a legitimate footprint.
const BRICK_MAX_EXT = 64; // max px per side a single brick thread may scan
// NDC magnitude past which a projected AABB corner is treated as a near-plane BLOWUP (its tiny
// positive w makes ndc = p/w shoot off-screen). Such a corner has no trustworthy 2D position and
// is excluded from the brick's screen bbox (it would otherwise smear the footprint into empty sky
// — the iter-1 monolith columns). 3 = three screens past centre: looser than any real on-screen
// foliage brick, so it never reclassifies a legitimately on-screen corner.
const NDC_EXPLODE = 3;

// BUILD-TIME BIT-BUDGET ASSERT (§6.2): the max payload (a qVoxRaster item index, < QVOX_CAP)
// MUST fit below BUCKET_SHIFT so the bucket field never collides with payload bits, and the
// bucket field itself must fit in a u32 above BUCKET_SHIFT. QVOX_CAP=2M ≪ 1<<28 (268M).
{
  if (QVOX_CAP - 1 >= 1 << BUCKET_SHIFT) {
    throw new Error(
      `NaniteVoxelRaster: payload cap ${QVOX_CAP - 1} >= 1<<BUCKET_SHIFT (${1 << BUCKET_SHIFT}) — bucket collides with payload`,
    );
  }
  if (BUCKET_SHIFT + KBITS > 32) {
    throw new Error(`NaniteVoxelRaster: bucket field overflows u32 (BUCKET_SHIFT ${BUCKET_SHIFT} + KBITS ${KBITS} > 32)`);
  }
}

export interface VoxelRasterDeps {
  gpu: RegistryGpu;
  cam: NaniteCam;
  /** voxel-raster work queue (cull §A1): [0]=(count,0); items 1.. = (instId, ci). */
  qVoxRasterRO: BufOf<UV2>;
  /** 2D-split dispatch args over the voxel-cluster count (cull.voxRasterDispatchAttr). */
  voxRasterDispatchAttr: IndirectStorageBufferAttribute;
  /** the SAME 24-bit depth key the world1 raster + resolve use (§6.7). */
  depthKey24: (cz: NF) => NU;
  /** global election buffers (the on-chip wgElect flushes here via atomicMax). */
  visPayloadV: { atomic: AtomicBuf };
  visBV: { atomic: AtomicBuf };
  width: number;
  height: number;
}

export interface VoxelRasterHandles {
  /** dispatch kVoxBin then kRasterVox (call AFTER world1+hwRender so the on-chip
   *  election can pre-seed from the global near-field triangle winners, §6.6). */
  dispatchVoxel: (renderer: Renderer) => void;
  /** [0] = per-pixel BRICK-WRITE counter (the Stage-2 overdraw-overlay number, §A2):
   *  one increment per wgElect win — the occlusion-skip should drive this FAR below the
   *  overlapping triangle fragments. Read AFTER dispatchVoxel. */
  readWriteCount: (renderer: Renderer) => Promise<number>;
  tilesX: number;
  tilesY: number;
  nTiles: number;
}

export function buildNaniteVoxelRaster(deps: VoxelRasterDeps): VoxelRasterHandles {
  const { gpu, cam, qVoxRasterRO, voxRasterDispatchAttr, depthKey24, visPayloadV, visBV, width, height } = deps;

  // ?voxdither=0|1 — the PERF-QUALITY TENSION knob (Stage-3b-perf, spec §3/§6.0/§6.4).
  //   0 = OPAQUE bricks (DEFAULT): every covered pixel that survives the occlusion skip is
  //       elected OPAQUE — NO coverHash, NO per-block density buffer read (both are pure
  //       overhead here). This is the CHEAP-AND-CORRECT path: a sub-pixel (≤1-2 px) opaque
  //       brick is not visibly blocky, and because it OCCLUDES, the front-to-back early-skip
  //       eliminates everything behind it ⇒ the occlusion-collapse that makes voxels net-win.
  //       The lever to make it look correct is SMALL bricks (finer ?voxgrid= / farther
  //       ?voxnear=), NOT see-through dither.
  //   1 = the Stage-3b density-modulated DITHER: a stable per-(pixel,block) coverHash drops
  //       (1-density) of the footprint so the band reads as SPARSE see-through foliage. Looks
  //       correct when bricks are COARSE/near, but see-through ⇒ no occlusion ⇒ the election
  //       explodes (the +18.7 ms regression the task is resolving). Kept as the A/B control.
  // The coverHash + density read are emitted ONLY in dither mode (build-time gate); and in
  // dither mode the hash now runs AFTER the occlusion skip (cheap reorder — occluded pixels
  // never pay the hash). DEFAULT OPAQUE so the sweep measures the cheap path first.
  const voxDither = new URLSearchParams(window.location.search).get('voxdither') === '1';

  // ?voxraster=scatter|bin — THE PER-BRICK-COST LEVER (the task; spec §6.0 THE SCAR).
  //   scatter (DEFAULT): each voxel block DIRECTLY atomicMax-elects its footprint into the
  //     GLOBAL visPayloadV/visBV — EXACTLY like world1 rasterizes triangles (NaniteRaster.ts
  //     :947-960). NO kVoxBin per-tile binning, NO kRasterVox K near→far passes, NO on-chip
  //     wgElect/flush-merge. The rigorous sweep proved the bin's SETUP FLOOR (the SCAR) — not
  //     the leaf-overdraw it replaces — is what makes the bin path net-LOSE at every config
  //     (+4.8..+14.1 ms vs the 41.3 ms triangle baseline). This is precisely the bin/K-pass
  //     mechanism that was REFUTED + DELETED for triangles (eecf046, +11.7/+18.1 ms). The
  //     production triangle raster that WON is SCATTER (world1). So the principled fix is a
  //     SCATTER voxel raster: for small opaque sub-pixel blocks the per-pixel overlap is low,
  //     so direct scatter is cheap and the bin's early-skip machinery is unnecessary overhead.
  //     The atomicMax handles depth (nearest block wins). A coarse PER-BLOCK OCCLUSION CULL
  //     (one global depth read at the projected centre) replaces the bin's per-pixel early-skip
  //     for the deep-canopy occluded blocks — one read per block, not per footprint pixel.
  //   bin: the prior code (kVoxBin + kRasterVox) kept as the A/B fallback behind the flag.
  const voxRasterMode = new URLSearchParams(window.location.search).get('voxraster') ?? 'scatter';
  const useScatter = voxRasterMode !== 'bin';

  const tilesX = Math.ceil(width / TILE);
  const tilesY = Math.ceil(height / TILE);
  const nTiles = tilesX * tilesY;

  // ---- buffers (split by ACCESS TYPE, tileproto §6.6) -------------------------------
  // atomicBuf (atomicAdd targets — the only true contention): [0,nTiles) per-tile flat-
  // list counts · [WRITE_CTR] the debug brick-write counter (Stage-2 overlay, §A2).
  const COUNTS = nTiles;
  const WRITE_CTR = COUNTS;
  const atomicWords = COUNTS + 1;
  const atomicBufAttr = new StorageBufferAttribute(new Uint32Array(atomicWords), 1);
  const atomicBuf = sU32Views(atomicBufAttr, atomicWords);
  // dataBuf (PLAIN stores, unique owner per slot): [tile·FLAT_TILE_CAP + slot] per-tile
  // flat list, each entry = payload | bucket<<BUCKET_SHIFT.
  const dataWords = nTiles * FLAT_TILE_CAP;
  const dataBufAttr = new StorageBufferAttribute(new Uint32Array(Math.max(1, dataWords)), 1);
  const dataBuf = sU32Views(dataBufAttr, Math.max(1, dataWords));

  // ---- kClearBins: clear per-tile counts + the write counter, before kVoxBin ----------
  const kClearBins = Fn(() => {
    const i = instanceIndex.toVar();
    If(i.lessThan(uint(atomicWords)), () => {
      atomicStore(atomicBuf.atomic.element(i), uint(0));
    });
  })().compute(atomicWords, [256]);
  (kClearBins as { setName(n: string): unknown }).setName('nanVoxClearBins');

  // workgroup-array plain store (wgId is a non-atomic workgroupArray).
  const wgSet = (arr: ReturnType<typeof workgroupArray>, i: NU, v: NU): void => {
    (arr.element(i) as unknown as { assign(x: NU): void }).assign(v);
  };

  // STABLE per-(pixel,block) dither in [0,1) for the density-coverage gate (above).
  // PCG-style integer hash (mirror of NaniteCommon.hashColor's mixer) keyed on the
  // pixel position AND the block payload, so a given footprint pixel of a given block
  // always lands the SAME side of the density threshold → ZERO temporal shimmer under a
  // moving camera (the dither pattern is locked to screen-space, intentionally — TAA
  // sees a steady stipple, not crawling noise; a temporal jitter is a deliberate
  // future option, NOT added here so the default is rock-stable, §Risk #1).
  const coverHash = (px: NU, py: NU, salt: NU): NF => {
    const a = px
      .mul(uint(0x9e3779b9))
      .add(py.mul(uint(0x85ebca77)))
      .add(salt.mul(uint(0xc2b2ae3d)))
      .add(uint(0x27d4eb2f))
      .toVar();
    const b = a.shiftRight(uint(15)).bitXor(a).mul(uint(0x2c1b3c6d)).toVar();
    const c = b.shiftRight(uint(12)).bitXor(b).mul(uint(0x297a2d39)).toVar();
    const h = c.shiftRight(uint(15)).bitXor(c).toVar();
    return toF(h.bitAnd(uint(0xffffff))).div(16777216) as unknown as NF;
  };

  // ---- kVoxBin (§6.2): one thread = one VOXEL CLUSTER work-item -----------------------
  const kVoxBin = Fn(() => {
    const itemIdx = wgLinear(DISPATCH_ROW).toVar();
    const itemCount = qVoxRasterRO.element(0).x;
    If(itemIdx.lessThan(itemCount), () => {
      const item = qVoxRasterRO.element(itemIdx.add(uint(1)));
      const instId = item.x.toVar();
      const ci = item.y.toVar();
      // instance transform (rigid — voxels carry NO wind, §8)
      const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
      const B = gpu.instances.element(instId.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
      const yawSc = instYaw(B);
      // the cluster's BLOCK AABB bound: word0-2 = local-space center, word3 = radius
      // (the block AABB half-diagonal, §A1). Build a conservative world-space AABB cube
      // of half-extent = the block radius about the transformed center, then project its
      // 8 corners (spec §6.2 step 2: "its 8 corners through cam.vp"). The cube is a
      // conservative superset of the block (radius = half-diagonal ≥ any half-axis).
      const cBase = ci.mul(uint(CLUSTER_WORDS)).toVar();
      const cxL = elemU(gpu.clusters, cBase).toVar();
      const cyL = elemU(gpu.clusters, cBase.add(uint(1))).toVar();
      const czL = elemU(gpu.clusters, cBase.add(uint(2))).toVar();
      const rL = elemU(gpu.clusters, cBase.add(uint(3))).toVar();
      // bitcast the f32-bits cluster sphere words
      const cLocal = vec3(bcU2F(cxL), bcU2F(cyL), bcU2F(czL)) as unknown as NV3;
      const rLocal = bcU2F(rL).toVar();
      // conservative world center + radius under the contract transform (no swayPad).
      const wCenter = instTransformPoint(A, B, yawSc, cLocal);
      const wR = instSphereRadius(A, B, rLocal as unknown as NF, float(0)).toVar();

      // project 8 corners of the world AABB cube [wCenter ± wR] → screen bbox + nearest z
      const sMinX = float(1e9).toVar();
      const sMinY = float(1e9).toVar();
      const sMaxX = float(-1e9).toVar();
      const sMaxY = float(-1e9).toVar();
      const nearZ = float(1e9).toVar(); // nearest NDC z (= smallest cz)
      const allBehind = uint(1).toVar(); // 1 until any corner has w>NEAR_EPS
      const W = float(cam.uW);
      const H = float(cam.uH);
      loopI('cz3', toI(0), toI(1), (zc) => {
        loopI('cy3', toI(0), toI(1), (yc) => {
          loopI('cx3', toI(0), toI(1), (xc) => {
            const sx = toF(xc).mul(2).sub(1);
            const sy = toF(yc).mul(2).sub(1);
            const sz = toF(zc).mul(2).sub(1);
            const wp = vec3(
              wCenter.x.add(sx.mul(wR)),
              wCenter.y.add(sy.mul(wR)),
              wCenter.z.add(sz.mul(wR)),
            ) as unknown as NV3;
            const p = (cam.vp.mul(vec4(wp, 1)) as unknown as NV4).toVar();
            If(p.w.greaterThan(float(NEAR_EPS)), () => {
              allBehind.assign(uint(0));
              const ndc = p.xyz.div(p.w).toVar();
              const scx = ndc.x.add(1).mul(0.5).mul(W);
              const scy = ndc.y.add(1).mul(0.5).mul(H);
              sMinX.assign(sMinX.min(scx));
              sMaxX.assign(sMaxX.max(scx));
              sMinY.assign(sMinY.min(scy));
              sMaxY.assign(sMaxY.max(scy));
              nearZ.assign(nearZ.min(ndc.z));
            });
          });
        });
      });
      If(allBehind.equal(uint(0)), () => {
        // clamp the screen bbox to the framebuffer; cull if it falls fully outside
        const bbMinX = maxI(toI(0), toI(sMinX.floor())).toVar();
        const bbMaxX = minI(toI(width - 1), toI(sMaxX.ceil())).toVar();
        const bbMinY = maxI(toI(0), toI(sMinY.floor())).toVar();
        const bbMaxY = minI(toI(height - 1), toI(sMaxY.ceil())).toVar();
        const validBB = bbMinX.lessThanEqual(bbMaxX).and(bbMinY.lessThanEqual(bbMaxY));
        If(validBB, () => {
          // nearest NDC z clamped to [0,1] for the depth key (near-plane crossing
          // corners are dropped above; a fully-near block keeps nearZ ≥ 0).
          const nz = nearZ.clamp(0, 1).toVar();
          const bucket = minU(
            depthKey24(nz as unknown as NF).shiftRight(uint(24 - KBITS)),
            uint(K_BUCKETS - 1),
          ).toVar();
          const payload = itemIdx.bitAnd(uint(PAYLOAD_MASK)).toVar();
          const entry = payload.bitOr(bucket.shiftLeft(uint(BUCKET_SHIFT))).toVar();
          // scatter the bucket-tagged entry into each 16×16 tile the bbox overlaps
          const tMinX = maxI(toI(0), bbMinX.div(toI(TILE)));
          const tMaxX = minI(toI(tilesX - 1), bbMaxX.div(toI(TILE)));
          const tMinY = maxI(toI(0), bbMinY.div(toI(TILE)));
          const tMaxY = minI(toI(tilesY - 1), bbMaxY.div(toI(TILE)));
          loopI('ty', tMinY as unknown as NI, tMaxY as unknown as NI, (ty) => {
            loopI('tx', tMinX as unknown as NI, tMaxX as unknown as NI, (tx) => {
              const tileIdx = ty.mul(toI(tilesX)).add(tx);
              const slot = atomicAdd(atomicBuf.atomic.element(uint(tileIdx)), uint(1)) as unknown as NU;
              If(slot.lessThan(uint(FLAT_TILE_CAP)), () => {
                const base = uint(tileIdx).mul(uint(FLAT_TILE_CAP)).add(slot);
                elemUW(dataBuf.rw, base).assign(entry);
              });
              // FLAT-list overflow drops a block silently (NOT mis-elected) — the
              // original tileproto behaviour; far fewer voxel blocks than tri cut.
            });
          });
        });
      });
    });
    // [1] local size: the cull's voxRasterDispatchAttr is split2D OVER the voxel-cluster
    // count (n workgroups), so one single-thread workgroup == one cluster work-item.
  })().compute(DISPATCH_ROW, [1]);
  (kVoxBin as { setName(n: string): unknown }).setName('nanVoxBin');

  // ---- kRasterVox (§6.3): one workgroup per 16×16 tile, on-chip election --------------
  const TILE_PX = TILE * TILE;
  const kRasterVox = Fn(() => {
    const tileIdx = wgLinear(DISPATCH_ROW).toVar();
    const local = localX().toVar();
    const tileX = tileIdx.mod(uint(tilesX));
    const tileY = tileIdx.div(uint(tilesX));
    const px0 = tileX.mul(uint(TILE)).toVar();
    const py0 = tileY.mul(uint(TILE)).toVar();
    // tiles past the grid (the last partial dispatch row) do nothing.
    If(tileIdx.lessThan(uint(nTiles)), () => {
      // ON-CHIP election (§6.3): wgElect[lpx] = packed (depthKey24<<8|id8); wgId = full
      // voxel payload. 2 × 256 × 4 B = 2 KB workgroup memory.
      const wgElect = workgroupArray('uint', TILE_PX);
      (wgElect as unknown as { bufferType: string }).bufferType = 'atomic<u32>';
      const wgId = workgroupArray('uint', TILE_PX);

      // PRE-SEED (§6.2): copy the GLOBAL near-field triangle winners into wgElect, so
      // bricks occluded by NEAR-FIELD leaves also early-skip (extends the occlusion skip
      // across the tri/voxel tier boundary). The wgId stays 0 for seeded pixels — the
      // flush only writes visBV when THIS wave's election strictly beats the global, so a
      // seeded triangle winner is never overwritten by a 0 id.
      loopU(local, uint(TILE_PX), (i) => {
        const lx = i.mod(uint(TILE));
        const ly = i.div(uint(TILE));
        const gx = px0.add(lx);
        const gy = py0.add(ly);
        const inImg = gx.lessThan(uint(width)).and(gy.lessThan(uint(height)));
        // FORCE the atomic-load result into a plain uint VAR at the loop-body root BEFORE it
        // becomes the value-arg of the atomicStore. An atomicLoad (AtomicFunctionNode) wrapped
        // in a .select() and then handed straight to atomicStore is the SAME TSL r184 pathology
        // as the inner cand/e bugs: AtomicFunctionNode.generate calls valueNode.build(...,'uint'),
        // and the un-materialized select-over-atomicLoad yields '' in that nested scope ⇒
        // "Invalid generated code, expected a uint" ⇒ generateConst('uint') (0u) ⇒ the pre-seed
        // is silently dropped (no cross-tier occlusion) AND three logs the codegen error. Reading
        // the atomic INTO `loaded` (its own statement) materializes a real uint temp first; the
        // select then composes plain uint temps, and the atomicStore value-arg is a built var.
        const loaded = aLoadU(visPayloadV.atomic.element(gy.mul(uint(width)).add(gx))).toVar();
        const seed = inImg.select(loaded, uint(0)).toVar();
        atomicStore(wgElect.element(i), seed);
        wgSet(wgId, i, uint(0));
      }, WG);
      workgroupBarrier();

      const tileX0 = (int(px0) as unknown as NI).toVar();
      const tileY0 = (int(py0) as unknown as NI).toVar();
      const tileX1 = minI(tileX0.add(toI(TILE - 1)), toI(width - 1));
      const tileY1 = minI(tileY0.add(toI(TILE - 1)), toI(height - 1));

      // read the per-tile count + increment the WRITE_CTR through the SAME atomic view
      // (mixing the .ro and .atomic views of one buffer in a single kernel is illegal
      // aliasing — the N0 same-scope law). aLoadU is a relaxed atomic load.
      const count = minU(aLoadU(atomicBuf.atomic.element(tileIdx)), uint(FLAT_TILE_CAP)).toVar();
      const listBase = tileIdx.mul(uint(FLAT_TILE_CAP)).toVar();
      const nRounds = count.add(uint(WG - 1)).div(uint(WG)).toVar();

      // K near→far passes (bk = 7 NEAREST .. 0 FARTHEST). Each pass walks the single flat
      // list and processes ONLY entries whose bucket==bk; each block hits exactly one pass.
      loopU(uint(0), uint(K_BUCKETS), (bi) => {
        const bk = uint(K_BUCKETS - 1).sub(bi).toVar();
        loopU(uint(0), nRounds, (r) => {
          const slot = local.add(r.mul(uint(WG))).toVar();
          If(slot.lessThan(count), () => {
            const entry = elemU(dataBuf.ro, listBase.add(slot)).toVar();
            const eBucket = entry.shiftRight(uint(BUCKET_SHIFT)).toVar();
            If(eBucket.equal(bk), () => {
              const payload = entry.bitAnd(uint(PAYLOAD_MASK)).toVar();
              // re-decode the block (the bin computed it; we re-project here — the bin
              // stored only the 1-word bucket-tagged entry, §6.2 "read ONCE in the fine
              // pass via the work-item index"). Cheap: 8-corner project of the AABB cube.
              const bitem = qVoxRasterRO.element(payload.add(uint(1)));
              const binstId = bitem.x.toVar();
              const bci = bitem.y.toVar();
              const bA = gpu.instances.element(binstId.mul(uint(2))).toVar() as unknown as NV4;
              const bB = gpu.instances.element(binstId.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
              const byaw = instYaw(bB);
              const cbase = bci.mul(uint(CLUSTER_WORDS)).toVar();
              const cl = vec3(
                bcU2F(elemU(gpu.clusters, cbase)),
                bcU2F(elemU(gpu.clusters, cbase.add(uint(1)))),
                bcU2F(elemU(gpu.clusters, cbase.add(uint(2)))),
              ) as unknown as NV3;
              const rl = bcU2F(elemU(gpu.clusters, cbase.add(uint(3)))).toVar();
              const wc = instTransformPoint(bA, bB, byaw, cl);
              const wr = instSphereRadius(bA, bB, rl as unknown as NF, float(0)).toVar();

              // DENSITY-MODULATED COVERAGE (§3 Risk #1 / §4.3 word4.A) — the bloat fix.
              // Read the block's representative coverage from its FIRST brick's density
              // (word4.A): brickBase = the voxel cluster's word6 (§4.1). This is the
              // coarse one-sample-per-block proxy (§6.4) — one buffer read per block,
              // loop-invariant over the footprint. Clamp into [FLOOR,CEIL] so the block
              // never fully vanishes (silhouette survives) nor paints a perfect wall.
              // OPAQUE MODE (?voxdither=0): this buffer read is pure overhead (no coverage
              // gate downstream) ⇒ skip it entirely at build time (the spec §6.4 cheap path).
              const density = voxDither
                ? (() => {
                    const brickBase = elemU(gpu.clusters, cbase.add(uint(6))).toVar();
                    const albWord = elemU(
                      gpu.voxelBricks,
                      brickBase.mul(uint(BRICK_WORDS)).add(uint(BRICK_ALBEDO)),
                    ).toVar();
                    return toF(albWord.shiftRight(uint(24)).bitAnd(uint(0xff)))
                      .div(255)
                      .clamp(COVER_FLOOR, COVER_CEIL)
                      .toVar();
                  })()
                : null;

              // screen bbox + nearest+farthest NDC z of the 8 world-AABB corners
              const W = float(cam.uW);
              const H = float(cam.uH);
              const sMinX = float(1e9).toVar();
              const sMinY = float(1e9).toVar();
              const sMaxX = float(-1e9).toVar();
              const sMaxY = float(-1e9).toVar();
              const zNear = float(1e9).toVar();
              loopI('rz3', toI(0), toI(1), (zc) => {
                loopI('ry3', toI(0), toI(1), (yc) => {
                  loopI('rx3', toI(0), toI(1), (xc) => {
                    const sx = toF(xc).mul(2).sub(1);
                    const sy = toF(yc).mul(2).sub(1);
                    const sz = toF(zc).mul(2).sub(1);
                    const wp = vec3(
                      wc.x.add(sx.mul(wr)),
                      wc.y.add(sy.mul(wr)),
                      wc.z.add(sz.mul(wr)),
                    ) as unknown as NV3;
                    const p = (cam.vp.mul(vec4(wp, 1)) as unknown as NV4).toVar();
                    If(p.w.greaterThan(float(NEAR_EPS)), () => {
                      const ndc = p.xyz.div(p.w).toVar();
                      sMinX.assign(sMinX.min(ndc.x.add(1).mul(0.5).mul(W)));
                      sMaxX.assign(sMaxX.max(ndc.x.add(1).mul(0.5).mul(W)));
                      sMinY.assign(sMinY.min(ndc.y.add(1).mul(0.5).mul(H)));
                      sMaxY.assign(sMaxY.max(ndc.y.add(1).mul(0.5).mul(H)));
                      zNear.assign(zNear.min(ndc.z));
                    });
                  });
                });
              });
              const nz = zNear.clamp(0, 1).toVar();
              // nearKey = the block's NEAREST-possible election key (AABB front-slab, exact
              // for the coarse one-sample-per-block path, §6.4); cand ≤ nearKey at EVERY
              // pixel ⇒ the skip discards a provable loser (loss-exact, §6.3 proof).
              const nearKey = depthKey24(nz as unknown as NF).shiftLeft(uint(8)).bitOr(uint(0xff)).toVar();
              // voxCz: the block's AABB-slab front depth (ONE voxCz per block — coarse
              // one-sample default, §6.4). NDC-z clamped, biased toward the nearest corner.
              const voxCz = nz.toVar();
              // the full voxel id: bit31 marker + the qVoxRaster item index (≤30 bits)
              const voxId = uint(VOX_BIT).bitOr(payload.bitAnd(uint(PAYLOAD_MASK))).toVar();

              // intersect the screen bbox with the tile + framebuffer
              const startX = maxI(maxI(toI(0), toI(sMinX.floor())), tileX0).toVar();
              const endX = minI(minI(toI(width - 1), toI(sMaxX.ceil())), tileX1).toVar();
              const startY = maxI(maxI(toI(0), toI(sMinY.floor())), tileY0).toVar();
              const endY = minI(minI(toI(height - 1), toI(sMaxY.ceil())), tileY1).toVar();
              const validBB = startX.lessThanEqual(endX).and(startY.lessThanEqual(endY));
              If(validBB, () => {
                loopI('vy', startY as unknown as NI, endY as unknown as NI, (y) => {
                  loopI('vx', startX as unknown as NI, endX as unknown as NI, (x) => {
                    const lpx = uint(y).sub(py0).mul(uint(TILE)).add(uint(x).sub(px0)).toVar();
                    const prevE = aLoadU(wgElect.element(lpx)).toVar();
                    // THE ELECTION (loss-exact, the codegen-safe tileproto shape). Factored
                    // into a closure so both modes emit IDENTICAL flow: the `cand` `.toVar()`
                    // declaration and the `atomicMax` that consumes it stay in the SAME
                    // ConditionalNode block (whether that block is the occlusion-skip `If` in
                    // opaque mode, or the nested coverage `If(keep)` in dither mode).
                    //
                    // BUILD `cand` HERE, INSIDE the election conditional — verbatim the working
                    // tileproto shape (dfe6518:NaniteTileRaster.ts:617-639: `cand` is declared
                    // inside the skip `If`, in the SAME ConditionalNode flow as the atomicMax
                    // that consumes it). The Stage-2 port originally DEVIATED by HOISTING this
                    // `cand` `.toVar()` to the OUTER function scope (it's loop-invariant per
                    // block) — but in TSL r184 a `.toVar()` whose declaration sits in the outer
                    // scope while its FIRST build is the value argument of an atomicMax nested
                    // several ConditionalNodes deep gets its flow-code placed in the wrong block.
                    // When AtomicFunctionNode.generate (three/src/nodes/gpgpu/AtomicFunctionNode.js)
                    // calls `valueNode.build(builder,'uint')`, the hoisted var yields '' in that
                    // scope ⇒ Node.build hits the `result===''` guard (three/src/nodes/core/Node.js
                    // :969-976) ⇒ "TSL: Invalid generated code, expected a uint" ⇒ three falls back
                    // to `generateConst('uint')` (a const 0u) ⇒ the election word/depth is silently
                    // corrupted (blocky slabs; the occlusion early-skip never fires). Declaring
                    // `cand` in the atomic's own conditional flow (the tileproto position) keeps
                    // its var declaration and the atomic in the same block ⇒ valid uint code.
                    // voxCz is loop-invariant (one voxCz per block, §6.4), so the node-graph
                    // PLACEMENT of `cand` here is loss-EXACT — identical arithmetic.
                    const electHere = (): void => {
                      const cand = depthKey24(voxCz as unknown as NF)
                        .shiftLeft(uint(8))
                        .bitOr(payload.bitAnd(uint(0xff)))
                        .toVar();
                      If(cand.greaterThan(prevE), () => {
                        const wonE = atomicMax(wgElect.element(lpx), cand) as unknown as NU;
                        If(cand.greaterThan(wonE), () => {
                          wgSet(wgId, lpx, voxId);
                          // debug per-pixel BRICK-WRITE counter (Stage-2 overlay, §A2)
                          atomicAdd(atomicBuf.atomic.element(uint(WRITE_CTR)), uint(1));
                        });
                      });
                    };
                    // OCCLUSION SKIP (§6.3) FIRST: nearKey can't beat the on-chip winner ⇒ this
                    // block loses at this pixel ⇒ skip the election entirely (loss-exact: cand ≤
                    // nearKey at every pixel). This is the cheapest gate (an on-chip atomic load
                    // + compare), so it runs BEFORE any density read / coverHash — occluded
                    // pixels pay NOTHING, including (in dither mode) the hash. This reorder is a
                    // pure win: the Stage-3b code ran coverHash on every covered pixel up front
                    // and `.and()`ed it with the occlusion test, paying the hash even on pixels
                    // the skip would reject. Now the hash runs ONLY past the skip.
                    If(nearKey.greaterThan(prevE), () => {
                      if (voxDither && density) {
                        // DENSITY-MODULATED COVERAGE GATE (§3 Risk #1 / §4.3 word4.A) — the
                        // bloat-to-blob fix. The block keeps THIS pixel only if a STABLE
                        // per-(pixel,block) dither falls below the block density: a 0.16-density
                        // conifer block covers ~16% of its footprint, and the dropped ~84% of
                        // pixels are never elected ⇒ the bricks/background BEHIND win there ⇒
                        // the coarse band reads as sparse, see-through FOLIAGE, not a solid slab.
                        // Salt the hash with the block payload so each block has its OWN stipple
                        // (no aligned holes across overlapping blocks). Computed HERE, after the
                        // occlusion skip — occluded pixels never pay it. NOTE the tension this
                        // knob resolves: see-through ⇒ no occlusion ⇒ the election explodes; the
                        // OPAQUE default (?voxdither=0) drops this gate entirely.
                        const keep = coverHash(uint(x), uint(y), payload).lessThan(density).toVar();
                        If(keep, electHere);
                      } else {
                        // OPAQUE (?voxdither=0, DEFAULT): every pixel past the occlusion skip is
                        // elected opaque — no hash, no see-through. Cheap AND (at sub-pixel brick
                        // size via finer ?voxgrid / farther ?voxnear) correct (spec §6.4).
                        electHere();
                      }
                    });
                  });
                });
              });
            });
          });
        });
      });

      // FLUSH-MERGE (§6.3): one barrier, then merge wgElect/wgId into the GLOBAL
      // visPayloadV/visBV via atomicMax — preserving any nearer near-field-tri winner
      // (since we pre-seeded, this is a no-op where the tri already won). The id is
      // written only when THIS wave's word strictly beats the global (so a seeded tri's
      // visBV id is never clobbered by a 0 wgId).
      workgroupBarrier();
      loopU(local, uint(TILE_PX), (i) => {
        const e = aLoadU(wgElect.element(i)).toVar();
        // GATE ON `e` AT THE BODY ROOT — verbatim the working tileproto flush shape
        // (dfe6518:NaniteTileRaster.ts:667-680: `If(e.notEqual(uint(0)))` wraps the
        // atomicMax, so `e`'s FIRST build is this root-level condition, NOT the deeply-
        // nested atomicMax value-arg). The Stage-2 port DEVIATED by gating on `id` and
        // adding an extra `If(inImg)` layer, so `e`.toVar()'s declaration sat at body root
        // while its FIRST build was the value argument of `atomicMax` two ConditionalNodes
        // deep — the SAME TSL r184 pathology as the inner `cand` bug: AtomicFunctionNode
        // .generate calls `e.build(builder,'uint')`, the hoisted var yields '' in that
        // scope ⇒ Node.build's `result===''` guard ⇒ "Invalid generated code, expected a
        // uint" ⇒ three substitutes `generateConst('uint')` (a const 0u) ⇒ the flush
        // atomicMax merges 0 (no-op) ⇒ wave winners never reach the GLOBAL visPayloadV.
        // Building `e` in this root condition forces its declaration to emit before the
        // nested atomic ⇒ valid uint code. (Out-of-image pixels are never elected — the
        // election clamps to width-1/height-1 — so they keep wgElect==seed/0; a seeded-
        // only pixel re-merges its own already-global value, a harmless atomicMax no-op,
        // and its wgId==0 guard below skips the id store. The pre-seed `id != 0` semantics
        // are preserved as the INNER gate, loss-exact.)
        If(e.notEqual(uint(0)), () => {
          const id = (wgId.element(i) as unknown as NU).toVar();
          // only flush pixels THIS wave actually elected a voxel into (wgId != 0). Seeded-
          // only pixels (wgId==0) carry a triangle winner already in the global buffer.
          If(id.notEqual(uint(0)), () => {
            const lx = i.mod(uint(TILE));
            const ly = i.div(uint(TILE));
            const gx = px0.add(lx);
            const gy = py0.add(ly);
            const inImg = gx.lessThan(uint(width)).and(gy.lessThan(uint(height)));
            If(inImg, () => {
              const px = gy.mul(uint(width)).add(gx).toVar();
              const won = atomicMax(visPayloadV.atomic.element(px), e) as unknown as NU;
              If(e.greaterThan(won), () => {
                atomicStore(visBV.atomic.element(px), id);
              });
            });
          });
        });
      }, WG);
    });
    // one workgroup per 16×16 tile (nTiles workgroups of WG threads). wgLinear handles
    // the >65535-workgroup 2D tiling three.js applies (workgroupId.y) transparently.
  })().compute(nTiles * WG, [WG]);
  (kRasterVox as { setName(n: string): unknown }).setName('nanRasterVox');

  // ---- kVoxScatter (?voxraster=scatter, the DEFAULT — the SCAR fix) -------------------
  // ONE thread = ONE voxel CLUSTER work-item from qVoxRaster (the SAME cull fanout the bin
  // uses). Each block DIRECTLY atomicMax-elects its footprint into the GLOBAL visPayloadV/
  // visBV — VERBATIM the world1 triangle election (NaniteRaster.ts:947-960): read prevE,
  // gate cand>prevE, atomicMax(visPayloadV), and the WINNER atomicStore's the full id into
  // visBV. NO per-tile bin, NO K near→far passes, NO on-chip wgElect/flush-merge — those
  // are the bin SETUP FLOOR (the SCAR §6.0) the rigorous sweep proved is the net-loss.
  //
  // DEPTH is handled by the atomicMax alone (nearest block wins per pixel) — order-free,
  // loss-exact across all overlapping blocks, exactly as the unordered world1 scatter is
  // for triangles. Because the elected key carries depthKey24 in the high 24 bits, two
  // scatter threads racing the same pixel resolve to the NEAREST block deterministically.
  //
  // PER-BLOCK OCCLUSION CULL (lever 2): BEFORE the footprint loop, read the current GLOBAL
  // winner ONCE at the block's projected-centre pixel; if the block's NEAREST-possible key
  // (nearKey, the AABB front-slab) can't beat it, the block is fully behind the near mesh /
  // a nearer block ⇒ SKIP the whole block. ONE global read per block (vs the bin's per-
  // footprint-pixel early-skip) culls the deep-canopy occluded blocks (worst poses #78-111)
  // cheaply. This is COARSE+CONSERVATIVE: a block whose centre is occluded but whose silhou-
  // ette edge peeks past the occluder is dropped — acceptable for sub-pixel foliage blocks
  // (the same coarse-grain tradeoff the triangle world1 + HZB make at cluster level), and
  // never produces a WRONG winner (it only ever DROPS a would-be loser-or-edge-sliver, never
  // overwrites a nearer pixel). ?voxoccl=0 disables it for the A/B.
  const voxOccl = (new URLSearchParams(window.location.search).get('voxoccl') ?? '1') !== '0';
  // ── COOPERATIVE-RASTER WORKGROUP MEMORY (the close-up overdraw-imbalance fix) ──────
  // Per-cluster shared brick records (Phase A fills, Phase B consumes). MAX_BRICKS_PER_
  // CLUSTER=128 entries each. wgCand packs the brick's election key (depthKey24<<8|id8);
  // bbX0/bbY0/bbW/bbH are the brick's CLAMPED screen bbox; wgBrickAbs[b]/wgDensBits[b] are
  // only consumed in dither mode. ~6×128×4B ≈ 3 KB workgroup memory — comfortable.
  const WG_RASTER = MAX_BRICKS_PER_CLUSTER; // 128 lanes per cluster workgroup
  const kVoxScatter = Fn(() => {
    // ── WORKGROUP-COOPERATIVE FOOTPRINT RASTER (CudaRaster T6/T8 + Lucid balanced ──────
    // dispatch; the residual-overdraw / per-lane-imbalance fix). Iteration-1 went
    // [1]-thread-per-workgroup → ONE thread per BRICK (localX), filling the SIMD waves and
    // removing the single-lane near-plane blowup. BUT each brick-thread still SERIALLY
    // looped its OWN footprint, so a brick that projects large at close range (up to the
    // BRICK_MAX_EXT box) grinds ~16K px on ONE lane while sibling lanes (small bricks)
    // sit idle — EXACTLY the "one thread per primitive, loop its pixels" model the GPU-
    // rasterizer literature condemns as badly under-utilized on MIXED-SIZE primitives
    // (Laine&Karras CudaRaster §2/§6/Fig.5: FreePipe's scheme; the per-lane footprint
    // variance flattens the waves to the WORST brick). UE/Nanite keeps every SW-rasterized
    // primitive PROVABLY SMALL (clusters ≥32 px go to the HW raster, never SW-looped),
    // and CudaRaster's fix (T6/T8) is to DECOUPLE coverage from work-distribution: flatten
    // the whole batch's footprint into one flat pixel space and spread it EVENLY across
    // all lanes (Lucid's dispatchLargeTriBalanced splits a wide primitive's per-row writes
    // across the subgroup the same way). We apply that here:
    //
    //   PHASE A (1 lane = 1 brick): decode + project + CLAMP each brick → store its bbox +
    //     election key into workgroup shared arrays; area = bbW·bbH. (Same per-brick math
    //     as before, but it now only SETS UP the footprint, it does not RASTER it.)
    //   PHASE B (ALL lanes, cooperative per-BRICK): loop b over the cluster's bricks; WITHIN
    //     each brick stride its OWN footprint [0, area) across all WG_RASTER lanes
    //     (loopU(brickLocal, area, …, WG_RASTER)). A 16K-px brick's pixels are spread across
    //     all 128 lanes (~128 iters/lane) instead of 16K on one — O(area/WG) per lane, not
    //     O(area); waves stay full regardless of per-brick footprint variance. localPx is
    //     [0, area) BY CONSTRUCTION, so every footprint pixel of every non-empty brick reaches
    //     the election (54c3947 density) and the ly = localPx/bbW overflow column is
    //     STRUCTURALLY impossible — no flat-index reconstruction, no prefix array, no binary
    //     search (the old flat-domain scheme's 99.2% loss + duplicate-prefix hazard, deleted).
    //
    // The election (depthKey24 atomicMax + winner visBV store), the per-brick geometry
    // (54c3947), the BRICK_MAX_EXT clamp, the per-block occlusion cull, and the ?voxdither
    // path are all PRESERVED bit-exact — only the WORK DISTRIBUTION of the footprint loop
    // changes. ≤10 buffers unchanged (no new storage buffers; only workgroup memory added).
    const itemIdx = wgLinear(DISPATCH_ROW).toVar();
    const brickLocal = localX().toVar(); // Phase A: this lane's brick index within the cluster
    const itemCount = qVoxRasterRO.element(0).x;
    // shared per-brick records (Phase A → Phase B). Plain (non-atomic) workgroup arrays.
    const wgBbX0 = workgroupArray('uint', WG_RASTER);
    const wgBbY0 = workgroupArray('uint', WG_RASTER);
    const wgBbW = workgroupArray('uint', WG_RASTER);
    const wgBbH = workgroupArray('uint', WG_RASTER);
    const wgCand = workgroupArray('uint', WG_RASTER); // depthKey24<<8 | id8 (loss-exact key)
    const wgBrickAbs = voxDither ? workgroupArray('uint', WG_RASTER) : null; // dither salt
    const wgDensBits = voxDither ? workgroupArray('uint', WG_RASTER) : null; // dither raw density byte (0..255)
    If(itemIdx.lessThan(itemCount), () => {
      const item = qVoxRasterRO.element(itemIdx.add(uint(1)));
      const instId = item.x.toVar();
      const ci = item.y.toVar();
      const A = gpu.instances.element(instId.mul(uint(2))).toVar() as unknown as NV4;
      const B = gpu.instances.element(instId.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
      const yawSc = instYaw(B);
      const cBase = ci.mul(uint(CLUSTER_WORDS)).toVar();
      // PER-BRICK FIX (the oversized-square bug, 54c3947): paint EACH brick's OWN small AABB
      // at its real grid cell (§6.2), NOT the whole block slab. word6 = brickBase, word7 low
      // byte = brickCount (§4.1). One thread = brick (brickBase + brickLocal).
      const brickBase = elemU(gpu.clusters, cBase.add(uint(6))).toVar();
      const brickCount = elemU(gpu.clusters, cBase.add(uint(7))).bitAnd(uint(0xff)).toVar();
      // the block sphere (word0-3) is the coarse PER-BLOCK occlusion-cull bound (one global
      // read at the projected block centre, computed once on thread 0), not a footprint.
      const blkLocal = vec3(
        bcU2F(elemU(gpu.clusters, cBase)),
        bcU2F(elemU(gpu.clusters, cBase.add(uint(1)))),
        bcU2F(elemU(gpu.clusters, cBase.add(uint(2)))),
      ) as unknown as NV3;
      const blkRLocal = bcU2F(elemU(gpu.clusters, cBase.add(uint(3)))).toVar();
      const blkWCenter = instTransformPoint(A, B, yawSc, blkLocal);
      const blkWR = instSphereRadius(A, B, blkRLocal as unknown as NF, float(0)).toVar();
      const W = float(cam.uW);
      const H = float(cam.uH);

      const payload = itemIdx.bitAnd(uint(PAYLOAD_MASK)).toVar();
      const voxId = uint(VOX_BIT).bitOr(payload).toVar();

      // PER-BLOCK OCCLUSION CULL (lever 2) — computed ONCE per workgroup on thread 0 into a
      // workgroup-shared flag, then broadcast via a barrier. One global read per BLOCK (not
      // per-brick, not per-pixel). If the block's NEAREST-possible key (its AABB front-slab)
      // can't beat the global winner at its centre pixel, every brick-thread skips. Conserva-
      // tive (a centre-occluded block whose silhouette edge peeks is dropped — fine for sub-
      // pixel foliage; never produces a WRONG winner). ?voxoccl=0 disables it for the A/B.
      const wgVisible = workgroupArray('uint', 1);
      if (voxOccl) {
        If(brickLocal.equal(uint(0)), () => {
          wgSet(wgVisible, uint(0), uint(1));
          const bp = (cam.vp.mul(vec4(blkWCenter, 1)) as unknown as NV4).toVar();
          If(bp.w.greaterThan(float(NEAR_EPS)), () => {
            const bndc = bp.xyz.div(bp.w).toVar();
            const bnz = bndc.z.sub(blkWR.div(bp.w)).clamp(0, 1).toVar();
            const bNearKey = depthKey24(bnz as unknown as NF).shiftLeft(uint(8)).bitOr(uint(0xff)).toVar();
            const bscx = minI(maxI(toI(0), toI(bndc.x.add(1).mul(0.5).mul(W))), toI(width - 1)).toVar();
            const bscy = minI(maxI(toI(0), toI(bndc.y.add(1).mul(0.5).mul(H))), toI(height - 1)).toVar();
            const bcpx = uint(bscy).mul(uint(width)).add(uint(bscx)).toVar();
            const centreE = aLoadU(visPayloadV.atomic.element(bcpx)).toVar();
            If(bNearKey.lessThanEqual(centreE), () => {
              wgSet(wgVisible, uint(0), uint(0)); // centre fully occluded ⇒ skip the block
            });
          });
        });
        workgroupBarrier();
      } else {
        // no occl cull → every live brick-thread proceeds; seed the flag so the read below
        // is uniform (thread 0 still writes it; barrier makes the broadcast well-defined).
        If(brickLocal.equal(uint(0)), () => {
          wgSet(wgVisible, uint(0), uint(1));
        });
        workgroupBarrier();
      }

      // ── PHASE A (1 lane = 1 brick): decode + project + CLAMP → store the brick's footprint
      // RECORD into shared memory. This lane does NOT raster the footprint — it only SETS UP
      // the bbox + election key so Phase B can spread the pixels across all lanes. Seed the
      // record EMPTY (bbW=0) for every lane first so a culled/idle lane contributes 0 area.
      wgSet(wgBbW, brickLocal, uint(0));
      wgSet(wgBbH, brickLocal, uint(0));
      wgSet(wgCand, brickLocal, uint(0));
      wgSet(wgBbX0, brickLocal, uint(0));
      wgSet(wgBbY0, brickLocal, uint(0));
      const brickActive = brickLocal.lessThan(brickCount).and((wgVisible.element(uint(0)) as unknown as NU).equal(uint(1)));
      If(brickActive, () => {
        const bAbs = brickBase.add(brickLocal).toVar();      // absolute brick index
        const bWordBase = bAbs.mul(uint(BRICK_WORDS)).toVar();
        // per-brick LOCAL center (words 5..7) + half-extent (word8) — the real grid cell.
        const brLocal = vec3(
          bcU2F(elemU(gpu.voxelBricks, bWordBase.add(uint(BRICK_POS_X)))),
          bcU2F(elemU(gpu.voxelBricks, bWordBase.add(uint(BRICK_POS_X + 1)))),
          bcU2F(elemU(gpu.voxelBricks, bWordBase.add(uint(BRICK_POS_X + 2)))),
        ) as unknown as NV3;
        const brHalf = bcU2F(elemU(gpu.voxelBricks, bWordBase.add(uint(BRICK_HALF)))).toVar();
        const brWCenter = instTransformPoint(A, B, yawSc, brLocal);
        const brWR = instSphereRadius(A, B, brHalf as unknown as NF, float(0)).toVar();

        // DENSITY-MODULATED COVERAGE (?voxdither=1 only): stash THIS brick's salt + its RAW
        // density byte (word4.A, the high byte of BRICK_ALBEDO — already a uint, no float
        // cast needed) for Phase B, which rebuilds the clamped float. Skip both entirely in
        // the OPAQUE default (no coverage gate ⇒ no per-brick density read).
        if (voxDither && wgBrickAbs && wgDensBits) {
          const densByte = elemU(gpu.voxelBricks, bWordBase.add(uint(BRICK_ALBEDO)))
            .shiftRight(uint(24))
            .bitAnd(uint(0xff))
            .toVar();
          wgSet(wgBrickAbs, brickLocal, bAbs);
          wgSet(wgDensBits, brickLocal, densByte);
        }

        // project the 8 corners of THIS brick's world AABB cube [brWCenter ± brWR] → screen bbox
        // + near z. A near-plane-straddling brick's surviving corners can fling the bbox to
        // extreme coords; those degenerate corners are detected + excluded (see STRADDLE FLAG)
        // and the brick is routed to a stable centre-box footprint instead.
        const sMinX = float(1e9).toVar();
        const sMinY = float(1e9).toVar();
        const sMaxX = float(-1e9).toVar();
        const sMaxY = float(-1e9).toVar();
        const nearZ = float(1e9).toVar();
        const allBehind = uint(1).toVar();
        // STRADDLE FLAG: set if ANY of the 8 AABB corners is degenerate — i.e. at/behind the
        // near plane (w ≤ NEAR_EPS) OR projecting wildly off-screen (|ndc.x|>NDC_EXPLODE or
        // |ndc.y|>NDC_EXPLODE). Both signal a corner near the near plane whose tiny positive w
        // makes ndc = p/w shoot to extreme coords; such a corner has NO trustworthy 2D position,
        // so we EXCLUDE it from sMin/sMax (it would otherwise drag the bbox into empty sky — the
        // root of the iter-1 monolith columns) and route the whole brick to the stable centre-box
        // SILHOUETTE branch below. A genuinely on-screen corner (|ndc| ≤ NDC_EXPLODE=3, i.e. up
        // to three screens past centre — far looser than any real foliage brick) is UNCHANGED, so
        // every normal in-front brick keeps 54c3947's exact column-free footprint.
        const straddles = uint(0).toVar();
        loopI('sz3', toI(0), toI(1), (zc) => {
          loopI('sy3', toI(0), toI(1), (yc) => {
            loopI('sx3', toI(0), toI(1), (xc) => {
              const sx = toF(xc).mul(2).sub(1);
              const sy = toF(yc).mul(2).sub(1);
              const sz = toF(zc).mul(2).sub(1);
              const wp = vec3(
                brWCenter.x.add(sx.mul(brWR)),
                brWCenter.y.add(sy.mul(brWR)),
                brWCenter.z.add(sz.mul(brWR)),
              ) as unknown as NV3;
              const p = (cam.vp.mul(vec4(wp, 1)) as unknown as NV4).toVar();
              If(p.w.greaterThan(float(NEAR_EPS)), () => {
                const ndc = p.xyz.div(p.w).toVar();
                const exploded = ndc.x
                  .abs()
                  .greaterThan(float(NDC_EXPLODE))
                  .or(ndc.y.abs().greaterThan(float(NDC_EXPLODE)));
                If(exploded, () => {
                  straddles.assign(uint(1)); // near-plane blowup ⇒ exclude this corner from the bbox
                }).Else(() => {
                  allBehind.assign(uint(0));
                  // RAW projected corner — bit-IDENTICAL to 54c3947's per-brick bbox accumulation
                  // for an all-in-front (non-straddling) brick: the COLUMN-FREE, KNOWN-CORRECT path.
                  sMinX.assign(sMinX.min(ndc.x.add(1).mul(0.5).mul(W)));
                  sMaxX.assign(sMaxX.max(ndc.x.add(1).mul(0.5).mul(W)));
                  sMinY.assign(sMinY.min(ndc.y.add(1).mul(0.5).mul(H)));
                  sMaxY.assign(sMaxY.max(ndc.y.add(1).mul(0.5).mul(H)));
                  nearZ.assign(nearZ.min(ndc.z));
                });
              }).Else(() => {
                straddles.assign(uint(1)); // a corner at/behind the near plane ⇒ degenerate bbox
              });
            });
          });
        });
        // ENTER if the brick has ANY usable footprint: a clean non-straddle bbox (some in-front
        // corner survived ⇒ allBehind=0) OR a straddle whose stable centre is in front (handled
        // inside). A brick with no in-front corner AND no in-front centre is genuinely off-screen
        // /behind and is dropped.
        If(allBehind.equal(uint(0)).or(straddles.equal(uint(1))), () => {
          // ── FOOTPRINT BBOX. Two regimes — the fix for the iter-1 monolith columns:
          //
          //  (1) NON-STRADDLER (all 8 corners in front + on-screen, the normal/far/mid case): use
          //      54c3947's RAW framebuffer-clamped projected AABB EXACTLY (floor(sMin)..ceil(sMax)
          //      clamped to the framebuffer). This is the column-free, KNOWN-CORRECT footprint,
          //      restored bit-for-bit. A SPAN CAP to 2·BRICK_MAX_EXT px (measured from the bbox
          //      min) is applied purely as the cooperative-raster stall guard; a real foliage
          //      brick is ≪ this, so it is a no-op here and the silhouette matches 54c3947.
          //
          //  (2) STRADDLER (≥1 corner near/behind the near plane — camera literally inside the
          //      brick, forcevox=all close-up): the projected AABB is DEGENERATE — a surviving
          //      near-plane corner with tiny w explodes ndc, which dragged sMin/sMax to extreme
          //      coords, and 4fe6821's centre-anchored ±64 box then painted a SOLID 128-px slab
          //      of EMPTY SKY ⇒ the tall vertical monolith columns. For a straddler we IGNORE the
          //      degenerate sMin/sMax and instead paint a small BRICK_MAX_EXT box centred on the
          //      brick's STABLE projected centre IF that centre is in front (depth taken from the
          //      centre, clamped near so the up-close brick still wins its election); if the
          //      centre itself is behind the near plane (camera dead-centre in the brick) we DROP
          //      the brick rather than smear a column. This bounds the close-up footprint to the
          //      brick's true on-screen position — no sky columns.
          const startX = toI(0).toVar();
          const startY = toI(0).toVar();
          const endX = toI(-1).toVar(); // endX<startX ⇒ invalid until a regime sets it
          const endY = toI(-1).toVar();
          const bbNearZ = nearZ.toVar();
          If(straddles.equal(uint(0)), () => {
            const sx0 = maxI(toI(0), toI(sMinX.floor())).toVar();
            const sy0 = maxI(toI(0), toI(sMinY.floor())).toVar();
            // SPAN CAP (stall guard only): far edge ≤ near edge + 2·BRICK_MAX_EXT px.
            const capX = minI(toI(width - 1), sx0.add(toI(2 * BRICK_MAX_EXT))).toVar();
            const capY = minI(toI(height - 1), sy0.add(toI(2 * BRICK_MAX_EXT))).toVar();
            startX.assign(sx0);
            startY.assign(sy0);
            endX.assign(minI(minI(toI(width - 1), toI(sMaxX.ceil())), capX));
            endY.assign(minI(minI(toI(height - 1), toI(sMaxY.ceil())), capY));
          }).Else(() => {
            // straddler → small box at the STABLE projected centre. CLAMP, do not DROP: a brick
            // the camera sits INSIDE is the NEAREST thing on screen, so it MUST win its election,
            // not vanish (the iter-1 close-up sparseness bug = this centre being dropped when it
            // fell behind the near plane). Clamp w to a small positive epsilon so a centre at/
            // behind the near plane still yields a STABLE on-screen position; paint the bounded
            // BRICK_MAX_EXT centre box there (NOT the degenerate sMin/sMax slab ⇒ no columns),
            // and pin bbNearZ via the centre's clamped ndc.z (nz.clamp(0,1) below maps an at/
            // behind-near depth to 0 = nearest). This keeps the 048f451 silhouette fix (bounded
            // box, no tall sky monolith) while recovering every close-up brick.
            const cp = (cam.vp.mul(vec4(brWCenter, 1)) as unknown as NV4).toVar();
            const cw = cp.w.max(float(NEAR_EPS)).toVar(); // clamp w ≥ ε so a behind-near centre is still placeable
            const cndc = cp.xyz.div(cw).toVar();
            const ccx = cndc.x.add(1).mul(0.5).mul(W).toVar();
            const ccy = cndc.y.add(1).mul(0.5).mul(H).toVar();
            startX.assign(maxI(toI(0), toI(ccx.sub(float(BRICK_MAX_EXT)).floor())));
            startY.assign(maxI(toI(0), toI(ccy.sub(float(BRICK_MAX_EXT)).floor())));
            endX.assign(minI(toI(width - 1), toI(ccx.add(float(BRICK_MAX_EXT)).ceil())));
            endY.assign(minI(toI(height - 1), toI(ccy.add(float(BRICK_MAX_EXT)).ceil())));
            // depth from the centre (the in-front corners may all have been excluded ⇒ nearZ
            // would still be its 1e9 seed). min with the centre's clamped ndc.z, then clamp to
            // [0,1] below — a behind-near centre lands at/below 0 ⇒ nearest, wins its election.
            bbNearZ.assign(bbNearZ.min(cndc.z));
          });
          const validBB = startX.lessThanEqual(endX).and(startY.lessThanEqual(endY));
          If(validBB, () => {
            const nz = bbNearZ.clamp(0, 1).toVar();
            // ONE voxCz per BRICK (coarse one-sample default, §6.4) — the brick AABB front-slab.
            // Precompute the FULL election key ONCE per brick (loss-exact — voxCz is loop-
            // invariant, exactly as the prior per-pixel re-derivation produced) so Phase B
            // never re-derives depthKey24 per pixel.
            const cand = depthKey24(nz as unknown as NF)
              .shiftLeft(uint(8))
              .bitOr(payload.bitAnd(uint(0xff)))
              .toVar();
            // store the brick's footprint RECORD (clamped bbox origin + width/height + key).
            const bbW = uint(endX).sub(uint(startX)).add(uint(1)).toVar();
            const bbH = uint(endY).sub(uint(startY)).add(uint(1)).toVar();
            wgSet(wgBbX0, brickLocal, uint(startX));
            wgSet(wgBbY0, brickLocal, uint(startY));
            wgSet(wgBbW, brickLocal, bbW);
            wgSet(wgBbH, brickLocal, bbH);
            wgSet(wgCand, brickLocal, cand);
          });
        });
      });
      workgroupBarrier(); // Phase A records complete before Phase B reads them

      // ── PHASE B (ALL lanes, cooperative per-BRICK raster): loop over every brick in the
      // cluster; WITHIN each brick, stride its OWN footprint [0, area) across all WG_RASTER
      // lanes. localPx is BY CONSTRUCTION in [0, area) ⇒ it can NEVER exceed bbW·bbH, so the
      // ly = localPx/bbW overflow column is structurally impossible and NO pixel is ever
      // dropped. This restores 54c3947's density (every footprint pixel of every non-empty
      // brick reaches the election) with no flat-index reconstruction, no prefix array, and
      // no binary search (the 99.2%-loss + duplicate-prefix hazard, all deleted). A big brick
      // still spreads across all 128 lanes via its inner stride ⇒ the 4fe6821 stall-fix
      // (O(area/WG) per lane, BRICK_MAX_EXT caps any one brick at ≤128×128) is preserved;
      // tiny/empty bricks (area 0) contribute a zero-trip inner loop.
      const nBricks = minU(brickCount, uint(WG_RASTER)).toVar();
      loopU(uint(0), nBricks, (b) => {
        // recover this brick's footprint RECORD (set up in Phase A).
        const bbX0 = (wgBbX0.element(b) as unknown as NU).toVar();
        const bbY0 = (wgBbY0.element(b) as unknown as NU).toVar();
        const bbW = (wgBbW.element(b) as unknown as NU).toVar();
        const bbH = (wgBbH.element(b) as unknown as NU).toVar();
        const cand = (wgCand.element(b) as unknown as NU).toVar();
        const area = bbW.mul(bbH).toVar(); // 0 for a culled/idle brick ⇒ zero-trip inner loop
        // each lane strides this brick's footprint: localPx = brickLocal, brickLocal+128, …
        loopU(brickLocal, area, (localPx) => {
          const lx = localPx.mod(bbW).toVar();
          const ly = localPx.div(bbW).toVar(); // < bbH since localPx < bbW·bbH (no column)
          const x = bbX0.add(lx).toVar();
          const y = bbY0.add(ly).toVar();
          const px = y.mul(uint(width)).add(x).toVar();
          // emit closure — built INSIDE the per-pixel flow so `cand`'s use as the atomicMax
          // value-arg stays in the same conditional subtree (TSL r184 codegen-safe).
          const electHere = (): void => {
            // VERBATIM the world1 election (NaniteRaster.ts:937-959). `cand` was precomputed
            // per brick (loss-exact); copy it into a local var HERE so its FIRST build is in
            // the SAME ConditionalNode subtree as the atomicMax that consumes it (the hoist-
            // pathology fix). prevE relaxed-load gates the atomic.
            const candL = cand.toVar();
            const prevE = aLoadU(visPayloadV.atomic.element(px)).toVar();
            If(candL.greaterThan(prevE), () => {
              const wonE = atomicMax(visPayloadV.atomic.element(px), candL) as unknown as NU;
              If(candL.greaterThan(wonE), () => {
                atomicStore(visBV.atomic.element(px), voxId);
                // debug per-pixel BRICK-WRITE counter (Stage-2 overlay, §A2)
                atomicAdd(atomicBuf.atomic.element(uint(WRITE_CTR)), uint(1));
              });
            });
          };
          if (voxDither && wgBrickAbs && wgDensBits) {
            // density-modulated see-through dither (?voxdither=1): stable per-(pixel,brick)
            // stipple drops (1-density) of the footprint. The raw density byte (0..255) was
            // stashed in Phase A; rebuild the clamped [FLOOR,CEIL] float exactly as before.
            // Salt with the absolute brick index so neighbouring bricks have independent stipples.
            const dens = toF(wgDensBits.element(b) as unknown as NU)
              .div(255)
              .clamp(COVER_FLOOR, COVER_CEIL)
              .toVar();
            const keep = coverHash(x, y, wgBrickAbs.element(b) as unknown as NU).lessThan(dens).toVar();
            If(keep, electHere);
          } else {
            electHere();
          }
        }, WG_RASTER);
      });
    });
    // ONE WORKGROUP per voxel cluster work-item (split2D indirect args over the fanned count,
    // unchanged), WG_RASTER (=MAX_BRICKS_PER_CLUSTER) threads each. Phase A = 1 lane/brick;
    // Phase B = all lanes cooperatively rasterize the cluster's flat footprint. The baked
    // `.compute(count,[wg])` is the static fallback; the indirect args drive the real WG count.
  })().compute(DISPATCH_ROW * WG_RASTER, [WG_RASTER]);
  (kVoxScatter as { setName(n: string): unknown }).setName('nanVoxScatter');

  const dispatchVoxel = (renderer: Renderer): void => {
    if (useScatter) {
      // SCATTER (DEFAULT): no bins to clear except the shared WRITE_CTR (slot WRITE_CTR). The
      // per-tile counts are unused here, but kClearBins also zeroes the WRITE_CTR debug counter
      // cheaply (one small dispatch); then ONE kernel directly elects into the global vis buffer.
      dispatch(renderer, kClearBins);
      dispatchIndirect(renderer, kVoxScatter as never, voxRasterDispatchAttr);
      return;
    }
    // BIN (A/B fallback, ?voxraster=bin): the prior two-kernel binned path.
    dispatch(renderer, kClearBins);
    // kVoxBin: one workgroup-thread per voxel cluster (2D-split over the fanned count).
    dispatchIndirect(renderer, kVoxBin as never, voxRasterDispatchAttr);
    // kRasterVox: one workgroup per 16×16 tile (static — every tile runs the K passes;
    // an empty tile's count==0 ⇒ zero rounds, but it still pre-seeds+flushes cheaply).
    dispatch(renderer, kRasterVox);
  };

  const readWriteCount = async (renderer: Renderer): Promise<number> => {
    const buf = await readBuffer(renderer, atomicBufAttr, WRITE_CTR * 4, 4);
    return new Uint32Array(buf)[0] ?? 0;
  };

  return { dispatchVoxel, readWriteCount, tilesX, tilesY, nTiles };
}
