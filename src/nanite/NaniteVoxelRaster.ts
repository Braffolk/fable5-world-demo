/**
 * NaniteVoxelRaster — the voxel-brick raster (the SCAR fix; spec §6.0). ONE SCATTER
 * kernel (kVoxScatter): each voxel BLOCK directly atomicMax-elects its footprint into
 * the GLOBAL visPayloadV/visBV — EXACTLY like world1 rasterizes triangles
 * (NaniteRaster.ts:937-959). NO per-tile bin, NO K near→far passes, NO on-chip
 * wgElect/flush-merge. The rigorous sweep proved the depth-bucketed bin's SETUP FLOOR —
 * not the leaf-overdraw it replaces — is what net-LOSES at every config (the SCAR §6.0):
 * it is the SAME bin/K-pass mechanism that was REFUTED + DELETED for triangles (eecf046),
 * so the bin voxel path was REFUTED + REMOVED too. For small opaque sub-pixel blocks the
 * per-pixel overlap is low, so direct scatter is cheap and the bin's early-skip is
 * unnecessary overhead. The atomicMax handles depth (nearest block wins, order-free +
 * loss-exact, exactly as world1's unordered triangle scatter). A coarse PER-BLOCK
 * OCCLUSION CULL (?voxoccl=1, one global depth read at the projected centre) replaces the
 * deep-canopy occluded blocks' early-skip — one read per block, not per footprint pixel.
 *
 * SPATIAL UNIT = the CLUSTER (≤128-brick BLOCK): the cluster sphere (word0-3) bounds the
 * block AABB and the per-brick records (word6 brickBase / word7 brickCount) give each
 * brick's own grid cell. The election id is the qVoxRaster item index (brick range
 * recovered in the resolve via the cluster's word6/7). A block covers ONE or SEVERAL
 * pixels — coarse, NOT a 1-write/pixel model (§6.1).
 *
 * BIT BUDGET (§4.5): the election WORD is `depthKey24(24) | id8`; the full id is plain-
 * stored into visBV by the winner. The voxel id carries bit31=1 (the only bit free at
 * both the 128- and 256-tri caps) + the qVoxRaster item index in the low ≤30 bits.
 *
 * ≤10 STORAGE BUFFERS (the hard constraint): kVoxScatter drops tri-only verts/indices/
 * hfVerts and binds clusters/meshes/instances/voxelBricks/qVoxRaster + visPayloadV/visBV.
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
  uint,
  vec3,
  vec4,
  workgroupArray,
  workgroupBarrier,
} from 'three/tsl';
import type { NF, NU, NV3, NV4 } from '../gpu/TSLTypes';
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

// PAYLOAD bit budget (§4.5): the election id's low bits hold the qVoxRaster item index;
// PAYLOAD_MASK clamps it below BUCKET_SHIFT (so the index always fits — QVOX_CAP ≪ 1<<28).
const BUCKET_SHIFT = 28;
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

// BUILD-TIME BIT-BUDGET ASSERT (§4.5): the max payload (a qVoxRaster item index, < QVOX_CAP)
// MUST fit below BUCKET_SHIFT so PAYLOAD_MASK never truncates the index. QVOX_CAP=2M ≪ 1<<28.
{
  if (QVOX_CAP - 1 >= 1 << BUCKET_SHIFT) {
    throw new Error(
      `NaniteVoxelRaster: payload cap ${QVOX_CAP - 1} >= 1<<BUCKET_SHIFT (${1 << BUCKET_SHIFT}) — index would truncate`,
    );
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
  /** dispatch the scatter voxel raster (call AFTER world1+hwRender so the per-block
   *  occlusion cull reads the global near-field triangle winners, §6.6). */
  dispatchVoxel: (renderer: Renderer) => void;
  /** [WRITE_CTR] = per-pixel BRICK-WRITE counter (the Stage-2 overdraw-overlay number,
   *  §A2): one increment per election win. Read AFTER dispatchVoxel. */
  readWriteCount: (renderer: Renderer) => Promise<number>;
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

  // The ?voxraster flag (scatter vs the REFUTED depth-bucketed bin) is gone: scatter is the
  // ONLY voxel raster (the bin path lost the frame at every config and was removed). The param
  // is still read defensively so a stale ?voxraster=bin URL is a harmless no-op (scatter runs).
  new URLSearchParams(window.location.search).get('voxraster');

  // ---- atomicBuf: a single debug BRICK-WRITE counter (Stage-2 overdraw overlay, §A2).
  // One word [WRITE_CTR]; scatter atomicAdds it per election win and kClearWrite zeroes it.
  const WRITE_CTR = 0;
  const atomicWords = 1;
  const atomicBufAttr = new StorageBufferAttribute(new Uint32Array(atomicWords), 1);
  const atomicBuf = sU32Views(atomicBufAttr, atomicWords);

  // ---- kClearWrite: zero the per-pixel brick-write counter before kVoxScatter ----------
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

  // ---- kVoxScatter (the voxel raster — the SCAR fix) ----------------------------------
  // ONE thread = ONE voxel CLUSTER work-item from qVoxRaster (the cull fanout). Each block
  // DIRECTLY atomicMax-elects its footprint into the GLOBAL visPayloadV/visBV — VERBATIM the
  // world1 triangle election (NaniteRaster.ts:947-960): read prevE, gate cand>prevE,
  // atomicMax(visPayloadV), and the WINNER atomicStore's the full id into visBV. NO per-tile
  // bin, NO K near→far passes, NO on-chip wgElect/flush-merge — those were the depth-bucketed
  // bin's SETUP FLOOR (the SCAR §6.0) the rigorous sweep proved is the net-loss (now removed).
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
    // zero the debug WRITE_CTR (one small dispatch), then ONE scatter kernel directly elects
    // each voxel block's footprint into the global vis buffer (the per-block occlusion cull
    // reads the near-field triangle winners already in visPayloadV from world1/hwRender).
    dispatch(renderer, kClearBins);
    dispatchIndirect(renderer, kVoxScatter as never, voxRasterDispatchAttr);
  };

  const readWriteCount = async (renderer: Renderer): Promise<number> => {
    const buf = await readBuffer(renderer, atomicBufAttr, WRITE_CTR * 4, 4);
    return new Uint32Array(buf)[0] ?? 0;
  };

  return { dispatchVoxel, readWriteCount };
}
