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
import type { Renderer, StorageBufferNode } from 'three/webgpu';
import { Vector4 } from 'three';
import {
  Fn,
  If,
  atomicAdd,
  atomicMax,
  atomicStore,
  countOneBits,
  float,
  instanceIndex,
  uint,
  vec3,
  vec4,
  workgroupArray,
  workgroupBarrier,
} from 'three/tsl';
import type { NB, NF, NU, NV3, NV4 } from '../gpu/TSLTypes';
import { CLUSTER_WORDS } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import { BRICK_ALBEDO, BRICK_DIM, BRICK_HALF, BRICK_OCC_HI, BRICK_OCC_LO, BRICK_POS_X, BRICK_WORDS, MAX_BRICKS_PER_CLUSTER } from './VoxelBrick';
import { DISPATCH_ROW, QVOX_CAP } from './NaniteCommon';
import type { NaniteCam } from './NaniteCommon';
import { instTransformPoint, instYaw, instSphereRadius } from './NaniteCommon';
import {
  aLoadU,
  bcU2F,
  dispatch,
  dispatchBatchMixed,
  dispatchIndirect,
  elemU,
  loopI,
  loopU,
  localX,
  maxI,
  minI,
  minU,
  dispatchBatch,
  readBuffer,
  sU32Views,
  setIndirectDispatch,
  toF,
  toI,
  uniformArrV4,
  uniformF,
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
  /** DEPTH-BUCKET F2B (?voxf2b): per-bucket (base,count) the K near→far scatter
   *  instances read (STORAGE, cull.voxBucketRangeRO). */
  voxBucketRangeRO: BufOf<UV2>;
  /** DEPTH-BUCKET F2B: per-bucket 2D-split indirect dispatch args (cull.voxBucketDispatchAttr,
   *  length = voxF2bK), ordered NEAR→FAR (index 0 = nearest). */
  voxBucketDispatchAttr: IndirectStorageBufferAttribute[];
  /** DEPTH-BUCKET F2B: the BUILD-TIME bucket count K (must equal cull.voxF2bK so the K
   *  baked scatter instances line up with the K cull dispatch attrs). */
  voxF2bK: number;
  /** DEPTH-BUCKET F2B: front-to-back ordering. DEFAULT FALSE since 2026-06-26 — the K
   *  bucket dispatches are barrier-serialized and the intended per-block-cull pre-seed is
   *  never realized (the occlusion pyramid is built once, not rebuilt between buckets), so
   *  F2B measured a pure net-loss scaling with K (worst single-tree pose 25.4→10.8 ms gpuWall,
   *  42→121 fps with it OFF). The default is now the single unordered whole-list dispatch;
   *  ?voxf2b=1 restores the old K-bucket path (A/B control / future large-batch opt-in). */
  voxF2bEnabled: boolean;
  /** the SAME 24-bit depth key the world1 raster + resolve use (§6.7). */
  depthKey24: (cz: NF) => NU;
  /** global election buffers (the on-chip wgElect flushes here via atomicMax). The
   *  per-block occlusion cull also needs a NON-atomic read-only view of payloadV to BUILD
   *  the min-pooled footprint pyramid (a reduction, no atomicity needed); the full
   *  sU32Views bundle is passed at the call site so `.ro` is present at runtime. */
  visPayloadV: { atomic: AtomicBuf; ro: StorageBufferNode<'uint'> };
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
  const { voxBucketRangeRO, voxBucketDispatchAttr, voxF2bK, voxF2bEnabled } = deps;

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

  // ?voxlod (G1, DEFAULT ON). When ON, the per-brick OCCUPANCY GATE (Rank-2) is compiled into
  // Phase A/B: a COARSE brick (whose tight [center+-half] cube still has empty interior between its
  // sparse occupied children) paints ONLY the screen buckets that contain an occupied 4x4x4 sub-
  // cell, NOT the full projected box. This is the OVER-COARSENING/blob guard for the band-anchored
  // pyramid: Phase B strides the FULL [0, bbW*bbH) box and elects every winning pixel, so a coarse
  // box's footprint AREA — not its brick COUNT — sets the cost, and an unguarded coarse box paints
  // a SOLID blob. The build (downsampleBrickGrid) re-bins each coarse brick's occLo/occHi into its
  // tight [center+-half] cube precisely so this gate skips the empty interior => a coarse brick
  // paints only the OCCUPIED SILHOUETTE (O(occupied), never a blob). NOTE: this is the architecture-
  // compatible realization of the spec's "occupancy DDA march" — the EXISTING election uses a single
  // per-BRICK depth key `cand` with front-to-back per-brick bucket ordering, so a per-PIXEL DDA depth
  // would fight that ordering; the Phase-A occupancy MASK (projects occupied cells, bit-tested per
  // pixel in Phase B) gives the same occupancy-silhouette result against THIS architecture, reads
  // BRICK_OCC_LO/HI, adds no storage buffer / no 64-bit atomic / no wave op, and is no-hole-validated.
  // ?voxlod=0 NEVER compiles the gate (mask arrays null, Phase-A build + Phase-B branch build-time
  // absent) => BYTE-IDENTICAL to the single-level path.
  const voxLod = new URLSearchParams(window.location.search).get('voxlod') !== '0';
  // ?voxocc=0 forces the OLD solid-AABB election (disables JUST the occupancy gate while keeping the
  // pyramid/DAG cut) — the A/B control isolating the gate's silhouette/blob-guard contribution.
  const voxOccGate = voxLod && new URLSearchParams(window.location.search).get('voxocc') !== '0';
  // Footprint area (px) below which the occupancy gate is a no-op skip: a screen-tiny brick
  // (<= this many px) has no meaningful empty interior to remove and the 4x4-bucket mask cannot
  // beat painting it solid, so the gate only ARMS on bricks large enough for the interior waste
  // to dominate. Bricks below it paint solid exactly as voxlod=0 (no behaviour/quality change at
  // the FINE/NEAR end — near voxels stay a few px, never shrink toward 1px). The mask grid is
  // OCC_MASK_DIM x OCC_MASK_DIM buckets over the bbox; the gate arms once the box exceeds one
  // bucket per cell so each mask bit covers >=1 px.
  const OCC_MASK_DIM = 4; // 4x4 screen buckets over the footprint bbox (fits a 16-bit mask)
  const OCC_GATE_MIN_AREA = OCC_MASK_DIM * OCC_MASK_DIM; // 16 px — one bucket per cell at threshold
  // occupancy popcount above which the mask build is skipped (paint solid): a brick filling
  // > 75% of its 64 cells carves too little to pay the ≤512-projection build ⇒ net loss. 48/64.
  const OCC_MASK_FULL = 48;

  // The ?voxraster flag (scatter vs the REFUTED depth-bucketed bin) is gone: scatter is the
  // ONLY voxel raster (the bin path lost the frame at every config and was removed). The param
  // is still read defensively so a stale ?voxraster=bin URL is a harmless no-op (scatter runs).
  new URLSearchParams(window.location.search).get('voxraster');

  // ---- atomicBuf: a single debug BRICK-WRITE counter (Stage-2 overdraw overlay, §A2).
  // One word [WRITE_CTR]; scatter atomicAdds it per election win and kClearWrite zeroes it.
  // DEFAULT OFF: that per-win atomicAdd targets ONE global word, so every election win on the
  // GPU serializes on a single cache line (cross-core ping-pong). At a close single tree that is
  // ~1.5-2.1M wins/frame all contending on one address — measured as the dominant cost of the
  // close-up 120→30fps cliff (wf wlbc8kgla, 2026-06-26). Build-time flag ⇒ when off, the atomicAdd
  // node is never built (byte-identical removal). ?voxwrites=1 restores the counter for debugging.
  const voxWrites = new URLSearchParams(window.location.search).get('voxwrites') === '1';
  // ?voxrdbg=2 — MEASUREMENT ablation (default 0/OFF), mirror of world1's ?rdbg=2: BUILD-TIME
  // STOP point right before the Phase-B per-footprint-pixel election loop. Phase A still runs
  // (decode + project + clamp + occ-mask build per brick), but NO pixel is ever elected into
  // visPayloadV/visBV. (baseline − voxrdbg2) gpuWall = the voxel-scatter per-pixel FILL share —
  // the ONLY isolating ablation for the forcevox=all canopy fill the prior pass could not split.
  const voxRdbg = Number(new URLSearchParams(window.location.search).get('voxrdbg') ?? '0');
  // ?voxrecip=0 disables — DEFAULT ON (FIX): in Phase B, derive the per-pixel (lx,ly) from a
  // per-BRICK float reciprocal (invW = 1/bbW) instead of a per-PIXEL integer div+mod. Apple/Metal
  // has no native integer divide — `localPx % bbW` / `localPx / bbW` each lower to a microcoded
  // multi-instruction sequence, paid on EVERY footprint fragment (tens of millions/frame at the
  // canopy poses). The reciprocal is computed ONCE per brick; per pixel it costs a float mul +
  // floor + an int mul-sub. Measured −8.8ms at the worst 200k pose (49.1→40.3ms). Loss-EXACT
  // (bit-identical, not just within-noise): ly = floor((localPx+0.5)·invW) reproduces the integer
  // quotient for ALL localPx∈[0,area) — fp32 error in the product is ≤ area·2⁻²³ ≈ 1.5e-5, far
  // below the +0.5 bias's 0.5/bbW ≥ 0.004 rounding margin — so px and election are identical.
  // ?voxrecip=0 restores the int div+mod path (A/B control).
  const voxRecip = new URLSearchParams(window.location.search).get('voxrecip') !== '0';
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

  // ── MIN-POOLED FOOTPRINT PYRAMID over THIS-frame visPayloadV (the per-block occlusion
  // cull's conservative occluder structure). REPLACES the old sparse FP_TAPS×FP_TAPS raw
  // grid: a fixed N-tap sample of a wide footprint box is NOT conservative — as the block
  // sphere grows (up to 128 bricks ⇒ rPx is tens-to-hundreds of px) the inter-tap gaps are
  // large, so a VISIBLE pixel (empty key 0 / a see-through canopy gap) BETWEEN taps is
  // missed, occK overestimates the nearest occluder, and the cull can fire on a block that
  // has a visible pixel = a HOLE (the FIX-review blocker). A min/max-POOLED pyramid is the
  // only structure that conservatively tests a WHOLE box at any size: it aggregates EVERY
  // texel (no gaps), and a mip level is picked so the footprint fits a 2×2 window — exactly
  // the NaniteHzb.sphereOccluded pattern, but:
  //   • MIN-pooled (not max): the election packs depthKey24(cz)<<8|id8 and is atomicMax, so
  //     NEARER ⇒ LARGER key. The conservative occluder over a window is the SMALLEST key =
  //     FARTHEST occluder surface = the most see-through pixel = the EASIEST pixel for the
  //     block to win. Cull iff bNearKey ≤ that min ⇒ even the block's nearest point is behind
  //     the FARTHEST occluder anywhere it projects ⇒ zero visible pixels (loss-exact). Any
  //     EMPTY texel (key 0, the kVisClear sentinel) drives the window min to 0 ⇒ KEEP — a
  //     block peeking through ANY canopy gap is never culled (no holes). MIN, and a pyramid
  //     that pools EVERY pixel (not a sparse sample), are jointly load-bearing for no-holes.
  //   • SOURCED FROM FRESH visPayloadV (THIS frame's SW+HW triangle election; dispatchVoxel
  //     runs AFTER hwRender per §6.6), NOT the HZB. The HZB at voxel-raster time holds LAST
  //     frame's depth (built later in NaniteFrame) and would downgrade the occluder + add
  //     prev-VP disocclusion-hole risk.
  //   • NO Y-FLIP — visPayloadV rows are bottom-up (SW raster s.y=(ndc.y+1)/2·H); the centre
  //     mapping bndc.y·0.5+0.5 is copied verbatim (a texture-style Y-flip would over-cull).
  // INITIAL fill = 0 ⇒ "all see-through" ⇒ KEEP everything before the first build (frame-0 /
  // resize pass-through, the safe direction). The pyramid is built ONLY when voxOccl (no
  // extra cost under ?voxoccl=0). ≤2 storage buffers per build kernel; the cull reads ONE
  // extra ro view (voxOccPyr) in kVoxScatter — still within the ≤10-buffer cap (§4.6).
  const VOX_PYR_LEVELS = 16;
  const pyrLevels: { offset: number; w: number; h: number }[] = [];
  {
    let lw = Math.max(1, Math.ceil(width / 2));
    let lh = Math.max(1, Math.ceil(height / 2));
    let off = 0;
    while (pyrLevels.length < VOX_PYR_LEVELS) {
      pyrLevels.push({ offset: off, w: lw, h: lh });
      off += lw * lh;
      if (lw === 1 && lh === 1) break;
      lw = Math.max(1, Math.ceil(lw / 2));
      lh = Math.max(1, Math.ceil(lh / 2));
    }
  }
  const pyrLevelCount = pyrLevels.length;
  const pyrTotal = pyrLevels.reduce((a, l) => a + l.w * l.h, 0);
  // level lookup table (offset, w, h, 0) — clamped pad to VOX_PYR_LEVELS for a fixed uniform.
  const pyrTable = uniformArrV4(
    Array.from({ length: VOX_PYR_LEVELS }, (_, k) => {
      const l = pyrLevels[Math.min(k, pyrLevelCount - 1)] as { offset: number; w: number; h: number };
      return new Vector4(l.offset, l.w, l.h, 0);
    }),
  );
  const pyrLevelCountU = uniformF(pyrLevelCount);
  // ALL-SEE-THROUGH (key 0) initial fill ⇒ before the first build nothing occludes ⇒ KEEP.
  const voxOccPyrAttr = new StorageBufferAttribute(new Uint32Array(Math.max(1, pyrTotal)), 1);
  const voxOccPyr = sU32Views(voxOccPyrAttr, Math.max(1, pyrTotal));
  // per-level 2×2-MIN reduction kernels (built only when the cull is on). Level 0 reduces
  // full-res visPayloadV.ro; level k reduces level k−1 THROUGH THE SAME rw view (a 2nd ro
  // view of one buffer in one dispatch is a same-scope usage violation — same as NaniteHzb).
  const voxPyrKernels: unknown[] = [];
  if (voxOccl) {
    for (let k = 0; k < pyrLevelCount; k++) {
      const info = pyrLevels[k] as { offset: number; w: number; h: number };
      const kn = Fn(() => {
        const lw = uint(info.w);
        const lh = uint(info.h);
        If(instanceIndex.lessThan(lw.mul(lh)), () => {
          const x = instanceIndex.mod(lw);
          const y = instanceIndex.div(lw);
          const sx = x.mul(uint(2));
          const sy = y.mul(uint(2));
          // seed MAX so the running reduction is a true MIN over the 2×2 window.
          const m = uint(0xffffffff).toVar();
          if (k === 0) {
            const sw = uint(width - 1);
            const sh = uint(height - 1);
            for (let dy = 0; dy < 2; dy++) {
              for (let dx = 0; dx < 2; dx++) {
                const tx = minU(sx.add(uint(dx)), sw);
                const ty = minU(sy.add(uint(dy)), sh);
                const e = elemU(visPayloadV.ro, ty.mul(uint(width)).add(tx)).toVar();
                m.assign(minU(m, e));
              }
            }
          } else {
            const srcL = pyrLevels[k - 1] as { offset: number; w: number; h: number };
            const srcW = uint(srcL.w);
            const swMax = uint(srcL.w - 1);
            const shMax = uint(srcL.h - 1);
            for (let dy = 0; dy < 2; dy++) {
              for (let dx = 0; dx < 2; dx++) {
                const tx = minU(sx.add(uint(dx)), swMax);
                const ty = minU(sy.add(uint(dy)), shMax);
                const e = elemU(voxOccPyr.rw, uint(srcL.offset).add(ty.mul(srcW)).add(tx)).toVar();
                m.assign(minU(m, e));
              }
            }
          }
          (voxOccPyr.rw.element(uint(info.offset).add(y.mul(lw)).add(x)) as unknown as { assign(v: NU): void }).assign(m);
        });
      })().compute(info.w * info.h, [64]);
      (kn as { setName(n: string): unknown }).setName(`nanVoxOccPyrL${k}`);
      voxPyrKernels.push(kn);
    }
  }
  // ── COOPERATIVE-RASTER WORKGROUP MEMORY (the close-up overdraw-imbalance fix) ──────
  // Per-cluster shared brick records (Phase A fills, Phase B consumes). MAX_BRICKS_PER_
  // CLUSTER=128 entries each. wgCand packs the brick's election key (depthKey24<<8|id8);
  // bbX0/bbY0/bbW/bbH are the brick's CLAMPED screen bbox; wgBrickAbs[b]/wgDensBits[b] are
  // only consumed in dither mode. ~6×128×4B ≈ 3 KB workgroup memory — comfortable.
  const WG_RASTER = MAX_BRICKS_PER_CLUSTER; // 128 lanes per cluster workgroup
  // ── DEPTH-BUCKET F2B: the scatter body is a FACTORY closed over a per-dispatch RANGE.
  // getRange() yields this dispatch's absolute itemIdx + the live/guard predicate. The
  // WHOLE-LIST instance (F2B off) reads qVoxRaster[0].x; each per-bucket instance (F2B
  // on) reads voxBucketRange[b]=(base_b,count_b) from STORAGE (NOT a CPU uniform — all K
  // share ONE submit) and offsets itemIdx by base_b. EVERYTHING below the range hook is
  // BYTE-IDENTICAL across instances (election math, Phase-A/B cooperative loop, per-block
  // occlusion cull, dither path) — only the input slice changes.
  const makeVoxScatter = (getRange: () => { itemIdx: NU; guard: NB }) => Fn(() => {
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
    // RANGE HOOK (the ONLY F2B-vs-legacy difference): absolute itemIdx + the live guard.
    const { itemIdx, guard } = getRange();
    const brickLocal = localX().toVar(); // Phase A: this lane's brick index within the cluster
    // shared per-brick records (Phase A → Phase B). Plain (non-atomic) workgroup arrays.
    const wgBbX0 = workgroupArray('uint', WG_RASTER);
    const wgBbY0 = workgroupArray('uint', WG_RASTER);
    const wgBbW = workgroupArray('uint', WG_RASTER);
    const wgBbH = workgroupArray('uint', WG_RASTER);
    const wgCand = workgroupArray('uint', WG_RASTER); // depthKey24<<8 | id8 (loss-exact key)
    const wgBrickAbs = voxDither ? workgroupArray('uint', WG_RASTER) : null; // dither salt
    const wgDensBits = voxDither ? workgroupArray('uint', WG_RASTER) : null; // dither raw density byte (0..255)
    // OCCUPANCY GATE (?voxlod=1): per-brick 16-bit (OCC_MASK_DIM x OCC_MASK_DIM) SCREEN mask of
    // which footprint-bbox buckets contain a projected occupied 4x4x4 sub-cell. Phase A builds it
    // by projecting each occupied cell centre into the brick's clamped bbox + DILATING by +-1
    // bucket (the no-hole guard against projection rounding); Phase B paints a pixel only if its
    // (su,sv) bucket bit is set. 0xffff (all set) for a brick the gate did not arm (small/full) or
    // when voxlod=0 (the array is null and the gate code is build-time absent => byte-identical).
    const wgOccMask = voxOccGate ? workgroupArray('uint', WG_RASTER) : null;
    If(guard, () => {
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
      // PYRAMID LEVEL (word7 bits 10-15; 0 = finest/L0, higher = coarser) — read ONLY under
      // ?voxlod=1 (build-time absent otherwise ⇒ voxlod=0 byte-identical). The occupancy gate
      // ARMS only on COARSE bricks (dagLevel>0), the levels the DAG cut emits at distance: L0
      // (near) paints SOLID (no mask build, no near carving, fine-end rule intact) AND the per-cell
      // mask build (≤512 projections) is confined to the few coarse bricks with empty interior to
      // remove, so the build cost can never exceed the elections it saves on the dense near crown.
      const dagLevel = voxOccGate
        ? elemU(gpu.clusters, cBase.add(uint(7))).shiftRight(uint(10)).bitAnd(uint(0x3f)).toVar()
        : null;
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
      // workgroup-shared flag, then broadcast via a barrier. CONSERVATIVE FOOTPRINT test via a
      // MIN-POOLED PYRAMID (voxOccPyr, built fresh THIS frame above): pick the mip level whose
      // 2×2 window covers the block's projected footprint, take the MIN election key (the
      // farthest-conservative occluder) over that window, and cull iff the block's NEAREST-
      // possible key (its AABB front-slab) is at-or-behind it. This REPLACES the prior sparse
      // FP_TAPS×FP_TAPS RAW grid, which was NOT conservative: a fixed N-tap sample of a wide box
      // (rPx grows to tens-to-hundreds of px for a ≤128-brick block) leaves large inter-tap gaps,
      // so a VISIBLE pixel (empty key 0 / a see-through canopy gap) between taps is missed and the
      // cull can drop a block that has a visible pixel = a HOLE. The pooled pyramid aggregates
      // EVERY texel under the window (no gaps) ⇒ conservative at ANY footprint size.
      //
      // CONSERVATIVE (NEVER drops a block with ANY visible pixel). Polarity: the election packs
      // depthKey24(cz)<<8|id8 and is atomicMax ⇒ NEARER = LARGER key. occK = MIN over the 2×2
      // window = SMALLEST key = FARTHEST occluder surface = the most see-through pixel the
      // footprint covers (the EASIEST pixel for the block to win). bNearKey = the block front
      // slab's key (the LARGEST key any block pixel could elect; |0xff id-tiebreak ⇒ keep-on-tie).
      // Cull iff bNearKey ≤ occK ⇒ even the block's nearest point is behind the FARTHEST occluder
      // anywhere it projects ⇒ zero visible pixels (loss-exact). Any EMPTY texel under the window
      // (key 0, the kVisClear/initial-fill sentinel) drives the pooled min to 0 ⇒ block KEPT — a
      // block peeking through ANY canopy gap is never culled (no holes). MIN-pooling + a pyramid
      // that pools EVERY pixel are jointly load-bearing for that no-hole property. NO Y-FLIP
      // (visPayloadV rows are bottom-up; the centre mapping bndc.y·0.5+0.5 is verbatim — a
      // texture-style Y-flip would over-cull). ?voxoccl=0 disables it for the A/B; with NO param
      // the upgraded footprint cull is the DEFAULT production path.
      const wgVisible = workgroupArray('uint', 1);
      if (voxOccl) {
        If(brickLocal.equal(uint(0)), () => {
          wgSet(wgVisible, uint(0), uint(1));
          const bp = (cam.vp.mul(vec4(blkWCenter, 1)) as unknown as NV4).toVar();
          If(bp.w.greaterThan(float(NEAR_EPS)), () => {
            const bndc = bp.xyz.div(bp.w).toVar();
            const bnz = bndc.z.sub(blkWR.div(bp.w)).clamp(0, 1).toVar();
            const bNearKey = depthKey24(bnz as unknown as NF).shiftLeft(uint(8)).bitOr(uint(0xff)).toVar();
            // FULL-RES projected pixel-radius of the block sphere (world→pixel scale = cotHalfFov·
            // H/2; bp.w is the view-space distance for a standard proj ⇒ rPx = blkWR·cotHalfFov·H/
            // (2·bp.w)). Level 0 of voxOccPyr is HALF-res, so a full-res radius rPx maps to rPx/2
            // texels at level 0; the level whose 2×2 window COVERS the footprint diameter (2·rPx
            // full-res) is ceil(log2(max(1, rPx))) — exactly NaniteHzb.ts:176's
            // radiusTexels·2 .max(1).log2().ceil() with radiusTexels = rPx/2. Clamp to the valid
            // level range. Over-coarse is SAFE (a larger window only pools more far/neighbour
            // texels ⇒ min trends toward KEEP); under-coarse is the hole risk and is precluded by
            // the ceil + the 2×2 (not 1×1) window.
            const rPx = blkWR
              .mul(cam.cotHalfFov as unknown as NF)
              .mul(H)
              .div(bp.w.mul(2))
              .toVar();
            const levelF = (rPx as unknown as { max(o: number): NF })
              .max(1)
              .log2()
              .ceil()
              .clamp(0, (pyrLevelCountU as unknown as { sub(o: number): NF }).sub(1))
              .toVar();
            const info = pyrTable.element(uint(levelF));
            const lw = uint(info.y).toVar();
            const lh = uint(info.z).toVar();
            const lo = uint(info.x).toVar();
            // projected centre → level-lvl texel coords (NO Y-FLIP, bottom-up rows). 2×2 window
            // around the centre texel (NaniteHzb.sphereOccluded pattern); the level pick guarantees
            // the footprint diameter ≤ one window edge, so this 2×2 fully covers the footprint.
            const px = bndc.x.mul(0.5).add(0.5).mul(toF(lw)).toVar();
            const py = bndc.y.mul(0.5).add(0.5).mul(toF(lh)).toVar();
            const x0 = uint((px.sub(0.5) as unknown as { clamp(a: number, b: NF): NF }).clamp(0, toF(lw.sub(uint(1))))).toVar();
            const y0 = uint((py.sub(0.5) as unknown as { clamp(a: number, b: NF): NF }).clamp(0, toF(lh.sub(uint(1))))).toVar();
            const x1 = minU(x0.add(uint(1)), lw.sub(uint(1))).toVar();
            const y1 = minU(y0.add(uint(1)), lh.sub(uint(1))).toVar();
            const z00 = elemU(voxOccPyr.ro, lo.add(y0.mul(lw)).add(x0)).toVar();
            const z01 = elemU(voxOccPyr.ro, lo.add(y0.mul(lw)).add(x1)).toVar();
            const z10 = elemU(voxOccPyr.ro, lo.add(y1.mul(lw)).add(x0)).toVar();
            const z11 = elemU(voxOccPyr.ro, lo.add(y1.mul(lw)).add(x1)).toVar();
            const occK = minU(minU(z00, z01), minU(z10, z11)).toVar();
            If(bNearKey.lessThanEqual(occK), () => {
              wgSet(wgVisible, uint(0), uint(0)); // footprint fully occluded ⇒ skip the block
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
      // OCCUPANCY-GATE seed (?voxlod=1): default 0xffff = ALL buckets painted ⇒ a brick whose
      // gate never arms (small footprint / mask stays full) behaves EXACTLY like voxlod=0 (no
      // pixel dropped). Phase A overwrites it with the real screen mask only when the gate arms.
      if (voxOccGate && wgOccMask) wgSet(wgOccMask, brickLocal, uint(0xffff));
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
            // ── OCCUPANCY-GATE MASK BUILD (?voxlod=1, Phase A). For a COARSE (dagLevel>0),
            // LARGE-footprint (area ≥ OCC_GATE_MIN_AREA) brick — i.e. screen-big enough for empty
            // interior to matter — build a OCC_MASK_DIM×OCC_MASK_DIM screen mask of which bbox
            // buckets a projected OCCUPIED 4×4×4 sub-cell overlaps, so Phase B skips the empty
            // buckets between sparse children. The build re-binned occLo/occHi into THIS brick's
            // tight [center±half] cube (downsampleBrickGrid), so each cell's LOCAL centre is
            // brLocal + (cell+0.5−BRICK_DIM/2)·cellLocalSize and its half-extent is cellLocalSize/2.
            // For each OCCUPIED cell we project its 8 LOCAL-AABB corners, take their SCREEN bbox,
            // and mark EVERY bbox bucket that rect overlaps — a CONSERVATIVE SUPERSET of the cell's
            // true coverage ⇒ a covered pixel's bucket is ALWAYS set ⇒ provably NO holes at ANY
            // perspective (validated by tools/voxlod-occgate-nohole.ts: 0 holes / 5966 armed
            // bricks). A small or L0 brick keeps the 0xffff seed (paints solid, no behaviour change;
            // near-dense crown intact, fine-end rule preserved).
            if (voxOccGate && wgOccMask && dagLevel) {
              const gateArea = bbW.mul(bbH).toVar();
              // ARM only on a COARSE brick (dagLevel>0) whose footprint is big enough for empty
              // interior to matter. A fine/near (L0) brick keeps the 0xffff seed ⇒ paints solid.
              If(gateArea.greaterThanEqual(uint(OCC_GATE_MIN_AREA)).and(dagLevel.greaterThan(uint(0))), () => {
                const occLo = elemU(gpu.voxelBricks, bWordBase.add(uint(BRICK_OCC_LO))).toVar();
                const occHi = elemU(gpu.voxelBricks, bWordBase.add(uint(BRICK_OCC_HI))).toVar();
                // SPARSITY GUARD: build the mask only when the brick is sparse enough that the
                // carved buckets save MORE elections than the ≤512-projection build costs. A
                // near-full brick (>OCC_MASK_FULL of 64 cells set) carves little, so building the
                // mask would be a net GPU LOSS — skip it and paint solid (the 0xffff seed). This is
                // the cost-model guard that keeps the gate from re-introducing a slowdown on DENSE
                // coarse bricks (the prior iterations' failure mode), measured by popcount once
                // per brick (cheap) — NOT per pixel.
                const occCount = (countOneBits(occLo) as unknown as NU)
                  .add(countOneBits(occHi) as unknown as NU)
                  .toVar();
                const sparseEnough = occCount.lessThanEqual(uint(OCC_MASK_FULL)).toVar();
                If(sparseEnough, () => {
                  const cellLocalSize = brHalf.mul(2).div(float(BRICK_DIM)).toVar(); // local edge of one cell
                  const halfDim = float(BRICK_DIM).mul(0.5).toVar();
                  const fbbW = toF(bbW).toVar();
                  const fbbH = toF(bbH).toVar();
                  const mask = uint(0).toVar();
                  // loop the 64 cells; only OCCUPIED cells contribute (occLo/occHi bit test).
                  loopI('ocz', toI(0), toI(BRICK_DIM), (czc) => {
                    loopI('ocy', toI(0), toI(BRICK_DIM), (cyc) => {
                      loopI('ocx', toI(0), toI(BRICK_DIM), (cxc) => {
                        // cell linear index = x + y*4 + z*16 (VoxelBrick.brickCellIndex order).
                        const cellIdx = (cxc as unknown as { toUint(): NU })
                          .toUint()
                          .add((cyc as unknown as { toUint(): NU }).toUint().mul(uint(BRICK_DIM)))
                          .add((czc as unknown as { toUint(): NU }).toUint().mul(uint(BRICK_DIM * BRICK_DIM)))
                          .toVar();
                        // occupied? bit cellIdx of the 64-bit occLo|occHi mask. Pick the word by
                        // the hi-bit (cellIdx≥32) and shift by cellIdx&31 — both shifts use an
                        // IN-RANGE [0,31] amount (no UB underflow shift, robust on Metal); select
                        // then keeps the LO result for cells 0..31 and the HI result for 32..63.
                        const cellLow = cellIdx.bitAnd(uint(31)).toVar();
                        const loBit = occLo.shiftRight(cellLow).bitAnd(uint(1)).toVar();
                        const hiBit = occHi.shiftRight(cellLow).bitAnd(uint(1)).toVar();
                        const occBit = cellIdx.lessThan(uint(32)).select(loBit, hiBit).toVar();
                        If(occBit.equal(uint(1)), () => {
                          // CONSERVATIVE per-cell mask: project the cell's 8 LOCAL-AABB corners,
                          // accumulate their SCREEN bbox, and mark EVERY bbox bucket the cell's
                          // screen rect overlaps. This is a SUPERSET of the cell's true coverage,
                          // so a covered pixel's bucket is ALWAYS set ⇒ provably NO holes at ANY
                          // perspective — including the close-up case where a cell projects across
                          // many buckets (the centre+dilate scheme missed those: a near cell spans
                          // > ±1 bucket and a covered pixel fell outside the dilated neighbourhood).
                          // Near a cell spans the whole bbox ⇒ marks all buckets ⇒ brick stays
                          // solid (near dense, no carving); far a cell is ~1 bucket ⇒ carves the
                          // empty interior. The cell LOCAL centre = brLocal + (cell+0.5−halfDim)·
                          // cellLocalSize; half-cell extent = cellLocalSize·0.5.
                          const clx = brLocal.x.add(toF(cxc).add(0.5).sub(halfDim).mul(cellLocalSize)).toVar();
                          const cly = brLocal.y.add(toF(cyc).add(0.5).sub(halfDim).mul(cellLocalSize)).toVar();
                          const clz = brLocal.z.add(toF(czc).add(0.5).sub(halfDim).mul(cellLocalSize)).toVar();
                          const hc = cellLocalSize.mul(0.5).toVar();
                          const cMinX = float(1e9).toVar();
                          const cMinY = float(1e9).toVar();
                          const cMaxX = float(-1e9).toVar();
                          const cMaxY = float(-1e9).toVar();
                          const cFront = uint(0).toVar();
                          loopI('ccz', toI(0), toI(2), (kz) => {
                            loopI('ccy', toI(0), toI(2), (ky) => {
                              loopI('ccx', toI(0), toI(2), (kx) => {
                                const ccl = vec3(
                                  clx.add(toF(kx).mul(2).sub(1).mul(hc)),
                                  cly.add(toF(ky).mul(2).sub(1).mul(hc)),
                                  clz.add(toF(kz).mul(2).sub(1).mul(hc)),
                                ) as unknown as NV3;
                                const ccw = instTransformPoint(A, B, yawSc, ccl);
                                const ccp = (cam.vp.mul(vec4(ccw, 1)) as unknown as NV4).toVar();
                                If(ccp.w.greaterThan(float(NEAR_EPS)), () => {
                                  cFront.assign(uint(1));
                                  const cndc = ccp.xyz.div(ccp.w).toVar();
                                  const csx = cndc.x.add(1).mul(0.5).mul(W).toVar();
                                  const csy = cndc.y.add(1).mul(0.5).mul(H).toVar();
                                  cMinX.assign(cMinX.min(csx));
                                  cMaxX.assign(cMaxX.max(csx));
                                  cMinY.assign(cMinY.min(csy));
                                  cMaxY.assign(cMaxY.max(csy));
                                });
                              });
                            });
                          });
                          If(cFront.equal(uint(1)), () => {
                            // screen bbox → INCLUSIVE bucket range, clamped to [0, OCC_MASK_DIM).
                            const fU0 = cMinX.sub(toF(startX)).div(fbbW).mul(float(OCC_MASK_DIM)).toVar();
                            const fU1 = cMaxX.sub(toF(startX)).div(fbbW).mul(float(OCC_MASK_DIM)).toVar();
                            const fV0 = cMinY.sub(toF(startY)).div(fbbH).mul(float(OCC_MASK_DIM)).toVar();
                            const fV1 = cMaxY.sub(toF(startY)).div(fbbH).mul(float(OCC_MASK_DIM)).toVar();
                            const u0 = maxI(toI(0), minI(toI(OCC_MASK_DIM - 1), toI(fU0.floor()))).toVar();
                            const u1 = maxI(toI(0), minI(toI(OCC_MASK_DIM - 1), toI(fU1.floor()))).toVar();
                            const v0 = maxI(toI(0), minI(toI(OCC_MASK_DIM - 1), toI(fV0.floor()))).toVar();
                            const v1 = maxI(toI(0), minI(toI(OCC_MASK_DIM - 1), toI(fV1.floor()))).toVar();
                            // FIXED 0..OCC_MASK_DIM loops with an in-[u0,u1]×[v0,v1] guard (constant
                            // loop bounds = the codegen-safe pattern; variable bounds avoided).
                            loopI('mbv', toI(0), toI(OCC_MASK_DIM), (vv) => {
                              loopI('mbu', toI(0), toI(OCC_MASK_DIM), (uu) => {
                                const inR = vv
                                  .greaterThanEqual(v0)
                                  .and(vv.lessThanEqual(v1))
                                  .and(uu.greaterThanEqual(u0))
                                  .and(uu.lessThanEqual(u1));
                                If(inR, () => {
                                  const bit = (vv as unknown as { toUint(): NU })
                                    .toUint()
                                    .mul(uint(OCC_MASK_DIM))
                                    .add((uu as unknown as { toUint(): NU }).toUint())
                                    .toVar();
                                  mask.assign(mask.bitOr(uint(1).shiftLeft(bit)));
                                });
                              });
                            });
                          });
                        });
                      });
                    });
                  });
                  // a degenerate all-behind-near projection could leave mask 0 ⇒ that would paint
                  // NOTHING (a hole). Guard: an empty mask falls back to 0xffff (paint solid).
                  const safeMask = mask.equal(uint(0)).select(uint(0xffff), mask).toVar();
                  wgSet(wgOccMask, brickLocal, safeMask);
                });
              });
            }
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
      // ?voxrdbg=2 STOP point: skip the ENTIRE Phase-B election (no per-pixel atomicMax / visBV
      // store). Build-time gate ⇒ when OFF (production) this whole block is emitted byte-identical.
      if (voxRdbg < 2)
      loopU(uint(0), nBricks, (b) => {
        // recover this brick's footprint RECORD (set up in Phase A).
        const bbX0 = (wgBbX0.element(b) as unknown as NU).toVar();
        const bbY0 = (wgBbY0.element(b) as unknown as NU).toVar();
        const bbW = (wgBbW.element(b) as unknown as NU).toVar();
        const bbH = (wgBbH.element(b) as unknown as NU).toVar();
        const cand = (wgCand.element(b) as unknown as NU).toVar();
        // OCCUPANCY-GATE mask (?voxlod=1): which OCC_MASK_DIM×OCC_MASK_DIM bbox buckets a
        // projected occupied sub-cell touched (Phase A). 0xffff (all) for an unarmed brick.
        const occMask = voxOccGate && wgOccMask ? (wgOccMask.element(b) as unknown as NU).toVar() : null;
        // per-brick: does the mask actually CARVE (≠ 0xffff)? Only then does Phase B pay the
        // per-pixel bucket test. A full (0xffff) mask — every L0/near/dense brick — takes the
        // plain solid path with ZERO added per-pixel arithmetic (near dense path unchanged).
        const gateActive = occMask ? occMask.notEqual(uint(0xffff)).toVar() : null;
        const area = bbW.mul(bbH).toVar(); // 0 for a culled/idle brick ⇒ zero-trip inner loop
        // ?voxrecip: ONE reciprocal per brick replaces the per-pixel integer div+mod below
        // (footprint addressing AND, for carved bricks, the occupancy-gate su/sv bucket index).
        const invW = voxRecip ? float(1).div(toF(bbW)).toVar() : null;
        const invH = voxRecip ? float(1).div(toF(bbH)).toVar() : null;
        // each lane strides this brick's footprint: localPx = brickLocal, brickLocal+128, …
        loopU(brickLocal, area, (localPx) => {
          // ly/lx: integer div+mod by default; per-brick float reciprocal under ?voxrecip
          // (loss-exact, see flag note) — Apple has no HW int-divide so the default path pays
          // a microcoded sequence on every fragment.
          const ly = (voxRecip && invW
            ? uint(toF(localPx).add(0.5).mul(invW).floor())
            : localPx.div(bbW)).toVar(); // < bbH since localPx < bbW·bbH (no column)
          const lx = (voxRecip ? localPx.sub(ly.mul(bbW)) : localPx.mod(bbW)).toVar();
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
                // debug per-pixel BRICK-WRITE counter (Stage-2 overlay, §A2). DEFAULT OFF —
                // single-global-address atomicAdd per win serializes ~1.5-2.1M wins/frame on one
                // cache line (the close-up cliff; see voxWrites note above). ?voxwrites=1 restores it.
                if (voxWrites) {
                  atomicAdd(atomicBuf.atomic.element(uint(WRITE_CTR)), uint(1));
                }
              });
            });
          };
          // OCCUPANCY GATE (?voxlod=1): paint this pixel only if its bbox bucket carries an
          // occupied sub-cell. su=lx·DIM/bbW, sv=ly·DIM/bbH ∈ [0,DIM); bit=sv·DIM+su. The mask is
          // 0xffff (all buckets) for any unarmed/full/degenerate brick ⇒ that pixel always paints
          // ⇒ no behaviour change there. The +1 dilation (Phase A) makes this provably hole-free.
          // voxlod=0: occMask is null and this whole branch is build-time absent (byte-identical).
          const dispatchElect = (): void => {
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
          };
          if (voxOccGate && occMask && gateActive) {
            // carved brick → gate the pixel by its bucket bit; full-mask brick → solid (no test).
            If(gateActive, () => {
              const su = (voxRecip && invW
                ? minU(uint(toF(lx.mul(uint(OCC_MASK_DIM))).add(0.5).mul(invW).floor()), uint(OCC_MASK_DIM - 1))
                : minU(lx.mul(uint(OCC_MASK_DIM)).div(bbW), uint(OCC_MASK_DIM - 1))).toVar();
              const sv = (voxRecip && invH
                ? minU(uint(toF(ly.mul(uint(OCC_MASK_DIM))).add(0.5).mul(invH).floor()), uint(OCC_MASK_DIM - 1))
                : minU(ly.mul(uint(OCC_MASK_DIM)).div(bbH), uint(OCC_MASK_DIM - 1))).toVar();
              const bit = sv.mul(uint(OCC_MASK_DIM)).add(su).toVar();
              const occupied = occMask.shiftRight(bit).bitAnd(uint(1)).equal(uint(1));
              If(occupied, dispatchElect);
            }).Else(dispatchElect);
          } else {
            dispatchElect();
          }
        }, WG_RASTER);
      });
    });
    // ONE WORKGROUP per voxel cluster work-item (split2D indirect args over the fanned count,
    // unchanged), WG_RASTER (=MAX_BRICKS_PER_CLUSTER) threads each. Phase A = 1 lane/brick;
    // Phase B = all lanes cooperatively rasterize the cluster's flat footprint. The baked
    // `.compute(count,[wg])` is the static fallback; the indirect args drive the real WG count.
  })().compute(DISPATCH_ROW * WG_RASTER, [WG_RASTER]);

  // WHOLE-LIST scatter (F2B OFF, ?voxf2b=0): the legacy unordered single dispatch over the
  // full qVoxRaster — itemIdx = wgLinear, guard = itemIdx < qVoxRaster[0].x. Byte-identical
  // to today's kVoxScatter for the A/B control.
  const kVoxScatter = makeVoxScatter(() => {
    const itemIdx = wgLinear(DISPATCH_ROW).toVar();
    const itemCount = qVoxRasterRO.element(0).x;
    return { itemIdx, guard: itemIdx.lessThan(itemCount) as unknown as NB };
  });
  (kVoxScatter as { setName(n: string): unknown }).setName('nanVoxScatter');

  // K PER-BUCKET scatter instances (F2B ON), ordered NEAR→FAR (bucket 0 = nearest). Each
  // closes over its bucket's (base_b, count_b) read from the voxBucketRange STORAGE buffer:
  // local = wgLinear; itemIdx = local + base_b (absolute slot); guard = local < count_b. The
  // qVoxRaster slice [base_b, base_b+count_b) is this bucket's contiguous depth slab. In-pass
  // UAV auto-sync (dispatchBatchMixed) serializes bucket b's visPayloadV/visBV writes before
  // bucket b+1's early-Z reads — so the near bricks pre-seed the far bricks' skip.
  const kVoxScatterB: unknown[] = [];
  if (voxF2bEnabled) {
    for (let b = 0; b < voxF2bK; b++) {
      const kb = makeVoxScatter(() => {
        const range = voxBucketRangeRO.element(uint(b));
        const base = range.x.toVar();
        const count = range.y.toVar();
        const local = wgLinear(DISPATCH_ROW).toVar();
        const itemIdx = local.add(base).toVar();
        return { itemIdx, guard: local.lessThan(count) as unknown as NB };
      });
      (kb as { setName(n: string): unknown }).setName(`nanVoxScatterB${b}`);
      setIndirectDispatch(kb, voxBucketDispatchAttr[b]);
      kVoxScatterB.push(kb);
    }
  }

  const dispatchVoxel = (renderer: Renderer): void => {
    // PER-BLOCK OCCLUSION CULL prerequisite: build the MIN-POOLED footprint pyramid over THIS
    // frame's visPayloadV (already holds the SW+HW triangle election — world1/hwRender ran
    // before this). ONE submit for the whole min-pool chain (level k reads level k−1; they share
    // one buffer + one compute pass ⇒ WebGPU auto-syncs between dispatches, same as NaniteHzb).
    // Built ONLY when voxOccl (no cost under ?voxoccl=0, where voxPyrKernels is empty). The
    // scatter's per-block cull below then reads voxOccPyr.ro at the footprint-covering level.
    if (voxOccl && voxPyrKernels.length > 0) dispatchBatch(renderer, voxPyrKernels);
    // zero the debug WRITE_CTR (one small dispatch), then the scatter elects each voxel
    // block's footprint into the global vis buffer (the per-block occlusion cull reads the
    // near-field triangle winners already in visPayloadV from world1/hwRender).
    if (voxF2bEnabled) {
      // K near→far dispatches in ONE submit: kClearBins then dispatchBatchMixed over the K
      // bucket kernels (each tagged with its tight per-bucket indirect args). The in-pass
      // barriers serialize the buckets near→far so the early-Z gate fires across slabs.
      dispatchBatchMixed(renderer, [kClearBins, ...kVoxScatterB]);
    } else {
      // ?voxf2b=0 — EXACTLY today: kClearBins + ONE dispatchIndirect over the whole list.
      dispatch(renderer, kClearBins);
      dispatchIndirect(renderer, kVoxScatter as never, voxRasterDispatchAttr);
    }
  };

  const readWriteCount = async (renderer: Renderer): Promise<number> => {
    const buf = await readBuffer(renderer, atomicBufAttr, WRITE_CTR * 4, 4);
    return new Uint32Array(buf)[0] ?? 0;
  };

  return { dispatchVoxel, readWriteCount };
}
