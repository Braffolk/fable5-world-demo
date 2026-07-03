/**
 * NaniteGrass — PROCEDURAL ZERO-STORAGE grass field (grass rethink 2026-07-03,
 * docs/perf-runs/2026-07-03-grass-arc.md §REBUILD).
 *
 * The GroundRing meadow (the user's lushness reference) re-expressed as a
 * nanite emit class: a fused cull+raster COMPUTE kernel walks a toroidal
 * world-cell grid around the camera each frame, re-derives every clump from
 * pcg(worldCell) (density law / thin×widen / band ladder VERBATIM from
 * GroundRing.ts), and rasters blade quads straight into the vis-buffer
 * election. NO stored geometry, NO instances, NO DAGs, NO boot cost:
 * the 30-bit election id IS the blade (slot 24b | prim 3b | seg 2b | tri 1b
 * under the 0xC0000000 namespace — bit31|bit30 is free: voxels are bit31 with
 * bits 28-30 zero, mesh ids stay below bit30 at the 128-tri cluster config).
 *
 * COST DISCIPLINE (the three per-element paths are deliberately asymmetric):
 *  - the KERNEL derives each clump once (hoisted per-prim constants, shared
 *    row corners — 2+2·segN corner evals per blade, ~40 ALU each, no selects
 *    in the hot path) and rasters small quads through the fixed-point SW
 *    scanline (the world1 election emit, cloned);
 *  - NEAR/oversized/near-plane quads append their ALREADY-COMPUTED corners to
 *    a grass HW queue (stride-10 f32) drawn inside the shared hwRender pass —
 *    the HW vertex stage is a pure unpack, zero re-derivation;
 *  - the RESOLVE reconstructs shading inputs ANALYTICALLY: tip parameter
 *    t = (wp.y − rootY)/bladeHeight and a per-prim mean normal pulled toward
 *    the terrain normal — no corner re-derivation, no barycentrics, ~10
 *    spatially-coherent texture taps per grass pixel. (t is a shading-only
 *    approximation; geometry/depth come from the raster.)
 *
 * LOD is continuous + world-space hash-jittered per clump (no screen dither,
 * no bands, no checkerboard by construction): thin×widen conserves coverage;
 * 5→3 blades ~30 m, segs 4→2→1, blades→crossed cards ~70 m, coarse-grid
 * super-tufts 150-265 m, terrain splat beyond.
 */

import { DoubleSide, Mesh, Scene, Sphere, Vector3 } from 'three';
import { BufferGeometry, Float32BufferAttribute, RenderTarget } from 'three';
import type { PerspectiveCamera } from 'three';
import {
  IndirectStorageBufferAttribute,
  NodeMaterial,
  StorageBufferAttribute,
  type Renderer,
  type StorageTexture,
} from 'three/webgpu';
import { tagGpu } from '../core/GpuProfiler';
import {
  Break,
  Fn,
  If,
  atomicAdd,
  atomicMax,
  atomicStore,
  float,
  instanceIndex,
  mix,
  normalize,
  positionGeometry,
  screenCoordinate,
  smoothstep,
  texture,
  time,
  uint,
  varyingProperty,
  vec2,
  vec3,
  vec4,
  vertexIndex,
} from 'three/tsl';
import type { NB, NF, NI, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import { canopyAt, cellHash, cellHash2 } from '../gpu/passes/Scatter';
import { gustAt, windContext, windExposure, windU } from '../render/Wind';
import { terrainDispAt, type TerrainDisp } from './NaniteFetch';
import type { Heightfield } from '../world/Heightfield';
import { WORLD_SIZE } from '../world/WorldConst';
import type { NaniteCam } from './NaniteCommon';
import type { NaniteVisBuffers } from './NaniteRaster';
import {
  aLoadU,
  bcF2U,
  bcU2F,
  dispatch,
  elemU,
  loopI,
  loopUN,
  maxI,
  minI,
  readBuffer,
  returnIf,
  sU32Views,
  toF,
  toI,
  uniformF,
} from './Tsl';

// ---- constants (GroundRing parity — the tuned reference) ---------------------------
const GRID = 3072;
const CELL = 0.105; // m → ±161 m ring, ~90 slots/m²
const R = 155;
const FAR_GRID = 768;
const FAR_CELL = 0.7; // ±269 m coarse super-tuft ring
const FAR_R0 = 150;
const FAR_R = 265;
const SALT = 0x51a55e & 0x7fffffff;
/** election id namespace: bit31|bit30 (voxel = bit31 only, mesh < bit30) */
const GRASS_FLAGS = 0xc0000000;
/** far-tuft ids live above this body offset (fine max = GRID²·64 ≈ 604M).
 *  Exported for the resolve's far-pixel cheap path. */
export const GRASS_FAR_BASE = 0x28000000; // 671M
const FAR_BASE = GRASS_FAR_BASE;
// SW scanline bbox bound (px) — MEASURED 2026-07-03 (grass2-swtest, eye meadow
// dpr1.5): the compute scanline is ~10× costlier per fragment than the HW pass
// (one lane walks pixels serially vs packed hardware raster; both write the same
// election). swpx 4→16→24 ⇒ grass compute 3.4→15.7→18 ms while the HW pass only
// +1.1→+0.6→+0.1. DEFAULT 4: SW keeps only sub-4px slivers (where HW quad-occupancy
// waste + fixed per-prim cost dominate); everything bigger rides the HW queue.
// ?grassswpx=N A/B knob (≤48 stays exact i32 — (48·256)² < 2^31).
const MAX_SW_PX = ((): number => {
  const v = Number(new URLSearchParams(window.location.search).get('grassswpx') ?? '4');
  return Number.isFinite(v) && v >= 2 && v <= 48 ? v : 4;
})();
/** ?grassdbg=funnel|corners — BUILD-TIME kernel stop points for cost attribution:
 *  funnel = cull chain only (no clump derive/raster); corners = full derive +
 *  corner math + projection, no scanline/election. Production pristine when unset. */
const GRASS_DBG = new URLSearchParams(window.location.search).get('grassdbg');
/** lane select: ?grass=ray → per-pixel analytic raycast (zero memory/emission,
 *  best-measured 15.1 ms @ dpr1 eye — NOT yet at the ≤2 ms mandate);
 *  ?grass=1|geo → the emission+HW-queue lane (+13 ms eye @ dpr1.5).
 *  DEFAULT OFF until a lane meets the mandate (user law: ≤2 ms worst case). */
const GEO_LANE = new URLSearchParams(window.location.search).get('grass') !== 'ray';
/** raycast bands: fine-stride march to RAY_L0_END, coarse-stride (16 fine sub-cells
 *  per step) to RAY_END; terrain splat beyond. Step caps bound the grazing worst case
 *  (a capped miss falls through to the splat — the correct infinite-distance limit). */
const RAY_L0_END = 25;
/** ?grassrayend=N — band-end knob for cost attribution (default 155) */
const RAY_END = ((): number => {
  const v = Number(new URLSearchParams(window.location.search).get('grassrayend') ?? '155');
  return Number.isFinite(v) && v >= 20 && v <= 300 ? v : 155;
})();
const RAY_SHELL_H = 1.5; // max blade reach above ground (incl. mid-card 2× + wind)
const NEAR_EPS = 1e-4;
/** grass HW queue capacity (per-TRI entries, stride 10 u32: body + 3 packed f32
 *  corners). With the 4-px SW bound nearly ALL visible blade tris ride this queue —
 *  the 786k cap measured SATURATED at an eye meadow (dropped tris). 1.5M × 40 B =
 *  60 MB; revisit with a packed u16 corner encoding if memory ever matters. */
const HW_CAP = 1_572_864;
const HW_STRIDE = 10;

/** continuous distance thinning, conserved by widening (GroundRing verbatim) */
function grassThin(dist: NF): NF {
  const base = float(58).div(dist.max(1).add(42)).min(1).pow(1.15);
  const far = float(120).div(dist.max(120)).pow(1.6);
  return base.mul(far);
}

// ---- JS-constant blade/card tables --------------------------------------------------
// bladeClump(5,4)'s mini-LCG is seed-fixed ⇒ the 5 blades' in-clump params are
// CONSTANTS (per-clump variety comes from the slot hash yaw/tilt, as shipped).
interface BladePar {
  c: number;
  s: number;
  ox: number;
  oz: number;
  hk: number;
  lean: number;
  /** mean rounded-cross-section normal (L+R average), blade-yaw-rotated — the
   *  resolve's shading normal (per-side curvature is sub-pixel at the widths
   *  grass renders at; the terrain pull dominates from 8 m out anyway) */
  nm: [number, number, number];
}
function bladeTable(blades: number, segs: number): BladePar[] {
  let s = 1234567 + blades * 77 + segs * 13;
  const rnd = (): number => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const CS = 0.788;
  const out: BladePar[] = [];
  for (let b = 0; b < blades; b++) {
    const yaw = rnd() * Math.PI * 2;
    const c = Math.cos(yaw);
    const sn = Math.sin(yaw);
    const ox = (rnd() - 0.5) * 0.16;
    const oz = (rnd() - 0.5) * 0.16;
    const hk = 0.62 + rnd() * 0.65;
    const lean = (rnd() - 0.5) * 0.42;
    // mean of (±SN, .25, −CS) = (0, .25, −CS), yaw-rotated, ~normalized
    const l = Math.hypot(0.25, CS);
    out.push({
      c,
      s: sn,
      ox,
      oz,
      hk,
      lean,
      nm: [(-CS * sn) / l, 0.25 / l, (-CS * c) / l],
    });
  }
  return out;
}
const BLADES = bladeTable(5, 4);

/** crossed-card tufts (tuftGeometry parity): 3 cards at k·1.92+0.4 */
interface CardPar {
  c: number;
  s: number;
  nm: [number, number, number];
}
const CARDS: CardPar[] = Array.from({ length: 3 }, (_, k) => {
  const a = k * 1.92 + 0.4;
  const c = Math.cos(a);
  const s = Math.sin(a);
  // tuftGeometry normal at sgn=0 (mean of ±): (−s·0.97·0.788, 0.25, c·0.97·0.788)
  const nx = -s * 0.97 * 0.788;
  const nz = c * 0.97 * 0.788;
  const l = Math.hypot(nx, 0.25, nz);
  return { c, s, nm: [nx / l, 0.25 / l, nz / l] };
});

export interface GrassBuildOpts {
  cam: NaniteCam;
  vis: NaniteVisBuffers;
  hf: Heightfield;
  canopyTex: StorageTexture | null;
  /** terrain micro-displacement (NaniteFrame's disp) — blades must root on the
   *  DISPLACED surface or short blades sink into the near-field relief. */
  disp?: TerrainDisp;
}

/** per-clump derived state (everything blade corners need) */
interface ClumpCtx {
  wpos: NV2;
  y: NF;
  dist: NF;
  widen: NF;
  bladeH: NF;
  cc: NF;
  cs: NF;
  tiltX: NF;
  tiltY: NF;
  /** wind: hoisted amplitude terms + the per-clump flutter sine (0 = still air) */
  bendAmp: NF;
  flutS: NF; // sin(time·5.2 + phase)·flutAmp — corner scales by tN
  dirX: NF;
  dirY: NF;
  /** representation ladder jitter (per clump, world-anchored) */
  jit: NF;
}

export interface GrassField {
  /** compute kernels for world1's batched submit (after kVisClear, before kHwArgs) */
  batch: readonly unknown[];
  /** the grass HW blade pass (own render target, HARDWARE EARLY-Z): a fullscreen
   *  prime writes the election depth into a real depth buffer, then the blade
   *  draw runs depth-tested — occluded blade fragments never invoke the election
   *  shader. Call right after the raster's hwRender (world1 only). */
  renderHw(renderer: Renderer, camera: PerspectiveCamera): void;
  /** resolve-side shading reconstruction (call INSIDE the resolve fragment Fn) */
  resolveDerive(body: NU, wp: NV3): { t: NF; nrm: NV3 };
  setEnabled(v: boolean): void;
  enabled(): boolean;
  readCounts(renderer: Renderer): Promise<{ clumps: number; hwTris: number }>;
}

export function buildGrassField(opts: GrassBuildOpts): GrassField {
  const { cam, vis, hf } = opts;
  const canopyTex = opts.canopyTex;
  const heightRes = (hf as unknown as { res: number }).res ?? 2048;
  const hasWater = (hf as unknown as { waterY: unknown }).waterY != null;
  const biomeTex = hf.biomeTex as NonNullable<typeof hf.biomeTex>;
  const fieldsTex = hf.fieldsTex as NonNullable<typeof hf.fieldsTex>;
  const uOn = uniformF(1);
  let onCpu = true;

  // manual-bilinear height from the heightTex TEXTURE (not the height storage
  // buffer — the resolve fragment is at the 10-storage-buffer ceiling; texture
  // taps are free there, and both stages MUST share one derivation).
  const heightAt = (p: NV2): NF => {
    const g = p
      .div(WORLD_SIZE)
      .add(0.5)
      .clamp(0, 1)
      .mul(heightRes)
      .sub(0.5) as unknown as NV2;
    const i0 = g.floor() as unknown as NV2;
    const f = g.fract() as unknown as NV2;
    const tap = (dx: number, dy: number): NF => {
      const uvT = i0
        .add(vec2(dx + 0.5, dy + 0.5))
        .div(heightRes)
        .clamp(0, 1) as unknown as NV2;
      return (texture(hf.heightTex, uvT, 0) as unknown as NV4).x as unknown as NF;
    };
    return mix(
      mix(tap(0, 0), tap(1, 0), f.x),
      mix(tap(0, 1), tap(1, 1), f.x),
      f.y,
    ) as unknown as NF;
  };

  /** ground height a blade roots on = heightfield + terrain micro-displacement */
  const groundAt = (p: NV2): NF =>
    (opts.disp
      ? heightAt(p).add(terrainDispAt(opts.disp, p))
      : heightAt(p)) as unknown as NF;

  /** toroidal slot → nearest congruent world cell (GroundRing idiom, shared camPos) */
  const worldCell = (sx: NF, sy: NF, grid: number, cell: number): NV2 => {
    const camC = vec2(cam.camPos.x, cam.camPos.z).div(cell);
    const wx = camC.x.sub(sx).div(grid).round().mul(grid).add(sx);
    const wy = camC.y.sub(sy).div(grid).round().mul(grid).add(sy);
    return vec2(wx, wy);
  };

  /** clump state from its world cell. `ground` (pre-sampled root height) is passed
   *  in where the caller already paid the taps. `liteAmp`: when given, use it as the
   *  wind gust amplitude instead of sampling gustAt/windExposure (the ray lane hoists
   *  those 3 taps to burst level — the gust field varies at 17-85 m, a burst spans
   *  ≤1.7 m); flutter is dropped on the lite path (sub-pixel at those distances). */
  const deriveClump = (wc: NV2, far: boolean, ground: NF, liteAmp?: NF): ClumpCtx => {
    const salt = far ? SALT ^ 0x6f21 : SALT;
    const jit = cellHash2(wc, salt);
    const wpos = wc.add(jit).mul(far ? FAR_CELL : CELL) as unknown as NV2;
    const dist = wpos.sub(vec2(cam.camPos.x, cam.camPos.z)).length() as unknown as NF;
    const h2 = cellHash2(wc, salt ^ 0x9191);
    const tilt = cellHash2(wc, salt ^ 0x4545).sub(0.5).mul(0.5) as unknown as NV2;
    const widen = (far
      ? h2.y.mul(0.8).add(1.6)
      : float(1).div(grassThin(dist).sqrt()).clamp(1, 4)) as unknown as NF;
    const bladeH = h2.x
      .pow(1.3)
      .mul(far ? 0.42 : 0.3)
      .add(far ? 0.34 : 0.2)
      .mul(widen.sub(1).mul(0.3).add(1)) as unknown as NF;
    const yawA = h2.y.mul(6.2831853);
    let bendAmp: NF = float(0) as unknown as NF;
    let flutS: NF = float(0) as unknown as NF;
    let dirX: NF = float(0) as unknown as NF;
    let dirY: NF = float(0) as unknown as NF;
    if (windContext()) {
      const wd = vec2(windU.dir as unknown as NV2);
      const st = windU.strength as unknown as NF;
      const amp = (liteAmp ??
        st.mul(gustAt(wpos).mul(0.9).add(0.3)).mul(windExposure(wpos))) as unknown as NF;
      // lean² rule (ring verbatim): deflection = bendAmp·tN² per corner
      bendAmp = amp.mul(st.mul(0.55).add(0.6)).mul(bladeH.mul(0.42)) as unknown as NF;
      // shimmer: ring formula + the leaf-style ~120 m fade; the SINE is per-clump —
      // corners scale it by tN. Tap-free (hash phase + whichever amp), so the lite
      // path keeps it too — near shimmer is part of the reference look.
      const flutAtten = float(1).sub(dist.sub(40).div(80).clamp(0, 1));
      const flutA = (far ? float(0) : amp.mul(0.05).mul(flutAtten)) as unknown as NF;
      const flutPh = h2.x.mul(6.2832).add(wpos.x.add(wpos.y).mul(0.9)) as unknown as NF;
      flutS = time.mul(5.2).add(flutPh).sin().mul(flutA) as unknown as NF;
      dirX = wd.x as unknown as NF;
      dirY = wd.y as unknown as NF;
    }
    return {
      wpos,
      y: ground,
      dist,
      widen,
      bladeH,
      cc: yawA.cos() as unknown as NF,
      cs: yawA.sin() as unknown as NF,
      tiltX: tilt.x as unknown as NF,
      tiltY: tilt.y as unknown as NF,
      bendAmp,
      flutS,
      dirX,
      dirY,
      jit: cellHash(wc, salt ^ 0xbeef),
    };
  };

  /** representation ladder (continuous, per-clump world-jittered — no dither bands) */
  const ladder = (C: ClumpCtx): { cardMode: NB; prims: NU; segN: NU; segNF: NF } => {
    const j = C.jit.mul(0.3).add(0.85) as unknown as NF; // 0.85..1.15
    const cardMode = C.dist.greaterThan(float(70).mul(j)) as unknown as NB;
    const prims = cardMode.select(
      uint(3),
      C.dist.greaterThan(float(30).mul(j)).select(uint(3), uint(5)),
    ) as unknown as NU;
    const segN = cardMode.select(
      uint(1),
      C.dist
        .lessThan(float(22).mul(j))
        .select(uint(4), C.dist.lessThan(float(48).mul(j)).select(uint(2), uint(1))),
    ) as unknown as NU;
    // FLOAT selects, not toF(segN): a u32→f32 conversion node on the select chain
    // was dropped in the fragment build (WGSL f32/u32 parse error).
    const segNF = cardMode.select(
      float(1),
      C.dist
        .lessThan(float(22).mul(j))
        .select(float(4), C.dist.lessThan(float(48).mul(j)).select(float(2), float(1))),
    ) as unknown as NF;
    return { cardMode, prims, segN, segNF };
  };

  /** select a JS-const table entry by runtime index (5-way chain — used ONCE per
   *  prim loop iteration, hoisted into vars; never in the per-corner path) */
  const sel = (idx: NU, vals: number[]): NF => {
    let e: NF = float(vals[vals.length - 1] ?? 0) as unknown as NF;
    for (let i = vals.length - 2; i >= 0; i--) {
      e = idx.equal(uint(i)).select(float(vals[i] ?? 0), e) as unknown as NF;
    }
    return e;
  };

  /** per-prim params hoisted to vars at the top of the prim loop (one 5-way select
   *  chain per scalar per ITERATION — the corner path below reads plain vars) */
  interface PrimVars {
    bc: NF;
    bs: NF;
    box: NF;
    boz: NF;
    bhk: NF;
    blean: NF;
    kc: NF;
    ks: NF;
  }
  const primVars = (prim: NU): PrimVars => ({
    bc: sel(prim, BLADES.map((b) => b.c)).toVar() as unknown as NF,
    bs: sel(prim, BLADES.map((b) => b.s)).toVar() as unknown as NF,
    box: sel(prim, BLADES.map((b) => b.ox)).toVar() as unknown as NF,
    boz: sel(prim, BLADES.map((b) => b.oz)).toVar() as unknown as NF,
    bhk: sel(prim, BLADES.map((b) => b.hk)).toVar() as unknown as NF,
    blean: sel(prim, BLADES.map((b) => b.lean)).toVar() as unknown as NF,
    kc: sel(prim, CARDS.map((k) => k.c)).toVar() as unknown as NF,
    ks: sel(prim, CARDS.map((k) => k.s)).toVar() as unknown as NF,
  });

  /** one blade/card corner in WORLD space (~40 ALU, no selects, no normals —
   *  the raster needs positions only). side ∈ {−1,+1,0=tip/center}. */
  const corner = (
    C: ClumpCtx,
    P: PrimVars,
    cardMode: NB,
    far: boolean,
    tt: NF,
    side: NF,
  ): NV3 => {
    // blade local (GroundCover/bladeClump verbatim, ×1.25 clump x-scale)
    const bw = float(0.014).mul(float(1).sub(tt.mul(0.85))).mul(1.25).mul(side) as unknown as NF;
    const bby = tt.mul(float(1).sub(tt.mul(tt).mul(0.06))).mul(P.bhk) as unknown as NF;
    const bbz = tt.mul(tt).mul(0.28) as unknown as NF;
    const bx = bw.mul(P.bc).add(bbz.mul(P.bs)).add(P.box).add(P.blean.mul(bby).mul(P.bc)) as unknown as NF;
    const bz = bbz.mul(P.bc).sub(bw.mul(P.bs)).add(P.boz).add(P.blean.mul(bby).mul(P.bs)) as unknown as NF;
    // card local (tuftGeometry verbatim: top width ×0.55, straight)
    const cw = float(far ? 0.21 : 0.04)
      .mul(float(1).sub(tt.mul(0.45)))
      .mul(side) as unknown as NF;
    const lx = cardMode.select(cw.mul(P.kc), bx) as unknown as NF;
    const ly = cardMode.select(tt, bby) as unknown as NF;
    const lz = cardMode.select(cw.mul(P.ks), bz) as unknown as NF;
    // clump transform (grassMaterial verbatim): scale → yaw → tilt shear → wind
    const xScale = C.widen.mul(cardMode.select(float(1.5), float(1.15))) as unknown as NF;
    const yScale = C.bladeH.mul(
      far ? (float(1) as unknown as NF) : cardMode.select(float(2), float(1)),
    ) as unknown as NF;
    const sx = lx.mul(xScale) as unknown as NF;
    const sy = ly.mul(yScale) as unknown as NF;
    const rx = sx.mul(C.cc).add(lz.mul(C.cs)) as unknown as NF;
    const rz = lz.mul(C.cc).sub(sx.mul(C.cs)) as unknown as NF;
    const tN = ly; // pre-scale local y — the ring's positionLocal.y
    const bend = C.bendAmp.mul(tN).mul(tN) as unknown as NF;
    const flut = C.flutS.mul(tN) as unknown as NF;
    const dx = C.dirX.mul(bend).sub(C.dirY.mul(flut)) as unknown as NF;
    const dz = C.dirY.mul(bend).add(C.dirX.mul(flut)) as unknown as NF;
    const dy = bend.mul(tN).mul(-0.4) as unknown as NF;
    return vec3(
      rx.add(C.tiltX.mul(sy)).add(dx).add(C.wpos.x),
      sy.add(C.y).add(dy),
      rz.add(C.tiltY.mul(sy)).add(dz).add(C.wpos.y),
    ) as unknown as NV3;
  };

  // ---- counters + HW queue ------------------------------------------------------------
  const countAttr = new StorageBufferAttribute(new Uint32Array(4), 1);
  const countV = sU32Views(countAttr, 4);
  // the 60 MB blade queue exists ONLY in the geometry reference lane (?grassgeo=1);
  // the ray lane's whole point is zero grass memory.
  const hwWords = GEO_LANE ? 1 + HW_CAP * HW_STRIDE : 4;
  const hwAttr = new StorageBufferAttribute(new Uint32Array(hwWords), 1);
  const hwV = sU32Views(hwAttr, hwWords);
  const hwDrawAttr = new IndirectStorageBufferAttribute(new Uint32Array(4), 4);
  const hwDrawBuf = sU32Views(hwDrawAttr as unknown as StorageBufferAttribute, 4).rw;

  const kClear = Fn(() => {
    If(instanceIndex.equal(uint(0)), () => {
      atomicStore(hwV.atomic.element(0), uint(0));
      atomicStore(countV.atomic.element(0), uint(0));
      atomicStore(countV.atomic.element(1), uint(0));
    });
  })().compute(1, [1]);
  (kClear as unknown as { setName(n: string): void }).setName('grassClear');

  // ---- election emit (world1 clone: guarded depthKey24 atomicMax + winner store) ------
  const depthKey24 = (cz: NF): NU =>
    uint(float(1).sub(cz).mul(16777215).clamp(0, 16777215)) as unknown as NU;
  const emitPx = (px: NU, cz: NF, body: NU): void => {
    const cand = depthKey24(cz).shiftLeft(uint(8)).bitOr(body.bitAnd(uint(0xff))).toVar();
    const prevE = aLoadU(vis.payloadV.atomic.element(px));
    If(cand.greaterThan(prevE), () => {
      const wonE = atomicMax(vis.payloadV.atomic.element(px), cand) as unknown as NU;
      If(cand.greaterThan(wonE), () => {
        atomicStore(vis.visBV.atomic.element(px), uint(GRASS_FLAGS).bitOr(body));
      });
    });
  };

  /** append one tri (WORLD corners, already computed) to the HW queue */
  const hwAppend = (body: NU, w0: NV3, w1: NV3, w2: NV3): void => {
    const slot = atomicAdd(hwV.atomic.element(0), uint(1)) as unknown as NU;
    If(slot.lessThan(uint(HW_CAP)), () => {
      const base = slot.mul(uint(HW_STRIDE)).add(uint(1)).toVar();
      atomicStore(hwV.atomic.element(base), body);
      atomicStore(hwV.atomic.element(base.add(uint(1))), bcF2U(w0.x as unknown as NF));
      atomicStore(hwV.atomic.element(base.add(uint(2))), bcF2U(w0.y as unknown as NF));
      atomicStore(hwV.atomic.element(base.add(uint(3))), bcF2U(w0.z as unknown as NF));
      atomicStore(hwV.atomic.element(base.add(uint(4))), bcF2U(w1.x as unknown as NF));
      atomicStore(hwV.atomic.element(base.add(uint(5))), bcF2U(w1.y as unknown as NF));
      atomicStore(hwV.atomic.element(base.add(uint(6))), bcF2U(w1.z as unknown as NF));
      atomicStore(hwV.atomic.element(base.add(uint(7))), bcF2U(w2.x as unknown as NF));
      atomicStore(hwV.atomic.element(base.add(uint(8))), bcF2U(w2.y as unknown as NF));
      atomicStore(hwV.atomic.element(base.add(uint(9))), bcF2U(w2.z as unknown as NF));
    });
  };

  const edgeFn = (a: NV2, b: NV2, p: NV2): NF =>
    p.y.sub(a.y).mul(b.x.sub(a.x)).sub(p.x.sub(a.x).mul(b.y.sub(a.y))) as unknown as NF;

  /** fixed-point SW scanline of ONE small triangle (mesh core parity: 1/256 snap,
   *  top-left rule, unbiased-weight depth). Oversized → HW append (corners ride). */
  const rasterTri = (w0: NV3, w1: NV3, w2: NV3, body: NU): void => {
    const p0 = cam.vp.mul(vec4(w0, 1)).toVar();
    const p1 = cam.vp.mul(vec4(w1, 1)).toVar();
    const p2 = cam.vp.mul(vec4(w2, 1)).toVar();
    const nearOK = p0.w
      .greaterThan(NEAR_EPS)
      .and(p1.w.greaterThan(NEAR_EPS))
      .and(p2.w.greaterThan(NEAR_EPS));
    If(nearOK.not(), () => {
      // near-plane crossing → HW clips it (unless fully behind)
      If(
        p0.w
          .greaterThan(NEAR_EPS)
          .or(p1.w.greaterThan(NEAR_EPS))
          .or(p2.w.greaterThan(NEAR_EPS)),
        () => {
          hwAppend(body, w0, w1, w2);
        },
      );
    }).Else(() => {
      const nd0 = p0.xyz.div(p0.w).toVar() as unknown as NV3;
      const nd1 = p1.xyz.div(p1.w).toVar() as unknown as NV3;
      const nd2 = p2.xyz.div(p2.w).toVar() as unknown as NV3;
      const areaNdc = edgeFn(
        nd0.xy as unknown as NV2,
        nd1.xy as unknown as NV2,
        nd2.xy as unknown as NV2,
      );
      // two-sided: re-wind back-faces in place
      const flip = areaNdc.lessThan(0);
      const v1 = vec3(flip.select(nd2, nd1)).toVar() as unknown as NV3;
      const v2 = vec3(flip.select(nd1, nd2)).toVar() as unknown as NV3;
      If(areaNdc.notEqual(0), () => {
        const W = float(cam.uW);
        const H = float(cam.uH);
        const s0 = nd0.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
        const s1 = v1.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
        const s2 = v2.xy.add(1).mul(0.5).mul(vec2(W, H)).toVar();
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
          .lessThanEqual(toI(MAX_SW_PX))
          .and(bbMaxY.sub(bbMinY).lessThanEqual(toI(MAX_SW_PX)));
        const startX = maxI(toI(0), bbMinX).toVar();
        const endX = minI(toI(cam.width - 1), bbMaxX).toVar();
        const startY = maxI(toI(0), bbMinY).toVar();
        const endY = minI(toI(cam.height - 1), bbMaxY).toVar();
        const validBB = startX.lessThanEqual(endX).and(startY.lessThanEqual(endY));
        If((smallEnough as unknown as { and(o: unknown): NB }).and(validBB), () => {
          const area2 = yi2
            .sub(yi0)
            .mul(xi1.sub(xi0))
            .sub(xi2.sub(xi0).mul(yi1.sub(yi0)))
            .toVar();
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
            const sx0 = ex0.mul(toI(256)).toVar();
            const sx1 = ex1.mul(toI(256)).toVar();
            const sx2 = ex2.mul(toI(256)).toVar();
            const sy0 = ey0.mul(toI(256)).toVar();
            const sy1 = ey1.mul(toI(256)).toVar();
            const sy2 = ey2.mul(toI(256)).toVar();
            const rcpArea = float(1).div(toF(area2 as unknown as NI)).toVar();
            const zz0 = nd0.z.toVar();
            const zz1 = v1.z.toVar();
            const zz2 = v2.z.toVar();
            loopI('gy', startY as unknown as NI, endY as unknown as NI, (yy) => {
              const cw0 = rw0.toVar();
              const cw1 = rw1.toVar();
              const cw2 = rw2.toVar();
              // SCANLINE X-SPAN (mesh raster parity): solve each edge's row crossing,
              // clip the loop to a guaranteed superset of the covered span. Thin
              // DIAGONAL blade slivers cover ~10-25% of their bbox — this is the
              // difference between visiting the bbox and visiting the blade.
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
              loopI('gx', xLo as unknown as NI, xHi as unknown as NI, (xx) => {
                If(
                  cw0
                    .greaterThanEqual(toI(0))
                    .and(cw1.greaterThanEqual(toI(0)))
                    .and(cw2.greaterThanEqual(toI(0))),
                  () => {
                    const uw0 = cw0.sub(bias0);
                    const uw1 = cw1.sub(bias1);
                    const uw2 = cw2.sub(bias2);
                    const cz = toF(uw0 as unknown as NI)
                      .mul(zz0)
                      .add(toF(uw1 as unknown as NI).mul(zz1))
                      .add(toF(uw2 as unknown as NI).mul(zz2))
                      .mul(rcpArea)
                      .toVar();
                    If(cz.greaterThanEqual(0).and(cz.lessThanEqual(1)), () => {
                      const px = uint(yy).mul(uint(cam.uW)).add(uint(xx));
                      emitPx(px, cz as unknown as NF, body);
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
        }).Else(() => {
          If(validBB, () => {
            hwAppend(body, w0, w1, w2);
          });
        });
      });
    });
  };

  /** raster every prim of one surviving clump. Rows are SHARED between segments:
   *  2 + 2·segN corner evals per blade (not 4·segN). bodyBase = slot<<6 (fine) or
   *  FAR_BASE + farSlot<<3 (far). */
  const rasterClump = (C: ClumpCtx, bodyBase: NU, far: boolean): void => {
    const lad = far
      ? {
          cardMode: uint(1).equal(uint(1)) as unknown as NB,
          prims: uint(3),
          segN: uint(1),
          segNF: float(1) as unknown as NF,
        }
      : ladder(C);
    loopUN(far ? 'gfp' : 'gp', uint(0), lad.prims, (prim) => {
      const P = primVars(prim);
      const rowL = vec3(corner(C, P, lad.cardMode, far, float(0) as unknown as NF, float(-1) as unknown as NF)).toVar() as unknown as NV3;
      const rowR = vec3(corner(C, P, lad.cardMode, far, float(0) as unknown as NF, float(1) as unknown as NF)).toVar() as unknown as NV3;
      loopUN(far ? 'gfs' : 'gs', uint(0), lad.segN, (seg) => {
        const t1 = toF(seg.add(uint(1))).div(lad.segNF) as unknown as NF;
        const isTip = seg.equal(lad.segN.sub(uint(1)));
        // blade tip collapses to a point (ring parity); cards keep their top width
        const wTop = isTip.select(
          lad.cardMode.select(float(1), float(0)),
          float(1),
        ) as unknown as NF;
        const topL = vec3(
          corner(C, P, lad.cardMode, far, t1, (float(-1) as unknown as NF).mul(wTop) as unknown as NF),
        ).toVar() as unknown as NV3;
        const topR = vec3(
          corner(C, P, lad.cardMode, far, t1, (float(1) as unknown as NF).mul(wTop) as unknown as NF),
        ).toVar() as unknown as NV3;
        const bodyPS = bodyBase
          .add(far ? uint(0) : prim.shiftLeft(uint(3)))
          .add(far ? prim.shiftLeft(uint(1)) : seg.shiftLeft(uint(1)))
          .toVar();
        if (GRASS_DBG === 'corners') {
          // attribution stop: corners computed + consumed via a never-taken branch
          // (compiler can't DCE), no scanline/election work.
          If(topR.x.add(rowL.x).add(topL.y).lessThan(-1e30), () => {
            hwAppend(bodyPS as unknown as NU, rowL, rowR, topR);
          });
        } else {
          // tri0 = (rowL,rowR,topR); tri1 = (rowL,topR,topL) — tri1 degenerates at the
          // blade tip (topL==topR) and falls out at the area test.
          rasterTri(rowL, rowR, topR, bodyPS as unknown as NU);
          rasterTri(rowL, topR, topL, bodyPS.add(uint(1)) as unknown as NU);
        }
        rowL.assign(topL);
        rowR.assign(topR);
      });
    });
  };

  // ---- density law (GroundRing cull verbatim) -----------------------------------------
  const byBio = (b: NF, vals: number[]): NF => {
    let e: NF = float(vals[5] ?? 0) as unknown as NF;
    for (let i = 4; i >= 0; i--) {
      e = b.equal(float(i)).select(float(vals[i] ?? 0), e) as unknown as NF;
    }
    return e;
  };
  const inFrustum = (center: NV3, slack: number): NB => {
    let inside: NB | null = null;
    for (let p = 0; p < 6; p++) {
      const pl = cam.planes.element(p) as unknown as NV4;
      const t = pl.xyz.dot(center).add(pl.w).greaterThan(-slack) as unknown as NB;
      inside = inside ? ((inside as unknown as { and(o: NB): NB }).and(t)) : t;
    }
    return inside as NB;
  };

  /** shared field-density (fine + far kernels; far skips scruff) */
  const densityAt = (wpos: NV2, h: NF, dist: NF, scruff: boolean): NF => {
    const uvW = wpos.div(WORLD_SIZE).add(0.5);
    const bio = texture(biomeTex, uvW, 0) as unknown as NV4;
    const fl = texture(fieldsTex, uvW, 0) as unknown as NV4;
    const ns = texture(hf.normalTex, uvW, 0) as unknown as NV4;
    const bioId = bio.x.mul(8).add(0.5).floor() as unknown as NF;
    const above = hasWater
      ? (h.sub(hf.sampleWaterYNearest(wpos)) as unknown as NF)
      : (float(1) as unknown as NF);
    const bank = smoothstep(0.06, 0.5, above).mul(
      float(1).sub(smoothstep(0.2, 1.1, fl.z).mul(0.78)),
    ) as unknown as NF;
    const canopy = canopyTex ? canopyAt(canopyTex, wpos) : (float(0) as unknown as NF);
    let dens = byBio(bioId, [0.18, 0.7, 0.62, 0.7, 1.5, 1.1])
      .mul(bank)
      .mul(bio.z.mul(0.85).add(0.15))
      .mul(float(1).sub(bio.w.mul(0.55)))
      .mul(float(1).sub(canopy.mul(0.45)))
      .mul(fl.x.mul(0.35).add(0.75)) as unknown as NF;
    if (scruff) {
      dens = dens.max(
        float(0.3).mul(float(1).sub(smoothstep(8, 14, dist))).mul(bank),
      ) as unknown as NF;
    }
    dens = dens
      .mul(float(1).sub(bio.y.mul(0.95)))
      .mul(float(1).sub(smoothstep(0.55, 0.95, ns.w))) as unknown as NF;
    // hard water gate (ring: return when above < 0.04)
    return (hasWater
      ? dens.mul(above.greaterThanEqual(0.04).select(float(1), float(0)))
      : dens) as unknown as NF;
  };

  // ---- FINE kernel: 3072² toroidal slots → clumps → blades ---------------------------
  // 2D TILE REMAP: a 256-thread workgroup covers a 16×16 SLOT TILE (1.68 m ground
  // square), not a 27 m row strip — every SIMD group shares distance/frustum/density
  // fate, so early-exits retire whole waves and surviving waves raster spatially
  // coherent clumps (adjacent pixels → cache + low election contention).
  const TILE = 16;
  const TILES_X = GRID / TILE; // 192
  const kFine = Fn(() => {
    returnIf((uOn as unknown as NF).lessThan(0.5) as unknown as NB);
    const i = instanceIndex;
    returnIf(i.greaterThanEqual(uint(GRID * GRID)));
    const tile = i.div(uint(TILE * TILE)).toVar();
    const within = i.mod(uint(TILE * TILE)).toVar();
    const sxU = tile.mod(uint(TILES_X)).mul(uint(TILE)).add(within.mod(uint(TILE)));
    const syU = tile.div(uint(TILES_X)).mul(uint(TILE)).add(within.div(uint(TILE)));
    const sx = toF(sxU);
    const sy = toF(syU);
    const wc = worldCell(sx, sy, GRID, CELL);
    const jit = cellHash2(wc, SALT);
    const wpos = wc.add(jit).mul(CELL) as unknown as NV2;
    const dist = wpos.sub(vec2(cam.camPos.x, cam.camPos.z)).length() as unknown as NF;
    returnIf(dist.greaterThan(R) as unknown as NB);
    // cheap pre-exit before ANY texture tap: accept needs hash < dens·edge·thin
    // with dens ≤ ~1.5 — kills most far slots for a few ALU.
    const thin = grassThin(dist);
    const edge = float(1).sub(smoothstep(R * 0.9, R, dist)) as unknown as NF;
    const hash = cellHash(wc, SALT ^ 0x77a1);
    returnIf(hash.greaterThanEqual(edge.mul(thin).mul(1.5)) as unknown as NB);
    const h = heightAt(wpos);
    returnIf(inFrustum(vec3(wpos.x, h.add(0.5), wpos.y) as unknown as NV3, 1.4).not());
    const dens = densityAt(wpos, h, dist, true);
    returnIf(hash.greaterThanEqual(dens.mul(edge).mul(thin)) as unknown as NB);
    if (GRASS_DBG === 'funnel') {
      // attribution stop: cull chain only — the accept is pinned by a counter write.
      atomicAdd(countV.atomic.element(0), uint(1));
      returnIf(uOn.greaterThanEqual(0) as unknown as NB);
    }
    const ground = (opts.disp ? h.add(terrainDispAt(opts.disp, wpos)) : h) as unknown as NF;
    const C = deriveClump(wc, false, ground);
    // id carries the ROW-MAJOR slot (the resolve decodes slot%GRID / slot÷GRID) —
    // NOT the tile-order thread index.
    const slotRM = syU.mul(uint(GRID)).add(sxU).toVar();
    rasterClump(C, slotRM.shiftLeft(uint(6)) as unknown as NU, false);
  })().compute(GRID * GRID, [256]);
  (kFine as unknown as { setName(n: string): void }).setName('grassFine');

  // ---- FAR kernel: 768² coarse super-tufts (150→265 m) --------------------------------
  const kFar = Fn(() => {
    returnIf((uOn as unknown as NF).lessThan(0.5) as unknown as NB);
    const i = instanceIndex;
    returnIf(i.greaterThanEqual(uint(FAR_GRID * FAR_GRID)));
    const sx = toF(i.mod(uint(FAR_GRID)));
    const sy = toF(i.div(uint(FAR_GRID)));
    const wc = worldCell(sx, sy, FAR_GRID, FAR_CELL);
    const jit = cellHash2(wc, SALT ^ 0x6f21);
    const wpos = wc.add(jit).mul(FAR_CELL) as unknown as NV2;
    const dist = wpos.sub(vec2(cam.camPos.x, cam.camPos.z)).length() as unknown as NF;
    returnIf(dist.lessThan(FAR_R0 - 16).or(dist.greaterThan(FAR_R)) as unknown as NB);
    const fadeIn = smoothstep(FAR_R0 - 16, FAR_R0 + 14, dist) as unknown as NF;
    const edge = float(1).sub(smoothstep(FAR_R * 0.93, FAR_R, dist)) as unknown as NF;
    const hash = cellHash(wc, SALT ^ 0x55aa);
    returnIf(hash.greaterThanEqual(fadeIn.mul(edge).mul(0.55).mul(1.5)) as unknown as NB);
    const h = heightAt(wpos);
    returnIf(inFrustum(vec3(wpos.x, h.add(0.6), wpos.y) as unknown as NV3, 1.6).not());
    const dens = densityAt(wpos, h, dist, false);
    returnIf(hash.greaterThanEqual(dens.mul(fadeIn).mul(edge).mul(0.55)) as unknown as NB);
    // far band starts at 150 m — outside the 85 m displacement fade ⇒ raw height
    const C = deriveClump(wc, true, h);
    rasterClump(C, uint(FAR_BASE).add(i.shiftLeft(uint(3))) as unknown as NU, true);
  })().compute(FAR_GRID * FAR_GRID, [256]);
  (kFar as unknown as { setName(n: string): void }).setName('grassFar');

  // ---- HW indirect args ----------------------------------------------------------------
  const kHwArgs = Fn(() => {
    const nRaw = aLoadU(hwV.atomic.element(0));
    const n = nRaw.greaterThan(uint(HW_CAP)).select(uint(HW_CAP), nRaw);
    hwDrawBuf.element(0).assign(n.mul(uint(3)));
    hwDrawBuf.element(1).assign(uint(1));
    hwDrawBuf.element(2).assign(uint(0));
    hwDrawBuf.element(3).assign(uint(0));
    atomicStore(countV.atomic.element(1), n);
  })().compute(1, [1]);
  (kHwArgs as unknown as { setName(n: string): void }).setName('grassHwArgs');

  // ---- HW material (vertex = pure unpack of kernel-computed corners) ------------------
  const hwMat = new NodeMaterial();
  // body id split across TWO 16-bit varyings — a single f32 varying rounds above
  // 2^24 and fine ids reach ~604M (the mesh HW path's vPayLo/vPayHi idiom).
  const vBodyLo = varyingProperty('float', 'grassBodyLo') as unknown as NF;
  const vBodyHi = varyingProperty('float', 'grassBodyHi') as unknown as NF;
  const vZ = varyingProperty('float', 'grassZ') as unknown as NF;
  const vW = varyingProperty('float', 'grassW') as unknown as NF;
  hwMat.vertexNode = Fn(() => {
    const triIndex = vertexIndex.div(3) as unknown as NU;
    const cornerI = vertexIndex.mod(3) as unknown as NU;
    const base = triIndex.mul(uint(HW_STRIDE)).add(uint(1)).toVar();
    const body = elemU(hwV.ro, base).toVar();
    const off = base.add(uint(1)).add(cornerI.mul(uint(3))).toVar();
    const world = vec3(
      bcU2F(elemU(hwV.ro, off)),
      bcU2F(elemU(hwV.ro, off.add(uint(1)))),
      bcU2F(elemU(hwV.ro, off.add(uint(2)))),
    ) as unknown as NV3;
    const clip = cam.vp.mul(vec4(world, 1)).toVar();
    (vBodyLo as unknown as { assign(v: unknown): void }).assign(
      toF((body as unknown as NU).bitAnd(uint(0xffff))),
    );
    (vBodyHi as unknown as { assign(v: unknown): void }).assign(
      toF((body as unknown as NU).shiftRight(uint(16))),
    );
    (vZ as unknown as { assign(v: unknown): void }).assign(clip.z);
    (vW as unknown as { assign(v: unknown): void }).assign(clip.w);
    return clip;
  })() as unknown as typeof hwMat.vertexNode;
  hwMat.fragmentNode = Fn(() => {
    const z = vZ.div(vW).toVar();
    const body = uint(vBodyLo.round())
      .bitOr(uint(vBodyHi.round()).shiftLeft(uint(16)))
      .toVar();
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const px = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    If(z.greaterThanEqual(0).and(z.lessThanEqual(1)), () => {
      emitPx(px, z as unknown as NF, body as unknown as NU);
    });
    return vec4(0, 0, 0, 0);
  })() as unknown as typeof hwMat.fragmentNode;
  // HARDWARE EARLY-Z (measured 2026-07-03: the blade fragment work is election
  // atomics — depth-testing against the frame's current winners culls occluded
  // fragments BEFORE they run): the pass owns a real depth buffer, a fullscreen
  // PRIME writes the election depth (mesh SW+HW + grass SW slivers are already
  // in it), then blades draw depth-tested + depth-writing (blade-on-blade
  // overdraw self-limits as the queue drains).
  hwMat.depthTest = true;
  hwMat.depthWrite = true;
  hwMat.colorWrite = false;
  hwMat.fog = false;
  hwMat.lights = false;
  hwMat.side = DoubleSide; // grass is two-sided

  const hwGeometry = new BufferGeometry();
  hwGeometry.setAttribute('position', new Float32BufferAttribute(new Float32Array(3), 3));
  hwGeometry.setIndirect(hwDrawAttr, 0);
  hwGeometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);
  const hwMesh = new Mesh(hwGeometry, hwMat);
  hwMesh.name = 'grassHw';
  hwMesh.frustumCulled = false;
  hwMesh.renderOrder = 1;

  // depth-prime: fullscreen triangle whose depthNode decodes the election key
  // (uncovered pixels decode to far plane 1.0 — no clear needed).
  const primeGeometry = new BufferGeometry();
  primeGeometry.setAttribute(
    'position',
    new Float32BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
  );
  primeGeometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);
  const primeMat = new NodeMaterial();
  primeMat.vertexNode = vec4(
    positionGeometry.xy,
    0,
    1,
  ) as unknown as typeof primeMat.vertexNode;
  primeMat.fragmentNode = vec4(0, 0, 0, 0) as unknown as typeof primeMat.fragmentNode;
  primeMat.depthNode = Fn(() => {
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    const elect = elemU(vis.payloadV.ro, pixelIndex);
    // cz = 1 − key/2^24; uncovered (elect==0) decodes to 1.0 = far plane
    return float(1).sub(toF(elect.shiftRight(uint(8))).div(16777215)) as unknown as NF;
  })() as unknown as typeof primeMat.depthNode;
  primeMat.depthTest = false;
  primeMat.depthWrite = true;
  primeMat.colorWrite = false;
  primeMat.fog = false;
  primeMat.lights = false;
  const primeMesh = new Mesh(primeGeometry, primeMat);
  primeMesh.frustumCulled = false;

  // TWO passes over one depth target: the prime READS the election (payloadV.ro)
  // while blade fragments WRITE it (atomic) — WebGPU forbids read-only + writable
  // usage of one buffer inside a single pass, so they cannot share one.
  const primeScene = new Scene();
  primeScene.add(primeMesh);
  const bladeScene = new Scene();
  bladeScene.add(hwMesh);
  const hwRT = new RenderTarget(cam.width, cam.height, { depthBuffer: true });
  tagGpu(hwRT, 'grass.hw');
  const renderHw = (renderer: Renderer, camera: PerspectiveCamera): void => {
    if (!onCpu) return;
    const prevRT = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    renderer.setRenderTarget(hwRT);
    // no clear anywhere: the prime rewrites every depth texel (uncovered → far
    // plane); the blade pass LOADS that depth; color is never written or read.
    renderer.autoClear = false;
    renderer.render(primeScene, camera);
    renderer.render(bladeScene, camera);
    renderer.autoClear = prevAutoClear;
    renderer.setRenderTarget(prevRT);
  };

  // ---- resolve-side shading reconstruction (ANALYTIC — no corner re-derivation) -------
  // t = (wp.y − rootY)/bladeHeight: exact up to the wind-bend dip (≤ ~4% of height)
  // and the 0.06 curve term — shading-only jitter, sub-visible at blade widths.
  // Normal = per-prim MEAN rounded normal, clump-yaw rotated; the shading side pulls
  // it toward the terrain normal from 8 m out (S0), so per-side curvature loss is
  // invisible at rendered blade widths. FAR super-tufts: constant mid-tip + up
  // (their normal is fully terrain-pulled at upK=1).
  const resolveDerive = (body: NU, wp: NV3): { t: NF; nrm: NV3 } => {
    const isFar = body.greaterThanEqual(uint(FAR_BASE));
    const tOut = float(0.55).toVar() as unknown as NF;
    const nOut = vec3(0, 1, 0).toVar() as unknown as NV3;
    // distance cheap path (the ?resfar idiom): beyond ~40 m a blade is 1-3 px wide —
    // the tip ramp is sub-pixel and the normal is ≥66% terrain-pulled; skip the
    // root-height taps entirely (t=0.55 + up-normal → terrain lighting).
    const camD = wp.sub(vec3(cam.camPos) as unknown as NV3).length() as unknown as NF;
    If((isFar as unknown as NB).not().and(camD.lessThan(40)), () => {
      const slot = body.shiftRight(uint(6)).toVar();
      const prim = body.shiftRight(uint(3)).bitAnd(uint(7)).toVar();
      const sx = toF(slot.mod(uint(GRID)));
      const sy = toF(slot.div(uint(GRID)));
      const wc = worldCell(sx, sy, GRID, CELL);
      const jit = cellHash2(wc, SALT);
      const wpos = wc.add(jit).mul(CELL) as unknown as NV2;
      const dist = wpos.sub(vec2(cam.camPos.x, cam.camPos.z)).length() as unknown as NF;
      const h2 = cellHash2(wc, SALT ^ 0x9191);
      const widen = float(1).div(grassThin(dist).sqrt()).clamp(1, 4) as unknown as NF;
      const bladeH = h2.x
        .pow(1.3)
        .mul(0.3)
        .add(0.2)
        .mul(widen.sub(1).mul(0.3).add(1)) as unknown as NF;
      const j = cellHash(wc, SALT ^ 0xbeef).mul(0.3).add(0.85) as unknown as NF;
      const cardMode = dist.greaterThan(float(70).mul(j)) as unknown as NB;
      const rootY = groundAt(wpos);
      // world height of this prim: blades = 0.94·hk·bladeH; mid cards = 2·bladeH
      const hk = sel(prim as unknown as NU, BLADES.map((b) => b.hk));
      const hWorld = cardMode.select(
        bladeH.mul(2),
        bladeH.mul(hk).mul(0.94),
      ) as unknown as NF;
      (tOut as unknown as { assign(v: unknown): void }).assign(
        wp.y.sub(rootY).div(hWorld.max(0.05)).clamp(0, 1),
      );
      // per-prim mean normal (blade- or card-table), clump-yaw rotated
      const yawA = h2.y.mul(6.2831853);
      const cc = yawA.cos();
      const cs = yawA.sin();
      const nx = cardMode.select(
        sel(prim as unknown as NU, CARDS.map((k) => k.nm[0])),
        sel(prim as unknown as NU, BLADES.map((b) => b.nm[0])),
      ) as unknown as NF;
      const ny = cardMode.select(
        sel(prim as unknown as NU, CARDS.map((k) => k.nm[1])),
        sel(prim as unknown as NU, BLADES.map((b) => b.nm[1])),
      ) as unknown as NF;
      const nz = cardMode.select(
        sel(prim as unknown as NU, CARDS.map((k) => k.nm[2])),
        sel(prim as unknown as NU, BLADES.map((b) => b.nm[2])),
      ) as unknown as NF;
      (nOut as unknown as { assign(v: unknown): void }).assign(
        vec3(nx.mul(cc).add(nz.mul(cs)), ny, nz.mul(cc).sub(nx.mul(cs))),
      );
    });
    return { t: tOut, nrm: normalize(nOut) as unknown as NV3 };
  };

  const readCounts = async (
    renderer: Renderer,
  ): Promise<{ clumps: number; hwTris: number }> => {
    // ray lane emits per-pixel — no counters exist (and countAttr never gets a GPU
    // buffer, so a readback would throw)
    if (!GEO_LANE) return { clumps: 0, hwTris: 0 };
    const buf = await readBuffer(renderer, countAttr, 0, 8);
    const u = new Uint32Array(buf);
    return { clumps: u[0] ?? 0, hwTris: u[1] ?? 0 };
  };

  // ================= RAY LANE (default) =================================================
  // Per-pixel analytic raycast of the SAME procedural clump field — zero memory,
  // zero emission, zero overdraw (≤1 election write per pixel). Bounded everywhere:
  //  - above the sward: Lipschitz height-jumps (slope bound), geometric for rising rays;
  //  - inside dense sward: hits arrive within 1-3 cells (density = speed, not cost);
  //  - inside sparse sward: probabilistic coarse-cell skip + hard step caps, and a
  //    capped miss falls through to the terrain splat — the correct far-field limit.
  // A hit emits the SAME self-describing blade id + depth into the election, so the
  // resolve/shadows/GTAO/TRAA pipeline is untouched.
  const kRay = ((): unknown => {
    if (GEO_LANE) return null;
    const W = cam.width;
    const H = cam.height;
    // (8×8 pixel tiling was measured NEUTRAL-to-worse here — rows are already
    // coherent; linear indexing kept)
    const k = Fn(() => {
      returnIf((uOn as unknown as NF).lessThan(0.5) as unknown as NB);
      const px = instanceIndex;
      returnIf(px.greaterThanEqual(uint(W * H)));
      const xI = px.mod(uint(W));
      const yI = px.div(uint(W)); // bottom-up rows (raster convention)
      const ndcX = toF(xI).add(0.5).div(W).mul(2).sub(1);
      const ndcY = toF(yI).add(0.5).div(H).mul(2).sub(1);
      const hf4 = cam.invVp.mul(vec4(ndcX, ndcY, 1, 1));
      const ro = vec3(cam.camPos).toVar() as unknown as NV3;
      const rd = (hf4.xyz.div(hf4.w).sub(ro).normalize().toVar()) as unknown as NV3;
      // scene early-out: current election depth bounds the march
      const elect = aLoadU(vis.payloadV.atomic.element(px));
      const tMax = float(1e9).toVar() as unknown as NF;
      If(elect.notEqual(uint(0)), () => {
        const czS = float(1).sub(toF(elect.shiftRight(uint(8))).div(16777215));
        const hs = cam.invVp.mul(vec4(ndcX, ndcY, czS, 1));
        (tMax as unknown as { assign(v: unknown): void }).assign(
          hs.xyz.div(hs.w).sub(ro).length().add(0.3),
        );
      });
      const dirL = rd.xz.length().max(1e-4).toVar() as unknown as NF;
      const tEnd = tMax.min(float(RAY_END).div(dirL)).toVar() as unknown as NF;
      const tCur = float(0.05).toVar() as unknown as NF;
      const tBest = float(1e9).toVar() as unknown as NF;
      const bodyBest = uint(0).toVar() as unknown as NU;
      if (GRASS_DBG === 'raysetup') {
        // attribution stop: ray gen + scene-depth reconstruct only
        If(tEnd.lessThan(-1), () => {
          emitPx(px as unknown as NU, tCur as unknown as NF, bodyBest);
        });
        returnIf(tBest.greaterThan(0) as unknown as NB);
      }

      /** test the prims of the clump in FINE cell (fx,fz) against the ray.
       *  unroll=true (near band): JS-unrolled prims with CONSTANT blade params (no
       *  select chains) — the near per-clump battery is the kernel's dominant cost.
       *  Every prim gets a ~15-ALU ray-to-root 2D capsule pre-reject (a ray hits ≤2
       *  of 5 blades). Ground: near = per-clump heightAt (4 taps) + burst disp;
       *  lite (≥25 m) = burst ground, zero taps. */
      const testCell = (
        fx: NF,
        fz: NF,
        thr: NF,
        lite: NB,
        groundB: NF,
        dispB: NF,
        ampB: NF,
        widenTB: NF,
        midKB: NF,
      ): void => {
        const wcF = vec2(fx, fz) as unknown as NV2;
        if (GRASS_DBG === 'raymarch') {
          // attribution stop: full march/DDA, hash accept only — no clump testing.
          // Pin the hash + a fake "hit" so nothing is DCE'd (never true in practice).
          If(cellHash(wcF, SALT ^ 0x77a1).lessThan(thr.mul(1e-9)), () => {
            (tBest as unknown as { assign(v: unknown): void }).assign(float(1e8));
          });
          return;
        }
        // PRE-DERIVE clump-top reject (~30 ALU vs the ~400-ALU derive+ladder): in the
        // skim band — ray above the MEDIAN blade top but under the burst max — this
        // is where the kernel's time went (bisect 2026-07-03: march 2.4 ms, full 47).
        const h2x = cellHash2(wcF, SALT ^ 0x9191).x as unknown as NF;
        const clumpTop = h2x
          .pow(1.3)
          .mul(0.3)
          .add(0.2)
          .mul(widenTB)
          .mul(midKB)
          .add(0.3) as unknown as NF;
        const wposQ = wcF.add(0.5).mul(CELL) as unknown as NV2;
        const tQ = wposQ.sub(ro.xz as unknown as NV2).length().div(dirL) as unknown as NF;
        const rayYq = ro.y.add(rd.y.mul(tQ)).sub(groundB) as unknown as NF;
        If(
          rayYq
            .lessThan(clumpTop)
            .and(cellHash(wcF, SALT ^ 0x77a1).lessThan(thr)),
          () => {
          const wposC = wcF.add(cellHash2(wcF, SALT)).mul(CELL) as unknown as NV2;
          const ground = groundB.toVar() as unknown as NF;
          If((lite as unknown as { not(): NB }).not(), () => {
            // near: per-clump root height (4 taps) — a whole warp is near or not,
            // so lite warps skip the taps entirely
            (ground as unknown as { assign(v: unknown): void }).assign(
              heightAt(wposC).add(dispB),
            );
          });
          const C = deriveClump(wcF, false, ground, ampB);
          const lad = ladder(C);
          const tClump = wposC.sub(ro.xz as unknown as NV2).length().div(dirL) as unknown as NF;
          const rayYc = ro.y.add(rd.y.mul(tClump)).sub(ground).toVar() as unknown as NF;
          const sxs = fx.sub(fx.div(GRID).floor().mul(GRID));
          const sys = fz.sub(fz.div(GRID).floor().mul(GRID));
          const slot = uint(sys.mul(GRID).add(sxs)).toVar() as unknown as NU;

          // ANALYTIC blade intersection (closed form — replaces 10 corner evals +
          // 8 Möller–Trumbore per blade; ~120 ALU, ~15 live registers — the kernel
          // was occupancy-bound on the old graph, not ALU-bound):
          //   P(by,u) = A + B·by + C·by² + E·u·W(by)
          // by = pre-scale blade height param (tN), u ∈ [−1,1] width coordinate.
          // A: root; B: yScale/lean/tilt/flutter (linear); C: curve+wind bend
          // (quadratic; the −0.4·bend·tN dip is dropped — ≤4% of height); E·W(by):
          // tapered width. Ray substitution ⇒ ONE quadratic in s.
          const testPrim = (prim: NU, PV: PrimVars, hkC: NF): void => {
            const card = lad.cardMode;
            const byMax = card.select(float(1), hkC.mul(0.94)) as unknown as NF;
            const primTop = C.bladeH
              .mul(card.select(float(2), float(1)))
              .mul(byMax)
              .add(0.25) as unknown as NF;
            If(rayYc.lessThan(primTop), () => {
              const xScale = C.widen.mul(card.select(float(1.5), float(1.15))) as unknown as NF;
              const yScale = C.bladeH.mul(card.select(float(2), float(1))) as unknown as NF;
              // width dir (bc,0,−bs) and bend dir (bs,0,bc), x pre-scaled, clump-yawed
              const bcx = card.select(PV.kc, PV.bc) as unknown as NF;
              const bsx = card.select(PV.ks, PV.bs) as unknown as NF;
              const Ex = bcx.mul(xScale).mul(C.cc).add(bsx.negate().mul(C.cs)) as unknown as NF;
              const Ez = bsx.negate().mul(C.cc).sub(bcx.mul(xScale).mul(C.cs)) as unknown as NF;
              const Gx = bsx.mul(xScale).mul(C.cc).add(bcx.mul(C.cs)) as unknown as NF;
              const Gz = bcx.mul(C.cc).sub(bsx.mul(xScale).mul(C.cs)) as unknown as NF;
              // root (cards sit at the clump center)
              const ox = card.select(float(0), PV.box) as unknown as NF;
              const oz = card.select(float(0), PV.boz) as unknown as NF;
              const Ax = ox.mul(xScale).mul(C.cc).add(oz.mul(C.cs)).add(C.wpos.x) as unknown as NF;
              const Az = oz.mul(C.cc).sub(ox.mul(xScale).mul(C.cs)).add(C.wpos.y) as unknown as NF;
              // linear coeff: lean (blades) + tilt shear + flutter; y = yScale
              const lean = card.select(float(0), PV.blean) as unknown as NF;
              const Bx = lean
                .mul(bcx)
                .mul(xScale)
                .mul(C.cc)
                .add(lean.mul(bsx).mul(C.cs))
                .add(C.tiltX.mul(yScale))
                .sub(C.dirY.mul(C.flutS)) as unknown as NF;
              const Bz = lean
                .mul(bsx)
                .mul(C.cc)
                .sub(lean.mul(bcx).mul(xScale).mul(C.cs))
                .add(C.tiltY.mul(yScale))
                .add(C.dirX.mul(C.flutS)) as unknown as NF;
              // quadratic coeff: built-in arc 0.28·t² (t≈by/hk) + wind bend·by²
              const k2 = card.select(
                float(0),
                float(0.28).div(hkC.mul(hkC).max(0.05)),
              ) as unknown as NF;
              const Cx = Gx.mul(k2).add(C.dirX.mul(C.bendAmp)) as unknown as NF;
              const Cz = Gz.mul(k2).add(C.dirY.mul(C.bendAmp)) as unknown as NF;
              // by(s) = alpha + beta·s  (vertical solve; yScale > 0 always)
              const alpha = ro.y.sub(ground).div(yScale) as unknown as NF;
              const beta = rd.y.div(yScale) as unknown as NF;
              // project x/z onto F ⊥ E
              const Fx = Ez.negate() as unknown as NF;
              const Fz = Ex as unknown as NF;
              const f0 = Fx.mul(ro.x.sub(Ax)).add(Fz.mul(ro.z.sub(Az))) as unknown as NF;
              const f1 = Fx.mul(rd.x).add(Fz.mul(rd.z)) as unknown as NF;
              const b1 = Fx.mul(Bx).add(Fz.mul(Bz)) as unknown as NF;
              const c1 = Fx.mul(Cx).add(Fz.mul(Cz)) as unknown as NF;
              // c1(α+βs)² + b1(α+βs) − f0 − f1 s = 0
              const qa = c1.mul(beta).mul(beta) as unknown as NF;
              const qb = c1.mul(2).mul(alpha).mul(beta).add(b1.mul(beta)).sub(f1) as unknown as NF;
              const qc = c1.mul(alpha).mul(alpha).add(b1.mul(alpha)).sub(f0) as unknown as NF;
              const disc = qb.mul(qb).sub(qa.mul(qc).mul(4)) as unknown as NF;
              If(disc.greaterThanEqual(0), () => {
                const sq = disc.sqrt();
                const isLin = qa.abs().lessThan(1e-7);
                const inv2a = float(0.5).div(
                  qa.abs().max(1e-7).mul(qa.greaterThanEqual(0).select(float(1), float(-1))),
                ) as unknown as NF;
                const sLin = qc.negate().div(
                  qb.abs().max(1e-7).mul(qb.greaterThanEqual(0).select(float(1), float(-1))),
                ) as unknown as NF;
                const r1 = isLin.select(sLin, qb.negate().sub(sq).mul(inv2a)) as unknown as NF;
                const r2 = isLin.select(float(1e9), qb.negate().add(sq).mul(inv2a)) as unknown as NF;
                const sNear = r1.min(r2).toVar() as unknown as NF;
                const sFar = r1.max(r2).toVar() as unknown as NF;
                // try near root, else far root (root order vs travel direction)
                const tryRoot = (s: NF): void => {
                  const by = alpha.add(beta.mul(s)) as unknown as NF;
                  const okBy = by.greaterThanEqual(0).and(by.lessThanEqual(byMax));
                  If(
                    s.greaterThan(0.02)
                      .and(s.lessThan(tBest))
                      .and(s.lessThan(tMax))
                      .and(okBy),
                    () => {
                      // width coordinate: residual along E over tapered width
                      const hx = ro.x
                        .add(rd.x.mul(s))
                        .sub(Ax)
                        .sub(Bx.mul(by))
                        .sub(Cx.mul(by).mul(by)) as unknown as NF;
                      const hz = ro.z
                        .add(rd.z.mul(s))
                        .sub(Az)
                        .sub(Bz.mul(by))
                        .sub(Cz.mul(by).mul(by)) as unknown as NF;
                      const e2 = Ex.mul(Ex).add(Ez.mul(Ez)).max(1e-8) as unknown as NF;
                      const uw = hx.mul(Ex).add(hz.mul(Ez)).div(e2) as unknown as NF;
                      const tb = by.div(byMax) as unknown as NF;
                      const wAt = card.select(
                        float(0.04).mul(float(1).sub(tb.mul(0.45))),
                        float(0.0175).mul(float(1).sub(tb.mul(0.85))),
                      ) as unknown as NF;
                      If(uw.abs().lessThanEqual(wAt), () => {
                        (tBest as unknown as { assign(v: unknown): void }).assign(s);
                        const seg = uint(
                          tb.mul(lad.segNF).min(lad.segNF.sub(0.01)),
                        ) as unknown as NU;
                        (bodyBest as unknown as { assign(v: unknown): void }).assign(
                          slot
                            .shiftLeft(uint(6))
                            .bitOr(prim.shiftLeft(uint(3)))
                            .bitOr(seg.shiftLeft(uint(1)))
                            .bitOr(uw.greaterThan(0).select(uint(1), uint(0))),
                        );
                      });
                    },
                  );
                };
                tryRoot(sNear as unknown as NF);
                tryRoot(sFar as unknown as NF);
              });
            });
          };

          // (JS-unrolled prims measured SLOWER — register bloat beat the saved
          // selects; one compact runtime loop keeps occupancy up)
          loopUN('gp2', uint(0), lad.prims, (prim) => {
            const PV = primVars(prim);
            testPrim(prim, PV, PV.bhk);
          });
        });
      };

      loopUN('gro', uint(0), uint(56), () => {
        If(tCur.greaterThanEqual(tEnd).or(tCur.greaterThan(tBest.add(0.3))), () => {
          Break();
        });
        const pos = ro.add(rd.mul(tCur)).toVar() as unknown as NV3;
        const hC = heightAt(pos.xz as unknown as NV2).toVar() as unknown as NF;
        // TIGHT analytic sward top at this distance (the blade-height law is closed
        // form — no bake needed): blades = bladeHmax·hkMax·widenTerm; mid cards ×2;
        // + tilt/wind margin. Replaces the loose 1.5 m shell — skimming rays exit the
        // candidate band 2-3× sooner, which is where the march cost lives.
        const distH0 = tCur.mul(dirL) as unknown as NF;
        const widenT = float(1)
          .div(grassThin(distH0).sqrt())
          .clamp(1, 4)
          .sub(1)
          .mul(0.3)
          .add(1) as unknown as NF;
        const topB = distH0
          .greaterThan(60)
          .select(float(1.0).mul(widenT), float(0.62).mul(widenT))
          .min(RAY_SHELL_H)
          .add(0.25)
          .toVar() as unknown as NF;
        const head = pos.y.sub(hC).sub(topB).toVar() as unknown as NF;
        If(head.greaterThan(0.2), () => {
          // Lipschitz skip over open air (terrain slope bound 1.4) — geometric for
          // ascending rays, linear for grazing; always terrain-safe.
          tCur.addAssign(head.div(1.4).max(0.3));
        }).Else(() => {
          // in-shell DDA burst at the distance-appropriate stride (a 2×CELL near
          // stride with 2×2 sub-cells was measured WORSE — reverted to per-cell)
          const distH = distH0;
          const nearL0 = distH.lessThan(RAY_L0_END);
          const P = nearL0.select(float(CELL), float(CELL * 4)).toVar() as unknown as NF;
          const burstN = nearL0.select(uint(8), uint(4)) as unknown as NU;
          // burst context, hoisted (fields vary ≥1.5 m; a burst spans ≤1.7 m):
          // density (3 taps + water), gust amplitude (3 taps), ground (already hC).
          const thin = grassThin(distH);
          const edge = float(1).sub(smoothstep(R * 0.9, R, distH)) as unknown as NF;
          const pFine = densityAt(pos.xz as unknown as NV2, hC, distH, true)
            .mul(thin)
            .mul(edge)
            .toVar() as unknown as NF;
          const ampB = (windContext()
            ? (windU.strength as unknown as NF)
                .mul(gustAt(pos.xz as unknown as NV2).mul(0.9).add(0.3))
                .mul(windExposure(pos.xz as unknown as NV2))
            : float(0)
          ).toVar() as unknown as NF;
          // burst displacement (terrainDispAt self-gates beyond its 85 m fade) —
          // near clumps pay only their own 4-tap heightAt on top of this
          const dispB = (opts.disp
            ? terrainDispAt(opts.disp, pos.xz as unknown as NV2)
            : (float(0) as unknown as NF)
          ).toVar() as unknown as NF;
          const groundB = hC.add(dispB).toVar() as unknown as NF;
          const midKB = distH.greaterThan(60).select(float(2), float(1.12)).toVar() as unknown as NF;
          // DDA state (t values ABSOLUTE along the ray)
          const cellX = pos.x.div(P).floor().toVar() as unknown as NF;
          const cellZ = pos.z.div(P).floor().toVar() as unknown as NF;
          const sdx = rd.x
            .greaterThanEqual(0)
            .select(rd.x.max(1e-6), rd.x.min(-1e-6))
            .toVar() as unknown as NF;
          const sdz = rd.z
            .greaterThanEqual(0)
            .select(rd.z.max(1e-6), rd.z.min(-1e-6))
            .toVar() as unknown as NF;
          const stX = rd.x.greaterThanEqual(0).select(float(1), float(-1)).toVar() as unknown as NF;
          const stZ = rd.z.greaterThanEqual(0).select(float(1), float(-1)).toVar() as unknown as NF;
          const tNx = cellX
            .add(stX.mul(0.5).add(0.5))
            .mul(P)
            .sub(ro.x)
            .div(sdx)
            .toVar() as unknown as NF;
          const tNz = cellZ
            .add(stZ.mul(0.5).add(0.5))
            .mul(P)
            .sub(ro.z)
            .div(sdz)
            .toVar() as unknown as NF;
          const tDx = P.div(sdx.abs()).toVar() as unknown as NF;
          const tDz = P.div(sdz.abs()).toVar() as unknown as NF;
          loopUN('gri', uint(0), burstN as unknown as NU, () => {
            // per-cell height rejection BEFORE any hashing (~5 ALU): the min ray
            // height across this cell vs the burst's tight sward top. This is what
            // collapses the sward-top-skim pathology — cells the ray clears are
            // pure DDA advances.
            const tCellExit = tNx.min(tNz).min(tEnd);
            const rayYmin = ro.y
              .add(rd.y.mul(rd.y.lessThan(0).select(tCellExit, tCur)))
              .sub(groundB) as unknown as NF;
            // ONE testCell inline site for both strides (kernel size = occupancy);
            // probabilistic whole-cell gate keeps sparse marches cheap at both levels
            // (joint accept per fine cell = gate·thr = pFine — statistics preserved).
            const subs = nearL0.select(float(1), float(4)) as unknown as NF;
            const pAgg = pFine.mul(subs).mul(subs).min(1) as unknown as NF;
            const gate = cellHash(vec2(cellX, cellZ) as unknown as NV2, SALT ^ 0x9e37);
            If(rayYmin.lessThanEqual(topB).and(gate.lessThan(pAgg)), () => {
              const thr = pFine.div(pAgg.max(1e-4)).min(1) as unknown as NF;
              const subN = nearL0.select(uint(1), uint(4)) as unknown as NU;
              const lite = nearL0.not() as unknown as NB;
              loopUN('gu', uint(0), subN, (u) => {
                loopUN('gv', uint(0), subN, (v) => {
                  const fx = cellX.mul(subs).add(toF(u)) as unknown as NF;
                  const fz = cellZ.mul(subs).add(toF(v)) as unknown as NF;
                  testCell(
                    fx,
                    fz,
                    thr as unknown as NF,
                    lite,
                    groundB,
                    dispB,
                    ampB,
                    widenT,
                    midKB,
                  );
                });
              });
            });
            // advance to the next cell; stop past the burst window
            If(tNx.lessThan(tNz), () => {
              (cellX as unknown as { addAssign(v: unknown): void }).addAssign(stX);
              tCur.assign(tNx);
              tNx.addAssign(tDx);
            }).Else(() => {
              (cellZ as unknown as { addAssign(v: unknown): void }).addAssign(stZ);
              tCur.assign(tNz);
              tNz.addAssign(tDz);
            });
            If(tCur.greaterThanEqual(tEnd).or(tCur.greaterThan(tBest.add(0.3))), () => {
              Break();
            });
          });
        });
      });

      If(tBest.lessThan(1e8), () => {
        const hit = ro.add(rd.mul(tBest));
        const clip = cam.vp.mul(vec4(hit, 1));
        const cz = clip.z.div(clip.w.max(NEAR_EPS));
        If(cz.greaterThanEqual(0).and(cz.lessThanEqual(1)), () => {
          emitPx(px as unknown as NU, cz as unknown as NF, bodyBest);
        });
      });
    })().compute(W * H, [256]);
    (k as unknown as { setName(n: string): void }).setName('grassRay');
    return k;
  })();

  const runGrass = (renderer: Renderer, camera: PerspectiveCamera): void => {
    if (!onCpu) return;
    if (kRay) {
      dispatch(renderer, kRay);
      return;
    }
    renderHw(renderer, camera);
  };

  return {
    batch: GEO_LANE ? [kClear, kFine, kFar, kHwArgs] : [],
    renderHw: runGrass,
    resolveDerive,
    setEnabled(v: boolean): void {
      onCpu = v;
      uOn.value = v ? 1 : 0;
    },
    enabled: () => onCpu,
    readCounts,
  };
}
