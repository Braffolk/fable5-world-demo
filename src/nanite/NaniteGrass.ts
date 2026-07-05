/**
 * NaniteGrass — the Sannikov precomputed-raycast grass lane (G-E, default-on;
 * docs/deep-review/grass-raycast.txt + docs/perf-runs/2026-07-03-grass-arc.md).
 *
 * A per-frame GUIDE FIELD (camera-centered ctx+mask storage buffers) carries
 * the world into the march: ground + gradient, sward top, gust amplitude and
 * the per-cell density-law occupancy bits. kRay walks guide texels per pixel;
 * inside occupied sward each step is answered by ONE trilinear fetch of the
 * boot-baked (x, z-in-tile, angle) raycast tile (GrassRayBake.ts — R = path
 * length 1/(1+d), G/B = normal azimuth/y, A = root-cell id). Wind and blade
 * arcs ride a QUADRATIC oblique march-space basis (off = Sl·h + Sq·h²,
 * re-linearized per fetch); anti-tiling = a continuous swirl rotation θ(pos)
 * + a golden-angle overlay layer. Hits emit a self-describing id + depth into
 * the vis-buffer election; the hit normal + tip param ride a screen
 * StorageTexture to the resolve (grassProc.ray).
 *
 * ⚠️ every march-space FIELD must be constant-per-STEP but continuous across
 * space — per-TILE-constant anything (values OR sampling granularity) reads
 * as a 0.84 m grid from altitude (five-round user-debugged lesson, ledger
 * §G-E GRID POST-MORTEM).
 *
 * The pre-G-E lanes (geo emission + HW queue + SW scanline, hybrid seam,
 * analytic all-bands march, statistical far band) were DELETED 2026-07-04
 * (user call) — git history has them.
 */

import { Data3DTexture, RepeatWrapping } from 'three';
import { HalfFloatType, LinearFilter, NearestFilter, RGBAFormat } from 'three';
import type { PerspectiveCamera } from 'three';
import { StorageBufferAttribute, StorageTexture, type Renderer } from 'three/webgpu';
import {
  Break,
  Fn,
  If,
  atan,
  atomicMax,
  atomicStore,
  float,
  instanceIndex,
  mix,
  normalize,
  smoothstep,
  texture,
  texture3D,
  textureStore,
  time,
  uint,
  uvec2,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { NB, NF, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import { canopyAt, cellHash, cellHash2 } from '../gpu/passes/Scatter';
import { gustAt, windContext, windExposure, windU } from '../render/Wind';
import { terrainDispAt, type TerrainDisp } from './NaniteFetch';
import type { Heightfield } from '../world/Heightfield';
import { WORLD_SIZE } from '../world/WorldConst';
import type { NaniteCam } from './NaniteCommon';
import type { NaniteVisBuffers } from './NaniteRaster';
import { bakeGrassRayTile } from './GrassRayBake';
import {
  aLoadU,
  bcF2U,
  bcU2F,
  dispatch,
  loopUN,
  packHalfU,
  returnIf,
  sU32Views,
  sUvec2,
  sUvec4RO,
  toF,
  uniformF,
  unpackHalfU,
} from './Tsl';

// ---- constants (GroundRing parity — the tuned reference) ---------------------------
const GRID = 3072;
const CELL = 0.105; // m → ±161 m ring, ~90 slots/m²
const R = 155;
const SALT = 0x51a55e & 0x7fffffff;
/** election id namespace: bit31|bit30 (voxel = bit31 only, mesh < bit30) */
const GRASS_FLAGS = 0xc0000000;
/** far-tuft ids live above this body offset (fine max = GRID²·64 ≈ 604M).
 *  Exported for the resolve's far-pixel cheap path. */
export const GRASS_FAR_BASE = 0x28000000; // 671M (no producer since the geo-lane delete)
/** ?grassdbg=raysetup — BUILD-TIME kernel stop (ray gen + scene-depth reconstruct
 *  only); ?grassdbg=flatres lives in NaniteResolve. Production pristine unset. */
const GRASS_DBG = new URLSearchParams(window.location.search).get('grassdbg');
// THE LANE (single, DEFAULT-ON — user calls 2026-07-04): the Sannikov article
// algorithm (docs/deep-review/grass-raycast.txt) in ALL bands — a boot-baked
// (x, z, angle) raycast tile texture answers each march step with ONE fetch
// (O(1), no per-clump batteries); wind = his march-space TBN shear; anti-tiling
// = a continuous swirl rotation + the golden-angle overlay. The old lanes
// (geo emission+HW queue, hybrid near-raster seam, analytic all-bands march,
// SW scanline) were DELETED 2026-07-04 (user call) — resurrect from git if a
// reference is ever needed. ?grass=0 disables grass entirely (NaniteFrame gate).
// ⚠️ the ≤2 ms law is NOT met yet (eye +14.5 gpuWall after the look passes) —
// the ray-march perf pass is the open follow-up.
/** article-lane knobs (all bake/runtime params of the G-E algorithm):
 *  ?grassbakres  — tile texels per edge (article 64; "1 texel ≈ 1 screen px")
 *  ?grassbakang  — angle slices (article uses 8, up to 64)
 *  ?grassshiftk  — bake fiber-shift heuristic, cells/cell (inclined-blade approx)
 *  ?grassthickk  — bake thicken-with-distance heuristic, fraction/cell (taper approx)
 *  ?grassbomb=0  — disable texture bombing (see the raw tiling)
 *  ?grassshear=0 — disable the march-space wind shear */
const qNum = (k: string, d: number, lo: number, hi: number): number => {
  const v = Number(new URLSearchParams(window.location.search).get(k) ?? String(d));
  return Number.isFinite(v) && v >= lo && v <= hi ? v : d;
};
const BAKE_RES = Math.round(qNum('grassbakres', 64, 16, 256));
const BAKE_ANG = Math.round(qNum('grassbakang', 8, 4, 64));
const BAKE_SHIFTK = qNum('grassshiftk', 0.22, 0, 1);
const BAKE_THICKK = qNum('grassthickk', 0.18, 0, 2);
const RAY_BOMB = new URLSearchParams(window.location.search).get('grassbomb') !== '0';
const RAY_SHEAR = new URLSearchParams(window.location.search).get('grassshear') !== '0';
/** static per-tile swirl lean (?grasstilt=0 off): rides the SAME oblique basis as
 *  the wind (the article's variable-incline-fibers case) — whole 0.84 m patches
 *  lean in hash-varied directions, breaking the straight-vertical-prism look. */
const RAY_TILT = new URLSearchParams(window.location.search).get('grasstilt') !== '0';
/** sway amplitude scale (?grasssway=K, 0 = steady gust-bend only) */
const RAY_SWAY = qNum('grasssway', 1, 0, 5);
/** golden-angle overlay layer (?grasslayers=1 disables): the article's layered
 *  anti-tiling — tile space rotated by φ·π with an independent arc direction.
 *  Fetched ONLY where layer 1 landed no blade: fills the top-down holes with
 *  criss-cross sweeps for ~zero cost in dense sward (user placement call). */
const RAY_LAYER2 = Math.round(qNum('grasslayers', 2, 1, 2)) === 2;
/** ?grassrayend=N — band-end knob for cost attribution (default 155) */
const RAY_END = ((): number => {
  const v = Number(new URLSearchParams(window.location.search).get('grassrayend') ?? '155');
  return Number.isFinite(v) && v >= 20 && v <= 300 ? v : 155;
})();
/** G-D LEAN resolve lighting (?grasslean=1, MEASURED NO-WIN 2026-07-04 → opt-in):
 *  bake sunVis (PCSS×cloud×far) + probe irradiance PER GUIDE TEXEL, resolve takes
 *  ONE filtered tap instead of the shadow-upsample + GI chain. Built on the
 *  "resolve wall" hypothesis — REFUTED by direct A/B: lean on/off identical at
 *  eye (p50 24.9/25.0 @dpr1.5), even under ?shalfres=0 full-res PCSS. The ~7 ms
 *  hwnoemit delta is the EMIT PATH (election atomics × blade overdraw), not the
 *  resolve shading. Kept opt-in: correct, verified, may win on tap-bound GPUs. */
const GRASS_LEAN = new URLSearchParams(window.location.search).get('grasslean') === '1';
const RAY_SHELL_H = 1.5; // max blade reach above ground (incl. mid-card 2× + wind)
// ---- GUIDE FIELD (ray lane) — the precomputed-intersection lever ---------------------
// A camera-centered world-space context field REBAKED EVERY FRAME by a tiny compute
// pass (O(area), blade-count-independent), so the per-pixel march FETCHES its world
// instead of deriving it in-register (the kernel is occupancy-bound: 47.6→15.1 ms came
// from shrinking live state, not ALU). Per 0.84 m texel (= 8×8 fine cells):
//   ctx  (uvec4): ground+disp f32 | ground gradient half2 | (swardTop, gustAmp) half2
//   mask (uvec2): 64-bit fine-cell occupancy — the DENSITY LAW baked to bits
//     (bit = cellHash(cell) < dens·thin·edge, the exact kernel accept)
// March: 1 ctx load per texel step (replaces 4-tap heightAt + widen/top math), empty
// texels (dirt gaps / banks / canopy scruff) skip on mask==0 without hashing — the
// measured near-band fire was exactly these miss-path pixels burning clump batteries.
// Blades root on PLANE-RECONSTRUCTED ground (center + gradient·Δ) — exact on slopes,
// which fixes the user-reported sunken blades (burst-level ground was up to ~0.5 m
// low). Wind stays fully animated: amp is rebaked per frame. Storage buffers, not
// textures (uint StorageTexture is mistyped 'float' by the node builder; float-texture
// roundtrips could canonicalize mask NaN bit patterns).
const GUIDE_SUB = 8; // fine cells per texel edge
const GUIDE_PITCH = CELL * GUIDE_SUB; // 0.84 m
const GUIDE_RES = Math.max(384, Math.ceil(((RAY_END + 4) * 2) / GUIDE_PITCH)); // ≥ ray reach
const GUIDE_N = GUIDE_RES * GUIDE_RES; // 384² = 147k texels ≈ 3.5 MB total
const NEAR_EPS = 1e-4;

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


export interface GrassBuildOpts {
  cam: NaniteCam;
  vis: NaniteVisBuffers;
  hf: Heightfield;
  canopyTex: StorageTexture | null;
  /** terrain micro-displacement (NaniteFrame's disp) — blades must root on the
   *  DISPLACED surface or short blades sink into the near-field relief. */
  disp?: TerrainDisp;
}


/** lighting providers for the per-texel light bake (built AFTER the grass field —
 *  the shadow system doesn't exist yet when buildGrassField runs) */
export interface GrassLightOpts {
  /** composed sun visibility: clipmap PCSS × cloud transmittance × far-shadow
   *  (the water-arc sunVis closure — NaniteFrame builds it from NaniteShadow).
   *  pix = explicit IGN-noise coord (compute has no fragCoord — MUST be passed) */
  sunVis: (wp: NV3, n: NV3, pix?: NV2) => NF;
  /** probe-GI irradiance (world.gi); null → ambient floor only */
  gi: { irradiance(wp: NV3, n: NV3, lift?: number, groundY?: NF): NV3 } | null;
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
  /** G-D lean lighting: one filtered tap of the per-frame texel light field →
   *  vec4(sunVis, irradianceRGB). null when the lean path is off (?grasslean=0,
   *  geo lane, or attachLightBake never called). Call INSIDE the resolve Fn. */
  resolveLean: ((wpXZ: NV2) => NV4) | null;
  /** G-E article lane (?grass=ray): the algorithm's own output IS depth+normal —
   *  kRay writes the baked-fetch hit normal + tip param per pixel into a screen
   *  StorageTexture; the resolve taps it instead of the analytic resolveDerive
   *  (whose per-blade id decode the article lane doesn't have). vec4(nrm, t).
   *  null on every other lane. */
  resolveRay: ((px: NU) => NV4) | null;
  /** attach the per-texel light bake kernel (call once the shadow system + GI
   *  exist — NaniteFrame, after buildNaniteShadow*). No-op on non-lean lanes. */
  attachLightBake(l: GrassLightOpts): void;
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

  // ---- density law (GroundRing cull verbatim) -----------------------------------------
  const byBio = (b: NF, vals: number[]): NF => {
    let e: NF = float(vals[5] ?? 0) as unknown as NF;
    for (let i = 4; i >= 0; i--) {
      e = b.equal(float(i)).select(float(vals[i] ?? 0), e) as unknown as NF;
    }
    return e;
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

  /** legacy resolve hook — NEVER CALLED on this lane (grassProc.ray supplies
   *  normal+tip from the raycast's screen texture; the analytic per-blade id
   *  reconstruction died with the geo lane). Kept as a type-shaped stub. */
  const resolveDerive = (_body: NU, _wp: NV3): { t: NF; nrm: NV3 } => ({
    t: float(0.55) as unknown as NF,
    nrm: vec3(0, 1, 0) as unknown as NV3,
  });

  /** the ray lane emits per-pixel — no counters exist */
  const readCounts = async (): Promise<{ clumps: number; hwTris: number }> => ({
    clumps: 0,
    hwTris: 0,
  });

  // ================= THE RAY LANE (the only lane) ==================================
  // Per-pixel raycast: guide-texel walk + ONE baked-tile fetch per in-sward step.
  // A hit emits a self-describing id + depth into the election, so the resolve/
  // shadows/GTAO/TRAA pipeline is untouched; normal+tip ride rayNrmTex.

  // ---- guide buffers + per-frame bake (ray lane only; 4-word ctx + 2-word mask) ------
  const gWords = GUIDE_N * 4;
  const guideCtxAttr = new StorageBufferAttribute(new Uint32Array(gWords), 1);
  const guideCtxW = sU32Views(guideCtxAttr, gWords);
  const guideCtx4 = sUvec4RO(guideCtxAttr, GUIDE_N);
  const mWords = GUIDE_N * 2;
  const guideMaskAttr = new StorageBufferAttribute(new Uint32Array(mWords), 1);
  const guideMaskW = sU32Views(guideMaskAttr, mWords);
  const guideMask2 = sUvec2(guideMaskAttr, GUIDE_N);
  /** guide origin = fine-cell index of texel (0,0)'s first cell, snapped to the
   *  8-cell texel grid — mask bits stay congruent with WORLD cells (the field is
   *  world-anchored, only the window moves). Integer-valued floats, exact ≤ 2^23. */
  const uGFx = uniformF(0);
  const uGFz = uniformF(0);

  /** per-texel FIELD bake (perf): the march re-derived every smooth field PER
   *  STEP (3× value noise = 12 hashes + swirl/wind trig) — measured ~12 ms of
   *  kRay's 23 @dpr2 eye. The SAME 2.5-texel-period fields are now evaluated
   *  once per texel per frame (the guide rebakes every frame for wind anyway)
   *  and HW-bilinear-sampled at the ray's world pos: piecewise-BILINEAR is
   *  continuous across space — the grid class was piecewise-CONSTANT.
   *  T1 = (Slx, Slz, Swx, Swz) lean + wind quad-term (time folded at bake);
   *  T2 = (ca, sa, a1x, a1z) swirl rotation + static arc. L1's quad = Sw + a1,
   *  L2's = Sw − a1 (negated arc — still criss-crossing, no third texture). */
  const mkFieldTex = (name: string): StorageTexture => {
    const t = new StorageTexture(GUIDE_RES, GUIDE_RES);
    t.type = HalfFloatType;
    t.format = RGBAFormat;
    t.magFilter = LinearFilter;
    t.minFilter = LinearFilter;
    t.generateMipmaps = false;
    t.name = name;
    return t;
  };
  const guideFieldT1 = mkFieldTex('grassGuideField1');
  const guideFieldT2 = mkFieldTex('grassGuideField2');

  const kGuideBake = ((): unknown => {
    const k = Fn(() => {
      returnIf((uOn as unknown as NF).lessThan(0.5) as unknown as NB);
      const i = instanceIndex;
      returnIf(i.greaterThanEqual(uint(GUIDE_N)));
      const tx = toF(i.mod(uint(GUIDE_RES)));
      const tz = toF(i.div(uint(GUIDE_RES)));
      // fine-cell base + world-space center of this texel
      const fb = vec2(uGFx as unknown as NF, uGFz as unknown as NF)
        .add(vec2(tx, tz).mul(GUIDE_SUB))
        .toVar() as unknown as NV2;
      const wpos = fb.add(GUIDE_SUB / 2).mul(CELL).toVar() as unknown as NV2;
      const dist = wpos.sub(vec2(cam.camPos.x, cam.camPos.z)).length().toVar() as unknown as NF;
      // ground (heightfield + micro-displacement) and its gradient (central diff at
      // ±half pitch). Blades plane-reconstruct off these: ≤ cm error at 0.84 m pitch
      // over the ~1 m-bilinear heightfield; meadow disp amplitude is ~0.04 m
      // (veg-gated) — this is what fixes the lite-path sunken blades on slopes.
      const g = groundAt(wpos).toVar() as unknown as NF;
      const hp = GUIDE_PITCH / 2;
      const dgdx = groundAt(wpos.add(vec2(hp, 0)) as unknown as NV2)
        .sub(groundAt(wpos.sub(vec2(hp, 0)) as unknown as NV2))
        .div(GUIDE_PITCH) as unknown as NF;
      const dgdz = groundAt(wpos.add(vec2(0, hp)) as unknown as NV2)
        .sub(groundAt(wpos.sub(vec2(0, hp)) as unknown as NV2))
        .div(GUIDE_PITCH) as unknown as NF;
      // per-cell accept probability: the density law sampled at the FOUR texel
      // CORNERS and bilinearly blended PER CELL. A single texel-center sample
      // (the old law) made pT a 0.84 m STAIRCASE — in sparse/gradient meadows
      // adjacent texels got visibly different fill, the user's persistent grid
      // quilt (2026-07-04 pics; every other per-tile field had already been
      // smoothed). Corners are shared between neighbors ⇒ the per-cell field is
      // CONTINUOUS across the world (and closer to kFine's exact per-cell law —
      // tighter hybrid seam).
      const pCorner = (dx: number, dz: number): NF => {
        const cw = wpos.add(vec2(dx * (GUIDE_PITCH / 2), dz * (GUIDE_PITCH / 2))) as unknown as NV2;
        const dC = cw.sub(vec2(cam.camPos.x, cam.camPos.z)).length() as unknown as NF;
        const eC = float(1).sub(smoothstep(R * 0.9, R, dC)) as unknown as NF;
        return densityAt(cw, heightAt(cw), dC, true)
          .mul(grassThin(dC))
          .mul(eC)
          .toVar() as unknown as NF;
      };
      const p00 = pCorner(-1, -1);
      const p10 = pCorner(1, -1);
      const p01 = pCorner(-1, 1);
      const p11 = pCorner(1, 1);
      const pMax = p00.max(p10).max(p01).max(p11) as unknown as NF;
      // 64-bit occupancy: bit v·8+u = fine cell (fb.x+u, fb.y+v) holds a clump.
      // SAME accept hash as the geo kFine (0x77a1) — the hybrid seam needs BOTH
      // lanes to place the identical clump set or clumps reshuffle at the boundary.
      const m0 = uint(0).toVar() as unknown as NU;
      const m1 = uint(0).toVar() as unknown as NU;
      If(pMax.greaterThan(0), () => {
        loopUN('gbv', uint(0), uint(GUIDE_SUB), (v) => {
          loopUN('gbu', uint(0), uint(GUIDE_SUB), (u) => {
            const wcF = fb.add(vec2(toF(u), toF(v))) as unknown as NV2;
            const fu = toF(u).add(0.5).mul(1 / GUIDE_SUB) as unknown as NF;
            const fv = toF(v).add(0.5).mul(1 / GUIDE_SUB) as unknown as NF;
            const pC = mix(mix(p00, p10, fu), mix(p01, p11, fu), fv) as unknown as NF;
            If(cellHash(wcF, SALT ^ 0x77a1).lessThan(pC), () => {
              const bit = (v as unknown as NU).mul(uint(GUIDE_SUB)).add(u) as unknown as NU;
              If(bit.lessThan(uint(32)), () => {
                (m0 as unknown as { assign(v: unknown): void }).assign(
                  m0.bitOr(uint(1).shiftLeft(bit)),
                );
              }).Else(() => {
                (m1 as unknown as { assign(v: unknown): void }).assign(
                  m1.bitOr(uint(1).shiftLeft(bit.bitAnd(uint(31)))),
                );
              });
            });
          });
        });
      });
      // conservative sward-top offset (the kernel's tight topB law) — 0 when empty,
      // which doubles as the march's "nothing here" flag
      const thin = grassThin(dist);
      const widenT = float(1)
        .div(thin.sqrt())
        .clamp(1, 4)
        .sub(1)
        .mul(0.3)
        .add(1) as unknown as NF;
      const topOff = dist
        .greaterThan(60)
        .select(float(1.0).mul(widenT), float(0.62).mul(widenT))
        .min(RAY_SHELL_H)
        .add(0.25) as unknown as NF;
      const occ = m0.bitOr(m1).notEqual(uint(0));
      const topOut = occ.select(topOff, float(0)) as unknown as NF;
      // gust amplitude at the texel — rebaked EVERY frame, so wind stays live
      const amp = (windContext()
        ? (windU.strength as unknown as NF)
            .mul(gustAt(wpos).mul(0.9).add(0.3))
            .mul(windExposure(wpos))
        : (float(0) as unknown as NF)) as unknown as NF;
      const base = i.mul(uint(4));
      guideCtxW.rw.element(base).assign(bcF2U(g));
      guideCtxW.rw.element(base.add(uint(1))).assign(packHalfU(vec2(dgdx, dgdz) as unknown as NV2));
      guideCtxW.rw.element(base.add(uint(2))).assign(packHalfU(vec2(topOut, amp) as unknown as NV2));
      // word 3: widenT (kRay ladder widen) — baked here so the march skips the
      // per-step 1/sqrt(grassThin) re-derive; exact f32 bitcast (was dead uint(0))
      guideCtxW.rw.element(base.add(uint(3))).assign(bcF2U(widenT));
      const mb = i.mul(uint(2));
      guideMaskW.rw.element(mb).assign(m0);
      guideMaskW.rw.element(mb.add(uint(1))).assign(m1);
      // ---- per-texel FIELD bake (guideFieldT1/T2 — see decl): the kRay per-step
      // smNoise+trig monster, folded to texel rate. Same fields, same salts.
      const smN = (salt: number): NV2 => {
        const qx = wpos.x.mul(1 / (GUIDE_PITCH * 2.5)) as unknown as NF;
        const qz = wpos.y.mul(1 / (GUIDE_PITCH * 2.5)) as unknown as NF;
        const ix = qx.floor().toVar() as unknown as NF;
        const iz = qz.floor().toVar() as unknown as NF;
        const fx = smoothstep(0, 1, qx.sub(ix)) as unknown as NF;
        const fz = smoothstep(0, 1, qz.sub(iz)) as unknown as NF;
        const c = (dx: number, dz: number): NV2 =>
          cellHash2(
            vec2(ix.add(dx), iz.add(dz)) as unknown as NV2,
            SALT ^ salt,
          ) as unknown as NV2;
        return mix(
          mix(c(0, 0), c(1, 0), fx),
          mix(c(0, 1), c(1, 1), fx),
          fz,
        ) as unknown as NV2;
      };
      let caB: NF = float(1) as unknown as NF;
      let saB: NF = float(0) as unknown as NF;
      if (RAY_BOMB) {
        const th = (smN(0x0b0b).x as unknown as NF).mul(6.2831853).toVar() as unknown as NF;
        caB = th.cos() as unknown as NF;
        saB = th.sin() as unknown as NF;
      }
      let SwxB: NF = float(0) as unknown as NF;
      let SwzB: NF = float(0) as unknown as NF;
      if (RAY_SHEAR && windContext()) {
        const st = (windU.strength as unknown as NF).toVar() as unknown as NF;
        const K = amp
          .mul(st.mul(0.55).add(0.6))
          .mul(0.45)
          .div(topOut.max(0.35))
          .toVar() as unknown as NF;
        const wd = vec2(windU.dir as unknown as NV2);
        const swn = smN(0x5151);
        const ph = (swn.x as unknown as NF).mul(6.2831853).toVar() as unknown as NF;
        const gustE = time.mul(0.22).add(ph).sin().mul(0.35).add(0.65) as unknown as NF;
        const lowS = time.mul(0.55).add(ph).sin() as unknown as NF;
        const highS = time.mul(2.4).add(ph.mul(1.7)).sin() as unknown as NF;
        const shelter = amp.mul(1.5).add(0.25).min(1) as unknown as NF;
        const ffall = float(1).sub(smoothstep(50, 110, dist)) as unknown as NF;
        const swayA = lowS
          .mul(gustE)
          .mul(0.45)
          .add(highS.mul(0.1))
          .mul(st.mul(0.7).add(0.15))
          .mul(shelter)
          .mul(ffall)
          .mul(RAY_SWAY)
          .toVar() as unknown as NF;
        const wob = (swn.y as unknown as NF).sub(0.5).mul(0.8) as unknown as NF;
        const swx = wd.x.sub(wd.y.mul(wob)) as unknown as NF;
        const swz = wd.y.add(wd.x.mul(wob)) as unknown as NF;
        SwxB = wd.x.mul(K).add(swx.mul(swayA)) as unknown as NF;
        SwzB = wd.y.mul(K).add(swz.mul(swayA)) as unknown as NF;
      }
      let SlxB: NF = float(0) as unknown as NF;
      let SlzB: NF = float(0) as unknown as NF;
      let a1xB: NF = float(0) as unknown as NF;
      let a1zB: NF = float(0) as unknown as NF;
      if (RAY_TILT) {
        // staticArc(0x3333) + the ~120°-offset lean, riding the same noise
        const n = smN(0x3333);
        const ba = (n.x as unknown as NF).mul(6.2831853).toVar() as unknown as NF;
        const bm = (n.y as unknown as NF).mul(0.25).add(0.12) as unknown as NF;
        a1xB = ba.cos().mul(bm) as unknown as NF;
        a1zB = ba.sin().mul(bm) as unknown as NF;
        const la = (n.x as unknown as NF).mul(6.2831853).add(2.1) as unknown as NF;
        const lm = (n.y as unknown as NF).mul(0.05).add(0.02) as unknown as NF;
        SlxB = la.cos().mul(lm) as unknown as NF;
        SlzB = la.sin().mul(lm) as unknown as NF;
      }
      const txu = i.mod(uint(GUIDE_RES));
      const tzu = i.div(uint(GUIDE_RES));
      textureStore(
        guideFieldT1,
        uvec2(txu, tzu),
        vec4(SlxB, SlzB, SwxB, SwzB),
      ).toWriteOnly();
      textureStore(
        guideFieldT2,
        uvec2(txu, tzu),
        vec4(caB, saB, a1xB, a1zB),
      ).toWriteOnly();
    })().compute(GUIDE_N, [256]);
    (k as unknown as { setName(n: string): void }).setName('grassGuide');
    return k;
  })();

  // ---- G-D per-texel LIGHT bake (lean resolve): sunVis + probe irradiance ----------
  // rgba16float StorageTexture — filterable (the resolve gets HW bilinear in ONE
  // tap) and a texture, not a buffer (the resolve fragment rides the 10-storage-
  // buffer ceiling; float lighting has none of the uint-mistype/NaN-bit hazards
  // that forced the ctx/mask to buffers). The kernel is attached LATER
  // (attachLightBake) because the shadow system is built after the grass field.
  // It reads THIS frame's guide ctx (own dispatch, after kGuideBake) and lights
  // the texel at sward mid-height with the terrain normal; shadow maps are the
  // last-updated ones (toroidal clipmap ≈ static — 1-frame lag is invisible).
  const guideLightTex = ((): StorageTexture | null => {
    if (!GRASS_LEAN) return null;
    const t = new StorageTexture(GUIDE_RES, GUIDE_RES);
    t.type = HalfFloatType;
    t.format = RGBAFormat;
    t.magFilter = LinearFilter;
    t.minFilter = LinearFilter;
    t.generateMipmaps = false;
    t.name = 'grassGuideLight';
    return t;
  })();
  let kGuideLight: unknown = null;
  const attachLightBake = (l: GrassLightOpts): void => {
    if (!guideLightTex || kGuideLight) return;
    const k = Fn(() => {
      returnIf((uOn as unknown as NF).lessThan(0.5) as unknown as NB);
      const i = instanceIndex;
      returnIf(i.greaterThanEqual(uint(GUIDE_N)));
      const tx = i.mod(uint(GUIDE_RES));
      const tz = i.div(uint(GUIDE_RES));
      const fb = vec2(uGFx as unknown as NF, uGFz as unknown as NF).add(
        vec2(toF(tx), toF(tz)).mul(GUIDE_SUB),
      ) as unknown as NV2;
      const wpos = fb.add(GUIDE_SUB / 2).mul(CELL).toVar() as unknown as NV2;
      const dist = wpos.sub(vec2(cam.camPos.x, cam.camPos.z)).length() as unknown as NF;
      // circle gate — the square guide's corners are beyond grass reach (−21%)
      returnIf(dist.greaterThan(R + 1) as unknown as NB);
      const cv = guideCtx4.element(i);
      const ground = bcU2F(cv.x as unknown as NU).toVar() as unknown as NF;
      const grad = unpackHalfU(cv.y as unknown as NU).toVar() as unknown as NV2;
      const ta = unpackHalfU(cv.z as unknown as NU).toVar() as unknown as NV2;
      const tn = normalize(
        vec3(grad.x.negate(), 1, grad.y.negate()) as unknown as NV3,
      ) as unknown as NV3;
      const wp3 = vec3(
        wpos.x,
        ground.add(ta.x.mul(0.5)).add(0.05),
        wpos.y,
      ) as unknown as NV3;
      // IGN noise coord = the texel index (compute has no fragCoord — the
      // ShadowHalf idiom); per-texel phase keeps the PCSS dither decorrelated
      const sun = l.sunVis(wp3, tn, vec2(toF(tx), toF(tz)) as unknown as NV2).clamp(0, 1) as unknown as NF;
      let irr = (
        l.gi ? l.gi.irradiance(wp3, tn, 2.0, heightAt(wpos)) : (vec3(0) as unknown as NV3)
      ) as unknown as NV3;
      if (l.gi && canopyTex) {
        // same canopy damping the resolve's full GI path applies
        irr = irr.mul(canopyAt(canopyTex, wpos).mul(0.18).oneMinus()) as unknown as NV3;
      }
      textureStore(guideLightTex, uvec2(tx, tz), vec4(sun, irr)).toWriteOnly();
    })().compute(GUIDE_N, [256]);
    (k as unknown as { setName(n: string): void }).setName('grassLight');
    kGuideLight = k;
  };
  /** lean resolve tap: world XZ → guide uv → ONE filtered sample. Texel i's data
   *  sits at uv (i+0.5)/RES; a point at a texel center has (wp/CELL − gf)/8 =
   *  i+0.5 — the mapping is exactly center-aligned, HW bilinear does the rest. */
  const resolveLean = guideLightTex
    ? (wpXZ: NV2): NV4 => {
        const uv = wpXZ
          .div(CELL)
          .sub(vec2(uGFx as unknown as NF, uGFz as unknown as NF))
          .div(GUIDE_SUB * GUIDE_RES)
          .clamp(0, 1) as unknown as NV2;
        return texture(guideLightTex, uv, 0) as unknown as NV4;
      }
    : null;

  // ---- G-E: THE ARTICLE'S PRECOMPUTATION (⚠️ USER DIRECTIVE 2026-07-04) ---------------
  // Boot-baked raycast tile (GrassRayBake.ts): 3D texture (x, z in tile, angle) →
  // R = path length 1/(1+d), GBA = normal — traced over infinitely-tiled FULL-density
  // clump geometry with the article's shift+thicken heuristics. LINEAR filter +
  // REPEAT wrap on all three axes (his interpolation, incl. across angle slices).
  // The tile = one guide texel footprint (0.84 m, 8×8 fine cells).
  /** height-banded volumes (?grassbands=2|3): real per-band taper/arc parallax,
   *  but band-hop refetches cost ~2.5× on look-down poses — default 1 (the
   *  per-fiber arcs live in the bake at mid-height; root-id validation carries
   *  the overhang) */
  const RAY_BANDS = Math.round(qNum('grassbands', 1, 1, 3));
  const rayBake = ((): { texs: Data3DTexture[]; dMaxTile: number } | null => {

    const b = bakeGrassRayTile({
      res: BAKE_RES,
      angles: BAKE_ANG,
      blades: BLADES,
      sub: GUIDE_SUB,
      cellM: CELL,
      shiftK: BAKE_SHIFTK,
      thickK: BAKE_THICKK,
      // user-called 2026-07-04: 0.011/0.0045 read as FAT uniform columns — the
      // original blades' visually dominant upper half is ≤15 mm and edge-on ~4 mm
      halfW: qNum('grassbakw', 0.0055, 0.001, 0.05),
      halfT: qNum('grassbakt', 0.003, 0.001, 0.02),
      // 8 spread fibers per cell (was 5 clumped — the dot-tufts-with-holes call);
      // bake-side density is FREE at runtime (shorter fetch distances)
      fibers: Math.round(qNum('grassbakn', 8, 2, 16)),
      // height-banded volumes (user grid+batch calls): per-fiber radial arcs +
      // real tip taper live in the bake; runtime picks a band by ray height
      bands: RAY_BANDS,
      arcK: qNum('grassarck', 1, 0, 3),
    });
    const texs = b.data.map((d, i) => {
      const t = new Data3DTexture(d, b.res, b.res, b.angles);
      t.format = RGBAFormat;
      t.minFilter = LinearFilter;
      t.magFilter = LinearFilter;
      t.wrapS = RepeatWrapping;
      t.wrapT = RepeatWrapping;
      t.wrapR = RepeatWrapping; // angle axis wraps (θ is periodic)
      t.generateMipmaps = false;
      t.needsUpdate = true;
      t.name = `grassRayTile${i}`;
      return t;
    });
    return { texs, dMaxTile: b.dMaxTile };
  })();
  // the article's OUTPUT is depth+normal — the hit normal/tip can't ride the 30-bit
  // election id, so kRay writes them per pixel into a screen StorageTexture (a
  // texture, not a buffer: the resolve fragment is at the 10-storage-buffer ceiling).
  const rayNrmTex = ((): StorageTexture | null => {
    if (!rayBake) return null;
    const t = new StorageTexture(cam.width, cam.height);
    t.type = HalfFloatType;
    t.format = RGBAFormat;
    t.magFilter = NearestFilter;
    t.minFilter = NearestFilter;
    t.generateMipmaps = false;
    t.name = 'grassRayNrm';
    return t;
  })();
  /** resolve-side tap: pixel index (bottom-up rows, the election convention —
   *  kRay stores with the same row indexing, so no flip) → vec4(worldNrm, t) */
  const resolveRay = rayNrmTex
    ? (px: NU): NV4 => {
        const uv = vec2(
          toF(px.mod(uint(cam.width))).add(0.5).div(cam.width),
          toF(px.div(uint(cam.width))).add(0.5).div(cam.height),
        ) as unknown as NV2;
        return texture(rayNrmTex, uv, 0) as unknown as NV4;
      }
    : null;

  const kRay = ((): unknown => {
    const W = cam.width;
    const H = cam.height;
    /** march-res decouple (?grassquad=1|2): the march cost is ∝ rays × steps —
     *  at retina dpr2 the full-res dispatch alone doubled grass cost vs dpr1.5.
     *  Q=2 marches ONE ray per 2×2 pixel quad (quad-center) and fans the hit out
     *  to the quad's pixels. True-geometry edges stay pixel-crisp — the election
     *  atomicMax is still per PIXEL, so nearer scene depth wins individually;
     *  only grass-over-background silhouettes quantize to the quad (≈ dpr1
     *  grass edges, TRAA-softened). Auto-on at dpr ≥ 1.75, off below. */
    const Q = Math.round(
      qNum('grassquad', W / Math.max(1, window.innerWidth) >= 1.75 ? 2 : 1, 1, 2),
    );
    const Wq = Math.ceil(W / Q);
    const Hq = Math.ceil(H / Q);
    // (8×8 pixel tiling was measured NEUTRAL-to-worse here — rows are already
    // coherent; linear indexing kept)
    const k = Fn(() => {
      returnIf((uOn as unknown as NF).lessThan(0.5) as unknown as NB);
      const qi = instanceIndex;
      returnIf(qi.greaterThanEqual(uint(Wq * Hq)));
      const xI = qi.mod(uint(Wq)).mul(uint(Q)).toVar() as unknown as NU; // base pixel
      const yI = qi.div(uint(Wq)).mul(uint(Q)).toVar() as unknown as NU; // bottom-up rows
      const px = yI.mul(uint(W)).add(xI).toVar() as unknown as NU;
      const ndcX = toF(xI).add(Q * 0.5).div(W).mul(2).sub(1);
      const ndcY = toF(yI).add(Q * 0.5).div(H).mul(2).sub(1);
      const hf4 = cam.invVp.mul(vec4(ndcX, ndcY, 1, 1));
      const ro = vec3(cam.camPos).toVar() as unknown as NV3;
      const rd = (hf4.xyz.div(hf4.w).sub(ro).normalize().toVar()) as unknown as NV3;
      // scene early-out: current election depth bounds the march. Q>1: the
      // FARTHEST bound across the quad's pixels (conservative — a nearer
      // neighbor must not clip a farther pixel's grass), unbounded if any
      // pixel is electionless.
      const tMax = float(1e9).toVar() as unknown as NF;
      if (Q === 1) {
        const elect = aLoadU(vis.payloadV.atomic.element(px));
        If(elect.notEqual(uint(0)), () => {
          const czS = float(1).sub(toF(elect.shiftRight(uint(8))).div(16777215));
          const hs = cam.invVp.mul(vec4(ndcX, ndcY, czS, 1));
          (tMax as unknown as { assign(v: unknown): void }).assign(
            hs.xyz.div(hs.w).sub(ro).length().add(0.3),
          );
        });
      } else {
        const bound = float(0).toVar() as unknown as NF;
        for (let dy = 0; dy < Q; dy++) {
          for (let dx = 0; dx < Q; dx++) {
            const x2 = uint(toF(xI.add(uint(dx))).min(W - 1)) as unknown as NU;
            const y2 = uint(toF(yI.add(uint(dy))).min(H - 1)) as unknown as NU;
            const e2 = aLoadU(vis.payloadV.atomic.element(y2.mul(uint(W)).add(x2)));
            If(e2.equal(uint(0)), () => {
              (bound as unknown as { assign(v: unknown): void }).assign(1e9);
            }).Else(() => {
              const czS = float(1).sub(toF(e2.shiftRight(uint(8))).div(16777215));
              const hs = cam.invVp.mul(vec4(ndcX, ndcY, czS, 1));
              (bound as unknown as { assign(v: unknown): void }).assign(
                bound.max(hs.xyz.div(hs.w).sub(ro).length().add(0.3)),
              );
            });
          }
        }
        (tMax as unknown as { assign(v: unknown): void }).assign(bound.min(1e9));
      }
      const dirL = rd.xz.length().max(1e-4).toVar() as unknown as NF;
      const tEnd = tMax.min(float(RAY_END).div(dirL)).toVar() as unknown as NF;
      const tCur = float(0.05).toVar() as unknown as NF;
      const tBest = float(1e9).toVar() as unknown as NF;
      const bodyBest = uint(0).toVar() as unknown as NU;
      // G-E article lane: the accepted hit's baked normal + tip param, stored to
      // the screen StorageTexture next to the election emit (the article's output
      // is depth+normal — see rayNrmTex above)
      const nrmV = rayBake ? (vec3(0, 1, 0).toVar() as unknown as NV3) : null;
      const tParV = rayBake ? (float(0.5).toVar() as unknown as NF) : null;
      /** per-PIXEL golden-layer budget — L2 is a hole-filler, not a second march */
      const l2n = rayBake && RAY_LAYER2 ? (uint(0).toVar() as unknown as NU) : null;
      if (GRASS_DBG === 'raysetup') {
        // attribution stop: ray gen + scene-depth reconstruct only
        If(tEnd.lessThan(-1), () => {
          emitPx(px as unknown as NU, tCur as unknown as NF, bodyBest);
        });
        returnIf(tBest.greaterThan(0) as unknown as NB);
      }


      // ---- GUIDE-DRIVEN MARCH: texel DDA → 1 uvec4 fetch → descend on occupancy ------
      // Everything the old march DERIVED per step/burst (4-tap ground, density taps,
      // gust taps, disp, sward-top math) is a FETCH from the per-frame guide bake —
      // the kernel is occupancy-bound, and this collapses its live-register state.
      const gfx = uGFx as unknown as NF;
      const gfz = uGFz as unknown as NF;
      const MAXTOP = RAY_SHELL_H + 0.25; // conservative shell for Lipschitz jumps
      const sdx = rd.x
        .greaterThanEqual(0)
        .select(rd.x.max(1e-6), rd.x.min(-1e-6))
        .toVar() as unknown as NF;
      const sdz = rd.z
        .greaterThanEqual(0)
        .select(rd.z.max(1e-6), rd.z.min(-1e-6))
        .toVar() as unknown as NF;
      loopUN('gro', uint(0), uint(256), () => {
        If(tCur.greaterThanEqual(tEnd).or(tCur.greaterThan(tBest.add(0.3))), () => {
          Break();
        });
        const pos = ro.add(rd.mul(tCur)).toVar() as unknown as NV3;
        // guide texel under pos (fine-cell space → texel index; in-range by
        // construction — tEnd caps horizontal travel inside the guide window)
        const txf = pos.x
          .div(CELL)
          .sub(gfx)
          .div(GUIDE_SUB)
          .floor()
          .clamp(0, GUIDE_RES - 1)
          .toVar() as unknown as NF;
        const tzf = pos.z
          .div(CELL)
          .sub(gfz)
          .div(GUIDE_SUB)
          .floor()
          .clamp(0, GUIDE_RES - 1)
          .toVar() as unknown as NF;
        const ti = uint(tzf.mul(GUIDE_RES).add(txf)).toVar() as unknown as NU;
        const cv = guideCtx4.element(ti);
        const ground = bcU2F(cv.x).toVar() as unknown as NF;
        const grad = (unpackHalfU(cv.y) as unknown as { toVar(): NV2 }).toVar() as unknown as NV2;
        const ta = (unpackHalfU(cv.z) as unknown as { toVar(): NV2 }).toVar() as unknown as NV2;
        // texel exit t (0.84 m world grid anchored at the guide origin)
        const bx = gfx
          .add(txf.add(rd.x.greaterThanEqual(0).select(float(1), float(0))).mul(GUIDE_SUB))
          .mul(CELL) as unknown as NF;
        const bz = gfz
          .add(tzf.add(rd.z.greaterThanEqual(0).select(float(1), float(0))).mul(GUIDE_SUB))
          .mul(CELL) as unknown as NF;
        const tEx = bx
          .sub(ro.x)
          .div(sdx)
          .min(bz.sub(ro.z).div(sdz))
          .max(tCur.add(1e-3))
          .toVar() as unknown as NF; // always progress
        const tExC = tEx.min(tEnd).toVar() as unknown as NF;
        // texel-level vertical reject: min ray Y across [tCur, tExC] (linear in t —
        // min is at an endpoint) vs the texel's baked sward top. topOff==0 ⇔ empty.
        const top = ground.add(ta.x) as unknown as NF;
        const rayYmin = ro.y.add(rd.y.mul(rd.y.lessThan(0).select(tExC, tCur))) as unknown as NF;
        If(ta.x.lessThanEqual(0).or(rayYmin.greaterThan(top)), () => {
          // skip the texel; when well above, add a terrain-Lipschitz jump vs the
          // CONSERVATIVE shell (the local top can jump 0→1.75 at mask boundaries,
          // so the tight local top must NOT drive the jump; ground slope bound 1.4)
          const head = pos.y.sub(ground).sub(MAXTOP) as unknown as NF;
          tCur.assign(tEx.add(1e-3).max(tCur.add(head.max(0).div(1.4))));
        }).Else(() => {
          const mv = guideMask2.ro.element(ti);
          const m0 = mv.x.toVar() as unknown as NU;
          const m1 = mv.y.toVar() as unknown as NU;
          // per-texel ladder context (dist-based; was per-burst before — same grain)
          const distT = tCur.mul(dirL) as unknown as NF;
          // widenT baked per-texel in kGuideBake (ctx word 3, exact f32 bitcast) —
          // was a per-step 1/sqrt(grassThin) chain (2×pow+sqrt+div); texel-center
          // dist vs step-exact is negligible (guide grain 0.84 m, widenT slowly-varying)
          const widenT = bcU2F(cv.w).toVar() as unknown as NF;
          // ---- G-E ARTICLE STEP (?grass=ray): ONE FETCH of the baked raycast tile
          // answers this march step — the article's O(1) core. Adaptations (the
          // "minor modifications"): hits are clamped to the CURRENT tile instance
          // (texture bombing changes the transform per tile, and the world density
          // mask varies — the walk just refetches in the next texel), the fetched
          // fiber is validated against the WORLD cell's occupancy bit (density law)
          // and blade-height law (the bake is 2D/height-free), and a reject steps
          // past the fiber and refetches (expected ≤2-3 per pixel at meadow fill).
          const bakedTexel = (): void => {
            if (!rayBake || !nrmV || !tParV) return;
            const txI = gfx.div(GUIDE_SUB).add(txf) as unknown as NF;
            const tzI = gfz.div(GUIDE_SUB).add(tzf) as unknown as NF;
            // ---- FIELD FETCH (perf rewrite 2026-07-04): every smooth per-step
            // field (swirl θ as cos/sin, wind quad-term, static arc, lean) comes
            // from the per-frame per-texel bake in kGuideBake, HW-bilinear at the
            // ray's CONTINUOUS world pos (resolveLean's center-aligned uv idiom).
            // ⚠️ GRID HISTORY: fields must be continuous ACROSS SPACE — piecewise-
            // BILINEAR of texel-center samples qualifies; the user's 0.84m quilt
            // was piecewise-CONSTANT fields. This replaces 3× value noise (12
            // hashes) + trig per march step — measured ~12 ms of kRay's 23 @dpr2.
            const guv = vec2(
              pos.x.div(CELL).sub(gfx).div(GUIDE_SUB * GUIDE_RES),
              pos.z.div(CELL).sub(gfz).div(GUIDE_SUB * GUIDE_RES),
            ).clamp(0, 1) as unknown as NV2;
            const f1 = (texture(guideFieldT1, guv, 0) as unknown as { toVar(): NV4 }).toVar();
            const f2 = (texture(guideFieldT2, guv, 0) as unknown as { toVar(): NV4 }).toVar();
            // swirl rotation: bilinear of (cos,sin) renormalized — bombI must be
            // the EXACT inverse of bombF (root-id lattice mapping needs ca²+sa²=1)
            const cl = vec2(f2.x, f2.y).length().max(1e-4).toVar() as unknown as NF;
            const ca = (f2.x as unknown as NF).div(cl).toVar() as unknown as NF;
            const sa = (f2.y as unknown as NF).div(cl).toVar() as unknown as NF;
            /** tile-space forward rotation */
            const bombF = (vx: NF, vz: NF): NV2 =>
              vec2(
                vx.mul(ca).sub(vz.mul(sa)),
                vx.mul(sa).add(vz.mul(ca)),
              ) as unknown as NV2;
            /** inverse rotation */
            const bombI = (vx: NF, vz: NF): NV2 =>
              vec2(
                vx.mul(ca).add(vz.mul(sa)),
                vz.mul(ca).sub(vx.mul(sa)),
              ) as unknown as NV2;
            // THE WIND — the article's march-space shear: an oblique TRUE-derivative
            // TBN basis (thetenthplanet.de/archives/1180; deliberately NON-orthonormal)
            // tilts the march space by the per-texel gust deflection while the baked
            // tile stays static. S = horizontal tip deflection per meter of height
            // (deriveClump's bendAmp law normalized by blade height); ta.y is rebaked
            // every frame, so the field animates.
            // the basis has a LINEAR part (whole-blade tilt, Sl) and a QUADRATIC
            // part (ARC — deflection Sq·h², the ring's bendAmp·tN² law). The march
            // uses the LOCAL tangent (∂off/∂y = Sl + 2·Sq·h), re-linearized at
            // every fetch, so blade silhouettes render genuinely CURVED (user call
            // 2026-07-04: "bends in the grass blades is a must") — zero extra
            // fetches, the cheapest bend variant.
            /** lean (linear term) + wind quad-term — baked in kGuideBake */
            const Slx = (f1.x as unknown as NF).toVar() as unknown as NF;
            const Slz = (f1.y as unknown as NF).toVar() as unknown as NF;
            const Swx = (f1.z as unknown as NF).toVar() as unknown as NF;
            const Swz = (f1.w as unknown as NF).toVar() as unknown as NF;
            /** quad basis = wind + static arc (f2.zw; 0 when ?grasstilt=0) */
            const Sqx = Swx.add(f2.z).toVar() as unknown as NF;
            const Sqz = Swz.add(f2.w).toVar() as unknown as NF;
            const texOx = gfx.add(txf.mul(GUIDE_SUB)).mul(CELL).toVar() as unknown as NF;
            const texOz = gfz.add(tzf.mul(GUIDE_SUB)).mul(CELL).toVar() as unknown as NF;
            const texCx = texOx.add(GUIDE_PITCH / 2).toVar() as unknown as NF;
            const texCz = texOz.add(GUIDE_PITCH / 2).toVar() as unknown as NF;
            // shear the CURRENT march point into tile space (height above the
            // texel's ground plane drives the shear), then bomb it
            const gP = ground
              .add(grad.x.mul(pos.x.sub(texCx)))
              .add(grad.y.mul(pos.z.sub(texCz))) as unknown as NF;
            // shear height clamped: past ~0.6 m the arc would wrap tile space by
            // whole tiles (far-field tall swards) — sub-pixel there anyway
            const hgt = pos.y.sub(gP).max(0).min(0.6).toVar() as unknown as NF;
            const offX = Slx.add(Sqx.mul(hgt)).mul(hgt) as unknown as NF; // Sl·h + Sq·h²
            const offZ = Slz.add(Sqz.mul(hgt)).mul(hgt) as unknown as NF;
            const lxT = pos.x.sub(offX).sub(texOx).div(GUIDE_PITCH) as unknown as NF;
            const lzT = pos.z.sub(offZ).sub(texOz).div(GUIDE_PITCH) as unknown as NF;
            const qb0 = bombF(lxT.sub(0.5), lzT.sub(0.5)) as unknown as NV2;
            const qbx = qb0.x.add(0.5).toVar() as unknown as NF;
            const qbz = qb0.y.add(0.5).toVar() as unknown as NF;
            // sheared march direction (LOCAL tangent of the quadratic basis) →
            // tile-space azimuth = the fetch's 3rd axis
            const tanX = Slx.add(Sqx.mul(hgt).mul(2)).toVar() as unknown as NF;
            const tanZ = Slz.add(Sqz.mul(hgt).mul(2)).toVar() as unknown as NF;
            const ex = rd.x.sub(tanX.mul(rd.y)).toVar() as unknown as NF;
            const ez = rd.z.sub(tanZ.mul(rd.y)).toVar() as unknown as NF;
            const eLen = vec2(ex, ez).length().max(1e-5).toVar() as unknown as NF;
            const ebT = bombF(ex as unknown as NF, ez as unknown as NF) as unknown as NV2;
            const ebx = ebT.x.toVar() as unknown as NF;
            const ebz = ebT.y.toVar() as unknown as NF;
            const az = (atan(ebz, ebx) as unknown as NF)
              .mul(1 / (Math.PI * 2))
              .fract() as unknown as NF;
            // HEIGHT BAND pick: per-fiber radial arcs + tip taper live in the
            // per-band bakes (a single height-free bake can't hold them — the
            // per-tile arc workaround combed tiles into the user's grid quilt)
            const HB = (ta.x as unknown as NF).mul(0.8).max(0.15).toVar() as unknown as NF;
            const hBandF = pos.y
              .sub(gP)
              .div(HB)
              .clamp(0, 0.999)
              .mul(RAY_BANDS)
              .floor()
              .toVar() as unknown as NF;
            const fetchBand = (u: NF, v: NF, w: NF): NV4 => {
              const smpAt = (i: number): NV4 =>
                texture3D(
                  (rayBake as { texs: Data3DTexture[] }).texs[i] as unknown as Parameters<
                    typeof texture3D
                  >[0],
                  vec3(u, v, w) as unknown as NV3,
                  0,
                ) as unknown as NV4;
              if (RAY_BANDS === 1) return (smpAt(0) as unknown as { toVar(): NV4 }).toVar();
              const out = vec4(0, 0, 0, 0).toVar() as unknown as NV4;
              const asg = (i: number): void => {
                (out as unknown as { assign(v: unknown): void }).assign(smpAt(i));
              };
              if (RAY_BANDS === 2) {
                If(hBandF.lessThan(1), () => {
                  asg(0);
                }).Else(() => {
                  asg(1);
                });
              } else {
                If(hBandF.lessThan(1), () => {
                  asg(0);
                })
                  .ElseIf(hBandF.lessThan(2), () => {
                    asg(1);
                  })
                  .Else(() => {
                    asg(2);
                  });
              }
              return out;
            };
            /** bilinear-smooth ground across texels (statTexel's gAt idiom) — the
             *  per-texel FACETED gRoot plane gave every tile's blades a coherent
             *  tip-param offset → a brightness step in the albedo ramp (part of
             *  the altitude grid). Accept-path only: the height TEST keeps the
             *  cheap facet (cm tolerance), only the SHADING t uses this. */
            const smoothGroundAt = (wx: NF, wz: NF): NF => {
              const qx = wx.div(CELL).sub(gfx).div(GUIDE_SUB).sub(0.5) as unknown as NF;
              const qz = wz.div(CELL).sub(gfz).div(GUIDE_SUB).sub(0.5) as unknown as NF;
              const ix = qx.floor().clamp(0, GUIDE_RES - 2).toVar() as unknown as NF;
              const iz = qz.floor().clamp(0, GUIDE_RES - 2).toVar() as unknown as NF;
              const fxg = qx.sub(ix).clamp(0, 1) as unknown as NF;
              const fzg = qz.sub(iz).clamp(0, 1) as unknown as NF;
              const gAt = (dx: number, dz: number): NF =>
                bcU2F(
                  guideCtx4.element(
                    uint(iz.add(dz).mul(GUIDE_RES).add(ix.add(dx))) as unknown as NU,
                  ).x as unknown as NU,
                ) as unknown as NF;
              return mix(
                mix(gAt(0, 0), gAt(1, 0), fxg),
                mix(gAt(0, 1), gAt(1, 1), fxg),
                fzg,
              ) as unknown as NF;
            };
            /** occupancy bit of an ARBITRARY world cell — cross-border hits root
             *  in the NEIGHBOR texel (the periodic copy's cells), so the owning
             *  texel's mask word is fetched when it differs (minority path).
             *  Validating copies against THIS texel's cells shifted the density
             *  field by ±1 tile at borders — the user's "sparsing happens at the
             *  wrong end of the square" pic. */
            const maskBitAt = (wcxA: NF, wczA: NF): NB => {
              const txN = wcxA
                .sub(gfx)
                .div(GUIDE_SUB)
                .floor()
                .clamp(0, GUIDE_RES - 1)
                .toVar() as unknown as NF;
              const tzN = wczA
                .sub(gfz)
                .div(GUIDE_SUB)
                .floor()
                .clamp(0, GUIDE_RES - 1)
                .toVar() as unknown as NF;
              const luN = wcxA.sub(gfx).sub(txN.mul(GUIDE_SUB)).clamp(0, GUIDE_SUB - 1) as unknown as NF;
              const lvN = wczA.sub(gfz).sub(tzN.mul(GUIDE_SUB)).clamp(0, GUIDE_SUB - 1) as unknown as NF;
              const bitN = uint(lvN.mul(GUIDE_SUB).add(luN)).toVar() as unknown as NU;
              const wm0 = m0.toVar() as unknown as NU;
              const wm1 = m1.toVar() as unknown as NU;
              If(txN.notEqual(txf).or(tzN.notEqual(tzf)), () => {
                const mvN = guideMask2.ro.element(
                  uint(tzN.mul(GUIDE_RES).add(txN)) as unknown as NU,
                );
                (wm0 as unknown as { assign(v: unknown): void }).assign(mvN.x);
                (wm1 as unknown as { assign(v: unknown): void }).assign(mvN.y);
              });
              const w = bitN.lessThan(uint(32)).select(wm0, wm1) as unknown as NU;
              return w
                .shiftRight(bitN.bitAnd(uint(31)))
                .bitAnd(uint(1))
                .equal(uint(1)) as unknown as NB;
            };
            // THE FETCH (O(1)): R = 1/(1+d) in tile widths, GBA = normal. Linear
            // filter interpolates x, z AND angle (repeat-wrapped) — his encoding.
            const smp = fetchBand(qbx, qbz, az);
            const dTile = float(1)
              .div((smp.x as unknown as NF).max(1 / 255))
              .sub(1)
              .toVar() as unknown as NF;
            const tHit = tCur.add(dTile.mul(GUIDE_PITCH).div(eLen)).toVar() as unknown as NF;
            const isMiss = dTile.greaterThan(rayBake.dMaxTile * 0.94) as unknown as NB;
            // accept hits up to a quarter-tile BEYOND the texel exit: the hard
            // border clamp dropped every blade whose hit crossed the line — a
            // blade-width seam along the whole 0.84 m grid (the user's residual
            // squares). The baked root id keeps cross-border hits consistent
            // (the fiber provably belongs to THIS tile instance).
            const tAcc = tExC.add(float(GUIDE_PITCH * 0.25).div(eLen)) as unknown as NF;
            // march position BEFORE the layer-1 advance — layer 2 refetches from here
            const tCur0 = RAY_LAYER2 ? (tCur.add(0).toVar() as unknown as NF) : null;
            If(isMiss.or(tHit.greaterThanEqual(tAcc)), () => {
              if (RAY_BANDS > 1) {
                // no fiber in THIS band — next texel, or for a descending ray the
                // nearer event: drop into the (denser) band below and refetch
                const bandBot = gP.add(hBandF.mul(HB).mul(1 / RAY_BANDS)) as unknown as NF;
                const tDropB = rd.y
                  .lessThan(-1e-4)
                  .select(
                    bandBot.sub(pos.y).div(rd.y).max(1e-3),
                    float(1e9),
                  ) as unknown as NF;
                tCur.assign(tEx.add(1e-3).min(tCur.add(tDropB).add(1e-4)));
              } else {
                // no fiber inside THIS tile instance — next texel (fresh bomb/mask)
                tCur.assign(tEx.add(1e-3));
              }
            }).Else(() => {
              const yH = ro.y.add(rd.y.mul(tHit)).toVar() as unknown as NF;
              // hit position in tile space: RAW for the copy index (a hit past
              // the border lives in the next periodic COPY — its root cells are
              // the neighbor texel's), wrapped for cell-exit + column jitter
              const qrx = qbx.add(ebx.div(eLen).mul(dTile)).toVar() as unknown as NF;
              const qrz = qbz.add(ebz.div(eLen).mul(dTile)).toVar() as unknown as NF;
              const qhx = qrx.fract().toVar() as unknown as NF;
              const qhz = qrz.fract().toVar() as unknown as NF;
              // the fiber's ROOT cell comes from the BAKED id (A channel), mapped
              // through the inverse bomb. The density law applies to ROOTS — an
              // arcing blade legally overhangs empty neighbor cells (the ring's
              // behavior). Validating the cell UNDER the hit culled every
              // overhanging blade → the user's dot-batch look.
              const idT = (smp.w as unknown as NF)
                .mul(GUIDE_SUB * GUIDE_SUB)
                .floor()
                .clamp(0, GUIDE_SUB * GUIDE_SUB - 1)
                .toVar() as unknown as NF;
              const tcx = idT
                .mod(GUIDE_SUB)
                .add(0.5)
                .div(GUIDE_SUB)
                .sub(0.5) as unknown as NF;
              const tcz = idT
                .div(GUIDE_SUB)
                .floor()
                .add(0.5)
                .div(GUIDE_SUB)
                .sub(0.5) as unknown as NF;
              // root cell center + the hit's COPY offset, both through the
              // inverse bomb (rotations keep the integer lattice exact)
              const lw = bombI(tcx, tcz) as unknown as NV2;
              const co = bombI(qrx.floor() as unknown as NF, qrz.floor() as unknown as NF) as unknown as NV2;
              const lu = lw.x
                .add(co.x)
                .add(0.5)
                .mul(GUIDE_SUB)
                .floor()
                .toVar() as unknown as NF; // UNclamped — may be the neighbor's
              const lv = lw.y
                .add(co.y)
                .add(0.5)
                .mul(GUIDE_SUB)
                .floor()
                .toVar() as unknown as NF;
              // world cell + its blade-height law: the bake is 2D (fibers are
              // infinite-height, article-style) — the runtime prunes by height
              const wcx = gfx.add(txf.mul(GUIDE_SUB)).add(lu).toVar() as unknown as NF;
              const wcz = gfz.add(tzf.mul(GUIDE_SUB)).add(lv).toVar() as unknown as NF;
              const occB = maskBitAt(wcx, wcz);
              const h2x = cellHash2(vec2(wcx, wcz) as unknown as NV2, SALT ^ 0x9191)
                .x as unknown as NF;
              // smooth stand-in for the geo ladder's blade→card height doubling
              const yK = mix(float(1.12), float(2), smoothstep(55, 85, distT)) as unknown as NF;
              const topB = h2x
                .pow(1.3)
                .mul(0.3)
                .add(0.2)
                .mul(widenT)
                .mul(yK)
                .toVar() as unknown as NF;
              // per-COLUMN ragged top (user-called: flat per-cell tops render every
              // fiber as an unbroken straight column): a ~13 mm world-stable hash
              // scales the cell top 0.55..1.05 — tapered/spiky silhouettes + blade
              // height variety inside a clump, at the price of one hash.
              const colH = cellHash(
                vec2(
                  txI.mul(64).add(qhx.mul(64).floor()),
                  tzI.mul(64).add(qhz.mul(64).floor()),
                ) as unknown as NV2,
                SALT ^ 0x7c01,
              ) as unknown as NF;
              const topEff = topB.mul(colH.mul(0.5).add(0.55)).toVar() as unknown as NF;
              const gRoot = ground
                .add(grad.x.mul(wcx.add(0.5).mul(CELL).sub(texCx)))
                .add(grad.y.mul(wcz.add(0.5).mul(CELL).sub(texCz)))
                .toVar() as unknown as NF;
              const tPar = yH.sub(gRoot).div(topEff.max(0.05)) as unknown as NF;
              If(
                occB
                  .and(tPar.lessThanEqual(1))
                  .and(yH.greaterThan(gRoot.sub(0.05)))
                  .and(tHit.lessThan(tMax)),
                () => {
                  // ACCEPT — fetch answers are ordered along the ray: first valid
                  // hit IS the nearest. Body = the world cell's slot (the resolve's
                  // shading comes from rayNrmTex, not the id decode).
                  const sxs = wcx.sub(wcx.div(GRID).floor().mul(GRID));
                  const sys = wcz.sub(wcz.div(GRID).floor().mul(GRID));
                  (tBest as unknown as { assign(v: unknown): void }).assign(tHit);
                  (bodyBest as unknown as { assign(v: unknown): void }).assign(
                    uint(sys.mul(GRID).add(sxs)).shiftLeft(uint(6)),
                  );
                  // baked normal (azimuth in G, y in B — A carries the root id) →
                  // world: inverse bomb on xz. (Shear inverse-transpose skipped —
                  // n is a shading mean.) A WORLD-CELL azimuth twist decorrelates
                  // each tile's mean-normal from its bomb rotation — the flatres
                  // stop proved the residual altitude grid was per-tile normal
                  // STATISTICS rotating with the bombing, not coverage.
                  const azn = (smp.y as unknown as NF)
                    .mul(6.2831853)
                    .add(
                      cellHash(vec2(wcx, wcz) as unknown as NV2, SALT ^ 0x6a6a)
                        .sub(0.5)
                        .mul(1.6),
                    ) as unknown as NF;
                  const nyT = (smp.z as unknown as NF).mul(2).sub(1) as unknown as NF;
                  const sxz = float(1).sub(nyT.mul(nyT)).max(0).sqrt() as unknown as NF;
                  const nW = bombI(
                    azn.cos().mul(sxz) as unknown as NF,
                    azn.sin().mul(sxz) as unknown as NF,
                  ) as unknown as NV2;
                  (nrmV as unknown as { assign(v: unknown): void }).assign(
                    vec3(nW.x, nyT, nW.y),
                  );
                  const gS = smoothGroundAt(
                    wcx.add(0.5).mul(CELL) as unknown as NF,
                    wcz.add(0.5).mul(CELL) as unknown as NF,
                  );
                  (tParV as unknown as { assign(v: unknown): void }).assign(
                    yH.sub(gS).div(topEff.max(0.05)).clamp(0, 1),
                  );
                  tCur.assign(tEnd); // done — break the outer walk
                },
              ).Else(() => {
                // baked fiber fails the WORLD tests (empty cell / above this cell's
                // blade top / under terrain). Every fiber in the HIT CELL shares its
                // mask bit and height law, so advance to the cell's EXIT in tile
                // space and refetch there — 0.03 m stepping burned a fetch per
                // ~blade-width across skim bands (measured eye +6.0 before this).
                const iux = ebx.div(eLen) as unknown as NF;
                const iuz = ebz.div(eLen) as unknown as NF;
                const gux = iux
                  .greaterThanEqual(0)
                  .select(iux.max(1e-5), iux.min(-1e-5)) as unknown as NF;
                const guz = iuz
                  .greaterThanEqual(0)
                  .select(iuz.max(1e-5), iuz.min(-1e-5)) as unknown as NF;
                const cellW = 1 / GUIDE_SUB; // tile units
                const ex2 = qhx
                  .mul(GUIDE_SUB)
                  .floor()
                  .add(iux.greaterThanEqual(0).select(float(1), float(0)))
                  .mul(cellW) as unknown as NF;
                const ez2 = qhz
                  .mul(GUIDE_SUB)
                  .floor()
                  .add(iuz.greaterThanEqual(0).select(float(1), float(0)))
                  .mul(cellW) as unknown as NF;
                const dExit = ex2
                  .sub(qhx)
                  .div(gux)
                  .min(ez2.sub(qhz).div(guz))
                  .max(0.002) as unknown as NF;
                const tCell = dExit.mul(GUIDE_PITCH).div(eLen) as unknown as NF;
                // STEEP rays (look-down): tCell divides by the tiny horizontal
                // speed and overshoots the whole sward → bare-dirt holes from
                // above (user call). The right next event for a descending ray is
                // "drop to the rejected cell's blade-top and refetch there".
                const tDrop = rd.y
                  .lessThan(-1e-4)
                  .select(
                    gRoot.add(topEff).sub(yH).div(rd.y).max(0.002),
                    float(1e9),
                  ) as unknown as NF;
                tCur.assign(tHit.add(tCell.min(tDrop)).add(1e-4));
              });
            });
            if (RAY_LAYER2 && tCur0 && l2n) {
              // GATES (measured +11.4 eye ungated — L2 fired on every non-accepting
              // step): downward rays only (the holes it exists for only READ from
              // above) AND a 3-per-pixel budget — a hole-filler, not a second march.
              If(
                tBest
                  .greaterThan(1e8)
                  .and(rd.y.lessThan(-0.3))
                  .and(l2n.lessThan(uint(3))),
                () => {
                  l2n.addAssign(uint(1));
                // LAYER 2 — the article's golden-angle overlay (tile space rotated
                // by φ·π, independent arc direction): fetched ONLY when layer 1
                // landed nothing at this step, so dense sward pays ~nothing while
                // the top-down holes gain a second, criss-crossing population.
                // The rotation isn't grid-preserving, so validation maps the hit's
                // WORLD position to its cell directly (the hit IS the fiber ±cm).
                const GC = -0.737369; // cos(φ·π)
                const GS = 0.67549; // sin(φ·π)
                // compose with the swirl: L2's frame = θ + φ·π (angle addition —
                // a fixed golden offset would be per-tile-coherent again)
                const ca2 = ca.mul(GC).sub(sa.mul(GS)).toVar() as unknown as NF;
                const sa2 = sa.mul(GC).add(ca.mul(GS)).toVar() as unknown as NF;
                // L2 arc = NEGATED L1 arc (f2.zw) — an independent direction
                // without a third baked field; 0 when ?grasstilt=0, same as L1
                const Q2x = Swx.sub(f2.z) as unknown as NF;
                const Q2z = Swz.sub(f2.w) as unknown as NF;
                const of2x = Slx.add(Q2x.mul(hgt)).mul(hgt) as unknown as NF;
                const of2z = Slz.add(Q2z.mul(hgt)).mul(hgt) as unknown as NF;
                const l2x = pos.x.sub(of2x).sub(texOx).div(GUIDE_PITCH).sub(0.5).toVar() as unknown as NF;
                const l2z = pos.z.sub(of2z).sub(texOz).div(GUIDE_PITCH).sub(0.5).toVar() as unknown as NF;
                const q2x = l2x.mul(ca2).sub(l2z.mul(sa2)).add(0.5) as unknown as NF;
                const q2z = l2x.mul(sa2).add(l2z.mul(ca2)).add(0.5) as unknown as NF;
                const t2x = Slx.add(Q2x.mul(hgt).mul(2)) as unknown as NF;
                const t2z = Slz.add(Q2z.mul(hgt).mul(2)) as unknown as NF;
                const e2x = rd.x.sub(t2x.mul(rd.y)).toVar() as unknown as NF;
                const e2z = rd.z.sub(t2z.mul(rd.y)).toVar() as unknown as NF;
                const e2L = vec2(e2x, e2z).length().max(1e-5).toVar() as unknown as NF;
                const r2x = e2x.mul(ca2).sub(e2z.mul(sa2)) as unknown as NF;
                const r2z = e2x.mul(sa2).add(e2z.mul(ca2)) as unknown as NF;
                const az2 = (atan(r2z, r2x) as unknown as NF)
                  .mul(1 / (Math.PI * 2))
                  .fract() as unknown as NF;
                const smp2 = fetchBand(q2x, q2z, az2);
                const dT2 = float(1)
                  .div((smp2.x as unknown as NF).max(1 / 255))
                  .sub(1)
                  .toVar() as unknown as NF;
                const tHit2 = tCur0.add(dT2.mul(GUIDE_PITCH).div(e2L)).toVar() as unknown as NF;
                If(
                  dT2
                    .lessThan((rayBake as { dMaxTile: number }).dMaxTile * 0.94)
                    .and(tHit2.lessThan(tAcc))
                    .and(tHit2.lessThan(tMax)),
                  () => {
                    const yH2 = ro.y.add(rd.y.mul(tHit2)).toVar() as unknown as NF;
                    const hx2 = ro.x.add(rd.x.mul(tHit2)).toVar() as unknown as NF;
                    const hz2 = ro.z.add(rd.z.mul(tHit2)).toVar() as unknown as NF;
                    // ROOT cell from the baked id, inverse-golden-rotated (a point
                    // maps exactly even though the rotation isn't grid-preserving)
                    const idT2 = (smp2.w as unknown as NF)
                      .mul(GUIDE_SUB * GUIDE_SUB)
                      .floor()
                      .clamp(0, GUIDE_SUB * GUIDE_SUB - 1)
                      .toVar() as unknown as NF;
                    const rc2x = idT2
                      .mod(GUIDE_SUB)
                      .add(0.5)
                      .div(GUIDE_SUB)
                      .sub(0.5) as unknown as NF;
                    const rc2z = idT2
                      .div(GUIDE_SUB)
                      .floor()
                      .add(0.5)
                      .div(GUIDE_SUB)
                      .sub(0.5) as unknown as NF;
                    // root + the hit's COPY offset in the L2 frame, inverse-rotated
                    // together (same off-by-one-tile hazard as L1)
                    const co2x = q2x.add(r2x.div(e2L).mul(dT2)).floor() as unknown as NF;
                    const co2z = q2z.add(r2z.div(e2L).mul(dT2)).floor() as unknown as NF;
                    const rrx = rc2x.add(co2x) as unknown as NF;
                    const rrz = rc2z.add(co2z) as unknown as NF;
                    const lu2 = rrx
                      .mul(ca2)
                      .add(rrz.mul(sa2))
                      .add(0.5)
                      .mul(GUIDE_SUB)
                      .floor()
                      .toVar() as unknown as NF; // UNclamped — may be the neighbor's
                    const lv2 = rrz
                      .mul(ca2)
                      .sub(rrx.mul(sa2))
                      .add(0.5)
                      .mul(GUIDE_SUB)
                      .floor()
                      .toVar() as unknown as NF;
                    const wc2x = gfx.add(txf.mul(GUIDE_SUB)).add(lu2).toVar() as unknown as NF;
                    const wc2z = gfz.add(tzf.mul(GUIDE_SUB)).add(lv2).toVar() as unknown as NF;
                    const occ2 = maskBitAt(wc2x, wc2z);
                    const h2b = cellHash2(vec2(wc2x, wc2z) as unknown as NV2, SALT ^ 0x9191)
                      .x as unknown as NF;
                    const yK2 = mix(float(1.12), float(2), smoothstep(55, 85, distT)) as unknown as NF;
                    const col2 = cellHash(
                      vec2(
                        hx2.mul(64 / GUIDE_PITCH).floor(),
                        hz2.mul(64 / GUIDE_PITCH).floor(),
                      ) as unknown as NV2,
                      SALT ^ 0x7c02,
                    ) as unknown as NF;
                    const top2 = h2b
                      .pow(1.3)
                      .mul(0.3)
                      .add(0.2)
                      .mul(widenT)
                      .mul(yK2)
                      .mul(col2.mul(0.5).add(0.55))
                      .toVar() as unknown as NF;
                    const gR2 = ground
                      .add(grad.x.mul(wc2x.add(0.5).mul(CELL).sub(texCx)))
                      .add(grad.y.mul(wc2z.add(0.5).mul(CELL).sub(texCz)))
                      .toVar() as unknown as NF;
                    const tP2 = yH2.sub(gR2).div(top2.max(0.05)) as unknown as NF;
                    If(
                      occ2.and(tP2.lessThanEqual(1)).and(yH2.greaterThan(gR2.sub(0.05))),
                      () => {
                        const sxs2 = wc2x.sub(wc2x.div(GRID).floor().mul(GRID));
                        const sys2 = wc2z.sub(wc2z.div(GRID).floor().mul(GRID));
                        (tBest as unknown as { assign(v: unknown): void }).assign(tHit2);
                        (bodyBest as unknown as { assign(v: unknown): void }).assign(
                          uint(sys2.mul(GRID).add(sxs2)).shiftLeft(uint(6)),
                        );
                        // normal decode (azimuth/y) + inverse golden rotation on xz
                        // (+ the same world-cell twist — see the L1 accept)
                        const az2n = (smp2.y as unknown as NF)
                          .mul(6.2831853)
                          .add(
                            cellHash(vec2(wc2x, wc2z) as unknown as NV2, SALT ^ 0x6a6a)
                              .sub(0.5)
                              .mul(1.6),
                          ) as unknown as NF;
                        const ny2 = (smp2.z as unknown as NF).mul(2).sub(1) as unknown as NF;
                        const sxz2 = float(1).sub(ny2.mul(ny2)).max(0).sqrt() as unknown as NF;
                        const nx2t = az2n.cos().mul(sxz2) as unknown as NF;
                        const nz2t = az2n.sin().mul(sxz2) as unknown as NF;
                        const n2x = nx2t.mul(ca2).add(nz2t.mul(sa2)) as unknown as NF;
                        const n2z = nz2t.mul(ca2).sub(nx2t.mul(sa2)) as unknown as NF;
                        (nrmV as unknown as { assign(v: unknown): void }).assign(
                          vec3(n2x, ny2, n2z),
                        );
                        const gS2 = smoothGroundAt(
                          wc2x.add(0.5).mul(CELL) as unknown as NF,
                          wc2z.add(0.5).mul(CELL) as unknown as NF,
                        );
                        (tParV as unknown as { assign(v: unknown): void }).assign(
                          yH2.sub(gS2).div(top2.max(0.05)).clamp(0, 1),
                        );
                        tCur.assign(tEnd);
                      },
                    );
                    // L2 reject: no own advance — layer 1's advance already moved
                    // the march; L2 simply re-tries at the next step.
                  },
                );
              });
            }
          };
          bakedTexel();
        });
      });

      If(tBest.lessThan(1e8), () => {
        const hit = ro.add(rd.mul(tBest));
        const clip = cam.vp.mul(vec4(hit, 1));
        const cz = clip.z.div(clip.w.max(NEAR_EPS));
        If(cz.greaterThanEqual(0).and(cz.lessThanEqual(1)), () => {
          // Q>1: fan the hit out to every pixel of the quad — the emit atomics
          // stay per-pixel, so nearer TRUE geometry still wins individually
          for (let dy = 0; dy < Q; dy++) {
            for (let dx = 0; dx < Q; dx++) {
              const x2 = (dx === 0 && dy === 0 ? xI : xI.add(uint(dx))) as unknown as NU;
              const y2 = (dx === 0 && dy === 0 ? yI : yI.add(uint(dy))) as unknown as NU;
              const doEmit = (): void => {
                emitPx(
                  y2.mul(uint(W)).add(x2) as unknown as NU,
                  cz as unknown as NF,
                  bodyBest,
                );
                if (rayNrmTex && nrmV && tParV) {
                  // the article's depth+normal output: normal + tip param ride a
                  // screen texture to the resolve (the 30-bit election id can't
                  // carry them). Same bottom-up row indexing — resolveRay matches.
                  textureStore(rayNrmTex, uvec2(x2, y2), vec4(nrmV, tParV)).toWriteOnly();
                }
              };
              if (dx === 0 && dy === 0) doEmit();
              else
                If(
                  x2.lessThan(uint(W)).and(y2.lessThan(uint(H))),
                  doEmit,
                );
            }
          }
        });
      });
    })().compute(Wq * Hq, [256]);
    (k as unknown as { setName(n: string): void }).setName('grassRay');
    return k;
  })();

  const runGrass = (renderer: Renderer, camera: PerspectiveCamera): void => {
    if (!onCpu) return;
    // snap the guide window to the texel grid in FINE-CELL units: mask bits stay
    // congruent with world cells (the field is world-anchored; only the window
    // moves), and the bake + march + resolve all share this frame's origin.
    const half = (GUIDE_RES / 2) * GUIDE_SUB;
    uGFx.value = Math.round(camera.position.x / GUIDE_PITCH) * GUIDE_SUB - half;
    uGFz.value = Math.round(camera.position.z / GUIDE_PITCH) * GUIDE_SUB - half;
    // separate dispatches (own submits) — keeps c.grassGuide / c.grassRay pass
    // timers clean (batched compute overlaps the render passes and smears their
    // timestamps; the whole-frame A/B is the ground truth either way)
    dispatch(renderer, kGuideBake);
    if (kGuideLight) dispatch(renderer, kGuideLight); // reads this frame's ctx
    dispatch(renderer, kRay);
  };

  return {
    batch: [],
    renderHw: runGrass,
    resolveDerive,
    resolveLean,
    resolveRay,
    attachLightBake,
    setEnabled(v: boolean): void {
      onCpu = v;
      uOn.value = v ? 1 : 0;
    },
    enabled: () => onCpu,
    readCounts,
  };
}
