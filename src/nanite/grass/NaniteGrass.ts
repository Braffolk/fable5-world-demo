/**
 * NaniteGrass — the Sannikov precomputed-raycast grass lane (G-E, default-on;
 * docs/deep-review/grass-raycast.txt + docs/perf-runs/2026-07-03-grass-arc.md).
 *
 * TRUE single-fetch runtime (2026-07-20 — the 256-step march is deleted):
 * per pixel, the elected SCENE DEPTH gives the shell point O (terrain
 * rasterizes every frame), the guide texel's ground plane reconstructs the
 * sward-top entry E, and ONE trilinear fetch of the boot-baked (x, z-in-tile,
 * angle) raycast tile (GrassRayBake.ts — R = path 1/(1+d), G/B = normal
 * azimuth/y, A = root-cell id; traced over INFINITELY TILED fibers, so a
 * grazing ray crossing many tiles is still one fetch) answers the whole ray:
 * |OB| = |OA|/cos α, B = E + rd·|OB|. Spatially-varying world density is the
 * one deviation from the article's uniform tiling — solved by DENSITY-TIERED
 * bakes (nested cell thinning) lerped by the guide density field, still one
 * (pairwise-lerped) fetch. Wind/blade arcs ride the QUADRATIC oblique basis
 * (off = Sl·h + Sq·h², linearized at the entry height); anti-tiling = the
 * continuous swirl rotation θ(pos). Hits emit a self-describing id + depth
 * into the vis-buffer election; normal + tip param ride a screen
 * StorageTexture to the resolve (grassProc.ray).
 *
 * A per-frame GUIDE FIELD (camera-centered ctx storage buffer + bilinear
 * field textures) carries the world in: ground + gradient, density
 * (mask popcount), swirl/wind/arc coefficients, gust amplitude.
 *
 * ⚠️ every ray-space FIELD must be continuous across space — per-TILE-
 * constant anything (values OR sampling granularity) reads as a 0.84 m grid
 * from altitude (five-round user-debugged lesson, ledger §G-E GRID
 * POST-MORTEM). This includes GEOMETRY fields: sward height and tip-param
 * ground are analytic/bilinear, never per-texel constants.
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
  Fn,
  If,
  atan,
  atomicMax,
  atomicStore,
  float,
  instanceIndex,
  mix,
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
import type { NB, NF, NU, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import { canopyAt, cellHash, cellHash2 } from '../../gpu/passes/Scatter';
import { gustAt, windContext, windExposure, windU } from '../../render/Wind';
import type { TerrainField } from '../world/TerrainField';
import type { NaniteCam } from '../NaniteCommon';
import type { NaniteVisBuffers } from '../raster/NaniteRaster';
import { bakeGrassRayTile } from '../build/GrassRayBake';
import {
  aLoadU,
  bcF2U,
  bcU2F,
  dispatch,
  loopUN,
  packHalfU,
  returnIf,
  sU32Views,
  sUvec4RO,
  toF,
  uniformF,
  unpackHalfU,
} from '../Tsl';

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
// algorithm (docs/deep-review/grass-raycast.txt) as a TRUE per-pixel single
// fetch — boot-baked (x, z, angle) raycast tiles answer the whole ray at the
// terrain-anchored sward entry (O(1), no loop; 2026-07-20 rebuild — the
// 256-step guide march this replaced is in git history); wind = his
// march-space TBN shear; anti-tiling = a continuous swirl rotation. The old
// lanes (geo emission+HW queue, hybrid near-raster seam, analytic all-bands
// march, SW scanline) were DELETED 2026-07-04 (user call). ?grass=0 disables
// grass entirely (NaniteFrame gate).
/** article-lane knobs (all bake/runtime params of the G-E algorithm):
 *  ?grassbakres  — tile texels per edge (article 64; "1 texel ≈ 1 screen px")
 *  ?grassbakang  — angle slices (article uses 8, up to 64)
 *  ?grassshiftk  — bake fiber-shift heuristic, cells/cell (inclined-blade approx)
 *  ?grassthickk  — bake thicken-with-distance heuristic, fraction/cell (taper approx) */
const qNum = (k: string, d: number, lo: number, hi: number): number => {
  const v = Number(new URLSearchParams(window.location.search).get(k) ?? String(d));
  return Number.isFinite(v) && v >= lo && v <= hi ? v : d;
};
const BAKE_RES = Math.round(qNum('grassbakres', 64, 16, 256));
const BAKE_ANG = Math.round(qNum('grassbakang', 8, 4, 64));
const BAKE_SHIFTK = qNum('grassshiftk', 0.22, 0, 1);
const BAKE_THICKK = qNum('grassthickk', 0.18, 0, 2);
/** sway amplitude scale (?grasssway=K, 0 = steady gust-bend only) */
const RAY_SWAY = qNum('grasssway', 1, 0, 5);
/** ?grassrayend=N — band-end knob for cost attribution (default 155) */
const RAY_END = ((): number => {
  const v = Number(new URLSearchParams(window.location.search).get('grassrayend') ?? '155');
  return Number.isFinite(v) && v >= 20 && v <= 300 ? v : 155;
})();
const RAY_SHELL_H = 1.5; // max blade reach above ground (incl. mid-card 2× + wind)
/** DENSITY TIERS (cell-keep fractions, descending): one baked LUT volume each.
 *  The world thins grass per cell; the single-fetch runtime picks the two tiers
 *  bracketing the local guide density and lerps — sparse regions render the
 *  correct PRESENT blades in one fetch instead of the march's reject speckle.
 *  Below the last tier the fetch fades to miss (bare ground). */
const TIER_FRACS = [1, 0.55, 0.3, 0.12];
// ---- GUIDE FIELD (ray lane) — the world context the single fetch anchors on ---------
// A camera-centered world-space context field REBAKED EVERY FRAME by a tiny compute
// pass (O(area), blade-count-independent): kRay FETCHES its world instead of deriving
// it in-register. Per 0.84 m texel (= 8×8 fine cells):
//   ctx  (uvec4): ground f32 | ground gradient half2 | (swardTop, gustAmp) half2
//   mask (uvec2): 64-bit fine-cell occupancy — the DENSITY LAW baked to bits
//     (bit = cellHash(cell) < dens·thin·edge); kRay consumes only its POPCOUNT
//     (the guideFieldT3 density that drives tier select + the bare gate)
// Blades root on PLANE-RECONSTRUCTED ground (center + gradient·Δ) — exact on slopes,
// which fixes the user-reported sunken blades (burst-level ground was up to ~0.5 m
// low). Wind stays fully animated: amp is rebaked per frame. Storage buffers, not
// textures (uint StorageTexture is mistyped 'float' by the node builder; float-texture
// roundtrips could canonicalize mask NaN bit patterns).
const GUIDE_SUB = 8; // fine cells per texel edge
const GUIDE_PITCH = CELL * GUIDE_SUB; // 0.84 m
// ≥ ray reach (default stays 384)
const GUIDE_RES = Math.max(384, Math.ceil(((RAY_END + 4) * 2) / GUIDE_PITCH));
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
  /** the TerrainField planes — THE grass terrain source (guide-bake height
   *  4-tap → height plane L0, density fields/biome/slope → plane taps +
   *  height-plane CD, water gate → waterY plane). */
  field: TerrainField;
  canopyTex: StorageTexture | null;
}


export interface GrassField {
  /** compute kernels for world1's batched submit (after kVisClear, before kHwArgs) */
  batch: readonly unknown[];
  /** the grass HW blade pass (own render target, HARDWARE EARLY-Z): a fullscreen
   *  prime writes the election depth into a real depth buffer, then the blade
   *  draw runs depth-tested — occluded blade fragments never invoke the election
   *  shader. Call right after the raster's hwRender (world1 only). */
  renderHw(renderer: Renderer, camera: PerspectiveCamera): void;
  /** resolve-side shading tap (call INSIDE the resolve fragment Fn): kRay writes
   *  the hit normal + tip param per pixel into a screen StorageTexture — the
   *  algorithm's own output is depth+normal. vec4(nrm, t). */
  resolveRay(px: NU): NV4;
  setEnabled(v: boolean): void;
  enabled(): boolean;
  readCounts(renderer: Renderer): Promise<{ clumps: number; hwTris: number }>;
}

export function buildGrassField(opts: GrassBuildOpts): GrassField {
  const { cam, vis, field } = opts;
  const canopyTex = opts.canopyTex;
  const uOn = uniformF(1);
  let onCpu = true;
  // S6c: absolute world coords reach ~311 km only on the streamed (Estonia) path;
  // the generated world stays < 3 km where f32 is exact. Build-time flag so the
  // guide-relative march below compiles ONLY for streamed — generated keeps the
  // verbatim absolute expressions (byte-identical shader ⇒ A/A gate holds).
  const streamed = new URLSearchParams(window.location.search).get('src') === 'estonia';

  // Format-1's finest window contains the whole guide ring, so retain its hoisted
  // level-0 path. Format-2 LOD -2 is only a 96 m window while the guide spans
  // ±161 m; streamed grass must fall through -2 -> -1 -> 0 exactly like terrain.
  const heightAt = (p: NV2): NF =>
    streamed ? field.fieldHeightFinestHot(p) : field.fieldHeight(p, 0);

  /** ground height a blade roots on = the real cooked heightfield (runtime
   *  terrain displacement is gone — cook-side-synthesis law) */
  const groundAt = (p: NV2): NF => heightAt(p);

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

  /** shared field-density (fine + far kernels; far skips scruff) — the
   *  GroundRing law over TerrainField planes: biome plane [classId byte,
   *  vegDensity], fields plane [moisture, flow, snow, rockExposure], slope =
   *  height-plane CD (the retired normalTex.w stencil), water gate = waterY
   *  plane (−1e4 sentinel when the source has no water ⇒ bank/gate pass).
   *  riverDepth is NOT stored (spec §3): derived waterY − h, max 0 — the
   *  legacy filtered sim-res riverDepth spilled ~a texel onto banks, the
   *  derived point value doesn't; the bank smoothstep + hard gate carry the
   *  visible suppression (S3b parity gate was the witness). Fields/biome stay
   *  on the finest-containing CHAIN (the generated L0 u8 window is
   *  world-centered pre-S5 — a far camera's ring falls through to L1). */
  const densityAt = (wpos: NV2, h: NF, dist: NF, scruff: boolean): NF => {
    const bio = field.biomeAt(wpos);
    const fld = field.fieldsAt(wpos);
    const bioId = bio.x.mul(255).add(0.5).floor() as unknown as NF;
    const vegDensity = bio.y as unknown as NF;
    const rockExposure = fld.w as unknown as NF;
    const snow = fld.z as unknown as NF;
    const moisture = fld.x as unknown as NF;
    const slope = streamed ? field.fieldSlopeHot(wpos) : field.fieldSlope(wpos, 0);
    const waterY = field.fieldWaterYNearest(wpos);
    const above = h.sub(waterY) as unknown as NF;
    const riverDepth = waterY.sub(h).max(0) as unknown as NF;
    const bank = smoothstep(0.06, 0.5, above).mul(
      float(1).sub(smoothstep(0.2, 1.1, riverDepth).mul(0.78)),
    ) as unknown as NF;
    const canopy = canopyTex ? canopyAt(canopyTex, wpos) : (float(0) as unknown as NF);
    let dens = byBio(bioId, [0.18, 0.7, 0.62, 0.7, 1.5, 1.1])
      .mul(bank)
      .mul(vegDensity.mul(0.85).add(0.15))
      .mul(float(1).sub(rockExposure.mul(0.55)))
      .mul(float(1).sub(canopy.mul(0.45)))
      .mul(moisture.mul(0.35).add(0.75)) as unknown as NF;
    if (scruff) {
      dens = dens.max(
        float(0.3).mul(float(1).sub(smoothstep(8, 14, dist))).mul(bank),
      ) as unknown as NF;
    }
    dens = dens
      .mul(float(1).sub(snow.mul(0.95)))
      .mul(float(1).sub(smoothstep(0.55, 0.95, slope))) as unknown as NF;
    // hard water gate (ring: return when above < 0.04); the −1e4 dry/no-water
    // sentinel makes `above` huge ⇒ gate self-passes.
    return dens.mul(above.greaterThanEqual(0.04).select(float(1), float(0))) as unknown as NF;
  };

  /** the ray lane emits per-pixel — no counters exist */
  const readCounts = async (): Promise<{ clumps: number; hwTris: number }> => ({
    clumps: 0,
    hwTris: 0,
  });

  // ================= THE RAY LANE (the only lane) ==================================
  // Per-pixel raycast: guide-texel walk + ONE baked-tile fetch per in-sward step.
  // A hit emits a self-describing id + depth into the election, so the resolve/
  // shadows/GTAO/TRAA pipeline is untouched; normal+tip ride rayNrmTex.

  // ---- guide buffers + per-frame bake (ray lane only) ------------------------------
  // MERGED RECORD (P3(b)): one 8-word / 32-byte line-aligned record per guide texel.
  // Word layout per texel i (base = i*8):
  //   [0] ground     f32 bitcast   (bcF2U)      ← kRay: O/E planes + gB corners
  //   [1] grad       half2         (dgdx, dgdz) ← kRay: O/E planes
  //   [2] (topOut,amp) half2       packHalfU
  //   [3] widenT     f32 bitcast   (bcF2U)
  //   [4] mask0      u32 bits 0..31
  //   [5] mask1      u32 bits 32..63
  //   [6..7] spare   (written 0)   — pads to 32B so records stay line-aligned
  // ⚠️ single-fetch rebuild (2026-07-20): kRay now reads ONLY words 0..1; the
  // mask survives as kGuideBake's own popcount → guideFieldT3 density, and
  // words 2..5 have NO consumer — shrinking the record is task #39's guide-
  // bake diet (kept here to hold this change to the kernel swap).
  const REC_WORDS_PER = 8;
  const recWords = GUIDE_N * REC_WORDS_PER;
  const guideRecAttr = new StorageBufferAttribute(new Uint32Array(recWords), 1);
  guideRecAttr.name = 'grassGuideRec';
  // raw u32 view — bake writes go here (one record's 8 words per texel)
  const guideRecW = sU32Views(guideRecAttr, recWords);
  // uvec4 view over the record halves: element(2*i) = ctx words 0..3
  const guideRec4 = sUvec4RO(guideRecAttr, GUIDE_N * 2);
  const recElem = (i: NU | number): NU =>
    (typeof i === 'number' ? uint(i) : i).mul(uint(2)) as unknown as NU;
  const guideCtx4 = { element: (i: NU | number) => guideRec4.element(recElem(i)) };
  /** guide origin = fine-cell index of texel (0,0)'s first cell, snapped to the
   *  8-cell texel grid — mask bits stay congruent with WORLD cells (the field is
   *  world-anchored, only the window moves). Integer-valued floats, exact ≤ 2^23. */
  const uGFx = uniformF(0);
  const uGFz = uniformF(0);
  // S6c PRECISION (streamed only): the guide-origin-RELATIVE ray origin in world
  // metres (camPos − guideOriginWorld), computed CPU-side each frame. On Estonia
  // camPos ≈ 311 km ⇒ the march's `pos.x/CELL − gfx` texel-index and the DDA
  // `boundary − ro.x` are 311 km−311 km f32 cancellations (ULP ≈ 3 cm) → the ray
  // picks the wrong guide texel with a view-dependent staircase = the terraced
  // grass bands. Re-expressed against this small relative origin the whole march
  // stays sub-metre ⇒ exact. GENERATED world keeps the verbatim absolute path
  // (streamed=false below), so its shader is byte-identical (A/A gate). uGFx is
  // an exact integer so guideOriginWorld = uGFx·CELL is exact; the CPU subtract
  // is f64 ⇒ uRoM is the precise small offset.
  const uRoMx = uniformF(0);
  const uRoMz = uniformF(0);
  // WORLD-ANCHORED tile origin (streamed only): the exact-integer guide origin in
  // TILE units, reduced mod 4096 tiles on the CPU (f64) so the fp32 shader stays
  // precise. The anti-tiling hex lattice must be locked to the WORLD, not the
  // camera — building it on the guide-origin-relative phE (which slides as the
  // camera moves) made the whole voronoi pattern reset on every position change.
  // worldTile = phE/PITCH + uOT ⇒ camera-invariant (mod a 3.4 km reshuffle line).
  const uOTx = uniformF(0);
  const uOTz = uniformF(0);

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
  /** T3.x = per-texel grass DENSITY (mask popcount / 64) — the tier-select
   *  field. Rides its own HW-bilinear texture: continuous across space (a
   *  buffer texel read is a staircase; encoding it as swirl-vector length
   *  shrinks under angle mixing). yzw spare. ~1.2 MB. */
  const guideFieldT3 = mkFieldTex('grassGuideField3');

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
      // S6e: cam.camPos is ANCHOR-relative since S6d, while wpos is ABSOLUTE —
      // subtracting them on streamed put the camera ~311 km away from every texel
      // (density/thin gates → 0 ⇒ empty guide ⇒ the ray lane painted NO grass).
      // Streamed distances are therefore formed in the GUIDE-ORIGIN frame: texel
      // offset (exact small math) vs uRoM (the CPU f64 camera − guideOrigin,
      // already computed for the march). Generated keeps the verbatim absolute
      // expression — its shader stays byte-identical (A/A gate).
      const wposG = vec2(tx, tz)
        .mul(GUIDE_SUB)
        .add(GUIDE_SUB / 2)
        .mul(CELL)
        .toVar() as unknown as NV2;
      const camG = vec2(uRoMx as unknown as NF, uRoMz as unknown as NF) as unknown as NV2;
      const dist = (streamed ? wposG.sub(camG) : wpos.sub(vec2(cam.camPos.x, cam.camPos.z)))
        .length()
        .toVar() as unknown as NF;
      // ground (the real cooked heightfield) and its gradient (central diff at
      // ±half pitch). Blades plane-reconstruct off these: ≤ cm error at 0.84 m pitch
      // over the ~1 m-bilinear heightfield — this is what fixes the lite-path
      // sunken blades on slopes.
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
        // S6e: guide-origin-frame distance on streamed (see dist above); cw stays
        // ABSOLUTE — the density/height field samplers below take world coords.
        const dC = (
          streamed
            ? wposG.add(vec2(dx * (GUIDE_PITCH / 2), dz * (GUIDE_PITCH / 2))).sub(camG)
            : cw.sub(vec2(cam.camPos.x, cam.camPos.z))
        ).length() as unknown as NF;
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
      /** occupied-cell count 0..64 — /64 = the texel's DENSITY (kRay tier select) */
      const cnt = uint(0).toVar() as unknown as NU;
      If(pMax.greaterThan(0), () => {
        loopUN('gbv', uint(0), uint(GUIDE_SUB), (v) => {
          loopUN('gbu', uint(0), uint(GUIDE_SUB), (u) => {
            const wcF = fb.add(vec2(toF(u), toF(v))) as unknown as NV2;
            const fu = toF(u).add(0.5).mul(1 / GUIDE_SUB) as unknown as NF;
            const fv = toF(v).add(0.5).mul(1 / GUIDE_SUB) as unknown as NF;
            const pC = mix(mix(p00, p10, fu), mix(p01, p11, fu), fv) as unknown as NF;
            If(cellHash(wcF, SALT ^ 0x77a1).lessThan(pC), () => {
              (cnt as unknown as { addAssign(v: unknown): void }).addAssign(uint(1));
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
      // (word 2 is consumer-less since the single-fetch rebuild — #39 diet)
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
      // merged 8-word record (see layout at decl): ctx words 0..3, mask words 4..5,
      // spare words 6..7 (write 0). Same values/pack fns as the pre-merge split.
      const base = i.mul(uint(REC_WORDS_PER));
      guideRecW.rw.element(base).assign(bcF2U(g));
      guideRecW.rw.element(base.add(uint(1))).assign(packHalfU(vec2(dgdx, dgdz) as unknown as NV2));
      guideRecW.rw.element(base.add(uint(2))).assign(packHalfU(vec2(topOut, amp) as unknown as NV2));
      // word 3: widenT — consumer-less since the single-fetch rebuild (kRay
      // derives widen analytically); dies with the #39 record diet
      guideRecW.rw.element(base.add(uint(3))).assign(bcF2U(widenT));
      guideRecW.rw.element(base.add(uint(4))).assign(m0);
      guideRecW.rw.element(base.add(uint(5))).assign(m1);
      // spare tail — keep the 32B record fully defined (line-aligned padding)
      guideRecW.rw.element(base.add(uint(6))).assign(uint(0));
      guideRecW.rw.element(base.add(uint(7))).assign(uint(0));
      // ---- per-texel FIELD bake (guideFieldT1/T2 — see decl): the kRay per-step
      // smNoise+trig monster, folded to texel rate. Same fields, same salts.
      const smN = (salt: number): NV2 => {
        // 4-texel period (was 2.5): the single fetch carries content up to
        // ~1.5 tiles through THIS rotation field — the steeper gradient read
        // as curved "kelp" smears at near range; slower swirl keeps the
        // anti-tiling while halving the warp per fetched blade
        const qx = wpos.x.mul(1 / (GUIDE_PITCH * 4)) as unknown as NF;
        const qz = wpos.y.mul(1 / (GUIDE_PITCH * 4)) as unknown as NF;
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
      // texture bombing: per-cell random rotation breaks the raw tiling
      const thB = (smN(0x0b0b).x as unknown as NF).mul(6.2831853).toVar() as unknown as NF;
      const caB = thB.cos() as unknown as NF;
      const saB = thB.sin() as unknown as NF;
      let SwxB: NF = float(0) as unknown as NF;
      let SwzB: NF = float(0) as unknown as NF;
      // march-space wind shear — only where there is wind to shear
      if (windContext()) {
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
      // static per-tile swirl lean: staticArc(0x3333) + the ~120°-offset lean, riding the
      // SAME oblique basis as the wind — whole 0.84 m patches lean in hash-varied directions,
      // breaking the straight-vertical-prism look.
      const nT = smN(0x3333);
      const baT = (nT.x as unknown as NF).mul(6.2831853).toVar() as unknown as NF;
      const bmT = (nT.y as unknown as NF).mul(0.25).add(0.12) as unknown as NF;
      const a1xB = baT.cos().mul(bmT) as unknown as NF;
      const a1zB = baT.sin().mul(bmT) as unknown as NF;
      const laT = (nT.x as unknown as NF).mul(6.2831853).add(2.1) as unknown as NF;
      const lmT = (nT.y as unknown as NF).mul(0.05).add(0.02) as unknown as NF;
      const SlxB = laT.cos().mul(lmT) as unknown as NF;
      const SlzB = laT.sin().mul(lmT) as unknown as NF;
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
      textureStore(
        guideFieldT3,
        uvec2(txu, tzu),
        vec4(toF(cnt).mul(1 / 64), 0, 0, 0),
      ).toWriteOnly();
    })().compute(GUIDE_N, [256]);
    (k as unknown as { setName(n: string): void }).setName('grassGuide');
    return k;
  })();

  // ---- G-E: THE ARTICLE'S PRECOMPUTATION (⚠️ USER DIRECTIVE 2026-07-04) ---------------
  // Boot-baked raycast tile (GrassRayBake.ts): 3D texture (x, z in tile, angle) →
  // R = path length 1/(1+d), GBA = normal — traced over infinitely-tiled FULL-density
  // clump geometry with the article's shift+thicken heuristics. LINEAR filter +
  // REPEAT wrap on all three axes (his interpolation, incl. across angle slices).
  // The tile = one guide texel footprint (0.84 m, 8×8 fine cells).
  const rayBake = ((): { texs: Data3DTexture[]; dMaxTile: number; meanR: number } => {
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
      tiers: TIER_FRACS,
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
    return { texs, dMaxTile: b.dMaxTile, meanR: b.meanR };
  })();
  // the article's OUTPUT is depth+normal — the hit normal/tip can't ride the 30-bit
  // election id, so kRay writes them per pixel into a screen StorageTexture (a
  // texture, not a buffer: the resolve fragment is at the 10-storage-buffer ceiling).
  const rayNrmTex = ((): StorageTexture => {
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
  const resolveRay = (px: NU): NV4 => {
    const uv = vec2(
      toF(px.mod(uint(cam.width))).add(0.5).div(cam.width),
      toF(px.div(uint(cam.width))).add(0.5).div(cam.height),
    ) as unknown as NV2;
    return texture(rayNrmTex, uv, 0) as unknown as NV4;
  };

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
      const tBest = float(1e9).toVar() as unknown as NF;
      const bodyBest = uint(0).toVar() as unknown as NU;
      // G-E article lane: the accepted hit's baked normal + tip param, stored to
      // the screen StorageTexture next to the election emit (the article's output
      // is depth+normal — see rayNrmTex above)
      const nrmV = vec3(0, 1, 0).toVar() as unknown as NV3;
      const tParV = float(0.5).toVar() as unknown as NF;
      // ---- O: the scene hit (the article's shell fragment). The terrain raster
      // already elected a depth for every ground pixel — no election means sky,
      // which has no ground to root grass on.
      returnIf(tMax.greaterThan(1e8) as unknown as NB);
      const tScene = tMax.sub(0.3).max(0.05).toVar() as unknown as NF;
      if (GRASS_DBG === 'raysetup') {
        // attribution stop: ray gen + scene-depth reconstruct only
        If(tScene.lessThan(-1), () => {
          emitPx(px as unknown as NU, tScene as unknown as NF, bodyBest);
        });
        returnIf(tBest.greaterThan(0) as unknown as NB);
      }


      // ---- G-E SINGLE FETCH (the article's runtime, terrain-anchored) ----------------
      // The 256-step march is DELETED (2026-07-20): it re-walked per pixel what the
      // precompute already answered. The article's runtime is exactly: at the shell
      // fragment O, transform the view ray into the tile's oblique frame, ONE LUT
      // fetch → in-tile path |OA| → 3D hit |OB| = |OA|/cos α → reconstruct B, emit
      // B's depth+normal. Our shell fragment is the ELECTED SCENE HIT (terrain
      // rasterizes every frame); the sward-top entry E is reconstructed from O via
      // the guide texel's ground plane. The infinite-tiling precompute IS the
      // multi-tile first-hit, so a grazing ray crossing many tiles is still one
      // fetch. Per-pixel cost: 2 guide ctx loads + 4 ground-corner loads +
      // 3 field taps + 2 tier LUT taps — density- and distance-independent.
      // ⚠️ flat-local approximation: E extrapolates O's texel plane backward along
      // the ray (grazing rays: metres of extrapolation) — exact on flat meadows,
      // soft on strongly undulating ground. The march's per-cell height/occupancy
      // rejects are gone (that's what made it a march); density is the tier axis.
      const gfx = uGFx as unknown as NF;
      const gfz = uGFz as unknown as NF;
      // band cap: same horizontal reach law as the march's tEnd (guide window ±161 m)
      returnIf(tScene.mul(dirL).greaterThan(RAY_END) as unknown as NB);
      /** ph-frame (S6c): streamed = guide-origin-relative metres (exact), else
       *  absolute world metres — every helper below takes coordinates in it */
      const phAt = (t: NF): { x: NF; z: NF } =>
        streamed
          ? {
              x: (uRoMx as unknown as NF).add(rd.x.mul(t)) as unknown as NF,
              z: (uRoMz as unknown as NF).add(rd.z.mul(t)) as unknown as NF,
            }
          : { x: ro.x.add(rd.x.mul(t)) as unknown as NF, z: ro.z.add(rd.z.mul(t)) as unknown as NF };
      /** ph-frame metres → fine-cell offset from the guide origin */
      const relCells = (ph: NF, gf: NF): NF =>
        (streamed ? ph.div(CELL) : ph.div(CELL).sub(gf)) as unknown as NF;
      /** texel origin (ph-frame metres) of texel index tf on one axis */
      const texOrg = (tf: NF, gf: NF): NF =>
        (streamed
          ? tf.mul(GUIDE_SUB).mul(CELL)
          : gf.add(tf.mul(GUIDE_SUB)).mul(CELL)) as unknown as NF;

      // ---- O's texel plane → sward height + the entry point E ----------------------
      const phO = phAt(tScene);
      const phOx = phO.x.toVar() as unknown as NF;
      const phOz = phO.z.toVar() as unknown as NF;
      const rcOx = relCells(phOx, gfx).toVar() as unknown as NF;
      const rcOz = relCells(phOz, gfz).toVar() as unknown as NF;
      const txfO = rcOx.div(GUIDE_SUB).floor().clamp(0, GUIDE_RES - 1).toVar() as unknown as NF;
      const tzfO = rcOz.div(GUIDE_SUB).floor().clamp(0, GUIDE_RES - 1).toVar() as unknown as NF;
      const cvO = guideCtx4.element(uint(tzfO.mul(GUIDE_RES).add(txfO)) as unknown as NU);
      const groundO = bcU2F(cvO.x).toVar() as unknown as NF;
      const gradO = (unpackHalfU(cvO.y) as unknown as { toVar(): NV2 }).toVar() as unknown as NV2;
      // steep-slope gate (~50°): the flat-local frame is meaningless on cliff
      // faces (Taevaskoja terraces painted grass curtains); the cook's density
      // law zeroes these anyway — this kills the bilinear bleed band too
      returnIf(
        gradO.x.mul(gradO.x).add(gradO.y.mul(gradO.y)).greaterThan(1.0) as unknown as NB,
      );
      // ANALYTIC sward height — continuous across space (the baked per-texel top
      // is a 0.84 m STAIRCASE; any geometry keyed to it re-grows the grid quilt).
      // Same laws as the guide bake: distance widen-conservation + far height-up,
      // × a world-anchored 0.42 m value noise for ragged (non-mowed) sward tops.
      // Noise lattice = 4 fine cells: gf is texel-snapped (8-cell multiple) so
      // gf/4 is an exact integer — hash coords absolute, fractions from the small
      // relative frame (S6c: no 311 km cancellation).
      const distO = tScene.mul(dirL).toVar() as unknown as NF;
      const widenO = float(1)
        .div(grassThin(distO).sqrt())
        .clamp(1, 4)
        .sub(1)
        .mul(0.3)
        .add(1) as unknown as NF;
      const rag = ((): NF => {
        const qx = rcOx.mul(0.25).toVar() as unknown as NF;
        const qz = rcOz.mul(0.25).toVar() as unknown as NF;
        const ix = qx.floor().toVar() as unknown as NF;
        const iz = qz.floor().toVar() as unknown as NF;
        const fx = smoothstep(0, 1, qx.sub(ix)) as unknown as NF;
        const fz = smoothstep(0, 1, qz.sub(iz)) as unknown as NF;
        const h = (dx: number, dz: number): NF =>
          cellHash(
            vec2(
              ix.add(dx).add(gfx.mul(0.25)),
              iz.add(dz).add(gfz.mul(0.25)),
            ) as unknown as NV2,
            SALT ^ 0x5a17,
          ) as unknown as NF;
        return mix(mix(h(0, 0), h(1, 0), fx), mix(h(0, 1), h(1, 1), fx), fz) as unknown as NF;
      })();
      const swardH = mix(float(0.5), float(0.85), smoothstep(50, 90, distO))
        .mul(widenO)
        .mul(rag.mul(0.24).add(0.88))
        .clamp(0.12, 1.1)
        .toVar() as unknown as NF;
      const texCOx = texOrg(txfO, gfx).add(GUIDE_PITCH / 2) as unknown as NF;
      const texCOz = texOrg(tzfO, gfz).add(GUIDE_PITCH / 2) as unknown as NF;
      const pOy = ro.y.add(rd.y.mul(tScene)) as unknown as NF;
      // height of O above its texel's ground plane (≈0 on terrain; >0 on a trunk/
      // rock — grass in front of it still renders, hits behind it lose the election)
      const hO = pOy
        .sub(groundO.add(gradO.x.mul(phOx.sub(texCOx))).add(gradO.y.mul(phOz.sub(texCOz))))
        .toVar() as unknown as NF;
      // descent rate of ray height ABOVE the ground plane (oblique frame vertical);
      // a terrain-hitting ray always approaches from above ⇒ clamp keeps it sane
      const dhdt = rd.y
        .sub(gradO.x.mul(rd.x))
        .sub(gradO.y.mul(rd.z))
        .min(-1e-3)
        .toVar() as unknown as NF;
      // E = where the ray crosses the sward top (h = swardH), walked back from O —
      // CAPPED to ±3 tiles of horizontal travel: a grazing ray's uncapped
      // walk-back extrapolates O's texel plane tens of metres (floating bright
      // slabs at the far band); capping keeps the fetch anchored to LOCAL guide
      // data (a capped entry starts inside the sward — blades above the segment
      // are clipped, sub-pixel at the ranges where the cap binds). Camera inside
      // the sward ⇒ clamp to the march's old near start.
      const tBackMax = float(GUIDE_PITCH * 3).div(dirL) as unknown as NF;
      const tE = tScene
        .add(swardH.sub(hO).div(dhdt).clamp(tBackMax.negate(), tBackMax))
        .max(0.05)
        .toVar() as unknown as NF;

      // ---- E's texel: local ground plane + per-frame fields (bomb/wind/arc) ---------
      const phE = phAt(tE);
      const phEx = phE.x.toVar() as unknown as NF;
      const phEz = phE.z.toVar() as unknown as NF;
      const rcEx = relCells(phEx, gfx).toVar() as unknown as NF;
      const rcEz = relCells(phEz, gfz).toVar() as unknown as NF;
      // bilinear ground under E (the same 4-corner scheme as gB): the old faceted
      // per-texel plane JUMPED at every 0.84 m guide-cell edge, so hgt jumped, the
      // sampled tile shifted a hair, and a dark seam appeared on the SQUARE guide
      // grid (visible at oblique angles where hgt is large). Bilinear ⇒ continuous.
      const gPE = ((): NF => {
        const qx = rcEx.div(GUIDE_SUB).sub(0.5) as unknown as NF;
        const qz = rcEz.div(GUIDE_SUB).sub(0.5) as unknown as NF;
        const ix = qx.floor().clamp(0, GUIDE_RES - 2).toVar() as unknown as NF;
        const iz = qz.floor().clamp(0, GUIDE_RES - 2).toVar() as unknown as NF;
        const fxg = qx.sub(ix).clamp(0, 1) as unknown as NF;
        const fzg = qz.sub(iz).clamp(0, 1) as unknown as NF;
        const gAt = (dx: number, dz: number): NF =>
          bcU2F(
            guideCtx4.element(uint(iz.add(dz).mul(GUIDE_RES).add(ix.add(dx))) as unknown as NU)
              .x as unknown as NU,
          ) as unknown as NF;
        return mix(
          mix(gAt(0, 0), gAt(1, 0), fxg),
          mix(gAt(0, 1), gAt(1, 1), fxg),
          fzg,
        ) as unknown as NF;
      })();
      // shear height at the entry (the article evaluates the oblique basis at the
      // fragment); clamped like the march — past ~0.6 m the arc wraps tile space
      const hgt = ro.y.add(rd.y.mul(tE)).sub(gPE).max(0).min(0.6).toVar() as unknown as NF;
      // per-frame fields, HW-bilinear at E's continuous position (grid lesson:
      // piecewise-bilinear across space, never per-texel-constant)
      const guv = vec2(
        rcEx.div(GUIDE_SUB * GUIDE_RES),
        rcEz.div(GUIDE_SUB * GUIDE_RES),
      ).clamp(0, 1) as unknown as NV2;
      const f1 = (texture(guideFieldT1, guv, 0) as unknown as { toVar(): NV4 }).toVar();
      const f2 = (texture(guideFieldT2, guv, 0) as unknown as { toVar(): NV4 }).toVar();
      // local grass DENSITY (guide mask popcount, HW-bilinear ⇒ continuous):
      // the tier-select field + the bare-ground gate
      const dens = (texture(guideFieldT3, guv, 0).x as unknown as NF)
        .clamp(0, 1)
        .toVar() as unknown as NF;
      returnIf(dens.lessThan(0.02) as unknown as NB);
      // the wind — the article's march-space shear: oblique TRUE-derivative basis
      // (thetenthplanet.de/archives/1180, deliberately NON-orthonormal), linear
      // lean + quadratic arc re-linearized at the entry height
      const Slx = (f1.x as unknown as NF).toVar() as unknown as NF;
      const Slz = (f1.y as unknown as NF).toVar() as unknown as NF;
      const Sqx = (f1.z as unknown as NF).add(f2.z).toVar() as unknown as NF;
      const Sqz = (f1.w as unknown as NF).add(f2.w).toVar() as unknown as NF;
      const offX = Slx.add(Sqx.mul(hgt)).mul(hgt) as unknown as NF; // Sl·h + Sq·h²
      const offZ = Slz.add(Sqz.mul(hgt)).mul(hgt) as unknown as NF;
      // WORLD-ANCHORED tile coordinate (uOT folds the exact-integer guide origin
      // back in, mod 4096 tiles): worldTile = phE/PITCH + uOT ⇒ camera-invariant,
      // so the anti-tiling hex lattice locks to the WORLD instead of resetting on
      // every camera-position move. (The old per-cell (phE − texOE) bomb rotated
      // about each 0.84 m cell centre — that fixed point emptied top-down centres
      // and the phase jump seamed adjacent cells.)
      const wtx = (
        streamed ? phEx.div(GUIDE_PITCH).add(uOTx as unknown as NF) : phEx.div(GUIDE_PITCH)
      ).toVar() as unknown as NF;
      const wtz = (
        streamed ? phEz.div(GUIDE_PITCH).add(uOTz as unknown as NF) : phEz.div(GUIDE_PITCH)
      ).toVar() as unknown as NF;
      // wind shear (metres → tiles) displaces the SAMPLING coord; the hex LATTICE
      // stays on the unsheared world coord so the voronoi doesn't wobble with wind
      const Ptx = wtx.sub(offX.div(GUIDE_PITCH)).toVar() as unknown as NF;
      const Ptz = wtz.sub(offZ.div(GUIDE_PITCH)).toVar() as unknown as NF;
      // sheared view-ray horizontal → the fetch's angle axis; eLen (rotation-
      // invariant) is the in-tile projection scale ⇒ dividing |OA| by it IS
      // |OB|=|OA|/cos α
      const tanX = Slx.add(Sqx.mul(hgt).mul(2)) as unknown as NF;
      const tanZ = Slz.add(Sqz.mul(hgt).mul(2)) as unknown as NF;
      const ex = rd.x.sub(tanX.mul(rd.y)).toVar() as unknown as NF;
      const ez = rd.z.sub(tanZ.mul(rd.y)).toVar() as unknown as NF;
      const eLen = vec2(ex, ez).length().max(1e-5).toVar() as unknown as NF;

      // ---- one density-tier-mixed LUT fetch at (qx,qz,az) ---------------------------
      // R = 1/(1+d), GBA = normal/root-id. Tiers are NESTED (shared blades bake
      // identical texels), lerped by where dens falls between their cell-keep
      // fractions; below the sparsest tier R fades to the miss encoding.
      const missNorm = 1 / (1 + rayBake.dMaxTile);
      const tierW = (hi: number, lo: number): NF =>
        dens
          .sub(TIER_FRACS[lo] as number)
          .div((TIER_FRACS[hi] as number) - (TIER_FRACS[lo] as number))
          .clamp(0, 1) as unknown as NF;
      const lutSample = (qx: NF, qz: NF, azA: NF): NV4 => {
        const tap = (i: number): NV4 =>
          texture3D(
            rayBake.texs[i] as unknown as Parameters<typeof texture3D>[0],
            vec3(qx, qz, azA) as unknown as NV3,
            0,
          ) as unknown as NV4;
        const s = vec4(0, 0, 0, 0).toVar() as unknown as NV4;
        If(dens.greaterThanEqual(TIER_FRACS[1] as number), () => {
          (s as unknown as { assign(v: unknown): void }).assign(mix(tap(1), tap(0), tierW(0, 1)));
        })
          .ElseIf(dens.greaterThanEqual(TIER_FRACS[2] as number), () => {
            (s as unknown as { assign(v: unknown): void }).assign(mix(tap(2), tap(1), tierW(1, 2)));
          })
          .ElseIf(dens.greaterThanEqual(TIER_FRACS[3] as number), () => {
            (s as unknown as { assign(v: unknown): void }).assign(mix(tap(3), tap(2), tierW(2, 3)));
          })
          .Else(() => {
            const s3 = (tap(3) as unknown as { toVar(): NV4 }).toVar();
            const w = dens.div(TIER_FRACS[3] as number) as unknown as NF;
            (s as unknown as { assign(v: unknown): void }).assign(
              vec4(mix(float(missNorm), s3.x as unknown as NF, w), s3.y, s3.z, s3.w),
            );
          });
        return s;
      };

      // ---- ANTI-TILING: Sannikov texture-bombing (hex 3-tap, decoupled grid) --------
      // shadertoy tsVGRd / his 2023 flow-map update: three hex-lattice nodes, each
      // an integer-hashed rotation + phase (pcg2d — stable, no fp drift), triangle-
      // weight blended so there is NO seam anywhere. Variance-preservation restores
      // the silhouette contrast that averaging N taps softens. Hex region ≈ 2 tiles.
      const HEXR = 1.7320508; // √3
      const HEX_TILES = qNum('grasshex', 2, 0.5, 8); // hex region size, in tiles
      const uhx = wtx.div(HEX_TILES).toVar() as unknown as NF;
      const uhz = wtz.div(HEX_TILES).toVar() as unknown as NF;
      const NODEOFF: [number, number][] = [
        [0, 0],
        [1, 1],
        [1, -1],
      ];
      const mkNode = (ni: number): { w: NF; Rx: NF; nx: NF; ny: NF; nz: NF } => {
        // uv = worldTile/HEX_TILES + nodeOffset/hexRatio·0.5  (hexRatio = (1, √3))
        const ux = uhx.add((NODEOFF[ni] as [number, number])[0] * 0.5) as unknown as NF;
        const uz = uhz.add(((NODEOFF[ni] as [number, number])[1] * 0.5) / HEXR) as unknown as NF;
        // even lattice a, odd lattice b; pick the nearer hex centre (Shane)
        const ax = ux.add(0.5).floor() as unknown as NF;
        const azi = uz.div(HEXR).add(0.5).floor() as unknown as NF;
        const cAx = ax as unknown as NF;
        const cAz = azi.mul(HEXR) as unknown as NF;
        const bx = ux.floor() as unknown as NF; // round(ux−0.5)
        const bz = uz.sub(1).div(HEXR).add(0.5).floor() as unknown as NF;
        const cBx = bx.add(0.5) as unknown as NF;
        const cBz = bz.add(0.5).mul(HEXR) as unknown as NF;
        const oAx = ux.sub(cAx);
        const oAz = uz.sub(cAz);
        const oBx = ux.sub(cBx);
        const oBz = uz.sub(cBz);
        const dA = oAx.mul(oAx).add(oAz.mul(oAz)) as unknown as NF;
        const dB = oBx.mul(oBx).add(oBz.mul(oBz)) as unknown as NF;
        const useA = dA.lessThanEqual(dB) as unknown as NB;
        const sel = (a: NF, b: NF): NF =>
          (useA as unknown as { select(a: unknown, b: unknown): NF }).select(a, b);
        const cX = sel(cAx, cBx);
        const cZ = sel(cAz, cBz);
        const idX = sel(ax, bx);
        // fold the even/odd lattice bit into the key (disjoint integer ranges) so
        // a- and b-cells never collide in the hash (pcg2d truncates to uint)
        const idZ = sel(azi, bz.add(131072));
        // weight = HexSDF(uv−centre)·2 (∈[0,1], the 3 nodes sum ≈ 1)
        const pX = ux.sub(cX).abs() as unknown as NF;
        const pZ = uz.sub(cZ).abs() as unknown as NF;
        const w = float(0.5)
          .sub(pX.mul(0.5).add(pZ.mul(HEXR * 0.5)).max(pX))
          .mul(2)
          .max(0)
          .toVar() as unknown as NF;
        // this node's hashed rotation + phase
        const key = vec2(idX, idZ) as unknown as NV2;
        const ang = (cellHash(key, SALT ^ 0x4e1b) as unknown as NF)
          .mul(Math.PI * 2)
          .toVar() as unknown as NF;
        const cs = ang.cos().toVar() as unknown as NF;
        const sn = ang.sin().toVar() as unknown as NF;
        const ph = cellHash2(key, SALT ^ 0x2c9d) as unknown as NV2;
        // rotate tile coord + view ray by this node's angle (LUT repeat-wraps qx,qz)
        const qx = Ptx.mul(cs).sub(Ptz.mul(sn)).add(ph.x) as unknown as NF;
        const qz = Ptx.mul(sn).add(Ptz.mul(cs)).add(ph.y) as unknown as NF;
        const rex = ex.mul(cs).sub(ez.mul(sn)) as unknown as NF;
        const rez = ex.mul(sn).add(ez.mul(cs)) as unknown as NF;
        const azN = (atan(rez, rex) as unknown as NF).mul(1 / (Math.PI * 2)).fract() as unknown as NF;
        const tap = lutSample(qx, qz, azN);
        // baked normal azimuth is TILE space → world = R(−ang)·(cos,sin)·sinθ
        const bAz = (tap.y as unknown as NF).mul(Math.PI * 2) as unknown as NF;
        const nyy = (tap.z as unknown as NF).mul(2).sub(1) as unknown as NF;
        const sxz = float(1).sub(nyy.mul(nyy)).max(0).sqrt() as unknown as NF;
        const txx = bAz.cos().mul(sxz) as unknown as NF;
        const tzz = bAz.sin().mul(sxz) as unknown as NF;
        return {
          w,
          Rx: tap.x as unknown as NF,
          nx: txx.mul(cs).add(tzz.mul(sn)) as unknown as NF, // R(−ang)·xz
          ny: nyy,
          nz: tzz.mul(cs).sub(txx.mul(sn)) as unknown as NF,
        };
      };
      const N0 = mkNode(0);
      const N1 = mkNode(1);
      const N2 = mkNode(2);
      const sumW = N0.w.add(N1.w).add(N2.w).max(1e-4) as unknown as NF;
      const invW = float(1).div(sumW) as unknown as NF;
      const sumR = N0.w.mul(N0.Rx).add(N1.w.mul(N1.Rx)).add(N2.w.mul(N2.Rx)) as unknown as NF;
      const sumW2 = N0.w.mul(N0.w).add(N1.w.mul(N1.w)).add(N2.w.mul(N2.w)) as unknown as NF;
      // variance-preserve the blended R toward the baked mean → crisp silhouettes
      const Rb = sumR.mul(invW) as unknown as NF;
      const m2 = sumW2.mul(invW).mul(invW).max(1e-4) as unknown as NF; // Σ(wᵢ/Σw)²
      const Rv = Rb.sub(rayBake.meanR)
        .div(m2.sqrt())
        .add(rayBake.meanR)
        .clamp(0, 1) as unknown as NF;
      const dTile = float(1).div(Rv.max(1 / 255)).sub(1).toVar() as unknown as NF;
      returnIf(dTile.greaterThan(rayBake.dMaxTile * 0.94) as unknown as NB); // baked miss
      // blended world normal (pre per-cell twist — applied once B's cell is known)
      const nWx0 = N0.w.mul(N0.nx).add(N1.w.mul(N1.nx)).add(N2.w.mul(N2.nx)).mul(invW)
        .toVar() as unknown as NF;
      const nWy0 = N0.w.mul(N0.ny).add(N1.w.mul(N1.ny)).add(N2.w.mul(N2.ny)).mul(invW)
        .toVar() as unknown as NF;
      const nWz0 = N0.w.mul(N0.nz).add(N1.w.mul(N1.nz)).add(N2.w.mul(N2.nz)).mul(invW)
        .toVar() as unknown as NF;
      // |OB| = |OA|/cos α: dTile is the in-tile 2D path, eLen the projection scale
      const tHit = tE.add(dTile.mul(GUIDE_PITCH).div(eLen)).toVar() as unknown as NF;
      // behind the scene hit (incl. under terrain): the election would lose anyway —
      // skip the atomics
      returnIf(tHit.greaterThanEqual(tMax) as unknown as NB);

      // ---- B: depth + normal + tip param → the unchanged emit/resolve path ----------
      const yH = ro.y.add(rd.y.mul(tHit)).toVar() as unknown as NF;
      const phB = phAt(tHit);
      const phBx = phB.x.toVar() as unknown as NF;
      const phBz = phB.z.toVar() as unknown as NF;
      // bilinear-smooth ground under B (the march's smoothGroundAt lesson): the
      // per-texel FACETED plane gives every tile a coherent tip-param offset →
      // a brightness step per tile (the altitude-grid class). 4 corner loads.
      const gB = ((): NF => {
        const qx = relCells(phBx, gfx).div(GUIDE_SUB).sub(0.5) as unknown as NF;
        const qz = relCells(phBz, gfz).div(GUIDE_SUB).sub(0.5) as unknown as NF;
        const ix = qx.floor().clamp(0, GUIDE_RES - 2).toVar() as unknown as NF;
        const iz = qz.floor().clamp(0, GUIDE_RES - 2).toVar() as unknown as NF;
        const fxg = qx.sub(ix).clamp(0, 1) as unknown as NF;
        const fzg = qz.sub(iz).clamp(0, 1) as unknown as NF;
        const gAt = (dx: number, dz: number): NF =>
          bcU2F(
            guideCtx4.element(uint(iz.add(dz).mul(GUIDE_RES).add(ix.add(dx))) as unknown as NU)
              .x as unknown as NU,
          ) as unknown as NF;
        return mix(
          mix(gAt(0, 0), gAt(1, 0), fxg),
          mix(gAt(0, 1), gAt(1, 1), fxg),
          fzg,
        ) as unknown as NF;
      })().toVar() as unknown as NF;
      returnIf(yH.lessThan(gB.sub(0.05)) as unknown as NB); // fiber below local ground
      // flat-local validity check: O's texel plane extrapolated to B must agree
      // with the bilinear ground field — they diverge exactly where terrain
      // breaks (cliff edges, gorge lips), which is where the single-fetch frame
      // hallucinates hanging-grass curtains. Flat/gentle meadows agree to cm.
      const gOB = groundO
        .add(gradO.x.mul(phBx.sub(texCOx)))
        .add(gradO.y.mul(phBz.sub(texCOz))) as unknown as NF;
      returnIf(gB.sub(gOB).abs().greaterThan(0.35) as unknown as NB);
      // world fine cell under B — election body id + the normal's decorrelation twist
      const wcx = (streamed ? phBx.div(CELL).add(gfx) : phBx.div(CELL))
        .floor()
        .toVar() as unknown as NF;
      const wcz = (streamed ? phBz.div(CELL).add(gfz) : phBz.div(CELL))
        .floor()
        .toVar() as unknown as NF;
      const sxs = wcx.sub(wcx.div(GRID).floor().mul(GRID));
      const sys = wcz.sub(wcz.div(GRID).floor().mul(GRID));
      (tBest as unknown as { assign(v: unknown): void }).assign(tHit);
      (bodyBest as unknown as { assign(v: unknown): void }).assign(
        uint(sys.mul(GRID).add(sxs)).shiftLeft(uint(6)),
      );
      // blended world normal (from the hex taps) + the WORLD-CELL azimuth twist
      // (per-tile normal statistics decorrelation — the flatres-stop lesson): rotate
      // the blended xz by a per-fine-cell angle so adjacent tiles don't share a
      // normal signature
      const twist = cellHash(vec2(wcx, wcz) as unknown as NV2, SALT ^ 0x6a6a)
        .sub(0.5)
        .mul(1.6) as unknown as NF;
      const tc = twist.cos() as unknown as NF;
      const ts = twist.sin() as unknown as NF;
      const nWx = nWx0.mul(tc).sub(nWz0.mul(ts)) as unknown as NF;
      const nWz = nWx0.mul(ts).add(nWz0.mul(tc)) as unknown as NF;
      (nrmV as unknown as { assign(v: unknown): void }).assign(vec3(nWx, nWy0, nWz));
      // per-cell blade-top jitter (march parity: topEff = topB·(0.55..1.05)) —
      // shading-only here: adjacent blades get different tip params, which is
      // the fine-grain ragged ramp the march had (a REJECT would speckle)
      const colH = cellHash(vec2(wcx, wcz) as unknown as NV2, SALT ^ 0x7c01) as unknown as NF;
      (tParV as unknown as { assign(v: unknown): void }).assign(
        yH.sub(gB).div(swardH.mul(colH.mul(0.5).add(0.55)).max(0.05)).clamp(0, 1),
      );

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
                // the article's depth+normal output: normal + tip param ride a
                // screen texture to the resolve (the 30-bit election id can't
                // carry them). Same bottom-up row indexing — resolveRay matches.
                textureStore(rayNrmTex, uvec2(x2, y2), vec4(nrmV, tParV)).toWriteOnly();
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
    // S6c: guide-origin-relative ray origin (metres) — f64 subtract of two ~311 km
    // values done HERE on the CPU (exact) so the GPU march never forms the huge
    // absolute coordinate. guideOriginWorld = uGFx·CELL (uGFx exact integer).
    uRoMx.value = camera.position.x - uGFx.value * CELL;
    uRoMz.value = camera.position.z - uGFz.value * CELL;
    // world-anchored tile origin (exact int guide origin / GUIDE_SUB = tiles),
    // reduced mod 4096 tiles in f64 so the shader's world-tile coord is precise
    // AND camera-invariant (the anti-tiling lattice locks to the world).
    const otx = uGFx.value / GUIDE_SUB;
    const otz = uGFz.value / GUIDE_SUB;
    uOTx.value = otx - Math.floor(otx / 4096) * 4096;
    uOTz.value = otz - Math.floor(otz / 4096) * 4096;
    // separate dispatches (own submits) — keeps c.grassGuide / c.grassRay pass
    // timers clean (batched compute overlaps the render passes and smears their
    // timestamps; the whole-frame A/B is the ground truth either way)
    dispatch(renderer, kGuideBake);
    dispatch(renderer, kRay);
  };

  return {
    batch: [],
    renderHw: runGrass,
    resolveRay,
    setEnabled(v: boolean): void {
      onCpu = v;
      uOn.value = v ? 1 : 0;
    },
    enabled: () => onCpu,
    readCounts,
  };
}
