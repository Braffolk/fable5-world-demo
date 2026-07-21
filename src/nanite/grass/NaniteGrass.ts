/**
 * NaniteGrass — the Sannikov precomputed-raycast grass lane (G-E, default-on;
 * docs/deep-review/grass-raycast.txt + docs/perf-runs/2026-07-03-grass-arc.md).
 *
 * TRUE fixed-fetch O(1) runtime (2026-07-20 — the 256-step march is deleted):
 * per pixel, the elected SCENE DEPTH gives the shell point O (terrain
 * rasterizes every frame), the guide field reconstructs the sward-top entry E,
 * and a bounded set of trilinear fetches from the boot-baked (x, z-in-tile,
 * angle) raycast volumes (GrassRayBake.ts — R = path 1/(1+d), G/B = normal
 * azimuth/y, A = root-cell id; traced over INFINITELY TILED fibers) answers the
 * whole ray without any step loop:
 * |OB| = |OA|/cos α, B = E + rd·|OB|. Spatially-varying world density is the
 * one deviation from the article's uniform tiling — solved by DENSITY-TIERED
 * bakes (nested cell thinning): root validity selects one complete bracketing
 * record, never a blended depth. Wind rides the exact non-orthogonal quadratic
 * basis (off = Sl·h + Sq·h²). Anti-tiling is a nearest-hit union of two global,
 * incommensurately scaled geometry layers at golden-ratio·π separation. Hits emit
 * a self-describing id + depth into the vis-buffer election; normal + tip ride a screen
 * StorageTexture to the resolve (grassProc.ray).
 *
 * A per-frame GUIDE FIELD (camera-centered ctx storage buffer + bilinear
 * field textures) carries the world in: ground + gradient, density
 * (mask popcount), wind/shear coefficients, gust amplitude.
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
/** Shared with GrassRayBake's pcg2d root keep threshold. */
const TIER_KEEP_SALT = SALT ^ 0x37a1;
/** election id namespace: bit31|bit30 (voxel = bit31 only, mesh < bit30) */
const GRASS_FLAGS = 0xc0000000;
/** far-tuft ids live above this body offset (fine max = GRID²·64 ≈ 604M).
 *  Exported for the resolve's far-pixel cheap path. */
export const GRASS_FAR_BASE = 0x28000000; // 671M (no producer since the geo-lane delete)
/** ?grassdbg=raysetup — BUILD-TIME kernel stop (ray gen + scene-depth reconstruct
 *  only); ?grassdbg=flatres lives in NaniteResolve. Production pristine unset. */
const GRASS_DBG = new URLSearchParams(window.location.search).get('grassdbg');
// THE LANE (single, DEFAULT-ON — user calls 2026-07-04): the Sannikov article
// algorithm (docs/deep-review/grass-raycast.txt) as a TRUE per-pixel fixed-fetch
// query — boot-baked (x, z, angle) raycast tiles answer the whole ray at the
// terrain-anchored sward entry (O(1), no march; 2026-07-20 rebuild — the
// 256-step guide march this replaced is in git history); wind = his
// non-orthogonal derivative basis; anti-tiling = two global geometry layers. The old
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
// Production defaults stay on Sannikov's geometrically exact parallel-extrusion
// case. His shift/thicken extensions are retained as explicit research knobs;
// the source warns that both can artifact when viewed along the fibers.
const BAKE_SHIFTK = qNum('grassshiftk', 0, 0, 1);
const BAKE_THICKK = qNum('grassthickk', 0, 0, 2);
/** sway amplitude scale (?grasssway=K, 0 = steady gust-bend only) */
const RAY_SWAY = qNum('grasssway', 1, 0, 5);
/** ?grassrayend=N — band-end knob for cost attribution (default 155) */
const RAY_END = ((): number => {
  const v = Number(new URLSearchParams(window.location.search).get('grassrayend') ?? '155');
  return Number.isFinite(v) && v >= 20 && v <= 300 ? v : 155;
})();
/** DENSITY TIERS (cell-keep fractions, descending): one baked LUT volume each.
 *  The world thins grass per cell; runtime tests the denser bracketing tier's
 *  nearest root and atomically chooses its full record or the nested sparser
 *  fallback. Below the last tier the same root test chooses hit or miss. */
const TIER_FRACS = [1, 0.55, 0.3, 0.12];
// ---- GUIDE FIELD (ray lane) — world context for the fixed-cost query ----------------
// A camera-centered world-space context field REBAKED EVERY FRAME by a tiny compute
// pass (O(area), blade-count-independent): kRay FETCHES its world instead of deriving
// it in-register. Per 0.84 m texel (= 8×8 fine cells):
//   ctx  (uvec4): ground f32 | ground gradient half2 | (swardTop, gustAmp) half2
//   mask (uvec2): 64-bit fine-cell occupancy — the DENSITY LAW baked to bits
//     (bit = cellHash(cell) < dens·edge); kRay consumes only its POPCOUNT
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
}
function bladeTable(blades: number, segs: number): BladePar[] {
  let s = 1234567 + blades * 77 + segs * 13;
  const rnd = (): number => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const out: BladePar[] = [];
  for (let b = 0; b < blades; b++) {
    const yaw = rnd() * Math.PI * 2;
    const c = Math.cos(yaw);
    const sn = Math.sin(yaw);
    const ox = (rnd() - 0.5) * 0.16;
    const oz = (rnd() - 0.5) * 0.16;
    const hk = 0.62 + rnd() * 0.65;
    const lean = (rnd() - 0.5) * 0.42;
    out.push({
      c,
      s: sn,
      ox,
      oz,
      hk,
      lean,
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
  //   [2] typeA|typeB<<8|blend<<16|vigor<<24
  //   [3] clumpLo|clumpHi<<8|moisture<<16|canopyProximity<<24
  //   [4..5] reserved (zero)
  //   [6..7] exact mirror of [2..3] for the approved external 32B contract
  // Words 2..3 ride the same uvec4 reads kRay already performs for ground/grad;
  // the mirrored tail is not fetched by kRay. No second record-half load is added.
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
  // precise. The two anti-tiling geometry layers must be locked to the WORLD,
  // not the camera; camera-relative transforms previously reset the pattern.
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
  /** T3.x = continuous per-texel ground-cover density — the tier-select
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
        return densityAt(cw, heightAt(cw), dC, false)
          .mul(eC)
          .toVar() as unknown as NF;
      };
      const p00 = pCorner(-1, -1);
      const p10 = pCorner(1, -1);
      const p01 = pCorner(-1, 1);
      const p11 = pCorner(1, 1);
      // The retired 8x8 occupancy/popcount loop only estimated the mean of this
      // bilinear probability field; no consumer used its mask after the O(1)
      // rebuild. Store the analytic expectation directly: smoother, fixed ALU,
      // and no per-frame shader loop.
      const densOut = p00.add(p10).add(p01).add(p11).mul(0.25).clamp(0, 1)
        .toVar() as unknown as NF;
      const topOff = float(0.75) as unknown as NF;
      const topOut = densOut.greaterThanEqual(0.02).select(topOff, float(0)) as unknown as NF;
      // gust amplitude at the texel — rebaked EVERY frame, so wind stays live
      const amp = (windContext()
        ? (windU.strength as unknown as NF)
            .mul(gustAt(wpos).mul(0.9).add(0.3))
            .mul(windExposure(wpos))
        : (float(0) as unknown as NF)) as unknown as NF;
      // Cook-side two-type control. Missing/legacy manifests compile to canonical
      // grass (id 0) and the current procedural density, preserving old body ids.
      const controlA = field.hasGroundCover
        ? field.groundCoverAt(wpos)
        : (vec4(0) as unknown as NV4);
      const controlB = field.hasGroundCover
        ? field.groundCoverLinearAt(wpos)
        : (vec4(0, densOut, 0, 0) as unknown as NV4);
      const byte = (v: NF): NU => uint(v.mul(255).add(0.5).floor()) as unknown as NU;
      const typeA = byte(controlA.x as unknown as NF);
      const typeB = byte(controlA.y as unknown as NF);
      const blendB = byte(controlB.x as unknown as NF);
      const vigorB = byte(controlB.y as unknown as NF);
      const clumpLo = byte(controlA.z as unknown as NF);
      const clumpHi = byte(controlA.w as unknown as NF);
      const moistureB = byte(controlB.z as unknown as NF);
      const canopyB = byte(controlB.w as unknown as NF);
      const mixWord = typeA
        .bitOr(typeB.shiftLeft(uint(8)))
        .bitOr(blendB.shiftLeft(uint(16)))
        .bitOr(vigorB.shiftLeft(uint(24))) as unknown as NU;
      const envWord = clumpLo
        .bitOr(clumpHi.shiftLeft(uint(8)))
        .bitOr(moistureB.shiftLeft(uint(16)))
        .bitOr(canopyB.shiftLeft(uint(24))) as unknown as NU;
      // merged 8-word record (see layout at declaration).
      const base = i.mul(uint(REC_WORDS_PER));
      guideRecW.rw.element(base).assign(bcF2U(g));
      guideRecW.rw.element(base.add(uint(1))).assign(packHalfU(vec2(dgdx, dgdz) as unknown as NV2));
      guideRecW.rw.element(base.add(uint(2))).assign(mixWord);
      guideRecW.rw.element(base.add(uint(3))).assign(envWord);
      guideRecW.rw.element(base.add(uint(4))).assign(uint(0));
      guideRecW.rw.element(base.add(uint(5))).assign(uint(0));
      guideRecW.rw.element(base.add(uint(6))).assign(mixWord);
      guideRecW.rw.element(base.add(uint(7))).assign(envWord);
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
      // NO static per-tile lean/arc. It deliberately tipped whole 0.84 m patches in
      // hash-RANDOM directions to "break the vertical-prism look" — which is exactly
      // the user's "grass grows from the side / random unexplainable directions."
      // Blades grow STRAIGHT UP; only WIND (coherent + directional, below) leans
      // them, all the same way like real wind. (?grassrandlean=K re-adds it, 0..1.)
      const randLean = qNum('grassrandlean', 0, 0, 1);
      const nT = smN(0x3333);
      const baT = (nT.x as unknown as NF).mul(6.2831853).toVar() as unknown as NF;
      const bmT = (nT.y as unknown as NF).mul(0.25 * randLean).add(0.12 * randLean) as unknown as NF;
      const a1xB = baT.cos().mul(bmT) as unknown as NF;
      const a1zB = baT.sin().mul(bmT) as unknown as NF;
      const laT = (nT.x as unknown as NF).mul(6.2831853).add(2.1) as unknown as NF;
      const lmT = (nT.y as unknown as NF).mul(0.05 * randLean).add(0.02 * randLean) as unknown as NF;
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
        vec4(densOut, toF(typeA).div(255), toF(typeB).div(255), toF(blendB).div(255)),
      ).toWriteOnly();
    })().compute(GUIDE_N, [256]);
    (k as unknown as { setName(n: string): void }).setName('grassGuide');
    return k;
  })();

  // ---- G-E: THE ARTICLE'S PRECOMPUTATION (⚠️ USER DIRECTIVE 2026-07-04) ---------------
  // Boot-baked raycast tile (GrassRayBake.ts): 3D texture (x, z in tile, angle) →
  // R = path length 1/(1+d), GBA = normal — traced over infinitely-tiled FULL-density
  // exact parallel clump geometry by default (optional shift/thicken research
  // knobs remain zero). LINEAR filter +
  // REPEAT wrap on all three axes (his interpolation, incl. across angle slices).
  // The tile = one guide texel footprint (0.84 m, 8×8 fine cells).
  const rayBake = ((): { texs: Data3DTexture[]; dMaxTile: number } => {
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
      // Two globally transformed geometry layers are composited below. Six
      // actual fibers per layer replace the coverage that the retired distance-
      // thicken heuristic had been fabricating.
      fibers: Math.round(qNum('grassbakn', 6, 2, 16)),
      tiers: TIER_FRACS,
      keepSalt: TIER_KEEP_SALT,
      arcK: qNum('grassarck', 0, 0, 3),
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
    // One independent fixed-cost ray per output pixel. The retired 2×2 shortcut
    // copied one hit/depth/normal into four pixels and produced screen-space
    // swimming under motion; full-resolution is still O(1), only correctly sampled.
    const k = Fn(() => {
      returnIf((uOn as unknown as NF).lessThan(0.5) as unknown as NB);
      const qi = instanceIndex;
      returnIf(qi.greaterThanEqual(uint(W * H)));
      const xI = qi.mod(uint(W)).toVar() as unknown as NU;
      const yI = qi.div(uint(W)).toVar() as unknown as NU; // bottom-up rows
      const px = yI.mul(uint(W)).add(xI).toVar() as unknown as NU;
      const ndcX = toF(xI).add(0.5).div(W).mul(2).sub(1);
      const ndcY = toF(yI).add(0.5).div(H).mul(2).sub(1);
      const hf4 = cam.invVp.mul(vec4(ndcX, ndcY, 1, 1));
      const ro = vec3(cam.camPos).toVar() as unknown as NV3;
      const rd = (hf4.xyz.div(hf4.w).sub(ro).normalize().toVar()) as unknown as NV3;
      // Scene early-out: this pixel's current election depth bounds the lookup.
      const tMax = float(1e9).toVar() as unknown as NF;
      const elect = aLoadU(vis.payloadV.atomic.element(px));
      If(elect.notEqual(uint(0)), () => {
        const czS = float(1).sub(toF(elect.shiftRight(uint(8))).div(16777215));
        const hs = cam.invVp.mul(vec4(ndcX, ndcY, czS, 1));
        (tMax as unknown as { assign(v: unknown): void }).assign(
          hs.xyz.div(hs.w).sub(ro).length().add(0.3),
        );
      });
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


      // ---- G-E FIXED-COST QUERY (the article's runtime, terrain-anchored) ------------
      // The 256-step march is DELETED (2026-07-20): it re-walked per pixel what the
      // precompute already answered. The article's runtime is exactly: at the shell
      // fragment O, transform the view ray into the tile's oblique frame, fixed LUT
      // candidates → in-tile path |OA| → 3D hit |OB| = |OA|/cos α → reconstruct B, emit
      // B's depth+normal. Our shell fragment is the ELECTED SCENE HIT (terrain
      // rasterizes every frame); the sward-top entry E is reconstructed from O via
      // the guide texel's ground plane. The infinite-tiling precompute IS the
      // multi-tile first-hit, so a grazing ray crossing many tiles never starts a
      // march. Candidate count and cost are density- and distance-independent.
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
      /** bilinear guide sampler (ground + gradient) at a ph-frame position given as
       *  its fine-cell offset rc. The O/E/B planes were NEAREST per 0.84 m texel, so
       *  the entry, height and descent JUMPED at every guide-cell edge — the hard
       *  cell-edge clipping the user still sees, and a per-cell warp as the camera
       *  moves. Bilinear ⇒ C0-continuous across space. */
      const bguide = (rcx: NF, rcz: NF): { g: NF; grad: NV2; mixWord: NU; envWord: NU } => {
        const qx = rcx.div(GUIDE_SUB).sub(0.5) as unknown as NF;
        const qz = rcz.div(GUIDE_SUB).sub(0.5) as unknown as NF;
        const ix = qx.floor().clamp(0, GUIDE_RES - 2).toVar() as unknown as NF;
        const iz = qz.floor().clamp(0, GUIDE_RES - 2).toVar() as unknown as NF;
        const fx = qx.sub(ix).clamp(0, 1) as unknown as NF;
        const fz = qz.sub(iz).clamp(0, 1) as unknown as NF;
        const cv = (dx: number, dz: number): NV4 =>
          guideCtx4.element(
            uint(iz.add(dz).mul(GUIDE_RES).add(ix.add(dx))) as unknown as NU,
          ) as unknown as NV4;
        const c00 = cv(0, 0);
        const c10 = cv(1, 0);
        const c01 = cv(0, 1);
        const c11 = cv(1, 1);
        const g = mix(
          mix(bcU2F(c00.x as unknown as NU), bcU2F(c10.x as unknown as NU), fx),
          mix(bcU2F(c01.x as unknown as NU), bcU2F(c11.x as unknown as NU), fx),
          fz,
        ) as unknown as NF;
        const grad = mix(
          mix(
            unpackHalfU(c00.y as unknown as NU) as unknown as NV2,
            unpackHalfU(c10.y as unknown as NU) as unknown as NV2,
            fx,
          ),
          mix(
            unpackHalfU(c01.y as unknown as NU) as unknown as NV2,
            unpackHalfU(c11.y as unknown as NU) as unknown as NV2,
            fx,
          ),
          fz,
        ) as unknown as NV2;
        // Categorical control comes from the nearest guide record; interpolating
        // ids or clump identity would invent nonexistent species at ecotones.
        // c00..c11 are already live for ground/gradient, so this adds no load.
        const nearMix0 = (fx.lessThan(0.5) as unknown as { select(a: unknown, b: unknown): NU })
          .select(c00.z, c10.z);
        const nearMix1 = (fx.lessThan(0.5) as unknown as { select(a: unknown, b: unknown): NU })
          .select(c01.z, c11.z);
        const nearEnv0 = (fx.lessThan(0.5) as unknown as { select(a: unknown, b: unknown): NU })
          .select(c00.w, c10.w);
        const nearEnv1 = (fx.lessThan(0.5) as unknown as { select(a: unknown, b: unknown): NU })
          .select(c01.w, c11.w);
        const mixWord = (fz.lessThan(0.5) as unknown as { select(a: unknown, b: unknown): NU })
          .select(nearMix0, nearMix1);
        const envWord = (fz.lessThan(0.5) as unknown as { select(a: unknown, b: unknown): NU })
          .select(nearEnv0, nearEnv1);
        return { g, grad, mixWord, envWord };
      };

      // ---- O's texel plane → sward height + the entry point E ----------------------
      const phO = phAt(tScene);
      const phOx = phO.x.toVar() as unknown as NF;
      const phOz = phO.z.toVar() as unknown as NF;
      const rcOx = relCells(phOx, gfx).toVar() as unknown as NF;
      const rcOz = relCells(phOz, gfz).toVar() as unknown as NF;
      const og = bguide(rcOx, rcOz);
      const groundO = (og.g as unknown as { toVar(): NF }).toVar() as unknown as NF;
      const gradO = (og.grad as unknown as { toVar(): NV2 }).toVar() as unknown as NV2;
      // steep-slope gate (~50°): the flat-local frame is meaningless on cliff
      // faces (Taevaskoja terraces painted grass curtains); the cook's density
      // law zeroes these anyway — this kills the bilinear bleed band too
      returnIf(
        gradO.x.mul(gradO.x).add(gradO.y.mul(gradO.y)).greaterThan(1.0) as unknown as NB,
      );
      // ANALYTIC sward height — continuous across space (the baked per-texel top
      // is a 0.84 m STAIRCASE; any geometry keyed to it re-grows the grid quilt).
      // Physical height is world-stable; only a world-anchored 0.42 m value noise
      // varies the ragged (non-mowed) sward top. Camera distance may choose a
      // representation, never change the grass's real height.
      // Noise lattice = 4 fine cells: gf is texel-snapped (8-cell multiple) so
      // gf/4 is an exact integer — hash coords absolute, fractions from the small
      // relative frame (S6c: no 311 km cancellation).
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
      const swardH = float(0.5)
        .mul(rag.mul(0.24).add(0.88))
        .clamp(0.12, 1.1)
        .toVar() as unknown as NF;
      const pOy = ro.y.add(rd.y.mul(tScene)) as unknown as NF;
      // height of O above the bilinear ground at O (≈0 on terrain; >0 on a trunk/
      // rock — grass in front of it still renders, hits behind it lose the election)
      const hO = pOy.sub(groundO).toVar() as unknown as NF;
      // Guide fields are anchored at O, the exact scene/terrain point. Sampling
      // them at walked-back E made patch ownership move up-ray at every camera
      // translation. O supplies both density and the local oblique basis.
      const guv = vec2(
        rcOx.div(GUIDE_SUB * GUIDE_RES),
        rcOz.div(GUIDE_SUB * GUIDE_RES),
      ).clamp(0, 1) as unknown as NV2;
      const f1 = (texture(guideFieldT1, guv, 0) as unknown as { toVar(): NV4 }).toVar();
      const f2 = (texture(guideFieldT2, guv, 0) as unknown as { toVar(): NV4 }).toVar();
      const dens = (texture(guideFieldT3, guv, 0).x as unknown as NF)
        .clamp(0, 1)
        .toVar() as unknown as NF;
      returnIf(dens.lessThan(0.02) as unknown as NB);
      const Slx = (f1.x as unknown as NF).toVar() as unknown as NF;
      const Slz = (f1.y as unknown as NF).toVar() as unknown as NF;
      const Sqx = (f1.z as unknown as NF).add(f2.z).toVar() as unknown as NF;
      const Sqz = (f1.w as unknown as NF).add(f2.w).toVar() as unknown as NF;

      // Exact non-orthogonal local coordinates for x=P+F(h), y=g(P)+h,
      // F(h)=Sl·h+Sq·h² and planar g with gradient m:
      //   Q(h)=(1-m·Sl)h-(m·Sq)h², Q(h)-Q(hO)=k·Δt,
      //   k=rd.y-m·rd.xz.
      // This is closed-form algebra; no ray steps are introduced.
      const basisC = float(1)
        .sub(gradO.x.mul(Slx).add(gradO.y.mul(Slz)))
        .toVar() as unknown as NF;
      const basisB = gradO.x.mul(Sqx).add(gradO.y.mul(Sqz)).toVar() as unknown as NF;
      const qAt = (h: NF): NF => basisC.mul(h).sub(basisB.mul(h).mul(h)) as unknown as NF;
      const kGround = rd.y
        .sub(gradO.x.mul(rd.x))
        .sub(gradO.y.mul(rd.z))
        .min(-1e-3)
        .toVar() as unknown as NF;
      // O is normally the terrain (hO≈0). If an opaque object supplied O above
      // the sward, shear is undefined there; use ordinary world height until the
      // ray enters the sward instead of extrapolating F beyond its physical domain.
      const qO = (hO.lessThanEqual(swardH) as unknown as { select(a: unknown, b: unknown): NF })
        .select(qAt(hO), hO)
        .toVar() as unknown as NF;
      const qTop = qAt(swardH).toVar() as unknown as NF;
      // E = where the ray crosses the sward top (h = swardH), walked back from O —
      // CAPPED to ±3 tiles of horizontal travel: a grazing ray's uncapped
      // walk-back extrapolates O's texel plane tens of metres (floating bright
      // slabs at the far band); capping keeps the fetch anchored to LOCAL guide
      // data (a capped entry starts inside the sward — blades above the segment
      // are clipped, sub-pixel at the ranges where the cap binds). Camera inside
      // the sward ⇒ clamp to the march's old near start.
      const tBackMax = float(GUIDE_PITCH * 3).div(dirL) as unknown as NF;
      const dtE = qTop.sub(qO).div(kGround).clamp(tBackMax.negate(), tBackMax)
        .toVar() as unknown as NF;
      const tE = tScene
        .add(dtE)
        .max(0.05)
        .toVar() as unknown as NF;

      // ---- E: exact shear height + world-fixed layer coordinates -------------------
      const phE = phAt(tE);
      const phEx = phE.x.toVar() as unknown as NF;
      const phEz = phE.z.toVar() as unknown as NF;
      // Invert Q stably: h=2q/(c+sqrt(c²-4bq)); this also handles b→0.
      const hFromQ = (q: NF): NF => {
        const disc = basisC.mul(basisC).sub(basisB.mul(q).mul(4)).max(1e-5) as unknown as NF;
        return q.mul(2).div(basisC.add(disc.sqrt()).max(1e-4)) as unknown as NF;
      };
      const qE = qO.add(kGround.mul(tE.sub(tScene))).clamp(0, qTop.max(1e-4)) as unknown as NF;
      const hgt = hFromQ(qE).clamp(0, swardH).toVar() as unknown as NF;
      const offX = Slx.add(Sqx.mul(hgt)).mul(hgt) as unknown as NF; // Sl·h + Sq·h²
      const offZ = Slz.add(Sqz.mul(hgt)).mul(hgt) as unknown as NF;
      // WORLD-ANCHORED tile coordinate (uOT folds the exact-integer guide origin
      // back in, mod 4096 tiles): worldTile = phE/PITCH + uOT ⇒ camera-invariant,
      // so both geometry layers lock to the WORLD instead of resetting on every
      // camera-position move.
      const wtx = (
        streamed ? phEx.div(GUIDE_PITCH).add(uOTx as unknown as NF) : phEx.div(GUIDE_PITCH)
      ).toVar() as unknown as NF;
      const wtz = (
        streamed ? phEz.div(GUIDE_PITCH).add(uOTz as unknown as NF) : phEz.div(GUIDE_PITCH)
      ).toVar() as unknown as NF;
      // wind shear (metres → tiles) displaces the sampling coordinate
      const Ptx = wtx.sub(offX.div(GUIDE_PITCH)).toVar() as unknown as NF;
      const Ptz = wtz.sub(offZ.div(GUIDE_PITCH)).toVar() as unknown as NF;
      // sheared view-ray horizontal → the fetch's angle axis; eLen (rotation-
      // invariant) is the in-tile projection scale ⇒ dividing |OA| by it IS
      // |OB|=|OA|/cos α
      const tanX = Slx.add(Sqx.mul(hgt).mul(2)) as unknown as NF;
      const tanZ = Slz.add(Sqz.mul(hgt).mul(2)) as unknown as NF;
      const basisDen = basisC.sub(basisB.mul(hgt).mul(2)).toVar() as unknown as NF;
      returnIf(basisDen.lessThanEqual(0.05) as unknown as NB);
      const dhdt = kGround.div(basisDen).toVar() as unknown as NF;
      const ex = rd.x.sub(tanX.mul(dhdt)).toVar() as unknown as NF;
      const ez = rd.z.sub(tanZ.mul(dhdt)).toVar() as unknown as NF;
      const eLen = vec2(ex, ez).length().max(1e-5).toVar() as unknown as NF;

      // ---- density-conditioned COMPLETE hit record at (qx,qz,az) -------------------
      // R = 1/(1+d), GB = normal, A = root-cell id. A first-hit record is not a
      // colour: linearly mixing two tiers invents a depth/normal that belongs to
      // neither geometry. The tiers are nested. For density between two tiers,
      // test the denser tier's nearest root against the exact bake-time keep hash;
      // keep that whole record when valid, otherwise use the sparser tier's whole
      // record. This is a fixed pair of taps, never a march.
      const missNorm = 1 / (1 + rayBake.dMaxTile);
      const lutSample = (qx: NF, qz: NF, azA: NF): { rec: NV4; near: NV4 } => {
        const tap = (i: number): NV4 =>
          texture3D(
            rayBake.texs[i] as unknown as Parameters<typeof texture3D>[0],
            vec3(qx, qz, azA) as unknown as NV3,
            0,
          ) as unknown as NV4;
        // The filtered hit may straddle texels; root ownership follows the nearest
        // texel. Sampling the exact texel centre through the SAME bound 3D texture
        // returns that texel exactly even with a linear sampler, so no extra binding
        // is needed and the integer root id cannot be interpolated.
        const tc = vec3(
          qx.fract().mul(BAKE_RES).floor().add(0.5).div(BAKE_RES),
          qz.fract().mul(BAKE_RES).floor().add(0.5).div(BAKE_RES),
          azA.fract().mul(BAKE_ANG).floor().add(0.5).div(BAKE_ANG),
        ) as unknown as NV3;
        const nearTap = (i: number): NV4 =>
          texture3D(
            rayBake.texs[i] as unknown as Parameters<typeof texture3D>[0],
            tc,
            0,
          ) as unknown as NV4;
        const rootKeep = (rt: NV4): NF => {
          const ri = (rt.w as unknown as NF).mul(GUIDE_SUB * GUIDE_SUB).floor().clamp(0, 63)
            .toVar() as unknown as NF;
          const rv = ri.div(GUIDE_SUB).floor() as unknown as NF;
          const ru = ri.sub(rv.mul(GUIDE_SUB)) as unknown as NF;
          return cellHash(vec2(ru, rv) as unknown as NV2, TIER_KEEP_SALT) as unknown as NF;
        };
        const miss = vec4(missNorm, 0, 166 / 255, 1) as unknown as NV4;
        const s = miss.toVar() as unknown as NV4;
        const n = miss.toVar() as unknown as NV4;
        const choosePair = (hi: number, lo: number): void => {
          const h = tap(hi);
          const l = tap(lo);
          const hn = nearTap(hi);
          const ln = nearTap(lo);
          const useHi = rootKeep(hn).lessThan(dens) as unknown as NB;
          (s as unknown as { assign(v: unknown): void }).assign(
            (useHi as unknown as { select(a: unknown, b: unknown): NV4 })
              .select(h, l),
          );
          (n as unknown as { assign(v: unknown): void }).assign(
            (useHi as unknown as { select(a: unknown, b: unknown): NV4 })
              .select(hn, ln),
          );
        };
        If(dens.greaterThanEqual(0.999), () => {
          (s as unknown as { assign(v: unknown): void }).assign(tap(0));
          (n as unknown as { assign(v: unknown): void }).assign(nearTap(0));
        })
          .ElseIf(dens.greaterThanEqual(TIER_FRACS[1] as number), () => {
            choosePair(0, 1);
          })
          .ElseIf(dens.greaterThanEqual(TIER_FRACS[2] as number), () => {
            choosePair(1, 2);
          })
          .ElseIf(dens.greaterThanEqual(TIER_FRACS[3] as number), () => {
            choosePair(2, 3);
          })
          .Else(() => {
            const h = tap(3);
            const hn = nearTap(3);
            const useHi = rootKeep(hn).lessThan(dens) as unknown as NB;
            (s as unknown as { assign(v: unknown): void }).assign(
              (useHi as unknown as { select(a: unknown, b: unknown): NV4 })
                .select(h, miss),
            );
            (n as unknown as { assign(v: unknown): void }).assign(
              (useHi as unknown as { select(a: unknown, b: unknown): NV4 })
                .select(hn, miss),
            );
          });
        return { rec: s, near: n };
      };

      // ---- ANTI-TILING: two real geometry layers, nearest complete hit wins ---------
      // The source article's geometry-safe prescription is to overlay layers at
      // different global angles (golden-ratio*pi maximizes their repeat period).
      // Applying the later colour-texture hex blend directly to first-hit depths
      // fabricated hollow Voronoi cells. These transforms never change by region,
      // so each layer is a continuous infinite extrusion; their union is the nearer
      // of two complete fixed-cost candidates.
      const sampleLayer = (
        ang: number,
        scale: number,
        phaseX: number,
        phaseZ: number,
      ): { dWorldTile: NF; nx: NF; ny: NF; nz: NF } => {
        const cs = Math.cos(ang);
        const sn = Math.sin(ang);
        const qx = Ptx.mul(scale).mul(cs).sub(Ptz.mul(scale).mul(sn)).add(phaseX) as unknown as NF;
        const qz = Ptx.mul(scale).mul(sn).add(Ptz.mul(scale).mul(cs)).add(phaseZ) as unknown as NF;
        const rex = ex.mul(cs).sub(ez.mul(sn)) as unknown as NF;
        const rez = ex.mul(sn).add(ez.mul(cs)) as unknown as NF;
        const azN = (atan(rez, rex) as unknown as NF).mul(1 / (Math.PI * 2)).fract() as unknown as NF;
        const hit = lutSample(qx, qz, azN);
        const origin = (hit.near.x as unknown as NF).greaterThanEqual(254.5 / 255) as unknown as NB;
        // The exact d=0 sample is categorical occupancy, not a filterable depth.
        // Preserve its whole record so |OA|/cos(alpha) remains exactly zero for
        // vertical/near-vertical rays instead of exploding a filtered epsilon.
        const rec = (origin as unknown as { select(a: unknown, b: unknown): NV4 })
          .select(hit.near, hit.rec);
        // baked normal azimuth is layer space -> rotate it back to world space
        const bAz = (rec.y as unknown as NF).mul(Math.PI * 2) as unknown as NF;
        const ny = (rec.z as unknown as NF).mul(2).sub(1) as unknown as NF;
        const sxz = float(1).sub(ny.mul(ny)).max(0).sqrt() as unknown as NF;
        const tx = bAz.cos().mul(sxz) as unknown as NF;
        const tz = bAz.sin().mul(sxz) as unknown as NF;
        const dTile = float(1).div((rec.x as unknown as NF).max(1 / 255)).sub(1) as unknown as NF;
        const valid = dTile.lessThan(rayBake.dMaxTile * 0.94) as unknown as NB;
        return {
          // q-space advances `scale` times faster than world-tile space.
          dWorldTile: (valid as unknown as { select(a: unknown, b: unknown): NF })
            .select(dTile.div(scale), float(1e6)),
          nx: tx.mul(cs).add(tz.mul(sn)) as unknown as NF,
          ny,
          nz: tz.mul(cs).sub(tx.mul(sn)) as unknown as NF,
        };
      };
      const L0 = sampleLayer(0, 1, 0, 0);
      const L1 = sampleLayer(Math.PI * 1.618033988749895, 1.071773462536293, 0.371, 0.619);
      const take0 = L0.dWorldTile.lessThanEqual(L1.dWorldTile) as unknown as NB;
      const pick = (a: NF, b: NF): NF =>
        (take0 as unknown as { select(x: unknown, y: unknown): NF }).select(a, b);
      const dWorldTile = pick(L0.dWorldTile, L1.dWorldTile).toVar() as unknown as NF;
      const nWx0 = pick(L0.nx, L1.nx).toVar() as unknown as NF;
      const nWy0 = pick(L0.ny, L1.ny).toVar() as unknown as NF;
      const nWz0 = pick(L0.nz, L1.nz).toVar() as unknown as NF;
      returnIf(dWorldTile.greaterThan(1e5) as unknown as NB); // both layers miss
      // |OB| = |OA|/cos α: dTile is the in-tile 2D path, eLen the projection scale
      const tHit = tE.add(dWorldTile.mul(GUIDE_PITCH).div(eLen)).toVar() as unknown as NF;
      // behind the scene hit (incl. under terrain): the election would lose anyway —
      // skip the atomics
      returnIf(tHit.greaterThanEqual(tMax) as unknown as NB);

      // ---- B: recover base-space root + physical height ----------------------------
      const yH = ro.y.add(rd.y.mul(tHit)).toVar() as unknown as NF;
      const phB = phAt(tHit);
      const phBx = phB.x.toVar() as unknown as NF;
      const phBz = phB.z.toVar() as unknown as NF;
      const qBraw = qO.add(kGround.mul(tHit.sub(tScene))).toVar() as unknown as NF;
      returnIf(qBraw.lessThan(-0.03).or(qBraw.greaterThan(qTop.add(0.08))) as unknown as NB);
      const hB = hFromQ(qBraw.clamp(0, qTop.max(1e-4)) as unknown as NF)
        .clamp(0, swardH)
        .toVar() as unknown as NF;
      const offBX = Slx.add(Sqx.mul(hB)).mul(hB) as unknown as NF;
      const offBZ = Slz.add(Sqz.mul(hB)).mul(hB) as unknown as NF;
      const baseBx = phBx.sub(offBX).toVar() as unknown as NF;
      const baseBz = phBz.sub(offBZ).toVar() as unknown as NF;
      // Ground and identity belong to P (the blade's base coordinate), not to the
      // horizontally displaced surface point x. Using g(x) made leaned blades on
      // slopes read as downhill/upside-down curtains.
      const bg = bguide(relCells(baseBx, gfx), relCells(baseBz, gfz));
      const gB = (bg.g as unknown as { toVar(): NF }).toVar() as unknown as NF;
      const mixWordB = bg.mixWord.toVar() as unknown as NU;
      const envWordB = bg.envWord.toVar() as unknown as NU;
      returnIf(yH.sub(gB.add(hB)).abs().greaterThan(0.15) as unknown as NB);
      // flat-local validity check: O's texel plane extrapolated to B must agree
      // with the bilinear ground field — they diverge exactly where terrain
      // breaks (cliff edges, gorge lips), which is where the single-fetch frame
      // hallucinates hanging-grass curtains. Flat/gentle meadows agree to cm.
      const gOB = groundO
        .add(gradO.x.mul(baseBx.sub(phOx)))
        .add(gradO.y.mul(baseBz.sub(phOz))) as unknown as NF;
      returnIf(gB.sub(gOB).abs().greaterThan(0.35) as unknown as NB);
      // world fine ROOT cell under B — election identity is base-space stable
      const wcx = (streamed ? baseBx.div(CELL).add(gfx) : baseBx.div(CELL))
        .floor()
        .toVar() as unknown as NF;
      const wcz = (streamed ? baseBz.div(CELL).add(gfz) : baseBz.div(CELL))
        .floor()
        .toVar() as unknown as NF;
      const sxs = wcx.sub(wcx.div(GRID).floor().mul(GRID));
      const sys = wcz.sub(wcz.div(GRID).floor().mul(GRID));
      const typeA = mixWordB.bitAnd(uint(0xff));
      const typeB = mixWordB.shiftRight(uint(8)).bitAnd(uint(0xff));
      const blend = toF(mixWordB.shiftRight(uint(16)).bitAnd(uint(0xff))).mul(1 / 255) as unknown as NF;
      const clumpLo = envWordB.bitAnd(uint(0xff));
      const clumpHi = envWordB.shiftRight(uint(8)).bitAnd(uint(0xff));
      // Stable whole-record type election. It is rooted in the recovered base
      // cell and cook-side clump id, never screen/pixel noise, so adjacent pixels
      // on one fiber agree and type boundaries remain world-anchored.
      const typePick = cellHash(
        vec2(wcx.add(toF(clumpLo)), wcz.add(toF(clumpHi))) as unknown as NV2,
        SALT ^ 0x6c31,
      ) as unknown as NF;
      const coverId = (typePick.lessThan(blend) as unknown as { select(a: unknown, b: unknown): NU })
        .select(typeB, typeA)
        .bitAnd(uint(0x3f));
      (tBest as unknown as { assign(v: unknown): void }).assign(tHit);
      (bodyBest as unknown as { assign(v: unknown): void }).assign(
        uint(sys.mul(GRID).add(sxs)).shiftLeft(uint(6)).bitOr(coverId),
      );
      // Keep the selected geometry record intact. A post-hit per-cell normal twist
      // changed shading without changing the surface and visibly reintroduced a
      // square cell signature; the baked face normal is already decorrelated by
      // the two global layers.
      (nrmV as unknown as { assign(v: unknown): void }).assign(vec3(nWx0, nWy0, nWz0));
      (tParV as unknown as { assign(v: unknown): void }).assign(
        hB.div(swardH.max(0.05)).clamp(0, 1),
      );

      If(tBest.lessThan(1e8), () => {
        const hit = ro.add(rd.mul(tBest));
        const clip = cam.vp.mul(vec4(hit, 1));
        const cz = clip.z.div(clip.w.max(NEAR_EPS));
        If(cz.greaterThanEqual(0).and(cz.lessThanEqual(1)), () => {
          emitPx(px as unknown as NU, cz as unknown as NF, bodyBest);
          // The article's depth+normal output: same bottom-up pixel convention.
          textureStore(rayNrmTex, uvec2(xI, yI), vec4(nrmV, tParV)).toWriteOnly();
        });
      });
    })().compute(W * H, [256]);
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
