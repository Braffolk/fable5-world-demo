/**
 * NaniteGrass — the Sannikov precomputed-raycast grass lane (G-E, default-on;
 * docs/deep-research/grass/Predraschyot-raycasta PDF +
 * docs/tasks/2026-07-21/GRASS-STATUS-AND-ISSUES.md).
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

import { ClampToEdgeWrapping, Data3DTexture, RepeatWrapping } from 'three';
import { HalfFloatType, LinearFilter, NearestFilter, RGBAFormat } from 'three';
import type { PerspectiveCamera } from 'three';
import { StorageBufferAttribute, StorageTexture, type Renderer } from 'three/webgpu';
import {
  Fn,
  If,
  atan,
  atomicMax,
  atomicStore,
  cross,
  float,
  int,
  instanceIndex,
  mix,
  normalize,
  smoothstep,
  texture,
  texture3D,
  textureLoad,
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
import { bakeGrassRayTile, packGroundCoverRayAtlas } from '../build/GrassRayBake';
import { GROUND_COVER_ID_MASK, GroundCoverId } from '../groundcover/GroundCoverTypes';
import {
  GROUND_COVER_PROFILE_COUNT,
  GROUND_COVER_PROFILE_FUNCTIONAL_IDS,
  GroundCoverProfileId,
  type LoadedPeriodicProfile,
} from '../groundcover/GroundCoverProfiles';
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
/** Diagnostic-only species isolation. `?grassprofile=2` replaces every live
 * cooked ground-cover patch with the selected canonical GCAR layer so one
 * authored geometry can be judged without moss/forb/shrub silhouettes. This is
 * a graph-build constant: the ordinary production graph is unchanged. */
const GRASS_PROFILE_OVERRIDE = (() => {
  const raw = new URLSearchParams(window.location.search).get('grassprofile');
  if (raw === null) return null;
  const value = Number(raw);
  return Number.isInteger(value) && value >= 0 && value < GROUND_COVER_PROFILE_COUNT
    ? value
    : null;
})();
/** Experimental local-plane reprojection retained for exact A/B diagnosis only.
 * Arbitrary first-hit records do not carry triangle extent/correspondence, so
 * extrapolating their tangent planes across angular bins can create unbounded
 * grazing streaks. The accepted path keeps complete stored hits instead. */
const PERIODIC_PLANE_REPROJECT = new URLSearchParams(window.location.search)
  .get('grassreproject') === '1';
/** Retain the exact-owner correspondence experiment for diagnosis without
 * making its reject-on-mismatch behavior the authored-profile default. */
const PERIODIC_EXACT_OWNER = new URLSearchParams(window.location.search)
  .get('grassowner') === '1';
/** The literal analytic extrusion remains available as an explicit diagnostic;
 * the ordinary isolated-species URL renders the authored periodic carrier. */
const BASE_EXTRUSION_DIAGNOSTIC = new URLSearchParams(window.location.search)
  .get('grassbase') === '1';
// THE LANE (single, DEFAULT-ON — user calls 2026-07-04): the Sannikov article
// algorithm (the archived GameDev.ru PDF under docs/deep-research/grass) as a
// TRUE per-pixel fixed-fetch
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
/** Rejected analytic cushion fixture. It remains available only for numerical/
 * visual comparison and is never compiled into the production shader graph. */
const GROUND_COVER_CAP_ORACLE = new URLSearchParams(window.location.search)
  .get('groundcovercaporacle') === '1';
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
/** Rounded-cushion ray elevation slices: normalized height lost per horizontal
 * tile. Log spacing covers grazing through near-vertical ground views while the
 * runtime still selects one precomputed block and never marches. */
const MOSS_DROP_BINS = [0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24] as const;
const PROFILE_BLOCK_BASE = GROUND_COVER_CAP_ORACLE
  ? [0, 1, 1 + MOSS_DROP_BINS.length, 10, 11, 12] as const
  : [0, 1, 2, 3, 4, 5] as const;
const COVER_PARAM_TABLE = [
  [0.5, 0.24, 1, PROFILE_BLOCK_BASE[0]],
  [0.085, 0.08, 0, PROFILE_BLOCK_BASE[1]],
  [0.68, 0.22, 0.72, PROFILE_BLOCK_BASE[2]],
  [0.045, 0.06, 0, PROFILE_BLOCK_BASE[3]],
  [0.22, 0.16, 0.22, PROFILE_BLOCK_BASE[4]],
  [0.3, 0.12, 0.12, PROFILE_BLOCK_BASE[5]],
] as const;
/** Four rgba8 density tiers combined. The ordinary atlas is ~2.3 MiB. This
 * fail-closed cap prevents research URL maxima from requesting ~900 MiB before
 * device-limit validation; production GPU-baked profiles will have an explicit
 * offline budget instead of growing this boot-time reference allocation. */
const RAY_ATLAS_MAX_BYTES = 64 * 1024 * 1024;
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
  /** Validated before graph construction. One filterable atlas binding is used
   * by each authored profile; absent profiles stay on explicit integration
   * fixtures and are never mistaken for botanical assets. */
  periodicProfiles: readonly LoadedPeriodicProfile[];
}


export interface GrassField {
  /** compute kernels for world1's batched submit (after kVisClear, before kHwArgs) */
  batch: readonly unknown[];
  /** the grass HW blade pass (own render target, HARDWARE EARLY-Z): a fullscreen
   *  prime writes the election depth into a real depth buffer, then the blade
   *  draw runs depth-tested — occluded blade fragments never invoke the election
   *  shader. Call right after the raster's hwRender (world1 only). */
  renderHw(renderer: Renderer, camera: PerspectiveCamera): void;
  /** Height of the rasterized source-method outer shell, or null when this
   * graph does not consume one. */
  shellHeight: number | null;
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
  const periodicProfilesById = new Map(
    opts.periodicProfiles.map((profile) => [profile.profileId, profile] as const),
  );
  const periodicProfileArrayTexture = opts.periodicProfiles[0]?.texture ?? null;
  const isolatedPeriodicProfile = GRASS_PROFILE_OVERRIDE !== null
    && opts.periodicProfiles.length === 1
    && opts.periodicProfiles[0]!.profileId === GRASS_PROFILE_OVERRIDE
    && opts.periodicProfiles[0]!.textureLayer === null;
  /** Acceptance checkpoint: establish the article's exact 2D extrusion basis at
   * Calamagrostis' authored stature before height-varying botanical detail is
   * reintroduced. This is a graph-build choice, so the four-view GCRP sampler,
   * its extra texture reads, and its elevation interpolation do not survive in
   * the generated acceptance shader. Production/multi-profile code is retained. */
  const baseExtrusionAcceptance = isolatedPeriodicProfile
    && GRASS_PROFILE_OVERRIDE === GroundCoverProfileId.CalamagrostisCanescens
    && !PERIODIC_EXACT_OWNER
    && BASE_EXTRUSION_DIAGNOSTIC;
  const shellHeight = isolatedPeriodicProfile
    ? opts.periodicProfiles[0]!.topH
    : null;
  if (field.hasGroundCoverClosure) {
    const canonicalArray =
      opts.periodicProfiles.length !== GROUND_COVER_PROFILE_COUNT
        ? false
        : !opts.periodicProfiles.some((profile) =>
          profile.texture !== periodicProfileArrayTexture
          || profile.textureLayer !== profile.profileId
        );
    if (!canonicalArray && !isolatedPeriodicProfile) {
      throw new Error(
        'ground-cover closure requires the canonical array or its explicit isolated-profile acceptance carrier',
      );
    }
    if (canonicalArray && opts.periodicProfiles.some((profile) =>
        profile.texture !== periodicProfileArrayTexture
        || profile.textureLayer !== profile.profileId
    )) {
      throw new Error('ground-cover closure requires one canonical profile array with layer == profile id');
    }
  }
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

  /** Branchless atlas-profile parameters: [physical height, rag amplitude,
   * deformation gain, first atlas block]. The TypeScript loop expands the six comparisons while
   * building TSL; it does not emit a WGSL loop. */
  const coverParams = (id: NU): NV4 => {
    let e = vec4(...COVER_PARAM_TABLE[5]) as unknown as NV4;
    for (let i = 4; i >= 0; i--) {
      e = (id.equal(uint(i)) as unknown as { select(a: unknown, b: unknown): NV4 })
        .select(vec4(...COVER_PARAM_TABLE[i]!) as unknown as NV4, e);
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
  //   [2] typeA|typeB<<8|candidateMaskLo<<16|candidateMaskHi<<24
  //   [3] clumpLo|clumpHi<<8|profileA<<16|profileB<<24
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
      // v2 carries the cook-proven root-reach candidate closure and exact
      // profile pair. Old v1 manifests fail safe with all six functional bits:
      // bounded over-query is slower, but it cannot punch ecotone holes.
      const controlC = field.hasGroundCoverClosure
        ? field.groundCoverProfilesAt(wpos)
        : (vec4(63 / 255, 0, 0, 0) as unknown as NV4);
      const coverDens = (field.hasGroundCover
        ? (controlB.y as unknown as NF)
        : densOut).clamp(0, 1) as unknown as NF;
      const topOut = coverDens.greaterThanEqual(0.02).select(float(0.75), float(0)) as unknown as NF;
      const byte = (v: NF): NU => uint(v.mul(255).add(0.5).floor()) as unknown as NU;
      const typeA = byte(controlA.x as unknown as NF);
      const typeB = byte(controlA.y as unknown as NF);
      const clumpLo = byte(controlA.z as unknown as NF);
      const clumpHi = byte(controlA.w as unknown as NF);
      const candidateMaskLo = byte(controlC.x as unknown as NF);
      const candidateMaskHi = byte(controlC.y as unknown as NF);
      const profileA = field.hasGroundCoverClosure
        ? byte(controlC.z as unknown as NF)
        : typeA;
      const profileB = field.hasGroundCoverClosure
        ? byte(controlC.w as unknown as NF)
        : typeB;
      const mixWord = typeA
        .bitOr(typeB.shiftLeft(uint(8)))
        .bitOr(candidateMaskLo.shiftLeft(uint(16)))
        .bitOr(candidateMaskHi.shiftLeft(uint(24))) as unknown as NU;
      const envWord = clumpLo
        .bitOr(clumpHi.shiftLeft(uint(8)))
        .bitOr(profileA.shiftLeft(uint(16)))
        .bitOr(profileB.shiftLeft(uint(24))) as unknown as NU;
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
        // Continuous carrier only. Categorical ids/clump remain nearest in the
        // guide record; filtering them would invent species. This existing tap
        // now also keeps blend/moisture/canopy C0-continuous across guide cells.
        vec4(coverDens, controlB.x, controlB.z, controlB.w),
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
  const rayBake = ((): {
    texs: Data3DTexture[];
    dMaxTile: number;
    angleStride: number;
    atlasDepth: number;
  } => {
    // Stable functional-form metadata matches GroundCoverId 0..5. These
    // boot-time analytic profiles are integration fixtures, not authored native
    // species. The rejected moss cap ladder is opt-in oracle-only. A last-slice
    // prefix and first-slice suffix preserve periodic angle interpolation inside
    // every block.
    const profiles = [
      {
        id: GroundCoverId.Grass,
        label: 'grass',
        section: 'rectangle' as const,
        halfW: qNum('grassbakw', 0.0055, 0.001, 0.05),
        halfT: qNum('grassbakt', 0.003, 0.001, 0.02),
        fibers: Math.round(qNum('grassbakn', 6, 2, 16)),
        spread: 1.3,
        arcK: qNum('grassarck', 0, 0, 3),
      },
      {
        id: GroundCoverId.Moss,
        label: 'moss-cushion',
        section: 'ellipse' as const,
        halfW: 0.052,
        halfT: 0.044,
        fibers: 4,
        spread: 1.0,
        arcK: 0,
      },
      {
        id: GroundCoverId.Sedge,
        label: 'sedge',
        section: 'rectangle' as const,
        halfW: 0.0038,
        halfT: 0.0022,
        fibers: 8,
        spread: 1.25,
        arcK: 0,
      },
      {
        id: GroundCoverId.Lichen,
        label: 'lichen-rosette',
        section: 'ellipse' as const,
        halfW: 0.041,
        halfT: 0.026,
        fibers: 3,
        spread: 0.9,
        arcK: 0,
      },
      {
        id: GroundCoverId.Forb,
        label: 'forb-rosette',
        section: 'ellipse' as const,
        halfW: 0.039,
        halfT: 0.009,
        fibers: 5,
        spread: 1.05,
        arcK: 0,
      },
      {
        id: GroundCoverId.DwarfShrub,
        label: 'dwarf-shrub',
        section: 'ellipse' as const,
        halfW: 0.024,
        halfT: 0.018,
        fibers: 5,
        spread: 1.1,
        arcK: 0,
      },
    ];
    const bakes = profiles.flatMap((profile) => {
      const drops = profile.id === GroundCoverId.Moss && GROUND_COVER_CAP_ORACLE
        ? MOSS_DROP_BINS
        : [undefined] as const;
      return drops.map((dropPerTile) => bakeGrassRayTile({
        res: BAKE_RES,
        angles: BAKE_ANG,
        blades: BLADES,
        sub: GUIDE_SUB,
        cellM: CELL,
        shiftK: profile.id === GroundCoverId.Grass ? BAKE_SHIFTK : 0,
        thickK: profile.id === GroundCoverId.Grass ? BAKE_THICKK : 0,
        halfW: profile.halfW,
        halfT: profile.halfT,
        fibers: profile.fibers,
        section: profile.section,
        shape: profile.id === GroundCoverId.Moss && GROUND_COVER_CAP_ORACLE
          ? 'ellipsoid-cap'
          : 'extruded',
        dropPerTile,
        spread: profile.spread,
        tiers: TIER_FRACS,
        keepSalt: TIER_KEEP_SALT,
        arcK: profile.arcK,
        label: dropPerTile === undefined ? profile.label : `${profile.label}@drop${dropPerTile}`,
      }));
    });
    for (let pi = 0; pi < profiles.length; pi++) {
      if (profiles[pi]?.id !== pi) throw new Error('ground-cover atlas ids must be contiguous 0..5');
      const expectedBlock = GROUND_COVER_CAP_ORACLE
        ? (pi < 2 ? pi : pi + MOSS_DROP_BINS.length - 1)
        : pi;
      if (PROFILE_BLOCK_BASE[pi] !== expectedBlock) {
        throw new Error('ground-cover profile block table is inconsistent');
      }
    }
    const atlas = packGroundCoverRayAtlas(bakes);
    const atlasBytes = atlas.data.reduce((sum, volume) => sum + volume.byteLength, 0);
    if (atlasBytes > RAY_ATLAS_MAX_BYTES) {
      throw new Error(
        `ground-cover ray atlas ${Math.ceil(atlasBytes / 1048576)} MiB exceeds `
        + `${RAY_ATLAS_MAX_BYTES / 1048576} MiB boot-reference budget`,
      );
    }
    const texs = atlas.data.map((d, i) => {
      const t = new Data3DTexture(d, atlas.res, atlas.res, atlas.depth);
      t.format = RGBAFormat;
      // A first-hit path is discontinuous at every silhouette. Linear sampling
      // between a hit and the finite miss sentinel invents a third, much longer
      // ray, which presents as a view-radial extrusion. The literal-base
      // checkpoint uses the complete nearest record; authored/production
      // carriers remain unchanged while this shared contract is isolated.
      t.minFilter = baseExtrusionAcceptance ? NearestFilter : LinearFilter;
      t.magFilter = baseExtrusionAcceptance ? NearestFilter : LinearFilter;
      t.wrapS = RepeatWrapping;
      t.wrapT = RepeatWrapping;
      t.wrapR = ClampToEdgeWrapping; // duplicated per-type seam owns angle wrap
      t.generateMipmaps = false;
      t.needsUpdate = true;
      t.name = `groundCoverRayAtlas${i}`;
      return t;
    });
    return {
      texs,
      dMaxTile: atlas.dMaxTile,
      angleStride: atlas.angleStride,
      atlasDepth: atlas.depth,
    };
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

  // All accepted GCRP layers share one direction lattice and atlas layout.
  // Depth is evaluated inside each exact root-closure candidate, but the normal
  // belongs only to the final elected profile/layer. Keeping those as two named
  // WGSL functions prevents twelve graph copies and, critically, avoids decoding
  // and normalizing four oct normals for every rejected candidate. The elected
  // normal re-fetches four cache-local records once. One array binding, no loop,
  // no barrier, no march; the hit geometry and Cartesian normal blend are exact.
  const periodicLayout = opts.periodicProfiles[0] ?? null;
  const ownedPeriodicProfile = PERIODIC_EXACT_OWNER && isolatedPeriodicProfile
    && periodicLayout?.version === 4
    && periodicLayout.ownerTexture
    && periodicLayout.vertexTexture
    && periodicLayout.triangleTexture
    && periodicLayout.sourceBounds
    ? periodicLayout
    : null;
  /** v4 exact-owner query. The four canonical directions surrounding the live
   * ray each supply one statically expanded correspondence refinement. Every
   * refinement projects that direction's first-hit height back onto the live
   * ray, then names one source triangle which is intersected exactly. This is
   * fixed-address reprojection, not a mesh traversal or ray march; WGSL has no
   * loop. */
  const sampleOwnedPeriodic = ownedPeriodicProfile
    ? (qxz: NV2, nd: NV3, tile: NV4): { t: NF; nrm: NV3; color: NV3 } => {
      const profile = ownedPeriodicProfile;
      const azF = (atan(nd.z, nd.x) as unknown as NF)
        .mul(1 / (Math.PI * 2)).fract().mul(profile.lattice.azimuthCount)
        .toVar() as unknown as NF;
      const az0 = azF.floor().toVar() as unknown as NF;
      const az1 = az0.add(1).mod(profile.lattice.azimuthCount).toVar() as unknown as NF;
      const elevation = (atan(
        nd.y.negate(),
        vec2(nd.x, nd.z).length().max(1e-5),
      ) as unknown as NF).toVar() as unknown as NF;
      const lastInterval = profile.lattice.elevationCount - 2;
      let elevation0: NF = float(lastInterval) as unknown as NF;
      let elevation1: NF = float(lastInterval + 1) as unknown as NF;
      for (let row = lastInterval - 1; row >= 0; row--) {
        const below = elevation.lessThan(profile.lattice.elevations[row + 1]!) as unknown as {
          select(a: unknown, b: unknown): NF;
        };
        elevation0 = below.select(float(row), elevation0);
        elevation1 = below.select(float(row + 1), elevation1);
      }
      const fracX = qxz.x.sub(tile.x).div(tile.z).fract().toVar() as unknown as NF;
      const fracZ = qxz.y.sub(tile.y).div(tile.w).fract().toVar() as unknown as NF;
      const wrappedX = tile.x.add(fracX.mul(tile.z)).toVar() as unknown as NF;
      const wrappedZ = tile.y.add(fracZ.mul(tile.w)).toVar() as unknown as NF;
      const rowValue = (values: NV4, row: NF): NF => {
        let value = values.w as unknown as NF;
        value = (row.equal(float(2)) as unknown as { select(a: unknown, b: unknown): NF })
          .select(values.z, value);
        value = (row.equal(float(1)) as unknown as { select(a: unknown, b: unknown): NF })
          .select(values.y, value);
        return (row.equal(float(0)) as unknown as { select(a: unknown, b: unknown): NF })
          .select(values.x, value);
      };
      const depthRows = Array.from({ length: profile.lattice.elevationCount }, (_, row) => {
        const slice = profile.slices[
          profile.lattice.order === 'azimuth-major'
            ? row
            : row * profile.lattice.azimuthCount
        ]!;
        return slice;
      });
      const depthMins = vec4(...depthRows.map((slice) => slice.depthMin)) as unknown as NV4;
      const depthMaxs = vec4(...depthRows.map((slice) => slice.depthMax)) as unknown as NV4;
      const atlasTap = (
        query: NV2,
        azimuth: NF,
        elevationRow: NF,
      ): { owner: NU; record: NV4; shiftX: NF; shiftZ: NF } => {
        const queryFracX = query.x.sub(tile.x).div(tile.z).fract().toVar() as unknown as NF;
        const queryFracZ = query.y.sub(tile.y).div(tile.w).fract().toVar() as unknown as NF;
        const texelX = queryFracX.mul(profile.interiorTileWidth).floor()
          .clamp(0, profile.interiorTileWidth - 1).toVar() as unknown as NF;
        const texelY = float(1).sub(queryFracZ).mul(profile.interiorTileHeight).floor()
          .clamp(0, profile.interiorTileHeight - 1).toVar() as unknown as NF;
        const slice = profile.lattice.order === 'azimuth-major'
          ? azimuth.mul(profile.lattice.elevationCount).add(elevationRow) as unknown as NF
          : elevationRow.mul(profile.lattice.azimuthCount).add(azimuth) as unknown as NF;
        const column = slice.mod(profile.atlasColumns) as unknown as NF;
        const atlasRow = slice.div(profile.atlasColumns).floor() as unknown as NF;
        const x = uint(column.mul(profile.storedTileWidth).add(profile.gutter).add(texelX));
        const y = uint(atlasRow.mul(profile.storedTileHeight).add(profile.gutter).add(texelY));
        const queryWrappedX = tile.x.add(queryFracX.mul(tile.z)).toVar() as unknown as NF;
        const queryWrappedZ = tile.y.add(queryFracZ.mul(tile.w)).toVar() as unknown as NF;
        return {
          owner: (textureLoad(profile.ownerTexture!, uvec2(x, y)) as unknown as { x: NU }).x,
          record: textureLoad(profile.texture, uvec2(x, y)) as unknown as NV4,
          shiftX: query.x.sub(qxz.x).sub(queryWrappedX.sub(wrappedX)).toVar() as unknown as NF,
          shiftZ: query.y.sub(qxz.y).sub(queryWrappedZ.sub(wrappedZ)).toVar() as unknown as NF,
        };
      };
      const refine = (azimuth: NF, elevationRow: NF): {
        owner: NU; record: NV4; shiftX: NF; shiftZ: NF;
      } => {
        const initial = atlasTap(qxz, azimuth, elevationRow);
        const canonicalElevation = rowValue(
          vec4(...profile.lattice.elevations) as unknown as NV4,
          elevationRow,
        ).toVar() as unknown as NF;
        const canonicalAzimuth = azimuth
          .mul((Math.PI * 2) / profile.lattice.azimuthCount).toVar() as unknown as NF;
        const canonicalHorizontal = canonicalElevation.cos().toVar() as unknown as NF;
        const canonicalDirection = vec3(
          canonicalHorizontal.mul(canonicalAzimuth.cos()),
          canonicalElevation.sin().negate(),
          canonicalHorizontal.mul(canonicalAzimuth.sin()),
        ).toVar() as unknown as NV3;
        const depthMin = rowValue(depthMins, elevationRow).toVar() as unknown as NF;
        const depthMax = rowValue(depthMaxs, elevationRow).toVar() as unknown as NF;
        const storedT = depthMin.add((initial.record.x as unknown as NF)
          .mul(depthMax.sub(depthMin))).toVar() as unknown as NF;
        const hitY = float(profile.topH)
          .add((canonicalDirection.y as unknown as NF).mul(storedT)).toVar() as unknown as NF;
        const liveSlope = vec2(nd.x, nd.z).div(nd.y.min(-1e-4)) as unknown as NV2;
        const canonicalSlope = vec2(canonicalDirection.x, canonicalDirection.z)
          .div((canonicalDirection.y as unknown as NF).min(-1e-4)) as unknown as NV2;
        const projected = qxz.add(
          liveSlope.sub(canonicalSlope).mul(hitY.sub(profile.topH)),
        ).toVar() as unknown as NV2;
        const present = (initial.record.w as unknown as NF).greaterThan(0.5) as unknown as {
          select(a: unknown, b: unknown): NV2;
        };
        const query = present.select(projected, qxz) as unknown as NV2;
        return atlasTap(query, azimuth, elevationRow);
      };
      const tableLoad = (
        textureNode: NonNullable<LoadedPeriodicProfile['vertexTexture']>,
        index: NU,
        width: number,
      ): { x: NU; y: NU; z: NU; w: NU } => textureLoad(
        textureNode,
        uvec2(index.mod(uint(width)), index.div(uint(width))),
      ) as unknown as { x: NU; y: NU; z: NU; w: NU };
      const lo16 = (word: NU): NU => word.bitAnd(uint(0xffff)) as unknown as NU;
      const hi16 = (word: NU): NU => word.shiftRight(uint(16)) as unknown as NU;
      const unit16 = (word: NU): NF => toF(word).div(65535) as unknown as NF;
      const bounds = profile.sourceBounds!;
      const decodeOctPair = (x: NF, y: NF): NV3 => {
        const ox = x.mul(2).sub(1).toVar() as unknown as NF;
        const oy = y.mul(2).sub(1).toVar() as unknown as NF;
        const oz = float(1).sub(ox.abs()).sub(oy.abs()).toVar() as unknown as NF;
        If(oz.lessThan(0), () => {
          const oldX = (ox as unknown as { toVar(): NF }).toVar() as unknown as NF;
          const sx = (oldX.greaterThanEqual(0) as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), float(-1));
          const sy = (oy.greaterThanEqual(0) as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), float(-1));
          (ox as unknown as { assign(v: unknown): void }).assign(float(1).sub(oy.abs()).mul(sx));
          (oy as unknown as { assign(v: unknown): void }).assign(float(1).sub(oldX.abs()).mul(sy));
        });
        return normalize(vec3(ox, oy, oz) as unknown as NV3) as unknown as NV3;
      };
      const bestT = float(1e6).toVar() as unknown as NF;
      const bestN = vec3(0, 1, 0).toVar() as unknown as NV3;
      const bestColor = vec3(0.05, 0.12, 0.03).toVar() as unknown as NV3;
      const considerOwner = (owner: NU, shiftX: NF, shiftZ: NF): void => {
        const present = owner.notEqual(uint(0xffff_ffff)) as unknown as NB;
        const triangleId = owner.bitAnd(uint(0x3f_ffff)).toVar() as unknown as NU;
        const copyX = toF(owner.shiftRight(uint(22)).bitAnd(uint(0x1f))).sub(16)
          .mul(tile.z).add(shiftX).toVar() as unknown as NF;
        const copyZ = toF(owner.shiftRight(uint(27)).bitAnd(uint(0x1f))).sub(16)
          .mul(tile.w).add(shiftZ).toVar() as unknown as NF;
        const triangle = tableLoad(profile.triangleTexture!, triangleId, profile.triangleTextureWidth);
        const decodeVertex = (vertexId: NU): { p: NV3; n: NV3; color: NV3 } => {
          const record = tableLoad(profile.vertexTexture!, vertexId, profile.vertexTextureWidth);
          const x = unit16(lo16(record.x)).mul(bounds[3] - bounds[0]).add(bounds[0])
            .add(copyX).sub(tile.x).sub(fracX.mul(tile.z)) as unknown as NF;
          const y = unit16(hi16(record.x)).mul(bounds[4] - bounds[1]).add(bounds[1]) as unknown as NF;
          const z = unit16(lo16(record.y)).mul(bounds[5] - bounds[2]).add(bounds[2])
            .add(copyZ).sub(tile.y).sub(fracZ.mul(tile.w)) as unknown as NF;
          return {
            p: vec3(x, y, z) as unknown as NV3,
            color: vec3(
              unit16(hi16(record.y)),
              unit16(lo16(record.z)),
              unit16(hi16(record.z)),
            ) as unknown as NV3,
            n: decodeOctPair(unit16(lo16(record.w)), unit16(hi16(record.w))),
          };
        };
        const a = decodeVertex(triangle.x);
        const b = decodeVertex(triangle.y);
        const c = decodeVertex(triangle.z);
        const e1 = b.p.sub(a.p).toVar() as unknown as NV3;
        const e2 = c.p.sub(a.p).toVar() as unknown as NV3;
        const pvec = cross(nd, e2) as unknown as NV3;
        const determinant = e1.dot(pvec).toVar() as unknown as NF;
        const inverse = float(1).div(determinant) as unknown as NF;
        const originToA = vec3(a.p.x.negate(), float(profile.topH).sub(a.p.y), a.p.z.negate())
          .toVar() as unknown as NV3;
        const u = originToA.dot(pvec).mul(inverse).toVar() as unknown as NF;
        const qvec = cross(originToA, e1) as unknown as NV3;
        const v = nd.dot(qvec).mul(inverse).toVar() as unknown as NF;
        const t = e2.dot(qvec).mul(inverse).toVar() as unknown as NF;
        const w = float(1).sub(u).sub(v).toVar() as unknown as NF;
        const valid = present
          .and(determinant.abs().greaterThan(1e-8) as unknown as NB)
          .and(u.greaterThanEqual(-2e-4) as unknown as NB)
          .and(v.greaterThanEqual(-2e-4) as unknown as NB)
          .and(w.greaterThanEqual(-2e-4) as unknown as NB)
          .and(t.greaterThanEqual(0) as unknown as NB)
          .and(t.lessThan(bestT) as unknown as NB) as unknown as NB;
        If(valid, () => {
          (bestT as unknown as { assign(v: unknown): void }).assign(t);
          const shade = normalize(a.n.mul(w).add(b.n.mul(u)).add(c.n.mul(v)) as unknown as NV3)
            .toVar() as unknown as NV3;
          If(shade.dot(nd).greaterThan(0), () => {
            (shade as unknown as { assign(v: unknown): void }).assign(shade.negate());
          });
          (bestN as unknown as { assign(v: unknown): void }).assign(shade);
          (bestColor as unknown as { assign(v: unknown): void }).assign(
            a.color.mul(w).add(b.color.mul(u)).add(c.color.mul(v)),
          );
        });
      };
      const sample0 = refine(az0, elevation0);
      const sample1 = refine(az1, elevation0);
      const sample2 = refine(az0, elevation1);
      const sample3 = refine(az1, elevation1);
      considerOwner(sample0.owner, sample0.shiftX, sample0.shiftZ);
      considerOwner(sample1.owner, sample1.shiftX, sample1.shiftZ);
      considerOwner(sample2.owner, sample2.shiftX, sample2.shiftZ);
      considerOwner(sample3.owner, sample3.shiftX, sample3.shiftZ);
      return { t: bestT, nrm: bestN, color: bestColor };
    }
    : null;
  const periodicSamplers = periodicProfileArrayTexture && periodicLayout
    ? (() => {
      const decodeRecordNormal = (record: NV4): NV3 => {
        const coverage = (record.w as unknown as NF).clamp(0, 1) as unknown as NF;
        const missWeight = float(1).sub(coverage) as unknown as NF;
        const safeCoverage = coverage.max(1 / 65535) as unknown as NF;
        const ox = (record.y as unknown as NF)
          .sub(missWeight.mul(0.5)).div(safeCoverage).clamp(0, 1)
          .mul(2).sub(1).toVar() as unknown as NF;
        const oy = (record.z as unknown as NF)
          .sub(missWeight.mul(0.5)).div(safeCoverage).clamp(0, 1)
          .mul(2).sub(1).toVar() as unknown as NF;
        const oz = float(1).sub(ox.abs()).sub(oy.abs()).toVar() as unknown as NF;
        If(oz.lessThan(0), () => {
          const oldX = (ox as unknown as { toVar(): NF }).toVar() as unknown as NF;
          const sx = (oldX.greaterThanEqual(0) as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), float(-1));
          const sy = (oy.greaterThanEqual(0) as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), float(-1));
          (ox as unknown as { assign(value: unknown): void }).assign(float(1).sub(oy.abs()).mul(sx));
          (oy as unknown as { assign(value: unknown): void }).assign(float(1).sub(oldX.abs()).mul(sy));
        });
        return normalize(vec3(ox, oy, oz) as unknown as NV3) as unknown as NV3;
      };
      const address = (
        qxz: NV2,
        nd: NV3,
        tile: NV4,
        profileLayer: NU,
      ): {
        nd: NV3;
        r00: NV4;
        r10: NV4;
        r01: NV4;
        r11: NV4;
        elevation0: NF;
        elevation1: NF;
        rowValue(values: NV4, row: NF): NF;
        w00: NF;
        w10: NF;
        w01: NF;
        w11: NF;
        weight: NF;
        bakeDirection: NV3;
        sampleDelta: NV2;
      } => {
        const azF = (atan(nd.z, nd.x) as unknown as NF)
          .mul(1 / (Math.PI * 2))
          .fract()
          .mul(periodicLayout.lattice.azimuthCount)
          .toVar() as unknown as NF;
        const az0 = azF.floor().toVar() as unknown as NF;
        const az1 = az0.add(1).mod(periodicLayout.lattice.azimuthCount).toVar() as unknown as NF;
        const azMix = azF.sub(az0).clamp(0, 1).toVar() as unknown as NF;
        const elevation = (atan(
          nd.y.negate(),
          vec2(nd.x, nd.z).length().max(1e-5),
        ) as unknown as NF).toVar() as unknown as NF;
        const lastInterval = periodicLayout.lattice.elevationCount - 2;
        let elevation0: NF = float(lastInterval) as unknown as NF;
        let elevation1: NF = float(lastInterval + 1) as unknown as NF;
        for (let i = lastInterval - 1; i >= 0; i--) {
          const below = elevation.lessThan(periodicLayout.lattice.elevations[i + 1]!) as unknown as {
            select(a: unknown, b: unknown): NF;
          };
          elevation0 = below.select(float(i), elevation0);
          elevation1 = below.select(float(i + 1), elevation1);
        }
        const rowValue = (values: NV4, row: NF): NF => {
          let value = values.w as unknown as NF;
          value = (row.equal(float(2)) as unknown as { select(a: unknown, b: unknown): NF })
            .select(values.z, value);
          value = (row.equal(float(1)) as unknown as { select(a: unknown, b: unknown): NF })
            .select(values.y, value);
          return (row.equal(float(0)) as unknown as { select(a: unknown, b: unknown): NF })
            .select(values.x, value);
        };
        const elevationLo = rowValue(
          vec4(...periodicLayout.lattice.elevations) as unknown as NV4,
          elevation0,
        );
        const elevationHi = rowValue(
          vec4(...periodicLayout.lattice.elevations) as unknown as NV4,
          elevation1,
        );
        const elevationMix = elevation.sub(elevationLo)
          .div(elevationHi.sub(elevationLo).max(1e-5))
          .clamp(0, 1)
          .toVar() as unknown as NF;
        const u = qxz.x.sub(tile.x).div(tile.z).fract().toVar() as unknown as NF;
        // The baker's framebuffer row zero is the tile's +Z edge.
        const v = float(1).sub(qxz.y.sub(tile.y).div(tile.w).fract()).toVar() as unknown as NF;
        const atlasWidth = periodicLayout.storedTileWidth * periodicLayout.atlasColumns;
        const atlasHeight = periodicLayout.storedTileHeight * periodicLayout.atlasRows;
        const tap = (azimuth: NF, row: NF): NV4 => {
          const slice = periodicLayout.lattice.order === 'azimuth-major'
            ? azimuth.mul(periodicLayout.lattice.elevationCount).add(row) as unknown as NF
            : row.mul(periodicLayout.lattice.azimuthCount).add(azimuth) as unknown as NF;
          const column = slice.mod(periodicLayout.atlasColumns).toVar() as unknown as NF;
          const atlasRow = slice.div(periodicLayout.atlasColumns).floor().toVar() as unknown as NF;
          const sampleU = u;
          const sampleV = v;
          const uv = vec2(
            column.mul(periodicLayout.storedTileWidth)
              .add(periodicLayout.gutter)
              .add(sampleU.mul(periodicLayout.interiorTileWidth))
              .div(atlasWidth),
            atlasRow.mul(periodicLayout.storedTileHeight)
              .add(periodicLayout.gutter)
              .add(sampleV.mul(periodicLayout.interiorTileHeight))
              .div(atlasHeight),
          ) as unknown as NV2;
          if (isolatedPeriodicProfile) {
            return texture(periodicProfileArrayTexture, uv, 0) as unknown as NV4;
          }
          return (texture(periodicProfileArrayTexture, uv) as unknown as {
            depth(layer: unknown): { level(lod: unknown): NV4 };
          }).depth(int(profileLayer)).level(float(0));
        };
        const r00 = tap(az0, elevation0);
        const r10 = tap(az1, elevation0);
        const r01 = tap(az0, elevation1);
        const r11 = tap(az1, elevation1);
        const oneAz = float(1).sub(azMix) as unknown as NF;
        const oneEl = float(1).sub(elevationMix) as unknown as NF;
        // Coverage is the precomputed ray's hit/miss component. It must weight
        // the complete first-hit record in the standalone acceptance carrier
        // too: treating miss records as weight 1 blends the finite miss sentinel
        // into a long false path, exactly the camera-radial stretching pattern.
        const c00 = r00.w.clamp(0, 1) as unknown as NF;
        const c10 = r10.w.clamp(0, 1) as unknown as NF;
        const c01 = r01.w.clamp(0, 1) as unknown as NF;
        const c11 = r11.w.clamp(0, 1) as unknown as NF;
        const w00 = oneAz.mul(oneEl).mul(c00).toVar() as unknown as NF;
        const w10 = azMix.mul(oneEl).mul(c10).toVar() as unknown as NF;
        const w01 = oneAz.mul(elevationMix).mul(c01).toVar() as unknown as NF;
        const w11 = azMix.mul(elevationMix).mul(c11).toVar() as unknown as NF;
        const weight = w00.add(w10).add(w01).add(w11).toVar() as unknown as NF;
        return {
          nd,
          r00,
          r10,
          r01,
          r11,
          elevation0,
          elevation1,
          rowValue,
          w00,
          w10,
          w01,
          w11,
          weight,
          bakeDirection: nd,
          sampleDelta: vec2(0, 0) as unknown as NV2,
        };
      };
      const depth = Fn(([
          qxz,
          nd,
          tile,
          depthMin,
          depthMax,
          profileLayer,
        ]: [NV2, NV3, NV4, NV4, NV4, NU]): NF => {
          const a = address(qxz, nd, tile, profileLayer);
          if (isolatedPeriodicProfile && !PERIODIC_PLANE_REPROJECT) {
            const maxProjectedTiles = Math.max(...periodicLayout.slices.map((slice) =>
              slice.depthMax * Math.hypot(slice.direction[0], slice.direction[2])
                / periodicLayout.tileSizeX));
            const missInverse = 1 / (1 + maxProjectedTiles);
            // A bilinear footprint that straddles a silhouette contains the
            // neutral miss record. Recover the covered hit carrier before the
            // angular blend; otherwise the finite miss value becomes a long
            // fabricated path. This mirrors decodeRecordNormal's coverage
            // unmixing and uses only the bake's existing RGBA record.
            const hitInverse = (record: NV4): NF => {
              const coverage = (record.w as unknown as NF).clamp(0, 1) as unknown as NF;
              return (record.x as unknown as NF)
                .sub(float(1).sub(coverage).mul(missInverse))
                .div(coverage.max(1 / 65535))
                .clamp(1 / 65535, 1) as unknown as NF;
            };
            const inversePath = hitInverse(a.r00).mul(a.w00)
              .add(hitInverse(a.r10).mul(a.w10))
              .add(hitInverse(a.r01).mul(a.w01))
              .add(hitInverse(a.r11).mul(a.w11))
              .div(a.weight.max(1e-5))
              .max(1 / 65535)
              .toVar() as unknown as NF;
            const projectedTiles = float(1).div(inversePath).sub(1).toVar() as unknown as NF;
            const liveHorizontal = vec2(a.nd.x, a.nd.z).length().max(1e-4) as unknown as NF;
            const liveT = projectedTiles.mul(tile.z).div(liveHorizontal).toVar() as unknown as NF;
            const valid = a.weight.greaterThan(0.02)
              .and(projectedTiles.lessThan(maxProjectedTiles * 0.97) as unknown as NB)
              .and(a.nd.y.lessThan(-1e-4) as unknown as NB) as unknown as NB;
            return (valid as unknown as { select(a: unknown, b: unknown): NF })
              .select(liveT, float(1e6));
          }
          const unpackDepth = (record: NV4, row: NF): NF => {
            const coverage = (record.w as unknown as NF).clamp(0, 1) as unknown as NF;
            const missWeight = float(1).sub(coverage) as unknown as NF;
            const depth01 = (record.x as unknown as NF)
              .sub(missWeight).div(coverage.max(1 / 65535)).clamp(0, 1) as unknown as NF;
            const lo = a.rowValue(depthMin, row);
            const hi = a.rowValue(depthMax, row);
            const storedT = lo.add(depth01.mul(hi.sub(lo))).toVar() as unknown as NF;
            if (!isolatedPeriodicProfile) return storedT;
            // Diagnostic-only local-plane experiment. A first-hit record lacks
            // primitive extent, so this cannot be accepted as generic geometry.
            const normal = decodeRecordNormal(record);
            const fromQueryToStoredHit = a.bakeDirection.mul(storedT).sub(vec3(
              a.sampleDelta.x,
              0,
              a.sampleDelta.y,
            ) as unknown as NV3) as unknown as NV3;
            return normal.dot(fromQueryToStoredHit)
              .div(normal.dot(a.nd).min(-1e-4)) as unknown as NF;
          };
          const tProfile = unpackDepth(a.r00, a.elevation0).mul(a.w00)
            .add(unpackDepth(a.r10, a.elevation0).mul(a.w10))
            .add(unpackDepth(a.r01, a.elevation1).mul(a.w01))
            .add(unpackDepth(a.r11, a.elevation1).mul(a.w11))
            .div(a.weight.max(1e-5))
            .toVar() as unknown as NF;
          const valid = a.weight.greaterThan(0.02)
            .and(a.nd.y.lessThan(-1e-4) as unknown as NB) as unknown as NB;
          return (valid as unknown as { select(a: unknown, b: unknown): NF })
            .select(tProfile, float(1e6));
        }).setLayout({
          name: 'groundCoverPeriodicDepth',
          type: 'float',
          inputs: [
            { name: 'qxz', type: 'vec2' },
            { name: 'nd', type: 'vec3' },
            { name: 'tile', type: 'vec4' },
            { name: 'depthMin', type: 'vec4' },
            { name: 'depthMax', type: 'vec4' },
            { name: 'profileLayer', type: 'uint' },
          ],
        });
      const normal = Fn(([
          qxz,
          nd,
          tile,
          profileLayer,
        ]: [NV2, NV3, NV4, NU]): NV3 => {
          const a = address(qxz, nd, tile, profileLayer);
          return normalize(
            decodeRecordNormal(a.r00).mul(a.w00)
              .add(decodeRecordNormal(a.r10).mul(a.w10))
              .add(decodeRecordNormal(a.r01).mul(a.w01))
              .add(decodeRecordNormal(a.r11).mul(a.w11))
              .add(vec3(0, 1e-6, 0)) as unknown as NV3,
          ) as unknown as NV3;
        }).setLayout({
          name: 'groundCoverPeriodicNormal',
          type: 'vec3',
          inputs: [
            { name: 'qxz', type: 'vec2' },
            { name: 'nd', type: 'vec3' },
            { name: 'tile', type: 'vec4' },
            { name: 'profileLayer', type: 'uint' },
          ],
        });
      // Authored colour is sampled only for the elected standalone hit. Keeping
      // this separate from `address()` prevents four extra texture taps and the
      // associated live records from entering every depth candidate.
      const color = isolatedPeriodicProfile && periodicLayout.colorTexture
        ? Fn(([
            qxz,
            nd,
            tile,
          ]: [NV2, NV3, NV4]): NV3 => {
            const azF = (atan(nd.z, nd.x) as unknown as NF)
              .mul(1 / (Math.PI * 2))
              .fract()
              .mul(periodicLayout.lattice.azimuthCount)
              .toVar() as unknown as NF;
            const az0 = azF.floor().toVar() as unknown as NF;
            const az1 = az0.add(1).mod(periodicLayout.lattice.azimuthCount).toVar() as unknown as NF;
            const azMix = azF.sub(az0).clamp(0, 1).toVar() as unknown as NF;
            const elevation = (atan(
              nd.y.negate(),
              vec2(nd.x, nd.z).length().max(1e-5),
            ) as unknown as NF).toVar() as unknown as NF;
            const lastInterval = periodicLayout.lattice.elevationCount - 2;
            let elevation0: NF = float(lastInterval) as unknown as NF;
            let elevation1: NF = float(lastInterval + 1) as unknown as NF;
            for (let row = lastInterval - 1; row >= 0; row--) {
              const below = elevation.lessThan(periodicLayout.lattice.elevations[row + 1]!) as unknown as {
                select(a: unknown, b: unknown): NF;
              };
              elevation0 = below.select(float(row), elevation0);
              elevation1 = below.select(float(row + 1), elevation1);
            }
            const rowValue = (values: NV4, row: NF): NF => {
              let value = values.w as unknown as NF;
              value = (row.equal(float(2)) as unknown as { select(a: unknown, b: unknown): NF })
                .select(values.z, value);
              value = (row.equal(float(1)) as unknown as { select(a: unknown, b: unknown): NF })
                .select(values.y, value);
              return (row.equal(float(0)) as unknown as { select(a: unknown, b: unknown): NF })
                .select(values.x, value);
            };
            const elevationLo = rowValue(
              vec4(...periodicLayout.lattice.elevations) as unknown as NV4,
              elevation0,
            );
            const elevationHi = rowValue(
              vec4(...periodicLayout.lattice.elevations) as unknown as NV4,
              elevation1,
            );
            const elevationMix = elevation.sub(elevationLo)
              .div(elevationHi.sub(elevationLo).max(1e-5))
              .clamp(0, 1)
              .toVar() as unknown as NF;
            const u = qxz.x.sub(tile.x).div(tile.z).fract().toVar() as unknown as NF;
            const v = float(1).sub(qxz.y.sub(tile.y).div(tile.w).fract()).toVar() as unknown as NF;
            const atlasWidth = periodicLayout.storedTileWidth * periodicLayout.atlasColumns;
            const atlasHeight = periodicLayout.storedTileHeight * periodicLayout.atlasRows;
            const tap = (azimuth: NF, row: NF): NV4 => {
              const slice = periodicLayout.lattice.order === 'azimuth-major'
                ? azimuth.mul(periodicLayout.lattice.elevationCount).add(row) as unknown as NF
                : row.mul(periodicLayout.lattice.azimuthCount).add(azimuth) as unknown as NF;
              const column = slice.mod(periodicLayout.atlasColumns).toVar() as unknown as NF;
              const atlasRow = slice.div(periodicLayout.atlasColumns).floor().toVar() as unknown as NF;
              const uv = vec2(
                column.mul(periodicLayout.storedTileWidth)
                  .add(periodicLayout.gutter)
                  .add(u.mul(periodicLayout.interiorTileWidth))
                  .div(atlasWidth),
                atlasRow.mul(periodicLayout.storedTileHeight)
                  .add(periodicLayout.gutter)
                  .add(v.mul(periodicLayout.interiorTileHeight))
                  .div(atlasHeight),
              ) as unknown as NV2;
              return texture(periodicLayout.colorTexture!, uv, 0) as unknown as NV4;
            };
            const r00 = tap(az0, elevation0);
            const r10 = tap(az1, elevation0);
            const r01 = tap(az0, elevation1);
            const r11 = tap(az1, elevation1);
            const oneAz = float(1).sub(azMix) as unknown as NF;
            const oneEl = float(1).sub(elevationMix) as unknown as NF;
            const b00 = oneAz.mul(oneEl).toVar() as unknown as NF;
            const b10 = azMix.mul(oneEl).toVar() as unknown as NF;
            const b01 = oneAz.mul(elevationMix).toVar() as unknown as NF;
            const b11 = azMix.mul(elevationMix).toVar() as unknown as NF;
            // Each spatial tap is already premultiplied by its hardware-filtered
            // coverage. Apply angular weights once, then unpremultiply once.
            const weight = (r00.w as unknown as NF).mul(b00)
              .add((r10.w as unknown as NF).mul(b10))
              .add((r01.w as unknown as NF).mul(b01))
              .add((r11.w as unknown as NF).mul(b11))
              .toVar() as unknown as NF;
            const rgb = (r00.xyz as unknown as NV3).mul(b00)
              .add((r10.xyz as unknown as NV3).mul(b10))
              .add((r01.xyz as unknown as NV3).mul(b01))
              .add((r11.xyz as unknown as NV3).mul(b11))
              .div(weight.max(1 / 255)) as unknown as NV3;
            return (weight.greaterThan(1 / 255) as unknown as {
              select(a: unknown, b: unknown): NV3;
            }).select(rgb, vec3(0.05, 0.12, 0.03));
          }).setLayout({
            name: 'groundCoverPeriodicColor',
            type: 'vec3',
            inputs: [
              { name: 'qxz', type: 'vec2' },
              { name: 'nd', type: 'vec3' },
              { name: 'tile', type: 'vec4' },
            ],
          })
        : null;
      return { depth, normal, color };
    })()
    : null;


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
      // Periodic normals are reconstructed only for the final elected candidate.
      // The closure body already packs profile id; only the anti-tile bit stays
      // live instead of per-candidate oct-normal vectors.
      const bestProfileId = field.hasGroundCoverClosure
        ? null
        : uint(0).toVar() as unknown as NU;
      const bestAntiLayer = uint(0).toVar() as unknown as NU;
      // ---- O/E: the source method's rasterized OUTER shell. In the isolated
      // authored-profile graph vis.depthV contains the actual terrain triangles
      // displaced by topH; it is not the later underlying scene hit. payloadV
      // remains only the ordinary scene-occlusion bound. This also preserves
      // shell silhouettes in pixels where the base terrain itself is not visible.
      let tScene: NF;
      if (shellHeight !== null) {
        const shellBits = aLoadU(vis.depthV.atomic.element(px));
        returnIf(shellBits.equal(uint(0xffffffff)) as unknown as NB);
        const czE = bcU2F(shellBits).toVar() as unknown as NF;
        const he = cam.invVp.mul(vec4(ndcX, ndcY, czE, 1));
        tScene = he.xyz.div(he.w).sub(ro).length().max(0.05).toVar() as unknown as NF;
      } else {
        // Dormant multi-cover path retains its prior terrain-anchored entry
        // until its per-profile shell contract is generalized.
        returnIf(tMax.greaterThan(1e8) as unknown as NB);
        tScene = tMax.sub(0.3).max(0.05).toVar() as unknown as NF;
      }
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
      // B's depth+normal. The isolated authored-profile graph now supplies that
      // shell fragment directly from the displaced terrain raster. The infinite-
      // tiling precompute IS the
      // multi-tile first-hit, so a grazing ray crossing many tiles never starts a
      // march. Candidate count and cost are density- and distance-independent.
      const gfx = uGFx as unknown as NF;
      const gfz = uGFz as unknown as NF;
      // Band cap is measured from the real shell entry, never the later terrain
      // depth whose horizontal distance diverges at grazing angles.
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
      // One already-bound, HW-linear control tap at O. Density/vigor, blend,
      // moisture and canopy are continuous; only categorical ids/clump below
      // remain nearest. Keeping height on nearest packed vigor rebuilt a visible
      // 0.84 m stair even after density itself had been filtered.
      const guv = vec2(
        rcOx.div(GUIDE_SUB * GUIDE_RES),
        rcOz.div(GUIDE_SUB * GUIDE_RES),
      ).clamp(0, 1) as unknown as NV2;
      const f1 = (texture(guideFieldT1, guv, 0) as unknown as { toVar(): NV4 }).toVar();
      const f2 = (texture(guideFieldT2, guv, 0) as unknown as { toVar(): NV4 }).toVar();
      const f3 = (texture(guideFieldT3, guv, 0) as unknown as { toVar(): NV4 }).toVar();
      const dens = (f3.x as unknown as NF).clamp(0, 1).toVar() as unknown as NF;
      // The source-method shell point is only the ray origin. At a cover/bare
      // boundary it can lie over bare ground while the precomputed ray reaches
      // geometry whose recovered root is inside the covered region. Rejecting
      // here clips those side surfaces into a floating top sheet. The exact
      // shell path validates density together with categorical ownership at the
      // recovered root below; the dormant ground-derived path keeps its prior
      // early-out until it receives the same exact-entry contract.
      if (shellHeight === null) returnIf(dens.lessThan(0.02) as unknown as NB);
      const mixWordO = og.mixWord;
      const candidateMask = GRASS_PROFILE_OVERRIDE === null
        ? mixWordO.shiftRight(uint(16)).bitAnd(uint(0xffff))
        : uint(1 << GRASS_PROFILE_OVERRIDE);
      const vigor = field.hasGroundCover ? dens : (float(1) as unknown as NF);
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
      const pOy = ro.y.add(rd.y.mul(tScene)) as unknown as NF;

      // A categorical profile cannot be chosen from shell point O: at an
      // oblique view O can be metres from the root returned by the LUT, so one
      // physical blade changed profile across pixels/views. Query the bounded
      // cooked A/B set instead, validate each complete candidate at its recovered
      // world root, and retain the nearer valid surface. The two calls below are
      // graph-build expansion, never a shader loop. Pure A==B patches execute one
      // candidate; only actual mixtures execute the second fixed query.
      const considerProfile = (profileId: number): void => {
      // Profile lookup happens while TypeScript expands the fixed candidate graph.
      // It emits neither a runtime map lookup nor a WGSL loop.
      const periodicProfile = periodicProfilesById.get(profileId) ?? null;
      const rigidPeriodicProfile = isolatedPeriodicProfile
        && periodicProfile?.profileId === GRASS_PROFILE_OVERRIDE;
      const usesPeriodicCarrier = periodicProfile !== null && !baseExtrusionAcceptance;
      const profileIdU = uint(profileId);
      const coverIdValue = field.hasGroundCoverClosure
        ? GROUND_COVER_PROFILE_FUNCTIONAL_IDS[profileId]!
        : profileId;
      const coverId = uint(coverIdValue);
      // profileId is a JS graph-build constant. Do not emit the dynamic six-way
      // functional-form select inside every guarded candidate.
      const params = vec4(...COVER_PARAM_TABLE[coverIdValue]!).toVar() as unknown as NV4;
      const baseHeight = params.x as unknown as NF;
      const ragAmp = params.y as unknown as NF;
      const profileBaseBlock = params.w as unknown as NF;
      const vigorScale = field.hasGroundCover
        ? vigor.mul(0.55).add(0.65)
        : (float(1) as unknown as NF);
      const swardH = (
        rigidPeriodicProfile
          ? float(periodicProfile!.topH)
          : baseHeight
              .mul(vigorScale)
              .mul(rag.mul(ragAmp).add(float(1).sub(ragAmp.mul(0.5))))
              .clamp(0, 1.1)
      ).toVar() as unknown as NF;
      // height of O above the bilinear ground at O (≈0 on terrain; >0 on a trunk/
      // rock — grass in front of it still renders, hits behind it lose the election)
      const hO = (
        shellHeight !== null
          ? swardH
          : pOy.sub(groundO)
      ).toVar() as unknown as NF;
      // Guide fields are anchored at O, the exact scene/terrain point. Sampling
      // them at walked-back E made patch ownership move up-ray at every camera
      // translation. O supplies density and the local oblique basis (f1/f2/f3
      // were fetched above so continuous controls also participate in election).
      const deformK = rigidPeriodicProfile ? (float(0) as unknown as NF) : params.z as unknown as NF;
      const Slx = (f1.x as unknown as NF).mul(deformK).toVar() as unknown as NF;
      const Slz = (f1.y as unknown as NF).mul(deformK).toVar() as unknown as NF;
      const Sqx = (f1.z as unknown as NF).add(f2.z).mul(deformK).toVar() as unknown as NF;
      const Sqz = (f1.w as unknown as NF).add(f2.w).mul(deformK).toVar() as unknown as NF;

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
      // E = where the ray crosses the sward top (h = swardH), walked back from O.
      // A periodic authored profile requires this exact top plane: starting a
      // capped ray inside the stand destroys tall stems and changes geometry by
      // view angle. The older analytic fallback retains its local three-tile cap
      // because it has no bounded authored height field beyond that region.
      const dtEUnbounded = qTop.sub(qO).div(kGround) as unknown as NF;
      const dtE = (
        shellHeight !== null
          ? float(0)
          : rigidPeriodicProfile
          ? dtEUnbounded
          : dtEUnbounded.clamp(
              float(GUIDE_PITCH * 3).div(dirL).negate(),
              float(GUIDE_PITCH * 3).div(dirL),
            )
      ).toVar() as unknown as NF;
      const tE = tScene.add(dtE).max(0.05).toVar() as unknown as NF;

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
      const basisValid = basisDen.greaterThan(0.05) as unknown as NB;
      const dhdt = kGround.div(basisDen.max(0.05)).toVar() as unknown as NF;
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
      const lutSample = (
        qx: NF,
        qz: NF,
        azA: NF,
        typeSlot: NF,
        queryDens: NF,
      ): { rec: NV4; near: NV4 } => {
        const atlasZ = typeSlot
          .mul(rayBake.angleStride)
          .add(1)
          .add(azA.fract().mul(BAKE_ANG))
          .div(rayBake.atlasDepth) as unknown as NF;
        const tap = (i: number): NV4 =>
          texture3D(
            rayBake.texs[i] as unknown as Parameters<typeof texture3D>[0],
            vec3(qx, qz, atlasZ) as unknown as NV3,
            0,
          ) as unknown as NV4;
        // The filtered hit may straddle texels; root ownership follows the nearest
        // texel. Sampling the exact texel centre through the SAME bound 3D texture
        // returns that texel exactly even with a linear sampler, so no extra binding
        // is needed and the integer root id cannot be interpolated.
        const tc = vec3(
          qx.fract().mul(BAKE_RES).floor().add(0.5).div(BAKE_RES),
          qz.fract().mul(BAKE_RES).floor().add(0.5).div(BAKE_RES),
          typeSlot
            .mul(rayBake.angleStride)
            .add(1.5)
            .add(azA.fract().mul(BAKE_ANG).floor())
            .div(rayBake.atlasDepth),
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
          const useHi = rootKeep(hn).lessThan(queryDens) as unknown as NB;
          (s as unknown as { assign(v: unknown): void }).assign(
            (useHi as unknown as { select(a: unknown, b: unknown): NV4 })
              .select(h, l),
          );
          (n as unknown as { assign(v: unknown): void }).assign(
            (useHi as unknown as { select(a: unknown, b: unknown): NV4 })
              .select(hn, ln),
          );
        };
        If(queryDens.greaterThanEqual(0.999), () => {
          (s as unknown as { assign(v: unknown): void }).assign(tap(0));
          (n as unknown as { assign(v: unknown): void }).assign(nearTap(0));
        })
          .ElseIf(queryDens.greaterThanEqual(TIER_FRACS[1] as number), () => {
            choosePair(0, 1);
          })
          .ElseIf(queryDens.greaterThanEqual(TIER_FRACS[2] as number), () => {
            choosePair(1, 2);
          })
          .ElseIf(queryDens.greaterThanEqual(TIER_FRACS[3] as number), () => {
            choosePair(2, 3);
          })
          .Else(() => {
            const h = tap(3);
            const hn = nearTap(3);
            const useHi = rootKeep(hn).lessThan(queryDens) as unknown as NB;
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
      const samplePeriodicLayer = (
        profile: LoadedPeriodicProfile,
        ang: number,
        scale: number,
        phaseX: number,
        phaseZ: number,
      ): {
        dt: NF; nx: NF; ny: NF; nz: NF; cr: NF; cg: NF; cb: NF;
        tip: NF; rootDx: NF; rootDz: NF;
      } => {
        const cs = Math.cos(ang);
        const sn = Math.sin(ang);
        // The legacy array treated guide-tile units as authored metres, shrinking
        // a 0.52 m source tile to 0.437 m. The isolated acceptance carrier keeps
        // the generator's metric XZ scale exactly.
        const coordinateScale = rigidPeriodicProfile ? GUIDE_PITCH : 1;
        const qx = Ptx.mul(scale * coordinateScale).mul(cs)
          .sub(Ptz.mul(scale * coordinateScale).mul(sn)).add(phaseX) as unknown as NF;
        const qz = Ptx.mul(scale * coordinateScale).mul(sn)
          .add(Ptz.mul(scale * coordinateScale).mul(cs)).add(phaseZ) as unknown as NF;
        // World-ray speed in the authored mesh's anisotropic profile space.
        // Its normalized vector selects the fixed 16x4 direction lattice; the
        // metric length converts the baked profile-space t back to world t.
        const profileDerivativeScale = scale * coordinateScale / GUIDE_PITCH;
        const dpx = ex.mul(profileDerivativeScale).mul(cs)
          .sub(ez.mul(profileDerivativeScale).mul(sn)) as unknown as NF;
        const dpz = ex.mul(profileDerivativeScale).mul(sn)
          .add(ez.mul(profileDerivativeScale).mul(cs)) as unknown as NF;
        const dpy = dhdt.mul(profile.topH).div(swardH.max(1e-4)) as unknown as NF;
        const metricSpeed = vec3(dpx, dpy, dpz).length().max(1e-5).toVar() as unknown as NF;
        const ndx = dpx.div(metricSpeed).toVar() as unknown as NF;
        const ndy = dpy.div(metricSpeed).toVar() as unknown as NF;
        const ndz = dpz.div(metricSpeed).toVar() as unknown as NF;
        if ((!periodicSamplers && !sampleOwnedPeriodic) || (profile.textureLayer === null && !isolatedPeriodicProfile)) {
          throw new Error(`periodic profile ${profile.profileId} has no compatible texture carrier`);
        }
        const depthRows = Array.from({ length: profile.lattice.elevationCount }, (_, row) => {
          const slice = profile.slices[
            profile.lattice.order === 'azimuth-major'
              ? row
              : row * profile.lattice.azimuthCount
          ]!;
          return slice;
        });
        const ownedHit = rigidPeriodicProfile && sampleOwnedPeriodic
          ? sampleOwnedPeriodic(
              vec2(qx, qz) as unknown as NV2,
              vec3(ndx, ndy, ndz) as unknown as NV3,
              vec4(profile.tileOriginX, profile.tileOriginZ, profile.tileSizeX, profile.tileSizeZ) as unknown as NV4,
            )
          : null;
        const tProfile = (ownedHit
          ? ownedHit.t
          : periodicSamplers!.depth(
              vec2(qx, qz),
              vec3(ndx, ndy, ndz),
              vec4(profile.tileOriginX, profile.tileOriginZ, profile.tileSizeX, profile.tileSizeZ),
              vec4(...depthRows.map((slice) => slice.depthMin)),
              vec4(...depthRows.map((slice) => slice.depthMax)),
              uint(profile.textureLayer ?? 0),
            )).toVar() as unknown as NF;
        // Periodic profiles are baked from the authored top plane, so the ray
        // distance already contains the exact botanical hit elevation. Preserve
        // that value instead of reconstructing a generic blade height from the
        // terrain shell after the lookup. The latter is only equivalent for the
        // analytic grass fixture and collapsed connected carpets toward t=0.
        const profileTip = float(profile.topH)
          .add(ndy.mul(tProfile))
          .div(profile.topH)
          .clamp(0, 1)
          .toVar() as unknown as NF;
        const dt = tProfile.div(metricSpeed).toVar() as unknown as NF;
        // A carpet has no discrete blade root. Its categorical owner is the
        // periodic source tile containing the hit, which is stable everywhere
        // except the measure-zero shared seam and remains inside the 4 m closure.
        const hitQx = qx.add(ndx.mul(tProfile)) as unknown as NF;
        const hitQz = qz.add(ndz.mul(tProfile)) as unknown as NF;
        const rootQx = hitQx.sub(profile.tileOriginX).div(profile.tileSizeX)
          .floor().add(0.5).mul(profile.tileSizeX).add(profile.tileOriginX) as unknown as NF;
        const rootQz = hitQz.sub(profile.tileOriginZ).div(profile.tileSizeZ)
          .floor().add(0.5).mul(profile.tileSizeZ).add(profile.tileOriginZ) as unknown as NF;
        const qa = rootQx.sub(phaseX) as unknown as NF;
        const qb = rootQz.sub(phaseZ) as unknown as NF;
        const rootPtx = qa.mul(cs).add(qb.mul(sn)).div(scale * coordinateScale) as unknown as NF;
        const rootPtz = qb.mul(cs).sub(qa.mul(sn)).div(scale * coordinateScale) as unknown as NF;
        const hitPtx = Ptx.add(ex.mul(dt).div(GUIDE_PITCH)) as unknown as NF;
        const hitPtz = Ptz.add(ez.mul(dt).div(GUIDE_PITCH)) as unknown as NF;
        const valid = tProfile.lessThan(5e5) as unknown as NB;
        const ownedWorldNormal = ownedHit
          ? (() => {
              const nxBase = (ownedHit.nrm.x as unknown as NF).mul(cs)
                .add((ownedHit.nrm.z as unknown as NF).mul(sn)) as unknown as NF;
              const nzBase = (ownedHit.nrm.z as unknown as NF).mul(cs)
                .sub((ownedHit.nrm.x as unknown as NF).mul(sn)) as unknown as NF;
              const nyScale = (ownedHit.nrm.y as unknown as NF)
                .mul(profile.topH).div(swardH.max(1e-4)) as unknown as NF;
              return normalize(vec3(
                nxBase.mul(profileDerivativeScale).sub(nyScale.mul(gradO.x)),
                nyScale,
                nzBase.mul(profileDerivativeScale).sub(nyScale.mul(gradO.y)),
              ) as unknown as NV3) as unknown as NV3;
            })()
          : (vec3(0, 1, 0) as unknown as NV3);
        return {
          dt: (valid as unknown as { select(a: unknown, b: unknown): NF })
            .select(dt, float(1e6)),
          // Winner-only normal reconstruction runs after profile/root election.
          nx: ownedWorldNormal.x as unknown as NF,
          ny: ownedWorldNormal.y as unknown as NF,
          nz: ownedWorldNormal.z as unknown as NF,
          cr: ownedHit ? ownedHit.color.x as unknown as NF : float(0.05) as unknown as NF,
          cg: ownedHit ? ownedHit.color.y as unknown as NF : float(0.12) as unknown as NF,
          cb: ownedHit ? ownedHit.color.z as unknown as NF : float(0.03) as unknown as NF,
          tip: profileTip,
          rootDx: rootPtx.sub(hitPtx).mul(GUIDE_PITCH) as unknown as NF,
          rootDz: rootPtz.sub(hitPtz).mul(GUIDE_PITCH) as unknown as NF,
        };
      };
      const sampleAnalyticLayer = (
        ang: number,
        scale: number,
        phaseX: number,
        phaseZ: number,
      ): {
        dt: NF; nx: NF; ny: NF; nz: NF; cr: NF; cg: NF; cb: NF;
        tip: NF; rootDx: NF; rootDz: NF;
      } => {
        const cs = Math.cos(ang);
        const sn = Math.sin(ang);
        const qx = Ptx.mul(scale).mul(cs).sub(Ptz.mul(scale).mul(sn)).add(phaseX) as unknown as NF;
        const qz = Ptx.mul(scale).mul(sn).add(Ptz.mul(scale).mul(cs)).add(phaseZ) as unknown as NF;
        const rex = ex.mul(cs).sub(ez.mul(sn)) as unknown as NF;
        const rez = ex.mul(sn).add(ez.mul(cs)) as unknown as NF;
        const azN = (atan(rez, rex) as unknown as NF).mul(1 / (Math.PI * 2)).fract() as unknown as NF;
        let isCushion: NB | null = null;
        let blockSlot = profileBaseBlock;
        // The basis checkpoint is the literal base algorithm: one uniformly
        // repeated 2D extrusion. Density tiers are a later world-control
        // extension and must not participate while the shared view transform is
        // being judged.
        let queryDens = baseExtrusionAcceptance
          ? (float(1) as unknown as NF)
          : dens as unknown as NF;
        // This entire branch is removed at TypeScript graph-build time in the
        // ordinary path. `select` alone would still make every cover candidate
        // pay the cap's elevation/metric ALU and register live ranges.
        if (GROUND_COVER_CAP_ORACLE) {
          isCushion = coverId.equal(uint(GroundCoverId.Moss)) as unknown as NB;
          const dropActual = dhdt
            .negate()
            .mul(GUIDE_PITCH)
            .div(swardH.max(1e-4).mul(eLen).mul(scale).max(1e-5))
            .clamp(MOSS_DROP_BINS[0], MOSS_DROP_BINS[MOSS_DROP_BINS.length - 1]) as unknown as NF;
          let elevationBlock: NF = float(0) as unknown as NF;
          for (let i = 1; i < MOSS_DROP_BINS.length; i++) {
            const threshold = Math.sqrt(MOSS_DROP_BINS[i - 1]! * MOSS_DROP_BINS[i]!);
            elevationBlock = (dropActual.greaterThanEqual(threshold) as unknown as {
              select(a: unknown, b: unknown): NF;
            }).select(float(i), elevationBlock);
          }
          blockSlot = (isCushion as unknown as { select(a: unknown, b: unknown): NF })
            .select(profileBaseBlock.add(elevationBlock), profileBaseBlock) as unknown as NF;
          // Oracle-only carpet fixture keeps all nested cells. It exists to
          // compare the reference intersection, not to represent production moss.
          queryDens = (isCushion as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), dens) as unknown as NF;
        }
        const hit = lutSample(qx, qz, azN, blockSlot, queryDens);
        const origin = (hit.near.x as unknown as NF).greaterThanEqual(254.5 / 255) as unknown as NB;
        // The exact d=0 sample is categorical occupancy, not a filterable depth.
        // Preserve its whole record so |OA|/cos(alpha) remains exactly zero for
        // vertical/near-vertical rays instead of exploding a filtered epsilon.
        const rec = (origin as unknown as { select(a: unknown, b: unknown): NV4 })
          .select(hit.near, hit.rec);
        // Baked normal is filterable octahedral XY in layer space. The old
        // scalar azimuth wrapped at 0/1 and linear filtering produced the
        // opposite normal at that payload seam. Oct decode uses only bounded
        // ALU and one normalization (no trig, fetch, branch loop, or binding).
        const ox = (rec.y as unknown as NF).mul(2).sub(1).toVar() as unknown as NF;
        const oy = (rec.z as unknown as NF).mul(2).sub(1).toVar() as unknown as NF;
        const oz = float(1).sub(ox.abs()).sub(oy.abs()).toVar() as unknown as NF;
        If(oz.lessThan(0), () => {
          const oldX = (ox as unknown as { toVar(): NF }).toVar() as unknown as NF;
          const sx = (oldX.greaterThanEqual(0) as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), float(-1));
          const sy = (oy.greaterThanEqual(0) as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), float(-1));
          (ox as unknown as { assign(v: unknown): void }).assign(float(1).sub(oy.abs()).mul(sx));
          (oy as unknown as { assign(v: unknown): void }).assign(float(1).sub(oldX.abs()).mul(sy));
        });
        const octN = normalize(vec3(ox, oy, oz) as unknown as NV3) as unknown as NV3;
        const tx = octN.x as unknown as NF;
        const ny = octN.y as unknown as NF;
        const tz = octN.z as unknown as NF;
        const dParam = float(1).div((rec.x as unknown as NF).max(1 / 255)).sub(1) as unknown as NF;
        const valid = dParam.lessThan(rayBake.dMaxTile * 0.94) as unknown as NB;
        const rx = tx.mul(cs).add(tz.mul(sn)) as unknown as NF;
        const rz = tz.mul(cs).sub(tx.mul(sn)) as unknown as NF;
        const qSpeed = eLen.mul(scale / GUIDE_PITCH).max(1e-5) as unknown as NF;
        let dt = dParam.div(qSpeed) as unknown as NF;
        let nx = rx;
        let nyOut = ny;
        let nz = rz;
        if (GROUND_COVER_CAP_ORACLE && isCushion) {
          // The oracle stores its gradient in (q-tile, normalized-height)
          // coordinates. Transform it back to physical space for comparison.
          const capN = normalize(vec3(
            rx.mul(scale / GUIDE_PITCH),
            ny.div(swardH.max(1e-4)),
            rz.mul(scale / GUIDE_PITCH),
          ) as unknown as NV3) as unknown as NV3;
          const hSpeed = dhdt.negate().div(swardH.max(1e-4)) as unknown as NF;
          const metricSpeed = qSpeed.mul(qSpeed).add(hSpeed.mul(hSpeed)).sqrt().max(1e-5) as unknown as NF;
          dt = (isCushion as unknown as { select(a: unknown, b: unknown): NF })
            .select(dParam.div(metricSpeed), dt) as unknown as NF;
          nx = (isCushion as unknown as { select(a: unknown, b: unknown): NF })
            .select(capN.x, rx) as unknown as NF;
          nyOut = (isCushion as unknown as { select(a: unknown, b: unknown): NF })
            .select(capN.y, ny) as unknown as NF;
          nz = (isCushion as unknown as { select(a: unknown, b: unknown): NF })
            .select(capN.z, rz) as unknown as NF;
        }
        // The categorical A payload is the canonical SUB×SUB root cell. Recover
        // the nearest repeated instance at this hit, invert this layer's fixed
        // transform, and carry its offset from the reconstructed surface base.
        // Control/type identity is sampled there below; `floor(baseB)` is only a
        // surface cell and can change across pixels of one overhanging object.
        const rootId = (hit.near.w as unknown as NF)
          .mul(GUIDE_SUB * GUIDE_SUB).floor().clamp(0, GUIDE_SUB * GUIDE_SUB - 1) as unknown as NF;
        const rootV = rootId.div(GUIDE_SUB).floor() as unknown as NF;
        const rootU = rootId.sub(rootV.mul(GUIDE_SUB)) as unknown as NF;
        const localRootX = rootU.add(0.5).div(GUIDE_SUB) as unknown as NF;
        const localRootZ = rootV.add(0.5).div(GUIDE_SUB) as unknown as NF;
        const hitQx = qx.add(rex.div(eLen).mul(dParam)) as unknown as NF;
        const hitQz = qz.add(rez.div(eLen).mul(dParam)) as unknown as NF;
        const rootQx = hitQx.sub(localRootX).add(0.5).floor().add(localRootX) as unknown as NF;
        const rootQz = hitQz.sub(localRootZ).add(0.5).floor().add(localRootZ) as unknown as NF;
        const qa = rootQx.sub(phaseX) as unknown as NF;
        const qb = rootQz.sub(phaseZ) as unknown as NF;
        const rootPtx = qa.mul(cs).add(qb.mul(sn)).div(scale) as unknown as NF;
        const rootPtz = qb.mul(cs).sub(qa.mul(sn)).div(scale) as unknown as NF;
        const hitPtx = Ptx.add(ex.mul(dt).div(GUIDE_PITCH)) as unknown as NF;
        const hitPtz = Ptz.add(ez.mul(dt).div(GUIDE_PITCH)) as unknown as NF;
        return {
          dt: (valid as unknown as { select(a: unknown, b: unknown): NF })
            .select(dt, float(1e6)),
          nx,
          ny: nyOut,
          nz,
          cr: float(0.05) as unknown as NF,
          cg: float(0.12) as unknown as NF,
          cb: float(0.03) as unknown as NF,
          // Analytic fixtures still reconstruct their normalized physical height
          // below; this sentinel is graph-dead for those TypeScript-expanded ids.
          tip: float(-1) as unknown as NF,
          rootDx: rootPtx.sub(hitPtx).mul(GUIDE_PITCH) as unknown as NF,
          rootDz: rootPtz.sub(hitPtz).mul(GUIDE_PITCH) as unknown as NF,
        };
      };
      const L0 = usesPeriodicCarrier
        ? samplePeriodicLayer(periodicProfile, 0, 1, 0, 0)
        : sampleAnalyticLayer(0, 1, 0, 0);
      const L1 = baseExtrusionAcceptance
        ? L0
        : usesPeriodicCarrier
          ? samplePeriodicLayer(
              periodicProfile,
              Math.PI * 1.618033988749895,
              1.071773462536293,
              0.371,
              0.619,
            )
          : sampleAnalyticLayer(Math.PI * 1.618033988749895, 1.071773462536293, 0.371, 0.619);
      const take0 = L0.dt.lessThanEqual(L1.dt) as unknown as NB;
      const pick = (a: NF, b: NF): NF =>
        (take0 as unknown as { select(x: unknown, y: unknown): NF }).select(a, b);
      const dtHit = pick(L0.dt, L1.dt).toVar() as unknown as NF;
      const exactOwned = usesPeriodicCarrier && ownedPeriodicProfile !== null;
      const nWx0 = usesPeriodicCarrier && !exactOwned ? (float(0) as unknown as NF) : pick(L0.nx, L1.nx).toVar() as unknown as NF;
      const nWy0 = usesPeriodicCarrier && !exactOwned ? (float(1) as unknown as NF) : pick(L0.ny, L1.ny).toVar() as unknown as NF;
      const nWz0 = usesPeriodicCarrier && !exactOwned ? (float(0) as unknown as NF) : pick(L0.nz, L1.nz).toVar() as unknown as NF;
      const ownedColorBody = exactOwned
        ? uint(pick(L0.cr, L1.cr).clamp(0, 1).mul(255).add(0.5).floor())
            .shiftLeft(uint(20))
            .bitOr(uint(pick(L0.cg, L1.cg).clamp(0, 1).mul(255).add(0.5).floor())
              .shiftLeft(uint(12)))
            .bitOr(uint(pick(L0.cb, L1.cb).clamp(0, 1).mul(255).add(0.5).floor())
              .shiftLeft(uint(4)))
            .bitOr(profileIdU.bitAnd(uint(0xf))) as unknown as NU
        : (uint(0) as unknown as NU);
      const profileTip = pick(L0.tip, L1.tip).toVar() as unknown as NF;
      const rootDx = pick(L0.rootDx, L1.rootDx).toVar() as unknown as NF;
      const rootDz = pick(L0.rootDz, L1.rootDz).toVar() as unknown as NF;
      const tHit = tE.add(dtHit).toVar() as unknown as NF;
      const hitInRange = basisValid
        .and(dtHit.lessThan(1e5) as unknown as NB)
        .and(tHit.lessThan(tMax) as unknown as NB) as unknown as NB;

      const preCandidate = hitInRange
        .and(tHit.lessThan(tBest) as unknown as NB) as unknown as NB;
      If(preCandidate, () => {

      // ---- B: recover base-space root + physical height ----------------------------
      const yH = ro.y.add(rd.y.mul(tHit)).toVar() as unknown as NF;
      const phB = phAt(tHit);
      const phBx = phB.x.toVar() as unknown as NF;
      const phBz = phB.z.toVar() as unknown as NF;
      const absPhBx = (
        streamed ? phBx.add(gfx.mul(CELL)) : phBx
      ).toVar() as unknown as NF;
      const absPhBz = (
        streamed ? phBz.add(gfz.mul(CELL)) : phBz
      ).toVar() as unknown as NF;
      const directHB = yH.sub(groundAt(vec2(absPhBx, absPhBz) as unknown as NV2))
        .toVar() as unknown as NF;
      const qBraw = (
        shellHeight !== null
          ? directHB
          : qO.add(kGround.mul(tHit.sub(tScene)))
      ).toVar() as unknown as NF;
      const qInRange = qBraw.greaterThanEqual(-0.03)
        .and(qBraw.lessThanEqual(
          (shellHeight !== null ? swardH : qTop).add(0.08),
        ) as unknown as NB) as unknown as NB;
      const hB = (
        shellHeight !== null
          ? directHB.clamp(0, swardH)
          : hFromQ(qBraw.clamp(0, qTop.max(1e-4)) as unknown as NF).clamp(0, swardH)
      ).toVar() as unknown as NF;
      const offBX = Slx.add(Sqx.mul(hB)).mul(hB) as unknown as NF;
      const offBZ = Slz.add(Sqz.mul(hB)).mul(hB) as unknown as NF;
      const baseBx = phBx.sub(offBX).toVar() as unknown as NF;
      const baseBz = phBz.sub(offBZ).toVar() as unknown as NF;
      // Ground and identity belong to P (the blade's base coordinate), not to the
      // horizontally displaced surface point x. Using g(x) made leaned blades on
      // slopes read as downhill/upside-down curtains.
      const rootBx = baseBx.add(rootDx).toVar() as unknown as NF;
      const rootBz = baseBz.add(rootDz).toVar() as unknown as NF;
      const baseRcx = relCells(baseBx, gfx).toVar() as unknown as NF;
      const baseRcz = relCells(baseBz, gfz).toVar() as unknown as NF;
      const rootRcx = relCells(rootBx, gfx).toVar() as unknown as NF;
      const rootRcz = relCells(rootBz, gfz).toVar() as unknown as NF;
      const surfaceGuide = bguide(baseRcx, baseRcz);
      const gB = (surfaceGuide.g as unknown as { toVar(): NF }).toVar() as unknown as NF;
      const surfaceValid = yH.sub(gB.add(hB)).abs().lessThanEqual(0.15) as unknown as NB;
      // flat-local validity check: O's texel plane extrapolated to B must agree
      // with the bilinear ground field — they diverge exactly where terrain
      // breaks (cliff edges, gorge lips), which is where the single-fetch frame
      // hallucinates hanging-grass curtains. Flat/gentle meadows agree to cm.
      const gOB = groundO
        .add(gradO.x.mul(baseBx.sub(phOx)))
        .add(gradO.y.mul(baseBz.sub(phOz))) as unknown as NF;
      const planeValid = gB.sub(gOB).abs().lessThanEqual(0.35) as unknown as NB;
      const surfaceCandidate = shellHeight !== null
        ? qInRange
        : qInRange.and(surfaceValid).and(planeValid) as unknown as NB;
      If(surfaceCandidate, () => {
      const rootGuide = bguide(rootRcx, rootRcz);
      // world fine ROOT cell under B — election identity is base-space stable
      const wcx = (streamed ? rootBx.div(CELL).floor().add(gfx) : rootBx.div(CELL).floor())
        .toVar() as unknown as NF;
      const wcz = (streamed ? rootBz.div(CELL).floor().add(gfz) : rootBz.div(CELL).floor())
        .toVar() as unknown as NF;
      const rootMix = rootGuide.mixWord;
      const rootEnv = rootGuide.envWord;
      const rootTypeA = rootMix.bitAnd(uint(0xff));
      const rootTypeB = rootMix.shiftRight(uint(8)).bitAnd(uint(0xff));
      const rootClumpLo = rootEnv.bitAnd(uint(0xff));
      const rootClumpHi = rootEnv.shiftRight(uint(8)).bitAnd(uint(0xff));
      const rootProfileA = rootEnv.shiftRight(uint(16)).bitAnd(uint(0xff));
      const rootProfileB = rootEnv.shiftRight(uint(24)).bitAnd(uint(0xff));
      const rootGuv = vec2(
        rootRcx.div(GUIDE_SUB * GUIDE_RES),
        rootRcz.div(GUIDE_SUB * GUIDE_RES),
      ).clamp(0, 1) as unknown as NV2;
      const rootControl = (texture(guideFieldT3, rootGuv, 0) as unknown as {
        toVar(): NV4;
      }).toVar();
      const rootDensity = (rootControl.x as unknown as NF).clamp(0, 1) as unknown as NF;
      const rootBlend = field.hasGroundCover
        ? (rootControl.y as unknown as NF)
        : (float(0) as unknown as NF);
      const rootPick = cellHash(
        vec2(
          wcx.add(toF(rootClumpLo)),
          wcz.add(toF(rootClumpHi)),
        ) as unknown as NV2,
        SALT ^ 0x6c31,
      ) as unknown as NF;
      const rootCover = (rootPick.lessThan(rootBlend) as unknown as { select(a: unknown, b: unknown): NU })
        .select(rootTypeB, rootTypeA)
        .bitAnd(uint(GROUND_COVER_ID_MASK));
      const rootProfile = (rootPick.lessThan(rootBlend) as unknown as { select(a: unknown, b: unknown): NU })
        .select(rootProfileB, rootProfileA);
      const rootMatches = GRASS_PROFILE_OVERRIDE === null
        ? rootCover.equal(coverId)
            .and(rootProfile.equal(profileIdU) as unknown as NB) as unknown as NB
        : rootProfile.notEqual(uint(0xff))
            .and(rootDensity.greaterThanEqual(0.02) as unknown as NB) as unknown as NB;
      const sxs = wcx.sub(wcx.div(GRID).floor().mul(GRID));
      const sys = wcz.sub(wcz.div(GRID).floor().mul(GRID));
      If(rootMatches, () => {
        (tBest as unknown as { assign(v: unknown): void }).assign(tHit);
        (bodyBest as unknown as { assign(v: unknown): void }).assign(
          exactOwned
            ? ownedColorBody
            : field.hasGroundCoverClosure
            ? uint(sys.mul(GRID).add(sxs))
                .bitAnd(uint(0x3fffff))
                .shiftLeft(uint(8))
                .bitOr(profileIdU)
            : uint(sys.mul(GRID).add(sxs)).shiftLeft(uint(6)).bitOr(coverId),
        );
        if (usesPeriodicCarrier) {
          if (bestProfileId) {
            (bestProfileId as unknown as { assign(v: unknown): void }).assign(profileIdU);
          }
          (bestAntiLayer as unknown as { assign(v: unknown): void }).assign(
            (take0 as unknown as { select(a: unknown, b: unknown): NU })
              .select(uint(0), uint(1)),
          );
          if (exactOwned) {
            (nrmV as unknown as { assign(v: unknown): void }).assign(vec3(nWx0, nWy0, nWz0));
          }
        } else {
          // Keep the selected geometry record intact. A post-hit per-cell normal
          // twist changed shading without changing the surface and visibly
          // reintroduced a square cell signature.
          (nrmV as unknown as { assign(v: unknown): void }).assign(vec3(nWx0, nWy0, nWz0));
        }
        (tParV as unknown as { assign(v: unknown): void }).assign(
          usesPeriodicCarrier ? profileTip : hB.div(swardH.max(0.05)).clamp(0, 1),
        );
      });
      });
      });
      };

      // Fixed closure: TypeScript expands guarded candidates; WGSL contains no
      // loop. Cooked v2 masks normally execute two nearby profiles. The legacy
      // no-closure path retains its bounded six analytic fixtures.
      if (GRASS_PROFILE_OVERRIDE !== null) {
        // Acceptance graph contains exactly the selected profile; no dormant
        // eleven-way closure guards or analytic fallbacks survive generation.
        considerProfile(GRASS_PROFILE_OVERRIDE);
      } else {
        const candidateCount = field.hasGroundCoverClosure ? GROUND_COVER_PROFILE_COUNT : 6;
        for (let profileId = 0; profileId < candidateCount; profileId++) {
          If(candidateMask.bitAnd(uint(1 << profileId)).notEqual(uint(0)), () => {
            considerProfile(profileId);
          });
        }
      }

      If(tBest.lessThan(1e8), () => {
        const reconstructPeriodicNormal = (): void => {
        if (periodicSamplers && opts.periodicProfiles.length > 0) {
          // Rebuild the winning oblique frame once, after exact depth/root
          // election. Carrying q/nd/frame scalars through every guarded profile
          // would lengthen live ranges and reduce occupancy. The fixed selects
          // below are metadata selection, not a runtime loop or ray march.
          let winningProfileId: NU;
          let profileMeta: NV4;
          let tileSizeZ: NF;
          let bestCoverId: NU;
          if (isolatedPeriodicProfile) {
            const profile = opts.periodicProfiles[0]!;
            winningProfileId = uint(profile.profileId) as unknown as NU;
            profileMeta = vec4(
              profile.topH,
              profile.tileOriginX,
              profile.tileOriginZ,
              profile.tileSizeX,
            ) as unknown as NV4;
            tileSizeZ = float(profile.tileSizeZ) as unknown as NF;
            bestCoverId = uint(
              GROUND_COVER_PROFILE_FUNCTIONAL_IDS[profile.profileId]!,
            ) as unknown as NU;
          } else {
            const orderedProfiles = Array.from(
              { length: GROUND_COVER_PROFILE_COUNT },
              (_, profileId) => periodicProfilesById.get(profileId),
            );
            if (orderedProfiles.some((profile) => !profile)) {
              throw new Error('winner normal reconstruction requires the canonical profile array');
            }
            const profiles = orderedProfiles as LoadedPeriodicProfile[];
            winningProfileId = field.hasGroundCoverClosure
              ? bodyBest.bitAnd(uint(0xff)) as unknown as NU
              : bestProfileId!;
            profileMeta = vec4(
              profiles[GROUND_COVER_PROFILE_COUNT - 1]!.topH,
              profiles[GROUND_COVER_PROFILE_COUNT - 1]!.tileOriginX,
              profiles[GROUND_COVER_PROFILE_COUNT - 1]!.tileOriginZ,
              profiles[GROUND_COVER_PROFILE_COUNT - 1]!.tileSizeX,
            ) as unknown as NV4;
            tileSizeZ = float(
              profiles[GROUND_COVER_PROFILE_COUNT - 1]!.tileSizeZ,
            ) as unknown as NF;
            bestCoverId = uint(
              GROUND_COVER_PROFILE_FUNCTIONAL_IDS[GROUND_COVER_PROFILE_COUNT - 1]!,
            ) as unknown as NU;
            for (let profileId = GROUND_COVER_PROFILE_COUNT - 2; profileId >= 0; profileId--) {
              const isProfile = winningProfileId.equal(uint(profileId));
              const profile = profiles[profileId]!;
              profileMeta = (isProfile as unknown as { select(a: unknown, b: unknown): NV4 })
                .select(
                  vec4(
                    profile.topH,
                    profile.tileOriginX,
                    profile.tileOriginZ,
                    profile.tileSizeX,
                  ),
                  profileMeta,
                );
              tileSizeZ = (isProfile as unknown as { select(a: unknown, b: unknown): NF })
                .select(float(profile.tileSizeZ), tileSizeZ);
              bestCoverId = (isProfile as unknown as { select(a: unknown, b: unknown): NU })
                .select(uint(GROUND_COVER_PROFILE_FUNCTIONAL_IDS[profileId]!), bestCoverId);
            }
          }
          const topH = profileMeta.x as unknown as NF;
          const params = coverParams(bestCoverId).toVar() as unknown as NV4;
          const bestSwardH = (
            isolatedPeriodicProfile
              ? topH
              : (params.x as unknown as NF)
                  .mul(field.hasGroundCover ? vigor.mul(0.55).add(0.65) : float(1))
                  .mul(rag.mul(params.y).add(float(1).sub((params.y as unknown as NF).mul(0.5))))
                  .clamp(0, 1.1)
          ).toVar() as unknown as NF;
          const bestDeformK = isolatedPeriodicProfile
            ? (float(0) as unknown as NF)
            : params.z as unknown as NF;
          const bestSlx = (f1.x as unknown as NF).mul(bestDeformK).toVar() as unknown as NF;
          const bestSlz = (f1.y as unknown as NF).mul(bestDeformK).toVar() as unknown as NF;
          const bestSqx = (f1.z as unknown as NF).add(f2.z).mul(bestDeformK).toVar() as unknown as NF;
          const bestSqz = (f1.w as unknown as NF).add(f2.w).mul(bestDeformK).toVar() as unknown as NF;
          const bestBasisC = float(1)
            .sub(gradO.x.mul(bestSlx).add(gradO.y.mul(bestSlz)))
            .toVar() as unknown as NF;
          const bestBasisB = gradO.x.mul(bestSqx).add(gradO.y.mul(bestSqz)).toVar() as unknown as NF;
          const bestQAt = (h: NF): NF =>
            bestBasisC.mul(h).sub(bestBasisB.mul(h).mul(h)) as unknown as NF;
          const bestKGround = rd.y
            .sub(gradO.x.mul(rd.x))
            .sub(gradO.y.mul(rd.z))
            .min(-1e-3)
            .toVar() as unknown as NF;
          const bestHO = (
            shellHeight !== null
              ? bestSwardH
              : pOy.sub(groundO)
          ).toVar() as unknown as NF;
          const bestQO = (
            bestHO.lessThanEqual(bestSwardH) as unknown as { select(a: unknown, b: unknown): NF }
          ).select(bestQAt(bestHO), bestHO).toVar() as unknown as NF;
          const bestQTop = bestQAt(bestSwardH).toVar() as unknown as NF;
          const bestDtEUnbounded = bestQTop.sub(bestQO).div(bestKGround) as unknown as NF;
          const bestDtE = (
            shellHeight !== null
              ? float(0)
              : isolatedPeriodicProfile
              ? bestDtEUnbounded
              : bestDtEUnbounded.clamp(
                  float(GUIDE_PITCH * 3).div(dirL).negate(),
                  float(GUIDE_PITCH * 3).div(dirL),
                )
          ).toVar() as unknown as NF;
          const bestTE = tScene.add(bestDtE).max(0.05).toVar() as unknown as NF;
          const bestPhE = phAt(bestTE);
          const bestQE = bestQO.add(bestKGround.mul(bestTE.sub(tScene)))
            .clamp(0, bestQTop.max(1e-4)) as unknown as NF;
          const bestDisc = bestBasisC.mul(bestBasisC)
            .sub(bestBasisB.mul(bestQE).mul(4)).max(1e-5) as unknown as NF;
          const bestHgt = bestQE.mul(2)
            .div(bestBasisC.add(bestDisc.sqrt()).max(1e-4))
            .clamp(0, bestSwardH)
            .toVar() as unknown as NF;
          const bestOffX = bestSlx.add(bestSqx.mul(bestHgt)).mul(bestHgt) as unknown as NF;
          const bestOffZ = bestSlz.add(bestSqz.mul(bestHgt)).mul(bestHgt) as unknown as NF;
          const bestWtx = (
            streamed
              ? bestPhE.x.div(GUIDE_PITCH).add(uOTx as unknown as NF)
              : bestPhE.x.div(GUIDE_PITCH)
          ).toVar() as unknown as NF;
          const bestWtz = (
            streamed
              ? bestPhE.z.div(GUIDE_PITCH).add(uOTz as unknown as NF)
              : bestPhE.z.div(GUIDE_PITCH)
          ).toVar() as unknown as NF;
          const bestPtx = bestWtx.sub(bestOffX.div(GUIDE_PITCH)).toVar() as unknown as NF;
          const bestPtz = bestWtz.sub(bestOffZ.div(GUIDE_PITCH)).toVar() as unknown as NF;
          const bestTanX = bestSlx.add(bestSqx.mul(bestHgt).mul(2)) as unknown as NF;
          const bestTanZ = bestSlz.add(bestSqz.mul(bestHgt).mul(2)) as unknown as NF;
          const bestBasisDen = bestBasisC.sub(bestBasisB.mul(bestHgt).mul(2)).toVar() as unknown as NF;
          const bestDhdt = bestKGround.div(bestBasisDen.max(0.05)).toVar() as unknown as NF;
          const bestEx = rd.x.sub(bestTanX.mul(bestDhdt)).toVar() as unknown as NF;
          const bestEz = rd.z.sub(bestTanZ.mul(bestDhdt)).toVar() as unknown as NF;

          const useLayer0 = bestAntiLayer.equal(uint(0));
          const layer1Angle = Math.PI * 1.618033988749895;
          const layer1Scale = 1.071773462536293;
          const layerCs = (useLayer0 as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), float(Math.cos(layer1Angle)))
            .toVar() as unknown as NF;
          const layerSn = (useLayer0 as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(0), float(Math.sin(layer1Angle)))
            .toVar() as unknown as NF;
          const layerScale = (useLayer0 as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(1), float(layer1Scale))
            .toVar() as unknown as NF;
          const layerScaleOverPitch = (
            useLayer0 as unknown as { select(a: unknown, b: unknown): NF }
          ).select(float(1 / GUIDE_PITCH), float(layer1Scale / GUIDE_PITCH))
            .toVar() as unknown as NF;
          const phaseX = (useLayer0 as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(0), float(0.371)) as unknown as NF;
          const phaseZ = (useLayer0 as unknown as { select(a: unknown, b: unknown): NF })
            .select(float(0), float(0.619)) as unknown as NF;
          const coordinateScale = isolatedPeriodicProfile ? GUIDE_PITCH : 1;
          const queryX = bestPtx.mul(coordinateScale).mul(layerScale).mul(layerCs)
            .sub(bestPtz.mul(coordinateScale).mul(layerScale).mul(layerSn)).add(phaseX) as unknown as NF;
          const queryZ = bestPtx.mul(coordinateScale).mul(layerScale).mul(layerSn)
            .add(bestPtz.mul(coordinateScale).mul(layerScale).mul(layerCs)).add(phaseZ) as unknown as NF;
          const derivativeScale = isolatedPeriodicProfile ? layerScale : layerScaleOverPitch;
          const dpx = bestEx.mul(derivativeScale).mul(layerCs)
            .sub(bestEz.mul(derivativeScale).mul(layerSn)) as unknown as NF;
          const dpz = bestEx.mul(derivativeScale).mul(layerSn)
            .add(bestEz.mul(derivativeScale).mul(layerCs)) as unknown as NF;
          const dpy = bestDhdt.mul(topH).div(bestSwardH.max(1e-4)) as unknown as NF;
          const metricSpeed = vec3(dpx, dpy, dpz).length().max(1e-5).toVar() as unknown as NF;
          const nProfile = periodicSamplers.normal(
            vec2(queryX, queryZ),
            vec3(dpx.div(metricSpeed), dpy.div(metricSpeed), dpz.div(metricSpeed)),
            vec4(profileMeta.y, profileMeta.z, profileMeta.w, tileSizeZ),
            winningProfileId,
          ) as unknown as NV3;
          const nxBase = (nProfile.x as unknown as NF).mul(layerCs)
            .add((nProfile.z as unknown as NF).mul(layerSn)) as unknown as NF;
          const nzBase = (nProfile.z as unknown as NF).mul(layerCs)
            .sub((nProfile.x as unknown as NF).mul(layerSn)) as unknown as NF;
          const nyScale = (nProfile.y as unknown as NF)
            .mul(topH).div(bestSwardH.max(1e-4)) as unknown as NF;
          (nrmV as unknown as { assign(v: unknown): void }).assign(normalize(vec3(
            nxBase.mul(derivativeScale).sub(nyScale.mul(gradO.x)),
            nyScale,
            nzBase.mul(derivativeScale).sub(nyScale.mul(gradO.y)),
          ) as unknown as NV3));
          if (isolatedPeriodicProfile && periodicSamplers.color) {
            const authoredColor = periodicSamplers.color(
              vec2(queryX, queryZ),
              vec3(dpx.div(metricSpeed), dpy.div(metricSpeed), dpz.div(metricSpeed)),
              vec4(profileMeta.y, profileMeta.z, profileMeta.w, tileSizeZ),
            ) as unknown as NV3;
            const authoredBody = uint((authoredColor.x as unknown as NF)
              .clamp(0, 1).mul(255).add(0.5).floor())
              .shiftLeft(uint(20))
              .bitOr(uint((authoredColor.y as unknown as NF)
                .clamp(0, 1).mul(255).add(0.5).floor()).shiftLeft(uint(12)))
              .bitOr(uint((authoredColor.z as unknown as NF)
                .clamp(0, 1).mul(255).add(0.5).floor()).shiftLeft(uint(4)))
              .bitOr(winningProfileId.bitAnd(uint(0xf))) as unknown as NU;
            (bodyBest as unknown as { assign(v: unknown): void }).assign(authoredBody);
          }
        }
        };
        const hit = ro.add(rd.mul(tBest));
        const clip = cam.vp.mul(vec4(hit, 1));
        const cz = clip.z.div(clip.w.max(NEAR_EPS));
        If(cz.greaterThanEqual(0).and(cz.lessThanEqual(1)), () => {
          if (!ownedPeriodicProfile && !baseExtrusionAcceptance) reconstructPeriodicNormal();
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
    shellHeight,
    resolveRay,
    setEnabled(v: boolean): void {
      onCpu = v;
      uOn.value = v ? 1 : 0;
    },
    enabled: () => onCpu,
    readCounts,
  };
}
