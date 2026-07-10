/**
 * TerrainField — the streamed terrain field set (SPEC-STREAMING-WORLD §3, S5).
 * The scene handle threaded where the one-shot Heightfield went: a camera-window
 * plane pyramid whose texels arrive EXCLUSIVELY as StreamBrain fill packets
 * through the main-thread mailbox (the universal path both the generated world
 * and Estonia ride) — this class allocates the planes from a placement PLAN
 * (PlaneFill.planField — the same pure math the brain plans against) and
 * applies fills/origin commits; it never fetches.
 *
 *  - HEIGHT plane pyramid — r32float, level k texel = base·4^k, res per level
 *    by the cover-or-window rule. A level whose window covers the layer span is
 *    PINNED (plan.wraps=false — every generated level): its samplers compile
 *    the exact pre-S5 form, bit-identical shaders. A wrapping level (Estonia's
 *    near rungs) samples toroidally via its phase uniform and scrolls by brain
 *    packets: fills land first, the origin commit re-points uOrigin/uPhase
 *    after (promote-after-fill is packet order — F-8).
 *  - BIOME/CANOPY + SURFACE-FIELDS planes — rgba8 (filtered), same rule.
 *  - WATER planes — waterY (r32float) + ×8 min-reduced far level.
 *  - CPU mirrors — the DataTexture backing stores double as the mirrors (zero
 *    extra RAM) and stay coherent under partial fills, so the ?profile=1
 *    device swap re-uploads correct state and heightAt/waterAt serve walk
 *    probe, spawn and bookmarks at any moment.
 *
 * Lattice honesty (A11) lives in PlaneFill.placeLevel: plane texel p IS source
 * lattice sample n0+p — L0 fills are bit-identical to the source raster.
 */

import { DataTexture, FloatType, LinearFilter, NearestFilter, RGBAFormat, RedFormat, UnsignedByteType, Vector2 } from 'three';
import { If, clamp, float, floor, fract, mix, texture, uniform, vec2, vec3, vec4 } from 'three/tsl';
import { texLoadR } from '../Tsl';
import type { NB, NF, NU, NV2, NV4 } from '../../gpu/TSLTypes';
import type { CoverageBox, FieldPlan, PlanePlan } from './PlaneFill';
import type { PlaneKind } from './StreamProtocol';

export type { CoverageBox } from './PlaneFill';

/** full 5-level Estonia set ≈ 84 (height) + 17 (biome) + ~40 (fields+water) MB;
 *  the generated full-coverage set ≈ 130 MB — anything past this is a leak, not
 *  a config (throw-loud VRAM law). */
const VRAM_CEILING_MB = 160;

export interface FieldLevel {
  /** data LOD — texel = layer base texel · lodStep^lod */
  lod: number;
  res: number;
  texel: number;
  /** world coord of texel (0,0)'s SAMPLE point (texel-centered source lattice) */
  originX: number;
  originZ: number;
  /** origin as a uniform — scroll commits re-point it without shader rebuilds */
  uOrigin: { value: Vector2 };
  /** toroidal phase (n0 mod res) — only wrapping levels compile reads of it */
  uPhase: { value: Vector2 };
  /** frozen at plan time: window < coverage ⇒ the level scrolls toroidally.
   *  Pinned levels (all generated ones) compile the exact pre-S5 samplers. */
  wraps: boolean;
  tex: DataTexture;
}

interface HeightLevel extends FieldLevel {
  data: Float32Array;
}

export class TerrainField {
  /** finest → coarsest; index = the `level` arg of fieldHeight */
  readonly heightLevels: readonly HeightLevel[];
  /** rgba8 [classId, vegDensity, canopyHeight, cover] */
  readonly biomeLevels: readonly FieldLevel[];
  /** rgba8 [moisture, flow, snow, rockExposure] */
  readonly fieldsLevels: readonly FieldLevel[];
  /** waterY at ~sim-res texel (r32float) — null when the source has no water */
  readonly water: HeightLevel | null;
  /** ×8 min-reduced far waterY (conservative: channels vanish, lakes survive) */
  readonly waterFar: HeightLevel | null;
  readonly coverageBox: CoverageBox;

  private constructor(
    heightLevels: HeightLevel[],
    biomeLevels: FieldLevel[],
    fieldsLevels: FieldLevel[],
    water: HeightLevel | null,
    waterFar: HeightLevel | null,
    coverageBox: CoverageBox,
  ) {
    if (heightLevels.length === 0) throw new Error('TerrainField: needs at least one height level');
    this.heightLevels = heightLevels;
    this.biomeLevels = biomeLevels;
    this.fieldsLevels = fieldsLevels;
    this.water = water;
    this.waterFar = waterFar;
    this.coverageBox = coverageBox;
    const mb = this.vramBytes() / 2 ** 20;
    // eslint-disable-next-line no-console
    console.log(
      `[laas] terrain field: height [${heightLevels.map((l) => `${l.res}²@${l.texel}m${l.wraps ? '~' : ''}`).join(' ')}] r32f + ` +
        `biome [${biomeLevels.map((l) => `${l.res}²`).join(' ')}] + fields [${fieldsLevels.map((l) => `${l.res}²`).join(' ')}] rgba8 + ` +
        `water ${water ? `${water.res}² r32f (+far ${waterFar?.res ?? 0}²)` : 'none'} = ` +
        `${mb.toFixed(1)} MB VRAM (CPU mirrors share the backing; ~ = camera-window level)`,
    );
    if (mb > VRAM_CEILING_MB) {
      throw new Error(`TerrainField: ${mb.toFixed(1)} MB exceeds the ${VRAM_CEILING_MB} MB ceiling`);
    }
  }

  /** Allocate every plane from the placement plan (PlaneFill.planField) —
   *  texels arrive as StreamBrain fill packets (the S5 streaming path). */
  static fromPlan(plan: FieldPlan): TerrainField {
    const heightLevels = plan.height.map((p) => makeHeightLevel(`terrainFieldHeightL${p.lod}`, p));
    const biomeLevels = plan.biome.map((p) => makeU8Level(`terrainFieldBiomeL${p.lod}`, p));
    const fieldsLevels = plan.fields.map((p) => makeU8Level(`terrainFieldFieldsL${p.lod}`, p));
    const water = plan.water ? makeHeightLevel('terrainFieldWaterY', plan.water) : null;
    const waterFar = plan.waterFar ? makeHeightLevel('terrainFieldWaterYFar', plan.waterFar) : null;
    return new TerrainField(heightLevels, biomeLevels, fieldsLevels, water, waterFar, plan.coverageBox);
  }

  /** One-level field over a flat/explicit height array — forest/gallery-class
   *  scenes whose world is not source-streamed (F11: the fake-hf hack's grave). */
  static singleLevel(opts: {
    res: number;
    texel: number;
    /** world min corner of the covered square */
    worldMinX: number;
    worldMinZ: number;
    /** res² heights; omit for a flat-0 world */
    heights?: Float32Array;
  }): TerrainField {
    const data = opts.heights ?? new Float32Array(opts.res * opts.res);
    if (data.length !== opts.res * opts.res) throw new Error('TerrainField.singleLevel: heights length != res²');
    const plan: PlanePlan = {
      lod: 0,
      res: opts.res,
      wraps: false,
      originX: opts.worldMinX + opts.texel * 0.5,
      originZ: opts.worldMinZ + opts.texel * 0.5,
      texel: opts.texel,
      stride: 1,
      n0x: 0,
      n0z: 0,
      nMinX: 0,
      nMaxX: opts.res - 1,
      nMinZ: 0,
      nMaxZ: opts.res - 1,
    };
    const lvl = makeHeightLevel('terrainFieldHeightL0', plan, data);
    return new TerrainField([lvl], [], [], null, null, {
      minX: opts.worldMinX,
      minZ: opts.worldMinZ,
      maxX: opts.worldMinX + opts.res * opts.texel,
      maxZ: opts.worldMinZ + opts.res * opts.texel,
    });
  }

  // ---- stream mailbox application (the ONLY mutation surface) ----------------------

  /** the level record a fill/origin packet targets. */
  levelFor(plane: PlaneKind, level: number): FieldLevel {
    const lvl =
      plane === 'height'
        ? this.heightLevels[level]
        : plane === 'biome'
          ? this.biomeLevels[level]
          : plane === 'fields'
            ? this.fieldsLevels[level]
            : plane === 'water'
              ? this.water
              : this.waterFar;
    if (!lvl) throw new Error(`TerrainField: no ${plane} level ${level}`);
    return lvl;
  }

  /** write a physical-rect fill into the CPU backing (the mirror). The GPU
   *  upload is the mailbox's job (writeTexture, or a full needsUpdate flip at
   *  boot) — backing-first keeps the ?profile device swap coherent. */
  applyFill(plane: PlaneKind, level: number, x: number, y: number, w: number, h: number, data: Float32Array | Uint8Array): void {
    const lvl = this.levelFor(plane, level);
    const backing = (lvl.tex.image as { data: Float32Array | Uint8Array }).data;
    const ch = backing instanceof Float32Array ? 1 : 4;
    if (data.length !== w * h * ch) throw new Error(`TerrainField: fill ${plane}L${level} ${w}×${h} data length mismatch`);
    for (let row = 0; row < h; row++) {
      backing.set(data.subarray(row * w * ch, (row + 1) * w * ch), ((y + row) * lvl.res + x) * ch);
    }
  }

  /** re-point a wrapping level after its fills landed (F-8 packet order). */
  commitOrigin(plane: PlaneKind, level: number, originX: number, originZ: number, phaseX: number, phaseY: number): void {
    const lvl = this.levelFor(plane, level);
    if (!lvl.wraps) throw new Error(`TerrainField: origin commit on pinned plane ${plane}L${level}`);
    lvl.originX = originX;
    lvl.originZ = originZ;
    lvl.uOrigin.value.set(originX, originZ);
    lvl.uPhase.value.set(phaseX, phaseY);
  }

  /** flip every plane's full backing to the GPU (boot: after the unbudgeted
   *  mailbox drain filled the mirrors). */
  markAllDirty(): void {
    for (const lvl of this.allLevels()) lvl.tex.needsUpdate = true;
  }

  private allLevels(): FieldLevel[] {
    return [...this.heightLevels, ...this.biomeLevels, ...this.fieldsLevels, ...(this.water ? [this.water] : []), ...(this.waterFar ? [this.waterFar] : [])];
  }

  // ---- CPU sampling (walk probe, spawn, bookmarks) --------------------------------

  /** bilinear height (m) — finest level whose window contains the point */
  heightAt(x: number, z: number): number {
    const L = this.heightLevels;
    for (let i = 0; i < L.length - 1; i++) {
      const lvl = L[i] as HeightLevel;
      const gx = (x - lvl.originX) / lvl.texel;
      const gz = (z - lvl.originZ) / lvl.texel;
      // 1-texel margin: the plane rim is clamp-extended fill (or the scroll
      // rim), the coarser level holds the real slope there
      if (gx >= 1 && gx <= lvl.res - 2 && gz >= 1 && gz <= lvl.res - 2) {
        return bilerpCpu(lvl, gx, gz);
      }
    }
    const lvl = L[L.length - 1] as HeightLevel;
    return bilerpCpu(lvl, (x - lvl.originX) / lvl.texel, (z - lvl.originZ) / lvl.texel);
  }

  /** bilinear waterY (m); dry cells sit well below the bed (sentinel/bed−2), so
   *  max(ground, waterAt + ε) stays a safe camera floor. −1e4 without water. */
  waterAt(x: number, z: number): number {
    const w = this.water;
    if (!w) return -1e4;
    return bilerpCpu(w, (x - w.originX) / w.texel, (z - w.originZ) / w.texel);
  }

  // ---- TSL sampling ----------------------------------------------------------------

  /** bilinear height at a KNOWN level — the hoisted form for draw classes whose
   *  band is known per draw/tile (hot S3b consumers hoist the level per tile) */
  fieldHeight(wxz: NV2, level = 0): NF {
    return planeBilerp(this.heightLevels[level] as HeightLevel, wxz);
  }

  /** nearest-texel height at a known level (cost-insensitive probe taps) */
  fieldHeightNearest(wxz: NV2, level = 0): NF {
    return planeNearest(this.heightLevels[level] as HeightLevel, wxz);
  }

  /** bilinear height, finest resident level selected per sample */
  fieldHeightFinest(wxz: NV2): NF {
    return this.finestSelect(wxz, planeBilerp);
  }

  /** nearest-texel height, finest resident level selected per sample */
  fieldHeightFinestNearest(wxz: NV2): NF {
    return this.finestSelect(wxz, planeNearest);
  }

  /** bilinear waterY (near consumers: caustics, water gates) */
  fieldWaterY(wxz: NV2): NF {
    const w = this.water;
    if (!w) throw new Error('TerrainField: no water plane');
    return planeBilerp(w, wxz);
  }

  /** bilinear waterY from the min-reduced far level (distant clipmap rings) */
  fieldWaterYFar(wxz: NV2): NF {
    const w = this.waterFar;
    if (!w) throw new Error('TerrainField: no far water plane');
    return planeBilerp(w, wxz);
  }

  // ---- HOT samplers (S3b) — Fn-stack only (If/ElseIf level chains: exactly ONE
  // arm executes, and the containment conditions are near-uniform per cluster).
  // The cold material-build-time samplers above use select() chains instead.

  /** height for hot per-vertex paths: finest containing level via a BRANCH
   *  chain. Level 0 reads the exact texel — terrain verts sit ON the finest
   *  lattice, so this is bit-identical to the retired global heightTex tap —
   *  and coarser levels bilerp (their lattice is coarser than the verts'). */
  fieldHeightHot(wxz: NV2): NF {
    const out = float(0).toVar();
    hotLevelChain(this.heightLevels, wxz, (lvl, finest) => {
      out.assign(finest ? planeNearest(lvl, wxz) : planeBilerp(lvl, wxz));
    });
    return out as unknown as NF;
  }

  /** central-difference slope (rise/run) from the height planes — the in-shader
   *  replacement for the retired normalTex.w: the SAME ±1-texel stencil the old
   *  bake ran (Heightfield derived-maps kernel), evaluated at the nearest texel
   *  of the finest containing level. `level` HOISTS the select for consumers
   *  whose window containment is guaranteed (grass guide ring ≪ the L0 window). */
  fieldSlope(wxz: NV2, level?: number): NF {
    if (level !== undefined) return slope4(this.heightLevels[level] as HeightLevel, wxz);
    const out = float(0).toVar();
    hotLevelChain(this.heightLevels, wxz, (lvl) => out.assign(slope4(lvl, wxz)));
    return out as unknown as NF;
  }

  /** central-difference world normal (xyz) + slope (w) — the retired normalTex's
   *  EXACT bake stencil (n = normalize(hl−hr, 2·texel, hd−hu); slope = |∇h|/2texel)
   *  evaluated in-shader at the finest containing level (S3b resolve). */
  fieldNormalSlope(wxz: NV2): NV4 {
    const out = vec4(0, 1, 0, 0).toVar();
    hotLevelChain(this.heightLevels, wxz, (lvl) => out.assign(normalSlope4(lvl, wxz)));
    return out as unknown as NV4;
  }

  /** surface-fields sample [moisture, flowStrength, snow, rockExposure] — one
   *  hardware-filtered rgba8 tap at the finest containing level. vec4(0) when
   *  the source has no fields layer (forest/gallery single-level fields). */
  fieldsAt(wxz: NV2): NV4 {
    if (this.fieldsLevels.length === 0) return vec4(0) as unknown as NV4;
    const out = vec4(0).toVar();
    hotLevelChain(this.fieldsLevels, wxz, (lvl) => out.assign(planeLinear(lvl, wxz)));
    return out as unknown as NV4;
  }

  /** biome/canopy sample [classId (raw id byte ⇒ ×255 to decode), vegDensity,
   *  canopyHeight, cover] — one filtered rgba8 tap, finest containing level.
   *  vec4(0) when the source has no biome layer. */
  biomeAt(wxz: NV2): NV4 {
    if (this.biomeLevels.length === 0) return vec4(0) as unknown as NV4;
    const out = vec4(0).toVar();
    hotLevelChain(this.biomeLevels, wxz, (lvl) => out.assign(planeLinear(lvl, wxz)));
    return out as unknown as NV4;
  }

  /** nearest-texel waterY — hot gates (raster riverDepth, grass water gate).
   *  −1e4 (the dry sentinel, far below any bed) when the source has no water. */
  fieldWaterYNearest(wxz: NV2): NF {
    const w = this.water;
    if (!w) return float(-1e4) as unknown as NF;
    return planeNearest(w, wxz);
  }

  /** a height level's plane texture — raster/debug paths that bind the texture
   *  directly (their in-shader reads migrate at S3b) */
  heightPlane(level = 0): DataTexture {
    return (this.heightLevels[level] as HeightLevel).tex;
  }

  vramBytes(): number {
    let b = 0;
    for (const l of this.allLevels()) b += l.res * l.res * 4;
    return b;
  }

  private finestSelect(wxz: NV2, sample: (lvl: HeightLevel, wxz: NV2) => NF): NF {
    const L = this.heightLevels;
    let h = sample(L[L.length - 1] as HeightLevel, wxz);
    for (let i = L.length - 2; i >= 0; i--) {
      const lvl = L[i] as HeightLevel;
      h = insideLevel(lvl, wxz).select(sample(lvl, wxz), h) as NF;
    }
    return h;
  }
}

// ---- plane sampling helpers (TSL) ----------------------------------------------------
// planeBilerp / planeNearest / insideLevel are exported for the S3b hot-shader
// fetch (NaniteFetch composes its own level chain over field.heightLevels).
// A PINNED level (wraps=false — every generated level) compiles the exact
// pre-S5 addressing; a wrapping level adds the phase-mod on its texel indices.

/** world → continuous sample-grid coords (integer = exact source sample),
 *  window-relative on wrapping levels. */
export function gridCoords(lvl: FieldLevel, wxz: NV2): NV2 {
  return wxz.sub(vec2(lvl.uOrigin as unknown as NV2)).div(lvl.texel);
}

/** logical texel index → physical (toroidal) index on wrapping levels. `i` is
 *  a float in [0, res-1]; the phase add stays < 2·res so one mod suffices. */
function physF(lvl: FieldLevel, i: NF, axis: 0 | 1): NF {
  if (!lvl.wraps) return i;
  const p = vec2(lvl.uPhase as unknown as NV2);
  const ph = (axis === 0 ? p.x : p.y) as unknown as NF;
  return i.add(ph).mod(lvl.res) as unknown as NF;
}

function texelU(lvl: FieldLevel, ix: NF, iy: NF): { x: NU; y: NU } {
  return {
    x: floor(physF(lvl, ix, 0)).toUint() as NU,
    y: floor(physF(lvl, iy, 1)).toUint() as NU,
  };
}

/** finest-containing-level branch chain: runs `arm` for exactly one level (the
 *  last level is the unconditional backstop — always resident, spec §3). */
export function hotLevelChain(
  levels: readonly FieldLevel[],
  wxz: NV2,
  arm: (lvl: FieldLevel, finest: boolean) => void,
): void {
  const n = levels.length;
  if (n === 0) throw new Error('TerrainField: hotLevelChain over zero levels');
  if (n === 1) {
    arm(levels[0] as FieldLevel, true);
    return;
  }
  let chain = If(insideLevel(levels[0] as FieldLevel, wxz), () => arm(levels[0] as FieldLevel, true));
  for (let i = 1; i < n - 1; i++) {
    const lvl = levels[i] as FieldLevel;
    chain = chain.ElseIf(insideLevel(lvl, wxz), () => arm(lvl, false));
  }
  chain.Else(() => arm(levels[n - 1] as FieldLevel, false));
}

/** one hardware-filtered tap (LinearFilter rgba8 planes) at a level. Wrapping
 *  levels take the 4 texel loads instead (a filtered tap would blend across
 *  the toroidal seam). */
function planeLinear(lvl: FieldLevel, wxz: NV2): NV4 {
  if (!lvl.wraps) {
    const uv = gridCoords(lvl, wxz).add(0.5).div(lvl.res);
    return texture(lvl.tex, uv as unknown as NV2, 0) as unknown as NV4;
  }
  const g = clamp(gridCoords(lvl, wxz), 0, lvl.res - 1);
  const i0 = floor(g);
  const f = fract(g);
  const x0 = physF(lvl, i0.x as unknown as NF, 0);
  const y0 = physF(lvl, i0.y as unknown as NF, 1);
  const x1 = physF(lvl, clamp(i0.x.add(1), 0, lvl.res - 1) as unknown as NF, 0);
  const y1 = physF(lvl, clamp(i0.y.add(1), 0, lvl.res - 1) as unknown as NF, 1);
  const uvOf = (x: NF, y: NF): NV2 => vec2(x.add(0.5), y.add(0.5)).div(lvl.res) as unknown as NV2;
  const s00 = texture(lvl.tex, uvOf(x0, y0), 0) as unknown as NV4;
  const s10 = texture(lvl.tex, uvOf(x1, y0), 0) as unknown as NV4;
  const s01 = texture(lvl.tex, uvOf(x0, y1), 0) as unknown as NV4;
  const s11 = texture(lvl.tex, uvOf(x1, y1), 0) as unknown as NV4;
  return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y) as unknown as NV4;
}

/** ±1-texel central differences at the nearest texel — the normalTex bake's
 *  exact stencil (see Heightfield rebuildDerivedMaps history) */
function cdTaps(lvl: FieldLevel, wxz: NV2): { dx: NF; dz: NF } {
  const g = clamp(gridCoords(lvl, wxz).add(0.5), 1, lvl.res - 2);
  const gx = floor(g.x) as unknown as NF;
  const gy = floor(g.y) as unknown as NF;
  const tl = (ix: NF, iy: NF): NF => {
    const t = texelU(lvl, ix, iy);
    return texLoadR(lvl.tex, t.x, t.y);
  };
  const hl = tl(gx.sub(1) as unknown as NF, gy);
  const hr = tl(gx.add(1) as unknown as NF, gy);
  const hd = tl(gx, gy.sub(1) as unknown as NF);
  const hu = tl(gx, gy.add(1) as unknown as NF);
  return { dx: hl.sub(hr) as unknown as NF, dz: hd.sub(hu) as unknown as NF };
}

function slope4(lvl: FieldLevel, wxz: NV2): NF {
  const { dx, dz } = cdTaps(lvl, wxz);
  return vec2(dx, dz).length().div(lvl.texel * 2) as unknown as NF;
}

/** vec4(world normal, slope) — the retired normalTex texel, derived live:
 *  n = normalize(hl−hr, 2·texel, hd−hu); slope = |(hl−hr, hd−hu)| / 2·texel */
function normalSlope4(lvl: FieldLevel, wxz: NV2): NV4 {
  const { dx, dz } = cdTaps(lvl, wxz);
  const n = vec3(dx, lvl.texel * 2, dz).normalize();
  return vec4(n, vec2(dx, dz).length().div(lvl.texel * 2)) as unknown as NV4;
}

export function planeBilerp(lvl: FieldLevel, wxz: NV2): NF {
  const g = clamp(gridCoords(lvl, wxz), 0, lvl.res - 1);
  const i0 = floor(g);
  const f = fract(g);
  const x0i = i0.x as unknown as NF;
  const y0i = i0.y as unknown as NF;
  const x1i = clamp(i0.x.add(1), 0, lvl.res - 1) as unknown as NF;
  const y1i = clamp(i0.y.add(1), 0, lvl.res - 1) as unknown as NF;
  const t00 = texelU(lvl, x0i, y0i);
  const t10 = texelU(lvl, x1i, y0i);
  const t01 = texelU(lvl, x0i, y1i);
  const t11 = texelU(lvl, x1i, y1i);
  const s00 = texLoadR(lvl.tex, t00.x, t00.y);
  const s10 = texLoadR(lvl.tex, t10.x, t10.y);
  const s01 = texLoadR(lvl.tex, t01.x, t01.y);
  const s11 = texLoadR(lvl.tex, t11.x, t11.y);
  return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

export function planeNearest(lvl: FieldLevel, wxz: NV2): NF {
  const g = clamp(gridCoords(lvl, wxz).add(0.5), 0, lvl.res - 1);
  const t = texelU(lvl, floor(g.x) as unknown as NF, floor(g.y) as unknown as NF);
  return texLoadR(lvl.tex, t.x, t.y);
}

/** true where the point sits ≥1 texel inside the level's window (the rim texel is
 *  clamp-extended fill or the scroll seam — the next-coarser level owns it) */
export function insideLevel(lvl: FieldLevel, wxz: NV2): NB {
  const g = gridCoords(lvl, wxz);
  return g.x
    .greaterThanEqual(1)
    .and(g.y.greaterThanEqual(1))
    .and(g.x.lessThanEqual(lvl.res - 2))
    .and(g.y.lessThanEqual(lvl.res - 2)) as NB;
}

// ---- CPU bilinear ---------------------------------------------------------------------

function bilerpCpu(lvl: HeightLevel, gx: number, gz: number): number {
  const res = lvl.res;
  const cx = Math.min(Math.max(gx, 0), res - 1.001);
  const cz = Math.min(Math.max(gz, 0), res - 1.001);
  const x0 = Math.floor(cx);
  const z0 = Math.floor(cz);
  const fx = cx - x0;
  const fz = cz - z0;
  const px = lvl.wraps ? lvl.uPhase.value.x : 0;
  const pz = lvl.wraps ? lvl.uPhase.value.y : 0;
  const at = (x: number, z: number): number => {
    const ix = (Math.min(x, res - 1) + px) % res;
    const iz = (Math.min(z, res - 1) + pz) % res;
    return lvl.data[iz * res + ix] ?? 0;
  };
  const a = at(x0, z0) * (1 - fx) + at(x0 + 1, z0) * fx;
  const b = at(x0, z0 + 1) * (1 - fx) + at(x0 + 1, z0 + 1) * fx;
  return a * (1 - fz) + b * fz;
}

// ---- level construction -----------------------------------------------------------------

function makeHeightLevel(name: string, plan: PlanePlan, data?: Float32Array): HeightLevel {
  const backing = data ?? new Float32Array(plan.res * plan.res);
  const tex = new DataTexture(backing, plan.res, plan.res, RedFormat, FloatType);
  configurePlane(tex, name);
  return { ...levelCommon(plan), tex, data: backing };
}

function makeU8Level(name: string, plan: PlanePlan): FieldLevel {
  const tex = new DataTexture(new Uint8Array(plan.res * plan.res * 4), plan.res, plan.res, RGBAFormat, UnsignedByteType);
  // rgba8 IS filterable — hot consumers (raster disp, grass density, resolve
  // shading) take ONE hardware-filtered tap instead of a manual 4-tap bilerp
  configurePlane(tex, name, LinearFilter);
  return { ...levelCommon(plan), tex };
}

function levelCommon(plan: PlanePlan): Omit<FieldLevel, 'tex'> {
  return {
    lod: plan.lod,
    res: plan.res,
    texel: plan.texel,
    originX: plan.originX,
    originZ: plan.originZ,
    uOrigin: uniform(new Vector2(plan.originX, plan.originZ)) as unknown as FieldLevel['uOrigin'],
    uPhase: uniform(new Vector2(0, 0)) as unknown as FieldLevel['uPhase'],
    wraps: plan.wraps,
  };
}

function configurePlane(tex: DataTexture, name: string, filter: typeof NearestFilter | typeof LinearFilter = NearestFilter): void {
  tex.name = name;
  tex.magFilter = filter;
  tex.minFilter = filter;
  tex.generateMipmaps = false;
  tex.flipY = false;
  tex.needsUpdate = true;
}
