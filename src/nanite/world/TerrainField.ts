/**
 * TerrainField — the streamed terrain field set (SPEC-STREAMING-WORLD §3, S3a).
 * The scene handle threaded where the one-shot Heightfield goes: a camera-window
 * plane pyramid filled EXCLUSIVELY through WorldSource.fetch (the universal path
 * both the generated world and Estonia ride), never by copying a source's GPU
 * textures. Owns:
 *
 *  - HEIGHT plane pyramid — r32float, level k texel = base·4^k, res per level by
 *    the COVER-OR-WINDOW rule (levelRes): the 2048² spec window, grown ≤4096²
 *    when that covers the source's whole span (the generated world rides
 *    full-coverage static planes; Estonia keeps camera windows). Levels = the
 *    source's height lods (generated: [0,1], statically filled once; S5 adds
 *    toroidal scroll — per-level origins are already uniforms so scrolling
 *    re-points them without shader rebuilds).
 *  - BIOME/CANOPY planes — rgba8 (cover-or-window, 1024² window ≤2048² cap):
 *    classId, vegDensity, canopyHeight, cover. Generated fill provides
 *    class+vegDensity; canopyHeight/cover stay 0 until S4 fills them from
 *    resident tree records + the asset-gen canopy pyramid.
 *  - SURFACE-FIELDS planes (F-1) — rgba8, same res rule: moisture, flow, snow,
 *    rockExposure. riverDepth is NOT stored: it is exactly waterY − height at wet
 *    texels, both of which are planes here (one derivation, no duplicate field).
 *  - WATER planes (F-1) — waterY at ~sim-res texel + a ×8 min-reduced far level.
 *    r32float, NOT r16float: f16 ULP at the ~140-300 m elevations both worlds use
 *    is 0.125-0.25 m — useless against 1 cm water levels (§10 rejected r16 planes
 *    for the same reason).
 *  - CPU mirrors — the DataTexture backing stores double as the mirrors (zero
 *    extra RAM): heightAt/waterAt serve walk probe, spawn and bookmarks.
 *
 * Lattice: every level is anchored on the SOURCE's payload lattice (A11): plane
 * texel p samples world = origin + p·texel where origin is the exact world coord
 * of texel (0,0)'s sample point. L0 fills are therefore bit-identical to the
 * source raster. F-7: static fill centers all levels on the coverage centroid —
 * generated (0,0), where the coarsest level covers the full extent; Estonia's
 * coarsest (L4) must center on the AOI centroid (it never scrolls). S5 re-centers
 * the finer levels on the camera.
 */

import { DataTexture, FloatType, LinearFilter, NearestFilter, RGBAFormat, RedFormat, UnsignedByteType, Vector2 } from 'three';
import { If, clamp, float, floor, fract, mix, texture, uniform, vec2, vec4 } from 'three/tsl';
import { texLoadR } from '../Tsl';
import type { NB, NF, NU, NV2, NV4 } from '../../gpu/TSLTypes';
import type { LayerName, WorldManifest, WorldSource } from '../../world/source/WorldSource';

const HEIGHT_PLANE_RES = 2048;
const U8_PLANE_RES = 1024;
/** cover-or-window rule (S3b): a level's plane res grows past the default window
 *  res up to this cap when it makes the plane cover the source's WHOLE layer
 *  span — a small world (the generated 4096 m) then rides full-coverage static
 *  planes with zero window seams, while a large one (Estonia) keeps the spec §3
 *  camera windows unchanged. The caps bound VRAM (throw-loud ceiling below). */
const HEIGHT_PLANE_RES_CAP = 4096;
const U8_PLANE_RES_CAP = 2048;
const WATER_FAR_FACTOR = 8;
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
  /** origin as a uniform — S5 toroidal scroll re-points it per frame */
  uOrigin: { value: Vector2 };
  tex: DataTexture;
}

interface HeightLevel extends FieldLevel {
  data: Float32Array;
}

interface LatticePlacement {
  originX: number;
  originZ: number;
  texel: number;
  stride: number;
  /** plane texel p ↔ source lattice index n = n0 + p (units of stride base texels) */
  n0x: number;
  n0z: number;
}

interface RasterGeom {
  texel0: number;
  chunkRes: number;
  originX: number;
  originZ: number;
  lodStep: number;
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

  private constructor(
    heightLevels: HeightLevel[],
    biomeLevels: FieldLevel[],
    fieldsLevels: FieldLevel[],
    water: HeightLevel | null,
    waterFar: HeightLevel | null,
  ) {
    if (heightLevels.length === 0) throw new Error('TerrainField: needs at least one height level');
    this.heightLevels = heightLevels;
    this.biomeLevels = biomeLevels;
    this.fieldsLevels = fieldsLevels;
    this.water = water;
    this.waterFar = waterFar;
    const mb = this.vramBytes() / 2 ** 20;
    // eslint-disable-next-line no-console
    console.log(
      `[laas] terrain field: height [${heightLevels.map((l) => `${l.res}²@${l.texel}m`).join(' ')}] r32f + ` +
        `biome [${biomeLevels.map((l) => `${l.res}²`).join(' ')}] + fields [${fieldsLevels.map((l) => `${l.res}²`).join(' ')}] rgba8 + ` +
        `water ${water ? `${water.res}² r32f (+far ${waterFar?.res ?? 0}²)` : 'none'} = ` +
        `${mb.toFixed(1)} MB VRAM (CPU mirrors share the backing)`,
    );
    if (mb > VRAM_CEILING_MB) {
      throw new Error(`TerrainField: ${mb.toFixed(1)} MB exceeds the ${VRAM_CEILING_MB} MB ceiling`);
    }
  }

  /** Build + statically fill every plane from the source via fetch — the SAME
   *  path Estonia uses (S5 turns the static fill into scroll updates). */
  static async fromSource(source: WorldSource, manifest: WorldManifest): Promise<TerrainField> {
    const heightMeta = manifest.layers.height;
    if (!heightMeta) throw new Error('TerrainField: source has no height layer');
    const { cx, cz } = coverageCenter(manifest);

    const heightGeo = layerGeom(manifest, 'height');
    const heightSpan = layerSpanM(manifest, 'height');
    const heightLevels: HeightLevel[] = [];
    for (const lod of heightMeta.lods) {
      const res = levelRes(heightSpan, heightGeo.texel0 * heightGeo.lodStep ** lod, HEIGHT_PLANE_RES, HEIGHT_PLANE_RES_CAP);
      const place = placeLevel(heightGeo, lod, res, cx, cz);
      const data = await fillF32Level(source, manifest, 'height', lod, res, place, heightGeo);
      heightLevels.push(makeHeightLevel(`terrainFieldHeightL${lod}`, lod, res, place, data));
    }

    const biomeLevels = await fillU8Layer(source, manifest, 'biome', cx, cz, [
      ['classId', 0],
      ['vegDensity', 1],
      // canopyHeight (2) + cover (3) stay 0 — S4 fills them from tree records
    ]);
    const fieldsLevels = await fillU8Layer(source, manifest, 'fields', cx, cz, [
      ['moisture', 0],
      ['flowStrength', 1],
      ['snow', 2],
      ['rockExposure', 3],
    ]);

    let water: HeightLevel | null = null;
    let waterFar: HeightLevel | null = null;
    const waterMeta = manifest.layers.water;
    if (waterMeta && waterMeta.lods.includes(0)) {
      const geo = layerGeom(manifest, 'water');
      const waterRes = levelRes(layerSpanM(manifest, 'water'), geo.texel0, HEIGHT_PLANE_RES, HEIGHT_PLANE_RES);
      const place = placeLevel(geo, 0, waterRes, cx, cz);
      // Estonia dry texels decode to NaN (§9a) — a NaN reaching bilinear poisons
      // whole quads, so map to the dry sentinel the generated field already uses
      // downstream of its bed−2 encoding. Generated payloads carry no NaN.
      const data = await fillF32Level(source, manifest, 'water', 0, waterRes, place, geo, -1e4);
      water = makeHeightLevel('terrainFieldWaterY', 0, waterRes, place, data);
      const farRes = waterRes / WATER_FAR_FACTOR;
      const farData = minReduce(data, waterRes, WATER_FAR_FACTOR);
      waterFar = makeHeightLevel('terrainFieldWaterYFar', 0, farRes, {
        ...place,
        texel: place.texel * WATER_FAR_FACTOR,
        // block (0,0) spans samples 0..7 — its representative point is their center
        originX: place.originX + (place.texel * (WATER_FAR_FACTOR - 1)) / 2,
        originZ: place.originZ + (place.texel * (WATER_FAR_FACTOR - 1)) / 2,
      }, farData);
    }

    return new TerrainField(heightLevels, biomeLevels, fieldsLevels, water, waterFar);
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
    const place: LatticePlacement = {
      originX: opts.worldMinX + opts.texel * 0.5,
      originZ: opts.worldMinZ + opts.texel * 0.5,
      texel: opts.texel,
      stride: 1,
      n0x: 0,
      n0z: 0,
    };
    return new TerrainField([makeHeightLevel('terrainFieldHeightL0', 0, opts.res, place, data)], [], [], null, null);
  }

  // ---- CPU sampling (walk probe, spawn, bookmarks) --------------------------------

  /** bilinear height (m) — finest level whose window contains the point */
  heightAt(x: number, z: number): number {
    const L = this.heightLevels;
    for (let i = 0; i < L.length - 1; i++) {
      const lvl = L[i] as HeightLevel;
      const gx = (x - lvl.originX) / lvl.texel;
      const gz = (z - lvl.originZ) / lvl.texel;
      // 1-texel margin: the plane rim is clamp-extended fill, the coarser level
      // holds the real slope there
      if (gx >= 1 && gx <= lvl.res - 2 && gz >= 1 && gz <= lvl.res - 2) {
        return bilerpCpu(lvl.data, lvl.res, gx, gz);
      }
    }
    const lvl = L[L.length - 1] as HeightLevel;
    return bilerpCpu(lvl.data, lvl.res, (x - lvl.originX) / lvl.texel, (z - lvl.originZ) / lvl.texel);
  }

  /** bilinear waterY (m); dry cells sit well below the bed (sentinel/bed−2), so
   *  max(ground, waterAt + ε) stays a safe camera floor. −1e4 without water. */
  waterAt(x: number, z: number): number {
    const w = this.water;
    if (!w) return -1e4;
    return bilerpCpu(w.data, w.res, (x - w.originX) / w.texel, (z - w.originZ) / w.texel);
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
   *  of the finest containing level. */
  fieldSlope(wxz: NV2): NF {
    const out = float(0).toVar();
    hotLevelChain(this.heightLevels, wxz, (lvl) => out.assign(slope4(lvl, wxz)));
    return out as unknown as NF;
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
    for (const l of this.heightLevels) b += l.res * l.res * 4;
    for (const l of this.biomeLevels) b += l.res * l.res * 4;
    for (const l of this.fieldsLevels) b += l.res * l.res * 4;
    if (this.water) b += this.water.res * this.water.res * 4;
    if (this.waterFar) b += this.waterFar.res * this.waterFar.res * 4;
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

/** world → continuous sample-grid coords (integer = exact source sample) */
export function gridCoords(lvl: FieldLevel, wxz: NV2): NV2 {
  return wxz.sub(vec2(lvl.uOrigin as unknown as NV2)).div(lvl.texel);
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

/** one hardware-filtered tap (LinearFilter rgba8 planes) at a level */
function planeLinear(lvl: FieldLevel, wxz: NV2): NV4 {
  const uv = gridCoords(lvl, wxz).add(0.5).div(lvl.res);
  return texture(lvl.tex, uv as unknown as NV2, 0) as unknown as NV4;
}

/** ±1-texel central-difference slope at the nearest texel — the normalTex
 *  bake's exact stencil (see Heightfield rebuildDerivedMaps history) */
function slope4(lvl: FieldLevel, wxz: NV2): NF {
  const g = clamp(gridCoords(lvl, wxz).add(0.5), 1, lvl.res - 2);
  const x = floor(g.x).toUint() as NU;
  const y = floor(g.y).toUint() as NU;
  const hl = texLoadR(lvl.tex, x.sub(1) as unknown as NU, y);
  const hr = texLoadR(lvl.tex, x.add(1) as unknown as NU, y);
  const hd = texLoadR(lvl.tex, x, y.sub(1) as unknown as NU);
  const hu = texLoadR(lvl.tex, x, y.add(1) as unknown as NU);
  return vec2(hl.sub(hr), hd.sub(hu)).length().div(lvl.texel * 2) as unknown as NF;
}

export function planeBilerp(lvl: FieldLevel, wxz: NV2): NF {
  const g = clamp(gridCoords(lvl, wxz), 0, lvl.res - 1);
  const i0 = floor(g);
  const f = fract(g);
  const x0 = i0.x.toUint() as NU;
  const y0 = i0.y.toUint() as NU;
  const x1 = clamp(i0.x.add(1), 0, lvl.res - 1).toUint() as NU;
  const y1 = clamp(i0.y.add(1), 0, lvl.res - 1).toUint() as NU;
  const s00 = texLoadR(lvl.tex, x0, y0);
  const s10 = texLoadR(lvl.tex, x1, y0);
  const s01 = texLoadR(lvl.tex, x0, y1);
  const s11 = texLoadR(lvl.tex, x1, y1);
  return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

export function planeNearest(lvl: FieldLevel, wxz: NV2): NF {
  const g = clamp(gridCoords(lvl, wxz).add(0.5), 0, lvl.res - 1);
  return texLoadR(lvl.tex, floor(g.x).toUint() as NU, floor(g.y).toUint() as NU);
}

/** true where the point sits ≥1 texel inside the level's window (the rim texel is
 *  clamp-extended fill — the next-coarser level owns it) */
export function insideLevel(lvl: FieldLevel, wxz: NV2): NB {
  const g = gridCoords(lvl, wxz);
  return g.x
    .greaterThanEqual(1)
    .and(g.y.greaterThanEqual(1))
    .and(g.x.lessThanEqual(lvl.res - 2))
    .and(g.y.lessThanEqual(lvl.res - 2)) as NB;
}

// ---- CPU bilinear ---------------------------------------------------------------------

function bilerpCpu(data: Float32Array, res: number, gx: number, gz: number): number {
  const cx = Math.min(Math.max(gx, 0), res - 1.001);
  const cz = Math.min(Math.max(gz, 0), res - 1.001);
  const x0 = Math.floor(cx);
  const z0 = Math.floor(cz);
  const fx = cx - x0;
  const fz = cz - z0;
  const at = (x: number, z: number): number => data[Math.min(z, res - 1) * res + Math.min(x, res - 1)] ?? 0;
  const a = at(x0, z0) * (1 - fx) + at(x0 + 1, z0) * fx;
  const b = at(x0, z0 + 1) * (1 - fx) + at(x0 + 1, z0 + 1) * fx;
  return a * (1 - fz) + b * fz;
}

// ---- construction: lattice placement + fetch fills --------------------------------------

function layerGeom(manifest: WorldManifest, layer: LayerName): RasterGeom {
  const meta = manifest.layers[layer];
  const t0 = meta?.texelMeters;
  if (!meta || !t0) throw new Error(`TerrainField: layer '${layer}' missing texelMeters`);
  const g = manifest.grid;
  return {
    texel0: t0,
    chunkRes: Math.round(g.chunkMeters / t0),
    originX: g.originX,
    originZ: g.originZ,
    lodStep: g.lodStep,
  };
}

/** span (m) of a layer's finest-lod chunk coverage — the honest data extent
 *  the cover-or-window rule (levelRes) measures against */
function layerSpanM(manifest: WorldManifest, layer: LayerName): number {
  const meta = manifest.layers[layer];
  if (!meta) return 0;
  const finest = Math.min(...meta.lods);
  const keys = manifest.chunks(layer, finest);
  if (keys.length === 0) return 0;
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const k of keys) {
    minX = Math.min(minX, k.cx);
    maxX = Math.max(maxX, k.cx);
    minZ = Math.min(minZ, k.cz);
    maxZ = Math.max(maxZ, k.cz);
  }
  const span = manifest.grid.chunkMeters * manifest.grid.lodStep ** finest;
  return Math.max(maxX - minX + 1, maxZ - minZ + 1) * span;
}

/** cover-or-window plane res: the smallest power of two whose window at this
 *  texel covers the layer span, clamped to [256, cap]; a span past the cap
 *  falls back to the camera-window default (Estonia semantics, spec §3). */
function levelRes(spanM: number, texel: number, windowRes: number, cap: number): number {
  const cov = Math.round(spanM / texel);
  if (cov <= 0) return windowRes;
  let r = 256;
  while (r < cov && r < cap) r *= 2;
  return r >= cov ? r : windowRes;
}

/** coverage centroid from the FINEST height lod's chunk set (F-7) */
function coverageCenter(manifest: WorldManifest): { cx: number; cz: number } {
  const meta = manifest.layers.height as NonNullable<WorldManifest['layers']['height']>;
  const finest = Math.min(...meta.lods);
  const keys = manifest.chunks('height', finest);
  if (keys.length === 0) throw new Error('TerrainField: height layer has no chunks');
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const k of keys) {
    minX = Math.min(minX, k.cx);
    maxX = Math.max(maxX, k.cx);
    minZ = Math.min(minZ, k.cz);
    maxZ = Math.max(maxZ, k.cz);
  }
  const g = manifest.grid;
  const span = g.chunkMeters * g.lodStep ** finest;
  return {
    cx: g.originX + ((minX + maxX + 1) / 2) * span,
    cz: g.originZ + ((minZ + maxZ + 1) / 2) * span,
  };
}

/** anchor a res² plane at `lod` on the source's payload lattice, centered as close
 *  to (centerX, centerZ) as the lattice allows. Payload sample i of chunk c sits at
 *  world = gridOrigin + ((c·chunkRes + i)·stride + off + 0.5)·texel0 with
 *  off = stride>>1 (the offset-centered coarse subsample, A11) — so the plane's
 *  sample n0+p lands EXACTLY on payload samples; L0 fills are bit-identical. */
function placeLevel(geo: RasterGeom, lod: number, res: number, centerX: number, centerZ: number): LatticePlacement {
  const stride = geo.lodStep ** lod;
  const texel = geo.texel0 * stride;
  const off = stride >> 1;
  const n0x = Math.round((centerX - (res / 2) * texel - geo.originX) / texel);
  const n0z = Math.round((centerZ - (res / 2) * texel - geo.originZ) / texel);
  return {
    n0x,
    n0z,
    stride,
    texel,
    originX: geo.originX + (n0x * stride + off + 0.5) * geo.texel0,
    originZ: geo.originZ + (n0z * stride + off + 0.5) * geo.texel0,
  };
}

interface FilledBox {
  x0: number;
  x1: number;
  z0: number;
  z1: number;
}

async function fillF32Level(
  source: WorldSource,
  manifest: WorldManifest,
  layer: LayerName,
  lod: number,
  res: number,
  place: LatticePlacement,
  geo: RasterGeom,
  mapNaN?: number,
): Promise<Float32Array> {
  const out = new Float32Array(res * res);
  let box: FilledBox | null = null;
  for (const key of manifest.chunks(layer, lod)) {
    const payload = await source.fetch(layer, key);
    if (!payload || payload.kind !== 'height') continue;
    box = copyChunkF32(out, res, place, geo, key.cx, key.cz, payload.heights, payload.res, box, mapNaN);
  }
  if (!box) {
    // eslint-disable-next-line no-console
    console.warn(`[laas] terrain field: layer '${layer}' lod ${lod} — no chunks landed, plane is zero`);
    return out;
  }
  clampExtend(out, res, 1, box);
  return out;
}

function copyChunkF32(
  out: Float32Array,
  res: number,
  place: LatticePlacement,
  geo: RasterGeom,
  ccx: number,
  ccz: number,
  src: Float32Array,
  srcRes: number,
  box: FilledBox | null,
  mapNaN?: number,
): FilledBox {
  // chunk (ccx) holds lattice n ∈ [ccx·chunkRes, (ccx+1)·chunkRes]; skip the apron
  // sample (n = (c+1)·chunkRes) — the east/south neighbor owns it, the world rim
  // is clamp-extended afterwards
  const pxLo = Math.max(0, ccx * geo.chunkRes - place.n0x);
  const pxHi = Math.min(res - 1, (ccx + 1) * geo.chunkRes - 1 - place.n0x);
  const pzLo = Math.max(0, ccz * geo.chunkRes - place.n0z);
  const pzHi = Math.min(res - 1, (ccz + 1) * geo.chunkRes - 1 - place.n0z);
  for (let pz = pzLo; pz <= pzHi; pz++) {
    const i0 = (place.n0z + pz - ccz * geo.chunkRes) * srcRes + (place.n0x + pxLo - ccx * geo.chunkRes);
    const o0 = pz * res + pxLo;
    for (let k = 0; k <= pxHi - pxLo; k++) {
      const v = src[i0 + k] as number;
      out[o0 + k] = mapNaN !== undefined && Number.isNaN(v) ? mapNaN : v;
    }
  }
  if (pxLo > pxHi || pzLo > pzHi) return box ?? { x0: res, x1: -1, z0: res, z1: -1 };
  if (!box) return { x0: pxLo, x1: pxHi, z0: pzLo, z1: pzHi };
  return {
    x0: Math.min(box.x0, pxLo),
    x1: Math.max(box.x1, pxHi),
    z0: Math.min(box.z0, pzLo),
    z1: Math.max(box.z1, pzHi),
  };
}

/** extend the filled region's border texels to the plane rim (channels = words per
 *  texel: 1 for f32 planes, 4 for interleaved rgba8) */
function clampExtend(data: Float32Array | Uint8Array, res: number, channels: number, box: FilledBox): void {
  if (box.x1 < box.x0 || box.z1 < box.z0) return;
  const row = res * channels;
  for (let z = box.z0; z <= box.z1; z++) {
    const r = z * row;
    for (let c = 0; c < channels; c++) {
      const lo = data[r + box.x0 * channels + c] as number;
      const hi = data[r + box.x1 * channels + c] as number;
      for (let x = 0; x < box.x0; x++) data[r + x * channels + c] = lo;
      for (let x = box.x1 + 1; x < res; x++) data[r + x * channels + c] = hi;
    }
  }
  for (let z = 0; z < box.z0; z++) data.copyWithin(z * row, box.z0 * row, box.z0 * row + row);
  for (let z = box.z1 + 1; z < res; z++) data.copyWithin(z * row, box.z1 * row, box.z1 * row + row);
}

/** fill every lod of a u8-planes layer into rgba8 levels; channelMap = payload
 *  plane name → output channel. Missing layer/planes log once and stay zero. */
async function fillU8Layer(
  source: WorldSource,
  manifest: WorldManifest,
  layer: LayerName,
  centerX: number,
  centerZ: number,
  channelMap: readonly (readonly [string, number])[],
): Promise<FieldLevel[]> {
  const meta = manifest.layers[layer];
  if (!meta) {
    // eslint-disable-next-line no-console
    console.log(`[laas] terrain field: source has no '${layer}' layer — planes stay zero (derived at fill on sources that carry equivalents)`);
    return [];
  }
  const names = meta.planes ?? [];
  const geo = layerGeom(manifest, layer);
  const span = layerSpanM(manifest, layer);
  const levels: FieldLevel[] = [];
  for (const lod of meta.lods) {
    const res = levelRes(span, geo.texel0 * geo.lodStep ** lod, U8_PLANE_RES, U8_PLANE_RES_CAP);
    const place = placeLevel(geo, lod, res, centerX, centerZ);
    const out = new Uint8Array(res * res * 4);
    let box: FilledBox | null = null;
    for (const key of manifest.chunks(layer, lod)) {
      const payload = await source.fetch(layer, key);
      if (!payload || payload.kind !== 'planes') continue;
      const pxLo = Math.max(0, key.cx * geo.chunkRes - place.n0x);
      const pxHi = Math.min(res - 1, (key.cx + 1) * geo.chunkRes - 1 - place.n0x);
      const pzLo = Math.max(0, key.cz * geo.chunkRes - place.n0z);
      const pzHi = Math.min(res - 1, (key.cz + 1) * geo.chunkRes - 1 - place.n0z);
      if (pxLo > pxHi || pzLo > pzHi) continue;
      for (const [name, ch] of channelMap) {
        const pi = names.indexOf(name);
        if (pi < 0) continue;
        const src = payload.planes[pi] as Uint8Array;
        for (let pz = pzLo; pz <= pzHi; pz++) {
          const i0 = (place.n0z + pz - key.cz * geo.chunkRes) * payload.res + (place.n0x + pxLo - key.cx * geo.chunkRes);
          const o0 = (pz * res + pxLo) * 4 + ch;
          for (let k = 0; k <= pxHi - pxLo; k++) out[o0 + k * 4] = src[i0 + k] as number;
        }
      }
      box = !box
        ? { x0: pxLo, x1: pxHi, z0: pzLo, z1: pzHi }
        : { x0: Math.min(box.x0, pxLo), x1: Math.max(box.x1, pxHi), z0: Math.min(box.z0, pzLo), z1: Math.max(box.z1, pzHi) };
    }
    for (const [name] of channelMap) {
      if (names.length > 0 && !names.includes(name)) {
        // eslint-disable-next-line no-console
        console.log(`[laas] terrain field: '${layer}' has no '${name}' plane — channel stays zero`);
      }
    }
    if (box) clampExtend(out, res, 4, box);
    const tex = new DataTexture(out, res, res, RGBAFormat, UnsignedByteType);
    // rgba8 IS filterable — hot consumers (raster disp, grass density, resolve
    // shading) take ONE hardware-filtered tap instead of a manual 4-tap bilerp
    configurePlane(tex, `terrainField${layer[0]?.toUpperCase()}${layer.slice(1)}L${lod}`, LinearFilter);
    levels.push({
      lod,
      res,
      texel: place.texel,
      originX: place.originX,
      originZ: place.originZ,
      uOrigin: uniform(new Vector2(place.originX, place.originZ)) as unknown as FieldLevel['uOrigin'],
      tex,
    });
  }
  return levels;
}

function makeHeightLevel(name: string, lod: number, res: number, place: LatticePlacement, data: Float32Array): HeightLevel {
  const tex = new DataTexture(data, res, res, RedFormat, FloatType);
  configurePlane(tex, name);
  return {
    lod,
    res,
    texel: place.texel,
    originX: place.originX,
    originZ: place.originZ,
    uOrigin: uniform(new Vector2(place.originX, place.originZ)) as unknown as FieldLevel['uOrigin'],
    tex,
    data,
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

function minReduce(src: Float32Array, res: number, factor: number): Float32Array {
  const farRes = Math.floor(res / factor);
  const out = new Float32Array(farRes * farRes);
  for (let z = 0; z < farRes; z++) {
    for (let x = 0; x < farRes; x++) {
      let mn = Infinity;
      for (let oz = 0; oz < factor; oz++) {
        const r = (z * factor + oz) * res + x * factor;
        for (let ox = 0; ox < factor; ox++) mn = Math.min(mn, src[r + ox] as number);
      }
      out[z * farRes + x] = mn;
    }
  }
  return out;
}
