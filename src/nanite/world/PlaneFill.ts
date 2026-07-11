/**
 * PlaneFill — the pure lattice/placement/fill math for the TerrainField plane
 * pyramid (SPEC-STREAMING-WORLD §3/§4, S5). No three/DOM/GPU deps: the SAME
 * code computes placements and assembles texel regions on the main thread
 * (TerrainField construction / plan) and inside the StreamBrain worker (window
 * fills + scrolls), so the two can never drift — the S5 parity gate rests on
 * this file being the single source of the lattice truth.
 *
 * Lattice contract (A11): payload sample i of chunk c sits at world =
 * gridOrigin + ((c·chunkRes + i)·stride + off + 0.5)·texel0 with off = stride>>1
 * (offset-centered coarse subsample). A plane of res² texels at level lod is
 * anchored so its texel p IS payload sample n0+p — L0 fills are bit-identical
 * to the source raster.
 */

import type { ChunkKey, LayerName, WorldManifest } from '../../world/source/WorldSource';

/** plane texel p ↔ source lattice index n = n0 + p (units of stride base texels) */
export interface LatticePlacement {
  originX: number;
  originZ: number;
  texel: number;
  stride: number;
  n0x: number;
  n0z: number;
}

export interface RasterGeom {
  texel0: number;
  chunkRes: number;
  originX: number;
  originZ: number;
  lodStep: number;
}

/** chunk-aligned world box of a source's height coverage (S4 window clamps). */
export interface CoverageBox {
  minX: number;
  minZ: number;
  maxX: number;
  maxZ: number;
}

export interface FilledBox {
  x0: number;
  x1: number;
  z0: number;
  z1: number;
}

/** one plane level's frozen plan: placement + whether it can ever scroll
 *  (window smaller than the layer's chunk coverage at this lod — Estonia's
 *  near levels; every generated level covers ⇒ pinned ⇒ wraps false and the
 *  sampling shaders compile the exact pre-S5 form). */
export interface PlanePlan extends LatticePlacement {
  lod: number;
  res: number;
  wraps: boolean;
  /** lattice index bounds of the layer's chunk coverage at this lod (inclusive
   *  sample range) — scroll clamps + rim clamp-extend measure against these. */
  nMinX: number;
  nMaxX: number;
  nMinZ: number;
  nMaxZ: number;
}

export interface FieldPlan {
  height: PlanePlan[];
  biome: PlanePlan[];
  fields: PlanePlan[];
  water: PlanePlan | null;
  waterFar: PlanePlan | null;
  coverageBox: CoverageBox;
}

export const HEIGHT_PLANE_RES = 2048;
export const U8_PLANE_RES = 1024;
/** cover-or-window rule (S3b): a level's plane res grows past the default
 *  window res up to the cap when that makes it cover the source's WHOLE layer
 *  span (generated world ⇒ full-coverage static planes); a larger world keeps
 *  the spec §3 camera windows. */
export const HEIGHT_PLANE_RES_CAP = 4096;
export const U8_PLANE_RES_CAP = 2048;
export const WATER_FAR_FACTOR = 8;
export const BIOME_CHANNELS: readonly (readonly [string, number])[] = [
  ['classId', 0],
  ['vegDensity', 1],
  // canopyHeight (2) + cover (3): L0 stays 0 (near = the S4 canopy window); the
  // far levels (LODs 1-4) merge the canopy layer via CANOPY_CHANNELS at boot.
];
/** far-forest canopy layer planes [heightM, cover] → biome plane channels 2/3.
 *  Merged onto the biome plane for the levels the canopy layer cooks (LODs 1-4).
 *  heightM = mean canopy height in meters (u8); cover = canopy-cover fraction×255
 *  (0 = treeless/unmeasured ⇒ no far tint/displacement). */
export const CANOPY_CHANNELS: readonly (readonly [string, number])[] = [
  ['heightM', 2],
  ['cover', 3],
];
export const FIELDS_CHANNELS: readonly (readonly [string, number])[] = [
  ['moisture', 0],
  ['flowStrength', 1],
  ['snow', 2],
  ['rockExposure', 3],
];
/** Estonia dry water texels decode to NaN (§9a) — mapped to the dry sentinel
 *  the generated field uses downstream of its bed−2 encoding. */
export const WATER_DRY_SENTINEL = -1e4;

export function layerGeom(manifest: WorldManifest, layer: LayerName): RasterGeom {
  const meta = manifest.layers[layer];
  const t0 = meta?.texelMeters;
  if (!meta || !t0) throw new Error(`PlaneFill: layer '${layer}' missing texelMeters`);
  const g = manifest.grid;
  return {
    texel0: t0,
    chunkRes: Math.round(g.chunkMeters / t0),
    originX: g.originX,
    originZ: g.originZ,
    lodStep: g.lodStep,
  };
}

/** chunk box (indices) of a layer's chunk set at one lod. */
export function chunkBox(
  keys: readonly ChunkKey[],
): { minX: number; maxX: number; minZ: number; maxZ: number } | null {
  if (keys.length === 0) return null;
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
  return { minX, maxX, minZ, maxZ };
}

/** span (m) of a layer's finest-lod chunk coverage — what levelRes measures. */
export function layerSpanM(manifest: WorldManifest, layer: LayerName): number {
  const meta = manifest.layers[layer];
  if (!meta) return 0;
  const finest = Math.min(...meta.lods);
  const box = chunkBox(manifest.chunks(layer, finest));
  if (!box) return 0;
  const span = manifest.grid.chunkMeters * manifest.grid.lodStep ** finest;
  return Math.max(box.maxX - box.minX + 1, box.maxZ - box.minZ + 1) * span;
}

/** cover-or-window plane res (see HEIGHT_PLANE_RES_CAP note). */
export function levelRes(spanM: number, texel: number, windowRes: number, cap: number): number {
  const cov = Math.round(spanM / texel);
  if (cov <= 0) return windowRes;
  let r = 256;
  while (r < cov && r < cap) r *= 2;
  return r >= cov ? r : windowRes;
}

function heightChunkBoxOf(manifest: WorldManifest): {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
  span: number;
} {
  const meta = manifest.layers.height;
  if (!meta) throw new Error('PlaneFill: source has no height layer');
  const finest = Math.min(...meta.lods);
  const box = chunkBox(manifest.chunks('height', finest));
  if (!box) throw new Error('PlaneFill: height layer has no chunks');
  return { ...box, span: manifest.grid.chunkMeters * manifest.grid.lodStep ** finest };
}

/** coverage centroid from the FINEST height lod's chunk set (F-7 anchoring). */
export function coverageCenter(manifest: WorldManifest): { cx: number; cz: number } {
  const b = heightChunkBoxOf(manifest);
  const g = manifest.grid;
  return {
    cx: g.originX + ((b.minX + b.maxX + 1) / 2) * b.span,
    cz: g.originZ + ((b.minZ + b.maxZ + 1) / 2) * b.span,
  };
}

/** chunk-aligned world box of the height coverage (generated: ±WORLD_HALF). */
export function coverageBoxM(manifest: WorldManifest): CoverageBox {
  const b = heightChunkBoxOf(manifest);
  const g = manifest.grid;
  return {
    minX: g.originX + b.minX * b.span,
    minZ: g.originZ + b.minZ * b.span,
    maxX: g.originX + (b.maxX + 1) * b.span,
    maxZ: g.originZ + (b.maxZ + 1) * b.span,
  };
}

/** aerial haze fully obscures terrain past ~50 km, so terrain (and the tile
 *  clipmap that renders it) reaches at most this far — the camera far plane, the
 *  tile-residency box, AND the coarsest height-plane floor all cap here so they
 *  agree by construction. */
export const VIEW_FAR_CAP_M = 150000;
/** view-far margin over the coarsest LOD's world half-extent (a small overshoot
 *  so the horizon is never shy of the data). */
export const VIEW_FAR_MARGIN = 1.15;
/** coarsest ("country floor") height-plane res ceiling — one r32f level spanning
 *  the whole tile-residency box at the coarsest LOD texel. 4096² r32f = 64 MB is
 *  the throw-loud wall (the TerrainField VRAM ceiling catches it too); Estonia's
 *  418 km box at 256 m needs 2048² = 16 MB. A box that needs more than this is a
 *  view-far / world-scale mistake to surface, not silently allocate. */
export const HEIGHT_FLOOR_RES_CAP = 4096;

/**
 * The tile-residency COVERAGE BOX in finest-lattice texels (the domain the
 * terrain tile clipmap partitions, hence the ONLY region terrain can render in) —
 * the SINGLE source both StreamBrainClient (tree latMin/latMax) and planField (the
 * coarsest floor level's extent) derive from, so the tree domain and its coarse
 * source can never drift (S6g/S8c). Box = the finest-LOD centre (where the camera
 * lives) ± the view-far horizon, clipped to the union of EVERY LOD's chunk
 * footprint (the country-wide coarse LODs), squared. See buildInit's historic note
 * for why the finest-LOD footprint alone was the wrong, void-opening source.
 */
export function coverageExtentLattice(manifest: WorldManifest): { latMin: number; latMax: number } {
  const geo = layerGeom(manifest, 'height');
  const lods = manifest.layers.height?.lods ?? [0];
  const finestLod = Math.min(...lods);
  let uMinX = Infinity;
  let uMaxX = -Infinity;
  let uMinZ = Infinity;
  let uMaxZ = -Infinity;
  let extentHalf = 0;
  let fine: ReturnType<typeof chunkBox> = null;
  for (const lod of lods) {
    const keys = manifest.chunks('height', lod);
    // a LONE coarse chunk is PADDING that merely contains the finer world (skip it
    // so the box tracks real data, not chunk-footprint overshoot — keeps the
    // generated world's box at its finest-only value ⇒ bit-identical).
    if (lod !== finestLod && keys.length <= 1) continue;
    const b = chunkBox(keys);
    if (!b) continue;
    const f = geo.chunkRes * geo.lodStep ** lod; // lattice texels per LOD-k chunk
    uMinX = Math.min(uMinX, b.minX * f);
    uMaxX = Math.max(uMaxX, (b.maxX + 1) * f);
    uMinZ = Math.min(uMinZ, b.minZ * f);
    uMaxZ = Math.max(uMaxZ, (b.maxZ + 1) * f);
    extentHalf = Math.max(extentHalf, ((b.maxX - b.minX + 1) * f) / 2, ((b.maxZ - b.minZ + 1) * f) / 2);
    if (lod === finestLod) fine = b;
  }
  if (!fine || !Number.isFinite(uMinX)) throw new Error('PlaneFill: height layer has no chunks');
  const f0 = geo.chunkRes * geo.lodStep ** finestLod;
  const ccx = ((fine.minX + fine.maxX + 1) / 2) * f0;
  const ccz = ((fine.minZ + fine.maxZ + 1) / 2) * f0;
  const viewFar = Math.min(VIEW_FAR_CAP_M / geo.texel0, extentHalf * VIEW_FAR_MARGIN);
  // squared (min/max across axes) — the tile clipmap lattice is square. Round
  // OUTWARD so coverage is never shaved (integer generated bounds are unchanged).
  const latMin = Math.floor(Math.min(Math.max(uMinX, ccx - viewFar), Math.max(uMinZ, ccz - viewFar)));
  const latMax = Math.ceil(Math.max(Math.min(uMaxX, ccx + viewFar), Math.min(uMaxZ, ccz + viewFar))) - 1;
  return { latMin, latMax };
}

/** anchor a res² plane at `lod` on the source's payload lattice, centered as
 *  close to (centerX, centerZ) as the lattice allows. */
export function placeLevel(
  geo: RasterGeom,
  lod: number,
  res: number,
  centerX: number,
  centerZ: number,
): LatticePlacement {
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

/** world coord of lattice sample n at a level (same formula as placeLevel). */
export function latticeWorld(geo: RasterGeom, lod: number, n: number, axis: 'x' | 'z'): number {
  const stride = geo.lodStep ** lod;
  const off = stride >> 1;
  const o = axis === 'x' ? geo.originX : geo.originZ;
  return o + (n * stride + off + 0.5) * geo.texel0;
}

/** the full per-level plan for one raster layer (placement + coverage bounds +
 *  the frozen wraps flag). Centered on the coverage centroid (F-7) — S5 scroll
 *  re-centers wrapping levels on the camera, pinned levels never move. */
export function planLayer(
  manifest: WorldManifest,
  layer: LayerName,
  windowRes: number,
  cap: number,
): PlanePlan[] {
  const meta = manifest.layers[layer];
  if (!meta) return [];
  const geo = layerGeom(manifest, layer);
  const span = layerSpanM(manifest, layer);
  const { cx, cz } = coverageCenter(manifest);
  const plans: PlanePlan[] = [];
  for (const lod of meta.lods) {
    const texel = geo.texel0 * geo.lodStep ** lod;
    const res = levelRes(span, texel, windowRes, cap);
    const place = placeLevel(geo, lod, res, cx, cz);
    const box = chunkBox(manifest.chunks(layer, lod));
    // lattice sample bounds of the chunk coverage at this lod (apron sample
    // (c+1)·chunkRes belongs to the east/south neighbor; the rim keeps it)
    const nMinX = box ? box.minX * geo.chunkRes : 0;
    const nMaxX = box ? (box.maxX + 1) * geo.chunkRes : 0;
    const nMinZ = box ? box.minZ * geo.chunkRes : 0;
    const nMaxZ = box ? (box.maxZ + 1) * geo.chunkRes : 0;
    // wraps measures the honest LAYER span (levelRes's metric), NOT the chunk
    // box — a coarse chunk's footprint can overhang the world in clamp-padding
    // (the generated lod1 chunk spans 8192 m of a 4096 m world) and must not
    // demote a covering level to a scrolling one (pinned ⇒ the samplers
    // compile the exact pre-S5 form — the parity + register gates).
    const wraps = res < Math.round(span / texel);
    plans.push({ ...place, lod, res, wraps, nMinX, nMaxX, nMinZ, nMaxZ });
  }
  return plans;
}

/**
 * Grow the COARSEST height level ("country floor") to span the tile-residency
 * coverage box, so every tile the tree can create has a resident REAL coarse
 * source — the S8c fix that makes "a frustum region with coarse data but flat/
 * absent terrain" unrepresentable. Before this, `levelRes` sized every level to
 * the FINEST-lod (pilot) footprint (~16 km), so the coarsest window reached only
 * ~33 km and every tile past it baked from the clamped window edge (a flat dead
 * plane that coarsen/eviction fell back onto — the one-direction "terrain
 * disappears on retreat" bug). The finer levels stay camera-windowed (mid-field
 * detail); only the coarsest becomes the whole-box floor. No-op when the coarsest
 * already spans the box (the generated world ⇒ bit-identical).
 */
function ensureFloorCoversBox(manifest: WorldManifest, height: PlanePlan[]): void {
  const geo = layerGeom(manifest, 'height');
  const { latMin, latMax } = coverageExtentLattice(manifest);
  const c = height[height.length - 1] as PlanePlan;
  const S = c.stride;
  // does the existing (square) coarsest window already cover [latMin,latMax]²?
  const covers = c.n0x * S <= latMin && (c.n0x + c.res) * S > latMax && c.n0z * S <= latMin && (c.n0z + c.res) * S > latMax;
  if (covers) return; // generated world (and any source whose floor already spans the box)
  const n0 = Math.floor(latMin / S);
  let res = 256;
  while ((n0 + res) * S <= latMax && res < HEIGHT_FLOOR_RES_CAP) res *= 2;
  if ((n0 + res) * S <= latMax) {
    // even at the res cap the coarsest LOD (texel c.texel m) can't STATICALLY span
    // the box: this source has no cookable country floor at this texel. Leave the
    // level as planned (it WRAPS — the far field rides the scrolling coarse window,
    // the pre-S8c behaviour) rather than allocate a giant fine plane. SURFACE it —
    // a real streamed world cooks coarse LODs so this never fires (Estonia's 256 m
    // L4 floors the 419 km box in one 2048² level).
    // eslint-disable-next-line no-console
    console.warn(
      `[laas] PlaneFill: coarsest height LOD (texel ${c.texel} m) too fine to statically floor the ` +
        `${Math.round(((latMax - latMin + 1) * geo.texel0) / 1000)} km box within res ${HEIGHT_FLOOR_RES_CAP} — ` +
        `far terrain rides the scrolling window (cook coarser LODs for a static floor)`,
    );
    return;
  }
  const off = S >> 1;
  height[height.length - 1] = {
    ...c,
    res,
    wraps: false, // spans the whole renderable domain ⇒ pinned (never scrolls)
    n0x: n0,
    n0z: n0,
    originX: geo.originX + (n0 * S + off + 0.5) * geo.texel0,
    originZ: geo.originZ + (n0 * S + off + 0.5) * geo.texel0,
  };
}

/** the whole field's plan — heights, biome, fields, water(+far). */
export function planField(manifest: WorldManifest): FieldPlan {
  const height = planLayer(manifest, 'height', HEIGHT_PLANE_RES, HEIGHT_PLANE_RES_CAP);
  if (height.length === 0) throw new Error('PlaneFill: source has no height layer');
  ensureFloorCoversBox(manifest, height);
  const biome = planLayer(manifest, 'biome', U8_PLANE_RES, U8_PLANE_RES_CAP);
  const fields = planLayer(manifest, 'fields', U8_PLANE_RES, U8_PLANE_RES_CAP);
  let water: PlanePlan | null = null;
  let waterFar: PlanePlan | null = null;
  const waterMeta = manifest.layers.water;
  if (waterMeta && waterMeta.lods.includes(0)) {
    const wplans = planLayer(manifest, 'water', HEIGHT_PLANE_RES, HEIGHT_PLANE_RES);
    water = wplans.find((p) => p.lod === 0) ?? null;
    if (water) {
      const farRes = water.res / WATER_FAR_FACTOR;
      waterFar = {
        ...water,
        res: farRes,
        texel: water.texel * WATER_FAR_FACTOR,
        // block (0,0) spans samples 0..7 — its representative point is their center
        originX: water.originX + (water.texel * (WATER_FAR_FACTOR - 1)) / 2,
        originZ: water.originZ + (water.texel * (WATER_FAR_FACTOR - 1)) / 2,
      };
    }
  }
  return { height, biome, fields, water, waterFar, coverageBox: coverageBoxM(manifest) };
}

// ---- region assembly (chunk payload → plane texels) -------------------------------

/** copy one height-kind chunk payload into a w×h region whose texel (0,0) is
 *  lattice sample (place.n0x, place.n0z); returns the union filled box. Skips
 *  the apron sample (the east/south neighbor owns it). */
export function copyChunkF32(
  out: Float32Array,
  w: number,
  h: number,
  place: LatticePlacement,
  geo: RasterGeom,
  ccx: number,
  ccz: number,
  src: Float32Array,
  srcRes: number,
  box: FilledBox | null,
  mapNaN?: number,
): FilledBox {
  const pxLo = Math.max(0, ccx * geo.chunkRes - place.n0x);
  const pxHi = Math.min(w - 1, (ccx + 1) * geo.chunkRes - 1 - place.n0x);
  const pzLo = Math.max(0, ccz * geo.chunkRes - place.n0z);
  const pzHi = Math.min(h - 1, (ccz + 1) * geo.chunkRes - 1 - place.n0z);
  for (let pz = pzLo; pz <= pzHi; pz++) {
    const i0 = (place.n0z + pz - ccz * geo.chunkRes) * srcRes + (place.n0x + pxLo - ccx * geo.chunkRes);
    const o0 = pz * w + pxLo;
    for (let k = 0; k <= pxHi - pxLo; k++) {
      const v = src[i0 + k] as number;
      out[o0 + k] = mapNaN !== undefined && Number.isNaN(v) ? mapNaN : v;
    }
  }
  if (pxLo > pxHi || pzLo > pzHi) return box ?? { x0: w, x1: -1, z0: h, z1: -1 };
  if (!box) return { x0: pxLo, x1: pxHi, z0: pzLo, z1: pzHi };
  return {
    x0: Math.min(box.x0, pxLo),
    x1: Math.max(box.x1, pxHi),
    z0: Math.min(box.z0, pzLo),
    z1: Math.max(box.z1, pzHi),
  };
}

/** copy one planes-kind chunk payload into an interleaved rgba8 w×h region. */
export function copyChunkU8(
  out: Uint8Array,
  w: number,
  h: number,
  place: LatticePlacement,
  geo: RasterGeom,
  ccx: number,
  ccz: number,
  planes: readonly Uint8Array[],
  srcRes: number,
  planeNames: readonly string[],
  channelMap: readonly (readonly [string, number])[],
  box: FilledBox | null,
): FilledBox {
  const pxLo = Math.max(0, ccx * geo.chunkRes - place.n0x);
  const pxHi = Math.min(w - 1, (ccx + 1) * geo.chunkRes - 1 - place.n0x);
  const pzLo = Math.max(0, ccz * geo.chunkRes - place.n0z);
  const pzHi = Math.min(h - 1, (ccz + 1) * geo.chunkRes - 1 - place.n0z);
  if (pxLo > pxHi || pzLo > pzHi) return box ?? { x0: w, x1: -1, z0: h, z1: -1 };
  for (const [name, ch] of channelMap) {
    const pi = planeNames.indexOf(name);
    if (pi < 0) continue;
    const src = planes[pi] as Uint8Array;
    for (let pz = pzLo; pz <= pzHi; pz++) {
      const i0 = (place.n0z + pz - ccz * geo.chunkRes) * srcRes + (place.n0x + pxLo - ccx * geo.chunkRes);
      const o0 = (pz * w + pxLo) * 4 + ch;
      for (let k = 0; k <= pxHi - pxLo; k++) out[o0 + k * 4] = src[i0 + k] as number;
    }
  }
  if (!box) return { x0: pxLo, x1: pxHi, z0: pzLo, z1: pzHi };
  return {
    x0: Math.min(box.x0, pxLo),
    x1: Math.max(box.x1, pxHi),
    z0: Math.min(box.z0, pzLo),
    z1: Math.max(box.z1, pzHi),
  };
}

/** extend the filled region's border texels to the w×h region rim (channels =
 *  words per texel: 1 for f32 planes, 4 for interleaved rgba8). */
export function clampExtend(
  data: Float32Array | Uint8Array,
  w: number,
  h: number,
  channels: number,
  box: FilledBox,
): void {
  if (box.x1 < box.x0 || box.z1 < box.z0) return;
  const row = w * channels;
  for (let z = box.z0; z <= box.z1; z++) {
    const r = z * row;
    for (let c = 0; c < channels; c++) {
      const lo = data[r + box.x0 * channels + c] as number;
      const hi = data[r + box.x1 * channels + c] as number;
      for (let x = 0; x < box.x0; x++) data[r + x * channels + c] = lo;
      for (let x = box.x1 + 1; x < w; x++) data[r + x * channels + c] = hi;
    }
  }
  for (let z = 0; z < box.z0; z++) data.copyWithin(z * row, box.z0 * row, box.z0 * row + row);
  for (let z = box.z1 + 1; z < h; z++) data.copyWithin(z * row, box.z1 * row, box.z1 * row + row);
}

/** ×factor min-reduce (far water: channels vanish, lakes survive). */
export function minReduce(src: Float32Array, res: number, factor: number): Float32Array {
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

/** chunk keys overlapping a w×h sample window at (n0x, n0z) of one lod (demand
 *  law: a level-k chunk is fetched iff it overlaps level k's window). */
export function chunksInWindow(
  geo: RasterGeom,
  lod: number,
  n0x: number,
  n0z: number,
  w: number,
  h = w,
): ChunkKey[] {
  const cx0 = Math.floor(n0x / geo.chunkRes);
  const cx1 = Math.floor((n0x + w - 1) / geo.chunkRes);
  const cz0 = Math.floor(n0z / geo.chunkRes);
  const cz1 = Math.floor((n0z + h - 1) / geo.chunkRes);
  const out: ChunkKey[] = [];
  for (let cz = cz0; cz <= cz1; cz++) for (let cx = cx0; cx <= cx1; cx++) out.push({ lod, cx, cz });
  return out;
}

// ---- toroidal scroll rect math ------------------------------------------------------

/** an axis-aligned texel rect in PHYSICAL (wrapped) plane coordinates. */
export interface PhysRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** split a LOGICAL window rect [lx0,lx0+w)×[lz0,lz0+h) (window-relative texels)
 *  into ≤4 physical rects under toroidal phase (n0 mod res). */
export function wrapRects(lx0: number, lz0: number, w: number, h: number, phaseX: number, phaseY: number, res: number): PhysRect[] {
  const px = (((lx0 + phaseX) % res) + res) % res;
  const py = (((lz0 + phaseY) % res) + res) % res;
  const xs: [number, number][] = px + w <= res ? [[px, w]] : [[px, res - px], [0, px + w - res]];
  const ys: [number, number][] = py + h <= res ? [[py, h]] : [[py, res - py], [0, py + h - res]];
  const out: PhysRect[] = [];
  for (const [y, hh] of ys) for (const [x, ww] of xs) out.push({ x, y, w: ww, h: hh });
  return out;
}
