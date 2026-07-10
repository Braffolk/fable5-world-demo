/**
 * ChunkContent — the ONE placement consumer (SPEC-STREAMING-WORLD §2/§7, S2).
 * Record payloads from ANY WorldSource become the per-species instance streams
 * {a,b,fill} that the boot bindInstances bridge (WorldRegistry) consumes — the
 * subsystem never knows which source fed it (law 3).
 *
 * Word layout (Scatter.ts contract): A = (x, y, z, scale), B = (yaw, leanX, leanZ, idF).
 *
 * Exactness columns present (generated source) → used VERBATIM: instance words are
 * byte-identical to the direct scatter path (the S2 gate). Absent (Estonia) → the ONE
 * derive rule, shared with nothing else:
 *   y    = height sample at (x,z) − scale·0.12 sink (trees only — root flare seam)
 *   yaw  = pcg2d hash(cx,cz, xq,zq) · τ         (xq/zq = u16-quantized local coords,
 *          the wire lattice — stable across refetches)
 *   lean = slope-normal.xz · 0.18 + (hash − ½)·0.12   (Scatter.ts:469-470 formula)
 */
import type { ChunkKey, ChunkPayload, LayerName, WorldManifest, WorldSource } from '../../world/source/WorldSource';

export interface InstanceStream {
  a: Float32Array;
  b: Float32Array;
  fill: number;
}

export interface ChunkContentStreams {
  perId: Map<number, InstanceStream>;
  total: number;
}

/** record layers a source may serve, in the canonical stream-build order. */
const RECORD_LAYER_ORDER: readonly LayerName[] = ['trees', 'understory', 'extras', 'stones', 'boulders'];

const TAU = 6.2831853;

/**
 * Build the boot instance streams: every record chunk of every record layer the
 * manifest declares, chunks in row-major (cz,cx) order, in-chunk scatter order
 * preserved. idFOf maps (species, variant) → geometry-pool idF; the default is the
 * generated source's identity rule (species IS the VegClass, idF = cls·8 + variant).
 * Estonia's SpeciesMap resolver replaces it at S7.
 */
export async function buildChunkContentStreams(
  source: WorldSource,
  manifest: WorldManifest,
  idFOf: (species: number, variant: number) => number = (s, v) => s * 8 + v,
): Promise<ChunkContentStreams> {
  const jobs: { layer: LayerName; key: ChunkKey; payload: ChunkPayload & { kind: 'records' } }[] = [];
  for (const layer of RECORD_LAYER_ORDER) {
    if (!manifest.layers[layer]) continue;
    const keys = manifest.chunks(layer, 0).sort((p, q) => p.cz - q.cz || p.cx - q.cx);
    for (const key of keys) {
      const payload = await source.fetch(layer, key);
      if (!payload) continue;
      if (payload.kind !== 'records') throw new Error(`ChunkContent: ${layer} chunk is not records`);
      jobs.push({ layer, key, payload });
    }
  }

  // two passes: exact per-id sizes, then fill (same shape the registry pools expect)
  const counts = new Map<number, number>();
  for (const { payload } of jobs) {
    const { species, variant } = payload.cols;
    for (let i = 0; i < payload.count; i++) {
      const id = idFOf(species[i] as number, variant[i] as number);
      counts.set(id, (counts.get(id) ?? 0) + 1);
    }
  }
  const perId = new Map<number, InstanceStream>();
  for (const [id, n] of counts) perId.set(id, { a: new Float32Array(n * 4), b: new Float32Array(n * 4), fill: 0 });

  let total = 0;
  for (const { layer, key, payload } of jobs) {
    const { cols, count } = payload;
    const footprint = manifest.grid.chunkMeters * manifest.grid.lodStep ** key.lod;
    const minX = manifest.grid.originX + key.cx * footprint;
    const minZ = manifest.grid.originZ + key.cz * footprint;
    const derive = cols.y && cols.yaw && cols.leanX && cols.leanZ ? null : await makeDeriver(source, manifest, key);
    for (let i = 0; i < count; i++) {
      const id = idFOf(cols.species[i] as number, cols.variant[i] as number);
      const s = perId.get(id) as InstanceStream;
      const d = s.fill * 4;
      const x = cols.xw ? (cols.xw[i] as number) : minX + (cols.x[i] as number);
      const z = cols.zw ? (cols.zw[i] as number) : minZ + (cols.z[i] as number);
      const scale = cols.scale[i] as number;
      s.a[d] = x;
      s.a[d + 2] = z;
      s.a[d + 3] = scale;
      s.b[d + 3] = id;
      if (derive) {
        const dv = derive(cols.x[i] as number, cols.z[i] as number);
        s.a[d + 1] = cols.y ? (cols.y[i] as number) : dv.h - (layer === 'trees' ? scale * 0.12 : 0);
        s.b[d] = cols.yaw ? (cols.yaw[i] as number) : dv.yaw;
        s.b[d + 1] = cols.leanX ? (cols.leanX[i] as number) : dv.leanX;
        s.b[d + 2] = cols.leanZ ? (cols.leanZ[i] as number) : dv.leanZ;
      } else {
        s.a[d + 1] = (cols.y as Float32Array)[i] as number;
        s.b[d] = (cols.yaw as Float32Array)[i] as number;
        s.b[d + 1] = (cols.leanX as Float32Array)[i] as number;
        s.b[d + 2] = (cols.leanZ as Float32Array)[i] as number;
      }
      s.fill++;
      total++;
    }
  }
  return { perId, total };
}

// --- derive-if-absent --------------------------------------------------------------------

type Deriver = (xLocal: number, zLocal: number) => { h: number; yaw: number; leanX: number; leanZ: number };

/** grounding + cosmetic-orientation deriver for one chunk: fetches the chunk's height
 *  window once, samples bilinearly on the texel-centered lattice, slope via central
 *  differences. Estonia trees lean on slopes exactly like generated ones (one rule). */
async function makeDeriver(source: WorldSource, manifest: WorldManifest, key: ChunkKey): Promise<Deriver> {
  const payload = await source.fetch('height', key);
  if (!payload || payload.kind !== 'height') {
    throw new Error(`ChunkContent: no height chunk (${key.lod},${key.cx},${key.cz}) to ground records on`);
  }
  const { res, heights } = payload;
  const footprint = manifest.grid.chunkMeters * manifest.grid.lodStep ** key.lod;
  const texel = footprint / (res - 1);
  const sample = (gx: number, gz: number): number => {
    const cgx = Math.min(Math.max(gx, 0), res - 1.001);
    const cgz = Math.min(Math.max(gz, 0), res - 1.001);
    const x0 = Math.floor(cgx);
    const z0 = Math.floor(cgz);
    const fx = cgx - x0;
    const fz = cgz - z0;
    const at = (xx: number, zz: number): number => heights[Math.min(zz, res - 1) * res + Math.min(xx, res - 1)] as number;
    return (
      at(x0, z0) * (1 - fx) * (1 - fz) +
      at(x0 + 1, z0) * fx * (1 - fz) +
      at(x0, z0 + 1) * (1 - fx) * fz +
      at(x0 + 1, z0 + 1) * fx * fz
    );
  };
  return (xLocal, zLocal) => {
    const gx = xLocal / texel - 0.5; // texel (i,j) centers at (i+0.5)·texel
    const gz = zLocal / texel - 0.5;
    const h = sample(gx, gz);
    // slope normal: central differences, normalized (-dh/dx, 1, -dh/dz)
    const dhdx = (sample(gx + 1, gz) - sample(gx - 1, gz)) / (2 * texel);
    const dhdz = (sample(gx, gz + 1) - sample(gx, gz - 1)) / (2 * texel);
    const inv = 1 / Math.hypot(dhdx, 1, dhdz);
    const xq = Math.min(65535, Math.max(0, Math.round((xLocal / footprint) * 65535)));
    const zq = Math.min(65535, Math.max(0, Math.round((zLocal / footprint) * 65535)));
    const [h1, h2] = pcg2d(((key.cx & 0x7fff) << 16) ^ xq, ((key.cz & 0x7fff) << 16) ^ zq);
    return {
      h,
      yaw: h1 * TAU,
      leanX: -dhdx * inv * 0.18 + (h2 - 0.5) * 0.12,
      leanZ: -dhdz * inv * 0.18 + (pcg2d(xq ^ 0x5b1e, zq ^ 0x2c9d)[0] - 0.5) * 0.12,
    };
  };
}

/** CPU pcg2d, same mix as gpu/passes/Scatter.ts pcg2d — returns two [0,1) floats. */
function pcg2d(px: number, pz: number): [number, number] {
  const M = 1664525;
  let a = Math.imul(px >>> 0, M) + 1013904223;
  let b = Math.imul(pz >>> 0, M) + 1013904223;
  a = (a + Math.imul(b, M)) >>> 0;
  b = (b + Math.imul(a, M)) >>> 0;
  a = (a ^ (a >>> 16)) >>> 0;
  b = (b ^ (b >>> 16)) >>> 0;
  a = (a + Math.imul(b, M)) >>> 0;
  b = (b + Math.imul(a, M)) >>> 0;
  a = (a ^ (a >>> 16)) >>> 0;
  b = (b ^ (b >>> 16)) >>> 0;
  return [(a & 0xffffff) / 16777216, (b & 0xffffff) / 16777216];
}
