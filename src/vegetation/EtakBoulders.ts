/**
 * EtakBoulders — ETAK survey boulder records → rock-library instances
 * (SPEC-ROCKS §H, SPEC-STREAMING-WORLD S7/S9). Pure/node-testable: no GPU/DOM.
 *
 * Record (asset-gen boulders cook, LAC1 layer 7): chunk-local x/z (m), kind u8
 * (0=single, 1=pile), size u8 (= size_m·40, clamp 8..255 ⇒ 0.2..6.4 m; raw 40/80
 * are usually the cook's 1 m/2 m DEFAULT — kind + positions are real survey data),
 * variant u8. size_m = size/40 (AG3-corrected: meters, not cm/40).
 *
 * Output rides the SAME instance pool as trees: flat {a,b} where A=(x,y,z,scale),
 * B=(yaw,leanX,leanZ,idF). leanX/leanZ = 0 (rocks sit, they don't lean). scale =
 * size_m / nominalRadius(class) so the rendered radius ≈ size_m; y = height − sink
 * (bedded 25–35 % on a slope-scaled bed). Piles are a DETERMINISTIC composition of
 * 6–14 fieldstone/flatCobble members (not a merged mesh) on a ring+top layout.
 */

import { VegClass } from '../gpu/passes/Scatter';
import { ETAK_ERRATIC_CLASS } from './RockGen';

const TAU = 6.2831853;

/** One chunk's ETAK boulder records, as they arrive on the decoded ChunkPayload
 *  (Lac1 maps the boulders layer onto the fixed record cols: species=kind,
 *  scale=size_m meters, variant). */
export interface EtakBoulderRecords {
  count: number;
  /** chunk-local meters (added to origin). */
  x: Float32Array;
  z: Float32Array;
  kind: Uint8Array; // 0 = single, 1 = pile  (payload cols.species)
  sizeM: Float32Array; // meters  (payload cols.scale — already size/40)
  variant: Uint8Array;
}

export interface EtakBoulderCfg {
  /** chunk world min-corner (game space). */
  originX: number;
  originZ: number;
  /** nominal (bound) radius per rock class — lib.clsRadius[cls]; scale = size_m/this. */
  radiusOf: (cls: number) => number;
  /** ground height sampler in GAME space (TerrainField mirror / height chunk). */
  heightAt: (wx: number, wz: number) => number;
}

/** the accumulating flat instance list (A/B word pairs; idF in b[·*4+3]). */
export interface EtakInstances {
  a: number[];
  b: number[];
}

/** class ladder by physical size (§H): <0.45 StoneM, <1.2 StoneL, <2.5 Boulder,
 *  ≥2.5 EtakErratic hero. */
function classForSize(sizeM: number): number {
  if (sizeM < 0.45) return VegClass.StoneM;
  if (sizeM < 1.2) return VegClass.StoneL;
  if (sizeM < 2.5) return VegClass.Boulder;
  return ETAK_ERRATIC_CLASS;
}

/** deterministic [0,1) hashes from integer coords (same mix family as ChunkContent). */
function hash2(px: number, pz: number): [number, number] {
  const M = 1664525;
  let a = (Math.imul(px >>> 0, M) + 1013904223) >>> 0;
  let b = (Math.imul(pz >>> 0, M) + 1013904223) >>> 0;
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

/** slope-scaled bed factor (a perched sphere on an incline embeds deeper). */
function bedAt(cfg: EtakBoulderCfg, wx: number, wz: number): number {
  const e = 0.75;
  const dhx = (cfg.heightAt(wx + e, wz) - cfg.heightAt(wx - e, wz)) / (2 * e);
  const dhz = (cfg.heightAt(wx, wz + e) - cfg.heightAt(wx, wz - e)) / (2 * e);
  const slope = Math.min(1, Math.hypot(dhx, dhz));
  return slope * 0.9 + 1;
}

function push(out: EtakInstances, x: number, y: number, z: number, scale: number, yaw: number, idF: number): void {
  out.a.push(x, y, z, scale);
  out.b.push(yaw, 0, 0, idF);
}

/** place one bedded single (or one pile member). */
function placeSingle(out: EtakInstances, cfg: EtakBoulderCfg, wx: number, wz: number, sizeM: number, cls: number, variant: number, salt: number, sinkFrac: number): void {
  const nomR = Math.max(0.1, cfg.radiusOf(cls));
  const scale = sizeM / nomR;
  const [hy, hs] = hash2((Math.round(wx * 4) ^ salt) >>> 0, (Math.round(wz * 4) ^ (salt << 1)) >>> 0);
  const bed = bedAt(cfg, wx, wz);
  const sink = sizeM * sinkFrac * (0.9 + hs * 0.2) * bed; // 25–35 % window × bed
  const y = cfg.heightAt(wx, wz) - sink;
  const idF = cls * 8 + (variant & 3);
  push(out, wx, y, wz, scale, hy * TAU, idF);
}

/**
 * Ingest one chunk's ETAK boulder records → flat rock instances. Singles bed on
 * the class ladder; piles compose 6–14 fieldstone(Boulder v2/3)/flatCobble(StoneM
 * v0/1) members on a footprint ring+top, mutual overlap allowed (reads as contact).
 */
export function ingestEtakBoulders(recs: EtakBoulderRecords, cfg: EtakBoulderCfg): EtakInstances {
  const out: EtakInstances = { a: [], b: [] };
  for (let i = 0; i < recs.count; i++) {
    const wx = cfg.originX + (recs.x[i] as number);
    const wz = cfg.originZ + (recs.z[i] as number);
    const sizeM = Math.max(0.1, recs.sizeM[i] as number);
    const variant = recs.variant[i] as number;
    if ((recs.kind[i] as number) === 0) {
      placeSingle(out, cfg, wx, wz, sizeM, classForSize(sizeM), variant, i * 2654435761, 0.28);
      continue;
    }
    // pile: 6–14 members on a ring of footprint max(1.5, 1.2·size_m), sunk 25–35 %
    const [hn, hr] = hash2((Math.round(wx) ^ 0x9e37) >>> 0, (Math.round(wz) ^ 0x79b9) >>> 0);
    const n = 6 + Math.floor(hn * 9); // 6..14
    const foot = Math.max(1.5, 1.2 * sizeM);
    for (let k = 0; k < n; k++) {
      const isTop = k === n - 1;
      const [ha, hb] = hash2((k * 2246822519) >>> 0, (i * 3266489917) >>> 0);
      const ang = (k / Math.max(1, n - 1)) * TAU + ha * 0.7;
      const rad = isTop ? foot * 0.15 : foot * (0.45 + hr * 0.35);
      const mx = wx + Math.cos(ang) * rad;
      const mz = wz + Math.sin(ang) * rad;
      const mSize = sizeM * (0.35 + hb * 0.3); // members are fractions of the pile size
      // fieldstone = Boulder variants 2/3; flatCobble = StoneM variants 0/1 — alternate by k
      const useBoulder = k % 3 === 0;
      const mCls = useBoulder ? VegClass.Boulder : VegClass.StoneM;
      const mVar = useBoulder ? 2 + (k & 1) : (k & 1);
      placeSingle(out, cfg, mx, mz, mSize, mCls, mVar, (i * 97 + k) >>> 0, 0.3);
    }
  }
  return out;
}
