/**
 * GRASS PATCH library (31-grass-plan S2): a ~4×4 m patch of blade clumps merged
 * into ONE mesh, registered as a matClass-5 nanite instance with an aggregate
 * DAG (remove-whole-blades + grow-survivors — BuildAggregateDag treats each
 * blade strip as an atomic island, so coarser DAG levels are the ring's own
 * thin×widen conservation, derived instead of hand-authored).
 *
 * DENSITY IS THE LAW (memory grass-lushness-law): the shipped GroundRing look
 * is ~90 clump slots/m² near-field, 5-blade clumps, blade overlap — the patch
 * bakes THAT density at L0. Perf comes from the DAG cut + the election, never
 * from thinning below the ring's coverage.
 *
 * Blades are baked WORLD-size (heights ~0.14–0.55 m like the ring's composed
 * scale) so patch instances carry scale 1; per-clump variety (yaw, height,
 * lean, dryness-driving position) is baked by a deterministic mini-rng keyed
 * on the variant seed — patch variants tile without visible repetition because
 * the resolve's albedo dryness/brightness are WORLD-anchored hashes, not baked.
 */

import { BufferAttribute, BufferGeometry } from 'three';
import { grassBladeGeometry } from './GroundCover';

export const GRASS_PATCH_SIZE = 4; // m — instance granularity for the hier cull
export const GRASS_CLUMPS_PER_M2 = 90; // the ring's near-band slot density
export const GRASS_PATCH_VARIANTS = 4;

/** one ~4×4 m grass patch mesh at FULL near-band density (≈1440 clumps, ~50k tris) */
export function grassPatchGeometry(variantSeed: number, size = GRASS_PATCH_SIZE): BufferGeometry {
  let s = (0x9e3779b9 ^ (variantSeed * 2654435761)) >>> 0;
  const rnd = (): number => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  // base blade (unit height, ±38° rounded-cross-section normals, uv.y tip ramp)
  const base = grassBladeGeometry(4);
  const bp = base.attributes.position as BufferAttribute;
  const bn = base.attributes.normal as BufferAttribute;
  const bu = base.attributes.uv as BufferAttribute;
  const bi = base.index as BufferAttribute;
  const clumps = Math.round(size * size * GRASS_CLUMPS_PER_M2);
  const BLADES = 5;
  const vPerBlade = bp.count;
  const tPerBlade = bi.count;
  const totalV = clumps * BLADES * vPerBlade;
  const pos = new Float32Array(totalV * 3);
  const nrm = new Float32Array(totalV * 3);
  const uvA = new Float32Array(totalV * 2);
  // vdata (4×u8 through geometryToSource, survives the DAG in slots 8-11):
  // x/y = the blade's ROOT xz within the patch (0..1 over size) — the grass
  // channel conforms the WHOLE blade to the terrain height under its root
  // (per-vertex sampling at displaced xz sheared blades along slopes and fed
  // wind motion back into height — the ravine-wall glitch, 2026-07-03);
  // z = per-blade wind phase (the whole patch waving on one hoisted sine read
  // as marching-band repetition); w spare (flex, later).
  const vdat = new Float32Array(totalV * 4);
  const idx = new Uint32Array(clumps * BLADES * tPerBlade);
  let vw = 0;
  let iw = 0;
  for (let ci = 0; ci < clumps; ci++) {
    // clump anchor: jittered grid keeps coverage even (no bald cells)
    const g = Math.ceil(Math.sqrt(clumps));
    const gx = ci % g;
    const gz = Math.floor(ci / g);
    const cx = ((gx + rnd()) / g) * size;
    const cz = ((gz + rnd()) / g) * size;
    // clump height family (the ring's c.h × per-blade composition → ~0.14–0.55 m)
    const ch = 0.55 + rnd() * 0.75;
    for (let b = 0; b < BLADES; b++) {
      const yaw = rnd() * Math.PI * 2;
      const c = Math.cos(yaw);
      const sn = Math.sin(yaw);
      const ox = cx + (rnd() - 0.5) * 0.16;
      const oz = cz + (rnd() - 0.5) * 0.16;
      const hk = ch * (0.55 + rnd() * 0.7) * 0.42;
      const lean = (rnd() - 0.5) * 0.42;
      const rootX = Math.max(0, Math.min(1, ox / size));
      const rootZ = Math.max(0, Math.min(1, oz / size));
      const phase = rnd();
      const v0 = vw;
      for (let i = 0; i < vPerBlade; i++) {
        const x = bp.getX(i) * 1.25;
        const y = bp.getY(i) * hk;
        const z = bp.getZ(i);
        pos[vw * 3] = x * c + z * sn + ox + lean * y * c;
        pos[vw * 3 + 1] = y;
        pos[vw * 3 + 2] = z * c - x * sn + oz + lean * y * sn;
        nrm[vw * 3] = bn.getX(i) * c + bn.getZ(i) * sn;
        nrm[vw * 3 + 1] = bn.getY(i);
        nrm[vw * 3 + 2] = bn.getZ(i) * c - bn.getX(i) * sn;
        uvA[vw * 2] = bu.getX(i);
        uvA[vw * 2 + 1] = bu.getY(i);
        vdat[vw * 4] = rootX;
        vdat[vw * 4 + 1] = rootZ;
        vdat[vw * 4 + 2] = phase;
        vdat[vw * 4 + 3] = 0.85;
        vw++;
      }
      for (let i = 0; i < tPerBlade; i++) idx[iw++] = v0 + bi.getX(i);
    }
  }
  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(pos, 3));
  geo.setAttribute('normal', new BufferAttribute(nrm, 3));
  geo.setAttribute('uv', new BufferAttribute(uvA, 2));
  geo.setAttribute('vdata', new BufferAttribute(vdat, 4));
  geo.setIndex(new BufferAttribute(idx, 1));
  return geo;
}
