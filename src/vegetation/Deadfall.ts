/**
 * Deadfall: fallen logs (3 decay states), stumps with root flare. Built on
 * the tube mesher with jagged broken caps; decay drives material moss/rot
 * via vdata.z (mossiness override channel on deadfall).
 */

import { Vector3 } from 'three';
import type { BufferGeometry } from 'three';
import type { Rng } from '../core/Seed';
import { MeshGrower, tubeForBranch } from './TubeMesh';
import type { SkelBranch } from './VegTypes';

export type DecayState = 'fresh' | 'mossy' | 'rotten';

export interface BuiltDeadfall {
  geometry: BufferGeometry;
  tris: number;
  /** ground-contact length for placement */
  length: number;
}

export function buildLog(rng: Rng, decay: DecayState): BuiltDeadfall {
  const g = new MeshGrower();
  const len = 2.6 + rng.float() * 2.6;
  const r0 = 0.16 + rng.float() * 0.16;
  const segs = 9;
  const pts: Vector3[] = [];
  const radii: number[] = [];
  const dirs: Vector3[] = [];
  // gentle ground-hugging curve with a slight rise at one end (root plate)
  const wob = rng.float() * Math.PI * 2;
  const squish = decay === 'rotten' ? 0.72 : 1;
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const sag = Math.sin(t * Math.PI) * 0.03;
    pts.push(
      new Vector3(
        (t - 0.5) * len,
        (r0 * 0.82 - sag + Math.sin(t * 7 + wob) * 0.015) * squish,
        Math.sin(t * 2.2 + wob) * len * 0.035,
      ),
    );
    const taper = decay === 'rotten' ? 0.88 : 0.94;
    radii.push(r0 * (1 - t * (1 - taper)) * (1 + Math.sin(t * 13 + wob) * 0.05));
    dirs.push(new Vector3(1, 0, Math.cos(t * 2.2 + wob) * 0.08).normalize());
  }
  const br: SkelBranch = { level: 0, pts, radii, dirs, len, tParent: 0, broken: true, parentIdx: -1 };
  tubeForBranch(
    g, br,
    {
      ringSegs: 13,
      uRepeats: 3,
      vScale: 1,
      capBase: true, // free-lying: both ends visible
      swayPhase: 0,
      swayFlexBase: 0,
      swayFlexTip: 0,
      hue: rng.float() * 2 - 1,
    },
    rng,
  );
  // mossiness/rot channel: overwrite vdata.z per decay
  const mossK = decay === 'fresh' ? 0.15 : decay === 'mossy' ? 0.8 : 1.0;
  const geo = g.build();
  const dat = geo.getAttribute('vdata');
  for (let i = 0; i < dat.count; i++) dat.setZ(i, mossK);
  return { geometry: geo, tris: g.triCount, length: len };
}

export function buildStump(rng: Rng): BuiltDeadfall {
  const g = new MeshGrower();
  const h = 0.5 + rng.float() * 0.6;
  const r0 = 0.22 + rng.float() * 0.14;
  const segs = 5;
  const pts: Vector3[] = [];
  const radii: number[] = [];
  const dirs: Vector3[] = [];
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    pts.push(new Vector3((rng.float() - 0.5) * 0.02, t * h, (rng.float() - 0.5) * 0.02));
    radii.push(r0 * (1 - t * 0.18));
    dirs.push(new Vector3(0, 1, 0));
  }
  const br: SkelBranch = { level: 0, pts, radii, dirs, len: h, tParent: 0, broken: true, parentIdx: -1 };
  tubeForBranch(
    g, br,
    {
      ringSegs: 14,
      uRepeats: 3,
      vScale: 1,
      flare: { amp: 0.85, height: h * 0.55, lobes: 5, phase: rng.float() * 6.28 },
      swayPhase: 0,
      swayFlexBase: 0,
      swayFlexTip: 0,
      hue: rng.float() * 2 - 1,
    },
    rng,
  );
  const geo = g.build();
  const dat = geo.getAttribute('vdata');
  for (let i = 0; i < dat.count; i++) dat.setZ(i, 0.5 + rng.float() * 0.3);
  return { geometry: geo, tris: g.triCount, length: h };
}
