/**
 * Mesh assembly helpers. MeshGrower accumulates one big indexed buffer
 * (position/normal/uv/vdata) — every generator appends into it, so a whole
 * asset is 1–2 draw calls. TubeMesh turns skeleton branches into generalized
 * cylinders via parallel-transport frames, with root flare/buttresses on the
 * trunk and jagged caps on broken branches.
 *
 * vdata layout (vec4, consumed by VegMaterials):
 *   x: hue jitter (−1..1)   y: sway flexibility (0 rigid .. 1 tip)
 *   z: sway phase (0..2π)   w: baked AO (0 dark .. 1 open)
 */

import { BufferAttribute, BufferGeometry, Vector3 } from 'three';
import type { Rng } from '../core/Seed';
import type { SkelBranch, Skeleton } from './VegTypes';

export class MeshGrower {
  private pos: number[] = [];
  private nrm: number[] = [];
  private uv: number[] = [];
  private dat: number[] = [];
  private idx: number[] = [];
  vertCount = 0;

  vertex(
    px: number, py: number, pz: number,
    nx: number, ny: number, nz: number,
    u: number, v: number,
    d0: number, d1: number, d2: number, d3: number,
  ): number {
    this.pos.push(px, py, pz);
    this.nrm.push(nx, ny, nz);
    this.uv.push(u, v);
    this.dat.push(d0, d1, d2, d3);
    return this.vertCount++;
  }

  tri(a: number, b: number, c: number): void {
    this.idx.push(a, b, c);
  }

  quad(a: number, b: number, c: number, d: number): void {
    this.idx.push(a, b, c, a, c, d);
  }

  get triCount(): number {
    return this.idx.length / 3;
  }

  /** blend normals toward a sphere around `center` (foliage cohesion trick) */
  bendNormals(center: Vector3, radius: number, k: number, fromVert = 0): void {
    const inv = 1 / Math.max(0.001, radius);
    for (let i = fromVert; i < this.vertCount; i++) {
      const px = this.pos[i * 3] as number;
      const py = this.pos[i * 3 + 1] as number;
      const pz = this.pos[i * 3 + 2] as number;
      let sx = (px - center.x) * inv;
      let sy = (py - center.y) * inv;
      let sz = (pz - center.z) * inv;
      const sl = Math.hypot(sx, sy, sz) || 1;
      sx /= sl; sy /= sl; sz /= sl;
      const nx = (this.nrm[i * 3] as number) * (1 - k) + sx * k;
      const ny = (this.nrm[i * 3 + 1] as number) * (1 - k) + sy * k;
      const nz = (this.nrm[i * 3 + 2] as number) * (1 - k) + sz * k;
      const l = Math.hypot(nx, ny, nz) || 1;
      this.nrm[i * 3] = nx / l;
      this.nrm[i * 3 + 1] = ny / l;
      this.nrm[i * 3 + 2] = nz / l;
    }
  }

  /** depth-in-crown AO: vdata.w *= darkening for verts inside the crown hull */
  crownAO(center: Vector3, radius: number, strength: number, fromVert = 0): void {
    const inv = 1 / Math.max(0.001, radius);
    for (let i = fromVert; i < this.vertCount; i++) {
      const dx = ((this.pos[i * 3] as number) - center.x) * inv;
      const dy = ((this.pos[i * 3 + 1] as number) - center.y) * inv;
      const dz = ((this.pos[i * 3 + 2] as number) - center.z) * inv;
      const d = Math.min(1, Math.hypot(dx, dy, dz));
      const ao = 1 - strength * (1 - d) * (1 - d);
      this.dat[i * 4 + 3] = (this.dat[i * 4 + 3] as number) * ao;
    }
  }

  build(): BufferGeometry {
    const g = new BufferGeometry();
    g.setAttribute('position', new BufferAttribute(new Float32Array(this.pos), 3));
    g.setAttribute('normal', new BufferAttribute(new Float32Array(this.nrm), 3));
    g.setAttribute('uv', new BufferAttribute(new Float32Array(this.uv), 2));
    g.setAttribute('vdata', new BufferAttribute(new Float32Array(this.dat), 4));
    g.setIndex(
      this.vertCount > 65535
        ? new BufferAttribute(new Uint32Array(this.idx), 1)
        : new BufferAttribute(new Uint16Array(this.idx), 1),
    );
    g.computeBoundingSphere();
    return g;
  }
}

export interface TubeOpts {
  /** ring vertex count at the branch base (tapers down along the branch) */
  ringSegs: number;
  /** around-tube texture repeats at the base */
  uRepeats: number;
  /** lengthwise texture scale (v per meter ≈ uRepeats / circumference) */
  vScale: number;
  /** trunk-only root flare */
  flare?: { amp: number; height: number; lobes: number; phase: number };
  /** per-branch sway phase + flexibility for vdata */
  swayPhase: number;
  swayFlexBase: number;
  swayFlexTip: number;
  hue: number;
}

const _N = new Vector3();
const _B = new Vector3();
const _T = new Vector3();
const _v = new Vector3();

/** generalized cylinder along a skeleton branch via parallel transport */
export function tubeForBranch(
  g: MeshGrower,
  br: SkelBranch,
  opts: TubeOpts,
  rng: Rng,
): void {
  const n = br.pts.length;
  if (n < 2) return;
  const rings: number[][] = [];
  let lastRingPos: number[] = [];
  // initial frame
  _T.copy(br.dirs[0] as Vector3);
  const ref = Math.abs(_T.y) < 0.94 ? new Vector3(0, 1, 0) : new Vector3(1, 0, 0);
  _N.crossVectors(ref, _T).normalize();
  _B.crossVectors(_T, _N).normalize();

  const segsAround = Math.max(4, opts.ringSegs);
  let vAlong = 0;
  const baseR = Math.max(br.radii[0] as number, 1e-4);

  for (let i = 0; i < n; i++) {
    const p = br.pts[i] as Vector3;
    const r = br.radii[i] as number;
    if (i > 0) {
      const prev = br.pts[i - 1] as Vector3;
      vAlong += _v.subVectors(p, prev).length();
      // parallel transport: rotate N,B by the rotation prev-tangent → tangent
      const tPrev = br.dirs[i - 1] as Vector3;
      const tCur = br.dirs[i] as Vector3;
      const axis = _v.crossVectors(tPrev, tCur);
      const s = axis.length();
      if (s > 1e-6) {
        axis.multiplyScalar(1 / s);
        const ang = Math.asin(Math.min(1, s));
        _N.applyAxisAngle(axis, ang).normalize();
        _B.applyAxisAngle(axis, ang).normalize();
      }
    }
    const tt = i / (n - 1);
    // taper slope tilts ring normals toward the tangent
    const rNext = br.radii[Math.min(n - 1, i + 1)] as number;
    const rPrev = br.radii[Math.max(0, i - 1)] as number;
    const slope = (rPrev - rNext) * (n - 1) / Math.max(0.05, br.len) * 0.5;
    const ring: number[] = [];
    const ringPos: number[] = [];
    const ao = 1; // bark AO baked later via crownAO/groundAO passes
    const flex = opts.swayFlexBase + (opts.swayFlexTip - opts.swayFlexBase) * tt;
    for (let k = 0; k <= segsAround; k++) {
      const a = (k / segsAround) * Math.PI * 2;
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      let rr = r;
      if (opts.flare && br.level === 0) {
        const h = (br.pts[i] as Vector3).y - (br.pts[0] as Vector3).y;
        const lobe = Math.pow(
          Math.max(0, Math.cos(opts.flare.lobes * a + opts.flare.phase)),
          1.6,
        );
        rr *= 1 + opts.flare.amp * Math.exp(-h / opts.flare.height) * (0.45 + 0.9 * lobe);
      }
      const dx = _N.x * ca + _B.x * sa;
      const dy = _N.y * ca + _B.y * sa;
      const dz = _N.z * ca + _B.z * sa;
      const tan = br.dirs[i] as Vector3;
      let nx = dx + tan.x * slope;
      let ny = dy + tan.y * slope;
      let nz = dz + tan.z * slope;
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx /= nl; ny /= nl; nz /= nl;
      ringPos.push(p.x + dx * rr, p.y + dy * rr, p.z + dz * rr);
      ring.push(
        g.vertex(
          p.x + dx * rr, p.y + dy * rr, p.z + dz * rr,
          nx, ny, nz,
          (k / segsAround) * opts.uRepeats,
          (vAlong / (Math.PI * 2 * baseR)) * opts.uRepeats * opts.vScale,
          opts.hue, flex, opts.swayPhase, ao,
        ),
      );
    }
    rings.push(ring);
    lastRingPos = ringPos;
  }

  for (let i = 0; i < rings.length - 1; i++) {
    const a = rings[i] as number[];
    const b = rings[i + 1] as number[];
    for (let k = 0; k < segsAround; k++) {
      g.quad(a[k] as number, b[k] as number, b[k + 1] as number, a[k + 1] as number);
    }
  }

  // cap
  const last = rings[rings.length - 1] as number[];
  const tipP = br.pts[n - 1] as Vector3;
  const tipD = br.dirs[n - 1] as Vector3;
  const tipR = br.radii[n - 1] as number;
  if (br.broken && tipR > 0.015) {
    // jagged break: ring of inward spikes at randomized heights
    const center = g.vertex(
      tipP.x + tipD.x * tipR * 0.4,
      tipP.y + tipD.y * tipR * 0.4,
      tipP.z + tipD.z * tipR * 0.4,
      tipD.x, tipD.y, tipD.z,
      0.5, 0.5, opts.hue, opts.swayFlexTip, opts.swayPhase, 0.55,
    );
    const jag: number[] = [];
    for (let k = 0; k <= segsAround; k++) {
      const px = tipP.x + ((lastRingPos[k * 3] as number) - tipP.x) * 0.45;
      const py = tipP.y + ((lastRingPos[k * 3 + 1] as number) - tipP.y) * 0.45;
      const pz = tipP.z + ((lastRingPos[k * 3 + 2] as number) - tipP.z) * 0.45;
      const spike = (rng.float() * 0.9 + 0.25) * tipR * 1.4;
      jag.push(
        g.vertex(
          px + tipD.x * spike, py + tipD.y * spike, pz + tipD.z * spike,
          tipD.x, tipD.y, tipD.z,
          0.5, 0.5, opts.hue, opts.swayFlexTip, opts.swayPhase, 0.5,
        ),
      );
    }
    for (let k = 0; k < segsAround; k++) {
      g.quad(last[k] as number, jag[k] as number, jag[k + 1] as number, last[k + 1] as number);
      g.tri(jag[k] as number, center, jag[k + 1] as number);
    }
  } else {
    // taper to a point
    const tip = g.vertex(
      tipP.x + tipD.x * tipR * 2.0,
      tipP.y + tipD.y * tipR * 2.0,
      tipP.z + tipD.z * tipR * 2.0,
      tipD.x, tipD.y, tipD.z,
      0.5, vAlong / (Math.PI * 2 * baseR) + 0.2,
      opts.hue, opts.swayFlexTip, opts.swayPhase, 1,
    );
    for (let k = 0; k < segsAround; k++) {
      g.tri(last[k] as number, tip, last[k + 1] as number);
    }
  }
}

/** ring resolution by branch level (LOD scales these down) */
export function ringsForLevel(level: number, lodK: number): number {
  const base = level === 0 ? 14 : level === 1 ? 8 : level === 2 ? 6 : 5;
  return Math.max(4, Math.round(base * lodK));
}

/** mesh every branch of a skeleton into the grower */
export function tubesForSkeleton(
  g: MeshGrower,
  skel: Skeleton,
  rng: Rng,
  opts: {
    lodK: number;
    uRepeats: number;
    flare?: { amp: number; height: number; lobes: number; phase: number };
    /** skip branches at or above this level (LOD cut) */
    maxLevel?: number;
    /** keep only every Nth branch of level ≥ 1 (far-LOD bark diet) */
    branchStride?: number;
  },
): void {
  const maxLevel = opts.maxLevel ?? 99;
  const stride = opts.branchStride ?? 1;
  let bi = 0;
  for (const br of skel.branches) {
    if (br.level > maxLevel) continue;
    if (br.level >= 1 && stride > 1 && bi++ % stride !== 0) continue;
    // sway: trunk rigid, outer levels flexible
    const flexB = br.level === 0 ? 0 : br.level === 1 ? 0.12 : 0.3;
    const flexT = br.level === 0 ? 0.05 : br.level === 1 ? 0.35 : 0.7;
    tubeForBranch(
      g,
      br,
      {
        ringSegs: ringsForLevel(br.level, opts.lodK),
        uRepeats: br.level === 0 ? opts.uRepeats : Math.max(1, Math.round(opts.uRepeats * 0.4)),
        vScale: 1,
        ...(br.level === 0 && opts.flare ? { flare: opts.flare } : {}),
        swayPhase: rng.float() * Math.PI * 2,
        swayFlexBase: flexB,
        swayFlexTip: flexT,
        hue: rng.float() * 2 - 1,
      },
      rng,
    );
  }
}
