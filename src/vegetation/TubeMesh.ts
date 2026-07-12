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
import { barkMacro, type BarkFieldParams } from './BarkField';
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
  /** jagged cap over ring 0 — tubes historically had NO start cap (invisible
   *  on branches attached to a parent; an open hole on free-lying deadfall) */
  capBase?: boolean;
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
  let firstRingPos: number[] = [];
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
    if (i === 0) firstRingPos = ringPos;
  }

  // winding: rings are built on (N, B=T×N) — increasing angle is CCW viewed
  // from −T, so quads must run base-ring-first to put front faces OUTWARD
  // (the old b-first order rendered tube interiors on FrontSide materials)
  for (let i = 0; i < rings.length - 1; i++) {
    const a = rings[i] as number[];
    const b = rings[i + 1] as number[];
    for (let k = 0; k < segsAround; k++) {
      g.quad(a[k] as number, a[k + 1] as number, b[k + 1] as number, b[k] as number);
    }
  }

  // base cap (free-lying pieces): jagged disc facing −T0. Winding note: the
  // cap advances along −T, which flips handedness vs the wall quads — the
  // outward order here is the MIRROR of the tip-cap order.
  if (opts.capBase && baseR > 0.015) {
    const baseP = br.pts[0] as Vector3;
    const baseD = br.dirs[0] as Vector3;
    const first = rings[0] as number[];
    const center = g.vertex(
      baseP.x - baseD.x * baseR * 0.4,
      baseP.y - baseD.y * baseR * 0.4,
      baseP.z - baseD.z * baseR * 0.4,
      -baseD.x, -baseD.y, -baseD.z,
      0.5, 0.5, opts.hue, opts.swayFlexBase, opts.swayPhase, 0.55,
    );
    const jag: number[] = [];
    for (let k = 0; k <= segsAround; k++) {
      const px = baseP.x + ((firstRingPos[k * 3] as number) - baseP.x) * 0.45;
      const py = baseP.y + ((firstRingPos[k * 3 + 1] as number) - baseP.y) * 0.45;
      const pz = baseP.z + ((firstRingPos[k * 3 + 2] as number) - baseP.z) * 0.45;
      const spike = (rng.float() * 0.9 + 0.25) * baseR * 1.4;
      jag.push(
        g.vertex(
          px - baseD.x * spike, py - baseD.y * spike, pz - baseD.z * spike,
          -baseD.x, -baseD.y, -baseD.z,
          0.5, 0.5, opts.hue, opts.swayFlexBase, opts.swayPhase, 0.5,
        ),
      );
    }
    for (let k = 0; k < segsAround; k++) {
      g.quad(first[k] as number, jag[k] as number, jag[k + 1] as number, first[k + 1] as number);
      g.tri(jag[k] as number, center, jag[k + 1] as number);
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
      g.quad(last[k] as number, last[k + 1] as number, jag[k + 1] as number, jag[k] as number);
      g.tri(jag[k + 1] as number, center, jag[k] as number);
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
      g.tri(last[k + 1] as number, tip, last[k] as number);
    }
  }
}

/** ring resolution by branch level (LOD scales these down) */
function ringsForLevel(level: number, lodK: number): number {
  const base = level === 0 ? 14 : level === 1 ? 8 : level === 2 ? 6 : 5;
  return Math.max(4, Math.round(base * lodK));
}

// ───────────────────────── junction-aware bark meshing ───────────────────────
//
// The legacy path meshed every branch INDEPENDENTLY: each branch's base ring sat
// on the parent centerline, buried, and OPEN. An open ring is an edge loop used
// by ≠2 triangles, which the QEM simplifier LOCKS (BuildDag.ts:393-422) — so the
// far-field never collapsed (~46M stuck sub-pixel tris).
//
// This path makes the whole bark skin ONE connected closed manifold:
//   • trunk base  → smooth fan disc (buried; the only legit cap — no parent).
//   • every branch → a HOLE is stencilled in the parent wall and the branch's
//     flared MOUTH ring is ZIPPER-WELDED (shared vertex IDs) to that hole rim.
//   • degenerate junctions (fork / childR≈parentR / overlap / coarse grid) fall
//     back to a closed buried collar cap — still 0 open edges, perf preserved.
//   • tips stay capped (taper / jagged-break) exactly as before.
// Result: 0 open-boundary edges ⇒ QEM collapses each tree to ~1 root.
//
// Determinism: this pass draws ZERO rng (pure functions of stored skeleton
// fields); the per-branch swayPhase/hue/broken-tip draws keep their order, so a
// twin build is bit-identical (probe-dag determinism gate).

interface EmittedRing {
  /** length segsAround+1; index segsAround is the UV-seam duplicate of 0 */
  ids: number[];
  pos: Vector3[];
}

/** ref-vector frame for a tube axis — IDENTICAL to tubeForBranch's frame init,
 *  so the base ring winds consistently. */
function axisFrame(dir: Vector3, outN: Vector3, outB: Vector3): void {
  const ref = Math.abs(dir.y) < 0.94 ? new Vector3(0, 1, 0) : new Vector3(1, 0, 0);
  outN.crossVectors(ref, dir).normalize();
  outB.crossVectors(dir, outN).normalize();
}

/** smooth fan disc over a ring (trunk ground cap / buried fallback base cap).
 *  rng-free. Winding is non-critical (these caps are always buried/invisible). */
function smoothDisc(
  g: MeshGrower,
  ring: EmittedRing,
  center: Vector3,
  normal: Vector3,
  hue: number,
  flex: number,
  swayPhase: number,
): void {
  const seg = ring.ids.length - 1;
  const c = g.vertex(
    center.x, center.y, center.z,
    normal.x, normal.y, normal.z,
    0.5, 0.5, hue, flex, swayPhase, 0.6,
  );
  for (let k = 0; k < seg; k++) {
    g.tri(ring.ids[k] as number, c, ring.ids[k + 1] as number);
  }
}

interface MeshBranchOpts {
  /** ring vertex count for SMOOTH (undisplaced) branches (ringsForLevel). */
  ringSegs: number;
  /** world metres per bark tile — sets uRepeats (integer, seam-safe) and
   *  v = vAlong / tileW (world-proportional; replaces the old ~0.5 m barkRepeats
   *  mapping). Per branch: uRepeats = max(1, round(2π·baseR / tileW)). */
  tileW: number;
  /** bark macro-relief field, or null (smooth ring). When present, thick rings
   *  (r ≥ ~0.06 m) are displaced into real furrows/ridges + normals recomputed
   *  from the displaced surface; thin/young bark (r < 0.05 m) stays smooth. */
  relief: BarkFieldParams | null;
  flare?: { amp: number; height: number; lobes: number; phase: number };
  swayPhase: number;
  swayFlexBase: number;
  swayFlexTip: number;
  hue: number;
}

interface RingSample {
  p: Vector3;
  r: number;
  dir: Vector3;
}

/** length of a branch polyline (for the along-branch ring-insertion schedule). */
function branchLength(br: SkelBranch): number {
  let L = 0;
  for (let i = 1; i < br.pts.length; i++) {
    L += (br.pts[i] as Vector3).distanceTo(br.pts[i - 1] as Vector3);
  }
  return Math.max(1e-3, L);
}

/** resample a branch polyline to a target ring spacing (m) that tightens toward
 *  the base (relief needs along-v resolution there); endpoints preserved so the
 *  base disc and tip cap still connect. Undisplaced branches skip this. */
function resampleBranch(br: SkelBranch): RingSample[] {
  const n = br.pts.length;
  const total = branchLength(br);
  const out: RingSample[] = [
    { p: (br.pts[0] as Vector3).clone(), r: br.radii[0] as number, dir: (br.dirs[0] as Vector3).clone() },
  ];
  let acc = 0;
  const spacingAt = (frac: number): number => (frac < 0.25 ? 0.2 : frac < 0.6 ? 0.5 : 1.0);
  for (let i = 1; i < n; i++) {
    const a = br.pts[i - 1] as Vector3;
    const b = br.pts[i] as Vector3;
    const segLen = b.distanceTo(a);
    const spacing = spacingAt(acc / total);
    const subd = Math.max(1, Math.ceil(segLen / spacing));
    const ra = br.radii[i - 1] as number;
    const rb = br.radii[i] as number;
    const da = br.dirs[i - 1] as Vector3;
    const db = br.dirs[i] as Vector3;
    for (let s = 1; s <= subd; s++) {
      const t = s / subd;
      out.push({
        p: new Vector3().lerpVectors(a, b, t),
        r: ra + (rb - ra) * t,
        dir: new Vector3().lerpVectors(da, db, t).normalize(),
      });
    }
    acc += segLen;
  }
  return out;
}

/** displacement amplitude ramp (m): 0 below ~0.05 m (young bark smooth), rising
 *  to macroAmp for thick trunks (≥ 0.25 m). Smooth ramp ⇒ no step at the cutoff. */
function reliefAmp(field: BarkFieldParams, r: number): number {
  const gate = smoothstep01(0.05, 0.11, r);
  return field.macroAmp * gate * Math.min(1, r / 0.25);
}
function smoothstep01(a: number, b: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** vertex normals from the DISPLACED surface (neighbour finite differences over
 *  the ring grid). Seam-consistent: k and k=seg read the same modular neighbours
 *  ⇒ identical normals, preserving the welded seam. Oriented outward. */
function computeSurfaceNormals(
  posGrid: Vector3[][],
  samples: RingSample[],
  seg: number,
): Vector3[][] {
  const m = posGrid.length;
  const out: Vector3[][] = [];
  const around = new Vector3();
  const axial = new Vector3();
  const nrm = new Vector3();
  const outward = new Vector3();
  for (let i = 0; i < m; i++) {
    const ring = posGrid[i] as Vector3[];
    const ringPrev = posGrid[Math.max(0, i - 1)] as Vector3[];
    const ringNext = posGrid[Math.min(m - 1, i + 1)] as Vector3[];
    const center = (samples[i] as RingSample).p;
    const rn: Vector3[] = [];
    for (let k = 0; k <= seg; k++) {
      const kp = (k + 1) % seg;
      const km = (k - 1 + seg) % seg;
      around.subVectors(ring[kp] as Vector3, ring[km] as Vector3);
      axial.subVectors(ringNext[k] as Vector3, ringPrev[k] as Vector3);
      nrm.crossVectors(axial, around);
      if (nrm.lengthSq() < 1e-12) nrm.subVectors(ring[k] as Vector3, center);
      nrm.normalize();
      outward.subVectors(ring[k] as Vector3, center);
      if (nrm.dot(outward) < 0) nrm.negate();
      rn.push(nrm.clone());
    }
    out.push(rn);
  }
  return out;
}

/** Mesh one branch as a closed tube: full wall + tip cap + (when junctions on) a
 *  small buried disc closing the base ring so the open boundary loop is gone and
 *  the QEM simplifier can collapse the tree. The parent wall is left SOLID — the
 *  child just interpenetrates it (legacy poke-through), which is what is actually
 *  visible. When `o.relief` is set and the branch is thick, the ring loop cuts the
 *  bark macro field into the tube (real furrows) and recomputes vertex normals
 *  from the displaced surface; the u-seam stays bit-identical (field periodic in
 *  u, uRepeats integer) so the closed-manifold/QEM invariant holds. */
function meshBranch(
  g: MeshGrower,
  br: SkelBranch,
  o: MeshBranchOpts,
  rng: Rng,
  jx: {
    junctions: boolean;
  },
): void {
  const n = br.pts.length;
  if (n < 2) return;
  const baseR = Math.max(br.radii[0] as number, 1e-4);
  // displace thick trunk + primary rings only (thin twigs stay smooth + cheap).
  const displaced = o.relief !== null && baseR >= 0.06 && br.level <= 1;
  const field = o.relief;
  const seg = displaced
    ? Math.max(14, Math.min(64, Math.round((2 * Math.PI * baseR) / 0.05)))
    : Math.max(4, o.ringSegs);
  // world-proportional tiling: integer repeats around (seam law) + v per metre.
  const uRepeats = Math.max(1, Math.round((2 * Math.PI * baseR) / o.tileW));
  const invTileW = 1 / o.tileW;

  const samples = displaced ? resampleBranch(br) : br.pts.map((p, i): RingSample => ({
    p: p as Vector3,
    r: br.radii[i] as number,
    dir: br.dirs[i] as Vector3,
  }));
  const m = samples.length;

  const T = new Vector3().copy(samples[0]!.dir);
  const N = new Vector3();
  const B = new Vector3();
  axisFrame(T, N, B);

  const baseY = samples[0]!.p.y;
  let vAlong = 0;
  const rings: EmittedRing[] = [];
  // per-ring scratch for the displaced-surface normal recompute
  const posGrid: Vector3[][] = [];
  const smoothN: Vector3[][] = [];
  const uvGrid: { u: number[]; v: number }[] = [];
  const flexArr: number[] = [];
  let lastRingPos: number[] = [];

  for (let i = 0; i < m; i++) {
    const s = samples[i]!;
    const p = s.p;
    const r = s.r;
    if (i > 0) {
      const prev = samples[i - 1]!.p;
      vAlong += _v.subVectors(p, prev).length();
      const tPrev = samples[i - 1]!.dir;
      const tCur = s.dir;
      const axis = _v.crossVectors(tPrev, tCur);
      const sLen = axis.length();
      if (sLen > 1e-6) {
        axis.multiplyScalar(1 / sLen);
        const ang = Math.asin(Math.min(1, sLen));
        N.applyAxisAngle(axis, ang).normalize();
        B.applyAxisAngle(axis, ang).normalize();
      }
    }
    const rNext = samples[Math.min(m - 1, i + 1)]!.r;
    const rPrev = samples[Math.max(0, i - 1)]!.r;
    const slope = ((rPrev - rNext) * (m - 1)) / Math.max(0.05, br.len) * 0.5;
    const tt = i / (m - 1);
    const flex = o.swayFlexBase + (o.swayFlexTip - o.swayFlexBase) * tt;
    const tan = s.dir;
    const amp = displaced ? reliefAmp(field as BarkFieldParams, r) : 0;
    const vv = vAlong * invTileW;
    const pos: Vector3[] = [];
    const sn: Vector3[] = [];
    const us: number[] = [];
    for (let k = 0; k <= seg; k++) {
      const a = (k / seg) * Math.PI * 2;
      const ca = Math.cos(a);
      const sa = Math.sin(a);
      let rr = r;
      if (o.flare && br.level === 0) {
        const h = p.y - baseY;
        const lobe = Math.pow(Math.max(0, Math.cos(o.flare.lobes * a + o.flare.phase)), 1.6);
        rr *= 1 + o.flare.amp * Math.exp(-h / o.flare.height) * (0.45 + 0.9 * lobe);
      }
      const uu = (k / seg) * uRepeats;
      if (amp > 0) rr += amp * barkMacro(field as BarkFieldParams, uu, vv).d;
      const dx = N.x * ca + B.x * sa;
      const dy = N.y * ca + B.y * sa;
      const dz = N.z * ca + B.z * sa;
      pos.push(new Vector3(p.x + dx * rr, p.y + dy * rr, p.z + dz * rr));
      // smooth analytic normal (used for undisplaced rings; unchanged behaviour)
      let nx = dx + tan.x * slope;
      let ny = dy + tan.y * slope;
      let nz = dz + tan.z * slope;
      const nl = Math.hypot(nx, ny, nz) || 1;
      sn.push(new Vector3(nx / nl, ny / nl, nz / nl));
      us.push(uu);
    }
    posGrid.push(pos);
    smoothN.push(sn);
    uvGrid.push({ u: us, v: vv });
    flexArr.push(flex);
  }

  // normals: displaced rings recompute from the displaced surface (neighbour
  // finite differences); undisplaced rings keep the analytic ring normal.
  const nrmGrid: Vector3[][] = displaced
    ? computeSurfaceNormals(posGrid, samples, seg)
    : smoothN;

  // emit vertices
  for (let i = 0; i < m; i++) {
    const pos = posGrid[i] as Vector3[];
    const nrm = nrmGrid[i] as Vector3[];
    const uvr = uvGrid[i] as { u: number[]; v: number };
    const flex = flexArr[i] as number;
    const ids: number[] = [];
    for (let k = 0; k <= seg; k++) {
      const pk = pos[k] as Vector3;
      const nk = nrm[k] as Vector3;
      ids.push(
        g.vertex(
          pk.x, pk.y, pk.z, nk.x, nk.y, nk.z,
          (uvr.u[k] as number), uvr.v,
          o.hue, flex, o.swayPhase, 1,
        ),
      );
    }
    rings.push({ ids, pos });
    if (i === m - 1) {
      lastRingPos = [];
      for (const v of pos) lastRingPos.push(v.x, v.y, v.z);
    }
  }

  const R = rings.length;
  if (R < 1) return;

  // ── PARENT role: none. CAP-ONLY junctions keep the parent wall SOLID (no hole
  // stencil); children simply interpenetrate it as in legacy, then cap their base.
  // (childFps/rimMap are still threaded through jx for the A/B plumbing but unused.)
  const removed = new Set<number>();

  // ── wall quads (full solid wall) ────────────────────────────────────────────
  for (let r = 0; r < R - 1; r++) {
    const a = rings[r] as EmittedRing;
    const b = rings[r + 1] as EmittedRing;
    for (let kmod = 0; kmod < seg; kmod++) {
      if (removed.has(r * seg + kmod)) continue;
      g.quad(a.ids[kmod] as number, a.ids[kmod + 1] as number, b.ids[kmod + 1] as number, b.ids[kmod] as number);
    }
  }

  // ── CHILD role base: close the (buried) base ring with a small disc ──────────
  const ring0 = rings[0] as EmittedRing;
  if (jx.junctions) {
    // close the open base ring (trunk ground cap AND every child base). The disc is
    // buried just behind the base ring inside the opaque parent, so it is invisible
    // yet removes the open edge loop the QEM simplifier would otherwise lock.
    const baseP = br.pts[0] as Vector3;
    const baseD = br.dirs[0] as Vector3;
    const center = new Vector3(
      baseP.x - baseD.x * baseR * 0.4,
      baseP.y - baseD.y * baseR * 0.4,
      baseP.z - baseD.z * baseR * 0.4,
    );
    smoothDisc(g, ring0, center, new Vector3(-baseD.x, -baseD.y, -baseD.z), o.hue, o.swayFlexBase, o.swayPhase);
  }
  // (junctions === false → base left open, reproducing legacy geometry for A/B)

  // ── tip cap (unchanged; keeps its rng draws so the stream stays in order) ────
  const last = rings[R - 1] as EmittedRing;
  const tipP = br.pts[n - 1] as Vector3;
  const tipD = br.dirs[n - 1] as Vector3;
  const tipR = br.radii[n - 1] as number;
  if (br.broken && tipR > 0.015) {
    const center = g.vertex(
      tipP.x + tipD.x * tipR * 0.4, tipP.y + tipD.y * tipR * 0.4, tipP.z + tipD.z * tipR * 0.4,
      tipD.x, tipD.y, tipD.z, 0.5, 0.5, o.hue, o.swayFlexTip, o.swayPhase, 0.55,
    );
    // jag ring has exactly `seg` verts and WRAPS modularly — a seam-duplicate
    // (k=seg) would carry an independent random spike, so jag[seg]≠jag[0] and the
    // cap would leak 4 open edges at the UV seam (the disc/taper caps avoid this
    // by reusing the ring's own seam-dup, which welds).
    const jag: number[] = [];
    for (let k = 0; k < seg; k++) {
      const px = tipP.x + ((lastRingPos[k * 3] as number) - tipP.x) * 0.45;
      const py = tipP.y + ((lastRingPos[k * 3 + 1] as number) - tipP.y) * 0.45;
      const pz = tipP.z + ((lastRingPos[k * 3 + 2] as number) - tipP.z) * 0.45;
      const spike = (rng.float() * 0.9 + 0.25) * tipR * 1.4;
      jag.push(
        g.vertex(
          px + tipD.x * spike, py + tipD.y * spike, pz + tipD.z * spike,
          tipD.x, tipD.y, tipD.z, 0.5, 0.5, o.hue, o.swayFlexTip, o.swayPhase, 0.5,
        ),
      );
    }
    for (let k = 0; k < seg; k++) {
      const k1 = (k + 1) % seg;
      g.quad(last.ids[k] as number, last.ids[k + 1] as number, jag[k1] as number, jag[k] as number);
      g.tri(jag[k1] as number, center, jag[k] as number);
    }
  } else {
    const tip = g.vertex(
      tipP.x + tipD.x * tipR * 2.0, tipP.y + tipD.y * tipR * 2.0, tipP.z + tipD.z * tipR * 2.0,
      tipD.x, tipD.y, tipD.z, 0.5, vAlong * invTileW + 0.2,
      o.hue, o.swayFlexTip, o.swayPhase, 1,
    );
    for (let k = 0; k < seg; k++) {
      g.tri(last.ids[k + 1] as number, tip, last.ids[k] as number);
    }
  }
}

/** mesh every branch of a skeleton into the grower (junction-aware) */
export function tubesForSkeleton(
  g: MeshGrower,
  skel: Skeleton,
  rng: Rng,
  opts: {
    lodK: number;
    /** world metres per bark tile (species field.tileW) — world-proportional UV. */
    tileW: number;
    /** bark macro-relief field (lod-0 hero only); null ⇒ smooth rings (r1/r2). */
    relief: BarkFieldParams | null;
    flare?: { amp: number; height: number; lobes: number; phase: number };
    /** skip branches at or above this level (LOD cut) */
    maxLevel?: number;
    /** keep only every Nth branch of level ≥ 1 (far-LOD bark diet) */
    branchStride?: number;
    /** false → legacy independent open-tube meshing (G5 A/B ablation) */
    junctions?: boolean;
  },
): void {
  const maxLevel = opts.maxLevel ?? 99;
  const stride = opts.branchStride ?? 1;
  const junctions = opts.junctions ?? true;
  const branches = skel.branches;

  // keptSet — replicate the legacy filter EXACTLY (incl. the level≥1 stride
  // counter), so the rng draw order and LOD selection are byte-identical.
  const kept = new Uint8Array(branches.length);
  {
    let bi = 0;
    for (let i = 0; i < branches.length; i++) {
      const br = branches[i] as SkelBranch;
      if (br.level > maxLevel) continue;
      if (br.level >= 1 && stride > 1 && bi++ % stride !== 0) continue;
      kept[i] = 1;
    }
  }

  for (let i = 0; i < branches.length; i++) {
    if (!kept[i]) continue;
    const br = branches[i] as SkelBranch;
    const flexB = br.level === 0 ? 0 : br.level === 1 ? 0.12 : 0.3;
    const flexT = br.level === 0 ? 0.05 : br.level === 1 ? 0.35 : 0.7;
    meshBranch(
      g,
      br,
      {
        ringSegs: ringsForLevel(br.level, opts.lodK),
        tileW: opts.tileW,
        relief: opts.relief,
        ...(br.level === 0 && opts.flare ? { flare: opts.flare } : {}),
        swayPhase: rng.float() * Math.PI * 2,
        swayFlexBase: flexB,
        swayFlexTip: flexT,
        hue: rng.float() * 2 - 1,
      },
      rng,
      { junctions },
    );
  }
}
