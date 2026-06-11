/**
 * Branching skeleton growth — parametric recursive grammar (SpeedTree-style):
 * each branch is a polyline walked segment by segment with tropisms
 * (gravity/light), wander noise, and cantilever droop; children spawn along
 * the parent with whorled or spiral phyllotaxis, their length shaped by a
 * crown envelope + light-competition asymmetry. Deterministic per Rng stream.
 */

import { Quaternion, Vector3 } from 'three';
import type { Rng } from '../core/Seed';
import type {
  CrownShape,
  GrowthInstance,
  LeafAnchor,
  LevelParams,
  Skeleton,
  SkelBranch,
  SpeciesParams,
} from './VegTypes';

const UP = new Vector3(0, 1, 0);
const GOLDEN = 2.39996323; // golden angle (rad)

function crownEnvelope(shape: CrownShape, t: number, rng: Rng): number {
  switch (shape) {
    case 'cone':
      // longest near the base of the crown, tapering to the leader
      return 0.18 + 0.82 * Math.pow(1 - t, 0.9);
    case 'ellipsoid':
      return Math.max(0.12, Math.sin(Math.PI * (0.08 + 0.88 * t)));
    case 'dome':
      // flat-bottomed dome: long low arms, shorter top
      return Math.max(0.15, Math.sqrt(Math.max(0, 1 - t * t * 0.92)));
    case 'column':
      return 0.55 + 0.45 * Math.sin(Math.PI * Math.min(1, t * 1.15));
    case 'irregular':
      return 0.3 + 0.7 * Math.abs(Math.sin(t * 9.7 + rng.float() * 6.28)) * (1 - t * 0.4);
  }
}

/** stable orthonormal basis perpendicular to dir */
function perpBasis(dir: Vector3, outN: Vector3, outB: Vector3): void {
  const ref = Math.abs(dir.y) < 0.94 ? UP : new Vector3(1, 0, 0);
  outN.crossVectors(ref, dir).normalize();
  outB.crossVectors(dir, outN).normalize();
}

interface GrowCtx {
  sp: SpeciesParams;
  rng: Rng;
  inst: GrowthInstance;
  branches: SkelBranch[];
  anchors: LeafAnchor[];
  /** total branch count guard */
  budget: number;
}

interface BranchSpec {
  level: number;
  basePos: Vector3;
  baseDir: Vector3;
  len: number;
  baseR: number;
  tParent: number;
  /** phyllotaxis azimuth on the parent (for anchor frames) */
  azimuth: number;
  stub: boolean;
}

function growBranch(ctx: GrowCtx, spec: BranchSpec): SkelBranch | null {
  if (ctx.budget <= 0) return null;
  ctx.budget--;

  const { sp, rng, inst } = ctx;
  const lp = sp.levels[spec.level] as LevelParams;
  const segs = Math.max(2, lp.segs);
  const isTrunk = spec.level === 0;
  const len = spec.stub ? spec.len * (0.12 + rng.float() * 0.18) : spec.len;
  const broken =
    spec.stub || (isTrunk && sp.brokenTop > 0 && sp.brokenTop < 1);
  const effLen = isTrunk && sp.brokenTop > 0 ? len * sp.brokenTop : len;

  const pts: Vector3[] = [];
  const radii: number[] = [];
  const dirs: Vector3[] = [];
  const dir = spec.baseDir.clone().normalize();
  const pos = spec.basePos.clone();
  const segLen = effLen / segs;

  // per-branch wander stream so sibling branches decorrelate
  const wanderPhase = rng.float() * Math.PI * 2;
  const wanderFreq = 1.5 + rng.float() * 2.5;

  // cantilever droop accumulates toward the tip; tip curl opposes it late
  const droopTotal = lp.droop * (0.7 + rng.float() * 0.6);

  const N = new Vector3();
  const B = new Vector3();
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    // a broken trunk's points span only the kept length — taper accordingly
    const tTaper = isTrunk && sp.brokenTop > 0 ? t * sp.brokenTop : t;
    let r = spec.baseR * Math.pow(Math.max(0, 1 - tTaper), lp.taper);
    if (isTrunk) r = Math.max(r, spec.baseR * 0.012);
    else r = Math.max(r, 0.0035);
    pts.push(pos.clone());
    radii.push(r);
    dirs.push(dir.clone());
    if (i === segs) break;

    // direction update: tropism + wander + droop/curl
    perpBasis(dir, N, B);
    const wob = lp.wander * (isTrunk ? 1 : 1.4);
    const a1 = Math.sin(t * wanderFreq * Math.PI * 2 + wanderPhase) * wob + (rng.float() - 0.5) * wob;
    const a2 = Math.cos(t * wanderFreq * Math.PI * 1.7 + wanderPhase * 1.7) * wob + (rng.float() - 0.5) * wob;
    dir.addScaledVector(N, a1).addScaledVector(B, a2);
    dir.addScaledVector(UP, lp.gravitropism * (isTrunk ? 1 : 0.4 + t));
    // droop: rotate down progressively (stronger at the unsupported tip)
    dir.y -= droopTotal * t * (1 / segs) * 2.4;
    dir.y += lp.tipCurl * Math.max(0, t - 0.62) * (1 / segs) * 5.2;
    if (isTrunk) {
      dir.x += inst.leanX * (1 / segs) * (1.6 - t);
      dir.z += inst.leanZ * (1 / segs) * (1.6 - t);
    }
    dir.normalize();
    pos.addScaledVector(dir, segLen * (0.92 + rng.float() * 0.16));
  }

  const branch: SkelBranch = {
    level: spec.level,
    pts,
    radii,
    dirs,
    len: effLen,
    tParent: spec.tParent,
    broken,
  };
  ctx.branches.push(branch);
  if (spec.stub) return branch;

  // ---- children ------------------------------------------------------------
  const childLevel = spec.level + 1;
  if (childLevel < sp.levels.length) {
    const cp = sp.levels[childLevel] as LevelParams;
    const span = Math.max(0, cp.childEnd - cp.childStart);
    const densityScale = 0.75 + inst.age * 0.45;
    const count = Math.round(effLen * span * cp.density * densityScale);
    if (count > 0) {
      const whorl = cp.whorl;
      const planar = cp.planar ?? 0;
      const groups = whorl >= 2 ? Math.max(1, Math.round(count / whorl)) : count;
      let azimuth = rng.float() * Math.PI * 2;
      for (let gi = 0; gi < groups; gi++) {
        const tG = cp.childStart + span * ((gi + 0.5) / groups);
        const inWhorl = whorl >= 2 ? whorl : 1;
        azimuth += whorl >= 2 ? GOLDEN * 0.5 + rng.float() * 0.4 : 0;
        for (let wi = 0; wi < inWhorl; wi++) {
          const t = Math.min(0.985, tG + (rng.float() - 0.5) * (span / groups) * 0.6);
          let az: number;
          if (rng.float() < planar) {
            // two-sided in the bough plane: ±N (horizontal perp), alternating
            az = (gi + wi) % 2 === 0 ? 0 : Math.PI;
            az += (rng.float() - 0.5) * 0.55;
          } else if (whorl >= 2) {
            az = azimuth + (wi / inWhorl) * Math.PI * 2 + (rng.float() - 0.5) * 0.5;
          } else {
            az = azimuth += GOLDEN + (rng.float() - 0.5) * 0.35;
          }
          // parent frame at t
          const idxF = t * segs;
          const i0 = Math.min(segs - 1, Math.floor(idxF));
          const f = idxF - i0;
          const pPos = new Vector3().lerpVectors(
            pts[i0] as Vector3,
            pts[i0 + 1] as Vector3,
            f,
          );
          const pDir = new Vector3()
            .lerpVectors(dirs[i0] as Vector3, dirs[i0 + 1] as Vector3, f)
            .normalize();
          const pR = (radii[i0] as number) * (1 - f) + (radii[i0 + 1] as number) * f;
          perpBasis(pDir, N, B);
          const side = new Vector3()
            .addScaledVector(N, Math.cos(az))
            .addScaledVector(B, Math.sin(az));
          const angle =
            cp.angleBase + (cp.angleTip - cp.angleBase) * t + (rng.float() - 0.5) * 0.16;
          const cDir = new Vector3()
            .addScaledVector(pDir, Math.cos(angle))
            .addScaledVector(side, Math.sin(angle))
            .normalize();
          // crown envelope + light-competition asymmetry
          const env = crownEnvelope(sp.crown, isTrunk ? t : t * 0.6 + 0.4, rng);
          const asymK =
            1 +
            sp.asym *
              (cDir.x * inst.biasX + cDir.z * inst.biasZ) *
              (isTrunk ? 1 : 0.4);
          const cLen =
            effLen * cp.lenRatio * env * asymK * (1 + (rng.float() - 0.5) * 2 * cp.lenJitter);
          if (cLen < 0.05) continue;
          const cR = Math.min(pR * cp.radRatio * (0.55 + env * 0.45), pR * 0.8);
          const stub = sp.stubChance > 0 && rng.chance(sp.stubChance);
          growBranch(ctx, {
            level: childLevel,
            basePos: pPos,
            baseDir: cDir,
            len: cLen,
            baseR: cR,
            tParent: t,
            azimuth: az,
            stub,
          });
        }
      }
    }
  }

  // ---- foliage anchors -----------------------------------------------------
  const fol = sp.foliage;
  if (fol && spec.level === fol.anchorLevel && !branch.broken) {
    const from = Math.max(0, fol.tStart);
    const along = effLen * (1 - from);
    const n = Math.max(1, Math.round(along / fol.spacing));
    const q = new Quaternion();
    const qTwist = new Quaternion();
    const qTilt = new Quaternion();
    for (let i = 0; i <= n; i++) {
      const t = Math.min(1, from + (1 - from) * (i / n));
      const idxF = t * segs;
      const i0 = Math.min(segs - 1, Math.floor(idxF));
      const f = idxF - i0;
      const aPos = new Vector3().lerpVectors(pts[i0] as Vector3, pts[i0 + 1] as Vector3, f);
      const aDir = new Vector3()
        .lerpVectors(dirs[i0] as Vector3, dirs[i0 + 1] as Vector3, f)
        .normalize();
      const terminal = i === n;
      // frame: +z along outgrowth direction, +y leaf-up
      perpBasis(aDir, N, B);
      const az = terminal
        ? 0
        : fol.planarLeaves
          ? (i % 2 === 0 ? 0 : Math.PI) + (rng.float() - 0.5) * 0.6
          : GOLDEN * i + (rng.float() - 0.5) * 0.7;
      const out = terminal
        ? aDir.clone()
        : new Vector3()
            .addScaledVector(aDir, Math.cos(fol.tilt))
            .addScaledVector(
              new Vector3().addScaledVector(N, Math.cos(az)).addScaledVector(B, Math.sin(az)),
              Math.sin(fol.tilt),
            )
            .normalize();
      // build quaternion: z→out, then twist so local y ≈ world up-ish
      q.setFromUnitVectors(new Vector3(0, 0, 1), out);
      const localY = new Vector3(0, 1, 0).applyQuaternion(q);
      const horizUp = new Vector3().addScaledVector(out, -out.y).add(UP).normalize();
      const twistAngle = Math.atan2(
        new Vector3().crossVectors(localY, horizUp).dot(out),
        localY.dot(horizUp),
      );
      qTwist.setFromAxisAngle(out, twistAngle + (rng.float() - 0.5) * 0.5);
      q.premultiply(qTwist);
      // slight droop of the spray itself
      const sideAxis = new Vector3(1, 0, 0).applyQuaternion(q);
      qTilt.setFromAxisAngle(sideAxis, (rng.float() - 0.5) * 0.24 + 0.08);
      q.premultiply(qTilt);
      const sc =
        (fol.scale[0] + rng.float() * (fol.scale[1] - fol.scale[0])) *
        (terminal ? 0.85 : 0.72 + t * 0.42);
      ctx.anchors.push({
        pos: aPos.clone().addScaledVector(out, sc * 0.06),
        quat: q.clone(),
        scale: sc,
        hue: rng.float() * 2 - 1,
        age: Math.max(0, 1 - t) * 0.7 + rng.float() * 0.3,
      });
    }
  }

  return branch;
}

export function growSkeleton(
  sp: SpeciesParams,
  rng: Rng,
  inst?: Partial<GrowthInstance>,
): Skeleton {
  const instance: GrowthInstance = {
    leanX: (rng.float() - 0.5) * 0.12,
    leanZ: (rng.float() - 0.5) * 0.12,
    biasX: 0,
    biasZ: 0,
    age: 0.5 + rng.float() * 0.5,
    ...inst,
  };
  if (instance.biasX === 0 && instance.biasZ === 0) {
    const a = rng.float() * Math.PI * 2;
    instance.biasX = Math.cos(a);
    instance.biasZ = Math.sin(a);
  }

  const height =
    (sp.height[0] + rng.float() * (sp.height[1] - sp.height[0])) *
    (0.72 + instance.age * 0.36);
  const ctx: GrowCtx = {
    sp,
    rng,
    inst: instance,
    branches: [],
    anchors: [],
    budget: 9000,
  };
  growBranch(ctx, {
    level: 0,
    basePos: new Vector3(0, 0, 0),
    baseDir: new Vector3(instance.leanX * 0.7, 1, instance.leanZ * 0.7).normalize(),
    len: height,
    baseR: height * sp.trunkRadiusK,
    tParent: 0,
    azimuth: 0,
    stub: false,
  });

  // crown bounds for normal bending + capture framing
  let minY = Infinity;
  let maxY = -Infinity;
  let maxR = 0.5;
  for (const a of ctx.anchors) {
    minY = Math.min(minY, a.pos.y);
    maxY = Math.max(maxY, a.pos.y);
    maxR = Math.max(maxR, Math.hypot(a.pos.x, a.pos.z));
  }
  if (!Number.isFinite(minY)) {
    minY = height * 0.3;
    maxY = height;
  }
  return {
    branches: ctx.branches,
    anchors: ctx.anchors,
    height,
    crownCenterY: (minY + maxY) * 0.5,
    crownRadius: maxR,
  };
}
