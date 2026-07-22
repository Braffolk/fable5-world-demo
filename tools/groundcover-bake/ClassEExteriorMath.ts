/**
 * Renderer-independent mathematics for one opaque class-E affine extrusion.
 *
 * This file deliberately knows nothing about WebGPU, textures, shaders, or
 * the legacy ground-cover direction atlas.  A finite sampled atlas supplies
 * `rho`; these functions only perform the exact affine reduction and lift.
 */

export type ClassEVec2 = readonly [number, number];
export type ClassEVec3 = readonly [number, number, number];

export interface ClassEAffine3 {
  /** Row-major canonical-to-world linear map. */
  linear: readonly [
    number, number, number,
    number, number, number,
    number, number, number,
  ];
  translation: ClassEVec3;
}

export interface ClassECanonicalRay {
  q0: ClassEVec2;
  h0: number;
  p: ClassEVec2;
  eta: number;
  projectedSpeed: number;
  omega: ClassEVec2 | null;
}

export interface ClassERayInterval {
  tMin: number;
  tMax: number;
}

function determinant3(m: ClassEAffine3['linear']): number {
  return m[0] * (m[4] * m[8] - m[5] * m[7])
    - m[1] * (m[3] * m[8] - m[5] * m[6])
    + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

export function inverseLinear3(m: ClassEAffine3['linear']): ClassEAffine3['linear'] {
  const det = determinant3(m);
  if (!(Math.abs(det) > 1e-15)) throw new Error('class-E affine map is singular');
  const s = 1 / det;
  return [
    (m[4] * m[8] - m[5] * m[7]) * s,
    (m[2] * m[7] - m[1] * m[8]) * s,
    (m[1] * m[5] - m[2] * m[4]) * s,
    (m[5] * m[6] - m[3] * m[8]) * s,
    (m[0] * m[8] - m[2] * m[6]) * s,
    (m[2] * m[3] - m[0] * m[5]) * s,
    (m[3] * m[7] - m[4] * m[6]) * s,
    (m[1] * m[6] - m[0] * m[7]) * s,
    (m[0] * m[4] - m[1] * m[3]) * s,
  ];
}

export function mulMat3Vec3(
  m: ClassEAffine3['linear'],
  v: ClassEVec3,
): ClassEVec3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

export function toCanonicalRay(
  affine: ClassEAffine3,
  worldOrigin: ClassEVec3,
  worldDirection: ClassEVec3,
): ClassECanonicalRay {
  const inverse = inverseLinear3(affine.linear);
  const relative: ClassEVec3 = [
    worldOrigin[0] - affine.translation[0],
    worldOrigin[1] - affine.translation[1],
    worldOrigin[2] - affine.translation[2],
  ];
  const origin = mulMat3Vec3(inverse, relative);
  const direction = mulMat3Vec3(inverse, worldDirection);
  const projectedSpeed = Math.hypot(direction[0], direction[1]);
  return {
    q0: [origin[0], origin[1]],
    h0: origin[2],
    p: [direction[0], direction[1]],
    eta: direction[2],
    projectedSpeed,
    omega: projectedSpeed === 0
      ? null
      : [direction[0] / projectedSpeed, direction[1] / projectedSpeed],
  };
}

/** Exact intersection of the forward finite ray with the canonical axial slab. */
export function clipCanonicalAxialInterval(
  ray: ClassECanonicalRay,
  hMin: number,
  hMax: number,
  maximumT: number,
): ClassERayInterval | null {
  if (!(hMax >= hMin) || !(maximumT >= 0)) return null;
  let tMin = 0;
  let tMax = maximumT;
  if (ray.eta === 0) {
    if (ray.h0 < hMin || ray.h0 > hMax) return null;
  } else {
    let a = (hMin - ray.h0) / ray.eta;
    let b = (hMax - ray.h0) / ray.eta;
    if (a > b) [a, b] = [b, a];
    tMin = Math.max(tMin, a);
    tMax = Math.min(tMax, b);
  }
  return tMin <= tMax ? { tMin, tMax } : null;
}

/**
 * Lift an exact two-dimensional first passage into world-ray parameter.
 * `occupiedAtEntry` is the categorical pole record for projectedSpeed == 0.
 */
export function liftClassEFirstPassage(
  ray: ClassECanonicalRay,
  interval: ClassERayInterval,
  rho: number | null,
  occupiedAtEntry: boolean,
): number | null {
  if (ray.projectedSpeed === 0) return occupiedAtEntry ? interval.tMin : null;
  if (rho === null || !(rho >= 0) || !Number.isFinite(rho)) return null;
  const t = interval.tMin + rho / ray.projectedSpeed;
  return t <= interval.tMax ? t : null;
}

/** Exact live-ray intersection with the stored tangent line. */
export function tangentReintersection(
  sampledHit: ClassEVec2,
  sampledNormal: ClassEVec2,
  livePhase: ClassEVec2,
  liveOmega: ClassEVec2,
): number | null {
  const denominator = sampledNormal[0] * liveOmega[0]
    + sampledNormal[1] * liveOmega[1];
  if (denominator === 0) return null;
  const numerator = sampledNormal[0] * (sampledHit[0] - livePhase[0])
    + sampledNormal[1] * (sampledHit[1] - livePhase[1]);
  const rho = numerator / denominator;
  return rho >= 0 ? rho : null;
}

export function applyRootLocalWind(
  root: ClassEVec3,
  point: ClassEVec3,
  beta: ClassEVec2,
  heightScale = 1,
): ClassEVec3 {
  const h = (point[1] - root[1]) * heightScale;
  return [
    root[0] + (point[0] - root[0]) + beta[0] * h,
    root[1] + h,
    root[2] + (point[2] - root[2]) + beta[1] * h,
  ];
}

/**
 * Integral of c + 2 Re(k exp(i(xiDotZ0 + xiDotV t))) on [t0,t1].
 * The zero-frequency limit is handled exactly rather than with an epsilon.
 */
export function integrateRealFourierMode(
  constant: number,
  coefficientReal: number,
  coefficientImaginary: number,
  phaseAtZero: number,
  angularSpeed: number,
  t0: number,
  t1: number,
): number {
  const length = t1 - t0;
  if (angularSpeed === 0) {
    const value = constant + 2 * (
      coefficientReal * Math.cos(phaseAtZero)
      - coefficientImaginary * Math.sin(phaseAtZero)
    );
    return value * length;
  }
  const primitive = (t: number): number => {
    const phase = phaseAtZero + angularSpeed * t;
    return 2 * (
      coefficientReal * Math.sin(phase)
      + coefficientImaginary * Math.cos(phase)
    ) / angularSpeed;
  };
  return constant * length + primitive(t1) - primitive(t0);
}
