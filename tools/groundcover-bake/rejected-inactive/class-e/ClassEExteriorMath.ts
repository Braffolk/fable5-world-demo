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
 * Filled-solid reference helper retained for comparison/counterexamples.
 * The selected cap-free lateral core uses `liftClassECurveFirstPassage`
 * instead and has no inside/outside occupancy state.
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

/**
 * Lift the first contact with a periodic closed curve set from the clipped
 * phase.  The curve may contain compact arcs with endpoints; it need not be
 * the boundary of a filled planar mask.  There is therefore no occupancy or
 * entry/exit state.
 *
 * At the projected-axis pole the ideal zero-thickness rule is categorical
 * coincidence: q is on the curve or it is not.  `coincidentAtPole` is that
 * one-bit mathematical oracle; a sampled implementation may route this
 * measure-zero case to its filtered residual instead of thickening the curve.
 */
export function liftClassECurveFirstPassage(
  ray: ClassECanonicalRay,
  interval: ClassERayInterval,
  rho: number | null,
  coincidentAtPole = false,
): number | null {
  if (ray.projectedSpeed === 0) return coincidentAtPole ? interval.tMin : null;
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
  minimumAbsoluteIncidence = 0,
): number | null {
  const denominator = sampledNormal[0] * liveOmega[0]
    + sampledNormal[1] * liveOmega[1];
  if (Math.abs(denominator) <= minimumAbsoluteIncidence) return null;
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

export interface ClassERealFourierMode {
  coefficientReal: number;
  coefficientImaginary: number;
  phaseAtZero: number;
  angularSpeed: number;
}

export interface ClassEFourierIntervalMoments {
  opticalDepth: number;
  firstMoment: number;
  centroid: number | null;
}

export type ClassEQuadraticRayWeight = readonly [
  constant: number,
  linear: number,
  quadratic: number,
];

interface ComplexValue {
  real: number;
  imaginary: number;
}

function sinc(value: number): number {
  const magnitude = Math.abs(value);
  if (magnitude >= 1e-3) return Math.sin(value) / value;
  const square = value * value;
  return 1 - square / 6 + square * square / 120 - square * square * square / 5040;
}

function sincDerivative(value: number): number {
  const magnitude = Math.abs(value);
  if (magnitude >= 1e-3) return (value * Math.cos(value) - Math.sin(value)) / (value * value);
  const square = value * value;
  return -value / 3 + value * square / 30 - value * square * square / 840;
}

function sincDerivativesThroughThird(value: number): readonly [number, number, number, number] {
  const magnitude = Math.abs(value);
  if (magnitude < 0.1) {
    const z2 = value * value;
    const z3 = z2 * value;
    const z4 = z2 * z2;
    const z5 = z4 * value;
    const z6 = z4 * z2;
    const z7 = z6 * value;
    const z8 = z4 * z4;
    return [
      1 - z2 / 6 + z4 / 120 - z6 / 5040 + z8 / 362880,
      -value / 3 + z3 / 30 - z5 / 840 + z7 / 45360,
      -1 / 3 + z2 / 10 - z4 / 168 + z6 / 6480 - z8 / 443520,
      value / 5 - z3 / 42 + z5 / 1080 - z7 / 55440,
    ];
  }
  const sine = Math.sin(value);
  const cosine = Math.cos(value);
  const z2 = value * value;
  const z3 = z2 * value;
  const z4 = z2 * z2;
  return [
    sine / value,
    (value * cosine - sine) / z2,
    (-z2 * sine - 2 * value * cosine + 2 * sine) / z3,
    (-z3 * cosine + 3 * z2 * sine + 6 * value * cosine - 6 * sine) / z4,
  ];
}

function multiplyComplex(a: ComplexValue, b: ComplexValue): ComplexValue {
  return {
    real: a.real * b.real - a.imaginary * b.imaginary,
    imaginary: a.real * b.imaginary + a.imaginary * b.real,
  };
}

function iRealPower(value: number, power: number): ComplexValue {
  const magnitude = value ** power;
  switch (power & 3) {
    case 0: return { real: magnitude, imaginary: 0 };
    case 1: return { real: 0, imaginary: magnitude };
    case 2: return { real: -magnitude, imaginary: 0 };
    default: return { real: 0, imaginary: -magnitude };
  }
}

const BINOMIAL_THROUGH_THREE = [
  [1, 0, 0, 0],
  [1, 1, 0, 0],
  [1, 2, 1, 0],
  [1, 3, 3, 1],
] as const;

/** Exact complex raw moment integral of t^power exp(i angularSpeed t). */
function complexExponentialRawMoment(
  power: 0 | 1 | 2 | 3,
  angularSpeed: number,
  t0: number,
  t1: number,
): ComplexValue {
  const length = t1 - t0;
  const midpoint = (t0 + t1) / 2;
  const halfLength = length / 2;
  const z = angularSpeed * halfLength;
  const derivatives = sincDerivativesThroughThird(z);
  let derivative: ComplexValue = { real: 0, imaginary: 0 };
  for (let order = 0; order <= power; order++) {
    const phasePower = iRealPower(midpoint, power - order);
    const scale = BINOMIAL_THROUGH_THREE[power][order]
      * halfLength ** order
      * derivatives[order];
    derivative = {
      real: derivative.real + phasePower.real * scale,
      imaginary: derivative.imaginary + phasePower.imaginary * scale,
    };
  }
  const phase = angularSpeed * midpoint;
  const phased = multiplyComplex(derivative, {
    real: Math.cos(phase),
    imaginary: Math.sin(phase),
  });
  const minusIPower: readonly ComplexValue[] = [
    { real: 1, imaginary: 0 },
    { real: 0, imaginary: -1 },
    { real: -1, imaginary: 0 },
    { real: 0, imaginary: 1 },
  ];
  const raw = multiplyComplex(phased, minusIPower[power]);
  return { real: length * raw.real, imaginary: length * raw.imaginary };
}

/**
 * Exact pole-stable moments after multiplying the Fourier density by a
 * quadratic ray-time window w(t)=w0+w1*t+w2*t^2.
 *
 * This is the closed form needed by the plane-free residual.  A clipped
 * root-local height window becomes precisely such a polynomial on the ray.
 */
export function integrateQuadraticWindowedRealFourierSeriesMoments(
  constant: number,
  modes: readonly ClassERealFourierMode[],
  weight: ClassEQuadraticRayWeight,
  t0: number,
  t1: number,
): ClassEFourierIntervalMoments {
  if (!(t1 >= t0)) throw new Error('Fourier interval must be ordered');
  const raw = ([0, 1, 2, 3] as const).map((power) => (
    complexExponentialRawMoment(power, 0, t0, t1)
  ));
  let opticalDepth = constant * (
    weight[0] * raw[0].real + weight[1] * raw[1].real + weight[2] * raw[2].real
  );
  let firstMoment = constant * (
    weight[0] * raw[1].real + weight[1] * raw[2].real + weight[2] * raw[3].real
  );

  for (const mode of modes) {
    const moments = ([0, 1, 2, 3] as const).map((power) => (
      complexExponentialRawMoment(power, mode.angularSpeed, t0, t1)
    ));
    const weighted: ComplexValue = {
      real: weight[0] * moments[0].real
        + weight[1] * moments[1].real
        + weight[2] * moments[2].real,
      imaginary: weight[0] * moments[0].imaginary
        + weight[1] * moments[1].imaginary
        + weight[2] * moments[2].imaginary,
    };
    const weightedFirst: ComplexValue = {
      real: weight[0] * moments[1].real
        + weight[1] * moments[2].real
        + weight[2] * moments[3].real,
      imaginary: weight[0] * moments[1].imaginary
        + weight[1] * moments[2].imaginary
        + weight[2] * moments[3].imaginary,
    };
    const phasedCoefficient = multiplyComplex(
      {
        real: mode.coefficientReal,
        imaginary: mode.coefficientImaginary,
      },
      {
        real: Math.cos(mode.phaseAtZero),
        imaginary: Math.sin(mode.phaseAtZero),
      },
    );
    opticalDepth += 2 * multiplyComplex(phasedCoefficient, weighted).real;
    firstMoment += 2 * multiplyComplex(phasedCoefficient, weightedFirst).real;
  }

  return {
    opticalDepth,
    firstMoment,
    centroid: opticalDepth > 0 ? firstMoment / opticalDepth : null,
  };
}

/**
 * Pole-stable optical depth and unattenuated first moment of
 *
 *   constant + 2 Re sum_j(c_j exp(i(phi_j + lambda_j t)))
 *
 * over [t0,t1].  This is the renderer-independent residual-medium identity.
 */
export function integrateRealFourierSeriesMoments(
  constant: number,
  modes: readonly ClassERealFourierMode[],
  t0: number,
  t1: number,
): ClassEFourierIntervalMoments {
  if (!(t1 >= t0)) throw new Error('Fourier interval must be ordered');
  const length = t1 - t0;
  const midpoint = (t0 + t1) / 2;
  let opticalDepth = constant * length;
  let firstMoment = constant * length * midpoint;

  for (const mode of modes) {
    const z = mode.angularSpeed * length / 2;
    const s = sinc(z);
    const sp = sincDerivative(z);
    const phase = mode.phaseAtZero + mode.angularSpeed * midpoint;
    const cosine = Math.cos(phase);
    const sine = Math.sin(phase);

    const integralReal = length * s * cosine;
    const integralImaginary = length * s * sine;
    opticalDepth += 2 * (
      mode.coefficientReal * integralReal
      - mode.coefficientImaginary * integralImaginary
    );

    const momentRealPart = midpoint * length * s;
    const momentImaginaryPart = -length * length * sp / 2;
    const momentReal = cosine * momentRealPart - sine * momentImaginaryPart;
    const momentImaginary = sine * momentRealPart + cosine * momentImaginaryPart;
    firstMoment += 2 * (
      mode.coefficientReal * momentReal
      - mode.coefficientImaginary * momentImaginary
    );
  }

  return {
    opticalDepth,
    firstMoment,
    centroid: opticalDepth > 0 ? firstMoment / opticalDepth : null,
  };
}
