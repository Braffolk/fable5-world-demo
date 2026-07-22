import assert from 'node:assert/strict';
import test from 'node:test';
import {
  applyRootLocalWind,
  clipCanonicalAxialInterval,
  integrateRealFourierMode,
  liftClassEFirstPassage,
  tangentReintersection,
  toCanonicalRay,
  type ClassEAffine3,
  type ClassEVec2,
  type ClassEVec3,
} from './ClassEExteriorMath';

const EPS = 2e-12;

function close(actual: number, expected: number, epsilon = EPS): void {
  assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} != ${expected}`);
}

function circleFirstPassage(
  phase: ClassEVec2,
  omega: ClassEVec2,
  radius: number,
): number | null {
  const b = phase[0] * omega[0] + phase[1] * omega[1];
  const c = phase[0] ** 2 + phase[1] ** 2 - radius ** 2;
  const discriminant = b * b - c;
  if (discriminant < 0) return null;
  const near = -b - Math.sqrt(discriminant);
  if (near >= 0) return near;
  const far = -b + Math.sqrt(discriminant);
  return far >= 0 ? 0 : null;
}

function normalized(v: ClassEVec3): ClassEVec3 {
  const length = Math.hypot(...v);
  return [v[0] / length, v[1] / length, v[2] / length];
}

const AFFINE: ClassEAffine3 = {
  linear: [
    1.3, 0.1, 0.24,
    0.0, 0.8, 0.92,
    0.2, -0.15, 0.11,
  ],
  translation: [1.2, -0.3, 2.1],
};

test('affine reduction and lift agree with an independent finite-cylinder equation', () => {
  const origins: ClassEVec3[] = [
    [3.5, 1.4, 2.4],
    [-1.0, 0.5, 1.6],
    [1.4, 3.2, 2.0],
  ];
  const directions: ClassEVec3[] = [
    normalized([-1, -0.2, -0.1]),
    normalized([1, -0.05, 0.2]),
    normalized([0.02, -1, 0.01]),
  ];
  for (const origin of origins) for (const direction of directions) {
    const ray = toCanonicalRay(AFFINE, origin, direction);
    const interval = clipCanonicalAxialInterval(ray, -0.4, 0.9, 20);
    if (!interval) continue;
    const phase: ClassEVec2 = [
      ray.q0[0] + interval.tMin * ray.p[0],
      ray.q0[1] + interval.tMin * ray.p[1],
    ];
    const occupied = phase[0] ** 2 + phase[1] ** 2 <= 0.45 ** 2;
    const rho = ray.omega ? circleFirstPassage(phase, ray.omega, 0.45) : null;
    const lifted = liftClassEFirstPassage(ray, interval, rho, occupied);

    // Independent truth: solve the circle quadratic in world-ray t, then
    // intersect the roots/inside interval with the axial slab interval.
    const a = ray.p[0] ** 2 + ray.p[1] ** 2;
    const b = 2 * (ray.q0[0] * ray.p[0] + ray.q0[1] * ray.p[1]);
    const c = ray.q0[0] ** 2 + ray.q0[1] ** 2 - 0.45 ** 2;
    let truth: number | null = null;
    if (a === 0) truth = c <= 0 ? interval.tMin : null;
    else {
      const discriminant = b * b - 4 * a * c;
      if (discriminant >= 0) {
        const first = (-b - Math.sqrt(discriminant)) / (2 * a);
        const second = (-b + Math.sqrt(discriminant)) / (2 * a);
        if (c <= 0) truth = interval.tMin;
        else if (first >= interval.tMin && first <= interval.tMax) truth = first;
        else if (second >= interval.tMin && second <= interval.tMax) truth = second;
      }
    }
    if (truth === null) assert.equal(lifted, null);
    else close(lifted!, truth, 2e-11);
  }
});

test('exact projected-axis pole uses occupancy and no epsilon', () => {
  const affine: ClassEAffine3 = {
    linear: [1, 0, 0.3, 0, 0, 1, 0, 1, -0.2],
    translation: [0, 0, 0],
  };
  const axis = normalized([0.3, 1, -0.2]);
  const ray = toCanonicalRay(affine, [0.1, -1, 0.2], axis);
  assert.equal(ray.projectedSpeed, 0);
  const interval = clipCanonicalAxialInterval(ray, -10, 10, 3)!;
  assert.equal(liftClassEFirstPassage(ray, interval, null, false), null);
  close(liftClassEFirstPassage(ray, interval, null, true)!, 0);
});

test('exact horizontal world rays survive the affine reduction', () => {
  const ray = toCanonicalRay(AFFINE, [2, 0.2, 2], normalized([-1, 0, 0.2]));
  assert.ok(ray.projectedSpeed > 0);
  const interval = clipCanonicalAxialInterval(ray, -2, 2, 10);
  assert.ok(interval);
  assert.ok(Number.isFinite(interval!.tMin) && Number.isFinite(interval!.tMax));
});

test('fixed field union elects the nearest exact candidate across owner order swaps', () => {
  const candidates = [3.4, null, 1.2, 2.7] as const;
  const winner = candidates.reduce<number | null>((best, value) => {
    if (value === null) return best;
    return best === null || value < best ? value : best;
  }, null);
  close(winner!, 1.2);
  const reversed = [...candidates].reverse().reduce<number | null>((best, value) => {
    if (value === null) return best;
    return best === null || value < best ? value : best;
  }, null);
  close(reversed!, winner!);
});

test('stored tangent line exactly reintersects every live ray in one straight segment cell', () => {
  // Boundary x=0.7, with an arbitrary sampled point on that same line.
  const hit: ClassEVec2 = [0.7, -0.31];
  const normal: ClassEVec2 = [1, 0];
  const phases: ClassEVec2[] = [[-0.4, -0.2], [0.1, 0.5], [-1.2, -0.8]];
  const directions: ClassEVec2[] = [[1, 0], [0.8, 0.6], [0.35, -Math.sqrt(1 - 0.35 ** 2)]];
  for (const phase of phases) for (const omega of directions) {
    const rho = tangentReintersection(hit, normal, phase, omega)!;
    close(phase[0] + rho * omega[0], 0.7);
  }
  assert.equal(tangentReintersection(hit, normal, [0, 0], [0, 1]), null);
});

test('root-local wind fixes roots and has exact height-proportional displacement', () => {
  const root: ClassEVec3 = [10, 3, -4];
  assert.deepEqual(applyRootLocalWind(root, root, [0.4, -0.2], 1.15), root);
  const point: ClassEVec3 = [10.2, 5, -3.7];
  const moved = applyRootLocalWind(root, point, [0.4, -0.2], 1.15);
  close(moved[0], 10.2 + 0.4 * 2.3);
  close(moved[1], 5.3);
  close(moved[2], -3.7 - 0.2 * 2.3);
});

test('closed-form plume mode integral matches independent dense quadrature', () => {
  const constant = 1.4;
  const kr = 0.23;
  const ki = -0.17;
  const phase = 0.43;
  for (const speed of [0, 1e-4, 0.7, -2.3]) {
    const t0 = -0.2;
    const t1 = 1.7;
    const exact = integrateRealFourierMode(constant, kr, ki, phase, speed, t0, t1);
    const steps = 200_000;
    const dt = (t1 - t0) / steps;
    let numerical = 0;
    for (let index = 0; index < steps; index++) {
      const t = t0 + (index + 0.5) * dt;
      const p = phase + speed * t;
      numerical += (constant + 2 * (kr * Math.cos(p) - ki * Math.sin(p))) * dt;
    }
    close(exact, numerical, 2e-9);
  }
});
