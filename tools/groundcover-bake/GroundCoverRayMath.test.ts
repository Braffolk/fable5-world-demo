import assert from 'node:assert/strict';
import test from 'node:test';
import {
  affineEnvelopeEntryFromGround,
  bilinearEnvelopeRoots,
  makeProfileRay,
  profileNormalToWorld,
  rayDot,
  rayInverse,
  rayMatVec,
  rayNormalize,
  rayPlaneT,
  rayPoint,
  rayForTranslatedSurface,
  reprojectCanonicalPlaneHit,
  translatedSurfacePoint,
  worldDeltaFromProfileDistance,
  worldDeltaFromProjectedDistance,
  type RayMat3,
  type RayVec3,
} from './GroundCoverRayMath';

type Triangle = readonly [RayVec3, RayVec3, RayVec3];

function cross(a: RayVec3, b: RayVec3): RayVec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function sub(a: RayVec3, b: RayVec3): RayVec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function translatedTriangle(triangle: Triangle, translation: RayVec3): Triangle {
  return triangle.map((point) => [
    point[0] + translation[0],
    point[1] + translation[1],
    point[2] + translation[2],
  ] as RayVec3) as unknown as Triangle;
}

function triangleHit(
  ray: { origin: RayVec3; direction: RayVec3 },
  triangle: Triangle,
): { t: number; barycentric: RayVec3 } | null {
  const edge1 = sub(triangle[1], triangle[0]);
  const edge2 = sub(triangle[2], triangle[0]);
  const p = cross(ray.direction, edge2);
  const determinant = rayDot(edge1, p);
  if (Math.abs(determinant) <= 1e-12) return null;
  const inverse = 1 / determinant;
  const fromA = sub(ray.origin, triangle[0]);
  const u = rayDot(fromA, p) * inverse;
  const q = cross(fromA, edge1);
  const v = rayDot(ray.direction, q) * inverse;
  const t = rayDot(edge2, q) * inverse;
  const w = 1 - u - v;
  return t >= 0 && u >= -1e-10 && v >= -1e-10 && w >= -1e-10
    ? { t, barycentric: [w, u, v] }
    : null;
}

function firstTriangleHit(
  ray: { origin: RayVec3; direction: RayVec3 },
  triangles: readonly Triangle[],
): { t: number; owner: number; barycentric: RayVec3 } | null {
  let first: { t: number; owner: number; barycentric: RayVec3 } | null = null;
  triangles.forEach((triangle, owner) => {
    const hit = triangleHit(ray, triangle);
    if (hit && (!first || hit.t < first.t)) first = { ...hit, owner };
  });
  return first;
}

function close(actual: number, expected: number, tolerance = 1e-10): void {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`,
  );
}

function closeVec(actual: RayVec3, expected: RayVec3, tolerance = 1e-10): void {
  close(actual[0], expected[0], tolerance);
  close(actual[1], expected[1], tolerance);
  close(actual[2], expected[2], tolerance);
}

test('full inverse basis reconstructs an affine profile hit exactly', () => {
  const profileToWorld: RayMat3 = [
    [0.73, 0.18, -0.41],
    [0.22, 1.37, 0.31],
    [0.49, -0.12, 0.86],
  ];
  const worldDirection = rayNormalize([0.91, -0.37, 0.19]);
  const profileRay = makeProfileRay(worldDirection, profileToWorld);
  const profileOrigin: RayVec3 = [1.4, 1.2, -0.7];
  const profileDistance = 2.35;
  const profileHit: RayVec3 = [
    profileOrigin[0] + profileRay.direction[0] * profileDistance,
    profileOrigin[1] + profileRay.direction[1] * profileDistance,
    profileOrigin[2] + profileRay.direction[2] * profileDistance,
  ];
  const worldOrigin = rayMatVec(profileToWorld, profileOrigin);
  const expectedWorldHit = rayMatVec(profileToWorld, profileHit);
  const worldDelta = worldDeltaFromProfileDistance(profileDistance, profileRay);
  closeVec(rayPoint({ origin: worldOrigin, direction: worldDirection }, worldDelta), expectedWorldHit);
});

test('Sannikov projected path divides by projected profile speed, not a world cosine', () => {
  const profileToWorld: RayMat3 = [
    [1.8, 0.35, 0],
    [0.25, 0.9, -0.3],
    [0.2, 0.1, 0.55],
  ];
  const profileRay = makeProfileRay(rayNormalize([0.94, -0.31, 0.12]), profileToWorld);
  const projectedPath = 0.83;
  const worldDelta = worldDeltaFromProjectedDistance(projectedPath, profileRay);
  close(
    Math.hypot(profileRay.velocity[0] * worldDelta, profileRay.velocity[2] * worldDelta),
    projectedPath,
  );
  const naiveWorldCosine = Math.hypot(0.94, 0.12) / Math.hypot(0.94, -0.31, 0.12);
  assert.ok(Math.abs(worldDelta - projectedPath / naiveWorldCosine) > 0.25);
});

test('ground-to-top reconstruction is exact when point and derivative share one affine chart', () => {
  const slopeX = 0.23;
  const slopeZ = -0.17;
  // q=(x,height,z) -> world=(x, slopeX*x + height + slopeZ*z, z)
  const profileToWorld: RayMat3 = [
    [1, 0, 0],
    [slopeX, 1, slopeZ],
    [0, 0, 1],
  ];
  const direction = rayNormalize([0.91, -0.34, 0.24]);
  const profileVelocity = rayMatVec(rayInverse(profileToWorld), direction);
  close(profileVelocity[1], direction[1] - slopeX * direction[0] - slopeZ * direction[2]);
  const groundPoint: RayVec3 = [3.1, slopeX * 3.1 + slopeZ * -1.7, -1.7];
  const groundT = 12.4;
  const origin: RayVec3 = [
    groundPoint[0] - direction[0] * groundT,
    groundPoint[1] - direction[1] * groundT,
    groundPoint[2] - direction[2] * groundT,
  ];
  const topHeight = 1.176;
  const entryT = affineEnvelopeEntryFromGround(groundT, 0, topHeight, profileVelocity[1]);
  const entry = rayPoint({ origin, direction }, entryT);
  close(entry[1] - slopeX * entry[0] - slopeZ * entry[2], topHeight);
});

test('constant translated envelope equals an unchanged carrier queried from the shifted origin', () => {
  const carrier: readonly Triangle[] = [
    [[0, 0, -2], [3, 0, -2], [0, 0, 2]],
    [[0, 0.35, -2], [3, 0.35, -2], [0, 0.35, 2]],
  ];
  const translation: RayVec3 = [0, 1.176, 0];
  const ray = {
    origin: [0.4, 3.2, 0] as RayVec3,
    direction: rayNormalize([0.2, -1, 0.05]),
  };
  const shellHit = firstTriangleHit(
    ray,
    carrier.map((triangle) => translatedTriangle(triangle, translation)),
  );
  const shiftedRay = rayForTranslatedSurface(ray, translation);
  const carrierHit = firstTriangleHit(shiftedRay, carrier);
  assert.ok(shellHit);
  assert.ok(carrierHit);
  close(shellHit.t, carrierHit.t);
  assert.equal(shellHit.owner, carrierHit.owner);
  closeVec(shellHit.barycentric, carrierHit.barycentric);
  closeVec(rayPoint(ray, shellHit.t), translatedSurfacePoint(shiftedRay, translation, carrierHit.t));

  // Same-orientation perspective sees exactly the same camera-relative vector,
  // hence the same clip coordinates, NDC depth, finite edges and silhouettes.
  const shellPoint = rayPoint(ray, shellHit.t);
  const carrierPoint = rayPoint(shiftedRay, carrierHit.t);
  closeVec(sub(shellPoint, ray.origin), sub(carrierPoint, shiftedRay.origin));
});

test('shifted-origin equivalence preserves shell silhouettes absent from the ordinary ground ray', () => {
  const carrier: readonly Triangle[] = [
    [[0, 0, -1], [2, 0, -1], [1, 0, 1]],
  ];
  const translation: RayVec3 = [0, 1, 0];
  const ray = {
    origin: [-1, 0.5, 0] as RayVec3,
    direction: rayNormalize([1, 0.25, 0]),
  };
  assert.equal(firstTriangleHit(ray, carrier), null, 'ordinary camera ray has no ground hit');
  const shellHit = firstTriangleHit(
    ray,
    carrier.map((triangle) => translatedTriangle(triangle, translation)),
  );
  const carrierHit = firstTriangleHit(rayForTranslatedSurface(ray, translation), carrier);
  assert.ok(shellHit);
  assert.ok(carrierHit);
  close(shellHit.t, carrierHit.t);
  assert.equal(shellHit.owner, carrierHit.owner);
});

test('one top-entry first-hit record cannot resolve arbitrary camera-inside visibility', () => {
  const topEntry = -2;
  const profileWithLaterVisibleSurface = [-1, 1];
  const profileWithoutLaterVisibleSurface = [-1];
  const storedFirst = (hits: readonly number[]): number | undefined =>
    hits.find((t) => t >= topEntry);
  const firstVisible = (hits: readonly number[]): number | undefined =>
    hits.find((t) => t >= 0);
  assert.equal(storedFirst(profileWithLaterVisibleSurface), -1);
  assert.equal(storedFirst(profileWithoutLaterVisibleSurface), -1);
  assert.equal(firstVisible(profileWithLaterVisibleSurface), 1);
  assert.equal(firstVisible(profileWithoutLaterVisibleSurface), undefined);
});

test('mixing raster depth with a different height/gradient chart amplifies error at grazing angles', () => {
  const elevation = 5 * Math.PI / 180;
  const direction: RayVec3 = [Math.cos(elevation), -Math.sin(elevation), 0];
  const groundT = 20;
  const topHeight = 1.176;
  const exact = affineEnvelopeEntryFromGround(groundT, 0, topHeight, direction[1]);
  // Same raster point, but a separately sampled guide says ground is 8 mm high
  // and its slope differs by 0.008. This is the former mixed-chart operation.
  const proxyHeight = -0.008;
  const proxyHeightSpeed = direction[1] - 0.008 * direction[0];
  const mixed = affineEnvelopeEntryFromGround(groundT, proxyHeight, topHeight, proxyHeightSpeed);
  assert.ok(Math.abs(mixed - exact) > 1.0, `expected grazing amplification, got ${mixed - exact} m`);
});

test('value and gradient at the ground hit cannot determine a curved envelope intersection', () => {
  const elevation = 8 * Math.PI / 180;
  const direction: RayVec3 = [Math.cos(elevation), -Math.sin(elevation), 0];
  const groundT = 10;
  const origin: RayVec3 = [
    -direction[0] * groundT,
    -direction[1] * groundT,
    0,
  ];
  const topHeight = 1;
  const tangentPrediction = affineEnvelopeEntryFromGround(groundT, 0, topHeight, direction[1]);
  // g0(x)=0 and g1(x)=0.002*x^2 have the same value and derivative at x=0.
  // Their top-envelope intersections differ, proving that the local jet alone
  // is insufficient; this is an information mismatch, not a shader detail.
  const curvature = 0.002;
  const a = -curvature * direction[0] * direction[0];
  const b = direction[1] - 2 * curvature * origin[0] * direction[0];
  const c = origin[1] - curvature * origin[0] * origin[0] - topHeight;
  const discriminant = b * b - 4 * a * c;
  const curvedRoots = [
    (-b - Math.sqrt(discriminant)) / (2 * a),
    (-b + Math.sqrt(discriminant)) / (2 * a),
  ].filter((value) => value >= 0 && value <= groundT);
  assert.equal(curvedRoots.length, 1);
  assert.ok(Math.abs(curvedRoots[0]! - tangentPrediction) > 0.5);
});

test('extending the ground-hit triangle is not exact when the envelope enters over its neighbour', () => {
  const elevation = 10 * Math.PI / 180;
  const direction: RayVec3 = [Math.cos(elevation), -Math.sin(elevation), 0];
  const slope = 0.4;
  const topHeight = 1.176;
  const groundT = 20;
  const groundPoint: RayVec3 = [1, slope, 0];
  const origin: RayVec3 = [
    groundPoint[0] - direction[0] * groundT,
    groundPoint[1] - direction[1] * groundT,
    0,
  ];
  // Actual piecewise terrain: g(x)=0 for x<=0, g(x)=slope*x for x>0.
  // The true top-envelope point lies over the flat neighbour.
  const actualEntry = (topHeight - origin[1]) / direction[1];
  const actualPoint = rayPoint({ origin, direction }, actualEntry);
  assert.ok(actualPoint[0] < 0);
  close(actualPoint[1], topHeight);
  // Extending T's sloped triangle to infinity satisfies the wrong affine chart.
  const slopedHeightSpeed = direction[1] - slope * direction[0];
  const extendedTriangleEntry = affineEnvelopeEntryFromGround(
    groundT,
    0,
    topHeight,
    slopedHeightSpeed,
  );
  const extendedPoint = rayPoint({ origin, direction }, extendedTriangleEntry);
  close(extendedPoint[1] - slope * extendedPoint[0], topHeight);
  assert.ok(Math.abs(extendedTriangleEntry - actualEntry) > 2);
});

test('a known bilinear ground patch has a closed-form exact envelope solve', () => {
  const ray = {
    origin: [-2.1, 4.7, -1.3] as RayVec3,
    direction: rayNormalize([0.72, -0.61, 0.33]),
  };
  const ground = { constant: 0.4, x: 0.13, z: -0.08, xz: 0.035 };
  const roots = bilinearEnvelopeRoots(ray, ground, 1.176);
  assert.ok(roots.length > 0);
  for (const t of roots) {
    const [x, y, z] = rayPoint(ray, t);
    close(y - (ground.constant + ground.x * x + ground.z * z + ground.xz * x * z), 1.176);
  }
});

test('geometric-plane reprojection is exact where linear directional depth blending is not', () => {
  const origin: RayVec3 = [0.17, 1.4, -0.21];
  const planePoint: RayVec3 = [0.3, 0.45, 0.1];
  const planeNormal = rayNormalize([0.31, 0.88, -0.36]);
  const d0 = rayNormalize([0.6, -0.75, 0.28]);
  const d1 = rayNormalize([-0.2, -0.91, 0.37]);
  const live = rayNormalize([
    d0[0] * 0.57 + d1[0] * 0.43,
    d0[1] * 0.57 + d1[1] * 0.43,
    d0[2] * 0.57 + d1[2] * 0.43,
  ]);
  const t0 = rayPlaneT({ origin, direction: d0 }, planePoint, planeNormal);
  const t1 = rayPlaneT({ origin, direction: d1 }, planePoint, planeNormal);
  const exact = rayPlaneT({ origin, direction: live }, planePoint, planeNormal);
  close(reprojectCanonicalPlaneHit(origin, d0, t0, planeNormal, live), exact);
  assert.ok(Math.abs((t0 * 0.57 + t1 * 0.43) - exact) > 1e-3);
});

test('profile normals use inverse-transpose and remain orthogonal after shear', () => {
  const profileToWorld: RayMat3 = [
    [1.4, 0.25, 0.1],
    [0.45, 0.8, -0.2],
    [0.05, 0.3, 1.1],
  ];
  const profileNormal = rayNormalize([0.27, 0.93, -0.24]);
  const tangentA = rayNormalize([profileNormal[1], -profileNormal[0], 0]);
  const tangentB = rayNormalize([
    profileNormal[0] * profileNormal[2],
    profileNormal[1] * profileNormal[2],
    -(profileNormal[0] ** 2 + profileNormal[1] ** 2),
  ]);
  const worldNormal = profileNormalToWorld(profileNormal, profileToWorld);
  close(rayDot(worldNormal, rayMatVec(profileToWorld, tangentA)), 0);
  close(rayDot(worldNormal, rayMatVec(profileToWorld, tangentB)), 0);
  const incorrectlyForwardTransformed = rayNormalize(rayMatVec(profileToWorld, profileNormal));
  assert.ok(Math.abs(rayDot(incorrectlyForwardTransformed, rayMatVec(profileToWorld, tangentA))) > 0.1);
});

test('vertical origin occupancy is finite while nonzero projected travel is undefined', () => {
  const profileRay = makeProfileRay([0, -1, 0], [
    [1, 0, 0],
    [0.2, 1, -0.1],
    [0, 0, 1],
  ]);
  close(worldDeltaFromProjectedDistance(0, profileRay), 0);
  assert.throws(() => worldDeltaFromProjectedDistance(0.1, profileRay), /perpendicular ray/);
});
