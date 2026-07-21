/**
 * Shader-independent reference math for Sannikov-style precomputed-ray cover.
 *
 * Conventions:
 * - q is profile space; x is world space.
 * - `profileToWorld` is the full, non-normalized, possibly non-orthogonal
 *   Jacobian J = dx/dq, stored row-major.
 * - A ray is x(t) = origin + direction * t. The caller decides whether t is
 *   metres by normalizing `direction`; none of the identities require it.
 */

export type RayVec3 = readonly [number, number, number];
export type RayMat3 = readonly [RayVec3, RayVec3, RayVec3];

export interface Ray3 {
  origin: RayVec3;
  direction: RayVec3;
}

export interface ProfileRay {
  /** Unit direction in the Euclidean metric used by the profile bake. */
  direction: RayVec3;
  /** Profile-distance travelled per unit of the world-ray parameter. */
  speed: number;
  /** Unnormalized dq/dt = inverse(J) * dx/dt. */
  velocity: RayVec3;
}

const EPS = 1e-12;

export function rayDot(a: RayVec3, b: RayVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function rayLength(v: RayVec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

export function rayNormalize(v: RayVec3): RayVec3 {
  const length = rayLength(v);
  if (!(length > EPS)) throw new Error('cannot normalize a zero ray vector');
  return [v[0] / length, v[1] / length, v[2] / length];
}

export function rayPoint(ray: Ray3, t: number): RayVec3 {
  return [
    ray.origin[0] + ray.direction[0] * t,
    ray.origin[1] + ray.direction[1] * t,
    ray.origin[2] + ray.direction[2] * t,
  ];
}

/**
 * Exact conjugate ray for a surface translated by one constant vector.
 *
 * For S_v = { p + v | p in S }:
 *
 *   origin + direction*t in S_v
 *     iff
 *   (origin - v) + direction*t in S.
 *
 * The complete set of ray parameters is therefore unchanged, including the
 * first positive hit, primitive ownership, ties, misses, and finite-domain
 * silhouettes. This is the exact shell-free geometric identity for a rigid
 * ground-cover envelope S + (0,H,0).
 */
export function rayForTranslatedSurface(ray: Ray3, translation: RayVec3): Ray3 {
  return {
    origin: [
      ray.origin[0] - translation[0],
      ray.origin[1] - translation[1],
      ray.origin[2] - translation[2],
    ],
    direction: ray.direction,
  };
}

/** Map the carrier hit back to the translated surface without changing t. */
export function translatedSurfacePoint(
  carrierRay: Ray3,
  translation: RayVec3,
  t: number,
): RayVec3 {
  const point = rayPoint(carrierRay, t);
  return [
    point[0] + translation[0],
    point[1] + translation[1],
    point[2] + translation[2],
  ];
}

export function rayMatVec(matrix: RayMat3, vector: RayVec3): RayVec3 {
  return [
    rayDot(matrix[0], vector),
    rayDot(matrix[1], vector),
    rayDot(matrix[2], vector),
  ];
}

export function rayTranspose(matrix: RayMat3): RayMat3 {
  return [
    [matrix[0][0], matrix[1][0], matrix[2][0]],
    [matrix[0][1], matrix[1][1], matrix[2][1]],
    [matrix[0][2], matrix[1][2], matrix[2][2]],
  ];
}

export function rayInverse(matrix: RayMat3): RayMat3 {
  const [a, b, c] = matrix[0];
  const [d, e, f] = matrix[1];
  const [g, h, i] = matrix[2];
  const A = e * i - f * h;
  const B = c * h - b * i;
  const C = b * f - c * e;
  const D = f * g - d * i;
  const E = a * i - c * g;
  const F = c * d - a * f;
  const G = d * h - e * g;
  const H = b * g - a * h;
  const I = a * e - b * d;
  const determinant = a * A + b * D + c * G;
  if (Math.abs(determinant) <= EPS) throw new Error('profile-to-world basis is singular');
  const inverse = 1 / determinant;
  return [
    [A * inverse, B * inverse, C * inverse],
    [D * inverse, E * inverse, F * inverse],
    [G * inverse, H * inverse, I * inverse],
  ];
}

/** Transform a world ray through the exact inverse of one affine cover chart. */
export function makeProfileRay(
  worldDirection: RayVec3,
  profileToWorld: RayMat3,
): ProfileRay {
  const velocity = rayMatVec(rayInverse(profileToWorld), worldDirection);
  const speed = rayLength(velocity);
  if (!(speed > EPS)) throw new Error('world ray has zero speed in profile space');
  return {
    velocity,
    speed,
    direction: [velocity[0] / speed, velocity[1] / speed, velocity[2] / speed],
  };
}

/** Lift a full 3D distance returned by the profile bake back to world-ray t. */
export function worldDeltaFromProfileDistance(profileDistance: number, profileRay: ProfileRay): number {
  return profileDistance / profileRay.speed;
}

/**
 * Sannikov's |OB| = |OA| / cos(alpha), expressed without assuming an
 * orthonormal basis. The bake measures OA in profile axes `axisA/axisB`; the
 * denominator is the actual projected profile speed, not a world-space cosine.
 */
export function worldDeltaFromProjectedDistance(
  projectedProfileDistance: number,
  profileRay: ProfileRay,
  axisA = 0,
  axisB = 2,
): number {
  const projectedSpeed = Math.hypot(
    profileRay.velocity[axisA] as number,
    profileRay.velocity[axisB] as number,
  );
  if (!(projectedSpeed > EPS)) {
    if (Math.abs(projectedProfileDistance) <= EPS) return 0;
    throw new Error('nonzero projected path is undefined for a perpendicular ray');
  }
  return projectedProfileDistance / projectedSpeed;
}

/**
 * Exact top-envelope parameter in one affine chart. `heightSpeed` and
 * `heightAtGroundPoint` must come from the same inverse basis as the ground
 * point; substituting another heightfield or derivative invalidates the solve.
 */
export function affineEnvelopeEntryFromGround(
  groundT: number,
  heightAtGroundPoint: number,
  topHeight: number,
  heightSpeed: number,
): number {
  if (Math.abs(heightSpeed) <= EPS) throw new Error('ray is parallel to the affine cover envelope');
  return groundT + (topHeight - heightAtGroundPoint) / heightSpeed;
}

export interface BilinearGroundPatch {
  /** g(x,z) = constant + x*x + z*z + xz*x*z. */
  constant: number;
  x: number;
  z: number;
  xz: number;
}

/**
 * Exact roots of y(t) - g(x(t),z(t)) = topHeight for one known bilinear patch.
 * A caller must still verify that a returned root lies inside that patch. This
 * deliberately does not pretend that one patch can solve a piecewise surface.
 */
export function bilinearEnvelopeRoots(
  ray: Ray3,
  ground: BilinearGroundPatch,
  topHeight: number,
): number[] {
  const [ox, oy, oz] = ray.origin;
  const [dx, dy, dz] = ray.direction;
  const g0 = ground.constant + ground.x * ox + ground.z * oz + ground.xz * ox * oz;
  const g1 = ground.x * dx + ground.z * dz + ground.xz * (ox * dz + oz * dx);
  const g2 = ground.xz * dx * dz;
  const a = -g2;
  const b = dy - g1;
  const c = oy - g0 - topHeight;
  if (Math.abs(a) <= EPS) {
    if (Math.abs(b) <= EPS) return [];
    return [-c / b];
  }
  const discriminant = b * b - 4 * a * c;
  if (discriminant < 0) return [];
  if (discriminant <= EPS) return [-b / (2 * a)];
  const root = Math.sqrt(discriminant);
  const q = -0.5 * (b + Math.sign(b || 1) * root);
  const first = q / a;
  const second = c / q;
  return first < second ? [first, second] : [second, first];
}

/** Exact intersection with the infinite plane through `point` with covector `normal`. */
export function rayPlaneT(
  ray: Ray3,
  point: RayVec3,
  normal: RayVec3,
): number {
  const denominator = rayDot(normal, ray.direction);
  if (Math.abs(denominator) <= EPS) throw new Error('ray is parallel to the reconstruction plane');
  const delta: RayVec3 = [
    point[0] - ray.origin[0],
    point[1] - ray.origin[1],
    point[2] - ray.origin[2],
  ];
  return rayDot(normal, delta) / denominator;
}

/**
 * Reproject one canonical first hit onto a live direction while its geometric
 * plane/owner remains valid. Unlike linear depth blending, this is exact for
 * that plane at every non-parallel direction.
 */
export function reprojectCanonicalPlaneHit(
  origin: RayVec3,
  canonicalDirection: RayVec3,
  canonicalDistance: number,
  geometricNormal: RayVec3,
  liveDirection: RayVec3,
): number {
  const canonicalHit: RayVec3 = [
    origin[0] + canonicalDirection[0] * canonicalDistance,
    origin[1] + canonicalDirection[1] * canonicalDistance,
    origin[2] + canonicalDirection[2] * canonicalDistance,
  ];
  return rayPlaneT({ origin, direction: liveDirection }, canonicalHit, geometricNormal);
}

/** Profile normals are covectors and therefore transform by inverse-transpose. */
export function profileNormalToWorld(
  profileNormal: RayVec3,
  profileToWorld: RayMat3,
): RayVec3 {
  return rayNormalize(rayMatVec(rayTranspose(rayInverse(profileToWorld)), profileNormal));
}
