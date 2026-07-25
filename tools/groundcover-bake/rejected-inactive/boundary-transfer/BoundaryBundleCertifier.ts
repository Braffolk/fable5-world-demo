/** Conservative offline sufficient certificate for one 4D boundary-ray cell. */

import {
  TriangleBvh,
  type CensusVec3,
  type DecodedOwnedProfileGeometry,
} from '../../ExteriorClosureCensus';
import type { OriginAwareTruthHit } from '../../OriginAwareRayTruth';

const EPSILON = 2e-8;

export interface BoundaryBundleCertificate {
  readonly plane: readonly [number, number, number, number];
  readonly supportMarginMetres: number;
  readonly incidenceMargin: number;
  readonly orderMarginMetres: number;
  readonly hitPointRadiusMetres: number;
  readonly rayParameterRadiusMetres: number;
  readonly broadPhaseTriangles: number;
}

function add(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function subtract(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(a: CensusVec3, amount: number): CensusVec3 {
  return [a[0] * amount, a[1] * amount, a[2] * amount];
}

function dot(a: CensusVec3, b: CensusVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: CensusVec3, b: CensusVec3): CensusVec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length(a: CensusVec3): number { return Math.hypot(a[0], a[1], a[2]); }

function normalized(a: CensusVec3): CensusVec3 {
  const magnitude = length(a);
  return magnitude > 0 ? scale(a, 1 / magnitude) : [0, 1, 0];
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function vertex(geometry: DecodedOwnedProfileGeometry, triangleId: number, corner: number): CensusVec3 {
  const index = geometry.triangles[triangleId * 3 + corner]! * 3;
  return [geometry.positions[index]!, geometry.positions[index + 1]!, geometry.positions[index + 2]!];
}

function copiedVertex(
  geometry: DecodedOwnedProfileGeometry,
  triangleId: number,
  corner: number,
  copyX: number,
  copyZ: number,
): CensusVec3 {
  const point = vertex(geometry, triangleId, corner);
  return [
    point[0] + copyX * geometry.tileSizeX,
    point[1],
    point[2] + copyZ * geometry.tileSizeZ,
  ];
}

function pointSegmentDistance(point: CensusVec3, a: CensusVec3, b: CensusVec3): number {
  const edge = subtract(b, a);
  const denominator = dot(edge, edge);
  const t = denominator > 0 ? clamp(dot(subtract(point, a), edge) / denominator, 0, 1) : 0;
  return length(subtract(point, add(a, scale(edge, t))));
}

function segmentSegmentDistance(a0: CensusVec3, a1: CensusVec3, b0: CensusVec3, b1: CensusVec3): number {
  const u = subtract(a1, a0);
  const v = subtract(b1, b0);
  const w = subtract(a0, b0);
  const a = dot(u, u);
  const b = dot(u, v);
  const c = dot(v, v);
  const d = dot(u, w);
  const e = dot(v, w);
  const determinant = a * c - b * b;
  let sN = 0;
  let sD = determinant;
  let tN = 0;
  let tD = determinant;
  if (determinant < 1e-30) {
    sN = 0;
    sD = 1;
    tN = e;
    tD = c;
  } else {
    sN = b * e - c * d;
    tN = a * e - b * d;
    if (sN < 0) { sN = 0; tN = e; tD = c; }
    else if (sN > sD) { sN = sD; tN = e + b; tD = c; }
  }
  if (tN < 0) {
    tN = 0;
    if (-d < 0) sN = 0;
    else if (-d > a) sN = sD;
    else { sN = -d; sD = a; }
  } else if (tN > tD) {
    tN = tD;
    if (-d + b < 0) sN = 0;
    else if (-d + b > a) sN = sD;
    else { sN = -d + b; sD = a; }
  }
  const sc = Math.abs(sN) < 1e-30 ? 0 : sN / sD;
  const tc = Math.abs(tN) < 1e-30 ? 0 : tN / tD;
  return length(subtract(add(w, scale(u, sc)), scale(v, tc)));
}

function pointTriangleDistance(point: CensusVec3, a: CensusVec3, b: CensusVec3, c: CensusVec3): number {
  // Ericson's closest-point regions, with an edge fallback for degenerate input.
  const ab = subtract(b, a);
  const ac = subtract(c, a);
  const ap = subtract(point, a);
  const d1 = dot(ab, ap);
  const d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0) return length(ap);
  const bp = subtract(point, b);
  const d3 = dot(ab, bp);
  const d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3) return length(bp);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) return pointSegmentDistance(point, a, b);
  const cp = subtract(point, c);
  const d5 = dot(ab, cp);
  const d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6) return length(cp);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) return pointSegmentDistance(point, a, c);
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) return pointSegmentDistance(point, b, c);
  const normal = cross(ab, ac);
  const magnitude = length(normal);
  return magnitude > 0 ? Math.abs(dot(ap, normal)) / magnitude : Math.min(
    pointSegmentDistance(point, a, b),
    pointSegmentDistance(point, b, c),
    pointSegmentDistance(point, c, a),
  );
}

function segmentTriangleDistance(
  s0: CensusVec3,
  s1: CensusVec3,
  a: CensusVec3,
  b: CensusVec3,
  c: CensusVec3,
): number {
  const direction = subtract(s1, s0);
  const e1 = subtract(b, a);
  const e2 = subtract(c, a);
  const p = cross(direction, e2);
  const determinant = dot(e1, p);
  if (Math.abs(determinant) > 1e-30) {
    const inverse = 1 / determinant;
    const tvec = subtract(s0, a);
    const u = dot(tvec, p) * inverse;
    const q = cross(tvec, e1);
    const v = dot(direction, q) * inverse;
    const t = dot(e2, q) * inverse;
    if (u >= 0 && v >= 0 && u + v <= 1 && t >= 0 && t <= 1) return 0;
  }
  return Math.min(
    pointTriangleDistance(s0, a, b, c),
    pointTriangleDistance(s1, a, b, c),
    segmentSegmentDistance(s0, s1, a, b),
    segmentSegmentDistance(s0, s1, b, c),
    segmentSegmentDistance(s0, s1, c, a),
  );
}

function supportRadius(point: CensusVec3, a: CensusVec3, b: CensusVec3, c: CensusVec3): number {
  const normal = normalized(cross(subtract(b, a), subtract(c, a)));
  const edgeDistance = (p0: CensusVec3, p1: CensusVec3): number => {
    const edge = subtract(p1, p0);
    return Math.abs(dot(subtract(point, p0), cross(normal, normalized(edge))));
  };
  return Math.min(edgeDistance(a, b), edgeDistance(b, c), edgeDistance(c, a));
}

/**
 * Prove one center winner remains the first hit for the complete product cell.
 *
 * `boundaryRadius` bounds every q in the boundary cell around q0.
 * `directionRadius` bounds ||d-d0|| for every Lambert direction in the cell.
 * The proof is deliberately strict: the pre-hit swept bundle must be a clear
 * capsule except for the selected triangle, and the complete possible hit
 * disk must remain inside that triangle.
 */
export function certifyBoundaryRayBundle(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  q0: CensusVec3,
  d0: CensusVec3,
  winner: OriginAwareTruthHit,
  boundaryRadius: number,
  directionRadius: number,
): BoundaryBundleCertificate | null {
  const a = copiedVertex(geometry, winner.triangleId, 0, winner.copyX, winner.copyZ);
  const b = copiedVertex(geometry, winner.triangleId, 1, winner.copyX, winner.copyZ);
  const c = copiedVertex(geometry, winner.triangleId, 2, winner.copyX, winner.copyZ);
  const n = normalized(cross(subtract(b, a), subtract(c, a)));
  const incidence = Math.abs(dot(n, d0));
  const incidenceMargin = incidence - directionRadius;
  if (!(incidenceMargin > EPSILON)) return null;
  const parameterRadius = (boundaryRadius + winner.t * directionRadius) / incidenceMargin + EPSILON;
  const hitRadius = boundaryRadius + parameterRadius + (winner.t + parameterRadius) * directionRadius + EPSILON;
  const centerHit = add(q0, scale(d0, winner.t));
  const rawSupport = supportRadius(centerHit, a, b, c);
  const supportMargin = rawSupport - hitRadius;
  if (!(supportMargin > EPSILON)) return null;

  const preHitT = Math.max(0, winner.t - parameterRadius);
  const segmentEnd = add(q0, scale(d0, preHitT));
  const bundleRadius = boundaryRadius + preHitT * directionRadius + EPSILON;
  const corridorMinimum: CensusVec3 = [
    Math.min(q0[0], segmentEnd[0]) - bundleRadius,
    Math.min(q0[1], segmentEnd[1]) - bundleRadius,
    Math.min(q0[2], segmentEnd[2]) - bundleRadius,
  ];
  const corridorMaximum: CensusVec3 = [
    Math.max(q0[0], segmentEnd[0]) + bundleRadius,
    Math.max(q0[1], segmentEnd[1]) + bundleRadius,
    Math.max(q0[2], segmentEnd[2]) + bundleRadius,
  ];
  const minCopyX = Math.ceil((corridorMinimum[0] - geometry.bounds.max[0]) / geometry.tileSizeX - EPSILON);
  const maxCopyX = Math.floor((corridorMaximum[0] - geometry.bounds.min[0]) / geometry.tileSizeX + EPSILON);
  const minCopyZ = Math.ceil((corridorMinimum[2] - geometry.bounds.max[2]) / geometry.tileSizeZ - EPSILON);
  const maxCopyZ = Math.floor((corridorMaximum[2] - geometry.bounds.min[2]) / geometry.tileSizeZ + EPSILON);
  let minimumClearance = Number.POSITIVE_INFINITY;
  let broadPhaseTriangles = 0;
  for (let copyZ = minCopyZ; copyZ <= maxCopyZ; copyZ++) {
    for (let copyX = minCopyX; copyX <= maxCopyX; copyX++) {
      const offsetX = copyX * geometry.tileSizeX;
      const offsetZ = copyZ * geometry.tileSizeZ;
      const localMinimum: CensusVec3 = [corridorMinimum[0] - offsetX, corridorMinimum[1], corridorMinimum[2] - offsetZ];
      const localMaximum: CensusVec3 = [corridorMaximum[0] - offsetX, corridorMaximum[1], corridorMaximum[2] - offsetZ];
      broadPhaseTriangles += bvh.forEachTriangleOverlappingBounds(localMinimum, localMaximum, (triangleId) => {
        if (triangleId === winner.triangleId && copyX === winner.copyX && copyZ === winner.copyZ) return;
        const ta = copiedVertex(geometry, triangleId, 0, copyX, copyZ);
        const tb = copiedVertex(geometry, triangleId, 1, copyX, copyZ);
        const tc = copiedVertex(geometry, triangleId, 2, copyX, copyZ);
        minimumClearance = Math.min(minimumClearance, segmentTriangleDistance(q0, segmentEnd, ta, tb, tc));
      });
    }
  }
  const orderMargin = minimumClearance - bundleRadius;
  if (!(orderMargin > EPSILON)) return null;
  return Object.freeze({
    plane: Object.freeze([n[0], n[1], n[2], dot(n, a)] as const),
    supportMarginMetres: supportMargin,
    incidenceMargin,
    orderMarginMetres: orderMargin,
    hitPointRadiusMetres: hitRadius,
    rayParameterRadiusMetres: parameterRadius,
    broadPhaseTriangles,
  });
}
