import {
  GroundCoverBoundaryFace,
  GroundCoverPatchFlags,
  type GroundCoverBoundaryFaceValue,
  type GroundCoverFixedChart,
  type GroundCoverCarrierPatch,
  type GroundCoverWinnerPlane,
} from './GroundCoverCarrierFormat';

export interface CarrierRay {
  readonly origin: readonly [number, number, number];
  /** Must be finite and non-zero. It need not already be normalized. */
  readonly direction: readonly [number, number, number];
}

export interface CarrierEntry {
  readonly distance: number;
  readonly exitDistance: number;
  readonly chartId: number;
  readonly patchId: number;
  readonly communityState: number;
  readonly boundaryFace: GroundCoverBoundaryFaceValue;
  /** Categorical face token: patchId in the high bits, face in the low 3. */
  readonly boundaryToken: number;
  readonly entryTransferClass: number;
  readonly closureCertificate: number;
  readonly insideFade: false;
}

export interface CarrierInsideFade {
  readonly insideFade: true;
  readonly chartId: number;
  readonly patchId: number;
}


function normalize(direction: readonly [number, number, number]): readonly [number, number, number] {
  const length = Math.hypot(direction[0], direction[1], direction[2]);
  if (!(length > 0) || !Number.isFinite(length)) throw new Error('carrier ray direction is invalid');
  return [direction[0] / length, direction[1] / length, direction[2] / length];
}

function halfOpenContains(value: number, low: number, high: number): boolean {
  return value >= low && value < high;
}

/**
 * Exact scalar reference for one analytic rectangular carrier component.
 *
 * This is a fixed six-slab calculation, not a march or plant query. Parallel
 * components are handled categorically, so exact vertical and horizontal rays
 * never divide by an epsilon. The deterministic face order is X, height, Z.
 */
export function intersectGroundCoverCarrier(
  chart: GroundCoverFixedChart,
  patch: GroundCoverCarrierPatch,
  ray: CarrierRay,
  horizonMetres: number,
): CarrierEntry | CarrierInsideFade | null {
  if (patch.chartId !== chart.chartId) throw new Error('carrier patch/chart mismatch');
  if (!(horizonMetres > 0) || !Number.isFinite(horizonMetres)) {
    throw new Error('carrier horizon must be finite and positive');
  }
  const d = normalize(ray.direction);
  const lx = ray.origin[0] - chart.originX;
  const lz = ray.origin[2] - chart.originZ;
  const h = ray.origin[1] - (
    chart.groundHeight + chart.slopeX * lx + chart.slopeZ * lz
  );
  const dh = d[1] - chart.slopeX * d[0] - chart.slopeZ * d[2];
  const origin = [lx, h, lz] as const;
  const direction = [d[0], dh, d[2]] as const;
  const fullPlane = (patch.flags & GroundCoverPatchFlags.FullPeriodicPlane) !== 0;
  const minimum = [
    fullPlane ? Number.NEGATIVE_INFINITY : patch.minX,
    patch.minHeight,
    fullPlane ? Number.NEGATIVE_INFINITY : patch.minZ,
  ] as const;
  const maximum = [
    fullPlane ? Number.POSITIVE_INFINITY : patch.maxX,
    patch.maxHeight,
    fullPlane ? Number.POSITIVE_INFINITY : patch.maxZ,
  ] as const;

  if (
    (fullPlane || halfOpenContains(origin[0], minimum[0], maximum[0]))
    && halfOpenContains(origin[1], minimum[1], maximum[1])
    && (fullPlane || halfOpenContains(origin[2], minimum[2], maximum[2]))
  ) return Object.freeze({ insideFade: true, chartId: chart.chartId, patchId: patch.patchId });

  let near = Number.NEGATIVE_INFINITY;
  let far = Number.POSITIVE_INFINITY;
  let nearFace: GroundCoverBoundaryFaceValue = GroundCoverBoundaryFace.MinX;
  const minFaces = [
    GroundCoverBoundaryFace.MinX,
    GroundCoverBoundaryFace.MinHeight,
    GroundCoverBoundaryFace.MinZ,
  ] as const;
  const maxFaces = [
    GroundCoverBoundaryFace.MaxX,
    GroundCoverBoundaryFace.MaxHeight,
    GroundCoverBoundaryFace.MaxZ,
  ] as const;
  for (let axis = 0; axis < 3; axis++) {
    if (fullPlane && axis !== 1) continue;
    const component = direction[axis]!;
    if (component === 0) {
      if (!halfOpenContains(origin[axis]!, minimum[axis]!, maximum[axis]!)) return null;
      continue;
    }
    let t0 = (minimum[axis]! - origin[axis]!) / component;
    let t1 = (maximum[axis]! - origin[axis]!) / component;
    let face0: GroundCoverBoundaryFaceValue = minFaces[axis]!;
    let face1: GroundCoverBoundaryFaceValue = maxFaces[axis]!;
    if (t0 > t1) {
      [t0, t1] = [t1, t0];
      [face0, face1] = [face1, face0];
    }
    // Frozen total order: lower face id wins an exact edge/corner tie.
    if (t0 > near || (t0 === near && face0 < nearFace)) {
      near = t0;
      nearFace = face0;
    }
    far = Math.min(far, t1);
    if (near > far) return null;
  }
  if (!(near >= 0 && near <= far && near <= horizonMetres)) return null;
  return Object.freeze({
    distance: near,
    exitDistance: Math.min(far, horizonMetres),
    chartId: chart.chartId,
    patchId: patch.patchId,
    communityState: patch.communityState,
    boundaryFace: nearFace,
    boundaryToken: ((patch.patchId << 3) | nearFace) >>> 0,
    entryTransferClass: patch.entryTransferClass,
    closureCertificate: patch.closureCertificate,
    insideFade: false,
  });
}

/** Apply exact packed-terrain/root translation to one elected source plane. */
export function transformWinnerPlaneAtRoot(
  chart: GroundCoverFixedChart,
  winner: GroundCoverWinnerPlane,
  packedRootHeight: number,
  heightScale = 1,
  windShearX = 0,
  windShearZ = 0,
): { readonly normal: readonly [number, number, number]; readonly constant: number } {
  if (winner.chartId !== chart.chartId) throw new Error('winner/chart mismatch');
  if (![packedRootHeight, heightScale, windShearX, windShearZ].every(Number.isFinite)) {
    throw new Error('winner root transform is non-finite');
  }
  if (!(heightScale > 0)) throw new Error('winner height scale must be positive');
  // Source local -> world linear map for world-up, root-fixed affine shear:
  // x'=x+windShearX*y, y'=heightScale*y, z'=z+windShearZ*y.
  // A plane transforms by A^{-T} n and translates by the exact packed root.
  const ny = (
    winner.normalY
    - windShearX * winner.normalX
    - windShearZ * winner.normalZ
  ) / heightScale;
  const normal = [winner.normalX, ny, winner.normalZ] as const;
  const rootWorldX = chart.originX + winner.rootX;
  const rootWorldZ = chart.originZ + winner.rootZ;
  const rootWorldY = packedRootHeight;
  const constant = winner.planeConstant
    + normal[0] * rootWorldX
    + normal[1] * rootWorldY
    + normal[2] * rootWorldZ;
  return Object.freeze({ normal, constant });
}

/** Intersect the transformed elected plane with the original live world ray. */
export function intersectWinnerPlane(
  ray: CarrierRay,
  normal: readonly [number, number, number],
  constant: number,
): number | null {
  const d = normalize(ray.direction);
  const denominator = normal[0] * d[0] + normal[1] * d[1] + normal[2] * d[2];
  if (denominator === 0) return null;
  const numerator = constant - (
    normal[0] * ray.origin[0]
    + normal[1] * ray.origin[1]
    + normal[2] * ray.origin[2]
  );
  const distance = numerator / denominator;
  return Number.isFinite(distance) && distance >= 0 ? distance : null;
}
