/**
 * Runtime-only contract for the exterior common-slab/boundary-transfer model.
 *
 * This file deliberately contains no asset parser and no fallback approximation.
 * A loader may expose this contract only after its cook has produced the fixed
 * chart, carrier-owner, regular-winner, mixed-appearance, and certificate data
 * named below.  The query is fixed work: one chart, one carrier record, one
 * transfer record, and one winner-root correction.  There is no traversal,
 * candidate list, ray march, or runtime ground-cover geometry.
 *
 * Governing mathematics:
 * - docs/tasks/2026-07-23/GRASS-BOX-BOUNDARY-TRANSFER-MATH.md
 * - docs/tasks/2026-07-23/GRASS-BOX-TERRAIN-MOTION-CLOSURE.md
 */

import {
  float,
  normalize,
  smoothstep,
  uint,
  vec2,
  vec3,
} from 'three/tsl';
import type { NB, NF, NU, NV2, NV3, NV4 } from '../../gpu/TSLTypes';
import type { LoadedGroundCoverCarrierClosure } from './GroundCoverCarrierFormat';
import type { LoadedGbr4V4Profile } from './GroundCoverGbr4V4';

export const GROUND_COVER_TRANSFER_MISS = 0;
export const GROUND_COVER_TRANSFER_REGULAR = 1;
export const GROUND_COVER_TRANSFER_MIXED = 2;

/** One cooked, world-fixed affine terrain chart.  `anchorXZ`, `groundY`, and
 * `slopeXZ` never follow the camera or an elected scene surface. */
export interface GroundCoverFixedChartSample {
  valid: NB;
  chartId: NU;
  anchorXZ: NV2;
  groundY: NF;
  slopeXZ: NV2;
  carrierMinH: NF;
  carrierMaxH: NF;
}

/** Categorical first boundary primitive of a finite control mask.  Its live
 * supporting-line intersection reconstructs the entry exactly; `stateId` is
 * the complete jointly baked community selected before botanical transfer. */
export interface GroundCoverCarrierOwnerSample {
  valid: NB;
  boundaryFace: NU;
  lineNormal: NV2;
  lineOffset: NF;
  spanAxis: NV2;
  spanMin: NF;
  spanMax: NF;
  stateId: NU;
  chartId: NU;
  /** Cook proof: the edge-conditioned transfer either hits before patch exit,
   * or no later patch can contribute inside the finite horizon. */
  firstEntryClosed: NB;
}

/** Complete transfer payload.  Runtime records always contain both layouts;
 * `kind` selects exactly one.  Unused fields are inert container padding, not
 * alternate candidates.
 *
 * Regular records carry a coupled categorical token.  Mixed records carry
 * filtered radiance/coverage and a conservative interval only; they do not
 * contain a representative depth or normal and may never enter surface
 * lighting as if they did. */
export interface GroundCoverTransferSample {
  kind: NU;
  regularCertified: NB;
  stateId: NU;
  chartId: NU;
  ownerToken: NU;
  copyToken: NU;
  planeToken: NU;
  rootToken: NU;
  materialToken: NU;
  sourcePlaneNormal: NV3;
  sourcePlaneOffset: NF;
  sourceRootOffsetXZ: NV2;
  sourceRootHeight: NF;
  periodicCopyXZ: NV2;
  deformationToken: NU;
  supportMargin: NF;
  incidenceMargin: NF;
  orderMargin: NF;
  terrainMotionMargin: NF;
  regularAlbedo: NV3;
  triangleDomainToken: NU;
  mixedPremulRadiance: NV3;
  mixedCoverage: NF;
  mixedTauNear: NF;
  mixedTauFar: NF;
  mixedNormalMoment: NV3;
  mixedMaterialMoment: NV4;
}

/** Winner-root correction output.  The callback evaluates packed terrain and
 * smooth affine deformation at `rootToken`, transforms that one source plane,
 * and carries the matching categorical attributes. */
export interface GroundCoverWinnerPlaneSample {
  valid: NB;
  firstnessCertified: NB;
  domainValid: NB;
  worldNormal: NV3;
  worldPlaneOffset: NF;
  albedo: NV3;
  materialToken: NU;
  normalizedHeight: NF;
}

/** Resource-facing operations.  Every function is one direct fixed lookup (or
 * an analytic token decode).  An implementation which walks mask cells,
 * retries successors, blends owners, or searches corrections does not satisfy
 * this interface even if it has the same TypeScript shape. */
export interface GroundCoverBoundaryResources {
  /** Strict wire resources.  Keeping these on the runtime contract prevents a
   * lookalike v2 depth/normal tensor from being adapted by structural typing. */
  readonly carrierClosure: LoadedGroundCoverCarrierClosure;
  readonly transferProfile: LoadedGbr4V4Profile;
  readonly horizonMetres: number;
  readonly interiorFadeMetres: number;
  /** `true` is the explicitly encoded P=R^2 periodic-plane specialization.
   * It has cap entry only; finite ecological cover must use the owner field. */
  readonly fullPeriodicPlane: boolean;

  selectFixedChart(preferredAnchorXZ: NV2, cameraXZ: NV2): GroundCoverFixedChartSample;
  carrierOccupancy(chartId: NU, worldXZ: NV2): NB;
  carrierInteriorDistance(chartId: NU, worldXZ: NV2): NF;
  communityStateAt(chartId: NU, worldXZ: NV2): { valid: NB; stateId: NU };
  firstCarrierOwner(
    chartId: NU,
    qWorldXZ: NV2,
    omegaWorldXZ: NV2,
    /** True only for the fixed eta/rho inside-union continuation contract. */
    strictAfterExit: NB,
  ): GroundCoverCarrierOwnerSample;
  lookupTransfer(
    chartId: NU,
    stateId: NU,
    boundaryFace: NU,
    entryWorld: NV3,
    canonicalDirection: NV3,
    pixelFootprintWorld: NF,
  ): GroundCoverTransferSample;
  resolveWinnerPlane(
    transfer: GroundCoverTransferSample,
    chart: GroundCoverFixedChartSample,
  ): GroundCoverWinnerPlaneSample;
}

export interface GroundCoverBoundaryQueryInput {
  rayOrigin: NV3;
  rayDirection: NV3;
  /** Optional opaque cutoff.  Its value is ignored when `hasSceneCutoff=false`. */
  sceneCutoffMetres: NF;
  hasSceneCutoff: NB;
  /** Preferred fixed-chart anchor when a real scene endpoint exists. */
  preferredAnchorXZ: NV2;
  /** Full angular pitch of one output pixel.  The query converts it to the
   * correlated boundary footprint using the live carrier-entry standoff. */
  pixelAngleRadians: NF;
}

export interface GroundCoverBoundaryQueryResult {
  regularHit: NB;
  regularDistance: NF;
  regularNormal: NV3;
  regularAlbedo: NV3;
  regularMaterial: NU;
  regularNormalizedHeight: NF;
  mixedHit: NB;
  mixedPremulRadiance: NV3;
  mixedCoverage: NF;
  mixedDistanceNear: NF;
  mixedDistanceFar: NF;
  /** One at an exterior origin; smoothly reaches zero only after entering the
   * declared guarded P x I carrier. */
  laneFade: NF;
}

function chooseF(test: NB, yes: NF, no: NF): NF {
  return (test as unknown as { select(a: unknown, b: unknown): NF }).select(yes, no);
}

function chooseU(test: NB, yes: NU, no: NU): NU {
  return (test as unknown as { select(a: unknown, b: unknown): NU }).select(yes, no);
}

function chooseV2(test: NB, yes: NV2, no: NV2): NV2 {
  return (test as unknown as { select(a: unknown, b: unknown): NV2 }).select(yes, no);
}

/** Exact fixed-cost transcription of the continuous exterior factorisation.
 * Divisors use exact categorical zero cases; a safe divisor is selected only
 * to keep the inactive arithmetic finite.  No epsilon changes ownership or
 * rejects horizontal/vertical rays. */
export function queryGroundCoverBoundary(
  resources: GroundCoverBoundaryResources,
  input: GroundCoverBoundaryQueryInput,
): GroundCoverBoundaryQueryResult {
  const ro = input.rayOrigin;
  const rd = normalize(input.rayDirection) as unknown as NV3;
  const horizon = float(resources.horizonMetres) as unknown as NF;
  const sceneCutoff = input.sceneCutoffMetres.max(0) as unknown as NF;
  const cutoff = chooseF(
    input.hasSceneCutoff,
    sceneCutoff.min(horizon) as unknown as NF,
    horizon,
  ).toVar() as unknown as NF;
  const chart = resources.selectFixedChart(input.preferredAnchorXZ, ro.xz as unknown as NV2);

  // The fixed chart maps world (x,z,y) to canonical (u.x,h,u.y).  Curved
  // terrain is corrected only after the categorical winner names its root.
  const cameraU = (ro.xz as unknown as NV2).sub(chart.anchorXZ) as unknown as NV2;
  const cameraGround = chart.groundY.add(chart.slopeXZ.dot(cameraU)) as unknown as NF;
  const cameraH = (ro.y as unknown as NF).sub(cameraGround).toVar() as unknown as NF;
  const k = (rd.y as unknown as NF)
    .sub(chart.slopeXZ.x.mul(rd.x))
    .sub(chart.slopeXZ.y.mul(rd.z))
    .toVar() as unknown as NF;

  const kZero = k.equal(float(0)) as unknown as NB;
  const safeK = chooseF(kZero, float(1) as unknown as NF, k).toVar() as unknown as NF;
  const cap0 = chart.carrierMinH.sub(cameraH).div(safeK).toVar() as unknown as NF;
  const cap1 = chart.carrierMaxH.sub(cameraH).div(safeK).toVar() as unknown as NF;
  const capNear = cap0.min(cap1).toVar() as unknown as NF;
  const capFar = cap0.max(cap1).toVar() as unknown as NF;
  const heightInside = cameraH.greaterThanEqual(chart.carrierMinH)
    .and(cameraH.lessThanEqual(chart.carrierMaxH) as unknown as NB) as unknown as NB;
  const slabEnter = chooseF(kZero, float(0) as unknown as NF, capNear.max(0) as unknown as NF)
    .toVar() as unknown as NF;
  const slabExit = chooseF(kZero, cutoff, capFar.min(cutoff) as unknown as NF)
    .toVar() as unknown as NF;
  const slabValid = chart.valid
    .and((kZero as unknown as { select(a: unknown, b: unknown): NB }).select(
      heightInside,
      slabEnter.lessThanEqual(slabExit) as unknown as NB,
    ))
    .and(cutoff.greaterThanEqual(0) as unknown as NB) as unknown as NB;

  const cameraOccupied = resources.carrierOccupancy(
    chart.chartId,
    ro.xz as unknown as NV2,
  );
  const cameraInside = heightInside.and(cameraOccupied) as unknown as NB;
  const insideDistance = resources.carrierInteriorDistance(
    chart.chartId,
    ro.xz as unknown as NV2,
  ).max(0) as unknown as NF;
  const fadeWidth = Math.max(resources.interiorFadeMetres, Number.EPSILON);
  const insideFade = float(1).sub(smoothstep(0, fadeWidth, insideDistance)) as unknown as NF;
  const laneFade = chooseF(cameraInside, insideFade, float(1) as unknown as NF)
    .clamp(0, 1).toVar() as unknown as NF;

  const qAtSlab = (ro.xz as unknown as NV2)
    .add((rd.xz as unknown as NV2).mul(slabEnter)) as unknown as NV2;
  const horizontalSpeed = (rd.xz as unknown as NV2).length().toVar() as unknown as NF;
  const verticalPole = horizontalSpeed.equal(float(0)) as unknown as NB;
  const safeHorizontalSpeed = chooseF(
    verticalPole,
    float(1) as unknown as NF,
    horizontalSpeed,
  ).toVar() as unknown as NF;
  const omega = chooseV2(
    verticalPole,
    vec2(1, 0) as unknown as NV2,
    (rd.xz as unknown as NV2).div(safeHorizontalSpeed) as unknown as NV2,
  ).toVar() as unknown as NV2;
  const qOccupied = resources.fullPeriodicPlane
    ? (slabValid as unknown as NB)
    : resources.carrierOccupancy(chart.chartId, qAtSlab);

  // Finite masks return one categorical boundary owner.  The full periodic
  // plane specialization encodes P=R^2 explicitly and therefore has ell=0.
  const owner = resources.firstCarrierOwner(
    chart.chartId,
    qAtSlab,
    omega,
    cameraInside,
  );
  const ownerDen = owner.lineNormal.dot(omega).toVar() as unknown as NF;
  const ownerParallel = ownerDen.equal(float(0)) as unknown as NB;
  const safeOwnerDen = chooseF(
    ownerParallel,
    float(1) as unknown as NF,
    ownerDen,
  ).toVar() as unknown as NF;
  const ell = owner.lineOffset.sub(owner.lineNormal.dot(qAtSlab))
    .div(safeOwnerDen).toVar() as unknown as NF;
  const sidePoint = qAtSlab.add(omega.mul(ell)) as unknown as NV2;
  const spanCoord = owner.spanAxis.dot(sidePoint).toVar() as unknown as NF;
  const ownerLiveValid = owner.valid
    .and((ownerParallel as unknown as { not(): NB }).not())
    .and(ell.greaterThanEqual(0) as unknown as NB)
    .and(spanCoord.greaterThanEqual(owner.spanMin) as unknown as NB)
    .and(spanCoord.lessThanEqual(owner.spanMax) as unknown as NB)
    .and(owner.firstEntryClosed) as unknown as NB;

  const capEntry = verticalPole.or(qOccupied) as unknown as NB;
  const fullPlaneEntry = resources.fullPeriodicPlane ? (slabValid as unknown as NB) : (uint(0).equal(1) as unknown as NB);
  const useCap = fullPlaneEntry.or(capEntry) as unknown as NB;
  const sideDistance = slabEnter.add(ell.div(safeHorizontalSpeed)).toVar() as unknown as NF;
  const entryDistance = chooseF(useCap, slabEnter, sideDistance).toVar() as unknown as NF;
  const finiteCarrierValid = (useCap as unknown as { select(a: unknown, b: unknown): NB }).select(
    slabValid.and(qOccupied) as unknown as NB,
    slabValid.and(ownerLiveValid) as unknown as NB,
  );
  // A camera in the full-periodic carrier has no later component in this lane;
  // it fades instead of inventing a cap behind the camera.  Finite-mask adapters
  // may implement the fixed eta/rho continuation through `strictAfterExit`.
  const insideAllowed = resources.fullPeriodicPlane
    ? (cameraInside as unknown as { not(): NB }).not()
    : (uint(0).equal(0) as unknown as NB);
  const carrierValid = finiteCarrierValid
    .and(insideAllowed)
    .and(entryDistance.greaterThanEqual(0) as unknown as NB)
    .and(entryDistance.lessThanEqual(slabExit) as unknown as NB)
    .and(entryDistance.lessThanEqual(cutoff) as unknown as NB) as unknown as NB;

  const entryXZ = (ro.xz as unknown as NV2)
    .add((rd.xz as unknown as NV2).mul(entryDistance)) as unknown as NV2;
  const entryWorld = ro.add(rd.mul(entryDistance)).toVar() as unknown as NV3;
  const capState = resources.communityStateAt(chart.chartId, entryXZ);
  const stateId = chooseU(useCap, capState.stateId, owner.stateId).toVar() as unknown as NU;
  const stateValid = (useCap as unknown as { select(a: unknown, b: unknown): NB }).select(
    capState.valid,
    ownerLiveValid,
  );

  const metricSpeed = vec3(rd.x, k, rd.z).length().toVar() as unknown as NF;
  const metricZero = metricSpeed.equal(float(0)) as unknown as NB;
  const safeMetricSpeed = chooseF(
    metricZero,
    float(1) as unknown as NF,
    metricSpeed,
  ).toVar() as unknown as NF;
  const canonicalDirection = normalize(vec3(rd.x, k, rd.z).div(safeMetricSpeed)) as unknown as NV3;
  const transfer = resources.lookupTransfer(
    chart.chartId,
    stateId,
    chooseU(
      useCap,
      chooseU(
        k.lessThan(0) as unknown as NB,
        uint(3) as unknown as NU,
        uint(2) as unknown as NU,
      ),
      owner.boundaryFace,
    ),
    entryWorld,
    canonicalDirection,
    entryDistance.mul(input.pixelAngleRadians.mul(0.5).tan()).mul(2) as unknown as NF,
  );
  const transferHeaderValid = carrierValid
    .and(stateValid)
    .and((metricZero as unknown as { not(): NB }).not())
    .and(transfer.chartId.equal(chart.chartId) as unknown as NB)
    .and(transfer.stateId.equal(stateId) as unknown as NB) as unknown as NB;

  const winner = resources.resolveWinnerPlane(transfer, chart);
  const planeDen = winner.worldNormal.dot(rd).toVar() as unknown as NF;
  const planeParallel = planeDen.equal(float(0)) as unknown as NB;
  const safePlaneDen = chooseF(
    planeParallel,
    float(1) as unknown as NF,
    planeDen,
  ).toVar() as unknown as NF;
  const regularDistance = winner.worldPlaneOffset
    .sub(winner.worldNormal.dot(ro))
    .div(safePlaneDen).toVar() as unknown as NF;
  const regularHit = transferHeaderValid
    .and(transfer.kind.equal(uint(GROUND_COVER_TRANSFER_REGULAR)) as unknown as NB)
    .and(transfer.regularCertified)
    .and(winner.valid)
    .and(winner.firstnessCertified)
    .and(winner.domainValid)
    .and((planeParallel as unknown as { not(): NB }).not())
    .and(regularDistance.greaterThanEqual(entryDistance) as unknown as NB)
    .and(regularDistance.lessThanEqual(cutoff) as unknown as NB) as unknown as NB;

  const mixedNear = entryDistance.add(
    transfer.mixedTauNear.div(safeMetricSpeed),
  ).toVar() as unknown as NF;
  const mixedFar = entryDistance.add(
    transfer.mixedTauFar.div(safeMetricSpeed),
  ).toVar() as unknown as NF;
  // A mixed interval is accepted only when wholly in front of the opaque
  // cutoff.  An overlap needs a jointly filtered scene/cover record; choosing a
  // representative depth here would recreate the forbidden mosaic surface.
  const mixedHit = transferHeaderValid
    .and(transfer.kind.equal(uint(GROUND_COVER_TRANSFER_MIXED)) as unknown as NB)
    .and(transfer.mixedCoverage.greaterThan(0) as unknown as NB)
    .and(transfer.mixedTauNear.greaterThanEqual(0) as unknown as NB)
    .and(transfer.mixedTauFar.greaterThanEqual(transfer.mixedTauNear) as unknown as NB)
    .and(mixedFar.lessThanEqual(cutoff) as unknown as NB) as unknown as NB;

  return {
    regularHit,
    regularDistance,
    regularNormal: winner.worldNormal,
    regularAlbedo: winner.albedo,
    regularMaterial: winner.materialToken,
    regularNormalizedHeight: winner.normalizedHeight,
    mixedHit,
    mixedPremulRadiance: transfer.mixedPremulRadiance,
    mixedCoverage: transfer.mixedCoverage.clamp(0, 1) as unknown as NF,
    mixedDistanceNear: mixedNear,
    mixedDistanceFar: mixedFar,
    laneFade,
  };
}
