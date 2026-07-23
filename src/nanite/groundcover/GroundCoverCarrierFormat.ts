/**
 * Fixed-chart carrier/control closure for the boundary-transfer ground-cover path.
 *
 * This module deliberately contains no renderer or shader policy.  It is the
 * strict wire parser for the carrier part of the boundary-transfer contract:
 *
 *   - a cooked, world-fixed affine terrain chart;
 *   - one guarded rectangular component of P x I;
 *   - exact cap/side/pole entry with a categorical boundary token;
 *   - a finite world horizon; and
 *   - coupled root/plane and firstness-certificate records.
 *
 * A general ecological mask is represented by a separately compiled direct
 * first-owner table.  GCC1 refuses to label an ordinary distance field or an
 * uncertified sampled direction table as rho_M.  The v1 producer therefore
 * publishes only analytically complete rectangular components; an irregular
 * mask is a cook failure until its direct first-owner pages exist.
 */

export const GROUND_COVER_CARRIER_MAGIC = 'GCC1';
export const GROUND_COVER_CARRIER_VERSION = 1;
export const GROUND_COVER_CARRIER_HEADER_BYTES = 256;
export const GROUND_COVER_CARRIER_ALIGNMENT = 64;

export const CALAMAGROSTIS_CARRIER_CLOSURE_URL = new URL(
  '../../assets/groundcover/calamagrostis-canescens.gcc1',
  import.meta.url,
);

export const GroundCoverCarrierFlags = {
  FixedWorldCharts: 1 << 0,
  GuardedCommonSlab: 1 << 1,
  ExactRectFirstEntry: 1 << 2,
  CoupledWinnerPlane: 1 << 3,
  FirstEntryClosure: 1 << 4,
} as const;

export const GroundCoverChartFlags = {
  Regular: 1 << 0,
  Transition: 1 << 1,
  WorldUp: 1 << 2,
} as const;

export const GroundCoverPatchFlags = {
  ExactRectangle: 1 << 0,
  HalfOpenMaxEdges: 1 << 1,
  EntryConditionedTransfer: 1 << 2,
  ClosureCertified: 1 << 3,
  /** P = R^2. This is the exact periodic diagnostic carrier, not a finite patch. */
  FullPeriodicPlane: 1 << 4,
} as const;

export const GroundCoverCertificateFlags = {
  TerrainBounded: 1 << 0,
  IncidencePositive: 1 << 1,
  OrderPositive: 1 << 2,
  BarycentricPositive: 1 << 3,
  HitBeforePatchExit: 1 << 4,
  NoLaterPatchBeforeHorizon: 1 << 5,
} as const;

export const GroundCoverBoundaryFace = {
  MinX: 0,
  MaxX: 1,
  MinHeight: 2,
  MaxHeight: 3,
  MinZ: 4,
  MaxZ: 5,
} as const;

export type GroundCoverBoundaryFaceValue =
  (typeof GroundCoverBoundaryFace)[keyof typeof GroundCoverBoundaryFace];

export interface GroundCoverFixedChart {
  readonly chartId: number;
  readonly flags: number;
  /** World X/Z anchor. Stored as float64 on wire; never camera-relative. */
  readonly originX: number;
  readonly originZ: number;
  readonly groundHeight: number;
  readonly slopeX: number;
  readonly slopeZ: number;
  /** Assigned world-horizontal chart domain, relative to originX/originZ. */
  readonly minX: number;
  readonly minZ: number;
  readonly maxX: number;
  readonly maxZ: number;
  readonly erodedMargin: number;
  readonly terrainRemainder: number;
  readonly interactionReach: number;
  readonly firstPatch: number;
  readonly patchCount: number;
  readonly transitionRecord: number;
}

export interface GroundCoverCarrierPatch {
  readonly patchId: number;
  readonly chartId: number;
  readonly communityState: number;
  readonly flags: number;
  /** Guarded footprint P in chart-local horizontal coordinates. */
  readonly minX: number;
  readonly minZ: number;
  readonly maxX: number;
  readonly maxZ: number;
  /** Common canonical height interval I. */
  readonly minHeight: number;
  readonly maxHeight: number;
  readonly entryTransferClass: number;
  readonly closureCertificate: number;
  readonly winnerPayloadFirst: number;
  readonly winnerPayloadCount: number;
}

/** Root prototypes in one periodic community tile. The live copy offset is
 * categorical winner data and is applied only after winner selection. */
export interface GroundCoverCommunityRoot {
  readonly rootId: number;
  readonly profileId: number;
  readonly localX: number;
  readonly localZ: number;
  readonly sourceGroundHeight: number;
  readonly maximumHeight: number;
  readonly stiffness: number;
  readonly flags: number;
}

export interface GroundCoverWinnerPlane {
  readonly planeId: number;
  readonly sourceTriangleId: number;
  readonly rootId: number;
  readonly materialSpecies: number;
  readonly periodicCopyX: number;
  readonly periodicCopyZ: number;
  readonly chartId: number;
  readonly certificateId: number;
  /** Source root relative to the fixed chart anchor. */
  readonly rootX: number;
  readonly rootZ: number;
  /** Source-space plane n dot x = c. */
  readonly normalX: number;
  readonly normalY: number;
  readonly normalZ: number;
  readonly planeConstant: number;
  readonly interactionReach: number;
  readonly incidenceMargin: number;
  readonly orderMargin: number;
  readonly barycentricMargin: number;
}

export interface GroundCoverFirstnessCertificate {
  readonly certificateId: number;
  readonly flags: number;
  readonly interactionReach: number;
  readonly terrainRemainder: number;
  readonly deformationError: number;
  readonly incidenceMargin: number;
  readonly orderMargin: number;
  readonly barycentricMargin: number;
  readonly hitBeforeExitMargin: number;
  readonly noLaterPatchHorizon: number;
}

export interface LoadedGroundCoverCarrierClosure {
  readonly profileId: number;
  readonly flags: number;
  readonly horizonMetres: number;
  readonly sourceSha256: string;
  readonly recipeSha256: string;
  readonly charts: readonly GroundCoverFixedChart[];
  readonly patches: readonly GroundCoverCarrierPatch[];
  readonly roots: readonly GroundCoverCommunityRoot[];
  readonly winnerPlanes: readonly GroundCoverWinnerPlane[];
  readonly certificates: readonly GroundCoverFirstnessCertificate[];
  readonly containerBytes: number;
}


const CHART_BYTES = 96;
const PATCH_BYTES = 64;
const PLANE_BYTES = 96;
const CERTIFICATE_BYTES = 64;
const ROOT_BYTES = 32;
const REQUIRED_FLAGS =
  GroundCoverCarrierFlags.FixedWorldCharts
  | GroundCoverCarrierFlags.GuardedCommonSlab
  | GroundCoverCarrierFlags.ExactRectFirstEntry
  | GroundCoverCarrierFlags.FirstEntryClosure;

function ascii(bytes: Uint8Array, offset: number, count: number): string {
  return String.fromCharCode(...bytes.subarray(offset, offset + count));
}

function hex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

function finite(view: DataView, offset: number, label: string): number {
  const value = view.getFloat32(offset, true);
  if (!Number.isFinite(value)) throw new Error(`GCC1 ${label} must be finite`);
  return value;
}

function finite64(view: DataView, offset: number, label: string): number {
  const value = view.getFloat64(offset, true);
  if (!Number.isFinite(value)) throw new Error(`GCC1 ${label} must be finite`);
  return value;
}

function positiveFinite(view: DataView, offset: number, label: string): number {
  const value = finite(view, offset, label);
  if (!(value > 0)) throw new Error(`GCC1 ${label} must be positive`);
  return value;
}

function section(
  bytes: Uint8Array,
  offset: number,
  count: number,
  stride: number,
  label: string,
): readonly [number, number] {
  const length = count * stride;
  if (
    !Number.isSafeInteger(length)
    || offset < GROUND_COVER_CARRIER_HEADER_BYTES
    || offset % GROUND_COVER_CARRIER_ALIGNMENT !== 0
    || length <= 0
    || offset + length > bytes.byteLength
  ) throw new Error(`GCC1 ${label} section is invalid`);
  return [offset, offset + length];
}

function requireZero(bytes: Uint8Array, begin: number, end: number, label: string): void {
  for (let index = begin; index < end; index++) {
    if (bytes[index] !== 0) throw new Error(`GCC1 ${label} reserved bytes must be zero`);
  }
}

/** Parse a complete, fail-closed carrier closure. */
export function parseGroundCoverCarrierClosure(bytes: Uint8Array): LoadedGroundCoverCarrierClosure {
  if (
    bytes.byteLength < GROUND_COVER_CARRIER_HEADER_BYTES
    || ascii(bytes, 0, 4) !== GROUND_COVER_CARRIER_MAGIC
  ) throw new Error('ground-cover carrier is not a GCC1 container');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(4, true) !== GROUND_COVER_CARRIER_VERSION) {
    throw new Error('ground-cover runtime requires GCC1/v1');
  }
  if (view.getUint32(8, true) !== GROUND_COVER_CARRIER_HEADER_BYTES) {
    throw new Error('GCC1 header size is not canonical');
  }
  if (view.getUint32(12, true) !== bytes.byteLength) {
    throw new Error('GCC1 container byte count does not match the file');
  }
  const flags = view.getUint32(16, true);
  if ((flags & REQUIRED_FLAGS) !== REQUIRED_FLAGS) {
    throw new Error('GCC1 is missing a required closure flag');
  }
  const profileId = view.getUint32(20, true);
  const horizonMetres = positiveFinite(view, 24, 'horizon');
  const chartCount = view.getUint32(28, true);
  const patchCount = view.getUint32(32, true);
  const planeCount = view.getUint32(36, true);
  const certificateCount = view.getUint32(40, true);
  const rootCount = view.getUint32(128, true);
  if (chartCount === 0 || patchCount === 0 || certificateCount === 0) {
    throw new Error('GCC1 requires non-empty chart, patch, and certificate sections');
  }
  const chartRange = section(bytes, view.getUint32(44, true), chartCount, CHART_BYTES, 'chart');
  const patchRange = section(bytes, view.getUint32(48, true), patchCount, PATCH_BYTES, 'patch');
  const planeRange = planeCount > 0
    ? section(bytes, view.getUint32(52, true), planeCount, PLANE_BYTES, 'plane')
    : [view.getUint32(52, true), view.getUint32(52, true)] as const;
  const certificateRange = section(
    bytes,
    view.getUint32(56, true),
    certificateCount,
    CERTIFICATE_BYTES,
    'certificate',
  );
  const rootRange = rootCount > 0
    ? section(bytes, view.getUint32(132, true), rootCount, ROOT_BYTES, 'root')
    : [view.getUint32(132, true), view.getUint32(132, true)] as const;
  const ranges = [
    chartRange,
    patchRange,
    certificateRange,
    ...(planeCount > 0 ? [planeRange] : []),
    ...(rootCount > 0 ? [rootRange] : []),
  ]
    .sort((a, b) => a[0] - b[0]);
  let previous = GROUND_COVER_CARRIER_HEADER_BYTES;
  for (const [begin, end] of ranges) {
    if (begin < previous) throw new Error('GCC1 sections overlap');
    previous = end;
  }
  requireZero(bytes, 136, GROUND_COVER_CARRIER_HEADER_BYTES, 'header');

  const charts: GroundCoverFixedChart[] = [];
  for (let index = 0; index < chartCount; index++) {
    const base = chartRange[0] + index * CHART_BYTES;
    const chart: GroundCoverFixedChart = Object.freeze({
      chartId: view.getUint32(base, true),
      flags: view.getUint32(base + 4, true),
      originX: finite64(view, base + 8, `chart ${index} originX`),
      originZ: finite64(view, base + 16, `chart ${index} originZ`),
      groundHeight: finite(view, base + 24, `chart ${index} groundHeight`),
      slopeX: finite(view, base + 28, `chart ${index} slopeX`),
      slopeZ: finite(view, base + 32, `chart ${index} slopeZ`),
      minX: finite(view, base + 36, `chart ${index} minX`),
      minZ: finite(view, base + 40, `chart ${index} minZ`),
      maxX: finite(view, base + 44, `chart ${index} maxX`),
      maxZ: finite(view, base + 48, `chart ${index} maxZ`),
      erodedMargin: finite(view, base + 52, `chart ${index} erodedMargin`),
      terrainRemainder: finite(view, base + 56, `chart ${index} terrainRemainder`),
      interactionReach: finite(view, base + 60, `chart ${index} interactionReach`),
      firstPatch: view.getUint32(base + 64, true),
      patchCount: view.getUint32(base + 68, true),
      transitionRecord: view.getUint32(base + 72, true),
    });
    if (chart.chartId !== index) throw new Error('GCC1 chart ids must be dense and ordered');
    if ((chart.flags & GroundCoverChartFlags.Regular) === 0) {
      throw new Error(`GCC1 chart ${index} is not a certified regular chart`);
    }
    if (!(chart.minX < chart.maxX && chart.minZ < chart.maxZ)) {
      throw new Error(`GCC1 chart ${index} domain is empty`);
    }
    if (chart.firstPatch + chart.patchCount > patchCount) {
      throw new Error(`GCC1 chart ${index} patch range is invalid`);
    }
    requireZero(bytes, base + 76, base + CHART_BYTES, `chart ${index}`);
    charts.push(chart);
  }

  const patches: GroundCoverCarrierPatch[] = [];
  for (let index = 0; index < patchCount; index++) {
    const base = patchRange[0] + index * PATCH_BYTES;
    const patch: GroundCoverCarrierPatch = Object.freeze({
      patchId: view.getUint32(base, true),
      chartId: view.getUint32(base + 4, true),
      communityState: view.getUint32(base + 8, true),
      flags: view.getUint32(base + 12, true),
      minX: finite(view, base + 16, `patch ${index} minX`),
      minZ: finite(view, base + 20, `patch ${index} minZ`),
      maxX: finite(view, base + 24, `patch ${index} maxX`),
      maxZ: finite(view, base + 28, `patch ${index} maxZ`),
      minHeight: finite(view, base + 32, `patch ${index} minHeight`),
      maxHeight: finite(view, base + 36, `patch ${index} maxHeight`),
      entryTransferClass: view.getUint32(base + 40, true),
      closureCertificate: view.getUint32(base + 44, true),
      winnerPayloadFirst: view.getUint32(base + 48, true),
      winnerPayloadCount: view.getUint32(base + 52, true),
    });
    if (patch.patchId !== index) throw new Error('GCC1 patch ids must be dense and ordered');
    if (patch.chartId >= chartCount) throw new Error(`GCC1 patch ${index} chart is invalid`);
    const analyticFootprint = patch.flags & (
      GroundCoverPatchFlags.ExactRectangle | GroundCoverPatchFlags.FullPeriodicPlane
    );
    if (analyticFootprint === 0 || analyticFootprint === (
      GroundCoverPatchFlags.ExactRectangle | GroundCoverPatchFlags.FullPeriodicPlane
    )) {
      throw new Error(`GCC1 patch ${index} does not name one exact analytic footprint`);
    }
    if ((patch.flags & GroundCoverPatchFlags.ClosureCertified) === 0) {
      throw new Error(`GCC1 patch ${index} lacks first-entry closure`);
    }
    if (
      !(patch.minHeight < patch.maxHeight)
      || (
        (patch.flags & GroundCoverPatchFlags.ExactRectangle) !== 0
        && !(patch.minX < patch.maxX && patch.minZ < patch.maxZ)
      )
    ) {
      throw new Error(`GCC1 patch ${index} carrier is empty`);
    }
    if (patch.closureCertificate >= certificateCount) {
      throw new Error(`GCC1 patch ${index} certificate is invalid`);
    }
    if (patch.winnerPayloadFirst + patch.winnerPayloadCount > planeCount) {
      throw new Error(`GCC1 patch ${index} winner payload range is invalid`);
    }
    requireZero(bytes, base + 56, base + PATCH_BYTES, `patch ${index}`);
    patches.push(patch);
  }

  const roots: GroundCoverCommunityRoot[] = [];
  for (let index = 0; index < rootCount; index++) {
    const base = rootRange[0] + index * ROOT_BYTES;
    const root: GroundCoverCommunityRoot = Object.freeze({
      rootId: view.getUint32(base, true),
      profileId: view.getUint32(base + 4, true),
      localX: finite(view, base + 8, `root ${index} localX`),
      localZ: finite(view, base + 12, `root ${index} localZ`),
      sourceGroundHeight: finite(view, base + 16, `root ${index} sourceGroundHeight`),
      maximumHeight: finite(view, base + 20, `root ${index} maximumHeight`),
      stiffness: finite(view, base + 24, `root ${index} stiffness`),
      flags: view.getUint32(base + 28, true),
    });
    if (root.rootId !== index) throw new Error('GCC1 root ids must be dense and ordered');
    if (root.profileId !== profileId) throw new Error(`GCC1 root ${index} profile mismatch`);
    if (!(root.maximumHeight > 0) || root.stiffness < 0) {
      throw new Error(`GCC1 root ${index} physical parameters are invalid`);
    }
    roots.push(root);
  }

  const winnerPlanes: GroundCoverWinnerPlane[] = [];
  for (let index = 0; index < planeCount; index++) {
    const base = planeRange[0] + index * PLANE_BYTES;
    const record: GroundCoverWinnerPlane = Object.freeze({
      planeId: view.getUint32(base, true),
      sourceTriangleId: view.getUint32(base + 4, true),
      rootId: view.getUint32(base + 8, true),
      materialSpecies: view.getUint32(base + 12, true),
      periodicCopyX: view.getInt32(base + 16, true),
      periodicCopyZ: view.getInt32(base + 20, true),
      chartId: view.getUint32(base + 24, true),
      certificateId: view.getUint32(base + 28, true),
      rootX: finite(view, base + 32, `plane ${index} rootX`),
      rootZ: finite(view, base + 36, `plane ${index} rootZ`),
      normalX: finite(view, base + 40, `plane ${index} normalX`),
      normalY: finite(view, base + 44, `plane ${index} normalY`),
      normalZ: finite(view, base + 48, `plane ${index} normalZ`),
      planeConstant: finite(view, base + 52, `plane ${index} constant`),
      interactionReach: finite(view, base + 56, `plane ${index} reach`),
      incidenceMargin: finite(view, base + 60, `plane ${index} incidence`),
      orderMargin: finite(view, base + 64, `plane ${index} order`),
      barycentricMargin: finite(view, base + 68, `plane ${index} barycentric`),
    });
    if (record.planeId !== index) throw new Error('GCC1 plane ids must be dense and ordered');
    if (record.chartId >= chartCount || record.certificateId >= certificateCount) {
      throw new Error(`GCC1 plane ${index} references an invalid record`);
    }
    const normalLength = Math.hypot(record.normalX, record.normalY, record.normalZ);
    if (!(normalLength > 1e-8)) throw new Error(`GCC1 plane ${index} has a zero normal`);
    requireZero(bytes, base + 72, base + PLANE_BYTES, `plane ${index}`);
    winnerPlanes.push(record);
  }

  const certificates: GroundCoverFirstnessCertificate[] = [];
  for (let index = 0; index < certificateCount; index++) {
    const base = certificateRange[0] + index * CERTIFICATE_BYTES;
    const record: GroundCoverFirstnessCertificate = Object.freeze({
      certificateId: view.getUint32(base, true),
      flags: view.getUint32(base + 4, true),
      interactionReach: finite(view, base + 8, `certificate ${index} reach`),
      terrainRemainder: finite(view, base + 12, `certificate ${index} terrain`),
      deformationError: finite(view, base + 16, `certificate ${index} deformation`),
      incidenceMargin: finite(view, base + 20, `certificate ${index} incidence`),
      orderMargin: finite(view, base + 24, `certificate ${index} order`),
      barycentricMargin: finite(view, base + 28, `certificate ${index} barycentric`),
      hitBeforeExitMargin: finite(view, base + 32, `certificate ${index} exit margin`),
      noLaterPatchHorizon: finite(view, base + 36, `certificate ${index} later horizon`),
    });
    if (record.certificateId !== index) {
      throw new Error('GCC1 certificate ids must be dense and ordered');
    }
    if ((record.flags & GroundCoverCertificateFlags.TerrainBounded) === 0) {
      throw new Error(`GCC1 certificate ${index} does not bound terrain`);
    }
    if (
      (record.flags & GroundCoverCertificateFlags.HitBeforePatchExit) === 0
      && (record.flags & GroundCoverCertificateFlags.NoLaterPatchBeforeHorizon) === 0
    ) throw new Error(`GCC1 certificate ${index} does not close first entry`);
    requireZero(bytes, base + 40, base + CERTIFICATE_BYTES, `certificate ${index}`);
    certificates.push(record);
  }

  for (const patch of patches) {
    if (charts[patch.chartId]!.firstPatch > patch.patchId) {
      throw new Error(`GCC1 patch ${patch.patchId} precedes its chart range`);
    }
    const certificate = certificates[patch.closureCertificate]!;
    if ((certificate.flags & GroundCoverCertificateFlags.HitBeforePatchExit) === 0) {
      if (certificate.noLaterPatchHorizon + 1e-5 < horizonMetres) {
        throw new Error(`GCC1 patch ${patch.patchId} closure horizon is incomplete`);
      }
    }
  }

  return Object.freeze({
    profileId,
    flags,
    horizonMetres,
    sourceSha256: hex(bytes.subarray(64, 96)),
    recipeSha256: hex(bytes.subarray(96, 128)),
    charts: Object.freeze(charts),
    patches: Object.freeze(patches),
    roots: Object.freeze(roots),
    winnerPlanes: Object.freeze(winnerPlanes),
    certificates: Object.freeze(certificates),
    containerBytes: bytes.byteLength,
  });
}

export async function loadGroundCoverCarrierClosure(
  source: URL = CALAMAGROSTIS_CARRIER_CLOSURE_URL,
): Promise<LoadedGroundCoverCarrierClosure> {
  const response = await fetch(source);
  if (!response.ok) throw new Error(`ground-cover carrier load failed (${response.status})`);
  return parseGroundCoverCarrierClosure(new Uint8Array(await response.arrayBuffer()));
}
