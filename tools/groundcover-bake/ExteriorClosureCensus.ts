/**
 * Shader-independent Stage-1 feasibility census for a predicate-filtered
 * periodic exterior owner closure.
 *
 * The census deliberately measures a lower bound, not a continuous-domain
 * certificate. For each deterministic exterior ray it finds the nearest
 * analytic triangle hit in every periodic tile copy. If tile-copy roots may be
 * independently enabled, every one of those root-first owners is an
 * unavoidable closure candidate for that ray.
 */

export type CensusVec3 = readonly [number, number, number];

export interface CensusBounds {
  min: CensusVec3;
  max: CensusVec3;
}

export interface DecodedOwnedProfileGeometry {
  version: 4;
  profileId: number;
  topH: number;
  tileOriginX: number;
  tileOriginZ: number;
  tileSizeX: number;
  tileSizeZ: number;
  bounds: CensusBounds;
  positions: Float64Array;
  triangles: Uint32Array;
  vertexCount: number;
  triangleCount: number;
}

export interface CensusRay {
  phaseX: number;
  phaseZ: number;
  azimuthDegrees: number;
  elevationDegrees: number;
  direction: CensusVec3;
}

export interface PeriodicRootHit {
  triangleId: number;
  copyX: number;
  copyZ: number;
  t: number;
}

export interface TriangleBvhMetrics {
  leafSize: number;
  leafCount: number;
  nodeCount: number;
  buildMilliseconds: number;
}

export interface ExteriorCensusOptions {
  phaseGrid: number;
  phaseOffsets?: readonly number[];
  azimuthCount: number;
  elevationsDegrees: readonly number[];
  worstRayLimit?: number;
}

export interface ExteriorCensusWorstRay {
  phase: readonly [number, number];
  phaseNormalized: readonly [number, number];
  azimuthDegrees: number;
  elevationDegrees: number;
  direction: CensusVec3;
  rootHitCount: number;
  owners: readonly PeriodicRootHit[];
}

export interface ExteriorClosureCensusReport {
  domain: {
    phaseGrid: number;
    phaseOffsets: readonly number[];
    azimuthCount: number;
    elevationsDegrees: readonly number[];
    minimumElevationDegrees: number;
    topH: number;
    minimumY: number;
    verticalDropHorizon: number;
    maximumRayDistance: number;
    maximumHorizontalReach: number;
  };
  accelerator: TriangleBvhMetrics;
  samples: {
    traceMilliseconds: number;
    rayCount: number;
    rootHitReferences: number;
    distinctOwnerKeys: number;
    distinctTriangles: number;
    distinctCopies: number;
    histogram: Record<string, number>;
    byElevation: Record<string, {
      rayCount: number;
      rootHitReferences: number;
      maximum: number;
      histogram: Record<string, number>;
    }>;
    p50: number;
    p95: number;
    p99: number;
    maximum: number;
    missOnlyRays: number;
    copyRange: {
      minX: number;
      maxX: number;
      minZ: number;
      maxZ: number;
      maximumAbsolute: number;
    } | null;
    worstRays: readonly ExteriorCensusWorstRay[];
  };
  interpretation: readonly string[];
}

const GCRP_HEADER_BYTES_V4 = 128;
const GCRP_SLICE_BYTES = 64;
const GCRP_GEOMETRY_TEXEL_BYTES = 8;
const GCRP_OWNER_TEXEL_BYTES = 4;
const GCRP_VERTEX_BYTES = 16;
const GCRP_TRIANGLE_BYTES = 16;
const PARALLEL_EPSILON = 1e-14;
const BARYCENTRIC_EPSILON = 1e-12;
const BOUNDS_EPSILON = 1e-12;

function positiveInteger(value: number, label: string): number {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer`);
  return value;
}

function finiteFloat(view: DataView, offset: number, label: string): number {
  const value = view.getFloat32(offset, true);
  if (!Number.isFinite(value)) throw new Error(`${label} is not finite`);
  return value;
}

/** Decode only the geometry needed by the census. The large v4 geometry and
 * owner atlases are validated by offset but intentionally not copied. */
export function decodeOwnedProfileGeometry(bytes: Uint8Array): DecodedOwnedProfileGeometry {
  if (
    bytes.byteLength < GCRP_HEADER_BYTES_V4
    || bytes[0] !== 0x47
    || bytes[1] !== 0x43
    || bytes[2] !== 0x52
    || bytes[3] !== 0x50
  ) throw new Error('exterior closure census requires a GCRP container');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(4, true);
  if (version !== 4) throw new Error(`exterior closure census requires GCRP/v4, got v${version}`);
  const profileId = view.getUint32(8, true);
  const storedTileWidth = positiveInteger(view.getUint32(12, true), 'stored tile width');
  const storedTileHeight = positiveInteger(view.getUint32(16, true), 'stored tile height');
  const atlasColumns = positiveInteger(view.getUint32(20, true), 'atlas columns');
  const atlasRows = positiveInteger(view.getUint32(24, true), 'atlas rows');
  const sliceCount = positiveInteger(view.getUint32(28, true), 'slice count');
  if (sliceCount > atlasColumns * atlasRows) throw new Error('GCRP/v4 slices do not fit the atlas');
  if (view.getUint32(32, true) !== GCRP_GEOMETRY_TEXEL_BYTES) {
    throw new Error('GCRP/v4 geometry texel size is not canonical');
  }
  const geometryOffset = view.getUint32(36, true);
  if (view.getUint32(40, true) !== 1) throw new Error('GCRP/v4 projection is not periodic top-plane XZ');
  const interiorTileWidth = positiveInteger(view.getUint32(44, true), 'interior tile width');
  const interiorTileHeight = positiveInteger(view.getUint32(48, true), 'interior tile height');
  const gutter = positiveInteger(view.getUint32(52, true), 'gutter');
  if (
    storedTileWidth !== interiorTileWidth + gutter * 2
    || storedTileHeight !== interiorTileHeight + gutter * 2
  ) throw new Error('GCRP/v4 stored dimensions do not match the wrapped gutter');
  const topH = finiteFloat(view, 56, 'top height');
  const tileOriginX = finiteFloat(view, 60, 'tile origin X');
  const tileOriginZ = finiteFloat(view, 64, 'tile origin Z');
  const tileSizeX = finiteFloat(view, 68, 'tile size X');
  const tileSizeZ = finiteFloat(view, 72, 'tile size Z');
  if (!(topH > 0 && tileSizeX > 0 && tileSizeZ > 0)) {
    throw new Error('GCRP/v4 top height and tile dimensions must be positive');
  }
  if (view.getUint32(76, true) !== GCRP_HEADER_BYTES_V4) {
    throw new Error('GCRP/v4 header size is not canonical');
  }
  const ownerOffset = view.getUint32(80, true);
  const vertexOffset = view.getUint32(84, true);
  const triangleOffset = view.getUint32(88, true);
  const vertexCount = positiveInteger(view.getUint32(92, true), 'vertex count');
  const triangleCount = positiveInteger(view.getUint32(96, true), 'triangle count');
  const bounds: CensusBounds = {
    min: [
      finiteFloat(view, 100, 'source minimum X'),
      finiteFloat(view, 104, 'source minimum Y'),
      finiteFloat(view, 108, 'source minimum Z'),
    ],
    max: [
      finiteFloat(view, 112, 'source maximum X'),
      finiteFloat(view, 116, 'source maximum Y'),
      finiteFloat(view, 120, 'source maximum Z'),
    ],
  };
  if (
    !(bounds.max[0] > bounds.min[0])
    || !(bounds.max[1] > bounds.min[1])
    || !(bounds.max[2] > bounds.min[2])
    || !(topH > bounds.max[1])
  ) throw new Error('GCRP/v4 source bounds or top height are invalid');
  const atlasTexels = storedTileWidth * storedTileHeight * atlasColumns * atlasRows;
  const expectedGeometryOffset = GCRP_HEADER_BYTES_V4 + sliceCount * GCRP_SLICE_BYTES;
  const expectedOwnerOffset = expectedGeometryOffset + atlasTexels * GCRP_GEOMETRY_TEXEL_BYTES;
  const expectedVertexOffset = expectedOwnerOffset + atlasTexels * GCRP_OWNER_TEXEL_BYTES;
  const expectedTriangleOffset = expectedVertexOffset + vertexCount * GCRP_VERTEX_BYTES;
  const expectedEnd = expectedTriangleOffset + triangleCount * GCRP_TRIANGLE_BYTES;
  if (
    geometryOffset !== expectedGeometryOffset
    || ownerOffset !== expectedOwnerOffset
    || vertexOffset !== expectedVertexOffset
    || triangleOffset !== expectedTriangleOffset
    || expectedEnd !== bytes.byteLength
  ) throw new Error('GCRP/v4 section offsets or length are inconsistent');

  const positions = new Float64Array(vertexCount * 3);
  const spanX = bounds.max[0] - bounds.min[0];
  const spanY = bounds.max[1] - bounds.min[1];
  const spanZ = bounds.max[2] - bounds.min[2];
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const source = vertexOffset + vertex * GCRP_VERTEX_BYTES;
    const target = vertex * 3;
    positions[target] = bounds.min[0] + view.getUint16(source, true) / 65535 * spanX;
    positions[target + 1] = bounds.min[1] + view.getUint16(source + 2, true) / 65535 * spanY;
    positions[target + 2] = bounds.min[2] + view.getUint16(source + 4, true) / 65535 * spanZ;
  }
  const triangles = new Uint32Array(triangleCount * 3);
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const source = triangleOffset + triangle * GCRP_TRIANGLE_BYTES;
    const target = triangle * 3;
    for (let corner = 0; corner < 3; corner++) {
      const vertex = view.getUint32(source + corner * 4, true);
      if (vertex >= vertexCount) throw new Error(`triangle ${triangle} references vertex ${vertex} outside ${vertexCount}`);
      triangles[target + corner] = vertex;
    }
  }
  return {
    version: 4,
    profileId,
    topH,
    tileOriginX,
    tileOriginZ,
    tileSizeX,
    tileSizeZ,
    bounds,
    positions,
    triangles,
    vertexCount,
    triangleCount,
  };
}

function expandMortonBits(value: number): number {
  let result = value & 0x3ff;
  result = (result | (result << 16)) & 0x030000ff;
  result = (result | (result << 8)) & 0x0300f00f;
  result = (result | (result << 4)) & 0x030c30c3;
  result = (result | (result << 2)) & 0x09249249;
  return result >>> 0;
}

function morton3(x: number, y: number, z: number): number {
  return (expandMortonBits(x) | (expandMortonBits(y) << 1) | (expandMortonBits(z) << 2)) >>> 0;
}

function radixSortMorton(keys: Uint32Array, ids: Uint32Array): Uint32Array {
  const scratchKeys = new Uint32Array(keys.length);
  const scratchIds = new Uint32Array(ids.length);
  const counts = new Uint32Array(1024);
  let sourceKeys: Uint32Array<ArrayBufferLike> = keys;
  let sourceIds: Uint32Array<ArrayBufferLike> = ids;
  let targetKeys: Uint32Array<ArrayBufferLike> = scratchKeys;
  let targetIds: Uint32Array<ArrayBufferLike> = scratchIds;
  for (const shift of [0, 10, 20]) {
    counts.fill(0);
    for (let i = 0; i < sourceKeys.length; i++) counts[(sourceKeys[i]! >>> shift) & 1023]++;
    let sum = 0;
    for (let bucket = 0; bucket < counts.length; bucket++) {
      const count = counts[bucket]!;
      counts[bucket] = sum;
      sum += count;
    }
    for (let i = 0; i < sourceKeys.length; i++) {
      const key = sourceKeys[i]!;
      const bucket = (key >>> shift) & 1023;
      const target = counts[bucket]!;
      counts[bucket] = target + 1;
      targetKeys[target] = key;
      targetIds[target] = sourceIds[i]!;
    }
    [sourceKeys, targetKeys] = [targetKeys, sourceKeys];
    [sourceIds, targetIds] = [targetIds, sourceIds];
  }
  return sourceIds;
}

export class TriangleBvh {
  readonly metrics: TriangleBvhMetrics;
  private readonly orderedTriangles: Uint32Array;
  private readonly left: Int32Array;
  private readonly right: Int32Array;
  private readonly first: Uint32Array;
  private readonly count: Uint32Array;
  private readonly minX: Float64Array;
  private readonly minY: Float64Array;
  private readonly minZ: Float64Array;
  private readonly maxX: Float64Array;
  private readonly maxY: Float64Array;
  private readonly maxZ: Float64Array;
  private readonly traversalStack = new Int32Array(64);

  private constructor(
    private readonly geometry: DecodedOwnedProfileGeometry,
    orderedTriangles: Uint32Array,
    leafSize: number,
    buildStarted: number,
  ) {
    this.orderedTriangles = orderedTriangles;
    const leafCount = Math.ceil(geometry.triangleCount / leafSize);
    const nodeCapacity = leafCount * 2 - 1;
    this.left = new Int32Array(nodeCapacity);
    this.right = new Int32Array(nodeCapacity);
    this.left.fill(-1);
    this.right.fill(-1);
    this.first = new Uint32Array(nodeCapacity);
    this.count = new Uint32Array(nodeCapacity);
    this.minX = new Float64Array(nodeCapacity);
    this.minY = new Float64Array(nodeCapacity);
    this.minZ = new Float64Array(nodeCapacity);
    this.maxX = new Float64Array(nodeCapacity);
    this.maxY = new Float64Array(nodeCapacity);
    this.maxZ = new Float64Array(nodeCapacity);
    let cursor = 0;
    const build = (firstLeaf: number, lastLeaf: number): number => {
      const node = cursor++;
      if (lastLeaf - firstLeaf === 1) {
        const firstTriangle = firstLeaf * leafSize;
        const triangleCount = Math.min(leafSize, geometry.triangleCount - firstTriangle);
        this.first[node] = firstTriangle;
        this.count[node] = triangleCount;
        let x0 = Number.POSITIVE_INFINITY;
        let y0 = Number.POSITIVE_INFINITY;
        let z0 = Number.POSITIVE_INFINITY;
        let x1 = Number.NEGATIVE_INFINITY;
        let y1 = Number.NEGATIVE_INFINITY;
        let z1 = Number.NEGATIVE_INFINITY;
        for (let ordered = firstTriangle; ordered < firstTriangle + triangleCount; ordered++) {
          const triangle = orderedTriangles[ordered]! * 3;
          for (let corner = 0; corner < 3; corner++) {
            const vertex = geometry.triangles[triangle + corner]! * 3;
            const x = geometry.positions[vertex]!;
            const y = geometry.positions[vertex + 1]!;
            const z = geometry.positions[vertex + 2]!;
            x0 = Math.min(x0, x);
            y0 = Math.min(y0, y);
            z0 = Math.min(z0, z);
            x1 = Math.max(x1, x);
            y1 = Math.max(y1, y);
            z1 = Math.max(z1, z);
          }
        }
        this.minX[node] = x0;
        this.minY[node] = y0;
        this.minZ[node] = z0;
        this.maxX[node] = x1;
        this.maxY[node] = y1;
        this.maxZ[node] = z1;
        return node;
      }
      const middle = Math.floor((firstLeaf + lastLeaf) / 2);
      const left = build(firstLeaf, middle);
      const right = build(middle, lastLeaf);
      this.left[node] = left;
      this.right[node] = right;
      this.minX[node] = Math.min(this.minX[left]!, this.minX[right]!);
      this.minY[node] = Math.min(this.minY[left]!, this.minY[right]!);
      this.minZ[node] = Math.min(this.minZ[left]!, this.minZ[right]!);
      this.maxX[node] = Math.max(this.maxX[left]!, this.maxX[right]!);
      this.maxY[node] = Math.max(this.maxY[left]!, this.maxY[right]!);
      this.maxZ[node] = Math.max(this.maxZ[left]!, this.maxZ[right]!);
      return node;
    };
    build(0, leafCount);
    this.metrics = {
      leafSize,
      leafCount,
      nodeCount: cursor,
      buildMilliseconds: performance.now() - buildStarted,
    };
  }

  static build(geometry: DecodedOwnedProfileGeometry, leafSize = 8): TriangleBvh {
    positiveInteger(leafSize, 'BVH leaf size');
    const started = performance.now();
    const keys = new Uint32Array(geometry.triangleCount);
    const ids = new Uint32Array(geometry.triangleCount);
    const spanX = Math.max(1e-30, geometry.bounds.max[0] - geometry.bounds.min[0]);
    const spanY = Math.max(1e-30, geometry.bounds.max[1] - geometry.bounds.min[1]);
    const spanZ = Math.max(1e-30, geometry.bounds.max[2] - geometry.bounds.min[2]);
    for (let triangleId = 0; triangleId < geometry.triangleCount; triangleId++) {
      const triangle = triangleId * 3;
      let cx = 0;
      let cy = 0;
      let cz = 0;
      for (let corner = 0; corner < 3; corner++) {
        const vertex = geometry.triangles[triangle + corner]! * 3;
        cx += geometry.positions[vertex]!;
        cy += geometry.positions[vertex + 1]!;
        cz += geometry.positions[vertex + 2]!;
      }
      const qx = Math.max(0, Math.min(1023, Math.floor(((cx / 3 - geometry.bounds.min[0]) / spanX) * 1023)));
      const qy = Math.max(0, Math.min(1023, Math.floor(((cy / 3 - geometry.bounds.min[1]) / spanY) * 1023)));
      const qz = Math.max(0, Math.min(1023, Math.floor(((cz / 3 - geometry.bounds.min[2]) / spanZ) * 1023)));
      keys[triangleId] = morton3(qx, qy, qz);
      ids[triangleId] = triangleId;
    }
    return new TriangleBvh(geometry, radixSortMorton(keys, ids), leafSize, started);
  }

  private boundsEntry(
    node: number,
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    maximumT: number,
  ): number {
    let near = 0;
    let far = maximumT;
    const slab = (origin: number, direction: number, minimum: number, maximum: number): boolean => {
      if (Math.abs(direction) <= 1e-30) return origin >= minimum - BOUNDS_EPSILON && origin <= maximum + BOUNDS_EPSILON;
      const inverse = 1 / direction;
      let a = (minimum - origin) * inverse;
      let b = (maximum - origin) * inverse;
      if (a > b) [a, b] = [b, a];
      near = Math.max(near, a - BOUNDS_EPSILON);
      far = Math.min(far, b + BOUNDS_EPSILON);
      return near <= far;
    };
    return slab(ox, dx, this.minX[node]!, this.maxX[node]!)
      && slab(oy, dy, this.minY[node]!, this.maxY[node]!)
      && slab(oz, dz, this.minZ[node]!, this.maxZ[node]!)
      ? near
      : Number.POSITIVE_INFINITY;
  }

  intersectNearest(origin: CensusVec3, direction: CensusVec3, maximumT: number): { triangleId: number; t: number } | null {
    let bestT = maximumT;
    let bestTriangle = -1;
    let stackSize = 0;
    this.traversalStack[stackSize++] = 0;
    while (stackSize > 0) {
      const node = this.traversalStack[--stackSize]!;
      if (this.boundsEntry(
        node,
        origin[0], origin[1], origin[2],
        direction[0], direction[1], direction[2],
        bestT,
      ) === Number.POSITIVE_INFINITY) continue;
      const triangleCount = this.count[node]!;
      if (triangleCount > 0) {
        const first = this.first[node]!;
        for (let ordered = first; ordered < first + triangleCount; ordered++) {
          const triangleId = this.orderedTriangles[ordered]!;
          const triangle = triangleId * 3;
          const ia = this.geometry.triangles[triangle]! * 3;
          const ib = this.geometry.triangles[triangle + 1]! * 3;
          const ic = this.geometry.triangles[triangle + 2]! * 3;
          const ax = this.geometry.positions[ia]!;
          const ay = this.geometry.positions[ia + 1]!;
          const az = this.geometry.positions[ia + 2]!;
          const e1x = this.geometry.positions[ib]! - ax;
          const e1y = this.geometry.positions[ib + 1]! - ay;
          const e1z = this.geometry.positions[ib + 2]! - az;
          const e2x = this.geometry.positions[ic]! - ax;
          const e2y = this.geometry.positions[ic + 1]! - ay;
          const e2z = this.geometry.positions[ic + 2]! - az;
          const px = direction[1] * e2z - direction[2] * e2y;
          const py = direction[2] * e2x - direction[0] * e2z;
          const pz = direction[0] * e2y - direction[1] * e2x;
          const determinant = e1x * px + e1y * py + e1z * pz;
          if (Math.abs(determinant) <= PARALLEL_EPSILON) continue;
          const inverse = 1 / determinant;
          const sx = origin[0] - ax;
          const sy = origin[1] - ay;
          const sz = origin[2] - az;
          const u = (sx * px + sy * py + sz * pz) * inverse;
          if (u < -BARYCENTRIC_EPSILON || u > 1 + BARYCENTRIC_EPSILON) continue;
          const qx = sy * e1z - sz * e1y;
          const qy = sz * e1x - sx * e1z;
          const qz = sx * e1y - sy * e1x;
          const v = (direction[0] * qx + direction[1] * qy + direction[2] * qz) * inverse;
          if (v < -BARYCENTRIC_EPSILON || u + v > 1 + BARYCENTRIC_EPSILON) continue;
          const t = (e2x * qx + e2y * qy + e2z * qz) * inverse;
          if (t < 0 || t > bestT + BOUNDS_EPSILON) continue;
          if (
            bestTriangle < 0
            || t < bestT
            || (Math.abs(t - bestT) <= BOUNDS_EPSILON && triangleId < bestTriangle)
          ) {
            bestT = t;
            bestTriangle = triangleId;
          }
        }
        continue;
      }
      const left = this.left[node]!;
      const right = this.right[node]!;
      const leftEntry = this.boundsEntry(
        left,
        origin[0], origin[1], origin[2],
        direction[0], direction[1], direction[2],
        bestT,
      );
      const rightEntry = this.boundsEntry(
        right,
        origin[0], origin[1], origin[2],
        direction[0], direction[1], direction[2],
        bestT,
      );
      if (leftEntry < rightEntry) {
        if (rightEntry !== Number.POSITIVE_INFINITY) this.traversalStack[stackSize++] = right;
        if (leftEntry !== Number.POSITIVE_INFINITY) this.traversalStack[stackSize++] = left;
      } else {
        if (leftEntry !== Number.POSITIVE_INFINITY) this.traversalStack[stackSize++] = left;
        if (rightEntry !== Number.POSITIVE_INFINITY) this.traversalStack[stackSize++] = right;
      }
    }
    return bestTriangle < 0 ? null : { triangleId: bestTriangle, t: bestT };
  }
}

function rayBoxEntry(
  origin: CensusVec3,
  direction: CensusVec3,
  bounds: CensusBounds,
  maximumT: number,
): number {
  let near = 0;
  let far = maximumT;
  for (let axis = 0; axis < 3; axis++) {
    const d = direction[axis]!;
    const o = origin[axis]!;
    if (Math.abs(d) <= 1e-30) {
      if (o < bounds.min[axis]! - BOUNDS_EPSILON || o > bounds.max[axis]! + BOUNDS_EPSILON) {
        return Number.POSITIVE_INFINITY;
      }
      continue;
    }
    const inverse = 1 / d;
    let a = (bounds.min[axis]! - o) * inverse;
    let b = (bounds.max[axis]! - o) * inverse;
    if (a > b) [a, b] = [b, a];
    near = Math.max(near, a - BOUNDS_EPSILON);
    far = Math.min(far, b + BOUNDS_EPSILON);
    if (near > far) return Number.POSITIVE_INFINITY;
  }
  return near;
}

/** Find one analytic first owner in every periodic tile-copy root touched by a
 * ray. Copy coordinates remain full JS integers; v4's signed five-bit packing
 * is intentionally not used. */
export function enumeratePeriodicRootHits(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  ray: CensusRay,
): PeriodicRootHit[] {
  const verticalDrop = geometry.topH - geometry.bounds.min[1];
  const maximumT = verticalDrop / -ray.direction[1];
  const endX = ray.phaseX + ray.direction[0] * maximumT;
  const endZ = ray.phaseZ + ray.direction[2] * maximumT;
  const rayMinX = Math.min(ray.phaseX, endX);
  const rayMaxX = Math.max(ray.phaseX, endX);
  const rayMinZ = Math.min(ray.phaseZ, endZ);
  const rayMaxZ = Math.max(ray.phaseZ, endZ);
  const minCopyX = Math.ceil((rayMinX - geometry.bounds.max[0]) / geometry.tileSizeX - BOUNDS_EPSILON);
  const maxCopyX = Math.floor((rayMaxX - geometry.bounds.min[0]) / geometry.tileSizeX + BOUNDS_EPSILON);
  const minCopyZ = Math.ceil((rayMinZ - geometry.bounds.max[2]) / geometry.tileSizeZ - BOUNDS_EPSILON);
  const maxCopyZ = Math.floor((rayMaxZ - geometry.bounds.min[2]) / geometry.tileSizeZ + BOUNDS_EPSILON);
  const hits: PeriodicRootHit[] = [];
  for (let copyZ = minCopyZ; copyZ <= maxCopyZ; copyZ++) {
    for (let copyX = minCopyX; copyX <= maxCopyX; copyX++) {
      const stableCopyX = Object.is(copyX, -0) ? 0 : copyX;
      const stableCopyZ = Object.is(copyZ, -0) ? 0 : copyZ;
      const offsetX = stableCopyX * geometry.tileSizeX;
      const offsetZ = stableCopyZ * geometry.tileSizeZ;
      const localOrigin: CensusVec3 = [
        ray.phaseX - offsetX,
        geometry.topH,
        ray.phaseZ - offsetZ,
      ];
      if (rayBoxEntry(localOrigin, ray.direction, geometry.bounds, maximumT) === Number.POSITIVE_INFINITY) continue;
      const hit = bvh.intersectNearest(localOrigin, ray.direction, maximumT);
      if (hit) hits.push({ triangleId: hit.triangleId, copyX: stableCopyX, copyZ: stableCopyZ, t: hit.t });
    }
  }
  return hits.sort((left, right) => left.t - right.t
    || left.copyX - right.copyX
    || left.copyZ - right.copyZ
    || left.triangleId - right.triangleId);
}

export function makeDeterministicExteriorRays(
  geometry: DecodedOwnedProfileGeometry,
  options: ExteriorCensusOptions,
): CensusRay[] {
  positiveInteger(options.phaseGrid, 'phase grid');
  positiveInteger(options.azimuthCount, 'azimuth count');
  if (options.elevationsDegrees.length === 0) throw new Error('at least one elevation is required');
  const phaseOffsets = options.phaseOffsets ?? [0.5];
  if (phaseOffsets.length === 0 || phaseOffsets.some((value) => !(value >= 0 && value < 1))) {
    throw new Error('phase offsets must be in [0,1)');
  }
  const rays: CensusRay[] = [];
  for (const elevationDegrees of options.elevationsDegrees) {
    if (!(elevationDegrees >= 5 && elevationDegrees <= 90)) {
      throw new Error(`elevation ${elevationDegrees} is outside the declared [5,90] degree domain`);
    }
    const elevation = elevationDegrees * Math.PI / 180;
    const azimuthSamples = Math.abs(elevationDegrees - 90) <= 1e-12 ? 1 : options.azimuthCount;
    for (let azimuthIndex = 0; azimuthIndex < azimuthSamples; azimuthIndex++) {
      const azimuthDegrees = azimuthIndex * 360 / azimuthSamples;
      const azimuth = azimuthDegrees * Math.PI / 180;
      const horizontal = Math.abs(elevationDegrees - 90) <= 1e-12 ? 0 : Math.cos(elevation);
      const direction: CensusVec3 = [
        horizontal * Math.cos(azimuth),
        -Math.sin(elevation),
        horizontal * Math.sin(azimuth),
      ];
      for (const offsetZ of phaseOffsets) {
        for (let phaseZ = 0; phaseZ < options.phaseGrid; phaseZ++) {
          for (const offsetX of phaseOffsets) {
            for (let phaseX = 0; phaseX < options.phaseGrid; phaseX++) {
              rays.push({
                phaseX: geometry.tileOriginX
                  + (phaseX + offsetX) / options.phaseGrid * geometry.tileSizeX,
                phaseZ: geometry.tileOriginZ
                  + (phaseZ + offsetZ) / options.phaseGrid * geometry.tileSizeZ,
                azimuthDegrees,
                elevationDegrees,
                direction,
              });
            }
          }
        }
      }
    }
  }
  return rays;
}

function histogramQuantile(histogram: Map<number, number>, total: number, fraction: number): number {
  const target = Math.ceil(total * fraction);
  let accumulated = 0;
  const counts = [...histogram.keys()].sort((a, b) => a - b);
  for (const count of counts) {
    accumulated += histogram.get(count)!;
    if (accumulated >= target) return count;
  }
  return counts.at(-1) ?? 0;
}

export function runExteriorClosureCensus(
  geometry: DecodedOwnedProfileGeometry,
  bvh: TriangleBvh,
  options: ExteriorCensusOptions,
): ExteriorClosureCensusReport {
  const traceStarted = performance.now();
  const rays = makeDeterministicExteriorRays(geometry, options);
  const histogram = new Map<number, number>();
  const elevationHistograms = new Map<number, Map<number, number>>();
  const ownerKeys = new Set<string>();
  const triangles = new Set<number>();
  const copies = new Set<string>();
  const worstRayLimit = options.worstRayLimit ?? 8;
  if (!Number.isInteger(worstRayLimit) || worstRayLimit < 0) throw new Error('worst-ray limit must be a non-negative integer');
  const worstRays: ExteriorCensusWorstRay[] = [];
  let rootHitReferences = 0;
  let missOnlyRays = 0;
  let minCopyX = Number.POSITIVE_INFINITY;
  let maxCopyX = Number.NEGATIVE_INFINITY;
  let minCopyZ = Number.POSITIVE_INFINITY;
  let maxCopyZ = Number.NEGATIVE_INFINITY;
  for (const ray of rays) {
    const hits = enumeratePeriodicRootHits(geometry, bvh, ray);
    histogram.set(hits.length, (histogram.get(hits.length) ?? 0) + 1);
    let elevationHistogram = elevationHistograms.get(ray.elevationDegrees);
    if (!elevationHistogram) {
      elevationHistogram = new Map<number, number>();
      elevationHistograms.set(ray.elevationDegrees, elevationHistogram);
    }
    elevationHistogram.set(hits.length, (elevationHistogram.get(hits.length) ?? 0) + 1);
    rootHitReferences += hits.length;
    if (hits.length === 0) missOnlyRays++;
    for (const hit of hits) {
      ownerKeys.add(`${hit.triangleId}:${hit.copyX}:${hit.copyZ}`);
      triangles.add(hit.triangleId);
      copies.add(`${hit.copyX}:${hit.copyZ}`);
      minCopyX = Math.min(minCopyX, hit.copyX);
      maxCopyX = Math.max(maxCopyX, hit.copyX);
      minCopyZ = Math.min(minCopyZ, hit.copyZ);
      maxCopyZ = Math.max(maxCopyZ, hit.copyZ);
    }
    if (worstRayLimit > 0) {
      const candidate: ExteriorCensusWorstRay = {
        phase: [ray.phaseX, ray.phaseZ],
        phaseNormalized: [
          (ray.phaseX - geometry.tileOriginX) / geometry.tileSizeX,
          (ray.phaseZ - geometry.tileOriginZ) / geometry.tileSizeZ,
        ],
        azimuthDegrees: ray.azimuthDegrees,
        elevationDegrees: ray.elevationDegrees,
        direction: ray.direction,
        rootHitCount: hits.length,
        owners: hits,
      };
      worstRays.push(candidate);
      worstRays.sort((left, right) => right.rootHitCount - left.rootHitCount
        || left.elevationDegrees - right.elevationDegrees
        || left.azimuthDegrees - right.azimuthDegrees
        || left.phaseNormalized[1] - right.phaseNormalized[1]
        || left.phaseNormalized[0] - right.phaseNormalized[0]);
      if (worstRays.length > worstRayLimit) worstRays.length = worstRayLimit;
    }
  }
  const histogramObject: Record<string, number> = {};
  for (const count of [...histogram.keys()].sort((a, b) => a - b)) {
    histogramObject[String(count)] = histogram.get(count)!;
  }
  const byElevation: ExteriorClosureCensusReport['samples']['byElevation'] = {};
  for (const elevation of [...elevationHistograms.keys()].sort((a, b) => a - b)) {
    const elevationHistogram = elevationHistograms.get(elevation)!;
    const entryHistogram: Record<string, number> = {};
    let rayCount = 0;
    let hitReferences = 0;
    let elevationMaximum = 0;
    for (const count of [...elevationHistogram.keys()].sort((a, b) => a - b)) {
      const countRays = elevationHistogram.get(count)!;
      entryHistogram[String(count)] = countRays;
      rayCount += countRays;
      hitReferences += count * countRays;
      elevationMaximum = Math.max(elevationMaximum, count);
    }
    byElevation[String(elevation)] = {
      rayCount,
      rootHitReferences: hitReferences,
      maximum: elevationMaximum,
      histogram: entryHistogram,
    };
  }
  const maximum = Math.max(0, ...histogram.keys());
  const minimumElevation = Math.min(...options.elevationsDegrees);
  const verticalDropHorizon = geometry.topH - geometry.bounds.min[1];
  const minimumElevationRadians = minimumElevation * Math.PI / 180;
  const hasCopies = copies.size > 0;
  return {
    domain: {
      phaseGrid: options.phaseGrid,
      phaseOffsets: options.phaseOffsets ?? [0.5],
      azimuthCount: options.azimuthCount,
      elevationsDegrees: [...options.elevationsDegrees],
      minimumElevationDegrees: minimumElevation,
      topH: geometry.topH,
      minimumY: geometry.bounds.min[1],
      verticalDropHorizon,
      maximumRayDistance: verticalDropHorizon / Math.sin(minimumElevationRadians),
      maximumHorizontalReach: verticalDropHorizon / Math.tan(minimumElevationRadians),
    },
    accelerator: bvh.metrics,
    samples: {
      traceMilliseconds: performance.now() - traceStarted,
      rayCount: rays.length,
      rootHitReferences,
      distinctOwnerKeys: ownerKeys.size,
      distinctTriangles: triangles.size,
      distinctCopies: copies.size,
      histogram: histogramObject,
      byElevation,
      p50: histogramQuantile(histogram, rays.length, 0.5),
      p95: histogramQuantile(histogram, rays.length, 0.95),
      p99: histogramQuantile(histogram, rays.length, 0.99),
      maximum,
      missOnlyRays,
      copyRange: hasCopies ? {
        minX: minCopyX,
        maxX: maxCopyX,
        minZ: minCopyZ,
        maxZ: maxCopyZ,
        maximumAbsolute: Math.max(
          Math.abs(minCopyX),
          Math.abs(maxCopyX),
          Math.abs(minCopyZ),
          Math.abs(maxCopyZ),
        ),
      } : null,
      worstRays,
    },
    interpretation: [
      'each count is one analytic first owner per independently selectable periodic tile-copy root',
      'the maximum is a sampled pointwise lower bound on fixed K and cannot be reduced by adaptive cell subdivision',
      'the census is not a continuous four-dimensional owner-closure certificate',
      'geometry is the decoded GCRP/v4 u16 mesh; fixed-function owner texels are not used as analytic truth',
      'copy coordinates are wide signed integers and are not truncated to the v4 signed five-bit fields',
      'intersections use decoded positions in f64 with explicit parallel, barycentric, and bounds tolerances',
    ],
  };
}
