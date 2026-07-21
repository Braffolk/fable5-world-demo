import { createHash } from 'node:crypto';

/**
 * Standalone storage contract for the certified sparse owner closure which
 * supersedes GCRP/v4's sampled first-owner atlas. This module deliberately does
 * not extend the runtime GCRP parser: it is a bake-side v5 contract until an
 * asset has passed closure and memory certification.
 *
 * The exterior domain is four-dimensional:
 *   [periodic phase X, periodic phase Z, azimuth, elevation].
 * Every adaptive hyperoctree leaf points at one fixed-width candidate page.
 * Runtime work is therefore bounded by K exact live ray/triangle tests; the
 * file contains no interpolated first-hit depths.
 */

export const SPARSE_EXTERIOR_CLOSURE_MAGIC = 'GCCL';
export const SPARSE_EXTERIOR_CLOSURE_VERSION = 5;

export const SPARSE_EXTERIOR_HEADER_BYTES = 256;
export const SPARSE_EXTERIOR_LEAF_BYTES = 24;
export const SPARSE_EXTERIOR_PAGE_HEADER_BYTES = 16;
export const SPARSE_EXTERIOR_CANDIDATE_BYTES = 16;
export const SPARSE_EXTERIOR_TRIANGLE_REMAP_BYTES = 8;

const UINT32_MAX = 0xffff_ffff;
const INT32_MIN = -0x8000_0000;
const INT32_MAX = 0x7fff_ffff;
const MAX_TYPED_ARRAY_BYTES = 0x7fff_ffff;
const TWO_PI = Math.PI * 2;
const DOMAIN_EPSILON = 1e-6;

const HEADER_FLAG_AZIMUTH_WRAP = 1 << 0;
const HEADER_FLAG_COMPLETE_EXTERIOR_PARTITION = 1 << 1;
const HEADER_FLAG_INSIDE_DIRECTORY_RESERVED = 1 << 2;
const HEADER_FLAG_PREDICATE_FILTERED = 1 << 3;

export const EXTERIOR_LEAF_FLAG_CERTIFIED_OWNER_CLOSURE = 1 << 0;
export const EXTERIOR_LEAF_FLAG_PREDICATE_FILTERED_SUCCESSOR = 1 << 1;
export const EXTERIOR_LEAF_FLAG_BOUNDARY_TIES_INCLUDED = 1 << 2;

export enum SparseClosurePredicateKind {
  /** Every periodic root is eligible. */
  AllRoots = 0,
  /** Pages were certified after applying one immutable cooked-root predicate. */
  BakedEligibleSuccessor = 1,
}

export interface SparseClosureTileDomain {
  originX: number;
  originZ: number;
  sizeX: number;
  sizeZ: number;
  topH: number;
}

export interface SparseClosureDirectionDomain {
  azimuthMin: number;
  azimuthMax: number;
  elevationMin: number;
  elevationMax: number;
  /** True only when the two azimuth endpoints are the same periodic boundary. */
  azimuthWrap: boolean;
}

export interface SparseClosurePredicateMetadata {
  kind: SparseClosurePredicateKind;
  revision: number;
  /** Hash of the exact eligibility field/recipe used during successor certification. */
  sha256: string;
}

export interface SparseExteriorClosureLeaf {
  /** Hyperoctree depth. One subdivision bisects all four dimensions. */
  level: number;
  /** Integer cell coordinate at `level`, ordered phaseX/phaseZ/azimuth/elevation. */
  origin: readonly [number, number, number, number];
  pageIndex: number;
  flags: number;
}

export interface SparseExteriorClosureCandidate {
  profileId: number;
  sourceTriangleId: number;
  /** Signed periodic owner-copy offsets. Int32 replaces v4's signed five bits. */
  copyX: number;
  copyZ: number;
  /** Opaque root/predicate identity retained for exact eligibility and tie policy. */
  predicateToken: number;
}

export interface SparseExteriorClosurePage {
  candidates: readonly SparseExteriorClosureCandidate[];
  flags?: number;
}

export interface SparseExteriorClosureDefinition {
  tile: SparseClosureTileDomain;
  direction: SparseClosureDirectionDomain;
  /** Finite forward profile-space distance over which closure was certified. */
  horizon: number;
  /** Runtime candidate count. Every page occupies exactly this many slots. */
  candidateCapacity: number;
  /** Fixed maximum lookup depth; runtime traversal may be statically unrolled to it. */
  maxSubdivisionDepth: number;
  predicate: SparseClosurePredicateMetadata;
  /** Hash of the source geometry/profile set against which triangle ids were certified. */
  sourceSha256: string;
  exteriorLeaves: readonly SparseExteriorClosureLeaf[];
  exteriorPages: readonly SparseExteriorClosurePage[];
}

/** No pack is allowed without explicit limits. These are bake rejection gates. */
export interface SparseExteriorClosureByteGates {
  maxTotalBytes: number;
  maxExteriorLeafBytes: number;
  maxExteriorPageBytes: number;
  maxTriangleRemapBytes: number;
  maxExteriorLeaves: number;
  maxExteriorPages: number;
  maxTriangleRemapEntries: number;
  maxCandidateCapacity: number;
  maxSubdivisionDepth: number;
  maxAbsCopyOffset: number;
}

export interface SparseExteriorClosureByteLedger {
  headerBytes: number;
  exteriorLeafBytes: number;
  exteriorPageBytes: number;
  triangleRemapBytes: number;
  paddingBytes: number;
  totalBytes: number;
  exteriorLeafOffset: number;
  exteriorPageOffset: number;
  triangleRemapOffset: number;
  /** Reserved directory entry; v5 exterior packs leave the section empty. */
  insideLeafOffset: number;
  /** Reserved directory entry; v5 exterior packs leave the section empty. */
  insidePageOffset: number;
}

export interface SparseClosureTriangleRemapEntry {
  profileId: number;
  sourceTriangleId: number;
}

export interface ParsedSparseClosureCandidate {
  triangleRemapIndex: number;
  copyX: number;
  copyZ: number;
  predicateToken: number;
}

export interface ParsedSparseClosurePage {
  flags: number;
  candidates: ParsedSparseClosureCandidate[];
}

export interface SparseExteriorClosureHeader {
  version: number;
  candidateCapacity: number;
  maxSubdivisionDepth: number;
  exteriorLeafCount: number;
  exteriorPageCount: number;
  triangleRemapCount: number;
  exteriorLeafOffset: number;
  exteriorPageOffset: number;
  triangleRemapOffset: number;
  endOffset: number;
  insideLeafOffset: number;
  insidePageOffset: number;
  insideLeafCount: number;
  insidePageCount: number;
  /** Reserved K_in for a future, separately budgeted successor-page section. */
  insideCandidateCapacity: number;
  horizon: number;
  direction: SparseClosureDirectionDomain;
  tile: SparseClosureTileDomain;
  predicate: SparseClosurePredicateMetadata;
  sourceSha256: string;
}

export interface PackedSparseExteriorClosureV5 {
  bytes: Uint8Array;
  sha256: string;
  header: SparseExteriorClosureHeader;
  ledger: SparseExteriorClosureByteLedger;
}

export interface ParsedSparseExteriorClosureV5 {
  header: SparseExteriorClosureHeader;
  ledger: SparseExteriorClosureByteLedger;
  exteriorLeaves: SparseExteriorClosureLeaf[];
  exteriorPages: ParsedSparseClosurePage[];
  triangleRemap: SparseClosureTriangleRemapEntry[];
  sha256: string;
}

interface NormalizedCandidate extends SparseExteriorClosureCandidate {
  triangleRemapIndex: number;
}

interface NormalizedPage {
  flags: number;
  candidates: NormalizedCandidate[];
}

interface LeafInterval {
  leaf: SparseExteriorClosureLeaf;
  start: bigint;
  end: bigint;
}

interface NormalizedClosure {
  definition: SparseExteriorClosureDefinition;
  leaves: SparseExteriorClosureLeaf[];
  pages: NormalizedPage[];
  triangleRemap: SparseClosureTriangleRemapEntry[];
  maxAbsCopyOffset: number;
  ledger: SparseExteriorClosureByteLedger;
}

function finite(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new Error(`${label} must be finite`);
  return value;
}

function uint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > UINT32_MAX) {
    throw new Error(`${label} must be a uint32`);
  }
  return value;
}

function int32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < INT32_MIN || value > INT32_MAX) {
    throw new Error(`${label} must be an int32`);
  }
  return value;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label} must be a positive safe integer`);
  }
  return value;
}

function nonNegativeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} must be a non-negative safe integer`);
  }
  return value;
}

function checkedAdd(left: number, right: number, label: string): number {
  const value = left + right;
  if (!Number.isSafeInteger(value)) throw new Error(`${label} exceeds safe integer range`);
  return value;
}

function checkedMultiply(left: number, right: number, label: string): number {
  const value = left * right;
  if (!Number.isSafeInteger(value)) throw new Error(`${label} exceeds safe integer range`);
  return value;
}

function align16(value: number): number {
  return checkedMultiply(Math.ceil(value / 16), 16, 'aligned closure offset');
}

function parseHash(hash: string, label: string): Uint8Array {
  if (!/^[0-9a-f]{64}$/i.test(hash)) throw new Error(`${label} must be a 64-character SHA-256 hex string`);
  const bytes = new Uint8Array(32);
  for (let i = 0; i < bytes.length; i++) bytes[i] = Number.parseInt(hash.slice(i * 2, i * 2 + 2), 16);
  return bytes;
}

function hashHex(bytes: Uint8Array): string {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
}

function safeOffset(view: DataView, offset: number, label: string): number {
  const value = view.getBigUint64(offset, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} exceeds safe integer range`);
  return Number(value);
}

function setOffset(view: DataView, offset: number, value: number): void {
  view.setBigUint64(offset, BigInt(value), true);
}

function validateDomain(definition: SparseExteriorClosureDefinition): void {
  const tile = definition.tile;
  finite(tile.originX, 'tile originX');
  finite(tile.originZ, 'tile originZ');
  finite(tile.topH, 'tile topH');
  if (!(finite(tile.sizeX, 'tile sizeX') > 0) || !(finite(tile.sizeZ, 'tile sizeZ') > 0)) {
    throw new Error('tile dimensions must be positive');
  }
  if (!(finite(definition.horizon, 'closure horizon') > 0)) {
    throw new Error('closure horizon must be positive');
  }
  const direction = definition.direction;
  const azimuthMin = finite(direction.azimuthMin, 'azimuth minimum');
  const azimuthMax = finite(direction.azimuthMax, 'azimuth maximum');
  const elevationMin = finite(direction.elevationMin, 'elevation minimum');
  const elevationMax = finite(direction.elevationMax, 'elevation maximum');
  const azimuthSpan = azimuthMax - azimuthMin;
  if (!(azimuthSpan > 0) || azimuthSpan > TWO_PI + DOMAIN_EPSILON) {
    throw new Error('azimuth domain must have positive span no greater than 2pi');
  }
  if (direction.azimuthWrap && Math.abs(azimuthSpan - TWO_PI) > DOMAIN_EPSILON) {
    throw new Error('wrapped azimuth domain must span exactly 2pi');
  }
  if (
    elevationMin < 0
    || elevationMax > Math.PI / 2 + DOMAIN_EPSILON
    || !(elevationMax > elevationMin)
  ) {
    throw new Error('elevation domain must be a positive interval within [0, pi/2]');
  }
  positiveInteger(definition.candidateCapacity, 'candidate capacity');
  uint32(definition.candidateCapacity, 'candidate capacity');
  nonNegativeInteger(definition.maxSubdivisionDepth, 'maximum subdivision depth');
  if (definition.maxSubdivisionDepth > 31) {
    throw new Error('maximum subdivision depth exceeds uint32 leaf coordinates');
  }
  uint32(definition.predicate.revision, 'predicate revision');
  if (
    definition.predicate.kind !== SparseClosurePredicateKind.AllRoots
    && definition.predicate.kind !== SparseClosurePredicateKind.BakedEligibleSuccessor
  ) {
    throw new Error('unsupported closure predicate kind');
  }
  parseHash(definition.predicate.sha256, 'predicate hash');
  parseHash(definition.sourceSha256, 'source hash');
}

function mortonPrefix(origin: readonly number[], level: number): bigint {
  let code = 0n;
  for (let bit = level - 1; bit >= 0; bit--) {
    for (let dimension = 0; dimension < 4; dimension++) {
      code = (code << 1n) | ((BigInt(origin[dimension]!) >> BigInt(bit)) & 1n);
    }
  }
  return code;
}

function validateAndSortLeaves(
  leavesInput: readonly SparseExteriorClosureLeaf[],
  pageCount: number,
  maxDepth: number,
  predicateKind: SparseClosurePredicateKind,
): SparseExteriorClosureLeaf[] {
  if (leavesInput.length === 0) throw new Error('exterior closure requires at least one leaf');
  const intervals: LeafInterval[] = leavesInput.map((input, index) => {
    const level = nonNegativeInteger(input.level, `leaf ${index} level`);
    if (level > maxDepth) throw new Error(`leaf ${index} exceeds maximum subdivision depth`);
    const side = 1n << BigInt(level);
    const origin = input.origin.map((coordinate, dimension) => {
      uint32(coordinate, `leaf ${index} origin ${dimension}`);
      if (BigInt(coordinate) >= side) throw new Error(`leaf ${index} origin lies outside its level`);
      return coordinate;
    }) as unknown as [number, number, number, number];
    const pageIndex = uint32(input.pageIndex, `leaf ${index} page index`);
    if (pageIndex >= pageCount) throw new Error(`leaf ${index} page index is outside the page table`);
    const flags = uint32(input.flags, `leaf ${index} flags`);
    if (flags > 0xffff) throw new Error(`leaf ${index} flags exceed uint16 storage`);
    if ((flags & EXTERIOR_LEAF_FLAG_CERTIFIED_OWNER_CLOSURE) === 0) {
      throw new Error(`leaf ${index} is not marked as a certified owner closure`);
    }
    if ((flags & EXTERIOR_LEAF_FLAG_BOUNDARY_TIES_INCLUDED) === 0) {
      throw new Error(`leaf ${index} does not include boundary/tie owners`);
    }
    if (
      predicateKind === SparseClosurePredicateKind.BakedEligibleSuccessor
      && (flags & EXTERIOR_LEAF_FLAG_PREDICATE_FILTERED_SUCCESSOR) === 0
    ) {
      throw new Error(`leaf ${index} is not a predicate-filtered successor closure`);
    }
    const leaf: SparseExteriorClosureLeaf = { level, origin, pageIndex, flags };
    const prefix = mortonPrefix(origin, level);
    const shift = BigInt(4 * (maxDepth - level));
    return { leaf, start: prefix << shift, end: (prefix + 1n) << shift };
  });
  intervals.sort((left, right) => left.start < right.start ? -1 : left.start > right.start ? 1 : 0);
  let cursor = 0n;
  for (const interval of intervals) {
    if (interval.start < cursor) throw new Error('adaptive exterior leaves overlap');
    if (interval.start > cursor) throw new Error('adaptive exterior leaves leave a domain gap');
    cursor = interval.end;
  }
  const domainEnd = 1n << BigInt(4 * maxDepth);
  if (cursor !== domainEnd) throw new Error('adaptive exterior leaves leave a domain gap');
  return intervals.map((interval) => interval.leaf);
}

function candidateCompare(
  left: SparseExteriorClosureCandidate,
  right: SparseExteriorClosureCandidate,
): number {
  return left.profileId - right.profileId
    || left.sourceTriangleId - right.sourceTriangleId
    || left.copyX - right.copyX
    || left.copyZ - right.copyZ
    || left.predicateToken - right.predicateToken;
}

function candidateKey(candidate: SparseExteriorClosureCandidate): string {
  return `${candidate.profileId}:${candidate.sourceTriangleId}:${candidate.copyX}:${candidate.copyZ}:${candidate.predicateToken}`;
}

function normalize(definition: SparseExteriorClosureDefinition): NormalizedClosure {
  validateDomain(definition);
  const candidateCapacity = definition.candidateCapacity;
  if (definition.exteriorPages.length === 0) throw new Error('exterior closure requires at least one candidate page');
  uint32(definition.exteriorPages.length, 'exterior page count');

  const referencedTriangles = new Map<string, SparseClosureTriangleRemapEntry>();
  let maxAbsCopyOffset = 0;
  const canonicalPages = definition.exteriorPages.map((page, pageIndex) => {
    const flags = uint32(page.flags ?? 0, `page ${pageIndex} flags`);
    if (page.candidates.length > candidateCapacity) {
      throw new Error(`page ${pageIndex} exceeds fixed candidate capacity K=${candidateCapacity}`);
    }
    const seen = new Set<string>();
    const candidates = page.candidates.map((candidate, candidateIndex) => {
      const value: SparseExteriorClosureCandidate = {
        profileId: uint32(candidate.profileId, `page ${pageIndex} candidate ${candidateIndex} profile id`),
        sourceTriangleId: uint32(
          candidate.sourceTriangleId,
          `page ${pageIndex} candidate ${candidateIndex} source triangle id`,
        ),
        copyX: int32(candidate.copyX, `page ${pageIndex} candidate ${candidateIndex} copy X`),
        copyZ: int32(candidate.copyZ, `page ${pageIndex} candidate ${candidateIndex} copy Z`),
        predicateToken: uint32(
          candidate.predicateToken,
          `page ${pageIndex} candidate ${candidateIndex} predicate token`,
        ),
      };
      const key = candidateKey(value);
      if (seen.has(key)) throw new Error(`page ${pageIndex} contains a duplicate candidate`);
      seen.add(key);
      referencedTriangles.set(`${value.profileId}:${value.sourceTriangleId}`, {
        profileId: value.profileId,
        sourceTriangleId: value.sourceTriangleId,
      });
      maxAbsCopyOffset = Math.max(maxAbsCopyOffset, Math.abs(value.copyX), Math.abs(value.copyZ));
      return value;
    }).sort(candidateCompare);
    return { flags, candidates };
  });

  const triangleRemap = Array.from(referencedTriangles.values()).sort((left, right) =>
    left.profileId - right.profileId || left.sourceTriangleId - right.sourceTriangleId);
  const remapIndex = new Map(
    triangleRemap.map((entry, index) => [`${entry.profileId}:${entry.sourceTriangleId}`, index] as const),
  );
  const pages: NormalizedPage[] = canonicalPages.map((page) => ({
    flags: page.flags,
    candidates: page.candidates.map((candidate) => ({
      ...candidate,
      triangleRemapIndex: remapIndex.get(`${candidate.profileId}:${candidate.sourceTriangleId}`)!,
    })),
  }));

  const leaves = validateAndSortLeaves(
    definition.exteriorLeaves,
    pages.length,
    definition.maxSubdivisionDepth,
    definition.predicate.kind,
  );
  const referencedPages = new Uint8Array(pages.length);
  for (const leaf of leaves) referencedPages[leaf.pageIndex] = 1;
  for (let page = 0; page < referencedPages.length; page++) {
    if (referencedPages[page] === 0) throw new Error(`candidate page ${page} is not referenced by any leaf`);
  }

  const pageRecordBytes = checkedAdd(
    SPARSE_EXTERIOR_PAGE_HEADER_BYTES,
    checkedMultiply(candidateCapacity, SPARSE_EXTERIOR_CANDIDATE_BYTES, 'candidate page payload'),
    'candidate page record',
  );
  const exteriorLeafBytes = checkedMultiply(leaves.length, SPARSE_EXTERIOR_LEAF_BYTES, 'exterior leaf bytes');
  const exteriorPageBytes = checkedMultiply(pages.length, pageRecordBytes, 'exterior page bytes');
  const triangleRemapBytes = checkedMultiply(
    triangleRemap.length,
    SPARSE_EXTERIOR_TRIANGLE_REMAP_BYTES,
    'triangle remap bytes',
  );
  const exteriorLeafOffset = SPARSE_EXTERIOR_HEADER_BYTES;
  const exteriorPageOffset = align16(checkedAdd(exteriorLeafOffset, exteriorLeafBytes, 'page offset'));
  const triangleRemapOffset = align16(checkedAdd(exteriorPageOffset, exteriorPageBytes, 'remap offset'));
  const totalBytes = align16(checkedAdd(triangleRemapOffset, triangleRemapBytes, 'closure byte size'));
  const payloadBytes = exteriorLeafBytes + exteriorPageBytes + triangleRemapBytes;
  const ledger: SparseExteriorClosureByteLedger = {
    headerBytes: SPARSE_EXTERIOR_HEADER_BYTES,
    exteriorLeafBytes,
    exteriorPageBytes,
    triangleRemapBytes,
    paddingBytes: totalBytes - SPARSE_EXTERIOR_HEADER_BYTES - payloadBytes,
    totalBytes,
    exteriorLeafOffset,
    exteriorPageOffset,
    triangleRemapOffset,
    insideLeafOffset: 0,
    insidePageOffset: 0,
  };
  return { definition, leaves, pages, triangleRemap, maxAbsCopyOffset, ledger };
}

function validateGates(gates: SparseExteriorClosureByteGates): void {
  positiveInteger(gates.maxTotalBytes, 'total byte gate');
  nonNegativeInteger(gates.maxExteriorLeafBytes, 'exterior leaf byte gate');
  nonNegativeInteger(gates.maxExteriorPageBytes, 'exterior page byte gate');
  nonNegativeInteger(gates.maxTriangleRemapBytes, 'triangle remap byte gate');
  positiveInteger(gates.maxExteriorLeaves, 'exterior leaf count gate');
  positiveInteger(gates.maxExteriorPages, 'exterior page count gate');
  nonNegativeInteger(gates.maxTriangleRemapEntries, 'triangle remap count gate');
  positiveInteger(gates.maxCandidateCapacity, 'candidate capacity gate');
  nonNegativeInteger(gates.maxSubdivisionDepth, 'subdivision depth gate');
  nonNegativeInteger(gates.maxAbsCopyOffset, 'copy offset gate');
  if (gates.maxAbsCopyOffset > INT32_MAX) throw new Error('copy offset gate exceeds int32 storage');
}

function applyGates(
  normalized: Pick<NormalizedClosure, 'definition' | 'leaves' | 'pages' | 'triangleRemap' | 'maxAbsCopyOffset' | 'ledger'>,
  gates: SparseExteriorClosureByteGates,
): void {
  validateGates(gates);
  const { definition, leaves, pages, triangleRemap, maxAbsCopyOffset, ledger } = normalized;
  if (definition.candidateCapacity > gates.maxCandidateCapacity) throw new Error('candidate capacity exceeds hard K gate');
  if (definition.maxSubdivisionDepth > gates.maxSubdivisionDepth) throw new Error('subdivision depth exceeds hard gate');
  if (leaves.length > gates.maxExteriorLeaves) throw new Error('exterior leaf count exceeds hard gate');
  if (pages.length > gates.maxExteriorPages) throw new Error('exterior page count exceeds hard gate');
  if (triangleRemap.length > gates.maxTriangleRemapEntries) throw new Error('triangle remap count exceeds hard gate');
  if (maxAbsCopyOffset > gates.maxAbsCopyOffset) throw new Error('signed periodic copy offset exceeds hard gate');
  if (ledger.exteriorLeafBytes > gates.maxExteriorLeafBytes) throw new Error('exterior leaf bytes exceed hard gate');
  if (ledger.exteriorPageBytes > gates.maxExteriorPageBytes) throw new Error('exterior page bytes exceed hard gate');
  if (ledger.triangleRemapBytes > gates.maxTriangleRemapBytes) throw new Error('triangle remap bytes exceed hard gate');
  if (ledger.totalBytes > gates.maxTotalBytes) throw new Error('total closure bytes exceed hard gate');
  if (ledger.totalBytes > MAX_TYPED_ARRAY_BYTES) throw new Error('closure exceeds typed-array implementation limit');
}

function makeHeader(normalized: NormalizedClosure): SparseExteriorClosureHeader {
  const { definition, leaves, pages, triangleRemap, ledger } = normalized;
  return {
    version: SPARSE_EXTERIOR_CLOSURE_VERSION,
    candidateCapacity: definition.candidateCapacity,
    maxSubdivisionDepth: definition.maxSubdivisionDepth,
    exteriorLeafCount: leaves.length,
    exteriorPageCount: pages.length,
    triangleRemapCount: triangleRemap.length,
    exteriorLeafOffset: ledger.exteriorLeafOffset,
    exteriorPageOffset: ledger.exteriorPageOffset,
    triangleRemapOffset: ledger.triangleRemapOffset,
    endOffset: ledger.totalBytes,
    insideLeafOffset: 0,
    insidePageOffset: 0,
    insideLeafCount: 0,
    insidePageCount: 0,
    insideCandidateCapacity: 0,
    horizon: definition.horizon,
    direction: { ...definition.direction },
    tile: { ...definition.tile },
    predicate: { ...definition.predicate },
    sourceSha256: definition.sourceSha256,
  };
}

export function measureSparseExteriorClosureV5(
  definition: SparseExteriorClosureDefinition,
): SparseExteriorClosureByteLedger {
  return { ...normalize(definition).ledger };
}

export function packSparseExteriorClosureV5(
  definition: SparseExteriorClosureDefinition,
  gates: SparseExteriorClosureByteGates,
): PackedSparseExteriorClosureV5 {
  const normalized = normalize(definition);
  applyGates(normalized, gates);
  const { ledger, leaves, pages, triangleRemap } = normalized;
  const bytes = new Uint8Array(ledger.totalBytes);
  const view = new DataView(bytes.buffer);
  bytes.set(Array.from(SPARSE_EXTERIOR_CLOSURE_MAGIC, (character) => character.charCodeAt(0)), 0);
  view.setUint32(4, SPARSE_EXTERIOR_CLOSURE_VERSION, true);
  view.setUint32(8, SPARSE_EXTERIOR_HEADER_BYTES, true);
  let headerFlags = HEADER_FLAG_COMPLETE_EXTERIOR_PARTITION | HEADER_FLAG_INSIDE_DIRECTORY_RESERVED;
  if (definition.direction.azimuthWrap) headerFlags |= HEADER_FLAG_AZIMUTH_WRAP;
  if (definition.predicate.kind === SparseClosurePredicateKind.BakedEligibleSuccessor) {
    headerFlags |= HEADER_FLAG_PREDICATE_FILTERED;
  }
  view.setUint32(12, headerFlags, true);
  view.setUint32(16, definition.candidateCapacity, true);
  view.setUint32(20, definition.maxSubdivisionDepth, true);
  view.setUint32(24, leaves.length, true);
  view.setUint32(28, pages.length, true);
  view.setUint32(32, triangleRemap.length, true);
  view.setUint32(36, SPARSE_EXTERIOR_LEAF_BYTES, true);
  view.setUint32(40, SPARSE_EXTERIOR_CANDIDATE_BYTES, true);
  view.setUint32(
    44,
    SPARSE_EXTERIOR_PAGE_HEADER_BYTES + definition.candidateCapacity * SPARSE_EXTERIOR_CANDIDATE_BYTES,
    true,
  );
  view.setUint32(48, SPARSE_EXTERIOR_TRIANGLE_REMAP_BYTES, true);
  view.setUint32(52, 0, true); // reserved K_in for separate inside successor pages
  setOffset(view, 56, ledger.exteriorLeafOffset);
  setOffset(view, 64, ledger.exteriorPageOffset);
  setOffset(view, 72, ledger.triangleRemapOffset);
  setOffset(view, 80, ledger.totalBytes);
  setOffset(view, 88, 0);
  setOffset(view, 96, 0);
  view.setUint32(104, 0, true);
  view.setUint32(108, 0, true);
  view.setFloat32(112, definition.horizon, true);
  view.setFloat32(116, definition.direction.azimuthMin, true);
  view.setFloat32(120, definition.direction.azimuthMax, true);
  view.setFloat32(124, definition.direction.elevationMin, true);
  view.setFloat32(128, definition.direction.elevationMax, true);
  view.setFloat32(132, definition.tile.originX, true);
  view.setFloat32(136, definition.tile.originZ, true);
  view.setFloat32(140, definition.tile.sizeX, true);
  view.setFloat32(144, definition.tile.sizeZ, true);
  view.setFloat32(148, definition.tile.topH, true);
  view.setUint32(152, definition.predicate.kind, true);
  view.setUint32(156, definition.predicate.revision, true);
  bytes.set(parseHash(definition.predicate.sha256, 'predicate hash'), 160);
  bytes.set(parseHash(definition.sourceSha256, 'source hash'), 192);
  setOffset(view, 224, ledger.exteriorLeafBytes);
  setOffset(view, 232, ledger.exteriorPageBytes);
  setOffset(view, 240, ledger.triangleRemapBytes);
  setOffset(view, 248, ledger.totalBytes);

  leaves.forEach((leaf, index) => {
    const offset = ledger.exteriorLeafOffset + index * SPARSE_EXTERIOR_LEAF_BYTES;
    for (let dimension = 0; dimension < 4; dimension++) {
      view.setUint32(offset + dimension * 4, leaf.origin[dimension]!, true);
    }
    view.setUint32(offset + 16, leaf.pageIndex, true);
    view.setUint16(offset + 20, leaf.level, true);
    view.setUint16(offset + 22, leaf.flags, true);
  });

  const pageRecordBytes = SPARSE_EXTERIOR_PAGE_HEADER_BYTES
    + definition.candidateCapacity * SPARSE_EXTERIOR_CANDIDATE_BYTES;
  pages.forEach((page, pageIndex) => {
    const offset = ledger.exteriorPageOffset + pageIndex * pageRecordBytes;
    view.setUint32(offset, page.candidates.length, true);
    view.setUint32(offset + 4, page.flags, true);
    view.setUint32(offset + 8, SPARSE_EXTERIOR_CANDIDATE_BYTES, true);
    for (let slot = 0; slot < definition.candidateCapacity; slot++) {
      const candidateOffset = offset + SPARSE_EXTERIOR_PAGE_HEADER_BYTES
        + slot * SPARSE_EXTERIOR_CANDIDATE_BYTES;
      const candidate = page.candidates[slot];
      if (!candidate) {
        view.setUint32(candidateOffset, UINT32_MAX, true);
        continue;
      }
      view.setUint32(candidateOffset, candidate.triangleRemapIndex, true);
      view.setInt32(candidateOffset + 4, candidate.copyX, true);
      view.setInt32(candidateOffset + 8, candidate.copyZ, true);
      view.setUint32(candidateOffset + 12, candidate.predicateToken, true);
    }
  });

  triangleRemap.forEach((entry, index) => {
    const offset = ledger.triangleRemapOffset + index * SPARSE_EXTERIOR_TRIANGLE_REMAP_BYTES;
    view.setUint32(offset, entry.profileId, true);
    view.setUint32(offset + 4, entry.sourceTriangleId, true);
  });

  const sha256 = createHash('sha256').update(bytes).digest('hex');
  return { bytes, sha256, header: makeHeader(normalized), ledger: { ...ledger } };
}

function parseHeader(bytes: Uint8Array): {
  header: SparseExteriorClosureHeader;
  flags: number;
  pageRecordBytes: number;
  ledger: SparseExteriorClosureByteLedger;
} {
  if (
    bytes.byteLength < SPARSE_EXTERIOR_HEADER_BYTES
    || String.fromCharCode(...bytes.subarray(0, 4)) !== SPARSE_EXTERIOR_CLOSURE_MAGIC
  ) {
    throw new Error('not a standalone GCCL closure');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(4, true);
  if (version !== SPARSE_EXTERIOR_CLOSURE_VERSION) throw new Error(`expected GCCL/v5, got v${version}`);
  if (view.getUint32(8, true) !== SPARSE_EXTERIOR_HEADER_BYTES) throw new Error('noncanonical GCCL header size');
  const flags = view.getUint32(12, true);
  const requiredFlags = HEADER_FLAG_COMPLETE_EXTERIOR_PARTITION | HEADER_FLAG_INSIDE_DIRECTORY_RESERVED;
  if ((flags & requiredFlags) !== requiredFlags) throw new Error('GCCL lacks complete-partition or inside-directory contract');
  const candidateCapacity = view.getUint32(16, true);
  const maxSubdivisionDepth = view.getUint32(20, true);
  const exteriorLeafCount = view.getUint32(24, true);
  const exteriorPageCount = view.getUint32(28, true);
  const triangleRemapCount = view.getUint32(32, true);
  if (view.getUint32(36, true) !== SPARSE_EXTERIOR_LEAF_BYTES) throw new Error('noncanonical GCCL leaf record size');
  if (view.getUint32(40, true) !== SPARSE_EXTERIOR_CANDIDATE_BYTES) throw new Error('noncanonical GCCL candidate record size');
  const pageRecordBytes = view.getUint32(44, true);
  if (pageRecordBytes !== SPARSE_EXTERIOR_PAGE_HEADER_BYTES + candidateCapacity * SPARSE_EXTERIOR_CANDIDATE_BYTES) {
    throw new Error('noncanonical GCCL page record size');
  }
  if (view.getUint32(48, true) !== SPARSE_EXTERIOR_TRIANGLE_REMAP_BYTES) {
    throw new Error('noncanonical GCCL triangle remap record size');
  }
  const insideCandidateCapacity = view.getUint32(52, true);
  const exteriorLeafOffset = safeOffset(view, 56, 'exterior leaf offset');
  const exteriorPageOffset = safeOffset(view, 64, 'exterior page offset');
  const triangleRemapOffset = safeOffset(view, 72, 'triangle remap offset');
  const endOffset = safeOffset(view, 80, 'closure end offset');
  const insideLeafOffset = safeOffset(view, 88, 'inside leaf offset');
  const insidePageOffset = safeOffset(view, 96, 'inside page offset');
  const insideLeafCount = view.getUint32(104, true);
  const insidePageCount = view.getUint32(108, true);
  if (
    insideCandidateCapacity !== 0
    || insideLeafOffset !== 0
    || insidePageOffset !== 0
    || insideLeafCount !== 0
    || insidePageCount !== 0
  ) {
    throw new Error('GCCL/v5 reserves but does not yet encode inside successor pages');
  }
  const predicateKind = view.getUint32(152, true);
  if (
    predicateKind !== SparseClosurePredicateKind.AllRoots
    && predicateKind !== SparseClosurePredicateKind.BakedEligibleSuccessor
  ) throw new Error('unsupported GCCL predicate kind');
  const direction: SparseClosureDirectionDomain = {
    azimuthMin: view.getFloat32(116, true),
    azimuthMax: view.getFloat32(120, true),
    elevationMin: view.getFloat32(124, true),
    elevationMax: view.getFloat32(128, true),
    azimuthWrap: (flags & HEADER_FLAG_AZIMUTH_WRAP) !== 0,
  };
  const predicate: SparseClosurePredicateMetadata = {
    kind: predicateKind,
    revision: view.getUint32(156, true),
    sha256: hashHex(bytes.subarray(160, 192)),
  };
  const tile: SparseClosureTileDomain = {
    originX: view.getFloat32(132, true),
    originZ: view.getFloat32(136, true),
    sizeX: view.getFloat32(140, true),
    sizeZ: view.getFloat32(144, true),
    topH: view.getFloat32(148, true),
  };
  const horizon = view.getFloat32(112, true);
  const definitionForDomain: SparseExteriorClosureDefinition = {
    tile,
    direction,
    horizon,
    candidateCapacity,
    maxSubdivisionDepth,
    predicate,
    sourceSha256: hashHex(bytes.subarray(192, 224)),
    exteriorLeaves: [],
    exteriorPages: [],
  };
  validateDomain(definitionForDomain);
  if (
    predicateKind === SparseClosurePredicateKind.BakedEligibleSuccessor
    && (flags & HEADER_FLAG_PREDICATE_FILTERED) === 0
  ) throw new Error('predicate-filtered GCCL metadata flag is missing');
  const exteriorLeafBytes = safeOffset(view, 224, 'exterior leaf bytes');
  const exteriorPageBytes = safeOffset(view, 232, 'exterior page bytes');
  const triangleRemapBytes = safeOffset(view, 240, 'triangle remap bytes');
  const recordedTotal = safeOffset(view, 248, 'recorded total bytes');
  const expectedLeafBytes = checkedMultiply(exteriorLeafCount, SPARSE_EXTERIOR_LEAF_BYTES, 'parsed leaf bytes');
  const expectedPageBytes = checkedMultiply(exteriorPageCount, pageRecordBytes, 'parsed page bytes');
  const expectedRemapBytes = checkedMultiply(
    triangleRemapCount,
    SPARSE_EXTERIOR_TRIANGLE_REMAP_BYTES,
    'parsed remap bytes',
  );
  if (
    exteriorLeafBytes !== expectedLeafBytes
    || exteriorPageBytes !== expectedPageBytes
    || triangleRemapBytes !== expectedRemapBytes
  ) throw new Error('GCCL section byte ledger is inconsistent');
  const expectedPageOffset = align16(SPARSE_EXTERIOR_HEADER_BYTES + exteriorLeafBytes);
  const expectedRemapOffset = align16(expectedPageOffset + exteriorPageBytes);
  const expectedEnd = align16(expectedRemapOffset + triangleRemapBytes);
  if (
    exteriorLeafOffset !== SPARSE_EXTERIOR_HEADER_BYTES
    || exteriorPageOffset !== expectedPageOffset
    || triangleRemapOffset !== expectedRemapOffset
    || endOffset !== expectedEnd
    || recordedTotal !== expectedEnd
    || bytes.byteLength !== expectedEnd
  ) throw new Error('GCCL table offsets or payload length are noncanonical');
  const payloadBytes = exteriorLeafBytes + exteriorPageBytes + triangleRemapBytes;
  const ledger: SparseExteriorClosureByteLedger = {
    headerBytes: SPARSE_EXTERIOR_HEADER_BYTES,
    exteriorLeafBytes,
    exteriorPageBytes,
    triangleRemapBytes,
    paddingBytes: expectedEnd - SPARSE_EXTERIOR_HEADER_BYTES - payloadBytes,
    totalBytes: expectedEnd,
    exteriorLeafOffset,
    exteriorPageOffset,
    triangleRemapOffset,
    insideLeafOffset,
    insidePageOffset,
  };
  return {
    flags,
    pageRecordBytes,
    ledger,
    header: {
      version,
      candidateCapacity,
      maxSubdivisionDepth,
      exteriorLeafCount,
      exteriorPageCount,
      triangleRemapCount,
      exteriorLeafOffset,
      exteriorPageOffset,
      triangleRemapOffset,
      endOffset,
      insideLeafOffset,
      insidePageOffset,
      insideLeafCount,
      insidePageCount,
      insideCandidateCapacity,
      horizon,
      direction,
      tile,
      predicate,
      sourceSha256: definitionForDomain.sourceSha256,
    },
  };
}

export function parseSparseExteriorClosureV5(
  bytes: Uint8Array,
  gates?: SparseExteriorClosureByteGates,
): ParsedSparseExteriorClosureV5 {
  const { header, pageRecordBytes, ledger } = parseHeader(bytes);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const exteriorLeaves: SparseExteriorClosureLeaf[] = [];
  for (let index = 0; index < header.exteriorLeafCount; index++) {
    const offset = ledger.exteriorLeafOffset + index * SPARSE_EXTERIOR_LEAF_BYTES;
    exteriorLeaves.push({
      origin: [
        view.getUint32(offset, true),
        view.getUint32(offset + 4, true),
        view.getUint32(offset + 8, true),
        view.getUint32(offset + 12, true),
      ],
      pageIndex: view.getUint32(offset + 16, true),
      level: view.getUint16(offset + 20, true),
      flags: view.getUint16(offset + 22, true),
    });
  }
  const sortedLeaves = validateAndSortLeaves(
    exteriorLeaves,
    header.exteriorPageCount,
    header.maxSubdivisionDepth,
    header.predicate.kind,
  );
  if (sortedLeaves.some((leaf, index) => leaf !== exteriorLeaves[index])) {
    // Object identity differs after reconstruction, so compare the canonical keys.
    for (let index = 0; index < sortedLeaves.length; index++) {
      const left = sortedLeaves[index]!;
      const right = exteriorLeaves[index]!;
      if (
        left.level !== right.level
        || left.pageIndex !== right.pageIndex
        || left.flags !== right.flags
        || left.origin.some((coordinate, dimension) => coordinate !== right.origin[dimension])
      ) throw new Error('GCCL exterior leaves are not in canonical Morton order');
    }
  }

  const triangleRemap: SparseClosureTriangleRemapEntry[] = [];
  for (let index = 0; index < header.triangleRemapCount; index++) {
    const offset = ledger.triangleRemapOffset + index * SPARSE_EXTERIOR_TRIANGLE_REMAP_BYTES;
    triangleRemap.push({
      profileId: view.getUint32(offset, true),
      sourceTriangleId: view.getUint32(offset + 4, true),
    });
  }
  for (let index = 1; index < triangleRemap.length; index++) {
    const previous = triangleRemap[index - 1]!;
    const current = triangleRemap[index]!;
    if (
      previous.profileId > current.profileId
      || (previous.profileId === current.profileId && previous.sourceTriangleId >= current.sourceTriangleId)
    ) throw new Error('GCCL triangle remap is not strictly canonical');
  }

  const exteriorPages: ParsedSparseClosurePage[] = [];
  let maxAbsCopyOffset = 0;
  for (let pageIndex = 0; pageIndex < header.exteriorPageCount; pageIndex++) {
    const offset = ledger.exteriorPageOffset + pageIndex * pageRecordBytes;
    const candidateCount = view.getUint32(offset, true);
    if (candidateCount > header.candidateCapacity) throw new Error('GCCL page exceeds fixed candidate capacity');
    if (view.getUint32(offset + 8, true) !== SPARSE_EXTERIOR_CANDIDATE_BYTES) {
      throw new Error('GCCL page candidate stride is invalid');
    }
    if (view.getUint32(offset + 12, true) !== 0) throw new Error('GCCL page reserved field is nonzero');
    const candidates: ParsedSparseClosureCandidate[] = [];
    for (let slot = 0; slot < header.candidateCapacity; slot++) {
      const candidateOffset = offset + SPARSE_EXTERIOR_PAGE_HEADER_BYTES
        + slot * SPARSE_EXTERIOR_CANDIDATE_BYTES;
      const triangleRemapIndex = view.getUint32(candidateOffset, true);
      const copyX = view.getInt32(candidateOffset + 4, true);
      const copyZ = view.getInt32(candidateOffset + 8, true);
      const predicateToken = view.getUint32(candidateOffset + 12, true);
      if (slot >= candidateCount) {
        if (triangleRemapIndex !== UINT32_MAX || copyX !== 0 || copyZ !== 0 || predicateToken !== 0) {
          throw new Error('GCCL unused fixed-width candidate slot is not canonical');
        }
        continue;
      }
      if (triangleRemapIndex >= triangleRemap.length) throw new Error('GCCL candidate remap index is outside the table');
      maxAbsCopyOffset = Math.max(maxAbsCopyOffset, Math.abs(copyX), Math.abs(copyZ));
      candidates.push({ triangleRemapIndex, copyX, copyZ, predicateToken });
    }
    exteriorPages.push({ flags: view.getUint32(offset + 4, true), candidates });
  }
  const referencedPages = new Uint8Array(header.exteriorPageCount);
  for (const leaf of exteriorLeaves) referencedPages[leaf.pageIndex] = 1;
  for (let page = 0; page < referencedPages.length; page++) {
    if (referencedPages[page] === 0) throw new Error(`GCCL candidate page ${page} is unreferenced`);
  }

  if (gates) {
    applyGates({
      definition: {
        tile: header.tile,
        direction: header.direction,
        horizon: header.horizon,
        candidateCapacity: header.candidateCapacity,
        maxSubdivisionDepth: header.maxSubdivisionDepth,
        predicate: header.predicate,
        sourceSha256: header.sourceSha256,
        exteriorLeaves,
        exteriorPages: [],
      },
      leaves: exteriorLeaves,
      pages: exteriorPages.map((page) => ({ flags: page.flags, candidates: [] })),
      triangleRemap,
      maxAbsCopyOffset,
      ledger,
    }, gates);
  }
  return {
    header,
    ledger,
    exteriorLeaves,
    exteriorPages,
    triangleRemap,
    sha256: createHash('sha256').update(bytes).digest('hex'),
  };
}
