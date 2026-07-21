import assert from 'node:assert/strict';
import test from 'node:test';
import {
  EXTERIOR_LEAF_FLAG_BOUNDARY_TIES_INCLUDED,
  EXTERIOR_LEAF_FLAG_CERTIFIED_OWNER_CLOSURE,
  EXTERIOR_LEAF_FLAG_PREDICATE_FILTERED_SUCCESSOR,
  SPARSE_EXTERIOR_CANDIDATE_BYTES,
  SPARSE_EXTERIOR_HEADER_BYTES,
  SPARSE_EXTERIOR_LEAF_BYTES,
  SPARSE_EXTERIOR_PAGE_HEADER_BYTES,
  SPARSE_EXTERIOR_TRIANGLE_REMAP_BYTES,
  SparseClosurePredicateKind,
  measureSparseExteriorClosureV5,
  packSparseExteriorClosureV5,
  parseSparseExteriorClosureV5,
  type SparseExteriorClosureByteGates,
  type SparseExteriorClosureDefinition,
  type SparseExteriorClosureLeaf,
} from './SparseExteriorClosureV5';

const CERTIFIED_FLAGS = EXTERIOR_LEAF_FLAG_CERTIFIED_OWNER_CLOSURE
  | EXTERIOR_LEAF_FLAG_PREDICATE_FILTERED_SUCCESSOR
  | EXTERIOR_LEAF_FLAG_BOUNDARY_TIES_INCLUDED;

function adaptiveLeaves(): SparseExteriorClosureLeaf[] {
  const leaves: SparseExteriorClosureLeaf[] = [];
  // Split the level-zero domain into 16 level-one children, then split the
  // [0,0,0,0] child once more. The resulting 31 leaves are a complete adaptive
  // 4D partition and exercise Morton-prefix ordering independently of input order.
  for (let phaseX = 0; phaseX < 2; phaseX++) {
    for (let phaseZ = 0; phaseZ < 2; phaseZ++) {
      for (let azimuth = 0; azimuth < 2; azimuth++) {
        for (let elevation = 0; elevation < 2; elevation++) {
          if (phaseX === 0 && phaseZ === 0 && azimuth === 0 && elevation === 0) continue;
          leaves.push({
            level: 1,
            origin: [phaseX, phaseZ, azimuth, elevation],
            pageIndex: (phaseX + phaseZ + azimuth + elevation) & 1,
            flags: CERTIFIED_FLAGS,
          });
        }
      }
    }
  }
  for (let phaseX = 0; phaseX < 2; phaseX++) {
    for (let phaseZ = 0; phaseZ < 2; phaseZ++) {
      for (let azimuth = 0; azimuth < 2; azimuth++) {
        for (let elevation = 0; elevation < 2; elevation++) {
          leaves.push({
            level: 2,
            origin: [phaseX, phaseZ, azimuth, elevation],
            pageIndex: (phaseX ^ phaseZ ^ azimuth ^ elevation) & 1,
            flags: CERTIFIED_FLAGS,
          });
        }
      }
    }
  }
  return leaves.reverse();
}

function definition(): SparseExteriorClosureDefinition {
  return {
    tile: { originX: -0.26, originZ: -0.26, sizeX: 0.52, sizeZ: 0.52, topH: 1.176 },
    direction: {
      azimuthMin: -Math.PI,
      azimuthMax: Math.PI,
      elevationMin: 5 * Math.PI / 180,
      elevationMax: 89 * Math.PI / 180,
      azimuthWrap: true,
    },
    horizon: 155,
    candidateCapacity: 4,
    maxSubdivisionDepth: 2,
    predicate: {
      kind: SparseClosurePredicateKind.BakedEligibleSuccessor,
      revision: 7,
      sha256: '11'.repeat(32),
    },
    sourceSha256: 'ab'.repeat(32),
    exteriorLeaves: adaptiveLeaves(),
    exteriorPages: [
      {
        flags: 3,
        candidates: [
          { profileId: 2, sourceTriangleId: 900_001, copyX: 4_096, copyZ: -8_192, predicateToken: 91 },
          { profileId: 2, sourceTriangleId: 17, copyX: -2_048, copyZ: 3_072, predicateToken: 22 },
        ],
      },
      {
        flags: 5,
        candidates: [
          // Same source triangle as page zero proves compact remap deduplication.
          { profileId: 2, sourceTriangleId: 17, copyX: 32_000, copyZ: -31_999, predicateToken: 23 },
          { profileId: 9, sourceTriangleId: 5, copyX: -512, copyZ: 768, predicateToken: 44 },
        ],
      },
    ],
  };
}

function gates(): SparseExteriorClosureByteGates {
  return {
    maxTotalBytes: 1 << 20,
    maxExteriorLeafBytes: 1 << 18,
    maxExteriorPageBytes: 1 << 18,
    maxTriangleRemapBytes: 1 << 16,
    maxExteriorLeaves: 1_024,
    maxExteriorPages: 64,
    maxTriangleRemapEntries: 1_024,
    maxCandidateCapacity: 8,
    maxSubdivisionDepth: 8,
    maxAbsCopyOffset: 100_000,
  };
}

test('v5 round-trips adaptive 4D leaves, fixed K pages, remap, and wide signed copies', () => {
  const source = definition();
  const packed = packSparseExteriorClosureV5(source, gates());
  const parsed = parseSparseExteriorClosureV5(packed.bytes, gates());

  assert.equal(parsed.sha256, packed.sha256);
  assert.equal(parsed.header.version, 5);
  assert.equal(parsed.header.candidateCapacity, 4);
  assert.equal(parsed.header.maxSubdivisionDepth, 2);
  assert.equal(parsed.header.exteriorLeafCount, 31);
  assert.equal(parsed.header.exteriorPageCount, 2);
  assert.equal(parsed.header.insideLeafCount, 0);
  assert.equal(parsed.header.insidePageCount, 0);
  assert.equal(parsed.header.insideCandidateCapacity, 0);
  assert.equal(parsed.header.insideLeafOffset, 0);
  assert.equal(parsed.header.insidePageOffset, 0);
  assert.equal(parsed.header.predicate.kind, SparseClosurePredicateKind.BakedEligibleSuccessor);
  assert.equal(parsed.header.predicate.sha256, '11'.repeat(32));
  assert.equal(parsed.header.sourceSha256, 'ab'.repeat(32));

  assert.deepEqual(parsed.triangleRemap, [
    { profileId: 2, sourceTriangleId: 17 },
    { profileId: 2, sourceTriangleId: 900_001 },
    { profileId: 9, sourceTriangleId: 5 },
  ]);
  assert.equal(parsed.exteriorPages[0]!.candidates.length, 2);
  assert.equal(parsed.exteriorPages[1]!.candidates.length, 2);
  assert.deepEqual(parsed.exteriorPages[1]!.candidates[0], {
    triangleRemapIndex: 0,
    copyX: 32_000,
    copyZ: -31_999,
    predicateToken: 23,
  });
  assert.ok(
    parsed.exteriorPages.some((page) => page.candidates.some((candidate) => Math.abs(candidate.copyX) > 16)),
    'copy range must materially exceed v4 signed five-bit ownership',
  );

  const pageRecordBytes = SPARSE_EXTERIOR_PAGE_HEADER_BYTES
    + source.candidateCapacity * SPARSE_EXTERIOR_CANDIDATE_BYTES;
  assert.equal(packed.ledger.headerBytes, SPARSE_EXTERIOR_HEADER_BYTES);
  assert.equal(packed.ledger.exteriorLeafBytes, 31 * SPARSE_EXTERIOR_LEAF_BYTES);
  assert.equal(packed.ledger.exteriorPageBytes, 2 * pageRecordBytes);
  assert.equal(packed.ledger.triangleRemapBytes, 3 * SPARSE_EXTERIOR_TRIANGLE_REMAP_BYTES);
  assert.equal(packed.ledger.totalBytes, packed.bytes.byteLength);
  assert.equal(measureSparseExteriorClosureV5(source).totalBytes, packed.bytes.byteLength);
});

test('packing is deterministic and candidate order is canonicalized', () => {
  const firstDefinition = definition();
  const secondDefinition = definition();
  secondDefinition.exteriorPages = secondDefinition.exteriorPages.map((page) => ({
    ...page,
    candidates: [...page.candidates].reverse(),
  }));
  const first = packSparseExteriorClosureV5(firstDefinition, gates());
  const second = packSparseExteriorClosureV5(secondDefinition, gates());
  assert.equal(first.sha256, second.sha256);
  assert.deepEqual(first.bytes, second.bytes);
});

test('adaptive leaf validation rejects gaps, overlaps, and uncertified predicate leaves', () => {
  const gap = definition();
  gap.exteriorLeaves = [{
    level: 1,
    origin: [0, 0, 0, 0],
    pageIndex: 0,
    flags: CERTIFIED_FLAGS,
  }];
  assert.throws(() => packSparseExteriorClosureV5(gap, gates()), /domain gap/);

  const overlap = definition();
  overlap.exteriorLeaves = [
    { level: 0, origin: [0, 0, 0, 0], pageIndex: 0, flags: CERTIFIED_FLAGS },
    { level: 1, origin: [0, 0, 0, 0], pageIndex: 1, flags: CERTIFIED_FLAGS },
  ];
  assert.throws(() => packSparseExteriorClosureV5(overlap, gates()), /overlap/);

  const unfiltered = definition();
  unfiltered.exteriorLeaves = unfiltered.exteriorLeaves.map((leaf) => ({
    ...leaf,
    flags: leaf.flags & ~EXTERIOR_LEAF_FLAG_PREDICATE_FILTERED_SUCCESSOR,
  }));
  assert.throws(() => packSparseExteriorClosureV5(unfiltered, gates()), /predicate-filtered successor/);
});

test('fixed K and every independent memory/copy gate fail closed before allocation', () => {
  const source = definition();
  const ledger = measureSparseExteriorClosureV5(source);

  const overK = definition();
  overK.candidateCapacity = 1;
  assert.throws(() => packSparseExteriorClosureV5(overK, gates()), /exceeds fixed candidate capacity/);

  const checks: Array<[keyof SparseExteriorClosureByteGates, number, RegExp]> = [
    ['maxTotalBytes', ledger.totalBytes - 1, /total closure bytes/],
    ['maxExteriorLeafBytes', ledger.exteriorLeafBytes - 1, /exterior leaf bytes/],
    ['maxExteriorPageBytes', ledger.exteriorPageBytes - 1, /exterior page bytes/],
    ['maxTriangleRemapBytes', ledger.triangleRemapBytes - 1, /triangle remap bytes/],
    ['maxExteriorLeaves', source.exteriorLeaves.length - 1, /exterior leaf count/],
    ['maxExteriorPages', source.exteriorPages.length - 1, /exterior page count/],
    ['maxTriangleRemapEntries', 2, /triangle remap count/],
    ['maxCandidateCapacity', source.candidateCapacity - 1, /hard K gate/],
    ['maxSubdivisionDepth', source.maxSubdivisionDepth - 1, /subdivision depth/],
    ['maxAbsCopyOffset', 31_999, /signed periodic copy offset/],
  ];
  for (const [key, value, message] of checks) {
    const constrained = { ...gates(), [key]: value };
    assert.throws(() => packSparseExteriorClosureV5(source, constrained), message, key);
  }
});

test('parser rejects truncation and nonzero reserved inside-successor directory entries', () => {
  const packed = packSparseExteriorClosureV5(definition(), gates());
  assert.throws(
    () => parseSparseExteriorClosureV5(packed.bytes.subarray(0, packed.bytes.length - 16)),
    /table offsets or payload length/,
  );

  const inside = packed.bytes.slice();
  new DataView(inside.buffer).setUint32(104, 1, true);
  assert.throws(() => parseSparseExteriorClosureV5(inside), /reserves but does not yet encode inside successor pages/);
});
