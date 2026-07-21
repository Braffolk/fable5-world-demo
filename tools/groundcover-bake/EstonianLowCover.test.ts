import assert from 'node:assert/strict';
import test from 'node:test';
import { makeDirection, validateIndexedMesh } from './ProfileFormat';
import { makePeriodicSlice, periodicAddress } from './PeriodicProfile';
import {
  ESTONIAN_LOW_COVER_PROFILE_IDS,
  makeAllEstonianLowCoverFixtures,
  makeCallunaVulgarisFixture,
  makeCladoniaRangiferinaFixture,
  makeEstonianLowCoverFixture,
  makeMaianthemumBifoliumFixture,
  makeOxalisAcetosellaFixture,
  makePleuroziumSchreberiFixture,
  makeVacciniumMyrtillusFixture,
} from './EstonianLowCover';

test('profile IDs 6-11 have deterministic, distinct botanical identities', () => {
  assert.deepEqual(ESTONIAN_LOW_COVER_PROFILE_IDS, [6, 7, 8, 9, 10, 11]);
  const first = makeAllEstonianLowCoverFixtures();
  const second = makeAllEstonianLowCoverFixtures();
  assert.deepEqual(first, second);
  assert.deepEqual(first.map((fixture) => fixture.species), [
    'Pleurozium schreberi',
    'Cladonia rangiferina',
    'Oxalis acetosella',
    'Maianthemum bifolium',
    'Vaccinium myrtillus',
    'Calluna vulgaris',
  ]);
  assert.equal(new Set(first.map((fixture) => fixture.generator)).size, 6);
  assert.equal(new Set(first.map((fixture) => fixture.morphology.growthForm)).size, 6);
  for (const fixture of first) {
    validateIndexedMesh(fixture.mesh);
    assert.deepEqual(makeEstonianLowCoverFixture(fixture.profileId), fixture);
  }
});

test('all geometry is indexed, edge-sharing, finite, and unit-normalized', () => {
  for (const fixture of makeAllEstonianLowCoverFixtures()) {
    const edgeUse = new Map<string, number>();
    for (let triangle = 0; triangle < fixture.mesh.indices.length; triangle += 3) {
      const vertices = [
        fixture.mesh.indices[triangle] as number,
        fixture.mesh.indices[triangle + 1] as number,
        fixture.mesh.indices[triangle + 2] as number,
      ];
      for (let edge = 0; edge < 3; edge++) {
        const a = vertices[edge]!;
        const b = vertices[(edge + 1) % 3]!;
        const key = a < b ? `${a}:${b}` : `${b}:${a}`;
        edgeUse.set(key, (edgeUse.get(key) ?? 0) + 1);
      }
    }
    const sharedEdges = [...edgeUse.values()].filter((uses) => uses >= 2).length;
    assert.ok(sharedEdges > fixture.mesh.indices.length / 12, `${fixture.species} is not meaningfully edge-connected indexed geometry`);
    for (let index = 0; index < fixture.mesh.normals.length; index += 3) {
      const length = Math.hypot(
        fixture.mesh.normals[index] as number,
        fixture.mesh.normals[index + 1] as number,
        fixture.mesh.normals[index + 2] as number,
      );
      assert.ok(Math.abs(length - 1) < 1e-10, `${fixture.species} has a non-unit normal`);
    }
  }
});

test('Pleurozium is a low simple-pinnate mat with many explicit concave leaves', () => {
  const fixture = makePleuroziumSchreberiFixture();
  assert.equal(fixture.morphology.growthForm, 'pleurocarp-mat');
  assert.equal(fixture.morphology.primaryAxisCount, 24);
  assert.equal(fixture.morphology.branchCount, 96);
  assert.equal(fixture.morphology.leafCount, 480);
  assert.ok(fixture.tile.topH < 0.08);
  assert.ok(fixture.mesh.indices.length / 3 > 5_000);
});

test('Cladonia is a dense terete podetial forest with repeated crown forks', () => {
  const fixture = makeCladoniaRangiferinaFixture();
  assert.equal(fixture.morphology.growthForm, 'fruticose-lichen');
  assert.equal(fixture.morphology.primaryAxisCount, 30);
  assert.ok(fixture.morphology.forkCount >= 100);
  assert.ok(fixture.morphology.branchCount >= 230);
  assert.equal(fixture.morphology.leafCount, 0);
  assert.ok(fixture.tile.topH > 0.07 && fixture.tile.topH < 0.15);
});

test('Oxalis carries exactly three deeply notched leaflets per long petiole', () => {
  const fixture = makeOxalisAcetosellaFixture();
  assert.equal(fixture.morphology.growthForm, 'trifoliate-herb');
  assert.equal(fixture.morphology.shootCount, 18);
  assert.equal(fixture.morphology.leafletCount, fixture.morphology.shootCount * 3);
  assert.equal(fixture.morphology.leafCount, fixture.morphology.leafletCount);
  assert.equal(fixture.morphology.flowerCount, 3);
  assert.ok(fixture.morphology.rhizomeSegments > 0);
});

test('Maianthemum distinguishes one-leaf sterile shoots from two-leaf flowering shoots and terminal racemes', () => {
  const fixture = makeMaianthemumBifoliumFixture();
  assert.equal(fixture.morphology.growthForm, 'two-leaved-herb');
  assert.equal(fixture.morphology.shootCount, 12);
  assert.equal(fixture.morphology.twoLeafShootCount, 8);
  assert.equal(fixture.morphology.leafCount, 20);
  assert.ok(fixture.morphology.flowerCount >= 80);
  assert.ok(fixture.morphology.rhizomeSegments > 0);
});

test('Vaccinium has three-sided axes and alternate serrated ovate leaves', () => {
  const fixture = makeVacciniumMyrtillusFixture();
  assert.equal(fixture.morphology.growthForm, 'dwarf-shrub');
  assert.equal(fixture.morphology.primaryAxisCount, 9);
  assert.equal(fixture.morphology.branchCount, 36);
  assert.equal(fixture.morphology.serratedLeafCount, 252);
  assert.equal(fixture.morphology.leafCount, fixture.morphology.serratedLeafCount);
  assert.equal(fixture.morphology.flowerCount, 0);
});

test('Calluna has dense four-ranked imbricate scale leaves on richly branched shoots', () => {
  const fixture = makeCallunaVulgarisFixture();
  assert.equal(fixture.morphology.growthForm, 'scale-leaved-shrub');
  assert.equal(fixture.morphology.primaryAxisCount, 18);
  assert.equal(fixture.morphology.branchCount, 432);
  assert.equal(fixture.morphology.forkCount, 288);
  assert.equal(fixture.morphology.fourRankScaleNodes, 5_760);
  assert.equal(fixture.morphology.leafCount, fixture.morphology.fourRankScaleNodes * 4);
  assert.equal(fixture.morphology.serratedLeafCount, 0);
});

test('every botanical tile repeats exactly through the canonical periodic address and conservative copy range', () => {
  for (const fixture of makeAllEstonianLowCoverFixtures()) {
    const sample = { x: fixture.tile.sizeX * 0.137, z: fixture.tile.sizeZ * 0.819 };
    const expected = periodicAddress(sample.x, sample.z, fixture.tile);
    for (let ix = -3; ix <= 3; ix++) {
      for (let iz = -3; iz <= 3; iz++) {
        const repeated = periodicAddress(
          sample.x + ix * fixture.tile.sizeX,
          sample.z + iz * fixture.tile.sizeZ,
          fixture.tile,
        );
        assert.ok(Math.abs(repeated.u - expected.u) < 2e-15);
        assert.ok(Math.abs(repeated.v - expected.v) < 2e-15);
      }
    }
    const forward = makePeriodicSlice(fixture.mesh, fixture.tile, makeDirection(31, 15));
    const reverse = makePeriodicSlice(fixture.mesh, fixture.tile, makeDirection(211, 15));
    assert.ok(forward.copyRange.maxX > 0 && reverse.copyRange.minX < 0, `${fixture.species} misses repeated X seam copies`);
    assert.ok(forward.copyRange.maxZ > 0 && reverse.copyRange.minZ < 0, `${fixture.species} misses repeated Z seam copies`);
    assert.ok(forward.copies.length > 1 && reverse.copies.length > 1, `${fixture.species} did not derive periodic source copies`);
  }
});
