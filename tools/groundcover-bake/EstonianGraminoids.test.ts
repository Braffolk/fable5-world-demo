import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { makeDirection } from './ProfileFormat';
import {
  ESTONIAN_GRAMINOID_PROFILE_IDS,
  makeAllEstonianGraminoidFixtures,
  makeEstonianGraminoidFixture,
} from './EstonianGraminoids';
import { makePeriodicSlice, periodicAddress } from './PeriodicProfile';

function meshHash(mesh: ReturnType<typeof makeEstonianGraminoidFixture>['mesh']): string {
  return createHash('sha256').update(JSON.stringify(mesh)).digest('hex');
}

function triangleTwiceArea(
  mesh: ReturnType<typeof makeEstonianGraminoidFixture>['mesh'],
  triangleOffset: number,
): number {
  const point = (index: number): [number, number, number] => [
    mesh.positions[index * 3] as number,
    mesh.positions[index * 3 + 1] as number,
    mesh.positions[index * 3 + 2] as number,
  ];
  const a = point(mesh.indices[triangleOffset] as number);
  const b = point(mesh.indices[triangleOffset + 1] as number);
  const c = point(mesh.indices[triangleOffset + 2] as number);
  const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const cross = [
    ab[1]! * ac[2]! - ab[2]! * ac[1]!,
    ab[2]! * ac[0]! - ab[0]! * ac[2]!,
    ab[0]! * ac[1]! - ab[1]! * ac[0]!,
  ];
  return Math.hypot(cross[0], cross[1], cross[2]);
}

function connectedComponents(mesh: ReturnType<typeof makeEstonianGraminoidFixture>['mesh']): number {
  const vertexCount = mesh.positions.length / 3;
  const adjacency: number[][] = Array.from({ length: vertexCount }, () => []);
  const referenced = new Set<number>();
  for (let index = 0; index < mesh.indices.length; index += 3) {
    assert.ok(triangleTwiceArea(mesh, index) > 1e-14, `triangle ${index / 3} must have real geometric area`);
    const a = mesh.indices[index] as number;
    const b = mesh.indices[index + 1] as number;
    const c = mesh.indices[index + 2] as number;
    adjacency[a]!.push(b, c);
    adjacency[b]!.push(a, c);
    adjacency[c]!.push(a, b);
    referenced.add(a).add(b).add(c);
  }
  assert.equal(referenced.size, vertexCount, 'every authored vertex must participate in indexed geometry');
  const seen = new Set<number>();
  let components = 0;
  for (let start = 0; start < vertexCount; start++) {
    if (seen.has(start)) continue;
    components++;
    seen.add(start);
    const stack = [start];
    while (stack.length > 0) {
      const current = stack.pop() as number;
      for (const neighbor of adjacency[current]!) {
        if (!seen.has(neighbor)) {
          seen.add(neighbor);
          stack.push(neighbor);
        }
      }
    }
  }
  return components;
}

test('five exact native profile identities are deterministic and geometrically distinct', () => {
  const fixtures = makeAllEstonianGraminoidFixtures();
  assert.deepEqual(fixtures.map((fixture) => fixture.profileId), [0, 1, 2, 3, 4]);
  assert.deepEqual(fixtures.map((fixture) => fixture.species), [
    'Agrostis capillaris',
    'Avenella flexuosa',
    'Calamagrostis canescens',
    'Carex cespitosa',
    'Eriophorum vaginatum',
  ]);
  const hashes = fixtures.map((fixture) => meshHash(fixture.mesh));
  assert.equal(new Set(hashes).size, 5, 'no profile may be a renamed copy');
  for (const fixture of fixtures) {
    const repeat = makeEstonianGraminoidFixture(fixture.profileId);
    assert.deepEqual(fixture, repeat);
    assert.ok(fixture.mesh.positions.length / 3 > 1_000);
    assert.ok(fixture.mesh.indices.length / 3 > 1_000);
  }
});

test('growth forms remain connected at the authored plant or rhizome-network scale', () => {
  const agrostis = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.AGROSTIS_CAPILLARIS);
  const avenella = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.AVENELLA_FLEXUOSA);
  const calamagrostis = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
  const carex = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CAREX_CESPITOSA);
  const eriophorum = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.ERIOPHORUM_VAGINATUM);
  assert.equal(connectedComponents(agrostis.mesh), 1, 'elongated rhizomes must join the Agrostis turf');
  assert.equal(connectedComponents(calamagrostis.mesh), 1, 'short rhizomes must join the loose Calamagrostis stand');
  assert.equal(connectedComponents(avenella.mesh), avenella.structure.tufts, 'Avenella must remain discrete caespitose tufts');
  assert.equal(connectedComponents(carex.mesh), carex.structure.tufts, 'Carex must remain separate dense tussocks');
  assert.equal(connectedComponents(eriophorum.mesh), eriophorum.structure.tufts, 'Eriophorum must remain separate compact tussocks');
});

test('every authored triangle has nonzero area, including rhizome endpoint fans', () => {
  for (const fixture of makeAllEstonianGraminoidFixtures()) {
    for (let offset = 0; offset < fixture.mesh.indices.length; offset += 3) {
      assert.ok(
        triangleTwiceArea(fixture.mesh, offset) > 1e-14,
        `${fixture.species} triangle ${offset / 3} is degenerate`,
      );
    }
  }
});

test('species-discriminating reproductive architecture is encoded as geometry', () => {
  const [agrostis, avenella, calamagrostis, carex, eriophorum] = makeAllEstonianGraminoidFixtures();
  assert.ok(agrostis!.structure.rhizomes > 0 && agrostis!.structure.panicleBranches > 0);
  assert.equal(avenella!.structure.rhizomes, 0);
  assert.ok(avenella!.structure.leaves / avenella!.structure.tufts > agrostis!.structure.leaves / agrostis!.structure.tufts);
  assert.ok(calamagrostis!.structure.panicleBranches > agrostis!.structure.panicleBranches);
  assert.equal(carex!.structure.culmCrossSectionSides, 3);
  assert.equal(carex!.structure.maleSpikes, carex!.structure.culms);
  assert.equal(carex!.structure.femaleSpikes, carex!.structure.culms * 2);
  assert.equal(eriophorum!.structure.culmCrossSectionSides, 3);
  assert.equal(eriophorum!.structure.cottonHeads, eriophorum!.structure.culms);
  assert.equal(eriophorum!.structure.cottonBristles, eriophorum!.structure.cottonHeads * 42);
  assert.equal(eriophorum!.structure.spikelets, 0, 'cotton heads are explicit bristle assemblies, not renamed grass spikelets');
});

test('all normals are unit length and exact periodic repetition covers grazing projections', () => {
  for (const fixture of makeAllEstonianGraminoidFixtures()) {
    for (let index = 0; index < fixture.mesh.normals.length; index += 3) {
      const length = Math.hypot(
        fixture.mesh.normals[index] as number,
        fixture.mesh.normals[index + 1] as number,
        fixture.mesh.normals[index + 2] as number,
      );
      assert.ok(Math.abs(length - 1) < 1e-10);
    }
    const slice = makePeriodicSlice(fixture.mesh, fixture.tile, makeDirection(31, 15));
    assert.ok(slice.copies.length > 9, `${fixture.species} must derive a real grazing copy range`);
    for (const center of fixture.tuftCenters) {
      const base = periodicAddress(center.x, center.z, fixture.tile);
      for (let ix = -3; ix <= 3; ix++) {
        for (let iz = -3; iz <= 3; iz++) {
          const repeated = periodicAddress(
            center.x + ix * fixture.tile.sizeX,
            center.z + iz * fixture.tile.sizeZ,
            fixture.tile,
          );
          assert.ok(Math.abs(base.u - repeated.u) < 2e-15);
          assert.ok(Math.abs(base.v - repeated.v) < 2e-15);
        }
      }
    }
  }
});
