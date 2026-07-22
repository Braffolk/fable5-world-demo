import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { makeDirection } from './ProfileFormat';
import {
  ESTONIAN_GRAMINOID_PROFILE_IDS,
  makeAllEstonianGraminoidFixtures,
  makeEstonianGraminoidFixture,
  type GraminoidBladeRecipe,
  type GraminoidHairFilamentRecipe,
  type GraminoidLanceolateRecipe,
  type GraminoidTubeRecipe,
} from './EstonianGraminoids';
import { makePeriodicSlice, periodicAddress } from './PeriodicProfile';

const ACCEPTED_CALAMAGROSTIS_MESH = {
  sha256: '37b0cf1d33f632b5cde3dd60410ad987eec2e977661d0bc804dd6f2bee7054c0',
  vertices: 2_049_985,
  triangles: 2_171_134,
  primitiveRecipes: 270_541,
} as const;

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
    assert.ok(fixture.primitiveRecipes.length > 0, `${fixture.species} must expose deterministic primitive recipes`);
    assert.ok(fixture.mesh.positions.length / 3 > 1_000);
    assert.ok(fixture.mesh.indices.length / 3 > 1_000);
  }
});

test('accepted Calamagrostis mesh stays byte-identical while recipes partition every source triangle', () => {
  const fixture = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
  assert.equal(fixture.mesh.positions.length / 3, ACCEPTED_CALAMAGROSTIS_MESH.vertices);
  assert.equal(fixture.mesh.indices.length / 3, ACCEPTED_CALAMAGROSTIS_MESH.triangles);
  assert.equal(meshHash(fixture.mesh), ACCEPTED_CALAMAGROSTIS_MESH.sha256);
  assert.equal(fixture.primitiveRecipes.length, ACCEPTED_CALAMAGROSTIS_MESH.primitiveRecipes);

  let triangleCursor = 0;
  let crispTriangles = 0;
  let plumeTriangles = 0;
  for (let index = 0; index < fixture.primitiveRecipes.length; index++) {
    const recipe = fixture.primitiveRecipes[index]!;
    assert.equal(recipe.primitiveId, index, 'primitive IDs must be stable emission-order IDs');
    assert.equal(recipe.sourceTriangleStart, triangleCursor, `primitive ${index} leaves a gap or overlaps its predecessor`);
    assert.ok(Number.isInteger(recipe.sourceTriangleCount) && recipe.sourceTriangleCount >= 0);
    assert.equal(recipe.profileId, fixture.profileId);
    assert.equal(recipe.species, 'Calamagrostis canescens');
    assert.equal(recipe.owner, 'estonia-native/graminoid/calamagrostis-canescens');
    assert.equal(recipe.family, fixture.family);
    assert.ok(recipe.rootSemantics.length > 0 && recipe.endSemantics.length > 0 && recipe.capSemantics.length > 0);

    const firstIndex = recipe.sourceTriangleStart * 3;
    const endIndex = (recipe.sourceTriangleStart + recipe.sourceTriangleCount) * 3;
    for (const anchor of recipe.sharedAnchorVertices) {
      assert.ok(anchor < recipe.emittedVertexStart, `primitive ${index} shared anchor must predate emitted vertices`);
      let attributed = false;
      for (let source = firstIndex; source < endIndex; source++) {
        if (fixture.mesh.indices[source] === anchor) {
          attributed = true;
          break;
        }
      }
      assert.ok(attributed, `primitive ${index} declares an anchor unused by its attributed triangles`);
    }

    if (recipe.disposition === 'crisp') crispTriangles += recipe.sourceTriangleCount;
    else plumeTriangles += recipe.sourceTriangleCount;
    triangleCursor += recipe.sourceTriangleCount;
  }
  assert.equal(triangleCursor, ACCEPTED_CALAMAGROSTIS_MESH.triangles);
  assert.equal(crispTriangles + plumeTriangles, ACCEPTED_CALAMAGROSTIS_MESH.triangles);
  assert.ok(crispTriangles > 0 && plumeTriangles > 0);
  assert.ok(
    fixture.primitiveRecipes.filter((recipe) => recipe.kind === 'hub')
      .every((recipe) => recipe.sourceTriangleCount === 0 && recipe.sharedAnchorVertices.length === 0),
    'hubs own vertices only; incident root-fan triangles are attributed once to the emitting blade/tube',
  );
});

test('Calamagrostis recipes preserve compiler-critical geometry and crisp/plume disposition', () => {
  const fixture = makeEstonianGraminoidFixture(ESTONIAN_GRAMINOID_PROFILE_IDS.CALAMAGROSTIS_CANESCENS);
  const culm = fixture.primitiveRecipes.find(
    (recipe): recipe is GraminoidTubeRecipe => recipe.kind === 'tube' && recipe.role === 'culm',
  );
  const rhizome = fixture.primitiveRecipes.find(
    (recipe): recipe is GraminoidTubeRecipe => recipe.kind === 'rhizome',
  );
  const blade = fixture.primitiveRecipes.find(
    (recipe): recipe is GraminoidBladeRecipe => recipe.kind === 'blade',
  );
  const glume = fixture.primitiveRecipes.find(
    (recipe): recipe is GraminoidLanceolateRecipe => recipe.kind === 'lanceolate-surface' && recipe.role === 'glume',
  );
  const hair = fixture.primitiveRecipes.find(
    (recipe): recipe is GraminoidHairFilamentRecipe => recipe.kind === 'hair-filament',
  );
  assert.ok(culm && culm.centerline.length === culm.radii.length && culm.sides === 7);
  assert.ok(rhizome && rhizome.centerline.length === 3 && rhizome.endAnchorVertex !== null);
  assert.ok(blade && blade.centerline.length === blade.segments + 1 && blade.halfWidths.length === blade.centerline.length);
  assert.ok(glume && glume.centerline.length === glume.halfWidths.length && glume.role === 'glume');
  assert.ok(hair && hair.disposition === 'plume' && hair.centerline.length === 3);
  assert.ok(
    fixture.primitiveRecipes.every((recipe) =>
      recipe.kind === 'hair-filament' || recipe.kind === 'cotton-bristle'
        ? recipe.disposition === 'plume'
        : recipe.disposition === 'crisp'),
  );
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
