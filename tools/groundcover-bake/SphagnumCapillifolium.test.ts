import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import {
  makeSphagnumCapillifoliumFixture,
  SPHAGNUM_CAPILLIFOLIUM_PROFILE_ID,
  SPHAGNUM_CAPILLIFOLIUM_SPECIES,
  type SphagnumBranchRecipe,
} from './SphagnumCapillifolium';

function meshBinarySha256(mesh: ReturnType<typeof makeSphagnumCapillifoliumFixture>['mesh']): string {
  const digest = createHash('sha256');
  for (const values of [
    Float64Array.from(mesh.positions),
    Float64Array.from(mesh.normals),
    Uint32Array.from(mesh.indices),
  ]) digest.update(new Uint8Array(values.buffer));
  return digest.digest('hex');
}

test('S. capillifolium fixture is deterministic and has the audited palette identity', () => {
  const first = makeSphagnumCapillifoliumFixture();
  const second = makeSphagnumCapillifoliumFixture();
  assert.equal(SPHAGNUM_CAPILLIFOLIUM_PROFILE_ID, 5);
  assert.equal(SPHAGNUM_CAPILLIFOLIUM_SPECIES, 'Sphagnum capillifolium');
  assert.equal(first.generator, 'original-procedural-sphagnum-capillifolium-vegetative-v1');
  assert.deepEqual(first, second);
  assert.equal(first.mesh.positions.length / 3, 25_397);
  assert.equal(first.mesh.indices.length / 3, 41_184);
  assert.equal(first.carpetVertexCount, 1_089);
  assert.equal(first.carpetTriangleCount, 2_048);
  assert.equal(first.capitula.length, 88);
  assert.equal(meshBinarySha256(first.mesh), 'ec8612e197dcc9a801c6146059cf8f85b7549ef33ac189252e91c8efbf01caa7');
});

test('primitive recipes deterministically and exactly disposition every source triangle', () => {
  const fixture = makeSphagnumCapillifoliumFixture();
  const triangleCount = fixture.mesh.indices.length / 3;
  const disposition = new Int32Array(triangleCount);
  const stableIds = new Set<string>();
  let expectedTriangleStart = 0;

  assert.equal(fixture.primitiveRecipes.length, 1_405);
  for (const [primitiveId, recipe] of fixture.primitiveRecipes.entries()) {
    assert.equal(recipe.primitiveId, primitiveId);
    assert.equal(recipe.sourceTriangleStart, expectedTriangleStart);
    assert.ok(recipe.sourceTriangleCount > 0);
    assert.ok(recipe.emittedVertexCount > 0);
    assert.ok(recipe.rootSemantics.length > 0);
    assert.ok(recipe.capSemantics.length > 0);
    assert.equal(recipe.owner, 'estonia-native/bryophyte/sphagnum-capillifolium');
    assert.ok(!stableIds.has(recipe.stableId));
    stableIds.add(recipe.stableId);
    for (
      let triangle = recipe.sourceTriangleStart;
      triangle < recipe.sourceTriangleStart + recipe.sourceTriangleCount;
      triangle++
    ) {
      assert.equal(disposition[triangle], 0);
      disposition[triangle]++;
    }
    expectedTriangleStart += recipe.sourceTriangleCount;
  }
  assert.equal(expectedTriangleStart, triangleCount);
  assert.ok(disposition.every((owners) => owners === 1));

  const carpet = fixture.primitiveRecipes[0]!;
  assert.equal(carpet.kind, 'carpet-support');
  if (carpet.kind !== 'carpet-support') assert.fail('first recipe must be the carpet support');
  assert.equal(carpet.sourceTriangleCount, fixture.carpetTriangleCount);
  assert.equal(carpet.controlPoints.length, fixture.carpetVertexCount);
  assert.equal(carpet.disposition, 'crisp');
  assert.equal(carpet.classEAxisCandidacy, 'not-an-axis');

  for (let capitulumIndex = 0; capitulumIndex < fixture.capitula.length; capitulumIndex++) {
    const capitulum = fixture.capitula[capitulumIndex]!;
    const recipes = fixture.primitiveRecipes.filter((recipe) => recipe.capitulumIndex === capitulumIndex);
    assert.equal(recipes.filter((recipe) => recipe.kind === 'stem').length, 1);
    assert.equal(recipes.filter((recipe) => recipe.kind === 'core').length, 1);
    assert.equal(recipes.filter((recipe) => recipe.kind === 'primary-branch').length, capitulum.primaryBranches);
    assert.equal(recipes.filter((recipe) => recipe.kind === 'fork-branch').length, capitulum.forkedBranches);
  }

  const stems = fixture.primitiveRecipes.filter((recipe) => recipe.kind === 'stem');
  const cores = fixture.primitiveRecipes.filter((recipe) => recipe.kind === 'core');
  const branches = fixture.primitiveRecipes.filter((recipe): recipe is SphagnumBranchRecipe =>
    recipe.kind === 'primary-branch' || recipe.kind === 'fork-branch');
  assert.equal(stems.length, 88);
  assert.equal(cores.length, 88);
  assert.equal(branches.length, 1_228);
  assert.ok(stems.every((recipe) => recipe.sourceTriangleCount === 24));
  assert.ok(cores.every((recipe) => recipe.sourceTriangleCount === 30));
  assert.ok(branches.every((recipe) => recipe.sourceTriangleCount === 28));
  assert.ok(branches.every((recipe) => recipe.disposition === 'medium-candidate'));
  assert.ok(branches.every((recipe) => recipe.classEAxisCandidacy === 'rejected-near-horizontal-curved-axis'));
  assert.ok(branches.every((recipe) => recipe.centerline.length === 5));
  assert.ok(branches.every((recipe) => recipe.halfWidths.length === recipe.centerline.length));
  assert.ok(branches.every((recipe) => recipe.halfThicknesses.length === recipe.centerline.length));
  assert.ok(branches.every((recipe) => recipe.kind !== 'fork-branch' || recipe.parentPrimaryBranchIndex !== null));
});

test('carpet is one connected indexed surface with exactly matching periodic seams', () => {
  const fixture = makeSphagnumCapillifoliumFixture();
  const adjacency: number[][] = Array.from({ length: fixture.carpetVertexCount }, () => []);
  for (let triangle = 0; triangle < fixture.carpetTriangleCount; triangle++) {
    const offset = triangle * 3;
    const a = fixture.mesh.indices[offset] as number;
    const b = fixture.mesh.indices[offset + 1] as number;
    const c = fixture.mesh.indices[offset + 2] as number;
    assert.ok(a < fixture.carpetVertexCount && b < fixture.carpetVertexCount && c < fixture.carpetVertexCount);
    adjacency[a]!.push(b, c);
    adjacency[b]!.push(a, c);
    adjacency[c]!.push(a, b);
  }
  const seen = new Set<number>([0]);
  const queue = [0];
  while (queue.length > 0) {
    const vertex = queue.pop() as number;
    for (const neighbor of adjacency[vertex]!) {
      if (!seen.has(neighbor)) {
        seen.add(neighbor);
        queue.push(neighbor);
      }
    }
  }
  assert.equal(seen.size, fixture.carpetVertexCount);

  const stride = Math.round(Math.sqrt(fixture.carpetVertexCount));
  assert.equal(stride * stride, fixture.carpetVertexCount);
  const component = (vertex: number, array: number[], axis: number): number => array[vertex * 3 + axis] as number;
  for (let offset = 0; offset < stride; offset++) {
    const pairs = [[offset * stride, offset * stride + stride - 1], [offset, (stride - 1) * stride + offset]];
    for (const [a, b] of pairs) {
      assert.ok(Math.abs(component(a!, fixture.mesh.positions, 1) - component(b!, fixture.mesh.positions, 1)) < 1e-12);
      for (let axis = 0; axis < 3; axis++) {
        assert.ok(Math.abs(component(a!, fixture.mesh.normals, axis) - component(b!, fixture.mesh.normals, axis)) < 1e-10);
      }
    }
  }
});

test('capitula preserve star-like branching and resolvable inter-head negative space', () => {
  const fixture = makeSphagnumCapillifoliumFixture();
  assert.equal(fixture.capitula.length, 88);
  assert.ok(fixture.capitula.every((capitulum) => capitulum.primaryBranches >= 9));
  assert.ok(fixture.capitula.every((capitulum) => capitulum.forkedBranches >= 3));
  const texelPitch = fixture.tile.sizeX / 64;
  assert.ok(Math.min(...fixture.capitula.map((capitulum) => capitulum.radius * 2)) / texelPitch >= 4);

  for (let vertex = 0; vertex < fixture.mesh.normals.length; vertex += 3) {
    const length = Math.hypot(
      fixture.mesh.normals[vertex] as number,
      fixture.mesh.normals[vertex + 1] as number,
      fixture.mesh.normals[vertex + 2] as number,
    );
    assert.ok(Math.abs(length - 1) < 1e-10);
  }
});
