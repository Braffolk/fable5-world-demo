import assert from 'node:assert/strict';
import test from 'node:test';
import {
  makeSphagnumCapillifoliumFixture,
  SPHAGNUM_CAPILLIFOLIUM_PROFILE_ID,
  SPHAGNUM_CAPILLIFOLIUM_SPECIES,
} from './SphagnumCapillifolium';

test('S. capillifolium fixture is deterministic and has the audited palette identity', () => {
  const first = makeSphagnumCapillifoliumFixture();
  const second = makeSphagnumCapillifoliumFixture();
  assert.equal(SPHAGNUM_CAPILLIFOLIUM_PROFILE_ID, 5);
  assert.equal(SPHAGNUM_CAPILLIFOLIUM_SPECIES, 'Sphagnum capillifolium');
  assert.equal(first.generator, 'original-procedural-sphagnum-capillifolium-vegetative-v1');
  assert.deepEqual(first, second);
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
