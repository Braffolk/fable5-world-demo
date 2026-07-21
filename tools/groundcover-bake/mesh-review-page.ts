/**
 * Direct authoring-mesh renderer for the ground-cover QA harness.
 *
 * This deliberately bypasses GCRP/GCAR baking and NaniteGrass: every image is a
 * direct WebGPU raster of the source IndexedMesh.  Four cardinal azimuths expose
 * view-dependent modelling failures; detail and fixed-metric framing separate
 * geometric legibility from honest stature.
 */

import {
  AmbientLight,
  Box3,
  BufferAttribute,
  BufferGeometry,
  Color,
  DirectionalLight,
  DoubleSide,
  GridHelper,
  Group,
  HemisphereLight,
  Mesh,
  MeshStandardMaterial,
  OrthographicCamera,
  PlaneGeometry,
  Scene,
  Vector3,
} from 'three';
import { WebGPURenderer } from 'three/webgpu';
type CardinalView = 'north' | 'east' | 'south' | 'west';
type Framing = 'detail' | 'metric' | 'apex';

interface ReviewMesh {
  positions: Float32Array;
  normals: Float32Array;
  colors?: Float32Array;
  indices: Uint32Array;
}

interface ReviewFixture {
  profileId: number;
  species: string;
  generator: string;
  mesh: ReviewMesh;
}

interface FixtureTransport {
  profileId: number;
  species: string;
  generator: string;
  positionsUrl: string;
  normalsUrl: string;
  colorsUrl?: string;
  indicesUrl: string;
}

interface ReviewResult {
  profileId: number;
  species: string;
  generator: string;
  view: CardinalView;
  framing: Framing;
  dimensionsM: { x: number; y: number; z: number };
  vertices: number;
  triangles: number;
}

declare global {
  interface Window {
    __groundCoverMeshReview: {
      ready: boolean;
      error: string | null;
      diagnostics: string[];
      loadBinary: (transport: FixtureTransport) => Promise<void>;
      render: (view: CardinalView, framing: Framing) => Promise<ReviewResult>;
    };
  }
}

const params = new URLSearchParams(location.search);
const pixels = Math.max(256, Math.min(2048, Number(params.get('px') ?? 1024)));
const METRIC_SPAN_M = 1.6;

function geometryFrom(mesh: ReviewMesh): BufferGeometry {
  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new BufferAttribute(mesh.positions, 3));
  geometry.setAttribute('normal', new BufferAttribute(mesh.normals, 3));
  if (mesh.colors) {
    geometry.setAttribute('color', new BufferAttribute(mesh.colors, 3));
  }
  geometry.setIndex(new BufferAttribute(mesh.indices, 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function direction(view: CardinalView): Vector3 {
  switch (view) {
    case 'north': return new Vector3(0, 0, 1);
    case 'east': return new Vector3(1, 0, 0);
    case 'south': return new Vector3(0, 0, -1);
    case 'west': return new Vector3(-1, 0, 0);
  }
}

function niceRulerHeight(height: number): number {
  const target = Math.max(0.05, height * 0.9);
  for (const candidate of [0.1, 0.25, 0.5, 1, 1.5]) {
    if (candidate >= target) return candidate;
  }
  return 2;
}

function addRuler(scene: Scene, x: number, height: number): void {
  const bands = 10;
  const bandH = height / bands;
  const width = Math.max(0.004, height * 0.018);
  for (let band = 0; band < bands; band++) {
    const material = new MeshStandardMaterial({
      color: band % 2 === 0 ? 0x171717 : 0xf4f1e8,
      roughness: 0.85,
      metalness: 0,
    });
    const marker = new Mesh(new PlaneGeometry(width, bandH), material);
    marker.position.set(x, bandH * (band + 0.5), 0);
    scene.add(marker);
  }
}

let renderer: WebGPURenderer;
let camera: OrthographicCamera;
let loadedFixture: ReviewFixture | null = null;
let loadedGeometry: BufferGeometry | null = null;

async function typedArray(url: string, kind: 'float32' | 'uint32'): Promise<Float32Array | Uint32Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`ground-cover mesh review could not fetch ${url}: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  if ((bytes.byteLength & 3) !== 0) throw new Error(`ground-cover mesh review ${url} is not 32-bit aligned`);
  return kind === 'float32' ? new Float32Array(bytes) : new Uint32Array(bytes);
}

async function loadBinary(transport: FixtureTransport): Promise<void> {
  if (!Number.isInteger(transport.profileId) || transport.profileId < 0 || transport.profileId > 11) {
    throw new Error(`ground-cover mesh review profile ${transport.profileId} is outside 0..11`);
  }
  const [positions, normals, colors, indices] = await Promise.all([
    typedArray(transport.positionsUrl, 'float32') as Promise<Float32Array>,
    typedArray(transport.normalsUrl, 'float32') as Promise<Float32Array>,
    transport.colorsUrl
      ? typedArray(transport.colorsUrl, 'float32') as Promise<Float32Array>
      : Promise.resolve(undefined),
    typedArray(transport.indicesUrl, 'uint32') as Promise<Uint32Array>,
  ]);
  if (
    positions.length === 0
    || positions.length !== normals.length
    || (colors !== undefined && colors.length !== positions.length)
  ) throw new Error(`ground-cover mesh review profile ${transport.profileId} has invalid attributes`);
  loadedGeometry?.dispose();
  loadedFixture = {
    profileId: transport.profileId,
    species: transport.species,
    generator: transport.generator,
    mesh: { positions, normals, ...(colors ? { colors } : {}), indices },
  };
  loadedGeometry = geometryFrom(loadedFixture.mesh);
}

async function render(view: CardinalView, framing: Framing): Promise<ReviewResult> {
  if (!loadedFixture) throw new Error('ground-cover mesh review render before load');
  const fixture = loadedFixture;
  if (!loadedGeometry) throw new Error('ground-cover mesh review geometry missing after load');
  const geometry = loadedGeometry;
  const material = new MeshStandardMaterial({
    color: 0xffffff,
    roughness: 0.72,
    metalness: 0,
    side: DoubleSide,
    vertexColors: fixture.mesh.colors !== undefined,
  });
  const plant = new Mesh(geometry, material);
  const sourceBounds = new Box3().setFromObject(plant);
  const sourceSize = sourceBounds.getSize(new Vector3());
  const sourceCenter = sourceBounds.getCenter(new Vector3());
  let apex = new Vector3(sourceCenter.x, Number.NEGATIVE_INFINITY, sourceCenter.z);
  const positions = fixture.mesh.positions;
  for (let index = 0; index < positions.length; index += 3) {
    if (positions[index + 1]! > apex.y) {
      apex = new Vector3(positions[index]!, positions[index + 1]!, positions[index + 2]!);
    }
  }
  const root = new Group();
  plant.position.set(-sourceCenter.x, -sourceBounds.min.y, -sourceCenter.z);
  root.add(plant);

  const scene = new Scene();
  scene.background = new Color(0xb8b4aa);
  scene.add(root);

  const horizontalExtent = Math.max(sourceSize.x, sourceSize.z);
  const detailSpan = Math.max(
    sourceSize.y * 1.24,
    horizontalExtent * 1.42,
    0.09,
  );
  const span = framing === 'metric' ? METRIC_SPAN_M : framing === 'apex' ? 0.30 : detailSpan;
  const target = framing === 'apex'
    ? new Vector3(
      apex.x - sourceCenter.x,
      sourceSize.y - Math.min(0.105, sourceSize.y * 0.12),
      apex.z - sourceCenter.z,
    )
    : new Vector3(0, span * 0.47, 0);
  const groundSpan = Math.max(METRIC_SPAN_M * 1.5, horizontalExtent * 4);
  const ground = new Mesh(
    new PlaneGeometry(groundSpan, groundSpan),
    new MeshStandardMaterial({ color: 0xaaa69c, roughness: 1, metalness: 0 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.001;
  scene.add(ground);
  const grid = new GridHelper(groundSpan, Math.round(groundSpan / 0.1), 0x6f6d68, 0x96928a);
  grid.position.y = 0.0005;
  scene.add(grid);

  if (framing !== 'apex') {
    const rulerHeight = framing === 'metric' ? 1 : niceRulerHeight(sourceSize.y);
    addRuler(scene, Math.min(span * 0.38, horizontalExtent * 0.62 + span * 0.09), rulerHeight);
  }

  const key = new DirectionalLight(0xfff4dd, 3.2);
  key.position.set(-1.6, 2.4, 1.8);
  scene.add(key);
  const fill = new DirectionalLight(0xdce9ff, 1.15);
  fill.position.set(1.7, 1.25, -1.4);
  scene.add(fill);
  scene.add(new HemisphereLight(0xd9e8ff, 0x524d42, 1.0));
  scene.add(new AmbientLight(0xffffff, 0.18));

  const half = span * 0.5;
  camera.left = -half;
  camera.right = half;
  camera.top = half;
  camera.bottom = -half;
  camera.near = 0.01;
  camera.far = 20;
  const eyeDirection = direction(view);
  camera.position.set(
    target.x + eyeDirection.x * 4,
    target.y + span * 0.055,
    target.z + eyeDirection.z * 4,
  );
  camera.up.set(0, 1, 0);
  camera.lookAt(target);
  camera.updateProjectionMatrix();

  renderer.render(scene, camera);
  await new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));

  material.dispose();
  return {
    profileId: fixture.profileId,
    species: fixture.species,
    generator: fixture.generator,
    view,
    framing,
    dimensionsM: { x: sourceSize.x, y: sourceSize.y, z: sourceSize.z },
    vertices: fixture.mesh.positions.length / 3,
    triangles: fixture.mesh.indices.length / 3,
  };
}

async function main(): Promise<void> {
  window.__groundCoverMeshReview = {
    ready: false,
    error: null,
    diagnostics: [],
    loadBinary,
    render,
  };
  try {
    renderer = new WebGPURenderer({ antialias: true });
    await renderer.init();
    renderer.setPixelRatio(1);
    renderer.setSize(pixels, pixels);
    document.body.appendChild(renderer.domElement);
    camera = new OrthographicCamera(-1, 1, 1, -1, 0.01, 20);
    const backend = (renderer as unknown as { backend?: { device?: GPUDevice } }).backend;
    backend?.device?.addEventListener('uncapturederror', (event) => {
      window.__groundCoverMeshReview.diagnostics.push(event.error.message);
    });
    window.__groundCoverMeshReview.ready = true;
  } catch (error) {
    window.__groundCoverMeshReview.error = error instanceof Error
      ? `${error.message}\n${error.stack ?? ''}`
      : String(error);
  }
}

void main();
