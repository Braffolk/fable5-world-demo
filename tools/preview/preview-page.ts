/**
 * Browser-side of the offscreen mesh-preview harness (tools/veg-preview.ts).
 *
 * Mirrors the app's render tech: three r184 WebGPURenderer (three/webgpu) with
 * ACES tone mapping, core scene classes from 'three'. It dynamically imports a
 * vegetation-mesh module (served by Vite), calls its `buildPreview(rng)` export
 * to get a THREE.Object3D, drops it into a neutral studio (grey background, soft
 * key+fill+hemi light, a faint ground plane + a 1 m banded scale stick), and
 * exposes `window.__preview` so the Node harness can pick a camera view and
 * screenshot the canvas. Rendering happens in the browser so the mesh module's
 * own materials (plain three or TSL NodeMaterials) render faithfully — no
 * geometry/material serialization across the process boundary.
 */

import {
  ACESFilmicToneMapping,
  AmbientLight,
  Box3,
  BoxGeometry,
  Color,
  DirectionalLight,
  GridHelper,
  Group,
  HemisphereLight,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  Sphere,
  Vector3,
} from 'three';
import { WorldSeed } from '../../src/core/Seed';

declare global {
  interface Window {
    __preview: {
      ready: boolean;
      error: string | null;
      dims: { x: number; y: number; z: number } | null;
      tris: number;
      views: readonly string[];
      render: (view: string) => Promise<void>;
    };
  }
}

const DEG = Math.PI / 180;
const VIEWS = ['front', 'threeq', 'side', 'top'] as const;
type ViewName = (typeof VIEWS)[number];

function qp(name: string, fallback: string): string {
  const v = new URLSearchParams(location.search).get(name);
  return v === null || v === '' ? fallback : v;
}

/** Camera direction (unit vector FROM target TO camera) + up, per view. */
function viewDir(view: ViewName): { dir: Vector3; up: Vector3 } {
  const up = new Vector3(0, 1, 0);
  switch (view) {
    case 'front': {
      // eye near ground level, looking slightly DOWN — the natural angle on a ~0.3 m plant
      const e = 9 * DEG;
      return { dir: new Vector3(0, Math.sin(e), Math.cos(e)).normalize(), up };
    }
    case 'threeq': {
      const az = 45 * DEG;
      const e = 22 * DEG;
      return {
        dir: new Vector3(Math.sin(az) * Math.cos(e), Math.sin(e), Math.cos(az) * Math.cos(e)).normalize(),
        up,
      };
    }
    case 'side': {
      const e = 8 * DEG;
      return { dir: new Vector3(Math.cos(e), Math.sin(e), 0).normalize(), up };
    }
    case 'top':
      // straight down; up must not be parallel to the view direction
      return { dir: new Vector3(0, 1, 0), up: new Vector3(0, 0, -1) };
  }
}

let renderer: import('three/webgpu').WebGPURenderer;
let scene: Scene;
let camera: PerspectiveCamera;
let target = new Vector3();
let sphere = new Sphere();

/** Distance so the object's bounding sphere fills ~70% of the smaller frame axis. */
function fitDistance(radius: number): number {
  const fovV = camera.fov * DEG;
  const halfV = fovV / 2;
  const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
  const distV = radius / Math.sin(halfV);
  const distH = radius / Math.sin(halfH);
  return Math.max(distV, distH) / 0.7;
}

async function renderView(view: string): Promise<void> {
  const v = (VIEWS as readonly string[]).includes(view) ? (view as ViewName) : 'front';
  const { dir, up } = viewDir(v);
  const dist = fitDistance(Math.max(sphere.radius, 1e-3));
  camera.up.copy(up);
  camera.position.copy(target).addScaledVector(dir, dist);
  camera.lookAt(target);
  camera.near = Math.max(dist - sphere.radius * 3, 0.001);
  camera.far = dist + sphere.radius * 3 + 10;
  camera.updateProjectionMatrix();
  await renderer.renderAsync(scene, camera);
  // let the presented frame settle before the screenshot
  await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
}

/** Vertical 1 m ruler with 0.1 m alternating light/dark bands — reads both 10 cm & 1 m. */
function buildScaleStick(xOffset: number): Object3D {
  const g = new Group();
  const bands = 10;
  const bandH = 0.1;
  const w = 0.018;
  for (let i = 0; i < bands; i++) {
    const mat = new MeshStandardMaterial({
      color: i % 2 === 0 ? 0x1a1a1a : 0xf2f2f2,
      roughness: 0.85,
      metalness: 0,
    });
    const m = new Mesh(new BoxGeometry(w, bandH, w), mat);
    m.position.set(xOffset, bandH * (i + 0.5), 0);
    g.add(m);
  }
  return g;
}

function countTris(obj: Object3D): number {
  let tris = 0;
  obj.traverse((o) => {
    const mesh = o as Mesh;
    if (mesh.isMesh && mesh.geometry) {
      const geo = mesh.geometry;
      const idx = geo.getIndex();
      const pos = geo.getAttribute('position');
      if (idx) tris += idx.count / 3;
      else if (pos) tris += pos.count / 3;
    }
  });
  return Math.round(tris);
}

async function main(): Promise<void> {
  window.__preview = {
    ready: false,
    error: null,
    dims: null,
    tris: 0,
    views: VIEWS,
    render: renderView,
  };

  try {
    const modPath = qp('mod', '');
    const expName = qp('exp', 'buildPreview');
    const seedN = Number(qp('seed', '1'));
    const stream = qp('stream', 'preview');
    if (!modPath) throw new Error('missing ?mod= (served module path, e.g. /src/vegetation/bog/CottonGrass.ts)');

    // dynamic import of the Vite-served mesh module
    const mod = (await import(/* @vite-ignore */ modPath)) as Record<string, unknown>;
    const fn = mod[expName];
    if (typeof fn !== 'function') {
      throw new Error(`export '${expName}' not found or not a function in ${modPath}`);
    }
    const rng = new WorldSeed(seedN).rng(stream);
    const objRaw = (fn as (r: ReturnType<WorldSeed['rng']>) => unknown)(rng);
    if (!(objRaw instanceof Object3D)) {
      throw new Error(`export '${expName}' did not return a THREE.Object3D`);
    }
    const obj = objRaw;

    // --- studio ------------------------------------------------------------
    scene = new Scene();
    scene.background = new Color(0x8c8c8c);

    // recenter: sit the object base on y=0, center it in x/z
    const box = new Box3().setFromObject(obj);
    const size = new Vector3();
    const center = new Vector3();
    box.getSize(size);
    box.getCenter(center);
    const wrap = new Group();
    obj.position.set(-center.x, -box.min.y, -center.z);
    wrap.add(obj);
    scene.add(wrap);

    window.__preview.dims = { x: size.x, y: size.y, z: size.z };
    window.__preview.tris = countTris(obj);

    // camera target = middle of the object's height above the ground
    target = new Vector3(0, size.y / 2, 0);
    const radiusXZ = 0.5 * Math.hypot(size.x, size.z);
    sphere = new Sphere(target.clone(), 0.5 * Math.max(size.x, size.y, size.z, 0.05));

    // faint ground plane + 0.1 m grid
    const groundSpan = Math.max(2, radiusXZ * 6 + 1);
    const ground = new Mesh(
      new PlaneGeometry(groundSpan, groundSpan),
      new MeshStandardMaterial({ color: 0x9a9a9a, roughness: 1, metalness: 0 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.0005;
    scene.add(ground);
    const divisions = Math.max(2, Math.round(groundSpan / 0.1));
    const grid = new GridHelper(groundSpan, divisions, 0x666666, 0x808080);
    grid.position.y = 0.0005;
    scene.add(grid);

    // 1 m banded scale stick just to the +X side of the plant
    scene.add(buildScaleStick(radiusXZ + 0.12));

    // --- lighting (soft studio; ACES to match the app) ---------------------
    const key = new DirectionalLight(0xfff4e6, 2.6);
    key.position.set(-1.2, 2.0, 1.4);
    scene.add(key);
    const fill = new DirectionalLight(0xdfe8ff, 1.0);
    fill.position.set(1.6, 1.0, -1.2);
    scene.add(fill);
    scene.add(new HemisphereLight(0xbfd4ff, 0x555044, 0.8));
    scene.add(new AmbientLight(0xffffff, 0.25));

    // --- renderer (WebGPU + TSL, mirroring the app) ------------------------
    const size2 = Number(qp('px', '1024'));
    const { WebGPURenderer } = await import('three/webgpu');
    renderer = new WebGPURenderer({ antialias: true });
    await renderer.init();
    renderer.setPixelRatio(1);
    renderer.setSize(size2, size2);
    renderer.toneMapping = ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    const canvas = renderer.domElement;
    canvas.style.width = `${size2}px`;
    canvas.style.height = `${size2}px`;
    document.getElementById('stage')!.appendChild(canvas);

    camera = new PerspectiveCamera(35, 1, 0.01, 100);

    await renderView('front');
    window.__preview.ready = true;
  } catch (err) {
    window.__preview.error = err instanceof Error ? `${err.message}\n${err.stack ?? ''}` : String(err);
    // eslint-disable-next-line no-console
    console.error('[veg-preview:page]', window.__preview.error);
  }
}

void main();
