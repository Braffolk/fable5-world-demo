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
  type BufferGeometry,
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
      /** wind mode (?wind=1): filmstrip metadata + per-cell displaced render. */
      wind: {
        rows: readonly { label: string; h0: number; freq: number }[];
        frames: number;
        render: (row: number, frame: number) => Promise<void>;
      } | null;
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

// ─────────────────────────── wind (verification) ─────────────────────────────
// A CPU per-vertex transcription of the ENGINE sway (NaniteFetch.windOffset +
// makeCtx leaf/trunk block). Drives ONLY off vdata.y (flex) — the in-world truth
// (vdata.z phase is dead; phase is per-instance). Fixed per-instance block:
// e=1, s=0.45, g/gL canned, one hashed instPhase; per-vertex prof/flex exactly as
// the shader writes it. Reproduces the stuck-head/stretch on the current meshes and
// (row 2 / --ab) previews the crown with shrub params (h0 0.9 / freq 1.8, issue 1b).
const WIND_DIR = { x: 0.78, y: 0.63 }; // ≈ windU.dir (already ~unit)
// windU.strength default is 0.45; the preview exaggerates it (diagnostic gain) so a
// flex discontinuity — the head sitting a few cm off its culm tip in EVERY frame,
// growing with s² via the lean term — is unmistakable at plant scale. The FORMULA is
// verbatim; only the input strength is scaled, exactly as ?wind=N does in-world.
const WIND_S = 0.45 * 1.6; // diagnostic strength
const WIND_G = 0.7; // canned gust g∈[0,1]
const WIND_GL = 0.55; // canned lagged gust
const WIND_INST_PHASE = 0.37; // one representative hashed per-instance phase
const WIND_FRAMES = 5; // time samples across ~one natural period

interface WindMesh {
  geo: BufferGeometry;
  orig: Float32Array; // original positions (xyz stride 3)
  flex: Float32Array; // per-vertex flex (vdata.y)
}
let windMeshes: WindMesh[] = [];
let windRows: { label: string; h0: number; freq: number }[] = [];

/** per-instance scalars for one (h0, freq) channel config (S=1, e=1, near). */
function windInst(freq: number): { leanBase: number; swayABase: number; branchBase: number; natW: number; ph: number } {
  const eks = 1;
  const leanBase = WIND_S * WIND_S * (0.9 * WIND_G + 0.5) * eks * 1.1;
  const swayABase = WIND_S * (0.75 * WIND_G + 0.25) * eks * 0.5;
  const branchBase = (WIND_GL - 0.45) * 1 * eks * 0.55;
  const fJit = (WIND_INST_PHASE * 7.31) % 1;
  const natW = (0.15 + 0.3 * fJit) * (Math.PI * 2) * freq; // /sqrt(max(S,.25)) = /1 (S=1)
  const ph = WIND_INST_PHASE * Math.PI * 2;
  return { leanBase, swayABase, branchBase, natW, ph };
}

/** displace every wind mesh into (row, frame) and render from the fixed side view. */
async function renderWind(row: number, frame: number): Promise<void> {
  const cfg = windRows[row] as { label: string; h0: number; freq: number };
  const inst = windInst(cfg.freq);
  const period = (Math.PI * 2) / Math.max(inst.natW, 1e-4);
  const t = (frame / (WIND_FRAMES - 1)) * period;
  const swayPhase = Math.sin(t * inst.natW + inst.ph);
  const swayXPhase = Math.sin(t * inst.natW * 1.31 + inst.ph * 1.7);
  for (const wm of windMeshes) {
    const pos = wm.geo.getAttribute('position');
    const arr = pos.array as Float32Array;
    for (let i = 0; i < wm.flex.length; i++) {
      const px = wm.orig[i * 3] as number;
      const py = wm.orig[i * 3 + 1] as number;
      const pz = wm.orig[i * 3 + 2] as number;
      const flex = wm.flex[i] as number;
      const localY = py; // S=1, base at y≈0
      const yn = localY / (localY + cfg.h0);
      const prof = Math.min(yn * yn * 1.7 + flex * 0.3, 1.6);
      const swayA = inst.swayABase * prof;
      const sway = swayPhase * swayA;
      const swayX = swayXPhase * swayA * 0.45;
      const along = inst.leanBase * prof + sway + inst.branchBase * flex;
      const dy = -0.2 * flex * (Math.abs(along) + Math.abs(swayX));
      arr[i * 3] = px + WIND_DIR.x * along - WIND_DIR.y * swayX;
      arr[i * 3 + 1] = py + dy;
      arr[i * 3 + 2] = pz + WIND_DIR.y * along + WIND_DIR.x * swayX;
    }
    pos.needsUpdate = true;
  }
  await renderer.renderAsync(scene, camera);
  await new Promise<void>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));
}

/** Capture the meshes' rest positions + flex, freeze a stable side camera, and
 *  publish window.__preview.wind. `ab` adds a second row with the shrub params. */
function setupWind(obj: Object3D, size: Vector3, ab: boolean): void {
  windMeshes = [];
  obj.traverse((o) => {
    const mesh = o as Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    const geo = mesh.geometry;
    const pos = geo.getAttribute('position');
    if (!pos) return;
    const orig = (pos.array as Float32Array).slice();
    const vd = geo.getAttribute('vdata');
    const flex = new Float32Array(pos.count);
    if (vd) for (let i = 0; i < pos.count; i++) flex[i] = vd.getY(i);
    windMeshes.push({ geo, orig, flex });
  });
  windRows = [{ label: 'leaf h0=6 f=1 (in-world default)', h0: 6, freq: 1 }];
  if (ab) windRows.push({ label: 'shrub h0=0.9 f=1.8 (1b flag)', h0: 0.9, freq: 1.8 });

  // freeze a stable THREE-QUARTER camera framed on the rest object (no per-frame
  // refit, so a lagging/shearing head is visible as motion against a fixed frame;
  // the wind-dir ≈ (0.78,0.63) motion reads laterally from this angle).
  const { dir, up } = viewDir('threeq');
  const radius = 0.5 * Math.max(size.x, size.y, size.z, 0.05);
  // widen so the exaggerated peak sway stays in frame
  const dist = fitDistance(radius) * 1.6;
  target = new Vector3(0, size.y / 2, 0);
  camera.up.copy(up);
  camera.position.copy(target).addScaledVector(dir, dist);
  camera.lookAt(target);
  camera.near = Math.max(dist - radius * 3, 0.001);
  camera.far = dist + radius * 3 + 10;
  camera.updateProjectionMatrix();

  window.__preview.wind = { rows: windRows, frames: WIND_FRAMES, render: renderWind };
}

async function main(): Promise<void> {
  window.__preview = {
    ready: false,
    error: null,
    dims: null,
    tris: 0,
    views: VIEWS,
    render: renderView,
    wind: null,
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

    // wind verification mode (?wind=1): capture rest state + a stable side camera
    // and publish the filmstrip render hook (Node harness drives the cells).
    if (qp('wind', '') !== '') {
      setupWind(obj, size, qp('windab', '') !== '');
    }

    window.__preview.ready = true;
  } catch (err) {
    window.__preview.error = err instanceof Error ? `${err.message}\n${err.stack ?? ''}` : String(err);
    // eslint-disable-next-line no-console
    console.error('[veg-preview:page]', window.__preview.error);
  }
}

void main();
