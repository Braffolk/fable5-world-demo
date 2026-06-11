/**
 * Octahedral impostor capture (spec §2: ≥ 8×8 views, albedo+normal+depth).
 * Hemi-octahedral view grid; each view renders the asset three times
 * (albedo / world-normal / linear depth) into a small RT, read back and
 * composed into two atlases:
 *   albedoAtlas: rgb sqrt-encoded albedo, a coverage
 *   normalAtlas: rgb world normal (0..1), a linear depth01 along the view ray
 * The Phase-5 impostor material blends the 3 nearest views; the gallery
 * shows fixed-view preview cards for verification.
 */

import {
  DataTexture,
  DoubleSide,
  LinearFilter,
  LinearMipmapLinearFilter,
  Mesh,
  NoColorSpace,
  OrthographicCamera,
  RenderTarget,
  Scene,
  type Texture,
  Vector3,
} from 'three';
import { MeshStandardNodeMaterial, type Renderer } from 'three/webgpu';
import { flipRows } from './FoliageCards';
import {
  attribute,
  cameraPosition,
  normalWorld,
  positionWorld,
  sqrt,
  texture,
  transformNormalToView,
  uv,
  vec2,
  vec3,
} from 'three/tsl';
import type { BufferGeometry } from 'three';
import type { NF, NV2, NV3, NV4 } from '../gpu/TSLTypes';

export const IMPOSTOR_GRID = 8;
export const IMPOSTOR_TILE = 256;

/** hemi-octahedral uv → hemisphere direction (y-up) */
export function hemiOctDecode(u: number, v: number): Vector3 {
  const x = u * 2 - 1;
  const z = v * 2 - 1;
  const px = (x + z) * 0.5;
  const pz = (z - x) * 0.5;
  const y = 1 - Math.abs(px) - Math.abs(pz);
  return new Vector3(px, Math.max(y, 0.02), pz).normalize();
}

export interface ImpostorPart {
  geometry: BufferGeometry;
  kind: 'bark' | 'cards' | 'mesh';
  /** card atlas (cards) */
  atlas?: Texture;
  /** synthesized bark maps (bark) */
  barkTex?: { texA: Texture; texB: Texture };
  /** base tint (mesh foliage) */
  color?: { r: number; g: number; b: number };
}

export interface ImpostorAtlas {
  albedo: DataTexture;
  /** rgb = world normal enc, a = linear depth01 along the view ray */
  normalDepth: DataTexture;
  radius: number;
  centerY: number;
}

type PassKind = 'albedo' | 'normal' | 'depth';

function passMaterial(part: ImpostorPart, pass: PassKind, camDist: number, radius: number): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  mat.colorNode = vec3(0);
  mat.side = DoubleSide;
  mat.roughness = 1;
  let alphaTex: NV4 | null = null;
  if (part.kind === 'cards' && part.atlas) {
    alphaTex = texture(part.atlas, uv() as never) as unknown as NV4;
    mat.opacityNode = alphaTex.w;
    mat.alphaTest = 0.32;
  }
  if (pass === 'albedo') {
    if (part.kind === 'cards' && alphaTex) {
      mat.emissiveNode = alphaTex.rgb; // already sqrt-encoded in the atlas
    } else if (part.kind === 'bark' && part.barkTex) {
      const a = texture(part.barkTex.texA, uv() as never) as unknown as NV4;
      mat.emissiveNode = a.rgb; // sqrt-encoded
    } else {
      const c = part.color ?? { r: 0.05, g: 0.1, b: 0.04 };
      const d = attribute('vdata', 'vec4') as unknown as NV4;
      const tinted = vec3(c.r, c.g, c.b).mul(d.w.mul(0.6).add(0.4));
      mat.emissiveNode = sqrt(tinted as unknown as NF) as unknown as NV3;
    }
  } else if (pass === 'normal') {
    mat.emissiveNode = normalWorld.mul(0.5).add(0.5);
  } else {
    // linear depth along the view ray, 0 at near tangent, 1 at far tangent
    const dist = positionWorld.sub(cameraPosition).length();
    const d01 = dist.sub(camDist - radius).div(2 * radius).clamp(0, 1);
    mat.emissiveNode = vec3(d01, d01, d01);
  }
  return mat;
}

export async function captureImpostor(
  renderer: Renderer,
  parts: ImpostorPart[],
  bounds: { centerY: number; radius: number },
): Promise<ImpostorAtlas> {
  const grid = IMPOSTOR_GRID;
  const tile = IMPOSTOR_TILE;
  const atlasRes = grid * tile;
  const albedoPx = new Uint8Array(atlasRes * atlasRes * 4);
  const normalPx = new Uint8Array(atlasRes * atlasRes * 4);

  const center = new Vector3(0, bounds.centerY, 0);
  const camDist = bounds.radius * 2.2;
  const rt = new RenderTarget(tile, tile);
  rt.texture.colorSpace = NoColorSpace;
  const cam = new OrthographicCamera(
    -bounds.radius * 1.04, bounds.radius * 1.04,
    bounds.radius * 1.04, -bounds.radius * 1.04,
    0.1, camDist + bounds.radius * 2,
  );

  const prevTarget = renderer.getRenderTarget();
  renderer.setClearColor(0x000000, 0);

  // build one scene per pass kind (materials differ)
  const passes: PassKind[] = ['albedo', 'normal', 'depth'];
  const scenes = new Map<PassKind, Scene>();
  for (const pk of passes) {
    const sc = new Scene();
    for (const part of parts) {
      const m = new Mesh(part.geometry, passMaterial(part, pk, camDist, bounds.radius));
      m.frustumCulled = false;
      sc.add(m);
    }
    scenes.set(pk, sc);
  }

  for (let gy = 0; gy < grid; gy++) {
    for (let gx = 0; gx < grid; gx++) {
      const dir = hemiOctDecode((gx + 0.5) / grid, (gy + 0.5) / grid);
      cam.position.copy(center).addScaledVector(dir, camDist);
      cam.up.set(0, 1, 0);
      if (Math.abs(dir.y) > 0.985) cam.up.set(0, 0, -1);
      cam.lookAt(center);
      cam.updateMatrixWorld();

      const out: Partial<Record<PassKind, Uint8Array>> = {};
      for (const pk of passes) {
        renderer.setRenderTarget(rt);
        renderer.render(scenes.get(pk) as Scene, cam);
        renderer.setRenderTarget(prevTarget);
        const raw = (await renderer.readRenderTargetPixelsAsync(rt, 0, 0, tile, tile)) as Uint8Array;
        const px = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
        flipRows(px, tile, tile);
        out[pk] = px;
      }
      // compose into atlases: normal.a := depth.r
      const alb = out.albedo as Uint8Array;
      const nrm = out.normal as Uint8Array;
      const dep = out.depth as Uint8Array;
      for (let py = 0; py < tile; py++) {
        const dstRow = ((gy * tile + py) * atlasRes + gx * tile) * 4;
        const srcRow = py * tile * 4;
        albedoPx.set(alb.subarray(srcRow, srcRow + tile * 4), dstRow);
        for (let px = 0; px < tile; px++) {
          const si = srcRow + px * 4;
          const di = dstRow + px * 4;
          normalPx[di] = nrm[si] as number;
          normalPx[di + 1] = nrm[si + 1] as number;
          normalPx[di + 2] = nrm[si + 2] as number;
          // alpha: depth where covered, 0 where background
          normalPx[di + 3] = (alb[si + 3] as number) > 8 ? Math.max(1, dep[si] as number) : 0;
        }
      }
    }
  }
  rt.dispose();

  const mk = (px: Uint8Array): DataTexture => {
    const t = new DataTexture(px, atlasRes, atlasRes);
    t.colorSpace = NoColorSpace;
    t.generateMipmaps = true;
    t.minFilter = LinearMipmapLinearFilter;
    t.magFilter = LinearFilter;
    t.anisotropy = 4;
    t.needsUpdate = true;
    return t;
  };
  return {
    albedo: mk(albedoPx),
    normalDepth: mk(normalPx),
    radius: bounds.radius,
    centerY: bounds.centerY,
  };
}

/**
 * Fixed-view preview card material (gallery verification): albedo × N·L
 * relighting from the captured normals for one tile of the atlas.
 */
export function impostorPreviewMaterial(
  atlas: ImpostorAtlas,
  view: { gx: number; gy: number },
): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const grid = IMPOSTOR_GRID;
  const u = uv() as unknown as NV2;
  const tileUv = u.div(grid).add(vec2(view.gx / grid, view.gy / grid));
  const a = texture(atlas.albedo, tileUv as never) as unknown as NV4;
  const nd = texture(atlas.normalDepth, tileUv as never) as unknown as NV4;
  const albedo = a.rgb.mul(a.rgb);
  const nW = nd.rgb.mul(2).sub(1).normalize();
  mat.colorNode = albedo;
  mat.normalNode = transformNormalToView(nW); // captured world-space normals
  mat.opacityNode = a.w;
  mat.alphaTest = 0.3;
  mat.side = DoubleSide;
  mat.roughness = 0.7;
  return mat;
}
