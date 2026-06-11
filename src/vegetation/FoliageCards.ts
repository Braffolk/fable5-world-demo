/**
 * Foliage cluster cards (the ez-tree look, zero external assets):
 * a lush twig/spray — dozens of REAL leaf/needle meshes — is rendered ONCE
 * into a per-species 2×2 variant atlas; the tree then places big alpha-tested
 * cards at its foliage anchors. One card = a whole leafy cluster at 2–4 tris,
 * which is where the crown fullness comes from. The same capture rig later
 * feeds branch cards and octahedral impostors.
 *
 * Capture detail: albedo is written sqrt-encoded (8-bit linear murders dark
 * greens), background-dilated on CPU (no dark halos in mips), and decoded in
 * the card material.
 */

import {
  DataTexture,
  DoubleSide,
  LinearFilter,
  LinearMipmapLinearFilter,
  Matrix4,
  Mesh,
  NoColorSpace,
  OrthographicCamera,
  Quaternion,
  RenderTarget,
  Scene,
  Vector3,
} from 'three';
import { MeshStandardNodeMaterial, type Renderer } from 'three/webgpu';
import { attribute, float, mix, smoothstep, sqrt, uv, vec3 } from 'three/tsl';
import type { Rng } from '../core/Seed';
import type { NF, NV2, NV4 } from '../gpu/TSLTypes';
import { buildLeaf, buildNeedleSpray } from './LeafMesh';
import { MeshGrower } from './TubeMesh';
import type { LeafAnchor, SpeciesParams } from './VegTypes';

export const ATLAS_RES = 1024;

const _m = new Matrix4();
const _q = new Quaternion();
const _q2 = new Quaternion();
const X = new Vector3(1, 0, 0);
const Z = new Vector3(0, 0, 1);

/** twig content for one capture tile, centered at (cx, cy), tile size 1 */
function buildTwigTile(
  g: MeshGrower,
  sp: SpeciesParams,
  rng: Rng,
  cx: number,
  cy: number,
): void {
  const fol = sp.foliage;
  if (!fol) return;
  const half = 0.46;
  if (fol.kind === 'needleSpray') {
    const brush = fol.leaf.brush > 0.5;
    const frond = fol.captureStyle === 'frond';
    // main spray growing +y from tile bottom; needle scale in tile units
    const scaleToTile = (2 * half) / (fol.scale[1] * 1.15);
    const leaf = {
      ...fol.leaf,
      len: fol.leaf.len * scaleToTile * (frond ? 1.45 : 1),
      width: fol.leaf.width * scaleToTile * 1.15,
      needleCount: Math.round(fol.leaf.needleCount * (frond ? 1.5 : brush ? 1.4 : 1.2)),
    };
    const sprayLen = fol.scale[1] * scaleToTile * (frond ? 1.3 : 1);
    const sub = frond ? 0 : brush ? 6 : 9;
    for (let i = -1; i < sub; i++) {
      const t = i < 0 ? 0 : (i + 0.6) / sub;
      const along = -half + t * sprayLen * 0.8;
      const side = i < 0 ? 0 : (i % 2 === 0 ? 1 : -1);
      const ang = i < 0 ? 0 : side * (0.75 + rng.float() * 0.65) * (brush ? 1.1 : 1);
      _q.setFromAxisAngle(Z, ang);
      _q2.setFromAxisAngle(X, -Math.PI / 2); // local +z → tile +y
      _q.multiply(_q2);
      const s = i < 0 ? 1 : (0.5 + rng.float() * 0.32) * (1.1 - t * 0.35);
      _m.compose(
        new Vector3(cx + (i < 0 ? 0 : Math.sin(ang) * 0.06), cy + along, 0),
        _q,
        new Vector3(s, s, s),
      );
      buildNeedleSpray(
        g, _m, leaf, sprayLen * (i < 0 ? 1 : s * 0.8), rng,
        rng.float() * 2 - 1, 0.5, rng.float() * 6.28,
        0.72 + rng.float() * 0.28,
      );
    }
  } else {
    // broadleaf cluster: short stem fan + 14–20 leaves facing mostly +z
    const n = 14 + rng.int(7);
    const leafScale = (2 * half) / (fol.leaf.len * 2.1);
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      // leaves arranged along a loose wide fan from bottom
      const spread = 0.5 + t * 0.6;
      const ang = (rng.float() - 0.5) * 3.0 * spread;
      const r = (0.15 + t * 0.85) * half * (0.75 + rng.float() * 0.45);
      const px = cx + Math.sin(ang) * r;
      const py = cy - half * 0.82 + (t * 1.45 + rng.float() * 0.3) * half;
      // orientation: blade up the fan direction, tilted toward camera
      _q.setFromAxisAngle(Z, ang * 0.8 + (rng.float() - 0.5) * 0.5);
      _q2.setFromAxisAngle(X, -Math.PI / 2 + 0.45 + (rng.float() - 0.3) * 0.7);
      _q.multiply(_q2);
      const s = leafScale * (0.75 + rng.float() * 0.5);
      _m.compose(new Vector3(px, py, (rng.float() - 0.5) * 0.05), _q, new Vector3(s, s, s));
      buildLeaf(
        g, _m, fol.leaf,
        rng.float() * 2 - 1, 0.5, rng.float() * 6.28,
        0.65 + rng.float() * 0.35,
      );
    }
  }
}

/** capture material: sqrt-encoded albedo as emissive, no lights involved */
function captureMaterial(sp: SpeciesParams): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const d = attribute('vdata', 'vec4') as unknown as NV4;
  const u = uv() as unknown as NV2;
  const c = sp.foliageColor;
  const base = vec3(c.r, c.g, c.b);
  // per-leaf hue/value variation + simple vein/tip accents painted into uv
  const k = d.x.mul(c.hueVar);
  const warmed = base
    .mul(mix(vec3(1), vec3(1.25, 1.05, 0.5), k.clamp(0, 1)))
    .mul(mix(vec3(1), vec3(0.72, 0.95, 1.2), k.negate().clamp(0, 1)));
  const midrib = smoothstep(0.0, 0.05, u.x.sub(0.5).abs()) as unknown as NF;
  const tipLight = mix(0.92, 1.18, u.y) as unknown as NF;
  let albedo = warmed
    .mul(d.w)
    .mul(midrib.mul(0.18).add(0.82))
    .mul(tipLight);
  if (sp.blossom) {
    // leaves whose hue jitter exceeds the blossom threshold become flowers
    const bl = sp.blossom;
    const isBlossom = d.x.greaterThan(1 - bl.frac * 2).select(float(1), float(0));
    albedo = mix(
      albedo,
      vec3(bl.r, bl.g, bl.b).mul(mix(0.75, 1.15, u.y)).mul(d.w.mul(0.4).add(0.6)),
      isBlossom,
    );
  }
  mat.colorNode = vec3(0);
  // sqrt-encode for 8-bit storage (decoded by squaring in the card material)
  mat.emissiveNode = sqrt(albedo.clamp(0, 1) as unknown as NF) as unknown as ReturnType<typeof vec3>;
  mat.roughness = 1;
  mat.side = DoubleSide;
  return mat;
}

/** WebGPU readbacks are top-left origin; UV space expects v=0 at bottom */
export function flipRows(px: Uint8Array, w: number, h: number): void {
  const row = w * 4;
  const tmp = new Uint8Array(row);
  for (let y = 0; y < h >> 1; y++) {
    const a = y * row;
    const b = (h - 1 - y) * row;
    tmp.set(px.subarray(a, a + row));
    px.copyWithin(a, b, b + row);
    px.set(tmp, b);
  }
}

/** alpha-aware dilation: bleed cluster color into transparent texels */
function dilate(px: Uint8Array, res: number, passes: number): void {
  const idx = (x: number, y: number): number => (y * res + x) * 4;
  for (let p = 0; p < passes; p++) {
    const src = px.slice();
    for (let y = 0; y < res; y++) {
      for (let x = 0; x < res; x++) {
        const i = idx(x, y);
        if ((src[i + 3] as number) > 8) continue;
        let r = 0, g = 0, b = 0, n = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= res || yy >= res) continue;
            const j = idx(xx, yy);
            if ((src[j + 3] as number) > 8) {
              r += src[j] as number; g += src[j + 1] as number; b += src[j + 2] as number;
              n++;
            }
          }
        }
        if (n > 0) {
          px[i] = Math.round(r / n);
          px[i + 1] = Math.round(g / n);
          px[i + 2] = Math.round(b / n);
          px[i + 3] = 9; // mark as filled so later passes spread further
        }
      }
    }
  }
  // dilation alpha markers must not pass the alpha test
  for (let i = 3; i < px.length; i += 4) {
    if ((px[i] as number) <= 9) px[i] = 0;
  }
}

/**
 * Render the species' twig atlas (2×2 variants) and return a mipmapped
 * texture. ~tens of ms once per species; deterministic per seed stream.
 */
export async function captureFoliageAtlas(
  renderer: Renderer,
  sp: SpeciesParams,
  rng: Rng,
): Promise<DataTexture> {
  const scene = new Scene();
  const g = new MeshGrower();
  for (let v = 0; v < 4; v++) {
    buildTwigTile(g, sp, rng.fork(`tile${v}`), (v % 2) - 0.5, Math.floor(v / 2) - 0.5);
  }
  const mesh = new Mesh(g.build(), captureMaterial(sp));
  mesh.frustumCulled = false;
  scene.add(mesh);

  const cam = new OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  cam.position.set(0, 0, 5);
  cam.lookAt(0, 0, 0);

  const rt = new RenderTarget(ATLAS_RES, ATLAS_RES);
  rt.texture.colorSpace = NoColorSpace;

  const prevTarget = renderer.getRenderTarget();
  const prevClearAlpha = renderer.getClearAlpha();
  renderer.setClearColor(0x000000, 0);
  renderer.setRenderTarget(rt);
  renderer.render(scene, cam);
  renderer.setRenderTarget(prevTarget);
  renderer.setClearAlpha(prevClearAlpha);

  const raw = (await renderer.readRenderTargetPixelsAsync(
    rt,
    0, 0, ATLAS_RES, ATLAS_RES,
  )) as Uint8Array;
  const px = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
  flipRows(px, ATLAS_RES, ATLAS_RES);
  dilate(px, ATLAS_RES, 6);
  rt.dispose();

  const tex = new DataTexture(px, ATLAS_RES, ATLAS_RES);
  tex.colorSpace = NoColorSpace;
  tex.generateMipmaps = true;
  tex.minFilter = LinearMipmapLinearFilter;
  tex.magFilter = LinearFilter;
  tex.anisotropy = 4;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Card geometry at every anchor: length axis along the anchor's +z (texture
 * v), 'lying' plane = bough plane (normal +y), 'cross' adds a second plane
 * for volumetric read. vdata carries hue/sway/AO as usual.
 */
export function buildFoliageCards(
  g: MeshGrower,
  anchors: readonly LeafAnchor[],
  opts: { mode: 'lying' | 'cross'; sizeK: number; bend?: number },
  rng: Rng,
): void {
  const right = new Vector3();
  const upL = new Vector3();
  const out = new Vector3();
  const p = new Vector3();
  const rowPos = new Vector3();
  const dirRow = new Vector3();
  const nrmRow = new Vector3();
  const bend = opts.bend ?? 0;
  const rows = bend !== 0 ? 3 : 1; // length segments
  for (const a of anchors) {
    const tile = rng.int(4);
    const u0 = (tile % 2) * 0.5;
    const v0 = Math.floor(tile / 2) * 0.5;
    const s = a.scale * opts.sizeK;
    const roll = (rng.float() - 0.5) * 0.7;
    _q.copy(a.quat);
    _q2.setFromAxisAngle(Z, roll);
    _q.multiply(_q2);
    right.set(1, 0, 0).applyQuaternion(_q);
    upL.set(0, 1, 0).applyQuaternion(_q);
    out.set(0, 0, 1).applyQuaternion(_q);
    const flex = 0.45 + rng.float() * 0.35;
    const phase = rng.float() * Math.PI * 2;
    const planes = opts.mode === 'cross' ? 2 : 1;
    const bendJ = bend * (0.75 + rng.float() * 0.5);
    for (let pl = 0; pl < planes; pl++) {
      // plane 0: width=right, normal=upL; plane 1: width=upL, normal=right
      const w = pl === 0 ? right : upL;
      const nrm = pl === 0 ? upL : right;
      const base = g.vertCount;
      // march the card spine, bending away from the plane normal
      rowPos.copy(a.pos).addScaledVector(out, -0.08 * s);
      for (let iv = 0; iv <= rows; iv++) {
        const t = iv / rows;
        const ang = bendJ * t;
        dirRow.copy(out).multiplyScalar(Math.cos(ang)).addScaledVector(nrm, -Math.sin(ang));
        nrmRow.copy(nrm).multiplyScalar(Math.cos(ang)).addScaledVector(out, Math.sin(ang));
        for (let iu = 0; iu <= 1; iu++) {
          p.copy(rowPos).addScaledVector(w, (iu - 0.5) * s);
          g.vertex(
            p.x, p.y, p.z,
            nrmRow.x, nrmRow.y, nrmRow.z,
            u0 + iu * 0.5, v0 + t * 0.5,
            a.hue, flex, phase, 1 - a.age * 0.25,
          );
        }
        if (iv < rows) rowPos.addScaledVector(dirRow, s / rows);
      }
      for (let iv = 0; iv < rows; iv++) {
        const r0 = base + iv * 2;
        g.quad(r0, r0 + 1, r0 + 3, r0 + 2);
      }
    }
  }
}
