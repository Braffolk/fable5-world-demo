/**
 * Ground cover: grass blades (instanced, clumped placement preview of the
 * Phase-5 scatter law) + near-field debris classes (twigs, bark chips, leaf
 * litter cards reusing the broadleaf capture atlas as dry litter).
 *
 * Grass instancing: per-instance vec4 `idata` (hue, dryness, swayPhase,
 * height) on an InstancedBufferAttribute; the blade geometry itself carries
 * uv.y for the base→tip ramp.
 */

import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  Euler,
  InstancedBufferAttribute,
  InstancedMesh,
  Matrix4,
  Quaternion,
  type Texture,
  Vector3,
} from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { attribute, float, mix, smoothstep, texture, uv, vec3 } from 'three/tsl';
import type { Rng } from '../core/Seed';
import type { NF, NV3, NV4 } from '../gpu/TSLTypes';
import { grassTranslucency } from '../render/VegMaterials';
import { MeshGrower } from './TubeMesh';

/** single grass blade: tapered 4-segment strip with a built-in bend */
export function grassBladeGeometry(): BufferGeometry {
  const SEG = 4;
  const pos: number[] = [];
  const nrm: number[] = [];
  const uvA: number[] = [];
  const idx: number[] = [];
  const W = 0.014;
  const H = 1; // unit height; instance scales
  let bendZ = 0;
  for (let i = 0; i <= SEG; i++) {
    const t = i / SEG;
    const w = W * (1 - t * 0.85);
    bendZ = t * t * 0.28;
    const y = t * H * (1 - t * t * 0.06);
    if (i < SEG) {
      pos.push(-w, y, bendZ, w, y, bendZ);
      nrm.push(0, 0.25, -1, 0, 0.25, -1);
      uvA.push(0, t, 1, t);
    } else {
      pos.push(0, y, bendZ);
      nrm.push(0, 0.25, -1);
      uvA.push(0.5, 1);
    }
  }
  for (let i = 0; i < SEG; i++) {
    const a = i * 2;
    if (i < SEG - 1) idx.push(a, a + 1, a + 3, a, a + 3, a + 2);
    else idx.push(a, a + 1, a + 2);
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(new Float32Array(pos), 3));
  g.setAttribute('normal', new BufferAttribute(new Float32Array(nrm), 3));
  g.setAttribute('uv', new BufferAttribute(new Float32Array(uvA), 2));
  g.setIndex(idx);
  return g;
}

export function grassMaterial(): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const id = attribute('idata', 'vec4') as unknown as NV4;
  const t = uv().y as unknown as NF;
  const fresh = mix(
    vec3(0.026, 0.06, 0.012),
    vec3(0.1, 0.16, 0.035),
    t.mul(t),
  ) as unknown as NV3;
  const dry = mix(
    vec3(0.07, 0.055, 0.02),
    vec3(0.21, 0.16, 0.07),
    t,
  ) as unknown as NV3;
  let albedo = mix(fresh, dry, id.y) as unknown as NV3;
  albedo = albedo.mul(id.x.mul(0.16).add(1)) as unknown as NV3;
  mat.colorNode = albedo;
  mat.emissiveNode = grassTranslucency(albedo, t);
  // fake self-shadowing at the base
  mat.aoNode = smoothstep(0.0, 0.55, t).mul(0.55).add(0.45);
  mat.roughness = 0.88;
  mat.metalness = 0;
  mat.side = DoubleSide;
  return mat;
}

/**
 * Clumped grass patch: parent clump points + child blades (light-competition
 * clumping per spec §3.5). Returns an InstancedMesh ready to place.
 */
export function grassPatch(
  rng: Rng,
  count: number,
  size: number,
  opts?: { dryBase?: number },
): InstancedMesh {
  const geo = grassBladeGeometry();
  const mesh = new InstancedMesh(geo, grassMaterial(), count);
  const idata = new Float32Array(count * 4);
  const m = new Matrix4();
  const q = new Quaternion();
  const p = new Vector3();
  const s = new Vector3();
  const clumps: { x: number; z: number; h: number; dry: number }[] = [];
  const nClumps = Math.max(3, Math.round((size * size) / 1.1));
  for (let i = 0; i < nClumps; i++) {
    clumps.push({
      x: (rng.float() - 0.5) * size,
      z: (rng.float() - 0.5) * size,
      h: 0.55 + rng.float() * 0.75,
      dry: rng.float(),
    });
  }
  const axis = new Vector3(0, 1, 0);
  for (let i = 0; i < count; i++) {
    const c = clumps[rng.int(nClumps)] as { x: number; z: number; h: number; dry: number };
    const rr = Math.sqrt(rng.float()) * 0.62;
    const aa = rng.float() * Math.PI * 2;
    const x = c.x + Math.cos(aa) * rr;
    const z = c.z + Math.sin(aa) * rr;
    if (Math.abs(x) > size / 2 || Math.abs(z) > size / 2) {
      // recycle out-of-bounds onto a uniform filler
      p.set((rng.float() - 0.5) * size, 0, (rng.float() - 0.5) * size);
    } else {
      p.set(x, 0, z);
    }
    q.setFromAxisAngle(axis, rng.float() * Math.PI * 2);
    const h = c.h * (0.55 + rng.float() * 0.7) * 0.42;
    s.set(1 + rng.float() * 0.5, h, 1);
    m.compose(p, q, s);
    mesh.setMatrixAt(i, m);
    idata[i * 4] = rng.float() * 2 - 1;
    idata[i * 4 + 1] = Math.min(1, c.dry * 0.55 + rng.float() * 0.3 + (opts?.dryBase ?? 0));
    idata[i * 4 + 2] = rng.float() * Math.PI * 2;
    idata[i * 4 + 3] = h;
  }
  geo.setAttribute('idata', new InstancedBufferAttribute(idata, 4));
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

// ---------------------------------------------------------------------------
// Debris
// ---------------------------------------------------------------------------

/** bent twig: 4-sided micro tube, 2 kinks */
export function twigGeometry(rng: Rng): BufferGeometry {
  const g = new MeshGrower();
  const segs = 3;
  const pts: Vector3[] = [];
  const dir = new Vector3(1, 0.08 + rng.float() * 0.1, (rng.float() - 0.5) * 0.4).normalize();
  const p = new Vector3(0, 0.012, 0);
  const len = 0.22 + rng.float() * 0.3;
  for (let i = 0; i <= segs; i++) {
    pts.push(p.clone());
    dir.x += (rng.float() - 0.5) * 0.5;
    dir.z += (rng.float() - 0.5) * 0.5;
    dir.normalize();
    p.addScaledVector(dir, len / segs);
  }
  const radii = pts.map((_, i) => 0.008 * (1 - (i / segs) * 0.6));
  const dirs = pts.map((_, i) =>
    i === 0
      ? (pts[1] as Vector3).clone().sub(pts[0] as Vector3).normalize()
      : (pts[i] as Vector3).clone().sub(pts[i - 1] as Vector3).normalize(),
  );
  // reuse tube path via a minimal inline branch
  const hue = rng.float() * 2 - 1;
  const ring = 4;
  const rings: number[][] = [];
  const N = new Vector3();
  const B = new Vector3();
  for (let i = 0; i <= segs; i++) {
    const d = dirs[i] as Vector3;
    const ref = Math.abs(d.y) < 0.9 ? new Vector3(0, 1, 0) : new Vector3(1, 0, 0);
    N.crossVectors(ref, d).normalize();
    B.crossVectors(d, N).normalize();
    const r: number[] = [];
    for (let k = 0; k <= ring; k++) {
      const a = (k / ring) * Math.PI * 2;
      const nx = N.x * Math.cos(a) + B.x * Math.sin(a);
      const ny = N.y * Math.cos(a) + B.y * Math.sin(a);
      const nz = N.z * Math.cos(a) + B.z * Math.sin(a);
      const rr = radii[i] as number;
      const pt = pts[i] as Vector3;
      r.push(
        g.vertex(pt.x + nx * rr, pt.y + ny * rr, pt.z + nz * rr, nx, ny, nz, k / ring, i / segs, hue, 0, 0, 0.85),
      );
    }
    rings.push(r);
  }
  for (let i = 0; i < segs; i++) {
    const a = rings[i] as number[];
    const b = rings[i + 1] as number[];
    for (let k = 0; k < ring; k++) {
      g.quad(a[k] as number, b[k] as number, b[k + 1] as number, a[k + 1] as number);
    }
  }
  return g.build();
}

/** curled bark chip: bent shard */
export function barkChipGeometry(rng: Rng): BufferGeometry {
  const g = new MeshGrower();
  const L = 0.1 + rng.float() * 0.12;
  const W = L * (0.3 + rng.float() * 0.25);
  const curl = 0.5 + rng.float() * 0.9;
  const hue = rng.float() * 2 - 1;
  const rows = 3;
  const ids: number[][] = [];
  for (let i = 0; i <= rows; i++) {
    const t = i / rows;
    const ang = (t - 0.5) * curl;
    const y = 0.012 + Math.cos(ang) * W * 0.18;
    const off = Math.sin(ang) * W * 0.5;
    ids.push([
      g.vertex(t * L, y, -W / 2 + off * 0.3, 0, 1, 0, 0, t, hue, 0, 0, 0.8),
      g.vertex(t * L, y + off * 0.12, W / 2 + off * 0.3, 0, 1, 0, 1, t, hue, 0, 0, 0.8),
    ]);
  }
  for (let i = 0; i < rows; i++) {
    const a = ids[i] as number[];
    const b = ids[i + 1] as number[];
    g.quad(a[0] as number, b[0] as number, b[1] as number, a[1] as number);
  }
  return g.build();
}

export function debrisMaterial(kind: 'twig' | 'chip'): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const d = attribute('vdata', 'vec4') as unknown as NV4;
  const base = kind === 'twig' ? vec3(0.1, 0.075, 0.05) : vec3(0.085, 0.06, 0.04);
  mat.colorNode = base.mul(d.x.mul(0.2).add(1)).mul(d.w);
  mat.roughness = 0.95;
  mat.metalness = 0;
  mat.side = DoubleSide;
  return mat;
}

/** dry leaf-litter card material: reuses a foliage atlas, browned */
export function litterMaterial(atlas: Texture): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const t = texture(atlas, uv() as never) as unknown as NV4;
  const albedo = t.rgb.mul(t.rgb);
  // shift green leaf clusters toward dry browns
  const browned = mix(
    albedo.mul(vec3(1.7, 1.0, 0.42)).add(vec3(0.012, 0.006, 0.002)),
    vec3(0.11, 0.072, 0.034),
    float(0.5),
  );
  mat.colorNode = browned;
  mat.opacityNode = t.w;
  mat.alphaTest = 0.32;
  mat.roughness = 0.92;
  mat.metalness = 0;
  mat.side = DoubleSide;
  return mat;
}

/** scatter helper for small instanced debris over a square */
export function scatterInstances(
  mesh: InstancedMesh,
  rng: Rng,
  size: number,
  yJitter: number,
  scale: [number, number],
  flat: boolean,
): void {
  const m = new Matrix4();
  const p = new Vector3();
  const q = new Quaternion();
  const e = new Vector3();
  const eu = new Euler();
  const s = new Vector3();
  for (let i = 0; i < mesh.count; i++) {
    p.set((rng.float() - 0.5) * size, rng.float() * yJitter, (rng.float() - 0.5) * size);
    if (flat) {
      q.setFromAxisAngle(e.set(0, 1, 0), rng.float() * Math.PI * 2);
      const tilt = new Quaternion().setFromAxisAngle(
        e.set(1, 0, 0).applyQuaternion(q),
        (rng.float() - 0.5) * 0.5,
      );
      q.premultiply(tilt);
    } else {
      q.setFromEuler(eu.set(rng.float() * 6.28, rng.float() * 6.28, rng.float() * 6.28));
    }
    const sc = scale[0] + rng.float() * (scale[1] - scale[0]);
    s.set(sc, sc, sc);
    m.compose(p, q, s);
    mesh.setMatrixAt(i, m);
  }
  mesh.castShadow = true;
  mesh.receiveShadow = true;
}
