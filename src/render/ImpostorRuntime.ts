/**
 * ImpostorRuntime — the draw-side of the octahedral impostors (closes D-4).
 *
 * One camera-facing (cylindrical-billboard) quad per instance; the view
 * direction — rotated into the instance's capture frame by its yaw — selects
 * a hemi-octahedral grid cell, and the FOUR neighboring view tiles are
 * blended bilinearly (smooth view interpolation, no tile pop). Albedo is
 * sqrt-decoded and relit through the captured world-space normals (rotated
 * back by yaw), so impostors respond to sun/GI like real geometry. Depth
 * parallax from normalDepth.a is NOT applied (documented in DEVIATIONS D-4).
 */

import { PlaneGeometry } from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import {
  cameraPosition,
  clamp,
  float,
  positionLocal,
  texture,
  transformNormalToView,
  uv,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { IMPOSTOR_GRID, type ImpostorAtlas } from '../vegetation/Impostors';
import type { NF, NV2, NV4 } from '../gpu/TSLTypes';
import {
  applyDitherFade,
  applyInstanceTint,
  fetchInstance,
  type InstanceBinding,
} from './VegInstance';

/** unit quad geometry shared by all impostor draws (x,y ∈ −1..1) */
export function impostorQuad(): PlaneGeometry {
  return new PlaneGeometry(2, 2);
}

export function impostorRuntimeMaterial(
  atlas: ImpostorAtlas,
  bind: InstanceBinding,
): MeshStandardNodeMaterial {
  const mat = new MeshStandardNodeMaterial();
  const { A, B, slot } = fetchInstance(bind);
  const s = A.w;
  const r = s.mul(atlas.radius);
  const cy = s.mul(atlas.centerY);

  // ---- cylindrical billboard --------------------------------------------------
  const toCam = cameraPosition.sub(A.xyz);
  const hl = vec2(toCam.x, toCam.z).length().max(1e-3);
  const right = vec3(toCam.z.div(hl), 0, toCam.x.negate().div(hl));
  mat.positionNode = A.xyz
    .add(right.mul(positionLocalX(r)))
    .add(vec3(0, 1, 0).mul(cy.add(positionLocalY(r))));

  // ---- hemi-octahedral tile select (capture frame = world rotated by −yaw) ---
  const c = B.x.cos();
  const sn = B.x.sin();
  const center = A.xyz.add(vec3(0, 1, 0).mul(cy));
  const dirW = cameraPosition.sub(center).normalize();
  // inverse of the instance yaw rotation used by instanceVeg
  const dl = vec3(
    dirW.x.mul(c).sub(dirW.z.mul(sn)),
    dirW.y.max(0.03),
    dirW.x.mul(sn).add(dirW.z.mul(c)),
  ).normalize();
  const k = dl.x.abs().add(dl.y).add(dl.z.abs());
  const px = dl.x.div(k);
  const pz = dl.z.div(k);
  // inverse of hemiOctDecode: x = px − pz, z = px + pz
  const g = clamp(
    vec2(px.sub(pz).add(1).mul(0.5), px.add(pz).add(1).mul(0.5)).mul(IMPOSTOR_GRID).sub(0.5),
    0,
    IMPOSTOR_GRID - 1.001,
  );
  const g0 = varying(g.floor()) as unknown as NV2;
  const f = varying(g.fract()) as unknown as NV2;
  const cs = varying(vec2(c, sn)) as unknown as NV2;

  // ---- bilinear 4-tile blend ---------------------------------------------------
  const inv = 1 / IMPOSTOR_GRID;
  const sampleTiles = (tex: ImpostorAtlas['albedo']): NV4 => {
    let acc: NV4 = vec4(0, 0, 0, 0);
    for (let j = 0; j <= 1; j++) {
      for (let i = 0; i <= 1; i++) {
        const tile = clamp(g0.add(vec2(i, j)), 0, IMPOSTOR_GRID - 1);
        const w = (i === 1 ? f.x : float(1).sub(f.x)).mul(
          j === 1 ? f.y : float(1).sub(f.y),
        );
        const tuv = tile.add(uv() as unknown as NV2).mul(inv);
        acc = acc.add((texture(tex, tuv as never) as unknown as NV4).mul(w)) as NV4;
      }
    }
    return acc;
  };

  const alb = sampleTiles(atlas.albedo);
  const nd = sampleTiles(atlas.normalDepth);

  mat.colorNode = alb.rgb.mul(alb.rgb); // sqrt-encoded at capture
  mat.opacityNode = alb.a;
  mat.alphaTest = 0.28;
  mat.roughness = 0.75;
  mat.metalness = 0;

  // captured world-space normal, rotated by the instance yaw
  const n = nd.rgb.mul(2).sub(1);
  const nW = vec3(
    n.x.mul(cs.x).add(n.z.mul(cs.y)),
    n.y,
    n.z.mul(cs.x).sub(n.x.mul(cs.y)),
  );
  mat.normalNode = transformNormalToView(nW as never);

  const dist = A.xyz.sub(cameraPosition).length();
  if (bind.fade) applyDitherFade(mat, dist, bind.fade);
  applyInstanceTint(mat, slot, bind.tint ?? 0.14);
  return mat;
}

// local-position helpers (PlaneGeometry XY in −1..1)
function positionLocalX(r: NF): NF {
  return positionLocal.x.mul(r);
}
function positionLocalY(r: NF): NF {
  return positionLocal.y.mul(r);
}
