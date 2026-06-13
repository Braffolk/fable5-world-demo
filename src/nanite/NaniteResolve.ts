/**
 * Material resolve (N4) — the fullscreen-triangle mesh that turns the vis
 * buffer into shaded pixels INSIDE the main scene pass (D-N18): it renders
 * first (renderOrder −1000, before the VegPrepass twins at −100), writes the
 * raster's REAL f32 depth via depthNode (classic depth — verbatim bits), and
 * Discards uncovered pixels so the cleared depth survives for sky. Every
 * later hardware draw (grass, cards, water) depth-tests against nanite
 * geometry, and the post chain (aerial/AO/contact/TRAA — all post-space from
 * beauty+depth) applies unchanged.
 *
 * C0 shading = the debug view's flat matClass palette + fixed-light lambert
 * (structural-integration milestone). The N4-C1+ material übershader replaces
 * the fragment body per ported class.
 */

import { Mesh, Sphere, Vector3 } from 'three';
import { BufferGeometry, Float32BufferAttribute } from 'three';
import type { Texture } from 'three';
import { NodeMaterial } from 'three/webgpu';
import {
  Discard,
  Fn,
  If,
  cross,
  dot,
  float,
  max,
  normalize,
  positionGeometry,
  screenCoordinate,
  uint,
  vec3,
  vec4,
} from 'three/tsl';
import type { NF, NU, NV3 } from '../gpu/TSLTypes';
import type { RegistryGpu } from './GeometryRegistry';
import type { NaniteCam } from './NaniteCommon';
import { makeFetch } from './NaniteFetch';
import type { NaniteVisBuffers } from './NaniteRaster';
import { bcU2F, elemU } from './Tsl';
import type { BufOf, UV2 } from './Tsl';

export interface NaniteResolveHandles {
  /** add to engine.scene; renderOrder −1000, castShadow off */
  mesh: Mesh;
}

export function buildNaniteResolve(
  gpu: RegistryGpu,
  heightTex: Texture,
  cam: NaniteCam,
  cull: { qRasterRO: BufOf<UV2> },
  vis: NaniteVisBuffers,
): NaniteResolveHandles {
  const { makeCtx, fetchWorldVert, meshWord } = makeFetch(gpu, heightTex);

  const geometry = new BufferGeometry();
  geometry.setAttribute(
    'position',
    new Float32BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
  );
  geometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);

  const mat = new NodeMaterial();
  mat.vertexNode = vec4(positionGeometry.xy, 0, 1) as unknown as typeof mat.vertexNode;
  mat.fragmentNode = Fn(() => {
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    const dRaw = elemU(vis.depthV.ro, pixelIndex);
    const pRaw = elemU(vis.payloadV.ro, pixelIndex);
    If(dRaw.equal(uint(0xffffffff)), () => {
      Discard();
    });
    // orphan (payload never matched pass-1 depth): background, not a garbage
    // decode — black-pixel probes then see it as a hole
    If(pRaw.equal(uint(0xffffffff)), () => {
      Discard();
    });
    const itemIdx = pRaw.shiftRight(uint(7));
    const localTri = pRaw.bitAnd(uint(127));
    const item = cull.qRasterRO.element(itemIdx.add(uint(1)));
    const instId = item.x.toVar();
    const ci = item.y.toVar();
    const ctx = makeCtx(instId, ci);

    const w0 = fetchWorldVert(ctx, localTri, 0);
    const w1 = fetchWorldVert(ctx, localTri, 1);
    const w2 = fetchWorldVert(ctx, localTri, 2);
    const faceN = normalize(
      cross(w1.sub(w0) as unknown as NV3, w2.sub(w0) as unknown as NV3),
    ) as unknown as NV3;
    const L = normalize(vec3(0.55, 0.8, 0.25)) as unknown as NV3;
    const lambert = max(dot(faceN, L), 0).mul(0.85).add(0.18) as unknown as NF;

    // ?nandbg=dist — paint the fetched vis depth as view-distance/2000 in R
    // (numeric forensics; pair with waterdbg=8 + skyveldbg=raw)
    if (new URLSearchParams(window.location.search).get('nandbg') === 'dist') {
      const dist = float(0.3).div(
        float(1).sub(bcU2F(dRaw).mul(float(29999.7).div(30000))),
      );
      return vec4(dist.div(2000).clamp(0, 1), 0, 0, 1);
    }

    const matClass = meshWord(ctx.meshId as unknown as NU, 6)
      .shiftRight(uint(8))
      .bitAnd(uint(0xff))
      .toVar();
    const isT = matClass.equal(uint(0));
    const isR = matClass.equal(uint(1));
    const isB = matClass.equal(uint(2));
    const isD = matClass.equal(uint(3));
    const albedo = isT
      .select(
        vec3(0.3, 0.36, 0.22),
        isR.select(
          vec3(0.42, 0.41, 0.4),
          isB.select(vec3(0.36, 0.27, 0.19), isD.select(vec3(0.33, 0.28, 0.22), vec3(0.35, 0.33, 0.3))),
        ),
      )
      .toVar();
    return vec4((albedo as unknown as NV3).mul(lambert), 1);
  })() as unknown as typeof mat.fragmentNode;
  const nandepth = new URLSearchParams(window.location.search).get('nandepth');
  mat.depthNode = Fn(() => {
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    // ?nandepth=half — frag_depth liveness probe: constant 0.5 must erase
    // everything beyond ~0.6 m if the write reaches the attachment
    if (nandepth === 'half') return float(0.5);
    return bcU2F(elemU(vis.depthV.ro, pixelIndex));
  })() as unknown as typeof mat.depthNode;
  // always-pass + write (backend: depthTest=false → compare 'always';
  // depthWriteEnabled tracks depthWrite independently) — the cleared depth
  // only survives where the fragment Discards. ?nandepth=0 = bisect probe:
  // color without the depth write (later draws see cleared depth)
  mat.depthTest = false;
  mat.depthWrite = nandepth !== '0';
  mat.fog = false;
  mat.lights = false;

  const mesh = new Mesh(geometry, mat);
  mesh.name = 'naniteResolve';
  mesh.frustumCulled = false;
  mesh.renderOrder = -1000;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return { mesh };
}
