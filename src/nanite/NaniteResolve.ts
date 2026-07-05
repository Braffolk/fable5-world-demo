/**
 * Material resolve (N4) — the fullscreen-triangle mesh that turns the vis
 * buffer into shaded pixels INSIDE the main scene pass (D-N18): renders first
 * (renderOrder −1000), writes the raster's REAL f32 depth via depthNode, and
 * Discards uncovered pixels so the cleared depth survives for the sky.
 *
 * ARCHITECTURE (reset after the C1 fragility — D-N20): a CLIP-SPACE fullscreen
 * triangle + a plain NodeMaterial with a fragmentNode. This is the C0 path
 * that provably rendered near-camera terrain (the `?nanitedbg=flat` view).
 * The earlier camera-glued near-plane triangle + MeshPhysicalNodeMaterial
 * failed to compile its lighting (missing geometry normal) and fell back to a
 * material that ignored every node — leaving the near terrain transparent.
 *
 * Shading is computed in the fragment from the reconstructed surface
 * (manual lighting, D-N17): TERRAIN runs buildTerrainShading on the
 * reconstructed world position, then sun + sky-ambient (+ probe GI). CSM
 * receive + exact IBL parity are layered in next; the immediate contract is
 * that the terrain is THERE (opaque, correct albedo), not transparent.
 *
 * Bisects (URL-gated): ?nandbg=flat (albedo, no shading) | albedo | normal |
 * cov (covered px red) | cls (matClass tint) | cluster (meshlet hash tint, like
 * the ?nanitedbg=cluster view but for the full-frame migrated set); ?nandepth=0
 * (depth write off); ?nanshadow=0; ?nanwind=0; ?nanbark=const|lN|grad.
 */

import { Mesh, Sphere, Vector3 } from 'three';
import { BufferGeometry, Float32BufferAttribute } from 'three';
import type { Texture } from 'three';
import { NodeMaterial, type StorageTexture } from 'three/webgpu';
import {
  Discard,
  Fn,
  If,
  cameraProjectionMatrixInverse,
  cameraWorldMatrix,
  cross,
  dot,
  float,
  floor,
  fract,
  getViewPosition,
  int,
  max,
  mix,
  nodeObject,
  normalize,
  positionGeometry,
  screenCoordinate,
  screenUV,
  sin,
  smoothstep,
  texture,
  uint,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { NB, NF, NU, NV2, NV3, NV4 } from '../gpu/TSLTypes';
import type { CSMShadowNode } from 'three/addons/csm/CSMShadowNode.js';
import type { NaniteShadow } from './NaniteShadow';
import type { ShadowHalf } from './NaniteShadowHalf';
import { causticContext, causticDepth, causticTint } from '../render/Caustics';
import { buildTerrainShading } from '../render/TerrainMaterial';
import { sunU } from '../render/VegMaterials';
import { canopyAt } from '../gpu/passes/Scatter';
import { BARK_RES } from '../gpu/passes/BarkSynth';
import { fbm3, valueNoise3 } from '../gpu/noise/NoiseTSL';
import type { ProbeGI } from '../gpu/passes/ProbeGI';
import type { Heightfield } from '../world/Heightfield';
import { CLUSTER_TRI_BITS, CLUSTER_TRI_MASK, CLUSTER_WORDS, MESH_FLAG_FARTILE, MESH_WORDS, readVertex } from './GeometryRegistry';
import type { RegistryGpu } from './GeometryRegistry';
import { brickNormalTsl, brickWord, BRICK_ALBEDO, BRICK_NORMAL, BRICK_POS_X } from './VoxelBrick';
import { makeFetch, slotHash } from './NaniteFetch';
import { GRASS_FAR_BASE } from './NaniteGrass';
import { hashColor, instRotateDir, instTransformPoint, instYaw, type NaniteCam } from './NaniteCommon';
import { clusterHwClass } from './NaniteHwClass';
import type { NaniteVisBuffers } from './NaniteRaster';
import { bcU2F, elemU, toF, uniformF } from './Tsl';
import type { BufOf, UV2, UniformF } from './Tsl';

export interface NaniteResolveHandles {
  /** add to engine.scene; renderOrder −1000, castShadow off */
  mesh: Mesh;
  /** voxel-foliage two-pass resolve (spec §4.6 / §7): a SECOND fullscreen pass that
   *  shades ONLY voxel-winner pixels (bit31 marker). Present only when the voxel queue
   *  is wired (?voxreg/?forcevox). The MAIN `mesh` above then DROPS voxelBricks/
   *  qVoxRasterRO entirely (skips voxel pixels via Discard) so it stays ≤10 fragment
   *  storage buffers, and THIS pass drops the tri-only verts/indices/qRasterRO (a voxel
   *  pixel never re-derives a triangle) so it also stays ≤10. Add to engine.scene right
   *  after `mesh` (renderOrder −999, just after the main resolve). */
  voxMesh?: Mesh;
  /** runtime gate for the full-screen three-CSM `keep` sample (1=full, 0=corner-only). Shared
   *  by both resolve passes. Exposed for a thermal-invariant within-boot A/B (setKeepFull). */
  keepFullU: UniformF;
}

export interface ResolveWorld {
  hf: Heightfield;
  /** RP-1 (deep-review 16 tri-class specialization): matClass ids with ≥1 registered
   *  mesh (GeometryRegistry.presentClasses). When set, the tri resolve SKIPS BUILDING
   *  the shading subgraphs of absent classes — output-identical (their If() guards can
   *  never fire) but strips samples/ALU from the shader ⇒ register pressure/occupancy
   *  (the vox-pass 37.5ms-cliff mechanism, comment at the terrain block). undefined or
   *  ?resclasses=0 ⇒ build everything (legacy). */
  presentClasses?: ReadonlySet<number>;
  gi: ProbeGI | null;
  canopyTex: StorageTexture | null;
  /** sun CSM cascades (D-N17 shadow receive) — sampled at the reconstructed
   *  world position via receivedShadowPositionNode; null = no sun shadows.
   *  At runtime this is the CachedCsmShadowNode (default), whose
   *  shadowPositionWorld-based cascade-select is what makes per-pixel
   *  reconstructed positions select the right cascade (?shadowcache=0 falls
   *  back to the base positionView.z select — a debug-only A/B). */
  csm: CSMShadowNode | null;
  /** P2 (shadow arc): "scene has sun shadows" WITHOUT a legacy CSM node. The
   *  nanite-only world severs the CSM entirely (its maps were empty; the keepalive
   *  rendered ~2 blank 2048² cascades/frame and the full-screen `keep` sample paid
   *  a per-pixel CSM PCSS purely to keep the node built). */
  sunShadows?: boolean;
  /** P2: world-space cloud sun-transmittance gate — pre-sever it rode the CSM
   *  filterNode (pcssFilter × clouds.shadowAt) and reached us through `keep`; now
   *  multiplied into the sun term directly (full-res, NaN-guarded). */
  cloudShadow?: ((wxz: NV2) => NF) | null;
  /** P4: baked heightfield sun-visibility (FarShadow, 1 bilinear tap) — the
   *  beyond-clipmap far-field term (mountains shade valleys at any distance). */
  farShadow?: ((wxz: NV2) => NF) | null;
  /** bark/deadwood texture-array (texA albedo+cavity, texB normal+rough+height);
   *  sampled at the per-mesh layer slice (mesh word 7). null = bark unported. */
  barkTexA: Texture | null;
  barkTexB: Texture | null;
  /** N5-R0 (D-N28): nanite's own depth-only shadow path. When present, the resolve
   *  takes the sun-shadow factor from shadowFactor() (PCSS over our r32 cascade
   *  textures) INSTEAD of three's CSM node — three's shadow map stays empty in the
   *  black slate. null = fall back to the old csm receive (?oldgeo). */
  naniteShadow: NaniteShadow | null;
  /** S0 (D-N29): when present, the sun-shadow factor is taken from a HALF-RES PCSS
   *  eval + depth-aware bilateral upsample (NaniteShadowHalf) instead of the per-
   *  pixel shadowFactor — ~4× fewer PCSS taps. null = full-res (?shalfres=0). */
  shadowHalf: ShadowHalf | null;
  /** procedural grass (NaniteGrass, grass rethink 2026-07-03): grass pixels carry
   *  the bit31|bit30 id namespace and shade in the vox-side pass ('vox'/'both').
   *  derive() re-builds the pixel's blade triangle from its self-describing id
   *  (zero storage buffers — the binding budget is untouched). */
  grassProc?: {
    derive(body: NU, wp: NV3): { t: NF; nrm: NV3 };
    /** G-D lean lighting: one filtered tap of the per-frame guide light field →
     *  vec4(sunVis, irradianceRGB). Non-null ⇒ grass pixels SKIP the bilateral
     *  shadow upsample + GI probe chain (the measured ~7 ms pixel-proportional
     *  wall) and use this instead. ?grasslean=0 → null → old path. */
    lean?: ((wpXZ: NV2) => NV4) | null;
    /** G-E article lane (?grass=ray): per-pixel vec4(worldNrm, tipParam) from the
     *  raycast lane's screen texture (the algorithm's own depth+normal output).
     *  Non-null ⇒ grass pixels use it INSTEAD of derive() (whose per-blade id
     *  reconstruction the baked-fetch lane doesn't carry). */
    ray?: ((px: NU) => NV4) | null;
  } | null;
}

/** TextureNode sample-config chain (depth = array slice, grad = explicit deriv) */
interface TexSample {
  depth(d: unknown): TexSample;
  grad(a: unknown, b: unknown): TexSample;
}

/** hue jitter (port of VegMaterials.hueShift): warm/cool tint by vdata.x */
function hueShift(base: NV3, hue: NF, amount: number): NV3 {
  const k = hue.mul(amount);
  const warm = vec3(1.18, 1.0, 0.55);
  const cool = vec3(0.7, 0.95, 1.25);
  return base
    .mul(warm)
    .mul(k.clamp(0, 1))
    .add(base.mul(cool).mul(k.negate().clamp(0, 1)))
    .add(base.mul(float(1).sub(k.abs()))) as unknown as NV3;
}

/** 3D barycentric of p inside triangle (a,b,c) — perspective-correct because
 *  p is the real reconstructed world point on the rasterized surface (not a
 *  screen-interpolated value). Ericson's method. */
function baryWeights(p: NV3, a: NV3, b: NV3, c: NV3): NV3 {
  const v0 = b.sub(a);
  const v1 = c.sub(a);
  const v2 = p.sub(a);
  const d00 = dot(v0, v0);
  const d01 = dot(v0, v1);
  const d11 = dot(v1, v1);
  const d20 = dot(v2, v0);
  const d21 = dot(v2, v1);
  const denom = d00.mul(d11).sub(d01.mul(d01)).max(float(1e-12)) as unknown as NF;
  const v = d11.mul(d20).sub(d01.mul(d21)).div(denom) as unknown as NF;
  const w = d00.mul(d21).sub(d01.mul(d20)).div(denom) as unknown as NF;
  const u = float(1).sub(v).sub(w);
  return vec3(u, v, w) as unknown as NV3;
}

/** registry vdata word (4×u8 unorm, WorldRegistry.geometryToSource) → vec4 */
function unpackVdata(packed: NU): NV4 {
  return vec4(
    toF(packed.bitAnd(uint(0xff))),
    toF(packed.shiftRight(uint(8)).bitAnd(uint(0xff))),
    toF(packed.shiftRight(uint(16)).bitAnd(uint(0xff))),
    toF(packed.shiftRight(uint(24)).bitAnd(uint(0xff))),
  ).div(255) as unknown as NV4;
}

/** ROCK material (port of VegMaterials.rockMaterial, defaults) — strata banding
 *  from vdata.y, lichen via vdata.z, AO/moss via vdata.w, on world-space noise.
 *  Returns albedo (incl. the colorNode AO darkening) + ao (the aoNode, applied
 *  to indirect only by the caller). Roughness omitted — terrain is matte-lit. */
function rockShade(d: NV4, wp: NV3, nrm: NV3): { albedo: NV3; ao: NF } {
  const strataT = d.y;
  const upness = nrm.y.max(0);
  const bandTint = valueNoise3(vec3(float(0), strataT.mul(7.3), float(0)).add(wp.mul(0.02)));
  const grain = fbm3(wp.mul(2.1), 3).mul(0.5).add(0.5);
  const tr = 0.285;
  const tg = 0.255;
  const tb = 0.215;
  let albedo = mix(
    vec3(tr * 0.42, tg * 0.44, tb * 0.55),
    vec3(tr, tg, tb),
    bandTint.mul(0.55).add(grain.mul(0.45)).clamp(0, 1),
  ) as unknown as NV3;
  const lich = smoothstep(0.62, 0.78, valueNoise3(wp.mul(3.7))).mul(d.z.mul(0.7).add(0.3));
  albedo = mix(albedo, vec3(0.16, 0.175, 0.14), lich.mul(0.55)) as unknown as NV3;
  albedo = mix(albedo, vec3(0.17, 0.15, 0.12), upness.pow(2).mul(0.3)) as unknown as NV3;
  const steep = float(1).sub(upness);
  const streakN = valueNoise3(vec3(wp.x.mul(2.6), wp.y.mul(0.22), wp.z.mul(2.6)));
  const streak = smoothstep(0.55, 0.82, streakN).mul(smoothstep(0.45, 0.8, steep)).mul(0.55);
  albedo = mix(albedo, albedo.mul(vec3(0.5, 0.46, 0.4)), streak) as unknown as NV3;
  // moss (default amount 0.25 → ×0.5 gate)
  const mossN = smoothstep(0.45, 0.75, fbm3(wp.mul(1.7), 3).mul(0.5).add(0.5));
  const moss = smoothstep(0.45, 0.85, upness).mul(mossN).mul(d.w).mul(0.5).clamp(0, 1);
  albedo = mix(albedo, vec3(0.045, 0.085, 0.03), moss) as unknown as NV3;
  albedo = albedo.mul(d.w.mul(0.35).add(0.65)) as unknown as NV3; // colorNode AO darkening
  return { albedo, ao: d.w as unknown as NF };
}

export function buildNaniteResolve(
  gpu: RegistryGpu,
  heightTex: Texture,
  cam: NaniteCam,
  cull: {
    qRasterRO: BufOf<UV2>;
    /** voxel-foliage (Stage 2 §7): the voxel-raster work queue — present only when
     *  ?voxreg/?forcevox is active. A bit31 winner (the voxel marker) decodes its
     *  (instId, ci) here, then the cluster's brickBase + gpu.voxelBricks normal slice. */
    qVoxRasterRO?: BufOf<UV2>;
  },
  vis: NaniteVisBuffers,
  world: ResolveWorld,
): NaniteResolveHandles {
  const hf = world.hf;
  if (!hf.biomeTex || !hf.fieldsTex || !hf.noiseA || !hf.noiseB) {
    throw new Error('NaniteResolve: heightfield derived maps missing (boot order)');
  }
  const q = new URLSearchParams(window.location.search);
  // ROCK (and future explicit-mesh classes) need per-vertex attributes →
  // re-fetch the cluster triangle. Terrain reconstructs wp from depth and
  // never touches this. ?nanwind=0 A/Bs the trunk wind — MUST match the raster's
  // makeFetch (both read this flag) so their windy positions stay bit-identical.
  const windOn = q.get('nanwind') !== '0';
  // RP-1: presence-gated class subgraphs (?resclasses=0 = build all, legacy).
  const present = q.get('resclasses') !== '0' ? (world.presentClasses ?? null) : null;
  const hasClass = (id: number): boolean => present === null || present.has(id);
  if (present !== null) {
    const stripped = [0, 1, 2, 3, 4].filter((id) => !present.has(id));
    if (stripped.length > 0)
      // eslint-disable-next-line no-console
      console.log(
        `[nanite] resolve tri-class strip (RP-1): matClass ${stripped.join(',')} absent from the registry — shading subgraphs not built (?resclasses=0 to disable)`,
      );
  }
  // bindHfVerts=false: the resolve reconstructs terrain world pos from DEPTH and only
  // calls fetchWorldVert for rock/bark (the explicit-mesh else branch), so it must NOT
  // bind the stride-1 terrain buffer — one fewer storage buffer in the fragment stage (2e).
  const fetch = makeFetch(gpu, heightTex, undefined, windOn ? { camPos: cam.camPos } : undefined, false);
  const nandepth = q.get('nandepth');
  const nandbg = q.get('nandbg');
  // ?clhwmax — SW/HW crossover px for the ?nandbg=clhw split tint (matches the raster's default).
  const clhwMax = Math.max(2, Number(q.get('clhwmax') ?? '16') || 16);
  // ?nanbark= bisect: const (flat brown) | lN (force mip N) | grad (anisotropic
  // ray-plane derivatives — known NaN on near trunks, default is analytic LOD)
  const nanbark = q.get('nanbark');
  // ?nanshadow=0 — master off for the whole nanite shadow system (this receive
  // term AND the per-cascade producer in NaniteFrame read the same flag). Default
  // ON. With the producer on, world.naniteShadow drives the PCSS branch below; the
  // csm-only branch is the ?oldgeo fallback (receives the old caster maps).
  const shadowsOn =
    (world.csm !== null || world.sunShadows === true) && q.get('nanshadow') !== '0';
  // ?voxao=0 — disable the per-brick DIRECTIONAL self-shading on VOXEL foliage. Voxels are shaded
  // with each brick's BAKED mean normal (VoxelBrick word2), which drives the sun N·L (line ~722) +
  // the normal.y ambient floor (line ~782) below, so brick faces angled away from the sun read
  // darker — the "shadow/AO on the voxel cube faces" the user sees (it is NOT ?nanshadow, the sun
  // SHADOW pass, nor ?occl, the cull). Default ON. =0 skips the brick-normal decode (also drops the
  // gpu.voxelBricks read in the vox material) and leaves the flat up-normal ⇒ uniformly flat-lit
  // voxels, for an A/B of the LOOK. HONEST PERF NOTE: the normal is baked, so this term is one
  // storage read + a dot/normalize per voxel pixel in a pass that runs regardless — disabling it
  // is expected to change GPU cost ~negligibly; the value is the visual A/B, not a perf win.
  const voxShade = q.get('voxao') !== '0';
  // ?voxbn (DEFAULT ON — must match NaniteVoxelRaster): the election payload carries the
  // winning brick's index in bits 21-27, so shade with THAT brick's baked normal + albedo
  // instead of brick[0]-of-the-block. Fixes the per-block flat shading that fused adjacent
  // bricks into giant single-color plates / near-black crowns (2026-07-01 review).
  const voxBrickShade = q.get('voxbn') !== '0';
  // ?voxbead=k — per-pixel "round normal" for voxel foliage (user: far crowns read as flat
  // blocky quads — one normal+albedo per brick = piecewise-constant shading). Bends the
  // shading normal toward the BEAD direction normalize(wp − brickCenterWorld), so every
  // brick gets ball-like within-brick gradients through the existing wrap+ambient lighting
  // (the classic SpeedTree crown-normal trick, applied per brick). k = blend weight 0..1.
  // Costs 3 f32 brick-center loads + ~20 ALU on vox-pass pixels only. 0 = off (exact old).
  const voxBeadRaw = Number(q.get('voxbead') ?? '0.6');
  const voxBeadK = Number.isFinite(voxBeadRaw) ? Math.max(0, Math.min(1, voxBeadRaw)) : 0.6;
  // ?voxjit=a — world-anchored per-cell VALUE jitter on voxel albedo (breaks the few-greens
  // camouflage tiling of the far field). Hash of floor(wp·1.4) ⇒ ~0.7 m cells, stable under
  // camera motion (no payload bits needed). a = ± value amplitude. 0 = off (exact old).
  const voxJitRaw = Number(q.get('voxjit') ?? '0.12');
  const voxJitK = Number.isFinite(voxJitRaw) ? Math.max(0, Math.min(1, voxJitRaw)) : 0.12;
  // ?ftnrm=k — far-TILE normal up-blend strength (0 = off/legacy, 1 = fully flat-lit).
  // See the MESH_FLAG_FARTILE block in the voxel shade path below.
  const ftNrmRaw = Number(q.get('ftnrm') ?? '0.65');
  const ftNrmK = Number.isFinite(ftNrmRaw) ? Math.max(0, Math.min(1, ftNrmRaw)) : 0.65;
  // ?voxgrad=k — crown-scale vertical light gradient on voxel albedo. DEFAULT 0 (OFF,
  // 2026-07-02): measured with voxjit2 at eye +4.9 / obl +4.6 ms COMBINED (per-voxel-pixel
  // instance fetch + hash in the resolve) — fails the user's looks-per-ms bar for a subtle
  // albedo effect. Knob kept for the next beauty pass (batch the fetch first).
  const voxGradRaw = Number(q.get('voxgrad') ?? '0');
  const voxGradK = Number.isFinite(voxGradRaw) ? Math.max(0, Math.min(1, voxGradRaw)) : 0;
  // ?voxjit2=k — CLUMP-scale (~3 m) albedo variation (grain larger than far bricks groups
  // them into organic patches). DEFAULT 0 (OFF) — same measured-cost story as voxgrad.
  const voxJit2Raw = Number(q.get('voxjit2') ?? '0');
  const voxJit2K = Number.isFinite(voxJit2Raw) ? Math.max(0, Math.min(1, voxJit2Raw)) : 0;
  // ?leafcheap=all — ATTRIBUTION LEVER: route EVERY mesh-leaf pixel through the ?resfar
  // cheap far-leaf path (species tint × quad normal, no makeCtx/gust/3-vert interp). The
  // measured delta vs default = the exact ceiling of any "make leaf decode cheaper" work
  // (shade-binning et al) with a real candidate look, not an ugly const color.
  const leafCheapAll = q.get('leafcheap') === 'all';
  // ?resfar=N — bark micro-detail distance gate (m); beyond it moss fbm + normal-map are
  // skipped (sub-texel there). 0 disables (legacy full-detail everywhere). Default 60.
  const resFarRaw = Number(q.get('resfar') ?? '60');
  const resFarDist = Number.isFinite(resFarRaw) && resFarRaw >= 0 ? resFarRaw : 60;
  // ?reskeep=0 — drop the redundant three-CSM `keep` factor in the lighting (below).
  // When OUR nanite depth-shadow is active (default), the resolve ALSO references three's
  // CSMShadowNode purely to multiply in `keep` — but three's cascade maps are EMPTY in the
  // black slate (all casters render through the nanite path, castShadow=false), so every
  // PCSS/PCF tap returns "lit" ⇒ keep≡1 and the whole per-pixel sample is wasted work that the
  // half-res shadow optimisation does NOT quarter (it is full-res). Dropping it also makes
  // `receivedShadowPositionNode` (which exists ONLY to feed that CSM node) dead → a SECOND
  // per-pixel wp reconstruction vanishes. Default ON (=keep present = bit-identical old path).
  // CAVEAT: referencing the CSM node is also what makes three run its per-frame cascade FIT
  // that NaniteShadow.run consumes; with keep dropped the fit is driven explicitly (see
  // NaniteFrame). Static measurement + shotdiff are fit-independent. Naniteshadow-active only.
  const keepOn = q.get('reskeep') !== '0';
  // Runtime-toggleable gate for the keep sample (see the lighting block). Default mirrors the
  // reskeep flag; window.__laasNanite.setKeepFull(0|1) flips it WITHIN a boot for a clean,
  // thermal-invariant A/B of the full-screen CSM sample cost (cross-boot gpuWall is thermally
  // noisy). Shared by both resolve passes so the toggle affects tri + vox together.
  const keepFullU = uniformF(keepOn ? 1 : 0);
  // PERF-VB4 (D-N45): the WORLD raster is single-pass — ONE SW+HW pass elects the 24-bit
  // depth key into visPayloadV (high bits) and stores the full 25-bit id into the side
  // buffer visBV. The resolve takes the id from visBV and reconstructs depth from the
  // election key (cz = 1 − (key>>8)/16777215); there is no exact depthV.

  // CLIP-SPACE fullscreen triangle (covers ndc [-1,1]² via (-1,-1),(3,-1),(-1,3))
  const geometry = new BufferGeometry();
  geometry.setAttribute(
    'position',
    new Float32BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3),
  );
  geometry.boundingSphere = new Sphere(new Vector3(), Number.POSITIVE_INFINITY);

  // voxel-foliage (spec §4.6 / §7): the resolve is TWO fullscreen passes when the voxel
  // queue is wired. The single-pass design bound the full tri-fetch set PLUS voxelBricks +
  // qVoxRasterRO in ONE fragment shader (WGSL/NodeMaterial binds every REFERENCED storage
  // buffer regardless of the runtime isV branch) → it busts the Metal storage-buffer ceiling
  // ⇒ the pipeline is INVALID and NOTHING shades. Fix: build the material TWICE —
  //   pass 'tri': the MAIN resolve. Shades TRIANGLE pixels (terrain/rock/bark/leaf), and
  //     SKIPS voxel-winner pixels via Discard. Does NOT reference voxelBricks/qVoxRasterRO at
  //     all, so it stays at its prior ≤10 (8: payloadV/visBV/qRasterRO/clusters/meshes +
  //     verts/indices/instances).
  //   pass 'vox': a SECOND small resolve. Shades ONLY voxel-winner pixels (bit31 marker) and
  //     SKIPS triangle pixels via Discard. DROPS the tri-only verts/indices/qRasterRO (a voxel
  //     pixel never re-derives a triangle), binding instead voxelBricks + qVoxRasterRO — so it
  //     stays at 7 (payloadV/visBV/clusters/meshes/instances + voxelBricks/qVoxRasterRO).
  // Both passes are provably ≤10 fragment storage buffers. The 'vox' pass is built only when
  // the voxel queue is present (cull.qVoxRasterRO) — a pure-triangle world has ONE pass.
  const buildMat = (pass: 'tri' | 'vox' | 'both'): NodeMaterial => {
  const mat = new NodeMaterial();
  mat.name = `naniteResolve_${pass}`;
  mat.vertexNode = vec4(positionGeometry.xy, 0, 1) as unknown as typeof mat.vertexNode;

  // D-N17 shadow receive: the CSM cascade-select + sampling read
  // `shadowPositionWorld`, which ShadowBaseNode sources from
  // material.receivedShadowPositionNode. The fullscreen triangle's
  // positionWorld is the clip-space vertex (useless), so supply the
  // per-pixel RECONSTRUCTED world position — self-contained like depthNode,
  // not a closure var, so it builds inside the shadow subgraph cleanly.
  // P2: exists ONLY to feed the legacy CSM node — gate on csm (the severed
  // world skips this SECOND per-pixel wp reconstruction entirely).
  if (shadowsOn && world.csm) {
    (mat as unknown as { receivedShadowPositionNode?: unknown }).receivedShadowPositionNode = Fn(
      () => {
        const fy = float(cam.uH).sub(screenCoordinate.y);
        const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
        const zDev = float(1).sub(
          toF(elemU(vis.payloadV.ro, pixelIndex).shiftRight(uint(8))).div(16777215),
        ) as unknown as NF;
        const wpv = getViewPosition(
          screenUV,
          zDev,
          cameraProjectionMatrixInverse,
        ) as unknown as NV3;
        return (
          (cameraWorldMatrix as unknown as { mul(v: NV4): NV4 }).mul(
            (vec4 as unknown as (a: NV3, b: number) => NV4)(wpv, 1),
          ) as unknown as NV4
        ).xyz;
      },
    )();
  }

  mat.fragmentNode = Fn(() => {
    // vis-buffer fetch (bottom-up rows: raster writes y·W+x with y bottom-up,
    // screenCoordinate is top-down, so flip)
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    // id from the full-id side buffer (visBV); depth + covered-test from the 24-bit
    // election key (visPayloadV high bits). Anchor 0 = no fragment (cleared) → background.
    const pRaw = elemU(vis.visBV.ro, pixelIndex).toVar();
    const elect = elemU(vis.payloadV.ro, pixelIndex).toVar();
    If(elect.equal(uint(0)), () => {
      Discard();
    });

    // voxel-foliage (Stage 2 §7.1 + two-pass §4.6): the bit31 VOXEL marker partitions the
    // two fullscreen passes. A voxel id is NOT a triangle itemIdx, so the tri pass must NOT
    // index qRasterRO with one (it would read garbage). Each pass keeps ONLY the buffers it
    // needs (so neither busts the storage-buffer ceiling): the 'tri' pass Discards voxel
    // pixels and never touches voxelBricks/qVoxRasterRO; the 'vox' pass Discards non-voxel
    // pixels and never touches verts/indices/qRasterRO. The 'both' pass (?respass=1, RP-4)
    // shades the WHOLE partition in one draw — no partition Discard — and binds BOTH queues:
    // legal ONLY where the fragment storage-buffer union is ≤10 (forest, gi null → exactly
    // 10); a config with a probe buffer must keep two-pass or the 11th binding silently
    // kills the pipeline.
    // RP-5: the partition Discard runs BEFORE the wp reconstruction below, so the majority
    // tier each pass skips never pays the view-pos math (pure reorder — the discard needs
    // only pRaw; the reconstruction is side-effect-free).
    const isV = pRaw.shiftRight(uint(31)).bitAnd(uint(1)).toVar();
    // procedural grass marker: bits 31|30 (voxel ids never set bit30; mesh ids never
    // set bit31) — grass pixels ride the vox-side pass via the isV partition above.
    const grassProcOn = !!world.grassProc && pass !== 'tri';
    const isGP: NB | null = grassProcOn
      ? (pRaw.shiftRight(uint(30)).equal(uint(3)).toVar() as unknown as NB)
      : null;
    if (pass === 'tri') {
      // MAIN pass: skip voxel-winner pixels (the 'vox' pass shades them). On a pure-triangle
      // world (no voxel queue) bit31 is never set, so this never fires.
      If(isV.equal(uint(1)), () => {
        Discard();
      });
    } else if (pass === 'vox') {
      // VOXEL pass: skip every pixel that is NOT a voxel winner (the 'tri' pass shades them).
      If(isV.equal(uint(0)), () => {
        Discard();
      });
    }
    // 24-bit depth in the high bits (8-bit id tiebreak below) ⇒ cz = 1 − (key>>8)/16777215
    const zDev = float(1).sub(toF(elect.shiftRight(uint(8))).div(16777215)) as unknown as NF;
    const wpv = getViewPosition(screenUV, zDev, cameraProjectionMatrixInverse) as unknown as NV3;
    const wp = (
      (cameraWorldMatrix as unknown as { mul(v: NV4): NV4 }).mul(
        (vec4 as unknown as (a: NV3, b: number) => NV4)(wpv, 1),
      ) as unknown as NV4
    ).xyz.toVar() as unknown as NV3;
    // (instId, ci): the 'tri' pass decodes from qRasterRO (the unchanged triangle path); the
    // 'vox' pass decodes from qVoxRaster (low 30 bits = the qVoxRaster item index → the voxel
    // cluster's (instId, ci); its word6 = brickBase points at gpu.voxelBricks, §4.1). The two
    // decodes are in mutually-exclusive branches, so only ONE of qRasterRO / qVoxRasterRO is
    // referenced per material build ⇒ each pass binds only its own queue. The 'both' pass
    // branches per PIXEL on isV instead (same partition, same decodes — both queues bound).
    let instId: NU;
    let ci: NU;
    if (pass === 'tri') {
      const triItemIdx = pRaw.shiftRight(uint(CLUSTER_TRI_BITS)).toVar();
      const triItem = cull.qRasterRO.element(triItemIdx.add(uint(1)));
      instId = triItem.x.toVar();
      ci = triItem.y.toVar();
    } else if (pass === 'vox') {
      const qVox = cull.qVoxRasterRO;
      if (!qVox) {
        // grass-only vox pass (no voxel queue in this config): every surviving pixel
        // is a procedural-grass winner; there is no (instId, ci) to decode. The
        // constants keep the shared decode chain shaped (matClass is forced 255 for
        // grass pixels below, so no mesh subgraph ever fires on them).
        if (!world.grassProc)
          throw new Error('NaniteResolve: vox pass built without qVoxRasterRO');
        instId = uint(0).toVar();
        ci = uint(0).toVar();
      } else {
        // item index = bits 0-20 (QVOX_CAP = 2^21). Bits 21-27 carry the winning BRICK index
        // under ?voxbn (default on; zero when off — masking is safe in both modes).
        const voxIdx = pRaw.bitAnd(uint(0x1fffff)).toVar();
        const voxItem = qVox.element(voxIdx.add(uint(1)));
        instId = voxItem.x.toVar();
        ci = voxItem.y.toVar();
      }
    } else {
      const qVox = cull.qVoxRasterRO;
      if (!qVox) throw new Error('NaniteResolve: both pass built without qVoxRasterRO');
      const instIdV = uint(0).toVar();
      const ciV = uint(0).toVar();
      If(isV.equal(uint(0)), () => {
        const triItemIdx = pRaw.shiftRight(uint(CLUSTER_TRI_BITS));
        const triItem = cull.qRasterRO.element(triItemIdx.add(uint(1)));
        instIdV.assign(triItem.x);
        ciV.assign(triItem.y);
      }).Else(() => {
        const voxIdx = pRaw.bitAnd(uint(0x1fffff));
        const voxItem = qVox.element(voxIdx.add(uint(1)));
        instIdV.assign(voxItem.x);
        ciV.assign(voxItem.y);
      });
      instId = instIdV as unknown as NU;
      ci = ciV as unknown as NU;
    }
    const meshId = elemU(gpu.clusters, ci.mul(uint(CLUSTER_WORDS)).add(uint(7))).shiftRight(uint(16));
    // (localTri = pRaw & CLUSTER_TRI_MASK is read inside the rock/bark/leaf branches; those
    // branches run ONLY in the 'tri' pass, where pRaw IS a triangle id. In the 'vox' pass the
    // low bits are the qVoxRaster index and the explicit-mesh branches are not built at all.)
    const matClass = elemU(gpu.meshes, meshId.mul(uint(MESH_WORDS)).add(uint(6)))
      .shiftRight(uint(8))
      .bitAnd(uint(0xff))
      .toVar();
    // procedural-grass pixels: the id is a blade, not a work item — the (instId, ci)
    // above decoded garbage (in-bounds, harmless). Force a sentinel matClass so NO
    // mesh-class subgraph (terrain/rock/bark/leaf/voxel) can fire on them; the grass
    // block below owns their albedo/normal/ao outright.
    if (isGP) {
      If(isGP, () => {
        matClass.assign(uint(255));
      });
    }
    const item = { x: instId, y: ci } as unknown as { x: NU; y: NU };
    const isT = matClass.equal(uint(0));

    // ---- TERRAIN shading on the reconstructed surface — GATED on isT (UE5-gap win #4):
    // non-terrain pixels skip the terrain prelude tex fetches + shading. The isT.select
    // below already discards this for non-terrain pixels, so the default (vec3(0.3) /
    // up-normal) reaching the select changes nothing ⇒ BIT-IDENTICAL. Mirrors the
    // isR/isBD/isL gating. (roughnessNode was computed-then-void'd/unused — dropped.)
    const camPos = vec3(cam.camPos) as unknown as NV3;
    const terrainCol = vec3(0.3).toVar() as unknown as NV3;
    const terrainNrm = vec3(0, 1, 0).toVar() as unknown as NV3;
    // The ROCK/BARK/LEAF material branches re-fetch the cluster triangle (gpu.verts /
    // gpu.indices) — the tri-only buffers the 'vox' pass must NOT bind. Each of those three
    // If() blocks is therefore guarded by `pass === 'tri'` below: in the 'vox' pass they would
    // never fire anyway (matClass=voxel(7) ⇒ isR/isBD/isL all false), and skipping their
    // CONSTRUCTION is what keeps verts/indices out of the voxel material's binding set. The
    // const declarations stay at top-level so the shared mux + lighting compile in BOTH passes.
    // TERRAIN reads no storage buffer, so it was previously left unguarded as "harmless" — but it
    // is NOT free in the 'vox' pass: buildTerrainShading's implicit-derivative texture() samples are
    // the vox shader's ONLY demote-forcing op, and the whole subgraph (~14 samples + fbm + caustics)
    // inflates register/instruction pressure → collapsed occupancy → the per-pixel voxel-decode
    // latency chain can't be hidden. Measured as the dominant driver of the close-up voxel r.scene
    // cliff (37.5ms inside a crown). Voxel pixels are never matClass 0 (isT always false in 'vox'),
    // so guarding by `pass === 'tri'` is output-identical and strips the graph from the vox shader.
    if (pass !== 'vox' && hasClass(0)) If(isT, () => {
      const shading = buildTerrainShading({
        normalTex: hf.normalTex,
        biomeTex: hf.biomeTex as StorageTexture,
        fieldsTex: hf.fieldsTex as StorageTexture,
        noiseA: hf.noiseA as StorageTexture,
        noiseB: hf.noiseB as StorageTexture,
        mp: hf.mp,
        far: false,
        surf: { wp, camPos },
      });
      let tc: NV3 = shading.colorNode;
      const cctx = causticContext();
      if (cctx) {
        const d = causticDepth(wp);
        const fringe = smoothstep(-0.45, -0.04, d);
        const caust = causticTint(wp, d);
        const biofilm = smoothstep(0.04, 0.5, d);
        let wetCol = tc
          .mul(fringe.mul(0.38).oneMinus())
          .mul(biofilm.mul(0.42).oneMinus()) as unknown as NV3;
        wetCol = mix(wetCol, wetCol.mul(vec3(0.72, 0.86, 0.55)), biofilm.mul(0.65)) as unknown as NV3;
        tc = wetCol.mul(caust.mul(1.7).add(1)) as unknown as NV3;
      }
      terrainCol.assign(tc);
      terrainNrm.assign(shading.worldNormalNode);
    });

    // ---- ROCK shading (N4-C2): re-fetch the cluster triangle, barycentric-
    // interpolate vdata + normal at the reconstructed surface point, run the
    // ported rockMaterial. Gated on isR so terrain (heightfield clusters, no
    // explicit verts) never enters the explicit-mesh fetch.
    const isR = matClass.equal(uint(1)).toVar();
    const rockCol = vec3(0.3).toVar() as unknown as NV3;
    const rockNrm = vec3(0, 1, 0).toVar() as unknown as NV3;
    const rockAo = float(1).toVar() as unknown as NF;
    if (pass !== 'vox' && hasClass(1)) If(isR, () => {
      const instId = item.x;
      const localTri = pRaw.bitAnd(uint(CLUSTER_TRI_MASK));
      const ctx = fetch.makeCtx(instId, ci);
      const w0 = fetch.fetchWorldVert(ctx, localTri, 0);
      const w1 = fetch.fetchWorldVert(ctx, localTri, 1);
      const w2 = fetch.fetchWorldVert(ctx, localTri, 2);
      const bw = baryWeights(wp, w0, w1, w2);
      const tb = ctx.triStart.add(localTri).mul(uint(3));
      const a = readVertex(gpu.verts, elemU(gpu.indices, tb));
      const b = readVertex(gpu.verts, elemU(gpu.indices, tb.add(uint(1))));
      const c = readVertex(gpu.verts, elemU(gpu.indices, tb.add(uint(2))));
      const dv = unpackVdata(a.vdata)
        .mul(bw.x)
        .add(unpackVdata(b.vdata).mul(bw.y))
        .add(unpackVdata(c.vdata).mul(bw.z)) as unknown as NV4;
      const nrm = normalize(
        instRotateDir(ctx.yawSc, a.nrm)
          .mul(bw.x)
          .add(instRotateDir(ctx.yawSc, b.nrm).mul(bw.y))
          .add(instRotateDir(ctx.yawSc, c.nrm).mul(bw.z)),
      ) as unknown as NV3;
      const rk = rockShade(dv, wp, nrm);
      rockCol.assign(rk.albedo);
      rockNrm.assign(nrm);
      rockAo.assign(rk.ao);
    });

    // ---- BARK + DEADWOOD shading (N4-C3): textured trunks/snags. Same
    // explicit-mesh fetch as rock, plus per-vertex UV interpolation, a tangent
    // frame from the triangle edges for the normal map, analytic UV gradients
    // (neighbour-pixel rays intersected with THIS triangle's plane → no
    // silhouette mip spike), and the bark texture-ARRAY sampled at the per-mesh
    // layer slice (mesh word 7). Diffuse-only (roughness unused, like terrain/
    // rock). Trunk WIND rides in via fetchWorldVert at the C3 second commit.
    const isB = matClass.equal(uint(2)).toVar();
    const isD = matClass.equal(uint(3)).toVar();
    const isBD = isB.or(isD).toVar();
    const barkCol = vec3(0.3).toVar() as unknown as NV3;
    const barkNrm = vec3(0, 1, 0).toVar() as unknown as NV3;
    const barkAo = float(1).toVar() as unknown as NF;
    if (pass !== 'vox' && world.barkTexA && world.barkTexB && (hasClass(2) || hasClass(3))) {
      const barkTexA = world.barkTexA;
      const barkTexB = world.barkTexB;
      If(isBD, () => {
        const instId = item.x;
        const localTri = pRaw.bitAnd(uint(CLUSTER_TRI_MASK));
        const ctx = fetch.makeCtx(instId, ci);
        const w0 = fetch.fetchWorldVert(ctx, localTri, 0);
        const w1 = fetch.fetchWorldVert(ctx, localTri, 1);
        const w2 = fetch.fetchWorldVert(ctx, localTri, 2);
        const bw = baryWeights(wp, w0, w1, w2);
        const tb = ctx.triStart.add(localTri).mul(uint(3));
        const va = readVertex(gpu.verts, elemU(gpu.indices, tb));
        const vb = readVertex(gpu.verts, elemU(gpu.indices, tb.add(uint(1))));
        const vc = readVertex(gpu.verts, elemU(gpu.indices, tb.add(uint(2))));
        const uvv = va.uv
          .mul(bw.x)
          .add(vb.uv.mul(bw.y))
          .add(vc.uv.mul(bw.z)) as unknown as NV2;
        const dv = unpackVdata(va.vdata)
          .mul(bw.x)
          .add(unpackVdata(vb.vdata).mul(bw.y))
          .add(unpackVdata(vc.vdata).mul(bw.z)) as unknown as NV4;
        const gnrm = normalize(
          instRotateDir(ctx.yawSc, va.nrm)
            .mul(bw.x)
            .add(instRotateDir(ctx.yawSc, vb.nrm).mul(bw.y))
            .add(instRotateDir(ctx.yawSc, vc.nrm).mul(bw.z)),
        ) as unknown as NV3;

        // tangent frame (world-space): T along +U, Gram-Schmidt vs the normal,
        // Bi = N×T. Edge/uv-delta solve (Lengyel) — bark texB.xy perturbs it.
        const e1 = w1.sub(w0);
        const e2 = w2.sub(w0);
        const du1 = vb.uv.x.sub(va.uv.x);
        const dq1 = vb.uv.y.sub(va.uv.y);
        const du2 = vc.uv.x.sub(va.uv.x);
        const dq2 = vc.uv.y.sub(va.uv.y);
        const r = float(1).div(du1.mul(dq2).sub(du2.mul(dq1)).add(1e-8));
        const Traw = e1.mul(dq2).sub(e2.mul(dq1)).mul(r) as unknown as NV3;
        const Braw = e2.mul(du1).sub(e1.mul(du2)).mul(r) as unknown as NV3;
        const T = normalize(Traw.sub(gnrm.mul(dot(gnrm, Traw)))) as unknown as NV3;
        const Bi = normalize(cross(gnrm, T)) as unknown as NV3;

        // analytic mip LOD (NaN-proof, isotropic): world size of one screen
        // pixel at the surface vs world size of one bark texel. |Traw|/|Braw| =
        // world metres per uv unit; bark tiles once per uv unit (BARK_RES texels/
        // unit). Conservative axis (min world-per-texel) to anti-alias. The
        // hardware auto-mip is unusable here (uv is computed in non-uniform
        // control flow → undefined derivatives); anisotropic .grad() is future
        // work (?nanbark=grad — the ray-plane neighbour path NaNs on near trunks).
        const C = vec3(cam.camPos) as unknown as NV3;
        const dist = wp.sub(C).length();
        const pixWorld = dist.mul(2).div(float(cam.cotHalfFov).mul(float(cam.uH)));
        const wPerTexel = Traw.length().min(Braw.length()).div(BARK_RES).max(1e-6);
        const lod = pixWorld.div(wPerTexel).max(1e-4).log2().max(0);
        const planeN = normalize(cross(e1, e2)) as unknown as NV3;
        const rayDir = (suv: NV2): NV3 => {
          const vpN = getViewPosition(suv, zDev, cameraProjectionMatrixInverse) as unknown as NV3;
          const wd = (
            (cameraWorldMatrix as unknown as { mul(v: NV4): NV4 }).mul(
              (vec4 as unknown as (a: NV3, b: number) => NV4)(vpN, 0),
            ) as unknown as NV4
          ).xyz as unknown as NV3;
          return normalize(wd) as unknown as NV3;
        };
        const uvAt = (dir: NV3): NV2 => {
          const tt = dot(planeN, w0.sub(C)).div(dot(planeN, dir).add(1e-8));
          const bh = baryWeights(C.add(dir.mul(tt)) as unknown as NV3, w0, w1, w2);
          return va.uv.mul(bh.x).add(vb.uv.mul(bh.y)).add(vc.uv.mul(bh.z)) as unknown as NV2;
        };

        const layer = int(fetch.meshWord(ctx.meshId, 7).bitAnd(uint(0xff)));
        // ?nanbark=const — flat brown, no texture (fetch/branch sanity)
        if (nanbark === 'const') {
          barkCol.assign(vec3(0.4, 0.25, 0.13) as unknown as NV3);
          barkNrm.assign(gnrm);
          barkAo.assign(float(1) as unknown as NF);
          return;
        }
        // ?nanbark=lN — force mip level N (inspect the generated chain)
        const lvlMatch = nanbark ? /^l(\d+)$/.exec(nanbark) : null;
        const sample = (t: Texture): NV4 => {
          const base = texture(t, uvv as never) as unknown as TexSample;
          if (lvlMatch)
            return (base.depth(layer) as unknown as { level(n: number): NV4 }).level(
              Number(lvlMatch[1]),
            );
          if (nanbark === 'grad') {
            const dUVdx = uvAt(
              rayDir(screenUV.add(vec2(float(1).div(cam.uW), 0)) as unknown as NV2),
            ).sub(uvv) as unknown as NV2;
            const dUVdy = uvAt(
              rayDir(screenUV.add(vec2(0, float(1).div(cam.uH))) as unknown as NV2),
            ).sub(uvv) as unknown as NV2;
            return base.depth(layer).grad(dUVdx, dUVdy) as unknown as NV4;
          }
          // DEFAULT: analytic isotropic mip LOD (NaN-proof)
          return (base.depth(layer) as unknown as { level(n: unknown): NV4 }).level(lod);
        };
        const tA = sample(barkTexA);

        // albedo: sqrt-decoded texture, bark (hue+cavity) vs deadwood (dim+moss+rot)
        const tex = tA.rgb.mul(tA.rgb) as unknown as NV3;
        const barkAlb = hueShift(tex, dv.x, 0.14).mul(dv.w.mul(0.45).add(0.55)) as unknown as NV3;
        // ?resfar (default 60 m, 0 = off): DISTANCE-GATED micro-detail. Beyond the gate the
        // 3-octave fbm moss noise and the tangent-frame + normal-map perturbation are
        // sub-texel (<1 px) — pure per-pixel ALU with no visible contribution. The pixel-
        // scaling law (2026-07-02) makes per-pixel cost 60-73% of the frame, so far bark
        // pixels take the cheap side: geometric normal + no moss. Branches are distance-
        // coherent on screen (low divergence).
        const detailNear = resFarDist > 0 ? dist.lessThan(float(resFarDist)) : null;
        // deadwood dim (logDim, representative — energy-correct, not per-pool)
        let deadAlb = tex.mul(vec3(0.6, 0.52, 0.44)) as unknown as NV3;
        const moss = float(0).toVar();
        const mossBody = (): void => {
          const mossN = smoothstep(0.24, 0.58, fbm3(wp.mul(2.6), 3).mul(0.5).add(0.5));
          moss.assign(smoothstep(0.05, 0.65, gnrm.y).mul(dv.z).mul(mossN).clamp(0, 1));
        };
        if (detailNear) If(detailNear, mossBody);
        else mossBody();
        deadAlb = mix(deadAlb, vec3(0.05, 0.1, 0.032), moss) as unknown as NV3;
        deadAlb = deadAlb.mul(float(1).sub(dv.z.mul(0.25))) as unknown as NV3; // rot
        deadAlb = hueShift(deadAlb, dv.x, 0.1) as unknown as NV3;

        // tangent-space normal map (three normalMap: n = tex·2−1, z kept = 1) — near only
        const pert = (vec3(gnrm.x, gnrm.y, gnrm.z) as unknown as NV3).toVar();
        const pertBody = (): void => {
          const tB = sample(barkTexB); // the normal-map sample is paid on the near side only
          pert.assign(
            normalize(
              T.mul(tB.x.mul(2).sub(1))
                .add(Bi.mul(tB.y.mul(2).sub(1)))
                .add(gnrm),
            ) as unknown as NV3,
          );
        };
        if (detailNear) If(detailNear, pertBody);
        else pertBody();

        // AUDIT-1a: per-instance warm/cool + value jitter (slotHash 17/91) — the
        // variation law the old path applied via applyInstanceTint (tintK 0.12).
        // Without it a mesh's ~4k instances share one colour (the original's
        // "migration clones trees" — banned). Same math, keyed on the persistent
        // scatter slot (instId), on TOP of the per-vertex hueShift above.
        const tK = 0.12;
        const h1 = slotHash(instId, 17);
        const h2 = slotHash(instId, 91);
        const warmCool = mix(
          vec3(1 + tK, 1, 1 - tK * 0.8),
          vec3(1 - tK * 0.8, 1, 1 + tK),
          h1,
        ) as unknown as NV3;
        const tintVal = h2.mul(tK * 1.6).add(1 - tK * 0.8);
        barkCol.assign((isD.select(deadAlb, barkAlb) as unknown as NV3).mul(warmCool).mul(tintVal));
        barkNrm.assign(pert);
        barkAo.assign(tA.w as unknown as NF);
      });
    }

    // ---- LEAF shading (N9-C0): the real mesh-leaf crown, OPAQUE + DOUBLE-SIDED.
    // Port of VegMaterials.foliageMaterial: per-species tint (matParam, packed
    // linear RGB + hueVar) × per-leaf hue jitter (vdata.x) × crown-depth AO
    // (vdata.w); a warm translucent BACKLIGHT is added to `lit` below; NO specular.
    // Same explicit-mesh fetch as bark, minus UV/TBN/texture (leaves have no detail
    // map). The geometric normal is FLIPPED to face the camera (two-sided lighting).
    const isL = matClass.equal(uint(4)).toVar();
    const leafCol = vec3(0.1, 0.2, 0.08).toVar() as unknown as NV3;
    const leafNrm = vec3(0, 1, 0).toVar() as unknown as NV3;
    if (pass !== 'vox' && hasClass(4)) If(isL, () => {
      const instId = item.x;
      const localTri = pRaw.bitAnd(uint(CLUSTER_TRI_MASK));
      // per-species tint from matParam (mesh word 7): linear RGB + hueVar (4×u8) —
      // needed by BOTH the full and the ?resfar cheap path (meshId is already decoded).
      const mp = fetch.meshWord(meshId, 7);
      const base = vec3(
        toF(mp.bitAnd(uint(0xff))),
        toF(mp.shiftRight(uint(8)).bitAnd(uint(0xff))),
        toF(mp.shiftRight(uint(16)).bitAnd(uint(0xff))),
      ).div(255) as unknown as NV3;
      const fullLeaf = (): void => {
        const ctx = fetch.makeCtx(instId, ci);
        const w0 = fetch.fetchWorldVert(ctx, localTri, 0);
        const w1 = fetch.fetchWorldVert(ctx, localTri, 1);
        const w2 = fetch.fetchWorldVert(ctx, localTri, 2);
        const bw = baryWeights(wp, w0, w1, w2);
        const tb = ctx.triStart.add(localTri).mul(uint(3));
        const va = readVertex(gpu.verts, elemU(gpu.indices, tb));
        const vb = readVertex(gpu.verts, elemU(gpu.indices, tb.add(uint(1))));
        const vc = readVertex(gpu.verts, elemU(gpu.indices, tb.add(uint(2))));
        const dv = unpackVdata(va.vdata)
          .mul(bw.x)
          .add(unpackVdata(vb.vdata).mul(bw.y))
          .add(unpackVdata(vc.vdata).mul(bw.z)) as unknown as NV4;
        const hueVar = toF(mp.shiftRight(uint(24)).bitAnd(uint(0xff))).div(255);
        // hueShift with a NODE amount (per-species hueVar) — inline of the resolve
        // hueShift helper, whose `amount` is a compile-time constant for bark.
        const k = (dv.x as unknown as NF).mul(hueVar);
        const tintedHue = base
          .mul(vec3(1.18, 1.0, 0.55))
          .mul(k.clamp(0, 1))
          .add(base.mul(vec3(0.7, 0.95, 1.25)).mul(k.negate().clamp(0, 1)))
          .add(base.mul(float(1).sub(k.abs()))) as unknown as NV3;
        leafCol.assign(tintedHue.mul(dv.w.mul(0.8).add(0.2)) as unknown as NV3);
        // instance-rotated geometric normal, flipped to face the camera (two-sided)
        const gnrm = normalize(
          instRotateDir(ctx.yawSc, va.nrm)
            .mul(bw.x)
            .add(instRotateDir(ctx.yawSc, vb.nrm).mul(bw.y))
            .add(instRotateDir(ctx.yawSc, vc.nrm).mul(bw.z)),
        ) as unknown as NV3;
        const toCam = normalize(camPos.sub(wp)) as unknown as NV3;
        leafNrm.assign(dot(gnrm, toCam).lessThan(0).select(gnrm.negate(), gnrm) as unknown as NV3);
      };
      if (resFarDist > 0 || leafCheapAll) {
        // ?resfar cheap FAR-leaf path (beyond resfar·0.6 ≈ 36 m by default; the voxel band
        // takes over at 45 m with flat brick shading anyway): species tint × mid crown-AO +
        // the leaf QUAD's single-vertex normal (leaves are flat quads — va.nrm ≈ the face),
        // no makeCtx (its gust samples + instance decode were paid PER PIXEL), no wind, no
        // 3-vertex interp. Matches the voxel handoff look; per-leaf hue jitter is ≤2 px there.
        const distL = wp.sub(vec3(camPos) as unknown as NV3).length();
        If(distL.greaterThan(float(leafCheapAll ? -1 : resFarDist * 0.6)), () => {
          const triStart = elemU(gpu.clusters, ci.mul(uint(CLUSTER_WORDS)).add(uint(6)));
          const vi = elemU(gpu.indices, triStart.add(localTri).mul(uint(3)));
          const va = readVertex(gpu.verts, vi);
          const B = gpu.instances.element(instId.mul(uint(2)).add(uint(1))) as unknown as NV4;
          const yawSc = instYaw(B);
          const gn = normalize(instRotateDir(yawSc, va.nrm)) as unknown as NV3;
          const toCam = normalize(camPos.sub(wp)) as unknown as NV3;
          leafNrm.assign(dot(gn, toCam).lessThan(0).select(gn.negate(), gn) as unknown as NV3);
          leafCol.assign(base.mul(0.68) as unknown as NV3);
        }).Else(fullLeaf);
      } else {
        fullLeaf();
      }
    });

    // ---- GRASS shading (S0, 31-grass-plan §5): matClass 5 — the GroundRing blade
    // material ported to the resolve. ALU + explicit-LOD taps only (the vox-cliff
    // rule: no implicit-derivative samples, no new storage buffers). Blade rounded
    // normal pulled toward the TERRAIN normal, harder with distance, so swards
    // light like their hillside (the GoT move; per-blade card normals sparkle).
    // Albedo = fresh/dry tip ramps × world-anchored ~1.6 m patch dryness ×
    // canopy shade-darkening (dry straw is a full-sun phenomenon).
    // Smooth ~1.6 m patch field (user: yellow dryness in PERFECT SQUARES ruins
    // immersion — the old floor() cell hash cut hard 1.6 m boundaries). Value
    // noise: 4 corner hashes + smoothstep-fade bilinear, domain rotated ~40° so
    // the lattice never reads axis-aligned. Coverage recalibrated (Monte-Carlo):
    // dryK = smoothstep(0.64, 0.82, x) ≈ the old smoothstep(0.7, 0.95, hash)
    // mean/rms; brightness drift scale 0.3 → 0.4 keeps the old std (smooth
    // interpolation compresses the extremes). x = dryness, y = brightness.
    const patchField = (xz: NV2): NV2 => {
      const p = vec2(
        (xz.x as unknown as NF).mul(0.766).sub((xz.y as unknown as NF).mul(0.643)),
        (xz.x as unknown as NF).mul(0.643).add((xz.y as unknown as NF).mul(0.766)),
      ).mul(1 / 1.6) as unknown as NV2;
      const ip = floor(p) as unknown as NV2;
      const fp = fract(p) as unknown as NV2;
      const u = fp.mul(fp).mul(fp.mul(-2).add(3)) as unknown as NV2;
      const h = (c: NV2): NV2 =>
        fract(
          sin(
            vec2(
              dot(c as unknown as NV3, vec2(127.1, 311.7) as unknown as NV3),
              dot(c as unknown as NV3, vec2(269.5, 183.3) as unknown as NV3),
            ),
          ).mul(vec2(43758.5453, 28461.7331) as unknown as NV2),
        ) as unknown as NV2;
      const hx = mix(h(ip), h(ip.add(vec2(1, 0)) as unknown as NV2), u.x) as unknown as NV2;
      const hy = mix(
        h(ip.add(vec2(0, 1)) as unknown as NV2),
        h(ip.add(vec2(1, 1)) as unknown as NV2),
        u.x,
      ) as unknown as NV2;
      return mix(hx, hy, u.y) as unknown as NV2;
    };
    const isG = matClass.equal(uint(5)).toVar();
    const grassCol = vec3(0.04, 0.09, 0.02).toVar() as unknown as NV3;
    const grassNrm = vec3(0, 1, 0).toVar() as unknown as NV3;
    const grassTip = float(0.5).toVar() as unknown as NF;
    if (pass !== 'vox' && hasClass(5)) If(isG, () => {
      const instId = item.x;
      const localTri = pRaw.bitAnd(uint(CLUSTER_TRI_MASK));
      const distG = wp.sub(vec3(camPos) as unknown as NV3).length();
      const t = float(0.5).toVar() as unknown as NF;
      const nG = vec3(0, 1, 0).toVar() as unknown as NV3;
      // tip param (uv.y) + normal: 3-vert bary interp near (the rounded
      // cross-section reads as a half-cylinder), single-vertex beyond 30 m
      // (the ?resfar cheap-path law — sub-pixel-width blades shade the same).
      If(distG.lessThan(float(30)), () => {
        const ctx = fetch.makeCtx(instId, ci);
        const w0 = fetch.fetchWorldVert(ctx, localTri, 0);
        const w1 = fetch.fetchWorldVert(ctx, localTri, 1);
        const w2 = fetch.fetchWorldVert(ctx, localTri, 2);
        const bw = baryWeights(wp, w0, w1, w2);
        const tb = ctx.triStart.add(localTri).mul(uint(3));
        const va = readVertex(gpu.verts, elemU(gpu.indices, tb));
        const vb = readVertex(gpu.verts, elemU(gpu.indices, tb.add(uint(1))));
        const vc = readVertex(gpu.verts, elemU(gpu.indices, tb.add(uint(2))));
        t.assign(
          (va.uv.y as unknown as NF)
            .mul(bw.x)
            .add((vb.uv.y as unknown as NF).mul(bw.y))
            .add((vc.uv.y as unknown as NF).mul(bw.z)) as unknown as NF,
        );
        nG.assign(
          normalize(
            instRotateDir(ctx.yawSc, va.nrm)
              .mul(bw.x)
              .add(instRotateDir(ctx.yawSc, vb.nrm).mul(bw.y))
              .add(instRotateDir(ctx.yawSc, vc.nrm).mul(bw.z)),
          ) as unknown as NV3,
        );
      }).Else(() => {
        const triStart = elemU(gpu.clusters, ci.mul(uint(CLUSTER_WORDS)).add(uint(6)));
        const va = readVertex(gpu.verts, elemU(gpu.indices, triStart.add(localTri).mul(uint(3))));
        const B = gpu.instances.element(instId.mul(uint(2)).add(uint(1))) as unknown as NV4;
        t.assign(va.uv.y as unknown as NF);
        nG.assign(normalize(instRotateDir(instYaw(B), va.nrm)) as unknown as NV3);
      });
      // two-sided: flip camera-ward, then pull toward the terrain normal
      const toCamG = normalize(camPos.sub(wp)) as unknown as NV3;
      const nF = dot(nG, toCamG).lessThan(0).select(nG.negate(), nG) as unknown as NV3;
      const tNrm = (
        texture(hf.normalTex, hf.uvFromWorld(wp.xz as unknown as NV2), 0) as unknown as NV4
      ).xyz.normalize() as unknown as NV3;
      const upK = smoothstep(8, 70, distG).mul(0.35).add(0.5) as unknown as NF;
      grassNrm.assign(normalize(mix(nF, tNrm, upK)) as unknown as NV3);
      const fresh = mix(
        vec3(0.02, 0.062, 0.011),
        vec3(0.065, 0.148, 0.028),
        t.mul(t),
      ) as unknown as NV3;
      const dryC = mix(vec3(0.085, 0.07, 0.024), vec3(0.21, 0.17, 0.075), t) as unknown as NV3;
      // world-anchored smooth ~1.6 m patch field (stable under camera motion,
      // TAA-safe): x = dryness drift, y = brightness drift
      const patch = patchField(wp.xz as unknown as NV2).toVar() as unknown as NV2;
      const patchX = patch.x as unknown as NF;
      const patchY = patch.y as unknown as NF;
      const cov = (world.canopyTex
        ? canopyAt(world.canopyTex, wp.xz as unknown as NV2)
        : float(0)) as unknown as NF;
      const dryK = smoothstep(0.64, 0.82, patchX).mul(float(1).sub(cov.mul(0.85))) as unknown as NF;
      let alb = mix(fresh, dryC, dryK) as unknown as NV3;
      alb = alb.mul(patchY.sub(0.5).mul(0.4).add(1)) as unknown as NV3;
      alb = mix(alb, vec3(0.018, 0.052, 0.014) as unknown as NV3, cov.mul(0.55)) as unknown as NV3;
      grassCol.assign(alb);
      grassTip.assign(t);
    });

    // ---- VOXEL shading (Stage 2 §7.2): matClass=voxel(7). The SECOND resolve pass shades
    // ONLY voxel-winner pixels (the 'tri' pass Discarded them). Reuses the SAME reconstructed
    // wp (no new depth math), decodes the BRICK-MEAN normal from gpu.voxelBricks (the coarse
    // one-sample-per-brick default §6.4; word3 spread is carried for a future SGGX fallback but
    // the mean-normal path ignores it, §7.2.3), rotates it by the instance yaw + flips it
    // camera-ward via the EXISTING leaf idiom (voxels are inherently two-sided), and colors via
    // the leaf matParam TINT (§7.2.6 — brick word4 albedo is raster/build-only). Built ONLY in
    // the 'vox' pass, so voxelBricks/qVoxRasterRO are referenced ONLY there.
    // GRAY-SLAB FIX (§7.2): in the 'vox' pass EVERY surviving pixel is a bit31 voxel
    // winner (non-voxel pixels were Discarded at the isV gate above), so the foliage
    // shade + default MUST be gated on `isV` (the bit31 marker), NOT on `isVox`
    // (matClass==7). A stale/garbage visBV id can decode a matClass != 7 even on a real
    // voxel pixel; gating the fall-through on `isVox` then dropped that pixel to the gray
    // `palette` ⇒ the GRAY SLABS. Gating on `isV` keeps every vox-pass pixel reading as
    // FOLIAGE (the brick-mean normal when the decode is clean, the dark-green foliage
    // default otherwise) — never gray. The 'tri' pass is unchanged (its grass/debris
    // fall-through still uses `palette`; voxel pixels were already Discarded there).
    const isVox = matClass.equal(uint(7)).toVar();
    const voxCol = vec3(0.1, 0.2, 0.08).toVar() as unknown as NV3;
    const voxNrm = vec3(0, 1, 0).toVar() as unknown as NV3;
    if (pass !== 'tri' && cull.qVoxRasterRO) {
      If(isVox, () => {
        // ?voxao= (default ON): the per-brick DIRECTIONAL self-shading. Decode the BAKED brick-mean
        // normal, rotate by instance yaw, flip camera-ward — it then drives the sun N·L + ambient
        // floor below, giving the darkening on faces angled away from the sun. =0 SKIPS this block
        // (drops the gpu.voxelBricks normal read) so voxNrm stays the flat up-normal ⇒ flat-lit.
        // ?voxbn: the winning BRICK index rides visBV bits 21-27 (0 when the flag is off in
        // the raster ⇒ brickBase+0 = the legacy block-representative decode, byte-identical).
        const brickSel = voxBrickShade
          ? pRaw.shiftRight(uint(21)).bitAnd(uint(0x7f)).toVar()
          : uint(0).toVar();
        const brickBase = elemU(gpu.clusters, ci.mul(uint(CLUSTER_WORDS)).add(uint(6))).toVar();
        const bi = brickBase.add(brickSel).toVar();
        if (voxShade) {
          const vInstId = item.x;
          const vB = gpu.instances.element(vInstId.mul(uint(2)).add(uint(1))).toVar() as unknown as NV4;
          const yawSc = instYaw(vB);
          // per-brick mean normal (word BRICK_NORMAL of the WINNING brick under ?voxbn;
          // brick[brickBase] block-representative otherwise, §6.4).
          const nrmWord = elemU(gpu.voxelBricks, brickWord(bi, uint(BRICK_NORMAL)));
          const localN = brickNormalTsl(nrmWord) as unknown as NV3;
          const gn = normalize(instRotateDir(yawSc, localN)) as unknown as NV3;
          const toCamV = normalize(camPos.sub(wp)) as unknown as NV3;
          if (voxBeadK > 0) {
            // ?voxbead v2: LOW-FREQUENCY crown field. v1 bent toward wp−brickCenter — a
            // per-brick radial field that made every brick a bright-tip/dark-flank facet;
            // rows of them read as SHARP CONES (user report). The SpeedTree trick works at
            // TREE scale: one smooth outward field per crown, so neighboring bricks shade
            // alike and the whole crown reads as a soft blob. Crown dir = horizontal
            // radial from the tree origin (A.xyz) + fixed up-tilt; a faint 25% within-
            // brick term keeps near-band rounding. Far-tile bricks ride identity
            // instances (origin 0) → world-anchored dir, ~constant per 64m tile.
            const vA = gpu.instances.element(vInstId.mul(uint(2))).toVar() as unknown as NV4;
            const bCtr = vec3(
              bcU2F(elemU(gpu.voxelBricks, brickWord(bi, uint(BRICK_POS_X)))),
              bcU2F(elemU(gpu.voxelBricks, brickWord(bi, uint(BRICK_POS_X + 1)))),
              bcU2F(elemU(gpu.voxelBricks, brickWord(bi, uint(BRICK_POS_X + 2)))),
            ) as unknown as NV3;
            const ctrW = instTransformPoint(vA, vB, yawSc, bCtr);
            const d0 = ctrW.sub(vA.xyz) as unknown as NV3;
            const horiz = vec3(d0.x, 0, d0.z) as unknown as NV3;
            // max(len, 0.05) guards crown-axis bricks (d0.xz≈0 → normalize NaN)
            const crownDir = normalize(
              horiz.div(max(horiz.length(), float(0.05))).add(vec3(0, 0.55, 0)),
            ) as unknown as NV3;
            const beadPix = normalize(wp.sub(ctrW)) as unknown as NV3;
            const bead = normalize(crownDir.mul(0.75).add(beadPix.mul(0.25))) as unknown as NV3;
            const blend = normalize(
              gn.mul(1 - voxBeadK).add(bead.mul(voxBeadK)),
            ) as unknown as NV3;
            voxNrm.assign(dot(blend, toCamV).lessThan(0).select(blend.negate(), blend) as unknown as NV3);
          } else {
            voxNrm.assign(dot(gn, toCamV).lessThan(0).select(gn.negate(), gn) as unknown as NV3);
          }
          // (the earlier 30% sunward normal wrap was REPLACED by proper WRAP LIGHTING on
          // the sun term below — bending the normal also skewed the ambient hemisphere
          // and still let fully-away crowns crash; see the nDotL wrap.)
          if (ftNrmK > 0) {
            // ?ftnrm (2026-07-02 beautification, default 0.65): FAR-TILE pixels only
            // (MESH_FLAG_FARTILE, word6 bits 16-23) — blend the brick mean normal toward
            // up. The splat-averaged tile normals carry a tile-pitch bias that N·L turns
            // into repeating dark bands (voxao=0 A/B proved the bands are 100% normal-
            // driven); a real canopy from >140 m reads near-lambertian-flat anyway.
            // Per-tree crowns (45-140 m) keep their full baked normals.
            // SLOPE-AWARE (2026-07-03, world hookup): the blend is weighted by how
            // up-facing the baked normal already is (clamp(n.y,0,1)) — flat-forest
            // canopy TOPS keep the full banding fix, but hillside tile WALLS (world
            // terrain; bricks seen side-on) keep their directional normals instead of
            // washing out into uniformly-lit pale plates (user report, w6-oblique).
            const mFlags = fetch.meshWord(meshId, 6).shiftRight(uint(16)).bitAnd(uint(0xff));
            If(mFlags.bitAnd(uint(MESH_FLAG_FARTILE)).notEqual(uint(0)), () => {
              const upK = float(ftNrmK).mul(voxNrm.y.clamp(0, 1));
              voxNrm.assign(
                normalize(mix(voxNrm, vec3(0, 1, 0) as unknown as NV3, upK)) as unknown as NV3,
              );
            });
          }
        }
        if (voxBrickShade) {
          // per-brick baked ALBEDO (BRICK_ALBEDO rgb of the winning brick) — breaks the
          // single-tint plate look; same 0.8 factor as the tint path so overall crown
          // brightness is unchanged.
          const albWord = elemU(gpu.voxelBricks, brickWord(bi, uint(BRICK_ALBEDO)));
          const bAlb = vec3(
            toF(albWord.bitAnd(uint(0xff))),
            toF(albWord.shiftRight(uint(8)).bitAnd(uint(0xff))),
            toF(albWord.shiftRight(uint(16)).bitAnd(uint(0xff))),
          ).div(255) as unknown as NV3;
          voxCol.assign(bAlb.mul(0.8) as unknown as NV3);
        } else {
          // leaf-tint color path (§7.2.6): mesh word7 = packed linear RGB + hueVar. No per-leaf
          // vdata jitter (no triangle) — use the mid tint (k=0 ⇒ base) × a mid crown-AO (0.6).
          const mp = fetch.meshWord(meshId, 7);
          const base = vec3(
            toF(mp.bitAnd(uint(0xff))),
            toF(mp.shiftRight(uint(8)).bitAnd(uint(0xff))),
            toF(mp.shiftRight(uint(16)).bitAnd(uint(0xff))),
          ).div(255) as unknown as NV3;
          voxCol.assign(base.mul(0.8) as unknown as NV3);
        }
        if (voxJitK > 0) {
          // ?voxjit: world-anchored per-cell value jitter (flag comment above) — breaks the
          // few-greens tiling without payload bits; ~0.7 m cells, camera-motion stable.
          const cellQ = floor(wp.mul(1.4)) as unknown as NV3;
          const h = fract(
            sin(dot(cellQ, vec3(12.9898, 78.233, 37.719) as unknown as NV3)).mul(43758.5453),
          ) as unknown as NF;
          voxCol.assign(voxCol.mul(h.mul(2 * voxJitK).add(1 - voxJitK)) as unknown as NV3);
        }
        if (voxJit2K > 0) {
          // ?voxjit2: clump-scale (~3 m) variation — see the knob comment above.
          const clumpQ = floor(wp.mul(0.34)) as unknown as NV3;
          const h2 = fract(
            sin(dot(clumpQ, vec3(41.017, 17.933, 91.381) as unknown as NV3)).mul(28461.7331),
          ) as unknown as NF;
          voxCol.assign(voxCol.mul(h2.mul(2 * voxJit2K).add(1 - voxJit2K)) as unknown as NV3);
        }
        if (voxGradK > 0) {
          // ?voxgrad (2026-07-02 beautification, default 0.35): CROWN-SCALE vertical light
          // gradient — tops brighten, undersides darken (fake sky-occlusion). Per-cube N·L
          // variance can't make box stacks read as foliage (user-confirmed); lighting
          // variance at CROWN scale is what sells the fluffy-blob look. World-anchored
          // (instance origin), TAA-stable, ~3 ALU. Height ramp saturates ~12.5 m up.
          const vInstIdG = item.x;
          const vAg = gpu.instances.element(vInstIdG.mul(uint(2))).toVar() as unknown as NV4;
          const hgt = wp.y.sub(vAg.y).mul(0.08).clamp(0, 1) as unknown as NF;
          voxCol.assign(
            voxCol.mul(hgt.mul(voxGradK).add(1 - voxGradK * 0.5)) as unknown as NV3,
          );
        }
      });
    }

    // unported explicit classes (grass/debris — N10) keep a flat gray; voxel pixels take
    // the innermost fall-through. In the 'vox' pass gate on `isV` (every surviving pixel
    // IS a voxel) so a mis-decoded matClass never drops to the gray slab; in the 'tri'
    // pass voxel pixels were Discarded, so `isVTri` is the original `isVox` (matClass==7),
    // leaving grass/debris on the gray `palette` exactly as before (loss-exact for tri).
    const palette = vec3(0.35, 0.33, 0.3) as unknown as NV3;
    const isVoxDefault = pass === 'tri' ? isVox : isV.equal(uint(1));
    const voxAlbDefault = isVoxDefault.select(voxCol, palette) as unknown as NV3;
    const voxNrmDefault = isVoxDefault.select(voxNrm, vec3(0, 1, 0)) as unknown as NV3;
    const albedo = isT
      .select(
        terrainCol,
        isR.select(
          rockCol,
          isBD.select(barkCol, isL.select(leafCol, isG.select(grassCol, voxAlbDefault))),
        ),
      )
      .toVar() as unknown as NV3;
    const wNormal = isT
      .select(
        terrainNrm,
        isR.select(
          rockNrm,
          isBD.select(barkNrm, isL.select(leafNrm, isG.select(grassNrm, voxNrmDefault))),
        ),
      )
      .toVar() as unknown as NV3;
    // aoNode (rock/bark cavity + grass tip-AO fake base self-shadow): indirect only
    const grassAo = smoothstep(0.0, 0.55, grassTip).mul(0.55).add(0.45) as unknown as NF;
    const ao = (isR.select(
      rockAo,
      isBD.select(barkAo, isG.select(grassAo, float(1))),
    ) as unknown as NF).toVar() as unknown as NF;

    // ---- PROCEDURAL GRASS shading (grass rethink 2026-07-03, NaniteGrass) — the
    // bit31|bit30 pixels. Re-derive the blade triangle from the self-describing id
    // (bit-identical to the raster's corners), bary-interpolate tip/normal at the
    // reconstructed wp, then the S0 GroundRing material port (fresh/dry tip ramps ×
    // ~1.6 m patch dryness × canopy shade; blade normal pulled to the terrain normal
    // hardening with distance). Zero storage buffers — texture taps + ALU only.
    const gpTip = float(0.5).toVar() as unknown as NF;
    // G-D lean lighting (vec4 sunVis+irradiance from the per-frame guide light
    // field, ONE filtered tap) — fetched here inside the isGP branch, consumed by
    // the sun + GI blocks below, which then SKIP the bilateral shadow upsample +
    // probe chain for grass pixels (the ~7 ms pixel-proportional wall).
    const gpLean = (isGP && world.grassProc?.lean) || null;
    const gpSun = float(1).toVar() as unknown as NF;
    const gpIrr = vec3(0).toVar() as unknown as NV3;
    // ?grassdbg=flatres — attribution stop: grass pixels keep their election/depth
    // but the resolve stubs derive+material+per-pixel work to constants. Splits
    // "grass pixels EXIST downstream" from "grass resolve work" in the frame A/B.
    const gpFlat =
      new URLSearchParams(window.location.search).get('grassdbg') === 'flatres';
    if (isGP && world.grassProc) {
      const gp = world.grassProc;
      If(isGP, () => {
        if (gpFlat) {
          albedo.assign(vec3(0.05, 0.12, 0.03) as unknown as NV3);
          wNormal.assign(vec3(0, 1, 0) as unknown as NV3);
          ao.assign(float(1));
          gpTip.assign(float(0.5));
          return;
        }
        if (gpLean) {
          const L = gpLean(wp.xz as unknown as NV2) as unknown as NV4;
          gpSun.assign(L.x as unknown as NF);
          gpIrr.assign((L as unknown as { yzw: NV3 }).yzw);
        }
        const body = pRaw.bitAnd(uint(0x3fffffff));
        // G-E article lane: normal + tip come straight from the raycast lane's
        // screen texture (ONE tap) — the analytic id re-derivation is bypassed.
        const g = gp.ray
          ? ((): { t: NF; nrm: NV3 } => {
              const rv = gp.ray!(pixelIndex as unknown as NU) as unknown as NV4;
              return {
                t: rv.w as unknown as NF,
                nrm: normalize(rv.xyz as unknown as NV3) as unknown as NV3,
              };
            })()
          : gp.derive(body as unknown as NU, wp);
        const distG = wp.sub(vec3(camPos) as unknown as NV3).length();
        const toCamG = normalize(camPos.sub(wp)) as unknown as NV3;
        const nF = dot(g.nrm, toCamG).lessThan(0).select(g.nrm.negate(), g.nrm) as unknown as NV3;
        const tNrm = (
          texture(hf.normalTex, hf.uvFromWorld(wp.xz as unknown as NV2), 0) as unknown as NV4
        ).xyz.normalize() as unknown as NV3;
        // far super-tufts (body ≥ GRASS_FAR_BASE): full terrain-normal pull (ring far mode)
        const isFarG = body.greaterThanEqual(uint(GRASS_FAR_BASE));
        const upK = isFarG.select(
          float(1),
          smoothstep(8, 70, distG).mul(0.35).add(0.5),
        ) as unknown as NF;
        const t = g.t;
        const fresh = mix(
          vec3(0.02, 0.062, 0.011),
          vec3(0.065, 0.148, 0.028),
          t.mul(t),
        ) as unknown as NV3;
        const dryC = mix(vec3(0.085, 0.07, 0.024), vec3(0.21, 0.17, 0.075), t) as unknown as NV3;
        const patch = patchField(wp.xz as unknown as NV2).toVar() as unknown as NV2;
        const patchX = patch.x as unknown as NF;
        const patchY = patch.y as unknown as NF;
        const cov = (world.canopyTex
          ? canopyAt(world.canopyTex, wp.xz as unknown as NV2)
          : float(0)) as unknown as NF;
        const dryK = smoothstep(0.64, 0.82, patchX).mul(float(1).sub(cov.mul(0.85))) as unknown as NF;
        let alb = mix(fresh, dryC, dryK) as unknown as NV3;
        alb = alb.mul(patchY.sub(0.5).mul(0.4).add(1)) as unknown as NV3;
        alb = mix(alb, vec3(0.018, 0.052, 0.014) as unknown as NV3, cov.mul(0.55)) as unknown as NV3;
        albedo.assign(alb);
        wNormal.assign(normalize(mix(nF, tNrm, upK)) as unknown as NV3);
        ao.assign(smoothstep(0.0, 0.55, t).mul(0.55).add(0.45));
        gpTip.assign(t);
      });
    }

    // ---- MANUAL lighting (D-N17): sun lambert × CSM shadow + sky ambient +
    // probe GI. The CSM node (proven on the old path) is referenced as a
    // multiplicative factor exactly like AnalyticLightNode does
    // (colorNode.mul(shadowNode)); it carries OUR pcssFilter + the cloud
    // gate, sampling at receivedShadowPositionNode set above. Exact IBL
    // parity for the ambient is the remaining N4-C1 term.
    const sunDir = normalize(vec3(sunU.dir)) as unknown as NV3;
    let nDotL = max(dot(wNormal, sunDir), 0) as unknown as NF;
    // WRAP LIGHTING for VOXEL foliage (user report round 2: whole trees "extremely dark
    // next to lit ones" at oblique, aerial fine): the baked crown mean-normals rotate
    // with each instance's YAW, so a tree whose rotated normals face away from the sun
    // dropped to the ambient floor as a UNIT while its neighbor glowed (aerial is immune
    // — top bricks point up regardless of yaw). Foliage is translucent: use the standard
    // wrapped diffuse N·L·0.5+0.5 for voxel pixels — per-brick variation survives, but
    // no yaw can crash a crown to the floor.
    if (pass !== 'tri') {
      // FLOOR at 0.18: albedo-debug proved the residual far dark clumps are lighting-side —
      // deeply down/away-facing tile-brick mean normals bottom the plain wrap out at ~0.05.
      // No foliage pixel drops below ~0.16·sun; per-brick variation survives above the floor.
      const wrapped = dot(wNormal, sunDir)
        .mul(0.5)
        .add(0.5)
        .clamp(0, 1)
        .max(0.25)
        .mul(0.9) as unknown as NF;
      // procedural-grass pixels are bit31 too but shade like the S0 mesh grass —
      // standard N·L on the terrain-pulled normal, NOT the voxel crown wrap.
      const wrapGate = isGP
        ? (isV.equal(uint(1)).and((isGP as unknown as { not(): NB }).not()) as unknown as NB)
        : (isV.equal(uint(1)) as unknown as NB);
      nDotL = ((wrapGate as unknown as { select(a: NF, b: NF): NF }).select(wrapped, nDotL) as unknown as NF).toVar() as unknown as NF;
    }
    const sunCol = (sunU.color as unknown as NV3).mul(float(sunU.intensity)) as unknown as NV3;
    let direct: NF = nDotL;
    if (shadowsOn && world.naniteShadow) {
      // N5-R0 (D-N28): OUR depth-only shadow — PCSS over our r32 cascade textures,
      // sampled at the reconstructed world pos. We still REFERENCE three's CSM node
      // (keep) so three runs its per-frame cascade FIT (NaniteShadow.run reads the
      // fitted cascade VPs); its own map is EMPTY in the black slate → keep == 1 →
      // folds out (and a cheap blocker-search-only sample). ?oldgeo → csm path.
      // S0: half-res PCSS + bilateral upsample when wired (default), else the
      // full-res per-pixel sample (?shalfres=0). camDist drives the bilateral.
      const sf = float(1).toVar() as unknown as NF;
      const fullShadow = (): void => {
        const camDist = (wp as unknown as { sub(o: NV3): { length(): NF } })
          .sub(camPos)
          .length();
        const myRaw = world.shadowHalf
          ? world.shadowHalf.upsample(wp as unknown as NV3, camDist)
          : world.naniteShadow!.shadowFactor(wp as unknown as NV3, wNormal as unknown as NV3);
        sf.assign((myRaw as unknown as { clamp(a: number, b: number): NF }).clamp(0, 1));
        // keep ≡ three's CSM factor — ≈1 here (empty black-slate maps). keepFullU (default =
        // !reskeep) gates whether the FULL-SCREEN per-pixel CSM cascade-select + PCSS sample runs:
        //   1 → sampled on every covered pixel (old path, value = sf·keep).
        //   0 → sampled ONLY the [0,0] corner pixel, so three's CSM node stays BUILT (its per-frame
        //       cascade FIT — consumed by NaniteShadow.run, NaniteFrame:484 — keeps running) while
        //       every real pixel skips the wasted ≈1 sample. That sample is FULL-res, NOT quartered
        //       by the half-res shadow, so it is pure waste when our nanite shadow is active.
        // Bit-identical for real pixels (keep≡1); the corner pixel only keeps the node alive.
        if (world.csm) {
          const keep = (nodeObject(world.csm) as unknown as NV4).x.clamp(0, 1) as unknown as NF;
          const isCorner = (screenCoordinate.x as unknown as NF)
            .lessThan(float(1))
            .and((screenCoordinate.y as unknown as NF).lessThan(float(1)));
          If(isCorner.or((keepFullU as unknown as NF).greaterThan(float(0.5))), () => {
            sf.assign((sf as unknown as { mul(o: NF): NF }).mul(keep));
          });
        }
        if (world.cloudShadow) {
          // P2: the cloud sun-transmittance gate, applied directly (it used to reach
          // this pixel through the CSM filterNode via `keep`). Clamp + self-equality
          // guard mirror ShadowSetup: one NaN from the cloud sample would otherwise
          // poison the multiply and erase ALL cast shadows.
          const c = world.cloudShadow(wp.xz as unknown as NV2);
          const safe = c.equal(c).select(c.clamp(0, 1), float(1)) as unknown as NF;
          sf.assign((sf as unknown as { mul(o: NF): NF }).mul(safe));
        }
        if (world.farShadow) {
          // P4: beyond-clipmap terrain shadowing (baked heightfield sun-visibility) —
          // one bilinear tap, applied at ALL distances (a mountain shades the valley
          // even when the caster is outside every clipmap ring).
          const fv = world.farShadow(wp.xz as unknown as NV2).clamp(0, 1) as unknown as NF;
          sf.assign((sf as unknown as { mul(o: NF): NF }).mul(fv));
        }
      };
      // G-D lean: grass pixels take the pre-composed texel sunVis (PCSS×cloud×far
      // baked in kGuideLight) — the whole upsample/keep/cloud/far chain above is
      // skipped for them. Non-grass pixels run the identical old path.
      if (gpLean) {
        If(isGP as NB, () => {
          sf.assign(gpSun);
        }).Else(fullShadow);
      } else {
        fullShadow();
      }
      direct = nDotL.mul(sf) as unknown as NF;
    } else if (shadowsOn && world.csm) {
      const sf = (nodeObject(world.csm) as unknown as NV4).x.clamp(0, 1).toVar() as unknown as NF;
      direct = nDotL.mul(sf) as unknown as NF;
    }
    // ENERGY-CORRECT lighting (D-N22, user choice — NOT pixel-parity with the
    // old terrain). Uses three's BRDF energy exactly: BRDF_Lambert = albedo/π
    // on BOTH the direct sun term (irradiance = NdotL·sunColor, sunColor =
    // color·intensity, no π — three src 303/600/624) AND the indirect probe
    // irradiance (the old path's IrradianceNode → context.irradiance → ×albedo/π
    // — line 713). The probe field is the sole sky-diffuse ambient here
    // (occlusion-aware, ray-marches the atmosphere). DELIBERATE DIVERGENCE from
    // the old terrain, which ALSO adds a full env-IBL skylight term
    // (scene.environment, intensity 1.0) on top of the probe — making it
    // brighter; we do NOT replicate that (it double-counts the unoccluded sky).
    // So nanite terrain is dimmer than old by design; parity was abandoned.
    // Accumulate radiance, divide once by π.
    let radiance: NV3 = sunCol.mul(direct) as unknown as NV3;
    if (world.gi) {
      const irrV = vec3(0).toVar() as unknown as NV3;
      const fullGi = (): void => {
        // F9: read ground height from heightTex (TEXTURE — plentiful) so the
        // resolve does not bind the height STORAGE buffer (10-buffer/stage cap)
        const groundY = (
          texture(hf.heightTex, hf.uvFromWorld(wp.xz)) as unknown as NV4
        ).x as unknown as NF;
        let irr = world.gi!.irradiance(wp, wNormal, 2.0, groundY) as unknown as NV3;
        if (world.canopyTex) {
          irr = irr.mul(canopyAt(world.canopyTex, wp.xz).mul(0.18).oneMinus()) as unknown as NV3;
        }
        irrV.assign(irr);
      };
      // G-D lean: grass pixels take the texel-baked probe irradiance (canopy
      // damping already applied in kGuideLight); the probe chain is skipped.
      if (gpLean) {
        If(isGP as NB, () => {
          irrV.assign(gpIrr);
        }).Else(fullGi);
      } else {
        fullGi();
      }
      radiance = radiance.add(irrV.mul(ao)) as unknown as NV3;
    }
    // AMBIENT FLOOR (fixes black back-faces; bdb24c7 dropped the hemisphere ambient to
    // de-bright TERRAIN, but foliage/bark have back-faces where the probe SH-L1 self-clamps
    // to ~0 and mat.lights=false gives no env IBL ⇒ they crushed to pure black). A MAX floor
    // (NOT the old ADD) only catches faces the probe leaves dark — lit faces keep their
    // energy-correct probe value, so bdb24c7's no-double-count parity is preserved. Magnitude
    // = the old hemisphere floor × π (radiance is ÷π just below), restoring the pre-bdb24c7
    // soft dark on shaded sides. Tune the .mul() factor if it reads too bright/dark.
    let ambUp = wNormal.y.mul(0.5).add(0.5).clamp(0, 1) as unknown as NF;
    // VOX pixels: up-bias the hemisphere — a crown chunk whose MEAN normal tilts down
    // otherwise gets the dark ground ambient (3× darker than sky) and reads as an oddly
    // dark tree even with the wrapped sun floor (user round 3: "still some, half less").
    // A canopy always sees sky; floor its ambient mix at 0.6.
    if (pass !== 'tri') {
      // (grass pixels keep the plain hemisphere — their normal is terrain-pulled,
      // and the 0.6 sky floor was tuned for canopy chunks, not ground cover)
      const ambGate = isGP
        ? (isV.equal(uint(1)).and((isGP as unknown as { not(): NB }).not()) as unknown as NB)
        : (isV.equal(uint(1)) as unknown as NB);
      ambUp = ((ambGate as unknown as { select(a: NF, b: NF): NF }).select(ambUp.max(0.6), ambUp) as unknown as NF).toVar() as unknown as NF;
    }
    const ambFloor = mix(vec3(0.18, 0.16, 0.12), vec3(0.4, 0.5, 0.62), ambUp).mul(0.5 * Math.PI) as unknown as NV3;
    radiance = max(radiance, ambFloor) as unknown as NV3;
    let lit: NV3 = albedo.mul(radiance).mul(float(1 / Math.PI)) as unknown as NV3;
    // N9-C0: leaf BACKLIGHT — warm translucent forward-scatter toward the sun
    // (port of VegMaterials.translucency, k=0.032), added on top of the diffuse
    // (the old path's emissiveNode). Leaves only; 0 elsewhere.
    {
      const viewDir = normalize(wp.sub(camPos)) as unknown as NV3;
      const toward = dot(viewDir, sunDir.negate()).clamp(0, 1);
      const glow = toward.pow(5).mul(float(sunU.intensity)).mul(0.032);
      // VOXEL pixels get the same translucent forward-scatter (they ARE foliage) —
      // its absence was part of the "suddenly very dark trees" report: mesh crowns
      // glowed toward the sun while voxel crowns did not.
      // 'both': voxel pixels glow with voxCol, leaf pixels with leafCol — the per-pixel
      // union of what the two split passes each applied to their own tier.
      // GRASS (S0): tip-weighted forward scatter — grassTranslucency's k=0.09·tipT
      // vs the leaf constant 0.032 (VegMaterials.grassTranslucency port).
      const grassOn = pass !== 'vox' && hasClass(5);
      const blSrc =
        pass === 'tri'
          ? grassOn
            ? (isG.select(grassCol, leafCol) as unknown as NV3)
            : leafCol
          : pass === 'vox'
            ? voxCol
            : (isV
                .equal(uint(1))
                .select(voxCol, grassOn ? isG.select(grassCol, leafCol) : leafCol) as unknown as NV3);
      let blGate =
        pass === 'tri'
          ? grassOn
            ? (isL.or(isG) as unknown as typeof isL)
            : isL
          : pass === 'vox'
            ? (isV.equal(uint(1)) as unknown as typeof isL)
            : (isV.equal(uint(1)).or(grassOn ? isL.or(isG) : isL) as unknown as typeof isL);
      // procedural-grass pixels: excluded from the voxel-tier backlight (their voxCol
      // is a garbage decode) — they get their own tip-weighted term below.
      if (isGP) {
        blGate = blGate.and(
          (isGP as unknown as { not(): NB }).not() as unknown as typeof isL,
        ) as unknown as typeof isL;
      }
      const kBl = grassOn
        ? (isG.select(grassTip.mul(0.09), float(0.032)) as unknown as NF)
        : (float(0.032) as unknown as NF);
      const backlight = blSrc
        .mul(sunU.color as unknown as NV3)
        .mul(glow.div(0.032).mul(kBl))
        .mul(vec3(0.9, 1.05, 0.55)) as unknown as NV3;
      lit = lit.add(blGate.select(backlight, vec3(0))) as unknown as NV3;
      if (isGP) {
        // grassTranslucency port for the procedural lane: albedo already holds the
        // grass color on these pixels; k = 0.09 × tip weight (S0 parity).
        const gBl = albedo
          .mul(sunU.color as unknown as NV3)
          .mul(glow.div(0.032).mul(gpTip.mul(0.09)))
          .mul(vec3(0.9, 1.05, 0.55)) as unknown as NV3;
        lit = lit.add(
          (isGP as unknown as { select(a: NV3, b: NV3): NV3 }).select(gBl, vec3(0) as unknown as NV3),
        ) as unknown as NV3;
      }
    }

    // ---- debug overrides ------------------------------------------------------
    if (nandbg === 'flat') return vec4(albedo, 1);
    if (nandbg === 'albedo') return vec4(albedo, 1);
    if (nandbg === 'normal') return vec4(wNormal.mul(0.5).add(0.5), 1);
    // ?nandbg=shadow — the raw nanite shadow factor (white=lit, black=shadow);
    // ?nandbg=shadowc — which cascade covers each pixel (r/g/b/yellow = 0/1/2/3,
    // black = none). N5-R0 debug.
    if (
      (nandbg === 'shadow' || nandbg === 'shadowc' || nandbg === 'shadowd') &&
      world.naniteShadow
    ) {
      if (nandbg === 'shadowc')
        return vec4(world.naniteShadow.cascadeTint(wp as unknown as NV3), 1) as unknown as NV4;
      if (nandbg === 'shadowd') {
        const dd = world.naniteShadow.debugDepth(wp as unknown as NV3) as unknown as NF;
        return vec4(dd, dd, dd, 1) as unknown as NV4;
      }
      const s = world.naniteShadow.shadowFactor(
        wp as unknown as NV3,
        wNormal as unknown as NV3,
      ) as unknown as NF;
      return vec4(s, s, s, 1) as unknown as NV4;
    }
    if (nandbg === 'cov') return vec4(1, 0, 0, 1); // every covered pixel red
    // ?nandbg=ftonly — ISOLATE the far-tile aggregated field: fartile pixels keep
    // their real lit colour, EVERY other pixel (near voxels, mesh, terrain, grass)
    // is painted flat MAGENTA. The fartile bricks then read against magenta so the
    // griddy rectangles + the coverage GAPS (magenta showing between bricks) pop.
    // ?nandbg=ftlevel — colour each fartile pixel by its DAG pyramid level (word7
    // bits 10-15) via hashColor, non-fartile black: adjacent tiles at DIFFERENT
    // levels (LOD-crack) show as different colours with seams at the boundary; a
    // uniform colour band = single level (coverage, not crack). Both DEBUG-only.
    if (nandbg === 'ftonly' || nandbg === 'ftlevel') {
      const mF = fetch.meshWord(meshId, 6).shiftRight(uint(16)).bitAnd(uint(0xff));
      // gate on matClass==7 (voxel) so grass/terrain pixels with a garbage meshId
      // can never false-positive into "fartile"; near voxels lack the flag ⇒ magenta.
      const isFt = matClass
        .equal(uint(7))
        .and(mF.bitAnd(uint(MESH_FLAG_FARTILE)).notEqual(uint(0)));
      if (nandbg === 'ftlevel') {
        const lvl = elemU(gpu.clusters, ci.mul(uint(CLUSTER_WORDS)).add(uint(7)))
          .shiftRight(uint(10))
          .bitAnd(uint(0x3f));
        return vec4(isFt.select(hashColor(lvl), vec3(0) as unknown as NV3), 1) as unknown as NV4;
      }
      return vec4(isFt.select(lit, vec3(1, 0, 1) as unknown as NV3), 1) as unknown as NV4;
    }
    // per-cluster hash tint (matches the ?nanitedbg=cluster view, but for the
    // full-frame migrated set) — visualises meshlet boundaries on the resolve
    if (nandbg === 'cluster') return vec4(hashColor(ci), 1) as unknown as NV4;
    // ?nandbg=clhw — visualise the per-cluster SW/HW split (?clhw): RED = cluster the split
    // routes to the HW instanced draw, GREEN = kept on the SW compute raster. Recomputes the
    // SHARED clusterHwClass (bit-identical to the cull partition + SW-skip), so the tint IS the
    // routing decision. Sweep ?clhwmax to watch clusters cross the boundary; works even with
    // ?clhw off (it just shows how the split WOULD classify each cluster).
    if (nandbg === 'clhw') {
      const projK = cam.cotHalfFov.mul(cam.uH).mul(0.5) as unknown as NF;
      const isHw = clusterHwClass(
        gpu,
        vec3(cam.camPos) as unknown as NV3,
        projK,
        instId,
        ci,
        clhwMax,
      );
      return vec4(
        isHw.select(
          vec3(1, 0.12, 0.08) as unknown as NV3,
          vec3(0.1, 0.7, 0.2) as unknown as NV3,
        ),
        1,
      ) as unknown as NV4;
    }
    if (nandbg === 'cls')
      // matClass tint: terrain green / rock red / bark blue / deadwood cyan /
      // leaf bright-green / other (grass/debris) magenta
      return vec4(
        isT.select(
          vec3(0.1, 0.6, 0.1),
          isR.select(
            vec3(0.95, 0.1, 0.1),
            isB.select(
              vec3(0.1, 0.1, 0.95),
              isD.select(
                vec3(0.1, 0.7, 0.8),
                isL.select(vec3(0.2, 0.85, 0.2), vec3(0.8, 0.1, 0.8)),
              ),
            ),
          ),
        ),
        1,
      ) as unknown as NV4;
    return vec4(lit, 1);
  })() as unknown as typeof mat.fragmentNode;

  mat.depthNode = Fn(() => {
    const fy = float(cam.uH).sub(screenCoordinate.y);
    const pixelIndex = uint(fy).mul(uint(cam.uW)).add(uint(screenCoordinate.x));
    const elect = elemU(vis.payloadV.ro, pixelIndex);
    return float(1).sub(toF(elect.shiftRight(uint(8))).div(16777215)) as unknown as typeof mat.depthNode;
  })() as unknown as typeof mat.depthNode;
  mat.depthTest = false;
  mat.depthWrite = nandepth !== '0';
  mat.fog = false;
  mat.lights = false;
  return mat;
  }; // end buildMat

  // RP-4 ?respass — SINGLE merged resolve pass (tri+vox in one fullscreen draw, removes
  // the second pass's payloadV+visBV loads + discard over every pixel). Legal only where
  // the union binding set is ≤10 storage buffers: forest (gi null) is EXACTLY 10 — any
  // config with a probe buffer (world.gi) would hit the 11th-binding silent pipeline death,
  // so the merge is refused there. DEFAULT ON in exactly the proven config shape (voxel
  // queue + no gi + no csm = forest; identity gates 2026-07-02 §5m: TAA-still pairs in the
  // D0 band + no-TAA phase-aligned pairs 0.13-0.45% vs ~5-6% ambient floor). ?respass=0
  // reverts to two-pass; ?respass=1 forces the merge in any gi-free config (e.g. with csm).
  const respassFlag = q.get('respass');
  const singlePass =
    respassFlag === '0'
      ? false
      : respassFlag === '1'
        ? !!cull.qVoxRasterRO && !world.gi
        : !!cull.qVoxRasterRO && !world.gi && world.csm === null;
  if (respassFlag === '1' && cull.qVoxRasterRO && world.gi)
    // eslint-disable-next-line no-console
    console.warn(
      '[nanite] ?respass=1 refused: probe-GI buffer would push the merged resolve past the 10-storage-buffer cliff — keeping two-pass',
    );
  if (singlePass)
    // eslint-disable-next-line no-console
    console.log('[nanite] resolve single-pass (RP-4): tri+vox merged into one fullscreen draw');

  // MAIN pass (always): shades triangle pixels, skips voxel pixels — or, under ?respass=1,
  // shades BOTH tiers in one draw.
  const mesh = new Mesh(geometry, buildMat(singlePass ? 'both' : 'tri'));
  mesh.name = 'naniteResolve';
  mesh.frustumCulled = false;
  mesh.renderOrder = -1000;
  mesh.castShadow = false;
  mesh.receiveShadow = false;

  // VOXEL pass (only when the voxel queue is wired): shades ONLY voxel-winner pixels. Both the
  // MAIN and the VOXEL material have depthTest=false + the same depthNode, so the two fullscreen
  // draws composite by Discard partition (each pass discards the OTHER tier's pixels + uncovered
  // pixels) with no read-modify-write conflict; renderOrder −999 runs it right after the main
  // resolve, still well before the sky/scene remainder.
  let voxMesh: Mesh | undefined;
  // built when the voxel queue is wired OR procedural grass is on (grass pixels are
  // bit31|bit30 — they shade in this pass; without it they would discard to sky).
  if ((cull.qVoxRasterRO || world.grassProc) && !singlePass) {
    voxMesh = new Mesh(geometry, buildMat('vox'));
    voxMesh.name = 'naniteResolveVox';
    voxMesh.frustumCulled = false;
    voxMesh.renderOrder = -999;
    voxMesh.castShadow = false;
    voxMesh.receiveShadow = false;
  }
  return { mesh, voxMesh, keepFullU };
}
