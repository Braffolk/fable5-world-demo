/**
 * Forests — GPU-driven rendering of the scattered world (spec §3.6 core).
 *
 * Per frame, compute passes run before render:
 *   clear counters → cull each scatter layer → write indirect args.
 * The cull kernel does: per-class distance bound → frustum sphere test (6
 * planes) → terrain-occlusion march (heightfield ray test camera→crown-top,
 * the "Hi-Z" of a heightfield world) → LOD ring classification with overlap
 * bands → atomic append of the instance slot into per-(pool,ring) compact
 * regions. Draw instance counts go straight into an indirect buffer
 * (geometry.setIndirect) — instance data and counts never touch the CPU.
 * Cull granularity = instance (tree/shrub/rock), not 64-tri meshlets — the
 * deviation and rationale are documented in DEVIATIONS D-5.
 *
 * SHADOW CASTERS ARE CULLED PER CASCADE, not by the view frustum: the same
 * cull kernel also tests every instance against each CSM cascade's ortho
 * frustum (24 extra plane uniforms, refreshed from the cascade cameras each
 * frame — one frame stale, hidden inside the CSM lightMargin slack) and
 * appends into per-(pool,ring,cascade) caster regions. Each casting draw
 * gets 4 shadow-only sibling meshes on layers 2+c that ONLY cascade c's
 * shadow camera renders (ShadowNode keeps a custom camera layer mask), while
 * the main meshes stop casting. This fixes shadows of off-screen casters
 * (sun behind you, golden-hour edges) — the view-frustum compact lists used
 * to silently drop them from every cascade map — and skips the
 * camera-occlusion march for casters (a ridge-hidden tree still casts into
 * the visible slope).
 *
 * LOD rings (dithered crossfades in the materials):
 *   trees:  R0 hero ≤26 m (full bark + cards + real mesh leaves, ≥100k tris)
 *           → R1 full cards ≤150 m → R2 branch-cards ≤460 m → octahedral
 *           impostors beyond (4-tile view blend, relit — D-4 runtime)
 *   understory: single ring with per-class max distance
 *   extras: boulders/slabs swap to low-detail rock at 120 m, live to 700 m
 */

import {
  BufferAttribute,
  BufferGeometry,
  Color,
  Group,
  IcosahedronGeometry,
  Mesh,
  Vector3,
  Vector4,
} from 'three';
import type { PerspectiveCamera } from 'three';
import { Frustum, Matrix4 } from 'three';
import {
  IndirectStorageBufferAttribute,
  MeshStandardNodeMaterial,
  StorageBufferAttribute,
  type Renderer,
  type StorageBufferNode,
  type StorageTexture,
} from 'three/webgpu';
import { IrradianceNode } from 'three/webgpu';
import {
  Fn,
  If,
  Return,
  atomicAdd,
  atomicLoad,
  atomicStore,
  float,
  instanceIndex,
  instancedArray,
  int,
  normalWorld,
  positionLocal,
  positionWorld,
  smoothstep,
  storage,
  uint,
  uniform,
  uniformArray,
  varying,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import type { Heightfield } from '../world/Heightfield';
import { hash12 } from '../gpu/noise/NoiseTSL';
import type { ProbeGI } from '../gpu/passes/ProbeGI';
import { canopyAt, type ScatterLayer, type ScatterResult } from '../gpu/passes/Scatter';
import { impostorQuad, impostorRuntimeMaterial } from '../render/ImpostorRuntime';
import { instanceVeg, updateVegViewPos, type RingFade } from '../render/VegInstance';
import { depthPrepassTwin } from '../render/VegPrepass';
import type { NF, NI, NU, NV3, NV4 } from '../gpu/TSLTypes';
import type { VegLib } from './VegLibrary';

// ring distances (m) + dither bands (user feedback: transitions read too
// close — full-card trees hold to 150 m, impostors start at 460 m).
// Hero ring 0 (≤26 m): full bark + cards + REAL mesh leaves — the nanite-
// equivalence near field (spec floor: hero tree ≥100k tris).
const R0_FAR = 26;
const BAND0 = 5;
const R1_FAR = 150;
const BAND1 = 14;
const R2_FAR = 460;
const BAND2 = 36;
const EX_R1_FAR = 120;
const EX_BAND = 15;

// per-group compact-region capacities
const CAP_HERO = 48;
const CAP_TREE_R1 = 6144;
const CAP_TREE_R2 = 8192;
const CAP_IMPOSTOR = 49152;
const CAP_UNDER = 4096;
const CAP_EX_R1 = 1024;
const CAP_EX_R2 = 2048;

const MAIN_GROUPS = 170;
/**
 * Per-cascade caster groups: trees r1/r2 (48) + hero r0 (24) + extras/stones
 * (64) + impostor-band crown proxies per species (6). The impostor band
 * casts so tree shadows don't end in a hard circle at the R2 boundary —
 * they fade out by IMP_CAST_FAR instead.
 */
const CASC_LOCALS = 142;
const CASCADES = 4;
const GROUPS = MAIN_GROUPS + CASCADES * CASC_LOCALS;
/** crown-proxy shadows fade out across this band (m from camera) */
const IMP_CAST_FADE0 = 620;
const IMP_CAST_FAR = 1100;

function groupOf(cls: number, variant: number, ring: 0 | 1 | 2 | 3): number {
  if (cls < 6) {
    if (ring === 0) return 146 + cls * 4 + variant;
    if (ring === 3) return 48 + cls;
    return (cls * 4 + variant) * 2 + (ring - 1);
  }
  if (cls < 15) return 54 + (cls - 8) * 4 + variant;
  const pe = (cls - 16) * 4 + variant;
  return 82 + pe * 2 + (ring - 1);
}

/**
 * Caster-group index for cascade c. Local layout:
 *   0..47   tree pools × rings r1/r2  (pool*2 + ring-1)
 *   48..71  hero r0 per pool
 *   72..135 extras/stones pe × rings  (72 + pe*2 + ring-1)
 */
function casterGroupOf(
  c: number,
  cls: number,
  variant: number,
  ring: 0 | 1 | 2 | 3,
): number {
  const base = MAIN_GROUPS + c * CASC_LOCALS;
  if (cls < 6) {
    if (ring === 3) return base + 136 + cls;
    const pool = cls * 4 + variant;
    if (ring === 0) return base + 48 + pool;
    return base + pool * 2 + (ring - 1);
  }
  const pe = (cls - 16) * 4 + variant;
  return base + 72 + pe * 2 + (ring - 1);
}

function capOf(g: number): number {
  if (g >= MAIN_GROUPS) {
    // caster regions: a cascade box covers a slice of the frustum, so the
    // worst case is well under the main-view caps
    const local = (g - MAIN_GROUPS) % CASC_LOCALS;
    if (local >= 136) return 8192; // impostor-band crown proxies (per cls)
    if (local < 48) return local % 2 === 0 ? 3072 : 6144; // tree r1/r2
    if (local < 72) return CAP_HERO;
    const pe = (local - 72) >> 1;
    const cls = 16 + (pe >> 2);
    const isR1 = (local - 72) % 2 === 0;
    if (cls < 20) return isR1 ? 512 : 1024; // extras
    if (cls === 20) return isR1 ? 2048 : 12288; // StoneL → 900 m
    if (cls === 21) return isR1 ? 4096 : 8192; // StoneM
    if (cls === 22) return isR1 ? 12288 : 64; // StoneS — single ring
    return 4096; // Branch
  }
  if (g < 48) return g % 2 === 0 ? CAP_TREE_R1 : CAP_TREE_R2;
  if (g < 54) return CAP_IMPOSTOR;
  if (g < 82) return CAP_UNDER;
  if (g >= 146) return CAP_HERO;
  if (g < 114) return (g - 82) % 2 === 0 ? CAP_EX_R1 : CAP_EX_R2;
  // size-stratified stones/branches (cls 20–23)
  const cls = 16 + ((g - 82) >> 3);
  const isR1 = (g - 82) % 2 === 0;
  if (cls === 20) return isR1 ? 4096 : 24576; // StoneL → 900 m
  if (cls === 21) return isR1 ? 8192 : 16384; // StoneM → 280 m
  if (cls === 22) return isR1 ? 24576 : 64; // StoneS — single ring
  return 8192; // Branch
}

/**
 * Crown shadow density per tree class (spruce/pine/beech/birch/karst/snag).
 * Real closed canopy transmits 2–5% at noon; hollow card-shell crowns leak
 * 40%+ through their alpha gradients and PCSS averages the speckle into a
 * flat half-lit wash (no dapple, no dark interior). The shadow proxy core
 * (dithered to this density) restores bulk occlusion; cards keep the edges
 * ragged in the near ring. Snag crowns are bare — no core.
 */
const CROWN_SHADOW_DENSITY = [0.9, 0.84, 0.92, 0.74, 0.85, 0] as const;

/** crown proxy dims, FITTED to a pool's actual ring geometry (meters, scale 1) */
interface CrownDims {
  cy: number;
  ry: number;
  rxz: number;
}

/**
 * Shadow-proxy tree: 80-tri ellipsoid crown + 12-tri trunk prism, fitted to
 * the pool's own geometry bounds (class-max dims made small variants throw
 * giant blob shadows — user-reported). This is the ONLY tree caster beyond
 * R1 (a cascade texel out there is ≥0.5 m — card raggedness is invisible)
 * and the bulk-density core inside R1's card edges.
 */
function crownProxyGeometry(d: CrownDims): BufferGeometry {
  // PolyhedronGeometry is non-indexed: 80 faces × 3 verts at detail 1
  const core = new IcosahedronGeometry(1, 1);
  const cpos = core.attributes.position as BufferAttribute;
  const cy = d.cy;
  const nCore = cpos.count;
  const tr = 0.035 * d.rxz + 0.03;
  const merged = new Float32Array(nCore * 3 + 6 * 3);
  for (let i = 0; i < nCore; i++) {
    merged[i * 3] = cpos.getX(i) * d.rxz;
    merged[i * 3 + 1] = cpos.getY(i) * d.ry + cy;
    merged[i * 3 + 2] = cpos.getZ(i) * d.rxz;
  }
  // trunk prism: 3 quads, base→crown center
  const idx: number[] = [];
  for (let i = 0; i < nCore; i++) idx.push(i);
  for (let k = 0; k < 3; k++) {
    const a = (k / 3) * Math.PI * 2;
    const o = (nCore + k * 2) * 3;
    merged[o] = Math.cos(a) * tr;
    merged[o + 1] = 0;
    merged[o + 2] = Math.sin(a) * tr;
    merged[o + 3] = Math.cos(a) * tr * 0.6;
    merged[o + 4] = cy;
    merged[o + 5] = Math.sin(a) * tr * 0.6;
  }
  for (let k = 0; k < 3; k++) {
    const n = (k + 1) % 3;
    idx.push(
      nCore + k * 2, nCore + n * 2, nCore + k * 2 + 1,
      nCore + n * 2, nCore + n * 2 + 1, nCore + k * 2 + 1,
    );
  }
  const nrm = new Float32Array(merged.length);
  for (let i = 0; i < nrm.length; i += 3) nrm[i + 1] = 1;
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(merged, 3));
  g.setAttribute('normal', new BufferAttribute(nrm, 3));
  g.setIndex(idx);
  return g;
}

export class Forests {
  readonly group = new Group();
  /** depth twins render before color draws (renderOrder; kept in one child
   *  group so a future bundle path can rely on traversal order too) */
  private prepassGroup = new Group();
  /** camera-pass meshes of pool OPAQUE parts (parts[0] — exactly what
   *  WorldRegistry migrates) keyed by VegClass — hidden per migrated class in
   *  the nanite full-frame mode (D-N19). Caster siblings are separate meshes
   *  on cascade layers and NEVER appear here: suppression cannot touch
   *  shadows. */
  private migratable: { mesh: Mesh; cls: number }[] = [];

  /**
   * Hide the camera draws of pools whose material class is nanite-migrated.
   * `matClassOf` = WorldRegistry.migratedMatClass (passed in by the caller —
   * the registry module stays a dynamic import). Returns hidden-mesh count.
   */
  suppressMigrated(
    matClassOf: (cls: number) => string | null,
    classes: ReadonlySet<string>,
  ): number {
    let n = 0;
    for (const m of this.migratable) {
      const mc = matClassOf(m.cls);
      if (mc !== null && classes.has(mc)) {
        m.mesh.visible = false;
        n++;
      }
    }
    return n;
  }

  private compact!: StorageBufferNode<'uint'>;
  private counters!: ReturnType<StorageBufferNode<'uint'>['toAtomic']>;
  private kernels: object[] = [];
  private camU = uniform(new Vector3());
  private planesU = uniformArray(
    Array.from({ length: 6 }, () => new Vector4()),
  );
  /** 6 planes × 4 cascade ortho frusta; w=-1e9 ⇒ reject-all until CSM exists */
  private planesCsmU = uniformArray(
    Array.from({ length: 6 * CASCADES }, () => new Vector4(0, 0, 0, -1e9)),
  );
  private csm: object | null = null;
  private frustum = new Frustum();
  private projView = new Matrix4();
  private cascM = new Matrix4();
  private cascFrustum = new Frustum();
  private indirectAttr!: IndirectStorageBufferAttribute;
  private groupTris = new Float32Array(GROUPS);
  private groupCaps = new Uint32Array(GROUPS);
  private reading = false;
  private frame = 0;
  private hud: Record<string, number> = {};

  constructor(
    private hf: Heightfield,
    private scatter: ScatterResult,
    private lib: VegLib,
    private gi: ProbeGI | null,
    private canopyTex: StorageTexture | null = null,
  ) {}

  private patchGI(mat: MeshStandardNodeMaterial): void {
    const gi = this.gi;
    if (!gi) return;
    let irr = gi.irradiance(positionWorld as unknown as NV3, normalWorld as unknown as NV3);
    if (this.canopyTex) {
      // probe field is canopy-aware (crown-slab extinction in the gather) —
      // this is only the 4 m-texel residual the 16 m probe grid can't carry
      irr = irr.mul(
        canopyAt(this.canopyTex, (positionWorld as unknown as NV3).xz)
          .mul(0.12)
          .oneMinus(),
      ) as typeof irr;
    }
    // vertex-stage probe GI (Phase 7 perf): the probe grid is 16 m and the
    // canopy residual 4 m — across ≤2 m cards or cm-tessellated hero meshes
    // vertex eval + interpolation is identical, and drops 4 texture taps
    // from every overdrawn foliage fragment
    const irrV = varying(irr as unknown as Parameters<typeof varying>[0]);
    (mat as unknown as { setupLightMap: () => unknown }).setupLightMap = () =>
      new IrradianceNode(irrV as unknown as ConstructorParameters<typeof IrradianceNode>[0]);
  }

  init(renderer: Renderer): void {
    void renderer;
    this.group.add(this.prepassGroup);
    const lib = this.lib;

    // ---- compact regions / group tables ------------------------------------
    const offsets = new Uint32Array(GROUPS);
    let off = 0;
    for (let g = 0; g < GROUPS; g++) {
      offsets[g] = off;
      this.groupCaps[g] = capOf(g);
      off += capOf(g);
    }
    this.compact = instancedArray(off, 'uint');
    this.counters = instancedArray(GROUPS, 'uint').toAtomic();
    const offBuf = storage(new StorageBufferAttribute(offsets, 1), 'uint', GROUPS);
    const capBuf = storage(
      new StorageBufferAttribute(this.groupCaps.slice(), 1),
      'uint',
      GROUPS,
    );

    // per-class cull info: (height, radius, maxDist, hasR2)
    const clsInfo = new Float32Array(24 * 4);
    for (let c = 0; c < 24; c++) {
      clsInfo[c * 4 + 0] = this.lib.clsHeight[c] ?? 1;
      clsInfo[c * 4 + 1] = this.lib.clsRadius[c] ?? 1;
      clsInfo[c * 4 + 2] = this.lib.clsMaxDist[c] ?? 150;
      const hasR2 = c < 6 || c === 18 || c === 19 || c === 20 || c === 21 || c === 23;
      clsInfo[c * 4 + 3] = hasR2 ? 1 : 0;
    }
    const clsBuf = storage(new StorageBufferAttribute(clsInfo, 4), 'vec4', 24);

    // ---- draws ---------------------------------------------------------------
    interface DrawSpec {
      group: number;
      indexCount: number;
    }
    const draws: DrawSpec[] = [];
    const meshes: Mesh[] = [];

    const prepassQ = new URLSearchParams(window.location.search).get('prepass');
    const noPrepass = prepassQ === '0' || prepassQ === 'grass';
    const addDraw = (
      geo: import('three').BufferGeometry,
      mat: MeshStandardNodeMaterial,
      g: number,
      tris: number,
      shadowLayer: number | null = null,
    ): Mesh => {
      const indexCount = geo.index ? geo.index.count : geo.attributes.position?.count ?? 0;
      draws.push({ group: g, indexCount });
      const mesh = new Mesh(geo, mat);
      mesh.frustumCulled = false;
      if (shadowLayer === null) {
        // visible draw — casting is owned by the per-cascade sibling meshes
        this.groupTris[g] += tris;
        mesh.castShadow = false;
        mesh.receiveShadow = true;
        // depth prepass for CARD parts only: crowns shade 3-8x per covered
        // pixel (alpha-tested cutouts defeat early-Z); opaque bark/rock
        // overdraw is shallow and each twin costs a CPU draw (~29 us) +
        // doubled vertex wind, so twinning everything LOST wall time.
        // (A DoubleSide quad rasterizes each triangle ONCE — no duplicate
        // face depth-ties; EQUAL is safe. Verified vs no-prepass at a
        // frame-aligned capture: differences at the deterministic floor.)
        if (!noPrepass && mat.alphaTest > 0) {
          const matS = mat as unknown as {
            positionNode: unknown;
            maskNode: unknown;
            opacityNode: unknown;
          };
          this.prepassGroup.add(
            depthPrepassTwin(mesh, {
              positionNode: matS.positionNode,
              maskNode: matS.maskNode ?? undefined,
              ...(mat.alphaTest > 0
                ? { opacityNode: matS.opacityNode, alphaTest: mat.alphaTest }
                : {}),
              side: mat.side,
            }),
          );
        }
      } else {
        // shadow-only caster: lives on the cascade's layer, so ONLY that
        // cascade's shadow camera ever renders it
        mesh.castShadow = true;
        mesh.receiveShadow = false;
        mesh.layers.set(shadowLayer);
      }
      meshes.push(mesh);
      this.group.add(mesh);
      return mesh;
    };

    /** geometry view sharing attributes/index but with its own indirect slot */
    const geoView = (src: import('three').BufferGeometry): BufferGeometry => {
      const g = new BufferGeometry();
      for (const [name, attr] of Object.entries(src.attributes)) {
        g.setAttribute(name, attr as import('three').BufferAttribute);
      }
      if (src.index) g.setIndex(src.index);
      return g;
    };

    const layerOf = (cls: number): ScatterLayer =>
      cls < 6
        ? this.scatter.trees
        : cls < 15
          ? this.scatter.understory
          : cls < 20
            ? this.scatter.extras
            : this.scatter.stones;

    /**
     * Shadow-proxy caster material: world-anchored hash dither (screen-space
     * IGN swims when CSM refits its boxes — user-visible shadow flicker) at
     * species density, with a crown-edge falloff so the rim breaks up into
     * a ragged crown instead of a solid oval. Impostor-band proxies fade out
     * toward IMP_CAST_FAR (fade distance uses vegViewPos via instanceVeg).
     */
    const proxyCasterMat = (
      bind: Parameters<typeof instanceVeg>[1],
      density: number,
      dims: CrownDims,
      impostorBand: boolean,
    ): MeshStandardNodeMaterial => {
      const pmat = new MeshStandardNodeMaterial();
      const handles = instanceVeg(pmat, bind);
      const e = positionLocal
        .sub(vec3(0, dims.cy, 0))
        .div(vec3(dims.rxz, dims.ry, dims.rxz))
        .length();
      let dens: NF = float(density).mul(
        float(1).sub(e.pow(3).mul(0.55)),
      );
      if (impostorBand) {
        dens = dens.mul(
          float(1).sub(smoothstep(IMP_CAST_FADE0, IMP_CAST_FAR - 50, handles.dist)),
        );
      }
      (pmat as unknown as { maskShadowNode: unknown }).maskShadowNode = hash12(
        positionWorld.xz.mul(13.73).add(positionWorld.yy.mul(5.19)),
      ).lessThan(dens);
      return pmat;
    };

    const fadeFor = (cls: number, ring: 0 | 1 | 2 | 3): RingFade => {
      if (cls < 6) {
        if (ring === 0) return { fadeOutAt: R0_FAR, band: BAND0 };
        if (ring === 1)
          return { fadeInAt: R0_FAR, inBand: BAND0, fadeOutAt: R1_FAR, band: BAND1 };
        // bands MUST match across each boundary (in-band here = out-band of
        // the nearer ring) or the complementary dither doesn't partition
        // pixels and holes reappear — hence inBand: ring2's out edge pairs
        // with the impostor's BAND2 while its in edge pairs with BAND1.
        if (ring === 2)
          return { fadeInAt: R1_FAR, inBand: BAND1, fadeOutAt: R2_FAR, band: BAND2 };
        return { fadeInAt: R2_FAR, band: BAND2 };
      }
      const maxD = this.lib.clsMaxDist[cls] ?? 150;
      if (cls < 15) return { fadeOutAt: maxD - 15, band: 15 };
      const hasR2 = cls === 18 || cls === 19 || cls === 20 || cls === 21 || cls === 23;
      if (ring === 1)
        return hasR2
          ? { fadeOutAt: EX_R1_FAR, band: EX_BAND }
          : { fadeOutAt: maxD - 20, band: 20 };
      return { fadeInAt: EX_R1_FAR, fadeOutAt: maxD - 20, band: EX_BAND };
    };

    for (const pool of lib.pools) {
      const layer = layerOf(pool.cls);
      const rings: { ring: 0 | 1 | 2; parts: typeof pool.r1 }[] = [];
      if (pool.r0) rings.push({ ring: 0, parts: pool.r0 });
      if (pool.r1) rings.push({ ring: 1, parts: pool.r1 });
      if (pool.r2) rings.push({ ring: 2, parts: pool.r2 });
      // shadow budget: tree rings 0–2 cast (per-cascade caster lists);
      // understory is grounded by contact shadows + AO instead.
      // ?ablate=casters drops ALL veg caster draws (perf attribution).
      const ablateCasters = (
        new URLSearchParams(window.location.search).get('ablate') ?? ''
      )
        .split(',')
        .includes('casters');
      const ringCasts =
        !ablateCasters && (pool.cls < 6 ? true : pool.cls < 15 ? false : true);
      const crownDensity = pool.cls < 6 ? CROWN_SHADOW_DENSITY[pool.cls] ?? 0 : 0;
      // fit the shadow proxy to THIS pool's real extents (R1 union bbox)
      let poolDims: CrownDims | null = null;
      if (crownDensity > 0) {
        const fitParts = pool.r1 ?? pool.r2 ?? null;
        if (fitParts) {
          let top = 2;
          let rxz = 0.8;
          for (const part of fitParts) {
            part.geo.computeBoundingBox();
            const bb = part.geo.boundingBox;
            if (!bb) continue;
            top = Math.max(top, bb.max.y);
            rxz = Math.max(
              rxz,
              Math.abs(bb.min.x),
              bb.max.x,
              Math.abs(bb.min.z),
              bb.max.z,
            );
          }
          // cards overhang the foliage mass — pull the core in
          const bot = top * 0.32;
          poolDims = {
            cy: (top + bot) / 2,
            ry: ((top - bot) / 2) * 0.95,
            rxz: rxz * 0.74,
          };
        }
      }
      // wind opt-in: living vegetation sways (trees + understory); deadfall,
      // stumps and stones stay rigid (their vdata.y means moss/decay, and
      // a swaying log reads broken instantly). Trees rock slowly around the
      // trunk-bend knee; understory is light + springy (faster natural
      // frequency, knee near the ground); bare snags are stiff dead wood.
      const windBind =
        pool.cls < 6
          ? pool.cls === 5
            ? { k: 0.45, freq: 0.8, h0: 6 }
            : { k: 1, freq: 1, h0: 6 }
          : pool.cls < 15
            ? { k: 1, freq: 1.8, h0: 0.9 }
            : undefined;
      for (const { ring, parts } of rings) {
        if (!parts) continue;
        const g = groupOf(pool.cls, pool.variant, ring);
        for (const part of parts) {
          const mat = part.make();
          instanceVeg(mat, {
            bufA: layer.bufA,
            bufB: layer.bufB,
            compact: this.compact,
            groupBase: offsets[g] ?? 0,
            fade: fadeFor(pool.cls, ring),
            wind: windBind,
          });
          this.patchGI(mat);
          // ?clsdbg=1 — flat-color every draw by VegClass (artifact triage:
          // "which pool is that?"); keeps alpha cutouts so silhouettes read
          if (new URLSearchParams(window.location.search).get('clsdbg') === '1') {
            const hue = (pool.cls * 47) % 360;
            const cdbg = new Color().setHSL(hue / 360, 0.95, 0.55);
            const op = mat.opacityNode as unknown as NF | null;
            mat.colorNode = vec4(vec3(cdbg.r, cdbg.g, cdbg.b), 1);
            if (op) mat.opacityNode = op;
          }
          const visMesh = addDraw(part.geo, mat, g, part.tris);
          // opaque part = parts[0] (WorldRegistry's migration unit) — track
          // it for D-N19 camera-draw suppression
          if (part === parts[0]) this.migratable.push({ mesh: visMesh, cls: pool.cls });
          // per-cascade caster siblings. Tree R2 skips its card/bark parts —
          // the crown proxy below carries the whole far shadow (a cascade
          // texel ≥0.5 m out there; 1.8k-tri cards bought nothing but raster)
          const proxyOwnsRing = pool.cls < 6 && ring === 2 && crownDensity > 0;
          // far cascades have 0.5–21 m texels — card-level caster geometry
          // buys nothing there; the crown proxies (added below for rings
          // 1+2 in EVERY cascade) own the far shadow. Ring-1 real casters
          // only feed the two near cascades (~15 ms → ~7 ms caster raster).
          const cascadeMax =
            pool.cls < 6 && ring === 1 && crownDensity > 0 ? 2 : CASCADES;
          if (part.castShadow && ringCasts && !proxyOwnsRing) {
            for (let c = 0; c < cascadeMax; c++) {
              const cg = casterGroupOf(c, pool.cls, pool.variant, ring);
              const cmat = part.make();
              instanceVeg(cmat, {
                bufA: layer.bufA,
                bufB: layer.bufB,
                compact: this.compact,
                groupBase: offsets[cg] ?? 0,
                fade: null,
                wind: windBind,
              });
              addDraw(geoView(part.geo), cmat, cg, 0, 2 + c);
            }
          }
        }
        // crown shadow proxy (rings 1+2): bulk occlusion at CROWN density —
        // cards alone leak 40%+ and PCSS flattens the speckle into a half-lit
        // wash; the core brings noon interiors down to real 5–15% so dapple
        // pools only survive at TRUE crown gaps
        if (ring > 0 && ringCasts && crownDensity > 0 && poolDims) {
          const proxyGeo = crownProxyGeometry(poolDims);
          for (let c = 0; c < CASCADES; c++) {
            const cg = casterGroupOf(c, pool.cls, pool.variant, ring);
            const pmat = proxyCasterMat(
              {
                bufA: layer.bufA,
                bufB: layer.bufB,
                compact: this.compact,
                groupBase: offsets[cg] ?? 0,
                fade: null,
              },
              crownDensity,
              poolDims,
              false,
            );
            addDraw(geoView(proxyGeo), pmat, cg, 0, 2 + c);
          }
          // impostor band (variant 0 carries it: one caster group per cls)
          if (ring === 2 && pool.variant === 0) {
            for (let c = 0; c < CASCADES; c++) {
              const cg = casterGroupOf(c, pool.cls, 0, 3);
              const pmat = proxyCasterMat(
                {
                  bufA: layer.bufA,
                  bufB: layer.bufB,
                  compact: this.compact,
                  groupBase: offsets[cg] ?? 0,
                  fade: null,
                },
                crownDensity,
                poolDims,
                true,
              );
              addDraw(geoView(crownProxyGeometry(poolDims)), pmat, cg, 0, 2 + c);
            }
          }
        }
      }
    }

    // tree impostors: one billboard draw per species
    for (const [cls, atlas] of lib.impostors) {
      const g = groupOf(cls, 0, 3);
      const mat = impostorRuntimeMaterial(atlas, {
        bufA: this.scatter.trees.bufA,
        bufB: this.scatter.trees.bufB,
        compact: this.compact,
        groupBase: offsets[g] ?? 0,
        fade: fadeFor(cls, 3),
      });
      this.patchGI(mat);
      addDraw(impostorQuad(), mat, g, 2);
    }

    // ---- indirect buffer -------------------------------------------------------
    const D = draws.length;
    const indirectData = new Uint32Array(D * 5);
    const drawGroups = new Uint32Array(D);
    for (let d = 0; d < D; d++) {
      const spec = draws[d] as DrawSpec;
      indirectData[d * 5] = spec.indexCount;
      drawGroups[d] = spec.group;
    }
    this.indirectAttr = new IndirectStorageBufferAttribute(indirectData, 5);
    for (let d = 0; d < D; d++) {
      (meshes[d] as Mesh).geometry.setIndirect(this.indirectAttr, d * 20);
    }
    const indirectStore = storage(this.indirectAttr, 'uint', D * 5);
    const drawGroupBuf = storage(new StorageBufferAttribute(drawGroups, 1), 'uint', D);

    // ---- kernels ---------------------------------------------------------------
    const counters = this.counters;
    const compact = this.compact;
    const camU = this.camU;
    const planesU = this.planesU;
    const hf = this.hf;

    const clearK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(GROUPS), () => {
        Return();
      });
      atomicStore(counters.element(i), uint(0));
    })().compute(GROUPS);
    clearK.setName('vegClear');

    const inFrustum = (center: NV3, rad: NF): NF => {
      // product of per-plane step(−r ≤ dist) — 1 inside, 0 outside
      let inside: NF = float(1);
      for (let p = 0; p < 6; p++) {
        const pl = planesU.element(int(p)) as unknown as NV4;
        const d = pl.xyz.dot(center).add(pl.w);
        inside = inside.mul(d.greaterThan(rad.negate()).select(float(1), float(0)));
      }
      return inside;
    };

    const planesCsmU = this.planesCsmU;
    // +30 m slack: the planes are one frame stale (CSM fits its boxes during
    // the upcoming render) — without it, casters at box edges pop while the
    // camera moves
    const inCascade = (c: number, center: NV3, rad: NF): NF => {
      let inside: NF = float(1);
      for (let p = 0; p < 6; p++) {
        const pl = planesCsmU.element(int(c * 6 + p)) as unknown as NV4;
        const d = pl.xyz.dot(center).add(pl.w);
        inside = inside.mul(
          d.greaterThan(rad.add(30).negate()).select(float(1), float(0)),
        );
      }
      return inside;
    };

    const appendTo = (g: NI | NU, slot: NU): void => {
      const idx = atomicAdd(counters.element(g), uint(1)) as unknown as NU;
      If(idx.lessThan(capBuf.element(g) as unknown as NU), () => {
        compact
          .element((offBuf.element(g) as unknown as NU).add(idx))
          .assign(slot);
      });
    };

    const makeCull = (
      layer: ScatterLayer,
      kind: 'trees' | 'under' | 'extras',
    ): object => {
      const N = layer.count;
      const k = Fn(() => {
        const i = instanceIndex;
        If(i.greaterThanEqual(uint(Math.max(N, 1))), () => {
          Return();
        });
        const A = layer.bufA.element(i) as unknown as NV4;
        const B = layer.bufB.element(i) as unknown as NV4;
        const idF = B.w;
        const cls = idF.div(8).floor();
        const variant = idF.sub(cls.mul(8));
        const info = clsBuf.element(cls.toInt()) as unknown as NV4;
        const scl = A.w;
        const hgt = info.x.mul(scl);
        const rad = info.y.mul(scl);
        const center = A.xyz.add(vec3(0, 1, 0).mul(hgt.mul(0.5)));
        const dist = A.xyz.sub(camU).length();

        if (kind !== 'trees') {
          // hard reach bound — applies to main view AND casters (beyond it
          // no ring geometry exists at all)
          If(dist.greaterThanEqual(info.z), () => {
            Return();
          });
        }

        if (kind === 'under') {
          // understory never casts — keep the cheap early-out path
          If(inFrustum(center, rad).lessThan(0.5), () => {
            Return();
          });
          const g = cls.sub(8).mul(4).add(variant).add(54).toInt();
          appendTo(g as unknown as NI, i as unknown as NU);
          return;
        }

        // main-view visibility: frustum + terrain-occlusion march (camera
        // sight line) — casters intentionally skip BOTH (an off-screen or
        // ridge-hidden tree still casts into the visible scene)
        const visMain = inFrustum(center, rad).toVar();
        If(visMain.greaterThan(0.5).and(dist.greaterThan(140)), () => {
          const top = vec3(A.x, A.y.add(hgt), A.z);
          const occ = float(0).toVar();
          for (let st = 1; st <= 7; st++) {
            const t = st / 8;
            const sp = camU.mul(1 - t).add(top.mul(t)) as unknown as NV3;
            const th = hf.sampleHeightNearest(vec2(sp.x, sp.z));
            occ.assign(occ.max(th.sub(sp.y)));
          }
          If(occ.greaterThan(4), () => {
            visMain.assign(0);
          });
        });

        if (kind === 'trees') {
          const pool = cls.mul(4).add(variant).toInt();
          If(visMain.greaterThan(0.5), () => {
            If(dist.lessThan(R0_FAR + BAND0), () => {
              appendTo(pool.add(146) as unknown as NI, i as unknown as NU);
            });
            If(
              dist.greaterThanEqual(R0_FAR - BAND0).and(dist.lessThan(R1_FAR + BAND1)),
              () => {
                appendTo(pool.mul(2) as unknown as NI, i as unknown as NU);
              },
            );
            If(
              dist.greaterThanEqual(R1_FAR - BAND1).and(dist.lessThan(R2_FAR + BAND2)),
              () => {
                appendTo(pool.mul(2).add(1) as unknown as NI, i as unknown as NU);
              },
            );
            If(dist.greaterThanEqual(R2_FAR - BAND2), () => {
              appendTo(cls.add(48).toInt() as unknown as NI, i as unknown as NU);
            });
          });
          // casters per cascade — same ring choice as the main view so the
          // shadow silhouette matches the rendered crown
          for (let c = 0; c < CASCADES; c++) {
            const base = MAIN_GROUPS + c * CASC_LOCALS;
            If(inCascade(c, center, rad).greaterThan(0.5), () => {
              If(dist.lessThan(R0_FAR + BAND0), () => {
                appendTo(
                  pool.add(base + 48) as unknown as NI,
                  i as unknown as NU,
                );
              });
              If(
                dist.greaterThanEqual(R0_FAR - BAND0).and(dist.lessThan(R1_FAR + BAND1)),
                () => {
                  appendTo(pool.mul(2).add(base) as unknown as NI, i as unknown as NU);
                },
              );
              If(
                dist.greaterThanEqual(R1_FAR - BAND1).and(dist.lessThan(R2_FAR + BAND2)),
                () => {
                  appendTo(
                    pool.mul(2).add(base + 1) as unknown as NI,
                    i as unknown as NU,
                  );
                },
              );
              // impostor band: crown proxies keep casting past R2 so the
              // shadow field fades out instead of ending in a camera circle
              If(
                dist.greaterThanEqual(R2_FAR - BAND2).and(dist.lessThan(IMP_CAST_FAR)),
                () => {
                  appendTo(
                    cls.add(base + 136).toInt() as unknown as NI,
                    i as unknown as NU,
                  );
                },
              );
            });
          }
        } else {
          const pe = cls.sub(16).mul(4).add(variant);
          const hasR2 = info.w.greaterThan(0.5);
          If(visMain.greaterThan(0.5), () => {
            If(hasR2, () => {
              If(dist.lessThan(EX_R1_FAR + EX_BAND), () => {
                appendTo(pe.mul(2).add(82).toInt() as unknown as NI, i as unknown as NU);
              });
              If(dist.greaterThanEqual(EX_R1_FAR - EX_BAND), () => {
                appendTo(pe.mul(2).add(83).toInt() as unknown as NI, i as unknown as NU);
              });
            }).Else(() => {
              appendTo(pe.mul(2).add(82).toInt() as unknown as NI, i as unknown as NU);
            });
          });
          for (let c = 0; c < CASCADES; c++) {
            const base = MAIN_GROUPS + c * CASC_LOCALS + 72;
            If(inCascade(c, center, rad).greaterThan(0.5), () => {
              If(hasR2, () => {
                If(dist.lessThan(EX_R1_FAR + EX_BAND), () => {
                  appendTo(
                    pe.mul(2).add(base).toInt() as unknown as NI,
                    i as unknown as NU,
                  );
                });
                If(dist.greaterThanEqual(EX_R1_FAR - EX_BAND), () => {
                  appendTo(
                    pe.mul(2).add(base + 1).toInt() as unknown as NI,
                    i as unknown as NU,
                  );
                });
              }).Else(() => {
                appendTo(
                  pe.mul(2).add(base).toInt() as unknown as NI,
                  i as unknown as NU,
                );
              });
            });
          }
        }
      })().compute(Math.max(N, 1));
      k.setName(`vegCull_${kind}`);
      return k;
    };

    const indirectK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(D), () => {
        Return();
      });
      const g = drawGroupBuf.element(i) as unknown as NU;
      const raw = atomicLoad(counters.element(g)) as unknown as NU;
      const cap = capBuf.element(g) as unknown as NU;
      const n = raw.greaterThan(cap).select(cap, raw);
      indirectStore.element(i.mul(5).add(1)).assign(n);
    })().compute(D);
    indirectK.setName('vegIndirect');

    this.kernels = [
      clearK,
      makeCull(this.scatter.trees, 'trees'),
      makeCull(this.scatter.understory, 'under'),
      makeCull(this.scatter.extras, 'extras'),
      makeCull(this.scatter.stones, 'extras'),
      indirectK,
    ];
  }

  /** wire the CSM rig (cascade cameras feed the caster cull) */
  setCSM(csm: object | null): void {
    this.csm = csm;
  }

  /** per-frame: update frustum/camera uniforms, run cull+indirect computes */
  update(renderer: Renderer, camera: PerspectiveCamera): void {
    this.camU.value.copy(camera.position);
    updateVegViewPos(camera);
    this.projView.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projView);
    const arr = this.planesU.array as Vector4[];
    for (let p = 0; p < 6; p++) {
      const pl = this.frustum.planes[p];
      if (!pl) continue;
      (arr[p] as Vector4).set(pl.normal.x, pl.normal.y, pl.normal.z, pl.constant);
    }
    // cascade ortho frusta → caster-cull planes (read one frame stale —
    // CSMShadowNode positions its lwLights during the upcoming render; the
    // lightMargin slack swallows the lag). Also pins each cascade camera to
    // its caster layer once the lazy CSM init has built the lights.
    interface CascCam {
      projectionMatrix: Matrix4;
      matrixWorldInverse: Matrix4;
      layers: { enable(ch: number): void };
      left?: number;
    }
    const lights = (
      this.csm as { lights?: { shadow?: { camera?: CascCam } }[] } | null
    )?.lights;
    if (lights) {
      const carr = this.planesCsmU.array as Vector4[];
      for (let c = 0; c < CASCADES; c++) {
        const cam = lights[c]?.shadow?.camera;
        if (!cam || !Number.isFinite(cam.left ?? NaN)) continue;
        cam.layers.enable(2 + c);
        this.cascM.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
        this.cascFrustum.setFromProjectionMatrix(this.cascM);
        for (let p = 0; p < 6; p++) {
          const pl = this.cascFrustum.planes[p];
          if (!pl) continue;
          (carr[c * 6 + p] as Vector4).set(
            pl.normal.x,
            pl.normal.y,
            pl.normal.z,
            pl.constant,
          );
        }
      }
    }
    for (const k of this.kernels) {
      renderer.compute(k as Parameters<Renderer['compute']>[0]);
    }
    this.frame++;
    if (this.frame % 90 === 0 && !this.reading) {
      this.reading = true;
      void this.readStats(renderer);
    }
  }

  /** HUD stats (throttled async readback of the group counters) */
  counterSnapshot(): Record<string, number> {
    return this.hud;
  }

  private async readStats(renderer: Renderer): Promise<void> {
    try {
      const attr = (this.counters as unknown as { value: unknown }).value;
      const ab = await renderer.getArrayBufferAsync(
        attr as Parameters<Renderer['getArrayBufferAsync']>[0],
      );
      const counts = new Uint32Array(ab);
      let hero = 0;
      let r1 = 0;
      let r2 = 0;
      let imp = 0;
      let under = 0;
      let extras = 0;
      let cast = 0;
      let tris = 0;
      for (let g = 0; g < GROUPS; g++) {
        const n = Math.min(counts[g] ?? 0, this.groupCaps[g] ?? 0);
        tris += n * (this.groupTris[g] ?? 0);
        if (g >= MAIN_GROUPS) {
          cast += n;
        } else if (g < 48) {
          if (g % 2 === 0) r1 += n;
          else r2 += n;
        } else if (g < 54) imp += n;
        else if (g < 82) under += n;
        else if (g < 146) extras += n;
        else hero += n;
      }
      this.hud = {
        'veg.hero': hero,
        'veg.r1': r1,
        'veg.r2': r2,
        'veg.imp': imp,
        'veg.underDrawn': under,
        'veg.extraDrawn': extras,
        'veg.cast': cast,
        'veg.tris': Math.round(tris),
      };
    } finally {
      this.reading = false;
    }
  }
}
