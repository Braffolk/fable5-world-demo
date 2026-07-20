/**
 * Sphagnum — the raised-bog peat-moss CARPET. This is not a single plant: it is a
 * ground-covering CUSHION mesh made of many packed moss shoots, each topped by a
 * compact star CAPITULUM (a pom-pom rosette of densely packed young branch tips).
 * A sphagnum lawn/hummock, seen from above and at low angle, reads as a soft
 * bumpy field of packed capitula over a peaty felt, in rusty-red → ochre → green.
 *
 * Reference (real Sphagnum morphology, grounded FIRST — not guessed):
 *   • CAPITULUM — "at the top of the plant is a compact cluster of young branches
 *     called the capitulum, giving a characteristic tuft-like / pom-pom
 *     appearance" (IUCN UK Peatland Programme, Species Showcase: Sphagnum;
 *     Community Wetlands Forum, "A Beginner's Guide to the Sphagnum Mosses of
 *     Raised Bogs"). From above each capitulum is a small rounded rosette with a
 *     denser central bud and radiating spreading branch tips — a domed star.
 *   • CARPET / MICROTOPOGRAPHY — "very small, but grow closely together forming
 *     spongy carpets; hummocks are created when the mosses grow to form mounds";
 *     species form lawns on the flat surface and hummocks/mounds, producing the
 *     bog's small-scale surface patterning (IUCN UK; Moors for the Future, Moor
 *     Moss Field Guide). So the visible unit is the PACKED CAPITULA FIELD over a
 *     mounded cushion, never a lone stalk.
 *   • COLOUR — species-dependent across a bog: S. rubellum / S. magellanicum
 *     (medium) reds & crimsons on hummocks, S. papillosum / S. fuscum ochres &
 *     browns, S. cuspidatum yellow-greens in wet hollows. A real carpet MIXES
 *     these, so the mesh spreads a per-capitulum hue over rust-red↔ochre↔green.
 *   Sources: iucn-uk-peatlandprogramme.org/biodiversity/species-showcase-sphagnum,
 *     communitywetlandsforum.ie (Beginner's Guide to Sphagnum Mosses of Raised
 *     Bogs), moorsforthefuture.org.uk (Moor Moss Field Guide), JNCC Sphagnum: a
 *     field guide (M. O. Hill).
 *
 * ── Routing: CARPET (the ground-cover layer), NOT understory. ─────────────────
 * Rigid / non-wind channel (vdata.y ≈ 0): moss does not sway. Two builders:
 *   • buildSphagnumPatch  — a low LAWN cushion TILE, gap-free packed capitula on a
 *     gently undulating felt. This is the near-field carpet MESH (also voxelised
 *     for the mid band). Footprint SPHAGNUM_PATCH_FOOTPRINT, base at y=0.
 *   • buildSphagnumHummock — a raised domed cushion (the sparse HERO class) that
 *     gives silhouette relief right around the camera. Same capitula, tall mound.
 * Both are prune-and-preserve friendly: they take a BogLodCtx (default native =
 * LOD0) so the seam pass drives the same crown-LOD ladder + voxel sibling as the
 * other bog pools — whole capitula prune to λ, survivors widen ×1/λ, arm count
 * drops on the detail tier, the felt sheet is structural (kept, widened).
 *
 * vdata layout (consumed by the leaf/foliage material, per MeshGrower):
 *   x = leaf-band hue 0..1 (0 green ↔ 0.5 ochre ↔ 1 red; the pool spreads
 *       foliageColor by it) · y = sway flex (≈0, RIGID carpet) · z = phase (0) ·
 *       w = baked AO (capitulum tops bright, valleys/felt dark → real shaded depth).
 */

import { Group, Mesh, MeshStandardMaterial, DoubleSide, Vector3 } from 'three';
import type { BufferGeometry, Object3D } from 'three';
import { mix32, type Rng } from '../../core/Seed';
import { MeshGrower } from '../TubeMesh';
import { BOG_LOD_NATIVE, type BogLodCtx } from './BogLod';

// ── recommended integration params ──────────────────────────────────────────
/** Lawn patch tile footprint (m, square). The seam pass scatters patches on a
 *  deterministic carpet grid at ≈ this step (edges skirt to y=0 + capitula
 *  overhang the rim → gap-free tiling). */
export const SPHAGNUM_PATCH_FOOTPRINT = 0.45;
/** Hero hummock cushion footprint (m) and mound-height range (m). */
export const SPHAGNUM_HUMMOCK_FOOTPRINT = 0.46;
export const SPHAGNUM_HUMMOCK_HEIGHT: readonly [number, number] = [0.16, 0.26];

/** POOL tint (foliageColor) — LINEAR albedo, NOT a display colour. This feeds BOTH
 *  the near leaf resolve (mesh word 7 base) and the voxel-brick albedo, so the near
 *  and far bands share one value. The earlier picks (0.4/0.27/0.13 → 0.34/0.22/0.1)
 *  were authored as if they were sRGB display values: a linear albedo that bright,
 *  lit by full sun and gamma-encoded to the screen, reads as pale SAND (≈ sRGB
 *  0.61/0.51/0.35 — the boot-2 failure). The muted-green tree/bog foliage tints sit
 *  at linear channels ~0.04–0.24 (Heather 0.09/0.16/0.06, LabradorTea 0.13/0.22/
 *  0.09); a dusty RUST-OCHRE moss belongs in that same regime, red-dominant. This
 *  displays as a desaturated rusty ochre-brown, never sand. hueVar spreads the
 *  per-capitulum vdata.x hue warm-ward (the shared leaf resolve applies k = vdata.x·
 *  hueVar ≥ 0 ⇒ only the warm branch fires: the carpet breathes ochre↔rust; the
 *  vivid green-hollow band folds into the ochre-rust dominant — see report). The
 *  channels sit CLOSE together (g≈0.8·r, b lifted off zero) so the display colour
 *  is a DUSTY, low-saturation rust-tan, not orange paint — a wide r≫g≫b gap read as
 *  pure orange (boot-verified). The shared leaf resolve only ever WARMS (k = vdata.x·
 *  hueVar ≥ 0 ⇒ base → rustier, never cooler), so hueVar stays gentle (0.32) — big
 *  hueVar just re-saturates the warm peaks back to orange. The olive-green hollow
 *  band is unreachable through the shared warm-only palette without editing it for
 *  ALL foliage (out of scope); the carpet spans dusty tan-olive → dusty rust. */
export const SPHAGNUM_FOLIAGE = { r: 0.115, g: 0.092, b: 0.055, hueVar: 0.32 };
/** The three species colour bands the carpet mixes (reference tints; the preview
 *  materials + the pool hueVar reproduce this rust-red→ochre→green spread). */
export const SPHAGNUM_BANDS = {
  red: { r: 0.49, g: 0.12, b: 0.09 }, // S. rubellum / magellanicum
  ochre: { r: 0.56, g: 0.42, b: 0.13 }, // S. papillosum / fuscum
  green: { r: 0.31, g: 0.42, b: 0.14 }, // S. cuspidatum
} as const;
/** Peaty felt beneath the capitula (fills gaps; darker reddish-brown). */
export const SPHAGNUM_FELT = { r: 0.23, g: 0.14, b: 0.08 };

/** Carpet distances (m) — consumed by the SPHAGNUM_CARPET spec (carpet/SphagnumCarpet.ts,
 *  the CARPET-layer routing entry); tune in-engine there. */
export const SPHAGNUM_VOX_NEAR = 30; // patch mesh → voxel sibling handoff
export const SPHAGNUM_MAX_DIST = 160; // voxel band end (≤ uband 160)

// ── deterministic value noise (base cushion micro-relief) ────────────────────
function hash2(ix: number, iz: number, seed: number): number {
  const h = Math.imul(ix | 0, 73856093) ^ Math.imul(iz | 0, 19349663) ^ Math.imul(seed | 0, 0x9e3779b9);
  return (mix32(h >>> 0) >>> 0) / 4294967296;
}
function vnoise(x: number, z: number, seed: number): number {
  const ix = Math.floor(x);
  const iz = Math.floor(z);
  const fx = x - ix;
  const fz = z - iz;
  const sx = fx * fx * (3 - 2 * fx);
  const sz = fz * fz * (3 - 2 * fz);
  const a = hash2(ix, iz, seed);
  const b = hash2(ix + 1, iz, seed);
  const c = hash2(ix, iz + 1, seed);
  const d = hash2(ix + 1, iz + 1, seed);
  return (a * (1 - sx) + b * sx) * (1 - sz) + (c * (1 - sx) + d * sx) * sz;
}

interface Preset {
  footprint: number;
  /** dome apex height at centre (m). Lawn small, hummock tall. */
  domeH: number;
  /** micro-relief noise amplitude (m). */
  noiseAmp: number;
  /** base grid resolution (verts per side − 1). */
  baseN: number;
  /** capitulum planting step (m) and jitter (m). */
  capStep: number;
  capJit: number;
  capSizeMin: number;
  capSizeMax: number;
  /** discrete cushion (hummock) → fade capitula density to 0 at the rim; a lawn
   *  tile keeps full density to the edge so tiles abut gap-free. */
  radialFade: boolean;
  noiseSeed: number;
}

const LAWN: Omit<Preset, 'noiseSeed'> = {
  footprint: SPHAGNUM_PATCH_FOOTPRINT,
  domeH: 0.03,
  noiseAmp: 0.017,
  baseN: 12,
  capStep: 0.034,
  capJit: 0.011,
  capSizeMin: 0.032,
  capSizeMax: 0.05,
  radialFade: false,
};
const HUMMOCK: Omit<Preset, 'noiseSeed'> = {
  footprint: SPHAGNUM_HUMMOCK_FOOTPRINT,
  domeH: 0.21,
  noiseAmp: 0.028,
  baseN: 14,
  capStep: 0.03,
  capJit: 0.011,
  capSizeMin: 0.03,
  capSizeMax: 0.046,
  radialFade: true,
};

/** normalised radius 0 (centre) .. 1 (rim of the inscribed circle) for a square
 *  footprint of half-width `half`. */
function radial(x: number, z: number, half: number): number {
  return Math.min(1, Math.hypot(x, z) / half);
}

/** Base cushion surface height (m) at (x,z): a smooth dome windowed to 0 at the
 *  rim + windowed value-noise micro-relief. Tiles meet at y≈0. */
function baseHeight(x: number, z: number, p: Preset, half: number): number {
  const rn = radial(x, z, half);
  const win = 1 - rn * rn; // 1 centre → 0 rim
  const dome = p.domeH * win * win;
  const cell = 0.11;
  const n1 = vnoise(x / cell, z / cell, p.noiseSeed);
  const n2 = vnoise(x / (cell * 0.5) + 31.7, z / (cell * 0.5) - 11.3, p.noiseSeed ^ 0x5bd1e995);
  const noise = ((n1 * 0.66 + n2 * 0.34) * 2 - 1) * p.noiseAmp * win;
  return dome + noise;
}

/** Two unit vectors perpendicular to `dir` and to each other. */
function perp(dir: Vector3, u: Vector3, v: Vector3): void {
  const ref = Math.abs(dir.y) < 0.94 ? UP : RIGHT;
  u.crossVectors(ref, dir).normalize();
  v.crossVectors(dir, u).normalize();
}
const UP = new Vector3(0, 1, 0);
const RIGHT = new Vector3(1, 0, 0);
const RIGID = 0.03; // near-zero sway flex (carpet does not move in the wind)

/** Surface normal of the base cushion at (x,z) via central differences. */
function baseNormal(x: number, z: number, p: Preset, half: number, out: Vector3): void {
  const e = 0.01;
  const hx = baseHeight(x + e, z, p, half) - baseHeight(x - e, z, p, half);
  const hz = baseHeight(x, z + e, p, half) - baseHeight(x, z - e, p, half);
  out.set(-hx, 2 * e, -hz).normalize();
}

/**
 * One CAPITULUM: a domed star rosette — a raised central bud, a ring of `armN`
 * radiating branch tips (the arms) that spread out and droop, with valley notches
 * between them. Real folded geometry (ridge down each arm, valleys either side),
 * not a flat painted star. Built in the local frame whose up is the base normal.
 * `hue01` (0 green ↔ 0.5 ochre ↔ 1 red) rides vdata.x → the pool spreads the tint.
 */
function addCapitulum(g: MeshGrower, center: Vector3, up: Vector3, size: number, hue01: number, armN: number, rng: Rng): void {
  const u = new Vector3();
  const v = new Vector3();
  perp(up, u, v);
  const hx = Math.max(0.03, Math.min(0.97, hue01));
  // a CONVEX rounded pom-pom (not a flat spiky star): tall centre bud, a low
  // scalloped rim, radiating ridges (arm tips ride a touch higher than the
  // valley notches → subtle moss texture, rounded outline).
  const domeH = size * (0.62 + rng.float() * 0.22);
  const baseAng = rng.float() * Math.PI * 2;
  const dir = new Vector3();
  const nrm = new Vector3();
  const pos = new Vector3();

  // central bud (bright, highest)
  pos.copy(center).addScaledVector(up, domeH);
  const cIdx = g.vertex(pos.x, pos.y, pos.z, up.x, up.y, up.z, 0.5, 0.5, hx, RIGID, 0, 1);

  const valley: number[] = [];
  const tip: number[] = [];
  for (let k = 0; k < armN; k++) {
    // valley notch between two arms — slightly in + low
    const av = baseAng + (k / armN) * Math.PI * 2;
    dir.copy(u).multiplyScalar(Math.cos(av)).addScaledVector(v, Math.sin(av));
    const rv = size * 0.8 * (0.92 + rng.float() * 0.14);
    const hv = domeH * 0.32;
    pos.copy(center).addScaledVector(dir, rv).addScaledVector(up, hv);
    nrm.copy(dir).multiplyScalar(0.45).add(up).normalize();
    valley.push(g.vertex(pos.x, pos.y, pos.z, nrm.x, nrm.y, nrm.z, 0.5 + Math.cos(av) * 0.5, 0.5 + Math.sin(av) * 0.5, hx, RIGID, 0, 0.62));

    // arm tip — the high scallop of the rim; only slightly further out than the
    // valley (rounded outline, not a splayed star), a touch higher (radiating ridge)
    const at = baseAng + ((k + 0.5) / armN) * Math.PI * 2;
    dir.copy(u).multiplyScalar(Math.cos(at)).addScaledVector(v, Math.sin(at));
    const rt = size * (0.92 + rng.float() * 0.14);
    const ht = domeH * (0.42 + rng.float() * 0.12);
    pos.copy(center).addScaledVector(dir, rt).addScaledVector(up, ht);
    nrm.copy(dir).multiplyScalar(0.35).add(up).normalize();
    tip.push(g.vertex(pos.x, pos.y, pos.z, nrm.x, nrm.y, nrm.z, 0.5 + Math.cos(at) * 0.5, 0.5 + Math.sin(at) * 0.5, hx, RIGID, 0, 0.85));
  }
  for (let k = 0; k < armN; k++) {
    const kn = (k + 1) % armN;
    g.tri(cIdx, valley[k] as number, tip[k] as number);
    g.tri(cIdx, tip[k] as number, valley[kn] as number);
  }
}

interface Parts {
  /** peaty felt cushion — structural, kept at every LOD (widened, never pruned). */
  felt: MeshGrower;
  /** capitula split by species colour band (for the preview's 3 materials). The
   *  integration builder merges all four into ONE geo; vdata.x carries the hue. */
  green: MeshGrower;
  ochre: MeshGrower;
  red: MeshGrower;
  capitula: number;
}

/** Build the felt cushion sheet (a displaced grid over the footprint). */
function buildFelt(g: MeshGrower, p: Preset, half: number, lod: BogLodCtx): void {
  // detail tier thins the sheet grid (structural → kept, just coarser far away).
  const N = Math.max(6, Math.round(p.baseN * (lod.detail === 0 ? 1 : lod.detail === 1 ? 0.75 : 0.55)));
  const nrm = new Vector3();
  const rows: number[][] = [];
  for (let i = 0; i <= N; i++) {
    const z = -half + (i / N) * 2 * half;
    const row: number[] = [];
    for (let j = 0; j <= N; j++) {
      const x = -half + (j / N) * 2 * half;
      const y = baseHeight(x, z, p, half);
      baseNormal(x, z, p, half, nrm);
      const rn = radial(x, z, half);
      const ao = 0.48 + 0.18 * (1 - rn); // felt is the shaded understorey of the carpet
      row.push(g.vertex(x, y, z, nrm.x, nrm.y, nrm.z, (x + half) / (2 * half), (z + half) / (2 * half), 0.5, RIGID, 0, ao));
    }
    rows.push(row);
  }
  for (let i = 0; i < N; i++) {
    const a = rows[i] as number[];
    const b = rows[i + 1] as number[];
    for (let j = 0; j < N; j++) {
      g.quad(a[j] as number, a[j + 1] as number, b[j + 1] as number, b[j] as number);
    }
  }
}

/** Assemble a sphagnum cushion (felt + packed capitula) into colour-band growers. */
function buildSphagnumParts(rng: Rng, p: Preset, lod: BogLodCtx): Parts {
  const felt = new MeshGrower();
  const green = new MeshGrower();
  const ochre = new MeshGrower();
  const red = new MeshGrower();
  const half = p.footprint * 0.5;

  buildFelt(felt, p, half, lod);

  const nrm = new Vector3();
  const center = new Vector3();
  const armCap = lod.detail === 0 ? 99 : lod.detail === 1 ? 6 : 5;
  let salt = 0;
  let capitula = 0;

  // jittered planting grid → densely packed, overlapping capitula (gap-free).
  for (let z = -half; z <= half + 1e-6; z += p.capStep) {
    for (let x = -half; x <= half + 1e-6; x += p.capStep) {
      const jx = x + (rng.float() - 0.5) * 2 * p.capJit;
      const jz = z + (rng.float() - 0.5) * 2 * p.capJit;
      const rn = radial(jx, jz, half);
      if (rn > 1.02) continue;
      // discrete cushion → moss covers the WHOLE mound; only FEATHER the outer
      // rim (rn>0.82) so it ends as a soft cushion, not a square slab. A lawn
      // tile keeps full density to the edge (tiles abut gap-free).
      if (p.radialFade && rn > 0.82 && rng.float() < (rn - 0.82) / 0.2) {
        salt++;
        continue;
      }
      const y = baseHeight(jx, jz, p, half);
      baseNormal(jx, jz, p, half, nrm);
      // capitulum leans partly toward the surface normal, partly upright.
      const up = new Vector3().copy(nrm).multiplyScalar(0.6).addScaledVector(UP, 0.4).normalize();
      center.set(jx, y + 0.003, jz);
      const size = (p.capSizeMin + rng.float() * (p.capSizeMax - p.capSizeMin)) * lod.widthMul;
      // COHERENT colour: low-frequency noise over (x,z) → real bog colour REGIONS
      // (a red hummock here, a green hollow there), not per-head confetti. A small
      // per-head jitter breaks banding. Warm-biased across green(0)↔ochre↔red(1).
      const region = vnoise(jx / 0.17, jz / 0.17, p.noiseSeed ^ 0x1b56c4e9);
      const hue01 = Math.max(0, Math.min(1, 0.52 + (region * 2 - 1) * 0.9 + (rng.float() - 0.5) * 0.2));
      const band = hue01 < 0.4 ? green : hue01 < 0.68 ? ochre : red;
      const kept = lod.keep(center, salt);
      const target = lod.target(band, center, salt);
      const armN = Math.min(armCap, 7 + rng.int(4)); // 7..10 arms
      addCapitulum(target, center, up, size, hue01, armN, rng.fork(`cap${salt}`));
      if (kept) capitula++;
      salt++;
    }
  }
  return { felt, green, ochre, red, capitula };
}

/** Weld a built geometry into `g`, preserving all four vertex streams + index. */
function weld(g: MeshGrower, src: BufferGeometry): void {
  const pos = src.getAttribute('position');
  const nrm = src.getAttribute('normal');
  const uv = src.getAttribute('uv');
  const dat = src.getAttribute('vdata');
  const idx = src.getIndex();
  const base = g.vertCount;
  for (let i = 0; i < pos.count; i++) {
    g.vertex(
      pos.getX(i), pos.getY(i), pos.getZ(i),
      nrm ? nrm.getX(i) : 0, nrm ? nrm.getY(i) : 1, nrm ? nrm.getZ(i) : 0,
      uv ? uv.getX(i) : 0, uv ? uv.getY(i) : 0,
      dat ? dat.getX(i) : 0.5, dat ? dat.getY(i) : RIGID, dat ? dat.getZ(i) : 0, dat ? dat.getW(i) : 1,
    );
  }
  if (idx) {
    for (let i = 0; i < idx.count; i += 3) g.tri(base + idx.getX(i), base + idx.getX(i + 1), base + idx.getX(i + 2));
  } else {
    for (let i = 0; i < pos.count; i += 3) g.tri(base + i, base + i + 1, base + i + 2);
  }
}

/** Merge the four band growers into ONE leaf-pool geometry (vdata.x = hue). */
function mergeParts(parts: Parts): { geo: BufferGeometry; tris: number } {
  const g = new MeshGrower();
  weld(g, parts.felt.build());
  weld(g, parts.green.build());
  weld(g, parts.ochre.build());
  weld(g, parts.red.build());
  return { geo: g.build(), tris: g.triCount };
}

/**
 * Integration builder — LAWN patch TILE (near-field carpet MESH + voxel sibling).
 * `lod` (default native = LOD0, byte-identical) regenerates a coarser crown-LOD
 * rung from the same seed via BogLod. Consumed as a FOLIAGE/leaf pool, rigid chan.
 */
export function buildSphagnumPatch(rng: Rng, lod: BogLodCtx = BOG_LOD_NATIVE): { geo: BufferGeometry; tris: number } {
  const seed = 1 + (rng.int(1 << 20) | 0);
  return mergeParts(buildSphagnumParts(rng, { ...LAWN, noiseSeed: seed }, lod));
}

/**
 * Integration builder — HERO hummock cushion (sparse silhouette class). A raised
 * domed mound of the same packed capitula, density fading at the rim so it reads
 * as a discrete cushion. Same lod/pool contract as the patch.
 */
export function buildSphagnumHummock(rng: Rng, lod: BogLodCtx = BOG_LOD_NATIVE): { geo: BufferGeometry; tris: number } {
  const seed = 1 + (rng.int(1 << 20) | 0);
  const domeH = SPHAGNUM_HUMMOCK_HEIGHT[0] + rng.float() * (SPHAGNUM_HUMMOCK_HEIGHT[1] - SPHAGNUM_HUMMOCK_HEIGHT[0]);
  return mergeParts(buildSphagnumParts(rng, { ...HUMMOCK, domeH, noiseSeed: seed }, lod));
}

// ── preview materials (rusty-red → ochre → green range + peaty felt) ─────────
function feltMat(): MeshStandardMaterial {
  return new MeshStandardMaterial({ color: 0x54401f, roughness: 0.94, metalness: 0, side: DoubleSide });
}
function bandMats(): { green: MeshStandardMaterial; ochre: MeshStandardMaterial; red: MeshStandardMaterial } {
  return {
    green: new MeshStandardMaterial({ color: 0x4f6a24, roughness: 0.72, metalness: 0, side: DoubleSide }),
    ochre: new MeshStandardMaterial({ color: 0x8f6a22, roughness: 0.7, metalness: 0, side: DoubleSide }),
    red: new MeshStandardMaterial({ color: 0x842016, roughness: 0.66, metalness: 0, side: DoubleSide, emissive: 0x1c0705, emissiveIntensity: 0.18 }),
  };
}

function addPartsMeshes(g: Group, parts: Parts, offset: Vector3): void {
  const fm = feltMat();
  const bm = bandMats();
  const meshes = [
    new Mesh(parts.felt.build(), fm),
    new Mesh(parts.green.build(), bm.green),
    new Mesh(parts.ochre.build(), bm.ochre),
    new Mesh(parts.red.build(), bm.red),
  ];
  for (const m of meshes) {
    m.position.copy(offset);
    g.add(m);
  }
}

/**
 * Fully-assembled, materialed QA scene: a lawn carpet tile with two hummock
 * cushions rising from it, so one render shows the packed capitula, the hummocky
 * relief AND the rust-red→ochre→green colour range. Meters, +Y up, base at y=0.
 */
export function buildPreview(rng: Rng): Object3D {
  const g = new Group();
  const lawn = buildSphagnumParts(rng.fork('lawn'), { ...LAWN, noiseSeed: 1 + rng.int(1 << 20) }, BOG_LOD_NATIVE);
  addPartsMeshes(g, lawn, new Vector3(0, 0, 0));

  const spots: Vector3[] = [new Vector3(0.12, 0, -0.09), new Vector3(-0.14, 0, 0.13)];
  for (let i = 0; i < spots.length; i++) {
    const domeH = SPHAGNUM_HUMMOCK_HEIGHT[0] + rng.float() * (SPHAGNUM_HUMMOCK_HEIGHT[1] - SPHAGNUM_HUMMOCK_HEIGHT[0]);
    const hum = buildSphagnumParts(rng.fork(`hum${i}`), { ...HUMMOCK, footprint: 0.3, domeH: domeH * 0.7, noiseSeed: 7 + rng.int(1 << 20) }, BOG_LOD_NATIVE);
    addPartsMeshes(g, hum, spots[i] as Vector3);
  }
  return g;
}
