/**
 * Cotton-grass — Eriophorum vaginatum (hare's-tail cotton-grass), the tussock-
 * forming sedge that defines Estonian raised bogs (raba). A leaf-only FOLIAGE
 * plant with a two-material split: a dense arching TUSSOCK of fine keeled sedge
 * blades (green, drying straw at the tips) + several erect culms each topped by
 * ONE unmistakable pure-white fluffy COTTON seed-head held above the clump.
 *
 * Reference (real photos + botany, matched at close range):
 *   - Wikipedia "Eriophorum vaginatum" / RHS / Minnesota Wildflowers / iNaturalist:
 *     single upright cottony seed-head per culm (unlike E. angustifolium's several
 *     drooping heads); tussock of stiff fine grass-like (actually sedge) leaves,
 *     narrow 1-3 mm, keeled/triangular, arching, 20-50 cm.
 *   - IUCN UK Peatland Programme species showcase: the white "cotton" is a dense
 *     mass of hundreds of elongated silky pappus bristles (perianth), ~2-5 cm,
 *     rounded/obovoid, held clear of the leaf clump on thin pale culms.
 *
 * The hero is the cotton head: it is built as a dense radial "fur" of ~40 fine
 * tapered white bristles (each a 2-plane cross that tapers to a feathery point and
 * sags under gravity), emanating from a small obovoid core — a genuinely cottony
 * puff, not a ball-on-a-stick.
 *
 * ROUTING: FOLIAGE (leaf-only). Because the head is PURE WHITE against GREEN
 * blades — and the leaf pool carries ONE tint per geometry — this returns TWO
 * leaf-class geometries (like buildShrub returns bark+crown, but both foliage):
 *   { blades, cotton } → integrate as two leaf pools bound to the same instance
 *   (green tint on `blades` incl. culms; near-white tint on `cotton`). vdata.x
 *   marks the parts too: 0 = blade/culm (green), 1 = cotton bristle (white).
 *
 * vdata: x part/hue (0 blade/culm, 1 cotton) · y sway flex (0 base .. 1 tip) ·
 *        z sway phase · w AO / luminance.
 */

import {
  BufferGeometry,
  Color,
  DoubleSide,
  Float32BufferAttribute,
  Group,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  SRGBColorSpace,
  Vector3,
} from 'three';
import type { Rng } from '../../core/Seed';
import { MeshGrower } from '../TubeMesh';
import { BOG_LOD_NATIVE, type BogLodCtx } from './BogLod';

// ── recommended integration params (the integrator wires enum/VegLibrary/etc.) ──

/** Real-world height: leaf clump ~0.35 m; culms + cotton heads reach ~0.45-0.5 m. */
export const COTTONGRASS_HEIGHT: readonly [number, number] = [0.42, 0.5];
/** Green blade+culm tint (leaf pool). Mid yellow-green with a high hueVar so the
 *  per-vertex vdata.x jitter spreads warm↔cool; blade TIPS dry straw (see below). */
export const COTTONGRASS_BLADE_TINT = { r: 0.13, g: 0.2, b: 0.045, hueVar: 0.34 };
/** Cotton seed-head tint (separate white leaf pool). Near-white, faintly warm. */
export const COTTONGRASS_COTTON_TINT = { r: 0.92, g: 0.92, b: 0.89, hueVar: 0.05 };
/** Short-range dense bog cover — cull distance. */
export const COTTONGRASS_CLS_MAXDIST = 90;
/** This species routes as FOLIAGE (leaf-only), returned as two leaf geometries. */
export const COTTONGRASS_ROUTING = 'FOLIAGE' as const;

const UP = new Vector3(0, 1, 0);
const X = new Vector3(1, 0, 0);

// ───────────────────────────── blades (tussock) ──────────────────────────────

/** scratch */
const _T = new Vector3();
const _side = new Vector3();
const _face = new Vector3();
const _p1p0 = new Vector3();
const _p2p1 = new Vector3();
const _C = new Vector3();
const _L = new Vector3();
const _M = new Vector3();
const _R = new Vector3();
const _Nl = new Vector3();
const _Nr = new Vector3();

/** blade / culm curve segments per LOD detail tier (emission-only — no rng). */
const SEGS_BY_DETAIL = [4, 3, 2] as const;

/** One keeled sedge blade: a curved (quadratic-Bezier) tapered ribbon folded along
 *  a raised midrib so the cross-section is a shallow tent (keel) — reads as a stiff
 *  triangular sedge blade and catches light on two facets. LOD: `widthMul` widens
 *  the survivor blade (Cook §3.3), `segs` coarsens the curve; both emission-only,
 *  so the rng draw order is identical at every rung. */
function addBlade(
  g: MeshGrower,
  base: Vector3,
  az: number,
  rng: Rng,
  widthMul: number,
  segs: number,
): void {
  // erect-to-arching stiff tussock: bias toward upright blades (arch^1.4), and
  // keep tips well off the ground so nothing sprawls flat like a stray wire.
  const arch = Math.pow(rng.float(), 1.4); // 0 = near-vertical, 1 = most arching
  const len = 0.18 + rng.float() * 0.2; // 0.18 .. 0.38 m
  // tip horizontal reach; capped so the tussock stays compact (~≤0.3 m across)
  let reach = 0.08 + 0.42 * arch;
  if (reach * len > 0.15) reach = 0.15 / len;
  const tipUp = 0.9 - 0.4 * arch; // tips stay high (ascending), never drooping to ground
  const ctrlUp = 0.62 + 0.26 * arch; // arch-over height of the control point
  const ctrlOut = 0.22 + 0.12 * arch;
  const hw0 = (0.0009 + rng.float() * 0.0008) * widthMul; // base half-width (0.9-1.7 mm blade at LOD0)
  const keel = 0.55 + rng.float() * 0.4; // keel height as fraction of half-width
  const phase = rng.float() * Math.PI * 2;
  // Blades carry their warm/cool hue jitter in vdata.x (like ferns/trees). The nanite
  // leaf resolve reads vdata.x≥~0.85 as the "cotton head" part-id, so keep the jitter
  // clear of that band. geometryToSource clamps vdata to [0,1], which would crush the
  // COOL half of a raw ±jitter to 0 (6.2) — so author it 0.5-CENTRED: 0.5+jit/2 spans
  // [0.2,0.8], preserving the signed jitter AND staying below the petal band.
  const hueJit = 0.5 + (rng.float() * 2 - 1) * 0.3;

  const ox = Math.cos(az);
  const oz = Math.sin(az);
  // Bezier control points
  const p0 = base;
  const p2x = base.x + ox * reach * len;
  const p2y = base.y + tipUp * len;
  const p2z = base.z + oz * reach * len;
  const p1x = base.x + ox * reach * len * ctrlOut;
  const p1y = base.y + ctrlUp * len;
  const p1z = base.z + oz * reach * len * ctrlOut;

  let prev: [number, number, number] | null = null;
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const mt = 1 - t;
    // point
    _C.set(
      mt * mt * p0.x + 2 * mt * t * p1x + t * t * p2x,
      mt * mt * p0.y + 2 * mt * t * p1y + t * t * p2y,
      mt * mt * p0.z + 2 * mt * t * p1z + t * t * p2z,
    );
    // tangent B'(t)
    _p1p0.set(p1x - p0.x, p1y - p0.y, p1z - p0.z);
    _p2p1.set(p2x - p1x, p2y - p1y, p2z - p1z);
    _T.copy(_p1p0).multiplyScalar(2 * mt).addScaledVector(_p2p1, 2 * t);
    if (_T.lengthSq() < 1e-9) _T.copy(UP);
    _T.normalize();
    // horizontal side + face normal
    _side.copy(UP).cross(_T);
    if (_side.lengthSq() < 1e-8) _side.copy(X);
    _side.normalize();
    _face.copy(_T).cross(_side).normalize();
    if (_face.y < 0) _face.negate(); // keep the fold facing up

    const hw = hw0 * Math.pow(1 - t, 0.7); // taper to a fine point
    const keelH = keel * hw;
    _L.copy(_C).addScaledVector(_side, -hw);
    _M.copy(_C).addScaledVector(_face, keelH);
    _R.copy(_C).addScaledVector(_side, hw);
    _Nl.copy(_face).multiplyScalar(0.8).addScaledVector(_side, -0.6).normalize();
    _Nr.copy(_face).multiplyScalar(0.8).addScaledVector(_side, 0.6).normalize();

    const flex = t; // tips sway
    const ao = 0.5 + 0.5 * t; // interior/base darker, tips bright
    const li = g.vertex(_L.x, _L.y, _L.z, _Nl.x, _Nl.y, _Nl.z, 0, t, hueJit, flex, phase, ao);
    const mi = g.vertex(_M.x, _M.y, _M.z, _face.x, _face.y, _face.z, 0.5, t, hueJit, flex, phase, ao);
    const ri = g.vertex(_R.x, _R.y, _R.z, _Nr.x, _Nr.y, _Nr.z, 1, t, hueJit, flex, phase, ao);
    if (prev) {
      const [pl, pm, pr] = prev;
      g.quad(pl, pm, mi, li); // left facet
      g.quad(pm, pr, ri, mi); // right facet
    }
    prev = [li, mi, ri];
  }
}

// ─────────────────────────── culms (flower stalks) ───────────────────────────

const _cT = new Vector3();
const _cP1 = new Vector3();
const _cP2 = new Vector3();
const _perp1 = new Vector3();
const _perp2 = new Vector3();

/** Thin erect culm: a tapered 2-plane cross (a slim stalk) following a gently
 *  leaning + nodding Bezier from the clump base up to the head. Pale green
 *  (goes in the blade/green geometry). Returns the head anchor point + tip dir.
 *  LOD: culms are STRUCTURAL (each carries the hero cotton head) so they are
 *  never pruned; they widen ×`widthMul` so the ~1 mm stalk stays a coherent
 *  line at range instead of dissolving into the speckle being fixed. */
function addCulm(
  g: MeshGrower,
  base: Vector3,
  az: number,
  rng: Rng,
  widthMul: number,
  segs: number,
): { top: Vector3; dir: Vector3 } {
  const h = 0.4 + rng.float() * 0.08; // 0.40 .. 0.48 m
  const leanR = 0.02 + rng.float() * 0.06; // outward lean of the top
  const nod = 0.01 + rng.float() * 0.03; // extra forward nod of the very tip
  const ox = Math.cos(az);
  const oz = Math.sin(az);
  const phase = rng.float() * Math.PI * 2;

  const p0 = base;
  const p2x = base.x + ox * (leanR + nod);
  const p2y = base.y + h;
  const p2z = base.z + oz * (leanR + nod);
  const p1x = base.x + ox * leanR * 0.4;
  const p1y = base.y + h * 0.55;
  const p1z = base.z + oz * leanR * 0.4;

  const r0 = 0.0011 * widthMul; // base radius ~1.1 mm at LOD0
  const r1 = 0.0006 * widthMul;
  // build both perpendicular planes as tapered strips
  const rings: { ids: number[] }[] = [];
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const mt = 1 - t;
    _cT.set(
      mt * mt * p0.x + 2 * mt * t * p1x + t * t * p2x,
      mt * mt * p0.y + 2 * mt * t * p1y + t * t * p2y,
      mt * mt * p0.z + 2 * mt * t * p1z + t * t * p2z,
    );
    _cP1.set(p1x - p0.x, p1y - p0.y, p1z - p0.z);
    _cP2.set(p2x - p1x, p2y - p1y, p2z - p1z);
    _T.copy(_cP1).multiplyScalar(2 * mt).addScaledVector(_cP2, 2 * t);
    if (_T.lengthSq() < 1e-9) _T.copy(UP);
    _T.normalize();
    _perp1.copy(_T).cross(UP);
    if (_perp1.lengthSq() < 1e-8) _perp1.copy(X);
    _perp1.normalize();
    _perp2.copy(_T).cross(_perp1).normalize();
    const r = r0 + (r1 - r0) * t;
    const ao = 0.6 + 0.4 * t;
    // 4 cross verts (two planes), normals point out along the opposing perp
    const a = g.vertex(_cT.x - _perp1.x * r, _cT.y - _perp1.y * r, _cT.z - _perp1.z * r, -_perp1.x, -_perp1.y, -_perp1.z, 0, t, 0.2, t, phase, ao);
    const b = g.vertex(_cT.x + _perp1.x * r, _cT.y + _perp1.y * r, _cT.z + _perp1.z * r, _perp1.x, _perp1.y, _perp1.z, 1, t, 0.2, t, phase, ao);
    const c = g.vertex(_cT.x - _perp2.x * r, _cT.y - _perp2.y * r, _cT.z - _perp2.z * r, -_perp2.x, -_perp2.y, -_perp2.z, 0, t, 0.2, t, phase, ao);
    const d = g.vertex(_cT.x + _perp2.x * r, _cT.y + _perp2.y * r, _cT.z + _perp2.z * r, _perp2.x, _perp2.y, _perp2.z, 1, t, 0.2, t, phase, ao);
    rings.push({ ids: [a, b, c, d] });
  }
  for (let i = 0; i < segs; i++) {
    const lo = rings[i]!.ids;
    const hi = rings[i + 1]!.ids;
    g.quad(lo[0]!, lo[1]!, hi[1]!, hi[0]!); // plane 1
    g.quad(lo[2]!, lo[3]!, hi[3]!, hi[2]!); // plane 2
  }

  const top = new Vector3(p2x, p2y, p2z);
  const dir = new Vector3(p2x - p1x, p2y - p1y, p2z - p1z).normalize();
  return { top, dir };
}

// ─────────────────────────── cotton seed-head (hero) ──────────────────────────

const _bdir = new Vector3();
const _broot = new Vector3();
const _btip = new Vector3();
const _bp1 = new Vector3();
const _bq1 = new Vector3();
const _bq2 = new Vector3();
const _bow = new Vector3();
const _off = new Vector3();

/** One cotton fibre: a fine tapered 2-plane cross from an interior ROOT to a shell
 *  TIP, gently bowed (a little outward + gravity) so it reads silky. Because roots
 *  are scattered through the head and tips lie on a rounded shell, neighbouring
 *  fibres CRISSCROSS and overlap into an opaque cloud with a soft rounded edge —
 *  cotton-wool, not a radial spike-star. */
function addFiber(
  g: MeshGrower,
  root: Vector3,
  tip: Vector3,
  center: Vector3,
  phase: number,
  widthMul: number,
  coarse: boolean,
): void {
  _bdir.copy(tip).sub(root);
  const len = _bdir.length();
  if (len < 1e-5) return;
  _bdir.multiplyScalar(1 / len);
  // bow: midpoint pushed slightly away from head centre + a touch of gravity
  _bp1.copy(root).add(tip).multiplyScalar(0.5);
  _bow.copy(_bp1).sub(center);
  _bow.addScaledVector(_bdir, -_bow.dot(_bdir)); // component ⊥ fibre axis
  _bp1.addScaledVector(_bow, 0.18).addScaledVector(UP, -len * 0.07);
  // width planes ⊥ the fibre
  _bq1.copy(_bdir).cross(UP);
  if (_bq1.lengthSq() < 1e-9) _bq1.copy(_bdir).cross(X);
  _bq1.normalize();
  _bq2.copy(_bdir).cross(_bq1).normalize();

  const w0 = 0.0007 * widthMul; // root (0.7 mm at LOD0)
  const w1 = 0.00052 * widthMul; // mid
  const w2 = 0.00006 * widthMul; // feathery point
  // coarse tier: drop the bow midpoint — one segment per plane (emission-only).
  const pts: [Vector3, number, number][] = coarse
    ? [
        [root, w0, 0],
        [tip, w2, 1],
      ]
    : [
        [root, w0, 0],
        [_bp1, w1, 0.5],
        [tip, w2, 1],
      ];
  for (const [plane, nrm] of [
    [_bq1, _bq2],
    [_bq2, _bq1],
  ] as [Vector3, Vector3][]) {
    let prev: number[] | null = null;
    for (const [p, w, tt] of pts) {
      // The cotton head is a RIGID sub-object welded onto the culm tip (flex 1.0),
      // so every fibre vertex carries that ONE constant attach flex — the head
      // translates as one with its culm instead of lagging/shearing (1a).
      const flex = 1;
      const ao = 0.9 + 0.1 * tt;
      const a = g.vertex(p.x - plane.x * w, p.y - plane.y * w, p.z - plane.z * w, nrm.x, nrm.y, nrm.z, 0, tt, 1, flex, phase, ao);
      const b = g.vertex(p.x + plane.x * w, p.y + plane.y * w, p.z + plane.z * w, nrm.x, nrm.y, nrm.z, 1, tt, 1, flex, phase, ao);
      if (prev) g.quad(prev[0]!, prev[1]!, b, a);
      prev = [a, b];
    }
  }
}

/** A random point in a unit ball (biased toward the surface by `shellBias`). */
function ballPoint(out: Vector3, rng: Rng, shellBias: number): void {
  out.set(rng.gauss(), rng.gauss(), rng.gauss());
  const l = out.length() || 1;
  const r = shellBias + (1 - shellBias) * Math.cbrt(rng.float());
  out.multiplyScalar(r / l);
}

/** A dense, rounded cotton-wool puff at a culm top: fibres run from scattered
 *  interior roots to tips on a slightly obovoid, gravity-sagged shell, crisscrossing
 *  into an opaque soft ball (hundreds of pappus hairs) — the E. vaginatum single
 *  hare's-tail head, nodding on its culm. */
function addCottonHead(g: MeshGrower, top: Vector3, rng: Rng, lod: BogLodCtx): void {
  const R = 0.011 + rng.float() * 0.004; // head radius ~1.1-1.5 cm (head ~2-3 cm)
  const phase = rng.float() * Math.PI * 2;
  // head centre sits just above the culm tip; obovoid (taller than wide)
  _C.copy(top).addScaledVector(UP, R * 0.55);

  const n = 150 + rng.int(50);
  for (let i = 0; i < n; i++) {
    // tip on a jittered shell, biased slightly upward/obovoid + gravity sag below
    ballPoint(_off, rng, 0.86);
    _off.y = _off.y * 1.18 + 0.08; // obovoid, lift the mass up
    _btip.copy(_C).addScaledVector(_off, R);
    const upness = Math.max(-1, Math.min(1, _off.y));
    _btip.y -= R * 0.42 * (1 - upness) * rng.float(); // lower fibres sag (silky nod)
    // root scattered through the interior (crisscross fill → opaque, un-spiky)
    ballPoint(_off, rng, 0);
    _broot.copy(_C).addScaledVector(_off, R * 0.42);
    // LOD: prune whole fibres to λ (like conifer intra-spray needles), survivors
    // width ×1/λ — the puff keeps its covered area with fewer, silkier-wide hairs.
    addFiber(lod.target(g, _btip, i), _broot, _btip, _C, phase, lod.widthMul, lod.detail === 2);
  }
}

// ─────────────────────────────── assembly ────────────────────────────────────

const _cbase = new Vector3();

/** Integration builder. Returns two leaf-class geometries: `blades` (green tussock
 *  + culms) and `cotton` (pure-white seed-heads). Both bind to one instance.
 *  `lod` (default native = LOD0, byte-identical) regenerates a coarser crown-LOD
 *  rung: blades/fibres pruned to λ + widened ×1/λ, culms widened, curve segments
 *  coarsened — same seed, same rng draw order (pruned elements build into the
 *  ctx's discard sink), so survivors match LOD0 exactly except for width. */
export function buildCottonGrass(rng: Rng, lod: BogLodCtx = BOG_LOD_NATIVE): {
  blades: BufferGeometry;
  cotton: BufferGeometry;
  bladeTris: number;
  cottonTris: number;
} {
  const greenG = new MeshGrower();
  const whiteG = new MeshGrower();
  const segs = SEGS_BY_DETAIL[lod.detail] as number;

  // dense tussock of fine blades fanning from a tight base
  const blades = 74 + rng.int(26);
  for (let i = 0; i < blades; i++) {
    const az = rng.float() * Math.PI * 2;
    const rBase = rng.float() * 0.03; // clump base radius (2-3 cm tight tussock)
    _cbase.set(Math.cos(az) * rBase, 0, Math.sin(az) * rBase);
    // blade splays roughly outward from where it sits in the clump
    const bladeAz = az + (rng.float() - 0.5) * 1.4;
    addBlade(lod.target(greenG, _cbase, i), _cbase, bladeAz, rng, lod.widthMul, segs);
  }

  // a few culms, each with ONE white cotton head above the clump
  const culms = 3 + rng.int(4); // 3-6
  for (let i = 0; i < culms; i++) {
    const az = (i / culms) * Math.PI * 2 + rng.float() * 0.9;
    const rBase = rng.float() * 0.02;
    _cbase.set(Math.cos(az) * rBase, 0, Math.sin(az) * rBase);
    const { top } = addCulm(greenG, _cbase, az, rng, lod.widthMul, segs);
    addCottonHead(whiteG, top, rng, lod);
  }

  return {
    blades: greenG.build(),
    cotton: whiteG.build(),
    bladeTris: greenG.triCount,
    cottonTris: whiteG.triCount,
  };
}

// ─────────────────────────────── preview ─────────────────────────────────────

/** Bake per-vertex colours for the green geometry so the QA render shows the real
 *  look: mid yellow-green blades drying STRAW at the tips (uv.v gradient), with a
 *  little per-blade hue jitter (vdata.x) and interior AO (vdata.w). Engine-side the
 *  same read comes from the tint + vdata channels; this is preview fidelity only. */
function greenVertexColors(geo: BufferGeometry): void {
  const uv = geo.getAttribute('uv');
  const vd = geo.getAttribute('vdata');
  const n = uv.count;
  const cols = new Float32Array(n * 3);
  const base = new Color().setRGB(0.34, 0.45, 0.15, SRGBColorSpace);
  const straw = new Color().setRGB(0.7, 0.62, 0.33, SRGBColorSpace);
  const c = new Color();
  for (let i = 0; i < n; i++) {
    const v = uv.getY(i); // 0 base .. 1 tip
    const hue = vd.getX(i); // -1 .. 1
    const ao = vd.getW(i); // 0 .. 1
    const strawT = Math.max(0, Math.min(1, (v - 0.5) / 0.5));
    const s = strawT * strawT;
    c.copy(base).lerp(straw, s);
    const warm = 1 + hue * 0.12; // subtle warm/cool jitter
    const lum = 0.66 + 0.34 * ao;
    cols[i * 3] = c.r * warm * lum;
    cols[i * 3 + 1] = c.g * lum;
    cols[i * 3 + 2] = c.b * (2 - warm) * lum;
  }
  geo.setAttribute('color', new Float32BufferAttribute(cols, 3));
}

/** Fully-assembled, materialed plant for the QA harness. Green tussock + culms in
 *  one material; pure-white cotton heads in a separate near-white material. Meters,
 *  +Y up, base at y=0. */
export function buildPreview(rng: Rng): Object3D {
  const { blades, cotton } = buildCottonGrass(rng);
  greenVertexColors(blades);

  const group = new Group();

  const greenMat = new MeshStandardMaterial({
    color: 0xffffff,
    vertexColors: true,
    roughness: 0.85,
    metalness: 0,
    side: DoubleSide,
  });
  group.add(new Mesh(blades, greenMat));

  // pure-white silky cotton: a touch of emissive keeps it reading white/silky in
  // shade under ACES, roughness 1 for a soft matte fibre look.
  const cottonMat = new MeshStandardMaterial({
    color: new Color().setRGB(0.95, 0.95, 0.93, SRGBColorSpace),
    emissive: new Color().setRGB(0.22, 0.22, 0.21, SRGBColorSpace),
    roughness: 1,
    metalness: 0,
    side: DoubleSide,
  });
  group.add(new Mesh(cotton, cottonMat));

  return group;
}
