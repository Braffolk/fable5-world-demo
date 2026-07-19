/**
 * Cloudberry — Rubus chamaemorus. A LOW herbaceous perennial of the raised bog
 * (10–25 cm), NOT a woody shrub and NOT leggy: a person recognises it as a
 * LEAF-DOMINATED little plant — a few big, round, pleated leaves held UP toward
 * the sky on short reddish stalks, with (in season) a single amber berry sitting
 * among / just above the leaves. REBUILD (v2): the first attempt was rejected as
 * too tall & leggy (leaves + berry stranded high on bare stems), the round leaves
 * DROOPED downward, and the foliage was pure green. This build fixes all three.
 *
 * Diagnostic reads (grounded in real reference):
 *   • Habit — "creeping plant, 10–25 cm; fruit borne singly on a mostly upright
 *     stem; leaves grow on straight branchless stalks" and typically form a low
 *     leafy rosette. Stems are SHORT — the leaves are the plant's whole read.
 *   • The SIGNATURE leaf — simple, alternate, ROUND / kidney-shaped in outline,
 *     4–11 cm wide, palmately veined with 5–7 broadly ROUNDED lobes, cordate base,
 *     finely toothed, RUGOSE and "at first strongly folded": the blade PLEATS UP
 *     along the veins into a shallow parasol/funnel, held FACING THE SKY (never
 *     drooping). This palmate rounded lobed leaf is THE identifier — it dominates.
 *   • Colour — blades warm/olive green; "stems, stalks and stipules are typically
 *     REDDISH". So the petioles/stems read distinctly red-brown and the blade
 *     margins carry a reddish tinge — not pure green.
 *   • Fruit — a single raspberry-like AGGREGATE berry of a few plump drupelets
 *     that ripens red → translucent AMBER/gold; it sits among / just above the
 *     leaves, not stranded high. Flower (spring, secondary): one white 5-petal.
 * Reference: minnesotawildflowers.info/shrub/cloudberry,
 *   en.wikipedia.org/wiki/Rubus_chamaemorus, ediblewildfood.com/cloudberry.aspx,
 *   nordgen.org/.../cloudberry-rubus-chamaemorus, naturewatch.ca/plantwatch/cloudberry.
 *
 * Routes as FOLIAGE (leaf-only, single BufferGeometry). vdata.x masks the two
 * tint bands the way buildFlower does: 0 = stem / petiole / leaf (foliage tint),
 * 1 = berry drupelets (amber tint). The integrator applies a flower-style material
 * (flowerMaterial) that blends the amber tint in on vdata.x → the aggregate berry
 * reads golden against the green palmate leaves. The white spring flower is a
 * PREVIEW-ONLY feature (not returned by buildCloudberry).
 *   vdata: x = part id (0 foliage, 1 berry) · y = sway flex (0 base .. 1 tip) ·
 *          z = sway phase · w = baked AO.
 */

import { BufferAttribute, Group, Mesh, MeshStandardMaterial, DoubleSide, Vector3 } from 'three';
import type { BufferGeometry, Object3D } from 'three';
import type { Rng } from '../../core/Seed';
import { MeshGrower } from '../TubeMesh';
import { growStem, type StemSample } from './EricaceousKit';

// ---- recommended integration params ----------------------------------------
/** low leaf-dominated plant: leaves held up on short stalks reach ~10–20 cm. */
export const CLOUDBERRY_HEIGHT: readonly [number, number] = [0.1, 0.2];
/** lateral spread of the small leafy clump (m) — informational for scatter footprint. */
export const CLOUDBERRY_SPREAD = 0.16;
/** warm / olive green herb foliage carrying a slight reddish cast; hueVar lets
 *  the per-vertex vdata jitter breathe it. */
export const CLOUDBERRY_FOLIAGE = { r: 0.22, g: 0.33, b: 0.12, hueVar: 0.16 };
/** amber / golden ripe aggregate berry; frac ≈ share of tris carrying vdata.x=1. */
export const CLOUDBERRY_BERRY = { r: 0.9, g: 0.56, b: 0.14, frac: 0.14 };
export const CLOUDBERRY_CLS_MAX_DIST = 75;
/** Routes as FOLIAGE (leaf-only): one merged BufferGeometry, berry masked by vdata.x. */
export const CLOUDBERRY_ROUTING = 'FOLIAGE' as const;

const UP = new Vector3(0, 1, 0);

interface Parts {
  /** round palmate leaf blades, vdata.x = 0 (foliage tint). */
  leaf: MeshGrower;
  /** short basal stems + reddish petioles + berry stalks, vdata.x = 0. */
  stem: MeshGrower;
  /** amber aggregate berries, vdata.x = 1 (amber tint). */
  berry: MeshGrower;
  /** white 5-petal flowers — PREVIEW ONLY (not returned by buildCloudberry). */
  flower: MeshGrower;
}

/** small low-res sphere with outward normals welded into `g`. */
function sphere(g: MeshGrower, center: Vector3, radius: number, vdx: number, swayPhase: number, aoScale = 1): void {
  const stacks = 4;
  const slices = 6;
  const rows: number[][] = [];
  for (let i = 0; i <= stacks; i++) {
    const phi = (i / stacks) * Math.PI;
    const y = Math.cos(phi);
    const rr = Math.sin(phi);
    const row: number[] = [];
    const ao = (0.6 + 0.4 * (0.5 + 0.5 * y)) * aoScale;
    for (let k = 0; k <= slices; k++) {
      const th = (k / slices) * Math.PI * 2;
      const nx = rr * Math.cos(th);
      const nz = rr * Math.sin(th);
      row.push(
        g.vertex(
          center.x + nx * radius, center.y + y * radius, center.z + nz * radius,
          nx, y, nz, k / slices, i / stacks, vdx, 0.4, swayPhase, ao,
        ),
      );
    }
    rows.push(row);
  }
  for (let i = 0; i < stacks; i++) {
    const a = rows[i] as number[];
    const b = rows[i + 1] as number[];
    for (let k = 0; k < slices; k++) g.quad(a[k] as number, a[k + 1] as number, b[k + 1] as number, b[k] as number);
  }
}

/**
 * The palmate, rounded-lobed, rugose cloudberry blade — the identifier. A round /
 * reniform disc of concentric rings whose RIM undulates into `lobes` (5–7) broadly
 * rounded lobes with shallow sinuses, and a small cordate notch at the petiole
 * (the < 360° sweep leaves a basal gap). The blade is "at first strongly folded":
 * it PLEATS UP along the veins into a shallow parasol — lobe centres ride up,
 * sinuses form fold valleys, and the whole disc gently funnels UPWARD toward the
 * sky (centre lowest at the petiole). Rugose micro-relief + a finely toothed rim
 * complete the read. Normals are recomputed from the displaced grid. vdata.x = 0.
 *
 *   base   petiole attachment (blade centre / cordate base = lowest point)
 *   nrm    unit blade face normal (≈ up: the leaf faces the sky)
 *   fwd    unit in-plane forward axis (θ=0), horizontal, away from the petiole
 *   R      blade radius (m); blade width ≈ 2R
 */
function palmateLeaf(g: MeshGrower, base: Vector3, fwd: Vector3, nrm: Vector3, R: number, lobes: number, swayPhase: number, rng: Rng): void {
  const right = new Vector3().crossVectors(nrm, fwd).normalize();
  const sweep = 2.95; // radians each side of θ=0 → ~338° total → ~22° cordate notch
  const teethPerLobe = 7; // doubly-serrate marginal teeth per lobe
  const Na = lobes * teethPerLobe * 2; // 2 angular samples / tooth → crisp serration
  // denser rings toward the rim so the teeth stay sharp and thin
  const ringF = [0.0, 0.14, 0.28, 0.42, 0.55, 0.67, 0.78, 0.87, 0.94, 0.98, 1.0];
  const m = ringF.length;
  const lobeDepth = 0.26; // broadly-rounded lobes (sinus depth as fraction of R)
  const phi0 = Math.PI; // a SINUS on the midline → lobes flank the forward axis
  const funnel = R * 0.42; // overall upward cupping (parasol) toward the rim
  const pleat = R * 0.2; // lobe-centre lift over the sinus fold-valleys (rugose pleats)
  const quiltAmp = R * 0.09; // ± pucker along/between veins → the deeply rugose surface
  const toothAmp = 0.2; // marginal serration depth as fraction of local radius
  const ribCount = 5; // pinnate secondary ribs per lobe (radial cadence)
  const chevronSkew = 3.4; // herringbone slant of the pinnate ribs off the midrib
  const veinW2 = 0.09 * 0.09; // squared angular half-width of an impressed main vein

  // ---- per-angle margin profile: doubly-serrate teeth + fine noise --------
  const toothVal: number[] = []; // 0 (notch) .. 1 (tooth tip)
  const rimNoise: number[] = [];
  for (let j = 0; j <= Na; j++) {
    const tphase = (j / Na) * lobes * teethPerLobe; // teeth marching around the margin
    const prim = 1 - Math.abs(2 * (tphase - Math.floor(tphase)) - 1); // primary saw teeth
    const sec = 1 - Math.abs(2 * (2 * tphase - Math.floor(2 * tphase)) - 1); // sub-teeth
    toothVal.push(prim * 0.72 + sec * 0.28); // doubly serrate: teeth carrying sub-teeth
    rimNoise.push((rng.float() - 0.5) * 0.05);
  }

  // ---- build displaced position grid [ring][angle] ------------------------
  const pos: Vector3[][] = [];
  for (let ri = 0; ri < m; ri++) {
    const f = ringF[ri] as number;
    const ring: Vector3[] = [];
    const rimGate = Math.max(0, (f - 0.72) / 0.28); // teeth bite only near the rim
    const quiltGate = Math.min(1, f / 0.32); // venation relief fades to 0 at the petiole well
    for (let j = 0; j <= Na; j++) {
      const th = (j / Na - 0.5) * 2 * sweep; // -sweep .. +sweep
      // lobed rim: maxima (lobe tips) where cos=+1, minima (sinuses) where cos=-1
      const lobed = 1 - lobeDepth * (0.5 - 0.5 * Math.cos(lobes * th + phi0));
      // TOOTHED margin cut into the silhouette (real geometry, not a smooth edge)
      const teeth = 1 + toothAmp * ((toothVal[j] as number) - 0.5) * rimGate * rimGate;
      const margin = 1 + (rimNoise[j] as number) * f;
      const rho = R * f * lobed * margin * teeth;
      const dir = new Vector3().copy(fwd).multiplyScalar(Math.cos(th)).addScaledVector(right, Math.sin(th));

      // ---- vertical relief --------------------------------------------------
      // 1) parasol funnel up toward the rim; 2) lobe-centre pleat lift
      const cupUp = funnel * f * f;
      const lobeUp = pleat * f * (0.5 + 0.5 * Math.cos(lobes * th + phi0));
      // 3) IMPRESSED VENATION → rugose quilt: palmate main veins run along each lobe
      //    axis, pinnate ribs branch off in a herringbone; veins sink into channels,
      //    the areoles between them bulge up. veinField≈1 on a vein, ≈0 between.
      const cell = lobes * th + phi0;
      const u = (((cell / (Math.PI * 2)) % 1) + 1) % 1; // 0 = lobe axis, .5 = sinus
      const off = u < 0.5 ? u : u - 1; // signed angular distance from the lobe midrib
      const mainVein = Math.exp(-(off * off) / veinW2); // ridge/channel along the midrib
      const ribPhase = f * ribCount - Math.abs(off) * chevronSkew; // herringbone slant
      const rib = 0.5 - 0.5 * Math.cos(ribPhase * Math.PI * 2); // 0..1 pinnate rib channels
      const veinField = Math.min(1, mainVein + rib * 0.55);
      const quilt = quiltAmp * (1 - 2 * veinField) * quiltGate; // + areole, − vein channel
      const micro = (rimNoise[j] as number) * R * 0.05 * f;

      const dz = cupUp + lobeUp + quilt + micro;
      const p = new Vector3().copy(base).addScaledVector(dir, rho).addScaledVector(nrm, dz);
      ring.push(p);
    }
    pos.push(ring);
  }

  // ---- normals from the displaced grid (finite differences) ---------------
  const nrmGrid: Vector3[][] = [];
  const around = new Vector3();
  const radial = new Vector3();
  const nn = new Vector3();
  for (let ri = 0; ri < m; ri++) {
    const rn: Vector3[] = [];
    for (let j = 0; j <= Na; j++) {
      const jp = Math.min(Na, j + 1);
      const jm = Math.max(0, j - 1);
      const rp = Math.min(m - 1, ri + 1);
      const rm = Math.max(0, ri - 1);
      around.subVectors((pos[ri] as Vector3[])[jp] as Vector3, (pos[ri] as Vector3[])[jm] as Vector3);
      radial.subVectors((pos[rp] as Vector3[])[j] as Vector3, (pos[rm] as Vector3[])[j] as Vector3);
      nn.crossVectors(radial, around);
      if (nn.lengthSq() < 1e-12) nn.copy(nrm);
      nn.normalize();
      if (nn.dot(nrm) < 0) nn.negate(); // orient to the leaf's up face
      rn.push(nn.clone());
    }
    nrmGrid.push(rn);
  }

  // ---- emit vertices + triangles ------------------------------------------
  const ids: number[][] = [];
  for (let ri = 0; ri < m; ri++) {
    const f = ringF[ri] as number;
    const row: number[] = [];
    for (let j = 0; j <= Na; j++) {
      const p = (pos[ri] as Vector3[])[j] as Vector3;
      const nv = (nrmGrid[ri] as Vector3[])[j] as Vector3;
      const ao = 0.5 + 0.45 * f; // centre shaded (petiole well), rim open to sky
      // uv.v carries the centre→rim fraction (drives the preview's reddish-margin tint)
      row.push(g.vertex(p.x, p.y, p.z, nv.x, nv.y, nv.z, j / Na, f, 0, 0.2 + 0.6 * f, swayPhase, ao));
    }
    ids.push(row);
  }
  for (let ri = 0; ri < m - 1; ri++) {
    const a = ids[ri] as number[];
    const b = ids[ri + 1] as number[];
    for (let j = 0; j < Na; j++) {
      g.quad(a[j] as number, a[j + 1] as number, b[j + 1] as number, b[j] as number);
    }
  }
}

/** flat white 5-petal flower (preview only) at `center`, facing `up`. */
function whiteFlower(g: MeshGrower, center: Vector3, up: Vector3, size: number, swayPhase: number): void {
  const ref = Math.abs(up.y) < 0.94 ? UP : new Vector3(1, 0, 0);
  const u = new Vector3().crossVectors(ref, up).normalize();
  const v = new Vector3().crossVectors(up, u).normalize();
  const petals = 5;
  const dir = new Vector3();
  const perp = new Vector3();
  const tip = new Vector3();
  const s0 = new Vector3();
  const s1 = new Vector3();
  for (let i = 0; i < petals; i++) {
    const a = (i / petals) * Math.PI * 2;
    dir.copy(u).multiplyScalar(Math.cos(a)).addScaledVector(v, Math.sin(a));
    perp.copy(u).multiplyScalar(-Math.sin(a)).addScaledVector(v, Math.cos(a));
    const baseR = size * 0.18;
    const w = size * 0.44;
    tip.copy(center).addScaledVector(dir, size).addScaledVector(up, size * 0.05);
    s0.copy(center).addScaledVector(dir, baseR).addScaledVector(perp, -w);
    s1.copy(center).addScaledVector(dir, baseR).addScaledVector(perp, w);
    const c = g.vertex(center.x, center.y, center.z, up.x, up.y, up.z, 0.5, 0, 0.5, 0.35, swayPhase, 0.7);
    const p0 = g.vertex(s0.x, s0.y, s0.z, up.x, up.y, up.z, 0, 0.5, 1, 0.5, swayPhase, 0.9);
    const p1 = g.vertex(s1.x, s1.y, s1.z, up.x, up.y, up.z, 1, 0.5, 1, 0.5, swayPhase, 0.9);
    const pt = g.vertex(tip.x, tip.y, tip.z, up.x, up.y, up.z, 0.5, 1, 1, 0.6, swayPhase, 1);
    g.tri(c, p0, pt);
    g.tri(c, pt, p1);
  }
}

/** amber raspberry-like aggregate berry: a cluster of a few plump drupelets on a dome. */
function aggregateBerry(g: MeshGrower, center: Vector3, R: number, swayPhase: number, rng: Rng): void {
  const drupR = R * 0.46;
  const n = 7 + rng.int(4); // 7–10 plump drupelets
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) / n;
    const y = t; // 0 (equator) .. 1 (top) — upper hemisphere only
    const rr = Math.sqrt(Math.max(0, 1 - y * y));
    const a = i * 2.399963; // golden-angle spiral
    const c = new Vector3(
      center.x + Math.cos(a) * rr * (R - drupR * 0.5),
      center.y + y * (R - drupR * 0.4),
      center.z + Math.sin(a) * rr * (R - drupR * 0.5),
    );
    sphere(g, c, drupR * (0.85 + rng.float() * 0.3), 1, swayPhase, 1);
  }
}

/** one LOW shoot: a very short reddish basal stem, 2–3 round palmate leaves held
 *  UP on ascending reddish petioles, and (optionally) a single amber berry sitting
 *  among / just above the leaf canopy — or, preview-only, a white flower. */
function buildShoot(parts: Parts, origin: Vector3, rng: Rng, opts: { berry: boolean; flower: boolean }): void {
  const swayPhase = rng.float() * Math.PI * 2;
  // SHORT basal stem (mostly hidden among the leaves) — the plant is leaf-forward,
  // not a tall bare stalk. 2–4.5 cm.
  const stemH = 0.02 + rng.float() * 0.025;
  const stem = growStem(
    parts.stem,
    {
      origin: origin.clone(),
      azimuth: rng.float() * Math.PI * 2,
      lean: 0.05 + rng.float() * 0.06,
      height: stemH,
      baseR: 0.0016,
      tipR: 0.0011,
      segs: 4,
      wander: 0.08,
      ascend: 0.6,
      recurve: 0.0,
      swayPhase,
      hue: 0,
    },
    rng.fork('stem'),
  );

  // 2–3 leaves, each on its own ascending petiole so the big round blade is held
  // UP to the sky. Petioles fan out around the shoot; leaves are the dominant mass.
  const nLeaves = 2 + rng.int(2);
  const leafTops: Vector3[] = [];
  const az0 = rng.float() * Math.PI * 2;
  for (let i = 0; i < nLeaves; i++) {
    // attach up the short stem (alternate), fanned in azimuth
    const s = stem[Math.min(stem.length - 1, 1 + Math.floor((i / nLeaves) * (stem.length - 1)))] as StemSample;
    const az = az0 + (i / nLeaves) * Math.PI * 2 + (rng.float() - 0.5) * 0.7;
    const outward = new Vector3(Math.cos(az), 0, Math.sin(az));
    // petiole: ascends STEEPLY (mostly up, slight outward) so the blade rides high
    // and faces the sky — 3–5 cm long.
    const petH = 0.045 + rng.float() * 0.03;
    const pet = growStem(
      parts.stem,
      {
        origin: s.p.clone(),
        azimuth: az,
        lean: 0.28 + rng.float() * 0.12,
        height: petH,
        baseR: 0.001,
        tipR: 0.0007,
        segs: 4,
        wander: 0.05,
        ascend: 0.5,
        recurve: 0.0,
        swayPhase,
        hue: 0,
      },
      rng.fork(`pet${i}`),
    );
    const tip = pet[pet.length - 1] as StemSample;
    leafTops.push(tip.p.clone());
    // blade held facing the sky: normal ≈ up (tiny outward tilt so leaves don't
    // stack perfectly flat); forward axis horizontal, away from the petiole.
    const nrmL = new Vector3().copy(UP).multiplyScalar(0.98).addScaledVector(outward, 0.05).normalize();
    let fwd = new Vector3().copy(outward).addScaledVector(nrmL, -outward.dot(nrmL));
    if (fwd.lengthSq() < 1e-8) fwd = new Vector3(1, 0, 0);
    fwd.normalize();
    const R = 0.026 + rng.float() * 0.016; // blade radius 2.6–4.2 cm → 5–8 cm wide (dominant)
    const lobes = 5 + rng.int(3); // 5–7 rounded lobes
    palmateLeaf(parts.leaf, tip.p.clone(), fwd, nrmL, R, lobes, swayPhase, rng.fork(`leaf${i}`));
  }

  // berry / flower sits among / just above the leaf canopy on a short reddish stalk.
  if (opts.berry || opts.flower) {
    // canopy top = mean leaf-top height
    let cy = 0;
    for (const p of leafTops) cy += p.y;
    cy = leafTops.length ? cy / leafTops.length : stemH;
    const top = stem[stem.length - 1] as StemSample;
    const stalkBase = top.p.clone();
    // short stalk up to just above the canopy — a touch more clearance so the
    // amber berry always peeks clear of the up-cupped leaf rims from the side.
    const stalkTopY = cy + 0.02 + rng.float() * 0.015;
    const stalkTop = new Vector3(stalkBase.x + (rng.float() - 0.5) * 0.01, stalkTopY, stalkBase.z + (rng.float() - 0.5) * 0.01);
    thinStalk(parts.stem, stalkBase, stalkTop, 0.0009, swayPhase);
    if (opts.berry) {
      aggregateBerry(parts.berry, stalkTop.clone().addScaledVector(UP, 0.006), 0.007 + rng.float() * 0.003, swayPhase, rng.fork('berry'));
    } else {
      whiteFlower(parts.flower, stalkTop, UP, 0.012 + rng.float() * 0.004, swayPhase);
    }
  }
}

/** a short thin 4-sided reddish stalk from a→b (berry/flower pedicel), vdata.x = 0. */
function thinStalk(g: MeshGrower, a: Vector3, b: Vector3, r: number, swayPhase: number): void {
  const dir = new Vector3().subVectors(b, a);
  const len = dir.length() || 1e-4;
  dir.multiplyScalar(1 / len);
  const ref = Math.abs(dir.y) < 0.94 ? UP : new Vector3(1, 0, 0);
  const u = new Vector3().crossVectors(ref, dir).normalize();
  const v = new Vector3().crossVectors(dir, u).normalize();
  const sides = 4;
  const ringA: number[] = [];
  const ringB: number[] = [];
  for (let k = 0; k <= sides; k++) {
    const th = (k / sides) * Math.PI * 2;
    const rad = new Vector3().copy(u).multiplyScalar(Math.cos(th)).addScaledVector(v, Math.sin(th));
    ringA.push(g.vertex(a.x + rad.x * r, a.y + rad.y * r, a.z + rad.z * r, rad.x, rad.y, rad.z, k / sides, 0, 0, 0.2, swayPhase, 0.7));
    ringB.push(g.vertex(b.x + rad.x * r * 0.7, b.y + rad.y * r * 0.7, b.z + rad.z * r * 0.7, rad.x, rad.y, rad.z, k / sides, 1, 0, 0.8, swayPhase, 0.9));
  }
  for (let k = 0; k < sides; k++) {
    g.quad(ringA[k] as number, ringA[k + 1] as number, ringB[k + 1] as number, ringB[k] as number);
  }
}

export function buildCloudberryParts(rng: Rng): Parts {
  const parts: Parts = { leaf: new MeshGrower(), stem: new MeshGrower(), berry: new MeshGrower(), flower: new MeshGrower() };
  const shoots = 3 + rng.int(4); // small clump of 3–6 shoots reads as a low leafy patch
  for (let i = 0; i < shoots; i++) {
    const az = (i / shoots) * Math.PI * 2 + rng.float() * 0.9;
    const rad = rng.float() * 0.05; // shoots scattered within ~5 cm
    const origin = new Vector3(Math.cos(az) * rad, 0, Math.sin(az) * rad);
    // most shoots purely leafy; some carry the terminal amber berry, one may flower
    const berry = rng.chance(0.45);
    const flower = !berry && i === 0 && rng.chance(0.5);
    buildShoot(parts, origin, rng.fork(`shoot${i}`), { berry, flower });
  }
  return parts;
}

/** weld a built geometry into `g` preserving all four vertex streams + index. */
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
      dat ? dat.getX(i) : 0, dat ? dat.getY(i) : 0, dat ? dat.getZ(i) : 0, dat ? dat.getW(i) : 1,
    );
  }
  if (idx) {
    for (let i = 0; i < idx.count; i += 3) g.tri(base + idx.getX(i), base + idx.getX(i + 1), base + idx.getX(i + 2));
  } else {
    for (let i = 0; i < pos.count; i += 3) g.tri(base + i, base + i + 1, base + i + 2);
  }
}

/** Integration builder: ONE merged leaf-only BufferGeometry (stems + petioles +
 *  leaves carry vdata.x = 0 → foliage tint; amber berries carry vdata.x = 1 →
 *  amber tint under a flower-style material). White spring flowers are NOT included
 *  (preview-only). */
export function buildCloudberry(rng: Rng): { geo: BufferGeometry; tris: number; berryTris: number } {
  const parts = buildCloudberryParts(rng);
  const g = new MeshGrower();
  weld(g, parts.stem.build());
  weld(g, parts.leaf.build());
  weld(g, parts.berry.build());
  return { geo: g.build(), tris: g.triCount, berryTris: parts.berry.triCount };
}

/** per-vertex colours for the PREVIEW leaf material: warm green at the centre
 *  reddening toward the finely-toothed rim (uv.v = centre→rim), so the reviewer's
 *  "slight reddish tinge" is visible. (Shipped geometry uses the single foliage
 *  tint band; this is preview dressing only.) */
function leafColors(geo: BufferGeometry): BufferGeometry {
  const uv = geo.getAttribute('uv');
  const n = uv.count;
  const col = new Float32Array(n * 3);
  const cen = [0.19, 0.33, 0.12]; // medium warm-green centre
  const rim = [0.34, 0.26, 0.1]; // reddish-olive margin (the "slight reddish tinge")
  for (let i = 0; i < n; i++) {
    const f = uv.getY(i); // 0 centre .. 1 rim
    const t = f * f; // push the red out toward the very margin
    col[i * 3] = cen[0] as number + ((rim[0] as number) - (cen[0] as number)) * t;
    col[i * 3 + 1] = cen[1] as number + ((rim[1] as number) - (cen[1] as number)) * t;
    col[i * 3 + 2] = cen[2] as number + ((rim[2] as number) - (cen[2] as number)) * t;
  }
  geo.setAttribute('color', new BufferAttribute(col, 3));
  return geo;
}

/** Fully-assembled, materialed plant for the QA harness. Meters, +Y up, base at
 *  y=0. Round green palmate leaves (reddish-margin vertex tint) + reddish stems/
 *  petioles + amber berries; the (spring) white flower in a preview-only material. */
export function buildPreview(rng: Rng): Object3D {
  const parts = buildCloudberryParts(rng);
  const g = new Group();
  const leafMat = new MeshStandardMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.68, metalness: 0, side: DoubleSide });
  const stemMat = new MeshStandardMaterial({ color: 0x7c4326, roughness: 0.7, metalness: 0, side: DoubleSide });
  const berryMat = new MeshStandardMaterial({ color: 0xe0921f, roughness: 0.42, metalness: 0, side: DoubleSide, emissive: 0x3a2404, emissiveIntensity: 0.28 });
  const flowerMat = new MeshStandardMaterial({ color: 0xf4f2ec, roughness: 0.8, metalness: 0, side: DoubleSide, emissive: 0x2a2a28, emissiveIntensity: 0.25 });
  g.add(new Mesh(leafColors(parts.leaf.build()), leafMat));
  g.add(new Mesh(parts.stem.build(), stemMat));
  g.add(new Mesh(parts.berry.build(), berryMat));
  g.add(new Mesh(parts.flower.build(), flowerMat));
  return g;
}
