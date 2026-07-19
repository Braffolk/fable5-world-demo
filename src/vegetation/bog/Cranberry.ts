/**
 * Small cranberry — Vaccinium oxycoccos. A creeping evergreen dwarf shrub of the
 * raised bog. REBUILD (v2): the first attempt read as a "dead spider" — a flat
 * star of long BARE stems splayed on bare ground, with lone berries scattered far
 * apart. That is not what a person sees on the moss. What you actually see is a
 * COMPACT LEAFY LITTLE PLANT: short erect/ascending leafy shoots rising a few cm
 * above the sphagnum, densely clothed in tiny leathery leaves, with a couple of
 * small CLUSTERS of a few deep-red berries nodding on thin stalks near the shoot
 * tips. The horizontal creeping runners are kept low and are themselves leafy —
 * they give the "creeps through the moss" read without looking like bare wire.
 *
 * Diagnostic reads (grounded in real reference):
 *   • Habit — "new branches erect to ascending, 4–5 in, rising above the moss;
 *     older branches dark reddish-brown, spreading horizontally, rooting at the
 *     nodes" (Minnesota Wildflowers). So the VISIBLE plant is leafy ascending
 *     shoots, not a bare radial fan. Compact, not a 30-cm splay.
 *   • Leaves — simple, ALTERNATE, ovate/elliptic 3–10 mm, leathery (coriaceous),
 *     dark green & shining above / glaucous-whitish beneath, with margins STRONGLY
 *     revolute (rolled under) → they read narrow with acute tips (Go Botany; FNA;
 *     Minnesota Wildflowers). Dense along the shoots — full coverage, not sparse.
 *   • Berries — "1 to 4 nodding on long stalks" near the shoot tips (Minnesota
 *     Wildflowers), i.e. small GROUPS, not scattered singletons; round, shiny
 *     deep red, 6–12 mm, borne a couple cm above the foliage (FNA; USFS FEIS).
 * Reference: minnesotawildflowers.info/shrub/small-cranberry,
 *   gobotany.nativeplanttrust.org/species/vaccinium/oxycoccos,
 *   floranorthamerica.org/Vaccinium_oxycoccos, en.wikipedia.org/wiki/
 *   Vaccinium_oxycoccos, research.fs.usda.gov/feis/species-reviews/vacoxy.
 *
 * Routes as FOLIAGE (leaf-only, single BufferGeometry). vdata.x masks the two
 * tint bands the way buildFlower does: 0 = stem/leaf (foliage tint), 1 = berry
 * (deep-red berry tint). The integrator applies a flower-style material that
 * blends the berry tint in on vdata.x → berries read red against green foliage.
 *   vdata: x = part id (0 foliage, 1 berry) · y = sway flex (0 base .. 1 tip) ·
 *          z = sway phase · w = baked AO.
 */

import { Group, Mesh, MeshStandardMaterial, DoubleSide, Vector3 } from 'three';
import type { BufferGeometry, Object3D } from 'three';
import type { Rng } from '../../core/Seed';
import { MeshGrower } from '../TubeMesh';
import { growStem, walkStem, leafBlade, perpFrame, stemFlexAt, STEM_FLEX_TIP, type StemSample } from './EricaceousKit';

// ---- recommended integration params ----------------------------------------
/** compact leafy plant height (m): ascending shoots + berry stalks reach ~6–10 cm. */
export const CRANBERRY_HEIGHT: readonly [number, number] = [0.06, 0.1];
/** lateral spread of the leafy clump (m) — informational for scatter footprint. */
export const CRANBERRY_SPREAD = 0.16;
/** dark evergreen foliage; hueVar lets per-vertex vdata.x jitter breathe it. */
export const CRANBERRY_FOLIAGE = { r: 0.09, g: 0.19, b: 0.07, hueVar: 0.18 };
/** deep shiny ripe berry; frac ≈ share of tris carrying vdata.x=1 (berry band). */
export const CRANBERRY_BERRY = { r: 0.52, g: 0.05, b: 0.06, frac: 0.24 };
export const CRANBERRY_CLS_MAX_DIST = 70;
/** Routes as FOLIAGE (leaf-only): one merged BufferGeometry, berries masked by vdata.x. */
export const CRANBERRY_ROUTING = 'FOLIAGE' as const;

interface Parts {
  /** stems + leaves + berry stalks, vdata.x = 0 (foliage tint). */
  foliage: MeshGrower;
  /** nodding round berries, vdata.x = 1 (deep-red berry tint). */
  berry: MeshGrower;
}

const UP = new Vector3(0, 1, 0);
const GOLDEN = 2.399963; // golden-angle phyllotaxy for alternate leaves

/** A small round berry (low-res lat/long sphere) with outward normals, welded
 *  into the berry grower. vdata.x = 1 so the flower-style material tints it red. */
function berrySphere(g: MeshGrower, center: Vector3, radius: number, swayPhase: number, attachFlex: number): void {
  const stacks = 5;
  const slices = 7;
  const rows: number[][] = [];
  for (let i = 0; i <= stacks; i++) {
    const phi = (i / stacks) * Math.PI; // 0..π (pole to pole)
    const y = Math.cos(phi);
    const rr = Math.sin(phi);
    const row: number[] = [];
    // berries darken slightly toward the shaded underside (lower AO for -y)
    const ao = 0.6 + 0.4 * (0.5 + 0.5 * y);
    for (let k = 0; k <= slices; k++) {
      const th = (k / slices) * Math.PI * 2;
      const nx = rr * Math.cos(th);
      const ny = y;
      const nz = rr * Math.sin(th);
      row.push(
        g.vertex(
          center.x + nx * radius, center.y + ny * radius, center.z + nz * radius,
          nx, ny, nz,
          k / slices, i / stacks,
          1, attachFlex, swayPhase, ao,
        ),
      );
    }
    rows.push(row);
  }
  for (let i = 0; i < stacks; i++) {
    const a = rows[i] as number[];
    const b = rows[i + 1] as number[];
    for (let k = 0; k < slices; k++) {
      g.quad(a[k] as number, a[k + 1] as number, b[k + 1] as number, b[k] as number);
    }
  }
}

/**
 * Clothe a shoot centerline with tiny leathery ovate leaves, alternate (golden
 * angle) and DENSE (spacing `spacing` m) so the shoot reads as a leafy shoot, not
 * a bare stem. `spread` blends the leaf axis between "held out from the stem" and
 * "up" — ascending shoots hold leaves out+up; low runners hold them nearly flat.
 */
function dressStem(parts: Parts, samples: StemSample[], spacing: number, spread: number, tStart: number, rng: Rng): void {
  const u = new Vector3();
  const v = new Vector3();
  let node = 0;
  walkStem(samples, spacing, tStart, (p, dir, t) => {
    perpFrame(dir, u, v);
    const a = node * GOLDEN + rng.float() * 0.35;
    node++;
    const outN = new Vector3().copy(u).multiplyScalar(Math.cos(a)).addScaledVector(v, Math.sin(a));
    // leaf axis: mostly outward from the stem, lifted toward UP, a touch forward
    const axis = new Vector3()
      .copy(outN).multiplyScalar(0.82)
      .addScaledVector(UP, spread)
      .addScaledVector(dir, 0.16)
      .normalize();
    const side = new Vector3().crossVectors(axis, UP);
    if (side.lengthSq() < 1e-6) side.crossVectors(axis, u);
    side.normalize();
    const len = 0.005 + rng.float() * 0.0033; // 5–8.3 mm (comparable to a berry)
    const hue = (rng.float() - 0.5) * 0.55;
    // narrow-ovate & leathery: broadest below mid, acute tip, margins STRONGLY
    // rolled under (revolute) → reads narrow/needle-like like the real leaf
    leafBlade(
      parts.foliage, p, axis, side,
      len, len * 0.24, len * 0.34, len * 0.05,
      0.16, 0.6, hue, stemFlexAt(t), 0.5 + 0.4 * t, 0.98, 2,
    );
  });
}

/** Sample a shoot centerline at fraction `frac` of its node count (≥1, ≤tip). */
function sampleAt(samples: StemSample[], frac: number): StemSample {
  const idx = Math.min(samples.length - 1, Math.max(1, Math.round(frac * (samples.length - 1))));
  return samples[idx] as StemSample;
}

/**
 * A small CLUSTER of 2–3 berries borne in the leafy tip of a shoot. Each berry
 * hangs at the end of a SHORT (6–13 mm) arching pedicel that springs from an
 * upper node, curves over and nods the berry down among the terminal leaves — so
 * the berry reads as attached to the plant, not a floating lollipop on a stiff
 * bare stalk. Berry ~5–7 mm — comparable to a leaf. The pedicels spring from the
 * same top-of-shoot zone so the berries group naturally.
 */
function addBerryCluster(parts: Parts, samples: StemSample[], baseAz: number, rng: Rng): void {
  const n = 2 + rng.int(2); // 2–3 berries per cluster
  for (let b = 0; b < n; b++) {
    const seat = sampleAt(samples, 0.72 + rng.float() * 0.22); // upper part of the shoot
    const phase = rng.float() * Math.PI * 2;
    const az = baseAz + (rng.float() - 0.5) * 2.4;
    const stalkH = 0.006 + rng.float() * 0.007; // 6–13 mm short pedicel
    const stalk = growStem(
      parts.foliage,
      {
        origin: seat.p.clone(),
        azimuth: az,
        lean: 0.45, // spring up-and-out from the node
        height: stalkH,
        baseR: 0.00035,
        tipR: 0.00028,
        segs: 4,
        wander: 0.14,
        ascend: 0.1,
        recurve: 1.0, // strong nod → the pedicel arches over, berry hangs close
        swayPhase: phase,
        hue: 0,
      },
      rng.fork(`ped${b}`),
    );
    const tip = stalk[stalk.length - 1] as StemSample;
    const rBerry = 0.0026 + rng.float() * 0.001; // 5.2–7.2 mm diameter (≈ leaf size)
    // berry sits AT the pedicel tip along its (nodding) direction → visibly joined.
    // RIGID: one CONSTANT flex = the pedicel-tip flex (growStem tip), so it sways
    // as one unit with the stalk instead of shearing.
    const c = new Vector3().copy(tip.p).addScaledVector(tip.dir, rBerry * 0.85);
    berrySphere(parts.berry, c, rBerry, phase, STEM_FLEX_TIP);
  }
}

export function buildCranberryParts(rng: Rng): Parts {
  const foliage = new MeshGrower();
  const berry = new MeshGrower();
  const parts: Parts = { foliage, berry };

  // ── 1) low creeping runners: a short LEAFY tangle hugging the moss. Kept low
  //    and clothed with leaves so they read as creeping shoots, not bare wire. ──
  const nRunners = 2 + rng.int(2); // 2–3
  const runnerTips: { p: Vector3; az: number }[] = [];
  for (let i = 0; i < nRunners; i++) {
    const az = (i / nRunners) * Math.PI * 2 + rng.float() * 1.2;
    const len = 0.04 + rng.float() * 0.025; // 4–6.5 cm trailing (kept short & leafy)
    const samples = growStem(
      foliage,
      {
        origin: new Vector3(Math.cos(az) * 0.012 * rng.float(), 0.006 + rng.float() * 0.004, Math.sin(az) * 0.012 * rng.float()),
        azimuth: az,
        lean: 0.9, // nearly horizontal creeper
        height: len,
        baseR: 0.0006,
        tipR: 0.0004,
        segs: 6,
        wander: 0.18,
        ascend: 0.12, // hug the moss but lift the tip a touch (not a dead flat wire)
        recurve: 0.0,
        swayPhase: rng.float() * Math.PI * 2,
        hue: 0,
      },
      rng.fork(`runner${i}`),
    );
    dressStem(parts, samples, 0.0045, 0.3, 0.05, rng.fork(`rleaf${i}`));
    const tip = samples[samples.length - 1] as StemSample;
    runnerTips.push({ p: tip.p.clone(), az: az + (rng.float() - 0.5) });
  }

  // ── 2) erect/ascending leafy shoots — the recognisable, dominant foliage.
  //    Rise from near the crown AND from runner tips (nodes root & turn up). ──
  const nShoots = 7 + rng.int(4); // 7–10
  for (let i = 0; i < nShoots; i++) {
    const fromRunner = i < runnerTips.length && rng.chance(0.7);
    const seat = fromRunner ? (runnerTips[i] as { p: Vector3; az: number }) : null;
    const baseAz = seat ? seat.az : rng.float() * Math.PI * 2;
    const origin = seat
      ? new Vector3(seat.p.x, Math.max(0.005, seat.p.y), seat.p.z)
      : new Vector3(Math.cos(baseAz) * 0.03 * rng.float(), 0.006 + rng.float() * 0.006, Math.sin(baseAz) * 0.03 * rng.float());
    const h = 0.045 + rng.float() * 0.035; // 4.5–8 cm ascending shoot
    const samples = growStem(
      foliage,
      {
        origin,
        azimuth: baseAz,
        lean: 0.22 + rng.float() * 0.22, // mostly upright, a little outward tilt
        height: h,
        baseR: 0.0005,
        tipR: 0.0003,
        segs: 6,
        wander: 0.13,
        ascend: 0.55, // pull upright → erect leafy shoot
        recurve: 0.12,
        swayPhase: rng.float() * Math.PI * 2,
        hue: 0,
      },
      rng.fork(`shoot${i}`),
    );
    // dense leaves, held out+up, over most of the shoot
    dressStem(parts, samples, 0.003, 0.42, 0.06, rng.fork(`sleaf${i}`));

    // ── berries: distributed across ~60% of the shoots (not just 1–2), each a
    //    small cluster nestled in that shoot's leafy tip. ──
    if (rng.chance(0.6)) {
      addBerryCluster(parts, samples, baseAz, rng.fork(`berry${i}`));
    }
  }

  return parts;
}

/** Integration builder: ONE merged leaf-only BufferGeometry (berries masked by
 *  vdata.x = 1). Consumed as a FOLIAGE leaf pool. */
export function buildCranberry(rng: Rng): { geo: BufferGeometry; tris: number; berryTris: number } {
  const parts = buildCranberryParts(rng);
  const g = new MeshGrower();
  weld(g, parts.foliage.build());
  weld(g, parts.berry.build());
  return { geo: g.build(), tris: parts.foliage.triCount + parts.berry.triCount, berryTris: parts.berry.triCount };
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

/** Fully-assembled, materialed plant for the QA harness. Meters, +Y up, base at
 *  y=0. Dark-green leafy clump in one material; deep-red berries in a second. */
export function buildPreview(rng: Rng): Object3D {
  const parts = buildCranberryParts(rng);
  const g = new Group();
  const foliageMat = new MeshStandardMaterial({ color: 0x2f4d1a, roughness: 0.58, metalness: 0, side: DoubleSide });
  const berryMat = new MeshStandardMaterial({ color: 0x861013, roughness: 0.34, metalness: 0, side: DoubleSide, emissive: 0x230304, emissiveIntensity: 0.22 });
  g.add(new Mesh(parts.foliage.build(), foliageMat));
  g.add(new Mesh(parts.berry.build(), berryMat));
  return g;
}
