/**
 * ?scene=gallery — specimen gallery (spec §4): every species × 3 seeds on
 * labeled pedestals, rock wall, dressed cliff, debris ground square. Primary
 * review surface for the Phase-4 macro–meso–micro audit. Full lighting/post
 * pipeline (sun/sky, CSM+PCSS, GTAO, TRAA, grade) so review = world shading.
 *
 * ?row=trees|rocks|ground|dead frames the camera on one exhibit row.
 */

import {
  CanvasTexture,
  CircleGeometry,
  CylinderGeometry,
  InstancedMesh,
  Mesh,
  PlaneGeometry,
  SRGBColorSpace,
  Vector3,
} from 'three';
import { MeshStandardNodeMaterial } from 'three/webgpu';
import { float, mix, positionWorld, smoothstep, vec3 } from 'three/tsl';
import type { NF } from '../gpu/TSLTypes';
import { hash12 } from '../gpu/noise/NoiseTSL';
import { bakeBarkTextures, type BarkTextures } from '../gpu/passes/BarkSynth';
import { PostStack } from '../render/PostStack';
import {
  barkTexturedMaterial,
  deadwoodMaterial,
  flowerMaterial,
  foliageMaterial,
  mushroomMaterial,
  rockMaterial,
  updateSunUniforms,
} from '../render/VegMaterials';
import { SunSky } from '../sky/SunSky';
import { buildLog, buildStump, type DecayState } from '../vegetation/Deadfall';
import { buildMushroom, buildVines } from '../vegetation/Dressing';
import {
  barkChipGeometry,
  debrisMaterial,
  grassPatch,
  scatterInstances,
  twigGeometry,
} from '../vegetation/GroundCover';
import { buildRock, type RockPreset } from '../vegetation/RockBuilder';
import { TREE_SPECIES } from '../vegetation/Species';
import { buildTree } from '../vegetation/TreeBuilder';
import {
  buildFlower,
  buildShrub,
  type FlowerKind,
  UNDERSTORY_SPECIES,
} from '../vegetation/Understory';
import type { WorldContext } from './Scenes';

const ROW_Z = { hero: -26, trees: 0, rocks: 40, ground: 70, dead: 100 } as const;

function labelSprite(text: string, sub: string): Mesh {
  const cv = document.createElement('canvas');
  cv.width = 512;
  cv.height = 144;
  const c = cv.getContext('2d');
  if (c) {
    c.fillStyle = 'rgba(20,24,28,0.92)';
    c.fillRect(0, 0, 512, 144);
    c.fillStyle = '#e8eef2';
    c.font = '600 44px system-ui, sans-serif';
    c.fillText(text, 18, 58);
    c.fillStyle = '#9fb2bf';
    c.font = '400 32px system-ui, sans-serif';
    c.fillText(sub, 18, 110);
  }
  const tex = new CanvasTexture(cv);
  tex.colorSpace = SRGBColorSpace;
  const mat = new MeshStandardNodeMaterial();
  mat.map = tex;
  mat.roughness = 0.9;
  const m = new Mesh(new PlaneGeometry(2.6, 0.73), mat);
  return m;
}

export async function buildGalleryScene(ctx: WorldContext): Promise<void> {
  const { engine, params, seed } = ctx;
  const q = new URLSearchParams(window.location.search);
  // ?noleaves — DEBUG: bark-only (skip foliage) so trunk/branch junctions are visible
  const noLeaves = q.get('noleaves') !== null;
  // ?nojunctions — A/B ablation: legacy independent open-tube bark in the gallery too
  const junctionsOn = q.get('nojunctions') === null;

  ctx.progress(0.05, 'gallery: sky');
  const sunSky = new SunSky(engine, params.timeOfDay);
  await sunSky.init(engine.renderer);
  updateSunUniforms(sunSky.sun);

  // ---- ground: neutral matte with a faint 5 m scale grid ---------------------
  const groundMat = new MeshStandardNodeMaterial();
  {
    const wxz = positionWorld.xz;
    const n = hash12(wxz.mul(0.71).floor()) as NF;
    const base = mix(
      vec3(0.085, 0.1, 0.06),
      vec3(0.12, 0.125, 0.085),
      n.mul(0.7).add(hash12(wxz.mul(0.093).floor()).mul(0.3)),
    );
    const gx = smoothstep(0.0, 0.06, wxz.x.div(5).fract().sub(0.5).abs());
    const gz = smoothstep(0.0, 0.06, wxz.y.div(5).fract().sub(0.5).abs());
    groundMat.colorNode = base.mul(gx.min(gz).mul(0.12).add(0.88));
    groundMat.roughness = 0.96;
  }
  const ground = new Mesh(new CircleGeometry(420, 64), groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  engine.scene.add(ground);

  // ---- pedestal helper -------------------------------------------------------
  const pedestalMat = new MeshStandardNodeMaterial();
  pedestalMat.colorNode = vec3(0.32, 0.31, 0.3).mul(
    hash12(positionWorld.xz.mul(31)).mul(0.15).add(float(0.85)),
  );
  pedestalMat.roughness = 0.88;
  const pedestalGeo = new CylinderGeometry(2.0, 2.3, 0.42, 28);

  const exhibit = (
    x: number,
    z: number,
    title: string,
    sub: string,
    opts?: { pedestal?: boolean },
  ): { x: number; z: number } => {
    if (opts?.pedestal !== false) {
      const ped = new Mesh(pedestalGeo, pedestalMat);
      ped.position.set(x, 0.21, z);
      ped.receiveShadow = true;
      ped.castShadow = true;
      engine.scene.add(ped);
      const label = labelSprite(title, sub);
      label.position.set(x, 0.62, z + 2.45);
      label.rotation.x = -0.42;
      engine.scene.add(label);
    } else {
      // floating label behind the exhibit (never occludes it)
      const label = labelSprite(title, sub);
      label.position.set(x, 2.3, z - 4.6);
      engine.scene.add(label);
    }
    return { x, z };
  };

  // ---- bark textures (synthesized per species layer) -------------------------
  ctx.progress(0.09, 'gallery: synthesizing bark');
  const barks = new Map<number, BarkTextures>();
  for (const sp of TREE_SPECIES) {
    if (barks.has(sp.barkLayer)) continue;
    barks.set(
      sp.barkLayer,
      await bakeBarkTextures(engine.renderer, sp.barkLayer, seed.sub(`bark/${sp.barkLayer}`) % 977),
    );
  }

  // ---- tree row: 6 species × 3 seeds ------------------------------------------
  let totalTris = 0;
  const spacing = 13;
  const groupGap = 6;
  const nSpecies = TREE_SPECIES.length;
  const rowWidth = nSpecies * 3 * spacing + (nSpecies - 1) * groupGap;
  let x = -rowWidth / 2;
  for (let si = 0; si < nSpecies; si++) {
    const sp = TREE_SPECIES[si];
    if (!sp) continue;
    for (let vi = 0; vi < 3; vi++) {
      ctx.progress(
        0.1 + (0.8 * (si * 3 + vi)) / (nSpecies * 3),
        `gallery: growing ${sp.id} #${vi}`,
      );
      // yield so boot UI can paint between heavy builds
      await new Promise((r) => setTimeout(r, 0));
      const rng = seed.rng(`tree/${sp.id}/${vi}`);
      const built = buildTree(sp, rng, { junctions: junctionsOn });
      totalTris += built.stats.tris;
      const at = exhibit(
        x,
        ROW_Z.trees,
        sp.label,
        `seed ${vi} · ${(built.stats.tris / 1000).toFixed(0)}k tris · ${built.stats.height.toFixed(1)} m`,
      );
      const barkTex = barks.get(sp.barkLayer) as BarkTextures;
      const barkMesh = new Mesh(built.bark, barkTexturedMaterial(barkTex));
      barkMesh.position.set(at.x, 0.42, at.z);
      barkMesh.castShadow = true;
      barkMesh.receiveShadow = true;
      engine.scene.add(barkMesh);
      // TODO(missing-leaves, CRITICAL): this LOD tree-review row is now BARK-ONLY.
      // It showed card foliage (deleted, S8); the live crown is the MESH leaf head
      // (see the HERO row, foliageMode 'mesh'). Give this row a budgeted mesh crown
      // to restore it as the species leaf-review surface.
      x += spacing;
    }
    x += groupGap;
  }
  engine.stats.counters['veg.tris'] = totalTris;

  // ---- rock row ----------------------------------------------------------------
  ctx.progress(0.9, 'gallery: carving rocks');
  await new Promise((r) => setTimeout(r, 0));
  let rockTris = 0;
  const addRock = (
    preset: RockPreset,
    detail: number,
    seedTag: string,
    x: number,
    z: number,
    moss: number,
    label?: string,
  ): void => {
    const rock = buildRock(preset, seed.rng(`rock/${preset}/${seedTag}`), detail);
    rockTris += rock.stats.tris;
    const m = new Mesh(rock.geometry, rockMaterial({ moss }));
    // settle into the ground a bit
    const bs = rock.geometry.boundingSphere;
    m.position.set(x, bs ? bs.radius * 0.52 : 1, z);
    m.rotation.y = seed.rng(`rockrot/${preset}/${seedTag}`).float() * Math.PI * 2;
    m.castShadow = true;
    m.receiveShadow = true;
    engine.scene.add(m);
    if (label) {
      exhibit(x, z, label, `${(rock.stats.tris / 1000).toFixed(0)}k tris`);
    }
  };
  const RZ = ROW_Z.rocks;
  addRock('hero', 7, '0', -60, RZ, 0.35, 'Hero rock (tor)');
  for (let i = 0; i < 3; i++) addRock('boulder', 5, String(i), -38 + i * 9, RZ, 0.45, i === 0 ? 'Boulders ×3' : undefined);
  for (let i = 0; i < 3; i++) addRock('angular', 5, String(i), -8 + i * 8, RZ, 0.1, i === 0 ? 'Angular scree ×3' : undefined);
  addRock('slab', 6, '0', 18, RZ, 0.2, 'Slab');
  // cobble cluster
  for (let i = 0; i < 14; i++) {
    const r = seed.rng(`cob/${i}`);
    addRock('cobble', 3, String(i), 28 + r.float() * 5 - 2.5, RZ + r.float() * 4 - 2, 0.12);
  }
  exhibit(28, RZ, 'Cobbles', 'water-rounded');
  // rock wall: stacked slabs
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 4; col++) {
      const r = seed.rng(`wall/${row}/${col}`);
      const rock = buildRock('slab', r.fork('s'), 5);
      rockTris += rock.stats.tris;
      const m = new Mesh(rock.geometry, rockMaterial({ moss: 0.3 }));
      m.position.set(44 + col * 2.6 + (row % 2) * 1.2, 0.8 + row * 1.35, RZ + (r.float() - 0.5));
      m.rotation.set((r.float() - 0.5) * 0.3, r.float() * 6.28, (r.float() - 0.5) * 0.3);
      m.scale.setScalar(0.85 + r.float() * 0.4);
      m.castShadow = true;
      m.receiveShadow = true;
      engine.scene.add(m);
    }
  }
  exhibit(48, RZ, 'Rock wall', 'stacked slabs');

  // dressed cliff: leaning slab + dirt streaks + hanging vine STEMS
  // TODO(missing-leaves, CRITICAL): vine leaves + ledge ferns were foliage cards
  // (deleted, S8) — the vines hang bare and the ferns are gone. Rebuild as mesh.
  {
    const cliffRock = buildRock('cliffFace', seed.rng('cliff/0'), 6);
    rockTris += cliffRock.stats.tris;
    const m = new Mesh(cliffRock.geometry, rockMaterial({ moss: 0.4 }));
    m.scale.set(1.5, 1.7, 1.2);
    m.position.set(72, 4.2, RZ);
    m.rotation.set(0.05, 0.2, -0.02);
    m.castShadow = true;
    m.receiveShadow = true;
    engine.scene.add(m);
    const vineRng = seed.rng('cliff/vines');
    const vines = buildVines(vineRng, 5.2, 4.2, 9);
    const stemMat = barkTexturedMaterial(barks.get(4) as BarkTextures);
    const vs = new Mesh(vines.stems, stemMat);
    vs.position.set(71.8, 7.6, RZ + 1.7);
    vs.castShadow = true;
    engine.scene.add(vs);
    exhibit(72, RZ + 5, 'Dressed cliff', 'streaks + bare vine stems', { pedestal: false });
  }
  engine.stats.counters['rock.tris'] = rockTris;

  // ---- ground row: grass, debris square, ferns, flowers, shrubs ---------------
  ctx.progress(0.93, 'gallery: ground cover');
  await new Promise((r) => setTimeout(r, 0));
  const GZ = ROW_Z.ground;
  {
    // meadow squares
    const small = grassPatch(seed.rng('grass/small'), 60_000, 6);
    small.position.set(-62, 0, GZ);
    engine.scene.add(small);
    exhibit(-62, GZ + 4, 'Grass 6×6 m', '60k blades', { pedestal: false });
    const meadow = grassPatch(seed.rng('grass/meadow'), 200_000, 13);
    meadow.position.set(-44, 0, GZ);
    engine.scene.add(meadow);
    exhibit(-44, GZ + 7.5, 'Meadow 13×13 m', '200k blades', { pedestal: false });

    // 2×2 m ground debris square
    const sq = 2.4;
    const cobbleGeo = buildRock('cobble', seed.rng('gc/cob'), 2).geometry;
    const cobInst = new InstancedMesh(cobbleGeo, rockMaterial({ moss: 0.08 }), 42);
    scatterInstances(cobInst, seed.rng('gc/cobs'), sq, 0.02, [0.35, 1.0], true);
    cobInst.position.set(-26, 0.05, GZ);
    engine.scene.add(cobInst);
    const pebInst = new InstancedMesh(
      buildRock('cobble', seed.rng('gc/peb'), 1).geometry,
      rockMaterial({ moss: 0 }),
      260,
    );
    scatterInstances(pebInst, seed.rng('gc/pebs'), sq, 0.02, [0.16, 0.5], true);
    pebInst.position.set(-26, 0.03, GZ);
    engine.scene.add(pebInst);
    for (let v = 0; v < 3; v++) {
      const tw = new InstancedMesh(
        twigGeometry(seed.rng(`gc/twig${v}`)),
        debrisMaterial('twig'),
        36,
      );
      scatterInstances(tw, seed.rng(`gc/twigs${v}`), sq, 0.03, [0.6, 1.5], true);
      tw.position.set(-26, 0.02, GZ);
      engine.scene.add(tw);
      const ch = new InstancedMesh(
        barkChipGeometry(seed.rng(`gc/chip${v}`)),
        debrisMaterial('chip'),
        45,
      );
      scatterInstances(ch, seed.rng(`gc/chips${v}`), sq, 0.02, [0.5, 1.3], true);
      ch.position.set(-26, 0.015, GZ);
      engine.scene.add(ch);
    }
    // TODO(missing-leaves): leaf litter (beech foliage-atlas quads) removed with
    // the card pipeline (S8) — the atlas capture it sampled is gone.
    exhibit(-26, GZ + 2, 'Ground square 2.4 m', 'cobbles+twigs+chips', { pedestal: false });

    // TODO(missing-leaves, CRITICAL): fern row removed — ferns were entirely
    // foliage cards (buildFern deleted, S8). Rebuild as real mesh fronds.

    // flower patches
    const flowerKinds: { kind: FlowerKind; color: { r: number; g: number; b: number }; n: number; label: string }[] = [
      { kind: 'umbel', color: { r: 0.75, g: 0.75, b: 0.7 }, n: 14, label: 'White umbel' },
      { kind: 'bell', color: { r: 0.28, g: 0.14, b: 0.5 }, n: 14, label: 'Bellflower' },
      { kind: 'daisy', color: { r: 0.85, g: 0.72, b: 0.12 }, n: 18, label: 'Yellow daisy' },
    ];
    let fx = 0;
    for (const fk of flowerKinds) {
      const rngF = seed.rng(`flower/${fk.kind}`);
      for (let i = 0; i < fk.n; i++) {
        const fl = new Mesh(buildFlower(fk.kind, rngF.fork(String(i))), flowerMaterial(fk.color));
        fl.position.set(fx + (rngF.float() - 0.5) * 2.4, 0, GZ + (rngF.float() - 0.5) * 2.4);
        fl.rotation.y = rngF.float() * 6.28;
        fl.castShadow = true;
        engine.scene.add(fl);
      }
      exhibit(fx, GZ + 2, fk.label, `×${fk.n}`, { pedestal: false });
      fx += 7;
    }

    // shrubs ×3 (incl. pink flowering)
    let sx = 26;
    for (const sp of UNDERSTORY_SPECIES) {
      const shrub = buildShrub(sp, seed.rng(`shrub/${sp.id}`));
      const bm = new Mesh(shrub.bark, barkTexturedMaterial(barks.get(sp.barkLayer) as BarkTextures));
      bm.position.set(sx, 0, GZ);
      bm.castShadow = true;
      bm.receiveShadow = true;
      engine.scene.add(bm);
      // TODO(missing-leaves, CRITICAL): shrubs are BARK-ONLY — foliage was card
      // geometry (deleted, S8). Needs a real mesh leaf crown per stem.
      exhibit(sx, GZ + 2.5, sp.label, 'multi-stem (bark only)', { pedestal: false });
      sx += 9;
    }
  }

  // ---- dead row: logs (3 decay states), stumps --------------------------------
  ctx.progress(0.95, 'gallery: deadfall');
  const DZ = ROW_Z.dead;
  const spruceBark = barks.get(0) as BarkTextures;
  const decays: DecayState[] = ['fresh', 'mossy', 'rotten'];
  for (let i = 0; i < decays.length; i++) {
    const log = buildLog(seed.rng(`log/${i}`), decays[i] as DecayState);
    const m = new Mesh(log.geometry, deadwoodMaterial(spruceBark));
    m.position.set(-22 + i * 9, 0, DZ);
    // keep logs near-perpendicular to the row so they present their length
    m.rotation.y = (seed.rng(`logr/${i}`).float() - 0.5) * 0.8;
    m.castShadow = true;
    m.receiveShadow = true;
    engine.scene.add(m);
    exhibit(-22 + i * 9, DZ + 2.5, `Log (${decays[i]})`, `${log.length.toFixed(1)} m`, { pedestal: false });
  }
  {
    const shelfRng = seed.rng('shelf');
    for (let i = 0; i < 4; i++) {
      const sh = new Mesh(buildMushroom(shelfRng.fork(String(i)), 'shelf'), mushroomMaterial());
      sh.position.set(-13.6 + i * 0.5, 0.32 + (i % 2) * 0.12, DZ + 0.28);
      sh.rotation.z = Math.PI / 2 - 0.3;
      sh.rotation.y = -Math.PI / 2;
      sh.castShadow = true;
      engine.scene.add(sh);
    }
  }
  for (let i = 0; i < 2; i++) {
    const st = buildStump(seed.rng(`stump/${i}`));
    const m = new Mesh(st.geometry, deadwoodMaterial(spruceBark));
    m.position.set(8 + i * 6, 0, DZ);
    m.castShadow = true;
    m.receiveShadow = true;
    engine.scene.add(m);
  }
  exhibit(11, DZ + 2.5, 'Stumps ×2', 'root flare, jagged top', { pedestal: false });

  // ---- hero row: mesh-foliage hero trees (>=100k tris floor) ------------------
  ctx.progress(0.96, 'gallery: hero trees');
  await new Promise((r) => setTimeout(r, 0));
  {
    const HZ = ROW_Z.hero;
    const heroSpecs = [TREE_SPECIES[0], TREE_SPECIES[2]];
    let hx = -14;
    for (const sp of heroSpecs) {
      if (!sp) continue;
      const built = buildTree(sp, seed.rng(`hero/${sp.id}`), { foliageMode: 'mesh', junctions: junctionsOn });
      const at = exhibit(hx, HZ, `HERO ${sp.label}`, `${(built.stats.tris / 1000).toFixed(0)}k tris (mesh foliage)`);
      const bm = new Mesh(built.bark, barkTexturedMaterial(barks.get(sp.barkLayer) as BarkTextures));
      bm.position.set(at.x, 0.42, at.z);
      bm.castShadow = true;
      bm.receiveShadow = true;
      engine.scene.add(bm);
      if (built.foliageMesh && !noLeaves) {
        const fm2 = new Mesh(built.foliageMesh, foliageMaterial({ color: sp.foliageColor }));
        fm2.position.copy(bm.position);
        fm2.castShadow = true;
        fm2.receiveShadow = true;
        engine.scene.add(fm2);
      }
      engine.stats.counters[`hero.${sp.id}`] = built.stats.tris;
      hx += 28;
    }
    // tree-base dressing: mushroom cluster at the beech hero
    // TODO(missing-leaves): the beech-atlas litter ring was removed with the card
    // pipeline (S8) — it sampled the deleted foliage atlas capture.
    const mrng = seed.rng('fungi');
    for (let i = 0; i < 6; i++) {
      const mush = new Mesh(buildMushroom(mrng.fork(String(i)), 'cap'), mushroomMaterial());
      const a2 = mrng.float() * 6.28;
      const rr = 0.5 + mrng.float() * 1.3;
      mush.position.set(14 + Math.cos(a2) * rr, 0.42, HZ + Math.sin(a2) * rr);
      mush.castShadow = true;
      engine.scene.add(mush);
    }
  }

  // ---- post stack (no clouds in the gallery) ----------------------------------
  ctx.progress(0.95, 'gallery: post pipeline');
  const post = new PostStack(engine, sunSky.atmosphere, params.timeOfDay, null);
  engine.post = post;

  ctx.hooks.setTimeOfDay = (t: number) => {
    void (async () => {
      await sunSky.setTimeOfDay(t);
      updateSunUniforms(sunSky.sun);
      post.setTimeOfDay(t);
    })();
  };

  // ---- camera ------------------------------------------------------------------
  if (params.cam === null) {
    const row = (q.get('row') ?? 'trees') as keyof typeof ROW_Z;
    const z = ROW_Z[row] ?? 0;
    engine.camera.position.set(0, 13, z + 64);
    engine.camera.lookAt(new Vector3(0, 9, z));
  }
  engine.onUpdate(() => {
    if (engine.camera.position.y < 0.6) engine.camera.position.y = 0.6;
  });

  ctx.progress(1, 'gallery ready');
}
