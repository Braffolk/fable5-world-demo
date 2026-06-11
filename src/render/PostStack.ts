/**
 * Post-processing stack (HDR, in order):
 *   scene pass (MRT: color, view normal, velocity, depth)
 *   → aerial perspective (Hillaire in-scatter from depth — Pillar D haze)
 *   → GTAO multiply (Phase-3 refines to indirect-only)
 *   → TRAA (temporal AA — the geometric density shimmers without it)
 *   → bloom (HDR threshold)
 *   → auto-exposure (GPU histogram-free log-average, smoothed, no readback)
 *   → filmic grade (per-ToD color script: white balance, teal–orange split
 *     toning, saturation, contrast) → AgX via renderer.toneMapping
 */

import { AgXToneMapping, Matrix4, NoToneMapping, Vector3 } from 'three';
import type { Renderer, StorageBufferNode } from 'three/webgpu';
import { RenderPipeline } from 'three/webgpu';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import {
  Fn,
  If,
  Return,
  clamp,
  dot,
  float,
  getScreenPosition,
  getViewPosition,
  instanceIndex,
  instancedArray,
  log2,
  exp2,
  luminance,
  mix,
  mrt,
  output,
  pass,
  rtt,
  screenUV,
  smoothstep,
  texture,
  uniform,
  vec2,
  vec3,
  vec4,
  velocity,
} from 'three/tsl';
import type { Engine } from '../core/Engine';
import type { Froxels } from '../gpu/passes/Froxels';
import { hash12 } from '../gpu/noise/NoiseTSL';
import type { NF, NV3, NV4 } from '../gpu/TSLTypes';
import type { Atmosphere } from '../sky/Atmosphere';
import { CLOUD_BOTTOM, CLOUD_TOP, type Clouds } from '../sky/Clouds';
import { GradeUniforms, gradeParamsAt } from './ColorScript';

export class PostStack {
  readonly post: RenderPipeline;
  private grade = new GradeUniforms();
  private exposureBuf: StorageBufferNode<'float'>;
  private exposureKernel: Parameters<Renderer['compute']>[0];

  constructor(
    engine: Engine,
    atmosphere: Atmosphere,
    tod: number,
    clouds: Clouds | null = null,
    froxels: Froxels | null = null,
  ) {
    const { renderer, scene, camera } = engine;
    const q = new URLSearchParams(window.location.search);
    const cloudview = q.get('cloudview');
    // perf attribution: ?ablate=clouds,ao,taa,bloom disables stages
    const ablate = new Set((q.get('ablate') ?? '').split(','));
    // debug probes need raw values — tone mapping would garble them
    renderer.toneMapping = cloudview ? NoToneMapping : AgXToneMapping;
    renderer.toneMappingExposure = 1.0;
    const frameU = uniform(0);
    // The post quad pass binds its own orthographic camera: `cameraPosition`,
    // `cameraWorldMatrix` and `cameraProjectionMatrixInverse` all resolve to
    // THAT camera inside outputNode (verified via ?cloudview probes — depth
    // reconstruction gave quad-local ~1 m distances). Feed the scene camera
    // explicitly, like three's own GTAO/TRAA nodes do.
    const uCamPos = uniform(new Vector3());
    const uProjInv = uniform(new Matrix4());
    const uCamWorld = uniform(new Matrix4());
    const uProj = uniform(new Matrix4());
    const uView = uniform(new Matrix4());
    engine.onUpdate(() => {
      frameU.value = (frameU.value + 1) % 1024;
      uCamPos.value.copy(camera.position);
      uProjInv.value.copy(camera.projectionMatrixInverse);
      uCamWorld.value.copy(camera.matrixWorld);
      uProj.value.copy(camera.projectionMatrix);
      uView.value.copy(camera.matrixWorldInverse);
    });
    const camPosW = vec3(uCamPos);

    const scenePass = pass(scene, camera);
    // ?postmin=1 — bisect probe: bare scene pass through the pipeline, no
    // MRT/effects. If shadows survive here but not in the full stack, an
    // effect node is the culprit; if they die here, pass() itself is.
    if (q.get('postmin') === '1') {
      if (q.get('postmrt') === '1') {
        scenePass.setMRT(mrt({ output, velocity }));
        this.post = new RenderPipeline(renderer);
        this.post.outputNode = scenePass.getTextureNode('output');
      } else {
        this.post = new RenderPipeline(renderer);
        this.post.outputNode = scenePass;
      }
      this.exposureBuf = instancedArray(2, 'float');
      this.exposureKernel = Fn(() => {})().compute(1);
      return;
    }
    scenePass.setMRT(
      mrt({
        output,
        velocity,
      }),
    );
    const beauty = scenePass.getTextureNode('output');
    const depthTex = scenePass.getTextureNode('depth');
    const velocityTex = scenePass.getTextureNode('velocity');

    // --- half-res volumetric cloud layer (own quad pass) -----------------------
    // The march is by far the most expensive screen-space work; running it at
    // half res quarters the ray count. Jitter + TRAA absorb the upsample.
    let cloudTex: NV4 | null = null;
    if (clouds && !ablate.has('clouds')) {
      const cloudLayer = Fn((): NV4 => {
        const d = depthTex.x;
        const viewDirV = getViewPosition(screenUV, float(0.5), uProjInv).normalize();
        const dirW = uCamWorld.mul(vec4(viewDirV, 0)).xyz.normalize().toVar();
        const dist = getViewPosition(screenUV, d, uProjInv).length();
        const isSky = d.lessThanEqual(1e-7).or(d.greaterThanEqual(0.9999999));
        const maxD = isSky.select(float(1e9), dist);
        const jitter = hash12(
          screenUV.mul(vec2(911.3, 423.7)).add(float(frameU).mul(0.61803)),
        );
        const cl = clouds.march(camPosW, dirW, maxD, jitter);
        return vec4(cl.color, cl.alpha);
      })();
      const cloudRtt = rtt(cloudLayer);
      const sizeClouds = (): void => {
        const dpr = renderer.getPixelRatio();
        cloudRtt.setSize(
          Math.max(2, Math.floor(window.innerWidth * dpr * 0.5)),
          Math.max(2, Math.floor(window.innerHeight * dpr * 0.5)),
        );
      };
      sizeClouds();
      window.addEventListener('resize', sizeClouds);
      cloudTex = cloudRtt as unknown as NV4;
    }

    // --- aerial perspective from depth -----------------------------------------
    const aerialNode = Fn((): NV3 => {
      const d = depthTex.x.toVar();
      const col = beauty.rgb.toVar();
      // ray direction from a FIXED finite depth (the far-plane depth value
      // degenerates through the inverse projection)
      const viewDirV = getViewPosition(screenUV, float(0.5), uProjInv).normalize();
      const dirW = uCamWorld.mul(vec4(viewDirV, 0)).xyz.normalize().toVar();
      const viewPos = getViewPosition(screenUV, d, uProjInv);
      const dist = viewPos.length();
      const distKm = dist.div(1000);
      const camAltKm = camPosW.y.div(1000).max(0.005);
      // sky = cleared depth; tolerate either depth convention (0 or 1 at far)
      const isSky = d.lessThanEqual(1e-7).or(d.greaterThanEqual(0.9999999));
      // froxel volumetrics first (local shafts/valley fog ≤ ~480 m), the
      // km-scale Hillaire haze integrates on top of the fogged radiance
      if (froxels) {
        const fogDist = isSky.select(float(1e5), dist);
        col.assign(froxels.apply(col, fogDist, screenUV));
      }
      const hazed = atmosphere.aerial(col, dirW, camAltKm, distKm);
      // reversed-z: far plane clears to 0 → sky already carries the atmosphere
      const scenePart = isSky.select(col, hazed).toVar();

      if (clouds && !ablate.has('clouds')) {
        const maxD = isSky.select(float(1e9), dist);
        if (cloudview === '2') {
          // constant output; march not built at all (graph-pollution bisect)
          scenePart.assign(vec3(1, 0, 0));
        } else if (cloudview === '7') {
          // ray-direction probe: R = dir.y, G = -dir.y, B = horizontalness
          scenePart.assign(
            vec3(
              clamp(dirW.y, 0, 1),
              clamp(dirW.y.negate(), 0, 1),
              dirW.y.abs().lessThan(1e-3).select(float(1), float(0)),
            ),
          );
        } else if (cloudview === '6') {
          // slab-intersection probe: R = valid, G = tEnter/10km, B = tExit/10km
          const t0 = float(CLOUD_BOTTOM).sub(camPosW.y).div(dirW.y);
          const t1 = float(CLOUD_TOP).sub(camPosW.y).div(dirW.y);
          const tEnterRaw = t0.min(t1);
          const tExitRaw = t0.max(t1);
          const ins = camPosW.y
            .greaterThan(CLOUD_BOTTOM)
            .and(camPosW.y.lessThan(CLOUD_TOP));
          const tEnter = ins.select(float(0), tEnterRaw.max(0));
          const tExit = tExitRaw.min(maxD).min(26000);
          const valid = tExit.greaterThan(tEnter).and(dirW.y.abs().greaterThan(1e-4));
          scenePart.assign(
            vec3(
              valid.select(clamp(tExit.div(10000), 0, 1), float(0)),
              clamp(tEnter.div(10000), 0, 1),
              clamp(dist.div(10000), 0, 1),
            ),
          );
        } else if (cloudview === '5') {
          // camera uniform probe: gray = camera height / 3000
          scenePart.assign(vec3(camPosW.y.div(3000)));
        } else if (cloudview === '3') {
          // isSky probe: white = far-plane depth
          scenePart.assign(isSky.select(vec3(1), vec3(0)));
        } else if (cloudTex) {
          const cl4 = cloudTex;
          if (cloudview === '1') {
            // march alpha as magenta overlay
            scenePart.assign(mix(scenePart, vec3(1, 0, 1), clamp(cl4.a, 0, 1)));
          } else {
            // depth-aware upsample gate: the cloud RTT is half-res, and
            // bilinear upsampling smears sky texels (visible through leaf
            // gaps) onto near geometry — clouds painted over close trees in
            // a woven pattern (user screenshot). A solid surface nearer
            // than the cloud-slab entry can have no cloud in front of it:
            // zero the contribution there. 300 m floor covers downward /
            // near-horizontal rays where the slab math degenerates.
            const t0 = float(CLOUD_BOTTOM).sub(camPosW.y).div(dirW.y);
            const t1 = float(CLOUD_TOP).sub(camPosW.y).div(dirW.y);
            const ins = camPosW.y
              .greaterThan(CLOUD_BOTTOM)
              .and(camPosW.y.lessThan(CLOUD_TOP));
            const tEnter = ins.select(float(0), t0.min(t1).max(0));
            const nearSolid = isSky.not().and(dist.lessThan(tEnter.max(300)));
            const k = nearSolid.select(float(0), float(1));
            scenePart.assign(
              scenePart.mul(float(1).sub(cl4.a.mul(k))).add(cl4.rgb.mul(k)),
            );
          }
        }
      }
      if (cloudview === '4') {
        // context probe: R/G = screenUV gradients, B = raw depth ×100
        scenePart.assign(vec3(screenUV.x, screenUV.y, clamp(d.mul(100), 0, 1)));
      }
      return scenePart;
    })();

    // --- GTAO (full res: resolutionScale 0.5 produced row-streak artifacts) ------
    // defaults are mesh-viewer scale: 16 samples cost ~50 ms on terrain vistas.
    // Normals = null → derived from depth: material normals carry strong
    // far-detail perturbation that disagrees with depth geometry — GTAO's
    // cones bent into the surface and printed black facets on steep ridges.
    const aoPass = ao(depthTex, null as unknown as typeof depthTex, camera);
    const aoCfg = aoPass as unknown as {
      samples: { value: number };
      radius: { value: number };
      distanceFallOff: { value: number };
    };
    aoCfg.samples.value = 8;
    aoCfg.radius.value = 1.6;
    aoCfg.distanceFallOff.value = 0.6;
    // AO is a near-field cue — fade it out with distance (far AO from a
    // 1.6 m radius is subpixel anyway and only adds instability)
    const aoFaded = Fn((): NF => {
      const dist = getViewPosition(screenUV, depthTex.x, uProjInv).length();
      const k = smoothstep(700, 1800, dist);
      // indirect-only approximation: sun-lit pixels (high HDR luminance)
      // shed most of the post-AO — occlusion belongs to ambient light.
      // (True aoNode-into-lighting wiring lands with the Phase-4 material
      // restructure; see DEVIATIONS.md.)
      const directK = smoothstep(1.2, 4.0, luminance(beauty.rgb)).mul(0.75);
      const aoRaw = aoPass.getTextureNode().x;
      return mix(mix(aoRaw, float(1), directK), float(1), k);
    })();
    // --- screen-space contact shadows (spec §2 floor) ---------------------------
    // Short depth-buffer march toward the sun: picks up the ~0.1–2 m contact
    // occlusion the 2048² cascades can't resolve. Near field only; floored so
    // it stays a contact CUE (never pitch black — no-black-shadows law).
    const SSCS_STEPS = 12;
    const contactNode = Fn((): NF => {
      const result = float(1).toVar();
      const d = depthTex.x;
      const isSky = d.lessThanEqual(1e-7).or(d.greaterThanEqual(0.9999999));
      const viewPos = getViewPosition(screenUV, d, uProjInv);
      const dist = viewPos.length();
      If(isSky.not().and(dist.lessThan(240)), () => {
        const sunW = vec3(atmosphere.sunDir).normalize();
        const sunV = uView.mul(vec4(sunW, 0)).xyz;
        const jit = hash12(screenUV.mul(vec2(517.7, 893.3)).add(float(frameU).mul(0.7548)))
          .mul(0.8)
          .add(0.4);
        const range = float(1.7);
        const occl = float(0).toVar();
        for (let s = 1; s <= SSCS_STEPS; s++) {
          // quadratic step distribution: dense near the surface
          const f = (s / SSCS_STEPS) ** 1.6;
          const sampleV = viewPos.add(sunV.mul(range).mul(jit).mul(f));
          const uvS = getScreenPosition(sampleV, uProj);
          const inFrame = uvS.x
            .greaterThan(0.001)
            .and(uvS.x.lessThan(0.999))
            .and(uvS.y.greaterThan(0.001))
            .and(uvS.y.lessThan(0.999));
          const dS = texture(depthTex.value, uvS).x;
          const bufV = getViewPosition(uvS, dS, uProjInv);
          const dz = bufV.z.sub(sampleV.z); // >0: buffer closer to camera
          const hit = dz.greaterThan(0.05).and(dz.lessThan(1.4)).and(inFrame);
          occl.assign(occl.max(hit.select(float(1).sub(float(f).mul(0.5)), float(0))));
        }
        // distance fade + floor
        const fade = smoothstep(240, 140, dist);
        result.assign(float(1).sub(occl.mul(0.6).mul(fade)));
      });
      return result;
    })();

    const withAO = ablate.has('ao')
      ? aerialNode
      : aerialNode.mul(aoFaded).mul(ablate.has('contact') ? float(1) : contactNode);

    // --- screen-space bounce / color bleed (DEVIATIONS D-2) --------------------
    // Half-res gather of nearby on-screen radiance, depth-gated, added back
    // modulated by the receiver's chroma. Subtle by design: probes carry the
    // large-scale bounce; this adds the local green-on-trunk / warm-on-rock
    // bleed that probes are too coarse for. ?ablate=bounce to A/B.
    let withBounce = withAO;
    if (!ablate.has('bounce')) {
      const bounceLayer = Fn((): NV4 => {
        const res = vec4(0).toVar();
        const d = depthTex.x;
        const isSky = d.lessThanEqual(1e-7).or(d.greaterThanEqual(0.9999999));
        If(isSky.not(), () => {
          const viewPos = getViewPosition(screenUV, d, uProjInv);
          const dist = viewPos.length();
          // ≈0.6 m world-space gather radius projected to screen
          const rPx = clamp(float(0.55).div(dist), 0.004, 0.07);
          const sum = vec3(0).toVar();
          const wsum = float(0).toVar();
          for (let i = 0; i < 8; i++) {
            const ga = i * 2.399963 + 0.7;
            const rr = Math.sqrt((i + 0.5) / 8);
            const offX = Math.cos(ga) * rr;
            const offY = Math.sin(ga) * rr;
            const uvS = screenUV.add(vec2(offX, offY).mul(rPx));
            const dS = texture(depthTex.value, uvS).x;
            const pS = getViewPosition(uvS, dS, uProjInv);
            const w = smoothstep(1.8, 0.25, pS.sub(viewPos).length());
            sum.addAssign(texture(beauty.value, uvS).rgb.mul(w));
            wsum.addAssign(w);
          }
          res.assign(vec4(sum.div(wsum.max(1e-3)), wsum.mul(0.125)));
        });
        return res;
      })();
      const bounceRtt = rtt(bounceLayer);
      const sizeBounce = (): void => {
        const dpr = renderer.getPixelRatio();
        bounceRtt.setSize(
          Math.max(2, Math.floor(window.innerWidth * dpr * 0.5)),
          Math.max(2, Math.floor(window.innerHeight * dpr * 0.5)),
        );
      };
      sizeBounce();
      window.addEventListener('resize', sizeBounce);
      // receiver albedo proxy: scene color normalized by its own luminance
      const recLum = luminance(withAO.rgb).add(0.25);
      const recTint = withAO.rgb.div(recLum);
      withBounce = withAO.rgb.add(
        bounceRtt.rgb.mul(recTint).mul(bounceRtt.a).mul(0.16),
      ) as unknown as typeof withAO;
    }

    // --- TRAA ----------------------------------------------------------------------
    const taaed = ablate.has('taa')
      ? (withBounce as unknown as ReturnType<typeof traa>)
      : traa(withBounce, depthTex, velocityTex, camera);

    // --- bloom -----------------------------------------------------------------------
    const taaedRgb = (taaed as unknown as NV4).rgb;
    const withBloom = ablate.has('bloom')
      ? taaedRgb
      : taaedRgb.add((bloom(taaed, 0.28, 0.45, 1.5) as unknown as NV4).rgb);

    // --- auto exposure (GPU-only feedback) ----------------------------------------------
    this.exposureBuf = instancedArray(2, 'float');
    const expInit = Fn(() => {
      this.exposureBuf.element(0).assign(1);
      this.exposureBuf.element(1).assign(1);
    })().compute(1);
    void renderer.computeAsync(expInit);

    const beautyForMeter = scenePass.getTextureNode('output');
    this.exposureKernel = Fn(() => {
      If(instanceIndex.greaterThanEqual(1), () => {
        Return();
      });
      const logSum = float(0).toVar();
      const N = 12;
      for (let gy = 0; gy < N; gy++) {
        for (let gx = 0; gx < N; gx++) {
          const u = (gx + 0.5) / N;
          const v = (gy + 0.5) / N;
          // center-weighted metering
          const w = 1 - 0.55 * Math.hypot(u - 0.5, (v - 0.5) * 0.9);
          const c = texture(beautyForMeter.value, vec2(u, v)).rgb;
          const lum = luminance(c).max(1e-4);
          logSum.addAssign(log2(lum).mul(w));
        }
      }
      let wTot = 0;
      for (let gy = 0; gy < N; gy++) {
        for (let gx = 0; gx < N; gx++) {
          wTot += 1 - 0.55 * Math.hypot((gx + 0.5) / N - 0.5, ((gy + 0.5) / N - 0.5) * 0.9);
        }
      }
      const avgLum = exp2(logSum.div(wTot));
      // key: auto-exposure normalizes the frame to mid-gray — the key sets
      // WHICH gray. 0.125 floated forest scenes into a washy high-key; 0.1
      // keeps deep canopy darks so the sun reads (user: "washed out").
      // Gain cap 4 (was 7): a fully canopy-shadowed interior must STAY a
      // dark frame (scene1 value structure: dark frame → lit mid → bright
      // bg) — at ×7 the meter dragged it to pastel mid-gray and noon
      // interiors read overcast.
      const target = clamp(float(0.1).div(avgLum), 0.18, 4.0);
      const prev = this.exposureBuf.element(0);
      this.exposureBuf.element(0).assign(mix(prev, target, 0.07));
    })().compute(1);
    this.exposureKernel.setName('autoExposure');

    // --- grade ------------------------------------------------------------------------------
    const uWB = uniform(this.grade.whiteBalance);
    const uShadowTint = uniform(this.grade.shadowTint);
    const uHighlightTint = uniform(this.grade.highlightTint);
    const uShadowAmt = uniform(0.3);
    const uHighlightAmt = uniform(0.2);
    const uSat = uniform(1.0);
    const uContrast = uniform(1.03);
    this.uniformsRefresh = (): void => {
      uShadowAmt.value = this.grade.shadowAmt;
      uHighlightAmt.value = this.grade.highlightAmt;
      uSat.value = this.grade.saturation;
      uContrast.value = this.grade.contrast;
    };

    const graded = Fn((): NV3 => {
      let c: NV3 = withBloom.mul(this.exposureBuf.element(0));
      c = c.mul(vec3(uWB));
      const lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
      const shadowMask = smoothstep(0.45, 0.08, lum).mul(float(uShadowAmt));
      c = mix(c, c.mul(vec3(uShadowTint)), shadowMask);
      const hiMask = smoothstep(0.35, 0.95, lum).mul(float(uHighlightAmt));
      c = mix(c, c.mul(vec3(uHighlightTint)), hiMask);
      // saturation + gentle contrast around mid-gray
      c = mix(vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), c, float(uSat));
      c = c.div(0.18).pow(vec3(float(uContrast))).mul(0.18);
      // restrained vignette + static grain (freeze-deterministic)
      const v = screenUV.sub(0.5);
      const vig = float(1).sub(dot(v, v).mul(0.42));
      const grain = hash12(screenUV.mul(vec2(1923.7, 1671.3))).sub(0.5).mul(0.012);
      return c.mul(vig).add(grain);
    })();

    this.post = new RenderPipeline(renderer);
    // chain bisect: 9 = constant at pipeline output, 8 = aerial only (no
    // AO/TRAA/bloom/exposure/grade), default = full chain
    this.post.outputNode =
      cloudview === '9' ? vec3(1, 0, 0)
      : cloudview !== null && cloudview !== '' ? aerialNode
      : graded;

    this.setTimeOfDay(tod);
  }

  private uniformsRefresh: () => void = () => undefined;

  setTimeOfDay(tod: number): void {
    this.grade.apply(gradeParamsAt(tod));
    this.uniformsRefresh();
  }

  /** call once per frame after render — updates exposure feedback */
  meter(renderer: Renderer): void {
    renderer.compute(this.exposureKernel);
  }

  render(): void {
    this.post.render();
  }
}

