/**
 * Physical sky — Hillaire-style LUT atmosphere (units: kilometers).
 *
 *  - transmittance LUT (256×64): optical depth to atmosphere top
 *  - multiple-scattering LUT (32×32): isotropic 2nd-order Ψ term
 *  - sky-view LUT (192×108): full in-scatter panorama around the camera,
 *    re-baked whenever the sun moves (time of day)
 *
 * Provides: background node (sky + limb-darkened sun disc), aerial
 * perspective (applied in post from depth), CPU sun transmittance for the
 * DirectionalLight color, and an IBL refresh hook.
 */

import { Vector3 } from 'three';
import type { ComputeNode, Renderer } from 'three/webgpu';
import { StorageTexture } from 'three/webgpu';
import { HalfFloatType } from 'three';
import {
  Fn,
  sign,
  If,
  Loop,
  Return,
  asin,
  atan,
  clamp,
  cos,
  exp,
  float,
  instanceIndex,
  max,
  mix,
  positionWorldDirection,
  pow,
  sin,
  smoothstep,
  sqrt,
  texture,
  textureStore,
  uniform,
  uvec2,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';
import { vexp3 } from '../gpu/TSLTypes';
import type { NF, NI, NV2, NV3 } from '../gpu/TSLTypes';

// --- physical constants (km) --------------------------------------------------
const RG = 6360; // ground radius
const RT = 6460; // atmosphere top
const HR = 8.0; // rayleigh scale height
const HM = 1.2; // mie scale height
// rayleigh standard; mie boosted ~2.4× over the clean-air baseline — the
// references are humid scenes with pronounced horizon haze and depth layering
const BETA_R: [number, number, number] = [5.802e-3, 13.558e-3, 33.1e-3];
const BETA_M_S = 9.6e-3;
const BETA_M_E = 1.07e-2;
const BETA_O: [number, number, number] = [0.65e-3, 1.881e-3, 0.085e-3];
const MIE_G = 0.8;
// 3x the physical 0.265° radius: games oversize the disc — at true scale
// it reads as a tiny dot (user feedback batch 2 item 10). Disc radiance is
// dimmed below to keep total flux (and the bloom response) in range.
const SUN_ANGULAR_RADIUS = 0.014;

const T_W = 256;
const T_H = 64;
const MS_RES = 32;
const SV_W = 192;
const SV_H = 108;

/**
 * Sun irradiance at the top of atmosphere in RENDER UNITS. LUTs are baked
 * with E=1; every sky/aerial sample is scaled by this, and the sun
 * DirectionalLight uses the same constant — one knob keeps sun:sky physical.
 */
export const SUN_E = 8.0;

// --- TSL atmosphere helpers ----------------------------------------------------

const betaR = vec3(...BETA_R);
const betaO = vec3(...BETA_O);

function densities(h: NF): { dr: NF; dm: NF; doz: NF } {
  const dr = exp(h.div(-HR));
  const dm = exp(h.div(-HM));
  const doz = max(0, float(1).sub(h.sub(25).abs().div(15)));
  return { dr, dm, doz };
}

/** distance from (r, mu) to the atmosphere top along the ray */
function distToTop(r: NF, mu: NF): NF {
  const disc = r.mul(r).mul(mu.mul(mu).sub(1)).add(RT * RT);
  return max(0, r.negate().mul(mu).add(sqrt(max(0, disc))));
}

/** distance to ground, or -1 if the ray misses */
function distToGround(r: NF, mu: NF): NF {
  const disc = r.mul(r).mul(mu.mul(mu).sub(1)).add(RG * RG);
  const d = r.negate().mul(mu).sub(sqrt(max(0, disc)));
  return disc.lessThan(0).or(d.lessThan(0)).select(float(-1), d);
}

/** transmittance LUT uv from (r, mu) — Bruneton mapping */
function transmittanceUv(r: NF, mu: NF): NV2 {
  const H = Math.sqrt(RT * RT - RG * RG);
  const rho = sqrt(max(0, r.mul(r).sub(RG * RG)));
  const d = distToTop(r, mu);
  const dMin = float(RT).sub(r);
  const dMax = rho.add(H);
  const xMu = d.sub(dMin).div(dMax.sub(dMin));
  const xR = rho.div(H);
  return vec2(clamp(xMu, 0, 1), clamp(xR, 0, 1));
}

const RAYLEIGH_PHASE_K = 3 / (16 * Math.PI);

function rayleighPhase(nu: NF): NF {
  return nu.mul(nu).add(1).mul(RAYLEIGH_PHASE_K);
}

/** Cornette–Shanks */
function miePhase(nu: NF): NF {
  const g = MIE_G;
  const g2 = g * g;
  const k = (3 / (8 * Math.PI)) * ((1 - g2) / (2 + g2));
  const num = nu.mul(nu).add(1);
  const den = pow(float(1 + g2).sub(nu.mul(2 * g)), 1.5);
  return num.div(den).mul(k);
}

export class Atmosphere {
  readonly transmittanceLUT: StorageTexture;
  readonly multiScatterLUT: StorageTexture;
  readonly skyViewLUT: StorageTexture;
  /** sun direction (world, normalized, y up) */
  readonly sunDir = uniform(new Vector3(0.3, 0.6, 0.2).normalize());
  private renderer: Renderer | null = null;
  private skyCompute: ComputeNode | null = null;

  constructor() {
    this.transmittanceLUT = this.makeLUT(T_W, T_H, 'atmoTransmittanceLut');
    this.multiScatterLUT = this.makeLUT(MS_RES, MS_RES, 'atmoMultiScatterLut');
    this.skyViewLUT = this.makeLUT(SV_W, SV_H, 'atmoSkyViewLut');
  }

  private makeLUT(w: number, h: number, name: string): StorageTexture {
    const t = new StorageTexture(w, h);
    t.name = name;
    t.type = HalfFloatType;
    t.generateMipmaps = false;
    return t;
  }

  /** sample transmittance toward the sun/top from radius r, cos zenith mu */
  sampleTransmittance(r: NF, mu: NF): NV3 {
    return texture(this.transmittanceLUT, transmittanceUv(r, mu)).rgb;
  }

  private sampleMultiScatter(r: NF, muS: NF): NV3 {
    const u = clamp(muS.mul(0.5).add(0.5), 0, 1);
    const v = clamp(r.sub(RG).div(RT - RG), 0, 1);
    return texture(this.multiScatterLUT, vec2(u, v)).rgb;
  }

  async init(renderer: Renderer): Promise<void> {
    this.renderer = renderer;

    // --- transmittance bake ----------------------------------------------------
    const tK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(T_W * T_H), () => {
        Return();
      });
      const x = i.mod(T_W);
      const y = i.div(T_W);
      const uv = vec2(float(x).add(0.5).div(T_W), float(y).add(0.5).div(T_H));
      const H = Math.sqrt(RT * RT - RG * RG);
      const rho = uv.y.mul(H);
      const r = sqrt(rho.mul(rho).add(RG * RG));
      const dMin = float(RT).sub(r);
      const dMax = rho.add(H);
      const d = mix(dMin, dMax, uv.x);
      const mu = d
        .lessThanEqual(1e-4)
        .select(float(1), float(H * H).sub(rho.mul(rho)).sub(d.mul(d)).div(r.mul(d).mul(2)));
      const muC = clamp(mu, -1, 1);

      const STEPS = 44;
      const dt = d.div(STEPS);
      const tau = vec3(0).toVar();
      Loop(STEPS, ({ i: si }: { readonly i: NI }) => {
        const t = float(si).add(0.5).mul(dt);
        const rx = sqrt(t.mul(t).add(r.mul(r)).add(t.mul(r).mul(muC).mul(2)));
        const h = rx.sub(RG);
        const { dr, dm, doz } = densities(h);
        tau.addAssign(
          betaR.mul(dr).add(vec3(BETA_M_E).mul(dm)).add(betaO.mul(doz)).mul(dt),
        );
      });
      textureStore(this.transmittanceLUT, uvec2(x.toUint(), y.toUint()), vec4(vexp3(tau.negate()), 1)).toWriteOnly();
    })().compute(T_W * T_H);
    tK.setName('atmoTransmittance');
    await renderer.computeAsync(tK);

    // --- multiple-scattering bake -----------------------------------------------
    // 64 uniform sphere directions (golden spiral), unrolled as constants
    const dirs: [number, number, number][] = [];
    for (let k = 0; k < 64; k++) {
      const zc = 1 - (2 * k + 1) / 64;
      const rr = Math.sqrt(Math.max(0, 1 - zc * zc));
      const phi = k * 2.399963229728653;
      dirs.push([rr * Math.cos(phi), zc, rr * Math.sin(phi)]);
    }
    const msK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(MS_RES * MS_RES), () => {
        Return();
      });
      const x = i.mod(MS_RES);
      const y = i.div(MS_RES);
      const muS = float(x).add(0.5).div(MS_RES).mul(2).sub(1);
      const r = float(y).add(0.5).div(MS_RES).mul(RT - RG).add(RG).min(RT - 0.01);
      const sunD = vec3(sqrt(max(0, float(1).sub(muS.mul(muS)))), muS, 0);

      const L2 = vec3(0).toVar();
      const fms = vec3(0).toVar();
      for (const [dx, dy, dz] of dirs) {
        const dir = vec3(dx, dy, dz);
        const mu = dir.y;
        const dGround = distToGround(r, mu);
        const dTop = distToTop(r, mu);
        const dEnd = dGround.greaterThan(0).select(dGround, dTop);
        const STEPS = 18;
        const dt = dEnd.div(STEPS);
        const T = vec3(1).toVar();
        Loop(STEPS, ({ i: si }: { readonly i: NI }) => {
          const t = float(si).add(0.5).mul(dt);
          const rx = sqrt(t.mul(t).add(r.mul(r)).add(t.mul(r).mul(mu).mul(2)));
          const h = rx.sub(RG).max(0);
          const { dr, dm, doz } = densities(h);
          const sigmaS = betaR.mul(dr).add(vec3(BETA_M_S).mul(dm));
          const sigmaE = betaR.mul(dr).add(vec3(BETA_M_E).mul(dm)).add(betaO.mul(doz));
          const muSx = rx.mul(muS).add(t.mul(dir.dot(sunD))).div(rx).clamp(-1, 1);
          const tSun = this.sampleTransmittance(rx, muSx);
          const stepT = vexp3(sigmaE.mul(dt).negate());
          // isotropic phase 1/4π; energy-conserving integration
          const Sint = sigmaS.mul(tSun).mul(1 / (4 * Math.PI));
          L2.addAssign(T.mul(Sint.sub(Sint.mul(stepT))).div(sigmaE.max(1e-6)));
          fms.addAssign(T.mul(sigmaS.sub(sigmaS.mul(stepT))).div(sigmaE.max(1e-6)));
          T.mulAssign(stepT);
        });
        // ground bounce (albedo 0.3)
        const groundHit = dGround.greaterThan(0);
        const muSg = clamp(muS, 0, 1);
        const tg = this.sampleTransmittance(float(RG + 0.01), muSg);
        L2.addAssign(groundHit.select(T.mul(tg).mul(muSg).mul(0.3 / Math.PI), vec3(0)));
      }
      const inv = 1 / 64;
      const L2a = L2.mul(inv);
      const fmsA = fms.mul(inv);
      const psi = L2a.div(max(1e-4, float(1).sub(fmsA.x.add(fmsA.y).add(fmsA.z).div(3))));
      textureStore(this.multiScatterLUT, uvec2(x.toUint(), y.toUint()), vec4(psi, 1)).toWriteOnly();
    })().compute(MS_RES * MS_RES);
    msK.setName('atmoMultiScatter');
    await renderer.computeAsync(msK);

    // --- sky-view kernel (re-run per sun change) ----------------------------------
    const svK = Fn(() => {
      const i = instanceIndex;
      If(i.greaterThanEqual(SV_W * SV_H), () => {
        Return();
      });
      const x = i.mod(SV_W);
      const y = i.div(SV_W);
      const u = float(x).add(0.5).div(SV_W);
      const v = float(y).add(0.5).div(SV_H);
      // sqrt-warped elevation: horizon detail concentrated at v=0.5
      const lRaw = v.mul(2).sub(1);
      const elev = lRaw.abs().mul(lRaw.abs()).mul(Math.PI / 2).mul(sign(lRaw));
      // azimuth relative to sun azimuth
      const sunAz = atan(this.sunDir.z, this.sunDir.x);
      const az = u.mul(2 * Math.PI).sub(Math.PI).add(sunAz);
      const dir = vec3(cos(elev).mul(cos(az)), sin(elev), cos(elev).mul(sin(az)));

      const r = float(RG + 0.35);
      const mu = dir.y;
      const dGround = distToGround(r, mu);
      const dTop = distToTop(r, mu);
      const dEnd = dGround.greaterThan(0).select(dGround, dTop).min(180);

      const sunDirN = this.sunDir.normalize();
      const nu = dir.dot(sunDirN).clamp(-1, 1);
      const pR = rayleighPhase(nu);
      const pM = miePhase(nu);

      const STEPS = 40;
      const dt = dEnd.div(STEPS);
      const L = vec3(0).toVar();
      const T = vec3(1).toVar();
      Loop(STEPS, ({ i: si }: { readonly i: NI }) => {
        const t = float(si).add(0.5).mul(dt);
        const rx = sqrt(t.mul(t).add(r.mul(r)).add(t.mul(r).mul(mu).mul(2)));
        const h = rx.sub(RG).max(0);
        const { dr, dm, doz } = densities(h);
        const sigmaSR = betaR.mul(dr);
        const sigmaSM = vec3(BETA_M_S).mul(dm);
        const sigmaE = betaR.mul(dr).add(vec3(BETA_M_E).mul(dm)).add(betaO.mul(doz)).max(1e-9);
        const muSx = rx.mul(sunDirN.y).add(t.mul(nu)).div(rx).clamp(-1, 1);
        const tSun = this.sampleTransmittance(rx, muSx);
        const psi = this.sampleMultiScatter(rx, muSx);
        const S = sigmaSR
          .mul(pR)
          .add(sigmaSM.mul(pM))
          .mul(tSun)
          .add(psi.mul(sigmaSR.add(sigmaSM)));
        const stepT = vexp3(sigmaE.mul(dt).negate());
        L.addAssign(T.mul(S.sub(S.mul(stepT))).div(sigmaE));
        T.mulAssign(stepT);
      });
      // rays that hit ground: add sun-lit distant land (hazy luminous horizon,
      // not a black void below the horizon line)
      const hitGround = dGround.greaterThan(0);
      const muSg = clamp(sunDirN.y, 0, 1);
      const tSunG = this.sampleTransmittance(float(RG + 0.05), muSg);
      const groundL = T.mul(tSunG).mul(muSg).mul(0.32 / Math.PI);
      L.addAssign(hitGround.select(groundL, vec3(0)));
      textureStore(this.skyViewLUT, uvec2(x.toUint(), y.toUint()), vec4(L, 1)).toWriteOnly();
    })().compute(SV_W * SV_H);
    svK.setName('atmoSkyView');
    this.skyCompute = svK;
    await renderer.computeAsync(svK);
  }

  /** point the sun (unit world dir) and re-bake the sky-view LUT */
  async setSun(dir: Vector3): Promise<void> {
    this.sunDir.value.copy(dir).normalize();
    if (this.renderer && this.skyCompute) {
      await this.renderer.computeAsync(this.skyCompute);
    }
  }

  /** sample the sky-view LUT for an arbitrary world direction */
  skyColor(dir: NV3): NV3 {
    const elev = asin(clamp(dir.y, -1, 1));
    const lRaw = elev.div(Math.PI / 2);
    const v = sqrt(lRaw.abs()).mul(sign(lRaw)).mul(0.5).add(0.5);
    const sunAz = atan(this.sunDir.z, this.sunDir.x);
    const az = atan(dir.z, dir.x).sub(sunAz);
    const u = az.div(2 * Math.PI).add(0.5).fract();
    return texture(this.skyViewLUT, vec2(u, v)).rgb.mul(SUN_E);
  }

  /** full background: sky + transmittance-tinted, limb-darkened sun disc */
  backgroundNode(): NV3 {
    const dir = positionWorldDirection.normalize();
    const sky = this.skyColor(dir);
    const sunDirN = this.sunDir.normalize();
    const cosA = dir.dot(sunDirN);
    const cosR = Math.cos(SUN_ANGULAR_RADIUS);
    const inDisc = smoothstep(cosR, cosR + 0.00008, cosA);
    const centerT = clamp(
      cosA.sub(cosR).div(1 - cosR),
      0,
      1,
    );
    const limb = pow(centerT, 0.45).mul(0.55).add(0.45);
    const tSun = this.sampleTransmittance(float(RG + 0.35), clamp(sunDirN.y, -1, 1));
    const sunL = tSun.mul(limb).mul(inDisc).mul(50 * SUN_E);
    return sky.add(sunL);
  }

  /**
   * Aerial perspective for post: altitude-aware analytic atmosphere PLUS a
   * boundary-layer haze term (humid valley mist — the references' depth
   * layering is this, not clean-air rayleigh, which is subtle under 15 km).
   */
  aerial(color: NV3, viewDir: NV3, camAltKm: NF, distKm: NF): NV3 {
    const fragAlt = camAltKm.add(viewDir.y.mul(distKm)).max(0);
    const hAvg = camAltKm.add(fragAlt).mul(0.5).max(0.02);
    const dr = exp(hAvg.div(-HR));
    const dm = exp(hAvg.div(-HM));
    const tauR = betaR.mul(dr).mul(distKm);
    const tauM = float(BETA_M_E).mul(dm).mul(distKm);

    // boundary-layer fog: density k·exp(−(h−h0)/Hf); exact path integral
    const FOG_K = 0.22; // extinction at the reference height, per km
    const FOG_H0 = 0.16; // reference altitude (≈ valley floor), km
    const FOG_HF = 0.38; // scale height, km
    const y0 = camAltKm.sub(FOG_H0).div(FOG_HF);
    const y1 = fragAlt.sub(FOG_H0).div(FOG_HF);
    const dy = y1.sub(y0);
    const flat = dy.abs().lessThan(1e-4);
    const integ = flat.select(
      exp(y0.negate()).mul(distKm),
      exp(y0.negate()).sub(exp(y1.negate())).div(dy).mul(distKm),
    );
    const tauF = integ.mul(FOG_K).max(0);

    const T = vexp3(tauR.add(tauM).add(tauF).negate());
    const sky = this.skyColor(viewDir);
    // per-channel energy exchange: blue extinguishes first → haze reads blue
    return color.mul(T).add(sky.mul(vec3(1).sub(T)));
  }

  /** CPU transmittance toward the sun at ground level (for light color) */
  sunTransmittanceCpu(sunDir: Vector3): [number, number, number] {
    const r = RG + 0.35;
    const mu = Math.max(-1, Math.min(1, sunDir.y));
    const disc = r * r * (mu * mu - 1) + RT * RT;
    const d = Math.max(0, -r * mu + Math.sqrt(Math.max(0, disc)));
    const N = 40;
    const dt = d / N;
    const tau = [0, 0, 0];
    for (let s = 0; s < N; s++) {
      const t = (s + 0.5) * dt;
      const rx = Math.sqrt(t * t + r * r + 2 * t * r * mu);
      const h = Math.max(0, rx - RG);
      const drD = Math.exp(-h / HR);
      const dmD = Math.exp(-h / HM);
      const dozD = Math.max(0, 1 - Math.abs(h - 25) / 15);
      for (let c = 0; c < 3; c++) {
        tau[c] =
          (tau[c] as number) +
          ((BETA_R[c] as number) * drD + BETA_M_E * dmD + (BETA_O[c] as number) * dozD) * dt;
      }
    }
    return [Math.exp(-(tau[0] as number)), Math.exp(-(tau[1] as number)), Math.exp(-(tau[2] as number))];
  }
}
