/**
 * PERF-4 effect #3 — a leaner TAA (USER-sanctioned fork, LOG be / ROADMAP PERF-4-TAA).
 * `TRAANode.resolve` is fetch-bound (~6.8 ms at the worst view; ~18 texture
 * loads/pixel): a 3×3 = 9-tap depth neighborhood (velocity dilation + edge
 * detect) + an 8-tap variance neighborhood (the ghosting guard) + history. three
 * exposes no knob for those, so this SUBCLASSES TRAANode, reuses ALL its infra
 * (history/resolve RTs, jitter, velocity, camera-matrix uniforms) via super.setup,
 * then swaps in a FAITHFUL copy of three 0.184's resolve with the two neighborhoods
 * shrunk. Every other line (clipAABB, variance math, disocclusion, flicker,
 * weights) is three's logic — only the offset lists change.
 *
 * MEASURED (LOG bf, high-res GPU-bound — the only regime that cuts the native
 * thermal noise): the win is MARGINAL. Cutting 4 fetches (variance 8→4) saves
 * ~1 ms at 3888×2520 ⇒ ~0.45 ms native; the full cut (also depth 9→5) ~2×. NOT the
 * ~3 ms the fetch count projected — because `TRAANode.resolve` is ALU+drain-bound,
 * not fetch-bound (the variance/clip/flicker MATH dominates, the loads don't). So
 * this trades the ghosting guard for a sub-ms native gain ⇒ DEFAULT OFF, not shipped;
 * kept gated for the user's in-motion A/B + as the foundation if TAA is ever reworked.
 * `?taacheap`: 1 = variance 8→4 (full 3×3 depth kept); 2 = also depth 9→5.
 *
 * Default OFF. Re-verify against the pinned three source on any upgrade. Assumes
 * CLASSIC depth (our renderer: reversedDepthBuffer + logarithmicDepthBuffer both
 * unset, D-N18) so three's reversed/log branches are omitted.
 *
 * TSL is dynamically typed; `TN` is a permissive chainable view of a node and the
 * `_`-prefixed locals are TN-typed wrappers of the three/tsl imports, so the ported
 * body below reads like three's JS with all the casts contained to this header.
 */

import type { Camera, NodeBuilder, TextureNode } from 'three/webgpu';
import TRAANode from 'three/addons/tsl/display/TRAANode.js';
import {
  Fn,
  If,
  add,
  convertToTexture,
  float,
  getViewPosition,
  ivec2,
  luminance,
  max,
  mix,
  struct,
  texture,
  uv,
  vec2,
  vec4,
  viewZToOrthographicDepth,
  viewZToPerspectiveDepth,
} from 'three/tsl';

/** permissive chainable view of a TSL node (only the ops this resolve uses). */
interface TN {
  add(x: TN | number): TN; sub(x: TN | number): TN; mul(x: TN | number): TN; div(x: TN | number): TN;
  max(x: TN | number): TN; min(x: TN | number): TN; clamp(a: TN | number, b: TN | number): TN;
  pow2(): TN; abs(): TN; sqrt(): TN; negate(): TN; oneMinus(): TN; fract(): TN; length(): TN; saturate(): TN; normalize(): TN;
  lessThan(x: TN | number): TN; greaterThan(x: TN | number): TN; greaterThanEqual(x: TN | number): TN; lessThanEqual(x: TN | number): TN;
  and(x: TN): TN; or(x: TN): TN; not(): TN; all(): TN;
  select(a: TN | number, b: TN | number): TN;
  assign(x: TN | number): void; addAssign(x: TN | number): void; mulAssign(x: TN | number): void;
  toVar(): TN; get(k: string): TN;
  load(t: TN): TN; sample(uv: TN): TN; offset(o: TN): TN; size(): TN;
  r: TN; g: TN; b: TN; a: TN; x: TN; y: TN; z: TN; w: TN; xy: TN; xyz: TN; rgb: TN;
}

const t = (x: unknown): TN => x as TN;
// TN-typed wrappers of the three/tsl imports (casts contained here).
const _float = float as unknown as (n: number) => TN;
const _vec2 = vec2 as unknown as (...a: Array<TN | number>) => TN;
const _vec4 = vec4 as unknown as (...a: Array<TN | number>) => TN;
const _ivec2 = ivec2 as unknown as (a: number, b: number) => TN;
const _uv = uv as unknown as () => TN;
const _max = max as unknown as (...a: Array<TN | number>) => TN;
const _add = add as unknown as (...a: Array<TN | number>) => TN;
const _mix = mix as unknown as (a: TN | number, b: TN | number, c: TN | number) => TN;
const _texture = texture as unknown as (tex: unknown) => TN;
const _struct = struct as unknown as (def: Record<string, string>) => (...a: TN[]) => TN;
const _luminance = luminance as unknown as (x: TN) => TN;
const _getViewPosition = getViewPosition as unknown as (uvN: TN, depth: TN, projInv: unknown) => TN;
const _viewZToPerspectiveDepth = viewZToPerspectiveDepth as unknown as (z: TN, n: TN, f: TN) => TN;
const _viewZToOrthographicDepth = viewZToOrthographicDepth as unknown as (z: TN, n: TN, f: TN) => TN;
const _If = If as unknown as (cond: TN, cb: () => void) => void;
const _Fn = Fn as unknown as (cb: () => TN) => () => TN;

/** the private TRAANode internals the resolve reads (set up by super.setup). */
interface TraaInternals {
  _previousDepthNode: { sample(uv: TN): TN };
  _historyRenderTarget: { texture: unknown };
  _cameraNearFar: { x: TN; y: TN };
  _previousCameraProjectionMatrixInverse: unknown;
  _previousCameraWorldMatrix: { mul(v: TN): TN };
  _cameraWorldMatrixInverse: { mul(v: TN): TN };
  _resolveMaterial: { colorNode: unknown };
}

const CROSS5: ReadonlyArray<readonly [number, number]> = [
  [0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
];
const BOX9: ReadonlyArray<readonly [number, number]> = [
  [-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1],
];
// the variance neighbourhood, cut from three's 8 to a 4-tap cross (the win).
const VAR4: ReadonlyArray<readonly [number, number]> = [[1, 0], [0, -1], [0, 1], [-1, 0]];

class LeanTraaNode extends TRAANode {
  private readonly cutLevel: number;

  constructor(
    beautyNode: TextureNode,
    depthNode: TextureNode,
    velocityNode: TextureNode,
    camera: Camera,
    cutLevel: number,
  ) {
    super(beautyNode, depthNode, velocityNode, camera);
    this.cutLevel = cutLevel;
  }

  override setup(builder: NodeBuilder): ReturnType<TRAANode['setup']> {
    // build three's full node (RTs, jitter, velocity wiring, its own resolve)…
    const out = (super.setup as (b: NodeBuilder) => ReturnType<TRAANode['setup']>).call(this, builder);
    // …then overwrite ONLY the resolve fragment with the leaner one.
    const priv = this as unknown as TraaInternals;
    priv._resolveMaterial.colorNode = this.leanResolve(priv);
    return out;
  }

  private leanResolve(priv: TraaInternals): unknown {
    // the variance cut IS the win (LOG be); the depth cut adds ~0 ms so keep the
    // full 3×3 unless level 2 explicitly asks to shrink it too.
    const depthOffsets = this.cutLevel >= 2 ? CROSS5 : BOX9;
    const varOffsets = VAR4;
    const beauty = t(this.beautyNode);
    const depthN = t(this.depthNode);
    const velN = t(this.velocityNode);
    const near = priv._cameraNearFar.x;
    const far = priv._cameraNearFar.y;
    const isOrtho = (this.camera as { isOrthographicCamera?: boolean }).isOrthographicCamera === true;
    const historyNode = _texture(priv._historyRenderTarget.texture);
    const currentDepthStruct = _struct({
      closestDepth: 'float',
      closestPositionTexel: 'vec2',
      farthestDepth: 'float',
    });

    // 3×3 → cross: closest (velocity-dilation target) + farthest (edge detect).
    const sampleCurrentDepth = (positionTexel: TN): TN => {
      const closestDepth = _float(2).toVar();
      const closestPositionTexel = _vec2(0, 0).toVar();
      const farthestDepth = _float(-1).toVar();
      for (const [x, y] of depthOffsets) {
        const neighbor = positionTexel.add(_vec2(x, y)).toVar();
        const depth = depthN.load(neighbor).r.toVar();
        _If(depth.lessThan(closestDepth), () => {
          closestDepth.assign(depth);
          closestPositionTexel.assign(neighbor);
        });
        _If(depth.greaterThan(farthestDepth), () => {
          farthestDepth.assign(depth);
        });
      }
      return currentDepthStruct(closestDepth, closestPositionTexel, farthestDepth);
    };

    // reproject a previous-frame depth with the current camera matrices.
    const samplePreviousDepth = (uvN: TN): TN => {
      const depth = priv._previousDepthNode.sample(uvN).r;
      const positionView = _getViewPosition(uvN, depth, priv._previousCameraProjectionMatrixInverse);
      const positionWorld = priv._previousCameraWorldMatrix.mul(_vec4(positionView, 1)).xyz;
      const viewZ = priv._cameraWorldMatrixInverse.mul(_vec4(positionWorld, 1)).z;
      return isOrtho
        ? _viewZToOrthographicDepth(viewZ, near, far)
        : _viewZToPerspectiveDepth(viewZ, near, far);
    };

    // optimized AABB clipping (playdead temporal) — three's math.
    const clipAABB = (currentColor: TN, historyColor: TN, minColor: TN, maxColor: TN): TN => {
      const pClip = maxColor.rgb.add(minColor.rgb).mul(0.5);
      const eClip = maxColor.rgb.sub(minColor.rgb).mul(0.5).add(1e-7);
      const vClip = historyColor.sub(_vec4(pClip, currentColor.a));
      const vUnit = vClip.xyz.div(eClip);
      const absUnit = vUnit.abs();
      const maxUnit = _max(absUnit.x, absUnit.y, absUnit.z);
      return maxUnit.greaterThan(1).select(
        _vec4(pClip, currentColor.a).add(vClip.div(maxUnit)),
        historyColor,
      );
    };

    // variance clipping (Salvi GDC16) — only the neighbour list changes.
    const varianceClipping = (positionTexel: TN, currentColor: TN, historyColor: TN, gamma: TN): TN => {
      const moment1 = currentColor.toVar();
      const moment2 = currentColor.pow2().toVar();
      for (const [x, y] of varOffsets) {
        const neighbor = beauty.offset(_ivec2(x, y)).load(positionTexel).max(0);
        moment1.addAssign(neighbor);
        moment2.addAssign(neighbor.pow2());
      }
      const N = _float(varOffsets.length + 1);
      const mean = moment1.div(N);
      const variance = moment2.div(N).sub(mean.pow2()).max(0).sqrt().mul(gamma);
      const minColor = mean.sub(variance);
      const maxColor = mean.add(variance);
      return clipAABB(mean.clamp(minColor, maxColor), historyColor, minColor, maxColor);
    };

    // subpixel weight in [0,1] — three's math.
    const subpixelCorrection = (velocityUV: TN, textureSize: TN): TN => {
      const velocityTexel = velocityUV.mul(textureSize);
      const phase = velocityTexel.fract().abs();
      const weight = _max(phase, phase.oneMinus());
      return weight.x.mul(weight.y).oneMinus().div(0.75);
    };

    // luminance-weighted flicker reduction — three's math.
    const flickerReduction = (currentColor: TN, historyColor: TN, currentWeight: TN): TN => {
      const historyWeight = currentWeight.oneMinus();
      const compressedCurrent = currentColor.mul(_float(1).div(_max(currentColor.r, currentColor.g, currentColor.b).add(1)));
      const compressedHistory = historyColor.mul(_float(1).div(_max(historyColor.r, historyColor.g, historyColor.b).add(1)));
      const luminanceCurrent = _luminance(compressedCurrent.rgb);
      const luminanceHistory = _luminance(compressedHistory.rgb);
      currentWeight.mulAssign(_float(1).div(luminanceCurrent.add(1)));
      historyWeight.mulAssign(_float(1).div(luminanceHistory.add(1)));
      return _add(currentColor.mul(currentWeight), historyColor.mul(historyWeight))
        .div(_max(currentWeight.add(historyWeight), 0.00001))
        .toVar();
    };

    const resolve = _Fn((): TN => {
      const uvNode = _uv();
      const textureSize = beauty.size();
      const positionTexel = uvNode.mul(textureSize);

      const currentDepth = sampleCurrentDepth(positionTexel);
      const closestDepth = currentDepth.get('closestDepth');
      const closestPositionTexel = currentDepth.get('closestPositionTexel');
      const farthestDepth = currentDepth.get('farthestDepth');

      const offsetUV = velN.load(closestPositionTexel).xy.mul(_vec2(0.5, -0.5));

      const historyUV = uvNode.sub(offsetUV);
      const previousDepth = samplePreviousDepth(historyUV);

      const isValidUV = historyUV.greaterThanEqual(0).all().and(historyUV.lessThanEqual(1).all());
      const isEdge = farthestDepth.sub(closestDepth).greaterThan(this.edgeDepthDiff);
      const isDisocclusion = closestDepth.sub(previousDepth).greaterThan(this.depthThreshold);
      const hasValidHistory = isValidUV.and(isEdge.or(isDisocclusion.not()));

      const currentColor = beauty.sample(uvNode);
      const historyColor = historyNode.sample(uvNode.sub(offsetUV));

      const motionFactor = uvNode.sub(historyUV).mul(textureSize).length().div(this.maxVelocityLength).saturate();
      const currentWeight = _float(0.05).toVar();
      if (this.useSubpixelCorrection) {
        currentWeight.addAssign(subpixelCorrection(offsetUV, textureSize).mul(0.25));
      }
      currentWeight.assign(hasValidHistory.select(currentWeight.add(motionFactor).saturate(), 1));

      const varianceGamma = _mix(0.5, 1, motionFactor.oneMinus().pow2());
      const clippedHistoryColor = varianceClipping(positionTexel, currentColor, historyColor, varianceGamma);

      return flickerReduction(currentColor, clippedHistoryColor, currentWeight);
    });

    return resolve();
  }
}

/** Drop-in for three's `traa()`, with a neighborhood cut level (1 depth-only, 2 full). */
export function leanTraa(
  beautyNode: unknown,
  depthNode: TextureNode,
  velocityNode: TextureNode,
  camera: Camera,
  cutLevel: number,
): TRAANode {
  // three's traa() factory wraps the beauty node in convertToTexture so it has a
  // passNode/renderTarget (read by the inherited updateBefore) — match it.
  const beautyTex = convertToTexture(
    beautyNode as Parameters<typeof convertToTexture>[0],
  ) as unknown as TextureNode;
  return new LeanTraaNode(beautyTex, depthNode, velocityNode, camera, cutLevel);
}
