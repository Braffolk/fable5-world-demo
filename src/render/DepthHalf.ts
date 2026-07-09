/**
 * DepthHalf — a tiny prefix pass (one half-res r16float attachment) that the
 * merged HalfResMrt über-fragment's GTAO/bounce taps read INSTEAD of the
 * full-res raw-depth buffer (PERF-P3).
 *
 * Why it exists: the HalfResMrt pass already renders at half resolution, yet
 * GTAO (12 march taps + a 3×3 normal stencil) and the bounce gather (8 taps)
 * each did a full-res depth textureLoad + an inverse-projection unproject per
 * tap — full-res random-stride fetches that are cache-hostile and carry no
 * information the half-res consumer can resolve. This pass point-samples the
 * TOP-LEFT full-res depth texel of each 2×2 block (a DETERMINISTIC downsample —
 * no min/max/jitter/dither) and stores LINEAR view-Z. The taps then read this
 * compact, cache-resident texture with a nearest sampler and reconstruct view
 * position straight from view-Z (getViewPositionFromViewZ), dropping the
 * per-tap unproject entirely.
 *
 * VRAM: exactly one half-res r16float texture (RedFormat + HalfFloatType),
 * depthBuffer:false. At a 2560×1440 dpr2 drawing buffer that is 1280×720×2 B ≈
 * 1.84 MB; at a 2880×1800 retina buffer 1440×900×2 B ≈ 2.59 MB. Nothing else.
 *
 * r16f precision: view-Z is stored linearly, so ulp ≈ |viewZ|·2⁻¹¹. At the AO
 * fade end (~240 m) that is ≈0.06–0.12 m — well under the 1 m GTAO thickness
 * test, so the horizon/thickness classification is unaffected across the whole
 * marched band (AO is forced to 1 past ~240 m anyway).
 *
 * Ordering: this node is referenced by the GTAO/bounce fragment nodes, so it is
 * registered as a child (post-order) sequential node of the HalfResMrt
 * material. Its FRAME updateBefore therefore fires during that pass's quad
 * render, BEFORE the über-quad draws — i.e. "right before the HalfResMRT quad"
 * — reading the current-frame scene depth (already produced by the scene pass
 * earlier in the main pipeline). Mirrors HalfResMrtNode's render idiom exactly.
 */

import {
  ClampToEdgeWrapping,
  HalfFloatType,
  NearestFilter,
  RedFormat,
  Vector2,
} from 'three';
import type { NodeBuilder, NodeFrame, Renderer, TextureNode } from 'three/webgpu';
import {
  NodeMaterial,
  NodeUpdateType,
  QuadMesh,
  RenderTarget,
  RendererUtils,
  TempNode,
} from 'three/webgpu';
import {
  Fn,
  floor,
  ivec2,
  min,
  passTexture,
  screenSize,
  screenUV,
  textureLoad,
  textureSize,
  vec4,
} from 'three/tsl';
import { tagGpu } from '../core/GpuProfiler';
import type { NF, NIV2, NV4 } from '../gpu/TSLTypes';
import { getViewPositionFast } from './Gtao';
import { internalSize } from './RenderScale';

interface DepthTexLike {
  value: unknown;
}

type RendererState = unknown;

export class DepthHalfNode extends TempNode {
  private readonly rt: RenderTarget;
  private readonly material = new NodeMaterial();
  private readonly quad = new QuadMesh();
  private readonly depthTex: DepthTexLike;
  private readonly projInv: unknown;
  private texNode: TextureNode | null = null;
  private rendererState: RendererState;

  /**
   * @param depthTex full-res scene-pass depth texture node (its `.value` is the
   *   depth texture; point-sampled via textureLoad, like the bounce gather).
   * @param projInv the live inverse-projection uniform (jitter-carrying).
   */
  constructor(depthTex: DepthTexLike, projInv: unknown) {
    super('vec4');
    this.depthTex = depthTex;
    this.projInv = projInv;
    this.updateBeforeType = NodeUpdateType.FRAME;

    this.rt = new RenderTarget(1, 1, {
      count: 1,
      depthBuffer: false,
      type: HalfFloatType,
    });
    const tex = this.rt.textures[0];
    if (tex) {
      tex.name = 'viewZHalf';
      tex.format = RedFormat; // RedFormat + HalfFloatType ⇒ r16float
      // nearest, hardware-clamped: replaces the manual mip/size-guard chains the
      // full-res taps used, and keeps the downsample a pure point sample.
      tex.minFilter = NearestFilter;
      tex.magFilter = NearestFilter;
      tex.wrapS = ClampToEdgeWrapping;
      tex.wrapT = ClampToEdgeWrapping;
    }
    tagGpu(this.rt, 'half.viewz');
    this.material.name = 'DepthHalfViewZ';
  }

  getTextureNode(): TextureNode {
    if (!this.texNode) {
      this.texNode = passTexture(
        this as unknown as Parameters<typeof passTexture>[0],
        this.rt.textures[0],
      ) as unknown as TextureNode;
    }
    return this.texNode;
  }

  private setSize(width: number, height: number): void {
    const w = Math.max(2, Math.round(width * 0.5));
    const h = Math.max(2, Math.round(height * 0.5));
    this.rt.setSize(w, h); // no-op when unchanged
  }

  override updateBefore(frame: NodeFrame): boolean | undefined {
    const renderer = (frame as unknown as { renderer: Renderer }).renderer;
    // half of the INTERNAL res (?rscale) — must track the scene depth, like HalfResMrt
    const size = internalSize(renderer, _size);
    this.setSize(size.width, size.height);

    this.rendererState = RendererUtils.resetRendererState(
      renderer,
      this.rendererState as Parameters<typeof RendererUtils.resetRendererState>[1],
    );
    renderer.setRenderTarget(this.rt);
    this.quad.material = this.material;
    this.quad.name = 'DepthHalfViewZ';
    this.quad.render(renderer);
    RendererUtils.restoreRendererState(
      renderer,
      this.rendererState as Parameters<typeof RendererUtils.restoreRendererState>[1],
    );
    return undefined;
  }

  override setup(_builder: NodeBuilder): ReturnType<DepthHalfNode['getTextureNode']> {
    this.material.fragmentNode = Fn((): NV4 => {
      const dval = this.depthTex.value as unknown as Parameters<typeof textureLoad>[0];
      // full-res depth dims (no-coord textureLoad = a bare handle for textureSize)
      const full = textureSize(textureLoad(dval)) as unknown as NIV2;
      // screenSize == the bound RT (this half-res target) ⇒ this is the half-res
      // pixel coord; ×2 = the TOP-LEFT full-res texel of its 2×2 block, clamped
      // (min componentwise on ivec) so an odd-sized last texel never reads OOB.
      const hp = ivec2(floor(screenUV.mul(screenSize)));
      const topLeft = min(
        hp.mul(2) as unknown as NF,
        full.sub(ivec2(1, 1)) as unknown as NF,
      ) as unknown as NIV2;
      const depth = textureLoad(
        dval,
        topLeft as unknown as Parameters<typeof textureLoad>[1],
      ) as unknown as NF;
      // view-Z is independent of ndc.xy (proj row 3 = (0,0,-1,0)); passing the
      // fragment's own screenUV is exact for the .z we keep.
      const viewZ = getViewPositionFast(screenUV, depth, this.projInv as never).z;
      return vec4(viewZ, 0, 0, 1);
    })() as never;
    this.material.needsUpdate = true;
    return this.getTextureNode();
  }

  override dispose(): void {
    this.rt.dispose();
    this.material.dispose();
  }
}

const _size = new Vector2();
