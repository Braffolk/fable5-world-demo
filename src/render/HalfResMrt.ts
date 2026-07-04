/**
 * One half-res MRT quad pass for the screen-space layers that don't need
 * full resolution: volumetric cloud march, GTAO, screen-space bounce gather.
 * Replaces three separate passes (two RTTNodes + GTAONode) with a single
 * raster — one encoder, shared depth fetches, no per-pass RT round-trips.
 *
 * Mechanics mirror three's own pass nodes (GTAONode/RTTNode, 0.184):
 * - `material.fragmentNode = mrt({...})` — MRTNode is an OutputStructNode,
 *   so this skips the whole lighting build (NodeMaterial.setup fragmentNode
 *   path) and maps each entry to the attachment whose TEXTURE NAME matches
 *   the entry key (MRTNode.setup → getTextureIndex), read off the render
 *   target bound at build time — which is ours, because the first
 *   quad.render with our RT bound triggers the compile.
 * - consumers pull per-attachment PassTextureNodes via getTextureNode(name);
 *   referencing one in a graph keeps this node's updateBefore in the frame.
 */

import { HalfFloatType, RGFormat, RedFormat, UnsignedByteType, Vector2 } from 'three';
import type { NodeBuilder, NodeFrame, Renderer, TextureNode } from 'three/webgpu';
import {
  NodeMaterial,
  NodeUpdateType,
  QuadMesh,
  RenderTarget,
  RendererUtils,
  TempNode,
} from 'three/webgpu';
import { mrt, passTexture } from 'three/tsl';
import { tagGpu } from '../core/GpuProfiler';
import { runiform } from '../gpu/RenderUniform';
import { internalSize } from './RenderScale';

export interface HalfResEntry {
  name: string;
  node: unknown; // TSL node producing this attachment (vec4-compatible)
  /** r8unorm attachment (e.g. AO) instead of rgba16f */
  red?: boolean;
  /** rg16f attachment (2-channel payloads, e.g. AO+viewZ — RP-3b, deep-review 08-L2b):
   *  halves the write + 4-tap-read bandwidth vs rgba16f. The producing node may still
   *  return vec4 (WebGPU discards the extra output components). */
  rg?: boolean;
}

type RendererState = unknown;

export class HalfResMrtNode extends TempNode {
  /** half-res dimensions, live-updated — the GTAO noise tiling reads this */
  readonly resolution = runiform(new Vector2());

  private readonly rt: RenderTarget;
  private readonly material = new NodeMaterial();
  private readonly quad = new QuadMesh();
  private readonly entries: HalfResEntry[];
  private readonly scale: number;
  private readonly texNodes = new Map<string, TextureNode>();
  private rendererState: RendererState;

  constructor(entries: HalfResEntry[], scale = 0.5) {
    super('vec4');
    if (entries.length === 0) throw new Error('HalfResMrtNode needs at least one entry');
    this.entries = entries;
    this.scale = scale;
    this.updateBeforeType = NodeUpdateType.FRAME;

    this.rt = new RenderTarget(1, 1, {
      count: entries.length,
      depthBuffer: false,
      type: HalfFloatType,
    });
    tagGpu(this.rt, 'half.mrt');
    entries.forEach((e, i) => {
      const tex = this.rt.textures[i];
      if (!tex) return;
      tex.name = e.name;
      if (e.red === true) {
        tex.format = RedFormat;
        tex.type = UnsignedByteType;
      } else if (e.rg === true) {
        tex.format = RGFormat; // stays HalfFloatType ⇒ rg16float (renderable)
      }
    });
    this.material.name = 'HalfResMRT';
  }

  getTextureNode(name: string): TextureNode {
    let node = this.texNodes.get(name);
    if (!node) {
      const idx = this.entries.findIndex((e) => e.name === name);
      const tex = this.rt.textures[idx];
      if (idx < 0 || !tex) throw new Error(`HalfResMrtNode: no entry '${name}'`);
      node = passTexture(
        this as unknown as Parameters<typeof passTexture>[0],
        tex,
      ) as unknown as TextureNode;
      this.texNodes.set(name, node);
    }
    return node;
  }

  private setSize(width: number, height: number): void {
    const w = Math.max(2, Math.round(width * this.scale));
    const h = Math.max(2, Math.round(height * this.scale));
    this.resolution.value.set(w, h);
    this.rt.setSize(w, h); // no-op when unchanged
  }

  override updateBefore(frame: NodeFrame): boolean | undefined {
    const renderer = (frame as unknown as { renderer: Renderer }).renderer;
    // half of the INTERNAL res (?rscale) — must track the scene pass, not the canvas
    const size = internalSize(renderer, _size);
    this.setSize(size.width, size.height);

    this.rendererState = RendererUtils.resetRendererState(
      renderer,
      this.rendererState as Parameters<typeof RendererUtils.resetRendererState>[1],
    );
    renderer.setRenderTarget(this.rt);
    this.quad.material = this.material;
    this.quad.name = 'HalfResMRT';
    this.quad.render(renderer);
    RendererUtils.restoreRendererState(
      renderer,
      this.rendererState as Parameters<typeof RendererUtils.restoreRendererState>[1],
    );
    return undefined;
  }

  override setup(_builder: NodeBuilder): ReturnType<HalfResMrtNode['getTextureNode']> {
    const outputs: Record<string, unknown> = {};
    for (const e of this.entries) outputs[e.name] = e.node;
    // the MRTNode must be the fragmentNode DIRECTLY: NodeMaterial routes the
    // multi-output path on `fragmentNode.isOutputStructNode` — wrapping it
    // (e.g. .context()) hides the flag and the builder vec4-wraps it, so the
    // WGSL output struct loses its members ("struct member m0 not found")
    this.material.fragmentNode = mrt(outputs as Parameters<typeof mrt>[0]) as never;
    this.material.needsUpdate = true;
    return this.getTextureNode(this.entries[0]?.name ?? '');
  }

  override dispose(): void {
    this.rt.dispose();
    this.material.dispose();
  }
}

const _size = new Vector2();
