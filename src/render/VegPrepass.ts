/**
 * Position-invariance install for depth-prepass correctness (Phase 7 perf directive).
 */

import { Mesh, type WebGPURenderer } from 'three/webgpu';

/**
 * Mark every vertex shader's clip-space position `@invariant` (WGSL's
 * depth-prepass tool): without it Metal may fuse/reassociate the position
 * math differently between the depth-only and shaded pipelines, the
 * last-ulp depth mismatch fails depthFunc=EQUAL, and the shaded pass drops
 * out (background showing through blade-shaped holes). The builder class
 * isn't exported from the three/webgpu bundle — take the prototype from a
 * live instance. Cost: only position-expression optimizations are
 * restricted, identically in every pipeline.
 */
export function installPositionInvariance(renderer: WebGPURenderer): void {
  const backend = renderer.backend as unknown as {
    createNodeBuilder(o: object, r: unknown): object;
  };
  const builder = backend.createNodeBuilder(new Mesh(), renderer);
  const proto = Object.getPrototypeOf(builder) as {
    _getWGSLVertexCode(d: unknown): string;
    __laasInvariant?: boolean;
  };
  if (proto.__laasInvariant === true) return;
  proto.__laasInvariant = true;
  const orig = proto._getWGSLVertexCode;
  proto._getWGSLVertexCode = function (this: unknown, d: unknown): string {
    return orig.call(this, d).replace(
      '@builtin( position ) builtinClipSpace',
      '@invariant @builtin( position ) builtinClipSpace',
    );
  };
}
