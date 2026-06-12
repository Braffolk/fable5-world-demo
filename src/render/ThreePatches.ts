/**
 * Targeted three-internals patches (Phase 7 perf). Verified against the
 * pinned 0.184 source — re-check on any upgrade (docs/THREE-NOTES.md).
 *
 * THE SHADOW-PASS HASH STORM (profiled ~4.5-8.4 ms/frame at ~600 draws,
 * scaling with cascade renders, gone with ?ablate=shadows): the renderer
 * mutates the per-light shared shadow override material PER OBJECT —
 * `overrideMaterial.alphaTest = material.alphaTest` — and Material's
 * alphaTest accessor bumps `version` on every 0↔cutout crossing (bark=0,
 * cards=0.32 alternate constantly). A version bump makes RenderObjects.get
 * re-validate EVERY shadow render object sharing that material, and each
 * validation re-hashes the full material node graph (getMaterialCacheKey →
 * customProgramCacheKey → graph walk + cyrb53): ~328 full hashes/frame.
 *
 * Fix 1 — freezeShadowAlphaTest: shadow-pass materials get an instance-own
 * PLAIN `alphaTest` property (shadows the prototype accessor). The value
 * still updates per object (the alpha-test threshold is a per-draw uniform
 * read at bind time), but version stops thrashing. Each shadow render
 * object keeps the pipeline built for its own alphaTest class — initial
 * cache keys already encode alphaTest as a 0/1 bucket per object.
 *
 * Fix 2 — per-RenderObject getMaterialCacheKey memo (belt and braces for
 * any remaining gate fire): the key reads material state + FIXED
 * per-render-object bits (object.uuid, context.id, receiveShadow), so it
 * is exact keyed on (material identity, material.version,
 * contextNode.version). We never mutate node graphs post-build without
 * bumping needsUpdate (three would miss the pipeline rebuild anyway).
 */

import { Mesh } from 'three';
import type { WebGPURenderer } from 'three/webgpu';

interface RenderObjectShape {
  material: { version: number };
  renderer: { contextNode: { version: number } };
  getMaterialCacheKey(): number;
}

interface RenderObjectsShape {
  createRenderObject(...args: unknown[]): RenderObjectShape;
}

function freezeShadowAlphaTest(mat: object): void {
  if (Object.prototype.hasOwnProperty.call(mat, 'alphaTest')) return;
  const current = (mat as { alphaTest: number }).alphaTest;
  Object.defineProperty(mat, 'alphaTest', {
    value: current,
    writable: true,
    enumerable: false, // keep it out of the cache-key property loop, like the accessor
    configurable: true,
  });
}

/**
 * FRAGMENT-STAGE STORAGE WRITES (nanite vis-buffer HW path). three 0.184
 * structurally forbids writable storage buffers outside compute:
 *  (a) WGSLNodeBuilder.getNodeAccess returns READ_ONLY for any non-compute
 *      stage (atomics get READ_WRITE plus a console warning), so plain
 *      stores emit `var<storage, read>` → WGSL parse error;
 *  (b) WebGPUBindingUtils.createBindingsLayout hardcodes ReadOnlyStorage
 *      for every binding without COMPUTE visibility → layout/shader mismatch
 *      even when the WGSL side is read_write (the atomic case).
 * WebGPU itself allows read_write storage in fragment shaders — only the
 * library blocks it. This patch is strictly OPT-IN per buffer attribute:
 * unmarked materials keep stock behavior.
 *
 * markFragmentWritable(attr) — call for each StorageBufferAttribute that
 * fragment-stage node code writes (plain or atomic).
 */
const fragWritableAttrs = new WeakSet<object>();

export function markFragmentWritable(attr: object): void {
  fragWritableAttrs.add(attr);
}

export function installFragmentStorageWrites(renderer: WebGPURenderer): void {
  // (a) WGSL access: honor node.access for marked attributes in any stage
  const backend = renderer.backend as unknown as {
    createNodeBuilder(o: object, r: unknown): object;
    bindingUtils: object;
  };
  const builder = backend.createNodeBuilder(new Mesh(), renderer);
  const bProto = Object.getPrototypeOf(builder) as {
    getNodeAccess(node: { value?: object; access?: string }, stage: string): string;
    __laasFragWrite?: boolean;
  };
  if (bProto.__laasFragWrite !== true) {
    bProto.__laasFragWrite = true;
    const origAccess = bProto.getNodeAccess;
    bProto.getNodeAccess = function (
      this: unknown,
      node: { value?: object; access?: string },
      stage: string,
    ): string {
      if (stage !== 'compute' && node.value && fragWritableAttrs.has(node.value)) {
        return (node.access ?? 'readWrite') as string;
      }
      return origAccess.call(this, node, stage);
    };
  }

  // (b) layout: marked bindings get COMPUTE visibility OR-ed in, routing them
  // through the access-honoring branch (FRAGMENT|COMPUTE + Storage is valid;
  // the buffers never appear in a vertex stage, which WOULD be invalid)
  const utilsProto = Object.getPrototypeOf(backend.bindingUtils) as {
    createBindingsLayout(bindGroup: { bindings: unknown[] }): unknown;
    __laasFragWrite?: boolean;
  };
  if (utilsProto.__laasFragWrite !== true) {
    utilsProto.__laasFragWrite = true;
    const origLayout = utilsProto.createBindingsLayout;
    utilsProto.createBindingsLayout = function (
      this: unknown,
      bindGroup: { bindings: unknown[] },
    ): unknown {
      for (const b of bindGroup.bindings) {
        const sb = b as {
          isStorageBuffer?: boolean;
          visibility: number;
          nodeUniform?: { value?: object; node?: { value?: object } };
        };
        if (sb.isStorageBuffer !== true) continue;
        const attr = sb.nodeUniform?.node?.value ?? sb.nodeUniform?.value;
        if (attr && fragWritableAttrs.has(attr)) {
          sb.visibility |= 4; // GPUShaderStage.COMPUTE
        }
      }
      return origLayout.call(this, bindGroup);
    };
  }
}

export function installMaterialKeyMemo(renderer: WebGPURenderer): void {
  const objects = (renderer as unknown as { _objects: RenderObjectsShape })._objects;
  const managerProto = Object.getPrototypeOf(objects) as RenderObjectsShape & {
    __laasKeyMemo?: boolean;
  };
  if (managerProto.__laasKeyMemo === true) return;
  managerProto.__laasKeyMemo = true;

  const memo = new WeakMap<
    object,
    { mat: object; v: number; ctxV: number; key: number }
  >();
  const origCreate = managerProto.createRenderObject;
  let protoPatched = false;

  managerProto.createRenderObject = function (
    this: RenderObjectsShape,
    ...args: unknown[]
  ): RenderObjectShape {
    // args: nodes, geometries, renderer, object, material, scene, camera,
    //       lightsNode, renderContext, clippingContext, passId
    const mat = args[4] as { isShadowPassMaterial?: boolean } | undefined;
    if (mat?.isShadowPassMaterial === true) freezeShadowAlphaTest(mat);
    const ro = origCreate.apply(this, args);
    if (!protoPatched) {
      protoPatched = true;
      const proto = Object.getPrototypeOf(ro) as RenderObjectShape;
      const origKey = proto.getMaterialCacheKey;
      proto.getMaterialCacheKey = function (this: RenderObjectShape): number {
        const m = this.material as unknown as object & { version: number };
        const ctxV = this.renderer.contextNode.version;
        const hit = memo.get(this);
        if (
          hit !== undefined &&
          hit.mat === m &&
          hit.v === m.version &&
          hit.ctxV === ctxV
        ) {
          return hit.key;
        }
        const key = origKey.call(this);
        memo.set(this, { mat: m, v: m.version, ctxV, key });
        return key;
      };
    }
    return ro;
  };
}
