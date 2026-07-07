/**
 * Splat.ts — `nanSplatElect`, the sub-pixel election consumer (task #76).
 *
 * The sub-pixel majority is not rasterized as triangles: world1 APPENDS (px,cand,payload)
 * to splatQueue and this lean kernel (24 regs measured) runs the depthKey24 (<<8|id)
 * atomicMax election into visPayloadV/visBV at full occupancy. INDIRECT dispatch
 * (kSplatArgs → this, the cull BFS kArgs→kTraverse in-batch pattern) launches only
 * ceil(N/64) workgroups. Binds visPayloadV/visBV/splatQueue = 3 storage buffers ⇒ lean.
 */

import { Fn, instanceIndex, uint } from 'three/tsl';
import { IndirectStorageBufferAttribute } from 'three/webgpu';
import type { NU } from '../../gpu/TSLTypes';
import { aLoadU, minU, returnIf, setIndirectDispatch, sU32Views } from '../Tsl';
import { SPLAT_CAP } from './Queues';

type U32Views = ReturnType<typeof sU32Views>;

interface ComputeKernel {
  setName(name: string): unknown;
}

/** Build `nanSplatElect` (null when the splat queue is absent). */
export function buildSplat(p: {
  splatQueueV: U32Views | null;
  splatDrawAttr: IndirectStorageBufferAttribute | null;
  /** the shipped depth-keyed election bound to the vis buffers (VisBuffer.makeElect). */
  elect: (px: NU, cand: NU, idStore: NU) => void;
}): unknown | null {
  const { splatQueueV, splatDrawAttr, elect } = p;
  if (!(splatQueueV && splatDrawAttr)) return null;
  const kn = Fn(() => {
    const i = instanceIndex;
    returnIf(
      i.greaterThanEqual(
        minU(aLoadU(splatQueueV.atomic.element(0)), uint(SPLAT_CAP)),
      ),
    );
    const b = uint(1).add(i.mul(uint(3)));
    const px = aLoadU(splatQueueV.atomic.element(b));
    const cand = aLoadU(splatQueueV.atomic.element(b.add(uint(1))));
    const pay = aLoadU(splatQueueV.atomic.element(b.add(uint(2))));
    // verbatim relect===1 election (VisBuffer.makeElect):
    elect(px, cand, pay);
  })().compute(SPLAT_CAP, [64]); // grid overridden by setIndirectDispatch
  (kn as unknown as ComputeKernel).setName('nanSplatElect');
  setIndirectDispatch(kn, splatDrawAttr);
  return kn;
}
