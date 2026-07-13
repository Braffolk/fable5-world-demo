import assert from 'node:assert/strict';
import test from 'node:test';

import { RemoteWorldSource } from './RemoteWorldSource';

function signedIndex(records: readonly (readonly [number, number, number])[]): Uint8Array {
  const bytes = new Uint8Array(records.length * 21);
  const view = new DataView(bytes.buffer);
  records.forEach(([lod, cx, cz], i) => {
    const off = i * 21;
    view.setInt8(off, lod);
    view.setInt32(off + 1, cx, true);
    view.setInt32(off + 5, cz, true);
    view.setUint32(off + 9, 64, true);
    view.setBigUint64(off + 13, BigInt(i + 1), true);
  });
  return bytes;
}

test('RemoteWorldSource opens a format-2 signed index with mixed physical LODs', async (t) => {
  const previousFetch = globalThis.fetch;
  const previousWorker = globalThis.Worker;
  class StubWorker {
    onmessage: ((event: MessageEvent) => void) | null = null;
    onerror: ((event: ErrorEvent) => void) | null = null;
    postMessage(): void {}
    terminate(): void {}
  }
  globalThis.Worker = StubWorker as unknown as typeof Worker;
  const manifest = {
    format: 2,
    containers: ['LAC1', 'LAC2'],
    codec: 'deflate',
    grid: { anchorE: 368640, anchorN: 6635520, chunkMeters: 2048, chunkRes: 2048, lodStep: 4 },
    layers: {
      height: {
        enc: 1, lods: [-2, -1, 0], count: 3, bytes: 192, index: 'index/height.bin',
        baseTexelMeters: 1, finestLod: -2, authorityLod: 0, synthesis: 'microtopography-v1',
      },
    },
  };
  const index = signedIndex([[-2, 0, 0], [-1, 0, 0], [0, 0, 0]]);
  globalThis.fetch = (async (input: string | URL | Request) => {
    const url = String(input);
    if (url.endsWith('/latest.json')) return Response.json({ manifest: 'm/mixed/manifest.json' });
    if (url.endsWith('/manifest.json')) return Response.json(manifest);
    if (url.endsWith('/index/height.bin')) return new Response(index.slice().buffer as ArrayBuffer);
    return new Response('missing', { status: 404 });
  }) as typeof fetch;
  t.after(() => {
    globalThis.fetch = previousFetch;
    globalThis.Worker = previousWorker;
  });

  const source = new RemoteWorldSource('https://fixture.invalid');
  const opened = await source.open();
  assert.equal(opened.format, 2);
  assert.deepEqual(opened.containers, ['LAC1', 'LAC2']);
  assert.equal(opened.layers.height?.finestLod, -2);
  assert.equal(opened.coverage('height', { lod: -2, cx: 0, cz: 0 })?.lod, -2);
  assert.equal(opened.coverage('height', { lod: 0, cx: 0, cz: 0 })?.lod, 0);
  assert.equal(opened.coverage('height', { lod: -2, cx: 1, cz: 0 }), null);
  assert.deepEqual(opened.chunks('height', -1), [{ lod: -1, cx: 0, cz: 0 }]);
  source.close();
});
